# Copyright 2026-present Kensho Technologies, LLC.
"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

Unlike BasicTokenizer:
- RegexTokenizer handles an optional regex splitting pattern.
- RegexTokenizer handles optional special tokens.

- this will be regular BPE with Picky BPE deletions
- with counts for merges

"""

from typing import Optional, Pattern, Iterator, Any
import os, time, json, copy
from collections import defaultdict
from heapdict import heapdict  # type: ignore[import-untyped]
from .util import frombytes, verify_dicts, frombytespair, blow_up, merge, create_initial_vocab
from .inferencedata import InferenceData
from .pretokenize import Pretokenizer
from .inference import Tokenizer
from .vocabulary import Vocabulary

# Default frequency (in merges) for printing `*` progress rows when not verbose.
# Aligns with the default checkpoint_iterations (8192 = 8 * 1024).
DEFAULT_PROGRESS_INTERVAL = 1024


def _resolve_progress_interval(progress_interval: Optional[int], verbose: bool) -> int:
    """Resolve the progress_interval sentinel.

    None -> 1 if verbose else DEFAULT_PROGRESS_INTERVAL. An explicit value (including 0,
    which disables progress rows) is used as-is regardless of verbose.
    """
    if progress_interval is None:
        return 1 if verbose else DEFAULT_PROGRESS_INTERVAL
    return progress_interval

# Generate sequential IDs for merge/deletion operations.
# Each trainer instance has its own counter to avoid conflicts in parallel training.
class IndexGenerator:
    """Per-instance index generator for merge/deletion operations."""

    def __init__(self) -> None:
        self._next_index = 0

    def get_next_index(self) -> int:
        current_index = self._next_index
        self._next_index += 1
        return current_index

    def reset(self) -> None:
        """Reset the index counter. Called at the start of each training run."""
        self._next_index = 0


class BaseBpeTrainer:

    def __init__(self, pretokenizer: Optional[Pretokenizer] = None) -> None:
        """
        Initialize the BPE trainer.

        Args:
            pretokenizer: Pretokenizer instance to use for text splitting.
                         If None, creates a default Pretokenizer in simple mode.
        """
        # Create inference data instance - holds all data needed for inference
        if pretokenizer is None:
            pretokenizer = Pretokenizer()
        self.inf_data = InferenceData.create_for_training(pretokenizer)

        # Per-instance index generator for merge/deletion operations
        # This avoids conflicts when running multiple trainers in parallel
        self.index_gen = IndexGenerator()

        # Initial vocabulary of valid single bytes (243 bytes for utf-8)
        self.initial_vocab = create_initial_vocab()

        # Training-only data structures below
        # These are NOT serialized and are only used during training

        # vocab will be initialized in train() method
        # - For first pass (regular BPE): starts with initial bytes via Vocabulary.create_initial()
        # - For second pass BoundlessBPE: starts with initial bytes via Vocabulary.create_initial()
        # - For second pass SuperBPE: copies word model vocab via Vocabulary.create_from_vocab()
        self.vocab: Optional[Vocabulary] = None

        # single_counts: token -> count for the current tokenization state
        # Used during training to track token frequencies; final counts saved to Vocabulary
        self.single_counts: dict[bytes, int] = {}

        # text_chunks and text_counts are now returned from pretokenize methods
        # and passed as parameters to training methods, not stored as instance variables

        # heapdict of all pairs with composite priority for efficient best pair selection
        # maps pair -> (-both_unlocked, -count, pair)
        # unlocked pairs automatically sort to top, then by count, then lexicographically
        # initialized as empty, will be populated in initial_pair_counts()
        self.pair_counts : heapdict = heapdict()

        # index mapping each token to pairs involving it (only used in super merge training)
        # enables O(k) updates when token is unlocked instead of O(n) scan
        # initialized as empty, will be populated in train() if is_super
        # not used if superbpe_mode
        self.token_to_pairs : defaultdict[bytes, set[Any]] = defaultdict(set)

        # index mapping each token to chunk indices containing it
        # enables O(k) iteration in merge/delete instead of O(n) over all chunks
        # initialized as empty, will be populated in initial_token_locations()
        self.token_locations: dict[bytes, set[int]] = defaultdict(set)

        # is a token unlocked to merge with something else
        # initialized as empty, will be reassigned in train() based on is_super
        self.unlocked : defaultdict[bytes, bool] = defaultdict(lambda: True)

        # number of single token chunks, for stats
        self.whole_words = 0

        # a static set of single bytes, for computing the total number of single bytes in stats
        self.single_bytes = set([bytes([idx]) for idx in range(256)])

        # for supermerge training: track position in word model operations
        self.word_model : Optional[Tokenizer] = None
        self.word_operations_list : list[tuple[int, str, Any]] = []  # (index, 'merge'|'delete', data)
        self.current_word_op_idx : int = 0
        self.target_vocab_size : int = 0

        # pending special tokens for two-pass training (indices computed at save time)
        self._pending_special_tokens : list[str] = []

    def _reset_training_state(self, pretokenizer: Optional[Pretokenizer] = None) -> None:
        """
        Reset all training state to allow retraining on the same instance.

        This is called at the start of each train() call to ensure clean state.
        Allows the same trainer instance to be used for multiple training runs.

        Args:
            pretokenizer: Optional new pretokenizer. If None, keeps the existing one.
        """
        # Reset index generator
        self.index_gen.reset()

        # Reset or recreate inference data
        if pretokenizer is not None:
            self.inf_data = InferenceData.create_for_training(pretokenizer)
        else:
            self.inf_data = InferenceData.create_for_training(self.inf_data.pretokenizer)

        # Reset training-only data structures
        self.vocab = None
        self.single_counts = {}
        self.pair_counts = heapdict()
        self.token_to_pairs = defaultdict(set)
        self.token_locations = defaultdict(set)
        self.unlocked = defaultdict(lambda: True)
        self.whole_words = 0

        # Reset supermerge training state
        self.word_model = None
        self.word_operations_list = []
        self.current_word_op_idx = 0
        self.target_vocab_size = 0
        self._pending_special_tokens = []

    # work at a chunk level
    # where a document is stored as a list of chunks
    # call this for regular merges
    def pretokenize(self, filepath : str, num_lines : int, max_bytes : int = 1000000000, save_pretokens : Optional[str] = None, verbose : bool = True) -> tuple[list[list[bytes]], list[int]]:

        start_time = time.time()

        # for regular words tally things up
        assert not self.inf_data.is_super

       # get counts of pre-tokenized chunks for each document
        chunk_tally : dict[str, int] = {}

        total_bytes = 0
        total_chars = 0

        with open(filepath) as f:
            for i in range(num_lines):
                if verbose and i % 10000 == 0:
                    print("document", i, time.time() - start_time, total_chars, total_bytes)
                line = f.readline()
                text = json.loads(line.rstrip())["text"]

                # split the text up into text chunks
                for chunk in self.inf_data.pretokenizer.pretokenize(text):
                    chunk_tally[chunk] = chunk_tally.get(chunk, 0) + 1

                total_chars += len(text)
                total_bytes += len(text.encode("utf-8"))

                if total_bytes >= max_bytes:
                    if verbose:
                        print('at max_bytes', i, max_bytes, total_chars, total_bytes)
                    break

        # lets make parallel list of chunks and counts before we split up the chunks
        # sort descending by count for neatness, TODO: can take out later
        # convert to bytes here
        cnt_chk = sorted([(cnt,chk.encode("utf-8")) for chk,cnt in chunk_tally.items()], reverse=True)

        if len(cnt_chk) == 0:
            print("WARNING: no pre-tokenization chunks found!")
            return [], []

        # save them to a file for use in another project
        if save_pretokens is not None:
            parent = os.path.dirname(save_pretokens)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(save_pretokens, "wt") as out:
                for (cnt, chk) in cnt_chk:
                    out.write(str(cnt) + "\t" + frombytes(chk) + "\n")
            print(f"Saved {len(cnt_chk)} pretokens to {save_pretokens}")

        if verbose:
            print("number of pre-tokenization chunks:", len(cnt_chk))
            print("top 10 pre-tokenization chunks:")
            for j in range(min(10, len(cnt_chk))):
                print(j, cnt_chk[j])
            print()

        # convert to lists
        counts_t, chunks_t = zip(*cnt_chk)
        text_counts = list(counts_t)
        text_chunks = list(chunks_t)

        # should always stay in parallel
        assert len(text_chunks) == len(text_counts)

        total_unique_bytes = sum([len(chk) for chk in text_chunks])
        total_counts = sum(text_counts)

        # split bytes into single bytes to get started
        text_chunks = [[bytes([b]) for b in ch] for ch in text_chunks]

        if verbose:
            print("pretokenization time:",time.time() - start_time, len(text_counts), total_counts, total_chars, total_bytes, total_unique_bytes, total_unique_bytes/total_bytes)
            print()

        return text_chunks, text_counts

    # call this for
    def pretokenize_super(self,
                          filepath : str,
                          num_lines : int,
                          max_bytes : int = 1000000000,
                          save_pretokens : Optional[str] = None,
                          verbose : bool = True) -> tuple[list[list[bytes]], list[int]]:

        start_time = time.time()

        # for super merges need to do regular merges first
        assert self.inf_data.is_super
        assert self.word_model is not None

        total_bytes = 0
        total_chars = 0

        counts: dict[tuple[bytes, ...], int] = {}

        # Shortcut: precompute which pretokens (as raw bytes) are both
        # single-token after merges (reachable) AND in possible_superwords.
        # This lets us skip the expensive per-document fast_merge_delete entirely.
        assert self.word_model.reachable_vocab is not None
        assert self.word_model.possible_superwords is not None
        supermerge_eligible = self.word_model.reachable_vocab & self.word_model.possible_superwords
        if verbose:
            print(f"supermerge_eligible: {len(supermerge_eligible)} tokens "
                  f"(reachable={len(self.word_model.reachable_vocab)}, "
                  f"possible_superwords={len(self.word_model.possible_superwords)})")

        with open(filepath) as f:
            for i in range(num_lines):
                if verbose and i % 10000 == 0:
                    print("document", i, time.time() - start_time, total_chars, total_bytes)
                line = f.readline()
                text = json.loads(line.rstrip())["text"]

                # split the text up into text chunks using Pretokenizer
                # and just convert them to list[bytes]
                document_bytes = [chunk.encode("utf-8") for chunk in self.inf_data.pretokenizer.pretokenize(text)]

                # Find runs of consecutive eligible pretokens directly,
                # skipping the full merge replay.
                n = len(document_bytes)
                idx = 0
                while idx < n:
                    if document_bytes[idx] not in supermerge_eligible:
                        idx += 1
                        continue
                    # start of a run
                    end = idx + 1
                    while end < n and document_bytes[end] in supermerge_eligible:
                        end += 1
                    # need at least 2 consecutive eligible pretokens
                    if end - idx >= 2:
                        superword_run = tuple(document_bytes[idx:end])
                        counts[superword_run] = counts.get(superword_run, 0) + 1
                    idx = end

                total_chars += len(text)
                total_bytes += len(text.encode("utf-8"))

                if total_bytes >= max_bytes:
                    if verbose:
                        print('at max_bytes', i, max_bytes, total_chars, total_bytes)
                    break

            # lets make parallel list of chunks and counts before we split up the chunks
            # sort descending by count for neatness, TODO: can take out later
            # convert from tuple(bytes) to list[bytes)
            cnt_chk = sorted([(cnt,list(chk)) for chk,cnt in counts.items()], reverse=True)

            if len(cnt_chk) == 0:
                print("WARNING: no superword pre-tokenization chunks found!")
                return [], []

            # save them to a file for use in another project
            if save_pretokens is not None:
                parent = os.path.dirname(save_pretokens)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                with open(save_pretokens, "wt") as out:
                    for (cnt, chk) in cnt_chk:
                        out.write(str(cnt) + "\t" + "\t".join(frombytes(tok) for tok in chk) + "\n")
                print(f"Saved {len(cnt_chk)} superword pretokens to {save_pretokens}")

            if verbose:
                print("number of superword pre-tokenization chunks:", len(cnt_chk))
                print("top 100 superword pre-tokenization chunks:")
                for j in range(min(100, len(cnt_chk))):
                    print(j, cnt_chk[j])
                print()

            # convert to lists
            counts_t, chunks_t = zip(*cnt_chk)
            text_counts = list(counts_t)
            text_chunks = list(chunks_t)

        # should always stay in parallel
        assert len(text_chunks) == len(text_counts)

        # how many bytes unique
        total_unique_bytes = sum([len(chk) for chk in text_chunks])

        total_counts = sum(text_counts)

        if verbose:
            print("pretokenization time:",time.time() - start_time, len(text_counts), total_counts, total_chars, total_bytes, total_unique_bytes, total_unique_bytes/total_bytes)
            print()

        return text_chunks, text_counts

    def print_tokenization(self, text_chunks: list[list[bytes]], text_counts: list[int]) -> None:

        print()
        print("final tokenization:")
        for tokens, cnt in zip(text_chunks, text_counts):

            # I'm curious about ones with a single byte in them
            minlen = min([len(tok) for tok in tokens])
            output = " ".join([frombytes(tok) for tok in tokens])

            print("!", minlen, len(tokens), output, cnt)

    #############################################################################

    # for this to be quick, we have to rely on the fact that most pretokens are skipped entirely
    def get_stats(self,
                  tokens : list[bytes],
                  counts : Optional[dict[tuple[bytes,bytes],int]],
                  multiplier : int,  # +cnt or -cnt depending
                  ) -> dict[tuple[bytes,bytes],int]:
        """
        Given a list of bytes object tokens,
        return a dictionary of counts of consecutive pairs
        Example: [b'a', b'b', b'c', b'a', b'b'] -> {(b'a', b'b'): 2,
                                                    (b'b', b'c'): 1,
                                                    (b'c', b'a'): 1}
        Optionally allows an existing dictionary of counts to update

        Note: Counts ALL pairs regardless of unlock status. Filtering by unlock
        status happens in choose_best_pair().
        """
        counts = {} if counts is None else counts
        prev_pair = None
        for pair in zip(tokens, tokens[1:]): # iterate consecutive elements
            sameasprevious = pair[0] == pair[1] and pair == prev_pair
            if not sameasprevious:
                counts[pair] = counts.get(pair, 0) + multiplier
                prev_pair = pair
            else:
                prev_pair = None # do the next one but skip this one

        return counts

    # returns True if we found any merges
    def get_stats_faster(self,
                         first: bytes,
                         second: bytes,
                         tokens : list[bytes],
                         counts : dict[tuple[bytes,bytes],int],
                         multiplier : int,  # +cnt or -cnt depending
                         ) -> int:
            """
            TODO update this
            Given a list of bytes object tokens, 
            return a dictionary of counts of consecutive pairs
            Example: [b'a', b'b', b'c', b'a', b'b'] -> {(b'a', b'b'): 2, 
                                                        (b'b', b'c'): 1, 
                                                        (b'c', b'a'): 1}
            Optionally allows an existing dictionary of counts to update
            """
            
            verbose = False # first == b'\xc2' and second == b'\xa0'
            
            assert self.unlocked[first] 
            assert self.unlocked[second]
            assert counts is not None

            combined = first + second

            # Note: combined will be unlocked at the end of merge_and_update
            # before apply_pair_count_changes, so priorities will be correct

            # complicating case 1:
            # if merging b' ' and b't', we make a mistake on the change in the pairwise count 
            # of (b't', b't') because it has overlap!!!!
            # [b' ', b't', b't', b't', b't', b't', b'g', b'y', b'y', b't', b't', b't', b't', b't', b't', b't', b'y']
            # can we fix it?

            # if we have a run of odd length starting with second
            # [b' ', b't', b't', b't', b't', b't', b'g', b'y', b'y', b't', b't', b't', b't', b't', b't', b't', b'y']
            # we can make two b'tt' pairs
            # however, after making b' t' we are left with 4, so we can still make 2 b'tt' pairs!!!!

            # complicating case 2:
            # a b first second c d
            # b first second c d
            # first second c d
            # but we might have a == first and b == second, 
            # or c == first, and d == second, in which case we have non-overalapping runs of merges

            # first second first second c d -> firstsecond firstsecond c d after the merge
            # in which case we don't want to add (second, firstsecond) to the counts

            # complicating case 3:
            # say we have [ ... b'i', b'n', b'i', b'n', ...]
            
            merge_cnt = 0  # did we do any merges?

            n = len(tokens)

            i = 0
            while i < n-1:

                # does `first` appear next to `second` in the list?
                # I'm hoping this is faster for long lists of tokens
                # since only need one pass throught the list
                if tokens[i] == first and tokens[i+1] == second:
                                    
                    # now find out how many repeated copies of (first, second) appear after this
                    k = i 
                    while (k+3 < n) and tokens[k+2] == first and tokens[k+3] == second:
                        k += 2

                    # this gives us one or more copies of (first, second) from (i,i+1) to (k,k+1)
                    assert tokens[k] == first 
                    assert tokens[k+1] == second

                    copies = ((k+1)-i+1)   # upper - lower + 1
                    assert copies % 2 == 0 # should be even, since takes two slots
                    copies = copies // 2
                    assert copies >= 1

                    if verbose:
                        print("copies:", copies, tokens)

                    merge_cnt += copies

                    # this always goes down by multiplier*copies
                    counts[(first,second)] = counts.get((first,second), 0) - multiplier*copies

                    # for b'e', b'd', b'e', b'd', b'e', b'd' 
                    # you'll have b'ed', b'ed', b'ed', you only end up with one b'ed' pair
                    # with b'ed', b'ed', b'ed', b'ed' you get two
                    if (copies % 2 == 0):
                        if verbose:
                            print("even branch", counts.get((combined,combined), 0), multiplier, (copies-1))
                        counts[(combined,combined)] = counts.get((combined,combined), 0) + multiplier*(copies // 2)
                    else:
                        # if just one, then won't be making any of these 
                        if copies > 1:
                            if verbose: 
                                print('odd branch', counts.get((combined,combined),0), multiplier, (copies-1)//2)
                            # ignore the odd one, and then also have n-1 intervals 
                            counts[(combined,combined)] = counts.get((combined,combined), 0) + multiplier*((copies-1)//2)
                    
                    # may have some combined after (first, second)
                    # to avoid double counting, add the number of copies
                    if copies > 1:

                        # also need to reduce the counts that were there for (n,i)
                        # in between the middle i n in i n i n 
                        # again there are counts-1 of these 
                        counts[(second,first)] = counts.get((second, first), 0) - multiplier*(copies-1)

                    # ok, so now we want to find the previous and after tokens (if any)
                    # and update our counts 
                    # we have: ... prev2, prev, first, second, after, after2, .... 
                    # so we decrease (prev, first), (first,second), and (second,after)
                    # and increase (prev, combined) and (combined, after)      
                    # # however we might have prev2 == first 
                    # and prev == second or after==first and after2 == second, or both
                    if i-1 >= 0:

                        # is there a previous token
                        prev = tokens[i-1]
                        if verbose:
                            print("prev:", prev)

                        # don't have a combined before so this count goes up
                        counts[(prev,combined)] = counts.get((prev,combined), 0) + multiplier

                        # are there any runs of first before the matches
                        # now this only goes up if we don't have an odd length run to the left of first
                        j = i  # start with i, and look backward
                        while (j-1 >= 0) and tokens[j-1] == first:
                            j -= 1

                        # if non, then i == j
                        runlen = i - j + 1

                        # only do it if even (counting i)
                        if (runlen == 1) or runlen % 2 == 0:
                            counts[(prev,first)] = counts.get((prev,first), 0) - multiplier

                    # is there anything beyond our run
                    if k+2 < len(tokens):
                        after = tokens[k+2]

                        if verbose:
                            print("after:", after)

                        # don't have (first,second) after so this count goes down
                        counts[(combined,after)] = counts.get((combined,after), 0) + multiplier

                        # now this only goes up if we don't have an odd length run to the right of second
                        j = k+1   # start on second
                        while (j+1 < len(tokens)) and tokens[j+1] == second:
                            j += 1

                        runlen = j - (k+1) + 1

                        # only do it if even (counting from i+1)
                        if (runlen == 1) or (runlen % 2 == 0):
                            counts[(second,after)] = counts.get((second,after), 0) - multiplier
                        
                    # skip over the runs of (first,second)
                    i = k+2

                else:
                    # advance to the next one
                    i += 1 
                    
            return merge_cnt

    # return the pair counts without any side effects
    def _calc_pair_counts(self, text_chunks: list[list[bytes]], text_counts: list[int]) -> tuple[dict[tuple[bytes, bytes], int], Any]:
        """
        Build pair counts from scratch.
        Returns: (pair_counts_dict, pair_counts_heap)
            - pair_counts_dict: simple dict mapping pair -> count (for verification)
            - pair_counts_heap: heapdict with composite priority (for actual use)
        """
        # First build regular dict of counts
        pair_counts_dict : dict[tuple[bytes,bytes], int] = {}
        for tokens, cnt in zip(text_chunks, text_counts):
            # passing in stats will update it in place, adding up counts,
            # incrementing each occurence by cnt
            # be sure to use get_stats here to handle overlapping tokens correctly
            self.get_stats(tokens, pair_counts_dict, cnt)

        # Convert to heapdict with composite priority
        pair_counts_heap = heapdict()
        for pair, count in pair_counts_dict.items():
            both_unlocked = self.unlocked[pair[0]] and self.unlocked[pair[1]]
            priority = (-int(both_unlocked), -count, pair)
            pair_counts_heap[pair] = priority

        return pair_counts_dict, pair_counts_heap

    # get initial pair_counts, which we'll keep updating
    def initial_pair_counts(self, text_chunks: list[list[bytes]], text_counts: list[int]) -> None:

        _, self.pair_counts = self._calc_pair_counts(text_chunks, text_counts)

        # token_to_pairs initialization removed — unlocking mechanism no longer needed.
        # The count-based competition ensures parents are created before their supermerges.

    # double check our dynamic pair counts are being updated correctly
    def verify_pair_counts(self, text_chunks: list[list[bytes]], text_counts: list[int]) -> None:
        from_scratch_counts, _ = self._calc_pair_counts(text_chunks, text_counts)

        # Extract counts from current pair_counts heapdict
        current_counts = {pair: self.get_pair_count(pair) for pair in self.pair_counts.keys()}

        verify_dicts(from_scratch_counts, current_counts)

        # Note: pair_counts now contains ALL pairs (locked and unlocked)
        # Filtering by unlock status happens in choose_best_pair()
            
    # compute single counts from scratch, and return them
    def _calc_single_counts(self, text_chunks: list[list[bytes]], text_counts: list[int], verbose: bool = False) -> dict[bytes, int]:

        single_counts : dict[bytes, int] = {}
        for tokens, cnt in zip(text_chunks, text_counts):
            for tok in tokens:
                single_counts[tok] = single_counts.get(tok, 0) + cnt

        return single_counts

    # set up the single token counts, which also update over time
    def initial_single_counts(self, text_chunks: list[list[bytes]], text_counts: list[int]) -> None:
        self.single_counts = self._calc_single_counts(text_chunks, text_counts, verbose=True)

    # double check our dynamic single counts are being updated correctly
    def verify_single_counts(self, text_chunks: list[list[bytes]], text_counts: list[int]) -> None:
        from_scratch = self._calc_single_counts(text_chunks, text_counts)
        verify_dicts(from_scratch, self.single_counts)

    def _calc_token_locations(self, text_chunks: list[list[bytes]]) -> dict[bytes, set[int]]:
        token_locations: dict[bytes, set[int]] = defaultdict(set)
        for chunk_idx, tokens in enumerate(text_chunks):
            for token in set(tokens):  # use set to avoid duplicates within chunk
                token_locations[token].add(chunk_idx)
        return token_locations

    # set up the token_locations index mapping tokens to chunk indices
    def initial_token_locations(self, text_chunks: list[list[bytes]]) -> None:
        self.token_locations = self._calc_token_locations(text_chunks)

    def verify_token_locations(self, text_chunks: list[list[bytes]]) -> None:
        """
        Verify that token_locations index is consistent with text_chunks.
        """
        # Build expected token_locations from text_chunks
        expected = self._calc_token_locations(text_chunks)

        # Check all tokens in expected match token_locations
        all_tokens = set(expected.keys()) | set(self.token_locations.keys())
        for token in all_tokens:
            expected_chunks = expected.get(token, set())
            actual_chunks = self.token_locations.get(token, set())
            if expected_chunks != actual_chunks:
                print(f"token_locations mismatch for token {frombytes(token)}:")
                print(f"  expected chunks: {expected_chunks}")
                print(f"  actual chunks: {actual_chunks}")
                missing = expected_chunks - actual_chunks
                extra = actual_chunks - expected_chunks
                if missing:
                    print(f"  missing: {missing}")
                if extra:
                    print(f"  extra (stale): {extra}")
            assert expected_chunks == actual_chunks, f"token_locations mismatch for token"

    def _calc_whole_words(self, text_chunks: list[list[bytes]], text_counts: list[int]) -> int:
        whole_words = 0
        for tokens, cnt in zip(text_chunks, text_counts):
            if len(tokens) == 1:
                whole_words += cnt
        return whole_words

    def initial_whole_words(self, text_chunks: list[list[bytes]], text_counts: list[int]) -> None:
        self.whole_words = self._calc_whole_words(text_chunks, text_counts)

    def verify_whole_words(self, text_chunks: list[list[bytes]], text_counts: list[int]) -> None:
        from_scratch = self._calc_whole_words(text_chunks, text_counts)

        if self.whole_words != from_scratch:
            print("WARNING: whole_words mismatch:", self.whole_words, from_scratch)
        assert self.whole_words == from_scratch

    def verify_unlocked(self) -> None:
        """
        Verify that pair_counts priorities are consistent with actual unlocked status.

        For each pair in pair_counts, the stored priority's both_unlocked flag should
        match the actual unlocked status of both tokens.
        """
        for pair, priority in self.pair_counts.items():
            neg_both_unlocked, neg_count, stored_pair = priority

            # Verify stored pair matches key
            assert pair == stored_pair, f"Pair key mismatch: {pair} vs {stored_pair}"

            # Verify both_unlocked matches actual status
            actual_both_unlocked = self.unlocked[pair[0]] and self.unlocked[pair[1]]
            stored_both_unlocked = (neg_both_unlocked == -1)

            if actual_both_unlocked != stored_both_unlocked:
                print(f"Unlocked mismatch for pair ({frombytes(pair[0])}, {frombytes(pair[1])}):")
                print(f"  stored priority says both_unlocked={stored_both_unlocked}")
                print(f"  actual: unlocked[{frombytes(pair[0])}]={self.unlocked[pair[0]]}, unlocked[{frombytes(pair[1])}]={self.unlocked[pair[1]]}")
            assert actual_both_unlocked == stored_both_unlocked, "Unlocked status mismatch in pair_counts priority"

    def verify_token_to_pairs(self) -> None:
        """
        Verify that token_to_pairs index is consistent with pair_counts.

        Only applicable for BoundlessBPE (is_super=True, superbpe_mode=False).
        """
        if not (self.inf_data.is_super and not self.inf_data.superbpe_mode):
            return

        # Build expected token_to_pairs from pair_counts
        expected: dict[bytes, set[tuple[bytes, bytes]]] = {}
        for pair in self.pair_counts.keys():
            if pair[0] not in expected:
                expected[pair[0]] = set()
            if pair[1] not in expected:
                expected[pair[1]] = set()
            expected[pair[0]].add(pair)
            expected[pair[1]].add(pair)

        # Check all tokens in expected are in token_to_pairs with correct pairs
        for token, expected_pairs in expected.items():
            actual_pairs = self.token_to_pairs[token]
            if expected_pairs != actual_pairs:
                print(f"token_to_pairs mismatch for token {frombytes(token)}:")
                print(f"  expected: {len(expected_pairs)} pairs")
                print(f"  actual: {len(actual_pairs)} pairs")
                missing = expected_pairs - actual_pairs
                extra = actual_pairs - expected_pairs
                if missing:
                    print(f"  missing: {missing}")
                if extra:
                    print(f"  extra: {extra}")
            assert expected_pairs == actual_pairs, f"token_to_pairs mismatch for token"

        # Check no extra tokens in token_to_pairs
        for token, pairs in self.token_to_pairs.items():
            if pairs:  # Only check non-empty sets
                assert token in expected, f"Token in token_to_pairs but not in any pair_counts pair"

    # verify it all
    def verify_state(self, text_chunks: list[list[bytes]], text_counts: list[int]) -> None:
        assert self.vocab is not None
        self.vocab.verify_vocabulary()
        self.verify_pair_counts(text_chunks, text_counts)
        self.verify_single_counts(text_chunks, text_counts)
        self.verify_whole_words(text_chunks, text_counts)
        # verify_unlocked and verify_token_to_pairs skipped — unlocking mechanism removed
        self.verify_token_locations(text_chunks)
        self.inf_data.verify_indices()

    # compute our initial state from text_chunks and text_counts
    def initial_counts(self, text_chunks: list[list[bytes]], text_counts: list[int]) -> None:
        self.initial_pair_counts(text_chunks, text_counts)
        self.initial_single_counts(text_chunks, text_counts)
        self.initial_whole_words(text_chunks, text_counts)
        self.initial_token_locations(text_chunks)

    # a specialized version
    # this shouldn't change state
    def choose_best_pair(self) -> tuple[Optional[tuple[bytes, bytes]], int]:

        # Composite priority ensures unlocked pairs sort to top
        if len(self.pair_counts) == 0:
            return None, -1

        pair, priority = self.pair_counts.peekitem()
        neg_both_unlocked, neg_count, _ = priority

        # Check if top pair has both tokens unlocked
        if neg_both_unlocked != -1:
            # No unlocked pairs available
            return None, -1

        count = -neg_count

        # Verify both tokens are unlocked
        assert self.unlocked[pair[0]]
        assert self.unlocked[pair[1]]

        return pair, count

    def _process_pending_deletions(self) -> None:
        """Process any deletions at the front of the word operations queue.

        Deletions are processed immediately without competing with supermerges.
        Each deletion removes the token from vocab.
        """
        assert self.vocab is not None
        while self.current_word_op_idx < len(self.word_operations_list):
            idx, op_type, data = self.word_operations_list[self.current_word_op_idx]
            if op_type != 'delete':
                break
            token = data
            if token in self.vocab:
                self.vocab.delete(token)
            self.current_word_op_idx += 1

    def _get_next_regular_merge(self) -> tuple[int | None, tuple[bytes, bytes] | None, int, int]:
        """Get the next merge from word operations, processing any pending deletions first.

        Returns:
            (op_index, pair, c_ab, unlocked_flag) or (None, None, -1, 0) if no more merges
        """
        # First process any pending deletions
        self._process_pending_deletions()

        if self.current_word_op_idx >= len(self.word_operations_list):
            return None, None, -1, 0

        idx, op_type, data = self.word_operations_list[self.current_word_op_idx]
        assert op_type == 'merge', f"Expected merge at index {idx}, got {op_type}"
        pair, c_ab, unlocked_flag = data
        return idx, pair, c_ab, unlocked_flag

    def get_pair_count(self, pair: tuple[bytes, bytes]) -> int:
        """Extract count from pair_counts priority tuple."""
        if pair in self.pair_counts:
            _: int
            neg_count: int
            _, neg_count, _ = self.pair_counts[pair]
            return -neg_count
        return 0

    def apply_pair_count_changes(self, overall_change: dict[tuple[bytes, bytes], int]) -> None:
        """
        Apply deltas from overall_change to pair_counts (heapdict).
        Updates priorities and maintains token_to_pairs index.
        """
        for pair, delta in overall_change.items():
            if delta != 0:
                current_count = self.get_pair_count(pair)
                new_count = current_count + delta

                assert new_count >= 0, f"Pair count went negative: {pair} {current_count} + {delta} = {new_count}"

                if new_count == 0:
                    # Delete pair
                    if pair in self.pair_counts:
                        del self.pair_counts[pair]
                else:
                    # Update priority with new count
                    both_unlocked = self.unlocked[pair[0]] and self.unlocked[pair[1]]
                    priority = (-int(both_unlocked), -new_count, pair)
                    self.pair_counts[pair] = priority

    def merge_and_update(self, max_pair : tuple[bytes,bytes], text_chunks: list[list[bytes]], text_counts: list[int]) -> tuple[int, bytes | None]:

        # unpack once
        first, second = max_pair

        # these should have been unlocked
        assert self.unlocked[first], f"First token {first!r} must be unlocked"
        assert self.unlocked[second], f"Second token {second!r} must be unlocked"

        # pair count should be positive
        pair_count = self.get_pair_count(max_pair)
        assert pair_count > 0, f"Pair {max_pair} count should be positive before merging, got {pair_count}"

        # the combined changes of pair counts over all text_chunks
        # this way we can track the number of changing values from this merge
        # and only update those values in
        overall_change: dict[tuple[bytes, bytes], int] = {}

        # how many of the merged count were actually created
        # need this to be accurate even with overlapping merges
        total_merge_cnt = 0
        whole_word_increase = 0
        
        # is there a chunk that is a single token after merging max_pair
        new_unlocked = None

        # print("merge_and_update:", max_pair)

        combined = first + second

        # Get candidate chunks using token_locations index
        # Only iterate over chunks that contain both tokens
        # Must copy since we modify token_locations during iteration
        if first == second:
            candidates = list(self.token_locations.get(first, set()))
        else:
            candidates = list(self.token_locations.get(first, set()) & self.token_locations.get(second, set()))

        # Use fast path for is_super when first != second
        # The first == second case uses the slower path due to complex overlap handling
        if (first != second) and self.inf_data.is_super:

            # with no overlap we can do stuff faster
            for chunk_idx in candidates:
                tokens = text_chunks[chunk_idx]
                cnt = text_counts[chunk_idx]

                # handle the signs explicitly in the routines, so pass in positive
                # work directly with overall_change here
                merge_cnt_before = self.get_stats_faster(first, second, tokens, counts=overall_change, multiplier=cnt)

                # anything to do?
                if merge_cnt_before > 0:

                    before_len = len(tokens)

                    # then merge each chunk independently
                    # replace all occurrences of pair in tokens with the combined
                    # the ending number may be less than the number of pairs
                    # this is the unweighted number of merges
                    newtokens, merge_cnt = merge(tokens, max_pair)

                    if merge_cnt_before != merge_cnt:
                        print("warning merge_cnt different:", first, second, merge_cnt_before, merge_cnt)

                    # use the slice notation so the change persists outside of the iteration
                    tokens[:] = newtokens

                    # this tokens appears cnt times in the pre-tokenization
                    total_merge_cnt += merge_cnt*cnt

                    if len(tokens) == 1 and (before_len > 1):
                        whole_word_increase += cnt

                        # only merge space words
                        new_unlocked = combined

                    # Update token_locations index
                    # O(5) checks since chunks have ~5 tokens
                    if first not in tokens:
                        self.token_locations[first].discard(chunk_idx)
                        # Clean up empty sets to avoid memory bloat
                        if not self.token_locations[first]:
                            del self.token_locations[first]
                    if second not in tokens:
                        self.token_locations[second].discard(chunk_idx)
                        if not self.token_locations[second]:
                            del self.token_locations[second]
                    self.token_locations[combined].add(chunk_idx)
        else:

            # Slower path for regular BPE or first == second case
            for chunk_idx in candidates:
                tokens = text_chunks[chunk_idx]
                cnt = text_counts[chunk_idx]

                # Try the merge - returns merge_cnt=0 if pair not adjacent
                # This replaces the old 'possible' check since merge() does the same scan
                newtokens, merge_cnt = merge(tokens, max_pair)

                if merge_cnt > 0:
                    # Get the before state (tokens still has old values)
                    local_delta = self.get_stats(tokens, counts=None, multiplier=-cnt)

                    before_len = len(tokens)

                    # use the slice notation so the change persists outside of the iteration
                    tokens[:] = newtokens

                    # this tokens appears cnt times in the pre-tokenization
                    total_merge_cnt += merge_cnt*cnt

                    # and finally, increment local_delta for each ending pair
                    self.get_stats(tokens, counts=local_delta, multiplier=cnt)

                    # copy over the local ones to the overall change
                    # ignore the ones that cancelled out, which should be many
                    for pair, delta in local_delta.items():
                        if delta != 0:
                            overall_change[pair] = overall_change.get(pair, 0) + delta

                            if overall_change[pair] == 0:
                                del overall_change[pair]

                    if len(tokens) == 1 and (before_len > 1):
                        whole_word_increase += cnt

                        # only merge space words
                        new_unlocked = combined

                    # Update token_locations index
                    # O(5) checks since chunks have ~5 tokens
                    if first not in tokens:
                        self.token_locations[first].discard(chunk_idx)
                        # Clean up empty sets to avoid memory bloat
                        if not self.token_locations[first]:
                            del self.token_locations[first]
                    if first != second and second not in tokens:
                        self.token_locations[second].discard(chunk_idx)
                        if not self.token_locations[second]:
                            del self.token_locations[second]
                    self.token_locations[combined].add(chunk_idx)

        # Unlock the merged token BEFORE applying pair count changes
        # This maintains the invariant: if both inputs are unlocked, output is unlocked
        # need this even if new_unlocked is None
        # (for super merge training; in regular BPE everything is already unlocked via defaultdict)
        if self.inf_data.is_super:
            merged_token = first + second
            self.unlocked[merged_token] = True

        # apply the deltas to our self.pair_counts (heapdict)
        self.apply_pair_count_changes(overall_change)

        # should no longer have max_pair
        if max_pair in self.pair_counts:
            count = self.get_pair_count(max_pair)
            print("WARNING: merged pair still present in pair_counts:", frombytespair(max_pair), count)
        assert max_pair not in self.pair_counts

        # Update the single counts that changed because of the merge
        merged_pair = first + second
        if merged_pair in self.single_counts:
            print("warning 1:", frombytes(merged_pair), frombytespair(max_pair), "was", self.single_counts[merged_pair], total_merge_cnt)
        # assert merged_pair not in self.single_counts
        self.single_counts[merged_pair] = self.single_counts.get(merged_pair, 0) + total_merge_cnt

        # and decrease the count of first and second by the same amount
        # still works if first == second
        self.single_counts[first] -= total_merge_cnt
        self.single_counts[second] -= total_merge_cnt

        if self.single_counts[first] == 0:
            del self.single_counts[first]

        # we don't want to delete it twice if first == second
        if first != second and self.single_counts[second] == 0:
            del self.single_counts[second]

        self.whole_words += whole_word_increase

        # Note: We already unlocked new_unlocked before apply_pair_count_changes
        # so the priorities are correct. No need to call update_unlocked_token_priorities here.
        # That function is only needed when processing regular merges that unlock existing tokens.

        # the single counts decreased by total_merge_cnt
        return total_merge_cnt, new_unlocked

    # delete a particular token, splitting all occurences into single bytes
    # update counts accordingly
    # Note: deletions only occur in first-pass BPE where all tokens are unlocked
    def delete_and_update(self, bad_token : bytes, text_chunks: list[list[bytes]], text_counts: list[int]) -> None:

        parts = self.inf_data.get_replacement_parts(bad_token)

        # if ios==1 then we already deleted them all
        # but still need to delete from the vocab
        expected_cnt = self.single_counts.get(bad_token, 0)

        # the combined changes over all text_chunks
        # this way we can track the number of changing values from this deletion
        # and only update those values that changed
        overall_change: dict[tuple[bytes, bytes], int] = {}

        # how many of the merged count were actually created
        # need this to be accurate even with overlapping merges
        total_delete_cnt = 0
        # the number of whole words will be non-positive
        whole_word_increase = 0

        # Get candidate chunks using token_locations index
        # need a copy since we modify self.token_locations[bad_token]
        candidates = self.token_locations.get(bad_token, set()).copy()

        for chunk_idx in candidates:
            tokens = text_chunks[chunk_idx]
            cnt = text_counts[chunk_idx]

            # just find the deltas in the current text_chunk
            # handling overlapping tokens correctly
            # this is the change in *pairs*
            local_delta = self.get_stats(tokens, counts=None, multiplier=-cnt)

            before_len = len(tokens)

            # replace all occurrences of bad_token with its parts
            newtokens, deletions = blow_up(tokens, bad_token, parts)
            # use the slice notation so the change persists outside of the iteration
            tokens[:] = newtokens

            # tally how many deletions we did
            total_delete_cnt += deletions*cnt

            # and finally, increment local_delta for each ending pairs
            self.get_stats(tokens, counts=local_delta, multiplier=cnt)
            # copy over the local ones to the overall change
            # ignore the ones that cancelled out, which should be many
            for pair, delta in local_delta.items():
                if delta != 0:
                    overall_change[pair] = overall_change.get(pair, 0) + delta

                    if overall_change[pair] == 0:
                        del overall_change[pair]

            # did we go from a single token to more
            if before_len == 1 and len(tokens) > 1:
                whole_word_increase -= cnt

            # Update token_locations index
            # bad_token is gone, parts are added
            self.token_locations[bad_token].discard(chunk_idx)
            # Clean up empty sets to avoid memory bloat
            if not self.token_locations[bad_token]:
                del self.token_locations[bad_token]
            for part in parts:
                self.token_locations[part].add(chunk_idx)

        if expected_cnt != total_delete_cnt:
            print("WARNING: delete count mismatch:", frombytes(bad_token), expected_cnt, total_delete_cnt, cnt)
        assert expected_cnt == total_delete_cnt

        # adjust the whole words too 
        self.whole_words += whole_word_increase  #  a negative increase

        # apply the deltas to our self.pair_counts (heapdict)
        self.apply_pair_count_changes(overall_change)

        # if ios == 1, then bad_token was deleted already from self.single_counts
        # e.g. deleting Ġsugg after the merger of Ġsugg est
        if bad_token in self.single_counts:
            if self.single_counts[bad_token] != total_delete_cnt:
                print("WARNING: single count mismatch on delete:", frombytes(bad_token), self.single_counts[bad_token], total_delete_cnt)
            self.single_counts[bad_token] -= total_delete_cnt
            # only keep single_counts with nonzero counts, so delete this
            assert self.single_counts[bad_token] == 0
            del self.single_counts[bad_token]

            # increase the single counts of the single bytes or pair of parts
            # if it has repeated bytes we'll just do them separately
            # in the ios==1 case we have total_delete_cnt==0, so this is a no op
            # can be inside the if statement
            for tok in parts:
                before = self.single_counts.get(tok, 0)
                self.single_counts[tok] = before + total_delete_cnt
                # print("singles:", tok, before, total_delete_cnt)

        if bad_token in self.inf_data.deletions.values():
            print("WARNING: token already in deletions:", frombytes(bad_token))

        # save the deletion event
        self.inf_data.deletions[self.index_gen.get_next_index()] = bad_token


    # how many tokens are single bytes in the current tokenization
    def get_single_byte_cnt(self) -> int:
        return sum([self.single_counts.get(tok, 0) for tok in self.single_bytes])

    # update priorities when a token is newly unlocked
    def update_unlocked_token_priorities(self, new_unlocked: bytes) -> None:
        """
        When a token is newly unlocked, update priorities for all pairs involving it.
        Uses token_to_pairs index for O(k) instead of O(n) performance.
        Only called during super merge training.
        """
        assert self.inf_data.is_super, "update_unlocked_token_priorities should only be called during super merge training"
        assert not self.inf_data.superbpe_mode, "update_unlocked_token_priorities not required with superbpe_mode"
        assert self.unlocked[new_unlocked], f"Token {new_unlocked!r} should be unlocked before calling this"

        # Get pairs involving this token from index
        pairs_to_update = self.token_to_pairs[new_unlocked]

        # Update priority for each pair
        for pair in pairs_to_update:
            # Pair should exist in pair_counts
            assert pair in self.pair_counts, f"Pair {pair} in token_to_pairs but not in pair_counts"

            count = self.get_pair_count(pair)
            assert count > 0, f"Pair {pair} in pair_counts but has non-positive count {count}"

            # Recompute priority with updated unlock status
            both_unlocked = self.unlocked[pair[0]] and self.unlocked[pair[1]]
            priority = (-int(both_unlocked), -count, pair)
            self.pair_counts[pair] = priority

    # do a merge of best_pair, followed by a potential deletion
    def merge_and_delete(self, best_pair: tuple[bytes, bytes], i: int, text_chunks: list[list[bytes]], text_counts: list[int], print_row: bool, verbose: bool) -> tuple[float, float, bytes | None]:

            start_merge = time.time()

            left, right = best_pair

            assert self.unlocked[left], frombytes(left) + "," + str(self.inf_data.is_super) + str(frombytespair(best_pair))
            assert self.unlocked[right], frombytes(right) + "," + str(self.inf_data.is_super) + str(frombytespair(best_pair))

            # the c_ab and best_loss we used in choose_best_pair
            # to select best_pair may have changed by now
            # due to previous merges in batch, so recompute
            c_ab = self.get_pair_count(best_pair)
            # look up the corresponding single values to for output
            c_a = self.single_counts[left]
            c_b = self.single_counts[right]

            # compute the Intersection over Self metrics from
            # compute before merging
            ios_a = c_ab/c_a
            ios_b = c_ab/c_b

            # do the merges while updating pair_counts as appropriate
            total_merge_cnt, new_unlocked = self.merge_and_update(best_pair, text_chunks, text_counts)

            # did we merge what we expected to?
            if c_ab != total_merge_cnt:
                print("WARNING: merge count mismatch:", c_ab, total_merge_cnt, frombytespair(best_pair))
            assert c_ab == total_merge_cnt

            # create new token
            merged_tok = left + right

            # save the merge rule with next available index
            # keep the c_ab around as well to save
            # also save if the merge unlocked anything
            self.inf_data.merges[self.index_gen.get_next_index()] = (best_pair, c_ab, 0 if new_unlocked is None else 1)

            # Only add to vocab if not already present
            # (can happen when deletions blow up tokens creating new adjacencies
            # for a pair that was already merged)
            assert self.vocab is not None
            if merged_tok not in self.vocab:
                self.vocab.add(merged_tok, is_super=self.inf_data.is_super)
            elif verbose:
                print("info: duplicate merge (token exists):", frombytes(merged_tok), frombytespair(best_pair))

            if print_row:

                left_tok = frombytes(left)
                right_tok = frombytes(right)

                if new_unlocked is None:
                    nu = ""
                else:
                    nu = frombytes(new_unlocked)

                output = ["*", i+1, len(self.vocab), "m", \
                        left_tok, right_tok, c_ab, c_a, c_b, round(ios_a, 5), round(ios_b, 5), round(time.time() - start_merge, 5), \
                        len(self.pair_counts), \
                        self.get_single_byte_cnt(), self.whole_words, nu]
                print("\t".join([str(x) for x in output]))

            merge_time = time.time() - start_merge
            start_delete = time.time()

            # now see if we need delete any tokens
            for (ios, tok, direction) in [(ios_a, left, "left"), (ios_b, right, "right")]:
                # don't delete a single byte or a super merge
                # note if ios == 1 then we already deleted all occurences
                if (len(tok) > 1) and (ios >= self.inf_data.tau):

                    start_this_delete = time.time()

                    if verbose and tok in self.inf_data.deletions.values():
                        print("info: deleting token that was previously deleted (may have been recreated)", frombytes(tok))

                    # Populate deletion_parts before calling delete_and_update
                    if tok not in self.inf_data.deletion_parts:
                        if self.inf_data.blowup:
                            self.inf_data.deletion_parts[tok] = [bytes([b]) for b in tok]
                        else:
                            # Find the merge pair that created this token
                            parts = None
                            for ((left_m, right_m), cnt_m, unlocked_flag_m) in self.inf_data.merges.values():
                                if left_m + right_m == tok:
                                    parts = [left_m, right_m]
                                    break
                            assert parts is not None, "couldn't find merge for " + str(tok)
                            self.inf_data.deletion_parts[tok] = parts

                    self.delete_and_update(tok, text_chunks, text_counts)

                    # and delete from vocab
                    assert self.vocab is not None
                    if tok in self.vocab:
                        self.vocab.delete(tok)

                    if print_row:

                        if new_unlocked is None:
                            nu = ""
                        else:
                            nu = frombytes(new_unlocked)

                        output = ["*", i+1, len(self.vocab), "d", \
                            frombytes(tok), direction, c_ab, c_a, c_b, round(ios, 5), round(self.inf_data.tau, 5), round(time.time() - start_this_delete, 5), \
                            len(self.pair_counts), \
                            self.get_single_byte_cnt(), self.whole_words, nu]

                        print("\t".join([str(x) for x in output]))

            delete_time = time.time() - start_delete

            return merge_time, delete_time, new_unlocked


    # _____________________________________________
    # Helper methods for training
    # _____________________________________________

    def _print_checkpoint_stats(self, outprefix: str, i: int, overall_start: float,
                                 total_pretok: float, total_ic: float, total_max_value: float,
                                 total_merge: float, total_delete: float, total_unlocked: float,
                                 total_verify: float, total_checkpoint: float, total_init: float = 0.0,
                                 total_ngram_split: float = 0.0) -> None:
        """Print timing stats for checkpoint."""
        assert self.vocab is not None
        overall_time = time.time() - overall_start

        print(":i:", i)
        print(":len(vocab):", len(self.vocab))
        print(":single_counts:", len(self.single_counts))
        print(":pair_counts:", len(self.pair_counts))
        print(":single_byte_cnt:", self.get_single_byte_cnt())
        print(":whole_words:", self.whole_words)
        print(":merges:", len(self.inf_data.merges))
        print(":deletions:", len(self.inf_data.deletions))
        if self.inf_data.is_super and self.current_word_op_idx < len(self.word_operations_list):
            print(":current_word_op_idx:", self.current_word_op_idx)
            print(":target_vocab_size:", self.target_vocab_size)
        print(":training time breakdown")
        print(":total_init:", total_init)
        print(":total_pretok:", total_pretok)
        print(":total_ngram_split:", total_ngram_split)
        print(":total_initalize_counts:", total_ic)
        print(":total_max_value:", total_max_value)
        print(":total_merge:", total_merge)
        print(":total_delete:", total_delete)
        print(":total_unlocked:", total_unlocked)
        print(":total_verify:", total_verify)
        print(":total_checkpoint:", total_checkpoint)
        print(":overall_time:", overall_time)
        print(":missing:", overall_time - total_init - total_pretok - total_ngram_split - total_ic - total_max_value - total_merge - total_delete - total_verify - total_unlocked - total_checkpoint)

        print(":outprefix:", outprefix + "_" + str(len(self.vocab)))


    # _____________________________________________
    # Training methods
    # _____________________________________________

    def _train_internal_bpe(self,
                            filepath: str,
                            outprefix: str,
                            num_lines: int,
                            vocab_size: int,
                            recalc: int,
                            max_bytes: int,
                            checkpoint_iterations: int,
                            verbose: bool,
                            progress_interval: int,
                            save_pretokens: Optional[str] = None) -> None:
        """First-pass BPE training with PickyBPE deletions.

        Note: tau and blowup are read from self.inf_data (set by caller before calling this method).
        """
        overall_start = time.time()

        # BPE-specific initialization
        init_start = time.time()

        # First pass: vocab starts with initial single bytes
        self.vocab = Vocabulary.create_initial()

        # Regular merge, so no need for locking, everything can merge
        self.unlocked = defaultdict(lambda: True)
        total_init = time.time() - init_start

        # Pretokenization
        start_pretok = time.time()
        text_chunks, text_counts = self.pretokenize(filepath, num_lines, max_bytes, save_pretokens, verbose)
        total_pretok = time.time() - start_pretok

        # Set up initial counts
        start_ic = time.time()
        self.initial_counts(text_chunks, text_counts)
        total_ic = time.time() - start_ic

        # Timing variables
        total_max_value = 0.0
        total_merge = 0.0
        total_delete = 0.0
        total_unlocked = 0.0
        total_verify = 0.0
        total_checkpoint = 0.0

        # Print header
        if progress_interval > 0:
            header = ["*", "i", "vocab", "type",
                     "left", "rght", "c_ab", "c_a", "c_b", "ios_a", "ios_b", "time",
                     "pairs", "single_bytes", "whole_words", "new_unlocked"]
            print("\t".join(header))

        i = 0

        # Main BPE training loop
        while True:
            start_max = time.time()
            best_pair, best_cnt = self.choose_best_pair()
            total_max_value += (time.time() - start_max)

            # best_pair could be None if no pairs left, but merge_and_delete handles it
            assert best_pair is not None, "best_pair should not be None in BPE training"

            print_row = progress_interval > 0 and i % progress_interval == 0
            merge_time, delete_time, new_unlocked = self.merge_and_delete(
                best_pair, i, text_chunks, text_counts, print_row, verbose
            )

            total_merge += merge_time
            total_delete += delete_time

            # Periodic verification
            if recalc > 0 and i % recalc == 0 and i > 0:
                verify_start = time.time()
                self.verify_state(text_chunks, text_counts)
                total_verify += time.time() - verify_start

            i += 1

            # Check stopping conditions
            should_stop = False
            stop_reason = ""

            if len(self.pair_counts) == 0:
                should_stop = True
                stop_reason = f"only single element chunks at iteration {i}"
            elif len(self.vocab) >= vocab_size:
                should_stop = True
                stop_reason = f"reached vocab_size {vocab_size}"

            # Checkpoint saving - self.save() routes to inf_data.save for is_super=False
            if (checkpoint_iterations > 0 and len(self.vocab) % checkpoint_iterations == 0 and len(self.vocab) > 0) or should_stop:
                checkpoint_start = time.time()
                self.save(outprefix + "_" + str(len(self.vocab)))
                total_checkpoint += time.time() - checkpoint_start
                if verbose:
                    self._print_checkpoint_stats(
                        outprefix, i, overall_start,
                        total_pretok, total_ic, total_max_value, total_merge,
                        total_delete, total_unlocked, total_verify, total_checkpoint, total_init
                    )

            if should_stop:
                print(f"Stopping: {stop_reason}")
                break

    def _train_internal_super(self,
                              filepath: str,
                              outprefix: str,
                              num_lines: int,
                              vocab_size: int,
                              recalc: int,
                              word_model_file: Optional[str],
                              superbpe_mode: bool,
                              max_bytes: int,
                              checkpoint_iterations: int,
                              verbose: bool,
                              progress_interval: int,
                              save_pretokens: Optional[str] = None,
                              greedy_split: bool = False,
                              min_count: int = 15,
                              max_ngram_len: int = 30) -> None:
        """Second-pass training (SuperBPE or BoundlessBPE)."""
        overall_start = time.time()

        # Super-specific initialization
        init_start = time.time()

        # Load the word model
        assert word_model_file is not None
        self.word_model = Tokenizer()
        self.word_model.load(word_model_file)

        # Target vocab size is the final vocab size of the original word model
        assert self.word_model.vocab is not None
        self.target_vocab_size = len(self.word_model.vocab)

        if superbpe_mode:
            # SuperBPE: vocab starts with copy of word model vocab
            # vocab_size parameter = number of supermerges to add on top
            self.vocab = Vocabulary.create_from_vocab(self.word_model.vocab)
            # Not used in SuperBPE mode, but initialize anyway
            self.word_operations_list = []
            self.current_word_op_idx = 0
        else:
            # BoundlessBPE: vocab starts with single bytes (same as first pass)
            # We'll add regular merges and supermerges, and process deletions
            self.vocab = Vocabulary.create_initial()

            # Build unified operations list from word model (merges and deletions sorted by index)
            assert self.word_model.words is not None
            self.word_operations_list = []
            for idx, (pair, c_ab, unlocked_flag) in self.word_model.words.merges.items():
                self.word_operations_list.append((idx, 'merge', (pair, c_ab, unlocked_flag)))
            for idx, token in self.word_model.words.deletions.items():
                self.word_operations_list.append((idx, 'delete', token))
            self.word_operations_list.sort(key=lambda x: x[0])
            self.current_word_op_idx = 0

        # Note: IndexGenerator was already reset by _reset_training_state()
        # Supermerges use independent index space starting from 0

        # Unlock all word vocab tokens — the count-based competition naturally
        # ensures parents are created before any supermerge that uses them.
        # Previously single bytes were never unlocked (bug), and the unlock-on-replay
        # mechanism for merged tokens was unnecessary overhead.
        self.unlocked = defaultdict(lambda: False)
        for tok in self.word_model.vocab.tokens:
            self.unlocked[tok] = True
        total_init = time.time() - init_start

        # Pretokenization
        start_pretok = time.time()
        assert word_model_file is not None
        text_chunks, text_counts = self.pretokenize_super(filepath, num_lines, max_bytes, save_pretokens, verbose)
        total_pretok = time.time() - start_pretok

        # Optional n-gram greedy split
        if greedy_split:
            from .ngram_split import ngram_split as _ngram_split
            start_ngram = time.time()
            text_chunks, text_counts = _ngram_split(text_chunks, text_counts,
                self.word_model, min_count, max_ngram_len)
            total_ngram_split = time.time() - start_ngram
        else:
            total_ngram_split = 0.0

        # Set up initial counts
        start_ic = time.time()
        self.initial_counts(text_chunks, text_counts)
        total_ic = time.time() - start_ic

        # Continue timing variables
        total_max_value = 0.0
        total_merge = 0.0
        total_delete = 0.0
        total_unlocked = 0.0
        total_verify = 0.0
        total_checkpoint = 0.0

        # Print header
        if progress_interval > 0:
            header = ["*", "i", "vocab", "type",
                     "left", "rght", "c_ab", "c_a", "c_b", "ios_a", "ios_b", "time",
                     "pairs", "single_bytes", "whole_words", "new_unlocked"]
            print("\t".join(header))

        i = 0
        # Cache the best supermerge across iterations. Since regular merge
        # replays don't change the supermerge heap (all tokens pre-unlocked),
        # we can reuse the cached result instead of re-peeking each time.
        cached_super = None

        # Main Super training loop
        while True:
            # Time each iteration

            print_row = progress_interval > 0 and i % progress_interval == 0

            # Get best supermerge — use cache if available
            if cached_super is not None:
                super_pair, super_c_ab = cached_super
                cached_super = None
            else:
                start_max = time.time()
                super_pair, super_c_ab = self.choose_best_pair()
                total_max_value += (time.time() - start_max)

            if superbpe_mode:
                # SuperBPE: just apply supermerge
                if super_pair is not None:
                    merge_time, delete_time, _ = self.merge_and_delete(
                        super_pair, i, text_chunks, text_counts, print_row, verbose
                    )
                    total_merge += merge_time
                    total_delete += delete_time
            else:
                # BoundlessBPE: compete with regular merges
                start_next_candidate = time.time()
                reg_op_idx, reg_pair, reg_c_ab, reg_unlocked_flag = \
                    self._get_next_regular_merge()
                time_next_candidate = time.time() - start_next_candidate

                # Strict > (not >=): ties go to the regular merge. This is
                # required because with all tokens pre-unlocked, a supermerge
                # could tie with the regular merge that creates one of its
                # parents. The parent must be created first.
                if (super_c_ab > reg_c_ab) and (super_pair is not None):
                    # Supermerge wins — apply it
                    assert super_pair[0] in self.vocab, f"supermerge parent not in vocab: {frombytes(super_pair[0])}"
                    assert super_pair[1] in self.vocab, f"supermerge parent not in vocab: {frombytes(super_pair[1])}"
                    merge_time, delete_time, _ = self.merge_and_delete(
                        super_pair, i, text_chunks, text_counts, print_row, verbose
                    )
                    total_merge += (merge_time + time_next_candidate)
                    total_delete += delete_time
                    # Supermerge changed the heap — don't cache
                elif reg_pair is not None:
                    # Regular merge wins — replay it (just add token to vocab)
                    assert self.vocab is not None
                    merged_token = reg_pair[0] + reg_pair[1]

                    if merged_token not in self.vocab:
                        self.vocab.add(merged_token, is_super=False)
                    elif verbose:
                        print("info: duplicate regular merge (token exists):", frombytes(merged_token), frombytespair(reg_pair))

                    self.current_word_op_idx += 1
                    self._process_pending_deletions()

                    # Heap unchanged — cache for next iteration
                    cached_super = (super_pair, super_c_ab)

            # Periodic verification
            if recalc > 0 and i % recalc == 0 and i > 0:
                verify_start = time.time()
                self.verify_state(text_chunks, text_counts)
                total_verify += time.time() - verify_start

            i += 1

            # Check stopping conditions
            should_stop = False
            stop_reason = ""

            if len(self.pair_counts) == 0:
                should_stop = True
                stop_reason = f"only single element chunks at iteration {i}"
            elif superbpe_mode:
                # SuperBPE: count supermerges added (vocab size - word model size)
                num_supermerges = len(self.vocab) - self.target_vocab_size
                if num_supermerges >= vocab_size:
                    should_stop = True
                    stop_reason = f"reached {vocab_size} supermerges"
            else:
                # BoundlessBPE stopping - same as first pass, use net vocab size
                if len(self.vocab) >= self.target_vocab_size:
                    should_stop = True
                    stop_reason = f"reached target vocab_size {self.target_vocab_size}"

            # Checkpoint saving - self.save() routes to save_two_pass for is_super=True
            if (checkpoint_iterations > 0 and len(self.vocab) % checkpoint_iterations == 0 and len(self.vocab) > 0) or should_stop:
                checkpoint_start = time.time()
                self.save(outprefix + "_" + str(len(self.vocab)))
                total_checkpoint += time.time() - checkpoint_start
                if verbose:
                    self._print_checkpoint_stats(
                        outprefix, i, overall_start,
                        total_pretok, total_ic, total_max_value, total_merge,
                        total_delete, total_unlocked, total_verify, total_checkpoint, total_init,
                        total_ngram_split
                    )

            if should_stop:
                print(f"Stopping: {stop_reason}")
                break

    def save_two_pass(self, file_prefix: str) -> None:
        """
        Save two-pass model (BoundlessBPE or SuperBPE) as unified file.
        Format: header, vocabulary, special_tokens, words section, superwords section

        - BoundlessBPE (superbpe_mode=False): Keeps word operations up to current_word_op_idx
        - SuperBPE (superbpe_mode=True): Keeps all N word operations, adds M supermerges

        Args:
            file_prefix: Path prefix for output file (will add .model extension)
        """
        assert self.inf_data.is_super, "save_two_pass() only for two-pass training (is_super=True)"
        assert self.word_model is not None, "word_model must be loaded"
        assert self.word_model.words is not None, "word_model.words must be loaded"
        assert self.word_model.vocab is not None, "word_model.vocab must be loaded"
        assert self.vocab is not None, "vocab must exist"

        num_supermerges = len(self.inf_data.merges)
        original_word_vocab_size = len(self.word_model.vocab)

        # Trim word model InferenceData if BoundlessBPE
        trimmed_words = copy.deepcopy(self.word_model.words)

        if self.inf_data.superbpe_mode:
            # SuperBPE: keep all word operations, add supermerges on top
            mode_name = "SuperBPE"
            model_type = "superbpe"
        else:
            # BoundlessBPE: trim word operations, replace with supermerges
            mode_name = "BoundlessBPE"
            model_type = "boundless"
            trimmed_words.trim_operations_to(self.current_word_op_idx)

        # Copy vocabulary for saving
        vocab_for_save = Vocabulary.create_from_vocab(self.vocab)

        # Collect all special tokens: from first-pass word model + any pending from second pass
        all_special_tokens: list[str] = []

        # First, preserve special tokens from the original word model
        if self.word_model.vocab.special_tokens:
            sorted_original = sorted(self.word_model.vocab.special_tokens.items(), key=lambda x: x[1])
            all_special_tokens.extend([tok for tok, _ in sorted_original])

        # Then add any pending special tokens from second pass
        for tok in self._pending_special_tokens:
            if tok not in all_special_tokens:
                all_special_tokens.append(tok)

        # Register special tokens on vocabulary
        if all_special_tokens:
            vocab_for_save.register_special_tokens(all_special_tokens)

        print(f"Saving {mode_name} model: {file_prefix}.model")
        print(f"  - Vocabulary: {len(vocab_for_save)} tokens" +
              (f" (trimmed from {original_word_vocab_size})" if not self.inf_data.superbpe_mode else ""))
        print(f"  - Supermerges: {num_supermerges}")
        if vocab_for_save.special_tokens:
            print(f"  - Special tokens: {len(vocab_for_save.special_tokens)}")
        print(f"  - Total vocab size: {vocab_for_save.total_size()}")

        model_file = file_prefix + ".model"
        with open(model_file, 'wt') as f:
            # Write version header
            f.write(f"BoundlessBPE v2 {model_type}\n")

            # Write unified vocabulary section
            vocab_for_save.save(f, self.single_counts)

            # Write words section
            f.write("words\n")
            trimmed_words._write_to_file(f)

            # Write superwords section
            f.write("superwords\n")
            self.inf_data._write_to_file(f)

    # Delegate methods for compatibility
    def register_special_tokens(self, special_tokens: list[str]) -> None:
        """
        Register special tokens.

        For single-pass (is_super=False): registers on vocab directly
        For two-pass (is_super=True): stores pending, registered at save time
        """
        if self.inf_data.is_super:
            # Two-pass: store pending tokens - will be registered with correct indices at save time
            self._pending_special_tokens = special_tokens
        else:
            # Single-pass: register on vocab directly
            assert self.vocab is not None, "vocab must exist"
            self.vocab.register_special_tokens(special_tokens)

    def save(self, file_prefix: str) -> None:
        """
        Save the model. Automatically determines correct format:
        - is_super=False: saves as single-pass "word" model
        - is_super=True: saves as two-pass unified model (boundless or superbpe)

        Format: header, vocabulary, special_tokens, words section, (optional) superwords section
        """
        if self.inf_data.is_super:
            # Two-pass training: save unified file with word + superword sections
            self.save_two_pass(file_prefix)
        else:
            # Single-pass training: save as word model
            assert self.vocab is not None, "vocab must exist"

            # Copy vocabulary for saving
            vocab_for_save = Vocabulary.create_from_vocab(self.vocab)

            # Copy special tokens if registered
            vocab_for_save.special_tokens = dict(self.vocab.special_tokens)
            vocab_for_save.inverse_special_tokens = dict(self.vocab.inverse_special_tokens)

            print(f"Saving word model: {file_prefix}.model")
            print(f"  - Vocabulary: {len(vocab_for_save)} tokens")
            if vocab_for_save.special_tokens:
                print(f"  - Special tokens: {len(vocab_for_save.special_tokens)}")
            print(f"  - Total vocab size: {vocab_for_save.total_size()}")

            model_file = file_prefix + ".model"
            with open(model_file, 'wt') as f:
                # Write version header
                f.write("BoundlessBPE v2 word\n")

                # Write vocabulary section
                vocab_for_save.save(f, self.single_counts)

                # Write words section
                f.write("words\n")
                self.inf_data._write_to_file(f)



# _____________________________________________
# Concrete trainer classes
# _____________________________________________

class BpeTrainer(BaseBpeTrainer):
    """
    First-pass BPE trainer with PickyBPE deletions.

    Trains a standard BPE model with optional token deletions based on the IOS metric.
    All tokens can merge freely (no locking mechanism).

    Args:
        script_specific_scripts: Optional list of script names to tokenize character-by-character.
                                If None, uses default scripts.
        simple_regex: Optional regex pattern for non-script-specific tokenization.
                     If None, uses default pattern.
        script_specific_regex: Optional regex pattern for script-specific tokenization.
                              If None, uses default pattern.
    """
    
    def train(self,
              tau: float,
              filepath: str,
              outprefix: str,
              num_lines: int,
              vocab_size: int,
              recalc: int,
              blowup: bool,
              max_bytes: int = 1000000000,
              checkpoint_iterations: int = 8192,
              verbose: bool = True,
              progress_interval: Optional[int] = None,
              save_pretokens: Optional[str] = None) -> None:
        """
        Train a first-pass BPE model.

        Args:
            tau: Deletion threshold for PickyBPE (0.0-1.0). Higher = more aggressive deletions.
            filepath: Path to training data (JSONL format).
            outprefix: Output file prefix for saving models.
            num_lines: Number of documents to read from training file.
            vocab_size: Target vocabulary size.
            recalc: How often to verify counts from scratch (for debugging).
            blowup: If True, delete tokens by splitting to bytes. If False, split to merge pair.
            max_bytes: Maximum bytes to process from training file (default 1GB).
            checkpoint_iterations: Save intermediate model every N net vocab size (after deletions) (default 8192).
            verbose: If True, print extra diagnostics (per-document progress, pretokenization
                summaries, timing breakdowns). Warnings and the final summary always print.
            progress_interval: How often to print a `*` merge-progress row. None (default) prints
                every merge when verbose, else every DEFAULT_PROGRESS_INTERVAL merges. 0 disables
                progress rows entirely.
            save_pretokens: If not None, save pretokenization data to this file path.
        """
        assert vocab_size >= len(self.initial_vocab)

        progress_interval = _resolve_progress_interval(progress_interval, verbose)

        # Reset all training state (allows retraining on same instance)
        self._reset_training_state()

        # Store configuration in inference_data
        self.inf_data.tau = tau
        self.inf_data.is_super = False
        self.inf_data.superbpe_mode = False
        self.inf_data.blowup = blowup

        # Call BPE training implementation
        self._train_internal_bpe(filepath, outprefix, num_lines, vocab_size, recalc,
                                max_bytes, checkpoint_iterations, verbose, progress_interval,
                                save_pretokens)


class BoundlessBpeTrainer(BaseBpeTrainer):
    """
    BoundlessBPE trainer - second-pass training with merge competition.

    Loads a first-pass word model and trains supermerges that compete with regular merges.
    Tokens start locked and unlock when they form whole words. Regular merges advance the
    iterator but supermerges are actually applied. Final model contains exactly N total operations
    (some regular, some super), matching the first-pass model's operation count.

    Args:
        script_specific_scripts: Optional list of script names to tokenize character-by-character.
                                If None, uses default scripts.
        simple_regex: Optional regex pattern for non-script-specific tokenization.
                     If None, uses default pattern.
        script_specific_regex: Optional regex pattern for script-specific tokenization.
                              If None, uses default pattern.
    """
    
    def train(self,
              filepath: str,
              outprefix: str,
              num_lines: int,
              recalc: int,
              word_model_file: str,
              max_bytes: int = 1000000000,
              checkpoint_iterations: int = 8192,
              verbose: bool = True,
              progress_interval: Optional[int] = None,
              save_pretokens: Optional[str] = None,
              greedy_split: bool = False,
              min_count: int = 15,
              max_ngram_len: int = 30) -> None:
        """
        Train a BoundlessBPE model (second pass with competition).

        Args:
            filepath: Path to training data (JSONL format).
            outprefix: Output file prefix for saving models.
            num_lines: Number of documents to read from training file.
            recalc: How often to verify counts from scratch (for debugging).
            word_model_file: Path to first-pass word model (required).
            max_bytes: Maximum bytes to process from training file (default 1GB).
            checkpoint_iterations: Save intermediate model every N net vocab size (after deletions) (default 8192).
            verbose: If True, print extra diagnostics (per-document progress, pretokenization
                summaries, timing breakdowns). Warnings and the final summary always print.
            progress_interval: How often to print a `*` merge-progress row. None (default) prints
                every merge when verbose, else every DEFAULT_PROGRESS_INTERVAL merges. 0 disables
                progress rows entirely.
            save_pretokens: If not None, save superword pretokenization data to this file path.
            greedy_split: If True, apply n-gram greedy split before initial_counts().
            min_count: Minimum count floor for n-gram counting (default 15).
            max_ngram_len: Maximum n-gram length (default 30).
        """
        progress_interval = _resolve_progress_interval(progress_interval, verbose)

        # Reset all training state (allows retraining on same instance)
        self._reset_training_state()

        # BoundlessBPE mode: tokens start locked, supermerges compete with regular merges
        superbpe_mode = False

        # vocab_size unused since BoundlessBPE uses number of regular merges from the first pass
        vocab_size = -1

        # Store configuration in inference_data
        self.inf_data.tau = 1.1  # No deletions for BoundlessBPE
        self.inf_data.is_super = True
        self.inf_data.superbpe_mode = superbpe_mode
        self.inf_data.blowup = False

        # Call BoundlessBPE training implementation
        self._train_internal_super(filepath, outprefix, num_lines, vocab_size, recalc,
                                   word_model_file, superbpe_mode, max_bytes,
                                   checkpoint_iterations, verbose, progress_interval, save_pretokens,
                                   greedy_split, min_count, max_ngram_len)


class SuperBpeTrainer(BaseBpeTrainer):
    """
    SuperBPE trainer - second-pass training with supermerges only.

    Loads a first-pass word model and trains only supermerges (no competition with regular merges).
    All word-level tokens start unlocked. Final model contains all N word operations plus M supermerges.

    Args:
        script_specific_scripts: Optional list of script names to tokenize character-by-character.
                                If None, uses default scripts.
        simple_regex: Optional regex pattern for non-script-specific tokenization.
                     If None, uses default pattern.
        script_specific_regex: Optional regex pattern for script-specific tokenization.
                              If None, uses default pattern.
    """
    
    def train(self,
              filepath: str,
              outprefix: str,
              num_lines: int,
              vocab_size: int,
              recalc: int,
              word_model_file: str,
              max_bytes: int = 1000000000,
              checkpoint_iterations: int = 8192,
              verbose: bool = True,
              progress_interval: Optional[int] = None,
              save_pretokens: Optional[str] = None,
              greedy_split: bool = False,
              min_count: int = 15,
              max_ngram_len: int = 30) -> None:
        """
        Train a SuperBPE model (second pass, supermerges only).

        Args:
            filepath: Path to training data (JSONL format).
            outprefix: Output file prefix for saving models.
            num_lines: Number of documents to read from training file.
            vocab_size: Target vocabulary size for supermerges.
            recalc: How often to verify counts from scratch (for debugging).
            word_model_file: Path to first-pass word model (required).
            max_bytes: Maximum bytes to process from training file (default 1GB).
            checkpoint_iterations: Save intermediate model every N net vocab size (after deletions) (default 8192).
            verbose: If True, print extra diagnostics (per-document progress, pretokenization
                summaries, timing breakdowns). Warnings and the final summary always print.
            progress_interval: How often to print a `*` merge-progress row. None (default) prints
                every merge when verbose, else every DEFAULT_PROGRESS_INTERVAL merges. 0 disables
                progress rows entirely.
            save_pretokens: If not None, save superword pretokenization data to this file path.
            greedy_split: If True, apply n-gram greedy split before initial_counts().
            min_count: Minimum count floor for n-gram counting (default 15).
            max_ngram_len: Maximum n-gram length (default 30).
        """
        assert vocab_size >= 1, "vocab_size must be at least 1 (number of supermerges to create)"

        progress_interval = _resolve_progress_interval(progress_interval, verbose)

        # Reset all training state (allows retraining on same instance)
        self._reset_training_state()

        # SuperBPE mode: all tokens unlocked, no competition with regular merges
        superbpe_mode = True

        # Store configuration in inference_data
        self.inf_data.tau = 1.1  # No deletions for SuperBPE
        self.inf_data.is_super = True
        self.inf_data.superbpe_mode = superbpe_mode
        self.inf_data.blowup = False

        # Call SuperBPE training implementation
        self._train_internal_super(filepath, outprefix, num_lines, vocab_size, recalc,
                                   word_model_file, superbpe_mode, max_bytes,
                                   checkpoint_iterations, verbose, progress_interval, save_pretokens,
                                   greedy_split, min_count, max_ngram_len)


