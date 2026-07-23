# Copyright 2026-present Kensho Technologies, LLC.
"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

- this will be regular BPE with Picky BPE deletions

"""

from typing import Optional
# note: need the fancy regex here for character classes (Unicode property support)
import time
import regex

from .util import frombytes, frombytespair, merge, blow_up
from .inferencedata import InferenceData
from .vocabulary import Vocabulary
from .export import export_tiktoken, export_huggingface

class Tokenizer():

    def __init__(self) -> None:
        """
        Initialize inference engine.
        All configuration comes from loaded models.
        """
        # Unified vocabulary (loaded from file)
        self.vocab: Optional[Vocabulary] = None

        # PRIMARY DATA: InferenceData instances (merge/deletion operations)
        self.words: Optional[InferenceData] = None
        self.superwords: Optional[InferenceData] = None

        # Optimization: reachable tokens set
        # Use self.words.blowup to determine which strategy applies
        self.reachable_vocab: Optional[set[bytes]] = None
        self.possible_superwords: Optional[set[bytes]] = None



    def find_reachable_tokens(self, verbose: bool = False) -> set[bytes]:
        r"""
        Find all elements in vocabulary which end up as single tokens when tokenized by themselves.

        Say you tokenize each token by itself with the tokenizer.  Most tokens
        not involved in a deletion end up as single tokens.  We'll call these
        tokens "reachable".  Other tokens will remain as two or more tokens,
        which we'll call unreachable.  We'll precompute the set of reachable tokens up front,
        since they provide a nice speed boost.  If a pretoken is in this set of reachable tokens,
        we know the end results, and avoid splitting it into bytes and following the normal merge rules. 

        Where do unreachable tokens come from?  There a number of categories:

        1. byte tokens which are not valid utf-8 clearly are unreachable, since they can't even be strings
        2. deletions might break up tokens, which don't have a chance to fully reform
        3. conditional matches in the regex can also cause these.
        For the ultimate regex, consider this branch " ?(?:\p{L}\p{M}*)+['\u2019](?:\p{L}\p{M}*)+",
        that identifies contractions.  It will leave "can't" as a single pretoken.  As that is
        broken up, you might have the merge (b"'",b"t") -> b"'t".  However if you split
        "'t" you get two pretokens: ["'","t"], since the apostrophe is restricted to be inside of letters.
        Thus this token, "'t" is usreachable.

        As another example consider "utils'\n//" with the gtp4o regex.
        The branch " ?[^\s\p{L}\p{N}]+[\r\n/]*", will break this as ['utils', "'\n//"]
        since the ' is a non-alphanumeric, followed by and \n and two forward slashes
        (why are they theres???).  However, if you remove the context and just have
        the token "\n//", it splits it into two: ['\n', '//'].  Thus that token is also
        unreachable.

        An even more obscure one is `anie` which is subtly affected by the deletion of `iel` 
        that only partially overlaps.  The `anie` merge rule then gets used by Daniel,
        but stays as two tokens with just `anie`, so it is unreachable

                text: 22 ,{anie}{aniel}{Daniel}
                text_chunks: [[',', '{'], ['a', 'n', 'i', 'e'], ['}{'], ['aniel'], ['}{'], ['D', 'a', 'n', 'i', 'e', 'l'], ['}']]
                * merge: (b'a', b'n') 22
                text_chunks: [[',', '{'], ['an', 'i', 'e'], ['}{'], ['aniel'], ['}{'], ['D', 'an', 'i', 'e', 'l'], ['}']]
                * merge: (b'e', b'l') 48
                text_chunks: [[',', '{'], ['an', 'i', 'e'], ['}{'], ['aniel'], ['}{'], ['D', 'an', 'i', 'el'], ['}']]
                * merge: (b'i', b'e') 247
                text_chunks: [[',', '{'], ['an', 'ie'], ['}{'], ['aniel'], ['}{'], ['D', 'an', 'i', 'el'], ['}']]
                * merge: (b'i', b'el') 677
                text_chunks: [[',', '{'], ['an', 'ie'], ['}{'], ['aniel'], ['}{'], ['D', 'an', 'iel'], ['}']]
                * deletion: b'iel' 732
                text_chunks: [[',', '{'], ['an', 'ie'], ['}{'], ['aniel'], ['}{'], ['D', 'an', 'i', 'e', 'l'], ['}']]
                * merge: (b'an', b'i') 4260
                text_chunks: [[',', '{'], ['an', 'ie'], ['}{'], ['aniel'], ['}{'], ['D', 'ani', 'e', 'l'], ['}']]
                * merge: (b'ani', b'e') 8110
                text_chunks: [[',', '{'], ['an', 'ie'], ['}{'], ['aniel'], ['}{'], ['D', 'anie', 'l'], ['}']]
                * merge: (b'anie', b'l') 8892
                text_chunks: [[',', '{'], ['an', 'ie'], ['}{'], ['aniel'], ['}{'], ['D', 'aniel'], ['}']]
                tokens: [',', '{', 'an', 'ie', '}{', 'aniel', '}{', 'D', 'aniel', '}']

        Our general approach here is simple.  We simply tokenize each word in the vocabulary,
        and add it to the reachable token set if it ends up as a single token.         
                        
        Returns:
            A subset of items from vocab that are reachable
        """
        start_time = time.time()
        if verbose:
            print("unreachable token analysis:")

        badutf8_cnt = 0
        contraction_cnt = 0
        assert self.words is not None, "words must be loaded before finding reachable tokens"
        assert self.vocab is not None, "vocab must be loaded before finding reachable tokens"

        deletion_cnt = 0
        returnforwardslashes = 0
        anie_cnt = 0
        other_cnt = 0

        # Work with only word-level tokens (not super-level)
        word_tokens = self.vocab.get_word_tokens()

        # there are cases that need it both ways with blowup
        # might not need both with regular deletions
        # ie.  'Something', ' Europe Euro Eur', or 'turn{tur}'

        # lets see if there are tokens in vocab that don't tokenize to a single token
        reachable_vocab = set()
        for v in word_tokens:
            try: 
                # TODO: error handle?
                text = v.decode("utf-8", errors="strict")  # only consider those that are valid utf-8 strings   

                # can't call supercharge yet, since we haven't set it up yet
                # this is computing the data for that
                tokens = self.encode_ordinary_chunks(text, supercharge=False)
                if len(tokens) > 1:

                    # it is fine if it contains a quote or a curly quote
                    # see comments for explanation
                    if "'" in text or "\u2019" in text:
                        contraction_cnt += 1
                        continue

                    # see comments for explanation
                    if v == b'\n//':
                        returnforwardslashes += 1 
                        continue

                    # or is involved with a deletion
                    if any(d in v for d in self.words.deletions.values()) or any(v in d for d in self.words.deletions.values()):
                        deletion_cnt += 1
                        continue

                    if v == b'anie':
                        anie_cnt += 1
                        continue

                    if verbose:
                        print("   unreachable:", frombytes(v), len(text), [frombytes(s.encode("utf-8")) for s in self.words.pretokenizer.pretokenize(text)])
                        # lets look up the merge rule that made it
                        for p, idx in self.words.merges_lookup.items():
                            if p[0] + p[1] == v:
                                print("   merge rule:", p, idx)
                                print()
                                break
                    other_cnt += 1

                else:
                    reachable_vocab.add(v)

            except UnicodeDecodeError:
                badutf8_cnt += 1

        if verbose:
            print("  time:", time.time() - start_time)
            print("  blowup:", self.words.blowup)
            print("  word vocab size:", len(word_tokens))
            print("  bad utf8:", badutf8_cnt)
            print("  contractions:", contraction_cnt)
            print("  deletions:", deletion_cnt)
            print("  \\n//:", returnforwardslashes)
            print("  anie:", anie_cnt)
            print("  other:", other_cnt)
            print("  reachable:", len(reachable_vocab))
            print()

        return reachable_vocab

    def load(self, model_file: str) -> None:
        """
        Load BoundlessBPE model from file. Auto-detects format.

        The `.model` file is a unified v2 text format. The full layout (this method
        reads the header and delegates each section to Vocabulary.load /
        InferenceData.load)::

            BoundlessBPE v2 <model_type>     # word | boundless | superbpe
            vocabulary
            <count>
            <idx> <token> <count> <is_super>
            ...
            special_tokens
            <count>
            <idx> <token_string>
            ...
            words
            <JSON config>                    # tau, is_super, regex patterns, etc.
            merges
            <count>
            <idx> <left> <right> <count> <unlocked_flag>
            ...
            deletions
            <count>
            <idx> <token>
            ...
            superwords                       # only for boundless / superbpe models
            <JSON config>
            merges
            ...
            deletions
            ...

        Tokens are written in the byte-to-unicode encoding of util.frombytes.

        Args:
            model_file: Path to .model file (single-pass or two-pass unified format)
        """
        with open(model_file, 'rt') as f:
            # Read header
            header = f.readline().strip()
            parts = header.split()
            assert len(parts) == 3, f"Invalid header: {header}"
            assert parts[0] == "BoundlessBPE", f"Invalid format: {parts[0]}"
            assert parts[1] == "v2", f"Unsupported version: {parts[1]}"
            model_type = parts[2]
            assert model_type in ("word", "boundless", "superbpe"), f"Unknown model type: {model_type}"

            # Load vocabulary section
            self.vocab = Vocabulary.load(f)

            print(f"Loaded vocabulary from {model_file}")
            print(f"  - Tokens: {len(self.vocab)}")
            if self.vocab.special_tokens:
                print(f"  - Special tokens: {len(self.vocab.special_tokens)}")
            print(f"  - Total size: {self.vocab.total_size()}")

            # Load words section
            words_marker = f.readline().strip()
            assert words_marker == "words", f"Expected 'words', got '{words_marker}'"
            self.words = InferenceData.load(f)

            print(f"Loaded words section")
            print(f"  - Merges: {len(self.words.merges)}")
            print(f"  - Deletions: {len(self.words.deletions)}")
            print(f"  - Blowup mode: {self.words.blowup}")

            # Load superwords section if present (two-pass model)
            if model_type in ["boundless", "superbpe"]:
                super_marker = f.readline().strip()
                assert super_marker == "superwords", f"Expected 'superwords', got '{super_marker}'"
                self.superwords = InferenceData.load(f)

                print(f"Loaded superwords section")
                print(f"  - Merges: {len(self.superwords.merges)}")

                print(f"Two-pass mode: {len(self.words.merges)} word merges, {len(self.superwords.merges)} supermerges")

        # Set up possible superwords (for supermerge optimization)
        self._setup_superwords()

        # Compute reachable tokens (optimization)
        print(f"Computing reachable tokens...")
        self.reachable_vocab = self.find_reachable_tokens()

    def _setup_superwords(self) -> None:
        """
        Set up possible superwords set.
        """
        assert self.words is not None, "words must be loaded"
        assert self.vocab is not None, "vocab must be loaded"

        # Build possible_superwords set for optimization (if superwords available)
        self.possible_superwords = set()

        if self.superwords:
            # Use supermerge merge pairs to determine possible superword tokens
            for (s1, s2) in self.superwords.merges_lookup.keys():
                self.possible_superwords.add(s1)
                self.possible_superwords.add(s2)
            print(f"Built possible_superwords set: {len(self.possible_superwords)} tokens")
        else:
            # in this case we're going to potentially pretokenize into runs of superwords
            for word in self.vocab.tokens:
                if self.words.pretokenizer.could_merge(word):
                    self.possible_superwords.add(word)

    #################################################


    def fast_merge_delete(self, text_chunks: list[list[bytes]], supercharge: bool = True, verbose: bool = False) -> None:
        """
        Apply all regular BPE merges and PickyBPE deletions to each chunk independently.
        This replaces the global while loop that uses overall_min with a chunk-by-chunk approach.

        Args:
            text_chunks: List of token chunks to process (modified in place)
            supercharge: Whether to use reachable token optimization (True) or process all chunks (False)
            verbose: Whether to print debug information
        """
        assert self.words is not None, "words must be loaded"

        for chunk_idx, tokens in enumerate(text_chunks):
            if verbose:
                print(f"Processing chunk {chunk_idx}: {[frombytes(t) for t in tokens]}")
            
            # Check if this chunk represents a single reachable token (98% of cases)
            if supercharge and len(tokens) > 1 and self.reachable_vocab is not None:  # Only check if supercharge enabled and chunk was split into bytes
                original_token = b''.join(tokens)  # Reconstruct original pretoken
                if original_token in self.reachable_vocab:
                    # Replace with single token and skip all processing
                    tokens[:] = [original_token]
                    if verbose:
                        print(f"  Chunk {chunk_idx} is reachable, skipping processing")
                    continue
            
            # Process this chunk until no more merges or deletions possible
            prev_ind = -1
            
            while True:
                # Find the best merge or deletion within this chunk only
                min_merge_pair = None
                min_merge_ind = 10**9
                min_deletion_tok = None
                min_deletion_ind = 10**9
                
                # Look for merges within this chunk
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    if pair in self.words.merges_lookup:
                        # Find the minimum index > prev_ind from the list of indices for this pair
                        valid_indices = [idx for idx in self.words.merges_lookup[pair]
                                        if idx > prev_ind]
                        if valid_indices:
                            merge_idx = min(valid_indices)
                            if merge_idx < min_merge_ind:
                                min_merge_ind = merge_idx
                                min_merge_pair = pair

                # Look for deletions within this chunk
                for tok in tokens:
                    if tok in self.words.deletions_lookup:
                        # Find the minimum index > prev_ind from the list of indices for this token
                        valid_indices = [idx for idx in self.words.deletions_lookup[tok] if idx > prev_ind]
                        if valid_indices:
                            deletion_idx = min(valid_indices)
                            if deletion_idx < min_deletion_ind:
                                min_deletion_ind = deletion_idx
                                min_deletion_tok = tok
                
                # Choose the operation with the lowest index
                if min_merge_ind < min_deletion_ind:
                    # Apply merge
                    if min_merge_pair is None:
                        break  # No more operations possible
                    
                    if verbose:
                        print(f"  merge: {min_merge_pair} (index {min_merge_ind})")
                    
                    newtokens, _ = merge(tokens, min_merge_pair)
                    tokens[:] = newtokens
                    prev_ind = min_merge_ind
                    
                elif min_deletion_ind < 10**9:
                    # Apply deletion
                    assert min_deletion_tok is not None
                    if verbose:
                        print(f"  deletion: {min_deletion_tok!r} (index {min_deletion_ind})")

                    parts = self.words.get_replacement_parts(min_deletion_tok)
                    newtokens, _ = blow_up(tokens, min_deletion_tok, parts)
                    tokens[:] = newtokens
                    prev_ind = min_deletion_ind
                    
                else:
                    # No more operations possible
                    break

    def supermerge_tokens(self, tokens: list[bytes]) -> list[bytes]:
        """
        Take a list of tokens and apply all possible supermerges in the correct order.
        This combines the logic of min_supermerge and supermerge but operates on a simple
        list of tokens rather than the complex text_chunks structure.

        Note: This function does NOT use prev_ind tracking like fast_merge_delete because
        supermerges do not have interleaved deletions. Each supermerge is simply applied
        in index order. If deletions were introduced for supermerges, this function would
        need to be updated to track prev_ind similar to fast_merge_delete.

        Args:
            tokens: List of bytes tokens to process

        Returns:
            List of tokens after applying all possible supermerges
        """
        assert self.superwords is not None, "superwords must be provided for supermerging"
        # Supermerges should never have deletions - this algorithm assumes no interleaved deletions
        assert len(self.superwords.deletions) == 0, "supermerge_tokens does not support deletions"
        # Work on a copy to avoid modifying the input
        result = tokens[:]
        
        while True:
            # Find the minimum supermerge pair in this token list
            min_pair = None
            min_ind = 10**9
            
            # Look for adjacent pairs that can be supermerged
            for i in range(len(result) - 1):
                pair = (result[i], result[i + 1])
                if pair in self.superwords.merges_lookup:
                    # Find the minimum index from the list of indices for this pair
                    # (For supermerges we don't use prev_ind limit, just find the smallest)
                    indices = self.superwords.merges_lookup[pair]
                    if indices:
                        merge_idx = min(indices)
                        if merge_idx < min_ind:
                            min_ind = merge_idx
                            min_pair = pair
            
            # If no more merges possible, we're done
            if min_pair is None:
                break
                
            # Apply the merge: build new list with merged tokens
            merged_token = min_pair[0] + min_pair[1]
            new_result = []
            i = 0
            while i < len(result):
                if i < len(result) - 1 and result[i] == min_pair[0] and result[i + 1] == min_pair[1]:
                    # Found the pair to merge
                    new_result.append(merged_token)
                    i += 2  # Skip both tokens of the pair
                else:
                    # Keep the token as-is
                    new_result.append(result[i])
                    i += 1
            result = new_result
        
        return result

    # do it all at once using FastBPE
    def fast_supermerge(self, text_chunks: list[list[bytes]], export_compatible: bool = False) -> list[bytes]:

        # If no superwords model, just flatten and return
        if not self.superwords:
            tokens = []
            for chunk in text_chunks:
                tokens.extend(chunk)
            return tokens

        # A chunk can start/continue a supermerge run in one of two ways:
        # - normal: it reduced to a single token that is a known supermerge participant.
        # - export_compatible: its original pretoken is word-like (could_merge), even if
        #   it reduced to several tokens. This reproduces what a plain single-pass BPE
        #   does under the coarse "trick" regex — i.e. what save_tiktoken /
        #   save_huggingface produce — and may apply supermerges normal inference would
        #   not (see README "Exporting"). No coarse regex is needed: could_merge on the
        #   model's own pretokens defines the same word runs.
        assert self.possible_superwords is not None, "possible_superwords must be initialized"
        assert self.words is not None, "words must be loaded"
        possible_superwords = self.possible_superwords

        # Precompute per-chunk run eligibility once, branching on the mode a single
        # time rather than per chunk (this stays on the inference hot path):
        # - normal: chunk reduced to a single known supermerge participant.
        # - export_compatible: original pretoken is word-like (could_merge), even if it
        #   reduced to several tokens — reproduces the exported tiktoken/HF tokenizer.
        if export_compatible:
            could_merge = self.words.pretokenizer.could_merge
            eligible = [could_merge(b"".join(chunk)) for chunk in text_chunks]
        else:
            eligible = [len(chunk) == 1 and chunk[0] in possible_superwords
                        for chunk in text_chunks]

        tokens = []
        i = 0
        n = len(text_chunks)
        while i < n:
            if not eligible[i]:
                tokens.extend(text_chunks[i])
                i += 1
                continue

            # find the end of this maximal run of supermerge-eligible chunks
            j = i + 1
            while j < n and eligible[j]:
                j += 1

            # a lone eligible chunk can't supermerge (rules are whole-word pairs)
            if j == i + 1:
                tokens.extend(text_chunks[i])
                i += 1
                continue

            # flatten the run's tokens and apply supermerges to the flat list
            flat: list[bytes] = []
            for k in range(i, j):
                flat.extend(text_chunks[k])
            tokens.extend(self.supermerge_tokens(flat))
            i = j
        return tokens
    
    # find valid runs of supermerges, without doing the supermerges
    # this is used to aggregate chunks
    # add these to the counts dict as a side effect
    def fast_supermerge_runs(self, text_chunks: list[list[bytes]], counts: dict[tuple[bytes, ...], int]) -> None:

        # we'll be calling this went training the superwords
        # so they shouldn't exist yet
        assert not self.superwords
        assert self.possible_superwords is not None

        i = 0
        while i < len(text_chunks):
            
            # print("tc:", text_chunks[i])

            # skip if not a single token or i not in possible superwords
            if len(text_chunks[i]) > 1 or \
                text_chunks[i][0] not in self.possible_superwords:
                i += 1 
                continue

            # find the end of this run of the single chunks
            j = i + 1  # beyond the range
            while j < len(text_chunks) and len(text_chunks[j]) == 1\
                and text_chunks[j][0] in self.possible_superwords:
                j += 1

            # if at least two then try a supermerge
            if j == i + 1:
                i += 1 
                continue

            # this is a list of single pretokens
            # TODO: remove this for speed
            for k in range(i,j):
                # print("i,j,k", i, j, k, text_chunks[k], text_chunks[k][0] in self.possible_superwords)
                assert len(text_chunks[k]) == 1
                assert text_chunks[k][0] in self.possible_superwords

            # note j is inclusive here and must be in range!!!!
            assert len(text_chunks[j-1]) == 1

            # map these to the list of ids that merge_ids expexts
            # merge_ids(self, ids: List[int])
            # work with flattened list here, converting from List[List[bytes]]
            # to list[bytes]
            # make a tuple so we can use as a dict key 
            superword_run = tuple([tc[0] for tc in text_chunks[i:j]])
            # print("before_tokens:", before_tokens)
            counts[superword_run] = counts.get(superword_run, 0) + 1

            # move to be j
            i = j
    
    def decode_bytes(self, ids: list[int]) -> bytes:
        """Given ids (list of integers), return raw bytes. Lossless."""
        assert self.vocab is not None, "model must be loaded"
        part_bytes = []
        for idx in ids:
            if idx in self.vocab.id_to_token:
                part_bytes.append(self.vocab.id_to_token[idx])
            elif idx in self.vocab.inverse_special_tokens:
                part_bytes.append(self.vocab.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        return b"".join(part_bytes)

    def decode(self, ids: list[int]) -> str:
        """Given ids (list of integers), return Python string.
        Uses lossy UTF-8 decoding (replaces invalid bytes with U+FFFD).
        Use decode_bytes() for lossless output."""
        return self.decode_bytes(ids).decode("utf-8", errors="replace")
    

    # take a document and process it as a list of chunks
    # returns a list of tokens, rather than the ids
    # since that can be useful sometimes
    # if blowup is True, then a deleted token is blown up into single bytes
    # as in our paper,  if False, then split into the pair that created it
    # as in the original paper.  This was used in the ablations
    def encode_ordinary_chunks(self, text : str, supercharge : bool = True, verbose : bool = False, export_compatible : bool = False) -> list[bytes]:
        """Encode text to a list of token byte strings, ignoring special tokens.

        export_compatible (default False): a comparison/testing aid, not for normal use.
        When True the supermerge step groups pretokens the way the coarse "trick" regex
        does and supermerges the flattened run, so it reproduces the *supermerges* the
        exported tiktoken / HuggingFace tokenizers make (see save_tiktoken /
        save_huggingface) — including some normal BoundlessBPE would not, e.g. a
        supermerge built from a pretoken fragment. This makes it much closer to the
        exports than normal inference, but it is NOT byte-identical to them: regular
        (non-super) merges here still stay within pretoken boundaries, whereas the
        exports (plain single-pass BPE over one fused pretoken) can additionally make a
        regular merge across a boundary. Use it to compare supermerge behavior against an
        export; leave it False for real encoding.
        """
        assert self.words is not None, "words must be loaded"

        if verbose:
            print("text:", len(text), text )

        # get the pre-tokenized chunks
        text_chunks_str = self.words.pretokenizer.pretokenize(text)

        # convert string to bytes, and split into single bytes to get started
        # (reachable token optimization is now handled in fast_merge_delete)
        text_chunks: list[list[bytes]] = [[bytes([b]) for b in ch.encode("utf-8")] for ch in text_chunks_str]

        # Apply all regular merges and deletions using the fast chunk-by-chunk approach
        self.fast_merge_delete(text_chunks, supercharge, verbose)

        # do all the supermerges (or just flatten if no self.superwords);
        # export_compatible reproduces the exported tiktoken/HuggingFace tokenizer
        tokens = self.fast_supermerge(text_chunks, export_compatible=export_compatible)

        if verbose:
            print("tokens:", [frombytes(t) for t in tokens])

        return tokens

    # finally do the integer lookup
    def encode_ordinary(self, text: str, supercharge: bool = True, export_compatible: bool = False) -> list[int]:
        """Encode text to token ids, ignoring special tokens.

        export_compatible (default False) is a testing aid that mimics the exported
        tiktoken/HuggingFace tokenizers and will make some merges real BoundlessBPE would
        not — see encode_ordinary_chunks. Leave it False for normal encoding.
        """
        assert self.vocab is not None, "vocab must be loaded"
        return [self.vocab.token_to_id[tok]
                for tok in self.encode_ordinary_chunks(text, supercharge=supercharge, export_compatible=export_compatible)]

    def encode(self, text: str, allowed_special: str = "none_raise", export_compatible: bool = False) -> list[int]:
        """
        Unlike encode_ordinary, this function handles special tokens.
        allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
        if none_raise, then an error is raised if any special token is encountered in text
        this is the default tiktoken behavior right now as well
        any other behavior is either annoying, or a major footgun
        export_compatible (default False) is a testing aid that more closely mimics the exported
        tiktoken/HuggingFace tokenizers and will make some merges real BoundlessBPE would
        not — see encode_ordinary_chunks. Leave it False for normal encoding.
        """
        # decode the user desire w.r.t. handling of special tokens
        assert self.vocab is not None, "vocab must be loaded"
        special = None
        if allowed_special == "all":
            special = self.vocab.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.vocab.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.vocab.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        if not special:
            # shortcut: if no special tokens, just use the ordinary encoding
            return self.encode_ordinary(text, export_compatible=export_compatible)
        # otherwise, we have to be careful with potential special tokens in text
        # we handle special tokens by splitting the text
        # based on the occurrence of any exact match with any of the special tokens
        # we can use regex.split for this. note that surrounding the pattern with ()
        # makes it into a capturing group, so the special tokens will be included
        special_pattern = "(" + "|".join(regex.escape(k) for k in special) + ")"
        special_chunks = regex.split(special_pattern, text)
        # this can have empty strings, if a special_pattern is at the start
        special_chunks = [ch for ch in special_chunks if len(ch) > 0]

        # now all the special characters are separated from the rest of the text
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for part in special_chunks:
            if part in special:
                # this is a special token, encode it separately as a special case
                ids.append(special[part])
            else:
                # this is an ordinary sequence, encode it normally
                ids.extend(self.encode_ordinary(part, export_compatible=export_compatible))
        return ids

    def encode_batch(self, texts: list[str], allowed_special: str = "none_raise", export_compatible: bool = False) -> list[list[int]]:
        """
        Encode multiple texts at once (HuggingFace-compatible).

        Args:
            texts: List of texts to encode
            allowed_special: How to handle special tokens ("all"|"none"|"none_raise" or set)
            export_compatible: If True, more closely match the exported tiktoken/HuggingFace tokenizer
                (see encode_ordinary).

        Returns:
            List of token ID lists, one for each input text
        """
        return [self.encode(text, allowed_special, export_compatible=export_compatible) for text in texts]

    def decode_batch(self, ids_list: list[list[int]]) -> list[str]:
        """
        Decode multiple token ID sequences at once (HuggingFace-compatible).

        Args:
            ids_list: List of token ID lists to decode

        Returns:
            List of decoded strings
        """
        return [self.decode(ids) for ids in ids_list]

    def get_vocab(self, with_added_tokens: bool = True) -> dict[bytes, int]:
        """
        Get vocabulary as bytes -> int mapping.

        Note: Unlike HuggingFace tokenizers which returns str -> int,
        this returns bytes -> int since BoundlessBPE works with bytes internally.

        Args:
            with_added_tokens: Whether to include special tokens in the result

        Returns:
            Dictionary mapping token bytes to their IDs
        """
        assert self.vocab is not None, "model must be loaded"

        if with_added_tokens and self.vocab.special_tokens:
            # Combine vocab with special tokens (encoding special tokens to bytes)
            result = dict(self.vocab.token_to_id)
            for tok_str, idx in self.vocab.special_tokens.items():
                result[tok_str.encode('utf-8')] = idx
            return result

        return dict(self.vocab.token_to_id)

    def get_vocab_size(self, with_added_tokens: bool = True) -> int:
        """
        Get vocabulary size (HuggingFace-compatible).

        Args:
            with_added_tokens: Whether to include special tokens in the count

        Returns:
            Size of vocabulary (optionally including special tokens)
        """
        assert self.vocab is not None, "model must be loaded"

        if with_added_tokens:
            return self.vocab.total_size()
        else:
            return self.vocab.vocab_size()

    def token_to_id(self, token: str) -> int | None:
        """
        Convert token string to ID (HuggingFace-compatible).

        Args:
            token: Token string to look up

        Returns:
            Token ID, or None if token not in vocabulary
        """
        assert self.vocab is not None, "model must be loaded"
        token_bytes = token.encode('utf-8')
        return self.vocab.token_to_id.get(token_bytes)

    def id_to_token(self, token_id: int) -> str | None:
        """
        Convert ID to token string (HuggingFace-compatible).

        Args:
            token_id: Token ID to look up

        Returns:
            Token string, or None if ID not in vocabulary
        """
        assert self.vocab is not None, "model must be loaded"
        token_bytes = self.vocab.id_to_token.get(token_id)
        if token_bytes is not None:
            return token_bytes.decode('utf-8', errors='replace')
        return None

    @classmethod
    def from_file(cls, path: str) -> 'Tokenizer':
        """
        Load tokenizer from file (HuggingFace-compatible).

        Args:
            path: Path to model file (auto-detects format)

        Returns:
            Loaded Tokenizer instance
        """
        instance = cls()
        instance.load(path)
        return instance

    def add_special_tokens(self, special_tokens: list[str]) -> int:
        """
        Add special tokens to the vocabulary (HuggingFace-compatible).

        Special tokens are indexed after regular vocabulary tokens.
        Can be called multiple times - new tokens are accumulated.

        Args:
            special_tokens: List of special token strings to add

        Returns:
            Number of tokens actually added (excludes duplicates)
        """
        assert self.vocab is not None, "vocab must be loaded before adding special tokens"

        # Count how many we actually add (excluding duplicates)
        added_count = 0

        for tok in special_tokens:
            if tok not in self.vocab.special_tokens:
                # Use vocab size + existing special tokens for correct indexing
                next_idx = len(self.vocab) + len(self.vocab.special_tokens)
                self.vocab.special_tokens[tok] = next_idx
                self.vocab.inverse_special_tokens[next_idx] = tok
                added_count += 1

        return added_count

    def save(self, file_prefix: str) -> None:
        """
        Save the tokenizer model to file.

        This allows loading a model, modifying it (e.g., adding special tokens),
        and saving it back.

        Args:
            file_prefix: Path prefix for output file (will add .model extension)
        """
        assert self.vocab is not None, "vocab must be loaded before saving"
        assert self.words is not None, "words must be loaded before saving"

        # Determine model type based on whether superwords exist
        if self.superwords is not None:
            # Two-pass model - check if it's superbpe or boundless based on superbpe_mode
            model_type = "superbpe" if self.superwords.superbpe_mode else "boundless"
        else:
            model_type = "word"

        model_file = file_prefix + ".model"

        print(f"Saving model: {model_file}")
        print(f"  - Vocabulary: {len(self.vocab)} tokens")
        if self.vocab.special_tokens:
            print(f"  - Special tokens: {len(self.vocab.special_tokens)}")
        print(f"  - Total size: {self.vocab.total_size()}")

        with open(model_file, 'wt') as f:
            # Write version header
            f.write(f"BoundlessBPE v2 {model_type}\n")

            # Write vocabulary section (use final_token_counts from loaded model)
            self.vocab.save(f, self.vocab.final_token_counts)

            # Write words section
            f.write("words\n")
            self.words._write_to_file(f)

            # Write superwords section if present
            if self.superwords is not None:
                f.write("superwords\n")
                self.superwords._write_to_file(f)

    def save_tiktoken(self, path: str, coarse_regex: Optional[str] = None) -> None:
        """
        Export this model to tiktoken format (a .tiktoken rank file + sidecar JSON).

        Requires a model with no PickyBPE deletions and no script-aware pretokenizer.
        For a superword (boundless/superbpe) model, `coarse_regex` is required — the
        coarse "trick" regex that keeps mergeable runs as a single pretoken. For a
        plain word model, `coarse_regex` must not be passed. See boundlessbpe.export.

        Args:
            path: Output path for the .tiktoken file (a `<path>.json` sidecar with
                the pattern and special tokens is written alongside it).
            coarse_regex: Coarse pretokenization regex (superword models only).
        """
        export_tiktoken(self, path, coarse_regex)

    def save_huggingface(self, path: str, coarse_regex: Optional[str] = None) -> None:
        """
        Export this model to HuggingFace format into directory `path`.

        Writes three files so the export loads both via
        `tokenizers.Tokenizer.from_file(path/"tokenizer.json")` and via
        `transformers.AutoTokenizer.from_pretrained(path)`:
          - tokenizer.json         (the tokenizer, built with the `tokenizers` library)
          - tokenizer_config.json  (tokenizer_class, special tokens, model_max_length)
          - special_tokens_map.json
        The config files are written by hand, so this does not require `transformers`.

        Requires a model with no PickyBPE deletions and no script-aware pretokenizer.
        For a superword (boundless/superbpe) model, `coarse_regex` is required; for a
        plain word model it must not be passed. See boundlessbpe.export.

        Args:
            path: Output directory for the HuggingFace tokenizer files.
            coarse_regex: Coarse pretokenization regex (superword models only).
        """
        export_huggingface(self, path, coarse_regex)
