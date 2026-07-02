# Copyright 2026-present Kensho Technologies, LLC.
"""
N-gram counting and greedy splitting for BoundlessBPE/SuperBPE training.

Operates directly on bytes (list[list[bytes]] + list[int]) — no string conversion.
Designed to be called from train.py after pretokenize_super() to reduce the number
of unique chunks before initial_counts().
"""

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .inference import Tokenizer


def get_cmin_from_word_model(word_model: "Tokenizer") -> int:
    """Extract c_min (count of the last merge) from an already-loaded word model.

    Args:
        word_model: A Tokenizer object with word_model.words.merges loaded.

    Returns:
        int: The count of the merge with the highest index.
    """
    assert word_model.words is not None, "word_model.words must be loaded"
    merges = word_model.words.merges
    max_idx = max(merges.keys())
    # merges[idx] = ((left, right), count, unlocked_flag)
    _, c_ab, _ = merges[max_idx]
    return c_ab


def count_ngrams_bytes(
    chunks: list[list[bytes]],
    counts: list[int],
    min_cnt: int,
    max_len: int = 30,
) -> dict[tuple[bytes, ...], int]:
    """Count all n-grams (n>=2) with count >= min_cnt using Apriori pruning.

    Operates on bytes: chunks is list[list[bytes]], counts is list[int].
    N-gram keys are tuple[bytes, ...].

    Args:
        chunks: list of token lists (each token is bytes)
        counts: list of corpus counts per chunk
        min_cnt: minimum count threshold
        max_len: maximum n-gram length (default 30)

    Returns:
        dict mapping n-gram tuples (of bytes) to counts (n>=2, count >= min_cnt)
    """
    overall_start = time.time()
    print(f"count_ngrams_bytes: {len(chunks)} chunks, min_cnt={min_cnt}")

    ngram_cnt: dict[tuple[bytes, ...], int] = {}

    for sz in range(1, max_len + 1):
        start_time = time.time()
        start_size = len(ngram_cnt)

        for j, (cnt, tokens) in enumerate(zip(counts, chunks)):
            if j % 500000 == 0:
                print(f"  sz={sz} j={j} ngrams={len(ngram_cnt)} t={time.time() - start_time:.1f}s")

            n = len(tokens)
            if n < sz:
                continue

            for i in range(n - sz + 1):
                if (sz == 1) or (
                    ngram_cnt.get(tuple(tokens[i:i + sz - 1]), 0) >= min_cnt
                    and ngram_cnt.get(tuple(tokens[i + 1:i + sz]), 0) >= min_cnt
                ):
                    ngram = tuple(tokens[i:i + sz])
                    ngram_cnt[ngram] = ngram_cnt.get(ngram, 0) + cnt

        # filter below threshold (keep unigrams for next pass's pruning)
        before = len(ngram_cnt)
        ngram_cnt = {
            ng: c for ng, c in ngram_cnt.items()
            if len(ng) == 1 or c >= min_cnt
        }
        elapsed = time.time() - start_time
        print(f"  sz={sz}: {before} -> {len(ngram_cnt)} ngrams ({elapsed:.1f}s)")

        if len(ngram_cnt) == start_size:
            print("  no new ngrams found, stopping")
            break

    # final: only n>=2 with count >= min_cnt
    result = {
        ng: c for ng, c in ngram_cnt.items()
        if len(ng) >= 2 and c >= min_cnt
    }

    # Length distribution
    len_dist: dict[int, int] = {}
    len_total: dict[int, int] = {}
    for ng, c in result.items():
        n = len(ng)
        len_dist[n] = len_dist.get(n, 0) + 1
        len_total[n] = len_total.get(n, 0) + c
    max_len_found = max(len_dist) if len_dist else 0
    total_ngrams = sum(result.values())
    unique_ngrams = len(result)
    print(f"count_ngrams_bytes: {unique_ngrams} results (max_len={max_len_found}), total time: {time.time() - overall_start:.1f}s")
    if total_ngrams > 0:
        print(f"N-grams: unique={unique_ngrams}, total={total_ngrams}, ratio={unique_ngrams / total_ngrams:.6f}")
    else:
        print(f"N-grams: unique={unique_ngrams}, total={total_ngrams}")
    for n in sorted(len_dist):
        print(f"  len={n}: {len_dist[n]} unique, {len_total[n]} total")
    return result


def build_prefix_set_bytes(ngram_dict: dict[tuple[bytes, ...], int]) -> set[tuple[bytes, ...]]:
    """Build a set of all prefixes of n-grams in the dict for fast lookup.

    Args:
        ngram_dict: dict mapping tuple[bytes, ...] -> count

    Returns:
        set of tuple[bytes, ...] prefixes
    """
    prefixes = set()
    for ng in ngram_dict:
        for length in range(1, len(ng) + 1):
            prefixes.add(ng[:length])
    return prefixes


def greedy_split_bytes(
    chunks: list[list[bytes]],
    counts: list[int],
    ngram_dict: dict[tuple[bytes, ...], int],
) -> tuple[list[list[bytes]], list[int]]:
    """Greedy left-to-right partition of chunks using n-gram dictionary.

    For each chunk, finds the longest n-gram (n>=2) starting at each position.
    Single tokens that don't start any n-gram become length-1 chunks.

    Args:
        chunks: list of token lists (each token is bytes)
        counts: list of corpus counts per chunk
        ngram_dict: dict of n-gram tuples -> counts (from count_ngrams_bytes)

    Returns:
        (new_chunks, new_counts): ready to pass to initial_counts()
            new_chunks: list[list[bytes]]
            new_counts: list[int]
    """
    start_time = time.time()
    print(f"greedy_split_bytes: {len(chunks)} chunks, {len(ngram_dict)} ngrams")

    prefix_set = build_prefix_set_bytes(ngram_dict)
    result: dict[tuple[bytes, ...], int] = {}  # tuple[bytes, ...] -> aggregated count

    for j, (cnt, tokens) in enumerate(zip(counts, chunks)):
        if j % 500000 == 0:
            print(f"  j={j} result_size={len(result)} t={time.time() - start_time:.1f}s")

        pos = 0
        while pos < len(tokens):
            # Find longest n-gram (n>=2) starting at pos
            best_len = 0
            for end in range(pos + 1, len(tokens) + 1):
                candidate = tuple(tokens[pos:end])
                if candidate not in prefix_set:
                    break
                if len(candidate) >= 2 and candidate in ngram_dict:
                    best_len = len(candidate)

            if best_len >= 2:
                ngram = tuple(tokens[pos:pos + best_len])
                result[ngram] = result.get(ngram, 0) + cnt
                pos += best_len
            else:
                # Single token — emit as a length-1 chunk
                single = (tokens[pos],)
                result[single] = result.get(single, 0) + cnt
                pos += 1

    print(f"greedy_split_bytes: {len(result)} unique chunks, time: {time.time() - start_time:.1f}s")

    # Convert to list format expected by initial_counts
    new_chunks = [list(ng) for ng in result]
    new_counts = [result[ng] for ng in result]

    return new_chunks, new_counts


def ngram_split(
    text_chunks: list[list[bytes]],
    text_counts: list[int],
    word_model: "Tokenizer",
    min_count_floor: int = 15,
    max_ngram_len: int = 30,
) -> tuple[list[list[bytes]], list[int]]:
    """Top-level entry point for n-gram splitting.

    Computes c_min from the word model, runs n-gram counting and greedy splitting,
    and returns replacement (text_chunks, text_counts) for initial_counts().

    Args:
        text_chunks: list[list[bytes]] from pretokenize_super()
        text_counts: list[int] from pretokenize_super()
        word_model: loaded Tokenizer object (word_model.words.merges must exist)
        min_count_floor: minimum count floor (default 15)
        max_ngram_len: maximum n-gram length (default 30)

    Returns:
        (new_text_chunks, new_text_counts): replacement data for initial_counts()
    """
    # Extract c_min from model
    model_cmin = get_cmin_from_word_model(word_model)
    min_cnt = max(model_cmin, min_count_floor)
    print(f"ngram_split: c_min: model={model_cmin}, floor={min_count_floor}, using={min_cnt}")

    # Input stats
    unique_input = len(text_chunks)
    total_input = sum(text_counts)
    print(f"ngram_split input: unique={unique_input}, total={total_input}, ratio={unique_input / total_input:.6f}")

    # Count n-grams
    ngram_dict = count_ngrams_bytes(text_chunks, text_counts, min_cnt, max_len=max_ngram_len)

    if len(ngram_dict) == 0:
        print("ngram_split: no n-grams found, returning original data unchanged")
        return text_chunks, text_counts

    # Greedy partition
    new_chunks, new_counts = greedy_split_bytes(text_chunks, text_counts, ngram_dict)

    # Output stats (all chunks including singletons, as passed to initial_counts)
    unique_output = len(new_chunks)
    total_output = sum(new_counts)
    print(f"ngram_split output: unique={unique_output}, total={total_output}, ratio={unique_output / total_output:.6f}")

    # Stats for n>=2 chunks only (comparable to compute_ngrams.py greedy_split output)
    ngram_unique = sum(1 for c in new_chunks if len(c) >= 2)
    ngram_total = sum(cnt for c, cnt in zip(new_chunks, new_counts) if len(c) >= 2)
    singleton_unique = unique_output - ngram_unique
    singleton_total = total_output - ngram_total
    print(f"ngram_split n>=2: unique={ngram_unique}, total={ngram_total}" +
          (f", ratio={ngram_unique / ngram_total:.6f}" if ngram_total > 0 else ""))
    print(f"ngram_split singletons: unique={singleton_unique}, total={singleton_total}")
    print(f"ngram_split reduction: {unique_input} -> {unique_output} unique chunks ({unique_output/unique_input:.4f}x)")

    return new_chunks, new_counts
