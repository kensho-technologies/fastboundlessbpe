# Copyright 2026-present Kensho Technologies, LLC.
"""
Count all byte-token n-grams up to max_len tokens
using the aggregated pretoken counts from pretokenize_super,
pruning below min_cnt.
"""

import time
from typing import Optional

from .util import frombytes


def ngrams(
    text_chunks: list[list[bytes]],
    text_counts: list[int],
    max_len: int,
    min_cnt: int,
    outfile: Optional[str] = None,
) -> dict[tuple[bytes, ...], int]:
    """Count all token n-grams up to max_len over the supermerge candidates.

    Args:
        text_chunks: list of token sequences (from pretokenize_super)
        text_counts: corpus count for each chunk
        max_len: maximum n-gram length (in tokens)
        min_cnt: minimum count threshold for keeping an n-gram
        outfile: optional path to write results as TSV

    Returns:
        dict mapping token n-gram tuples to their counts (filtered by min_cnt)
    """
    overall_start_time = time.time()

    print(f"ngrams: {len(text_chunks)} chunks, max_len={max_len}, min_cnt={min_cnt}")

    # do each size separately, but keep old counts around for bounding
    ngram_cnt: dict[tuple[bytes, ...], int] = {}

    # up to and including max_len so +1
    for sz in range(1, max_len + 1):
        start_time = time.time()
        start_size = len(ngram_cnt)
        for j, (cnt, tokens) in enumerate(zip(text_counts, text_chunks)):

            if j % 500000 == 0:
                print(sz, j, len(ngram_cnt), round(time.time() - start_time, 2))

            # skip if chunk is too small
            if len(tokens) < sz:
                continue

            for i in range(len(tokens) - sz + 1):
                # keep all values of size 1
                # can skip if either the prefix or suffix (n-1)-gram is below
                # min_cnt, since the full n-gram can't exceed either count
                if (sz == 1) or (
                    ngram_cnt.get(tuple(tokens[i:i + sz - 1]), 0) >= min_cnt
                    and ngram_cnt.get(tuple(tokens[i + 1:i + sz]), 0) >= min_cnt
                ):
                    ngram = tuple(tokens[i:i + sz])
                    ngram_cnt[ngram] = ngram_cnt.get(ngram, 0) + cnt

        # filter out entries below the limit (always keep size 1)
        before_size = len(ngram_cnt)
        ngram_cnt = {
            ngram: cnt for ngram, cnt in ngram_cnt.items()
            if len(ngram) == 1 or cnt >= min_cnt
        }

        print(sz, before_size, len(ngram_cnt), time.time() - start_time)

        # stop if no new n-grams were found at this size
        if len(ngram_cnt) == start_size:
            print("nothing new found, breaking")
            break

    # final filter: drop unigrams (only needed for subgram bound) and below threshold
    result = {
        ngram: cnt for ngram, cnt in ngram_cnt.items()
        if len(ngram) >= 2 and cnt >= min_cnt
    }

    if outfile is not None:
        ngram_sorted = sorted(result.items(), key=lambda x: -x[1])
        with open(outfile, "wt") as out:
            for ngram, cnt in ngram_sorted:
                token_str = " ".join(frombytes(tok) for tok in ngram)
                out.write(f"{cnt}\t{token_str}\n")
        print(f"wrote {len(ngram_sorted)} n-grams to {outfile}")

    print(f"ngrams: {len(result)} results, overall_time: {time.time() - overall_start_time:.1f}s")

    return result
