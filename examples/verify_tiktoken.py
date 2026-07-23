# Copyright 2026-present Kensho Technologies, LLC.
"""Verify a tiktoken export matches our tokenizer over a corpus.

For a word model, exported tokenization must be byte-identical. For a superword
(boundless/superbpe) model exported via the coarse-regex trick, exact parity is
best effort: the only permitted differences are supermerge choices (word-level
tokenization must be identical and bytes never corrupted). See
boundlessbpe.export.classify_export_difference.

Usage:
    python examples/verify_tiktoken.py --model-file model.model --dataset data.jsonl \
        [--coarse-regex-default] [--num-docs 1000]
"""
import argparse
import contextlib
import io
import json
import os
import tempfile
from collections import Counter

import tiktoken
from tiktoken.load import load_tiktoken_bpe

from boundlessbpe import GPT4O_COARSE_REGEX
from boundlessbpe.inference import Tokenizer
from boundlessbpe.export import classify_export_difference, _superword_expansion_map


def load_docs(path: str, n: int) -> list[str]:
    docs: list[str] = []
    with open(path) as f:
        for line in f:
            docs.append(json.loads(line)["text"])
            if len(docs) >= n:
                break
    return docs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-file", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--num-docs", type=int, default=1000)
    ap.add_argument("--coarse-regex-default", action="store_true",
                    help="Use GPT4O_COARSE_REGEX (required for superword models trained with GPT4O_EXPORT_REGEX).")
    args = ap.parse_args()

    tok = Tokenizer()
    with contextlib.redirect_stdout(io.StringIO()):
        tok.load(args.model_file)
    coarse = GPT4O_COARSE_REGEX if args.coarse_regex_default else None

    with tempfile.TemporaryDirectory() as d:
        tpath = os.path.join(d, "model.tiktoken")
        tok.save_tiktoken(tpath, coarse_regex=coarse)
        ranks = load_tiktoken_bpe(tpath)
        meta = json.load(open(tpath + ".json"))
    enc = tiktoken.Encoding(name="export", pat_str=meta["pat_str"],
                            mergeable_ranks=ranks, special_tokens=meta["special_tokens"])

    assert tok.vocab is not None  # loaded above
    parent = _superword_expansion_map(tok)
    docs = load_docs(args.dataset, args.num_docs)
    counts: Counter = Counter()
    total_tokens = 0        # BoundlessBPE token count (denominator)
    extra_supermerges = 0   # extra merges the export applied (bytes identical => len diff)
    applied_supermerges = 0  # supermerges BoundlessBPE itself applied
    for text in docs:
        ours = [tok.vocab.id_to_token[i] for i in tok.encode_ordinary(text)]
        theirs = [enc.decode_single_token_bytes(i) for i in enc.encode_ordinary(text)]
        counts[classify_export_difference(tok, ours, theirs, parent)] += 1
        total_tokens += len(ours)
        extra_supermerges += max(0, len(ours) - len(theirs))
        applied_supermerges += sum(1 for t in ours if t in parent)

    print(f"Checked {len(docs)} docs against tiktoken export:")
    for k in ("identical", "supermerge", "byte-mismatch", "wordlevel"):
        print(f"  {k:14}: {counts.get(k, 0)}")
    if total_tokens:
        pct_tok = 100 * extra_supermerges / total_tokens
        pct_sm = 100 * extra_supermerges / applied_supermerges if applied_supermerges else 0.0
        print(f"  over-applied supermerges: {extra_supermerges} "
              f"({pct_tok:.3f}% of {total_tokens} tokens, {pct_sm:.2f}% of applied supermerges)")

    # byte-mismatch and wordlevel indicate real bugs (not the known limitation).
    ok = counts.get("byte-mismatch", 0) == 0 and counts.get("wordlevel", 0) == 0
    if tok.superwords is None and counts.get("supermerge", 0) != 0:
        ok = False  # word models must be byte-identical
    print("RESULT:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
