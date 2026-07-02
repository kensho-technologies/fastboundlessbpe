#!/usr/bin/env python3
# Copyright 2026-present Kensho Technologies, LLC.

"""
Test script for BoundlessBPE inference.
Loads a model and times encode_ordinary for Python and Rust
implementations, with and without supercharge, over up to 1M documents.
Reports cumulative timing at checkpoint document counts.
Verifies all 4 results match per-document but does not charge
comparison time against the totals.
"""

import time
import json
import sys
import argparse

from boundlessbpe import Tokenizer, FastTokenizer, RUST_AVAILABLE


CHECKPOINTS = [10_000, 20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000]

# Labels for the 4 cases
LABELS = ["Py+SC", "Py-SC", "Rs+SC", "Rs-SC"]


def main() -> None:
    parser = argparse.ArgumentParser(description='Benchmark BoundlessBPE inference (Python vs Rust, with/without supercharge)')
    parser.add_argument('--model-file', required=True, help='Path to .model file')
    parser.add_argument('--dataset', default="./data/minipile.jsonl",
                        help='Path to JSONL dataset file (default: ./data/minipile.jsonl)')
    parser.add_argument('--num-lines', type=int, default=1_000_000,
                        help='Number of documents to process (default: 1000000)')
    args = parser.parse_args()

    if not RUST_AVAILABLE:
        print("Rust implementation not available - run 'maturin develop --release' first")
        sys.exit(1)

    # Load both tokenizers
    print(f"Loading model: {args.model_file}")
    py_tok = Tokenizer()
    py_tok.load(args.model_file)

    rs_tok = FastTokenizer()
    rs_tok.load(args.model_file)

    # Build checkpoint set for quick lookup
    checkpoints = set(c for c in CHECKPOINTS if c <= args.num_lines)
    checkpoints.add(args.num_lines)

    # Header
    print(f"\n! {'docs':>10s}  {'Py+SC':>10s}  {'Py-SC':>10s}  {'Rs+SC':>10s}  {'Rs-SC':>10s}  {'SC spdup':>8s}  {'noSC spdup':>10s}  {'Py SC ben':>9s}  {'Rs SC ben':>9s}")
    print(f"! {'':->10s}  {'':->10s}  {'':->10s}  {'':->10s}  {'':->10s}  {'':->8s}  {'':->10s}  {'':->9s}  {'':->9s}")

    times = [0.0, 0.0, 0.0, 0.0]  # Py+SC, Py-SC, Rs+SC, Rs-SC
    total_chars = 0
    doc_count = 0

    with open(args.dataset, "rt") as data:
        for i in range(args.num_lines):
            line = data.readline()
            if not line:
                break
            text = json.loads(line)['text']
            if len(text.strip()) == 0:
                continue

            # Time Python with supercharge
            start = time.time()
            py_sc_tokens = py_tok.encode_ordinary(text, supercharge=True)
            times[0] += time.time() - start

            # Time Python without supercharge
            start = time.time()
            py_nosc_tokens = py_tok.encode_ordinary(text, supercharge=False)
            times[1] += time.time() - start

            # Time Rust with supercharge
            start = time.time()
            rs_sc_tokens = rs_tok.encode_ordinary(text, supercharge=True)
            times[2] += time.time() - start

            # Time Rust without supercharge
            start = time.time()
            rs_nosc_tokens = rs_tok.encode_ordinary(text, supercharge=False)
            times[3] += time.time() - start

            # Verify all 4 match (not timed)
            all_results = [
                ("Py+SC", py_sc_tokens),
                ("Py-SC", py_nosc_tokens),
                ("Rs+SC", rs_sc_tokens),
                ("Rs-SC", rs_nosc_tokens),
            ]
            ref_label, ref_tokens = all_results[0]
            for label, tokens in all_results[1:]:
                if tokens != ref_tokens:
                    print(f"\n  ERROR: Token mismatch at document {i}: {ref_label} vs {label}")
                    print(f"    Text: '{text[:100]}...'")
                    print(f"    {ref_label} tokens: {len(ref_tokens)}, {label} tokens: {len(tokens)}")
                    for j in range(min(len(ref_tokens), len(tokens))):
                        if ref_tokens[j] != tokens[j]:
                            print(f"    First mismatch at position {j}: {ref_label}={ref_tokens[j]}, {label}={tokens[j]}")
                            break
                    sys.exit(1)

            total_chars += len(text)
            doc_count += 1

            if doc_count % 10_000 == 0:
                sc_speedup = times[0] / times[2] if times[2] > 0 else float('inf')
                nosc_speedup = times[1] / times[3] if times[3] > 0 else float('inf')
                py_sc_ben = times[1] / times[0] if times[0] > 0 else float('inf')
                rs_sc_ben = times[3] / times[2] if times[2] > 0 else float('inf')
                doc_str = f"{doc_count:,}"
                prefix = "!" if doc_count in checkpoints else "*"
                print(f"{prefix} {doc_str:>10s}  {times[0]:>10.2f}s  {times[1]:>10.2f}s  {times[2]:>10.2f}s  {times[3]:>10.2f}s  {sc_speedup:>7.2f}x  {nosc_speedup:>9.2f}x  {py_sc_ben:>8.2f}x  {rs_sc_ben:>8.2f}x", flush=True)

    # Final summary
    print(f"\nAll {doc_count:,} documents passed correctness check (all 4 cases identical).")
    print(f"\n! {'':>10s}  {'Py+SC':>10s}  {'Py-SC':>10s}  {'Rs+SC':>10s}  {'Rs-SC':>10s}")
    print(f"! {'time':>10s}  {times[0]:>10.2f}s  {times[1]:>10.2f}s  {times[2]:>10.2f}s  {times[3]:>10.2f}s")
    cps = [total_chars / t if t > 0 else 0 for t in times]
    print(f"! {'chars/s':>10s}  {cps[0]:>10,.0f}  {cps[1]:>10,.0f}  {cps[2]:>10,.0f}  {cps[3]:>10,.0f}")
    sc_speedup = times[0] / times[2] if times[2] > 0 else float('inf')
    nosc_speedup = times[1] / times[3] if times[3] > 0 else float('inf')
    py_sc_benefit = times[1] / times[0] if times[0] > 0 else float('inf')
    rs_sc_benefit = times[3] / times[2] if times[2] > 0 else float('inf')
    print(f"\n! Rust speedup with supercharge:    {sc_speedup:.2f}x")
    print(f"! Rust speedup without supercharge: {nosc_speedup:.2f}x")
    print(f"! Supercharge benefit (Python):     {py_sc_benefit:.2f}x")
    print(f"! Supercharge benefit (Rust):       {rs_sc_benefit:.2f}x")


if __name__ == "__main__":
    main()

# example:
# python -u test_inference.py --model-file ./models/twopass_1000000_10000000000_131072_1.1_0_fast_boundlessbpe_final.model 2>&1 | tee log_inference.txt
