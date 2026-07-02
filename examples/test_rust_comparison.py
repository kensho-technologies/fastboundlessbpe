#!/usr/bin/env python3
# Copyright 2026-present Kensho Technologies, LLC.

"""
Test script to compare Python vs Rust BoundlessBPE implementations over many documents.
Updated for v2 model format and new API (no blowup parameter).
"""

import time
import json
import sys

from boundlessbpe import Tokenizer, FastTokenizer, RUST_AVAILABLE

def load_some_data(dataset_path: str, num_lines: int = 500) -> list[str]:
    """Load test documents from JSONL file"""
    lines = []
    try:
        with open(dataset_path, "rt") as data:
            for i in range(num_lines):
                line = data.readline()
                if not line:
                    break
                doc = json.loads(line)
                lines.append(doc['text'])
        print(f"Loaded {len(lines)} documents from {dataset_path}")
    except FileNotFoundError:
        print(f"Dataset file '{dataset_path}' not found.")
        print("You can download a dataset like minipile.jsonl or use any JSONL file with 'text' field")
        sys.exit(1)

    return lines

def test_document(
    python_tokenizer: Tokenizer,
    rust_tokenizer: "FastTokenizer",
    i: int,
    text: str,
    verbose: bool = False,
) -> tuple[float, float, float, int]:
    """Test a single document with both implementations"""

    char_count = len(text)

    if verbose or i % 50 == 0:
        print(f"Test {i+1}: '{text[:50]}{'...' if len(text) > 50 else ''}'")

    # Test Python implementation
    start_time = time.time()
    python_tokens = python_tokenizer.encode_ordinary(text)
    python_time = time.time() - start_time

    # Test Rust implementation
    start_time = time.time()
    rust_tokens = rust_tokenizer.encode_ordinary(text)
    rust_time = time.time() - start_time

    # Calculate speedup
    if rust_time > 0:
        speedup = python_time / rust_time
        if verbose or i % 50 == 0:
            print(f"  Size: {char_count} chars, Python: {python_time:.4f}s, Rust: {rust_time:.4f}s, Speedup: {speedup:.2f}x")
    else:
        speedup = float('inf')

    # Compare results
    tokens_match = python_tokens == rust_tokens

    if not tokens_match:
        print(f"  ERROR: Token mismatch on document {i}!")
        print(f"    Text: '{text[:100]}...'")
        print(f"    Python tokens: {len(python_tokens)}")
        print(f"    Rust tokens:   {len(rust_tokens)}")

        # Show first few tokens for debugging
        print(f"    Python first 10: {python_tokens[:10]}")
        print(f"    Rust first 10:   {rust_tokens[:10]}")

        # Find first mismatch
        for j in range(min(len(python_tokens), len(rust_tokens))):
            if python_tokens[j] != rust_tokens[j]:
                print(f"    First mismatch at position {j}: Python={python_tokens[j]}, Rust={rust_tokens[j]}")
                break

        # Save debug files
        with open("python_debug.txt", "wt") as out:
            for t in python_tokens:
                out.write(str(t) + "\n")

        with open("rust_debug.txt", "wt") as out:
            for t in rust_tokens:
                out.write(str(t) + "\n")

        print("Saved python_debug.txt and rust_debug.txt for comparison")
        sys.exit(1)

    return python_time, rust_time, speedup, char_count

def test_implementations(model_file: str, test_texts: list[str], verbose: bool = False) -> None:
    """Test both implementations with the same inputs and compare results"""

    if not RUST_AVAILABLE:
        print("Rust implementation not available - run 'maturin develop' first")
        return

    print(f"Loading model: {model_file}")

    # Load both implementations
    python_tokenizer = Tokenizer()
    python_tokenizer.load(model_file)

    rust_tokenizer = FastTokenizer()
    rust_tokenizer.load(model_file)

    print(f"Both tokenizers loaded successfully")
    print(f"Testing with {len(test_texts)} documents")
    print("=" * 80)

    total_python_time = 0.0
    total_rust_time = 0.0
    total_chars = 0
    count = 0

    for i, text in enumerate(test_texts):
        if len(text.strip()) == 0:
            continue

        try:
            python_time, rust_time, speedup, char_count = test_document(
                python_tokenizer, rust_tokenizer, i, text, verbose
            )

            total_python_time += python_time
            total_rust_time += rust_time
            total_chars += char_count
            count += 1

        except Exception as e:
            print(f"Error on document {i}: {e}")
            if verbose:
                raise
            continue

        # Print intermediate results every 100 documents
        if (i + 1) % 100 == 0:
            speedup = total_python_time / total_rust_time if total_rust_time > 0 else 0
            python_cps = total_chars / total_python_time if total_python_time > 0 else 0
            rust_cps = total_chars / total_rust_time if total_rust_time > 0 else 0

            print(f"\n--- Progress: {i+1}/{len(test_texts)} documents ---")
            print(f"Total characters: {total_chars:,}")
            print(f"Speedup: {speedup:.2f}x")
            print(f"Python: {python_cps:,.0f} chars/sec, Rust: {rust_cps:,.0f} chars/sec")
            print("-" * 50)

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS:")
    print(f"All {count} documents passed correctness checks")

    if total_rust_time > 0:
        speedup = total_python_time / total_rust_time
        python_cps = total_chars / total_python_time if total_python_time > 0 else 0
        rust_cps = total_chars / total_rust_time if total_rust_time > 0 else 0

        print(f"Total characters: {total_chars:,}")
        print(f"Total time: Python {total_python_time:.2f}s vs Rust {total_rust_time:.2f}s")
        print(f"Throughput: Python {python_cps:,.0f} chars/sec vs Rust {rust_cps:,.0f} chars/sec")
        print(f"Overall speedup: {speedup:.2f}x")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Compare Python vs Rust BoundlessBPE implementations')
    parser.add_argument('--model-file', default="./models/boundless_1000000_1000000000_131072_0.9_1000_ultimate_1_40960.model",
                        help='Path to .model file')
    parser.add_argument('--dataset', default="./data/minipile.jsonl",
                       help='Path to JSONL dataset file')
    parser.add_argument('--num-docs', type=int, default=50000,
                       help='Number of documents to test (default: 50000)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')

    args = parser.parse_args()

    print("BoundlessBPE Python vs Rust Comparison")
    print("=" * 80)

    # Load test data
    test_texts = load_some_data(args.dataset, args.num_docs)

    try:
        test_implementations(args.model_file, test_texts, args.verbose)
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during testing: {e}")
        if args.verbose:
            raise
        sys.exit(1)
