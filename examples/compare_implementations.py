#!/usr/bin/env python3
# Copyright 2026-present Kensho Technologies, LLC.
"""
Comparison driver to verify Rust and Python implementations produce identical results.

This script loads both implementations, runs them on test cases, and compares:
- Tokenization results (token sequences)
- Decoding results  
- Performance (timing)
- Memory usage
"""

import time
import sys
from pathlib import Path

from boundlessbpe import Tokenizer, FastTokenizer, RUST_AVAILABLE

def load_test_cases() -> list[str]:
    """Load various test cases to verify correctness"""
    return [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "GPT-4 tokenization with special characters: <|endoftext|>",
        "Unicode test: 你好世界 🌍 café naïve résumé",
        "Code snippet: def hello(): print('world')",
        "Mixed content: Hello world! 123 @#$%^&*()",
        "Long text: " + "This is a longer piece of text to test performance. " * 100,
        "Edge cases: '', single char: 'a', numbers: 12345",
        "Contractions: don't, won't, can't, it's, they're",
        "Snake_case and camelCase and PascalCase variables",
    ]

def compare_tokenizers(model_path: str) -> bool:
    """Compare Python and Rust tokenizer implementations"""
    if not RUST_AVAILABLE:
        print("❌ Rust implementation not available - run 'maturin develop' first")
        return False
    
    print("🔄 Loading tokenizers...")
    
    # Load Python tokenizer
    py_tokenizer = Tokenizer()
    py_tokenizer.load(model_path)
    
    # Load Rust tokenizer 
    rust_tokenizer = FastTokenizer()
    rust_tokenizer.load(model_path)
    
    print("✅ Both tokenizers loaded successfully")
    
    test_cases = load_test_cases()
    all_passed = True
    
    print(f"\n🧪 Testing {len(test_cases)} test cases...")
    
    for i, text in enumerate(test_cases, 1):
        print(f"\nTest {i}: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        # Time Python implementation
        start = time.perf_counter()
        py_tokens = py_tokenizer.encode_ordinary(text)
        py_time = time.perf_counter() - start
        
        # Time Rust implementation  
        start = time.perf_counter()
        rust_tokens = rust_tokenizer.encode_ordinary(text)
        rust_time = time.perf_counter() - start
        
        # Compare results
        if py_tokens == rust_tokens:
            speedup = py_time / rust_time if rust_time > 0 else float('inf')
            print(f"  ✅ Tokens match ({len(py_tokens)} tokens)")
            print(f"  ⚡ Speedup: {speedup:.1f}x (Python: {py_time*1000:.2f}ms, Rust: {rust_time*1000:.2f}ms)")
        else:
            print(f"  ❌ Token mismatch!")
            print(f"  Python:  {py_tokens[:10]}{'...' if len(py_tokens) > 10 else ''}")
            print(f"  Rust:    {rust_tokens[:10]}{'...' if len(rust_tokens) > 10 else ''}")
            all_passed = False
        
        # Test decoding
        py_decoded = py_tokenizer.decode(py_tokens)
        rust_decoded = rust_tokenizer.decode(rust_tokens)
        
        if py_decoded == rust_decoded == text:
            print(f"  ✅ Decode matches original")
        else:
            print(f"  ❌ Decode mismatch!")
            all_passed = False
    
    print(f"\n{'🎉 All tests passed!' if all_passed else '❌ Some tests failed!'}")
    return all_passed

def benchmark_performance(model_path: str, text: str, iterations: int = 100) -> None:
    """Detailed performance comparison"""
    if not RUST_AVAILABLE:
        print("❌ Rust implementation not available")
        return
    
    print(f"\n🏃 Performance benchmark ({iterations} iterations)")
    print(f"Text length: {len(text)} characters")
    
    # Load tokenizers
    py_tokenizer = Tokenizer()
    py_tokenizer.load(model_path)
    
    rust_tokenizer = FastTokenizer()  
    rust_tokenizer.load(model_path)
    
    # Warm up
    for _ in range(5):
        py_tokenizer.encode_ordinary(text)
        rust_tokenizer.encode_ordinary(text)
    
    # Benchmark Python
    start = time.perf_counter()
    for _ in range(iterations):
        py_tokenizer.encode_ordinary(text)
    py_total = time.perf_counter() - start
    
    # Benchmark Rust
    start = time.perf_counter()
    for _ in range(iterations):
        rust_tokenizer.encode_ordinary(text)
    rust_total = time.perf_counter() - start
    
    print(f"Python: {py_total:.4f}s ({py_total/iterations*1000:.2f}ms per call)")
    print(f"Rust:   {rust_total:.4f}s ({rust_total/iterations*1000:.2f}ms per call)")
    print(f"Speedup: {py_total/rust_total:.1f}x")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python compare_implementations.py <model_file.model>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        sys.exit(1)
    
    print("🚀 BoundlessBPE Implementation Comparison")
    print("=" * 50)
    
    # Run correctness tests
    success = compare_tokenizers(model_path)
    
    if success:
        # Run performance benchmark
        long_text = "This is a performance test with a longer piece of text. " * 200
        benchmark_performance(model_path, long_text)
    
    print("\n✨ Comparison complete!")