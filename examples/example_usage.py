#!/usr/bin/env python
# Copyright 2026-present Kensho Technologies, LLC.
"""
Example usage of the Pretokenizer module.

This script demonstrates how to use the cleaned up tokenization functionality
for various text processing scenarios.
"""

from boundlessbpe import Pretokenizer
from boundlessbpe.pretokenize import pretokenize

def main() -> None:
    print("Pretokenizer Usage Examples")
    print("=" * 50)

    # Method 1: Use the convenience function
    print("\n1. Using convenience function:")
    text1 = "Hello 世界! How are you?"
    tokens1 = pretokenize(text1)
    print(f"Input:  '{text1}'")
    print(f"Tokens: {tokens1}")

    # Method 2: Create tokenizer instance for multiple uses
    print("\n2. Using tokenizer instance:")
    tokenizer = Pretokenizer()

    multilingual_texts = [
        "English text with 中文 mixed in",
        "สวัสดี مرحبا नमस्ते",
        "XMLHttpRequest.prototype.send",
        "we're can't won't don't",
        "🏳️‍🌈 👨‍👩‍👧‍👦 🇺🇸 emojis work too",
        "123,456.789 numbers work",
    ]

    for text in multilingual_texts:
        tokens = tokenizer.pretokenize(text)
        print(f"'{text}' -> {len(tokens)} tokens")
        print(f"  Tokens: {tokens}")

        # Verify tokenization is reversible
        reconstructed = "".join(tokens)
        if reconstructed == text:
            print("  ✓ Reversible tokenization")
        else:
            print("  ✗ Tokenization not reversible!")
        print()

    # Method 3: Compare with GPT-4o style tokenization
    print("\n3. Comparison with GPT-4o style:")
    comparison_text = "testing 123   hello  世界"

    our_tokens = tokenizer.pretokenize(comparison_text)
    gpt4o_tokens = tokenizer.pretokenize_gpt4o_style(comparison_text)

    print(f"Text: '{comparison_text}'")
    print(f"Our tokenizer:   {our_tokens}")
    print(f"GPT-4o style:    {gpt4o_tokens}")

    # Method 4: Custom script configuration
    print("\n4. Custom script configuration:")
    mixed_script_text = "English עברית العربية 中文 ひらがな カタカナ ไทย"

    # Default tokenization
    default_tokenizer = Pretokenizer()
    tokens = default_tokenizer.pretokenize(mixed_script_text)

    print(f"Mixed script text: '{mixed_script_text}'")
    print("Script-aware tokens:")
    for i, token in enumerate(tokens):
        print(f"  {i+1:2}: '{token}'")

if __name__ == "__main__":
    main()