# Pretokenizer

A pretokenization module for multilingual text processing which is an enhanced version of the GPT4o regex.

## Overview

This module provides utilities for pretokenizing text while respecting grapheme boundaries and handling special cases for non-space-delimited Asian scripts. It was extracted and cleaned up from exploratory research code in `Graphemes.ipynb`.

## Features

- **Script-aware pretokenization**: Handles non-space-delimited Asian scripts (CJK, Thai, Myanmar, Khmer, Lao) character by character
- **Word-level processing**: Processes other scripts with intelligent word-level pretokenization
- **Grapheme boundaries**: Respects complex grapheme clusters (emojis, combining marks, etc.)
- **Comprehensive patterns**: Supports contractions, camelCase, numbers, punctuation, and whitespace
- **Reversible**: All pretokenization is fully reversible (joining pretokens recreates original text)

## Usage

### Quick Start

```python
from pretokenize import pretokenize

# Simple pretokenization
text = "Hello 世界! How are you?"
pretokens = pretokenize(text)
print(pretokens)
# ['Hello', ' 世', '界', '!', ' How', ' are', ' you', '?']
```

### Advanced Usage

```python
from pretokenize import Pretokenizer

# Create pretokenizer instance for multiple uses (default settings)
pretokenizer = Pretokenizer()

# Multilingual text
text = "English text with 中文 mixed in"
pretokens = pretokenizer.pretokenize(text)
print(pretokens)
# ['English', ' text', ' with', ' 中', '文', ' mixed', ' in']

# Compare with GPT-4o style pretokenization
gpt4o_pretokens = pretokenizer.pretokenize_gpt4o_style(text)
```

### Customizing Script Handling

You can customize which scripts are pretokenized character-by-character:

```python
from pretokenize import Pretokenizer

# Default: Han, Hiragana, Katakana, Thai, Myanmar, Khmer, Lao are character-by-character
pretokenizer = Pretokenizer()

# Custom: Add low-resource non-space-delimited scripts
pretokenizer = Pretokenizer(script_specific_scripts=['Han', 'Hiragana', 'Katakana', 'Thai',
                                                      'Myanmar', 'Khmer', 'Lao', 'Javanese',
                                                      'Balinese', 'Batak', 'Yi', 'Bopomofo',
                                                      'Tagbanwa', 'Tai_Tham'])

# Only treat Han (Chinese/Kanji) character-by-character
pretokenizer = Pretokenizer(script_specific_scripts=['Han'])

# Treat all scripts at word-level (no character-by-character splitting)
pretokenizer = Pretokenizer(script_specific_scripts=[])

# Example with custom configuration for low-resource scripts
text = "ꦲꦏ꧀ꦱꦫꦗꦮ"  # Javanese script
default_pretokens = Pretokenizer().pretokenize(text)  # word-level
javanese_char_pretokens = Pretokenizer(script_specific_scripts=['Javanese']).pretokenize(text)  # char-level
```

## Key Components

### Pretokenizer Class

Main pretokenizer class with the following methods:

- `__init__(script_specific_scripts=None)`: Initialize pretokenizer. `script_specific_scripts` is an optional list of script names to pretokenize character-by-character. If None, uses DEFAULT_SCRIPT_SPECIFIC_SCRIPTS.
- `pretokenize(text)`: Main pretokenization method using script-aware approach
- `pretokenize_gpt4o_style(text)`: GPT-4o style pretokenization for comparison

### Convenience Functions

- `create_pretokenizer()`: Create a new Pretokenizer instance with default settings
- `pretokenize(text)`: Simple function using default pretokenizer (creates new instance each call)

**Performance Note**: The `pretokenize()` convenience function creates a new `Pretokenizer` instance on each call, which includes building a Unicode script lookup array (~0.12 seconds). For processing multiple texts, create a single `Pretokenizer` instance and reuse it.

## Supported Scripts

### Asian Scripts (Character-by-character)

- Han (Chinese or Japanese (Kanji) characters)
- Hiragana (Japanese)
- Katakana (Japanese)
- Thai
- Myanmar (Burmese)
- Khmer (Cambodian)
- Lao

### Other Scripts (Word-level)
- Latin, Cyrillic, Arabic, Hebrew, Greek, and others
- Intelligent handling of:
  - Contractions (we're, can't)
  - camelCase and PascalCase
  - Numbers with proper grouping
  - Emojis and symbols
  - Various types of whitespace

## Testing

Run the module directly to see example outputs:

```bash
source ../.venv/bin/activate  # Activate virtual environment
python pretokenize.py
```

Or run the comprehensive examples:

```bash
python example_usage.py
```

## Dependencies

- `regex`: Enhanced regular expression library

## Performance

The pretokenizer uses precompiled regex patterns and a Unicode script lookup array for efficient processing. Performance testing showed it processes multilingual text efficiently while maintaining accuracy.

## Migration from Notebook

This module was cleaned up from `Graphemes.ipynb` with the following changes:

- Removed all cell markers (`# In[X]:`)
- Extracted core functionality into a clean class structure
- Removed exploratory/debugging code
- Added proper documentation and type hints
- Created convenience functions for common use cases
- Added comprehensive test examples
- Maintained backward compatibility with legacy patterns

The original research and analysis remains in the notebook for reference.
