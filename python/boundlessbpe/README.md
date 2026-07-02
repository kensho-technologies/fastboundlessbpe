# BoundlessBPE

The repo contains the code to accompany the BoundlessBPE papers:
[Boundless Byte Pair Encoding](https://arxiv.org/abs/2504.00178) and
[Faster Superword Tokenization](https://arxiv.org/abs/2604.05192).

This implementation is based on the following research:

- **BoundlessBPE**: [Boundless Byte Pair Encoding: Breaking the Pre-tokenization Barrier](https://arxiv.org/abs/2504.00178)
- **Faster Superword Tokenization**: [Faster Superword Tokenization](https://arxiv.org/abs/2604.05192)
- **SuperBPE**: [SuperBPE: Space Travel for Language Models](https://superbpe.github.io/)
- **PickyBPE**: [PickyBPE: Compression-Efficient Tokenizers for Large Language Models](https://arxiv.org/abs/2409.04599)

## Overview

BoundlessBPE implements a tokenization pipeline with the following key components:

```
Text → Pretokenization → Byte-level BPE → Supermerges → Token IDs
```

### Architecture

```
boundlessbpe/
├── __init__.py         # Package exports
├── train.py            # Training: BpeTrainer, SuperBpeTrainer, BoundlessBpeTrainer
├── inference.py        # Tokenizer class for encoding/decoding
├── inferencedata.py    # InferenceData: model serialization/deserialization
├── pretokenize.py      # Pretokenizer: script-aware text splitting
├── regexconstants.py   # Regex patterns (GPT-2, GPT-4, GPT-4o, custom)
└── util.py             # Byte encoding utilities, file I/O helpers
```

## Key Concepts

### 1. Pretokenization (`pretokenize.py`)

The `Pretokenizer` class splits text into chunks before BPE processing:

- **Single script per pretoken**: Pretokens containing multiple scripts are split, so any pretoken will only contain a single script. 
- **Script-aware tokenization**: Asian scripts (CJK, Thai, Myanmar, Khmer, Lao) are tokenized character-by-character, while Latin and other scripts use word-level patterns
- **Grapheme clustering**: Properly handles combining marks, ZWJ sequences, emoji modifiers
- **Number grouping**: Numbers are split into groups of 1-3 digits (right-to-left grouping), based on [Tokenization Counts: the impact of tokenization on arithmetic in frontier LLMs](https://arxiv.org/abs/2402.14903)

```python
from boundlessbpe import Pretokenizer

pretokenizer = Pretokenizer()

# Basic words with leading spaces preserved
pretokenizer.pretokenize("Hello world")
# → ["Hello", " world"]

# Script-aware: Asian scripts split character-by-character
# Multiple scripts are split apart
pretokenizer.pretokenize("漢字カタカナABC")
# → ["漢", "字", "カ", "タ", "カ", "ナ", "ABC"]

# Mixed scripts split at boundaries
pretokenizer.pretokenize("Hello мир 你好 world!")
# → ["Hello", " мир", " 你", "好", " world", "!"]

# CamelCase splitting
pretokenizer.pretokenize("XMLHttpRequest")
# → ["XMLHttp", "Request"]

# Contractions kept together
pretokenizer.pretokenize("we're can't you've")
# → ["we're", " can't", " you've"]

# Numbers grouped right-to-left in 1-3 digit chunks
pretokenizer.pretokenize("1234567")
# → ["1", "234", "567"]

# Emoji with ZWJ sequences, skin tones, flags
pretokenizer.pretokenize("👍🏽👩‍👩‍👧‍👦🇺🇳")
# → ["👍🏽", "👩‍👩‍👧‍👦", "🇺🇳"]

# Devanagari with combining marks and ZWJ conjuncts
pretokenizer.pretokenize("क्‍ष")
# → ["क्‍ष"]
```

### 2. First-Pass Training: BPE with Optional PickyBPE Deletions (`train.py`)

The `BpeTrainer` class implements standard BPE with an extension called **PickyBPE**:

- Starts with 243 valid UTF-8 byte tokens as the initial vocabulary
- Iteratively merges the most frequent adjacent token pairs
- **PickyBPE deletion**: After each merge, if either input token has an Intersection-over-Self (IOS) ratio ≥ τ (tau), that token is deleted from the vocabulary

**IOS Metric**: For a merge (A, B) → AB with counts c_a, c_b, c_ab:
- ios_a = c_ab / c_a (what fraction of A's occurrences are in this merge)
- ios_b = c_ab / c_b (what fraction of B's occurrences are in this merge)

If ios ≥ τ, the token is "subsumed" by the merge and can be deleted to save vocabulary space. Set tao > 1 to disable deletions. 

**Blowup modes**: When deleting a token, it can be replaced with:
- `blowup=True`: Individual bytes (the approach in the BoundlessBPE paper)
- `blowup=False`: The pair of tokens that created it (the approach in the original PickyBPE paper)

### 3. Second-Pass Training: Supermerges

After first-pass training, a second pass can create **supermerges** - merges that span pretokenization boundaries. Two strategies are available:

#### BoundlessBPE (`BoundlessBpeTrainer`)

- Tokens start locked; they unlock when they form complete "words" (pretokens)
- Supermerges compete with regular merges at each step
- If a supermerge has higher count than the next regular merge, it's applied
- Otherwise, the regular merge just advances (unlocking tokens as needed)
- Final model maintains the N total operations as the first pass training (trading regular merges for supermerges)

#### SuperBPE (`SuperBpeTrainer`)
- All word-level tokens are immediately unlocked for merging
- Trains additional supermerge tokens on top of the word model
- Final model = N word operations + M supermerges

### 4. Inference (`inference.py`)

The `Tokenizer` class provides encoding and decoding:

```python
from boundlessbpe import Tokenizer

# Load a trained model
tokenizer = Tokenizer.from_file("path/to/model.model")

# Encode text to token IDs
ids = tokenizer.encode("Hello, world!")

# Decode back to text
text = tokenizer.decode(ids)

# Batch operations (HuggingFace-compatible)
# Note that these are not run in parallel at present, so there is no speedup from using `encode_batch`
ids_list = tokenizer.encode_batch(["Hello", "World"])
texts = tokenizer.decode_batch(ids_list)

# Vocabulary inspection
vocab_size = tokenizer.get_vocab_size()
vocab = tokenizer.get_vocab()  # str -> int mapping
```

**Reachable Token Optimization**: During loading, the tokenizer pre-computes which vocabulary tokens are "reachable" - tokens that tokenize back to themselves. This covers ~98% of cases and provides a significant speedup.

### 5. Model Serialization (`inferencedata.py`)

Models are saved in a text-based `.model` format:

```
BoundlessBPE v2 <model_type>     # Header: word, boundless, or superbpe
{JSON config}                     # tau, blowup, regex patterns, etc.
vocab
<vocab entries with counts>
special_tokens
<special token entries>
merges
<merge rules with counts>
deletions
<deleted tokens>
[superwords]                      # Only for two-pass models
{superword config}
...
```

## Usage

### Training a First-Pass Model (BPE with PickyBPE)

```python
from boundlessbpe import BpeTrainer

trainer = BpeTrainer()

trainer.train(
    tau=0.9,                    # Deletion threshold (0.0-1.0, >1 to disable)
    filepath="data.jsonl",      # Training data (JSONL with "text" field)
    outprefix="models/word",    # Output path prefix
    num_lines=1000000,          # Documents to process
    vocab_size=32768,           # Target vocabulary size
    recalc=10000,               # Verification interval
    blowup=True,                # Deletion strategy
    verbose=True
)

trainer.save("models/word_final")
```

### Training a Two-Pass Model (BoundlessBPE)

```python
from boundlessbpe import BoundlessBpeTrainer

trainer = BoundlessBpeTrainer()

trainer.train(
    filepath="data.jsonl",
    outprefix="models/boundless",
    num_lines=1000000,
    recalc=10000,
    word_model_file="models/word_final.model",
    verbose=True
)

trainer.save("models/boundless_final")
```

### Training a Two-Pass Model (SuperBPE)

```python
from boundlessbpe import SuperBpeTrainer

trainer = SuperBpeTrainer()

trainer.train(
    filepath="data.jsonl",
    outprefix="models/super",
    num_lines=1000000,
    vocab_size=8192,            # Number of supermerge tokens
    recalc=10000,
    word_model_file="models/word_final.model",  # First-pass model
    verbose=True
)

trainer.save("models/super_final")
```

### Using a Trained Model

```python
from boundlessbpe import Tokenizer

# Load any model type (auto-detects format)
tokenizer = Tokenizer.from_file("models/boundless_final.model")

# Basic encoding/decoding
text = "Hello, world! 你好世界"
ids = tokenizer.encode(text)
decoded = tokenizer.decode(ids)
assert decoded == text

# With special tokens
tokenizer.add_special_tokens(["<|endoftext|>", "<|pad|>"])
ids = tokenizer.encode("Hello<|endoftext|>", allowed_special="all")

# Vocabulary info
print(f"Vocab size: {tokenizer.get_vocab_size()}")
print(f"Token 'hello' -> {tokenizer.token_to_id('hello')}")
print(f"ID 256 -> '{tokenizer.id_to_token(256)}'")
```

## Data Structures

### InferenceData

Core data class holding everything needed for inference:

| Field | Type | Description |
|-------|------|-------------|
| `vocab` | `dict[bytes, int]` | Token → ID mapping |
| `inv_vocab` | `dict[int, bytes]` | ID → Token mapping |
| `merges` | `dict[int, tuple]` | Index → ((left, right), count, unlocked_flag) |
| `deletions` | `dict[int, bytes]` | Index → deleted token |
| `merges_lookup` | `dict[tuple, list[int]]` | Pair → list of merge indices |
| `deletions_lookup` | `dict[bytes, list[int]]` | Token → list of deletion indices |
| `tau` | `float` | Deletion threshold |
| `blowup` | `bool` | Deletion strategy |
| `is_super` | `bool` | Is this a superword model? |
| `superbpe_mode` | `bool` | SuperBPE (True) vs BoundlessBPE (False) |
| `pretokenizer` | `Pretokenizer` | Pretokenization patterns |

### Tokenizer

The main inference class wraps InferenceData:

| Field | Type | Description |
|-------|------|-------------|
| `words` | `InferenceData` | Word-level model |
| `superwords` | `InferenceData \| None` | Superword model (two-pass only) |
| `vocab` | `dict[bytes, int]` | Unified vocabulary |
| `inv_vocab` | `dict[int, bytes]` | Unified inverse vocabulary |
| `reachable_vocab` | `set[bytes]` | Tokens that encode to themselves |
| `possible_superwords` | `set[bytes]` | Tokens eligible for supermerges |

## Regex Patterns (`regexconstants.py`)

Several pretokenization patterns are available:

| Pattern | Description |
|---------|-------------|
| `GPT2_SPLIT_PATTERN` | Original GPT-2 tokenizer pattern |
| `GPT4_SPLIT_PATTERN` | GPT-4 tokenizer pattern |
| `GPT4O_SPLIT_PATTERN` | GPT-4o tokenizer pattern |
| `ULTIMATE_PATTERN` | Enhanced pattern with better Unicode/emoji support |
| `IMPROVED_MERGE_PATTERN` | Pattern for determining supermerge eligibility |

The `IMPROVED_MERGE_PATTERN` (`^(?=.+\p{L})(?:\p{L}\p{M}*|[ _'\u2019])+$`) defines which tokens can participate in supermerges - essentially letter-based tokens with spaces, underscores, or apostrophes.

## HuggingFace Compatibility

The `Tokenizer` class provides HuggingFace-compatible methods:

- `encode(text, allowed_special)` / `decode(ids)`
- `encode_batch(texts)` / `decode_batch(ids_list)`
- `get_vocab(with_added_tokens)` / `get_vocab_size(with_added_tokens)`
- `token_to_id(token)` / `id_to_token(token_id)`
- `add_special_tokens(special_tokens)`
- `from_file(path)` (class method)

## Algorithm Details

### Encoding Process

1. **Pretokenize**: Split text into chunks using script-aware regex patterns
2. **Byte conversion**: Convert each chunk to list of single-byte tokens
3. **Reachable check**: If chunk is a reachable token, skip to step 6
4. **Merge/delete loop**: Apply merges and deletions in index order
   - If next operation is a merge with lower index, apply it
   - If next operation is a deletion with lower index, blow up the token
5. **Supermerge**: If two-pass model, apply supermerges to adjacent word tokens
6. **Vocabulary lookup**: Convert byte tokens to integer IDs

### Training Process (BPE)

1. **Pretokenize** training corpus, counting chunk frequencies
2. **Initialize** vocabulary with 243 valid UTF-8 bytes
3. **Compute** initial pair counts using heapdict for O(log n) selection
4. **Main loop** until vocab_size reached:
   - Select highest-count pair
   - Merge all occurrences, updating pair and single counts incrementally
   - Check IOS for deletion candidates
   - If IOS ≥ τ, delete token and update counts
5. **Save** model with vocabulary, merges, deletions, and config

### Training Process (Two-Pass)

1. **Load** first-pass word model
2. **Pretokenize** corpus using word model (apply all word merges)
3. **Find** superword runs (consecutive tokens eligible for supermerges)
4. **Initialize** pair counts on superword runs only
5. **Main loop**:
   - **SuperBPE**: Just apply best supermerge
   - **BoundlessBPE**: Compare supermerge count vs next regular merge count
     - If supermerge wins: apply it
     - If regular wins: just unlock tokens (don't apply merge)
6. **Save** unified model (trimmed word + superwords section)

## Dependencies

- `regex`: Advanced regex with Unicode property support
- `heapdict`: Heap-based dictionary for efficient pair selection

## License

See the project root for license information.
