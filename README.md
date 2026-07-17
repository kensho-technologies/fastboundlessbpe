# BoundlessBPE

[![CI](https://github.com/kensho-technologies/fastboundlessbpe/actions/workflows/ci.yml/badge.svg)](https://github.com/kensho-technologies/fastboundlessbpe/actions/workflows/ci.yml)

A fast Rust implementation of the BoundlessBPE tokenizer with Python bindings, plus an
identical pure-Python implementation for environments where a Rust extension is unavailable.

The package provides:

- A **Rust inference engine** with Python bindings (`FastTokenizer`) and parallel batch inference.
- A **pure-Python inference** implementation (`Tokenizer`) with identical results, no Rust toolchain needed.
- **Training routines** for word-level BPE, BoundlessBPE, and SuperBPE models (both Rust and Python).

## Project Structure

```text
fastboundlessbpe/
├── src/                           # Rust implementation
│   ├── lib.rs                     # PyO3 bindings and module exports
│   ├── tokenizer.rs               # Core BPE inference logic
│   ├── trainer.rs                 # Training engine (BPE, BoundlessBPE, SuperBPE)
│   ├── training_types.rs          # Compact TokenId training state and chunk operations
│   ├── vocabulary.rs              # Token vocabulary management
│   ├── inference_data.rs          # Merge/deletion operations data
│   ├── pretokenize.rs             # Regex-based text pretokenization
│   ├── script_data.rs             # Unicode script lookup (generated)
│   ├── byte_encoding.rs           # Byte-to-character encoding utilities
│   ├── constants.rs               # Regex patterns and constants
│   └── error.rs                   # Error types
├── python/boundlessbpe/           # Python package
│   ├── __init__.py                # Package initialization and exports
│   ├── inference.py               # Python inference implementation (Tokenizer)
│   ├── train.py                   # Python training (BpeTrainer, BoundlessBpeTrainer, SuperBpeTrainer)
│   ├── vocabulary.py              # Vocabulary class
│   ├── inferencedata.py           # InferenceData class
│   ├── pretokenize.py             # Pretokenizer class
│   ├── script_data.py             # Unicode script lookup (generated)
│   ├── allngramcnt.py             # N-gram counting for BoundlessBPE candidates
│   ├── ngram_split.py             # N-gram greedy splitting
│   ├── regexconstants.py          # Regex pattern constants
│   └── util.py                    # Encoding and I/O utilities
├── python/tests/                  # Python tests
│   ├── test_pretokenize.py        # Pretokenizer tests
│   └── test_pretokenize_gpt4o.py  # GPT-4o pretokenizer tests
├── tests/                         # Rust integration tests
│   └── test_parity.rs             # Rust-Python parity tests
├── unicode_data/                  # Unicode data files
│   ├── Scripts.txt                # Unicode 17.0 script assignments
│   └── generate_script_data.py    # Generates script_data.py and script_data.rs
└── examples/                      # Runnable examples and driver scripts
    ├── example_usage.py           # Pretokenizer usage examples
    ├── compare_implementations.py # Verify Rust and Python produce identical results
    ├── test_rust_comparison.py    # Python vs Rust correctness & performance
    ├── test_inference.py          # Inference roundtrip / timing
    ├── train_word.py              # Train first-pass word model
    ├── train_boundless.py         # Train second-pass BoundlessBPE model
    └── train_super.py             # Train second-pass SuperBPE model
```

## Installation

Building from source requires a Rust toolchain.

```bash
git clone https://github.com/kensho-technologies/fastboundlessbpe
cd fastboundlessbpe

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install maturin (build tool for Rust+Python packages)
pip install maturin

# Build and install in development mode.
# This compiles the Rust extension and installs the runtime
# dependencies (regex, heapdict) automatically.
maturin develop --release

# Optional: install test dependencies (pytest, numpy)
pip install -e ".[test]"

# Optional: install dev dependencies (mypy) for type checking
pip install -e ".[dev]"

# Or install both at once
pip install -e ".[test,dev]"
```

## Quick Start

### Inference

There are two inference implementations with identical results:

- **`FastTokenizer`** (Rust with Python bindings and parallel batch execution)
- **`Tokenizer`** (pure Python) -- no Rust toolchain needed

Both share the same core API:

```python
from boundlessbpe import FastTokenizer  # Rust (fast)
# or: from boundlessbpe import Tokenizer  # Python (no Rust needed)

tokenizer = FastTokenizer()
tokenizer.load("path/to/model.model")

# Encode text to token IDs
token_ids = tokenizer.encode_ordinary("Hello, world!")

# Decode back to text
decoded = tokenizer.decode(token_ids)

# Encode with special token handling
tokens = tokenizer.encode("Hello <|endoftext|>", allowed_special="all")

# Batch encode/decode
batch_ids = tokenizer.encode_batch(["Hello", "World"])
batch_text = tokenizer.decode_batch(batch_ids)

# Get vocabulary size
size = tokenizer.get_vocab_size()
```

The Python `Tokenizer` exposes additional methods not available on `FastTokenizer`:

| Method | `Tokenizer` (Python) | `FastTokenizer` (Rust) |
|--------|:--------------------:|:----------------------:|
| `load(path)` | yes | yes |
| `encode_ordinary(text)` | yes | yes |
| `encode(text, allowed_special)` | yes | yes |
| `encode_batch(texts, allowed_special)` | yes | yes |
| `encode_ordinary_chunks(text)` | yes | yes |
| `decode(ids)` | yes | yes |
| `decode_bytes(ids)` | yes | yes |
| `decode_batch(ids_list)` | yes | yes |
| `get_vocab_size(with_added_tokens)` | yes | yes |
| `save(file_prefix)` | yes | -- |
| `add_special_tokens(tokens)` | yes | -- |
| `get_vocab(with_added_tokens)` | yes | -- |
| `token_to_id(token)` | yes | -- |
| `id_to_token(token_id)` | yes | -- |
| `from_file(path)` | yes | -- |
| `.vocab`, `.words`, `.superwords` | yes | -- |

You need a `.model` file to load. Train one yourself with the two-pass workflow below,
or use the driver scripts in `examples/`.

### Training

Training uses a two-pass approach:

1. **First pass** (`BpeTrainer`): Train a word-level BPE model with optional PickyBPE deletions
2. **Second pass** (`BoundlessBpeTrainer` or `SuperBpeTrainer`): Train supermerges using the word model

```python
from boundlessbpe import Pretokenizer, BpeTrainer, BoundlessBpeTrainer
from boundlessbpe import SCRIPT_SPECIFIC_REGEX, DEFAULT_SCRIPT_SPECIFIC_SCRIPTS

# Create pretokenizer with script-aware mode
pretokenizer = Pretokenizer(
    script_specific_regex=SCRIPT_SPECIFIC_REGEX,
    script_specific_scripts=DEFAULT_SCRIPT_SPECIFIC_SCRIPTS
)

# --- First pass: train word model ---
trainer = BpeTrainer(pretokenizer)
trainer.train(
    tau=0.9,                        # Deletion threshold (>1.0 disables deletions)
    filepath="data/minipile.jsonl",
    outprefix="./models/word_model",
    num_lines=100000,
    vocab_size=131072,
    recalc=8192,
    blowup=True,                    # Delete to bytes (True) or merge pair (False)
)
trainer.register_special_tokens(["<|endoftext|>"])
trainer.save("./models/word_model_final")

# --- Second pass: train BoundlessBPE model ---
trainer2 = BoundlessBpeTrainer(pretokenizer)
trainer2.train(
    filepath="data/minipile.jsonl",
    outprefix="./models/boundless_model",
    num_lines=100000,
    recalc=8192,
    word_model_file="./models/word_model_final.model",
)
trainer2.register_special_tokens(["<|endoftext|>"])
trainer2.save("./models/boundless_model_final")
```

Rust implementations of the trainers are also available as `FastBpeTrainer`,
`FastBoundlessBpeTrainer`, and `FastSuperBpeTrainer` with the same interface.

The `examples/` directory contains runnable driver scripts for each training pass:

```bash
# Train first-pass word model
python -u examples/train_word.py --num-lines 10000

# Train second-pass BoundlessBPE model (requires word model from first pass)
python -u examples/train_boundless.py --num-lines 10000

# Train second-pass SuperBPE model (requires word model from first pass)
python -u examples/train_super.py --num-lines 10000 --supermerges 53180
```

#### Command-line options

All three scripts share a common set of options:

| Option | Default | Description |
|--------|---------|-------------|
| `--num-lines` | *(required)* | Number of documents to read from the training file |
| `--filepath` | `data/minipile.jsonl` | Path to the JSONL training corpus |
| `--vocab-size` | `131072` | Target vocabulary size (word-model size for the second pass) |
| `--max-bytes` | `10000000000` | Maximum bytes to process (10 GB) |
| `--recalc` | `8192` | Verification frequency (`0` disables verification) |
| `--checkpoint-iterations` | `8192` | Save a checkpoint every N iterations |
| `--output-prefix` | `./models/twopass` | Output path prefix for model files |
| `--fast` | off | Use the Rust trainer instead of the Python one |
| `--simple` | off | Use plain `GPT4O_REGEX` with no script-specific pretokenization |
| `--save-pretokens` | `None` | Write pretokenization data to this TSV file |

**`train_word.py`** (first pass) adds:

| Option | Default | Description |
|--------|---------|-------------|
| `--tau` | `1.1` | PickyBPE deletion threshold (`>1.0` disables deletions) |
| `--blowup` | off | Delete tokens by splitting to bytes instead of re-merging the pair |

**`train_boundless.py`** and **`train_super.py`** (second pass) add:

| Option | Default | Description |
|--------|---------|-------------|
| `--word-model` | `None` | Explicit path to the first-pass word model (overrides the auto-constructed path) |
| `--first-pass-tau` | `1.1` | `tau` used in the first pass (only used to reconstruct the word-model filename) |
| `--blowup` | off | Blowup mode used in the first pass (filename reconstruction) |
| `--fast-word-model` | off | The word model was trained with `--fast` (filename reconstruction) |
| `--greedy-split` | off | Apply n-gram greedy split before training |
| `--min-count` | `15` | Minimum count floor for n-gram counting (with `--greedy-split`) |
| `--max-ngram-len` | `30` | Maximum n-gram length (with `--greedy-split`) |

**`train_super.py`** additionally requires `--supermerges N` — the number of supermerges to
create, which should match the supermerge count from the corresponding BoundlessBPE run.

> The second-pass scripts locate the word model by reconstructing its filename from
> `--vocab-size`, `--first-pass-tau`, `--blowup`, and `--fast-word-model`. If your word model
> lives elsewhere or was named differently, pass `--word-model /path/to/word_model.model`
> directly.

### Choosing a pretokenization regex

Pretokenization is the first step of the pipeline — it splits raw text into chunks before BPE
runs. The `Pretokenizer` accepts up to three patterns, all importable from `boundlessbpe`:

```python
from boundlessbpe import Pretokenizer
from boundlessbpe import GPT4O_REGEX, SCRIPT_SPECIFIC_GPT4O_REGEX, DEFAULT_SCRIPT_SPECIFIC_SCRIPTS

pretokenizer = Pretokenizer(
    main_regex=GPT4O_REGEX,                          # pattern for most text
    script_specific_regex=SCRIPT_SPECIFIC_GPT4O_REGEX,  # pattern for the scripts below
    script_specific_scripts=DEFAULT_SCRIPT_SPECIFIC_SCRIPTS,
)
```

- **`main_regex`** (default `GPT4O_REGEX`) — applied to all text. In script-aware mode it is
  applied to everything *except* the script-specific scripts.
- **`script_specific_regex`** (optional) — providing it enables *script-aware mode*, where the
  listed scripts (e.g. CJK, Thai) are split character-by-character. Leaving it `None` uses
  *simple mode*: `main_regex` over the whole text.
- **`script_specific_scripts`** — the scripts that use `script_specific_regex`. Required
  whenever `script_specific_regex` is set.
- **`merge_pattern`** (default `SIMPLE_MERGE_PATTERN`) — controls which tokens are eligible to
  participate in supermerges.

Available patterns (all importable from `boundlessbpe`):

| Constant | Role | Notes |
|----------|------|-------|
| `GPT4O_REGEX` | `main_regex` | Default; the GPT-4o split pattern |
| `GPT2_REGEX` | `main_regex` | The original GPT-2 pattern |
| `GPT4_REGEX` | `main_regex` | The GPT-4 pattern |
| `GPT4O_SPLIT_PATTERN` | `main_regex` | GPT-4o variant assembled from parts |
| `ULTIMATE_PATTERN` | `main_regex` | Splits camelCase / snake_case / acronyms more aggressively |
| `WORD_LEVEL_REGEX` | `main_regex` | Word-level pattern used for the script-aware default |
| `SCRIPT_SPECIFIC_REGEX` | `script_specific_regex` | Character-level pattern (pairs with `WORD_LEVEL_REGEX`) |
| `SCRIPT_SPECIFIC_GPT4O_REGEX` | `script_specific_regex` | Character-level pattern (pairs with `GPT4O_REGEX`) |
| `SIMPLE_MERGE_PATTERN` | `merge_pattern` | Default; token is merge-eligible if it contains any letter |
| `IMPROVED_MERGE_PATTERN` | `merge_pattern` | Letters plus spaces/underscores/apostrophes only |
| `DEFAULT_SCRIPT_SPECIFIC_SCRIPTS` | `script_specific_scripts` | `Han`, `Hiragana`, `Katakana`, `Thai`, `Myanmar`, `Khmer`, `Lao` |

For example, to train with the more aggressive `ULTIMATE_PATTERN` in simple mode:

```python
from boundlessbpe import Pretokenizer, BpeTrainer, ULTIMATE_PATTERN

pretokenizer = Pretokenizer(main_regex=ULTIMATE_PATTERN)  # simple mode, no script splitting
trainer = BpeTrainer(pretokenizer)
```

You can also pass your own regex string to `main_regex` — anything compatible with the
[`regex`](https://pypi.org/project/regex/) module works. The Rust trainers
(`FastBpeTrainer`, etc.) take the patterns as the `main_regex` / `script_specific_regex` /
`script_specific_scripts` constructor arguments rather than a `Pretokenizer` object.

### Verification & Benchmarking

```bash
# Run Python vs Rust inference comparison
python examples/test_rust_comparison.py --model-file path/to/model.model --dataset path/to/dataset.jsonl

# Run Rust unit tests
cargo test --no-default-features

# Run Python pretokenizer tests
python -m pytest python/tests/ -v
```

**Getting training data:** for training/benchmarking you need a JSONL corpus. The minipile
dataset works well:

```bash
mkdir -p data

pip install huggingface-hub
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='JeanKaddour/minipile', filename='data/train.jsonl', local_dir='data', local_dir_use_symlinks=False)"
mv data/data/train.jsonl data/minipile.jsonl
```

### Editing a model (adding special tokens)

To load an existing `.model`, add special tokens, and save it back, use the pure-Python
`Tokenizer`. Model authoring (loading, modifying, saving) lives on `Tokenizer`, not on
`FastTokenizer` or the `Fast*Trainer` classes — the Rust trainers only build a vocabulary
by training from scratch and have no `load()`.

```python
from boundlessbpe import Tokenizer

t = Tokenizer()
t.load("path/to/model.model")

# Append special tokens (accumulates, skips duplicates, HuggingFace-compatible indexing)
t.add_special_tokens(["<|mytoken|>", "<|another|>"])

# Writes path/to/new_model.model
t.save("path/to/new_model")
```

The resulting `.model` loads normally in `FastTokenizer` for fast inference afterward. Using
the slower Python `Tokenizer` here is fine: this is a one-time edit, not the inference hot path.

## Performance

**Current Rust vs Python batch-inference reference** (same current BPE model, 20,000 generated
texts, 971,282 output tokens, median of three isolated runs):

| Implementation | Rayon workers | Encode time | Throughput | Peak RSS |
|---|---:|---:|---:|---:|
| `Tokenizer` (pure Python) | serial | 2.784 s | 348,891 tokens/s | 33.1 MiB |
| `FastTokenizer` (current Rust) | 4 | **0.208 s** | **4,670,657 tokens/s** | 48.9 MiB |

This is a 13.39x encoding speedup for this batch-oriented workload. The Rust result includes four
Rayon workers; its higher RSS includes the extension and worker overhead. Single-text latency,
model, text length, CPU topology, and `RAYON_NUM_THREADS` can change the result, so this table is
a reference configuration rather than a universal claim.

**Correctness**: 100% identical results between Python and Rust implementations over 1,000,000 documents.

### Trainer migration and thread-scaling benchmark

The table below compares the original serial trainer (`c2dce31`) with the Rayon-only trainer
(`1c4d41d`) and the current TokenId trainer. Every model file was byte-identical across versions
and thread counts. Peak RSS is for the whole Python process, including the input batch and Python
runtime. The SuperBPE fixture is intentionally run-heavy to exercise the phase-2 aggregation map.

| Workload | Original serial | Rayon, 1 worker | Rayon, 4 workers | TokenId, 1 worker | TokenId, 4 workers |
|---|---:|---:|---:|---:|---:|
| BPE training: 20k JSONL documents, 240k unique pretokens | 4.102 s, 218.2 MiB | 3.970 s, 225.7 MiB | 3.896 s, 228.3 MiB | 1.130 s, 197.7 MiB | **0.756 s, 201.2 MiB** |
| SuperBPE training: 500k JSONL documents, 302,999 unique runs | 3.476 s, 269.4 MiB | 4.543 s, 321.4 MiB | 3.353 s, 322.1 MiB | 2.961 s, 151.4 MiB | **1.784 s, 151.7 MiB** |

At four workers, the TokenId trainer is 5.43x faster than the original BPE trainer and 1.95x
faster than the original SuperBPE trainer in these fixtures. The run-heavy SuperBPE case also
uses 43.7% less peak RSS than the original and 52.9% less than the Rayon-only trainer.

For this 12-logical-core host, four workers improve the run-heavy SuperBPE scan by 1.35x for the
Rayon-only trainer and 1.66x for the TokenId trainer. Peak RSS is nearly flat across one and four
workers; the phase-2 unique-run map, rather than Rayon worker count, is the dominant memory cost.
Thread count is workload- and machine-dependent, so use these as a tuning starting point rather
than a universal default.

### Technical trainer changes

The migration is internal to Rust training. It does not change the Python API, model-file format,
vocabulary numbering, or inference representation.

```rust
type TokenId = u32;
type Pair = (TokenId, TokenId);
type Chunk = Vec<TokenId>;
```

| Area | Before | Current implementation |
|---|---|---|
| Token storage | Every trainer map/chunk owned `Vec<u8>` tokens, often repeatedly. | `TokenArena` owns each byte string once and returns a stable, append-only `TokenId`. It also records the first merge parent for deletion reconstruction. |
| Corpus chunks | `Vec<Vec<Vec<u8>>>` through trainer initialization and rewrites. | Chunks are `Vec<TokenId>` after the raw corpus/trainer boundary. Initial BPE interning assigns IDs in byte order. |
| Hot trainer maps | Single counts, pair heap keys, pair adjacency, locations, and unlock state used byte-vector keys. | Those structures use `TokenId`/`Pair` keys. Only heap priority metadata retains byte copies, solely to preserve lexical equal-count tie-breaking. |
| Initial counting | One byte-backed pass built counts and locations. | Fixed-size Rayon batches build worker-local ID count maps, then reduce in batch order for deterministic location indices. |
| Merge/delete updates | Byte-token rewrites and byte-backed pair deltas. | Non-overlapping merge and blow-up operate on IDs. Workers mutate disjoint chunks in place and reduce local `Pair` deltas before the trainer commits locations/counts. |
| Sparse rewrites | The parallel path could scan the full corpus for a small candidate set. | Small or sparse candidate sets update serially; large dense sets use Rayon with a candidate bitmap. |
| SuperBPE phase 2 | `AHashMap<Vec<Vec<u8>>, i64>` stored each unique run with nested byte allocations. | Word-vocabulary tokens are pre-interned; the scan stores `AHashMap<Vec<TokenId>, i64>`. Eligibility is an ID set rather than cloned token bytes. |
| Optional n-grams | N-gram and greedy-split maps used nested byte-vector chunks. | Both use `AHashMap<Vec<TokenId>, i64>` / `AHashSet<Vec<TokenId>>`. |

The heap still compares the selected pair’s underlying bytes when frequencies tie. This is
intentional: `TokenId`s reflect creation order, while the previous trainer selected equal-count
pairs in lexical byte order. Keeping that ordering is what allows the current trainer to produce
byte-identical models.

### Parallel execution

The Rust implementation uses Rayon for bounded corpus-scan batches and for sufficiently
large `encode_batch` / `decode_batch` calls. Corpus reading remains ordered, and training
applies document and byte limits in that order, so parallel work does not change which
documents are included. Python bindings release the GIL while Rust performs loading,
training, encoding, and decoding.

Rayon uses its default worker count unless configured. Set `RAYON_NUM_THREADS` when the
tokenizer shares a machine with other CPU-intensive work:

```bash
RAYON_NUM_THREADS=12 python examples/train_super.py ...
```

### Compact trainer representation and memory scaling

Rust training uses a private, append-only `TokenId` (`u32`) arena. Token bytes and their merge
parents are stored once; the hot trainer structures—chunks, pair counts, token locations, unlock
state, and rewrite deltas—store IDs instead of cloned `Vec<u8>` values. Vocabulary indices are
not used as training IDs because BoundlessBPE deletions can renumber vocabulary entries.

For word-level BPE, raw pretokens are interned once at the corpus/trainer boundary. For the
SuperBPE second pass, word-vocabulary tokens are interned before scanning, so the phase-2 map is
`AHashMap<Vec<TokenId>, i64>` rather than a nested byte-vector map. Optional n-gram counting and
greedy splitting use the same ID chunks. Merge selection still breaks equal-frequency ties by
token bytes, preserving historical deterministic model output.

Dense rewrite sets are processed in parallel; sparse sets are updated serially to avoid scanning
the complete corpus and allocating a corpus-sized candidate bitmap for a small change.

This is an in-memory trainer, not an out-of-core one. `max_bytes` limits input consumed, but does
not impose a fixed memory ceiling: peak memory still depends on the number and total ID-length of
unique chunks/runs, active pairs, and—when `--greedy-split` is enabled—the number of retained
n-grams. Use a representative pilot corpus and conservative `--min-count` / `--max-ngram-len`
settings before attempting multi-gigabyte training.

## Model File Format

BoundlessBPE uses `.model` files in a unified v2 format:

```text
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
<JSON config>                     # tau, is_super, regex patterns, etc.
merges
<count>
<idx> <left> <right> <count> <unlocked_flag>
...
deletions
<count>
<idx> <token>
...
superwords                        # (only for boundless/superbpe models)
<JSON config>
merges
...
deletions
...
```

## Requirements

- **Python**: 3.10+
- **Rust**: 1.70+ (only needed for building the Rust extension from source)
- **Python dependencies**: `regex`, `heapdict` (installed automatically)
- **Rust dependencies**: `pyo3`, `fancy-regex`, `ahash`, `serde`, `serde_json`, `priority-queue`

## Acknowledgments

This project builds upon and extends [minBPE](https://github.com/karpathy/minbpe) by Andrej
Karpathy. Several base components are derived from the original minBPE implementation, though
substantially evolved and extended for the BoundlessBPE algorithm.

## License

Apache License 2.0 - see LICENSE file for details.

## Citation

If you use BoundlessBPE in your research, please cite:

```bibtex
@misc{schmidt2025boundlessbytepairencoding,
      title={Boundless Byte Pair Encoding: Breaking the Pre-tokenization Barrier},
      author={Craig W. Schmidt and Varshini Reddy and Chris Tanner and Yuval Pinter},
      year={2025},
      eprint={2504.00178},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2504.00178},
}

@misc{schmidt2026fastersuperwordtokenization,
      title={Faster Superword Tokenization}, 
      author={Craig W. Schmidt and Chris Tanner and Yuval Pinter},
      year={2026},
      eprint={2604.05192},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.05192}, 
}
```
