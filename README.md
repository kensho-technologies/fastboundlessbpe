# BoundlessBPE

[![CI](https://github.com/kensho-technologies/fastboundlessbpe/actions/workflows/ci.yml/badge.svg)](https://github.com/kensho-technologies/fastboundlessbpe/actions/workflows/ci.yml)

BoundlessBPE is a byte-pair-encoding tokenizer that can merge tokens *across* pretokenization
boundaries ("supermerges"), producing multi-word "superword" tokens for better compression than
standard BPE. It also supports optional PickyBPE token deletion. This package lets you **train**
such tokenizers and **use** them for fast encoding/decoding, and can **export** trained models to
the HuggingFace and tiktoken formats for use in standard runtimes. See the papers under
[Citation](#citation) for the method.

The package provides:

- A **Rust inference engine** with Python bindings (`FastTokenizer`) — the fast path, ~2.7x faster.
- A **pure-Python inference** implementation (`Tokenizer`) with identical results, no Rust toolchain needed.
- **Training routines** for word-level BPE, BoundlessBPE, and SuperBPE models (both Rust and Python).
- **Export** to HuggingFace `tokenizer.json` and tiktoken formats.

If you just want to use a trained tokenizer, jump to [Quick Start](#quick-start). To train one,
see [Training](#training). To export one, see [Exporting](#exporting-to-huggingface-and-tiktoken-formats).

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

# Optional: export dependencies (tokenizers, tiktoken) for HuggingFace / tiktoken export
pip install -e ".[export]"

# Optional: test dependencies (pytest, numpy)
pip install -e ".[test]"

# Optional: dev dependencies (mypy) for type checking
pip install -e ".[dev]"

# Or install several at once
pip install -e ".[export,test,dev]"
```

## Quick Start

### Inference

There are two inference implementations with identical results:

- **`FastTokenizer`** (Rust with Python bindings) -- ~2.7x faster
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
| `save_huggingface(path, coarse_regex)` | yes | -- |
| `save_tiktoken(path, coarse_regex)` | yes | -- |
| `add_special_tokens(tokens)` | yes | -- |
| `get_vocab(with_added_tokens)` | yes | -- |
| `token_to_id(token)` | yes | -- |
| `id_to_token(token_id)` | yes | -- |
| `from_file(path)` | yes | -- |
| `.vocab`, `.words`, `.superwords` | yes | -- |

You need a `.model` file to load. Train one yourself with the two-pass workflow below,
or use the driver scripts in `examples/`.

### Training

First you need a JSONL training corpus (one `{"text": ...}` object per line). The minipile
dataset works well:

```bash
pip install huggingface-hub

# minipile is a *dataset* repo and stores the file at data/train.jsonl. Downloading
# with local_dir='.' places it at ./data/train.jsonl; then rename it in place.
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='JeanKaddour/minipile', repo_type='dataset', filename='data/train.jsonl', local_dir='.')"
mv data/train.jsonl data/minipile.jsonl
```

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

Pretokenization is the first step of the pipeline: before any BPE merging, the raw text is split
into chunks (pretokens), and regular merges are only ever applied *within* a chunk — the regex
therefore sets the hard boundaries that ordinary tokens can never cross. (BoundlessBPE's supermerges
combine whole pretokens into a superword, so a superword is always made of complete pretokens,
never a fragment of one.) The choice of regex thus has a large effect on the resulting vocabulary,
so it is a first-class, configurable input.

The `Pretokenizer` supports two modes:

- **Simple mode** — a single `main_regex` is applied to the whole text. This is the default
  (`GPT4O_REGEX`).
- **Script-aware mode** — you additionally supply a `script_specific_regex` and the set of Unicode
  scripts it applies to (`script_specific_scripts`). Text in those scripts is split with the
  script-specific pattern while everything else uses `main_regex`. This is useful for scripts that
  aren't space-delimited: the default `DEFAULT_SCRIPT_SPECIFIC_SCRIPTS` (Han, Hiragana, Katakana,
  Thai, Myanmar, Khmer, Lao) are split character-by-character so they don't form runaway pretokens.

You are **not** limited to the built-in patterns below — `main_regex`, `script_specific_regex`, and
`merge_pattern` accept any pattern string compatible with the [`regex`](https://pypi.org/project/regex/)
module (the built-ins are just convenient, tested defaults). The `Pretokenizer` accepts up to three
patterns, all importable from `boundlessbpe`:

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
| `ULTIMATE_PATTERN` | `main_regex` | Splits camelCase / snake_case / acronyms more aggressively; ~2x slower to pretokenize than `GPT4O_REGEX` (larger alternation with lookaheads) |
| `WORD_LEVEL_REGEX` | `main_regex` | Word-level pattern used for the script-aware default |
| `GPT4O_EXPORT_REGEX` | `main_regex` | GPT-4o variant with leading-space-only words and internal-only apostrophes; train with this to export a superword model (see "Exporting" below) |
| `GPT4O_COARSE_REGEX` | export only | Coarse companion to `GPT4O_EXPORT_REGEX`; passed as `coarse_regex` when exporting a superword model, not for training |
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

`ULTIMATE_PATTERN` splits more aggressively (camelCase, snake_case, acronyms, contractions),
which can improve token quality on code and identifier-heavy text, but it is roughly **2x slower
to pretokenize** than `GPT4O_REGEX` because it is a much larger alternation with lookaheads.
Pretokenization is a fixed per-document cost paid in both training and inference, so weigh the
split quality against the throughput hit for your corpus.

The Rust trainers (`FastBpeTrainer`, etc.) take the patterns as `main_regex` /
`script_specific_regex` / `script_specific_scripts` constructor arguments directly, rather than a
`Pretokenizer` object.

### Verification & Benchmarking

```bash
# Run Python vs Rust inference comparison
python examples/test_rust_comparison.py --model-file path/to/model.model --dataset path/to/dataset.jsonl

# Run Rust unit tests
cargo test --no-default-features

# Run Python pretokenizer tests
python -m pytest python/tests/ -v
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

# Add special tokens to the vocabulary
t.add_special_tokens(["<|endoftext|>", "<|pad|>"])

# Writes path/to/new_model.model
t.save("path/to/new_model")
```

The resulting `.model` loads normally in `FastTokenizer` for fast inference afterward. Using
the slower Python `Tokenizer` here is fine: this is a one-time edit, not the inference hot path.

### Exporting to HuggingFace and tiktoken formats

A loaded `Tokenizer` can export to the two standard byte-level BPE formats. Install the extra
dependencies with `pip install boundlessbpe[export]` (adds `tokenizers` and `tiktoken`).

```python
from boundlessbpe import Tokenizer, GPT4O_COARSE_REGEX

t = Tokenizer()
t.load("path/to/word_model.model")

# Word (plain BPE) model — no coarse regex needed:
t.save_huggingface("out_dir")            # writes tokenizer.json + tokenizer_config.json + special_tokens_map.json
t.save_tiktoken("out.tiktoken")          # writes out.tiktoken + out.tiktoken.json sidecar
```

Load the exports back in their native libraries:

```python
# HuggingFace — either loader works on the output directory
from tokenizers import Tokenizer as HFTokenizer
HFTokenizer.from_file("out_dir/tokenizer.json")
# or: from transformers import AutoTokenizer; AutoTokenizer.from_pretrained("out_dir")

# tiktoken — combine the rank file with its .json sidecar (pattern + special tokens)
import json, tiktoken
from tiktoken.load import load_tiktoken_bpe
meta = json.load(open("out.tiktoken.json"))
enc = tiktoken.Encoding(name="bbpe", pat_str=meta["pat_str"],
                        mergeable_ranks=load_tiktoken_bpe("out.tiktoken"),
                        special_tokens=meta["special_tokens"])
```

**Superword models** (BoundlessBPE / SuperBPE) use the "SuperBPE trick": a *coarse*
pretokenization regex that keeps a run of mergeable words as one pretoken, so single-pass BPE
can rebuild the cross-boundary supermerges. You supply the coarse regex (we ship one for the
default). Train the model with `GPT4O_EXPORT_REGEX` and export with `GPT4O_COARSE_REGEX`:

```python
from boundlessbpe import BoundlessBpeTrainer, Pretokenizer, GPT4O_EXPORT_REGEX, GPT4O_COARSE_REGEX

# ... train word + boundless models with Pretokenizer(main_regex=GPT4O_EXPORT_REGEX) ...

t = Tokenizer(); t.load("path/to/boundless_model.model")
t.save_huggingface("out_dir", coarse_regex=GPT4O_COARSE_REGEX)
t.save_tiktoken("out.tiktoken", coarse_regex=GPT4O_COARSE_REGEX)
```

**The exported model is not exactly the same model.** In BoundlessBPE, supermerges are confined to
whole single-token pretokens and no regular merge can cross a pretoken boundary. When the coarse
regex fuses a run of fine pretokens into one pretoken, *neither* limitation holds any more: plain
single-pass BPE running over that fused pretoken can make regular merges across the original
boundaries, and can build superwords out of pretoken *fragments*. So the exported tokenizer will
occasionally make merges the real BoundlessBPE model never would — and these are not graceful:
a merge crossing a boundary can cut through the middle of a multi-byte character, giving an ugly
partial-character token (the bytes still round-trip, but the split is poor). They are rare and
concentrated in no-separator runs — camelCase identifiers or joined non-Latin scripts; ordinary
space-separated text is unaffected. On 1,000,000 minipile documents (vocab 40960) only 24 differed
vs tiktoken and 26 vs HuggingFace (under 1 in 30,000). See the Details below for the mechanism.

**Designing the coarse regex is your responsibility.** It must group exactly the runs of fine
pretokens that *should* be candidates to combine into superwords — no more, no less. We provide
`GPT4O_COARSE_REGEX` as a correct companion to `GPT4O_EXPORT_REGEX`; for any other pretokenization
you must construct your own coarse regex, and getting it wrong will change the tokenization. This
is an advanced feature — know what your fine and coarse regexes do before relying on the export.

Requirements: the model must have **no PickyBPE deletions** (train with `tau >= 1.1`) and **no
script-aware pretokenizer** — neither maps to a single plain-BPE pipeline, and export raises a
clear `ValueError` otherwise. `coarse_regex` is required for superword models and must not be
passed for word models.

**Word models export exactly** (byte-identical token IDs).

**Superword models differ from normal BoundlessBPE by design.** The coarse-regex trick reproduces
supermerges but also makes the fragment/boundary-crossing merges described above, so the export is
*not* a byte-identical stand-in for the `boundlessbpe` tokenizer — the two disagree on a meaningful
fraction of documents (any that contain a run the export merges differently). If you need exact
BoundlessBPE behavior, use the `boundlessbpe` tokenizer directly; the export is for deploying in
standard runtimes where that small tokenization difference is acceptable.

`Tokenizer` can get *close* to the export with `export_compatible=True` (see Details): that mode
makes the same *supermerges* the export does, so the two agree on the vast majority of text — on
1,000,000 minipile documents (vocab 40960) fewer than 1 in 30,000 differed. It is still not
byte-identical to the export, and the residual differences are exactly the cases the export gets
wrong: a regular merge crossing a pretoken boundary, which can split in the middle of a multi-byte
character into an ugly partial-character token. (The decoded text is always correct — the bytes
round-trip — but the segmentation at that spot is not something you'd want.)

Verify an export against your own corpus:

```bash
python examples/verify_tiktoken.py --model-file model.model --dataset data.jsonl [--coarse-regex-default]
python examples/verify_hf.py       --model-file model.model --dataset data.jsonl [--coarse-regex-default]
```

<details>
<summary><b>Details: why superword export isn't always byte-identical</b></summary>

BoundlessBPE lets *supermerges* cross pretoken boundaries but never lets a *regular* merge do so.
The coarse export regex collapses a run of adjacent word-pretokens into one flat pretoken, and
plain single-pass BPE — which has no memory of the original boundaries inside that pretoken — can
then apply a regular merge across one. The decoded text is always identical, but the segmentation
at that spot can be poor — the merge may cut through the middle of a multi-byte character.

This only happens for **no-separator runs** — camelCase identifiers or joined non-Latin scripts,
where the fine regex splits on the case/script change but there is no space at the seam. For
example the Cyrillic camelCase identifier `ФичаДляПроверки` is three pretokens to us
(`Фича|Для|Проверки`) but one coarse pretoken to the export, which then merges bytes across the
`Фича|Для` seam. Ordinary space-separated text never triggers it (the leading space is a byte at
the seam that blocks the cross-word merge).

Measured on 1,000,000 minipile documents (vocab 40960): 24 documents differed vs tiktoken and 26
vs HuggingFace — all boundary-crossing merges, zero corrupted bytes. `examples/benchmark_export.py`
trains, exports, and classifies every difference this way.

**Approximating the export from Python.** Pass `export_compatible=True` to make our tokenizer
group pretokens the way the coarse regex does and apply the export's *supermerges* (no coarse
regex needed — it uses the model's own pretokenizer):

```python
t.encode_ordinary(text, export_compatible=True)                    # export's supermerges, but
t.encode(text, allowed_special="all", export_compatible=True)      # not its cross-boundary regular merges
```

This matches the export on the vast majority of text but is not byte-identical: our regular merges
still respect pretoken boundaries, so the boundary-crossing merges above (the <1-in-30,000 cases)
still differ. Default is `False` (normal BoundlessBPE); available on both `Tokenizer` and
`FastTokenizer`.

**If instead you train a fresh BPE with HuggingFace** (rather than exporting ours), note its
`BpeTrainer` counts *overlapping* adjacent pairs while BoundlessBPE counts only *non-overlapping*
(applicable) merges — e.g. `"   "` gives HF two `(space, space)` pairs but us one. On
whitespace-heavy corpora this can change even the first merge, so an HF-trained model is not
bit-for-bit reproducible against ours. `examples/verify_hf_training.py` measures this. (This does
not affect *exporting* an already-trained model.)

</details>

## Performance

**Inference benchmarks** (1,000,000 documents from minipile):

```text
   Overall speedup:    2.66x
   Total time: Python 3,569s vs Rust 1,341s
   Throughput: Python 1,644,124 chars/sec vs Rust 4,375,271 chars/sec
```

**Correctness**: 100% identical results between Python and Rust implementations over 1,000,000 documents.

## Project Structure

```text
fastboundlessbpe/
├── src/                           # Rust implementation
│   ├── lib.rs                     # PyO3 bindings and module exports
│   ├── tokenizer.rs               # Core BPE inference logic
│   ├── trainer.rs                 # Training engine (BPE, BoundlessBPE, SuperBPE)
│   ├── vocabulary.rs              # Token vocabulary management
│   ├── inference_data.rs          # Merge/deletion operations data
│   ├── pretokenize.rs             # Regex-based text pretokenization
│   ├── script_data.rs             # Unicode script lookup (generated)
│   ├── byte_encoding.rs           # Byte-to-character encoding utilities
│   ├── constants.rs               # Regex patterns and constants
│   └── error.rs                   # Error types
├── python/boundlessbpe/           # Python package
│   ├── inference.py               # Python inference implementation (Tokenizer)
│   ├── train.py                   # Python training (BpeTrainer, BoundlessBpeTrainer, SuperBpeTrainer)
│   ├── export.py                  # HuggingFace / tiktoken export
│   ├── vocabulary.py              # Vocabulary class
│   ├── inferencedata.py           # InferenceData class
│   ├── pretokenize.py             # Pretokenizer class
│   ├── regexconstants.py          # Regex pattern constants
│   ├── allngramcnt.py             # N-gram counting for BoundlessBPE candidates
│   ├── ngram_split.py             # N-gram greedy splitting
│   ├── script_data.py             # Unicode script lookup (generated)
│   └── util.py                    # Encoding and I/O utilities
├── python/tests/                  # Python tests
├── tests/                         # Rust integration tests (test_parity.rs)
├── unicode_data/                  # Unicode data + script-table generator
└── examples/                      # Runnable examples and driver scripts
    ├── train_word.py              # Train first-pass word model
    ├── train_boundless.py         # Train second-pass BoundlessBPE model
    ├── train_super.py             # Train second-pass SuperBPE model
    ├── example_usage.py           # Pretokenizer usage examples
    ├── compare_implementations.py # Verify Rust and Python produce identical results
    ├── test_rust_comparison.py    # Python vs Rust correctness & performance
    ├── test_inference.py          # Inference roundtrip / timing
    ├── benchmark_export.py        # Train, export, and verify tiktoken/HF parity + timing
    ├── verify_tiktoken.py         # Verify a tiktoken export matches our tokenizer
    ├── verify_hf.py               # Verify a HuggingFace export matches our tokenizer
    └── verify_hf_training.py      # Compare an HF-trained BPE to ours (counting differences)
```

## Requirements

- **Python**: 3.10+
- **Rust**: 1.70+ (only needed for building the Rust extension from source)
- **Python dependencies**: `regex`, `heapdict` (installed automatically)
- **Optional (`[export]` extra)**: `tokenizers`, `tiktoken` — only needed for HuggingFace / tiktoken export
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
