# Copyright 2026-present Kensho Technologies, LLC.
"""Tests for HuggingFace and tiktoken export.

Trains tiny word and boundless models, exports them, and checks:
- word models export byte-identically to both formats;
- boundless models export best-effort: the ONLY differences are supermerge choices
  (word-level tokenization identical, bytes never corrupted);
- error paths (deletions, script-aware, coarse_regex misuse) raise clearly.
"""
import contextlib
import glob
import io
import json
import os
import tempfile

import pytest

from boundlessbpe import GPT4O_EXPORT_REGEX, GPT4O_COARSE_REGEX
from boundlessbpe.train import BpeTrainer, BoundlessBpeTrainer
from boundlessbpe.inference import Tokenizer
from boundlessbpe.export import (
    build_export_view,
    classify_export_difference,
    _superword_expansion_map,
)

tiktoken = pytest.importorskip("tiktoken", reason="tiktoken not installed")
tokenizers = pytest.importorskip("tokenizers", reason="tokenizers not installed")
from tiktoken.load import load_tiktoken_bpe  # noqa: E402
from tokenizers import Tokenizer as HFTokenizer  # noqa: E402
from boundlessbpe.util import tobytes  # noqa: E402


_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In the United States of America, one of the best things is freedom.",
    "She said she isn't sure whether we're going to make it, y'all.",
    "def foo(x): return x + 1  # a comment",
    "Numbers like 123 and 4567 and dates 2024-01-02.",
    "café résumé naïve — em dashes and accents.",
    "Multiple    spaces\tand\ttabs\nand newlines here.",
] * 40  # 280 docs


def _quiet(fn):
    with contextlib.redirect_stdout(io.StringIO()):
        fn()


SPECIAL_TOKENS = ["<|endoftext|>", "<|pad|>"]


@pytest.fixture(scope="module")
def models():
    """Train a tiny word model and a boundless model with the export regex.

    The temp dir is kept for the module's lifetime (not a context manager) so the
    boundless .model path can be reloaded by FastTokenizer for a Python/Rust check;
    the path is stashed on the boundless tokenizer as ._model_path.
    """
    d = tempfile.mkdtemp()
    try:
        jsonl = os.path.join(d, "train.jsonl")
        with open(jsonl, "w") as f:
            for t in _TEXTS:
                f.write(json.dumps({"text": t}) + "\n")

        pre_kwargs = dict(main_regex=GPT4O_EXPORT_REGEX)
        _quiet(lambda: BpeTrainer(**_pretok(pre_kwargs)).train(
            tau=1.1, filepath=jsonl, outprefix=os.path.join(d, "w"),
            num_lines=len(_TEXTS), vocab_size=800, recalc=0, blowup=False,
            checkpoint_iterations=0, verbose=False))
        wm = sorted(glob.glob(os.path.join(d, "w_*.model")),
                    key=lambda p: int(p.split("_")[-1].split(".")[0]))[-1]

        _quiet(lambda: BoundlessBpeTrainer(**_pretok(pre_kwargs)).train(
            filepath=jsonl, outprefix=os.path.join(d, "b"),
            num_lines=len(_TEXTS), recalc=0, word_model_file=wm,
            checkpoint_iterations=0, verbose=False))
        bm = sorted(glob.glob(os.path.join(d, "b_*.model")),
                    key=lambda p: int(p.split("_")[-1].split(".")[0]))[-1]

        word, bnd = Tokenizer(), Tokenizer()
        _quiet(lambda: word.load(wm))
        _quiet(lambda: bnd.load(bm))
        # Register special tokens on both so export paths are exercised with them.
        # (encode_ordinary ignores specials and regular-token ids are unchanged.)
        word.add_special_tokens(SPECIAL_TOKENS)
        bnd.add_special_tokens(SPECIAL_TOKENS)
        bnd._model_path = bm  # for the Python/Rust export_compatible check
        yield word, bnd
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


def _pretok(kwargs):
    """BpeTrainer/BoundlessBpeTrainer take a Pretokenizer; build one with our regex."""
    from boundlessbpe.pretokenize import Pretokenizer
    return {"pretokenizer": Pretokenizer(main_regex=kwargs["main_regex"])}


def _tik_tokens(enc, text):
    return [enc.decode_single_token_bytes(i) for i in enc.encode_ordinary(text)]


def _hf_tokens(hf, text):
    return [tobytes(hf.id_to_token(i)) for i in hf.encode(text).ids]


def _load_tiktoken(tok, coarse):
    d = tempfile.mkdtemp()
    p = os.path.join(d, "m.tiktoken")
    tok.save_tiktoken(p, coarse_regex=coarse)
    ranks = load_tiktoken_bpe(p)
    meta = json.load(open(p + ".json"))
    return tiktoken.Encoding(name="t", pat_str=meta["pat_str"],
                             mergeable_ranks=ranks, special_tokens=meta["special_tokens"])


def _load_hf(tok, coarse):
    d = tempfile.mkdtemp()
    tok.save_huggingface(d, coarse_regex=coarse)
    return HFTokenizer.from_file(os.path.join(d, "tokenizer.json"))


# --- word model: must be byte-identical -----------------------------------

def test_word_tiktoken_identical(models):
    word, _ = models
    enc = _load_tiktoken(word, None)
    for t in _TEXTS[:7]:
        ours = [word.vocab.id_to_token[i] for i in word.encode_ordinary(t)]
        assert _tik_tokens(enc, t) == ours, f"tiktoken word mismatch: {t!r}"


def test_word_hf_identical(models):
    word, _ = models
    hf = _load_hf(word, None)
    for t in _TEXTS[:7]:
        ours = [word.vocab.id_to_token[i] for i in word.encode_ordinary(t)]
        assert _hf_tokens(hf, t) == ours, f"HF word mismatch: {t!r}"


# --- boundless model: best effort, only supermerge differences ------------

@pytest.mark.parametrize("fmt", ["tiktoken", "hf"])
def test_boundless_only_supermerge_differences(models, fmt):
    _, bnd = models
    parent = _superword_expansion_map(bnd)
    if fmt == "tiktoken":
        enc = _load_tiktoken(bnd, GPT4O_COARSE_REGEX)
        toks = lambda t: _tik_tokens(enc, t)
    else:
        hf = _load_hf(bnd, GPT4O_COARSE_REGEX)
        toks = lambda t: _hf_tokens(hf, t)
    for t in _TEXTS[:7]:
        ours = [bnd.vocab.id_to_token[i] for i in bnd.encode_ordinary(t)]
        kind = classify_export_difference(bnd, ours, toks(t), parent)
        # Never corruption, never a word-level (non-supermerge) divergence.
        assert kind in ("identical", "supermerge"), f"{fmt} {kind} on {t!r}"


def test_export_compatible_matches_exports(models):
    """export_compatible=True reproduces the tiktoken AND HuggingFace exports exactly
    (byte-identical), i.e. it *is* the plain-BPE tokenization the export produces."""
    _, bnd = models
    enc = _load_tiktoken(bnd, GPT4O_COARSE_REGEX)
    hf = _load_hf(bnd, GPT4O_COARSE_REGEX)
    for t in _TEXTS[:7]:
        ours = [bnd.vocab.id_to_token[i] for i in bnd.encode_ordinary(t, export_compatible=True)]
        assert _tik_tokens(enc, t) == ours, f"tiktoken != export_compatible: {t!r}"
        assert _hf_tokens(hf, t) == ours, f"HF != export_compatible: {t!r}"


def test_export_compatible_python_matches_rust(models):
    """Python and Rust agree in export_compatible mode (shared vocab id space)."""
    from boundlessbpe import FastTokenizer, RUST_AVAILABLE
    if not RUST_AVAILABLE:
        pytest.skip("Rust extension not built")
    _, bnd = models
    # bnd was trained/loaded in a temp dir; reload via the same .model is not retained,
    # so exercise Rust on the same in-memory vocab by re-encoding through FastTokenizer
    # loaded from the fixture's model file if available; otherwise compare is skipped.
    fast_path = getattr(bnd, "_model_path", None)
    if fast_path is None:
        pytest.skip("model path not retained on fixture")
    fast = FastTokenizer()
    fast.load(fast_path)
    for t in _TEXTS[:7]:
        py = [bnd.vocab.id_to_token[i] for i in bnd.encode_ordinary(t, export_compatible=True)]
        rust = [bnd.vocab.id_to_token[i] for i in fast.encode_ordinary(t, export_compatible=True)]
        assert py == rust, f"Python != Rust (export_compatible): {t!r}"


# --- special tokens -------------------------------------------------------

def test_special_tokens_in_export_view(models):
    """Exported special-token ids sit above the (renumbered) regular token block."""
    word, bnd = models
    for tok, coarse in ((word, None), (bnd, GPT4O_COARSE_REGEX)):
        view = build_export_view(tok, coarse)
        n = len(view.tokens)
        # Contiguous ids immediately above the regular block, in registration order.
        assert view.special_tokens == {s: n + i for i, s in enumerate(SPECIAL_TOKENS)}


def test_special_tokens_tiktoken(models):
    """tiktoken export encodes specials to the same ids we do, atomically."""
    word, _ = models
    enc = _load_tiktoken(word, None)
    text = "hello <|endoftext|> world <|pad|>"
    ours = word.encode(text, allowed_special="all")
    assert enc.encode(text, allowed_special="all") == ours
    # Each special is a single token with our id.
    for s in SPECIAL_TOKENS:
        assert enc.encode_single_token(s) == word.vocab.special_tokens[s]


def test_special_tokens_hf(models):
    """HF export marks specials atomic (special=True) with matching ids."""
    import json as _json
    word, _ = models
    with tempfile.TemporaryDirectory() as d:
        word.save_huggingface(d)
        files = set(os.listdir(d))
        assert {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"} <= files
        hf = HFTokenizer.from_file(os.path.join(d, "tokenizer.json"))
        for s in SPECIAL_TOKENS:
            assert hf.token_to_id(s) == word.vocab.special_tokens[s]
        # Specials are recognized as atomic single tokens, not split.
        ids = hf.encode("x <|endoftext|> y").ids
        assert word.vocab.special_tokens["<|endoftext|>"] in ids
        # tokenizer_config marks them special in added_tokens_decoder.
        cfg = _json.load(open(os.path.join(d, "tokenizer_config.json")))
        dec = cfg["added_tokens_decoder"]
        for s in SPECIAL_TOKENS:
            entry = dec[str(word.vocab.special_tokens[s])]
            assert entry["content"] == s and entry["special"] is True


# --- error paths ----------------------------------------------------------

def test_error_coarse_on_word_model(models):
    word, _ = models
    with pytest.raises(ValueError, match="coarse_regex must not be passed"):
        build_export_view(word, coarse_regex=GPT4O_COARSE_REGEX)


def test_error_missing_coarse_on_superword(models):
    _, bnd = models
    with pytest.raises(ValueError, match="coarse_regex is required"):
        build_export_view(bnd)


def test_error_deletions_model():
    """A model trained with PickyBPE deletions (tau < 1.1) cannot be exported."""
    from boundlessbpe.pretokenize import Pretokenizer
    with tempfile.TemporaryDirectory() as d:
        jsonl = os.path.join(d, "t.jsonl")
        with open(jsonl, "w") as f:
            for _ in range(400):
                f.write(json.dumps({"text": "crystal crystallize crystals crystalline"}) + "\n")
        _quiet(lambda: BpeTrainer(Pretokenizer(main_regex=GPT4O_EXPORT_REGEX)).train(
            tau=0.6, filepath=jsonl, outprefix=os.path.join(d, "m"),
            num_lines=400, vocab_size=350, recalc=0, blowup=False,
            checkpoint_iterations=0, verbose=False))
        m = sorted(glob.glob(os.path.join(d, "m_*.model")),
                   key=lambda p: int(p.split("_")[-1].split(".")[0]))[-1]
        tok = Tokenizer()
        _quiet(lambda: tok.load(m))
    with pytest.raises(ValueError, match="PickyBPE deletions"):
        build_export_view(tok)


def test_error_script_aware_model():
    """A model using the script-aware pretokenizer cannot be exported."""
    from boundlessbpe.pretokenize import Pretokenizer
    from boundlessbpe import SCRIPT_SPECIFIC_REGEX, DEFAULT_SCRIPT_SPECIFIC_SCRIPTS
    with tempfile.TemporaryDirectory() as d:
        jsonl = os.path.join(d, "t.jsonl")
        with open(jsonl, "w") as f:
            for _ in range(400):
                f.write(json.dumps({"text": "the quick brown fox and friends"}) + "\n")
        pretok = Pretokenizer(
            main_regex=GPT4O_EXPORT_REGEX,
            script_specific_regex=SCRIPT_SPECIFIC_REGEX,
            script_specific_scripts=DEFAULT_SCRIPT_SPECIFIC_SCRIPTS,
        )
        _quiet(lambda: BpeTrainer(pretok).train(
            tau=1.1, filepath=jsonl, outprefix=os.path.join(d, "m"),
            num_lines=400, vocab_size=350, recalc=0, blowup=False,
            checkpoint_iterations=0, verbose=False))
        m = sorted(glob.glob(os.path.join(d, "m_*.model")),
                   key=lambda p: int(p.split("_")[-1].split(".")[0]))[-1]
        tok = Tokenizer()
        _quiet(lambda: tok.load(m))
    with pytest.raises(ValueError, match="script-aware"):
        build_export_view(tok)
