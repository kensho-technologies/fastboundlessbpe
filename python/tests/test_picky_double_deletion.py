# Copyright 2026-present Kensho Technologies, LLC.
"""Regression test for the PickyBPE double-deletion bug.

Reproduction case contributed by Adam Wiemerslage.

When a token is deleted, recreated by a later merge, and its replacement parts
reference another token that was itself deleted, encoding could strand a token
that is not in the final vocabulary and raise KeyError at ID lookup. See
InferenceData.resolve_deletion_parts for the full explanation and fix.
"""

import glob
import json
import os
import tempfile

import pytest

from boundlessbpe.train import BpeTrainer
from boundlessbpe.inference import Tokenizer

try:
    from boundlessbpe import FastTokenizer, RUST_AVAILABLE
except ImportError:  # pragma: no cover
    FastTokenizer = None
    RUST_AVAILABLE = False


# Corpus that provokes overlapping deletions of cr / cry / crys / cryst
# (tau=0.6, blowup=False), so cry's replacement parts reference the
# already-deleted token cr.
_TEXTS = (
    ["crystal crystal crystal crystal crystal crystal crystal"] * 800
    + ["crystallize crystallize crystallize crystallize"] * 400
    + ["crystallization crystallization crystallization"] * 300
    + ["crystals crystals crystals crystals crystals"] * 400
    + ["crystalline crystalline crystalline crystalline"] * 300
    + ["the and is of to in a for on with"] * 500
)

# Inputs that exercise the deleted tokens directly.
_ENCODE_CASES = [
    "crycry",
    "crystalcrystal",
    "crystal crystal",
    "crystallization",
    "crystalline stuff",
    "cry",
    "cr",
    "ccrryy",
]


@pytest.fixture(scope="module")
def picky_model_file():
    """Train the tiny double-deletion model once and yield its path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl = os.path.join(tmpdir, "train.jsonl")
        with open(jsonl, "w") as f:
            for t in _TEXTS:
                f.write(json.dumps({"text": t}) + "\n")

        BpeTrainer().train(
            tau=0.6,
            filepath=jsonl,
            outprefix=os.path.join(tmpdir, "m"),
            num_lines=len(_TEXTS),
            vocab_size=2000,
            recalc=0,
            blowup=False,
            checkpoint_iterations=0,
            verbose=False,
        )
        model_file = sorted(glob.glob(os.path.join(tmpdir, "m_*.model")))[-1]
        yield model_file


def test_python_encodes_without_keyerror(picky_model_file):
    """Encoding must not raise and must round-trip through decode."""
    tok = Tokenizer()
    tok.load(picky_model_file)
    for text in _ENCODE_CASES:
        ids = tok.encode_ordinary(text)  # must not raise KeyError
        assert tok.decode(ids) == text, f"round-trip failed for {text!r}"


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not built")
def test_rust_matches_python(picky_model_file):
    """FastTokenizer must agree with the Python Tokenizer on every case."""
    py = Tokenizer()
    py.load(picky_model_file)
    fast = FastTokenizer()
    fast.load(picky_model_file)
    for text in _ENCODE_CASES:
        expected = py.encode_ordinary(text)
        assert fast.encode_ordinary(text) == expected, f"parity failed for {text!r}"
