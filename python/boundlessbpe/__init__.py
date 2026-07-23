# Copyright 2026-present Kensho Technologies, LLC.
"""
BoundlessBPE - Fast Rust implementation of BPE tokenizer with Python fallback

This package provides both a fast Rust implementation and the original Python
implementation for comparison and testing.
"""

# Import Python implementation
from .inference import Tokenizer
from .train import BpeTrainer, SuperBpeTrainer, BoundlessBpeTrainer
from .pretokenize import Pretokenizer
from .vocabulary import Vocabulary
from .regexconstants import *

# Try to import Rust implementation
try:
    from .boundlessbpe import FastTokenizer
    from .boundlessbpe import FastBpeTrainer, FastBoundlessBpeTrainer, FastSuperBpeTrainer
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    FastTokenizer = None
    FastBpeTrainer = None
    FastBoundlessBpeTrainer = None
    FastSuperBpeTrainer = None

# Export main classes
__all__ = [
    'Tokenizer',
    'Pretokenizer',
    'Vocabulary',
    'BpeTrainer',
    'SuperBpeTrainer',
    'BoundlessBpeTrainer',
    'FastTokenizer',
    'FastBpeTrainer',
    'FastBoundlessBpeTrainer',
    'FastSuperBpeTrainer',
    'RUST_AVAILABLE',
    # Re-export constants
    'GPT2_REGEX',
    'GPT4_REGEX',
    'GPT4O_REGEX',
    'GPT4O_SPLIT_PATTERN',
    'ULTIMATE_PATTERN',
    'IMPROVED_MERGE_PATTERN',
    'SIMPLE_MERGE_PATTERN',
    'SCRIPT_SPECIFIC_REGEX',
    'SCRIPT_SPECIFIC_GPT4O_REGEX',
    'DEFAULT_SCRIPT_SPECIFIC_SCRIPTS',
    'WORD_LEVEL_REGEX',
    'GPT4O_EXPORT_REGEX',
    'GPT4O_COARSE_REGEX',
]

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("boundlessbpe")
except PackageNotFoundError:  # not installed (e.g. imported from source tree)
    __version__ = "0.0.0+unknown"