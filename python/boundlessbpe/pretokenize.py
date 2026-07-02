#!/usr/bin/env python
# Copyright 2026-present Kensho Technologies, LLC.
"""
Grapheme-aware tokenization for multilingual text processing.

This module provides utilities for tokenizing text while respecting grapheme boundaries
and handling special cases for non-space-delimited Asian scripts.
"""

import regex
from typing import List, Optional, cast

from .script_data import (
    init_script_array, SCRIPT_NAME_TO_INDEX, NUM_SCRIPTS,
    UNKNOWN, COMMON, INHERITED,
)

from .regexconstants import (
    SIMPLE_MERGE_PATTERN,
    DEFAULT_SCRIPT_SPECIFIC_SCRIPTS,
    GPT4O_REGEX,
    SCRIPT_SPECIFIC_GPT4O_REGEX,
)


class Pretokenizer:
    """
    A pretokenizer that handles grapheme clusters and script-aware tokenization.

    Features:
    - Handles Asian scripts (CJK, Thai, Myanmar, Khmer, Lao) character by character
    - Processes other scripts with word-level tokenization
    - Supports contractions, camelCase, numbers, punctuation, and whitespace, emoji, flags, keycaps sequences
    - Breaks runs of space to try to maximize spaces before words and punctuation
    """

    def __init__(self,
                 main_regex: Optional[str] = None,
                 script_specific_regex: Optional[str] = None,
                 script_specific_scripts: Optional[List[str]] = None,
                 merge_pattern: Optional[str] = None):
        """
        Initialize the pretokenizer.

        By default, uses simple mode with pattern matching over entire text.
        To enable script-aware splitting, provide script_specific_regex.

        Args:
            main_regex: Main tokenization pattern. Used for all text in simple mode,
                       or for non-script-specific scripts in script-aware mode.
                       Defaults to GPT4O_REGEX.
            script_specific_regex: If provided, enables script-aware mode. Pattern for
                                  scripts in script_specific_scripts (e.g., CJK).
                                  If None, operates in simple mode.
            script_specific_scripts: Scripts that use script_specific_regex.
                                    Only used in script-aware mode.
                                    Must be provided if script_specific_regex is not None
            merge_pattern: Pattern for merge eligibility.
                          Defaults to SIMPLE_MERGE_PATTERN.
        """

        # Compile patterns (used in both modes)
        self.main_pattern = regex.compile(
            main_regex if main_regex is not None else GPT4O_REGEX
        )

        # Determine mode based on script_specific_regex
        if script_specific_regex is None:
            # SIMPLE MODE - no script splitting
            self.script_aware_mode = False
            # Placeholder for simple mode - None when script_aware_mode is False
            # This allows inferencedata._write_to_file to check script_aware_mode first
            self.script_specific_pattern: Optional[regex.Pattern[str]] = None
            if script_specific_scripts is not None:
                raise ValueError(
                    "script_specific_scripts provided but script_specific_regex is None. "
                    "To use script-aware mode, provide script_specific_regex."
                )
        else:
            # SCRIPT-AWARE MODE
            self.script_aware_mode = True

            if script_specific_scripts is None:
                raise ValueError(
                    "script_specific_scripts must be provided when script_specific_regex is not None."
                )

            # Build bool lookup table: is_script_specific[script_idx] = 1 if script-specific
            self.script_specific_scripts_set = set(script_specific_scripts)
            self.is_script_specific = bytearray(NUM_SCRIPTS)
            for name in script_specific_scripts:
                self.is_script_specific[SCRIPT_NAME_TO_INDEX[name]] = 1

            # Build script array for fast script detection
            self._setup_script_array()
            # Compile script-specific pattern (only needed in script-aware mode)
            self.script_specific_pattern = regex.compile(script_specific_regex)

        self.gpt4o_pattern = regex.compile(GPT4O_REGEX)

        merge_pattern_str = merge_pattern if merge_pattern is not None else SIMPLE_MERGE_PATTERN
        self.merge_pattern = regex.compile(merge_pattern_str)

    def _setup_script_array(self) -> None:
        """Build script lookup array for fast script detection."""
        self.script_array = init_script_array()

    def can_merge(self, left: bytes, right: bytes) -> bool:
        """
        Check if two byte tokens can be merged (UTF-8 decode version).

        Args:
            left: Left byte token
            right: Right byte token

        Returns:
            True if both tokens match the merge pattern after UTF-8 decoding
        """
        # decode both sides; if decoding fails, we treat it as "cannot merge"
        try:
            left_str = left.decode('utf-8')
            right_str = right.decode('utf-8')
        except UnicodeDecodeError:
            return False

        return bool(self.merge_pattern.search(left_str)) and bool(self.merge_pattern.search(right_str))

    def could_merge(self, tok: bytes) -> bool:
        """
        Check if a byte token could be involved in a merge (UTF-8 decode version).

        Args:
            tok: Byte token to check

        Returns:
            True if token matches the merge pattern after UTF-8 decoding
        """
        try:
            tok_str = tok.decode('utf-8')
        except UnicodeDecodeError:
            return False

        return bool(self.merge_pattern.search(tok_str))

    def pretokenize(self, text: str) -> List[str]:
        """
        Pretokenize text.

        If in simple mode: Applies main_pattern to entire text.
        If in script-aware mode: Splits by Unicode script and applies 
                                 script_specific_pattern to scripts in asian_scripts_set
                                 and main_pattern to all other scripts

        Args:
            text: Input text to split

        Returns:
            List of pretokens
        """
        if not text:
            return []

        # Simple mode - just apply main_pattern to entire text
        if not self.script_aware_mode:
            return cast(List[str], self.main_pattern.findall(text))

        # Script-aware mode - split by script boundaries
        assert self.script_specific_pattern is not None  # guaranteed in script_aware_mode
        tokens: List[str] = []
        run_start = 0
        current_script = -1  # -1 = no script yet
        last_non_common_index = -1
        script_array = self.script_array
        is_script_specific = self.is_script_specific

        for i, char in enumerate(text):
            char_script = script_array[ord(char)]

            # Same script continues
            if char_script == current_script:
                last_non_common_index = i
                continue

            # Inherited stays with current script
            if char_script == INHERITED:
                if current_script == -1:
                    current_script = INHERITED
                last_non_common_index = i
                continue

            # Common or Unknown - skip if no real script yet
            if char_script == COMMON or char_script == UNKNOWN:
                if current_script == -1:
                    current_script = COMMON
                # Don't update last_non_common_index - allows trailing Common/Unknown to move
                continue

            # New real script starts a new run
            if current_script != -1:
                # Only process if we had a real script (not pure Common/Inherited leading)
                if current_script != COMMON and current_script != INHERITED:
                    chunk_end = last_non_common_index + 1

                    if is_script_specific[current_script]:
                        tokens.extend(self.script_specific_pattern.findall(text[run_start:chunk_end]))
                    else:
                        tokens.extend(self.main_pattern.findall(text[run_start:chunk_end]))

                    run_start = chunk_end  # Start next chunk after the previous one
                # else: keep run_start where it is - Common/Inherited will attach to new script

            current_script = char_script
            last_non_common_index = i

        # Last run
        if current_script != -1:
            if current_script == COMMON or current_script == INHERITED:
                # Pure Common/Inherited at end - process it
                tokens.extend(self.main_pattern.findall(text[run_start:]))
            elif is_script_specific[current_script]:
                tokens.extend(self.script_specific_pattern.findall(text[run_start:]))
            else:
                tokens.extend(self.main_pattern.findall(text[run_start:]))

        return tokens

    def pretokenize_gpt4o_style(self, text: str) -> List[str]:
        """
        Pretokenize using GPT-4o style pattern for comparison.

        Args:
            text: Input text to pretokenize

        Returns:
            List of pretokens using GPT-4o pattern
        """
        result: List[str] = self.gpt4o_pattern.findall(text)
        return result


# Convenience functions
def pretokenize(text: str) -> List[str]:
    """
    Pretokenize text using simple mode (word-level tokenization, no script splitting).

    For processing multiple texts, create a single Pretokenizer instance
    and reuse it for better performance.

    Args:
        text: Input text to pretokenize

    Returns:
        List of pretokens
    """
    pretokenizer = Pretokenizer()
    return pretokenizer.pretokenize(text)


def pretokenize_script_aware(text: str) -> List[str]:
    """
    Pretokenize text using script-aware mode with character-level splitting for CJK scripts.

    Warning: This function creates a new Pretokenizer instance on each call,
    which includes building a Unicode script lookup array (~0.12 seconds).
    For processing multiple texts, create a single Pretokenizer instance
    and reuse it for better performance.

    Args:
        text: Input text to pretokenize

    Returns:
        List of pretokens
    """
    pretokenizer = Pretokenizer(
        script_specific_regex=SCRIPT_SPECIFIC_GPT4O_REGEX,
        script_specific_scripts=DEFAULT_SCRIPT_SPECIFIC_SCRIPTS
    )
    return pretokenizer.pretokenize(text)

