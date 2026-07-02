# Copyright 2026-present Kensho Technologies, LLC.
"""
Unified vocabulary management for BoundlessBPE tokenizers.

This module provides the Vocabulary class which handles token-to-index mappings,
special tokens, and serialization. Used by both training (incrementally built)
and inference (loaded from file).
"""

from typing import Optional, TextIO

from .util import create_initial_vocab, frombytes, tobytes


class Vocabulary:
    """
    Unified vocabulary for training and inference.

    Manages an ordered list of tokens with bidirectional mappings between
    tokens (bytes) and indices (int). Supports add/delete operations during
    training and serialization for persistence.
    """

    def __init__(self) -> None:
        """Initialize an empty vocabulary."""
        self.tokens: list[bytes] = []           # ordered token list
        self.token_to_id: dict[bytes, int] = {} # token -> index
        self.id_to_token: dict[int, bytes] = {} # index -> token
        self.is_super: dict[bytes, bool] = {}   # token -> is super-level
        self.special_tokens: dict[str, int] = {}
        self.inverse_special_tokens: dict[int, str] = {}
        self.final_token_counts: dict[bytes, int] = {}  # loaded for analysis

    @classmethod
    def create_initial(cls) -> 'Vocabulary':
        """
        Create vocabulary with 243 valid UTF-8 single bytes.

        Used for first-pass BPE and BoundlessBPE training.

        Returns:
            Vocabulary initialized with valid single-byte tokens.
        """
        vocab = cls()
        for tok in create_initial_vocab():
            vocab.add(tok, is_super=False)
        return vocab

    @classmethod
    def create_from_vocab(cls, source_vocab: 'Vocabulary') -> 'Vocabulary':
        """
        Copy all tokens from source vocabulary.

        Used for SuperBPE second pass which starts with the complete
        word model vocabulary.

        Args:
            source_vocab: Vocabulary to copy from.

        Returns:
            New Vocabulary with all tokens from source.
        """
        vocab = cls()
        for tok in source_vocab.tokens:
            vocab.add(tok, is_super=source_vocab.is_super[tok])
        # Copy final_token_counts if present
        vocab.final_token_counts = dict(source_vocab.final_token_counts)
        return vocab

    def add(self, token: bytes, is_super: bool) -> int:
        """
        Add a token to the vocabulary.

        If the token already exists, returns the existing index without
        adding a duplicate.

        Args:
            token: The token to add.
            is_super: Whether this is a super-level token (vs word-level).

        Returns:
            The index assigned to the token.
        """
        if token in self.token_to_id:
            return self.token_to_id[token]

        idx = len(self.tokens)
        self.tokens.append(token)
        self.token_to_id[token] = idx
        self.id_to_token[idx] = token
        self.is_super[token] = is_super
        return idx

    def delete(self, token: bytes) -> None:
        """
        Remove a token from the vocabulary and re-index subsequent tokens.

        This is O(n) where n is the number of tokens after the deleted one,
        but deletions are rare in practice (only PickyBPE uses them).

        Args:
            token: The token to remove.
        """
        if token not in self.token_to_id:
            return

        idx = self.token_to_id[token]

        # Remove from list
        self.tokens.pop(idx)

        # Remove from forward mapping
        del self.token_to_id[token]

        # Remove from inverse mapping
        del self.id_to_token[idx]

        # Remove from is_super mapping
        del self.is_super[token]

        # Rebuild mappings for tokens after deleted index
        for i in range(idx, len(self.tokens)):
            tok = self.tokens[i]
            self.token_to_id[tok] = i
            self.id_to_token[i] = tok

        # Remove old highest index from inverse mapping (now invalid)
        old_max_idx = len(self.tokens)  # This was the old last index
        if old_max_idx in self.id_to_token:
            del self.id_to_token[old_max_idx]

    def __contains__(self, token: bytes) -> bool:
        """Check if a token is in the vocabulary."""
        return token in self.token_to_id

    def __len__(self) -> int:
        """Return the number of tokens in the vocabulary."""
        return len(self.tokens)

    def __getitem__(self, token: bytes) -> int:
        """Get the index of a token."""
        return self.token_to_id[token]

    def get_token(self, idx: int) -> bytes:
        """
        Get the token at a given index.

        Args:
            idx: The token index.

        Returns:
            The token bytes at that index.

        Raises:
            KeyError: If the index is not in the vocabulary.
        """
        return self.id_to_token[idx]

    def vocab_size(self) -> int:
        """Return the number of tokens, excluding special tokens."""
        return len(self.tokens)

    def register_special_tokens(self, tokens: list[str]) -> None:
        """
        Register special tokens indexed after regular vocabulary.

        Special tokens are stored separately from regular tokens and
        their indices start at len(self.tokens).

        Args:
            tokens: List of special token strings to register.
        """
        start_idx = len(self.tokens)
        for tok in tokens:
            if tok not in self.special_tokens:
                idx = start_idx + len(self.special_tokens)
                self.special_tokens[tok] = idx
                self.inverse_special_tokens[idx] = tok

    def save(self, f: TextIO, token_counts: dict[bytes, int], max_size: Optional[int] = None) -> None:
        """
        Save vocabulary section to file.

        Writes the vocabulary and special_tokens sections.

        Args:
            f: File handle to write to.
            token_counts: Dict mapping tokens to their counts (from trainer's single_counts).
            max_size: If provided, only save first max_size tokens.
                     Used by BoundlessBPE to trim the vocabulary.
        """
        # Determine how many tokens to write
        tokens_to_write = self.tokens
        if max_size is not None:
            tokens_to_write = self.tokens[:max_size]

        # Write vocabulary section
        f.write("vocabulary\n")
        f.write(f"{len(tokens_to_write)}\n")
        for idx, tok in enumerate(tokens_to_write):
            count = token_counts.get(tok, 0)
            is_super_flag = 1 if self.is_super[tok] else 0
            f.write(f"{idx} {frombytes(tok)} {count} {is_super_flag}\n")

        # Write special_tokens section
        f.write("special_tokens\n")
        f.write(f"{len(self.special_tokens)}\n")
        # Sort by index to ensure consistent ordering
        sorted_special = sorted(self.special_tokens.items(), key=lambda x: x[1])
        for tok_str, idx in sorted_special:
            # Recompute index based on trimmed vocab size if needed
            if max_size is not None:
                adjusted_idx = max_size + (idx - len(self.tokens))
            else:
                adjusted_idx = idx
            f.write(f"{adjusted_idx} {tok_str}\n")

    @classmethod
    def load(cls, f: TextIO) -> 'Vocabulary':
        """
        Load vocabulary from file.

        Reads the vocabulary and special_tokens sections.

        Args:
            f: File handle to read from.

        Returns:
            Loaded Vocabulary instance.
        """
        vocab = cls()

        # Read vocabulary section header
        line = f.readline().strip()
        assert line == "vocabulary", f"Expected 'vocabulary', got '{line}'"

        # Read token count
        count = int(f.readline().strip())

        # Read tokens
        for _ in range(count):
            parts = f.readline().strip().split(" ")
            assert len(parts) == 4, f"Expected 4 parts, got {len(parts)}: {parts}"
            idx_str, tok_str, count_str, is_super_str = parts
            idx = int(idx_str)
            tok = tobytes(tok_str)
            tok_count = int(count_str)
            is_super = (int(is_super_str) == 1)

            # Add token (should match expected index)
            added_idx = vocab.add(tok, is_super=is_super)
            assert added_idx == idx, f"Index mismatch: expected {idx}, got {added_idx}"

            if tok_count > 0:
                vocab.final_token_counts[tok] = tok_count

        # Read special_tokens section header
        line = f.readline().strip()
        assert line == "special_tokens", f"Expected 'special_tokens', got '{line}'"

        # Read special token count
        special_count = int(f.readline().strip())

        # Read special tokens
        for _ in range(special_count):
            parts = f.readline().strip().split(" ", 1)
            assert len(parts) == 2, f"Expected 2 parts for special token, got {len(parts)}: {parts}"
            idx_str, tok_str = parts
            idx = int(idx_str)
            vocab.special_tokens[tok_str] = idx
            vocab.inverse_special_tokens[idx] = tok_str

        return vocab

    def get_all_token_ids(self) -> set[int]:
        """
        Get set of all valid token IDs (regular + special).

        Returns:
            Set of all valid token indices.
        """
        ids = set(self.id_to_token.keys())
        ids.update(self.special_tokens.values())
        return ids

    def total_size(self) -> int:
        """
        Get total vocabulary size including special tokens.

        Returns:
            Number of regular tokens plus number of special tokens.
        """
        return len(self.tokens) + len(self.special_tokens)

    def get_word_tokens(self) -> list[bytes]:
        """
        Get only word-level tokens (not super-level).

        Returns:
            List of tokens where is_super is False.
        """
        return [t for t in self.tokens if not self.is_super[t]]

    def get_super_tokens(self) -> list[bytes]:
        """
        Get only super-level tokens.

        Returns:
            List of tokens where is_super is True.
        """
        return [t for t in self.tokens if self.is_super[t]]

    def verify_vocabulary(self) -> None:
        """
        Verify internal consistency between all vocabulary fields.

        Raises:
            AssertionError: If any inconsistency is found.
        """
        # Check sizes match - if sizes match and all entries validate, no extras possible
        assert len(self.tokens) == len(self.token_to_id), \
            f"tokens length {len(self.tokens)} != token_to_id length {len(self.token_to_id)}"
        assert len(self.tokens) == len(self.id_to_token), \
            f"tokens length {len(self.tokens)} != id_to_token length {len(self.id_to_token)}"
        assert len(self.tokens) == len(self.is_super), \
            f"tokens length {len(self.tokens)} != is_super length {len(self.is_super)}"

        # Check each token's mappings are consistent
        for idx, token in enumerate(self.tokens):
            assert token in self.token_to_id, \
                f"token {token!r} at index {idx} not in token_to_id"
            assert self.token_to_id[token] == idx, \
                f"token_to_id[{token!r}] = {self.token_to_id[token]}, expected {idx}"
            assert idx in self.id_to_token, \
                f"index {idx} not in id_to_token"
            assert self.id_to_token[idx] == token, \
                f"id_to_token[{idx}] = {self.id_to_token[idx]!r}, expected {token!r}"
            assert token in self.is_super, \
                f"token {token!r} at index {idx} not in is_super"

        # Check special tokens consistency
        assert len(self.special_tokens) == len(self.inverse_special_tokens), \
            f"special_tokens length {len(self.special_tokens)} != inverse_special_tokens length {len(self.inverse_special_tokens)}"
        for tok_str, idx in self.special_tokens.items():
            assert idx >= len(self.tokens), \
                f"special token {tok_str} has index {idx} < vocab size {len(self.tokens)}"
            assert idx in self.inverse_special_tokens, \
                f"special token index {idx} not in inverse_special_tokens"
            assert self.inverse_special_tokens[idx] == tok_str, \
                f"inverse_special_tokens[{idx}] = {self.inverse_special_tokens[idx]}, expected {tok_str}"
