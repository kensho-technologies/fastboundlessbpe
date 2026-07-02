# Copyright 2026-present Kensho Technologies, LLC.
"""
Data and methods needed for tokenization inference.

This module contains the InferenceData class which encapsulates merge/deletion
operations and configuration. Vocabulary is now handled separately by the
Vocabulary class in vocabulary.py.
"""

from typing import Optional, TextIO
import json

from .util import frombytes, frombytespair, _write_sorted_dict_intkey, _read_sorted_dict_intkey, _write_sorted_dict_intkey_with_counts, _read_sorted_dict_intkey_with_counts
from .pretokenize import Pretokenizer


class InferenceData:
    """
    Data and methods needed for tokenization inference.

    This class encapsulates merge/deletion operations and configuration.
    Vocabulary is handled separately by the Vocabulary class.
    """

    def __init__(
        self,
        tau: float,
        is_super: bool,
        superbpe_mode: bool,
        blowup: bool,
        pretokenizer: Pretokenizer,
        merges: dict[int, tuple[tuple[bytes, bytes], int, int]],
        deletions: dict[int, bytes],
        merges_lookup: dict[tuple[bytes, bytes], list[int]],
        deletions_lookup: dict[bytes, list[int]],
        deletion_parts: dict[bytes, list[bytes]],
    ) -> None:
        """
        Initialize InferenceData with all required fields.
        Use InferenceData.load() to construct from a file.
        """
        # Merge and deletion rules
        self.merges = merges
        self.deletions = deletions

        # Configuration parameters
        self.tau = tau
        self.is_super = is_super
        self.superbpe_mode = superbpe_mode
        self.blowup = blowup

        # Pretokenizer (required)
        self.pretokenizer = pretokenizer

        # Inverted indices for fast O(1) lookup during encoding
        # Index lists are sorted to enable binary search in Rust
        self.merges_lookup = merges_lookup
        self.deletions_lookup = deletions_lookup

        # Precomputed replacement parts for deleted tokens (avoids O(n) scan at inference time)
        self.deletion_parts = deletion_parts

    @classmethod
    def create_for_training(cls, pretokenizer: Pretokenizer) -> 'InferenceData':
        """
        Factory method to create an empty InferenceData instance for training.
        Fields will be populated during the training process.
        """
        from collections import defaultdict
        return cls(
            tau=1.1,
            is_super=False,
            superbpe_mode=False,  # Must be False when is_super is False
            blowup=False,
            pretokenizer=pretokenizer,
            merges={},
            deletions={},
            merges_lookup=defaultdict(list),
            deletions_lookup=defaultdict(list),
            deletion_parts={},
        )

    def verify_indices(self) -> None:
        """Verify that merge and deletion indices are sequential starting from the minimum index."""
        indices = sorted(list(self.merges.keys()) + list(self.deletions.keys()))

        # should be an ordered list of indices
        if len(indices) > 0:
            min_index = indices[0]
            for i, ind in enumerate(indices):
                expected_index = min_index + i
                # about to die so dump them
                if expected_index != ind:
                    print("debug 3: expected", expected_index, "got", ind)
                    print("word merges:")
                    for idx, (pair, c_ab, unlocked_flag) in self.merges.items():
                        print(idx, frombytespair(pair), c_ab, unlocked_flag)
                    print()
                    print("word deletions:")
                    for idx, tok in self.deletions.items():
                        print(idx, frombytes(tok))
                    print()

                assert expected_index == ind

    def get_replacement_parts(self, bad_token: bytes) -> list[bytes]:
        """
        Get the parts to replace a deleted token with.
        Either returns the merge pair that created it, or single bytes.
        Uses precomputed deletion_parts map for O(1) lookup.
        """
        assert bad_token in self.deletion_parts, "couldn't find replacement parts for " + str(bad_token)
        return self.deletion_parts[bad_token]

    def trim_operations_to(self, num_ops: int) -> None:
        """
        Trim merges and deletions to keep only first num_ops operations.
        Operations are ordered by index.

        This should only be called on word models (is_super=False), not superword sections.
        Note: Vocabulary trimming is handled separately by Vocabulary.save(max_size=N).

        Args:
            num_ops: Number of operations to keep (by index)
        """
        assert not self.is_super, "trim_operations_to should only be called on word models, not superword sections"
        assert len(self.merges) > 0 or len(self.deletions) > 0, "No operations to trim"

        all_indices = sorted(list(self.merges.keys()) + list(self.deletions.keys()))
        indices_to_keep = set(all_indices[:num_ops])

        # Remove operations beyond num_ops
        self.merges = {k: v for k, v in self.merges.items() if k in indices_to_keep}
        self.deletions = {k: v for k, v in self.deletions.items() if k in indices_to_keep}

        # Rebuild lookup indices - store lists of indices for duplicate pairs/tokens
        # Note: A pair/token can have multiple indices when it's deleted and later recreated
        from collections import defaultdict
        self.merges_lookup = defaultdict(list)
        for idx, (pair, count, unlocked_flag) in self.merges.items():
            self.merges_lookup[pair].append(idx)

        self.deletions_lookup = defaultdict(list)
        for idx, token in self.deletions.items():
            self.deletions_lookup[token].append(idx)

        # Sort index lists to enable binary search
        for idx_list in self.merges_lookup.values():
            idx_list.sort()
        for idx_list in self.deletions_lookup.values():
            idx_list.sort()

        # Rebuild deletion_parts
        self.deletion_parts = {}
        for token in self.deletions.values():
            if token not in self.deletion_parts:
                if not self.blowup:
                    parts = None
                    for ((left, right), cnt, unlocked_flag) in self.merges.values():
                        if left + right == token:
                            parts = [left, right]
                            break
                    assert parts is not None, "couldn't find merge for " + str(token)
                    self.deletion_parts[token] = parts
                else:
                    self.deletion_parts[token] = [bytes([b]) for b in token]

    def _write_to_file(self, f: TextIO) -> None:
        """
        Write inference data to an open file handle (internal method).

        Writes config JSON, merges section, and deletions section.
        Vocabulary is handled separately by the Vocabulary class.

        Args:
            f: Open file handle to write to
        """
        # write pretokenizer configuration as JSON
        # Save actual regex pattern strings from compiled patterns
        config = {
            "tau": self.tau,
            "is_super": self.is_super,
            "superbpe_mode": self.superbpe_mode,
            "blowup": self.blowup,
            "main_regex": self.pretokenizer.main_pattern.pattern,
            "script_specific_regex": self.pretokenizer.script_specific_pattern.pattern if self.pretokenizer.script_specific_pattern is not None else None,
            "script_specific_scripts": sorted(list(self.pretokenizer.script_specific_scripts_set)) if self.pretokenizer.script_aware_mode else None,
            "merge_pattern": self.pretokenizer.merge_pattern.pattern
        }
        f.write(json.dumps(config) + "\n")

        f.write("merges\n")
        _write_sorted_dict_intkey_with_counts(self.merges, f, ispair=True, isstr=False)

        f.write("deletions\n")
        _write_sorted_dict_intkey(self.deletions, f, ispair=False, isstr=False)


    @staticmethod
    def load(f: TextIO) -> "InferenceData":
        """
        Load inference data from an open file handle.

        Reads config JSON, merges section, and deletions section.
        Vocabulary is handled separately by the Vocabulary class.

        Args:
            f: Open file handle to read from

        Returns:
            Fully initialized InferenceData instance
        """
        # read pretokenizer config
        config = json.loads(f.readline().strip())

        tau = config.get("tau")
        is_super = config.get("is_super")
        superbpe_mode = config.get("superbpe_mode")
        blowup = config.get("blowup")

        assert tau is not None, "tau must be in config"
        assert is_super is not None, "is_super must be in config"
        assert superbpe_mode is not None, "superbpe_mode must be in config"
        assert blowup is not None, "blowup must be in config"

        # Recreate the pretokenizer with saved patterns
        pretokenizer = Pretokenizer(
            main_regex=config.get("main_regex"),
            script_specific_regex=config.get("script_specific_regex"),
            script_specific_scripts=config.get("script_specific_scripts"),
            merge_pattern=config.get("merge_pattern")
        )

        # Read merges section
        line = f.readline().strip()
        assert line == "merges", f"Expected merges, got {line}"
        merges, _ = _read_sorted_dict_intkey_with_counts(f, ispair=True, isstr=False)

        # Print smallest regular merge count if there are any merges
        if len(merges) > 0:
            min_c_ab = min(c_ab for (pair, c_ab, unlocked_flag) in merges.values())
            print(f"Loaded {len(merges)} merges, minimum count: {min_c_ab}")

        # Read deletions section
        line = f.readline().strip()
        assert line == "deletions", f"Expected deletions, got {line}"
        deletions = _read_sorted_dict_intkey(f, ispair=False, isstr=False)

        # Build inverted lookup indices for fast encoding.
        # A pair/token can have multiple indices when it's deleted and later recreated.
        # During encoding (fast_merge_delete), prev_ind is tracked to ensure operations
        # are applied in the correct order: we only use merge/deletion indices > prev_ind.
        from collections import defaultdict
        merges_lookup = defaultdict(list)
        for idx, (pair, count, unlocked_flag) in merges.items():
            merges_lookup[pair].append(idx)

        total_merge_len = 0
        for pair, idx_list in merges_lookup.items():
            total_merge_len += len(idx_list)
            if len(idx_list) > 1:
                print("info: read multiple merge entries:", pair, idx_list)

        deletions_lookup = defaultdict(list)
        for idx, token in deletions.items():
            deletions_lookup[token].append(idx)

        total_deletions_len = 0
        for token, idx_list in deletions_lookup.items():
            total_deletions_len += len(idx_list)
            if len(idx_list) > 1:
                print("info: read multiple deletion entries: ", token, idx_list)

        # Sort index lists to enable binary search
        for idx_list in merges_lookup.values():
            idx_list.sort()
        for idx_list in deletions_lookup.values():
            idx_list.sort()

        print(f"Built lookup indices: {len(merges_lookup)} unique merge pairs with {total_merge_len} values, {len(deletions_lookup)} unique deletion tokens with {total_deletions_len} values")

        # Precompute replacement parts for each deleted token
        deletion_parts: dict[bytes, list[bytes]] = {}
        for token in deletions.values():
            if token not in deletion_parts:
                if not blowup:
                    # Find the merge pair that created this token
                    parts = None
                    for ((left, right), cnt, unlocked_flag) in merges.values():
                        if left + right == token:
                            parts = [left, right]
                            break
                    assert parts is not None, "couldn't find merge for " + str(token)
                    deletion_parts[token] = parts
                else:
                    deletion_parts[token] = [bytes([b]) for b in token]

        # Construct and return the instance
        instance = InferenceData(
            tau=tau,
            is_super=is_super,
            superbpe_mode=superbpe_mode,
            blowup=blowup,
            pretokenizer=pretokenizer,
            merges=merges,
            deletions=deletions,
            merges_lookup=merges_lookup,
            deletions_lookup=deletions_lookup,
            deletion_parts=deletion_parts,
        )

        # Verify indices are consistent
        instance.verify_indices()

        return instance
