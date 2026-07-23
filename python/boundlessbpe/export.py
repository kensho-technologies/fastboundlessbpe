# Copyright 2026-present Kensho Technologies, LLC.
"""
Export BoundlessBPE models to standard tokenizer formats (HuggingFace, tiktoken).

Neither target format understands BoundlessBPE's cross-pretoken supermerges or
PickyBPE deletions. Export therefore relies on the "SuperBPE trick": emit the
vocabulary and merges as a plain byte-level BPE, paired with a *coarser*
pretokenization regex that leaves runs of mergeable word-pretokens as a single
pretoken, so standard BPE reproduces the supermerges.

Constraints (raise ValueError otherwise):
- The model must NOT have been trained with PickyBPE deletions.
- The model must NOT use the script-aware / script-specific pretokenizer.
- coarse_regex must be supplied for a superword model, and must NOT be supplied
  for a plain word model.

Inference in BoundlessBPE is two-phase: all word merges are applied to each
pretoken, then all supermerges are applied across the run. A single flat BPE
merge list reproduces this as two blocks: word merges (in index order) followed
by supermerges (in index order).
"""

import base64
import json
import os
from typing import TYPE_CHECKING, Optional

from .util import frombytes

if TYPE_CHECKING:
    from .inference import Tokenizer


class ExportView:
    """A plain byte-level BPE view of a model, ready to serialize.

    In BoundlessBPE inference, all word merges are applied to each pretoken before
    any supermerge is applied across the run. A single-pass BPE reproduces this by
    ordering merges (and therefore ranks) as: base-alphabet bytes, then word-merge
    results in index order, then supermerge results in index order. This "two-block"
    order is the merge priority — NOT the model's own vocabulary id order, which
    interleaves word and super tokens by training-competition order. For word-only
    models the two coincide; for superword models the exported ids/ranks are
    renumbered and therefore differ from the model's vocabulary ids.

    Attributes:
        tokens: token bytes in BPE rank order (index in this list == rank == id).
        rank: token bytes -> rank/id (inverse of tokens).
        merges: ordered list of (left_bytes, right_bytes) merge pairs, in apply order.
        regex: the pretokenization pattern to pair with this BPE — the model's own
            main regex for a word model, or the coarse regex for a superword model.
        special_tokens: special token string -> id (ids continue after len(tokens)).
    """

    def __init__(
        self,
        tokens: list[bytes],
        merges: list[tuple[bytes, bytes]],
        regex: str,
        special_tokens: dict[str, int],
    ) -> None:
        self.tokens = tokens
        self.rank: dict[bytes, int] = {tok: i for i, tok in enumerate(tokens)}
        self.merges = merges
        self.regex = regex
        self.special_tokens = special_tokens


def build_export_view(tokenizer: "Tokenizer", coarse_regex: Optional[str] = None) -> ExportView:
    """Build a plain-BPE ExportView from a loaded Tokenizer, validating constraints.

    Args:
        tokenizer: a loaded Tokenizer (word, boundless, or superbpe model).
        coarse_regex: required for superword models, forbidden for word models.

    Returns:
        ExportView with vocab, ordered merges, regex, and special tokens.

    Raises:
        ValueError: if the model has deletions, uses script-aware mode, or the
            coarse_regex argument violates the word/superword contract.
    """
    assert tokenizer.vocab is not None, "model must be loaded"
    assert tokenizer.words is not None, "model must be loaded"

    words = tokenizer.words
    superwords = tokenizer.superwords
    is_superword_model = superwords is not None

    # --- Validation --------------------------------------------------------
    if words.deletions:
        raise ValueError(
            "Cannot export a model trained with PickyBPE deletions to a plain BPE "
            f"format: the word model has {len(words.deletions)} deletion(s). Deleted "
            "tokens have no representation in HuggingFace/tiktoken merge lists. "
            "Retrain with tau >= 1.1 (deletions disabled) to export."
        )
    if superwords is not None and superwords.deletions:
        raise ValueError(
            "Cannot export a model whose superword section has deletions to a plain "
            f"BPE format ({len(superwords.deletions)} deletion(s))."
        )

    # Script-aware mode splits some scripts character-by-character with a second
    # regex; a single plain-BPE pretokenizer cannot reproduce that.
    if words.pretokenizer.script_specific_pattern is not None:
        raise ValueError(
            "Cannot export a model that uses the script-aware pretokenizer "
            "(script_specific_regex is set) to a plain BPE format, which supports "
            "only a single pretokenization regex."
        )

    if is_superword_model and coarse_regex is None:
        raise ValueError(
            "coarse_regex is required to export a superword (boundless/superbpe) "
            "model. Supply the coarse 'trick' regex that keeps mergeable runs as a "
            "single pretoken (e.g. GPT4O_COARSE_REGEX for the GPT4O default)."
        )
    if not is_superword_model and coarse_regex is not None:
        raise ValueError(
            "coarse_regex must not be passed for a regular (word) BPE model; its own "
            "pretokenization regex is exported as-is."
        )

    # --- Merges (word block, then superword block), in apply order ---------
    word_merges: list[tuple[bytes, bytes]] = [
        words.merges[idx][0] for idx in sorted(words.merges.keys())
    ]
    super_merges: list[tuple[bytes, bytes]] = []
    if superwords is not None:
        super_merges = [
            superwords.merges[idx][0] for idx in sorted(superwords.merges.keys())
        ]
    merges: list[tuple[bytes, bytes]] = word_merges + super_merges

    # --- Rank order: base bytes, then word-merge results, then supermerges --
    # This is the BPE merge priority. It equals vocabulary-id order for word-only
    # models but is renumbered for superword models (see class docstring).
    produced = {left + right for (left, right) in merges}
    base_tokens = [t for t in tokenizer.vocab.tokens if t not in produced]
    ordered = base_tokens + [l + r for (l, r) in word_merges] + [l + r for (l, r) in super_merges]
    tokens: list[bytes] = []
    seen: set[bytes] = set()
    for t in ordered:
        if t not in seen:
            seen.add(t)
            tokens.append(t)
    assert len(tokens) == len(tokenizer.vocab.tokens), (
        f"rank ordering lost tokens: {len(tokens)} != {len(tokenizer.vocab.tokens)}"
    )

    # --- Regex -------------------------------------------------------------
    if is_superword_model:
        assert coarse_regex is not None  # guaranteed by validation above
        regex_str: str = coarse_regex
    else:
        regex_str = words.pretokenizer.main_pattern.pattern

    # --- Special tokens (ids continue after the regular tokens) ------------
    # Renumber so specials sit directly above the exported token block.
    n = len(tokens)
    ordered_specials = sorted(tokenizer.vocab.special_tokens.items(), key=lambda kv: kv[1])
    special_tokens: dict[str, int] = {tok: n + i for i, (tok, _old) in enumerate(ordered_specials)}

    return ExportView(tokens=tokens, merges=merges, regex=regex_str, special_tokens=special_tokens)


def export_tiktoken(
    tokenizer: "Tokenizer",
    path: str,
    coarse_regex: Optional[str] = None,
) -> None:
    """Export a model to tiktoken format.

    Writes two files next to `path`:
      - `<path>` : the `.tiktoken` rank file, one line per token
        `base64(token_bytes) SPACE rank`, sorted by rank (rank == token id).
      - `<path>.json` : a sidecar with the `pat_str` (pretokenization regex) and
        the special-token map, since a `.tiktoken` file alone carries neither.

    Reconstruct with:
        ranks = tiktoken.load.load_tiktoken_bpe(path)
        meta = json.load(open(path + ".json"))
        enc = tiktoken.Encoding(name=..., pat_str=meta["pat_str"],
                                mergeable_ranks=ranks, special_tokens=meta["special_tokens"])
    """
    view = build_export_view(tokenizer, coarse_regex)

    # tiktoken derives merges from ranks, so we only emit ranks (base64 token + rank),
    # in rank order (base bytes, then word merges, then supermerges).
    lines = []
    for rank, token in enumerate(view.tokens):
        lines.append(base64.b64encode(token).decode("ascii") + " " + str(rank))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

    sidecar = {
        "pat_str": view.regex,
        "special_tokens": view.special_tokens,
    }
    with open(path + ".json", "w") as f:
        json.dump(sidecar, f, ensure_ascii=False, indent=2)


def export_huggingface(
    tokenizer: "Tokenizer",
    path: str,
    coarse_regex: Optional[str] = None,
) -> None:
    """Export a model to HuggingFace format into directory `path`.

    Builds the tokenizer via the `tokenizers` library (not hand-written JSON) so the
    artifact is guaranteed loadable: a byte-level BPE with a Split(regex) + ByteLevel
    pre-tokenizer sequence and a ByteLevel decoder. Writes three files:
      - tokenizer.json         (the tokenizer)
      - tokenizer_config.json  (tokenizer_class, special tokens, model_max_length)
      - special_tokens_map.json
    so the directory loads via both tokenizers.Tokenizer.from_file(.../tokenizer.json)
    and transformers.AutoTokenizer.from_pretrained(path). The two config files are
    written by hand to avoid a dependency on `transformers`.

    Requires the `tokenizers` package (install the `export` extra).
    """
    try:
        from tokenizers import Tokenizer as HFTokenizer, models, pre_tokenizers, decoders, Regex
        from tokenizers import AddedToken
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "HuggingFace export requires the `tokenizers` package. "
            "Install with: pip install boundlessbpe[export]"
        ) from e

    view = build_export_view(tokenizer, coarse_regex)

    # HF vocab keys are the GPT-2 byte-to-unicode strings (our frombytes), same as
    # the ByteLevel pretokenizer emits at encode time. Ids follow BPE rank order.
    hf_vocab = {frombytes(token): rank for token, rank in view.rank.items()}
    hf_merges = [(frombytes(left), frombytes(right)) for (left, right) in view.merges]

    bpe = models.BPE(vocab=hf_vocab, merges=hf_merges, byte_fallback=False)
    hf = HFTokenizer(bpe)

    # Split on our pretokenization regex, then apply ByteLevel (no second regex).
    # `behavior`/`invert` chosen to match our regex.findall semantics; verified by
    # the parity harness (examples/verify_hf.py).
    hf.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(pattern=Regex(view.regex), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    hf.decoder = decoders.ByteLevel()

    ordered_specials = sorted(view.special_tokens.items(), key=lambda kv: kv[1])
    if ordered_specials:
        # Add in id order so HF assigns matching ids (they already sit above vocab).
        hf.add_special_tokens([AddedToken(tok, special=True) for tok, _ in ordered_specials])

    os.makedirs(path, exist_ok=True)
    hf.save(os.path.join(path, "tokenizer.json"))

    # Also write tokenizer_config.json and special_tokens_map.json so the export
    # loads via transformers.AutoTokenizer.from_pretrained(path), not only via
    # tokenizers.Tokenizer.from_file(). Written by hand to avoid a dependency on
    # the (heavy) transformers package; these mirror what PreTrainedTokenizerFast
    # would emit for a fast (tokenizer.json-backed) tokenizer.
    special_strs = [tok for tok, _ in ordered_specials]
    added_tokens_decoder = {
        str(idx): {
            "content": tok,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True,
        }
        for tok, idx in ordered_specials
    }
    tokenizer_config: dict[str, object] = {
        "added_tokens_decoder": added_tokens_decoder,
        "clean_up_tokenization_spaces": False,
        "model_max_length": 1000000000000000019884624838656,
        "tokenizer_class": "PreTrainedTokenizerFast",
    }
    # Surface eos/pad if the usual names were registered (best effort, harmless if absent).
    for name, cand in (("eos_token", "<|endoftext|>"), ("pad_token", "<|pad|>"),
                       ("bos_token", "<|startoftext|>"), ("unk_token", "<unk>")):
        if cand in view.special_tokens:
            tokenizer_config[name] = cand
    with open(os.path.join(path, "tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)

    special_tokens_map: dict[str, object] = {}
    for name, cand in (("eos_token", "<|endoftext|>"), ("pad_token", "<|pad|>"),
                       ("bos_token", "<|startoftext|>"), ("unk_token", "<unk>")):
        if cand in view.special_tokens:
            special_tokens_map[name] = cand
    if special_strs:
        special_tokens_map["additional_special_tokens"] = special_strs
    with open(os.path.join(path, "special_tokens_map.json"), "w") as f:
        json.dump(special_tokens_map, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------


def _superword_expansion_map(tokenizer: "Tokenizer") -> dict[bytes, tuple[bytes, bytes]]:
    """Map each superword token to the (left, right) pair that produced it."""
    if tokenizer.superwords is None:
        return {}
    return {l + r: (l, r) for ((l, r), _c, _f) in tokenizer.superwords.merges.values()}


def expand_superwords(tokens: list[bytes], parent: dict[bytes, tuple[bytes, bytes]]) -> list[bytes]:
    """Recursively expand every superword token to its word-level pieces.

    Two token sequences over the same text are equivalent-up-to-supermerging iff
    their expansions are equal. This is the acceptance criterion for exported
    superword models: word-level tokenization must be identical, and the only
    permitted differences are which supermerges got applied.
    """
    out: list[bytes] = []
    stack = list(reversed(tokens))
    while stack:
        tok = stack.pop()
        if tok in parent:
            l, r = parent[tok]
            stack.append(r)
            stack.append(l)
        else:
            out.append(tok)
    return out


def classify_export_difference(
    tokenizer: "Tokenizer",
    our_tokens: list[bytes],
    exported_tokens: list[bytes],
    parent: Optional[dict[bytes, tuple[bytes, bytes]]] = None,
) -> str:
    """Classify how an exported tokenization differs from ours for one text.

    Returns one of:
      - "identical"      : byte-identical token sequences (the goal for word models).
      - "supermerge"     : differs only in applied supermerges (word-level identical);
                           this is the documented, best-effort limitation for superword
                           export (a supermerge participant that is also a prefix
                           sub-token of a longer word).
      - "byte-mismatch"  : the underlying bytes differ (real corruption — a bug).
      - "wordlevel"      : bytes match but word-level expansion differs (a real bug that
                           is NOT the known supermerge limitation).
    """
    if our_tokens == exported_tokens:
        return "identical"
    if b"".join(our_tokens) != b"".join(exported_tokens):
        return "byte-mismatch"
    if parent is None:
        parent = _superword_expansion_map(tokenizer)
    if expand_superwords(our_tokens, parent) == expand_superwords(exported_tokens, parent):
        return "supermerge"
    return "wordlevel"
