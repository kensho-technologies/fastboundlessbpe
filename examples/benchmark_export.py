# Copyright 2026-present Kensho Technologies, LLC.
"""End-to-end export benchmark and correctness check.

What it does:
  1. Trains a superword (BoundlessBPE) model over a corpus with GPT4O_EXPORT_REGEX.
  2. Exports it to tiktoken and HuggingFace using the coarse "trick" regex.
  3. Encodes every document with all three tokenizers (our Rust FastTokenizer in
     export_compatible mode, tiktoken, HuggingFace), timing each.
  4. Classifies every per-document difference and reports per-method inference time.

Why the regexes: the model is trained with GPT4O_EXPORT_REGEX (leading-space-only
words, internal-only apostrophes) so the coarse companion GPT4O_COARSE_REGEX can
reproduce cross-pretoken supermerges under plain single-pass BPE. `export_compatible=True`
on our tokenizer is the mode that mirrors that plain-BPE behavior, so it is the fair
thing to compare the exports against. Comparison is on token *bytes*, not ids, because
superword export renumbers ids. All three encode with allowed_special="all" so a literal
special-token string in a document (e.g. "<|endoftext|>") is handled the same way by all.

What "correct" means here: the exports are NOT byte-identical to BoundlessBPE in general.
The coarse regex collapses a run of word-pretokens into one flat pretoken, so plain BPE
can apply a *regular* merge that crosses a fine-pretoken boundary — something BoundlessBPE
never does (it only lets *supermerges* cross boundaries). Such differences are expected and
benign; the script classifies each difference and PASSES as long as every one is a
boundary-crossing merge (byte-identical text, only the segmentation differs). It FAILS only
on genuine defects: corrupted bytes, or a difference inside a single pretoken.

Empirically (1,000,000 minipile docs, vocab 40960): 24/1M differ vs tiktoken and 26/1M vs
HuggingFace — all boundary-crossing merges in no-space multi-script runs (see the examples
by the classifier below). Throughput (Rust): tiktoken ~4.1M tok/s, ours ~1.5M, HF ~1.4M.

Usage:
    python examples/benchmark_export.py --num-lines 1000000 --vocab-size 40960 \
        --special-token "<|endoftext|>" --output-dir ./export_benchmark_1M \
        --mismatch-log mism.jsonl
    python examples/benchmark_export.py --num-lines 20000 --vocab-size 8192   # quick check

Requires the export extra: pip install boundlessbpe[export]
"""
import argparse
import contextlib
import glob
import io
import json
import os
import tempfile
import time
from collections import Counter

import tiktoken
from tiktoken.load import load_tiktoken_bpe
from tokenizers import Tokenizer as HFTokenizer

from boundlessbpe import (
    FastBpeTrainer, FastBoundlessBpeTrainer, Tokenizer, FastTokenizer,
    GPT4O_EXPORT_REGEX, GPT4O_COARSE_REGEX,
)
from boundlessbpe.util import tobytes, frombytes


def _latest_model(prefix: str) -> str:
    models = glob.glob(prefix + "_*.model")
    return sorted(models, key=lambda p: int(p.split("_")[-1].split(".")[0]))[-1]


def read_docs(path: str, n: int) -> list[str]:
    docs: list[str] = []
    with open(path) as f:
        for line in f:
            docs.append(json.loads(line)["text"])
            if len(docs) >= n:
                break
    return docs


def train_model(filepath: str, num_lines: int, vocab_size: int, workdir: str) -> str:
    """Train word + boundless models with GPT4O_EXPORT_REGEX; return boundless .model path."""
    wprefix = os.path.join(workdir, "word")
    bprefix = os.path.join(workdir, "bnd")
    t0 = time.time()
    FastBpeTrainer(main_regex=GPT4O_EXPORT_REGEX).train(
        tau=1.1, filepath=filepath, outprefix=wprefix, num_lines=num_lines,
        vocab_size=vocab_size, recalc=0, blowup=False, checkpoint_iterations=0,
        verbose=False, progress_interval=0)
    word_model = _latest_model(wprefix)
    print(f"  word model trained in {time.time() - t0:.1f}s -> {os.path.basename(word_model)}")
    t0 = time.time()
    FastBoundlessBpeTrainer(main_regex=GPT4O_EXPORT_REGEX).train(
        filepath=filepath, outprefix=bprefix, num_lines=num_lines, recalc=0,
        word_model_file=word_model, checkpoint_iterations=0, verbose=False,
        progress_interval=0)
    bnd_model = _latest_model(bprefix)
    print(f"  boundless model trained in {time.time() - t0:.1f}s -> {os.path.basename(bnd_model)}")
    return bnd_model


def build_encoders(
    bnd_model: str, outdir: str, special_tokens: list[str]
) -> "tuple[Tokenizer, FastTokenizer, tiktoken.Encoding, HFTokenizer]":
    """Load our tokenizer (registering any special tokens) and build the exported
    tiktoken + HuggingFace tokenizers. Artifacts are written under outdir."""
    ours = Tokenizer()
    with contextlib.redirect_stdout(io.StringIO()):
        ours.load(bnd_model)
    if special_tokens:
        ours.add_special_tokens(special_tokens)
        print(f"  registered special tokens: {special_tokens}")

    os.makedirs(outdir, exist_ok=True)
    # Also save our own .model (with the special tokens) alongside the exports.
    model_path = os.path.join(outdir, "model.model")
    ours.save(os.path.join(outdir, "model"))
    tik_path = os.path.join(outdir, "export.tiktoken")
    hf_dir = os.path.join(outdir, "hf")
    ours.save_tiktoken(tik_path, coarse_regex=GPT4O_COARSE_REGEX)
    ours.save_huggingface(hf_dir, coarse_regex=GPT4O_COARSE_REGEX)
    print(f"  exports written to {outdir}/ (model.model, export.tiktoken[+.json], hf/)")

    # Time inference with the Rust FastTokenizer (our production path), so all three
    # timed tokenizers are Rust-backed and the comparison is apples-to-apples.
    fast = FastTokenizer()
    with contextlib.redirect_stdout(io.StringIO()):
        fast.load(model_path)

    ranks = load_tiktoken_bpe(tik_path)
    meta = json.load(open(tik_path + ".json"))
    tik = tiktoken.Encoding(name="bbpe", pat_str=meta["pat_str"],
                            mergeable_ranks=ranks, special_tokens=meta["special_tokens"])
    hf = HFTokenizer.from_file(os.path.join(hf_dir, "tokenizer.json"))
    return ours, fast, tik, hf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-lines", type=int, required=True)
    ap.add_argument("--vocab-size", type=int, default=131072)
    ap.add_argument("--filepath", type=str, default="data/minipile.jsonl")
    ap.add_argument("--output-dir", type=str, default="./export_benchmark",
                    help="Directory for the persisted model + exports (default: ./export_benchmark)")
    ap.add_argument("--special-token", action="append", default=[], dest="special_tokens",
                    help="Special token to register before export (repeatable)")
    ap.add_argument("--mismatch-log", type=str, default=None,
                    help="If set, write each differing document to this JSONL file for diagnosis")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as workdir:
        print(f"Training superword model: {args.num_lines} docs, vocab {args.vocab_size}")
        bnd_model = train_model(args.filepath, args.num_lines, args.vocab_size, workdir)

        print("Exporting to tiktoken and HuggingFace...")
        ours, fast, tik, hf = build_encoders(bnd_model, args.output_dir, args.special_tokens)

        print(f"Reading {args.num_lines} docs for the inference comparison...")
        docs = read_docs(args.filepath, args.num_lines)

        # Encode each document with all three tokenizers back to back, timing each call
        # and comparing on the spot, then discarding — so nothing is held across docs and
        # the three run under the same conditions. All three are Rust-backed: our
        # FastTokenizer (export_compatible), tiktoken, HuggingFace. Compare bytes (not
        # ids): superword export renumbers ids; `fast` shares the word-model vocab, so
        # ours.vocab.id_to_token maps its ids to bytes.
        print(f"Encoding {len(docs):,} docs with all three tokenizers (per-document)...")
        assert ours.vocab is not None and ours.words is not None  # loaded above
        id_to_token = ours.vocab.id_to_token
        inv_special = ours.vocab.inverse_special_tokens  # our special id -> str
        # tiktoken/HF special id -> bytes, so all three render specials identically.
        tik_special_bytes = {i: name.encode("utf-8") for name, i in tik._special_tokens.items()} \
            if hasattr(tik, "_special_tokens") else {}
        hf_special_ids = {hf.token_to_id(s): s for s in args.special_tokens if hf.token_to_id(s) is not None}

        def ours_bytes(i: int) -> bytes:
            t = id_to_token.get(i)
            return t if t is not None else inv_special[i].encode("utf-8")

        def tik_bytes(i: int) -> bytes:
            b = tik_special_bytes.get(i)
            return b if b is not None else tik.decode_single_token_bytes(i)

        def hf_bytes(i: int) -> bytes:
            s = hf_special_ids.get(i)
            return s.encode("utf-8") if s is not None else tobytes(hf.id_to_token(i))

        ours_time = tik_time = hf_time = 0.0
        total_tokens = 0
        tik_cat: Counter = Counter()
        hf_cat: Counter = Counter()
        perf = time.perf_counter
        mlog = open(args.mismatch_log, "w") if args.mismatch_log else None

        def first_divergence(a: list[bytes], b: list[bytes]) -> dict:
            """Index of first differing token + a small readable window around it."""
            k = 0
            while k < len(a) and k < len(b) and a[k] == b[k]:
                k += 1
            return {
                "index": k,
                "context": frombytes(b"".join(a[:k]))[-40:],
                "ours": [frombytes(t) for t in a[k:k + 6]],
                "other": [frombytes(t) for t in b[k:k + 6]],
            }

        pretok = ours.words.pretokenizer

        def _token_starts(toks: list[bytes]) -> list[int]:
            """Byte offset before each token, plus the final total offset."""
            starts, off = [], 0
            for t in toks:
                starts.append(off)
                off += len(t)
            starts.append(off)
            return starts

        def _tokens_in_span(toks: list[bytes], lo: int, hi: int) -> list[bytes]:
            """Tokens whose start offset lies in [lo, hi)."""
            res, off = [], 0
            for t in toks:
                if lo <= off < hi:
                    res.append(t)
                off += len(t)
            return res

        # Real strings from the 1M-doc run that produced boundary-crossing merges. They
        # are all no-separator runs of "words" — camelCase or joined scripts — where the
        # fine regex splits on the case/script change but the coarse regex keeps the run
        # whole, letting plain BPE merge across the (invisible) seam:
        #
        #   "ФичаДляПроверкиМетода"   Cyrillic camelCase identifier; fine splits
        #                             Фича|Для|Проверки|Метода, export merges the 'а'+'Д'
        #                             bytes across the Фича|Для seam (a partial-UTF-8 token).
        #   "ЗаведующийЗам.директора" likewise across Заведующий|Зам.
        #   "ТолстыйКлиентОбычно"     likewise across Толстый|Клиент|Обычно.
        #
        # Space-separated text (" the quick brown") never triggers this: the leading space
        # is itself a byte at the seam, so no cross-word merge forms. The affected inputs
        # are code identifiers and non-Latin scripts without word spacing.

        def classify(text: str, ours_toks: list[bytes], other_toks: list[bytes]) -> str:
            """Classify how an export tokenization differs from ours for one document.

              - "identical"      : same tokens.
              - "boundary_merge" : bytes match and every span where the two disagree
                    straddles a fine-pretoken boundary — i.e. the only differences are
                    regular merges crossing a boundary our tokenizer never crosses. This
                    is the documented, benign limitation of the coarse-regex export.
              - "byte_mismatch"  : underlying bytes differ (real corruption / bug).
              - "other"          : bytes match but some disagreeing span lies entirely
                    within one fine pretoken — not explained by a boundary-crossing
                    merge, so a real bug to investigate.

            Method: the two token lists cover the same bytes. Their shared cut offsets
            partition the text; within each [a,b) between adjacent shared cuts they cover
            the same bytes, so if the tokens there differ it is a "divergent span". A
            divergent span is benign iff a fine-pretoken boundary falls strictly inside it
            (the export merged across it); otherwise both stayed inside one pretoken and
            still tokenized differently — a bug.
            """
            if ours_toks == other_toks:
                return "identical"
            if b"".join(ours_toks) != b"".join(other_toks):
                return "byte_mismatch"
            fine = set()
            off = 0
            for p in pretok.pretokenize(text):
                fine.add(off)
                off += len(p.encode("utf-8"))
            fine.add(off)
            common = sorted(set(_token_starts(ours_toks)) & set(_token_starts(other_toks)))
            for a, b in zip(common, common[1:]):
                if _tokens_in_span(ours_toks, a, b) != _tokens_in_span(other_toks, a, b):
                    if not any(a < fb < b for fb in fine):
                        return "other"
            return "boundary_merge"

        for n, d in enumerate(docs):
            # All three encode with special-token handling enabled, so a literal
            # special string in the text (e.g. "<|endoftext|>") is emitted atomically
            # by all three — an apples-to-apples comparison.
            t0 = perf(); ours_ids = fast.encode(d, allowed_special="all", export_compatible=True); ours_time += perf() - t0
            t0 = perf(); tik_ids = tik.encode(d, allowed_special="all"); tik_time += perf() - t0
            t0 = perf(); hf_ids = hf.encode(d).ids; hf_time += perf() - t0

            ours_toks = [ours_bytes(i) for i in ours_ids]
            tik_toks = [tik_bytes(i) for i in tik_ids]
            hf_toks = [hf_bytes(i) for i in hf_ids]
            total_tokens += len(ours_toks)

            for fmt, other, cat in (("tik", tik_toks, tik_cat), ("hf", hf_toks, hf_cat)):
                kind = classify(d, ours_toks, other)
                cat[kind] += 1
                if kind != "identical" and mlog is not None:
                    rec = {"doc": n, "fmt": fmt, "kind": kind,
                           "diff": first_divergence(ours_toks, other), "text": d}
                    mlog.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    mlog.flush()

            if (n + 1) % 10000 == 0:
                print(f"  {n + 1:,} docs  "
                      f"(tik boundary/other/bad={tik_cat['boundary_merge']}/{tik_cat['other']}/{tik_cat['byte_mismatch']}, "
                      f"hf={hf_cat['boundary_merge']}/{hf_cat['other']}/{hf_cat['byte_mismatch']})", flush=True)

        if mlog is not None:
            mlog.close()

        def report(name: str, cat: "Counter") -> None:
            diffs = cat["boundary_merge"] + cat["other"] + cat["byte_mismatch"]
            print(f"  {name}: {diffs} differing docs "
                  f"(boundary_merge={cat['boundary_merge']}, other={cat['other']}, "
                  f"byte_mismatch={cat['byte_mismatch']})")

        print()
        print("=" * 60)
        print(f"Documents:            {len(docs):,}")
        print(f"Total tokens (ours):  {total_tokens:,}")
        print(f"Vocab size:           {args.vocab_size:,}")
        print("-" * 60)
        print("Differences vs our export_compatible tokenization:")
        report("tiktoken   ", tik_cat)
        report("huggingface", hf_cat)
        print("  (boundary_merge = expected: a regular merge crossing a fine-pretoken")
        print("   boundary, which the coarse-regex export permits but BoundlessBPE does not)")
        print("-" * 60)
        print("Inference time (total over corpus):")
        print(f"  ours FastTokenizer (export_compatible): {ours_time:8.1f}s  {total_tokens / ours_time:12,.0f} tok/s")
        print(f"  tiktoken:                               {tik_time:8.1f}s  {total_tokens / tik_time:12,.0f} tok/s")
        print(f"  huggingface:                            {hf_time:8.1f}s  {total_tokens / hf_time:12,.0f} tok/s")
        print("=" * 60)
        # Only genuine defects fail: byte corruption, or a difference within a single
        # pretoken (not explained by a boundary-crossing merge).
        bugs = (tik_cat["other"] + tik_cat["byte_mismatch"]
                + hf_cat["other"] + hf_cat["byte_mismatch"])
        if bugs == 0:
            print("RESULT: PASS (all differences are expected boundary-crossing merges)")
        else:
            print(f"RESULT: FAIL ({bugs} difference(s) not explained by boundary-crossing merges)")
            raise SystemExit(1)


if __name__ == "__main__":
    main()
