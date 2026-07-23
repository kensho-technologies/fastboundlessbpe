# Copyright 2026-present Kensho Technologies, LLC.
"""Train a HuggingFace BPE on the same corpus and compare to one of our word models.

This is a VALIDATION script, not a shipped code path. It checks how close HF's own
BpeTrainer (ByteLevel + Split with our regex) comes to our word-level BPE. Exact
vocab/merge equality is NOT achievable, so we cannot compare training bit-for-bit;
expect a model that is slightly different. Two independent reasons:

1. Overlapping-pair counting (the main one). HF's BpeTrainer counts EVERY adjacent
   pair in a pretoken, including overlapping occurrences inside a run of a repeated
   token. Our trainer counts only NON-overlapping pairs -- the merges BPE can
   actually apply -- so the 3-space pretoken "   " contributes one (space,space)
   pair, whereas HF counts two. (Verified: HF trained on ["   "]*100 + ["ab"]*130
   picks (space,space) first, count 200 > 130.)

   This is a real count difference, not a tie, and it can change which merge is
   selected -- including the very first one. Worked example, same 2000 minipile
   docs, same pretokens:

       pair          naive/overlap (HF)   non-overlapping (ours)
       (space,space)      253888                133322
       (space,t)          182834                182834
       first merge        (space,space)         (space,t)

   So HF's first merge is "ĠĠ" while ours is the usual "Ġt". Whether this crossover
   happens is corpus-dependent: prose (e.g. CulturaX) has few long space runs, so
   (space,t) wins even under overlap counting and HF also starts with "Ġt";
   whitespace/indentation-heavy corpora like minipile trip the crossover. Either
   way the two models end up slightly different because the counting differs.
2. Tie-breaking. When counts are genuinely equal, we break ties lexicographically
   by token bytes; HF uses its own order.

The script reports set overlap and the first ordered-merge divergence rather than
asserting identity. Expect high merge-INVENTORY overlap with an early ordering split.

Usage:
    python examples/verify_hf_training.py --dataset data.jsonl --num-docs 20000 --vocab-size 4096
"""
import argparse
import contextlib
import io
import json
from typing import Iterator

from tokenizers import Tokenizer as HFTokenizer, models, pre_tokenizers, decoders, trainers, Regex

from boundlessbpe import GPT4O_EXPORT_REGEX
from boundlessbpe.train import BpeTrainer
from boundlessbpe.pretokenize import Pretokenizer


def iter_texts(path: str, n: int) -> Iterator[str]:
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            yield json.loads(line)["text"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--num-docs", type=int, default=20000)
    ap.add_argument("--vocab-size", type=int, default=4096)
    args = ap.parse_args()

    texts = list(iter_texts(args.dataset, args.num_docs))

    # --- Our word model ----------------------------------------------------
    import tempfile, os, glob
    d = tempfile.mkdtemp()
    jsonl = os.path.join(d, "c.jsonl")
    with open(jsonl, "w") as f:
        for t in texts:
            f.write(json.dumps({"text": t}) + "\n")
    with contextlib.redirect_stdout(io.StringIO()):
        BpeTrainer(Pretokenizer(main_regex=GPT4O_EXPORT_REGEX)).train(
            tau=1.1, filepath=jsonl, outprefix=os.path.join(d, "w"),
            num_lines=len(texts), vocab_size=args.vocab_size, recalc=0,
            blowup=False, checkpoint_iterations=0, verbose=False)
    from boundlessbpe.inference import Tokenizer
    ours = Tokenizer()
    with contextlib.redirect_stdout(io.StringIO()):
        ours.load(sorted(glob.glob(os.path.join(d, "w_*.model")),
                         key=lambda p: int(p.split("_")[-1].split(".")[0]))[-1])
    from boundlessbpe.export import build_export_view
    our_view = build_export_view(ours)
    our_merges = list(our_view.merges)

    # --- HF-trained BPE ----------------------------------------------------
    hf = HFTokenizer(models.BPE(byte_fallback=False))
    hf.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(pattern=Regex(GPT4O_EXPORT_REGEX), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    hf.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=[],
        show_progress=False,
    )
    hf.train_from_iterator(texts, trainer=trainer)

    from boundlessbpe.util import tobytes
    hf_model = json.loads(hf.to_str())["model"]
    hf_merges = []
    for m in hf_model["merges"]:
        a, b = (m if isinstance(m, list) else m.split(" "))
        hf_merges.append((tobytes(a), tobytes(b)))

    # --- Compare -----------------------------------------------------------
    print(f"our merges: {len(our_merges)}  hf merges: {len(hf_merges)}")
    our_set, hf_set = set(our_merges), set(hf_merges)
    inter = our_set & hf_set
    print(f"merge overlap (set): {len(inter)} "
          f"({100 * len(inter) / max(1, len(our_set)):.1f}% of ours)")
    # first index where ordered merge lists differ
    first = next((i for i in range(min(len(our_merges), len(hf_merges)))
                  if our_merges[i] != hf_merges[i]), None)
    print(f"first ordered-merge divergence at index: {first}")
    if first is not None:
        print(f"  ours[{first}] = {our_merges[first]}")
        print(f"  hf  [{first}] = {hf_merges[first]}")
    print("NOTE: exact equality is not expected; tie-breaking may differ between "
          "HuggingFace and BoundlessBPE.")


if __name__ == "__main__":
    main()
