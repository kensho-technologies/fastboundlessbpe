# Copyright 2026-present Kensho Technologies, LLC.
# Train first-pass word model
# This creates a word-level BPE model that will be used as input for second-pass training
#
# Usage: python -u train_word.py --num-lines 10000 2>&1 | tee log_word_10k.txt

import argparse

from boundlessbpe import Pretokenizer, GPT4O_REGEX, SCRIPT_SPECIFIC_GPT4O_REGEX, DEFAULT_SCRIPT_SPECIFIC_SCRIPTS


def main(args: argparse.Namespace) -> None:
    if args.simple:
        # Simple mode: standard GPT4O regex, no script-specific pretokenization
        pretokenizer = Pretokenizer()
        ss_regex = None
        ss_scripts = None
    else:
        # Script-aware mode (default)
        pretokenizer = Pretokenizer(
            script_specific_regex=SCRIPT_SPECIFIC_GPT4O_REGEX,
            script_specific_scripts=DEFAULT_SCRIPT_SPECIFIC_SCRIPTS
        )
        ss_regex = SCRIPT_SPECIFIC_GPT4O_REGEX
        ss_scripts = DEFAULT_SCRIPT_SPECIFIC_SCRIPTS

    if args.fast:
        from boundlessbpe import FastBpeTrainer
        assert FastBpeTrainer is not None, "Rust extension not available - run 'maturin develop --release' first"
        tokenizer = FastBpeTrainer(
            script_specific_regex=ss_regex,
            script_specific_scripts=ss_scripts,
        )
    else:
        from boundlessbpe.train import BpeTrainer
        tokenizer = BpeTrainer(pretokenizer)

    # Print training parameters
    print("num_lines:", args.num_lines)
    print("vocab_size:", args.vocab_size)
    print("tau:", args.tau)
    print("blowup:", args.blowup)
    print("max_bytes:", args.max_bytes)
    print("recalc:", args.recalc)
    print("checkpoint_iterations:", args.checkpoint_iterations)
    print("fast:", args.fast)
    print("simple:", args.simple)
    print("save_pretokens:", args.save_pretokens)
    assert args.recalc >= 0

    impl = "fast" if args.fast else "py"
    outprefix = f"{args.output_prefix}_{args.num_lines}_{args.max_bytes}_{args.vocab_size}_{args.tau}_{int(args.blowup)}_{impl}_word"
    print("outprefix:", outprefix)

    # Train first-pass word model
    tokenizer.train(args.tau, args.filepath, outprefix, args.num_lines,
                    args.vocab_size, args.recalc, args.blowup, args.max_bytes,
                    args.checkpoint_iterations, args.verbose, args.progress_interval,
                    save_pretokens=args.save_pretokens)

    # Register special tokens
    tokenizer.register_special_tokens(["<|endoftext|>"])

    print("saving word model")
    tokenizer.save(outprefix + "_final")

    # Test loading with Tokenizer
    print("loading")
    from boundlessbpe.inference import Tokenizer
    tokenizer2 = Tokenizer()
    tokenizer2.load(outprefix + "_final.model")

    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train first-pass word model")
    parser.add_argument("--num-lines", type=int, required=True, help="Number of documents to read from training file")
    parser.add_argument("--vocab-size", type=int, default=131072, help="Target vocabulary size (default: 131072)")
    parser.add_argument("--tau", type=float, default=1.1, help="Deletion threshold (default: 1.1, >1.0 disables deletions)")
    parser.add_argument("--blowup", action="store_true", help="Delete tokens by splitting to bytes instead of merge pair")
    parser.add_argument("--max-bytes", type=int, default=10_000_000_000, help="Maximum bytes to process (default: 10GB)")
    parser.add_argument("--recalc", type=int, default=8192, help="Verification frequency (default: 8192, 0 to disable)")
    parser.add_argument("--filepath", type=str, default="data/minipile.jsonl", help="Path to training data (default: data/minipile.jsonl)")
    parser.add_argument("--checkpoint-iterations", type=int, default=8192, help="Save checkpoint every N iterations (default: 8192)")
    parser.add_argument("--output-prefix", type=str, default="./models/twopass", help="Output path prefix for model files (default: ./models/twopass)")
    parser.add_argument("--fast", action="store_true", help="Use Rust implementation for training")
    parser.add_argument("--simple", action="store_true", help="Use standard GPT4O regex without script-specific pretokenization")
    parser.add_argument("--save-pretokens", type=str, default=None, help="Save pretokenization data to this file (TSV format)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed diagnostics (per-document progress, pretokenization summaries, timing breakdowns)")
    parser.add_argument("--progress-interval", type=int, default=None, help="Print a progress row every N merges. Default: 1 if --verbose else 1024. 0 disables progress rows.")
    main(parser.parse_args())


# Example commands:
# python -u train_word.py --num-lines 10000 2>&1 | tee log_word_10k.txt
# python -u train_word.py --num-lines 10000 --fast 2>&1 | tee log_word_fast_10k.txt
# python -u train_word.py --num-lines 10000 --vocab-size 40960 2>&1 | tee log_word_10k_40k.txt
# python -u train_word.py --num-lines 100000 --tau 0.8 --blowup 2>&1 | tee log_word_100k.txt
# python -u train_word.py --num-lines 1000000 --filepath data/other.jsonl 2>&1 | tee log_word_1M.txt
# python -u train_word.py  --num-lines 30000 --vocab-size 131072 --recalc 0 --filepath data/en_English_30000.jsonl --checkpoint-iterations 200000 --fast --save-pretokens save_pretokens/en_pretokens_30000.tsv --output-prefix ./save_pretokens