# Copyright 2026-present Kensho Technologies, LLC.
# Train second-pass BoundlessBPE model
# This loads the first-pass word model and trains supermerges that compete with regular merges
# Final model will have exactly vocab_size operations (same as first-pass model)
#
# Usage: python -u train_boundless.py --num-lines 10000 2>&1 | tee log_boundless_10k.txt

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
        from boundlessbpe import FastBoundlessBpeTrainer
        assert FastBoundlessBpeTrainer is not None, "Rust extension not available - run 'maturin develop --release' first"
        tokenizer = FastBoundlessBpeTrainer(
            script_specific_regex=ss_regex,
            script_specific_scripts=ss_scripts,
        )
    else:
        from boundlessbpe.train import BoundlessBpeTrainer
        tokenizer = BoundlessBpeTrainer(pretokenizer)

    # Print training parameters
    print("num_lines:", args.num_lines)
    print("vocab_size:", args.vocab_size)
    print("first_pass_tau:", args.first_pass_tau)
    print("blowup:", args.blowup)
    print("max_bytes:", args.max_bytes)
    print("recalc:", args.recalc)
    print("checkpoint_iterations:", args.checkpoint_iterations)
    print("fast:", args.fast)
    print("simple:", args.simple)
    print("save_pretokens:", args.save_pretokens)
    print("greedy_split:", args.greedy_split)
    print("min_count:", args.min_count)
    print("max_ngram_len:", args.max_ngram_len)
    assert args.recalc >= 0

    impl = "fast" if args.fast else "py"
    word_impl = "fast" if args.fast_word_model else "py"

    # Construct word model path (or use explicit override)
    if args.word_model:
        word_model_file = args.word_model
    else:
        word_model_file = f"{args.output_prefix}_{args.num_lines}_{args.max_bytes}_{args.vocab_size}_{args.first_pass_tau}_{int(args.blowup)}_{word_impl}_word_final.model"
    print("word_model_file:", word_model_file)

    # BoundlessBPE second pass: no deletions (tau=1.1)
    tau = 1.1
    outprefix = f"{args.output_prefix}_{args.num_lines}_{args.max_bytes}_{args.vocab_size}_{tau}_{int(args.blowup)}_{impl}_boundlessbpe"
    print("outprefix:", outprefix)

    # Train BoundlessBPE (second pass with competition)
    tokenizer.train(args.filepath, outprefix, args.num_lines, args.recalc, word_model_file, args.max_bytes,
                    args.checkpoint_iterations, args.verbose, args.progress_interval,
                    save_pretokens=args.save_pretokens,
                    greedy_split=args.greedy_split, min_count=args.min_count,
                    max_ngram_len=args.max_ngram_len)

    # Register special tokens
    tokenizer.register_special_tokens(["<|endoftext|>"])

    print("saving unified BoundlessBPE model")
    tokenizer.save(outprefix + "_final")

    # Test loading with Tokenizer
    print("loading unified model")
    from boundlessbpe.inference import Tokenizer
    tok = Tokenizer()
    tok.load(outprefix + "_final.model")

    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train second-pass BoundlessBPE model")
    parser.add_argument("--num-lines", type=int, required=True, help="Number of documents to read from training file")
    parser.add_argument("--vocab-size", type=int, default=131072, help="Word model vocabulary size (default: 131072)")
    parser.add_argument("--first-pass-tau", type=float, default=1.1, help="Tau used in first-pass word model (default: 1.1)")
    parser.add_argument("--blowup", action="store_true", help="Blowup mode used in first-pass word model")
    parser.add_argument("--max-bytes", type=int, default=10_000_000_000, help="Maximum bytes to process (default: 10GB)")
    parser.add_argument("--recalc", type=int, default=8192, help="Verification frequency (default: 8192, 0 to disable)")
    parser.add_argument("--filepath", type=str, default="data/minipile.jsonl", help="Path to training data (default: data/minipile.jsonl)")
    parser.add_argument("--word-model", type=str, default=None, help="Explicit path to word model file (overrides auto-constructed path)")
    parser.add_argument("--checkpoint-iterations", type=int, default=8192, help="Save checkpoint every N iterations (default: 8192)")
    parser.add_argument("--fast", action="store_true", help="Use Rust implementation for training")
    parser.add_argument("--simple", action="store_true", help="Use standard GPT4O regex without script-specific pretokenization")
    parser.add_argument("--output-prefix", type=str, default="./models/twopass", help="Output path prefix for model files (default: ./models/twopass)")
    parser.add_argument("--fast-word-model", action="store_true", help="Word model was trained with --fast (affects filename lookup)")
    parser.add_argument("--save-pretokens", type=str, default=None, help="Save superword pretokenization data to this file (TSV format)")
    parser.add_argument("--greedy-split", action="store_true", help="Apply n-gram greedy split before training")
    parser.add_argument("--min-count", type=int, default=15, help="Minimum count floor for n-gram counting (default: 15)")
    parser.add_argument("--max-ngram-len", type=int, default=30, help="Maximum n-gram length (default: 30)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed diagnostics (per-document progress, pretokenization summaries, timing breakdowns)")
    parser.add_argument("--progress-interval", type=int, default=None, help="Print a progress row every N merges. Default: 1 if --verbose else 1024. 0 disables progress rows.")
    main(parser.parse_args())


# Example commands:
# python -u train_boundless.py --num-lines 10000 2>&1 | tee log_boundless_10k.txt
# python -u train_boundless.py --num-lines 10000 --fast 2>&1 | tee log_boundless_fast_10k.txt
# python -u train_boundless.py --num-lines 10000 --vocab-size 40960 2>&1 | tee log_boundless_10k_40k.txt
# python -u train_boundless.py --num-lines 100000 --word-model ./models/my_word_model.model 2>&1 | tee log_boundless_100k.txt
# python -u train_boundless.py --num-lines 1000000 --filepath data/other.jsonl 2>&1 | tee log_boundless_1M.txt
