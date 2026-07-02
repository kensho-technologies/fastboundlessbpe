// Copyright 2026-present Kensho Technologies, LLC.
use std::time::Instant;
use std::io::{BufRead, BufReader};
use std::fs::File;

use ahash::{AHashMap, AHashSet};
use priority_queue::PriorityQueue;

use crate::byte_encoding::ByteEncoder;
use crate::error::{TokenizerError, TokenizerResult};
use crate::inference_data::InferenceData;
use crate::pretokenize::Pretokenizer;
use crate::tokenizer::Tokenizer;
use crate::vocabulary::Vocabulary;

// ---------------------------------------------------------------------------
// IndexGenerator
// ---------------------------------------------------------------------------

struct IndexGenerator {
    next_index: i32,
}

impl IndexGenerator {
    fn new() -> Self {
        Self { next_index: 0 }
    }
    fn get_next_index(&mut self) -> i32 {
        let idx = self.next_index;
        self.next_index += 1;
        idx
    }
    fn reset(&mut self) {
        self.next_index = 0;
    }
}

// ---------------------------------------------------------------------------
// Priority type for the pair heap
// ---------------------------------------------------------------------------
// Python uses heapdict with priority = (-both_unlocked_as_int, -count, pair).
// `priority-queue` is a max-heap; wrapping in Reverse gives min-heap.
// We use a comparable tuple type.

/// Priority stored on the heap: (neg_both_unlocked, neg_count, pair).
/// Ordered ascending, so Reverse gives us the item with the "smallest" tuple first
/// matching Python's heapdict which is a min-heap.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PairPriority {
    neg_both_unlocked: i32,
    neg_count: i64,
    pair: (Vec<u8>, Vec<u8>),
}

impl PartialOrd for PairPriority {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PairPriority {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // We want the *smallest* tuple to win in the max-heap,
        // so we invert: Reverse ordering so the PQ pops the min.
        other
            .neg_both_unlocked
            .cmp(&self.neg_both_unlocked)
            .then_with(|| other.neg_count.cmp(&self.neg_count))
            .then_with(|| other.pair.cmp(&self.pair))
    }
}

// ---------------------------------------------------------------------------
// Operation type for word operations list
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum OpType {
    Merge,
    Delete,
}

#[derive(Debug, Clone)]
enum OpData {
    MergeData {
        pair: (Vec<u8>, Vec<u8>),
        c_ab: i64,
        unlocked_flag: i32,
    },
    DeleteData {
        token: Vec<u8>,
    },
}

// ---------------------------------------------------------------------------
// BaseBpeTrainer
// ---------------------------------------------------------------------------

pub struct BaseBpeTrainer {
    pub inf_data: InferenceData,
    index_gen: IndexGenerator,
    initial_vocab: Vec<Vec<u8>>,
    encoder: ByteEncoder,

    // Training state
    pub vocab: Option<Vocabulary>,
    single_counts: AHashMap<Vec<u8>, i64>,
    pair_counts: PriorityQueue<(Vec<u8>, Vec<u8>), PairPriority>,
    token_to_pairs: AHashMap<Vec<u8>, AHashSet<(Vec<u8>, Vec<u8>)>>,
    token_locations: AHashMap<Vec<u8>, AHashSet<usize>>,
    // `unlocked` maps are stored explicitly; missing keys have a default
    unlocked: AHashMap<Vec<u8>, bool>,
    unlocked_default: bool, // true for BPE, false for super
    whole_words: i64,

    // Supermerge training state
    word_model: Option<Tokenizer>,
    word_operations_list: Vec<(i32, OpType, OpData)>,
    current_word_op_idx: usize,
    target_vocab_size: i32,
    pending_special_tokens: Vec<String>,

    // Static helper set
    single_bytes: AHashSet<Vec<u8>>,
}

impl BaseBpeTrainer {
    pub fn new(pretokenizer: Option<Pretokenizer>) -> TokenizerResult<Self> {
        let pretok = match pretokenizer {
            Some(p) => p,
            None => Pretokenizer::new(None, None, None, None)?,
        };
        let inf_data = InferenceData::create_for_training(pretok);

        // Build initial vocab (243 valid UTF-8 single bytes)
        let initial_vocab: Vec<Vec<u8>> = (0u8..192)
            .chain(194u8..245)
            .map(|b| vec![b])
            .collect();

        let single_bytes: AHashSet<Vec<u8>> = (0u8..=255).map(|b| vec![b]).collect();

        Ok(Self {
            inf_data,
            index_gen: IndexGenerator::new(),
            initial_vocab,
            encoder: ByteEncoder::new(),
            vocab: None,
            single_counts: AHashMap::new(),
            pair_counts: PriorityQueue::new(),
            token_to_pairs: AHashMap::new(),
            token_locations: AHashMap::new(),
            unlocked: AHashMap::new(),
            unlocked_default: true,
            whole_words: 0,
            word_model: None,
            word_operations_list: Vec::new(),
            current_word_op_idx: 0,
            target_vocab_size: 0,
            pending_special_tokens: Vec::new(),
            single_bytes,
        })
    }

    fn is_unlocked(&self, tok: &[u8]) -> bool {
        self.unlocked
            .get(tok)
            .copied()
            .unwrap_or(self.unlocked_default)
    }

    fn reset_training_state(&mut self, pretokenizer: Option<Pretokenizer>) -> TokenizerResult<()> {
        self.index_gen.reset();
        if let Some(p) = pretokenizer {
            self.inf_data = InferenceData::create_for_training(p);
        } else {
            // re-create with the same pretokenizer config
            let pretok = Pretokenizer::new(
                self.inf_data.pretokenizer.main_pattern_str.as_deref(),
                self.inf_data.pretokenizer.script_specific_pattern_str.as_deref(),
                self.inf_data.pretokenizer.script_specific_scripts.as_deref(),
                self.inf_data.pretokenizer.merge_pattern_str.as_deref(),
            )?;
            self.inf_data = InferenceData::create_for_training(pretok);
        }
        self.vocab = None;
        self.single_counts = AHashMap::new();
        self.pair_counts = PriorityQueue::new();
        self.token_to_pairs = AHashMap::new();
        self.token_locations = AHashMap::new();
        self.unlocked = AHashMap::new();
        self.unlocked_default = true;
        self.whole_words = 0;
        self.word_model = None;
        self.word_operations_list = Vec::new();
        self.current_word_op_idx = 0;
        self.target_vocab_size = 0;
        self.pending_special_tokens = Vec::new();
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Pretokenization
    // -----------------------------------------------------------------------

    /// First-pass pretokenization: read JSONL, pretokenize, aggregate, split to bytes.
    fn pretokenize(
        &self,
        filepath: &str,
        num_lines: usize,
        max_bytes: usize,
        save_pretokens: Option<&str>,
        verbose: bool,
    ) -> TokenizerResult<(Vec<Vec<Vec<u8>>>, Vec<i64>)> {
        let start = Instant::now();
        assert!(!self.inf_data.is_super);

        let mut chunk_tally: AHashMap<String, i64> = AHashMap::new();
        let mut total_bytes: usize = 0;
        let mut total_chars: usize = 0;

        let f = File::open(filepath)
            .map_err(|e| TokenizerError::IoError(e))?;
        let reader = BufReader::new(f);

        for (i, line_result) in reader.lines().enumerate() {
            if i >= num_lines {
                break;
            }
            if verbose && i % 10000 == 0 {
                println!(
                    "document {} {:.3} {} {}",
                    i,
                    start.elapsed().as_secs_f64(),
                    total_chars,
                    total_bytes
                );
            }
            let line = line_result.map_err(|e| TokenizerError::IoError(e))?;
            let json: serde_json::Value = serde_json::from_str(line.trim_end())
                .map_err(|e| TokenizerError::JsonError(e))?;
            let text = json["text"]
                .as_str()
                .ok_or_else(|| TokenizerError::ModelError("missing 'text' field".into()))?;

            for chunk in self.inf_data.pretokenizer.pretokenize(text) {
                *chunk_tally.entry(chunk).or_insert(0) += 1;
            }

            total_chars += text.len();
            total_bytes += text.len(); // approximate (UTF-8)

            if total_bytes >= max_bytes {
                if verbose {
                    println!("at max_bytes {} {} {} {}", i, max_bytes, total_chars, total_bytes);
                }
                break;
            }
        }

        // Sort descending by count
        let mut cnt_chk: Vec<(i64, Vec<u8>)> = chunk_tally
            .into_iter()
            .map(|(chk, cnt)| (cnt, chk.into_bytes()))
            .collect();
        cnt_chk.sort_by(|a, b| b.0.cmp(&a.0));

        if cnt_chk.is_empty() {
            println!("WARNING: no pre-tokenization chunks found!");
            return Ok((vec![], vec![]));
        }

        // save them to a file for use in another project
        if let Some(path) = save_pretokens {
            use std::io::Write;
            if let Some(parent) = std::path::Path::new(path).parent() {
                std::fs::create_dir_all(parent).map_err(|e| TokenizerError::IoError(e))?;
            }
            let mut out = std::io::BufWriter::new(
                File::create(path).map_err(|e| TokenizerError::IoError(e))?,
            );
            for (cnt, chk) in &cnt_chk {
                writeln!(out, "{}\t{}", cnt, self.encoder.from_bytes(chk))
                    .map_err(|e| TokenizerError::IoError(e))?;
            }
            println!("Saved {} pretokens to {}", cnt_chk.len(), path);
        }

        if verbose {
            println!("number of pre-tokenization chunks: {}", cnt_chk.len());
            println!("top 10 pre-tokenization chunks:");
            for j in 0..cnt_chk.len().min(10) {
                println!("{} {:?}", j, cnt_chk[j]);
            }
            println!();
        }

        let text_counts: Vec<i64> = cnt_chk.iter().map(|(cnt, _)| *cnt).collect();
        let text_chunks: Vec<Vec<Vec<u8>>> = cnt_chk
            .iter()
            .map(|(_, chk)| chk.iter().map(|&b| vec![b]).collect())
            .collect();

        if verbose {
            println!(
                "pretokenization time: {:.3} {} {}",
                start.elapsed().as_secs_f64(),
                text_counts.len(),
                text_counts.iter().sum::<i64>(),
            );
            println!();
        }

        Ok((text_chunks, text_counts))
    }

    /// Second-pass pretokenization: apply word model, find supermerge runs.
    fn pretokenize_super(
        &mut self,
        filepath: &str,
        num_lines: usize,
        max_bytes: usize,
        save_pretokens: Option<&str>,
        verbose: bool,
    ) -> TokenizerResult<(Vec<Vec<Vec<u8>>>, Vec<i64>)> {
        let start = Instant::now();
        assert!(self.inf_data.is_super);

        let word_model = self
            .word_model
            .as_ref()
            .expect("word_model must be loaded");

        let mut total_bytes: usize = 0;
        let mut total_chars: usize = 0;
        let mut counts: AHashMap<Vec<Vec<u8>>, i64> = AHashMap::new();

        // Shortcut: precompute which pretokens (as raw bytes) are both
        // single-token after merges (reachable) AND in possible_superwords.
        // This lets us skip the expensive per-document fast_merge_delete entirely.
        let reachable = word_model.reachable_vocab.as_ref()
            .expect("reachable_vocab must be initialized");
        let possible = word_model.possible_superwords.as_ref()
            .expect("possible_superwords must be initialized");
        let supermerge_eligible: AHashSet<Vec<u8>> = reachable
            .intersection(possible)
            .cloned()
            .collect();
        if verbose {
            println!(
                "supermerge_eligible: {} tokens (reachable={}, possible_superwords={})",
                supermerge_eligible.len(),
                reachable.len(),
                possible.len()
            );
        }

        let f = File::open(filepath)
            .map_err(|e| TokenizerError::IoError(e))?;
        let reader = BufReader::new(f);

        for (i, line_result) in reader.lines().enumerate() {
            if i >= num_lines {
                break;
            }
            if verbose && i % 10000 == 0 {
                println!(
                    "document {} {:.3} {} {}",
                    i,
                    start.elapsed().as_secs_f64(),
                    total_chars,
                    total_bytes
                );
            }
            let line = line_result.map_err(|e| TokenizerError::IoError(e))?;
            let json: serde_json::Value = serde_json::from_str(line.trim_end())
                .map_err(|e| TokenizerError::JsonError(e))?;
            let text = json["text"]
                .as_str()
                .ok_or_else(|| TokenizerError::ModelError("missing 'text' field".into()))?;

            let document_bytes: Vec<Vec<u8>> = self
                .inf_data
                .pretokenizer
                .pretokenize(text)
                .into_iter()
                .map(|chunk| chunk.into_bytes())
                .collect();

            // Find runs of consecutive eligible pretokens directly,
            // skipping the full merge replay.
            let n = document_bytes.len();
            let mut idx = 0;
            while idx < n {
                if !supermerge_eligible.contains(&document_bytes[idx]) {
                    idx += 1;
                    continue;
                }
                // start of a run
                let mut end = idx + 1;
                while end < n && supermerge_eligible.contains(&document_bytes[end]) {
                    end += 1;
                }
                // need at least 2 consecutive eligible pretokens
                if end - idx >= 2 {
                    let superword_run: Vec<Vec<u8>> = document_bytes[idx..end].to_vec();
                    *counts.entry(superword_run).or_insert(0) += 1;
                }
                idx = end;
            }

            total_chars += text.len();
            total_bytes += text.len();

            if total_bytes >= max_bytes {
                if verbose {
                    println!("at max_bytes {} {} {} {}", i, max_bytes, total_chars, total_bytes);
                }
                break;
            }
        }

        // Sort descending by count
        let mut cnt_chk: Vec<(i64, Vec<Vec<u8>>)> = counts
            .into_iter()
            .map(|(chk, cnt)| (cnt, chk))
            .collect();
        cnt_chk.sort_by(|a, b| b.0.cmp(&a.0));

        if cnt_chk.is_empty() {
            println!("WARNING: no superword pre-tokenization chunks found!");
            return Ok((vec![], vec![]));
        }

        // save them to a file for use in another project
        if let Some(path) = save_pretokens {
            use std::io::Write;
            if let Some(parent) = std::path::Path::new(path).parent() {
                std::fs::create_dir_all(parent).map_err(|e| TokenizerError::IoError(e))?;
            }
            let mut out = std::io::BufWriter::new(
                File::create(path).map_err(|e| TokenizerError::IoError(e))?,
            );
            for (cnt, chk) in &cnt_chk {
                let tokens: Vec<String> = chk.iter().map(|tok| self.encoder.from_bytes(tok)).collect();
                writeln!(out, "{}\t{}", cnt, tokens.join("\t"))
                    .map_err(|e| TokenizerError::IoError(e))?;
            }
            println!("Saved {} superword pretokens to {}", cnt_chk.len(), path);
        }

        if verbose {
            println!(
                "number of superword pre-tokenization chunks: {}",
                cnt_chk.len()
            );
            println!("top 100 superword pre-tokenization chunks:");
            for j in 0..cnt_chk.len().min(100) {
                println!("{} {:?}", j, cnt_chk[j]);
            }
            println!();
        }

        let text_counts: Vec<i64> = cnt_chk.iter().map(|(cnt, _)| *cnt).collect();
        let text_chunks: Vec<Vec<Vec<u8>>> = cnt_chk.into_iter().map(|(_, chk)| chk).collect();

        if verbose {
            println!(
                "pretokenization time: {:.3} {} {}",
                start.elapsed().as_secs_f64(),
                text_counts.len(),
                text_counts.iter().sum::<i64>(),
            );
            println!();
        }

        Ok((text_chunks, text_counts))
    }

    // -----------------------------------------------------------------------
    // Pair counting
    // -----------------------------------------------------------------------

    /// Count adjacent pairs in token list with overlap handling.
    fn get_stats(
        tokens: &[Vec<u8>],
        counts: &mut AHashMap<(Vec<u8>, Vec<u8>), i64>,
        multiplier: i64,
    ) {
        let mut prev_pair: Option<(Vec<u8>, Vec<u8>)> = None;
        for i in 0..tokens.len().saturating_sub(1) {
            let pair = (tokens[i].clone(), tokens[i + 1].clone());
            let same_as_previous = pair.0 == pair.1
                && prev_pair.as_ref() == Some(&pair);
            if !same_as_previous {
                *counts.entry(pair.clone()).or_insert(0) += multiplier;
                prev_pair = Some(pair);
            } else {
                prev_pair = None;
            }
        }
    }

    /// Optimized incremental pair count updates for non-overlapping merge case.
    /// Returns number of merges found. All changes are written to `counts` as deltas.
    fn get_stats_faster(
        &self,
        first: &[u8],
        second: &[u8],
        tokens: &[Vec<u8>],
        counts: &mut AHashMap<(Vec<u8>, Vec<u8>), i64>,
        multiplier: i64,
    ) -> i64 {
        assert!(self.is_unlocked(first));
        assert!(self.is_unlocked(second));

        let mut combined = first.to_vec();
        combined.extend_from_slice(second);

        let mut merge_cnt: i64 = 0;
        let n = tokens.len();

        let mut i = 0;
        while i < n.saturating_sub(1) {
            if tokens[i] == first && tokens[i + 1] == second {
                // Find how many repeated copies of (first, second)
                let mut k = i;
                while k + 3 < n && tokens[k + 2] == first && tokens[k + 3] == second {
                    k += 2;
                }
                // copies from (i,i+1) to (k,k+1)
                assert_eq!(tokens[k].as_slice(), first);
                assert_eq!(tokens[k + 1].as_slice(), second);

                let copies_raw = (k + 1) - i + 1;
                assert_eq!(copies_raw % 2, 0);
                let copies = (copies_raw / 2) as i64;
                assert!(copies >= 1);

                merge_cnt += copies;

                // (first, second) count always goes down
                *counts
                    .entry((first.to_vec(), second.to_vec()))
                    .or_insert(0) -= multiplier * copies;

                // Handle (combined, combined) pairs
                if copies % 2 == 0 {
                    *counts
                        .entry((combined.clone(), combined.clone()))
                        .or_insert(0) += multiplier * (copies / 2);
                } else if copies > 1 {
                    *counts
                        .entry((combined.clone(), combined.clone()))
                        .or_insert(0) += multiplier * ((copies - 1) / 2);
                }

                // Reduce (second, first) counts between repeated pairs
                if copies > 1 {
                    *counts
                        .entry((second.to_vec(), first.to_vec()))
                        .or_insert(0) -= multiplier * (copies - 1);
                }

                // Handle previous token
                if i >= 1 {
                    let prev = &tokens[i - 1];
                    // New pair (prev, combined)
                    *counts
                        .entry((prev.clone(), combined.clone()))
                        .or_insert(0) += multiplier;

                    // Check run of `first` to the left
                    let mut j = i;
                    while j >= 1 && tokens[j - 1] == first {
                        j -= 1;
                    }
                    let runlen = i - j + 1;

                    if runlen == 1 || runlen % 2 == 0 {
                        *counts
                            .entry((prev.clone(), first.to_vec()))
                            .or_insert(0) -= multiplier;
                    }
                }

                // Handle token after the run
                if k + 2 < tokens.len() {
                    let after = &tokens[k + 2];
                    // New pair (combined, after)
                    *counts
                        .entry((combined.clone(), after.clone()))
                        .or_insert(0) += multiplier;

                    // Check run of `second` to the right
                    let mut j = k + 1;
                    while j + 1 < tokens.len() && tokens[j + 1] == second {
                        j += 1;
                    }
                    let runlen = j - (k + 1) + 1;

                    if runlen == 1 || runlen % 2 == 0 {
                        *counts
                            .entry((second.to_vec(), after.clone()))
                            .or_insert(0) -= multiplier;
                    }
                }

                i = k + 2;
            } else {
                i += 1;
            }
        }

        merge_cnt
    }

    // -----------------------------------------------------------------------
    // State initialization
    // -----------------------------------------------------------------------

    fn _calc_pair_counts(
        &self,
        text_chunks: &[Vec<Vec<u8>>],
        text_counts: &[i64],
    ) -> (
        AHashMap<(Vec<u8>, Vec<u8>), i64>,
        PriorityQueue<(Vec<u8>, Vec<u8>), PairPriority>,
    ) {
        let mut pair_counts_dict: AHashMap<(Vec<u8>, Vec<u8>), i64> = AHashMap::new();
        for (tokens, &cnt) in text_chunks.iter().zip(text_counts.iter()) {
            Self::get_stats(tokens, &mut pair_counts_dict, cnt);
        }

        let mut pair_counts_heap = PriorityQueue::new();
        for (pair, &count) in &pair_counts_dict {
            let both_unlocked = self.is_unlocked(&pair.0) && self.is_unlocked(&pair.1);
            let priority = PairPriority {
                neg_both_unlocked: -(both_unlocked as i32),
                neg_count: -count,
                pair: pair.clone(),
            };
            pair_counts_heap.push(pair.clone(), priority);
        }

        (pair_counts_dict, pair_counts_heap)
    }

    fn initial_pair_counts(
        &mut self,
        text_chunks: &[Vec<Vec<u8>>],
        text_counts: &[i64],
    ) {
        let (_, heap) = self._calc_pair_counts(text_chunks, text_counts);
        self.pair_counts = heap;

        // token_to_pairs initialization removed — unlocking mechanism no longer needed.
        // The count-based competition ensures parents are created before their supermerges.
    }

    fn _calc_single_counts(
        text_chunks: &[Vec<Vec<u8>>],
        text_counts: &[i64],
    ) -> AHashMap<Vec<u8>, i64> {
        let mut single_counts: AHashMap<Vec<u8>, i64> = AHashMap::new();
        for (tokens, &cnt) in text_chunks.iter().zip(text_counts.iter()) {
            for tok in tokens {
                *single_counts.entry(tok.clone()).or_insert(0) += cnt;
            }
        }
        single_counts
    }

    fn initial_single_counts(
        &mut self,
        text_chunks: &[Vec<Vec<u8>>],
        text_counts: &[i64],
    ) {
        self.single_counts = Self::_calc_single_counts(text_chunks, text_counts);
    }

    fn _calc_token_locations(text_chunks: &[Vec<Vec<u8>>]) -> AHashMap<Vec<u8>, AHashSet<usize>> {
        let mut token_locations: AHashMap<Vec<u8>, AHashSet<usize>> = AHashMap::new();
        for (chunk_idx, tokens) in text_chunks.iter().enumerate() {
            let unique: AHashSet<&Vec<u8>> = tokens.iter().collect();
            for token in unique {
                token_locations
                    .entry(token.clone())
                    .or_default()
                    .insert(chunk_idx);
            }
        }
        token_locations
    }

    fn initial_token_locations(&mut self, text_chunks: &[Vec<Vec<u8>>]) {
        self.token_locations = Self::_calc_token_locations(text_chunks);
    }

    fn _calc_whole_words(text_chunks: &[Vec<Vec<u8>>], text_counts: &[i64]) -> i64 {
        let mut whole_words: i64 = 0;
        for (tokens, &cnt) in text_chunks.iter().zip(text_counts.iter()) {
            if tokens.len() == 1 {
                whole_words += cnt as i64;
            }
        }
        whole_words
    }

    fn initial_whole_words(
        &mut self,
        text_chunks: &[Vec<Vec<u8>>],
        text_counts: &[i64],
    ) {
        self.whole_words = Self::_calc_whole_words(text_chunks, text_counts);
    }

    fn initial_counts(
        &mut self,
        text_chunks: &[Vec<Vec<u8>>],
        text_counts: &[i64],
    ) {
        self.initial_pair_counts(text_chunks, text_counts);
        self.initial_single_counts(text_chunks, text_counts);
        self.initial_whole_words(text_chunks, text_counts);
        self.initial_token_locations(text_chunks);
    }

    // -----------------------------------------------------------------------
    // Verification
    // -----------------------------------------------------------------------

    fn verify_pair_counts(
        &self,
        text_chunks: &[Vec<Vec<u8>>],
        text_counts: &[i64],
    ) {
        let (from_scratch, _) = self._calc_pair_counts(text_chunks, text_counts);
        let current: AHashMap<(Vec<u8>, Vec<u8>), i64> = self
            .pair_counts
            .iter()
            .map(|(pair, _)| (pair.clone(), self.get_pair_count(pair)))
            .collect();
        verify_maps(&from_scratch, &current);
    }

    fn verify_single_counts(
        &self,
        text_chunks: &[Vec<Vec<u8>>],
        text_counts: &[i64],
    ) {
        let from_scratch = Self::_calc_single_counts(text_chunks, text_counts);
        verify_maps(&from_scratch, &self.single_counts);
    }

    fn verify_whole_words(
        &self,
        text_chunks: &[Vec<Vec<u8>>],
        text_counts: &[i64],
    ) {
        let from_scratch = Self::_calc_whole_words(text_chunks, text_counts);
        assert_eq!(
            self.whole_words, from_scratch,
            "whole_words mismatch: {} vs {}",
            self.whole_words, from_scratch
        );
    }

    #[allow(dead_code)]
    fn verify_unlocked(&self) {
        for (pair, priority) in self.pair_counts.iter() {
            assert_eq!(pair, &priority.pair);
            let actual_both_unlocked =
                self.is_unlocked(&pair.0) && self.is_unlocked(&pair.1);
            let stored_both_unlocked = priority.neg_both_unlocked == -1;
            assert_eq!(
                actual_both_unlocked, stored_both_unlocked,
                "Unlocked status mismatch for pair ({:?}, {:?})",
                pair.0, pair.1
            );
        }
    }

    #[allow(dead_code)]
    fn verify_token_to_pairs(&self) {
        if !(self.inf_data.is_super && !self.inf_data.superbpe_mode) {
            return;
        }
        let mut expected: AHashMap<Vec<u8>, AHashSet<(Vec<u8>, Vec<u8>)>> = AHashMap::new();
        for (pair, _) in self.pair_counts.iter() {
            expected.entry(pair.0.clone()).or_default().insert(pair.clone());
            expected.entry(pair.1.clone()).or_default().insert(pair.clone());
        }

        for (token, expected_pairs) in &expected {
            let actual_pairs = self
                .token_to_pairs
                .get(token)
                .cloned()
                .unwrap_or_default();
            assert_eq!(
                *expected_pairs, actual_pairs,
                "token_to_pairs mismatch for token {:?}",
                token
            );
        }
    }

    fn verify_token_locations(&self, text_chunks: &[Vec<Vec<u8>>]) {
        let expected = Self::_calc_token_locations(text_chunks);
        let all_tokens: AHashSet<&Vec<u8>> = expected
            .keys()
            .chain(self.token_locations.keys())
            .collect();
        for token in all_tokens {
            let expected_chunks = expected.get(token).cloned().unwrap_or_default();
            let actual_chunks = self.token_locations.get(token).cloned().unwrap_or_default();
            assert_eq!(
                expected_chunks, actual_chunks,
                "token_locations mismatch for token {:?}",
                token
            );
        }
    }

    fn verify_state(
        &self,
        text_chunks: &[Vec<Vec<u8>>],
        text_counts: &[i64],
    ) {
        if let Some(ref vocab) = self.vocab {
            vocab.verify_vocabulary();
        }
        self.verify_pair_counts(text_chunks, text_counts);
        self.verify_single_counts(text_chunks, text_counts);
        self.verify_whole_words(text_chunks, text_counts);
        // verify_unlocked and verify_token_to_pairs skipped — unlocking mechanism removed
        self.verify_token_locations(text_chunks);
        self.inf_data.verify_indices();
    }

    // -----------------------------------------------------------------------
    // Core operations
    // -----------------------------------------------------------------------

    fn get_pair_count(&self, pair: &(Vec<u8>, Vec<u8>)) -> i64 {
        match self.pair_counts.get(pair) {
            Some((_, priority)) => -priority.neg_count,
            None => 0,
        }
    }

    fn choose_best_pair(&self) -> (Option<(Vec<u8>, Vec<u8>)>, i64) {
        if self.pair_counts.is_empty() {
            return (None, -1);
        }

        let (pair, priority) = self.pair_counts.peek().unwrap();
        if priority.neg_both_unlocked != -1 {
            return (None, -1);
        }

        let count = -priority.neg_count;
        assert!(self.is_unlocked(&pair.0));
        assert!(self.is_unlocked(&pair.1));

        (Some(pair.clone()), count)
    }

    fn apply_pair_count_changes(
        &mut self,
        overall_change: &AHashMap<(Vec<u8>, Vec<u8>), i64>,
    ) {
        for (pair, &delta) in overall_change {
            if delta != 0 {
                let current_count = self.get_pair_count(pair);
                let new_count = current_count + delta;

                assert!(
                    new_count >= 0,
                    "Pair count went negative: {:?} {} + {} = {}",
                    pair,
                    current_count,
                    delta,
                    new_count
                );

                if new_count == 0 {
                    self.pair_counts.remove(pair);
                } else {
                    let both_unlocked =
                        self.is_unlocked(&pair.0) && self.is_unlocked(&pair.1);
                    let priority = PairPriority {
                        neg_both_unlocked: -(both_unlocked as i32),
                        neg_count: -new_count,
                        pair: pair.clone(),
                    };
                    // push_increase or change_priority
                    if self.pair_counts.get(pair).is_some() {
                        self.pair_counts.change_priority(pair, priority);
                    } else {
                        self.pair_counts.push(pair.clone(), priority);
                    }
                }
            }
        }
    }

    fn get_single_byte_cnt(&self) -> i64 {
        let mut cnt: i64 = 0;
        for tok in &self.single_bytes {
            if let Some(&c) = self.single_counts.get(tok) {
                cnt += c as i64;
            }
        }
        cnt
    }

    fn merge_and_update(
        &mut self,
        max_pair: &(Vec<u8>, Vec<u8>),
        text_chunks: &mut [Vec<Vec<u8>>],
        text_counts: &[i64],
    ) -> (i64, Option<Vec<u8>>) {
        let first = &max_pair.0;
        let second = &max_pair.1;

        assert!(self.is_unlocked(first));
        assert!(self.is_unlocked(second));

        let pair_count = self.get_pair_count(max_pair);
        assert!(pair_count > 0);

        let mut overall_change: AHashMap<(Vec<u8>, Vec<u8>), i64> = AHashMap::new();
        let mut total_merge_cnt: i64 = 0;
        let mut whole_word_increase: i64 = 0;
        let mut new_unlocked: Option<Vec<u8>> = None;

        let mut combined = first.clone();
        combined.extend_from_slice(second);

        // Get candidate chunks
        let candidates: Vec<usize> = if first == second {
            self.token_locations
                .get(first)
                .map(|s| s.iter().copied().collect())
                .unwrap_or_default()
        } else {
            let first_locs = self.token_locations.get(first);
            let second_locs = self.token_locations.get(second);
            match (first_locs, second_locs) {
                (Some(fl), Some(sl)) => fl.intersection(sl).copied().collect(),
                _ => vec![],
            }
        };

        // Fast path for is_super when first != second
        if first != second && self.inf_data.is_super {
            for chunk_idx in &candidates {
                let tokens = &text_chunks[*chunk_idx];
                let cnt = text_counts[*chunk_idx];

                let merge_cnt_before = self.get_stats_faster(
                    first,
                    second,
                    tokens,
                    &mut overall_change,
                    cnt,
                );

                if merge_cnt_before > 0 {
                    let before_len = tokens.len();

                    let (newtokens, merge_cnt) =
                        Tokenizer::merge(tokens, (first, second));

                    if merge_cnt_before != merge_cnt {
                        println!(
                            "warning merge_cnt different: {:?} {:?} {} {}",
                            first, second, merge_cnt_before, merge_cnt
                        );
                    }

                    text_chunks[*chunk_idx] = newtokens;
                    total_merge_cnt += merge_cnt * cnt;

                    let tokens = &text_chunks[*chunk_idx];
                    if tokens.len() == 1 && before_len > 1 {
                        whole_word_increase += cnt as i64;
                        new_unlocked = Some(combined.clone());
                    }

                    // Update token_locations
                    if !tokens.contains(first) {
                        if let Some(set) = self.token_locations.get_mut(first) {
                            set.remove(chunk_idx);
                            if set.is_empty() {
                                self.token_locations.remove(first);
                            }
                        }
                    }
                    if !tokens.contains(second) {
                        if let Some(set) = self.token_locations.get_mut(second) {
                            set.remove(chunk_idx);
                            if set.is_empty() {
                                self.token_locations.remove(second);
                            }
                        }
                    }
                    self.token_locations
                        .entry(combined.clone())
                        .or_default()
                        .insert(*chunk_idx);
                }
            }
        } else {
            // Slower path for regular BPE or first == second
            for chunk_idx in &candidates {
                let tokens = &text_chunks[*chunk_idx];
                let cnt = text_counts[*chunk_idx];

                let (newtokens, merge_cnt) =
                    Tokenizer::merge(tokens, (first, second));

                if merge_cnt > 0 {
                    let mut local_delta: AHashMap<(Vec<u8>, Vec<u8>), i64> = AHashMap::new();
                    Self::get_stats(tokens, &mut local_delta, -cnt);

                    let before_len = tokens.len();
                    text_chunks[*chunk_idx] = newtokens;
                    total_merge_cnt += merge_cnt * cnt;

                    let tokens = &text_chunks[*chunk_idx];
                    Self::get_stats(tokens, &mut local_delta, cnt);

                    for (pair, delta) in &local_delta {
                        if *delta != 0 {
                            let entry = overall_change.entry(pair.clone()).or_insert(0);
                            *entry += delta;
                            if *entry == 0 {
                                overall_change.remove(pair);
                            }
                        }
                    }

                    if tokens.len() == 1 && before_len > 1 {
                        whole_word_increase += cnt as i64;
                        new_unlocked = Some(combined.clone());
                    }

                    // Update token_locations
                    if !tokens.contains(first) {
                        if let Some(set) = self.token_locations.get_mut(first) {
                            set.remove(chunk_idx);
                            if set.is_empty() {
                                self.token_locations.remove(first);
                            }
                        }
                    }
                    if first != second && !tokens.contains(second) {
                        if let Some(set) = self.token_locations.get_mut(second) {
                            set.remove(chunk_idx);
                            if set.is_empty() {
                                self.token_locations.remove(second);
                            }
                        }
                    }
                    self.token_locations
                        .entry(combined.clone())
                        .or_default()
                        .insert(*chunk_idx);
                }
            }
        }

        // Unlock merged token for super training
        if self.inf_data.is_super {
            let merged_token = combined.clone();
            self.unlocked.insert(merged_token, true);
        }

        // Apply deltas to heap
        self.apply_pair_count_changes(&overall_change);

        // Should no longer have max_pair in heap
        assert!(
            !self.pair_counts.get(max_pair).is_some(),
            "max_pair should have been removed"
        );

        // Update single counts
        let merged_pair = combined;
        *self.single_counts.entry(merged_pair.clone()).or_insert(0) += total_merge_cnt;
        *self.single_counts.entry(first.clone()).or_insert(0) -= total_merge_cnt;
        *self.single_counts.entry(second.clone()).or_insert(0) -= total_merge_cnt;

        if self.single_counts.get(first).copied() == Some(0) {
            self.single_counts.remove(first);
        }
        if first != second && self.single_counts.get(second).copied() == Some(0) {
            self.single_counts.remove(second);
        }

        self.whole_words += whole_word_increase;

        (total_merge_cnt, new_unlocked)
    }

    fn delete_and_update(
        &mut self,
        bad_token: &[u8],
        text_chunks: &mut [Vec<Vec<u8>>],
        text_counts: &[i64],
    ) {
        let parts = self.inf_data.get_replacement_parts(bad_token).to_vec();
        let expected_cnt = self.single_counts.get(bad_token).copied().unwrap_or(0);

        let mut overall_change: AHashMap<(Vec<u8>, Vec<u8>), i64> = AHashMap::new();
        let mut total_delete_cnt: i64 = 0;
        let mut whole_word_increase: i64 = 0;

        let candidates: Vec<usize> = self
            .token_locations
            .get(bad_token)
            .map(|s| s.iter().copied().collect())
            .unwrap_or_default();

        for chunk_idx in &candidates {
            let tokens = &text_chunks[*chunk_idx];
            let cnt = text_counts[*chunk_idx];

            let mut local_delta: AHashMap<(Vec<u8>, Vec<u8>), i64> = AHashMap::new();
            Self::get_stats(tokens, &mut local_delta, -cnt);

            let before_len = tokens.len();

            let (newtokens, deletions) = Tokenizer::blow_up(tokens, bad_token, &parts);
            text_chunks[*chunk_idx] = newtokens;

            total_delete_cnt += deletions * cnt;

            let tokens = &text_chunks[*chunk_idx];
            Self::get_stats(tokens, &mut local_delta, cnt);

            for (pair, delta) in &local_delta {
                if *delta != 0 {
                    let entry = overall_change.entry(pair.clone()).or_insert(0);
                    *entry += delta;
                    if *entry == 0 {
                        overall_change.remove(pair);
                    }
                }
            }

            if before_len == 1 && tokens.len() > 1 {
                whole_word_increase -= cnt as i64;
            }

            // Update token_locations
            if let Some(set) = self.token_locations.get_mut(bad_token) {
                set.remove(chunk_idx);
                if set.is_empty() {
                    self.token_locations.remove(bad_token);
                }
            }
            for part in &parts {
                self.token_locations
                    .entry(part.clone())
                    .or_default()
                    .insert(*chunk_idx);
            }
        }

        assert_eq!(
            expected_cnt, total_delete_cnt,
            "delete count mismatch: expected {}, got {}",
            expected_cnt, total_delete_cnt
        );

        self.whole_words += whole_word_increase;

        self.apply_pair_count_changes(&overall_change);

        // Update single counts
        if self.single_counts.contains_key(bad_token) {
            *self.single_counts.get_mut(bad_token).unwrap() -= total_delete_cnt;
            assert_eq!(
                self.single_counts.get(bad_token).copied().unwrap_or(0),
                0
            );
            self.single_counts.remove(bad_token);

            for tok in &parts {
                *self.single_counts.entry(tok.clone()).or_insert(0) += total_delete_cnt;
            }
        }

        // Save the deletion event
        self.inf_data
            .deletions
            .insert(self.index_gen.get_next_index(), bad_token.to_vec());
    }

    #[allow(dead_code)]
    fn update_unlocked_token_priorities(&mut self, new_unlocked: &[u8]) {
        assert!(self.inf_data.is_super);
        assert!(!self.inf_data.superbpe_mode);
        assert!(self.is_unlocked(new_unlocked));

        let pairs_to_update: Vec<(Vec<u8>, Vec<u8>)> = self
            .token_to_pairs
            .get(new_unlocked)
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default();

        for pair in &pairs_to_update {
            // Only update if the other token is also unlocked — otherwise
            // both_unlocked stays false and the priority doesn't change.
            let other = if pair.0 == new_unlocked { &pair.1 } else { &pair.0 };
            if !self.is_unlocked(other) {
                continue;
            }
            let count = self.get_pair_count(pair);
            assert!(count > 0);
            let priority = PairPriority {
                neg_both_unlocked: -1, // both are unlocked
                neg_count: -count,
                pair: pair.clone(),
            };
            self.pair_counts.change_priority(pair, priority);
        }
    }

    fn merge_and_delete(
        &mut self,
        best_pair: &(Vec<u8>, Vec<u8>),
        i: usize,
        text_chunks: &mut [Vec<Vec<u8>>],
        text_counts: &[i64],
        print_row: bool,
        verbose: bool,
    ) -> (f64, f64, Option<Vec<u8>>) {
        let start_merge = Instant::now();

        let left = &best_pair.0;
        let right = &best_pair.1;

        assert!(self.is_unlocked(left));
        assert!(self.is_unlocked(right));

        let c_ab = self.get_pair_count(best_pair);
        let c_a = *self.single_counts.get(left).expect("left not in single_counts");
        let c_b = *self.single_counts.get(right).expect("right not in single_counts");

        let ios_a = c_ab as f64 / c_a as f64;
        let ios_b = c_ab as f64 / c_b as f64;

        let (total_merge_cnt, new_unlocked) =
            self.merge_and_update(best_pair, text_chunks, text_counts);

        assert_eq!(
            c_ab, total_merge_cnt,
            "merge count mismatch: {} vs {}",
            c_ab, total_merge_cnt
        );

        let merged_tok = {
            let mut t = left.clone();
            t.extend_from_slice(right);
            t
        };

        // Save merge rule
        self.inf_data.merges.insert(
            self.index_gen.get_next_index(),
            (best_pair.clone(), c_ab, if new_unlocked.is_some() { 1 } else { 0 }),
        );

        // Add to vocab
        let vocab = self.vocab.as_mut().expect("vocab must exist");
        if !vocab.contains(&merged_tok) {
            vocab.add(merged_tok.clone(), self.inf_data.is_super);
        } else if verbose {
            println!(
                "info: duplicate merge (token exists): {:?}",
                self.encoder.from_bytes(&merged_tok)
            );
        }

        if print_row {
            let left_tok = self.encoder.from_bytes(left);
            let right_tok = self.encoder.from_bytes(right);
            let nu = match &new_unlocked {
                Some(t) => self.encoder.from_bytes(t),
                None => String::new(),
            };
            println!(
                "*\t{}\t{}\tm\t{}\t{}\t{}\t{}\t{}\t{:.5}\t{:.5}\t{:.5}\t{}\t{}\t{}\t{}",
                i + 1,
                vocab.len(),
                left_tok,
                right_tok,
                c_ab,
                c_a,
                c_b,
                ios_a,
                ios_b,
                start_merge.elapsed().as_secs_f64(),
                self.pair_counts.len(),
                self.get_single_byte_cnt(),
                self.whole_words,
                nu
            );
        }

        let merge_time = start_merge.elapsed().as_secs_f64();
        let start_delete = Instant::now();

        // Check for deletions
        for &(ios, ref tok, ref direction) in
            &[(ios_a, left.clone(), "left"), (ios_b, right.clone(), "right")]
        {
            if tok.len() > 1 && ios >= self.inf_data.tau {
                let start_this_delete = Instant::now();

                // Populate deletion_parts before calling delete_and_update
                if !self.inf_data.deletion_parts.contains_key(tok) {
                    if self.inf_data.blowup {
                        let parts: Vec<Vec<u8>> = tok.iter().map(|&b| vec![b]).collect();
                        self.inf_data.deletion_parts.insert(tok.clone(), parts);
                    } else {
                        let mut parts = None;
                        for ((left_m, right_m), _cnt_m, _unlocked_flag_m) in self.inf_data.merges.values() {
                            let mut combined = left_m.clone();
                            combined.extend_from_slice(right_m);
                            if combined == *tok {
                                parts = Some(vec![left_m.clone(), right_m.clone()]);
                                break;
                            }
                        }
                        let parts = parts.expect("couldn't find merge for deleted token");
                        self.inf_data.deletion_parts.insert(tok.clone(), parts);
                    }
                }

                self.delete_and_update(tok, text_chunks, text_counts);

                let vocab = self.vocab.as_mut().expect("vocab must exist");
                if vocab.contains(tok) {
                    vocab.delete(tok);
                }

                if print_row {
                    let nu = match &new_unlocked {
                        Some(t) => self.encoder.from_bytes(t),
                        None => String::new(),
                    };
                    println!(
                        "*\t{}\t{}\td\t{}\t{}\t{}\t{}\t{}\t{:.5}\t{:.5}\t{:.5}\t{}\t{}\t{}\t{}",
                        i + 1,
                        vocab.len(),
                        self.encoder.from_bytes(tok),
                        direction,
                        c_ab,
                        c_a,
                        c_b,
                        ios,
                        self.inf_data.tau,
                        start_this_delete.elapsed().as_secs_f64(),
                        self.pair_counts.len(),
                        self.get_single_byte_cnt(),
                        self.whole_words,
                        nu
                    );
                }
            }
        }

        let delete_time = start_delete.elapsed().as_secs_f64();

        (merge_time, delete_time, new_unlocked)
    }

    fn _process_pending_deletions(&mut self) {
        let vocab = self.vocab.as_mut().expect("vocab must exist");
        while self.current_word_op_idx < self.word_operations_list.len() {
            let (_idx, ref op_type, ref data) = self.word_operations_list[self.current_word_op_idx];
            match op_type {
                OpType::Delete => {
                    if let OpData::DeleteData { token } = data {
                        if vocab.contains(token) {
                            vocab.delete(token);
                        }
                    }
                    self.current_word_op_idx += 1;
                }
                OpType::Merge => break,
            }
        }
    }

    fn _get_next_regular_merge(
        &mut self,
    ) -> (Option<i32>, Option<(Vec<u8>, Vec<u8>)>, i64, i32) {
        self._process_pending_deletions();

        if self.current_word_op_idx >= self.word_operations_list.len() {
            return (None, None, -1, 0);
        }

        let (idx, ref _op_type, ref data) = self.word_operations_list[self.current_word_op_idx];
        if let OpData::MergeData {
            pair,
            c_ab,
            unlocked_flag,
        } = data
        {
            (Some(idx), Some(pair.clone()), *c_ab, *unlocked_flag)
        } else {
            panic!("Expected merge operation");
        }
    }

    // -----------------------------------------------------------------------
    // Save methods
    // -----------------------------------------------------------------------

    fn save_single_pass(&self, file_prefix: &str) -> TokenizerResult<()> {
        let vocab = self.vocab.as_ref().expect("vocab must exist");

        println!("Saving word model: {}.model", file_prefix);
        println!("  - Vocabulary: {} tokens", vocab.len());
        if !vocab.special_tokens.is_empty() {
            println!("  - Special tokens: {}", vocab.special_tokens.len());
        }
        println!("  - Total vocab size: {}", vocab.total_size());

        let model_file = format!("{}.model", file_prefix);
        let file = std::fs::File::create(&model_file)
            .map_err(|e| TokenizerError::IoError(e))?;
        let mut f = std::io::BufWriter::new(file);

        use std::io::Write;

        writeln!(f, "BoundlessBPE v2 word")?;
        vocab.save(&mut f, &self.single_counts, None, &self.encoder)?;
        writeln!(f, "words")?;
        self.inf_data.write_to_file(&mut f, &self.encoder)?;

        Ok(())
    }

    fn save_two_pass(&self, file_prefix: &str) -> TokenizerResult<()> {
        assert!(self.inf_data.is_super);
        let word_model = self.word_model.as_ref().expect("word_model must be loaded");
        let word_words = word_model.words.as_ref().expect("word_model.words must be loaded");
        let word_vocab = word_model.vocab.as_ref().expect("word_model.vocab must be loaded");
        let vocab = self.vocab.as_ref().expect("vocab must exist");

        let num_supermerges = self.inf_data.merges.len();

        // Deep copy and potentially trim word model InferenceData
        let mut trimmed_words = InferenceData::create_for_training(
            Pretokenizer::new(
                word_words.pretokenizer.main_pattern_str.as_deref(),
                word_words.pretokenizer.script_specific_pattern_str.as_deref(),
                word_words.pretokenizer.script_specific_scripts.as_deref(),
                word_words.pretokenizer.merge_pattern_str.as_deref(),
            )?,
        );
        // Copy all data from word model's inference data
        trimmed_words.merges = word_words.merges.clone();
        trimmed_words.deletions = word_words.deletions.clone();
        trimmed_words.merges_lookup = word_words.merges_lookup.clone();
        trimmed_words.deletions_lookup = word_words.deletions_lookup.clone();
        trimmed_words.deletion_parts = word_words.deletion_parts.clone();
        trimmed_words.tau = word_words.tau;
        trimmed_words.is_super = word_words.is_super;
        trimmed_words.superbpe_mode = word_words.superbpe_mode;
        trimmed_words.blowup = word_words.blowup;

        let (mode_name, model_type) = if self.inf_data.superbpe_mode {
            ("SuperBPE", "superbpe")
        } else {
            trimmed_words.trim_operations_to(self.current_word_op_idx);
            ("BoundlessBPE", "boundless")
        };

        // Copy vocabulary for saving
        let mut vocab_for_save = Vocabulary::create_from_vocab(vocab);

        // Collect all special tokens
        let mut all_special_tokens: Vec<String> = Vec::new();

        if !word_vocab.special_tokens.is_empty() {
            let mut sorted_original: Vec<(&String, &i32)> =
                word_vocab.special_tokens.iter().collect();
            sorted_original.sort_by_key(|&(_, &idx)| idx);
            all_special_tokens.extend(sorted_original.into_iter().map(|(tok, _)| tok.clone()));
        }

        for tok in &self.pending_special_tokens {
            if !all_special_tokens.contains(tok) {
                all_special_tokens.push(tok.clone());
            }
        }

        if !all_special_tokens.is_empty() {
            vocab_for_save.register_special_tokens(&all_special_tokens);
        }

        println!("Saving {} model: {}.model", mode_name, file_prefix);
        println!("  - Vocabulary: {} tokens", vocab_for_save.len());
        println!("  - Supermerges: {}", num_supermerges);
        if !vocab_for_save.special_tokens.is_empty() {
            println!(
                "  - Special tokens: {}",
                vocab_for_save.special_tokens.len()
            );
        }
        println!("  - Total vocab size: {}", vocab_for_save.total_size());

        let model_file = format!("{}.model", file_prefix);
        let file = std::fs::File::create(&model_file)
            .map_err(|e| TokenizerError::IoError(e))?;
        let mut f = std::io::BufWriter::new(file);

        use std::io::Write;

        writeln!(f, "BoundlessBPE v2 {}", model_type)?;
        vocab_for_save.save(&mut f, &self.single_counts, None, &self.encoder)?;
        writeln!(f, "words")?;
        trimmed_words.write_to_file(&mut f, &self.encoder)?;
        writeln!(f, "superwords")?;
        self.inf_data.write_to_file(&mut f, &self.encoder)?;

        Ok(())
    }

    pub fn save(&self, file_prefix: &str) -> TokenizerResult<()> {
        if self.inf_data.is_super {
            self.save_two_pass(file_prefix)
        } else {
            self.save_single_pass(file_prefix)
        }
    }

    pub fn register_special_tokens(&mut self, special_tokens: Vec<String>) {
        if self.inf_data.is_super {
            self.pending_special_tokens = special_tokens;
        } else {
            if let Some(ref mut vocab) = self.vocab {
                vocab.register_special_tokens(&special_tokens);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Training loops
    // -----------------------------------------------------------------------

    fn _train_internal_bpe(
        &mut self,
        filepath: &str,
        outprefix: &str,
        num_lines: usize,
        vocab_size: i32,
        recalc: usize,
        max_bytes: usize,
        checkpoint_iterations: usize,
        verbose: bool,
        progress_interval: usize,
        save_pretokens: Option<&str>,
    ) -> TokenizerResult<()> {
        let overall_start = Instant::now();

        // BPE-specific initialization
        let init_start = Instant::now();
        self.vocab = Some(Vocabulary::create_initial());
        self.unlocked = AHashMap::new();
        self.unlocked_default = true;
        let total_init = init_start.elapsed().as_secs_f64();

        // Pretokenization
        let start_pretok = Instant::now();
        let (mut text_chunks, text_counts) = self.pretokenize(filepath, num_lines, max_bytes, save_pretokens, verbose)?;
        let total_pretok = start_pretok.elapsed().as_secs_f64();

        // Set up initial counts
        let start_ic = Instant::now();
        self.initial_counts(&text_chunks, &text_counts);
        let total_ic = start_ic.elapsed().as_secs_f64();

        let mut total_max_value = 0.0;
        let mut total_merge = 0.0;
        let mut total_delete = 0.0;
        let total_unlocked = 0.0;
        let mut total_verify = 0.0;
        let mut total_checkpoint = 0.0;

        if progress_interval > 0 {
            println!("*\ti\tvocab\ttype\tleft\trght\tc_ab\tc_a\tc_b\tios_a\tios_b\ttime\tpairs\tsingle_bytes\twhole_words\tnew_unlocked");
        }

        let mut i: usize = 0;

        loop {
            let start_max = Instant::now();
            let (best_pair, _best_cnt) = self.choose_best_pair();
            total_max_value += start_max.elapsed().as_secs_f64();

            let best_pair = best_pair.expect("best_pair should not be None in BPE training");

            let print_row = progress_interval > 0 && i % progress_interval == 0;
            let (merge_time, delete_time, _new_unlocked) =
                self.merge_and_delete(&best_pair, i, &mut text_chunks, &text_counts, print_row, verbose);

            total_merge += merge_time;
            total_delete += delete_time;

            // Periodic verification
            if recalc > 0 && i % recalc == 0 && i > 0 {
                let verify_start = Instant::now();
                self.verify_state(&text_chunks, &text_counts);
                total_verify += verify_start.elapsed().as_secs_f64();
            }

            i += 1;

            // Check stopping conditions
            let vocab_len = self.vocab.as_ref().unwrap().len() as i32;
            let mut should_stop = false;
            let mut stop_reason = String::new();

            if self.pair_counts.is_empty() {
                should_stop = true;
                stop_reason = format!("only single element chunks at iteration {}", i);
            } else if vocab_len >= vocab_size {
                should_stop = true;
                stop_reason = format!("reached vocab_size {}", vocab_size);
            }

            // Checkpoint saving
            let vocab_len = self.vocab.as_ref().unwrap().len();
            if (checkpoint_iterations > 0
                && vocab_len % checkpoint_iterations == 0
                && vocab_len > 0)
                || should_stop
            {
                let checkpoint_start = Instant::now();
                let checkpoint_prefix = format!("{}_{}", outprefix, vocab_len);
                self.save(&checkpoint_prefix)?;
                total_checkpoint += checkpoint_start.elapsed().as_secs_f64();

                if verbose {
                    self._print_checkpoint_stats(
                        outprefix,
                        i,
                        overall_start,
                        total_pretok,
                        total_ic,
                        total_max_value,
                        total_merge,
                        total_delete,
                        total_unlocked,
                        total_verify,
                        total_checkpoint,
                        total_init,
                        0.0,
                    );
                }
            }

            if should_stop {
                println!("Stopping: {}", stop_reason);
                break;
            }
        }

        Ok(())
    }

    fn _train_internal_super(
        &mut self,
        filepath: &str,
        outprefix: &str,
        num_lines: usize,
        vocab_size: i32,
        recalc: usize,
        word_model_file: &str,
        superbpe_mode: bool,
        max_bytes: usize,
        checkpoint_iterations: usize,
        verbose: bool,
        progress_interval: usize,
        save_pretokens: Option<&str>,
        greedy_split: bool,
        min_count: i64,
        max_ngram_len: usize,
    ) -> TokenizerResult<()> {
        let overall_start = Instant::now();

        // Super-specific initialization
        let init_start = Instant::now();

        // Load the word model
        let mut word_model = Tokenizer::new();
        word_model.load(word_model_file)?;

        let word_vocab = word_model.vocab.as_ref().expect("word model vocab");
        self.target_vocab_size = word_vocab.len() as i32;

        if superbpe_mode {
            self.vocab = Some(Vocabulary::create_from_vocab(word_vocab));
            self.word_operations_list = Vec::new();
            self.current_word_op_idx = 0;
        } else {
            self.vocab = Some(Vocabulary::create_initial());

            let word_words = word_model.words.as_ref().expect("word model words");
            self.word_operations_list = Vec::new();
            for (&idx, ((left, right), c_ab, unlocked_flag)) in &word_words.merges {
                self.word_operations_list.push((
                    idx,
                    OpType::Merge,
                    OpData::MergeData {
                        pair: (left.clone(), right.clone()),
                        c_ab: *c_ab,
                        unlocked_flag: *unlocked_flag,
                    },
                ));
            }
            for (&idx, token) in &word_words.deletions {
                self.word_operations_list.push((
                    idx,
                    OpType::Delete,
                    OpData::DeleteData {
                        token: token.clone(),
                    },
                ));
            }
            self.word_operations_list.sort_by_key(|&(idx, _, _)| idx);
            self.current_word_op_idx = 0;
        }

        // Unlock all word vocab tokens — the count-based competition naturally
        // ensures parents are created before any supermerge that uses them.
        // Previously single bytes were never unlocked (bug), and the unlock-on-replay
        // mechanism for merged tokens was unnecessary overhead.
        self.unlocked = AHashMap::new();
        self.unlocked_default = false;
        for tok in &word_vocab.tokens {
            self.unlocked.insert(tok.clone(), true);
        }

        self.word_model = Some(word_model);
        let total_init = init_start.elapsed().as_secs_f64();

        // Pretokenization
        let start_pretok = Instant::now();
        let (text_chunks_raw, text_counts_raw) =
            self.pretokenize_super(filepath, num_lines, max_bytes, save_pretokens, verbose)?;
        let total_pretok = start_pretok.elapsed().as_secs_f64();

        // Optional n-gram greedy split
        let total_ngram_split;
        let (mut text_chunks, text_counts) = if greedy_split {
            let start_ngram = Instant::now();
            let word_model_ref = self.word_model.as_ref().expect("word model must be loaded");
            let result = ngram_split(text_chunks_raw, text_counts_raw, word_model_ref, min_count, max_ngram_len);
            total_ngram_split = start_ngram.elapsed().as_secs_f64();
            result
        } else {
            total_ngram_split = 0.0;
            (text_chunks_raw, text_counts_raw)
        };

        // Set up initial counts
        let start_ic = Instant::now();
        self.initial_counts(&text_chunks, &text_counts);
        let total_ic = start_ic.elapsed().as_secs_f64();

        let mut total_max_value = 0.0;
        let mut total_merge = 0.0;
        let mut total_delete = 0.0;
        let total_unlocked = 0.0;
        let mut total_verify = 0.0;
        let mut total_checkpoint = 0.0;

        if progress_interval > 0 {
            println!("*\ti\tvocab\ttype\tleft\trght\tc_ab\tc_a\tc_b\tios_a\tios_b\ttime\tpairs\tsingle_bytes\twhole_words\tnew_unlocked");
        }

        let mut i: usize = 0;
        // Cache the best supermerge across iterations. Since regular merge
        // replays don't change the supermerge heap (all tokens pre-unlocked),
        // we can reuse the cached result instead of re-peeking each time.
        let mut cached_super: Option<(Option<(Vec<u8>, Vec<u8>)>, i64)> = None;

        loop {
            let print_row = progress_interval > 0 && i % progress_interval == 0;

            let (super_pair, super_c_ab) = if let Some(cached) = cached_super.take() {
                cached
            } else {
                let start_max = Instant::now();
                let result = self.choose_best_pair();
                total_max_value += start_max.elapsed().as_secs_f64();
                result
            };

            if superbpe_mode {
                if let Some(ref sp) = super_pair {
                    let sp_clone = sp.clone();
                    let (merge_time, delete_time, _) =
                        self.merge_and_delete(&sp_clone, i, &mut text_chunks, &text_counts, print_row, verbose);
                    total_merge += merge_time;
                    total_delete += delete_time;
                }
            } else {
                // BoundlessBPE: compete with regular merges
                let start_next = Instant::now();
                let (_reg_op_idx, reg_pair, reg_c_ab, _reg_unlocked_flag) =
                    self._get_next_regular_merge();
                let time_next = start_next.elapsed().as_secs_f64();

                // Strict > (not >=): ties go to the regular merge. This is
                // required because with all tokens pre-unlocked, a supermerge
                // could tie with the regular merge that creates one of its
                // parents. The parent must be created first.
                if super_c_ab > reg_c_ab && super_pair.is_some() {
                    let sp = super_pair.unwrap();
                    let vocab = self.vocab.as_ref().expect("vocab must exist");
                    assert!(
                        vocab.contains(&sp.0),
                        "supermerge parent not in vocab: {:?}",
                        self.encoder.from_bytes(&sp.0)
                    );
                    assert!(
                        vocab.contains(&sp.1),
                        "supermerge parent not in vocab: {:?}",
                        self.encoder.from_bytes(&sp.1)
                    );
                    let (merge_time, delete_time, _) =
                        self.merge_and_delete(&sp, i, &mut text_chunks, &text_counts, print_row, verbose);
                    total_merge += merge_time + time_next;
                    total_delete += delete_time;
                    // Supermerge changed the heap — don't cache
                } else if let Some(reg_pair) = reg_pair {
                    // Regular merge wins — replay it (just add token to vocab)
                    let vocab = self.vocab.as_mut().expect("vocab must exist");
                    let mut merged_token = reg_pair.0.clone();
                    merged_token.extend_from_slice(&reg_pair.1);

                    if !vocab.contains(&merged_token) {
                        vocab.add(merged_token.clone(), false);
                    } else if verbose {
                        println!(
                            "info: duplicate regular merge (token exists): {:?}",
                            self.encoder.from_bytes(&merged_token)
                        );
                    }

                    self.current_word_op_idx += 1;
                    self._process_pending_deletions();

                    // Heap unchanged — cache for next iteration
                    cached_super = Some((super_pair, super_c_ab));
                }
            }

            // Periodic verification
            if recalc > 0 && i % recalc == 0 && i > 0 {
                let verify_start = Instant::now();
                self.verify_state(&text_chunks, &text_counts);
                total_verify += verify_start.elapsed().as_secs_f64();
            }

            i += 1;

            // Check stopping conditions
            let mut should_stop = false;
            let mut stop_reason = String::new();

            if self.pair_counts.is_empty() {
                should_stop = true;
                stop_reason = format!("only single element chunks at iteration {}", i);
            } else if superbpe_mode {
                let num_supermerges =
                    self.vocab.as_ref().unwrap().len() as i32 - self.target_vocab_size;
                if num_supermerges >= vocab_size {
                    should_stop = true;
                    stop_reason = format!("reached {} supermerges", vocab_size);
                }
            } else {
                let vocab_len = self.vocab.as_ref().unwrap().len() as i32;
                if vocab_len >= self.target_vocab_size {
                    should_stop = true;
                    stop_reason = format!("reached target vocab_size {}", self.target_vocab_size);
                }
            }

            let vocab_len = self.vocab.as_ref().unwrap().len();
            if (checkpoint_iterations > 0
                && vocab_len % checkpoint_iterations == 0
                && vocab_len > 0)
                || should_stop
            {
                let checkpoint_start = Instant::now();
                let checkpoint_prefix = format!("{}_{}", outprefix, vocab_len);
                self.save(&checkpoint_prefix)?;
                total_checkpoint += checkpoint_start.elapsed().as_secs_f64();

                if verbose {
                    self._print_checkpoint_stats(
                        outprefix,
                        i,
                        overall_start,
                        total_pretok,
                        total_ic,
                        total_max_value,
                        total_merge,
                        total_delete,
                        total_unlocked,
                        total_verify,
                        total_checkpoint,
                        total_init,
                        total_ngram_split,
                    );
                }
            }

            if should_stop {
                println!("Stopping: {}", stop_reason);
                break;
            }
        }

        Ok(())
    }

    fn _print_checkpoint_stats(
        &self,
        outprefix: &str,
        i: usize,
        overall_start: Instant,
        total_pretok: f64,
        total_ic: f64,
        total_max_value: f64,
        total_merge: f64,
        total_delete: f64,
        total_unlocked: f64,
        total_verify: f64,
        total_checkpoint: f64,
        total_init: f64,
        total_ngram_split: f64,
    ) {
        let overall_time = overall_start.elapsed().as_secs_f64();
        let vocab = self.vocab.as_ref().unwrap();

        println!(":i: {}", i);
        println!(":len(vocab): {}", vocab.len());
        println!(":single_counts: {}", self.single_counts.len());
        println!(":pair_counts: {}", self.pair_counts.len());
        println!(":single_byte_cnt: {}", self.get_single_byte_cnt());
        println!(":whole_words: {}", self.whole_words);
        println!(":merges: {}", self.inf_data.merges.len());
        println!(":deletions: {}", self.inf_data.deletions.len());
        if self.inf_data.is_super
            && self.current_word_op_idx < self.word_operations_list.len()
        {
            println!(":current_word_op_idx: {}", self.current_word_op_idx);
            println!(":target_vocab_size: {}", self.target_vocab_size);
        }
        println!(":training time breakdown");
        println!(":total_init: {}", total_init);
        println!(":total_pretok: {}", total_pretok);
        println!(":total_ngram_split: {}", total_ngram_split);
        println!(":total_initalize_counts: {}", total_ic);
        println!(":total_max_value: {}", total_max_value);
        println!(":total_merge: {}", total_merge);
        println!(":total_delete: {}", total_delete);
        println!(":total_unlocked: {}", total_unlocked);
        println!(":total_verify: {}", total_verify);
        println!(":total_checkpoint: {}", total_checkpoint);
        println!(":overall_time: {}", overall_time);
        println!(
            ":missing: {}",
            overall_time
                - total_init
                - total_pretok
                - total_ngram_split
                - total_ic
                - total_max_value
                - total_merge
                - total_delete
                - total_verify
                - total_unlocked
                - total_checkpoint
        );
        println!(":outprefix: {}_{}", outprefix, vocab.len());
    }
}

// ---------------------------------------------------------------------------
// Concrete trainer types
// ---------------------------------------------------------------------------

pub struct BpeTrainer {
    pub base: BaseBpeTrainer,
}

impl BpeTrainer {
    pub fn new(pretokenizer: Option<Pretokenizer>) -> TokenizerResult<Self> {
        Ok(Self {
            base: BaseBpeTrainer::new(pretokenizer)?,
        })
    }

    pub fn train(
        &mut self,
        tau: f64,
        filepath: &str,
        outprefix: &str,
        num_lines: usize,
        vocab_size: i32,
        recalc: usize,
        blowup: bool,
        max_bytes: usize,
        checkpoint_iterations: usize,
        verbose: bool,
        progress_interval: usize,
        save_pretokens: Option<&str>,
    ) -> TokenizerResult<()> {
        assert!(vocab_size >= self.base.initial_vocab.len() as i32);

        self.base.reset_training_state(None)?;

        self.base.inf_data.tau = tau;
        self.base.inf_data.is_super = false;
        self.base.inf_data.superbpe_mode = false;
        self.base.inf_data.blowup = blowup;

        self.base._train_internal_bpe(
            filepath,
            outprefix,
            num_lines,
            vocab_size,
            recalc,
            max_bytes,
            checkpoint_iterations,
            verbose,
            progress_interval,
            save_pretokens,
        )
    }

    pub fn register_special_tokens(&mut self, tokens: Vec<String>) {
        self.base.register_special_tokens(tokens);
    }
}

pub struct BoundlessBpeTrainer {
    pub base: BaseBpeTrainer,
}

impl BoundlessBpeTrainer {
    pub fn new(pretokenizer: Option<Pretokenizer>) -> TokenizerResult<Self> {
        Ok(Self {
            base: BaseBpeTrainer::new(pretokenizer)?,
        })
    }

    pub fn train(
        &mut self,
        filepath: &str,
        outprefix: &str,
        num_lines: usize,
        recalc: usize,
        word_model_file: &str,
        max_bytes: usize,
        checkpoint_iterations: usize,
        verbose: bool,
        progress_interval: usize,
        save_pretokens: Option<&str>,
        greedy_split: bool,
        min_count: i64,
        max_ngram_len: usize,
    ) -> TokenizerResult<()> {
        self.base.reset_training_state(None)?;

        self.base.inf_data.tau = 1.1;
        self.base.inf_data.is_super = true;
        self.base.inf_data.superbpe_mode = false;
        self.base.inf_data.blowup = false;

        self.base._train_internal_super(
            filepath,
            outprefix,
            num_lines,
            -1, // vocab_size unused for BoundlessBPE
            recalc,
            word_model_file,
            false, // superbpe_mode
            max_bytes,
            checkpoint_iterations,
            verbose,
            progress_interval,
            save_pretokens,
            greedy_split,
            min_count,
            max_ngram_len,
        )
    }

    pub fn register_special_tokens(&mut self, tokens: Vec<String>) {
        self.base.register_special_tokens(tokens);
    }
}

pub struct SuperBpeTrainer {
    pub base: BaseBpeTrainer,
}

impl SuperBpeTrainer {
    pub fn new(pretokenizer: Option<Pretokenizer>) -> TokenizerResult<Self> {
        Ok(Self {
            base: BaseBpeTrainer::new(pretokenizer)?,
        })
    }

    pub fn train(
        &mut self,
        filepath: &str,
        outprefix: &str,
        num_lines: usize,
        vocab_size: i32,
        recalc: usize,
        word_model_file: &str,
        max_bytes: usize,
        checkpoint_iterations: usize,
        verbose: bool,
        progress_interval: usize,
        save_pretokens: Option<&str>,
        greedy_split: bool,
        min_count: i64,
        max_ngram_len: usize,
    ) -> TokenizerResult<()> {
        assert!(vocab_size >= 1);

        self.base.reset_training_state(None)?;

        self.base.inf_data.tau = 1.1;
        self.base.inf_data.is_super = true;
        self.base.inf_data.superbpe_mode = true;
        self.base.inf_data.blowup = false;

        self.base._train_internal_super(
            filepath,
            outprefix,
            num_lines,
            vocab_size,
            recalc,
            word_model_file,
            true, // superbpe_mode
            max_bytes,
            checkpoint_iterations,
            verbose,
            progress_interval,
            save_pretokens,
            greedy_split,
            min_count,
            max_ngram_len,
        )
    }

    pub fn register_special_tokens(&mut self, tokens: Vec<String>) {
        self.base.register_special_tokens(tokens);
    }
}

// ---------------------------------------------------------------------------
// N-gram greedy split
// ---------------------------------------------------------------------------

/// Extract c_min (count of the last merge) from a loaded word model.
fn get_cmin_from_word_model(word_model: &Tokenizer) -> i64 {
    let words = word_model.words.as_ref().expect("word model words must be loaded");
    let max_idx = *words.merges.keys().max().expect("word model must have merges");
    let (_, c_ab, _) = &words.merges[&max_idx];
    *c_ab
}

/// Count all n-grams (n>=2) with count >= min_cnt using Apriori pruning.
/// Operates on Vec<Vec<Vec<u8>>> chunks + Vec<i64> counts.
/// Returns HashMap<Vec<Vec<u8>>, i64> for n>=2 with count >= min_cnt.
fn count_ngrams_bytes(
    chunks: &[Vec<Vec<u8>>],
    counts: &[i64],
    min_cnt: i64,
    max_len: usize,
) -> AHashMap<Vec<Vec<u8>>, i64> {
    let overall_start = Instant::now();
    println!("count_ngrams_bytes: {} chunks, min_cnt={}", chunks.len(), min_cnt);

    let mut ngram_cnt: AHashMap<Vec<Vec<u8>>, i64> = AHashMap::new();

    for sz in 1..=max_len {
        let start_time = Instant::now();
        let start_size = ngram_cnt.len();

        for (j, (tokens, &cnt)) in chunks.iter().zip(counts.iter()).enumerate() {
            if j % 500000 == 0 {
                println!(
                    "  sz={} j={} ngrams={} t={:.1}s",
                    sz, j, ngram_cnt.len(), start_time.elapsed().as_secs_f64()
                );
            }

            let n = tokens.len();
            if n < sz {
                continue;
            }

            for i in 0..=(n - sz) {
                let should_count = if sz == 1 {
                    true
                } else {
                    let prefix = tokens[i..i + sz - 1].to_vec();
                    let suffix = tokens[i + 1..i + sz].to_vec();
                    ngram_cnt.get(&prefix).copied().unwrap_or(0) >= min_cnt
                        && ngram_cnt.get(&suffix).copied().unwrap_or(0) >= min_cnt
                };

                if should_count {
                    let ngram = tokens[i..i + sz].to_vec();
                    *ngram_cnt.entry(ngram).or_insert(0) += cnt as i64;
                }
            }
        }

        // Filter below threshold (keep unigrams for next pass's pruning)
        let before = ngram_cnt.len();
        ngram_cnt.retain(|ng, c| ng.len() == 1 || *c >= min_cnt);
        let elapsed = start_time.elapsed().as_secs_f64();
        println!("  sz={}: {} -> {} ngrams ({:.1}s)", sz, before, ngram_cnt.len(), elapsed);

        if ngram_cnt.len() == start_size {
            println!("  no new ngrams found, stopping");
            break;
        }
    }

    // Final: only n>=2 with count >= min_cnt
    ngram_cnt.retain(|ng, c| ng.len() >= 2 && *c >= min_cnt);

    // Length distribution
    let mut len_dist: AHashMap<usize, usize> = AHashMap::new();
    let mut len_total: AHashMap<usize, i64> = AHashMap::new();
    for (ng, &c) in &ngram_cnt {
        let n = ng.len();
        *len_dist.entry(n).or_insert(0) += 1;
        *len_total.entry(n).or_insert(0) += c;
    }
    let max_len_found = len_dist.keys().max().copied().unwrap_or(0);
    let total_ngrams: i64 = ngram_cnt.values().sum();
    let unique_ngrams = ngram_cnt.len();
    println!(
        "count_ngrams_bytes: {} results (max_len={}), total time: {:.1}s",
        unique_ngrams, max_len_found, overall_start.elapsed().as_secs_f64()
    );
    if total_ngrams > 0 {
        println!(
            "N-grams: unique={}, total={}, ratio={:.6}",
            unique_ngrams, total_ngrams, unique_ngrams as f64 / total_ngrams as f64
        );
    } else {
        println!("N-grams: unique={}, total={}", unique_ngrams, total_ngrams);
    }
    let mut sorted_lens: Vec<usize> = len_dist.keys().copied().collect();
    sorted_lens.sort();
    for n in sorted_lens {
        println!("  len={}: {} unique, {} total", n, len_dist[&n], len_total[&n]);
    }

    ngram_cnt
}

/// Build a set of all prefixes of n-grams for fast lookup.
fn build_prefix_set_bytes(ngram_dict: &AHashMap<Vec<Vec<u8>>, i64>) -> AHashSet<Vec<Vec<u8>>> {
    let mut prefixes = AHashSet::new();
    for ng in ngram_dict.keys() {
        for length in 1..=ng.len() {
            prefixes.insert(ng[..length].to_vec());
        }
    }
    prefixes
}

/// Greedy left-to-right partition of chunks using n-gram dictionary.
/// Returns (new_chunks, new_counts) ready for initial_counts().
fn greedy_split_bytes(
    chunks: &[Vec<Vec<u8>>],
    counts: &[i64],
    ngram_dict: &AHashMap<Vec<Vec<u8>>, i64>,
) -> (Vec<Vec<Vec<u8>>>, Vec<i64>) {
    let start_time = Instant::now();
    println!("greedy_split_bytes: {} chunks, {} ngrams", chunks.len(), ngram_dict.len());

    let prefix_set = build_prefix_set_bytes(ngram_dict);
    let mut result: AHashMap<Vec<Vec<u8>>, i64> = AHashMap::new();

    for (j, (tokens, &cnt)) in chunks.iter().zip(counts.iter()).enumerate() {
        if j % 500000 == 0 {
            println!(
                "  j={} result_size={} t={:.1}s",
                j, result.len(), start_time.elapsed().as_secs_f64()
            );
        }

        let mut pos = 0;
        while pos < tokens.len() {
            let mut best_len = 0usize;
            for end in (pos + 1)..=tokens.len() {
                let candidate = tokens[pos..end].to_vec();
                if !prefix_set.contains(&candidate) {
                    break;
                }
                if candidate.len() >= 2 && ngram_dict.contains_key(&candidate) {
                    best_len = candidate.len();
                }
            }

            if best_len >= 2 {
                let ngram = tokens[pos..pos + best_len].to_vec();
                *result.entry(ngram).or_insert(0) += cnt as i64;
                pos += best_len;
            } else {
                // Single token — emit as a length-1 chunk
                let single = vec![tokens[pos].clone()];
                *result.entry(single).or_insert(0) += cnt as i64;
                pos += 1;
            }
        }
    }

    println!(
        "greedy_split_bytes: {} unique chunks, time: {:.1}s",
        result.len(), start_time.elapsed().as_secs_f64()
    );

    let mut new_chunks: Vec<Vec<Vec<u8>>> = Vec::with_capacity(result.len());
    let mut new_counts: Vec<i64> = Vec::with_capacity(result.len());
    for (ng, c) in result {
        new_chunks.push(ng);
        new_counts.push(c);
    }

    (new_chunks, new_counts)
}

/// Top-level entry point for n-gram splitting.
fn ngram_split(
    text_chunks: Vec<Vec<Vec<u8>>>,
    text_counts: Vec<i64>,
    word_model: &Tokenizer,
    min_count_floor: i64,
    max_ngram_len: usize,
) -> (Vec<Vec<Vec<u8>>>, Vec<i64>) {
    let model_cmin = get_cmin_from_word_model(word_model);
    let min_cnt = model_cmin.max(min_count_floor);
    println!("ngram_split: c_min: model={}, floor={}, using={}", model_cmin, min_count_floor, min_cnt);

    let unique_input = text_chunks.len();
    let total_input: i64 = text_counts.iter().map(|&c| c as i64).sum();
    println!("ngram_split input: unique={}, total={}, ratio={:.6}", unique_input, total_input, unique_input as f64 / total_input as f64);

    let ngram_dict = count_ngrams_bytes(&text_chunks, &text_counts, min_cnt, max_ngram_len);

    if ngram_dict.is_empty() {
        println!("ngram_split: no n-grams found, returning original data unchanged");
        return (text_chunks, text_counts);
    }

    let (new_chunks, new_counts) = greedy_split_bytes(&text_chunks, &text_counts, &ngram_dict);

    let unique_output = new_chunks.len();
    let total_output: i64 = new_counts.iter().map(|&c| c as i64).sum();
    println!("ngram_split output: unique={}, total={}, ratio={:.6}", unique_output, total_output, unique_output as f64 / total_output as f64);

    // Stats for n>=2 chunks only (comparable to compute_ngrams.py greedy_split output)
    let mut ngram_unique: usize = 0;
    let mut ngram_total: i64 = 0;
    for (c, &cnt) in new_chunks.iter().zip(new_counts.iter()) {
        if c.len() >= 2 {
            ngram_unique += 1;
            ngram_total += cnt as i64;
        }
    }
    let singleton_unique = unique_output - ngram_unique;
    let singleton_total = total_output - ngram_total;
    if ngram_total > 0 {
        println!("ngram_split n>=2: unique={}, total={}, ratio={:.6}", ngram_unique, ngram_total, ngram_unique as f64 / ngram_total as f64);
    } else {
        println!("ngram_split n>=2: unique={}, total={}", ngram_unique, ngram_total);
    }
    println!("ngram_split singletons: unique={}, total={}", singleton_unique, singleton_total);
    println!(
        "ngram_split reduction: {} -> {} unique chunks ({:.4}x)",
        unique_input, unique_output, unique_output as f64 / unique_input as f64
    );

    (new_chunks, new_counts)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn verify_maps<K: std::fmt::Debug + std::hash::Hash + Eq, V: std::fmt::Debug + PartialEq>(
    expected: &AHashMap<K, V>,
    actual: &AHashMap<K, V>,
) {
    for (key, expected_val) in expected {
        let actual_val = actual.get(key);
        assert!(
            actual_val.is_some(),
            "Key {:?} missing from actual (expected {:?})",
            key,
            expected_val
        );
        assert_eq!(
            expected_val,
            actual_val.unwrap(),
            "Value mismatch for key {:?}",
            key
        );
    }
    for (key, actual_val) in actual {
        assert!(
            expected.contains_key(key),
            "Extra key {:?} in actual (value {:?})",
            key,
            actual_val
        );
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_stats_basic() {
        let tokens: Vec<Vec<u8>> = vec![
            vec![b'a'],
            vec![b'b'],
            vec![b'c'],
            vec![b'a'],
            vec![b'b'],
        ];
        let mut counts: AHashMap<(Vec<u8>, Vec<u8>), i64> = AHashMap::new();
        BaseBpeTrainer::get_stats(&tokens, &mut counts, 1);

        assert_eq!(
            counts.get(&(vec![b'a'], vec![b'b'])).copied().unwrap_or(0),
            2
        );
        assert_eq!(
            counts.get(&(vec![b'b'], vec![b'c'])).copied().unwrap_or(0),
            1
        );
        assert_eq!(
            counts.get(&(vec![b'c'], vec![b'a'])).copied().unwrap_or(0),
            1
        );
    }

    #[test]
    fn test_get_stats_overlap() {
        // Test overlap handling: [a, a, a] should give (a,a):1 not 2
        let tokens: Vec<Vec<u8>> = vec![vec![b'a'], vec![b'a'], vec![b'a']];
        let mut counts: AHashMap<(Vec<u8>, Vec<u8>), i64> = AHashMap::new();
        BaseBpeTrainer::get_stats(&tokens, &mut counts, 1);

        assert_eq!(
            counts.get(&(vec![b'a'], vec![b'a'])).copied().unwrap_or(0),
            1
        );
    }

    #[test]
    fn test_vocabulary_create_initial() {
        let vocab = Vocabulary::create_initial();
        assert_eq!(vocab.len(), 243);
        // Check 0xC0 and 0xC1 are excluded
        assert!(!vocab.contains(&[0xC0]));
        assert!(!vocab.contains(&[0xC1]));
        // Check 0xF5 excluded
        assert!(!vocab.contains(&[0xF5]));
        // Check valid bytes present
        assert!(vocab.contains(&[0x00]));
        assert!(vocab.contains(&[0x41])); // 'A'
        assert!(vocab.contains(&[0xC2]));
        assert!(vocab.contains(&[0xF4]));
    }

    #[test]
    fn test_vocabulary_delete() {
        let mut vocab = Vocabulary::create_initial();
        let initial_len = vocab.len();

        // Delete a token
        vocab.delete(&[0x41]); // 'A'
        assert_eq!(vocab.len(), initial_len - 1);
        assert!(!vocab.contains(&[0x41]));

        // Verify re-indexing
        vocab.verify_vocabulary();
    }
}
