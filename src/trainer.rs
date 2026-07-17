// Copyright 2026-present Kensho Technologies, LLC.
use std::fs::File;
use std::hash::Hash;
use std::io::{BufRead, BufReader};
use std::time::Instant;

use ahash::{AHashMap, AHashSet};
use priority_queue::PriorityQueue;
use rayon::prelude::*;

use crate::byte_encoding::ByteEncoder;
use crate::error::{TokenizerError, TokenizerResult};
use crate::inference_data::InferenceData;
use crate::pretokenize::Pretokenizer;
use crate::tokenizer::Tokenizer;
use crate::training_types::{
    blow_up_chunk, intern_chunks, merge_chunk, Chunk, InitialCounts, Pair, TokenArena, TokenId,
};
use crate::vocabulary::Vocabulary;

const SCAN_BATCH_DOCUMENTS: usize = 512;
const PARALLEL_REWRITE_MIN_CANDIDATES: usize = 1_024;

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
// `priority-queue` is a max-heap, so `Ord` below reverses these fields.

/// Logical priority is `(both_unlocked, count, lexical_pair)`, with higher
/// values preferred. The stored fields and `Ord` implementation invert that
/// ordering because `PriorityQueue` is a max-heap.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PairPriority {
    neg_both_unlocked: i32,
    neg_count: i64,
    /// IDs are creation-ordered, while historical BPE tie-breaking is
    /// lexicographic by token bytes. This copy exists only in the priority;
    /// heap items and trainer maps remain compact `Pair` IDs.
    lexical_pair: (Box<[u8]>, Box<[u8]>),
}

/// Worker-local state from an in-place rewrite. Workers mutate disjoint chunks
/// and return aggregate deltas and changed indices; replacement chunks are not
/// retained in the worker result.
#[derive(Default)]
struct LocalRewrite {
    pair_deltas: AHashMap<Pair, i64>,
    changed_chunks: Vec<usize>,
    replacements: i64,
    whole_word_delta: i64,
}

impl PartialOrd for PairPriority {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PairPriority {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse the logical ordering because PriorityQueue is a max-heap.
        other
            .neg_both_unlocked
            .cmp(&self.neg_both_unlocked)
            .then_with(|| other.neg_count.cmp(&self.neg_count))
            .then_with(|| other.lexical_pair.cmp(&self.lexical_pair))
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
    /// Stable training identities, independent of `Vocabulary` IDs, which
    /// BoundlessBPE may renumber during deletion.
    arena: TokenArena,

    // Training state
    pub vocab: Option<Vocabulary>,
    single_counts: AHashMap<TokenId, i64>,
    pair_counts: PriorityQueue<Pair, PairPriority>,
    token_to_pairs: AHashMap<TokenId, AHashSet<Pair>>,
    token_locations: AHashMap<TokenId, AHashSet<usize>>,
    // `unlocked` maps are stored explicitly; missing keys have a default
    unlocked: AHashMap<TokenId, bool>,
    unlocked_default: bool, // true for BPE, false for super
    whole_words: i64,

    // Supermerge training state
    word_model: Option<Tokenizer>,
    word_operations_list: Vec<(i32, OpType, OpData)>,
    current_word_op_idx: usize,
    target_vocab_size: i32,
    pending_special_tokens: Vec<String>,
}

impl BaseBpeTrainer {
    pub fn new(pretokenizer: Option<Pretokenizer>) -> TokenizerResult<Self> {
        let pretok = match pretokenizer {
            Some(p) => p,
            None => Pretokenizer::new(None, None, None, None)?,
        };
        let inf_data = InferenceData::create_for_training(pretok);

        // Build the model's 243 initial byte tokens.
        let initial_vocab: Vec<Vec<u8>> = (0u8..192).chain(194u8..245).map(|b| vec![b]).collect();

        Ok(Self {
            inf_data,
            index_gen: IndexGenerator::new(),
            initial_vocab,
            encoder: ByteEncoder::new(),
            arena: TokenArena::new(),
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
        })
    }

    fn is_unlocked(&self, tok: TokenId) -> bool {
        self.unlocked
            .get(&tok)
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
                self.inf_data
                    .pretokenizer
                    .script_specific_pattern_str
                    .as_deref(),
                self.inf_data
                    .pretokenizer
                    .script_specific_scripts
                    .as_deref(),
                self.inf_data.pretokenizer.merge_pattern_str.as_deref(),
            )?;
            self.inf_data = InferenceData::create_for_training(pretok);
        }
        self.vocab = None;
        self.arena = TokenArena::new();
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

    fn merge_count_maps<K>(target: &mut AHashMap<K, i64>, source: AHashMap<K, i64>)
    where
        K: Eq + Hash,
    {
        for (key, count) in source {
            *target.entry(key).or_insert(0) += count;
        }
    }

    fn parse_jsonl_text_batch(lines: &[String]) -> Vec<TokenizerResult<String>> {
        // Keep batches ordered so the serial caller can apply num_lines and
        // max_bytes exactly as before. Errors stay attached to their source
        // line: a record after the cap must not fail a scan that would never
        // have read it in the serial implementation.
        lines
            .par_iter()
            .map(|line| {
                let json: serde_json::Value =
                    serde_json::from_str(line.trim_end()).map_err(TokenizerError::JsonError)?;
                json["text"]
                    .as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| TokenizerError::ModelError("missing 'text' field".into()))
            })
            .collect()
    }

    fn collect_documents_until_max_bytes(
        parsed_documents: Vec<TokenizerResult<String>>,
        max_bytes: usize,
        total_chars: &mut usize,
        total_bytes: &mut usize,
        documents_read: &mut usize,
    ) -> TokenizerResult<(Vec<String>, bool)> {
        let mut documents = Vec::with_capacity(parsed_documents.len());
        for result in parsed_documents {
            let text = result?;
            *total_chars += text.chars().count();
            *total_bytes += text.len(); // UTF-8 byte length
            *documents_read += 1;
            documents.push(text);
            if *total_bytes >= max_bytes {
                return Ok((documents, true));
            }
        }
        Ok((documents, false))
    }

    fn tally_pretoken_batch(
        pretokenizer: &Pretokenizer,
        documents: &[String],
    ) -> AHashMap<String, i64> {
        documents
            .par_iter()
            .fold(AHashMap::new, |mut local_counts, text| {
                for chunk in pretokenizer.pretokenize(text) {
                    *local_counts.entry(chunk).or_insert(0) += 1;
                }
                local_counts
            })
            .reduce(AHashMap::new, |mut left, right| {
                Self::merge_count_maps(&mut left, right);
                left
            })
    }

    fn add_superword_run(counts: &mut AHashMap<Chunk, i64>, run: &mut Chunk) {
        if run.len() >= 2 {
            *counts.entry(std::mem::take(run)).or_insert(0) += 1;
        } else {
            run.clear();
        }
    }

    fn tally_superword_batch(
        pretokenizer: &Pretokenizer,
        supermerge_eligible: &AHashSet<TokenId>,
        arena: &TokenArena,
        documents: &[String],
    ) -> AHashMap<Chunk, i64> {
        documents
            .par_iter()
            .fold(AHashMap::new, |mut local_counts, text| {
                let mut run = Vec::new();
                for chunk in pretokenizer.pretokenize(text) {
                    if let Some(token) = arena
                        .id_for(chunk.as_bytes())
                        .filter(|token| supermerge_eligible.contains(token))
                    {
                        run.push(token);
                    } else {
                        Self::add_superword_run(&mut local_counts, &mut run);
                    }
                }
                Self::add_superword_run(&mut local_counts, &mut run);
                local_counts
            })
            .reduce(AHashMap::new, |mut left, right| {
                Self::merge_count_maps(&mut left, right);
                left
            })
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
        let mut documents_read: usize = 0;

        let f = File::open(filepath).map_err(|e| TokenizerError::IoError(e))?;
        let mut reader = BufReader::new(f);
        let mut raw_batch = Vec::with_capacity(SCAN_BATCH_DOCUMENTS);
        let mut source_lines = 0;
        let mut reached_max_bytes = false;

        while source_lines < num_lines && !reached_max_bytes {
            raw_batch.clear();
            while source_lines < num_lines && raw_batch.len() < SCAN_BATCH_DOCUMENTS {
                if verbose && source_lines % 10_000 == 0 {
                    println!(
                        "document {} {:.3} {} {}",
                        source_lines,
                        start.elapsed().as_secs_f64(),
                        total_chars,
                        total_bytes
                    );
                }
                let mut line = String::new();
                if reader
                    .read_line(&mut line)
                    .map_err(TokenizerError::IoError)?
                    == 0
                {
                    break;
                }
                raw_batch.push(line);
                source_lines += 1;
            }
            if raw_batch.is_empty() {
                break;
            }

            let (document_batch, reached_max) = Self::collect_documents_until_max_bytes(
                Self::parse_jsonl_text_batch(&raw_batch),
                max_bytes,
                &mut total_chars,
                &mut total_bytes,
                &mut documents_read,
            )?;
            if reached_max {
                reached_max_bytes = true;
                if verbose {
                    println!(
                        "at max_bytes {} {} {} {}",
                        documents_read - 1,
                        max_bytes,
                        total_chars,
                        total_bytes
                    );
                }
            }
            if !document_batch.is_empty() {
                let local_counts =
                    Self::tally_pretoken_batch(&self.inf_data.pretokenizer, &document_batch);
                Self::merge_count_maps(&mut chunk_tally, local_counts);
            }
        }

        println!(
            "phase 1 scan: documents={} bytes={} unique_pretokens={} elapsed={:.1}s",
            documents_read,
            total_bytes,
            chunk_tally.len(),
            start.elapsed().as_secs_f64(),
        );

        // Sort by count and bytes so parallel aggregation has deterministic
        // output independent of hash-map iteration order.
        let mut cnt_chk: Vec<(i64, Vec<u8>)> = chunk_tally
            .into_iter()
            .map(|(chk, cnt)| (cnt, chk.into_bytes()))
            .collect();
        cnt_chk.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));

        if cnt_chk.is_empty() {
            println!("WARNING: no pre-tokenization chunks found!");
            return Ok((vec![], vec![]));
        }

        // Write optional TSV output.
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
    ) -> TokenizerResult<(Vec<Chunk>, Vec<i64>)> {
        let start = Instant::now();
        assert!(self.inf_data.is_super);

        let word_model = self.word_model.as_ref().expect("word_model must be loaded");

        let mut total_bytes: usize = 0;
        let mut total_chars: usize = 0;
        let mut counts: AHashMap<Chunk, i64> = AHashMap::new();
        let mut documents_read: usize = 0;

        // Precompute IDs of pretokens that are both single-token
        // after word merges (reachable) and possible superwords. The word
        // vocabulary was interned during initialization, so this avoids a
        // second byte-owning eligibility set.
        let reachable = word_model
            .reachable_vocab
            .as_ref()
            .expect("reachable_vocab must be initialized");
        let possible = word_model
            .possible_superwords
            .as_ref()
            .expect("possible_superwords must be initialized");
        let supermerge_eligible: AHashSet<TokenId> = reachable
            .intersection(possible)
            .map(|token| {
                self.arena
                    .id_for(token)
                    .expect("word vocabulary token was not interned")
            })
            .collect();
        if verbose {
            println!(
                "supermerge_eligible: {} tokens (reachable={}, possible_superwords={})",
                supermerge_eligible.len(),
                reachable.len(),
                possible.len()
            );
        }

        let f = File::open(filepath).map_err(TokenizerError::IoError)?;
        let mut reader = BufReader::new(f);
        let mut raw_batch = Vec::with_capacity(SCAN_BATCH_DOCUMENTS);
        let mut source_lines = 0;
        let mut reached_max_bytes = false;

        // Reading stays serial to preserve source order and the exact byte cap.
        // JSON parsing and regex/run tally execute in parallel for each bounded
        // batch, avoiding unbounded corpus buffering.
        while source_lines < num_lines && !reached_max_bytes {
            raw_batch.clear();
            while source_lines < num_lines && raw_batch.len() < SCAN_BATCH_DOCUMENTS {
                if verbose && source_lines % 10_000 == 0 {
                    println!(
                        "document {} {:.3} {} {}",
                        source_lines,
                        start.elapsed().as_secs_f64(),
                        total_chars,
                        total_bytes
                    );
                }
                let mut line = String::new();
                if reader
                    .read_line(&mut line)
                    .map_err(TokenizerError::IoError)?
                    == 0
                {
                    break;
                }
                raw_batch.push(line);
                source_lines += 1;
            }
            if raw_batch.is_empty() {
                break;
            }

            let (document_batch, reached_max) = Self::collect_documents_until_max_bytes(
                Self::parse_jsonl_text_batch(&raw_batch),
                max_bytes,
                &mut total_chars,
                &mut total_bytes,
                &mut documents_read,
            )?;
            if reached_max {
                reached_max_bytes = true;
                if verbose {
                    println!(
                        "at max_bytes {} {} {} {}",
                        documents_read - 1,
                        max_bytes,
                        total_chars,
                        total_bytes
                    );
                }
            }

            if !document_batch.is_empty() {
                let local_counts = Self::tally_superword_batch(
                    &self.inf_data.pretokenizer,
                    &supermerge_eligible,
                    &self.arena,
                    &document_batch,
                );
                Self::merge_count_maps(&mut counts, local_counts);
            }
        }

        println!(
            "phase 2 scan: documents={} bytes={} unique_runs={} elapsed={:.1}s",
            documents_read,
            total_bytes,
            counts.len(),
            start.elapsed().as_secs_f64(),
        );

        // Preserve deterministic chunk ordering after parallel aggregation.
        let mut cnt_chk: Vec<(i64, Chunk)> =
            counts.into_iter().map(|(chk, cnt)| (cnt, chk)).collect();
        cnt_chk.sort_by(|a, b| {
            b.0.cmp(&a.0).then_with(|| {
                a.1.iter()
                    .map(|&token| self.arena.bytes(token))
                    .cmp(b.1.iter().map(|&token| self.arena.bytes(token)))
            })
        });

        if cnt_chk.is_empty() {
            println!("WARNING: no superword pre-tokenization chunks found!");
            return Ok((vec![], vec![]));
        }

        // Write optional TSV output.
        if let Some(path) = save_pretokens {
            use std::io::Write;
            if let Some(parent) = std::path::Path::new(path).parent() {
                std::fs::create_dir_all(parent).map_err(|e| TokenizerError::IoError(e))?;
            }
            let mut out = std::io::BufWriter::new(
                File::create(path).map_err(|e| TokenizerError::IoError(e))?,
            );
            for (cnt, chk) in &cnt_chk {
                let tokens: Vec<String> = chk
                    .iter()
                    .map(|&token| self.encoder.from_bytes(self.arena.bytes(token)))
                    .collect();
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
        let text_chunks: Vec<Chunk> = cnt_chk.into_iter().map(|(_, chk)| chk).collect();

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
    fn get_stats(tokens: &[TokenId], counts: &mut AHashMap<Pair, i64>, multiplier: i64) {
        let mut prev_pair: Option<Pair> = None;
        for i in 0..tokens.len().saturating_sub(1) {
            let pair = (tokens[i], tokens[i + 1]);
            let same_as_previous = pair.0 == pair.1 && prev_pair.as_ref() == Some(&pair);
            if !same_as_previous {
                *counts.entry(pair).or_insert(0) += multiplier;
                prev_pair = Some(pair);
            } else {
                prev_pair = None;
            }
        }
    }

    // -----------------------------------------------------------------------
    // State initialization
    // -----------------------------------------------------------------------

    fn _calc_pair_counts(
        &self,
        text_chunks: &[Chunk],
        text_counts: &[i64],
    ) -> (AHashMap<Pair, i64>, PriorityQueue<Pair, PairPriority>) {
        let mut pair_counts_dict: AHashMap<Pair, i64> = AHashMap::new();
        for (tokens, &cnt) in text_chunks.iter().zip(text_counts.iter()) {
            Self::get_stats(tokens, &mut pair_counts_dict, cnt);
        }

        let pair_counts_heap = self.build_pair_heap(&pair_counts_dict);

        (pair_counts_dict, pair_counts_heap)
    }

    fn build_pair_heap(
        &self,
        pair_counts: &AHashMap<Pair, i64>,
    ) -> PriorityQueue<Pair, PairPriority> {
        let mut pair_counts_heap = PriorityQueue::new();
        for (pair, &count) in pair_counts {
            let priority = self.pair_priority(*pair, count);
            pair_counts_heap.push(*pair, priority);
        }

        pair_counts_heap
    }

    fn _calc_single_counts(text_chunks: &[Chunk], text_counts: &[i64]) -> AHashMap<TokenId, i64> {
        let mut single_counts: AHashMap<TokenId, i64> = AHashMap::new();
        for (tokens, &cnt) in text_chunks.iter().zip(text_counts.iter()) {
            for tok in tokens {
                *single_counts.entry(*tok).or_insert(0) += cnt;
            }
        }
        single_counts
    }

    fn _calc_token_locations(text_chunks: &[Chunk]) -> AHashMap<TokenId, AHashSet<usize>> {
        let mut token_locations: AHashMap<TokenId, AHashSet<usize>> = AHashMap::new();
        for (chunk_idx, tokens) in text_chunks.iter().enumerate() {
            let unique: AHashSet<TokenId> = tokens.iter().copied().collect();
            for token in unique {
                token_locations.entry(token).or_default().insert(chunk_idx);
            }
        }
        token_locations
    }

    fn _calc_whole_words(text_chunks: &[Chunk], text_counts: &[i64]) -> i64 {
        let mut whole_words: i64 = 0;
        for (tokens, &cnt) in text_chunks.iter().zip(text_counts.iter()) {
            if tokens.len() == 1 {
                whole_words += cnt as i64;
            }
        }
        whole_words
    }

    fn initial_counts(&mut self, text_chunks: &[Chunk], text_counts: &[i64]) {
        let start = Instant::now();
        let InitialCounts {
            pair_counts,
            single_counts,
            token_locations,
            whole_words,
        } = InitialCounts::from_chunks_parallel(text_chunks, text_counts);

        self.pair_counts = self.build_pair_heap(&pair_counts);
        self.single_counts = single_counts;
        self.token_locations = token_locations
            .into_iter()
            .map(|(token, locations)| (token, locations.into_iter().collect()))
            .collect();
        self.whole_words = whole_words;
        println!(
            "initial counts: pairs={} tokens={} locations={} elapsed={:.1}s",
            self.pair_counts.len(),
            self.single_counts.len(),
            self.token_locations.len(),
            start.elapsed().as_secs_f64(),
        );
    }

    // -----------------------------------------------------------------------
    // Verification
    // -----------------------------------------------------------------------

    fn verify_pair_counts(&self, text_chunks: &[Chunk], text_counts: &[i64]) {
        let (from_scratch, _) = self._calc_pair_counts(text_chunks, text_counts);
        let current: AHashMap<Pair, i64> = self
            .pair_counts
            .iter()
            .map(|(pair, _)| (*pair, self.get_pair_count(pair)))
            .collect();
        verify_maps(&from_scratch, &current);
    }

    fn verify_single_counts(&self, text_chunks: &[Chunk], text_counts: &[i64]) {
        verify_maps(
            &Self::_calc_single_counts(text_chunks, text_counts),
            &self.single_counts,
        );
    }

    fn verify_whole_words(&self, text_chunks: &[Chunk], text_counts: &[i64]) {
        assert_eq!(
            self.whole_words,
            Self::_calc_whole_words(text_chunks, text_counts)
        );
    }

    fn verify_token_locations(&self, text_chunks: &[Chunk]) {
        verify_maps(
            &Self::_calc_token_locations(text_chunks),
            &self.token_locations,
        );
    }

    fn verify_state(&self, text_chunks: &[Chunk], text_counts: &[i64]) {
        if let Some(ref vocab) = self.vocab {
            vocab.verify_vocabulary();
        }
        self.verify_pair_counts(text_chunks, text_counts);
        self.verify_single_counts(text_chunks, text_counts);
        self.verify_whole_words(text_chunks, text_counts);
        self.verify_token_locations(text_chunks);
        self.inf_data.verify_indices();
    }

    // -----------------------------------------------------------------------
    // Core operations
    // -----------------------------------------------------------------------

    fn get_pair_count(&self, pair: &Pair) -> i64 {
        match self.pair_counts.get(pair) {
            Some((_, priority)) => -priority.neg_count,
            None => 0,
        }
    }

    fn pair_priority(&self, pair: Pair, count: i64) -> PairPriority {
        PairPriority {
            neg_both_unlocked: -((self.is_unlocked(pair.0) && self.is_unlocked(pair.1)) as i32),
            neg_count: -count,
            lexical_pair: (
                self.arena.bytes(pair.0).into(),
                self.arena.bytes(pair.1).into(),
            ),
        }
    }

    fn choose_best_pair(&self) -> (Option<Pair>, i64) {
        if self.pair_counts.is_empty() {
            return (None, -1);
        }

        let (pair, priority) = self.pair_counts.peek().unwrap();
        if priority.neg_both_unlocked != -1 {
            return (None, -1);
        }

        let count = -priority.neg_count;
        assert!(self.is_unlocked(pair.0));
        assert!(self.is_unlocked(pair.1));

        (Some(*pair), count)
    }

    fn apply_pair_count_changes(&mut self, overall_change: &AHashMap<Pair, i64>) {
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
                    let priority = self.pair_priority(*pair, new_count);
                    // Refresh an existing entry or insert a newly observed pair.
                    if self.pair_counts.get(pair).is_some() {
                        self.pair_counts.change_priority(pair, priority);
                    } else {
                        self.pair_counts.push(*pair, priority);
                    }
                }
            }
        }
    }

    fn get_single_byte_cnt(&self) -> i64 {
        self.single_counts
            .iter()
            .filter(|(token, _)| self.arena.bytes(**token).len() == 1)
            .map(|(_, count)| *count)
            .sum()
    }

    fn byte_single_counts(&self) -> AHashMap<Vec<u8>, i64> {
        self.single_counts
            .iter()
            .map(|(&token, &count)| (self.arena.bytes(token).to_vec(), count))
            .collect()
    }

    fn merge_candidates(&self, (first, second): Pair) -> Vec<usize> {
        if first == second {
            let mut result: Vec<usize> = self
                .token_locations
                .get(&first)
                .map(|locations| locations.iter().copied().collect())
                .unwrap_or_default();
            result.sort_unstable();
            return result;
        }
        let (Some(left), Some(right)) = (
            self.token_locations.get(&first),
            self.token_locations.get(&second),
        ) else {
            return Vec::new();
        };
        let mut result: Vec<usize> = left.intersection(right).copied().collect();
        result.sort_unstable();
        result
    }

    fn add_location(&mut self, token: TokenId, chunk_index: usize) {
        self.token_locations
            .entry(token)
            .or_default()
            .insert(chunk_index);
    }

    fn remove_location(&mut self, token: TokenId, chunk_index: usize) {
        let mut empty = false;
        if let Some(locations) = self.token_locations.get_mut(&token) {
            locations.remove(&chunk_index);
            empty = locations.is_empty();
        }
        if empty {
            self.token_locations.remove(&token);
        }
    }

    fn combine_local_rewrites(left: LocalRewrite, right: LocalRewrite) -> LocalRewrite {
        let mut combined = left;
        for (pair, delta) in right.pair_deltas {
            *combined.pair_deltas.entry(pair).or_insert(0) += delta;
        }
        combined.changed_chunks.extend(right.changed_chunks);
        combined.replacements += right.replacements;
        combined.whole_word_delta += right.whole_word_delta;
        combined
    }

    fn record_merge_rewrite(
        local: &mut LocalRewrite,
        index: usize,
        tokens: &mut Chunk,
        max_pair: Pair,
        merged: TokenId,
        count: i64,
    ) {
        let before_len = tokens.len();
        let mut deltas = AHashMap::new();
        Self::get_stats(tokens, &mut deltas, -count);
        let (replacement, replacements) = merge_chunk(tokens, max_pair, merged);
        if replacements == 0 {
            return;
        }
        Self::get_stats(&replacement, &mut deltas, count);
        for (pair, delta) in deltas {
            *local.pair_deltas.entry(pair).or_insert(0) += delta;
        }
        local.replacements += replacements * count;
        if before_len > 1 && replacement.len() == 1 {
            local.whole_word_delta += count;
        }
        *tokens = replacement;
        local.changed_chunks.push(index);
    }

    fn record_deletion_rewrite(
        local: &mut LocalRewrite,
        index: usize,
        tokens: &mut Chunk,
        bad_token: TokenId,
        parts: &[TokenId],
        count: i64,
    ) {
        let before_len = tokens.len();
        let mut deltas = AHashMap::new();
        Self::get_stats(tokens, &mut deltas, -count);
        let (replacement, replacements) = blow_up_chunk(tokens, bad_token, parts);
        if replacements == 0 {
            return;
        }
        Self::get_stats(&replacement, &mut deltas, count);
        for (pair, delta) in deltas {
            *local.pair_deltas.entry(pair).or_insert(0) += delta;
        }
        local.replacements += replacements * count;
        if before_len == 1 && replacement.len() > 1 {
            local.whole_word_delta -= count;
        }
        *tokens = replacement;
        local.changed_chunks.push(index);
    }

    fn merge_rewrite_in_place(
        &mut self,
        max_pair: Pair,
        merged: TokenId,
        candidates: &[usize],
        text_chunks: &mut [Chunk],
        text_counts: &[i64],
    ) -> (AHashMap<Pair, i64>, i64, i64) {
        let mut result = if candidates.len() < PARALLEL_REWRITE_MIN_CANDIDATES
            || candidates.len().saturating_mul(4) < text_chunks.len()
        {
            let mut local = LocalRewrite::default();
            for &index in candidates {
                Self::record_merge_rewrite(
                    &mut local,
                    index,
                    &mut text_chunks[index],
                    max_pair,
                    merged,
                    text_counts[index],
                );
            }
            local
        } else {
            let mut is_candidate = vec![false; text_chunks.len()];
            for &index in candidates {
                is_candidate[index] = true;
            }
            text_chunks
                .par_iter_mut()
                .enumerate()
                .fold(LocalRewrite::default, |mut local, (index, tokens)| {
                    if is_candidate[index] {
                        Self::record_merge_rewrite(
                            &mut local,
                            index,
                            tokens,
                            max_pair,
                            merged,
                            text_counts[index],
                        );
                    }
                    local
                })
                .reduce(LocalRewrite::default, Self::combine_local_rewrites)
        };

        result.pair_deltas.retain(|_, delta| *delta != 0);
        result.changed_chunks.sort_unstable();
        for index in &result.changed_chunks {
            let tokens = &text_chunks[*index];
            if !tokens.contains(&max_pair.0) {
                self.remove_location(max_pair.0, *index);
            }
            if max_pair.0 != max_pair.1 && !tokens.contains(&max_pair.1) {
                self.remove_location(max_pair.1, *index);
            }
            self.add_location(merged, *index);
        }
        (result.pair_deltas, result.replacements, result.whole_word_delta)
    }

    fn deletion_rewrite_in_place(
        &mut self,
        bad_token: TokenId,
        parts: &[TokenId],
        candidates: &[usize],
        text_chunks: &mut [Chunk],
        text_counts: &[i64],
    ) -> (AHashMap<Pair, i64>, i64, i64) {
        let mut result = if candidates.len() < PARALLEL_REWRITE_MIN_CANDIDATES
            || candidates.len().saturating_mul(4) < text_chunks.len()
        {
            let mut local = LocalRewrite::default();
            for &index in candidates {
                Self::record_deletion_rewrite(
                    &mut local,
                    index,
                    &mut text_chunks[index],
                    bad_token,
                    parts,
                    text_counts[index],
                );
            }
            local
        } else {
            let mut is_candidate = vec![false; text_chunks.len()];
            for &index in candidates {
                is_candidate[index] = true;
            }
            text_chunks
                .par_iter_mut()
                .enumerate()
                .fold(LocalRewrite::default, |mut local, (index, tokens)| {
                    if is_candidate[index] {
                        Self::record_deletion_rewrite(
                            &mut local,
                            index,
                            tokens,
                            bad_token,
                            parts,
                            text_counts[index],
                        );
                    }
                    local
                })
                .reduce(LocalRewrite::default, Self::combine_local_rewrites)
        };

        result.pair_deltas.retain(|_, delta| *delta != 0);
        result.changed_chunks.sort_unstable();
        for index in &result.changed_chunks {
            self.remove_location(bad_token, *index);
            for &part in parts {
                self.add_location(part, *index);
            }
        }
        (result.pair_deltas, result.replacements, result.whole_word_delta)
    }

    fn merge_and_update(
        &mut self,
        max_pair: &Pair,
        text_chunks: &mut [Chunk],
        text_counts: &[i64],
    ) -> (i64, Option<TokenId>) {
        let (first, second) = *max_pair;
        assert!(self.is_unlocked(first) && self.is_unlocked(second));
        assert!(self.get_pair_count(max_pair) > 0);
        let mut merged_bytes = self.arena.bytes(first).to_vec();
        merged_bytes.extend_from_slice(self.arena.bytes(second));
        let merged = self.arena.intern_merge(&merged_bytes, *max_pair);
        let candidates = self.merge_candidates(*max_pair);
        let (deltas, total_merges, whole_word_delta) = self.merge_rewrite_in_place(
            *max_pair,
            merged,
            &candidates,
            text_chunks,
            text_counts,
        );
        self.apply_pair_count_changes(&deltas);
        assert!(
            !self.pair_counts.get(max_pair).is_some(),
            "selected pair should be gone"
        );
        *self.single_counts.entry(merged).or_insert(0) += total_merges;
        *self.single_counts.entry(first).or_insert(0) -= total_merges;
        *self.single_counts.entry(second).or_insert(0) -= total_merges;
        if self.single_counts.get(&first) == Some(&0) {
            self.single_counts.remove(&first);
        }
        if first != second && self.single_counts.get(&second) == Some(&0) {
            self.single_counts.remove(&second);
        }
        self.whole_words += whole_word_delta;
        if self.inf_data.is_super {
            self.unlocked.insert(merged, true);
        }
        (total_merges, (whole_word_delta > 0).then_some(merged))
    }

    fn delete_and_update(
        &mut self,
        bad_token: TokenId,
        parts: &[TokenId],
        text_chunks: &mut [Chunk],
        text_counts: &[i64],
    ) {
        let expected = self.single_counts.get(&bad_token).copied().unwrap_or(0);
        let mut candidates: Vec<usize> = self
            .token_locations
            .get(&bad_token)
            .map(|locations| locations.iter().copied().collect())
            .unwrap_or_default();
        candidates.sort_unstable();
        let (deltas, total_deletions, whole_word_delta) = self.deletion_rewrite_in_place(
            bad_token,
            parts,
            &candidates,
            text_chunks,
            text_counts,
        );
        assert_eq!(expected, total_deletions, "delete count mismatch");
        self.apply_pair_count_changes(&deltas);
        self.whole_words += whole_word_delta;
        *self.single_counts.entry(bad_token).or_insert(0) -= total_deletions;
        assert_eq!(self.single_counts.get(&bad_token), Some(&0));
        self.single_counts.remove(&bad_token);
        for &part in parts {
            *self.single_counts.entry(part).or_insert(0) += total_deletions;
        }
        self.inf_data.deletions.insert(
            self.index_gen.get_next_index(),
            self.arena.bytes(bad_token).to_vec(),
        );
    }

    fn merge_and_delete(
        &mut self,
        best_pair: &Pair,
        i: usize,
        text_chunks: &mut [Chunk],
        text_counts: &[i64],
        print_row: bool,
        verbose: bool,
    ) -> (f64, f64, Option<TokenId>) {
        let start_merge = Instant::now();
        let (left, right) = *best_pair;
        assert!(self.is_unlocked(left) && self.is_unlocked(right));
        let c_ab = self.get_pair_count(best_pair);
        let c_a = *self
            .single_counts
            .get(&left)
            .expect("left not in single_counts");
        let c_b = *self
            .single_counts
            .get(&right)
            .expect("right not in single_counts");

        let ios_a = c_ab as f64 / c_a as f64;
        let ios_b = c_ab as f64 / c_b as f64;

        let (total_merge_cnt, new_unlocked) =
            self.merge_and_update(best_pair, text_chunks, text_counts);

        assert_eq!(
            c_ab, total_merge_cnt,
            "merge count mismatch: {} vs {}",
            c_ab, total_merge_cnt
        );

        let left_bytes = self.arena.bytes(left).to_vec();
        let right_bytes = self.arena.bytes(right).to_vec();
        let mut merged_tok = left_bytes.clone();
        merged_tok.extend_from_slice(&right_bytes);

        // Save merge rule
        self.inf_data.merges.insert(
            self.index_gen.get_next_index(),
            (
                (left_bytes.clone(), right_bytes.clone()),
                c_ab,
                if new_unlocked.is_some() { 1 } else { 0 },
            ),
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
            let left_tok = self.encoder.from_bytes(&left_bytes);
            let right_tok = self.encoder.from_bytes(&right_bytes);
            let nu = match new_unlocked {
                Some(t) => self.encoder.from_bytes(self.arena.bytes(t)),
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
        for (ios, token, direction) in [(ios_a, left, "left"), (ios_b, right, "right")] {
            let token_bytes = self.arena.bytes(token).to_vec();
            if token_bytes.len() > 1
                && ios >= self.inf_data.tau
                && self.single_counts.contains_key(&token)
            {
                let start_this_delete = Instant::now();
                let part_ids: Vec<TokenId> = if self.inf_data.blowup {
                    token_bytes
                        .iter()
                        .map(|byte| self.arena.intern_leaf(&[*byte]))
                        .collect()
                } else {
                    self.arena
                        .parent(token)
                        .map(|(left, right)| vec![left, right])
                        .expect("deleted token has no merge parent")
                };
                let part_bytes: Vec<Vec<u8>> = part_ids
                    .iter()
                    .map(|&part| self.arena.bytes(part).to_vec())
                    .collect();
                self.inf_data
                    .deletion_parts
                    .entry(token_bytes.clone())
                    .or_insert(part_bytes);
                self.delete_and_update(token, &part_ids, text_chunks, text_counts);

                let vocab = self.vocab.as_mut().expect("vocab must exist");
                if vocab.contains(&token_bytes) {
                    vocab.delete(&token_bytes);
                }

                if print_row {
                    let nu = match new_unlocked {
                        Some(t) => self.encoder.from_bytes(self.arena.bytes(t)),
                        None => String::new(),
                    };
                    println!(
                        "*\t{}\t{}\td\t{}\t{}\t{}\t{}\t{}\t{:.5}\t{:.5}\t{:.5}\t{}\t{}\t{}\t{}",
                        i + 1,
                        vocab.len(),
                        self.encoder.from_bytes(&token_bytes),
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

    fn _get_next_regular_merge(&mut self) -> (Option<i32>, Option<(Vec<u8>, Vec<u8>)>, i64, i32) {
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
        let file = std::fs::File::create(&model_file).map_err(|e| TokenizerError::IoError(e))?;
        let mut f = std::io::BufWriter::new(file);

        use std::io::Write;

        writeln!(f, "BoundlessBPE v2 word")?;
        vocab.save(&mut f, &self.byte_single_counts(), None, &self.encoder)?;
        writeln!(f, "words")?;
        self.inf_data.write_to_file(&mut f, &self.encoder)?;

        Ok(())
    }

    fn save_two_pass(&self, file_prefix: &str) -> TokenizerResult<()> {
        assert!(self.inf_data.is_super);
        let word_model = self.word_model.as_ref().expect("word_model must be loaded");
        let word_words = word_model
            .words
            .as_ref()
            .expect("word_model.words must be loaded");
        let word_vocab = word_model
            .vocab
            .as_ref()
            .expect("word_model.vocab must be loaded");
        let vocab = self.vocab.as_ref().expect("vocab must exist");

        let num_supermerges = self.inf_data.merges.len();

        // Copy word-model inference data for the saved model.
        let mut trimmed_words = InferenceData::create_for_training(Pretokenizer::new(
            word_words.pretokenizer.main_pattern_str.as_deref(),
            word_words
                .pretokenizer
                .script_specific_pattern_str
                .as_deref(),
            word_words.pretokenizer.script_specific_scripts.as_deref(),
            word_words.pretokenizer.merge_pattern_str.as_deref(),
        )?);
        // Preserve the word model's operation and configuration data.
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
        let file = std::fs::File::create(&model_file).map_err(|e| TokenizerError::IoError(e))?;
        let mut f = std::io::BufWriter::new(file);

        use std::io::Write;

        writeln!(f, "BoundlessBPE v2 {}", model_type)?;
        vocab_for_save.save(&mut f, &self.byte_single_counts(), None, &self.encoder)?;
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
        let (text_chunks_raw, text_counts) =
            self.pretokenize(filepath, num_lines, max_bytes, save_pretokens, verbose)?;
        let mut text_chunks = intern_chunks(&mut self.arena, text_chunks_raw);
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
            let (merge_time, delete_time, _new_unlocked) = self.merge_and_delete(
                &best_pair,
                i,
                &mut text_chunks,
                &text_counts,
                print_row,
                verbose,
            );

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

        // Make word-model tokens available as supermerge parents. Newly created
        // supermerge tokens are unlocked by merge_and_update.
        self.unlocked = AHashMap::new();
        self.unlocked_default = false;
        for tok in &word_vocab.tokens {
            self.unlocked.insert(self.arena.intern_leaf(tok), true);
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
            let result = ngram_split(
                text_chunks_raw,
                text_counts_raw,
                word_model_ref,
                min_count,
                max_ngram_len,
            );
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
        let mut cached_super: Option<(Option<Pair>, i64)> = None;

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
                    let sp_clone = *sp;
                    let (merge_time, delete_time, _) = self.merge_and_delete(
                        &sp_clone,
                        i,
                        &mut text_chunks,
                        &text_counts,
                        print_row,
                        verbose,
                    );
                    total_merge += merge_time;
                    total_delete += delete_time;
                }
            } else {
                // BoundlessBPE: compete with regular merges
                let start_next = Instant::now();
                let (_reg_op_idx, reg_pair, reg_c_ab, _reg_unlocked_flag) =
                    self._get_next_regular_merge();
                let time_next = start_next.elapsed().as_secs_f64();

                // A tie goes to the regular merge. A supermerge must not be
                // applied before the regular merge that creates one of its
                // parents.
                if super_c_ab > reg_c_ab && super_pair.is_some() {
                    let sp = super_pair.unwrap();
                    let vocab = self.vocab.as_ref().expect("vocab must exist");
                    assert!(
                        vocab.contains(self.arena.bytes(sp.0)),
                        "supermerge parent not in vocab: {:?}",
                        self.encoder.from_bytes(self.arena.bytes(sp.0))
                    );
                    assert!(
                        vocab.contains(self.arena.bytes(sp.1)),
                        "supermerge parent not in vocab: {:?}",
                        self.encoder.from_bytes(self.arena.bytes(sp.1))
                    );
                    let (merge_time, delete_time, _) = self.merge_and_delete(
                        &sp,
                        i,
                        &mut text_chunks,
                        &text_counts,
                        print_row,
                        verbose,
                    );
                    total_merge += merge_time + time_next;
                    total_delete += delete_time;
                    // Supermerge updates the heap; discard the cached candidate.
                } else if let Some(reg_pair) = reg_pair {
                    // Replay the regular merge by adding its token to the vocabulary.
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

                    // The heap is unchanged; reuse this supermerge candidate.
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
        if self.inf_data.is_super && self.current_word_op_idx < self.word_operations_list.len() {
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
    let words = word_model
        .words
        .as_ref()
        .expect("word model words must be loaded");
    let max_idx = *words
        .merges
        .keys()
        .max()
        .expect("word model must have merges");
    let (_, c_ab, _) = &words.merges[&max_idx];
    *c_ab
}

/// Count all ID n-grams (n>=2) with count >= min_cnt using Apriori pruning.
fn count_ngrams_ids(
    chunks: &[Chunk],
    counts: &[i64],
    min_cnt: i64,
    max_len: usize,
) -> AHashMap<Chunk, i64> {
    let overall_start = Instant::now();
    println!(
        "count_ngrams_ids: {} chunks, min_cnt={}",
        chunks.len(),
        min_cnt
    );

    let mut ngram_cnt: AHashMap<Chunk, i64> = AHashMap::new();

    for sz in 1..=max_len {
        let start_time = Instant::now();
        let start_size = ngram_cnt.len();

        for (j, (tokens, &cnt)) in chunks.iter().zip(counts.iter()).enumerate() {
            if j % 500000 == 0 {
                println!(
                    "  sz={} j={} ngrams={} t={:.1}s",
                    sz,
                    j,
                    ngram_cnt.len(),
                    start_time.elapsed().as_secs_f64()
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
                    *ngram_cnt.entry(ngram).or_insert(0) += cnt;
                }
            }
        }

        // Filter below threshold (keep unigrams for next pass's pruning)
        let before = ngram_cnt.len();
        ngram_cnt.retain(|ng, c| ng.len() == 1 || *c >= min_cnt);
        let elapsed = start_time.elapsed().as_secs_f64();
        println!(
            "  sz={}: {} -> {} ngrams ({:.1}s)",
            sz,
            before,
            ngram_cnt.len(),
            elapsed
        );

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
        "count_ngrams_ids: {} results (max_len={}), total time: {:.1}s",
        unique_ngrams,
        max_len_found,
        overall_start.elapsed().as_secs_f64()
    );
    if total_ngrams > 0 {
        println!(
            "N-grams: unique={}, total={}, ratio={:.6}",
            unique_ngrams,
            total_ngrams,
            unique_ngrams as f64 / total_ngrams as f64
        );
    } else {
        println!("N-grams: unique={}, total={}", unique_ngrams, total_ngrams);
    }
    let mut sorted_lens: Vec<usize> = len_dist.keys().copied().collect();
    sorted_lens.sort();
    for n in sorted_lens {
        println!(
            "  len={}: {} unique, {} total",
            n, len_dist[&n], len_total[&n]
        );
    }

    ngram_cnt
}

/// Build a set of all prefixes of n-grams for fast lookup.
fn build_prefix_set_ids(ngram_dict: &AHashMap<Chunk, i64>) -> AHashSet<Chunk> {
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
fn greedy_split_ids(
    chunks: &[Chunk],
    counts: &[i64],
    ngram_dict: &AHashMap<Chunk, i64>,
) -> (Vec<Chunk>, Vec<i64>) {
    let start_time = Instant::now();
    println!(
        "greedy_split_ids: {} chunks, {} ngrams",
        chunks.len(),
        ngram_dict.len()
    );

    let prefix_set = build_prefix_set_ids(ngram_dict);
    let mut result: AHashMap<Chunk, i64> = AHashMap::new();

    for (j, (tokens, &cnt)) in chunks.iter().zip(counts.iter()).enumerate() {
        if j % 500000 == 0 {
            println!(
                "  j={} result_size={} t={:.1}s",
                j,
                result.len(),
                start_time.elapsed().as_secs_f64()
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
                *result.entry(ngram).or_insert(0) += cnt;
                pos += best_len;
            } else {
                // Single token — emit as a length-1 chunk
                let single = vec![tokens[pos]];
                *result.entry(single).or_insert(0) += cnt;
                pos += 1;
            }
        }
    }

    println!(
        "greedy_split_ids: {} unique chunks, time: {:.1}s",
        result.len(),
        start_time.elapsed().as_secs_f64()
    );

    let mut new_chunks: Vec<Chunk> = Vec::with_capacity(result.len());
    let mut new_counts: Vec<i64> = Vec::with_capacity(result.len());
    for (ng, c) in result {
        new_chunks.push(ng);
        new_counts.push(c);
    }

    (new_chunks, new_counts)
}

/// Top-level entry point for n-gram splitting.
fn ngram_split(
    text_chunks: Vec<Chunk>,
    text_counts: Vec<i64>,
    word_model: &Tokenizer,
    min_count_floor: i64,
    max_ngram_len: usize,
) -> (Vec<Chunk>, Vec<i64>) {
    let model_cmin = get_cmin_from_word_model(word_model);
    let min_cnt = model_cmin.max(min_count_floor);
    println!(
        "ngram_split: c_min: model={}, floor={}, using={}",
        model_cmin, min_count_floor, min_cnt
    );

    let unique_input = text_chunks.len();
    let total_input: i64 = text_counts.iter().copied().sum();
    println!(
        "ngram_split input: unique={}, total={}, ratio={:.6}",
        unique_input,
        total_input,
        unique_input as f64 / total_input as f64
    );

    let ngram_dict = count_ngrams_ids(&text_chunks, &text_counts, min_cnt, max_ngram_len);

    if ngram_dict.is_empty() {
        println!("ngram_split: no n-grams found, returning original data unchanged");
        return (text_chunks, text_counts);
    }

    let (new_chunks, new_counts) = greedy_split_ids(&text_chunks, &text_counts, &ngram_dict);

    let unique_output = new_chunks.len();
    let total_output: i64 = new_counts.iter().copied().sum();
    println!(
        "ngram_split output: unique={}, total={}, ratio={:.6}",
        unique_output,
        total_output,
        unique_output as f64 / total_output as f64
    );

    // Stats for n>=2 chunks only (comparable to compute_ngrams.py greedy_split output)
    let mut ngram_unique: usize = 0;
    let mut ngram_total: i64 = 0;
    for (c, &cnt) in new_chunks.iter().zip(new_counts.iter()) {
        if c.len() >= 2 {
            ngram_unique += 1;
            ngram_total += cnt;
        }
    }
    let singleton_unique = unique_output - ngram_unique;
    let singleton_total = total_output - ngram_total;
    if ngram_total > 0 {
        println!(
            "ngram_split n>=2: unique={}, total={}, ratio={:.6}",
            ngram_unique,
            ngram_total,
            ngram_unique as f64 / ngram_total as f64
        );
    } else {
        println!(
            "ngram_split n>=2: unique={}, total={}",
            ngram_unique, ngram_total
        );
    }
    println!(
        "ngram_split singletons: unique={}, total={}",
        singleton_unique, singleton_total
    );
    println!(
        "ngram_split reduction: {} -> {} unique chunks ({:.4}x)",
        unique_input,
        unique_output,
        unique_output as f64 / unique_input as f64
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
        let tokens = vec![1, 2, 3, 1, 2];
        let mut counts: AHashMap<Pair, i64> = AHashMap::new();
        BaseBpeTrainer::get_stats(&tokens, &mut counts, 1);

        assert_eq!(counts.get(&(1, 2)), Some(&2));
        assert_eq!(counts.get(&(2, 3)), Some(&1));
        assert_eq!(counts.get(&(3, 1)), Some(&1));
    }

    #[test]
    fn test_get_stats_overlap() {
        // Verify overlap handling: [a, a, a] contributes (a, a) once.
        let tokens = vec![1, 1, 1];
        let mut counts: AHashMap<Pair, i64> = AHashMap::new();
        BaseBpeTrainer::get_stats(&tokens, &mut counts, 1);

        assert_eq!(counts.get(&(1, 1)), Some(&1));
    }

    #[test]
    fn pair_priority_uses_lexical_bytes_for_equal_counts() {
        let mut trainer = BaseBpeTrainer::new(None).unwrap();
        let b = trainer.arena.intern_leaf(b"b");
        let a = trainer.arena.intern_leaf(b"a");
        let mut heap = PriorityQueue::new();
        heap.push((b, a), trainer.pair_priority((b, a), 1));
        heap.push((a, b), trainer.pair_priority((a, b), 1));

        assert_eq!(heap.peek().map(|(pair, _)| *pair), Some((a, b)));
    }

    #[test]
    fn id_initial_counts_match_serial_and_locations_are_ordered() {
        let chunks = vec![vec![1, 1, 1], vec![2, 1], vec![1, 2, 1], vec![3]];
        let counts = vec![5, 3, 2, 7];
        let parallel = InitialCounts::from_chunks_parallel(&chunks, &counts);
        let serial = InitialCounts::from_chunks(&chunks, &counts);
        assert_eq!(parallel, serial);
        assert_eq!(parallel.token_locations.get(&1), Some(&vec![0, 1, 2]));
    }

    #[test]
    fn parallel_id_rewrites_preserve_all_trainer_invariants() {
        let mut trainer = BaseBpeTrainer::new(None).unwrap();
        let a = trainer.arena.intern_leaf(b"a");
        let b = trainer.arena.intern_leaf(b"b");
        let mut chunks = vec![vec![a, b, a, b], vec![a, b], vec![b, a], vec![a, a, a]];
        let counts = vec![4, 3, 2, 5];
        trainer.initial_counts(&chunks, &counts);

        let (merges, _) = trainer.merge_and_update(&(a, b), &mut chunks, &counts);
        assert_eq!(merges, 11);
        trainer.verify_state(&chunks, &counts);

        let ab = trainer.arena.id_for(b"ab").unwrap();
        trainer.delete_and_update(ab, &[a, b], &mut chunks, &counts);
        trainer.verify_state(&chunks, &counts);
        assert_eq!(chunks[0], vec![a, b, a, b]);
    }

    #[test]
    fn large_parallel_rewrites_preserve_all_trainer_invariants() {
        let mut trainer = BaseBpeTrainer::new(None).unwrap();
        let a = trainer.arena.intern_leaf(b"a");
        let b = trainer.arena.intern_leaf(b"b");
        let chunk_count = 2_065;
        let mut chunks = vec![vec![a, b, a]; chunk_count];
        let counts = vec![1; chunk_count];
        trainer.initial_counts(&chunks, &counts);

        let (merges, _) = trainer.merge_and_update(&(a, b), &mut chunks, &counts);
        assert_eq!(merges, chunk_count as i64);
        assert!(chunks.iter().all(|chunk| chunk.len() == 2));
        trainer.verify_state(&chunks, &counts);
    }

    #[test]
    fn id_trainer_writes_a_deterministic_loadable_model() {
        use std::fs;

        let stem = format!("/tmp/boundlessbpe-id-migration-{}", std::process::id());
        let input = format!("{stem}.jsonl");
        fs::write(&input, "{\"text\":\"abab abab\"}\n{\"text\":\"abab\"}\n").unwrap();

        let train_once = |suffix: &str| {
            let prefix = format!("{stem}-{suffix}");
            let mut trainer = BpeTrainer::new(None).unwrap();
            trainer
                .train(
                    2.0, &input, &prefix, 10, 244, 1, false, 0, 0, false, 0, None,
                )
                .unwrap();
            let model = format!("{prefix}_244.model");
            let bytes = fs::read(&model).unwrap();
            let mut tokenizer = Tokenizer::new();
            tokenizer.load(&model).unwrap();
            assert!(!tokenizer.encode_ordinary_chunks("abab", true).is_empty());
            fs::remove_file(model).unwrap();
            bytes
        };

        let first = train_once("one");
        let second = train_once("two");
        assert_eq!(
            first, second,
            "training output must not depend on Rayon scheduling"
        );
        fs::remove_file(input).unwrap();
    }

    #[test]
    fn superbpe_phase_two_uses_interned_runs_end_to_end() {
        use std::fs;

        let stem = format!("/tmp/boundlessbpe-super-id-migration-{}", std::process::id());
        let input = format!("{stem}.jsonl");
        fs::write(
            &input,
            "{\"text\":\"abab abab abab\"}\n{\"text\":\"abab abab abab\"}\n",
        )
        .unwrap();

        let word_prefix = format!("{stem}-word");
        let mut word_trainer = BpeTrainer::new(None).unwrap();
        word_trainer
            .train(
                2.0,
                &input,
                &word_prefix,
                10,
                248,
                1,
                false,
                usize::MAX,
                0,
                false,
                0,
                None,
            )
            .unwrap();
        // The tiny fixture exhausts its pairs at 246 entries before reaching
        // the requested ceiling of 248.
        let word_model = format!("{word_prefix}_246.model");

        let train_super = |suffix: &str| {
            let super_prefix = format!("{stem}-super-{suffix}");
            let mut super_trainer = SuperBpeTrainer::new(None).unwrap();
            super_trainer
                .train(
                    &input,
                    &super_prefix,
                    10,
                    1,
                    1,
                    &word_model,
                    usize::MAX,
                    0,
                    false,
                    0,
                    None,
                    true,
                    1,
                    3,
                )
                .unwrap();

            let super_model = format!("{super_prefix}_247.model");
            let bytes = fs::read(&super_model).unwrap();
            let mut tokenizer = Tokenizer::new();
            tokenizer.load(&super_model).unwrap();
            assert!(tokenizer.superwords.is_some());
            fs::remove_file(super_model).unwrap();
            bytes
        };
        assert_eq!(train_super("one"), train_super("two"));
        fs::remove_file(word_model).unwrap();
        fs::remove_file(input).unwrap();
    }

    #[test]
    fn parallel_scan_tallies_match_serial_counting() {
        let pretokenizer = Pretokenizer::new(None, None, None, None).unwrap();
        let documents = vec![
            "The quick brown fox.".to_string(),
            "The quick brown fox jumps.".to_string(),
            "Numbers 123 and symbols!".to_string(),
        ];

        let parallel_pretokens = BaseBpeTrainer::tally_pretoken_batch(&pretokenizer, &documents);
        let mut serial_pretokens = AHashMap::new();
        for document in &documents {
            for chunk in pretokenizer.pretokenize(document) {
                *serial_pretokens.entry(chunk).or_insert(0) += 1;
            }
        }
        assert_eq!(parallel_pretokens, serial_pretokens);

        // Use a subset of the observed chunks so this also exercises runs
        // ending at document boundaries and runs separated by ineligible text.
        let eligible_bytes: AHashSet<Vec<u8>> = serial_pretokens
            .keys()
            .filter(|chunk| chunk.chars().any(char::is_alphabetic))
            .map(|chunk| chunk.as_bytes().to_vec())
            .collect();
        let mut arena = TokenArena::new();
        let mut eligible_tokens: Vec<&Vec<u8>> = eligible_bytes.iter().collect();
        eligible_tokens.sort();
        for token in eligible_tokens {
            arena.intern_leaf(token);
        }
        let eligible: AHashSet<TokenId> = eligible_bytes
            .iter()
            .map(|token| arena.id_for(token).unwrap())
            .collect();
        let parallel_runs = BaseBpeTrainer::tally_superword_batch(
            &pretokenizer,
            &eligible,
            &arena,
            &documents,
        );
        let mut serial_runs = AHashMap::new();
        for document in &documents {
            let mut run = Vec::new();
            for chunk in pretokenizer.pretokenize(document) {
                if let Some(token) = arena
                    .id_for(chunk.as_bytes())
                    .filter(|token| eligible.contains(token))
                {
                    run.push(token);
                } else {
                    BaseBpeTrainer::add_superword_run(&mut serial_runs, &mut run);
                }
            }
            BaseBpeTrainer::add_superword_run(&mut serial_runs, &mut run);
        }
        assert_eq!(parallel_runs, serial_runs);
    }

    #[test]
    fn id_ngram_split_counts_and_partitions_without_token_bytes() {
        let chunks = vec![vec![11, 12, 13, 99], vec![11, 12, 13]];
        let counts = vec![4, 3];
        let ngrams = count_ngrams_ids(&chunks, &counts, 3, 3);

        assert_eq!(ngrams.get(&vec![11, 12, 13]), Some(&7));
        assert_eq!(ngrams.get(&vec![11, 12]), Some(&7));

        let (split_chunks, split_counts) = greedy_split_ids(&chunks, &counts, &ngrams);
        let split: AHashMap<Chunk, i64> = split_chunks
            .into_iter()
            .zip(split_counts)
            .collect();
        assert_eq!(split.get(&vec![11, 12, 13]), Some(&7));
        assert_eq!(split.get(&vec![99]), Some(&4));
        assert_eq!(split.len(), 2);
    }

    #[test]
    fn parallel_json_parsing_preserves_document_order() {
        let lines = vec![
            "{\"text\":\"first\"}\n".to_string(),
            "{\"text\":\"second\"}\n".to_string(),
            "{\"text\":\"third\"}\n".to_string(),
        ];

        assert_eq!(
            BaseBpeTrainer::parse_jsonl_text_batch(&lines)
                .into_iter()
                .collect::<TokenizerResult<Vec<_>>>()
                .unwrap(),
            vec!["first", "second", "third"]
        );
    }

    #[test]
    fn scan_ignores_parse_errors_after_the_byte_cap() {
        let lines = vec![
            "{\"text\":\"accepted\"}\n".to_string(),
            "this is not JSON\n".to_string(),
        ];
        let mut total_chars = 0;
        let mut total_bytes = 0;
        let mut documents_read = 0;

        let (documents, reached_cap) = BaseBpeTrainer::collect_documents_until_max_bytes(
            BaseBpeTrainer::parse_jsonl_text_batch(&lines),
            "accepted".len(),
            &mut total_chars,
            &mut total_bytes,
            &mut documents_read,
        )
        .unwrap();

        assert!(reached_cap);
        assert_eq!(documents, vec!["accepted"]);
        assert_eq!((total_chars, total_bytes, documents_read), (8, 8, 1));
    }

    #[test]
    fn scan_reports_character_and_byte_lengths_separately() {
        let parsed = vec![Ok("hé".to_string())];
        let mut total_chars = 0;
        let mut total_bytes = 0;
        let mut documents_read = 0;

        BaseBpeTrainer::collect_documents_until_max_bytes(
            parsed,
            usize::MAX,
            &mut total_chars,
            &mut total_bytes,
            &mut documents_read,
        )
        .unwrap();

        assert_eq!((total_chars, total_bytes, documents_read), (2, 3, 1));
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
