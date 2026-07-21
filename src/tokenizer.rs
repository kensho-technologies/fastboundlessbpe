// Copyright 2026-present Kensho Technologies, LLC.
use ahash::{AHashMap, AHashSet};
use std::io::BufRead;

use crate::byte_encoding::ByteEncoder;
use crate::error::{TokenizerError, TokenizerResult};
use crate::inference_data::InferenceData;
use crate::vocabulary::Vocabulary;

/// Core BPE tokenizer implementation.
///
/// Translates Python inference.py. Performs encoding/decoding using the
/// BoundlessBPE algorithm with merge/deletion operations, supermerges,
/// and reachable token optimization.
pub struct Tokenizer {
    pub vocab: Option<Vocabulary>,
    pub words: Option<InferenceData>,
    pub superwords: Option<InferenceData>,
    pub reachable_vocab: Option<AHashSet<Vec<u8>>>,
    pub possible_superwords: Option<AHashSet<Vec<u8>>>,
}

impl Tokenizer {
    pub fn new() -> Self {
        Self {
            vocab: None,
            words: None,
            superwords: None,
            reachable_vocab: None,
            possible_superwords: None,
        }
    }

    /// Load a BoundlessBPE v2 model from file.
    ///
    /// File format:
    /// ```text
    /// BoundlessBPE v2 <model_type>
    /// <vocabulary section>
    /// <special_tokens section>
    /// words
    /// <config JSON + merges + deletions>
    /// [superwords]
    /// [<config JSON + merges + deletions>]
    /// ```
    pub fn load(&mut self, model_file: &str) -> TokenizerResult<()> {
        let file = std::fs::File::open(model_file)?;
        let mut reader = std::io::BufReader::new(file);
        let encoder = ByteEncoder::new();

        // Read header
        let mut line = String::new();
        reader.read_line(&mut line)?;
        let header = line.trim().to_string();
        let parts: Vec<&str> = header.split_whitespace().collect();
        if parts.len() != 3 {
            return Err(TokenizerError::ModelError(
                format!("Invalid header: {}", header),
            ));
        }
        if parts[0] != "BoundlessBPE" {
            return Err(TokenizerError::ModelError(
                format!("Invalid format: {}", parts[0]),
            ));
        }
        if parts[1] != "v2" {
            return Err(TokenizerError::ModelError(
                format!("Unsupported version: {}", parts[1]),
            ));
        }
        let model_type = parts[2];
        if model_type != "word" && model_type != "boundless" && model_type != "superbpe" {
            return Err(TokenizerError::ModelError(
                format!("Unknown model type: {}", model_type),
            ));
        }

        // Load vocabulary section
        self.vocab = Some(Vocabulary::load(&mut reader, &encoder)?);

        println!(
            "Loaded vocabulary: {} tokens, {} special tokens, total {}",
            self.vocab.as_ref().unwrap().vocab_size(),
            self.vocab.as_ref().unwrap().special_tokens.len(),
            self.vocab.as_ref().unwrap().total_size(),
        );

        // Read "words" marker
        line.clear();
        reader.read_line(&mut line)?;
        let marker = line.trim();
        if marker != "words" {
            return Err(TokenizerError::ModelError(
                format!("Expected 'words', got '{}'", marker),
            ));
        }

        // Load words section
        self.words = Some(InferenceData::load(&mut reader, &encoder)?);

        println!(
            "Loaded words: {} merges, {} deletions, blowup={}",
            self.words.as_ref().unwrap().merges.len(),
            self.words.as_ref().unwrap().deletions.len(),
            self.words.as_ref().unwrap().blowup,
        );

        // Load superwords section if present
        if model_type == "boundless" || model_type == "superbpe" {
            line.clear();
            reader.read_line(&mut line)?;
            let marker = line.trim();
            if marker != "superwords" {
                return Err(TokenizerError::ModelError(
                    format!("Expected 'superwords', got '{}'", marker),
                ));
            }
            self.superwords = Some(InferenceData::load(&mut reader, &encoder)?);

            println!(
                "Loaded superwords: {} merges",
                self.superwords.as_ref().unwrap().merges.len(),
            );
        }

        // Resolve deletion parts against the final vocabulary. This repairs
        // "double deletion" cases where a deleted token's replacement parts
        // reference another token that was itself deleted (see
        // InferenceData::resolve_deletion_parts).
        let vocab_tokens: AHashSet<Vec<u8>> = self
            .vocab
            .as_ref()
            .expect("vocab must be loaded")
            .token_to_id
            .keys()
            .cloned()
            .collect();
        self.words
            .as_mut()
            .expect("words must be loaded")
            .resolve_deletion_parts(&vocab_tokens);
        if let Some(superwords) = self.superwords.as_mut() {
            superwords.resolve_deletion_parts(&vocab_tokens);
        }

        // Set up possible superwords
        self.setup_superwords();

        // Compute reachable tokens
        println!("Computing reachable tokens...");
        self.reachable_vocab = Some(self.find_reachable_tokens());

        println!(
            "Reachable tokens: {}",
            self.reachable_vocab.as_ref().unwrap().len(),
        );

        Ok(())
    }

    /// Set up the possible_superwords set for supermerge optimization.
    fn setup_superwords(&mut self) {
        let vocab = self.vocab.as_ref().expect("vocab must be loaded");
        let words = self.words.as_ref().expect("words must be loaded");

        let mut possible_superwords = AHashSet::new();

        if let Some(superwords) = &self.superwords {
            // Use supermerge merge pairs to determine possible superword tokens
            for (s1, s2) in superwords.merges_lookup.keys() {
                possible_superwords.insert(s1.clone());
                possible_superwords.insert(s2.clone());
            }
        } else {
            // Word-only model: check could_merge() for each vocab token
            for word in &vocab.tokens {
                if words.pretokenizer.could_merge(word) {
                    possible_superwords.insert(word.clone());
                }
            }
        }

        self.possible_superwords = Some(possible_superwords);
    }

    /// Find all vocabulary tokens that encode to themselves (single token output).
    ///
    /// These "reachable" tokens can skip the full merge/delete pipeline.
    /// This is a major optimization: ~98% of tokens are reachable.
    fn find_reachable_tokens(&self) -> AHashSet<Vec<u8>> {
        let vocab = self.vocab.as_ref().expect("vocab must be loaded");

        // Only word-level tokens
        let word_tokens = vocab.get_word_tokens();

        let mut reachable = AHashSet::new();

        for v in word_tokens {
            // Only consider valid UTF-8 tokens
            if let Ok(text) = std::str::from_utf8(v) {
                // Encode without supercharge (we're computing the supercharge data)
                let tokens = self.encode_ordinary_chunks(text, false);
                if tokens.len() == 1 {
                    reachable.insert(v.clone());
                }
            }
        }

        reachable
    }

    // -----------------------------------------------------------------------
    // Static helpers: merge and blow_up
    // -----------------------------------------------------------------------

    /// Replace all adjacent occurrences of `pair` with their concatenation.
    pub fn merge(tokens: &[Vec<u8>], pair: (&[u8], &[u8])) -> (Vec<Vec<u8>>, i64) {
        let mut result = Vec::with_capacity(tokens.len());
        let merged: Vec<u8> = [pair.0, pair.1].concat();
        let mut i = 0;
        let mut count = 0;
        while i < tokens.len() {
            if i < tokens.len() - 1 && tokens[i] == pair.0 && tokens[i + 1] == pair.1 {
                result.push(merged.clone());
                count += 1;
                i += 2;
            } else {
                result.push(tokens[i].clone());
                i += 1;
            }
        }
        (result, count)
    }

    /// Replace all occurrences of `bad_token` with `parts`.
    pub fn blow_up(tokens: &[Vec<u8>], bad_token: &[u8], parts: &[Vec<u8>]) -> (Vec<Vec<u8>>, i64) {
        let mut result = Vec::with_capacity(tokens.len() + parts.len());
        let mut count = 0;
        for tok in tokens {
            if tok.as_slice() == bad_token {
                result.extend_from_slice(parts);
                count += 1;
            } else {
                result.push(tok.clone());
            }
        }
        (result, count)
    }

    // -----------------------------------------------------------------------
    // Core algorithms
    // -----------------------------------------------------------------------

    /// Apply all merge and deletion operations to each chunk.
    ///
    /// Uses prev_ind tracking with partition_point for O(log n) binary search
    /// on sorted index lists.
    pub fn fast_merge_delete(
        &self,
        text_chunks: &mut [Vec<Vec<u8>>],
        supercharge: bool,
    ) {
        let words = self.words.as_ref().expect("words must be loaded");

        for tokens in text_chunks.iter_mut() {
            // Supercharge optimization: check if chunk is a single reachable token
            if supercharge && tokens.len() > 1 {
                if let Some(ref reachable) = self.reachable_vocab {
                    let original: Vec<u8> = tokens.iter().flatten().copied().collect();
                    if reachable.contains(&original) {
                        *tokens = vec![original];
                        continue;
                    }
                }
            }

            let mut prev_ind: i32 = -1;

            loop {
                let mut min_merge_pair: Option<(Vec<u8>, Vec<u8>)> = None;
                let mut min_merge_ind: i32 = i32::MAX;
                let mut min_deletion_tok: Option<Vec<u8>> = None;
                let mut min_deletion_ind: i32 = i32::MAX;

                // Look for merges
                for i in 0..tokens.len().saturating_sub(1) {
                    let pair = (tokens[i].clone(), tokens[i + 1].clone());
                    if let Some(indices) = words.merges_lookup.get(&pair) {
                        // Binary search: find first index > prev_ind
                        let pos = indices.partition_point(|&idx| idx <= prev_ind);
                        if pos < indices.len() {
                            let merge_idx = indices[pos];
                            if merge_idx < min_merge_ind {
                                min_merge_ind = merge_idx;
                                min_merge_pair = Some(pair);
                            }
                        }
                    }
                }

                // Look for deletions
                for tok in tokens.iter() {
                    if let Some(indices) = words.deletions_lookup.get(tok) {
                        let pos = indices.partition_point(|&idx| idx <= prev_ind);
                        if pos < indices.len() {
                            let deletion_idx = indices[pos];
                            if deletion_idx < min_deletion_ind {
                                min_deletion_ind = deletion_idx;
                                min_deletion_tok = Some(tok.clone());
                            }
                        }
                    }
                }

                // Choose operation with lowest index
                if min_merge_ind < min_deletion_ind {
                    if let Some(pair) = min_merge_pair {
                        let (new_tokens, _) =
                            Self::merge(tokens, (pair.0.as_slice(), pair.1.as_slice()));
                        *tokens = new_tokens;
                        prev_ind = min_merge_ind;
                    } else {
                        break;
                    }
                } else if min_deletion_ind < i32::MAX {
                    if let Some(bad_tok) = min_deletion_tok {
                        let parts = words.get_replacement_parts(&bad_tok);
                        let (new_tokens, _) = Self::blow_up(tokens, &bad_tok, parts);
                        *tokens = new_tokens;
                        prev_ind = min_deletion_ind;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
        }
    }

    /// Apply supermerges to a list of tokens. No prev_ind tracking (no deletions).
    fn supermerge_tokens(&self, tokens: &[Vec<u8>]) -> Vec<Vec<u8>> {
        let superwords = self.superwords.as_ref().expect("superwords must be loaded");
        let mut result = tokens.to_vec();

        loop {
            let mut min_pair: Option<(Vec<u8>, Vec<u8>)> = None;
            let mut min_ind = i32::MAX;

            for i in 0..result.len().saturating_sub(1) {
                let pair = (result[i].clone(), result[i + 1].clone());
                if let Some(indices) = superwords.merges_lookup.get(&pair) {
                    // For supermerges, find the global minimum index (no prev_ind limit)
                    if let Some(&first) = indices.first() {
                        if first < min_ind {
                            min_ind = first;
                            min_pair = Some(pair);
                        }
                    }
                }
            }

            let Some(pair) = min_pair else { break };

            let (new_result, _) = Self::merge(&result, (pair.0.as_slice(), pair.1.as_slice()));
            result = new_result;
        }

        result
    }

    /// Find runs of possible superwords and apply supermerge_tokens to each run.
    fn fast_supermerge(&self, text_chunks: &[Vec<Vec<u8>>]) -> Vec<Vec<u8>> {
        if self.superwords.is_none() {
            // No superwords model, just flatten
            return text_chunks.iter().flatten().cloned().collect();
        }

        let possible_superwords = self
            .possible_superwords
            .as_ref()
            .expect("possible_superwords must be initialized");

        let mut tokens = Vec::new();
        let mut i = 0;

        while i < text_chunks.len() {
            // Skip if not a single token or not in possible superwords
            if text_chunks[i].len() != 1
                || !possible_superwords.contains(&text_chunks[i][0])
            {
                tokens.extend(text_chunks[i].iter().cloned());
                i += 1;
                continue;
            }

            // Find end of run of single-token chunks that could be superwords
            let mut j = i + 1;
            while j < text_chunks.len()
                && text_chunks[j].len() == 1
                && possible_superwords.contains(&text_chunks[j][0])
            {
                j += 1;
            }

            // Need at least two consecutive candidates
            if j == i + 1 {
                tokens.extend(text_chunks[i].iter().cloned());
                i += 1;
                continue;
            }

            // Extract single tokens from the run
            let before_tokens: Vec<Vec<u8>> = (i..j)
                .map(|k| text_chunks[k][0].clone())
                .collect();

            // Apply supermerges
            let merged = self.supermerge_tokens(&before_tokens);
            tokens.extend(merged);

            i = j;
        }

        tokens
    }

    /// Find runs of consecutive single-token chunks in possible_superwords,
    /// create tuple keys, and increment counts. Used during second-pass
    /// pretokenization (SuperBPE/BoundlessBPE) -- counting only, no merging.
    pub fn fast_supermerge_runs(
        &self,
        text_chunks: &[Vec<Vec<u8>>],
        counts: &mut AHashMap<Vec<Vec<u8>>, i64>,
    ) {
        assert!(
            self.superwords.is_none(),
            "fast_supermerge_runs is for training, superwords should not exist yet"
        );
        let possible_superwords = self
            .possible_superwords
            .as_ref()
            .expect("possible_superwords must be initialized");

        let mut i = 0;
        while i < text_chunks.len() {
            // Skip if not a single token or not in possible superwords
            if text_chunks[i].len() != 1
                || !possible_superwords.contains(&text_chunks[i][0])
            {
                i += 1;
                continue;
            }

            // Find end of run of single-token chunks that could be superwords
            let mut j = i + 1;
            while j < text_chunks.len()
                && text_chunks[j].len() == 1
                && possible_superwords.contains(&text_chunks[j][0])
            {
                j += 1;
            }

            // Need at least two consecutive candidates
            if j == i + 1 {
                i += 1;
                continue;
            }

            // Build the tuple key: Vec<Vec<u8>> of single tokens
            let superword_run: Vec<Vec<u8>> = (i..j)
                .map(|k| text_chunks[k][0].clone())
                .collect();
            *counts.entry(superword_run).or_insert(0) += 1;

            i = j;
        }
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Encode text into token byte vectors (without integer ID lookup).
    ///
    /// Steps:
    /// 1. Pretokenize using regex
    /// 2. Split each pretoken into individual bytes
    /// 3. Apply merge/deletion operations (fast_merge_delete)
    /// 4. Apply supermerges (fast_supermerge)
    pub fn encode_ordinary_chunks(&self, text: &str, supercharge: bool) -> Vec<Vec<u8>> {
        let words = self.words.as_ref().expect("words must be loaded");

        // Pretokenize
        let text_chunks_str = words.pretokenizer.pretokenize(text);

        // Convert to bytes and split into single bytes
        let mut text_chunks: Vec<Vec<Vec<u8>>> = text_chunks_str
            .iter()
            .map(|ch| ch.as_bytes().iter().map(|&b| vec![b]).collect())
            .collect();

        // Apply all regular merges and deletions
        self.fast_merge_delete(&mut text_chunks, supercharge);

        // Apply supermerges (or just flatten)
        self.fast_supermerge(&text_chunks)
    }

    /// Encode text to token IDs (ignoring special tokens).
    pub fn encode_ordinary(&self, text: &str) -> TokenizerResult<Vec<i32>> {
        let vocab = self.vocab.as_ref().ok_or_else(|| {
            TokenizerError::VocabularyError("vocab must be loaded".to_string())
        })?;
        let tokens = self.encode_ordinary_chunks(text, true);
        let mut ids = Vec::with_capacity(tokens.len());
        for tok in &tokens {
            let id = vocab.token_to_id.get(tok).ok_or_else(|| {
                TokenizerError::VocabularyError(format!(
                    "Token not found in vocabulary: {:?}",
                    String::from_utf8_lossy(tok)
                ))
            })?;
            ids.push(*id);
        }
        Ok(ids)
    }

    /// Encode text with special token handling.
    ///
    /// `allowed_special`: "all", "none", "none_raise"
    pub fn encode(&self, text: &str, allowed_special: &str) -> TokenizerResult<Vec<i32>> {
        let vocab = self.vocab.as_ref().ok_or_else(|| {
            TokenizerError::VocabularyError("vocab must be loaded".to_string())
        })?;

        let special: &ahash::AHashMap<String, i32> = match allowed_special {
            "all" => &vocab.special_tokens,
            "none" => return self.encode_ordinary(text),
            "none_raise" => {
                for special_token in vocab.special_tokens.keys() {
                    if text.contains(special_token.as_str()) {
                        return Err(TokenizerError::SpecialTokenError(format!(
                            "Special token '{}' found in text",
                            special_token
                        )));
                    }
                }
                return self.encode_ordinary(text);
            }
            _ => {
                return Err(TokenizerError::SpecialTokenError(format!(
                    "allowed_special='{}' not understood",
                    allowed_special
                )));
            }
        };

        if special.is_empty() {
            return self.encode_ordinary(text);
        }

        // Split text by special tokens
        let special_pattern_str = format!(
            "({})",
            special
                .keys()
                .map(|k| fancy_regex::escape(k))
                .collect::<Vec<_>>()
                .join("|")
        );
        let special_regex = fancy_regex::Regex::new(&special_pattern_str)?;

        let mut chunks = Vec::new();
        let mut last_end = 0;

        for mat in special_regex.find_iter(text) {
            let mat = mat?;
            if mat.start() > last_end {
                chunks.push(&text[last_end..mat.start()]);
            }
            chunks.push(&text[mat.start()..mat.end()]);
            last_end = mat.end();
        }
        if last_end < text.len() {
            chunks.push(&text[last_end..]);
        }

        let mut ids = Vec::new();
        for part in chunks {
            if part.is_empty() {
                continue;
            }
            if let Some(&special_id) = special.get(part) {
                ids.push(special_id);
            } else {
                ids.extend(self.encode_ordinary(part)?);
            }
        }

        Ok(ids)
    }

    /// Decode token IDs to raw bytes (lossless).
    pub fn decode_bytes(&self, ids: &[i32]) -> TokenizerResult<Vec<u8>> {
        let vocab = self.vocab.as_ref().ok_or_else(|| {
            TokenizerError::VocabularyError("vocab must be loaded".to_string())
        })?;

        let mut part_bytes = Vec::new();
        for &id in ids {
            if let Some(token) = vocab.id_to_token.get(&id) {
                part_bytes.extend_from_slice(token);
            } else if let Some(special) = vocab.inverse_special_tokens.get(&id) {
                part_bytes.extend_from_slice(special.as_bytes());
            } else {
                return Err(TokenizerError::VocabularyError(format!(
                    "invalid token id: {}",
                    id
                )));
            }
        }
        Ok(part_bytes)
    }

    /// Decode token IDs to string (lossy UTF-8).
    pub fn decode(&self, ids: &[i32]) -> TokenizerResult<String> {
        let bytes = self.decode_bytes(ids)?;
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// Encode multiple texts at once.
    pub fn encode_batch(
        &self,
        texts: &[&str],
        allowed_special: &str,
    ) -> TokenizerResult<Vec<Vec<i32>>> {
        texts
            .iter()
            .map(|text| self.encode(text, allowed_special))
            .collect()
    }

    /// Decode multiple token ID sequences at once.
    pub fn decode_batch(&self, ids_list: &[Vec<i32>]) -> TokenizerResult<Vec<String>> {
        ids_list.iter().map(|ids| self.decode(ids)).collect()
    }

    /// Get vocabulary size.
    pub fn get_vocab_size(&self, with_added_tokens: bool) -> TokenizerResult<i32> {
        let vocab = self.vocab.as_ref().ok_or_else(|| {
            TokenizerError::VocabularyError("vocab must be loaded".to_string())
        })?;
        if with_added_tokens {
            Ok(vocab.total_size())
        } else {
            Ok(vocab.vocab_size())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge() {
        let tokens = vec![
            b"a".to_vec(),
            b"b".to_vec(),
            b"c".to_vec(),
            b"a".to_vec(),
            b"b".to_vec(),
        ];
        let (result, count) = Tokenizer::merge(&tokens, (b"a", b"b"));
        assert_eq!(count, 2);
        assert_eq!(result, vec![b"ab".to_vec(), b"c".to_vec(), b"ab".to_vec()]);
    }

    #[test]
    fn test_blow_up() {
        let tokens = vec![b"ab".to_vec(), b"c".to_vec(), b"ab".to_vec()];
        let parts = vec![b"a".to_vec(), b"b".to_vec()];
        let (result, count) = Tokenizer::blow_up(&tokens, b"ab", &parts);
        assert_eq!(count, 2);
        assert_eq!(
            result,
            vec![
                b"a".to_vec(),
                b"b".to_vec(),
                b"c".to_vec(),
                b"a".to_vec(),
                b"b".to_vec()
            ]
        );
    }
}
