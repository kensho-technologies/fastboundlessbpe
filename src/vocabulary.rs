// Copyright 2026-present Kensho Technologies, LLC.
use std::io::{BufRead, Write};
use ahash::AHashMap;

use crate::byte_encoding::ByteEncoder;
use crate::error::{TokenizerError, TokenizerResult};

/// Unified vocabulary for BoundlessBPE tokenizers.
///
/// Manages an ordered list of tokens with bidirectional mappings between
/// tokens (bytes) and indices (i32). Translates Python vocabulary.py.
pub struct Vocabulary {
    pub tokens: Vec<Vec<u8>>,
    pub token_to_id: AHashMap<Vec<u8>, i32>,
    pub id_to_token: AHashMap<i32, Vec<u8>>,
    pub is_super: AHashMap<Vec<u8>, bool>,
    pub special_tokens: AHashMap<String, i32>,
    pub inverse_special_tokens: AHashMap<i32, String>,
}

impl Vocabulary {
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            token_to_id: AHashMap::new(),
            id_to_token: AHashMap::new(),
            is_super: AHashMap::new(),
            special_tokens: AHashMap::new(),
            inverse_special_tokens: AHashMap::new(),
        }
    }

    /// Add a token to the vocabulary. Returns existing index if already present.
    pub fn add(&mut self, token: Vec<u8>, is_super_flag: bool) -> i32 {
        if let Some(&idx) = self.token_to_id.get(&token) {
            return idx;
        }
        let idx = self.tokens.len() as i32;
        self.token_to_id.insert(token.clone(), idx);
        self.id_to_token.insert(idx, token.clone());
        self.is_super.insert(token.clone(), is_super_flag);
        self.tokens.push(token);
        idx
    }

    /// Number of regular tokens (excluding special tokens).
    pub fn vocab_size(&self) -> i32 {
        self.tokens.len() as i32
    }

    /// Total size including special tokens.
    pub fn total_size(&self) -> i32 {
        self.tokens.len() as i32 + self.special_tokens.len() as i32
    }

    /// Get only word-level tokens (not super-level).
    pub fn get_word_tokens(&self) -> Vec<&Vec<u8>> {
        self.tokens
            .iter()
            .filter(|t| !self.is_super.get(*t).copied().unwrap_or(false))
            .collect()
    }

    /// Load vocabulary from a reader.
    ///
    /// Reads the vocabulary and special_tokens sections from a v2 model file.
    ///
    /// File format:
    /// ```text
    /// vocabulary
    /// <count>
    /// <idx> <tok_str> <count> <is_super_flag>
    /// ...
    /// special_tokens
    /// <count>
    /// <idx> <tok_str>
    /// ...
    /// ```
    pub fn load<R: BufRead>(reader: &mut R, encoder: &ByteEncoder) -> TokenizerResult<Self> {
        let mut vocab = Self::new();
        let mut line = String::new();

        // Read "vocabulary" header
        line.clear();
        reader.read_line(&mut line)?;
        let header = line.trim();
        if header != "vocabulary" {
            return Err(TokenizerError::ModelError(
                format!("Expected 'vocabulary', got '{}'", header),
            ));
        }

        // Read token count
        line.clear();
        reader.read_line(&mut line)?;
        let count: usize = line.trim().parse()?;

        // Read tokens
        for i in 0..count {
            line.clear();
            reader.read_line(&mut line)?;
            let parts: Vec<&str> = line.trim().split(' ').collect();
            if parts.len() != 4 {
                return Err(TokenizerError::ModelError(
                    format!("Expected 4 parts on vocab line {}, got {}: {:?}", i, parts.len(), parts),
                ));
            }
            let idx: i32 = parts[0].parse()?;
            let tok = encoder.to_bytes(parts[1]);
            let _tok_count: i64 = parts[2].parse()?;
            let is_super_flag = parts[3].parse::<i32>()? == 1;

            let added_idx = vocab.add(tok, is_super_flag);
            if added_idx != idx {
                return Err(TokenizerError::ModelError(
                    format!("Index mismatch: expected {}, got {}", idx, added_idx),
                ));
            }
        }

        // Read "special_tokens" header
        line.clear();
        reader.read_line(&mut line)?;
        let header = line.trim();
        if header != "special_tokens" {
            return Err(TokenizerError::ModelError(
                format!("Expected 'special_tokens', got '{}'", header),
            ));
        }

        // Read special token count
        line.clear();
        reader.read_line(&mut line)?;
        let special_count: usize = line.trim().parse()?;

        // Read special tokens
        for i in 0..special_count {
            line.clear();
            reader.read_line(&mut line)?;
            let trimmed = line.trim();
            // Split on first space only (token string may contain spaces)
            let (idx_str, tok_str) = trimmed.split_once(' ').ok_or_else(|| {
                TokenizerError::ModelError(
                    format!("Expected 2 parts for special token on line {}, got: {:?}", i, trimmed),
                )
            })?;
            let idx: i32 = idx_str.parse()?;
            vocab.special_tokens.insert(tok_str.to_string(), idx);
            vocab.inverse_special_tokens.insert(idx, tok_str.to_string());
        }

        Ok(vocab)
    }

    /// Register special tokens indexed after regular vocabulary.
    pub fn register_special_tokens(&mut self, tokens: &[String]) {
        let start_idx = self.tokens.len() as i32;
        for tok in tokens {
            if !self.special_tokens.contains_key(tok) {
                let idx = start_idx + self.special_tokens.len() as i32;
                self.special_tokens.insert(tok.clone(), idx);
                self.inverse_special_tokens.insert(idx, tok.clone());
            }
        }
    }

    // -----------------------------------------------------------------------
    // Training methods
    // -----------------------------------------------------------------------

    /// Create vocabulary with 243 valid UTF-8 single bytes.
    /// Excludes 0xC0, 0xC1, and 0xF5-0xFF.
    pub fn create_initial() -> Self {
        let mut vocab = Self::new();
        // valid bytes = 0..192, 194..245
        let valid_bytes: Vec<u8> = (0u8..192)
            .chain(194u8..245)
            .collect();
        for b in valid_bytes {
            vocab.add(vec![b], false);
        }
        vocab
    }

    /// Deep-copy all tokens from source vocabulary, preserving is_super flags.
    pub fn create_from_vocab(source: &Vocabulary) -> Self {
        let mut vocab = Self::new();
        for tok in &source.tokens {
            let is_super_flag = source.is_super.get(tok).copied().unwrap_or(false);
            vocab.add(tok.clone(), is_super_flag);
        }
        vocab
    }

    /// Remove token from vocabulary with O(n) re-indexing.
    /// Silently returns if token not found (matching Python).
    pub fn delete(&mut self, token: &[u8]) {
        let idx = match self.token_to_id.get(token) {
            Some(&idx) => idx as usize,
            None => return,
        };

        // Remove from list
        self.tokens.remove(idx);

        // Remove from forward mapping
        self.token_to_id.remove(token);

        // Remove from inverse mapping
        self.id_to_token.remove(&(idx as i32));

        // Remove from is_super mapping
        self.is_super.remove(token);

        // Rebuild mappings for tokens after deleted index
        for i in idx..self.tokens.len() {
            let tok = &self.tokens[i];
            self.token_to_id.insert(tok.clone(), i as i32);
            self.id_to_token.insert(i as i32, tok.clone());
        }

        // Remove old highest index from inverse mapping (now invalid)
        let old_max_idx = self.tokens.len() as i32;
        self.id_to_token.remove(&old_max_idx);
    }

    /// Save vocabulary section to file.
    /// `max_size`: If Some, only save first max_size tokens (BoundlessBPE trimming).
    pub fn save<W: Write>(
        &self,
        writer: &mut W,
        token_counts: &AHashMap<Vec<u8>, i64>,
        max_size: Option<usize>,
        encoder: &ByteEncoder,
    ) -> TokenizerResult<()> {
        let tokens_to_write = match max_size {
            Some(n) => &self.tokens[..n.min(self.tokens.len())],
            None => &self.tokens[..],
        };

        // Write vocabulary section
        writeln!(writer, "vocabulary")?;
        writeln!(writer, "{}", tokens_to_write.len())?;
        for (idx, tok) in tokens_to_write.iter().enumerate() {
            let count = token_counts.get(tok).copied().unwrap_or(0);
            let is_super_flag = if self.is_super.get(tok).copied().unwrap_or(false) { 1 } else { 0 };
            writeln!(writer, "{} {} {} {}", idx, encoder.from_bytes(tok), count, is_super_flag)?;
        }

        // Write special_tokens section
        writeln!(writer, "special_tokens")?;
        writeln!(writer, "{}", self.special_tokens.len())?;
        // Sort by index for consistent ordering
        let mut sorted_special: Vec<(&String, &i32)> = self.special_tokens.iter().collect();
        sorted_special.sort_by_key(|&(_, &idx)| idx);
        for (tok_str, &idx) in sorted_special {
            let adjusted_idx = if let Some(n) = max_size {
                n as i32 + (idx - self.tokens.len() as i32)
            } else {
                idx
            };
            writeln!(writer, "{} {}", adjusted_idx, tok_str)?;
        }

        Ok(())
    }

    /// Check if a token is in the vocabulary.
    pub fn contains(&self, token: &[u8]) -> bool {
        self.token_to_id.contains_key(token)
    }

    /// Return the number of tokens in the vocabulary.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Check if vocabulary is empty.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Verify internal consistency between all vocabulary fields.
    pub fn verify_vocabulary(&self) {
        // Check sizes match
        assert_eq!(
            self.tokens.len(),
            self.token_to_id.len(),
            "tokens length {} != token_to_id length {}",
            self.tokens.len(),
            self.token_to_id.len()
        );
        assert_eq!(
            self.tokens.len(),
            self.id_to_token.len(),
            "tokens length {} != id_to_token length {}",
            self.tokens.len(),
            self.id_to_token.len()
        );
        assert_eq!(
            self.tokens.len(),
            self.is_super.len(),
            "tokens length {} != is_super length {}",
            self.tokens.len(),
            self.is_super.len()
        );

        // Check each token's mappings are consistent
        for (idx, token) in self.tokens.iter().enumerate() {
            assert!(
                self.token_to_id.contains_key(token),
                "token {:?} at index {} not in token_to_id",
                token,
                idx
            );
            assert_eq!(
                *self.token_to_id.get(token).unwrap(),
                idx as i32,
                "token_to_id mismatch at index {}",
                idx
            );
            assert!(
                self.id_to_token.contains_key(&(idx as i32)),
                "index {} not in id_to_token",
                idx
            );
            assert_eq!(
                self.id_to_token.get(&(idx as i32)).unwrap(),
                token,
                "id_to_token mismatch at index {}",
                idx
            );
            assert!(
                self.is_super.contains_key(token),
                "token {:?} at index {} not in is_super",
                token,
                idx
            );
        }

        // Check special tokens consistency
        assert_eq!(
            self.special_tokens.len(),
            self.inverse_special_tokens.len(),
        );
        for (tok_str, &idx) in &self.special_tokens {
            assert!(
                idx >= self.tokens.len() as i32,
                "special token {} has index {} < vocab size {}",
                tok_str,
                idx,
                self.tokens.len()
            );
            assert!(self.inverse_special_tokens.contains_key(&idx));
            assert_eq!(self.inverse_special_tokens.get(&idx).unwrap(), tok_str);
        }
    }
}
