// Copyright 2026-present Kensho Technologies, LLC.
use std::io::{BufRead, Write};
use ahash::AHashMap;

use crate::byte_encoding::ByteEncoder;
use crate::pretokenize::Pretokenizer;
use crate::error::{TokenizerError, TokenizerResult};

/// Data and methods needed for tokenization inference.
///
/// Translates Python inferencedata.py. Encapsulates merge/deletion operations,
/// configuration, and the pretokenizer. Vocabulary is handled separately.
pub struct InferenceData {
    /// idx -> ((left, right), count, unlocked_flag)
    pub merges: AHashMap<i32, ((Vec<u8>, Vec<u8>), i64, i32)>,
    /// idx -> token
    pub deletions: AHashMap<i32, Vec<u8>>,
    /// (left, right) -> sorted list of indices
    pub merges_lookup: AHashMap<(Vec<u8>, Vec<u8>), Vec<i32>>,
    /// token -> sorted list of indices
    pub deletions_lookup: AHashMap<Vec<u8>, Vec<i32>>,
    /// deleted token -> replacement parts
    pub deletion_parts: AHashMap<Vec<u8>, Vec<Vec<u8>>>,
    pub tau: f64,
    pub is_super: bool,
    pub superbpe_mode: bool,
    pub blowup: bool,
    pub pretokenizer: Pretokenizer,
}

impl InferenceData {
    /// Load inference data from a reader.
    ///
    /// Reads JSON config line, merges section, and deletions section.
    /// Builds inverted lookup indices and precomputes deletion_parts.
    pub fn load<R: BufRead>(reader: &mut R, encoder: &ByteEncoder) -> TokenizerResult<Self> {
        let mut line = String::new();

        // Read JSON config line
        line.clear();
        reader.read_line(&mut line)?;
        let config: serde_json::Value = serde_json::from_str(line.trim())?;

        let tau = config["tau"].as_f64().ok_or_else(|| {
            TokenizerError::ModelError("tau must be in config".to_string())
        })?;
        let is_super = config["is_super"].as_bool().ok_or_else(|| {
            TokenizerError::ModelError("is_super must be in config".to_string())
        })?;
        let superbpe_mode = config["superbpe_mode"].as_bool().ok_or_else(|| {
            TokenizerError::ModelError("superbpe_mode must be in config".to_string())
        })?;
        let blowup = config["blowup"].as_bool().ok_or_else(|| {
            TokenizerError::ModelError("blowup must be in config".to_string())
        })?;

        // Extract regex patterns from config
        let main_regex = config["main_regex"].as_str();
        let script_specific_regex = config["script_specific_regex"].as_str();
        let script_specific_scripts: Option<Vec<String>> = config["script_specific_scripts"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            });
        let merge_pattern = config["merge_pattern"].as_str();

        // Create Pretokenizer from config
        let pretokenizer = Pretokenizer::new(
            main_regex,
            script_specific_regex,
            script_specific_scripts.as_deref(),
            merge_pattern,
        )?;

        // Read "merges" header
        line.clear();
        reader.read_line(&mut line)?;
        let header = line.trim();
        if header != "merges" {
            return Err(TokenizerError::ModelError(
                format!("Expected 'merges', got '{}'", header),
            ));
        }

        // Read merges with counts (v2 format: idx tok1 tok2 count unlocked_flag)
        let merges = read_merges_with_counts(reader, encoder)?;

        // Read "deletions" header
        line.clear();
        reader.read_line(&mut line)?;
        let header = line.trim();
        if header != "deletions" {
            return Err(TokenizerError::ModelError(
                format!("Expected 'deletions', got '{}'", header),
            ));
        }

        // Read deletions (v2 format: idx token)
        let deletions = read_deletions(reader, encoder)?;

        // Build inverted lookup indices
        let mut merges_lookup: AHashMap<(Vec<u8>, Vec<u8>), Vec<i32>> = AHashMap::new();
        for (&idx, ((left, right), _count, _unlocked_flag)) in &merges {
            merges_lookup
                .entry((left.clone(), right.clone()))
                .or_default()
                .push(idx);
        }

        let mut deletions_lookup: AHashMap<Vec<u8>, Vec<i32>> = AHashMap::new();
        for (&idx, token) in &deletions {
            deletions_lookup
                .entry(token.clone())
                .or_default()
                .push(idx);
        }

        // Sort index lists for binary search
        for idx_list in merges_lookup.values_mut() {
            idx_list.sort();
        }
        for idx_list in deletions_lookup.values_mut() {
            idx_list.sort();
        }

        // Precompute replacement parts for each deleted token
        let mut deletion_parts: AHashMap<Vec<u8>, Vec<Vec<u8>>> = AHashMap::new();
        for token in deletions.values() {
            if !deletion_parts.contains_key(token) {
                if !blowup {
                    // Find the merge pair that created this token
                    let mut parts = None;
                    for ((left, right), _cnt, _unlocked_flag) in merges.values() {
                        let mut combined = left.clone();
                        combined.extend_from_slice(right);
                        if combined == *token {
                            parts = Some(vec![left.clone(), right.clone()]);
                            break;
                        }
                    }
                    let parts = parts.ok_or_else(|| {
                        TokenizerError::ModelError(format!(
                            "couldn't find merge for deleted token: {:?}",
                            token
                        ))
                    })?;
                    deletion_parts.insert(token.clone(), parts);
                } else {
                    // Blow up to individual bytes
                    let parts: Vec<Vec<u8>> = token.iter().map(|&b| vec![b]).collect();
                    deletion_parts.insert(token.clone(), parts);
                }
            }
        }

        Ok(Self {
            merges,
            deletions,
            merges_lookup,
            deletions_lookup,
            deletion_parts,
            tau,
            is_super,
            superbpe_mode,
            blowup,
            pretokenizer,
        })
    }

    /// Get replacement parts for a deleted token (O(1) lookup).
    pub fn get_replacement_parts(&self, bad_token: &[u8]) -> &[Vec<u8>] {
        self.deletion_parts
            .get(bad_token)
            .expect("couldn't find replacement parts for deleted token")
    }

    // -----------------------------------------------------------------------
    // Training methods
    // -----------------------------------------------------------------------

    /// Factory creating empty instance for training.
    pub fn create_for_training(pretokenizer: Pretokenizer) -> Self {
        Self {
            merges: AHashMap::new(),
            deletions: AHashMap::new(),
            merges_lookup: AHashMap::new(),
            deletions_lookup: AHashMap::new(),
            deletion_parts: AHashMap::new(),
            tau: 1.1,
            is_super: false,
            superbpe_mode: false,
            blowup: false,
            pretokenizer,
        }
    }

    /// JSON-encode a string with non-ASCII chars escaped as \uXXXX (matching Python's ensure_ascii=True).
    fn json_string_ascii(s: &str) -> String {
        let mut out = String::with_capacity(s.len() + 2);
        out.push('"');
        for ch in s.chars() {
            match ch {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c if (c as u32) < 0x20 => {
                    out.push_str(&format!("\\u{:04x}", c as u32));
                }
                c if !c.is_ascii() => {
                    let cp = c as u32;
                    if cp <= 0xFFFF {
                        out.push_str(&format!("\\u{:04x}", cp));
                    } else {
                        // Surrogate pair for chars outside BMP
                        let cp = cp - 0x10000;
                        out.push_str(&format!("\\u{:04x}\\u{:04x}", 0xD800 + (cp >> 10), 0xDC00 + (cp & 0x3FF)));
                    }
                }
                c => out.push(c),
            }
        }
        out.push('"');
        out
    }

    /// Write inference data to a writer (config JSON, merges, deletions).
    pub fn write_to_file<W: Write>(
        &self,
        writer: &mut W,
        encoder: &ByteEncoder,
    ) -> TokenizerResult<()> {
        // Build config JSON
        let main_regex = self.pretokenizer.main_pattern_str.as_deref();
        let script_specific_regex = self.pretokenizer.script_specific_pattern_str.as_deref();
        let merge_pattern = self.pretokenizer.merge_pattern_str.as_deref();

        let script_specific_scripts_json = match &self.pretokenizer.script_specific_scripts {
            Some(scripts) => {
                let mut sorted = scripts.clone();
                sorted.sort();
                let items: Vec<String> = sorted.iter().map(|s| format!("\"{}\"", s)).collect();
                format!("[{}]", items.join(", "))
            }
            None => "null".to_string(),
        };

        let config = format!(
            "{{\"tau\": {}, \"is_super\": {}, \"superbpe_mode\": {}, \"blowup\": {}, \"main_regex\": {}, \"script_specific_regex\": {}, \"script_specific_scripts\": {}, \"merge_pattern\": {}}}",
            self.tau,
            self.is_super,
            self.superbpe_mode,
            self.blowup,
            match main_regex {
                Some(r) => Self::json_string_ascii(r),
                None => "null".to_string(),
            },
            match script_specific_regex {
                Some(r) => Self::json_string_ascii(r),
                None => "null".to_string(),
            },
            script_specific_scripts_json,
            match merge_pattern {
                Some(r) => Self::json_string_ascii(r),
                None => "null".to_string(),
            },
        );
        writeln!(writer, "{}", config)?;

        // Write merges section
        writeln!(writer, "merges")?;
        writeln!(writer, "{}", self.merges.len())?;
        // Sort by index for consistent output
        let mut merge_entries: Vec<_> = self.merges.iter().collect();
        merge_entries.sort_by_key(|(&idx, _)| idx);
        for (&idx, ((left, right), count, unlocked_flag)) in &merge_entries {
            writeln!(
                writer,
                "{} {} {} {} {}",
                idx,
                encoder.from_bytes(left),
                encoder.from_bytes(right),
                count,
                unlocked_flag
            )?;
        }

        // Write deletions section
        writeln!(writer, "deletions")?;
        writeln!(writer, "{}", self.deletions.len())?;
        let mut deletion_entries: Vec<_> = self.deletions.iter().collect();
        deletion_entries.sort_by_key(|(&idx, _)| idx);
        for (&idx, token) in &deletion_entries {
            writeln!(writer, "{} {}", idx, encoder.from_bytes(token))?;
        }

        Ok(())
    }

    /// Trim merges and deletions to keep only first num_ops operations (by index order).
    pub fn trim_operations_to(&mut self, num_ops: usize) {
        assert!(!self.is_super, "trim_operations_to should only be called on word models");
        assert!(
            !self.merges.is_empty() || !self.deletions.is_empty(),
            "No operations to trim"
        );

        // Collect all indices, sort, keep first num_ops
        let mut all_indices: Vec<i32> = self
            .merges
            .keys()
            .chain(self.deletions.keys())
            .copied()
            .collect();
        all_indices.sort();
        let indices_to_keep: ahash::AHashSet<i32> =
            all_indices.iter().take(num_ops).copied().collect();

        // Filter operations
        self.merges.retain(|k, _| indices_to_keep.contains(k));
        self.deletions.retain(|k, _| indices_to_keep.contains(k));

        // Rebuild merges_lookup
        self.merges_lookup.clear();
        for (&idx, ((left, right), _count, _unlocked_flag)) in &self.merges {
            self.merges_lookup
                .entry((left.clone(), right.clone()))
                .or_default()
                .push(idx);
        }
        for idx_list in self.merges_lookup.values_mut() {
            idx_list.sort();
        }

        // Rebuild deletions_lookup
        self.deletions_lookup.clear();
        for (&idx, token) in &self.deletions {
            self.deletions_lookup
                .entry(token.clone())
                .or_default()
                .push(idx);
        }
        for idx_list in self.deletions_lookup.values_mut() {
            idx_list.sort();
        }

        // Rebuild deletion_parts
        self.deletion_parts.clear();
        for token in self.deletions.values() {
            if !self.deletion_parts.contains_key(token) {
                if !self.blowup {
                    let mut parts = None;
                    for ((left, right), _cnt, _unlocked_flag) in self.merges.values() {
                        let mut combined = left.clone();
                        combined.extend_from_slice(right);
                        if combined == *token {
                            parts = Some(vec![left.clone(), right.clone()]);
                            break;
                        }
                    }
                    let parts = parts.expect("couldn't find merge for deleted token");
                    self.deletion_parts.insert(token.clone(), parts);
                } else {
                    let parts: Vec<Vec<u8>> = token.iter().map(|&b| vec![b]).collect();
                    self.deletion_parts.insert(token.clone(), parts);
                }
            }
        }
    }

    /// Verify that merge+deletion indices are sequential from minimum.
    pub fn verify_indices(&self) {
        let mut indices: Vec<i32> = self
            .merges
            .keys()
            .chain(self.deletions.keys())
            .copied()
            .collect();
        indices.sort();

        if !indices.is_empty() {
            let min_index = indices[0];
            for (i, &ind) in indices.iter().enumerate() {
                let expected = min_index + i as i32;
                assert_eq!(
                    expected, ind,
                    "Index gap: expected {}, got {}",
                    expected, ind
                );
            }
        }
    }
}

impl std::fmt::Debug for InferenceData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InferenceData")
            .field("merges_count", &self.merges.len())
            .field("deletions_count", &self.deletions.len())
            .field("tau", &self.tau)
            .field("is_super", &self.is_super)
            .field("blowup", &self.blowup)
            .finish()
    }
}

/// Read merges section with counts (v2 format).
/// Each line: <idx> <tok1> <tok2> <count> <unlocked_flag>
fn read_merges_with_counts<R: BufRead>(
    reader: &mut R,
    encoder: &ByteEncoder,
) -> TokenizerResult<AHashMap<i32, ((Vec<u8>, Vec<u8>), i64, i32)>> {
    let mut line = String::new();

    // Read count
    line.clear();
    reader.read_line(&mut line)?;
    let n: usize = line.trim().parse()?;

    let mut merges = AHashMap::with_capacity(n);

    for i in 0..n {
        line.clear();
        reader.read_line(&mut line)?;
        let parts: Vec<&str> = line.trim().split(' ').collect();
        if parts.len() != 5 {
            return Err(TokenizerError::ModelError(
                format!("Expected 5 fields on merge line {}, got {}: {:?}", i, parts.len(), parts),
            ));
        }
        let idx: i32 = parts[0].parse()?;
        let tok1 = encoder.to_bytes(parts[1]);
        let tok2 = encoder.to_bytes(parts[2]);
        let count: i64 = parts[3].parse()?;
        let unlocked_flag: i32 = parts[4].parse()?;

        merges.insert(idx, ((tok1, tok2), count, unlocked_flag));
    }

    Ok(merges)
}

/// Read deletions section (v2 format).
/// Each line: <idx> <token>
fn read_deletions<R: BufRead>(
    reader: &mut R,
    encoder: &ByteEncoder,
) -> TokenizerResult<AHashMap<i32, Vec<u8>>> {
    let mut line = String::new();

    // Read count
    line.clear();
    reader.read_line(&mut line)?;
    let n: usize = line.trim().parse()?;

    let mut deletions = AHashMap::with_capacity(n);

    for i in 0..n {
        line.clear();
        reader.read_line(&mut line)?;
        let parts: Vec<&str> = line.trim().split(' ').collect();
        if parts.len() != 2 {
            return Err(TokenizerError::ModelError(
                format!("Expected 2 fields on deletion line {}, got {}: {:?}", i, parts.len(), parts),
            ));
        }
        let idx: i32 = parts[0].parse()?;
        let tok = encoder.to_bytes(parts[1]);

        deletions.insert(idx, tok);
    }

    Ok(deletions)
}
