// Copyright 2026-present Kensho Technologies, LLC.
use fancy_regex::Regex;

use crate::constants::{SIMPLE_MERGE_PATTERN, gpt4o_regex};
use crate::error::{TokenizerError, TokenizerResult};
use crate::script_data::{script_of, script_name_to_index, COMMON, INHERITED, UNKNOWN};

/// Convert Python-style `\uXXXX` Unicode escapes to literal Unicode characters.
/// This allows patterns stored in Python format to be compiled by fancy-regex
/// (which only supports `\u{XXXX}` with braces).
/// Handles `\\uXXXX` (escaped backslash) correctly by not converting it.
fn convert_python_unicode_escapes(pattern: &str) -> String {
    let mut result = String::with_capacity(pattern.len());
    let chars: Vec<char> = pattern.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '\\' && i + 1 < chars.len() {
            if chars[i + 1] == '\\' {
                // Escaped backslash — pass through both
                result.push('\\');
                result.push('\\');
                i += 2;
                continue;
            }
            if chars[i + 1] == 'u' && i + 5 < chars.len() && chars.get(i + 2) != Some(&'{') {
                let hex: String = chars[i + 2..i + 6].iter().collect();
                if hex.chars().all(|c| c.is_ascii_hexdigit()) {
                    if let Ok(cp) = u32::from_str_radix(&hex, 16) {
                        if let Some(c) = char::from_u32(cp) {
                            result.push(c);
                            i += 6;
                            continue;
                        }
                    }
                }
            }
        }
        result.push(chars[i]);
        i += 1;
    }
    result
}

/// Pretokenizer that handles script-aware tokenization.
///
/// Translates Python pretokenize.py. Supports two modes:
/// - Simple mode: applies main_pattern to entire text
/// - Script-aware mode: splits by Unicode script, applies script_specific_pattern
///   to CJK/Thai/Myanmar/Khmer/Lao and main_pattern to everything else
pub struct Pretokenizer {
    main_pattern: Regex,
    pub main_pattern_str: Option<String>,
    script_specific_pattern: Option<Regex>,
    pub script_specific_pattern_str: Option<String>,
    pub script_specific_scripts: Option<Vec<String>>,
    is_script_specific: [bool; 256],
    script_aware_mode: bool,
    pub merge_pattern: Regex,
    pub merge_pattern_str: Option<String>,
}

impl Pretokenizer {
    /// Create a new pretokenizer.
    ///
    /// If `script_specific_regex` is None, operates in simple mode.
    /// If provided, enables script-aware mode with the given scripts list.
    pub fn new(
        main_regex: Option<&str>,
        script_specific_regex: Option<&str>,
        script_specific_scripts: Option<&[String]>,
        merge_pattern: Option<&str>,
    ) -> TokenizerResult<Self> {
        // Keep the generated default pattern in scope while compiling it.
        let main_regex_owned;
        let main_regex_str = if let Some(mr) = main_regex {
            mr
        } else {
            main_regex_owned = gpt4o_regex();
            &main_regex_owned
        };

        let main_compile_str = convert_python_unicode_escapes(main_regex_str);
        let main_pattern = Regex::new(&main_compile_str)
            .map_err(|e| TokenizerError::InvalidPattern(format!("main regex '{}': {}", main_regex_str, e)))?;
        let main_pattern_str_owned = Some(main_regex_str.to_string());

        let (script_aware_mode, script_specific_pattern, is_script_specific,
             ss_pattern_str, ss_scripts_vec) =
            if let Some(ss_regex) = script_specific_regex {
                let ss_compile_str = convert_python_unicode_escapes(ss_regex);
                let ss_pattern = Regex::new(&ss_compile_str)
                    .map_err(|e| TokenizerError::InvalidPattern(format!("script specific regex '{}': {}", ss_regex, e)))?;
                let scripts_list = script_specific_scripts.ok_or_else(|| {
                    TokenizerError::ModelError(
                        "script_specific_scripts must be provided when script_specific_regex is not None".to_string(),
                    )
                })?;
                let mut lookup = [false; 256];
                for name in scripts_list {
                    if let Some(idx) = script_name_to_index(name) {
                        lookup[idx as usize] = true;
                    }
                }
                let scripts_vec: Vec<String> = scripts_list.to_vec();
                (true, Some(ss_pattern), lookup,
                 Some(ss_regex.to_string()), Some(scripts_vec))
            } else {
                (false, None, [false; 256], None, None)
            };

        let merge_pattern_str_ref = merge_pattern.unwrap_or(SIMPLE_MERGE_PATTERN);
        let merge_compile_str = convert_python_unicode_escapes(merge_pattern_str_ref);
        let merge_pat = Regex::new(&merge_compile_str)
            .map_err(|e| TokenizerError::InvalidPattern(format!("merge pattern '{}': {}", merge_pattern_str_ref, e)))?;
        let merge_pattern_str_owned = Some(merge_pattern_str_ref.to_string());

        Ok(Self {
            main_pattern,
            main_pattern_str: main_pattern_str_owned,
            script_specific_pattern,
            script_specific_pattern_str: ss_pattern_str,
            script_specific_scripts: ss_scripts_vec,
            is_script_specific,
            script_aware_mode,
            merge_pattern: merge_pat,
            merge_pattern_str: merge_pattern_str_owned,
        })
    }

    /// Check if two byte tokens can be merged.
    pub fn can_merge(&self, left: &[u8], right: &[u8]) -> bool {
        let left_str = match std::str::from_utf8(left) {
            Ok(s) => s,
            Err(_) => return false,
        };
        let right_str = match std::str::from_utf8(right) {
            Ok(s) => s,
            Err(_) => return false,
        };
        self.merge_pattern.is_match(left_str).unwrap_or(false)
            && self.merge_pattern.is_match(right_str).unwrap_or(false)
    }

    /// Check if a byte token could participate in a merge.
    pub fn could_merge(&self, tok: &[u8]) -> bool {
        match std::str::from_utf8(tok) {
            Ok(tok_str) => self.merge_pattern.is_match(tok_str).unwrap_or(false),
            Err(_) => false,
        }
    }

    /// Pretokenize text into a list of string tokens.
    ///
    /// In simple mode, applies main_pattern findall to entire text.
    /// In script-aware mode, splits by Unicode script boundaries first.
    pub fn pretokenize(&self, text: &str) -> Vec<String> {
        if text.is_empty() {
            return Vec::new();
        }

        if !self.script_aware_mode {
            return self.findall(&self.main_pattern, text);
        }

        // Script-aware mode
        let script_specific_pattern = self.script_specific_pattern.as_ref().unwrap();
        let mut tokens: Vec<String> = Vec::new();
        let chars: Vec<char> = text.chars().collect();

        let mut run_start = 0; // byte offset
        let mut current_script: Option<u8> = None;
        let mut last_non_common_byte_end = 0usize; // byte offset of end of last non-Common/Inherited char

        let mut byte_offset = 0usize;

        for &ch in chars.iter() {
            let char_len = ch.len_utf8();
            let cs_idx = script_of(ch);

            // Same script continues
            if Some(cs_idx) == current_script {
                last_non_common_byte_end = byte_offset + char_len;
                byte_offset += char_len;
                continue;
            }

            // Inherited stays with current script
            if cs_idx == INHERITED {
                if current_script.is_none() {
                    current_script = Some(INHERITED);
                }
                last_non_common_byte_end = byte_offset + char_len;
                byte_offset += char_len;
                continue;
            }

            // Common or Unknown - skip if no real script yet
            if cs_idx == COMMON || cs_idx == UNKNOWN {
                if current_script.is_none() {
                    current_script = Some(COMMON);
                }
                // Don't update last_non_common_byte_end
                byte_offset += char_len;
                continue;
            }

            // New real script starts a new run
            if let Some(cs) = current_script {
                if cs != COMMON && cs != INHERITED {
                    let chunk_end = last_non_common_byte_end;
                    let chunk = &text[run_start..chunk_end];

                    if self.is_script_specific[cs as usize] {
                        tokens.extend(self.findall(script_specific_pattern, chunk));
                    } else {
                        tokens.extend(self.findall(&self.main_pattern, chunk));
                    }

                    run_start = chunk_end;
                }
                // else: keep run_start where it is
            }

            current_script = Some(cs_idx);
            last_non_common_byte_end = byte_offset + char_len;
            byte_offset += char_len;
        }

        // Last run
        if let Some(cs) = current_script {
            let remaining = &text[run_start..];
            if cs == COMMON || cs == INHERITED {
                tokens.extend(self.findall(&self.main_pattern, remaining));
            } else if self.is_script_specific[cs as usize] {
                tokens.extend(self.findall(script_specific_pattern, remaining));
            } else {
                tokens.extend(self.findall(&self.main_pattern, remaining));
            }
        }

        tokens
    }

    /// Collect all non-overlapping matches from a regex pattern (like Python findall).
    fn findall(&self, pattern: &Regex, text: &str) -> Vec<String> {
        let mut results = Vec::new();
        let mut pos = 0;
        while pos < text.len() {
            match pattern.find(&text[pos..]) {
                Ok(Some(m)) => {
                    let match_str = &text[pos + m.start()..pos + m.end()];
                    results.push(match_str.to_string());
                    // Advance past this match
                    let advance = m.end();
                    if advance == 0 {
                        // Prevent infinite loop on zero-length match
                        pos += text[pos..].chars().next().map_or(1, |c| c.len_utf8());
                    } else {
                        pos += advance;
                    }
                }
                _ => break,
            }
        }
        results
    }
}


impl std::fmt::Debug for Pretokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pretokenizer")
            .field("script_aware_mode", &self.script_aware_mode)
            .finish()
    }
}
