#![allow(dead_code)]
// Copyright 2026-present Kensho Technologies, LLC.
/// Regex constants ported from Python regexconstants.py
///
/// Most patterns are now loaded from model files at runtime.
/// These constants serve as defaults and for reference.

/// Improved merge pattern (legacy) -- validates merge eligibility with full match.
/// Requires at least one letter, allows letters with combining marks, spaces, underscores, apostrophes.
pub const IMPROVED_MERGE_PATTERN: &str = "^(?=.+\\p{L})(?:\\p{L}\\p{M}*|[ _'\\u2019])+$";

/// Simple merge pattern -- token contains at least one Unicode letter.
/// Used as a containment check (not full match): any letter anywhere makes the token merge-eligible.
pub const SIMPLE_MERGE_PATTERN: &str = "\\p{L}";

/// Word-level regex parts for pretokenization
/// Note: Unicode escapes use literal chars (not \u{...}) so patterns are portable to Python.
pub const WORD_LEVEL_REGEX_PARTS: &[&str] = &[
    // Words with lowercase and optional contractions
    " ?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}\u{200d}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}\u{200d}]+(?:['\u{2019}][\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+)?",
    // Words with uppercase and optional contractions
    " ?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}\u{200d}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}\u{200d}]*(?:['\u{2019}][\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+)?",
    // Keycap sequences (0-9, #, *) - BEFORE general emoji
    "[0-9#*]\u{FE0F}?\u{20E3}",
    // Regional Indicator emoji pairs (flags)
    "[\u{1F1E6}-\u{1F1FF}]{2}",
    // Emoji with modifiers (skin tones, variation selectors, ZWJ)
    "\\p{Extended_Pictographic}[\u{1F3FB}-\u{1F3FF}]?[\u{FE0F}\u{FE0E}]?(?:\u{200d}\\p{Extended_Pictographic}[\u{1F3FB}-\u{1F3FF}]?[\u{FE0F}\u{FE0E}]?)*",
    // Other symbols and punctuation with optional leading space (max 16)
    " ?(?:[\\p{S}\\p{P}]\\p{M}*){1,16}",
    // Horizontal whitespace with following line endings (max 16 chars each)
    "[^\\S\\r\\n]{0,16}[\\r\\n]{1,16}",
    // Remaining whitespace (max 16 chars)
    "[^\\S\\r\\n]{1,16}(?![^\\s\\p{N}\\p{C}\\p{M}])",
    // Cleanup remaining whitespace
    "\\s{1,16}",
    // Numbers (1-3 digits, grouping by threes right to left)
    "\\p{N}{1,3}(?=(?:\\p{N}{3})*(?:\\P{N}|$))",
    // Separator, control, or combining marks
    "[\\p{Z}\\p{C}\\p{M}]+",
];

/// Default scripts requiring character-by-character tokenization
pub const DEFAULT_SCRIPT_SPECIFIC_SCRIPTS: &[&str] = &[
    "Han", "Hiragana", "Katakana", "Thai",
    "Myanmar", "Khmer", "Lao",
];

/// Helper function to create word-level regex pattern
pub fn word_level_regex() -> String {
    WORD_LEVEL_REGEX_PARTS.join("|")
}

/// GPT4O regex parts for pretokenization
pub const GPT4O_REGEX_PARTS: &[&str] = &[
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"\p{N}{1,3}",
    r" ?[^\s\p{L}\p{N}]+[\r\n/]*",
    r"\s*[\r\n]+",
    r"\s+(?!\S)",
    r"\s+",
];

/// Helper function to create GPT4O regex pattern
pub fn gpt4o_regex() -> String {
    GPT4O_REGEX_PARTS.join("|")
}

/// Script-specific GPT4O regex parts (character-level for CJK etc.)
pub const SCRIPT_SPECIFIC_GPT4O_REGEX_PARTS: &[&str] = &[
    r" ?\p{L}\p{M}*",
    r"\p{N}{1,3}",
    r" ?[^\s\p{L}\p{N}]+[\r\n/]*",
    r"\s*[\r\n]+",
    r"\s+(?!\S)",
    r"\s+",
];

/// Helper function to create script-specific GPT4O regex pattern
pub fn script_specific_gpt4o_regex() -> String {
    SCRIPT_SPECIFIC_GPT4O_REGEX_PARTS.join("|")
}
