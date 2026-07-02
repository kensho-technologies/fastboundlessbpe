// Copyright 2026-present Kensho Technologies, LLC.
use std::fmt;

/// Custom error types for BoundlessBPE tokenizer operations
#[derive(Debug)]
pub enum TokenizerError {
    /// Model file format or loading errors
    ModelError(String),
    
    /// Invalid regex pattern errors
    InvalidPattern(String),
    
    /// Vocabulary or token lookup errors
    VocabularyError(String),
    
    /// Special token handling errors
    SpecialTokenError(String),
    
    /// I/O errors during file operations
    IoError(std::io::Error),
    
    /// JSON parsing errors
    JsonError(serde_json::Error),
    
    /// Regex compilation errors
    RegexError(fancy_regex::Error),
    
    /// UTF-8 decoding errors
    EncodingError(std::str::Utf8Error),
    
    /// Parse errors for numbers
    ParseError(std::num::ParseIntError),

    /// Parse errors for floats
    ParseFloatError(std::num::ParseFloatError),
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenizerError::ModelError(msg) => write!(f, "Model error: {}", msg),
            TokenizerError::InvalidPattern(msg) => write!(f, "Invalid pattern: {}", msg),
            TokenizerError::VocabularyError(msg) => write!(f, "Vocabulary error: {}", msg),
            TokenizerError::SpecialTokenError(msg) => write!(f, "Special token error: {}", msg),
            TokenizerError::IoError(err) => write!(f, "I/O error: {}", err),
            TokenizerError::JsonError(err) => write!(f, "JSON error: {}", err),
            TokenizerError::RegexError(err) => write!(f, "Regex error: {}", err),
            TokenizerError::EncodingError(err) => write!(f, "Encoding error: {}", err),
            TokenizerError::ParseError(err) => write!(f, "Parse error: {}", err),
            TokenizerError::ParseFloatError(err) => write!(f, "Parse float error: {}", err),
        }
    }
}

impl std::error::Error for TokenizerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            TokenizerError::IoError(err) => Some(err),
            TokenizerError::JsonError(err) => Some(err),
            TokenizerError::RegexError(err) => Some(err),
            TokenizerError::EncodingError(err) => Some(err),
            TokenizerError::ParseError(err) => Some(err),
            TokenizerError::ParseFloatError(err) => Some(err),
            _ => None,
        }
    }
}

// Automatic conversions from common error types
impl From<std::io::Error> for TokenizerError {
    fn from(err: std::io::Error) -> Self {
        TokenizerError::IoError(err)
    }
}

impl From<serde_json::Error> for TokenizerError {
    fn from(err: serde_json::Error) -> Self {
        TokenizerError::JsonError(err)
    }
}

impl From<fancy_regex::Error> for TokenizerError {
    fn from(err: fancy_regex::Error) -> Self {
        TokenizerError::RegexError(err)
    }
}

impl From<std::str::Utf8Error> for TokenizerError {
    fn from(err: std::str::Utf8Error) -> Self {
        TokenizerError::EncodingError(err)
    }
}

impl From<std::num::ParseIntError> for TokenizerError {
    fn from(err: std::num::ParseIntError) -> Self {
        TokenizerError::ParseError(err)
    }
}

impl From<std::num::ParseFloatError> for TokenizerError {
    fn from(err: std::num::ParseFloatError) -> Self {
        TokenizerError::ParseFloatError(err)
    }
}

/// Result type alias for tokenizer operations
pub type TokenizerResult<T> = Result<T, TokenizerError>;