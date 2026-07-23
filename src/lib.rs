// Copyright 2026-present Kensho Technologies, LLC.
#![allow(non_local_definitions)]
mod constants;
mod script_data;
pub mod byte_encoding;
pub mod vocabulary;
pub mod pretokenize;
pub mod inference_data;
pub mod tokenizer;
pub mod trainer;
pub mod error;

pub use tokenizer::Tokenizer;
pub use trainer::{BpeTrainer, BoundlessBpeTrainer, SuperBpeTrainer};
pub use error::{TokenizerError, TokenizerResult};

#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;
#[cfg(feature = "python-bindings")]
use pyo3::types::PyBytes;

/// Python wrapper for the Rust tokenizer
#[cfg(feature = "python-bindings")]
#[pyclass(name = "FastTokenizer")]
pub struct PyTokenizer {
    inner: Tokenizer,
}

#[cfg(feature = "python-bindings")]
#[allow(non_local_definitions)]
#[pymethods]
impl PyTokenizer {
    #[new]
    fn new() -> Self {
        Self {
            inner: Tokenizer::new(),
        }
    }

    /// Load model from file
    fn load(&mut self, model_file: &str) -> PyResult<()> {
        self.inner
            .load(model_file)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Encode text to token IDs (ignoring special tokens)
    #[pyo3(signature = (text, supercharge = true, export_compatible = false))]
    fn encode_ordinary(&self, text: &str, supercharge: bool, export_compatible: bool) -> PyResult<Vec<i32>> {
        let vocab = self.inner.vocab.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("vocab must be loaded"))?;
        let tokens = self.inner.encode_ordinary_chunks(text, supercharge, export_compatible);
        let mut ids = Vec::with_capacity(tokens.len());
        for tok in &tokens {
            let id = vocab.token_to_id.get(tok)
                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(
                    format!("Token not found in vocabulary: {:?}", tok)
                ))?;
            ids.push(*id);
        }
        Ok(ids)
    }

    /// Encode text to token byte vectors
    #[pyo3(signature = (text, supercharge = true, export_compatible = false))]
    fn encode_ordinary_chunks(&self, text: &str, supercharge: bool, export_compatible: bool) -> PyResult<Vec<Vec<u8>>> {
        Ok(self.inner.encode_ordinary_chunks(text, supercharge, export_compatible))
    }

    /// Encode text with special token handling
    #[pyo3(signature = (text, allowed_special = "none_raise", export_compatible = false))]
    fn encode(&self, text: &str, allowed_special: &str, export_compatible: bool) -> PyResult<Vec<i32>> {
        self.inner
            .encode(text, allowed_special, export_compatible)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Encode multiple texts at once
    #[pyo3(signature = (texts, allowed_special = "none_raise", export_compatible = false))]
    fn encode_batch(&self, texts: Vec<&str>, allowed_special: &str, export_compatible: bool) -> PyResult<Vec<Vec<i32>>> {
        self.inner
            .encode_batch(&texts, allowed_special, export_compatible)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Decode token IDs back to text
    fn decode(&self, ids: Vec<i32>) -> PyResult<String> {
        self.inner
            .decode(&ids)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Decode token IDs to raw bytes
    fn decode_bytes<'py>(&self, py: Python<'py>, ids: Vec<i32>) -> PyResult<&'py PyBytes> {
        let bytes = self
            .inner
            .decode_bytes(&ids)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// Decode multiple token ID sequences at once
    fn decode_batch(&self, ids_list: Vec<Vec<i32>>) -> PyResult<Vec<String>> {
        self.inner
            .decode_batch(&ids_list)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Pretokenize text using the loaded model's pretokenizer
    fn pretokenize(&self, text: &str) -> PyResult<Vec<String>> {
        let words = self.inner.words.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("model not loaded"))?;
        Ok(words.pretokenizer.pretokenize(text))
    }

    /// Get vocabulary size
    #[pyo3(signature = (with_added_tokens = true))]
    fn get_vocab_size(&self, with_added_tokens: bool) -> PyResult<i32> {
        self.inner
            .get_vocab_size(with_added_tokens)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// Helper: build a Rust Pretokenizer from optional Python string arguments
// ---------------------------------------------------------------------------

#[cfg(feature = "python-bindings")]
fn build_pretokenizer(
    main_regex: Option<&str>,
    script_specific_regex: Option<&str>,
    script_specific_scripts: Option<Vec<String>>,
) -> PyResult<pretokenize::Pretokenizer> {
    let ss_scripts_ref: Option<Vec<String>> = script_specific_scripts;
    let ss_scripts_slice: Option<&[String]> = ss_scripts_ref.as_deref();
    pretokenize::Pretokenizer::new(
        main_regex,
        script_specific_regex,
        ss_scripts_slice,
        None,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// Default frequency (in merges) for printing `*` progress rows when not verbose.
/// Aligns with the default checkpoint_iterations (8192 = 8 * 1024).
#[cfg(feature = "python-bindings")]
const DEFAULT_PROGRESS_INTERVAL: usize = 1024;

/// Resolve the progress_interval sentinel: None -> 1 if verbose else DEFAULT_PROGRESS_INTERVAL.
/// An explicit value (including 0, which disables progress rows) is used as-is.
#[cfg(feature = "python-bindings")]
fn resolve_progress_interval(progress_interval: Option<usize>, verbose: bool) -> usize {
    match progress_interval {
        Some(v) => v,
        None => if verbose { 1 } else { DEFAULT_PROGRESS_INTERVAL },
    }
}

// ---------------------------------------------------------------------------
// Python trainer wrappers
// ---------------------------------------------------------------------------

#[cfg(feature = "python-bindings")]
#[pyclass(name = "FastBpeTrainer")]
pub struct PyBpeTrainer {
    inner: trainer::BpeTrainer,
}

#[cfg(feature = "python-bindings")]
#[allow(non_local_definitions)]
#[pymethods]
impl PyBpeTrainer {
    #[new]
    #[pyo3(signature = (main_regex = None, script_specific_regex = None, script_specific_scripts = None))]
    fn new(
        main_regex: Option<&str>,
        script_specific_regex: Option<&str>,
        script_specific_scripts: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let pretok = build_pretokenizer(main_regex, script_specific_regex, script_specific_scripts)?;
        let inner = trainer::BpeTrainer::new(Some(pretok))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (tau, filepath, outprefix, num_lines, vocab_size, recalc, blowup, max_bytes = 1000000000, checkpoint_iterations = 8192, verbose = true, progress_interval = None, save_pretokens = None))]
    fn train(
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
        progress_interval: Option<usize>,
        save_pretokens: Option<&str>,
    ) -> PyResult<()> {
        let progress_interval = resolve_progress_interval(progress_interval, verbose);
        self.inner
            .train(
                tau,
                filepath,
                outprefix,
                num_lines,
                vocab_size,
                recalc,
                blowup,
                max_bytes,
                checkpoint_iterations,
                verbose,
                progress_interval,
                save_pretokens,
            )
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn register_special_tokens(&mut self, tokens: Vec<String>) -> PyResult<()> {
        self.inner.register_special_tokens(tokens);
        Ok(())
    }

    fn save(&self, file_prefix: &str) -> PyResult<()> {
        self.inner
            .base
            .save(file_prefix)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }
}

#[cfg(feature = "python-bindings")]
#[pyclass(name = "FastBoundlessBpeTrainer")]
pub struct PyBoundlessBpeTrainer {
    inner: trainer::BoundlessBpeTrainer,
}

#[cfg(feature = "python-bindings")]
#[allow(non_local_definitions)]
#[pymethods]
impl PyBoundlessBpeTrainer {
    #[new]
    #[pyo3(signature = (main_regex = None, script_specific_regex = None, script_specific_scripts = None))]
    fn new(
        main_regex: Option<&str>,
        script_specific_regex: Option<&str>,
        script_specific_scripts: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let pretok = build_pretokenizer(main_regex, script_specific_regex, script_specific_scripts)?;
        let inner = trainer::BoundlessBpeTrainer::new(Some(pretok))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (filepath, outprefix, num_lines, recalc, word_model_file, max_bytes = 1000000000, checkpoint_iterations = 8192, verbose = true, progress_interval = None, save_pretokens = None, greedy_split = false, min_count = 15, max_ngram_len = 30))]
    fn train(
        &mut self,
        filepath: &str,
        outprefix: &str,
        num_lines: usize,
        recalc: usize,
        word_model_file: &str,
        max_bytes: usize,
        checkpoint_iterations: usize,
        verbose: bool,
        progress_interval: Option<usize>,
        save_pretokens: Option<&str>,
        greedy_split: bool,
        min_count: i64,
        max_ngram_len: usize,
    ) -> PyResult<()> {
        let progress_interval = resolve_progress_interval(progress_interval, verbose);
        self.inner
            .train(
                filepath,
                outprefix,
                num_lines,
                recalc,
                word_model_file,
                max_bytes,
                checkpoint_iterations,
                verbose,
                progress_interval,
                save_pretokens,
                greedy_split,
                min_count,
                max_ngram_len,
            )
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn register_special_tokens(&mut self, tokens: Vec<String>) -> PyResult<()> {
        self.inner.register_special_tokens(tokens);
        Ok(())
    }

    fn save(&self, file_prefix: &str) -> PyResult<()> {
        self.inner
            .base
            .save(file_prefix)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }
}

#[cfg(feature = "python-bindings")]
#[pyclass(name = "FastSuperBpeTrainer")]
pub struct PySuperBpeTrainer {
    inner: trainer::SuperBpeTrainer,
}

#[cfg(feature = "python-bindings")]
#[allow(non_local_definitions)]
#[pymethods]
impl PySuperBpeTrainer {
    #[new]
    #[pyo3(signature = (main_regex = None, script_specific_regex = None, script_specific_scripts = None))]
    fn new(
        main_regex: Option<&str>,
        script_specific_regex: Option<&str>,
        script_specific_scripts: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let pretok = build_pretokenizer(main_regex, script_specific_regex, script_specific_scripts)?;
        let inner = trainer::SuperBpeTrainer::new(Some(pretok))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (filepath, outprefix, num_lines, vocab_size, recalc, word_model_file, max_bytes = 1000000000, checkpoint_iterations = 8192, verbose = true, progress_interval = None, save_pretokens = None, greedy_split = false, min_count = 15, max_ngram_len = 30))]
    fn train(
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
        progress_interval: Option<usize>,
        save_pretokens: Option<&str>,
        greedy_split: bool,
        min_count: i64,
        max_ngram_len: usize,
    ) -> PyResult<()> {
        let progress_interval = resolve_progress_interval(progress_interval, verbose);
        self.inner
            .train(
                filepath,
                outprefix,
                num_lines,
                vocab_size,
                recalc,
                word_model_file,
                max_bytes,
                checkpoint_iterations,
                verbose,
                progress_interval,
                save_pretokens,
                greedy_split,
                min_count,
                max_ngram_len,
            )
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn register_special_tokens(&mut self, tokens: Vec<String>) -> PyResult<()> {
        self.inner.register_special_tokens(tokens);
        Ok(())
    }

    fn save(&self, file_prefix: &str) -> PyResult<()> {
        self.inner
            .base
            .save(file_prefix)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }
}

/// Python module definition
#[cfg(feature = "python-bindings")]
#[pymodule]
fn boundlessbpe(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyTokenizer>()?;
    m.add_class::<PyBpeTrainer>()?;
    m.add_class::<PyBoundlessBpeTrainer>()?;
    m.add_class::<PySuperBpeTrainer>()?;
    Ok(())
}
