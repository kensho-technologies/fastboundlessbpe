// Copyright 2026-present Kensho Technologies, LLC.

//! Compact, stable token identities used only while training.
//!
//! These IDs must not be confused with [`Vocabulary`](crate::vocabulary::Vocabulary)
//! indices: vocabulary indices are renumbered when BoundlessBPE deletes a token,
//! while a training token ID must stay valid for the entire training run.

use std::collections::BTreeSet;
use std::hash::{Hash, Hasher};

use ahash::{AHashMap, AHasher};
use rayon::prelude::*;

/// A stable, append-only identity for a token during one training run.
pub(crate) type TokenId = u32;

/// An adjacent pair of training tokens.
pub(crate) type Pair = (TokenId, TokenId);

/// A training chunk represented solely by stable token IDs.
pub(crate) type Chunk = Vec<TokenId>;

/// All trainer state derived from a fixed set of aggregated chunks.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct InitialCounts {
    pub(crate) pair_counts: AHashMap<Pair, i64>,
    pub(crate) single_counts: AHashMap<TokenId, i64>,
    pub(crate) token_locations: AHashMap<TokenId, Vec<usize>>,
    pub(crate) whole_words: i64,
}

impl InitialCounts {
    pub(crate) fn from_chunks(chunks: &[Chunk], counts: &[i64]) -> Self {
        assert_eq!(chunks.len(), counts.len());
        let mut result = Self::default();

        for (chunk_index, (tokens, &count)) in chunks.iter().zip(counts).enumerate() {
            let mut previous_pair = None;
            for pair in tokens.windows(2).map(|window| (window[0], window[1])) {
                let overlaps_same_pair = pair.0 == pair.1 && previous_pair == Some(pair);
                if overlaps_same_pair {
                    previous_pair = None;
                } else {
                    *result.pair_counts.entry(pair).or_insert(0) += count;
                    previous_pair = Some(pair);
                }
            }

            for &token in tokens {
                *result.single_counts.entry(token).or_insert(0) += count;
            }
            if tokens.len() == 1 {
                result.whole_words += count;
            }

            let mut unique_tokens = tokens.clone();
            unique_tokens.sort_unstable();
            unique_tokens.dedup();
            for token in unique_tokens {
                result
                    .token_locations
                    .entry(token)
                    .or_default()
                    .push(chunk_index);
            }
        }

        result
    }

    /// Count fixed-size chunk batches concurrently, then reduce them in input
    /// order. This keeps location indices stable and tie-breaking independent
    /// of Rayon scheduling.
    pub(crate) fn from_chunks_parallel(chunks: &[Chunk], counts: &[i64]) -> Self {
        const BATCH_SIZE: usize = 1024;

        assert_eq!(chunks.len(), counts.len());
        let partials: Vec<Self> = chunks
            .par_chunks(BATCH_SIZE)
            .zip(counts.par_chunks(BATCH_SIZE))
            .map(|(chunk_batch, count_batch)| Self::from_chunks(chunk_batch, count_batch))
            .collect();

        let mut result = Self::default();
        for (batch_index, partial) in partials.into_iter().enumerate() {
            for (pair, count) in partial.pair_counts {
                *result.pair_counts.entry(pair).or_insert(0) += count;
            }
            for (token, count) in partial.single_counts {
                *result.single_counts.entry(token).or_insert(0) += count;
            }
            for (token, locations) in partial.token_locations {
                let offset = batch_index * BATCH_SIZE;
                result
                    .token_locations
                    .entry(token)
                    .or_default()
                    .extend(locations.into_iter().map(|location| location + offset));
            }
            result.whole_words += partial.whole_words;
        }

        result
    }
}

/// Interned token bytes and the merge that first created each token.
///
/// `byte_index` maps a 64-bit hash to candidate token IDs. Hash collisions are
/// resolved by comparing the bytes held in `bytes`.
#[derive(Debug, Default)]
pub(crate) struct TokenArena {
    bytes: Vec<Box<[u8]>>,
    parents: Vec<Option<Pair>>,
    byte_index: AHashMap<u64, Vec<TokenId>>,
}

impl TokenArena {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Intern a token and return its stable training ID.
    ///
    /// If distinct merge histories produce identical bytes, retain the first
    /// parent. This preserves the trainer's first-creation behavior and keeps
    /// parent reconstruction deterministic when input and merge order are
    /// deterministic.
    pub(crate) fn intern(&mut self, token: &[u8], parent: Option<Pair>) -> TokenId {
        let hash = Self::hash(token);
        if let Some(candidates) = self.byte_index.get(&hash) {
            for &id in candidates {
                if self.bytes[id as usize].as_ref() == token {
                    return id;
                }
            }
        }

        if let Some((left, right)) = parent {
            debug_assert!((left as usize) < self.bytes.len());
            debug_assert!((right as usize) < self.bytes.len());
        }

        let id = TokenId::try_from(self.bytes.len()).expect("training token ID overflowed u32");
        self.bytes.push(token.into());
        self.parents.push(parent);
        self.byte_index.entry(hash).or_default().push(id);
        id
    }

    pub(crate) fn intern_leaf(&mut self, token: &[u8]) -> TokenId {
        self.intern(token, None)
    }

    pub(crate) fn intern_merge(&mut self, token: &[u8], pair: Pair) -> TokenId {
        self.intern(token, Some(pair))
    }

    pub(crate) fn id_for(&self, token: &[u8]) -> Option<TokenId> {
        let hash = Self::hash(token);
        self.byte_index.get(&hash).and_then(|candidates| {
            candidates
                .iter()
                .copied()
                .find(|&id| self.bytes[id as usize].as_ref() == token)
        })
    }

    pub(crate) fn bytes(&self, id: TokenId) -> &[u8] {
        self.bytes
            .get(id as usize)
            .expect("unknown training token ID")
            .as_ref()
    }

    pub(crate) fn parent(&self, id: TokenId) -> Option<Pair> {
        self.parents
            .get(id as usize)
            .expect("unknown training token ID")
            .as_ref()
            .copied()
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.bytes.len()
    }

    fn hash(token: &[u8]) -> u64 {
        let mut hasher = AHasher::default();
        token.hash(&mut hasher);
        hasher.finish()
    }
}

/// Intern a byte-backed training corpus at the corpus/model I/O boundary.
pub(crate) fn intern_chunks(arena: &mut TokenArena, chunks: Vec<Vec<Vec<u8>>>) -> Vec<Chunk> {
    // Establish IDs in byte order before converting aggregated chunks so pair
    // tie-breaking does not depend on hash-map iteration order.
    let mut unseen_tokens: BTreeSet<&[u8]> = BTreeSet::new();
    for chunk in &chunks {
        for token in chunk {
            unseen_tokens.insert(token);
        }
    }
    for token in unseen_tokens {
        arena.intern_leaf(token);
    }

    chunks
        .into_iter()
        .map(|chunk| {
            chunk
                .into_iter()
                .map(|token| arena.id_for(&token).expect("token was just interned"))
                .collect()
        })
        .collect()
}

/// Replace non-overlapping occurrences of `pair` with its already-interned ID.
pub(crate) fn merge_chunk(tokens: &[TokenId], pair: Pair, merged: TokenId) -> (Chunk, i64) {
    let mut result = Vec::with_capacity(tokens.len());
    let mut index = 0;
    let mut count = 0;

    while index < tokens.len() {
        if index + 1 < tokens.len() && (tokens[index], tokens[index + 1]) == pair {
            result.push(merged);
            count += 1;
            index += 2;
        } else {
            result.push(tokens[index]);
            index += 1;
        }
    }

    (result, count)
}

/// Replace every occurrence of `bad_token` with its replacement IDs.
pub(crate) fn blow_up_chunk(
    tokens: &[TokenId],
    bad_token: TokenId,
    parts: &[TokenId],
) -> (Chunk, i64) {
    let mut result = Vec::with_capacity(tokens.len() + parts.len());
    let mut count = 0;

    for &token in tokens {
        if token == bad_token {
            result.extend_from_slice(parts);
            count += 1;
        } else {
            result.push(token);
        }
    }

    (result, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interning_reuses_ids_without_copying_token_keys() {
        let mut arena = TokenArena::new();
        let a = arena.intern_leaf(b"a");
        let a_again = arena.intern_leaf(b"a");
        let b = arena.intern_leaf(b"b");

        assert_eq!(a, a_again);
        assert_ne!(a, b);
        assert_eq!(arena.len(), 2);
        assert_eq!(arena.id_for(b"a"), Some(a));
        assert_eq!(arena.bytes(b), b"b");
        assert_eq!(arena.parent(a), None);
    }

    #[test]
    fn merge_parent_is_stable_when_bytes_are_reencountered() {
        let mut arena = TokenArena::new();
        let a = arena.intern_leaf(b"a");
        let b = arena.intern_leaf(b"b");
        let ab = arena.intern_merge(b"ab", (a, b));

        let existing = arena.intern_leaf(b"ab");
        assert_eq!(existing, ab);
        assert_eq!(arena.parent(ab), Some((a, b)));
    }

    #[test]
    fn id_chunks_preserve_merge_overlap_and_blow_up_semantics() {
        let mut arena = TokenArena::new();
        let a = arena.intern_leaf(b"a");
        let b = arena.intern_leaf(b"b");
        let ab = arena.intern_merge(b"ab", (a, b));

        let chunks = intern_chunks(&mut arena, vec![vec![b"a".to_vec(), b"b".to_vec()]]);
        assert_eq!(chunks, vec![vec![a, b]]);

        let (merged, merges) = merge_chunk(&[a, a, a], (a, a), ab);
        assert_eq!((merged, merges), (vec![ab, a], 1));

        let (expanded, deletions) = blow_up_chunk(&[ab, b, ab], ab, &[a, b]);
        assert_eq!((expanded, deletions), (vec![a, b, b, a, b], 2));
    }

    #[test]
    fn chunk_interning_assigns_stable_ids_despite_chunk_order() {
        let source = vec![vec![b"bb".to_vec(), b"a".to_vec()], vec![b"c".to_vec()]];
        let mut first_arena = TokenArena::new();
        let first = intern_chunks(&mut first_arena, source.clone());

        let mut reordered = source;
        reordered.reverse();
        let mut second_arena = TokenArena::new();
        let second = intern_chunks(&mut second_arena, reordered);

        assert_eq!(first_arena.id_for(b"a"), second_arena.id_for(b"a"));
        assert_eq!(first_arena.id_for(b"bb"), second_arena.id_for(b"bb"));
        assert_eq!(first_arena.id_for(b"c"), second_arena.id_for(b"c"));
        assert_ne!(first[0], second[0]); // input order remains faithfully represented
    }

    #[test]
    fn initial_counts_uses_ids_and_records_each_chunk_once_per_token() {
        let chunks = vec![vec![1, 1, 1], vec![1, 2], vec![2]];
        let counts = vec![5, 3, 7];
        let result = InitialCounts::from_chunks(&chunks, &counts);

        assert_eq!(result.pair_counts.get(&(1, 1)), Some(&5));
        assert_eq!(result.pair_counts.get(&(1, 2)), Some(&3));
        assert_eq!(result.single_counts.get(&1), Some(&18));
        assert_eq!(result.single_counts.get(&2), Some(&10));
        assert_eq!(result.token_locations.get(&1), Some(&vec![0, 1]));
        assert_eq!(result.token_locations.get(&2), Some(&vec![1, 2]));
        assert_eq!(result.whole_words, 7);
    }
}
