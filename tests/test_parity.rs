// Copyright 2026-present Kensho Technologies, LLC.
use boundlessbpe::Tokenizer;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_creation() {
        let tokenizer = Tokenizer::new();
        assert!(tokenizer.vocab.is_none());
        assert!(tokenizer.words.is_none());
    }

    #[test]
    fn test_merge() {
        let tokens = vec![
            b"a".to_vec(),
            b"b".to_vec(),
            b"c".to_vec(),
            b"a".to_vec(),
            b"b".to_vec(),
        ];
        let (result, count) = boundlessbpe::tokenizer::Tokenizer::merge(&tokens, (b"a", b"b"));
        assert_eq!(count, 2);
        assert_eq!(result, vec![b"ab".to_vec(), b"c".to_vec(), b"ab".to_vec()]);
    }

    #[test]
    fn test_blow_up() {
        let tokens = vec![b"ab".to_vec(), b"c".to_vec(), b"ab".to_vec()];
        let parts = vec![b"a".to_vec(), b"b".to_vec()];
        let (result, count) = boundlessbpe::tokenizer::Tokenizer::blow_up(&tokens, b"ab", &parts);
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

    #[test]
    fn test_byte_encoding_round_trip() {
        let encoder = boundlessbpe::byte_encoding::ByteEncoder::new();
        for b in 0u8..=255 {
            let s = encoder.from_bytes(&[b]);
            let result = encoder.to_bytes(&s);
            assert_eq!(result, vec![b]);
        }
    }

    // Full integration tests require model files
    // Use test_rust_comparison.py for parity testing
}
