// Copyright 2026-present Kensho Technologies, LLC.
use ahash::AHashMap;

/// Byte encoding utilities matching Python's frombytes/tobytes (util.py:32-54).
///
/// Maps all 256 byte values to Unicode characters using the HuggingFace/GPT-2
/// byte-level BPE encoding scheme:
/// - 0x21-0x7E (printable ASCII): direct mapping (95 chars)
/// - 0xA1-0xAC (Latin-1 supplement): direct mapping (12 chars)
/// - 0xAE-0xFF (Latin-1 supplement): direct mapping (82 chars)
/// - Remaining 67 bytes: mapped to chr(256 + n) for n=0..66
pub struct ByteEncoder {
    byte_to_char: [char; 256],
    inv_byte_map: AHashMap<char, u8>,
}

impl ByteEncoder {
    pub fn new() -> Self {
        let mut bs: Vec<u8> = Vec::new();

        // Same ranges as Python: printable ASCII, Latin-1 supplement
        bs.extend(b'!'..=b'~');       // 0x21-0x7E (94 chars)
        bs.extend(0xA1u8..=0xAC);     // 0xA1-0xAC (12 chars)
        bs.extend(0xAEu8..=0xFF);     // 0xAE-0xFF (82 chars)

        let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();
        let mut n: u32 = 0;

        // Map remaining bytes to Unicode private use area (256+n)
        for b in 0u8..=255 {
            if !bs.contains(&b) {
                bs.push(b);
                cs.push(256 + n);
                n += 1;
            }
        }

        let mut byte_to_char = ['\0'; 256];
        let mut inv_byte_map = AHashMap::with_capacity(256);

        for (&b, &c) in bs.iter().zip(cs.iter()) {
            let ch = char::from_u32(c).unwrap();
            byte_to_char[b as usize] = ch;
            inv_byte_map.insert(ch, b);
        }

        Self {
            byte_to_char,
            inv_byte_map,
        }
    }

    /// Convert bytes to string representation (Python: frombytes)
    pub fn from_bytes(&self, bytes: &[u8]) -> String {
        bytes.iter().map(|&b| self.byte_to_char[b as usize]).collect()
    }

    /// Convert string representation back to bytes (Python: tobytes)
    pub fn to_bytes(&self, s: &str) -> Vec<u8> {
        s.chars().map(|c| self.inv_byte_map[&c]).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_trip_all_bytes() {
        let encoder = ByteEncoder::new();
        for b in 0u8..=255 {
            let s = encoder.from_bytes(&[b]);
            let result = encoder.to_bytes(&s);
            assert_eq!(result, vec![b], "Round-trip failed for byte {}", b);
        }
    }

    #[test]
    fn test_round_trip_multi_byte() {
        let encoder = ByteEncoder::new();
        let bytes: Vec<u8> = (0u8..=255).collect();
        let s = encoder.from_bytes(&bytes);
        let result = encoder.to_bytes(&s);
        assert_eq!(result, bytes);
    }

    #[test]
    fn test_printable_ascii_direct() {
        let encoder = ByteEncoder::new();
        // Printable ASCII should map to themselves
        for b in b'!'..=b'~' {
            let ch = encoder.byte_to_char[b as usize];
            assert_eq!(ch, b as char, "Byte {} should map to itself", b);
        }
    }
}
