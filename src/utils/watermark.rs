// src/utils/watermark.rs
use rand::RngCore;
use rand::rngs::OsRng;

use reed_solomon_rs::fec::fec::{FEC, Share};

use crate::error::{AuraMarkError, Result};
use crate::utils::hmac_utils;

// Constants for Reed-Solomon configuration
// PAYLOAD_SIZE now refers to the total data size including HMAC
pub const PAYLOAD_SIZE: usize = hmac_utils::PAYLOAD_WITH_HMAC_LEN; // This is 'k' for Reed-Solomon (48 bytes)

// Total number of shares for Reed-Solomon (n).
// (160 - 48) / 2 = 56 errors (exactly 35% error correction of 160 shares).
// This means it can correct up to 56 byte errors.
pub const ENCODED_PAYLOAD_SIZE: usize = 160;

// Total bits expected to be embedded/extracted by the robust module.
pub const TOTAL_BITS_WATERMARK: usize = ENCODED_PAYLOAD_SIZE * 8; // 160 * 8 = 1280 bits

/// Prepares generic message data into a robust byte payload with RS-ECC and an HMAC for authentication.
/// The output is a `Vec<u8>` where each byte is a Reed-Solomon share.
///
/// `message_data`: The data (e.g., a hash, an ID) to be watermarked.
///                 This must be exactly 16 bytes long to fit the defined payload structure.
/// `secret_key`: The secret key used for HMAC generation.
pub fn prepare_watermark_payload(message_data: &[u8], secret_key: &[u8]) -> Result<Vec<u8>> {
    // Validate input message_data length
    if message_data.len() != 16 {
        return Err(AuraMarkError::Error(format!(
            "Input 'message_data' must be exactly 16 bytes long, got {}",
            message_data.len()
        )));
    }

    // Generate the full payload including HMAC using the hmac module
    let data_to_encode = hmac_utils::generate_data_with_hmac(message_data, secret_key)?;

    // Initialize the FEC encoder
    let fec = FEC::new(PAYLOAD_SIZE, ENCODED_PAYLOAD_SIZE)
        .map_err(|e| AuraMarkError::Error(format!("Failed to create FEC encoder: {}", e)))?;

    // Prepare a vector to hold the encoded shares.
    // The `reed_solomon_rs` library expects a callback or an iterator for output.
    // We'll collect the single-byte shares into our `encoded_payload_bytes`.
    let mut encoded_payload_bytes = Vec::with_capacity(ENCODED_PAYLOAD_SIZE);

    // Closure to collect the encoded shares. Each share should contain exactly one byte.
    let output_collector = |s: Share| {
        if s.number < ENCODED_PAYLOAD_SIZE {
            if let Some(&byte) = s.data.first() {
                // Ensure we don't go out of bounds if shares were pre-allocated
                // and push new shares if collecting dynamically.
                // Since we're pushing, capacity is sufficient.
                encoded_payload_bytes.push(byte);
            } else {
                // This case indicates an unexpected empty share, which should ideally not happen
                // with the reed_solomon_rs library if setup correctly for 1-byte shares.
                // For robustness, consider how to handle this if it could occur.
                // For now, it's a silent skip which might lead to incorrect length.
                // A panic! or a Result return here would be better if this is a critical error.
                // Given the current design, `fec.encode` returning Result for entire process is enough.
            }
        }
    };

    fec.encode(&data_to_encode, output_collector)
        .map_err(|e| AuraMarkError::Error(format!("Failed to encode Reed-Solomon data: {}", e)))?;

    // Verify the final length of the encoded payload.
    if encoded_payload_bytes.len() != ENCODED_PAYLOAD_SIZE {
        return Err(AuraMarkError::Error(format!(
            "Internal encoding error: Expected {} encoded bytes, but got {}",
            ENCODED_PAYLOAD_SIZE,
            encoded_payload_bytes.len()
        )));
    }

    Ok(encoded_payload_bytes)
}

/// Decodes the watermark payload, corrects errors, and authenticates using HMAC.
/// Returns Ok(Some(Vec<u8>)) with the 16-byte original message data if authenticated,
/// Ok(None) if not authenticated (HMAC mismatch), Err(AuraMarkError) on severe decode failure.
///
/// `encoded`: The received encoded bytes, which may contain corruptions.
/// `secret_key`: The secret key used for HMAC verification.
pub fn decode_watermark_payload(encoded: &[u8], secret_key: &[u8]) -> Result<Option<Vec<u8>>> {
    if encoded.len() != ENCODED_PAYLOAD_SIZE {
        return Err(AuraMarkError::Error(format!(
            "Invalid encoded payload length: expected {}, got {}",
            ENCODED_PAYLOAD_SIZE,
            encoded.len()
        )));
    }

    // Convert the received bytes into `Share` objects for the decoder.
    let shares_to_decode: Vec<Share> = encoded
        .iter()
        .enumerate()
        .map(|(i, &byte)| Share {
            number: i,
            data: vec![byte], // Each share holds one byte of data
        })
        .collect();

    let fec = FEC::new(PAYLOAD_SIZE, ENCODED_PAYLOAD_SIZE)
        .map_err(|e| AuraMarkError::Error(format!("Failed to create FEC decoder: {}", e)))?;

    // The `decode` method expects an iterable of shares (could be `Vec<Share>` or `&[Share]`).
    // It returns the recovered data (the original `PAYLOAD_SIZE` bytes).
    let recovered_full_payload = fec
        .decode(Vec::new(), shares_to_decode)
        .map_err(|e| AuraMarkError::Error(format!("Failed to decode Reed-Solomon data: {}", e)))?;

    // Ensure the recovered payload has the expected size before HMAC verification.
    if recovered_full_payload.len() != PAYLOAD_SIZE {
        return Err(AuraMarkError::Error(format!(
            "Internal decoding error: Expected {} recovered bytes, but got {}",
            PAYLOAD_SIZE,
            recovered_full_payload.len()
        )));
    }

    // Now, pass the full recovered payload (including HMAC) to the hmac module for verification.
    hmac_utils::verify_data_with_hmac(&recovered_full_payload, secret_key)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{self, Rng};
    use sha2::{Digest, Sha256}; // Ensure Sha256 and Digest are imported for hashing // For generating random data for corruption

    // Helper to get a consistent 16-byte message for testing
    fn get_test_message_data(input_str: &str) -> Vec<u8> {
        let mut hasher = Sha256::new();
        hasher.update(input_str.as_bytes());
        hasher.finalize()[..16].to_vec()
    }

    // Helper to generate a sufficiently long random secret key
    fn get_test_secret_key() -> Vec<u8> {
        let mut key = vec![0u8; 32];
        rand::rngs::OsRng.fill_bytes(&mut key);
        key
    }

    #[test]
    fn test_prepare_with_invalid_message_length() {
        let secret_key = get_test_secret_key();
        let too_short_msg = b"short";
        let too_long_msg = b"this_is_a_very_long_message_that_exceeds_16_bytes_by_far";

        // Test too short message
        let result = prepare_watermark_payload(too_short_msg, &secret_key);
        assert!(result.is_err());
        if let Err(AuraMarkError::Error(msg)) = result {
            assert!(msg.contains("Input 'message_data' must be exactly 16 bytes long"));
        } else {
            panic!("Unexpected error type for too short message: {:?}", result);
        }

        // Test too long message
        let result = prepare_watermark_payload(too_long_msg, &secret_key);
        assert!(result.is_err());
        if let Err(AuraMarkError::Error(msg)) = result {
            assert!(msg.contains("Input 'message_data' must be exactly 16 bytes long"));
        } else {
            panic!("Unexpected error type for too long message: {:?}", result);
        }
    }

    #[test]
    fn test_encode_decode_round_trip() {
        let message_data = get_test_message_data("aura.mark/test@example.com");
        let secret_key = get_test_secret_key();

        let encoded =
            prepare_watermark_payload(&message_data, &secret_key).expect("Encoding failed");
        assert_eq!(encoded.len(), ENCODED_PAYLOAD_SIZE);

        let decoded_message_option =
            decode_watermark_payload(&encoded, &secret_key).expect("Decoding failed");
        assert!(
            decoded_message_option.is_some(),
            "HMAC verification failed on round trip"
        );

        let decoded_message = decoded_message_option.unwrap();
        assert_eq!(decoded_message.len(), 16);
        assert_eq!(decoded_message, message_data);
    }

    #[test]
    fn test_decode_with_corruption_within_limits() {
        let message_data = get_test_message_data("corruption@test.com");
        let secret_key = get_test_secret_key();

        let mut encoded =
            prepare_watermark_payload(&message_data, &secret_key).expect("Encoding failed");

        // Calculate maximum correctable errors
        let max_correctable_errors = (ENCODED_PAYLOAD_SIZE - PAYLOAD_SIZE) / 2;
        // Corrupt exactly the maximum number of correctable errors
        let num_corruptions = max_correctable_errors;

        println!(
            "Attempting to corrupt {} shares out of {} total shares (max correctable: {})",
            num_corruptions, ENCODED_PAYLOAD_SIZE, max_correctable_errors
        );

        // Introduce corruption: flip bits in `num_corruptions` distinct bytes
        let mut rng = rand::rng();
        let mut corrupted_indices: Vec<usize> = Vec::new();
        while corrupted_indices.len() < num_corruptions {
            let idx = rng.random_range(0..encoded.len());
            if !corrupted_indices.contains(&idx) {
                encoded[idx] ^= 0xFF; // Flip all bits in the byte
                corrupted_indices.push(idx);
            }
        }

        let decoded_message_option = decode_watermark_payload(&encoded, &secret_key)
            .expect("Decoding with corruption failed");
        assert!(
            decoded_message_option.is_some(),
            "HMAC verification failed after corruption correction"
        );

        let decoded_message = decoded_message_option.unwrap();
        assert_eq!(decoded_message, message_data);
    }

    #[test]
    fn test_decode_fails_on_too_much_corruption() {
        let message_data = get_test_message_data("too_much_corruption@test.com");
        let secret_key = get_test_secret_key();

        let mut encoded =
            prepare_watermark_payload(&message_data, &secret_key).expect("Encoding failed");

        let max_correctable_errors = (ENCODED_PAYLOAD_SIZE - PAYLOAD_SIZE) / 2;
        let num_corruptions_to_fail = max_correctable_errors + 1; // One more than correctable

        println!(
            "Attempting to corrupt {} shares (expected to fail, max correctable: {})",
            num_corruptions_to_fail, max_correctable_errors
        );

        // Introduce corruption
        let mut rng = rand::rng();
        let mut corrupted_indices: Vec<usize> = Vec::new();
        while corrupted_indices.len() < num_corruptions_to_fail {
            let idx = rng.random_range(0..encoded.len());
            if !corrupted_indices.contains(&idx) {
                if idx < encoded.len() {
                    // Ensure index is within bounds
                    encoded[idx] ^= 0xFF; // Flip all bits
                    corrupted_indices.push(idx);
                }
            }
        }

        let result = decode_watermark_payload(&encoded, &secret_key);
        assert!(result.is_err());
        if let Err(AuraMarkError::Error(msg)) = result {
            assert!(msg.contains("Failed to decode Reed-Solomon data"));
        } else {
            panic!(
                "Expected Reed-Solomon decode failure, but got: {:?}",
                result
            );
        }
    }

    #[test]
    fn test_decode_fails_on_invalid_length() {
        let secret_key = get_test_secret_key();

        // Test too short encoded payload
        let too_short = vec![0u8; 10]; // Much shorter than ENCODED_PAYLOAD_SIZE
        let result = decode_watermark_payload(&too_short, &secret_key);
        assert!(result.is_err());
        if let Err(AuraMarkError::Error(msg)) = result {
            assert!(msg.contains(&format!(
                "Invalid encoded payload length: expected {}, got {}",
                ENCODED_PAYLOAD_SIZE,
                too_short.len()
            )));
        } else {
            panic!(
                "Expected length error for short payload, but got: {:?}",
                result
            );
        }

        // Test too long encoded payload
        let too_long = vec![0u8; ENCODED_PAYLOAD_SIZE + 5];
        let result = decode_watermark_payload(&too_long, &secret_key);
        assert!(result.is_err());
        if let Err(AuraMarkError::Error(msg)) = result {
            assert!(msg.contains(&format!(
                "Invalid encoded payload length: expected {}, got {}",
                ENCODED_PAYLOAD_SIZE,
                too_long.len()
            )));
        } else {
            panic!(
                "Expected length error for long payload, but got: {:?}",
                result
            );
        }
    }

    #[test]
    fn test_decode_hmac_mismatch() {
        let message_data = get_test_message_data("hmac_mismatch_test");
        let secret_key_embed = get_test_secret_key();
        let secret_key_extract = get_test_secret_key(); // Different key

        // Test 1: Mismatched secret key during extraction
        let encoded =
            prepare_watermark_payload(&message_data, &secret_key_embed).expect("Encoding failed");

        let decoded_message_option = decode_watermark_payload(&encoded, &secret_key_extract)
            .expect("Decoding failed unexpectedly with wrong key");
        assert!(
            decoded_message_option.is_none(),
            "HMAC verification succeeded with wrong key, expected failure"
        );

        // Test 2: Tamper with a byte in the encoded shares (which could be original data or HMAC or parity)
        // This simulates a scenario where Reed-Solomon *can* correct the error,
        // but the underlying payload has been tampered with, causing HMAC to fail.
        let mut tampered_encoded =
            prepare_watermark_payload(&message_data, &secret_key_embed).expect("Encoding failed");

        // Corrupt a single byte that is within Reed-Solomon's correction capability
        let corruption_index = 0; // Corrupt the very first share
        if tampered_encoded.len() > corruption_index {
            tampered_encoded[corruption_index] ^= 0x01; // Flip a bit
        } else {
            panic!("Encoded payload too short for intended corruption index");
        }

        let tampered_decode_result = decode_watermark_payload(&tampered_encoded, &secret_key_embed)
            .expect("Decoding failed unexpectedly on tampered data");
        assert!(
            tampered_decode_result.is_none(),
            "HMAC verification succeeded on tampered (but correctable) data, expected failure"
        );
    }
}
