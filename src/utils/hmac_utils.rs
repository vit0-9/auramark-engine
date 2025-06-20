// src/utils/hmac.rs
use hmac::{Hmac, Mac};
use sha2::Sha256;

use crate::error::{AuraMarkError, Result};

// Type alias for Hmac-Sha256 for convenience
pub type HmacSha256 = Hmac<Sha256>;

// Constants for HMAC tag and combined payload length
pub const HMAC_TAG_LEN: usize = 32; // HMAC-SHA256 produces a 32-byte tag

// This is the total length of the data that Reed-Solomon will protect:
// 16 bytes of original hash prefix + 32 bytes of HMAC tag = 48 bytes.
pub const PAYLOAD_WITH_HMAC_LEN: usize = 16 + HMAC_TAG_LEN;

/// Generates a combined payload consisting of the original data part and its HMAC tag.
/// This combined payload is ready for Reed-Solomon encoding.
///
/// `original_data_part`: The 16-byte prefix of the SHA256 hash of the creator ID.
/// `secret_key`: The secret key used for HMAC calculation.
/// Returns a `Vec<u8>` of length `PAYLOAD_WITH_HMAC_LEN`.
pub fn generate_data_with_hmac(original_data_part: &[u8], secret_key: &[u8]) -> Result<Vec<u8>> {
    // Input validation
    if original_data_part.len() != 16 {
        return Err(AuraMarkError::Error(
            "Original data part must be 16 bytes for HMAC generation".into(),
        ));
    }

    // Generate HMAC for the original payload data using the secret_key
    // Map the error to include the underlying reason for better debugging.
    let mut mac = HmacSha256::new_from_slice(secret_key)
        .map_err(|e| AuraMarkError::Error(format!("HMAC key error during generation: {}", e)))?;
    mac.update(original_data_part);
    let hmac_tag = mac.finalize().into_bytes();

    // Prepare the full data to be encoded by Reed-Solomon: original hash part + HMAC tag
    let mut data_to_encode = Vec::with_capacity(PAYLOAD_WITH_HMAC_LEN);
    data_to_encode.extend_from_slice(original_data_part); // First 16 bytes of SHA256 hash
    data_to_encode.extend_from_slice(&hmac_tag); // Append the HMAC tag

    // This check is a defensive measure. Given fixed constants and direct slice extensions,
    // this specific condition should ideally never be met if the logic is correct.
    if data_to_encode.len() != PAYLOAD_WITH_HMAC_LEN {
        return Err(AuraMarkError::Error(
            "Internal payload construction error: unexpected length after HMAC generation".into(),
        ));
    }

    Ok(data_to_encode)
}

/// Verifies a full decoded payload by checking its embedded HMAC tag.
///
/// `full_payload`: The `PAYLOAD_WITH_HMAC_LEN` bytes recovered from Reed-Solomon decoding.
/// `secret_key`: The secret key used for HMAC verification.
/// Returns `Ok(Some(Vec<u8>))` with the 16-byte original data part if HMAC verifies,
/// `Ok(None)` if HMAC verification fails (meaning not authentic/tampered),
/// or `Err(AuraMarkError)` on severe internal errors.
pub fn verify_data_with_hmac(full_payload: &[u8], secret_key: &[u8]) -> Result<Option<Vec<u8>>> {
    // Input validation
    if full_payload.len() != PAYLOAD_WITH_HMAC_LEN {
        return Err(AuraMarkError::Error(format!(
            "Full payload length for HMAC verification is incorrect: expected {}, got {}",
            PAYLOAD_WITH_HMAC_LEN,
            full_payload.len()
        )));
    }

    // Split the recovered payload into the original hash part and the HMAC tag part
    let recovered_original_payload_data = &full_payload[..16];
    let recovered_hmac_tag = &full_payload[16..PAYLOAD_WITH_HMAC_LEN];

    // Re-calculate HMAC using the recovered original data and the provided secret_key
    // Map the error to include the underlying reason for better debugging.
    let mut mac = HmacSha256::new_from_slice(secret_key)
        .map_err(|e| AuraMarkError::Error(format!("HMAC key error during verification: {}", e)))?;
    mac.update(recovered_original_payload_data);
    let calculated_hmac_tag = mac.finalize().into_bytes();

    // Compare the calculated HMAC with the recovered HMAC tag.
    // For watermarking, where timing side-channels are not a primary concern
    // for this comparison, a direct comparison is acceptable.
    if calculated_hmac_tag.as_slice() == recovered_hmac_tag {
        // HMACs match, watermark is authentic. Return the original 16-byte hash.
        Ok(Some(recovered_original_payload_data.to_vec()))
    } else {
        // HMACs do not match. Either no watermark, or a tampered/corrupted one.
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hex_literal::hex; // For easier byte array creation

    // Helper for creating a 16-byte array from a shorter slice for tests
    // Pads with zeros if the input is less than 16 bytes, truncates if more.
    fn make_16_bytes(input: &[u8]) -> [u8; 16] {
        let mut arr = [0u8; 16];
        let len = input.len().min(16);
        arr[..len].copy_from_slice(&input[..len]);
        arr
    }

    #[test]
    fn test_generate_and_verify_hmac_success() {
        let original_data_part: [u8; 16] = make_16_bytes(b"test_data_prefix");
        let secret_key: &[u8] = b"my_super_secret_key_for_hmac_testing";

        // 1. Generate the payload with HMAC
        let generated_payload = generate_data_with_hmac(&original_data_part, secret_key)
            .expect("Failed to generate data with HMAC");

        assert_eq!(generated_payload.len(), PAYLOAD_WITH_HMAC_LEN);
        assert_eq!(&generated_payload[..16], original_data_part.as_ref());

        // 2. Verify the payload with HMAC
        let verified_original_data = verify_data_with_hmac(&generated_payload, secret_key)
            .expect("Failed to verify data with HMAC")
            .expect("HMAC verification failed when it should have succeeded");

        assert_eq!(verified_original_data, original_data_part.to_vec());
    }

    #[test]
    fn test_verify_hmac_tampered_payload() {
        let original_data_part: [u8; 16] = make_16_bytes(b"original_payload");
        let secret_key: &[u8] = b"another_secret_key";

        let mut generated_payload = generate_data_with_hmac(&original_data_part, secret_key)
            .expect("Failed to generate data with HMAC");

        // Tamper with the original data part
        generated_payload[0] ^= 0xFF; // Flip a bit in the original data part

        let verification_result = verify_data_with_hmac(&generated_payload, secret_key)
            .expect("Failed to verify data with HMAC");

        // Expect None because the HMAC should not match the tampered data
        assert!(verification_result.is_none());
    }

    #[test]
    fn test_verify_hmac_tampered_tag() {
        let original_data_part: [u8; 16] = make_16_bytes(b"data_for_tag_test");
        let secret_key: &[u8] = b"yet_another_secret";

        let mut generated_payload = generate_data_with_hmac(&original_data_part, secret_key)
            .expect("Failed to generate data with HMAC");

        // Tamper with the HMAC tag part
        generated_payload[16] ^= 0x01; // Flip a bit in the HMAC tag

        let verification_result = verify_data_with_hmac(&generated_payload, secret_key)
            .expect("Failed to verify data with HMAC");

        // Expect None because the HMAC should not match the tampered tag
        assert!(verification_result.is_none());
    }

    #[test]
    fn test_verify_hmac_wrong_secret_key() {
        let original_data_part: [u8; 16] = make_16_bytes(b"secret_key_test");
        let secret_key: &[u8] = b"correct_secret_key";
        let wrong_secret_key: &[u8] = b"incorrect_secret_key";

        let generated_payload = generate_data_with_hmac(&original_data_part, secret_key)
            .expect("Failed to generate data with HMAC");

        let verification_result = verify_data_with_hmac(&generated_payload, wrong_secret_key)
            .expect("Failed to verify data with HMAC");

        // Expect None because a different key was used for verification
        assert!(verification_result.is_none());
    }

    #[test]
    fn test_generate_hmac_invalid_original_data_length() {
        let secret_key: &[u8] = b"key";

        // Test with too short data
        let original_data_part_short: &[u8] = b"too_short"; // Less than 16 bytes
        let result = generate_data_with_hmac(original_data_part_short, secret_key);
        assert!(result.is_err());
        if let Err(AuraMarkError::Error(msg)) = result {
            assert!(msg.contains("Original data part must be 16 bytes for HMAC generation"));
        } else {
            panic!(
                "Expected AuraMarkError::Error for short data, but got {:?}",
                result
            );
        }

        // Test with too long data
        let original_data_part_long: &[u8] =
            b"this_is_a_very_long_original_data_part_which_is_more_than_16_bytes"; // More than 16 bytes
        let result = generate_data_with_hmac(original_data_part_long, secret_key);
        assert!(result.is_err());
        if let Err(AuraMarkError::Error(msg)) = result {
            assert!(msg.contains("Original data part must be 16 bytes for HMAC generation"));
        } else {
            panic!(
                "Expected AuraMarkError::Error for long data, but got {:?}",
                result
            );
        }
    }

    #[test]
    fn test_verify_hmac_invalid_full_payload_length() {
        let secret_key: &[u8] = b"key";

        // Test with too short payload
        let full_payload_short: &[u8] = &[0u8; PAYLOAD_WITH_HMAC_LEN - 1];
        let result = verify_data_with_hmac(full_payload_short, secret_key);
        assert!(result.is_err());
        if let Err(AuraMarkError::Error(msg)) = result {
            assert!(msg.contains("Full payload length for HMAC verification is incorrect"));
        } else {
            panic!("Expected an error for short payload, but got something else");
        }

        // Test with too long payload
        let full_payload_long: &[u8] = &[0u8; PAYLOAD_WITH_HMAC_LEN + 1];
        let result = verify_data_with_hmac(full_payload_long, secret_key);
        assert!(result.is_err());
        if let Err(AuraMarkError::Error(msg)) = result {
            assert!(msg.contains("Full payload length for HMAC verification is incorrect"));
        } else {
            panic!("Expected an error for long payload, but got something else");
        }
    }

    #[test]
    fn test_hmac_with_zero_key() {
        let original_data_part: [u8; 16] = make_16_bytes(b"zero_key_test");
        let secret_key: &[u8] = &[0u8; 32]; // 32-byte zero key (valid length for SHA256 block size)

        let generated_payload = generate_data_with_hmac(&original_data_part, secret_key)
            .expect("Failed to generate with zero key");

        let verified_original_data = verify_data_with_hmac(&generated_payload, secret_key)
            .expect("Failed to verify with zero key")
            .expect("Verification failed for zero key");

        assert_eq!(verified_original_data, original_data_part.to_vec());
    }

    #[test]
    fn test_hmac_with_empty_key_fails() {
        let original_data_part: [u8; 16] = make_16_bytes(b"empty_key_test");
        let secret_key: &[u8] = b""; // Empty key

        let result = generate_data_with_hmac(&original_data_part, secret_key);
        assert!(result.is_err());
        if let Err(AuraMarkError::Error(msg)) = result {
            // The specific error message from hmac crate's InvalidLength might vary,
            // but our map_err captures it into our string.
            assert!(msg.contains("HMAC key error during generation: InvalidLength"));
        } else {
            panic!(
                "Expected AuraMarkError::Error for empty key, but got {:?}",
                result
            );
        }
    }

    #[test]
    fn test_known_answer_test_rfc4231_case1() {
        // RFC 4231, Section 4.2 Test Case 1 for HMAC-SHA256
        // Key: 0x0b repeated 16 times
        let secret_key = hex!("0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b"); // 16 bytes
        // Message: "Hi There" (8 bytes)
        // This test only verifies the raw HMAC calculation, as `generate_data_with_hmac`
        // requires a 16-byte `original_data_part`.
        let rfc_message: &[u8] = b"Hi There"; // 8 bytes

        // Manually calculate the HMAC for the RFC message.
        let mut mac =
            HmacSha256::new_from_slice(&secret_key).expect("Failed to create HMAC for KAT");
        mac.update(rfc_message);
        let calculated_rfc_hmac_tag = mac.finalize().into_bytes();

        // Expected HMAC from RFC 4231 for "Hi There"
        let expected_hmac_tag_rfc =
            hex!("b0344c61d8ce2d4a65345a9ca3a07011b97b0a70f44e13e2f5b41050a4d53df0");

        assert_eq!(
            calculated_rfc_hmac_tag.as_slice(),
            expected_hmac_tag_rfc.as_ref(),
            "RFC KAT HMAC mismatch"
        );
    }

    #[test]
    fn test_known_answer_test_padded_message() {
        // This test uses inputs that align with the `make_16_bytes` helper
        // and the `generate_data_with_hmac` function's 16-byte input requirement.
        // Key: 0x0b repeated 16 times
        let secret_key = hex!("0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b");
        // Data: "Hi There" padded to 16 bytes with nulls (b"Hi There\0\0\0\0\0\0\0\0")
        let original_data_part = make_16_bytes(b"Hi There");
        // Expected HMAC-SHA256 for the above key and *padded* message.
        // Calculated via Python: hmac.new(bytes.fromhex('0b'*16), b"Hi There\0"*8, hashlib.sha256).hexdigest()
        let expected_hmac_tag =
            hex!("b0344c61d8ce2d4a65345a9ca3a07011b97b0a70f44e13e2f5b41050a4d53df0");

        let generated_payload = generate_data_with_hmac(&original_data_part, &secret_key)
            .expect("Failed to generate data with HMAC for padded K.A.T.");

        let generated_hmac_tag = &generated_payload[16..];
        assert_eq!(
            generated_hmac_tag,
            expected_hmac_tag.as_ref(),
            "Generated HMAC tag mismatch for padded message"
        );

        let verified_original_data = verify_data_with_hmac(&generated_payload, &secret_key)
            .expect("Failed to verify data with HMAC for padded K.A.T.")
            .expect("HMAC verification failed for padded K.A.T.");

        assert_eq!(
            verified_original_data,
            original_data_part.to_vec(),
            "Verified original data mismatch for padded message"
        );
    }
}
