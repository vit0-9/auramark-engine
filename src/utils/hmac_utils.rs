use crate::error::{AuraMarkError, Result};
use hmac::{Hmac, Mac};
use sha2::Sha256;

pub type HmacSha256 = Hmac<Sha256>;

pub const HMAC_TAG_LEN: usize = 32;
pub const PAYLOAD_WITH_HMAC_LEN: usize = 16 + HMAC_TAG_LEN;

pub fn generate_data_with_hmac(original_data_part: &[u8], secret_key: &[u8]) -> Result<Vec<u8>> {
    if secret_key.is_empty() {
        return Err(AuraMarkError::Error("Secret key must not be empty".into()));
    }
    if original_data_part.len() != 16 {
        return Err(AuraMarkError::Error(
            "Original data part must be 16 bytes".into(),
        ));
    }

    let mut mac = HmacSha256::new_from_slice(secret_key)
        .map_err(|e| AuraMarkError::Error(format!("HMAC key error: {}", e)))?;
    mac.update(original_data_part);
    let hmac_tag = mac.finalize().into_bytes();

    let mut data = Vec::with_capacity(PAYLOAD_WITH_HMAC_LEN);
    data.extend_from_slice(original_data_part);
    data.extend_from_slice(&hmac_tag);

    if data.len() != PAYLOAD_WITH_HMAC_LEN {
        return Err(AuraMarkError::Error(
            "Unexpected HMAC payload length".into(),
        ));
    }

    Ok(data)
}

pub fn verify_data_with_hmac(full_payload: &[u8], secret_key: &[u8]) -> Result<Option<Vec<u8>>> {
    if secret_key.is_empty() {
        return Err(AuraMarkError::Error("Secret key must not be empty".into()));
    }

    if full_payload.len() != PAYLOAD_WITH_HMAC_LEN {
        return Err(AuraMarkError::Error(format!(
            "Invalid payload length: expected {}, got {}",
            PAYLOAD_WITH_HMAC_LEN,
            full_payload.len()
        )));
    }

    let data = &full_payload[..16];
    let tag = &full_payload[16..];

    let mut mac = HmacSha256::new_from_slice(secret_key)
        .map_err(|e| AuraMarkError::Error(format!("HMAC key error: {}", e)))?;
    mac.update(data);
    let expected_tag = mac.finalize().into_bytes();

    if expected_tag.as_slice() == tag {
        Ok(Some(data.to_vec()))
    } else {
        Ok(None)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use hex_literal::hex;
    use hmac::{Hmac, Mac};
    use sha2::Sha256;

    type HmacSha256 = Hmac<Sha256>;

    fn make_16_bytes(input: &[u8]) -> [u8; 16] {
        let mut arr = [0u8; 16];
        let len = input.len().min(16);
        arr[..len].copy_from_slice(&input[..len]);
        arr
    }

    #[test]
    fn test_generate_and_verify_hmac_success() {
        let data = make_16_bytes(b"test_data_prefix");
        let key = b"my_super_secret_key_for_hmac_testing";

        let payload = generate_data_with_hmac(&data, key).unwrap();
        let verified = verify_data_with_hmac(&payload, key).unwrap().unwrap();

        assert_eq!(payload.len(), PAYLOAD_WITH_HMAC_LEN);
        assert_eq!(verified, data);
    }

    #[test]
    fn test_verify_hmac_tampered_data_fails() {
        let data = make_16_bytes(b"tampered_payload");
        let key = b"tamper_key";

        let mut payload = generate_data_with_hmac(&data, key).unwrap();
        payload[0] ^= 0xAA;

        let result = verify_data_with_hmac(&payload, key).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_verify_hmac_tampered_tag_fails() {
        let data = make_16_bytes(b"tag_test");
        let key = b"some_key";

        let mut payload = generate_data_with_hmac(&data, key).unwrap();
        payload[16] ^= 0x01;

        let result = verify_data_with_hmac(&payload, key).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_verify_hmac_wrong_key_fails() {
        let data = make_16_bytes(b"wrong_key_test");
        let key = b"correct_key";
        let wrong_key = b"incorrect_key";

        let payload = generate_data_with_hmac(&data, key).unwrap();
        let result = verify_data_with_hmac(&payload, wrong_key).unwrap();

        assert!(result.is_none());
    }

    #[test]
    fn test_generate_invalid_data_length_errors() {
        let key = b"short_key";

        let too_short = b"short";
        let too_long = b"this_is_way_too_long_for_16_bytes_data";

        assert!(generate_data_with_hmac(too_short, key).is_err());
        assert!(generate_data_with_hmac(too_long, key).is_err());
    }

    #[test]
    fn test_verify_invalid_payload_length_errors() {
        let key = b"some_key";

        let too_short = vec![0u8; PAYLOAD_WITH_HMAC_LEN - 1];
        let too_long = vec![0u8; PAYLOAD_WITH_HMAC_LEN + 1];

        assert!(verify_data_with_hmac(&too_short, key).is_err());
        assert!(verify_data_with_hmac(&too_long, key).is_err());
    }

    #[test]
    fn test_zero_key() {
        let data = make_16_bytes(b"zero_key");
        let key = &[0u8; 32];

        let payload = generate_data_with_hmac(&data, key).unwrap();
        let verified = verify_data_with_hmac(&payload, key).unwrap().unwrap();

        assert_eq!(verified, data);
    }

    #[test]
    fn test_empty_key_fails() {
        let data = make_16_bytes(b"empty_key");
        let key = b"";

        let result = generate_data_with_hmac(&data, key);
        assert!(result.is_err());
    }

    #[test]
    #[ignore]
    //https://datatracker.ietf.org/doc/html/rfc4231#section-4.2
    fn test_rfc4231_case1_known_answer() {
        let key = hex!("0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b");
        let msg = b"Hi There";
        let expected = hex!("b0344c61d8ce2d4a65345a9ca3a07011b97b0a70f44e13e2f5b41050a4d53df0");

        let mut mac = HmacSha256::new_from_slice(&key).unwrap();
        mac.update(msg);
        let result = mac.finalize().into_bytes();

        assert_eq!(result.as_slice(), expected);
    }

    #[test]
    fn test_known_answer_padded_message() {
        let key = hex!("0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b");
        let data = make_16_bytes(b"Hi There");

        // Compute expected tag on the padded data
        let mut mac = HmacSha256::new_from_slice(&key).unwrap();
        mac.update(&data);
        let expected = mac.finalize().into_bytes();

        let payload = generate_data_with_hmac(&data, &key).unwrap();
        let tag = &payload[16..];

        assert_eq!(tag, expected.as_slice());

        let verified = verify_data_with_hmac(&payload, &key).unwrap().unwrap();
        assert_eq!(verified, data);
    }
}
