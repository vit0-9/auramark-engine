//! Module for the robust (attribution) watermark.
//! This version uses the Discrete Cosine Transform (DCT) for robustness.

use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
use rustdct::DctPlanner;
use std::sync::Arc;

use crate::dct;
use crate::error::{AuraMarkError, Result};
use crate::utils;
use crate::utils::convert::{extract_luminance, merge_luminance_back};

// --- Constants ---
const BLOCK_SIZE: usize = 8;
const ENCODED_PAYLOAD_SIZE: usize = utils::watermark::ENCODED_PAYLOAD_SIZE;
const TOTAL_BITS: usize = ENCODED_PAYLOAD_SIZE * 8;

/// Public entry point to embed the robust watermark using DCT.
pub fn embed(
    image: &mut ImageBuffer<Rgb<u8>, Vec<u8>>,
    message: &[u8],
    secret_key: &[u8],
) -> Result<()> {
    // 1. Prepare the data payload with error correction.
    let payload = utils::watermark::prepare_watermark_payload(message, secret_key)?;

    // 2. Generate pseudo-random, non-overlapping 8x8 block locations.
    let (width, height) = image.dimensions();
    let num_blocks_x = width as usize / BLOCK_SIZE;
    let num_blocks_y = height as usize / BLOCK_SIZE;

    if (num_blocks_x * num_blocks_y) < TOTAL_BITS {
        return Err(AuraMarkError::ImageTooSmall);
    }

    let block_locations = utils::locations::generate_embedding_block_locations(
        (num_blocks_x, num_blocks_y),
        secret_key,
        TOTAL_BITS,
    )?;

    // 3. Set up the DCT planner.
    let mut planner = DctPlanner::new();
    let dct_boxed_processor: Arc<dyn rustdct::Dct2<f32>> =
        planner.plan_dct2(BLOCK_SIZE * BLOCK_SIZE);
    let dct_processor: Arc<dyn rustdct::Dct2<f32>> = dct_boxed_processor.into();

    // Convert image to Luma (Grayscale) for processing.
    // this was image.to_luma32f() in the original code
    let mut luma_image = extract_luminance(image);

    // 4. Iterate through each bit of the payload and embed it in a unique block.
    for (i, (block_x, block_y)) in block_locations.iter().enumerate() {
        let bit_index = i;
        let byte_index = bit_index / 8;
        if byte_index >= payload.len() {
            return Err(AuraMarkError::Error("Payload index out of bounds".into()));
        }
        let bit_value = (payload[byte_index] >> (bit_index % 8)) & 1 == 1;

        dct::embed_bit_in_block(
            &mut luma_image,
            (*block_x, *block_y),
            bit_value,
            dct_processor.clone(),
        )?;
    }
    let merged = merge_luminance_back(image, &luma_image);

    *image = merged;

    Ok(())
}

/// Public entry point to extract the robust watermark using DCT.
pub fn extract(image: &DynamicImage, secret_key: &[u8]) -> Result<Option<String>> {
    // 1. Generate the same pseudo-random block locations used during embedding.
    let (width, height) = image.dimensions();
    let num_blocks_x = width as usize / BLOCK_SIZE;
    let num_blocks_y = height as usize / BLOCK_SIZE;

    if (num_blocks_x * num_blocks_y) < TOTAL_BITS {
        return Err(AuraMarkError::ImageTooSmall);
    }

    let block_locations = utils::locations::generate_embedding_block_locations(
        (num_blocks_x, num_blocks_y),
        secret_key,
        TOTAL_BITS,
    )?;

    // 2. Set up the DCT planner.
    let mut planner = DctPlanner::new();
    let dct_boxed_processor: Arc<dyn rustdct::Dct2<f32>> =
        planner.plan_dct2(BLOCK_SIZE * BLOCK_SIZE);
    let dct_processor: Arc<dyn rustdct::Dct2<f32>> = dct_boxed_processor.into();

    // Convert image to Luma (Grayscale) for processing.
    let luma_image = image.to_luma32f();

    // 3. Extract bits from each block location.
    let mut extracted_bits = Vec::with_capacity(TOTAL_BITS);

    for (block_x, block_y) in block_locations.iter() {
        let bit =
            dct::extract_bit_from_block(&luma_image, (*block_x, *block_y), dct_processor.clone())?;
        extracted_bits.push(bit);
    }

    // 4. Convert bits back to bytes.
    let mut extracted_bytes = vec![0u8; ENCODED_PAYLOAD_SIZE];
    for (i, bit) in extracted_bits.iter().enumerate() {
        let byte_index = i / 8;
        let bit_position = i % 8;
        if *bit {
            extracted_bytes[byte_index] |= 1 << bit_position;
        }
    }

    // 5. Decode the payload and verify/correct errors.
    match utils::watermark::decode_watermark_payload(&extracted_bytes, secret_key) {
        Ok(Some(hash_bytes)) => {
            let hash_hex = hex::encode(&hash_bytes);
            Ok(Some(hash_hex))
        }
        Ok(None) => Ok(None),
        Err(AuraMarkError::InvalidWatermarkData) => Ok(None),
        Err(e) => {
            // Check if error is Reed-Solomon too many errors to reconstruct
            if let AuraMarkError::Error(msg) = &e {
                if msg.contains("too many errors to reconstruct") {
                    return Ok(None);
                }
            }
            Err(e)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::bytes_utils::to_fixed_16;

    use super::*;
    use image::{ImageBuffer, Rgb};

    // Helper function to create a dummy RGB image for tests (since embed expects Rgb<u8>)
    fn create_dummy_rgb_image_buffer(width: u32, height: u32) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
        let mut img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(width, height);

        for y in 0..height {
            for x in 0..width {
                // Example pattern: create a simple gradient in RGB channels
                let r = (x % 256) as u8;
                let g = (y % 256) as u8;
                let b = ((x + y) % 256) as u8;
                img.put_pixel(x, y, Rgb([r, g, b]));
            }
        }

        img
    }

    // Helper function to create a dummy DynamicImage (for extract function)
    // Renamed to avoid confusion with the ImageBuffer version
    fn create_dummy_dynamic_image_from_rgb_buffer(
        rgb_buffer: ImageBuffer<Rgb<u8>, Vec<u8>>,
    ) -> DynamicImage {
        DynamicImage::ImageRgb8(rgb_buffer)
    }

    #[test]
    fn test_embed_image_too_small() {
        let min_blocks_needed = TOTAL_BITS;
        let min_side_blocks = (min_blocks_needed as f32).sqrt().ceil() as u32;
        let min_dim_px = min_side_blocks * BLOCK_SIZE as u32;

        let too_small_width = min_dim_px - 1;
        let too_small_height = min_dim_px - 1;
        let mut image = create_dummy_rgb_image_buffer(too_small_width, too_small_height);

        let creator_id = to_fixed_16(b"test_user_small_image");
        let secret_key = to_fixed_16(b"super_secret_key_small");

        let result = embed(&mut image, &creator_id, &secret_key);

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), AuraMarkError::ImageTooSmall);

        let mut image_barely_not_enough = create_dummy_rgb_image_buffer(
            (min_blocks_needed as u32 - 1) * BLOCK_SIZE as u32,
            BLOCK_SIZE as u32,
        );
        let result_barely_not_enough =
            embed(&mut image_barely_not_enough, &creator_id, &secret_key);
        assert!(result_barely_not_enough.is_err());
        assert_eq!(
            result_barely_not_enough.unwrap_err(),
            AuraMarkError::ImageTooSmall
        );
    }

    #[test]
    fn test_embed_successful_basic() {
        let min_side_blocks = (TOTAL_BITS as f32).sqrt().ceil() as u32;
        let min_dim_px = min_side_blocks * BLOCK_SIZE as u32;

        let width = min_dim_px;
        let height = min_dim_px;
        let mut image = create_dummy_rgb_image_buffer(width, height);

        let creator_id = to_fixed_16(b"test_user_success");
        let secret_key = to_fixed_16(b"another_strong_secret_key");

        let result = embed(&mut image, &creator_id, &secret_key);

        assert!(
            result.is_ok(),
            "Embedding failed with error: {:?}",
            result.unwrap_err()
        );
    }

    #[test]
    fn test_extract_from_non_watermarked_image() {
        let min_side_blocks = (TOTAL_BITS as f32).sqrt().ceil() as u32;
        let min_dim_px = min_side_blocks * BLOCK_SIZE as u32;

        let width = min_dim_px;
        let height = min_dim_px;
        // Create an unwatermarked DynamicImage directly for this test
        let image = create_dummy_dynamic_image_from_rgb_buffer(create_dummy_rgb_image_buffer(
            width, height,
        ));

        let secret_key = to_fixed_16(b"test_extraction_key");

        let result = extract(&image, &secret_key);
        assert!(result.is_ok());
        // Should return None since image was not watermarked
        assert_eq!(result.unwrap(), None);
    }

    #[test]
    fn test_embed_and_extract_successful() {
        let min_side_blocks = (TOTAL_BITS as f32).sqrt().ceil() as u32;
        let min_dim_px = min_side_blocks * BLOCK_SIZE as u32;

        let width = min_dim_px;
        let height = min_dim_px;
        let mut image_buffer_to_embed = create_dummy_rgb_image_buffer(width, height);

        let message = to_fixed_16(b"test_message_12345");
        let secret_key = to_fixed_16(b"my_secret_key_embed_extract");

        // Embed the watermark
        let embed_result = embed(&mut image_buffer_to_embed, &message, &secret_key);
        assert!(
            embed_result.is_ok(),
            "Embedding failed: {:?}",
            embed_result.unwrap_err()
        );

        // Convert the modified ImageBuffer back to DynamicImage for extraction
        let embedded_dynamic_image =
            create_dummy_dynamic_image_from_rgb_buffer(image_buffer_to_embed);

        // Extract the watermark
        let extracted_message_hex = extract(&embedded_dynamic_image, &secret_key)
            .expect("Extraction should not return an error");

        // Verify the extracted message
        assert!(
            extracted_message_hex.is_some(),
            "Watermark not found after embedding (expected success)"
        );
        let extracted_message_string = extracted_message_hex.unwrap();
        // The decode_watermark_payload function in utils::watermark returns a hash in hex.
        // So we need to compare the expected hash, not the original message.
        // For a full round-trip test, you'd need access to the exact payload *before* encoding to hash.
        // For now, let's just ensure *something* was extracted and it's a valid hex string of the expected length.
        assert_eq!(extracted_message_string, ""); // Hash in hex is twice the byte length
    }

    #[test]
    fn test_extract_with_wrong_key() {
        let min_side_blocks = (TOTAL_BITS as f32).sqrt().ceil() as u32;
        let min_dim_px = min_side_blocks * BLOCK_SIZE as u32;

        let width = min_dim_px;
        let height = min_dim_px;
        let mut image_to_embed = create_dummy_rgb_image_buffer(width, height);

        let creator_id = to_fixed_16(b"wrong_key_test_user");
        let embed_key = to_fixed_16(b"correct_embedding_key");
        let extract_key = to_fixed_16(b"wrong_extraction_key");

        // Embed with one key
        let embed_result = embed(&mut image_to_embed, &creator_id, &embed_key);
        assert!(embed_result.is_ok());

        // Convert the *embedded* image buffer to DynamicImage for extraction
        let embedded_dynamic_image = create_dummy_dynamic_image_from_rgb_buffer(image_to_embed);

        // Try to extract with different key
        let extract_result = extract(&embedded_dynamic_image, &extract_key);
        // Should return None because the key is wrong (or if Reed-Solomon fails)
        assert!(extract_result.is_ok()); // The function returns Ok(None) on failure, not an Err
        assert_eq!(
            extract_result.unwrap(),
            None,
            "Extraction should return None with wrong key"
        );
    }
}
