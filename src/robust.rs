//! Module for the robust (attribution) watermark.
//! This version uses the Discrete Cosine Transform (DCT) for robustness.

use image::{DynamicImage, GenericImageView};
use rustdct::DctPlanner;
use std::sync::Arc;

use crate::dct;
use crate::error::{AuraMarkError, Result};
use crate::utils;

// --- Constants ---
const BLOCK_SIZE: usize = 8;
const ENCODED_PAYLOAD_SIZE: usize = utils::watermark::ENCODED_PAYLOAD_SIZE;
const TOTAL_BITS: usize = ENCODED_PAYLOAD_SIZE * 8;

/// Public entry point to embed the robust watermark using DCT.
pub fn embed(image: &mut DynamicImage, message: &[u8], secret_key: &[u8]) -> Result<()> {
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
    let mut luma_image = image.to_luma32f();

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

    // 5. Convert the modified Luma image back to the original color format.
    *image = DynamicImage::from(luma_image);

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
    use image::buffer::ConvertBuffer;
    use image::{ImageBuffer, Luma, Rgb};

    // Helper function to create a dummy grayscale image for tests
    fn create_dummy_luma_image(width: u32, height: u32) -> DynamicImage {
        let mut img = ImageBuffer::new(width, height);
        for y in 0..height {
            for x in 0..width {
                img.put_pixel(x, y, Luma([(x % 255) as f32 / 255.0]));
            }
        }
        DynamicImage::ImageLuma8(img.convert())
    }

    // Helper function to create a dummy RGB image
    fn create_dummy_rgb_image(width: u32, height: u32) -> DynamicImage {
        let mut img = ImageBuffer::new(width, height);
        for y in 0..height {
            for x in 0..width {
                img.put_pixel(x, y, Rgb([x as u8, y as u8, (x + y) as u8]));
            }
        }
        DynamicImage::ImageRgb8(img)
    }

    #[test]
    fn test_embed_image_too_small() {
        let min_blocks_needed = TOTAL_BITS;
        let min_side_blocks = (min_blocks_needed as f32).sqrt().ceil() as u32;
        let min_dim_px = min_side_blocks * BLOCK_SIZE as u32;

        let too_small_width = min_dim_px - 1;
        let too_small_height = min_dim_px - 1;
        let mut image = create_dummy_luma_image(too_small_width, too_small_height);

        let creator_id = to_fixed_16(b"test_user_small_image");
        let secret_key = to_fixed_16(b"super_secret_key_small");

        let result = embed(&mut image, &creator_id, &secret_key);

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), AuraMarkError::ImageTooSmall);

        let mut image_barely_not_enough = create_dummy_luma_image(
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
        let mut image = create_dummy_luma_image(width, height);

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
    fn test_embed_with_rgb_image() {
        let min_side_blocks = (TOTAL_BITS as f32).sqrt().ceil() as u32;
        let min_dim_px = min_side_blocks * BLOCK_SIZE as u32;

        let width = min_dim_px;
        let height = min_dim_px;
        let mut image = create_dummy_rgb_image(width, height);

        let creator_id = to_fixed_16(b"rgb_test_user");
        let secret_key = to_fixed_16(b"rgb_secret_key");

        let result = embed(&mut image, &creator_id, &secret_key);
        assert!(
            result.is_ok(),
            "Embedding failed on RGB image: {:?}",
            result.unwrap_err()
        );
    }

    #[test]
    fn test_extract_from_non_watermarked_image() {
        let min_side_blocks = (TOTAL_BITS as f32).sqrt().ceil() as u32;
        let min_dim_px = min_side_blocks * BLOCK_SIZE as u32;

        let width = min_dim_px;
        let height = min_dim_px;
        let image = create_dummy_luma_image(width, height);

        let secret_key = to_fixed_16(b"test_extraction_key");

        let result = extract(&image, &secret_key);
        assert!(result.is_ok());
        // Should return None since image was not watermarked
        assert_eq!(result.unwrap(), None);
    }

    #[test]
    fn test_extract_with_wrong_key() {
        let min_side_blocks = (TOTAL_BITS as f32).sqrt().ceil() as u32;
        let min_dim_px = min_side_blocks * BLOCK_SIZE as u32;

        let width = min_dim_px;
        let height = min_dim_px;
        let mut image = create_dummy_rgb_image(width, height);

        let creator_id = to_fixed_16(b"wrong_key_test_user");
        let embed_key = to_fixed_16(b"correct_embedding_key");
        let extract_key = to_fixed_16(b"wrong_extraction_key");

        // Embed with one key
        let embed_result = embed(&mut image, &creator_id, &embed_key);

        assert!(embed_result.is_ok());
        // Try to extract with different key
        let extract_result = extract(&image, &extract_key);
        // Should return Error since wrong key was used
        assert!(extract_result.is_ok());
        assert_eq!(
            extract_result.unwrap(),
            None,
            "Extraction should fail with wrong key"
        );
    }
}
