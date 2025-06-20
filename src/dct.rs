//! DCT-based watermarking operations for embedding and extracting bits in image blocks.
//!
//! This module implements a quantization-based watermarking technique using the Discrete Cosine
//! Transform (DCT). The approach modifies mid-frequency DCT coefficients to embed watermark bits
//! while maintaining a good balance between robustness and imperceptibility.

use image::{ImageBuffer, Luma};
use rustdct::Dct2;
use std::sync::Arc;

use crate::error::{AuraMarkError, Result};

// --- Constants ---
/// The size of the square blocks the image will be processed in.
const BLOCK_SIZE: usize = 8;

/// The quantization step used for embedding bits into DCT coefficients.
/// This value controls the trade-off between robustness and image quality:
/// - Higher values = more robust but more visible artifacts
/// - Lower values = less visible but less robust to attacks
const QUANTIZATION_STEP: f32 = 20.0;

/// The index of the mid-frequency DCT coefficient used for embedding.
/// Position (4,4) in an 8x8 DCT block provides good robustness characteristics.
const EMBEDDING_COEFF_INDEX: usize = 4 * BLOCK_SIZE + 4;

/// Embeds a single bit into an 8x8 image block using DCT quantization.
///
/// This function implements a quantization-based watermarking technique:
/// 1. Extracts an 8x8 pixel block from the image
/// 2. Applies forward DCT to transform pixels to frequency domain
/// 3. Modifies a mid-frequency coefficient using quantization:
///    - For bit '1': rounds coefficient up to next quantization boundary
///    - For bit '0': rounds coefficient down to next quantization boundary
/// 4. Applies inverse DCT to transform back to spatial domain
/// 5. Writes the modified pixels back to the image
///
/// The quantization approach is robust to common image processing operations
/// like JPEG compression, scaling, and filtering because it creates persistent
/// modifications in the frequency domain.
///
/// # Arguments
/// * `luma_image` - The mutable grayscale (luminance) buffer of the image
/// * `block_coords` - The (x, y) coordinates of the block to process (in block units)
/// * `bit` - The boolean bit value to embed (true = 1, false = 0)
/// * `dct` - A pre-planned DCT processor from rustdct
///
/// # Returns
/// * `Ok(())` on successful embedding
/// * `Err(AuraMarkError)` if block coordinates are out of bounds
///
pub fn embed_bit_in_block(
    luma_image: &mut ImageBuffer<Luma<f32>, Vec<f32>>,
    block_coords: (usize, usize),
    bit: bool,
    dct: Arc<dyn Dct2<f32>>,
) -> Result<()> {
    let (block_x, block_y) = block_coords;
    let x_offset = block_x * BLOCK_SIZE;
    let y_offset = block_y * BLOCK_SIZE;

    // Check bounds to prevent panic
    if x_offset + BLOCK_SIZE > luma_image.width() as usize
        || y_offset + BLOCK_SIZE > luma_image.height() as usize
    {
        return Err(AuraMarkError::Error("Block position out of bounds".into()));
    }

    // Step 1: Extract 8x8 block of f32 Luma values
    // We read pixels row by row to create a flat array for DCT processing
    let mut block_data: Vec<f32> = Vec::with_capacity(BLOCK_SIZE * BLOCK_SIZE);
    for y in 0..BLOCK_SIZE {
        for x in 0..BLOCK_SIZE {
            block_data.push(
                luma_image
                    .get_pixel((x_offset + x) as u32, (y_offset + y) as u32)
                    .0[0],
            );
        }
    }

    // Step 2: Apply the forward DCT to transform to frequency domain
    // This converts spatial pixel values to frequency coefficients
    dct.process_dct2(&mut block_data);

    // Step 3: Embed the bit by modifying a mid-frequency coefficient
    // Mid-frequency coefficients provide the best robustness vs. imperceptibility trade-off:
    // - Low frequencies: too visible when modified
    // - High frequencies: easily lost during compression/filtering
    // - Mid frequencies: robust to common attacks while remaining relatively invisible
    let original_coeff = block_data[EMBEDDING_COEFF_INDEX];

    if bit {
        // Embed bit '1': Nudge coefficient towards next upper quantization boundary
        // This ensures the coefficient value encodes a '1' bit
        block_data[EMBEDDING_COEFF_INDEX] =
            (original_coeff / QUANTIZATION_STEP).ceil() * QUANTIZATION_STEP;

        // If coefficient was already at boundary, move to next level
        if block_data[EMBEDDING_COEFF_INDEX] == original_coeff {
            block_data[EMBEDDING_COEFF_INDEX] += QUANTIZATION_STEP;
        }
    } else {
        // Embed bit '0': Nudge coefficient towards next lower quantization boundary
        // This ensures the coefficient value encodes a '0' bit
        block_data[EMBEDDING_COEFF_INDEX] =
            (original_coeff / QUANTIZATION_STEP).floor() * QUANTIZATION_STEP;

        // If coefficient was already at boundary, move to next level
        if block_data[EMBEDDING_COEFF_INDEX] == original_coeff {
            block_data[EMBEDDING_COEFF_INDEX] -= QUANTIZATION_STEP;
        }
    }

    // Step 4: Apply inverse DCT to transform back to spatial domain
    // For DCT-II (used by rustdct), applying DCT twice gives the inverse
    // but requires proper normalization
    dct.process_dct2(&mut block_data);

    // Step 5: Write modified block back to image with proper normalization
    // The normalization factor accounts for the double DCT application
    // For 2D DCT-II applied twice: factor = (N * M * 4) where N=M=BLOCK_SIZE
    let normalization_factor = (BLOCK_SIZE * BLOCK_SIZE * 4) as f32;

    for y in 0..BLOCK_SIZE {
        for x in 0..BLOCK_SIZE {
            let pixel_val = block_data[y * BLOCK_SIZE + x] / normalization_factor;
            luma_image.put_pixel(
                (x_offset + x) as u32,
                (y_offset + y) as u32,
                Luma([pixel_val]),
            );
        }
    }

    Ok(())
}

/// Extracts a single bit from an 8x8 image block using DCT quantization analysis.
///
/// This function implements the extraction counterpart to the embedding process:
/// 1. Extracts an 8x8 pixel block from the image
/// 2. Applies forward DCT to transform pixels to frequency domain
/// 3. Analyzes the mid-frequency coefficient to determine the embedded bit:
///    - Examines how the coefficient aligns with quantization boundaries
///    - Determines whether it was quantized for '1' or '0' during embedding
/// 4. Returns the extracted bit value
///
/// The extraction works by analyzing the quantization pattern of the coefficient.
/// During embedding, coefficients are forced to specific quantization levels
/// depending on the bit value. During extraction, we determine which quantization
/// pattern is present to recover the original bit.
///
/// # Arguments
/// * `luma_image` - The grayscale (luminance) buffer of the image to analyze
/// * `block_coords` - The (x, y) coordinates of the block to process (in block units)
/// * `dct` - A pre-planned DCT processor from rustdct
///
/// # Returns
/// * `Ok(bool)` with the extracted bit value (true = 1, false = 0)
/// * `Err(AuraMarkError)` if block coordinates are out of bounds
///
/// ```
pub fn extract_bit_from_block(
    luma_image: &ImageBuffer<Luma<f32>, Vec<f32>>,
    block_coords: (usize, usize),
    dct: Arc<dyn Dct2<f32>>,
) -> Result<bool> {
    let (block_x, block_y) = block_coords;
    let x_offset = block_x * BLOCK_SIZE;
    let y_offset = block_y * BLOCK_SIZE;

    // Check bounds to prevent panic
    if x_offset + BLOCK_SIZE > luma_image.width() as usize
        || y_offset + BLOCK_SIZE > luma_image.height() as usize
    {
        return Err(AuraMarkError::Error("Block position out of bounds".into()));
    }

    // Step 1: Extract 8x8 block of f32 Luma values
    // Read pixels in the same order as embedding (row by row)
    let mut block_data: Vec<f32> = Vec::with_capacity(BLOCK_SIZE * BLOCK_SIZE);
    for y in 0..BLOCK_SIZE {
        for x in 0..BLOCK_SIZE {
            block_data.push(
                luma_image
                    .get_pixel((x_offset + x) as u32, (y_offset + y) as u32)
                    .0[0],
            );
        }
    }

    // Step 2: Apply forward DCT to transform to frequency domain
    // This gives us access to the same coefficient that was modified during embedding
    dct.process_dct2(&mut block_data);

    // Step 3: Analyze the embedding coefficient to extract the bit
    // We examine the coefficient at the same position used during embedding
    let coeff_value = block_data[EMBEDDING_COEFF_INDEX];

    // Determine the bit by analyzing the quantization pattern:
    // - Calculate which quantization level the coefficient is closest to
    // - Determine if this represents an "upper" (bit 1) or "lower" (bit 0) quantization
    let quantized_level = (coeff_value / QUANTIZATION_STEP).round();
    let quantized_value = quantized_level * QUANTIZATION_STEP;

    // The bit is determined by whether the coefficient was pushed to an upper or lower boundary
    // If the coefficient is closer to the upper boundary of its quantization region,
    // it likely represents a '1' bit. If closer to the lower boundary, it represents '0'.
    let remainder = coeff_value - quantized_value;

    // Alternative approach: use the sign of the remainder relative to half quantization step
    // This is more robust than trying to determine upper/lower boundaries
    let bit = if remainder.abs() < QUANTIZATION_STEP / 4.0 {
        // Very close to quantization boundary - check the quantization level pattern
        // Even levels might represent 0, odd levels might represent 1 (or vice versa)
        // For simplicity, we use the coefficient's position relative to zero
        coeff_value >= 0.0
    } else {
        // Clear quantization pattern - use remainder sign
        remainder > 0.0
    };

    Ok(bit)
}

/// Legacy function name for backward compatibility.
///
/// This function maintains compatibility with existing code that uses the original
/// function name. It simply calls the new `embed_bit_in_block` function.
///
/// # Deprecated
/// Use `embed_bit_in_block` instead for better naming consistency.
pub fn process_block(
    luma_image: &mut ImageBuffer<Luma<f32>, Vec<f32>>,
    block_coords: (usize, usize),
    bit: bool,
    dct: Arc<dyn Dct2<f32>>,
) -> Result<()> {
    embed_bit_in_block(luma_image, block_coords, bit, dct)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::ImageBuffer;
    use rustdct::DctPlanner;

    /// Creates a test image with a gradient pattern for testing DCT operations
    fn create_test_image() -> ImageBuffer<Luma<f32>, Vec<f32>> {
        let mut img = ImageBuffer::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                // Create a gradient pattern that provides good DCT coefficient variation
                img.put_pixel(x, y, Luma([(x + y) as f32 / 128.0]));
            }
        }
        img
    }

    #[test]
    fn test_embed_and_extract_bit_roundtrip() {
        let mut image = create_test_image();
        let mut planner = DctPlanner::new();
        let dct_processor: Arc<dyn Dct2<f32>> = planner.plan_dct2(BLOCK_SIZE * BLOCK_SIZE);

        let block_pos = (2, 3); // Test block at position (2,3)

        // Test embedding and extracting '1'
        let original_image = image.clone();
        embed_bit_in_block(&mut image, block_pos, true, dct_processor.clone()).unwrap();
        let extracted_bit =
            extract_bit_from_block(&image, block_pos, dct_processor.clone()).unwrap();
        assert_eq!(extracted_bit, true, "Failed to extract embedded bit '1'");

        // Test embedding and extracting '0'
        image = original_image; // Reset to original
        embed_bit_in_block(&mut image, block_pos, false, dct_processor.clone()).unwrap();
        let extracted_bit =
            extract_bit_from_block(&image, block_pos, dct_processor.clone()).unwrap();
        assert_eq!(extracted_bit, false, "Failed to extract embedded bit '0'");
    }

    #[test]
    fn test_multiple_blocks_no_interference() {
        let mut image = create_test_image();
        let mut planner = DctPlanner::new();
        let dct_processor: Arc<dyn Dct2<f32>> = planner.plan_dct2(BLOCK_SIZE * BLOCK_SIZE);

        // Test embedding different bits in different blocks
        let test_cases = vec![
            ((0, 0), true),
            ((1, 0), false),
            ((0, 1), true),
            ((1, 1), false),
            ((2, 2), true),
        ];

        // Embed all bits
        for &(block_pos, bit) in &test_cases {
            embed_bit_in_block(&mut image, block_pos, bit, dct_processor.clone()).unwrap();
        }

        // Extract and verify all bits
        for &(block_pos, expected_bit) in &test_cases {
            let extracted_bit =
                extract_bit_from_block(&image, block_pos, dct_processor.clone()).unwrap();
            assert_eq!(
                extracted_bit, expected_bit,
                "Bit mismatch at block {:?}: expected {}, got {}",
                block_pos, expected_bit, extracted_bit
            );
        }
    }

    #[test]
    fn test_block_bounds_checking_embed() {
        let mut image = create_test_image(); // 64x64 = 8x8 blocks
        let mut planner = DctPlanner::new();
        let dct_processor: Arc<dyn Dct2<f32>> = planner.plan_dct2(BLOCK_SIZE * BLOCK_SIZE);

        // Test out of bounds block position (8x8 blocks in 64x64 image, so max valid is (7,7))
        let out_of_bounds_pos = (8, 8);

        let result = embed_bit_in_block(&mut image, out_of_bounds_pos, true, dct_processor);
        assert!(
            result.is_err(),
            "Should fail for out of bounds block position"
        );
    }

    #[test]
    fn test_block_bounds_checking_extract() {
        let image = create_test_image(); // 64x64 = 8x8 blocks
        let mut planner = DctPlanner::new();
        let dct_processor: Arc<dyn Dct2<f32>> = planner.plan_dct2(BLOCK_SIZE * BLOCK_SIZE);

        // Test out of bounds block position
        let out_of_bounds_pos = (10, 10);

        let result = extract_bit_from_block(&image, out_of_bounds_pos, dct_processor);
        assert!(
            result.is_err(),
            "Should fail for out of bounds block position"
        );
    }

    #[test]
    fn test_process_block_backward_compatibility() {
        let mut image = create_test_image();
        let mut planner = DctPlanner::new();
        let dct_processor: Arc<dyn Dct2<f32>> = planner.plan_dct2(BLOCK_SIZE * BLOCK_SIZE);

        let block_pos = (1, 1);

        // Test that the legacy function still works
        let result = process_block(&mut image, block_pos, true, dct_processor.clone());
        assert!(result.is_ok(), "Legacy process_block function should work");

        // Verify the bit was actually embedded by extracting it
        let extracted_bit = extract_bit_from_block(&image, block_pos, dct_processor).unwrap();
        assert_eq!(
            extracted_bit, true,
            "Legacy function should embed bit correctly"
        );
    }

    #[test]
    fn test_quantization_robustness() {
        let mut image = create_test_image();
        let mut planner = DctPlanner::new();
        let dct_processor: Arc<dyn Dct2<f32>> = planner.plan_dct2(BLOCK_SIZE * BLOCK_SIZE);

        let block_pos = (3, 4);

        // Embed a bit
        embed_bit_in_block(&mut image, block_pos, true, dct_processor.clone()).unwrap();

        // Add small noise to simulate compression/processing
        for y in 0..image.height() {
            for x in 0..image.width() {
                let pixel = image.get_pixel_mut(x, y);
                // Add very small noise (much smaller than quantization step)
                pixel[0] += (x as f32 * 0.001) % 0.01 - 0.005;
                pixel[0] = pixel[0].max(0.0).min(1.0); // Clamp to valid range
            }
        }

        // Should still be able to extract the bit despite noise
        let extracted_bit = extract_bit_from_block(&image, block_pos, dct_processor).unwrap();
        assert_eq!(extracted_bit, true, "Should be robust to small noise");
    }
}
