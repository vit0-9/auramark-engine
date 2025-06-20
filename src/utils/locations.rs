//! Utilities for generating pseudo-random block locations for watermarking.

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::collections::HashSet;

use crate::error::{AuraMarkError, Result};

/// Generate pseudo-random, non-overlapping block locations for watermark embedding.
///
/// # Arguments
/// * `block_dimensions` - (blocks_x, blocks_y) dimensions of the image in blocks
/// * `secret_key` - Secret key used to seed the random number generator
/// * `count` - Number of unique block locations to generate
///
/// # Returns
/// A sorted vector of (x, y) block coordinates to ensure deterministic ordering
pub fn generate_embedding_block_locations(
    block_dimensions: (usize, usize),
    secret_key: &[u8],
    count: usize,
) -> Result<Vec<(usize, usize)>> {
    let (blocks_x, blocks_y) = block_dimensions;

    // Create deterministic seed from secret key
    let seed: [u8; 32] = {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(secret_key);
        hasher.finalize().into()
    };
    let mut rng = ChaCha8Rng::from_seed(seed);

    // Check if we have enough blocks
    if (blocks_x * blocks_y) < count {
        return Err(AuraMarkError::ImageTooSmall);
    }

    // Generate unique random locations
    let mut locations = HashSet::with_capacity(count);
    while locations.len() < count {
        let x = rng.random_range(0..blocks_x);
        let y = rng.random_range(0..blocks_y);
        locations.insert((x, y));
    }

    // Sort the locations to ensure deterministic order
    let mut result: Vec<(usize, usize)> = locations.into_iter().collect();
    result.sort();
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_embedding_block_locations_deterministic() {
        let block_dimensions = (100, 100);
        let count = 50;
        let secret_key_1 = b"my_secret_key_for_locations";
        let secret_key_2 = b"my_secret_key_for_locations";

        let locations_1 =
            generate_embedding_block_locations(block_dimensions, secret_key_1, count).unwrap();

        let locations_2 =
            generate_embedding_block_locations(block_dimensions, secret_key_2, count).unwrap();

        assert_eq!(locations_1, locations_2);

        let secret_key_3 = b"a_DIFFERENT_secret_key";
        let locations_3 =
            generate_embedding_block_locations(block_dimensions, secret_key_3, count).unwrap();

        assert_ne!(locations_1, locations_3);
    }

    #[test]
    fn test_generate_embedding_block_locations_no_duplicates() {
        let block_dimensions = (20, 20);
        let count = 100;
        let secret_key = b"unique_locations_test";

        let locations =
            generate_embedding_block_locations(block_dimensions, secret_key, count).unwrap();

        let unique_count = locations.iter().collect::<HashSet<_>>().len();
        assert_eq!(
            locations.len(),
            unique_count,
            "Generated locations should be unique"
        );
        assert_eq!(
            locations.len(),
            count,
            "Should generate the exact number of requested locations"
        );
    }

    #[test]
    fn test_insufficient_blocks() {
        let block_dimensions = (5, 5); // 25 blocks total
        let count = 30; // Need more than available
        let secret_key = b"test_key";

        let result = generate_embedding_block_locations(block_dimensions, secret_key, count);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), AuraMarkError::ImageTooSmall);
    }
}
