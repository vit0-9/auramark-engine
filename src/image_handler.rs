use anyhow::{Result, anyhow};
use image::{DynamicImage, ImageFormat};
use std::io::Cursor;

/// Load image from bytes (supports PNG, JPEG, BMP, etc.)
pub fn load_image_from_bytes(bytes: &[u8]) -> Result<DynamicImage> {
    let img = image::load_from_memory(bytes).map_err(|e| anyhow!("Image decode error: {}", e))?;
    Ok(img)
}

/// Save image to PNG bytes
pub fn save_image_to_bytes(image: &DynamicImage) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    image
        .write_to(&mut Cursor::new(&mut buf), ImageFormat::Png)
        .map_err(|e| anyhow!("Image encode error: {}", e))?;
    Ok(buf)
}
