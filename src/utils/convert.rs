use image::{DynamicImage, ImageBuffer, Luma, Rgb, RgbImage};

/// Converts an ImageBuffer with f32 grayscale pixels (Luma32F)
/// to an 8-bit grayscale image (Luma8) suitable for PNG encoding.
pub fn luma32f_to_luma8(img: &ImageBuffer<Luma<f32>, Vec<f32>>) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    let (width, height) = img.dimensions();
    let mut out = ImageBuffer::new(width, height);

    for (x, y, pixel) in img.enumerate_pixels() {
        let clamped = pixel[0].clamp(0.0, 1.0);
        let val = (clamped * 255.0).round() as u8;
        out.put_pixel(x, y, Luma([val]));
    }

    out
}

/// Converts an 8-bit grayscale image (Luma8) to a floating-point grayscale image (Luma32F).
pub fn luma8_to_luma32f(img: &ImageBuffer<Luma<u8>, Vec<u8>>) -> ImageBuffer<Luma<f32>, Vec<f32>> {
    let (width, height) = img.dimensions();
    let mut out = ImageBuffer::new(width, height);

    for (x, y, pixel) in img.enumerate_pixels() {
        let val = pixel[0] as f32 / 255.0;
        out.put_pixel(x, y, Luma([val]));
    }

    out
}

/// Converts a DynamicImage of any kind to a floating-point grayscale image (Luma32F).
// Consider removing this and using `img.to_luma32f()` directly where possible.
pub fn dynamic_to_luma32f(img: &DynamicImage) -> ImageBuffer<Luma<f32>, Vec<f32>> {
    let gray8 = img.to_luma8();
    luma8_to_luma32f(&gray8)
}

/// Converts a floating-point grayscale image (Luma32F) to a DynamicImage (Luma8).
/// Suitable for saving as PNG or other 8-bit grayscale formats.
pub fn luma32f_to_dynamic(img: &ImageBuffer<Luma<f32>, Vec<f32>>) -> DynamicImage {
    DynamicImage::ImageLuma8(luma32f_to_luma8(img))
}

/// Convert an RGB image to grayscale f32 Luma32F (normalized 0.0-1.0).
/// Useful if you want to extract luminance from a color image before embedding.
pub fn rgb_to_luma32f(img: &RgbImage) -> ImageBuffer<Luma<f32>, Vec<f32>> {
    let (width, height) = img.dimensions();
    let mut out = ImageBuffer::new(width, height);

    for (x, y, pixel) in img.enumerate_pixels() {
        // Use standard luminance conversion (Rec. 709)
        let r = pixel[0] as f32 / 255.0;
        let g = pixel[1] as f32 / 255.0;
        let b = pixel[2] as f32 / 255.0;
        let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        out.put_pixel(x, y, Luma([lum]));
    }

    out
}

/// Convert grayscale Luma32F image back to RGB image by duplicating luminance.
/// This can be used if you want to create a visual RGB image from grayscale data.
/// This simply sets R=G=B=luminance for each pixel.
pub fn luma32f_to_rgb(img: &ImageBuffer<Luma<f32>, Vec<f32>>) -> RgbImage {
    let (width, height) = img.dimensions();
    let mut out = ImageBuffer::new(width, height);

    for (x, y, pixel) in img.enumerate_pixels() {
        let clamped = pixel[0].clamp(0.0, 1.0);
        let val = (clamped * 255.0).round() as u8;
        out.put_pixel(x, y, Rgb([val, val, val]));
    }

    out
}

// YCbCr formulas per ITU-R BT.601 (Y in [0,1], Cb, Cr in [-0.5, 0.5])
// Note: The Cb/Cr definitions often use a range of [-0.5, 0.5] from the get-go,
// but for pixel data, they might be scaled to [0, 1] or [16, 240] for limited range.
// Your `rgb_to_ycbcr` outputs Cb/Cr in [0,1], so the inverse should account for that.

pub fn rgb_to_ycbcr(pixel: Rgb<u8>) -> (f32, f32, f32) {
    let r = pixel[0] as f32 / 255.0;
    let g = pixel[1] as f32 / 255.0;
    let b = pixel[2] as f32 / 255.0;

    // YCbCr formulas per ITU-R BT.601 (often used in JPEG)
    // Y in [0, 1], Cb, Cr in [0, 1] (or similar scaled range)
    // For full range 0-255 RGB:
    // Y = 0.299*R + 0.587*G + 0.114*B
    // Cb = 0.5 * (B - Y) / (1 - 0.114) + 0.5  => Cb = -0.1687*R - 0.3313*G + 0.5*B + 0.5
    // Cr = 0.5 * (R - Y) / (1 - 0.299) + 0.5  => Cr = 0.5*R - 0.4187*G - 0.0813*B + 0.5

    let y = 0.299 * r + 0.587 * g + 0.114 * b;
    let cb = -0.168736 * r - 0.331264 * g + 0.500000 * b + 0.5; // Simplified from the previous form
    let cr = 0.500000 * r - 0.418688 * g - 0.081312 * b + 0.5; // Simplified from the previous form

    (y, cb, cr)
}

/// Convert YCbCr pixel to RGB pixel (clamping output to 0..255)
pub fn ycbcr_to_rgb(y: f32, cb: f32, cr: f32) -> Rgb<u8> {
    // Clamp inputs to expected range [0, 1]
    let y = y.clamp(0.0, 1.0);
    let cb = cb.clamp(0.0, 1.0);
    let cr = cr.clamp(0.0, 1.0);

    // Convert Cb and Cr back to the [-0.5, 0.5] range for inverse calculation
    let cb_centered = cb - 0.5;
    let cr_centered = cr - 0.5;

    // Standard ITU-R BT.601 inverse formulas
    let r_linear = y + 1.402 * cr_centered;
    let g_linear = y - 0.344136 * cb_centered - 0.714136 * cr_centered;
    let b_linear = y + 1.772 * cb_centered;

    // Clamp and convert to u8
    let r_u8 = (r_linear * 255.0).round().clamp(0.0, 255.0) as u8;
    let g_u8 = (g_linear * 255.0).round().clamp(0.0, 255.0) as u8;
    let b_u8 = (b_linear * 255.0).round().clamp(0.0, 255.0) as u8;

    Rgb([r_u8, g_u8, b_u8])
}

/// Merge luminance with original chrominance and reconstruct RGB image
pub fn merge_luminance_back(
    original: &RgbImage,
    watermarked_luma: &ImageBuffer<Luma<f32>, Vec<f32>>,
) -> RgbImage {
    let (width, height) = original.dimensions();
    let mut out_img = ImageBuffer::new(width, height);

    for (x, y, orig_pixel) in original.enumerate_pixels() {
        let y_new = watermarked_luma.get_pixel(x, y).0[0];
        // Extract original chrominance components
        let (_, cb_orig, cr_orig) = rgb_to_ycbcr(*orig_pixel);

        // Reconstruct RGB with new luminance and original chrominance
        let rgb_pixel = ycbcr_to_rgb(y_new, cb_orig, cr_orig);
        out_img.put_pixel(x, y, rgb_pixel);
    }

    out_img
}

pub fn extract_luminance(image: &RgbImage) -> ImageBuffer<Luma<f32>, Vec<f32>> {
    let (width, height) = image.dimensions();
    let mut luma_img = ImageBuffer::new(width, height);

    for (x, y, pixel) in image.enumerate_pixels() {
        let (y_val, _cb, _cr) = rgb_to_ycbcr(*pixel);
        luma_img.put_pixel(x, y, Luma([y_val]));
    }

    luma_img
}
