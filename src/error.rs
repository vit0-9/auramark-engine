use thiserror::Error;

/// The central error type for all operations in the auramark_engine.
#[derive(Error, Debug)]
pub enum AuraMarkError {
    #[error("Error: {0}")]
    Error(String),
    #[error("Image processing error: {0}")]
    ImageError(#[from] image::ImageError),

    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Watermark data is invalid or corrupt")]
    InvalidWatermarkData,

    #[error("The provided image dimensions are too small to embed a watermark")]
    ImageTooSmall,

    #[error("An unknown error has occurred")]
    Unknown,
}

// Manually implement PartialEq for AuraMarkError
impl PartialEq for AuraMarkError {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (AuraMarkError::ImageTooSmall, AuraMarkError::ImageTooSmall) => true,
            // For errors with String messages, compare the messages
            (AuraMarkError::Error(s1), AuraMarkError::Error(s2)) => s1 == s2,
            // For errors with `#[from]` foreign types, only compare the *variant*
            // We cannot compare the inner `image::ImageError` or `std::io::Error` directly.
            // If you need more specific comparison for tests, you'd inspect the inner error's kind/message.
            (AuraMarkError::ImageError(_), AuraMarkError::ImageError(_)) => {
                // Here you might add more specific logic if needed, e.g.:
                // self.to_string() == other.to_string()
                // For a basic test, just matching the variant is often enough.
                true
            }
            (AuraMarkError::IoError(_), AuraMarkError::IoError(_)) => {
                // Same here, usually you wouldn't compare `std::io::Error` directly.
                true
            }
            // Add other variants if you have them
            _ => false, // All other variant combinations are not equal
        }
    }
}

/// A centralized result type for our library.
pub type Result<T> = std::result::Result<T, AuraMarkError>;
