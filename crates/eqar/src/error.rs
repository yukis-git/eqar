//! Error kinds returned instead of panicking.

use std::fmt;

/// Invalid input or state. Nothing in the crate panics on bad caller input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EqarError {
    /// Image size is not usable (width must equal 2 * height, width a multiple of 4).
    Size,
    /// Unknown pixel format.
    Format,
    /// Stride smaller than a row, or the buffer is too short for the view.
    Stride,
    /// Scratch allocation failed or the arena is exhausted.
    Alloc,
    /// Unknown dictionary.
    Dict,
    /// Output capacity too small.
    Cap,
    /// Detector not initialised.
    NoInit,
    /// A parameter is outside its allowed range.
    Params,
}

impl fmt::Display for EqarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            EqarError::Size => "invalid image size",
            EqarError::Format => "invalid pixel format",
            EqarError::Stride => "invalid stride or buffer too short",
            EqarError::Alloc => "allocation failed or arena exhausted",
            EqarError::Dict => "unknown dictionary",
            EqarError::Cap => "output capacity too small",
            EqarError::NoInit => "detector not initialised",
            EqarError::Params => "parameter out of range",
        };
        f.write_str(s)
    }
}

impl std::error::Error for EqarError {}
