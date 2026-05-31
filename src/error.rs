//! Error type returned by the fallible parts of the crate.

use core::fmt;

/// Errors produced while configuring or running an STFT/ISTFT.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum StftError {
    /// No window was supplied to a builder, or a zero-length window was given.
    MissingWindow,
    /// The frame length (window length) is zero.
    InvalidFrameLength,
    /// The hop size is invalid (must be `1..=frame_len`).
    InvalidHopSize {
        /// The requested hop size.
        hop: usize,
        /// The frame length it was checked against.
        frame_len: usize,
    },
    /// The FFT size is smaller than the frame length.
    InvalidFftSize {
        /// The requested FFT size.
        fft_size: usize,
        /// The frame length it must be at least as large as.
        frame_len: usize,
    },
    /// A supplied buffer did not have the expected length.
    LengthMismatch {
        /// The length that was expected.
        expected: usize,
        /// The length that was supplied.
        got: usize,
    },
    /// [`Scaling::Density`](crate::Scaling) was requested without a sample rate.
    MissingSampleRate,
    /// A processing call needed more buffered samples than were available.
    NotEnoughData {
        /// Samples required to compute a frame.
        needed: usize,
        /// Samples currently available.
        available: usize,
    },
    /// The underlying FFT backend reported an error.
    Fft,
}

impl fmt::Display for StftError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingWindow => f.write_str("no window was supplied"),
            Self::InvalidFrameLength => f.write_str("frame length must be greater than zero"),
            Self::InvalidHopSize { hop, frame_len } => write!(
                f,
                "hop size {hop} is invalid for frame length {frame_len} (expected 1..={frame_len})"
            ),
            Self::InvalidFftSize {
                fft_size,
                frame_len,
            } => write!(
                f,
                "fft size {fft_size} must be at least the frame length {frame_len}"
            ),
            Self::LengthMismatch { expected, got } => {
                write!(f, "buffer length mismatch: expected {expected}, got {got}")
            }
            Self::MissingSampleRate => {
                f.write_str("density scaling requires a sample rate to be configured")
            }
            Self::NotEnoughData { needed, available } => write!(
                f,
                "not enough buffered samples: needed {needed}, have {available}"
            ),
            Self::Fft => f.write_str("the FFT backend reported an error"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for StftError {}
