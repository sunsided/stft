//! Configuration enums shared by the forward and inverse transforms.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// How the complex STFT coefficients are scaled.
///
/// The variants mirror the scaling modes of SciPy's `ShortTimeFFT`:
///
/// - [`Scaling::None`] leaves the raw FFT output untouched.
/// - [`Scaling::Magnitude`] divides every bin by the sum of the window
///   coefficients, so a sinusoid's bin reflects its amplitude.
/// - [`Scaling::Density`] divides by `sqrt(fs * Σ wᵢ²)`, so that `|S|²`
///   approximates a power spectral density. Requires a configured sample rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Scaling {
    /// No scaling: raw FFT coefficients.
    #[default]
    None,
    /// Amplitude (magnitude) scaling: divide by the window sum.
    Magnitude,
    /// Power-spectral-density scaling: divide by `sqrt(fs * Σ wᵢ²)`.
    Density,
}

/// How a signal is padded when centered framing is enabled in batch mode.
///
/// With [`center`](crate::StftBuilder::center) enabled the signal is padded by
/// `fft_size / 2` samples on each side so that frame `t` is centered on sample
/// `t * hop`, matching librosa's convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum PadMode {
    /// Pad with zeros.
    #[default]
    Zero,
    /// Mirror the signal at the boundary (without repeating the edge sample).
    Reflect,
    /// Repeat the edge sample.
    Edge,
}
