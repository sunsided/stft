//! Mel filterbank, (log-)mel spectrograms and MFCCs.
//!
//! This module is pure math and builds without `std`. It operates on power
//! spectra (`|S|²`), which you obtain from a [`Spectrogram`](crate::Spectrogram)
//! column via [`spectrum::power`](crate::spectrum::power).
//!
//! Conventions follow librosa's defaults: the Slaney mel scale with
//! area-normalized triangular filters, and an orthonormal DCT-II for MFCCs.

use crate::sample::{cast, Sample};
use alloc::vec;
use alloc::vec::Vec;
use core::f64::consts::PI;

#[cfg(not(feature = "std"))]
use num_traits::Float;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// The mel frequency scale convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum MelScale {
    /// Slaney (Auditory Toolbox) scale, librosa's default.
    #[default]
    Slaney,
    /// HTK scale: `2595·log₁₀(1 + f/700)`.
    Htk,
}

const SLANEY_F_SP: f64 = 200.0 / 3.0;
const SLANEY_MIN_LOG_HZ: f64 = 1000.0;

/// Convert a frequency in Hz to mels.
#[must_use]
pub fn hz_to_mel(hz: f64, scale: MelScale) -> f64 {
    match scale {
        MelScale::Htk => 2595.0 * (1.0 + hz / 700.0).log10(),
        MelScale::Slaney => {
            let min_log_mel = SLANEY_MIN_LOG_HZ / SLANEY_F_SP;
            let logstep = 6.4f64.ln() / 27.0;
            if hz >= SLANEY_MIN_LOG_HZ {
                min_log_mel + (hz / SLANEY_MIN_LOG_HZ).ln() / logstep
            } else {
                hz / SLANEY_F_SP
            }
        }
    }
}

/// Convert mels to a frequency in Hz.
#[must_use]
pub fn mel_to_hz(mel: f64, scale: MelScale) -> f64 {
    match scale {
        MelScale::Htk => 700.0 * (10.0f64.powf(mel / 2595.0) - 1.0),
        MelScale::Slaney => {
            let min_log_mel = SLANEY_MIN_LOG_HZ / SLANEY_F_SP;
            let logstep = 6.4f64.ln() / 27.0;
            if mel >= min_log_mel {
                SLANEY_MIN_LOG_HZ * (logstep * (mel - min_log_mel)).exp()
            } else {
                mel * SLANEY_F_SP
            }
        }
    }
}

/// A bank of triangular mel filters mapping `n_freqs` linear bins to `n_mels`
/// mel bands.
#[derive(Debug, Clone, PartialEq)]
pub struct MelFilterBank<T> {
    weights: Vec<T>,
    n_mels: usize,
    n_freqs: usize,
}

impl<T: Sample> MelFilterBank<T> {
    /// Construct a mel filterbank (librosa `mel` with `norm='slaney'`).
    ///
    /// - `n_fft` is the FFT size (the bank has `n_fft / 2 + 1` linear bins).
    /// - `fmin`/`fmax` bound the mel band edges in Hz.
    #[must_use]
    pub fn new(
        n_mels: usize,
        n_fft: usize,
        sample_rate: f64,
        fmin: f64,
        fmax: f64,
        scale: MelScale,
    ) -> Self {
        let n_freqs = n_fft / 2 + 1;
        let mut weights = vec![T::zero(); n_mels * n_freqs];

        // Linear FFT bin frequencies.
        let fft_freqs: Vec<f64> = (0..n_freqs)
            .map(|k| k as f64 * sample_rate / n_fft as f64)
            .collect();

        // `n_mels + 2` mel band edges, converted back to Hz.
        let mel_min = hz_to_mel(fmin, scale);
        let mel_max = hz_to_mel(fmax, scale);
        let hz_points: Vec<f64> = (0..n_mels + 2)
            .map(|i| {
                let mel = mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64;
                mel_to_hz(mel, scale)
            })
            .collect();

        for m in 0..n_mels {
            let lower_edge = hz_points[m];
            let center = hz_points[m + 1];
            let upper_edge = hz_points[m + 2];
            let lower_width = center - lower_edge;
            let upper_width = upper_edge - center;
            // Slaney area normalization.
            let enorm = 2.0 / (upper_edge - lower_edge);
            for (k, &f) in fft_freqs.iter().enumerate() {
                let lower = if lower_width > 0.0 {
                    (f - lower_edge) / lower_width
                } else {
                    0.0
                };
                let upper = if upper_width > 0.0 {
                    (upper_edge - f) / upper_width
                } else {
                    0.0
                };
                let w = lower.min(upper).max(0.0) * enorm;
                weights[m * n_freqs + k] = cast(w);
            }
        }

        Self {
            weights,
            n_mels,
            n_freqs,
        }
    }

    /// Number of mel bands.
    #[must_use]
    pub fn n_mels(&self) -> usize {
        self.n_mels
    }

    /// Number of linear frequency bins expected as input.
    #[must_use]
    pub fn n_freqs(&self) -> usize {
        self.n_freqs
    }

    /// Filter weights, row-major `[n_mels, n_freqs]`.
    #[must_use]
    pub fn weights(&self) -> &[T] {
        &self.weights
    }

    /// Apply the filterbank to one power-spectrum column into `out`.
    ///
    /// # Panics
    /// Panics if `power.len() != n_freqs` or `out.len() != n_mels`.
    pub fn transform_into(&self, power: &[T], out: &mut [T]) {
        assert_eq!(power.len(), self.n_freqs, "mel input length mismatch");
        assert_eq!(out.len(), self.n_mels, "mel output length mismatch");
        for (m, slot) in out.iter_mut().enumerate() {
            let row = &self.weights[m * self.n_freqs..(m + 1) * self.n_freqs];
            let mut acc = T::zero();
            for (&w, &p) in row.iter().zip(power) {
                acc = acc + w * p;
            }
            *slot = acc;
        }
    }

    /// Apply the filterbank to one power-spectrum column, allocating the result.
    #[must_use]
    pub fn transform(&self, power: &[T]) -> Vec<T> {
        let mut out = vec![T::zero(); self.n_mels];
        self.transform_into(power, &mut out);
        out
    }
}

/// An orthonormal type-II DCT, precomputed as a basis matrix.
///
/// Matches `scipy.fftpack.dct(type=2, norm='ortho')`, which is what librosa
/// uses to turn a log-mel spectrum into MFCCs.
#[derive(Debug, Clone, PartialEq)]
pub struct DctII<T> {
    basis: Vec<T>,
    n_in: usize,
    n_out: usize,
}

impl<T: Sample> DctII<T> {
    /// Build a DCT-II that maps `n_in` inputs to the first `n_out` coefficients.
    #[must_use]
    pub fn new(n_in: usize, n_out: usize) -> Self {
        let mut basis = vec![T::zero(); n_out * n_in];
        let n = n_in as f64;
        for k in 0..n_out {
            let f = if k == 0 {
                (1.0 / (4.0 * n)).sqrt()
            } else {
                (1.0 / (2.0 * n)).sqrt()
            };
            for m in 0..n_in {
                let v = 2.0 * f * (PI * k as f64 * (2.0 * m as f64 + 1.0) / (2.0 * n)).cos();
                basis[k * n_in + m] = cast(v);
            }
        }
        Self { basis, n_in, n_out }
    }

    /// Number of input samples.
    #[must_use]
    pub fn n_in(&self) -> usize {
        self.n_in
    }

    /// Number of output coefficients.
    #[must_use]
    pub fn n_out(&self) -> usize {
        self.n_out
    }

    /// Transform one input column into `out`.
    ///
    /// # Panics
    /// Panics if `input.len() != n_in` or `out.len() != n_out`.
    pub fn transform_into(&self, input: &[T], out: &mut [T]) {
        assert_eq!(input.len(), self.n_in, "DCT input length mismatch");
        assert_eq!(out.len(), self.n_out, "DCT output length mismatch");
        for (k, slot) in out.iter_mut().enumerate() {
            let row = &self.basis[k * self.n_in..(k + 1) * self.n_in];
            let mut acc = T::zero();
            for (&b, &x) in row.iter().zip(input) {
                acc = acc + b * x;
            }
            *slot = acc;
        }
    }

    /// Transform one input column, allocating the result.
    #[must_use]
    pub fn transform(&self, input: &[T]) -> Vec<T> {
        let mut out = vec![T::zero(); self.n_out];
        self.transform_into(input, &mut out);
        out
    }
}
