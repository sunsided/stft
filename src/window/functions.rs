//! Raw window-coefficient generators, all computed in `f64`.
//!
//! Every generator builds a *symmetric* window of the requested length using
//! the conventional `denom = M - 1` normalization. Periodic ("DFT-even")
//! windows are obtained by [`super::mod`] generating a symmetric window one
//! sample longer and truncating, which matches NumPy/SciPy `fftbins=True` and
//! librosa.

use alloc::vec;
use alloc::vec::Vec;
use core::f64::consts::PI;
// In `no_std` builds `f64` has no inherent transcendental methods, so the
// `Float` trait (backed by `libm`) supplies them. Under `std` the inherent
// methods are used and importing the trait would be flagged as unused.
#[cfg(not(feature = "std"))]
use num_traits::Float;

/// Sum-of-cosines window of length `m` with the given coefficients
/// `[a0, a1, a2, ...]`: `w[i] = a0 - a1·cos(θ) + a2·cos(2θ) - a3·cos(3θ) + ...`
/// where `θ = 2π·i/(m-1)`.
pub(super) fn cosine_sum(m: usize, coeffs: &[f64]) -> Vec<f64> {
    if m == 1 {
        return vec![1.0];
    }
    let denom = (m - 1) as f64;
    (0..m)
        .map(|i| {
            let theta = 2.0 * PI * (i as f64) / denom;
            let mut acc = 0.0;
            for (k, &a) in coeffs.iter().enumerate() {
                let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
                acc += sign * a * (k as f64 * theta).cos();
            }
            acc
        })
        .collect()
}

pub(super) fn rectangular(m: usize) -> Vec<f64> {
    vec![1.0; m]
}

pub(super) fn hann(m: usize) -> Vec<f64> {
    cosine_sum(m, &[0.5, 0.5])
}

pub(super) fn hamming(m: usize) -> Vec<f64> {
    cosine_sum(m, &[0.54, 0.46])
}

pub(super) fn blackman(m: usize) -> Vec<f64> {
    cosine_sum(m, &[0.42, 0.5, 0.08])
}

pub(super) fn blackman_harris(m: usize) -> Vec<f64> {
    cosine_sum(m, &[0.358_75, 0.488_29, 0.141_28, 0.011_68])
}

pub(super) fn nuttall(m: usize) -> Vec<f64> {
    cosine_sum(m, &[0.363_581_9, 0.489_177_5, 0.136_599_5, 0.010_641_1])
}

pub(super) fn flat_top(m: usize) -> Vec<f64> {
    cosine_sum(
        m,
        &[
            0.215_578_95,
            0.416_631_58,
            0.277_263_158,
            0.083_578_947,
            0.006_947_368,
        ],
    )
}

/// Sine (a.k.a. cosine) window: `w[i] = sin(π·i/(m-1))`.
pub(super) fn cosine(m: usize) -> Vec<f64> {
    if m == 1 {
        return vec![1.0];
    }
    let denom = (m - 1) as f64;
    (0..m).map(|i| (PI * (i as f64) / denom).sin()).collect()
}

/// Bartlett (triangular with zero endpoints), matching `numpy.bartlett`.
pub(super) fn bartlett(m: usize) -> Vec<f64> {
    if m == 1 {
        return vec![1.0];
    }
    let half = (m - 1) as f64 / 2.0;
    (0..m)
        .map(|i| 1.0 - ((i as f64 - half) / half).abs())
        .collect()
}

/// Triangular window without zero endpoints, matching `scipy.signal.windows.triang`.
pub(super) fn triangular(m: usize) -> Vec<f64> {
    if m == 1 {
        return vec![1.0];
    }
    let mf = m as f64;
    (0..m)
        .map(|i| {
            // distance from the center, in samples
            let d = (i as f64 - (m - 1) as f64 / 2.0).abs();
            if m % 2 == 0 {
                1.0 - (2.0 * d) / mf
            } else {
                1.0 - (2.0 * d) / (mf + 1.0)
            }
        })
        .collect()
}

/// Welch (parabolic) window.
pub(super) fn welch(m: usize) -> Vec<f64> {
    if m == 1 {
        return vec![1.0];
    }
    let half = (m - 1) as f64 / 2.0;
    (0..m)
        .map(|i| {
            let x = (i as f64 - half) / half;
            1.0 - x * x
        })
        .collect()
}

/// Tukey (tapered cosine) window, matching `scipy.signal.windows.tukey`.
pub(super) fn tukey(m: usize, alpha: f64) -> Vec<f64> {
    if m == 1 {
        return vec![1.0];
    }
    let alpha = alpha.clamp(0.0, 1.0);
    if alpha == 0.0 {
        return rectangular(m);
    }
    if alpha >= 1.0 {
        return hann(m);
    }
    let denom = (m - 1) as f64;
    let width = alpha * denom / 2.0;
    (0..m)
        .map(|i| {
            let n = i as f64;
            if n < width {
                0.5 * (1.0 + (PI * (-1.0 + 2.0 * n / (alpha * denom))).cos())
            } else if n <= denom - width {
                1.0
            } else {
                0.5 * (1.0 + (PI * (-2.0 / alpha + 1.0 + 2.0 * n / (alpha * denom))).cos())
            }
        })
        .collect()
}

/// Modified Bessel function of the first kind, order zero (series expansion).
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let mut k = 1.0;
    loop {
        let r = x / (2.0 * k);
        term *= r * r;
        sum += term;
        if term <= 1e-16 * sum || k > 1_000.0 {
            break;
        }
        k += 1.0;
    }
    sum
}

/// Kaiser window with shape parameter `beta`.
pub(super) fn kaiser(m: usize, beta: f64) -> Vec<f64> {
    if m == 1 {
        return vec![1.0];
    }
    let denom = (m - 1) as f64;
    let i0_beta = bessel_i0(beta);
    (0..m)
        .map(|i| {
            let r = 2.0 * (i as f64) / denom - 1.0;
            bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) / i0_beta
        })
        .collect()
}

/// Gaussian window with standard deviation `std` (in samples), matching
/// `scipy.signal.windows.gaussian`.
pub(super) fn gaussian(m: usize, std: f64) -> Vec<f64> {
    if m == 1 {
        return vec![1.0];
    }
    let center = (m - 1) as f64 / 2.0;
    (0..m)
        .map(|i| {
            let n = (i as f64 - center) / std;
            (-0.5 * n * n).exp()
        })
        .collect()
}
