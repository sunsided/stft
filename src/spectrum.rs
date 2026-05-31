//! Helpers for turning complex STFT coefficients into magnitudes, powers,
//! phases and decibels.
//!
//! All functions operate on plain slices so they are usable on a single
//! spectrogram column, a whole flattened spectrogram, or any other layout.

use crate::sample::{cast, Sample};
use alloc::vec::Vec;
use num_complex::Complex;

/// Compute the magnitude (`|z|`) of each complex bin into `out`.
///
/// # Panics
/// Panics if `spectrum.len() != out.len()`.
pub fn magnitude_into<T: Sample>(spectrum: &[Complex<T>], out: &mut [T]) {
    assert_eq!(spectrum.len(), out.len(), "magnitude_into length mismatch");
    for (dst, z) in out.iter_mut().zip(spectrum) {
        *dst = z.norm();
    }
}

/// Allocate and return the magnitudes of `spectrum`.
#[must_use]
pub fn magnitude<T: Sample>(spectrum: &[Complex<T>]) -> Vec<T> {
    spectrum.iter().map(|z| z.norm()).collect()
}

/// Compute the power (`|z|²`) of each complex bin into `out`.
///
/// # Panics
/// Panics if `spectrum.len() != out.len()`.
pub fn power_into<T: Sample>(spectrum: &[Complex<T>], out: &mut [T]) {
    assert_eq!(spectrum.len(), out.len(), "power_into length mismatch");
    for (dst, z) in out.iter_mut().zip(spectrum) {
        *dst = z.norm_sqr();
    }
}

/// Allocate and return the power of `spectrum`.
#[must_use]
pub fn power<T: Sample>(spectrum: &[Complex<T>]) -> Vec<T> {
    spectrum.iter().map(|z| z.norm_sqr()).collect()
}

/// Compute the phase angle (in radians, `-π..=π`) of each bin into `out`.
///
/// # Panics
/// Panics if `spectrum.len() != out.len()`.
pub fn phase_into<T: Sample>(spectrum: &[Complex<T>], out: &mut [T]) {
    assert_eq!(spectrum.len(), out.len(), "phase_into length mismatch");
    for (dst, z) in out.iter_mut().zip(spectrum) {
        *dst = z.arg();
    }
}

/// Allocate and return the phase angles of `spectrum`.
#[must_use]
pub fn phase<T: Sample>(spectrum: &[Complex<T>]) -> Vec<T> {
    spectrum.iter().map(|z| z.arg()).collect()
}

/// Smallest value clamped to before taking a logarithm, avoiding `-inf`.
fn amin<T: Sample>() -> T {
    cast(1e-10)
}

/// Convert amplitudes to decibels in place: `20·log₁₀(max(|a|, amin) / reference)`.
///
/// If `top_db` is `Some(d)`, values are floored at `max - d`, matching
/// librosa's `amplitude_to_db`.
pub fn amplitude_to_db<T: Sample>(amplitudes: &mut [T], reference: T, top_db: Option<T>) {
    let amin = amin::<T>();
    let twenty = cast::<T>(20.0);
    let log_ref = reference.abs().max(amin).log10();
    convert_to_db(amplitudes, twenty, log_ref, amin, top_db);
}

/// Convert powers to decibels in place: `10·log₁₀(max(p, amin) / reference)`.
///
/// If `top_db` is `Some(d)`, values are floored at `max - d`, matching
/// librosa's `power_to_db`.
pub fn power_to_db<T: Sample>(powers: &mut [T], reference: T, top_db: Option<T>) {
    let amin = amin::<T>();
    let ten = cast::<T>(10.0);
    let log_ref = reference.abs().max(amin).log10();
    convert_to_db(powers, ten, log_ref, amin, top_db);
}

fn convert_to_db<T: Sample>(values: &mut [T], factor: T, log_ref: T, amin: T, top_db: Option<T>) {
    let mut max_db = T::neg_infinity();
    for v in values.iter_mut() {
        let db = factor * (v.max(amin).log10() - log_ref);
        *v = db;
        if db > max_db {
            max_db = db;
        }
    }
    if let Some(top) = top_db {
        let floor = max_db - top.abs();
        for v in values.iter_mut() {
            if *v < floor {
                *v = floor;
            }
        }
    }
}
