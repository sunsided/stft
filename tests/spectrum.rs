//! Spectrum helpers checked against exact, hand-derived values.

use approx::assert_abs_diff_eq;
use ruststft::spectrum::{
    magnitude, magnitude_into, phase, phase_into, power, power_into, power_to_db,
};
use ruststft::Complex;

/// Three bins with exact magnitudes `5`, `1` and `0`.
const BINS: [Complex<f64>; 3] = [
    Complex::new(3.0, 4.0),
    Complex::new(0.0, -1.0),
    Complex::new(0.0, 0.0),
];

/// Assert that two slices are equal to within `1e-12`.
fn assert_slice_eq(got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "length mismatch");
    for (&g, &w) in got.iter().zip(want) {
        assert_abs_diff_eq!(g, w, epsilon = 1e-12);
    }
}

/// The allocating and in-place forms of magnitude, power and phase agree with
/// the exact values.
#[test]
fn magnitude_power_phase() {
    let mags = [5.0, 1.0, 0.0];
    let pows = [25.0, 1.0, 0.0];
    let phases = [4.0f64.atan2(3.0), -core::f64::consts::FRAC_PI_2, 0.0];
    let mut out = [f64::NAN; 3];

    assert_slice_eq(&magnitude(&BINS), &mags);
    magnitude_into(&BINS, &mut out);
    assert_slice_eq(&out, &mags);

    assert_slice_eq(&power(&BINS), &pows);
    power_into(&BINS, &mut out);
    assert_slice_eq(&out, &pows);

    assert_slice_eq(&phase(&BINS), &phases);
    phase_into(&BINS, &mut out);
    assert_slice_eq(&out, &phases);
}

/// `power_to_db` divides by the reference and clamps tiny values to `1e-10`.
#[test]
fn power_to_db_reference_and_amin() {
    let mut values = [100.0, 1.0, 1e-20];
    power_to_db(&mut values, 10.0, None);

    assert_slice_eq(&values, &[10.0, -10.0, -110.0]);
}

/// `power_to_db` with `top_db` floors every value at `max - top_db`.
#[test]
fn power_to_db_top_db_floor() {
    let mut values = [100.0, 1.0, 1e-20];
    power_to_db(&mut values, 1.0, Some(30.0));

    assert_slice_eq(&values, &[20.0, 0.0, -10.0]);
}
