//! Window coefficients checked against reference values from SciPy.
//!
//! The table comes from `scipy.signal.windows` (SciPy 1.16) for lengths 6 and 7
//! in both symmetries. Even and odd lengths take different code paths in some
//! families, so the test uses both.

use approx::assert_abs_diff_eq;
use ruststft::{Symmetry, Window, WindowFunction};

/// One reference case: the window family, its symmetry and the SciPy coefficients.
type Case = (WindowFunction, Symmetry, &'static [f64]);

/// SciPy reference coefficients. The length of each window is the slice length.
#[rustfmt::skip]
const CASES: &[Case] = &[
    (WindowFunction::Rectangular, Symmetry::Symmetric, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    (WindowFunction::Rectangular, Symmetry::Periodic, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    (WindowFunction::Rectangular, Symmetry::Symmetric, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    (WindowFunction::Rectangular, Symmetry::Periodic, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    (WindowFunction::Hann, Symmetry::Symmetric, &[0.0, 0.34549150281252633, 0.9045084971874737, 0.9045084971874737, 0.34549150281252633, 0.0]),
    (WindowFunction::Hann, Symmetry::Periodic, &[0.0, 0.2499999999999999, 0.75, 1.0, 0.7500000000000002, 0.25000000000000033]),
    (WindowFunction::Hann, Symmetry::Symmetric, &[0.0, 0.2499999999999999, 0.75, 1.0, 0.7500000000000002, 0.25000000000000033, 0.0]),
    (WindowFunction::Hann, Symmetry::Periodic, &[0.0, 0.18825509907063326, 0.6112604669781572, 0.9504844339512095, 0.9504844339512095, 0.6112604669781572, 0.18825509907063326]),
    (WindowFunction::Hamming, Symmetry::Symmetric, &[0.08000000000000007, 0.3978521825875243, 0.9121478174124759, 0.9121478174124759, 0.3978521825875243, 0.08000000000000007]),
    (WindowFunction::Hamming, Symmetry::Periodic, &[0.08000000000000007, 0.30999999999999994, 0.77, 1.0, 0.7700000000000002, 0.31000000000000033]),
    (WindowFunction::Hamming, Symmetry::Symmetric, &[0.08000000000000007, 0.30999999999999994, 0.77, 1.0, 0.7700000000000002, 0.31000000000000033, 0.08000000000000007]),
    (WindowFunction::Hamming, Symmetry::Periodic, &[0.08000000000000007, 0.25319469114498266, 0.6423596296199047, 0.9544456792351128, 0.9544456792351128, 0.6423596296199047, 0.25319469114498266]),
    (WindowFunction::Blackman, Symmetry::Symmetric, &[-1.3877787807814457e-17, 0.2007701432625305, 0.8492298567374694, 0.8492298567374694, 0.2007701432625305, -1.3877787807814457e-17]),
    (WindowFunction::Blackman, Symmetry::Periodic, &[-1.3877787807814457e-17, 0.1299999999999999, 0.6299999999999999, 0.9999999999999999, 0.6300000000000002, 0.1300000000000002]),
    (WindowFunction::Blackman, Symmetry::Symmetric, &[-1.3877787807814457e-17, 0.1299999999999999, 0.6299999999999999, 0.9999999999999999, 0.6300000000000002, 0.1300000000000002, -1.3877787807814457e-17]),
    (WindowFunction::Blackman, Symmetry::Periodic, &[-1.3877787807814457e-17, 0.09045342435412808, 0.45918295754596367, 0.9203636180999082, 0.9203636180999082, 0.45918295754596367, 0.09045342435412808]),
    (WindowFunction::BlackmanHarris, Symmetry::Symmetric, &[6.0000000000001025e-05, 0.10301148934566379, 0.7938335106543363, 0.7938335106543363, 0.10301148934566379, 6.0000000000001025e-05]),
    (WindowFunction::BlackmanHarris, Symmetry::Periodic, &[6.0000000000001025e-05, 0.05564499999999996, 0.5205749999999999, 1.0, 0.5205750000000002, 0.05564500000000015]),
    (WindowFunction::BlackmanHarris, Symmetry::Symmetric, &[6.0000000000001025e-05, 0.05564499999999996, 0.5205749999999999, 1.0, 0.5205750000000002, 0.05564500000000015, 6.0000000000001025e-05]),
    (WindowFunction::BlackmanHarris, Symmetry::Periodic, &[6.0000000000001025e-05, 0.0333917234781512, 0.33283350429856506, 0.8893697722232838, 0.8893697722232838, 0.33283350429856506, 0.0333917234781512]),
    (WindowFunction::Nuttall, Symmetry::Symmetric, &[0.0003628000000000381, 0.11051525304987181, 0.7982580969501283, 0.7982580969501283, 0.11051525304987181, 0.0003628000000000381]),
    (WindowFunction::Nuttall, Symmetry::Periodic, &[0.0003628000000000381, 0.06133449999999996, 0.5292298, 1.0, 0.5292298000000003, 0.06133450000000015]),
    (WindowFunction::Nuttall, Symmetry::Symmetric, &[0.0003628000000000381, 0.06133449999999996, 0.5292298, 1.0, 0.5292298000000003, 0.06133450000000015, 0.0003628000000000381]),
    (WindowFunction::Nuttall, Symmetry::Periodic, &[0.0003628000000000381, 0.03777576895352028, 0.34272761996881956, 0.8918518610776603, 0.8918518610776603, 0.34272761996881956, 0.03777576895352028]),
    (WindowFunction::FlatTop, Symmetry::Symmetric, &[-0.0004210510000000013, -0.06771425207621193, 0.606872152576212, 0.606872152576212, -0.06771425207621193, -0.0004210510000000013]),
    (WindowFunction::FlatTop, Symmetry::Periodic, &[-0.0004210510000000013, -0.05126315599999999, 0.19821052999999986, 1.000000003, 0.1982105300000003, -0.05126315600000008]),
    (WindowFunction::FlatTop, Symmetry::Symmetric, &[-0.0004210510000000013, -0.05126315599999999, 0.19821052999999986, 1.000000003, 0.1982105300000003, -0.05126315600000008, -0.0004210510000000013]),
    (WindowFunction::FlatTop, Symmetry::Periodic, &[-0.0004210510000000013, -0.03684078115492349, 0.010703716716153475, 0.78087391493877, 0.78087391493877, 0.010703716716153475, -0.03684078115492349]),
    (WindowFunction::Bartlett, Symmetry::Symmetric, &[0.0, 0.4, 0.8, 0.8, 0.3999999999999999, 0.0]),
    (WindowFunction::Bartlett, Symmetry::Periodic, &[0.0, 0.3333333333333333, 0.6666666666666666, 1.0, 0.6666666666666667, 0.33333333333333326]),
    (WindowFunction::Bartlett, Symmetry::Symmetric, &[0.0, 0.3333333333333333, 0.6666666666666666, 1.0, 0.6666666666666667, 0.33333333333333326, 0.0]),
    (WindowFunction::Bartlett, Symmetry::Periodic, &[0.0, 0.2857142857142857, 0.5714285714285714, 0.8571428571428571, 0.8571428571428572, 0.5714285714285714, 0.2857142857142858]),
    (WindowFunction::Triangular, Symmetry::Symmetric, &[0.16666666666666666, 0.5, 0.8333333333333334, 0.8333333333333334, 0.5, 0.16666666666666666]),
    (WindowFunction::Triangular, Symmetry::Periodic, &[0.25, 0.5, 0.75, 1.0, 0.75, 0.5]),
    (WindowFunction::Triangular, Symmetry::Symmetric, &[0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25]),
    (WindowFunction::Triangular, Symmetry::Periodic, &[0.125, 0.375, 0.625, 0.875, 0.875, 0.625, 0.375]),
    (WindowFunction::Tukey { alpha: 0.5 }, Symmetry::Symmetric, &[0.0, 0.9045084971874737, 1.0, 1.0, 0.9045084971874735, 0.0]),
    (WindowFunction::Tukey { alpha: 0.5 }, Symmetry::Periodic, &[0.0, 0.75, 1.0, 1.0, 1.0, 0.7499999999999999]),
    (WindowFunction::Tukey { alpha: 0.5 }, Symmetry::Symmetric, &[0.0, 0.75, 1.0, 1.0, 1.0, 0.7499999999999999, 0.0]),
    (WindowFunction::Tukey { alpha: 0.5 }, Symmetry::Periodic, &[0.0, 0.6112604669781572, 1.0, 1.0, 1.0, 1.0, 0.6112604669781576]),
    (WindowFunction::Tukey { alpha: 0.0 }, Symmetry::Symmetric, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    (WindowFunction::Tukey { alpha: 0.0 }, Symmetry::Periodic, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    (WindowFunction::Tukey { alpha: 0.0 }, Symmetry::Symmetric, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    (WindowFunction::Tukey { alpha: 0.0 }, Symmetry::Periodic, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    (WindowFunction::Tukey { alpha: 1.0 }, Symmetry::Symmetric, &[0.0, 0.34549150281252633, 0.9045084971874737, 0.9045084971874737, 0.34549150281252633, 0.0]),
    (WindowFunction::Tukey { alpha: 1.0 }, Symmetry::Periodic, &[0.0, 0.2499999999999999, 0.75, 1.0, 0.7500000000000002, 0.25000000000000033]),
    (WindowFunction::Tukey { alpha: 1.0 }, Symmetry::Symmetric, &[0.0, 0.2499999999999999, 0.75, 1.0, 0.7500000000000002, 0.25000000000000033, 0.0]),
    (WindowFunction::Tukey { alpha: 1.0 }, Symmetry::Periodic, &[0.0, 0.18825509907063326, 0.6112604669781572, 0.9504844339512095, 0.9504844339512095, 0.6112604669781572, 0.18825509907063326]),
    (WindowFunction::Kaiser { beta: 8.6 }, Symmetry::Symmetric, &[0.0013325139979024198, 0.20105486954533092, 0.8494161890910448, 0.8494161890910448, 0.20105486954533092, 0.0013325139979024198]),
    (WindowFunction::Kaiser { beta: 8.6 }, Symmetry::Periodic, &[0.0013325139979024198, 0.13040194654166437, 0.6304119273359405, 1.0, 0.6304119273359405, 0.13040194654166437]),
    (WindowFunction::Kaiser { beta: 8.6 }, Symmetry::Symmetric, &[0.0013325139979024198, 0.13040194654166437, 0.6304119273359405, 1.0, 0.6304119273359405, 0.13040194654166437, 0.0013325139979024198]),
    (WindowFunction::Kaiser { beta: 8.6 }, Symmetry::Periodic, &[0.0013325139979024198, 0.09113651292826536, 0.45964377459338085, 0.9204615832581574, 0.9204615832581574, 0.45964377459338085, 0.09113651292826536]),
    (WindowFunction::Kaiser { beta: 0.5 }, Symmetry::Symmetric, &[0.9403061933191574, 0.9782962393705391, 0.9975765035372048, 0.9975765035372048, 0.9782962393705391, 0.9403061933191574]),
    (WindowFunction::Kaiser { beta: 0.5 }, Symmetry::Periodic, &[0.9403061933191574, 0.9732402256999829, 0.9932754654413776, 1.0, 0.9932754654413776, 0.9732402256999829]),
    (WindowFunction::Kaiser { beta: 0.5 }, Symmetry::Symmetric, &[0.9403061933191574, 0.9732402256999829, 0.9932754654413776, 1.0, 0.9932754654413776, 0.9732402256999829, 0.9403061933191574]),
    (WindowFunction::Kaiser { beta: 0.5 }, Symmetry::Periodic, &[0.9403061933191574, 0.969312119993129, 0.9888964026825293, 0.9987631475060962, 0.9987631475060962, 0.9888964026825293, 0.969312119993129]),
    (WindowFunction::Gaussian { std: 1.5 }, Symmetry::Symmetric, &[0.24935220877729622, 0.6065306597126334, 0.9459594689067654, 0.9459594689067654, 0.6065306597126334, 0.24935220877729622]),
    (WindowFunction::Gaussian { std: 1.5 }, Symmetry::Periodic, &[0.1353352832366127, 0.41111229050718745, 0.8007374029168081, 1.0, 0.8007374029168081, 0.41111229050718745]),
    (WindowFunction::Gaussian { std: 1.5 }, Symmetry::Symmetric, &[0.1353352832366127, 0.41111229050718745, 0.8007374029168081, 1.0, 0.8007374029168081, 0.41111229050718745, 0.1353352832366127]),
    (WindowFunction::Gaussian { std: 1.5 }, Symmetry::Periodic, &[0.06572852861653047, 0.24935220877729622, 0.6065306597126334, 0.9459594689067654, 0.9459594689067654, 0.6065306597126334, 0.24935220877729622]),
];

/// Every window family matches SciPy to within `1e-12`.
#[test]
fn windows_match_scipy() {
    for &(function, symmetry, expected) in CASES {
        let window = Window::<f64>::new(function, expected.len(), symmetry);
        assert_eq!(window.len(), expected.len(), "{function:?} {symmetry:?}");

        for (i, (&got, &want)) in window.coefficients().iter().zip(expected).enumerate() {
            assert!(
                (got - want).abs() < 1e-12,
                "{function:?} {symmetry:?} len {} index {i}: got {got}, want {want}",
                expected.len()
            );
        }
    }
}

/// The Welch window has no SciPy equivalent. Its closed form `1 - x²` gives
/// exact values at length 5.
#[test]
fn welch_matches_closed_form() {
    let window = Window::<f64>::new(WindowFunction::Welch, 5, Symmetry::Symmetric);
    let expected = [0.0, 0.75, 1.0, 0.75, 0.0];

    for (&got, want) in window.coefficients().iter().zip(expected) {
        assert_abs_diff_eq!(got, want, epsilon = 1e-12);
    }
}

/// The cosine window follows its documented form `sin(π·i/(m-1))` with zero
/// endpoints.
///
/// SciPy's `cosine` uses `sin(π·(i+½)/m)` instead, so it is not in [`CASES`].
#[test]
fn cosine_matches_documented_form() {
    let window = Window::<f64>::new(WindowFunction::Cosine, 5, Symmetry::Symmetric);
    let half = core::f64::consts::FRAC_1_SQRT_2;
    let expected = [0.0, half, 1.0, half, 0.0];

    for (&got, want) in window.coefficients().iter().zip(expected) {
        assert_abs_diff_eq!(got, want, epsilon = 1e-12);
    }
}

/// The periodic cosine window is power-complementary at 50% overlap:
/// `w[i]² + w[i + n/2]² = 1`. Overlap-add and MDCT rely on this property.
#[test]
fn periodic_cosine_is_power_complementary() {
    for n in [4, 16, 64] {
        let window = Window::<f64>::cosine(n);
        let c = window.coefficients();

        for i in 0..n / 2 {
            let sum = c[i] * c[i] + c[i + n / 2] * c[i + n / 2];
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-12);
        }
    }
}

/// A window of length 1 is `[1.0]` and a window of length 0 is empty, for every family.
#[test]
fn degenerate_lengths() {
    for &(function, symmetry, _) in CASES {
        assert_eq!(
            Window::<f64>::new(function, 1, symmetry).coefficients(),
            &[1.0]
        );
        assert!(Window::<f64>::new(function, 0, symmetry).is_empty());
    }
}

/// The cached sums agree with the coefficients.
#[test]
fn cached_sums_match_coefficients() {
    for &(function, symmetry, expected) in CASES {
        let window = Window::<f64>::new(function, expected.len(), symmetry);
        let sum: f64 = expected.iter().sum();
        let sum_sq: f64 = expected.iter().map(|w| w * w).sum();

        assert_abs_diff_eq!(window.sum(), sum, epsilon = 1e-12);
        assert_abs_diff_eq!(window.sum_squared(), sum_sq, epsilon = 1e-12);
        assert!(!window.is_empty());
    }
}

/// A NaN Tukey `alpha` gives a rectangular window, not NaN coefficients.
#[test]
fn tukey_nan_alpha_is_rectangular() {
    let window = Window::<f64>::new(
        WindowFunction::Tukey { alpha: f64::NAN },
        7,
        Symmetry::Symmetric,
    );
    assert_eq!(window.coefficients(), &[1.0; 7]);
}

/// Kaiser coefficients stay finite up to the documented `|beta| <= 700` bound.
#[test]
fn kaiser_large_beta_is_finite() {
    for beta in [-700.0, 700.0] {
        let window = Window::<f64>::kaiser(64, beta);
        assert!(
            window.coefficients().iter().all(|c| c.is_finite()),
            "beta = {beta}"
        );
    }
}
