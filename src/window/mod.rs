//! Analysis/synthesis windows.
//!
//! [`Window`] holds the (already evaluated) coefficients together with their
//! sum and sum-of-squares, which the scaling modes and overlap-add
//! normalization need. [`WindowFunction`] is a serializable description of a
//! parametric window family that can be evaluated into a [`Window`].

mod functions;

use crate::sample::{cast, Sample};
use alloc::vec::Vec;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Whether a window is *symmetric* (for filter design) or *periodic* (a.k.a.
/// "DFT-even", for spectral analysis).
///
/// A periodic window of length `N` equals the symmetric window of length
/// `N + 1` with its last sample removed, matching NumPy/SciPy `fftbins=True`
/// and librosa. [`Symmetry::Periodic`] is the default because it is the right
/// choice for STFT spectral analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Symmetry {
    /// DFT-even window, the correct choice for spectral analysis.
    #[default]
    Periodic,
    /// Symmetric window, the correct choice for FIR filter design.
    Symmetric,
}

/// A parametric window family that can be evaluated into a [`Window`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum WindowFunction {
    /// Rectangular (boxcar) window: all ones.
    Rectangular,
    /// Hann window.
    Hann,
    /// Hamming window.
    Hamming,
    /// Blackman window.
    Blackman,
    /// 4-term Blackman-Harris window.
    BlackmanHarris,
    /// 4-term Nuttall window.
    Nuttall,
    /// 5-term flat-top window (excellent amplitude accuracy).
    FlatTop,
    /// Bartlett (triangular, zero endpoints) window.
    Bartlett,
    /// Triangular window without zero endpoints.
    Triangular,
    /// Welch (parabolic) window.
    Welch,
    /// Cosine (sine) window.
    Cosine,
    /// Tukey (tapered cosine) window; `alpha` is the taper fraction in `0..=1`.
    Tukey {
        /// Fraction of the window inside the cosine tapers (`0` = rectangular,
        /// `1` = Hann).
        alpha: f64,
    },
    /// Kaiser window with shape parameter `beta`.
    Kaiser {
        /// Shape parameter; larger values trade main-lobe width for side-lobe
        /// attenuation.
        beta: f64,
    },
    /// Gaussian window with standard deviation `std` (in samples).
    Gaussian {
        /// Standard deviation in samples.
        std: f64,
    },
}

impl WindowFunction {
    /// Evaluate this window family into raw `f64` coefficients of length `len`
    /// with the given symmetry.
    fn coefficients(self, len: usize, symmetry: Symmetry) -> Vec<f64> {
        if len == 0 {
            return Vec::new();
        }
        if len == 1 {
            return alloc::vec![1.0];
        }
        // Build a symmetric window of length `m` (= len for symmetric, len + 1
        // for periodic) and truncate to `len`.
        let m = match symmetry {
            Symmetry::Symmetric => len,
            Symmetry::Periodic => len + 1,
        };
        let mut coeffs = match self {
            Self::Rectangular => functions::rectangular(m),
            Self::Hann => functions::hann(m),
            Self::Hamming => functions::hamming(m),
            Self::Blackman => functions::blackman(m),
            Self::BlackmanHarris => functions::blackman_harris(m),
            Self::Nuttall => functions::nuttall(m),
            Self::FlatTop => functions::flat_top(m),
            Self::Bartlett => functions::bartlett(m),
            Self::Triangular => functions::triangular(m),
            Self::Welch => functions::welch(m),
            Self::Cosine => functions::cosine(m),
            Self::Tukey { alpha } => functions::tukey(m, alpha),
            Self::Kaiser { beta } => functions::kaiser(m, beta),
            Self::Gaussian { std } => functions::gaussian(m, std),
        };
        coeffs.truncate(len);
        coeffs
    }

    /// Evaluate this window family into a typed [`Window`].
    #[must_use]
    pub fn generate<T: Sample>(self, len: usize, symmetry: Symmetry) -> Window<T> {
        let coeffs = self
            .coefficients(len, symmetry)
            .into_iter()
            .map(cast)
            .collect();
        Window::from_coefficients(coeffs)
    }
}

/// A window: its coefficients plus cached `sum` and `sum_of_squares`.
#[derive(Debug, Clone, PartialEq)]
pub struct Window<T> {
    coeffs: Vec<T>,
    sum: T,
    sum_sq: T,
}

impl<T: Sample> Window<T> {
    /// Build a window from explicit coefficients, computing the cached sums.
    #[must_use]
    pub fn from_coefficients(coeffs: Vec<T>) -> Self {
        let mut sum = T::zero();
        let mut sum_sq = T::zero();
        for &c in &coeffs {
            sum = sum + c;
            sum_sq = sum_sq + c * c;
        }
        Self {
            coeffs,
            sum,
            sum_sq,
        }
    }

    /// Build a window from a [`WindowFunction`] family.
    #[must_use]
    pub fn new(function: WindowFunction, len: usize, symmetry: Symmetry) -> Self {
        function.generate(len, symmetry)
    }

    /// The window coefficients.
    #[must_use]
    pub fn coefficients(&self) -> &[T] {
        &self.coeffs
    }

    /// Number of coefficients (the frame length).
    #[must_use]
    pub fn len(&self) -> usize {
        self.coeffs.len()
    }

    /// Whether the window is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.coeffs.is_empty()
    }

    /// Sum of the coefficients (`Σ wᵢ`). Used by magnitude scaling.
    #[must_use]
    pub fn sum(&self) -> T {
        self.sum
    }

    /// Sum of the squared coefficients (`Σ wᵢ²`). Used by density scaling and
    /// overlap-add normalization.
    #[must_use]
    pub fn sum_squared(&self) -> T {
        self.sum_sq
    }
}

/// Generate a shorthand constructor for a parameter-free window family.
macro_rules! window_ctor {
    ($name:ident, $variant:ident, $doc:literal) => {
        #[doc = $doc]
        #[doc = ""]
        #[doc = "Uses [`Symmetry::Periodic`]; use [`Window::new`] for symmetric windows."]
        #[must_use]
        pub fn $name(len: usize) -> Self {
            WindowFunction::$variant.generate(len, Symmetry::Periodic)
        }
    };
}

impl<T: Sample> Window<T> {
    window_ctor!(rectangular, Rectangular, "A rectangular (boxcar) window.");
    window_ctor!(hann, Hann, "A periodic Hann window.");
    window_ctor!(hamming, Hamming, "A periodic Hamming window.");
    window_ctor!(blackman, Blackman, "A periodic Blackman window.");
    window_ctor!(
        blackman_harris,
        BlackmanHarris,
        "A periodic Blackman-Harris window."
    );
    window_ctor!(nuttall, Nuttall, "A periodic Nuttall window.");
    window_ctor!(flat_top, FlatTop, "A periodic flat-top window.");
    window_ctor!(bartlett, Bartlett, "A Bartlett window (zero endpoints).");
    window_ctor!(triangular, Triangular, "A triangular window.");
    window_ctor!(welch, Welch, "A Welch (parabolic) window.");
    window_ctor!(cosine, Cosine, "A cosine (sine) window.");

    /// A periodic Tukey (tapered cosine) window with taper fraction `alpha`.
    #[must_use]
    pub fn tukey(len: usize, alpha: f64) -> Self {
        WindowFunction::Tukey { alpha }.generate(len, Symmetry::Periodic)
    }

    /// A periodic Kaiser window with shape parameter `beta`.
    #[must_use]
    pub fn kaiser(len: usize, beta: f64) -> Self {
        WindowFunction::Kaiser { beta }.generate(len, Symmetry::Periodic)
    }

    /// A periodic Gaussian window with standard deviation `std` (in samples).
    #[must_use]
    pub fn gaussian(len: usize, std: f64) -> Self {
        WindowFunction::Gaussian { std }.generate(len, Symmetry::Periodic)
    }
}
