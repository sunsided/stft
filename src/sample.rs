//! Numeric scalar abstraction shared by the whole crate.

use num_traits::{Float, FromPrimitive};

/// Scalar sample type supported by the crate.
///
/// This is a convenience trait alias implemented for every type that is both a
/// floating-point number ([`num_traits::Float`]) and constructible from
/// primitives ([`num_traits::FromPrimitive`]). In practice that means
/// [`f32`] and [`f64`].
///
/// The FFT-backed processors ([`Stft`](crate::Stft), [`Istft`](crate::Istft))
/// additionally require the backend's `FftNum` bound, which is also satisfied
/// by `f32`/`f64`.
pub trait Sample: Float + FromPrimitive + 'static {}

impl<T: Float + FromPrimitive + 'static> Sample for T {}

/// Convert an `f64` constant into the sample type `T`.
///
/// Window coefficients, mel scale conversions and decibel references are all
/// computed in `f64` and cast once. For `f32`/`f64` this conversion is
/// infallible, hence the `expect`.
#[inline]
pub(crate) fn cast<T: FromPrimitive>(value: f64) -> T {
    T::from_f64(value).expect("f32/f64 are always constructible from f64")
}
