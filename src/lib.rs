//! # ruststft
//!
//! A complete short-time Fourier transform (STFT) toolkit for Rust:
//!
//! - **Forward STFT** over real-valued signals, in both *streaming* and *batch*
//!   modes, backed by [`realfft`] (≈2× faster and half the memory of a full
//!   complex FFT on real input).
//! - **Inverse STFT** with weighted overlap-add (WOLA) for perfect
//!   reconstruction.
//! - A **rich window library** (Hann, Hamming, Blackman/-Harris, Nuttall,
//!   Bartlett, triangular, Welch, cosine, Tukey, Kaiser, Gaussian, flat-top)
//!   with both periodic (spectral-analysis) and symmetric (filter-design)
//!   variants.
//! - **Spectrum helpers**: magnitude, power, phase and decibel conversions.
//! - Optional **mel** spectrograms and **MFCC**s (`mel` feature).
//! - Optional `ndarray` I/O (`ndarray`), parallel batch processing
//!   (`rayon`) and configuration (de)serialization (`serde`).
//!
//! ## Quick start (batch)
//!
//! ```
//! # #[cfg(feature = "std")] {
//! use ruststft::{Stft, Window};
//!
//! // A 1 kHz tone sampled at 8 kHz.
//! let fs = 8_000.0;
//! let signal: Vec<f64> = (0..8_000)
//!     .map(|n| (2.0 * std::f64::consts::PI * 1_000.0 * n as f64 / fs).sin())
//!     .collect();
//!
//! let mut stft = Stft::builder()
//!     .window(Window::<f64>::hann(1024))
//!     .hop_size(256)
//!     .build()
//!     .unwrap();
//!
//! let spec = stft.spectrogram(&signal);
//! assert_eq!(spec.n_freqs(), 1024 / 2 + 1); // includes the Nyquist bin
//! # }
//! ```
//!
//! ## Streaming
//!
//! ```
//! # #[cfg(feature = "std")] {
//! use ruststft::{Stft, Window};
//!
//! let mut stft = Stft::builder()
//!     .window(Window::<f32>::hann(1024))
//!     .hop_size(512)
//!     .build()
//!     .unwrap();
//!
//! let mut column = vec![num_complex::Complex::new(0.0f32, 0.0); stft.n_freqs()];
//! let chunk: Vec<f32> = (0..3000).map(|x| x as f32).collect();
//!
//! stft.append(&chunk);
//! while stft.ready() {
//!     stft.process_into(&mut column).unwrap();
//!     // ... use `column` ...
//!     stft.step();
//! }
//! # }
//! ```
//!
//! ## `no_std`
//!
//! The crate is `#![no_std]` (with `alloc`). The FFT-backed processors
//! ([`Stft`], [`Istft`], batch spectrograms) require the default `std`
//! feature because the underlying FFT backend needs `std`. The window
//! library, [`crate::mel`] filterbank/MFCC math and the
//! [`crate::spectrum`] helpers build without `std`.

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![forbid(unsafe_code)]
#![deny(missing_docs)]

extern crate alloc;

pub mod error;
pub mod sample;
pub mod spectrum;
pub mod window;

mod config;

#[cfg(feature = "mel")]
#[cfg_attr(docsrs, doc(cfg(feature = "mel")))]
pub mod mel;

#[cfg(feature = "std")]
mod batch;
#[cfg(feature = "std")]
mod istft;
#[cfg(feature = "std")]
mod stft;

pub use config::{PadMode, Scaling};
pub use error::StftError;
pub use sample::Sample;
pub use window::{Symmetry, Window, WindowFunction};

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
pub use batch::Spectrogram;
#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
pub use istft::{Istft, IstftBuilder};
#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
pub use stft::{Stft, StftBuilder};

// Re-export the complex number type so downstream users do not need to track
// the exact `num-complex` version themselves.
pub use num_complex::Complex;
