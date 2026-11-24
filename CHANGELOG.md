# Changelog

All notable changes to this project are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `Istft::drain_ready()` streams reconstructed samples out between columns, so
  long-running inverse transforms use bounded memory.
- `Istft::reconstruct_with_len()` returns exactly the requested number of
  samples. `reconstruct()` can be up to `hop - 1` samples shorter than the
  analyzed signal.
- `StftError::InvalidWindow` and `StftError::InvalidScale`.

### Changed

- `amplitude_to_db` clamps amplitudes at `1e-5` (was `1e-10`), so silence maps
  to `-100 dB` like librosa's `amplitude_to_db`.
- `MelFilterBank::new` panics on `n_fft == 0`, a non-positive sample rate, or
  band edges outside `0 <= fmin < fmax <= sample_rate / 2`.
- `MelFilterBank::transform` only visits the non-zero bins of each filter.
- `Spectrogram::from_flat` panics on `n_freqs == 0`, instead of a later panic
  in `columns()`.

### Fixed

- `Stft` and `Istft` builders reject windows with NaN or infinite
  coefficients, and scale factors that are not finite and positive (zero-sum
  window with `Scaling::Magnitude`, a sample rate of `0` with
  `Scaling::Density`, a forward scale of `0`). These used to produce NaN
  spectrograms silently.
- A NaN Tukey `alpha` gives a rectangular window instead of NaN coefficients.

## [0.4.0]

This is a ground-up redesign with a clean, encapsulated API. The old `STFT`
type, `WindowType` enum, `FromF64` trait and `log10_positive` free function
have been removed.

### Added

- `Istft` / `IstftBuilder`: inverse STFT with weighted overlap-add for perfect
  reconstruction, plus `Stft::inverse()` to mirror a forward transform.
- One-shot batch processing: `Stft::spectrogram()` returning a `Spectrogram`,
  with optional centered framing (`reflect`/`edge`/`zero` padding) and optional
  `rayon` parallelism.
- A full window library (`Window`, `WindowFunction`, `Symmetry`): rectangular,
  Hann, Hamming, Blackman, Blackman-Harris, Nuttall, flat-top, Bartlett,
  triangular, Welch, cosine, Tukey, Kaiser and Gaussian, in periodic and
  symmetric variants.
- Coefficient scaling modes (`Scaling`: none, magnitude, density).
- `spectrum` helpers: magnitude, power, phase, and decibel conversions.
- `mel` feature: mel filterbank, mel scale conversions, and an orthonormal
  DCT-II for MFCCs (librosa-compatible defaults).
- Optional integrations: `ndarray` (`Array2` output), `rayon` (parallel batch),
  `serde` (config (de)serialization).
- `no_std` support (with `alloc`) for the window, spectrum and mel math.
- `#![forbid(unsafe_code)]` across the crate.

### Changed

- Switched the FFT backend to `realfft`, roughly halving time and memory for
  real-valued input.

### Fixed

- The number of frequency bins is now `fft_size / 2 + 1`, correctly including
  the Nyquist bin (previously `fft_size / 2`, which dropped it).
- Bin center frequencies are now `k · fs / fft_size` (previously
  `k · fs / (2·(n_freqs − 1))`, which was off).

[Unreleased]: https://github.com/sunsided/stft/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/sunsided/stft/compare/v0.3.1...v0.4.0
