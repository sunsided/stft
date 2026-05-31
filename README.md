# ruststft

[![CI](https://github.com/sunsided/stft/actions/workflows/ci.yml/badge.svg)](https://github.com/sunsided/stft/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/ruststft.svg)](https://crates.io/crates/ruststft)
[![docs.rs](https://img.shields.io/docsrs/ruststft)](https://docs.rs/ruststft)
[![unsafe forbidden](https://img.shields.io/badge/unsafe-forbidden-success.svg)](https://github.com/rust-secure-code/safety-dance/)
![license](https://img.shields.io/crates/l/ruststft.svg)

A complete [short-time Fourier transform](https://en.wikipedia.org/wiki/Short-time_Fourier_transform)
toolkit for Rust: forward **and** inverse STFT, a rich window library, batch and
streaming APIs, and optional mel spectrograms / MFCCs.

- **Forward STFT** over real signals, streaming or batch, backed by
  [`realfft`](https://crates.io/crates/realfft) (≈2× faster and half the memory
  of a full complex FFT on real input).
- **Inverse STFT** with weighted overlap-add (WOLA) for perfect reconstruction.
- **Windows**: rectangular, Hann, Hamming, Blackman, Blackman-Harris, Nuttall,
  flat-top, Bartlett, triangular, Welch, cosine, Tukey, Kaiser, Gaussian -
  periodic (spectral-analysis) or symmetric (filter-design).
- **Spectrum helpers**: magnitude, power, phase, and decibel conversions.
- **Mel & MFCC** (`mel` feature): librosa-compatible filterbank, mel scales and
  an orthonormal DCT-II.
- **`#![forbid(unsafe_code)]`** - 100% safe Rust.
- **`no_std`** (with `alloc`) for the window, spectrum and mel math.

## Install

```toml
[dependencies]
ruststft = "0.4"
```

## Batch spectrogram

```rust
use ruststft::{Stft, Window};

let fs = 8_000.0;
let signal: Vec<f64> = (0..8_000)
    .map(|n| (2.0 * std::f64::consts::PI * 1_000.0 * n as f64 / fs).sin())
    .collect();

let mut stft = Stft::builder()
    .window(Window::<f64>::hann(1024))
    .hop_size(256)
    .center(true)
    .build()
    .unwrap();

let spec = stft.spectrogram(&signal);
assert_eq!(spec.n_freqs(), 1024 / 2 + 1); // includes the Nyquist bin
```

## Perfect reconstruction (STFT → ISTFT)

```rust
use ruststft::{Stft, Window};

let signal: Vec<f64> = (0..8_000).map(|n| (n as f64 * 0.01).sin()).collect();

let mut stft = Stft::builder()
    .window(Window::<f64>::hann(1024))
    .hop_size(256)       // 75% overlap: Hann is COLA-compliant
    .center(true)
    .build()
    .unwrap();

let spec = stft.spectrogram(&signal);
let recon = stft.inverse().unwrap().reconstruct(&spec).unwrap();
// recon matches `signal` in the interior to ~machine precision.
```

## Streaming

```rust
use ruststft::{Complex, Stft, Window};

let mut stft = Stft::builder()
    .window(Window::<f32>::hann(1024))
    .hop_size(512)
    .build()
    .unwrap();

let mut column = vec![Complex::new(0.0f32, 0.0); stft.n_freqs()];
let chunk: Vec<f32> = (0..3000).map(|x| x as f32).collect();

stft.append(&chunk);
while stft.ready() {
    stft.process_into(&mut column).unwrap();
    // ... use `column` ...
    stft.step();
}
```

## Feature flags

| Feature   | Default | Description                                              |
|-----------|:-------:|----------------------------------------------------------|
| `std`     |   yes   | FFT-backed processors (`Stft`, `Istft`, batch). Required for the transforms. |
| `mel`     |   no    | Mel filterbank, mel scales, and DCT-II for MFCCs.        |
| `ndarray` |   no    | `Spectrogram::to_array2` (`[n_freqs, n_frames]`).        |
| `rayon`   |   no    | Parallel per-frame batch spectrograms.                   |
| `serde`   |   no    | (De)serialize configuration and window descriptions.     |
| `wasm_simd` | no    | WASM `simd128` FFT kernels (implies `std`; build with `-C target-feature=+simd128`). |

Without the default `std` feature the crate builds as `no_std` (with `alloc`),
exposing the window library, the [`spectrum`](https://docs.rs/ruststft/latest/ruststft/spectrum/)
helpers and the [`mel`](https://docs.rs/ruststft/latest/ruststft/mel/) math. The
FFT processors require `std` because the FFT backend does.

## Migrating from 0.3

`0.4` is a breaking redesign. Rough mapping:

| 0.3                                   | 0.4                                            |
|---------------------------------------|------------------------------------------------|
| `STFT::new(WindowType::Hanning, w, s)`| `Stft::builder().window(Window::hann(w)).hop_size(s).build()?` |
| `WindowType::Hanning` (etc.)          | `Window::hann(len)` / `WindowFunction::Hann`   |
| `stft.append_samples(x)`              | `stft.append(x)`                               |
| `stft.contains_enough_to_compute()`   | `stft.ready()`                                 |
| `stft.compute_complex_column(&mut c)` | `stft.process_into(&mut c)?`                   |
| `stft.move_to_next_column()`          | `stft.step()`                                  |
| `stft.output_size()` (= `fft/2`)      | `stft.n_freqs()` (= `fft/2 + 1`, fixes Nyquist)|
| `compute_magnitude_column` / `compute_column` | `spectrum::magnitude` / `power_to_db` on a column |
| (no inverse)                          | `stft.inverse()?.reconstruct(&spec)?`          |

See [`CHANGELOG.md`](CHANGELOG.md) for details.

## [Contributing](contributing.md)

Licensed under either of [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT) at
your option.
