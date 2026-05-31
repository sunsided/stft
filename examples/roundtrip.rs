//! Demonstrate perfect reconstruction: signal -> STFT -> ISTFT -> signal.
//!
//! Run with: `cargo run --example roundtrip`

use ruststft::{Stft, Window};

fn main() {
    let fs = 8_000.0;
    let n = 8_000usize;
    let signal: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 / fs;
            (2.0 * std::f64::consts::PI * 220.0 * t).sin()
                + 0.3 * (2.0 * std::f64::consts::PI * 880.0 * t).sin()
        })
        .collect();

    let mut stft = Stft::builder()
        .window(Window::<f64>::hann(1024))
        .hop_size(256) // 75% overlap -> Hann is COLA compliant
        .center(true)
        .build()
        .expect("valid configuration");

    let spec = stft.spectrogram(&signal);
    let recon = stft
        .inverse()
        .expect("invertible")
        .reconstruct(&spec)
        .expect("reconstruction");

    // Maximum reconstruction error over the interior (edges taper off).
    let frame = 1024;
    let max_err = (frame..n - frame)
        .map(|i| (recon[i] - signal[i]).abs())
        .fold(0.0_f64, f64::max);

    println!("frames: {}", spec.n_frames());
    println!("interior max reconstruction error: {max_err:.3e}");
    assert!(max_err < 1e-6, "reconstruction should be near-perfect");
    println!("reconstruction is near-perfect ✔");
}
