//! Compute a magnitude spectrogram of a linear chirp and print it in decibels.
//!
//! Run with: `cargo run --example spectrogram`

use ruststft::spectrum::{magnitude, power_to_db};
use ruststft::{Stft, Window};

fn main() {
    let fs = 16_000.0;
    let n = fs as usize * 2; // 2 seconds

    // A linear chirp sweeping from 200 Hz to 4 kHz.
    let signal: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / fs;
            let f = 200.0 + (4000.0 - 200.0) * (t / 2.0);
            (std::f32::consts::PI * f * t).sin()
        })
        .collect();

    let mut stft = Stft::builder()
        .window(Window::<f32>::hann(1024))
        .hop_size(256)
        .center(true)
        .build()
        .expect("valid configuration");

    let spec = stft.spectrogram(&signal);
    println!(
        "spectrogram: {} frames x {} freqs",
        spec.n_frames(),
        spec.n_freqs()
    );

    let freqs = stft.freqs(fs as f64);
    // Report the dominant frequency in a handful of frames.
    for frame in (0..spec.n_frames()).step_by(spec.n_frames().max(1) / 8 + 1) {
        let mags = magnitude(spec.column(frame));
        let mut powers: Vec<f32> = mags.iter().map(|m| m * m).collect();
        power_to_db(&mut powers, 1.0, Some(80.0));
        let peak = mags
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(k, _)| k)
            .unwrap_or(0);
        println!(
            "frame {frame:>4}: peak ~ {:>6.0} Hz ({:>5.1} dB)",
            freqs[peak], powers[peak]
        );
    }
}
