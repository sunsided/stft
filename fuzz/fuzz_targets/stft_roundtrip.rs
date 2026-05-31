#![no_main]
//! Forward STFT then inverse STFT must never panic on arbitrary input.

use libfuzzer_sys::fuzz_target;
use ruststft::{Stft, Window};

fuzz_target!(|data: &[u8]| {
    // Interpret the raw bytes as little-endian f32 samples, dropping
    // non-finite values so the transform sees a real signal.
    let samples: Vec<f32> = data
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .filter(|x| x.is_finite())
        .collect();
    if samples.len() < 256 {
        return;
    }

    let mut stft = match Stft::builder()
        .window(Window::<f32>::hann(256))
        .hop_size(64)
        .center(true)
        .build()
    {
        Ok(s) => s,
        Err(_) => return,
    };

    let spec = stft.spectrogram(&samples);
    if let Ok(istft) = stft.inverse() {
        let _ = istft.reconstruct(&spec);
    }
});
