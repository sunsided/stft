//! Compute MFCCs from a tone, mirroring librosa's pipeline:
//! power spectrogram -> mel filterbank -> dB -> DCT-II.
//!
//! Run with: `cargo run --example mfcc --features mel`

use ruststft::mel::{DctII, MelFilterBank, MelScale};
use ruststft::spectrum::{power, power_to_db};
use ruststft::{Stft, Window};

fn main() {
    let fs = 16_000.0;
    let n_fft = 1024usize;
    let n_mels = 40usize;
    let n_mfcc = 13usize;

    let signal: Vec<f64> = (0..fs as usize)
        .map(|i| {
            let t = i as f64 / fs;
            (2.0 * std::f64::consts::PI * 440.0 * t).sin()
        })
        .collect();

    let mut stft = Stft::builder()
        .window(Window::<f64>::hann(n_fft))
        .hop_size(n_fft / 4)
        .center(true)
        .build()
        .expect("valid configuration");
    let spec = stft.spectrogram(&signal);

    let bank = MelFilterBank::<f64>::new(n_mels, n_fft, fs, 0.0, fs / 2.0, MelScale::Slaney);
    let dct = DctII::<f64>::new(n_mels, n_mfcc);

    // Transform the middle frame.
    let frame = spec.n_frames() / 2;
    let mut mel = bank.transform(&power(spec.column(frame)));
    power_to_db(&mut mel, 1.0, None);
    let mfcc = dct.transform(&mel);

    println!("MFCCs (frame {frame}):");
    for (i, c) in mfcc.iter().enumerate() {
        println!("  c{i:<2} = {c:+.4}");
    }
}
