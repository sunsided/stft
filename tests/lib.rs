//! Integration tests exercising the public API against analytic ground truth.

use approx::assert_abs_diff_eq;
use core::f64::consts::PI;
#[cfg(feature = "mel")]
use ruststft::mel::{hz_to_mel, mel_to_hz, DctII, MelFilterBank, MelScale};
use ruststft::spectrum::{amplitude_to_db, magnitude};
use ruststft::{Complex, Scaling, Stft, Symmetry, Window, WindowFunction};

// ---------------------------------------------------------------------------
// Windows
// ---------------------------------------------------------------------------

#[test]
fn periodic_hann_matches_truncated_symmetric() {
    let w = Window::<f64>::hann(4);
    // Periodic Hann(4) == symmetric Hann(5) without its last sample.
    let expected = [0.0, 0.5, 1.0, 0.5];
    for (got, want) in w.coefficients().iter().zip(expected) {
        assert_abs_diff_eq!(*got, want, epsilon = 1e-12);
    }
}

#[test]
fn symmetric_window_is_symmetric() {
    let w = Window::<f64>::new(WindowFunction::Hann, 9, Symmetry::Symmetric);
    let c = w.coefficients();
    for i in 0..c.len() {
        assert_abs_diff_eq!(c[i], c[c.len() - 1 - i], epsilon = 1e-12);
    }
    assert_abs_diff_eq!(c[0], 0.0, epsilon = 1e-12);
}

#[test]
fn rectangular_window_sums() {
    let w = Window::<f64>::rectangular(16);
    assert_abs_diff_eq!(w.sum(), 16.0, epsilon = 1e-12);
    assert_abs_diff_eq!(w.sum_squared(), 16.0, epsilon = 1e-12);
}

// ---------------------------------------------------------------------------
// Forward STFT correctness
// ---------------------------------------------------------------------------

#[test]
fn nyquist_bin_is_included() {
    let stft = Stft::builder()
        .window(Window::<f64>::hann(1024))
        .hop_size(256)
        .build()
        .unwrap();
    assert_eq!(stft.n_freqs(), 1024 / 2 + 1);
}

#[test]
fn frequencies_are_correct() {
    let fft_size = 1024usize;
    let fs = 8_000.0;
    let stft = Stft::builder()
        .window(Window::<f64>::rectangular(fft_size))
        .hop_size(256)
        .build()
        .unwrap();
    let freqs = stft.freqs(fs);
    assert_eq!(freqs.len(), fft_size / 2 + 1);
    assert_abs_diff_eq!(freqs[0], 0.0, epsilon = 1e-9);
    // Last bin is exactly Nyquist.
    assert_abs_diff_eq!(*freqs.last().unwrap(), fs / 2.0, epsilon = 1e-9);
    // Arbitrary bin k -> k * fs / fft_size.
    assert_abs_diff_eq!(freqs[10], 10.0 * fs / fft_size as f64, epsilon = 1e-9);
}

#[test]
fn pure_tone_peaks_at_expected_bin() {
    let n = 1024usize;
    let fs = 1024.0;
    let k0 = 64usize; // exact bin
    let signal: Vec<f64> = (0..n)
        .map(|i| (2.0 * PI * k0 as f64 * i as f64 / n as f64).cos())
        .collect();

    let mut stft = Stft::builder()
        .window(Window::<f64>::rectangular(n))
        .hop_size(n)
        .build()
        .unwrap();
    let spec = stft.spectrogram(&signal);
    assert_eq!(spec.n_frames(), 1);

    let mags = magnitude(spec.column(0));
    let argmax = mags
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    assert_eq!(argmax, k0);
    assert_eq!(spec.column(0).len(), fs as usize / 2 + 1);
}

#[test]
fn magnitude_scaling_recovers_sine_amplitude() {
    let n = 2048usize;
    let amplitude = 0.7;
    let k0 = 100usize;
    let signal: Vec<f64> = (0..n)
        .map(|i| amplitude * (2.0 * PI * k0 as f64 * i as f64 / n as f64).cos())
        .collect();

    let mut stft = Stft::builder()
        .window(Window::<f64>::rectangular(n))
        .hop_size(n)
        .scaling(Scaling::Magnitude)
        .build()
        .unwrap();
    let spec = stft.spectrogram(&signal);
    let mags = magnitude(spec.column(0));
    // One-sided magnitude of a real cosine at an interior bin is amplitude/2.
    assert_abs_diff_eq!(mags[k0], amplitude / 2.0, epsilon = 1e-6);
}

#[test]
fn streaming_and_batch_agree() {
    let n = 4096usize;
    let signal: Vec<f64> = (0..n).map(|i| (i as f64 * 0.01).sin()).collect();

    let mut batch = Stft::builder()
        .window(Window::<f64>::hann(256))
        .hop_size(64)
        .build()
        .unwrap();
    let spec = batch.spectrogram(&signal);

    let mut stream = Stft::builder()
        .window(Window::<f64>::hann(256))
        .hop_size(64)
        .build()
        .unwrap();
    stream.append(&signal);
    let columns: Vec<Vec<Complex<f64>>> = stream.columns().collect();

    assert_eq!(columns.len(), spec.n_frames());
    for (frame, batch_col) in columns.iter().zip(spec.columns()) {
        for (a, b) in frame.iter().zip(batch_col) {
            assert_abs_diff_eq!(a.re, b.re, epsilon = 1e-9);
            assert_abs_diff_eq!(a.im, b.im, epsilon = 1e-9);
        }
    }
}

// ---------------------------------------------------------------------------
// Round-trip reconstruction (STFT -> ISTFT)
// ---------------------------------------------------------------------------

#[test]
fn round_trip_reconstructs_signal() {
    let n = 8000usize;
    let fs = 8000.0;
    let signal: Vec<f64> = (0..n)
        .map(|i| {
            (2.0 * PI * 220.0 * i as f64 / fs).sin()
                + 0.3 * (2.0 * PI * 600.0 * i as f64 / fs).sin()
        })
        .collect();

    let mut stft = Stft::builder()
        .window(Window::<f64>::hann(1024))
        .hop_size(256) // 75% overlap: Hann is COLA-compliant
        .center(true)
        .build()
        .unwrap();
    let spec = stft.spectrogram(&signal);

    let istft = stft.inverse().unwrap();
    let recon = istft.reconstruct(&spec).unwrap();

    // Compare the interior, away from edge-taper artifacts.
    let lo = 1024;
    let hi = n - 1024;
    for i in lo..hi {
        assert_abs_diff_eq!(recon[i], signal[i], epsilon = 1e-6);
    }
}

#[test]
fn round_trip_rectangular_no_overlap_is_exact_interior() {
    let n = 4096usize;
    let signal: Vec<f64> = (0..n).map(|i| ((i * 7 % 13) as f64) - 6.0).collect();

    let mut stft = Stft::builder()
        .window(Window::<f64>::rectangular(512))
        .hop_size(512) // contiguous, non-overlapping frames
        .build()
        .unwrap();
    let spec = stft.spectrogram(&signal);
    let istft = stft.inverse().unwrap();
    let recon = istft.reconstruct(&spec).unwrap();

    for i in 0..recon.len() {
        assert_abs_diff_eq!(recon[i], signal[i], epsilon = 1e-9);
    }
}

// ---------------------------------------------------------------------------
// Spectrum helpers
// ---------------------------------------------------------------------------

#[test]
fn amplitude_to_db_floor_and_reference() {
    let mut v = vec![1.0f64, 0.1, 0.01, 0.0];
    amplitude_to_db(&mut v, 1.0, Some(80.0));
    assert_abs_diff_eq!(v[0], 0.0, epsilon = 1e-9); // 20*log10(1) = 0
    assert_abs_diff_eq!(v[1], -20.0, epsilon = 1e-9); // 20*log10(0.1)
                                                      // Floored at max - 80 = -80.
    assert_abs_diff_eq!(v[3], -80.0, epsilon = 1e-9);
}

// ---------------------------------------------------------------------------
// Mel + MFCC
// ---------------------------------------------------------------------------

#[cfg(feature = "mel")]
#[test]
fn mel_scale_round_trips_and_is_monotonic() {
    for scale in [MelScale::Slaney, MelScale::Htk] {
        assert_abs_diff_eq!(hz_to_mel(0.0, scale), 0.0, epsilon = 1e-9);
        for f in [100.0, 440.0, 1000.0, 4000.0, 8000.0] {
            assert_abs_diff_eq!(mel_to_hz(hz_to_mel(f, scale), scale), f, epsilon = 1e-6);
        }
        assert!(hz_to_mel(2000.0, scale) > hz_to_mel(1000.0, scale));
    }
}

#[cfg(feature = "mel")]
#[test]
fn mel_filterbank_shape_and_nonnegativity() {
    let bank = MelFilterBank::<f64>::new(40, 1024, 16_000.0, 0.0, 8_000.0, MelScale::Slaney);
    assert_eq!(bank.n_mels(), 40);
    assert_eq!(bank.n_freqs(), 1024 / 2 + 1);
    assert!(bank.weights().iter().all(|&w| w >= 0.0));

    // Applying to a flat power spectrum yields positive energy in every band.
    let power = vec![1.0f64; bank.n_freqs()];
    let mel = bank.transform(&power);
    assert_eq!(mel.len(), 40);
    assert!(mel.iter().all(|&m| m > 0.0));
}

#[cfg(feature = "mel")]
#[test]
fn dct2_of_constant_is_a_single_coefficient() {
    let n = 32usize;
    let dct = DctII::<f64>::new(n, n);
    let x = vec![1.0f64; n];
    let y = dct.transform(&x);
    assert_abs_diff_eq!(y[0], (n as f64).sqrt(), epsilon = 1e-9);
    for &c in &y[1..] {
        assert_abs_diff_eq!(c, 0.0, epsilon = 1e-9);
    }
}
