//! Property-based tests: STFT linearity and STFT/ISTFT round-trip fidelity.

use proptest::prelude::*;
use ruststft::{Complex, PadMode, Scaling, Spectrogram, Stft, Symmetry, Window, WindowFunction};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    /// The STFT is linear: `STFT(a·x + b·y) == a·STFT(x) + b·STFT(y)`.
    #[test]
    fn stft_is_linear(
        x in prop::collection::vec(-1.0f64..1.0, 1024..2048),
        a in -3.0f64..3.0,
        b in -3.0f64..3.0,
    ) {
        let n = x.len();
        // A deterministic second signal correlated with the first.
        let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.013).sin()).collect();
        let z: Vec<f64> = (0..n).map(|i| a * x[i] + b * y[i]).collect();

        let build = || Stft::builder()
            .window(Window::<f64>::hann(256))
            .hop_size(128)
            .build()
            .unwrap();

        let sx = build().spectrogram(&x);
        let sy = build().spectrogram(&y);
        let sz = build().spectrogram(&z);

        prop_assert_eq!(sx.n_frames(), sz.n_frames());
        for f in 0..sz.n_frames() {
            for (k, zc) in sz.column(f).iter().enumerate() {
                let lhs = *zc;
                let rhs = sx.column(f)[k] * a + sy.column(f)[k] * b;
                prop_assert!((lhs.re - rhs.re).abs() < 1e-6);
                prop_assert!((lhs.im - rhs.im).abs() < 1e-6);
            }
        }
    }

    /// A Hann window at 75% overlap is COLA-compliant, so STFT->ISTFT
    /// reconstructs the interior of the signal.
    #[test]
    fn round_trip_reconstructs_interior(
        signal in prop::collection::vec(-1.0f64..1.0, 4096..6000),
    ) {
        let mut stft = Stft::builder()
            .window(Window::<f64>::hann(512))
            .hop_size(128)
            .center(true)
            .build()
            .unwrap();
        let spec = stft.spectrogram(&signal);
        let recon = stft.inverse().unwrap().reconstruct(&spec).unwrap();

        let frame = 512;
        let hi = signal.len().min(recon.len());
        for i in frame..(hi - frame) {
            prop_assert!((recon[i] - signal[i]).abs() < 1e-6);
        }
    }
}

/// Every window family that the properties below draw from.
fn any_window_function() -> impl Strategy<Value = WindowFunction> {
    prop_oneof![
        Just(WindowFunction::Rectangular),
        Just(WindowFunction::Hann),
        Just(WindowFunction::Hamming),
        Just(WindowFunction::Blackman),
        Just(WindowFunction::BlackmanHarris),
        Just(WindowFunction::Nuttall),
        Just(WindowFunction::FlatTop),
        Just(WindowFunction::Bartlett),
        Just(WindowFunction::Triangular),
        Just(WindowFunction::Welch),
        Just(WindowFunction::Cosine),
        (0.0f64..=1.0).prop_map(|alpha| WindowFunction::Tukey { alpha }),
        (0.0f64..12.0).prop_map(|beta| WindowFunction::Kaiser { beta }),
        (0.5f64..8.0).prop_map(|std| WindowFunction::Gaussian { std }),
    ]
}

/// Any padding mode.
fn any_pad_mode() -> impl Strategy<Value = PadMode> {
    prop_oneof![
        Just(PadMode::Zero),
        Just(PadMode::Reflect),
        Just(PadMode::Edge)
    ]
}

/// Pad `signal` by `pad` samples on each side, as NumPy `np.pad` does.
///
/// The oracle needs `pad < signal.len()`, so that one reflection is enough.
fn numpy_pad(signal: &[f64], pad: usize, mode: PadMode) -> Vec<f64> {
    let len = signal.len();
    let left = (0..pad).map(|i| match mode {
        PadMode::Zero => 0.0,
        PadMode::Edge => signal[0],
        PadMode::Reflect => signal[pad - i],
    });
    let right = (0..pad).map(|j| match mode {
        PadMode::Zero => 0.0,
        PadMode::Edge => signal[len - 1],
        PadMode::Reflect => signal[len - 2 - j],
    });

    left.chain(signal.iter().copied()).chain(right).collect()
}

/// Assert that two spectrograms are equal to within `tol`, column by column.
fn assert_spec_close(
    a: &[Vec<Complex<f64>>],
    b: &Spectrogram<f64>,
    tol: f64,
) -> Result<(), TestCaseError> {
    prop_assert_eq!(a.len(), b.n_frames());
    for (col_a, col_b) in a.iter().zip(b.columns()) {
        prop_assert_eq!(col_a.len(), col_b.len());
        for (x, y) in col_a.iter().zip(col_b) {
            prop_assert!((x - y).norm() < tol, "{x} != {y}");
        }
    }

    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// A periodic window of length `n` is the symmetric window of length
    /// `n + 1` without its last sample.
    #[test]
    fn periodic_is_truncated_symmetric(
        function in any_window_function(),
        len in 2usize..40,
    ) {
        let periodic = Window::<f64>::new(function, len, Symmetry::Periodic);
        let symmetric = Window::<f64>::new(function, len + 1, Symmetry::Symmetric);

        prop_assert_eq!(periodic.len(), len);
        for (p, s) in periodic.coefficients().iter().zip(symmetric.coefficients()) {
            prop_assert!((p - s).abs() < 1e-12);
        }
    }

    /// A symmetric window is a palindrome.
    #[test]
    fn symmetric_window_is_palindrome(
        function in any_window_function(),
        len in 1usize..40,
    ) {
        let window = Window::<f64>::new(function, len, Symmetry::Symmetric);
        let c = window.coefficients();
        for i in 0..len {
            prop_assert!((c[i] - c[len - 1 - i]).abs() < 1e-12);
        }
    }

    /// Streaming and batch processing give the same columns for any frame,
    /// hop, FFT size and scaling. The frame count is `1 + (len - frame) / hop`.
    #[test]
    fn streaming_matches_batch(
        function in any_window_function(),
        frame in 2usize..48,
        hop_frac in 0.0f64..1.0,
        extra_fft in 0usize..9,
        scaling in prop_oneof![Just(Scaling::None), Just(Scaling::Magnitude), Just(Scaling::Density)],
        signal in prop::collection::vec(-1.0f64..1.0, 0..160),
    ) {
        let hop = 1 + (hop_frac * (frame - 1) as f64) as usize;
        let build = || Stft::builder()
            .window(Window::<f64>::new(function, frame, Symmetry::Periodic))
            .hop_size(hop)
            .fft_size(frame + extra_fft)
            .scaling(scaling)
            .sample_rate(1000.0)
            .build()
            .unwrap();

        let spec = build().spectrogram(&signal);
        let mut stream = build();
        stream.append(&signal);
        let columns: Vec<_> = stream.columns().collect();

        let expected_frames = if signal.len() >= frame { 1 + (signal.len() - frame) / hop } else { 0 };
        prop_assert_eq!(spec.n_frames(), expected_frames);
        prop_assert_eq!(spec.n_freqs(), (frame + extra_fft) / 2 + 1);
        prop_assert!(stream.buffered() < frame);
        assert_spec_close(&columns, &spec, 1e-9)?;
    }

    /// Centered framing equals uncentered framing on a NumPy-padded signal.
    #[test]
    fn center_pads_like_numpy(
        frame in 2usize..32,
        hop in 1usize..8,
        mode in any_pad_mode(),
        signal in prop::collection::vec(-1.0f64..1.0, 33..120),
    ) {
        let hop = hop.min(frame);
        let build = |center| Stft::builder()
            .window(Window::<f64>::hann(frame))
            .hop_size(hop)
            .center(center)
            .pad_mode(mode)
            .build()
            .unwrap();

        let centered = build(true).spectrogram(&signal);
        let padded = numpy_pad(&signal, frame / 2, mode);
        let manual = build(false).spectrogram(&padded);

        let manual_cols: Vec<_> = manual.columns().map(<[_]>::to_vec).collect();
        assert_spec_close(&manual_cols, &centered, 1e-12)?;
    }

    /// STFT followed by ISTFT reconstructs the interior for any hop up to half
    /// the frame, any FFT zero padding and any scaling mode.
    #[test]
    fn round_trip_any_config(
        frame in 4usize..48,
        hop_frac in 0.0f64..1.0,
        extra_fft in 0usize..9,
        scaling in prop_oneof![Just(Scaling::None), Just(Scaling::Magnitude), Just(Scaling::Density)],
        center in any::<bool>(),
        signal in prop::collection::vec(-1.0f64..1.0, 100..200),
    ) {
        let hop = 1 + (hop_frac * (frame / 2 - 1) as f64) as usize;
        let mut stft = Stft::builder()
            .window(Window::<f64>::hann(frame))
            .hop_size(hop)
            .fft_size(frame + extra_fft)
            .scaling(scaling)
            .sample_rate(8000.0)
            .center(center)
            .build()
            .unwrap();
        let spec = stft.spectrogram(&signal);
        let istft = stft.inverse().unwrap();
        prop_assert_eq!(istft.n_freqs(), spec.n_freqs());
        let recon = istft.reconstruct(&spec).unwrap();

        // The output covers whole frames only. Centered output also has
        // `frame / 2` samples of padding trimmed from each end, so it starts at
        // sample 0 and can be shorter than the signal.
        let span = (spec.n_frames() - 1) * hop + frame;
        let covered = if center { span - 2 * (frame / 2) } else { span };
        prop_assert_eq!(recon.len(), covered);
        for i in frame..covered.saturating_sub(frame) {
            prop_assert!((recon[i] - signal[i]).abs() < 1e-9, "sample {i}: {} vs {}", recon[i], signal[i]);
        }
    }
}
