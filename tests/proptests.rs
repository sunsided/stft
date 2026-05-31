//! Property-based tests: STFT linearity and STFT/ISTFT round-trip fidelity.

use proptest::prelude::*;
use ruststft::{Stft, Window};

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
