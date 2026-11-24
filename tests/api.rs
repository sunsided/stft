//! Accessors, error display and builder validation of the public API.

use ruststft::{Complex, Istft, Scaling, Spectrogram, Stft, StftError, Window};

/// `Spectrogram` reports its shape and slices columns frame-major.
#[test]
fn spectrogram_shape_and_columns() {
    let data: Vec<Complex<f64>> = (0..6).map(|i| Complex::new(f64::from(i), 0.0)).collect();
    let spec = Spectrogram::from_flat(data, 3, 2);

    assert_eq!((spec.n_frames(), spec.n_freqs()), (3, 2));
    assert!(!spec.is_empty());
    assert_eq!(
        spec.column(2),
        &[Complex::new(4.0, 0.0), Complex::new(5.0, 0.0)]
    );
    assert_eq!(spec.as_flat().len(), 6);
    assert_eq!(spec.clone().into_flat(), spec.as_flat());
    assert!(Spectrogram::<f64>::from_flat(Vec::new(), 0, 2).is_empty());
}

/// Every error variant has a non-empty message that names its values.
#[test]
fn error_messages() {
    let cases = [
        (StftError::MissingWindow, "window"),
        (StftError::InvalidFrameLength, "frame length"),
        (
            StftError::InvalidHopSize {
                hop: 7,
                frame_len: 4,
            },
            "hop size 7",
        ),
        (
            StftError::InvalidFftSize {
                fft_size: 3,
                frame_len: 4,
            },
            "fft size 3",
        ),
        (
            StftError::LengthMismatch {
                expected: 5,
                got: 6,
            },
            "expected 5, got 6",
        ),
        (StftError::MissingSampleRate, "sample rate"),
        (
            StftError::NotEnoughData {
                needed: 8,
                available: 2,
            },
            "needed 8, have 2",
        ),
        (StftError::Fft, "FFT"),
    ];

    for (error, needle) in cases {
        let message = error.to_string();
        assert!(
            message.contains(needle),
            "{message:?} does not contain {needle:?}"
        );
    }
}

/// Both builders default the hop to `frame_len / 4`, at least 1, and reject a
/// hop of 0 or a hop longer than the frame.
#[test]
fn builders_default_and_validate_hop() {
    let stft = |frame: usize, hop: Option<usize>| {
        let builder = Stft::builder().window(Window::<f64>::hann(frame));
        match hop {
            Some(h) => builder.hop_size(h).build(),
            None => builder.build(),
        }
    };
    let istft = |hop: usize| {
        Istft::builder()
            .window(Window::<f64>::hann(8))
            .hop_size(hop)
            .build()
    };

    assert_eq!(stft(8, None).unwrap().hop(), 2);
    assert_eq!(stft(2, None).unwrap().hop(), 1);
    for hop in [0, 9] {
        let expected = StftError::InvalidHopSize { hop, frame_len: 8 };
        assert_eq!(stft(8, Some(hop)).err(), Some(expected.clone()));
        assert_eq!(istft(hop).err(), Some(expected));
    }

    assert!(istft(8).is_ok());
}

/// The default ISTFT hop is `frame_len / 4`, so it inverts a default STFT.
#[test]
fn istft_default_hop_inverts_default_stft() {
    let signal: Vec<f64> = (0..64).map(|i| (f64::from(i) * 0.3).sin()).collect();
    let mut stft = Stft::builder()
        .window(Window::<f64>::hann(16))
        .build()
        .unwrap();
    let spec = stft.spectrogram(&signal);

    let istft = Istft::builder()
        .window(Window::<f64>::hann(16))
        .build()
        .unwrap();
    let recon = istft.reconstruct(&spec).unwrap();

    assert_eq!(recon.len(), (spec.n_frames() - 1) * 4 + 16);
    for i in 16..48 {
        assert!((recon[i] - signal[i]).abs() < 1e-9, "sample {i}");
    }
}

/// Density scaling divides by `sqrt(fs · Σw²)`.
#[test]
fn density_scale_value() {
    let stft = Stft::builder()
        .window(Window::<f64>::rectangular(4))
        .scaling(Scaling::Density)
        .sample_rate(2.0)
        .build()
        .unwrap();

    assert!((stft.scale() - 1.0 / 8.0f64.sqrt()).abs() < 1e-15);
}

/// `process_into` checks the output length and the buffered sample count.
#[test]
fn process_into_errors() {
    let mut stft = Stft::builder()
        .window(Window::<f64>::hann(8))
        .hop_size(4)
        .build()
        .unwrap();
    let mut out = vec![Complex::new(0.0, 0.0); stft.n_freqs()];
    let mut short = vec![Complex::new(0.0, 0.0); stft.n_freqs() - 1];

    stft.append(&[1.0; 6]);
    assert_eq!(stft.buffered(), 6);
    assert_eq!(
        stft.process_into(&mut short),
        Err(StftError::LengthMismatch {
            expected: 5,
            got: 4
        })
    );
    assert_eq!(
        stft.process_into(&mut out),
        Err(StftError::NotEnoughData {
            needed: 8,
            available: 6
        })
    );

    stft.append(&[1.0; 2]);
    assert_eq!(stft.process_into(&mut out), Ok(()));
    assert!(
        (out[0].re - 4.0).abs() < 1e-12,
        "DC of a Hann(8) over ones is 4"
    );

    stft.reset();
    assert_eq!(stft.buffered(), 0);
    assert!(!stft.ready());
}

/// `Istft` counts frames and accepts imaginary parts in the DC and Nyquist
/// bins, for even and odd FFT sizes.
#[test]
fn istft_frames_and_real_bins() {
    for fft_size in [8, 9] {
        let mut istft = Istft::builder()
            .window(Window::<f64>::hann(8))
            .hop_size(4)
            .fft_size(fft_size)
            .build()
            .unwrap();
        let column = vec![Complex::new(1.0, 0.5); istft.n_freqs()];

        assert_eq!(istft.process_column(&column), Ok(()));
        assert_eq!(istft.process_column(&column), Ok(()));
        assert_eq!(istft.frames(), 2);
        assert_eq!(istft.finish().len(), 12);
    }
}

/// `to_array2` transposes to the librosa `[n_freqs, n_frames]` layout.
#[cfg(feature = "ndarray")]
#[test]
fn spectrogram_to_array2_layout() {
    let data: Vec<Complex<f64>> = (0..6).map(|i| Complex::new(f64::from(i), 0.0)).collect();
    let array = Spectrogram::from_flat(data, 3, 2).to_array2();

    assert_eq!(array.dim(), (2, 3));
    assert_eq!(array[[1, 2]], Complex::new(5.0, 0.0));
    assert_eq!(array[[0, 1]], Complex::new(2.0, 0.0));
}

/// Centered `finish` trims `frame_len / 2` samples from each end, and returns
/// an empty signal when there is less output than padding.
#[test]
fn istft_center_trim_edges() {
    let build = |frame: usize| {
        Istft::builder()
            .window(Window::<f64>::rectangular(frame))
            .hop_size(frame)
            .center(true)
            .build()
            .unwrap()
    };

    assert!(build(7).finish().is_empty());

    let mut istft = build(2);
    istft.process_column(&[Complex::new(2.0, 0.0); 2]).unwrap();
    assert!(istft.finish().is_empty());
}
