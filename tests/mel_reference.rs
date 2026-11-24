//! Mel scale, mel filterbank and DCT-II checked against librosa and SciPy.
//!
//! Reference values come from librosa 0.11 (`librosa.filters.mel`,
//! `librosa.hz_to_mel`) and `scipy.fft.dct(type=2, norm="ortho")`.
#![cfg(feature = "mel")]

use approx::{assert_abs_diff_eq, assert_relative_eq};
use ruststft::mel::{hz_to_mel, DctII, MelFilterBank, MelScale};

/// librosa `filters.mel(sr=8000, n_fft=16, n_mels=4, fmin=100, fmax=3500, htk=False)`.
const SLANEY_BANK: [f64; 36] = [
    0.0,
    0.002236011903733015,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0019052065908908844,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0001589191087987274,
    0.00144827866461128,
    0.0004988981527276337,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    4.557692227535881e-05,
    0.000659274635836482,
    0.000793510873336345,
    0.0003967554366681725,
    0.0,
    0.0,
];

/// The same filterbank with `htk=True`.
const HTK_BANK: [f64; 36] = [
    0.0,
    0.002138787880539894,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.00037141586653888226,
    0.0014474077615886927,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0003295188653282821,
    0.001309913583099842,
    0.0005058069946244359,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    4.1966417484218255e-05,
    0.0006191072170622647,
    0.0008284759824164212,
    0.0004142379912082106,
    0.0,
    0.0,
];

/// `hz_to_mel` matches librosa on both sides of the Slaney 1 kHz break and for HTK.
#[test]
fn hz_to_mel_matches_librosa() {
    let hz = [0.0, 500.0, 1000.0, 4000.0];
    let slaney = [0.0, 7.5, 15.0, 35.163_760_314_616_646];
    let htk = [
        0.0,
        607.445_919_656_59,
        999.985_537_139_624_4,
        2_146.064_527_506_19,
    ];

    for ((&f, &s), &h) in hz.iter().zip(&slaney).zip(&htk) {
        assert_abs_diff_eq!(hz_to_mel(f, MelScale::Slaney), s, epsilon = 1e-9);
        assert_abs_diff_eq!(hz_to_mel(f, MelScale::Htk), h, epsilon = 1e-9);
    }
}

/// The filterbank weights match librosa for both mel scales.
///
/// librosa returns `float32` weights, so the check uses a relative tolerance.
#[test]
fn filterbank_matches_librosa() {
    for (scale, expected) in [(MelScale::Slaney, &SLANEY_BANK), (MelScale::Htk, &HTK_BANK)] {
        let bank = MelFilterBank::<f64>::new(4, 16, 8000.0, 100.0, 3500.0, scale);
        assert_eq!((bank.n_mels(), bank.n_freqs()), (4, 9));
        assert_eq!(bank.weights().len(), expected.len());

        for (&got, &want) in bank.weights().iter().zip(expected) {
            assert_relative_eq!(got, want, max_relative = 1e-6);
        }
    }
}

/// `transform` is the matrix product of the weights with the power column.
#[test]
fn filterbank_transform_is_matrix_product() {
    let bank = MelFilterBank::<f64>::new(4, 16, 8000.0, 100.0, 3500.0, MelScale::Slaney);
    let power: Vec<f64> = (0..9).map(|k| f64::from(k) + 0.5).collect();

    let mel = bank.transform(&power);

    for (m, &got) in mel.iter().enumerate() {
        let row = &SLANEY_BANK[m * 9..(m + 1) * 9];
        let want: f64 = row.iter().zip(&power).map(|(w, p)| w * p).sum();
        assert_relative_eq!(got, want, max_relative = 1e-6);
    }
}

/// The orthonormal DCT-II matches SciPy for the first three coefficients.
#[test]
fn dct2_matches_scipy() {
    let dct = DctII::<f64>::new(5, 3);
    assert_eq!((dct.n_in(), dct.n_out()), (5, 3));

    let y = dct.transform(&[1.0, -2.0, 3.5, 0.25, -1.0]);
    let expected = [
        0.782_623_792_124_926_5,
        0.366_568_832_479_676_3,
        -1.871_575_223_844_369_7,
    ];

    for (&got, want) in y.iter().zip(expected) {
        assert_abs_diff_eq!(got, want, epsilon = 1e-12);
    }
}
