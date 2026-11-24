//! One-shot batch spectrograms and the [`Spectrogram`] container.

use crate::config::PadMode;
use crate::sample::Sample;
use crate::stft::Stft;
use alloc::vec;
use alloc::vec::Vec;
use num_complex::Complex;
use realfft::FftNum;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// A dense spectrogram stored frame-major: `n_frames` columns of `n_freqs`
/// complex bins each.
#[derive(Debug, Clone, PartialEq)]
pub struct Spectrogram<T> {
    data: Vec<Complex<T>>,
    n_frames: usize,
    n_freqs: usize,
}

impl<T: Sample> Spectrogram<T> {
    /// Build a spectrogram from a frame-major flat buffer.
    ///
    /// # Panics
    /// Panics if `data.len() != n_frames * n_freqs`.
    #[must_use]
    pub fn from_flat(data: Vec<Complex<T>>, n_frames: usize, n_freqs: usize) -> Self {
        assert_eq!(data.len(), n_frames * n_freqs, "spectrogram shape mismatch");
        Self {
            data,
            n_frames,
            n_freqs,
        }
    }

    /// Number of frames (columns).
    #[must_use]
    pub fn n_frames(&self) -> usize {
        self.n_frames
    }

    /// Number of frequency bins per frame.
    #[must_use]
    pub fn n_freqs(&self) -> usize {
        self.n_freqs
    }

    /// Whether the spectrogram has no frames.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n_frames == 0
    }

    /// Borrow frame `index` as a slice of `n_freqs` bins.
    #[must_use]
    pub fn column(&self, index: usize) -> &[Complex<T>] {
        let start = index * self.n_freqs;
        &self.data[start..start + self.n_freqs]
    }

    /// Iterate over the frames (columns).
    pub fn columns(&self) -> impl Iterator<Item = &[Complex<T>]> {
        self.data.chunks_exact(self.n_freqs)
    }

    /// The underlying frame-major buffer.
    #[must_use]
    pub fn as_flat(&self) -> &[Complex<T>] {
        &self.data
    }

    /// Consume into the frame-major buffer.
    #[must_use]
    pub fn into_flat(self) -> Vec<Complex<T>> {
        self.data
    }

    /// Convert to a `[n_freqs, n_frames]` [`ndarray::Array2`] (librosa layout).
    #[cfg(feature = "ndarray")]
    #[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
    #[must_use]
    pub fn to_array2(&self) -> ndarray::Array2<Complex<T>> {
        ndarray::Array2::from_shape_fn((self.n_freqs, self.n_frames), |(bin, frame)| {
            self.data[frame * self.n_freqs + bin]
        })
    }
}

/// Reflect an index `p` into `0..len` using mirror (no edge repeat) semantics,
/// matching NumPy's `reflect` padding.
fn reflect_index(p: isize, len: isize) -> usize {
    if len == 1 {
        return 0;
    }
    let period = 2 * (len - 1);
    let mut m = p.rem_euclid(period);
    if m >= len {
        m = period - m;
    }
    m as usize
}

/// Pad `signal` by `pad` samples on each side according to `mode`.
fn pad_signal<T: Sample>(signal: &[T], pad: usize, mode: PadMode) -> Vec<T> {
    let len = signal.len();
    if pad == 0 {
        return signal.to_vec();
    }
    let mut out = vec![T::zero(); len + 2 * pad];
    let len_i = len as isize;
    for (i, slot) in out.iter_mut().enumerate() {
        let p = i as isize - pad as isize;
        *slot = if p >= 0 && p < len_i {
            signal[p as usize]
        } else if len == 0 {
            T::zero()
        } else {
            match mode {
                PadMode::Zero => T::zero(),
                PadMode::Edge => signal[p.clamp(0, len_i - 1) as usize],
                PadMode::Reflect => signal[reflect_index(p, len_i)],
            }
        };
    }
    out
}

/// Return `(pad, n_frames)` for a signal of `signal_len` samples.
///
/// `pad` is the padding on each side (`frame_len / 2` if `center` is set, else
/// `0`). `n_frames` is the number of full frames in the padded signal. The
/// caller must make sure that `hop > 0`. The sum `signal_len + 2 * pad` does not
/// overflow because both terms are lengths of in-memory buffers (at most
/// `isize::MAX` each).
fn frame_count(signal_len: usize, frame_len: usize, hop: usize, center: bool) -> (usize, usize) {
    let pad = if center { frame_len / 2 } else { 0 };
    let padded_len = signal_len + 2 * pad;
    let n_frames = if padded_len >= frame_len {
        1 + (padded_len - frame_len) / hop
    } else {
        0
    };

    (pad, n_frames)
}

impl<T: Sample + FftNum> Stft<T> {
    /// Compute the full spectrogram of `signal` in one call.
    ///
    /// Resets the internal streaming buffer. With
    /// [`center`](crate::StftBuilder::center) enabled the signal is padded by
    /// `frame_len / 2` on each side using the configured [`PadMode`]. With the
    /// `rayon` feature the frames are computed in parallel.
    #[must_use]
    pub fn spectrogram(&mut self, signal: &[T]) -> Spectrogram<T> {
        self.reset();
        let (pad, n_frames) = frame_count(signal.len(), self.frame_len(), self.hop(), self.center);
        let n_freqs = self.n_freqs();
        let frame_len = self.frame_len();
        let hop = self.hop();

        let padded = pad_signal(signal, pad, self.pad_mode);
        let zero = Complex::new(T::zero(), T::zero());
        let mut data = vec![zero; n_frames * n_freqs];

        #[cfg(feature = "rayon")]
        {
            let fft = self.fft_handle();
            let win = self.window().coefficients();
            let scale = self.scale();
            let one = T::one();
            data.par_chunks_mut(n_freqs).enumerate().for_each_init(
                || (fft.make_input_vec(), fft.make_scratch_vec()),
                |(input, scratch), (frame_idx, out_col)| {
                    let start = frame_idx * hop;
                    let frame = &padded[start..start + frame_len];
                    let (head, tail) = input.split_at_mut(frame_len);
                    for ((dst, &w), &s) in head.iter_mut().zip(win).zip(frame) {
                        *dst = s * w;
                    }
                    for dst in tail {
                        *dst = T::zero();
                    }
                    fft.process_with_scratch(input, out_col, scratch)
                        .expect("realfft forward");
                    if scale != one {
                        for bin in out_col.iter_mut() {
                            *bin = *bin * scale;
                        }
                    }
                },
            );
        }

        #[cfg(not(feature = "rayon"))]
        {
            for frame_idx in 0..n_frames {
                let start = frame_idx * hop;
                let frame = &padded[start..start + frame_len];
                let spectrum = self.compute_frame(frame).expect("realfft forward");
                let out_start = frame_idx * n_freqs;
                data[out_start..out_start + n_freqs].copy_from_slice(spectrum);
            }
        }

        Spectrogram::from_flat(data, n_frames, n_freqs)
    }
}

/// Kani proofs for the index arithmetic of batch framing and padding.
///
/// The proofs cover integer logic only. The FFT and the float math are out of
/// scope; property tests cover them.
#[cfg(kani)]
mod verification {
    use super::{frame_count, pad_signal, reflect_index, PadMode};

    /// `reflect_index` stays in `0..len` for every position `p`, and is the
    /// identity inside the signal.
    ///
    /// `len` is at most `isize::MAX / 2`, so `2 * (len - 1)` does not overflow.
    /// Slices of `f32` or `f64` are always shorter than that.
    #[kani::proof]
    fn reflect_index_in_bounds() {
        let len: isize = kani::any();
        let p: isize = kani::any();
        kani::assume(len >= 1 && len <= isize::MAX / 2);

        let m = reflect_index(p, len);

        assert!(m < len.unsigned_abs());
        if p >= 0 && p < len {
            assert_eq!(m, p.unsigned_abs());
        }
    }

    /// The frame count never overflows or divides by zero, over the full range
    /// of valid lengths.
    #[kani::proof]
    fn frame_count_no_overflow() {
        let signal_len: usize = kani::any();
        let frame_len: usize = kani::any();
        let hop: usize = kani::any();
        kani::assume(signal_len <= isize::MAX.unsigned_abs());
        kani::assume(frame_len >= 1 && frame_len <= isize::MAX.unsigned_abs());
        kani::assume(hop >= 1 && hop <= frame_len);

        let _ = frame_count(signal_len, frame_len, hop, kani::any());
    }

    /// Every frame fits in the padded signal and one more frame would not.
    ///
    /// The lengths are bounded to `2^10` to keep the nonlinear arithmetic
    /// tractable for the solver.
    #[kani::proof]
    fn frame_count_frames_fit() {
        let signal_len: usize = kani::any();
        let frame_len: usize = kani::any();
        let hop: usize = kani::any();
        let center: bool = kani::any();
        kani::assume(signal_len <= 1 << 10);
        kani::assume(frame_len >= 1 && frame_len <= 1 << 10);
        kani::assume(hop >= 1 && hop <= frame_len);

        let (pad, n_frames) = frame_count(signal_len, frame_len, hop, center);
        let padded_len = signal_len + 2 * pad;

        if n_frames > 0 {
            assert!((n_frames - 1) * hop + frame_len <= padded_len);
            assert!(n_frames * hop + frame_len > padded_len);
        } else {
            assert!(padded_len < frame_len);
        }
    }

    /// `pad_signal` never indexes out of bounds, keeps the signal in the
    /// middle, and returns `len + 2 * pad` samples for every mode.
    #[kani::proof]
    #[kani::unwind(10)]
    fn pad_signal_shape() {
        let len: usize = kani::any();
        let pad: usize = kani::any();
        kani::assume(len <= 3 && pad <= 3);
        let mode = match kani::any::<u8>() % 3 {
            0 => PadMode::Zero,
            1 => PadMode::Edge,
            _ => PadMode::Reflect,
        };
        let signal: [f32; 3] = [1.0, 2.0, 3.0];

        let out = pad_signal(&signal[..len], pad, mode);

        assert_eq!(out.len(), len + 2 * pad);
        for i in 0..len {
            assert_eq!(out[pad + i], signal[i]);
        }
    }
}
