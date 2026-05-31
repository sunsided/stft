//! Forward short-time Fourier transform: streaming processor and builder.

use crate::config::{PadMode, Scaling};
use crate::error::StftError;
use crate::sample::{cast, Sample};
use crate::window::Window;
use alloc::collections::VecDeque;
use alloc::sync::Arc;
use alloc::vec::Vec;
use num_complex::Complex;
use realfft::{FftNum, RealFftPlanner, RealToComplex};

/// Builder for [`Stft`].
///
/// A window is mandatory; everything else has a sensible default
/// (hop = `frame_len / 4`, fft size = `frame_len`, no scaling, no centering).
#[must_use]
pub struct StftBuilder<T: Sample + FftNum> {
    window: Option<Window<T>>,
    hop: Option<usize>,
    fft_size: Option<usize>,
    scaling: Scaling,
    center: bool,
    pad_mode: PadMode,
    sample_rate: Option<f64>,
}

impl<T: Sample + FftNum> Default for StftBuilder<T> {
    fn default() -> Self {
        Self {
            window: None,
            hop: None,
            fft_size: None,
            scaling: Scaling::None,
            center: false,
            pad_mode: PadMode::Zero,
            sample_rate: None,
        }
    }
}

impl<T: Sample + FftNum> StftBuilder<T> {
    /// Set the analysis window. Its length becomes the frame length.
    pub fn window(mut self, window: Window<T>) -> Self {
        self.window = Some(window);
        self
    }

    /// Set the hop size (samples advanced between frames). Defaults to
    /// `frame_len / 4`.
    pub fn hop_size(mut self, hop: usize) -> Self {
        self.hop = Some(hop);
        self
    }

    /// Set the FFT size; values larger than the frame length zero-pad each
    /// frame. Defaults to the frame length.
    pub fn fft_size(mut self, fft_size: usize) -> Self {
        self.fft_size = Some(fft_size);
        self
    }

    /// Set the coefficient [`Scaling`] mode.
    pub fn scaling(mut self, scaling: Scaling) -> Self {
        self.scaling = scaling;
        self
    }

    /// Enable centered framing for batch [`spectrogram`](Stft::spectrogram):
    /// the signal is padded by `frame_len / 2` on each side.
    pub fn center(mut self, center: bool) -> Self {
        self.center = center;
        self
    }

    /// Set how the signal is padded when centered framing is enabled.
    pub fn pad_mode(mut self, pad_mode: PadMode) -> Self {
        self.pad_mode = pad_mode;
        self
    }

    /// Set the sample rate, required for [`Scaling::Density`].
    pub fn sample_rate(mut self, fs: f64) -> Self {
        self.sample_rate = Some(fs);
        self
    }

    /// Validate the configuration and build the [`Stft`].
    ///
    /// # Errors
    /// Returns [`StftError`] if the window is missing/empty, the hop size is
    /// out of range, the FFT size is smaller than the frame length, or density
    /// scaling is requested without a sample rate.
    pub fn build(self) -> Result<Stft<T>, StftError> {
        let window = self.window.ok_or(StftError::MissingWindow)?;
        let frame_len = window.len();
        if frame_len == 0 {
            return Err(StftError::InvalidFrameLength);
        }

        let hop = self.hop.unwrap_or((frame_len / 4).max(1));
        if hop == 0 || hop > frame_len {
            return Err(StftError::InvalidHopSize { hop, frame_len });
        }

        let fft_size = self.fft_size.unwrap_or(frame_len);
        if fft_size < frame_len {
            return Err(StftError::InvalidFftSize {
                fft_size,
                frame_len,
            });
        }

        let scale = match self.scaling {
            Scaling::None => T::one(),
            Scaling::Magnitude => T::one() / window.sum(),
            Scaling::Density => {
                let fs = self.sample_rate.ok_or(StftError::MissingSampleRate)?;
                T::one() / (cast::<T>(fs) * window.sum_squared()).sqrt()
            }
        };

        let fft = RealFftPlanner::<T>::new().plan_fft_forward(fft_size);
        let input = fft.make_input_vec();
        let spectrum = fft.make_output_vec();
        let scratch = fft.make_scratch_vec();
        let n_freqs = spectrum.len();

        Ok(Stft {
            window,
            frame_len,
            hop,
            fft_size,
            n_freqs,
            scale,
            center: self.center,
            pad_mode: self.pad_mode,
            fft,
            input,
            spectrum,
            scratch,
            ring: VecDeque::new(),
        })
    }
}

/// A streaming forward short-time Fourier transform over real samples.
///
/// Feed samples with [`append`](Stft::append); whenever [`ready`](Stft::ready)
/// is true, compute a column with [`process_into`](Stft::process_into) (or use
/// the [`columns`](Stft::columns) iterator) and advance with
/// [`step`](Stft::step). For one-shot processing of a whole signal use
/// [`spectrogram`](Stft::spectrogram).
pub struct Stft<T: Sample + FftNum> {
    window: Window<T>,
    frame_len: usize,
    hop: usize,
    fft_size: usize,
    n_freqs: usize,
    scale: T,
    pub(crate) center: bool,
    pub(crate) pad_mode: PadMode,
    fft: Arc<dyn RealToComplex<T>>,
    input: Vec<T>,
    spectrum: Vec<Complex<T>>,
    scratch: Vec<Complex<T>>,
    ring: VecDeque<T>,
}

impl<T: Sample + FftNum> Stft<T> {
    /// Start building an [`Stft`].
    pub fn builder() -> StftBuilder<T> {
        StftBuilder::default()
    }

    /// Number of frequency bins per column: `fft_size / 2 + 1` (DC … Nyquist).
    #[must_use]
    pub fn n_freqs(&self) -> usize {
        self.n_freqs
    }

    /// The frame (window) length.
    #[must_use]
    pub fn frame_len(&self) -> usize {
        self.frame_len
    }

    /// The hop size.
    #[must_use]
    pub fn hop(&self) -> usize {
        self.hop
    }

    /// The FFT size.
    #[must_use]
    pub fn fft_size(&self) -> usize {
        self.fft_size
    }

    /// The multiplicative scaling factor applied to each coefficient.
    #[must_use]
    pub fn scale(&self) -> T {
        self.scale
    }

    /// The analysis window.
    #[must_use]
    pub fn window(&self) -> &Window<T> {
        &self.window
    }

    /// A cloned handle to the shared forward-FFT plan (used by batch workers).
    #[cfg(feature = "rayon")]
    pub(crate) fn fft_handle(&self) -> Arc<dyn RealToComplex<T>> {
        self.fft.clone()
    }

    /// The bin center frequencies for a sample rate `fs`: `freqs[k] = k·fs / fft_size`.
    #[must_use]
    pub fn freqs(&self, fs: f64) -> Vec<T> {
        let fft_size = self.fft_size as f64;
        (0..self.n_freqs)
            .map(|k| cast(k as f64 * fs / fft_size))
            .collect()
    }

    /// Append samples to the internal ring buffer.
    pub fn append(&mut self, samples: &[T]) {
        self.ring.extend(samples.iter().copied());
    }

    /// Number of buffered samples awaiting processing.
    #[must_use]
    pub fn buffered(&self) -> usize {
        self.ring.len()
    }

    /// Whether a full frame is available to process.
    #[must_use]
    pub fn ready(&self) -> bool {
        self.ring.len() >= self.frame_len
    }

    /// Clear the internal ring buffer.
    pub fn reset(&mut self) {
        self.ring.clear();
    }

    /// Compute the current column into `out` without advancing.
    ///
    /// # Errors
    /// Returns [`StftError::LengthMismatch`] if `out.len() != n_freqs`, or
    /// [`StftError::NotEnoughData`] if fewer than `frame_len` samples are
    /// buffered.
    pub fn process_into(&mut self, out: &mut [Complex<T>]) -> Result<(), StftError> {
        if out.len() != self.n_freqs {
            return Err(StftError::LengthMismatch {
                expected: self.n_freqs,
                got: out.len(),
            });
        }
        if self.ring.len() < self.frame_len {
            return Err(StftError::NotEnoughData {
                needed: self.frame_len,
                available: self.ring.len(),
            });
        }
        self.compute_from_ring()?;
        out.copy_from_slice(&self.spectrum);
        Ok(())
    }

    /// Drop `hop` samples from the front of the ring buffer.
    pub fn step(&mut self) {
        let drop = self.hop.min(self.ring.len());
        self.ring.drain(..drop);
    }

    /// Iterate over spectrogram columns, advancing by `hop` after each, until
    /// fewer than `frame_len` samples remain buffered.
    pub fn columns(&mut self) -> Columns<'_, T> {
        Columns { stft: self }
    }

    /// Fill `self.input` from the front `frame_len` samples of the ring,
    /// applying the window and zero-padding, then run the FFT and scaling.
    fn compute_from_ring(&mut self) -> Result<(), StftError> {
        let frame_len = self.frame_len;
        let win = self.window.coefficients();
        let (head, tail) = self.input.split_at_mut(frame_len);
        for ((dst, &w), &s) in head.iter_mut().zip(win).zip(self.ring.iter()) {
            *dst = s * w;
        }
        for dst in tail {
            *dst = T::zero();
        }
        self.run_fft()
    }

    /// Run the forward FFT on `self.input`, writing to `self.spectrum` and
    /// applying the scaling factor.
    fn run_fft(&mut self) -> Result<(), StftError> {
        self.fft
            .process_with_scratch(&mut self.input, &mut self.spectrum, &mut self.scratch)
            .map_err(|_| StftError::Fft)?;
        if self.scale != T::one() {
            let scale = self.scale;
            for bin in &mut self.spectrum {
                *bin = *bin * scale;
            }
        }
        Ok(())
    }

    /// Fill `self.input` from an arbitrary `frame` slice (length `frame_len`)
    /// and compute its spectrum. Used by serial batch processing.
    #[cfg(not(feature = "rayon"))]
    pub(crate) fn compute_frame(&mut self, frame: &[T]) -> Result<&[Complex<T>], StftError> {
        debug_assert_eq!(frame.len(), self.frame_len);
        let frame_len = self.frame_len;
        let win = self.window.coefficients();
        let (head, tail) = self.input.split_at_mut(frame_len);
        for ((dst, &w), &s) in head.iter_mut().zip(win).zip(frame) {
            *dst = s * w;
        }
        for dst in tail {
            *dst = T::zero();
        }
        self.run_fft()?;
        Ok(&self.spectrum)
    }
}

/// Iterator over spectrogram columns produced by [`Stft::columns`].
pub struct Columns<'a, T: Sample + FftNum> {
    stft: &'a mut Stft<T>,
}

impl<T: Sample + FftNum> Iterator for Columns<'_, T> {
    type Item = Vec<Complex<T>>;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.stft.ready() {
            return None;
        }
        // `ready()` guarantees a full frame, so `compute_from_ring` succeeds.
        self.stft.compute_from_ring().ok()?;
        let column = self.stft.spectrum.clone();
        self.stft.step();
        Some(column)
    }
}
