//! Inverse short-time Fourier transform via weighted overlap-add (WOLA).

use crate::batch::Spectrogram;
use crate::error::StftError;
use crate::sample::{cast, Sample};
use crate::stft::Stft;
use crate::window::Window;
use alloc::sync::Arc;
use alloc::vec::Vec;
use num_complex::Complex;
use realfft::{ComplexToReal, FftNum, RealFftPlanner};

/// Builder for [`Istft`].
#[must_use]
pub struct IstftBuilder<T: Sample + FftNum> {
    window: Option<Window<T>>,
    hop: Option<usize>,
    fft_size: Option<usize>,
    forward_scale: T,
    center: bool,
}

impl<T: Sample + FftNum> Default for IstftBuilder<T> {
    fn default() -> Self {
        Self {
            window: None,
            hop: None,
            fft_size: None,
            forward_scale: T::one(),
            center: false,
        }
    }
}

impl<T: Sample + FftNum> IstftBuilder<T> {
    /// Set the synthesis window. For perfect reconstruction this must be the
    /// same window used by the forward transform.
    pub fn window(mut self, window: Window<T>) -> Self {
        self.window = Some(window);
        self
    }

    /// Set the hop size. Must equal the forward hop size.
    pub fn hop_size(mut self, hop: usize) -> Self {
        self.hop = Some(hop);
        self
    }

    /// Set the FFT size. Must equal the forward FFT size.
    pub fn fft_size(mut self, fft_size: usize) -> Self {
        self.fft_size = Some(fft_size);
        self
    }

    /// Set the multiplicative scaling factor that the forward transform
    /// applied, so it can be undone. Defaults to `1`.
    pub fn forward_scale(mut self, scale: T) -> Self {
        self.forward_scale = scale;
        self
    }

    /// Indicate that the forward transform used centered framing, so that
    /// [`Istft::finish`] trims the `frame_len / 2` padding from each end.
    pub fn center(mut self, center: bool) -> Self {
        self.center = center;
        self
    }

    /// Validate the configuration and build the [`Istft`].
    ///
    /// # Errors
    /// Returns [`StftError`] for a missing/empty window, out-of-range hop, or
    /// an FFT size smaller than the frame length.
    pub fn build(self) -> Result<Istft<T>, StftError> {
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

        let fft = RealFftPlanner::<T>::new().plan_fft_inverse(fft_size);
        let spec_in = fft.make_input_vec();
        let frame_out = fft.make_output_vec();
        let scratch = fft.make_scratch_vec();
        let n_freqs = spec_in.len();

        // Undo the unnormalized inverse FFT (factor `fft_size`) and the forward
        // scaling factor in one division.
        let inv_scale = T::one() / (self.forward_scale * cast::<T>(fft_size as f64));

        Ok(Istft {
            window,
            frame_len,
            hop,
            fft_size,
            n_freqs,
            inv_scale,
            center: self.center,
            fft,
            spec_in,
            frame_out,
            scratch,
            output: Vec::new(),
            norm: Vec::new(),
            pos: 0,
            frames: 0,
        })
    }
}

/// An inverse short-time Fourier transform that reconstructs a real signal
/// from spectrogram columns using weighted overlap-add.
///
/// Feed columns with [`process_column`](Istft::process_column) (or a whole
/// [`Spectrogram`] via [`reconstruct`](Istft::reconstruct)) and obtain the
/// signal with [`finish`](Istft::finish).
pub struct Istft<T: Sample + FftNum> {
    window: Window<T>,
    frame_len: usize,
    hop: usize,
    fft_size: usize,
    n_freqs: usize,
    inv_scale: T,
    center: bool,
    fft: Arc<dyn ComplexToReal<T>>,
    spec_in: Vec<Complex<T>>,
    frame_out: Vec<T>,
    scratch: Vec<Complex<T>>,
    output: Vec<T>,
    norm: Vec<T>,
    pos: usize,
    frames: usize,
}

impl<T: Sample + FftNum> Istft<T> {
    /// Start building an [`Istft`].
    pub fn builder() -> IstftBuilder<T> {
        IstftBuilder::default()
    }

    /// Number of frequency bins expected per column (`fft_size / 2 + 1`).
    #[must_use]
    pub fn n_freqs(&self) -> usize {
        self.n_freqs
    }

    /// Number of columns processed so far.
    #[must_use]
    pub fn frames(&self) -> usize {
        self.frames
    }

    /// Overlap-add one spectrogram column.
    ///
    /// # Errors
    /// Returns [`StftError::LengthMismatch`] if `column.len() != n_freqs`, or
    /// [`StftError::Fft`] if the backend fails.
    pub fn process_column(&mut self, column: &[Complex<T>]) -> Result<(), StftError> {
        if column.len() != self.n_freqs {
            return Err(StftError::LengthMismatch {
                expected: self.n_freqs,
                got: column.len(),
            });
        }

        self.spec_in.copy_from_slice(column);
        // The inverse real FFT requires the DC bin (and the Nyquist bin for an
        // even-length transform) to be purely real; force it to avoid backend
        // errors from round-off.
        self.spec_in[0].im = T::zero();
        if self.fft_size % 2 == 0 {
            let last = self.n_freqs - 1;
            self.spec_in[last].im = T::zero();
        }

        self.fft
            .process_with_scratch(&mut self.spec_in, &mut self.frame_out, &mut self.scratch)
            .map_err(|_| StftError::Fft)?;

        let end = self.pos + self.frame_len;
        if self.output.len() < end {
            self.output.resize(end, T::zero());
            self.norm.resize(end, T::zero());
        }

        let inv = self.inv_scale;
        let frame_len = self.frame_len;
        let pos = self.pos;
        let win = self.window.coefficients();
        let out_seg = &mut self.output[pos..pos + frame_len];
        let norm_seg = &mut self.norm[pos..pos + frame_len];
        for (((o, n), &w), &fo) in out_seg
            .iter_mut()
            .zip(norm_seg.iter_mut())
            .zip(win)
            .zip(&self.frame_out[..frame_len])
        {
            let recon = fo * inv;
            *o = *o + w * recon;
            *n = *n + w * w;
        }

        self.pos += self.hop;
        self.frames += 1;
        Ok(())
    }

    /// Overlap-add an entire [`Spectrogram`] and return the reconstructed
    /// signal.
    ///
    /// # Errors
    /// Propagates errors from [`process_column`](Istft::process_column).
    pub fn reconstruct(mut self, spectrogram: &Spectrogram<T>) -> Result<Vec<T>, StftError> {
        for column in spectrogram.columns() {
            self.process_column(column)?;
        }
        Ok(self.finish())
    }

    /// Finish reconstruction: normalize by the accumulated window energy and
    /// return the signal, trimming centered padding if configured.
    #[must_use]
    pub fn finish(self) -> Vec<T> {
        let mut output = self.output;
        let eps = cast::<T>(1e-12);
        for (o, n) in output.iter_mut().zip(&self.norm) {
            if *n > eps {
                *o = *o / *n;
            } else {
                *o = T::zero();
            }
        }

        if self.center {
            let pad = self.frame_len / 2;
            if output.len() >= 2 * pad {
                output.truncate(output.len() - pad);
                output.drain(..pad);
            }
        }
        output
    }
}

impl<T: Sample + FftNum> Stft<T> {
    /// Build an [`Istft`] that exactly inverts this forward transform
    /// (same window, hop, FFT size, scaling and centering).
    ///
    /// # Errors
    /// Returns [`StftError`] if the mirrored configuration is invalid.
    pub fn inverse(&self) -> Result<Istft<T>, StftError> {
        IstftBuilder::default()
            .window(self.window().clone())
            .hop_size(self.hop())
            .fft_size(self.fft_size())
            .forward_scale(self.scale())
            .center(self.center)
            .build()
    }
}
