//! Inverse short-time Fourier transform via weighted overlap-add (WOLA).

use crate::batch::Spectrogram;
use crate::error::StftError;
use crate::sample::{cast, Sample};
use crate::stft::{is_valid_scale, FrameGeometry, Stft};
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
    /// Returns [`StftError`] for a missing/empty window or a non-finite
    /// coefficient, an out-of-range hop, an FFT size smaller than the frame
    /// length, or a forward scale that is not finite or is zero
    /// ([`StftError::InvalidScale`]).
    pub fn build(self) -> Result<Istft<T>, StftError> {
        let window = self.window.ok_or(StftError::MissingWindow)?;
        let FrameGeometry {
            frame_len,
            hop,
            fft_size,
        } = FrameGeometry::validate(&window, self.hop, self.fft_size)?;
        if !is_valid_scale(self.forward_scale) {
            return Err(StftError::InvalidScale);
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
            front_pad: if self.center { frame_len / 2 } else { 0 },
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
    /// Start of the next frame, relative to the first sample still in `output`.
    pos: usize,
    /// Number of columns processed so far.
    frames: usize,
    /// Centered padding at the signal start that is not yet trimmed. Becomes
    /// `0` once [`Istft::drain_ready`] has dropped it.
    front_pad: usize,
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
    /// The output covers only the samples that the frames reach. It can be up
    /// to `hop - 1` samples shorter than the analyzed signal. Use
    /// [`reconstruct_with_len`](Istft::reconstruct_with_len) to get an exact
    /// length.
    ///
    /// # Errors
    /// Propagates errors from [`process_column`](Istft::process_column).
    pub fn reconstruct(mut self, spectrogram: &Spectrogram<T>) -> Result<Vec<T>, StftError> {
        for column in spectrogram.columns() {
            self.process_column(column)?;
        }
        Ok(self.finish())
    }

    /// Like [`reconstruct`](Istft::reconstruct), but return exactly `len`
    /// samples: zero-pad or truncate the end.
    ///
    /// Pass the length of the analyzed signal to get a signal of the same
    /// length back.
    ///
    /// # Errors
    /// Propagates errors from [`process_column`](Istft::process_column).
    pub fn reconstruct_with_len(
        self,
        spectrogram: &Spectrogram<T>,
        len: usize,
    ) -> Result<Vec<T>, StftError> {
        let mut output = self.reconstruct(spectrogram)?;
        output.resize(len, T::zero());
        Ok(output)
    }

    /// Move all samples that no later column can change into `out`.
    ///
    /// Call this between [`process_column`](Istft::process_column) calls to
    /// stream the signal with bounded memory. The samples are normalized and
    /// centered padding at the start is dropped, as in
    /// [`finish`](Istft::finish). [`finish`](Istft::finish) then returns only
    /// the samples that remain. Returns the number of samples appended.
    ///
    /// With centered framing, the last `frame_len / 2` samples stay buffered,
    /// because they can be end padding that [`finish`](Istft::finish) trims.
    pub fn drain_ready(&mut self, out: &mut Vec<T>) -> usize {
        let end_pad = if self.center { self.frame_len / 2 } else { 0 };
        let ready = self.pos.min(self.output.len().saturating_sub(end_pad));
        normalize(&mut self.output[..ready], &self.norm[..ready]);

        let skip = self.front_pad.min(ready);
        self.front_pad -= skip;
        let before = out.len();
        out.extend(self.output.drain(..ready).skip(skip));
        self.norm.drain(..ready);
        self.pos -= ready;

        out.len() - before
    }

    /// Finish reconstruction: normalize by the accumulated window energy and
    /// return the signal, trimming centered padding if configured.
    ///
    /// After [`drain_ready`](Istft::drain_ready) this returns only the samples
    /// that were not drained.
    #[must_use]
    pub fn finish(self) -> Vec<T> {
        let mut output = self.output;
        normalize(&mut output, &self.norm);

        if self.center {
            let end_pad = self.frame_len / 2;
            let front = self.front_pad.min(output.len());
            output.drain(..front);
            output.truncate(output.len().saturating_sub(end_pad));
        }

        output
    }
}

/// Divide each overlap-added sample by its accumulated squared window weight.
///
/// Samples with a weight at or below `1e-12` get no window coverage and are set
/// to zero.
fn normalize<T: Sample>(output: &mut [T], norm: &[T]) {
    let eps = cast::<T>(1e-12);
    for (o, &n) in output.iter_mut().zip(norm) {
        *o = if n > eps { *o / n } else { T::zero() };
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
