#![no_main]
//! Arbitrary (but bounded) STFT configurations plus a streaming run must
//! never panic, regardless of window/hop/fft-size combination.

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use ruststft::{Complex, Stft, Window};

#[derive(Arbitrary, Debug)]
struct Input {
    win: u16,
    hop: u16,
    pad: u16,
    samples: Vec<f32>,
}

fuzz_target!(|input: Input| {
    let win = ((input.win % 2048) as usize).max(1);
    let hop = ((input.hop as usize) % win).max(1); // 1..=win
    let fft = win + (input.pad as usize % 4096); // >= win, bounded to avoid OOM

    let mut stft = match Stft::builder()
        .window(Window::<f32>::hann(win))
        .hop_size(hop)
        .fft_size(fft)
        .build()
    {
        Ok(s) => s,
        Err(_) => return,
    };

    let samples: Vec<f32> = input
        .samples
        .into_iter()
        .filter(|x| x.is_finite())
        .collect();
    stft.append(&samples);

    let mut column = vec![Complex::new(0.0f32, 0.0); stft.n_freqs()];
    while stft.ready() {
        if stft.process_into(&mut column).is_err() {
            break;
        }
        stft.step();
    }
});
