#![no_main]
//! Generating any window family at any (bounded) length and parameters must
//! never panic or hang.

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use ruststft::{Symmetry, Window, WindowFunction};

#[derive(Arbitrary, Debug)]
struct Input {
    len: u16,
    which: u8,
    alpha: f64,
    beta: f64,
    std: f64,
    symmetric: bool,
}

fuzz_target!(|input: Input| {
    let len = (input.len % 8192) as usize;
    let symmetry = if input.symmetric {
        Symmetry::Symmetric
    } else {
        Symmetry::Periodic
    };
    let func = match input.which % 14 {
        0 => WindowFunction::Rectangular,
        1 => WindowFunction::Hann,
        2 => WindowFunction::Hamming,
        3 => WindowFunction::Blackman,
        4 => WindowFunction::BlackmanHarris,
        5 => WindowFunction::Nuttall,
        6 => WindowFunction::FlatTop,
        7 => WindowFunction::Bartlett,
        8 => WindowFunction::Triangular,
        9 => WindowFunction::Welch,
        10 => WindowFunction::Cosine,
        11 => WindowFunction::Tukey { alpha: input.alpha },
        12 => WindowFunction::Kaiser { beta: input.beta },
        _ => WindowFunction::Gaussian { std: input.std },
    };
    let _ = Window::<f64>::new(func, len, symmetry);
});
