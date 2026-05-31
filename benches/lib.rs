//! Criterion benchmarks for windows, the forward/inverse STFT, spectrum
//! helpers and (optionally) the mel transforms.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ruststft::spectrum::{magnitude_into, power_to_db};
use ruststft::{Complex, Stft, Symmetry, Window, WindowFunction};

fn signal_f32(seconds: usize) -> Vec<f32> {
    let fs = 44_100usize;
    (0..fs * seconds)
        .map(|n| {
            let t = n as f32 / fs as f32;
            0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
        })
        .collect()
}

fn signal_f64(seconds: usize) -> Vec<f64> {
    let fs = 44_100usize;
    (0..fs * seconds)
        .map(|n| {
            let t = n as f64 / fs as f64;
            0.5 * (2.0 * std::f64::consts::PI * 440.0 * t).sin()
        })
        .collect()
}

fn bench_window(c: &mut Criterion) {
    let mut group = c.benchmark_group("window_generation");
    for &len in &[1024usize, 4096] {
        for (label, func) in [
            ("hann", WindowFunction::Hann),
            ("blackman_harris", WindowFunction::BlackmanHarris),
            ("kaiser", WindowFunction::Kaiser { beta: 8.6 }),
            ("gaussian", WindowFunction::Gaussian { std: 128.0 }),
        ] {
            group.bench_with_input(BenchmarkId::new(label, len), &len, |b, &len| {
                b.iter(|| Window::<f64>::new(func, len, Symmetry::Periodic));
            });
        }
    }
    group.finish();
}

fn bench_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("forward_spectrogram");
    let s32 = signal_f32(10);
    let s64 = signal_f64(10);
    group.throughput(Throughput::Elements(s32.len() as u64));

    for &win in &[1024usize, 4096] {
        group.bench_with_input(BenchmarkId::new("f32", win), &win, |b, &win| {
            let mut stft = Stft::builder()
                .window(Window::<f32>::hann(win))
                .hop_size(win / 4)
                .build()
                .unwrap();
            b.iter(|| stft.spectrogram(&s32));
        });
        group.bench_with_input(BenchmarkId::new("f64", win), &win, |b, &win| {
            let mut stft = Stft::builder()
                .window(Window::<f64>::hann(win))
                .hop_size(win / 4)
                .build()
                .unwrap();
            b.iter(|| stft.spectrogram(&s64));
        });
    }
    group.finish();
}

fn bench_streaming(c: &mut Criterion) {
    let signal = signal_f32(10);
    c.bench_function("streaming_columns_1024_f32", |b| {
        b.iter(|| {
            let mut stft = Stft::builder()
                .window(Window::<f32>::hann(1024))
                .hop_size(512)
                .build()
                .unwrap();
            let mut column = vec![Complex::new(0.0f32, 0.0); stft.n_freqs()];
            for chunk in signal.chunks(4096) {
                stft.append(chunk);
                while stft.ready() {
                    stft.process_into(&mut column).unwrap();
                    stft.step();
                }
            }
        });
    });
}

fn bench_round_trip(c: &mut Criterion) {
    let signal = signal_f64(5);
    c.bench_function("round_trip_1024_f64", |b| {
        let mut stft = Stft::builder()
            .window(Window::<f64>::hann(1024))
            .hop_size(256)
            .build()
            .unwrap();
        b.iter(|| {
            let spec = stft.spectrogram(&signal);
            let istft = stft.inverse().unwrap();
            istft.reconstruct(&spec).unwrap()
        });
    });
}

fn bench_spectrum(c: &mut Criterion) {
    let signal = signal_f64(5);
    let mut stft = Stft::builder()
        .window(Window::<f64>::hann(1024))
        .hop_size(256)
        .build()
        .unwrap();
    let spec = stft.spectrogram(&signal);

    let mut group = c.benchmark_group("spectrum");
    group.bench_function("magnitude_full", |b| {
        let mut mag = vec![0.0f64; spec.n_freqs()];
        b.iter(|| {
            for col in spec.columns() {
                magnitude_into(col, &mut mag);
            }
        });
    });
    group.bench_function("power_to_db_full", |b| {
        // `power_to_db` mutates in place, so restore a fresh power buffer each
        // iteration from an immutable baseline.
        let baseline: Vec<f64> = spec.as_flat().iter().map(|z| z.norm_sqr()).collect();
        b.iter_batched_ref(
            || baseline.clone(),
            |powers| power_to_db(powers, 1.0, Some(80.0)),
            criterion::BatchSize::SmallInput,
        );
    });
    group.finish();
}

#[cfg(feature = "mel")]
fn bench_mel(c: &mut Criterion) {
    use ruststft::mel::{DctII, MelFilterBank, MelScale};
    use ruststft::spectrum::power;

    let fs = 16_000.0;
    let n_fft = 1024usize;
    // Generate the signal at `fs` so the mel filterbank bins line up with the
    // spectrogram bins (signal_f64 is hardcoded to 44.1 kHz).
    let signal: Vec<f64> = (0..(fs as usize) * 5)
        .map(|n| {
            let t = n as f64 / fs;
            0.5 * (2.0 * std::f64::consts::PI * 440.0 * t).sin()
        })
        .collect();
    let mut stft = Stft::builder()
        .window(Window::<f64>::hann(n_fft))
        .hop_size(n_fft / 4)
        .build()
        .unwrap();
    let spec = stft.spectrogram(&signal);
    let bank = MelFilterBank::<f64>::new(40, n_fft, fs, 0.0, fs / 2.0, MelScale::Slaney);
    let dct = DctII::<f64>::new(40, 13);

    let mut group = c.benchmark_group("mel");
    group.bench_function("logmel_mfcc_full", |b| {
        let mut mel = vec![0.0f64; 40];
        let mut mfcc = vec![0.0f64; 13];
        b.iter(|| {
            for col in spec.columns() {
                let p = power(col);
                bank.transform_into(&p, &mut mel);
                power_to_db(&mut mel, 1.0, None);
                dct.transform_into(&mel, &mut mfcc);
            }
        });
    });
    group.finish();
}

#[cfg(feature = "mel")]
criterion_group!(
    benches,
    bench_window,
    bench_forward,
    bench_streaming,
    bench_round_trip,
    bench_spectrum,
    bench_mel
);

#[cfg(not(feature = "mel"))]
criterion_group!(
    benches,
    bench_window,
    bench_forward,
    bench_streaming,
    bench_round_trip,
    bench_spectrum
);

criterion_main!(benches);
