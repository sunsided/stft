//! Criterion benchmarks for the forward/inverse STFT.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ruststft::{Complex, Stft, Window};

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

criterion_group!(benches, bench_forward, bench_streaming, bench_round_trip);
criterion_main!(benches);
