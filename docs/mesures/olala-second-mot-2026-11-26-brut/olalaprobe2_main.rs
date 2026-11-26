// Data-free (H = I) Tetra encode of a raw f32 matrix, then a second Tetra word
// on the residual of the first, in the rotated basis. Stage 2 uses the same
// pieces as stage 1: frozen row scales, two gain centroids, retraction.
use llvq_quant::gptq::{quantize_layer_parallel_capturing, GptqConfig, TailPolicy, Weights};
use llvq_quant::linalg::GptqFactor;
use llvq_quant::quantizer::{fit_gain_centroids, BlockCode, BlockQuantizer, TetraShapeGain};
use llvq_quant::rotation::Rotation;
use llvq_search::tetra::{Encoder, Tetra};
use std::sync::Arc;

fn stage(w: &[f64], rows: usize, d_in: usize, factor: &GptqFactor, cfg: &GptqConfig, enc: &Arc<Encoder>) -> (Vec<f64>, Vec<f64>) {
    let mut weights = Weights::new(rows, d_in, w.to_vec());
    let centroids = fit_gain_centroids(&weights.w, rows, d_in, 24, 1, 40);
    let (e, levels) = (enc.clone(), centroids.clone());
    let make = move || -> Box<dyn BlockQuantizer> { Box::new(TetraShapeGain::with_encoder(e.clone(), levels.clone())) };
    let mut codes: Vec<Option<BlockCode>> = vec![None; rows * (d_in / 24)];
    quantize_layer_parallel_capturing(&mut weights, factor, None, &make, cfg, 4, Some(&mut codes));
    (weights.w, centroids)
}
fn rel(a: &[f64], b: &[f64]) -> f64 {
    let (mut e, mut n) = (0.0, 0.0);
    for (x, y) in a.iter().zip(b) { e += (x - y) * (x - y); n += x * x; }
    e / n
}
fn block_cv(w: &[f64]) -> f64 {
    // spread of the 24-block norms relative to their row's mean block norm is what one gain bit has to cover
    let n: Vec<f64> = w.chunks_exact(24).map(|b| b.iter().map(|v| v * v).sum::<f64>().sqrt()).collect();
    let m = n.iter().sum::<f64>() / n.len() as f64;
    (n.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / n.len() as f64).sqrt() / m
}
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (path, d_in, max_rows): (&str, usize, usize) = (&a[1], a[2].parse().unwrap(), a[3].parse().unwrap());
    let bytes = std::fs::read(path).unwrap();
    let rows = (bytes.len() / 4 / d_in).min(max_rows);
    let mut w0: Vec<f64> = bytes[..rows * d_in * 4].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64).collect();
    let mut h = vec![0.0f64; d_in * d_in];
    for i in 0..d_in { h[i * d_in + i] = 1.0; }
    let factor = GptqFactor::new(&h, d_in, 1e-2).ok().expect("identity factors");
    let cfg = GptqConfig { block: 24, retract: true, group_scales: false, design_c: false, spherical_feedback: false, lambda: 1e-2, tail: TailPolicy::KeepExact };
    Rotation::new(d_in, 0x110FEED).rotate_weight_rows(&mut w0, rows);
    let tetra = Tetra::new();
    let enc = Arc::new(Encoder::new(&tetra));
    let (q1, c1) = stage(&w0, rows, d_in, &factor, &cfg, &enc);
    let r1: Vec<f64> = w0.iter().zip(&q1).map(|(x, y)| x - y).collect();
    let (q2, c2) = stage(&r1, rows, d_in, &factor, &cfg, &enc);
    let s2: Vec<f64> = q1.iter().zip(&q2).map(|(x, y)| x + y).collect();
    let r2: Vec<f64> = w0.iter().zip(&s2).map(|(x, y)| x - y).collect();
    let (q3, _) = stage(&r2, rows, d_in, &factor, &cfg, &enc);
    let s3: Vec<f64> = s2.iter().zip(&q3).map(|(x, y)| x + y).collect();
    println!("{path} rows {rows} x {d_in}: one word (2 b/w) {:.5} | two words (4 b/w) {:.5} | three words (6 b/w) {:.5} | stage-2 own rel error {:.5} | block-norm CV: weights {:.3}, residual {:.3} | centroids {:?} then {:?}",
        rel(&w0, &q1), rel(&w0, &s2), rel(&w0, &s3), rel(&r1, &q2), block_cv(&w0), block_cv(&r1), c1, c2);
}
