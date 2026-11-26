// Data-free (H = I) encode of a raw f32 matrix with the sealed Tetra recipe's
// pieces: rotation, frozen row scales, two gain centroids, spherical retraction.
use llvq_quant::gptq::{quantize_layer_parallel_capturing, GptqConfig, TailPolicy, Weights};
use llvq_quant::linalg::GptqFactor;
use llvq_quant::quantizer::{fit_gain_centroids, BlockCode, BlockQuantizer, TetraShapeGain};
use llvq_quant::rotation::Rotation;
use llvq_search::tetra::{Encoder, Tetra};
use std::sync::Arc;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (path, d_out, d_in, max_rows): (&str, usize, usize, usize) =
        (&a[1], a[2].parse().unwrap(), a[3].parse().unwrap(), a[4].parse().unwrap());
    let rot = a.get(5).map(|s| s == "rot").unwrap_or(true);
    let bytes = std::fs::read(path).unwrap();
    let rows = (bytes.len() / 4 / d_in).min(max_rows);
    let _ = d_out;
    let w0: Vec<f64> = bytes[..rows * d_in * 4]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
        .collect();
    let mut h = vec![0.0f64; d_in * d_in];
    for i in 0..d_in { h[i * d_in + i] = 1.0; }
    let factor = GptqFactor::new(&h, d_in, 1e-2).ok().expect("identity factors");
    let cfg = GptqConfig { block: 24, retract: true, group_scales: false, design_c: false,
        spherical_feedback: false, lambda: 1e-2, tail: TailPolicy::KeepExact };
    let mut weights = Weights::new(rows, d_in, w0.clone());
    let q = Rotation::new(d_in, 0x110FEED);
    if rot { q.rotate_weight_rows(&mut weights.w, rows); }
    let centroids = fit_gain_centroids(&weights.w, rows, d_in, 24, 1, 40);
    let tetra = Tetra::new();
    let enc = Arc::new(Encoder::new(&tetra));
    let levels = centroids.clone();
    let make = move || -> Box<dyn BlockQuantizer> { Box::new(TetraShapeGain::with_encoder(enc.clone(), levels.clone())) };
    let mut codes: Vec<Option<BlockCode>> = vec![None; rows * (d_in / 24)];
    let t = std::time::Instant::now();
    quantize_layer_parallel_capturing(&mut weights, &factor, None, &make, &cfg, 4, Some(&mut codes));
    if rot { q.unrotate_weight_rows(&mut weights.w, rows); }
    let (mut e, mut n) = (0.0, 0.0);
    let mut worst: f64 = 0.0; let mut rel_rows = Vec::with_capacity(rows);
    for i in 0..rows {
        let (mut ei, mut ni) = (0.0, 0.0);
        for j in 0..d_in { let (x, y) = (w0[i * d_in + j], weights.w[i * d_in + j]); ei += (x - y) * (x - y); ni += x * x; }
        e += ei; n += ni; let r = if ni > 0.0 { ei / ni } else { 0.0 }; worst = worst.max(r); rel_rows.push(r);
    }
    rel_rows.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("{path} rows {rows} x {d_in} rot {rot}: rel MSE (energy-weighted) {:.5}  median row {:.5}  p99 row {:.5}  worst row {:.5}  centroids {:?}  {:.1}s",
        e / n, rel_rows[rows / 2], rel_rows[rows * 99 / 100], worst, centroids, t.elapsed().as_secs_f64());
}
