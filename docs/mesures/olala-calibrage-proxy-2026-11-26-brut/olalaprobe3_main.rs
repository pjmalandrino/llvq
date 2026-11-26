// Real Tetra encode, data-free (H = I), of every expert of one tensor, one word then a second on the residual.
// Writes the two dense reconstructions in the natural basis as raw f32.
// usage: olalaprobe3 <in.f32> <experts> <d_out> <d_in> <layer> <g> <out_prefix>
use llvq_quant::gptq::{quantize_layer_parallel_capturing, GptqConfig, TailPolicy, Weights};
use llvq_quant::linalg::GptqFactor;
use llvq_quant::quantizer::{fit_gain_centroids, BlockCode, BlockQuantizer, TetraShapeGain};
use llvq_quant::rotation::Rotation;
use llvq_search::tetra::{Encoder, Tetra};
use std::io::Write;
use std::sync::Arc;

fn stage(w: &[f64], rows: usize, d_in: usize, factor: &GptqFactor, cfg: &GptqConfig, enc: &Arc<Encoder>) -> Vec<f64> {
    let mut weights = Weights::new(rows, d_in, w.to_vec());
    let centroids = fit_gain_centroids(&weights.w, rows, d_in, 24, 1, 40);
    let (e, levels) = (enc.clone(), centroids);
    let make = move || -> Box<dyn BlockQuantizer> { Box::new(TetraShapeGain::with_encoder(e.clone(), levels.clone())) };
    let mut codes: Vec<Option<BlockCode>> = vec![None; rows * (d_in / 24)];
    quantize_layer_parallel_capturing(&mut weights, factor, None, &make, cfg, 4, Some(&mut codes));
    assert!(codes.iter().all(|c| c.is_some()), "a block emitted no code");
    weights.w
}
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let p = |i: usize| -> usize { a[i].parse().unwrap() };
    let (path, ne, d_out, d_in, layer, g, out) = (&a[1], p(2), p(3), p(4), p(5) as u64, p(6) as u64, &a[7]);
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(bytes.len(), ne * d_out * d_in * 4);
    let mut h = vec![0.0f64; d_in * d_in];
    for i in 0..d_in { h[i * d_in + i] = 1.0; }
    let factor = GptqFactor::new(&h, d_in, 1e-2).ok().expect("identity factors");
    let cfg = GptqConfig { block: 24, retract: true, group_scales: false, design_c: false, spherical_feedback: false, lambda: 1e-2, tail: TailPolicy::KeepExact };
    let tetra = Tetra::new();
    let enc = Arc::new(Encoder::new(&tetra));
    let mut f1 = std::io::BufWriter::new(std::fs::File::create(format!("{out}.w1.f32")).unwrap());
    let mut f2 = std::io::BufWriter::new(std::fs::File::create(format!("{out}.w2.f32")).unwrap());
    let (mut n, mut e1, mut e2) = (0.0f64, 0.0f64, 0.0f64);
    let t = std::time::Instant::now();
    let per = d_out * d_in;
    for e in 0..ne {
        let w0: Vec<f64> = bytes[e * per * 4..(e + 1) * per * 4].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64).collect();
        let q = Rotation::new(d_in, 0x110FEED ^ (layer << 32) ^ ((e as u64) << 20) ^ (g << 16));
        let mut wr = w0.clone();
        q.rotate_weight_rows(&mut wr, d_out);
        let mut q1 = stage(&wr, d_out, d_in, &factor, &cfg, &enc);
        let r1: Vec<f64> = wr.iter().zip(&q1).map(|(x, y)| x - y).collect();
        let q2 = stage(&r1, d_out, d_in, &factor, &cfg, &enc);
        let mut s2: Vec<f64> = q1.iter().zip(&q2).map(|(x, y)| x + y).collect();
        q.unrotate_weight_rows(&mut q1, d_out);
        q.unrotate_weight_rows(&mut s2, d_out);
        for i in 0..per {
            n += w0[i] * w0[i]; e1 += (w0[i] - q1[i]).powi(2); e2 += (w0[i] - s2[i]).powi(2);
            f1.write_all(&(q1[i] as f32).to_le_bytes()).unwrap();
            f2.write_all(&(s2[i] as f32).to_le_bytes()).unwrap();
        }
        if (e + 1) % 32 == 0 || e + 1 == ne {
            println!("L{layer} g{g} expert {}/{ne}: rel error one word {:.5}, two words {:.5}, {:.0} s", e + 1, e1 / n, e2 / n, t.elapsed().as_secs_f64());
        }
    }
}
