//! Stage 1 proper of the E8 arbitration: both arms through the SAME GPTQ loop.
//!
//! ```text
//! e8gptq <h.f64le> <rows.f64le> <n> <rows> <cap> <damping> <c0> <c1>
//! ```
//!
//! ## What the void run got wrong
//!
//! `docs/mesures/e8-etape1-2026-09-18.txt` compared a GPTQ witness against a
//! plain per-block encode, so its ratio measured the compensation and not the
//! codebook (deviation E3). This bin removes that confound the only way that
//! works: `llvq_quant::gptq::quantize_layer` runs both arms, with the same
//! `GptqFactor`, the same `GptqConfig`, the same gain centroids and the same
//! row scale, and the only difference left is which `BlockQuantizer` it calls.
//!
//! The trait's own doc invites this: the loop is meant to be exercised against
//! codebooks that have nothing to do with the Leech lattice.
//!
//! ## What it prints
//!
//! One TSV line per row and arm, for the wrapper to aggregate: the unweighted
//! squared error, the Hessian-weighted error, the isotropic reference for that
//! error energy, and the mean cosine over the row's blocks. The isotropic
//! reference is exact, `E[e'He] = |e|^2 tr(H)/n`, and it is what separates how
//! much error an arm makes from where it puts it.

use llvq_bench::e8::E8Cubed;
use llvq_quant::gptq::{quantize_layer, GptqConfig, TailPolicy, Weights};
use llvq_quant::linalg::GptqFactor;
use llvq_quant::quantizer::{BlockQuantizer, TetraShapeGain};

const DIM: usize = 24;

fn read_f64(path: &str, want: usize) -> Vec<f64> {
    let b = std::fs::read(path).expect("read the array");
    assert_eq!(b.len(), want * 8, "{path}: {} bytes for {want} f64", b.len());
    b.chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().expect("8 bytes")))
        .collect()
}

/// `e' H e` without forming `H e`.
fn quad(h: &[f64], n: usize, e: &[f64]) -> f64 {
    (0..n)
        .map(|i| {
            if e[i] == 0.0 {
                0.0
            } else {
                e[i] * (0..n).map(|j| h[i * n + j] * e[j]).sum::<f64>()
            }
        })
        .sum()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    assert_eq!(a.len(), 9, "usage: e8gptq H ROWS n rows cap damping c0 c1");
    let n: usize = a[3].parse().expect("n");
    let nrows: usize = a[4].parse().expect("rows");
    let cap: i32 = a[5].parse().expect("cap");
    let damping: f64 = a[6].parse().expect("damping");
    let cent = vec![a[7].parse().expect("c0"), a[8].parse().expect("c1")];

    let h = read_f64(&a[1], n * n);
    let rows = read_f64(&a[2], nrows * n);
    let factor = GptqFactor::new(&h, n, damping).expect("H is not positive definite");
    let trn = (0..n).map(|i| h[i * n + i]).sum::<f64>() / n as f64;

    let cfg = GptqConfig {
        block: DIM,
        retract: true,
        group_scales: false,
        design_c: false,
        lambda: 1e-2,
        tail: TailPolicy::KeepExact,
    };
    let nb = n / DIM;
    let e8 = E8Cubed::new(cap, cent.clone());
    let (pts, bits) = e8.size();
    println!("# e8gptq  n {n}  rows {nrows}  blocks {nb}  cap {cap}  damping {damping}");
    println!(
        "# arm B codebook {pts} points, {bits} bits an index, {} b/weight with one gain bit",
        (3 * bits + 1) as f64 / DIM as f64
    );
    println!("# row\tarm\terr\tjloc\tjiso\tcos");

    for r in 0..nrows {
        let orig = &rows[r * n..(r + 1) * n];
        let den = quad(&h, n, orig);
        for arm in ["tetra", "e8cubed"] {
            let mut q: Box<dyn BlockQuantizer> = if arm == "tetra" {
                Box::new(TetraShapeGain::with_encoder(
                    TetraShapeGain::encoder(),
                    cent.clone(),
                ))
            } else {
                Box::new(E8Cubed::new(cap, cent.clone()))
            };
            let mut w = Weights::new(1, n, orig.to_vec());
            quantize_layer(&mut w, &factor, None, q.as_mut(), &cfg);
            let e: Vec<f64> = (0..n).map(|i| orig[i] - w.w[i]).collect();
            let err: f64 = e[..nb * DIM].iter().map(|v| v * v).sum();
            let jloc = quad(&h, n, &e) / den;
            let jiso = e.iter().map(|v| v * v).sum::<f64>() * trn / den;
            let cos: f64 = (0..nb)
                .map(|b| {
                    let (x, y) = (&orig[b * DIM..(b + 1) * DIM], &w.w[b * DIM..(b + 1) * DIM]);
                    let d: f64 = x.iter().zip(y).map(|(p, q_)| p * q_).sum();
                    let nx: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
                    let ny: f64 = y.iter().map(|v| v * v).sum::<f64>().sqrt();
                    if nx == 0.0 || ny == 0.0 { 1.0 } else { d / (nx * ny) }
                })
                .sum::<f64>()
                / nb as f64;
            println!("{r}\t{arm}\t{err:.12e}\t{jloc:.12e}\t{jiso:.12e}\t{cos:.9}");
        }
    }
}
