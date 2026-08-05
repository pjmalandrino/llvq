//! The identity the whole fused inference path rests on.
//!
//! The sealed artifact stores weights in a **rotated** basis: the quantizer
//! computed `W' = W Qᵀ`, and `decode_matrix` un-rotates on the way out, which
//! is why the dense path needs nothing extra. A fused kernel does not
//! un-rotate — it reads `W'` as stored — so what it must compute is
//!
//! ```text
//! y = W x = (W Qᵀ)(Q x) = W' · rot(x)
//! ```
//!
//! This test is that equation, end to end, on the CPU:
//!
//!   * left side — `decode_matrix`, i.e. exactly the weights `bin/run` feeds
//!     to candle today;
//!   * right side — `transcode` to `Slot32`, `decode_block` per block, the
//!     tail applied verbatim, against a **rotated activation**.
//!
//! ## Why this had to exist before any GPU code
//!
//! Get the rotation backwards — `apply` where `apply_transpose` belongs, or
//! the rotation applied to the output instead of the input — and every row is
//! finite, plausible, and wrong. There is no crash, no NaN, no assertion: the
//! model simply produces worse text. Discovering that from a perplexity number
//! after writing a kernel, a loader and a forward pass would cost days; this
//! costs a second and needs no card.
//!
//! ## What it deliberately does NOT test
//!
//! The GPU. `tests/rotation_matches_rust.rs` in `llvq-cuda` pins the rotation
//! kernel against `Rotation`, and `bin/matvec` pins `slot_dot` against an f64
//! reference. This test pins the *composition* those two are supposed to
//! implement — which neither of them can see.

use llvq_artifact::runtime::{transcode, ClassTable, Layout};
use llvq_artifact::{decode_matrix, QuantizedMatrix};
use llvq_core::{SplitMix64, DIM};
use llvq_quant::quantizer::BlockCode;
use llvq_quant::rotation::Rotation;
use llvq_search::fastdec::FastDecoder;
use llvq_search::index::{Indexer, N13};

/// `(d_out, d_in, seed, what it is)`. `d_in % 24 != 0` on purpose in half the
/// cases: the tail is applied in the rotated basis too, and getting that wrong
/// is invisible on a width that has no tail.
const CASES: [(usize, usize, u64, &str); 5] = [
    (8, 240, 0x11, "no tail, k=1 (240 = 16·15 → m=16)"),
    (8, 2560, 0x22, "Qwen3-4B hidden, tail 16, k=5"),
    (16, 9728, 0x33, "Qwen3-4B down_proj input, tail 8, k=19"),
    (8, 4096, 0x44, "power of two, no tail, k=1"),
    (24, 96, 0x55, "small, tail 0, k=3"),
];

/// Build a quantized matrix whose indices and codes are the *same* object seen
/// two ways — the file carries indices, the dense decoder wants points.
fn make(d_out: usize, d_in: usize, seed: u64) -> (QuantizedMatrix, Vec<u64>, Vec<u32>) {
    let ix = Indexer::new();
    let mut rng = SplitMix64::new(seed);
    let nblocks = d_in / DIM;
    let tail_w = d_in % DIM;

    let mut indices = Vec::with_capacity(d_out * nblocks);
    let mut gains = Vec::with_capacity(d_out * nblocks);
    let mut codes = Vec::with_capacity(d_out * nblocks);
    for _ in 0..d_out * nblocks {
        // 1 + … skips index 0, the origin: a matrix of origins would satisfy
        // any rotation whatsoever, since 0 = Q·0.
        let idx = 1 + rng.next() % (N13 - 1);
        let point = ix.decode(idx).expect("index in the cap-13 ball");
        let gain = (rng.next() & 1) as u32;
        indices.push(idx);
        gains.push(gain);
        codes.push(BlockCode { point, gain });
    }

    (
        QuantizedMatrix {
            name: format!("test.{d_out}x{d_in}"),
            d_out,
            d_in,
            codes,
            // Deliberately not all equal: a row scale folded in the wrong
            // place cancels out when every row shares one.
            row_scales: (0..d_out).map(|_| 0.5 + rng.next_gaussian().abs()).collect(),
            centroids: vec![0.625, 1.375],
            rotation_seed: Some(seed ^ 0xbeef),
            shell_cap: 13,
            tail: (0..d_out * tail_w).map(|_| rng.next_gaussian()).collect(),
        },
        indices,
        gains,
    )
}

/// `y = W'·rot(x)` — what a fused kernel computes, in f64 on the CPU.
fn fused_product(
    m: &QuantizedMatrix,
    indices: &[u64],
    gains: &[u32],
    x: &[f64],
) -> Vec<f64> {
    let fd = FastDecoder::new();
    let table = ClassTable::new(&fd, 1);
    let rt = transcode(&fd, &table, indices, gains, Layout::Slot32).expect("transcode");

    let nblocks = m.d_in / DIM;
    let tail_w = m.d_in % DIM;

    // The activation, rotated. `apply`, not `apply_transpose`: the weights
    // were rotated by `rotate_weight_rows`, which is `W ← W Qᵀ`, so the input
    // owes the matching `Q`.
    let mut xr = x.to_vec();
    if let Some(seed) = m.rotation_seed {
        Rotation::new(m.d_in, seed).apply(&mut xr);
    }

    let mut y = vec![0.0f64; m.d_out];
    for (row, yv) in y.iter_mut().enumerate() {
        let mut acc = 0.0;
        for p in 0..nblocks {
            let (point, gain) = rt.decode_block(&table, row * nblocks + p);
            if let Some(shell) = llvq_core::Leech::shell_index(&point).filter(|&s| s > 0) {
                let k = m.centroids[gain as usize] * m.row_scales[row]
                    / ((16 * shell) as f64).sqrt();
                for (i, &v) in point.iter().enumerate() {
                    acc += v as f64 * k * xr[p * DIM + i];
                }
            }
        }
        for t in 0..tail_w {
            acc += m.tail[row * tail_w + t] * xr[nblocks * DIM + t];
        }
        *yv = acc;
    }
    y
}

#[test]
#[cfg_attr(debug_assertions, ignore = "transcodes five matrices, run in release")]
fn the_fused_path_computes_what_the_dense_path_computes() {
    let mut worst = 0.0f64;
    for (d_out, d_in, seed, what) in CASES {
        let (m, indices, gains) = make(d_out, d_in, seed);

        // Left side: the weights `bin/run` uses today.
        let w = decode_matrix(&m);
        let mut rng = SplitMix64::new(seed ^ 0xa11ce);
        let x: Vec<f64> = (0..d_in).map(|_| rng.next_gaussian()).collect();
        let dense: Vec<f64> = (0..d_out)
            .map(|r| (0..d_in).map(|c| w[r * d_in + c] as f64 * x[c]).sum())
            .collect();

        // Right side: stored weights against a rotated activation.
        let got = fused_product(&m, &indices, &gains, &x);

        // `decode_matrix` narrows to f32 on the way out, so the two sides
        // cannot be bit-equal; the scale is the sum of |contributions|, as
        // everywhere else in this repository.
        for r in 0..d_out {
            let scale: f64 = (0..d_in)
                .map(|c| (w[r * d_in + c] as f64 * x[c]).abs())
                .sum::<f64>()
                .max(1e-12);
            let e = (dense[r] - got[r]).abs() / scale;
            assert!(
                e < 1e-6,
                "{what}, ligne {r} : dense {} contre fusé {} ({e:.3e}·Σ|w·x|)",
                dense[r],
                got[r]
            );
            worst = worst.max(e);
        }
    }
    eprintln!("pire écart sur {} formes : {worst:.3e}·Σ|w·x|", CASES.len());
}

/// The rotation is not decorative: without it the two sides must disagree.
///
/// This is the assertion that makes the one above load-bearing. A test that
/// only checks agreement would pass just as well on a matrix quantized in the
/// natural basis, where `Q = I` — and every shape in the sealed artifact
/// carries a real rotation, so that would be exactly the wrong thing to prove.
#[test]
#[cfg_attr(debug_assertions, ignore = "transcodes a matrix, run in release")]
fn skipping_the_rotation_is_detected() {
    let (m, indices, gains) = make(8, 2560, 0x77);
    let w = decode_matrix(&m);
    let mut rng = SplitMix64::new(0x5a17);
    let x: Vec<f64> = (0..m.d_in).map(|_| rng.next_gaussian()).collect();
    let dense: Vec<f64> = (0..m.d_out)
        .map(|r| (0..m.d_in).map(|c| w[r * m.d_in + c] as f64 * x[c]).sum())
        .collect();

    let mut unrotated = m;
    unrotated.rotation_seed = None; // the kernel's view if nobody rotates `x`
    let got = fused_product(&unrotated, &indices, &gains, &x);

    let rel: f64 = dense
        .iter()
        .zip(&got)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
        / dense.iter().map(|v| v.abs()).fold(0.0, f64::max).max(1e-12);
    assert!(
        rel > 0.1,
        "sans rotation l'écart n'est que de {rel:.3e} — le test principal ne prouve rien"
    );
}
