// DRAFT, never compiled or run. Revised 2026-11-02 after review.
//
// Sketch of the measurement mode of prereg-brick-1.md: `LLVQ_ACTQ=<arms>` in `ppl` (dense path,
// Metal, f32) and `LLVQ_ACTQ_STATS=<dir>`. Refused beside `LLVQ_CONFIG`, like every measurement
// mode. What the review changed:
//   * Q comes from each Tetra record's own stored seed (`QuantizedMatrix::rotation_seed`,
//     llvq-artifact/src/format.rs:484), the seed `decode_matrix` un-rotated it with (format.rs:1527).
//     `Proj::Dense` carries no seed (llvq-llm/src/sealed.rs:116-117), so the loader keeps it beside
//     each dense Tetra projection. `effective_rotation_seed(ROTATION_SEED, block, act)` is only a
//     cross-check, printed per record (prereg check (a)).
//   * `QtQ = I` makes the transparency arm blind to a wrong basis, so `basis_check` ties the hook's Q
//     to the Q that rotated the weights (prereg check (b), gate 6).
//   * `ppl` reads no `LLVQ_THREADS` today; the injection pool reads it here and prints it (gate 9).
//   * `Off` with no statistics returns before touching the row, so the off arm is the hook-free run
//     (gate 7 checks it bit for bit).
// The int4 inputs (v, o, down in layers 12-23) and the head are never touched.

/// Which arm a window runs under.
#[derive(Clone, Copy, PartialEq)]
pub enum ActQ {
    Off,   // published
    IdRot, // rotate then unrotate: the transparency gate
    I16s,  // brick default: int16 per 384-slice, power-of-two exponent, spread capped at 3
    I24s,  // fallback: int24 per slice
    I8v,   // positive control: int8 per vector, amax/127 (ExLlamaV3 plain mode)
}

const SLICE: usize = 384;
const CAP: i32 = 3;

/// Exact binary exponent of a positive finite f32.
fn exp2_of(a: f32) -> i32 {
    let b = a.to_bits();
    let e = ((b >> 23) & 0xff) as i32;
    if e == 0 { -127 - ((b & 0x7f_ffff).leading_zeros() as i32 - 9) } else { e - 127 }
}

/// The brick quantizer on one rotated token, in place: x <- dequant(quant(x)).
/// `bits` = 16 or 24. Matches t48i_reference.hpp::quantize on the block coordinates;
/// the caller passes only the 24 * nblocks block coordinates (the f16 tail is excluded).
fn quant_slices(x: &mut [f32], bits: u32) {
    let qmax = (1i64 << (bits - 1)) - 1;
    let n = x.len();
    let ns = (n + SLICE - 1) / SLICE;
    let mut e = vec![i32::MIN; ns];
    for s in 0..ns {
        let seg = &x[s * SLICE..n.min((s + 1) * SLICE)];
        let amax = seg.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        if amax > 0.0 && amax.is_finite() {
            e[s] = exp2_of(amax) - (bits as i32 - 2);
        }
    }
    let e_hi = e.iter().copied().max().filter(|&v| v != i32::MIN).unwrap_or(0);
    let e_lo = e_hi - CAP;
    for s in 0..ns {
        let es = e[s].max(e_lo);
        let (down, up) = (f32::powi(2.0, -es), f32::powi(2.0, es)); // exact powers of two
        for v in &mut x[s * SLICE..n.min((s + 1) * SLICE)] {
            let q = ((*v * down).round_ties_even() as i64).clamp(-qmax, qmax);
            *v = q as f32 * up;
        }
    }
}

fn quant_vector_int8(x: &mut [f32]) {
    let amax = x.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-8);
    let q = amax / 127.0;
    for v in x.iter_mut() {
        *v = ((*v / q).round_ties_even()).clamp(-127.0, 127.0) * q;
    }
}

/// Per-token statistics for LLVQ_ACTQ_STATS, rotated basis: slice amax/rms, vector amax/rms,
/// and the relative errors of int16 and int24 against the f16 rounding of the same token.
pub struct TokenStats {
    pub slice_amax_over_rms: Vec<f32>,
    pub vec_amax_over_rms: f32,
    pub rel_err_i16: f64,
    pub rel_err_i24: f64,
    pub rel_err_f16: f64,
}

/// The hook, applied to every row of a Tetra projection's input before the dense matmul.
/// `rot` is built once at load from that record's stored seed: `Rotation::new(d_in, seed)`.
/// `nblock_coords` = 24 * nblocks (the f16 tail columns are excluded, as in the kernel).
pub fn inject(row: &mut [f32], rot: &llvq_quant::rotation::Rotation, nblock_coords: usize,
              arm: ActQ, stats: Option<&mut Vec<TokenStats>>) {
    if arm == ActQ::Off && stats.is_none() {
        return;                                           // untouched: the hook-free run
    }
    let mut v: Vec<f64> = row.iter().map(|&a| a as f64).collect();
    rot.apply(&mut v);                                    // into the basis the weights live in
    let mut r: Vec<f32> = v.iter().map(|&a| a as f32).collect(); // the kernel sees f32
    if let Some(st) = stats {
        st.push(token_stats(&r[..nblock_coords]));
    }
    match arm {
        ActQ::Off => return,                              // statistics only, row untouched
        ActQ::IdRot => {}
        ActQ::I16s => quant_slices(&mut r[..nblock_coords], 16),
        ActQ::I24s => quant_slices(&mut r[..nblock_coords], 24),
        ActQ::I8v => quant_vector_int8(&mut r[..nblock_coords]),
    }
    let mut back: Vec<f64> = r.iter().map(|&a| a as f64).collect();
    rot.apply_transpose(&mut back);                       // W' Q x~ = W Q^T (Q x~): the dense W
    for (o, b) in row.iter_mut().zip(back) {
        *o = b as f32;
    }
}

/// Check (b) of the prereg, at load, for one block of one record per projection type: the dense
/// natural-basis rows rebuilt by `decode_matrix` (W_dense = W_rot Q), times Q^T, must give back the
/// rotated-basis block that `reconstruct_shape_gain` decodes. A wrong seed or keying fails here,
/// where the transparency arm (Q then Q^T) cannot see it. Returns the worst relative error.
pub fn basis_check(w_dense_row: &[f32], rot: &llvq_quant::rotation::Rotation,
                   block: usize, decoded_rot_block: &[f64; 24]) -> f64 {
    // (W_dense Q^T) row = Q (W_dense row) for a row vector: apply Q to the row.
    let mut v: Vec<f64> = w_dense_row.iter().map(|&a| a as f64).collect();
    rot.apply(&mut v);
    let got = &v[24 * block..24 * block + 24];
    let norm = decoded_rot_block.iter().map(|a| a * a).sum::<f64>().sqrt().max(1e-300);
    got.iter().zip(decoded_rot_block).map(|(a, b)| (a - b).abs() / norm).fold(0.0, f64::max)
}

/// The injection pool size: `LLVQ_THREADS`, default ncpu - 4, at least 1. Printed in the run header
/// as `actq threads N`; anything that is not a positive integer is refused by name.
pub fn actq_threads() -> anyhow::Result<usize> {
    match std::env::var("LLVQ_THREADS") {
        Ok(s) => {
            let n: usize = s.parse().map_err(|e| anyhow::anyhow!("LLVQ_THREADS={s:?} is not a number: {e}"))?;
            anyhow::ensure!(n > 0, "LLVQ_THREADS={s:?}: at least one is needed");
            Ok(n)
        }
        Err(_) => Ok(std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8).saturating_sub(4).max(1)),
    }
}

fn token_stats(r: &[f32]) -> TokenStats {
    let rms = |s: &[f32]| (s.iter().map(|&a| (a as f64) * (a as f64)).sum::<f64>() / s.len() as f64).sqrt();
    let amax = |s: &[f32]| s.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let slice_amax_over_rms = r.chunks(SLICE).map(|s| (amax(s) as f64 / rms(s).max(1e-30)) as f32).collect();
    let rel = |q: &[f32]| {
        let num: f64 = r.iter().zip(q).map(|(&a, &b)| ((a - b) as f64).powi(2)).sum();
        let den: f64 = r.iter().map(|&a| (a as f64).powi(2)).sum();
        (num / den.max(1e-300)).sqrt()
    };
    let mut q16 = r.to_vec(); quant_slices(&mut q16, 16);
    let mut q24 = r.to_vec(); quant_slices(&mut q24, 24);
    let f16r: Vec<f32> = r.iter().map(|&a| half::f16::from_f32(a).to_f32()).collect();
    TokenStats {
        slice_amax_over_rms,
        vec_amax_over_rms: (amax(r) as f64 / rms(r).max(1e-30)) as f32,
        rel_err_i16: rel(&q16),
        rel_err_i24: rel(&q24),
        rel_err_f16: rel(&f16r),
    }
}
