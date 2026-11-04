// DRAFT, never compiled or run. CPU replay of the canonical attention arithmetic of
// wave5-deep/attention.md section 3.1, revision 2, sketched for llvq-llm (behind a device trait,
// as the CPU fake of `AttnKernel`). Not repository code.
//
// Purpose: the CUDA kernel uses explicit round-to-nearest intrinsics and a polynomial exp2, so
// this replay must return the card's bits exactly. Rust f32 `*`, `+`, `-`, `/` are IEEE RN and
// never contracted; `f32::mul_add` is a fused multiply-add like `__fmaf_rn`; `half::f16`
// converts with round-to-nearest-even like `cvt.rn.f16.f32`.
//
// The functions mirror attn_dec_sketch.cu one for one (dot8, tree16, tile_max_carry, tile_p,
// tile_accumulate, st_merge, deq_r1, quant_codes) so that the host probe can diff each of them
// against the kernel TEXT compiled by clang++, before any card runs it (METHODE section 4).

use half::f16;

pub const HD: usize = 128;
pub const LANES: usize = 16;
pub const DPL: usize = 8;
pub const TT: usize = 8; // the bench may time 4 before the freeze
pub const UNIT: usize = 64; // the bench may time 32 before the freeze
pub const SPLIT: usize = 256;

// Same placeholder coefficients as the kernel (Taylor, about 1e-5 relative error at f -> 1);
// freeze them in both files at once from a minimax fit.
const C: [f32; 6] = [0.693147182, 0.240226507, 0.0555041087, 0.00961812911, 0.00133335581, 0.000154035304];

pub fn ex2_fixed(x: f32) -> f32 {
    if x.is_nan() {
        return x; // NaN propagates, as in the kernel
    }
    if !(x > -126.0) {
        return 0.0;
    }
    let fl = x.floor();
    let f = x - fl;
    let mut p = C[5];
    for &c in C[..5].iter().rev() {
        p = p.mul_add(f, c);
    }
    p = p.mul_add(f, 1.0);
    let e = fl as i32; // in [-126, 0]
    f32::from_bits(p.to_bits().wrapping_add((e as u32) << 23)) // the kernel's unsigned wrap
}

#[derive(Clone)]
pub struct St {
    pub m: Vec<f32>,       // [G]
    pub d: Vec<f32>,       // [G]
    pub a: Vec<[f32; HD]>, // [G][HD], the 16 lanes' slices side by side
}

impl St {
    pub fn empty(g: usize) -> Self {
        St { m: vec![f32::NEG_INFINITY; g], d: vec![0.0; g], a: vec![[0.0; HD]; g] }
    }
}

pub fn st_merge(a: &mut St, b: &St) {
    for g in 0..a.m.len() {
        if b.m[g] == f32::NEG_INFINITY {
            continue;
        }
        if a.m[g] == f32::NEG_INFINITY {
            a.m[g] = b.m[g];
            a.d[g] = b.d[g];
            a.a[g] = b.a[g];
            continue;
        }
        let m = a.m[g].max(b.m[g]);
        let fa = ex2_fixed(a.m[g] - m);
        let fb = ex2_fixed(b.m[g] - m);
        a.d[g] = a.d[g] * fa + b.d[g] * fb; // two RN products, one RN sum: no fma
        for i in 0..HD {
            a.a[g][i] = a.a[g][i] * fa + b.a[g][i] * fb;
        }
        a.m[g] = m;
    }
}

/// One lane's partial: an fma chain over its 8 dims, from +0.
pub fn dot8(q: &[f32; HD], k: &[f16; HD], lane: usize) -> f32 {
    let mut t = 0f32;
    for i in 0..DPL {
        let d = lane * DPL + i;
        t = q[d].mul_add(k[d].to_f32(), t);
    }
    t
}

/// The canonical tree over 16 partials: offsets 8, 4, 2, 1, pairwise RN sums, read at lane 0.
pub fn tree16(v: [f32; LANES]) -> f32 {
    let mut x = v;
    for off in [8usize, 4, 2, 1] {
        let y = x;
        for l in 0..LANES {
            x[l] = y[l] + y[l ^ off];
        }
    }
    x[0]
}

pub fn score(q: &[f32; HD], k: &[f16; HD], c: f32) -> f32 {
    let mut parts = [0f32; LANES];
    for lane in 0..LANES {
        parts[lane] = dot8(q, k, lane);
    }
    tree16(parts) * c
}

/// (m_new, carry) for one head; `sc` holds the tile's nv valid scores.
pub fn tile_max_carry(m_old: f32, sc: &[f32]) -> (f32, f32) {
    let mut mt = sc[0];
    for &s in &sc[1..] {
        mt = mt.max(s);
    }
    if m_old == f32::NEG_INFINITY {
        (mt, 0.0)
    } else {
        let mn = m_old.max(mt);
        (mn, ex2_fixed(m_old - mn))
    }
}

pub fn tile_p(s: f32, m_new: f32) -> f32 {
    ex2_fixed(s - m_new)
}

/// d, ts (left fold in position order from +0) and the HD accumulators for one head.
pub fn tile_accumulate(d: &mut f32, acc: &mut [f32; HD], p: &[f32], carry: f32, v: &[[f16; HD]]) {
    let mut ts = 0f32;
    for &pj in p {
        ts += pj;
    }
    *d = *d * carry + ts;
    for i in 0..HD {
        let mut a = acc[i] * carry;
        for (j, &pj) in p.iter().enumerate() {
            a = pj.mul_add(v[j][i].to_f32(), a);
        }
        acc[i] = a;
    }
}

/// One tile of up to TT positions starting at absolute position p0 (p0 < l), all heads.
pub fn tile(s: &mut St, q: &[[f32; HD]], k: &dyn Fn(usize) -> [f16; HD], v: &dyn Fn(usize) -> [f16; HD],
            p0: usize, l: usize, c: f32) {
    let nv = TT.min(l - p0);
    let ks: Vec<[f16; HD]> = (0..nv).map(|j| k(p0 + j)).collect();
    let vs: Vec<[f16; HD]> = (0..nv).map(|j| v(p0 + j)).collect();
    for g in 0..q.len() {
        let sc: Vec<f32> = ks.iter().map(|kr| score(&q[g], kr, c)).collect();
        let (mn, carry) = tile_max_carry(s.m[g], &sc);
        let p: Vec<f32> = sc.iter().map(|&x| tile_p(x, mn)).collect();
        tile_accumulate(&mut s.d[g], &mut s.a[g], &p, carry, &vs);
        s.m[g] = mn;
    }
}

/// One row: absolute length `l` (positions 0..l), the canonical schedule, f16 output [G][HD].
pub fn row(q: &[[f32; HD]], k: &dyn Fn(usize) -> [f16; HD], v: &dyn Fn(usize) -> [f16; HD],
           l: usize, c: f32) -> Vec<[f16; HD]> {
    let g_n = q.len();
    let mut total = St::empty(g_n);
    for s0 in (0..l).step_by(SPLIT) {
        let mut split = St::empty(g_n);
        for u0 in (s0..s0 + SPLIT).step_by(UNIT) {
            if u0 >= l {
                break;
            }
            let mut unit = St::empty(g_n);
            for p0 in (u0..u0 + UNIT).step_by(TT) {
                if p0 >= l {
                    break;
                }
                tile(&mut unit, q, k, v, p0, l, c);
            }
            st_merge(&mut split, &unit);
        }
        st_merge(&mut total, &split);
    }
    (0..g_n)
        .map(|g| {
            let mut o = [f16::ZERO; HD];
            for d in 0..HD {
                o[d] = f16::from_f32(total.a[g][d] / total.d[g]);
            }
            o
        })
        .collect()
}

// ---------------------------------------------------------------- int8 store, R1 semantics
pub const INV255: f32 = (1.0f64 / 255.0) as f32;

/// kvq.rs's quantizer for one group of 64: (codes, s16, mn16). The min and max folds are exact;
/// a +-0 tie may store the other zero than candle does, with no effect on any dequantized value.
pub fn quant64(x: &[f16; 64]) -> ([u8; 64], f16, f16) {
    let xf: Vec<f32> = x.iter().map(|h| h.to_f32()).collect();
    let mn = xf.iter().cloned().fold(f32::INFINITY, f32::min);
    let mx = xf.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let s16 = f16::from_f32((mx - mn) * INV255);
    let mn16 = f16::from_f32(mn);
    let (s, mnr) = (s16.to_f32(), mn16.to_f32());
    let safe = if s > 0.0 { s } else { 1.0 };
    let mut codes = [0u8; 64];
    for i in 0..64 {
        codes[i] = (((xf[i] - mnr) / safe).round().clamp(0.0, 255.0)) as u8;
    }
    (codes, s16, mn16)
}

pub fn deq_r1(code: u8, s16: f16, mn16: f16) -> f16 {
    let (s, mn) = (s16.to_f32(), mn16.to_f32());
    if !(s > 0.0) {
        // The round trip returns (q*s + mn) * 0 + mn * 1: mn when mn != 0, +0 when mn is +-0.
        return if mn == 0.0 { f16::ZERO } else { mn16 };
    }
    f16::from_f32((code as f32) * s + mn) // two RN roundings, as candle's two kernels
}

// ---------------------------------------------------------------- tests to write (sketch)
// 1. rows_are_invariant: for L in {1, 7, 8, 9, 63, 64, 65, 255, 256, 257, 1000, 4096} and k in
//    1..=8, the k rows of one verify call (row r has length pos0 + r + 1) equal k separate decode
//    calls bit for bit. True on the CPU by construction; the same check, fed the card's outputs, is
//    the load-time and bench identity check, run for EVERY mapping variant (they must all agree).
// 2. against_f64: row() within 2e-6 relative of an f64 softmax attention on random inputs.
// 3. q8_store_is_the_round_trip: quant64 + deq_r1 equals candle's quantize_dequantize on the CPU
//    backend bit for bit on dequantized VALUES (inputs from kvq.rs's tests, plus a constant group,
//    a constant group of -0.0, a group mixing +0 and -0 at its minimum, and a group whose range
//    rounds to 0 in f16), and row() over deq_r1 values equals row() over the round-trip cache.
// 4. ex2_fixed: within 3 ulp of exp2 on [-126, 0] once the coefficients are fitted; 0 at -inf and
//    at -126; NaN for NaN; ex2_fixed(-0.0) == ex2_fixed(+0.0) == 1.0, bits included.
// 5. kernel_text_matches_replay (llvq-llm/tests, host probe): build host_attn_dec.cpp with clang++
//    -std=c++17 -O2 -ffp-contract=off -Wall -Wextra -Werror; feed it random q, K, V, scores and
//    states; require ex2_fixed (sweep including the edges of 4), dot8, tree16, tile_max_carry,
//    tile_p, tile_accumulate, st_merge, deq_r1 and quant_codes to return this file's bits. A missing
//    clang++ fails and names the tool (hard rule 10).
// Mutants each test must kill: split boundary relative to L instead of absolute; units merged in
// arrival order; fma in st_merge; tree16 levels reordered (1, 2, 4, 8); masked positions given
// p = 0 but still multiplied into acc with a stale V of inf; ts summed as a tree instead of a left
// fold; the quantizer's round() removed (kvq.rs's own mutant); deq_r1 fused into one fma; deq_r1
// returning mn16 for an ungraded -0 group; ex2_fixed masking NaN to 0. A signed shift in ex2_fixed
// may compute the right bits and still be undefined behaviour, so no value test kills it: one probe
// build with -fsanitize=undefined does (UBSan reports a left shift of a negative value).
