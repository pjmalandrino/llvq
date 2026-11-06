// The canonical decode-attention arithmetic, replayed on the CPU.
//
// `kernels/attn_dec.cu` computes split-KV grouped attention for decode and
// verify rows with explicit round-to-nearest operations and a polynomial
// exp2, so every bit of its output is a function of the inputs and of this
// arithmetic, on any NVIDIA card. This file states that arithmetic once, in
// Rust, so the card's output can be checked bit for bit: on the Mac against
// the kernel text executed by clang++ (`llvq-llm/tests/attn_dec_host.rs`), and
// on the card at load (`crate::attn`).
//
// Rust `*`, `+`, `-`, `/` on f32 are IEEE round-to-nearest-even and are never
// contracted; `f32::mul_add` is a fused multiply-add, like `__fmaf_rn`. The
// kernel uses `__fadd_rn`, `__fmul_rn`, `__fsub_rn`, `__fdiv_rn` and
// `__fmaf_rn` for every float operation, so NVRTC's `--fmad=true` default
// cannot move a bit.
//
// Plain `//` comments only: `llvq-llm` includes this file with `include!`,
// because `llvq-cuda` is a Linux-only dependency there and its CPU fake of the
// kernel must run on the Mac. One text, two crates, no second copy to drift.
//
// The arithmetic, bit-relevant (changing any of these changes output bits):
//   - lanes: 16 lanes own one (row, query head); lane l owns dims 8l..8l+7; its
//     partial dot is an fma chain over those 8 dims from +0 (`dot8`);
//   - score: the xor tree 8, 4, 2, 1 over the 16 partials (`tree16`), then one
//     RN product by c = scale x log2(e) rounded to f32 (`score_scale`);
//   - tile: TT = 8 positions aligned to absolute positions; tile maximum, one
//     carry ex2(m_old - m_new), p = ex2(s - m_new) per position, their sum as
//     a left fold from +0, d <- d*carry + ts, each accumulator <- acc*carry
//     then an fma chain over the positions in order. Positions at or past the
//     row's length do no arithmetic;
//   - unit: `unit` positions (64 served, 32 the bench's alternative), tiles in
//     order; split: SPLIT = 256 positions aligned to multiples of 256, units
//     folded left to right; row: splits folded left to right (`st_merge`);
//   - output: f16 round-to-nearest of acc / d;
//   - exp2: `ex2_fixed`, a fixed polynomial with an exponent add.
// Which tiles, units and splits a row visits depends on its length alone, so
// a row's output is a function of (q, K[0..L), V[0..L)) only: the same bits
// in a decode call, in a verify call of up to KMAX rows, under any grid.

/// Head dimension: Qwen3's, the only one the kernel compiles.
pub const HD: usize = 128;
/// Lanes that share one score.
pub const LANES: usize = 16;
/// Dims a lane owns.
pub const DPL: usize = HD / LANES;
/// Positions per online-softmax update.
pub const TT: usize = 8;
/// Positions per split, the unit of work of one CTA and of the combine.
pub const SPLIT: usize = 256;
/// Rows one call carries: decode is 1, verify up to 8.
pub const KMAX: usize = 8;
/// The unit size the served arithmetic uses until the kernel-only bench
/// freezes it (prereg-attention-2a, rule 3).
pub const UNIT_SERVED: usize = 64;
/// Floats of one partial record: 128 accumulators, the maximum, the sum.
pub const PSTRIDE: usize = HD + 2;

/// The exp2 polynomial: p(f) = 1 + f(C1 + f(C2 + ... + f C6)), f in [0, 1).
/// A minimax fit of 2^f in relative error with p(0) = 1 exact, rounded to
/// f32; its Horner evaluation by fma stays within 0.92 ulp of 2^f on [0, 1)
/// (computed on 210,000 points, 2026-11-04), and `ex2_fixed` within 1.18 ulp
/// of 2^x on [-126, 0] (`tests/attn_ref_props.rs`). The kernel spells the same
/// values in decimal; `ex2_fixed` is diffed against the kernel text, so a
/// constant that drifted in either file fails that test.
pub const EX2_COEF_BITS: [u32; 6] =
    [0x3f31_7217, 0x3e75_feb3, 0x3d63_4139, 0x3c1e_ab8f, 0x3aa2_28db, 0x3966_5cf6];

/// binary16 bits to f32, exact.
pub fn h2f(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = match (exp, man) {
        (0, 0) => sign,
        (0, m) => {
            // Subnormal: normalise. The value is m x 2^-24.
            let lz = m.leading_zeros() - 22; // zeros above bit 9
            let m = (m << (lz + 1)) & 0x3ff;
            sign | ((127 - 15 - lz) << 23) | (m << 13)
        }
        (31, 0) => sign | 0x7f80_0000,
        (31, m) => sign | 0x7fc0_0000 | (m << 13),
        (e, m) => sign | ((e + 127 - 15) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

/// f32 to binary16 bits, round to nearest even, what `cvt.rn.f16.f32` does.
pub fn f2h(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 255 {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e16 = exp - 127 + 15;
    if e16 >= 31 {
        return sign | 0x7c00;
    }
    if e16 <= 0 {
        if e16 < -10 {
            return sign;
        }
        let man = man | 0x80_0000;
        let shift = (14 - e16) as u32;
        let half = 1u32 << (shift - 1);
        let mut v = man >> shift;
        let rem = man & ((1 << shift) - 1);
        if rem > half || (rem == half && v & 1 == 1) {
            v += 1;
        }
        return sign | v as u16;
    }
    let mut v = (sign as u32) | ((e16 as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && v & 1 == 1) {
        v += 1;
    }
    v as u16
}

/// 2^x for x <= 0, exactly specified. NaN propagates; x <= -126 and -inf give
/// +0; otherwise floor, an exact fraction, six fmas, and the integer part
/// added to the exponent field. The result is never subnormal.
pub fn ex2_fixed(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    // NaN is gone, so this is the kernel's `!(x > -126.0f)`.
    if x <= -126.0 {
        return 0.0;
    }
    let fl = x.floor();
    let f = x - fl;
    let c = EX2_COEF_BITS.map(f32::from_bits);
    let mut p = c[5];
    p = p.mul_add(f, c[4]);
    p = p.mul_add(f, c[3]);
    p = p.mul_add(f, c[2]);
    p = p.mul_add(f, c[1]);
    p = p.mul_add(f, c[0]);
    p = p.mul_add(f, 1.0);
    let e = fl as i32;
    // The kernel adds in unsigned arithmetic: a left shift of a negative int
    // is undefined in C++17, and the host probe compiles that text.
    f32::from_bits(p.to_bits().wrapping_add((e as u32) << 23))
}

/// c = (1 / sqrt(head_dim)) x log2(e), formed in f64 and rounded once.
pub fn score_scale(head_dim: usize) -> f32 {
    ((head_dim as f64).sqrt().recip() * std::f64::consts::LOG2_E) as f32
}

/// The unit sizes the kernel compiles.
pub fn check_unit(unit: usize) -> Result<(), String> {
    match unit {
        32 | 64 => Ok(()),
        u => Err(format!("attention unit {u}: the kernel compiles 64 (served) and 32 only")),
    }
}

/// One online-softmax state of one (row, query head): running maximum, sum,
/// and the 128 accumulators (the 16 lanes' slices side by side).
#[derive(Clone, Debug)]
pub struct St {
    pub m: f32,
    pub d: f32,
    pub a: [f32; HD],
}

impl St {
    pub fn empty() -> Self {
        St { m: f32::NEG_INFINITY, d: 0.0, a: [0.0; HD] }
    }
}

/// Merge `b` (later positions) into `a` (earlier positions). An empty right
/// operand is skipped, an empty left operand is replaced; otherwise two RN
/// products and one RN sum per term, never an fma.
pub fn st_merge(a: &mut St, b: &St) {
    if b.m == f32::NEG_INFINITY {
        return;
    }
    if a.m == f32::NEG_INFINITY {
        *a = b.clone();
        return;
    }
    let m = a.m.max(b.m);
    let fa = ex2_fixed(a.m - m);
    let fb = ex2_fixed(b.m - m);
    a.d = a.d * fa + b.d * fb;
    for i in 0..HD {
        a.a[i] = a.a[i] * fa + b.a[i] * fb;
    }
    a.m = m;
}

/// One lane's partial dot: an fma chain over its 8 dims, from +0.
pub fn dot8(q: &[f32; HD], k: &[u16], lane: usize) -> f32 {
    let mut t = 0f32;
    for i in 0..DPL {
        let d = lane * DPL + i;
        t = q[d].mul_add(h2f(k[d]), t);
    }
    t
}

/// The xor tree over the 16 partials: offsets 8, 4, 2, 1, pairwise RN sums.
/// Every lane of the butterfly ends with these bits, since RN addition
/// commutes; read at lane 0.
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

/// One score: the tree of the 16 lane partials, times c.
pub fn score(q: &[f32; HD], k: &[u16], c: f32) -> f32 {
    let mut parts = [0f32; LANES];
    for (lane, p) in parts.iter_mut().enumerate() {
        *p = dot8(q, k, lane);
    }
    tree16(parts) * c
}

/// The K and V caches of one layer, `[n_kv][cap][HD]` as f16 bits.
#[derive(Clone, Copy)]
pub struct Kv<'a> {
    pub k: &'a [u16],
    pub v: &'a [u16],
    pub cap: usize,
}

impl Kv<'_> {
    fn krow(&self, h: usize, p: usize) -> &[u16] {
        let o = (h * self.cap + p) * HD;
        &self.k[o..o + HD]
    }
    fn vrow(&self, h: usize, p: usize) -> &[u16] {
        let o = (h * self.cap + p) * HD;
        &self.v[o..o + HD]
    }
}

/// One tile of up to TT positions from absolute position `p0` (< `l`).
fn tile(st: &mut St, q: &[f32; HD], kv: &Kv, h: usize, p0: usize, l: usize, c: f32) {
    let nv = TT.min(l - p0);
    let mut sc = [0f32; TT];
    for (j, s) in sc.iter_mut().enumerate().take(nv) {
        *s = score(q, kv.krow(h, p0 + j), c);
    }
    let mut mt = sc[0];
    for &s in &sc[1..nv] {
        mt = mt.max(s);
    }
    let (m_new, carry) = if st.m == f32::NEG_INFINITY {
        (mt, 0.0)
    } else {
        let m = st.m.max(mt);
        (m, ex2_fixed(st.m - m))
    };
    let mut p = [0f32; TT];
    let mut ts = 0f32;
    for j in 0..nv {
        p[j] = ex2_fixed(sc[j] - m_new);
        ts += p[j];
    }
    st.d = st.d * carry + ts;
    for i in 0..HD {
        let mut a = st.a[i] * carry;
        for (j, &pj) in p.iter().enumerate().take(nv) {
            a = pj.mul_add(h2f(kv.vrow(h, p0 + j)[i]), a);
        }
        st.a[i] = a;
    }
    st.m = m_new;
}

/// The state of one split: units of `unit` positions folded in order.
pub fn split_state(q: &[f32; HD], kv: &Kv, h: usize, s: usize, l: usize, c: f32, unit: usize) -> St {
    let mut split = St::empty();
    let s0 = s * SPLIT;
    for u0 in (s0..s0 + SPLIT).step_by(unit) {
        if u0 >= l {
            break;
        }
        let mut st = St::empty();
        for p0 in (u0..u0 + unit).step_by(TT) {
            if p0 >= l {
                break;
            }
            tile(&mut st, q, kv, h, p0, l, c);
        }
        st_merge(&mut split, &st);
    }
    split
}

/// One row of one query head over positions `0..l` of KV head `h`, f16 bits.
pub fn row(q: &[f32; HD], kv: &Kv, h: usize, l: usize, c: f32, unit: usize) -> [u16; HD] {
    assert!(l >= 1, "a row sees at least its own position");
    let mut total = St::empty();
    for s in 0..l.div_ceil(SPLIT) {
        st_merge(&mut total, &split_state(q, kv, h, s, l, c, unit));
    }
    let mut o = [0u16; HD];
    for (i, oi) in o.iter_mut().enumerate() {
        *oi = f2h(total.a[i] / total.d);
    }
    o
}

/// What one kernel call computes: `rows` rows at absolute positions
/// `pos0..pos0 + rows`, row r over positions `0..=pos0 + r`.
///
/// `q` is `[n_kv * groups][rows][HD]` f16 bits, query head `h * groups + g`
/// reading KV head `h` (the `repeat_kv` convention); the result has the same
/// layout. The cache must already hold every position the rows read.
#[allow(clippy::too_many_arguments)]
pub fn attend(
    q: &[u16],
    kv: &Kv,
    n_kv: usize,
    groups: usize,
    pos0: usize,
    rows: usize,
    c: f32,
    unit: usize,
) -> Vec<u16> {
    let n_heads = n_kv * groups;
    assert_eq!(q.len(), n_heads * rows * HD, "q is [n_heads][rows][HD]");
    assert!(pos0 + rows <= kv.cap, "rows past the cache capacity");
    let mut out = vec![0u16; q.len()];
    for hq in 0..n_heads {
        for r in 0..rows {
            let o = (hq * rows + r) * HD;
            let mut qf = [0f32; HD];
            for (i, x) in qf.iter_mut().enumerate() {
                *x = h2f(q[o + i]);
            }
            out[o..o + HD].copy_from_slice(&row(&qf, kv, hq / groups, pos0 + r + 1, c, unit));
        }
    }
    out
}

/// A deterministic fill for checks and benches: f16 values in about
/// [-amp, amp], never NaN or inf. `xorshift64*`, so the card's load-time
/// check and the Mac's tests can build the same data.
pub fn fill(n: usize, seed: u64, amp: f32) -> Vec<u16> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            let r = s.wrapping_mul(0x2545_f491_4f6c_dd1d);
            let u = ((r >> 40) as f32) / ((1u64 << 24) as f32); // [0, 1)
            f2h((2.0 * u - 1.0) * amp)
        })
        .collect()
}
