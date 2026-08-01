//! The fused kernel, at last: dequant + matvec on one **real layer**, against
//! an FP16 matvec on the same machine.
//!
//! This is the measurement G6 exists for. The paper's Table 7 reports its
//! (single-shell, M = 3) CUDA kernel at 11.94 µs against 16.3-17.69 µs FP16
//! on a 4096×4096-class layer — 1.36-1.48× — while conceding it is slower
//! than QTIP's. The multi-shell kernel (m ≤ 13, the 2 bit regime) exists
//! nowhere; this is it, on Apple Silicon, on the frozen Grouped32 format.
//!
//! ## Shape
//!
//! One SIMD group (32 lanes) per output row. The activation is staged in
//! threadgroup memory once per threadgroup. Each lane decodes every 32nd
//! block of the row — consecutive lanes hold consecutive blocks, so the
//! grouped stream is read almost sequentially — accumulates a partial dot,
//! and a `simd_sum` folds the row. Lane 0 adds the `KeepExact` tail columns
//! (kept f32, as the artifact stores them) and applies the row scale.
//!
//! The FP16 baseline has the same shape with `half4` loads, 256 coalesced
//! bytes per iteration per SIMD group — an honest baseline, not a straw man.
//!
//! ## Honesty
//!
//! * Both outputs are verified against f64 CPU references before timing —
//!   the LLVQ one against the transcoded blocks (pinned bit-for-bit on
//!   `Indexer::decode`), the FP16 one against the same weights rounded
//!   through f16, each with an error budget scaled to Σ|wᵢ·xᵢ|.
//! * One matvec is ~150 µs — the same order as the ~0.15 ms submission
//!   overhead, the exact trap the passation documents. So each measurement
//!   encodes 32 back-to-back dispatches in one command buffer and the
//!   overhead is subtracted once.
//! * Weights and scales are the real ones from the artifact — real centroids,
//!   real row scales, real tail — not synthetic stand-ins.
//!
//! Run: `cargo run --release -p llvq-metal --bin matvec [model.llvq] [name-substring]`

use llvq_artifact::runtime::{transcode, ClassTable, Layout};
use llvq_core::DIM;
use llvq_search::fastdec::FastDecoder;
use std::fs::File;
use std::io::BufReader;

/// Matvecs per command buffer — sizes the measurement in milliseconds.
const K: usize = 32;

const SRC: &str = r#"
struct Params { uint d_in; uint d_out; uint nblocks; uint tail_w; };

// ---------------------------------------------------------------------------
// Branchless variant of the payload decode. The naive form loses to FP16
// (205 µs vs 129): its per-slot level chain diverges — lanes decode different
// classes, so under predication every slot costs the deepest path any lane
// takes — and its running sign counter serializes the walk. Here every slot
// is unconditional and independent: three prefix popcounts, four mask tests,
// selects, and the sign bit found by a prefix popcount of the nonzero mask —
// which is just "not the zero level's mask in slot space", built on the fly.
// Two accumulators split the FMA chain.
// ---------------------------------------------------------------------------
static inline float decode_dot_ilp(Cursor c,
                                   constant ClassRec* tab,
                                   constant float* gscale,
                                   threadgroup const float* xs)
{
    uint id   = take(c, 9);
    uint gain = take(c, 1);
    constant ClassRec &r = tab[id];
    uint len = r.len;
    uint signs = take(c, r.nz);

    uint left = DIM;
    uint m0 = 0xffffffu, m1 = 0xffffffu, m2 = 0xffffffu, m3 = 0xffffffu;
    if (len > 1) { m0 = take(c, left); left -= r.counts[0];
        if (len > 2) { m1 = take(c, left); left -= r.counts[1];
            if (len > 3) { m2 = take(c, left); left -= r.counts[2];
                if (len > 4) { m3 = take(c, left); }
                else m3 = 0xffffffu;
            } else { m2 = 0xffffffu; }
        } else { m1 = 0xffffffu; }
    } else { m0 = 0xffffffu; }

    float v0 = r.vals[0], v1 = r.vals[1], v2 = r.vals[2],
          v3 = r.vals[3], v4 = r.vals[4];
    // Signed variants, picked by the slot's sign bit with one select.
    float n0 = -v0, n1 = -v1, n2 = -v2, n3 = -v3, n4 = -v4;

    float d0 = 0.0f, d1 = 0.0f;
    uint nzm = 0;   // nonzero mask over slots walked so far
    uint zlev = r.zlev;
    for (uint i = 0; i < DIM; ++i) {
        uint b0 = 1u << i;
        // Unconditional level resolution: prefix popcounts and tests.
        uint p1 = popcount(~m0 & (b0 - 1u));
        uint b1 = 1u << p1;
        uint p2 = popcount(~m1 & (b1 - 1u));
        uint b2 = 1u << p2;
        uint p3 = popcount(~m2 & (b2 - 1u));
        uint b3 = 1u << p3;
        bool t0 = m0 & b0, t1 = m1 & b1, t2 = m2 & b2, t3 = m3 & b3;
        uint lev = t0 ? 0u : t1 ? 1u : t2 ? 2u : t3 ? 3u : 4u;

        bool nzb = lev != zlev;
        bool neg = (signs >> popcount(nzm & (b0 - 1u))) & 1u;
        nzm |= uint(nzb) << i;
        neg = neg && nzb;
        float v = t0 ? (neg ? n0 : v0)
                : t1 ? (neg ? n1 : v1)
                : t2 ? (neg ? n2 : v2)
                : t3 ? (neg ? n3 : v3)
                :      (neg ? n4 : v4);
        if (i & 1) d1 = fma(v, xs[i], d1); else d0 = fma(v, xs[i], d0);
    }
    return (d0 + d1) * gscale[gain];
}

// ---------------------------------------------------------------------------
// FP16 baseline: one SIMD group per row, half4 loads, coalesced.
// ---------------------------------------------------------------------------
kernel void matvec_f16(device const half*   w   [[buffer(0)]],
                       device const float*  x   [[buffer(1)]],
                       device float*        y   [[buffer(2)]],
                       constant Params&     P   [[buffer(3)]],
                       threadgroup float*   xs  [[threadgroup(0)]],
                       uint gid  [[thread_position_in_grid]],
                       uint tid  [[thread_index_in_threadgroup]],
                       uint tgs  [[threads_per_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]])
{
    for (uint i = tid; i < P.d_in; i += tgs) xs[i] = x[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row = gid >> 5;
    device const half4* w4 = (device const half4*)(w + row * P.d_in);
    float acc = 0.0f;
    for (uint j = lane; j < (P.d_in >> 2); j += 32) {
        float4 wv = float4(w4[j]);
        uint c = j << 2;
        acc += wv.x * xs[c] + wv.y * xs[c + 1] + wv.z * xs[c + 2] + wv.w * xs[c + 3];
    }
    acc = simd_sum(acc);
    if (lane == 0) y[row] = acc;
}

// ---------------------------------------------------------------------------
// Shape floor: same rows, same G32 loads, same reduction — no decoding.
// What is left over the FP16 baseline after subtracting this is the cost of
// the decode itself; the floor is everything structural.
// ---------------------------------------------------------------------------
kernel void matvec_floor(device const uint*   words  [[buffer(0)]],
                         device const uint*   bases  [[buffer(1)]],
                         device const float*  rscale [[buffer(2)]],
                         device const float*  x      [[buffer(3)]],
                         device float*        y      [[buffer(4)]],
                         constant Params&     P      [[buffer(5)]],
                         threadgroup float*   xs     [[threadgroup(0)]],
                         uint gid  [[thread_position_in_grid]],
                         uint tid  [[thread_index_in_threadgroup]],
                         uint tgs  [[threads_per_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]])
{
    for (uint i = tid; i < P.d_in; i += tgs) xs[i] = x[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row = gid >> 5;
    float acc = 0.0f;
    for (uint j = lane; j < P.nblocks; j += 32) {
        Cursor c = cursor_g32(words, bases, row * P.nblocks + j);
        uint f = uint(c.lo) ^ uint(c.lo >> 32) ^ c.hi;
        threadgroup const float* xb = xs + j * DIM;
        float d0 = 0.0f, d1 = 0.0f;
        for (uint i = 0; i < DIM; i += 2) {
            d0 = fma(float(char((f >> (i & 7)) & 0x7f)), xb[i], d0);
            d1 = fma(float(char((f >> ((i + 1) & 7)) & 0x7f)), xb[i + 1], d1);
        }
        acc += d0 + d1;
    }
    acc = simd_sum(acc);
    if (lane == 0) y[row] = acc * rscale[row];
}

// ---------------------------------------------------------------------------
// LLVQ fused: decode Grouped32 blocks in flight, never materialize weights.
// ---------------------------------------------------------------------------
kernel void matvec_g32(device const uint*   words  [[buffer(0)]],
                       device const uint*   bases  [[buffer(1)]],
                       constant ClassRec*   tab    [[buffer(2)]],
                       constant float*      gscale [[buffer(3)]],
                       device const float*  rscale [[buffer(4)]],
                       device const float*  tail   [[buffer(5)]],
                       device const float*  x      [[buffer(6)]],
                       device float*        y      [[buffer(7)]],
                       constant Params&     P      [[buffer(8)]],
                       threadgroup float*   xs     [[threadgroup(0)]],
                       uint gid  [[thread_position_in_grid]],
                       uint tid  [[thread_index_in_threadgroup]],
                       uint tgs  [[threads_per_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]])
{
    for (uint i = tid; i < P.d_in; i += tgs) xs[i] = x[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row = gid >> 5;
    float acc = 0.0f;
    for (uint j = lane; j < P.nblocks; j += 32) {
        Cursor c = cursor_g32(words, bases, row * P.nblocks + j);
        acc += decode_payload(c, tab, gscale, xs + j * DIM);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        float t = 0.0f;
        uint c0 = P.nblocks * DIM;
        for (uint i = 0; i < P.tail_w; ++i) {
            t += tail[row * P.tail_w + i] * xs[c0 + i];
        }
        y[row] = acc * rscale[row] + t;
    }
}

// ---------------------------------------------------------------------------
// Flat32 fused matvec. The nested-mask decode pays prefix popcounts per
// slot; this payload was shaped for the kernel instead: slot-space masks
// (level 0 implicit), signs level-major. A level's slots come off its word
// with ctz, signs shift out sequentially, zero slots are never touched.
// ---------------------------------------------------------------------------

// 160-bit cursor: the flat worst case is 130 payload bits + a byte shift.
struct Cur5 { ulong lo; ulong hi; uint xt; };

static inline uint take5(thread Cur5 &c, uint width) {
    if (width == 0) return 0u;
    uint v = uint(c.lo & ((1ul << width) - 1ul));
    c.lo = (c.lo >> width) | (c.hi << (64u - width));
    c.hi = (c.hi >> width) | (ulong(c.xt) << (64u - width));
    c.xt = c.xt >> width;
    return v;
}

// One level's contribution: iterate its set bits, shift signs out as we go.
static inline float lrun(uint m, float v, uint s, threadgroup const float* xb) {
    float d = 0.0f;
    while (m) {
        uint i = ctz(m);
        m &= m - 1u;
        d = fma((s & 1u) ? -v : v, xb[i], d);
        s >>= 1u;
    }
    return d;
}

kernel void matvec_flat(device const uint*   words  [[buffer(0)]],
                        device const uint*   bases  [[buffer(1)]],
                        constant ClassRec*   tab    [[buffer(2)]],
                        constant float*      gscale [[buffer(3)]],
                        device const float*  rscale [[buffer(4)]],
                        device const float*  tail   [[buffer(5)]],
                        device const float*  x      [[buffer(6)]],
                        device float*        y      [[buffer(7)]],
                        constant Params&     P      [[buffer(8)]],
                        threadgroup float*   xs     [[threadgroup(0)]],
                        uint gid  [[thread_position_in_grid]],
                        uint tid  [[thread_index_in_threadgroup]],
                        uint tgs  [[threads_per_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]])
{
    for (uint i = tid; i < P.d_in; i += tgs) xs[i] = x[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row = gid >> 5;
    float acc = 0.0f;
    for (uint j = lane; j < P.nblocks; j += 32) {
        uint b = row * P.nblocks + j;
        uint g = b >> 5;
        uint base = bases[g];
        uint stride = (bases[g + 1] - base) >> 5;
        uint byte = base + (b & 31u) * stride;

        uint w = byte >> 2;
        uint w0 = words[w], w1 = words[w + 1], w2 = words[w + 2],
             w3 = words[w + 3], w4 = words[w + 4];
        uint sh = (byte & 3u) * 8u;
        ulong lo = (ulong(w1) << 32) | ulong(w0);
        ulong hi = (ulong(w3) << 32) | ulong(w2);
        uint xt = w4;
        if (sh != 0) {
            lo = (lo >> sh) | (hi << (64u - sh));
            hi = (hi >> sh) | (ulong(xt) << (64u - sh));
            xt >>= sh;
        }
        Cur5 c = { lo, hi, xt };

        uint id   = take5(c, 9);
        uint gain = take5(c, 1);
        constant ClassRec &r = tab[id];
        uint nlev = r.len;
        uint signs = take5(c, r.nz);
        uint m1 = 0, m2 = 0, m3 = 0, m4 = 0;
        if (nlev > 1) { m1 = take5(c, 24);
            if (nlev > 2) { m2 = take5(c, 24);
                if (nlev > 3) { m3 = take5(c, 24);
                    if (nlev > 4) { m4 = take5(c, 24); } } } }
        uint m0 = ~(m1 | m2 | m3 | m4) & 0xffffffu;

        threadgroup const float* xb = xs + j * DIM;
        uint zl = r.zlev;
        float d = 0.0f;
        if (zl != 0u) d += lrun(m0, r.vals[0], signs >> r.sbase[0], xb);
        if (nlev > 1 && zl != 1u) d += lrun(m1, r.vals[1], signs >> r.sbase[1], xb);
        if (nlev > 2 && zl != 2u) d += lrun(m2, r.vals[2], signs >> r.sbase[2], xb);
        if (nlev > 3 && zl != 3u) d += lrun(m3, r.vals[3], signs >> r.sbase[3], xb);
        if (nlev > 4 && zl != 4u) d += lrun(m4, r.vals[4], signs >> r.sbase[4], xb);
        acc = fma(d, gscale[gain], acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        float t = 0.0f;
        uint c0 = P.nblocks * DIM;
        for (uint i = 0; i < P.tail_w; ++i) {
            t += tail[row * P.tail_w + i] * xs[c0 + i];
        }
        y[row] = acc * rscale[row] + t;
    }
}
"#;

/// Mirror of the shader's `Params`.
#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    d_in: u32,
    d_out: u32,
    nblocks: u32,
    tail_w: u32,
}

/// f32 → IEEE 754 binary16 bits, round to nearest even. No `half` crate —
/// this crate stays at two dependencies.
fn f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 255 {
        // Inf / NaN.
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e16 = exp - 127 + 15;
    if e16 >= 31 {
        return sign | 0x7c00; // overflow to inf
    }
    if e16 <= 0 {
        // Subnormal (or zero): shift the implicit-1 mantissa down.
        if e16 < -10 {
            return sign;
        }
        let man = man | 0x80_0000;
        let shift = (14 - e16) as u32;
        let half = 1u32 << (shift - 1);
        let mut v = man >> shift;
        // Round to nearest even.
        let rem = man & ((1 << shift) - 1);
        if rem > half || (rem == half && v & 1 == 1) {
            v += 1;
        }
        return sign | v as u16;
    }
    let mut v = (sign as u32) | ((e16 as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && v & 1 == 1) {
        v += 1; // may carry into the exponent — that is correct rounding
    }
    v as u16
}

fn f16_to_f64(h: u16) -> f64 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = (h >> 10) & 0x1f;
    let man = (h & 0x3ff) as f64;
    sign * match exp {
        0 => man * (-24f64).exp2(),
        31 => {
            if man == 0.0 {
                f64::INFINITY
            } else {
                f64::NAN
            }
        }
        e => (1.0 + man / 1024.0) * ((e as f64) - 15.0).exp2(),
    }
}

/// `max |got − want| / Σ|wᵢxᵢ|` over the rows — the error unit a float dot
/// product is actually accountable to (relative error explodes when a row
/// cancels to near zero, which says nothing about the kernel).
fn check(name: &str, got: &[f32], want: &[f64], scale: &[f64]) {
    let mut worst = 0.0f64;
    for ((&g, &w), &s) in got.iter().zip(want).zip(scale) {
        worst = worst.max((g as f64 - w).abs() / s.max(1e-12));
    }
    assert!(
        worst < 1e-3,
        "{name}: worst row error {worst:.2e} of the row's Σ|w·x|"
    );
    println!("  {name:<10} : {} lignes vérifiées, pire erreur {worst:.1e}·Σ|w·x|", got.len());
}

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .unwrap_or_else(|| format!("{}/llvq-q4b.llvq", std::env::var("HOME").unwrap()));
    let target = args.next().unwrap_or_else(|| "layers.0.mlp.gate_proj".into());

    // ---- find the layer ----
    let f = File::open(&path).map_err(|e| format!("open {path}: {e}"))?;
    let mut r = BufReader::new(f);
    let h = llvq_artifact::read_header(&mut r).map_err(|e| e.to_string())?;
    let m = (0..h.matrices)
        .map(|_| llvq_artifact::read_matrix_raw(&mut r).expect("valid matrix"))
        .find(|m| m.name.contains(&target))
        .ok_or_else(|| format!("no matrix matching {target:?}"))?;
    let (d_out, d_in) = (m.d_out, m.d_in);
    let nblocks = d_in / DIM;
    let tail_w = d_in % DIM;
    assert_eq!(d_in % 4, 0, "the FP16 kernel loads half4");
    assert_eq!(m.centroids.len(), 2, "the shader hardcodes 1 gain bit");
    println!(
        "{} — {d_out}×{d_in}, {nblocks} blocs/ligne + queue {tail_w}, {} blocs",
        m.name,
        d_out * nblocks
    );

    // ---- transcode, and build the FP16 twin + f64 references ----
    let fd = FastDecoder::new();
    let table = ClassTable::new(&fd, 1);
    let rt = transcode(&fd, &table, &m.indices, &m.gains, Layout::Grouped32)
        .map_err(|e| e.to_string())?;
    let rtf = transcode(&fd, &table, &m.indices, &m.gains, Layout::Flat32)
        .map_err(|e| e.to_string())?;

    let mut rng = llvq_core::SplitMix64::new(0x6_AC71);
    let x: Vec<f32> = (0..d_in).map(|_| rng.next_gaussian() as f32).collect();

    // Row-major dequantized weights in f64 (rotated basis — the basis the
    // matvec runs in; the rotation of x is a separate per-layer cost that
    // neither kernel pays here).
    let mut w = vec![0.0f64; d_out * d_in];
    for row in 0..d_out {
        for p in 0..nblocks {
            let (pt, gain) = rt.decode_block(&table, row * nblocks + p);
            if let Some(shell) = llvq_core::Leech::shell_index(&pt).filter(|&s| s > 0) {
                let s = m.centroids[gain as usize] * m.row_scales[row]
                    / ((16 * shell) as f64).sqrt();
                for (i, &v) in pt.iter().enumerate() {
                    w[row * d_in + p * DIM + i] = v as f64 * s;
                }
            }
        }
        for t in 0..tail_w {
            w[row * d_in + nblocks * DIM + t] = m.tail[row * tail_w + t];
        }
    }
    let w16: Vec<u16> = w.iter().map(|&v| f16_bits(v as f32)).collect();

    // References and error scales, f64.
    let mut y_ref = vec![0.0f64; d_out];
    let mut y16_ref = vec![0.0f64; d_out];
    let mut scale = vec![0.0f64; d_out];
    for row in 0..d_out {
        let (mut a, mut b, mut s) = (0.0, 0.0, 0.0);
        for c in 0..d_in {
            let xv = x[c] as f64;
            a += w[row * d_in + c] * xv;
            b += f16_to_f64(w16[row * d_in + c]) * xv;
            s += (w[row * d_in + c] * xv).abs();
        }
        y_ref[row] = a;
        y16_ref[row] = b;
        scale[row] = s;
    }

    // ---- GPU ----
    let src = format!("{}{}", llvq_metal::PAYLOAD_MSL, SRC);
    let kf16 = llvq_metal::Kernel::new(&src, "matvec_f16")?;
    let kg32 = llvq_metal::Kernel::new(&src, "matvec_g32")?;
    println!("GPU : {}", kf16.device_name());
    let overhead = kf16.overhead(20);

    let params = [Params {
        d_in: d_in as u32,
        d_out: d_out as u32,
        nblocks: nblocks as u32,
        tail_w: tail_w as u32,
    }];
    let recs = llvq_metal::gpu_class_table(&fd);
    let gscale: Vec<f32> = m.centroids.iter().map(|&c| c as f32).collect();
    let rscale: Vec<f32> = m.row_scales.iter().map(|&s| s as f32).collect();
    let tailf: Vec<f32> = m.tail.iter().map(|&t| t as f32).collect();
    let mut g32_data = rt.data.clone();
    g32_data.extend_from_slice(&[0u8; 16]);
    let mut flat_data = rtf.data.clone();
    flat_data.extend_from_slice(&[0u8; 20]);

    let bw16 = kf16.buffer(&w16);
    let bx = kf16.buffer(&x);
    let by = kf16.empty::<f32>(d_out);
    let bp = kf16.buffer(&params);
    let bwords = kf16.buffer(&g32_data);
    let bbases = kf16.buffer(&rt.bases);
    let bwordsf = kf16.buffer(&flat_data);
    let bbasesf = kf16.buffer(&rtf.bases);
    let btab = kf16.buffer(&recs);
    let bgs = kf16.buffer(&gscale);
    let brs = kf16.buffer(&rscale);
    let btail = kf16.buffer(&tailf);

    let threads = (d_out * 32) as u64;
    let xs_bytes = (d_in * 4) as u64;

    let bind_f16 = |enc: &metal::ComputeCommandEncoderRef| {
        enc.set_buffer(0, Some(&bw16), 0);
        enc.set_buffer(1, Some(&bx), 0);
        enc.set_buffer(2, Some(&by), 0);
        enc.set_buffer(3, Some(&bp), 0);
        llvq_metal::Kernel::set_threadgroup_memory(enc, 0, xs_bytes);
    };
    let bind_g32 = |enc: &metal::ComputeCommandEncoderRef| {
        enc.set_buffer(0, Some(&bwords), 0);
        enc.set_buffer(1, Some(&bbases), 0);
        enc.set_buffer(2, Some(&btab), 0);
        enc.set_buffer(3, Some(&bgs), 0);
        enc.set_buffer(4, Some(&brs), 0);
        enc.set_buffer(5, Some(&btail), 0);
        enc.set_buffer(6, Some(&bx), 0);
        enc.set_buffer(7, Some(&by), 0);
        enc.set_buffer(8, Some(&bp), 0);
        llvq_metal::Kernel::set_threadgroup_memory(enc, 0, xs_bytes);
    };

    let kflat = llvq_metal::Kernel::new(&src, "matvec_flat")?;
    let bind_flat = |enc: &metal::ComputeCommandEncoderRef| {
        enc.set_buffer(0, Some(&bwordsf), 0);
        enc.set_buffer(1, Some(&bbasesf), 0);
        enc.set_buffer(2, Some(&btab), 0);
        enc.set_buffer(3, Some(&bgs), 0);
        enc.set_buffer(4, Some(&brs), 0);
        enc.set_buffer(5, Some(&btail), 0);
        enc.set_buffer(6, Some(&bx), 0);
        enc.set_buffer(7, Some(&by), 0);
        enc.set_buffer(8, Some(&bp), 0);
        llvq_metal::Kernel::set_threadgroup_memory(enc, 0, xs_bytes);
    };

    // ---- verify before timing ----
    kf16.dispatch(threads, 256, bind_f16);
    let got: Vec<f32> = unsafe { kf16.read(&by, d_out) };
    check("FP16", &got, &y16_ref, &scale);
    kg32.dispatch(threads, 256, bind_g32);
    let got: Vec<f32> = unsafe { kg32.read(&by, d_out) };
    check("LLVQ G32", &got, &y_ref, &scale);
    kflat.dispatch(threads, 256, bind_flat);
    let got: Vec<f32> = unsafe { kflat.read(&by, d_out) };
    check("LLVQ Flat32", &got, &y_ref, &scale);

    let kfl = llvq_metal::Kernel::new(&src, "matvec_floor")?;
    let bind_floor = |enc: &metal::ComputeCommandEncoderRef| {
        enc.set_buffer(0, Some(&bwords), 0);
        enc.set_buffer(1, Some(&bbases), 0);
        enc.set_buffer(2, Some(&brs), 0);
        enc.set_buffer(3, Some(&bx), 0);
        enc.set_buffer(4, Some(&by), 0);
        enc.set_buffer(5, Some(&bp), 0);
        llvq_metal::Kernel::set_threadgroup_memory(enc, 0, xs_bytes);
    };

    // ---- time: K dispatches per command buffer, overhead subtracted once ----
    let t16 = (kf16.time_many(threads, 256, K, 3, 15, bind_f16).seconds - overhead)
        / K as f64;
    let tfl = (kfl.time_many(threads, 256, K, 3, 15, bind_floor).seconds - overhead)
        / K as f64;
    let tg = (kg32.time_many(threads, 256, K, 3, 15, bind_g32).seconds - overhead)
        / K as f64;
    let tf = (kflat.time_many(threads, 256, K, 3, 15, bind_flat).seconds - overhead)
        / K as f64;

    let f16_bytes = (d_out * d_in * 2) as f64;
    let g32_bytes = rt.data.len() as f64
        + rt.bases.len() as f64 * 4.0
        + (d_out * tail_w) as f64 * 4.0
        + d_out as f64 * 4.0;
    println!("\n  {:<22}{:>10}{:>12}{:>14}", "noyau", "µs", "Go/s eff.", "vs FP16");
    println!("  {}", "-".repeat(60));
    println!(
        "  {:<22}{:>10.2}{:>12.0}{:>14}",
        "FP16 (half4)",
        t16 * 1e6,
        f16_bytes / t16 / 1e9,
        "1.00×"
    );
    println!(
        "  {:<22}{:>10.2}{:>12.0}{:>14}",
        "sol G32 (sans décodage)",
        tfl * 1e6,
        g32_bytes / tfl / 1e9,
        format!("{:.2}×", t16 / tfl)
    );
    println!(
        "  {:<22}{:>10.2}{:>12.0}{:>14}",
        "LLVQ fusé (G32)",
        tg * 1e6,
        g32_bytes / tg / 1e9,
        format!("{:.2}×", t16 / tg)
    );
    let flat_bytes = rtf.data.len() as f64
        + rtf.bases.len() as f64 * 4.0
        + (d_out * tail_w) as f64 * 4.0
        + d_out as f64 * 4.0;
    println!(
        "  {:<22}{:>10.2}{:>12.0}{:>14}",
        "LLVQ fusé (Flat32)",
        tf * 1e6,
        flat_bytes / tf / 1e9,
        format!("{:.2}×", t16 / tf)
    );
    println!(
        "\n  poids lus : FP16 {:.1} Mo, LLVQ G32 {:.1} Mo, Flat32 {:.1} Mo",
        f16_bytes / 1e6, g32_bytes / 1e6, flat_bytes / 1e6
    );
    println!(
        "\n  poids lus : FP16 {:.1} Mo, LLVQ {:.1} Mo (×{:.2} de moins)",
        f16_bytes / 1e6,
        g32_bytes / 1e6,
        f16_bytes / g32_bytes
    );
    println!(
        "  repère papier (Table 7, leur GPU, 4096×4104, M=3) : LLVQ 11,94 µs, \
         FP16 17,69 µs — 1,48×"
    );
    Ok(())
}
