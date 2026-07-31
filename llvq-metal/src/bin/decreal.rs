//! The runtime format, measured on **real blocks** — the number the
//! passation said to get before writing any matvec.
//!
//! `decode` (the synthetic bench) measured nested masks at 0.11 ns/block on
//! blocks that all had 4 magnitude levels, uniform layout, made-up codes.
//! This bench closes the three gaps it left open:
//!
//! * **real blocks** — read from the published 4B artifact, transcoded by
//!   `llvq_artifact::runtime` into the frozen payload;
//! * **real divergence** — 3, 4 and 5 levels mixed as the model mixes them,
//!   the level chain predicated per lane;
//! * **real addressing** — both finalists: `Fixed96` (12-byte stride, aligned
//!   loads) and `Grouped32` (byte strides via one u32 base per group,
//!   unaligned loads). Same payload bits, only the addressing differs; the
//!   verdict between 4.00 and 3.35 bits/weight is exactly the cost measured
//!   here.
//!
//! Every kernel's output is verified against the CPU reference decoder
//! (`RuntimeBlocks::decode_block`, itself pinned bit-for-bit against
//! `Indexer::decode`) for **every block**, before any timing is believed.
//! The shader is a third, independent implementation of the format spec —
//! a CPU encoder/decoder pair that agreed on a wrong reading would fail here.
//!
//! Run: `cargo run --release -p llvq-metal --bin decreal [model.llvq]`

use llvq_artifact::runtime::{transcode, ClassTable, Layout, RuntimeBlocks};
use llvq_core::DIM;
use llvq_search::fastdec::FastDecoder;
use std::fs::File;
use std::io::BufReader;

/// Blocks to measure — the synthetic bench's 16.7 M, for comparability.
const N: usize = 1 << 24;

const SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint DIM = 24;

// Mirror of the Rust-side record. 32 bytes; `vals` are the level values
// already divided by the shell norm, so a block's weights are
// vals[level] * gscale[gain] (row scale folded elsewhere).
struct ClassRec {
    float vals[5];
    uchar counts[4];
    uchar len;
    uchar nz;
    uchar pad[6];
};

// ---------------------------------------------------------------------------
// A 96-bit cursor in registers. Fields are ≤ 32 bits wide; width 0 is legal
// (the origin's empty sign field) and must not shift by 64.
// ---------------------------------------------------------------------------
struct Cursor { ulong lo; uint hi; };

static inline uint take(thread Cursor &c, uint width) {
    if (width == 0) return 0u;
    uint v = uint(c.lo & ((1ul << width) - 1ul));
    c.lo = (c.lo >> width) | (ulong(c.hi) << (64u - width));
    c.hi = c.hi >> width;
    return v;
}

// ---------------------------------------------------------------------------
// The payload decode both layouts share, from an aligned cursor.
//
// The level chain is the popcount form the synthetic bench measured: the
// mask of the block's *last* level is replaced by all-ones, so the chain
// needs no per-level count checks — it stops where the class says it stops.
// The sign index is a running nonzero counter, free in the serial walk.
// ---------------------------------------------------------------------------
static inline float decode_payload(Cursor c,
                                   constant ClassRec* tab,
                                   constant float* gscale,
                                   threadgroup const float* xs)
{
    uint id   = take(c, 9);
    uint gain = take(c, 1);            // the published 4B: 1 gain bit
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

    float dot = 0.0f;
    uint nz = 0;
    for (uint i = 0; i < DIM; ++i) {
        uint b0 = 1u << i;
        float v;
        if (m0 & b0) {
            v = r.vals[0];
        } else {
            uint p1 = popcount(~m0 & (b0 - 1u));
            uint b1 = 1u << p1;
            if (m1 & b1) {
                v = r.vals[1];
            } else {
                uint p2 = popcount(~m1 & (b1 - 1u));
                uint b2 = 1u << p2;
                if (m2 & b2) {
                    v = r.vals[2];
                } else {
                    uint p3 = popcount(~m2 & (b2 - 1u));
                    v = (m3 & (1u << p3)) ? r.vals[3] : r.vals[4];
                }
            }
        }
        if (v != 0.0f) {
            if ((signs >> nz) & 1u) v = -v;
            ++nz;
        }
        dot += v * xs[i];
    }
    return dot * gscale[gain];
}

// ---------------------------------------------------------------------------
// floor: read the fixed layout's 12 bytes, decode nothing.
// ---------------------------------------------------------------------------
kernel void floor96(device const uint*  words [[buffer(0)]],
                    device const float* x     [[buffer(1)]],
                    device float*       out   [[buffer(2)]],
                    uint gid [[thread_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float xs[DIM];
    if (tid < DIM) xs[tid] = x[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint acc = words[3*gid] ^ words[3*gid + 1] ^ words[3*gid + 2];
    float dot = 0.0f;
    for (uint i = 0; i < DIM; ++i) {
        dot += float(char((acc >> (i & 7)) & 0x7f)) * xs[i];
    }
    out[gid] = dot;
}

// ---------------------------------------------------------------------------
// Fixed96: three aligned u32 loads at a constant stride, then the payload.
// ---------------------------------------------------------------------------
kernel void decode_f96(device const uint*     words  [[buffer(0)]],
                       constant ClassRec*     tab    [[buffer(1)]],
                       constant float*        gscale [[buffer(2)]],
                       device const float*    x      [[buffer(3)]],
                       device float*          out    [[buffer(4)]],
                       uint gid [[thread_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float xs[DIM];
    if (tid < DIM) xs[tid] = x[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint w0 = words[3*gid], w1 = words[3*gid + 1], w2 = words[3*gid + 2];
    Cursor c = { (ulong(w1) << 32) | ulong(w0), w2 };
    out[gid] = decode_payload(c, tab, gscale, xs);
}

// ---------------------------------------------------------------------------
// Grouped32: byte-granular strides. The block's bytes start anywhere, so
// four aligned words are read and funnel-shifted down to the payload.
// ---------------------------------------------------------------------------
kernel void decode_g32(device const uint*     words  [[buffer(0)]],
                       device const uint*     bases  [[buffer(1)]],
                       constant ClassRec*     tab    [[buffer(2)]],
                       constant float*        gscale [[buffer(3)]],
                       device const float*    x      [[buffer(4)]],
                       device float*          out    [[buffer(5)]],
                       uint gid [[thread_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float xs[DIM];
    if (tid < DIM) xs[tid] = x[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint g = gid >> 5;
    uint base = bases[g];
    uint stride = (bases[g + 1] - base) >> 5;
    uint byte = base + (gid & 31u) * stride;

    uint w = byte >> 2;
    uint w0 = words[w], w1 = words[w + 1], w2 = words[w + 2], w3 = words[w + 3];
    uint sh = (byte & 3u) * 8u;
    ulong lo = (ulong(w1) << 32) | ulong(w0);
    uint  hi;
    if (sh == 0) {
        hi = w2;
    } else {
        lo = (lo >> sh) | (ulong(w2) << (64u - sh));
        hi = (w2 >> sh) | (w3 << (32u - sh));
    }
    Cursor c = { lo, hi };
    out[gid] = decode_payload(c, tab, gscale, xs);
}
"#;

/// GPU-side class record; must match the shader's `ClassRec` byte for byte.
#[repr(C)]
#[derive(Clone, Copy)]
struct GpuClassRec {
    vals: [f32; 5],
    counts: [u8; 4],
    len: u8,
    nz: u8,
    pad: [u8; 6],
}

fn gpu_class_table(fd: &FastDecoder) -> Vec<GpuClassRec> {
    let mut recs = Vec::with_capacity(fd.n_classes() + 1);
    // Entry 0: the origin.
    recs.push(GpuClassRec {
        vals: [0.0; 5],
        counts: [24, 0, 0, 0],
        len: 1,
        nz: 0,
        pad: [0; 6],
    });
    for ci in 0..fd.n_classes() {
        let lv = fd.levels(ci);
        let norm = ((16 * lv.shell) as f64).sqrt();
        let mut vals = [0.0f32; 5];
        for (v, &raw) in vals.iter_mut().zip(&lv.values[..lv.len]) {
            *v = (raw as f64 / norm) as f32;
        }
        recs.push(GpuClassRec {
            vals,
            counts: [lv.counts[0], lv.counts[1], lv.counts[2], lv.counts[3]],
            len: lv.len as u8,
            nz: lv.nonzero,
            pad: [0; 6],
        });
    }
    recs
}

/// The activation and gain levels the dot products run against.
const GSCALE: [f32; 2] = [0.9, 1.1];

fn xvec() -> Vec<f32> {
    (0..DIM).map(|i| 1.0 + i as f32 * 0.125).collect()
}

/// CPU expectation for one block, f64 end to end; the GPU is f32, so the
/// comparison carries a tolerance rather than an equality.
fn expected(rt: &RuntimeBlocks, table: &ClassTable, b: usize, x: &[f32]) -> f64 {
    let (p, gain) = rt.decode_block(table, b);
    match llvq_core::Leech::shell_index(&p) {
        Some(m) if m > 0 => {
            let norm = ((16 * m) as f64).sqrt();
            let dot: f64 = p
                .iter()
                .zip(x)
                .map(|(&v, &xi)| v as f64 / norm * xi as f64)
                .sum();
            dot * GSCALE[gain as usize] as f64
        }
        _ => 0.0,
    }
}

fn verify(name: &str, got: &[f32], rt: &RuntimeBlocks, table: &ClassTable, x: &[f32]) {
    for (b, &g) in got.iter().enumerate() {
        let want = expected(rt, table, b, x);
        let tol = 2e-3f64.max(2e-3 * want.abs());
        assert!(
            (g as f64 - want).abs() < tol,
            "{name}: block {b} decodes to {g}, reference says {want}"
        );
    }
    println!("  {name} : {} blocs vérifiés contre le décodeur CPU", got.len());
}

fn main() -> Result<(), String> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| format!("{}/llvq-q4b.llvq", std::env::var("HOME").unwrap()));

    // ---- collect real blocks: a contiguous prefix of every matrix ----
    let fd = FastDecoder::new();
    let table = ClassTable::new(&fd, 1);
    let f = File::open(&path).map_err(|e| format!("open {path}: {e}"))?;
    let mut r = BufReader::new(f);
    let h = llvq_artifact::read_header(&mut r).map_err(|e| e.to_string())?;
    let per_matrix = N.div_ceil(h.matrices as usize);
    let mut indices = Vec::with_capacity(N);
    let mut gains = Vec::with_capacity(N);
    for _ in 0..h.matrices {
        let m = llvq_artifact::read_matrix_raw(&mut r).map_err(|e| e.to_string())?;
        assert_eq!(
            m.centroids.len().next_power_of_two().trailing_zeros(),
            1,
            "{}: the bench's shader hardcodes 1 gain bit",
            m.name
        );
        let take = per_matrix.min(m.indices.len()).min(N - indices.len());
        indices.extend_from_slice(&m.indices[..take]);
        gains.extend_from_slice(&m.gains[..take]);
    }
    let n = indices.len();
    println!(
        "{n} blocs réels ({} matrices, préfixes contigus) — {path}",
        h.matrices
    );

    // ---- transcode both layouts ----
    let t = std::time::Instant::now();
    let f96 = transcode(&fd, &table, &indices, &gains, Layout::Fixed96)
        .map_err(|e| e.to_string())?;
    let g32 = transcode(&fd, &table, &indices, &gains, Layout::Grouped32)
        .map_err(|e| e.to_string())?;
    println!(
        "transcodés en {:.1} s — F96 {:.4} b/poids, G32 {:.4} b/poids\n",
        t.elapsed().as_secs_f64(),
        f96.bits_per_weight(),
        g32.bits_per_weight()
    );

    // ---- GPU setup ----
    let floor = llvq_metal::Kernel::new(SRC, "floor96")?;
    println!("GPU : {}", floor.device_name());
    let overhead = floor.overhead(20);
    println!("  surcoût de soumission {:.3} ms\n", overhead * 1e3);

    let x = xvec();
    let recs = gpu_class_table(&fd);
    let bf96 = floor.buffer(&f96.data);
    // Pad so the grouped kernel's fourth word never reads past the end.
    let mut g32_data = g32.data.clone();
    g32_data.extend_from_slice(&[0u8; 16]);
    let bg32 = floor.buffer(&g32_data);
    let bbases = floor.buffer(&g32.bases);
    let btab = floor.buffer(&recs);
    let bgs = floor.buffer(&GSCALE);
    let bx = floor.buffer(&x);
    let bout = floor.empty::<f32>(n);

    let report = |name: &str, secs: f64, floor_secs: f64| {
        let net = secs - overhead;
        let per_block = net / n as f64;
        println!(
            "  {name:<26}{:>9.3} ms{:>10.3} ns/bloc{:>13.2e} blocs/s{:>9}",
            net * 1e3,
            per_block * 1e9,
            1.0 / per_block,
            if floor_secs > 0.0 {
                format!("{:.2}×", net / floor_secs)
            } else {
                "—".into()
            }
        );
        net
    };

    println!(
        "  {:<26}{:>12}{:>18}{:>18}{:>9}",
        "noyau", "temps", "par bloc", "débit", "vs sol"
    );
    println!("  {}", "-".repeat(84));

    // ---- floor ----
    let t = floor.time(n as u64, 256, 3, 15, |enc| {
        enc.set_buffer(0, Some(&bf96), 0);
        enc.set_buffer(1, Some(&bx), 0);
        enc.set_buffer(2, Some(&bout), 0);
    });
    let t_floor = report("sol (12 o lus, rien décodé)", t.seconds, 0.0);

    // ---- Fixed96 ----
    let kf = llvq_metal::Kernel::new(SRC, "decode_f96")?;
    let t = kf.time(n as u64, 256, 3, 15, |enc| {
        enc.set_buffer(0, Some(&bf96), 0);
        enc.set_buffer(1, Some(&btab), 0);
        enc.set_buffer(2, Some(&bgs), 0);
        enc.set_buffer(3, Some(&bx), 0);
        enc.set_buffer(4, Some(&bout), 0);
    });
    let t_f96 = report("Fixed96 (aligné)", t.seconds, t_floor);
    let got: Vec<f32> = unsafe { kf.read(&bout, n) };
    verify("Fixed96", &got, &f96, &table, &x);

    // ---- Grouped32 ----
    let kg = llvq_metal::Kernel::new(SRC, "decode_g32")?;
    let t = kg.time(n as u64, 256, 3, 15, |enc| {
        enc.set_buffer(0, Some(&bg32), 0);
        enc.set_buffer(1, Some(&bbases), 0);
        enc.set_buffer(2, Some(&btab), 0);
        enc.set_buffer(3, Some(&bgs), 0);
        enc.set_buffer(4, Some(&bx), 0);
        enc.set_buffer(5, Some(&bout), 0);
    });
    let t_g32 = report("Grouped32 (strides octet)", t.seconds, t_floor);
    let got: Vec<f32> = unsafe { kg.read(&bout, n) };
    verify("Grouped32", &got, &g32, &table, &x);

    // ---- what it means ----
    println!("  {}", "-".repeat(84));
    let blocks_4b = 3_633_315_840f64 / 24.0;
    println!("\n  Pour un Qwen3-4B ({:.0} M blocs) et un token :", blocks_4b / 1e6);
    for (name, per_block, bpw) in [
        ("Fixed96", t_f96 / n as f64, f96.bits_per_weight()),
        ("Grouped32", t_g32 / n as f64, g32.bits_per_weight()),
    ] {
        let decode_ms = blocks_4b * per_block * 1e3;
        let gb = 3_633_315_840f64 * bpw / 8.0 / 1e9 + 0.778;
        println!(
            "    {name:<11} décodage {decode_ms:>6.2} ms/token — trafic {gb:.2} Go \
             → plafond mémoire {:.0} tok/s",
            400.0 / gb
        );
    }
    println!(
        "\n  Repère synthétique (4 niveaux uniformes, uint4) : 0,11 ns/bloc.\n  \
         Le sol lit les mêmes 12 octets que Fixed96 ; Grouped32 en lit ~{:.1}.",
        g32.data.len() as f64 / n as f64
    );
    Ok(())
}
