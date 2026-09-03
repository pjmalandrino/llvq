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

// Appended to `llvq_metal::PAYLOAD_MSL`, which provides ClassRec, the
// cursor, `decode_payload` and `cursor_g32` — the same source `matvec` uses.
const SRC: &str = r#"
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

    Cursor c = cursor_g32(words, bases, gid);
    out[gid] = decode_payload(c, tab, gscale, xs);
}
"#;

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
    println!("  {name}: {} blocks verified against the CPU decoder", got.len());
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
        "{n} real blocks ({} matrices, contiguous prefixes) — {path}",
        h.matrices
    );

    // ---- transcode both layouts ----
    let t = std::time::Instant::now();
    let f96 = transcode(&fd, &table, &indices, &gains, Layout::Fixed96)
        .map_err(|e| e.to_string())?;
    let g32 = transcode(&fd, &table, &indices, &gains, Layout::Grouped32)
        .map_err(|e| e.to_string())?;
    println!(
        "transcoded in {:.1} s — F96 {:.4} b/weight, G32 {:.4} b/weight\n",
        t.elapsed().as_secs_f64(),
        f96.bits_per_weight(),
        g32.bits_per_weight()
    );

    // ---- GPU setup ----
    let src = format!("{}{}", llvq_metal::PAYLOAD_MSL, SRC);
    let floor = llvq_metal::Kernel::new(&src, "floor96")?;
    println!("GPU: {}", floor.device_name());
    let overhead = floor.overhead(20);
    println!("  submission overhead {:.3} ms\n", overhead * 1e3);

    let x = xvec();
    let recs = llvq_metal::gpu_class_table(&fd);
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
            "  {name:<26}{:>9.3} ms{:>10.3} ns/block{:>13.2e} blocks/s{:>9}",
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
        "kernel", "time", "per block", "throughput", "vs sol"
    );
    println!("  {}", "-".repeat(84));

    // ---- floor ----
    let t = floor.time(n as u64, 256, 3, 15, |enc| {
        enc.set_buffer(0, Some(&bf96), 0);
        enc.set_buffer(1, Some(&bx), 0);
        enc.set_buffer(2, Some(&bout), 0);
    });
    let t_floor = report("sol (12 B read, no decode)", t.seconds, 0.0);

    // ---- Fixed96 ----
    let kf = llvq_metal::Kernel::new(&src, "decode_f96")?;
    let t = kf.time(n as u64, 256, 3, 15, |enc| {
        enc.set_buffer(0, Some(&bf96), 0);
        enc.set_buffer(1, Some(&btab), 0);
        enc.set_buffer(2, Some(&bgs), 0);
        enc.set_buffer(3, Some(&bx), 0);
        enc.set_buffer(4, Some(&bout), 0);
    });
    let t_f96 = report("Fixed96 (aligned)", t.seconds, t_floor);
    let got: Vec<f32> = unsafe { kf.read(&bout, n) };
    verify("Fixed96", &got, &f96, &table, &x);

    // ---- Grouped32 ----
    let kg = llvq_metal::Kernel::new(&src, "decode_g32")?;
    let t = kg.time(n as u64, 256, 3, 15, |enc| {
        enc.set_buffer(0, Some(&bg32), 0);
        enc.set_buffer(1, Some(&bbases), 0);
        enc.set_buffer(2, Some(&btab), 0);
        enc.set_buffer(3, Some(&bgs), 0);
        enc.set_buffer(4, Some(&bx), 0);
        enc.set_buffer(5, Some(&bout), 0);
    });
    let t_g32 = report("Grouped32 (byte strides)", t.seconds, t_floor);
    let got: Vec<f32> = unsafe { kg.read(&bout, n) };
    verify("Grouped32", &got, &g32, &table, &x);

    // ---- what it means ----
    println!("  {}", "-".repeat(84));
    let blocks_4b = 3_633_315_840f64 / 24.0;
    println!("\n  For one Qwen3-4B ({:.0} M blocks) and one token:", blocks_4b / 1e6);
    for (name, per_block, bpw) in [
        ("Fixed96", t_f96 / n as f64, f96.bits_per_weight()),
        ("Grouped32", t_g32 / n as f64, g32.bits_per_weight()),
    ] {
        let decode_ms = blocks_4b * per_block * 1e3;
        let gb = 3_633_315_840f64 * bpw / 8.0 / 1e9 + 0.778;
        println!(
            "    {name:<11} decode {decode_ms:>6.2} ms/token — traffic {gb:.2} GB \
             → memory ceiling {:.0} tok/s",
            400.0 / gb
        );
    }
    println!(
        "\n  Synthetic reference (4 uniform levels, uint4): 0.11 ns/block.\n  \
         The floor reads the same 12 bytes as Fixed96; Grouped32 reads ~{:.1}.",
        g32.data.len() as f64 / n as f64
    );
    Ok(())
}
