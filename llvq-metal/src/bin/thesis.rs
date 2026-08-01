//! The thesis bench: **one token's worth of linear algebra**, for the whole
//! model, LLVQ against FP16 on the same machine.
//!
//! Every earlier number was one layer. This is the claim itself — a 2-bit
//! model is smaller *and* faster than the FP16 it replaces — measured on all
//! 252 projection matrices of the published Qwen3-4B, in model order, in one
//! command buffer per format.
//!
//! ## Why one command buffer, and why that is the honest shape
//!
//! * **Naturally cold.** A pass touches 2.4 GB (LLVQ) or 7.3 GB (FP16) of
//!   distinct weights — nothing is re-read, so no matrix can hide in the
//!   48 MB system cache. The `matvec` bench had to force this with rotating
//!   buffer copies; here the workload does it by construction.
//! * **Submission overhead amortized honestly.** 252 dispatches share one
//!   commit, exactly as a real decode step would.
//! * **Serialized by the real dependency.** Every dispatch writes the same
//!   output buffer, so the write-write hazard drains between them — the same
//!   mechanism that serializes a transformer's actually-dependent layers.
//!
//! What it deliberately leaves out: attention, norms, activations, the
//! rotation applied to `x`, and the tied `lm_head` (unquantized in this
//! artifact, and identical for both sides). It measures the part
//! quantization changes, and reports what the rest costs on top.
//!
//! ## Tiling, and the shape the single-layer bench could not run
//!
//! `matvec` staged the whole activation in threadgroup memory — fine at
//! d_in = 2560 (10 KB), impossible at d_in = 9728 (38 KB against Metal's
//! 32 KB limit). The 36 `down_proj` matrices could not have run at all. Both
//! kernels here tile the activation in 128-block (3072-column, 12 KB) chunks,
//! so every shape of the model goes through the same code.
//!
//! ## Verification
//!
//! Every matrix's output — all 252, every row — is checked against an f64
//! CPU reference built from the transcoded blocks, before any timing is
//! believed. The LLVQ reference goes through `RuntimeBlocks::decode_block`
//! (pinned bit-for-bit on `Indexer::decode`); the FP16 one through the same
//! weights rounded to binary16. Errors are reported relative to Σ|wᵢ·xᵢ|,
//! the only scale a float dot product is accountable to.
//!
//! Run: `cargo run --release -p llvq-metal --bin thesis [model.llvq]`

use llvq_artifact::runtime::{transcode, ClassTable, Layout};
use llvq_core::{Leech, SplitMix64, DIM};
use llvq_search::fastdec::FastDecoder;
use std::fs::File;
use std::io::BufReader;
use std::time::Instant;

/// Blocks of activation staged per tile — 3072 columns, 12 KB, under the
/// 32 KB threadgroup budget with room for occupancy.
const TILE_BLOCKS: usize = 128;

const SRC: &str = r#"
struct Params { uint d_in; uint d_out; uint nblocks; uint tail_w; };

constant uint TILE_BLOCKS = 128;
constant uint TILE_COLS = TILE_BLOCKS * DIM;   // 3072

// ---------------------------------------------------------------------------
// FP16 baseline, tiled. One SIMD group per row, half4 loads.
// ---------------------------------------------------------------------------
kernel void tv_f16(device const half*   w   [[buffer(0)]],
                   device const float*  x   [[buffer(1)]],
                   device float*        y   [[buffer(2)]],
                   constant Params&     P   [[buffer(3)]],
                   threadgroup float*   xs  [[threadgroup(0)]],
                   uint gid  [[thread_position_in_grid]],
                   uint tid  [[thread_index_in_threadgroup]],
                   uint tgs  [[threads_per_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]])
{
    uint row = gid >> 5;
    float acc = 0.0f;
    uint ntiles = (P.d_in + TILE_COLS - 1u) / TILE_COLS;
    for (uint t = 0; t < ntiles; ++t) {
        uint c0 = t * TILE_COLS;
        uint n = min(TILE_COLS, P.d_in - c0);
        // Two barriers: the second orders the fill against the readers, the
        // first stops the next fill from racing the previous tile's readers.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < n; i += tgs) xs[i] = x[c0 + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device const half4* w4 =
            (device const half4*)(w + row * P.d_in + c0);
        for (uint j = lane; j < (n >> 2); j += 32) {
            float4 wv = float4(w4[j]);
            uint c = j << 2;
            acc += wv.x * xs[c] + wv.y * xs[c + 1]
                 + wv.z * xs[c + 2] + wv.w * xs[c + 3];
        }
    }
    acc = simd_sum(acc);
    if (lane == 0) y[row] = acc;
}

// ---------------------------------------------------------------------------
// LLVQ fused, tiled. `slot_dot` comes from the shared PAYLOAD_MSL — the
// same decoder the single-layer bench measured at 2.2x.
// ---------------------------------------------------------------------------
kernel void tv_slot(device const uint*   words  [[buffer(0)]],
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
    uint row = gid >> 5;
    uint b0r = row * P.nblocks;
    float acc = 0.0f;
    uint ntiles = (P.nblocks + TILE_BLOCKS - 1u) / TILE_BLOCKS;
    for (uint t = 0; t < ntiles; ++t) {
        uint jlo = t * TILE_BLOCKS;
        uint jhi = min(jlo + TILE_BLOCKS, P.nblocks);
        uint c0 = jlo * DIM;
        uint n = (jhi - jlo) * DIM;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < n; i += tgs) xs[i] = x[c0 + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint j = jlo + lane; j < jhi; j += 32) {
            acc += slot_dot(words, bases, tab, gscale, b0r + j,
                            xs + (j - jlo) * DIM);
        }
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        // The KeepExact tail columns fall outside the block tiling, so they
        // are read straight from device memory — at most 23 floats per row.
        float tv = 0.0f;
        uint c0 = P.nblocks * DIM;
        for (uint i = 0; i < P.tail_w; ++i) {
            tv += tail[row * P.tail_w + i] * x[c0 + i];
        }
        y[row] = acc * rscale[row] + tv;
    }
}
"#;

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    d_in: u32,
    d_out: u32,
    nblocks: u32,
    tail_w: u32,
}

/// Everything one matrix needs on the GPU, plus what it costs.
struct Mat {
    name: String,
    d_out: usize,
    params: metal::Buffer,
    // LLVQ side.
    words: metal::Buffer,
    bases: metal::Buffer,
    gscale: metal::Buffer,
    rscale: metal::Buffer,
    tail: metal::Buffer,
    slot_bytes: u64,
    // FP16 side.
    w16: metal::Buffer,
    f16_bytes: u64,
    // Verification.
    y_ref: Vec<f64>,
    y16_ref: Vec<f64>,
    scale: Vec<f64>,
}

/// `max |got − want| / Σ|wᵢxᵢ|` over the rows.
fn worst_error(got: &[f32], want: &[f64], scale: &[f64]) -> f64 {
    got.iter()
        .zip(want)
        .zip(scale)
        .map(|((&g, &w), &s)| (g as f64 - w).abs() / s.max(1e-12))
        .fold(0.0, f64::max)
}

fn main() -> Result<(), String> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| format!("{}/llvq-q4b.llvq", std::env::var("HOME").unwrap()));

    let src = format!("{}{}", llvq_metal::PAYLOAD_MSL, SRC);
    let kf16 = llvq_metal::Kernel::new(&src, "tv_f16")?;
    let kslot = llvq_metal::Kernel::new(&src, "tv_slot")?;
    println!("GPU : {}\n", kf16.device_name());

    let fd = FastDecoder::new();
    let recs = llvq_metal::gpu_class_table(&fd);
    let btab = kf16.buffer(&recs);

    // One activation, long enough for the widest layer. Which vector it is
    // does not change the cost; that it is the *same* one for both formats
    // does matter, and it is.
    let f = File::open(&path).map_err(|e| format!("open {path}: {e}"))?;
    let mut r = BufReader::new(f);
    let h = llvq_artifact::read_header(&mut r).map_err(|e| e.to_string())?;
    let mut rng = SplitMix64::new(0x6_7451);
    let xmax: Vec<f32> = (0..16384).map(|_| rng.next_gaussian() as f32).collect();
    let bx = kf16.buffer(&xmax);
    let mut by_len = 0usize;

    println!("Chargement et transcodage des {} matrices…", h.matrices);
    let t0 = Instant::now();
    let mut mats: Vec<Mat> = Vec::with_capacity(h.matrices as usize);
    let mut n_weights = 0u64;

    for mi in 0..h.matrices {
        let m = llvq_artifact::read_matrix_raw(&mut r).map_err(|e| e.to_string())?;
        let (d_out, d_in) = (m.d_out, m.d_in);
        let nblocks = d_in / DIM;
        let tail_w = d_in % DIM;
        assert_eq!(d_in % 4, 0, "{}: the FP16 kernel loads half4", m.name);
        assert_eq!(d_out % 8, 0, "{}: rows must fill whole threadgroups", m.name);
        assert!(d_in <= xmax.len(), "{}: d_in {d_in} exceeds the activation", m.name);
        let gain_bits = m.centroids.len().next_power_of_two().trailing_zeros();
        let table = ClassTable::new(&fd, gain_bits);
        let rt = transcode(&fd, &table, &m.indices, &m.gains, Layout::Slot32)
            .map_err(|e| e.to_string())?;
        n_weights += (d_out * d_in) as u64;
        by_len = by_len.max(d_out);

        // Dequantized weights, f64, in the rotated basis the matvec runs in.
        // Held for this matrix only — the whole model in f64 would be 29 GB.
        let mut w = vec![0.0f64; d_out * d_in];
        for row in 0..d_out {
            for p in 0..nblocks {
                let (pt, gain) = rt.decode_block(&table, row * nblocks + p);
                if let Some(shell) = Leech::shell_index(&pt).filter(|&s| s > 0) {
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
        let w16: Vec<u16> = w.iter().map(|&v| llvq_metal::f16_bits(v as f32)).collect();

        // References and the error scale, f64.
        let mut y_ref = vec![0.0f64; d_out];
        let mut y16_ref = vec![0.0f64; d_out];
        let mut scale = vec![0.0f64; d_out];
        for row in 0..d_out {
            let (mut a, mut b, mut s) = (0.0, 0.0, 0.0);
            for c in 0..d_in {
                let xv = xmax[c] as f64;
                let wv = w[row * d_in + c];
                a += wv * xv;
                b += llvq_metal::f16_to_f64(w16[row * d_in + c]) * xv;
                s += (wv * xv).abs();
            }
            y_ref[row] = a;
            y16_ref[row] = b;
            scale[row] = s;
        }
        drop(w);

        let mut words = rt.data.clone();
        words.extend_from_slice(&[0u8; 20]); // the 5-word read of the last block
        let params = [Params {
            d_in: d_in as u32,
            d_out: d_out as u32,
            nblocks: nblocks as u32,
            tail_w: tail_w as u32,
        }];
        let gscale: Vec<f32> = m.centroids.iter().map(|&c| c as f32).collect();
        let rscale: Vec<f32> = m.row_scales.iter().map(|&s| s as f32).collect();
        let tailf: Vec<f32> = m.tail.iter().map(|&t| t as f32).collect();
        let slot_bytes = rt.data.len() as u64
            + rt.bases.len() as u64 * 4
            + (d_out * tail_w) as u64 * 4
            + d_out as u64 * 4;

        mats.push(Mat {
            name: m.name.clone(),
            d_out,
            params: kf16.buffer(&params),
            words: kf16.buffer(&words),
            bases: kf16.buffer(&rt.bases),
            gscale: kf16.buffer(&gscale),
            rscale: kf16.buffer(&rscale),
            tail: kf16.buffer(if tailf.is_empty() { &[0.0f32] } else { &tailf }),
            slot_bytes,
            w16: kf16.buffer(&w16),
            f16_bytes: (d_out * d_in * 2) as u64,
            y_ref,
            y16_ref,
            scale,
        });
        if mi % 36 == 0 {
            println!(
                "  {mi:>3}/{}  {} ({d_out}×{d_in})",
                h.matrices, mats[mats.len() - 1].name
            );
        }
    }
    let load_s = t0.elapsed().as_secs_f64();
    let by = kf16.empty::<f32>(by_len);
    println!(
        "  {} matrices, {:.2} Md de poids, transcodées en {load_s:.0} s\n",
        mats.len(),
        n_weights as f64 / 1e9
    );

    let tg = 256u64;
    let xs_bytes = (TILE_BLOCKS * DIM * 4) as u64;
    let bind_f16 = |enc: &metal::ComputeCommandEncoderRef, m: &Mat| {
        enc.set_buffer(0, Some(&m.w16), 0);
        enc.set_buffer(1, Some(&bx), 0);
        enc.set_buffer(2, Some(&by), 0);
        enc.set_buffer(3, Some(&m.params), 0);
        llvq_metal::Kernel::set_threadgroup_memory(enc, 0, xs_bytes);
    };
    let bind_slot = |enc: &metal::ComputeCommandEncoderRef, m: &Mat| {
        enc.set_buffer(0, Some(&m.words), 0);
        enc.set_buffer(1, Some(&m.bases), 0);
        enc.set_buffer(2, Some(&btab), 0);
        enc.set_buffer(3, Some(&m.gscale), 0);
        enc.set_buffer(4, Some(&m.rscale), 0);
        enc.set_buffer(5, Some(&m.tail), 0);
        enc.set_buffer(6, Some(&bx), 0);
        enc.set_buffer(7, Some(&by), 0);
        enc.set_buffer(8, Some(&m.params), 0);
        llvq_metal::Kernel::set_threadgroup_memory(enc, 0, xs_bytes);
    };

    // ---- verify every matrix, every row, before any timing ----
    println!("Vérification des {} matrices contre la référence CPU f64…", mats.len());
    let (mut worst_q, mut worst_f, mut worst_name) = (0.0f64, 0.0f64, String::new());
    for m in &mats {
        let threads = (m.d_out * 32) as u64;
        kf16.dispatch(threads, tg, |e: &metal::ComputeCommandEncoderRef| bind_f16(e, m));
        let got: Vec<f32> = unsafe { kf16.read(&by, m.d_out) };
        let e = worst_error(&got, &m.y16_ref, &m.scale);
        assert!(e < 1e-3, "{}: FP16 kernel off by {e:.2e}·Σ|w·x|", m.name);
        worst_f = worst_f.max(e);

        kslot.dispatch(threads, tg, |e: &metal::ComputeCommandEncoderRef| bind_slot(e, m));
        let got: Vec<f32> = unsafe { kslot.read(&by, m.d_out) };
        let e = worst_error(&got, &m.y_ref, &m.scale);
        assert!(e < 1e-3, "{}: LLVQ kernel off by {e:.2e}·Σ|w·x|", m.name);
        if e > worst_q {
            worst_q = e;
            worst_name = m.name.clone();
        }
    }
    let rows: usize = mats.iter().map(|m| m.d_out).sum();
    println!(
        "  {rows} lignes sur {} matrices — pire erreur LLVQ {worst_q:.1e}·Σ|w·x| ({worst_name}),\n  \
         pire erreur FP16 {worst_f:.1e}·Σ|w·x|\n",
        mats.len()
    );

    // ---- one command buffer = one token's linear work ----
    let pass = |k: &llvq_metal::Kernel, slot: bool| -> f64 {
        let mut best = f64::INFINITY;
        for rep in 0..7 {
            let t = k.dispatch_batch(
                |enc, i| {
                    let m = &mats[i];
                    if slot {
                        bind_slot(enc, m);
                    } else {
                        bind_f16(enc, m);
                    }
                    ((m.d_out * 32) as u64, tg)
                },
                mats.len(),
            );
            if rep > 1 {
                best = best.min(t.seconds);
            }
        }
        best
    };
    let t16 = pass(&kf16, false);
    let tq = pass(&kslot, true);

    let f16_bytes: u64 = mats.iter().map(|m| m.f16_bytes).sum();
    let slot_bytes: u64 = mats.iter().map(|m| m.slot_bytes).sum();
    let bpw = slot_bytes as f64 * 8.0 / n_weights as f64;

    println!("UN TOKEN — les 252 projections, un command buffer, mémoire froide");
    println!("  {}", "-".repeat(72));
    println!(
        "  {:<26}{:>10}{:>12}{:>12}{:>10}",
        "format", "ms", "Go lus", "Go/s", "vs FP16"
    );
    println!(
        "  {:<26}{:>10.3}{:>12.2}{:>12.0}{:>10}",
        "FP16",
        t16 * 1e3,
        f16_bytes as f64 / 1e9,
        f16_bytes as f64 / t16 / 1e9,
        "1.00×"
    );
    println!(
        "  {:<26}{:>10.3}{:>12.2}{:>12.0}{:>10}",
        "LLVQ fusé (Slot32)",
        tq * 1e3,
        slot_bytes as f64 / 1e9,
        slot_bytes as f64 / tq / 1e9,
        format!("{:.2}×", t16 / tq)
    );
    println!("  {}", "-".repeat(72));
    println!(
        "\n  poids : {:.2} Md à {bpw:.3} b/poids — {:.2} Go contre {:.2} Go en FP16 (×{:.2})",
        n_weights as f64 / 1e9,
        slot_bytes as f64 / 1e9,
        f16_bytes as f64 / 1e9,
        f16_bytes as f64 / slot_bytes as f64
    );

    // The tied lm_head is unquantized in this artifact and read once per
    // token by both sides — it is the same constant added to each, and it is
    // what caps the end-to-end ratio.
    let head_bytes = 389_070_848f64 * 2.0;
    let bw = f16_bytes as f64 / t16; // this machine's measured streaming rate
    let head_s = head_bytes / bw;
    println!("\n  Débit d'un pas de décodage (projections + lm_head f16 non quantifié) :");
    println!("  {}", "-".repeat(72));
    for (name, t) in [("FP16", t16), ("LLVQ", tq)] {
        let total = t + head_s;
        println!(
            "  {name:<26}{:>8.2} ms{:>10.1} tok/s{:>14}",
            total * 1e3,
            1.0 / total,
            format!("(dont lm_head {:.2} ms)", head_s * 1e3)
        );
    }
    println!(
        "\n  Le lm_head lié ({:.0} M poids, f16) n'est pas quantifié dans cet artefact\n  \
         et coûte la même chose aux deux côtés : c'est lui qui plafonne le rapport\n  \
         de bout en bout. Attention, normes et activations ne sont pas mesurées ici.",
        389_070_848f64 / 1e6
    );
    Ok(())
}
