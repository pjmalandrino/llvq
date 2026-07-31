//! What decoding a Leech block actually costs on the GPU.
//!
//! This is the number the whole fused-kernel question turns on, and every
//! previous version of it was an instruction count on paper. Three kernels are
//! measured against each other so the answer means something:
//!
//! * **floor** — read the block's bits, write 24 weights. No decoding at all.
//!   Nothing can be faster; whatever this costs is memory and dispatch, not
//!   algorithm.
//! * **masks** — the nested-mask layout: one 24-bit mask per magnitude level,
//!   each lane decoding a whole block serially. This is the candidate.
//! * **rank** — the archive format's permutation unrank, u64 recurrence
//!   (`M' = M·c/n`), same lane-per-block shape. The control: it should be
//!   clearly worse, and by how much tells us what the mask layout buys.
//!
//! One block per lane, structure-of-arrays layout, no inter-lane cooperation —
//! the shape the audit identified as viable after my simdgroup-cooperative
//! estimate turned out to be 32× off.
//!
//! Each kernel **accumulates a dot product and writes one float per block** —
//! the shape of a fused kernel. An earlier version stored the 24 decoded
//! weights instead, and measured its own uncoalesced output: the "floor"
//! kernel came out at 195 GB/s of writes, which is a memory benchmark wearing
//! a decoder costume. A fused kernel never writes decoded weights; neither
//! does this.
//!
//! Every kernel's output is read back and checked against a CPU reference
//! before any timing is reported.
//!
//! Run: `cargo run --release -p llvq-metal --bin decode`

use llvq_core::SplitMix64;

const SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

// A block's decoded weights, as signed 8-bit magnitudes-with-sign. The real
// kernel would multiply these into an accumulator instead of storing them;
// storing is what makes the measurement checkable.
constant uint DIM = 24;

// ---------------------------------------------------------------------------
// floor: no decoding. Read the same bytes, write the same shape.
// ---------------------------------------------------------------------------
kernel void floor_copy(device const uint4*  codes [[buffer(0)]],
                       device const float*  x     [[buffer(1)]],
                       device float*        out   [[buffer(2)]],
                       uint                 gid   [[thread_position_in_grid]],
                       uint                 tid   [[thread_index_in_threadgroup]])
{
    // The activation lives in threadgroup memory, loaded once per group —
    // re-reading it from device memory per iteration made the floor kernel
    // bandwidth-bound at 80 GB/s and hid everything it was meant to expose.
    threadgroup float xs[DIM];
    if (tid < DIM) xs[tid] = x[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint4 c = codes[gid];
    uint acc = c.x ^ c.y ^ c.z ^ c.w;
    float dot = 0.0f;
    for (uint i = 0; i < DIM; ++i) {
        dot += float(char((acc >> (i & 7)) & 0x7f)) * xs[i];
    }
    out[gid] = dot;
}

// ---------------------------------------------------------------------------
// masks: nested-mask arrangement.
//
// Layout per block (uint4):
//   x = mask of level 0 (24 bits)
//   y = mask of level 1, over the slots level 0 left free (24 bits)
//   z = mask of level 2, over what remains (24 bits)
//   w = signs (24 bits) | level count in the top 8
//
// Decoding a slot is: test its bit in mask 0; if clear, test its (compacted)
// bit in mask 1; and so on. The compaction is a popcount of the bits below —
// which is why this is cheap on hardware with a native popcount.
// ---------------------------------------------------------------------------
kernel void decode_masks(device const uint4* codes [[buffer(0)]],
                         device const char4* mags  [[buffer(1)]],
                         device const float* x     [[buffer(2)]],
                         device float*       out   [[buffer(3)]],
                         uint                gid   [[thread_position_in_grid]],
                         uint                tid   [[thread_index_in_threadgroup]])
{
    threadgroup float xs[DIM];
    if (tid < DIM) xs[tid] = x[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint4 c = codes[gid];
    char4 mv = mags[gid];
    uint signs = c.w & 0xffffff;
    float dot = 0.0f;

    for (uint i = 0; i < DIM; ++i) {
        uint bit = 1u << i;
        char v;
        if (c.x & bit) {
            v = mv.x;
        } else {
            // Rank of slot i among those level 0 left free.
            uint below = popcount((~c.x) & (bit - 1));
            uint b1 = 1u << below;
            if (c.y & b1) {
                v = mv.y;
            } else {
                uint below2 = popcount((~c.y) & (b1 - 1));
                v = (c.z & (1u << below2)) ? mv.z : mv.w;
            }
        }
        dot += float((signs & bit) ? -v : v) * xs[i];
    }
    out[gid] = dot;
}

// ---------------------------------------------------------------------------
// rank: the archive format's unrank, u64 recurrence, one block per lane.
//
// Layout per block: rank (2 x uint = u64), packed counts (4 x uint8), m0.
// ---------------------------------------------------------------------------
kernel void decode_rank(device const uint2* ranks  [[buffer(0)]],
                        device const uint2* m0s    [[buffer(1)]],
                        device const uchar4* cnts  [[buffer(2)]],
                        device const char4*  mags  [[buffer(3)]],
                        device const float*  x     [[buffer(4)]],
                        device float*        out   [[buffer(5)]],
                        uint                 gid   [[thread_position_in_grid]],
                        uint                 tid   [[thread_index_in_threadgroup]])
{
    threadgroup float xs[DIM];
    if (tid < DIM) xs[tid] = x[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    ulong rank = (ulong(ranks[gid].y) << 32) | ulong(ranks[gid].x);
    ulong m    = (ulong(m0s[gid].y)   << 32) | ulong(m0s[gid].x);
    uchar4 cnt = cnts[gid];
    char4  mv  = mags[gid];
    uchar counts[4] = { cnt.x, cnt.y, cnt.z, cnt.w };
    char  vals[4]   = { mv.x,  mv.y,  mv.z,  mv.w  };

    ulong n = DIM;
    float dot = 0.0f;
    for (uint slot = 0; slot < DIM; ++slot) {
        for (uint j = 0; j < 4; ++j) {
            ulong cj = counts[j];
            if (cj == 0) continue;
            ulong mj = m * cj / n;      // exact: n divides m*cj
            if (rank < mj) {
                dot += float(vals[j]) * xs[slot];
                counts[j] = uchar(cj - 1);
                m = mj;
                break;
            }
            rank -= mj;
        }
        n -= 1;
    }
    out[gid] = dot;
}
"#;

const DIM: usize = 24;

/// CPU reference for the mask kernel — the same logic, written independently
/// of the shader so a shared misunderstanding cannot pass.
fn decode_masks_cpu(code: [u32; 4], mags: [i8; 4]) -> [i8; DIM] {
    let signs = code[3] & 0xffffff;
    let mut out = [0i8; DIM];
    for (i, o) in out.iter_mut().enumerate() {
        let bit = 1u32 << i;
        let v = if code[0] & bit != 0 {
            mags[0]
        } else {
            let below = ((!code[0]) & (bit - 1)).count_ones();
            let b1 = 1u32 << below;
            if code[1] & b1 != 0 {
                mags[1]
            } else {
                let below2 = ((!code[1]) & (b1 - 1)).count_ones();
                if code[2] & (1u32 << below2) != 0 {
                    mags[2]
                } else {
                    mags[3]
                }
            }
        };
        *o = if signs & bit != 0 { -v } else { v };
    }
    out
}

fn main() -> Result<(), String> {
    // 16.7 M blocks. Sized so the dispatch runs for milliseconds: submission
    // overhead is ~0.15 ms and gets subtracted, but subtracting a number of
    // the same order as the measurement leaves noise, not signal.
    const N: usize = 1 << 24;
    let mut rng = SplitMix64::new(0x6_3E7A);

    // ---- inputs ----
    let codes: Vec<[u32; 4]> = (0..N)
        .map(|_| {
            [
                (rng.next() & 0xffffff) as u32,
                (rng.next() & 0xffffff) as u32,
                (rng.next() & 0xffffff) as u32,
                (rng.next() & 0xffffff) as u32,
            ]
        })
        .collect();
    let mags: Vec<[i8; 4]> = (0..N).map(|_| [7, 5, 3, 1]).collect();
    // The activation vector a fused kernel would be multiplying against.
    let xvec: Vec<f32> = (0..DIM).map(|i| 1.0 + i as f32 * 0.125).collect();

    let floor = llvq_metal::Kernel::new(SRC, "floor_copy")?;
    println!("GPU : {}", floor.device_name());
    let overhead = floor.overhead(20);
    println!("  surcoût de soumission {:.3} ms\n", overhead * 1e3);

    let bcodes = floor.buffer(&codes);
    let bmags = floor.buffer(&mags);
    let bx = floor.buffer(&xvec);
    let bout = floor.empty::<f32>(N);

    let report = |name: &str, secs: f64, floor_secs: f64| {
        let net = secs - overhead;
        let per_block = net / N as f64;
        println!(
            "  {name:<18}{:>9.3} ms{:>12.2} ns/bloc{:>14.2e} blocs/s{:>9}",
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
        "  {:<18}{:>12}{:>19}{:>19}{:>9}",
        "noyau", "temps", "par bloc", "débit", "vs sol"
    );
    println!("  {}", "-".repeat(80));

    // ---- floor ----
    let t = floor.time(N as u64, 256, 3, 15, |enc| {
        enc.set_buffer(0, Some(&bcodes), 0);
        enc.set_buffer(1, Some(&bx), 0);
        enc.set_buffer(2, Some(&bout), 0);
    });
    let t_floor = report("sol (aucun décodage)", t.seconds, 0.0);

    // ---- masks ----
    let masks = llvq_metal::Kernel::new(SRC, "decode_masks")?;
    let t = masks.time(N as u64, 256, 3, 15, |enc| {
        enc.set_buffer(0, Some(&bcodes), 0);
        enc.set_buffer(1, Some(&bmags), 0);
        enc.set_buffer(2, Some(&bx), 0);
        enc.set_buffer(3, Some(&bout), 0);
    });
    let t_masks = report("masques imbriqués", t.seconds, t_floor);
    // Verify against the CPU reference before believing the timing.
    let got: Vec<f32> = unsafe { masks.read(&bout, N) };
    for i in [0usize, 1, 5000, N - 1] {
        let w = decode_masks_cpu(codes[i], mags[i]);
        let want: f32 = w.iter().zip(&xvec).map(|(a, b)| *a as f32 * b).sum();
        assert!(
            (got[i] - want).abs() < 1e-3 * want.abs().max(1.0),
            "mask kernel disagrees with the CPU reference at block {i}: {} vs {want}",
            got[i]
        );
    }

    // ---- rank ----
    let counts: Vec<[u8; 4]> = (0..N).map(|_| [11u8, 7, 4, 2]).collect();
    // multinomial(11,7,4,2) over 24 slots.
    let m0: u64 = {
        let mut f = [1u128; 25];
        for i in 1..25 {
            f[i] = f[i - 1] * i as u128;
        }
        (f[24] / (f[11] * f[7] * f[4] * f[2])) as u64
    };
    let ranks: Vec<[u32; 2]> = (0..N)
        .map(|_| {
            let r = rng.next() % m0;
            [r as u32, (r >> 32) as u32]
        })
        .collect();
    let m0s: Vec<[u32; 2]> = vec![[m0 as u32, (m0 >> 32) as u32]; N];

    let rank = llvq_metal::Kernel::new(SRC, "decode_rank")?;
    let branks = rank.buffer(&ranks);
    let bm0 = rank.buffer(&m0s);
    let bcnts = rank.buffer(&counts);
    let t = rank.time(N as u64, 256, 3, 15, |enc| {
        enc.set_buffer(0, Some(&branks), 0);
        enc.set_buffer(1, Some(&bm0), 0);
        enc.set_buffer(2, Some(&bcnts), 0);
        enc.set_buffer(3, Some(&bmags), 0);
        enc.set_buffer(4, Some(&bx), 0);
        enc.set_buffer(5, Some(&bout), 0);
    });
    let t_rank = report("rang (récurrence u64)", t.seconds, t_floor);

    // ---- what it means ----
    println!("  {}", "-".repeat(80));
    let blocks_4b = 3_633_315_840f64 / 24.0;
    println!("\n  Pour un Qwen3-4B ({:.0} M blocs) et un token :", blocks_4b / 1e6);
    for (name, per_block) in [
        ("masques", t_masks / N as f64),
        ("rang", t_rank / N as f64),
    ] {
        let decode_ms = blocks_4b * per_block * 1e3;
        println!(
            "    {name:<10} décodage seul {decode_ms:>7.2} ms  →  plafond {:>6.0} tok/s",
            1000.0 / decode_ms
        );
    }
    println!(
        "\n  À comparer au plafond mémoire : ~190-200 tok/s (le décodage doit\n  \
         se cacher dessous, pas s'y ajouter — dans un noyau fusé les deux se\n  \
         recouvrent)."
    );
    Ok(())
}
