// DRAFT, never compiled or run. Wave 5 deep dive, lead "mac", 2026-11-02.
//
// Proposed as `llvq-metal/src/bin/q4bench.rs`: the isolated rate of the two
// int4 kernels at the shapes of the sealed files, and the tile's residency
// effect, before prereg-mac-1's predictions are signed (mac.md, L2b).
//
// IT INFORMS ONLY. Isolated benches over-predicted in situ on this Mac by 9 to
// 18 % (commit 31a5acb) and by 15 % against 5 % (commit 5ec72b0). Its range,
// divided by 1.09-1.18, replaces the 75-250 GB/s bracket in prereg-mac-1's
// arithmetic before that file is stamped; nothing is decided on it. It runs
// under a one-page note stamped before the first measurement (METHODE section 1).
//
// Protocol (docs/format-noyau.md section 9, METHODE section 2):
//   - one process, all arms in the same order every round, 2 discarded + 7 timed,
//     ratios round by round;
//   - fast math ON (`Kernel::new`), the shipped compile (fused_metal.rs:397);
//     the bench of 31a5acb compiled with it off, one of its two defects;
//   - the DRAM regime forced cold: each arm rotates through enough distinct
//     weight copies to pass 96 MB, twice the 48 MB SLC;
//   - one shared, tracked output buffer, so the dispatches serialize on the
//     write-write hazard (`dispatch_many`'s doc), as one projection after
//     another does in a decode step;
//   - each measurement moves at least 1 GB, so the ~0.15 ms commit is under
//     5 % of it;
//   - GB/s = matrix bytes (int4, plus an f16 scale and bias per 128) x
//     dispatches / seconds. The activation (10 to 70 KB) is not counted.
//
// Usage, on a quiet machine, the binary by path:
//   nice -n 15 cargo build --release -p llvq-metal --bin q4bench -j 4
//   nice -n 10 target/release/q4bench 7

// The other llvq-metal binaries build a stub off macOS (matvec.rs:773); so does this one.
#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("q4bench measures a Metal device; it runs on macOS only");
    std::process::exit(2);
}

#[cfg(target_os = "macos")]
use llvq_core::SplitMix64;
#[cfg(target_os = "macos")]
use llvq_metal::{f16_bits, Kernel};

#[cfg(target_os = "macos")]
const SOURCE: &str = include_str!("../../../llvq-llm/kernels/tv_q4_h.metal");
const LANES: u64 = 32;
const GROUP: u64 = 256;
const DISCARDED: usize = 2;
const COLD_BYTES: f64 = 96e6;
const BYTES_A_MEASUREMENT: f64 = 1e9;

struct Arm {
    label: String,
    kernel: &'static str,
    d_out: usize,
    d_in: usize,
    tile: Option<usize>,
}

fn arms() -> Vec<Arm> {
    let mut v = vec![
        Arm { label: "untiled 1024x2560 (v, 4B)".into(), kernel: "tv_q4_metal", d_out: 1024, d_in: 2560, tile: None },
        Arm { label: "untiled 2560x4096 (o, 4B)".into(), kernel: "tv_q4_metal", d_out: 2560, d_in: 4096, tile: None },
    ];
    for (d_out, d_in, what) in [(2560usize, 9728usize, "down, 4B"), (4096, 12288, "down, 8B"), (5120, 17408, "down, 14B")] {
        for tile in [1024usize, 2048, 8192] {
            v.push(Arm {
                label: format!("tiled {d_out}x{d_in} ({what}) at {tile}"),
                kernel: "tv_q4_metal_tiled",
                d_out,
                d_in,
                tile: Some(tile),
            });
        }
    }
    v
}

fn bytes_of(d_out: usize, d_in: usize) -> f64 {
    (d_out * d_in) as f64 * 4.25 / 8.0
}

// (The constants, `Arm`, `arms` and `bytes_of` above carry the same
// `#[cfg(target_os = "macos")]` in the real file; omitted here for reading.)
#[cfg(target_os = "macos")]
fn main() {
    let timed: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(7);
    let arms = arms();
    let mut rng = SplitMix64::new(0x0_4BE_4C);

    // Compile once, upload once: the copies, the activation, one output each.
    struct Ready {
        k: Kernel,
        copies: Vec<(metal::Buffer, metal::Buffer, metal::Buffer)>,
        x: metal::Buffer,
        y: metal::Buffer,
        dispatches: usize,
    }
    let ready: Vec<Ready> = arms
        .iter()
        .map(|a| {
            let k = Kernel::new(SOURCE, a.kernel).expect("compiles");
            let bytes = bytes_of(a.d_out, a.d_in);
            let n_copies = ((COLD_BYTES / bytes).ceil() as usize).max(4);
            let gpr = a.d_in / 128;
            let copies = (0..n_copies)
                .map(|_| {
                    let words: Vec<u32> = (0..a.d_out * a.d_in / 8).map(|_| rng.next() as u32).collect();
                    let scales = vec![f16_bits(0.01); a.d_out * gpr];
                    let biases = vec![f16_bits(-0.08); a.d_out * gpr];
                    (k.buffer(&words), k.buffer(&scales), k.buffer(&biases))
                })
                .collect::<Vec<_>>();
            let x: Vec<f32> = (0..a.d_in).map(|i| ((i % 17) as f32 - 8.0) / 8.0).collect();
            let dispatches = ((BYTES_A_MEASUREMENT / bytes).ceil() as usize).max(n_copies);
            Ready { x: k.buffer(&x), y: k.empty::<f32>(a.d_out), copies, dispatches, k }
        })
        .collect();

    println!("q4bench: {} arms, fast math on, cold over >= 96 MB, {DISCARDED} discarded + {timed} timed rounds", arms.len());
    let mut gbs = vec![Vec::new(); arms.len()];
    for r in 0..DISCARDED + timed {
        for (i, (a, rd)) in arms.iter().zip(&ready).enumerate() {
            let (d_in, gpr) = (a.d_in as u32, (a.d_in / 128) as u32);
            let tc = a.tile.unwrap_or(0) as u32;
            let staged = a.tile.unwrap_or(a.d_in) * 4;
            let t = rd.k.dispatch_many(a.d_out as u64 * LANES, GROUP, rd.dispatches, |enc, j| {
                let (wq, sc, bi) = &rd.copies[j % rd.copies.len()];
                enc.set_buffer(0, Some(wq), 0);
                enc.set_buffer(1, Some(sc), 0);
                enc.set_buffer(2, Some(bi), 0);
                enc.set_buffer(3, Some(&rd.x), 0);
                enc.set_buffer(4, Some(&rd.y), 0);
                enc.set_bytes(5, 4, &d_in as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(6, 4, &gpr as *const u32 as *const std::ffi::c_void);
                if a.tile.is_some() {
                    enc.set_bytes(7, 4, &tc as *const u32 as *const std::ffi::c_void);
                }
                enc.set_threadgroup_memory_length(0, staged as u64);
            });
            if r >= DISCARDED {
                gbs[i].push(bytes_of(a.d_out, a.d_in) * rd.dispatches as f64 / t.seconds / 1e9);
            }
        }
    }

    let stats = |v: &[f64]| {
        let mut s = v.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        (s[s.len() / 2], s[0], s[s.len() - 1])
    };
    for (a, g) in arms.iter().zip(&gbs) {
        let (m, lo, hi) = stats(g);
        println!("{:<40} {m:6.1} GB/s [{lo:.1}-{hi:.1}] (measured, isolated)", a.label);
    }
    // The tile's residency effect, round by round, at each shape.
    for base in (2..arms.len()).step_by(3) {
        for (x, y) in [(base + 2, base + 1), (base, base + 1)] {
            let q: Vec<f64> = gbs[x].iter().zip(&gbs[y]).map(|(p, q)| p / q).collect();
            let (m, lo, hi) = stats(&q);
            println!("{} over {}: x{m:.3} [{lo:.3}-{hi:.3}]", arms[x].label, arms[y].label);
        }
    }
}
