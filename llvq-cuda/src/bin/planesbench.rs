//! Planes14 against Slot32 against the FP16 witness, on NVIDIA — plus the
//! M2 arm: `tv_planes12x`, the sparse overlay.
//!
//! `planesbench <model.llvq>` measures the published model: one token's worth
//! of linear algebra over its 252 projections, five arms interleaved in every
//! round — `tv_slot` (the published kernel, untouched), `tv_planes` (layout
//! E1a: uniform 14-byte records, no bases array, explicit bit-plane levels),
//! `tv_planes12x` (M2: a 12-byte main stream capped at 4 levels, the 5-level
//! blocks swapped for their best L ≤ 4 direction and corrected **exactly**
//! by an exception pass in the same launch — one memset + one atomicAdd per
//! row and per exception is the price, timed inside the arm), `tv_golay70`
//! (E2: a 9-byte main stream whose per-slot decode goes through the Golay
//! codeword rank — the residue-mod-4 plane of every block is a codeword —
//! with the E2 exception blocks holed to the origin and added back exactly
//! by the same-launch correction pass) and `tv_f16`.
//! With no argument it falls back to the synthetic seven-shape path, sized
//! past the card's L2 — for iterating on the kernel, nothing else.
//!
//! Since the v2 campaign the bench carries SEVEN arms: the six above (where
//! `tv_golay70` is the v2 decoder) plus `tv_golay70_v1`, the frozen witness
//! copy of the published Golay70 decode (`kernels/golay70_v1.cu`), which
//! shares the v2's device buffers — not one stored byte differs between the
//! two, so the witness arm costs zero VRAM.
//!
//! ## `LLVQ_BENCH_ARMS` — the arm selector (lot B, deviation É1)
//!
//! Semicolon-separated phases of comma-separated arm names (see
//! `llvq_cuda::arms` for the whole contract). One process can therefore run
//! a control table and a full table back to back — the §4 rule of
//! `proofs/preregistration-2026-08-10.md` — with, during each phase, ONLY
//! that phase's buffers resident: a deselected arm builds no transcode, no
//! device buffer, no verification, no timing, no table row. The NVRTC
//! translation unit and the register report never change with the
//! selection; only buffers and dispatch do. After a multi-phase run the
//! bench prints `Δ_contrôle` — the largest relative drift of the common
//! arms between consecutive phases, the number §4's decision rule needs.
//!
//! The protocol is `bin/matvec`'s, reproduced rather than referenced: rounds
//! with warmup discarded, all arms dispatched every round in the same order,
//! ratios formed **round by round** and reported as median with range — never
//! a quotient of two minima. Every row of every matrix is verified against an
//! f64 CPU reference before any timing, and the Planes14 stream is proved a
//! bit-exact bijection of the Slot32 content block by block during the build.
//!
//! The Planes14 stream itself comes from `planes14_from_slot32` (see
//! `src/planes14_host.rs`): a pure bit-level transcoding of the Slot32
//! stream, and the **single point to re-plug** once `llvq-artifact` grows the
//! layout.
//!
//! ## The A4 section, after the table: fusing what shares an input
//!
//! With a model path, a sixth and seventh arm run *after* the five-arm table,
//! in their own interleaved rounds: `tv_slot_seg` and `tv_planes_seg` over the
//! row-concatenation of q+k+v and of gate+up — 144 launches where there were
//! 252, and 768-block grids where k/v alone were 128. The published Slot32
//! delta is 0.803 ms (`docs/mesures/fusion-qkv-cuda-2026-08-05.txt`); the
//! Planes14 one is unknown, which is why **both** are re-timed here rather than
//! one being compared against a number from another job. Correctness first, and
//! bit-exact: a fused row runs the same blocks in the same order with the same
//! centroids, so nothing is reassociated and a tolerance would let a wrong
//! `gs_off` through. The concatenation itself (`src/seg_host.rs`) is proved on
//! the development machine by `tests/planes_segment_matches_unfused.rs`.
//!
//! Cost of the section: one extra transcode of the fusible matrices, and about
//! 4 GB of VRAM for the two fused streams on top of the ~15 GB the five arms
//! already hold. It is skipped entirely on the synthetic path.

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("planesbench targets NVIDIA GPUs; there is nothing to run here.");
}

#[cfg(target_os = "linux")]
fn main() -> Result<(), String> {
    llvq_cuda::check_removed_bench(|v| std::env::var(v).ok())?;
    linux::run()
}

#[cfg(target_os = "linux")]
#[path = "planesbench/linux.rs"]
mod linux;
