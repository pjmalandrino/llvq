//! Occupancy arithmetic that outlived A3: the staged block width and the
//! residency model.
//!
//! A3's occupancy arms (`LLVQ_SEG_ARMS`, `kernels/planes_occ.cu`) had their
//! verdict on 2026-09-02 (`docs/mesures/a3-occupation-banc-2026-09-01.txt`:
//! `persall` passed the bench gate and became the persistent line; the seven
//! others stayed under it) and were removed with their registry. What stays
//! here is what other code still reads: [`XS_DIM`] (`tile.rs`) and
//! [`residency`] (`bin/f1rankfloor`).
//!
//! Portable on purpose, for the reason `arms.rs` and `shared.rs` give: the
//! benches are `cfg(target_os = "linux")`, and arithmetic that sizes a launch
//! is tested where tests are free.

/// Floats per staged block under the reference stride: one block.
/// A literal, not `llvq_core::DIM` — that crate is a Linux-only dependency
/// of this one — and pinned to it by a test where the dev-dependency exists.
pub const XS_DIM: usize = 24;

/// CTAs of `threads` threads, `regs` registers a thread and `shared` bytes of
/// dynamic shared memory that one SM holds at once — the smallest of the
/// three limits, every one of them read off the card and the loaded function.
///
/// A model, not a measurement: the hardware allocates registers by warp in
/// granules and shared memory in 128-byte units, both of which round UP the
/// real footprint. The number is therefore an upper bound on residency, and
/// it is printed next to the grid it sizes so a reader can see it.
pub fn residency(
    regs: u32,
    threads: u32,
    shared: u32,
    max_threads_sm: u32,
    regs_sm: u32,
    shared_sm: u32,
) -> u32 {
    let by_threads = max_threads_sm / threads;
    let by_regs = regs_sm / (regs.max(1) * threads);
    let by_shared = shared_sm.checked_div(shared).unwrap_or(u32::MAX);
    by_threads.min(by_regs).min(by_shared)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_block_width_is_the_lattice_dimension() {
        assert_eq!(XS_DIM, llvq_core::DIM);
    }

    /// The L40S numbers of the A3 design note: 40 registers, 256 threads,
    /// 12,288 bytes → 6 CTAs an SM, 852 on 142 SM. A 28-float staging
    /// (14,336 bytes) still allows 7 by shared memory, so threads keep the
    /// limit at 6.
    #[test]
    fn residency_reproduces_the_l40s_figure() {
        let (thr, regs_sm, sh_sm) = (1536, 65_536, 102_400);
        assert_eq!(residency(40, 256, 12_288, thr, regs_sm, sh_sm), 6);
        assert_eq!(residency(40, 256, 14_336, thr, regs_sm, sh_sm), 6);
        assert_eq!(6 * 142, 852);
        // Registers become the limit past 42 a thread.
        assert_eq!(residency(48, 256, 12_288, thr, regs_sm, sh_sm), 5);
        assert_eq!(residency(64, 256, 12_288, thr, regs_sm, sh_sm), 4);
        // Shared memory becomes the limit for a whole-row staging.
        assert_eq!(residency(40, 256, 38_912, thr, regs_sm, sh_sm), 2);
        assert_eq!(residency(40, 256, 0, thr, regs_sm, sh_sm), 6);
    }
}
