//! The launch constants every child builds its grids on: the quad order, the SIMD-group width, the threadgroup
//! size and the threadgroup memory the kernels are compiled against. One file because they are facts about the
//! device and the shaders, not about any one launch.

// Read by intra-doc links only; rustc never compiles it, so it cannot go unused.
#[cfg(doc)]
use super::types::MetalRuntime;

/// The coordinate order the quads come out in. Mirrors `TETRA48_ORDER` in the
/// shader and `TETRA48_ORDER` in `llvq_tetra48.cuh`.
pub(super) const ORDER: [usize; llvq_core::DIM] = [
    0, 1, 2, 3, 4, 7, 10, 12, 6, 11, 13, 14, 16, 17, 18, 19, 5, 8, 9, 15, 20, 21, 22, 23,
];

/// Lanes in an Apple SIMD-group. The kernel gives one to a row.
pub(super) const LANES: usize = 32;
/// Threads a threadgroup, so eight rows each. The kernel carries no
/// `row < d_out` guard, exactly like the CUDA one, so `d_out` must be a
/// multiple of the rows a group covers. Asserted at upload.
pub(super) const GROUP: usize = 256;
/// Lanes `tv_tetra48_metal_ilp` gives one output row, and why the served
/// path does NOT use it.
///
/// Eight lanes instead of 32 gives each lane 13 blocks rather than 3.3, which
/// is what the roofline asks for: the 32-lane kernel reaches 22 % of the
/// measured 340 GB/s AND 19 % of the measured 12.5 TFLOPS, and a kernel that
/// saturates neither is waiting on latency.
///
/// It works, and it is refused. The lane count is part of the ARITHMETIC:
/// each lane accumulates a different subset and the butterfly has a different
/// depth, so the answer moves in the last bits. Measured 2026-09-21: the
/// 60-token identity against the dense arm BREAKS at token 57, "dans
/// l'atmosphere" becoming "dans la atmosphere", while the throughput gain in
/// situ is 5 percent, 48.1 to 50.4 tok/s. The isolated bench had promised 15.
///
/// Five percent is not worth the property the whole path is gated on.
#[allow(dead_code)]
const ROW_LANES: usize = 8;
/// Threadgroup memory an M3 Max offers, measured 2026-09-20 through
/// `MTLDevice.maxThreadgroupMemoryLength`.
///
/// A constant rather than a device query on purpose: the kernels are compiled
/// against it, so a machine that offered less has to be refused rather than
/// quietly served a kernel that stages past its limit. Asking the device and
/// adapting would hide exactly that. [`MetalRuntime::new`] and
/// [`MetalRuntime::from_tables`] refuse such a device by name
/// (`check_threadgroup_memory`).
pub(super) const THREADGROUP_LIMIT: usize = 32_768;
