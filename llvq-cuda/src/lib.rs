//! The fused Leech kernel, on NVIDIA hardware.
//!
//! `llvq-metal` measured one number on one GPU: what decoding a 24-weight
//! block costs, and what the fused decode+matvec buys over FP16. It answered
//! that on Apple silicon — 2.09× over the whole model, `docs/mesures/`. This
//! crate is the same question asked of a card anyone can rent, which is the
//! only way the answer becomes portable rather than anecdotal.
//!
//! ## Why the CUDA lives in a string
//!
//! The development machine is a Mac: no `nvcc`, no driver, no card. Every
//! compile, run and measurement happens inside a rented job. Compiling the
//! kernels with `nvcc` at build time would make each edit — a tile size, a
//! `maxrregcount`, a padding experiment — cost a 40-70 minute image rebuild.
//! So the kernel source ships as data and NVRTC compiles it at startup, in a
//! few hundred milliseconds. One image build for the whole campaign.
//!
//! The `.cuh` and the `.cu` are **concatenated** rather than `#include`d.
//! `CompileOptions::include_paths` exists and would work; the reason not to
//! use it is proof, not capability. An include resolved from a mounted
//! directory makes the compiled text open-ended — the header can change with
//! the string passed to NVRTC staying the same. Concatenation gives one
//! string, hashable, equal to what the driver saw.
//!
//! ## What is deliberately *not* here
//!
//! The other layouts. `llvq-metal` carries decoders for `Grouped32`,
//! `Flat32`, `Fixed96` and `Sorted32`; K−1(a) measured all of them on the
//! whole model and the answer was unambiguous — `Flat32` saves 0.254 b/weight
//! over `Slot32` and costs 2.3× the time, `Grouped32` saves 2.012 and costs
//! 3.0×. Porting them would be porting the losing half of a settled question.
//! Reclaiming bits happens *inside* `Slot32`, by capping levels.

#[cfg(target_os = "linux")]
pub mod gpu;

/// The CUDA sources, embedded so a run is reproducible from the binary alone.
///
/// `include_str!` and not a runtime read: a kernel loaded from a mounted
/// directory could differ from the one in the commit, and then the sha256 the
/// harness prints would describe a file nobody can retrieve. An override path
/// for iteration is a separate, explicit switch — not the default.
#[cfg(target_os = "linux")]
pub const SLOT_CUH: &str = include_str!("../kernels/llvq_slot.cuh");
#[cfg(target_os = "linux")]
pub const PREFLIGHT_CU: &str = include_str!("../kernels/preflight.cu");
