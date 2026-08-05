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
#[cfg(target_os = "linux")]
pub const SLOT_CUH: &str = include_str!("../kernels/llvq_slot.cuh");
#[cfg(target_os = "linux")]
pub const PREFLIGHT_CU: &str = include_str!("../kernels/preflight.cu");

/// Where the two sources come from, and whether that was the committed copy.
#[cfg(target_os = "linux")]
pub struct Sources {
    pub slot: String,
    pub cu: String,
    /// `None` when the embedded copies were used.
    pub overridden_from: Option<String>,
}

/// Load the kernel sources, honouring `LLVQ_KERNEL_DIR`.
///
/// This switch is what makes the campaign affordable, and it is worth being
/// precise about why. The kernels are `include_str!`'d, so *by default* a run
/// is reproducible from the binary alone — which is the property a published
/// number needs. But it also means a one-line kernel edit would rebuild the
/// image: forty to seventy minutes, on a step with a documented history of
/// dying to SIGKILL. That is not a tuning loop, it is a dare.
///
/// With the variable set, the binary reads the two files from a directory the
/// job wrote a moment earlier — a heredoc in the job command is enough, the
/// sources are a few kilobytes. Editing a tile size or a register cap then
/// costs one mini-job and no rebuild.
///
/// The safeguard is disclosure, not prohibition: when the override is used the
/// harness says so, loudly, and prints the sha256 of the exact string handed to
/// NVRTC. A figure taken from an overridden run is traceable to that hash and
/// to nothing else — which is the honest position, and a stricter one than
/// silently trusting a path.
#[cfg(target_os = "linux")]
pub fn load_sources() -> Result<Sources, String> {
    match std::env::var("LLVQ_KERNEL_DIR") {
        Err(_) => Ok(Sources {
            slot: SLOT_CUH.to_string(),
            cu: PREFLIGHT_CU.to_string(),
            overridden_from: None,
        }),
        Ok(dir) => {
            let read = |n: &str| {
                std::fs::read_to_string(std::path::Path::new(&dir).join(n))
                    .map_err(|e| format!("LLVQ_KERNEL_DIR={dir} : {n} : {e}"))
            };
            Ok(Sources {
                slot: read("llvq_slot.cuh")?,
                cu: read("preflight.cu")?,
                overridden_from: Some(dir),
            })
        }
    }
}
