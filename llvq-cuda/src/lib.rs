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

// Portable on purpose, like `f16_bits` below: the parsing rules of
// `LLVQ_BENCH_ARMS` — unknown names refused, witness required, phases
// monotone — are exactly the kind of logic a mutation can silently break,
// and the development machine has no CUDA, so the tests must run here.
pub mod arms;

/// f32 → binary16 bits, round to nearest even.
///
/// Portable on purpose, unlike everything else in this file: the rotation
/// kernel is checked on the development Mac as well as on a card, so both
/// sides need this and neither can be behind `cfg(linux)`. `llvq_metal` and
/// `bin/matvec.rs` carry their own copies — the shared home is lot K0.
pub fn f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 255 {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e16 = exp - 127 + 15;
    if e16 >= 31 {
        return sign | 0x7c00;
    }
    if e16 <= 0 {
        if e16 < -10 {
            return sign;
        }
        let man = man | 0x80_0000;
        let shift = (14 - e16) as u32;
        let half = 1u32 << (shift - 1);
        let mut v = man >> shift;
        let rem = man & ((1 << shift) - 1);
        if rem > half || (rem == half && v & 1 == 1) {
            v += 1;
        }
        return sign | v as u16;
    }
    let mut v = (sign as u32) | ((e16 as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && v & 1 == 1) {
        v += 1;
    }
    v as u16
}

/// binary16 bits → f64. Exact: every binary16 is a binary64.
pub fn f16_to_f64(h: u16) -> f64 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = (h >> 10) & 0x1f;
    let man = (h & 0x3ff) as f64;
    sign * match exp {
        0 => man * (-24f64).exp2(),
        31 => {
            if man == 0.0 {
                f64::INFINITY
            } else {
                f64::NAN
            }
        }
        e => (1.0 + man / 1024.0) * ((e as f64) - 15.0).exp2(),
    }
}

/// Mirrors `LLVQ_ROT_KMAX` in `kernels/llvq_rot.cuh`.
///
/// Duplicated across the language boundary, so it is pinned by
/// `rotation_kmax_matches_the_kernel` rather than by a comment.
pub const ROT_KMAX: usize = 32;

/// The CUDA sources, embedded so a run is reproducible from the binary alone.
#[cfg(target_os = "linux")]
pub const SLOT_CUH: &str = include_str!("../kernels/llvq_slot.cuh");
#[cfg(target_os = "linux")]
pub const PREFLIGHT_CU: &str = include_str!("../kernels/preflight.cu");
#[cfg(target_os = "linux")]
pub const MATVEC_CU: &str = include_str!("../kernels/matvec.cu");
#[cfg(target_os = "linux")]
pub const FLOOR_CUH: &str = include_str!("../kernels/llvq_floor.cuh");
#[cfg(target_os = "linux")]
pub const ROT_CUH: &str = include_str!("../kernels/llvq_rot.cuh");
#[cfg(target_os = "linux")]
pub const ROTATE_CU: &str = include_str!("../kernels/rotate.cu");

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
    load_sources_named(&["llvq_slot.cuh", "preflight.cu"])
}

/// [`load_sources`] for a binary whose second file is not `preflight.cu`.
#[cfg(target_os = "linux")]
pub fn load_sources_named(names: &[&str; 2]) -> Result<Sources, String> {
    let set = load_sources_many(names)?;
    let mut it = set.parts.into_iter();
    Ok(Sources {
        slot: it.next().expect("two parts"),
        cu: it.next().expect("two parts"),
        overridden_from: set.overridden_from,
    })
}

/// The same loader, for any number of parts.
///
/// NVRTC has no filesystem, so a multi-file kernel is **concatenated** by the
/// host rather than `#include`d — the `#ifndef … #include` guards at the top
/// of each header exist so the very same text also compiles under `clang++`,
/// which does have one. Order therefore matters: a part may only use what an
/// earlier part defined.
#[cfg(target_os = "linux")]
pub struct SourceSet {
    pub parts: Vec<String>,
    /// `None` when the embedded copies were used.
    pub overridden_from: Option<String>,
}

#[cfg(target_os = "linux")]
pub fn load_sources_many(names: &[&str]) -> Result<SourceSet, String> {
    let embedded = |n: &str| match n {
        "llvq_slot.cuh" => Ok(SLOT_CUH),
        "preflight.cu" => Ok(PREFLIGHT_CU),
        "matvec.cu" => Ok(MATVEC_CU),
        "llvq_floor.cuh" => Ok(FLOOR_CUH),
        "llvq_rot.cuh" => Ok(ROT_CUH),
        "rotate.cu" => Ok(ROTATE_CU),
        other => Err(format!("no embedded copy of {other}")),
    };
    match std::env::var("LLVQ_KERNEL_DIR") {
        Err(_) => Ok(SourceSet {
            parts: names
                .iter()
                .map(|n| embedded(n).map(str::to_string))
                .collect::<Result<_, _>>()?,
            overridden_from: None,
        }),
        Ok(dir) => Ok(SourceSet {
            parts: names
                .iter()
                .map(|n| {
                    std::fs::read_to_string(std::path::Path::new(&dir).join(n))
                        .map_err(|e| format!("LLVQ_KERNEL_DIR={dir} : {n} : {e}"))
                })
                .collect::<Result<_, _>>()?,
            overridden_from: Some(dir),
        }),
    }
}
