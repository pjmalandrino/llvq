//! The Linux half of planesbench: the kernel sources, the arm types, the launchers, and `run`.

use cudarc::driver::PushKernelArg;
use llvq_artifact::runtime::{transcode, transcode_planes12x, ClassTable, Layout};
use llvq_core::{SplitMix64, DIM};
use llvq_cuda::arms::{self, ArmSet};
use llvq_cuda::gpu::{Cuda, KernelSource};
use llvq_cuda::{f16_bits, f16_to_f64};
use llvq_search::fastdec::FastDecoder;
use llvq_search::index::N13;
use llvq_search::tetra::Tetra;
use llvq_search::Searcher;
use std::time::Instant;

#[path = "fusion.rs"]
mod fusion;
#[path = "matbuild.rs"]
mod matbuild;
#[path = "measure.rs"]
mod measure;
#[path = "report.rs"]
mod report;
#[path = "setup.rs"]
mod setup;
use matbuild::{build_mat, load_mats};
use measure::measure;
use setup::{compile, read_tetra, select_arms, upload_inputs};

include!("../../planes14_host.rs");
include!("../../golay70_host.rs");
include!("../../seg_host.rs");
include!("../../e1v_host.rs");

const THREADS: u32 = 256;
const GSCALE: [f32; 2] = [0.625, 1.375];
const TABLE_ENTRIES: usize = 512;
const REC_WORDS: usize = 6;
const TOL: f64 = 1e-5;
/// The competitor arm's threshold, looser by a factor of ~100, for a
/// reason of format and not of indulgence: `awq_gemv_g128` writes its
/// output in **binary16** (`f2h`, last line of `awq_gemv.cu`) where the
/// five in-house arms return f32. A binary16 output carries ~2⁻¹¹ of
/// relative error by construction, so demanding 1e-5 here would fail a
/// perfectly correct kernel. Its REFERENCE stays exact: `scale·(q − zero)`
/// is what the kernel reconstructs bit for bit.
const AWQ_TOL: f64 = 1e-3;
const ROUNDS: usize = 7;
const WARMUP: usize = 2;
// Arm order inside a round: `arms::ARM_NAMES`, the registration order —
// Slot32, Planes14, Planes12x, Golay70 v1, FP16, AWQ, Golay70 v2.
//
// ⚠️ The order is the REGISTRATION order, and an added arm ALWAYS goes
// LAST. Inserting an arm in the middle would change the dispatch order of
// the existing arms inside every round, hence the object the published
// table measures. AWQ (2026-08-10) then Golay70 v2 (2026-08-11) both
// follow this rule. An `LLVQ_BENCH_ARMS` selection never changes this
// order: it skips arms, it does not move them.

/// The seven projection shapes of Qwen3-4B, `(name, d_out, d_in)`.
const SHAPES: [(&str, usize, usize); 7] = [
    ("q_proj", 4096, 2560),
    ("k_proj", 1024, 2560),
    ("v_proj", 1024, 2560),
    ("o_proj", 2560, 4096),
    ("gate_proj", 9728, 2560),
    ("up_proj", 9728, 2560),
    ("down_proj", 2560, 9728),
];

/// The two Planes14 sources are not embedded in `lib.rs` — new files
/// only, that file belongs to another lot — so the bin carries its own
/// copies and honours `LLVQ_KERNEL_DIR` with the same contract as
/// `load_sources_many`: without the variable, the embedded text; with it,
/// the files from the directory, disclosed loudly by the caller.
const PLANES_CUH_EMBED: &str = include_str!("../../../kernels/llvq_planes.cuh");
const PLANES_CU_EMBED: &str = include_str!("../../../kernels/planes.cu");
const PLANES12_CUH_EMBED: &str = include_str!("../../../kernels/llvq_planes12.cuh");
const PLANES12_CU_EMBED: &str = include_str!("../../../kernels/planes12.cu");
const GOLAY_CUH_EMBED: &str = include_str!("../../../kernels/llvq_golay.cuh");
const GOLAY_CU_EMBED: &str = include_str!("../../../kernels/golay70.cu");
/// The segmented Planes14 kernel — item A4. One file, no header of its
/// own: it reuses `llvq_planes.cuh`'s `planes_dot`. It lives apart from
/// `planes.cu` on purpose — `llvq-llm/src/fused_cuda.rs` concatenates
/// `llvq_planes.cuh` + `planes.cu` + `tv_planes_h.cu` as the *shipped*
/// inference kernel, and appending to `planes.cu` would change that
/// string's bytes and its sha256 for a bench's convenience.
const PLANES_SEG_CU_EMBED: &str = include_str!("../../../kernels/planes_seg.cu");
/// The competitor arm. It depends on `h2f`/`f2h`, hence on `matvec.cu`,
/// so it is concatenated AFTER it, and last, so that adding a competitor
/// never reorders the fragments of the in-house arms.
const AWQ_CU_EMBED: &str = include_str!("../../../kernels/awq_gemv.cu");
/// The witness arm: the FROZEN copy of the published Golay70 decoder
/// (v1), symbols renamed, buffers shared with v2. Concatenated after AWQ,
/// since the arm added last reorders no existing fragment.
const GOLAY_V1_CU_EMBED: &str = include_str!("../../../kernels/golay70_v1.cu");

fn load_planes_sources() -> Result<(String, String, Option<String>), String> {
    match std::env::var("LLVQ_KERNEL_DIR") {
        Err(_) => Ok((PLANES_CUH_EMBED.to_string(), PLANES_CU_EMBED.to_string(), None)),
        Ok(dir) => {
            let rd = |n: &str| {
                std::fs::read_to_string(std::path::Path::new(&dir).join(n))
                    .map_err(|e| format!("LLVQ_KERNEL_DIR={dir}: {n}: {e}"))
            };
            let cuh = rd("llvq_planes.cuh")?;
            let cu = rd("planes.cu")?;
            Ok((cuh, cu, Some(dir)))
        }
    }
}

/// Same contract for the two Planes12x sources — the M2 arm's kernel.
fn load_planes12_sources() -> Result<(String, String, Option<String>), String> {
    match std::env::var("LLVQ_KERNEL_DIR") {
        Err(_) => Ok((
            PLANES12_CUH_EMBED.to_string(),
            PLANES12_CU_EMBED.to_string(),
            None,
        )),
        Ok(dir) => {
            let rd = |n: &str| {
                std::fs::read_to_string(std::path::Path::new(&dir).join(n))
                    .map_err(|e| format!("LLVQ_KERNEL_DIR={dir}: {n}: {e}"))
            };
            let cuh = rd("llvq_planes12.cuh")?;
            let cu = rd("planes12.cu")?;
            Ok((cuh, cu, Some(dir)))
        }
    }
}

/// Same contract for the two Golay70 sources — the E2 arm's kernel.
fn load_golay_sources() -> Result<(String, String, Option<String>), String> {
    match std::env::var("LLVQ_KERNEL_DIR") {
        Err(_) => Ok((GOLAY_CUH_EMBED.to_string(), GOLAY_CU_EMBED.to_string(), None)),
        Ok(dir) => {
            let rd = |n: &str| {
                std::fs::read_to_string(std::path::Path::new(&dir).join(n))
                    .map_err(|e| format!("LLVQ_KERNEL_DIR={dir}: {n}: {e}"))
            };
            let cuh = rd("llvq_golay.cuh")?;
            let cu = rd("golay70.cu")?;
            Ok((cuh, cu, Some(dir)))
        }
    }
}

/// Same contract for the one segmented source — the A4 arm's kernel.
/// Same contract as the others: without the variable, the embedded text;
/// with it, the file from the directory, announced loudly. It is also the
/// path by which a kernel we cannot vendor, QTIP being GPL-3, would enter
/// without being committed.
fn load_awq_source() -> Result<(String, Option<String>), String> {
    match std::env::var("LLVQ_KERNEL_DIR") {
        Err(_) => Ok((AWQ_CU_EMBED.to_string(), None)),
        Ok(dir) => {
            let cu =
                std::fs::read_to_string(std::path::Path::new(&dir).join("awq_gemv.cu"))
                    .map_err(|e| format!("LLVQ_KERNEL_DIR={dir}: awq_gemv.cu: {e}"))?;
            Ok((cu, Some(dir)))
        }
    }
}

/// Our QTIP shims — committed, so always available, and version-locked to
/// the binary that launches them.
const QTIP_GLUE_EMBED: &str = include_str!("../../../kernels/qtip_glue.cu");
/// The typedefs the fetched kernel needs and NVRTC does not carry. Ours,
/// and concatenated **before** the fetched half.
const QTIP_PRELUDE_EMBED: &str = include_str!("../../../kernels/qtip_prelude.cuh");

/// The QTIP device half, and the one loader here that reads a **different**
/// variable from every other.
///
/// 🚨 `LLVQ_QTIP_DIR`, not `LLVQ_KERNEL_DIR`, and the distinction is not
/// cosmetic. `LLVQ_KERNEL_DIR` means *override every kernel source from
/// this directory*: every other loader in this file and in
/// `load_sources_many` reads ITS OWN files from it and **fails hard** when
/// one is missing. Pointing it at the QTIP fetch output would therefore
/// break the whole bench on `matvec.cu: No such file or directory`. QTIP is
/// not an override of anything — it is an ADDITION — so it gets a variable
/// of its own, and the two compose instead of colliding.
///
/// Only the device half is read from disk: the kernel is GPL v3 and this
/// workspace is MIT OR Apache-2.0, so it is fetched at job time by
/// `ops/fetch-qtip.sh` and never committed. The shims above are ours.
/// `Ok(None)` means "no QTIP here", the normal state on any machine that
/// has not run the script; the arm is then simply not dispatchable, which
/// `arms::HAS_KERNEL` already says. What must never happen is a *silent*
/// fallback to some other text, which is why there is nothing to fall
/// back to.
fn load_qtip_sources() -> Result<Option<(String, String)>, String> {
    let Ok(dir) = std::env::var("LLVQ_QTIP_DIR") else {
        return Ok(None);
    };
    let cuh = std::fs::read_to_string(std::path::Path::new(&dir).join("qtip_device.cuh"))
        .map_err(|e| {
            format!(
                "LLVQ_QTIP_DIR={dir}: qtip_device.cuh: {e}, run ops/fetch-qtip.sh \
                 and point the variable at its output"
            )
        })?;
    // Prelude first — it declares the types the fetched half names.
    Ok(Some((format!("{QTIP_PRELUDE_EMBED}\n{cuh}"), QTIP_GLUE_EMBED.to_string())))
}

fn load_planes_seg_source() -> Result<(String, Option<String>), String> {
    match std::env::var("LLVQ_KERNEL_DIR") {
        Err(_) => Ok((PLANES_SEG_CU_EMBED.to_string(), None)),
        Ok(dir) => {
            let cu = std::fs::read_to_string(
                std::path::Path::new(&dir).join("planes_seg.cu"),
            )
            .map_err(|e| format!("LLVQ_KERNEL_DIR={dir}: planes_seg.cu: {e}"))?;
            Ok((cu, Some(dir)))
        }
    }
}

/// Same contract for the frozen v1 copy. Overriding the WITNESS is
/// allowed, but announced as loudly as the rest: a witness silently
/// replaced attests to nothing.
fn load_golay_v1_source() -> Result<(String, Option<String>), String> {
    match std::env::var("LLVQ_KERNEL_DIR") {
        Err(_) => Ok((GOLAY_V1_CU_EMBED.to_string(), None)),
        Ok(dir) => {
            let cu = std::fs::read_to_string(
                std::path::Path::new(&dir).join("golay70_v1.cu"),
            )
            .map_err(|e| format!("LLVQ_KERNEL_DIR={dir}: golay70_v1.cu: {e}"))?;
            Ok((cu, Some(dir)))
        }
    }
}

/// One matrix's inputs, whatever they came from — same split as
/// `bin/matvec`: everything downstream of `Src` is shared, so the
/// synthetic and artifact runs differ only in their data.
struct Src {
    name: String,
    d_out: usize,
    d_in: usize,
    indices: Vec<u64>,
    gains: Vec<u32>,
    centroids: [f32; 2],
    rscale: Vec<f32>,
    tail: Vec<f32>,
}

/// A buffer whose arm may enter only at a later phase: the bytes are
/// prepared once at build time (the host work, references and transcodes,
/// is not redone between phases), but they reach the card only at the
/// start of the first phase that times the arm. During a phase the
/// resident STREAMS (the GB) are therefore exactly those of that phase's
/// arms, the residency invariant the §4 control of the preregistration
/// must reproduce.
///
/// ⚠️ Exact scope, so the claim does not overreach: four CONSTANT buffers
/// follow the UNION of the phases and not the current phase.
/// `d_gtab`/`d_cw` (~32 KiB, as soon as a golay70 arm is selected
/// anywhere), `d_xh` (32 KiB) and `d_yh` (~19 KiB, as soon as awq is).
/// They do not go through this stage: ~83 KiB in all against ~15 GB of
/// stream, below any resolution of the bench, and declared here rather
/// than denied.
enum Staged<T> {
    Host(Vec<T>),
    Dev(cudarc::driver::CudaSlice<T>),
}

impl<T> Staged<T> {
    /// The device buffer. Named panic if the arm was not uploaded: a
    /// launch on an unstaged arm is a phase-sequencing bug, never a case
    /// to be recovered from in silence.
    fn dev(&self) -> &cudarc::driver::CudaSlice<T> {
        match self {
            Staged::Dev(d) => d,
            Staged::Host(_) => panic!("arm not uploaded, phase bug"),
        }
    }
}

fn stage_up_u32(cuda: &Cuda, s: &mut Staged<u32>) -> Result<(), String> {
    if let Staged::Host(v) = s {
        *s = Staged::Dev(cuda.up_u32(v)?);
    }
    Ok(())
}

fn stage_up_u16(cuda: &Cuda, s: &mut Staged<u16>) -> Result<(), String> {
    if let Staged::Host(v) = s {
        *s = Staged::Dev(cuda.up_u16(v)?);
    }
    Ok(())
}

// ---- a selected arm = one Option built, and nothing else ----
//
// Every arm carries its buffers AND its byte accounting: a discarded arm
// has neither, so no table row can cite an arm that did not run.

struct SlotArm {
    words: Staged<u32>,
    bases: Staged<u32>,
    bytes: u64,
}

struct PlanesArm {
    pwords: Staged<u32>,
    bytes: u64,
}

struct P12Arm {
    words: Staged<u32>,
    exc_idx: Staged<u32>,
    exc_words: Staged<u32>,
    n_exc: usize,
    bytes: u64,
}

/// SHARED by the golay70v1 and golay70v2 arms: not one format byte
/// distinguishes them, so a single set of buffers serves both, and the v1
/// witness arm costs no VRAM.
struct G70Arm {
    words: Staged<u32>,
    exc_idx: Staged<u32>,
    exc_words: Staged<u32>,
    n_exc: usize,
    bytes: u64,
}

/// P1c. The row-aligned E1v stream, its bases table, and the number of
/// blocks per ROW it was cut on, the same as the kernel's `nblocks`,
/// carried here rather than recomputed at launch.
struct E1vArm {
    data: Staged<u32>,
    bases: Staged<u32>,
    row_blocks: usize,
    bytes: u64,
}

// ---- the competitor arm (AWQ w4g128) ----
//
// Three buffers and its own reference: its content is no other arm's, so
// neither `y_ref` (the LLVQ content) nor `y16_ref` (the same weights in
// binary16) describes it. Same situation as the FP16 arm, same answer.
struct AwqArm {
    w: Staged<u32>,
    z: Staged<u32>,
    s: Staged<u16>,
    bytes: u64,
    y_ref: Vec<f64>,
    /// The arm's OWN error denominator: `Σ|w·x|` over its own weights.
    /// Reusing `scale`, computed on the LLVQ weights, would measure AWQ's
    /// error by another format's rule.
    scale: Vec<f64>,
}

/// The 2-bit competitor. Its payload is **synthetic and is not a
/// quantization of our weights** — a stronger caveat than the AWQ arm's,
/// and it is a property of the code, not a shortcut. QTIP is a fixed-rate
/// trellis: every bit pattern is a valid codeword, so a pseudo-random
/// buffer is a legitimate input, but encoding *given* weights would need
/// their Viterbi search. This arm therefore makes **no quality claim at
/// all**; it measures time and nothing else.
///
/// The timing stands regardless: the kernel has no data-dependent branch
/// and its traffic is a function of the shapes alone.
struct QtipArm {
    /// The trellis stream, two u16 per u32 in the order the kernel reads
    /// them (it casts the pointer to `const uint16_t*`).
    compressed: Staged<u32>,
    /// 512 half2 entries, flat as 1024 binary16 words.
    codebook: Staged<u16>,
    bytes: u64,
    /// Exact by construction: we know the bits, so we know the state, so we
    /// know the codebook entry and the sign.
    y_ref: Vec<f64>,
    /// `Σ|w·x|` over **its own** decoded weights — reusing the LLVQ scale
    /// would judge this arm by another format's rule.
    scale: Vec<f64>,
    /// The shim this shape resolves to, e.g. `qtip_mv_4096x2560`.
    kernel: String,
}

/// The served Tetra arm: the second file's stream and its own everything.
///
/// It carries its own `gscale`, `rscale` and `tail` — a `Mat` already has
/// all three, and they belong to the FIRST file. Same projection, same
/// shape, different quantization: reusing the first file's would decode
/// this stream against another file's scales, which is a silent wrong
/// number and not a crash. Its reference and its error denominator are its
/// own for the same reason the AWQ and QTIP arms' are.
struct TetraArm {
    /// Little-endian, **row-strided** words. Not flat: a 6-byte record at
    /// `6·b` is not 4-aligned, so a row must start on a u32 boundary or
    /// every row after the first reads at a shifted phase.
    words: Staged<u32>,
    /// The stream's OWN stride, never recomputed at launch — the E1v
    /// argument, and it is asserted against `tetra48::stride_u32` below.
    stride_u32: u32,
    gscale: cudarc::driver::CudaSlice<f32>,
    rscale: cudarc::driver::CudaSlice<f32>,
    tail: cudarc::driver::CudaSlice<f32>,
    bytes: u64,
    /// The row padding, which `bytes` deliberately does NOT include.
    ///
    /// It is real VRAM — the stream allocated and uploaded IS the padded
    /// one — and it is not billed, exactly as the slot arm's 20-byte pad
    /// and the planes arm's 4-byte pad are not. Carried so the run can say
    /// how much VRAM it holds without billing it into a rate, which is the
    /// difference between the two accountings this repository refuses to
    /// mix.
    pad_bytes: u64,
    /// Weights this arm covers. NOT the matrix's: `v_proj` is int4 in the
    /// served file and has no lattice arm, so the Tetra arm's b/weight
    /// denominator is its own and every arm's column is now formed from
    /// the weights that arm actually read.
    weights: u64,
    y_ref: Vec<f64>,
    scale: Vec<f64>,
}

/// One matrix of the **second** file, already in the served stream.
///
/// The indices and gains are dropped as soon as they are transcoded: the
/// reference below decodes the packed words back through
/// `Tetra48Blocks::decode_block`, which is the stronger check anyway — a
/// mistake in the packing is then a disagreement between two routes and
/// not an error both sides share. It also halves what this list costs:
/// `srcs` already holds ~1.8 GB of host RAM for the first file.
struct TetraSrc {
    d_out: usize,
    d_in: usize,
    blocks: llvq_artifact::tetra48::Tetra48Blocks,
    /// The **second** file's centroid pair, not the first's. Same
    /// projection, different quantization, different gain scale.
    centroids: [f32; 2],
    rscale: Vec<f32>,
    tail: Vec<f32>,
}

struct Mat {
    name: String,
    d_out: usize,
    d_in: usize,
    nblocks: usize,
    tail_w: usize,
    slot: Option<SlotArm>,
    planes: Option<PlanesArm>,
    p12: Option<P12Arm>,
    g70: Option<G70Arm>,
    e1v: Option<E1vArm>,
    tetra: Option<TetraArm>,
    awq: Option<AwqArm>,
    qtip: Option<QtipArm>,
    // The FP16 witness and the shared inputs, always built: fp16 cannot
    // be deselected (arms::parse_phases refuses it).
    gscale: cudarc::driver::CudaSlice<f32>,
    rscale: cudarc::driver::CudaSlice<f32>,
    tail: cudarc::driver::CudaSlice<f32>,
    w16: cudarc::driver::CudaSlice<u16>,
    f16_bytes: u64,
    y_ref: Vec<f64>,
    y16_ref: Vec<f64>,
    /// The cublasf16 arm's reference: f16 weights × binary16 input, built
    /// only when the arm is.
    y16h_ref: Option<Vec<f64>>,
    scale: Vec<f64>,
}

/// The bytes an arm reads for one matrix, in the bench's accounting. 0
/// for an arm that was not built, which has no table row to print it in
/// anyway. v1 and v2 share the golay70 row: same buffers, same bytes, by
/// construction.
/// `1/sqrt(16 m)`, entry 0 the origin — reconstructed without a branch and
/// without a division. `m ≤ 27` on this codebook, derived over the whole
/// table by `llvq-bench/examples/tetrashell.rs`; 32 is that bound rounded
/// up, and `llvq_tetra48.cuh` masks with it. Same table `bin/f1rankfloor`
/// builds, same three lines, because it is the same kernel.
fn invnorm_table() -> Vec<f32> {
    let mut t = vec![0.0f32; llvq_artifact::tetra48::TETRA48_SHELLS];
    for (m, e) in t.iter_mut().enumerate().skip(1) {
        *e = (1.0f64 / ((16 * m) as f64).sqrt()) as f32;
    }
    t
}

/// Bytes into little-endian u32, zero-padded — the rank table's prefix and
/// suffix streams are `u8` and the upload path is `u32`.
fn pack_u8(b: &[u8]) -> Vec<u32> {
    b.chunks(4)
        .map(|c| {
            let mut w = [0u8; 4];
            w[..c.len()].copy_from_slice(c);
            u32::from_le_bytes(w)
        })
        .collect()
}

/// The four rank-table buffers the F1 decoder walks, plus the two
/// constants the served Tetra decode adds. Uploaded once for the run.
/// A device pointer as the persistent sections take it.
fn dptr<T>(s: &cudarc::driver::CudaSlice<T>, cuda: &Cuda) -> u64 {
    use cudarc::driver::DevicePtr;
    s.device_ptr(cuda.stream()).0
}

struct TetraTabs {
    rows: cudarc::driver::CudaSlice<u32>,
    prefixes: cudarc::driver::CudaSlice<u32>,
    branches: cudarc::driver::CudaSlice<u16>,
    suffixes: cudarc::driver::CudaSlice<u32>,
    invnorm: cudarc::driver::CudaSlice<f32>,
}

/// The weights an arm's bytes cover, for one matrix.
///
/// 🕳️ It was `n_weights`, one accumulator for the whole table, and that
/// was correct only while every arm read every matrix. The served Tetra
/// arm does not: 36 of the 252 projections are `v_proj`, stored as int4
/// g128 and read by another kernel. Dividing its bytes by all 252
/// projections' weights understates its b/weight by the `v_proj` share —
/// 94,371,840 of 3,633,315,840 weights, **2.597 %** (*computed*, from the
/// seven shapes over 36 layers), which turns a true 2.1576 into a printed
/// 2.1017. Both clear an F1d gate of 2.20, so it would not have flipped
/// the verdict; it would have published a wrong number, which is worse for
/// being survivable.
///
/// Every arm now divides by what it actually read.
fn arm_weights(m: &Mat, a: usize) -> u64 {
    match a {
        arms::TETRA48 => m.tetra.as_ref().map_or(0, |x| x.weights),
        _ => (m.d_out * m.d_in) as u64,
    }
}

fn arm_bytes(m: &Mat, a: usize) -> u64 {
    match a {
        arms::SLOT32 => m.slot.as_ref().map_or(0, |x| x.bytes),
        arms::PLANES14 => m.planes.as_ref().map_or(0, |x| x.bytes),
        arms::PLANES12X => m.p12.as_ref().map_or(0, |x| x.bytes),
        arms::GOLAY70V1 | arms::GOLAY70V2 => m.g70.as_ref().map_or(0, |x| x.bytes),
        arms::E1V => m.e1v.as_ref().map_or(0, |x| x.bytes),
        arms::TETRA48 => m.tetra.as_ref().map_or(0, |x| x.bytes),
        // The floor reads NO weight, which is its definition. What is
        // left is what every LLVQ arm uploads anyway: the f32 tail and
        // the row scales. Its table row will therefore show a near-zero
        // b/weight, and that is not an anomaly, it is the point.
        arms::NULLK => ((m.d_out * m.tail_w) as u64 + m.d_out as u64) * 4,
        arms::FP16 => m.f16_bytes,
        // Same bytes as the witness: cublasf16 reads the SAME w16.
        arms::CUBLASF16 => m.f16_bytes,
        arms::AWQ => m.awq.as_ref().map_or(0, |x| x.bytes),
        arms::QTIP => m.qtip.as_ref().map_or(0, |x| x.bytes),
        _ => unreachable!("unknown arm"),
    }
}

use llvq_cuda::stats::worst_error;

use llvq_cuda::stats::spread;

/// Same shape as `gpu.rs`'s private `row_grid`, replicated rather than
/// exported: one warp per row, whole blocks only — the kernels have no
/// bounds guard (a return before `__syncthreads()` deadlocks).
fn row_grid(d_out: u32, threads: u32, shared: u32) -> cudarc::driver::LaunchConfig {
    assert_eq!(d_out % (threads / 32), 0, "rows must fill whole blocks");
    cudarc::driver::LaunchConfig {
        grid_dim: (d_out * 32 / threads, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: shared,
    }
}

/// `tv_planes(words, tab, gscale, rscale, tail, x, y, nblocks, tail_w)` —
/// `tv_slot` minus the bases array, which Planes14 does not have.
#[allow(clippy::too_many_arguments)]
fn launch_planes(
    cuda: &Cuda,
    f: &cudarc::driver::CudaFunction,
    words: &cudarc::driver::CudaSlice<u32>,
    tab: &cudarc::driver::CudaSlice<u32>,
    gscale: &cudarc::driver::CudaSlice<f32>,
    rscale: &cudarc::driver::CudaSlice<f32>,
    tail: &cudarc::driver::CudaSlice<f32>,
    x: &cudarc::driver::CudaSlice<f32>,
    y: &mut cudarc::driver::CudaSlice<f32>,
    nblocks: u32,
    tail_w: u32,
    d_out: u32,
    threads: u32,
    shared: u32,
) -> Result<(), String> {
    let cfg = row_grid(d_out, threads, shared);
    let mut b = cuda.stream().launch_builder(f);
    b.arg(words).arg(tab).arg(gscale).arg(rscale).arg(tail).arg(x).arg(y)
        .arg(&nblocks).arg(&tail_w);
    unsafe { b.launch(cfg) }.map_err(|e| format!("tv_planes: {e}"))?;
    Ok(())
}

/// `tv_f1r_v3g(words, row_stride_u32, rows, prefixes, branches, suffixes,
/// gscale, invnorm, rscale, tail, x, y, nblocks, tail_w)` — the served
/// Tetra decode.
///
/// Two things differ from every ball arm above, and both are the layout.
/// The words are **row-strided**, so the kernel takes a stride where the
/// others take nothing: a 6-byte record at `6·b` is not 4-aligned, and a
/// row that did not start on a u32 boundary would read every block after
/// the first at a shifted phase. And the four table pointers are the F1
/// rank decoder's, not a `ClassRec` table: Tetra resolves a magnitude
/// through a trellis rank, where the ball layouts index a class.
#[allow(clippy::too_many_arguments)]
fn launch_tetra48(
    cuda: &Cuda,
    f: &cudarc::driver::CudaFunction,
    words: &cudarc::driver::CudaSlice<u32>,
    row_stride_u32: u32,
    tabs: &TetraTabs,
    gscale: &cudarc::driver::CudaSlice<f32>,
    rscale: &cudarc::driver::CudaSlice<f32>,
    tail: &cudarc::driver::CudaSlice<f32>,
    x: &cudarc::driver::CudaSlice<f32>,
    y: &mut cudarc::driver::CudaSlice<f32>,
    nblocks: u32,
    tail_w: u32,
    d_out: u32,
    threads: u32,
    shared: u32,
) -> Result<(), String> {
    let cfg = row_grid(d_out, threads, shared);
    let mut b = cuda.stream().launch_builder(f);
    b.arg(words).arg(&row_stride_u32)
        .arg(&tabs.rows).arg(&tabs.prefixes).arg(&tabs.branches).arg(&tabs.suffixes)
        .arg(gscale).arg(&tabs.invnorm).arg(rscale).arg(tail)
        .arg(x).arg(y).arg(&nblocks).arg(&tail_w);
    unsafe { b.launch(cfg) }.map_err(|e| format!("tv_f1r_v3g: {e}"))?;
    Ok(())
}

/// `tv_tetra48_seg` — `launch_tetra48` with one pointer more.
///
/// The only difference is `gs_off`, inserted right after `gscale`, because
/// a row-concatenation of Tetra matrices carries one gain pair per segment
/// and the kernel resolves which pair a row owns. Everything else is
/// identical, deliberately: the words concatenate without re-encoding —
/// segments sharing `d_in` share `nblocks` and therefore `row_stride_u32` —
/// and `invnorm` is a property of the lattice rather than of a projection,
/// so both are passed exactly as the unfused arm passes them.
#[allow(clippy::too_many_arguments)]
fn launch_tetra48_seg(
    cuda: &Cuda,
    f: &cudarc::driver::CudaFunction,
    words: &cudarc::driver::CudaSlice<u32>,
    row_stride_u32: u32,
    tabs: &TetraTabs,
    gscale: &cudarc::driver::CudaSlice<f32>,
    gs_off: &cudarc::driver::CudaSlice<u32>,
    rscale: &cudarc::driver::CudaSlice<f32>,
    tail: &cudarc::driver::CudaSlice<f32>,
    x: &cudarc::driver::CudaSlice<f32>,
    y: &mut cudarc::driver::CudaSlice<f32>,
    nblocks: u32,
    tail_w: u32,
    d_out: u32,
    threads: u32,
    shared: u32,
) -> Result<(), String> {
    let cfg = row_grid(d_out, threads, shared);
    let mut b = cuda.stream().launch_builder(f);
    b.arg(words).arg(&row_stride_u32)
        .arg(&tabs.rows).arg(&tabs.prefixes).arg(&tabs.branches).arg(&tabs.suffixes)
        .arg(gscale).arg(gs_off).arg(&tabs.invnorm).arg(rscale).arg(tail)
        .arg(x).arg(y).arg(&nblocks).arg(&tail_w);
    unsafe { b.launch(cfg) }.map_err(|e| format!("tv_tetra48_seg: {e}"))?;
    Ok(())
}

/// `tv_planes_seg(words, tab, gscale, gs_off, rscale, tail, x, y, nblocks,
/// tail_w)` — `tv_planes` over a row-concatenation of projections sharing
/// an input, with one extra table naming each row's centroid pair.
///
/// Same grid function as the unfused arm, on the *summed* `d_out`: a
/// segmented matrix is one matrix, and the only thing the launch knows
/// about the segments is `gs_off`. No memset, no atomic — rows partition
/// the output, and `kernels/planes_seg.cu` argues that at length.
#[allow(clippy::too_many_arguments)]
fn launch_planes_seg(
    cuda: &Cuda,
    f: &cudarc::driver::CudaFunction,
    words: &cudarc::driver::CudaSlice<u32>,
    tab: &cudarc::driver::CudaSlice<u32>,
    gscale: &cudarc::driver::CudaSlice<f32>,
    gs_off: &cudarc::driver::CudaSlice<u32>,
    rscale: &cudarc::driver::CudaSlice<f32>,
    tail: &cudarc::driver::CudaSlice<f32>,
    x: &cudarc::driver::CudaSlice<f32>,
    y: &mut cudarc::driver::CudaSlice<f32>,
    nblocks: u32,
    tail_w: u32,
    d_out: u32,
    threads: u32,
    shared: u32,
) -> Result<(), String> {
    let cfg = row_grid(d_out, threads, shared);
    let mut b = cuda.stream().launch_builder(f);
    b.arg(words).arg(tab).arg(gscale).arg(gs_off).arg(rscale).arg(tail).arg(x).arg(y)
        .arg(&nblocks).arg(&tail_w);
    unsafe { b.launch(cfg) }.map_err(|e| format!("tv_planes_seg: {e}"))?;
    Ok(())
}

/// `tv_planes12x(words, exc_idx, exc_words, tab, gscale, rscale, tail, x,
/// y, nblocks, tail_w, row_cta, n_exc)` — the overlay arm: `row_cta` CTAs
/// of rows plus `ceil(n_exc/8)` CTAs of corrections, ONE launch. Both
/// regions accumulate into `y` atomically, so `y` is zeroed first — an
/// async memset on the same stream, deliberately inside the timed arm:
/// it is part of what the layout costs.
#[allow(clippy::too_many_arguments)]
fn launch_planes12x(
    cuda: &Cuda,
    f: &cudarc::driver::CudaFunction,
    words: &cudarc::driver::CudaSlice<u32>,
    exc_idx: &cudarc::driver::CudaSlice<u32>,
    exc_words: &cudarc::driver::CudaSlice<u32>,
    tab: &cudarc::driver::CudaSlice<u32>,
    gscale: &cudarc::driver::CudaSlice<f32>,
    rscale: &cudarc::driver::CudaSlice<f32>,
    tail: &cudarc::driver::CudaSlice<f32>,
    x: &cudarc::driver::CudaSlice<f32>,
    y: &mut cudarc::driver::CudaSlice<f32>,
    nblocks: u32,
    tail_w: u32,
    n_exc: u32,
    d_out: u32,
    threads: u32,
    shared: u32,
) -> Result<(), String> {
    assert_eq!(d_out % (threads / 32), 0, "rows must fill whole blocks");
    let row_cta = d_out * 32 / threads;
    let exc_cta = n_exc.div_ceil(threads / 32);
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (row_cta + exc_cta, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: shared,
    };
    cuda.stream()
        .memset_zeros(y)
        .map_err(|e| format!("memset y: {e}"))?;
    let mut b = cuda.stream().launch_builder(f);
    b.arg(words).arg(exc_idx).arg(exc_words).arg(tab).arg(gscale).arg(rscale)
        .arg(tail).arg(x).arg(y).arg(&nblocks).arg(&tail_w).arg(&row_cta).arg(&n_exc);
    unsafe { b.launch(cfg) }.map_err(|e| format!("tv_planes12x: {e}"))?;
    Ok(())
}

/// `tv_nullk(rscale, tail, x, y, nblocks, tail_w)`: the floor. Same grid,
/// same tiling, same epilogue, no weight buffer, so it has neither a
/// `Staged` nor an arm struct, only a launch.
#[allow(clippy::too_many_arguments)]
fn launch_nullk(
    cuda: &Cuda,
    f: &cudarc::driver::CudaFunction,
    rscale: &cudarc::driver::CudaSlice<f32>,
    tail: &cudarc::driver::CudaSlice<f32>,
    x: &cudarc::driver::CudaSlice<f32>,
    y: &mut cudarc::driver::CudaSlice<f32>,
    nblocks: u32,
    tail_w: u32,
    d_out: u32,
    threads: u32,
    shared: u32,
) -> Result<(), String> {
    let cfg = row_grid(d_out, threads, shared);
    let mut b = cuda.stream().launch_builder(f);
    b.arg(rscale).arg(tail).arg(x).arg(y).arg(&nblocks).arg(&tail_w);
    unsafe { b.launch(cfg) }.map_err(|e| format!("tv_nullk: {e}"))?;
    Ok(())
}

/// `tv_e1v(data, bases, tab, pay, binom, golay, gscale, rscale, tail, x, y,
/// nblocks, tail_w)`: the P1c arm. Same grid and same epilogue as
/// `tv_planes`, a single launch, no correction region and no zeroed `y`.
/// E1v has no exceptions, every row writes its own.
///
/// ⚠️ `nblocks` is the number of blocks per ROW, and it is the value the
/// stream was cut on. It comes from `E1vMat::row_blocks` and not from a
/// second computation: two values would move every group boundary and the
/// decode would go wrong from the second row on.
#[allow(clippy::too_many_arguments)]
fn launch_e1v(
    cuda: &Cuda,
    f: &cudarc::driver::CudaFunction,
    data: &cudarc::driver::CudaSlice<u32>,
    bases: &cudarc::driver::CudaSlice<u32>,
    tab: &cudarc::driver::CudaSlice<u32>,
    pay: &cudarc::driver::CudaSlice<u32>,
    binom: &cudarc::driver::CudaSlice<u32>,
    golay: &cudarc::driver::CudaSlice<u32>,
    gscale: &cudarc::driver::CudaSlice<f32>,
    rscale: &cudarc::driver::CudaSlice<f32>,
    tail: &cudarc::driver::CudaSlice<f32>,
    x: &cudarc::driver::CudaSlice<f32>,
    y: &mut cudarc::driver::CudaSlice<f32>,
    nblocks: u32,
    tail_w: u32,
    d_out: u32,
    threads: u32,
    shared: u32,
) -> Result<(), String> {
    let cfg = row_grid(d_out, threads, shared);
    let mut b = cuda.stream().launch_builder(f);
    b.arg(data).arg(bases).arg(tab).arg(pay).arg(binom).arg(golay).arg(gscale)
        .arg(rscale).arg(tail).arg(x).arg(y).arg(&nblocks).arg(&tail_w);
    unsafe { b.launch(cfg) }.map_err(|e| format!("tv_e1v: {e}"))?;
    Ok(())
}

/// `tv_golay70(words, exc_idx, exc_words, cwtab, gtab, tab, gscale,
/// rscale, tail, x, y, nblocks, tail_w, row_cta, n_exc)` — the E2 arm:
/// same two-region grid and same zeroed-y protocol as `tv_planes12x`,
/// with the codeword table and the dedicated class table added, and an
/// exact-only correction (the main stream holds the origin at every
/// exception). The memset is deliberately inside the timed arm.
#[allow(clippy::too_many_arguments)]
fn launch_golay70(
    cuda: &Cuda,
    f: &cudarc::driver::CudaFunction,
    words: &cudarc::driver::CudaSlice<u32>,
    exc_idx: &cudarc::driver::CudaSlice<u32>,
    exc_words: &cudarc::driver::CudaSlice<u32>,
    cwtab: &cudarc::driver::CudaSlice<u32>,
    gtab: &cudarc::driver::CudaSlice<u32>,
    tab: &cudarc::driver::CudaSlice<u32>,
    gscale: &cudarc::driver::CudaSlice<f32>,
    rscale: &cudarc::driver::CudaSlice<f32>,
    tail: &cudarc::driver::CudaSlice<f32>,
    x: &cudarc::driver::CudaSlice<f32>,
    y: &mut cudarc::driver::CudaSlice<f32>,
    nblocks: u32,
    tail_w: u32,
    n_exc: u32,
    d_out: u32,
    threads: u32,
    shared: u32,
) -> Result<(), String> {
    assert_eq!(d_out % (threads / 32), 0, "rows must fill whole blocks");
    let row_cta = d_out * 32 / threads;
    let exc_cta = n_exc.div_ceil(threads / 32);
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (row_cta + exc_cta, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: shared,
    };
    cuda.stream()
        .memset_zeros(y)
        .map_err(|e| format!("memset y: {e}"))?;
    let mut b = cuda.stream().launch_builder(f);
    b.arg(words).arg(exc_idx).arg(exc_words).arg(cwtab).arg(gtab).arg(tab)
        .arg(gscale).arg(rscale).arg(tail).arg(x).arg(y)
        .arg(&nblocks).arg(&tail_w).arg(&row_cta).arg(&n_exc);
    unsafe { b.launch(cfg) }.map_err(|e| format!("tv_golay70: {e}"))?;
    Ok(())
}

// ================= the competitor arm: AWQ 4-bit, w4g128 =================
//
// A group of 128 input channels, one binary16 scale factor and one 4-bit
// zero per group, four bits per weight. This is the configuration of the
// official Qwen3 AWQ checkpoints, hence the only one this arm carries.
const AWQ_GROUP: usize = 128;

/// The three row strides `awq_gemv_g128` computes for itself. Copied here
/// because the host must allocate and fill exactly what the kernel will
/// read, and two of them are **padded**, which is neither cosmetic nor
/// guessable from the shapes.
fn awq_strides(d_in: usize) -> (usize, usize, usize) {
    let g = d_in / AWQ_GROUP; // actual groups
    let npg = g.div_ceil(8); // `num_groups_packed`
    (d_in / 8, npg, npg * 8) // weight_w, zeros_w, sf_w
}

/// Bytes an AWQ arm reads for one matrix, **in the bench's accounting**:
/// the stream and nothing else, structural padding included because the
/// kernel really does index it.
///
/// No tail and no row scale here, unlike the LLVQ arms: w4g128 quantizes
/// *every* column, it has no tail policy. That is a real difference
/// between the formats, not a billing oversight, and it favours AWQ, so
/// it is declared.
fn awq_bytes(d_out: usize, d_in: usize) -> u64 {
    let (ww, zw, sw) = awq_strides(d_in);
    (d_out * ww * 4 + d_out * zw * 4 + d_out * sw * 2) as u64
}

/// Quantizes one row of weights into w4g128, packs it into the kernel's
/// three buffers, and returns the EXACT f64 dot product of what the kernel
/// will decode against the activation **as it sees it**.
///
/// ## Why the reference is computed here and nowhere else
///
/// The AWQ arm decodes a content that is no other arm's: neither `y_ref`
/// (the LLVQ content) nor `y16_ref` (the same weights in binary16)
/// describes it. It needs its own, on the FP16 arm's model, and that one
/// is **exact by construction**: we know `q`, `scale` and `zero`, so
/// `scale·(q − zero)` is the weight the kernel will reconstruct, bit for
/// bit, with no approximation to bound.
///
/// ⚠️ `xf` is the activation **rounded to binary16 then widened back**,
/// because that is what the kernel reads: its inputs are `float4` of eight
/// binary16. Using the f32 activation here would inflate the measured
/// error by a deviation that is not the arm's.
fn awq_quant_row(
    wrow: &[f64],
    xf: &[f64],
    d_in: usize,
    w_out: &mut [u32],
    z_out: &mut [u32],
    s_out: &mut [u16],
) -> f64 {
    let (ww, zw, sw) = awq_strides(d_in);
    let g = d_in / AWQ_GROUP;
    w_out[..ww].fill(0);
    z_out[..zw].fill(0);
    s_out[..sw].fill(0);
    let mut acc = 0.0f64;
    for gi in 0..g {
        let c0 = gi * AWQ_GROUP;
        let seg = &wrow[c0..c0 + AWQ_GROUP];
        let (lo, hi) = seg.iter().fold((f64::MAX, f64::MIN), |(a, b), &v| (a.min(v), b.max(v)));
        // Asymmetric 4-bit scale and zero, the AWQ convention:
        // `w ≈ scale·(q − zero)` with `q ∈ [0, 15]`. A zero scale happens
        // only on a constant group; we force it non-zero so the division
        // exists, and the zero then absorbs the value.
        let mut scale = (hi - lo) / 15.0;
        // `!(scale > 0.0)` and not `scale <= 0.0`: the negated form also
        // catches a NaN (a NaN weight upstream), which the positive form
        // would let through into the division.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if !(scale > 0.0) {
            scale = 1.0;
        }
        // The scale makes a binary16 round trip BEFORE serving as the
        // reference: that is the one the kernel will read, not the f64.
        let scale = f16_to_f64(f16_bits(scale as f32));
        let zero = (-lo / scale).round().clamp(0.0, 15.0);
        s_out[gi] = f16_bits(scale as f32);
        let zq = zero as u32;
        z_out[gi / 8] |= (zq & 0xF) << (4 * (gi % 8));
        for (k, &w) in seg.iter().enumerate() {
            let q = (w / scale + zero).round().clamp(0.0, 15.0);
            let c = c0 + k;
            w_out[c / 8] |= ((q as u32) & 0xF) << (4 * (c % 8));
            acc += scale * (q - zero) * xf[c];
        }
    }
    acc
}

/// `qtip_mv_<M>x<K>(out, compressed, x, codebook)` — the shim from
/// `kernels/qtip_glue.cu`.
///
/// The geometry is **theirs**, copied from `decompress_matvec_ptr`:
/// `<<<128, 1024, 1 << (S + 5 + V + 1)>>>`. Changing it would make this arm
/// a measurement of our tuning rather than of their kernel — the same rule
/// the AWQ arm follows.
///
/// ⚠️ The 64 KiB codebook is **dynamic** shared memory, above the 48 KiB
/// per-block default, so `f` must have come from
/// `Cuda::func_dynamic_shared(name, QTIP_SHARED_BYTES)` and not from
/// `Cuda::func`. A function obtained the wrong way compiles and fails at
/// launch.
fn launch_qtip(
    cuda: &Cuda,
    f: &cudarc::driver::CudaFunction,
    y: &mut cudarc::driver::CudaSlice<f32>,
    compressed: &cudarc::driver::CudaSlice<u32>,
    xh: &cudarc::driver::CudaSlice<u16>,
    codebook: &cudarc::driver::CudaSlice<u16>,
) -> Result<(), String> {
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (128, 1, 1),
        block_dim: (1024, 1, 1),
        shared_mem_bytes: llvq_cuda::qtip_host::QTIP_SHARED_BYTES,
    };
    let mut b = cuda.stream().launch_builder(f);
    b.arg(y).arg(compressed).arg(xh).arg(codebook);
    unsafe { b.launch(cfg) }.map_err(|e| format!("qtip_mv: {e}"))?;
    Ok(())
}

/// The trellis stream as the kernel addresses it: it casts `compressed` to
/// `const uint16_t*`, so u16 index `i` must sit at byte `2i`. On a
/// little-endian device that is the low half of u32 word `i / 2`.
///
/// A tile is 32 u16, so the length is always even and the last word is
/// never half-filled — asserted rather than assumed.
fn qtip_pack_u32(buf: &[u16]) -> Vec<u32> {
    assert!(buf.len().is_multiple_of(2), "QTIP buffer length must be even");
    buf.chunks_exact(2).map(|c| c[0] as u32 | ((c[1] as u32) << 16)).collect()
}

/// `awq_gemv_g128(inputs, weight, zeros, scaling_factors, outputs, IC, OC)`.
///
/// The geometry is **theirs**, copied from `gemv_forward_cuda`:
/// `dim3 num_blocks(1, OC/4, B)` and `dim3 num_threads(32, 4)`, one warp
/// per output channel and four channels per block. Changing it would make
/// this arm a measurement of our tuning, not of their kernel.
#[allow(clippy::too_many_arguments)]
fn launch_awq(
    cuda: &Cuda,
    f: &cudarc::driver::CudaFunction,
    xh: &cudarc::driver::CudaSlice<u16>,
    w: &cudarc::driver::CudaSlice<u32>,
    z: &cudarc::driver::CudaSlice<u32>,
    s: &cudarc::driver::CudaSlice<u16>,
    y: &mut cudarc::driver::CudaSlice<u16>,
    d_in: u32,
    d_out: u32,
) -> Result<(), String> {
    assert_eq!(d_out % 4, 0, "AWQ: OC/4 blocks, OC must be a multiple of 4");
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (1, d_out / 4, 1),
        block_dim: (32, 4, 1),
        shared_mem_bytes: 0,
    };
    let mut b = cuda.stream().launch_builder(f);
    b.arg(xh).arg(w).arg(z).arg(s).arg(y).arg(&d_in).arg(&d_out);
    unsafe { b.launch(cfg) }.map_err(|e| format!("awq_gemv_g128: {e}"))?;
    Ok(())
}

/// What `upload_inputs` builds before the matrices: the decoder tables, the activation,
/// their device copies, and the draw the synthetic path continues.
struct Inputs {
    reps: usize,
    fd: FastDecoder,
    table: ClassTable,
    d_tab: cudarc::driver::CudaSlice<u32>,
    tetra_tabs: Option<TetraTabs>,
    golay: llvq_core::Golay,
    g70cls: Vec<Golay70Class>,
    d_gtab: Option<cudarc::driver::CudaSlice<u32>>,
    d_cw: Option<cudarc::driver::CudaSlice<u32>>,
    d_e1vtab: Option<cudarc::driver::CudaSlice<u32>>,
    d_e1vpay: Option<cudarc::driver::CudaSlice<u32>>,
    d_e1vbinom: Option<cudarc::driver::CudaSlice<u32>>,
    d_e1vcw: Option<cudarc::driver::CudaSlice<u32>>,
    rng: SplitMix64,
    x: Vec<f32>,
    d_x: cudarc::driver::CudaSlice<f32>,
    d_xh: Option<cudarc::driver::CudaSlice<u16>>,
    searcher: Searcher,
}

/// One arm's launch over one matrix, the shape of `measure`'s closures, named so the
/// sections they are handed to need not spell the `Fn` type out.
trait RunMat<Y>: Fn(&Mat, &mut cudarc::driver::CudaSlice<Y>) -> Result<(), String> {}
impl<Y, F: Fn(&Mat, &mut cudarc::driver::CudaSlice<Y>) -> Result<(), String>> RunMat<Y> for F {}

/// The Golay70 launch, which takes the kernel too: v1 and v2 share every buffer.
trait RunG70: Fn(&Mat, &cudarc::driver::CudaFunction, &mut cudarc::driver::CudaSlice<f32>) -> Result<(), String> {}
impl<F: Fn(&Mat, &cudarc::driver::CudaFunction, &mut cudarc::driver::CudaSlice<f32>) -> Result<(), String>> RunG70 for F {}

/// One fused group's launch: `()` on the ball layouts, whether it ran on Tetra.
trait RunSeg<R>: Fn(&FusedMat, &mut cudarc::driver::CudaSlice<f32>) -> Result<R, String> {}
impl<R, F: Fn(&FusedMat, &mut cudarc::driver::CudaSlice<f32>) -> Result<R, String>> RunSeg<R> for F {}

// ---- the fusion arm (item A4) ----
//
// q/k/v share an activation, and so do gate/up. Row-concatenating them
// turns three grids of 512/128/128 blocks into one of 768 and removes
// 108 launches a token (252 → 144). On **Slot32** that was measured at
// 0.803 ms (`docs/mesures/fusion-qkv-cuda-2026-08-05.txt`); this arm
// exists because that number does not transfer to Planes14 and cannot
// be rescaled into it — see `kernels/planes_seg.cu`.
//
// Both layouts are fused here, and both are timed in the same process
// and the same rounds. That is the whole methodological point: a
// Planes14 delta compared against a Slot32 delta from another job would
// be exactly the inter-process subtraction this repository has already
// had to retract once.
//
// The concatenation is on the *indices*, before transcoding, so a fused
// matrix is one ordinary stream — and `seg_concat` is proved on the
// development machine by `tests/planes_segment_matches_unfused.rs`,
// nine mutants deep.
struct FusedMat {
    name: String,
    d_out: usize,
    nblocks: usize,
    tail_w: usize,
    words: cudarc::driver::CudaSlice<u32>,
    bases: cudarc::driver::CudaSlice<u32>,
    pwords: cudarc::driver::CudaSlice<u32>,
    gscale: cudarc::driver::CudaSlice<f32>,
    gs_off: cudarc::driver::CudaSlice<u32>,
    rscale: cudarc::driver::CudaSlice<f32>,
    tail: cudarc::driver::CudaSlice<f32>,
    slot_bytes: u64,
    planes_bytes: u64,
    /// The Tetra stream of the same group, present only when
    /// `tetra48` is in the selection. `Option` and not a sentinel: a
    /// group whose members are not all Tetra has no such stream, and
    /// an empty `CudaSlice` would launch and read garbage.
    ///
    /// It carries its own scales, like `TetraArm`. The `gscale`,
    /// `rscale` and `tail` above belong to the FIRST file; until
    /// 2026-11-04 the Tetra launch read them, on a stream transcoded
    /// from the first file's indices, and was compared against the
    /// second file's separate launches.
    tetra: Option<FusedTetra>,
    tstride: u32,
    tetra_bytes: u64,
    /// Indices into `mats` of the matrices this one replaces, in row
    /// order — the fused output must equal their outputs concatenated.
    parts: Vec<usize>,
}

/// The Tetra half of a fused group: the SECOND file's words, centroid
/// pairs, `gs_off`, row scales and tail, concatenated by
/// `seg_concat_tetra48` from the same members in the same order.
struct FusedTetra {
    words: cudarc::driver::CudaSlice<u32>,
    gscale: cudarc::driver::CudaSlice<f32>,
    gs_off: cudarc::driver::CudaSlice<u32>,
    rscale: cudarc::driver::CudaSlice<f32>,
    tail: cudarc::driver::CudaSlice<f32>,
}

pub fn run() -> Result<(), String> {
    let (phases, tetra_path, union) = select_arms()?;
    // The tile, resolved HERE, before any transcoding:
    // it opens the card, and a refusal that fires after three minutes of
    // transcoding on a rented card is worth nothing. It has to precede the
    // source anyway — it is a `#define` in it.
    let tile = llvq_cuda::tile::resolve(llvq_cuda::gpu::probe_compute_cap()?);
    println!("{}", tile.provenance());

    let g70_needed = union.has(arms::GOLAY70V1) || union.has(arms::GOLAY70V2);

    let (seg_tetra, pers_flags, src, cuda, dev) = compile(tile, union)?;
    let mut inputs = upload_inputs(&cuda, &dev, union, g70_needed)?;

    let phase0 = phases[0];
    let (tetra, tetra_int4) = read_tetra(&tetra_path, union)?;
    let build = |s: &Src| {
        build_mat(
            s, &inputs.x, &inputs.fd, &inputs.table, union, &inputs.searcher, g70_needed,
            &inputs.golay, &inputs.g70cls, phase0, &cuda, &tetra,
        )
    };
    let (t0, mats, srcs, n_weights, source) = load_mats(
        &build, union, g70_needed, &tetra_path, &tetra, tetra_int4, inputs.reps, &mut inputs.rng,
    )?;

    // 🕳️ **The fusion arm (A4) is built AFTER the five-arm table, and no
    // longer here.** It used to be here, and it moved the object the table
    // measures twice over, with no log line saying so:
    //
    //   * `max_dout` chained the fused `d_out`, so `d_y` went from 9,728
    //     to 19,456. But `Planes12x` and `Golay70` do their
    //     `memset_zeros(y)` over the **whole** slice and **inside the
    //     timer** (a claimed choice, see `launch_planes12x`). The two
    //     correction arms therefore paid ~9.8 MB more per pass, ~+0.4%,
    //     than in the run that published their numbers: a systematic and
    //     one-directional bias, on exactly the two arms we want to know
    //     whether they drifted.
    //   * its ~2.9 GB of fused streams stayed resident while the five arms
    //     were being timed, taking occupancy from ~15.3 to ~18.4 GB. VRAM
    //     pressure is the named suspect for the ×20 spread observed going
    //     from four arms to five.
    //
    // Neither is a bug: they are side effects of a lot added afterwards
    // (A4, commit 2d56cce, 2026-08-09). But they made the five-arm table
    // **incomparable with the run that published it**, and that is the one
    // thing this table has to guarantee.
    let max_dout = mats.iter().map(|m| m.d_out).max().unwrap();
    let d_y = cuda.zeros_f32(max_dout)?;
    // The AWQ arm writes a binary16 output, which is what their kernel
    // does. A separate buffer rather than a reinterpretation of `d_y`:
    // two types, two buffers, and nothing that can overlap, and no buffer
    // at all when the arm is not selected.
    let d_yh = match union.has(arms::AWQ) || union.has(arms::CUBLASF16) {
        true => Some(cuda.zeros_u16(max_dout)?),
        false => None,
    };
    // Printed because it was not: that silence is what let `d_y` double
    // with nobody able to read it in a log.
    println!("  d_y: {max_dout} f32, the table and nothing else (section A4 has its own)");
    println!(
        "  {} matrices, {:.2} G weights, in {:.0} s, block-by-block proofs of the \
         arms that were built",
        mats.len(),
        n_weights as f64 / 1e9,
        t0.elapsed().as_secs_f64()
    );
    measure(
        cuda, tile, dev, union, phases, g70_needed, seg_tetra, pers_flags, src, inputs, tetra, mats,
        srcs, source, max_dout, d_y, d_yh,
    )
}
