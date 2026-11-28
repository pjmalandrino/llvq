//! The Linux half of the bench: its imports, its constants, the host types every arm builds, and the byte
//! accounting. It is the module root so that the sibling files, which are its children, read these private
//! items and fields as the single file did.

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

include!("../../../planes14_host.rs");
include!("../../../golay70_host.rs");
include!("../../../seg_host.rs");
include!("../../../e1v_host.rs");

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

mod sources;
use sources::{
    load_awq_source, load_golay_sources, load_golay_v1_source, load_planes12_sources, load_planes_seg_source,
    load_planes_sources, load_qtip_sources,
};

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

mod launch;
use launch::{
    awq_bytes, awq_quant_row, awq_strides, launch_awq, launch_e1v, launch_golay70, launch_nullk, launch_planes,
    launch_planes12x, launch_planes_seg, launch_qtip, launch_tetra48, launch_tetra48_seg, qtip_pack_u32,
};

mod bench;
pub use bench::run;
