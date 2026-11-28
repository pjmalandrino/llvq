//! 4E on CUDA: the whole decode token in one cooperative launch, N tokens a
//! launch (`llvq-cuda/kernels/pers_4e.cu`). Built by `bin/fusedrun`'s
//! `LLVQ_PERS_4E` (a measurement), and by a served config that names the
//! persistent decode (`crate::served_decode`: the v9 configs at 4B, 8B and
//! 14B, kernel form from position 0).
//!
//! Every phase of the kernel equals the served chain bit for bit on the host
//! (`llvq-cuda/tests/pers_matches_served/`), attention included as the
//! "kernel" form computes it (`attn_ref.rs`). So on the card, with every
//! decode row of the eager path sent to the attention kernel
//! (`Qwen3::set_kernel_from_for_measurement(0)`), 4E owes that path's tokens.
//!
//! The runtime refuses a launch that starts below its threshold. Installed,
//! the threshold is [`crate::kvq::KERNEL_FROM`]: an eager config that names
//! the kernel form (`configs/qwen3-4b-tetra-e4-kernel-from1280.json`) runs
//! candle's grouped arithmetic below 1,280 (cuBLAS matmuls, candle's
//! softmax), whose bits no hand kernel reproduces, so 4E owes that config's
//! tokens only from there. The served configs,
//! `configs/qwen3-{4b,8b,14b}-tetra-e4-4e-v9.json` and the 4B's chain config
//! `configs/qwen3-4b-tetra-e4-4e-v9-chain.json`, name the persistent decode
//! and send every decode row to the kernel from position 0, the form the
//! operator's T8 extension of 2026-11-05 licenses on the sealed 4B
//! (`docs/mesures/attn-t8-below-4b-2026-11-05.txt`). `Served::decoder` sets
//! the threshold to 0 for them (`Runtime4e::set_served_kernel_from`). A
//! measurement may lower it to 0 too
//! ([`Runtime4e::set_kernel_from_for_measurement`]), against an eager path
//! with every row on the kernel.
//!
//! What only the card can decide is RoPE: candle's `rope` is `__half`
//! arithmetic the driver's ptxas may fuse, so [`install`] runs candle's rope
//! on the card and keeps the form (pers_rope_pair) that matches it, or
//! refuses.
//!
//! This version reads the three sealed shapes: q and k Tetra under one
//! rotation, v int4, o int4 (4B, 14B) or Tetra under its own rotation (8B, a
//! `Pers4eOT` image a layer), a Tetra or int4 down per layer, and the head on
//! the embedding's int4 g64 table (4B, tied) or on its own (8B, 14B). Anything
//! else is refused at install by name.
//!
//! v9 (row form 7) reads the int4 words tiled. Under a v9 config the loader
//! wrote them so and the eager path reads the same buffers
//! (`fused_cuda::upload_int4`, `tv_q4_tile.cu`): the int4 bytes are on the
//! card once. On the bench door's file-order runtime v9 relays a copy.

use candle_core::cuda_backend::cudarc::driver::{CudaFunction, CudaSlice, DevicePtr, LaunchConfig, PushKernelArg};
use candle_core::{DType, Device, Storage, Tensor};
use half::f16;

use crate::fused_cuda::{CudaInt4, CudaLattice, TetraView};
use crate::model::{Embed, Head, KvCache, Proj, Qwen3};
use llvq_cuda::pers;

mod decode;
mod install;
mod prepare;
mod tree;
mod tree_launch;
mod views;

pub use install::*;
pub use tree::*;
pub use views::*;

/// The positions 4E's attention scratch is sized for, at most. Defined in
/// `served_decode`, which every build compiles, so a harness checks its
/// prompts against it before a run on any host.
pub const MAX_CAP: usize = crate::served_decode::P4E_MAX_CAP;

pub struct Runtime4e {
    cuda: llvq_cuda::gpu::Cuda,
    f: CudaFunction,
    /// The phase-stamp symbol of the same form and block size, when one
    /// exists (512 threads): [`Runtime4e::decode_stamped`].
    f_stamp: Option<CudaFunction>,
    /// The logits symbol of an E1 module ([`install_e1`]): the token loop
    /// writing the head's binary16 logits ([`Runtime4e::decode_logits`]).
    f_logits: Option<CudaFunction>,
    /// The E1 arm of the module (`pers::P4E_E1_ARMS`), or none for a form's
    /// own module.
    pub e1_arm: Option<&'static str>,
    /// The local memory of the token symbol, in bytes: 0 on every served
    /// path; nonzero only for an arm installed by [`install_e1_local`]. The
    /// driver's own type (`CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES`).
    pub local_bytes: i32,
    pub threads: u32,
    /// The module's row form (`pers::P4E_ROWS_FORMS`).
    pub rows_form: u32,
    pub grid: u32,
    smem: u32,
    /// The launch asks for more shared bytes than the form needs.
    padded: bool,
    pub form: u32,
    layers_host: Vec<pers::P4eLayerHost>,
    args_host: pers::P4eArgsHost,
    layers: CudaSlice<u8>,
    args: CudaSlice<u8>,
    /// The barrier counter: its address is in the images; the host writes it
    /// back to 0 after a faulted launch ([`pers::p4e_settle`]).
    bar: CudaSlice<u32>,
    abort: CudaSlice<u32>,
    nan: CudaSlice<u32>,
    ring: CudaSlice<u32>,
    max_tokens: u32,
    bar_base: u32,
    n_layers: u32,
    /// Buffers whose addresses live in the images: held, never read.
    _held16: Vec<CudaSlice<u16>>,
    _held32: Vec<CudaSlice<u32>>,
    _heldf: Vec<CudaSlice<f32>>,
    _held64: Vec<CudaSlice<u64>>,
    /// The Tetra o images (`Pers4eOT`), one a layer that has one.
    _held8: Vec<CudaSlice<u8>>,
    /// The (cap, K, V addresses) the uploaded images carry.
    kv_seen: Vec<(usize, u64, u64)>,
    /// The first position a launch may start at: the served threshold, or 0
    /// for a measurement against the kernel on every row.
    kernel_from: usize,
    pub lines: Vec<String>,
    /// The tile define of the model's runtime, for the verify module's text.
    tile_define: String,
    inter: u32,
    /// The verify pass, once [`Runtime4e::install_verify`] has run.
    verify: Option<Verify4e>,
    /// The tree verify pass, once [`Runtime4e::install_tree`] has run.
    tree: Option<Tree4e>,
    /// Arm v9k8's int8 caches (`pers::p4e_e1_k8`), or none.
    k8: Option<Kv8>,
    /// How the tree modules load ([`Runtime4e::set_tree_modules`]): PTX at
    /// every start, as the benches ran, unless set.
    tree_modules: TreeModules,
}

/// Arm v9k8's int8 KV caches (`kernels/pers_4e_k8.cuh`, `kv8_ref.rs`): one K
/// and one V buffer a layer, `[n_kv][cap / 8][1,088]` bytes each, which the
/// layer images name in place of the binary16 caches. The prefill's binary16
/// caches stay the eager path's; `pers_kv8_fill` quantizes their positions
/// into these before a launch that hands them over (`pers::p4e_k8_step`).
/// The arm never writes the binary16 caches, and leaves their length alone.
struct Kv8 {
    fill: CudaFunction,
    n_kv: u32,
    bufs: Vec<(CudaSlice<u8>, CudaSlice<u8>)>,
    state: pers::K8State,
    /// The binary16 caches' (K, V) addresses the int8 ones were filled from.
    f16_seen: Vec<(u64, u64)>,
    /// Release each layer's binary16 cache once its positions are quantized
    /// ([`Runtime4e::set_k8_release`], `pers::p4e_k8_step_release`): the int8
    /// caches are then the only copy. False is the research arm as measured.
    release: bool,
}

/// Which symbol a launch runs: the token loop, its phase stamps, or an E1
/// module's logits.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode4e {
    Plain,
    Stamped,
    Logits,
}

/// The card's barrier counter and flags, as [`pers::p4e_settle`] reads and
/// clears them.
struct Counters4e<'a> {
    cuda: &'a llvq_cuda::gpu::Cuda,
    bar: &'a mut CudaSlice<u32>,
    abort: &'a mut CudaSlice<u32>,
    nan: &'a mut CudaSlice<u32>,
}

impl pers::P4eCounters for Counters4e<'_> {
    fn flags(&mut self) -> Result<(u32, u32), String> {
        Ok((self.cuda.down_u32(self.abort)?[0], self.cuda.down_u32(self.nan)?[0]))
    }

    fn zero(&mut self) -> Result<(), String> {
        let stream = self.cuda.stream();
        for d in [&mut *self.bar, &mut *self.abort, &mut *self.nan] {
            stream.memcpy_htod(&[0u32], d).map_err(|e| format!("H2D: {e}"))?;
        }
        stream.synchronize().map_err(|e| format!("sync: {e}"))
    }
}

/// 4E's verify pass (`pers_4ev.cu`): its own module (the v9 module and the
/// verify kernel), its block size and staging area, and its K-row buffers.
struct Verify4e {
    _cuda: llvq_cuda::gpu::Cuda,
    f: CudaFunction,
    threads: u32,
    smem: u32,
    act_floats: u32,
    part_stride: u32,
    toks: CudaSlice<u32>,
    ring: CudaSlice<u32>,
    _held16: Vec<CudaSlice<u16>>,
    held16_addr: [u64; 8],
    part: CudaSlice<f32>,
    tickets: CudaSlice<u32>,
    slots: CudaSlice<u64>,
}

/// The tree verify pass (`pers_4et.cu`, `docs/plan-verify-tree.md`): its own
/// module (the v9 module and the tree kernel), its area and the model's
/// slices, and buffers for `pers::P4T_RMAX` rows. The served path of the
/// 4B's chain config, whose prompt goes through it (`served_decode`'s
/// `install_chain`); `bin/vtreebench` measures it too.
struct Tree4e {
    _cuda: llvq_cuda::gpu::Cuda,
    f: CudaFunction,
    smem: u32,
    area: u32,
    /// The arms a launch picks from (job N1): each its slices and its form.
    arms: Vec<TreeArm>,
    /// The tiled module and the tree-tiled copies, when an arm is tiled.
    tiled: Option<TreeTiled>,
    part_stride: u32,
    ldx: u32,
    n_ot: u32,
    vocab: u32,
    toks: CudaSlice<u32>,
    depth: CudaSlice<u32>,
    path: CudaSlice<u8>,
    mv: CudaSlice<u32>,
    ring: CudaSlice<u32>,
    logits: CudaSlice<u16>,
    /// x_a, x_b, xin, xn, q_raw, k_raw, v_raw, q_out, ctx, g_raw, u_raw, tk, tv
    _held16: Vec<CudaSlice<u16>>,
    held16_addr: [u64; 13],
    part: CudaSlice<f32>,
    fpart: CudaSlice<f32>,
    tickets: CudaSlice<u32>,
    ftickets: CudaSlice<u32>,
    slots: CudaSlice<u64>,
}

/// The tree-tiled arms' modules and the tree-tiled copies of every
/// projection 4E reads, built once at install from the same words
/// (`p4t_tile_words`), and the images of `Pers4eLayer` and `Pers4eArgs` that
/// point at them, re-uploaded when the caches move. The tiled (a2) and the
/// stream (a3) kernels read the same copies.
struct TreeTiled {
    _cudas: Vec<llvq_cuda::gpu::Cuda>,
    /// The tiled symbol, when a tiled arm is installed.
    f: Option<CudaFunction>,
    /// The stream symbol, when a stream arm is installed.
    fs: Option<CudaFunction>,
    /// The a4 symbol, when an a4 arm is installed.
    f4: Option<CudaFunction>,
    /// The a5 symbol, when an a5 arm is installed.
    f5: Option<CudaFunction>,
    /// The a6 to a12ipg32ft symbols, when such an arm is installed.
    fx: [Option<CudaFunction>; 24],
    _held: Vec<CudaSlice<u32>>,
    /// Each layer's q, k, v, o, gate, up and down words in the copies.
    swaps: Vec<[u64; 7]>,
    head: u64,
    layers: CudaSlice<u8>,
    args: CudaSlice<u8>,
    seen: Vec<(usize, u64, u64)>,
}

fn addr16(s: &CudaSlice<u16>, st: &candle_core::cuda_backend::cudarc::driver::CudaStream) -> u64 {
    s.device_ptr(st).0
}
