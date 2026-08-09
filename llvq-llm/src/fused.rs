//! Loading the sealed artifact **without decoding it** — the shape a fused
//! kernel needs.
//!
//! ## What this is not
//!
//! [`crate::sealed::load`] rebuilds a model by calling `decode_matrix` on
//! every projection, which materializes 3.63 G weights as f16 tensors. That is
//! correct, it is what every published perplexity refers to, and it throws
//! away the entire point of the format: once decoded, the file occupies as
//! much memory as an f16 checkpoint and runs at exactly its speed. The
//! 2026-08-05 miniature run measured that plainly — 42.7 tok/s against 42.8.
//!
//! This module keeps the weights encoded. It transcodes each matrix to the
//! runtime layout the fused matvec reads — `Planes14` by default, `Planes12x`
//! or `Slot32` under `LLVQ_FUSED_LAYOUT` (see [`FusedLayout`]) — and hands
//! the host tables over. Nothing here touches a GPU: this is the portable
//! half, and it is testable without a card.
//!
//! ## The rotation tables, and why they are here
//!
//! The artifact is quantized in a rotated basis, so a kernel reading the
//! stored weights computes `W' x`, not `W x`. The identity it owes is
//!
//! ```text
//! y = W x = (W Qᵀ)(Q x) = W' · rot(x)
//! ```
//!
//! pinned by `llvq-artifact/tests/fused_path_matches_dense.rs`. A kernel
//! cannot rebuild `Q` from its seed — that would mean porting `SplitMix64` and
//! Gram–Schmidt to the device — so the host builds it once per distinct
//! `(width, seed)` and ships three small tables.
//!
//! Distinct pairs are far fewer than matrices: q/k/v share an activation and
//! therefore a rotation, and so do gate/up. Qwen3-4B has 252 projections and
//! **144** rotations, and the tables together weigh under a megabyte.

use std::collections::HashMap;
use std::io::Read;

use llvq_artifact::runtime::{
    transcode, transcode_planes12x, transcode_planes14, ClassTable, Layout, Planes12xBlocks,
    PlanesBlocks, RuntimeBlocks,
};
use llvq_quant::rotation::Rotation;
use llvq_search::fastdec::FastDecoder;
use llvq_search::Searcher;

/// Which runtime layout the fused path reads.
///
/// Resolved once, from `LLVQ_FUSED_LAYOUT`, before any transcoding: the whole
/// model is one layout, the kernel is chosen by it, and the two cannot drift
/// apart because every [`HostStream`] carries its variant with it.
///
/// `Planes14` is the default — the reference layout since C1 (2026-08-06,
/// 1.14× over `Slot32` at identical decoded content, 4.804 against 5.510
/// b/weight on the published 4B). `Planes12x` is the sparse overlay measured
/// at the bench on 2026-08-07 (4.342 b/weight, 1.98× against FP16, exact
/// reconstruction) and wired here so it can be measured *in the model*;
/// ⚠️ on the 8B it does not pay — `Planes14` + a q8 embedding already sits
/// under the AWQ 4-bit at 5.322 b/param — and the reason it is here anyway is
/// the 14B, where the carried share falls back to ~10.5 %
/// (`docs/mesures/rtbits-planes-8b-2026-08-09.txt`). `Slot32` stays as the
/// comparison arm and the fallback, bit-identical to what shipped before the
/// switch existed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FusedLayout {
    Planes14,
    Planes12x,
    Slot32,
}

impl FusedLayout {
    /// Parse the value of `LLVQ_FUSED_LAYOUT`. `None` (unset) and the empty
    /// string mean the default; anything else must name a layout exactly —
    /// a typo silently falling back to a default would make an A/B lie.
    pub fn parse(v: Option<&str>) -> Result<Self, String> {
        match v {
            None | Some("") => Ok(Self::Planes14),
            Some("planes14") => Ok(Self::Planes14),
            Some("planes12x") => Ok(Self::Planes12x),
            Some("slot32") => Ok(Self::Slot32),
            Some(other) => Err(format!(
                "LLVQ_FUSED_LAYOUT={other} : valeurs admises « planes14 » (défaut), \
                 « planes12x » et « slot32 »"
            )),
        }
    }

    /// Resolve from the environment.
    pub fn from_env() -> Result<Self, String> {
        let v = std::env::var("LLVQ_FUSED_LAYOUT").ok();
        Self::parse(v.as_deref())
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Planes14 => "planes14",
            Self::Planes12x => "planes12x",
            Self::Slot32 => "slot32",
        }
    }
}

/// How the carried embedding tables — the input table, and the `lm_head` when
/// the model unties it — sit on the device.
///
/// Resolved once, from `LLVQ_EMBED`, and printed next to the bytes it
/// decides. `F16` is the shipped behaviour — the embedding decoded to an f16
/// tensor, 778.1 MB on the 4B. `Q8` keeps it as the int8 g64 payload lot B
/// validated (ppl 16.9379 identical, MMLU within sigma): 388.96 MB of int8
/// plus 24.31 MB of f16 scales and biases, read by two dedicated kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbedMode {
    F16,
    Q8,
}

impl EmbedMode {
    /// Parse the value of `LLVQ_EMBED`. Same contract as [`FusedLayout`]:
    /// unset and empty mean the default, anything else must name a mode
    /// exactly — a typo silently falling back would make an A/B lie.
    pub fn parse(v: Option<&str>) -> Result<Self, String> {
        match v {
            None | Some("") => Ok(Self::F16),
            Some("f16") => Ok(Self::F16),
            Some("q8") => Ok(Self::Q8),
            Some(other) => Err(format!(
                "LLVQ_EMBED={other} : valeurs admises « f16 » (défaut) et « q8 »"
            )),
        }
    }

    /// Resolve from the environment.
    pub fn from_env() -> Result<Self, String> {
        let v = std::env::var("LLVQ_EMBED").ok();
        Self::parse(v.as_deref())
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Q8 => "q8",
        }
    }
}

/// The group width of the q8 embedding path — MLX's scheme, and the one
/// `bin/embedq` wrote into the artifact lot B scored. The kernels hard-code
/// it (`c >> 6`), so it is a constant, not a knob.
pub const EMBED_GROUP: usize = 64;

/// Turn a carried tensor into the int8 g64 payload the embedding kernels read.
///
/// Two accepted inputs, one output:
///
///  * an f16 tensor is quantized **by the same function `bin/embedq` calls**
///    (`embedquant::quantize_affine`, bits = 8, group = 64) — that call, not a
///    reimplementation, is what transfers lot B's quality verdict to this
///    path;
///  * a tensor already stored as int8 g64 (an artifact `embedq` produced) is
///    passed through byte-identical;
///  * anything else — int4, another group — is refused rather than silently
///    requantized: the validated object is q8 g64 and nothing next to it.
pub fn embed_q8(t: llvq_artifact::RawTensor) -> Result<llvq_artifact::RawTensor, String> {
    match &t.data {
        llvq_artifact::RawData::F16(_) => {
            crate::embedquant::quantize_affine(&t, 8, EMBED_GROUP).map_err(|e| e.to_string())
        }
        llvq_artifact::RawData::Quant(q) if q.bits == 8 && q.group == EMBED_GROUP => Ok(t),
        llvq_artifact::RawData::Quant(q) => Err(format!(
            "{} : int{} g{} porté par le fichier — le chemin q8 ne lit que int8 g{EMBED_GROUP}",
            t.name, q.bits, q.group
        )),
    }
}

/// Device bytes of the q8 embedding: `(packed, scales + biases)`.
///
/// Pure arithmetic on the dims, so the announced footprint is testable
/// without a card: on `[151936, 2560]` this is 388 956 160 + 24 309 760
/// bytes — the 413.3 MB the mission statement quotes.
pub fn q8_device_bytes(dims: &[usize]) -> (u64, u64) {
    let row_len = dims.last().copied().unwrap_or(1).max(1);
    let n: usize = dims.iter().product();
    let rows = n / row_len;
    let gpr = row_len.div_ceil(EMBED_GROUP);
    (n as u64, (rows * gpr) as u64 * 4)
}

/// The carried tensor holding the input embedding.
pub const EMBED_NAME: &str = "model.embed_tokens.weight";
/// The carried tensor holding the output projection. Present only when the
/// model unties its two ends — Qwen3-4B has `tie_word_embeddings = true` and
/// carries no such tensor, Qwen3-8B has it `false` and carries a second table
/// of the same shape and different values.
pub const HEAD_NAME: &str = "lm_head.weight";

/// The embedding tables the q8 path takes off the carried list, quantized.
///
/// One table when the model ties its two ends: a single device buffer feeds
/// the gather at the input and the `lm_head` at the output, which is the −365
/// MB lot B validated on the 4B. Two tables when it unties them: they are
/// different weights, so they are different buffers, and nothing in this path
/// may substitute one for the other.
pub struct EmbedTables {
    /// [`EMBED_NAME`], int8 g64.
    pub embed: llvq_artifact::RawTensor,
    /// [`HEAD_NAME`], int8 g64 — `None` exactly when the two ends are tied.
    pub head: Option<llvq_artifact::RawTensor>,
}

impl EmbedTables {
    /// The distinct payloads to upload, in buffer order.
    pub fn buffers(&self) -> Vec<&llvq_artifact::RawTensor> {
        let mut v = vec![&self.embed];
        v.extend(self.head.iter());
        v
    }

    /// Which uploaded buffer each end of the model reads: `(embedding, head)`.
    ///
    /// The wiring is returned as a **value** rather than decided at the call
    /// site, and that is deliberate. The call site sits behind
    /// `cfg(target_os = "linux", feature = "cuda")` and compiles on no machine
    /// this suite runs on, so an `lm_head` pointed at the embedding's buffer
    /// would be caught by nothing until a billed job produced plausible wrong
    /// logits. As data, it is pinned by a test that runs anywhere.
    pub fn wiring(&self) -> (usize, usize) {
        match self.head {
            None => (0, 0),
            Some(_) => (0, 1),
        }
    }
}

/// Take the embedding tables off the carried list and turn them into the int8
/// g64 payload the embedding kernels read.
///
/// `tie` is the model's `tie_word_embeddings`. Under `false` the file must
/// carry [`HEAD_NAME`] too, and its absence is an error that names it — never
/// a silent fall back on the embedding, which would produce a model that runs
/// and is wrong.
///
/// Both tables go through [`embed_q8`], so both accept either an f16 tensor
/// (quantized by the very function `bin/embedq` calls) or bytes `embedq`
/// already wrote, passed through byte-identical.
pub fn take_embed_tables(
    raw: &mut Vec<llvq_artifact::RawTensor>,
    tie: bool,
) -> Result<EmbedTables, String> {
    fn take(
        raw: &mut Vec<llvq_artifact::RawTensor>,
        name: &str,
    ) -> Option<llvq_artifact::RawTensor> {
        // `swap_remove` reorders what is left, so the second lookup below
        // searches the mutated vector by name rather than reusing an index.
        let i = raw.iter().position(|t| t.name == name)?;
        Some(raw.swap_remove(i))
    }
    let embed = take(raw, EMBED_NAME).ok_or_else(|| format!("ne porte pas {EMBED_NAME}"))?;
    let embed = embed_q8(embed)?;
    let head = if tie {
        None
    } else {
        let t = take(raw, HEAD_NAME).ok_or_else(|| {
            format!(
                "tie_word_embeddings=false, mais le fichier ne porte pas {HEAD_NAME} — \
                 les deux extrémités sont déliées, et le chemin q8 ne remplace jamais \
                 un lm_head manquant par l'embedding"
            )
        })?;
        Some(embed_q8(t)?)
    };
    Ok(EmbedTables { embed, head })
}

/// The embedding tables still carried as ordinary tensors, embedding first and
/// `lm_head` second whatever order the file wrote them in.
pub fn carried_embed_tables(raw: &[llvq_artifact::RawTensor]) -> Vec<&llvq_artifact::RawTensor> {
    [EMBED_NAME, HEAD_NAME]
        .iter()
        .filter_map(|n| raw.iter().find(|t| t.name == *n))
        .collect()
}

/// What the embedding tables cost on the device, and the line that says so.
///
/// Arithmetic on dims alone, so the announced footprint is checkable without a
/// card. It exists because the line it replaces announced **one** table's
/// bytes on a model carrying two: with the ends untied it under-reported by a
/// whole table — 1.24 GB on the 8B — while the total printed just below it was
/// right. Half-true is the worst kind of wrong in a footprint report, because
/// the two lines then contradict each other and neither is obviously the liar.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbedReport {
    pub mode: EmbedMode,
    /// One entry per table resident on the device, in buffer order:
    /// `(name, payload bytes, scale/bias bytes)`. The last is zero at f16.
    pub tables: Vec<(String, u64, u64)>,
}

impl EmbedReport {
    /// Build the report for the tables the device will actually hold.
    pub fn new(mode: EmbedMode, tables: &[&llvq_artifact::RawTensor]) -> Self {
        let tables = tables
            .iter()
            .map(|t| match mode {
                EmbedMode::F16 => (t.name.clone(), t.len() as u64 * 2, 0),
                EmbedMode::Q8 => {
                    let (packed, sb) = q8_device_bytes(&t.dims);
                    (t.name.clone(), packed, sb)
                }
            })
            .collect();
        Self { mode, tables }
    }

    /// Payload bytes over every table.
    pub fn packed(&self) -> u64 {
        self.tables.iter().map(|&(_, p, _)| p).sum()
    }

    /// Scale and bias bytes over every table; zero at f16.
    pub fn meta(&self) -> u64 {
        self.tables.iter().map(|&(_, _, s)| s).sum()
    }

    /// Device bytes of every embedding table together.
    pub fn total(&self) -> u64 {
        self.packed() + self.meta()
    }

    /// The load-time line — it names how many tables it is counting, so a
    /// reader can tell a tied model from an untied one without doing division.
    pub fn line(&self) -> String {
        let names: Vec<&str> = self.tables.iter().map(|(n, ..)| n.as_str()).collect();
        let which = if names.len() == 1 {
            format!("1 table ({}, lm_head lié dessus)", names[0])
        } else {
            format!("{} tables ({})", names.len(), names.join(" + "))
        };
        match self.mode {
            EmbedMode::F16 => format!(
                "embedding : f16 (LLVQ_EMBED) — {which}, {:.1} Mo sur la carte",
                self.total() as f64 / 1e6
            ),
            EmbedMode::Q8 => format!(
                "embedding : q8 g64 (LLVQ_EMBED) — {which}, {:.1} Mo sur la carte \
                 (int8 {:.1} + échelles/biais {:.1})",
                self.total() as f64 / 1e6,
                self.packed() as f64 / 1e6,
                self.meta() as f64 / 1e6
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// The kernel sources a layout needs
// ---------------------------------------------------------------------------
//
// These live here rather than next to their only caller (`fused_cuda`) for one
// reason, and it is the same reason `EmbedTables::wiring` returns a value: on
// this workspace's development machine, `fused_cuda` **compiles on no target**
// — it is behind `cfg(all(target_os = "linux", feature = "cuda"))`, and
// `candle-kernels` needs a real `nvcc` to build, so `--target
// x86_64-unknown-linux-gnu` does not reach it either. Anything left in that
// file is unchecked until a billed job. The source list is a real contract —
// NVRTC receives one concatenated string and the order of the parts decides
// whether it compiles — so it belongs on the side of the boundary a test can
// reach.

/// The bit-plane kernel sources, embedded like every other kernel so a run is
/// reproducible from the binary alone. The files from `llvq-cuda` are reused
/// verbatim (they belong to the C1 and M2 lots and are pinned by planesbench);
/// the `tv_*_h.cu` files are this crate's own — the half-storing entry points
/// only the inference path needs.
const PLANES_CUH_EMBED: &str = include_str!("../../llvq-cuda/kernels/llvq_planes.cuh");
const PLANES_CU_EMBED: &str = include_str!("../../llvq-cuda/kernels/planes.cu");
const PLANES_H_CU_EMBED: &str = include_str!("../kernels/tv_planes_h.cu");
const PLANES12_CUH_EMBED: &str = include_str!("../../llvq-cuda/kernels/llvq_planes12.cuh");
const PLANES12X_H_CU_EMBED: &str = include_str!("../kernels/tv_planes12x_h.cu");

/// The bit-plane sources a layout needs, **in NVRTC concatenation order**.
///
/// The order is not cosmetic: NVRTC has no filesystem, the host hands it one
/// string, and each part's `#ifndef` guard keys on a macro an earlier part
/// defines. `llvq_planes.cuh` needs `llvq_slot.cuh`, `planes.cu` needs
/// `matvec.cu` (both come from `load_sources_many` ahead of these),
/// `llvq_planes12.cuh` needs `llvq_planes.cuh`, and `tv_planes12x_h.cu` needs
/// both.
///
/// `Planes12x` is `Planes14`'s list plus two files, not a list of its own, and
/// that is deliberate: the two arms then share one translation-unit shape, so
/// the register report of `tv_planes_h` stays a drift detector on the
/// `Planes12x` build too. `Slot32` takes none of them, so its translation unit
/// is bit-identical to what shipped before any of this existed.
pub fn planes_source_names(layout: FusedLayout) -> &'static [&'static str] {
    match layout {
        FusedLayout::Slot32 => &[],
        FusedLayout::Planes14 => &["llvq_planes.cuh", "planes.cu", "tv_planes_h.cu"],
        FusedLayout::Planes12x => &[
            "llvq_planes.cuh",
            "planes.cu",
            "tv_planes_h.cu",
            "llvq_planes12.cuh",
            "tv_planes12x_h.cu",
        ],
    }
}

/// The `extern "C"` matvec entry point `layout` launches.
pub fn matvec_kernel_name(layout: FusedLayout) -> &'static str {
    match layout {
        FusedLayout::Planes14 => "tv_planes_h",
        FusedLayout::Planes12x => "tv_planes12x_h",
        FusedLayout::Slot32 => "tv_slot_h",
    }
}

/// The bit-plane parts of `layout`, honouring `LLVQ_KERNEL_DIR` with the same
/// contract as `llvq_cuda::load_sources_many`: without the variable, the
/// embedded text; with it, **all** the files from the directory, disclosed
/// loudly by the caller — mixing embedded and overridden parts would make the
/// printed sha256 untraceable.
///
/// The override reads every name from one directory even though the embedded
/// copies come from two crates. That is the existing contract and it is the
/// right one: a bench that overrides the kernels must present a complete,
/// self-consistent set, not a merge of a directory and a binary.
pub fn load_planes_sources(
    layout: FusedLayout,
) -> Result<(Vec<String>, Option<String>), String> {
    let names = planes_source_names(layout);
    let embedded = |n: &str| match n {
        "llvq_planes.cuh" => Ok(PLANES_CUH_EMBED),
        "planes.cu" => Ok(PLANES_CU_EMBED),
        "tv_planes_h.cu" => Ok(PLANES_H_CU_EMBED),
        "llvq_planes12.cuh" => Ok(PLANES12_CUH_EMBED),
        "tv_planes12x_h.cu" => Ok(PLANES12X_H_CU_EMBED),
        other => Err(format!("pas de copie embarquée de {other}")),
    };
    match std::env::var("LLVQ_KERNEL_DIR") {
        Err(_) => Ok((
            names
                .iter()
                .map(|n| embedded(n).map(str::to_string))
                .collect::<Result<_, _>>()?,
            None,
        )),
        Ok(dir) => Ok((
            names
                .iter()
                .map(|n| {
                    std::fs::read_to_string(std::path::Path::new(&dir).join(n))
                        .map_err(|e| format!("LLVQ_KERNEL_DIR={dir} : {n} : {e}"))
                })
                .collect::<Result<_, _>>()?,
            Some(dir),
        )),
    }
}

/// Mirrors `LLVQ_ROT_KMAX` in `llvq-cuda/kernels/llvq_rot.cuh`.
///
/// Duplicated across a language boundary on purpose — the kernel is in another
/// crate that does not build on this machine — and pinned by
/// `the_kmax_constant_matches_the_kernel` on the CUDA side.
pub const ROT_KMAX: usize = 32;

/// One matrix's payload in one runtime layout, as the words a kernel indexes.
///
/// An enum rather than optional fields, and that is the guarantee: a
/// `Planes14` stream **has no bases** — not an empty vector, no field at all —
/// so no code path can upload or launch with a bases array on the planes path.
/// The compiler enforces what a review would otherwise have to.
pub enum HostStream {
    Slot32 {
        /// `Slot32` payload as `u32` words, padded for the kernel's five-word
        /// read window past the last block.
        words: Vec<u32>,
        /// Byte offset of each group of 32 blocks, plus a final sentinel.
        bases: Vec<u32>,
    },
    Planes14 {
        /// The uniform 14-byte records as `u32` words, padded for the
        /// kernel's four-word read window past the last block.
        words: Vec<u32>,
    },
    Planes12x {
        /// The uniform 12-byte main-stream records as `u32` words. `12·b` is
        /// always word-aligned, so the three-word window needs no padding at
        /// all; the four spare bytes are packing symmetry with the other
        /// streams, not a correctness need.
        words: Vec<u32>,
        /// Matrix-wide block index of each exception, strictly ascending.
        exc_idx: Vec<u32>,
        /// One exact 14-byte `Planes14` record per exception, as `u32` words,
        /// padded for the four-word window `planes_fields` reads on it.
        exc_words: Vec<u32>,
        /// `d_out + 1` prefix offsets into `exc_idx`: output row `r` owns
        /// exceptions `row_exc[r] .. row_exc[r+1]`.
        ///
        /// This table is what lets the inference kernel keep one warp per
        /// output row and drop the bench kernel's memset-and-atomicAdd
        /// protocol — see `kernels/tv_planes12x_h.cu`. It exists because
        /// `exc_idx` is ascending and blocks are row-major, so a row's
        /// exceptions are contiguous; [`row_offsets`] asserts that rather
        /// than assuming it, since the kernel derives an exception's column
        /// as `b − row·nblocks` and a wrong row would read the wrong
        /// activation slice and produce finite, plausible, wrong numbers.
        row_exc: Vec<u32>,
    },
}

/// One projection, transcoded and ready to upload.
pub struct FusedMatrix {
    pub name: String,
    pub d_out: usize,
    pub d_in: usize,
    pub nblocks: usize,
    pub tail_w: usize,
    /// The payload, in the layout [`FusedModel::layout`] names.
    pub stream: HostStream,
    /// The two gain centroids. One bit of gain is hard-coded in every decoder.
    pub gscale: [f32; 2],
    pub rscale: Vec<f32>,
    /// `d_out × tail_w`, row-major, in the rotated basis.
    pub tail: Vec<f32>,
    /// Key into [`FusedModel::rotations`], or `None` in the natural basis.
    pub rotation: Option<RotKey>,
    /// Bytes this matrix costs at runtime — every device array the kernel
    /// reads: the payload, whatever addressing the layout carries beside it
    /// (`Slot32`'s bases, `Planes12x`'s exception table and row offsets), the
    /// tail and the row scales.
    pub bytes: u64,
}

/// A rotation is determined by its width and its seed, and shared by every
/// matrix consuming the same activation.
pub type RotKey = (usize, u64);

/// `Q`, flattened into what a kernel can read.
pub struct RotationTables {
    pub n: usize,
    /// Power-of-two factor: the width of one Walsh–Hadamard group.
    pub m: usize,
    /// Odd factor: the side of the dense block, and the number of groups.
    pub k: usize,
    /// One bit per coordinate, set when the sign is negative.
    pub signbits: Vec<u32>,
    /// `Q_odd`, zero-padded to `ROT_KMAX × ROT_KMAX` row-major. The padding is
    /// load-bearing: the kernel's inner loops run to a compile-time bound.
    pub small: Vec<f32>,
    /// `1/√m`, narrowed once here rather than recomputed on the device where
    /// `sqrtf` and `rsqrtf` would each land somewhere else.
    pub inv: f32,
}

impl RotationTables {
    pub fn build(n: usize, seed: u64) -> Result<Self, String> {
        let rot = Rotation::new(n, seed);
        let (m, k) = (rot.pow2(), rot.odd());
        if k > ROT_KMAX {
            return Err(format!(
                "largeur {n} : facteur impair {k} au-delà de KMAX={ROT_KMAX}. Le noyau de \
                 rotation ne la traite pas — cf. llvq_rot.cuh."
            ));
        }
        let mut signbits = vec![0u32; n.div_ceil(32)];
        for (i, &s) in rot.signs().iter().enumerate() {
            if s < 0.0 {
                signbits[i >> 5] |= 1 << (i & 31);
            }
        }
        let mut small = vec![0.0f32; ROT_KMAX * ROT_KMAX];
        for g in 0..k {
            for t in 0..k {
                small[g * ROT_KMAX + t] = rot.small()[g * k + t] as f32;
            }
        }
        Ok(Self {
            n,
            m,
            k,
            signbits,
            small,
            inv: (1.0f64 / (m as f64).sqrt()) as f32,
        })
    }
}

/// Everything a fused runtime needs, still on the host.
pub struct FusedModel {
    /// The runtime layout every matrix was transcoded to.
    pub layout: FusedLayout,
    pub matrices: Vec<FusedMatrix>,
    pub rotations: HashMap<RotKey, RotationTables>,
    /// Embedding and norms, carried verbatim, **still in the file's own
    /// encoding** — f16 bits, or int8 g64 for an `embedq` output. Decoding is
    /// the consumer's decision: the f16 path widens, the q8 path keeps the
    /// bytes. Holding the encoded form also halves what this struct weighs
    /// (the 778 MB embedding would be 1.5 GB as f32).
    pub raw: Vec<llvq_artifact::RawTensor>,
    pub config_json: Vec<u8>,
    pub tokenizer_json: Vec<u8>,
    pub quantized_weights: usize,
    pub carried_weights: usize,
    /// Size of the file on disk.
    pub file_bytes: u64,
    /// Bytes the projections occupy at runtime, in the chosen layout.
    pub runtime_bytes: u64,
}

impl FusedModel {
    /// Bits per weight the runtime actually spends on quantized projections.
    ///
    /// Not the same accounting as the file's: the runtime layout pays its
    /// addressing where the file packs 48 bits per block exactly. On the
    /// published 4B: `Slot32` (byte-rounded group strides plus a `u32` base
    /// per group) measured 5.510 b/weight, `Planes14` (uniform 14-byte
    /// records, no bases) 4.804, `Planes12x` (12-byte records plus the
    /// exception table) 4.342 at the bench, against 2.0702 effective in the
    /// file.
    ///
    /// ⚠️ This crate's `Planes12x` figure sits marginally *above* the bench's,
    /// on purpose: it also bills the `d_out + 1` row-offset table the
    /// inference kernel reads, which the bench arm has no equivalent of
    /// (32/`d_in` b/weight — 0.0125 at `d_in = 2560`). Under-reporting a
    /// device array is exactly the accounting sin the K-1 lot spent a dossier
    /// removing.
    pub fn runtime_bits_per_weight(&self) -> f64 {
        self.runtime_bytes as f64 * 8.0 / self.quantized_weights as f64
    }
}

fn read_u32(r: &mut impl Read) -> Result<u32, String> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).map_err(|e| e.to_string())?;
    Ok(u32::from_le_bytes(b))
}

/// Pack a transcoded stream into `u32` words the kernel can index.
///
/// The twenty trailing zero bytes are not slack: `slot_fields` reads five
/// consecutive words from `byte >> 2`, so the last block of the stream reads
/// past its own payload by design. Without the pad that is an out-of-bounds
/// read — benign on Metal, an illegal address on CUDA, and it kills the
/// context in the middle of a billed job.
fn pack_words(rt: &RuntimeBlocks) -> Vec<u32> {
    let mut bytes = rt.data.clone();
    bytes.extend_from_slice(&[0u8; 20]);
    while !bytes.len().is_multiple_of(4) {
        bytes.push(0);
    }
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// [`pack_words`] for a bit-plane stream — the four-word window's padding.
///
/// `planes_fields` reads four consecutive words from `(14·b) >> 2`, so the
/// last record's window reaches up to 2 bytes past the `14·n` payload. Four
/// spare bytes plus word alignment keep that read in bounds — the same
/// arithmetic as planesbench's upload, and skipping it is an illegal address
/// on CUDA, in the middle of a billed job.
///
/// Used for the `Planes14` stream *and* for the `Planes12x` exception table,
/// which is a `Planes14` record array read through the very same window. The
/// `Planes12x` main stream is word-aligned by construction (12 bytes a
/// record) and needs none of this; it goes through here anyway so one
/// function owns the packing.
///
/// 🕳️ **Mutation, 2026-08-09: deleting the four bytes kills no test, and the
/// reason is arithmetic, not a weak test.** For a 14-byte stride the
/// word-alignment step alone already supplies exactly what the window needs.
/// Write `L = ceil(14n/4)·4` for the padless length and `T = 4·⌊14(n−1)/4⌋ +
/// 16` for the far end of the last record's four-word window. `14(n−1) ≡ 0
/// (mod 4)` exactly when `n` is odd, and then `T = 14n + 2` while `14n ≡ 2
/// (mod 4)`, so `L = 14n + 2 = T`; when `n` is even, `T = 14n` and `14n ≡ 0
/// (mod 4)`, so `L = 14n = T`. Equal in both cases — never short, never
/// spare. The pad is therefore an *equivalent* mutant in the §5 sense (dead
/// code, not an untested branch), and it stays: it costs four bytes per
/// matrix, it is what planesbench uploads, and the identity above holds for
/// this stride and this window width only. A layout whose record size and
/// window stop coinciding this way would need it, and would get no warning.
fn pack_plane_bytes(bytes: &[u8]) -> Vec<u32> {
    let mut bytes = bytes.to_vec();
    bytes.extend_from_slice(&[0u8; 4]);
    while !bytes.len().is_multiple_of(4) {
        bytes.push(0);
    }
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// The `d_out + 1` prefix offsets of each output row's slice of `exc_idx`.
///
/// Blocks are row-major (`b = row·nblocks + col`) and `exc_idx` is strictly
/// ascending, so each row's exceptions form a contiguous, ascending run and a
/// single pass finds every boundary.
///
/// Why this is a function of its own rather than a loop inlined into
/// [`Transcoder::stream`]: `tv_planes12x_h` recovers an exception's column as
/// `b − row·nblocks`, using the row it already owns rather than the division
/// `planes12x_locate` pays to recover one. If a block were filed under the
/// wrong row, that subtraction would index a valid-looking slice of the
/// activation and the kernel would return finite, plausible, wrong numbers
/// with no error anywhere. So the two things the kernel actually depends on
/// are **checked**, not assumed:
///
///  * the table is strictly ascending — the premise of the single pass;
///  * every exception lands in some row — nothing past the last block.
///
/// 🕳️ **A third check used to sit here and mutation testing removed it
/// (2026-08-09).** It read `if (exc_idx[e] as usize) < r * nblocks { … }` —
/// "this block is below the row it is being filed under" — and deleting it
/// killed no test, because it is **unreachable**: row `r−1`'s `while` consumed
/// every entry below `r·nblocks`, and the ascending check above it (which runs
/// first in the same iteration) forbids going back. Row 0 is the trivial case,
/// `b < 0`. That is the §5 verdict "the code is dead", not "the test is weak" —
/// and the distinction was checked, not assumed: shifting the row boundary
/// **with the guard also deleted** still fails three tests
/// (`the_row_offsets_partition_the_exception_table`,
/// `planes12x_words_rebuild_the_planes14_content`,
/// `the_planes12x_half_kernel_decides_what_rust_decides`). The row attribution
/// is held by the assertions, so a guard that can never fire was buying
/// nothing but the appearance of care.
fn row_offsets(exc_idx: &[u32], d_out: usize, nblocks: usize) -> Result<Vec<u32>, String> {
    let mut off = Vec::with_capacity(d_out + 1);
    off.push(0u32);
    let mut e = 0usize;
    for r in 0..d_out {
        let hi = ((r + 1) * nblocks) as u32;
        while e < exc_idx.len() && exc_idx[e] < hi {
            if e > 0 && exc_idx[e] <= exc_idx[e - 1] {
                return Err(format!(
                    "table d'exceptions non strictement croissante en {e} \
                     ({} après {})",
                    exc_idx[e],
                    exc_idx[e - 1]
                ));
            }
            e += 1;
        }
        off.push(e as u32);
    }
    if e != exc_idx.len() {
        return Err(format!(
            "{} exceptions au-delà du dernier bloc de la matrice ({} lignes × {nblocks})",
            exc_idx.len() - e,
            d_out
        ));
    }
    Ok(off)
}

/// Everything a transcoding pass needs, built once for a whole model.
///
/// The searcher is here rather than at the call site because of what it costs
/// and what it means. `Planes12x` re-encodes every 5-level block — one exact
/// `nearest_angular` over the m ≤ 12 ball, ~0.7 ms of one core each, 5 096 688
/// of them on the published 4B — and the searcher carries the lattice tables
/// that search needs. Holding it in a struct whose constructor knows the
/// layout makes "the searcher exists exactly when the layout needs one" an
/// invariant of the type rather than a rule a caller has to remember: there is
/// no way to reach [`Self::stream`] for `Planes12x` without one, and no way to
/// build 252 of them by accident.
pub struct Transcoder {
    layout: FusedLayout,
    fd: FastDecoder,
    table: ClassTable,
    /// `Some` exactly for [`FusedLayout::Planes12x`].
    searcher: Option<Searcher>,
}

impl Transcoder {
    /// Build the tables for `layout`, checking up front what the layout needs
    /// from the class table.
    pub fn new(layout: FusedLayout) -> Result<Self, String> {
        let fd = FastDecoder::new();
        let table = ClassTable::new(&fd, 1);
        if matches!(layout, FusedLayout::Planes14 | FusedLayout::Planes12x) {
            // Three bit-planes address eight levels; both plane layouts are
            // only a bijection of the Slot32 content while every class stays
            // within five — `Planes12x` needs it for its exception records,
            // which are Planes14 records. True of the v1 table, asserted
            // rather than assumed — the same guard planesbench re-asserts
            // before any timing.
            if !(0..table.n_entries()).all(|e| table.record(e).len <= 5) {
                return Err("une classe dépasse 5 niveaux : les plans de bits ne \
                            peuvent pas la porter"
                    .into());
            }
        }
        let searcher = matches!(layout, FusedLayout::Planes12x).then(Searcher::new);
        Ok(Self { layout, fd, table, searcher })
    }

    pub fn layout(&self) -> FusedLayout {
        self.layout
    }

    pub fn decoder(&self) -> &FastDecoder {
        &self.fd
    }

    pub fn class_table(&self) -> &ClassTable {
        &self.table
    }

    /// Transcode one matrix's raw codes into the words of the chosen layout.
    ///
    /// Returns the stream and its **payload** bytes — the accounting the
    /// runtime reports, so `Slot32` counts its bases, `Planes14` has none to
    /// count, and `Planes12x` counts its exception table *and* the row-offset
    /// table it adds, because both are device arrays the kernel reads. (The
    /// read-window padding is deliberately excluded, as it always was for
    /// `Slot32`: it is upload slack, not format.)
    ///
    /// `d_out` and `nblocks` describe the row structure `indices` is laid out
    /// in. Only `Planes12x` reads them — it is the only layout whose kernel
    /// needs to know where a row ends — but they are required of every arm so
    /// the shape check below runs on all of them.
    pub fn stream(
        &self,
        indices: &[u64],
        gains: &[u32],
        d_out: usize,
        nblocks: usize,
    ) -> Result<(HostStream, u64), String> {
        if indices.len() != d_out * nblocks {
            return Err(format!(
                "{} codes pour {d_out} lignes × {nblocks} blocs",
                indices.len()
            ));
        }
        match self.layout {
            FusedLayout::Slot32 => {
                let rt = transcode(&self.fd, &self.table, indices, gains, Layout::Slot32)
                    .map_err(|e| e.to_string())?;
                let bytes = rt.data.len() as u64 + rt.bases.len() as u64 * 4;
                let words = pack_words(&rt);
                Ok((HostStream::Slot32 { words, bases: rt.bases }, bytes))
            }
            FusedLayout::Planes14 => {
                let pb: PlanesBlocks =
                    transcode_planes14(&self.fd, &self.table, indices, gains)
                        .map_err(|e| e.to_string())?;
                let bytes = pb.data.len() as u64;
                Ok((HostStream::Planes14 { words: pack_plane_bytes(&pb.data) }, bytes))
            }
            FusedLayout::Planes12x => {
                let s = self
                    .searcher
                    .as_ref()
                    .ok_or("Transcoder::new n'a pas construit de chercheur pour planes12x")?;
                let pb: Planes12xBlocks =
                    transcode_planes12x(&self.fd, &self.table, s, indices, gains)
                        .map_err(|e| e.to_string())?;
                let row_exc = row_offsets(&pb.exc_idx, d_out, nblocks)?;
                let bytes = pb.data.len() as u64
                    + pb.exc_idx.len() as u64 * 4
                    + pb.exc_data.len() as u64
                    + row_exc.len() as u64 * 4;
                Ok((
                    HostStream::Planes12x {
                        words: pack_plane_bytes(&pb.data),
                        exc_idx: pb.exc_idx,
                        exc_words: pack_plane_bytes(&pb.exc_data),
                        row_exc,
                    },
                    bytes,
                ))
            }
        }
    }
}

/// Load a sealed artifact in encoded form, transcoded to `layout`.
///
/// Costs one transcode of every matrix — 142 s for the 4B on eight cores,
/// measured 2026-08-05 — against `sealed::load`'s full decode, which took
/// 208.7 s on the same file and produced eight gigabytes of f16.
///
/// ⚠️ `Planes12x` is a different order of magnitude and it is not a defect:
/// it runs one exact lattice search per 5-level block on top of the same
/// work. Measured on this machine (M3 Max, 12 performance cores), see
/// `tests/fused_planes12x.rs::transcode_of_the_sealed_model_matches_planes14`.
pub fn load(path: &str, layout: FusedLayout) -> Result<FusedModel, String> {
    let file_bytes = std::fs::metadata(path).map_err(|e| e.to_string())?.len();
    let f = std::fs::File::open(path).map_err(|e| format!("open {path}: {e}"))?;
    let mut r = std::io::BufReader::with_capacity(1 << 20, f);
    let head = llvq_artifact::read_header(&mut r).map_err(|e| e.to_string())?;
    if !head.is_self_contained() {
        return Err(format!(
            "{path} n'est qu'un artefact de projections (format v{}) — il ne peut pas \
             tourner seul.",
            head.version
        ));
    }

    let tr = Transcoder::new(layout)?;
    let mut matrices = Vec::with_capacity(head.matrices as usize);
    let mut rotations: HashMap<RotKey, RotationTables> = HashMap::new();
    let mut quantized_weights = 0usize;
    let mut runtime_bytes = 0u64;

    for _ in 0..head.matrices {
        let m = llvq_artifact::read_matrix_raw(&mut r).map_err(|e| e.to_string())?;
        // Every decoder hard-codes one gain bit (`hdr >> 9`). A file with a
        // different gain width would transcode into a coherent stream and
        // decode into garbage — silently.
        if m.centroids.len() != 2 {
            return Err(format!(
                "{} : {} centroïdes, mais les noyaux codent 1 bit de gain en dur",
                m.name,
                m.centroids.len()
            ));
        }
        let nblocks = m.d_in / llvq_core::DIM;
        let tail_w = m.d_in % llvq_core::DIM;
        quantized_weights += m.d_out * m.d_in;

        let rotation = match m.rotation_seed {
            None => None,
            Some(seed) => {
                let key = (m.d_in, seed);
                // `entry` rather than contains/insert, but not for the reason
                // clippy gives: `RotationTables::build` runs a Gram–Schmidt and
                // allocates, and the naive form would do that work on every
                // matrix sharing a rotation — 252 builds where 144 are owed.
                if let std::collections::hash_map::Entry::Vacant(e) = rotations.entry(key) {
                    e.insert(RotationTables::build(m.d_in, seed)?);
                }
                Some(key)
            }
        };

        let (stream, payload) = tr
            .stream(&m.indices, &m.gains, m.d_out, nblocks)
            .map_err(|e| format!("{} : {e}", m.name))?;
        let bytes = payload + (m.d_out * tail_w) as u64 * 4 + m.d_out as u64 * 4;
        runtime_bytes += bytes;

        matrices.push(FusedMatrix {
            name: m.name,
            d_out: m.d_out,
            d_in: m.d_in,
            nblocks,
            tail_w,
            stream,
            gscale: [m.centroids[0] as f32, m.centroids[1] as f32],
            rscale: m.row_scales.iter().map(|&v| v as f32).collect(),
            tail: m.tail.iter().map(|&v| v as f32).collect(),
            rotation,
            bytes,
        });
    }

    let n_raw = read_u32(&mut r)?;
    let mut raw = Vec::with_capacity(n_raw as usize);
    let mut carried_weights = 0usize;
    for _ in 0..n_raw {
        let t = llvq_artifact::read_raw(&mut r, head.version).map_err(|e| e.to_string())?;
        carried_weights += t.len();
        raw.push(t);
    }

    let n_blob = read_u32(&mut r)?;
    let mut blobs: HashMap<String, Vec<u8>> = HashMap::new();
    for _ in 0..n_blob {
        let b = llvq_artifact::read_blob(&mut r).map_err(|e| e.to_string())?;
        blobs.insert(b.name, b.bytes);
    }

    Ok(FusedModel {
        layout,
        matrices,
        rotations,
        raw,
        config_json: blobs
            .remove("config.json")
            .ok_or_else(|| format!("{path} ne porte pas de config.json"))?,
        tokenizer_json: blobs
            .remove("tokenizer.json")
            .ok_or_else(|| format!("{path} ne porte pas de tokenizer.json"))?,
        quantized_weights,
        carried_weights,
        file_bytes,
        runtime_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The three tables must describe the transform `Rotation` computes.
    ///
    /// Not a restatement of `RotationTables::build`: it rebuilds `Q x` from
    /// the *tables alone*, the way a kernel would — sign bitmap, Walsh–
    /// Hadamard per group, dense block across groups — and requires the
    /// result to match. A packing bug (bit order, row-major vs column-major
    /// `small`, a wrong `inv`) shows up here and nowhere else on this machine.
    #[test]
    fn the_tables_describe_the_rotation() {
        for (n, seed) in [(240usize, 1u64), (2560, 2), (9728, 3), (4096, 4), (96, 5)] {
            let t = RotationTables::build(n, seed).expect("within KMAX");
            let rot = Rotation::new(n, seed);
            let mut rng = llvq_core::SplitMix64::new(seed ^ 0xf00d);
            let x: Vec<f64> = (0..n).map(|_| rng.next_gaussian()).collect();

            let mut want = x.clone();
            rot.apply(&mut want);

            // From the tables only.
            let mut s: Vec<f64> = x
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    if (t.signbits[i >> 5] >> (i & 31)) & 1 == 1 {
                        -v
                    } else {
                        v
                    }
                })
                .collect();
            let mut len = 1usize;
            while len < t.m {
                for p in 0..n / 2 {
                    let j = (p / len) * (len << 1) + (p % len);
                    let (a, b) = (s[j], s[j + len]);
                    s[j] = a + b;
                    s[j + len] = a - b;
                }
                len <<= 1;
            }
            let inv = t.inv as f64;
            let got: Vec<f64> = if t.k == 1 {
                s.iter().map(|v| v * inv).collect()
            } else {
                let mut out = vec![0.0f64; n];
                for j in 0..t.m {
                    for g in 0..t.k {
                        let mut acc = 0.0;
                        for c in 0..t.k {
                            acc += t.small[g * ROT_KMAX + c] as f64 * s[c * t.m + j] * inv;
                        }
                        out[g * t.m + j] = acc;
                    }
                }
                out
            };

            let nrm = want.iter().map(|v| v * v).sum::<f64>().sqrt();
            let dev = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();
            assert!(dev / nrm < 1e-6, "n={n} : écart relatif {:.3e}", dev / nrm);
        }
    }

    /// `inv` is `1/√m`, not `1/√n`. On a width with an odd factor the two
    /// differ by `√k` — a factor of 4.4 at `k = 19` — and every output would
    /// be uniformly wrong, which is exactly the kind of error a norm check
    /// catches and an eyeball does not.
    #[test]
    fn the_scale_is_over_the_power_of_two_factor() {
        let t = RotationTables::build(9728, 7).expect("within KMAX");
        assert_eq!((t.m, t.k), (512, 19));
        assert!((t.inv as f64 - 1.0 / 512f64.sqrt()).abs() < 1e-9);
        assert!((t.inv as f64 - 1.0 / 9728f64.sqrt()).abs() > 1e-3);
    }
}
