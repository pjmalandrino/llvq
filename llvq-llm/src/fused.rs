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

use llvq_artifact::CodeKind;
use llvq_artifact::runtime::{
    transcode, transcode_golay70, transcode_planes12x, transcode_planes14, ClassTable,
    Golay70Blocks, Golay70Table, Layout, Planes12xBlocks, PlanesBlocks, RuntimeBlocks,
    PLANES14_BYTES,
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
///
/// `Golay70` is the E2 layout, wired 2026-08-11 as step 5 of the v2 campaign
/// (`docs/archive/passation-golay70-2026-08-11.md`): the 9-byte main stream whose
/// per-slot decode goes through the Golay codeword rank, plus the exact
/// exception records — 3.589 b/weight at the bench, the only layout whose
/// whole-model b/param beats the deployed AWQ at every scale
/// (`docs/archive/projections-golay70-2026-08-11.md` §2). Its kernel is the v2
/// block-prologue decoder; whether it is SERVED is decided by the
/// pre-registered criterion of `proofs/preregistration-2026-08-11.md`, not
/// by this wiring — being selectable is what makes it measurable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FusedLayout {
    Planes14,
    Planes12x,
    Slot32,
    Golay70,
    /// The Tetra word, read as it is written. The only layout here that does
    /// **not** unfold a ball index into a kernel-shaped record — it addresses
    /// blocks row-strided rather than flat, because a six-byte record is not
    /// u32-aligned and a row must start on a boundary or every row after the
    /// first reads at a shifted phase (`llvq_artifact::tetra48`).
    Tetra48,
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
            Some("golay70") => Ok(Self::Golay70),
            Some("tetra48") => Ok(Self::Tetra48),
            Some(other) => Err(format!(
                "LLVQ_FUSED_LAYOUT={other}: accepted values \"planes14\" (default), \
                 \"planes12x\", \"slot32\", \"golay70\" and \"tetra48\""
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
            Self::Golay70 => "golay70",
            Self::Tetra48 => "tetra48",
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
                "LLVQ_EMBED={other}: accepted values \"f16\" (default) and \"q8\""
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
            "{}: int{} g{} carried by the file, the q8 path reads only int8 g{EMBED_GROUP}",
            t.name, q.bits, q.group
        )),
    }
}

/// Device bytes of the q8 embedding: `(packed, scales + biases)`.
///
/// Pure arithmetic on the dims, so the announced footprint is testable
/// without a card: on `[151936, 2560]` this is 388,956,160 + 24,309,760
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
    let embed = take(raw, EMBED_NAME).ok_or_else(|| format!("does not carry {EMBED_NAME}"))?;
    let embed = embed_q8(embed)?;
    let head = if tie {
        None
    } else {
        let t = take(raw, HEAD_NAME).ok_or_else(|| {
            format!(
                "tie_word_embeddings=false, but the file does not carry {HEAD_NAME}. The \
                 two ends are untied, and the q8 path never substitutes the embedding for \
                 a missing lm_head"
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
            format!("1 table ({}, lm_head tied on it)", names[0])
        } else {
            format!("{} tables ({})", names.len(), names.join(" + "))
        };
        match self.mode {
            EmbedMode::F16 => format!(
                "embedding: f16 (LLVQ_EMBED), {which}, {:.1} MB on the card",
                self.total() as f64 / 1e6
            ),
            EmbedMode::Q8 => format!(
                "embedding: q8 g64 (LLVQ_EMBED), {which}, {:.1} MB on the card \
                 (int8 {:.1} + scales/biases {:.1})",
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
const PLANES_SEG_H_CU_EMBED: &str = include_str!("../kernels/tv_planes_seg_h.cu");
const PLANES12_CUH_EMBED: &str = include_str!("../../llvq-cuda/kernels/llvq_planes12.cuh");
const PLANES12X_H_CU_EMBED: &str = include_str!("../kernels/tv_planes12x_h.cu");
const GOLAY_CUH_EMBED: &str = include_str!("../../llvq-cuda/kernels/llvq_golay.cuh");
const GOLAY70_H_CU_EMBED: &str = include_str!("../kernels/tv_golay70_h.cu");
// Tetra48's four. The first three come from `llvq-cuda` verbatim — the F1
// decoder chain the bench of 2026-09-08 measured and the served decode
// `tests/tetra48_matches_rust.rs` pins bit for bit — and the fourth is this
// crate's own half-storing entry point.
const F1RANK_CUH_EMBED: &str = include_str!("../../llvq-cuda/kernels/llvq_f1rank.cuh");
const F1RANK_V3_CUH_EMBED: &str = include_str!("../../llvq-cuda/kernels/llvq_f1rank_v3.cuh");
const TETRA48_CUH_EMBED: &str = include_str!("../../llvq-cuda/kernels/llvq_tetra48.cuh");
const TETRA48_H_CU_EMBED: &str = include_str!("../kernels/tv_tetra48_h.cu");
/// The int4 g128 projection kernel, for the mixed files `LLVQ_INT4_TYPES`
/// writes. Not a layout source: it is orthogonal to the lattice layout — a
/// mixed file carries int4 records **beside** Tetra or Ball ones — so it is
/// selected by the file's `KindSet` and not by `LLVQ_FUSED_LAYOUT`.
const Q4_H_CU_EMBED: &str = include_str!("../kernels/tv_q4_h.cu");

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
///
/// 🚨 **`tv_planes_seg_h.cu` is in all three lists unconditionally, and that
/// choice has a price the commit has to name.** The alternative — appending it
/// only under [`FuseMode::On`] — would give the two arms of a fusion A/B **two
/// different translation units**, while the fused arm still launches
/// `tv_planes_h` for `o_proj` and `down_proj`: it would inherit a possibly
/// different register allocation on those 72 launches, which no correctness
/// test can see. Including it always makes both arms share **one**
/// `tv_planes_h` so they differ by a launch count and nothing else. What it
/// costs: the served unit is no longer byte-identical to the one the
/// 2026-08-06 figures were measured on, so **the reference for this lot is the
/// `LLVQ_FUSE=0` arm of the same job**, never the published 5.079 ms.
pub fn planes_source_names(layout: FusedLayout) -> &'static [&'static str] {
    match layout {
        FusedLayout::Slot32 => &[],
        FusedLayout::Planes14 => &[
            "llvq_planes.cuh",
            "planes.cu",
            "tv_planes_h.cu",
            "tv_planes_seg_h.cu",
        ],
        FusedLayout::Planes12x => &[
            "llvq_planes.cuh",
            "planes.cu",
            "tv_planes_h.cu",
            "tv_planes_seg_h.cu",
            "llvq_planes12.cuh",
            "tv_planes12x_h.cu",
        ],
        // Planes12x's list plus two files, not a list of its own — the same
        // deliberate sharing of translation-unit shape as Planes12x itself:
        // the register report of `tv_planes_h` stays a drift detector on the
        // Golay70 build, and `llvq_golay.cuh` (the v2 decoder) finds every
        // guard of `llvq_planes12.cuh` already taken.
        FusedLayout::Golay70 => &[
            "llvq_planes.cuh",
            "planes.cu",
            "tv_planes_h.cu",
            "tv_planes_seg_h.cu",
            "llvq_planes12.cuh",
            "tv_planes12x_h.cu",
            "llvq_golay.cuh",
            "tv_golay70_h.cu",
        ],
        // A list of its own, and it shares nothing with the four above: Tetra
        // names no class, so `llvq_planes.cuh` and its ClassRec table have no
        // meaning here. What it needs is the F1 decoder chain the bench proved,
        // plus the served decode on top.
        FusedLayout::Tetra48 => &[
            "llvq_f1rank.cuh",
            "llvq_f1rank_v3.cuh",
            "llvq_tetra48.cuh",
            "tv_tetra48_h.cu",
        ],
    }
}

/// The lattice [`CodeKind`] a layout reads — the one it transcodes.
///
/// Not the only kind it can *serve*: see [`check_kinds`].
pub fn lattice_kind(layout: FusedLayout) -> CodeKind {
    match layout {
        FusedLayout::Planes14
        | FusedLayout::Planes12x
        | FusedLayout::Slot32
        | FusedLayout::Golay70 => CodeKind::Ball,
        FusedLayout::Tetra48 => CodeKind::Tetra,
    }
}

/// Whether `layout` can serve a record of `kind`.
///
/// Two kinds, and they arrive by different routes:
///
/// * the layout's own [`lattice_kind`], which its transcoder turns into a
///   kernel stream — and **only** that one. A Tetra word read as a ball index
///   names a class it does not have; a ball index read as a Tetra word decodes
///   to a different lattice point. Neither fails: both return plausible,
///   wrong weights, which is why `require_ball` exists and why this does not
///   soften it for the eleven sites that still use it;
/// * [`CodeKind::Int4G128`], which no layout transcodes at all. An int4 record
///   is stored group-affine in the natural basis — never GPTQ, never the
///   rotation, never the lattice (`calib.rs:793-812`) — and is served by
///   `tv_q4_h`, which is orthogonal to the layout. That is what makes a mixed
///   file possible: the two kinds do not share a code path, so they do not
///   have to share a layout.
pub fn serves_kind(layout: FusedLayout, kind: CodeKind) -> bool {
    kind == CodeKind::Int4G128 || kind == lattice_kind(layout)
}

/// [`serves_kind`] over a whole header set, refused by name.
///
/// This replaces `require_ball_kinds` on the fused path, and replacing it is
/// the point: that function was written when every layout read a ball index,
/// and it refuses `Tetra` and `Int4G128` unconditionally. It is still right
/// everywhere it is still called — a `ClassTable` consumer cannot read either.
/// Here the layout decides.
///
/// The early stop matters: the set is in the header, so a file this path
/// cannot serve is refused for the cost of a header rather than after 130
/// seconds of transcoding.
pub fn check_kinds(layout: FusedLayout, kinds: llvq_artifact::KindSet) -> Result<(), String> {
    for kind in kinds.iter() {
        if !serves_kind(layout, kind) {
            // The message names the layout that DOES read the kind, when one
            // exists. A refusal that only says no sends an operator to read
            // source; this one sends them to `LLVQ_FUSED_LAYOUT`.
            let go = match kind {
                CodeKind::Tetra => " Serve it with LLVQ_FUSED_LAYOUT=tetra48.",
                CodeKind::Ball => {
                    " Serve it with LLVQ_FUSED_LAYOUT=planes14 (or planes12x, slot32, golay70)."
                }
                CodeKind::Int4G128 => "",
            };
            return Err(format!(
                "LLVQ_FUSED_LAYOUT={}: this file declares a {kind:?} record, which {} \
                 cannot read — it transcodes {:?} and carries {:?} beside it, and any \
                 other kind names a different map that would decode to plausible, wrong \
                 weights.{go}",
                layout.name(),
                layout.name(),
                lattice_kind(layout),
                CodeKind::Int4G128,
            ));
        }
    }
    Ok(())
}

/// The four constant tables `tv_tetra48_h` reads, in the shapes it declares.
///
/// Built here rather than in `fused_cuda.rs` for the reason that module gives
/// about `golay70_gpu_class_table`: encoder and decoder must not drift, so the
/// tables come from the **same** `llvq_search::tetra::Tetra` the encoder used,
/// and they are built where a machine without a card can check them.
///
/// The two byte tables travel packed four to a `u32`, little-endian. The kernel
/// casts nothing — it declares `const unsigned char*` and reads byte `i` at
/// byte `i`, which on a little-endian device is what this packing put there.
/// `bin/f1rankfloor` uploads them the same way and its card-side control 1
/// pins the result against the Rust reference.
pub struct Tetra48Tables {
    /// 4,096 rank rows, two classes of 2,048. 16 KiB.
    pub rows: Vec<u32>,
    /// `prefixes[2·s8 + b1]`, 128 bytes packed into 32 u32.
    pub prefixes: Vec<u32>,
    /// `branches[16·s8 + b2]` = `byte | s16 << 8`, 1,024 u16.
    pub branches: Vec<u16>,
    /// `suffixes[2·s16 + b3]`, 128 bytes packed into 32 u32.
    pub suffixes: Vec<u32>,
    /// `1/√(16 m)` with entry 0 **zero** — the origin, reconstructed without a
    /// branch and without a division. `llvq_artifact::tetra48::TETRA48_SHELLS`
    /// entries; `m ≤ 27` on this codebook.
    pub invnorm: Vec<f32>,
}

pub fn tetra48_tables(t: &llvq_search::tetra::Tetra) -> Tetra48Tables {
    let pack = |b: &[u8]| -> Vec<u32> {
        b.chunks(4)
            .map(|c| {
                let mut w = [0u8; 4];
                w[..c.len()].copy_from_slice(c);
                u32::from_le_bytes(w)
            })
            .collect()
    };
    let mut prefixes = Vec::with_capacity(128);
    let mut suffixes = Vec::with_capacity(128);
    for s in 0..64usize {
        prefixes.extend_from_slice(&t.prefixes()[s]);
        suffixes.extend_from_slice(&t.suffixes()[s]);
    }
    let mut branches = Vec::with_capacity(1024);
    for s in 0..64usize {
        for b in 0..16usize {
            let (byte, s16) = t.branches()[s][b];
            branches.push(byte as u16 | (s16 as u16) << 8);
        }
    }
    let mut invnorm = vec![0.0f32; llvq_artifact::tetra48::TETRA48_SHELLS];
    for (m, e) in invnorm.iter_mut().enumerate().skip(1) {
        *e = (1.0f64 / ((16 * m) as f64).sqrt()) as f32;
    }
    Tetra48Tables {
        rows: t.rows().to_vec(),
        prefixes: pack(&prefixes),
        branches,
        suffixes: pack(&suffixes),
        invnorm,
    }
}

/// The int4 g128 projection source, for a file that carries `Int4G128` records.
///
/// It lives here rather than in `fused_cuda.rs` for the reason this whole
/// module is unconditional: `fused_cuda.rs` is `cfg(target_os = "linux",
/// feature = "cuda")` and the development machine is a Mac, so a source list
/// that lived there could not be reached by a test until it had already
/// reached a card. `tv_q4_h.cu` has **never run on a GPU** — it is verified as
/// host C++ by `tests/proj_q4.rs` and by nothing else — which makes it exactly
/// the file whose plumbing should be checkable without one.
///
/// `LLVQ_KERNEL_DIR` overrides it on the same all-or-nothing terms as
/// [`load_planes_sources`].
pub fn load_int4_sources() -> Result<(String, Option<String>), String> {
    match std::env::var("LLVQ_KERNEL_DIR") {
        Err(_) => Ok((Q4_H_CU_EMBED.to_string(), None)),
        Ok(dir) => {
            let p = std::path::Path::new(&dir).join("tv_q4_h.cu");
            let text = std::fs::read_to_string(&p)
                .map_err(|e| format!("LLVQ_KERNEL_DIR={dir}: tv_q4_h.cu: {e}"))?;
            Ok((text, Some(dir)))
        }
    }
}

/// The `extern "C"` entry point [`load_int4_sources`] defines.
pub const INT4_KERNEL_NAME: &str = "tv_q4_h";

/// The `extern "C"` matvec entry point `layout` launches.
pub fn matvec_kernel_name(layout: FusedLayout) -> &'static str {
    match layout {
        FusedLayout::Planes14 => "tv_planes_h",
        FusedLayout::Planes12x => "tv_planes12x_h",
        FusedLayout::Slot32 => "tv_slot_h",
        FusedLayout::Golay70 => "tv_golay70_h",
        FusedLayout::Tetra48 => "tv_tetra48_h",
    }
}

/// The `extern "C"` **segmented** matvec entry point, or `None` when `layout`
/// cannot be segmented.
///
/// A second entry point beside [`matvec_kernel_name`], never a replacement: a
/// fused build still launches `tv_planes_h` for the projections that stay alone
/// (`o_proj`, `down_proj`).
///
/// `Planes14` only, and that is not a scoping decision — the other three are
/// *wrong* under a row concatenation, silently:
///
///  * `Planes12x`'s overlay is indexed by the **local** row of a matrix
///    (`row_exc` has `d_out + 1` entries and `tv_planes12x_h` recovers an
///    exception's column as `b − row·nblocks`), so stacking rows moves every
///    exception to the wrong column and returns finite, plausible, wrong
///    numbers;
///  * `Golay70` inherits that machinery verbatim;
///  * `Slot32`'s stride is the widest record of a group of 32 blocks and its
///    bases table is per group, so a concatenation that regroups across a
///    segment boundary moves the byte total. `tv_slot_seg` exists at the bench
///    and wiring it is a separate lot with its own measurement.
pub fn seg_kernel_name(layout: FusedLayout) -> Option<&'static str> {
    match layout {
        FusedLayout::Planes14 => Some("tv_planes_seg_h"),
        _ => None,
    }
}

/// The PREFILL entry point a layout carries, if it has one.
///
/// `Some` exactly when the layout's unit defines a kernel that takes several
/// activation rows a launch. Only `Tetra48` does today, and the shape is
/// `seg_kernel_name`'s deliberately: an `Option` the runtime turns into an
/// authorisation, so no path can reach a kernel the unit never carried.
///
/// The one-row entry point of every layout is untouched and stays the decode
/// path — this is the second kernel every quantized-inference stack carries,
/// not a replacement for the first.
pub fn rows_kernel_name(layout: FusedLayout) -> Option<&'static str> {
    match layout {
        FusedLayout::Tetra48 => Some("tv_tetra48_rows_h"),
        _ => None,
    }
}

/// Whether the projections that share an activation are launched as one
/// row-concatenated matrix or one at a time.
///
/// The `Off` arm is not a fallback: it issues **today's launches, launch for
/// launch** — 252 a token on the published 4B against 144 — so the two arms of
/// a card measurement differ by a count and nothing else.
///
/// **The default is `Off`** until the card gate is green; the commit that flips
/// it carries the measurement in its message, exactly the [`crate::rotplan::
/// RotShare`] discipline. ⚠️ It also *cannot* be `On` today without a second
/// variable being set: [`check_fuse`] refuses `On` beside `RotShare::Off`, and
/// `RotShare`'s own default is `Off`, so a default of `On` here would make
/// every `fused_cuda::load` fail with no environment set at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FuseMode {
    /// One matvec per projection.
    Off,
    /// One matvec per shared activation.
    On,
}

impl FuseMode {
    /// Parse `LLVQ_FUSE`. Same contract as [`FusedLayout::parse`] and
    /// [`crate::rotplan::RotShare::parse`]: unset and empty mean the default,
    /// anything else must name a mode exactly — a typo quietly falling back to
    /// the default is how an A/B reports "no effect" for an arm that never ran.
    pub fn parse(v: Option<&str>) -> Result<Self, String> {
        match v {
            None | Some("") | Some("0") => Ok(Self::Off),
            Some("1") => Ok(Self::On),
            Some(other) => Err(format!(
                "LLVQ_FUSE={other}: accepted values \"0\" (default, one matvec per \
                 projection) and \"1\" (one matvec per shared activation)"
            )),
        }
    }

    /// Resolve from the environment.
    pub fn from_env() -> Result<Self, String> {
        let v = std::env::var("LLVQ_FUSE").ok();
        Self::parse(v.as_deref())
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Off => "0",
            Self::On => "1",
        }
    }
}

/// Whether `fuse` can be honoured beside `layout` and `share`.
///
/// Two refusals, and the second is about the *measurement* rather than about
/// correctness:
///
///  * a layout [`seg_kernel_name`] does not name cannot be segmented — see
///    there for the mechanism, which is silent in all three cases;
///  * `FuseMode::On` with `RotShare::Off` is refused, because a fused group is
///    **one** site: it rotates once per row whatever the mode says. Accepting
///    the pair would let an `LLVQ_FUSE` A/B move the hoist as well, and the
///    delta would then be two mechanisms added together — the confounder this
///    dossier spends its time forbidding. The only way to keep one variable is
///    to make `LLVQ_ROT_SHARE=1` mandatory on **both** arms of the fusion A/B.
pub fn check_fuse(
    layout: FusedLayout,
    share: crate::rotplan::RotShare,
    fuse: FuseMode,
) -> Result<(), String> {
    if fuse == FuseMode::Off {
        return Ok(());
    }
    if seg_kernel_name(layout).is_none() {
        return Err(format!(
            "LLVQ_FUSE=1 with LLVQ_FUSED_LAYOUT={}: only planes14 segments. The \
             exceptions of planes12x and golay70 are indexed by the LOCAL row of their \
             matrix, and slot32 carries one stride per group of 32 blocks. A row \
             concatenation there would return finite, plausible, wrong numbers.",
            layout.name()
        ));
    }
    if share == crate::rotplan::RotShare::Off {
        return Err(
            "LLVQ_FUSE=1 with LLVQ_ROT_SHARE=0: a fused group is ONE site, so it rotates \
             once per row whatever the mode says. Accepting the pair would move two \
             mechanisms inside one A/B. Set LLVQ_ROT_SHARE=1 on BOTH arms."
                .into(),
        );
    }
    Ok(())
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
        "tv_planes_seg_h.cu" => Ok(PLANES_SEG_H_CU_EMBED),
        "llvq_planes12.cuh" => Ok(PLANES12_CUH_EMBED),
        "tv_planes12x_h.cu" => Ok(PLANES12X_H_CU_EMBED),
        "llvq_golay.cuh" => Ok(GOLAY_CUH_EMBED),
        "tv_golay70_h.cu" => Ok(GOLAY70_H_CU_EMBED),
        "llvq_f1rank.cuh" => Ok(F1RANK_CUH_EMBED),
        "llvq_f1rank_v3.cuh" => Ok(F1RANK_V3_CUH_EMBED),
        "llvq_tetra48.cuh" => Ok(TETRA48_CUH_EMBED),
        "tv_tetra48_h.cu" => Ok(TETRA48_H_CU_EMBED),
        other => Err(format!("no embedded copy of {other}")),
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
    Golay70 {
        /// The 9-byte main-stream records as `u32` words. The transcoder
        /// itself word-aligns the stream and appends one zero word so the
        /// kernel's three-word window exists for the last block — no extra
        /// packing pad here, unlike the plane streams.
        words: Vec<u32>,
        /// Matrix-wide block index of each exception, strictly ascending.
        exc_idx: Vec<u32>,
        /// One exact 14-byte `Planes14` record per exception, as `u32` words,
        /// padded for the four-word window `planes_fields` reads on it —
        /// the exception machinery is Planes12x's, verbatim.
        exc_words: Vec<u32>,
        /// `d_out + 1` prefix offsets into `exc_idx` — same contract, same
        /// [`row_offsets`] check, same reason as the `Planes12x` field: one
        /// warp per row, no memset, no atomic (`kernels/tv_golay70_h.cu`).
        row_exc: Vec<u32>,
    },
    Tetra48 {
        /// The 48-bit words as `u32`, **row-strided**: `stride_u32` words per
        /// output row, the row's six-byte records at its start and the rest
        /// zero. Every other stream here is flat over the matrix's blocks;
        /// this one is not, and the reason is alignment rather than taste —
        /// a six-byte record at `6·b` is u32-aligned only every second block,
        /// so `f1r_load`'s two-word window would read at a shifted phase on
        /// every row after the first. `llvq_artifact::tetra48` builds the
        /// stride, rounds it to eight bytes so the last block's window fits,
        /// and asserts that rather than assuming it.
        words: Vec<u32>,
        /// Words per row. The kernel's second argument; a stream whose stride
        /// disagrees with it returns finite, plausible, wrong numbers.
        stride_u32: u32,
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
    /// `d_out × tail_w`, row-major, in the rotated basis, as **binary16
    /// bits** — see [`tail_f16_bits`] for why the width changed and what it
    /// is aligned to.
    pub tail: Vec<u16>,
    /// Key into [`FusedModel::rotations`], or `None` in the natural basis.
    pub rotation: Option<RotKey>,
    /// Bytes this matrix costs at runtime — every device array the kernel
    /// reads: the payload, whatever addressing the layout carries beside it
    /// (`Slot32`'s bases, `Planes12x`'s exception table and row offsets), and
    /// [`matrix_side_bytes`] for the tail and the row scales.
    pub bytes: u64,
}

/// One `Int4G128` projection, uploaded as it is stored.
///
/// Deliberately **not** a variant of [`FusedMatrix`]. That struct carries a
/// `gscale`, a `rscale`, a `tail` and a `rotation`, and an int4 record has
/// none of the four: it is stored group-affine in the natural basis, never
/// GPTQ, never the rotation, never the lattice (`calib.rs:793-812`). Folding
/// it in would mean four `Option`s on the hot struct and a reader who cannot
/// tell which combinations are legal. A separate list says it once.
///
/// The weights are the file's own bytes. Nothing here decodes, rescales or
/// reorders them — `tv_q4_h` reads **`scale · q + bias`** off exactly these
/// three arrays, and `tests/proj_q4.rs` pins that arithmetic against
/// [`llvq_artifact::Int4Matrix::to_f32`] on the development machine.
///
/// 🕳️ This said `scale · (q − zero)` until 2026-09-10, which is the OTHER
/// affine convention — the one where the offset is subtracted from the
/// quantum before scaling. `embedquant` stores `bias = min` and the kernel
/// spells the two roundings out (`__fmul_rn` then `__fadd_rn`, never
/// contracted) precisely so a served row equals the row the file decodes to.
/// A doc naming the wrong convention is what a second implementation would
/// have been written against.
#[derive(Clone)]
pub struct FusedInt4 {
    pub name: String,
    pub d_out: usize,
    pub d_in: usize,
    /// Groups along `d_in`; `d_in / group` per output row.
    pub group: usize,
    /// `d_out · d_in / 2` bytes, row-major, **low nibble first**. A lattice
    /// index in the same file is packed MSB-first: two orders in one file, and
    /// a reader that assumed one convention for both would produce plausible,
    /// wrong weights.
    pub packed: Vec<u8>,
    /// One binary16 per group, `d_out × (d_in / group)`, row-major.
    pub scales: Vec<u16>,
    /// Same shape, same order.
    pub biases: Vec<u16>,
    /// Bytes this matrix costs at runtime: the three arrays, and nothing else.
    /// There is no tail and no row scale to add.
    pub bytes: u64,
}

/// Bytes one `TailPolicy::KeepExact` weight costs on the device.
///
/// Two since 2026-08-09 (lot A7a), four before it. Named rather than spelled
/// so [`matrix_side_bytes`] and the tail upload cannot drift apart, and so a
/// mutation of the width has exactly one site.
pub const TAIL_BYTES: u64 = 2;

/// Bytes one row scale costs on the device. Still `f32`, and deliberately:
/// `rscale[row]` multiplies the whole accumulated block sum, so its rounding
/// is a *relative* error on every quantized weight of the row at once, where
/// the tail's is an absolute error on `d_in mod 24` of them. Different
/// exposure, different decision — and 1,105,920 rows on the 4B is 4.4 MB, a
/// tenth of what the tail was costing.
pub const ROW_SCALE_BYTES: u64 = 4;

/// Narrow a `KeepExact` tail to the precision the **reference** model holds it
/// at, and return its binary16 bits.
///
/// ## The only question that mattered here, and its answer in the code
///
/// The tail is full precision *by design*: `TailPolicy::KeepExact` leaves the
/// `d_in mod 24` unquantized columns alone so that they never contribute an
/// error of their own. Rounding them looks, at first sight, like spending
/// quality for bytes — which this dossier has refused for much better trades.
/// It is not, and the reason is that the arm the fused path is *compared to*
/// already rounds them:
///
/// * `bin/fusedrun` runs both arms at `DType::F16` (`bin/fusedrun.rs`, the
///   `let dtype` beside the device);
/// * its dense arm is `crate::sealed::load(path, dtype, device)`, which takes
///   `llvq_artifact::decode_matrix`'s `Vec<f32>` — tail columns restored into
///   it, then un-rotated — and calls `.to_dtype(dtype)` on the whole matrix
///   (`crate::sealed`). Nothing exempts the tail: it is a column range of a
///   tensor that gets narrowed to binary16 in one move;
/// * `bin/mmlu` defaults to F16 and `bin/ppl` prints the dtype it used, and
///   **both score through `sealed::load`** — the fused path is a matvec and
///   `bin/ppl` keeps the dense path for scoring. So no published perplexity
///   or MMLU figure reads this array at all; what it can move is
///   `bin/fusedrun`'s token agreement, and nothing else.
///
/// The `f32` that was here before was never a precision decision for the
/// card. It is the *file's* storage width — `calib.rs` narrows the tail to
/// `f32` before writing it, precisely so the file and the evaluated model are
/// one object, and `llvq_artifact::format` reads it back from 32 bits — and
/// the loader carried that width forward because it had no reason not to.
/// Keeping 24 mantissa bits resident bought precision the reference does not
/// have, at 2 bytes a weight.
///
/// ## Size of the rounding, against the rounding already there
///
/// Half an ulp of binary16 is `2⁻¹¹ ≈ 4.9e-4` relative. If tail and block
/// columns carry comparable energy, the tail's share of a row's dot product
/// scales as `√(tail_w / d_in)`, so the perturbation on `y` is about
/// `4.9e-4 · √(tail_w/d_in)`: **7.9 %** of it on the 4B's `d_in = 2560`
/// (tail 16), **2.9 %** on its `down_proj` (`d_in = 9728`, tail 8). The
/// kernel then narrows `y` itself to binary16 — a `4.9e-4` relative rounding,
/// i.e. 12× to 34× *larger*. ⚠️ Labelled honestly: that is an **estimate**
/// under an energy assumption, not a measurement, and the only thing that
/// settles it is a card diffing `fusedrun`'s two arms.
///
/// ## Why the double rounding is not one
///
/// `v` comes from the file, where the tail is stored as 32 bits and widened
/// to `f64` on read, so `v as f32` is exact and `f64 → f32 → f16` rounds
/// once. That is asserted by `narrowing_the_tail_rounds_once`.
pub fn tail_f16_bits(tail: &[f64]) -> Vec<u16> {
    tail.iter()
        .map(|&v| half::f16::from_f32(v as f32).to_bits())
        .collect()
}

/// Device bytes a matrix costs **beside** its block stream: the narrowed tail
/// and the row scales.
///
/// Extracted from `load` so the accounting is a function with a name and a
/// test rather than an expression inside a loop — `runtime_bytes` is the
/// number `fusedrun` prints as "GB on the card" and that
/// [`FusedModel::runtime_bits_per_weight`] divides, so an arithmetic slip
/// here is published, not caught.
/// Device bytes the quantized projections occupy — every list the model holds.
///
/// A function with a name and a test rather than an expression inside `load`,
/// for the reason [`matrix_side_bytes`] gives: this is the number `fusedrun`
/// prints as "GB on the card" and the numerator
/// [`FusedModel::runtime_bits_per_weight`] divides, so a term missing here is
/// published rather than caught.
///
/// 🕳️ The `int4` term WAS missing, and on one side of that division only:
/// `quantized_weights` counts an int4 record's weights while this sum did not
/// count its bytes. On the served mixed file that is 2.030 b/weight printed
/// against a true 2.141, and a footprint 50 MB light.
pub fn runtime_bytes_of(
    matrices: &[FusedMatrix],
    groups: &[FusedGroup],
    int4: &[FusedInt4],
) -> u64 {
    matrices.iter().map(|m| m.bytes).sum::<u64>()
        + groups.iter().map(|g| g.bytes).sum::<u64>()
        + int4.iter().map(|q| q.bytes).sum::<u64>()
}

pub fn matrix_side_bytes(d_out: usize, tail_w: usize) -> u64 {
    (d_out * tail_w) as u64 * TAIL_BYTES + d_out as u64 * ROW_SCALE_BYTES
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
                "width {n}: odd factor {k} above KMAX={ROT_KMAX}. The rotation kernel does \
                 not handle it, see llvq_rot.cuh."
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

// ---------------------------------------------------------------------------
// Segmentation — the row-concatenation one fused launch reads
// ---------------------------------------------------------------------------
//
// The twin of `llvq-cuda/src/seg_host.rs`, which does the same job for the
// bench. The two are **deliberately not shared**: `seg_host.rs` concatenates
// raw indices *before* transcoding, carries an `f32` tail, and targets the f32
// bench kernel, while the served path holds streams already packed into `u32`
// words and a binary16 tail. Making one file serve both would mean reworking
// the object that pins the bench, inside a lot whose subject is not the bench.
// The price is paid in tests: `llvq-llm/tests/fused_segment.rs` has to be as
// lethal as `llvq-cuda/tests/planes_segment_matches_unfused.rs`, not lighter.

/// `u32` words a `Planes14` payload of `n_blocks` occupies, **pad excluded**.
///
/// `14·n ≡ 2n (mod 4)`, so this is a whole number of words exactly when `n` is
/// even — which the served path guarantees, since `upload_matrix` refuses
/// `d_out % 8 != 0` and `n = d_out · nblocks`. Returned as an `Option` rather
/// than assumed: the day a ragged `d_out` reaches here, this is where it stops.
pub fn planes_payload_words(n_blocks: usize) -> Option<usize> {
    n_blocks
        .is_multiple_of(2)
        .then(|| PLANES14_BYTES * n_blocks / 4)
}

/// Splice the packed `Planes14` streams of a group into the single stream the
/// segmented kernel indexes. `parts` is `(stream, n_blocks)` **in row order**.
///
/// ## Why this cannot be a `concat()`
///
/// [`pack_plane_bytes`] appends four bytes and then word-aligns, so every
/// `HostStream::Planes14` carries **one trailing pad word**. Concatenating two
/// of them buries that word in the middle of the stream: the kernel computes
/// `byte = 14·b` from the base of the buffer, so every block of the second
/// segment would be read four bytes early, with `sh = (byte & 3)·8` on the
/// wrong parity. The class id, the gain, the sign mask and the three planes all
/// come out of a shifted bit field — a valid class, a point in the ball, a
/// model that runs.
///
/// And nothing crashes: with two pads the buffer is *longer* than the last
/// block's four-word window needs, so there is no illegal address to catch it
/// in the middle of a billed job. Finite, plausible, wrong.
///
/// ## Why the concatenation is correct at all
///
/// `transcode_planes14` opens a fresh `BitSink` at `bit0 = b·14·8` for every
/// block and carries **no inter-block state**: record `b` depends only on
/// `indices[b]`, `gains[b]` and the `ClassTable`. So the transcode of a
/// concatenation is byte for byte the concatenation of the transcodes, and the
/// block of local row `r` of segment `s` carries the global number
/// `(row0_s + r)·nblocks + p` — exactly what the kernel computes.
///
/// The length check below is therefore the whole safety of this function, and
/// it is written as an equality rather than as `len() - 1`: it also catches the
/// day a stream arrives whose block count is odd, where `pack_plane_bytes`
/// emits a different number of pad bytes.
pub fn splice_planes14(parts: &[(&HostStream, usize)]) -> Result<Vec<u32>, String> {
    let total: usize = parts.iter().map(|&(_, n)| n).sum();
    let want = planes_payload_words(total)
        .ok_or_else(|| format!("{total} blocks in total, an odd count"))?;
    let mut out = Vec::with_capacity(want + 1);
    for (i, &(s, n)) in parts.iter().enumerate() {
        let HostStream::Planes14 { words } = s else {
            return Err(format!("segment {i}: stream that is not Planes14"));
        };
        let pw = planes_payload_words(n).ok_or_else(|| {
            format!(
                "segment {i}: {n} blocks, an odd count, so 14·n is not a multiple of 4 and \
                 the boundary does not fall on a word"
            )
        })?;
        if words.len() != pw + 1 {
            return Err(format!(
                "segment {i}: {} words for {n} blocks, {} expected (payload {pw} + one \
                 end-of-stream padding word)",
                words.len(),
                pw + 1
            ));
        }
        // Payload ONLY, for every segment including the last.
        out.extend_from_slice(&words[..pw]);
    }
    // One single pad, for the four-word window of the LAST block — exactly what
    // `pack_plane_bytes` would have emitted over the concatenated payload.
    out.push(0);
    debug_assert_eq!(out.len(), want + 1);
    Ok(out)
}

/// One projection's identity and place inside a fused group.
pub struct SegPart {
    /// The artifact's fully qualified name — what an error message must name.
    pub name: String,
    pub layer: usize,
    /// Checkpoint suffix, e.g. `self_attn.q_proj` — the key `Block::proj` uses.
    pub proj: String,
    pub d_out: usize,
    /// First row of this projection inside the group.
    pub row0: usize,
    /// Rank inside the family, i.e. its index in [`crate::model::Act::consumers`].
    /// This *is* the row order, and `model::group_forward` re-checks it
    /// structurally rather than trusting it: two places have to agree, so
    /// neither may assume.
    pub rank: usize,
}

/// A row-concatenation of the projections that share one activation — what one
/// segmented launch needs, and nothing else.
///
/// `stream` is a `HostStream::Planes14` by construction ([`splice_planes14`]
/// refuses anything else), so no code path can hand the segmented kernel a
/// bases array or an exception table.
pub struct FusedGroup {
    /// `"{layer:03}.{act:?}"` — the grouping key, for logs and errors.
    pub key: String,
    /// The **total**: Σ of the parts.
    pub d_out: usize,
    pub d_in: usize,
    pub nblocks: usize,
    pub tail_w: usize,
    pub stream: HostStream,
    /// The concatenated `gscale`: two floats per part, in row order.
    pub gscale: Vec<f32>,
    /// `d_out` entries. `gs_off[row] = 2 · rank`, the index of the first of the
    /// pair that row's part owns — the one thing that does **not** concatenate,
    /// because a gain centroid belongs to a matrix while the gain bit belongs
    /// to a block.
    pub gs_off: Vec<u32>,
    pub rscale: Vec<f32>,
    pub tail: Vec<u16>,
    pub rotation: Option<RotKey>,
    pub parts: Vec<SegPart>,
    /// Device bytes: the payload, [`matrix_side_bytes`] for the tail and the
    /// row scales, **and `gs_off`** — 4 bytes a fused row, which no other
    /// layout pays. On the published 4B that last term is the whole difference
    /// between the two arms: 36 layers × 25,600 fused rows × 4 = 3,686,400
    /// bytes, i.e. +0.0081 b/weight. Counted here rather than dropped, because
    /// `runtime_bytes` is what `fusedrun` prints as "GB on the card".
    pub bytes: u64,
}

/// Check that the spans handed to one segmented launch are the whole group, in
/// row order. `spans` is `(rank, row0, d_out)` **in the caller's order**.
///
/// Lives here, beside the loader that assigns the row order, rather than inside
/// `model::group_forward` where it is called from: that function's segmented
/// branch exists on no target this workspace's development machine builds, and
/// this is the half of risk R3 a test can reach. Two places have to agree on
/// the row order and neither may assume — a group read in the order k,q,v would
/// otherwise return finite, plausible, wrong numbers with no assertion
/// anywhere.
pub fn check_seg_spans(
    key: &str,
    spans: &[(usize, usize, usize)],
    group_d_out: usize,
) -> Result<(), String> {
    if spans.is_empty() {
        return Err(format!("{key}: segmented group with no part at all"));
    }
    let mut at = 0usize;
    for (i, &(rank, row0, d_out)) in spans.iter().enumerate() {
        if rank != i {
            return Err(format!(
                "{key}: the projection at position {i} carries rank {rank}. The row order \
                 of the group is the one of Act::consumers(), and the split of the output \
                 depends on it."
            ));
        }
        if row0 != at {
            return Err(format!(
                "{key}: part {i} starts at row {row0}, {at} expected. The parts do not \
                 tile the group."
            ));
        }
        at += d_out;
    }
    if at != group_d_out {
        return Err(format!(
            "{key}: the parts cover {at} rows of the fused group's {group_d_out}"
        ));
    }
    Ok(())
}

/// One projection on its way into a group: its rank in `Act::consumers()`, the
/// checkpoint suffix that gave the rank, and the matrix itself.
type PendingPart = (usize, String, FusedMatrix);

/// Split a loaded model's matrices into fused groups and the projections that
/// stay alone (`o_proj`, `down_proj`).
///
/// **Consumes** the matrices: a group's parts are spliced into one stream, and
/// keeping the unfused copies alive would double the host's peak — 1.37 GB on
/// the published 4B — for nothing.
///
/// ## The grouping key is `(layer, Act)`, and never the name
///
/// `llvq-cuda/src/seg_host.rs::seg_family` matches `name.contains("q_proj")`
/// and is right to — its fixtures carry no rotation, so the name is the only
/// structure available. Here the artifact carries rotation keys, and
/// [`crate::rotplan::act_of_suffix`] states the rule outright: the name is a
/// **parallel channel** that can disagree with the file. So the family comes
/// from that function (which reads `Act::consumers()`, the model's own table)
/// and the rank is the position in that same slice — q,k,v then gate,up, the
/// order `seg_family` also produces, from one derivation instead of two. The
/// layer is part of the key because q/k/v of two different layers share a
/// `d_in` and would concatenate perfectly well into nonsense.
///
/// The rotation keys are re-asserted equal across the group even though
/// [`crate::rotplan::check_rotation_partition`] already ran, because this
/// function owns the row order and must not inherit a premise it cannot see.
pub fn segment_matrices(
    matrices: Vec<FusedMatrix>,
) -> Result<(Vec<FusedMatrix>, Vec<FusedGroup>), String> {
    use crate::model::Act;

    let mut singles: Vec<FusedMatrix> = Vec::new();
    // First-appearance order, so logs and errors follow the file rather than a
    // hash seed — `rotation_sites` does the same, for the same reason.
    let mut order: Vec<(usize, Act)> = Vec::new();
    let mut buckets: HashMap<(usize, Act), Vec<PendingPart>> = HashMap::new();

    for m in matrices {
        let (layer, suffix) = llvq_artifact::split_name(&m.name).map_err(|e| e.to_string())?;
        let act = crate::rotplan::act_of_suffix(&suffix).ok_or_else(|| {
            format!(
                "{}: \"{suffix}\" consumes none of the four activations of a Qwen3 block",
                m.name
            )
        })?;
        let consumers = act.consumers();
        if consumers.len() < 2 {
            // `o_proj` and `down_proj`: nothing else consumes their activation,
            // so there is no group to be part of.
            singles.push(m);
            continue;
        }
        let rank = consumers
            .iter()
            .position(|&c| c == suffix)
            .expect("act_of_suffix found this suffix in this very slice");
        let key = (layer, act);
        if !buckets.contains_key(&key) {
            order.push(key);
        }
        buckets.entry(key).or_default().push((rank, suffix, m));
    }

    let mut groups = Vec::with_capacity(order.len());
    for (layer, act) in order {
        let mut parts = buckets
            .remove(&(layer, act))
            .expect("every key pushed on `order` was inserted in the same breath");
        parts.sort_by_key(|&(rank, ..)| rank);
        groups.push(build_group(layer, act, parts)?);
    }
    Ok((singles, groups))
}

/// One group, from its parts already sorted by rank.
fn build_group(
    layer: usize,
    act: crate::model::Act,
    parts: Vec<PendingPart>,
) -> Result<FusedGroup, String> {
    let key = format!("{layer:03}.{act:?}");
    let want = act.consumers().len();
    if parts.len() != want {
        // An incomplete group is a broken file, never a degraded case: the
        // missing rows would silently shrink the fused matrix.
        return Err(format!(
            "{key}: {} projections out of the {want} this activation feeds ({})",
            parts.len(),
            act.consumers().join(", ")
        ));
    }
    let (d_in, nblocks, tail_w, rotation) = {
        let m = &parts[0].2;
        (m.d_in, m.nblocks, m.tail_w, m.rotation)
    };
    if rotation.is_none() {
        return Err(format!(
            "{key}: {} has no rotation. The fused path cannot read a matrix quantized in \
             the natural basis.",
            parts[0].2.name
        ));
    }

    // The spliced stream is built first, from borrows, so the parts can then be
    // consumed field by field without cloning 1.37 GB of payload.
    let words = {
        let streams: Vec<(&HostStream, usize)> = parts
            .iter()
            .map(|(_, _, m)| (&m.stream, m.d_out * nblocks))
            .collect();
        splice_planes14(&streams).map_err(|e| format!("{key}: {e}"))?
    };

    let mut d_out = 0usize;
    let mut payload = 0u64;
    let mut gscale: Vec<f32> = Vec::with_capacity(2 * want);
    let mut gs_off: Vec<u32> = Vec::new();
    let mut rscale: Vec<f32> = Vec::new();
    let mut tail: Vec<u16> = Vec::new();
    let mut seg_parts: Vec<SegPart> = Vec::with_capacity(want);

    for (i, (rank, suffix, m)) in parts.into_iter().enumerate() {
        if rank != i {
            return Err(format!(
                "{key}: {} carries rank {rank} at position {i}, so two projections share a \
                 rank, or a consumer is duplicated",
                m.name
            ));
        }
        if m.d_in != d_in {
            return Err(format!(
                "{key}: {} has d_in={}, the group has {d_in}. A group shares its \
                 activation, hence its block count and its tail.",
                m.name, m.d_in
            ));
        }
        if m.nblocks != nblocks || m.tail_w != tail_w {
            return Err(format!(
                "{key}: {} splits into {} blocks and a tail of {}, against {nblocks} and \
                 {tail_w} for the group",
                m.name, m.nblocks, m.tail_w
            ));
        }
        if m.rotation != rotation {
            return Err(format!(
                "{key}: {} carries rotation {:?}, {rotation:?} for the group. Hoisting \
                 would give one the basis of the other.",
                m.name, m.rotation
            ));
        }
        // Per part **and** on the total below. The kernels launch whole
        // 256-thread blocks of 8 rows and carry no bounds guard; asserting the
        // total alone would let an individually ragged part through on a lucky
        // sum, and then the *unfused* control arm could not be launched at all.
        if !m.d_out.is_multiple_of(8) {
            return Err(format!(
                "{key}: {} has d_out={}, which is not a multiple of 8",
                m.name, m.d_out
            ));
        }
        if m.rscale.len() != m.d_out || m.tail.len() != m.d_out * tail_w {
            return Err(format!(
                "{key}: {} carries {} row scales and {} tail values for {} \
                 rows of {tail_w}",
                m.name,
                m.rscale.len(),
                m.tail.len(),
                m.d_out
            ));
        }
        // The payload is 14 bytes a block whatever the grouping — structural
        // for `Planes14`, uniform stride and no bases. Cross-checked against
        // what `load` billed for this very matrix, so a drift between the two
        // accountings stops here rather than in a published b/weight.
        let part_payload = (PLANES14_BYTES * m.d_out * nblocks) as u64;
        if m.bytes != part_payload + matrix_side_bytes(m.d_out, tail_w) {
            return Err(format!(
                "{key}: {} bills {} bytes, {} expected (payload {part_payload} + \
                 tail and scales)",
                m.name,
                m.bytes,
                part_payload + matrix_side_bytes(m.d_out, tail_w)
            ));
        }

        seg_parts.push(SegPart {
            name: m.name,
            layer,
            proj: suffix,
            d_out: m.d_out,
            row0: d_out,
            rank,
        });
        gscale.extend_from_slice(&m.gscale);
        // Two centroids per part, so part `rank`'s pair starts at `2·rank`; one
        // entry per ROW, because the kernel reads it as `gs_off[row]`.
        gs_off.extend(std::iter::repeat_n(2 * rank as u32, m.d_out));
        rscale.extend_from_slice(&m.rscale);
        tail.extend_from_slice(&m.tail);
        payload += part_payload;
        d_out += m.d_out;
    }

    if !d_out.is_multiple_of(8) {
        return Err(format!(
            "{key}: d_out={d_out} in total, which is not a multiple of 8"
        ));
    }
    if gs_off.len() != d_out
        || rscale.len() != d_out
        || tail.len() != d_out * tail_w
        || gscale.len() != 2 * seg_parts.len()
    {
        return Err(format!(
            "{key}: inconsistent concatenation, {} gs_off, {} scales, {} tail values, \
             {} centroids for {d_out} rows of {tail_w} and {} parts",
            gs_off.len(),
            rscale.len(),
            tail.len(),
            gscale.len(),
            seg_parts.len()
        ));
    }

    Ok(FusedGroup {
        key,
        d_out,
        d_in,
        nblocks,
        tail_w,
        stream: HostStream::Planes14 { words },
        gscale,
        gs_off,
        rscale,
        tail,
        rotation,
        parts: seg_parts,
        // The payload does not move — 14 bytes a block, grouping or not — and
        // neither does `matrix_side_bytes`, which is additive in `d_out` at a
        // shared `tail_w`. What fusion adds is exactly `4 · d_out`.
        bytes: payload + matrix_side_bytes(d_out, tail_w) + d_out as u64 * 4,
    })
}

/// Everything a fused runtime needs, still on the host.
pub struct FusedModel {
    /// The runtime layout every matrix was transcoded to.
    pub layout: FusedLayout,
    /// The projections launched one at a time. **All** of them under
    /// [`FuseMode::Off`]; only `o_proj` and `down_proj` under `On`, the rest
    /// having moved into [`Self::groups`].
    pub matrices: Vec<FusedMatrix>,
    /// The row-concatenated groups, empty under [`FuseMode::Off`] — where this
    /// struct is then byte-identical to what shipped before this lot.
    pub groups: Vec<FusedGroup>,
    /// The `Int4G128` projections, empty for every file written before
    /// `LLVQ_INT4_TYPES` existed — which is every published artifact. They sit
    /// beside the lattice matrices rather than among them because they share
    /// no code path with them: a different kernel, a different basis, a
    /// different packing order.
    pub int4: Vec<FusedInt4>,
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
    /// 🚨 **This function no longer reproduces those three bench numbers, and
    /// the gap is a real difference of objects, not a drift.** Since lot A7a
    /// the inference path holds the `KeepExact` tail as binary16 while every
    /// bench arm (`planesbench`, and `rtbits`'s "b/weight kernel" column which
    /// models it) still uploads `f32`. The whole difference is
    /// `16 · tail_weights / weights`: **−0.0747 b/weight on the published 4B**
    /// (16,957,440 tail weights of 3,633,315,840) and **−0.0462 on the 8B**
    /// (20,054,016 of 6,945,767,424), so `Planes14` reads 4.729 here against
    /// the bench's 4.804 at 4B, and 4.706 against 4.752 at 8B. Two
    /// accountings of two different residencies — never subtract one from the
    /// other and call it a measurement.
    ///
    /// ⚠️ This crate's `Planes12x` figure additionally sits marginally *above*
    /// the bench's for an unrelated reason: it also bills the `d_out + 1`
    /// row-offset table the inference kernel reads, which the bench arm has no
    /// equivalent of (32/`d_in` b/weight — 0.0125 at `d_in = 2560`).
    /// Under-reporting a device array is exactly the accounting sin the K-1
    /// lot spent a dossier removing.
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

/// A byte stream the transcoder already word-aligned and padded — the
/// Golay70 main stream — as `u32` words, verbatim. `pack_plane_bytes` would
/// add a second pad; harmless zeros, but the byte accounting then no longer
/// reads off the vector's length.
fn pack_aligned_bytes(bytes: &[u8]) -> Vec<u32> {
    assert!(bytes.len().is_multiple_of(4), "the transcoder aligns its stream");
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// `u32` words per entry of the Golay70 GPU class table — the kernel's
/// `GolayClassRec { float v[4]; u32 flags; u32 is_odd; u32 pad0, pad1; }`.
pub const GOLAY70_REC_WORDS: usize = 8;
/// Entries of that table — 512 so the 9-bit class field cannot index out of
/// bounds even from a corrupt stream (the `ClassRec` reasoning, unchanged).
pub const GOLAY70_TABLE_ENTRIES: usize = 512;

/// The 512-entry Golay70 GPU class table, 8 `u32` words per entry, matching
/// `GolayClassRec` of `llvq_golay.cuh` field for field — values divided by
/// `sqrt(16·shell)` like every other arm's table, exception classes and the
/// origin left all-zero (the main stream never names an exception class, and
/// the origin decodes to zero through a zero entry).
///
/// Built from [`Golay70Table`] — the table [`transcode_golay70`] encodes
/// against — so encoder and decoder read one derivation, not two: the even
/// side takes `pairs` (residue pairs in canonical level order, single-value
/// residues already duplicated), the odd side `values` and `flags`. The
/// shell norm is the one thing the class entry does not carry; it comes from
/// the same [`FastDecoder`] the table was built from.
pub fn golay70_gpu_class_table(fd: &FastDecoder, g70: &Golay70Table) -> Vec<u32> {
    assert!(g70.n_entries() <= GOLAY70_TABLE_ENTRIES);
    let mut tab = vec![0u32; GOLAY70_TABLE_ENTRIES * GOLAY70_REC_WORDS];
    for ci in 0..fd.n_classes() {
        let id = 1 + ci;
        let cls = g70.class(id);
        if cls.exception {
            continue;
        }
        let lv = fd.levels(ci);
        let norm = f64::from(16 * lv.shell).sqrt();
        let vals: [i8; 4] = if cls.odd {
            cls.values
        } else {
            [cls.pairs[0][0], cls.pairs[0][1], cls.pairs[1][0], cls.pairs[1][1]]
        };
        let base = id * GOLAY70_REC_WORDS;
        for (k, &v) in vals.iter().enumerate() {
            tab[base + k] = ((f64::from(v) / norm) as f32).to_bits();
        }
        tab[base + 4] = u32::from(cls.flags);
        tab[base + 5] = u32::from(cls.odd);
    }
    tab
}

/// The canonical 4096-codeword table the kernel resolves the 12-bit rank
/// through — `Golay::codewords()` order, the order frozen by format v1.
pub fn golay70_gpu_codewords(g70: &Golay70Table) -> Vec<u32> {
    let cw = g70.golay().codewords();
    assert_eq!(cw.len(), 4096, "the canonical Golay table has 4096 codewords");
    cw.to_vec()
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
                    "exception table not strictly increasing at {e} \
                     ({} after {})",
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
            "{} exceptions past the last block of the matrix ({} rows × {nblocks})",
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
/// `nearest_angular` over the m ≤ 12 ball, ~0.7 ms of one core each, 5,096,688
/// of them on the published 4B — and the searcher carries the lattice tables
/// that search needs. Holding it in a struct whose constructor knows the
/// layout makes "the searcher exists exactly when the layout needs one" an
/// invariant of the type rather than a rule a caller has to remember: there is
/// no way to reach [`Self::stream`] for `Planes12x` without one, and no way to
/// build 252 of them by accident.
pub struct Transcoder {
    layout: FusedLayout,
    fd: FastDecoder,
    /// The v1 ball's class table — `Some` exactly when the LAYOUT reads a
    /// class, which is every layout but `Tetra48`. The `searcher`/`g70`
    /// pattern: the `Some` is the authorisation, so no arm can reach a table
    /// its format does not have.
    ///
    /// 🕳️ It was unconditional, and `ClassTable::for_kind` refuses a Tetra
    /// kind by design — the table describes 383 ball classes and a Tetra word
    /// names none. So `LLVQ_FUSED_LAYOUT=tetra48` died on the served file with
    /// "no runtime layout for Tetra: this reads the v1 ball's classes, and a Tetra word names none" **before reading one block**,
    /// on the first card run of the served path (2026-09-10, $0.02). The
    /// record gates of step 6.6 were open; this one was upstream of them and
    /// nothing on the Mac reached it, because building a `Transcoder` is what
    /// a card run does first.
    table: Option<ClassTable>,
    /// `Some` exactly for [`FusedLayout::Planes12x`].
    searcher: Option<Searcher>,
    /// `Some` exactly for [`FusedLayout::Golay70`] — the class table
    /// [`transcode_golay70`] encodes against, and the source of the GPU
    /// tables ([`golay70_gpu_class_table`], [`golay70_gpu_codewords`]).
    g70: Option<Golay70Table>,
}

impl Transcoder {
    /// Build the tables for `layout`, checking up front what the layout needs
    /// from the class table — for a Ball file, which every file before v5 is.
    pub fn new(layout: FusedLayout) -> Result<Self, String> {
        Self::for_kind(layout, CodeKind::Ball)
    }

    /// [`Self::new`] under the file's [`CodeKind`]. Every layout here is a
    /// rearrangement of a v1 ball index; a Tetra header is refused at the
    /// table, before a single block is looked at, with the artifact crate's
    /// own words — `tetra48`, the served Tetra layout, is F1d's
    /// (`docs/ROADMAP.md` §2.2 quater, step 6). Portable: this is the side of
    /// the boundary a test on a machine without a card reaches.
    pub fn for_kind(layout: FusedLayout, kind: CodeKind) -> Result<Self, String> {
        let fd = FastDecoder::new();
        // Built for the layouts that READ a class. `Tetra48` reads a rank and
        // a trellis, so asking for the ball's table would refuse a file the
        // layout serves perfectly well — and asking for it is what refused
        // one. The ball layouts keep the check exactly as it was: a Tetra file
        // under `planes14` still dies here, by name, before a block is read.
        let table = match layout {
            FusedLayout::Tetra48 => None,
            _ => Some(
                ClassTable::for_kind(kind, &fd, 1)
                    .map_err(|e| format!("{}: {e}", layout.name()))?,
            ),
        };
        if let Some(table) = table.as_ref() {
        if matches!(
            layout,
            FusedLayout::Planes14 | FusedLayout::Planes12x | FusedLayout::Golay70
        ) {
            // Three bit-planes address eight levels; both plane layouts are
            // only a bijection of the Slot32 content while every class stays
            // within five — `Planes12x` needs it for its exception records,
            // which are Planes14 records. True of the v1 table, asserted
            // rather than assumed — the same guard planesbench re-asserts
            // before any timing. `Golay70` inherits the need: its exception
            // records are the same Planes14 records.
            if !(0..table.n_entries()).all(|e| table.record(e).len <= 5) {
                return Err("a class exceeds 5 levels: the bit planes cannot \
                            carry it"
                    .into());
            }
        }
        }
        let searcher = matches!(layout, FusedLayout::Planes12x).then(Searcher::new);
        let g70 = matches!(layout, FusedLayout::Golay70).then(|| Golay70Table::new(&fd));
        Ok(Self { layout, fd, table, searcher, g70 })
    }

    /// The Golay70 class table — `Some` exactly when the layout is
    /// [`FusedLayout::Golay70`], like `searcher` for `Planes12x`.
    pub fn golay70_table(&self) -> Option<&Golay70Table> {
        self.g70.as_ref()
    }

    pub fn layout(&self) -> FusedLayout {
        self.layout
    }

    pub fn decoder(&self) -> &FastDecoder {
        &self.fd
    }

    /// The class table of a layout that has one, or a refusal naming the
    /// layout that does not. Only the ball arms of [`Self::stream`] call it,
    /// and each of them is unreachable for `Tetra48` — so this error is a
    /// guard against a future arm, not a path any caller takes today.
    fn ball_table(&self) -> Result<&ClassTable, String> {
        self.table.as_ref().ok_or_else(|| {
            format!("{}: this layout reads no class table", self.layout.name())
        })
    }

    /// `None` for `Tetra48`, which reads no class. See the field.
    pub fn class_table(&self) -> Option<&ClassTable> {
        self.table.as_ref()
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
                "{} codes for {d_out} rows × {nblocks} blocks",
                indices.len()
            ));
        }
        match self.layout {
            FusedLayout::Slot32 => {
                let rt = transcode(&self.fd, self.ball_table()?, indices, gains, Layout::Slot32)
                    .map_err(|e| e.to_string())?;
                let bytes = rt.data.len() as u64 + rt.bases.len() as u64 * 4;
                let words = pack_words(&rt);
                Ok((HostStream::Slot32 { words, bases: rt.bases }, bytes))
            }
            FusedLayout::Planes14 => {
                let pb: PlanesBlocks =
                    transcode_planes14(&self.fd, self.ball_table()?, indices, gains)
                        .map_err(|e| e.to_string())?;
                let bytes = pb.data.len() as u64;
                Ok((HostStream::Planes14 { words: pack_plane_bytes(&pb.data) }, bytes))
            }
            // The one arm that consults neither the fast decoder nor the class
            // table: a Tetra word names no class, so there is nothing to look
            // up and nothing to re-encode. The pad IS counted in `bytes` here,
            // unlike Slot32's read-window slack, because it is not slack — the
            // row stride is the format, the kernel reads over it every row, and
            // a rate that dropped it would be the 2026-07-31 accounting error
            // in a new place.
            FusedLayout::Tetra48 => {
                let tb = llvq_artifact::tetra48::transcode_tetra48(
                    indices, gains, d_out, nblocks,
                )
                .map_err(|e| e.to_string())?;
                let bytes = tb.data.len() as u64;
                let stride_u32 = u32::try_from(tb.stride_u32)
                    .map_err(|_| format!("row stride {} overflows u32", tb.stride_u32))?;
                Ok((
                    HostStream::Tetra48 { words: pack_plane_bytes(&tb.data), stride_u32 },
                    bytes,
                ))
            }
            FusedLayout::Planes12x => {
                let s = self
                    .searcher
                    .as_ref()
                    .ok_or("Transcoder::new built no searcher for planes12x")?;
                let pb: Planes12xBlocks =
                    transcode_planes12x(&self.fd, self.ball_table()?, s, indices, gains)
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
            FusedLayout::Golay70 => {
                let g70 = self
                    .g70
                    .as_ref()
                    .ok_or("Transcoder::new built no table for golay70")?;
                let gb: Golay70Blocks =
                    transcode_golay70(&self.fd, self.ball_table()?, g70, indices, gains)
                        .map_err(|e| e.to_string())?;
                let row_exc = row_offsets(&gb.exc_idx, d_out, nblocks)?;
                // Same accounting rule as every arm: what the vectors hold,
                // the transcoder's own tail padding included (it is read —
                // the last block's three-word window lands in it), never a
                // packing pad added here.
                let bytes = gb.data.len() as u64
                    + gb.exc_idx.len() as u64 * 4
                    + gb.exc_data.len() as u64
                    + row_exc.len() as u64 * 4;
                Ok((
                    HostStream::Golay70 {
                        words: pack_aligned_bytes(&gb.data),
                        exc_idx: gb.exc_idx,
                        exc_words: pack_plane_bytes(&gb.exc_data),
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
    load_with(path, layout, FuseMode::Off)
}

/// [`load`] with the fusion mode named by the caller rather than read from the
/// environment.
///
/// Under [`FuseMode::Off`] this is `load` as it always was: [`FusedModel::
/// groups`] is empty, `matrices` carries all 252 projections of the published
/// 4B, and `runtime_bytes` is the same sum it always was. Under `On` the
/// projections that share an activation are spliced into [`FusedGroup`]s — 72
/// groups plus 72 lone projections, so 144 matvec launches a token.
pub fn load_with(path: &str, layout: FusedLayout, fuse: FuseMode) -> Result<FusedModel, String> {
    // Refused here as well as in `fused_cuda::load_with`, and on purpose: this
    // is the side of the boundary a test on a machine without a card reaches.
    // Without it `segment_matrices` would be handed exception-carrying streams
    // whose only symptom is a wrong number.
    if fuse == FuseMode::On && seg_kernel_name(layout).is_none() {
        return Err(format!(
            "LLVQ_FUSE=1 with LLVQ_FUSED_LAYOUT={}: only planes14 segments (see \
             seg_kernel_name)",
            layout.name()
        ));
    }
    let file_bytes = std::fs::metadata(path).map_err(|e| e.to_string())?.len();
    let f = std::fs::File::open(path).map_err(|e| format!("open {path}: {e}"))?;
    let mut r = std::io::BufReader::with_capacity(1 << 20, f);
    let head = llvq_artifact::read_header(&mut r).map_err(|e| e.to_string())?;
    if !head.is_self_contained() {
        return Err(format!(
            "{path} is only a projections artifact (format v{}), it cannot run on its \
             own.",
            head.version
        ));
    }

    // The header's whole set of kinds, not its default: one Tetra matrix among
    // four hundred Ball ones is a file this path cannot serve, and the refusal
    // has to land here, before any record is read through `read_matrix_raw` as
    // a Ball record.
    check_kinds(layout, head.kinds())?;
    let tr = Transcoder::for_kind(layout, lattice_kind(layout))?;
    let mut matrices = Vec::with_capacity(head.matrices as usize);
    let mut int4: Vec<FusedInt4> = Vec::new();
    let mut rotations: HashMap<RotKey, RotationTables> = HashMap::new();
    let mut quantized_weights = 0usize;

    for _ in 0..head.matrices {
        // `read_record` and not `read_matrix_raw`: the latter reads every
        // record as a lattice body whatever its kind tag says, which for an
        // int4 record is not a refusal but a misparse — it would consume the
        // wrong number of bytes and desynchronise every record after it.
        let rec = llvq_artifact::read_record(&mut r, head.version).map_err(|e| e.to_string())?;
        // The record's own kind, not only the header's set: the set is a
        // declaration, and a file whose header under-reports it would reach
        // the transcoder here with the wrong map in hand.
        if !serves_kind(layout, rec.kind()) {
            return Err(format!(
                "{}: a {:?} record in a file served as {}",
                rec.name(),
                rec.kind(),
                layout.name()
            ));
        }
        let m = match rec {
            llvq_artifact::Record::Int4(q) => {
                // Nothing to transcode: the kernel reads these bytes. The
                // weights count toward the model's quantized total exactly as
                // a lattice matrix's do — they are quantized, just not by us.
                quantized_weights += q.d_out * q.d_in;
                let bytes = (q.packed.len() + 2 * q.scales.len() + 2 * q.biases.len()) as u64;
                int4.push(FusedInt4 {
                    name: q.name,
                    d_out: q.d_out,
                    d_in: q.d_in,
                    group: q.group,
                    packed: q.packed,
                    scales: q.scales,
                    biases: q.biases,
                    bytes,
                });
                continue;
            }
            llvq_artifact::Record::Lattice(m) => m,
        };
        // Every lattice decoder hard-codes one gain bit (`hdr >> 9`). A file
        // with a different gain width would transcode into a coherent stream
        // and decode into garbage — silently. Lattice-only since 2026-09-09:
        // an int4 record carries no centroid at all, so this check used to
        // fire on exactly the record a mixed file most needs.
        if m.centroids.len() != 2 {
            return Err(format!(
                "{}: {} centroids, but the kernels hardcode 1 gain bit",
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
        let bytes = payload + matrix_side_bytes(m.d_out, tail_w);

        matrices.push(FusedMatrix {
            name: m.name,
            d_out: m.d_out,
            d_in: m.d_in,
            nblocks,
            tail_w,
            stream,
            gscale: [m.centroids[0] as f32, m.centroids[1] as f32],
            rscale: m.row_scales.iter().map(|&v| v as f32).collect(),
            tail: tail_f16_bits(&m.tail),
            rotation,
            bytes,
        });
    }

    // The rotation keys must partition the activation sites *before* anything
    // is uploaded: lot A4 hoists one rotation per site, and a file where two
    // consumers of one activation carry different rotations — or where two
    // activations share one — would make that hoist hand a projection the
    // wrong basis. Refused by name here rather than degraded silently, and
    // checked on every load so the `LLVQ_ROT_SHARE=0` arm rests on the same
    // premise as the `=1` arm.
    crate::rotplan::check_rotation_partition(&matrices)?;

    let (matrices, groups) = match fuse {
        FuseMode::Off => (matrices, Vec::new()),
        FuseMode::On => segment_matrices(matrices)?,
    };
    // Re-derived rather than patched: the splice moves payload between the two
    // vectors and adds `gs_off`, so an accumulator carried through the read
    // loop would have to be corrected twice. One sum over what the model
    // actually holds cannot drift from what it holds.
    //
    // 🕳️ The int4 term was missing, and it was missing on ONE side of a
    // division. `quantized_weights` counts an int4 record's weights — it says
    // so where it does it, "they are quantized, just not by us" — so
    // `runtime_bits_per_weight` divided a numerator without those bytes by a
    // denominator with those weights. On the served mixed file that prints
    // **2.030 b/weight where the truth is 2.141**, and a card footprint 50 MB
    // light, on the line `fusedrun` publishes. Nothing failed; the number was
    // simply wrong, and the test below now divides the two the same way.
    let runtime_bytes = runtime_bytes_of(&matrices, &groups, &int4);

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
        int4,
        layout,
        matrices,
        groups,
        rotations,
        raw,
        config_json: blobs
            .remove("config.json")
            .ok_or_else(|| format!("{path} carries no config.json"))?,
        tokenizer_json: blobs
            .remove("tokenizer.json")
            .ok_or_else(|| format!("{path} carries no tokenizer.json"))?,
        quantized_weights,
        carried_weights,
        file_bytes,
        runtime_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `LLVQ_FUSE` refuses anything that is not a mode — the [`FusedLayout::
    /// parse`] contract, restated beside its siblings.
    #[test]
    fn fuse_mode_parse_refuses_anything_else() {
        assert_eq!(FuseMode::parse(None), Ok(FuseMode::Off));
        assert_eq!(FuseMode::parse(Some("")), Ok(FuseMode::Off));
        assert_eq!(FuseMode::parse(Some("0")), Ok(FuseMode::Off));
        assert_eq!(FuseMode::parse(Some("1")), Ok(FuseMode::On));
        for bad in ["on", "off", "true", "2", "1 ", "01", "yes", "planes14"] {
            let e = FuseMode::parse(Some(bad)).expect_err("must be refused");
            assert!(e.contains(bad), "the message must cite the value: {e}");
        }
    }

    /// The int4 bytes are counted, and they are counted where the weights are.
    ///
    /// 🕳️ The regression this pins is not hypothetical: `runtime_bytes`
    /// summed the lattice matrices and the fused groups and stopped, while
    /// `quantized_weights` counted an int4 record's weights — "they are
    /// quantized, just not by us". A numerator missing a term whose weights
    /// are in the denominator understates every b/weight the runtime prints,
    /// and nothing fails.
    #[test]
    fn the_int4_bytes_reach_the_footprint_the_runtime_prints() {
        let q = FusedInt4 {
            name: "model.layers.0.self_attn.v_proj.weight".into(),
            d_out: 1024,
            d_in: 2560,
            group: 128,
            packed: Vec::new(),
            scales: Vec::new(),
            biases: Vec::new(),
            // What one served `v_proj` actually costs: 1024·2560/2 packed,
            // plus 1024·20 groups of one binary16 scale and one binary16 bias.
            bytes: 1024 * 2560 / 2 + 1024 * 20 * 2 * 2,
        };
        assert_eq!(q.bytes, 1_392_640);
        // The term that was missing, alone, so its absence cannot hide behind
        // the other two.
        assert_eq!(runtime_bytes_of(&[], &[], std::slice::from_ref(&q)), q.bytes);
        assert_eq!(runtime_bytes_of(&[], &[], &[]), 0);
        // Over the served file's 36 of them, against the hand arithmetic: the
        // whole term is 50.1 MB, which is what the printed line was light by.
        let all: Vec<FusedInt4> = (0..36)
            .map(|l| FusedInt4 { name: format!("model.layers.{l}.self_attn.v_proj.weight"), ..q.clone() })
            .collect();
        assert_eq!(runtime_bytes_of(&[], &[], &all), 50_135_040);
    }

    /// The name the host hands the driver is the name the embedded source
    /// defines, and the group it hard-codes is the format's.
    ///
    /// `fused_cuda.rs` looks `INT4_KERNEL_NAME` up on a card and nowhere else,
    /// so a typo in it is a `named symbol not found` at the end of a load —
    /// which is exactly how `tv_planes_seg_h` was found, on a rented card.
    #[test]
    fn the_int4_kernel_names_and_constants_match_its_source() {
        let src = Q4_H_CU_EMBED;
        assert!(
            src.contains(&format!("__global__ void {INT4_KERNEL_NAME}(")),
            "`{INT4_KERNEL_NAME}` is not defined by the source the host appends"
        );
        // The group is the format's constant on both sides of the boundary.
        assert!(
            src.contains(&format!("#define LLVQ_Q4_GROUP {}u", llvq_artifact::INT4G128_GROUP)),
            "the kernel's group and `INT4G128_GROUP` disagree"
        );
        // It needs `matvec.cu` (h2f, f2h, warp_sum) and nothing else, and that
        // file is the second part of EVERY unit — so appending this source
        // last is always legal, whatever the layout.
        assert!(src.contains("#ifndef TILE_COLS"), "the composition guard moved");
    }

    /// Every kernel a runtime looks up has its SOURCE in that runtime's unit.
    ///
    /// 🕳️ The one invariant `fused_cuda.rs` cannot check for itself: that file
    /// does not compile on this machine — it needs nvcc and a Linux cross
    /// toolchain — so a name it asks the driver for is only ever tested on a
    /// rented card, at the end of a load. On 2026-09-10 it asked for
    /// `tv_planes_seg_h` on the `Tetra48` build, whose list shares nothing
    /// with the ball layouts, and the run died after compiling, loading and
    /// reporting 216 projections. Two of the three names it looks up are
    /// pinned here instead, where the check is free.
    ///
    /// `Slot32` is the exception and stays one: its kernel lives in the base
    /// list every layout gets (`llvq_slot.cuh`, `matvec.cu`), not in the
    /// per-layout list, which is precisely what keeps its translation unit
    /// bit-identical to what shipped before any layout switch existed.
    #[test]
    fn every_layout_only_names_kernels_its_own_unit_carries() {
        for layout in [
            FusedLayout::Slot32,
            FusedLayout::Planes14,
            FusedLayout::Planes12x,
            FusedLayout::Golay70,
            FusedLayout::Tetra48,
        ] {
            let srcs = planes_source_names(layout);
            // The segmented kernel, ONE WAY ONLY — and the asymmetry is the
            // point. `Planes12x` and `Golay70` carry `tv_planes_seg_h.cu` and
            // cannot launch it: it is compiled so the register report stays a
            // drift detector on those builds, which the list's own header
            // argues at length. So *compiled* is a superset of *launchable*,
            // and only the implication holds — a layout that can launch it
            // must have it.
            //
            // Which is why `fused_cuda`'s register report keys on the SOURCE
            // LIST and not on either the layout or `seg_kernel_name`: on
            // `Tetra48` all three answers differ, and it picked the wrong one.
            if seg_kernel_name(layout).is_some() {
                assert!(
                    srcs.contains(&"tv_planes_seg_h.cu"),
                    "{}: launches `tv_planes_seg_h` and its unit does not carry it: {srcs:?}",
                    layout.name()
                );
            }
            // Tetra48 has neither, and that is the case that crashed: it is
            // not Slot32, so "not Slot32" reported a kernel it never had.
            if layout == FusedLayout::Tetra48 {
                assert!(seg_kernel_name(layout).is_none());
                assert!(!srcs.contains(&"tv_planes_seg_h.cu"));
            }
            // The prefill entry point, same rule: named only where the unit
            // carries the source that defines it. `tests/served_unit.rs`
            // checks the other half — that the source really defines it.
            if let Some(n) = rows_kernel_name(layout) {
                assert_eq!(n, "tv_tetra48_rows_h");
                assert!(
                    srcs.contains(&"tv_tetra48_h.cu"),
                    "{}: names `{n}` and carries no source that defines it",
                    layout.name()
                );
            }
            // The matvec kernel: its own `.cu` is in the list, except Slot32's
            // which is in the base list every unit already carries.
            let m = matvec_kernel_name(layout);
            if layout != FusedLayout::Slot32 {
                assert!(
                    srcs.contains(&format!("{m}.cu").as_str()),
                    "{}: looks up `{m}` and carries no `{m}.cu`: {srcs:?}",
                    layout.name()
                );
            } else {
                assert!(srcs.is_empty(), "Slot32's unit is the base list and nothing else");
            }
        }
    }

    /// A Tetra header has no runtime layout **among the ball ones**: each of
    /// the four refuses it with the artifact crate's words, and the Ball arm
    /// is `new` — same tables, same layout.
    ///
    /// 🕳️ This test swept the four ball layouts and stopped there, so
    /// `Tetra48` never passed through `for_kind` on this machine at all. The
    /// layout that the whole of step 6 exists to serve was the one layout the
    /// portable test did not build — and it died on a card, at $0.02, on the
    /// first run of the served path, before reading a single block. A sweep
    /// that omits the subject is not a sweep. `Tetra48` is a POSITIVE case
    /// below, which is what makes the omission impossible to repeat.
    #[test]
    fn a_tetra_file_has_no_runtime_layout_among_the_ball_ones() {
        for layout in [
            FusedLayout::Planes14,
            FusedLayout::Planes12x,
            FusedLayout::Slot32,
            FusedLayout::Golay70,
        ] {
            let e = Transcoder::for_kind(layout, CodeKind::Tetra)
                .err()
                .unwrap_or_else(|| panic!("{}: a Tetra header must be refused", layout.name()));
            assert!(
                e.contains("no runtime layout for Tetra: this reads the v1 ball's classes, and a Tetra word names none"),
                "{}: the refusal must say why: {e}",
                layout.name()
            );
            assert!(e.contains(layout.name()), "{}: the refusal must name the layout: {e}", layout.name());
            let ball = Transcoder::for_kind(layout, CodeKind::Ball).expect("a Ball header builds");
            assert_eq!(ball.layout(), layout);
            assert_eq!(
                ball.class_table().expect("a ball layout has a class table").n_entries(),
                Transcoder::new(layout)
                    .expect("new")
                    .class_table()
                    .expect("a ball layout has a class table")
                    .n_entries()
            );
        }

        // And the layout that serves Tetra BUILDS on a Tetra header, with no
        // class table at all — a Tetra word names no class, so asking for the
        // ball's 383 of them is what refused a file this layout serves.
        let t = Transcoder::for_kind(FusedLayout::Tetra48, CodeKind::Tetra)
            .expect("tetra48 must build on a Tetra header");
        assert_eq!(t.layout(), FusedLayout::Tetra48);
        assert!(t.class_table().is_none(), "tetra48 must hold no class table");

        // A BALL file under `tetra48` is still refused — not here, where the
        // layout needs no table, but at the record gate. What this asserts is
        // that dropping the table did not drop that: the two refusals are
        // different, and only one of them was ever this function's business.
        assert!(
            Transcoder::for_kind(FusedLayout::Tetra48, CodeKind::Ball).is_ok(),
            "the table is not where a Ball file under tetra48 is refused"
        );
        assert!(!serves_kind(FusedLayout::Tetra48, CodeKind::Ball));
        assert!(serves_kind(FusedLayout::Tetra48, CodeKind::Tetra));
    }

    /// The default is `Off`, and it *has* to be: [`check_fuse`] refuses `On`
    /// beside `RotShare::Off`, whose own default is `Off`, so a default of `On`
    /// here would make every `fused_cuda::load` fail with no environment set.
    /// The two defaults are pinned together so neither can move alone.
    #[test]
    fn the_two_defaults_are_compatible() {
        let (fuse, share) = (
            FuseMode::parse(None).expect("default"),
            crate::rotplan::RotShare::parse(None).expect("default"),
        );
        assert_eq!(fuse, FuseMode::Off);
        assert_eq!(share, crate::rotplan::RotShare::Off);
        check_fuse(FusedLayout::Planes14, share, fuse).expect("the two defaults hold together");
        // And the pair this is guarding against, in both of its forms.
        let e = check_fuse(FusedLayout::Planes14, crate::rotplan::RotShare::Off, FuseMode::On)
            .expect_err("On + RotShare::Off must be refused");
        assert!(e.contains("LLVQ_ROT_SHARE"), "{e}");
        for layout in [FusedLayout::Planes12x, FusedLayout::Slot32, FusedLayout::Golay70] {
            let e = check_fuse(layout, crate::rotplan::RotShare::On, FuseMode::On)
                .expect_err("only planes14 segments");
            assert!(e.contains(layout.name()), "the refusal must name the layout: {e}");
            // …and `Off` stays launchable on every layout, since it is the
            // control arm.
            check_fuse(layout, crate::rotplan::RotShare::On, FuseMode::Off).expect("control");
        }
        check_fuse(FusedLayout::Planes14, crate::rotplan::RotShare::On, FuseMode::On)
            .expect("the only fusable pair");
    }

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
            assert!(dev / nrm < 1e-6, "n={n}: relative deviation {:.3e}", dev / nrm);
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
