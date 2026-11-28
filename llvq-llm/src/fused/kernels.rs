//! The kernel text a layout needs and the names its entry points carry, with
//! the `LLVQ_KERNEL_DIR` loaders and the int4 fixtures. One file because every
//! `include_str!` of the fused path lives here, so the served text has one home.

use super::*;

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
const PLANES_CUH_EMBED: &str = include_str!("../../../llvq-cuda/kernels/llvq_planes.cuh");
const PLANES_CU_EMBED: &str = include_str!("../../../llvq-cuda/kernels/planes.cu");
const PLANES_H_CU_EMBED: &str = include_str!("../../kernels/tv_planes_h.cu");
const PLANES_SEG_H_CU_EMBED: &str = include_str!("../../kernels/tv_planes_seg_h.cu");
const PLANES12_CUH_EMBED: &str = include_str!("../../../llvq-cuda/kernels/llvq_planes12.cuh");
const PLANES12X_H_CU_EMBED: &str = include_str!("../../kernels/tv_planes12x_h.cu");
const GOLAY_CUH_EMBED: &str = include_str!("../../../llvq-cuda/kernels/llvq_golay.cuh");
const GOLAY70_H_CU_EMBED: &str = include_str!("../../kernels/tv_golay70_h.cu");
// Tetra48's four. The first three come from `llvq-cuda` verbatim — the F1
// decoder chain the bench of 2026-09-08 measured and the served decode
// `tests/tetra48_matches_rust.rs` pins bit for bit — and the fourth is this
// crate's own half-storing entry point.
const F1RANK_CUH_EMBED: &str = include_str!("../../../llvq-cuda/kernels/llvq_f1rank.cuh");
const F1RANK_V3_CUH_EMBED: &str = include_str!("../../../llvq-cuda/kernels/llvq_f1rank_v3.cuh");
const TETRA48_CUH_EMBED: &str = include_str!("../../../llvq-cuda/kernels/llvq_tetra48.cuh");
const TETRA48_H_CU_EMBED: &str = include_str!("../../kernels/tv_tetra48_h.cu");
/// The int4 g128 projection kernel, for the mixed files `LLVQ_INT4_TYPES`
/// writes. Not a layout source: it is orthogonal to the lattice layout — a
/// mixed file carries int4 records **beside** Tetra or Ball ones — so it is
/// selected by the file's `KindSet` and not by `LLVQ_FUSED_LAYOUT`.
pub(super) const Q4_H_CU_EMBED: &str = include_str!("../../kernels/tv_q4_h.cu");

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
/// Built here rather than in `fused_cuda/` for the reason that module gives
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
    /// entries; `m ≤ 26` on this codebook.
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
/// It lives here rather than in `fused_cuda/` for the reason this whole
/// module is unconditional: `fused_cuda/` is `cfg(target_os = "linux",
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

/// The same matvec with an f16 activation, widened in the kernel while it is
/// staged: the served path's model carries f16, and this entry point saves the
/// separate cast launch [`INT4_KERNEL_NAME`] needs. Same shared request, same
/// output bits; the runtime checks the second claim on the card at load.
pub const INT4_HALF_KERNEL_NAME: &str = "tv_q4_hh";

/// The dynamic shared memory [`INT4_KERNEL_NAME`] asks for over a model's int4
/// records: the widest activation, staged whole in f32; `0` when there is none.
///
/// Not a hidden size any more. The sealed objects serve `o_proj` and
/// `down_proj` in int4, and a `down_proj` stages the intermediate width:
/// 9,728 floats at 4B (38,912 B), 12,288 at 8B (49,152 B, the default to the
/// byte) and 17,408 at 14B (69,632 B, past it). The 14B sealed file was
/// refused at load on 2026-09-25 against the 49,152 default (job
/// 6ab6212752d0dbd7f1d8eae0), while the card offers 101,376 on request.
pub fn int4_shared_bytes(d_ins: impl IntoIterator<Item = usize>) -> usize {
    d_ins.into_iter().map(|d| d * 4).max().unwrap_or(0)
}

/// The int4 entry point that stages its activation as f16 bits, `d_in * 2`
/// bytes, for records whose f32 staging passes the card's ceiling (an int4
/// `down_proj` of Qwen3-32B, d_in 25,600, on a 99 KiB card). Same bits as
/// [`INT4_HALF_KERNEL_NAME`]; it takes an f16 activation and an `x_off`.
pub const INT4_WIDE_KERNEL_NAME: &str = "tv_q4_hs";

/// `tv_q4_wide.cu`, the unit that defines [`INT4_WIDE_KERNEL_NAME`].
///
/// A unit of its own, compiled only when some record is F16-staged
/// (`llvq_cuda::wide::int4_staging`), so the served unit's text and printed
/// sha256 do not move. Composition: `llvq_slot.cuh`, `matvec.cu`, `tv_q4_h.cu`
/// (whose `q4_deq` it calls), then this file; [`int4_wide_unit_names`].
pub const Q4_WIDE_CU_EMBED: &str = include_str!("../../kernels/tv_q4_wide.cu");

/// The wide int4 unit, in the order NVRTC must see it. The first two come
/// from `llvq_cuda::embedded_source`, the last two from this crate.
pub fn int4_wide_unit_names() -> [&'static str; 4] {
    ["llvq_slot.cuh", "matvec.cu", "tv_q4_h.cu", "tv_q4_wide.cu"]
}

/// The wide int4 sources `tv_q4_h.cu` then `tv_q4_wide.cu`, honouring
/// `LLVQ_KERNEL_DIR` on [`load_int4_sources`]'s all-or-nothing terms.
pub fn load_int4_wide_sources() -> Result<([String; 2], Option<String>), String> {
    match std::env::var("LLVQ_KERNEL_DIR") {
        Err(_) => Ok(([Q4_H_CU_EMBED.to_string(), Q4_WIDE_CU_EMBED.to_string()], None)),
        Ok(dir) => {
            let read = |n: &str| {
                std::fs::read_to_string(std::path::Path::new(&dir).join(n))
                    .map_err(|e| format!("LLVQ_KERNEL_DIR={dir}: {n}: {e}"))
            };
            Ok(([read("tv_q4_h.cu")?, read("tv_q4_wide.cu")?], Some(dir)))
        }
    }
}

/// `tv_q4_tile.cu`: the int4 matvec on the tiled words 4E v9 reads
/// (`llvq_cuda::pers::q4_tile_index`), for a runtime that holds those words
/// alone ([`Int4Layout::Tiled`]). Appended to the unit only then, so the unit
/// of every other config, and its printed sha256, do not move. Composition:
/// `llvq_slot.cuh`, `matvec.cu`, `tv_q4_h.cu` (`q4_deq`), then this file.
/// Embedded only: a tiled runtime comes from a served config, which refuses
/// `LLVQ_KERNEL_DIR` by name.
pub const Q4_TILE_CU_EMBED: &str = include_str!("../../kernels/tv_q4_tile.cu");

/// [`INT4_KERNEL_NAME`] on tiled words, `x` in f32.
pub const INT4_TILE_KERNEL_NAME: &str = "tv_q4_ht";

/// [`INT4_HALF_KERNEL_NAME`] on tiled words, `x` in f16.
pub const INT4_TILE_HALF_KERNEL_NAME: &str = "tv_q4_hht";

/// How a runtime lays an int4 record's words on the card. The file always
/// stores the served order; the choice is made at load and moves no byte of
/// the count (a permutation inside each row).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Int4Layout {
    /// The file's order, read by `tv_q4_h` / `tv_q4_hh` and by 4E v3 to v8.
    Served,
    /// The tiled order of 4E v9, alone on the card, read by `tv_q4_ht` /
    /// `tv_q4_hht` and by 4E v9. Chosen by a config whose rows are "v9".
    Tiled,
}

impl Int4Layout {
    pub fn name(self) -> &'static str {
        match self {
            Self::Served => "served",
            Self::Tiled => "tiled",
        }
    }
}

/// The tiled layout's condition on a record: `d_in % 32 == 0`, so a row is a
/// whole number of 16-byte loads (nwords a multiple of 4). Every int4 record
/// of the three sealed files meets it (2,560, 4,096, 5,120, 9,728, 12,288,
/// 17,408).
pub fn int4_tileable(d_in: usize) -> bool {
    d_in.is_multiple_of(32)
}

/// A synthetic int4 g128 record and an f16 activation, drawn from `seed`.
///
/// The same bytes on the Mac and on a card: the host test
/// (`tests/proj_q4_wide.rs`) runs `tv_q4_hs` and `tv_q4_hh` on them and writes
/// the sha256 of the output to `tests/fixtures/q4-wide-hashes.txt`, and
/// `fusedrun`'s `LLVQ_INT4_WIDE_CHECK` runs the kernels on the card and
/// compares. Nibbles uniform, scales in about [2^-9, 2^-5], biases of both
/// signs, the activation over seven binades with exact zeros, as in the load
/// check `check_int4_half_parity`.
pub fn int4_synthetic(d_out: usize, d_in: usize, seed: u64) -> (FusedInt4, Vec<u16>) {
    let group = llvq_artifact::INT4G128_GROUP;
    let gpr = d_in.div_ceil(group);
    let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) as u32
    };
    let packed: Vec<u8> = (0..d_out * d_in / 2).map(|_| (next() & 0xff) as u8).collect();
    let unit = |v: u32| (v & 0xff_ffff) as f32 / (1u32 << 24) as f32;
    let scales: Vec<u16> = (0..d_out * gpr)
        .map(|_| half::f16::from_f32(0.002 + 0.03 * unit(next())).to_bits())
        .collect();
    let biases: Vec<u16> = (0..d_out * gpr)
        .map(|_| half::f16::from_f32(0.2 * (unit(next()) - 0.5)).to_bits())
        .collect();
    let x: Vec<u16> = (0..d_in)
        .map(|i| {
            let u = unit(next()) - 0.5;
            let v = if i % 17 == 0 { 0.0 } else { u * (1u32 << (i % 7)) as f32 };
            half::f16::from_f32(v).to_bits()
        })
        .collect();
    let bytes = (packed.len() + 2 * (scales.len() + biases.len())) as u64;
    let rec = FusedInt4 {
        name: format!("synthetic.int4.{d_out}x{d_in}.{seed:#x}"),
        d_out,
        d_in,
        group,
        packed,
        scales,
        biases,
        bytes,
    };
    (rec, x)
}

/// The synthetic records whose `tv_q4_hs` output the card compares against
/// the host's: `(d_out, d_in, seed, x_off)`. 25,344 is the widest d_in whose
/// f32 staging fits 101,376 B, so `tv_q4_hh` runs beside it on a 99 KiB card;
/// 25,600 is the Qwen3-32B `down_proj`, where only `tv_q4_hs` runs there.
pub const INT4_WIDE_CARD_CASES: [(usize, usize, u64, usize); 3] = [
    (256, 25_344, 0x41, 0),
    (256, 25_600, 0x42, 0),
    (256, 25_600, 0x43, 37),
];

/// `<d_out> <d_in> <seed hex> <x_off> <sha256 of the f16 output, LE>`, one
/// line per [`INT4_WIDE_CARD_CASES`] entry, written by the host test. Compiled
/// in, because the card image holds no repository tree.
pub const INT4_WIDE_FIXTURE: &str = include_str!("../../tests/fixtures/q4-wide-hashes.txt");

/// `(d_out, d_in, seed, x_off)` to the output sha256.
pub type Int4WideFixture = std::collections::BTreeMap<(usize, usize, u64, usize), String>;

/// [`INT4_WIDE_FIXTURE`], parsed: `(d_out, d_in, seed, x_off)` to the hash.
/// Refuses a malformed line or a duplicate case by line number.
pub fn parse_int4_wide_fixture(text: &str) -> Result<Int4WideFixture, String> {
    let mut m = std::collections::BTreeMap::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() != 5 {
            return Err(format!("fixture line {}: {} fields, expected 5", i + 1, f.len()));
        }
        let num = |k: usize| -> Result<usize, String> {
            f[k].parse().map_err(|e| format!("fixture line {}: field {}: {e}", i + 1, k + 1))
        };
        let seed = u64::from_str_radix(f[2].trim_start_matches("0x"), 16)
            .map_err(|e| format!("fixture line {}: seed: {e}", i + 1))?;
        if f[4].len() != 64 || !f[4].bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(format!("fixture line {}: {:?} is not a sha256", i + 1, f[4]));
        }
        let key = (num(0)?, num(1)?, seed, num(3)?);
        if m.insert(key, f[4].to_string()).is_some() {
            return Err(format!("fixture line {}: {key:?} twice", i + 1));
        }
    }
    Ok(m)
}

/// The q4 embedding source, `emb_q4.cu`, appended under [`EmbedMode::Q4`].
///
/// Here rather than in `fused_cuda/` for [`load_int4_sources`]'s reason: the
/// names the host looks up and the text that defines them are checked by a
/// test on a machine with no card (`the_q4_embedding_kernel_names_match_its_source`).
/// `LLVQ_KERNEL_DIR` overrides it on the same all-or-nothing terms.
pub fn load_emb_q4_sources() -> Result<(String, Option<String>), String> {
    match std::env::var("LLVQ_KERNEL_DIR") {
        Err(_) => Ok((EMB_Q4_CU_EMBED.to_string(), None)),
        Ok(dir) => {
            let p = std::path::Path::new(&dir).join("emb_q4.cu");
            let text = std::fs::read_to_string(&p)
                .map_err(|e| format!("LLVQ_KERNEL_DIR={dir}: emb_q4.cu: {e}"))?;
            Ok((text, Some(dir)))
        }
    }
}

pub(super) const EMB_Q4_CU_EMBED: &str = include_str!("../../kernels/emb_q4.cu");

/// The `(gather, lm_head)` entry points of a quantized embedding mode, the
/// pair the CUDA runtime looks up and reports registers for. `None` at f16,
/// which launches neither.
pub fn emb_kernel_names(mode: EmbedMode) -> Option<(&'static str, &'static str)> {
    match mode {
        EmbedMode::F16 => None,
        EmbedMode::Q8 => Some(("emb_q8_gather", "tv_q8_h")),
        EmbedMode::Q4 => Some(("emb_q4_gather", "tv_emb_q4_h")),
    }
}

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
