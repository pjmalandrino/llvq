//! Load-time options of the Metal path and the launch-shape rules they feed: `MetalTuning`, the Tetra kernel
//! names and their row loops, the int4 tile, and the threadgroup checks. One file because every one of them
//! answers the same question, which kernel and which launch shape a projection gets, before anything runs.

use candle_core::Result;
use candle_metal_kernels::metal::ComputePipeline;

use super::launch::{LANES, THREADGROUP_LIMIT};
use super::port::MetalLattice;
use super::source::source_of;

/// Columns `tv_q4_metal_tiled` stages a tile, for an int4 record whose whole
/// activation does not fit in `THREADGROUP_LIMIT`.
///
/// Every sealed file's int4 `down_proj` is past the wall: d_in 9,728 at 4B,
/// 12,288 at 8B, 17,408 at 14B. Records that fit keep `tv_q4_metal` and their
/// exact code path.
///
/// 2,048 and not the 8,192 that would fill the limit, for residency: a tile of
/// 8,192 asks all 32,768 B, so likely one threadgroup a core, while 8 KB leaves
/// room for several. Pinning 18.7 KB of threadgroup memory already capped
/// residency on this M3 Max and cost +8 to +23 % (commit 82c029c). llvq-tetra
/// ships 2,048 as the value its kernels were validated with. The tile moves no
/// bit: a multiple of 256 columns is a multiple of 32 words, so lane `l` walks
/// the untiled kernel's words in the untiled order
/// (`llvq-metal/tests/q4_tiled_matches_served.rs`).
pub const Q4_TILE_COLS: usize = 2048;

/// Load-time options of the Metal path, for measurement harnesses.
///
/// None of them changes a served bit: each kernel a field names is gated
/// equal to the served one, bit for bit. The served door takes
/// [`MetalTuning::default`], which is the served path as it stood before this
/// struct existed, plus the tiled int4 route for the records past the wall.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalTuning {
    /// Columns a tile for the int4 records past the wall.
    pub q4_tile_cols: usize,
    /// The Tetra matvec. `tv_tetra48_metal_lut` is the served one.
    pub tetra_kernel: &'static str,
    /// The int4 row loop: `false` is the served `tv_q4_metal` /
    /// `tv_q4_metal_tiled`, `true` their prefetching twins
    /// (`kernels/tv_q4_pf.metal`).
    pub q4_prefetch: bool,
    /// The half-io twins: the Tetra matvec stores f16 (`_luth`, `_lutdh`) and
    /// the int4 matvecs read and store f16 (`tv_q4_metal_h`,
    /// `tv_q4_metal_tiled_h`), so the adapter's casts around them, 336
    /// launches a token on the sealed 4B, are not issued. The same roundings
    /// in another place, so the same bits.
    pub half_io: bool,
    /// Prompt rows one launch of the Tetra prefill kernel takes
    /// (`tv_tetra48_metal_lutdr`). `1`, the served value, means no rows
    /// kernel: the prefill fans out row by row through the matvec. 2, 4 or 8
    /// decode each block once for that many rows, with every row's bits those
    /// of the matvec narrowed to f16.
    pub prefill_rows: usize,
    /// The int4 records take the prefill chunks too, through
    /// `tv_q4_metal_rows`; requires `prefill_rows` above 1. `false`, the
    /// served value, keeps their row-by-row loop.
    pub q4_rows: bool,
    /// The one-row rotations of width at most 8,192 run `rot_apply_tg_metal`,
    /// which stages the vector in threadgroup memory, instead of
    /// `rot_apply_metal`, which runs every stage through device memory. The
    /// same arithmetic in the same order, so the same bits. `false` is served.
    pub rot_tg: bool,
}

impl Default for MetalTuning {
    fn default() -> Self {
        Self {
            q4_tile_cols: Q4_TILE_COLS,
            tetra_kernel: SERVED_TETRA,
            q4_prefetch: false,
            half_io: false,
            prefill_rows: 1,
            q4_rows: false,
            rot_tg: false,
        }
    }
}

/// Rows the Tetra prefill kernel can take a launch (`TETRA48_PREFILL_MAX`).
pub(super) const TETRA_PREFILL_MAX: usize = 8;

impl MetalTuning {
    /// The tuning a `bin/metalrun` arm names, `lutd+h+pq` for one: a Tetra
    /// matvec, then modifiers. The one reader of that spelling, shared by the
    /// bench and by the served configs that name their Metal kernels
    /// ([`MetalTuning::of_served`]), so an arm a journal timed and a config
    /// that serves it cannot mean two things. `+grp` and `+ker` are attention
    /// forms, not kernels: the caller takes them off first.
    pub fn parse_arm(s: &str) -> Result<Self> {
        let mut parts = s.split('+');
        let kernel = match parts.next().unwrap_or("") {
            "lut" => SERVED_TETRA,
            "lutp" => "tv_tetra48_metal_lutp",
            "lutd" => "tv_tetra48_metal_lutd",
            "lutdp" => "tv_tetra48_metal_lutdp",
            "lutd2" => "tv_tetra48_metal_lutd2",
            "lutd4" => "tv_tetra48_metal_lutd4",
            "lutdt" => "tv_tetra48_metal_lutdt",
            other => candle_core::bail!("arm {s:?}: {other:?} is not lut, lutp, lutd, lutdp, lutd2, lutd4 or lutdt"),
        };
        let mut t = MetalTuning { tetra_kernel: kernel, ..Default::default() };
        for m in parts {
            match m {
                "q4p" => t.q4_prefetch = true,
                "h" => t.half_io = true,
                "p" => t.prefill_rows = TETRA_PREFILL_MAX,
                "r" => t.rot_tg = true,
                "pq" => {
                    t.prefill_rows = TETRA_PREFILL_MAX;
                    t.q4_rows = true;
                }
                c if c.starts_with('t') && c.len() > 1 => {
                    t.q4_tile_cols =
                        c[1..].parse().map_err(|_| candle_core::Error::msg(format!("arm {s:?}: tile {c:?}")))?
                }
                other => candle_core::bail!("arm {s:?}: unknown modifier {other:?}"),
            }
        }
        t.validate()?;
        Ok(t)
    }

    /// The tuning a served config names in `metal_kernels`. `Default` is
    /// [`MetalTuning::default`], what every config before 2026-11-26 served.
    pub fn of_served(k: crate::served::MetalKernels) -> Result<Self> {
        match k.arm() {
            None => Ok(Self::default()),
            Some(arm) => Self::parse_arm(arm),
        }
    }
}

/// What the runtime under a loaded model launches, found through its first
/// Tetra projection; `None` when the model has no Metal Tetra projection.
pub fn tuning_of(m: &crate::model::Qwen3) -> Option<MetalTuning> {
    m.blocks.iter().find_map(|b| {
        [&b.q_proj, &b.k_proj, &b.v_proj, &b.o_proj, &b.gate_proj, &b.up_proj, &b.down_proj].into_iter().find_map(|p| {
            match p {
                crate::model::Proj::Lattice(d) => {
                    d.as_any().and_then(|a| a.downcast_ref::<MetalLattice>()).map(|l| l.rt.tuning())
                }
                _ => None,
            }
        })
    })
}

/// The served Tetra matvec on Metal.
pub const SERVED_TETRA: &str = "tv_tetra48_metal_lut";

impl MetalTuning {
    /// Refuses a bad tile or an unknown kernel when the load starts, whether
    /// or not a record of this file needs it. A zero tile makes the kernel's
    /// `c0 += tile_cols` loop forever: a GPU hang, not an error.
    pub fn validate(&self) -> Result<()> {
        let c = self.q4_tile_cols;
        if c == 0 || !c.is_multiple_of(256) {
            candle_core::bail!(
                "q4 tile_cols {c} is not a positive multiple of 256. A tile of tile_cols/8 words must be a \
                 multiple of 32 or the lanes interleave differently and the f32 moves"
            );
        }
        if c * 4 > THREADGROUP_LIMIT {
            candle_core::bail!(
                "q4 tile_cols {c} stages {} B of threadgroup memory against {THREADGROUP_LIMIT}",
                c * 4
            );
        }
        source_of(self.tetra_kernel)?;
        let k = self.tetra_kernel;
        if k != SERVED_TETRA && !TETRA_ROW_LOOPS.contains(&k) && !TETRA_REFERENCES.contains(&k) {
            candle_core::bail!(
                "{k} is not a 32-lane Tetra matvec with the served arguments and an f32 output: \
                 {SERVED_TETRA}, a row loop of TETRA_ROW_LOOPS or a reference of TETRA_REFERENCES. \
                 The f16 twins come through half-io, the prefill kernel through prefill rows"
            );
        }
        if self.half_io {
            if tetra_half_twin(self.tetra_kernel).is_none() {
                candle_core::bail!(
                    "half-io: {} has no half-store twin; tv_tetra48_metal_lut and tv_tetra48_metal_lutd have one",
                    self.tetra_kernel
                );
            }
            if self.q4_prefetch {
                candle_core::bail!("half-io and the prefetching int4 twins are two arms; name one");
            }
        }
        if !matches!(self.prefill_rows, 1 | 2 | 4 | 8) {
            candle_core::bail!(
                "prefill rows {}: the Tetra prefill kernel takes 2, 4 or 8 rows a launch, and 1 is none",
                self.prefill_rows
            );
        }
        if self.q4_rows && self.prefill_rows == 1 {
            candle_core::bail!("q4 rows: the int4 prefill kernel takes the chunks prefill rows names, and 1 is none");
        }
        Ok(())
    }
}

/// The half-store twin of a Tetra matvec, if it has one.
pub(super) fn tetra_half_twin(name: &str) -> Option<&'static str> {
    match name {
        "tv_tetra48_metal_lut" => Some("tv_tetra48_metal_luth"),
        "tv_tetra48_metal_lutd" => Some("tv_tetra48_metal_lutdh"),
        "tv_tetra48_metal_lutdt" => Some("tv_tetra48_metal_lutdth"),
        _ => None,
    }
}

/// The row-loop twins of the served Tetra matvec (`kernels/llvq_tetra48_rows.metal`).
/// Each has the served kernel's arguments and, by gate, its bits.
pub const TETRA_ROW_LOOPS: [&str; 6] = [
    "tv_tetra48_metal_lutp",
    "tv_tetra48_metal_lutd",
    "tv_tetra48_metal_lutdp",
    "tv_tetra48_metal_lutd2",
    "tv_tetra48_metal_lutd4",
    "tv_tetra48_metal_lutdt",
];

/// The reference variants of the served Tetra matvec (`kernels/llvq_tetra48.metal`):
/// the faithful port and three other decoders of it (`_ar`, `_u2`, `_c3`).
/// Each has the served kernel's arguments, 32 lanes a row, an f32 output, and
/// no threadgroup memory but the staging at index 0 the host sizes.
const TETRA_REFERENCES: [&str; 4] =
    ["tv_tetra48_metal", "tv_tetra48_metal_ar", "tv_tetra48_metal_u2", "tv_tetra48_metal_c3"];

/// Whether a Tetra matvec stages the decoder tables in threadgroup memory and
/// loops its simdgroups over the rows (`_lutdt`, `_lutdth`): its own dispatch.
pub(super) fn tetra_loops_rows(name: &str) -> bool {
    matches!(name, "tv_tetra48_metal_lutdt" | "tv_tetra48_metal_lutdth")
}

/// Rows a simdgroup of `_lutdt` takes, on average, from one copy of the tables.
/// The threadgroup count is `ceil(d_out / (simdgroups * this))`.
pub(super) const TETRA_TG_LOOP_ROWS: usize = 4;

/// Threadgroup bytes of `_lutdt`'s five tables, in binding order: rank rows
/// (4,096 u32), branches (1,024 u16), prefixes and suffixes (128 u8 each),
/// the 64 values of `F1R_VAL`.
pub(super) const TETRA_TG_TABLES: [usize; 5] = [16_384, 2_048, 128, 128, 256];

/// Whether a Tetra matvec reads the activation from device memory, with no
/// threadgroup staging to size.
pub(super) fn tetra_reads_direct(name: &str) -> bool {
    matches!(
        name,
        "tv_tetra48_metal_lutd"
            | "tv_tetra48_metal_lutdp"
            | "tv_tetra48_metal_lutd2"
            | "tv_tetra48_metal_lutd4"
            | "tv_tetra48_metal_lutdh"
            | "tv_tetra48_metal_lutdt"
            | "tv_tetra48_metal_lutdth"
    )
}

/// Rows a simdgroup of a Tetra matvec carries: 1 for every kernel but the
/// multi-row twins, which keep 32 lanes a row and give a lane R rows.
pub(super) fn tetra_rows_per_simd(name: &str) -> usize {
    match name {
        "tv_tetra48_metal_lutd2" => 2,
        "tv_tetra48_metal_lutd4" => 4,
        _ => 1,
    }
}

/// `None` when the whole activation of an int4 record fits in threadgroup
/// memory, the tile otherwise.
pub(super) fn int4_tile_for(name: &str, d_in: usize, gpr: usize, tile: usize) -> Result<Option<usize>> {
    // The kernel indexes `s0 = row * gpr` and `g = s0 + c / 128`: a gpr that
    // is not ceil(d_in / 128) reads another row's scales, silently.
    if gpr != d_in.div_ceil(128) {
        candle_core::bail!(
            "{name}: {gpr} groups a row for d_in {d_in}, the kernel wants {}",
            d_in.div_ceil(128)
        );
    }
    if d_in * 4 <= THREADGROUP_LIMIT {
        return Ok(None);
    }
    MetalTuning { q4_tile_cols: tile, ..Default::default() }.validate()?;
    Ok(Some(tile))
}

/// Refuses by name a device that offers less threadgroup memory than
/// [`THREADGROUP_LIMIT`], the bound every staging of this file is sized to.
pub(super) fn check_threadgroup_memory(dev: &candle_core::MetalDevice) -> Result<()> {
    use objc2_metal::MTLDevice as _;
    let offered = dev.device().as_ref().maxThreadgroupMemoryLength();
    if offered < THREADGROUP_LIMIT {
        candle_core::bail!(
            "the Metal device offers {offered} B of threadgroup memory; the kernels of this file \
             stage up to {THREADGROUP_LIMIT} B"
        );
    }
    Ok(())
}

/// Threads a threadgroup for a launch that asks `want`: `want` when the
/// pipeline takes it, else the largest `want / 2^k` it takes, down to one
/// SIMD-group, else a refusal by name.
///
/// Only for a kernel whose result does not depend on the group: each row is
/// the SIMD-group of its global thread index (`gid >> 5`), and the staging
/// loops over `threads_per_threadgroup`. Halving keeps every threadgroup
/// whole on a grid `want` tiled.
pub(super) fn group_within(pipe: &ComputePipeline, name: &str, want: usize) -> Result<usize> {
    let max = pipe.max_total_threads_per_threadgroup();
    let mut threads = want;
    while threads > max && threads > LANES {
        threads /= 2;
    }
    if threads > max {
        candle_core::bail!(
            "{name}: the pipeline allows {max} threads a threadgroup, fewer than one SIMD-group of {LANES}"
        );
    }
    Ok(threads)
}

/// `want` threads a threadgroup, refused by name when the pipeline takes
/// fewer: for a kernel whose indexing assumes that count.
pub(super) fn group_exactly(pipe: &ComputePipeline, name: &str, want: usize) -> Result<usize> {
    let max = pipe.max_total_threads_per_threadgroup();
    if max < want {
        candle_core::bail!("{name}: the launch needs {want} threads a threadgroup, the pipeline allows {max}");
    }
    Ok(want)
}

/// The matvec named by `LLVQ_METAL_KERNEL`, or the served one.
pub(super) fn kernel_from_env() -> Result<&'static str> {
    kernel_named(std::env::var("LLVQ_METAL_KERNEL").ok().as_deref())
}

/// The matvec a value of `LLVQ_METAL_KERNEL` names. An unknown name is refused:
/// falling back to the served kernel would label an A/B arm with a kernel it
/// did not run.
///
/// The names are a subset of what [`MetalTuning::validate`] accepts. `ilp` is
/// not one: `tv_tetra48_metal_ilp` gives a row 8 lanes, so on the 32-lane grid
/// every launch uses it would write past the output.
pub(super) fn kernel_named(v: Option<&str>) -> Result<&'static str> {
    const SERVED: &str = "tv_tetra48_metal_lut";
    Ok(match v {
        None | Some("lut") => SERVED,
        Some("c3") => "tv_tetra48_metal_c3",
        Some("ar") => "tv_tetra48_metal_ar",
        Some("u2") => "tv_tetra48_metal_u2",
        Some("plain") => "tv_tetra48_metal",
        Some(other) => candle_core::bail!(
            "LLVQ_METAL_KERNEL={other:?} is not a matvec of this file: lut, c3, ar, u2 or plain"
        ),
    })
}
