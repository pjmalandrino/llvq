//! The tile: how many blocks of the activation one CTA stages in shared memory.
//!
//! `TILE_BLOCKS` is a **host-injected constant**, not a property of any format.
//! It is `#define`d into the kernel source before NVRTC sees it, and the same
//! number sizes the dynamic shared allocation at launch. Zero bits on disk,
//! bit-identical output whatever its value — it is a knob, and until
//! 2026-09-09 **no journal had ever varied it**.
//!
//! Varying it explained the whole two-card discrepancy. On 2026-09-08 the
//! served Tetra decode measured 1.4693× Planes14 on sm_120 and 0.5824× on
//! sm_89, and the table floor of 2026-09-09 refuted both the capacity and the
//! clock explanations — the decoder table is *faster* on Blackwell at every
//! footprint. What was left is geometry: shared memory and L1 are the same
//! 102,400 B of SRAM per SM, so the activation tile **steals L1 from the
//! decoder table**. At tile 128, six CTAs hold 73,728 B and leave ~28 KB of L1
//! to a table that wants 18 KiB. The sweep (`docs/mesures/tile-sweep-2026-09-09.txt`):
//!
//! | R = (v3g − nullk)/(planes14 − nullk) | 128 | 64 | 32 |
//! |---|---|---|---|
//! | sm_89 (L40S) | 0.5784 | **0.4668** | 0.4704 |
//! | sm_120 (RTX PRO 6000) | 1.4643 | 1.4472 | **0.8491** |
//!
//! There is no single best value: the optimum depends on the architecture.
//!
//! ## What this module ships, and what it deliberately does not
//!
//! It ships the **mechanism** — one parser, one table, one resolved value with
//! its provenance. It does **not** ship the policy: with `LLVQ_TILE_BLOCKS`
//! unset the answer is [`crate::TILE_BLOCKS`] = 128, the value every published
//! number was measured at. The two rows of [`TILE_BY_SM`] were measured on
//! `bin/f1rankfloor`, a *synthetic* bench with its own shapes, not on the
//! served kernel with the real model's `nblocks` per projection. Promoting a
//! synthetic optimum to a served default without measuring it on the served
//! path is the class of error this repository keeps catching. F1d measures all
//! three columns on the real path; the operator flips the default afterwards,
//! by changing one arm of [`resolve_with`].
//!
//! ## Why the provenance travels with the value
//!
//! A default that depends on the card makes a figure irreproducible unless the
//! card is named. So [`Tile`] carries [`TileSource`] and every bench prints it,
//! the way `bin/f1rankfloor` already prints its ⚠️ line. A number measured at
//! a tile nobody can reconstruct is not a measurement.
//!
//! Portable on purpose, like [`crate::arms`] and [`crate::shared`]: the
//! refusals below are exactly the kind of logic a mutation breaks in silence,
//! the development Mac has no CUDA, and a knob that reached a rented card
//! wrong would cost a billed job to notice.

use crate::TILE_BLOCKS;

/// Smallest tile that is still *this* kernel.
///
/// Every arm walks its blocks as `for (j = jlo + lane; j < jhi; j += 32)`.
/// Below 32 blocks in flight some lanes never enter the loop, which is a
/// different kernel with a different occupancy — not a smaller tile — and
/// comparing it to the others would compare two things.
pub const TILE_MIN: usize = 32;

/// Largest tile the shared budget admits **at the 24-float stride**:
/// `512 · 24 · 4` = 49,152 B, exactly
/// `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK` on every card measured
/// here. Past it the launch fails at run time on the card, which is the worst
/// place to find out.
///
/// The stride is not universal. The A3 arms `pad` and `mr2p` stage at
/// [`crate::occ::XS_PAD`] = 28 floats a block (`occ::XS_STRIDE`), where the
/// same 512 would ask for 57,344 B and be refused by the driver. Those arms
/// are refused off the served tile outright — see
/// [`crate::occ::refuse_off_served_tile`] — so this bound is the one that
/// applies to every launch that can actually happen.
pub const TILE_MAX: usize = 512;

/// Measured optima, one row per architecture: `(sm, blocks, journal)`.
///
/// `sm` is `major · 10 + minor`, the form NVIDIA prints and the form
/// `LLVQ_NVRTC_ARCH=compute_NN` names.
///
/// **No row is interpolated, and no row is inferred from a neighbour.** A card
/// absent from this table gets [`TILE_BLOCKS`], never a guess drawn through
/// two points — the residency trade-off depends on the SRAM budget, the table
/// footprint and the L1 policy of that specific architecture, and two measured
/// points are not a law.
pub const TILE_BY_SM: [(i32, usize, &str); 2] = [
    (89, 64, "docs/mesures/tile-sweep-2026-09-09.txt"),
    (120, 32, "docs/mesures/tile-sweep-2026-09-09.txt"),
];

/// What `LLVQ_TILE_BLOCKS` asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileRequest {
    /// Unset: the served constant, on every card.
    Served,
    /// `auto`: the measured row for this card, or the served constant when
    /// this card has no measured row.
    Auto,
    /// An explicit count, already validated.
    Fixed(usize),
}

/// Where the resolved tile came from. Printed beside the figure it produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileSource {
    /// `LLVQ_TILE_BLOCKS=<n>`.
    Env,
    /// `LLVQ_TILE_BLOCKS=auto` and this card has a measured row.
    Table(i32),
    /// `LLVQ_TILE_BLOCKS=auto` and this card does not. The fallback is loud on
    /// purpose: it is the case where an operator asked for the optimum and got
    /// the served value instead.
    AutoNoRow(i32),
    /// Unset — the served constant, the value every published number used.
    Served,
}

/// A tile and where it came from. The two never travel apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tile {
    pub blocks: usize,
    pub source: TileSource,
}

impl Tile {
    /// Bytes of dynamic shared memory one CTA stages at the 24-float stride:
    /// `blocks · DIM · 4`.
    ///
    /// Through [`crate::occ::XS_DIM`] rather than `llvq_core::DIM`, for the
    /// reason stated there: `llvq-core` is a Linux-only dependency of this
    /// crate and this module is portable. That literal is already pinned to
    /// the real dimension by `occ`'s own test, so this reuses one pin instead
    /// of adding a second literal that could drift from it.
    ///
    /// Every launch that reads a resolved tile forms its shared bytes here,
    /// and that is the point: the `#define` and the launch parameter were two
    /// independent expressions over one `const`, safe only while the const
    /// could not move.
    ///
    /// 🕳️ It is **not** the only place the product is formed in this crate.
    /// [`crate::occ::shared_bytes`] forms it again, clamping at
    /// [`crate::TILE_BLOCKS`] directly, and it is the live launch argument of
    /// the A3 section (`planesbench` :3310, :3318, :3328). Threading the
    /// resolved tile through `occ` would not be enough either: `planes_occ.cu`
    /// branches on `nblocks <= TILE_BLOCKS` at :259 and :416, so the `pers`
    /// arm takes a *different code path* below the tile and would report two
    /// algorithms under one name. That section is off by default
    /// (`occ::parse_seg_arms(None)` is empty) and off F1d's path, so it is
    /// refused off the served tile rather than resized — see
    /// [`crate::occ::refuse_off_served_tile`].
    pub fn shared_bytes(&self) -> u32 {
        (self.blocks * crate::occ::XS_DIM * 4) as u32
    }

    /// The `#define` line handed to NVRTC. Same source as [`Self::shared_bytes`].
    pub fn define(&self) -> String {
        format!("#define TILE_BLOCKS {}u\n", self.blocks)
    }

    /// Refuse a source whose tile is not this one.
    ///
    /// The invariant nothing else can hold: **the tile a report prints is the
    /// tile NVRTC compiled**. The value travels as a `#define` into a text of
    /// tens of thousands of lines assembled from thirty-odd files by the host,
    /// and a stale, duplicated or missing define changes the kernel's staging
    /// while every number keeps the old label. None of that is a compile
    /// error, on either side — a second `#define` of the same macro is a
    /// warning NVRTC does not fail on, and a report reads its own variable.
    ///
    /// Exactly one define, and it is this tile. Checked on the text handed to
    /// NVRTC, after assembly, which is the only place both facts are true at
    /// once.
    pub fn assert_defined_in(&self, src_text: &str) {
        let n = src_text.matches("#define TILE_BLOCKS").count();
        assert_eq!(
            n, 1,
            "the assembled source carries {n} `#define TILE_BLOCKS`, not 1: \
             the tile NVRTC compiles would not be the tile this run reports"
        );
        assert!(
            src_text.contains(self.define().trim_end()),
            "the assembled source defines a tile that is not {} ({}): \
             every figure of this run would carry the wrong tile",
            self.blocks,
            self.provenance()
        );
    }

    /// One line for the report header, naming the value and its provenance.
    pub fn provenance(&self) -> String {
        match self.source {
            TileSource::Served => format!("tile {} (served constant)", self.blocks),
            TileSource::Env => format!("tile {} (⚠️ LLVQ_TILE_BLOCKS overrides the served {TILE_BLOCKS})", self.blocks),
            TileSource::Table(sm) => format!(
                "tile {} (⚠️ measured optimum for sm_{sm}, not the served {TILE_BLOCKS})",
                self.blocks
            ),
            TileSource::AutoNoRow(sm) => format!(
                "tile {} (⚠️ LLVQ_TILE_BLOCKS=auto, but sm_{sm} has no measured row: served constant)",
                self.blocks
            ),
        }
    }
}

/// Parse one `LLVQ_TILE_BLOCKS` value.
///
/// Pure, so the refusals are tested without touching the process environment —
/// which is global, and which `cargo test` runs in parallel.
///
/// Refused **by name**, never silently defaulted: the `LLVQ_FUSED_LAYOUT` rule,
/// and the same one `gpu::arch()` applies to `LLVQ_NVRTC_ARCH`.
pub fn parse_request(s: &str) -> Result<TileRequest, String> {
    if s == "auto" {
        return Ok(TileRequest::Auto);
    }
    let n: usize = s.parse().map_err(|e| {
        format!("LLVQ_TILE_BLOCKS={s:?}: expected `auto` or an integer ({e})")
    })?;
    if !(TILE_MIN..=TILE_MAX).contains(&n) || !n.is_power_of_two() {
        return Err(format!(
            "LLVQ_TILE_BLOCKS={n}: expected a power of two in {TILE_MIN}..={TILE_MAX}; \
             below {TILE_MIN} the lane stride idles lanes and it is another kernel, \
             above {TILE_MAX} the tile exceeds the per-block shared budget"
        ));
    }
    Ok(TileRequest::Fixed(n))
}

/// What the environment asks for, pinned once per process.
///
/// Pinned for the reason `gpu::arch()` is: a value read twice could differ
/// between the `#define` and the launch, and those two must be the same number
/// or the kernel overruns its staging area.
pub fn request() -> TileRequest {
    use std::sync::OnceLock;
    static ONCE: OnceLock<TileRequest> = OnceLock::new();
    *ONCE.get_or_init(|| match std::env::var("LLVQ_TILE_BLOCKS") {
        Ok(s) => parse_request(&s).unwrap_or_else(|e| panic!("{e}")),
        Err(_) => TileRequest::Served,
    })
}

/// `sm` as NVIDIA prints it, from the driver's `(major, minor)`.
pub fn sm_of(compute_cap: (i32, i32)) -> i32 {
    compute_cap.0 * 10 + compute_cap.1
}

/// Resolve a request against a card. Pure: the whole policy in one function.
///
/// **This is the line that flips.** When F1d has measured the per-card optima
/// on the served path, [`TileRequest::Served`] stops meaning [`TILE_BLOCKS`]
/// and starts meaning the table — one arm, one commit, one journal.
pub fn resolve_with(req: TileRequest, compute_cap: (i32, i32)) -> Tile {
    let sm = sm_of(compute_cap);
    match req {
        TileRequest::Served => Tile {
            blocks: TILE_BLOCKS,
            source: TileSource::Served,
        },
        TileRequest::Fixed(n) => Tile {
            blocks: n,
            source: TileSource::Env,
        },
        TileRequest::Auto => match TILE_BY_SM.iter().find(|&&(s, _, _)| s == sm) {
            Some(&(_, blocks, _)) => Tile {
                blocks,
                source: TileSource::Table(sm),
            },
            None => Tile {
                blocks: TILE_BLOCKS,
                source: TileSource::AutoNoRow(sm),
            },
        },
    }
}

/// The tile this process runs at on this card.
pub fn resolve(compute_cap: (i32, i32)) -> Tile {
    resolve_with(request(), compute_cap)
}

#[cfg(test)]
mod tests {
    use super::*;

    const L40S: (i32, i32) = (8, 9);
    const BLACKWELL: (i32, i32) = (12, 0);
    const A100: (i32, i32) = (8, 0);

    #[test]
    fn sm_is_major_ten_plus_minor() {
        assert_eq!(sm_of(L40S), 89);
        assert_eq!(sm_of(BLACKWELL), 120);
        assert_eq!(sm_of(A100), 80);
    }

    /// Every row is a tile this repository's kernels can actually run, and no
    /// architecture appears twice — a duplicate row would make `find` return
    /// whichever came first, silently.
    #[test]
    fn every_measured_row_is_a_runnable_tile() {
        for &(sm, blocks, journal) in &TILE_BY_SM {
            assert!(
                (TILE_MIN..=TILE_MAX).contains(&blocks) && blocks.is_power_of_two(),
                "sm_{sm}: {blocks} is not a runnable tile"
            );
            assert!(journal.starts_with("docs/mesures/"), "sm_{sm}: a row without a journal");
        }
        for (i, a) in TILE_BY_SM.iter().enumerate() {
            for b in &TILE_BY_SM[i + 1..] {
                assert_ne!(a.0, b.0, "sm_{} appears twice", a.0);
            }
        }
    }

    /// The policy, stated as a test: an unset environment is the served
    /// constant on **every** card, including the two that have a measured row.
    /// This is what makes today's figures comparable to every published one.
    #[test]
    fn unset_is_the_served_constant_on_every_card() {
        for cap in [L40S, BLACKWELL, A100] {
            let t = resolve_with(TileRequest::Served, cap);
            assert_eq!(t.blocks, TILE_BLOCKS);
            assert_eq!(t.source, TileSource::Served);
        }
        assert_eq!(TILE_BLOCKS, 128, "the served constant moved without this test moving");
    }

    #[test]
    fn auto_reads_the_measured_row_and_never_interpolates() {
        assert_eq!(resolve_with(TileRequest::Auto, L40S).blocks, 64);
        assert_eq!(resolve_with(TileRequest::Auto, L40S).source, TileSource::Table(89));
        assert_eq!(resolve_with(TileRequest::Auto, BLACKWELL).blocks, 32);
        assert_eq!(resolve_with(TileRequest::Auto, BLACKWELL).source, TileSource::Table(120));
        // sm_80 sits between two measured rows. It gets neither, and it is not
        // averaged into 48: it gets the served constant, and says so.
        let a100 = resolve_with(TileRequest::Auto, A100);
        assert_eq!(a100.blocks, TILE_BLOCKS);
        assert_eq!(a100.source, TileSource::AutoNoRow(80));
    }

    #[test]
    fn an_explicit_value_wins_over_the_card() {
        for cap in [L40S, BLACKWELL, A100] {
            let t = resolve_with(TileRequest::Fixed(256), cap);
            assert_eq!(t.blocks, 256);
            assert_eq!(t.source, TileSource::Env);
        }
    }

    /// The `#define` and the launch parameter are the same number, formed from
    /// the same value. Decoupling them is a shared-memory overrun with no
    /// diagnostic, which is why they share one owner.
    #[test]
    fn the_define_and_the_shared_bytes_come_from_one_value() {
        for blocks in [32usize, 64, 128, 256, 512] {
            let t = Tile { blocks, source: TileSource::Env };
            assert_eq!(t.define(), format!("#define TILE_BLOCKS {blocks}u\n"));
            assert_eq!(t.shared_bytes(), (blocks * 24 * 4) as u32);
        }
        // The largest admissible tile is exactly the per-block shared budget
        // every card here reports; one step further would fail on the card.
        assert_eq!(
            Tile { blocks: TILE_MAX, source: TileSource::Env }.shared_bytes(),
            49_152
        );
    }

    /// No binary may fork the knob again.
    ///
    /// 🕳️ `request()` reads the environment through a `OnceLock`, so a binary
    /// that never calls it never reads `LLVQ_TILE_BLOCKS` — and says nothing.
    /// On 2026-09-10 four still did: `nullkbench` and `f1floorbench` wrote
    /// their own `#define` from the constant, `matvec` and `graphbench` each
    /// carried a private `const TILE_BLOCKS = 128`. `LLVQ_TILE_BLOCKS=32
    /// nullkbench` printed a tile-128 floor with no line saying so, and the
    /// journal records nullk moving 6.1 % between those two tiles — a wrong
    /// subtrahend, carried across processes into the very same-head ratio hard
    /// rule 4 exists to protect.
    ///
    /// Nothing could detect that: silence is what an unread `OnceLock` does.
    /// So the tree is the detector. Every binary spends its tile through
    /// [`Tile::define`] and [`Tile::shared_bytes`], and a new one that writes
    /// the `#define` itself fails here rather than on a card.
    #[test]
    fn no_binary_writes_its_own_tile() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/bin");
        let mut seen = 0;
        for e in std::fs::read_dir(&dir).expect("src/bin") {
            let path = e.expect("dir entry").path();
            if path.extension().and_then(|x| x.to_str()) != Some("rs") {
                continue;
            }
            let text = std::fs::read_to_string(&path).expect("read bin");
            seen += 1;
            for banned in ["#define TILE_BLOCKS ", "const TILE_BLOCKS"] {
                assert!(
                    !text.contains(banned),
                    "{}: writes `{banned}` itself. Spend the tile through \
                     `tile::resolve` — a private one is invisible to \
                     LLVQ_TILE_BLOCKS and to every reader of the output",
                    path.display()
                );
            }
        }
        assert!(seen >= 8, "only {seen} binaries scanned: the directory moved");
    }

    #[test]
    fn every_refusal_names_the_variable() {
        for bad in ["0", "16", "1024", "96", "-64", "", "yes", "128b", "AUTO"] {
            let e = parse_request(bad).expect_err("{bad} should be refused");
            assert!(e.contains("LLVQ_TILE_BLOCKS"), "{bad}: {e}");
        }
        assert_eq!(parse_request("auto"), Ok(TileRequest::Auto));
        for good in [32usize, 64, 128, 256, 512] {
            assert_eq!(parse_request(&good.to_string()), Ok(TileRequest::Fixed(good)));
        }
    }

    /// The guard that makes a stale, missing or duplicated define loud. The
    /// defect it is for cannot fail to compile on either side of the boundary.
    #[test]
    fn the_source_must_carry_this_tile_and_only_this_tile() {
        let t = Tile { blocks: 64, source: TileSource::Env };
        t.assert_defined_in("#define TILE_BLOCKS 64u\n__global__ void k() {}");
        // Same shape as `fused_segment.rs`: the four refusals below are the
        // point of the test, so their backtraces are not test output.
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        // Assembled with a leftover define from another part.
        let two = std::panic::catch_unwind(|| {
            t.assert_defined_in("#define TILE_BLOCKS 64u\n#define TILE_BLOCKS 128u\n")
        });
        // The host resolved 64 and the source says 128 — the exact shape of a
        // knob threaded to one call site and not the other.
        let stale = std::panic::catch_unwind(|| t.assert_defined_in("#define TILE_BLOCKS 128u\n"));
        // No define at all: `matvec.cu` `#error`s, but only once it reaches
        // NVRTC — on a rented card, in a billed job.
        let none = std::panic::catch_unwind(|| t.assert_defined_in("__global__ void k() {}"));
        // 64 must not be satisfied by a source defining 640.
        let wider = std::panic::catch_unwind(|| t.assert_defined_in("#define TILE_BLOCKS 640u\n"));
        std::panic::set_hook(prev);
        assert!(two.is_err(), "two defines accepted");
        assert!(stale.is_err(), "a stale define accepted");
        assert!(none.is_err(), "a missing define accepted");
        assert!(wider.is_err(), "a prefix match accepted: the trailing `u` is what separates 64 from 640");
    }

    /// Provenance is never empty and never silent about a departure from the
    /// served constant: a figure measured at another tile must say so on the
    /// same line, or it is not reconstructible.
    #[test]
    fn a_tile_that_is_not_the_served_one_says_so() {
        assert!(!resolve_with(TileRequest::Served, L40S).provenance().contains('⚠'));
        for t in [
            resolve_with(TileRequest::Auto, L40S),
            resolve_with(TileRequest::Auto, A100),
            resolve_with(TileRequest::Fixed(32), L40S),
        ] {
            assert!(t.provenance().contains('⚠'), "{}", t.provenance());
        }
    }
}
