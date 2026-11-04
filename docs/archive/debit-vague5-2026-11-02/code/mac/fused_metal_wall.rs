// DRAFT, never compiled or run. Wave 5 deep dive, lead "mac", 2026-11-02, revised the same day.
//
// The host-side change that routes the int4 records past the 32 KB wall to
// `tv_q4_metal_tiled` in `llvq-llm/src/fused_metal.rs`, plus the two load-time
// options the Mac harness needs (host copies for the oracle, null kernels for
// the floor). Written as fragments against the file as it stands on branch
// `debit-trims` (line numbers below), for the branch that implements the
// audit's step 4. It touches the host only: `llvq-llm/kernels/tv_q4_h.metal`
// is not edited, because the llvq-tetra wheel ships a copy of it and checks
// its sha256 (`llvq_tetra/metal.py:62-79`).
//
// Exactness: a tile of a multiple of 256 columns is a multiple of 32 words, so
// lane l walks words l, l+32, ... in the untiled order (tv_q4_h.metal:152-170).
// Records that fit keep `tv_q4_metal` and their exact code path.

// --- near fused_metal.rs:80 ---------------------------------------------------

/// Columns `tv_q4_metal_tiled` stages a tile, for a record whose whole
/// activation does not fit in `THREADGROUP_LIMIT`.
///
/// 2,048 and not the 8,192 that would fill the limit, for residency: a tile of
/// 8,192 asks all 32,768 B, so likely one threadgroup fits a core, while 8 KB
/// leaves room for several. Pinning 18.7 KB of threadgroup memory already
/// capped residency on this M3 Max and cost +8 to +23 % (commit 82c029c).
/// llvq-tetra ships 2,048 as the value its kernels were validated with
/// (`llvq_tetra/integration.py:25-28`). The value moves no bit; prereg-mac-1
/// times 8,192 against it and moves the default only on a ratio range wholly
/// on one side.
pub const Q4_TILE_COLS: usize = 2048;

/// Load-time options of the Metal path. None of them changes a served bit.
/// Measurement harnesses set them; the served door takes the default.
#[derive(Debug, Clone, Copy)]
pub struct MetalTuning {
    /// Columns a tile for the records past the wall.
    pub q4_tile_cols: usize,
    /// Keep host copies of the decoder tables and of every Tetra stream, so
    /// `TetraMatvec::cpu_fwd` can serve as the oracle of a record gate. Costs
    /// the Tetra streams twice in host memory (about 0.75 GB at 4B). Never on
    /// the served door.
    pub keep_host: bool,
    /// Map every matvec and head pipeline to an empty twin with the same
    /// bindings (kernels/llvq_null.metal). The analog of `nullk`
    /// (docs/format-noyau.md section 6): it prices host encoding, launches and
    /// candle's own ops with their overlap intact. Tokens become id 0. Never
    /// on the served door.
    pub null_kernels: bool,
}

impl Default for MetalTuning {
    fn default() -> Self {
        Self { q4_tile_cols: Q4_TILE_COLS, keep_host: false, null_kernels: false }
    }
}

impl MetalTuning {
    /// Refuses a bad tile when the load starts, whether or not a record of
    /// this file needs it. A zero tile makes the kernel's `c0 += tile_cols`
    /// loop forever: a GPU hang, not an error.
    pub fn validate(&self) -> Result<()> {
        let c = self.q4_tile_cols;
        if c == 0 || c % 256 != 0 {
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
        Ok(())
    }
}

// --- source_of, fused_metal.rs:93-102 -----------------------------------------
//
//      "tv_q4_metal" | "tv_q4_metal_tiled" => Ok(SRC_Q4),
//      // Null twins: a fifth translation unit, so no shipped unit changes.
//      "tv_tetra48_metal_lut_null" | "tv_q4_metal_null" | "tv_q4_metal_tiled_null"
//          | "tv_emb_q4_metal_null" | "tv_q8_metal_null" => Ok(SRC_NULL),
//
//  with  const SRC_NULL: &str = include_str!("../kernels/llvq_null.metal");
//  Each null entry point keeps its twin's exact argument list and writes
//  `y[row] = 0.0f` from lane 0, nothing else.

// --- MetalRuntime gains two fields, set by `from_tables_tuned` ------------------

//  pub struct MetalRuntime { ..., q4_tile_cols: usize, null_kernels: bool, ... }

// --- pipeline(), fused_metal.rs:385: the null remap ---------------------------
//
//      let name: &'static str = match (self.null_kernels, name) {
//          (true, "tv_tetra48_metal_lut") => "tv_tetra48_metal_lut_null",
//          (true, "tv_q4_metal") => "tv_q4_metal_null",
//          (true, "tv_q4_metal_tiled") => "tv_q4_metal_tiled_null",
//          (true, "tv_emb_q4_metal") => "tv_emb_q4_metal_null",
//          (true, "tv_q8_metal") => "tv_q8_metal_null",
//          (_, n) => n, // the rotation and the gathers stay real
//      };

// --- from_tables_tuned, beside fused_metal.rs:1385 ------------------------------
//
// `from_tables` with the host copies `MetalRuntime::new(.., keep_host)` already
// knows how to keep (fused_metal.rs:257-296). `Tetra48Tables` packs prefixes
// and suffixes into u32 for the CUDA upload; the host reader indexes bytes, so
// they are unpacked the way `compose_c3` already is at :1406-1409.
//
//      pub fn from_tables_tuned(device: &Device, t: &Tetra48Tables, tile: usize,
//                               tuning: MetalTuning) -> Result<Self> {
//          tuning.validate()?;
//          let mut rt = Self::from_tables(device, t, tile)?;
//          if tuning.keep_host {
//              let bytes = |w: &[u32]| w.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
//              rt.tables.host = Some(HostTables {
//                  rows: t.rows.clone(),
//                  prefixes: bytes(&t.prefixes),
//                  branches: t.branches.clone(),
//                  suffixes: bytes(&t.suffixes),
//                  invnorm: t.invnorm.clone(),
//              });
//          }
//          rt.q4_tile_cols = tuning.q4_tile_cols;
//          rt.null_kernels = tuning.null_kernels;
//          Ok(rt)
//      }
//
// `upload_stream` already copies each stream to its `HostProj` when
// `tables.host.is_some()` (:375-380), so nothing else changes for Tetra.

// --- MetalInt4Proj gains one field --------------------------------------------

pub struct MetalInt4Proj {
    pub name: String,
    pub d_out: usize,
    pub d_in: usize,
    gpr: usize,
    /// `None`: `tv_q4_metal` stages the whole activation.
    /// `Some(c)`: `tv_q4_metal_tiled` stages `c` columns a tile.
    tile_cols: Option<usize>,
    wq: Arc<Buffer>,
    scales: Arc<Buffer>,
    biases: Arc<Buffer>,
}

impl MetalInt4Proj {
    /// What the load line and the binding test read.
    pub fn tile_cols(&self) -> Option<usize> {
        self.tile_cols
    }
}

// --- the routing, used by upload_int4_tiled -------------------------------------

/// `None` when the whole activation fits, the tile otherwise. The tile was
/// validated when the load started (`MetalTuning::validate`).
fn int4_tile_for(name: &str, d_in: usize, gpr: usize, tile: usize) -> Result<Option<usize>> {
    // The kernel indexes `s0 = row * gpr` and `g = s0 + c / 128`: a gpr that
    // is not ceil(d_in / 128) reads another row's scales, silently.
    if gpr != d_in.div_ceil(128) {
        candle_core::bail!("{name}: {gpr} groups a row for d_in {d_in}, the kernel wants {}", d_in.div_ceil(128));
    }
    if d_in * 4 <= THREADGROUP_LIMIT {
        return Ok(None); // the served untiled path, unchanged
    }
    debug_assert!(tile > 0 && tile % 256 == 0 && tile * 4 <= THREADGROUP_LIMIT, "validate() first");
    Ok(Some(tile))
}

// --- upload_int4_tiled, replacing fused_metal.rs:970-977 ------------------------
//
//      pub fn upload_int4(&self, name, d_out, d_in, gpr, wq, scales, biases) -> Result<MetalInt4Proj> {
//          self.upload_int4_tiled(name, d_out, d_in, gpr, wq, scales, biases, self.q4_tile_cols)
//      }
//
//      /// The primitive the binding test drives at any tile.
//      pub fn upload_int4_tiled(&self, name, d_out, d_in, gpr, wq, scales, biases,
//                               tile: usize) -> Result<MetalInt4Proj> {
//          ... the d_out and d_in checks of :957-969, unchanged ...
//          MetalTuning { q4_tile_cols: tile, ..Default::default() }.validate()?;
//          let tile_cols = int4_tile_for(name, d_in, gpr, tile)?;
//          ... the length checks of :978-984, unchanged ...
//          Ok(MetalInt4Proj { ..., tile_cols, ... })
//      }

// --- Int4Op::metal_fwd, replacing fused_metal.rs:1031 and :1045 ---------------
//
//      let (kname, staged_cols) = match p.tile_cols {
//          None => ("tv_q4_metal", p.d_in),
//          Some(c) => ("tv_q4_metal_tiled", c),
//      };
//      let pipe = self.rt.pipeline(kname)?;
//      ... buffers 0..=6 exactly as today ...
//      if let Some(c) = p.tile_cols {
//          set_param(&enc, 7, c as u32);           // never leave it unbound
//      }
//      // The tile, never d_in: d_in * 4 is above the limit on this path.
//      enc.set_threadgroup_memory_length(0, staged_cols * DType::F32.size_in_bytes());

// --- load_resolved_tuned, beside fused_metal.rs:1448 ----------------------------
//
// `load_resolved` becomes `load_resolved_tuned(.., MetalTuning::default())`, so
// the served door is byte-for-byte the tuned path at its default. At :1500 the
// runtime is built by `from_tables_tuned(device, &tables, 64, tuning)`.
//
// After the int4 loop (:1539-1553), one provenance line. Gate 4 of
// prereg-mac-1 reads it: it proves the routing reached the op rather than only
// the parser.
//
//      let tiled = uploaded.iter().filter(|p| p.tile_cols().is_some()).count();
//      println!(
//          "int4: {} records, {tiled} through tv_q4_metal_tiled at {} columns a tile, {} through tv_q4_metal",
//          uploaded.len(), tuning.q4_tile_cols, uploaded.len() - tiled
//      );
//      if tuning.keep_host { println!("host copies kept: this load is an oracle's, never the served door"); }
//      if tuning.null_kernels { println!("NULL KERNELS: no weight is read, tokens are meaningless"); }

// --- host tests that need no device (llvq-llm, any machine) -------------------

#[cfg(test)]
mod tile_tests {
    use super::*;

    #[test]
    fn records_that_fit_keep_the_served_kernel() {
        // v_proj and o_proj of all three sealed sizes, and the limit itself.
        for d_in in [2560usize, 4096, 5120, 8192] {
            assert_eq!(int4_tile_for("x", d_in, d_in.div_ceil(128), Q4_TILE_COLS).unwrap(), None);
        }
    }

    #[test]
    fn the_sealed_down_proj_widths_route_to_the_tiled_kernel() {
        for d_in in [9728usize, 12288, 17408, 25600] {
            for tile in [Q4_TILE_COLS, 8192] {
                assert_eq!(int4_tile_for("x", d_in, d_in.div_ceil(128), tile).unwrap(), Some(tile));
            }
        }
    }

    #[test]
    fn a_bad_tile_is_refused_by_name_at_the_start() {
        for bad in [0usize, 128, 300, 16384] {
            let e = MetalTuning { q4_tile_cols: bad, ..Default::default() }.validate().unwrap_err().to_string();
            assert!(e.contains(&bad.to_string()), "{e}");
        }
        for good in [256usize, 1024, 2048, 8192] {
            MetalTuning { q4_tile_cols: good, ..Default::default() }.validate().unwrap();
        }
    }

    #[test]
    fn a_wrong_group_count_is_refused() {
        assert!(int4_tile_for("v", 2560, 19, Q4_TILE_COLS).is_err());
    }

    #[test]
    fn the_default_tuning_is_the_served_door() {
        let t = MetalTuning::default();
        assert_eq!((t.q4_tile_cols, t.keep_host, t.null_kernels), (2048, false, false));
    }
}
