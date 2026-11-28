//! From stored records to kernel words: the packers, the Golay70 device tables
//! and [`Transcoder`], which turns one lattice matrix into one [`HostStream`].

use super::*;

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
