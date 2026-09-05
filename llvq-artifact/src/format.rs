//! Reading and writing the `LVQ1` … `LVQ5` stream.
//!
//! ## What a matrix needs, and why each part is there
//!
//! | field | why it cannot be dropped |
//! |---|---|
//! | index + gain per block | the code itself |
//! | one scale per output row | the gain code is relative to it |
//! | the matrix's gain centroids | fitted per matrix, not global |
//! | the rotation seed | codes live in the **rotated** basis |
//! | the tail columns | `KeepExact` leaves them unquantized |
//!
//! `every_stored_field_is_load_bearing` in the test suite corrupts each of
//! them in turn and demands the reconstruction move.
//!
//! ## Bit-exactness, and where the f64s are
//!
//! Scales and centroids are stored in **f64**, not f16. `reconstruct` computes
//! `centroids[g] * row_scale` in f64; storing a rounded scale would change
//! that product and the decoded weights would differ from the evaluated ones
//! in the last bits. That costs 64 bits per output row instead of 16 —
//! 0.0146 bits/weight on Qwen3-4B — and it is **counted**, not waved away.
//! Making the scales exactly f32-representable at fit time would recover it;
//! that is an optimization, not a correctness fix, and it is not done here.
//!
//! ## Two code kinds, one record shape (`LVQ5`)
//!
//! An index has always been a v1 ball index — 47 or 48 bits, one of 383
//! classes. Trio ([`llvq_search::trio`]) is a second map from 47 bits to Λ₂₄
//! with the same width and none of the same meaning: a Trio word read as a
//! ball index is in range, decodes to a lattice point, and is wrong. So the
//! **file** says which map it uses — [`CodeKind`] in the v5 header, beside a
//! second fingerprint for the Trio map — and every reader of a record
//! consults the kind before it trusts any width ([`read_matrix_raw_for_kind`]).
//! The record itself does not change shape: a Trio record carries
//! [`TRIO_SHELL_CAP`] in its `shell_cap` field and one gain bit, so a v4 tool
//! copying records byte for byte still copies them right.

use crate::{Error, Result};
use core::fmt;
use llvq_core::{Point, DIM};
use llvq_quant::quantizer::{index_bits, reconstruct_shape_gain, BlockCode};
use llvq_search::index::Indexer;
use llvq_search::pack::{BitReader, BitWriter};
use llvq_search::trio::{Trio, LABEL_BITS};
use std::io::{Read, Write};

/// Format identifier of [`DEFAULT_VERSION`], what [`ArtifactWriter::new`]
/// emits.
///
/// `LVQ1` held quantized projections and nothing else, so a file needed the
/// original checkpoint beside it to run. `LVQ2` adds the raw tensors and the
/// blobs that make it self-contained. `LVQ3` tags each raw tensor with its
/// encoding so the embedding can be carried group-affine quantized instead of
/// f16 (see [`crate::sealed`]). `LVQ4` appends the writer's codebook
/// fingerprint to the header (see [`crate::codebook`]), the first field of
/// the whole format that describes what an index *means* rather than how
/// wide it is. `LVQ5` appends the Trio fingerprint and the file's
/// [`CodeKind`]: which of the two maps its indices are read through. All
/// five are readable; `LVQ4` is written for Ball, `LVQ5` for Trio. The
/// matrix records themselves are identical across versions — which is what
/// lets a tool copy them between two files of different versions untouched.
pub const MAGIC: &[u8; 4] = MAGIC_V4;
pub const MAGIC_V5: &[u8; 4] = b"LVQ5";
pub const MAGIC_V4: &[u8; 4] = b"LVQ4";
pub const MAGIC_V3: &[u8; 4] = b"LVQ3";
pub const MAGIC_V2: &[u8; 4] = b"LVQ2";
pub const MAGIC_V1: &[u8; 4] = b"LVQ1";

/// The newest version this crate reads and writes.
pub const VERSION: u32 = 5;

/// The version [`ArtifactWriter::new`] emits.
///
/// Deliberately not [`VERSION`]: a Ball file gains nothing from a v5 header,
/// and holding the default at 4 keeps every Ball file — the served 4B's
/// path included — byte-identical to what the same writer produced before
/// Trio existed (`the_default_writer_still_writes_v4`). A Trio file cannot
/// be a v4 file, and [`ArtifactWriter::with_kind`] picks its version from
/// the kind.
pub const DEFAULT_VERSION: u32 = 4;

/// First version whose header carries a codebook fingerprint.
///
/// Files below it were written before the field existed and are read without
/// the check — grandfathered deliberately: the published Qwen3-4B artifact is
/// `LVQ2`, and a reader that refused it would be worse than the hole it
/// closes. What protects *those* files is the pinned fingerprint in the test
/// suite, not a field they cannot have.
pub const FIRST_FINGERPRINTED_VERSION: u32 = 4;

/// First version whose header carries a [`CodeKind`] and the Trio
/// fingerprint. Every file below it is a Ball file: there was nothing else
/// to be.
pub const FIRST_KINDED_VERSION: u32 = 5;

/// What a Trio record carries in its `shell_cap` field.
///
/// The field is not read for Trio; 12 keeps `index_bits` at 47 so a reader
/// that ignored the kind would at least fail loudly on the fingerprint, not
/// silently on the width — its 47-bit reads would stay aligned with the
/// stream, and every one of them would be an index into the wrong map. The
/// reader consults the kind BEFORE any width ([`read_matrix_raw_for_kind`]),
/// takes Trio's width from [`LABEL_BITS`], and refuses a Trio record whose
/// field holds anything else: such a record was never written by this crate.
pub const TRIO_SHELL_CAP: u32 = 12;

/// Which map turns a file's indices into lattice points.
///
/// Stored as a `u32` tag in the v5 header (`0 = Ball`, `1 = Trio`); a tag
/// this build does not know is [`Error::UnknownCodeKind`], refused at the
/// header. Every file before v5 is `Ball`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CodeKind {
    /// The v1 ball: 383 classes of `Λ₂₄(13)`, indexed by
    /// [`llvq_search::index::Indexer`], fingerprinted by
    /// [`crate::codebook_fingerprint`].
    Ball,
    /// The three-section trellis word of [`llvq_search::trio`], fingerprinted
    /// by [`crate::codebook::trio_fingerprint`].
    Trio,
}

impl CodeKind {
    /// The header tag.
    pub const fn tag(self) -> u32 {
        match self {
            Self::Ball => 0,
            Self::Trio => 1,
        }
    }

    /// The kind a header tag names, or [`Error::UnknownCodeKind`].
    pub fn from_tag(tag: u32) -> Result<Self> {
        match tag {
            0 => Ok(Self::Ball),
            1 => Ok(Self::Trio),
            _ => Err(Error::UnknownCodeKind { tag }),
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Ball => "Ball",
            Self::Trio => "Trio",
        }
    }

    /// The version [`ArtifactWriter::with_kind`] emits for this kind:
    /// [`DEFAULT_VERSION`] for Ball, [`FIRST_KINDED_VERSION`] for Trio.
    pub const fn default_version(self) -> u32 {
        match self {
            Self::Ball => DEFAULT_VERSION,
            Self::Trio => FIRST_KINDED_VERSION,
        }
    }
}

impl fmt::Display for CodeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// The map of one [`CodeKind`], built once per process and shared across
/// matrices — building either enumerates tables that have no business being
/// rebuilt per record (383 classes for the ball, a 16 KiB table and a
/// trellis, all re-derived and asserted, for Trio).
pub enum Codebook {
    Ball(Box<Indexer>),
    Trio(Box<Trio>),
}

impl Codebook {
    pub fn new(kind: CodeKind) -> Self {
        match kind {
            CodeKind::Ball => Self::Ball(Box::new(Indexer::new())),
            CodeKind::Trio => Self::Trio(Box::new(Trio::new())),
        }
    }

    pub fn kind(&self) -> CodeKind {
        match self {
            Self::Ball(_) => CodeKind::Ball,
            Self::Trio(_) => CodeKind::Trio,
        }
    }

    /// The index of a point, or `None` for a point the map has no word for.
    /// A Trio word comes back with its gain bit (bit 47) clear.
    pub fn encode(&self, point: &Point) -> Option<u64> {
        match self {
            Self::Ball(ix) => ix.encode(point),
            Self::Trio(trio) => trio.encode(point),
        }
    }

    /// The point of a stored `(index, gain)` pair.
    ///
    /// For Trio the pair is put back together as the 48-bit word the kernel
    /// will read, `index | gain << 47`, before decoding — the gain bit is
    /// opaque to [`Trio::decode`], but the word is the unit of the format,
    /// and a reader that assembled it with the gain at any other bit would
    /// decode a different point (`the_gain_bit_sits_at_bit_47_on_disk`).
    pub fn decode(&self, index: u64, gain: u32) -> Option<Point> {
        match self {
            Self::Ball(ix) => ix.decode(index),
            Self::Trio(trio) => Some(trio.decode(index | u64::from(gain) << LABEL_BITS)),
        }
    }
}

/// One quantized matrix, everything a decoder needs.
pub struct QuantizedMatrix {
    pub name: String,
    pub d_out: usize,
    pub d_in: usize,
    /// Row-major `d_out × (d_in / 24)`.
    pub codes: Vec<BlockCode>,
    /// One per output row, in the rotated basis.
    pub row_scales: Vec<f64>,
    /// Gain levels fitted to this matrix, relative to the row scale.
    pub centroids: Vec<f64>,
    /// Seed of the incoherence rotation, or `None` for the natural basis.
    pub rotation_seed: Option<u64>,
    /// Shell cap of the direction code, which sets the index width for a
    /// Ball matrix; [`TRIO_SHELL_CAP`] on a Trio matrix, where it is a
    /// sentinel and not a cap.
    pub shell_cap: u32,
    /// Trailing columns kept at full precision, `d_out × (d_in % 24)`
    /// row-major, in the rotated basis.
    pub tail: Vec<f64>,
}

impl QuantizedMatrix {
    fn nblocks(&self) -> usize {
        self.d_in / DIM
    }

    fn gain_bits(&self) -> u32 {
        self.centroids.len().next_power_of_two().trailing_zeros()
    }

    fn index_bits(&self) -> u32 {
        index_bits(self.shell_cap)
    }

    /// Bits this matrix occupies in the stream.
    pub fn bits(&self) -> u64 {
        let per_block = (self.index_bits() + self.gain_bits()) as u64;
        self.codes.len() as u64 * per_block
            + self.row_scales.len() as u64 * 64
            + self.centroids.len() as u64 * 64
            + self.tail.len() as u64 * 32
    }
}

fn put_u32(w: &mut impl Write, v: u32) -> Result<()> {
    w.write_all(&v.to_le_bytes())?;
    Ok(())
}
fn put_u64(w: &mut impl Write, v: u64) -> Result<()> {
    w.write_all(&v.to_le_bytes())?;
    Ok(())
}
fn get_u32(r: &mut impl Read, what: &'static str) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).map_err(|_| Error::Truncated { reading: what })?;
    Ok(u32::from_le_bytes(b))
}
fn get_u64(r: &mut impl Read, what: &'static str) -> Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).map_err(|_| Error::Truncated { reading: what })?;
    Ok(u64::from_le_bytes(b))
}

/// Cap on any allocation sized from a length field of the file. A length
/// field is a claim, not a fact: reserving gigabytes because a corrupted u64
/// says so would abort the process on OOM before the reads could fail.
/// Legitimate vectors larger than this grow as their elements arrive.
const PREALLOC_CAP: usize = 1 << 20;

/// Read exactly `n` bytes declared by a length field of the file.
///
/// The buffer grows only as bytes actually arrive, so a lying length on a
/// short stream returns [`Error::Truncated`] instead of aborting on an
/// allocation the file could never back.
fn get_bytes(r: &mut impl Read, n: u64, what: &'static str) -> Result<Vec<u8>> {
    let mut buf = Vec::with_capacity(n.min(PREALLOC_CAP as u64) as usize);
    let got = (&mut *r).take(n).read_to_end(&mut buf)?;
    if (got as u64) < n {
        return Err(Error::Truncated { reading: what });
    }
    Ok(buf)
}

/// The index width of a record, from its kind first and its `shell_cap`
/// second — the one place both readers and both writers get it from.
///
/// Ball: the cap sets the width, and a cap past the supported ball is an
/// `Err` here rather than an assert inside llvq-search's class enumeration.
/// Trio: the width is [`LABEL_BITS`] whatever the field says, and the field
/// must say [`TRIO_SHELL_CAP`] — a Trio record with any other value was not
/// written by this crate, and the only honest reading of it is a refusal.
fn index_width(kind: CodeKind, name: &str, shell_cap: u32) -> Result<u32> {
    match kind {
        CodeKind::Ball => {
            if shell_cap > llvq_search::classes::MAX_SHELL {
                return Err(Error::Inconsistent {
                    name: name.to_string(),
                    detail: format!(
                        "shell cap {shell_cap} exceeds the supported ball (m ≤ {})",
                        llvq_search::classes::MAX_SHELL
                    ),
                });
            }
            Ok(index_bits(shell_cap))
        }
        CodeKind::Trio => {
            if shell_cap != TRIO_SHELL_CAP {
                return Err(Error::Inconsistent {
                    name: name.to_string(),
                    detail: format!(
                        "shell cap {shell_cap} on a Trio record: the field is not read for \
                         Trio and is written as {TRIO_SHELL_CAP}"
                    ),
                });
            }
            Ok(LABEL_BITS)
        }
    }
}

/// The gain width of a record. A Trio word has exactly one gain bit, at bit
/// 47: two centroids, no more, no fewer — a wider gain field would push the
/// word past 48 bits and [`Codebook::decode`]'s `gain << 47` past the word.
fn gain_width(kind: CodeKind, name: &str, n_centroids: usize) -> Result<u32> {
    let gb = n_centroids.next_power_of_two().trailing_zeros();
    if kind == CodeKind::Trio && gb != 1 {
        return Err(Error::Inconsistent {
            name: name.to_string(),
            detail: format!(
                "{n_centroids} centroids on a Trio record: a Trio word carries one gain bit \
                 at bit {LABEL_BITS}, so a Trio matrix has two"
            ),
        });
    }
    Ok(gb)
}

/// Everything of a record before its code stream, in the order the file
/// stores it — shared by both writers so the two cannot drift apart
/// (`raw_passthrough_is_byte_identical` is what would notice).
#[allow(clippy::too_many_arguments)] // a record has many fields; that is the point
fn put_record_head(
    w: &mut impl Write,
    name: &str,
    d_out: usize,
    d_in: usize,
    shell_cap: u32,
    centroids: &[f64],
    row_scales: &[f64],
    rotation_seed: Option<u64>,
    tail: &[f64],
) -> Result<()> {
    let name = name.as_bytes();
    put_u32(w, name.len() as u32)?;
    w.write_all(name)?;
    put_u32(w, d_out as u32)?;
    put_u32(w, d_in as u32)?;
    put_u32(w, shell_cap)?;
    put_u32(w, centroids.len() as u32)?;
    put_u64(w, rotation_seed.unwrap_or(0))?;
    put_u32(w, rotation_seed.is_some() as u32)?;
    for c in centroids {
        put_u64(w, c.to_bits())?;
    }
    for s in row_scales {
        put_u64(w, s.to_bits())?;
    }
    for t in tail {
        put_u32(w, (*t as f32).to_bits())?;
    }
    Ok(())
}

/// Serialize one matrix through `encode`, returning the bits its payload
/// occupies. The Ball and Trio writers differ only in the map.
fn write_codes(
    w: &mut impl Write,
    kind: CodeKind,
    encode: &dyn Fn(&Point) -> Option<u64>,
    m: &QuantizedMatrix,
) -> Result<u64> {
    if m.codes.len() != m.d_out * m.nblocks() {
        return Err(Error::Inconsistent {
            name: m.name.clone(),
            detail: format!(
                "{} codes for {} blocks",
                m.codes.len(),
                m.d_out * m.nblocks()
            ),
        });
    }
    if m.row_scales.len() != m.d_out {
        return Err(Error::Inconsistent {
            name: m.name.clone(),
            detail: format!("{} row scales for {} rows", m.row_scales.len(), m.d_out),
        });
    }
    let ib = index_width(kind, &m.name, m.shell_cap)?;
    let gb = gain_width(kind, &m.name, m.centroids.len())?;

    put_record_head(
        w,
        &m.name,
        m.d_out,
        m.d_in,
        m.shell_cap,
        &m.centroids,
        &m.row_scales,
        m.rotation_seed,
        &m.tail,
    )?;

    // The disk word. `push(idx, ib); push(gain, gb)` MSB-first (`pack.rs`),
    // so a Trio block's 48 bits sit big-endian across six bytes, its `p` bit
    // last. The card's decoders (`llvq_f1rank*.cuh`) read a little-endian
    // 48-bit word, bit 0 = `p` in the lowest byte: that reordering is F1d's
    // transcoder (`trio48`, roadmap §2.2 quater step 6), not this crate's,
    // and nothing here produces a device stream for Trio
    // (`runtime::require_ball`).
    let mut bw = BitWriter::with_capacity(m.codes.len() as u64 * (ib + gb) as u64);
    for c in &m.codes {
        let idx = encode(&c.point).ok_or_else(|| Error::PointOutsideCodebook {
            name: m.name.clone(),
        })?;
        if idx >= (1u64 << ib) {
            return Err(Error::IndexTooWide {
                name: m.name.clone(),
                index: idx,
                bits: ib,
            });
        }
        bw.push(idx, ib);
        bw.push(c.gain as u64, gb);
    }
    let bytes = bw.finish();
    put_u64(w, bytes.len() as u64)?;
    w.write_all(&bytes)?;
    Ok(m.bits())
}

/// Serialize one Ball matrix, returning the bits its payload occupies.
///
/// The `Indexer` is shared across calls — building it enumerates 383 classes
/// and has no business happening per matrix. [`write_matrix_with`] is the
/// same entry for either kind.
pub fn write_matrix(w: &mut impl Write, ix: &Indexer, m: &QuantizedMatrix) -> Result<u64> {
    write_codes(w, CodeKind::Ball, &|p| ix.encode(p), m)
}

/// Serialize one matrix through the file's [`Codebook`]: Ball → the v1
/// index, Trio → the Trio word ([`Error::PointOutsideCodebook`] for a point
/// the map refuses, which for Trio means a point off its label set).
pub fn write_matrix_with(w: &mut impl Write, cb: &Codebook, m: &QuantizedMatrix) -> Result<u64> {
    write_codes(w, cb.kind(), &|p| cb.encode(p), m)
}

/// One matrix with its codes left as raw `(index, gain)` pairs — what the
/// stream actually stores, before any lattice decoding.
///
/// This is the entry point for anything that walks a real artifact at scale:
/// the load-time transcoder and the format-accounting benches classify blocks
/// straight from the index (a class is a fixed multiset of |values|, so most
/// per-block facts need no decode at all). [`read_matrix`] is this plus a
/// decode of every index. Which map the indices belong to is the file's
/// business, not the record's: a `RawMatrix` of a Trio file holds Trio
/// words, and only [`read_matrix_raw_for_kind`]'s caller knows it.
pub struct RawMatrix {
    pub name: String,
    pub d_out: usize,
    pub d_in: usize,
    /// Row-major `d_out × (d_in / 24)`, one index per block, undecoded and
    /// **unvalidated** — decoding is where an out-of-range index surfaces.
    pub indices: Vec<u64>,
    /// Gain level rank per block, same order.
    pub gains: Vec<u32>,
    pub row_scales: Vec<f64>,
    pub centroids: Vec<f64>,
    pub rotation_seed: Option<u64>,
    pub shell_cap: u32,
    pub tail: Vec<f64>,
}

/// Serialize one matrix from its undecoded `(index, gain)` pairs.
///
/// This is [`write_matrix`] minus the lattice: a matrix read with
/// [`read_matrix_raw`] and written back through here must produce the same
/// bytes, which is what lets a tool rewrite a sealed file's raw-tensor section
/// without paying (or trusting) a decode/re-encode of 150 M blocks. The
/// byte-identity is pinned by `raw_passthrough_is_byte_identical`, at v4 and
/// at v5. The widths come from `shell_cap` as for a Ball record;
/// [`write_matrix_raw_for_kind`] is the same write with the Trio invariants
/// checked first.
pub fn write_matrix_raw(w: &mut impl Write, m: &RawMatrix) -> Result<u64> {
    write_matrix_raw_for_kind(w, CodeKind::Ball, m)
}

/// [`write_matrix_raw`] under the file's kind: a Trio record must carry
/// [`TRIO_SHELL_CAP`] and one gain bit, or it is refused before a byte is
/// written — a passthrough is not a licence to copy a record the file's own
/// reader would refuse.
pub fn write_matrix_raw_for_kind(w: &mut impl Write, kind: CodeKind, m: &RawMatrix) -> Result<u64> {
    let nblocks = m.d_in / DIM;
    if m.indices.len() != m.d_out * nblocks || m.gains.len() != m.indices.len() {
        return Err(Error::Inconsistent {
            name: m.name.clone(),
            detail: format!(
                "{} indices / {} gains for {} blocks",
                m.indices.len(),
                m.gains.len(),
                m.d_out * nblocks
            ),
        });
    }
    if m.row_scales.len() != m.d_out {
        return Err(Error::Inconsistent {
            name: m.name.clone(),
            detail: format!("{} row scales for {} rows", m.row_scales.len(), m.d_out),
        });
    }
    let ib = index_width(kind, &m.name, m.shell_cap)?;
    let gb = gain_width(kind, &m.name, m.centroids.len())?;

    put_record_head(
        w,
        &m.name,
        m.d_out,
        m.d_in,
        m.shell_cap,
        &m.centroids,
        &m.row_scales,
        m.rotation_seed,
        &m.tail,
    )?;

    let mut bw = BitWriter::with_capacity(m.indices.len() as u64 * (ib + gb) as u64);
    for (&idx, &gain) in m.indices.iter().zip(&m.gains) {
        if idx >= (1u64 << ib) {
            return Err(Error::IndexTooWide {
                name: m.name.clone(),
                index: idx,
                bits: ib,
            });
        }
        bw.push(idx, ib);
        bw.push(gain as u64, gb);
    }
    let bytes = bw.finish();
    put_u64(w, bytes.len() as u64)?;
    w.write_all(&bytes)?;
    Ok(m.indices.len() as u64 * (ib + gb) as u64
        + m.row_scales.len() as u64 * 64
        + m.centroids.len() as u64 * 64
        + m.tail.len() as u64 * 32)
}

/// Read one Ball matrix without decoding its indices — every file before
/// v5, and every v5 file whose header says `Ball`.
///
/// Kept at this signature on purpose: some twenty callers across the tree
/// walk archives through it, and every one of them reads a Ball file. A
/// caller that has opened a v5 header goes through
/// [`read_matrix_raw_for_kind`] with the header's kind instead.
pub fn read_matrix_raw(r: &mut impl Read) -> Result<RawMatrix> {
    read_matrix_raw_for_kind(r, CodeKind::Ball)
}

/// Read one matrix without decoding its indices, under the file's kind.
///
/// The kind decides the index width before the record's own `shell_cap` is
/// looked at ([`index_width`]): for Trio the width is [`LABEL_BITS`] and the
/// field is only checked to be [`TRIO_SHELL_CAP`]. A reader that took the
/// width from the field would read a Trio record whose field had been
/// corrupted to 13 as 48-bit words, in step with nothing.
pub fn read_matrix_raw_for_kind(r: &mut impl Read, kind: CodeKind) -> Result<RawMatrix> {
    let n = get_u32(r, "name length")? as usize;
    let name = get_bytes(r, n as u64, "name")?;
    let name = String::from_utf8(name).map_err(|_| Error::BadName)?;
    let d_out = get_u32(r, "d_out")? as usize;
    let d_in = get_u32(r, "d_in")? as usize;
    let shell_cap = get_u32(r, "shell cap")?;
    // Validated before any width is derived: `index_bits` asserts on the
    // supported ball inside llvq-search, and a wild cap from a corrupted file
    // must be an `Err`, not a panic — for either kind.
    let ib = index_width(kind, &name, shell_cap)?;
    let n_cent = get_u32(r, "centroid count")? as usize;
    let gb = gain_width(kind, &name, n_cent)?;
    let seed = get_u64(r, "rotation seed")?;
    let has_rot = get_u32(r, "rotation flag")? != 0;

    let mut centroids = Vec::with_capacity(n_cent.min(PREALLOC_CAP));
    for _ in 0..n_cent {
        centroids.push(f64::from_bits(get_u64(r, "centroids")?));
    }
    let mut row_scales = Vec::with_capacity(d_out.min(PREALLOC_CAP));
    for _ in 0..d_out {
        row_scales.push(f64::from_bits(get_u64(r, "row scales")?));
    }
    let tail_w = d_out * (d_in % DIM);
    let mut tail = Vec::with_capacity(tail_w.min(PREALLOC_CAP));
    for _ in 0..tail_w {
        tail.push(f32::from_bits(get_u32(r, "tail")?) as f64);
    }

    let nbytes = get_u64(r, "code length")?;
    let bytes = get_bytes(r, nbytes, "code stream")?;

    let nblocks = d_out * (d_in / DIM);
    // The stream must hold every block the dimensions promise. Checked here
    // because `BitReader::read` treats an over-read as a caller bug and
    // panics — its assert is an internal invariant, and the boundary where a
    // hostile file is still an `Err` is this one. The multiplication is
    // checked for the same reason as `get_u16s`'s: hostile u32 dimensions can
    // push it past u64 (a debug panic, a wrap in release that would let the
    // guard pass), and a bit count that overflows u64 names data no file
    // could hold.
    let need_bits = (nblocks as u64)
        .checked_mul((ib + gb) as u64)
        .ok_or(Error::Truncated {
            reading: "code stream",
        })?;
    if need_bits > bytes.len() as u64 * 8 {
        return Err(Error::Truncated {
            reading: "code stream",
        });
    }
    let mut br = BitReader::new(&bytes);
    let mut indices = Vec::with_capacity(nblocks.min(PREALLOC_CAP));
    let mut gains = Vec::with_capacity(nblocks.min(PREALLOC_CAP));
    for _ in 0..nblocks {
        indices.push(br.read(ib));
        gains.push(br.read(gb) as u32);
    }

    Ok(RawMatrix {
        name,
        d_out,
        d_in,
        indices,
        gains,
        row_scales,
        centroids,
        rotation_seed: has_rot.then_some(seed),
        shell_cap,
        tail,
    })
}

/// Decode every `(index, gain)` of a raw record through `decode`.
fn decode_codes(
    raw: RawMatrix,
    decode: &dyn Fn(u64, u32) -> Option<Point>,
) -> Result<QuantizedMatrix> {
    let mut codes = Vec::with_capacity(raw.indices.len());
    for (&idx, &gain) in raw.indices.iter().zip(&raw.gains) {
        let point = decode(idx, gain).ok_or(Error::IndexOutOfRange {
            name: raw.name.clone(),
            index: idx,
        })?;
        codes.push(BlockCode { point, gain });
    }

    Ok(QuantizedMatrix {
        name: raw.name,
        d_out: raw.d_out,
        d_in: raw.d_in,
        codes,
        row_scales: raw.row_scales,
        centroids: raw.centroids,
        rotation_seed: raw.rotation_seed,
        shell_cap: raw.shell_cap,
        tail: raw.tail,
    })
}

/// Read back what [`write_matrix`] wrote — a Ball record.
pub fn read_matrix(r: &mut impl Read, ix: &Indexer) -> Result<QuantizedMatrix> {
    decode_codes(read_matrix_raw_for_kind(r, CodeKind::Ball)?, &|idx, _| ix.decode(idx))
}

/// Read back what [`write_matrix_with`] wrote, through the file's
/// [`Codebook`]: Ball → `Indexer::decode(idx)`, Trio →
/// `Trio::decode(idx | gain << 47)`.
pub fn read_matrix_with(r: &mut impl Read, cb: &Codebook) -> Result<QuantizedMatrix> {
    decode_codes(read_matrix_raw_for_kind(r, cb.kind())?, &|idx, gain| cb.decode(idx, gain))
}

/// Rebuild the `d_out × d_in` weight matrix, in the **natural** basis, exactly
/// as the evaluated model holds it.
///
/// The order of operations mirrors the quantization loop: decode in the
/// rotated basis, restore the tail, un-rotate, and only then narrow to f32.
/// Doing the narrowing earlier, or un-rotating in f32, changes the last bits —
/// and the whole claim of this format is that it does not. The per-block
/// reconstruction is [`reconstruct_shape_gain`], the same function the
/// quantizer's own `reconstruct` calls: no direction code is consulted, so a
/// Ball and a Trio matrix rebuild through the same lines.
pub fn decode_matrix(m: &QuantizedMatrix) -> Vec<f32> {
    let nblocks = m.nblocks();
    let tail_w = m.d_in % DIM;

    let mut w = vec![0.0f64; m.d_out * m.d_in];
    let mut block = [0.0f64; DIM];
    for i in 0..m.d_out {
        for p in 0..nblocks {
            reconstruct_shape_gain(&m.codes[i * nblocks + p], &m.centroids, m.row_scales[i], &mut block);
            let at = i * m.d_in + p * DIM;
            w[at..at + DIM].copy_from_slice(&block);
        }
        if tail_w > 0 {
            let at = i * m.d_in + nblocks * DIM;
            w[at..at + tail_w].copy_from_slice(&m.tail[i * tail_w..(i + 1) * tail_w]);
        }
    }
    if let Some(seed) = m.rotation_seed {
        llvq_quant::rotation::Rotation::new(m.d_in, seed).unrotate_weight_rows(&mut w, m.d_out);
    }
    w.into_iter().map(|v| v as f32).collect()
}

/// `model.layers.{b}.{proj}.weight` → `(b, proj)`.
pub fn split_name(name: &str) -> Result<(usize, String)> {
    let parts: Vec<&str> = name.split('.').collect();
    if parts.len() < 5 {
        return Err(Error::UnexpectedName {
            name: name.to_string(),
        });
    }
    let block = parts[2].parse().map_err(|_| Error::UnexpectedName {
        name: name.to_string(),
    })?;
    Ok((block, parts[3..parts.len() - 1].join(".")))
}

/// Streaming writer for a whole model.
pub struct ArtifactWriter<W: Write> {
    out: W,
    codebook: Codebook,
    pub matrices: u32,
    pub payload_bits: u64,
}

impl<W: Write> ArtifactWriter<W> {
    /// `n_matrices` is written up front so the reader can size itself. A
    /// Ball file at [`DEFAULT_VERSION`], byte for byte what it always was.
    pub fn new(out: W, n_matrices: u32) -> Result<Self> {
        Self::with_version_kind(out, DEFAULT_VERSION, n_matrices, CodeKind::Ball)
    }

    /// Same, at a chosen format version — for a tool that rewrites an existing
    /// Ball file and must not silently upgrade it.
    pub fn with_version(out: W, version: u32, n_matrices: u32) -> Result<Self> {
        Self::with_version_kind(out, version, n_matrices, CodeKind::Ball)
    }

    /// A writer for `kind`, at the version the kind calls for
    /// ([`CodeKind::default_version`]): v4 for Ball, v5 for Trio.
    pub fn with_kind(out: W, kind: CodeKind, n_matrices: u32) -> Result<Self> {
        Self::with_version_kind(out, kind.default_version(), n_matrices, kind)
    }

    /// The general form. A Trio writer below [`FIRST_KINDED_VERSION`] is
    /// refused: the header would have nowhere to say what its records are.
    pub fn with_version_kind(
        mut out: W,
        version: u32,
        n_matrices: u32,
        kind: CodeKind,
    ) -> Result<Self> {
        write_header_kind(&mut out, version, n_matrices, kind)?;
        Ok(Self {
            out,
            codebook: Codebook::new(kind),
            matrices: 0,
            payload_bits: 0,
        })
    }

    pub fn kind(&self) -> CodeKind {
        self.codebook.kind()
    }

    pub fn push(&mut self, m: &QuantizedMatrix) -> Result<()> {
        self.payload_bits += write_matrix_with(&mut self.out, &self.codebook, m)?;
        self.matrices += 1;
        Ok(())
    }

    /// Push a matrix straight from its undecoded codes — see
    /// [`write_matrix_raw_for_kind`]: under a Trio writer the record's Trio
    /// invariants are checked, and nothing else about it is looked at.
    pub fn push_raw(&mut self, m: &RawMatrix) -> Result<()> {
        self.payload_bits += write_matrix_raw_for_kind(&mut self.out, self.codebook.kind(), m)?;
        self.matrices += 1;
        Ok(())
    }

    /// Close the file with the sections that make it self-contained.
    ///
    /// Taking them here rather than streaming them is deliberate: the counts
    /// have to precede the data, and a writer that cannot seek must therefore
    /// know them before writing. They are also small next to the codes —
    /// except the embedding, which is one tensor.
    pub fn seal(
        mut self,
        raws: &[crate::RawTensor],
        blobs: &[crate::Blob],
    ) -> Result<(u64, u64)> {
        put_u32(&mut self.out, raws.len() as u32)?;
        let mut extra = 0u64;
        for t in raws {
            extra += crate::write_raw(&mut self.out, t)?;
        }
        put_u32(&mut self.out, blobs.len() as u32)?;
        for b in blobs {
            extra += crate::write_blob(&mut self.out, b)?;
        }
        self.out.flush()?;
        Ok((self.payload_bits, extra * 8))
    }

    /// Close without the extra sections, producing a projections-only file.
    ///
    /// Kept for the quantization run, which streams matrices as it goes and
    /// has no checkpoint tensors to hand. `bin/seal` completes such a file.
    pub fn finish(mut self) -> Result<u64> {
        put_u32(&mut self.out, 0)?;
        put_u32(&mut self.out, 0)?;
        self.out.flush()?;
        Ok(self.payload_bits)
    }
}

/// What a file's header says: which version, how many matrices follow, from
/// [`FIRST_FINGERPRINTED_VERSION`] on which codebook wrote it, and from
/// [`FIRST_KINDED_VERSION`] on which map its indices belong to.
pub struct Header {
    /// 1 for projections only; 2+ for a self-contained file. 3 adds tagged
    /// raw-tensor encodings; 4 adds the codebook fingerprint; 5 adds the Trio
    /// fingerprint and the code kind.
    pub version: u32,
    pub matrices: u32,
    /// The writer's v1 codebook fingerprint, or `None` for a legacy file that
    /// predates the field.
    ///
    /// A `Some` here has already been checked against
    /// [`crate::codebook_fingerprint`]: [`read_header`] refuses a file whose
    /// codebook is not this build's, so reaching a `Header` at all means the
    /// indices about to be read mean what the writer meant.
    pub codebook: Option<u64>,
    /// The writer's Trio fingerprint, checked the same way against
    /// [`crate::codebook::trio_fingerprint`]; `None` below v5. Carried and
    /// checked by a v5 Ball file too — the header describes the build, not
    /// only the map in use.
    pub trio: Option<u64>,
    /// The map the records are read through; `Ball` for every version below
    /// v5, which had nothing else to be.
    pub kind: CodeKind,
}

impl Header {
    /// Whether the file carries embeddings, norms and tokenizer — i.e. whether
    /// it can run without the original checkpoint.
    pub fn is_self_contained(&self) -> bool {
        self.version >= 2
    }

    /// The map the records are read through — what every reader of a record
    /// and every runtime transcoder must consult first.
    pub fn kind(&self) -> CodeKind {
        self.kind
    }
}

/// Write a file header for `version`, kind `Ball`.
///
/// Exposed because the header is no longer `magic + count`: from
/// [`FIRST_FINGERPRINTED_VERSION`] on it also carries the fingerprint, from
/// [`FIRST_KINDED_VERSION`] on a second one and the kind, and a tool that
/// rewrites a file's matrix section byte for byte has to reproduce the whole
/// of it. Hand-rolling the writes is exactly how a `LVQ4` magic ends up over
/// a `LVQ3` body.
pub fn write_header(w: &mut impl Write, version: u32, matrices: u32) -> Result<()> {
    write_header_kind(w, version, matrices, CodeKind::Ball)
}

/// [`write_header`] for either kind. A kind other than `Ball` needs
/// [`FIRST_KINDED_VERSION`] or later, and is refused below it rather than
/// written as a header whose records would read as Ball.
pub fn write_header_kind(
    w: &mut impl Write,
    version: u32,
    matrices: u32,
    kind: CodeKind,
) -> Result<()> {
    let magic = match version {
        1 => MAGIC_V1,
        2 => MAGIC_V2,
        3 => MAGIC_V3,
        4 => MAGIC_V4,
        5 => MAGIC_V5,
        v => return Err(Error::UnknownVersion { version: v }),
    };
    if kind != CodeKind::Ball && version < FIRST_KINDED_VERSION {
        return Err(Error::Inconsistent {
            name: "header".into(),
            detail: format!(
                "a {kind} file needs format v{FIRST_KINDED_VERSION}: v{version} carries no \
                 code kind, and its records would read as Ball"
            ),
        });
    }
    w.write_all(magic)?;
    put_u32(w, matrices)?;
    if version >= FIRST_FINGERPRINTED_VERSION {
        put_u64(w, crate::codebook_fingerprint())?;
    }
    if version >= FIRST_KINDED_VERSION {
        put_u64(w, crate::codebook::trio_fingerprint())?;
        put_u32(w, kind.tag())?;
    }
    Ok(())
}

/// Read the file header, refusing a file this build cannot interpret.
///
/// Separate from [`read_all`] because a 4B model's codes are 14 GB of lattice
/// points: anything that walks a real artifact has to do it one matrix at a
/// time, and holding them all was never an option.
///
/// The fingerprints are checked here rather than at first decode so that a
/// codebook disagreement costs a few header bytes of I/O instead of a
/// gigabyte, and so that every caller gets the check without asking for it.
/// The kind is read last and refused if unknown: past an unknown kind no
/// record has a width.
pub fn read_header(r: &mut impl Read) -> Result<Header> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)
        .map_err(|_| Error::Truncated { reading: "magic" })?;
    let version = if &magic == MAGIC_V5 {
        5
    } else if &magic == MAGIC_V4 {
        4
    } else if &magic == MAGIC_V3 {
        3
    } else if &magic == MAGIC_V2 {
        2
    } else if &magic == MAGIC_V1 {
        1
    } else {
        return Err(Error::NotAnArtifact { got: magic });
    };
    let matrices = get_u32(r, "matrix count")?;
    let codebook = if version >= FIRST_FINGERPRINTED_VERSION {
        Some(check_fingerprint(
            get_u64(r, "codebook fingerprint")?,
            crate::codebook_fingerprint(),
            "v1 ball",
        )?)
    } else {
        None
    };
    let (trio, kind) = if version >= FIRST_KINDED_VERSION {
        let trio = check_fingerprint(
            get_u64(r, "trio fingerprint")?,
            crate::codebook::trio_fingerprint(),
            "Trio",
        )?;
        (Some(trio), CodeKind::from_tag(get_u32(r, "code kind")?)?)
    } else {
        (None, CodeKind::Ball)
    };
    Ok(Header {
        version,
        matrices,
        codebook,
        trio,
        kind,
    })
}

fn check_fingerprint(stored: u64, computed: u64, which: &'static str) -> Result<u64> {
    if stored != computed {
        return Err(Error::CodebookMismatch {
            which,
            stored,
            computed,
        });
    }
    Ok(stored)
}

/// Read every matrix back through the header's kind. Only safe for small
/// models — see [`read_header`].
pub fn read_all(r: &mut impl Read) -> Result<Vec<QuantizedMatrix>> {
    let h = read_header(r)?;
    let cb = Codebook::new(h.kind());
    (0..h.matrices).map(|_| read_matrix_with(r, &cb)).collect()
}
