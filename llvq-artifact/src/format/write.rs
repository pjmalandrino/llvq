//! The lattice matrix types and their writers: the record head both lattice
//! writers and the int4 writer share, the Ball and Tetra code writers, and
//! the raw passthrough with the code stream it writes.

use super::checks::{
    check_rotation, check_tail, gain_out_of_range, gain_width, index_width, max_index, put_u32,
    put_u64, u32_field,
};
use super::kinds::{CodeKind, Codebook, FIRST_KINDED_VERSION};
use crate::{Error, Result};
use llvq_core::{Point, DIM};
use llvq_search::index::Indexer;
use llvq_search::pack::BitWriter;
use llvq_search::shapegain::{index_bits, BlockCode};
use std::io::Write;
// Named only by the documentation's links, which resolved in the one file.
#[cfg(doc)]
use super::{read_matrix, read_matrix_raw, Int4Matrix, TETRA_SHELL_CAP};

/// One quantized matrix, everything a decoder needs.
///
/// Lattice only: every field below is a fact about `(index, gain)` pairs on
/// Λ₂₄. An [`CodeKind::Int4G128`] record has no shape in this type — no
/// centroids, no row scales, no tail, no shell cap — and is [`Int4Matrix`]
/// instead. Forcing it in here would give it four fields with nothing true to
/// put in them, which is how a sentinel becomes a value somebody reads.
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
    /// Ball matrix; [`TETRA_SHELL_CAP`] on a Tetra matrix, where it is a
    /// sentinel and not a cap.
    pub shell_cap: u32,
    /// Trailing columns kept at full precision, `d_out × (d_in % 24)`
    /// row-major, in the rotated basis.
    pub tail: Vec<f64>,
}

impl QuantizedMatrix {
    pub(super) fn nblocks(&self) -> usize {
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

/// Everything of a record before its code stream, in the order the file
/// stores it — shared by both writers so the two cannot drift apart
/// (`raw_passthrough_is_byte_identical` is what would notice).
///
/// The kind tag goes between the shell cap and the centroid count, and only
/// from [`FIRST_KINDED_VERSION`]: a v4 record has nowhere to put it, so a
/// non-Ball record at that version is refused here rather than written as a
/// record that would read back as Ball.
#[allow(clippy::too_many_arguments)] // a record has many fields; that is the point
pub(super) fn put_record_head(
    w: &mut impl Write,
    version: u32,
    name: &str,
    d_out: usize,
    d_in: usize,
    shell_cap: u32,
    kind: CodeKind,
    centroids: &[f64],
    row_scales: &[f64],
    rotation_seed: Option<u64>,
    tail: &[f64],
) -> Result<()> {
    if kind == CodeKind::Int4G128
        && (!centroids.is_empty()
            || !row_scales.is_empty()
            || !tail.is_empty()
            || rotation_seed.is_some())
    {
        return Err(Error::Inconsistent {
            name: name.to_string(),
            detail: format!(
                "an Int4G128 head with {} centroids, {} row scales, {} tail values and \
                 rotation {}: none of the four has a reader, and a head that carried them \
                 would put the payload length where nothing looks for it",
                centroids.len(),
                row_scales.len(),
                tail.len(),
                rotation_seed.is_some()
            ),
        });
    }
    if version < FIRST_KINDED_VERSION && kind != CodeKind::Ball {
        return Err(Error::Inconsistent {
            name: name.to_string(),
            detail: format!(
                "a {kind} record needs format v{FIRST_KINDED_VERSION}: a v{version} record \
                 carries no code kind, and this one would read back as Ball"
            ),
        });
    }
    check_rotation(name, d_in, rotation_seed.is_some())?;
    // Every u32 field is converted before the first byte goes out.
    let name_len = u32_field(name, "name length", name.len())?;
    let d_out32 = u32_field(name, "d_out", d_out)?;
    let d_in32 = u32_field(name, "d_in", d_in)?;
    let n_cent32 = u32_field(name, "centroid count", centroids.len())?;
    let name = name.as_bytes();
    put_u32(w, name_len)?;
    w.write_all(name)?;
    put_u32(w, d_out32)?;
    put_u32(w, d_in32)?;
    put_u32(w, shell_cap)?;
    if version >= FIRST_KINDED_VERSION {
        put_u32(w, kind.tag())?;
    }
    put_u32(w, n_cent32)?;
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
/// occupies. The Ball and Tetra writers differ only in the map.
fn write_codes(
    w: &mut impl Write,
    version: u32,
    kind: CodeKind,
    encode: &dyn Fn(&Point) -> Option<u64>,
    m: &QuantizedMatrix,
) -> Result<u64> {
    // Before any width is derived: `index_width` would answer for a kind that
    // has no codes at all.
    if kind == CodeKind::Int4G128 {
        return Err(Error::Inconsistent {
            name: m.name.clone(),
            detail: "an Int4G128 matrix has no lattice codes".to_string(),
        });
    }
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
    check_tail(&m.name, m.d_out, m.d_in, m.tail.len())?;
    let ib = index_width(kind, &m.name, m.shell_cap)?;
    let gb = gain_width(kind, &m.name, m.centroids.len())?;
    // Before the head goes out, like the checks above.
    if let Some(c) = m.codes.iter().find(|c| c.gain as usize >= m.centroids.len()) {
        return Err(gain_out_of_range(&m.name, c.gain, m.centroids.len()));
    }

    put_record_head(
        w,
        version,
        &m.name,
        m.d_out,
        m.d_in,
        m.shell_cap,
        kind,
        &m.centroids,
        &m.row_scales,
        m.rotation_seed,
        &m.tail,
    )?;

    // The disk word. `push(idx, ib); push(gain, gb)` MSB-first (`pack.rs`),
    // so a Tetra block's 48 bits sit big-endian across six bytes, its `p` bit
    // last. The card's decoders (`llvq_f1rank*.cuh`) read a little-endian
    // 48-bit word, bit 0 = `p` in the lowest byte: that reordering is F1d's
    // transcoder (`tetra48`, roadmap §2.2 quater step 6), not this crate's,
    // and nothing here produces a device stream for Tetra
    // (`runtime::require_ball`).
    let max_idx = max_index(kind, m.shell_cap);
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
        // Inside the field and still past the cap: the reader refuses it.
        if idx > max_idx {
            return Err(Error::IndexOutOfRange {
                name: m.name.clone(),
                index: idx,
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

/// Serialize one Ball matrix into a file of version `version`, returning the
/// bits its payload occupies.
///
/// The `Indexer` is shared across calls — building it enumerates 383 classes
/// and has no business happening per matrix. [`write_matrix_with`] is the
/// same entry for either kind.
pub fn write_matrix(
    w: &mut impl Write,
    version: u32,
    ix: &Indexer,
    m: &QuantizedMatrix,
) -> Result<u64> {
    write_codes(w, version, CodeKind::Ball, &|p| ix.encode(p), m)
}

/// Serialize one matrix through a [`Codebook`], which is what fixes the
/// record's kind: Ball → the v1 index, Tetra → the Tetra word
/// ([`Error::PointOutsideCodebook`] for a point the map refuses, which for
/// Tetra means a point off its label set).
pub fn write_matrix_with(
    w: &mut impl Write,
    version: u32,
    cb: &Codebook,
    m: &QuantizedMatrix,
) -> Result<u64> {
    write_codes(w, version, cb.kind(), &|p| cb.encode(p), m)
}

/// One matrix with its codes left as raw `(index, gain)` pairs — what the
/// stream actually stores, before any lattice decoding.
///
/// This is the entry point for anything that walks a real artifact at scale:
/// the load-time transcoder and the format-accounting benches classify blocks
/// straight from the index (a class is a fixed multiset of |values|, so most
/// per-block facts need no decode at all). [`read_matrix`] is this plus a
/// decode of every index.
pub struct RawMatrix {
    pub name: String,
    pub d_out: usize,
    pub d_in: usize,
    /// The map these indices belong to — the record's own field from v5 on,
    /// `Ball` below it. It is what a passthrough carries across
    /// ([`write_matrix_raw`]) and what a Ball-only tool checks before it
    /// files an index under one of the 383 classes.
    pub kind: CodeKind,
    /// Row-major `d_out × (d_in / 24)`, one index per block, undecoded. The
    /// reader checks a Ball index against its cap, at most `N(shell_cap)`;
    /// nothing else about an index is checked before decoding.
    pub indices: Vec<u64>,
    /// Gain level rank per block, same order.
    pub gains: Vec<u32>,
    pub row_scales: Vec<f64>,
    pub centroids: Vec<f64>,
    pub rotation_seed: Option<u64>,
    pub shell_cap: u32,
    pub tail: Vec<f64>,
}

/// The index and gain widths of a lattice record, in bits.
///
/// Public because the widths are a format fact and a reader outside this crate
/// has no other honest source for them. `llvq_llm::hfpack`, which rewrites a
/// sealed file into safetensors, asks here rather than recomputing
/// `LABEL_BITS` and `log2(centroids)` on its own: a packer that disagreed with
/// the format about a width would produce a file that loads and decodes
/// plausible, wrong weights.
///
/// Refuses an [`CodeKind::Int4G128`] record, which has neither field.
pub fn code_widths(
    kind: CodeKind,
    name: &str,
    shell_cap: u32,
    n_centroids: usize,
) -> Result<(u32, u32)> {
    Ok((
        index_width(kind, name, shell_cap)?,
        gain_width(kind, name, n_centroids)?,
    ))
}

/// The code stream of a lattice record: `(index, gain)` pairs packed
/// [MSB-first](llvq_search::pack), the exact bytes the record stores.
///
/// [`write_matrix_raw`] calls this and writes what it returns, so the two
/// cannot drift: the bytes a packer gets here are the bytes the file holds,
/// by construction rather than by a comment. `code_stream_is_the_record_bytes`
/// pins it anyway, because the two could be pulled apart by an edit that looks
/// local.
///
/// The record's invariants are checked here, before a bit is written: an
/// [`CodeKind::Int4G128`] kind on a lattice type, a block count the dimensions
/// do not support, a row-scale count that is not `d_out`, a tail count that is
/// not `d_out · (d_in mod 24)`, an index wider than its field, a Ball index
/// past its cap, a gain rank with no centroid.
pub fn code_stream(m: &RawMatrix) -> Result<Vec<u8>> {
    // A `RawMatrix` is a lattice type; one carrying the int4 kind was built by
    // hand. Writing it would put a lattice head over int4 bytes, which is the
    // one shape no reader of this crate can tell from a torn file.
    if m.kind == CodeKind::Int4G128 {
        return Err(Error::NotALatticeRecord {
            name: m.name.clone(),
            kind: m.kind,
        });
    }
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
    check_tail(&m.name, m.d_out, m.d_in, m.tail.len())?;
    let (ib, gb) = code_widths(m.kind, &m.name, m.shell_cap, m.centroids.len())?;
    let max_idx = max_index(m.kind, m.shell_cap);
    let mut bw = BitWriter::with_capacity(m.indices.len() as u64 * (ib + gb) as u64);
    for (&idx, &gain) in m.indices.iter().zip(&m.gains) {
        if idx >= (1u64 << ib) {
            return Err(Error::IndexTooWide {
                name: m.name.clone(),
                index: idx,
                bits: ib,
            });
        }
        // Inside the field and still past the cap: the reader refuses it.
        if idx > max_idx {
            return Err(Error::IndexOutOfRange {
                name: m.name.clone(),
                index: idx,
            });
        }
        if gain as usize >= m.centroids.len() {
            return Err(gain_out_of_range(&m.name, gain, m.centroids.len()));
        }
        bw.push(idx, ib);
        bw.push(gain as u64, gb);
    }
    Ok(bw.finish())
}

/// Serialize one matrix from its undecoded `(index, gain)` pairs, into a file
/// of version `version`.
///
/// This is [`write_matrix`] minus the lattice: a matrix read with
/// [`read_matrix_raw`] at some version and written back through here at the
/// same version must produce the same bytes, which is what lets a tool
/// rewrite a sealed file's raw-tensor section without paying (or trusting) a
/// decode/re-encode of 150 M blocks. The byte-identity is pinned by
/// `raw_passthrough_is_byte_identical`, at v4 and at v5.
///
/// The record's kind is `m.kind` — whatever the record was read as, not what
/// the file's default is: that is the whole of what makes a mixed file
/// copyable. Its invariants are checked first, so a passthrough is not a
/// licence to write a record the file's own reader would refuse.
pub fn write_matrix_raw(w: &mut impl Write, version: u32, m: &RawMatrix) -> Result<u64> {
    let kind = m.kind;
    // `code_stream` first, because it is the one that refuses an int4 kind by
    // name. `code_widths` would refuse it too, as a record with no index width,
    // and `an_int4_record_is_refused_by_every_lattice_entry` demands the name.
    let bytes = code_stream(m)?;
    let (ib, gb) = code_widths(kind, &m.name, m.shell_cap, m.centroids.len())?;

    put_record_head(
        w,
        version,
        &m.name,
        m.d_out,
        m.d_in,
        m.shell_cap,
        kind,
        &m.centroids,
        &m.row_scales,
        m.rotation_seed,
        &m.tail,
    )?;

    put_u64(w, bytes.len() as u64)?;
    w.write_all(&bytes)?;
    Ok(m.indices.len() as u64 * (ib + gb) as u64
        + m.row_scales.len() as u64 * 64
        + m.centroids.len() as u64 * 64
        + m.tail.len() as u64 * 32)
}
