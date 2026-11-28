//! The lattice reader: the record head every kind shares, and the lattice
//! body after it, where a hostile file must still come out as an `Err`.

use super::checks::{
    check_rotation, gain_out_of_range, gain_width, get_bytes, get_u32, get_u64, index_width,
    max_index, PREALLOC_CAP,
};
use super::kinds::{CodeKind, FIRST_KINDED_VERSION};
use super::write::RawMatrix;
use crate::{Error, Result};
use llvq_core::DIM;
use llvq_search::pack::BitReader;
use std::io::Read;
// Named only by the documentation's links, which resolved in the one file.
#[cfg(doc)]
use super::{KindSet, TETRA_SHELL_CAP};
#[cfg(doc)]
use llvq_search::tetra::LABEL_BITS;

/// Read one matrix without decoding its indices, from a file of version
/// `version`.
///
/// The version is what says whether the record carries a kind tag at all: it
/// does from [`FIRST_KINDED_VERSION`], and below that it is a Ball record
/// because nothing else existed. Passing the wrong version does not read the
/// wrong kind quietly — it reads the kind tag as a centroid count, or the
/// count as a kind — which is why the argument is here rather than defaulted
/// (`a_v5_record_read_as_v4_is_not_the_same_record`).
///
/// The kind then decides the index width before the record's own `shell_cap`
/// is looked at ([`index_width`]): for Tetra the width is [`LABEL_BITS`] and
/// the field is only checked to be [`TETRA_SHELL_CAP`]. A reader that took the
/// width from the field would read a Tetra record whose field had been
/// corrupted to 13 as 48-bit words, in step with nothing.
pub fn read_matrix_raw(r: &mut impl Read, version: u32) -> Result<RawMatrix> {
    let head = get_record_head(r, version)?;
    read_lattice_body(r, head)
}

/// Everything of a record up to and including its kind tag — the fields every
/// kind shares, read in one place so the two bodies below cannot disagree
/// about where they start.
pub(super) struct RecordHead {
    pub(super) name: String,
    pub(super) d_out: usize,
    pub(super) d_in: usize,
    pub(super) shell_cap: u32,
    pub(super) kind: CodeKind,
}

pub(super) fn get_record_head(r: &mut impl Read, version: u32) -> Result<RecordHead> {
    let n = get_u32(r, "name length")? as usize;
    let name = get_bytes(r, n as u64, "name")?;
    let name = String::from_utf8(name).map_err(|_| Error::BadName)?;
    let d_out = get_u32(r, "d_out")? as usize;
    let d_in = get_u32(r, "d_in")? as usize;
    let shell_cap = get_u32(r, "shell cap")?;
    // The record's own kind, before anything is derived from it. An unknown
    // tag stops here: past it the width, the gain field and every byte that
    // follows are a guess.
    let kind = if version >= FIRST_KINDED_VERSION {
        CodeKind::from_tag(get_u32(r, "record code kind")?)?
    } else {
        CodeKind::Ball
    };
    Ok(RecordHead {
        name,
        d_out,
        d_in,
        shell_cap,
        kind,
    })
}

/// The rest of a lattice record, after [`get_record_head`].
///
/// An [`CodeKind::Int4G128`] record is refused here, by name, before a width
/// is derived. That refusal is what protects every lattice-only tool of this
/// repository — the benches, the five Metal binaries, the two CUDA ones,
/// `fused/`, `export` — without any of them being touched, and it holds
/// even for a file whose header under-declares its own [`KindSet`].
pub(super) fn read_lattice_body(r: &mut impl Read, head: RecordHead) -> Result<RawMatrix> {
    let RecordHead {
        name,
        d_out,
        d_in,
        shell_cap,
        kind,
    } = head;
    if kind == CodeKind::Int4G128 {
        return Err(Error::NotALatticeRecord { name, kind });
    }
    // Validated before any width is derived: `index_bits` asserts on the
    // supported ball inside llvq-search, and a wild cap from a corrupted file
    // must be an `Err`, not a panic — for either kind.
    let ib = index_width(kind, &name, shell_cap)?;
    let n_cent = get_u32(r, "centroid count")? as usize;
    let gb = gain_width(kind, &name, n_cent)?;
    // A record with blocks and no centroid has no gain to give any of them.
    // Tetra cannot reach this: `gain_width` has already pinned it to two.
    if n_cent == 0 && d_out > 0 && d_in >= DIM {
        return Err(Error::Inconsistent {
            name,
            detail: format!(
                "no gain centroid for {} blocks: every block's gain rank names one",
                d_out * (d_in / DIM)
            ),
        });
    }
    let seed = get_u64(r, "rotation seed")?;
    let has_rot = get_u32(r, "rotation flag")? != 0;
    // Before anything is allocated from the record: with no rows, nothing
    // after this point would ever check `d_in` against the bytes.
    check_rotation(&name, d_in, has_rot)?;

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
    // One compare a block: a Ball index past `N(cap)` decodes, through the map
    // of the whole ball, to a point of a shell the record says it does not use.
    let max_idx = max_index(kind, shell_cap);
    let mut br = BitReader::new(&bytes);
    let mut indices = Vec::with_capacity(nblocks.min(PREALLOC_CAP));
    let mut gains = Vec::with_capacity(nblocks.min(PREALLOC_CAP));
    for _ in 0..nblocks {
        let idx = br.read(ib);
        if idx > max_idx {
            return Err(Error::IndexOutOfRange {
                name: name.clone(),
                index: idx,
            });
        }
        indices.push(idx);
        // A field of `gb` bits holds ranks up to `2^gb - 1`, past the last
        // centroid when the count is not a power of two. On Tetra the count
        // is two and the field one bit, so this never refuses there.
        let gain = br.read(gb) as u32;
        if gain as usize >= n_cent {
            return Err(gain_out_of_range(&name, gain, n_cent));
        }
        gains.push(gain);
    }

    Ok(RawMatrix {
        name,
        d_out,
        d_in,
        kind,
        indices,
        gains,
        row_scales,
        centroids,
        rotation_seed: has_rot.then_some(seed),
        shell_cap,
        tail,
    })
}
