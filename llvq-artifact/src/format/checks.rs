//! The little-endian field primitives and the checks both writers and the
//! reader share: field widths, the tail count, the rotation bound and the
//! allocation cap. Kept together so a writer and its reader cannot disagree.

use super::kinds::{CodeKind, TETRA_SHELL_CAP};
use crate::{Error, Result};
use llvq_core::DIM;
use llvq_search::shapegain::index_bits;
use llvq_search::tetra::LABEL_BITS;
use std::io::{Read, Write};
// Named only by the documentation's links, which resolved in the one file.
#[cfg(doc)]
use super::Codebook;

pub(super) fn put_u32(w: &mut impl Write, v: u32) -> Result<()> {
    w.write_all(&v.to_le_bytes())?;
    Ok(())
}
pub(super) fn put_u64(w: &mut impl Write, v: u64) -> Result<()> {
    w.write_all(&v.to_le_bytes())?;
    Ok(())
}
pub(super) fn get_u32(r: &mut impl Read, what: &'static str) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).map_err(|_| Error::Truncated { reading: what })?;
    Ok(u32::from_le_bytes(b))
}
pub(super) fn get_u64(r: &mut impl Read, what: &'static str) -> Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).map_err(|_| Error::Truncated { reading: what })?;
    Ok(u64::from_le_bytes(b))
}

/// Cap on any allocation sized from a length field of the file. A length
/// field is a claim, not a fact: reserving gigabytes because a corrupted u64
/// says so would abort the process on OOM before the reads could fail.
/// Legitimate vectors larger than this grow as their elements arrive.
pub(super) const PREALLOC_CAP: usize = 1 << 20;

/// `v` as the u32 field that stores it. A value past `u32::MAX` is refused by
/// name: written with `as` it would wrap, and read back as another record.
pub(crate) fn u32_field(name: &str, what: &str, v: usize) -> Result<u32> {
    u32::try_from(v).map_err(|_| Error::Inconsistent {
        name: name.to_string(),
        detail: format!("{what} {v} does not fit the u32 field that stores it"),
    })
}

/// The tail a `d_out × d_in` record carries: `d_in mod 24` values per row,
/// the count `read_lattice_body` reads back. A writer that stored another
/// count would shift every byte after the tail, and the reader would take a
/// tail value for the code length.
pub(super) fn check_tail(name: &str, d_out: usize, d_in: usize, tail_len: usize) -> Result<()> {
    let want = d_out * (d_in % DIM);
    if tail_len != want {
        return Err(Error::Inconsistent {
            name: name.to_string(),
            detail: format!(
                "{tail_len} tail values for {d_out} rows of {} columns past the last block \
                 (want {want})",
                d_in % DIM
            ),
        });
    }
    Ok(())
}

/// The largest odd factor of a rotated record's `d_in`.
///
/// `Rotation::new(d_in, seed)` builds a dense `k x k` block by Gram-Schmidt,
/// `k` the odd factor of `d_in`: O(k^2) memory and O(k^3) time from one u32
/// field, which on a record with no rows nothing else in the file backs. The
/// widths the product rotates have odd factors of at most 25: Qwen3 0.6B
/// (1024, 2048, 3072: 1 and 3), 4B (2560, 4096, 9728: 5, 1, 19), 8B (4096,
/// 12288: 1, 3), 14B (5120, 17408: 5, 17), 32B (5120, 8192, 25600: 5, 1, 25),
/// Inkling 2 Mini (1152, 2304: 9). The served kernels stop at 32
/// (`ROT_KMAX` in `llvq-cuda`). 255 is ten times the widest of them and costs
/// a fraction of a second to build; 2001 cost 74 s.
pub const MAX_ROTATION_ODD_FACTOR: usize = 255;

/// The largest `d_in` a rotated record may carry.
///
/// The rotation keeps one sign per column, so `d_in` alone sizes an
/// allocation even when its odd factor is 1. The widest rotated width above
/// is 25,600; 2^20 is forty times that, and its sign table is 8 MB.
pub const MAX_ROTATED_D_IN: usize = 1 << 20;

/// The rotation a record asks for has to be one a decoder can build: a
/// positive width (`Rotation::new` asserts it), no wider than
/// [`MAX_ROTATED_D_IN`], with an odd factor no larger than
/// [`MAX_ROTATION_ODD_FACTOR`]. Both writers and the reader call this, so a
/// writer never emits a record its own reader refuses.
pub(super) fn check_rotation(name: &str, d_in: usize, rotated: bool) -> Result<()> {
    if !rotated {
        return Ok(());
    }
    let refuse = |why: String| {
        Err(Error::Inconsistent {
            name: name.to_string(),
            detail: format!("a rotation of d_in {d_in}: {why}"),
        })
    };
    if d_in == 0 {
        return refuse("a rotation has no zero-width form".to_string());
    }
    if d_in > MAX_ROTATED_D_IN {
        return refuse(format!("wider than the {MAX_ROTATED_D_IN} a rotated record may carry"));
    }
    let odd = d_in >> d_in.trailing_zeros();
    if odd > MAX_ROTATION_ODD_FACTOR {
        return refuse(format!(
            "its odd factor {odd} is past {MAX_ROTATION_ODD_FACTOR}, and the decoder would \
             build a dense {odd} x {odd} block for it"
        ));
    }
    Ok(())
}

/// A gain rank with no centroid behind it. The gain field is
/// `ceil(log2 n)` bits wide, so when `n` is not a power of two it can hold
/// ranks `n..2^bits` that `reconstruct_shape_gain` would index past the
/// centroids.
pub(super) fn gain_out_of_range(name: &str, gain: u32, n_centroids: usize) -> Error {
    Error::Inconsistent {
        name: name.to_string(),
        detail: format!("gain rank {gain} with {n_centroids} centroids: no centroid has that rank"),
    }
}

/// Read exactly `n` bytes declared by a length field of the file.
///
/// The buffer grows only as bytes actually arrive, so a lying length on a
/// short stream returns [`Error::Truncated`] instead of aborting on an
/// allocation the file could never back.
pub(super) fn get_bytes(r: &mut impl Read, n: u64, what: &'static str) -> Result<Vec<u8>> {
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
/// Tetra: the width is [`LABEL_BITS`] whatever the field says, and the field
/// must say [`TETRA_SHELL_CAP`] — a Tetra record with any other value was not
/// written by this crate, and the only honest reading of it is a refusal.
pub(super) fn index_width(kind: CodeKind, name: &str, shell_cap: u32) -> Result<u32> {
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
        CodeKind::Tetra => {
            if shell_cap != TETRA_SHELL_CAP {
                return Err(Error::Inconsistent {
                    name: name.to_string(),
                    detail: format!(
                        "shell cap {shell_cap} on a Tetra record: the field is not read for \
                         Tetra and is written as {TETRA_SHELL_CAP}"
                    ),
                });
            }
            Ok(LABEL_BITS)
        }
        CodeKind::Int4G128 => Err(Error::Inconsistent {
            name: name.to_string(),
            detail: "an Int4G128 record has no lattice index width".to_string(),
        }),
    }
}

/// The largest index a record of `kind` and `shell_cap` can hold.
///
/// Ball: `N(cap)`, the point count of `Λ₂₄(cap)`, the origin being index 0.
/// The index map lays the shells out in increasing order
/// ([`llvq_search::index`]), so an index past `N(cap)` fits the field of
/// [`index_bits`]`(cap)` and still names a point of a shell above the cap: the
/// encoder at that cap never writes one. The count is the sum [`index_bits`]
/// takes, over `ClassSet::shell_cardinality`. Call it after [`index_width`],
/// which refuses a cap that `enumerate_classes` would assert on.
///
/// Tetra: every value of the 47-bit field is a label, so no bound but the
/// width.
pub(super) fn max_index(kind: CodeKind, shell_cap: u32) -> u64 {
    match kind {
        CodeKind::Ball => {
            let cs = llvq_search::classes::enumerate_classes(shell_cap);
            (2..=shell_cap).map(|m| cs.shell_cardinality(m)).sum()
        }
        CodeKind::Tetra | CodeKind::Int4G128 => u64::MAX,
    }
}

/// The gain width of a record. A Tetra word has exactly one gain bit, at bit
/// 47: two centroids, no more, no fewer — a wider gain field would push the
/// word past 48 bits and [`Codebook::decode`]'s `gain << 47` past the word.
pub(super) fn gain_width(kind: CodeKind, name: &str, n_centroids: usize) -> Result<u32> {
    if kind == CodeKind::Int4G128 {
        return Err(Error::Inconsistent {
            name: name.to_string(),
            detail: "an Int4G128 record has no gain field".to_string(),
        });
    }
    let gb = n_centroids.next_power_of_two().trailing_zeros();
    if kind == CodeKind::Tetra && gb != 1 {
        return Err(Error::Inconsistent {
            name: name.to_string(),
            detail: format!(
                "{n_centroids} centroids on a Tetra record: a Tetra word carries one gain bit \
                 at bit {LABEL_BITS}, so a Tetra matrix has two"
            ),
        });
    }
    Ok(gb)
}
