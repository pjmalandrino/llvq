//! From stored codes back to weights: the decode of a raw record through its
//! map, the two decoding readers, and `decode_matrix`, the bit-exact
//! reconstruction in the natural basis.

use super::kinds::{CodeKind, Codebooks};
use super::read::read_matrix_raw;
use super::write::{QuantizedMatrix, RawMatrix};
use crate::{Error, Result};
use llvq_core::{Point, DIM};
use llvq_search::index::Indexer;
use llvq_search::shapegain::{reconstruct_shape_gain, BlockCode};
use std::io::Read;
// Named only by the documentation's links, which resolved in the one file.
#[cfg(doc)]
use super::{read_record, write_matrix, write_matrix_with, Record};

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

/// Decode a [`Record::Lattice`] payload through the codebook its kind names.
///
/// [`read_matrix_with`] is this composed with [`read_matrix_raw`], and that
/// pair refuses a mixed file by construction: it cannot step over an int4
/// record. A reader that walks [`read_record`] instead has already consumed
/// the record and knows which arm it got, so it needs the decode half on its
/// own. `export` is that reader.
pub fn decode_raw(raw: RawMatrix, cbs: &Codebooks) -> Result<QuantizedMatrix> {
    let cb = cbs.get(raw.kind)?;
    decode_codes(raw, &|idx, gain| cb.decode(idx, gain))
}

/// Read back what [`write_matrix`] wrote — a Ball record, refusing any other
/// kind by name.
///
/// The refusal is the point: this entry has one map, and a caller that reaches
/// a Tetra record through it wanted the ball. [`read_matrix_with`] is the entry
/// that reads whatever the record says it is.
///
/// Neither can reach an [`CodeKind::Int4G128`] record: both go through
/// [`read_matrix_raw`], which refuses one with [`Error::NotALatticeRecord`]
/// before a width is derived. That is where the guarantee lives, so it is
/// written here too rather than left implicit at two call sites.
pub fn read_matrix(r: &mut impl Read, version: u32, ix: &Indexer) -> Result<QuantizedMatrix> {
    let raw = read_matrix_raw(r, version)?;
    if raw.kind != CodeKind::Ball {
        return Err(Error::WrongCodeKind {
            name: raw.name,
            want: CodeKind::Ball,
            got: raw.kind,
        });
    }
    decode_codes(raw, &|idx, _| ix.decode(idx))
}

/// Read back what [`write_matrix_with`] wrote, through the map the **record**
/// names: Ball → `Indexer::decode(idx)`, Tetra →
/// `Tetra::decode(idx | gain << 47)`.
pub fn read_matrix_with(
    r: &mut impl Read,
    version: u32,
    cbs: &Codebooks,
) -> Result<QuantizedMatrix> {
    let raw = read_matrix_raw(r, version)?;
    let cb = cbs.get(raw.kind)?;
    decode_codes(raw, &|idx, gain| cb.decode(idx, gain))
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
/// Ball and a Tetra matrix rebuild through the same lines.
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
    // With no row there is nothing to un-rotate, and building the rotation
    // anyway would cost its dense block for an empty matrix.
    if let (Some(seed), true) = (m.rotation_seed, m.d_out > 0) {
        llvq_search::rotation::Rotation::new(m.d_in, seed).unrotate_weight_rows(&mut w, m.d_out);
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
