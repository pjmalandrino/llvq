//! The int4 g128 record, its reader and writer, and the `Record` that walks a
//! mixed file. One file because this record shares only the head with the
//! lattice ones, and everything past the head is its own.

use super::checks::{get_bytes, get_u32, get_u64, put_u32, put_u64};
use super::kinds::{
    CodeKind, FIRST_KINDED_VERSION, INT4G128_BITS, INT4G128_GROUP, INT4G128_SHELL_CAP,
};
use super::read::{get_record_head, read_lattice_body, RecordHead};
use super::write::{put_record_head, write_matrix_raw, RawMatrix};
use crate::{Error, Result};
use std::io::{Read, Write};
// Named only by the documentation's links, which resolved in the one file.
#[cfg(doc)]
use super::read_matrix_raw;
#[cfg(doc)]
use llvq_search::pack::BitWriter;

// ---------------------------------------------------------------------------
// The int4 g128 record
// ---------------------------------------------------------------------------

/// One matrix stored as affine int4 over groups of [`INT4G128_GROUP`], in the
/// **natural** basis: `w = scale·q + bias`.
///
/// The reconstruction is the whole record. There is no map, no fingerprint,
/// no rotation and no tail, so the failure modes a lattice record has do not
/// exist here — and the one it does have, a group axis read along the wrong
/// dimension, is why the groups are pinned to `d_in` in one place
/// (`the_group_axis_runs_along_d_in`).
///
/// The encoder lives in `llvq_llm::embedquant`, not here: choosing `q` needs
/// f32→f16 **rounding**, and this crate has no dependencies to round with.
/// Decoding is exact widening and stays here, which is the same split
/// [`crate::RawTensor`] already makes.
pub struct Int4Matrix {
    pub name: String,
    pub d_out: usize,
    pub d_in: usize,
    /// [`INT4G128_BITS`]. Stored, not assumed, so a future width is a refusal
    /// and not a misread.
    pub bits: u8,
    /// [`INT4G128_GROUP`], along `d_in`.
    pub group: usize,
    /// `d_out · d_in / 2` bytes, row-major, **low nibble first** — the
    /// convention `llvq_llm::embedquant` writes and [`crate::RawTensor`]
    /// reads. A lattice index in the same file is packed MSB-first by
    /// [`BitWriter`]: two orders in one file, and a reader that assumed one
    /// convention for both would produce plausible, wrong weights.
    pub packed: Vec<u8>,
    /// One IEEE binary16 per group, `d_out × (d_in / group)`, row-major.
    pub scales: Vec<u16>,
    /// Same shape and order.
    pub biases: Vec<u16>,
}

impl Int4Matrix {
    /// Groups per output row. An exact division: `d_in % group != 0` is
    /// refused by the writer and the reader, because a short last group would
    /// make the rate below wrong.
    pub fn groups_per_row(&self) -> usize {
        self.d_in.checked_div(self.group).unwrap_or(0)
    }

    /// Bits this matrix occupies in the stream, framing excluded —
    /// `4 + 32/128 = 4.250` per weight exactly. Scale **and** bias are f16,
    /// where AWQ w4 g128 packs its zero and holds 4.15625: a declared
    /// difference of 0.09 b/weight, against us.
    pub fn bits(&self) -> u64 {
        self.packed.len() as u64 * 8 + (self.scales.len() + self.biases.len()) as u64 * 16
    }

    /// Rebuild the `d_out × d_in` weights, row-major, in the natural basis.
    ///
    /// Line for line the group-affine branch of
    /// [`crate::RawTensor::to_f32`], on purpose: the file path and the
    /// measurement path (`LLVQ_RESTORE_Q4`) have to agree bit for bit, and
    /// `the_two_int4_paths_agree_bit_for_bit` is what would notice if they
    /// stopped.
    pub fn to_f32(&self) -> Vec<f32> {
        let n = self.d_out * self.d_in;
        let gpr = self.groups_per_row();
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let (row, col) = (i / self.d_in, i % self.d_in);
            let g = row * gpr + col / self.group;
            let qv = ((self.packed[i / 2] >> (4 * (i % 2))) & 0xf) as u32;
            out.push(
                crate::sealed::f16_to_f32(self.scales[g]) * qv as f32
                    + crate::sealed::f16_to_f32(self.biases[g]),
            );
        }
        out
    }
}

/// One record of a v5 file, whatever its kind — the only thing that can be
/// read from a mixed file without knowing what comes next.
pub enum Record {
    /// A Ball or Tetra record, undecoded.
    Lattice(RawMatrix),
    Int4(Int4Matrix),
}

impl Record {
    pub fn name(&self) -> &str {
        match self {
            Self::Lattice(m) => &m.name,
            Self::Int4(m) => &m.name,
        }
    }

    pub fn kind(&self) -> CodeKind {
        match self {
            Self::Lattice(m) => m.kind,
            Self::Int4(_) => CodeKind::Int4G128,
        }
    }

    pub fn dims(&self) -> (usize, usize) {
        match self {
            Self::Lattice(m) => (m.d_out, m.d_in),
            Self::Int4(m) => (m.d_out, m.d_in),
        }
    }
}

/// Read one record of any kind, from a file of version `version`.
///
/// The head is read once and the kind decides the body: a lattice kind goes
/// to the same code [`read_matrix_raw`] runs, int4 to the reader below. This
/// is the only entry point that can walk a mixed file, and the reason every
/// other one refuses int4 rather than guessing.
pub fn read_record(r: &mut impl Read, version: u32) -> Result<Record> {
    let head = get_record_head(r, version)?;
    match head.kind {
        CodeKind::Int4G128 => Ok(Record::Int4(read_int4_body(r, head)?)),
        CodeKind::Ball | CodeKind::Tetra => Ok(Record::Lattice(read_lattice_body(r, head)?)),
    }
}

/// Write one record of any kind, at `version`, returning its payload bits.
pub fn write_record(w: &mut impl Write, version: u32, rec: &Record) -> Result<u64> {
    match rec {
        Record::Lattice(m) => write_matrix_raw(w, version, m),
        Record::Int4(m) => write_matrix_int4(w, version, m),
    }
}

/// The payload of an int4 record, after [`get_record_head`].
///
/// Every field of the head is checked against what this crate writes, and
/// each with its own message: three independent fields have to agree, so a
/// lattice record whose kind tag was corrupted to 2 contradicts itself in
/// three places rather than decoding to plausible weights
/// (`a_lattice_record_read_as_int4_is_refused`).
fn read_int4_body(r: &mut impl Read, head: RecordHead) -> Result<Int4Matrix> {
    let RecordHead {
        name,
        d_out,
        d_in,
        shell_cap,
        kind: _,
    } = head;
    let bad = |detail: String| Error::Inconsistent {
        name: name.clone(),
        detail,
    };
    if shell_cap != INT4G128_SHELL_CAP {
        return Err(bad(format!(
            "shell cap {shell_cap} on an Int4G128 record: it indexes no lattice and carries \
             {INT4G128_SHELL_CAP} in that field"
        )));
    }
    let n_cent = get_u32(r, "centroid count")? as usize;
    if n_cent != 0 {
        return Err(bad(format!(
            "{n_cent} centroids on an Int4G128 record: its weights are stored, so it has no \
             gain levels"
        )));
    }
    let _seed = get_u64(r, "rotation seed")?;
    let rot = get_u32(r, "rotation flag")?;
    if rot != 0 {
        return Err(bad(
            "a rotation seed on an Int4G128 record: it is stored in the natural basis, and \
             un-rotating it would decode plausible, wrong weights"
                .to_string(),
        ));
    }

    let payload_len = get_u64(r, "int4 payload length")?;
    let payload = get_bytes(r, payload_len, "int4 payload")?;
    let mut p = &payload[..];
    let bits = get_u32(&mut p, "int4 bits")?;
    let group = get_u32(&mut p, "int4 group")? as usize;
    if bits != u32::from(INT4G128_BITS) {
        return Err(bad(format!(
            "{bits} bits on an Int4G128 record: the only width this format stores is \
             {INT4G128_BITS}"
        )));
    }
    if group != INT4G128_GROUP {
        return Err(bad(format!(
            "group {group} on an Int4G128 record: the only group this format stores is \
             {INT4G128_GROUP}"
        )));
    }
    if !d_in.is_multiple_of(group) {
        return Err(bad(format!(
            "d_in {d_in} is not a multiple of the group {group}: a short last group would \
             cost more than {INT4G128_BITS} + 32/{INT4G128_GROUP} bits per weight and the \
             rate would be wrong"
        )));
    }
    let n = d_out
        .checked_mul(d_in)
        .ok_or(Error::Truncated { reading: "int4 payload" })?;
    let packed_len = get_u64(&mut p, "int4 packed length")?;
    if packed_len != (n / 2) as u64 {
        return Err(bad(format!(
            "{packed_len} packed bytes for {d_out}×{d_in} weights, expected {}",
            n / 2
        )));
    }
    let packed = get_bytes(&mut p, packed_len, "int4 packed")?;
    let groups = get_u64(&mut p, "int4 group count")?;
    let want_groups = (d_out * (d_in / group)) as u64;
    if groups != want_groups {
        return Err(bad(format!(
            "{groups} groups for {d_out} rows of {d_in}, expected {want_groups}"
        )));
    }
    let scales = get_u16s(&mut p, groups, "int4 scales")?;
    let biases = get_u16s(&mut p, groups, "int4 biases")?;
    if !p.is_empty() {
        return Err(bad(format!(
            "{} bytes left over in an Int4G128 payload of {payload_len}",
            p.len()
        )));
    }
    Ok(Int4Matrix {
        name,
        d_out,
        d_in,
        bits: INT4G128_BITS,
        group,
        packed,
        scales,
        biases,
    })
}

fn get_u16s(r: &mut impl Read, n: u64, what: &'static str) -> Result<Vec<u16>> {
    let bytes = get_bytes(r, n.checked_mul(2).ok_or(Error::Truncated { reading: what })?, what)?;
    Ok(bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

/// Serialize one int4 record into a file of version `version`, returning the
/// bits its payload occupies.
///
/// Every invariant is checked before a byte is written, for the same reason
/// [`write_matrix_raw`]'s are: a writer that emitted a record its own reader
/// refuses would put the failure a run away from its cause.
pub fn write_matrix_int4(w: &mut impl Write, version: u32, m: &Int4Matrix) -> Result<u64> {
    let bad = |detail: String| Error::Inconsistent {
        name: m.name.clone(),
        detail,
    };
    if version < FIRST_KINDED_VERSION {
        return Err(bad(format!(
            "an Int4G128 record needs format v{FIRST_KINDED_VERSION}: a v{version} record \
             carries no code kind, and this one would read back as Ball"
        )));
    }
    if m.bits != INT4G128_BITS {
        return Err(bad(format!(
            "{} bits: the only width this format stores is {INT4G128_BITS}",
            m.bits
        )));
    }
    if m.group != INT4G128_GROUP {
        return Err(bad(format!(
            "group {}: the only group this format stores is {INT4G128_GROUP}",
            m.group
        )));
    }
    if !m.d_in.is_multiple_of(m.group) {
        return Err(bad(format!(
            "d_in {} is not a multiple of the group {}",
            m.d_in, m.group
        )));
    }
    let n = m.d_out * m.d_in;
    if m.packed.len() != n / 2 {
        return Err(bad(format!(
            "{} packed bytes for {}×{} weights, expected {}",
            m.packed.len(),
            m.d_out,
            m.d_in,
            n / 2
        )));
    }
    let groups = m.d_out * m.groups_per_row();
    if m.scales.len() != groups || m.biases.len() != groups {
        return Err(bad(format!(
            "{} scales and {} biases for {groups} groups",
            m.scales.len(),
            m.biases.len()
        )));
    }

    put_record_head(
        w,
        version,
        &m.name,
        m.d_out,
        m.d_in,
        INT4G128_SHELL_CAP,
        CodeKind::Int4G128,
        &[],
        &[],
        None,
        &[],
    )?;

    // Built in memory and then length-prefixed, so the stored length is the
    // bytes written by construction rather than a formula that can drift from
    // the field list below it.
    let mut payload: Vec<u8> = Vec::with_capacity(16 + m.packed.len() + 8 + groups * 4);
    put_u32(&mut payload, u32::from(m.bits))?;
    put_u32(&mut payload, m.group as u32)?;
    put_u64(&mut payload, m.packed.len() as u64)?;
    payload.extend_from_slice(&m.packed);
    put_u64(&mut payload, groups as u64)?;
    for v in &m.scales {
        payload.extend_from_slice(&v.to_le_bytes());
    }
    for v in &m.biases {
        payload.extend_from_slice(&v.to_le_bytes());
    }
    put_u64(w, payload.len() as u64)?;
    w.write_all(&payload)?;
    Ok(m.bits())
}
