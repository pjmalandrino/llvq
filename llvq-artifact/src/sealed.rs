//! Everything a model needs beyond its quantized projections.
//!
//! A file holding only the 252 linear layers is not a deployable model: it
//! still needs the embedding, the norms, the config and the tokenizer, which
//! means it still needs the original checkpoint next to it. A 981 MB file that
//! requires 8 GB of company is not the point.
//!
//! So the format carries two more sections:
//!
//! * **raw tensors** — anything the quantizer did not touch. Embeddings and
//!   RMSNorm weights, stored as f16. Not "the tensors we listed", but *every
//!   tensor in the checkpoint that is not a quantized projection*: taking the
//!   complement rather than an allow-list is what stops a future architecture
//!   from silently losing a weight nobody thought to enumerate.
//! * **blobs** — `config.json` and `tokenizer.json`, byte for byte.
//!
//! The result opens with no network, no Hugging Face cache, and no checkpoint.

use crate::{Error, Result};
use std::io::{Read, Write};

/// A tensor stored verbatim, in f16.
///
/// f16 and not f32: these are the weights the model already ran at, and
/// doubling 778 MB of embedding to keep bits the forward pass discards would
/// be paying for nothing.
pub struct RawTensor {
    pub name: String,
    pub dims: Vec<usize>,
    /// Row-major, `dims.iter().product()` values, IEEE binary16.
    pub data: Vec<u16>,
}

impl RawTensor {
    pub fn len(&self) -> usize {
        self.dims.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Bytes this tensor occupies in the stream.
    pub fn bytes(&self) -> u64 {
        self.data.len() as u64 * 2
    }
}

/// A file carried along verbatim — config, tokenizer.
pub struct Blob {
    pub name: String,
    pub bytes: Vec<u8>,
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
    r.read_exact(&mut b)
        .map_err(|_| Error::Truncated { reading: what })?;
    Ok(u32::from_le_bytes(b))
}
fn get_u64(r: &mut impl Read, what: &'static str) -> Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)
        .map_err(|_| Error::Truncated { reading: what })?;
    Ok(u64::from_le_bytes(b))
}
fn put_str(w: &mut impl Write, s: &str) -> Result<()> {
    put_u32(w, s.len() as u32)?;
    w.write_all(s.as_bytes())?;
    Ok(())
}
fn get_str(r: &mut impl Read, what: &'static str) -> Result<String> {
    let n = get_u32(r, what)? as usize;
    let mut b = vec![0u8; n];
    r.read_exact(&mut b)
        .map_err(|_| Error::Truncated { reading: what })?;
    String::from_utf8(b).map_err(|_| Error::BadName)
}

pub fn write_raw(w: &mut impl Write, t: &RawTensor) -> Result<u64> {
    put_str(w, &t.name)?;
    put_u32(w, t.dims.len() as u32)?;
    for d in &t.dims {
        put_u64(w, *d as u64)?;
    }
    if t.data.len() != t.len() {
        return Err(Error::Inconsistent {
            name: t.name.clone(),
            detail: format!("{} values for dims {:?}", t.data.len(), t.dims),
        });
    }
    put_u64(w, t.data.len() as u64)?;
    // Little-endian f16, written in bulk rather than value by value: 389 M
    // half-words through `write_all` per element is minutes of syscalls.
    let mut bytes = Vec::with_capacity(t.data.len() * 2);
    for v in &t.data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    w.write_all(&bytes)?;
    Ok(t.bytes())
}

pub fn read_raw(r: &mut impl Read) -> Result<RawTensor> {
    let name = get_str(r, "raw tensor name")?;
    let rank = get_u32(r, "raw tensor rank")? as usize;
    let mut dims = Vec::with_capacity(rank);
    for _ in 0..rank {
        dims.push(get_u64(r, "raw tensor dims")? as usize);
    }
    let n = get_u64(r, "raw tensor length")? as usize;
    let mut bytes = vec![0u8; n * 2];
    r.read_exact(&mut bytes).map_err(|_| Error::Truncated {
        reading: "raw tensor data",
    })?;
    let data = bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    Ok(RawTensor { name, dims, data })
}

pub fn write_blob(w: &mut impl Write, b: &Blob) -> Result<u64> {
    put_str(w, &b.name)?;
    put_u64(w, b.bytes.len() as u64)?;
    w.write_all(&b.bytes)?;
    Ok(b.bytes.len() as u64)
}

pub fn read_blob(r: &mut impl Read) -> Result<Blob> {
    let name = get_str(r, "blob name")?;
    let n = get_u64(r, "blob length")? as usize;
    let mut bytes = vec![0u8; n];
    r.read_exact(&mut bytes)
        .map_err(|_| Error::Truncated { reading: "blob" })?;
    Ok(Blob { name, bytes })
}
