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
//!   RMSNorm weights. Not "the tensors we listed", but *every tensor in the
//!   checkpoint that is not a quantized projection*: taking the complement
//!   rather than an allow-list is what stops a future architecture from
//!   silently losing a weight nobody thought to enumerate.
//! * **blobs** — `config.json` and `tokenizer.json`, byte for byte.
//!
//! The result opens with no network, no Hugging Face cache, and no checkpoint.
//!
//! ## Raw tensor encodings, and why there are two
//!
//! Up to `LVQ2` every raw tensor was f16 — and the embedding, one tensor, was
//! 44 % of the sealed file (778 MB of 1.77 GB on Qwen3-4B). `LVQ3` lets a raw
//! tensor be stored **group-affine quantized** instead: `w = scale·q + bias`
//! over groups of `group` values along the innermost dimension, `q` in
//! `[0, 2^bits)`, scale and bias one f16 each per group. This is exactly the
//! scheme MLX ships as `q4`/`q8` (group size 64), which is the evidence that
//! it holds an embedding without measurable damage. At 4 bits + 2×16/64 the
//! embedding costs 4.5 bits/weight instead of 16.
//!
//! Norms stay f16: quantizing a 2560-value tensor saves nothing and risks the
//! one place a scalar actually matters.
//!
//! Legacy `LVQ1`/`LVQ2` files carry untagged f16 records and remain readable;
//! [`read_raw`] takes the header version to know which framing to expect.

use crate::{Error, Result};
use std::io::{Read, Write};

/// Encoding tag of a raw tensor record in an `LVQ3` stream.
const TAG_F16: u32 = 0;
const TAG_QUANT: u32 = 1;

/// Widen one IEEE binary16 to f32. Exact — widening has no rounding.
///
/// Hand-rolled so this crate keeps zero dependencies; `llvq-llm` pins it
/// against the `half` crate for all 65 536 bit patterns.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    let magnitude = match (exp, frac) {
        (0, 0) => 0,
        (0, f) => {
            // Subnormal: value = f · 2⁻²⁴, normalized into an f32.
            let z = f.leading_zeros(); // 22..=31 for f in 1..=1023
            ((134 - z) << 23) | ((f << (z - 8)) & 0x007f_ffff)
        }
        (0x1f, f) => (0xff << 23) | (f << 13), // ±inf and NaN
        (e, f) => ((e + 112) << 23) | (f << 13),
    };
    f32::from_bits(sign | magnitude)
}

/// A group-affine quantized payload: `w = scale·q + bias`.
///
/// Groups run along the **innermost** dimension; when the row length is not a
/// multiple of `group`, the last group of each row is short. `q` values are
/// stored in flat row-major order — one byte each at 8 bits, two per byte at
/// 4 bits with the **low nibble first**.
pub struct QuantData {
    /// 4 or 8.
    pub bits: u8,
    /// Values per scale/bias pair, along the innermost dimension.
    pub group: usize,
    pub packed: Vec<u8>,
    /// One IEEE binary16 per group, `rows × ceil(row_len / group)`, row-major.
    pub scales: Vec<u16>,
    /// Same shape and order as `scales`.
    pub biases: Vec<u16>,
}

/// What a raw tensor's values are, on the wire.
pub enum RawData {
    /// Row-major, `dims.iter().product()` values, IEEE binary16.
    F16(Vec<u16>),
    Quant(QuantData),
}

/// A tensor the quantizer did not touch, carried in the sealed file.
///
/// f16 (or group-affine int8/int4) and not f32: these are the weights the
/// model already ran at, and doubling 778 MB of embedding to keep bits the
/// forward pass discards would be paying for nothing.
pub struct RawTensor {
    pub name: String,
    pub dims: Vec<usize>,
    pub data: RawData,
}

impl RawTensor {
    pub fn len(&self) -> usize {
        self.dims.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Bytes this tensor's payload occupies in the stream (framing excluded).
    pub fn bytes(&self) -> u64 {
        match &self.data {
            RawData::F16(d) => d.len() as u64 * 2,
            RawData::Quant(q) => {
                q.packed.len() as u64 + (q.scales.len() + q.biases.len()) as u64 * 2
            }
        }
    }

    /// Decode to f32, whatever the encoding. This is the reader's one entry
    /// point: quality measurements must go through the same bytes the file
    /// stores, not through a shortcut.
    pub fn to_f32(&self) -> Vec<f32> {
        match &self.data {
            RawData::F16(d) => d.iter().map(|&b| f16_to_f32(b)).collect(),
            RawData::Quant(q) => {
                let n = self.len();
                let row_len = self.dims.last().copied().unwrap_or(1).max(1);
                let gpr = row_len.div_ceil(q.group);
                let mut out = Vec::with_capacity(n);
                for i in 0..n {
                    let (row, col) = (i / row_len, i % row_len);
                    let g = row * gpr + col / q.group;
                    let qv = match q.bits {
                        8 => q.packed[i] as u32,
                        _ => ((q.packed[i / 2] >> (4 * (i % 2))) & 0xf) as u32,
                    };
                    out.push(f16_to_f32(q.scales[g]) * qv as f32 + f16_to_f32(q.biases[g]));
                }
                out
            }
        }
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
    let n = get_u32(r, what)?;
    let b = get_bytes(r, n as u64, what)?;
    String::from_utf8(b).map_err(|_| Error::BadName)
}

/// Cap on any allocation sized from a length field of the file — same
/// reasoning as its twin in `format.rs`: a corrupted length must fail as
/// [`Error::Truncated`] when the bytes run out, not abort on OOM before the
/// read gets a chance to.
const PREALLOC_CAP: usize = 1 << 20;

/// Read exactly `n` bytes declared by a length field of the file, growing the
/// buffer only as bytes actually arrive.
fn get_bytes(r: &mut impl Read, n: u64, what: &'static str) -> Result<Vec<u8>> {
    let mut buf = Vec::with_capacity(n.min(PREALLOC_CAP as u64) as usize);
    let got = (&mut *r).take(n).read_to_end(&mut buf)?;
    if (got as u64) < n {
        return Err(Error::Truncated { reading: what });
    }
    Ok(buf)
}

fn put_u16s(w: &mut impl Write, vals: &[u16]) -> Result<()> {
    // In bulk rather than value by value: 389 M half-words through
    // `write_all` per element is minutes of syscalls.
    let mut bytes = Vec::with_capacity(vals.len() * 2);
    for v in vals {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    w.write_all(&bytes)?;
    Ok(())
}
fn get_u16s(r: &mut impl Read, n: usize, what: &'static str) -> Result<Vec<u16>> {
    // `n` can come straight from a length field; a count whose byte size
    // overflows u64 names data no file could hold, so it truncates by
    // definition rather than wrapping into a small, wrong allocation.
    let nbytes = (n as u64)
        .checked_mul(2)
        .ok_or(Error::Truncated { reading: what })?;
    let bytes = get_bytes(r, nbytes, what)?;
    Ok(bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

/// Serialize one raw tensor in `LVQ3` framing (tagged).
pub fn write_raw(w: &mut impl Write, t: &RawTensor) -> Result<u64> {
    let tag = match &t.data {
        RawData::F16(_) => TAG_F16,
        RawData::Quant(_) => TAG_QUANT,
    };
    put_u32(w, tag)?;
    put_str(w, &t.name)?;
    put_u32(w, t.dims.len() as u32)?;
    for d in &t.dims {
        put_u64(w, *d as u64)?;
    }
    match &t.data {
        RawData::F16(data) => {
            if data.len() != t.len() {
                return Err(Error::Inconsistent {
                    name: t.name.clone(),
                    detail: format!("{} values for dims {:?}", data.len(), t.dims),
                });
            }
            put_u64(w, data.len() as u64)?;
            put_u16s(w, data)?;
        }
        RawData::Quant(q) => {
            let n = t.len();
            let row_len = t.dims.last().copied().unwrap_or(1).max(1);
            let rows = n / row_len;
            let groups = rows * row_len.div_ceil(q.group);
            let need = match q.bits {
                8 => n,
                4 => n.div_ceil(2),
                b => {
                    return Err(Error::Inconsistent {
                        name: t.name.clone(),
                        detail: format!("unsupported quant width {b}"),
                    })
                }
            };
            if q.packed.len() != need || q.scales.len() != groups || q.biases.len() != groups {
                return Err(Error::Inconsistent {
                    name: t.name.clone(),
                    detail: format!(
                        "{} packed bytes / {} scales / {} biases for {n} values in \
                         groups of {} (want {need} / {groups} / {groups})",
                        q.packed.len(),
                        q.scales.len(),
                        q.biases.len(),
                        q.group
                    ),
                });
            }
            put_u32(w, q.bits as u32)?;
            put_u32(w, q.group as u32)?;
            put_u64(w, q.packed.len() as u64)?;
            w.write_all(&q.packed)?;
            put_u64(w, groups as u64)?;
            put_u16s(w, &q.scales)?;
            put_u16s(w, &q.biases)?;
        }
    }
    Ok(t.bytes())
}

/// Read one raw tensor. `version` is the file header's — `LVQ1`/`LVQ2`
/// records are untagged f16, `LVQ3` records carry an encoding tag.
pub fn read_raw(r: &mut impl Read, version: u32) -> Result<RawTensor> {
    let tag = if version >= 3 {
        get_u32(r, "raw tensor tag")?
    } else {
        TAG_F16
    };
    // Refused before anything else is parsed: past an unknown tag, every
    // following byte would be interpreted against the wrong layout.
    if tag != TAG_F16 && tag != TAG_QUANT {
        return Err(Error::BadRawEncoding { tag });
    }
    let name = get_str(r, "raw tensor name")?;
    let rank = get_u32(r, "raw tensor rank")? as usize;
    let mut dims = Vec::with_capacity(rank.min(PREALLOC_CAP));
    for _ in 0..rank {
        dims.push(get_u64(r, "raw tensor dims")? as usize);
    }
    let data = match tag {
        TAG_F16 => {
            let n = get_u64(r, "raw tensor length")? as usize;
            RawData::F16(get_u16s(r, n, "raw tensor data")?)
        }
        TAG_QUANT => {
            let bits = get_u32(r, "quant width")? as u8;
            let group = get_u32(r, "quant group")? as usize;
            let nbytes = get_u64(r, "quant packed length")?;
            let packed = get_bytes(r, nbytes, "quant packed data")?;
            let groups = get_u64(r, "quant group count")? as usize;
            let scales = get_u16s(r, groups, "quant scales")?;
            let biases = get_u16s(r, groups, "quant biases")?;
            RawData::Quant(QuantData {
                bits,
                group,
                packed,
                scales,
                biases,
            })
        }
        tag => return Err(Error::BadRawEncoding { tag }),
    };
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
    let n = get_u64(r, "blob length")?;
    let bytes = get_bytes(r, n, "blob")?;
    Ok(Blob { name, bytes })
}
