//! The state of an [`super::InklingCache`] as bytes, so a server can keep it
//! on disk and load it into a new cache (plan item P7,
//! `docs/plan-debit-inkling2-mini.md`). A cache loaded from these bytes holds
//! what the saved one held, value for value and buffer for buffer, so the
//! reads after it give the logits of the reads after the saved one, bit for
//! bit (`tests/inkling_prefix_state.rs`).
//!
//! Format, every integer little-endian, no padding:
//!
//! ```text
//! cache  := magic "INKCACHE" | u64 version (1) | u64 slots | u64 len | u64 n | n x u32 token | slots x slot
//! slot   := u64 len | u64 base | u64 min_capacity | u64 capacity
//!           | u8 bounded | [u64 window | u64 slack if bounded]
//!           | u8 has_kv | [kv if has_kv]
//!           | 4 x conv                           (key, value, post-attention, post-MLP)
//! kv     := u64 kv_heads | u64 head_dim | u64 held
//!           | tensor keys   [kv_heads, head_dim, held]
//!           | tensor values [kv_heads, held, head_dim]
//! conv   := u8 0 (empty) | u8 1, u64 rows, u64 cols, tensor [rows, cols]
//! tensor := u8 dtype (0 f32, 1 f16, 2 bf16) | the values row-major, raw bytes of that dtype
//! ```
//!
//! `held` is `len - base`: the positions the slot's buffers hold. A loaded slot
//! gets buffers of `capacity` positions with those rows at row 0, as the saved
//! one had: a bounded slot drops its oldest positions when its buffers are full,
//! so the capacity decides where the next drop happens, and with it which keys
//! the next rows' softmax reads. A convolution state held as a ring of one-row
//! handles is written as the block of its rows in time order: the same values.
//!
//! These bytes carry no integrity check and no key of the model or of the
//! settings: the caller (`llvq-runtime`'s prefix cache) wraps them in a file
//! that does.

use std::io::{Read, Write};

use candle_core::{DType, Device, Result, Tensor};

/// The first bytes of a cache state.
pub const MAGIC: &[u8; 8] = b"INKCACHE";
/// The version of the format above.
pub const VERSION: u64 = 1;

/// The largest tensor a state may declare, in bytes: a damaged length is
/// refused before anything is allocated for it.
const MAX_TENSOR_BYTES: u64 = 1 << 34;

fn io(e: std::io::Error) -> candle_core::Error {
    candle_core::Error::msg(format!("cache state: {e}"))
}

pub(crate) fn put_u64(w: &mut dyn Write, v: u64) -> Result<()> {
    w.write_all(&v.to_le_bytes()).map_err(io)
}

pub(crate) fn put_u8(w: &mut dyn Write, v: u8) -> Result<()> {
    w.write_all(&[v]).map_err(io)
}

pub(crate) fn get_u64(r: &mut dyn Read) -> Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).map_err(io)?;
    Ok(u64::from_le_bytes(b))
}

pub(crate) fn get_usize(r: &mut dyn Read, what: &str) -> Result<usize> {
    let v = get_u64(r)?;
    usize::try_from(v).map_err(|_| candle_core::Error::msg(format!("cache state: {what} {v} does not fit")))
}

pub(crate) fn get_u8(r: &mut dyn Read) -> Result<u8> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b).map_err(io)?;
    Ok(b[0])
}

fn dtype_tag(d: DType) -> Result<u8> {
    match d {
        DType::F32 => Ok(0),
        DType::F16 => Ok(1),
        DType::BF16 => Ok(2),
        o => candle_core::bail!("cache state: a tensor of dtype {o:?}, only f32, f16 and bf16 are written"),
    }
}

fn tag_dtype(t: u8) -> Result<DType> {
    match t {
        0 => Ok(DType::F32),
        1 => Ok(DType::F16),
        2 => Ok(DType::BF16),
        o => candle_core::bail!("cache state: dtype tag {o}"),
    }
}

/// The dtype tag of `t`, then its raw bytes in row-major order. `t` may be
/// a view: its values are what is written.
pub(crate) fn put_tensor_bytes(w: &mut dyn Write, t: &Tensor) -> Result<()> {
    put_u8(w, dtype_tag(t.dtype())?)?;
    let flat = t.flatten_all()?.to_device(&Device::Cpu)?;
    let bytes: Vec<u8> = match t.dtype() {
        DType::F32 => flat.to_vec1::<f32>()?.iter().flat_map(|x| x.to_le_bytes()).collect(),
        DType::F16 => flat.to_vec1::<half::f16>()?.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect(),
        DType::BF16 => flat.to_vec1::<half::bf16>()?.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect(),
        _ => unreachable!("refused by dtype_tag"),
    };
    w.write_all(&bytes).map_err(io)
}

/// A dtype tag then `shape`'s raw bytes, as a tensor on `dev`.
pub(crate) fn get_tensor_bytes(r: &mut dyn Read, shape: &[usize], dev: &Device) -> Result<Tensor> {
    let dtype = tag_dtype(get_u8(r)?)?;
    let n = shape.iter().try_fold(1u64, |a, &d| a.checked_mul(d as u64));
    let bytes = n.and_then(|n| n.checked_mul(dtype.size_in_bytes() as u64));
    let Some(bytes) = bytes.filter(|&b| b <= MAX_TENSOR_BYTES) else {
        candle_core::bail!("cache state: a tensor of shape {shape:?} is past the size a state may hold");
    };
    let mut buf = vec![0u8; bytes as usize];
    r.read_exact(&mut buf).map_err(io)?;
    Tensor::from_raw_buffer(&buf, dtype, shape, dev)
}
