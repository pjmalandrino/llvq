//! Reading and writing the `LVQ1` stream.
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

use crate::{Error, Result};
use llvq_core::DIM;
use llvq_quant::quantizer::BlockCode;
use llvq_search::index::Indexer;
use llvq_search::pack::{BitReader, BitWriter};
use std::io::{Read, Write};

/// Format identifier. Any change to the layout below bumps this.
///
/// `LVQ1` held quantized projections and nothing else, so a file needed the
/// original checkpoint beside it to run. `LVQ2` adds the raw tensors and the
/// blobs that make it self-contained. `LVQ3` tags each raw tensor with its
/// encoding so the embedding can be carried group-affine quantized instead of
/// f16 (see [`crate::sealed`]). `LVQ4` appends the writer's codebook
/// fingerprint to the header (see [`crate::codebook`]), which is the only
/// field of the whole format that describes what an index *means* rather than
/// how wide it is. All four are readable; only `LVQ4` is written. The matrix
/// records themselves are identical across versions — which is what lets a
/// tool copy them between two files of different versions untouched.
pub const MAGIC: &[u8; 4] = b"LVQ4";
pub const MAGIC_V3: &[u8; 4] = b"LVQ3";
pub const MAGIC_V2: &[u8; 4] = b"LVQ2";
pub const MAGIC_V1: &[u8; 4] = b"LVQ1";

/// The version [`ArtifactWriter`] emits.
pub const VERSION: u32 = 4;

/// First version whose header carries a codebook fingerprint.
///
/// Files below it were written before the field existed and are read without
/// the check — grandfathered deliberately: the published Qwen3-4B artifact is
/// `LVQ2`, and a reader that refused it would be worse than the hole it
/// closes. What protects *those* files is the pinned fingerprint in the test
/// suite, not a field they cannot have.
pub const FIRST_FINGERPRINTED_VERSION: u32 = 4;

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
    /// Shell cap of the direction code, which sets the index width.
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
        llvq_quant::quantizer::index_bits(self.shell_cap)
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

/// Serialize one matrix, returning the bits its payload occupies.
///
/// The `Indexer` is shared across calls — building it enumerates 383 classes
/// and has no business happening per matrix.
pub fn write_matrix(w: &mut impl Write, ix: &Indexer, m: &QuantizedMatrix) -> Result<u64> {
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

    let name = m.name.as_bytes();
    put_u32(w, name.len() as u32)?;
    w.write_all(name)?;
    put_u32(w, m.d_out as u32)?;
    put_u32(w, m.d_in as u32)?;
    put_u32(w, m.shell_cap)?;
    put_u32(w, m.centroids.len() as u32)?;
    put_u64(w, m.rotation_seed.unwrap_or(0))?;
    put_u32(w, m.rotation_seed.is_some() as u32)?;
    for c in &m.centroids {
        put_u64(w, c.to_bits())?;
    }
    for s in &m.row_scales {
        put_u64(w, s.to_bits())?;
    }
    for t in &m.tail {
        put_u32(w, (*t as f32).to_bits())?;
    }

    let (ib, gb) = (m.index_bits(), m.gain_bits());
    let mut bw = BitWriter::with_capacity(m.codes.len() as u64 * (ib + gb) as u64);
    for c in &m.codes {
        let idx = ix.encode(&c.point).ok_or_else(|| Error::PointOutsideCodebook {
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
/// byte-identity is pinned by `raw_passthrough_is_byte_identical`.
pub fn write_matrix_raw(w: &mut impl Write, m: &RawMatrix) -> Result<u64> {
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

    let name = m.name.as_bytes();
    put_u32(w, name.len() as u32)?;
    w.write_all(name)?;
    put_u32(w, m.d_out as u32)?;
    put_u32(w, m.d_in as u32)?;
    put_u32(w, m.shell_cap)?;
    put_u32(w, m.centroids.len() as u32)?;
    put_u64(w, m.rotation_seed.unwrap_or(0))?;
    put_u32(w, m.rotation_seed.is_some() as u32)?;
    for c in &m.centroids {
        put_u64(w, c.to_bits())?;
    }
    for s in &m.row_scales {
        put_u64(w, s.to_bits())?;
    }
    for t in &m.tail {
        put_u32(w, (*t as f32).to_bits())?;
    }

    let ib = llvq_quant::quantizer::index_bits(m.shell_cap);
    let gb = m.centroids.len().next_power_of_two().trailing_zeros();
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

/// Read one matrix without decoding its indices.
pub fn read_matrix_raw(r: &mut impl Read) -> Result<RawMatrix> {
    let n = get_u32(r, "name length")? as usize;
    let mut name = vec![0u8; n];
    r.read_exact(&mut name)
        .map_err(|_| Error::Truncated { reading: "name" })?;
    let name = String::from_utf8(name).map_err(|_| Error::BadName)?;
    let d_out = get_u32(r, "d_out")? as usize;
    let d_in = get_u32(r, "d_in")? as usize;
    let shell_cap = get_u32(r, "shell cap")?;
    let n_cent = get_u32(r, "centroid count")? as usize;
    let seed = get_u64(r, "rotation seed")?;
    let has_rot = get_u32(r, "rotation flag")? != 0;

    let mut centroids = Vec::with_capacity(n_cent);
    for _ in 0..n_cent {
        centroids.push(f64::from_bits(get_u64(r, "centroids")?));
    }
    let mut row_scales = Vec::with_capacity(d_out);
    for _ in 0..d_out {
        row_scales.push(f64::from_bits(get_u64(r, "row scales")?));
    }
    let tail_w = d_out * (d_in % DIM);
    let mut tail = Vec::with_capacity(tail_w);
    for _ in 0..tail_w {
        tail.push(f32::from_bits(get_u32(r, "tail")?) as f64);
    }

    let nbytes = get_u64(r, "code length")? as usize;
    let mut bytes = vec![0u8; nbytes];
    r.read_exact(&mut bytes)
        .map_err(|_| Error::Truncated { reading: "code stream" })?;

    let ib = llvq_quant::quantizer::index_bits(shell_cap);
    let gb = centroids.len().next_power_of_two().trailing_zeros();
    let nblocks = d_out * (d_in / DIM);
    let mut br = BitReader::new(&bytes);
    let mut indices = Vec::with_capacity(nblocks);
    let mut gains = Vec::with_capacity(nblocks);
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

/// Read back what [`write_matrix`] wrote.
pub fn read_matrix(r: &mut impl Read, ix: &Indexer) -> Result<QuantizedMatrix> {
    let raw = read_matrix_raw(r)?;
    let mut codes = Vec::with_capacity(raw.indices.len());
    for (&idx, &gain) in raw.indices.iter().zip(&raw.gains) {
        let point = ix.decode(idx).ok_or(Error::IndexOutOfRange {
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

/// Rebuild the `d_out × d_in` weight matrix, in the **natural** basis, exactly
/// as the evaluated model holds it.
///
/// The order of operations mirrors the quantization loop: decode in the
/// rotated basis, restore the tail, un-rotate, and only then narrow to f32.
/// Doing the narrowing earlier, or un-rotating in f32, changes the last bits —
/// and the whole claim of this format is that it does not.
pub fn decode_matrix(m: &QuantizedMatrix) -> Vec<f32> {
    use llvq_quant::quantizer::LeechShapeGain;
    let q = LeechShapeGain::with_shell_cap(m.centroids.clone(), m.shell_cap);
    let nblocks = m.nblocks();
    let tail_w = m.d_in % DIM;

    let mut w = vec![0.0f64; m.d_out * m.d_in];
    let mut block = [0.0f64; DIM];
    for i in 0..m.d_out {
        for p in 0..nblocks {
            q.reconstruct(&m.codes[i * nblocks + p], m.row_scales[i], &mut block);
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
    ix: Indexer,
    pub matrices: u32,
    pub payload_bits: u64,
}

impl<W: Write> ArtifactWriter<W> {
    /// `n_matrices` is written up front so the reader can size itself.
    pub fn new(out: W, n_matrices: u32) -> Result<Self> {
        Self::with_version(out, VERSION, n_matrices)
    }

    /// Same, at a chosen format version — for a tool that rewrites an existing
    /// file and must not silently upgrade it.
    pub fn with_version(mut out: W, version: u32, n_matrices: u32) -> Result<Self> {
        write_header(&mut out, version, n_matrices)?;
        Ok(Self {
            out,
            ix: Indexer::new(),
            matrices: 0,
            payload_bits: 0,
        })
    }

    pub fn push(&mut self, m: &QuantizedMatrix) -> Result<()> {
        self.payload_bits += write_matrix(&mut self.out, &self.ix, m)?;
        self.matrices += 1;
        Ok(())
    }

    /// Push a matrix straight from its undecoded codes — see
    /// [`write_matrix_raw`].
    pub fn push_raw(&mut self, m: &RawMatrix) -> Result<()> {
        self.payload_bits += write_matrix_raw(&mut self.out, m)?;
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

/// What a file's header says: which version, how many matrices follow, and —
/// from [`FIRST_FINGERPRINTED_VERSION`] on — which codebook wrote it.
pub struct Header {
    /// 1 for projections only; 2+ for a self-contained file. 3 adds tagged
    /// raw-tensor encodings; 4 adds the codebook fingerprint.
    pub version: u32,
    pub matrices: u32,
    /// The writer's codebook fingerprint, or `None` for a legacy file that
    /// predates the field.
    ///
    /// A `Some` here has already been checked against
    /// [`crate::codebook_fingerprint`]: [`read_header`] refuses a file whose
    /// codebook is not this build's, so reaching a `Header` at all means the
    /// indices about to be read mean what the writer meant.
    pub codebook: Option<u64>,
}

impl Header {
    /// Whether the file carries embeddings, norms and tokenizer — i.e. whether
    /// it can run without the original checkpoint.
    pub fn is_self_contained(&self) -> bool {
        self.version >= 2
    }
}

/// Write a file header for `version`.
///
/// Exposed because the header is no longer `magic + count`: from
/// [`FIRST_FINGERPRINTED_VERSION`] on it also carries the fingerprint, and a
/// tool that rewrites a file's matrix section byte for byte has to reproduce
/// the whole of it. Hand-rolling the two writes is exactly how a `LVQ4` magic
/// ends up over a `LVQ3` body.
pub fn write_header(w: &mut impl Write, version: u32, matrices: u32) -> Result<()> {
    let magic = match version {
        1 => MAGIC_V1,
        2 => MAGIC_V2,
        3 => MAGIC_V3,
        4 => MAGIC,
        v => return Err(Error::UnknownVersion { version: v }),
    };
    w.write_all(magic)?;
    put_u32(w, matrices)?;
    if version >= FIRST_FINGERPRINTED_VERSION {
        put_u64(w, crate::codebook_fingerprint())?;
    }
    Ok(())
}

/// Read the file header, refusing a file this build cannot interpret.
///
/// Separate from [`read_all`] because a 4B model's codes are 14 GB of lattice
/// points: anything that walks a real artifact has to do it one matrix at a
/// time, and holding them all was never an option.
///
/// The fingerprint is checked here rather than at first decode so that a
/// codebook disagreement costs sixteen bytes of I/O instead of a gigabyte, and
/// so that every caller gets the check without asking for it.
pub fn read_header(r: &mut impl Read) -> Result<Header> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)
        .map_err(|_| Error::Truncated { reading: "magic" })?;
    let version = if &magic == MAGIC {
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
        let stored = get_u64(r, "codebook fingerprint")?;
        let computed = crate::codebook_fingerprint();
        if stored != computed {
            return Err(Error::CodebookMismatch { stored, computed });
        }
        Some(stored)
    } else {
        None
    };
    Ok(Header {
        version,
        matrices,
        codebook,
    })
}

/// Read every matrix back. Only safe for small models — see [`read_header`].
pub fn read_all(r: &mut impl Read) -> Result<Vec<QuantizedMatrix>> {
    let h = read_header(r)?;
    let ix = Indexer::new();
    (0..h.matrices).map(|_| read_matrix(r, &ix)).collect()
}
