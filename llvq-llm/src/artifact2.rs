//! The real 2-bit artifact: lattice indices, not reconstructions.
//!
//! [`crate::artifact`] stores the floating-point weights a decoder *would*
//! produce — 6.8 GB for Qwen3-4B, useful for re-evaluating without
//! re-quantizing and useless as a deliverable. This module stores the codes:
//! one packed index per 24 weights, plus what it takes to turn them back into
//! weights. The file's size **is** the bit rate, which is the point — an
//! accounting error cannot hide in a file you have to write byte by byte.
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
//! ## Bit-exactness, and where the f64s are
//!
//! Scales and centroids are stored in **f64**, not f16. `reconstruct` computes
//! `centroids[g] * row_scale` in f64; storing a rounded scale would change
//! that product and the decoded weights would differ from the evaluated ones
//! in the last bits. That costs 64 bits per output row instead of 16 —
//! 0.0146 bits/weight on Qwen3-4B — and it is **counted**, not waved away.
//! Making the scales exactly f32-representable at fit time would recover it;
//! that is an optimization, not a correctness fix, and it is not done here.

use llvq_core::DIM;
use llvq_quant::quantizer::BlockCode;
use llvq_search::index::Indexer;
use llvq_search::pack::{BitReader, BitWriter};
use std::io::{Read, Write};

/// Format identifier. Any change to the layout below bumps this.
const MAGIC: &[u8; 4] = b"LVQ1";

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

    /// Bits this matrix occupies in the stream, header included.
    pub fn bits(&self) -> u64 {
        let per_block = (self.index_bits() + self.gain_bits()) as u64;
        self.codes.len() as u64 * per_block
            + self.row_scales.len() as u64 * 64
            + self.centroids.len() as u64 * 64
            + self.tail.len() as u64 * 32
    }
}

fn put_u32(w: &mut impl Write, v: u32) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn put_u64(w: &mut impl Write, v: u64) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn get_u32(r: &mut impl Read) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn get_u64(r: &mut impl Read) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

/// Serialize one matrix. The `Indexer` is shared across calls — building it
/// enumerates 383 classes and has no business happening per matrix.
pub fn write_matrix(
    w: &mut impl Write,
    ix: &Indexer,
    m: &QuantizedMatrix,
) -> anyhow::Result<u64> {
    anyhow::ensure!(
        m.codes.len() == m.d_out * m.nblocks(),
        "{}: {} codes for {} blocks",
        m.name,
        m.codes.len(),
        m.d_out * m.nblocks()
    );
    anyhow::ensure!(m.row_scales.len() == m.d_out, "{}: row scale count", m.name);

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
        let idx = ix
            .encode(&c.point)
            .ok_or_else(|| anyhow::anyhow!("{}: point outside the codebook", m.name))?;
        anyhow::ensure!(
            idx < (1u64 << ib),
            "{}: index {idx} does not fit in {ib} bits — the shell cap and the \
             codes disagree",
            m.name
        );
        bw.push(idx, ib);
        bw.push(c.gain as u64, gb);
    }
    let bytes = bw.finish();
    put_u64(w, bytes.len() as u64)?;
    w.write_all(&bytes)?;
    Ok(m.bits())
}

/// Read back what [`write_matrix`] wrote.
pub fn read_matrix(r: &mut impl Read, ix: &Indexer) -> anyhow::Result<QuantizedMatrix> {
    let n = get_u32(r)? as usize;
    let mut name = vec![0u8; n];
    r.read_exact(&mut name)?;
    let name = String::from_utf8(name)?;
    let d_out = get_u32(r)? as usize;
    let d_in = get_u32(r)? as usize;
    let shell_cap = get_u32(r)?;
    let n_cent = get_u32(r)? as usize;
    let seed = get_u64(r)?;
    let has_rot = get_u32(r)? != 0;

    let mut centroids = Vec::with_capacity(n_cent);
    for _ in 0..n_cent {
        centroids.push(f64::from_bits(get_u64(r)?));
    }
    let mut row_scales = Vec::with_capacity(d_out);
    for _ in 0..d_out {
        row_scales.push(f64::from_bits(get_u64(r)?));
    }
    let tail_w = d_out * (d_in % DIM);
    let mut tail = Vec::with_capacity(tail_w);
    for _ in 0..tail_w {
        tail.push(f32::from_bits(get_u32(r)?) as f64);
    }

    let nbytes = get_u64(r)? as usize;
    let mut bytes = vec![0u8; nbytes];
    r.read_exact(&mut bytes)?;

    let ib = llvq_quant::quantizer::index_bits(shell_cap);
    let gb = centroids.len().next_power_of_two().trailing_zeros();
    let nblocks = d_in / DIM;
    let mut br = BitReader::new(&bytes);
    let mut codes = Vec::with_capacity(d_out * nblocks);
    for _ in 0..d_out * nblocks {
        let idx = br.read(ib);
        let gain = br.read(gb) as u32;
        let point = ix
            .decode(idx)
            .ok_or_else(|| anyhow::anyhow!("{name}: index {idx} is out of range"))?;
        codes.push(BlockCode { point, gain });
    }

    Ok(QuantizedMatrix {
        name,
        d_out,
        d_in,
        codes,
        row_scales,
        centroids,
        rotation_seed: has_rot.then_some(seed),
        shell_cap,
        tail,
    })
}

/// Rebuild the `d_out × d_in` weight matrix, in the **natural** basis, exactly
/// as the evaluated model holds it.
///
/// The order of operations mirrors `quantize_model`: decode in the rotated
/// basis, restore the tail, un-rotate, and only then narrow to f32. Doing the
/// narrowing earlier, or un-rotating in f32, changes the last bits.
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
            w[at..at + tail_w]
                .copy_from_slice(&m.tail[i * tail_w..(i + 1) * tail_w]);
        }
    }
    if let Some(seed) = m.rotation_seed {
        llvq_quant::rotation::Rotation::new(m.d_in, seed)
            .unrotate_weight_rows(&mut w, m.d_out);
    }
    w.into_iter().map(|v| v as f32).collect()
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
    pub fn new(mut out: W, n_matrices: u32) -> anyhow::Result<Self> {
        out.write_all(MAGIC)?;
        put_u32(&mut out, n_matrices)?;
        Ok(Self {
            out,
            ix: Indexer::new(),
            matrices: 0,
            payload_bits: 0,
        })
    }

    pub fn push(&mut self, m: &QuantizedMatrix) -> anyhow::Result<()> {
        self.payload_bits += write_matrix(&mut self.out, &self.ix, m)?;
        self.matrices += 1;
        Ok(())
    }

    pub fn finish(mut self) -> anyhow::Result<u64> {
        self.out.flush()?;
        Ok(self.payload_bits)
    }
}

/// Read the file header, returning how many matrices follow.
///
/// Separate from [`read_all`] because a 4B model's codes are 14 GB of lattice
/// points: anything that walks a real artifact has to do it one matrix at a
/// time, and holding them all was never an option.
pub fn read_header(r: &mut impl Read) -> anyhow::Result<u32> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    anyhow::ensure!(&magic == MAGIC, "not an LLVQ artifact");
    Ok(get_u32(r)?)
}

/// Read every matrix back. Only safe for small models — see [`read_header`].
pub fn read_all(r: &mut impl Read) -> anyhow::Result<Vec<QuantizedMatrix>> {
    let n = read_header(r)?;
    let ix = Indexer::new();
    (0..n).map(|_| read_matrix(r, &ix)).collect()
}
