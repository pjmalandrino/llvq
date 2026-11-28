//! The native Inkling 2 Mini checkpoint, read by mmap one tensor at a time.
//!
//! The file is 19.5 GB; nothing here reads it whole. A dense tensor is copied
//! out of the mapping when it is asked for, the embedding one row per token,
//! and the experts one expert at a time.
//!
//! candle 0.9.2 reads the `U8` and `F8_E4M3` safetensors dtypes of the native
//! experts (`candle-core/src/safetensors.rs`, lines 47 and 56). The expert
//! bytes are still decoded here, from the raw view, so that one expert costs
//! one expert's bytes and the arithmetic is the reference's to the bit.
//!
//! NVFP4 layout, per `Inkling2LoopDequantizeExperts` (transformers 046bdb8cc2,
//! `core_model_loading.py` lines 772-816): two E2M1 codes a byte, the even
//! element in the low nibble; one E4M3 scale per 16 values; one factor
//! `amax / outer_scale_denominator` per expert. The value is
//! `(e2m1 * scale) * factor` in f32, in that order (line 813-814). The first
//! product is exact in f32. In `w13` the checkpoint interleaves rows, row `2j`
//! gate `j` and row `2j + 1` up `j`; the reference deinterleaves them with
//! `Interleave(dim=1)` (`core_model_loading.py` lines 232-251,
//! `conversion_mapping.py` line 2666). `W/ops/inkling/nvfp4.py` documents and
//! cross-checks the same decode.
//!
//! MXFP8 layout (layers 20 to 23, no `amax` tensor), same op, lines 806-808:
//! one E4M3 value a byte and one E8M0 scale byte `s` per 32 values; the value
//! is `e4m3 * 2^(s - 127)` in f32, exact. `w13` is interleaved the same way.

use std::path::Path;

use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device, Result, Tensor};

/// The 16 E2M1 values, indexed by the nibble (sign in bit 3).
pub const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// OCP float8 E4M3 ("fn": bias 7, no infinities, `0x7f`/`0xff` NaN) to f32. Exact.
pub fn e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let e = ((b >> 3) & 0xF) as i32;
    let m = (b & 0x7) as f32;
    if b & 0x7F == 0x7F {
        return f32::NAN;
    }
    let mag = if e == 0 {
        m / 8.0 * 2f32.powi(-6)
    } else {
        (1.0 + m / 8.0) * 2f32.powi(e - 7)
    };
    sign * mag
}

fn e4m3_table() -> [f32; 256] {
    let mut t = [0f32; 256];
    for (i, v) in t.iter_mut().enumerate() {
        *v = e4m3_to_f32(i as u8);
    }
    t
}

/// The mapped native file. Names are given without the `model.llm.` prefix.
pub struct NativeCheckpoint {
    st: MmapedSafetensors,
    prefix: String,
}

impl NativeCheckpoint {
    /// Map `path` (the native `model-*.safetensors`).
    ///
    /// The file must not be modified while the mapping is alive: candle maps
    /// it read-only, and a change underneath the mapping is undefined
    /// behaviour (the reason this crate allows `unsafe` at that boundary).
    pub fn open(path: &Path) -> Result<Self> {
        // SAFETY: assumes no process writes or truncates `path` while the map
        // lives: candle maps it read-only, and a change underneath is undefined
        // behaviour (a truncation faults with SIGBUS). Nothing in this process
        // writes it (the doc above); another process could, and nothing here can stop it:
        // the boundary every mmap reader in this crate accepts, not a guarantee.
        let st = unsafe { MmapedSafetensors::new(path)? };
        Ok(Self { st, prefix: "model.llm.".into() })
    }

    fn full(&self, name: &str) -> String {
        format!("{}{name}", self.prefix)
    }

    pub fn has(&self, name: &str) -> bool {
        self.st.get(&self.full(name)).is_ok()
    }

    /// The tensor `name`, copied out of the mapping, in f32 on `dev`. BF16 and
    /// F32 widen exactly.
    pub fn f32(&self, name: &str, dev: &Device) -> Result<Tensor> {
        self.st.load(&self.full(name), dev)?.to_dtype(DType::F32)
    }

    /// The tensor `name` in its file dtype on `dev`.
    pub fn raw(&self, name: &str, dev: &Device) -> Result<Tensor> {
        self.st.load(&self.full(name), dev)
    }

    /// Rows `ids` of the BF16 embedding `embed.weight`, in f32, `[ids.len(), hidden]`.
    /// Reads `ids.len()` rows of the mapping, not the 926 MB table.
    pub fn embed_rows(&self, ids: &[u32], dev: &Device) -> Result<Tensor> {
        let view = self.st.get(&self.full("embed.weight"))?;
        let shape = view.shape().to_vec();
        if shape.len() != 2 || format!("{:?}", view.dtype()) != "BF16" {
            candle_core::bail!("embed.weight: {:?} {shape:?}, expected BF16 rank 2", view.dtype());
        }
        let (vocab, hidden) = (shape[0], shape[1]);
        let data = view.data();
        let mut out = Vec::with_capacity(ids.len() * hidden);
        for &id in ids {
            let id = id as usize;
            if id >= vocab {
                candle_core::bail!("token {id} outside the vocabulary of {vocab}");
            }
            let row = &data[id * hidden * 2..(id + 1) * hidden * 2];
            out.extend(row.chunks_exact(2).map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32()));
        }
        Tensor::from_vec(out, (ids.len(), hidden), dev)
    }

    /// The experts of `layer` (NVFP4 or MXFP8), still in the mapping.
    pub fn experts(&self, layer: usize) -> Result<NativeExperts<'_>> {
        let base = format!("layers.{layer}.mlp.experts.");
        let w13 = ExpertMatrix::new(self, &format!("{base}w13_weight"))?;
        let w2 = ExpertMatrix::new(self, &format!("{base}w2_weight"))?;
        if w13.experts != w2.experts || w13.cols != w2.rows || w13.rows != 2 * w2.cols {
            candle_core::bail!(
                "layer {layer}: w13 {}x{}x{} and w2 {}x{}x{} do not form gate/up/down",
                w13.experts, w13.rows, w13.cols, w2.experts, w2.rows, w2.cols
            );
        }
        Ok(NativeExperts { w13, w2, e4m3: e4m3_table() })
    }
}

/// How one expert stack is stored.
enum Codec {
    /// NVFP4: two E2M1 codes a byte, an E4M3 scale per 16 values, a factor per expert.
    NvFp4 { factor: Vec<f32> },
    /// MXFP8: one E4M3 value a byte, a power-of-two scale `2^(s - 127)` (E8M0 byte `s`) per 32 values.
    Mxfp8,
}

/// One expert stack `[experts, rows, cols]` in the mapping, NVFP4 (layers 0 to 19) or MXFP8 (layers 20 to 23).
pub struct ExpertMatrix<'a> {
    codes: &'a [u8],
    scales: &'a [u8],
    codec: Codec,
    pub experts: usize,
    pub rows: usize,
    pub cols: usize,
}

/// `2^(s - 127)` in f32, exactly: what `torch.exp2(scale.float() - 127)` gives for an integer exponent
/// (`core_model_loading.py` line 808). `s = 0` is the subnormal `2^-127`; `s = 255` is not a scale.
pub fn e8m0_to_f32(s: u8) -> f32 {
    match s {
        0 => f32::from_bits(0x0040_0000),
        255 => f32::NAN,
        _ => f32::from_bits((s as u32) << 23),
    }
}

impl<'a> ExpertMatrix<'a> {
    fn new(ck: &'a NativeCheckpoint, name: &str) -> Result<Self> {
        let w = ck.st.get(&ck.full(name))?;
        let s = ck.st.get(&ck.full(&format!("{name}_scale")))?;
        let ws = w.shape().to_vec();
        let ss = s.shape().to_vec();
        let (wd, sd) = (format!("{:?}", w.dtype()), format!("{:?}", s.dtype()));
        // `amax` present means NVFP4 (`Inkling2LoopDequantizeExperts`, lines 797-799)
        let nvfp4 = ck.has(&format!("{name}_amax"));
        if nvfp4 {
            if wd != "U8" || sd != "F8_E4M3" {
                candle_core::bail!("{name}: dtypes {wd}/{sd}, expected U8/F8_E4M3 (NVFP4)");
            }
            let (experts, rows, cols) = (ws[0], ws[1], ws[2] * 2);
            if ss != [experts, rows, cols / 16] {
                candle_core::bail!("{name}_scale: shape {ss:?}, expected [{experts}, {rows}, {}]", cols / 16);
            }
            let amax =
                ck.st.load(&ck.full(&format!("{name}_amax")), &Device::Cpu)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            let den = ck
                .st
                .load(&ck.full(&format!("{name}_outer_scale_denominator")), &Device::Cpu)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            if amax.len() != experts || den.len() != 1 {
                candle_core::bail!("{name}: {} amax and {} denominators for {experts} experts", amax.len(), den.len());
            }
            // `amax.float() / denominator.float()`, line 804
            let factor = amax.iter().map(|a| a / den[0]).collect();
            Ok(Self { codes: w.data(), scales: s.data(), codec: Codec::NvFp4 { factor }, experts, rows, cols })
        } else {
            if wd != "F8_E4M3" || sd != "U8" {
                candle_core::bail!("{name}: dtypes {wd}/{sd}, expected F8_E4M3/U8 (MXFP8)");
            }
            let (experts, rows, cols) = (ws[0], ws[1], ws[2]);
            if ss != [experts, rows, cols / 32] {
                candle_core::bail!("{name}_scale: shape {ss:?}, expected [{experts}, {rows}, {}]", cols / 32);
            }
            Ok(Self { codes: w.data(), scales: s.data(), codec: Codec::Mxfp8, experts, rows, cols })
        }
    }

    pub fn is_mxfp8(&self) -> bool {
        matches!(self.codec, Codec::Mxfp8)
    }

    /// Row `r` of expert `e`, decoded into `out` (`cols` values).
    fn row(&self, e: usize, r: usize, table: &[f32; 256], out: &mut [f32]) {
        match &self.codec {
            Codec::NvFp4 { factor } => {
                let nb = self.cols / 2;
                let codes = &self.codes[(e * self.rows + r) * nb..][..nb];
                let scales = &self.scales[(e * self.rows + r) * (self.cols / 16)..][..self.cols / 16];
                let f = factor[e];
                for (g, &sb) in scales.iter().enumerate() {
                    let s = table[sb as usize];
                    for (k, &byte) in codes[g * 8..g * 8 + 8].iter().enumerate() {
                        out[g * 16 + 2 * k] = (E2M1[(byte & 0xF) as usize] * s) * f;
                        out[g * 16 + 2 * k + 1] = (E2M1[(byte >> 4) as usize] * s) * f;
                    }
                }
            }
            Codec::Mxfp8 => {
                // `weight.float() * exp2(scale - 127)`, lines 807-808
                let codes = &self.codes[(e * self.rows + r) * self.cols..][..self.cols];
                let scales = &self.scales[(e * self.rows + r) * (self.cols / 32)..][..self.cols / 32];
                for (g, &sb) in scales.iter().enumerate() {
                    let s = e8m0_to_f32(sb);
                    for (k, &byte) in codes[g * 32..g * 32 + 32].iter().enumerate() {
                        out[g * 32 + k] = table[byte as usize] * s;
                    }
                }
            }
        }
    }
}

/// The experts of one layer: `w13` `[E, 2I, H]` interleaved, `w2` `[E, H, I]`.
pub struct NativeExperts<'a> {
    pub w13: ExpertMatrix<'a>,
    pub w2: ExpertMatrix<'a>,
    e4m3: [f32; 256],
}

impl NativeExperts<'_> {
    pub fn count(&self) -> usize {
        self.w13.experts
    }

    /// Expert `e` in f32: `gate_up` `[2I, H]` with the gate rows first then
    /// the up rows (the reference's `gate_up_proj[e]`), and `down` `[H, I]`.
    /// Rows are decoded in parallel; each value is computed by the same
    /// expression whatever the split, so the result does not depend on it.
    pub fn dequant(&self, e: usize) -> (Vec<f32>, Vec<f32>) {
        use rayon::prelude::*;
        let (r13, c13) = (self.w13.rows, self.w13.cols);
        let inter = r13 / 2;
        let mut gu = vec![0f32; r13 * c13];
        // checkpoint row 2j is gate j, row 2j+1 is up j (`Interleave(dim=1)`)
        let (gate, up) = gu.split_at_mut(inter * c13);
        gate.par_chunks_mut(c13).enumerate().for_each(|(j, out)| self.w13.row(e, 2 * j, &self.e4m3, out));
        up.par_chunks_mut(c13).enumerate().for_each(|(j, out)| self.w13.row(e, 2 * j + 1, &self.e4m3, out));
        let (r2, c2) = (self.w2.rows, self.w2.cols);
        let mut down = vec![0f32; r2 * c2];
        down.par_chunks_mut(c2).enumerate().for_each(|(r, out)| self.w2.row(e, r, &self.e4m3, out));
        (gu, down)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The E4M3 decode against candle's own float8 conversion, on all 256
    /// bytes (NaN on both sides for 0x7f and 0xff).
    #[test]
    fn e4m3_matches_candle_float8() -> Result<()> {
        let bytes: Vec<u8> = (0..=255u8).collect();
        let t = Tensor::from_raw_buffer(&bytes, DType::F8E4M3, &[256], &Device::Cpu)?;
        let c = t.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        for (i, &v) in c.iter().enumerate() {
            let m = e4m3_to_f32(i as u8);
            assert!(
                (v.is_nan() && m.is_nan()) || v.to_bits() == m.to_bits(),
                "byte {i:#04x}: candle {v} ours {m}"
            );
        }
        assert_eq!(e4m3_to_f32(0x38), 1.0);
        assert_eq!(e4m3_to_f32(0x7E), 448.0);
        assert_eq!(e4m3_to_f32(0x01), 2f32.powi(-9));
        Ok(())
    }

    /// The E8M0 scale is `2^(s - 127)` exactly, the subnormal `s = 0` included.
    #[test]
    fn e8m0_is_an_exact_power_of_two() {
        for s in 0..=254u8 {
            assert_eq!(e8m0_to_f32(s), 2f64.powi(s as i32 - 127) as f32, "s = {s}");
        }
        assert_eq!(e8m0_to_f32(127), 1.0);
        assert!(e8m0_to_f32(255).is_nan());
    }
}
