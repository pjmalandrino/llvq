//! The bridge between the artifact format and candle.
//!
//! The format itself — writing, reading, decoding to `Vec<f32>` — lives in
//! [`llvq_artifact`], which has **no dependencies at all**. Someone who wants
//! to inspect a `.llvq`, port the reader to another language, or audit the
//! decoder before trusting a model does not have to compile a tensor runtime
//! to do it. That is the sovereignty argument applied to the format itself.
//!
//! What is left here is the only part that genuinely needs candle: turning
//! decoded weights into tensors and dropping them into a live `Qwen3`.

pub use llvq_artifact::{
    decode_matrix, read_all, read_header, read_matrix, split_name, write_matrix, ArtifactWriter,
    QuantizedMatrix,
};

/// Decode an artifact straight into a model, one matrix at a time.
///
/// Returns `(matrices, weights)` — both, because a partial artifact must never
/// be mistaken for a whole one when a compression ratio is quoted.
///
/// ## What the file does not carry
///
/// The **linear projections only**. Embeddings, RMSNorm weights and the config
/// still come from the checkpoint, so a `.llvq` is not yet a self-contained
/// model — it is the compressed 90 % of one. On Qwen3-4B the tied embedding is
/// 389 M weights at f16, 9.7 % of the model and the reason the end-to-end
/// ratio is ~4.6× rather than the ~7.6× the linear layers alone would suggest.
pub fn load(
    model: &mut crate::model::Qwen3,
    path: impl AsRef<std::path::Path>,
    device: &candle_core::Device,
) -> anyhow::Result<(usize, usize)> {
    let f = std::fs::File::open(path.as_ref())?;
    let mut r = std::io::BufReader::with_capacity(1 << 20, f);
    let n = read_header(&mut r)?;
    let ix = llvq_search::index::Indexer::new();
    let dtype = model.dtype();
    let mut weights = 0usize;
    for _ in 0..n {
        let m = read_matrix(&mut r, &ix)?;
        weights += m.d_out * m.d_in;
        let w = decode_matrix(&m);
        let (b, proj) = split_name(&m.name)?;
        let t = candle_core::Tensor::from_vec(w, (m.d_out, m.d_in), device)?.to_dtype(dtype)?;
        let lin = model.blocks[b].linear_mut(&proj);
        anyhow::ensure!(
            lin.weight().dims() == [m.d_out, m.d_in],
            "{}: artifact holds {:?}, model expects {:?}",
            m.name,
            [m.d_out, m.d_in],
            lin.weight().dims()
        );
        *lin = candle_nn::Linear::new(t, None);
    }
    Ok((n as usize, weights))
}
