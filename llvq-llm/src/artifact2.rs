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
//!
//! # Resuming a killed run
//!
//! A 32B quantization is ~11.4 h of rented card. Until now nothing could
//! restart one: an incident in the tenth hour lost the ten hours and the bill
//! both, which is the reason the 32B run kept not being launched.
//!
//! ## Why the shard *is* the state
//!
//! Calibration is sequential — block `t` is quantized against activations that
//! have flowed through blocks `0..t` **in their quantized form** — so resuming
//! means reconstituting that state. It looks like it needs a checkpoint of the
//! whole loop. It does not, and the reason is the property `verify_artifact`
//! has asserted on every run that ever wrote a file: [`decode_matrix`] returns
//! the evaluated weights **bit for bit** (6,945,767,424 of them on the 8B run,
//! 1,950,351,360 on the 32B de-risking pass). The already-written matrices are
//! therefore a lossless snapshot of every weight the loop has changed, and the
//! only thing left to rebuild is the hidden states — by re-running the forward
//! pass over the blocks that are already done, which is exactly what
//! [`crate::calib::RunConfig::start`] makes the loop do.
//!
//! Nothing else survives a block: the Hessians, the factors and the gain
//! centroids are all rebuilt per block from the weights and the activations.
//!
//! ## Why the fusion happens *here* rather than after the fact
//!
//! A resumed segment copies its input shard into its own output before
//! quantizing anything, so **the last segment's file is the whole artifact**.
//! There is no separate merge step to get wrong, and the invariant "a `.llvq`
//! covers blocks `0..k` contiguously" holds at every instant.
//!
//! Concatenating two files would not have worked anyway:
//! [`ArtifactWriter::finish`] closes a stream with two `u32` zeros (the empty
//! raw-tensor and blob sections), and a header carries the matrix count, so
//! naive concatenation yields a file whose header under-counts and whose
//! middle contains eight bytes of garbage. The copy goes through
//! [`llvq_artifact::read_matrix_raw`] / [`ArtifactWriter::push_raw`], which is
//! byte-identical by construction (`raw_passthrough_is_byte_identical`) and
//! validates every record on the way past — and, unlike a byte copy, it costs
//! no lattice re-encode of 150 M blocks.
//!
//! ## What is checked before a single block is re-quantized
//!
//! An artifact assembled from segments of *different* configurations would be
//! valid in form and wrong in substance — the failure mode this repo fears
//! most. [`resume_from_shard`] therefore refuses unless, matrix by matrix:
//! the record order is the one [`crate::calib::block_matrix_plan`] emits, the
//! dimensions are the model's, the shell cap and centroid count are the
//! resolved codebook's, and the stored rotation seed is exactly
//! [`crate::calib::effective_rotation_seed`] of the resolved base. The
//! codebook fingerprint is checked one level down, by
//! [`llvq_artifact::read_header`].
//!
//! The rest of the configuration — calibration corpus, seed, window count and
//! length, damping, mode, dtype, model id — leaves no trace in a shard and
//! cannot be checked from one. That is what [`RunState`] is for.

use std::io::Write;

pub use llvq_artifact::{
    decode_matrix, read_all, read_header, read_matrix, read_matrix_raw, split_name, write_matrix,
    ArtifactWriter, Blob, Header, QuantizedMatrix, RawMatrix, RawTensor,
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
    let head = read_header(&mut r)?;
    let ix = llvq_search::index::Indexer::new();
    let dtype = model.dtype();
    let mut weights = 0usize;
    for _ in 0..head.matrices {
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
    Ok((head.matrices as usize, weights))
}

// ---------------------------------------------------------------------------
// Resuming a killed run. See the module header for why this is possible at all.
// ---------------------------------------------------------------------------

/// How far a shard actually goes.
///
/// ## Why the header is not the answer
///
/// [`ArtifactWriter::new`] writes the matrix count **up front**, before a
/// single record exists — it is a promise, not a measurement. A run killed at
/// hour ten leaves a header claiming 252 matrices over a file holding 160 and
/// a half. Trusting the header there would make the resume read past the end
/// of the data and start the model's second half from garbage.
///
/// So this walks the record *headers*, seeking over the payloads: a record's
/// size is fully determined by its own fields (name length, dimensions,
/// centroid count, tail width, then a length-prefixed code stream), so the
/// scan costs one seek per matrix — 252 of them on a 4B — and never reads the
/// 11 GB. It stops at the first record that is not entirely present.
///
/// The result is truncated **down to a whole block**: a shard that stops after
/// four of a block's seven projections covers that block not at all. Appending
/// behind a torn block would produce an artifact missing one projection in the
/// middle, which the loader leaves at full precision — silently, in a model
/// advertised as 2-bit.
///
/// Returns `(complete_records, whole_blocks)`.
pub fn shard_extent(path: impl AsRef<std::path::Path>) -> anyhow::Result<(usize, usize)> {
    use std::io::{Read, Seek, SeekFrom};

    let path = path.as_ref();
    let f = std::fs::File::open(path).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    let end = f.metadata()?.len();
    let mut r = std::io::BufReader::new(f);
    // `read_header` is also where the codebook fingerprint is checked, so a
    // shard written by a different build of the index map is refused before
    // anything else is decided about it.
    let head = read_header(&mut r)?;

    // Every read below is bounds-checked against the file length rather than
    // trusted: a torn file's last bytes can decode to an enormous `name_len`,
    // and `vec![0u8; that]` would be an allocation failure reported as a bug
    // in the resume rather than as a truncated shard.
    let u32_at = |r: &mut std::io::BufReader<std::fs::File>| -> Option<u32> {
        let mut b = [0u8; 4];
        r.read_exact(&mut b).ok()?;
        Some(u32::from_le_bytes(b))
    };
    let mut complete = 0usize;
    loop {
        let at = r.stream_position()?;
        // The trailer is two `u32` zeros; anything shorter than one record's
        // fixed part is the end of the data one way or another.
        if end.saturating_sub(at) < 32 {
            break;
        }
        let Some(name_len) = u32_at(&mut r) else { break };
        // Every arithmetic step is checked, and that is not defensive
        // decoration: the bytes being added up here come from a **torn file**,
        // so `nbytes` can legitimately be `u64::MAX`. Wrapping would make
        // `next` land back inside the file, the scan would accept a record
        // that does not exist, and the resume would start the model's second
        // half from the middle of the first one. In release builds Rust wraps
        // silently, so the check has to be written.
        let Some(rest) = (|| {
            let after_name = at.checked_add(4)?.checked_add(u64::from(name_len))?;
            if after_name.checked_add(28)? > end {
                return None;
            }
            r.seek(SeekFrom::Start(after_name)).ok()?;
            let d_out = u64::from(u32_at(&mut r)?);
            let d_in = u64::from(u32_at(&mut r)?);
            let _shell = u32_at(&mut r)?;
            let n_cent = u64::from(u32_at(&mut r)?);
            // The fixed part behind the name: four `u32` dimensions, then the
            // rotation seed (`u64`) and its flag (`u32`).
            let payload = after_name.checked_add(16)?.checked_add(12)?;
            let skip = n_cent
                .checked_mul(8)?
                .checked_add(d_out.checked_mul(8)?)?
                .checked_add(d_out.checked_mul(d_in % llvq_core::DIM as u64)?.checked_mul(4)?)?;
            let code_len_at = payload.checked_add(skip)?;
            if code_len_at.checked_add(8)? > end {
                return None;
            }
            r.seek(SeekFrom::Start(code_len_at)).ok()?;
            let mut b = [0u8; 8];
            r.read_exact(&mut b).ok()?;
            let nbytes = u64::from_le_bytes(b);
            let next = code_len_at.checked_add(8)?.checked_add(nbytes)?;
            (next <= end).then_some(next)
        })() else {
            break;
        };
        r.seek(SeekFrom::Start(rest))?;
        complete += 1;
    }

    let per = crate::calib::matrices_per_block();
    let blocks = complete / per;
    anyhow::ensure!(
        blocks > 0,
        "{}: {complete} complete matrices, less than one block of {per}, \
         nothing to resume",
        path.display()
    );
    if complete != head.matrices as usize {
        eprintln!(
            "  WARNING: {}: the header announces {} matrices, the file carries {complete}. \
             The shard was interrupted. The resume keeps only the {blocks} whole blocks \
             ({} matrices) and drops the tail.",
            path.display(),
            head.matrices,
            blocks * per
        );
    }
    Ok((blocks * per, blocks))
}

/// The parts of the resolved codebook a shard *does* record, so a resume can
/// check them matrix by matrix.
pub struct ShardExpect {
    /// Shell cap of the direction code — what sets the index width.
    pub shell_cap: u32,
    /// Number of fitted gain levels, i.e. `1 << gain_bits`.
    pub centroids: usize,
    /// **Base** rotation seed of the run, not the per-matrix one. The check
    /// derives the per-matrix value through
    /// [`crate::calib::effective_rotation_seed`], which is the only place that
    /// derivation is written.
    pub rotation_seed: Option<u64>,
}

/// What a shard turned out to hold, for the report line that has to describe
/// the *whole* file rather than the segment that finished it.
#[derive(Debug, Default, Clone)]
pub struct ShardScan {
    pub matrices: usize,
    /// Contiguous blocks `0..blocks`, all seven projections each.
    pub blocks: usize,
    pub weights: u64,
    /// Weights that stayed at full precision in a tail.
    pub tail_weights: u64,
    /// Output rows, each carrying one scale.
    pub rows: u64,
    pub seconds: f64,
}

/// Copy `shard` into `out` and load it into `model`, in a single pass.
///
/// This is the resume, and it does three things at once because they all want
/// the same bytes and a 32B shard is ~11 GB:
///
/// 1. **validate** every record against the resolved configuration (see the
///    module header for the list, and for what a shard cannot check);
/// 2. **copy** it into the output, so the segment that finishes the model
///    produces the whole artifact and no merge step exists to get wrong;
/// 3. **load** it into `model`, which is what reconstitutes the sequential
///    state — the loop then re-runs the forward passes over these blocks and
///    arrives at block `blocks` with the hidden states a single run would have
///    carried there.
///
/// The copy is a raw passthrough: the indices are *also* decoded, for (3), but
/// they are never re-encoded. `raw_passthrough_is_byte_identical` pins the
/// byte-for-byte property that makes this a copy rather than a rewrite.
pub fn resume_from_shard<W: Write>(
    model: &mut crate::model::Qwen3,
    shard: impl AsRef<std::path::Path>,
    out: &mut ArtifactWriter<W>,
    expect: &ShardExpect,
    device: &candle_core::Device,
) -> anyhow::Result<ShardScan> {
    let t0 = std::time::Instant::now();
    let path = shard.as_ref();
    let (n_matrices, blocks) = shard_extent(path)?;
    anyhow::ensure!(
        blocks <= model.blocks.len(),
        "{}: the shard covers {blocks} blocks, the model has only {}. \
         This is not the same model.",
        path.display(),
        model.blocks.len()
    );

    let f = std::fs::File::open(path)?;
    let mut r = std::io::BufReader::with_capacity(1 << 20, f);
    let _ = read_header(&mut r)?;
    let ix = llvq_search::index::Indexer::new();
    let dtype = model.dtype();
    let plan = crate::calib::block_matrix_plan();

    let mut scan = ShardScan {
        matrices: n_matrices,
        blocks,
        ..Default::default()
    };
    for t in 0..blocks {
        for (act, proj) in &plan {
            let raw = read_matrix_raw(&mut r)?;
            // ---- the record has to be the one the loop would have written ----
            let want = crate::artifact::key(t, proj);
            anyhow::ensure!(
                raw.name == want,
                "{}: expected matrix {want:?}, read {:?}. The record order is not the \
                 one the loop writes, so this file is the prefix of no run.",
                path.display(),
                raw.name
            );
            let dims = model.blocks[t].linear(proj).weight().dims2()?;
            anyhow::ensure!(
                (raw.d_out, raw.d_in) == dims,
                "{want}: the shard carries {:?}, the model expects {:?}, not the \
                 same model",
                (raw.d_out, raw.d_in),
                dims
            );
            anyhow::ensure!(
                raw.shell_cap == expect.shell_cap,
                "{want}: the shard was quantized with shell {} and this run uses \
                 {}. Two codebooks in one file.",
                raw.shell_cap,
                expect.shell_cap
            );
            anyhow::ensure!(
                raw.centroids.len() == expect.centroids,
                "{want}: the shard carries {} gain levels and this run sets {}, \
                 two rates in one file.",
                raw.centroids.len(),
                expect.centroids
            );
            // Derived, never read off the file: storing the base seed instead
            // of the per-matrix one is a mistake this project has already made
            // once, and it decodes to plausible garbage rather than failing.
            let want_seed = expect
                .rotation_seed
                .map(|s| crate::calib::effective_rotation_seed(s, t, *act));
            anyhow::ensure!(
                raw.rotation_seed == want_seed,
                "{want}: rotation seed {:?} in the shard, {want_seed:?} for this run. \
                 The two halves of the model would be quantized in different bases.",
                raw.rotation_seed
            );

            scan.weights += (raw.d_out * raw.d_in) as u64;
            scan.tail_weights += (raw.d_out * (raw.d_in % llvq_core::DIM)) as u64;
            scan.rows += raw.d_out as u64;

            // ---- (2) copy the record on, untouched ----
            out.push_raw(&raw)?;

            // ---- (3) load it into the model ----
            //
            // The indices are decoded here and never re-encoded: the copy
            // above already carries them. Building the `QuantizedMatrix` by
            // hand rather than calling `read_matrix` is what lets one read of
            // the file serve all three jobs.
            let m = to_quantized(raw, &ix)?;
            let w = decode_matrix(&m);
            let tensor =
                candle_core::Tensor::from_vec(w, (m.d_out, m.d_in), device)?.to_dtype(dtype)?;
            *model.blocks[t].linear_mut(proj) = candle_nn::Linear::new(tensor, None);
        }
    }
    scan.seconds = t0.elapsed().as_secs_f64();
    Ok(scan)
}

/// The half of a run's configuration that leaves **no trace in a shard**.
///
/// A shard records the codebook (shell cap, centroid count), the dimensions,
/// the rotation seeds and — from `LVQ4` — the index map's fingerprint. It
/// records nothing about the calibration corpus, the window count and length,
/// the seed, the damping, the refinement mode or the dtype. Resume a run under
/// a different one of those and every check in [`resume_from_shard`] passes:
/// the file is well formed, every index is in range, the model loads and runs.
/// It is simply two different quantizations of two halves of a model, and the
/// only place the difference would ever show up is a perplexity nobody can
/// explain six weeks later. That is the failure mode this repo has paid for
/// three times.
///
/// So a run that writes an artifact writes this beside it, and a run that
/// resumes reads it back and refuses on any disagreement. Plain `key = value`
/// text, because an operator staring at a bucket at 3 a.m. should be able to
/// see why the resume was refused without a tool.
///
/// The fields are the caller's to choose — `bin/smoke` owns which parts of its
/// configuration matter — with one exception, [`RunState::BLOCKS`], which this
/// module reads: it is the sidecar's own claim about how far the shard goes,
/// cross-checked against [`shard_extent`]'s reading of the file. Two
/// independent witnesses of the same fact, which is what catches the operator
/// error a single source of truth cannot: a `.state` left over from a
/// *different* shard.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RunState {
    fields: Vec<(String, String)>,
}

impl RunState {
    /// Key holding the number of blocks the shard covers.
    pub const BLOCKS: &'static str = "blocks_done";

    pub fn new() -> Self {
        Self::default()
    }

    /// Record one field. Order is preserved, so the file reads like the
    /// journal banner it mirrors.
    pub fn set(&mut self, key: &str, value: impl std::fmt::Display) -> &mut Self {
        self.fields.push((key.to_string(), value.to_string()));
        self
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.fields
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    /// Where the sidecar of `artifact` lives.
    pub fn path_for(artifact: &str) -> String {
        format!("{artifact}.state")
    }

    pub fn render(&self) -> String {
        let mut s = String::from(
            "# llvq run state, written beside the artifact, read back by a resume.\n\
             # Any divergence from the resolved configuration makes the resume refuse.\n",
        );
        for (k, v) in &self.fields {
            s.push_str(&format!("{k} = {v}\n"));
        }
        s
    }

    pub fn write(&self, artifact: &str) -> anyhow::Result<String> {
        let p = Self::path_for(artifact);
        std::fs::write(&p, self.render())?;
        Ok(p)
    }

    pub fn read(artifact: &str) -> anyhow::Result<Self> {
        let p = Self::path_for(artifact);
        let text = std::fs::read_to_string(&p).map_err(|e| {
            anyhow::anyhow!(
                "{p}: {e}\nA resume requires the state written beside the shard. Without \
                 it, nothing says on which calibration, which damping and which dtype the \
                 shard was produced, and an artifact whose two halves do not share them \
                 would look valid and be wrong."
            )
        })?;
        Ok(Self::parse(&text))
    }

    pub fn parse(text: &str) -> Self {
        let mut s = Self::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some((k, v)) = line.split_once('=') {
                s.set(k.trim(), v.trim());
            }
        }
        s
    }

    /// Every field on which `self` (the shard's state) and `now` (the resolved
    /// configuration) disagree, rendered for a refusal message.
    ///
    /// A key present on one side and absent on the other counts: a run that
    /// stopped recording a field would otherwise get its resume waved through
    /// on the strength of the fields it kept.
    ///
    /// [`RunState::BLOCKS`] is skipped, and is the only key that is: it states
    /// how far the *shard* goes, which is precisely the thing a resume expects
    /// to differ. It is cross-checked against [`shard_extent`] instead — see
    /// the type doc.
    pub fn diff(&self, now: &Self) -> Vec<String> {
        let mut out = Vec::new();
        for (k, v) in self.fields.iter().filter(|(k, _)| k != Self::BLOCKS) {
            match now.get(k) {
                Some(w) if w == v => {}
                Some(w) => out.push(format!("{k}: shard {v:?}, this run {w:?}")),
                None => out.push(format!("{k}: shard {v:?}, this run does not record it")),
            }
        }
        for (k, w) in now.fields.iter().filter(|(k, _)| k != Self::BLOCKS) {
            if self.get(k).is_none() {
                out.push(format!("{k}: absent from the shard, this run {w:?}"));
            }
        }
        out
    }
}

/// Decode a raw record's indices into lattice points.
///
/// [`read_matrix`] is this plus the read; splitting it out is what lets the
/// resume read a record once and both copy and decode it.
fn to_quantized(
    raw: RawMatrix,
    ix: &llvq_search::index::Indexer,
) -> anyhow::Result<QuantizedMatrix> {
    let mut codes = Vec::with_capacity(raw.indices.len());
    for (&idx, &gain) in raw.indices.iter().zip(&raw.gains) {
        let point = ix.decode(idx).ok_or_else(|| {
            anyhow::anyhow!("{}: index {idx} outside the codebook", raw.name)
        })?;
        codes.push(llvq_quant::quantizer::BlockCode { point, gain });
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
