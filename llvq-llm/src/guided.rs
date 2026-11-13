//! Gradient-weighted input Hessians for `gate_proj` and `up_proj`: V1 of
//! `docs/mesures/recherche-qualite-r2-chiffrage-encodeur-2026-11-13.txt`.
//!
//! ## What changes
//!
//! Today every matrix of an activation is encoded against one Hessian, `H = Σ_t x_t x_tᵀ / N`. Under
//! [`ENV_DIR`], `mlp.gate_proj` and `mlp.up_proj` are each encoded against `G` Hessians instead, one
//! per group of consecutive output rows:
//!
//! ```text
//! H_g = Σ_t w̃_tg x_t x_tᵀ / N,   w̃_tg = w_tg / mean_t(w_tg),   w_tg = Σ_{rows r of group g} (∂ℓ/∂z_tr)²
//! ```
//!
//! `x_t` is the encoder's own input, exactly as today: its partly encoded prefix, its rotation, its
//! damping, its shrink. Only the per-token weights `w_tg` come from outside, written by
//! `ops/guided_weights.py` from one backward pass per calibration window on the FP16 checkpoint
//! (next-token cross-entropy on the window's own tokens, GuidedQuant's objective,
//! arXiv 2505.07004 Eq. 4 and §3.2). Every other matrix, and every other knob, is untouched.
//!
//! ## Why a group and not a row
//!
//! The encoder treats every output row independently (`llvq_quant::gptq`, "Rows therefore never
//! interact"), and a 24-weight block lies along the input axis of one row. A positive weight on a
//! row scales that row's objective and moves none of its codes. A weight per token does move them:
//! it changes `H`, so the factor, so the error feedback onto later columns. Rows of one group share
//! one `H_g`, and since rows never interact, encoding the matrix group by group through the
//! existing row-parallel loop is exact plumbing, not an approximation.
//!
//! ## Why the scale of `H_g` does not matter, and why it is normalised anyway
//!
//! The GPTQ path is invariant to `H → cH` for any `c > 0`: the damping is relative,
//! `λ·mean(diag H)` (`llvq_quant::linalg::damped`), the feedback applies `U_QQ⁻¹ U_QR`, a ratio of
//! the factor's entries, and the block search does not read `H`. So `G` groups with weights summing
//! to different totals would encode the same codes as with each group rescaled. The mean-1
//! normalisation keeps every `H_g` on the scale of today's `H`, so that the f32 accumulation sees
//! numbers of today's size whatever the absolute size of the squared gradients, which depends on
//! the loss, the model and the token.
//!
//! ## The file
//!
//! A directory with `manifest.json` and one little-endian f32 file per (layer, type), shape
//! `[tokens, 16]`, row-major: column `j` holds the squared gradients summed over output rows
//! `[j·d/16, (j+1)·d/16)`. `G ∈ {1, 2, 4, 8, 16}` is read by summing neighbouring columns, so one
//! card job serves every `G`. The manifest names the token fingerprint the weights were computed on,
//! and the encoder refuses it unless it is the fingerprint of its own calibration windows.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// The directory of the weights. Unset: the published path, byte for byte.
pub const ENV_DIR: &str = "LLVQ_GUIDED_WEIGHTS";
/// `G`, the number of row groups. Only read beside [`ENV_DIR`].
pub const ENV_GROUPS: &str = "LLVQ_GUIDED_GROUPS";
/// `G` when [`ENV_GROUPS`] is unset: GuidedQuant's choice for 7B and 13B models (§4).
pub const DEFAULT_GROUPS: usize = 4;
/// Columns of every weight file: the finest grouping one card job writes.
pub const G_MAX: usize = 16;
/// The manifest's `format` field.
pub const FORMAT: &str = "llvq-guided-weights-v1";
/// The token file's sidecar `format` field.
pub const TOKENS_FORMAT: &str = "llvq-calib-tokens-v1";
/// The matrices this applies to, as `Act::consumers` names them.
pub const TYPES: [&str; 2] = ["mlp.gate_proj", "mlp.up_proj"];

/// `G` from its variable: unset or empty is [`DEFAULT_GROUPS`]; anything but a divisor of
/// [`G_MAX`] is refused by name.
pub fn parse_groups(v: Option<&str>) -> Result<usize, String> {
    match v.map(str::trim) {
        None | Some("") => Ok(DEFAULT_GROUPS),
        Some(s) => match s.parse::<usize>() {
            Ok(g) if g > 0 && G_MAX % g == 0 => Ok(g),
            _ => Err(format!(
                "{ENV_GROUPS}={s:?}: accepted values 1, 2, 4, 8 and 16 (default {DEFAULT_GROUPS}); the \
                 weight files hold {G_MAX} columns and a group is a sum of neighbouring columns"
            )),
        },
    }
}

/// The two variables, read through `lookup`. `Ok(None)` is the published path.
///
/// [`ENV_GROUPS`] without [`ENV_DIR`] is refused: it would read as a guided run in a log and be the
/// default one. An empty [`ENV_DIR`] is refused too, rather than read as unset.
pub fn from_env(lookup: impl Fn(&str) -> Option<String>) -> Result<Option<(PathBuf, usize)>, String> {
    let dir = lookup(ENV_DIR);
    let groups = lookup(ENV_GROUPS);
    match dir {
        None => match groups {
            None => Ok(None),
            Some(g) => Err(format!(
                "{ENV_GROUPS}={g:?} without {ENV_DIR}: the group count means nothing without the weights; \
                 unset it, or name the weights directory"
            )),
        },
        Some(d) if d.trim().is_empty() => Err(format!(
            "{ENV_DIR} is set and empty: name the weights directory, or unset it for the published path"
        )),
        Some(d) => Ok(Some((PathBuf::from(d), parse_groups(groups.as_deref())?))),
    }
}

/// The file of one (layer, matrix): `w-003-gate_proj.f32`.
pub fn file_name(layer: usize, name: &str) -> String {
    let short = name.rsplit('.').next().unwrap_or(name);
    format!("w-{layer:03}-{short}.f32")
}

/// The rows of group `g` of `G` in a matrix of `d_out` rows: consecutive equal chunks.
///
/// `d_out` must be a multiple of [`G_MAX`], so that every group of `G` is a union of whole columns
/// of the file. The 4B's 9,728 rows are 16 × 608.
pub fn row_ranges(d_out: usize, groups: usize) -> Result<Vec<std::ops::Range<usize>>, String> {
    if groups == 0 || G_MAX % groups != 0 {
        return Err(format!("{groups} row groups: a divisor of {G_MAX} is needed"));
    }
    if d_out % G_MAX != 0 {
        return Err(format!(
            "a matrix of {d_out} output rows: guided weights need a multiple of {G_MAX}, the columns of \
             the weight files"
        ));
    }
    let per = d_out / groups;
    Ok((0..groups).map(|g| g * per..(g + 1) * per).collect())
}

/// One weight file, read raw: `tokens × G_MAX` f32, little endian. Refuses a size that is not that,
/// and any value that is not finite or is negative (a weight is a sum of squares).
pub fn read_raw(path: &Path, tokens: usize) -> anyhow::Result<Vec<f32>> {
    let bytes = std::fs::read(path).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    let want = tokens * G_MAX * 4;
    anyhow::ensure!(
        bytes.len() == want,
        "{}: {} bytes, expected {want} ({tokens} tokens × {G_MAX} columns × 4)",
        path.display(),
        bytes.len()
    );
    let v: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    if let Some((i, x)) = v.iter().enumerate().find(|(_, x)| !x.is_finite() || **x < 0.0) {
        anyhow::bail!(
            "{}: value {x} at token {}, column {}: a weight is a finite sum of squares",
            path.display(),
            i / G_MAX,
            i % G_MAX
        );
    }
    Ok(v)
}

/// The per-token weights of each of `groups` row groups, each scaled to mean 1 over the tokens.
///
/// Group `g` sums columns `[g·G_MAX/G, (g+1)·G_MAX/G)` of `raw` in f64; the mean is taken in f64 and
/// the quotient narrowed to f32 once. A group whose weights are all zero is refused: its `H_g` would
/// be zero and the factor the damping alone.
pub fn group_weights(raw: &[f32], tokens: usize, groups: usize) -> Result<Vec<Vec<f32>>, String> {
    if groups == 0 || G_MAX % groups != 0 {
        return Err(format!("{groups} row groups: a divisor of {G_MAX} is needed"));
    }
    if raw.len() != tokens * G_MAX {
        return Err(format!("{} weights for {tokens} tokens × {G_MAX} columns", raw.len()));
    }
    let span = G_MAX / groups;
    let mut out = Vec::with_capacity(groups);
    for g in 0..groups {
        let w: Vec<f64> = (0..tokens)
            .map(|t| raw[t * G_MAX + g * span..t * G_MAX + (g + 1) * span].iter().map(|&x| x as f64).sum())
            .collect();
        let mean = w.iter().sum::<f64>() / tokens as f64;
        if !(mean > 0.0 && mean.is_finite()) {
            return Err(format!("row group {g} of {groups}: every weight is zero (mean {mean})"));
        }
        out.push(w.into_iter().map(|x| (x / mean) as f32).collect());
    }
    Ok(out)
}

/// sha256 of `dir/manifest.json`: what the run state records, before the tokens are known.
pub fn manifest_sha256(dir: &Path) -> anyhow::Result<String> {
    let p = dir.join("manifest.json");
    crate::digest::sha256_file(&p).map_err(|e| anyhow::anyhow!("{ENV_DIR}: {}: {e}", p.display()))
}

/// A weights directory, opened and checked against the run that reads it.
#[derive(Clone, Debug)]
pub struct GuidedWeights {
    dir: PathBuf,
    groups: usize,
    tokens: usize,
    layers: usize,
    fingerprint: u64,
    files: BTreeMap<String, String>,
    manifest_sha256: String,
}

impl GuidedWeights {
    /// Open `dir` for a run of `expect_tokens` calibration rows whose token ids fingerprint to
    /// `expect_fingerprint` ([`crate::eval::token_fingerprint`] over the windows in order).
    ///
    /// Refused, by name: a manifest of another format or column count, another token count, another
    /// fingerprint, a layer or a type without its file, and any file whose sha256 is not the
    /// manifest's. Every file is hashed here, before the first block (72 files of 8 MiB at 4B, a few
    /// seconds), so that a corrupted late layer is refused at the start of an encode and not after an
    /// hour and a half. [`Self::load`] hashes each file again as it reads it, block by block.
    pub fn open(dir: &Path, groups: usize, expect_fingerprint: u64, expect_tokens: usize) -> anyhow::Result<Self> {
        parse_groups(Some(&groups.to_string())).map_err(anyhow::Error::msg)?;
        let mpath = dir.join("manifest.json");
        let raw = std::fs::read(&mpath).map_err(|e| anyhow::anyhow!("{}: {e}", mpath.display()))?;
        let manifest_sha256 = crate::digest::sha256_hex(&raw);
        let m: serde_json::Value =
            serde_json::from_slice(&raw).map_err(|e| anyhow::anyhow!("{}: {e}", mpath.display()))?;
        let field = |k: &str| m.get(k).ok_or_else(|| anyhow::anyhow!("{}: no field `{k}`", mpath.display()));
        let format = field("format")?.as_str().unwrap_or("");
        anyhow::ensure!(format == FORMAT, "{}: format {format:?}, expected {FORMAT:?}", mpath.display());
        let g_max = field("g_max")?.as_u64().unwrap_or(0) as usize;
        anyhow::ensure!(g_max == G_MAX, "{}: g_max {g_max}, expected {G_MAX}", mpath.display());
        let tokens = field("tokens")?.as_u64().unwrap_or(0) as usize;
        anyhow::ensure!(
            tokens == expect_tokens,
            "{}: weights for {tokens} tokens, and this run calibrates on {expect_tokens}",
            mpath.display()
        );
        let fp_s = field("token_fingerprint")?.as_str().unwrap_or("");
        let fingerprint = u64::from_str_radix(fp_s, 16)
            .map_err(|_| anyhow::anyhow!("{}: token_fingerprint {fp_s:?} is not 16 hex digits", mpath.display()))?;
        anyhow::ensure!(
            fingerprint == expect_fingerprint,
            "{}: token fingerprint {fingerprint:016x}, and this run's calibration windows fingerprint to \
             {expect_fingerprint:016x}: the weights were computed on other tokens",
            mpath.display()
        );
        let layers = field("layers")?.as_u64().unwrap_or(0) as usize;
        anyhow::ensure!(layers > 0, "{}: layers {layers}", mpath.display());
        let files: BTreeMap<String, String> = field("files")?
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("{}: `files` is not an object", mpath.display()))?
            .iter()
            .map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string()))
            .collect();
        for l in 0..layers {
            for t in TYPES {
                let n = file_name(l, t);
                let want = files.get(&n).ok_or_else(|| anyhow::anyhow!("{}: no entry for {n}", mpath.display()))?;
                let path = dir.join(&n);
                let got =
                    crate::digest::sha256_file(&path).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
                anyhow::ensure!(&got == want, "{}: sha256 {got}, the manifest says {want}", path.display());
            }
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            groups,
            tokens,
            layers,
            fingerprint,
            files,
            manifest_sha256,
        })
    }

    pub fn groups(&self) -> usize {
        self.groups
    }

    pub fn tokens(&self) -> usize {
        self.tokens
    }

    pub fn layers(&self) -> usize {
        self.layers
    }

    /// One line for the run state and the log: what a resume must agree with, and what tells a
    /// guided file from a default one.
    pub fn describe(&self) -> String {
        format!(
            "gate_proj,up_proj G={} tokens={} fingerprint={:016x} manifest_sha256={}",
            self.groups, self.tokens, self.fingerprint, self.manifest_sha256
        )
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The normalised weights of (layer, matrix), one vector per row group, each of [`Self::tokens`]
    /// entries. The file is hashed against the manifest before it is read as numbers.
    pub fn load(&self, layer: usize, name: &str) -> anyhow::Result<Vec<Vec<f32>>> {
        anyhow::ensure!(layer < self.layers, "guided weights cover {} layers, asked for {layer}", self.layers);
        let n = file_name(layer, name);
        let path = self.dir.join(&n);
        let want = self.files.get(&n).ok_or_else(|| anyhow::anyhow!("manifest has no entry for {n}"))?;
        let got = crate::digest::sha256_file(&path).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
        anyhow::ensure!(&got == want, "{}: sha256 {got}, the manifest says {want}", path.display());
        let raw = read_raw(&path, self.tokens)?;
        group_weights(&raw, self.tokens, self.groups).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))
    }
}

/// The calibration token ids of a run, as `bin/smoke` cuts them: the corpus text bounded by
/// [`crate::corpus::calib_chars`], tokenized whole, `n_calib` windows of `calib_len` at
/// [`crate::corpus::window_starts`].
///
/// One source of truth: `smoke` builds its windows from this, and `bin/calibtokens` writes this
/// for `ops/guided_weights.py`. A second copy of the cut could drift, and the weights would land on
/// other tokens than the ones whose activations they weight.
pub struct CalibIds {
    /// The windows, concatenated in order: `n_calib · calib_len` ids.
    pub ids: Vec<u32>,
    /// Tokens the corpus text held.
    pub read: usize,
    /// Whole windows of `calib_len` it held.
    pub available: usize,
    pub starts: Vec<usize>,
}

impl CalibIds {
    pub fn fingerprint(&self) -> u64 {
        crate::eval::token_fingerprint(&self.ids)
    }
}

pub fn calibration_ids(
    tok: &tokenizers::Tokenizer,
    calib: crate::corpus::CalibCorpus,
    n_calib: usize,
    calib_len: usize,
    calib_seed: Option<u64>,
) -> anyhow::Result<CalibIds> {
    let train = calib.text(crate::corpus::calib_chars(n_calib, calib_len))?;
    let train_ids = tok.encode(train.as_str(), false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec();
    anyhow::ensure!(n_calib > 0, "n_calib must be > 0");
    let available = train_ids.len() / calib_len;
    anyhow::ensure!(
        available >= n_calib,
        "the calibration corpus only holds {available} windows of {calib_len} ({} tokens read) — {n_calib} \
         requested. This run fails here rather than serving fewer in silence: a published volume must be \
         a measured volume.",
        train_ids.len()
    );
    let starts = crate::corpus::window_starts(n_calib, train_ids.len(), calib_len, calib_seed);
    let mut ids = Vec::with_capacity(n_calib * calib_len);
    for &s in &starts {
        ids.extend_from_slice(&train_ids[s..s + calib_len]);
    }
    Ok(CalibIds {
        ids,
        read: train_ids.len(),
        available,
        starts,
    })
}

/// Write `ids` as little-endian u32 to `path`, and its sidecar `path.json`. Returns the sha256 of
/// the id file.
pub fn write_token_file(path: &Path, ids: &[u32], windows: usize, window_len: usize, meta: &[(&str, String)]) -> anyhow::Result<String> {
    anyhow::ensure!(ids.len() == windows * window_len, "{} ids for {windows} × {window_len}", ids.len());
    let mut bytes = Vec::with_capacity(ids.len() * 4);
    for id in ids {
        bytes.extend_from_slice(&id.to_le_bytes());
    }
    std::fs::write(path, &bytes)?;
    let sha = crate::digest::sha256_hex(&bytes);
    let mut j = serde_json::Map::new();
    j.insert("format".into(), TOKENS_FORMAT.into());
    j.insert("windows".into(), windows.into());
    j.insert("window_len".into(), window_len.into());
    j.insert("tokens".into(), ids.len().into());
    j.insert("fingerprint".into(), format!("{:016x}", crate::eval::token_fingerprint(ids)).into());
    j.insert("sha256".into(), sha.clone().into());
    for (k, v) in meta {
        j.insert((*k).into(), v.clone().into());
    }
    let side = PathBuf::from(format!("{}.json", path.display()));
    std::fs::write(&side, serde_json::to_string_pretty(&serde_json::Value::Object(j))? + "\n")?;
    Ok(sha)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_parse_divisors_of_sixteen_only() {
        assert_eq!(parse_groups(None), Ok(DEFAULT_GROUPS));
        assert_eq!(parse_groups(Some("")), Ok(DEFAULT_GROUPS));
        for g in [1, 2, 4, 8, 16] {
            assert_eq!(parse_groups(Some(&g.to_string())), Ok(g));
        }
        for bad in ["0", "3", "32", "-1", "four", "4.0"] {
            let e = parse_groups(Some(bad)).expect_err(bad);
            assert!(e.contains(ENV_GROUPS), "{e}");
        }
    }

    #[test]
    fn the_variables_are_refused_by_name_when_malformed() {
        let env = |pairs: &'static [(&'static str, &'static str)]| {
            move |k: &str| pairs.iter().find(|(n, _)| *n == k).map(|(_, v)| v.to_string())
        };
        assert_eq!(from_env(env(&[])), Ok(None));
        let e = from_env(env(&[(ENV_GROUPS, "4")])).expect_err("groups alone");
        assert!(e.contains(ENV_GROUPS) && e.contains(ENV_DIR), "{e}");
        let e = from_env(env(&[(ENV_DIR, " ")])).expect_err("empty dir");
        assert!(e.contains(ENV_DIR), "{e}");
        let e = from_env(env(&[(ENV_DIR, "/w"), (ENV_GROUPS, "5")])).expect_err("bad G");
        assert!(e.contains(ENV_GROUPS), "{e}");
        assert_eq!(from_env(env(&[(ENV_DIR, "/w")])), Ok(Some((PathBuf::from("/w"), DEFAULT_GROUPS))));
        assert_eq!(from_env(env(&[(ENV_DIR, "/w"), (ENV_GROUPS, "16")])), Ok(Some((PathBuf::from("/w"), 16))));
    }

    /// Each group's weights have mean 1 after normalisation, and keep their proportions.
    #[test]
    fn group_weights_sum_neighbours_and_normalise_to_mean_one() {
        let tokens = 5;
        let raw: Vec<f32> = (0..tokens * G_MAX).map(|i| ((i % 7) as f32) * 0.5 + 0.25).collect();
        for groups in [1, 2, 4, 8, 16] {
            let w = group_weights(&raw, tokens, groups).expect("weights");
            assert_eq!(w.len(), groups);
            let span = G_MAX / groups;
            for (g, v) in w.iter().enumerate() {
                let mean: f64 = v.iter().map(|&x| x as f64).sum::<f64>() / tokens as f64;
                assert!((mean - 1.0).abs() < 1e-6, "G={groups} group {g}: mean {mean}");
                let s: Vec<f64> = (0..tokens)
                    .map(|t| raw[t * G_MAX + g * span..t * G_MAX + (g + 1) * span].iter().map(|&x| x as f64).sum())
                    .collect();
                let sm = s.iter().sum::<f64>() / tokens as f64;
                for t in 0..tokens {
                    assert!(((v[t] as f64) - s[t] / sm).abs() < 1e-6, "G={groups} group {g} token {t}");
                }
            }
        }
        let zero = vec![0.0f32; 2 * G_MAX];
        assert!(group_weights(&zero, 2, 1).expect_err("all zero").contains("zero"));
    }

    #[test]
    fn row_ranges_are_consecutive_equal_chunks() {
        assert_eq!(row_ranges(64, 2).expect("ok"), vec![0..32, 32..64]);
        assert_eq!(row_ranges(9728, 4).expect("ok")[3], 7296..9728);
        assert!(row_ranges(24, 1).is_err(), "24 rows is not a multiple of 16");
        assert!(row_ranges(64, 3).is_err());
    }
}
