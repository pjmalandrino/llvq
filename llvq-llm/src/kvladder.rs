//! The KV quality ladder: quantized cache formats measured as round trips.
//!
//! A MEASUREMENT MODE, `LLVQ_KV_LADDER=<arm>`, never a served config: it is in
//! [`crate::served::MEASUREMENT_MODES`], and `bin/fusedrun` reads it in the
//! `ladder` form of `LLVQ_GQA_PPL` only. Each arm quantizes the new keys and
//! values on their way into the cache and widens them straight back, as
//! [`crate::kvq::quantize_dequantize`] does for `LLVQ_KV=q8`: the cache keeps
//! its dtype, so the attention reads exactly what a reader of the packed
//! format would reconstruct, and nothing about the memory saving is measured.
//!
//! | arm | keys | values | bits a value, K / V |
//! |---|---|---|---|
//! | `k8v8` | int8 affine g64 | int8 affine g64 | 8.5 / 8.5 (it is `LLVQ_KV=q8`) |
//! | `k8v4` | int8 affine g64 | int4 affine g64 | 8.5 / 4.5 |
//! | `k4v4c` | int4 symmetric, static step per RoPE pair | int4 affine g64 | 4.0 / 4.5 |
//! | `k4v4h` | Hadamard, int4 affine g64, Hadamard back | same | 4.5 / 4.5 |
//!
//! ## Every arm codes a position from that position alone
//!
//! The groups run along `head_dim`, the Hadamard mixes one head of one token,
//! and the static key steps are constants fixed before the measured text. So
//! the bytes of a cached position are a function of its token only, and a
//! prefill of `l` tokens writes what `l` decode steps write (the
//! [`crate::kvq`] invariant; `every_arm_codes_a_token_alone` locks it).
//!
//! ## Why the static key step is shared by a RoPE pair
//!
//! Keys are cached after RoPE. `candle_nn::rotary_emb::rope`, the form
//! `model/block.rs` calls, rotates channel `i` with channel `i + head_dim/2` (the
//! rotate-half form, not the interleaved `rope_i`). A per-channel step fixed
//! on calibration positions would be wrong at other positions: the rotation
//! moves energy between the two channels of a pair as the position changes.
//! The pair's norm does not move, so the step is a statistic of the pair
//! norm, shared by both channels, and the same at every position
//! (`the_key_step_is_rope_invariant`).
//!
//! ## Rounding
//!
//! Explicit everywhere, as in [`crate::kvq`]: the affine arms go through
//! [`crate::kvq::affine_round_trip`] and its `round()`, the symmetric key arm
//! rounds before it clamps. Steps, scales and biases are read back through f16
//! before the code is chosen.

use candle_core::{DType, Device, Result, Tensor};

use crate::kvq::KvMode;
use crate::model::KvRoundTrip;

/// Values per int4 affine group, keys of `k4v4h` and values of every int4
/// arm. 64, not the 128 of the int4 weight records, for two reasons: it is
/// the group of the int8 arm, so every arm of the ladder lays a head row out
/// as two groups and one kernel layout reads them all; and the group is the
/// only knob of an affine int4 code against a heavy-tailed row, worth
/// 0.25 bit a value (4.5 against 4.25) where the ladder is asking how much
/// error the cache can take. `int4_g64_against_g128` reports both.
pub const V4_GROUP: usize = 64;

/// Levels of the symmetric key code of `k4v4c`: `q ∈ [−7, 7]`, 15 of the 16
/// codes of 4 bits, so that 0 is a code and the code is symmetric about it.
pub const KEY_LEVELS: f64 = 7.0;

/// The default clip quantile of `k4v4c`: the step of a pair is the 0.999
/// quantile of its norm over the calibration positions, divided by
/// [`KEY_LEVELS`]. One explicit parameter, set as `k4v4c:<q>`; `k4v4c:1` is
/// the max-abs statistic.
pub const DEFAULT_CLIP: f64 = 0.999;

/// Calibration positions of `k4v4c`: one prefill of this many `dclm-edu`
/// tokens, the corpus the sealed files were calibrated on and not the
/// WikiText-2 text the ladder scores.
pub const CALIB_TOKENS: usize = 4096;

/// One arm of the ladder, as `LLVQ_KV_LADDER` names it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ladder {
    pub mode: KvMode,
    /// The clip quantile of `k4v4c`; `None` for every other arm.
    pub clip: Option<f64>,
}

impl Ladder {
    /// Parse `LLVQ_KV_LADDER`. Unset is no arm; an empty value and any value
    /// that does not name an arm exactly are refused by name.
    pub fn parse(v: Option<&str>) -> std::result::Result<Option<Self>, String> {
        let accepted = "accepted values \"k8v8\", \"k8v4\", \"k4v4c\" (clip quantile 0.999), \
                        \"k4v4c:<q>\" with 0.5 < q <= 1, \"k4v4h\"";
        let Some(v) = v else { return Ok(None) };
        let arm = |mode| Ok(Some(Self { mode, clip: None }));
        match v {
            "k8v8" => arm(KvMode::Q8),
            "k8v4" => arm(KvMode::K8V4),
            "k4v4h" => arm(KvMode::K4V4H),
            "k4v4c" => Ok(Some(Self { mode: KvMode::K4V4C, clip: Some(DEFAULT_CLIP) })),
            other => match other.strip_prefix("k4v4c:").map(str::parse::<f64>) {
                Some(Ok(q)) if q > 0.5 && q <= 1.0 => Ok(Some(Self { mode: KvMode::K4V4C, clip: Some(q) })),
                _ => Err(format!("LLVQ_KV_LADDER={other:?}: {accepted}")),
            },
        }
    }

    /// Resolve from the environment. `LLVQ_KV` set to anything but its
    /// default beside an arm is a contradiction, refused rather than outvoted.
    pub fn from_env() -> std::result::Result<Option<Self>, String> {
        let ladder = Self::parse(std::env::var("LLVQ_KV_LADDER").ok().as_deref())?;
        if ladder.is_some() && KvMode::from_env()? != KvMode::F16 {
            return Err(
                "LLVQ_KV_LADDER beside LLVQ_KV: the ladder names the cache of its arm, and \
                 its reference arm is the f16 cache. Unset LLVQ_KV."
                    .to_string(),
            );
        }
        Ok(ladder)
    }

    /// The arm as the result lines print it, its bits included.
    pub fn label(&self) -> String {
        let name = match self.mode {
            KvMode::Q8 => "k8v8".to_string(),
            KvMode::K4V4C => format!("k4v4c (clip quantile {})", self.clip.unwrap_or(DEFAULT_CLIP)),
            m => m.name().to_string(),
        };
        format!(
            "{name}, K {} b / V {} b a value",
            self.mode.key_bits(DType::F16),
            self.mode.value_bits(DType::F16)
        )
    }
}

/// What an arm needs besides the tensors: the static key steps of `k4v4c`,
/// `[1, n_kv, 1, head_dim]` f32 for one layer, and the Hadamard matrix of
/// `k4v4h` on the model's device. Installed on each block, as that block's
/// [`KvRoundTrip`], by [`crate::model::Qwen3::set_kv_round_trip`] (see
/// [`LadderAux::boxed`]).
#[derive(Clone, Debug, Default)]
pub struct LadderAux {
    pub key_step: Option<Tensor>,
    pub hadamard: Option<Tensor>,
}

impl LadderAux {
    /// One entry a block, in the form [`crate::model::Qwen3::set_kv_round_trip`]
    /// takes.
    pub fn boxed(aux: Vec<LadderAux>) -> Vec<Box<dyn KvRoundTrip>> {
        aux.into_iter().map(|a| Box::new(a) as Box<dyn KvRoundTrip>).collect()
    }
}

/// The ladder's hook into the forward pass: a block's cache appends through
/// [`round_trip`] with this block's state, so the model names no ladder type.
impl KvRoundTrip for LadderAux {
    fn round_trip(&self, mode: KvMode, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        round_trip(mode, k, v, Some(self))
    }
}

/// Quantize this step's keys and values under a ladder arm and widen them
/// back, at the input's dtype and shape.
pub fn round_trip(mode: KvMode, k: &Tensor, v: &Tensor, aux: Option<&LadderAux>) -> Result<(Tensor, Tensor)> {
    match mode {
        KvMode::F16 => Ok((k.clone(), v.clone())),
        KvMode::Q8 => Ok((crate::kvq::quantize_dequantize(k)?, crate::kvq::quantize_dequantize(v)?)),
        KvMode::K8V4 => Ok((crate::kvq::quantize_dequantize(k)?, int4_affine(v)?)),
        KvMode::K4V4C => {
            let step = aux.and_then(|a| a.key_step.as_ref()).ok_or_else(|| {
                candle_core::Error::msg(
                    "k4v4c with no static key steps: calibrate (kvladder::calibrate) and install \
                     them (Qwen3::set_kv_round_trip) before the first position",
                )
            })?;
            Ok((static_key(k, step)?, int4_affine(v)?))
        }
        KvMode::K4V4H => {
            let built;
            let h = match aux.and_then(|a| a.hadamard.as_ref()) {
                Some(h) => h,
                None => {
                    built = hadamard_matrix(*k.dims().last().expect("4-D"), k.device())?;
                    &built
                }
            };
            Ok((hadamard_int4(k, h)?, hadamard_int4(v, h)?))
        }
    }
}

/// int4 group-affine over [`V4_GROUP`], `q ∈ [0, 15]`, f16 scale and bias.
pub fn int4_affine(x: &Tensor) -> Result<Tensor> {
    crate::kvq::affine_round_trip(x, V4_GROUP, 15.0)
}

/// The symmetric key code of `k4v4c`: `q = clamp(round(k / step), −7, 7)`,
/// read back as `q · step`. `step` is `[1, n_kv, 1, head_dim]`, already
/// rounded through f16. A step of 0 (a pair the calibration never saw move)
/// reads back 0.
pub fn static_key(k: &Tensor, step: &Tensor) -> Result<Tensor> {
    let (_, heads, _, hd) = k.dims4()?;
    let (_, sh, _, shd) = step.dims4()?;
    if (heads, hd) != (sh, shd) {
        candle_core::bail!(
            "k4v4c: keys of {heads} heads x {hd}, static steps for {sh} heads x {shd}. A store that \
             holds the expanded heads (LLVQ_KV_PREALLOC) has no static key steps"
        );
    }
    let graded = step.gt(0.0)?.to_dtype(DType::F32)?;
    let safe = ((step * &graded)? + (graded.ones_like()? - &graded)?)?;
    let q = k
        .to_dtype(DType::F32)?
        .broadcast_div(&safe)?
        // Explicit, and before the clamp: the code is the nearest level.
        .round()?
        .clamp(-KEY_LEVELS, KEY_LEVELS)?;
    q.broadcast_mul(step)?.to_dtype(k.dtype())
}

/// The orthonormal Sylvester-Hadamard matrix of order `n`, a power of two:
/// `H[i][j] = (−1)^popcount(i & j) / √n`. Symmetric, so it is its own
/// inverse: `H · H = I`.
pub fn hadamard_matrix(n: usize, dev: &Device) -> Result<Tensor> {
    if !n.is_power_of_two() {
        candle_core::bail!("k4v4h: head_dim {n} is not a power of two");
    }
    let s = 1.0 / (n as f32).sqrt();
    let h: Vec<f32> = (0..n * n)
        .map(|e| if ((e / n) & (e % n)).count_ones().is_multiple_of(2) { s } else { -s })
        .collect();
    Tensor::from_vec(h, (n, n), dev)
}

/// `k4v4h`'s round trip of one tensor: each head row times `H`, int4 affine
/// g64, times `H` again. The transform is the Walsh-Hadamard transform of the
/// row, applied as one f32 matmul rather than a butterfly: the two differ by
/// f32 rounding only, about 1e-7 of the row, far below an int4 quantum. For
/// keys, `(q H) · (k H) = q · k` in exact arithmetic, and the round trip
/// hands attention `Q(k H) H`, so the scores move by the quantization only.
pub fn hadamard_int4(x: &Tensor, h: &Tensor) -> Result<Tensor> {
    let dims = x.dims().to_vec();
    let hd = *dims.last().expect("at least 1-D");
    let rows = x.elem_count() / hd;
    let flat = x.to_dtype(DType::F32)?.reshape((rows, hd))?;
    let rotated = flat.matmul(h)?;
    let coded = int4_affine(&rotated)?;
    coded.matmul(h)?.reshape(dims)?.to_dtype(x.dtype())
}

/// The clip statistic of `k4v4c`, on the host: for each head `h` and RoPE pair
/// `i < hd/2`, the `clip` quantile (nearest rank) over the `t` positions of
/// `‖(k[h,t,i], k[h,t,i+hd/2])‖`. `keys` is `[n_kv, t, hd]` row-major. Returns
/// `n_kv · hd/2` values, head-major.
pub fn pair_norm_quantile(keys: &[f32], n_kv: usize, t: usize, hd: usize, clip: f64) -> Vec<f32> {
    assert_eq!(keys.len(), n_kv * t * hd, "keys are [n_kv, t, hd]");
    assert!(t > 0 && hd.is_multiple_of(2), "positions to calibrate on, and pairs");
    let half = hd / 2;
    let rank = ((clip * t as f64).ceil() as usize).clamp(1, t) - 1;
    let mut out = Vec::with_capacity(n_kv * half);
    let mut r = vec![0f32; t];
    for h in 0..n_kv {
        for i in 0..half {
            for (j, rj) in r.iter_mut().enumerate() {
                let row = &keys[(h * t + j) * hd..(h * t + j + 1) * hd];
                *rj = row[i].hypot(row[i + half]);
            }
            r.sort_unstable_by(f32::total_cmp);
            out.push(r[rank]);
        }
    }
    out
}

/// The static key steps of one layer from its pair statistics: each pair's
/// statistic over [`KEY_LEVELS`], rounded through f16 as a stored step would
/// be, and written to both channels of the pair. `[1, n_kv, 1, hd]` f32.
pub fn key_steps(pair_stat: &[f32], n_kv: usize, hd: usize, dev: &Device) -> Result<Tensor> {
    let half = hd / 2;
    assert_eq!(pair_stat.len(), n_kv * half, "one statistic a head and pair");
    let mut s = vec![0f32; n_kv * hd];
    for h in 0..n_kv {
        for i in 0..half {
            let step = half::f16::from_f32(pair_stat[h * half + i] / KEY_LEVELS as f32).to_f32();
            s[h * hd + i] = step;
            s[h * hd + i + half] = step;
        }
    }
    Tensor::from_vec(s, (1, n_kv, 1, hd), dev)
}

/// Calibrate `k4v4c`: one prefill of `ids` on an f16 cache, then each
/// layer's keys, as cached (after RoPE), to [`pair_norm_quantile`] and
/// [`key_steps`]. One tensor a layer, on the model's device. The model must
/// cache in f16: the statistic is of the keys, not of a code of them.
pub fn calibrate(model: &crate::model::Qwen3, ids: &[u32], chunk: usize, clip: f64) -> Result<Vec<Tensor>> {
    if model.kv_mode() != KvMode::F16 {
        candle_core::bail!("k4v4c calibration on a {} cache: it reads the f16 keys", model.kv_mode().name());
    }
    let mut caches = model.fresh_caches();
    model.prefill_chunked(ids, chunk, &mut caches)?;
    let mut steps = Vec::with_capacity(caches.len());
    for c in &caches {
        let (k, _) = c.held()?.ok_or_else(|| candle_core::Error::msg("k4v4c calibration: an empty cache"))?;
        let (_, n_kv, t, hd) = k.dims4()?;
        let keys = k.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let stat = pair_norm_quantile(&keys, n_kv, t, hd, clip);
        steps.push(key_steps(&stat, n_kv, hd, model.device())?);
    }
    Ok(steps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// Deterministic standard normals (SplitMix64 and Box-Muller), no crate.
    fn normals(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        let mut next = || {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..n)
            .map(|_| {
                let (u1, u2) = (next().max(1e-12), next());
                ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
            })
            .collect()
    }

    const T: usize = 64;
    const NKV: usize = 8;
    const HD: usize = 128;

    /// `[1, 8, t, 128]` f16: Gaussian rows, every head its own spread.
    fn gaussian(t: usize, seed: u64) -> Tensor {
        let g = normals(NKV * t * HD, seed);
        let v: Vec<f32> = g.iter().enumerate().map(|(i, x)| x * (1.0 + (i / (t * HD)) as f32)).collect();
        Tensor::from_vec(v, (1, NKV, t, HD), &Device::Cpu).unwrap().to_dtype(DType::F16).unwrap()
    }

    /// Gaussian rows with a few channels 20 times larger, at fixed indices
    /// across tokens: the outlier-channel shape of real keys.
    fn outlier_channels(t: usize, seed: u64) -> Tensor {
        let g = normals(NKV * t * HD, seed);
        let v: Vec<f32> = g
            .iter()
            .enumerate()
            .map(|(i, x)| if matches!(i % HD, 3 | 67 | 90) { x * 20.0 } else { *x })
            .collect();
        Tensor::from_vec(v, (1, NKV, t, HD), &Device::Cpu).unwrap().to_dtype(DType::F16).unwrap()
    }

    /// Keys shaped as Qwen3 caches them: per-channel spreads (one per pair,
    /// over two orders of magnitude, a few pairs large), then RoPE at the
    /// positions `offset..offset + t` with the model's form.
    fn rope_keys(t: usize, offset: usize, seed: u64) -> Tensor {
        let g = normals(NKV * t * HD, seed);
        let amp = |c: usize| {
            let p = c % (HD / 2);
            let base = 0.1 + 3.0 * (p as f32 / 64.0).powi(2);
            if matches!(p, 5 | 40 | 41) { base * 15.0 } else { base }
        };
        let v: Vec<f32> = g.iter().enumerate().map(|(i, x)| x * amp(i % HD)).collect();
        let k = Tensor::from_vec(v, (1, NKV, t, HD), &Device::Cpu).unwrap();
        let (cos, sin) = rope_tables(offset, t);
        candle_nn::rotary_emb::rope(&k, &cos, &sin).unwrap().to_dtype(DType::F16).unwrap()
    }

    fn rope_tables(offset: usize, t: usize) -> (Tensor, Tensor) {
        let half = HD / 2;
        let mut c = Vec::with_capacity(t * half);
        let mut s = Vec::with_capacity(t * half);
        for p in offset..offset + t {
            for i in 0..half {
                let theta = p as f64 / 1_000_000f64.powf(2.0 * i as f64 / HD as f64);
                c.push(theta.cos() as f32);
                s.push(theta.sin() as f32);
            }
        }
        (
            Tensor::from_vec(c, (t, half), &Device::Cpu).unwrap(),
            Tensor::from_vec(s, (t, half), &Device::Cpu).unwrap(),
        )
    }

    fn sqnr_db(x: &Tensor, y: &Tensor) -> f64 {
        let a = x.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = y.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let sig: f64 = a.iter().map(|v| (*v as f64).powi(2)).sum();
        let err: f64 = a.iter().zip(&b).map(|(u, v)| (*u as f64 - *v as f64).powi(2)).sum();
        10.0 * (sig / err).log10()
    }

    /// Calibration positions of the tests: enough that the 0.999 quantile is
    /// not the maximum (rank 2,046 of 2,048).
    const CALIB_T: usize = 2048;

    /// Steps for `rope_keys` from calibration positions far from the scored ones.
    fn calibrated_steps(clip: f64) -> Tensor {
        let calib = rope_keys(CALIB_T, 20_000, 99);
        let keys = calib.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        key_steps(&pair_norm_quantile(&keys, NKV, CALIB_T, HD, clip), NKV, HD, &Device::Cpu).unwrap()
    }

    fn aux_for(mode: KvMode) -> LadderAux {
        match mode {
            KvMode::K4V4C => LadderAux { key_step: Some(calibrated_steps(DEFAULT_CLIP)), hadamard: None },
            KvMode::K4V4H => LadderAux { key_step: None, hadamard: Some(hadamard_matrix(HD, &Device::Cpu).unwrap()) },
            _ => LadderAux::default(),
        }
    }

    const ARMS: [KvMode; 4] = [KvMode::Q8, KvMode::K8V4, KvMode::K4V4C, KvMode::K4V4H];

    #[test]
    fn the_variable_names_its_arms_exactly() {
        assert_eq!(Ladder::parse(None), Ok(None));
        assert_eq!(Ladder::parse(Some("k8v8")).unwrap().unwrap().mode, KvMode::Q8);
        assert_eq!(Ladder::parse(Some("k8v4")).unwrap().unwrap().mode, KvMode::K8V4);
        assert_eq!(Ladder::parse(Some("k4v4h")).unwrap().unwrap().mode, KvMode::K4V4H);
        let c = Ladder::parse(Some("k4v4c")).unwrap().unwrap();
        assert_eq!((c.mode, c.clip), (KvMode::K4V4C, Some(DEFAULT_CLIP)));
        assert_eq!(Ladder::parse(Some("k4v4c:1")).unwrap().unwrap().clip, Some(1.0));
        assert_eq!(Ladder::parse(Some("k4v4c:0.99")).unwrap().unwrap().clip, Some(0.99));
        for bad in ["", "q8", "f16", "K8V4", "k4v4", "k4v4c:", "k4v4c:0.5", "k4v4c:1.1", "k4v4c:x", " k8v4"] {
            let e = Ladder::parse(Some(bad)).expect_err(bad);
            assert!(e.contains("LLVQ_KV_LADDER") && e.contains(&format!("{bad:?}")), "{e}");
        }
        // The served variable does not learn the arms: a config cannot name one.
        for arm in ["k8v8", "k8v4", "k4v4c", "k4v4h"] {
            assert!(KvMode::parse(Some(arm)).is_err(), "LLVQ_KV accepted {arm}");
        }
        assert!(Ladder::parse(Some("k4v4c:0.999")).unwrap().unwrap().label().contains("0.999"));
    }

    /// The memory accounting of each arm on the 4B geometry, as a count.
    #[test]
    fn bytes_a_position_on_the_4b() {
        let b = |m| crate::kvq::bytes_per_token(36, 8, 128, m, DType::F16);
        assert_eq!(b(KvMode::F16), 147_456.0);
        assert_eq!(b(KvMode::Q8), 78_336.0);
        assert_eq!(b(KvMode::K8V4), 59_904.0);
        assert_eq!(b(KvMode::K4V4C), 39_168.0);
        assert_eq!(b(KvMode::K4V4H), 41_472.0);
        assert_eq!(KvMode::K8V4.key_bits(DType::F16), 8.5);
        assert_eq!(KvMode::K8V4.value_bits(DType::F16), 4.5);
        assert_eq!(KvMode::K4V4C.key_bits(DType::F16), 4.0);
        assert_eq!(KvMode::K4V4H.key_bits(DType::F16), 4.5);
    }

    /// The invariant of `kvq.rs`, per arm: a token coded alone equals the
    /// same token coded inside a prefill, bit for bit.
    #[test]
    fn every_arm_codes_a_token_alone() -> Result<()> {
        for mode in ARMS {
            let aux = aux_for(mode);
            let k = rope_keys(16, 3, 1);
            let v = gaussian(16, 2);
            let (ka, va) = round_trip(mode, &k, &v, Some(&aux))?;
            for t in [0usize, 1, 9, 15] {
                let (k1, v1) = (k.narrow(2, t, 1)?.contiguous()?, v.narrow(2, t, 1)?.contiguous()?);
                let (kb, vb) = round_trip(mode, &k1, &v1, Some(&aux))?;
                for (alone, all, name) in [(&kb, &ka, "k"), (&vb, &va, "v")] {
                    let within = all.narrow(2, t, 1)?.contiguous()?;
                    let x = alone.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                    let y = within.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                    assert!(
                        x.iter().zip(&y).all(|(a, b)| a.to_bits() == b.to_bits()),
                        "{mode:?} {name}: token {t} codes differently alone than inside a prefill"
                    );
                }
            }
        }
        Ok(())
    }

    /// SQNR of each arm on the synthetic sources, against the expected bands.
    /// The bands have upper bounds: an arm that stopped quantizing (a lost
    /// `round`, a level count gone to 255) reads far above them.
    #[test]
    fn sqnr_of_each_arm_on_synthetic_sources() -> Result<()> {
        let g = gaussian(T, 7);
        let o = outlier_channels(T, 8);
        let r = rope_keys(T, 100, 9);

        let q8 = sqnr_db(&g, &crate::kvq::quantize_dequantize(&g)?);
        let v4 = sqnr_db(&g, &int4_affine(&g)?);
        let v4_128 = sqnr_db(&g, &crate::kvq::affine_round_trip(&g, 128, 15.0)?);
        let h = hadamard_matrix(HD, &Device::Cpu)?;
        let hg = sqnr_db(&g, &hadamard_int4(&g, &h)?);
        let plain_o = sqnr_db(&o, &int4_affine(&o)?);
        let had_o = sqnr_db(&o, &hadamard_int4(&o, &h)?);
        let plain_r = sqnr_db(&r, &int4_affine(&r)?);
        let stat_r = sqnr_db(&r, &static_key(&r, &calibrated_steps(DEFAULT_CLIP))?);
        let stat_r_max = sqnr_db(&r, &static_key(&r, &calibrated_steps(1.0))?);
        println!(
            "SQNR dB: int8 g64 gaussian {q8:.2}; int4 g64 gaussian {v4:.2} (g128 {v4_128:.2}); \
             hadamard int4 g64 gaussian {hg:.3} (plain {v4:.3}); outlier channels: int4 g64 {plain_o:.2}, hadamard {had_o:.2}; \
             rope keys: int4 g64 per token {plain_r:.2}, static per pair q0.999 {stat_r:.2}, max {stat_r_max:.2}"
        );
        assert!((44.0..47.5).contains(&q8), "int8 g64 on a Gaussian: {q8}");
        assert!((19.0..23.5).contains(&v4), "int4 g64 on a Gaussian: {v4}");
        assert!(v4 > v4_128, "g64 must beat g128: {v4} against {v4_128}");
        assert!((19.0..23.5).contains(&hg), "Hadamard int4 on a Gaussian is still a Gaussian code: {hg}");
        assert!(had_o > plain_o + 4.0, "the Hadamard must spread outlier channels: {had_o} against {plain_o}");
        assert!(stat_r > plain_r, "static per-pair steps must beat per-token int4 on channel-shaped keys: {stat_r} against {plain_r}");
        assert!((15.0..19.0).contains(&stat_r), "k4v4c keys: {stat_r}");
        assert!((15.0..19.0).contains(&stat_r_max), "k4v4c:1 keys: {stat_r_max}");
        Ok(())
    }

    /// Each arm through [`round_trip`], keys and values apart: an 8-bit side
    /// reads above 35 dB, a 4-bit side between 12 and 26, on the same
    /// sources. Catches an arm that codes one side at the other's bits.
    #[test]
    fn each_arm_codes_each_side_at_its_bits() -> Result<()> {
        let k = rope_keys(T, 50, 21);
        let v = gaussian(T, 22);
        for mode in ARMS {
            let (kq, vq) = round_trip(mode, &k, &v, Some(&aux_for(mode)))?;
            let (sk, sv) = (sqnr_db(&k, &kq), sqnr_db(&v, &vq));
            println!("{mode:?}: keys {sk:.2} dB, values {sv:.2} dB");
            for (side, db, bits) in [("keys", sk, mode.key_bits(DType::F16)), ("values", sv, mode.value_bits(DType::F16))] {
                if bits > 8.0 {
                    assert!(db > 35.0, "{mode:?} {side}: an 8-bit code at {db:.2} dB");
                } else {
                    assert!((12.0..26.0).contains(&db), "{mode:?} {side}: a 4-bit code at {db:.2} dB");
                }
            }
        }
        Ok(())
    }

    /// The g64 choice for the int4 values, against g128, reported.
    #[test]
    fn int4_g64_against_g128() -> Result<()> {
        let o = outlier_channels(T, 11);
        let a = sqnr_db(&o, &int4_affine(&o)?);
        let b = sqnr_db(&o, &crate::kvq::affine_round_trip(&o, 128, 15.0)?);
        println!("outlier rows: int4 g64 {a:.2} dB, g128 {b:.2} dB");
        assert!(a > b);
        Ok(())
    }

    /// The pair statistic is unchanged by RoPE at any position: the steps
    /// fixed on calibration positions hold at every scored one.
    #[test]
    fn the_key_step_is_rope_invariant() -> Result<()> {
        let k = rope_keys(256, 0, 5);
        let raw = k.to_dtype(DType::F32)?;
        // The same keys rotated further by 12,345 positions.
        let (cos, sin) = rope_tables(12_345, 1);
        let (cos, sin) = (cos.repeat((256, 1))?, sin.repeat((256, 1))?);
        let moved = candle_nn::rotary_emb::rope(&raw, &cos, &sin)?;
        let a = pair_norm_quantile(&raw.flatten_all()?.to_vec1::<f32>()?, NKV, 256, HD, DEFAULT_CLIP);
        let b = pair_norm_quantile(&moved.flatten_all()?.to_vec1::<f32>()?, NKV, 256, HD, DEFAULT_CLIP);
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            assert!((x - y).abs() <= 1e-5 * x.max(1e-3), "pair {i}: {x} before RoPE, {y} after");
        }
        Ok(())
    }

    /// The nearest-rank quantile on a known list, and the two channels of a
    /// pair carrying one step.
    #[test]
    fn the_statistic_and_the_step_layout() -> Result<()> {
        // One head, 10 positions, hd 4: pair 0 has norms 1..=10, pair 1 is 0.
        let mut keys = vec![0f32; 10 * 4];
        for t in 0..10 {
            keys[t * 4] = (t + 1) as f32 * 0.6;
            keys[t * 4 + 2] = (t + 1) as f32 * 0.8;
        }
        let q = |c| pair_norm_quantile(&keys, 1, 10, 4, c);
        assert_eq!(q(1.0), vec![10.0, 0.0]);
        assert_eq!(q(0.9)[0], 9.0);
        assert_eq!(q(0.55)[0], 6.0);
        let s = key_steps(&[7.0, 0.0], 1, 4, &Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(s, vec![1.0, 0.0, 1.0, 0.0]);
        // A zero step reads back zero, and a graded one rounds then clamps.
        let k = Tensor::from_vec(vec![2.6f32, 5.0, -9.0, 0.3], (1, 1, 1, 4), &Device::Cpu)?;
        let step = Tensor::from_vec(vec![1.0f32, 0.0, 1.0, 0.0], (1, 1, 1, 4), &Device::Cpu)?;
        assert_eq!(static_key(&k, &step)?.flatten_all()?.to_vec1::<f32>()?, vec![3.0, 0.0, -7.0, 0.0]);
        Ok(())
    }

    /// The Hadamard is orthonormal and keeps every dot product.
    #[test]
    fn the_hadamard_is_orthonormal() -> Result<()> {
        let h = hadamard_matrix(HD, &Device::Cpu)?;
        let i = h.matmul(&h)?.flatten_all()?.to_vec1::<f32>()?;
        for (n, x) in i.iter().enumerate() {
            let want = if n / HD == n % HD { 1.0 } else { 0.0 };
            assert!((x - want).abs() < 1e-5, "H·H at {n}: {x}");
        }
        assert!(hadamard_matrix(96, &Device::Cpu).is_err());
        Ok(())
    }

    /// `k4v4c` without its steps is an error, never a silent f16 cache.
    #[test]
    fn k4v4c_without_steps_is_refused() {
        let k = rope_keys(2, 0, 3);
        let e = round_trip(KvMode::K4V4C, &k, &k, None).expect_err("no steps");
        assert!(format!("{e}").contains("static key steps"), "{e}");
        let wide = Tensor::zeros((1, 32, 1, HD), DType::F16, &Device::Cpu).unwrap();
        let e = static_key(&wide, &calibrated_steps(DEFAULT_CLIP)).expect_err("expanded heads");
        assert!(format!("{e}").contains("expanded heads"), "{e}");
    }

    /// The `kvq.rs` invariant through the cache, for every `LLVQ_KV_LADDER`
    /// arm and both stores a ladder run uses: positions appended in one call
    /// and the same positions appended one at a time hold the same bytes. The
    /// arm reaches the cache as the block does, through [`KvRoundTrip`].
    #[test]
    fn ladder_caches_hold_a_prefill_as_its_decode_steps() {
        use crate::kvq::KvStore;
        use crate::model::KvCache;
        let dev = Device::Cpu;
        let (n_kv, hd, l) = (2usize, 128usize, 7usize);
        let mut seed = 7u32;
        let mut gen = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    (seed >> 8) as f32 / (1u32 << 23) as f32 - 1.0
                })
                .collect()
        };
        let k = Tensor::from_vec(gen(n_kv * l * hd), (1, n_kv, l, hd), &dev).unwrap();
        let v = Tensor::from_vec(gen(n_kv * l * hd), (1, n_kv, l, hd), &dev).unwrap();
        let stat: Vec<f32> = (0..n_kv * hd / 2).map(|i| 0.5 + (i % 9) as f32 * 0.1).collect();
        let aux = LadderAux {
            key_step: Some(key_steps(&stat, n_kv, hd, &dev).unwrap()),
            hadamard: Some(hadamard_matrix(hd, &dev).unwrap()),
        };
        let hook: &dyn KvRoundTrip = &aux;
        for mode in [KvMode::Q8, KvMode::K8V4, KvMode::K4V4C, KvMode::K4V4H] {
            for store in [KvStore::Cat, KvStore::Grow(4)] {
                let mut whole = KvCache::with_store(mode, store);
                let mut steps = KvCache::with_store(mode, store);
                whole.append_with(&k, &v, Some(hook)).unwrap();
                for t in 0..l {
                    let (kt, vt) = (k.narrow(2, t, 1).unwrap(), v.narrow(2, t, 1).unwrap());
                    steps.append_with(&kt, &vt, Some(hook)).unwrap();
                }
                let (wk, wv) = whole.held().unwrap().unwrap();
                let (sk, sv) = steps.held().unwrap().unwrap();
                for (a, b, name) in [(&wk, &sk, "k"), (&wv, &sv, "v")] {
                    let a: Vec<u32> = a.flatten_all().unwrap().to_vec1::<f32>().unwrap().iter().map(|x| x.to_bits()).collect();
                    let b: Vec<u32> = b.flatten_all().unwrap().to_vec1::<f32>().unwrap().iter().map(|x| x.to_bits()).collect();
                    assert_eq!(a, b, "{mode:?} on {store:?}: {name} of a prefill differs from its decode steps");
                }
                // And the arm is a code, not a copy.
                let raw: Vec<f32> = k.flatten_all().unwrap().to_vec1().unwrap();
                let held: Vec<f32> = wk.flatten_all().unwrap().to_vec1().unwrap();
                assert!(raw != held, "{mode:?}: the keys came back unquantized");
            }
            // A measurement arm with no round trip installed is refused, never
            // appended as f16.
            if mode != KvMode::Q8 {
                let e = KvCache::new(mode).append_with(&k, &v, None).expect_err("no round trip");
                assert!(format!("{e}").contains("no round trip installed"), "{mode:?}: {e}");
            }
        }
    }
}
