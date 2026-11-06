//! Group-affine int8 quantization of the KV cache.
//!
//! The cache is the one tensor whose size grows with the conversation: on a
//! 70B at 8k it is 2.7 GB in f16, and it is what stands between a 32 GB card
//! and a long context. This module stores it as `w ≈ scale·q + bias` over
//! groups of 64, `q ∈ [0, 256)`, scale and bias one f16 each per group — the
//! same scheme [`crate::embedquant`] ships for the embedding, and the same one
//! MLX calls `q8`.
//!
//! ## The group must never cross the time axis
//!
//! This is the one design constraint, and it is not a preference. The cache is
//! `[b, n_kv, t, head_dim]`; grouping along `head_dim` (128 on Qwen3-4B, so two
//! groups of 64 per `(token, head)`) makes the bytes a **function of the token
//! alone**. A prefill of `l` tokens at once and `l` single-token decode steps
//! then produce **identical** bytes.
//!
//! Group along `t` instead and they diverge: the prefill would see a whole
//! window's min and max, the decode step only its own. Nothing in perplexity or
//! MMLU would flag it — the model would simply be a slightly different model in
//! generation than in scoring. `LLVQ_VERIFY_CACHE=1` is what catches it, and it
//! only catches it because the granularity here is `t`-independent.
//!
//! ## What the encoder must not get wrong
//!
//! `Tensor::to_dtype(U8)` **truncates toward zero** in candle's Metal cast
//! kernel (`cast.metal`, `static_cast<U>(static_cast<IR>(x))`). Rounding is
//! therefore explicit and load-bearing: dropping it costs half a quantum of
//! systematic bias, which neither perplexity nor MMLU would report as a bug.
//! `the_round_is_load_bearing` fails without it.
//!
//! Scale and bias are stored as f16 and **read back through f16** before `q` is
//! chosen, so the error the encoder minimizes is the error the reader sees —
//! the same discipline as `embedquant`.

use candle_core::{DType, Result, Tensor};

/// How the KV cache sits in memory.
///
/// Resolved once, from `LLVQ_KV`, and printed next to the dtype and the token
/// fingerprint on every result line.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum KvMode {
    /// The shipped behaviour: keys and values kept at the model's dtype.
    #[default]
    F16,
    /// int8 payload, group of 64 along `head_dim`, f16 scale and bias per group.
    Q8,
}

/// Values per group. Divides `head_dim` on every model this repository loads
/// (128 on Qwen3-4B and 8B), so no group is ever short and no group ever spans
/// two heads.
pub const GROUP: usize = 64;

impl KvMode {
    /// Parse the value of `LLVQ_KV`. Same contract as [`crate::fused::EmbedMode`]:
    /// unset and empty mean the default, anything else must name a mode
    /// exactly — a typo silently falling back would make an A/B lie.
    pub fn parse(v: Option<&str>) -> std::result::Result<Self, String> {
        match v {
            None | Some("") => Ok(Self::F16),
            Some("f16") => Ok(Self::F16),
            Some("q8") => Ok(Self::Q8),
            Some(other) => Err(format!(
                "LLVQ_KV={other}: accepted values \"f16\" (default) and \"q8\""
            )),
        }
    }

    /// Resolve from the environment.
    pub fn from_env() -> std::result::Result<Self, String> {
        let v = std::env::var("LLVQ_KV").ok();
        Self::parse(v.as_deref())
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Q8 => "q8",
        }
    }

    /// Bits per stored value, scale and bias included — the number a memory
    /// claim is made in. `8 + 32/64 = 8.5` for `Q8`.
    ///
    /// ⚠️ This is **bits per cache value**, never b/param: the cache is not a
    /// parameter, and its size is stated in bytes per token with the context,
    /// the batch and the geometry named.
    pub fn bits_per_value(&self, dtype: DType) -> f64 {
        match self {
            Self::F16 => (dtype.size_in_bytes() * 8) as f64,
            Self::Q8 => 8.0 + 32.0 / GROUP as f64,
        }
    }
}

/// Quantize along the innermost axis and widen straight back.
///
/// Returns a tensor of the input's dtype and shape, holding exactly what a
/// reader of the int8 payload would reconstruct. Storing the payload itself is
/// the memory win; this round trip is what makes the **quality** of that win
/// measurable with no kernel and no format change — the cache keeps its dtype,
/// so every consumer downstream (`repeat_kv`, the attention matmul) is
/// untouched.
///
/// ⚠️ Therefore: this function establishes what q8 **costs in quality**, and
/// nothing about what it **saves in memory**. The saving is a count (÷1.882 on
/// the 4B geometry) and is reported, not measured here.
pub fn quantize_dequantize(x: &Tensor) -> Result<Tensor> {
    let dt = x.dtype();
    let dims = x.dims().to_vec();
    let inner = *dims.last().expect("cache tensors are at least 1-D");
    assert_eq!(
        inner % GROUP,
        0,
        "head_dim {inner} is not a multiple of the group of {GROUP}: a short \
         group would make the last group of a head a different quantizer, and \
         nothing downstream would report it"
    );

    // [..., inner] -> [..., inner/GROUP, GROUP]: the group becomes an axis, so
    // min and max are one reduction and never reach across a head or a token.
    let mut g = dims.clone();
    g.pop();
    g.push(inner / GROUP);
    g.push(GROUP);
    let xg = x.reshape(g)?.to_dtype(DType::F32)?;

    let mn = xg.min_keepdim(candle_core::D::Minus1)?;
    let mx = xg.max_keepdim(candle_core::D::Minus1)?;

    // Scale and bias go through f16 exactly as the file would store them, so
    // `q` is chosen against the values a reader gets back.
    let qmax = 255.0f64;
    let s = ((mx - &mn)? / qmax)?
        .to_dtype(DType::F16)?
        .to_dtype(DType::F32)?;
    let mn = mn.to_dtype(DType::F16)?.to_dtype(DType::F32)?;

    // A group whose range rounds to zero in f16 has nothing to grade: every q
    // is 0 and the bias carries the value. Dividing by it would be a NaN, so
    // the guard is arithmetic, not a branch — `graded` is 0 or 1, and `safe`
    // is never zero.
    let graded = s.gt(&s.zeros_like()?)?.to_dtype(DType::F32)?;
    let ungraded = (graded.ones_like()? - &graded)?;
    let safe = ((&s * &graded)? + &ungraded)?;

    // Every op against a per-group statistic broadcasts over the group axis:
    // candle's `-`, `/`, `*` and `+` require equal shapes and would not.
    let q = xg
        .broadcast_sub(&mn)?
        .broadcast_div(&safe)?
        // Explicit: `to_dtype(U8)` truncates toward zero.
        .round()?
        .clamp(0.0, qmax)?
        .to_dtype(DType::U8)?;

    let graded_v = q.to_dtype(DType::F32)?.broadcast_mul(&s)?.broadcast_add(&mn)?;
    // Ungraded groups reconstruct to their bias, which is the constant value.
    let deq = graded_v
        .broadcast_mul(&graded)?
        .broadcast_add(&mn.broadcast_mul(&ungraded)?)?;
    deq.to_dtype(dt)?.reshape(dims)
}

/// Bytes a cache of this geometry occupies per token, both modes.
///
/// Stated per token with the geometry named, because that is the only form in
/// which a KV number means anything: `2 · n_layers · n_kv · head_dim` values
/// for keys and values together, times the bits per value.
pub fn bytes_per_token(n_layers: usize, n_kv: usize, head_dim: usize, mode: KvMode, dt: DType) -> f64 {
    let values = 2.0 * n_layers as f64 * n_kv as f64 * head_dim as f64;
    values * mode.bits_per_value(dt) / 8.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// Qwen3-4B and 8B geometry: 36 layers, 8 KV heads, head_dim 128.
    fn cache_like(t: usize, dev: &Device) -> Result<Tensor> {
        // Deterministic, and deliberately not uniform: real keys and values
        // have very different spreads per head, which is what a per-group
        // scale exists to track.
        // batch 1 × 8 KV heads × t tokens × head_dim 128.
        let n = 8 * t * 128;
        let v: Vec<f32> = (0..n)
            .map(|i| {
                let head = (i / (t * 128)) % 8;
                let amp = 1.0 + head as f32;
                let x = (i % 977) as f32 / 977.0 - 0.5;
                amp * x * x.signum() * 3.0
            })
            .collect();
        Tensor::from_vec(v, (1, 8, t, 128), dev)?.to_dtype(DType::F16)
    }

    #[test]
    fn the_env_contract_matches_embed_mode() {
        assert_eq!(KvMode::parse(None), Ok(KvMode::F16));
        assert_eq!(KvMode::parse(Some("")), Ok(KvMode::F16));
        assert_eq!(KvMode::parse(Some("f16")), Ok(KvMode::F16));
        assert_eq!(KvMode::parse(Some("q8")), Ok(KvMode::Q8));
        // A typo is an error, never a silent default — the whole point.
        assert!(KvMode::parse(Some("Q8")).is_err());
        assert!(KvMode::parse(Some("int8")).is_err());
        assert!(KvMode::parse(Some("q4")).is_err());
    }

    /// The constraint of the module header, as a test: the same token must
    /// produce the same bytes whether it arrives in a prefill of many or as a
    /// single decode step.
    ///
    /// This is what `LLVQ_VERIFY_CACHE=1` checks end to end; here it is checked
    /// at the quantizer, where a failure names its cause.
    #[test]
    fn prefill_and_decode_agree_bit_for_bit() -> Result<()> {
        let dev = Device::Cpu;
        let full = cache_like(16, &dev)?;
        let all = quantize_dequantize(&full)?;

        for t in [0usize, 1, 7, 15] {
            let step = full.narrow(2, t, 1)?.contiguous()?;
            let alone = quantize_dequantize(&step)?;
            let within = all.narrow(2, t, 1)?.contiguous()?;
            let d = (alone.to_dtype(DType::F32)? - within.to_dtype(DType::F32)?)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            assert_eq!(
                d, 0.0,
                "token {t} quantizes differently alone than inside a prefill: \
                 the group crosses the time axis"
            );
        }
        Ok(())
    }

    /// The degenerate branch is reachable and must not silently flatten a group.
    ///
    /// A group whose range rounds to zero in f16 takes the ungraded path. On a
    /// constant group that is exact; the test pins that it *is* exact rather
    /// than merely accepted.
    #[test]
    fn a_constant_group_reconstructs_exactly() -> Result<()> {
        let dev = Device::Cpu;
        // First group constant (ungraded path), second group graded — one
        // tensor, both paths, so a regression on either shows up here.
        let v: Vec<f32> = (0..128)
            .map(|i| if i < GROUP { 0.375 } else { i as f32 / 128.0 })
            .collect();
        let t = Tensor::from_vec(v, (1, 1, 1, 128), &dev)?.to_dtype(DType::F16)?;
        let out = quantize_dequantize(&t)?;

        let a = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let b = out.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        for i in 0..GROUP {
            assert_eq!(
                b[i], a[i],
                "slot {i} of a constant group must come back exactly: the bias \
                 carries the value, there is nothing to grade"
            );
        }
        // And the graded half is not flattened: it must still vary.
        let lo = b[GROUP..].iter().cloned().fold(f32::INFINITY, f32::min);
        let hi = b[GROUP..].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            hi - lo > 0.4,
            "the graded group collapsed to a constant ({lo} … {hi}): the \
             ungraded branch is leaking into groups that have something to grade"
        );
        Ok(())
    }

    /// Removing `round()` must break something — otherwise the truncation of
    /// `to_dtype(U8)` would ship a half-quantum bias that no evaluation reports.
    #[test]
    fn the_round_is_load_bearing() -> Result<()> {
        let dev = Device::Cpu;
        let x = cache_like(4, &dev)?;
        let got = quantize_dequantize(&x)?;

        // A truncating encoder, written here so the comparison is against a
        // concrete alternative rather than against a hope.
        let dims = x.dims().to_vec();
        let inner = *dims.last().unwrap();
        let mut g = dims.clone();
        g.pop();
        g.push(inner / GROUP);
        g.push(GROUP);
        let xg = x.reshape(g)?.to_dtype(DType::F32)?;
        let mn = xg
            .min_keepdim(candle_core::D::Minus1)?
            .to_dtype(DType::F16)?
            .to_dtype(DType::F32)?;
        let mx = xg.max_keepdim(candle_core::D::Minus1)?;
        let s = ((mx - &mn)? / 255.0)?
            .to_dtype(DType::F16)?
            .to_dtype(DType::F32)?;
        let trunc = xg
            .broadcast_sub(&mn)?
            .broadcast_div(&s)?
            .clamp(0.0, 255.0)?
            .to_dtype(DType::U8)?;
        let deq = trunc
            .to_dtype(DType::F32)?
            .broadcast_mul(&s)?
            .broadcast_add(&mn)?
            .to_dtype(x.dtype())?
            .reshape(dims)?;

        let bias_round = (got.to_dtype(DType::F32)? - x.to_dtype(DType::F32)?)?
            .mean_all()?
            .to_scalar::<f32>()?;
        let bias_trunc = (deq.to_dtype(DType::F32)? - x.to_dtype(DType::F32)?)?
            .mean_all()?
            .to_scalar::<f32>()?;
        assert!(
            bias_round.abs() * 4.0 < bias_trunc.abs(),
            "rounding must cut the systematic bias of truncation by far more \
             than a factor 4 — got {bias_round:e} rounded against {bias_trunc:e} \
             truncated; if these are close, round() is not load-bearing and the \
             gate of §3.3 is not doing its job"
        );
        Ok(())
    }

    /// The accounting the memory claim is made in, pinned.
    #[test]
    fn the_geometry_is_the_published_one() {
        // Qwen3-4B: 36 layers, 8 KV heads, head_dim 128 — 147,456 o/token f16.
        let f16 = bytes_per_token(36, 8, 128, KvMode::F16, DType::F16);
        assert_eq!(f16, 147_456.0);
        // q8 g64 with f16 scale and bias: 8.5 bits, so ÷1.882 and NOT ÷2.
        let q8 = bytes_per_token(36, 8, 128, KvMode::Q8, DType::F16);
        assert_eq!(q8, 78_336.0);
        let ratio = f16 / q8;
        assert!(
            (ratio - 1.882).abs() < 5e-4,
            "the ratio is 1.882, not 2 — the scales and biases are not free: {ratio}"
        );
    }

    /// The `LLVQ_KV_PREALLOC` contract, same shape as `LLVQ_KV`'s: unset and
    /// empty mean the default, `0` and typos are refused by name — a fallback
    /// would make an A/B lie.
    #[test]
    fn kv_store_parse_contract() {
        assert_eq!(KvStore::parse(None).unwrap(), KvStore::Cat);
        assert_eq!(KvStore::parse(Some("")).unwrap(), KvStore::Cat);
        assert_eq!(KvStore::parse(Some("256")).unwrap(), KvStore::Prealloc(256));
        for bad in ["0", "abc", "-1", "1.5", " 256"] {
            let e = KvStore::parse(Some(bad)).expect_err(bad);
            assert!(e.contains("LLVQ_KV_PREALLOC"), "{e}");
        }
        assert_eq!(KvStore::Cat.label(), "cat");
        assert_eq!(KvStore::Prealloc(256).label(), "prealloc(256)");
        assert_eq!(KvStore::Grow(256).label(), "grow(256)");
        assert_eq!(AttnHeads::default(), AttnHeads::Expanded);
    }

    /// The `attention` field's contract, which is NOT the variables': an
    /// absent field means the published form, and an empty one is refused.
    #[test]
    fn attn_heads_parse_contract() {
        assert_eq!(AttnHeads::parse(None).unwrap(), AttnHeads::Expanded);
        assert_eq!(AttnHeads::parse(Some("expanded")).unwrap(), AttnHeads::Expanded);
        assert_eq!(AttnHeads::parse(Some("grouped")).unwrap(), AttnHeads::Grouped);
        assert_eq!(AttnHeads::parse(Some("kernel")).unwrap(), AttnHeads::Kernel);
        assert_eq!(AttnHeads::Kernel.label(), "kernel");

        // Empty is refused, and so is whitespace: "  " trimmed is "", and a
        // field nobody wrote must not read as a value.
        for empty in ["", "   "] {
            let e = AttnHeads::parse(Some(empty)).expect_err("an empty field");
            assert!(e.contains("empty"), "{e}");
        }
        // Shared is refused BY NAME, with the reason, not as a typo.
        let e = AttnHeads::parse(Some("shared")).expect_err("shared is not served");
        assert!(e.contains("gqa-ab-2026-11-01"), "the refusal cites its journal: {e}");
        // And a typo is a typo, never a default.
        for bad in ["Grouped", "group", "gqa", "cat"] {
            let e = AttnHeads::parse(Some(bad)).expect_err(bad);
            assert!(e.contains(bad), "the message must name the value: {e}");
        }
    }

    /// The store rides the form. A `Grouped` config on a `Cat` store would
    /// measure neither arm of the A/B that licensed it.
    #[test]
    fn the_store_rides_the_attention_form() {
        assert_eq!(AttnHeads::Expanded.store(), KvStore::Cat);
        assert_eq!(AttnHeads::Grouped.store(), KvStore::Grow(GROW_CHUNK));
        assert_eq!(AttnHeads::Shared.store(), KvStore::Grow(GROW_CHUNK));
        assert_eq!(AttnHeads::Kernel.store(), KvStore::Grow(GROW_CHUNK));
        // The kernel stages whole splits: the chunk must be a multiple of one.
        assert_eq!(GROW_CHUNK % crate::attn::canon::SPLIT, 0);
        // The chunk the A/B of 2026-11-01 ran, named once.
        assert_eq!(GROW_CHUNK, 256);
    }
}

/// How the KV cache GROWS: by concatenation (shipped), into a buffer
/// preallocated to a fixed window, or into a buffer grown by whole chunks.
///
/// Resolved once, from `LLVQ_KV_PREALLOC`, by the binaries — `model.rs` reads
/// no environment variable — and printed on the arm line of every A/B, so an
/// unwired arm is visible instead of silently measuring `Cat`.
///
/// `Prealloc` exists for A2 (CUDA Graphs, preregistration `802006c5…`): a
/// static graph cannot capture a `Tensor::cat` whose output grows, so the
/// history must live at a stable address. The window is a hard bound — going
/// past it is a named error, never a silent wrap: a ring would silently
/// change what attention sees, and no experiment of this repository wants
/// that behaviour implicitly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum KvStore {
    /// The shipped behaviour: `Tensor::cat` per step, unbounded.
    #[default]
    Cat,
    /// Fixed buffers of this many positions, written in place per step.
    Prealloc(usize),
    /// Buffers grown by whole chunks of this many positions, written in place
    /// per step; attention reads a view of the valid prefix. The history is
    /// copied only when a chunk fills, not at every step as under `Cat`, and
    /// no window is promised in advance. Paired with
    /// [`AttnHeads::Grouped`], a decode step reads each cached byte once and
    /// copies none. Set by `bin/fusedrun`'s `LLVQ_GQA_AB` only, so far.
    Grow(usize),
}

impl KvStore {
    /// Parse the value of `LLVQ_KV_PREALLOC`. Same contract as [`KvMode`]:
    /// unset and empty mean the default, anything else must be a positive
    /// decimal window — `0` and typos are refused, a fallback would make an
    /// A/B lie.
    pub fn parse(v: Option<&str>) -> std::result::Result<Self, String> {
        match v {
            None | Some("") => Ok(Self::Cat),
            Some(other) => match other.parse::<usize>() {
                Ok(w) if w > 0 => Ok(Self::Prealloc(w)),
                _ => Err(format!(
                    "LLVQ_KV_PREALLOC={other}: accepted values \"\" (default, \
                     concatenation) or a whole window > 0"
                )),
            },
        }
    }

    /// Resolve from the environment.
    pub fn from_env() -> std::result::Result<Self, String> {
        let v = std::env::var("LLVQ_KV_PREALLOC").ok();
        Self::parse(v.as_deref())
    }

    /// What the arm line prints.
    pub fn label(&self) -> String {
        match self {
            Self::Cat => "cat".to_string(),
            Self::Prealloc(w) => format!("prealloc({w})"),
            Self::Grow(c) => format!("grow({c})"),
        }
    }

}

/// The chunk a `Grow` store is sized by when a binary picks it, in positions.
///
/// A trade between two small costs, *computed* on Qwen3-4B in f16 (147 KB a
/// position over 36 layers): at most 255 positions held and unused, 37 MB;
/// and one copy of the history every 256 tokens, which at 8,192 positions is
/// 32 positions' worth of copying a token against 8,192 read by attention.
pub const GROW_CHUNK: usize = 256;

/// The first position the `Kernel` form sends to the attention kernel. A row
/// at position `p` (the cache held `p` positions before it was written) goes
/// to the kernel iff `p >= KERNEL_FROM`, and to the grouped path otherwise.
/// The rule reads the position only, so the output is the same function of
/// it on every card and in every call shape: a verify call that straddles the
/// threshold is split by row.
///
/// 1,280, from the served 4B on the L40S (*computed* from
/// `docs/mesures/lot1-4b-2026-11-04.txt`, attention r2, *measured*): a decode
/// token costs the grouped form 7.58 / 8.73 ms at 256 / 2,048 positions,
/// linear in the position at 0.64 us each (five lengths, 5 to 8,192, all
/// within 0.64-0.69); it costs the kernel 8.14 / 8.26 ms there. If the
/// kernel's cost does not fall as the position grows, it is at least 8.14 ms
/// and at most 8.30 ms (the slow end of its 2,048 range) in between, so the
/// two forms cross between 1,130 and 1,380 positions; the straight line
/// through the two ratios puts it at 1,240. 1,024 lies below that bracket,
/// where the grouped form is the faster one (8.07 ms against at least 8.14).
/// 1,280 is the multiple of the 256-position split nearest the line's
/// crossing, inside the bracket and above it rather than below: a threshold
/// set too high costs a little speed over a few positions (under 1.2 % per
/// token at any position of the bracket), one set too low costs the same
/// speed AND hands more positions to arithmetic that is not the grouped
/// form's. A constant of the form: no variable moves it.
pub const KERNEL_FROM: usize = 1280;

/// How attention reads the `n_kv` key and value heads the cache stores.
///
/// `Expanded` is the shipped form: `repeat_kv` copies each head to the
/// `n_heads / n_kv` query heads that read it, the whole history at every step
/// on the `Cat` store. The two other forms read the stored heads where they
/// are, and copy no history:
///
/// - `Shared` runs one matmul per position in the group, each of the
///   expanded form's shape, so each score is computed exactly as before. It
///   reads every key head `n_heads / n_kv` times, as the expanded matmul
///   does, and launches more kernels than it.
/// - `Grouped` stacks the query heads of one key head as rows of one matrix,
///   one matmul per key head, which reads each cached byte once. The matmul
///   shape differs from the expanded one, so its rounding may too: on the CPU
///   a decode step moves the logits by about one unit in the last place.
/// - `Kernel` sends decode and verify rows (up to 8 a call, batch 1) at
///   position [`KERNEL_FROM`] or later to the split-KV decode-attention
///   kernel, `crate::attn`: one launch a layer, an f32 online softmax with
///   explicit rounding. Every other row (a position below the threshold, a
///   prefill chunk, a batch) takes the grouped path, which writes the same
///   cache bytes. Below the threshold a decode step is the grouped form's,
///   bit for bit. It is NOT bit for bit with `Grouped` on the rows the kernel
///   takes: its licence is the operator's T8, passed on the sealed 4B with
///   this threshold (`docs/mesures/attn-t8-r2-2026-11-04.txt`) and extended
///   below it, from position 0, for the persistent decode
///   (`docs/mesures/attn-t8-below-4b-2026-11-05.txt`). No T8 covers it on
///   Metal, or on the 8B and the 14B.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AttnHeads {
    #[default]
    Expanded,
    Shared,
    Grouped,
    Kernel,
}

impl AttnHeads {
    /// What the arm line prints.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Expanded => "expanded",
            Self::Shared => "shared",
            Self::Grouped => "grouped",
            Self::Kernel => "kernel",
        }
    }

    /// Parse the served config's `attention` field. No environment variable
    /// carries this one: the form belongs to the object, so it is a field or
    /// it is nothing.
    ///
    /// `None` — the field absent — is the ONE default this vocabulary allows,
    /// and it is honest where the five required fields' defaults were not:
    /// `Expanded` is the form every published number ran, so the six configs
    /// written before this field existed keep serving exactly what they
    /// served, down to the logits. An EMPTY field is refused like any other,
    /// for the reason `served.rs` gives: a value nobody wrote must not read as
    /// a value.
    ///
    /// `Shared` is refused by name. It is slower than `Grouped` at every
    /// length measured and it is not bit for bit on the card either
    /// (`docs/mesures/gqa-ab-2026-11-01.txt`), so nothing is served under it;
    /// it stays reachable from the A/B modes of `bin/fusedrun`, which is where
    /// those two facts were established.
    pub fn parse(v: Option<&str>) -> std::result::Result<Self, String> {
        match v {
            None => Ok(Self::Expanded),
            Some("expanded") => Ok(Self::Expanded),
            Some("grouped") => Ok(Self::Grouped),
            Some("kernel") => Ok(Self::Kernel),
            Some(s) if s.trim().is_empty() => Err(
                "attention is empty. Write \"expanded\", \"grouped\" or \"kernel\", or \
                 delete the field — absent means \"expanded\", the published form."
                    .to_string(),
            ),
            Some("shared") => Err(
                "attention \"shared\" is not a served form: it is slower than \"grouped\" \
                 at every length measured and not bit for bit on the card either \
                 (gqa-ab-2026-11-01). It exists in the A/B modes of bin/fusedrun only."
                    .to_string(),
            ),
            Some(other) => Err(format!(
                "attention {other:?}: accepted values \"expanded\" (the published form, \
                 also what an absent field means), \"grouped\" or \"kernel\""
            )),
        }
    }

    /// The cache growth store this form needs.
    ///
    /// Not a free choice beside the form, which is why it is read from the
    /// form rather than written beside it: `Expanded` copies the whole history
    /// every step and has always run on `Cat`, and the two unexpanded forms
    /// exist to read the history where it lies, which `Cat` cannot offer
    /// because its buffer moves at every step. A config that could name the
    /// two separately could name a pair no measurement ever ran.
    pub fn store(&self) -> KvStore {
        match self {
            Self::Expanded => KvStore::Cat,
            // `Kernel` reads the `Grow` buffers in place: whole splits of 256
            // positions, which a capacity in whole chunks of 256 guarantees.
            Self::Shared | Self::Grouped | Self::Kernel => KvStore::Grow(GROW_CHUNK),
        }
    }
}

