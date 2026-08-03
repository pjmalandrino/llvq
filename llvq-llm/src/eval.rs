//! Evaluation knobs that have to be identical across harnesses for their
//! numbers to be comparable to each other.
//!
//! ## Why this module exists
//!
//! `bin/ppl` built its model in F32 and `bin/mmlu` in F16, both hard-coded,
//! neither printed. Each metric was internally consistent — the same dtype on
//! the quantized arm and on the reference — so each number stands on its own.
//! What does not stand is the *cross* claim assembled from them: "we match the
//! paper in perplexity and lose much more in capability" requires the two
//! metrics to describe the same object, and they described an F32 model and an
//! F16 model.
//!
//! The sharper version of the same problem concerns the file we ship.
//! `verify_artifact` proves the artifact decodes bit for bit to the **F32**
//! weights perplexity was measured on. The narrowing to F16 that `bin/mmlu`
//! and `bin/run` apply afterwards happens outside that proof — so the object
//! that scored 57.59 on MMLU was never round-trip verified against anything.
//!
//! The dtype is therefore one named knob, overridable with `LLVQ_DTYPE`, and
//! **printed next to every result**. The per-binary defaults are deliberately
//! unchanged: silently redefining what `bin/ppl` reports would make every
//! published perplexity incomparable to its own documentation, and nobody
//! would see it happen.

use candle_core::{DType, Device};

/// Environment override for a harness's model dtype: `f16`, `bf16` or `f32`.
pub const DTYPE_ENV: &str = "LLVQ_DTYPE";

/// Resolve `cpu` / `cuda` / `metal` into a device, **failing loudly**.
///
/// ## Why the fallback had to go
///
/// Every binary used to write `Device::new_metal(0).unwrap_or(Device::Cpu)`,
/// which swallows the error. On a laptop that is a mild annoyance: a broken
/// Metal init silently runs ~7× slower. On rented hardware billed by the
/// minute it is a direct, invisible cost — a job on a GPU flavor that fails to
/// initialize CUDA would run the whole quantization on the flavor's handful of
/// vCPUs, at GPU prices, and report nothing wrong. A 32B run costs four times
/// more on the wrong device.
///
/// So an unavailable accelerator is now an error naming the feature that is
/// missing. Falling back to CPU stays possible — by asking for `cpu`, which is
/// a decision someone took rather than one a `unwrap_or` took for them.
pub fn device(name: &str) -> anyhow::Result<Device> {
    match name.trim().to_ascii_lowercase().as_str() {
        "cpu" => Ok(Device::Cpu),
        // candle defines both constructors unconditionally and returns
        // `NotCompiledWith*Support` when the feature is off, so there is no
        // `cfg` here — the error message already says which flag is missing.
        "cuda" | "gpu" => Device::new_cuda(0).map_err(|e| {
            anyhow::anyhow!(
                "CUDA requested but unavailable: {e}\n\
                 Build with `--features cuda`, or pass `cpu` deliberately."
            )
        }),
        "metal" => Device::new_metal(0).map_err(|e| {
            anyhow::anyhow!(
                "Metal requested but unavailable: {e}\n\
                 Build with `--features metal`, or pass `cpu` deliberately."
            )
        }),
        other => anyhow::bail!("unknown device {other:?} — use cpu, cuda or metal"),
    }
}

/// Resolve the model dtype: `default`, unless `LLVQ_DTYPE` overrides it.
pub fn dtype(default: DType) -> anyhow::Result<DType> {
    match std::env::var(DTYPE_ENV) {
        Ok(s) => parse_dtype(&s),
        Err(_) => Ok(default),
    }
}

/// An unknown name is an error, not a fallback to the default.
///
/// A typo silently scoring in the wrong precision is exactly the failure this
/// module exists to close; falling back would reinstate it one layer down.
pub fn parse_dtype(s: &str) -> anyhow::Result<DType> {
    match s.trim().to_ascii_lowercase().as_str() {
        "f16" | "fp16" | "half" => Ok(DType::F16),
        "bf16" | "bfloat16" => Ok(DType::BF16),
        "f32" | "fp32" | "float" => Ok(DType::F32),
        other => anyhow::bail!("{DTYPE_ENV}={other:?} is not a dtype — use f16, bf16 or f32"),
    }
}

/// The name to print beside a number, so that no result is dtype-anonymous.
pub fn dtype_name(d: DType) -> String {
    format!("{d:?}").to_ascii_lowercase()
}

/// Round an `f32` to what `dtype` can hold, then widen it back.
///
/// ## Why the artifact proof needs this
///
/// `verify_artifact` decodes the file and demands the evaluated weights back
/// **bit for bit**. The decoder produces `f32`; the model holds its weights at
/// the run's dtype. At F32 those are the same object and the comparison is
/// exact — which is why the proof has always held and must keep holding.
///
/// At F16 or BF16 they are not the same object: the reconstruction is computed
/// in f64, written to the model at the model's precision, and the file still
/// decodes to the full-precision value. The honest statement becomes "the file
/// decodes to the evaluated weights **at the precision the model stores
/// them**" — the same claim, evaluated at the run's dtype, and still strong
/// enough to catch a wrong codebook, a wrong rotation seed or a mis-ordered
/// index.
///
/// This is the move `calib.rs` already makes on tail columns: narrow to the
/// precision it is stored at, before anything else reads it.
pub fn narrow(v: f32, dtype: DType) -> f32 {
    match dtype {
        DType::F16 => half::f16::from_f32(v).to_f32(),
        DType::BF16 => half::bf16::from_f32(v).to_f32(),
        _ => v,
    }
}

/// A dependency-free fingerprint of a token stream (FNV-1a, 64 bits).
///
/// Two perplexities are comparable only if they scored the *same tokens*. That
/// is normally assumed — same corpus, same context length — and the assumption
/// holds right up until the two arms take their tokenizer from different
/// places, which is exactly what happens when the baseline loads a checkpoint
/// and the quantized arm loads a sealed artifact carrying its own
/// `tokenizer.json`.
///
/// A silent divergence there does not crash and does not look wrong: it
/// produces two plausible perplexities of two different token streams, and
/// their ratio is meaningless. Printing this on the result line turns an
/// assumption into something a reader checks by comparing two lines.
pub fn token_fingerprint(ids: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &id in ids {
        for b in id.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spellings_of_the_same_dtype_agree() {
        for s in ["f16", "FP16", " half "] {
            assert_eq!(parse_dtype(s).unwrap(), DType::F16, "{s}");
        }
        assert_eq!(parse_dtype("f32").unwrap(), DType::F32);
        assert_eq!(parse_dtype("bf16").unwrap(), DType::BF16);
    }

    /// The whole point: a misspelled override must stop the run, not quietly
    /// score in whatever the binary's default happened to be.
    #[test]
    fn an_unknown_name_is_refused() {
        assert!(parse_dtype("f8").is_err());
        assert!(parse_dtype("").is_err());
    }

    /// Names are printed next to published numbers, so they have to round-trip
    /// through the parser rather than being decorative strings.
    #[test]
    fn printed_names_parse_back() {
        for d in [DType::F16, DType::BF16, DType::F32] {
            assert_eq!(parse_dtype(&dtype_name(d)).unwrap(), d);
        }
    }

    /// At F32 the narrowing must be the identity, or every existing artifact
    /// proof silently changes meaning.
    #[test]
    fn narrowing_is_a_no_op_at_f32() {
        for v in [0.0f32, 1.0, -3.7e-9, 1.234_567_8, f32::MIN_POSITIVE] {
            assert_eq!(narrow(v, DType::F32).to_bits(), v.to_bits(), "{v:e}");
        }
    }

    /// And it must actually round at the half precisions — a no-op there would
    /// make `verify_artifact` compare a full-precision decode against a
    /// rounded model and fail for the wrong reason.
    #[test]
    fn narrowing_rounds_at_half_precision() {
        let v = 1.234_567_8f32;
        assert_ne!(narrow(v, DType::F16).to_bits(), v.to_bits());
        assert_ne!(narrow(v, DType::BF16).to_bits(), v.to_bits());

        // Idempotent: narrowing an already-narrowed value changes nothing.
        // This is what makes the comparison in `verify_artifact` stable — the
        // model's stored weight is already at `dtype`, so narrowing the decode
        // has to land on it and stay there.
        for d in [DType::F16, DType::BF16] {
            assert_eq!(narrow(narrow(v, d), d).to_bits(), narrow(v, d).to_bits());
        }

        // bf16 keeps f32's exponent range but only 8 mantissa bits against
        // f16's 11, so it is the coarser of the two — **on average**, not on
        // every value. A number whose mantissa terminates early rounds
        // identically in both, which is exactly what a single-value assertion
        // here got wrong: 1.2345678 lands on 1.234375 in f16 *and* in bf16.
        let (mut e16, mut ebf) = (0.0f64, 0.0f64);
        for i in 1..2000 {
            let v = i as f32 * 0.001_37;
            e16 += f64::from((narrow(v, DType::F16) - v).abs());
            ebf += f64::from((narrow(v, DType::BF16) - v).abs());
        }
        assert!(ebf > 4.0 * e16, "bf16 {ebf:e} devrait dominer f16 {e16:e}");
    }

    /// The fingerprint exists to catch two arms tokenizing differently, so it
    /// has to move on every way two streams can differ: content, order, and
    /// length. A checksum blind to any one of them would report "same tokens"
    /// on a stream that is not.
    #[test]
    fn the_fingerprint_separates_streams_that_differ() {
        let base: Vec<u32> = (0..64).collect();
        assert_eq!(token_fingerprint(&base), token_fingerprint(&base.clone()));

        let mut one_token_off = base.clone();
        one_token_off[31] = 999;
        assert_ne!(token_fingerprint(&base), token_fingerprint(&one_token_off));

        // Order: a permutation is a different stream, and a sum-based digest
        // would call it identical.
        let mut swapped = base.clone();
        swapped.swap(3, 4);
        assert_ne!(token_fingerprint(&base), token_fingerprint(&swapped));

        // Length: a prefix must not collide with the whole.
        assert_ne!(token_fingerprint(&base), token_fingerprint(&base[..63]));

        // And a leading zero must count — folding over `to_le_bytes` rather
        // than over a decimal rendering is what makes that true.
        assert_ne!(token_fingerprint(&[0, 1]), token_fingerprint(&[1]));
    }
}
