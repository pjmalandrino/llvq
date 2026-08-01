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

use candle_core::DType;

/// Environment override for a harness's model dtype: `f16`, `bf16` or `f32`.
pub const DTYPE_ENV: &str = "LLVQ_DTYPE";

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
}
