//! The CPU fake of the decode-attention kernel, against the canonical replay
//! the card is checked against, and the calls it refuses.
//!
//! `llvq_llm::attn::canon` is `llvq-cuda/src/attn_ref.rs` by `include!`; the
//! first test holds it against `llvq_cuda::attn_ref`, the same file compiled
//! in its own crate, on the same inputs, so the two copies cannot be two
//! arithmetics.

// `llvq-cuda` is a dev-dependency on macOS only (see `llvq-llm/Cargo.toml`): on a
// Linux CPU build it would pull `cudarc`, whose build script needs `nvcc`.
// Under `--features cuda` it is a normal dependency, so the file runs there.
#![cfg(any(target_os = "macos", feature = "cuda"))]

use candle_core::{DType, Device, Tensor};
use half::f16;
use llvq_llm::attn::{canon, CpuAttn};
use llvq_llm::device::AttnKernel;

fn halves(bits: &[u16], shape: (usize, usize, usize, usize)) -> Tensor {
    let v: Vec<f16> = bits.iter().map(|&b| f16::from_bits(b)).collect();
    Tensor::from_vec(v, shape, &Device::Cpu).unwrap()
}

fn bits(t: &Tensor) -> Vec<u16> {
    t.flatten_all().unwrap().to_vec1::<f16>().unwrap().into_iter().map(f16::to_bits).collect()
}

/// A `Grow`-like cache: a buffer of `cap` positions, viewed over `len`.
fn cache(n_kv: usize, cap: usize, len: usize, seed: u64) -> (Vec<u16>, Tensor) {
    let b = canon::fill(n_kv * cap * canon::HD, seed, 1.0);
    let view = halves(&b, (1, n_kv, cap, canon::HD)).narrow(2, 0, len).unwrap();
    (b, view)
}

#[test]
fn the_cpu_fake_is_the_replay_and_the_replay_is_one_text() {
    let k = CpuAttn::new(canon::UNIT_SERVED).unwrap();
    let c = canon::score_scale(canon::HD);
    for &(n_kv, g, len, l) in &[(2usize, 4usize, 1usize, 1usize), (2, 4, 300, 1), (2, 5, 300, 8), (1, 4, 513, 3)] {
        let cap = len.div_ceil(256) * 256;
        let (kb, kt) = cache(n_kv, cap, len, 1 + len as u64);
        let (vb, vt) = cache(n_kv, cap, len, 2 + len as u64);
        let qb = canon::fill(n_kv * g * l * canon::HD, 3 + l as u64, 4.0);
        let q = halves(&qb, (1, n_kv * g, l, canon::HD));
        let got = bits(&k.attend(&q, &kt, &vt).unwrap());
        let kv = canon::Kv { k: &kb, v: &vb, cap };
        let want = canon::attend(&qb, &kv, n_kv, g, len - l, l, c, 64);
        assert_eq!(got, want, "n_kv={n_kv} G={g} len={len} rows={l}: the fake is not the replay");
        let kv2 = llvq_cuda::attn_ref::Kv { k: &kb, v: &vb, cap };
        let other = llvq_cuda::attn_ref::attend(&qb, &kv2, n_kv, g, len - l, l, c, 64);
        assert_eq!(want, other, "the included copy and llvq-cuda's own differ");
    }
}

#[test]
fn the_f16_conversions_are_the_half_crates() {
    for h in 0..=u16::MAX {
        let a = canon::h2f(h);
        let b = f16::from_bits(h).to_f32();
        assert!(a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()), "{h:#06x}");
    }
    let mut s = 0x1234_5678_9abc_def0u64;
    for _ in 0..2_000_000 {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let x = f32::from_bits((s >> 32) as u32);
        if x.is_nan() {
            continue;
        }
        assert_eq!(canon::f2h(x), f16::from_f32(x).to_bits(), "{x:e}");
    }
}

#[test]
fn the_kernel_refuses_what_it_does_not_compute() {
    let k = CpuAttn::new(64).unwrap();
    let (_, kt) = cache(2, 256, 10, 1);
    let (_, vt) = cache(2, 256, 10, 2);
    let q = |n_heads: usize, l: usize| halves(&canon::fill(n_heads * l * canon::HD, 3, 1.0), (1, n_heads, l, canon::HD));
    let err = |r: candle_core::Result<Tensor>| r.expect_err("refused").to_string();
    assert!(err(k.attend(&q(8, 1).to_dtype(DType::F32).unwrap(), &kt, &vt)).contains("f16"));
    assert!(err(k.attend(&q(8, 9), &kt, &vt)).contains("rows"));
    assert!(err(k.attend(&q(7, 1), &kt, &vt)).contains("query heads"));
    let small = Tensor::zeros((1, 8, 1, 64), DType::F16, &Device::Cpu).unwrap();
    let smallkv = Tensor::zeros((1, 2, 10, 64), DType::F16, &Device::Cpu).unwrap();
    assert!(err(k.attend(&small, &smallkv, &smallkv)).contains("head dim"));
    let q2 = Tensor::zeros((2, 8, 1, canon::HD), DType::F16, &Device::Cpu).unwrap();
    let kv2 = Tensor::zeros((2, 2, 10, canon::HD), DType::F16, &Device::Cpu).unwrap();
    assert!(err(k.attend(&q2, &kv2, &kv2)).contains("batch"));
    assert!(CpuAttn::new(48).is_err(), "a unit the kernel does not compile");
    assert!(llvq_llm::attn::kernel_for(&Device::Cpu, 2, 4, 64).is_err());
}
