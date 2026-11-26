//! `inkling::attn_bias` on Metal against the host tables of `score_bias`.
//!
//! Class A: the kernel's `[heads, T, S]` tensor must be the host path's bit
//! for bit, the `-inf` of the masks and the signed zeros of the dropped
//! relative logits included. The shapes cover a full and a sliding layer, a
//! query block at the start of a slot and one past the extent, a key count
//! that is not a multiple of the threadgroup, and the real attention shape
//! (24 heads, extent 1,024, window 1,024, 128 rows against 2,560 keys).
//!
//! Tiny and synthetic: not a heavy job.
#![cfg(all(target_os = "macos", feature = "metal"))]

use candle_core::{Device, Tensor};
use llvq_llm::inkling::attn_bias;
use llvq_llm::inkling::layers::score_bias_host;

/// xorshift64*: the same logits on every run, about half of them negative.
fn logits(n: usize, seed: u64) -> Vec<f32> {
    let mut x = seed;
    (0..n)
        .map(|_| {
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            let u = (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32 / (1u64 << 24) as f32;
            (u - 0.5) * 7.0
        })
        .collect()
}

fn bits(t: &Tensor) -> Vec<u32> {
    t.flatten_all().unwrap().to_vec1::<f32>().unwrap().iter().map(|v| v.to_bits()).collect()
}

#[test]
fn bias_kernel_is_the_host_tables_bit_for_bit() {
    let dev = Device::new_metal(0).unwrap();
    // (heads, rows, keys, extent, offset, window)
    let cases: [(usize, usize, usize, usize, usize, Option<usize>); 7] = [
        (4, 5, 9, 6, 4, None),
        (4, 5, 9, 6, 4, Some(4)),
        (3, 7, 33, 8, 26, Some(5)),
        (2, 1, 1, 4, 0, None),
        (24, 16, 300, 1024, 284, None),
        (24, 128, 2560, 1024, 2432, Some(1024)),
        (24, 128, 9000, 1024, 8872, None),
    ];
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    for (heads, t, s, extent, offset, window) in cases {
        seed = seed.wrapping_add(1);
        let rel = Tensor::from_vec(logits(heads * t * extent, seed), (heads, t, extent), &dev).unwrap();
        let host = score_bias_host(&rel, offset, s, extent, window).unwrap();
        let kern = attn_bias::bias_once(&dev, &rel, offset, s, extent, window).unwrap().expect("the kernel applies");
        assert_eq!(kern.dims(), host.dims(), "shape, case {heads}x{t}x{s}");
        let (a, b) = (bits(&host), bits(&kern));
        let bad = a.iter().zip(&b).filter(|(x, y)| x != y).count();
        assert_eq!(bad, 0, "{bad} elements differ, case heads {heads} rows {t} keys {s} offset {offset} window {window:?}");
        // The case reaches every branch it should: masked keys, dropped logits, kept ones.
        let v = host.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(v.iter().any(|x| x.is_finite() && *x != 0.0), "no kept logit in case {heads}x{t}x{s}");
        if offset + 1 < s || window.is_some_and(|w| offset >= w) {
            assert!(v.contains(&f32::NEG_INFINITY), "no masked key in case {heads}x{t}x{s}");
        }
    }
}

#[test]
fn bias_switch_routes_score_bias_to_the_kernel() {
    let dev = Device::new_metal(0).unwrap();
    attn_bias::set_on_device(Some(&dev)).unwrap();
    assert!(attn_bias::on_device());
    let rel = Tensor::from_vec(logits(4 * 5 * 6, 7), (4, 5, 6), &dev).unwrap();
    let on = attn_bias::bias(&rel, 4, 9, 6, Some(4)).unwrap().expect("on");
    attn_bias::set_on_device(None).unwrap();
    assert!(attn_bias::bias(&rel, 4, 9, 6, Some(4)).unwrap().is_none());
    assert_eq!(bits(&on), bits(&score_bias_host(&rel, 4, 9, 6, Some(4)).unwrap()));
}
