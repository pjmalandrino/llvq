//! What the attention replay must be on its own, before any kernel is held
//! against it: an attention (it agrees with an f64 softmax), row invariant by
//! construction, absolute in its splits, and built on exact conversions and an
//! accurate exp2.
//!
//! Its agreement with the kernel text is `attn_dec_matches_rust.rs`; its
//! agreement with the card is the load-time check of `llvq_cuda::attn`.

use llvq_cuda::attn_ref::{self, ex2_fixed, f2h, h2f, st_merge, Kv, St, HD, SPLIT};

/// The f64 attention of one row over `0..l`, and the scale its error is
/// measured in: sum p |v| / sum p, per dim.
fn reference(q: &[u16], kv: &Kv, h: usize, l: usize) -> (Vec<f64>, Vec<f64>) {
    let qf: Vec<f64> = q.iter().map(|&x| h2f(x) as f64).collect();
    let scale = 1.0 / (HD as f64).sqrt();
    let s: Vec<f64> = (0..l)
        .map(|p| {
            let o = (h * kv.cap + p) * HD;
            (0..HD).map(|i| qf[i] * h2f(kv.k[o + i]) as f64).sum::<f64>() * scale
        })
        .collect();
    let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let p: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
    let z: f64 = p.iter().sum();
    let mut out = vec![0f64; HD];
    let mut mag = vec![0f64; HD];
    for (j, pj) in p.iter().enumerate() {
        let o = (h * kv.cap + j) * HD;
        for i in 0..HD {
            let v = h2f(kv.v[o + i]) as f64;
            out[i] += pj * v;
            mag[i] += pj * v.abs();
        }
    }
    (out.iter().map(|x| x / z).collect(), mag.iter().map(|x| x / z).collect())
}

/// The replay's f32 result before the f16 rounding, by its public pieces.
fn replay_f32(q: &[u16], kv: &Kv, h: usize, l: usize, unit: usize) -> Vec<f32> {
    let mut qf = [0f32; HD];
    for i in 0..HD {
        qf[i] = h2f(q[i]);
    }
    let c = attn_ref::score_scale(HD);
    let mut total = St::empty();
    for s in 0..l.div_ceil(SPLIT) {
        st_merge(&mut total, &attn_ref::split_state(&qf, kv, h, s, l, c, unit));
    }
    total.a.iter().map(|a| a / total.d).collect()
}

/// An f32 online softmax with a 1-ulp exp2 agrees with f64 to a few parts in
/// a million of the row's magnitude, at every length the kernel meets, and
/// its f16 output is the f16 rounding of that.
#[test]
fn the_replay_is_an_attention() {
    let cap = 4352;
    let kc = attn_ref::fill(cap * HD, 1, 1.0);
    let vc = attn_ref::fill(cap * HD, 2, 1.0);
    let kv = Kv { k: &kc, v: &vc, cap };
    let c = attn_ref::score_scale(HD);
    let mut worst = 0f64;
    for (i, &l) in [1usize, 7, 8, 9, 63, 64, 65, 255, 256, 257, 1000, 4096].iter().enumerate() {
        // A sharp query (scores of several units) and a flat one.
        for amp in [0.5f32, 6.0] {
            let q = attn_ref::fill(HD, 100 + i as u64, amp);
            let (want, mag) = reference(&q, &kv, 0, l);
            for unit in [64, 32] {
                let got = replay_f32(&q, &kv, 0, l, unit);
                for d in 0..HD {
                    let e = (got[d] as f64 - want[d]).abs() / mag[d];
                    worst = worst.max(e);
                    assert!(e < 4e-6, "L={l} amp={amp} unit={unit} dim {d}: {} vs {} (rel {e:e})", got[d], want[d]);
                }
                let mut qf = [0f32; HD];
                for d in 0..HD {
                    qf[d] = h2f(q[d]);
                }
                let o = attn_ref::row(&qf, &kv, 0, l, c, unit);
                for d in 0..HD {
                    assert_eq!(o[d], f2h(got[d]), "L={l} unit={unit} dim {d}: output is not the f16 of the state");
                }
            }
        }
    }
    eprintln!("worst relative error against f64: {worst:e}");
}

/// The rows of one call at pos0..pos0+k equal k calls of one row: true by
/// construction in the replay, stated so a refactor that broke it fails here
/// before it fails against the kernel.
#[test]
fn rows_are_invariant() {
    let (n_kv, g, cap) = (2, 4, 4352);
    let kc = attn_ref::fill(n_kv * cap * HD, 3, 1.0);
    let vc = attn_ref::fill(n_kv * cap * HD, 4, 1.0);
    let kv = Kv { k: &kc, v: &vc, cap };
    let c = attn_ref::score_scale(HD);
    for &pos0 in &[0usize, 6, 7, 8, 62, 63, 64, 248, 254, 255, 256, 999, 4088] {
        for k in 1..=8 {
            let q = attn_ref::fill(n_kv * g * k * HD, pos0 as u64 * 31 + k as u64, 4.0);
            let all = attn_ref::attend(&q, &kv, n_kv, g, pos0, k, c, 64);
            for r in 0..k {
                let mut q1 = Vec::new();
                for hq in 0..n_kv * g {
                    q1.extend_from_slice(&q[(hq * k + r) * HD..(hq * k + r + 1) * HD]);
                }
                let one = attn_ref::attend(&q1, &kv, n_kv, g, pos0 + r, 1, c, 64);
                for hq in 0..n_kv * g {
                    assert_eq!(
                        &all[(hq * k + r) * HD..(hq * k + r + 1) * HD],
                        &one[hq * HD..(hq + 1) * HD],
                        "pos0={pos0} k={k} row {r} head {hq}"
                    );
                }
            }
        }
    }
}

/// Splits sit at absolute multiples of 256: a row of 300 positions folds
/// [0, 256) then [256, 300), whatever its length. Stated through the public
/// pieces, so a split taken relative to the row's end fails here.
#[test]
fn splits_are_absolute() {
    let cap = 512;
    let kc = attn_ref::fill(cap * HD, 5, 1.0);
    let vc = attn_ref::fill(cap * HD, 6, 1.0);
    let kv = Kv { k: &kc, v: &vc, cap };
    let c = attn_ref::score_scale(HD);
    let q = attn_ref::fill(HD, 7, 4.0);
    let mut qf = [0f32; HD];
    for i in 0..HD {
        qf[i] = h2f(q[i]);
    }
    for l in [257usize, 300, 511] {
        let mut total = St::empty();
        st_merge(&mut total, &attn_ref::split_state(&qf, &kv, 0, 0, l, c, 64));
        st_merge(&mut total, &attn_ref::split_state(&qf, &kv, 0, 1, l, c, 64));
        let want: Vec<u16> = total.a.iter().map(|a| f2h(a / total.d)).collect();
        assert_eq!(attn_ref::row(&qf, &kv, 0, l, c, 64).to_vec(), want, "L={l}");
    }
}

/// Every binary16 widens exactly.
#[test]
fn h2f_is_exact_on_every_half() {
    for h in 0..=u16::MAX {
        let want = llvq_cuda::f16_to_f64(h);
        let got = h2f(h);
        if want.is_nan() {
            assert!(got.is_nan(), "{h:#06x}");
        } else {
            assert_eq!(got as f64, want, "{h:#06x}");
            assert_eq!(got.is_sign_negative(), want.is_sign_negative(), "{h:#06x} sign");
        }
    }
}

/// The narrowing is the crate's own `f16_bits`, round to nearest even, on
/// every half, every midpoint between halves and its two neighbours.
#[test]
fn f2h_rounds_to_nearest_even() {
    let mut xs = Vec::new();
    for h in 0..0x7c00u16 {
        for sign in [0u16, 0x8000] {
            let a = h2f(h | sign);
            xs.push(a);
            let b = h2f((h + 1) | sign);
            let mid = ((a as f64 + b as f64) / 2.0) as f32;
            for x in [mid, f32::from_bits(mid.to_bits() + 1), f32::from_bits(mid.to_bits().saturating_sub(1))] {
                xs.push(x);
            }
        }
    }
    xs.extend([f32::INFINITY, f32::NEG_INFINITY, 65520.0, 65519.99, 1e-8, -1e-8, 2.9802322e-8, 5.9604645e-8]);
    for x in xs {
        assert_eq!(f2h(x), llvq_cuda::f16_bits(x), "{x:e}");
    }
}

/// Within 1.25 ulp of 2^x over the domain the kernel meets, exact at the
/// points that matter, and never subnormal.
#[test]
fn ex2_is_accurate_and_exact_where_it_must_be() {
    assert_eq!(ex2_fixed(0.0).to_bits(), 1.0f32.to_bits());
    assert_eq!(ex2_fixed(-0.0).to_bits(), 1.0f32.to_bits());
    assert_eq!(ex2_fixed(-1.0), 0.5);
    assert_eq!(ex2_fixed(-125.0), f32::from_bits(2 << 23));
    assert_eq!(ex2_fixed(-126.0).to_bits(), 0);
    assert_eq!(ex2_fixed(f32::NEG_INFINITY).to_bits(), 0);
    assert_eq!(ex2_fixed(-1000.0).to_bits(), 0);
    assert!(ex2_fixed(f32::NAN).is_nan());
    let mut worst = 0f64;
    let mut s = 0xabcdefu64;
    for i in 0..400_000 {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let u = (s >> 40) as f64 / (1u64 << 24) as f64;
        let x = (if i % 2 == 0 { -125.9 * u } else { -2.0 * u }) as f32;
        let y = ex2_fixed(x);
        assert!(y.is_normal(), "ex2({x}) = {y} is not normal");
        let want = (x as f64).exp2();
        let ulp = (f32::from_bits(y.to_bits() + 1) - y) as f64;
        worst = worst.max((y as f64 - want).abs() / ulp);
    }
    eprintln!("ex2_fixed worst error: {worst:.3} ulp");
    // 1.18 ulp measured (2026-11-04): 0.92 from the polynomial on [0, 1),
    // the rest where 2^x sits just under a power of two and the ulp halves.
    assert!(worst < 1.25, "ex2_fixed is {worst} ulp away from 2^x");
}

/// The scale the host hands the kernel: (1 / sqrt(128)) x log2(e), one
/// rounding.
#[test]
fn the_score_scale_is_rounded_once() {
    let c = attn_ref::score_scale(128);
    assert_eq!(c, (std::f64::consts::LOG2_E / 128f64.sqrt()) as f32);
    assert_eq!(c.to_bits(), 0x3e02_93ee);
}
