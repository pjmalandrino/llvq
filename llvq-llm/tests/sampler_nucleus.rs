//! `Sampler::pick` against the sampler it replaced, token for token.
//!
//! `pick` selects the heaviest weights instead of sorting the whole row when
//! it can show the nucleus is the same vector, and sorts the row otherwise.
//! `pick_reference` below is `Sampler::pick` as it stood at `925930d`, copied
//! verbatim but for `self`: the whole row sorted, the nucleus cut, the draw.
//! Both samplers run from the same seed on the same rows, and the token and
//! the generator state must agree after every draw.
//!
//! The rows cover what the shortcut takes and what it refuses: peaked rows,
//! a wide head, flat rows, rows rounded to an f16 mantissa (ties), a NaN,
//! both infinities, a constant row, and degenerate temperatures and `top_p`.

use candle_core::{Device, Tensor, D};
use llvq_llm::repl::Sampler;
use std::time::Instant;

/// SplitMix64, stepped exactly as `Sampler` steps its state.
fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// `Sampler::pick` at `925930d`, verbatim but for `self`.
fn pick_reference(s: &mut Sampler, logits: &Tensor) -> u32 {
    if s.temp <= 0.0 {
        return logits
            .argmax(D::Minus1)
            .expect("an argmax")
            .to_scalar::<u32>()
            .expect("a scalar");
    }
    let v = logits.to_vec1::<f32>().expect("a row");
    let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut p: Vec<(u32, f32)> = v
        .iter()
        .enumerate()
        .map(|(i, &x)| (i as u32, ((x - max) / s.temp).exp()))
        .collect();
    p.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

    let total: f32 = p.iter().map(|x| x.1).sum();
    let mut acc = 0.0f32;
    let mut keep = p.len();
    for (i, (_, w)) in p.iter().enumerate() {
        acc += w / total;
        if acc >= s.top_p {
            keep = i + 1;
            break;
        }
    }
    p.truncate(keep.max(1));

    let mass: f32 = p.iter().map(|x| x.1).sum();
    let r = (splitmix(&mut s.state) >> 11) as f32 / (1u64 << 53) as f32 * mass;
    let mut acc = 0.0f32;
    for (id, w) in &p {
        acc += w;
        if acc >= r {
            return *id;
        }
    }
    p.last().expect("the nucleus keeps at least one token").0
}

/// Uniform in `[0, 1)`.
fn uniform(state: &mut u64) -> f32 {
    (splitmix(state) >> 40) as f32 / (1u64 << 24) as f32
}

/// `heads` tokens at `level` and above, over a bulk spread on `[-bulk, bulk]`.
fn row(seed: u64, n: usize, heads: usize, level: f32, bulk: f32) -> Vec<f32> {
    let mut st = seed;
    let mut v: Vec<f32> = (0..n).map(|_| bulk * (2.0 * uniform(&mut st) - 1.0)).collect();
    for h in 0..heads {
        let at = (uniform(&mut st) * n as f32) as usize % n;
        v[at] = level + 8.0 * uniform(&mut st) + h as f32 * 0.25;
    }
    v
}

/// The value with its mantissa cut to the ten bits of an f16, so that equal
/// logits, and equal weights, are common.
fn f16_mantissa(x: f32) -> f32 {
    f32::from_bits(x.to_bits() & 0xFFFF_E000)
}

fn rows(seed: u64, n: usize) -> Vec<(String, Vec<f32>)> {
    let one = row(seed, n, 1, 24.0, 4.0);
    let six = row(seed, n, 6, 24.0, 4.0);
    let wide = row(seed, n, 90, 12.0, 4.0);
    let mut out = vec![
        ("one head".to_string(), one.clone()),
        ("six heads".to_string(), six.clone()),
        ("ninety heads".to_string(), wide.clone()),
        ("flat".to_string(), row(seed, n, 0, 0.0, 1.0)),
        ("six heads, f16 mantissa".to_string(), six.iter().map(|&x| f16_mantissa(x)).collect()),
        ("ninety heads, f16 mantissa".to_string(), wide.iter().map(|&x| f16_mantissa(x)).collect()),
        ("constant".to_string(), vec![0.5; n]),
        ("all -inf".to_string(), vec![f32::NEG_INFINITY; n]),
    ];
    for (name, bad) in [("a NaN", f32::NAN), ("+inf", f32::INFINITY), ("-inf", f32::NEG_INFINITY)] {
        let mut v = one.clone();
        v[n / 3] = bad;
        out.push((name.to_string(), v));
    }
    let mut two = one;
    two[1] = 40.0;
    two[n - 1] = 40.0;
    out.push(("two equal maxima".to_string(), two));
    out
}

/// `draws` successive draws from `seed` on both samplers.
fn agree(v: &[f32], temp: f32, top_p: f32, seed: u64, draws: usize, what: &str) {
    let t = Tensor::from_slice(v, v.len(), &Device::Cpu).expect("a row");
    let mut new = Sampler { temp, top_p, state: seed };
    let mut old = Sampler { temp, top_p, state: seed };
    for d in 0..draws {
        let a = new.pick(&t).expect("a token");
        let b = pick_reference(&mut old, &t);
        assert_eq!(a, b, "{what}: temp {temp}, top_p {top_p}, seed {seed}, draw {d}");
        assert_eq!(new.state, old.state, "{what}: the generator drifted at draw {d}");
    }
}

#[test]
fn pick_draws_the_reference_token_on_every_row() {
    for n in [300usize, 4096] {
        for seed in 0..2u64 {
            for (what, v) in rows(seed, n) {
                for temp in [0.05f32, 0.7, 1.0, 2.0] {
                    for top_p in [0.0f32, 0.5, 0.9, 0.99, 1.0] {
                        agree(&v, temp, top_p, 0xC0FFEE ^ seed, 4, &format!("{what}, n {n}"));
                    }
                }
            }
        }
    }
}

#[test]
fn pick_draws_the_reference_token_at_degenerate_settings() {
    let v = row(7, 4096, 6, 24.0, 4.0);
    let temps = [0.0f32, -1.0, f32::MIN_POSITIVE, 1e-30, 1e30, f32::INFINITY, f32::NAN];
    for temp in temps {
        for top_p in [f32::NAN, -1.0, 0.0, 0.9, 1.5] {
            agree(&v, temp, top_p, 11, 4, "six heads");
        }
    }
}

/// Rows long enough that the selection gives up on its cap of 1,024 weights
/// while a quarter of the row would still allow more: the flat rows, the
/// ties, and 1,500 distinct heads over a masked bulk, whose total settles
/// only past the cap. Each falls back to the sort, which must be the
/// reference's own.
#[test]
fn pick_draws_the_reference_token_where_the_selection_gives_up() {
    let n = 16_384;
    let mut out = rows(1, n);
    let mut plateau = vec![f32::NEG_INFINITY; n];
    for (i, x) in plateau.iter_mut().take(1_500).enumerate() {
        *x = -(i as f32) * 1e-3;
    }
    out.push(("1,500 heads over -inf".to_string(), plateau));
    for (what, v) in out {
        for temp in [0.7f32, 1.0] {
            for top_p in [0.5f32, 0.9, 0.99] {
                agree(&v, temp, top_p, 0xFA11 ^ n as u64, 4, &format!("{what}, n {n}"));
            }
        }
    }
}

/// The vocabulary of Qwen3, where the shortcut is meant to pay.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn pick_draws_the_reference_token_at_vocabulary_size() {
    for seed in 0..4u64 {
        for (what, v) in rows(seed, 151_936) {
            for temp in [0.7f32, 1.0] {
                for top_p in [0.9f32, 0.95] {
                    agree(&v, temp, top_p, 0xBEEF ^ seed, 16, &what);
                }
            }
        }
    }
}

/// Times `pick` against `pick_reference` on three rows of the Qwen3
/// vocabulary, interleaved round by round: a peaked row, a flat row, and the
/// peaked row with its logits cut to an f16 mantissa, so equal weights are
/// common. The flat row never settles: it pays every selection, then the
/// sort. Prints the median time per draw over the rounds and the range the
/// rounds formed. An instrument, run by hand: `--ignored --nocapture`, in
/// release.
#[test]
#[ignore]
fn pick_times_against_the_sorted_reference() {
    const ROUNDS: usize = 9;
    const DRAWS: usize = 64;
    let arms = [
        ("six heads", row(3, 151_936, 6, 24.0, 4.0)),
        ("flat", row(3, 151_936, 0, 0.0, 1.0)),
        ("six heads, f16 mantissa", row(3, 151_936, 6, 24.0, 4.0).iter().map(|&x| f16_mantissa(x)).collect()),
    ];
    for (what, v) in arms {
        let t = Tensor::from_slice(v.as_slice(), v.len(), &Device::Cpu).expect("a row");
        let (mut old_us, mut new_us) = (Vec::new(), Vec::new());
        for round in 0..ROUNDS {
            let seed = 0x5EED ^ round as u64;
            let mut old = Sampler { temp: 0.7, top_p: 0.9, state: seed };
            let mut new = Sampler { temp: 0.7, top_p: 0.9, state: seed };
            let start = Instant::now();
            let a: Vec<u32> = (0..DRAWS).map(|_| pick_reference(&mut old, &t)).collect();
            old_us.push(start.elapsed().as_secs_f64() * 1e6 / DRAWS as f64);
            let start = Instant::now();
            let b: Vec<u32> = (0..DRAWS).map(|_| new.pick(&t).expect("a token")).collect();
            new_us.push(start.elapsed().as_secs_f64() * 1e6 / DRAWS as f64);
            assert_eq!(a, b, "{what}: round {round}");
        }
        for (arm, us) in [("pick_reference", &mut old_us), ("pick", &mut new_us)] {
            us.sort_by(f64::total_cmp);
            println!(
                "{what}, temp 0.7, top_p 0.9, n 151936: {arm} median {:.1} us per draw, \
                 range {:.1}..{:.1} over {ROUNDS} rounds of {DRAWS}",
                us[ROUNDS / 2],
                us[0],
                us[ROUNDS - 1]
            );
        }
    }
}
