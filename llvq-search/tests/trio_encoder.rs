//! # The Trio encoder against the word map
//!
//! What the encoder returns must be a Trio codeword and nothing else: its
//! word must be the word `Trio::encode` gives its point, its point the point
//! `Trio::decode` gives its word, in Λ₂₄, never the origin for a nonzero
//! block. Its agreement with the bench's rule, and its retention, are in
//! `llvq-bench/tests/trio_encoder.rs`, where the bench is visible.

use llvq_core::{Leech, SplitMix64, DIM};
use llvq_search::trio::encoder::t_of;
use llvq_search::trio::{Encoder, Scratch, Trio, LABEL_MASK};

/// Blocks per test: two thousand in release, two hundred in debug.
const N: usize = if cfg!(debug_assertions) { 200 } else { 2_000 };

/// The F1b seed: 4,000 training blocks skipped, then the evaluation blocks.
fn eval_blocks(n: usize) -> Vec<[f64; DIM]> {
    let mut rng = SplitMix64::new(0x0f1b_2026_0904);
    for _ in 0..4_000 {
        core::array::from_fn::<f64, DIM, _>(|_| rng.next_gaussian());
    }
    (0..n).map(|_| core::array::from_fn(|_| rng.next_gaussian())).collect()
}

/// (2) The word and the point of a code are the same object under the map:
/// `Trio::encode(point) == Some(word)`, `Trio::decode(word) == point`, gain
/// bit clear, `t` as stated.
#[test]
fn the_word_and_the_point_agree_with_the_map() {
    let trio = Trio::new();
    let enc = Encoder::new(&trio);
    let mut sc = Scratch::new();
    for x in eval_blocks(N) {
        let code = enc.encode(&x, &mut sc);
        assert_eq!(code.word >> 47, 0, "the gain bit is set");
        assert_eq!(code.word & !LABEL_MASK, 0);
        assert_eq!(trio.encode(&code.point), Some(code.word), "{:?}", code.point);
        assert_eq!(trio.decode(code.word), code.point, "{:#014x}", code.word);
        assert_eq!(code.t, t_of(&x, &code.point));
        assert!(code.t.is_finite() && code.t > 0.0, "t = {} for a Gaussian block", code.t);
    }
}

/// (3) Every point is in Λ₂₄ and none is the origin for `x ≠ 0` — at every
/// magnitude of `x`, since the scales adapt to `‖x‖`, down to blocks whose
/// squared norm underflows (1e-300) and blocks of denormals (1e-310); the
/// zero block alone is the origin, word 0, `t = −∞`. Coordinates stay within
/// the table's ±10, and at ordinary magnitudes the point is the direction's
/// alone.
#[test]
fn every_point_is_in_leech_and_never_the_origin() {
    let trio = Trio::new();
    let enc = Encoder::new(&trio);
    let leech = Leech::new();
    let mut sc = Scratch::new();
    for (i, x) in eval_blocks(N).into_iter().enumerate() {
        let scale = [1.0, 1e-6, 1e6, 1e-300, 1e-310][i % 5];
        let x: [f64; DIM] = core::array::from_fn(|j| x[j] * scale);
        let code = enc.encode(&x, &mut sc);
        assert!(leech.contains(&code.point), "block {i}: {:?} is outside Λ₂₄", code.point);
        assert_ne!(code.point, [0; DIM], "block {i} at scale {scale}: the origin");
        assert_ne!(code.word, 0);
        assert_eq!(trio.encode(&code.point), Some(code.word), "block {i} at scale {scale}");
        assert!(code.point.iter().all(|v| v.abs() <= 10), "block {i}: {:?}", code.point);
        assert!(code.t > 0.0, "block {i} at scale {scale}: t = {}", code.t);
        // The code is a function of the direction: the same block at another
        // magnitude gives the same point, as long as the targets do not lose
        // bits to the magnitude itself.
        if matches!(i % 5, 1 | 2) {
            let unit: [f64; DIM] = core::array::from_fn(|j| x[j] / scale);
            assert_eq!(enc.encode(&unit, &mut sc).point, code.point, "block {i}: the scale {scale} changed the point");
        }
    }
    let zero = enc.encode(&[0.0; DIM], &mut sc);
    assert_eq!((zero.word, zero.point, zero.t), (0, [0; DIM], f64::NEG_INFINITY));
}

/// `encode` is the better of its two scales, `ALPHA·‖x‖/√24` and `RATIO`
/// times it, by `t`, the lower scale keeping ties; both scales contribute
/// winners on Gaussian blocks. A different pair is a different encoder.
#[test]
fn encode_is_the_better_of_its_two_scales() {
    let trio = Trio::new();
    let enc = Encoder::new(&trio);
    let mut sc = Scratch::new();
    let mut wins = [0usize; 2];
    for x in eval_blocks(N) {
        let (_, s0) = Encoder::lower_scale(&x);
        assert!((s0 - Encoder::ALPHA * x.iter().map(|v| v * v).sum::<f64>().sqrt() / (DIM as f64).sqrt()).abs() < 1e-12);
        let a = enc.encode_at_scale(&x, s0, &mut sc);
        let b = enc.encode_at_scale(&x, s0 * Encoder::RATIO, &mut sc);
        let want = if b.t > a.t { b } else { a };
        wins[usize::from(b.t > a.t)] += 1;
        assert_eq!(enc.encode(&x, &mut sc), want);
    }
    assert!(wins[0] > N / 10 && wins[1] > N / 10, "the two scales win {wins:?} of {N} blocks: one of them is dead weight");
}

/// The pinned pair sits where the scale study looked: `α` in the band the
/// 18-point grid covers at the useful scales, the ratio one to two grid steps.
const _: () = assert!(Encoder::ALPHA > 0.2 && Encoder::ALPHA < 0.5 && Encoder::RATIO > 1.0 && Encoder::RATIO < 1.5);

/// The encoder is shared across threads (`Sync`) with one `Scratch` each,
/// and it is a pure function of its input: the same block encodes to the
/// same code on a reused scratch and on a fresh one.
#[test]
fn the_encoder_is_shared_and_deterministic() {
    fn assert_sync<T: Sync + Send>() {}
    assert_sync::<Encoder>();
    let trio = Trio::new();
    let enc = Encoder::new(&trio);
    let blocks = eval_blocks(N.min(400));
    let mut sc = Scratch::new();
    let first: Vec<_> = blocks.iter().map(|x| enc.encode(x, &mut sc)).collect();
    let again: Vec<_> = blocks.iter().map(|x| enc.encode(x, &mut Scratch::new())).collect();
    assert_eq!(first, again);
    std::thread::scope(|s| {
        for chunk in blocks.chunks(blocks.len().div_ceil(4)).zip(first.chunks(blocks.len().div_ceil(4))) {
            let enc = &enc;
            s.spawn(move || {
                let mut sc = Scratch::new();
                for (x, want) in chunk.0.iter().zip(chunk.1) {
                    assert_eq!(&enc.encode(x, &mut sc), want);
                }
            });
        }
    });
}

/// The truncated rule stays a Trio encoder: on blocks built so that two
/// coordinates of a section carry all its energy, and at a scale a hundred
/// times too small — where three shrinks leave every middle pattern without
/// a member and only the full rule's zero shrink finds one — every code is
/// still a Trio codeword and never the origin. (Its `t` is not ordered
/// against the full rule's: a farther point in the scaled metric can align
/// better.)
#[test]
fn the_truncated_rule_stays_feasible() {
    let trio = Trio::new();
    let (full, cut) = (Encoder::new(&trio), Encoder::with_shrinks(&trio, 3));
    let (leech, order) = (Leech::new(), *trio.order());
    let mut rng = SplitMix64::new(0x7210_2026_0905_0011);
    let mut sc = Scratch::new();
    for i in 0..N {
        // In trio order: two big coordinates per section, the rest small.
        let mut raw = [0.0f64; DIM];
        for k in 0..3 {
            let (a, b) = ((rng.next() % 8) as usize, ((rng.next() % 7) as usize + 1));
            raw[8 * k + a] = 3.0 + rng.next_gaussian().abs();
            raw[8 * k + (a + b) % 8] = 3.0 + rng.next_gaussian().abs();
            for j in 0..8 {
                raw[8 * k + j] += 0.05 * rng.next_gaussian();
            }
        }
        let mut x = [0.0f64; DIM];
        for (j, &v) in raw.iter().enumerate() {
            x[order[j] as usize] = v;
        }
        let tiny = Encoder::lower_scale(&x).1 / 100.0;
        for code in [full.encode(&x, &mut sc), cut.encode(&x, &mut sc), cut.encode_at_scale(&x, tiny, &mut sc), full.encode_at_scale(&x, tiny, &mut sc)] {
            assert!(leech.contains(&code.point), "block {i}: outside Λ₂₄");
            assert_ne!(code.point, [0; DIM], "block {i}: the origin");
            assert_eq!(trio.encode(&code.point), Some(code.word), "block {i}");
        }
    }
}
