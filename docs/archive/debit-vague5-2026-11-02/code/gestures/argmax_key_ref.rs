// DRAFT, never compiled or run. Wave-5 deep dive "gestures", 2026-11-02, revised after review.
// Not part of the repository; a sketch for the operator to review.
//
// Host reference for `tok_argmax_h`: a replay of candle 0.9.2's fast_argmax
// (reduce.cu:478-523, one 1024-thread block, shr at -inf and index 0xFFFFFFFF,
// strided strict `>` scan with `not_set`, left-kept tree) against the 64-bit
// tie key, on adversarial f16 vectors. Pure host Rust, so it runs on the Mac in
// the fast `cargo test` loop. It certifies the KEY. The card check at load
// certifies the KERNEL against candle's own `fast_argmax_f16` on the device,
// with the case list below in the order tok_argmax.cu gives (rule 10: card =
// reference), and the A/B mode kills the kernel mutants m1-m8 there.
//
// What changed after review: the random test of the first draft drew every
// element from six levels, so the maximum sat at 1/8 of positions and thread 0
// held it with probability 1 - (7/8)^149 = 1 - 2.3e-9 (computed): every case
// resolved to thread 0, and the bit-reversed cross-thread order, the crux of the
// key, was never exercised. The sparse cases below put the maximum at 2 to 6
// random positions, as the wave-2 simulation did (wave2-verify-fitA/argmax_sim.py),
// over a background of four levels that keeps signed zeros in play; one dense
// case keeps in-thread ties. The signed-zero case is also swapped so that the
// -0 sits at the winning position, which is what kills m1 (computed in
// calc/gestures-check.py: the first draft's layout returned the same index with
// and without m1).
//
// Where it would live: llvq-llm/src/argmax_key.rs (no CUDA, no unsafe), with
// `half` (already a dependency of llvq-llm).

use half::f16;

pub const N: usize = 151_936; // Qwen3's vocabulary at every size

/// candle's fast_argmax, replayed on the f32 widening of f16 values (exact, so
/// every `>` gives the result `__half` gives). Mirrors the kernel line by line:
/// `not_set || v > best`, then `if (tid < s && shr[tid + s] > shr[tid])`.
pub fn candle_fast_argmax(vals: &[f32]) -> u32 {
    const B: usize = 1024;
    assert!(vals.len() >= B, "the replay is for n >= 1024 (block exactly 1024 wide)");
    let mut shr = [f32::NEG_INFINITY; B];
    let mut idx = [u32::MAX; B];
    for t in 0..B {
        let mut not_set = true;
        let mut i = t;
        while i < vals.len() {
            if not_set || vals[i] > shr[t] {
                shr[t] = vals[i];
                idx[t] = i as u32;
                not_set = false;
            }
            i += B;
        }
    }
    let mut s = B / 2;
    while s > 0 {
        for t in 0..s {
            if shr[t + s] > shr[t] {
                shr[t] = shr[t + s];
                idx[t] = idx[t + s];
            }
        }
        s /= 2;
    }
    idx[0]
}

pub fn ord16(b: u16) -> u32 {
    let b = if b == 0x8000 { 0 } else { b } as u32;
    if b & 0x8000 != 0 { !b & 0xFFFF } else { b | 0x8000 }
}

pub fn is_nan16(b: u16) -> bool {
    b & 0x7C00 == 0x7C00 && b & 0x03FF != 0
}

pub fn bitrev10(t: u32) -> u32 {
    (t & 0x3FF).reverse_bits() >> 22
}

pub fn tie_key(b: u16, i: u32) -> u64 {
    ((ord16(b) as u64) << 32) | (((1023 - bitrev10(i & 1023)) as u64) << 22) | (0x3F_FFFF - (i >> 10)) as u64
}

pub fn key_index(k: u64) -> u32 {
    let t = bitrev10(1023 - ((k >> 22) & 0x3FF) as u32);
    let j = 0x3F_FFFF - (k & 0x3F_FFFF) as u32;
    (j << 10) | t
}

/// What the kernel computes, in any order: (index, NaN seen).
pub fn key_argmax(bits: &[u16]) -> (u32, bool) {
    let mut best = 0u64;
    let mut nan = false;
    for (i, &b) in bits.iter().enumerate() {
        if is_nan16(b) {
            nan = true;
            continue;
        }
        best = best.max(tie_key(b, i as u32));
    }
    (if nan { 0 } else { key_index(best) }, nan)
}

fn h(v: f32) -> u16 {
    f16::from_f32(v).to_bits()
}

fn filled(v: f32) -> Vec<u16> {
    vec![h(v); N]
}

/// The eight non-NaN adversarial cases, IN THE ORDER OF THE CARD SEQUENCE
/// (tok_argmax.cu, "THE CARD CHECK"), with candle's expected index where the
/// test pins it. Case 1 holds the largest key of the sequence on purpose.
pub fn adversarial_cases() -> Vec<(&'static str, Vec<u16>, Option<u32>)> {
    let mut out = Vec::new();
    let mut v = filled(-1.0);
    v[N - 1] = 0x7C00;
    out.push(("+inf last", v, Some((N - 1) as u32)));
    let mut v = filled(-1.0);
    v[10] = 0x7C00;
    v[N - 1] = 0x7C00; // thread 383, bitrev 1018, loses to thread 10, bitrev 320
    out.push(("+inf tie", v, Some(10)));
    let mut v = filled(-1.0);
    v[256] = h(7.5);
    v[512] = h(7.5); // candle: 512 (bitrev 1) beats 256 (bitrev 2)
    out.push(("256 vs 512", v, Some(512)));
    let mut v = filled(-1.0);
    v[1] = h(7.5);
    v[2] = h(7.5); // 2 (bitrev 256) beats 1 (bitrev 512)
    out.push(("1 vs 2", v, Some(2)));
    let mut v = filled(-1.0);
    v[5] = h(3.0);
    v[5 + 1024 * 7] = h(3.0); // same thread: the earliest wins
    out.push(("same thread", v, Some(5)));
    let mut v = filled(-2.0);
    v[1300] = 0x8000; // -0 at thread 276, bitrev 162: the winning position
    v[300] = 0x0000; // +0 at thread 300, bitrev 210
    out.push(("-0 wins the tie", v, Some(1300)));
    out.push(("all -inf", vec![0xFC00u16; N], Some(0)));
    let mut v = vec![0x8001u16; N]; // tiny negative subnormals
    v[4096] = 0x0001; // thread 0
    v[77] = 0x0001; // thread 77
    out.push(("subnormal tie", v, Some(4096)));
    out
}

/// Sparse tie sets: a background of four levels (signed zeros included), the
/// maximum 2.25 at 2 to 6 random positions. A fixed LCG keeps them
/// reproducible; the card check uploads the first 32, the host test runs 300.
pub fn sparse_cases(count: usize) -> Vec<Vec<u16>> {
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move || {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        s >> 33
    };
    let levels = [h(-3.0), 0x8000, 0x0000, h(1.5)];
    (0..count)
        .map(|_| {
            let mut v: Vec<u16> = (0..N).map(|_| levels[(next() % 4) as usize]).collect();
            let k = 2 + (next() % 5) as usize; // 2..=6 maxima
            for _ in 0..k {
                v[(next() % N as u64) as usize] = h(2.25);
            }
            v
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn widen(bits: &[u16]) -> Vec<f32> {
        bits.iter().map(|&b| f16::from_bits(b).to_f32()).collect()
    }

    fn agree(bits: &[u16], what: &str) -> u32 {
        let (k, nan) = key_argmax(bits);
        assert!(!nan, "{what}: no NaN in this case");
        assert_eq!(k, candle_fast_argmax(&widen(bits)), "{what}");
        k
    }

    #[test]
    fn adversarial_cases_agree_with_candle_and_their_pinned_index() {
        // Kills m1 (case "-0 wins the tie"), m2 (cases "256 vs 512", "1 vs 2"),
        // m3 ("same thread") and m4 ("256 vs 512") when they are applied to the
        // Rust key; their kernel twins die on the card.
        for (what, v, want) in adversarial_cases() {
            let got = agree(&v, what);
            if let Some(w) = want {
                assert_eq!(got, w, "{what}");
            }
        }
    }

    #[test]
    fn sparse_ties_agree_fast_loop() {
        // The 32 cases the card check uploads, in the debug loop.
        for (c, v) in sparse_cases(32).iter().enumerate() {
            agree(v, &format!("sparse case {c}"));
        }
    }

    #[test]
    #[cfg_attr(debug_assertions, ignore)] // 300 x two passes over 151,936: the release loop
    fn sparse_ties_agree_300() {
        for (c, v) in sparse_cases(300).iter().enumerate() {
            agree(v, &format!("sparse case {c}"));
        }
    }

    #[test]
    fn dense_ties_keep_the_in_thread_order() {
        // The first draft's distribution, kept as one case: the maximum at 1/8
        // of positions, so thread 0 holds several and must return its earliest.
        let mut s = 0x2545_f491_4f6c_dd1du64;
        let levels = [h(-3.0), 0x8000, 0x0000, h(1.5), h(1.5), h(2.25), h(-3.0), 0x0000];
        let v: Vec<u16> = (0..N)
            .map(|_| {
                s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                levels[(s >> 61) as usize]
            })
            .collect();
        agree(&v, "dense ties");
    }

    #[test]
    fn a_nan_raises_the_flag_and_yields_an_in_bounds_token() {
        // Kills m8 when applied to the Rust key.
        let mut v = filled(0.5);
        v[1234] = 0x7E00; // quiet NaN
        let (k, nan) = key_argmax(&v);
        assert!(nan);
        assert_eq!(k, 0, "token 0 keeps the next gather in bounds");
    }

    #[test]
    fn key_index_inverts_tie_key() {
        // Kills m4 (9-bit reversal): 512 and 1023 lose bit 9 in the decode.
        for &i in &[0u32, 1, 2, 511, 512, 1023, 1024, 151_935] {
            assert_eq!(key_index(tie_key(0x3C00, i)), i);
        }
    }
}

// The card check at load, in prose (fused_cuda.rs, beside check_int4_half_parity):
//   build adversarial_cases(), then sparse_cases(32), then the NaN case, as f16
//   bits on the host; upload once; for each case in that order run candle's
//   `argmax` on the device tensor (fast_argmax_f16) and tok_argmax_h into
//   ring[pos], all through ONE slot/ticket/flag state; read the ring back once;
//   any index that differs stops the load and names the case; the NaN case
//   checks the flag and token 0 only. One line:
//   "tok_argmax: candle's fast_argmax_f16 index on 40 cases in sequence, NaN flag raised".
//   Cost: 41 vectors x 304 KB uploaded once, about 12.5 MB; a few ms of device time.
