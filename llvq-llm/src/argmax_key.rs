//! The greedy token by a 64-bit key that reproduces candle's tie order.
//!
//! Candle 0.9.2 picks the greedy token with `fast_argmax` (`reduce.cu:478-523`):
//! one block of 1,024 threads for any row of at least 1,024 logits. Thread `t`
//! scans `i = t, t + 1024, …` with a strict `>` from −inf, so inside a thread
//! the EARLIEST index of the maximum wins. The tree `if (tid < s && shr[tid +
//! s] > shr[tid])`, `s = 512 … 1`, keeps the LEFT operand on ties; bit 9 of
//! the thread is decided first and bit 0 last, so among tied threads the one
//! with the smallest `bitrev10(t)` wins (512 beats 256, 2 beats 1).
//!
//! `tok_argmax_h` (`kernels/tok_argmax.cu`) encodes that order in a key,
//!
//! ```text
//! ord16(v) << 32 | (1023 − bitrev10(i & 1023)) << 22 | (0x3FFFFF − (i >> 10))
//! ```
//!
//! with −0 mapped to +0 first (they tie under `>`). Keys are unique per index,
//! so a max over them is order-free and any grid returns candle's index. A NaN
//! is the one input no commutative key can follow (candle's answer then
//! depends on where the NaN sits in the tree): the kernel raises a sticky flag
//! and writes token 0, and the loop refuses the generation.
//!
//! This module is the host side, with no CUDA and no `unsafe`: the replay of
//! candle's kernel, the key, the case sequence the card check runs, and the
//! card mutants. It certifies the KEY on every machine; the card check of
//! `fused_cuda::KeyArgmax` certifies the KERNEL against candle's own
//! `fast_argmax_f16` on the device, with the case list below.
//!
//! Exactness of the f16 argmax (gesture a1): candle's `fast_argmax_f16` and
//! `fast_argmax_f32` are one template, and widening f16 to f32 is exact and
//! monotone, so every `>` gives the same answer on the f16 logits as on their
//! f32 cast. Signed zeros tie in both, a NaN compares false in both.

use half::f16;

/// Qwen3's vocabulary, at every size: the row the key kernel reads.
pub const N: usize = 151_936;

/// The key kernel's source, embedded like `emb_q8.cu`. Its own translation
/// unit and NVRTC module: nothing of the served unit moves when it is loaded.
pub const TOK_ARGMAX_CU: &str = include_str!("../kernels/tok_argmax.cu");

/// Threads a block of `tok_argmax_h`.
pub const KEY_THREADS: u32 = 256;

/// The grid of `tok_argmax_h` for `n` logits: one 16-byte vector a thread,
/// 75 blocks at 151,936 (one wave on 142 SMs).
pub fn key_grid(n: usize) -> u32 {
    (n / 8).div_ceil(KEY_THREADS as usize) as u32
}

/// The key's preconditions, checked on the host before every launch: candle's
/// block is exactly 1,024 wide from 1,024 logits on, the row index fits the
/// key's 22-bit field, and the kernel reads aligned 16-byte vectors.
pub fn key_preconditions(n: usize, offset: usize) -> Result<(), String> {
    if n < 1024 {
        return Err(format!("{n} logits: the key reproduces candle's 1,024-wide block, which needs at least 1,024"));
    }
    if n >= 1024 << 22 {
        return Err(format!("{n} logits: past the key's 22-bit row field"));
    }
    if !n.is_multiple_of(8) || !offset.is_multiple_of(8) {
        return Err(format!(
            "{n} logits at offset {offset}: the kernel reads 16-byte vectors, so both must be multiples of 8"
        ));
    }
    Ok(())
}

/// candle's `fast_argmax` replayed on the f32 widening of f16 values (exact,
/// so every `>` gives the answer `__half` gives). Line for line: `not_set ||
/// v > best` in the scan, then `if (tid < s && shr[tid + s] > shr[tid])`.
pub fn candle_fast_argmax(vals: &[f32]) -> u32 {
    const B: usize = 1024;
    assert!(vals.len() >= B, "the replay is for n >= 1024, where the block is exactly 1024 wide");
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

/// The order of an f16 as an unsigned integer, −0 mapped to +0 first.
pub fn ord16(b: u16) -> u32 {
    let b = if b == 0x8000 { 0 } else { b } as u32;
    if b & 0x8000 != 0 {
        !b & 0xFFFF
    } else {
        b | 0x8000
    }
}

pub fn is_nan16(b: u16) -> bool {
    b & 0x7C00 == 0x7C00 && b & 0x03FF != 0
}

/// The low ten bits of `t`, reversed.
pub fn bitrev10(t: u32) -> u32 {
    (t & 0x3FF).reverse_bits() >> 22
}

pub fn tie_key(b: u16, i: u32) -> u64 {
    ((ord16(b) as u64) << 32) | (((1023 - bitrev10(i & 1023)) as u64) << 22) | (0x3F_FFFF - (i >> 10)) as u64
}

/// The index a key encodes. `bitrev10` is an involution.
pub fn key_index(k: u64) -> u32 {
    let t = bitrev10(1023 - ((k >> 22) & 0x3FF) as u32);
    let j = 0x3F_FFFF - (k & 0x3F_FFFF) as u32;
    (j << 10) | t
}

/// What the kernel computes, in any order: the index, and whether a NaN was
/// met (then the index is 0, an in-bounds token the caller refuses).
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

/// One case of the card sequence: f16 bits, the index candle must return when
/// the test pins one, and whether it holds a NaN.
pub struct KeyCase {
    pub name: String,
    pub bits: Vec<u16>,
    pub want: Option<u32>,
    pub nan: bool,
}

fn case(name: &str, bits: Vec<u16>, want: Option<u32>) -> KeyCase {
    KeyCase { name: name.to_string(), bits, want, nan: false }
}

/// The eight non-NaN adversarial cases, in the order of the card sequence.
/// Case 2 holds the largest key of the whole sequence (+inf at thread 10), so
/// a slot that is not reset after a call returns 10 from case 3 on. The draft
/// of 2026-11-02 gave that role to case 1, whose +inf sits at thread 383,
/// bitrev 1018, and loses to it.
pub fn adversarial_cases() -> Vec<KeyCase> {
    let mut out = Vec::new();
    let mut v = filled(-1.0);
    v[N - 1] = 0x7C00;
    out.push(case("+inf last", v, Some((N - 1) as u32)));
    let mut v = filled(-1.0);
    v[10] = 0x7C00;
    v[N - 1] = 0x7C00; // thread 383, bitrev 1018, loses to thread 10, bitrev 320
    out.push(case("+inf tie", v, Some(10)));
    let mut v = filled(-1.0);
    v[256] = h(7.5);
    v[512] = h(7.5); // 512 (bitrev 1) beats 256 (bitrev 2)
    out.push(case("256 vs 512", v, Some(512)));
    let mut v = filled(-1.0);
    v[1] = h(7.5);
    v[2] = h(7.5); // 2 (bitrev 256) beats 1 (bitrev 512)
    out.push(case("1 vs 2", v, Some(2)));
    let mut v = filled(-1.0);
    v[5] = h(3.0);
    v[5 + 1024 * 7] = h(3.0); // one thread: the earliest wins
    out.push(case("same thread", v, Some(5)));
    let mut v = filled(-2.0);
    v[1300] = 0x8000; // -0 at thread 276, bitrev 162: the winning position
    v[300] = 0x0000; // +0 at thread 300, bitrev 210
    out.push(case("-0 wins the tie", v, Some(1300)));
    out.push(case("all -inf", vec![0xFC00u16; N], Some(0)));
    let mut v = vec![0x8001u16; N]; // tiny negative subnormals
    v[4096] = 0x0001; // thread 0
    v[77] = 0x0001; // thread 77
    out.push(case("subnormal tie", v, Some(4096)));
    out
}

/// Sparse tie sets: a background of four levels with both signed zeros, the
/// maximum 2.25 at 2 to 6 random positions, so the cross-thread order decides.
/// (A dense draw puts the maximum in thread 0 with probability 1 − 2.3e-9 and
/// never reaches that order.) A fixed LCG keeps them reproducible.
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

/// The NaN case: a quiet NaN at 1234 and a unique maximum at 777, so a kernel
/// that ignored the NaN would return 777, not the 0 the flag path writes.
pub fn nan_case() -> KeyCase {
    let mut v = filled(0.5);
    v[777] = h(0.75);
    v[1234] = 0x7E00;
    KeyCase { name: "NaN".to_string(), bits: v, want: Some(0), nan: true }
}

/// Sparse cases the card check uploads.
pub const CARD_SPARSE: usize = 32;

/// The card sequence, in order: the 8 adversarial cases, the first
/// [`CARD_SPARSE`] sparse ones, the NaN case last (the flag is sticky).
pub fn card_sequence() -> Vec<KeyCase> {
    let mut out = adversarial_cases();
    for (c, bits) in sparse_cases(CARD_SPARSE).into_iter().enumerate() {
        out.push(case(&format!("sparse {c}"), bits, None));
    }
    out.push(nan_case());
    out
}

/// One card mutant of `tok_argmax.cu`: each `from` must occur exactly once in
/// the pristine text, or the mutant is refused (a replace that matches nothing
/// reads as a survivor).
pub struct KernelMutant {
    pub name: &'static str,
    pub edits: &'static [(&'static str, &'static str)],
    /// What kills it, for the log.
    pub killed_by: &'static str,
}

/// The card mutants (repo rule 10). m5, m6, m7 exist only on the card: the
/// host key has no slot, ticket or atomics. m9 is the vector load's half order.
pub const KERNEL_MUTANTS: [KernelMutant; 9] = [
    KernelMutant {
        name: "m1 -0 not mapped to +0",
        edits: &[("if (b == 0x8000u) b = 0u;", "")],
        killed_by: "-0 wins the tie",
    },
    KernelMutant {
        name: "m2 thread order by index",
        edits: &[("return __brev(t) >> 22;", "return t & 1023u;")],
        killed_by: "256 vs 512, 1 vs 2",
    },
    KernelMutant {
        name: "m3 latest index inside a thread",
        edits: &[
            ("(u64)(0x3FFFFFu - (i >> 10));", "(u64)(i >> 10);"),
            ("u32 j = 0x3FFFFFu - (u32)(k & 0x3FFFFFu);", "u32 j = (u32)(k & 0x3FFFFFu);"),
        ],
        killed_by: "same thread",
    },
    KernelMutant {
        name: "m4 nine-bit reversal",
        edits: &[("return __brev(t) >> 22;", "return __brev(t) >> 23;")],
        killed_by: "256 vs 512",
    },
    KernelMutant {
        name: "m5 ticket not reset",
        edits: &[("*ticket = 0u;", "")],
        killed_by: "every case after the first",
    },
    KernelMutant {
        name: "m6 slot not reset",
        edits: &[("const u64 k = atomicExch(slot, 0ull);", "const u64 k = *slot;")],
        killed_by: "every case after +inf last",
    },
    KernelMutant {
        name: "m7 plain store for the max",
        edits: &[("atomicMax(slot, best);", "*slot = best;")],
        killed_by: "the sparse cases",
    },
    KernelMutant {
        name: "m8 NaN flag dropped",
        edits: &[("if (nan) atomicOr(flag, 1u);", "")],
        killed_by: "NaN",
    },
    KernelMutant {
        name: "m9 halves of a word swapped",
        edits: &[("(words[q] >> (16u * hf))", "(words[q] >> (16u * (1u - hf)))")],
        killed_by: "+inf last",
    },
];

/// `text` with every edit of `m` applied, each `from` found exactly once.
pub fn apply_mutant(text: &str, m: &KernelMutant) -> Result<String, String> {
    let mut out = text.to_string();
    for (from, to) in m.edits {
        let count = text.matches(from).count();
        if count != 1 {
            return Err(format!("{}: {from:?} occurs {count} times in tok_argmax.cu, not once", m.name));
        }
        out = out.replacen(from, to, 1);
    }
    Ok(out)
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
        for c in adversarial_cases() {
            let got = agree(&c.bits, &c.name);
            assert_eq!(Some(got), c.want, "{}", c.name);
        }
    }

    #[test]
    fn sparse_ties_reach_the_cross_thread_order() {
        // The cases are only worth their name if the maximum is held by
        // several threads and the winner is not thread 0 every time.
        let mut multi_thread = 0;
        let mut not_thread0 = 0;
        for v in sparse_cases(CARD_SPARSE) {
            let max = *v.iter().map(|&b| ord16(b)).collect::<Vec<_>>().iter().max().unwrap();
            let mut threads: Vec<usize> =
                v.iter().enumerate().filter(|(_, &b)| ord16(b) == max).map(|(i, _)| i % 1024).collect();
            threads.sort_unstable();
            threads.dedup();
            multi_thread += usize::from(threads.len() >= 2);
            not_thread0 += usize::from(!key_argmax(&v).0.is_multiple_of(1024));
        }
        assert!(multi_thread >= 28, "{multi_thread} of {CARD_SPARSE} sparse cases tie across threads");
        assert!(not_thread0 >= 28, "{not_thread0} of {CARD_SPARSE} sparse cases resolve outside thread 0");
    }

    #[test]
    fn sparse_ties_agree_card_set() {
        for (c, v) in sparse_cases(CARD_SPARSE).iter().enumerate() {
            agree(v, &format!("sparse case {c}"));
        }
    }

    #[test]
    #[cfg_attr(debug_assertions, ignore)] // 300 × two passes over 151,936: the release loop
    fn sparse_ties_agree_300() {
        for (c, v) in sparse_cases(300).iter().enumerate() {
            agree(v, &format!("sparse case {c}"));
        }
    }

    #[test]
    fn dense_ties_keep_the_in_thread_order() {
        // The maximum at 1/8 of positions, so thread 0 holds several and must
        // return its earliest.
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
    fn a_nan_raises_the_flag_and_yields_token_zero() {
        let c = nan_case();
        let (k, nan) = key_argmax(&c.bits);
        assert!(nan);
        assert_eq!(Some(k), c.want, "token 0 keeps the next gather in bounds");
        // Without the NaN the unique maximum wins: the case can tell a kernel
        // that skipped the flag from one that raised it.
        let mut clean = c.bits.clone();
        clean[1234] = h(0.5);
        assert_eq!(agree(&clean, "NaN case without its NaN"), 777);
    }

    #[test]
    fn key_index_inverts_tie_key() {
        for &i in &[0u32, 1, 2, 511, 512, 1023, 1024, 151_935] {
            assert_eq!(key_index(tie_key(0x3C00, i)), i);
        }
    }

    #[test]
    fn the_card_sequence_is_41_cases_with_the_nan_last() {
        let seq = card_sequence();
        assert_eq!(seq.len(), 41);
        assert!(seq[..40].iter().all(|c| !c.nan) && seq[40].nan);
        // Case 2 holds the largest key of the sequence, so an unreset slot
        // poisons every later case.
        let keys: Vec<u64> = seq[..40]
            .iter()
            .map(|c| c.bits.iter().enumerate().map(|(i, &b)| tie_key(b, i as u32)).max().unwrap())
            .collect();
        assert!(keys[0] < keys[1] && keys[2..].iter().all(|&k| k < keys[1]));
    }

    #[test]
    fn every_card_mutant_applies_exactly_once_and_changes_the_text() {
        for m in &KERNEL_MUTANTS {
            let t = apply_mutant(TOK_ARGMAX_CU, m).unwrap_or_else(|e| panic!("{e}"));
            assert_ne!(t, TOK_ARGMAX_CU, "{}", m.name);
        }
        let ghost = KernelMutant { name: "ghost", edits: &[("no such text", "")], killed_by: "" };
        assert!(apply_mutant(TOK_ARGMAX_CU, &ghost).is_err());
    }

    #[test]
    fn key_preconditions_hold_for_qwen3_and_refuse_the_rest() {
        assert!(key_preconditions(N, 0).is_ok());
        assert!(key_preconditions(N, 151_936 * 3).is_ok());
        assert!(key_preconditions(1000, 0).is_err());
        assert!(key_preconditions(N, 4).is_err());
        assert!(key_preconditions(N + 4, 0).is_err());
        assert_eq!(key_grid(N), 75);
    }
}
