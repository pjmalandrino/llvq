//! **P1 §3.3 — the integral sweep**, for the two rank decoders the bench judges.
//!
//! `proofs/preregistration-p1-2026-08-13.md` §3 orders V0 in three steps: a CPU
//! reference written independently of the shader, a block-by-block check on a
//! broad sample plus the §1.6 synthetic fixture, and then **the integral sweep
//! of the 150,681,600 blocks** of the sealed 4B. The first two are
//! `llvq-search/src/rankdec.rs` and `llvq-metal/src/bin/p1v0.rs`. This file is
//! the third.
//!
//! It lives here and not in `llvq-search` because this is the only crate that
//! can open a `.llvq` — `llvq-search` has no dev-dependency and is meant to
//! keep none (pre-registration §3, *code placement*).
//!
//! ## The two arms are swept against their own standards, which differ
//!
//! | arm | what is asserted, on every block |
//! |---|---|
//! | `cascade_uniform` | the **point** it rebuilds equals `FastDecoder::decode`'s, coordinate for coordinate |
//! | `binomial_walk` | `rank → arrangement → rank` closes, and the arrangement realises the class's multiset |
//!
//! Amendment É2 is why the second is not an equality: a binomial walk unranks
//! in combinatorial order, the archive in multiset-permutation order, and
//! relating the two *is* the CNS re-bijection — that is P5, which this bench
//! gates. Demanding equality here would demand the transcoder P1's own verdict
//! authorises.
//!
//! ## What the sweep adds over the unit tests, and it is not "more of the same"
//!
//! `cascade_uniform_matches_the_archive` probes 43 ranks per class. Real blocks
//! are not probes: they are 150,681,600 ranks drawn by GPTQ from a distribution
//! nobody chose, concentrated on 243 of the 383 classes, and this is the only
//! harness that sees them. It is also the only one that exercises the
//! **composition** — the mixed-radix peel, the Golay lookup, the parity repair,
//! the three sign rules — on real data, and that composition is restated here
//! from `llvq-metal/shaders/cascade_uniform.metal` rather than imported from
//! `FastDecoder::decode`, which is the yardstick and must owe the thing under
//! test nothing.
//!
//! ⚠️ **What it does NOT cover.** No GPU runs here. This sweep bounds the CPU
//! references; the shaders are bounded by `bin/p1v0`, on 2^20 real blocks and
//! the synthetic fixture. Neither implies the other, and the journal must not
//! read this green as a statement about a kernel.
//!
//! The classes a real draw cannot reach — the origin, the 82 shell-13 classes,
//! the 15 unused cap-12 classes — are not reachable here either, by
//! construction: this sweep reads the file. They are `p1v0`'s pass 1.
//!
//! `#[ignore]`d unconditionally: it reads a 981 MB archive that is not in the
//! repo, and it fails loudly rather than skipping green when the file is absent
//! (see `common::sealed_artifact_path`).

use llvq_core::{Golay, SplitMix64, DIM};
use llvq_search::fastdec::{FastDecoder, MAX_KINDS};
use llvq_search::rankdec::{binomial_rank, binomial_walk, cascade_uniform, walk_radices};

mod common;

/// Entries of the class table, the origin included — the id convention shared
/// by `ClassTable` and `llvq-metal`'s walk table.
const ENTRIES: usize = 384;

/// Blocks per work unit. Large enough that the per-chunk merge into the shared
/// histogram is noise, small enough that 252 matrices of uneven size still
/// balance across threads.
const CHUNK: usize = 1 << 18;

/// Feed for the walk arm's ranks. Mixed with the matrix and chunk indices
/// rather than shared, so what each block is asked to decode does **not**
/// depend on how the threads happened to be scheduled: a sweep whose failing
/// block moved from run to run would be unusable as a bug report.
const SEED: u64 = 0x0000_5EED_B1A0;

/// Rebuild a block's point from its archive index using **`cascade_uniform`**
/// where `FastDecoder::decode` uses `unrank_multiset`.
///
/// The composition around it — the peel of `s_f`, `s_w`, `a_off`, `a_on`, the
/// Golay lookup, the parity repair and the three sign rules — is restated from
/// `llvq-metal/shaders/cascade_uniform.metal` (its steps 1 and 2), **not** from
/// `fastdec.rs`. That is deliberate and it is what makes the sweep worth
/// running twice over: the yardstick is `FastDecoder::decode`, so a
/// composition copied out of it would agree with it by construction, and the
/// sweep would check the arrangement decoder while silently endorsing whatever
/// recipe the shader follows. Restating the shader's recipe puts *both* under
/// the same yardstick.
///
/// The origin decodes to the zero point. No block of the published 4B is the
/// origin, so this branch is here for the format, not for the file.
fn point_via_cascade(fd: &FastDecoder, golay: &Golay, idx: u64) -> [i32; DIM] {
    let mut p = [0i32; DIM];
    if idx == 0 {
        return p;
    }
    let ci = fd.class_of(idx).expect("index inside the cap-13 ball");
    let c = fd.cascade_class(ci);
    let mut local = (idx - 1) - c.offset;

    if c.odd {
        // One 24-slot problem, indexed by the coordinate itself.
        let r_arr = local % c.m_arr;
        let gi = (local / c.m_arr) as usize;
        let cw = golay.codewords()[gi];
        let mut counts = c.counts;
        let mut kinds = [0u8; DIM];
        cascade_uniform(r_arr, &mut counts, c.n_kinds, c.m_arr, &mut kinds);
        for (slot, pi) in p.iter_mut().enumerate() {
            let v = c.vals[kinds[slot] as usize];
            // Shader: `neg_odd = cbit ^ ((v >> 1) & 1)`. For an odd |value|,
            // bit 1 is set exactly when v ≡ 3 (mod 4) — the whole forced-sign
            // rule as one XOR, which is why the shader can afford it.
            let neg = ((cw >> slot) & 1) ^ ((v as u32 >> 1) & 1);
            *pi = if neg == 1 { -v } else { v };
        }
        return p;
    }

    // Two problems, the support and its complement, and four fields peeled off
    // the local rank in this order.
    let r_sf = local % c.s_f;
    local /= c.s_f;
    let r_sw = local % c.s_w;
    local /= c.s_w;
    let r_off = local % c.a_off;
    local /= c.a_off;
    let r_on = local % c.a_on;
    let gi = (local / c.a_on) as usize;
    let cw = golay.of_weight(c.w as usize)[gi];
    let w = c.w as usize;

    let mut wc = [0u8; MAX_KINDS];
    wc[..4].copy_from_slice(&c.word_counts);
    let mut on_kinds = [0u8; DIM];
    cascade_uniform(r_on, &mut wc, c.n_word, c.a_on, &mut on_kinds[..w]);

    let mut oc = c.off_counts;
    let mut off_kinds = [0u8; DIM];
    cascade_uniform(r_off, &mut oc, c.n_off, c.a_off, &mut off_kinds[..DIM - w]);

    // The zero run is the last off-support kind, and its being a *kind* rather
    // than an absence is what lets both problems run 24 identical steps.
    let zero_kind = (c.n_off - 1) as u8;
    let (mut s_on, mut s_off) = (0usize, 0usize);
    let mut par = 0u32;
    let mut sf = r_sf;
    for (slot, pi) in p.iter_mut().enumerate() {
        if (cw >> slot) & 1 == 1 {
            let v = c.word_vals[on_kinds[s_on] as usize];
            let bfree = ((r_sw >> s_on) & 1) as u32;
            // The last support slot carries the parity repair instead of a
            // free bit: that is the bit the class does not spend.
            let neg = if s_on + 1 == w { par ^ c.p_req } else { bfree };
            if s_on + 1 != w {
                par ^= bfree;
            }
            *pi = if neg == 1 { -v } else { v };
            s_on += 1;
        } else {
            let k = off_kinds[s_off];
            if k != zero_kind {
                let v = c.free_vals[k as usize];
                *pi = if sf & 1 == 1 { -v } else { v };
                sf >>= 1;
            }
            s_off += 1;
        }
    }
    p
}

/// One block through the walk's own bijection: `rank → arrangement → rank`.
///
/// The multiset check is not decoration and it does not follow from the round
/// trip: `binomial_rank` walks `0..k-1`, the final kind having no rank of its
/// own, so a walk that misplaced the last kind closes the trip anyway. That
/// exact mutation survived five tests before the assertion existed
/// (`rankdec.rs`, `the_walk_round_trips_on_its_own_bijection`).
fn walk_round_trips(fd: &FastDecoder, ci: usize, rng: &mut SplitMix64) {
    let lv = fd.levels(ci);
    let k = lv.len;
    let mut counts = [0u8; MAX_KINDS];
    counts[..k].copy_from_slice(&lv.counts[..k]);
    let rad = walk_radices(&counts, k, DIM);

    let mut ranks = [0u64; MAX_KINDS];
    for (j, r) in ranks.iter_mut().enumerate().take(k) {
        *r = if rad[j] > 1 { rng.next() % rad[j] } else { 0 };
    }

    let mut arr = [0u8; DIM];
    binomial_walk(&ranks, &counts, k, &mut arr);
    for (j, &c) in counts.iter().enumerate().take(k) {
        assert_eq!(
            arr.iter().filter(|&&x| x == j as u8).count(),
            c as usize,
            "class {ci}, kind {j} misplaced: ranks {ranks:?}"
        );
    }
    let back = binomial_rank(&arr, &counts, k);
    for j in 0..k.saturating_sub(1) {
        assert_eq!(
            back[j], ranks[j],
            "class {ci}, kind {j}: the round trip does not close"
        );
    }
}

/// **The sealed sweep.** Every block of the published Qwen3-4B through both
/// candidate decoders, against the standard each one has.
#[test]
#[ignore = "reads the sealed 4B archive; minutes even in release"]
fn the_sealed_artifact_p1_decoders_are_exact() {
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::sync::Mutex;

    let path = common::sealed_artifact_path();
    let fd = FastDecoder::new();
    let golay = Golay::new();
    let f = std::fs::File::open(&path).expect("open the sealed artifact");
    let mut r = std::io::BufReader::new(f);
    let h = llvq_artifact::read_header(&mut r).expect("valid artifact header");

    let nthreads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let verified = AtomicU64::new(0);
    let hist = Mutex::new(vec![0u64; ENTRIES]);

    for mi in 0..h.matrices {
        let m = llvq_artifact::read_matrix_raw(&mut r).expect("valid matrix");
        assert_eq!(m.shell_cap, 12, "{}: the sealed file is leech1c12", m.name);
        let nchunks = m.indices.len().div_ceil(CHUNK);
        let next = AtomicUsize::new(0);
        std::thread::scope(|sc| {
            for _ in 0..nthreads {
                sc.spawn(|| loop {
                    let c = next.fetch_add(1, Ordering::Relaxed);
                    if c >= nchunks {
                        break;
                    }
                    let lo = c * CHUNK;
                    let hi = ((c + 1) * CHUNK).min(m.indices.len());
                    // Seeded from the position in the file, never from the
                    // thread: the same block is asked the same rank on every
                    // run, whatever the scheduling.
                    let mut rng =
                        SplitMix64::new(SEED ^ ((mi as u64) << 40) ^ ((c as u64) << 8));
                    let mut local = vec![0u64; ENTRIES];
                    for (b, &idx) in m.indices[lo..hi].iter().enumerate() {
                        // Arm 1 — point for point against the archive decoder.
                        let want = fd.decode(idx).expect("index inside the ball");
                        assert_eq!(
                            point_via_cascade(&fd, &golay, idx),
                            want,
                            "{} block {}: cascade_uniform against FastDecoder::decode",
                            m.name,
                            lo + b
                        );
                        // Arm 2 — round trip on its own bijection (É2), on the
                        // class this block actually has.
                        let id = match fd.class_of(idx) {
                            Some(ci) => {
                                walk_round_trips(&fd, ci, &mut rng);
                                1 + ci
                            }
                            None => 0,
                        };
                        local[id] += 1;
                    }
                    let mut g = hist.lock().expect("histogram");
                    for (a, b) in g.iter_mut().zip(&local) {
                        *a += b;
                    }
                    verified.fetch_add((hi - lo) as u64, Ordering::Relaxed);
                });
            }
        });
    }

    let n = verified.load(Ordering::Relaxed);
    let hist = hist.into_inner().expect("histogram");
    let touched = hist.iter().filter(|&&c| c > 0).count();
    let shell13 = (0..fd.n_classes())
        .filter(|&ci| fd.levels(ci).shell == 13)
        .count();

    // The count is the assertion, not a decoration: a sweep that read one
    // matrix and stopped would print the same green line.
    assert_eq!(n, 150_681_600, "the published 4B has 150,681,600 blocks");
    eprintln!(
        "P1 §3.3 — {n} blocks, {} matrices.\n  \
         cascade_uniform: point for point against FastDecoder::decode, 0 deviation\n  \
         binomial_walk  : rank → arrangement → rank closes, multiset realised\n  \
         {touched} of {ENTRIES} table entries touched ({} origin) — the {} \
         remaining,\n  including the {shell13} shell-13 classes, are not reachable from \
         ANY cap-12 file\n  and belong to the synthetic fixture of \
         `bin/p1v0` (preregistration §1.6).",
        h.matrices,
        hist[0],
        ENTRIES - touched,
    );
}
