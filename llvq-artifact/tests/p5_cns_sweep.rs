//! **P5 §2.2 / C2 — the integral sweep of the CNS re-bijection.**
//!
//! `proofs/preregistration-p5-2026-08-14.md` §3, V0.d: every one of the
//! **150,681,600 blocks** of the sealed Qwen3-4B goes archive index → CNS
//! record → point, and the point must equal `FastDecoder::decode`'s. Any
//! discrepancy buries E1v without a timing (rule of the 08-11).
//!
//! It lives here because this is the only crate that can open a `.llvq` —
//! `llvq-search` has no dev-dependency and is meant to keep none (P5 §2.5).
//!
//! ## What rides along, and why in the same pass
//!
//! * **C3.1** — the maximum step count, over real blocks rather than over a
//!   fixture. Green at ≤ 96, the bound the variant declares for itself.
//! * **C3.3** — that count is the same for every block of a class. A single
//!   number per class, merged across threads, and a mismatch names the class.
//! * **C1's raw width** — the CNS record's bits, averaged over the real class
//!   mix. This is the number amendment É0 substitutes for the published
//!   **51.87** bits/block: same composition, rounded **per kind** rather than
//!   per stage, because per-stage rounding means a packed peel and a packed
//!   peel means a division.
//!
//! ⚠️ **What this sweep does NOT give.** The *addressed* figure — 53.332
//! bits/block FO/warp-scan, and the 2.3709 b/weight kernel that derives from it
//! — needs the group-of-32 addressing model, which lives in `radixstudy` and
//! which P5 §4 forbids this side from reading. What is printed here is the
//! **raw** width; converting it is a separate deliverable with its own
//! accounting (P5 §1.1, §1.2), and doing it by eye would merge two
//! accountings that the pre-registration spends a page separating.
//!
//! ⚠️ **And the origin is not here.** The published 4B carries **zero** origin
//! block, so no sweep of it will ever exercise that path. It is covered by
//! `llvq_search::cns`'s own fixture, which pushes index 0 on purpose. Writing
//! "proven on 150.7 M blocks" without that caveat would be false — P5 §2.2
//! says so in as many words.
//!
//! `#[ignore]`d unconditionally: it reads a 981 MB archive that is not in the
//! repo, and it fails loudly rather than skipping green when the file is absent
//! (see `common::sealed_artifact_path`).

use llvq_core::Golay;
use llvq_search::cns::{cns_decode_counted, cns_encode, cns_layout};
use llvq_search::fastdec::FastDecoder;

mod common;

/// Blocks per work unit. No group structure to respect here — the CNS record is
/// per block — so this is only a scheduling grain.
const CHUNK: usize = 1 << 18;

/// C3.1's bound, from the variant's own declaration.
const MAX_STEPS: u32 = 96;

#[test]
#[ignore = "reads the sealed 4B archive; minutes even in release"]
fn the_sealed_artifact_cns_re_bijection_is_exact() {
    use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
    use std::sync::Mutex;

    let path = common::sealed_artifact_path();
    let fd = FastDecoder::new();
    let golay = Golay::new();
    let layouts: Vec<u64> = (0..fd.n_classes())
        .map(|ci| cns_layout(&fd, &golay, ci).bits())
        .collect();

    let f = std::fs::File::open(&path).expect("open the sealed artifact");
    let mut r = std::io::BufReader::new(f);
    let h = llvq_artifact::read_header(&mut r).expect("valid artifact header");

    let nthreads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let verified = AtomicU64::new(0);
    let bits = AtomicU64::new(0);
    let max_steps = AtomicU32::new(0);
    // One step count per class, or none yet. C3.3 is the assertion that a class
    // never shows two.
    let per_class: Mutex<Vec<Option<u32>>> = Mutex::new(vec![None; fd.n_classes()]);

    for _mi in 0..h.matrices {
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
                    let mut local: Vec<Option<u32>> = vec![None; fd.n_classes()];
                    let (mut local_max, mut local_bits) = (0u32, 0u64);

                    for (b, (&idx, &g)) in m.indices[lo..hi]
                        .iter()
                        .zip(&m.gains[lo..hi])
                        .enumerate()
                    {
                        let gain = u8::try_from(g).expect("one gain bit");
                        let rec = cns_encode(&fd, idx, gain)
                            .unwrap_or_else(|| panic!("{} block {}: index outside the ball", m.name, lo + b));
                        let mut steps = 0u32;
                        let got = cns_decode_counted(&fd, &golay, &rec, &mut steps);

                        // C2 — the re-bijection closes against the archive.
                        let want = fd
                            .decode(idx)
                            .unwrap_or_else(|| panic!("{} block {}: index outside the ball", m.name, lo + b));
                        assert_eq!(got, want, "{} block {}: CNS against the archive", m.name, lo + b);
                        assert_eq!(rec.gain, gain, "{} block {}: gain", m.name, lo + b);

                        // C3.1 and C3.3.
                        local_max = local_max.max(steps);
                        let ci = rec.class.expect("the 4B carries no origin block");
                        match local[ci] {
                            None => local[ci] = Some(steps),
                            Some(s) => assert_eq!(
                                s, steps,
                                "{} block {}: class {ci}, the step count moves with the rank",
                                m.name,
                                lo + b
                            ),
                        }
                        local_bits += layouts[ci];
                    }

                    max_steps.fetch_max(local_max, Ordering::Relaxed);
                    bits.fetch_add(local_bits, Ordering::Relaxed);
                    verified.fetch_add((hi - lo) as u64, Ordering::Relaxed);
                    let mut g = per_class.lock().expect("histogram");
                    for (ci, (a, b)) in g.iter_mut().zip(&local).enumerate() {
                        match (*a, *b) {
                            (None, x) => *a = x,
                            (Some(x), Some(y)) => assert_eq!(
                                x, y,
                                "class {ci}: two step counts ({x} and {y}) — C3.3 red"
                            ),
                            _ => {}
                        }
                    }
                });
            }
        });
    }

    let n = verified.load(Ordering::Relaxed);
    let steps = max_steps.load(Ordering::Relaxed);
    let mean = bits.load(Ordering::Relaxed) as f64 / n as f64;
    let touched = per_class
        .lock()
        .expect("histogram")
        .iter()
        .filter(|s| s.is_some())
        .count();

    // The count is the assertion, not a decoration: a sweep that read one
    // matrix and stopped would print the same green line.
    assert_eq!(n, 150_681_600, "the published 4B has 150,681,600 blocks");
    assert!(steps <= MAX_STEPS, "C3.1 RED — {steps} steps > {MAX_STEPS}");
    eprintln!(
        "P5 C2/C3 — {n} blocks, {matrices} matrices, {touched} classes touched.\n  \
         C2   bijection: archive → CNS → point == FastDecoder::decode, 0 deviation\n  \
         C3.1 max steps: {steps} / {MAX_STEPS}\n  \
         C3.3 per-class invariance: green on the {touched} classes of the file\n  \
         CNS width    : {mean:.4} bits/block RAW, 10 b header included — the PER-KIND \
         accounting\n               of amendment É0, to compare with the 51.87 \
         published for the PER-STAGE accounting.\n               WARNING: RAW — the \
         addressed figure (FO/warp-scan) needs the group-of-32\n               addressing \
         model, which §4 forbids this side from reading. Another deliverable.\n  \
         WARNING: ORIGIN not exercised — the 4B carries none. It is covered by the \
         fixture\n     of `llvq_search::cns`, never here (P5 §2.2).",
        matrices = h.matrices
    );
}
