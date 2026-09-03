//! **P5 C4 — what transcoding to `E1v` costs, against `Planes14`.**
//!
//! `T ≡ the wall-clock time, from a matrix's (indices, gains) array to the
//! layout's final byte buffer, allocation included, disk I/O excluded`. The ratio
//! is formed **pass by pass**; C4 is green if the **median of three ratios** is
//! ≤ 2.0, and the range of the three is published.
//!
//! ## Mono-thread, and why that is not a detail
//!
//! Both arms run on **one thread of the same process**, alternating within each
//! pass. `transcode_planes14`'s mono-thread cost on the 4B **has never been
//! measured in a defined thread model** — P5 §4 says so — so the anchor is
//! established here rather than quoted, and it is established next to the arm it
//! anchors. A ratio between a number measured today and one remembered from a
//! run with an unstated thread count would be worth nothing.
//!
//! ## What C4 can and cannot do
//!
//! 🚨 **C4 is a criterion of ADOPTION, never of burial** (P5 §4). Its ×2 is an
//! engineering judgement, not a derived quantity, and unlike E2's "2.0×" — which
//! was *the speed of a layout already served* — nothing here is a threshold some
//! served path already meets. A red C4 means `E1v` is not adopted for the served
//! path and is published as a point on the curve. It does not touch C1, C2 or
//! C3, and it does not touch the reopening those three grant.
//!
//! `#[ignore]`d unconditionally: it reads a 981 MB archive that is not in the
//! repo, and it fails loudly rather than skipping green when the file is absent.

use llvq_artifact::e1v::{transcode_e1v, E1V_GROUP};
use llvq_artifact::runtime::{transcode_planes14, ClassTable};
use llvq_core::Golay;
use llvq_search::fastdec::FastDecoder;
use std::time::Instant;

mod common;

/// Passes. Three, per §4 — the median of three ratios, and their range.
const PASSES: usize = 3;

/// C4's threshold, verbatim.
const C4_MAX: f64 = 2.0;

#[test]
#[ignore = "reads the sealed 4B archive; minutes even in release"]
fn the_e1v_transcode_is_within_twice_planes14() {
    let path = common::sealed_artifact_path();
    let fd = FastDecoder::new();
    let golay = Golay::new();
    let table = ClassTable::new(&fd, 1);

    // The largest matrix of the file: one matrix, as §4's `T` says, and the one
    // whose cost dominates any real transcode. Its name is printed — a ratio
    // measured on an unnamed slice of a model is not reproducible.
    let f = std::fs::File::open(&path).expect("open the sealed artifact");
    let mut r = std::io::BufReader::new(f);
    let h = llvq_artifact::read_header(&mut r).expect("valid artifact header");
    let mut best: Option<(String, Vec<u64>, Vec<u32>)> = None;
    for _ in 0..h.matrices {
        let m = llvq_artifact::read_matrix_raw(&mut r).expect("valid matrix");
        if best.as_ref().is_none_or(|(_, i, _)| m.indices.len() > i.len()) {
            best = Some((m.name.clone(), m.indices.clone(), m.gains.clone()));
        }
    }
    let (name, indices, gains) = best.expect("at least one matrix");
    assert_eq!(
        indices.len() % E1V_GROUP,
        0,
        "{name}: {} blocks, E1v addresses whole groups",
        indices.len()
    );

    // Warm the caches once, outside the timed passes, on both arms — otherwise
    // the first pass would price a cold page table and the ratio would move
    // with the order of the arms.
    let _ = transcode_planes14(&fd, &table, &indices[..E1V_GROUP], &gains[..E1V_GROUP]);
    let _ = transcode_e1v(&fd, &golay, &indices[..E1V_GROUP], &gains[..E1V_GROUP]);

    let mut ratios = Vec::with_capacity(PASSES);
    let (mut t14s, mut t1vs) = (Vec::new(), Vec::new());
    for p in 0..PASSES {
        // Alternating within the pass, so a machine that drifts over the run
        // drifts through both arms rather than through one.
        let t0 = Instant::now();
        let p14 = transcode_planes14(&fd, &table, &indices, &gains).expect("transcodes");
        let t14 = t0.elapsed().as_secs_f64();

        let t0 = Instant::now();
        let e1v = transcode_e1v(&fd, &golay, &indices, &gains).expect("transcodes");
        let t1v = t0.elapsed().as_secs_f64();

        assert_eq!(p14.n_blocks, indices.len());
        assert_eq!(e1v.n_blocks, indices.len());
        // Both buffers are read once, so neither call can be optimised away —
        // §7's "an arm the compiler eliminated measures nothing", applied to a
        // transcode rather than to a kernel.
        assert!(!p14.data.is_empty() && !e1v.data.is_empty());

        ratios.push(t1v / t14);
        t14s.push(t14);
        t1vs.push(t1v);
        eprintln!(
            "  pass {}: Planes14 {t14:.3} s · E1v {t1v:.3} s · ratio {:.3}×",
            p + 1,
            t1v / t14
        );
    }

    let mut sorted = ratios.clone();
    sorted.sort_by(f64::total_cmp);
    let median = sorted[PASSES / 2];
    let e1v = transcode_e1v(&fd, &golay, &indices, &gains).expect("transcodes");
    let p14 = transcode_planes14(&fd, &table, &indices, &gains).expect("transcodes");

    eprintln!(
        "\nP5 C4 — mono-thread transcode, matrix `{name}`, {} blocks, {PASSES} passes\n  \
         Planes14: {:.3} / {:.3} / {:.3} s   ({:.4} b/weight)\n  \
         E1v     : {:.3} / {:.3} / {:.3} s   ({:.4} b/weight, addressing included)\n  \
         ratio formed PASS BY PASS: median {median:.3}× [range {:.3}–{:.3}]\n  \
         C4 threshold: {C4_MAX} — {}\n  \
         WARNING: C4 is a criterion of ADOPTION, never of burial. A red touches neither C1,\n     \
         nor C2, nor C3, nor the reopening they grant.\n  \
         WARNING: the mono-thread Planes14 anchor is established HERE: it had never been \
         measured\n     in a defined thread model (§4).",
        indices.len(),
        t14s[0], t14s[1], t14s[2],
        p14.bits_per_weight(),
        t1vs[0], t1vs[1], t1vs[2],
        e1v.bits_per_weight(),
        sorted[0],
        sorted[PASSES - 1],
        if median <= C4_MAX { "GREEN" } else { "RED (not adopted for the served path)" }
    );
}
