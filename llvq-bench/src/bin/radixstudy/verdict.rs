//! The verdict: the priced points, the text that states which clause of the
//! spec they answer, and the marks the tables print.

use super::*;

/// One priced point of the verdict: a variant, an addressing mode, its cost and
/// the decode depth that decides which clause it answers.
pub(super) struct Point {
    pub(super) name: &'static str,
    pub(super) mode: Mode,
    pub(super) bpw: f64,
    pub(super) depth: Option<u32>,
}

/// The one sentence that asserts the spec's **conjunction** is satisfied.
///
/// A constant so that [`tests::a_depth_reopening_never_prints_the_spec_admission_sentence`]
/// can require its absence: printing it for a point that only cleared the bit
/// half is precisely the over-claim of the 2026-08-13 journals, and it is now a
/// test failure rather than a proofreading duty.
pub(super) const OPEN_AT_SPEC: &str = "The E3 workstream is open IN THE SENSE OF THE SPEC";

/// The verdict, as text, so it can be asserted on rather than eyeballed.
///
/// `admissible` holds only points satisfying **both** clauses; `reopening`
/// holds points under the bit criterion whose decoder is deeper than
/// [`MAX_DEPTH`]. The two are never merged, and the second can never reach
/// [`OPEN_AT_SPEC`].
pub(super) fn verdict_text(admissible: &[Point], reopening: &[Point], best: f64) -> String {
    use std::fmt::Write;
    let mut o = String::new();
    if admissible.is_empty() {
        let _ = writeln!(
            o,
            "  FAIL: NO ADMISSIBLE decomposition, shift-only AND fixed depth ≤ {MAX_DEPTH} \
             steps,\n     the CONJUNCTION the spec demands, gets under {E3_CRITERION} \
             kernel b/weight IN FILE ORDER,\n     neither in grp32-max nor in warp-scan."
        );
        if best.is_finite() {
            let _ = writeln!(
                o,
                "  The best ADMISSIBLE point is {best:.4} b/weight, {:.0}% above the threshold.",
                100.0 * (best / E3_CRITERION - 1.0)
            );
        }
        let _ = writeln!(
            o,
            "  The criterion for opening the E3 workstream is NOT met: E3 buries itself on\n  \
             paper, for 0 $, as E2 buried itself at the bench. The next rung of the memory\n  \
             ladder stays E1c (X1/X2), and the project's practical floor is the measured layout."
        );
    } else {
        let _ = writeln!(
            o,
            "  PASS: ADMISSIBLE decomposition(s) (shift-only AND depth ≤ {MAX_DEPTH}) under the \
             criterion,\n     in file order:"
        );
        for p in admissible {
            let _ = writeln!(
                o,
                "     {:<24}{:<6}{:.4} kernel b/weight   (depth {})",
                p.name,
                p.mode.tag(),
                p.bpw,
                depth_tag(p.depth)
            );
        }
        let _ = writeln!(
            o,
            "  {OPEN_AT_SPEC}. WARNING: this verdict is about the BITS\n  \
             alone. The speed of such a decoder is still unmeasured, and Golay70 recalls that a\n  \
             correct and compact format can die in the ALU (195 GB/s against 425)."
        );
    }
    if !reopening.is_empty() {
        let _ = writeln!(
            o,
            "\n  WARNING: UNDER THE BIT CRITERION, BUT OUTSIDE THE DEPTH CLAUSE ≤ {MAX_DEPTH}:"
        );
        for p in reopening {
            let _ = writeln!(
                o,
                "     {:<24}{:<6}{:.4} kernel b/weight   (depth {})",
                p.name,
                p.mode.tag(),
                p.bpw,
                depth_tag(p.depth)
            );
        }
        let _ = writeln!(
            o,
            "  These points are NOT admitted: the criterion for opening E3\n  \
             (spec-memoire-extreme-2026-08-12.md:186-187) is a CONJUNCTION, ≤ {E3_CRITERION} \
             b/weight\n  \
             AND a fixed-depth decoder ≤ {MAX_DEPTH} steps with no inter-slot serial state.\n  \
             They amount to a REOPENING OF THE DEPTH CLAUSE, which is a handover decision\n  \
             and not a result of this bench. Publishing them as an admission would be the\n  \
             over-claim this block exists to prevent."
        );
    }
    o
}

fn depth_tag(d: Option<u32>) -> String {
    d.map_or("serial".to_string(), |d| format!("{d}"))
}

pub(super) fn mark(ok: bool) -> &'static str {
    if ok {
        "✅"
    } else {
        "❌"
    }
}

/// Marker of the `S_alt` column, which has **three** states and not two.
///
/// The pre-registration §2 forbids publishing a result in `]S_spec ; S_alt]` as
/// a pass, but a bare ✅ in that band cites itself — it was printed there until
/// 2026-08-13. `⊘` is deliberately not a verdict glyph: it cannot be quoted
/// without its legend, which is the point.
pub(super) fn mark_alt(bpw: f64) -> &'static str {
    if bpw <= E3_CRITERION {
        "✅"
    } else if bpw <= S_ALT {
        "⊘"
    } else {
        "❌"
    }
}
