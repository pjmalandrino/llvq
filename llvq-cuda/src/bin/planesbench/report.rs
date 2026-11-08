//! One phase's table: times, bytes read, b/weight, and the ratios formed round by round.

use super::*;

/// The table of one phase, printed after its rounds.
#[allow(clippy::too_many_arguments)]
pub(super) fn report(
    mats: &[Mat],
    pi: usize,
    phase: ArmSet,
    times: &[Vec<f64>; arms::N_ARMS],
    n_phases: usize,
    per_round: &impl Fn(&[f64], &[f64]) -> Vec<f64>,
    tile: llvq_cuda::tile::Tile,
    source: &str,
    dev: &llvq_cuda::gpu::DeviceReport,
) {
    let bytes_of = |a: usize| -> u64 { mats.iter().map(|m| arm_bytes(m, a)).sum() };
    // The arm's own denominator, never the table's. `n_weights` stays
    // what it always was — the model's weight count, printed once in
    // the header — but it is no longer what a b/weight column divides
    // by, because not every arm reads every matrix any more.
    let weights_of = |a: usize| -> u64 { mats.iter().map(|m| arm_weights(m, a)).sum() };
    let bpw = |a: usize| -> f64 { bytes_of(a) as f64 * 8.0 / weights_of(a).max(1) as f64 };
    // How many matrices an arm's timed pass actually launched. Equal
    // for every ball arm; 216 of 252 for the served Tetra one.
    let covered = |a: usize| -> usize {
        mats.iter().filter(|m| arm_weights(m, a) > 0).count()
    };
    // Rule 8: a number carries its accounting. This column and the
    // `rtbits` kernel rate differ by exactly this, and by nothing else.
    let tetra_pad: u64 = mats.iter().filter_map(|m| m.tetra.as_ref()).map(|t| t.pad_bytes).sum();
    let t_f16 = &times[arms::FP16];
    let phase_tag = if n_phases > 1 {
        format!("  [phase {}/{}: {}]", pi + 1, n_phases, phase.label())
    } else {
        String::new()
    };
    println!(
        "\nONE TOKEN — {} matrices, one stream, {} arms interleaved{}",
        mats.len(),
        phase.len(),
        phase_tag
    );
    println!(
        "  {ROUNDS} rounds, {WARMUP} discarded; the ratios are formed ROUND BY ROUND"
    );
    println!("  {}", "-".repeat(80));
    // "GB/s(min)", not "GB/s": the column divides by the MINIMUM time
    // — the most favourable reading, that of a peak throughput — and
    // the header did not say so (raised by the 2026-08-18 audit).
    // Declared rather than changed: the published journals carry this
    // accounting, and what was missing is the label, not the formula.
    println!(
        "  {:<22}{:>9}{:>9}{:>9}{:>9}{:>9}{:>10}",
        "format", "min ms", "med ms", "max ms", "GB read", "b/weight", "GB/s(min)"
    );
    // 🚨 The tile, ON the table. Its provenance is printed once, some
    // thirty lines and a three-minute transcode earlier, and F1d runs
    // THREE processes at three tiles into one journal: a reader
    // comparing two tables would have to scroll past a transcode to
    // learn which is which, and the numbers only differ by the thing
    // that is off-screen.
    println!("  {}", tile.provenance());
    for a in arms::DISPLAY_ORDER {
        if !phase.has(a) {
            continue;
        }
        let (lo, md, hi) = spread(times[a].clone());
        let b = bytes_of(a);
        println!(
            "  {:<22}{:>9.3}{:>9.3}{:>9.3}{:>9.2}{:>9.3}{:>9.0}",
            arms::DISPLAY_NAMES[a],
            lo * 1e3,
            md * 1e3,
            hi * 1e3,
            b as f64 / 1e9,
            bpw(a),
            b as f64 / lo / 1e9
        );
    }
    println!("  {}", "-".repeat(80));
    if phase.has(arms::TETRA48) && tetra_pad > 0 {
        println!(
            "  tetra48 also HOLDS {:.1} MB of row landing pad, NOT billed above — the \
             convention\n  the slot arm's 20-byte pad and the planes arm's 4-byte pad \
             already follow.\n  Real VRAM, not a rate: `stride_u32` rounds each row to \
             eight bytes so the last\n  block's six-byte window cannot fault.",
            tetra_pad as f64 / 1e6,
        );
    }
    for a in arms::DISPLAY_ORDER {
        if a == arms::FP16 || !phase.has(a) {
            continue;
        }
        let (lo, md, hi) = spread(per_round(t_f16, &times[a]));
        // BOTH competitors carry the warning, and for the same
        // reason: their × against FP16 mechanically rewards whoever
        // reads the least. QTIP, at 2.0000 b/weight, is the extreme
        // case — its byte bound is 8.00× where ours is 3.33× — so a
        // bare ratio would be the most misleading one in the table.
        if a == arms::AWQ || a == arms::QTIP {
            println!(
                "  {:<16} vs FP16     : {md:.2}× [{lo:.2}–{hi:.2}]  \
                 (COMPETITOR — reads {:.3} b/weight;\n  the comparable quantity \
                 is the GB/s, not this ratio)",
                arms::DISPLAY_NAMES[a],
                bpw(a)
            );
        } else {
            println!(
                "  {:<16} vs FP16     : {md:.2}× [{lo:.2}–{hi:.2}]",
                arms::DISPLAY_NAMES[a]
            );
        }
    }
    // The arm-against-arm ratios: `num` against `den`, formed round by
    // round — >1 = `num` faster. They exist only if BOTH arms ran in
    // this phase.
    let pair = |num: usize, den: usize, note: &str| {
        if phase.has(num) && phase.has(den) {
            let (lo, md, hi) = spread(per_round(&times[den], &times[num]));
            println!(
                "  {:<10} vs {:<10}: {md:.2}× [{lo:.2}–{hi:.2}]  {note}",
                arms::ARM_NAMES[num],
                arms::ARM_NAMES[den]
            );
        }
    };
    pair(
        arms::PLANES14,
        arms::SLOT32,
        "(>1 = Planes14 faster, same content decoded)",
    );
    pair(
        arms::PLANES12X,
        arms::PLANES14,
        "(>1 = Planes12x faster; memset + correction included, same exact y)",
    );
    pair(
        arms::GOLAY70V1,
        arms::PLANES12X,
        "(>1 = Golay70 v1 faster; memset + correction included, same exact y)",
    );
    pair(
        arms::GOLAY70V2,
        arms::GOLAY70V1,
        "(>1 = v2 faster; SAME buffers, same exact y — the ratio of the \
         v2 campaign)",
    );
    // F1d's gate reads THIS line, and it is the one omission on the
    // printing path that would have been silent: every other consumer
    // of an arm picks it up from `DISPLAY_ORDER` automatically, and
    // `pair` is a closure called at four fixed sites that iterate
    // nothing. Without the fifth call the bench would build the Tetra
    // buffers, verify the arm against its own f64 reference, dispatch
    // it in every round, print its table row and its vs-FP16 line —
    // and never print the ratio the whole step exists to measure.
    if phase.has(arms::TETRA48) && phase.has(arms::PLANES14) {
        let (nt, np) = (covered(arms::TETRA48), covered(arms::PLANES14));
        pair(
            arms::TETRA48,
            arms::PLANES14,
            match nt == np {
                true => "(>1 = Tetra faster; DIFFERENT FILES, same 252 matrices — a layout ratio)",
                false => "(>1 = Tetra faster — see the caveat below)",
            },
        );
        // 🕳️ Printed unconditionally, and FALSE on the file F1d runs.
        // The prereg chose the PURE Tetra file precisely so the two
        // arms cover the same 252 matrices; a warning saying they do
        // not would travel into the dossier attached to the one line
        // the throughput gate reads, and a valid equal-support ratio
        // would be discounted for a reason that does not exist.
        if nt != np {
            println!(
                "    ⚠️ NOT the same support: Tetra launched over {nt} matrices, \
                 Planes14 over {np}.\n       The {} missing are int4 g128 in the Tetra \
                 file and are read by another kernel,\n       so this ratio is formed \
                 over unequal passes. b/weight is per arm and divides\n       by what \
                 that arm read.",
                np.saturating_sub(nt)
            );
        }
    }
    println!("\n  source: {source}");
    let light = arms::DISPLAY_ORDER
        .into_iter()
        .filter(|&a| {
            // `bytes_of` is 0 for an arm whose buffers were not built
            // — the synthetic path, or a run given no Tetra file — and
            // 0 wins a `min_by_key` outright. The paragraph below then
            // calls it "the lightest arm" and divides the L2 by it.
            //
            // 🕳️ And the FLOOR wins it whenever it is selected: `nullk`
            // reads no weight by definition, so its bytes are the tail
            // and the row scales alone — ~72 MB against Planes14's
            // ~2.1 GB. The paragraph would then report 0.8× the L2 and
            // condemn a run that measures DRAM perfectly well, on the
            // arm that was built to read nothing. It is excluded for
            // the reason FP16 and AWQ are: the sentence is about the
            // LLVQ arms' traffic.
            a != arms::FP16 && a != arms::AWQ && a != arms::NULLK
                && phase.has(a) && bytes_of(a) > 0
        })
        .min_by_key(|&a| bytes_of(a));
    if let Some(la) = light {
        let light = bytes_of(la);
        println!(
            "  {:.0} MB distinct per pass on the lightest arm ({}), which is \
             {:.1}× the L2 read.\n  Under 1× we would be measuring the cache and not \
             the DRAM — the trap that\n  made every LLVQ measurement before 2026-07-31 \
             optimistic.",
            light as f64 / 1e6,
            arms::DISPLAY_NAMES[la],
            light as f64 / dev.l2_bytes as f64
        );
    }
    if std::env::args().nth(1).is_none() {
        println!(
            "  WARNING: SYNTHETIC blocks, drawn uniformly on the ball m ≤ 13. The class\n  \
             mixture of a real artifact is not exercised, so the group strides of the\n  \
             Slot32 arm and the byte traffic differ from the published model. This report\n  \
             measures the KERNELS — pass the .llvq path as an argument for the model."
        );
    }
    // The tied lm_head, read once per token by every arm at the FP16
    // arm's measured rate — the constant that caps every ratio.
    let head_bytes = 389_070_848f64 * 2.0;
    let fb = bytes_of(arms::FP16) as f64;
    let (f_lo, _, _) = spread(t_f16.clone());
    let head_s = head_bytes / (fb / f_lo);
    let with_head: Vec<String> = arms::DISPLAY_ORDER
        .into_iter()
        .filter(|&a| a != arms::FP16 && a != arms::AWQ && phase.has(a))
        .map(|a| {
            let (lo, _, _) = spread(times[a].clone());
            format!("{} {:.2}×", arms::DISPLAY_NAMES[a], (f_lo + head_s) / (lo + head_s))
        })
        .collect();
    if !with_head.is_empty() {
        println!(
            "\n  With the unquantized f16 lm_head ({:.0} M weights, {:.2} ms at the \
             measured FP16\n  throughput, added to the LLVQ arms): {}.\n  Norms, \
             activations, attention and rotation are measured neither here nor there.",
            389_070_848f64 / 1e6,
            head_s * 1e3,
            with_head.join(", ")
        );
    }
    // 🚨 Hard rule 4 wants the same-head ratio BESIDE the raw one, and
    // hard rule 7 wants it formed round by round, never as a quotient
    // of two minima. The paragraph above gives neither for a PAIR: its
    // entries are per-arm × against FP16, and each is built from
    // `spread(...).0`, a minimum. For F1d's headline — the one line a
    // kill is read off — the two rules are met here explicitly.
    if phase.has(arms::TETRA48) && phase.has(arms::PLANES14) {
        // 🚨 TWO corrections, and this repository already uses the
        // phrase "same head" for the SECOND one. The F1 journals mean
        // *the launch floor removed* by it (`f1rankfloor` prints
        // "(v3g−nullk)/(planes14−nullk) … the SAME-HEAD ratio, launch
        // floor removed (rule 4)"), while this file's own paragraph
        // means *the f16 lm_head added*. They are different
        // quantities and they move in opposite directions: the floor
        // is common to both arms and pushes any ratio TOWARD 1, the
        // head is common too and pushes it toward 1 as well, but from
        // a different baseline. Printing one of them under the other's
        // name is how a dossier ends up comparing two things.
        //
        // So both are printed, each named for what it does, and both
        // formed round by round — never a quotient of two medians,
        // which is what a reader would otherwise compute off the table
        // above and which hard rule 7 forbids.
        let add = |v: &[f64], d: f64| -> Vec<f64> { v.iter().map(|t| t + d).collect() };
        let (lo, md, hi) = spread(per_round(
            &add(&times[arms::PLANES14], head_s),
            &add(&times[arms::TETRA48], head_s),
        ));
        println!(
            "  tetra48 vs planes14, WITH THE F16 HEAD : {md:.2}× [{lo:.2}–{hi:.2}]  \
             (the lm_head added to BOTH arms, round by round)"
        );
        if phase.has(arms::NULLK) {
            // The F1 dossier's same-head ratio, the one `f1rankfloor`
            // prints: the launch floor removed from BOTH arms, round
            // by round, so this run's number and that bench's are the
            // same function of the same measurement.
            let sub = |v: &[f64]| -> Vec<f64> {
                v.iter()
                    .zip(&times[arms::NULLK])
                    .map(|(t, f)| (t - f).max(f64::MIN_POSITIVE))
                    .collect()
            };
            let (lo, md, hi) = spread(per_round(
                &sub(&times[arms::PLANES14]),
                &sub(&times[arms::TETRA48]),
            ));
            println!(
                "  tetra48 vs planes14, FLOOR REMOVED : {md:.2}× [{lo:.2}–{hi:.2}]  \
                 (nullk subtracted from BOTH, round by round — the F1 journals' \
                 \"same head\")"
            );
        }
    }
    println!(
        "\n  WARNING: NEVER compare this line by line with the Metal figure, and never\n  \
         subtract it from a bin/matvec run: other rounds, other NVRTC translation unit.\n  \
         Every arm against its own f64 reference; the ratios form round by round,\n  \
         inside one process."
    );
}
