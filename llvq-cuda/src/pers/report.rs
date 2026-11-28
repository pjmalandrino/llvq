//! 4E's token and stamp symbols and its phase report: what a stamped launch
//! measured, read from its stamps. One file because the report reads the
//! event layout the symbols write.

use super::med_range;

/// The token kernel's symbol for a RoPE form and a block size.
pub fn p4e_symbol(form: u32, threads: u32) -> Result<&'static str, String> {
    Ok(match (form, threads) {
        (0, 768) => "pers_4e_tokens_f0",
        (1, 768) => "pers_4e_tokens_f1",
        (2, 768) => "pers_4e_tokens_f2",
        (0, 512) => "pers_4e_tokens_f0_t512",
        (1, 512) => "pers_4e_tokens_f1_t512",
        (2, 512) => "pers_4e_tokens_f2_t512",
        _ => return Err(format!("4E: no kernel for RoPE form {form} at {threads} threads")),
    })
}

/// The phase-stamp symbol for a RoPE form and a block size: the same body
/// and the same rows' variant as the token symbol of that size.
pub fn p4e_stamp_symbol(form: u32, threads: u32) -> Result<&'static str, String> {
    Ok(match (form, threads) {
        (0, 512) => "pers_4e_stamps_f0_t512",
        (1, 512) => "pers_4e_stamps_f1_t512",
        (2, 512) => "pers_4e_stamps_f2_t512",
        (0, 768) => "pers_4e_stamps_f0_t768",
        (1, 768) => "pers_4e_stamps_f1_t768",
        (2, 768) => "pers_4e_stamps_f2_t768",
        _ => return Err(format!("4E stamps: no stamp symbol for RoPE form {form} at {threads} threads")),
    })
}

/// Whether the 4E symbols of a block size keep the rows' next block or word
/// in flight: 512 threads yes, 768 no (its registers would spill).
pub fn p4e_rows_prefetch(threads: u32) -> bool {
    threads == 512
}

/// Stamp events of one 4E token (`pers_4e.cu`, STAMP = true): the
/// embedding's arrival and release, fifteen a layer in the order they happen
/// ([`P4E_LAYER`]), the head's end.
pub fn p4e_events(layers: u32) -> u32 {
    3 + 15 * layers
}

/// A layer's stamp events, offsets from its first (`pers_4e.cu`): q/k/v's
/// ln1 and rotation done, then each grid wait's arrival and release, with the
/// MLP's four inner points before its own wait.
pub const P4E_LAYER: [&str; 15] = [
    "q/k/v ln1+rotation done", "q/k/v arrival", "q/k/v release", "norm arrival", "norm release",
    "attention arrival", "attention release", "o arrival", "o release", "MLP ln2+rotation done",
    "MLP gate/up done", "MLP inner release", "MLP down input ready", "MLP arrival", "MLP release",
];

/// The parts of q/k/v and MLP the inner stamps split them into, as
/// [`p4e_phase_report`] names them; each is CTA work, no grid wall.
pub const P4E_SUBS: [&str; 7] = [
    "q/k/v: ln1+rotation", "q/k/v: rows", "MLP: ln2+rotation", "MLP: gate/up rows", "MLP: inner wait",
    "MLP: down input", "MLP: down rows",
];

/// The phases of a 4E token, in order, as [`p4e_phase_report`] names them.
const P4E_PHASES: [&str; 7] = ["embedding", "q/k/v", "norm/RoPE/append", "attention", "o", "MLP", "head"];

/// One phase of [`P4eReport`], summed over the layers of a token. Each is
/// the median over the tokens read, with its range.
#[derive(Clone, Debug, PartialEq)]
pub struct P4ePhase {
    pub name: &'static str,
    /// How far the grid's last release moved: the phase's share of the token.
    pub wall: (f64, f64, f64),
    /// The slowest CTA's work, from its own release to its own arrival.
    pub crit: (f64, f64, f64),
    /// The mean CTA's work.
    pub mean: (f64, f64, f64),
    /// The CTA that was slowest most often in this phase, and how often.
    pub crit_cta: (u32, f64),
}

/// [`p4e_phase_report`]'s reading of a stamped launch.
#[derive(Clone, Debug, PartialEq)]
pub struct P4eReport {
    pub phases: Vec<P4ePhase>,
    /// [`P4E_SUBS`], from the inner stamps: their `wall` is unread (zeros).
    pub subs: Vec<P4ePhase>,
    /// Release to release, the head's end of one token to the next's.
    pub token: (f64, f64, f64),
    pub tokens_read: usize,
    /// Smallest positive step between two stamps: the clock's grain.
    pub grain: u64,
}

/// Time per phase from the stamps of one launch of `n` tokens,
/// `[n][p4e_events(layers)][grid]`, in the clock's unit (ns on a card).
///
/// Refused, by name, when the stamps contradict the barriers they bracket: a
/// release before an arrival of the same wait on any CTA, or a CTA's stamps
/// going back within the launch. Token 0 has no previous end and is not read,
/// so `n` must be at least 2. The phases of a token start at the previous
/// release (the head's end for the embedding) and end at the arrival; the
/// head, whose W4 is inside it, ends at its release.
pub fn p4e_phase_report(stamps: &[u64], n: u32, layers: u32, grid: u32) -> Result<P4eReport, String> {
    let (ev, g) = (p4e_events(layers) as usize, grid as usize);
    if n < 2 || stamps.len() != n as usize * ev * g {
        return Err(format!("4E stamps: {} stamps for {n} tokens x {ev} events x {grid} CTAs (n >= 2)", stamps.len()));
    }
    let at = |t: usize, e: usize, c: usize| stamps[(t * ev + e) * g + c];
    for c in 0..g {
        let mut prev = 0u64;
        for t in 0..n as usize {
            for e in 0..ev {
                let v = at(t, e, c);
                if v == 0 || v < prev {
                    return Err(format!("4E stamps: CTA {c}, token {t}, event {e}: {v} after {prev}"));
                }
                prev = v;
            }
        }
    }
    // Every wait: its last arrival before its first release (the MLP's inner
    // wait too: its gate/up-done stamp is each CTA's arrival).
    let mut waits = vec![0usize];
    for l in 0..layers as usize {
        let b = 2 + 15 * l;
        waits.extend([b + 1, b + 3, b + 5, b + 7, b + 10, b + 13]);
    }
    for t in 0..n as usize {
        for &a in &waits {
            let last_arrival = (0..g).map(|c| at(t, a, c)).max().unwrap();
            let first_release = (0..g).map(|c| at(t, a + 1, c)).min().unwrap();
            if first_release < last_arrival {
                return Err(format!("4E stamps: token {t}, event {a}: a release at {first_release} before an arrival at {last_arrival}"));
            }
        }
    }
    let mut sorted: Vec<u64> = stamps.to_vec();
    sorted.sort_unstable();
    let grain = sorted.windows(2).map(|w| w[1] - w[0]).filter(|&d| d > 0).min().unwrap_or(0);
    let (k, ks) = (P4E_PHASES.len(), P4E_SUBS.len());
    let (mut wall, mut crit, mut mean) = (vec![Vec::new(); k + ks], vec![Vec::new(); k + ks], vec![Vec::new(); k + ks]);
    let mut crit_ctas = vec![vec![0u32; g]; k + ks];
    let mut token = Vec::new();
    let head_end = ev - 1;
    for t in 1..n as usize {
        let (mut w, mut cr, mut mn) = (vec![0.0; k + ks], vec![0.0; k + ks], vec![0.0; k + ks]);
        // (kind, start event or None for the previous token's end, end event,
        // whether the kind has a grid wall: the phases do, the parts do not)
        let mut segs: Vec<(usize, Option<usize>, usize, bool)> = vec![(0, None, 0, true)];
        let mut rel = 1usize;
        for l in 0..layers as usize {
            let b = 2 + 15 * l;
            segs.push((1, Some(rel), b + 1, true));
            segs.push((k, Some(rel), b, false));
            segs.push((k + 1, Some(b), b + 1, false));
            segs.push((2, Some(b + 2), b + 3, true));
            segs.push((3, Some(b + 4), b + 5, true));
            segs.push((4, Some(b + 6), b + 7, true));
            segs.push((5, Some(b + 8), b + 13, true));
            for (i, e) in (b + 8..b + 13).enumerate() {
                segs.push((k + 2 + i, Some(e), e + 1, false));
            }
            rel = b + 14;
        }
        segs.push((6, Some(rel), head_end, true));
        for (kind, start, end, walled) in segs {
            let s_of = |c: usize| match start {
                Some(e) => at(t, e, c),
                None => at(t - 1, head_end, c),
            };
            if walled {
                // the grid's release that ends this phase: its end event's release
                let end_rel = if end == head_end { end } else { end + 1 };
                let grid_from = (0..g).map(s_of).max().unwrap();
                let grid_to = (0..g).map(|c| at(t, end_rel, c)).max().unwrap();
                w[kind] += (grid_to - grid_from) as f64;
            }
            let works: Vec<u64> = (0..g).map(|c| at(t, end, c) - s_of(c)).collect();
            // the slowest CTA, the lowest index on a tie
            let (ci, cw) = works.iter().enumerate().fold((0usize, 0u64), |b, (i, &v)| if v > b.1 { (i, v) } else { b });
            cr[kind] += cw as f64;
            mn[kind] += works.iter().sum::<u64>() as f64 / g as f64;
            crit_ctas[kind][ci] += 1;
        }
        for i in 0..k + ks {
            wall[i].push(w[i]);
            crit[i].push(cr[i]);
            mean[i].push(mn[i]);
        }
        let t_end = (0..g).map(|c| at(t, head_end, c)).max().unwrap();
        let t_prev = (0..g).map(|c| at(t - 1, head_end, c)).max().unwrap();
        token.push((t_end - t_prev) as f64);
    }
    let mut all: Vec<P4ePhase> = (0..k + ks)
        .map(|i| {
            let total: u32 = crit_ctas[i].iter().sum();
            let (cta, hits) = crit_ctas[i].iter().enumerate().fold((0usize, 0u32), |b, (c, &h)| if h > b.1 { (c, h) } else { b });
            P4ePhase {
                name: if i < k { P4E_PHASES[i] } else { P4E_SUBS[i - k] },
                wall: med_range(&wall[i]),
                crit: med_range(&crit[i]),
                mean: med_range(&mean[i]),
                crit_cta: (cta as u32, hits as f64 / total.max(1) as f64),
            }
        })
        .collect();
    let subs = all.split_off(k);
    Ok(P4eReport { phases: all, subs, token: med_range(&token), tokens_read: n as usize - 1, grain })
}

/// [`P4eReport`] as printed lines, the clock in ns: µs a token per phase.
pub fn p4e_report_lines(r: &P4eReport) -> Vec<String> {
    let us = |v: (f64, f64, f64)| format!("{:8.1} [{:.1}-{:.1}]", v.0 / 1e3, v.1 / 1e3, v.2 / 1e3);
    let mut out = vec![format!(
        "  phase stamps: {} tokens read, clock grain {} ns; us a token, median [range] over tokens",
        r.tokens_read, r.grain
    )];
    out.push(format!("    {:<18} {:>24} {:>24} {:>24} {:>6} {:>12}", "phase", "wall", "slowest CTA's work", "mean CTA's work", "share", "slowest CTA"));
    for p in &r.phases {
        out.push(format!(
            "    {:<18} {} {} {} {:5.1}% {:>5} ({:.0}%)",
            p.name,
            us(p.wall),
            us(p.crit),
            us(p.mean),
            100.0 * p.wall.0 / r.token.0,
            p.crit_cta.0,
            100.0 * p.crit_cta.1
        ));
    }
    out.push(format!("    {:<18} {}", "token", us(r.token)));
    out.push("    inside q/k/v and MLP (CTA work, no wall)".to_string());
    for p in &r.subs {
        out.push(format!("      {:<22} {} {} {:>5} ({:.0}%)", p.name, us(p.crit), us(p.mean), p.crit_cta.0, 100.0 * p.crit_cta.1));
    }
    out
}
