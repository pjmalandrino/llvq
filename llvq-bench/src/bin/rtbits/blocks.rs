//! The per-block accumulators of a v1 sweep: the payload totals, the
//! grouped-32 layout, the levels of a gaussian point, and the whole-model
//! mean they feed.

use super::*;
/// Bits per parameter of the **whole model**: a weighted mean of `(count,
/// bits per parameter)` groups over their total count.
///
/// The only quantity comparable to an AWQ figure, and the one the dossier
/// got wrong twice by quoting a payload rate against a whole-model one. It
/// takes a list rather than two arguments because a real model has three
/// groups, not two — quantized linears, embedding tables, and the norms that
/// stay `f16` whatever is done to the embedding. Every count is printed
/// beside the result so the division is auditable.
pub(super) fn model_bpw(groups: &[(u64, f64)]) -> f64 {
    let (bits, n) = groups
        .iter()
        .fold((0.0, 0u64), |(b, n), &(c, r)| (b + c as f64 * r, n + c));
    bits / n as f64
}

#[derive(Default)]
pub(super) struct Acc {
    pub(super) blocks: u64,
    pub(super) mask: u64,
    pub(super) pos: u64,
    pub(super) levels: [u64; MAX_LEVELS + 1],
    pub(super) worst_mask: u64,
    pub(super) worst_pos: u64,
}

impl Acc {
    pub(super) fn push(&mut self, w: &Widths) {
        self.blocks += 1;
        self.mask += w.mask;
        self.pos += w.pos;
        self.levels[w.levels] += 1;
        self.worst_mask = self.worst_mask.max(w.mask);
        self.worst_pos = self.worst_pos.max(w.pos);
    }
}

/// Tracks the grouped-32 layout: every lane of a group reads the group's max
/// width, one u32 base pointer per group.
#[derive(Default)]
pub(super) struct Grouped {
    filled: usize,
    max_mask: u64,
    max_pos: u64,
    max_slot: u64,
    max_levels: usize,
    /// Does this group hold a block of exactly 4 levels? Under an `L ≤ 4`
    /// cap such a group's stride is *exactly* 14 bytes — see `close`.
    has_four: bool,
    /// Origin blocks seen in the group. When every lane is the origin the
    /// group's max width is the 10-bit header, which the general
    /// `9 + gain + 24·L` rule does not describe.
    origins: usize,
    pub(super) sum_levels: u64,
    /// Totals over closed groups.
    pub(super) mask_bits: u64,
    pub(super) pos_bits: u64,
    /// Same stride, rounded up to a whole byte — what lanes can actually
    /// address.
    pub(super) mask_bits_byte: u64,
    /// `Slot32`, byte-rounded stride: the production format's real cost.
    pub(super) slot_bits_byte: u64,
    pub(super) max_levels_sum: u64,
    /// Groups by their widest block's level count — the distribution the
    /// mean of maxima hides, and what decides the capped rate.
    pub(super) maxl_hist: [u64; MAX_LEVELS + 1],
    pub(super) groups_with_four: u64,
    /// Groups in which every lane is the origin, excluded from `maxl_hist`
    /// so the histogram cross-check stays a closed identity.
    pub(super) origin_only_groups: u64,
    pub(super) groups: u64,
}

impl Grouped {
    pub(super) fn push(&mut self, w: &Widths) {
        self.filled += 1;
        self.max_mask = self.max_mask.max(w.mask);
        self.max_pos = self.max_pos.max(w.pos);
        self.max_slot = self.max_slot.max(w.slot);
        self.max_levels = self.max_levels.max(w.levels);
        self.has_four |= w.levels == 4;
        self.origins += usize::from(w.origin);
        self.sum_levels += w.levels as u64;
        if self.filled == GROUP {
            self.close();
        }
    }

    pub(super) fn close(&mut self) {
        if self.filled == 0 {
            return;
        }
        // A partial trailing group still pays full-width lanes.
        self.mask_bits += GROUP as u64 * self.max_mask + 32;
        self.pos_bits += GROUP as u64 * self.max_pos + 32;
        self.mask_bits_byte += GROUP as u64 * self.max_mask.div_ceil(8) * 8 + 32;
        self.slot_bits_byte += GROUP as u64 * self.max_slot.div_ceil(8) * 8 + 32;
        self.max_levels_sum += self.max_levels as u64;
        if self.origins == self.filled {
            self.origin_only_groups += 1;
        } else {
            self.maxl_hist[self.max_levels] += 1;
        }
        self.groups_with_four += u64::from(self.has_four);
        self.groups += 1;
        self.filled = 0;
        self.max_mask = 0;
        self.max_pos = 0;
        self.max_slot = 0;
        self.max_levels = 0;
        self.has_four = false;
        self.origins = 0;
    }
}

/// Levels of a raw lattice point, canonical order — for the gaussian mode,
/// where blocks come from the searcher rather than from an artifact.
pub(super) fn levels_of_point(p: &[i32; DIM]) -> ClassLevels {
    let mut kinds: Vec<(i32, u8)> = Vec::new();
    for &v in p {
        let a = v.abs();
        match kinds.iter_mut().find(|(u, _)| *u == a) {
            Some((_, n)) => *n += 1,
            None => kinds.push((a, 1)),
        }
    }
    kinds.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(b.0.cmp(&a.0)));
    let mut lv = ClassLevels {
        len: kinds.len(),
        odd: p.iter().all(|v| v % 2 != 0),
        shell: 0, // unused here
        ..Default::default()
    };
    for (i, &(v, n)) in kinds.iter().enumerate() {
        lv.values[i] = v;
        lv.counts[i] = n;
        if v != 0 {
            lv.nonzero += n;
        }
    }
    lv
}
