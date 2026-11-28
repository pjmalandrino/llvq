//! The sweeps: the per-class width table, the class-major and file-order
//! group passes over it, and the mean and worst rates they report.

use super::*;

// ---------------------------------------------------------------------------
// The sweeps
// ---------------------------------------------------------------------------

/// Per-class width vectors, flattened `class-major × NVAR`, plus the width an
/// **origin** block costs each variant.
///
/// Flat rather than `Vec<Vec<_>>` on purpose: the file sweep touches this once
/// per block, 150.7 M times, and a slice of `NVAR` contiguous `u64` is one
/// cache line rather than a pointer chase.
pub(super) struct WidthTable {
    flat: Vec<u64>,
    pub(super) origin: [u64; NVAR],
    pub(super) per_class: Vec<Widths>,
}

impl WidthTable {
    pub(super) fn build(fd: &FastDecoder) -> Self {
        let cs = enumerate_classes(13);
        let mut by_key: HashMap<Key, Widths> = HashMap::new();
        for c in &cs.even {
            let free_n: u32 = c.free_vals.iter().map(|&(_, n)| u32::from(n)).sum();
            let n0 = 24 - c.w - free_n;
            let mut kinds: Vec<(i32, u8)> = c
                .word_vals
                .iter()
                .chain(&c.free_vals)
                .map(|&(v, n)| (i32::from(v), n))
                .collect();
            if n0 > 0 {
                kinds.push((0, n0 as u8));
            }
            by_key.insert(canonical(kinds, false), widths_of_even(c));
        }
        for c in &cs.odd {
            let kinds = c.vals.iter().map(|&(v, n)| (i32::from(v), n)).collect();
            by_key.insert(canonical(kinds, true), widths_of_odd(c));
        }

        // Per-class widths in FastDecoder order, so the sweep is a direct index.
        let per_class: Vec<Widths> = (0..fd.n_classes())
            .map(|ci| {
                let lv = fd.levels(ci);
                *by_key
                    .get(&key_of_levels(lv))
                    .unwrap_or_else(|| panic!("class {ci} missing from enumerate_classes: {lv:?}"))
            })
            .collect();
        let mut flat = Vec::with_capacity(per_class.len() * NVAR);
        for w in &per_class {
            for v in VARIANTS {
                flat.push((v.get)(w));
            }
        }
        // An origin block carries a class id, a gain bit and nothing else in
        // every *variable-width* runtime variant; in the archive it is still a
        // full index.
        //
        // ⚠️ And that is wrong for a **fixed-width** variant: `golay70` is a
        // flat 70-bit record whose origin would still be 70, not 10. Both
        // sealed files hold zero origin blocks, so nothing published depends on
        // this; `main` refuses by name to price a file that has any, rather
        // than repair a tariff no artifact exercises. Pinned with its direction
        // by `the_origin_tariff_is_wrong_for_fixed_width_variants`.
        let mut origin = [0u64; NVAR];
        for (k, v) in VARIANTS.iter().enumerate() {
            origin[k] = (v.get)(&Widths::default()).max(HEADER_BITS);
        }
        Self { flat, origin, per_class }
    }

    pub(super) fn row(&self, ci: usize) -> &[u64] {
        &self.flat[ci * NVAR..(ci + 1) * NVAR]
    }
}

/// The class-major sweep, from the histogram alone.
///
/// **Bit-for-bit the 2026-08-12 loop**, and deliberately so: it is the control
/// that makes every other column readable. Classes ascending, `n == 0`
/// skipped, one continuous run of lanes across class boundaries (the group
/// being built is *not* closed between classes), one close at the end.
///
/// The one deviation, inert on both sealed files: the `origins` run is
/// appended to the sweep instead of being left out of the grouping entirely.
/// The published file has **zero** origin blocks, so the control reproduces to
/// the last bit; a file that had them would otherwise have been priced with
/// its origins in the mean and absent from the stride, which is not a stream
/// anybody could read.
pub(super) fn sweep_class_major(t: &WidthTable, count: &[u64], origins: u64) -> Vec<Group> {
    let mut acc = vec![Group::default(); NVAR];
    for (ci, &n) in count.iter().enumerate() {
        if n == 0 {
            continue;
        }
        let ws = t.row(ci);
        for _ in 0..n {
            for (g, &w) in acc.iter_mut().zip(ws) {
                g.push(w);
            }
        }
    }
    for _ in 0..origins {
        for (g, &w) in acc.iter_mut().zip(&t.origin) {
            g.push(w);
        }
    }
    for g in &mut acc {
        g.close();
    }
    acc
}

/// What one pass over an artifact collects.
pub(super) struct Sweep {
    /// Blocks per class — the histogram the means and the `exact` column need.
    pub(super) count: Vec<u64>,
    /// Blocks whose index is outside the ball: the origin.
    pub(super) origins: u64,
    /// File-order groups, one accumulator per variant.
    pub(super) file: Vec<Group>,
    pub(super) shapes: Shapes,
    pub(super) matrices: u64,
    /// Matrices whose blocks-per-row is not a multiple of 32, i.e. where a
    /// group straddles two rows.
    pub(super) row_straddling: u64,
}

impl Sweep {
    pub(super) fn total(&self) -> u64 {
        self.count.iter().sum::<u64>() + self.origins
    }

    /// Fold one matrix into the histogram and into the file-order groups, then
    /// close the open group.
    ///
    /// Split out of [`sweep_file`] so the streaming glue can be exercised on
    /// **one** real matrix rather than only on the whole 981 MB archive: the
    /// per-block lookup, the group boundary and the matrix boundary all live
    /// here, and a sweep is the only other way to reach them.
    pub(super) fn push_matrix(&mut self, fd: &FastDecoder, t: &WidthTable, m: &llvq_artifact::RawMatrix) {
        self.shapes.push(m.d_out, m.d_in);
        self.matrices += 1;
        self.row_straddling += u64::from(!((m.d_in / DIM) as u64).is_multiple_of(GROUP));
        for &idx in &m.indices {
            let ws = match fd.class_of(idx) {
                Some(ci) => {
                    self.count[ci] += 1;
                    t.row(ci)
                }
                None => {
                    self.origins += 1;
                    &t.origin[..]
                }
            };
            for (g, &w) in self.file.iter_mut().zip(ws) {
                g.push(w);
            }
        }
        // A group never spans two matrices — see the module header. Closing
        // here is what makes that true, and `Group::partial` is what says how
        // much it cost.
        for g in &mut self.file {
            g.close();
        }
    }

    pub(super) fn empty(n_classes: usize) -> Self {
        Self {
            count: vec![0u64; n_classes],
            origins: 0,
            file: vec![Group::default(); NVAR],
            shapes: Shapes::default(),
            matrices: 0,
            row_straddling: 0,
        }
    }
}

/// One streaming pass over a sealed artifact: histogram **and** file-order
/// groups for every variant at once.
///
/// No `Vec` of 150.7 M widths anywhere — the widths are looked up per block
/// and folded straight into the accumulators, so the peak allocation is one
/// matrix's index vector, which `read_matrix_raw` allocates anyway.
pub(super) fn sweep_file(fd: &FastDecoder, t: &WidthTable, path: &str) -> Sweep {
    let f = File::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let mut r = BufReader::with_capacity(1 << 20, f);
    let h = llvq_artifact::read_header(&mut r).expect("valid artifact header");
    // The widths are per v1 class; a Tetra word would be filed under one.
    if let Err(e) = llvq_artifact::runtime::require_ball_kinds(h.kinds(), "radixstudy") {
        panic!(
            "{path}: a {} file (format v{}); radixstudy reads indices as v1 classes — {e}",
            h.kinds(),
            h.version
        );
    }
    let mut s = Sweep::empty(fd.n_classes());
    for _ in 0..h.matrices {
        let m = llvq_artifact::read_matrix_raw(&mut r, h.version).expect("valid matrix");
        // The record's own kind, not only the header's declared set.
        llvq_artifact::runtime::require_ball(m.kind, "radixstudy").unwrap_or_else(|e| {
            panic!(
                "{}: a {} record in a {} file — radixstudy reads indices as v1 classes; {e}",
                m.name, m.kind, h.kinds()
            )
        });
        s.push_matrix(fd, t, &m);
    }
    assert_eq!(s.matrices, u64::from(h.matrices), "matrices read ≠ matrices announced");
    s
}

/// Mean bits per block of one variant, from the histogram — the same order of
/// accumulation as the published run (classes ascending, then the origins).
pub(super) fn mean_bits(t: &WidthTable, count: &[u64], origins: u64, k: usize) -> f64 {
    let mut sum = 0f64;
    for (ci, &n) in count.iter().enumerate() {
        if n == 0 {
            continue;
        }
        sum += t.row(ci)[k] as f64 * n as f64;
    }
    sum += origins as f64 * t.origin[k] as f64;
    sum / (count.iter().sum::<u64>() + origins) as f64
}

pub(super) fn worst_bits(t: &WidthTable, count: &[u64], origins: u64, k: usize) -> u64 {
    let mut worst = if origins > 0 { t.origin[k] } else { 0 };
    for (ci, &n) in count.iter().enumerate() {
        if n > 0 {
            worst = worst.max(t.row(ci)[k]);
        }
    }
    worst
}
