//! `LLVQ_BOUNDARY_AB`, the portable half: the arms, their row splits, their
//! boundary counts and the round-by-round arithmetic.
//!
//! The mode times the decode cost of one added kernel boundary inside the
//! served 4B, by class (prereg-persistent-0.md, process 2b). Every arm adds
//! boundaries and changes no arithmetic, so all eight return the same tokens.
//! The card half lives in `fused_cuda` (the split launches, the empty kernel)
//! and in `bin/fusedrun` (the rounds). What can be wrong without a card is
//! here, with its tests: a split point off the 8-row grid, a count that does
//! not match its arm, a per-boundary cost formed from medians.

use std::sync::atomic::{AtomicU64, Ordering};

/// Rows one CTA of `tv_tetra48_h` computes: 256 threads, one warp a row.
pub const ROWS_PER_CTA: u32 = 8;

/// `gate_proj` rows of Qwen3-4B (its `intermediate_size`).
pub const GATE_D_OUT_4B: u32 = 9728;

/// `down_proj` rows of Qwen3-4B (its `hidden_size`).
pub const DOWN_D_OUT_4B: u32 = 2560;

/// Decoder layers of Qwen3-4B.
pub const LAYERS_4B: usize = 36;

/// `down_proj` records the sealed 4B serves as Tetra; the other 12 are int4.
pub const TETRA_DOWN_4B: u32 = 24;

/// `split@wave`: the first launch covers 852 CTAs, one full wave of the L40S
/// at six resident CTAs an SM (142 × 6), and the second the 364 left.
pub const WAVE_ROWS: u32 = 6816;

/// `split@half`: 608 + 608 CTAs.
pub const HALF_ROWS: u32 = 4864;

/// `split-down`: the Tetra `down_proj` as 160 + 160 CTAs.
pub const DOWN_HALF_ROWS: u32 = 1280;

/// CTAs of the `emptygrid` launch: the same wave as `split@wave`'s first half.
pub const EMPTY_GRID_CTAS: u32 = 852;

/// Threads of every empty CTA, the served kernel's block.
pub const EMPTY_THREADS: u32 = 256;

/// The empty kernel. Compiled as its own NVRTC module, so the served unit
/// keeps its bytes and its sha256.
pub const EMPTY_SRC: &str = r#"extern "C" __global__ void llvq_empty(unsigned int u) { (void)u; }"#;

/// The entry point of [`EMPTY_SRC`].
pub const EMPTY_FN: &str = "llvq_empty";

/// One arm of `LLVQ_BOUNDARY_AB`. The discriminant is what the process-wide
/// flag in `fused_cuda` stores.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum BoundaryArm {
    /// Nothing added.
    Served = 0,
    /// Each `gate_proj` as rows `[0, 6816)` then `[6816, 9728)`.
    SplitWave = 1,
    /// Each `gate_proj` as `[0, 4864)` then `[4864, 9728)`.
    SplitHalf = 2,
    /// Each Tetra `down_proj` as `[0, 1280)` then `[1280, 2560)`.
    SplitDown = 3,
    /// One 1-CTA empty kernel after each `down_proj`, Tetra and int4.
    Empty1 = 4,
    /// Four of them.
    Empty4 = 5,
    /// One 852-CTA empty grid after each `down_proj`.
    EmptyGrid = 6,
    /// One candle affine on a 1-element f16 tensor after each layer.
    CandleTiny = 7,
}

impl BoundaryArm {
    /// Every arm, in the order each round runs them. `Served` first: it is
    /// the reference every other arm is subtracted from.
    pub const ALL: [BoundaryArm; 8] = [
        BoundaryArm::Served,
        BoundaryArm::SplitWave,
        BoundaryArm::SplitHalf,
        BoundaryArm::SplitDown,
        BoundaryArm::Empty1,
        BoundaryArm::Empty4,
        BoundaryArm::EmptyGrid,
        BoundaryArm::CandleTiny,
    ];

    /// The arm a flag value names. Any value the flag was never set to is
    /// refused rather than read as `Served`.
    pub fn from_u8(v: u8) -> Option<Self> {
        Self::ALL.into_iter().find(|a| *a as u8 == v)
    }

    pub fn name(self) -> &'static str {
        match self {
            BoundaryArm::Served => "served",
            BoundaryArm::SplitWave => "split@wave",
            BoundaryArm::SplitHalf => "split@half",
            BoundaryArm::SplitDown => "split-down",
            BoundaryArm::Empty1 => "empty x1",
            BoundaryArm::Empty4 => "empty x4",
            BoundaryArm::EmptyGrid => "emptygrid x1",
            BoundaryArm::CandleTiny => "candle-tiny x1",
        }
    }

    /// Boundaries this arm adds to one decode step of the sealed 4B.
    pub fn boundaries_per_token(self) -> u32 {
        let layers = LAYERS_4B as u32;
        match self {
            BoundaryArm::Served => 0,
            BoundaryArm::SplitWave | BoundaryArm::SplitHalf => layers,
            BoundaryArm::SplitDown => TETRA_DOWN_4B,
            BoundaryArm::Empty1 | BoundaryArm::EmptyGrid | BoundaryArm::CandleTiny => layers,
            BoundaryArm::Empty4 => 4 * layers,
        }
    }

    /// The row where this arm splits the projection `name`, if it splits it.
    /// `name` is a record name such as `model.layers.3.mlp.gate_proj.weight`.
    pub fn split_at(self, name: &str) -> Option<u32> {
        match (self, proj_type(name)) {
            (BoundaryArm::SplitWave, "gate_proj") => Some(WAVE_ROWS),
            (BoundaryArm::SplitHalf, "gate_proj") => Some(HALF_ROWS),
            (BoundaryArm::SplitDown, "down_proj") => Some(DOWN_HALF_ROWS),
            _ => None,
        }
    }

    /// `(launches, CTAs a launch)` of empty kernels this arm adds after a
    /// projection named `name`: after every `down_proj`, nothing elsewhere.
    pub fn probe_after(self, name: &str) -> Option<(u32, u32)> {
        if proj_type(name) != "down_proj" {
            return None;
        }
        match self {
            BoundaryArm::Empty1 => Some((1, 1)),
            BoundaryArm::Empty4 => Some((4, 1)),
            BoundaryArm::EmptyGrid => Some((1, EMPTY_GRID_CTAS)),
            _ => None,
        }
    }

    /// Whether the model runs one candle op after each layer under this arm.
    pub fn candle_probe(self) -> bool {
        self == BoundaryArm::CandleTiny
    }
}

/// The projection type of a record name: `gate_proj` for
/// `model.layers.3.mlp.gate_proj.weight`, with or without the `.weight`.
pub fn proj_type(name: &str) -> &str {
    let n = name.strip_suffix(".weight").unwrap_or(name);
    n.rsplit('.').next().unwrap_or(n)
}

/// Refuses a split point the kernel cannot honour bit for bit: outside
/// `(0, d_out)`, or off the 8-row grid one CTA covers.
pub fn check_split(at: u32, d_out: usize) -> Result<(), String> {
    if at == 0 || at as usize >= d_out {
        return Err(format!("split at row {at}: outside (0, {d_out})"));
    }
    if !at.is_multiple_of(ROWS_PER_CTA) {
        return Err(format!(
            "split at row {at}: not a multiple of {ROWS_PER_CTA}, the rows one CTA computes"
        ));
    }
    Ok(())
}

/// The two launches of a split, `(first row, rows)` each. They partition
/// `[0, d_out)` in order.
pub fn split_ranges(at: u32, d_out: usize) -> Result<[(u32, u32); 2], String> {
    check_split(at, d_out)?;
    Ok([(0, at), (at, d_out as u32 - at)])
}

/// Refuses a model whose shapes are not the 4B's: the split points and the
/// boundary counts are the 4B's own.
pub fn check_4b_shape(layers: usize, hidden: usize, intermediate: usize) -> Result<(), String> {
    if (layers, hidden, intermediate)
        != (LAYERS_4B, DOWN_D_OUT_4B as usize, GATE_D_OUT_4B as usize)
    {
        return Err(format!(
            "LLVQ_BOUNDARY_AB: {layers} layers, hidden {hidden}, intermediate {intermediate}. \
             Its split points and boundary counts are Qwen3-4B's ({LAYERS_4B}, {DOWN_D_OUT_4B}, \
             {GATE_D_OUT_4B})"
        ));
    }
    Ok(())
}

/// Boundaries added since the last [`take_added`]. Written only by the arm
/// branches, never by the served path.
static ADDED: AtomicU64 = AtomicU64::new(0);

/// Counts `n` added boundaries.
pub fn note_added(n: u64) {
    ADDED.fetch_add(n, Ordering::Relaxed);
}

/// The boundaries added since the last call, and the counter back to zero.
pub fn take_added() -> u64 {
    ADDED.swap(0, Ordering::Relaxed)
}

/// The bounds a generate of `n_new` tokens from a prompt of `prompt` tokens
/// must count under `arm`. Each of the `n_new - 1` decode steps adds the
/// arm's count exactly. The prefill adds between none of it (its Tetra
/// projections run the rows kernel) and one count a prompt row.
pub fn counted_bounds(arm: BoundaryArm, n_new: usize, prompt: usize) -> (u64, u64) {
    let per = arm.boundaries_per_token() as u64;
    let steps = n_new.saturating_sub(1) as u64;
    (per * steps, per * (steps + prompt as u64))
}

/// Median, min and max of a series.
pub fn median_range(v: &[f64]) -> (f64, f64, f64) {
    assert!(!v.is_empty(), "median over zero rounds");
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let n = s.len();
    let med = if n % 2 == 1 { s[n / 2] } else { (s[n / 2 - 1] + s[n / 2]) / 2.0 };
    (med, s[0], s[n - 1])
}

/// µs per added boundary, one value a round: `(t_arm - t_served) /
/// boundaries` with `t` in seconds a token, the two arms paired by round.
pub fn per_boundary_us(served: &[f64], arm: &[f64], boundaries: u32) -> Result<Vec<f64>, String> {
    if served.len() != arm.len() || served.is_empty() {
        return Err(format!(
            "{} served rounds against {} arm rounds: the series pair round by round",
            served.len(),
            arm.len()
        ));
    }
    if boundaries == 0 {
        return Err("an arm that adds no boundary has no cost per boundary".into());
    }
    Ok(served
        .iter()
        .zip(arm)
        .map(|(s, a)| (a - s) / boundaries as f64 * 1e6)
        .collect())
}

/// The derived classes, `(name, formula)`, each a difference of two arms'
/// per-boundary series. Printed, never gated.
pub const DERIVED: [(&str, BoundaryArm, BoundaryArm); 4] = [
    ("drain+ramp", BoundaryArm::SplitWave, BoundaryArm::Empty1),
    ("underfill", BoundaryArm::SplitHalf, BoundaryArm::SplitWave),
    ("dispatch", BoundaryArm::EmptyGrid, BoundaryArm::Empty1),
    ("linearity", BoundaryArm::Empty4, BoundaryArm::Empty1),
];

/// One derived class, round by round: `per[a] - per[b]`, where `per` holds
/// each arm's per-boundary series in [`BoundaryArm::ALL`] order.
pub fn derived_series(per: &[Vec<f64>], a: BoundaryArm, b: BoundaryArm) -> Result<Vec<f64>, String> {
    let at = |x: BoundaryArm| BoundaryArm::ALL.iter().position(|&y| y == x).expect("every arm is in ALL");
    let (pa, pb) = (&per[at(a)], &per[at(b)]);
    if pa.len() != pb.len() || pa.is_empty() {
        return Err(format!(
            "{}: {} rounds against {}, and the classes pair round by round",
            a.name(),
            pa.len(),
            pb.len()
        ));
    }
    Ok(pa.iter().zip(pb).map(|(x, y)| x - y).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_split_point_sits_on_the_cta_grid_inside_its_projection() {
        let shapes = [
            ("model.layers.0.mlp.gate_proj.weight", GATE_D_OUT_4B as usize),
            ("model.layers.0.mlp.down_proj.weight", DOWN_D_OUT_4B as usize),
            ("model.layers.0.mlp.up_proj.weight", GATE_D_OUT_4B as usize),
            ("model.layers.0.self_attn.q_proj.weight", 4096),
        ];
        let mut splits = 0;
        for arm in BoundaryArm::ALL {
            for (name, d_out) in shapes {
                if let Some(at) = arm.split_at(name) {
                    splits += 1;
                    assert_eq!(at % ROWS_PER_CTA, 0, "{} on {name}", arm.name());
                    assert!(at > 0 && (at as usize) < d_out, "{} on {name}", arm.name());
                    let [(s0, n0), (s1, n1)] = split_ranges(at, d_out).unwrap();
                    assert_eq!((s0, s0 + n0, s1 + n1), (0, s1, d_out as u32));
                    assert_eq!(n1 % ROWS_PER_CTA, 0);
                }
            }
        }
        assert_eq!(splits, 3, "three split arms, one projection type each");
    }

    #[test]
    fn the_splits_are_the_prereg_ctas() {
        let ctas = |at: u32, d: u32| (at / ROWS_PER_CTA, (d - at) / ROWS_PER_CTA);
        assert_eq!(ctas(WAVE_ROWS, GATE_D_OUT_4B), (852, 364));
        assert_eq!(ctas(HALF_ROWS, GATE_D_OUT_4B), (608, 608));
        assert_eq!(ctas(DOWN_HALF_ROWS, DOWN_D_OUT_4B), (160, 160));
        assert_eq!(EMPTY_GRID_CTAS, WAVE_ROWS / ROWS_PER_CTA);
    }

    #[test]
    fn split_points_off_the_grid_or_outside_are_refused() {
        assert!(check_split(0, 9728).is_err());
        assert!(check_split(9728, 9728).is_err());
        assert!(check_split(9736, 9728).is_err());
        assert!(check_split(6812, 9728).is_err());
        assert!(check_split(6816, 9728).is_ok());
    }

    #[test]
    fn the_arms_match_by_projection_type_with_or_without_weight() {
        let a = BoundaryArm::SplitWave;
        assert_eq!(a.split_at("model.layers.7.mlp.gate_proj.weight"), Some(WAVE_ROWS));
        assert_eq!(a.split_at("model.layers.7.mlp.gate_proj"), Some(WAVE_ROWS));
        assert_eq!(a.split_at("model.layers.7.mlp.up_proj.weight"), None);
        assert_eq!(BoundaryArm::Served.split_at("model.layers.7.mlp.gate_proj.weight"), None);
        assert_eq!(BoundaryArm::Empty4.probe_after("model.layers.7.mlp.down_proj.weight"), Some((4, 1)));
        assert_eq!(BoundaryArm::Empty4.probe_after("model.layers.7.mlp.gate_proj.weight"), None);
        assert_eq!(BoundaryArm::SplitDown.probe_after("model.layers.7.mlp.down_proj.weight"), None);
        assert_eq!(
            BoundaryArm::EmptyGrid.probe_after("model.layers.7.mlp.down_proj.weight"),
            Some((1, EMPTY_GRID_CTAS))
        );
    }

    #[test]
    fn the_boundary_counts_are_the_prereg_table() {
        let counts: Vec<u32> = BoundaryArm::ALL.iter().map(|a| a.boundaries_per_token()).collect();
        assert_eq!(counts, [0, 36, 36, 24, 36, 144, 36, 36]);
        for (k, a) in BoundaryArm::ALL.into_iter().enumerate() {
            assert_eq!(a as u8 as usize, k);
            assert_eq!(BoundaryArm::from_u8(k as u8), Some(a));
        }
        assert_eq!(BoundaryArm::from_u8(8), None);
        assert_eq!(counted_bounds(BoundaryArm::Empty1, 256, 5), (36 * 255, 36 * 260));
        assert_eq!(counted_bounds(BoundaryArm::Served, 256, 5), (0, 0));
    }

    #[test]
    fn only_the_4b_shape_is_accepted() {
        assert!(check_4b_shape(36, 2560, 9728).is_ok());
        assert!(check_4b_shape(36, 4096, 12288).is_err());
        assert!(check_4b_shape(40, 5120, 17408).is_err());
    }

    /// The per-boundary series pairs the two arms ROUND BY ROUND. A version
    /// that sorted either series first, or divided a difference of medians,
    /// gives other numbers on these rounds, and fails here.
    #[test]
    fn the_cost_per_boundary_is_formed_round_by_round() {
        let served = [1.0e-3, 3.0e-3, 2.0e-3];
        let arm = [5.0e-3, 3.0e-3, 4.0e-3];
        let per = per_boundary_us(&served, &arm, 4).unwrap();
        let want = [1000.0, 0.0, 500.0];
        for (g, w) in per.iter().zip(want) {
            assert!((g - w).abs() < 1e-9, "{per:?}");
        }
        // Mutant 1: sort both series, then pair. [1,2,3] against [3,4,5].
        let mut s = served.to_vec();
        let mut a = arm.to_vec();
        s.sort_by(|x, y| x.total_cmp(y));
        a.sort_by(|x, y| x.total_cmp(y));
        let sorted = per_boundary_us(&s, &a, 4).unwrap();
        assert_ne!(median_range(&sorted).1, median_range(&per).1);
        // Mutant 2: the difference of the medians. On these rounds it is
        // 1000 us, and the median of the per-round series is 2000 us.
        let served = [1.0e-3, 2.0e-3, 3.0e-3];
        let arm = [3.0e-3, 2.0e-3, 10.0e-3];
        let per = per_boundary_us(&served, &arm, 1).unwrap();
        let of_medians = (median_range(&arm).0 - median_range(&served).0) * 1e6;
        assert!((median_range(&per).0 - 2000.0).abs() < 1e-9);
        assert!((of_medians - 1000.0).abs() < 1e-9);
        assert!(per_boundary_us(&served, &arm[..2], 1).is_err());
        assert!(per_boundary_us(&served, &arm, 0).is_err());
    }

    #[test]
    fn the_derived_classes_pair_round_by_round() {
        let mut per = vec![Vec::new(); BoundaryArm::ALL.len()];
        per[BoundaryArm::SplitWave as usize] = vec![3.0, 1.0, 2.0];
        per[BoundaryArm::Empty1 as usize] = vec![1.0, 2.0, 0.5];
        let d = derived_series(&per, BoundaryArm::SplitWave, BoundaryArm::Empty1).unwrap();
        assert_eq!(d, [2.0, -1.0, 1.5]);
        assert!(derived_series(&per, BoundaryArm::SplitHalf, BoundaryArm::SplitWave).is_err());
        for (_, a, b) in DERIVED {
            assert_ne!(a, b);
            assert_eq!(a.boundaries_per_token() > 0, b.boundaries_per_token() > 0);
        }
    }

    #[test]
    fn the_counter_is_taken_and_reset() {
        take_added();
        note_added(3);
        note_added(4);
        assert_eq!(take_added(), 7);
        assert_eq!(take_added(), 0);
    }

    #[test]
    fn the_empty_module_names_its_entry_point() {
        assert!(EMPTY_SRC.contains(&format!("void {EMPTY_FN}(")));
        assert!(EMPTY_SRC.starts_with("extern \"C\" __global__"));
    }
}
