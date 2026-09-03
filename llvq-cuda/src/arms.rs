//! Arm selection for `planesbench` — the `LLVQ_BENCH_ARMS` contract.
//!
//! Exists to close deviation É1 of `proofs/preregistration-2026-08-10.md` §7bis:
//! the six-arm job could not run its five-arm control in the same process,
//! because the bench had no way to leave an arm out — and a control taken
//! from another job, another image and another translation unit is exactly
//! the inter-process subtraction this repository has already had to retract.
//! The exit clause written into É1 is the specification here: *a control run
//! must be one environment variable away.*
//!
//! ## The contract
//!
//! `LLVQ_BENCH_ARMS` is a semicolon-separated list of **phases**, each a
//! comma-separated set of arm names. Unset means one phase, every arm.
//!
//! * A deselected arm **builds none of its buffers** — no transcode, no
//!   device upload, no verification, no timing, no table row. The resident
//!   STREAMS during a phase are that phase's arms' and nothing else; that
//!   is the whole point (the control must restitute the residency it claims
//!   to measure). Four constant micro-buffers (~83 KiB total: the golay70
//!   tables, the AWQ binary16 activation and its output buffer) follow the
//!   UNION of the phases rather than the current phase — disclosed in
//!   `planesbench`'s `Staged` comment rather than hidden in "exactly".
//! * The NVRTC translation unit **never changes with the selection** — every
//!   kernel is always compiled, and the register/spill report still covers
//!   all of them. Selecting arms by editing the source would make the
//!   control a different compiled object, which is the É1 mistake again.
//! * Dispatch order inside a round is the **registration order** below,
//!   whatever order the variable names the arms in. Reordering dispatch
//!   would change the measured object (`planesbench`'s ARMS comment).
//! * Each phase must contain every arm of the previous phase: buffers are
//!   built when first needed and never freed, so a shrinking phase would
//!   time its arms with a dead arm's buffers resident — a residency no
//!   published run has, and no log line would say so.
//! * `fp16` is required in every phase: it is the witness every published
//!   ratio is formed against.
//! * An unknown name is a hard error, never a silent fallback — the
//!   `LLVQ_FUSED_LAYOUT` rule: an A/B that silently runs the wrong arm is
//!   worse than one that fails. `golay70` without a version suffix is
//!   refused by name since the v2 exists.

/// Registration order — the dispatch order inside a round. Every arm added
/// since the published tables sits **after** them, in the order it was added:
/// `golay70v2` for the v2 campaign, then P4's eight
/// (`proofs/preregistration-p4-2026-08-14.md` §2.3, §2.5). An added arm must
/// never reorder the dispatch of the arms that produced a published number.
pub const ARM_NAMES: [&str; N_ARMS] = [
    // the six of the 2026-08-10 job
    "slot32",
    "planes14",
    "planes12x",
    "golay70v1",
    "fp16",
    "awq",
    // the v2 campaign
    "golay70v2",
    // P4 §2.5 — all eight still to be written on the kernel side
    "cublasf16",
    "mvkf16",
    "nullk",
    "planes14k",
    "planes12xk",
    "golay70v2k",
    "e1c14",
    "e1c12",
    // P1c/E1v — registered last again, and for the same reason: the dispatch
    // order of every arm that produced a published number must be fixed before
    // a job, not while one runs.
    "e1v",
    // F2 — the QTIP comparison arm. Registered last, same rule again: the
    // dispatch order of every arm that produced a published number is fixed
    // before a job, never while one runs.
    "qtip",
];
pub const N_ARMS: usize = 17;

pub const SLOT32: usize = 0;
pub const PLANES14: usize = 1;
pub const PLANES12X: usize = 2;
pub const GOLAY70V1: usize = 3;
pub const FP16: usize = 4;
pub const AWQ: usize = 5;
pub const GOLAY70V2: usize = 6;
/// The **publishable denominator**: f16 through cuBLAS, unhandicapped.
pub const CUBLASF16: usize = 7;
/// Matvec-k f16, ours. A **control**, never published alone — the
/// `broadcast_matmul` trap made this repo publish a ×2.03 that came from a
/// defect of its own dense arm (§2.5).
pub const MVKF16: usize = 8;
/// Same grid, same k, output written, **no weight read**. The floor.
pub const NULLK: usize = 9;
pub const PLANES14K: usize = 10;
pub const PLANES12XK: usize = 11;
pub const GOLAY70V2K: usize = 12;
pub const E1C14: usize = 13;
pub const E1C12: usize = 14;
/// The E1v stream, **row-aligned** — the only cut a warp-per-row matvec can
/// read (`llvq_artifact::e1v`, and X3's rotation argument). Measured at
/// **2.3983 kernel b/weight** on the sealed 4B's written bytes.
pub const E1V: usize = 15;
/// The **2-bit competitor**, and the one this repository's motivation was
/// inherited from without ever being reproduced: QTIP's trellis GEMV
/// (Cornell-RelaxML), ported unchanged into this harness the way `AWQ` was.
///
/// 🚨 Its source is **not in this repository** and never will be: the kernel
/// is GPL v3, this workspace is MIT OR Apache-2.0. It is fetched at job time
/// by `ops/fetch-qtip.sh` and reaches the bench through `LLVQ_QTIP_DIR` — its
/// own variable, not the `LLVQ_KERNEL_DIR` that overrides our kernels, since
/// this is an addition and not an override (`docs/qtip-provenance.md`).
pub const QTIP: usize = 16;

/// The six arms of the 2026-08-10 job — phase 1 of P4 §2.4, which reproduces
/// the published run.
pub const PHASE1: [usize; 6] = [SLOT32, PLANES14, PLANES12X, GOLAY70V1, FP16, AWQ];
/// The seven of the 08-11 job — phase 2, **the control**, the one that
/// manufactures `Δ_contrôle`.
pub const PHASE2: [usize; 7] = [SLOT32, PLANES14, PLANES12X, GOLAY70V1, FP16, AWQ, GOLAY70V2];
/// P4's eight new arms — phase 3 is `PHASE2` plus these.
pub const PHASE3_NEW: [usize; 8] = [
    CUBLASF16, MVKF16, NULLK, PLANES14K, PLANES12XK, GOLAY70V2K, E1C14, E1C12,
];

/// Which of [`ARM_NAMES`] have a kernel behind them.
///
/// 🚨 **Registering a name is not implementing an arm.** P4 §2.3 wants the
/// dispatch order fixed *before* the job — an added arm must never reorder the
/// published ones — so arms are registered while their kernels are still to be
/// written. Selecting one would dispatch a kernel that does not exist, on a
/// rented card, after the buffers were built. The parser therefore **refuses**
/// any arm whose entry here is `false`, by name, and the bare command
/// (`LLVQ_BENCH_ARMS` unset) selects only the ones that are `true`.
///
/// 🕳️ **This was a threshold, `IMPLEMENTED: usize`, and the threshold carried
/// an assumption nobody had had to state: that kernels land in registration
/// order.** The day a kernel landed out of order — E1v's, which has nothing to
/// do with P4's eight — a single index could not express it: `e1v` sits at 15,
/// so making it runnable would have declared the eight unwritten arms below it
/// runnable too, and a bare `planesbench` would have dispatched a missing
/// kernel on a rented card. Exactly what the threshold existed to prevent. One
/// flag per arm says the same thing without the ordering assumption.
pub const HAS_KERNEL: [bool; N_ARMS] = [
    // the six of the 2026-08-10 job, and the v2 campaign's seventh
    true, true, true, true, true, true, true,
    // cublasf16, written 2026-08-18 (F1 of the TACO plan): not a kernel of
    // ours. It is one cublasGemmEx call (16F/16F/16F, compute 32F, n = 1) on
    // the SAME w16 buffer as the fp16 witness, checked against its own f64
    // reference with binary16 inputs. This is the denominator every published
    // × was waiting for: if tv_f16 is not at cuBLAS level, it shows here,
    // before a referee makes it show.
    true,
    // P4 §2.5, mvkf16: still to be written
    false,
    // `nullk`, written 2026-08-16. It lands before P4's seven other arms for a
    // fundamental reason: it replaces no layout and has no admission criterion
    // to meet. It MEASURES the residue that the CUDA attribution obtains by
    // subtraction. It is not a candidate, so it does not wait for the
    // operator's A2/A4/A6 rulings.
    true,
    // planes14k, planes12xk, golay70v2k, e1c14, e1c12: still to be written
    false, false, false, false, false,
    // 🚨 e1v — written, wired, and **never compiled by nvcc**. `bin/cuhcheck`
    // says clang parses it and `tests/e1v_decoder_matches_rust.rs` runs its
    // decode against the Rust reference on this machine; neither is a device
    // compile. The flag is true because the arm is dispatchable — leaving it
    // false would make its wiring unreachable dead code — and what stands
    // between it and a published number is the bench's own V0 plus the
    // `local_bytes != 0` check at startup, not this table.
    true,
    // 🚨 qtip — registered, NOT runnable. P0 of F2 resolved the format and
    // wrote its host codec (`qtip_host`), but nothing has compiled the device
    // side: the kernel is fetched, not committed, so a bare `planesbench` on a
    // machine without `LLVQ_QTIP_DIR` has no kernel to dispatch at all. This
    // flag flips in P2, in the same commit that shows a device compile — not
    // before, and not on the strength of a host-side test.
    false,
];

/// Arms whose kernel exists but **not in this repository**, and which are
/// therefore fetched at job time.
///
/// This is a third state, and it is needed because the two that existed cannot
/// express QTIP. `HAS_KERNEL[QTIP]` is `false` and stays `false` — the
/// statement it makes ("there is no kernel here to dispatch") is literally
/// true, the file is GPL v3 and is not committed. But `false` also means "the
/// parser refuses this name", and that would leave a fetched arm unselectable
/// forever.
///
/// The rule this table adds is narrow on purpose: such an arm is **never** in
/// [`ArmSet::runnable`], so a bare `planesbench` on any machine — with or
/// without the fetch — dispatches exactly what it dispatched before this arm
/// existed. It becomes selectable only when a job **names it explicitly**,
/// which is the only context in which someone has also arranged for the
/// kernel to be there. The bench then fails loudly if it is not.
pub const FETCHED_AT_RUNTIME: [bool; N_ARMS] = {
    let mut f = [false; N_ARMS];
    f[QTIP] = true;
    f
};

/// Whether an arm may be named in `LLVQ_BENCH_ARMS` at all.
pub fn is_selectable(arm: usize) -> bool {
    HAS_KERNEL[arm] || FETCHED_AT_RUNTIME[arm]
}

/// The name each arm carries in a published table — prettier than
/// [`ARM_NAMES`], which is the `LLVQ_BENCH_ARMS` vocabulary.
///
/// 🕳️ **This lived in `bin/planesbench.rs`, and that is why the CUDA image was
/// un-buildable for a day without anyone knowing.** It is declared
/// `[&str; N_ARMS]` and indexed by the arm, so it must grow with the registry —
/// but `planesbench.rs` is entirely under `cfg(target_os = "linux")`, so the
/// development machine never type-checked it. P4 took `N_ARMS` from 7 to 15 on
/// 2026-08-15 and left seven literals behind; the error surfaced on
/// 2026-08-16, in a build launched for something else entirely.
///
/// It is here now for the reason this module exists at all — its own header
/// says it: *the development machine has no CUDA, so the tests must run here*.
/// A length mismatch is a compile error on a Mac from this line on.
pub const DISPLAY_NAMES: [&str; N_ARMS] = [
    "LLVQ Slot32",
    "LLVQ Planes14",
    "LLVQ Planes12x",
    "LLVQ Golay70 v1",
    "FP16 (128 bits)",
    "AWQ w4g128",
    "LLVQ Golay70 v2",
    "FP16 cuBLAS",
    "FP16 matvec-k",
    "floor (nullk)",
    "LLVQ Planes14-k",
    "LLVQ Planes12x-k",
    "LLVQ Golay70 v2-k",
    "LLVQ E1c14",
    "LLVQ E1c12",
    "LLVQ E1v",
    "QTIP 2 bits",
];

/// The order a table PRINTS its rows in — cosmetic, and deliberately not the
/// dispatch order: the witness first, v2 under v1, the competitor last.
///
/// Its length is its own, not [`N_ARMS`]: an arm with no kernel has no row.
pub const DISPLAY_ORDER: [usize; 11] = [
    // The floor first: it is the quantity every other row is read against, and
    // putting it at the top saves the reader from hunting for it. cublasf16
    // sits right under our own witness, so the two rows every published ×
    // divides are read one under the other.
    NULLK, FP16, CUBLASF16, SLOT32, PLANES14, PLANES12X, GOLAY70V1, GOLAY70V2, E1V, AWQ,
    // 🚨 The two competitors at the end of the table, together. QTIP got here
    // on 2026-08-20 and its absence was a SILENT defect of a particular kind:
    // everything `planesbench` prints iterates over this table, so the bench
    // would have built the payload, checked the arm against its f64 reference
    // and dispatched it in the seven rounds, then written NO line about it,
    // while "the QTIP row of the table" is the declared deliverable of its
    // job. The comment above says "an arm with no kernel has no row"; QTIP is
    // the case that shortcut does not cover: `HAS_KERNEL` is false and
    // `FETCHED_AT_RUNTIME` is true.
    QTIP,
];

/// A set of arms, at most one bit per registered arm.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct ArmSet {
    /// One bit per registered arm. Widened from `u8` when P4 took the count
    /// from 7 to 15, then from `u16` when E1v took it to 16 — at 16 arms
    /// `1u16 << N_ARMS` overflows the shift itself, so the old width could not
    /// even build the full set. A set that silently truncated would deselect an
    /// arm without a word, which is the failure mode this whole module exists
    /// to prevent. `the_set_holds_every_registered_arm` pins the width.
    bits: u32,
}

impl ArmSet {
    /// Every **registered** arm, implemented or not. Use [`Self::runnable`]
    /// for what a job may actually dispatch.
    pub fn all() -> Self {
        ArmSet { bits: (1u32 << N_ARMS) - 1 }
    }

    /// Every arm that has a kernel — what the bare command runs.
    pub fn runnable() -> Self {
        let mut s = ArmSet::empty();
        for (a, &ok) in HAS_KERNEL.iter().enumerate() {
            if ok {
                s.insert(a);
            }
        }
        s
    }

    pub fn empty() -> Self {
        ArmSet { bits: 0 }
    }

    pub fn has(self, arm: usize) -> bool {
        debug_assert!(arm < N_ARMS);
        self.bits & (1u32 << arm) != 0
    }

    pub fn insert(&mut self, arm: usize) {
        debug_assert!(arm < N_ARMS);
        self.bits |= 1u32 << arm;
    }

    pub fn is_superset_of(self, other: ArmSet) -> bool {
        self.bits & other.bits == other.bits
    }

    /// Arms of `self` that are not in `other`, registration order.
    pub fn minus(self, other: ArmSet) -> ArmSet {
        ArmSet { bits: self.bits & !other.bits }
    }

    pub fn len(self) -> usize {
        self.bits.count_ones() as usize
    }

    pub fn is_empty(self) -> bool {
        self.bits == 0
    }

    /// Member arms, in registration order.
    pub fn iter(self) -> impl Iterator<Item = usize> {
        (0..N_ARMS).filter(move |&a| self.has(a))
    }

    /// Canonical label: member names in registration order, comma-joined.
    pub fn label(self) -> String {
        self.iter().map(|a| ARM_NAMES[a]).collect::<Vec<_>>().join(",")
    }
}

/// Parse the `LLVQ_BENCH_ARMS` value into phases. `None` (variable unset)
/// is one phase of every arm — the bare command keeps its meaning.
pub fn parse_phases(spec: Option<&str>) -> Result<Vec<ArmSet>, String> {
    let Some(spec) = spec else {
        return Ok(vec![ArmSet::runnable()]);
    };
    let mut phases = Vec::new();
    for (pi, phase_txt) in spec.split(';').enumerate() {
        let mut set = ArmSet::empty();
        let mut named_any = false;
        for raw in phase_txt.split(',') {
            let name = raw.trim();
            if name.is_empty() {
                return Err(format!(
                    "LLVQ_BENCH_ARMS: empty name in phase {} (\"{}\")",
                    pi + 1,
                    phase_txt.trim()
                ));
            }
            named_any = true;
            if name == "golay70" {
                return Err(
                    "LLVQ_BENCH_ARMS: \"golay70\" is ambiguous since the v2. Name \
                     golay70v1 (the published decoder) or golay70v2"
                        .to_string(),
                );
            }
            if let Some(arm) = ARM_NAMES.iter().position(|&n| n == name) {
                if !is_selectable(arm) {
                    return Err(format!(
                        "LLVQ_BENCH_ARMS: \"{name}\" is registered to fix the dispatch \
                         order (P4 §2.3) but its kernel is NOT written. Selecting it \
                         would dispatch a kernel that does not exist, on a rented \
                         card. Runnable arms: {}",
                        ArmSet::runnable().label()
                    ));
                }
            }
            let Some(arm) = ARM_NAMES.iter().position(|&n| n == name) else {
                return Err(format!(
                    "LLVQ_BENCH_ARMS: unknown arm \"{name}\". Valid: {}",
                    ARM_NAMES.join(", ")
                ));
            };
            if set.has(arm) {
                return Err(format!(
                    "LLVQ_BENCH_ARMS: \"{name}\" named twice in phase {}",
                    pi + 1
                ));
            }
            set.insert(arm);
        }
        if !named_any {
            return Err(format!("LLVQ_BENCH_ARMS: phase {} is empty", pi + 1));
        }
        if !set.has(FP16) {
            return Err(format!(
                "LLVQ_BENCH_ARMS: phase {} without fp16. The witness cannot be \
                 deselected, every published ratio is formed against it",
                pi + 1
            ));
        }
        if let Some(prev) = phases.last() {
            if !set.is_superset_of(*prev) {
                return Err(format!(
                    "LLVQ_BENCH_ARMS: phase {} ({}) does not contain phase {} ({}). \
                     Buffers are built and never freed, so a shrinking phase would \
                     measure with a dead arm's residency",
                    pi + 1,
                    set.label(),
                    pi,
                    prev.label()
                ));
            }
        }
        phases.push(set);
    }
    Ok(phases)
}

/// P4 §2.4's phase plan, as a value rather than as prose in a job script.
///
/// Three phases, each a superset of the last: the six of the published run,
/// then the seven that manufacture `Δ_contrôle`, then those plus P4's eight.
///
/// 🚨 **A single-phase job produces NO `Δ_contrôle`, therefore no `R`,
/// therefore no decision rule — and nothing in the output says so** (§2.2,
/// which calls the phase plan *a validity condition*). Having the plan as
/// a function that can be compared against what a job actually ran is what
/// turns that from a footnote into something a test can check.
pub fn p4_phase_plan() -> Vec<ArmSet> {
    let of = |arms: &[usize]| {
        let mut s = ArmSet::empty();
        for &a in arms {
            s.insert(a);
        }
        s
    };
    let p1 = of(&PHASE1);
    let p2 = of(&PHASE2);
    let mut p3 = p2;
    for &a in &PHASE3_NEW {
        p3.insert(a);
    }
    vec![p1, p2, p3]
}

/// The plan's own `LLVQ_BENCH_ARMS` string — so a job is launched from the
/// same object a test checks, and not from a line retyped into a shell.
pub fn p4_phase_spec() -> String {
    p4_phase_plan()
        .iter()
        .map(|s| s.label())
        .collect::<Vec<_>>()
        .join(";")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unset_is_one_phase_of_every_arm() {
        // "Every arm" means every arm the bench can DISPATCH. It used to mean
        // every registered arm, and the two were the same set until P4
        // registered eight whose kernels are not written — selecting one would
        // launch a missing kernel on a rented card.
        let phases = parse_phases(None).unwrap();
        assert_eq!(phases, vec![ArmSet::runnable()]);
        assert_eq!(phases[0].len(), HAS_KERNEL.iter().filter(|&&k| k).count());
    }

    #[test]
    fn the_six_arm_control_then_the_seven_arm_table_parses() {
        let phases = parse_phases(Some(
            "fp16,slot32,planes14,planes12x,golay70v1,awq;\
             fp16,slot32,planes14,planes12x,golay70v1,awq,golay70v2",
        ))
        .unwrap();
        assert_eq!(phases.len(), 2);
        assert_eq!(phases[0].len(), 6);
        assert!(!phases[0].has(GOLAY70V2));
        // 🕳️ This line read `ArmSet::all()` until P4 registered eight more
        // arms, and it passed only because "every registered arm" and "the
        // seven of the v2 campaign" happened to be the same set. They are not
        // the same statement, and the day they diverged the test failed —
        // loudly, which is the whole design of this module.
        let seven = {
            let mut s = ArmSet::empty();
            for &a in &PHASE2 {
                s.insert(a);
            }
            s
        };
        assert_eq!(phases[1], seven);
        assert_eq!(phases[1].minus(phases[0]).label(), "golay70v2");
    }

    #[test]
    fn labels_are_in_registration_order_whatever_the_input_order() {
        let phases = parse_phases(Some("awq,fp16,slot32")).unwrap();
        assert_eq!(phases[0].label(), "slot32,fp16,awq");
    }

    #[test]
    fn an_unknown_arm_is_refused_with_the_valid_names() {
        let e = parse_phases(Some("fp16,planes15")).unwrap_err();
        assert!(e.contains("planes15"), "{e}");
        assert!(e.contains("golay70v2"), "{e}");
    }

    #[test]
    fn bare_golay70_is_refused_by_name() {
        let e = parse_phases(Some("fp16,golay70")).unwrap_err();
        assert!(e.contains("ambiguous"), "{e}");
    }

    #[test]
    fn a_phase_without_the_witness_is_refused() {
        let e = parse_phases(Some("slot32,planes14")).unwrap_err();
        assert!(e.contains("fp16"), "{e}");
    }

    #[test]
    fn a_duplicate_name_is_refused() {
        let e = parse_phases(Some("fp16,awq,awq")).unwrap_err();
        assert!(e.contains("twice"), "{e}");
    }

    #[test]
    fn an_empty_name_and_an_empty_phase_are_refused() {
        assert!(parse_phases(Some("fp16,,awq")).is_err());
        assert!(parse_phases(Some("fp16;;fp16")).is_err());
        assert!(parse_phases(Some("")).is_err());
    }

    #[test]
    fn a_shrinking_phase_is_refused() {
        let e = parse_phases(Some("fp16,slot32,awq;fp16,slot32")).unwrap_err();
        assert!(e.contains("residency"), "{e}");
    }

    #[test]
    fn an_equal_phase_is_allowed() {
        // Same set twice: identical residency, a legitimate repro check.
        let phases = parse_phases(Some("fp16,awq;fp16,awq")).unwrap();
        assert_eq!(phases[0], phases[1]);
    }

    /// The set must hold every registered arm. P4 took the count from 7 to 15
    /// and the backing integer from `u8` to `u16`; a set too narrow would
    /// deselect the top arms in silence, which is exactly the failure this
    /// module refuses everywhere else.
    #[test]
    fn the_set_holds_every_registered_arm() {
        assert_eq!(ARM_NAMES.len(), N_ARMS);
        let all = ArmSet::all();
        for (a, name) in ARM_NAMES.iter().enumerate() {
            assert!(all.has(a), "arm {a} ({name}) does not fit in the set");
        }
        assert_eq!(all.len(), N_ARMS);
        // And no name is registered twice — a duplicate would make one of the
        // two unreachable by name and the parser would never say which.
        let mut sorted = ARM_NAMES.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), N_ARMS, "an arm name is registered twice");
    }

    /// The six of the published run keep indices 0..5 and `golay70v2` keeps 6.
    /// P4 §2.3: an added arm never reorders the dispatch of the arms that
    /// produced a published table.
    #[test]
    fn the_published_arms_keep_their_dispatch_order() {
        assert_eq!(
            &ARM_NAMES[..7],
            &["slot32", "planes14", "planes12x", "golay70v1", "fp16", "awq", "golay70v2"]
        );
        for (i, &a) in PHASE3_NEW.iter().enumerate() {
            assert_eq!(a, 7 + i, "the new arm {} is not in last position", ARM_NAMES[a]);
        }
    }

    /// The plan is three monotone phases, each carrying the witness — the two
    /// properties the parser enforces, checked on the plan the job will use so
    /// a malformed plan fails here rather than on a rented card.
    #[test]
    fn the_p4_phase_plan_is_valid_by_its_own_parser() {
        let plan = p4_phase_plan();
        assert_eq!(plan.len(), 3, "a single-phase job produces no Δ_contrôle (§2.2)");
        assert_eq!(plan[0].len(), 6);
        assert_eq!(plan[1].len(), 7);
        // 🕳️ This read `N_ARMS`, and it passed only because P4's phase 3 and
        // "every registered arm" happened to be the same set — the very
        // accident the comment fifteen lines above describes for the previous
        // occurrence. E1v separated them: it is registered, and it is NOT part
        // of P4's plan (its own document governs it). The plan's size is what
        // the plan is made of.
        assert_eq!(plan[2].len(), PHASE2.len() + PHASE3_NEW.len());
        assert!(!plan[2].has(E1V), "e1v does not belong to P4's phase plan");
        for p in &plan {
            assert!(p.has(FP16), "the witness cannot be deselected");
        }
        // 🚨 And the plan is NOT runnable today, which is a fact worth
        // asserting rather than discovering at job start: phase 3 names arms
        // whose kernels are not written, so the parser refuses the spec and
        // says which one. This test flips to the round trip below the day
        // `IMPLEMENTED` reaches `N_ARMS`.
        let spec = p4_phase_spec();
        match parse_phases(Some(&spec)) {
            // Refused today, and the message says which arm is missing. The
            // day every kernel exists this falls into `Ok` and the round trip
            // is checked instead — no edit needed, and no window where the
            // test asserts nothing.
            Err(e) => assert!(e.contains("kernel is NOT written"), "{e}"),
            Ok(back) => assert_eq!(back, plan),
        }
        // The two phases that ARE runnable parse today — the control plan of
        // the published campaign is not held hostage by P4's unwritten arms.
        let runnable_spec = format!("{};{}", plan[0].label(), plan[1].label());
        let back = parse_phases(Some(&runnable_spec)).expect("phases 1 and 2 are runnable");
        assert_eq!(back, vec![plan[0], plan[1]]);
    }

    /// A registered-but-unwritten arm is refused **by name**, and the bare
    /// command never selects one. The `LLVQ_FUSED_LAYOUT` rule again: a
    /// selection that silently ran the wrong thing is worse than one that
    /// fails.
    #[test]
    fn an_arm_without_a_kernel_is_registered_but_not_runnable() {
        assert_eq!(parse_phases(None).unwrap(), vec![ArmSet::runnable()]);
        assert_eq!(
            ArmSet::runnable().len(),
            HAS_KERNEL.iter().filter(|&&k| k).count()
        );
        // Every arm of the published run and of the control phase must have a
        // kernel, or the campaign those two phases reproduce could not run.
        for &a in PHASE1.iter().chain(&PHASE2) {
            assert!(HAS_KERNEL[a], "{} carries a published number with no kernel", ARM_NAMES[a]);
        }
        // PHASE3_NEW is no longer "the arms with no kernel": `nullk` is one of
        // them and it has had a kernel since 2026-08-16. The test reads the
        // FLAG, which is the property it checks, and not a list that happened
        // to coincide with it. Same slip as `plan[2].len() == N_ARMS` fifteen
        // lines below.
        //
        // 🕳️ And the same slip happened ONE STEP UP on 2026-08-20: this filter
        // read `!HAS_KERNEL[a]`, which stopped being the property under test
        // the day an arm had a kernel **outside the repository**
        // (`FETCHED_AT_RUNTIME`). The property is "neither here nor
        // fetchable", so `!is_selectable`. The flag being read must be the one
        // the parser consults, not the one that resembled it.
        for a in (0..N_ARMS).filter(|&a| !is_selectable(a)) {
            let e = parse_phases(Some(&format!("fp16,{}", ARM_NAMES[a]))).unwrap_err();
            assert!(e.contains(ARM_NAMES[a]), "{e}");
            assert!(e.contains("kernel is NOT written"), "{e}");
        }
        // The one arm that is refused by `HAS_KERNEL` yet accepted by the
        // parser is pinned by name here: a second one appearing silently is
        // exactly what this test exists to catch.
        let exceptions: Vec<&str> = (0..N_ARMS)
            .filter(|&a| !HAS_KERNEL[a] && is_selectable(a))
            .map(|a| ARM_NAMES[a])
            .collect();
        assert_eq!(exceptions, vec!["qtip"], "an arm fetched at job time was added silently");

        // And every implemented arm still parses. `fp16` is the witness and
        // is already in the spec, so it is named once.
        for name in ArmSet::runnable().iter().map(|a| ARM_NAMES[a]) {
            let name = &name;
            let spec = if *name == "fp16" { name.to_string() } else { format!("fp16,{name}") };
            parse_phases(Some(&spec))
                .unwrap_or_else(|e| panic!("{name} should be selectable: {e}"));
        }
    }

    /// 🚨 **The property the threshold could not express.** `HAS_KERNEL` is a
    /// flag per arm, so an arm may be runnable while an arm registered BEFORE it
    /// is not — which is the situation E1v creates, its kernel having nothing to
    /// do with P4's eight.
    ///
    /// A threshold would have had to declare those eight runnable to reach
    /// index 15, and a bare `planesbench` would then have dispatched a missing
    /// kernel on a rented card. This asserts the shape rather than the current
    /// contents: `runnable()` is read off the flags and is not required to be a
    /// prefix of the registration order.
    #[test]
    fn runnability_is_per_arm_and_not_a_prefix() {
        for (a, &ok) in HAS_KERNEL.iter().enumerate() {
            assert_eq!(
                ArmSet::runnable().has(a),
                ok,
                "{}: the runnable set does not follow its flag",
                ARM_NAMES[a]
            );
        }
        // The registry HAS a hole today — e1v is runnable while the eight arms
        // registered before it are not — which is exactly the situation a
        // threshold could not express.
        assert!(HAS_KERNEL[E1V], "e1v has a kernel");
        assert!(
            PHASE3_NEW.iter().any(|&a| !HAS_KERNEL[a]),
            "the hole is gone: this test would lose its subject"
        );
        let flags: Vec<bool> = (0..N_ARMS).map(|a| ArmSet::runnable().has(a)).collect();
        assert_eq!(flags, HAS_KERNEL.to_vec());
    }

    /// The two display tables, checked where a Mac can check them.
    ///
    /// 🚨 The length of `DISPLAY_NAMES` is the whole point — it is what P4's
    /// change silently violated for a day. The rest guards the other ways a
    /// hand-kept parallel table goes wrong: a row printed twice, a row for an
    /// arm that cannot run, an empty label.
    #[test]
    fn a_fetched_arm_is_selectable_but_never_bare() {
        // The whole point of the third state: naming it works, and a bare
        // command still never picks it up. Both halves matter — the first
        // makes the arm usable at all, the second guarantees that adding it
        // changed nothing for every machine that does not fetch it.
        let qtip = ARM_NAMES[QTIP];
        assert!(!HAS_KERNEL[QTIP], "QTIP's kernel is not in this repository");
        assert!(FETCHED_AT_RUNTIME[QTIP]);
        assert!(is_selectable(QTIP));
        assert!(!ArmSet::runnable().has(QTIP), "a bare run must not select a fetched arm");
        let phases = parse_phases(Some(&format!("fp16,{qtip}"))).unwrap();
        assert_eq!(phases.len(), 1);
        assert!(phases[0].has(QTIP));
    }

    #[test]
    fn only_qtip_is_fetched_at_runtime() {
        // A second arm silently joining this table would make itself
        // selectable without a kernel, which is exactly what the refusal
        // exists to prevent.
        for a in 0..N_ARMS {
            assert_eq!(FETCHED_AT_RUNTIME[a], a == QTIP, "arm {} ({})", a, ARM_NAMES[a]);
            // And no arm may claim both: "in the repository" and "fetched"
            // are exclusive statements about where the file is.
            assert!(!(HAS_KERNEL[a] && FETCHED_AT_RUNTIME[a]), "arm {a}");
        }
    }

    #[test]
    fn the_display_tables_cover_the_registry() {
        assert_eq!(DISPLAY_NAMES.len(), N_ARMS);
        for (a, n) in DISPLAY_NAMES.iter().enumerate() {
            assert!(!n.is_empty(), "arm {} has no label", ARM_NAMES[a]);
        }
        let mut seen = ArmSet::empty();
        for &a in &DISPLAY_ORDER {
            assert!(a < N_ARMS, "the display order names an arm that does not exist");
            assert!(!seen.has(a), "{} is displayed twice", ARM_NAMES[a]);
            // 🕳️ THIRD time this slip happens in this file, and always the
            // same one: the test read `HAS_KERNEL`, which stopped being the
            // property under test the day an arm had a kernel OUTSIDE the
            // repository. The question here is "can this arm run", which is
            // `is_selectable`. The first two occurrences are documented on
            // `an_arm_without_a_kernel_is_registered_but_not_runnable`; this
            // is the third, and the pattern deserves a name: a flag that
            // coincides with the property is not the property.
            assert!(
                is_selectable(a),
                "{} has a table row but cannot run",
                ARM_NAMES[a]
            );
            seen.insert(a);
        }
        // Every SELECTABLE arm has a row — including the ones a bare run never
        // picks up. An arm that can be timed and has no row would be measured
        // and never printed, which is a worse failure than not measuring it:
        // the job costs the same and produces nothing.
        let selectable = {
            let mut s = ArmSet::empty();
            for a in 0..N_ARMS {
                if is_selectable(a) {
                    s.insert(a);
                }
            }
            s
        };
        assert_eq!(seen, selectable, "a selectable arm has no row");
        // And the rows a bare run produces are still exactly the runnable set:
        // adding a fetched arm must not change what a bare command prints.
        for a in ArmSet::runnable().iter() {
            assert!(seen.has(a), "{} runs bare with no row", ARM_NAMES[a]);
        }
    }

    #[test]
    fn whitespace_around_names_is_tolerated() {
        let phases = parse_phases(Some(" fp16 , slot32 ; fp16 ,slot32, awq ")).unwrap();
        assert_eq!(phases[0].label(), "slot32,fp16");
        assert_eq!(phases[1].label(), "slot32,fp16,awq");
    }
}
