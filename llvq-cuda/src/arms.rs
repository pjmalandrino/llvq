//! Arm selection for `planesbench` — the `LLVQ_BENCH_ARMS` contract.
//!
//! Exists to close écart É1 of `proofs/preregistration-2026-08-10.md` §7bis:
//! the six-arm job could not run its five-arm control in the same process,
//! because the bench had no way to leave an arm out — and a control taken
//! from another job, another image and another translation unit is exactly
//! the inter-process subtraction this repository has already had to retract.
//! The exit clause written into É1 is the specification here: *un run de
//! contrôle doit être à une variable d'environnement près.*
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
];
pub const N_ARMS: usize = 15;

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
/// `broadcast_matmul` trap made this repo publish a ×2,03 that came from a
/// defect of its own dense arm (§2.5).
pub const MVKF16: usize = 8;
/// Same grid, same k, output written, **no weight read**. The floor.
pub const NULLK: usize = 9;
pub const PLANES14K: usize = 10;
pub const PLANES12XK: usize = 11;
pub const GOLAY70V2K: usize = 12;
pub const E1C14: usize = 13;
pub const E1C12: usize = 14;

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

/// How many of [`ARM_NAMES`] have a kernel behind them.
///
/// 🚨 **Registering a name is not implementing an arm.** P4 §2.3 wants the
/// dispatch order fixed *before* the job — an added arm must never reorder the
/// published ones — so the eight arms of §2.5 are registered now, in order,
/// while their kernels are still to be written. Selecting one would dispatch a
/// kernel that does not exist, on a rented card, after the buffers were built.
///
/// So the parser **refuses** any arm at or above this index, by name, and the
/// bare command (`LLVQ_BENCH_ARMS` unset) selects only the arms below it.
/// Raise it by one as each kernel lands, and the test below will tell you if
/// you raised it past what exists.
pub const IMPLEMENTED: usize = 7;

/// A set of arms, at most one bit per registered arm.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct ArmSet {
    /// One bit per registered arm. Widened from `u8` when P4 took the count
    /// from 7 to 15 — a set that silently truncated would deselect an arm
    /// without a word, which is the failure mode this whole module exists to
    /// prevent. `the_set_holds_every_registered_arm` pins the width.
    bits: u16,
}

impl ArmSet {
    /// Every **registered** arm, implemented or not. Use [`Self::runnable`]
    /// for what a job may actually dispatch.
    pub fn all() -> Self {
        ArmSet { bits: (1u16 << N_ARMS) - 1 }
    }

    /// Every arm that has a kernel — what the bare command runs.
    pub fn runnable() -> Self {
        ArmSet { bits: (1u16 << IMPLEMENTED) - 1 }
    }

    pub fn empty() -> Self {
        ArmSet { bits: 0 }
    }

    pub fn has(self, arm: usize) -> bool {
        debug_assert!(arm < N_ARMS);
        self.bits & (1u16 << arm) != 0
    }

    pub fn insert(&mut self, arm: usize) {
        debug_assert!(arm < N_ARMS);
        self.bits |= 1u16 << arm;
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
                    "LLVQ_BENCH_ARMS : nom vide dans la phase {} («{}»)",
                    pi + 1,
                    phase_txt.trim()
                ));
            }
            named_any = true;
            if name == "golay70" {
                return Err(
                    "LLVQ_BENCH_ARMS : «golay70» est ambigu depuis la v2 — \
                     nommer golay70v1 (le décodeur publié) ou golay70v2"
                        .to_string(),
                );
            }
            if let Some(arm) = ARM_NAMES.iter().position(|&n| n == name) {
                if arm >= IMPLEMENTED {
                    return Err(format!(
                        "LLVQ_BENCH_ARMS : «{name}» est enregistré pour figer l'ordre de \
                         dispatch (P4 §2.3) mais son noyau n'est PAS écrit — le \
                         sélectionner dispatcherait un noyau inexistant sur une carte \
                         louée. Bras exécutables : {}",
                        ARM_NAMES[..IMPLEMENTED].join(", ")
                    ));
                }
            }
            let Some(arm) = ARM_NAMES.iter().position(|&n| n == name) else {
                return Err(format!(
                    "LLVQ_BENCH_ARMS : bras inconnu «{name}» — valides : {}",
                    ARM_NAMES.join(", ")
                ));
            };
            if set.has(arm) {
                return Err(format!(
                    "LLVQ_BENCH_ARMS : «{name}» nommé deux fois dans la phase {}",
                    pi + 1
                ));
            }
            set.insert(arm);
        }
        if !named_any {
            return Err(format!("LLVQ_BENCH_ARMS : phase {} vide", pi + 1));
        }
        if !set.has(FP16) {
            return Err(format!(
                "LLVQ_BENCH_ARMS : phase {} sans fp16 — le témoin n'est pas \
                 désélectionnable, tout rapport publié se forme contre lui",
                pi + 1
            ));
        }
        if let Some(prev) = phases.last() {
            if !set.is_superset_of(*prev) {
                return Err(format!(
                    "LLVQ_BENCH_ARMS : la phase {} ({}) ne contient pas la phase {} \
                     ({}) — les tampons se construisent et ne se libèrent pas, une \
                     phase qui rétrécit mesurerait avec la résidence d'un bras mort",
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
/// which calls the phase plan *une condition de validité*). Having the plan as
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
        assert_eq!(phases[0].len(), IMPLEMENTED);
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
        assert!(e.contains("ambigu"), "{e}");
    }

    #[test]
    fn a_phase_without_the_witness_is_refused() {
        let e = parse_phases(Some("slot32,planes14")).unwrap_err();
        assert!(e.contains("fp16"), "{e}");
    }

    #[test]
    fn a_duplicate_name_is_refused() {
        let e = parse_phases(Some("fp16,awq,awq")).unwrap_err();
        assert!(e.contains("deux fois"), "{e}");
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
        assert!(e.contains("résidence"), "{e}");
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
            assert!(all.has(a), "le bras {a} ({name}) ne tient pas dans le set");
        }
        assert_eq!(all.len(), N_ARMS);
        // And no name is registered twice — a duplicate would make one of the
        // two unreachable by name and the parser would never say which.
        let mut sorted = ARM_NAMES.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), N_ARMS, "un nom de bras est enregistré deux fois");
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
            assert_eq!(a, 7 + i, "le bras neuf {} n'est pas en dernière position", ARM_NAMES[a]);
        }
    }

    /// The plan is three monotone phases, each carrying the witness — the two
    /// properties the parser enforces, checked on the plan the job will use so
    /// a malformed plan fails here rather than on a rented card.
    #[test]
    fn the_p4_phase_plan_is_valid_by_its_own_parser() {
        let plan = p4_phase_plan();
        assert_eq!(plan.len(), 3, "un job à une phase ne produit aucun Δ_contrôle (§2.2)");
        assert_eq!(plan[0].len(), 6);
        assert_eq!(plan[1].len(), 7);
        assert_eq!(plan[2].len(), N_ARMS);
        for p in &plan {
            assert!(p.has(FP16), "le témoin n'est pas désélectionnable");
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
            Err(e) => assert!(e.contains("noyau n'est PAS écrit"), "{e}"),
            Ok(back) => assert_eq!(back, plan),
        }
        // The two phases that ARE runnable parse today — the control plan of
        // the published campaign is not held hostage by P4's unwritten arms.
        let runnable_spec = format!("{};{}", plan[0].label(), plan[1].label());
        let back = parse_phases(Some(&runnable_spec)).expect("phases 1 et 2 sont exécutables");
        assert_eq!(back, vec![plan[0], plan[1]]);
    }

    /// A registered-but-unwritten arm is refused **by name**, and the bare
    /// command never selects one. The `LLVQ_FUSED_LAYOUT` rule again: a
    /// selection that silently ran the wrong thing is worse than one that
    /// fails.
    #[test]
    fn an_arm_without_a_kernel_is_registered_but_not_runnable() {
        const _: () = assert!(IMPLEMENTED <= N_ARMS);
        assert_eq!(parse_phases(None).unwrap(), vec![ArmSet::runnable()]);
        assert_eq!(ArmSet::runnable().len(), IMPLEMENTED);
        for &a in &PHASE3_NEW {
            let e = parse_phases(Some(&format!("fp16,{}", ARM_NAMES[a]))).unwrap_err();
            assert!(e.contains(ARM_NAMES[a]), "{e}");
            assert!(e.contains("noyau n'est PAS écrit"), "{e}");
        }
        // And every implemented arm still parses. `fp16` is the witness and
        // is already in the spec, so it is named once.
        for name in &ARM_NAMES[..IMPLEMENTED] {
            let spec = if *name == "fp16" { name.to_string() } else { format!("fp16,{name}") };
            parse_phases(Some(&spec))
                .unwrap_or_else(|e| panic!("{name} devrait être sélectionnable : {e}"));
        }
    }

    #[test]
    fn whitespace_around_names_is_tolerated() {
        let phases = parse_phases(Some(" fp16 , slot32 ; fp16 ,slot32, awq ")).unwrap();
        assert_eq!(phases[0].label(), "slot32,fp16");
        assert_eq!(phases[1].label(), "slot32,fp16,awq");
    }
}
