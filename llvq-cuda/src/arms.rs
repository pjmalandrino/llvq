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

/// Registration order — the dispatch order inside a round. `golay70v2` is
/// LAST: it is the arm the v2 campaign adds, and an added arm must never
/// reorder the dispatch of the arms that produced the published tables.
pub const ARM_NAMES: [&str; N_ARMS] =
    ["slot32", "planes14", "planes12x", "golay70v1", "fp16", "awq", "golay70v2"];
pub const N_ARMS: usize = 7;

pub const SLOT32: usize = 0;
pub const PLANES14: usize = 1;
pub const PLANES12X: usize = 2;
pub const GOLAY70V1: usize = 3;
pub const FP16: usize = 4;
pub const AWQ: usize = 5;
pub const GOLAY70V2: usize = 6;

/// A set of arms, at most one bit per registered arm.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct ArmSet {
    bits: u8,
}

impl ArmSet {
    pub fn all() -> Self {
        ArmSet { bits: (1 << N_ARMS) - 1 }
    }

    pub fn empty() -> Self {
        ArmSet { bits: 0 }
    }

    pub fn has(self, arm: usize) -> bool {
        debug_assert!(arm < N_ARMS);
        self.bits & (1 << arm) != 0
    }

    pub fn insert(&mut self, arm: usize) {
        debug_assert!(arm < N_ARMS);
        self.bits |= 1 << arm;
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
        return Ok(vec![ArmSet::all()]);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unset_is_one_phase_of_every_arm() {
        let phases = parse_phases(None).unwrap();
        assert_eq!(phases, vec![ArmSet::all()]);
        assert_eq!(phases[0].len(), N_ARMS);
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
        assert_eq!(phases[1], ArmSet::all());
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

    #[test]
    fn whitespace_around_names_is_tolerated() {
        let phases = parse_phases(Some(" fp16 , slot32 ; fp16 ,slot32, awq ")).unwrap();
        assert_eq!(phases[0].label(), "slot32,fp16");
        assert_eq!(phases[1].label(), "slot32,fp16,awq");
    }
}
