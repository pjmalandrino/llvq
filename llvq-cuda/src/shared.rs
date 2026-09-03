//! Which of the card's **two** shared-memory allowances a staging kernel needs.
//!
//! `rot_apply` stages the whole activation in dynamic shared memory, in one
//! block, because a Walsh–Hadamard transform is `log₂ m` barriers and CUDA has
//! no barrier across blocks (`kernels/rotate.cu` argues the trade). So a model
//! runs only if its widest rotation fits what **one block** may hold — and on
//! sm_70 and later that is two numbers, not one:
//!
//!   * `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK` — 49,152 bytes on an
//!     L40S: what a block gets without asking;
//!   * `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN` — 101,376 bytes on
//!     the same card: what it gets once the **function** has been opted in
//!     with `cuFuncSetAttribute(…, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, …)`.
//!
//! Comparing against the first only is what refused Qwen3-14B on 2026-08-17
//! (job `6a82f40ce55292eada79b526`, $0.24, exit 1 after 488 s): 17,408 × 4 =
//! 69,632 bytes, over the default and comfortably under the opt-in. The kernel
//! was never the obstacle; the host's arithmetic was.
//!
//! ## Why the arithmetic lives here
//!
//! Both callers that need it — `bin/rotbench` and `llvq-llm`'s `fused_cuda`
//! — sit under `cfg(target_os = "linux")`, where nothing on the development
//! Mac compiles them and the only thing that exercises them is a 40-minute
//! image build (passation 2026-08-16 §3, three breakages in two days). This
//! module names no driver type and carries no `cfg`, so the decision itself
//! is tested where tests are free, and the un-netted files are left with a
//! call and a message.
//!
//! ## What this does *not* change
//!
//! The contract of `rot_apply`: **the caller is what keeps the staging under
//! the device limit.** The kernel has no way to check and would simply
//! corrupt. This module moves the bound the caller checks against; it does
//! not move the responsibility, and it does not make any check optional.

/// Bytes of dynamic shared memory `rot_apply` stages for a width of `n`.
///
/// One f32 per coordinate whatever the input dtype: the kernel widens the f16
/// activation on load (`rot_load`) and every stage of the transform reads and
/// writes f32.
pub const fn rot_bytes(n: usize) -> usize {
    n * 4
}

/// Which allowance a staging size needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fit {
    /// Under the default. Nothing to declare — launching is enough.
    Default,
    /// Over the default, under the opt-in ceiling. The function **must** be
    /// opted in before its first launch, or the driver refuses the launch
    /// with `CUDA_ERROR_INVALID_VALUE`.
    OptIn,
}

/// The ceiling one block can reach on this card, opt-in included.
///
/// `max`, not the opt-in figure alone. The driver documents the opt-in as
/// never below the default, but a card or a CUDA version that does not know
/// the attribute reports 0 for it, and a `0` must not be able to shrink an
/// allowance that already worked. Taking the max makes an unknown attribute
/// a no-op instead of a regression.
pub fn ceiling(default_limit: usize, optin_limit: usize) -> usize {
    if optin_limit > default_limit {
        optin_limit
    } else {
        default_limit
    }
}

/// Does `bytes` fit one block, and at what price?
///
/// `None` means it fits neither bound — there is no host-side remedy, and
/// nothing may be launched.
pub fn plan(bytes: usize, default_limit: usize, optin_limit: usize) -> Option<Fit> {
    if bytes <= default_limit {
        Some(Fit::Default)
    } else if bytes <= ceiling(default_limit, optin_limit) {
        Some(Fit::OptIn)
    } else {
        None
    }
}

/// [`plan`] for a rotation of width `n`, with the diagnostic the guard prints.
///
/// The message names **both** bounds and says which one is crossed, because
/// the two failures have different remedies and the earlier message conflated
/// them: over the default alone is a missing opt-in (a host bug, fixed here),
/// over the ceiling is a kernel that does not fit this card at this width (no
/// host fix exists — `rotate.cu`'s header names the two-kernel split as the
/// fallback, and it is not written).
pub fn rot_plan(n: usize, default_limit: usize, optin_limit: usize) -> Result<Fit, String> {
    let bytes = rot_bytes(n);
    if let Some(fit) = plan(bytes, default_limit, optin_limit) {
        return Ok(fit);
    }
    let top = ceiling(default_limit, optin_limit);
    // Two refusals wear the same shape and are not the same fact: on a card
    // that grants an opt-in, the ceiling is what was crossed; on one that
    // announces none, it is the default, and saying "opt-in" there would send
    // a reader hunting for an attribute the card does not have.
    let which = if top > default_limit {
        format!(
            "the card offers {default_limit} by default and {top} after opt-in: the OPT-IN bound \
             is the one crossed, by {} bytes",
            bytes - top
        )
    } else {
        format!(
            "the card offers {default_limit} by default and announces no opt-in beyond it: the \
             DEFAULT bound is the one crossed, by {} bytes",
            bytes - top
        )
    };
    Err(format!(
        "rotation of width {n}: {bytes} shared bytes requested, {which}, and there is no \
         host-side remedy. The one-block kernel does not suit this width (see rotate.cu)."
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What an L40S reports. The default is measured — `docs/mesures/
    /// cuda-preflight-2026-08-05.txt:29`, "49152 o/bloc, 102400 o/SM" — and
    /// the opt-in is sm_89's documented 99 KiB, i.e. 100 KiB minus the 1 KiB
    /// the driver reserves. **Not** derived from `shared_per_sm − 1024` in the
    /// code: the driver owns that reservation and reports the answer, so
    /// `DeviceReport` reads the attribute. This constant is the fixture, and
    /// a card that reports otherwise changes only the fixture.
    const L40S_DEFAULT: usize = 49_152;
    const L40S_OPTIN: usize = 101_376;

    /// The four widths the deliverable actually has: `intermediate_size`, the
    /// widest activation any projection of the model takes.
    ///
    /// This is the whole table of `docs/format-noyau.md`'s shared-memory
    /// section, and it is what the guard must reproduce.
    #[test]
    fn the_four_real_widths_land_where_the_card_says() {
        let want = [
            (9_728usize, Ok(Fit::Default)),  // Qwen3-4B  — 38,912 bytes
            (12_288, Ok(Fit::Default)),      // Qwen3-8B  — 49,152 bytes, ON the bound
            (17_408, Ok(Fit::OptIn)),        // Qwen3-14B — 69,632 bytes, the job that failed
            (25_600, Err(())),               // Qwen3-32B — 102,400 bytes, over both
        ];
        for (n, expect) in want {
            let got = rot_plan(n, L40S_DEFAULT, L40S_OPTIN);
            match (got, expect) {
                (Ok(f), Ok(w)) => assert_eq!(f, w, "width {n}"),
                (Err(_), Err(())) => {}
                (g, _) => panic!("width {n}: {g:?} is not what the card allows"),
            }
        }
    }

    /// The 8B sits **exactly** on the default bound — 12,288 × 4 = 49,152 —
    /// so the comparison is `<=` and one character decides whether the model
    /// that has been shipping all along still loads. A `<` here would refuse
    /// a width the card has been running since 2026-08-08.
    #[test]
    fn the_default_bound_is_inclusive_and_the_8b_sits_on_it() {
        assert_eq!(rot_bytes(12_288), L40S_DEFAULT);
        assert_eq!(plan(L40S_DEFAULT, L40S_DEFAULT, L40S_OPTIN), Some(Fit::Default));
        assert_eq!(plan(L40S_DEFAULT + 1, L40S_DEFAULT, L40S_OPTIN), Some(Fit::OptIn));
        assert_eq!(plan(L40S_OPTIN, L40S_DEFAULT, L40S_OPTIN), Some(Fit::OptIn));
        assert_eq!(plan(L40S_OPTIN + 1, L40S_DEFAULT, L40S_OPTIN), None);
    }

    /// The 32B misses the opt-in by 1,024 bytes — one kibibyte on 100, which is
    /// the reservation the driver keeps. The refusal must survive, and it
    /// must say by how much: a guard that let this through would corrupt
    /// silently, which is what `rotate.cu` promises it would do.
    #[test]
    fn the_32b_misses_the_optin_by_1024_bytes() {
        assert_eq!(rot_bytes(25_600) - L40S_OPTIN, 1_024);
        let e = rot_plan(25_600, L40S_DEFAULT, L40S_OPTIN).unwrap_err();
        assert!(e.contains("102400"), "the message hides the bytes requested: {e}");
        assert!(e.contains("49152"), "the message hides the default bound: {e}");
        assert!(e.contains("101376"), "the message hides the opt-in bound: {e}");
        assert!(e.contains("OPT-IN"), "the message does not say which bound is crossed: {e}");
        assert!(e.contains("by 1024 bytes"), "the message hides by how much: {e}");
    }

    /// A driver that does not know the opt-in attribute reports 0 for it.
    /// That must cost nothing: everything that fitted the default still
    /// fits, and everything past it is refused — the behaviour before this
    /// change, exactly.
    #[test]
    fn an_unknown_optin_attribute_cannot_shrink_the_default() {
        assert_eq!(rot_plan(9_728, L40S_DEFAULT, 0), Ok(Fit::Default));
        assert_eq!(rot_plan(12_288, L40S_DEFAULT, 0), Ok(Fit::Default));
        assert!(rot_plan(17_408, L40S_DEFAULT, 0).is_err());
        assert_eq!(ceiling(L40S_DEFAULT, 0), L40S_DEFAULT);
    }

    /// …and on such a card the refusal must not blame an opt-in that does not
    /// exist. Same width, two cards, two different facts: 17,408 crosses the
    /// **default** where there is no opt-in, and would cross nothing at all on
    /// an L40S. A message that said "opt-in" in both cases would send the
    /// reader after an attribute the card never offered.
    #[test]
    fn the_refusal_names_the_bound_that_the_card_actually_has() {
        let none = rot_plan(17_408, L40S_DEFAULT, 0).unwrap_err();
        assert!(none.contains("DEFAULT"), "{none}");
        assert!(none.contains("no opt-in"), "{none}");
        assert!(!none.contains("OPT-IN"), "opt-in bound invoked with no opt-in: {none}");
        assert!(none.contains("by 20480 bytes"), "the message hides by how much: {none}");

        let both = rot_plan(25_600, L40S_DEFAULT, L40S_OPTIN).unwrap_err();
        assert!(both.contains("OPT-IN"), "{both}");
        assert!(!both.contains("no opt-in"), "{both}");
    }

    /// `rot_bytes` is f32 per coordinate, and the section of
    /// `docs/format-noyau.md` that quotes 69,632 bytes for the 14B depends on it.
    /// Halving it — the f16-staging idea the doc names and does not adopt —
    /// would move every verdict in that table, which is why it is pinned
    /// rather than inlined.
    #[test]
    fn the_staging_is_four_bytes_a_coordinate() {
        assert_eq!(rot_bytes(9_728), 38_912);
        assert_eq!(rot_bytes(12_288), 49_152);
        assert_eq!(rot_bytes(17_408), 69_632);
        assert_eq!(rot_bytes(25_600), 102_400);
    }
}
