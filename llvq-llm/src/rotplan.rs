//! Rotation sharing — which projections may consume one rotated activation,
//! how many times a token pays for it, and the sequencing both the model and
//! its host simulator run.
//!
//! ## The gisement
//!
//! The artifact is quantized in a rotated basis, so the fused path owes
//! `y = W' · rot(x)` for every projection (see [`crate::fused`]). Today
//! `fused_cuda` launches `rot_apply` once per *projection*: 252 launches a
//! token on the published Qwen3-4B, for **144** distinct rotations — q/k/v
//! share an activation and therefore a rotation, and so do gate/up. 108 of
//! those launches recompute a vector that was computed two lines above.
//!
//! Measured cost of one `rot_apply` at `n = 2560`, in isolation on an empty
//! stream: **8.05 µs** (`docs/mesures/rotation-cuda-2026-08-05.txt`). That
//! figure is wall-clock per chained launch, so the 3.63 µs launch floor is
//! **inside** it — adding the two would double-count, which
//! `docs/archive/audit-perf-noyau-cuda-2026-08-05.md` §1 forbids explicitly. The
//! honest bracket is therefore
//!
//!  * upper bound (whole launches removed): `108 × 8.05 µs ≈ 0.87 ms/token`;
//!  * lower bound (work only, `8.05 − 3.33`): `108 × 4.7 µs ≈ 0.51 ms/token`.
//!
//! Against the 4B's 11.31 ms/token that is **[92.6 ; 95.8] tok/s, i.e. +4.2 to
//! +7.4** over the published 88.4–88.5. *Estimated, never measured in this
//! layout on this card.*
//!
//! ## Why there is no cache here
//!
//! Memoizing a rotation means deciding that two calls carry the *same input*,
//! and every key available is a liar:
//!
//!  * **a device pointer** — candle recycles its buffers, and `narrow` hands
//!    out views sharing one base, so two different activations compare equal;
//!  * **the [`crate::fused::RotKey`]** — it repeats at every token by
//!    construction;
//!  * **`d_in`** — the key of the scratch pool this lot removes; on the 4B,
//!    q/k/v, o_proj and gate/up all sit at 2560;
//!  * **`(RotKey, step)`** — no step counter crosses `CustomOp1`, and the
//!    prefill loops underneath it.
//!
//! None of them *crashes* on a false positive. They return finite, plausible,
//! wrong logits, which is the failure mode this dossier fears most. So the
//! rotated activation is a **value** with a lexical scope
//! ([`crate::model::Rotated`]), produced next to its uses and dropped with
//! them: there is no "is this the same input?" question left to get wrong.
//! What remains is one wiring mistake — handing a group's rotation to a
//! projection of another group — and [`check_key`] closes it by name.

use std::collections::{HashMap, HashSet};

use crate::fused::{FusedMatrix, RotKey};
use crate::model::Act;

/// Whether the rotation of a shared activation is computed once for its group
/// or once per projection.
///
/// The `Off` arm is not a fallback: it issues **today's launches, launch for
/// launch**, kept so the two arms of a card measurement differ by a count and
/// nothing else. An A/B whose control arm is a rewrite measures the rewrite.
///
/// ⚠️ **One qualification, because an earlier draft said "today's code" flat
/// and that is not true at load time.** [`check_rotation_partition`] runs on
/// *both* arms, so `Off` now refuses artifacts the previous code accepted: a
/// matrix with no rotation seed, a name whose suffix is none of Qwen3's seven,
/// a name `llvq_artifact::split_name` cannot parse. That is a deliberate
/// hardening — those files were broken before, they just failed later and less
/// clearly — but it widens the control arm's failure surface, and a control
/// arm that can fail where the old one did not is worth saying out loud. What
/// is unchanged is the *forward path*: same kernels, same arguments, same
/// order, same number of launches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RotShare {
    /// One `rot_apply` per projection — 252 a token on the 4B.
    Off,
    /// One `rot_apply` per shared activation — 144 a token on the 4B.
    On,
}

impl RotShare {
    /// Parse the value of `LLVQ_ROT_SHARE`. Unset and empty mean the default;
    /// anything else must name a mode exactly.
    ///
    /// Same contract as [`crate::fused::FusedLayout::parse`], for the same
    /// reason: a typo quietly falling back to the default makes an A/B report
    /// "no effect" for a bras that never ran.
    ///
    /// **The default is `Off`** until the card gate of §4 is green. The commit
    /// that flips it carries the measurement in its message.
    pub fn parse(v: Option<&str>) -> Result<Self, String> {
        match v {
            None | Some("") => Ok(Self::Off),
            Some("0") => Ok(Self::Off),
            Some("1") => Ok(Self::On),
            Some(other) => Err(format!(
                "LLVQ_ROT_SHARE={other} : valeurs admises « 0 » (défaut, une rotation par \
                 projection) et « 1 » (une rotation par activation partagée)"
            )),
        }
    }

    /// Resolve from the environment.
    pub fn from_env() -> Result<Self, String> {
        let v = std::env::var("LLVQ_ROT_SHARE").ok();
        Self::parse(v.as_deref())
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Off => "0",
            Self::On => "1",
        }
    }
}

/// The rotation a site was handed must be the rotation that site owes.
///
/// The single wiring mistake the value-based design leaves open: three lines
/// separate the q/k/v group from the gate/up group in
/// `Block::forward_cached`, both take an activation of width `hidden_size` on
/// Qwen3-4B, and handing one group's [`crate::model::Rotated`] to the other
/// would produce finite, plausible, wrong numbers. It plants a `bail!` naming
/// the projection instead.
pub fn check_key(site: &str, want: Option<RotKey>, got: Option<RotKey>) -> Result<(), String> {
    if want == got {
        return Ok(());
    }
    Err(format!(
        "{site} : activation tournée par la rotation {got:?}, alors que cette projection \
         est quantifiée sous {want:?}. Une activation tournée est portée, jamais retrouvée — \
         c'est le mauvais groupe qui a été passé."
    ))
}

/// The activation a projection consumes, from the name the artifact stores.
///
/// 🚨 **Never group by name.** `llvq-cuda/src/seg_host.rs:158` matches
/// `name.contains("q_proj")` and is right to: its fixtures carry no rotation,
/// so the name is the only structure available. Here the rotation key exists
/// in the file, and the name is a *parallel channel* that can disagree with
/// it. This function is used only to say **which sites ought to agree**, and
/// the key itself is what says whether they do.
pub fn act_of_suffix(suffix: &str) -> Option<Act> {
    Act::ALL
        .iter()
        .copied()
        .find(|a| a.consumers().contains(&suffix))
}

/// One activation site: where it sits, which rotation it carries, how many
/// projections consume it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RotSite {
    pub layer: usize,
    pub act: Act,
    pub key: RotKey,
    pub consumers: usize,
}

/// The activation sites of a loaded model, or the two matrices that disagree.
///
/// Two implications are checked, and they are **not** the same statement:
///
///  * every projection of a `(layer, activation)` pair carries the *same*
///    rotation key — otherwise hoisting the rotation to the group would hand
///    one of them the wrong basis;
///  * no key is shared *across* two pairs — otherwise the count this lot
///    saves would be wrong, and worse, a future file could make two genuinely
///    different activations look interchangeable.
///
/// Both directions matter and a check of one alone passes on a file the other
/// breaks. Verified on the two sealed artifacts by
/// `tests/rotplan.rs::rotation_keys_partition_the_sites`: 252 matrices, 144
/// sites, consumer histogram `{3: 36, 2: 36, 1: 72}`.
pub fn rotation_sites(m: &[FusedMatrix]) -> Result<Vec<RotSite>, String> {
    // Insertion order, so the error messages and the returned list follow the
    // file rather than a hash seed.
    let mut order: Vec<(usize, Act)> = Vec::new();
    let mut sites: HashMap<(usize, Act), (RotKey, String, usize)> = HashMap::new();
    let mut owner: HashMap<RotKey, (usize, Act, String)> = HashMap::new();

    for fm in m {
        let (layer, suffix) = llvq_artifact::split_name(&fm.name).map_err(|e| e.to_string())?;
        let act = act_of_suffix(&suffix).ok_or_else(|| {
            format!(
                "{} : « {suffix} » n'est le consommateur d'aucune des quatre activations \
                 d'un bloc Qwen3",
                fm.name
            )
        })?;
        let key = fm.rotation.ok_or_else(|| {
            format!(
                "{} : aucune rotation dans le fichier. Le chemin fusé ne sait pas lire une \
                 matrice quantifiée en base naturelle (cf. fused_cuda.rs), et la partager \
                 n'aurait pas de sens.",
                fm.name
            )
        })?;

        match sites.get_mut(&(layer, act)) {
            None => {
                order.push((layer, act));
                sites.insert((layer, act), (key, fm.name.clone(), 1));
            }
            Some((k0, n0, count)) => {
                if *k0 != key {
                    return Err(format!(
                        "{n0} et {} consomment la même activation ({act:?} de la couche \
                         {layer}) mais portent deux rotations, {k0:?} et {key:?}. Le \
                         hissage donnerait à l'une la base de l'autre.",
                        fm.name
                    ));
                }
                *count += 1;
            }
        }

        match owner.get(&key) {
            None => {
                owner.insert(key, (layer, act, fm.name.clone()));
            }
            Some((l0, a0, n0)) => {
                if (*l0, *a0) != (layer, act) {
                    return Err(format!(
                        "{n0} et {} partagent la rotation {key:?} alors qu'ils consomment \
                         deux activations distinctes ({a0:?} de la couche {l0}, {act:?} de \
                         la couche {layer}).",
                        fm.name
                    ));
                }
            }
        }
    }

    Ok(order
        .into_iter()
        .map(|(layer, act)| {
            let (key, _, consumers) = sites[&(layer, act)].clone();
            RotSite { layer, act, key, consumers }
        })
        .collect())
}

/// [`rotation_sites`], discarding the sites — the load-time gate.
///
/// Called from `fused::load` before a single byte reaches a card: a file whose
/// rotation keys do not partition its activation sites is **refused**, naming
/// the two matrices, rather than silently falling back to one rotation per
/// projection. A silent fallback is the shape of defect this dossier keeps
/// paying for — it turns a broken assumption into a performance mystery.
pub fn check_rotation_partition(m: &[FusedMatrix]) -> Result<(), String> {
    rotation_sites(m).map(|_| ())
}

/// `rot_apply` launches one decode token costs, under `share`.
///
/// The number `bin/fusedrun` prints on both arms. It is printed because a card
/// gate that reads "128 tokens identiques" while both arms launched 252
/// rotations proves the tokens and nothing about the lot.
///
/// Independent of [`crate::fused::FusedLayout`] by construction: the rotation
/// acts on `x`, the layout describes `W`. Pinned by
/// `the_hoist_is_not_gated_on_the_layout`.
pub fn rot_launches_per_token(share: RotShare, m: &[FusedMatrix]) -> usize {
    match share {
        RotShare::Off => m.iter().filter(|x| x.rotation.is_some()).count(),
        RotShare::On => m
            .iter()
            .filter_map(|x| x.rotation)
            .collect::<HashSet<_>>()
            .len(),
    }
}

/// `rot_apply` launches one decode token costs, over the projections that stay
/// alone **and** the groups that were fused.
///
/// Supersedes [`rot_launches_per_token`] on the fused path, which is kept
/// because `tests/rotplan.rs` pins it on 252/144 with no groups and because it
/// is the whole of the `Off` arm.
///
/// A fused group counts as **one** launch whatever `share` says, and that is
/// not an approximation: the group is one site, so `model::group_forward`
/// rotates it once per row. `share` therefore only reaches the projections that
/// stayed alone — where, the groups having taken every shared activation, its
/// two arms coincide anyway. On the published 4B: 72 singles + 72 groups = 144,
/// the same 144 the unfused `On` arm issues.
pub fn rot_launches(
    share: RotShare,
    singles: &[FusedMatrix],
    groups: &[crate::fused::FusedGroup],
) -> usize {
    rot_launches_per_token(share, singles)
        + groups.iter().filter(|g| g.rotation.is_some()).count()
}

/// Matvec launches one decode token costs — **252 unfused, 144 fused** on the
/// published 4B.
///
/// Printed on `bin/fusedrun`'s arm line for the reason [`rot_launches`] is
/// printed there: a gate reading "128 tokens identiques" while both arms issued
/// 252 matvecs proves the tokens and nothing about the lot.
pub fn matvec_launches_per_token(
    singles: &[FusedMatrix],
    groups: &[crate::fused::FusedGroup],
) -> usize {
    singles.len() + groups.len()
}

/// Whether a run of this shape can tell the two [`crate::fused::FuseMode`] arms
/// apart.
///
/// Distinct from [`arms_are_discriminating`], which was written for the
/// `RotShare` arms, and the mechanism really is different: hoisting only
/// removes a launch where an activation has several consumers, while fusion
/// removes matvec launches from `rows == 1` onwards, the prefill included.
///
/// The **bound** is nevertheless the same, and that is deliberate rather than
/// lazy: what it buys in both cases is margin, not mechanism. A run this short
/// spends most of its wall clock in load and warm-up, so a launch-count
/// difference of a few hundred microseconds is not separable from noise and a
/// green A/B on it would mean nothing. Two predicates instead of one alias so
/// that a future change to either arm's gate cannot silently move the other's.
///
/// Here rather than in `bin/fusedrun` for the reason that function's own
/// comment gives — that binary compiles on no machine this suite runs on, and a
/// mutant weakening a bound written out there survived the whole suite once
/// already.
pub fn fuse_arms_are_discriminating(prompt_len: usize, n_new: usize) -> bool {
    prompt_len >= 2 && n_new >= 2
}

/// Whether a run of this shape can tell the two [`RotShare`] arms apart.
///
/// The hoist removes a launch only when one activation is consumed by several
/// projections **inside a decode step**. Under two new tokens the run never
/// leaves the prefill; under a two-token prompt each site is visited once. In
/// either case both arms walk the same code, and `bin/fusedrun`'s token
/// comparison prints "128 tokens identiques" for a lot that could be entirely
/// broken.
///
/// A function here rather than an `if` in `bin/fusedrun`, for the reason
/// [`crate::model::time_phases_enabled`] is one: the whole of that binary's
/// body sits behind `cfg(all(target_os = "linux", feature = "cuda"))` and
/// compiles on no machine this suite runs on. Mutation testing found it — a
/// mutant weakening the bound to zero survived the entire suite until this
/// predicate moved out. The `if` that calls it is still unchecked; the
/// *threshold*, which is the part that can be wrong, is not.
///
/// ⚠️ **The bound is conservative, not tight, and the reason first written for
/// it was wrong.** Hoisting removes launches from `rows == 1` onwards — the
/// prefill included, since q/k/v share one rotation per row whatever the
/// length — so a one-token run *would* discriminate. What the bound actually
/// buys is margin: a run this short spends most of its wall clock in load and
/// warm-up, so a launch-count difference of a few hundred microseconds is not
/// separable from noise, and a green A/B on it would mean nothing. Erring
/// restrictive costs a refused short run; erring permissive would publish a
/// ratio that measures start-up.
pub fn arms_are_discriminating(prompt_len: usize, n_new: usize) -> bool {
    prompt_len >= 2 && n_new >= 2
}

/// Run one group of projections that share an activation, over `rows` rows.
///
/// **This is the whole mechanism, and it lives here so that two callers run
/// the same code**: `crate::model::group_forward` on a card, and the host
/// simulator of `tests/rotplan.rs` on a machine that has none. A launch count
/// measured on the simulator is therefore the launch count the model issues,
/// rather than a restatement of it that can drift.
///
/// The two arms differ only in where `prepare` sits:
///
/// ```text
/// Off:  for site: for row: prepare(site,row); apply(site,row)
/// On:   for row: prepare(sites[0],row); for site: apply(site,row)
/// ```
///
/// `Off` is exactly the order `FusedRuntime::forward` issued before this lot —
/// rotation then matvec, projection by projection, row by row — which is what
/// makes it a control arm rather than a second rewrite.
///
/// Returns one vector of results per site, in row order: `out[site][row]`.
pub fn drive_rows<P, R, T, E>(
    share: RotShare,
    sites: &[P],
    rows: usize,
    prepare: impl Fn(&P, usize) -> Result<R, E>,
    apply: impl Fn(&P, &R, usize) -> Result<T, E>,
) -> Result<Vec<Vec<T>>, E> {
    let mut out: Vec<Vec<T>> = (0..sites.len()).map(|_| Vec::with_capacity(rows)).collect();
    if sites.is_empty() {
        return Ok(out);
    }
    match share {
        RotShare::Off => {
            for (s, site) in sites.iter().enumerate() {
                for row in 0..rows {
                    let r = prepare(site, row)?;
                    out[s].push(apply(site, &r, row)?);
                }
            }
        }
        RotShare::On => {
            for row in 0..rows {
                // The group's representative. Every other site is required to
                // agree with it — `check_key`, called from `apply`, is what
                // turns "they agree" from an assumption into a failure.
                let r = prepare(&sites[0], row)?;
                for (s, site) in sites.iter().enumerate() {
                    out[s].push(apply(site, &r, row)?);
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rot_share_parse_refuses_anything_else() {
        assert_eq!(RotShare::parse(None), Ok(RotShare::Off));
        assert_eq!(RotShare::parse(Some("")), Ok(RotShare::Off));
        assert_eq!(RotShare::parse(Some("0")), Ok(RotShare::Off));
        assert_eq!(RotShare::parse(Some("1")), Ok(RotShare::On));
        for bad in ["on", "off", "true", "2", "1 ", "01", "yes"] {
            let e = RotShare::parse(Some(bad)).expect_err("must be refused");
            assert!(e.contains(bad), "le message doit citer la valeur : {e}");
        }
    }

    /// The suffix→activation map is the one `model::Act` declares, and nothing
    /// outside it resolves.
    #[test]
    fn every_projection_suffix_maps_to_its_activation() {
        assert_eq!(act_of_suffix("self_attn.q_proj"), Some(Act::Attn));
        assert_eq!(act_of_suffix("self_attn.k_proj"), Some(Act::Attn));
        assert_eq!(act_of_suffix("self_attn.v_proj"), Some(Act::Attn));
        assert_eq!(act_of_suffix("self_attn.o_proj"), Some(Act::AttnOut));
        assert_eq!(act_of_suffix("mlp.gate_proj"), Some(Act::Mlp));
        assert_eq!(act_of_suffix("mlp.up_proj"), Some(Act::Mlp));
        assert_eq!(act_of_suffix("mlp.down_proj"), Some(Act::MlpOut));
        assert_eq!(act_of_suffix("q_proj"), None);
        assert_eq!(act_of_suffix("self_attn.qkv_proj"), None);
    }

    #[test]
    fn check_key_names_the_projection() {
        assert!(check_key("m", Some((2560, 7)), Some((2560, 7))).is_ok());
        assert!(check_key("m", None, None).is_ok());
        let e = check_key("model.layers.3.mlp.up_proj.weight", Some((2560, 7)), Some((2560, 9)))
            .expect_err("mismatched keys must be refused");
        assert!(e.contains("mlp.up_proj"), "{e}");
    }

    /// The card gate's own guard clause, at its boundary.
    ///
    /// Found by mutation: weakening the bound to zero survived the whole suite
    /// while the `if` lived inside `bin/fusedrun`'s cfg-gated body. A gate that
    /// cannot discriminate is worse than no gate — it prints a tick.
    #[test]
    fn a_run_too_short_cannot_tell_the_arms_apart() {
        for (p, n) in [(0, 0), (1, 128), (128, 1), (1, 1), (5, 0), (0, 5)] {
            assert!(
                !arms_are_discriminating(p, n),
                "prompt {p}, {n} nouveaux : les deux bras parcourent le même chemin"
            );
        }
        for (p, n) in [(2, 2), (5, 128), (2, 128), (128, 2)] {
            assert!(arms_are_discriminating(p, n), "prompt {p}, {n} nouveaux");
        }
    }

    /// `drive_rows` on an empty group is empty, not a panic on `sites[0]`.
    #[test]
    fn an_empty_group_drives_nothing() {
        let out: Vec<Vec<u32>> = drive_rows(
            RotShare::On,
            &[] as &[u32],
            4,
            |_: &u32, _| Err::<u32, String>("never".into()),
            |_: &u32, _: &u32, _| Err::<u32, String>("never".into()),
        )
        .expect("empty group");
        assert!(out.is_empty());
    }
}
