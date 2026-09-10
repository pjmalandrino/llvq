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
    /// "no effect" for an arm that never ran.
    ///
    /// **The default is `Off`** until the card gate of §4 is green. The commit
    /// that flips it carries the measurement in its message.
    pub fn parse(v: Option<&str>) -> Result<Self, String> {
        match v {
            None | Some("") => Ok(Self::Off),
            Some("0") => Ok(Self::Off),
            Some("1") => Ok(Self::On),
            Some(other) => Err(format!(
                "LLVQ_ROT_SHARE={other}: accepted values \"0\" (default, one rotation per \
                 projection) and \"1\" (one rotation per shared activation)"
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
        "{site}: activation rotated by {got:?}, while this projection \
         is quantized under {want:?}. A rotated activation is carried, never recovered: \
         the wrong group was passed."
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
                "{}: \"{suffix}\" consumes none of the four activations \
                 of a Qwen3 block",
                fm.name
            )
        })?;
        let key = fm.rotation.ok_or_else(|| {
            format!(
                "{}: no rotation in the file. The fused path cannot read a \
                 matrix quantized in the natural basis (see fused_cuda.rs), and sharing \
                 it would make no sense.",
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
                        "{n0} and {} consume the same activation ({act:?} of layer \
                         {layer}) but carry two rotations, {k0:?} and {key:?}. The \
                         hoist would give one the other's basis.",
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
                        "{n0} and {} share rotation {key:?} while they consume \
                         two distinct activations ({a0:?} of layer {l0}, {act:?} of \
                         layer {layer}).",
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
///
/// ## Why `int4` is a third argument and not an oversight
///
/// The served object of 2026-09-08 is mixed: 216 lattice records and 36
/// `v_proj` in int4 g128. Those 36 live in their own vector because they take
/// their own kernel (`tv_q4_h`), and for one decode token each costs **one
/// launch**, exactly like a lone lattice projection. They neither fuse nor
/// group, so the term is a plain count.
///
/// Omitting it printed `216 matvec_launches/token for 252 projections` on the
/// first card run of the served object (job `6aa2e938`, 2026-09-10) — a line
/// that reads as "36 projections cost no launch". The tokens, the memory and
/// the speed on that run were right; only this number was wrong. It is the
/// number the sentence above says the gate rests on, so a counter that
/// undercounts weakens precisely the guard it exists to arm.
pub fn matvec_launches_per_token(
    singles: &[FusedMatrix],
    groups: &[crate::fused::FusedGroup],
    int4: usize,
) -> usize {
    singles.len() + groups.len() + int4
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
/// On:   for row: prepare(REP,row); for site: apply(site,row)   [if it shares]
///                                 else      prepare(site,row); apply(site,row)
/// ```
///
/// `shares` says whether a site takes the group's rotation. A site that does
/// not is **not disagreeing** with the group: it consumes the activation in
/// the basis it already arrives in, and handing it the group's rotated form
/// would hand it a basis its weights were never quantized in — which
/// [`check_key`] refuses, by design.
///
/// 🕳️ Before 2026-09-10 `On` prepared from `sites[0]` unconditionally and
/// handed that to everyone. That was right while every projection of a group
/// was a lattice one. It stopped being right when the served object became a
/// mixed file: its `v_proj` is int4, stored group-affine in the NATURAL basis,
/// so a q+k+v group under `LLVQ_ROT_SHARE=1` — the served setting — failed at
/// `check_key` with the group's own key. The refusal was correct and the
/// sharing was wrong.
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
    shares: impl Fn(&P) -> bool,
    prepare: impl Fn(&P, usize) -> Result<R, E>,
    apply: impl Fn(&P, &R, usize) -> Result<T, E>,
) -> Result<Vec<Vec<T>>, E> {
    // One row a call is a batch of one, and that is the whole relationship:
    // there is ONE mechanism, and the per-row path is its degenerate case.
    // Anything else would be two descriptions of the sharing rule.
    drive_rows_batched(share, sites, rows, 1, shares, prepare, |p, rs, row0| {
        Ok(vec![apply(p, rs[0], row0)?])
    })
}

/// [`drive_rows`], `batch` rows a call.
///
/// ## Why it exists
///
/// The served matvec reads ONE activation row a launch, so a prompt of N
/// tokens decodes the whole weight stream N times: on the served 4B a 5-shot
/// MMLU question is several hundred tokens, i.e. ~776 GB of re-reads and
/// ~200,000 launches. The prefill kernel takes `PREFILL_ROWS` rows a launch
/// and reads the stream that many times less; this is the loop that feeds it.
///
/// ## What `batch` does NOT change
///
///  * the number of `prepare` calls — one a row a site that does not share,
///    one a row for the group that does. A rotation is per row and batching
///    the matvec does not batch it;
///  * the ORDER of the results. `out[site][row]` is indexed by row, and the
///    chunks are walked in increasing row order;
///  * the sharing rule. Under [`RotShare::On`] one rotation a row serves every
///    site that has a key, and a site with none prepares its own — which for a
///    natural-basis projection is the activation untouched.
///
/// ## The tail
///
/// The last chunk is short when `rows % batch != 0`, and `apply_rows` is handed
/// exactly the rotations it must answer for. It returns one result per
/// rotation; a shorter or longer answer is a bug in the caller and is refused
/// here rather than silently reindexed — a row landing one position off is the
/// failure this file's comments keep naming, finite and plausible and wrong.
pub fn drive_rows_batched<P, R, T, E>(
    share: RotShare,
    sites: &[P],
    rows: usize,
    batch: usize,
    shares: impl Fn(&P) -> bool,
    prepare: impl Fn(&P, usize) -> Result<R, E>,
    apply_rows: impl Fn(&P, &[&R], usize) -> Result<Vec<T>, E>,
) -> Result<Vec<Vec<T>>, E> {
    let mut out: Vec<Vec<T>> = (0..sites.len()).map(|_| Vec::with_capacity(rows)).collect();
    if sites.is_empty() {
        return Ok(out);
    }
    let batch = batch.max(1);
    // One `push` a row a site, checked at the end: `apply_rows` returning the
    // wrong count would otherwise shift every row after it.
    let mut chunks: Vec<(usize, usize)> = Vec::new();
    let mut lo = 0usize;
    while lo < rows {
        let len = batch.min(rows - lo);
        chunks.push((lo, len));
        lo += len;
    }

    match share {
        RotShare::Off => {
            for (s, site) in sites.iter().enumerate() {
                for &(lo, len) in &chunks {
                    let mut rot: Vec<R> = Vec::with_capacity(len);
                    for row in lo..lo + len {
                        rot.push(prepare(site, row)?);
                    }
                    let refs: Vec<&R> = rot.iter().collect();
                    let got = apply_rows(site, &refs, lo)?;
                    if got.len() != len {
                        // Not an assert: the caller owns the kernel and this is
                        // its contract, so it must be able to see the message.
                        refuse_count(s, lo, len, got.len());
                    }
                    out[s].extend(got);
                }
            }
        }
        RotShare::On => {
            // The representative is the first site that HAS a rotation to
            // share — not `sites[0]`, which may be one that has none.
            let rep = sites.iter().position(&shares);
            for &(lo, len) in &chunks {
                // Prepared once a row, and only if anyone shares it. Every
                // site that does is required to agree with it — `check_key`,
                // called from `apply_rows`, is what turns "they agree" from an
                // assumption into a failure.
                let shared: Option<Vec<R>> = match rep {
                    Some(i) => {
                        let mut v = Vec::with_capacity(len);
                        for row in lo..lo + len {
                            v.push(prepare(&sites[i], row)?);
                        }
                        Some(v)
                    }
                    None => None,
                };
                for (s, site) in sites.iter().enumerate() {
                    let own: Option<Vec<R>> = match (&shared, shares(site)) {
                        (Some(_), true) => None,
                        // Its own, which for a natural-basis projection is the
                        // activation untouched and launches nothing.
                        _ => {
                            let mut v = Vec::with_capacity(len);
                            for row in lo..lo + len {
                                v.push(prepare(site, row)?);
                            }
                            Some(v)
                        }
                    };
                    let refs: Vec<&R> = match (&own, &shared) {
                        (Some(v), _) => v.iter().collect(),
                        (None, Some(v)) => v.iter().collect(),
                        (None, None) => unreachable!("no rotation for a site that shares none"),
                    };
                    let got = apply_rows(site, &refs, lo)?;
                    if got.len() != len {
                        refuse_count(s, lo, len, got.len());
                    }
                    out[s].extend(got);
                }
            }
        }
    }
    out
        .iter()
        .enumerate()
        .for_each(|(s, v)| assert_eq!(v.len(), rows, "site {s} produced {} of {rows} rows", v.len()));
    Ok(out)
}

/// A count mismatch from `apply_rows`, made loud where the caller can see it.
///
/// It cannot return `E` — this function is generic over the caller's error and
/// has no way to build one — so it panics, which is right: a kernel that
/// answered a different number of rows than it was asked for has no correct
/// continuation, and shifting every row after it is exactly the silent wrong
/// number this file exists to prevent.
fn refuse_count(site: usize, row0: usize, want: usize, got: usize) -> ! {
    panic!("site {site}, rows {row0}..{}: the batched apply answered {got} rows", row0 + want)
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
            assert!(e.contains(bad), "the message must cite the value: {e}");
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
                "prompt {p}, {n} new: both arms walk the same path"
            );
        }
        for (p, n) in [(2, 2), (5, 128), (2, 128), (128, 2)] {
            assert!(arms_are_discriminating(p, n), "prompt {p}, {n} new");
        }
    }

    /// `drive_rows` on an empty group is empty, not a panic on `sites[0]`.
    #[test]
    fn an_empty_group_drives_nothing() {
        let out: Vec<Vec<u32>> = drive_rows(
            RotShare::On,
            &[] as &[u32],
            4,
            |_: &u32| true,
            |_: &u32, _| Err::<u32, String>("never".into()),
            |_: &u32, _: &u32, _| Err::<u32, String>("never".into()),
        )
        .expect("empty group");
        assert!(out.is_empty());
    }

    /// The batched loop covers every row exactly once, in order, and the tail
    /// chunk is short.
    ///
    /// A row landing one position off is finite, plausible and wrong — the
    /// failure this file names three times — and no card would report it.
    #[test]
    fn the_batched_loop_covers_every_row_once_and_in_order() {
        for (rows, batch) in [(10usize, 4usize), (8, 4), (1, 4), (3, 4), (7, 1), (0, 4)] {
            let seen = std::cell::RefCell::new(Vec::<(usize, usize)>::new());
            let sites = [true, true];
            let out: Vec<Vec<usize>> = drive_rows_batched(
                RotShare::Off,
                &sites,
                rows,
                batch,
                |h: &bool| *h,
                |_: &bool, row: usize| Ok::<usize, String>(row),
                |_: &bool, rs: &[&usize], row0: usize| {
                    seen.borrow_mut().push((row0, rs.len()));
                    // The answer IS the rotation, so a chunk that reordered or
                    // reindexed its rows shows up in `out`.
                    Ok::<Vec<usize>, String>(rs.iter().map(|r| **r).collect())
                },
            )
            .expect("drives");
            assert_eq!(out.len(), 2);
            for v in &out {
                assert_eq!(*v, (0..rows).collect::<Vec<_>>(), "rows {rows}, batch {batch}");
            }
            // The chunks tile [0, rows) with no gap and no overlap, and only
            // the last one is short.
            let chunks = seen.borrow();
            let per_site: Vec<(usize, usize)> = chunks[..chunks.len() / 2].to_vec();
            let mut next = 0usize;
            for (i, &(lo, len)) in per_site.iter().enumerate() {
                assert_eq!(lo, next, "rows {rows}, batch {batch}: chunk {i} starts at {lo}");
                assert!(len <= batch && len > 0);
                if i + 1 < per_site.len() {
                    assert_eq!(len, batch, "only the last chunk may be short");
                }
                next += len;
            }
            assert_eq!(next, rows, "rows {rows}, batch {batch}: the chunks do not tile");
        }
    }

    /// A batched apply that answers the wrong number of rows is refused, not
    /// reindexed.
    ///
    /// 🕳️ Nothing tested this until a mutant survived: removing the check
    /// changed no test, because every fixture answered correctly. A kernel
    /// that returned three rows for four would otherwise shift every row after
    /// it — finite, plausible, wrong, and no card would say so.
    #[test]
    fn an_apply_that_answers_the_wrong_count_is_refused() {
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        for (asked, answered) in [(4usize, 3usize), (4, 5), (2, 0)] {
            let r = std::panic::catch_unwind(move || {
                let sites = [true];
                drive_rows_batched(
                    RotShare::Off,
                    &sites,
                    asked,
                    asked,
                    |h: &bool| *h,
                    |_: &bool, row: usize| Ok::<usize, String>(row),
                    move |_: &bool, _: &[&usize], _| {
                        Ok::<Vec<usize>, String>(vec![0; answered])
                    },
                )
            });
            assert!(
                r.is_err(),
                "{asked} rows asked, {answered} answered, and the loop accepted it"
            );
        }

        // 🚨 The case the TOTAL cannot see: one chunk short, the next long, and
        // the sum right. Only the per-chunk check catches it, and without it
        // every row after the first chunk sits one position off while the
        // count comes out perfect.
        let r = std::panic::catch_unwind(|| {
            let sites = [true];
            let n = std::cell::Cell::new(0usize);
            drive_rows_batched(
                RotShare::Off,
                &sites,
                8,
                4,
                |h: &bool| *h,
                |_: &bool, row: usize| Ok::<usize, String>(row),
                move |_: &bool, _: &[&usize], _| {
                    n.set(n.get() + 1);
                    // 3 then 5: eight in total, four asked each time.
                    Ok::<Vec<usize>, String>(vec![0; if n.get() == 1 { 3 } else { 5 }])
                },
            )
        });
        assert!(
            r.is_err(),
            "a short chunk balanced by a long one was accepted: the total is right and \
             every row after the first chunk is one position off"
        );

        // 🕳️ And the SAME case under `On`. The two arms carry the check
        // separately, and a mutant that removed only one of them survived a
        // sweep that exercised only `Off` — the harness had reported it as a
        // surviving mutant when it was a half-applied one.
        let r = std::panic::catch_unwind(|| {
            let sites = [true, false];
            let n = std::cell::Cell::new(0usize);
            drive_rows_batched(
                RotShare::On,
                &sites,
                8,
                4,
                |h: &bool| *h,
                |_: &bool, row: usize| Ok::<usize, String>(row),
                move |_: &bool, _: &[&usize], _| {
                    n.set(n.get() + 1);
                    Ok::<Vec<usize>, String>(vec![0; if n.get() == 1 { 3 } else { 4 }])
                },
            )
        });
        assert!(r.is_err(), "the shared-rotation arm accepted a short chunk");
        std::panic::set_hook(prev);
    }

    /// At `batch = 1` the batched loop IS the per-row one — same calls, same
    /// order — because `drive_rows` is written as that case and nothing else.
    #[test]
    fn a_batch_of_one_is_the_per_row_loop() {
        let sites = [true, false, true];
        let log_a = std::cell::RefCell::new(Vec::<String>::new());
        let a: Vec<Vec<usize>> = drive_rows(
            RotShare::On,
            &sites,
            3,
            |h: &bool| *h,
            |h: &bool, row: usize| {
                log_a.borrow_mut().push(format!("prep {h} {row}"));
                Ok::<usize, String>(row * 10 + usize::from(*h))
            },
            |h: &bool, r: &usize, row: usize| {
                log_a.borrow_mut().push(format!("app {h} {row}"));
                Ok::<usize, String>(*r)
            },
        )
        .expect("per row");
        let log_b = std::cell::RefCell::new(Vec::<String>::new());
        let b: Vec<Vec<usize>> = drive_rows_batched(
            RotShare::On,
            &sites,
            3,
            1,
            |h: &bool| *h,
            |h: &bool, row: usize| {
                log_b.borrow_mut().push(format!("prep {h} {row}"));
                Ok::<usize, String>(row * 10 + usize::from(*h))
            },
            |h: &bool, rs: &[&usize], row0: usize| {
                log_b.borrow_mut().push(format!("app {h} {row0}"));
                Ok::<Vec<usize>, String>(vec![*rs[0]])
            },
        )
        .expect("batched at one");
        assert_eq!(a, b, "the two loops disagree on the result");
        assert_eq!(*log_a.borrow(), *log_b.borrow(), "the two loops disagree on the call order");
    }

    /// Under `On`, a chunk prepares the group's rotation ONCE a row — not once
    /// a row a site — and the sites that share it all see the same value.
    #[test]
    fn a_chunk_prepares_the_shared_rotation_once_a_row() {
        let sites = [true, true, false];
        let preps = std::cell::RefCell::new(0usize);
        let out: Vec<Vec<usize>> = drive_rows_batched(
            RotShare::On,
            &sites,
            4,
            4,
            |h: &bool| *h,
            |h: &bool, row: usize| {
                *preps.borrow_mut() += 1;
                Ok::<usize, String>(if *h { 100 + row } else { row })
            },
            |_: &bool, rs: &[&usize], _| Ok::<Vec<usize>, String>(rs.iter().map(|r| **r).collect()),
        )
        .expect("drives");
        // Four rows: four for the group, four for the site that shares none.
        assert_eq!(*preps.borrow(), 8, "the shared rotation was prepared per site");
        assert_eq!(out[0], vec![100, 101, 102, 103]);
        assert_eq!(out[1], out[0], "the two sharers saw different rotations");
        assert_eq!(out[2], vec![0, 1, 2, 3], "the natural-basis site was handed the group's");
    }

    /// A site with no rotation gets its OWN, and the sharers still share one.
    ///
    /// 🕳️ The served object's `v_proj` is int4, stored in the natural basis.
    /// Under `RotShare::On` — the served setting — `drive_rows` used to prepare
    /// from `sites[0]` and hand that to everyone, so a q+k+v group failed at
    /// `check_key` with the group's own key. Correct refusal, wrong sharing.
    #[test]
    fn a_site_without_a_rotation_takes_no_share_of_the_group_s() {
        // `true` where a site has a key. q and k share; v (int4) does not.
        let sites = [true, true, false];
        let prepared = std::cell::RefCell::new(Vec::<(usize, usize)>::new());
        let out: Vec<Vec<usize>> = drive_rows(
            RotShare::On,
            &sites,
            2,
            |has: &bool| *has,
            |has: &bool, row: usize| -> Result<usize, String> {
                prepared.borrow_mut().push((row, usize::from(*has)));
                // The "rotation" is 10·row for a sharer, 0 for a natural-basis
                // site — so `apply` below can tell which one it was handed.
                Ok(if *has { 10 * row + 1 } else { 0 })
            },
            |has: &bool, r: &usize, _row: usize| -> Result<usize, String> {
                // The check the real `apply` makes: a natural-basis site must
                // never see a rotated form.
                if !*has && *r != 0 {
                    return Err("a natural-basis site was handed a rotation".into());
                }
                Ok(*r)
            },
        )
        .expect("the mixed group must drive");

        // One prepare a row for the two sharers TOGETHER, plus one for the
        // site that does not share: two rows × two prepares.
        assert_eq!(prepared.borrow().len(), 4);
        assert_eq!(*prepared.borrow(), vec![(0, 1), (0, 0), (1, 1), (1, 0)]);
        // q and k saw the SAME rotation, v saw none.
        assert_eq!(out[0], vec![1, 11]);
        assert_eq!(out[1], vec![1, 11]);
        assert_eq!(out[2], vec![0, 0]);
    }

    /// And a group where NOBODY shares still runs: the representative is an
    /// `Option`, not `sites[0]`.
    #[test]
    fn a_group_of_natural_basis_sites_needs_no_representative() {
        let sites = [false, false];
        let out: Vec<Vec<usize>> = drive_rows(
            RotShare::On,
            &sites,
            1,
            |has: &bool| *has,
            |_: &bool, row: usize| Ok::<usize, String>(row),
            |_: &bool, r: &usize, _| Ok::<usize, String>(*r),
        )
        .expect("no representative needed");
        assert_eq!(out, vec![vec![0], vec![0]]);
    }
}
