//! The 4E bench arms built as text edits of v9's module: experiment E1's
//! prefetch arms, the rc row bodies and the k8 int8 cache, with their
//! layouts. One file because every arm is v9's text plus edits, checked alike.

use super::{p4e_attn_ring, p4e_rows_define, pers_ft_edit, PERS_UNIT_PARTS};

// ---------------------------------------------------------------------------
// Experiment E1 (docs/mesures/large-4e-budget-2026-11-11.txt, cut 1): where
// 4E v9 asks L2 for later rows. Three bench modules, each 4E v9's text with
// edits applied as text, so the served v9 module (pinned below,
// `the_4e_v9_module_is_the_one_the_doors_gate_on`) does not move:
//   "v9"   v9 and a logits symbol (the reference the arms are compared with);
//   "v9np" "v9" without the attention phase's prefetch of the o and gate/up
//          rows: nothing else;
//   "v9mv" "v9np" with layer l + 1's q/k/v and o rows asked of L2 during
//          layer l's gate/up rows (after ln2), layer 0's before its ln1.
//   "v9ft" "v9" with the attention tile's full-tile path ([`pers_ft_edit`],
//          docs/mesures/large-attn-tuile-pleine-mac-2026-11-11.txt).
//   "v9ep" "v9" with every Tetra row phase on rowbench's `endpass` body
//          ([`p4e_rc_edits`], kernels/pers_4e_rc.cuh): the bits.
//   "v9fl" "v9" with every Tetra row phase on rowbench's `flat` body: a
//          bound of 1e-5 on rows, NOT the bits ([`p4e_e1_owes_bits`]).
// ---------------------------------------------------------------------------

/// The E1 arms, the first the reference of the others. `v9k8` (phase O13 of
/// docs/CERVEAU.md) is v9 with the KV cache held as group-affine int8
/// ([`p4e_k8_edits`], kernels/pers_4e_k8.cuh): its runtime keeps int8 caches
/// of its own, filled from the prefill's binary16 cache.
pub const P4E_E1_ARMS: [&str; 14] =
    ["v9", "v9np", "v9mv", "v9ft", "v9ep", "v9fl", "v9ep2", "v9ep3", "v9fl2", "v9fl3", "v9fn", "v9fg", "v9eg", "v9k8"];

/// The research bodies' arms and their body codes (`PERS_RC_ARM`): 1 endpass,
/// 2 flat (pers_4e_rc.cuh); 3 ep2, 4 ep3, 5 fl2, 6 fl3 (pers_4e_rc2.cuh, the
/// lighter variants after the first card run refused 1 and 2 for 8 B of
/// local memory, docs/mesures/recherche-debit-jeton-2026-11-13.txt).
pub(super) const P4E_RC_ARMS: [(&str, u32); 6] = [("v9ep", 1), ("v9fl", 2), ("v9ep2", 3), ("v9ep3", 4), ("v9fl2", 5), ("v9fl3", 6)];

/// Load probes of the endpass body (`LLVQ_PERS_4E=check:` only, never timed,
/// never launched): E1's v9 with pers_4e_rc.cuh inserted and the body on ONE
/// call site, or on none. The index is the site's edit in
/// [`p4e_rc_edits`]`(1)`: 0 none, 1 q/k, 2 gate/up, 3 down, 4 o. The six
/// whole arms were each refused for the same 8 B of local memory
/// (docs/mesures/recherche-debit-charge2-2026-11-14.txt); the probes say
/// which site, if one, brings them. They are not E1 arms: no chain test
/// covers a module with the body on one site.
pub const P4E_RC_PROBES: [(&str, usize); 5] = [("v9pe0", 0), ("v9peq", 1), ("v9peg", 2), ("v9ped", 3), ("v9peo", 4)];

/// The gate/up call site of an R6d arm ([`P4E_RC_GU_ARMS`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RcGu {
    /// v9's own gate/up text: edit 2 of [`p4e_rc_edits`] dropped.
    V9,
    /// [`P4E_RC_GU_TWO_CALLS`] in place of edit 2's replacement: the CTA's
    /// gate rows in one call, its up rows in a second, one matrix a call.
    TwoCalls,
}

/// R6d's arms (docs/mesures/recherche-debit-sondes-2026-11-14.txt: with the
/// endpass body on one call site, only gate/up brings the 8 B of local
/// memory): the body on q/k, the Tetra down and the Tetra o, and on gate/up
/// either v9's text or the two-call site. (arm, body code, gate/up site):
///   v9fn flat, v9's gate/up: flat's rows on the other sites, so flat's bound;
///   v9fg flat, two calls: flat's rows everywhere, so v9fl's chain byte for byte;
///   v9eg endpass, two calls: v9's arithmetic, so the bits.
pub(super) const P4E_RC_GU_ARMS: [(&str, u32, RcGu); 3] = [("v9fn", 2, RcGu::V9), ("v9fg", 2, RcGu::TwoCalls), ("v9eg", 1, RcGu::TwoCalls)];

/// The two-call gate/up site of R6d (replaces the same v9 text as edit 2 of
/// [`p4e_rc_edits`]). Call 1 parks silu(g) of each gate row in `act`; call 2
/// multiplies it by the row's up value: exact_hmul(silu[f2h(g)], f2h(u)), the
/// stream's own product. `act` rows [glo, ghi) are this CTA's alone, and no
/// CTA reads them before the grid barrier after the MLP's gate/up part; each
/// body ends on a barrier after its last epilogue, so call 2 reads what call 1
/// wrote. No value is kept in shared memory across the calls.
pub(super) const P4E_RC_GU_TWO_CALLS: &str = "        // rc, two calls: the CTA's gate rows, silu of each parked in act; after the body's last barrier its up
        // rows, each product exact_hmul(silu[g], u), the stream's own. One matrix a call, no value kept in shared.
        const u32 glo = PERS_SEG_GU(0u), ghi = PERS_SEG_GU(1u);
        const u32 nb = A->gate.nblocks, tw = A->gate.tail_w;
        {
            const auto row_at = [&](u32 idx) {
                const u32 r = glo + idx;
                return PersTRow{A->gate.words + r * A->gate.stride_u32, A->gate.gscale, A->gate.rscale, A->gate.tail, r};
            };
            const auto epi = [&](u32 idx, float y) { A->act[glo + idx] = A->silu[f2h(y)]; };
            pers_rc_rows<PERS_RC_ARM>(ghi - glo, nb, tw, tab, t_inv, v, row_at, epi, v + 2u * H, false);
        }
        {
            const auto row_at = [&](u32 idx) {
                const u32 r = glo + idx;
                return PersTRow{A->up.words + r * A->up.stride_u32, A->up.gscale, A->up.rscale, A->up.tail, r};
            };
            const auto epi = [&](u32 idx, float y) { A->act[glo + idx] = exact_hmul(A->act[glo + idx], f2h(y)); };
            pers_rc_rows<PERS_RC_ARM>(ghi - glo, nb, tw, tab, t_inv, v, row_at, epi, v + 2u * H, false);
        }
";

/// The body code of a research arm, or none.
pub fn p4e_rc_body(arm: &str) -> Option<u32> {
    P4E_RC_ARMS
        .iter()
        .find(|a| a.0 == arm)
        .map(|a| a.1)
        .or_else(|| P4E_RC_GU_ARMS.iter().find(|a| a.0 == arm).map(|a| a.1))
}

/// The edits of an R6d arm: [`p4e_rc_edits`] of its body, edit 2 (gate/up)
/// dropped (v9fn) or its replacement swapped for [`P4E_RC_GU_TWO_CALLS`]
/// (v9fg, v9eg). The text, q/k, down and o edits are the body's own.
pub(super) fn p4e_rc_gu_edits(arm: &str) -> Result<Vec<(&'static str, String, String)>, String> {
    let &(_, body, gu) =
        P4E_RC_GU_ARMS.iter().find(|a| a.0 == arm).ok_or_else(|| format!("4E rc: {arm:?} is not an R6d arm"))?;
    let mut v = p4e_rc_edits(body)?;
    if v.len() != 5 || v[2].0 != "pers_segment.cu" || !v[2].1.contains("A->act[pend_r] = exact_hmul(pend_sv, pend_u);") {
        return Err(format!("4E rc {arm}: edit 2 of body {body} is not v9's gate/up site"));
    }
    match gu {
        RcGu::V9 => {
            v.remove(2);
        }
        RcGu::TwoCalls => v[2].2 = P4E_RC_GU_TWO_CALLS.to_string(),
    }
    Ok(v)
}

/// Whether a body is flat's family (owes the bound, not the bits).
pub fn rc_body_is_flat(body: u32) -> bool {
    matches!(body, 2 | 5 | 6)
}

/// Whether a body runs the CTA's slice in one chunk of slots (it traps
/// otherwise): ep3, fl2, fl3.
pub fn rc_body_one_chunk(body: u32) -> bool {
    matches!(body, 4..=6)
}

/// Whether an E1 arm owes v9's bits (its tokens and logits equal v9's). Every
/// arm does but flat's (`v9fl`, `v9fl2`, `v9fl3`, `v9fn`, `v9fg`), whose rows owe a bound of
/// 1e-5 (pers_4e_rc.cuh): their tokens and logits are compared with v9's and
/// reported, never assumed equal.
pub fn p4e_e1_owes_bits(arm: &str) -> bool {
    !p4e_rc_body(arm).is_some_and(rc_body_is_flat) && !p4e_e1_k8(arm)
}

/// What an E1 arm's tokens are held to, as a harness prints it: v9's bits,
/// flat's bound on rows, or (v9k8) the int8 cache's values, whose tokens
/// and logits are compared with v9's and reported.
pub fn p4e_e1_debt(arm: &str) -> &'static str {
    if p4e_e1_k8(arm) {
        "the int8 cache's values (kv8_ref), not v9's bits"
    } else if p4e_e1_owes_bits(arm) {
        "the bits"
    } else {
        "a bound on rows"
    }
}

/// 4E's load rule on a token symbol: one CTA a SM, and no local memory. The
/// second half is waived only by `spill`, a MEASUREMENT (`fusedrun
/// LLVQ_PERS_4E`, an arm named `<arm>+local`): no arm that holds local memory
/// had been timed before 2026-11-15, the rule refusing it first, so what the
/// rule protects had no figure. No served path passes `spill`.
pub fn p4e_token_symbol_loads(local_bytes: i32, resident_a_sm: u32, spill: bool) -> bool {
    resident_a_sm >= 1 && (local_bytes == 0 || spill)
}

/// The suffix that names an arm timed with local memory in `LLVQ_PERS_4E`.
pub const P4E_LOCAL_SUFFIX: &str = "+local";

/// Whether an E1 arm holds the KV cache as int8 (`v9k8`): its runtime
/// allocates the int8 caches and fills them with `pers_kv8_fill`.
pub fn p4e_e1_k8(arm: &str) -> bool {
    arm == "v9k8"
}

/// The int8 cache's text (arm v9k8), inserted at the head of pers_4e.cu by
/// [`p4e_k8_edits`].
pub(super) const P4E_K8_PART: (&str, &str) = ("pers_4e_k8.cuh", include_str!("../../kernels/pers_4e_k8.cuh"));

/// The symbol of the fill a v9k8 module carries: positions of the prefill's
/// binary16 cache quantized into the int8 caches (pers_4e_k8.cuh).
pub const P4E_K8_FILL_SYMBOL: &str = "pers_kv8_fill";

/// Bytes of one int8 cache block (8 positions of one head): `kv8_ref::BLK`.
pub const P4E_K8_BLK: u32 = 1088;

/// Arm v9k8's int8 caches between launches, as its runtime keeps them:
/// their capacity in positions (a multiple of 256, 0 before the first
/// launch), the positions quantized from the binary16 cache the prefill
/// wrote (`[0, hand)`), and the end of the arm's own last launch (`own`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct K8State {
    pub cap: usize,
    pub hand: usize,
    pub own: usize,
}

/// What the runtime does before a v9k8 launch ([`p4e_k8_step`]): grow the
/// int8 caches (copying what they hold when they held something), fill
/// positions `[a, b)` from the binary16 caches, then run with `next`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct K8Step {
    pub grow: bool,
    pub copy_old: bool,
    pub fill: Option<(usize, usize)>,
    pub next: K8State,
}

/// The rule of a v9k8 launch of `n` tokens at `pos0`, the binary16 caches
/// holding `len16` positions. `moved`: their buffers are not the ones the
/// int8 caches were filled from (a new context, or a cache the eager path
/// grew), so nothing quantized before is trusted. Two launches are allowed:
/// a handoff (`len16 == pos0`: the binary16 caches hold every position
/// before the launch; positions `[hand, pos0)` are quantized, and the arm
/// owns what follows `pos0`), or the arm continuing its own decode (`pos0`
/// is where its last launch ended, past `len16`, nothing moved). Anything
/// else is refused: the int8 caches would hold positions nobody wrote. The
/// capacity is the next multiple of 256 at or past `pos0 + n`, never
/// shrinking, at most `max_cap` (the attention scratch). The contract the
/// rule rests on: the eager path writes a cache only at its length, and a
/// cache truncated below `hand` and written again arrives as `moved`.
pub fn p4e_k8_step(st: K8State, moved: bool, len16: usize, pos0: usize, n: usize, max_cap: usize) -> Result<K8Step, String> {
    let st = if moved { K8State { cap: st.cap, hand: 0, own: 0 } } else { st };
    let continuing = !moved && st.own > 0 && pos0 == st.own && len16 < pos0;
    if !continuing && len16 != pos0 {
        return Err(format!(
            "4E v9k8: a launch at position {pos0}, the binary16 caches holding {len16} and the arm's own decode \
             ending at {}{}: neither a handoff nor a continuation",
            st.own,
            if moved { " (the binary16 caches moved)" } else { "" }
        ));
    }
    let cap = st.cap.max((pos0 + n).div_ceil(256) * 256);
    if cap > max_cap {
        return Err(format!("4E v9k8: {} positions, the attention scratch holds {max_cap}", pos0 + n));
    }
    let grow = cap != st.cap;
    let fill = (!continuing && pos0 > st.hand).then_some((st.hand, pos0));
    let hand = if continuing { st.hand } else { pos0 };
    Ok(K8Step { grow, copy_old: grow && st.cap > 0 && st.own.max(hand) > 0, fill, next: K8State { cap, hand, own: st.own } })
}

/// The rule of a v9k8 launch when the binary16 caches are released at each
/// hand-over (the NOT SERVED `decode_cache: int8` form): after the fill, each
/// layer's binary16 cache is freed, and the int8 caches are the only copy.
/// `has_f16`: every layer's binary16 cache holds its buffers (a prefill just
/// wrote them), `len16` positions. Two launches are allowed: a hand-over
/// (`has_f16` and `len16 == pos0`: positions `[0, pos0)` are quantized, from
/// 0 since nothing quantized before survives a new prefill, the capacity
/// kept when it holds the launch), after which the runtime releases the
/// binary16 caches; or the arm continuing its own decode (`!has_f16`, `pos0`
/// where its last launch ended). Anything else is refused: a continuation
/// beside binary16 buffers would leave two copies that disagree, and a
/// hand-over with a gap would quantize positions nobody wrote.
pub fn p4e_k8_step_release(st: K8State, has_f16: bool, len16: usize, pos0: usize, n: usize, max_cap: usize) -> Result<K8Step, String> {
    let need = (pos0 + n).div_ceil(256) * 256;
    if need > max_cap {
        return Err(format!("4E v9k8: {} positions, the attention scratch holds {max_cap}", pos0 + n));
    }
    match has_f16 {
        true if len16 == pos0 && pos0 > 0 => {
            let cap = st.cap.max(need);
            Ok(K8Step { grow: cap != st.cap, copy_old: false, fill: Some((0, pos0)), next: K8State { cap, hand: pos0, own: 0 } })
        }
        true => Err(format!(
            "4E v9k8 (binary16 released at hand-over): a launch at position {pos0} beside binary16 caches holding {len16}: \
             a hand-over starts where a prefill ended, past position 0"
        )),
        false if st.own > 0 && pos0 == st.own => {
            let cap = st.cap.max(need);
            Ok(K8Step { grow: cap != st.cap, copy_old: cap != st.cap, fill: None, next: K8State { cap, hand: st.hand, own: st.own } })
        }
        false => Err(format!(
            "4E v9k8 (binary16 released at hand-over): a launch at position {pos0} with no binary16 cache, the arm's own \
             decode ending at {}: only a continuation from there",
            st.own
        )),
    }
}

/// The edits of arm v9k8: the int8 cache's text before `struct Pers4eOT`,
/// then v9's append and attention calls on the int8 caches, whose addresses
/// the layer's `kc` and `vc` carry (no struct moves). Nothing else.
pub fn p4e_k8_edits() -> Vec<(&'static str, String, String)> {
    let e4 = "pers_4e.cu";
    vec![
        (e4, "struct Pers4eOT {\n".into(), format!("{}\nstruct Pers4eOT {{\n", P4E_K8_PART.1)),
        (
            e4,
            "            pers_4e_norm_rope_append<FORM>(Y.qkv.q, Y.qkv.k, Y.qkv.v, Y.q_norm, Y.k_norm, A->cosr + pos * 64u,
                                           A->sinr + pos * 64u, A->eps, A->n_heads, A->n_kv, A->cap, pos, A->q_out,
                                           Y.kc, Y.vc);
"
            .into(),
            "            // k8: the append quantizes into the int8 caches (pers_4e_k8.cuh), whose addresses kc and vc carry
            pers_4e_norm_rope_append_k8<FORM>(Y.qkv.q, Y.qkv.k, Y.qkv.v, Y.q_norm, Y.k_norm, A->cosr + pos * 64u,
                                              A->sinr + pos * 64u, A->eps, A->n_heads, A->n_kv, A->cap, pos,
                                              A->q_out, reinterpret_cast<unsigned char*>(Y.kc),
                                              reinterpret_cast<unsigned char*>(Y.vc));
"
            .into(),
        ),
        (
            e4,
            "            pers_attention<PF ? 2u : 1u>(A->q_out, Y.kc, Y.vc, A->n_heads,".into(),
            "            pers_attention_k8<PF ? 2u : 1u>(A->q_out, reinterpret_cast<const unsigned char*>(Y.kc),
                                            reinterpret_cast<const unsigned char*>(Y.vc), A->n_heads,"
                .into(),
        ),
    ]
}

/// The research phase's row bodies as 4E row forms (step R6 of
/// docs/CERVEAU.md O11): inserted into llvq_pers.cuh by [`p4e_rc_edits`].
const PERS_RC_PART: (&str, &str) = ("pers_4e_rc.cuh", include_str!("../../kernels/pers_4e_rc.cuh"));

/// The lighter variants (bodies 3 to 6), inserted after [`PERS_RC_PART`].
const PERS_RC2_PART: (&str, &str) = ("pers_4e_rc2.cuh", include_str!("../../kernels/pers_4e_rc2.cuh"));

/// The f32 slots of a body (`PERS_RC_SLOTS` of pers_4e_rc.cuh).
const PERS_RC_SLOTS: u32 = 2048;

/// `pers_rc_padded_floats`: flat's padded staging of a `d_in` vector (block b
/// at 24 b + b / 8, the tail after), rounded to 16 bytes.
pub(super) fn rc_padded_floats(d_in: u32) -> u32 {
    let (nb, tw) = (d_in / 24, d_in % 24);
    (24 * nb + nb / 8 + tw + 3) & !3
}

/// The edits that put a 4E v9 module's Tetra row phases on a research body:
/// `body` 1 endpass, 2 flat. The body's text and its define before `struct
/// Pers4eQkv`; then the four call sites (q/k, gate/up, the Tetra down, the
/// 8B's Tetra o), each from v9's warp lists (pers_4e_rows) to the CTA's whole
/// slice (pers_rc_rows). Every phase, its free region (pers_4e_rc.cuh, header):
///   q/k     after xr and xn, v + 2 hidden; flat's padded copy there, the slots after;
///   gate/up the same, then the gate and up values (binary16) of the CTA's pairs;
///   down, o in place: flat restages the activation where it lies, the slots after it.
/// The gate/up list is the CTA's gate rows THEN its up rows (rowbench's gate/up
/// form), where v9 interleaves them a warp at a time (gate r, up r) and folds
/// each product under the next pair; here the products follow a CTA barrier.
///
/// Bodies 3 to 6 (pers_4e_rc2.cuh): the same call sites, naming
/// `pers_rc2_rows` and `pers_rc2_work_floats`, and both texts inserted. The
/// texts of bodies 1 and 2 are unchanged by them (their modules keep their
/// sha256, `the_rc_modules_keep_their_bytes`).
pub fn p4e_rc_edits(body: u32) -> Result<Vec<(&'static str, String, String)>, String> {
    if !(1..=6).contains(&body) {
        return Err(format!("4E rc: body {body}, the bodies are 1 (endpass), 2 (flat), 3 to 6 (pers_4e_rc2.cuh)"));
    }
    let v = p4e_rc_edits_12(body)?;
    if body <= 2 {
        return Ok(v);
    }
    Ok(v.into_iter()
        .map(|(f, from, to)| {
            let to = to
                .replace(PERS_RC_PART.1, &format!("{}\n{}", PERS_RC_PART.1, PERS_RC2_PART.1))
                .replace("pers_rc_rows<PERS_RC_ARM>(", "pers_rc2_rows<PERS_RC_ARM>(")
                .replace("pers_rc_work_floats<PERS_RC_ARM>(", "pers_rc2_work_floats<PERS_RC_ARM>(");
            (f, from, to)
        })
        .collect())
}

/// The edits of bodies 1 and 2, and the template of the others ([`p4e_rc_edits`]).
fn p4e_rc_edits_12(body: u32) -> Result<Vec<(&'static str, String, String)>, String> {
    let (cuh, seg, e4) = ("llvq_pers.cuh", "pers_segment.cu", "pers_4e.cu");
    Ok(vec![
        (cuh, "struct Pers4eQkv {\n".into(), format!("#define PERS_RC_ARM {body}u\n{}\nstruct Pers4eQkv {{\n", PERS_RC_PART.1)),
        (
            cuh,
            "        (void)scratch;
        const auto row_of = [&](u32 k) { return qk_row(lo + warp + k * nw); };
        const auto start = [](u32) {};
        const auto epi = [&](u32 k, float y) { qk_out(lo + warp + k * nw, y); };
        pers_4e_rows(pers_warp_rows(lo, thi, warp, nw), P.nblocks, P.tail_w, tab, invnorm, xr, lane, row_of, start,
                     epi);
"
            .into(),
            "        (void)scratch;  // rc: the CTA's q | k rows as one list; the free region after xr and xn
        const auto row_at = [&](u32 idx) { return qk_row(lo + idx); };
        const auto epi = [&](u32 idx, float y) { qk_out(lo + idx, y); };
        pers_rc_rows<PERS_RC_ARM>(thi - lo, P.nblocks, P.tail_w, tab, invnorm, const_cast<float*>(xr), row_at, epi,
                                  const_cast<float*>(xr) + 2u * P.d_in, false);
"
            .into(),
        ),
        (
            seg,
            "        const u32 warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
        const u32 glo = PERS_SEG_GU(0u), ghi = PERS_SEG_GU(1u);
        const u32 npairs = pers_warp_rows(glo, ghi, warp, nw);
        unsigned short g = 0u, pend_sv = 0u, pend_u = 0u;
        u32 pend_r = 0xffffffffu;
        const auto row_of = [&](u32 k) {
            const u32 r = glo + warp + (k >> 1) * nw;
            const SegTetra& M = (k & 1u) ? A->up : A->gate;
            return PersTRow{M.words + r * M.stride_u32, M.gscale, M.rscale, M.tail, r};
        };
        const auto start = [](u32) {};
        const auto epi = [&](u32 k, float y) {
            if ((k & 1u) == 0u) {
                g = f2h(y);
                return;
            }
            if (pend_r != 0xffffffffu) A->act[pend_r] = exact_hmul(pend_sv, pend_u);
            pend_sv = A->silu[g];
            pend_u = f2h(y);
            pend_r = glo + warp + (k >> 1) * nw;
        };
        pers_4e_rows(2u * npairs, A->gate.nblocks, A->gate.tail_w, tab, t_inv, v, lane, row_of, start, epi);
        if (lane == 0u && pend_r != 0xffffffffu) A->act[pend_r] = exact_hmul(pend_sv, pend_u);
"
            .into(),
            "        // rc: the CTA's gate rows, then its up rows, as one list; their binary16 values after the body's
        // region, the products after the body's last barrier (exact_hmul(silu[g], u), the stream's own product)
        const u32 glo = PERS_SEG_GU(0u), ghi = PERS_SEG_GU(1u), np = ghi - glo;
        const u32 nb = A->gate.nblocks, tw = A->gate.tail_w;
        float* const work = v + 2u * H;
        float* const gu_vals = work + pers_rc_work_floats<PERS_RC_ARM>(nb, tw, false);
        pers_rc_guard(gu_vals + np);
        unsigned short* const sgu = reinterpret_cast<unsigned short*>(gu_vals);
        const auto row_at = [&](u32 idx) {
            const bool up = idx >= np;
            const u32 r = glo + (up ? idx - np : idx);
            const SegTetra& M = up ? A->up : A->gate;
            return PersTRow{M.words + r * M.stride_u32, M.gscale, M.rscale, M.tail, r};
        };
        const auto epi = [&](u32 idx, float y) { sgu[idx] = f2h(y); };
        pers_rc_rows<PERS_RC_ARM>(2u * np, nb, tw, tab, t_inv, v, row_at, epi, work, false);
        for (u32 p = threadIdx.x; p < np; p += blockDim.x) A->act[glo + p] = exact_hmul(A->silu[sgu[p]], sgu[np + p]);
"
            .into(),
        ),
        (
            seg,
            "            unsigned short xr = 0u;
            const auto row_of = [&](u32 k) {
                const u32 r = dlo + warp + k * nw;
                return PersTRow{A->down.words + r * A->down.stride_u32, A->down.gscale, A->down.rscale, A->down.tail, r};
            };
            const auto start = [&](u32 k) { xr = __ldcg(L.x + dlo + warp + k * nw); };
            const auto epi = [&](u32 k, float y) { L.out[dlo + warp + k * nw] = exact_hadd(xr, f2h(y)); };
            pers_4e_rows(pers_warp_rows(dlo, dhi, warp, nw), A->down.nblocks, A->down.tail_w, tab, t_inv, v, lane,
                         row_of, start, epi);
"
            .into(),
            "            // rc: the CTA's down rows as one list; flat restages the activation in place, the slots after it
            const u32 nb = A->down.nblocks, tw = A->down.tail_w;
            const auto row_at = [&](u32 idx) {
                const u32 r = dlo + idx;
                return PersTRow{A->down.words + r * A->down.stride_u32, A->down.gscale, A->down.rscale, A->down.tail, r};
            };
            const auto epi = [&](u32 idx, float y) { L.out[dlo + idx] = exact_hadd(__ldcg(L.x + dlo + idx), f2h(y)); };
            pers_rc_rows<PERS_RC_ARM>(dhi - dlo, nb, tw, tab, t_inv, v, row_at, epi, v + pers_rc_padded_floats(nb, tw), true);
"
            .into(),
        ),
        (
            e4,
            "    const u32 lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    unsigned short xr = 0u;
    const auto row_of = [&](u32 k) {
        const u32 r = lo + warp + k * nw;
        return PersTRow{O.t.words + r * O.t.stride_u32, O.t.gscale, O.t.rscale, O.t.tail, r};
    };
    const auto start = [&](u32 k) { xr = __ldcg(x + lo + warp + k * nw); };
    const auto epi = [&](u32 k, float y) { out[lo + warp + k * nw] = exact_hadd(xr, f2h(y)); };
    pers_4e_rows(pers_warp_rows(lo, hi, warp, nw), O.t.nblocks, O.t.tail_w, tab, invnorm, cs, lane, row_of, start, epi);
"
            .into(),
            "    // rc: the CTA's o rows as one list; flat restages the context in place, the slots after it
    const u32 nb = O.t.nblocks, tw = O.t.tail_w;
    const auto row_at = [&](u32 idx) {
        const u32 r = lo + idx;
        return PersTRow{O.t.words + r * O.t.stride_u32, O.t.gscale, O.t.rscale, O.t.tail, r};
    };
    const auto epi = [&](u32 idx, float y) { out[lo + idx] = exact_hadd(__ldcg(x + lo + idx), f2h(y)); };
    pers_rc_rows<PERS_RC_ARM>(hi - lo, nb, tw, tab, invnorm, cs, row_at, epi, cs + pers_rc_padded_floats(nb, tw), true);
"
            .into(),
        ),
    ])
}

/// One phase of a research body in 4E's buffer: its name, the last float it
/// uses (from 4E's row vector `v`), and the v7 area's offset it must not pass.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RcPhase {
    pub phase: &'static str,
    pub end: u32,
    pub limit: u32,
}

/// Where `body` (1 endpass, 2 flat) puts its region at every Tetra phase of a
/// 4E v9 shape, as pers_4e_rc.cuh and the edits place it: `hidden`, `inter`,
/// `d_ctx` (the attention output's width; `tetra_o`: an o_proj in Tetra, the
/// 8B's), `groups` and the grid (the CTA's gate/up pairs at most). Refuses, by
/// phase and numbers, a shape where a phase would pass the v7 area: no byte is
/// added to 4E's launch, so a shape that does not fit is not run.
#[allow(clippy::too_many_arguments)]
pub fn p4e_rc_layout(body: u32, hidden: u32, inter: u32, d_ctx: u32, tetra_o: bool, tetra_down: bool, groups: u32, grid: u32) -> Result<Vec<RcPhase>, String> {
    if !(1..=6).contains(&body) || grid == 0 || groups == 0 {
        return Err(format!("4E rc: body {body}, grid {grid}, {groups} groups"));
    }
    let flat = rc_body_is_flat(body);
    let attn = 16 * 1024 * p4e_attn_ring(7) + 4 * 4 * groups * 130 + 4 * groups * 128;
    let limit = inter.max(2 * hidden).max(attn / 4);
    let work = |d_in: u32, in_place: bool| if flat && !in_place { rc_padded_floats(d_in) } else { 0 } + PERS_RC_SLOTS;
    let pairs = inter.div_ceil(grid);
    // gate/up: the one-call site's binary16 values after the body's region. R6d's arms (P4E_RC_GU_ARMS) keep no
    // value there (v9's site, or two calls ending at the body's region), so this end is conservative for them.
    let mut v = vec![
        RcPhase { phase: "q/k", end: 2 * hidden + work(hidden, false), limit },
        RcPhase { phase: "gate/up", end: 2 * hidden + work(hidden, false) + pairs, limit },
    ];
    if tetra_down {
        v.push(RcPhase { phase: "down", end: rc_padded_floats(inter) + work(inter, true), limit });
    }
    if tetra_o {
        v.push(RcPhase { phase: "o", end: rc_padded_floats(d_ctx) + work(d_ctx, true), limit });
    }
    let name = ["", "endpass", "flat", "ep2", "ep3", "fl2", "fl3"][body as usize];
    if let Some(p) = v.iter().find(|p| p.end > p.limit) {
        return Err(format!(
            "4E rc {name}: the {} phase needs floats up to {} of 4E's row vector, past the v7 area at {} (hidden {hidden}, \
             inter {inter}, d_ctx {d_ctx}, {groups} groups): refused, 4E's launch gets no byte more",
            p.phase, p.end, p.limit
        ));
    }
    if flat && inter.max(hidden).max(d_ctx).div_ceil(24).div_ceil(8) > PERS_RC_SLOTS {
        return Err(format!("4E rc {name}: a row of more than {PERS_RC_SLOTS} runs"));
    }
    // One chunk (ep3, fl2, fl3): the CTA's largest slice of every phase, its rows (endpass) or rows x runs (flat),
    // within the slots; the body traps otherwise.
    if rc_body_one_chunk(body) {
        let runs = |d_in: u32| if flat { (d_in / 24).div_ceil(8) } else { 1 };
        let mut need = vec![("q/k", (d_ctx + d_ctx / groups).div_ceil(grid) * runs(hidden)), ("gate/up", 2 * pairs * runs(hidden))];
        if tetra_down {
            need.push(("down", hidden.div_ceil(grid) * runs(inter)));
        }
        if tetra_o {
            need.push(("o", hidden.div_ceil(grid) * runs(d_ctx)));
        }
        if let Some((ph, n)) = need.iter().find(|p| p.1 > PERS_RC_SLOTS) {
            return Err(format!(
                "4E rc {name}: one chunk, and the {ph} phase's slice needs {n} slots of {PERS_RC_SLOTS} on {grid} CTAs: refused"
            ));
        }
    }
    Ok(v)
}

/// The mv arm's helpers (`pers_4e_e1.cuh`), inserted at the head of
/// `pers_segment.cu`'s text in that arm's module only.
const P4E_E1_PRE: (&str, &str) = ("pers_4e_e1.cuh", include_str!("../../kernels/pers_4e_e1.cuh"));

/// The logits symbols every E1 module carries: `pers_4e_body` with a
/// fourth template parameter, false by default (the token and stamp symbols
/// keep their code), true in these symbols, which write the head's binary16
/// logits of token t at `stamps + t x vocab` (the launch's stamp pointer,
/// read as a buffer of `n_tokens x vocab` binary16).
const P4E_E1_LOGITS_SYMBOLS: &str = "// e1-logits: the head's binary16 logits of every token of the launch, written at
// L.stamps + t x vocab (a measurement symbol, outside the timed rounds)
#define PERS_4E_LOGITS(sym, FORM_)                                                                       \\
    extern \"C\" __global__ void __launch_bounds__(512, 1) sym(const Pers4eArgs* __restrict__ A,          \\
                                                             const Pers4eLaunch L)                        \\
    {                                                                                                    \\
        pers_4e_body<FORM_, false, true, true>(A, L);                                                    \\
    }

PERS_4E_LOGITS(pers_4e_logits_f0_t512, 0)
PERS_4E_LOGITS(pers_4e_logits_f1_t512, 1)
PERS_4E_LOGITS(pers_4e_logits_f2_t512, 2)
";

/// The edits of an E1 arm: (part of [`PERS_UNIT_PARTS`], text, replacement),
/// each text present exactly once in its part. Every arm carries the logits
/// edits; np and mv add theirs. `tests/pers_matches_served/` builds the
/// host twin of each arm from the same edits.
pub fn p4e_e1_edits(arm: &str) -> Result<Vec<(&'static str, String, String)>, String> {
    let e4 = "pers_4e.cu";
    let seg = "pers_segment.cu";
    let mut v: Vec<(&'static str, String, String)> = vec![
        (
            e4,
            "template <int FORM, bool STAMP, bool PF>\n__device__ __forceinline__ void pers_4e_body(".into(),
            "template <int FORM, bool STAMP, bool PF, bool LOGITS = false>\n__device__ __forceinline__ void pers_4e_body(".into(),
        ),
        (
            e4,
            "PERS_BAR(), A->abort_flag, nullptr);".into(),
            "PERS_BAR(), A->abort_flag,\n                                          \
             LOGITS ? reinterpret_cast<unsigned short*>(L.stamps) + (size_t)t * A->vocab : nullptr);"
                .into(),
        ),
        (
            e4,
            "PERS_4E(pers_4e_stamps_f2_t768, 2, 768, true, false)\n".into(),
            format!("PERS_4E(pers_4e_stamps_f2_t768, 2, 768, true, false)\n\n{P4E_E1_LOGITS_SYMBOLS}"),
        ),
    ];
    let np = [
        (
            e4,
            "            if (c0 >= attn_jobs) prefetch_next();\n".to_string(),
            "            (void)attn_jobs;  // e1-pf: no o or gate/up row is asked of L2 in this phase\n            \
             (void)prefetch_next;  // e1-pf\n"
                .to_string(),
        ),
        (e4, "            if (c0 < attn_jobs) prefetch_next();\n".to_string(), String::new()),
    ];
    match arm {
        "v9" => {}
        // the attention tile's full-tile path (pers_ft_edit), nothing else
        "v9ft" => v.push(pers_ft_edit()),
        // the research bodies on every Tetra row phase (pers_4e_rc.cuh), nothing else
        "v9ep" | "v9fl" | "v9ep2" | "v9ep3" | "v9fl2" | "v9fl3" => {
            v.extend(p4e_rc_edits(p4e_rc_body(arm).expect("a research arm has a body"))?)
        }
        // R6d: the body on q/k, down and o; gate/up as v9 (v9fn) or in two calls (v9fg, v9eg)
        "v9fn" | "v9fg" | "v9eg" => v.extend(p4e_rc_gu_edits(arm)?),
        "v9np" => v.extend(np),
        // the KV cache as int8 (pers_4e_k8.cuh), nothing else
        "v9k8" => v.extend(p4e_k8_edits()),
        "v9mv" => {
            v.extend(np);
            v.extend([
                (seg, "struct SegTetra {\n".to_string(), format!("{}\nstruct SegTetra {{\n", P4E_E1_PRE.1)),
                (
                    seg,
                    "    PERS_SEG_SUB(0u);\n".to_string(),
                    "    PERS_SEG_SUB(0u);\n    \
                     if (IN4E) pers_e1_prefetch_staged();  // e1-pf: layer l + 1's q/k/v and o rows, during these gate/up rows\n"
                        .to_string(),
                ),
                (
                    e4,
                    "        pers_v7_stage_rot(area, g->signs_a, Ag->hidden, g->small_a, g->k_a);\n    };\n".to_string(),
                    "        pers_v7_stage_rot(area, g->signs_a, Ag->hidden, g->small_a, g->k_a);\n        \
                     pers_e1_stage_next(Ag->layers, l + 1u, Ag->n_layers, Ag->hidden, Ag->n_heads);  // e1-pf: layer l + 1's shares\n    };\n"
                        .to_string(),
                ),
                (
                    e4,
                    "            if (PF) {\n                const Pers4eQkv& N = Y.qkv;\n".to_string(),
                    "            if (l == 0u) {  // e1-pf: layer 0's q/k/v here, every later layer's during the gate/up rows before it\n            \
                     if (PF) {\n                const Pers4eQkv& N = Y.qkv;\n"
                        .to_string(),
                ),
                (
                    e4,
                    "            }\n            // ln1, the q/k rotation, the q/k/v rows\n".to_string(),
                    "            }\n            pers_e1_prefetch_o(Y, H, A->n_heads);  // e1-pf: and layer 0's o rows\n            \
                     }  // e1-pf\n            // ln1, the q/k rotation, the q/k/v rows\n"
                        .to_string(),
                ),
            ]);
        }
        _ => match P4E_RC_PROBES.iter().find(|p| p.0 == arm) {
            // a load probe: the endpass text, and its one site
            Some(&(_, site)) => {
                v.extend(p4e_rc_edits(1)?.into_iter().enumerate().filter(|(i, _)| *i == 0 || *i == site).map(|(_, e)| e))
            }
            None => return Err(format!("4E E1: no arm {arm:?}; the arms are {P4E_E1_ARMS:?}")),
        },
    }
    Ok(v)
}

/// The module text of an E1 arm at a tile: row form 7's define, then
/// [`PERS_UNIT_PARTS`] with the arm's edits ([`p4e_e1_edits`]), joined as
/// [`pers_unit_text`](crate::pers::pers_unit_text) joins them. Refuses an edit whose text is not in its
/// part exactly once.
pub fn p4e_e1_unit_text(tile_define: &str, arm: &str) -> Result<String, String> {
    let mut parts: Vec<(&str, String)> = PERS_UNIT_PARTS.iter().map(|p| (p.0, p.1.to_string())).collect();
    for (file, from, to) in p4e_e1_edits(arm)? {
        let part = parts.iter_mut().find(|p| p.0 == file).ok_or_else(|| format!("4E E1 {arm}: no part {file}"))?;
        let n = part.1.matches(from.as_str()).count();
        if n != 1 {
            return Err(format!("4E E1 {arm}: {n} sites in {file} for {from:?}, expected exactly one"));
        }
        part.1 = part.1.replacen(from.as_str(), &to, 1);
    }
    let body = std::iter::once(tile_define.to_string()).chain(parts.into_iter().map(|p| p.1)).collect::<Vec<_>>().join("\n");
    Ok(format!("{}{body}", p4e_rows_define(7)?))
}

/// The logits symbol of an E1 module for a RoPE form (512 threads).
pub fn p4e_logits_symbol(form: u32) -> Result<&'static str, String> {
    Ok(match form {
        0 => "pers_4e_logits_f0_t512",
        1 => "pers_4e_logits_f1_t512",
        2 => "pers_4e_logits_f2_t512",
        _ => return Err(format!("4E E1: no logits symbol for RoPE form {form}")),
    })
}

/// A line E1 may add, drop or change: a prefetch statement or the code that
/// steers one. It names a prefetch, or carries the `e1-pf` tag; and it
/// assigns nothing (no ` = ` outside a declaration of a prefetch helper).
fn p4e_e1_prefetch_line(l: &str) -> bool {
    let t = l.trim();
    (t.contains("prefetch") || t.contains("e1-pf"))
        && (!t.split("//").next().unwrap_or("").contains(" = ") || t.starts_with("const auto prefetch_next = "))
}

/// The structural check of an E1 arm against its reference (`"v9"`'s text):
/// with every prefetch line ([`p4e_e1_prefetch_line`]) and the mv helpers'
/// block (`// e1-pf begin` .. `// e1-pf end`) taken out of both, the two
/// texts are the same lines in the same order. The block itself may hold no
/// kernel, no atomic, no barrier, no grid wait, and may store only to its
/// header slots (`s[i] = ...`).
pub fn p4e_e1_structure(reference: &str, arm: &str) -> Result<(), String> {
    fn strip(t: &str) -> Result<Vec<&str>, String> {
        let mut out = Vec::new();
        let mut inside = false;
        for l in t.lines() {
            let s = l.trim();
            if s == "// e1-pf begin" {
                inside = true;
                continue;
            }
            if s == "// e1-pf end" {
                inside = false;
                continue;
            }
            if inside {
                for bad in ["__global__", "atomic", "__syncthreads", "pers_grid_wait", "PERS_BAR", "A->", "L."] {
                    if s.contains(bad) {
                        return Err(format!("E1 helpers: {bad:?} in {s:?}"));
                    }
                }
                // every assignment's target: a local being declared, or a slot
                let code = s.split("//").next().unwrap_or("");
                for (i, _) in code.match_indices(" = ") {
                    let before = &code[..i];
                    let cut = before.rfind([';', ',', '{']).map_or(0, |k| k + 1);
                    let mut lhs = before[cut..].trim();
                    lhs = lhs.strip_prefix("for (").unwrap_or(lhs);
                    if let Some(k) = lhs.rfind(") ") {
                        lhs = lhs[k + 2..].trim();
                    }
                    if (lhs.contains('[') && !lhs.starts_with("s[")) || lhs.starts_with('*') || lhs.contains("->") || lhs.contains('.') {
                        return Err(format!("E1 helpers: a store that is not a slot: {s:?}"));
                    }
                }
                continue;
            }
            // comments and blank lines are not code
            if s.is_empty() || s.starts_with("//") {
                continue;
            }
            if !p4e_e1_prefetch_line(l) {
                out.push(l);
            }
        }
        if inside {
            return Err("E1 helpers: a block opened and never closed".into());
        }
        Ok(out)
    }
    let (a, b) = (strip(reference)?, strip(arm)?);
    if let Some(i) = (0..a.len().min(b.len())).find(|&i| a[i] != b[i]) {
        return Err(format!("E1: past the prefetch lines, line {i} differs: {:?} against {:?}", a[i], b[i]));
    }
    if a.len() != b.len() {
        return Err(format!("E1: past the prefetch lines, {} lines against {}", a.len(), b.len()));
    }
    Ok(())
}
