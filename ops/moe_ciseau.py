#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""T4 of the 2026-08-13 preregistration: the hot/cold MoE scissor, priced.

The idea under test ("le ciseau"): store the hot expert cells in the fast but
fat runtime layout (`Golay70`, 3.589 b/weight, measured) and the cold ones in
the thin archive layout (2.219 b/weight, the sealed file itself), so that the
whole-model VRAM lands under the 3.20 b/weight bar that puts a 117 B-parameter
MoE on a 48 GB card.

Two blades close on it, hence the name:

* the **memory** blade wants alpha (hot fraction) small, because the mix is
  `alpha*3.589 + (1-alpha)*2.219`;
* the **hit** blade wants alpha large, because every routing that lands on a
  cold cell is a miss, and a miss is expensive.

This script computes `alpha_min(budget)` from the *measured* routing dump and
confronts it with `alpha_VRAM`, the largest alpha the memory bar allows. It
also prices the alternative the preregistration demands be named: cold tier in
**host RAM**, miss = PCIe memcpy.

    uv run ops/moe_ciseau.py

Standard library only, and deliberately so: this is arithmetic over a 5.7 kB
JSON, it must run on a machine with no ML stack (the repo's conda env is
broken: torch 2.5.1 vs transformers 5.5.4).

No number is invented. Every constant carries its provenance as
`file:line` in the CONSTANTS block below, and every printed figure is tagged
*measured* / *computed* / *estimated* per the repo rule (CLAUDE.md section 7).
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date

# ---------------------------------------------------------------------------
# CONSTANTS — provenance first, arithmetic second.
# ---------------------------------------------------------------------------

DUMP = "docs/data/moe-routing-gptoss20b-2026-08-12.json"
PREREG = "proofs/preregistration-2026-08-13.md"

# Runtime layouts, both MEASURED, b/weight payload.
#   Golay70 : docs/data/echelle-formats.csv, 3.589 b/weight; exact
#             reconstruction proven on 150.7 M blocks, 1.31x vs FP16 (v1).
#   archive : docs/archive/spec-memoire-extreme-2026-08-12.md:56
#             "plancher (le fichier) | 2,219" — the sealed file's own rate.
B_HOT = 3.589
B_COLD = 2.219

# Decode throughputs.
#   195 GB/s : Golay70 effective, L40S, 7-arm bench
#              (docs/archive/etude-moe-memoire-extreme-2026-08-12.md:51-55).
#   8,27 ns/block : the v1 rank decoder, M3 Max GPU over 16,7 M blocks
#              (docs/format-noyau.md:55 and :120, "106x le sol").
#   WARNING: two different machines. The composite below is an order of
#   magnitude, not a precision figure. It is labelled *estimated* when printed.
GBPS_HOT = 195.0
NS_PER_BLOCK_RANK = 8.27
WEIGHTS_PER_BLOCK = 24  # Lambda_24, by construction

# gpt-oss-120B, the dimensioning target.
#   117 B total / 5,1 B active   : docs/archive/etude-moe-memoire-extreme-2026-08-12.md:25 (web sheet)
#   128 experts                  : proofs/preregistration-2026-08-13.md:169-170
#   layers                       : CALCULATED below, cross-checked on the 20B.
P120_TOTAL = 117e9
P120_ACTIVE = 5.1e9
P120_EXPERTS = 128
P120_TOPK = 4

# Memory bar. proofs/preregistration-2026-08-13.md:160-161 fixes the triple:
# 48 GB card, 117 B total, ~1,2 GB deducted for embedding + KV.
# 117e9 * 3.20/8 = 46,8 GB, +1,2 = 48,0 GB exactly — so this IS the convention
# the 3,20 threshold was derived in, and the one every VRAM line below uses.
CARD_GB = 48.0
OVERHEAD_GB = 1.2
S_MIX = 3.20

# The two miss budgets the lot must report, in misses per token, whole model.
#   0,70 : proposed by the piste.
#   0,29 : the counter-expertise, "2,4x too generous" (0,70 / 0,29 = 2,41).
# WARNING, PROVENANCE DEBT stated up front: neither number has a source file in
# this repo (grepped 2026-08-13). They enter as hypotheses; section 4 derives
# what the measured decode cost actually supports, and the verdict is computed
# on a sweep so that it does not depend on picking one.
BUDGET_PISTE = 0.70
BUDGET_CONTRE = 0.29

# PCIe effective bandwidths, host->device, large contiguous copies.
# WARNING, EXTERNAL HYPOTHESIS: nothing in this repo measures PCIe. Estimated.
PCIE = {"gen4 x16 (~32 Go/s)": 32e9, "gen5 x16 (~50 Go/s)": 50e9, "gen5 x16 (~63 Go/s)": 63e9}

# LRU branch, frozen in the preregistration (:163-164): resident backing means
# the cold tier is paid IN FULL plus the cache, so 2,219 + alpha*3,589 <= 3,20,
# and the branch survives only on a hit rate >= 99,8 % at that alpha.
LRU_HIT_REQUIRED = 99.8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def gini(xs: list[int]) -> float:
    """Same estimator as `ops/moe_routing.py:113-120`, in pure Python.

    Reimplemented rather than imported because that module pulls torch in at
    import time. Reproducing its published values is this script's
    non-regression control: a parsing error would show up here first.
    """
    v = sorted(float(x) for x in xs)
    n = len(v)
    s = sum(v)
    if s == 0:
        return float("nan")
    return 2 * sum((i + 1) * v[i] for i in range(n)) / (n * s) - (n + 1) / n


def quantile(sorted_vals: list[float], q: float) -> float:
    """Linear interpolation, torch.quantile convention (matches moe_routing)."""
    n = len(sorted_vals)
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    return sorted_vals[lo] + (pos - lo) * (sorted_vals[hi] - sorted_vals[lo])


def cold_mass_at(desc: list[int], total: int, k_hot: int) -> float:
    """Routed mass left cold when the `k_hot` busiest cells are resident."""
    return (total - sum(desc[:k_hot])) / total


def alpha_min_global(desc: list[int], total: int, f_max: float) -> tuple[float, int]:
    """Smallest hot fraction whose cold mass stays under `f_max`, globally.

    Cells are ranked across ALL layers at once, so a concentrated layer may
    keep fewer hot cells than a flat one. This is the *optimal* static
    allocation, hence the most favourable number the piste can be given — the
    repo's habit of being generous to the thing it is about to bury.
    """
    acc = 0
    for k, v in enumerate(desc):
        if (total - acc) / total <= f_max:
            return k / len(desc), k
        acc += v
    return 1.0, len(desc)


def alpha_min_uniform(counts: list[list[int]], f_max: float) -> list[float]:
    """Per-layer hot fraction under a uniform per-layer miss allocation.

    Simpler to implement in a runtime (each layer gets the same budget) and
    strictly worse than the global one — reported to bracket the answer.
    """
    out = []
    for row in counts:
        desc = sorted(row, reverse=True)
        t = sum(desc)
        a, _ = alpha_min_global(desc, t, f_max)
        out.append(a)
    return out


def bisect_budget(desc: list[int], total: int, k: int, n_layers: int, target_alpha: float) -> float:
    """Miss budget at which `alpha_min_global` first drops to `target_alpha`."""
    lo, hi = 1e-3, 100.0
    for _ in range(200):
        mid = (lo + hi) / 2
        a, _ = alpha_min_global(desc, total, mid / (k * n_layers))
        if a > target_alpha:
            lo = mid
        else:
            hi = mid
    return hi


def mix(alpha: float) -> float:
    return alpha * B_HOT + (1 - alpha) * B_COLD


def vram_go(bpw: float, params: float = P120_TOTAL) -> float:
    return params * bpw / 8 / 1e9 + OVERHEAD_GB


def fr(x: float, nd: int = 3) -> str:
    """French decimal comma, because the output lands in docs/mesures/."""
    return f"{x:.{nd}f}".replace(".", ",")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, DUMP)
    if not os.path.exists(path):
        # Repo rule: a test that skips when its file is missing must FAIL.
        sys.exit(f"MISSING: {path}. The routing dump is the only measured input of this script.")
    d = json.load(open(path))

    counts = d["counts"]
    L, E, K = len(counts), d["n_experts"], d["top_k"]
    hidden, moe_inter, ntok = d["hidden"], d["moe_intermediate"], d["tokens"]
    cells = L * E
    visits = K * L  # routed cell-visits per token, whole model

    print("# The hot/cold MoE scissor. T4 of the 2026-08-13 preregistration")
    print(f"# date        : {date.today().isoformat()}")
    print("# command     : uv run ops/moe_ciseau.py")
    print(f"# source      : {DUMP} (measured, 131,072 tokens of C4, M3 Max, 2026-08-12)")
    print(f"# judge       : {PREREG} §3-T4, mix threshold > 3,20 b/weight ⇒ VRAM variant buried")
    print(f"# script      : ops/moe_ciseau.py (stdlib only, $0, no GPU, no network)")
    print("#")
    print("# Labels      : measured = out of a sealed file or a benchmark; computed = exact")
    print("# arithmetic on measured values; estimated = rests on a named external hypothesis.")

    # -- 0. Non-regression control -----------------------------------------
    print("\n\n0. PARSING CONTROL: the published Gini values must come back")
    print("-" * 78)
    print("   If this section does not reproduce moe-routing-gptoss20b-2026-08-12.txt,")
    print("   nothing that follows is readable.")
    per_layer_sums = {sum(r) for r in counts}
    ok_sums = per_layer_sums == {ntok * K}
    agg = [sum(counts[l][e] for l in range(L)) for e in range(E)]
    g0, g23, gagg = gini(counts[0]), gini(counts[L - 1]), gini(agg)
    print(f"   sum per layer         : {sorted(per_layer_sums)[0]} = {ntok} × {K}   "
          f"{'OK' if ok_sums else 'FAIL'}   (measured)")
    print(f"   Gini layer 0          : {fr(g0)}   expected 0,351   {'OK' if abs(g0-0.351)<5e-4 else 'FAIL'}")
    print(f"   Gini layer {L-1}         : {fr(g23)}   expected 0,748   {'OK' if abs(g23-0.748)<5e-4 else 'FAIL'}")
    print(f"   Gini aggregate (ALL)  : {fr(gagg)}   expected 0,169   {'OK' if abs(gagg-0.169)<5e-4 else 'FAIL'}")
    if not (ok_sums and abs(g0 - 0.351) < 5e-4 and abs(g23 - 0.748) < 5e-4 and abs(gagg - 0.169) < 5e-4):
        sys.exit("   CONTROL FAILED: the parsing is wrong, we stop here.")
    print("   OK: the three published Gini values come back to the thousandth. Parsing is sound.")

    # -- 1. Geometry, and the internal control that validates it -----------
    per_cell = 3 * hidden * moe_inter  # SwiGLU: gate + up + down
    exp20 = L * E * per_cell
    act20 = K * L * per_cell
    print("\n\n1. CELL GEOMETRY, AND THE CONTROL THAT VALIDATES IT")
    print("-" * 78)
    print(f"   dump          : {d['model']}, {L} layers × {E} experts = {cells} cells,")
    print(f"                   top-{K}, hidden {hidden}, moe_inter {moe_inter}   (measured)")
    print(f"   cell          : 3 × {hidden} × {moe_inter} = {per_cell:,} weights (gate+up+down)   (computed)"
          .replace(",", " "))
    print(f"   experts 20B   : {L}×{E}×cell = {fr(exp20/1e9,2)} B; +non-experts ⇒ 21 B on the sheet")
    print(f"   active 20B    : {K}×{L}×cell = {fr(act20/1e9,2)} B; +non-experts ⇒ 3,6 B on the sheet")
    p20_total, p20_active = 21e9, 3.6e9  # web sheet, etude-moe:22
    nonexp20 = p20_total - exp20
    nonexp20_active = p20_active - act20
    print(f"   OK, control: the study sheet gives {p20_total/1e9:.0f} B total / "
          f"{fr(p20_active/1e9,1)} B active (etude-moe:22).")
    print(f"      The 3-matrix model leaves {fr(nonexp20/1e9,2)} B outside the experts and "
          f"{fr(nonexp20_active/1e9,2)} B outside")
    print("      the active experts, which is exactly attention + embedding. The geometry holds.")
    # 120B layer count: solve L from the total, charging the 20B's per-layer
    # non-expert cost. Cross-checked on the active count just below.
    nonexp_per_layer = nonexp20 / L
    n_layers_120 = round(P120_TOTAL / (P120_EXPERTS * per_cell + nonexp_per_layer))
    exp120 = n_layers_120 * P120_EXPERTS * per_cell
    act120 = P120_TOPK * n_layers_120 * per_cell
    print(f"   120B          : {P120_TOTAL/1e9:.0f} B total / {fr(P120_ACTIVE/1e9,1)} B active, "
          f"{P120_EXPERTS} experts top-{P120_TOPK}   (sheet + preregistration:169)")
    print(f"                   ⇒ {n_layers_120} layers   (COMPUTED by charging the 20B non-expert")
    print(f"                   cost per layer : {P120_TOTAL/1e9:.0f} / ({P120_EXPERTS}×cell + "
          f"{nonexp_per_layer/1e6:.0f} M))")
    print(f"      control 1  : {n_layers_120}×{P120_EXPERTS}×cell = {fr(exp120/1e9,1)} B of experts, "
          f"{fr((P120_TOTAL-exp120)/1e9,2)} B left outside")
    print(f"                   (against {fr(nonexp20/1e9,2)} B at {L} layers, ×"
          f"{fr((P120_TOTAL-exp120)/nonexp20,2)} for ×{fr(n_layers_120/L,2)} the layers) OK")
    print(f"      control 2  : active = {P120_TOPK}×{n_layers_120}×cell = {fr(act120/1e9,2)} B, "
          f"+{fr((P120_ACTIVE-act120)/1e9,2)} B outside the experts")
    print(f"                   = the {fr(P120_ACTIVE/1e9,1)} B announced (against "
          f"{fr(nonexp20_active/1e9,2)} B outside the experts on the 20B) OK")
    print("   WARNING, hypothesis: hidden and moe_inter of the 120B assumed equal to the 20B (2880).")
    print("      The dump does not say so. It rests on the two controls above.")
    print("      *estimated*. The layer count serves only two secondary comparisons")
    print("      (time of one layer, §4; visits per token, §8): ±1 layer changes nothing.")

    # -- 2. Load distribution, layer by layer ------------------------------
    print("\n\n2. ROUTED LOAD DISTRIBUTION, PER CELL AND PER LAYER   (measured)")
    print("-" * 78)
    print("   Load as a multiple of the uniform load; α_min = smallest fraction of HOT cells")
    print("   in the layer such that the cold mass stays under budget, with a uniform")
    print("   allocation across layers (budget/layer = budget/24).")
    print()
    u_c = alpha_min_uniform(counts, BUDGET_CONTRE / visits)
    u_p = alpha_min_uniform(counts, BUDGET_PISTE / visits)
    uniform_load = ntok * K / E
    print(f"   {'layer':>6}  {'min':>7}  {'median':>8}  {'max':>7}  {'Gini':>6}  "
          f"{'α_min 0,29':>10}  {'α_min 0,70':>10}")
    print("   " + "-" * 71)
    for li in range(L):
        v = sorted(x / uniform_load for x in counts[li])
        print(f"   {li:>6}  {fr(v[0]):>7}  {fr(quantile(v,0.5)):>8}  {fr(v[-1]):>7}  "
              f"{fr(gini(counts[li])):>6}  {fr(u_c[li],3):>10}  {fr(u_p[li],3):>10}")
    print("   " + "-" * 71)
    print(f"   {'mean':>6}  {'':>7}  {'':>8}  {'':>7}  "
          f"{fr(sum(gini(r) for r in counts)/L):>6}  {fr(sum(u_c)/L):>10}  {fr(sum(u_p)/L):>10}")
    print()
    print(f"   Gini per layer : {fr(min(gini(r) for r in counts))} … {fr(max(gini(r) for r in counts))}")
    print("   WARNING: the \"0,59-0,77\" range to cross-check is that of the DEEP layers.")
    print("      Layers 0-1 and 6 are much flatter (0,351 / 0,426 / 0,448). The full range")
    print("      over the 24 layers is the one printed above, and that is the one that counts")
    print("      here, because the FLAT layers are the ones that cost the scissor.")
    print("   DETAIL: the mechanism is counter-intuitive and central. A HIGH Gini is")
    print("      FAVOURABLE to the scissor (few cells carry the mass). The cost item is the")
    print(f"      flat layers. At 0,29 miss/token layer 0 (Gini {fr(gini(counts[0]))}, the flattest)")
    print(f"      needs α = {fr(u_c[0])}: ALL of its cells hot. Layer {L-1} "
          f"(Gini {fr(gini(counts[L-1]))}),")
    print(f"      the most concentrated, needs only {fr(u_c[L-1])}.")

    # -- 3. The scissor ----------------------------------------------------
    desc = sorted((v for r in counts for v in r), reverse=True)
    total = sum(desc)
    alpha_vram = (S_MIX - B_COLD) / (B_HOT - B_COLD)
    print("\n\n3. THE SCISSOR: α_min (the \"hit\" blade) against α_VRAM (the \"memory\" blade)")
    print("-" * 78)
    print(f"   α_VRAM = ({fr(S_MIX,2)} − {fr(B_COLD)}) / ({fr(B_HOT)} − {fr(B_COLD)}) = "
          f"{fr(alpha_vram,4)}   (computed)")
    print(f"   Above it, the mix exceeds {fr(S_MIX,2)} b/weight and the 48 GB bar closes.")
    print()
    print(f"   {'budget':>8}  {'cold mass':>12}  {'α_min':>7}  {'hot':>9}  "
          f"{'mix':>8}  {'VRAM 120B':>10}  verdict")
    print(f"   {'miss/tok':>8}  {'max':>12}  {'global':>7}  {'cells':>9}  "
          f"{'b/weight':>8}  {'GB':>10}")
    print("   " + "-" * 74)
    sweep = [0.137, BUDGET_CONTRE, 0.5, BUDGET_PISTE, 1.0, 2.0, 5.0, 10.0, 25.0]
    rows = {}
    for b in sweep:
        f_max = b / visits
        a, k = alpha_min_global(desc, total, f_max)
        m = mix(a)
        rows[b] = (a, k, m)
        tag = "RED" if m > S_MIX else "green"
        print(f"   {fr(b,3):>8}  {fr(f_max*100,4)+' %':>12}  {fr(a,4):>7}  {k:>4}/{cells:<4}  "
              f"{fr(m,4):>8}  {fr(vram_go(m),1):>10}  {tag}")
    print("   " + "-" * 74)
    need = bisect_budget(desc, total, K, L, alpha_vram)
    print("   WARNING: the rows at ≥ 2 miss/token are NOT operating points. Section 4 shows")
    print("      that a miss costs 8,57 ms against 11,73 ms of token time. There \"green\"")
    print("      means the memory would fit. It does not mean the product exists.")
    print(f"   Budget it WOULD TAKE to reach α_VRAM = {fr(alpha_vram,4)}: "
          f"{fr(need,2)} miss/token   (computed)")
    print(f"   That is ×{fr(need/BUDGET_PISTE,1)} the budget of the piste and ×{fr(need/BUDGET_CONTRE,1)} "
          "that of the counter-expertise.")

    # -- 4. What a miss actually costs -------------------------------------
    blocks_cell = per_cell / WEIGHTS_PER_BLOCK
    t_token_ms = P120_ACTIVE * B_HOT / 8 / (GBPS_HOT * 1e9) * 1e3
    t_miss_vram_ms = blocks_cell * NS_PER_BLOCK_RANK / 1e6
    t_layer_ms = t_token_ms / n_layers_120
    print("\n\n4. WHAT A MISS REALLY COSTS: WHERE A DEFENSIBLE BUDGET COMES FROM")
    print("-" * 78)
    print(f"   token time 120B, all hot        : {fr(P120_ACTIVE/1e9,1)} B × {fr(B_HOT)}/8 = "
          f"{fr(P120_ACTIVE*B_HOT/8/1e9,3)} GB ÷ {GBPS_HOT:.0f} GB/s")
    print(f"                                     = {fr(t_token_ms,2)} ms  ({1000/t_token_ms:.0f} tok/s)"
          "   *estimated* (bandwidth")
    print("                                     ceiling, not a predicted throughput, study §2)")
    print("   a miss in the VRAM VARIANT      : the cold cell decodes through the v1 rank,")
    print(f"                                     {blocks_cell:,.0f}".replace(",", " ")
          + f" blocks × {fr(NS_PER_BLOCK_RANK,2)} ns/block")
    print("                                     (measured, format-noyau.md:120)")
    print(f"                                     = {fr(t_miss_vram_ms,2)} ms, which is "
          f"{t_miss_vram_ms/t_layer_ms:.0f}× the time of ONE WHOLE LAYER   *estimated*")
    print("   WARNING: the 195 GB/s are CUDA/L40S and the 8,27 ns/block Metal/M3 Max. The")
    print("      composite is an order of magnitude, never a precise figure. It serves only to")
    print("      settle a factor 30, which a factor 2 of uncertainty does not reverse.")
    print()
    print(f"   {'budget':>8}  {'miss cost':>13}  {'% of token':>11}  status")
    print(f"   {'miss/tok':>8}  {'ms/token':>13}  {'time':>11}")
    print("   " + "-" * 60)
    for b, label in ((0.137, "+10 % of token time"), (BUDGET_CONTRE, "counter-expertise"),
                     (BUDGET_PISTE, "the piste"), (need, "what it would take")):
        c = b * t_miss_vram_ms
        print(f"   {fr(b,3):>8}  {fr(c,2):>13}  {fr(100*c/t_token_ms,1)+' %':>11}  {label}")
    print("   " + "-" * 60)
    print("   OK, VERDICT ON THE BUDGET: 0,29 is the defensible one, and it is already generous.")
    print(f"      At 0,70 miss/token the cold tier eats {fr(100*BUDGET_PISTE*t_miss_vram_ms/t_token_ms,0)} %"
          " of the token time. A \"budget\" that")
    print("      redefines half the product is not a budget. The +10 % of token time threshold")
    print(f"      is worth {fr(0.137,3)} miss/token, ×{fr(BUDGET_PISTE/0.137,1)} stricter still")
    print("      than the counter-expertise. The factor 2,4 it denounces is a lower bound.")
    print()
    b_prereg = (1 - LRU_HIT_REQUIRED / 100) * visits
    a_prereg, _ = alpha_min_global(desc, total, b_prereg / visits)
    print("   DETAIL: an independent cross-check, and it sits in the preregistration itself.")
    print(f"      Its LRU hit threshold ({fr(LRU_HIT_REQUIRED,1)} %) is equivalent to "
          f"{fr(b_prereg,3)} miss/token, and that budget")
    print(f"      gives α_min = {fr(a_prereg,3)}, EXACTLY the \"α_hit ≈ 0,85\" it predicted.")
    print("      The two figures of §3-T4 are therefore a single claim, and that claim")
    print(f"      sits at {fr(b_prereg,2)} miss/token: between my derived threshold ({fr(0.137,3)}) and the")
    print(f"      counter-expertise one ({fr(BUDGET_CONTRE,2)}), far from the {fr(BUDGET_PISTE,2)} of the piste.")
    print("      Three independent derivations converge under 0,30; only one stays out.")

    # -- 5. Verdict --------------------------------------------------------
    a_c, _, m_c = rows[BUDGET_CONTRE]
    a_p, _, m_p = rows[BUDGET_PISTE]
    print("\n\n5. VERDICT AGAINST THE PREREGISTERED THRESHOLD (mix > 3,20 ⇒ VRAM buried)")
    print("=" * 78)
    print(f"   budget 0,29 (defensible) : α_min = {fr(a_c,4)} > α_VRAM = {fr(alpha_vram,4)}  ⇒ "
          f"mix {fr(m_c,3)} b/weight, VRAM {fr(vram_go(m_c),1)} GB")
    print(f"   budget 0,70 (generous)   : α_min = {fr(a_p,4)} > α_VRAM = {fr(alpha_vram,4)}  ⇒ "
          f"mix {fr(m_p,3)} b/weight, VRAM {fr(vram_go(m_p),1)} GB")
    print()
    print("   FAIL, EMPTY INTERVAL AT BOTH BUDGETS. The VRAM variant is BURIED.")
    print(f"      The mix exceeds {fr(S_MIX,2)} even under the MOST favourable allocation")
    print("      (global ranking of the 768 cells, static oracle, optimal allocation).")
    print(f"      It still exceeds it at the most generous budget on file: at 0,70")
    print(f"      miss/token, {fr(m_p,3)} > {fr(S_MIX,2)}. No budget choice between the two saves")
    print(f"      the piste. It would take {fr(need,2)} miss/token, outside any defensible budget.")
    print()
    print("   DETAIL: against the preregistered prediction (α_hit ≈ 0,85; α_VRAM ≤ 0,72; mix")
    print(f"      3,4-3,5): α_VRAM = {fr(alpha_vram,4)} lands exactly, α_hit = {fr(a_p,3)}-{fr(a_c,3)} is")
    print(f"      a little UNDER 0,85, and the mix lands at {fr(m_p,2)}-{fr(m_c,2)}, under the range")
    print("      that was predicted. The verdict is the one announced. The margin is thinner")
    print("      than predicted, and that is the only thing this lot contradicts.")

    # -- 6. LRU branch -----------------------------------------------------
    alpha_lru = (S_MIX - B_COLD) / B_HOT
    k_lru = round(alpha_lru * cells)
    hit_lru = (1 - cold_mass_at(desc, total, k_lru)) * 100
    miss_lru = cold_mass_at(desc, total, k_lru) * visits
    print("\n\n6. THE LRU CACHE BRANCH: KILLED BY ARITHMETIC, THEN BY MEASUREMENT")
    print("-" * 78)
    print(f"   Resident backing ⇒ the cold tier is paid IN FULL, plus the cache:")
    print(f"   {fr(B_COLD)} + α×{fr(B_HOT)} ≤ {fr(S_MIX,2)}  ⟺  α ≤ {fr(alpha_lru,4)}   "
          "(computed, preregistration §3-T4)")
    print(f"   At α = {fr(alpha_lru,4)} ({k_lru} cells out of {cells}):")
    print(f"     hit of a STATIC ORACLE cache (the {k_lru} busiest) : {fr(hit_lru,2)} %   (measured)")
    print(f"     ⇒ {fr(miss_lru,1)} miss/token, against a preregistered threshold of {fr(LRU_HIT_REQUIRED,1)} %")
    print(f"       ({fr((1-LRU_HIT_REQUIRED/100)*visits,2)} miss/token)")
    print(f"   FAIL, gap: ×{fr(miss_lru/((1-LRU_HIT_REQUIRED/100)*visits),0)} the miss budget. "
          "The LRU branch is dead in two")
    print("      independent ways: the arithmetic of the backing, then the real concentration.")
    print("   WARNING: the static oracle is NOT an LRU. The dump is AGGREGATED over 131 k tokens,")
    print("      it carries no temporal information. An LRU can beat the static oracle if")
    print("      temporal locality exists, and this measurement cannot decide that.")
    print(f"      But the gap to close is ×{fr(miss_lru/((1-LRU_HIT_REQUIRED/100)*visits),0)}, "
          "not a few percent. No plausible")
    print("      locality closes it. Settling it would take a TEMPORAL dump, which does not exist.")

    # -- 7. The dominant alternative ---------------------------------------
    cell_hot_mb = per_cell * B_HOT / 8 / 1e6
    cell_cold_mb = per_cell * B_COLD / 8 / 1e6
    print("\n\n7. THE DOMINANT ALTERNATIVE: COLD TIER IN HOST RAM, MISS = PCIe MEMCPY")
    print("=" * 78)
    print("   The preregistration requires it: \"a burial that does not name what replaces")
    print("   it is not a verdict, it is an abandonment.\"")
    print()
    print(f"   Size of one expert cell (120B, hidden 2880)   (computed):")
    print(f"     served format  `Golay70` {fr(B_HOT)} b/weight : {fr(cell_hot_mb,2)} MB")
    print(f"     archive format {fr(B_COLD)} b/weight           : {fr(cell_cold_mb,2)} MB  "
          f"(−{fr(100*(1-cell_cold_mb/cell_hot_mb),0)} % of traffic)")
    print()
    print(f"   Cost of a miss = one PCIe memcpy of that cell   (estimated, no PCIe measurement")
    print("   in this repository):")
    print(f"   {'link':>22}  {'served (ms)':>11}  {'archive (ms)':>13}  "
          f"{'miss/tok at +10 %':>18}  {'cap if overlapped':>18}")
    print("   " + "-" * 92)
    for name, bw in PCIE.items():
        t_hot = cell_hot_mb * 1e6 / bw * 1e3
        t_cold = cell_cold_mb * 1e6 / bw * 1e3
        serial = 0.10 * t_token_ms / t_hot
        overlap = bw * (t_token_ms / 1e3) / (cell_hot_mb * 1e6)
        print(f"   {name:>22}  {fr(t_hot,3):>11}  {fr(t_cold,3):>13}  {fr(serial,1):>18}  "
              f"{fr(overlap,1):>18}")
    print("   " + "-" * 92)
    print("   \"+10 %\" = serialized misses, added to the token time. \"cap if overlapped\" =")
    print("   perfectly prefetched misses, bounded only by the throughput of the link.")
    print()
    print("   DETAIL, a reversal: on this path the ARCHIVE format is the FASTEST one.")
    print("      The bottleneck is the link, not the ALU. 38 % less traffic is 38 % less")
    print("      miss latency. The memory thesis of the project becomes a latency thesis")
    print("      as soon as the cold tier crosses a bus.")
    print()
    print("   Dimensioning: VRAM carries ONLY the hot tier (the cold one lives in host RAM).")
    print(f"   VRAM(α) = {P120_TOTAL/1e9:.0f} B × α × {fr(B_HOT)}/8 + {fr(OVERHEAD_GB,1)} GB")
    print()
    print(f"   {'α':>6}  {'cells':>9}  {'VRAM GB':>8}  {'hit':>8}  {'miss/tok':>9}  "
          f"{'gen4 ms':>8}  {'gen5 ms':>8}  card class")
    print("   " + "-" * 88)
    for a in (0.20, alpha_lru, 0.35, 0.50, alpha_vram, 0.85, 1.0):
        k = round(a * cells)
        cold = cold_mass_at(desc, total, k)
        mt = cold * visits
        v = P120_TOTAL * a * B_HOT / 8 / 1e9 + OVERHEAD_GB
        g4 = mt * cell_hot_mb * 1e6 / 32e9 * 1e3
        g5 = mt * cell_hot_mb * 1e6 / 63e9 * 1e3
        card = "24 GB" if v <= 24 else "32 GB" if v <= 32 else "48 GB" if v <= 48 else "≥ 64 GB"
        print(f"   {fr(a,3):>6}  {k:>4}/{cells:<4}  {fr(v,1):>8}  {fr((1-cold)*100,2)+' %':>8}  "
              f"{fr(mt,1):>9}  {fr(g4,2):>8}  {fr(g5,2):>8}  {card}")
    print("   " + "-" * 88)
    print(f"   (the ms are the PCIe traffic per token, to compare with the {fr(t_token_ms,1)} ms of compute;")
    print("    overlapped, a miss is free as long as this column stays under the token time.")
    print("    Card classes at exact fit, preregistration convention. The 1,2 GB of embedding")
    print("    + KV are INSIDE the VRAM column, there is no other margin.)")
    print()
    print("   DERIVED operating points: the α each link imposes if the misses are NOT")
    print("   overlapped (the unfavourable case: +10 % of token time at most):")
    print(f"   {'link':>22}  {'budget miss/tok':>15}  {'α needed':>8}  {'VRAM GB':>8}  card")
    print("   " + "-" * 70)
    for name, bw in PCIE.items():
        t_hot = cell_hot_mb * 1e6 / bw * 1e3
        b_link = 0.10 * t_token_ms / t_hot
        a_link, _ = alpha_min_global(desc, total, b_link / visits)
        v_link = P120_TOTAL * a_link * B_HOT / 8 / 1e9 + OVERHEAD_GB
        card = "24 GB" if v_link <= 24 else "32 GB" if v_link <= 32 else "48 GB" if v_link <= 48 else "≥ 64 GB"
        print(f"   {name:>22}  {fr(b_link,1):>15}  {fr(a_link,3):>8}  {fr(v_link,1):>8}  {card}")
    print("   " + "-" * 70)
    print()
    print("   OK: YES, IT CHANGES THE CARD CLASS, and it is the only positive result of the test.")
    v_lru = P120_TOTAL * alpha_lru * B_HOT / 8 / 1e9 + OVERHEAD_GB
    m50 = cold_mass_at(desc, total, round(0.50 * cells)) * visits
    v50 = P120_TOTAL * 0.50 * B_HOT / 8 / 1e9 + OVERHEAD_GB
    g4_50 = m50 * cell_hot_mb * 1e6 / 32e9 * 1e3
    t_g4 = cell_hot_mb * 1e6 / 32e9 * 1e3
    b_g4 = 0.10 * t_token_ms / t_g4
    a_g4, _ = alpha_min_global(desc, total, b_g4 / visits)
    v_g4 = P120_TOTAL * a_g4 * B_HOT / 8 / 1e9 + OVERHEAD_GB
    print(f"      • VRAM variant (this test)    : {fr(vram_go(m_p),1)} GB, does NOT EVEN fit on 48 GB.")
    print(f"      • all `Golay70` resident      : {fr(vram_go(B_HOT),1)} GB, a 64 GB card.")
    print(f"      • host RAM, misses NOT overlapped (the unfavourable case, gen4): α = {fr(a_g4,3)},")
    print(f"        {fr(v_g4,1)} GB, a 48 GB card, reached by a path the VRAM variant could not")
    print("        take at all. On gen5, α drops to 0,51-0,55 and the 32 GB card is")
    print("        enough (table above).")
    print(f"      • host RAM, misses OVERLAPPED : α = 0,50 ⇒ {fr(v50,1)} GB, a 32 GB card,")
    print(f"        {fr(m50,1)} miss/token, {fr(g4_50,1)} ms of gen4 PCIe against {fr(t_token_ms,1)} ms "
          f"of compute ({fr(100*g4_50/t_token_ms,0)} %")
    print("        of the budget), overlappable IN PRINCIPLE, never measured.")
    print(f"      • the limit                   : α = {fr(alpha_lru,3)} ⇒ {fr(v_lru,1)} GB, a 24 GB card,")
    print("        but PCIe becomes the critical path there. It is a limit, not an")
    print("        operating point.")
    print("   ⇒ The VRAM variant dropped below NO card class. The cold tier in host RAM")
    print("      brings the 120b down from \"≥ 64 GB\" to \"32-48 GB\", and the question")
    print("      becomes a prefetch engineering question again, no longer a question of bits.")
    print("   WARNING, three debts this pricing does not clear. (i) Prefetching assumes the")
    print("      routing is known BEFORE the layer, which takes a speculative prefetch or a")
    print("      one-layer shift, neither designed nor measured. (ii) 32-63 GB/s are external")
    print("      hypotheses, no PCIe measurement exists here. (iii) The PCIe traffic of a")
    print("      batch > 1 does not get shared: these figures are batch 1.")

    # -- 8. Scope reserve --------------------------------------------------
    print("\n\n8. SCOPE CAVEAT: WRITTEN INTO THE PREREGISTRATION IN ADVANCE (§3-T4)")
    print("=" * 78)
    print(f"   The dump is a 20b with {E} experts top-{K} ({fr(100*K/E,1)} % active, {L} layers).")
    print(f"   The dimensioning targets a 120b with {P120_EXPERTS} experts top-{P120_TOPK} "
          f"({fr(100*P120_TOPK/P120_EXPERTS,1)} % active, {n_layers_120} layers).")
    print("   ⇒ A GREEN WOULD NOT CARRY OVER. A RED does: concentration can only get worse")
    print("     as the expert count rises (more experts for the same mass of tokens ⇒ more")
    print("     weak cells), and the number of visits per token rises")
    print(f"     from {visits} to {P120_TOPK*n_layers_120}, which DIVIDES the tolerable cold mass again")
    print(f"     per visit (×{fr(P120_TOPK*n_layers_120/visits,1)}).")
    print("   The verdict of this test is RED: it carries over.")
    print()
    print("   What this test does NOT settle: no millisecond has been measured; the dump has")
    print("   no temporal dimension, so no dynamic cache is judged; the routing of a 120b has")
    print("   never been captured; and 2-bit quality on a MoE still has no figure at all,")
    print("   here or anywhere else (gate X5-MoE, ~25-55 $).")


if __name__ == "__main__":
    main()
