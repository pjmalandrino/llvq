#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""T4 of the 2026-08-13 pre-registration: the hot/cold MoE scissor, priced.

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
also prices the alternative the pre-registration demands be named: cold tier in
**host RAM**, miss = PCIe memcpy.

    uv run ops/moe_ciseau.py

Standard library only, and deliberately so: this is arithmetic over a 5.7 kB
JSON, it must run on a machine with no ML stack (the repo's conda env is
broken: torch 2.5.1 vs transformers 5.5.4).

No number is invented. Every constant carries its provenance as
`file:line` in the CONSTANTS block below, and every printed figure is tagged
*mesuré* / *calculé* / *estimé* per the repo rule (CLAUDE.md section 7).
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
#   Golay70 : CLAUDE.md section 3 "échelle des formats", 3,589 — exact
#             reconstruction proven on 150,7 M blocks, 1,31x vs FP16 (v1).
#   archive : docs/archive/spec-memoire-extreme-2026-08-12.md:56
#             "plancher (le fichier) | 2,219" — the sealed file's own rate.
B_HOT = 3.589
B_COLD = 2.219

# Decode throughputs.
#   195 Go/s : Golay70 effective, L40S, 7-arm bench
#              (docs/archive/etude-moe-memoire-extreme-2026-08-12.md:51-55).
#   8,27 ns/bloc : the v1 rank decoder, M3 Max GPU over 16,7 M blocks
#              (docs/format-noyau.md:55 and :120, "106x le sol").
#   ⚠️ Two different machines. The composite below is an order of magnitude,
#   not a precision figure — labelled *estimé* wherever it is printed.
GBPS_HOT = 195.0
NS_PER_BLOCK_RANK = 8.27
WEIGHTS_PER_BLOCK = 24  # Lambda_24, by construction

# gpt-oss-120B, the dimensioning target.
#   117 Md total / 5,1 Md active : docs/archive/etude-moe-memoire-extreme-2026-08-12.md:25 (web sheet)
#   128 experts                  : proofs/preregistration-2026-08-13.md:169-170
#   layers                       : CALCULATED below, cross-checked on the 20B.
P120_TOTAL = 117e9
P120_ACTIVE = 5.1e9
P120_EXPERTS = 128
P120_TOPK = 4

# Memory bar. proofs/preregistration-2026-08-13.md:160-161 fixes the triple:
# 48 GB card, 117 Md total, ~1,2 GB deducted for embedding + KV.
# 117e9 * 3.20/8 = 46,8 GB, +1,2 = 48,0 GB exactly — so this IS the convention
# the 3,20 threshold was derived in, and the one every VRAM line below uses.
CARD_GB = 48.0
OVERHEAD_GB = 1.2
S_MIX = 3.20

# The two miss budgets the lot must report, in misses per token, whole model.
#   0,70 : proposed by the piste.
#   0,29 : the counter-expertise, "2,4x too generous" (0,70 / 0,29 = 2,41).
# ⚠️ PROVENANCE DEBT, stated up front: neither number has a source file in
# this repo (grepped 2026-08-13). They enter as hypotheses; section 4 derives
# what the measured decode cost actually supports, and the verdict is computed
# on a sweep so that it does not depend on picking one.
BUDGET_PISTE = 0.70
BUDGET_CONTRE = 0.29

# PCIe effective bandwidths, host->device, large contiguous copies.
# ⚠️ EXTERNAL HYPOTHESIS: nothing in this repo measures PCIe. Estimated.
PCIE = {"gen4 x16 (~32 Go/s)": 32e9, "gen5 x16 (~50 Go/s)": 50e9, "gen5 x16 (~63 Go/s)": 63e9}

# LRU branch, frozen in the pre-registration (:163-164): resident backing means
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
        sys.exit(f"ABSENT : {path} — le dump de routage est la seule entrée mesurée de ce script.")
    d = json.load(open(path))

    counts = d["counts"]
    L, E, K = len(counts), d["n_experts"], d["top_k"]
    hidden, moe_inter, ntok = d["hidden"], d["moe_intermediate"], d["tokens"]
    cells = L * E
    visits = K * L  # routed cell-visits per token, whole model

    print("# Le ciseau MoE chaud/froid — T4 du pré-enregistrement du 2026-08-13")
    print(f"# date        : {date.today().isoformat()}")
    print("# commande    : uv run ops/moe_ciseau.py")
    print(f"# source      : {DUMP} (mesuré — 131 072 tokens de C4, M3 Max, 2026-08-12)")
    print(f"# juge        : {PREREG} §3-T4 — seuil mélange > 3,20 b/poids ⇒ variante VRAM enterrée")
    print(f"# script      : ops/moe_ciseau.py (stdlib seule, 0 $, aucun GPU, aucun réseau)")
    print("#")
    print("# Étiquettes : mesuré = sort d'un fichier scellé ou d'un banc ; calculé = arithmétique")
    print("# exacte sur des mesurés ; estimé = repose sur une hypothèse externe, nommée sur place.")

    # -- 0. Non-regression control -----------------------------------------
    print("\n\n0. CONTRÔLE DE PARSING — les Gini publiés doivent retomber")
    print("-" * 78)
    print("   Si cette section ne reproduit pas moe-routing-gptoss20b-2026-08-12.txt,")
    print("   rien de ce qui suit n'est lisible.")
    per_layer_sums = {sum(r) for r in counts}
    ok_sums = per_layer_sums == {ntok * K}
    agg = [sum(counts[l][e] for l in range(L)) for e in range(E)]
    g0, g23, gagg = gini(counts[0]), gini(counts[L - 1]), gini(agg)
    print(f"   somme par couche      : {sorted(per_layer_sums)[0]} = {ntok} × {K}   "
          f"{'OK' if ok_sums else 'ÉCHEC'}   (mesuré)")
    print(f"   Gini couche 0         : {fr(g0)}   attendu 0,351   {'OK' if abs(g0-0.351)<5e-4 else 'ÉCHEC'}")
    print(f"   Gini couche {L-1}        : {fr(g23)}   attendu 0,748   {'OK' if abs(g23-0.748)<5e-4 else 'ÉCHEC'}")
    print(f"   Gini agrégé (TOUTES)  : {fr(gagg)}   attendu 0,169   {'OK' if abs(gagg-0.169)<5e-4 else 'ÉCHEC'}")
    if not (ok_sums and abs(g0 - 0.351) < 5e-4 and abs(g23 - 0.748) < 5e-4 and abs(gagg - 0.169) < 5e-4):
        sys.exit("   ÉCHEC du contrôle — parsing faux, on s'arrête ici.")
    print("   ✅ les trois Gini publiés retombent au millième : le parsing est bon.")

    # -- 1. Geometry, and the internal control that validates it -----------
    per_cell = 3 * hidden * moe_inter  # SwiGLU: gate + up + down
    exp20 = L * E * per_cell
    act20 = K * L * per_cell
    print("\n\n1. GÉOMÉTRIE DES CELLULES — et le contrôle qui la valide")
    print("-" * 78)
    print(f"   dump          : {d['model']} — {L} couches × {E} experts = {cells} cellules,")
    print(f"                   top-{K}, hidden {hidden}, moe_inter {moe_inter}   (mesuré)")
    print(f"   cellule       : 3 × {hidden} × {moe_inter} = {per_cell:,} poids (gate+up+down)   (calculé)"
          .replace(",", " "))
    print(f"   experts 20B   : {L}×{E}×cellule = {fr(exp20/1e9,2)} Md ; +hors-experts ⇒ 21 Md de fiche")
    print(f"   actifs 20B    : {K}×{L}×cellule = {fr(act20/1e9,2)} Md ; +hors-experts ⇒ 3,6 Md de fiche")
    p20_total, p20_active = 21e9, 3.6e9  # fiche web, etude-moe:22
    nonexp20 = p20_total - exp20
    nonexp20_active = p20_active - act20
    print(f"   ✅ contrôle : la fiche de l'étude donne {p20_total/1e9:.0f} Md totaux / "
          f"{fr(p20_active/1e9,1)} Md actifs (etude-moe:22).")
    print(f"      Le modèle à 3 matrices laisse {fr(nonexp20/1e9,2)} Md hors experts et "
          f"{fr(nonexp20_active/1e9,2)} Md hors")
    print("      experts actifs — soit exactement l'attention + l'embedding. La géométrie tient.")
    # 120B layer count: solve L from the total, charging the 20B's per-layer
    # non-expert cost. Cross-checked on the active count just below.
    nonexp_per_layer = nonexp20 / L
    n_layers_120 = round(P120_TOTAL / (P120_EXPERTS * per_cell + nonexp_per_layer))
    exp120 = n_layers_120 * P120_EXPERTS * per_cell
    act120 = P120_TOPK * n_layers_120 * per_cell
    print(f"   120B          : {P120_TOTAL/1e9:.0f} Md totaux / {fr(P120_ACTIVE/1e9,1)} Md actifs, "
          f"{P120_EXPERTS} experts top-{P120_TOPK}   (fiche + pré-enregistrement:169)")
    print(f"                   ⇒ {n_layers_120} couches   (CALCULÉ en chargeant le hors-experts")
    print(f"                   par couche du 20B : {P120_TOTAL/1e9:.0f} / ({P120_EXPERTS}×cellule + "
          f"{nonexp_per_layer/1e6:.0f} M))")
    print(f"      contrôle 1 : {n_layers_120}×{P120_EXPERTS}×cellule = {fr(exp120/1e9,1)} Md d'experts, "
          f"reste {fr((P120_TOTAL-exp120)/1e9,2)} Md hors experts")
    print(f"                   (contre {fr(nonexp20/1e9,2)} Md à {L} couches — ×"
          f"{fr((P120_TOTAL-exp120)/nonexp20,2)} pour ×{fr(n_layers_120/L,2)} de couches) ✅")
    print(f"      contrôle 2 : actifs = {P120_TOPK}×{n_layers_120}×cellule = {fr(act120/1e9,2)} Md, "
          f"+{fr((P120_ACTIVE-act120)/1e9,2)} Md hors experts")
    print(f"                   = les {fr(P120_ACTIVE/1e9,1)} Md annoncés (contre "
          f"{fr(nonexp20_active/1e9,2)} Md hors experts au 20B) ✅")
    print("   ⚠️ hypothèse : hidden et moe_inter du 120B supposés identiques au 20B (2880).")
    print("      Le dump ne le dit pas ; ce sont les deux contrôles ci-dessus qui la soutiennent.")
    print("      *estimé*. Le compte de couches ne sert qu'à deux comparaisons secondaires")
    print("      (temps d'une couche, §4 ; visites par token, §8) : ±1 couche ne change rien.")

    # -- 2. Load distribution, layer by layer ------------------------------
    print("\n\n2. DISTRIBUTION DE CHARGE ROUTÉE, PAR CELLULE ET PAR COUCHE   (mesuré)")
    print("-" * 78)
    print("   Charge en multiple de l'uniforme ; α_min = plus petite fraction de cellules")
    print("   CHAUDES de la couche telle que la masse froide reste sous le budget, allocation")
    print("   uniforme entre couches (budget/couche = budget/24).")
    print()
    u_c = alpha_min_uniform(counts, BUDGET_CONTRE / visits)
    u_p = alpha_min_uniform(counts, BUDGET_PISTE / visits)
    uniform_load = ntok * K / E
    print(f"   {'couche':>6}  {'min':>7}  {'médiane':>8}  {'max':>7}  {'Gini':>6}  "
          f"{'α_min 0,29':>10}  {'α_min 0,70':>10}")
    print("   " + "-" * 71)
    for li in range(L):
        v = sorted(x / uniform_load for x in counts[li])
        print(f"   {li:>6}  {fr(v[0]):>7}  {fr(quantile(v,0.5)):>8}  {fr(v[-1]):>7}  "
              f"{fr(gini(counts[li])):>6}  {fr(u_c[li],3):>10}  {fr(u_p[li],3):>10}")
    print("   " + "-" * 71)
    print(f"   {'moyenne':>6}  {'':>7}  {'':>8}  {'':>7}  "
          f"{fr(sum(gini(r) for r in counts)/L):>6}  {fr(sum(u_c)/L):>10}  {fr(sum(u_p)/L):>10}")
    print()
    print(f"   Gini par couche : {fr(min(gini(r) for r in counts))} … {fr(max(gini(r) for r in counts))}")
    print("   ⚠️ La fourchette « 0,59-0,77 » à recouper est celle des couches PROFONDES : les")
    print("      couches 0-1 et 6 sont bien plus plates (0,351 / 0,426 / 0,448). La plage")
    print("      complète des 24 couches est celle imprimée ci-dessus, et c'est elle qui compte")
    print("      ici, parce que ce sont les couches PLATES qui coûtent cher au ciseau.")
    print("   🔎 Mécanisme, contre-intuitif et central : un Gini ÉLEVÉ est FAVORABLE au ciseau")
    print("      (peu de cellules portent la masse). Le poste de coût, ce sont les couches")
    print(f"      plates — à 0,29 miss/token la couche 0 (Gini {fr(gini(counts[0]))}, la plus plate)")
    print(f"      exige α = {fr(u_c[0])} : TOUTES ses cellules chaudes. La couche {L-1} "
          f"(Gini {fr(gini(counts[L-1]))}),")
    print(f"      la plus concentrée, n'en exige que {fr(u_c[L-1])}.")

    # -- 3. The scissor ----------------------------------------------------
    desc = sorted((v for r in counts for v in r), reverse=True)
    total = sum(desc)
    alpha_vram = (S_MIX - B_COLD) / (B_HOT - B_COLD)
    print("\n\n3. LE CISEAU — α_min (lame « hit ») contre α_VRAM (lame « mémoire »)")
    print("-" * 78)
    print(f"   α_VRAM = ({fr(S_MIX,2)} − {fr(B_COLD)}) / ({fr(B_HOT)} − {fr(B_COLD)}) = "
          f"{fr(alpha_vram,4)}   (calculé)")
    print(f"   Au-dessus, le mélange dépasse {fr(S_MIX,2)} b/poids et le barreau 48 Go se ferme.")
    print()
    print(f"   {'budget':>8}  {'masse froide':>12}  {'α_min':>7}  {'cellules':>9}  "
          f"{'mélange':>8}  {'VRAM 120B':>10}  verdict")
    print(f"   {'miss/tok':>8}  {'max':>12}  {'global':>7}  {'chaudes':>9}  "
          f"{'b/poids':>8}  {'Go':>10}")
    print("   " + "-" * 74)
    sweep = [0.137, BUDGET_CONTRE, 0.5, BUDGET_PISTE, 1.0, 2.0, 5.0, 10.0, 25.0]
    rows = {}
    for b in sweep:
        f_max = b / visits
        a, k = alpha_min_global(desc, total, f_max)
        m = mix(a)
        rows[b] = (a, k, m)
        tag = "ROUGE" if m > S_MIX else "vert"
        print(f"   {fr(b,3):>8}  {fr(f_max*100,4)+' %':>12}  {fr(a,4):>7}  {k:>4}/{cells:<4}  "
              f"{fr(m,4):>8}  {fr(vram_go(m),1):>10}  {tag}")
    print("   " + "-" * 74)
    need = bisect_budget(desc, total, K, L, alpha_vram)
    print("   ⚠️ Les lignes ≥ 2 miss/token ne sont PAS des points de fonctionnement : le §4")
    print("      montre qu'un miss coûte 8,57 ms contre 11,73 ms de temps de token. « vert »")
    print("      y veut dire « la mémoire passerait », pas « le produit existe ».")
    print(f"   Budget qu'il FAUDRAIT pour atteindre α_VRAM = {fr(alpha_vram,4)} : "
          f"{fr(need,2)} miss/token   (calculé)")
    print(f"   soit ×{fr(need/BUDGET_PISTE,1)} le budget de la piste et ×{fr(need/BUDGET_CONTRE,1)} "
          "celui de la contre-expertise.")

    # -- 4. What a miss actually costs -------------------------------------
    blocks_cell = per_cell / WEIGHTS_PER_BLOCK
    t_token_ms = P120_ACTIVE * B_HOT / 8 / (GBPS_HOT * 1e9) * 1e3
    t_miss_vram_ms = blocks_cell * NS_PER_BLOCK_RANK / 1e6
    t_layer_ms = t_token_ms / n_layers_120
    print("\n\n4. CE QUE COÛTE VRAIMENT UN MISS — d'où sort un budget défendable")
    print("-" * 78)
    print(f"   temps/token 120B, tout chaud    : {fr(P120_ACTIVE/1e9,1)} Md × {fr(B_HOT)}/8 = "
          f"{fr(P120_ACTIVE*B_HOT/8/1e9,3)} Go ÷ {GBPS_HOT:.0f} Go/s")
    print(f"                                     = {fr(t_token_ms,2)} ms  ({1000/t_token_ms:.0f} tok/s)"
          "   *estimé* (plafond de bande")
    print("                                     passante, pas un débit prédit — étude §2)")
    print("   un miss en VARIANTE VRAM        : la cellule froide se décode par le rang v1,")
    print(f"                                     {blocks_cell:,.0f}".replace(",", " ")
          + f" blocs × {fr(NS_PER_BLOCK_RANK,2)} ns/bloc")
    print("                                     (mesuré, format-noyau.md:120)")
    print(f"                                     = {fr(t_miss_vram_ms,2)} ms, soit "
          f"{t_miss_vram_ms/t_layer_ms:.0f}× le temps d'UNE COUCHE entière   *estimé*")
    print("   ⚠️ Les 195 Go/s sont CUDA/L40S et les 8,27 ns/bloc Metal/M3 Max : le composite")
    print("      est un ordre de grandeur, jamais une précision. Il ne sert qu'à trancher un")
    print("      facteur 30, ce qu'un facteur 2 d'incertitude ne renverse pas.")
    print()
    print(f"   {'budget':>8}  {'coût des miss':>13}  {'% du temps':>11}  statut")
    print(f"   {'miss/tok':>8}  {'ms/token':>13}  {'de token':>11}")
    print("   " + "-" * 60)
    for b, label in ((0.137, "+10 % de temps de token"), (BUDGET_CONTRE, "contre-expertise"),
                     (BUDGET_PISTE, "la piste"), (need, "ce qu'il faudrait")):
        c = b * t_miss_vram_ms
        print(f"   {fr(b,3):>8}  {fr(c,2):>13}  {fr(100*c/t_token_ms,1)+' %':>11}  {label}")
    print("   " + "-" * 60)
    print("   ✅ VERDICT SUR LE BUDGET : c'est 0,29 qui est défendable, et il est déjà généreux.")
    print(f"      À 0,70 miss/token le tier froid mange {fr(100*BUDGET_PISTE*t_miss_vram_ms/t_token_ms,0)} %"
          " du temps de token — un « budget » qui")
    print("      redéfinit la moitié du produit n'est pas un budget. Le seuil à +10 % de temps")
    print(f"      de token vaut {fr(0.137,3)} miss/token, soit ×{fr(BUDGET_PISTE/0.137,1)} plus strict")
    print("      encore que la contre-expertise. Le facteur 2,4 qu'elle dénonce est un minorant.")
    print()
    b_prereg = (1 - LRU_HIT_REQUIRED / 100) * visits
    a_prereg, _ = alpha_min_global(desc, total, b_prereg / visits)
    print("   🔎 Recoupement indépendant, et il est dans le pré-enregistrement lui-même.")
    print(f"      Son seuil de hit LRU ({fr(LRU_HIT_REQUIRED,1)} %) équivaut à "
          f"{fr(b_prereg,3)} miss/token, et ce budget")
    print(f"      rend α_min = {fr(a_prereg,3)} — c'est-à-dire EXACTEMENT le « α_hit ≈ 0,85 » qu'il")
    print("      prédisait. Les deux chiffres du §3-T4 sont donc une seule affirmation, et elle")
    print(f"      se situe à {fr(b_prereg,2)} miss/token : entre mon seuil dérivé ({fr(0.137,3)}) et celui de")
    print(f"      la contre-expertise ({fr(BUDGET_CONTRE,2)}), très loin des {fr(BUDGET_PISTE,2)} de la piste.")
    print("      Trois dérivations indépendantes convergent sous 0,30 ; une seule reste dehors.")

    # -- 5. Verdict --------------------------------------------------------
    a_c, _, m_c = rows[BUDGET_CONTRE]
    a_p, _, m_p = rows[BUDGET_PISTE]
    print("\n\n5. VERDICT CONTRE LE SEUIL PRÉ-ENREGISTRÉ (mélange > 3,20 ⇒ VRAM enterrée)")
    print("=" * 78)
    print(f"   budget 0,29 (défendable) : α_min = {fr(a_c,4)} > α_VRAM = {fr(alpha_vram,4)}  ⇒ "
          f"mélange {fr(m_c,3)} b/poids, VRAM {fr(vram_go(m_c),1)} Go")
    print(f"   budget 0,70 (généreux)   : α_min = {fr(a_p,4)} > α_VRAM = {fr(alpha_vram,4)}  ⇒ "
          f"mélange {fr(m_p,3)} b/poids, VRAM {fr(vram_go(m_p),1)} Go")
    print()
    print("   ❌ INTERVALLE VIDE AUX DEUX BUDGETS — la variante VRAM est ENTERRÉE.")
    print(f"      Le mélange dépasse {fr(S_MIX,2)} même sous l'allocation la PLUS favorable")
    print("      (classement global des 768 cellules, oracle statique, allocation optimale).")
    print(f"      Et il le dépasse encore au budget le plus généreux du dossier : à 0,70")
    print(f"      miss/token, {fr(m_p,3)} > {fr(S_MIX,2)}. Aucun choix de budget entre les deux ne sauve")
    print(f"      la piste — il faudrait {fr(need,2)} miss/token, hors de tout budget défendable.")
    print()
    print("   🔎 Face à la prédiction pré-enregistrée (α_hit ≈ 0,85 ; α_VRAM ≤ 0,72 ; mélange")
    print(f"      3,4-3,5) : α_VRAM = {fr(alpha_vram,4)} tombe pile, α_hit = {fr(a_p,3)}-{fr(a_c,3)} est")
    print(f"      un peu SOUS 0,85, et le mélange atterrit à {fr(m_p,2)}-{fr(m_c,2)}, sous la fourchette")
    print("      prédite. Le verdict est celui qui était annoncé ; la marge est plus mince que")
    print("      prédit, et c'est la seule chose que ce lot dément.")

    # -- 6. LRU branch -----------------------------------------------------
    alpha_lru = (S_MIX - B_COLD) / B_HOT
    k_lru = round(alpha_lru * cells)
    hit_lru = (1 - cold_mass_at(desc, total, k_lru)) * 100
    miss_lru = cold_mass_at(desc, total, k_lru) * visits
    print("\n\n6. LA BRANCHE CACHE LRU — tuée par arithmétique, puis par la mesure")
    print("-" * 78)
    print(f"   Backing résident ⇒ le froid est payé EN ENTIER, plus le cache :")
    print(f"   {fr(B_COLD)} + α×{fr(B_HOT)} ≤ {fr(S_MIX,2)}  ⟺  α ≤ {fr(alpha_lru,4)}   "
          "(calculé — pré-enregistrement §3-T4)")
    print(f"   À α = {fr(alpha_lru,4)} ({k_lru} cellules sur {cells}) :")
    print(f"     hit d'un cache STATIQUE ORACLE (les {k_lru} plus chargées) : {fr(hit_lru,2)} %   (mesuré)")
    print(f"     ⇒ {fr(miss_lru,1)} miss/token, contre un seuil pré-enregistré de {fr(LRU_HIT_REQUIRED,1)} %")
    print(f"       ({fr((1-LRU_HIT_REQUIRED/100)*visits,2)} miss/token)")
    print(f"   ❌ Écart : ×{fr(miss_lru/((1-LRU_HIT_REQUIRED/100)*visits),0)} le budget de miss. "
          "La branche LRU est morte de deux façons")
    print("      indépendantes — l'arithmétique du backing, puis la concentration réelle.")
    print("   ⚠️ Et l'oracle statique n'est PAS un LRU : le dump est AGRÉGÉ sur 131 k tokens,")
    print("      il ne porte aucune information temporelle. Un LRU peut battre l'oracle statique")
    print("      s'il existe de la localité temporelle — cette mesure ne peut pas en décider.")
    print(f"      Mais l'écart à combler est ×{fr(miss_lru/((1-LRU_HIT_REQUIRED/100)*visits),0)}, "
          "pas quelques pour cent : aucune localité")
    print("      plausible ne le referme. Trancher exigerait un dump TEMPOREL, qui n'existe pas.")

    # -- 7. The dominant alternative ---------------------------------------
    cell_hot_mb = per_cell * B_HOT / 8 / 1e6
    cell_cold_mb = per_cell * B_COLD / 8 / 1e6
    print("\n\n7. L'ALTERNATIVE DOMINANTE — tier froid en RAM HÔTE, miss = memcpy PCIe")
    print("=" * 78)
    print("   Le pré-enregistrement l'exige : « un enterrement qui ne nomme pas ce qui le")
    print("   remplace n'est pas un verdict, c'est un abandon. »")
    print()
    print(f"   Taille d'une cellule d'expert (120B, hidden 2880)   (calculé) :")
    print(f"     format servi   `Golay70` {fr(B_HOT)} b/poids : {fr(cell_hot_mb,2)} Mo")
    print(f"     format archive {fr(B_COLD)} b/poids           : {fr(cell_cold_mb,2)} Mo  "
          f"(−{fr(100*(1-cell_cold_mb/cell_hot_mb),0)} % de trafic)")
    print()
    print(f"   Coût d'un miss = un memcpy PCIe de cette cellule   (estimé — aucune mesure PCIe")
    print("   dans ce dépôt) :")
    print(f"   {'lien':>22}  {'servi (ms)':>11}  {'archive (ms)':>13}  "
          f"{'miss/token à +10 %':>18}  {'plafond recouvert':>18}")
    print("   " + "-" * 92)
    for name, bw in PCIE.items():
        t_hot = cell_hot_mb * 1e6 / bw * 1e3
        t_cold = cell_cold_mb * 1e6 / bw * 1e3
        serial = 0.10 * t_token_ms / t_hot
        overlap = bw * (t_token_ms / 1e3) / (cell_hot_mb * 1e6)
        print(f"   {name:>22}  {fr(t_hot,3):>11}  {fr(t_cold,3):>13}  {fr(serial,1):>18}  "
              f"{fr(overlap,1):>18}")
    print("   " + "-" * 92)
    print("   « +10 % » = miss sérialisés, ajoutés au temps de token. « plafond recouvert » =")
    print("   miss parfaitement préchargés, bornés seulement par le débit du lien.")
    print()
    print("   🔎 Un renversement à noter : sur ce chemin, le format ARCHIVE est le plus RAPIDE.")
    print("      Le goulot est le lien, pas l'ALU — 38 % de trafic en moins, c'est 38 % de")
    print("      latence de miss en moins. La thèse mémoire du projet devient une thèse de")
    print("      latence dès que le tier froid traverse un bus.")
    print()
    print("   Dimensionnement : la VRAM ne porte QUE le tier chaud (le froid vit en RAM hôte).")
    print(f"   VRAM(α) = {P120_TOTAL/1e9:.0f} Md × α × {fr(B_HOT)}/8 + {fr(OVERHEAD_GB,1)} Go")
    print()
    print(f"   {'α':>6}  {'cellules':>9}  {'VRAM Go':>8}  {'hit':>8}  {'miss/tok':>9}  "
          f"{'gen4 ms':>8}  {'gen5 ms':>8}  classe de carte")
    print("   " + "-" * 88)
    for a in (0.20, alpha_lru, 0.35, 0.50, alpha_vram, 0.85, 1.0):
        k = round(a * cells)
        cold = cold_mass_at(desc, total, k)
        mt = cold * visits
        v = P120_TOTAL * a * B_HOT / 8 / 1e9 + OVERHEAD_GB
        g4 = mt * cell_hot_mb * 1e6 / 32e9 * 1e3
        g5 = mt * cell_hot_mb * 1e6 / 63e9 * 1e3
        card = "24 Go" if v <= 24 else "32 Go" if v <= 32 else "48 Go" if v <= 48 else "≥ 64 Go"
        print(f"   {fr(a,3):>6}  {k:>4}/{cells:<4}  {fr(v,1):>8}  {fr((1-cold)*100,2)+' %':>8}  "
              f"{fr(mt,1):>9}  {fr(g4,2):>8}  {fr(g5,2):>8}  {card}")
    print("   " + "-" * 88)
    print(f"   (les ms sont le trafic PCIe par token, à comparer aux {fr(t_token_ms,1)} ms de calcul ;")
    print("    recouvert, un miss est gratuit tant que cette colonne reste sous le temps de token.")
    print("    Classes de carte à l'ajustement exact, convention du pré-enregistrement — les")
    print("    1,2 Go d'embedding + KV sont DANS la colonne VRAM, il n'y a pas d'autre marge.)")
    print()
    print("   Points de fonctionnement DÉRIVÉS — le α que chaque lien impose si les miss ne")
    print("   sont PAS recouverts (le cas défavorable : +10 % de temps de token au plus) :")
    print(f"   {'lien':>22}  {'budget miss/tok':>15}  {'α requis':>8}  {'VRAM Go':>8}  carte")
    print("   " + "-" * 70)
    for name, bw in PCIE.items():
        t_hot = cell_hot_mb * 1e6 / bw * 1e3
        b_link = 0.10 * t_token_ms / t_hot
        a_link, _ = alpha_min_global(desc, total, b_link / visits)
        v_link = P120_TOTAL * a_link * B_HOT / 8 / 1e9 + OVERHEAD_GB
        card = "24 Go" if v_link <= 24 else "32 Go" if v_link <= 32 else "48 Go" if v_link <= 48 else "≥ 64 Go"
        print(f"   {name:>22}  {fr(b_link,1):>15}  {fr(a_link,3):>8}  {fr(v_link,1):>8}  {card}")
    print("   " + "-" * 70)
    print()
    print("   ✅ OUI, ÇA CHANGE LA CLASSE DE CARTE, et c'est le seul résultat positif du test.")
    v_lru = P120_TOTAL * alpha_lru * B_HOT / 8 / 1e9 + OVERHEAD_GB
    m50 = cold_mass_at(desc, total, round(0.50 * cells)) * visits
    v50 = P120_TOTAL * 0.50 * B_HOT / 8 / 1e9 + OVERHEAD_GB
    g4_50 = m50 * cell_hot_mb * 1e6 / 32e9 * 1e3
    t_g4 = cell_hot_mb * 1e6 / 32e9 * 1e3
    b_g4 = 0.10 * t_token_ms / t_g4
    a_g4, _ = alpha_min_global(desc, total, b_g4 / visits)
    v_g4 = P120_TOTAL * a_g4 * B_HOT / 8 / 1e9 + OVERHEAD_GB
    print(f"      • variante VRAM (ce test)     : {fr(vram_go(m_p),1)} Go — ne tient MÊME PAS sur 48 Go.")
    print(f"      • tout `Golay70` résident     : {fr(vram_go(B_HOT),1)} Go — une carte 64 Go.")
    print(f"      • RAM hôte, miss NON recouverts (le cas défavorable, gen4) : α = {fr(a_g4,3)},")
    print(f"        {fr(v_g4,1)} Go — une carte 48 Go, mais on y arrive par un chemin que la variante")
    print("        VRAM ne pouvait pas prendre du tout. En gen5, α tombe à 0,51-0,55 et la")
    print("        carte 32 Go suffit (tableau ci-dessus).")
    print(f"      • RAM hôte, miss RECOUVERTS   : α = 0,50 ⇒ {fr(v50,1)} Go, une carte 32 Go,")
    print(f"        {fr(m50,1)} miss/token, {fr(g4_50,1)} ms de PCIe gen4 contre {fr(t_token_ms,1)} ms "
          f"de calcul ({fr(100*g4_50/t_token_ms,0)} %")
    print("        du budget) — recouvrable EN PRINCIPE, jamais mesuré.")
    print(f"      • la limite                   : α = {fr(alpha_lru,3)} ⇒ {fr(v_lru,1)} Go, une carte 24 Go,")
    print("        mais le PCIe y devient le chemin critique. Une limite, pas un point de")
    print("        fonctionnement.")
    print("   ⇒ La variante VRAM ne descendait sous AUCUNE classe de carte ; le tier froid en")
    print("      RAM hôte fait tomber le 120b de « ≥ 64 Go » à « 32-48 Go », et la question")
    print("      redevient une question d'ingénierie de prefetch, pas une question de bits.")
    print("   ⚠️ Trois dettes que ce chiffrage n'éteint pas : (i) le préchargement suppose de")
    print("      connaître le routage AVANT la couche, ce qui exige un prefetch spéculatif ou")
    print("      un décalage d'une couche — non conçu, non mesuré ; (ii) 32-63 Go/s sont des")
    print("      hypothèses externes, aucune mesure PCIe n'existe ici ; (iii) le trafic PCIe")
    print("      d'un batch > 1 ne se partage pas : ces chiffres sont batch 1.")

    # -- 8. Scope reserve --------------------------------------------------
    print("\n\n8. RÉSERVE DE PÉRIMÈTRE — inscrite d'avance au pré-enregistrement (§3-T4)")
    print("=" * 78)
    print(f"   Le dump est un 20b à {E} experts top-{K} ({fr(100*K/E,1)} % actifs, {L} couches).")
    print(f"   Le dimensionnement vise un 120b à {P120_EXPERTS} experts top-{P120_TOPK} "
          f"({fr(100*P120_TOPK/P120_EXPERTS,1)} % actifs, {n_layers_120} couches).")
    print("   ⇒ UN VERT NE SE TRANSPORTERAIT PAS. Un ROUGE, si : la concentration ne peut")
    print("     qu'empirer en montant en nombre d'experts (plus d'experts pour la même masse")
    print("     de tokens ⇒ plus de cellules faibles), et le nombre de visites par token monte")
    print(f"     de {visits} à {P120_TOPK*n_layers_120}, ce qui DIVISE encore la masse froide tolérable")
    print(f"     par visite (×{fr(P120_TOPK*n_layers_120/visits,1)}).")
    print("   Le verdict de ce test est ROUGE : il se transporte.")
    print()
    print("   Ce que ce test NE tranche PAS : aucune milliseconde n'a été mesurée ; le dump")
    print("   n'a pas de dimension temporelle, donc aucun cache dynamique n'est jugé ; le")
    print("   routage d'un 120b n'a jamais été capturé ; et la qualité 2 bits sur un MoE reste")
    print("   sans aucun chiffre, chez nous comme ailleurs (gate X5-MoE, ~25-55 $).")


if __name__ == "__main__":
    main()
