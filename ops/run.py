# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface-hub>=1.26"]
# ///
"""Launch LLVQ quantization runs on Hugging Face Jobs.

Why this lives outside the Rust workspace: the crates are auditable and
dependency-light on purpose, and orchestration is neither. Python is also the
native API for Jobs, so the alternative would be shelling out to `hf jobs` and
parsing its output.

## The one thing this file exists to prevent

Launching a job whose cost you have not estimated. The `estimate` command is
not decoration: it is checked against the real Qwen3-4B run (`selftest`), and
`launch` refuses to start above a cost ceiling without an explicit `--yes`.

## What the estimate is built on

End-to-end constants from four real jobs, at 0.6B, 8B and 32B. Not a model of
the parts, which was wrong three times:

* the first version assumed the Cholesky dominated. `bin/cholbench` says it is
  1.5 % of a 4B run since `faer`;
* the second assumed the Leech encoder was the whole cost and concluded a GPU
  flavor pays for an idle accelerator. **Wrong**: the GPU run is ~4× cheaper
  in core-seconds, because the forward passes fall from 88 % to 1–2 %;
* the third assumed the cost per weight was width-independent. **Wrong too**:
  the n³ factorization is 1.6 % of a run at 0.6B, 5.5 % at 8B and 16.5 % at
  32B, and extrapolating the 8B constant undershot the 32B by 25 %.

What survives all three: the encoder is CPU-bound and dominates. The
accelerator's job is to get the forward passes out of the way; after that only
vCPU count and encoder speed buy anything — which is why a two-card flavor is
rented for its 46 cores and leaves the second GPU idle.

**De-risk before committing.** Four blocks of a 32B cost $5.43 and corrected a
25 % underestimate on a run that would have cost $62.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

# --- hardware ---------------------------------------------------------------
#
# From https://huggingface.co/docs/hub/en/jobs-pricing (checked 2026-08-02).
# `hf jobs hardware` prints the live table; refresh this one if it drifts.
#
# `usd_per_core_hour` is the number that actually decides, because the run is
# CPU-bound. Sorted worst to best on that metric in the comments below.
#
# ⚠️ `cap` is the CUDA compute capability, and it is a **hard filter, not a
# preference**. `ops/Dockerfile.cuda` pins `CUDA_COMPUTE_CAP=89` because the
# Hub's Space builder has no GPU, and `candle-kernels` therefore ships PTX.
# PTX is forward-compatible only: the driver can JIT it for any sm ≥ 89, and
# for nothing below. A flavor whose `cap` is under 89 will start, bill, pull
# the image, download the checkpoint — and then fail to load a single kernel.
#
# `a100-large` is the trap: 80 GB of VRAM at a fair price is exactly what one
# reaches for, and it is sm_80. Marked here so the choice is refused rather
# than regretted.
MIN_COMPUTE_CAP = 89

FLAVORS: dict[str, dict] = {
    #                       vCPU  RAM Go  VRAM Go  $/h        compute cap
    "cpu-upgrade":     dict(vcpu=8,   ram=32,   vram=0,   usd_h=0.03, cap=None),
    "cpu-xl":          dict(vcpu=16,  ram=124,  vram=0,   usd_h=1.00, cap=None),
    "cpu-performance": dict(vcpu=32,  ram=256,  vram=0,   usd_h=1.90, cap=None),
    "t4-medium":       dict(vcpu=8,   ram=30,   vram=16,  usd_h=0.60, cap=75),
    "l4x1":            dict(vcpu=8,   ram=30,   vram=24,  usd_h=0.80, cap=89),
    # Multi-card flavors are listed for their **vCPU**, not their VRAM: candle
    # drives one device, so the extra cards sit idle. What they buy is host
    # cores at the same or better price per core-hour, and the run is 90 %
    # CPU-side encoder. `rtx-pro-6000x2` costs exactly what the x1 costs for
    # the same job, in half the wall clock.
    "l4x4":            dict(vcpu=48,  ram=186,  vram=24,  usd_h=3.80, cap=89),
    "l40sx1":          dict(vcpu=8,   ram=62,   vram=48,  usd_h=1.80, cap=89),
    "a100-large":      dict(vcpu=12,  ram=142,  vram=80,  usd_h=2.50, cap=80),
    "rtx-pro-6000":    dict(vcpu=23,  ram=256,  vram=96,  usd_h=2.75, cap=120),
    "rtx-pro-6000x2":  dict(vcpu=46,  ram=512,  vram=96,  usd_h=5.50, cap=120),
    "h200":            dict(vcpu=23,  ram=256,  vram=141, usd_h=5.00, cap=90),
}


def cap_ok(flavor: str) -> tuple[bool, str]:
    """`(usable, why)` for the image this repo builds.

    CPU flavors are always fine — they never load a CUDA kernel.
    """
    cap = FLAVORS.get(flavor, {}).get("cap")
    if cap is None:
        return True, "CPU"
    if cap < MIN_COMPUTE_CAP:
        return False, f"sm_{cap} < sm_{MIN_COMPUTE_CAP} — l'image ne peut y charger aucun noyau"
    if cap > MIN_COMPUTE_CAP:
        return True, f"sm_{cap}, par JIT PTX depuis sm_{MIN_COMPUTE_CAP}"
    return True, f"sm_{cap} natif"


def device_ok(flavor: str, device: str) -> tuple[bool, str]:
    """`(coherent, why)` for a `--flavor` / `--device` pair, before it is billed.

    Nothing else crosses these two, and they are the pair the whole estimate
    rests on. `cost_table` charges a flavor with VRAM the `cuda` rate
    (6.36e-5 core-s/weight) **because** the forward passes fall to 1–2 % of the
    run there; ask that same flavor for `--device cpu` and the forwards come
    back at 88 %, i.e. the `cpu` rate (2.41e-4). The estimate is then 3.8× low
    on a machine billed by the minute.

    What turns that overspend into a total loss is the `timeout`: `cmd_launch`
    sets it to 1.5× the estimate, so a run 3.8× slower than estimated is killed
    at ~40 % of its blocks — every minute billed, and no artifact.

    The reverse, `--device cuda` on a CPU flavor, is the cheap half but still a
    paid failure: since `eval.rs::device` stopped falling back to CPU it fails
    loudly, and it does so *after* the job has started, pulled the image and
    downloaded the checkpoint (65 GB on a 32B, ~26 min without code item C5).

    ⚠️ Deliberately **not** overridable by `--yes`. That flag means "I accept
    the bill" and the README hands it to every run above the ceiling — which is
    exactly the 32B, the one run where a 3.8× miss costs the most. A guard that
    switches itself off on the expensive run is not a guard.

    Pure on purpose: it is the only launch check that can be exercised without
    a network, a token, or a billed minute.
    """
    spec = FLAVORS.get(flavor)
    if spec is None:
        # Not this function's refusal — `cmd_launch` already rejects an unknown
        # flavor by name, and inventing a second message for it would send the
        # operator looking for a device problem they do not have.
        return True, ""
    gpu = spec["vram"] > 0
    if device == "metal":
        return False, (
            f"--device metal sur {flavor} : aucune flavor HF Jobs n'est un Mac, et\n"
            f"l'image de ce dépôt n'est pas construite avec `--features metal`.\n"
            f"Le job démarrerait, serait facturé, téléchargerait le checkpoint, puis\n"
            f"`eval.rs::device` échouerait au premier appel.\n"
            f"Relance avec --device {'cuda' if gpu else 'cpu'}.")
    if gpu and device != "cuda":
        return False, (
            f"--device {device} sur {flavor}, qui a {spec['vram']} Go de VRAM.\n"
            f"Le devis facture à cette flavor le tarif `cuda` "
            f"({QUANT_CORE_SEC_PER_WEIGHT['cuda']:.2e} cœur-s/poids) parce que les passes\n"
            f"avant y tombent à 1–2 % du run. En `{device}` elles reviennent à 88 %, soit\n"
            f"{QUANT_CORE_SEC_PER_WEIGHT['cpu']:.2e} : ×"
            + f"{QUANT_CORE_SEC_PER_WEIGHT['cpu'] / QUANT_CORE_SEC_PER_WEIGHT['cuda']:.1f}".replace(".", ",")
            + " le devis. Et le `timeout` est posé à 1,5× ce devis,\n"
            f"donc le job serait tué vers 40 % des blocs — facturé plein, sans artefact.\n"
            f"Relance avec --device cuda, ou prends une flavor sans VRAM.")
    if not gpu and device != "cpu":
        return False, (
            f"--device {device} sur {flavor}, qui n'a aucun GPU.\n"
            f"`eval.rs::device` ne retombe plus sur le CPU en silence : il échoue en\n"
            f"nommant la feature manquante — mais après démarrage, pull de l'image et\n"
            f"téléchargement du checkpoint, tous facturés.\n"
            f"Relance avec --device cpu, ou prends une flavor avec de la VRAM.")
    return True, f"{device} sur {flavor} ({spec['vram']} Go de VRAM)"


# Measured, see the module docstring. Change these only with a bench to back it.
BLOCKS_PER_SEC_PER_CORE = 1469.0
CHOLESKY_MACS_PER_SEC = 110e9
DIM = 24

# Core-seconds per quantized weight, measured end to end on real jobs:
#
#   0.6B  cpu-upgrade      8 vCPU  cpu    47 185 920 poids en  1421 s → 2.41e-4
#   0.6B  l4x1             8 vCPU  cuda   47 185 920 poids en   371 s → 6.29e-5
#   8B    rtx-pro-6000    23 vCPU  cuda 6 925 713 408 poids en 14356 s → 4.77e-5
#   32B   rtx-pro-6000x2  46 vCPU  cuda 1 947 893 760 poids en  2694 s → 6.36e-5
#
# The GPU is ~4× cheaper *in core-seconds*: the forward passes fall from 88 %
# of the work to 1–2 %. What is left is the encoder, which is CPU-bound either
# way — so on a GPU flavor only vCPU count and encoder speed buy anything.
#
# ⚠️ The `cuda` figure is **not** width-independent. It dips at 8B and climbs
# again at 32B, because the n³ factorization goes from 1.6 % of a run (0.6B) to
# 5.5 % (8B) to 16.5 % (32B). The **largest** constant is used so that a
# projection errs high rather than low: extrapolating the 8B number to the 32B
# undershot by 25 %, which a 5.43 $ de-risking run caught before a 62 $ commit.
QUANT_CORE_SEC_PER_WEIGHT = {"cpu": 2.41e-4, "cuda": 6.36e-5}

# The published Qwen3-4B rate. Used to size the artifact, nothing else.
BITS_PER_WEIGHT = 2.1696

# What one already-quantized weight costs to bring back when a run resumes:
# read the shard's record, decode its Leech indices, copy the record into the
# new artifact, and hand the reconstruction to the model. Measured on
# Qwen3-0.6B, single-threaded, on the run that resumed at block 14 of 28 —
# 220 200 960 weights in 8.1 s, i.e. 27.2 M/s. (A 2-block shard gave 26.2 M/s,
# so the rate does not depend on how much is being reloaded.)
#
# 🔎 On that run the whole segmentation was **free within the noise**: 1747 s
# for one 28-block run against 1730 s for the two segments end to end, wall
# clock, same machine. The reload is 8 s of it; what the estimate below cannot
# see — a second checkpoint load, a second baseline perplexity, and the forward
# passes replayed over the inherited blocks — was under 20 s at this size. It
# will not stay that small: at 32B the replayed forwards run on 131 k
# calibration tokens instead of 2 048, and the reload is 488 M weights per
# block instead of 15.7 M.
#
# ⚠️ **This is the only new constant here, and it is an extrapolation.** It was
# measured on a 0.6B on an M3 Max, and the repo has already been taught what
# extrapolating a width costs: the 32B de-risking pass came in 25 % over the
# projection from the 8B. It is used below for *one* purpose — telling an
# operator roughly what an extra segmentation point costs — and never as a
# billing guard.
#
# What it deliberately does **not** cover, on the same doctrine as the rest of
# this estimator: the forward passes the resumed segment re-runs over the
# blocks it inherited. Dressing a guess up as an estimate is worse than
# omitting it, so the segmentation plan prints the decode term and says the
# forwards are missing.
RESUME_SEC_PER_WEIGHT = 3.8e-8


def fetch_config(repo: str) -> dict:
    """A model's `config.json`, straight from the Hub — no token, no download."""
    url = f"https://huggingface.co/{repo}/raw/main/config.json"
    with urllib.request.urlopen(url, timeout=30) as r:  # noqa: S310
        return json.load(r)


def weight_counts(cfg: dict) -> tuple[int, int]:
    """`(quantized, carried)` weights.

    The carried half is the trap. On Qwen3-4B the embedding is **tied**, so it
    counts once (389 M). On Qwen3-32B `tie_word_embeddings` is **false**, so
    `embed_tokens` and `lm_head` are two separate tensors — 1.556 G weights at
    16 bits, which is 4.7 % of the model but **27 % of the artifact**. Quoting
    the linear layers' bits/weight as the model's compression ratio is wrong by
    a large factor whenever this happens; see CLAUDE.md.
    """
    layers = cfg["num_hidden_layers"]
    hidden = cfg["hidden_size"]
    inter = cfg["intermediate_size"]
    head_dim = cfg["head_dim"]
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg["num_key_value_heads"]

    attn_out = head_dim * n_heads
    per_layer = (
        hidden * attn_out          # q_proj
        + hidden * head_dim * n_kv  # k_proj
        + hidden * head_dim * n_kv  # v_proj
        + attn_out * hidden        # o_proj
        + hidden * inter           # gate_proj
        + hidden * inter           # up_proj
        + inter * hidden           # down_proj
    )
    embed = cfg["vocab_size"] * hidden
    carried = embed if cfg.get("tie_word_embeddings", False) else 2 * embed
    return layers * per_layer, carried


def act_widths(cfg: dict) -> list[int]:
    """The four activation widths a block factorizes, one Cholesky each.

    Four and not seven: `q/k/v` consume the same tensor and `gate/up` too, so
    the factorization is per *activation*, reused by the matrices that share it.
    """
    return [
        cfg["hidden_size"],                                  # Attn   → q,k,v
        cfg["head_dim"] * cfg["num_attention_heads"],        # AttnOut→ o
        cfg["hidden_size"],                                  # Mlp    → gate,up
        cfg["intermediate_size"],                            # MlpOut → down
    ]


def estimate(cfg: dict, blocks: int | None = None) -> dict:
    """Core-hours and artifact size, per term."""
    layers = blocks if blocks is not None else cfg["num_hidden_layers"]
    frac = layers / cfg["num_hidden_layers"]
    all_quantized, carried = weight_counts(cfg)
    quantized = int(all_quantized * frac)

    leech_h = (quantized / DIM) / BLOCKS_PER_SEC_PER_CORE / 3600.0
    macs = sum(2.0 / 3.0 * n**3 for n in act_widths(cfg)) * layers
    chol_h = macs / CHOLESKY_MACS_PER_SEC / 3600.0

    artifact_gb = (quantized * BITS_PER_WEIGHT / 8 + carried * 2) / 1e9
    # **Every** weight, not the fraction being quantized. The whole model has
    # to be resident whatever `--blocks` says: calibration is sequential, so
    # the run still loads all 64 layers and forwards through the ones it is not
    # quantizing. Scaling this with `frac` made a 4-block 32B look like it fit
    # on a 16 GB card — it needs 65.5 GB in bf16, same as the full run.
    checkpoint_gb = (all_quantized + carried) * 2 / 1e9
    return dict(
        layers=layers,
        quantized=quantized,
        carried=carried,
        leech_core_h=leech_h,
        chol_core_h=chol_h,
        artifact_gb=artifact_gb,
        checkpoint_gb=checkpoint_gb,
    )


def cost_table(est: dict, dtype: str = "f32") -> list[tuple[str, float, float, str]]:
    """`(flavor, wall_hours, usd, warning)` for every flavor.

    Built on the end-to-end measured constants, not on a model of the parts:
    a GPU flavor is charged the `cuda` rate because its forward passes are
    free, a CPU flavor the `cpu` rate because they are not. Both numbers come
    from the same 3-block Qwen3-0.6B run.

    ⚠️ Extrapolating a 0.6B to a 32B assumes the cost stays linear in weight
    count. It will not, exactly — memory bandwidth and cache behave differently
    at 50× the size — so treat these as the right order of magnitude and let
    the 8B step correct them.
    """
    rows = []
    for name, f in FLAVORS.items():
        gpu = f["vram"] > 0
        rate = QUANT_CORE_SEC_PER_WEIGHT["cuda" if gpu else "cpu"]
        wall = est["quantized"] * rate / f["vcpu"] / 3600.0
        warn = ""
        # The model has to sit somewhere: VRAM on a GPU flavor, host RAM
        # otherwise. `smoke` loads in F32 today, which is 2× the bf16 figure —
        # that is what code item C3 would fix, and it is what decides whether
        # an 8B fits on the cheapest card.
        # `est["checkpoint_gb"]` is the bf16 size. Code item C3 landed, so the
        # run can hold the model at the checkpoint's own precision instead of
        # upcasting it: `LLVQ_DTYPE=bf16` halves this, and on a 32B that is the
        # difference between one 96 GB card and a two-card flavor.
        need = est["checkpoint_gb"] * (2 if dtype == "f32" else 1)
        if gpu and f["vram"] < need:
            warn = f"VRAM {f['vram']} Go < {need:.0f} Go en {dtype}"
        if not gpu and f["ram"] < need:
            warn = f"RAM {f['ram']} Go < {need:.0f} Go en {dtype}"
        rows.append((name, wall, wall * f["usd_h"], warn))
    return rows


# --- commands ---------------------------------------------------------------


def print_segment_plan(cfg: dict, est: dict, n: int, flavor: str) -> None:
    """What cutting a run into `n` segments costs, and what it buys.

    The question this answers is the one that has kept the 32B unlaunched: a
    single job is ~11.4 h with nothing to show for it if it dies at hour ten.
    Segmenting bounds the loss to one segment — the whole point of code item
    C4 — and the price is that every segment after the first re-reads what its
    predecessors wrote.

    ## The arithmetic, and why more segments get expensive faster than they look
    Segment `i` (0-based, equal shares) restarts at block `i·L/n` and therefore
    reloads `i·L/n` blocks. Summed over the `n-1` resumed segments that is
    `L·(n-1)/2` block-reloads — i.e. **half the model re-read per extra
    segmentation point**, on average. Doubling `n` roughly doubles the total
    reload, and it does *not* halve the exposure again in the same proportion:
    the worst-case loss falls as `1/n` while the overhead climbs as `n-1`.

    Two segments is where that trade is best, and it is a bound worth stating
    rather than a taste: it caps the loss at half a run for one reload of half
    a model.
    """
    layers = cfg["num_hidden_layers"]
    per_weight = est["quantized"] / max(layers, 1)
    vcpu = FLAVORS[flavor]["vcpu"]
    gpu = FLAVORS[flavor]["vram"] > 0
    rate = QUANT_CORE_SEC_PER_WEIGHT["cuda" if gpu else "cpu"]
    usd_h = FLAVORS[flavor]["usd_h"]

    cuts = [round(i * layers / n) for i in range(n + 1)]
    print(f"\n  découpage en {n} segments sur {flavor}")
    print(f"  {'segment':<9}{'blocs':<12}{'quantifie':>12}{'recharge':>12}"
          f"{'h':>8}{'$':>8}")
    print("  " + "-" * 62)
    total_h = 0.0
    reload_h = 0.0
    for i in range(n):
        lo, hi = cuts[i], cuts[i + 1]
        quant = (hi - lo) * per_weight
        back = lo * per_weight
        h_quant = quant * rate / vcpu / 3600.0
        h_back = back * RESUME_SEC_PER_WEIGHT / 3600.0
        total_h += h_quant + h_back
        reload_h += h_back
        print(f"  {i + 1:<9}{f'{lo}..{hi - 1}':<12}{quant / 1e9:>10.2f} Md"
              f"{back / 1e9:>10.2f} Md{h_quant + h_back:>8.1f}{(h_quant + h_back) * usd_h:>8.2f}")
    base_h = est["quantized"] * rate / vcpu / 3600.0
    longest = max(
        (cuts[i + 1] - cuts[i]) * per_weight * rate / vcpu / 3600.0
        + cuts[i] * per_weight * RESUME_SEC_PER_WEIGHT / 3600.0
        for i in range(n)
    )
    print(f"\n  job le plus long   {longest:8.1f} h   (contre {base_h:.1f} h d'un seul tenant)")
    print(f"  surcoût total      {reload_h:8.1f} h   soit +{100 * reload_h / max(base_h, 1e-9):.1f} %,"
          f" ~{reload_h * usd_h:.2f} $")
    print(f"  perte maximale     {longest:8.1f} h   au lieu de {base_h:.1f} h"
          f"  ← ce que le découpage achète")
    print("\n  ⚠️  Le terme « recharge » ne compte que le DÉCODAGE du shard (constante")
    print("      mesurée sur un 0,6B, mono-thread). Il ne modélise ni le rechargement")
    print("      du checkpoint, ni la perplexité de référence que chaque segment")
    print("      remesure, ni les passes avant que le segment repris rejoue sur les")
    print("      blocs hérités — comme le reste de ce devis, qui ne modélise pas les")
    print("      passes avant. Le surcoût réel est donc PLUS GRAND que celui affiché.")


def cmd_estimate(args) -> int:
    cfg = fetch_config(args.model)
    est = estimate(cfg, args.blocks)
    print(f"\n{args.model} — {est['layers']} couches")
    print(f"  poids quantifiés  {est['quantized'] / 1e9:8.2f} Md")
    print(f"  poids portés 16 b {est['carried'] / 1e9:8.2f} Md"
          f"   ({100 * est['carried'] / (est['quantized'] + est['carried']):.1f} %"
          f" des poids, {100 * est['carried'] * 2 / 1e9 / est['artifact_gb']:.0f} %"
          f" de l'artefact)")
    print(f"  checkpoint bf16   {est['checkpoint_gb']:8.1f} Go")
    print(f"  artefact projeté  {est['artifact_gb']:8.1f} Go"
          f"   ×{est['checkpoint_gb'] / est['artifact_gb']:.1f}")
    print(f"\n  encodage Leech    {est['leech_core_h']:8.1f} cœur-h")
    print(f"  Cholesky          {est['chol_core_h']:8.1f} cœur-h"
          f"   ({100 * est['chol_core_h'] / (est['leech_core_h'] + est['chol_core_h']):.1f} %)")
    print(f"\n  {'flavor':<18}{'h (CPU)':>10}{'$':>9}   remarque")
    print("  " + "-" * 72)
    for name, wall, usd, warn in sorted(cost_table(est, args.dtype), key=lambda r: r[2]):
        usable, why = cap_ok(name)
        mark = "" if usable else "  ⛔ "
        print(f"  {name:<18}{wall:>10.1f}{usd:>9.2f}   {mark}{warn if usable else why}")
    print("\n  Constantes mesurées de bout en bout sur un run 0,6B (CPU et CUDA). Les")
    print("  flavors GPU paient le tarif cuda, les autres le tarif cpu.")
    print("  ⛔ = l'image de ce dépôt n'y charge aucun noyau CUDA.")
    if args.segments and args.segments > 1:
        print_segment_plan(cfg, est, args.segments, args.flavor)
    print("\n  ⚠️  Ce devis modélise une QUANTIFICATION, et rien d'autre : il multiplie")
    print("      un nombre de poids par un coût mesuré de l'encodeur Leech et de la")
    print("      factorisation. Un job de MESURE (ppl, mmlu, oracle) n'exécute ni")
    print("      l'un ni l'autre — sur Qwen3-4B ce devis annonce ~8 h pour ce qui en")
    print("      prend 30 à 90 min. Ne pas s'en servir pour la campagne de mesure, ni")
    print("      de --max-usd comme garde : le plafond utile y est le `timeout` du job,")
    print("      qui est exact et connu avant lancement.")
    return 0


def cmd_selftest(args) -> int:
    """Check the estimator against the run that produced the published 16.94.

    An estimator nobody has confronted with a real run is a spreadsheet. This
    one is pinned to `~/llvq-run-4b-artefact.log`: 3 633 315 840 weights, and
    14 447 s of wall clock on 12 M3 Max performance cores with Metal.
    """
    cfg = fetch_config("Qwen/Qwen3-4B")
    quantized, carried = weight_counts(cfg)
    ok = True

    if quantized != 3_633_315_840:
        print(f"FAIL  poids quantifiés {quantized}, le run en rapporte 3 633 315 840")
        ok = False
    else:
        print("ok    poids quantifiés = 3 633 315 840, au poids près")

    if carried != 151_936 * 2_560:
        print(f"FAIL  embedding {carried}, attendu {151_936 * 2_560} (liée)")
        ok = False
    else:
        print("ok    embedding liée comptée une fois")

    est = estimate(cfg)
    leech_wall = est["leech_core_h"] / 12 * 3600
    chol_wall = est["chol_core_h"] / 12 * 3600
    print(f"\n      Leech    {leech_wall:7.0f} s sur 12 cœurs")
    print(f"      Cholesky {chol_wall:7.0f} s sur 12 cœurs")
    print(f"      mesuré   {14447:7d} s au total")
    share = 100 * leech_wall / 14447
    print(f"\n      l'encodage expliquerait {share:.0f} % du run mesuré")
    if not 40 <= share <= 80:
        print("FAIL  hors de la fourchette plausible — la constante a dérivé")
        ok = False
    return 0 if ok else 1


def cmd_launch(args) -> int:
    from huggingface_hub import Volume, run_job

    cfg = fetch_config(args.model)
    est = estimate(cfg, args.blocks)
    rows = {name: (wall, usd) for name, wall, usd, _ in cost_table(est, args.dtype or "f32")}
    if args.flavor not in rows:
        print(f"flavor inconnu: {args.flavor}", file=sys.stderr)
        return 2
    usable, why = cap_ok(args.flavor)
    if not usable:
        print(f"refus : {args.flavor} — {why}.\n"
              f"L'image fige CUDA_COMPUTE_CAP={MIN_COMPUTE_CAP} (le builder d'un Space n'a pas\n"
              f"de GPU, donc candle-kernels n'embarque que du PTX, compatible vers l'avant\n"
              f"seulement). Le job démarrerait, serait facturé, téléchargerait le checkpoint,\n"
              f"puis échouerait à charger le premier noyau.", file=sys.stderr)
        return 2
    coherent, why = device_ok(args.flavor, args.device)
    if not coherent:
        print(f"refus : {why}", file=sys.stderr)
        return 2
    wall, usd = rows[args.flavor]

    print(f"{args.model} sur {args.flavor} — {wall:.1f} h estimées, ~{usd:.2f} $")
    print("  (hors passes avant, cf. `estimate`)")
    if usd > args.max_usd and not args.yes:
        # Flush first: the estimate above is on stdout and the refusal below on
        # stderr, and unflushed they arrive in the wrong order — which reads as
        # a refusal without a reason.
        sys.stdout.flush()
        print(f"\nrefus : {usd:.2f} $ dépasse le plafond de {args.max_usd:.2f} $."
              f"\nRelance avec --yes, ou --max-usd plus haut.", file=sys.stderr)
        return 1

    # `--mount-model` works as of 2026-08-08: `Checkpoint::fetch` now decides
    # local-directory versus Hub repo **syntactically** (a leading `/`, `./`,
    # `../` or `~/` means a directory), so the `/model` this passes resolves to
    # the mounted volume instead of being tried as a repo id. Code item C5,
    # closed with no Python change.
    #
    # Still off by default: the volume has to exist and hold a full checkpoint,
    # and a mount that silently lacks a shard is a worse failure than a
    # download. Opt in when you know the volume is there.
    env = {
        "LLVQ_MODEL": args.model_mount if args.mount_model else args.model,
        "LLVQ_CALIB": args.calib,
        "LLVQ_THREADS": str(FLAVORS[args.flavor]["vcpu"]),
    }
    if args.bucket:
        env["LLVQ_ARTIFACT"] = f"{args.out_mount}/{args.name}.llvq"
    else:
        # No bucket: the artifact still gets written and round-trip verified
        # in-process, it just dies with the container. Fine for a plumbing
        # test, useless for a real run.
        env["LLVQ_ARTIFACT"] = f"/scratch/{args.name}.llvq"
        print("  ⚠️  sans --bucket, l'artefact ne survit pas au conteneur")
    for var, val in (("LLVQ_DAMPING", args.damping),
                     ("LLVQ_CALIB_SEED", args.calib_seed),
                     ("LLVQ_DTYPE", args.dtype)):
        if val is not None:
            env[var] = str(val)
    # Code item C4. The block this restarts at is **not** passed: `smoke` reads
    # it off the shard, so there is one source of truth and no way to hand the
    # job a start that disagrees with the file. What is passed is the shard,
    # and the guarantee is upstream of this script — `resume_from_shard`
    # refuses a shard whose codebook, dimensions, rotation or record order are
    # not this run's, and the `.state` file beside it carries the rest of the
    # configuration (calibration, damping, dtype, device) so that a hybrid
    # artifact cannot be assembled by accident.
    if args.resume:
        if not args.bucket:
            print("refus : --resume sans --bucket. Le shard vit sur le volume de "
                  "sortie ;\nsans lui il n'y a rien à reprendre et rien où écrire.",
                  file=sys.stderr)
            return 2
        env["LLVQ_RESUME"] = args.resume
        if args.resume == env["LLVQ_ARTIFACT"]:
            print(f"refus : --resume et la sortie désignent {args.resume}.\n"
                  f"La sortie est ouverte en création : le shard serait tronqué "
                  f"avant d'être lu.\nDonne un --name différent de celui du segment "
                  f"précédent.", file=sys.stderr)
            return 2
        print(f"  reprise depuis {args.resume}"
              f" → {env['LLVQ_ARTIFACT']}")
        print("  ⚠️  le devis ci-dessus chiffre TOUS les blocs, pas le reste à faire :")
        print("      il ne sait pas où s'arrête le shard, qui n'est lisible que dans")
        print("      le conteneur. Il majore donc, ce qui est le bon sens d'erreur")
        print("      pour un plafond de coût et pour un timeout.")

    command = [
        "smoke",
        str(args.calib_windows), str(args.calib_len),
        str(args.eval_windows), str(args.eval_ctx),
        args.device,
        "gs" if args.group_scales else "nogs",
        args.codebook,
        str(args.blocks if args.blocks else 1_000_000),
        "rot" if args.rotation else "norot",
    ]

    volumes = []
    if args.mount_model:
        volumes.append(Volume(type="model", source=args.model,
                              mount_path=args.model_mount, read_only=True))
    if args.bucket == "auto":
        # `sync_job_volume` creates the `jobs-artifacts` bucket on demand and
        # returns a mountable, writable volume. Worth the two lines: the 8B
        # artifact costs a full run to regenerate, so letting it die with the
        # container would be paying twice for the same file.
        from huggingface_hub import sync_job_volume
        out = Path(args.root_out) / args.name
        out.mkdir(parents=True, exist_ok=True)
        vol = sync_job_volume(str(out), args.out_mount, read_only=False)
        volumes.append(vol)
        print(f"  bucket : hf://buckets/{vol.source}/{vol.path} → {args.out_mount}")
    elif args.bucket:
        volumes.append(Volume(type="bucket", source=args.bucket,
                              mount_path=args.out_mount, read_only=False))

    job = run_job(
        image=args.image,
        command=command,
        flavor=args.flavor,
        env=env,
        volumes=volumes,
        # The default is 30 minutes, which would kill every real run. Padded
        # 50 % over the estimate — and the estimate excludes the forwards.
        timeout=f"{max(1, int(wall * 1.5))}h",
        name=args.name,
        namespace=args.namespace,
    )
    print(f"\nlancé : {job.url}\n  id {job.id}")
    print(f"  suivi : uv run ops/run.py watch {job.id}")
    return 0


SPACE_CARD = """---
title: LLVQ runner
emoji: 🧊
colorFrom: blue
colorTo: gray
sdk: docker
pinned: false
---

# LLVQ runner

**Ce Space ne sert rien.** C'est un *constructeur d'image* : Hugging Face
compile ici les binaires Rust de la quantification LLVQ, et
[HF Jobs](https://huggingface.co/docs/hub/jobs-overview) réutilise l'image
produite pour lancer les runs sur du matériel adapté.

Passer par un Space évite de pousser plusieurs gigaoctets d'image depuis un
poste de dev : c'est HF qui construit, à partir des sources.

Binaires : `smoke` (quantification), `ppl`, `mmlu`, `run`, `seal`, `oracle`.

Lancé depuis `ops/run.py` du dépôt LLVQ.
"""

# What `publish` uploads, as one pair of lists used **twice**: to filter the
# upload, and to decide whether the tree is clean where it matters. Two copies
# of this would be the exact bug the guard below exists to prevent — a
# perimeter that drifts from the payload checks the wrong files, and a
# provenance check that checks the wrong files is worse than none, because it
# reassures.
#
# Allow-list, not deny-list: `COPY . .` copies whatever ends up in the Space,
# so a forgotten exclusion means a slower build, and `target/` alone is tens of
# gigabytes.
#
# `Cargo.lock` is in the list on purpose. Without it the builder resolves
# dependencies fresh, so the image would not be built from the tree that was
# committed — and every number the image produces would be traceable to a
# commit that does not describe it. That is exactly the provenance gap this
# campaign exists to close.
UPLOAD_ALLOW = ("Cargo.toml", "Cargo.lock", "llvq-*/**")
UPLOAD_IGNORE = ("**/target/**", "**/*.log")


def dirty_in_upload_perimeter(dirty_files, recipe: str) -> list[str]:
    """The modified paths that would actually change the image being built.

    `manifest.git_state` looks at the **whole** tree, and applying that verdict
    verbatim here would make `publish` refuse whenever `docs/` moves — that is,
    in this repository's ordinary working state. A guard that fires on every
    invocation is a guard the operator learns to bypass, after which it
    protects nothing. So the question is narrowed to the only one that has an
    answer: *would this change the bytes the builder compiles?*

    In scope: the allow list above, minus its ignores, plus the Dockerfile
    actually uploaded. The recipe is not a detail — `--cuda` selects it and it
    pins `CUDA_COMPUTE_CAP=89`, i.e. which cards the image can load a kernel
    on at all.

    Deliberately out of scope: `ops/run.py` itself. It picks the command a Job
    runs, but it is never compiled into the image, and a launcher edit must not
    block a rebuild of unchanged sources.

    ⚠️ `git status --porcelain` quotes paths holding non-ASCII bytes and spells
    a rename `old -> new`. Both make a path *fail* to match a pattern here, so
    the residual risk is under-reporting on exotic filenames; every crate path
    in this workspace is plain ASCII.
    """
    from fnmatch import fnmatchcase          # case-sensitive, like git

    blocking = []
    for path in dirty_files:
        if path == recipe:
            blocking.append(path)
        elif (any(fnmatchcase(path, p) for p in UPLOAD_ALLOW)
              and not any(fnmatchcase(path, p) for p in UPLOAD_IGNORE)):
            blocking.append(path)
    return blocking


def cmd_publish(args) -> int:
    """Publish the workspace as a Docker Space, which HF then builds for us.

    A Space is used here purely as a build service: pushing a multi-gigabyte
    image from a laptop is slow, and HF will build from source for free.
    `run_job(image="hf.co/spaces/<user>/<name>")` then reuses the result.

    **The Space itself will settle into `RUNTIME_ERROR`, and that is normal.**
    Its own container runs on CPU hardware with no GPU, so a CUDA runtime
    image has nothing to attach to and the app exits. Only the *build* matters
    here, and a built image stays usable by `run_job` regardless — verified on
    2026-08-05, when `llvq-preflight` completed on `l40sx1` against a Space
    sitting in `RUNTIME_ERROR`. Wait for `APP_STARTING` or `RUNNING`, which is
    what says the build passed; do not wait for the app to stay up, and do not
    read `RUNTIME_ERROR` as a failed build. A real build failure shows as
    `BUILD_ERROR`.

    Private by default. This is someone's research repository, and making it
    public is a decision they take, not one a script takes for them.

    ## And it uploads a commit, or refuses

    `ops/README.md` and this module both promise that "un chiffre qu'elle
    produit est rattachable à un commit" — on the strength of `--locked` alone,
    which pins the *dependencies* and says nothing about the workspace itself.
    Nothing checked that the workspace being uploaded was committed at all, so
    the promise held only by habit. It is checked now, over the perimeter that
    is actually uploaded (see `dirty_in_upload_perimeter`), and the commit is
    shipped alongside the sources as `COMMIT` so the claim is verifiable rather
    than asserted.
    """
    from huggingface_hub import create_repo, upload_file, upload_folder

    # `ops/manifest.py` already asks git this exact question and already
    # refuses a dirty tree; reusing it keeps one definition of "clean" for the
    # whole `ops/` directory. (Sibling module: `uv run ops/run.py` puts `ops/`
    # on `sys.path[0]`.)
    from manifest import ManifestError, git_state, now_utc

    # A Space builds its Dockerfile as written, so the CPU/CUDA choice is made
    # by *which file* we upload — `--build-arg` never reaches the Hub builder.
    # Resolved here rather than at the upload below because the recipe is part
    # of what has to be committed.
    recipe = "Dockerfile.cuda" if args.cuda else "Dockerfile"
    try:
        git_st = git_state(args.root)
    except ManifestError as exc:
        # No git, no provenance — and `--allow-dirty` cannot rescue this one:
        # there is no revision to write into COMMIT, so publishing would ship
        # sources nobody can situate. A refusal the operator can act on, rather
        # than the traceback `git_out` would otherwise raise here.
        print(f"refus : git est muet sur {args.root} — {exc}\n"
              "Sans état git, `publish` ne peut rattacher l'image à aucun commit, et\n"
              "c'est toute la promesse de ops/README.md.", file=sys.stderr)
        return 1
    blocking = dirty_in_upload_perimeter(git_st.dirty_files, f"ops/{recipe}")
    if blocking and not args.allow_dirty:
        listing = "\n".join(f"    {f}" for f in blocking[:20])
        more = f"\n    … et {len(blocking) - 20} autres" if len(blocking) > 20 else ""
        print(f"refus : {len(blocking)} fichier(s) non commité(s) dans ce que "
              f"`publish` téléverse :\n{listing}{more}\n"
              "Le Space compile l'image depuis ces octets-là, et les deux recettes bâtissent\n"
              "avec `--locked` précisément pour qu'un chiffre produit par l'image soit\n"
              "rattachable à un commit. Téléverser un arbre sale rompt la promesse en\n"
              "silence : l'image existe, elle tourne, et le commit qu'elle annonce ne la\n"
              "décrit pas — c'est la panne de provenance que `ops/manifest.py` existe pour\n"
              "empêcher côté mesures.\n"
              "Le contrôle ne regarde QUE le périmètre téléversé "
              f"({', '.join(UPLOAD_ALLOW)}, ops/{recipe}) :\n"
              "`docs/` et `ops/run.py` peuvent bouger librement.\n"
              "Commite, ou `--allow-dirty` — et le fichier COMMIT dira alors que l'arbre\n"
              "était sale, plutôt que de laisser croire le contraire.", file=sys.stderr)
        return 1

    repo_id = args.space
    create_repo(repo_id, repo_type="space", space_sdk="docker",
                private=not args.public, exist_ok=True)
    print(f"space {repo_id} ({'public' if args.public else 'privé'})")
    print(f"commit {git_st.commit[:12]}"
          + (f"  ⚠️ {len(blocking)} fichier(s) sale(s) dans le périmètre" if blocking
             else "  (périmètre téléversé propre)"))

    upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=str(args.root),
        allow_patterns=list(UPLOAD_ALLOW),
        ignore_patterns=list(UPLOAD_IGNORE),
        commit_message="LLVQ workspace",
    )
    # The commit these sources came from, shipped with them. Without it the
    # Space holds a copy of the workspace and no way to say which revision it
    # is: `git log` on the Space only dates the upload, and two publishes of
    # different commits look identical from the Hub side.
    #
    # It lands in the **build context** — `COPY . .` puts it at `/src/COMMIT` —
    # but not in the runtime layer, whose `COPY --from=build` names the
    # binaries one by one. Carrying it into the running container is a one-line
    # addition to each recipe, and would let a Job print the revision it is.
    #
    # ⚠️ Under `--allow-dirty` this file says so, in as many words. A COMMIT
    # naming a revision the uploaded bytes do not match would be worse than no
    # COMMIT at all — same rule as `manifest.py`, where a dirty entry is
    # recorded, marked, and refused by `verify`.
    lines = [f"{git_st.commit}\n",
             "# Commit du dépôt LLVQ dont proviennent Cargo.lock et llvq-*/**.\n",
             f"# recette : ops/{recipe}\n",
             f"# téléversé : {now_utc()}\n"]
    if blocking:
        lines.append(f"# ⚠️ ARBRE SALE dans le périmètre téléversé — "
                     f"{len(blocking)} fichier(s), image NON rattachable à ce commit :\n")
        lines += [f"#     {f}\n" for f in blocking[:20]]
    else:
        lines.append("# Périmètre téléversé propre au moment du téléversement.\n")
    commit_note = "".join(lines)
    upload_file(
        path_or_fileobj=commit_note.encode(),
        path_in_repo="COMMIT",
        repo_id=repo_id, repo_type="space",
        commit_message=f"commit {git_st.commit[:12]}"
                       + (" (ARBRE SALE)" if blocking else ""),
    )
    upload_file(
        path_or_fileobj=str(args.root / "ops" / recipe),
        path_in_repo="Dockerfile",
        repo_id=repo_id, repo_type="space",
        commit_message=f"build recipe ({recipe})",
    )
    upload_file(
        path_or_fileobj=SPACE_CARD.encode(),
        path_in_repo="README.md",
        repo_id=repo_id, repo_type="space",
        commit_message="space card",
    )
    print(f"\nimage : hf.co/spaces/{repo_id}")
    print(f"build : https://huggingface.co/spaces/{repo_id}  (suivre les logs)")
    print("\nAttendre que le build passe en RUNNING avant de lancer un Job.")
    return 0


def cmd_oracle(args) -> int:
    """Run `bin/oracle` on a rented device — the gate before paying for more.

    The hand-written forward pass exists to expose linear-layer inputs, and
    every Hessian is built from it. If it diverges from `candle-transformers`'
    own Qwen3 on this backend, every number that follows is wrong and the
    failure would surface hours later as an unexplained perplexity.

    It is the cheapest job in this file — a 0.6B and 64 tokens — and it is the
    one that must never be skipped when the hardware changes.
    """
    from huggingface_hub import run_job

    # Same cross-check as `launch`, and it belongs here too: the whole point of
    # this command is to prove the forward pass on **the backend the next run
    # will use**, so a `(flavor, device)` pair that cannot run is not a cheap
    # mistake — it is a proof that was never taken while looking taken.
    coherent, why = device_ok(args.flavor, args.device)
    if not coherent:
        print(f"refus : {why}", file=sys.stderr)
        return 2

    job = run_job(
        image=args.image,
        command=["oracle", args.model, str(args.tokens), args.device],
        flavor=args.flavor,
        timeout="20m",
        name=f"oracle-{args.device}",
        namespace=args.namespace,
    )
    print(f"oracle sur {args.flavor}/{args.device} : {job.url}\n  id {job.id}")
    return 0


# Cards this repo will run a kernel bench on, and why the list is short.
#
# `cap_ok` only says a card *can load* the image's code; it accepts anything
# from sm_89 up. That is the wrong question for a bench. An `l4x1` runs at
# roughly 300 GB/s — **below** the M3 Max the Metal figure was taken on — so a
# ratio measured there would be structurally pessimistic and uninterpretable,
# and the money would be spent producing a number nobody could use. The
# `rtx-pro-6000` and `h200` are the opposite problem: faster, more expensive,
# and not the class of card the paper's readers rent.
BENCH_FLAVORS = ("l40sx1",)


def cmd_bench(args) -> int:
    """Run an arbitrary binary from the image on a rented card.

    The generic launcher this file did not have. `cmd_launch` hard-codes
    `smoke` and `cmd_oracle` hard-codes `oracle`, so anything else had to be
    smuggled in as a fake quantization — and the cost estimator that gates
    `launch` does not apply here at all: it multiplies a weight count by a
    Leech encoder cost, and on a measurement job it is wrong by a factor ~8.
    The useful ceiling is the job timeout, which is why it is mandatory and
    has no silent default.

    Two guards the other subcommands lack:

    * the card whitelist above, checked *before* anything is billed;
    * `set -euo pipefail` around the command. Without it a bench that dies
      leaves the rest of the line running and the job finishes COMPLETED with
      no result — which reads as a pass.
    """
    from huggingface_hub import run_job, Volume

    ok, why = cap_ok(args.flavor)
    if not ok:
        print(f"refus : {args.flavor} — {why}")
        return 2
    if args.flavor not in BENCH_FLAVORS and not args.any_flavor:
        print(
            f"refus : {args.flavor} n'est pas dans {BENCH_FLAVORS}.\n"
            "  Un rapport mesuré ailleurs n'est pas comparable au chiffre Metal —\n"
            "  voir docs/portage-noyau-cuda.md §4.11. Forcer avec --any-flavor,\n"
            "  et alors le dire dans tout chiffre publié."
        )
        return 2

    # A read-only model volume rather than a download inside the job: the
    # sealed 4B is 1.77 GB and a job is billed by the minute. `hf download`
    # would charge the transfer at the card's rate.
    volumes = []
    if args.mount_model:
        volumes.append(Volume(type="model", source=args.mount_model,
                              mount_path=args.model_mount, read_only=True))
        print(f"montage {args.mount_model} en lecture seule sur {args.model_mount}")

    # 🕳️ Un volume de SORTIE, que cette sous-commande n'avait pas — et son
    # absence était une facture différée, pas un manque cosmétique.
    #
    # `cmd_launch` en a un depuis toujours parce qu'il produit un artefact.
    # `bench` produisait, lui, des chiffres qu'on lisait dans les logs, ce qui
    # suffit pour un tableau de banc. Ça ne suffit plus depuis que `bin/mmlu`
    # sait écrire `LLVQ_MMLU_DUMP` : un dump par question est le seul objet
    # qui rende un écart MMLU testable en appariement, il pèse quelques Mo, et
    # sans chemin de retour il meurt avec le conteneur. On aurait donc payé un
    # rejeu pour découvrir qu'il faut le repayer.
    #
    # Même mécanique que `launch --bucket auto` : `sync_job_volume` crée le
    # bucket à la demande et rend un volume montable en écriture, de sorte que
    # ce qui atterrit sous `--out-mount` survit au job. Rien en local, ce qui
    # est aussi la consigne d'exploitation.
    if args.bucket == "auto":
        from huggingface_hub import sync_job_volume
        out = Path(args.root_out) / args.name
        out.mkdir(parents=True, exist_ok=True)
        vol = sync_job_volume(str(out), args.out_mount, read_only=False)
        volumes.append(vol)
        print(f"  bucket : hf://buckets/{vol.source}/{vol.path} → {args.out_mount}")
    elif args.bucket:
        volumes.append(Volume(type="bucket", source=args.bucket,
                              mount_path=args.out_mount, read_only=False))
        print(f"  bucket : {args.bucket} → {args.out_mount}")
    else:
        print("  ⚠️  sans --bucket, rien de ce que le job écrit ne survit au conteneur")

    script = "set -euo pipefail\n" + "\n".join(args.cmd)
    usd_h = FLAVORS.get(args.flavor, {}).get("usd_h")
    if usd_h:
        mins = _timeout_minutes(args.timeout)
        print(
            f"{args.flavor} à {usd_h:.2f} $/h = {usd_h / 60:.4f} $/min ; "
            f"plafond {args.timeout} → au pire {usd_h * mins / 60:.2f} $"
        )
    job = run_job(
        image=args.image,
        command=["bash", "-lc", script],
        flavor=args.flavor,
        timeout=args.timeout,
        name=args.name,
        namespace=args.namespace,
        volumes=volumes or None,
        # Without this a private repo is simply unreachable: `hf-hub` reads its
        # token from `$HF_HOME/token` and nothing writes that file inside a
        # job. Passed as a secret rather than inlined in the script, which
        # would put it in the logs.
        secrets={"HF_TOKEN": os.environ["HF_TOKEN"]} if os.environ.get("HF_TOKEN") else None,
    )
    print(f"{args.name} sur {args.flavor} : {job.url}\n  id {job.id}")
    print(f"suivre : uv run ops/run.py monitor {job.id} --flavor {args.flavor}")
    return 0


def cmd_dequant(args) -> int:
    """Rebuild an AWQ repository into dense f16, in a Job, straight into a bucket.

    Three constraints meet here and only one arrangement satisfies all three.

    * `ops/awq_dequant.py` is **Python**, and the image this file publishes is
      Rust-only — deliberately, it is a runtime layer with `ca-certificates`
      and `libssl3` and nothing else. So the reconstruction cannot run in it.
    * It cannot run on the dev machine either: ~10 GB of AWQ shards plus a
      dense f16 output that is 29.5 GB on a 14B, against 22 GiB free. The
      operator's rule — everything on the Hub, nothing local — says the same
      thing from the other side.
    * And the output must not be pushed to a model repository afterwards. It
      used to have to be; since `Checkpoint::fetch` learned to accept a local
      directory (code item C5), `bin/ppl` and `bin/mmlu` read it **from the
      mounted bucket**, so a 29.5 GB upload disappears from the campaign.

    `run_uv_job` resolves all three: it ships this repository's own script to a
    `uv` image, which reads the PEP 723 header for its dependencies. No image
    rebuild, no local disk, no intermediate repository.

    ⚠️ The controls are the point, not the reconstruction. Without an entry in
    `EXPECTED`, `awq_dequant` turns off structure, iso-perimeter and tokenizer
    checking, and the failure it exists to catch — an `unpack_cols` that skips
    `AWQ_REVERSE_ORDER`, permuting output channels in packets of 8 — produces
    weights that load, run and give numbers. So the repository is refused here
    unless it is known, rather than degraded silently.
    """
    from huggingface_hub import Volume, run_uv_job

    script = Path(__file__).resolve().parent / "awq_dequant.py"
    if args.awq_repo not in _expected_repos():
        print(
            f"refus : {args.awq_repo} n'a pas d'entrée EXPECTED dans {script.name}.\n"
            "Sans elle, structure, iso-périmètre et tokenizer ne sont PAS vérifiés,\n"
            "et une reconstruction fausse est indiscernable d'une bonne : elle charge,\n"
            "elle tourne, elle produit des nombres. Ajouter l'entrée d'abord — voir\n"
            "celle de Qwen/Qwen3-14B-AWQ, chaque constante y porte sa source.",
            file=sys.stderr,
        )
        return 2

    out_dir = f"{args.out_mount}/{args.name}"
    job = run_uv_job(
        script=str(script),
        script_args=[
            "dequant",
            "--awq-repo", args.awq_repo,
            "--base-repo", args.base_repo,
            "--out", out_dir,
        ],
        flavor=args.flavor,
        timeout=args.timeout,
        volumes=[Volume(type="bucket", source=args.bucket,
                        mount_path=args.out_mount, read_only=False)],
        # Same reason as `bench`: without it a private bucket or repo is simply
        # unreachable, and a secret rather than an inlined value keeps it out
        # of the logs.
        secrets={"HF_TOKEN": os.environ["HF_TOKEN"]} if os.environ.get("HF_TOKEN") else None,
        namespace=args.namespace,
    )
    usd_h = FLAVORS.get(args.flavor, {}).get("usd_h")
    if usd_h:
        mins = _timeout_minutes(args.timeout)
        print(f"{args.flavor} à {usd_h:.2f} $/h ; plafond {args.timeout} → "
              f"au pire {usd_h * mins / 60:.2f} $")
    print(f"  sortie : hf://buckets/{args.bucket}/{args.name} → {out_dir}")
    print(f"dequant {args.awq_repo} : {job.url}\n  id {job.id}")
    return 0


def _expected_repos() -> set[str]:
    """Repository ids `ops/awq_dequant.py` knows how to check.

    Parsed rather than imported: that file declares PEP 723 dependencies this
    one does not have, so importing it would drag numpy and safetensors into a
    launcher that needs neither.

    ⚠️ Keys come in two shapes and reading only one is a refusal of legitimate
    work. The 4B entry is keyed by the module constant `AWQ_REPO`, the 14B by a
    string literal — so a regex over literals alone finds the 14B, declares the
    4B unknown, and blocks a reconstruction the checks would have covered.
    Found the moment this was first run, which is the argument for running it.
    """
    import ast
    import re

    src = (Path(__file__).resolve().parent / "awq_dequant.py").read_text()
    body = src.split("EXPECTED: dict[str, dict] = {", 1)
    if len(body) < 2:
        return set()
    repos = set(re.findall(r'"([\w.-]+/[\w.-]+)"\s*:\s*dict\(', body[1]))
    # Constant-keyed entries: resolve the identifier against its own top-level
    # assignment rather than hard-coding a value that would drift.
    for ident in re.findall(r"^\s{4}([A-Z_][A-Z0-9_]*)\s*:\s*dict\(", body[1], re.M):
        m = re.search(rf"^{ident}\s*=\s*(['\"][^'\"]+['\"])", src, re.M)
        if m:
            repos.add(ast.literal_eval(m.group(1)))
    return repos


def _timeout_minutes(t: str) -> float:
    """`30m`, `2h`, `90s` → minutes. Used only to print a worst-case cost."""
    unit, n = t[-1], t[:-1]
    try:
        v = float(n)
    except ValueError:
        return 0.0
    return {"s": v / 60, "m": v, "h": v * 60}.get(unit, 0.0)


def cmd_monitor(args) -> int:
    """Follow a running Job: stage, accrued cost, utilisation, new log lines.

    Two things this shows that `watch` cannot.

    **What it has cost so far.** Jobs bill by the minute while Starting or
    Running, so the only honest progress bar on a rented run is a dollar
    counter next to the ETA.

    **Whether the accelerator is doing anything.** The phase breakdown says
    97.6 % of a GPU run is the CPU-side encoder, which predicts a GPU sitting
    near-idle for most of the run. If that prediction holds, the next run
    should rent cores rather than a bigger card — and this is where it gets
    confirmed or refuted, on the actual hardware.
    """
    import json
    import threading

    from huggingface_hub import fetch_job_logs, inspect_job

    usd_h = FLAVORS.get(args.flavor, {}).get("usd_h")

    # --- VRAM: peak, mean, and the samples behind them ----------------------
    #
    # `fetch_job_metrics` emits roughly one event per second carrying per-GPU
    # `memory_used_bytes`. It is a **live** stream and only a live one: the
    # library passes `tolerated_status_codes=(500,)` because the endpoint 500s
    # once a Job has finished. A detached job inspected afterwards therefore
    # has **no** metrics at all — which is why this lives in `monitor` and not
    # in a post-mortem command, and why `--detach` costs you the memory axis.
    #
    # Two things the published numbers must carry, and this records both:
    #
    #   * the **time-weighted** mean, not the arithmetic one. The stream has
    #     keep-alives and reconnections, so samples are not equally spaced and
    #     averaging them as if they were would weight a stall like a second.
    #   * the sample count and period, because a peak shorter than the sampling
    #     interval is invisible. 1 Hz is the resolution of the instrument, and
    #     a peak is only ever a lower bound on the true one.
    #
    # And what it measures is what the CUDA context has **reserved**, allocator
    # included — the "does it fit" quantity. It is not a count of live tensor
    # bytes, and it must never be compared against one.
    samples: list[tuple[float, int]] = []

    def meter():
        try:
            from huggingface_hub import fetch_job_metrics, inspect_job as _inspect
        except ImportError:
            print("[metrics] fetch_job_metrics absent de huggingface_hub — "
                  "axe mémoire perdu pour ce job", flush=True)
            return
        # **Re-subscribe in a loop, and it is not defensive coding.** The first
        # pilot subscribed while the Job was still SCHEDULING; the endpoint has
        # nothing to stream for a Job that is not running, so the generator
        # returned immediately, the thread exited, and by the time the card was
        # allocated nobody was listening. The job produced its numbers and the
        # memory axis was simply absent from the result.
        #
        # So: subscribing too EARLY loses the stream exactly as subscribing too
        # late does. Keep re-attaching until the Job reaches a terminal stage.
        attached = False
        while True:
            try:
                st = _inspect(job_id=args.job_id).status.stage
            except Exception:
                st = "UNKNOWN"
            if st in ("COMPLETED", "ERROR", "CANCELED", "DELETED"):
                if not attached:
                    print("[metrics] le Job s'est terminé sans qu'aucun "
                          "échantillon n'ait pu être lu", flush=True)
                return
            try:
                for ev in fetch_job_metrics(job_id=args.job_id):
                    gpus = (ev or {}).get("gpus") or {}
                    used = max((g.get("memory_used_bytes", 0) for g in gpus.values()),
                               default=0)
                    if used:
                        if not attached:
                            print("[metrics] flux attaché", flush=True)
                            attached = True
                        samples.append((time.time(), int(used)))
            except Exception as e:
                print(f"[metrics interrompues: {e}]", flush=True)
            time.sleep(2)

    threading.Thread(target=meter, daemon=True).start()

    # `fetch_job_logs` is a **live stream**: on a running Job it never ends, so
    # `list(...)` on it blocks forever. That is what silenced the first version
    # of this monitor for two hours. Consume it on a thread instead and let the
    # main loop poll status.
    def tail():
        try:
            for line in fetch_job_logs(job_id=args.job_id):
                print(line, flush=True)
        except Exception as e:  # a dead stream must not kill the monitor
            print(f"[logs interrompus: {e}]", flush=True)

    threading.Thread(target=tail, daemon=True).start()

    stage = None
    while True:
        info = inspect_job(job_id=args.job_id)
        if info.status.stage != stage:
            stage = info.status.stage
            print(f"[stage] {stage} {info.status.message or ''}", flush=True)
        # `durations.running_secs` is the **billed** time. Wall clock since
        # `created_at` is not: it keeps counting after the Job ends, and it
        # counts scheduling. Using it reported $35.80 for a run that cost
        # $11.48.
        d = getattr(info, "durations", None)
        # `or 0`, not a `getattr` default: while a Job sits in SCHEDULING the
        # attribute **exists and is None**, so the default never fires and the
        # next division raises. That killed the monitor of the first pilot
        # before the metrics stream had produced a single sample — and the
        # stream is live, so a monitor that dies is an axis that is lost.
        secs = (getattr(d, "running_secs", None) or 0) if d else 0
        cost = f"{secs / 3600 * usd_h:.2f} $" if usd_h else "coût n/d"
        print(f"[{secs / 60:.0f} min facturées · {cost}]", flush=True)
        if stage in ("COMPLETED", "ERROR", "CANCELED", "DELETED"):
            break
        time.sleep(args.every)

    print(f"\n[fin] {stage} — {secs / 3600:.2f} h facturées"
          + (f", {secs / 3600 * usd_h:.2f} $" if usd_h else ""))

    if samples:
        vals = [v for _, v in samples]
        peak = max(vals)
        # Time-weighted mean: each sample holds until the next one.
        span = samples[-1][0] - samples[0][0]
        if span > 0 and len(samples) > 1:
            area = sum(
                samples[i][1] * (samples[i + 1][0] - samples[i][0])
                for i in range(len(samples) - 1)
            )
            mean = area / span
        else:
            mean = float(vals[0])
        ordered = sorted(vals)
        p50 = ordered[len(ordered) // 2]
        p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
        gb = 1e9
        print(f"\n[VRAM] pic {peak / gb:.3f} Go · moyenne {mean / gb:.3f} Go "
              f"· p50 {p50 / gb:.3f} · p95 {p95 / gb:.3f}")
        print(f"       {len(samples)} échantillons sur {span:.0f} s "
              f"({span / max(1, len(samples) - 1):.2f} s entre deux)")
        if peak > 0 and (peak - min(vals)) / peak < 0.02:
            # The control the protocol demands before promising two numbers: if
            # the allocator never returns memory, "peak" and "mean" are two
            # names for one measurement and publishing both is false precision.
            print("       ⚠️  la série ne redescend jamais (< 2 % d'amplitude) : "
                  "pic et moyenne\n           décrivent la même chose, ne pas "
                  "les publier comme deux mesures")
        if args.metrics_out:
            with open(args.metrics_out, "w") as f:
                json.dump({
                    "job_id": args.job_id,
                    "flavor": args.flavor,
                    "billed_secs": secs,
                    "vram_peak_bytes": peak,
                    "vram_mean_bytes": mean,
                    "vram_p50_bytes": p50,
                    "vram_p95_bytes": p95,
                    "samples": len(samples),
                    "span_secs": span,
                    "series": samples,
                }, f, indent=2)
            print(f"       série écrite dans {args.metrics_out}")
    else:
        print("\n[VRAM] aucune métrique reçue — un Job lancé en --detach puis "
              "observé après coup\n       n'en a pas : le flux est live et "
              "l'endpoint 500 une fois le Job fini.")

    return 0 if stage == "COMPLETED" else 1


def cmd_watch(args) -> int:
    from huggingface_hub import fetch_job_logs, inspect_job

    info = inspect_job(job_id=args.job_id)
    print(f"{info.id}  {info.status.stage}  {info.flavor}")
    for line in fetch_job_logs(job_id=args.job_id):
        print(line)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("estimate", help="cœur-heures et coût, sans rien lancer")
    e.add_argument("model", nargs="?", default="Qwen/Qwen3-32B")
    e.add_argument("--blocks", type=int, default=None, help="limiter aux N premiers blocs")
    e.add_argument("--dtype", default="f32", choices=["f32", "bf16", "f16"],
                   help="precision du modele resident (C3)")
    # The C4 planner. A single 32B job is ~11.4 h and, if it dies at hour ten,
    # ten hours are billed for nothing — which is the reason that run keeps not
    # being launched. This prints what cutting it costs and what it bounds the
    # loss to.
    e.add_argument("--segments", type=int, default=None,
                   help="afficher le plan de découpage en N segments (reprise, C4)")
    e.add_argument("--flavor", default="rtx-pro-6000x2", choices=sorted(FLAVORS),
                   help="flavor sur laquelle chiffrer le plan de découpage")
    e.set_defaults(fn=cmd_estimate)

    s = sub.add_parser("selftest", help="confronter l'estimateur au run 4B réel")
    s.set_defaults(fn=cmd_selftest)

    l = sub.add_parser("launch", help="lancer un Job HF")
    l.add_argument("--model", default="Qwen/Qwen3-32B")
    l.add_argument("--flavor", default="cpu-performance", choices=sorted(FLAVORS))
    l.add_argument("--image", required=True, help="ex. <user>/llvq:cpu")
    l.add_argument("--name", default="llvq")
    l.add_argument("--namespace", default=None, help="facturer à une organisation")
    l.add_argument("--bucket", default=None,
                   help="Storage Bucket pour la sortie ; `auto` en crée un")
    l.add_argument("--root-out", default="/tmp/llvq-out",
                   help="dossier local synchronisé quand --bucket auto")
    l.add_argument("--model-mount", default="/model")
    l.add_argument("--mount-model", action="store_true",
                   help="monter le checkpoint en volume plutôt que le retélécharger")
    l.add_argument("--out-mount", default="/out")
    l.add_argument("--device", default="cpu", choices=["cpu", "cuda", "metal"],
                   help="doit s'accorder à la flavor — croisé par `device_ok`")
    l.add_argument("--blocks", type=int, default=None)
    # Code item C4. `--blocks` is an **absolute bound**, not a count, so a
    # second segment of a 64-layer model is `--resume <shard> --blocks 64`,
    # never `--blocks 32`. `smoke` refuses a bound at or below the block the
    # shard stops at rather than quantizing nothing.
    l.add_argument("--resume", default=None, metavar="PATH",
                   help="reprendre derrière un shard, ex. /out/qwen3-32b-seg1.llvq "
                        "(le bloc de reprise est lu dans le shard, pas passé ici)")
    # `leech1c12` — ball m ≤ 12, 47 index bits + 1 gain bit — is the protocol
    # every published number of this project was measured under, 4B and 8B
    # alike (docs/echelle-4b-8b-2026-08-08.md).
    #
    # The default here used to be `leech1c12L3`, and that trailing `L3` is not
    # a detail: it caps a block at three distinct magnitudes, which is what
    # sets the fused kernel's RAM width. It is **dead on quality** — the L ≤ 4
    # swap measured on the sealed 4B costs +4.75 % of perplexity (16.9415 →
    # 17.7459), which puts the file back ABOVE QTIP's 17.04 and loses the one
    # comparison the paper rests on (docs/verdicts-lot-b-2026-08-06.md §B6).
    #
    # And it has already been paid for once as a default: the 8B had to be
    # requantized at $12.61 to purge it (docs/data/jobs.csv, 2026-08-07,
    # « confondant L3 purge »). A VRAM experiment that wants the cap again
    # asks for it by name; it must never be what a bare `launch` produces.
    l.add_argument("--codebook", default="leech1c12",
                   help="défaut leech1c12, le protocole de tous les chiffres publiés ; "
                        "un suffixe L<n> plafonne les niveaux et coûte +4,75 %% de ppl")
    # Mirrors `CalibCorpus` in `llvq-llm/src/bin/smoke.rs`. The binary now
    # refuses an unknown corpus, but it does so **inside the container** —
    # i.e. after the job has started, been billed and pulled the image. The
    # same typo caught here costs nothing. Keep the two lists in step: a
    # corpus added there and not here is simply unreachable from `launch`.
    l.add_argument("--calib", default="c4",
                   choices=["c4", "wikitext2", "wikitext2-test"])
    l.add_argument("--calib-windows", type=int, default=64)
    l.add_argument("--calib-len", type=int, default=2048)
    l.add_argument("--eval-windows", type=int, default=12)
    l.add_argument("--eval-ctx", type=int, default=4096)
    l.add_argument("--calib-seed", type=int, default=None)
    l.add_argument("--damping", default=None)
    l.add_argument("--dtype", default=None, choices=[None, "f16", "bf16", "f32"])
    l.add_argument("--group-scales", action="store_true")
    # `action="store_true"` **with** `default=True` was a contradiction, and it
    # was silent: the flag stores True and the default is True, so `norot` was
    # unreachable from the command line and every `launch` shipped `rot`
    # whatever was typed. That is the one arm nobody could afford to lose — the
    # input rotation is the largest quality lever ever measured here (×2.290 →
    # ×1.811 on the 0.6B, 21 % of perplexity in one step, CLAUDE.md §6), so its
    # control arm has to stay dialable to attribute anything to it.
    l.add_argument("--rotation", action=argparse.BooleanOptionalAction, default=True,
                   help="rotation d'incohérence en entrée (défaut) ; "
                        "--no-rotation pour le bras de contrôle")
    l.add_argument("--max-usd", type=float, default=60.0)
    l.add_argument("--yes", action="store_true", help="passer outre le plafond")
    l.set_defaults(fn=cmd_launch)

    pu = sub.add_parser("publish", help="publier le workspace en Space docker")
    pu.add_argument("space", help="ex. Pier-Jean/llvq-runner")
    pu.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    pu.add_argument("--public", action="store_true",
                    help="par défaut le Space est privé")
    pu.add_argument("--cuda", action="store_true",
                    help="image CUDA (compute cap figée, cf. ops/Dockerfile.cuda)")
    pu.add_argument("--allow-dirty", action="store_true",
                    help="téléverser un périmètre non commité ; COMMIT le déclarera")
    pu.set_defaults(fn=cmd_publish)

    o = sub.add_parser("oracle", help="valider la passe avant sur le backend cible")
    o.add_argument("--image", required=True)
    o.add_argument("--flavor", default="l4x1", choices=sorted(FLAVORS))
    o.add_argument("--device", default="cuda", choices=["cpu", "cuda", "metal"])
    o.add_argument("--model", default="Qwen/Qwen3-0.6B")
    o.add_argument("--tokens", type=int, default=64)
    o.add_argument("--namespace", default=None)
    o.set_defaults(fn=cmd_oracle)

    b = sub.add_parser("bench", help="lancer une commande quelconque de l'image sur une carte")
    b.add_argument("cmd", nargs="+", help="lignes de shell, exécutées sous set -euo pipefail")
    b.add_argument("--image", required=True)
    b.add_argument("--flavor", default="l40sx1")
    b.add_argument("--any-flavor", action="store_true",
                   help="passer outre la liste blanche — à déclarer dans tout chiffre publié")
    b.add_argument("--timeout", default="30m",
                   help="plafond réel du coût : l'estimateur ne vaut que pour une quantification")
    b.add_argument("--mount-model", default=None,
                   help="dépôt du Hub à monter en lecture seule (ex. Pier-Jean/Qwen3-4B-LLVQ-2bit)")
    b.add_argument("--model-mount", default="/model")
    # Mêmes noms que sur `launch`, à dessein : une campagne enchaîne les deux
    # sous-commandes et un opérateur ne doit pas avoir à se rappeler laquelle
    # appelle sa sortie autrement.
    b.add_argument("--bucket", default=None,
                   help="'auto' crée/synchronise un bucket, ou un nom de bucket existant")
    b.add_argument("--root-out", default="/tmp/llvq-out",
                   help="dossier local synchronisé quand --bucket auto")
    b.add_argument("--out-mount", default="/out")
    b.add_argument("--name", default="llvq-bench")
    b.add_argument("--namespace", default=None)
    b.set_defaults(fn=cmd_bench)

    dq = sub.add_parser("dequant",
                        help="reconstruire un dépôt AWQ en f16 dense, dans un bucket")
    dq.add_argument("--awq-repo", required=True, help="ex. Qwen/Qwen3-14B-AWQ")
    dq.add_argument("--base-repo", required=True, help="ex. Qwen/Qwen3-14B")
    dq.add_argument("--bucket", required=True, help="ex. Pier-Jean/jobs-artifacts")
    dq.add_argument("--name", required=True,
                    help="sous-dossier du bucket, ex. qwen3-14b-awq-deq")
    dq.add_argument("--out-mount", default="/out")
    # CPU pur : la reconstruction est du déballage de quartets et de l'écriture
    # de shards. Une carte n'y sert à rien et coûterait 27× le cœur-heure.
    dq.add_argument("--flavor", default="cpu-upgrade", choices=sorted(FLAVORS))
    dq.add_argument("--timeout", default="4h")
    dq.add_argument("--namespace", default=None)
    dq.set_defaults(fn=cmd_dequant)

    m = sub.add_parser("monitor", help="suivre un Job : coût, utilisation, logs")
    m.add_argument("job_id")
    m.add_argument("--flavor", default=None, choices=sorted(FLAVORS),
                   help="pour chiffrer le coût accumulé")
    m.add_argument("--every", type=int, default=120, help="secondes entre relevés")
    m.add_argument("--metrics-out", default=None, metavar="FICHIER.json",
                   help="écrire la série VRAM complète (pic, moyenne pondérée, série brute)")
    m.set_defaults(fn=cmd_monitor)

    w = sub.add_parser("watch", help="statut et logs d'un Job")
    w.add_argument("job_id")
    w.set_defaults(fn=cmd_watch)

    args = p.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
