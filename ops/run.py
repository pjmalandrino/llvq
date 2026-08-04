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

    # Mounting the checkpoint as a volume is only useful once `LLVQ_MODEL`
    # accepts a local directory — code item C5. Until it lands, `--mount-model`
    # would hand the loader a path it will try to resolve as a Hub repo id, so
    # the default is to let the container download from the Hub. That is
    # 65 GB per run on a 32B, which is exactly why C5 matters.
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


def cmd_publish(args) -> int:
    """Publish the workspace as a Docker Space, which HF then builds for us.

    A Space is used here purely as a build service: pushing a multi-gigabyte
    image from a laptop is slow, and HF will build from source for free.
    `run_job(image="hf.co/spaces/<user>/<name>")` then reuses the result.

    Private by default. This is someone's research repository, and making it
    public is a decision they take, not one a script takes for them.
    """
    from huggingface_hub import create_repo, upload_file, upload_folder

    repo_id = args.space
    create_repo(repo_id, repo_type="space", space_sdk="docker",
                private=not args.public, exist_ok=True)
    print(f"space {repo_id} ({'public' if args.public else 'privé'})")

    # Allow-list, not deny-list: `COPY . .` copies whatever ends up here, so a
    # forgotten exclusion means a slower build, and `target/` alone is tens of
    # gigabytes.
    #
    # `Cargo.lock` is in the list on purpose. Without it the builder resolves
    # dependencies fresh, so the image would not be built from the tree that
    # was committed — and every number the image produces would be traceable to
    # a commit that does not describe it. That is exactly the provenance gap
    # this campaign exists to close.
    upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=str(args.root),
        allow_patterns=["Cargo.toml", "Cargo.lock", "llvq-*/**"],
        ignore_patterns=["**/target/**", "**/*.log"],
        commit_message="LLVQ workspace",
    )
    # A Space builds its Dockerfile as written, so the CPU/CUDA choice is made
    # by *which file* we upload — `--build-arg` never reaches the Hub builder.
    recipe = "Dockerfile.cuda" if args.cuda else "Dockerfile"
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
                   help="monter le checkpoint en volume — exige C5")
    l.add_argument("--out-mount", default="/out")
    l.add_argument("--device", default="cpu", choices=["cpu", "cuda", "metal"])
    l.add_argument("--blocks", type=int, default=None)
    l.add_argument("--codebook", default="leech1c12L3")
    l.add_argument("--calib", default="c4")
    l.add_argument("--calib-windows", type=int, default=64)
    l.add_argument("--calib-len", type=int, default=2048)
    l.add_argument("--eval-windows", type=int, default=12)
    l.add_argument("--eval-ctx", type=int, default=4096)
    l.add_argument("--calib-seed", type=int, default=None)
    l.add_argument("--damping", default=None)
    l.add_argument("--dtype", default=None, choices=[None, "f16", "bf16", "f32"])
    l.add_argument("--group-scales", action="store_true")
    l.add_argument("--rotation", action="store_true", default=True)
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
    pu.set_defaults(fn=cmd_publish)

    o = sub.add_parser("oracle", help="valider la passe avant sur le backend cible")
    o.add_argument("--image", required=True)
    o.add_argument("--flavor", default="l4x1", choices=sorted(FLAVORS))
    o.add_argument("--device", default="cuda", choices=["cpu", "cuda", "metal"])
    o.add_argument("--model", default="Qwen/Qwen3-0.6B")
    o.add_argument("--tokens", type=int, default=64)
    o.add_argument("--namespace", default=None)
    o.set_defaults(fn=cmd_oracle)

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
