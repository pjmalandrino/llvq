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

Two measured constants, both from this repo's own benches on an M3 Max:

* the Leech encoder runs at 1469 blocks/s/core (`bin/encbench`,
  `nearest_angular` — the path Phase 5 calls);
* `GptqFactor::new` reaches ~110 G mult-add/s (`bin/cholbench` at n = 4096,
  still climbing, so this is conservative).

The second one is the reason the flavor advice changed: the factorization is
**1.5 % of a run**, not the dominant term it was before `faer`. What costs is
the Leech encoding, which is pure CPU — so a GPU flavor is mostly paying for an
idle accelerator.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request

# --- hardware ---------------------------------------------------------------
#
# From https://huggingface.co/docs/hub/en/jobs-pricing (checked 2026-08-02).
# `hf jobs hardware` prints the live table; refresh this one if it drifts.
#
# `usd_per_core_hour` is the number that actually decides, because the run is
# CPU-bound. Sorted worst to best on that metric in the comments below.
FLAVORS: dict[str, dict] = {
    #                       vCPU  RAM Go  VRAM Go  $/h
    "cpu-upgrade":     dict(vcpu=8,   ram=32,   vram=0,   usd_h=0.03),
    "cpu-xl":          dict(vcpu=16,  ram=124,  vram=0,   usd_h=1.00),
    "cpu-performance": dict(vcpu=32,  ram=256,  vram=0,   usd_h=1.90),
    "l4x1":            dict(vcpu=8,   ram=30,   vram=24,  usd_h=0.80),
    "l40sx1":          dict(vcpu=8,   ram=62,   vram=48,  usd_h=1.80),
    "a100-large":      dict(vcpu=12,  ram=142,  vram=80,  usd_h=2.50),
    "rtx-pro-6000":    dict(vcpu=23,  ram=256,  vram=96,  usd_h=2.75),
    "h200":            dict(vcpu=23,  ram=256,  vram=141, usd_h=5.00),
}

# Measured, see the module docstring. Change these only with a bench to back it.
BLOCKS_PER_SEC_PER_CORE = 1469.0
CHOLESKY_MACS_PER_SEC = 110e9
DIM = 24

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
    quantized, carried = weight_counts(cfg)
    quantized = int(quantized * frac)

    leech_h = (quantized / DIM) / BLOCKS_PER_SEC_PER_CORE / 3600.0
    macs = sum(2.0 / 3.0 * n**3 for n in act_widths(cfg)) * layers
    chol_h = macs / CHOLESKY_MACS_PER_SEC / 3600.0

    artifact_gb = (quantized * BITS_PER_WEIGHT / 8 + carried * 2) / 1e9
    checkpoint_gb = (quantized + carried) * 2 / 1e9
    return dict(
        layers=layers,
        quantized=quantized,
        carried=carried,
        leech_core_h=leech_h,
        chol_core_h=chol_h,
        artifact_gb=artifact_gb,
        checkpoint_gb=checkpoint_gb,
    )


def cost_table(est: dict) -> list[tuple[str, float, float, str]]:
    """`(flavor, wall_hours, usd, warning)` for every flavor.

    The forward passes are deliberately **not** modelled. On a GPU they are
    ~10 minutes for a 32B; on CPU they are hours, and the spread of candle's
    CPU gemm throughput is a factor of 3. That unknown is exactly what the 8B
    step is for, so guessing it here would dress a guess up as an estimate.
    """
    rows = []
    parallel_h = est["leech_core_h"] + est["chol_core_h"]
    for name, f in FLAVORS.items():
        wall = parallel_h / f["vcpu"]
        warn = ""
        if f["vram"] == 0:
            warn = "CPU forward not counted — hours, not minutes"
        elif f["vram"] < est["checkpoint_gb"]:
            warn = f"VRAM {f['vram']} Go < modèle {est['checkpoint_gb']:.0f} Go"
        if f["ram"] < est["checkpoint_gb"]:
            warn = (warn + "; " if warn else "") + f"RAM {f['ram']} Go serrée"
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
    for name, wall, usd, warn in sorted(cost_table(est), key=lambda r: r[2]):
        print(f"  {name:<18}{wall:>10.1f}{usd:>9.2f}   {warn}")
    print("\n  Les passes avant ne sont PAS comptées : ~10 min sur GPU, plusieurs")
    print("  heures sur CPU. C'est l'étape 8B qui tranche entre les deux colonnes.")
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
    rows = {name: (wall, usd) for name, wall, usd, _ in cost_table(est)}
    if args.flavor not in rows:
        print(f"flavor inconnu: {args.flavor}", file=sys.stderr)
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

    # `LLVQ_MODEL` must accept a local directory for the mounted volume to be
    # of any use — that is code item C5, and until it lands the container will
    # re-download the checkpoint from the Hub instead.
    env = {
        "LLVQ_MODEL": args.model_mount,
        "LLVQ_CALIB": args.calib,
        "LLVQ_ARTIFACT": f"{args.out_mount}/{args.name}.llvq",
        "LLVQ_THREADS": str(FLAVORS[args.flavor]["vcpu"]),
    }
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

    volumes = [
        Volume(type="model", source=args.model, mount_path=args.model_mount,
               read_only=True),
    ]
    if args.bucket:
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
    e.set_defaults(fn=cmd_estimate)

    s = sub.add_parser("selftest", help="confronter l'estimateur au run 4B réel")
    s.set_defaults(fn=cmd_selftest)

    l = sub.add_parser("launch", help="lancer un Job HF")
    l.add_argument("--model", default="Qwen/Qwen3-32B")
    l.add_argument("--flavor", default="cpu-performance", choices=sorted(FLAVORS))
    l.add_argument("--image", required=True, help="ex. <user>/llvq:cpu")
    l.add_argument("--name", default="llvq")
    l.add_argument("--namespace", default=None, help="facturer à une organisation")
    l.add_argument("--bucket", default=None, help="Storage Bucket pour la sortie")
    l.add_argument("--model-mount", default="/model")
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

    w = sub.add_parser("watch", help="statut et logs d'un Job")
    w.add_argument("job_id")
    w.set_defaults(fn=cmd_watch)

    args = p.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
