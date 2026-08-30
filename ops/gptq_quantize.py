# This script does not run under `uv`: it runs **inside the vLLM job image**,
# with `gptqmodel` installed at job time.
"""Produce the GPTQ 2-bit arm of M3 — on **our** calibration corpus.

## Why we quantize it ourselves instead of downloading one

Every third-party arm the project has measured so far was calibrated on a corpus
we do not know: the official AWQ checkpoint is Qwen's, on Qwen's data. That is a
confound sitting under every quality comparison in the dossier, and it has never
been removable — until an arm is produced here.

This one is. It reads **C4 English validation, shard 1** — the shard
`llvq-llm/src/corpus.rs:187` reserves for calibration, shard 0 being reserved for
evaluation by construction — in **64 contiguous windows of 2048 tokens**, which
is the `n_calib × calib_len` of every published LLVQ artifact. Same corpus, same
shard, same volume, same contiguous-prefix regime (the published artifacts ran
without a seed).

⚠️ **Same corpus is not same procedure.** GPTQ compensates against a Hessian
built from these activations; Spherical GPTQ retracts onto a lattice. Removing
the corpus confound does not make the two methods commensurable in anything but
their input.

## What is fixed here and why

* `bits=2` — the point of the arm.
* `group_size=128` — AutoGPTQ's field default, and what `llvq-llm/src/bin/smoke.rs:333`
  already prices at `bits + 0.25` when it models the competition. Choosing
  anything else would make our own b/param model incomparable to itself.
* `desc_act=True` — act-order. It is what makes 2-bit GPTQ survive at all, and
  it is also the `g_idx` trap named in the preregistration §2.6. We do not
  unpack the weights ourselves, so the permutation is vLLM's to honour, with
  vLLM's code.

## The fingerprint

The calibration token stream gets the same FNV-1a fingerprint the rest of the
project uses. It is printed, and it is the only way a later reader can tell
whether two arms saw the same text.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path

BANNER = "=" * 78

C4_REPO = "allenai/c4"
C4_CALIB_SHARD = "en/c4-validation.00001-of-00008.json.gz"

N_CALIB = 64
CALIB_LEN = 2048

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x00000100000001B3
U64 = (1 << 64) - 1


class Refused(Exception):
    """Configuration refused. Nothing heavy has run."""


def token_fingerprint(ids) -> int:
    h = FNV_OFFSET
    for tid in ids:
        for b in int(tid).to_bytes(4, "little"):
            h = ((h ^ b) * FNV_PRIME) & U64
    return h


def calibration_windows(tok, want_windows: int, win: int):
    """The same stream `bin/smoke` builds: concatenate, then window.

    `smoke.rs:1007` divides one tokenized stream by `calib_len`; the windows are
    contiguous and non-overlapping, and the published artifacts took the prefix
    (no `LLVQ_CALIB_SEED`). Reproduced here rather than approximated.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=C4_REPO, filename=C4_CALIB_SHARD, repo_type="dataset"
    )
    need = want_windows * win
    ids: list[int] = []
    docs = 0
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            ids.extend(tok.encode(json.loads(line)["text"], add_special_tokens=False))
            docs += 1
            if len(ids) >= need:
                break
    if len(ids) < need:
        raise Refused(
            f"{len(ids)} tokens lus pour {need} demandés — le shard est trop court"
        )
    ids = ids[:need]
    print(f"  documents C4 lus      {docs}")
    print(f"  tokens de calibration {len(ids)} = {want_windows} × {win}")
    print(f"  empreinte calibration {token_fingerprint(ids):016x}")
    return [ids[i * win : (i + 1) * win] for i in range(want_windows)]


def bits_per_param(out_dir: Path, n_params: int) -> float:
    """Octets réellement écrits ÷ paramètres. *Mesuré*, pas modélisé.

    C'est la comptabilité que le §7 du CLAUDE.md impose — b/param **modèle
    entier, embedding compris** — et pour un dépôt sur disque elle se lit
    directement, sans modéliser quoi que ce soit.
    """
    total = sum(
        p.stat().st_size for p in out_dir.rglob("*") if p.suffix == ".safetensors"
    )
    return total * 8 / n_params


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--revision", default="1cfa9a7208912126459214e8b04321603b3df60c")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--n-calib", type=int, default=N_CALIB)
    ap.add_argument("--calib-len", type=int, default=CALIB_LEN)
    ap.add_argument(
        "--no-desc-act",
        action="store_true",
        help="désactiver l'act-order — à déclarer dans tout chiffre publié",
    )
    args = ap.parse_args(argv)

    print(BANNER)
    print(f"gptq_quantize — {args.model} @ {args.revision[:12]} → {args.bits} bits")
    print(BANNER)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    print(f"  modèle                {args.model} @ {args.revision[:12]}")
    windows = calibration_windows(tok, args.n_calib, args.calib_len)

    try:
        from gptqmodel import GPTQModel, QuantizeConfig
    except ImportError as e:
        raise Refused(
            "gptqmodel absent de l'image — l'installation au job a échoué, "
            "et rien de lourd n'a tourné"
        ) from e

    cfg = QuantizeConfig(
        bits=args.bits,
        group_size=args.group_size,
        desc_act=not args.no_desc_act,
        sym=True,
    )
    print(f"\n  bits {cfg.bits} · group_size {cfg.group_size} · desc_act {cfg.desc_act}")

    # 🕳️ `GPTQModel.load(..., revision=…)` NE MARCHE PAS : gptqmodel 7.3.5
    # forwarde ses kwargs inconnus jusqu'au constructeur du modèle, et
    # `Qwen3ForCausalLM.__init__()` les refuse — mort en 2 min pour 0,05 $ le
    # 2026-08-30 (job 6a940b8f…).
    #
    # La réparation ne consiste PAS à laisser tomber la révision : ce bras
    # existe pour être comparable à des chiffres datés, et un artefact produit
    # depuis « main » ne se compare à rien de daté. On résout la révision
    # d'abord, on charge un chemin local ensuite.
    from huggingface_hub import snapshot_download

    local = snapshot_download(repo_id=args.model, revision=args.revision)
    print(f"  révision résolue      {args.revision[:12]} → {local}")
    model = GPTQModel.load(local, cfg)

    # 🚨 LE DÉNOMINATEUR, ET IL A ÉTÉ FAUX. `sum(p.numel())` compte
    # `embed_tokens` ET `lm_head` alors que Qwen3-4B a les têtes **liées** :
    # 4 411 424 256 au lieu de 4 022 468 096, soit +9,67 % — exactement la part
    # de l'embedding. Le job du 2026-08-30 a imprimé 3,182 b/param là où la
    # bonne valeur est 3,489, et l'a mis en regard de nos 5,162.
    #
    # C'est la règle n°1 du §7 enfreinte : « toute comparaison mémoire se dit en
    # b/param MODÈLE ENTIER », et deux dénominateurs différents ne se comparent
    # pas. Les deux sont désormais imprimés, étiquetés, et c'est le nôtre qui
    # porte le b/param publiable.
    raw = sum(p.numel() for p in model.model.parameters())
    tied = bool(getattr(model.model.config, "tie_word_embeddings", False))
    vocab = int(model.model.config.vocab_size)
    hidden = int(model.model.config.hidden_size)
    n_params = raw - (vocab * hidden if tied else 0)
    print(f"  paramètres bruts      {raw}   (embed + lm_head comptés deux fois si liés)")
    print(f"  têtes liées           {tied}")
    print(f"  paramètres RÉELS      {n_params}   <-- le dénominateur publiable")

    model.quantize([{"input_ids": w} for w in windows])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    tok.save_pretrained(str(out))

    bpp = bits_per_param(out, n_params)
    bpp_raw = bits_per_param(out, raw)
    print(f"\n  écrit                 {out}")
    print(f"  b/param modèle entier {bpp:.3f}   (*mesuré*, dénominateur RÉEL)")
    print(f"  (pour mémoire)        {bpp_raw:.3f}   avec le compte brut — NE PAS PUBLIER")
    print("  repères               LLVQ 5,162 · AWQ 5,302 · MLX q4 4,50")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Refused as e:
        print(f"\nRefused: {e}", file=sys.stderr)
        sys.exit(2)
