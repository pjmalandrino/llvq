#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers", "datasets", "huggingface_hub", "accelerate"]
# ///
"""Histogramme de routage d'un MoE — étape 5 du runbook lot X.

La question, et elle n'est pas cosmétique : **la distribution de routage
est-elle assez plate pour que chaque expert voie de quoi former une hessienne
non singulière ?** Notre pipeline calibre sur ~131 k tokens ; sur un MoE, un
expert ne voit que la fraction des tokens que le routeur lui envoie. Si cette
fraction est très inégale, l'expert le moins servi reçoit moins d'échantillons
que la dimension de sa hessienne — qui devient singulière — et le volume de
calibration MoE explose, avec le devis du gate X5-MoE.

Ce script n'utilise PAS notre pipeline : n'importe quel runtime donne les
décisions du routeur, et les prendre chez `transformers` évite de confondre
un défaut de routage avec un défaut de notre passe avant.

    uv run ops/moe_routing.py                         # gpt-oss-20b, 131 k tokens de C4
    uv run ops/moe_routing.py --model X --tokens N

⚠️ Ce qui est compté est un **nombre de tokens routés**, pas une garantie de
rang : un expert peut voir 10 000 tokens fortement colinéaires et rendre une
hessienne singulière quand même. Le compte est un **majorant** de ce que la
calibration peut espérer, exactement comme les comptes de bits du lot X sont
des majorants d'espoir et pas des prédictions de vitesse.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="openai/gpt-oss-20b")
    p.add_argument("--tokens", type=int, default=131_072, help="tokens de calibration à router")
    p.add_argument("--ctx", type=int, default=2048, help="longueur de fenêtre")
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    p.add_argument("--dataset", default="allenai/c4")
    p.add_argument("--json", default=None, help="écrire le détail par couche ici")
    p.add_argument(
        "--from-json",
        default=None,
        help="rejouer le verdict depuis un JSON déjà produit, sans recharger le modèle",
    )
    return p.parse_args()


def load_calibration_tokens(tok, name: str, n_tokens: int, ctx: int) -> torch.Tensor:
    """Le même corpus que la calibration réelle : C4, en flux, tronqué au budget.

    Concaténer puis découper — et non tronquer chaque document à `ctx` — parce
    que c'est ce que fait notre harnais : un document court ne doit pas
    fabriquer une fenêtre courte, qui changerait la statistique de routage
    sans qu'on le voie.
    """
    from datasets import load_dataset

    ds = load_dataset(name, "en", split="train", streaming=True)
    ids: list[int] = []
    for row in ds:
        ids.extend(tok(row["text"]).input_ids)
        if len(ids) >= n_tokens:
            break
    ids = ids[: (n_tokens // ctx) * ctx]
    return torch.tensor(ids, dtype=torch.long).view(-1, ctx)


def attach_router_hooks(model, counts: dict[int, torch.Tensor], top_k: int, n_exp: int):
    """Compter les experts choisis, en accrochant les modules dont le nom finit
    par `router`.

    Un hook plutôt que `output_router_logits=True` : le drapeau n'existe pas
    dans toutes les architectures, et un `getattr` qui échoue en silence
    rendrait un histogramme vide qu'on lirait comme « parfaitement plat ».
    Ici, zéro hook accroché est une erreur franche.
    """
    handles = []
    layers = []

    def make(layer_idx: int):
        def hook(_mod, _inp, out):
            logits = out[0] if isinstance(out, (tuple, list)) else out
            if not torch.is_tensor(logits):
                return
            flat = logits.reshape(-1, logits.shape[-1]).float()
            if flat.shape[-1] != n_exp:
                return
            idx = flat.topk(top_k, dim=-1).indices.reshape(-1)
            counts[layer_idx] += torch.bincount(idx.cpu(), minlength=n_exp)

        return hook

    for name, mod in model.named_modules():
        if name.endswith("router") or name.endswith("gate"):
            layer_idx = len(layers)
            layers.append(name)
            counts[layer_idx] = torch.zeros(n_exp, dtype=torch.long)
            handles.append(mod.register_forward_hook(make(layer_idx)))
    if not handles:
        sys.exit("aucun module de routage accroché — l'architecture ne suit pas la convention `router`/`gate`")
    return handles, layers


def gini(x: torch.Tensor) -> float:
    """Inégalité de la charge, 0 = parfaitement plat, 1 = un seul expert."""
    v = x.double().sort().values
    n = v.numel()
    if v.sum() == 0:
        return float("nan")
    idx = torch.arange(1, n + 1, dtype=torch.float64)
    return float((2 * (idx * v).sum()) / (n * v.sum()) - (n + 1) / n)


def verdict(total: torch.Tensor, n_tokens: int, top_k: int, n_exp: int, dims: list[int]) -> None:
    """Ce que la calibration coûterait, cellule par cellule.

    ⚠️ **Une cellule, ce n'est pas un expert : c'est un couple (couche,
    expert)**, parce que chaque expert de chaque couche a sa propre hessienne.
    Agréger sur les couches donnerait une distribution beaucoup plus plate que
    la réalité — c'est le piège de ce tableau.

    Et surtout : le « pire » n'est PAS le bon résumé. Un expert mort (zéro
    routage) rend n'importe quelle division infinie et écrase la question
    réelle, qui est *combien de tokens pour que la plus grande partie des
    cellules atteigne le rang plein*. On répond donc par quantiles.
    """
    c = total.flatten().double()
    dead = int((c == 0).sum())
    print("\n  VERDICT — combien de tokens pour une hessienne de rang plein")
    print("  " + "-" * 84)
    print(f"  cellules (couche, expert) : {c.numel()}   dont MORTES (zéro routage) : {dead}")
    if dead:
        idx = (total == 0).nonzero().tolist()
        where = ", ".join(f"L{l}/e{e}" for l, e in idx[:8]) + (" …" if dead > 8 else "")
        print(f"    → {where} — aucun volume de calibration ne les sauve : ce n'est")
        print("      pas un problème de volume mais un expert que le routeur n'élit jamais")
    qs = torch.tensor([0.01, 0.05, 0.10, 0.25, 0.50], dtype=torch.float64)
    quants = torch.quantile(c, qs)
    for d in dims:
        print(f"\n  hessienne de dimension {d} — il faut {d} routages par cellule")
        under = int((c < d).sum())
        print(f"    à {n_tokens} tokens : {under} cellules sous le rang plein "
              f"({100 * under / c.numel():.1f} %)")
        for q, v in zip(qs.tolist(), quants.tolist()):
            if v <= 0:
                print(f"    couvrir {100 * (1 - q):.0f} % des cellules : impossible (cellule morte)")
                continue
            need = math.ceil(d * n_tokens / v)
            r = need / n_tokens
            print(
                f"    couvrir {100 * (1 - q):>3.0f} % des cellules : "
                f"{need:>12,} tokens (×{r:.0f})".replace(",", " ")
                if r >= 10
                else f"    couvrir {100 * (1 - q):>3.0f} % des cellules : "
                f"{need:>12,} tokens (×{r:.2f})".replace(",", " ")
            )
    print(
        "\n  ⚠️ « rang plein possible » n'est PAS « hessienne bien conditionnée » :\n"
        "     le compte est un majorant d'espoir, exactement comme les comptes de\n"
        "     bits du lot X ne prédisent aucune vitesse. Des tokens colinéaires\n"
        "     rendent une hessienne singulière quel qu'en soit le nombre.\n"
        f"  ⚠️ Ce modèle active {top_k}/{n_exp} experts ({100 * top_k / n_exp:.1f} %). Les cibles\n"
        "     réelles sont PLUS creuses (Qwen3-30B-A3B : 8/128 = 6,3 % ; K2.6 : 8/384\n"
        "     = 2,1 %), donc leur déséquilibre sera pire, pas meilleur."
    )


def main() -> None:
    a = parse_args()
    if a.from_json:
        d = json.load(open(a.from_json))
        total = torch.tensor(d["counts"], dtype=torch.long)
        print(f"{d['model']} — {d['n_experts']} experts, {d['top_k']} routés, "
              f"{d['tokens']} tokens (relu de {a.from_json})")
        verdict(total, d["tokens"], d["top_k"], d["n_experts"],
                sorted({d["hidden"], d["moe_intermediate"]}))
        return
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    cfg = AutoConfig.from_pretrained(a.model)
    text_cfg = getattr(cfg, "text_config", cfg)
    n_exp = getattr(text_cfg, "num_local_experts", None) or getattr(text_cfg, "num_experts")
    top_k = (
        getattr(text_cfg, "num_experts_per_tok", None)
        or getattr(text_cfg, "experts_per_token", None)
        or getattr(text_cfg, "top_k", 0)
    )
    hidden = getattr(text_cfg, "hidden_size")
    inter = getattr(text_cfg, "intermediate_size", hidden)
    moe_inter = getattr(text_cfg, "moe_intermediate_size", inter)

    print(f"{a.model} — {n_exp} experts, {top_k} routés, hidden {hidden}, moe_inter {moe_inter}")
    print(f"corpus {a.dataset}, {a.tokens} tokens, fenêtres de {a.ctx}, device {a.device}\n")

    tok = AutoTokenizer.from_pretrained(a.model)
    print("tokenisation du corpus…", flush=True)
    windows = load_calibration_tokens(tok, a.dataset, a.tokens, a.ctx)
    print(f"  {windows.shape[0]} fenêtres × {windows.shape[1]} = {windows.numel()} tokens\n")

    print("chargement du modèle (MXFP4 déquantifié en bf16 hors Hopper — c'est lent)…", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map=a.device)
    model.eval()
    print(f"  chargé en {time.time() - t0:.0f} s\n", flush=True)

    counts: dict[int, torch.Tensor] = {}
    handles, layer_names = attach_router_hooks(model, counts, top_k, n_exp)
    print(f"{len(handles)} couches MoE accrochées\n", flush=True)

    t0 = time.time()
    with torch.inference_mode():
        for w in range(windows.shape[0]):
            model(windows[w : w + 1].to(a.device))
            if (w + 1) % 8 == 0 or w + 1 == windows.shape[0]:
                el = time.time() - t0
                print(
                    f"  fenêtre {w + 1}/{windows.shape[0]} — {el:.0f} s "
                    f"({el / (w + 1):.1f} s/fenêtre)",
                    flush=True,
                )
    for h in handles:
        h.remove()

    total = torch.stack([counts[i] for i in sorted(counts)])  # [couches, experts]
    n_tokens = int(total[0].sum()) // top_k
    print(f"\n{n_tokens} tokens routés par couche, {total.shape[0]} couches\n")

    uniform = n_tokens * top_k / n_exp
    print("  charge par expert, en multiple de la charge uniforme")
    print("  " + "-" * 84)
    print(f"  {'couche':>7}  {'min':>8}  {'p10':>8}  {'médiane':>8}  {'p90':>8}  {'max':>8}  {'Gini':>6}")
    print("  " + "-" * 84)
    for li in range(total.shape[0]):
        v = total[li].double() / uniform
        q = torch.quantile(v, torch.tensor([0.10, 0.50, 0.90], dtype=torch.float64))
        if total.shape[0] <= 32 or li % max(1, total.shape[0] // 12) == 0:
            print(
                f"  {li:>7}  {v.min():>8.3f}  {q[0]:>8.3f}  {q[1]:>8.3f}  "
                f"{q[2]:>8.3f}  {v.max():>8.3f}  {gini(total[li]):>6.3f}"
            )
    agg = total.sum(0).double() / (uniform * total.shape[0])
    print("  " + "-" * 84)
    q = torch.quantile(agg, torch.tensor([0.10, 0.50, 0.90], dtype=torch.float64))
    print(
        f"  {'TOUTES':>7}  {agg.min():>8.3f}  {q[0]:>8.3f}  {q[1]:>8.3f}  "
        f"{q[2]:>8.3f}  {agg.max():>8.3f}  {gini(total.sum(0)):>6.3f}"
    )

    verdict(total, n_tokens, top_k, n_exp, sorted({hidden, moe_inter}))

    if a.json:
        with open(a.json, "w") as f:
            json.dump(
                {
                    "model": a.model,
                    "n_experts": n_exp,
                    "top_k": top_k,
                    "hidden": hidden,
                    "moe_intermediate": moe_inter,
                    "tokens": n_tokens,
                    "layer_names": layer_names,
                    "counts": total.tolist(),
                },
                f,
            )
        print(f"\n  détail par couche : {a.json}")


if __name__ == "__main__":
    main()
