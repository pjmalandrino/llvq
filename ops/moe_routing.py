#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers", "datasets", "huggingface_hub", "accelerate"]
# ///
"""Routing histogram of a MoE, step 5 of the lot X runbook.

The question, and it is not cosmetic: **is the routing distribution flat enough
for every expert to see what it takes to form a non-singular Hessian?** Our
pipeline calibrates on ~131 k tokens; on a MoE, an expert only sees the fraction
of the tokens the router sends it. If that fraction is very uneven, the least
served expert gets fewer samples than the dimension of its Hessian, which then
becomes singular, and the MoE calibration volume explodes, along with the
X5-MoE gate estimate.

This script does NOT use our pipeline: any runtime gives the router decisions,
and taking them from `transformers` avoids mistaking a routing defect for a
defect of our forward pass.

    uv run ops/moe_routing.py                         # gpt-oss-20b, 131 k tokens of C4
    uv run ops/moe_routing.py --model X --tokens N

WARNING: what is counted is a **number of routed tokens**, not a rank guarantee.
An expert can see 10,000 strongly collinear tokens and still give a singular
Hessian. The count is an **upper bound** on what calibration can hope for, in
the same way the bit counts of lot X bound what to hope for and predict no
speed.
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
    p.add_argument("--tokens", type=int, default=131_072, help="calibration tokens to route")
    p.add_argument("--ctx", type=int, default=2048, help="window length")
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    p.add_argument("--dataset", default="allenai/c4")
    p.add_argument("--json", default=None, help="write the per-layer detail here")
    p.add_argument(
        "--from-json",
        default=None,
        help="replay the verdict from a JSON already produced, without reloading the model",
    )
    return p.parse_args()


def load_calibration_tokens(tok, name: str, n_tokens: int, ctx: int) -> torch.Tensor:
    """The same corpus as the real calibration: C4, streamed, truncated to budget.

    Concatenate then cut, rather than truncate each document to `ctx`,
    because that is what our harness does. A short document must not make
    a short window, which would change the routing statistics without us
    noticing.
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
    """Count the chosen experts by hooking the modules whose name ends in
    `router`.

    A hook rather than `output_router_logits=True`: the flag does not exist in
    every architecture, and a `getattr` that fails silently would return an
    empty histogram we would read as "perfectly flat". Here, zero hooks
    attached is an outright error.
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
        sys.exit("no routing module hooked: the architecture does not follow the `router`/`gate` convention")
    return handles, layers


def gini(x: torch.Tensor) -> float:
    """Load inequality, 0 = perfectly flat, 1 = a single expert."""
    v = x.double().sort().values
    n = v.numel()
    if v.sum() == 0:
        return float("nan")
    idx = torch.arange(1, n + 1, dtype=torch.float64)
    return float((2 * (idx * v).sum()) / (n * v.sum()) - (n + 1) / n)


def verdict(total: torch.Tensor, n_tokens: int, top_k: int, n_exp: int, dims: list[int]) -> None:
    """What calibration would cost, cell by cell.

    WARNING: **a cell is not an expert, it is a (layer, expert) pair**, because
    every expert of every layer has its own Hessian. Aggregating over the layers
    would give a distribution much flatter than reality. That is the trap of
    this table.

    The "worst" is NOT the right summary. A dead expert (zero routing) makes any
    division infinite and crushes the real question, which is *how many tokens
    for the largest part of the cells to reach full rank*. The answer is given
    in quantiles.
    """
    c = total.flatten().double()
    dead = int((c == 0).sum())
    print("\n  VERDICT: how many tokens for a full-rank Hessian")
    print("  " + "-" * 84)
    print(f"  cells (layer, expert): {c.numel()}   of which DEAD (zero routing): {dead}")
    if dead:
        idx = (total == 0).nonzero().tolist()
        where = ", ".join(f"L{l}/e{e}" for l, e in idx[:8]) + (" …" if dead > 8 else "")
        print(f"    → {where}: no calibration volume saves them. The router simply")
        print("      never elects that expert, so volume is not the problem")
    qs = torch.tensor([0.01, 0.05, 0.10, 0.25, 0.50], dtype=torch.float64)
    quants = torch.quantile(c, qs)
    for d in dims:
        print(f"\n  Hessian of dimension {d}: it takes {d} routings per cell")
        under = int((c < d).sum())
        print(f"    at {n_tokens} tokens: {under} cells under full rank "
              f"({100 * under / c.numel():.1f} %)")
        for q, v in zip(qs.tolist(), quants.tolist()):
            if v <= 0:
                print(f"    cover {100 * (1 - q):.0f} % of the cells: impossible (dead cell)")
                continue
            need = math.ceil(d * n_tokens / v)
            r = need / n_tokens
            print(
                f"    cover {100 * (1 - q):>3.0f} % of the cells: "
                f"{need:>12,} tokens (×{r:.0f})".replace(",", " ")
                if r >= 10
                else f"    cover {100 * (1 - q):>3.0f} % of the cells: "
                f"{need:>12,} tokens (×{r:.2f})".replace(",", " ")
            )
    print(
        "\n  WARNING: \"full rank possible\" is NOT \"well-conditioned Hessian\".\n"
        "     The count bounds what to hope for, exactly as the bit counts of lot X\n"
        "     predict no speed. Collinear tokens make a Hessian singular whatever\n"
        "     their number.\n"
        f"  WARNING: this model activates {top_k}/{n_exp} experts ({100 * top_k / n_exp:.1f} %). The real\n"
        "     targets are SPARSER (Qwen3-30B-A3B: 8/128 = 6.3 %; K2.6: 8/384\n"
        "     = 2.1 %), so their imbalance will be worse, not better."
    )


def main() -> None:
    a = parse_args()
    if a.from_json:
        d = json.load(open(a.from_json))
        total = torch.tensor(d["counts"], dtype=torch.long)
        print(f"{d['model']}: {d['n_experts']} experts, {d['top_k']} routed, "
              f"{d['tokens']} tokens (reread from {a.from_json})")
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

    print(f"{a.model}: {n_exp} experts, {top_k} routed, hidden {hidden}, moe_inter {moe_inter}")
    print(f"corpus {a.dataset}, {a.tokens} tokens, windows of {a.ctx}, device {a.device}\n")

    tok = AutoTokenizer.from_pretrained(a.model)
    print("tokenizing the corpus…", flush=True)
    windows = load_calibration_tokens(tok, a.dataset, a.tokens, a.ctx)
    print(f"  {windows.shape[0]} windows × {windows.shape[1]} = {windows.numel()} tokens\n")

    print("loading the model (MXFP4 dequantized to bf16 off Hopper: this is slow)…", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map=a.device)
    model.eval()
    print(f"  loaded in {time.time() - t0:.0f} s\n", flush=True)

    counts: dict[int, torch.Tensor] = {}
    handles, layer_names = attach_router_hooks(model, counts, top_k, n_exp)
    print(f"{len(handles)} MoE layers hooked\n", flush=True)

    t0 = time.time()
    with torch.inference_mode():
        for w in range(windows.shape[0]):
            model(windows[w : w + 1].to(a.device))
            if (w + 1) % 8 == 0 or w + 1 == windows.shape[0]:
                el = time.time() - t0
                print(
                    f"  window {w + 1}/{windows.shape[0]}: {el:.0f} s "
                    f"({el / (w + 1):.1f} s/window)",
                    flush=True,
                )
    for h in handles:
        h.remove()

    total = torch.stack([counts[i] for i in sorted(counts)])  # [layers, experts]
    n_tokens = int(total[0].sum()) // top_k
    print(f"\n{n_tokens} tokens routed per layer, {total.shape[0]} layers\n")

    uniform = n_tokens * top_k / n_exp
    print("  load per expert, as a multiple of the uniform load")
    print("  " + "-" * 84)
    print(f"  {'layer':>7}  {'min':>8}  {'p10':>8}  {'median':>8}  {'p90':>8}  {'max':>8}  {'Gini':>6}")
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
        f"  {'ALL':>7}  {agg.min():>8.3f}  {q[0]:>8.3f}  {q[1]:>8.3f}  "
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
        print(f"\n  per-layer detail: {a.json}")


if __name__ == "__main__":
    main()
