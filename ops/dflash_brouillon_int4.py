# /// script
# requires-python = ">=3.10"
# dependencies = ["mlx-lm==0.28.3", "mlx==0.32.4", "numpy==2.5.4", "safetensors==0.8.0", "tokenizers==0.23.3",
#                 "pyarrow==26.0.1", "transformers==5.22.1", "huggingface_hub", "llvq-tetra"]
# [tool.uv.sources]
# llvq-tetra = { path = "../llvq-tetra" }
# ///
"""DFlash drafter with quantized weights: tree-of-15 acceptance on the Mac, at $0 (step M2, 2026-11-11).

    uv run ops/dflash_brouillon_int4.py <packed_dir> <house.jsonl> <gsm.jsonl> <sp8_brut_dir> <arbre32_brut_dir> \
        <out_dir> [n_gsm] [n_house]

The harness of ops/dflash_arbre32.py (imported, not copied): same text, same prompts, same tree of 15, same replay.
Only the drafter's weights change, by fake quantization: each matrix is quantized and dequantized once, and the
drafter runs f32 arithmetic over the result. Every arm reads the same target capture of the same text, so the arms
are paired prompt by prompt against the unquantized drafter ("ref", the checkpoint's bf16 widened to f32).

Quantizer: llvq_llm::embedquant::quantize_affine, the one int4_record (int4 g128 target records) and embedq (int4 g64
embedding) both call. The matrix is narrowed to f16; per group of `g` consecutive weights along the input dimension,
scale = f16((max - min) / qmax), bias = f16(min), q = round((w - bias) / scale) clamped to [0, qmax], w' = bias +
q * scale; a group whose f16 scale is 0 keeps q = 0. Round-to-nearest, no calibration, no GPTQ. int3 uses the same
formula with qmax = 7 (the Rust function refuses 3 bits; the formula is the only change).

The text for chat, code and rag is the MLX door's greedy decode cached by dflash_arbre32 (mac-greedy.json), copied
from its brut folder so that every arm and the earlier run read the same tokens.
"""
from __future__ import annotations

import gzip
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dflash_arbre32 as A  # noqa: E402

A.NODES = (15,)  # tree_depths reads the module global: only the served tree of 15
LINEAR = ("q", "k", "v", "o", "gate", "up", "down")

# name -> (bits or "f16" or None, group, quantize fc)
ARMS = {
    "ref": (None, 0, False),
    "f16": ("f16", 0, False),
    "int8g64": (8, 64, False),
    "int4g128": (4, 128, False),
    "int4g64": (4, 64, False),
    "int4g128+fc": (4, 128, True),
    "int4g64+fc": (4, 64, True),
    "int3g64": (3, 64, False),
}


def fake_quant(w: np.ndarray, bits: int, group: int) -> np.ndarray:
    """quantize_affine then dequantize, f32 out. w [out, in], in % group == 0."""
    qmax = float((1 << bits) - 1)
    d_out, d_in = w.shape
    assert d_in % group == 0, (w.shape, group)
    x = w.astype(np.float16).astype(np.float32).reshape(d_out, d_in // group, group)
    mn, mx = x.min(axis=-1, keepdims=True), x.max(axis=-1, keepdims=True)
    s = ((mx - mn) / np.float32(qmax)).astype(np.float16).astype(np.float32)
    graded = s > 0
    s = np.where(graded, s, np.float32(1.0))
    b = mn.astype(np.float16).astype(np.float32)
    q = np.clip(np.floor((x - b) / s + np.float32(0.5)), 0.0, qmax)
    q = np.where(graded, q, 0.0)
    return (b + q * s).astype(np.float32).reshape(d_out, d_in)


def build_arms(path: str, names: list[str]):
    import mlx.core as mx

    w = mx.load(path)
    dtypes = sorted({str(v.dtype) for v in w.values()})
    base = A.Drafter(path)  # the reference, f32
    out = {"ref": base}
    np_of = lambda a: np.array(a.astype(mx.float32))
    for name in names:
        if name == "ref":
            continue
        kind, group, do_fc = ARMS[name]
        d = A.Drafter.__new__(A.Drafter)
        d.__dict__.update({k: v for k, v in base.__dict__.items() if k != "layers"})
        d.layers = []
        for l in base.layers:
            nl = dict(l)
            for s in (LINEAR if kind is not None else ()):
                if kind == "f16":
                    nl[s] = l[s].astype(mx.float16).astype(mx.float32)
                else:
                    nl[s] = mx.array(fake_quant(np_of(l[s]), kind, group))
            if kind == "f16":
                for s in ("qn", "kn", "ln1", "ln2"):
                    nl[s] = l[s].astype(mx.float16).astype(mx.float32)
            d.layers.append(nl)
        if kind == "f16":
            d.norm = base.norm.astype(mx.float16).astype(mx.float32)
        if do_fc:
            d.fc = mx.array(fake_quant(np_of(base.fc), kind, group))
        mx.eval(d.fc, d.norm, [list(l.values()) for l in d.layers])
        out[name] = d
    # weight error of each arm against the reference, over the matrices it changed (diagnostic)
    err = {}
    for name, d in out.items():
        num = den = 0.0
        for l0, l1 in zip(base.layers, d.layers):
            for s in LINEAR:
                num += float(mx.sum((l1[s] - l0[s]) ** 2))
                den += float(mx.sum(l0[s] ** 2))
        fcn = float(mx.sum((d.fc - base.fc) ** 2)) / float(mx.sum(base.fc ** 2))
        err[name] = {"linear_rel_mse": num / den, "fc_rel_mse": fcn}
    return out, dtypes, err


def main(argv: list[str]) -> int:
    import mlx.core as mx
    from huggingface_hub import hf_hub_download
    from mlx_lm.generate import generate_step
    from tokenizers import Tokenizer

    import llvq_mlx as L

    packed, house_p, gsm_p, brut, a32, out_dir = (argv[1], Path(argv[2]), Path(argv[3]), Path(argv[4]),
                                                  Path(argv[5]), Path(argv[6]))
    n_gsm = int(argv[7]) if len(argv) > 7 else 200
    n_house = int(argv[8]) if len(argv) > 8 else 10**9
    arm_names = [a for a in ARMS if a in (argv[9].split(",") if len(argv) > 9 else ARMS)]
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    model, _, _ = L.build(packed)
    qe = model.model.embed_tokens
    table = mx.dequantize(qe.weight, qe.scales, qe.biases, group_size=qe.group_size, bits=qe.bits).astype(mx.float32)
    mx.eval(table)
    tok = Tokenizer.from_file(str(Path(packed) / "tokenizer.json"))
    dpath = hf_hub_download(A.DRAFT_REPO, "model.safetensors", revision=A.DRAFT_REV)
    arms, dtypes, err = build_arms(dpath, arm_names)
    print(f"target tetra ({packed}); drafter {A.DRAFT_REPO}@{A.DRAFT_REV[:8]} stored as {dtypes}; arms {arm_names}; "
          f"loaded in {time.perf_counter() - t0:.1f} s", flush=True)
    print("weight error " + json.dumps({k: {kk: float(f"{vv:.4e}") for kk, vv in v.items()} for k, v in err.items()}),
          flush=True)

    dump = {}
    for line in (brut / "gsm200.jsonl").read_text().splitlines():
        o = json.loads(line)
        if o["decoder"] == "pass":
            dump[o["id"]] = o["tokens"]
    card_house, card_gsm = A.card_log(brut / "spec-house.txt"), A.card_log(brut / "spec-gsm200.txt")
    shutil.copyfile(a32 / "mac-greedy.json", out_dir / "mac-greedy.json")
    mac_gen = json.loads((out_dir / "mac-greedy.json").read_text())

    jobs = []
    for r in A.load_prompts(house_p)[:n_house]:
        jobs.append(("house", r["id"], r["kind"], A.HOUSE_BUDGET, r, card_house.get(r["id"], {})))
    for r in A.load_prompts(gsm_p)[:n_gsm]:
        jobs.append(("gsm", r["id"], "gsm8k", A.GSM_BUDGET, r, card_gsm.get(r["id"], {})))

    rows, raw, arm_time = [], {}, {a: 0.0 for a in arms}
    for set_name, pid, kind, budget, rec, card in jobs:
        ids = A.prompt_ids(tok, rec["messages"])
        if card.get("prompt") is not None and card["prompt"] != len(ids):
            raise SystemExit(f"{pid}: prompt {len(ids)} tokens here, {card['prompt']} on the card")
        if pid in dump:
            first = None
            for (t, _), _ in zip(generate_step(mx.array(ids), model, max_tokens=1), range(1)):
                first = int(t)
            gen = [first] + dump[pid][:budget]
            source = "card"
        else:
            if pid not in mac_gen:
                raise SystemExit(f"{pid}: no cached Mac text in {a32 / 'mac-greedy.json'}")
            gen = mac_gen[pid]
            source = "mac"
        x, p = ids + gen, len(ids)
        feats, _ = A.capture(model, x)
        ms = list(range(p, len(x) - 1))
        line = {"set": set_name, "id": pid, "kind": kind, "text": source, "prompt": p, "tokens": len(gen) - 1}
        if "dflash:tree" in card:
            line["card_tree"] = {"tokens": card["dflash:tree"][0], "acc": card["dflash:tree"][1],
                                 "passes": card["dflash:tree"][2]}
        raw[f"{set_name}/{pid}"] = {}
        for name, d in arms.items():
            ta = time.perf_counter()
            ks, vs = d.context(feats)
            parts = [d.drafts(table, table, x, ms[i:i + A.BATCH], ks, vs) for i in range(0, len(ms), A.BATCH)]
            ti = np.concatenate([q[0] for q in parts])
            tl = np.concatenate([q[1] for q in parts])
            order = np.argsort(-tl, axis=-1, kind="stable")
            ti = np.take_along_axis(ti, order, axis=-1)
            tl = np.take_along_axis(tl, order, axis=-1)
            acc = A.tree_depths(x, p, ti, tl)[15]
            rr = A.replay(x, p, acc)
            line[name] = {"passes": len(rr), "accepted": sum(rr), "hist": np.bincount(rr, minlength=16).tolist()}
            raw[f"{set_name}/{pid}"][name] = acc
            arm_time[name] += time.perf_counter() - ta
            del ks, vs
        rows.append(line)
        short = {k: (v if not isinstance(v, dict) or "hist" not in v else round(v["accepted"] / v["passes"], 3))
                 for k, v in line.items()}
        print(json.dumps(short), flush=True)
        del feats
    meta = {"drafter": f"{A.DRAFT_REPO}@{A.DRAFT_REV}", "stored_dtypes": dtypes, "arms": {a: ARMS[a] for a in arms},
            "weight_error": err, "arm_seconds": {a: round(v, 1) for a, v in arm_time.items()}}
    (out_dir / "rows.json").write_text(json.dumps(rows, indent=1))
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    (out_dir / "accept-depths.json.gz").write_bytes(gzip.compress(json.dumps(raw).encode(), mtime=0))
    print(f"arm seconds {meta['arm_seconds']}")
    print(f"done in {time.perf_counter() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
