# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch",
#   "numpy",
#   "safetensors",
#   "tokenizers",
#   "pyarrow",
#   "transformers @ git+https://github.com/huggingface/transformers@046bdb8cc2",
#   "gguf @ git+https://github.com/ggml-org/llama.cpp@690b87d881#subdirectory=gguf-py",
# ]
# ///
"""The whole-model arms against the native teacher, resumable window by window.

    uv run evalfull_arms.py check  [--attn DIR] [--tables DIR]          # light: which arms have their inputs
    uv run evalfull_arms.py run ARM [--windows 0-7] [--attn DIR] ...     # heavy: the windows not yet done
    uv run evalfull_arms.py report                                       # light: summary of the finished arms

Arms (D = ~/llvq-inkling-mini):
  experts24        experts of all 24 layers from D/layers (dir:, `.f32` or `.f16`), everything else native
  experts24_tables + `embed` and `unembed` from `--tables`, attention native: the file the operator chose on 2026-11-22
                   (attention kept in 16 bits)
  experts24_attn   + the six attention matrices of every layer from `--attn` (B's inkwh reconstructions)
  whole            + `embed` and `unembed` from `--tables` (B's int4 g64 tables): the whole quantized text model
  attn_only        the attention matrices alone, experts and tables native: what the attention's type rule costs
  attn_lattice_only  the two attention matrices the lattice code holds (wq_du, wk_dv) alone, the rest native
  attn_int4_only   the four int4 attention matrices (wv_dv, wo_ud, gate_attn, wr_du) alone, the rest native
  tables_only      the two tables alone
  rival_iq2xxs     the UD-IQ2_XXS file whole: experts through ggufx, every other tensor through ggufmap
  rival_q2kxl      the UD-Q2_K_XL file whole
A rival arm transplants the file's weights into the reference forward pass (transformers 046bdb8cc2, float32). It
measures the weights, not llama.cpp's kernels, its activation rounding or its own forward code.

Each window's row goes to D/evalfull/arms/<arm>/w{i}.json with the arm's spec and a sha256 of it; a window already
there with the same spec is skipped, one with another spec is refused. `run` refuses an arm whose inputs are
incomplete (the `check` rules): all 24 layers with `VERIFIED` and both reconstructions; all 144 attention
reconstructions; both tables; a verified GGUF map with no unverified row. The summary (`report`) is evalkl.py's:
KL with its bootstrap interval over the 8 windows, top-1, paired NLL and its interval, plus the perplexity ratio.
"""

import _threads  # noqa: F401

import argparse
import glob
import hashlib
import json
import os
import sys
import time

from _threads import DATA

LAYERS = os.path.join(DATA, "layers")
ARMS_DIR = os.path.join(DATA, "evalfull", "arms")
GGUF = {"rival_iq2xxs": os.path.join(DATA, "gguf", "Inkling-2-Mini-UD-IQ2_XXS.gguf"),
        "rival_q2kxl": os.path.join(DATA, "gguf", "Inkling-2-Mini-UD-Q2_K_XL.gguf")}
ARMS = ("experts24", "experts24_tables", "experts24_attn", "whole", "attn_only", "attn_lattice_only", "attn_int4_only", "tables_only",
        "rival_iq2xxs", "rival_q2kxl")
# Which attention matrices the lattice code holds in the plan's default rule; the four others are int4 g128.
ATTN_LATTICE = ("wq_du", "wk_dv")


def spec_of(arm, attn, tables):
    e = f"experts:all=dir:{LAYERS}"
    a = f"attention:all=attn:{attn}"
    t = (f"model.embed_tokens.weight=table:{os.path.join(tables, 'embed.q4g64.llvqraw')};"
         f"lm_head.weight=table:{os.path.join(tables, 'unembed.q4g64.llvqraw')}")
    if arm == "experts24":
        return e
    if arm == "experts24_tables":
        return f"{e};{t}"
    if arm == "experts24_attn":
        return f"{e};{a}"
    if arm == "whole":
        return f"{e};{a};{t}"
    if arm == "attn_only":
        return a
    if arm in ("attn_lattice_only", "attn_int4_only"):
        # one matrix a key, so that the other attention matrices stay native
        import model_full

        idx = model_full.attn_index(attn) if attn and os.path.isdir(attn) else {}
        want = [n for n in model_full.ATTN_NATIVE if (n in ATTN_LATTICE) == (arm == "attn_lattice_only")]
        return ";".join(f"model.layers.{L}.{model_full.ATTN_NATIVE[n]}=file:{idx[(L, n)]}"
                        for L in range(24) for n in want if (L, n) in idx)
    if arm == "tables_only":
        return t
    if arm in GGUF:
        return f"experts:all=gguf:{GGUF[arm]};nonexperts=gguf:{GGUF[arm]}"
    raise KeyError(arm)


def missing(arm, attn, tables):
    """What an arm lacks; empty when it can run."""
    out = []
    if arm in ("experts24", "experts24_tables", "experts24_attn", "whole"):
        for L in range(24):
            d = os.path.join(LAYERS, f"L{L:02d}")
            ok = os.path.exists(os.path.join(d, "VERIFIED")) and all(
                os.path.exists(os.path.join(d, f"{s}.f32")) or os.path.exists(os.path.join(d, f"{s}.f16"))
                for s in ("w13_hat", "w2_hat"))
            if not ok:
                out.append(f"experts L{L:02d}")
    if arm in ("experts24_attn", "whole", "attn_only", "attn_lattice_only", "attn_int4_only"):
        import model_full

        if not attn or not os.path.isdir(attn):
            out.append(f"attention directory {attn!r}")
        else:
            idx = model_full.attn_index(attn)
            lack = [f"L{L}.{n}" for L in range(24) for n in model_full.ATTN_NATIVE if (L, n) not in idx]
            if lack:
                out.append(f"{len(lack)} of 144 attention reconstructions (first: {lack[:4]})")
    if arm in ("whole", "tables_only", "experts24_tables"):
        for t in ("embed.q4g64.llvqraw", "unembed.q4g64.llvqraw"):
            if not os.path.exists(os.path.join(tables or "", t)):
                out.append(f"table {t}")
    if arm in GGUF:
        stem = os.path.splitext(os.path.basename(GGUF[arm]))[0]
        p = os.path.join(DATA, "evalfull", f"ggufmap_{stem}.json")
        if not os.path.exists(GGUF[arm]):
            out.append(GGUF[arm])
        elif not os.path.exists(p):
            out.append(f"{p} (ggufmap.py verify)")
        else:
            c = json.load(open(p))["counts"]
            if set(c) != {"verified"}:
                out.append(f"GGUF map not fully verified: {c}")
    return out


def _windows(s):
    a, b = s.split("-") if "-" in s else (s, s)
    return list(range(int(a), int(b) + 1))


def run(arm, wins, attn, tables, device="mps"):
    lack = missing(arm, attn, tables)
    if lack:
        sys.exit(f"arm {arm}: inputs missing: {lack}")
    import evalfull

    spec = spec_of(arm, attn, tables)
    h = hashlib.sha256(spec.encode()).hexdigest()[:16]
    d = os.path.join(ARMS_DIR, arm)
    os.makedirs(d, exist_ok=True)
    todo = []
    for i in wins:
        p = os.path.join(d, f"w{i}.json")
        if os.path.exists(p):
            r = json.load(open(p))
            if r["spec_sha"] != h:
                sys.exit(f"{p} was made with another spec ({r['spec']}); move it away first")
            continue
        todo.append(i)
    if not todo:
        print(f"arm {arm}: windows {wins} already done")
        return
    t0 = time.time()
    rows, subst, fp, _ = evalfull.run_rows(spec, device, todo, 2048)
    for i in todo:
        r = dict(rows[i], spec=spec, spec_sha=h, tokens_fingerprint_of_8=fp, device=device,
                 n_substituted=len(subst), sec_job=time.time() - t0)
        tmp = os.path.join(d, f"w{i}.json.tmp")
        json.dump(r, open(tmp, "w"), indent=1)
        os.replace(tmp, os.path.join(d, f"w{i}.json"))
    print(f"arm {arm}: windows {todo} in {time.time() - t0:.0f}s, {len(subst)} non-expert tensors substituted",
          flush=True)


def report():
    import evalfull

    out = {}
    for arm in ARMS:
        rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(ARMS_DIR, arm, "w*.json")))]
        if len(rows) != 8:
            out[arm] = {"windows_done": len(rows)}
            continue
        rows.sort(key=lambda r: r["window"])
        out[arm] = {"spec": rows[0]["spec"], **evalfull.aggregate(rows, 10000)}
    json.dump(out, open(os.path.join(ARMS_DIR, "summary.json"), "w"), indent=1)
    for arm, r in out.items():
        if "kl_mean" in r:
            print(f"{arm:16s} KL {r['kl_mean']:.5f} [{r['kl_ci95'][0]:.5f}, {r['kl_ci95'][1]:.5f}]  "
                  f"top1 {100 * r['top1_agree']:.2f}%  dNLL {r['nll_diff_mean']:+.5f} "
                  f"[{r['nll_diff_ci95'][0]:+.5f}, {r['nll_diff_ci95'][1]:+.5f}]  ppl x{r['ppl_ratio']:.4f}")
        else:
            print(f"{arm:16s} {r['windows_done']}/8 windows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["check", "run", "report"])
    ap.add_argument("arm", nargs="?")
    ap.add_argument("--windows", default="0-7")
    ap.add_argument("--attn", default=os.environ.get("INKLING_ATTN_DIR", os.path.join(DATA, "lot1-b", "attn")))
    ap.add_argument("--tables", default=os.path.join(DATA, "lot1-b", "tables"))
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    if a.mode == "check":
        bad = 0
        for arm in ARMS:
            m = missing(arm, a.attn, a.tables)
            print(f"{arm:16s} {'ready' if not m else 'missing: ' + '; '.join(m)}")
            bad += bool(m)
        sys.exit(1 if bad else 0)
    if a.mode == "report":
        report()
        return
    run(a.arm, _windows(a.windows), a.attn, a.tables, a.device)


if __name__ == "__main__":
    main()
