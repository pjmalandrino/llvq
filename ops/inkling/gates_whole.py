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
"""Gates of the whole-model sources (attention files, tables), and the agreement of the arm path with last night's.

    uv run gates_whole.py attn  <native_dir> <b_dir>   # heavy: 4 evaluations of window 0
    uv run gates_whole.py table <tables_dir>           # heavy: decode check + 4 evaluations of window 0
    uv run gates_whole.py agree                        # heavy: experts 6-17 on 8 windows vs eval/arm_tetra_6-17.json

attn: `<native_dir>` holds native attention matrices written in B's reconstruction layout
(`model.layers.{L}.attn.{nat}.weight.f32`, raw f32 [out, in], what `attn_slices.py whprep` writes); `<b_dir>` holds
reconstructions written by B's `inkwh`. (1) layers 0 and 6 from the native files through `attention:` -> KL must be
exactly 0; (2) B's reconstructions present in `<b_dir>` -> KL > 0 and small; (3) one matrix read transposed and
(4) the key and value matrices of one layer swapped -> KL far above (2).
table: (0) B's int4 g64 tables decoded by `model_full.read_llvqraw` against the native table narrowed to f16: every
value within half a step of its group (the encoder's rounding bound), else the decode is wrong; (1) the native table
written as an f16 raw tensor in the same framing and read back through `table:` -> KL 0 when the f16 narrowing is
exact; (2) B's embed, (3) B's unembed -> KL > 0 and small; (4) the embed table forced into `lm_head` -> KL blows up,
and without the force it is refused by name.
Results: D/evalfull/gates_whole_<mode>.json.
"""

import _threads  # noqa: F401

import json
import os
import struct
import sys
import time

import numpy as np

from _threads import DATA, TEXTONLY

import evalfull  # noqa: E402
import model_full  # noqa: E402
from executor import load_model, torch  # noqa: E402

torch.set_grad_enabled(False)
OUT = evalfull.OUT


def ev(spec, model, name, nwin=1):
    rows, subst, _, _ = evalfull.run_rows(spec, "mps", range(nwin), 2048, model=model, restore=True)
    agg = evalfull.aggregate([rows[i] for i in range(nwin)], 1000)
    r = {"spec": spec, "n_substituted": len(subst),
         "max_abs_change": max((v["max_abs_change"] for v in subst.values()), default=0.0), **agg}
    print(f"{name}: KL {r['kl_mean']:.6g} (max {r['kl_max']:.3g}) top1 {100 * r['top1_agree']:.2f}% "
          f"ppl x{r['ppl_ratio']:.5f} substituted {r['n_substituted']} max change {r['max_abs_change']:.3g}",
          flush=True)
    return r


def gate_attn(native_dir, b_dir):
    model = load_model("mps")
    res = {}
    res["native_files_L0_L6"] = ev(f"attention:0+6=attn:{native_dir}", model, "native via file path")
    idx = model_full.attn_index(b_dir)
    if not idx:
        sys.exit(f"no reconstruction under {b_dir}")
    spec = ";".join(f"model.layers.{L}.{model_full.ATTN_NATIVE[nat]}=file:{p}" for (L, nat), p in sorted(idx.items()))
    res["b_reconstructions"] = {"files": {f"L{L}.{nat}": p for (L, nat), p in sorted(idx.items())},
                                **ev(spec, model, "B reconstructions")}
    nat = model_full.attn_index(native_dir)
    res["transposed_L6_wq_du"] = ev(
        f"model.layers.6.self_attn.q_proj.linear.weight=fileT:{nat[(6, 'wq_du')]}", model, "wq_du L6 transposed")
    res["swapped_L6_k_v"] = ev(f"model.layers.6.self_attn.k_proj.linear.weight=file:{nat[(6, 'wv_dv')]};"
                               f"model.layers.6.self_attn.v_proj.linear.weight=file:{nat[(6, 'wk_dv')]}",
                               model, "wk_dv and wv_dv of L6 swapped")
    res["ok"] = (res["native_files_L0_L6"]["kl_mean"] == 0.0 and res["native_files_L0_L6"]["kl_max"] == 0.0
                 and 0.0 < res["b_reconstructions"]["kl_mean"] < res["transposed_L6_wq_du"]["kl_mean"]
                 and res["b_reconstructions"]["kl_mean"] < res["swapped_L6_k_v"]["kl_mean"])
    return res


def write_f16_raw(path, name, arr16):
    """One f16 raw tensor in the sealed stream's framing (tag 0), as `llvq-artifact` `write_raw` writes it."""
    nb = name.encode()
    with open(path + ".tmp", "wb") as f:
        f.write(struct.pack("<II", 0, len(nb)) + nb + struct.pack("<I", arr16.ndim))
        f.write(b"".join(struct.pack("<Q", d) for d in arr16.shape))
        f.write(struct.pack("<Q", arr16.size))
        f.write(np.ascontiguousarray(arr16, dtype="<f2").tobytes())
    os.replace(path + ".tmp", path)


def decode_check(path, native_name):
    """Every decoded value within half a step (plus f16 rounding of the bound) of the f16-narrowed native value."""
    from safetensors import safe_open

    sealed, dims, deq = model_full.read_llvqraw(path)
    with safe_open(os.path.join(TEXTONLY, "model.safetensors"), framework="pt") as f:
        x16 = f.get_tensor(native_name).to(torch.float16).float().numpy()
    assert list(x16.shape) == dims, (x16.shape, dims)
    g = 64
    xs = x16.reshape(dims[0], -1, g)
    ds = deq.reshape(dims[0], -1, g)
    mn, mx = xs.min(-1), xs.max(-1)
    # the encoder's scale and bias (embedquant::quantize_affine): f16((max - min) / 15), f16(min); 1 for a flat group
    step = ((mx - mn) / np.float32(15)).astype(np.float32).astype(np.float16).astype(np.float32)
    step = np.where(step > 0, step, np.float32(1))
    b16 = mn.astype(np.float16).astype(np.float32)
    err = np.abs(ds - xs)
    bound = step / 2 * (1 + 1e-2) + np.abs(mn - b16) + 1e-7
    viol = err > bound[..., None]
    rel = float(np.linalg.norm(ds - xs) / np.linalg.norm(xs))
    shift = float(np.linalg.norm(ds[1:] - xs[:-1]) / np.linalg.norm(xs[:-1]))  # off-by-one-row control
    return {"file": path, "sealed_name": sealed, "dims": dims, "values": int(err.size),
            "values_beyond_half_step": int(viol.sum()), "max_err": float(err.max()), "rel_err": rel,
            "rel_err_if_rows_shifted_by_one": shift}


def gate_table(tdir):
    res = {"decode": [decode_check(os.path.join(tdir, "embed.q4g64.llvqraw"), "model.llm.embed.weight"),
                      decode_check(os.path.join(tdir, "unembed.q4g64.llvqraw"), "model.llm.unembed.weight")]}
    print(json.dumps(res["decode"]), flush=True)
    model = load_model("mps")
    from safetensors import safe_open

    with safe_open(os.path.join(TEXTONLY, "model.safetensors"), framework="pt") as f:
        x = f.get_tensor("model.llm.embed.weight")
    x32 = x.float().numpy()
    x16 = x.to(torch.float16).numpy()
    res["native_f16_narrowing_exact"] = bool(np.array_equal(x16.astype(np.float32), x32))
    tmp = os.path.join(OUT, "gw", "embed.native.f16.llvqraw")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    write_f16_raw(tmp, "model.embed.weight", x16)
    del x, x32, x16
    try:
        res["native_embed_via_raw"] = ev(f"model.embed_tokens.weight=table:{tmp}", model, "native embed via f16 raw")
    finally:
        os.unlink(tmp)
    e, u = os.path.join(tdir, "embed.q4g64.llvqraw"), os.path.join(tdir, "unembed.q4g64.llvqraw")
    res["b_embed"] = ev(f"model.embed_tokens.weight=table:{e}", model, "B embed q4g64")
    res["b_unembed"] = ev(f"lm_head.weight=table:{u}", model, "B unembed q4g64")
    try:
        model_full.tensor_plan(model_full.parse_student(f"lm_head.weight=table:{e}"), model_full.tensor_names(model))
        model_full._read_table(f"table:{e}", "lm_head.weight", (200058, 2304))
        res["misassigned_refused"] = False
    except ValueError as err:
        res["misassigned_refused"] = str(err)
    res["misassigned_forced_embed_as_lm_head"] = ev(f"lm_head.weight=table!:{e}", model, "embed forced as lm_head")
    exact = res["native_f16_narrowing_exact"]
    res["ok"] = (all(d["values_beyond_half_step"] == 0 for d in res["decode"])
                 and (res["native_embed_via_raw"]["kl_mean"] == 0.0 or not exact)
                 and 0 < res["b_embed"]["kl_mean"] < res["misassigned_forced_embed_as_lm_head"]["kl_mean"]
                 and 0 < res["b_unembed"]["kl_mean"] and bool(res["misassigned_refused"]))
    return res


def gate_agree():
    ref = json.load(open(os.path.join(DATA, "eval", "arm_tetra_6-17.json")))
    spec = f"experts:6-17=dir:{os.path.join(DATA, 'layers')}"
    rows, _, fp, _ = evalfull.run_rows(spec, "mps", range(8), 2048)
    agg = evalfull.aggregate([rows[i] for i in range(8)], 10000)
    same = [rows[i]["kl_mean"] == ref["per_window"][i]["kl_mean"] for i in range(8)]
    return {"spec": spec, "kl_mean": agg["kl_mean"], "ref_kl_mean": ref["kl_mean"], "top1": agg["top1_agree"],
            "ref_top1": ref["top1_agree"], "per_window_kl_bit_equal": same, "ok": all(same)}


def main():
    t0 = time.time()
    mode = sys.argv[1]
    if mode == "attn":
        res = gate_attn(sys.argv[2], sys.argv[3])
    elif mode == "table":
        res = gate_table(sys.argv[2])
    elif mode == "agree":
        res = gate_agree()
    else:
        sys.exit(__doc__)
    res["sec"] = time.time() - t0
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, f"gates_whole_{mode}.json"), "w"), indent=1)
    print(json.dumps({"mode": mode, "ok": res["ok"], "sec": res["sec"]}), flush=True)


if __name__ == "__main__":
    main()
