# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch",
#   "numpy",
#   "safetensors",
#   "tokenizers",
#   "pyarrow",
#   "transformers @ git+https://github.com/huggingface/transformers@046bdb8cc2",
# ]
# ///
"""The attention capture of `capture_attn.py` in resumable slices of sequences, folded into one set of Hessians.

    uv run attn_slices.py slice --out DIR --seqs A..B [--n 64] [--len 2048] [--layers all] [--fold]  # heavy
    uv run attn_slices.py fold  --out DIR      # fold the complete partials that extend the coverage (heavy: IO)
    uv run attn_slices.py merge --out DIR      # final: refuses any gap, writes DIR/L{xx}_{stream}.f64 + manifests
    uv run attn_slices.py done  --out DIR --seqs A..B   # exit 0 when that slice is folded or its partial complete
    uv run attn_slices.py whprep --out DIR [--layers all]   # native attention weights as raw f32, for inkwh

Slice. Sequences [A, B) of the `--n` calibration windows (`data.calib_windows(tok, n, len)`, always tokenized for the
whole `n`, so a slice sees the same tokens whatever the slicing) run through `capture_attn.capture`. The result is a
partial, `DIR/partials/sAAA-BBB/`: one float64 Gram file per (layer, stream) and `partial.json` with the range, the
settings, the token counts, the per-visit sums and the sha256 of every file. It is written under
`sAAA-BBB.tmp-<pid>` and renamed into place, so a final name is complete by construction. A slice whose partial (or
whose fold) is already there is skipped. A leftover `.tmp-*` directory, or a final one whose `partial.json` is missing,
unreadable, or does not match its files, is torn: it is removed and the slice is redone.

Settings that must agree across partials and the accumulator: corpus fingerprint (sha256 of the n windows' uint32
tokens), n, len, layers, streams, dtype, device, chunk, micro, model revision. A mismatch is refused.

Fold. The accumulator is `DIR/acc/`: `ledger.json` (settings, folded ranges in order, `k` the number of folds, a
`pending` range while a fold runs) and one file a (layer, stream), `acc/L{xx}_{stream}.k{k}.f64`. Only the partial
that starts where the coverage ends is folded, so the folding order is the sequence order whatever order the slices
ran in, and the result is reproducible bit for bit. A fold first records `pending`, then turns each file from `k` to
`k+1` (write tmp, rename, unlink the old one; a file already at `k+1` is skipped), then appends the range, clears
`pending`, and deletes the partial. A fold killed midway resumes from `pending`. Disk peak: accumulator + one partial +
one file (9.8 GB for all 24 layers). A partial whose range is already folded (duplicate) or overlaps the coverage or
another partial is refused. A partial whose files do not match their sha256 is torn: removed, slice to redo.

Merge. Folds what can be folded, then refuses unless the folded ranges cover [0, n) exactly with nothing pending, and
renames the files to `DIR/L{xx}_{stream}.f64` with a `manifest.json` that `capture_attn.py check` reads, and one
`inkwh` manifest a group, `DIR/wh/L{xx}_{q|k|v|o}.json` (see `wh_manifests`).

Numerics. A partial's Gram is the float32-block, float64-sum of `capture_attn.Accumulator`, started from zero. The
fold adds partials in sequence order: acc_{k+1} = acc_k + P_k, float64. A single run of the same sequences adds the
per-sequence blocks one by one into one sum. The two agree bit for bit when every slice holds one sequence (the
additions are then the same, in the same order); with several sequences a slice, P_k is summed separately before it is
added, and the two differ only by float64 reassociation (relative error at most about n_slices x 2^-52 of the sum of
absolute values).
"""

import _threads  # noqa: F401

import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np

from _threads import DATA, NATIVE, TEXTONLY

import capture_attn  # noqa: E402  (light: imports torch lazily)
from capture_attn import STREAMS  # noqa: E402

SHORT = {"query": "q", "key": "k", "value": "v", "gated_out": "o"}
# which native matrix reads which stream, and how the plan (docs/plan-inkling2-mini.md, section 1) stores it
WH_MATRICES = {
    "query": [("wq_du", "tetra"), ("wr_du", "int4"), ("gate_attn", "int4")],
    "key": [("wk_dv", "tetra")],
    "value": [("wv_dv", "int4")],
    "gated_out": [("wo_ud", "int4")],
}
SETTING_KEYS = ("config", "corpus_fingerprint", "n", "len", "layers", "streams", "dtype", "device", "chunk", "micro", "model")


class Refused(RuntimeError):
    pass


def parse_seqs(s):
    a, b = s.split("..")
    a, b = int(a), int(b)
    if not 0 <= a < b:
        raise ValueError(f"bad range {s}")
    return a, b


def _sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def _write_json_atomic(p, obj):
    tmp = p + f".tmp-{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def _write_array(p, a):
    with open(p, "wb") as f:
        f.write(a.tobytes())
        f.flush()
        os.fsync(f.fileno())


def stem(L, s):
    return f"L{L:02d}_{s}"


def pdir(out, a, b):
    return os.path.join(out, "partials", f"s{a:03d}-{b:03d}")


# ---------------------------------------------------------------- partials


def read_partial(d, deep=False):
    """partial.json of a final partial directory if it is complete, else None. `deep` also checks every sha256."""
    try:
        meta = json.load(open(os.path.join(d, "partial.json")))
        for fn, f in meta["files"].items():
            p = os.path.join(d, fn)
            if os.path.getsize(p) != f["bytes"]:
                return None
            if deep and _sha_file(p) != f["sha256"]:
                return None
        return meta
    except (OSError, ValueError, KeyError):
        return None


def clean_torn(out):
    """Remove leftover temporary partials and final partials that are not complete. Returns what was removed."""
    removed = []
    for d in sorted(glob.glob(os.path.join(out, "partials", "*"))):
        if ".tmp-" in os.path.basename(d) or read_partial(d) is None:
            shutil.rmtree(d, ignore_errors=True)
            removed.append(os.path.basename(d))
    return removed


def write_partial(out, a, b, acc, settings):
    final = pdir(out, a, b)
    tmp = final + f".tmp-{os.getpid()}"
    os.makedirs(tmp)
    files = {}
    for (L, s), g in sorted(acc.gram.items()):
        fn = stem(L, s) + ".f64"
        _write_array(os.path.join(tmp, fn), g)
        files[fn] = {"layer": L, "stream": s, "shape": list(g.shape), "bytes": g.nbytes,
                     "tokens": acc.tokens[(L, s)], "sha256": hashlib.sha256(g.tobytes()).hexdigest()}
    per_visit = {f"{L}|{p}|{s}": {"tokens": r["tokens"], "sum": r["sum"], "sumsq": r["sumsq"],
                                  "sha256_x_f32": r["h"].hexdigest() if acc.sha else None}
                 for (L, p, s), r in sorted(acc.per_visit.items())}
    _write_json_atomic(os.path.join(tmp, "partial.json"),
                       {"range": [a, b], "settings": settings, "files": files, "per_visit": per_visit,
                        "written": time.strftime("%F %T")})
    os.rename(tmp, final)
    return final


def settings_of(a, fingerprint, cfg_layers, config=None):
    config = config or _config_block(json.load(open(os.path.join(TEXTONLY, "config.json"))))
    return {"config": config, "corpus_fingerprint": fingerprint, "n": a.n, "len": a.len, "layers": cfg_layers,
            "streams": list(STREAMS), "dtype": "float32", "device": a.device, "chunk": a.chunk, "micro": 1,
            "model": "thinkingmachines/Inkling-2-Mini@ed40e7da625f594fc9c4990f5f66be24873c22d7"}


def corpus(n, length):
    from tokenizers import Tokenizer

    import data

    tok = Tokenizer.from_file(os.path.join(NATIVE, "tokenizer.json"))
    windows, stats = data.calib_windows(tok, n, length)
    fp = hashlib.sha256(np.asarray(windows, dtype=np.uint32).tobytes()).hexdigest()
    return windows, stats, fp


def run_slices(teacher, out, ranges, windows, settings, sha=True, log=print):
    """Capture each range not yet folded nor complete. `teacher` is an executor.Teacher (real or tiny)."""
    import torch

    os.makedirs(os.path.join(out, "partials"), exist_ok=True)
    removed = clean_torn(out)
    if removed and log:
        log(f"torn partials removed: {removed}")
    ids_all = torch.tensor(windows, dtype=torch.long)
    done = []
    for a, b in ranges:
        if is_done(out, a, b):
            if log:
                log(f"slice {a}..{b}: done, skipped")
            continue
        t0 = time.time()
        acc, _ = capture_attn.capture(teacher, ids_all[a:b], settings["layers"], group=b - a, micro=1,
                                      chunk=settings["chunk"], sha=sha, log=None)
        write_partial(out, a, b, acc, settings)
        done.append((a, b))
        if log:
            log(f"slice {a}..{b}: captured in {time.time() - t0:.0f}s")
    return done


def is_done(out, a, b):
    led = read_ledger(out)
    if led is not None and [a, b] in led["folded"]:
        return True
    return read_partial(pdir(out, a, b)) is not None


# ---------------------------------------------------------------- fold and merge


def read_ledger(out):
    p = os.path.join(out, "acc", "ledger.json")
    return json.load(open(p)) if os.path.exists(p) else None


def _coverage_end(led):
    end = 0
    for a, b in led["folded"]:
        if a != end:
            raise Refused(f"ledger not contiguous at {a} (coverage ends at {end})")
        end = b
    return end


def _acc_path(out, fn, k):
    return os.path.join(out, "acc", fn[:-4] + f".k{k}.f64")


def _fold_one(out, led, d, meta, log):
    """Fold partial `d` (already checked) into the accumulator at k -> k+1."""
    k = led["k"]
    led["pending"] = meta["range"]
    _write_json_atomic(os.path.join(out, "acc", "ledger.json"), led)
    for fn, f in sorted(meta["files"].items()):
        nxt = _acc_path(out, fn, k + 1)
        cur = _acc_path(out, fn, k)
        if os.path.exists(nxt):
            if os.path.exists(cur):
                os.unlink(cur)
            continue
        p = np.fromfile(os.path.join(d, fn), np.float64)
        base = np.fromfile(cur, np.float64) if k > 0 else np.zeros_like(p)
        if base.size != p.size:
            raise Refused(f"{fn}: accumulator holds {base.size} values, partial {p.size}")
        tmp = nxt + f".tmp-{os.getpid()}"
        _write_array(tmp, base + p)
        os.rename(tmp, nxt)
        if k > 0:
            os.unlink(cur)
    for key, r in meta["per_visit"].items():
        row = led["per_visit"].setdefault(key, {"tokens": 0, "sum": 0.0, "sumsq": 0.0, "slice_sha256": []})
        row["tokens"] += r["tokens"]
        row["sum"] += r["sum"]
        row["sumsq"] += r["sumsq"]
        row["slice_sha256"].append(r["sha256_x_f32"])
    for fn, f in meta["files"].items():
        led["tokens"][fn] = led["tokens"].get(fn, 0) + f["tokens"]
    led["folded"].append(meta["range"])
    led["pending"] = None
    led["k"] = k + 1
    _write_json_atomic(os.path.join(out, "acc", "ledger.json"), led)
    shutil.rmtree(d)
    if log:
        log(f"folded {meta['range']} -> k={k + 1}")


def fold(out, log=print):
    """Fold every complete partial that extends the coverage, in sequence order. Refuses duplicates, overlaps and
    settings mismatches; removes torn partials (they must be captured again)."""
    os.makedirs(os.path.join(out, "acc"), exist_ok=True)
    for t in glob.glob(os.path.join(out, "acc", "*.tmp-*")):  # a file of a fold that was killed before its rename
        os.unlink(t)
    removed = clean_torn(out)
    if removed and log:
        log(f"torn partials removed: {removed}")
    parts = {}
    for d in sorted(glob.glob(os.path.join(out, "partials", "s*"))):
        meta = read_partial(d)
        parts[tuple(meta["range"])] = (d, meta)
    led = read_ledger(out)
    if led is None:
        if not parts:
            return None
        first = next(iter(parts.values()))[1]
        led = {"settings": first["settings"], "folded": [], "k": 0, "pending": None, "per_visit": {}, "tokens": {}}
        _write_json_atomic(os.path.join(out, "acc", "ledger.json"), led)
    for (a, b), (d, meta) in parts.items():
        if meta["settings"] != led["settings"]:
            diff = {k: (meta["settings"].get(k), led["settings"].get(k)) for k in SETTING_KEYS
                    if meta["settings"].get(k) != led["settings"].get(k)}
            raise Refused(f"partial {a}..{b}: settings differ from the accumulator: {diff}")
        if [a, b] in led["folded"]:
            raise Refused(f"partial {a}..{b}: duplicate, this range is already folded")
        for fa, fb in led["folded"]:
            if a < fb and fa < b:
                raise Refused(f"partial {a}..{b}: overlaps the folded range {fa}..{fb}")
        for (oa, ob) in parts:
            if (oa, ob) != (a, b) and a < ob and oa < b:
                raise Refused(f"partial {a}..{b}: overlaps the partial {oa}..{ob}")
    if led["pending"] is not None:
        pa, pb = led["pending"]
        if (pa, pb) not in parts:
            raise Refused(f"fold of {pa}..{pb} was interrupted and its partial is gone: capture that slice again")
    while True:
        end = _coverage_end(led)
        nxt = led["pending"] if led["pending"] is not None else next(([a, b] for (a, b) in parts if a == end), None)
        if nxt is None:
            break
        d, meta = parts.pop(tuple(nxt))
        if read_partial(d, deep=True) is None:
            shutil.rmtree(d)
            if log:
                log(f"partial {nxt}: sha256 mismatch, torn, removed; capture it again")
            if led["pending"] is not None:
                raise Refused(f"pending fold {nxt} has a torn partial: capture that slice again")
            break
        _fold_one(out, led, d, meta, log)
    return led


def merge(out, log=print):
    led = fold(out, log)
    if led is None:
        raise Refused("nothing captured")
    n = led["settings"]["n"]
    end = _coverage_end(led)
    if led["pending"] is not None or end != n:
        gaps = f"covered [0, {end}) of [0, {n})"
        left = sorted(os.path.basename(d) for d in glob.glob(os.path.join(out, "partials", "s*")))
        raise Refused(f"merge refused: {gaps}; partials waiting: {left}")
    k = led["k"]
    files = {}
    s = led["settings"]
    for fn, tokens in sorted(led["tokens"].items()):
        src = _acc_path(out, fn, k)
        dst = os.path.join(out, fn)
        os.rename(src, dst)
        L, stream = int(fn[1:3]), fn[4:-4]
        width = int(round((os.path.getsize(dst) / 8) ** 0.5))
        files[fn] = {"layer": L, "stream": stream, "shape": [width, width], "tokens": tokens,
                     "sha256": _sha_file(dst)}
    per_visit = []
    for key, r in sorted(led["per_visit"].items(), key=lambda kv: tuple(int(x) if x.isdigit() else x
                                                                         for x in kv[0].split("|"))):
        L, p, stream = key.split("|")
        shas = r["slice_sha256"]
        per_visit.append({"layer": int(L), "pass": int(p), "stream": stream, "tokens": r["tokens"],
                          "sum": r["sum"], "sumsq": r["sumsq"],
                          "sha256_x_f32": hashlib.sha256("".join(shas).encode()).hexdigest()
                          if all(shas) else None})
    config = s["config"]
    manifest = {
        "model": s["model"], "settings": s, "n_seqs": n, "seq_len": s["len"], "n_tokens": n * s["len"],
        "layers": s["layers"], "streams": s["streams"], "readers": capture_attn.READERS, "config": config,
        "hessian": "unnormalised sum over visits and tokens of x x^T (weight 1 a token, so weight_sum = tokens); "
                   "float32 Gram blocks summed in float64; slices folded in sequence order",
        "slices": led["folded"], "files": files, "per_visit": per_visit,
        "per_visit_sha256": "sha256 of the concatenated per-slice sha256 of x, in slice order",
    }
    _write_json_atomic(os.path.join(out, "manifest.json"), manifest)
    wh_manifests(out, files)
    shutil.rmtree(os.path.join(out, "acc"))
    if log:
        log(f"merged {len(files)} files over {len(led['folded'])} slices into {out}")
    return manifest


def _config_block(cfg):
    loop = cfg.get("loop_config", {})
    n = cfg.get("num_hidden_layers", 24)
    types = cfg.get("layer_types") or ["hybrid" if (i + 1) % 6 == 0 else "hybrid_sliding" for i in range(n)]
    return {"num_hidden_layers": n, "loop_layer_index_start": loop.get("core_start", 6),
            "loop_layer_index_end": loop.get("core_end", 18), "loop_num_passes": loop.get("num_passes", 3),
            "layer_types": types}


def wh_manifests(out, files):
    """One `inkwh` manifest a (layer, stream): `llvq_llm::inkseal::run_wh` fields. `h` points at the merged file;
    the Hessian carries weight 1 a token, so `weight_sum` = `tokens`; `layer` + `stream` (q, k, v, o) derive the
    rotation seed (`inkseal::stream_seed`). `w` points at `DIR/w/<name>.f32`, which `whprep` writes."""
    d = os.path.join(out, "wh")
    os.makedirs(d, exist_ok=True)
    for fn, f in files.items():
        L, s = f["layer"], f["stream"]
        mats = []
        for nat, kind in WH_MATRICES[s]:
            name = f"model.layers.{L}.attn.{nat}.weight"
            mats.append({"name": name, "w": f"../w/{name}.f32", "kind": kind, "d_out": None})
        _write_json_atomic(os.path.join(d, f"L{L:02d}_{SHORT[s]}.json"),
                           {"n": f["shape"][0], "h": f"../{fn}", "weight_sum": float(f["tokens"]),
                            "tokens": f["tokens"], "layer": L, "stream": SHORT[s], "matrices": mats})
    shapes = os.path.join(out, "w", "shapes.json")
    if os.path.exists(shapes):
        fill_d_out(out)


def fill_d_out(out):
    """Write each matrix's `d_out` into the inkwh manifests from `w/shapes.json` (written by `whprep`)."""
    shp = json.load(open(os.path.join(out, "w", "shapes.json")))
    for p in sorted(glob.glob(os.path.join(out, "wh", "*.json"))):
        m = json.load(open(p))
        for x in m["matrices"]:
            if x["name"] not in shp:  # whprep ran on other layers; d_out stays unset until it covers this one
                continue
            out_in = shp[x["name"]]
            if out_in[1] != m["n"]:
                raise Refused(f"{x['name']}: d_in {out_in[1]} but the stream is {m['n']} wide")
            x["d_out"] = out_in[0]
        _write_json_atomic(p, m)


def whprep(out, layers):
    """The native attention matrices the inkwh manifests name, as raw little-endian float32 [out, in]."""
    import torch
    from safetensors import safe_open

    d = os.path.join(out, "w")
    os.makedirs(d, exist_ok=True)
    shapes = {}
    with safe_open(os.path.join(TEXTONLY, "model.safetensors"), framework="pt") as f:
        for L in layers:
            for s, mats in WH_MATRICES.items():
                for nat, _ in mats:
                    t = f.get_tensor(f"model.llm.layers.{L}.attn.{nat}.weight").to(torch.float32)
                    name = f"model.layers.{L}.attn.{nat}.weight"
                    _write_array(os.path.join(d, name + ".f32"), t.contiguous().numpy())
                    shapes[name] = list(t.shape)
    _write_json_atomic(os.path.join(d, "shapes.json"), shapes)
    if glob.glob(os.path.join(out, "wh", "*.json")):
        fill_d_out(out)
    print(json.dumps({"whprep": d, "matrices": len(shapes)}))


# ---------------------------------------------------------------- CLI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["slice", "fold", "merge", "done", "whprep"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seqs", default=None)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--len", type=int, default=2048)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--no-sha", action="store_true")
    ap.add_argument("--fold", action="store_true", help="fold after the slice, in the same process")
    a = ap.parse_args()
    try:
        if a.mode == "done":
            sys.exit(0 if is_done(a.out, *parse_seqs(a.seqs)) else 1)
        if a.mode == "fold":
            fold(a.out)
            return
        if a.mode == "merge":
            merge(a.out)
            return
        layers = list(range(24)) if a.layers == "all" else [int(x) for x in a.layers.split(",")]
        if a.mode == "whprep":
            whprep(a.out, layers)
            return
        from executor import Teacher, torch  # noqa: E402

        torch.set_grad_enabled(False)
        rng = parse_seqs(a.seqs)
        if rng[1] > a.n:
            raise Refused(f"range {a.seqs} beyond n={a.n}")
        windows, _, fp = corpus(a.n, a.len)
        settings = settings_of(a, fp, layers)
        led = read_ledger(a.out)
        if led is not None and led["settings"] != settings:
            raise Refused(f"settings differ from the accumulator in {a.out}")
        if is_done(a.out, *rng):
            print(f"slice {a.seqs}: done, skipped")
        else:
            teacher = Teacher(device=a.device)
            teacher.model.requires_grad_(False)
            run_slices(teacher, a.out, [rng], windows, settings, sha=not a.no_sha,
                       log=lambda m: print(m, flush=True))
        if a.fold:
            fold(a.out)
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
