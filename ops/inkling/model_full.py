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
"""Whole-model student of Inkling 2 Mini: any text tensor from any source, and lazy per-expert loading.

This module builds on `executor.py` (imported, never edited): its model loader, its layer-major `Teacher.run`, its
expert `Provider`. It adds three things.

1. A student mapping, `{key: source}`, parsed by `parse_student`. A key is either `experts:<layers>` (for example
   `experts:6-17`, `experts:all`) or an fnmatch pattern over the `transformers` parameter names of
   `Inkling2LoopForCausalLM` (for example `model.layers.*.self_attn.q_proj.linear.weight`, `lm_head.weight`). A source
   is `native`, `dir:<path>` or `gguf:<file>`. For experts the sources mean what `executor.Provider` means. For any
   other tensor, `dir:<path>` reads `<path>/<parameter name>.f32` or `.f16` (raw little-endian, the parameter's shape,
   `[out, in]` for a linear), and `gguf:<file>` reads the tensor that `ggufmap.py` mapped and verified (see there).
   Later keys win over earlier ones. A mapping is given inline (`k=v;k=v`) or as a JSON file.

2. `ExpertSource`: one expert of one layer at a time, for the decode path, with a byte-bounded LRU. Its values are
   the same as the full-layer path's, bit for bit (`uv run model_full.py selftest`): native experts go through the
   library's own `Inkling2LoopDequantizeExperts` and `Interleave`, which are elementwise per expert, on a one-expert
   slice; `dir:` reads a slice of the same raw file; `gguf:` calls `ggufx._load` on a one-expert range.

3. `LazyExperts`: `executor.ProviderExperts` with a forward that, when no full layer is set, fetches only the routed
   experts (decode) or the full layer (prefill, more than `FULL_LAYER_TOKENS` tokens). The per-expert arithmetic is the
   library's loop, verbatim, so a forward through `LazyExperts` computes what `Inkling2LoopExperts` computes.

Numerics: float32 everywhere, as `executor.py`.
"""

import _threads  # noqa: F401  (BLAS threads before numpy/torch)

import collections
import fnmatch
import json
import os
import sys

import numpy as np

from _threads import HIDDEN, INTER, NATIVE_FILE, NUM_EXPERTS

import executor  # noqa: E402  (patches the MoE block, applies torch threads)
from executor import Provider, ProviderExperts, _parse_layers, torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# Above this many routed (token, slot) pairs a visit loads the whole layer instead of single experts.
FULL_LAYER_TOKENS = 16


# ---------------------------------------------------------------- the student mapping


# native attention name (B's `inkwh` records: `model.layers.{L}.attn.{nat}.weight`) -> transformers suffix
ATTN_NATIVE = {
    "wq_du": "self_attn.q_proj.linear.weight",
    "wk_dv": "self_attn.k_proj.linear.weight",
    "wv_dv": "self_attn.v_proj.linear.weight",
    "wo_ud": "self_attn.o_proj.linear.weight",
    "gate_attn": "self_attn.o_proj.gate_proj.weight",
    "wr_du": "self_attn.relative_position_bias.linear.weight",
}
# sealed table name (B's `inktable`, through `inkseal::rename`) -> transformers parameter
TABLE_PARAM = {"model.embed.weight": "model.embed_tokens.weight", "model.unembed.weight": "lm_head.weight"}
_SOURCES = ("dir:", "gguf:", "attn:", "table:", "table!:", "file:", "fileT:")


def _check_source(src):
    if not (src == "native" or src.startswith(_SOURCES)):
        raise ValueError(f"unknown source {src!r}")
    return src


def parse_student(spec):
    """`"experts:6-17=dir:/p;attention:all=attn:/a;model.embed_tokens.weight=table:/e.llvqraw"` or a JSON file ->
    ordered list of (key, source). Keys: `experts:<layers>`, `attention:<layers>` (the six attention matrices of
    `ATTN_NATIVE`, source `attn:<dir>` or `native`), `nonexperts` (every non-expert tensor, for a whole GGUF), or an
    fnmatch pattern over parameter and buffer names. Sources for a pattern: `native`, `dir:<d>` (`<d>/<name>.f32`),
    `gguf:<file>` (through `ggufmap`), `table:<file.llvqraw>` (B's int4 g64 table, or an f16 raw tensor; its sealed
    name must be the parameter's, `table!:` skips that check, for the negative control only), `file:<f.f32>` (one raw
    f32 file in the parameter's shape) and `fileT:<f.f32>` (the same file read transposed: negative control only)."""
    if not spec or spec == "native":
        return []
    if os.path.isfile(spec) and spec.endswith(".json"):
        items = list(json.load(open(spec)).items())
    else:
        items = [tuple(part.split("=", 1)) for part in spec.split(";") if part.strip()]
    out = []
    for k, v in items:
        k, v = k.strip(), _check_source(v.strip())
        if k.startswith("experts:"):
            _parse_layers(k[len("experts:"):])  # validates
            if not (v == "native" or v.startswith(("dir:", "gguf:"))):
                raise ValueError(f"{k}: experts take native, dir: or gguf:, not {v!r}")
        elif k.startswith("attention:"):
            _parse_layers(k[len("attention:"):])
            if not (v == "native" or v.startswith("attn:")):
                raise ValueError(f"{k}: attention takes native or attn:<dir>, not {v!r}")
        elif v.startswith("attn:"):
            raise ValueError(f"{k}: attn:<dir> goes with an attention:<layers> key")
        out.append((k, v))
    return out


def expert_arm(student):
    """The `executor` arm {layer: source} of a student; later keys win."""
    arm = {}
    for k, v in student:
        if k.startswith("experts:"):
            for L in _parse_layers(k[len("experts:"):]):
                arm[L] = v
    return {L: s for L, s in arm.items() if s != "native"}


def attn_index(root):
    """{(layer, native name): path} of every `model.layers.{L}.attn.{nat}.weight.f32` under `root` (recursive)."""
    import re

    pat = re.compile(r"^model\.layers\.(\d+)\.attn\.(\w+)\.weight\.f32$")
    idx = {}
    for dp, _, fns in os.walk(root):
        for fn in fns:
            m = pat.match(fn)
            if m and m.group(2) in ATTN_NATIVE:
                key = (int(m.group(1)), m.group(2))
                if key in idx:
                    raise KeyError(f"{fn}: found twice under {root} ({idx[key]}, {os.path.join(dp, fn)})")
                idx[key] = os.path.join(dp, fn)
    return idx


def tensor_plan(student, param_names):
    """{parameter name: source} for every non-expert tensor a key names (later keys win)."""
    plan = {}
    idx_cache = {}
    for k, v in student:
        if k.startswith("experts:"):
            continue
        if k.startswith("attention:"):
            for L in _parse_layers(k[len("attention:"):]):
                for nat, suffix in ATTN_NATIVE.items():
                    name = f"model.layers.{L}.{suffix}"
                    if v == "native":
                        plan[name] = "native"
                        continue
                    root = v[len("attn:"):]
                    if root not in idx_cache:
                        idx_cache[root] = attn_index(root)
                    path = idx_cache[root].get((L, nat))
                    if path is None:
                        raise FileNotFoundError(f"model.layers.{L}.attn.{nat}.weight.f32 not under {root}")
                    plan[name] = "file:" + path
            continue
        if k == "nonexperts":
            hits = [n for n in param_names if ".mlp.experts." not in n]
        else:
            hits = [n for n in param_names if fnmatch.fnmatchcase(n, k)]
        if not hits:
            raise KeyError(f"pattern {k!r} names no parameter")
        for n in hits:
            if ".mlp.experts." in n:
                raise KeyError(f"{n}: experts go through an experts:<layers> key")
            plan[n] = v
    return {n: s for n, s in plan.items() if s != "native"}


def read_llvqraw(path):
    """(name, dims, float32 array) of one raw tensor in the sealed stream's tagged framing
    (`llvq-artifact/src/sealed.rs` `write_raw`, version >= 3): tag u32 (0 f16, 1 affine quant), name (u32 length +
    UTF-8), rank u32, dims u64 each; f16: count u64 + u16 values; quant: bits u32, group u32, packed length u64 +
    bytes, group count u64, scales u16 x groups, biases u16 x groups, all little-endian. The decode is
    `RawTensor::to_f32`: `f16(scale) * q + f16(bias)` in float32, groups of `group` along the last dimension, two
    4-bit codes a byte, the even element in the low nibble."""
    import struct

    buf = np.memmap(path, dtype=np.uint8, mode="r")
    pos = 0

    def rd(fmt):
        nonlocal pos
        v = struct.unpack_from(fmt, buf, pos)
        pos += struct.calcsize(fmt)
        return v[0]

    tag = rd("<I")
    n_name = rd("<I")
    name = bytes(buf[pos : pos + n_name]).decode()
    pos += n_name
    dims = [rd("<Q") for _ in range(rd("<I"))]
    n = int(np.prod(dims))
    row_len = dims[-1]
    if tag == 0:
        cnt = rd("<Q")
        assert cnt == n, (path, cnt, n)
        a = np.frombuffer(buf, dtype="<f2", count=n, offset=pos).astype(np.float32)
        pos += 2 * n
    elif tag == 1:
        bits, group = rd("<I"), rd("<I")
        nb = rd("<Q")
        packed = np.frombuffer(buf, dtype=np.uint8, count=nb, offset=pos)
        pos += nb
        ng = rd("<Q")
        scales = np.frombuffer(buf, dtype="<f2", count=ng, offset=pos).astype(np.float32)
        pos += 2 * ng
        biases = np.frombuffer(buf, dtype="<f2", count=ng, offset=pos).astype(np.float32)
        pos += 2 * ng
        if bits == 4:
            q = np.empty(2 * nb, np.uint8)
            q[0::2] = packed & 0xF
            q[1::2] = packed >> 4
            q = q[:n]
        elif bits == 8:
            q = np.asarray(packed[:n])
        else:
            raise ValueError(f"{path}: {bits}-bit quant")
        if row_len % group:
            raise ValueError(f"{path}: row {row_len} not a multiple of group {group} (short groups not handled)")
        gpr = row_len // group
        rows = n // row_len
        assert ng == rows * gpr, (path, ng, rows, gpr)
        a = (scales.reshape(rows, gpr, 1) * q.reshape(rows, gpr, group).astype(np.float32)
             + biases.reshape(rows, gpr, 1)).reshape(-1)
    else:
        raise ValueError(f"{path}: raw tag {tag}")
    if pos != buf.size:
        raise ValueError(f"{path}: {buf.size - pos} trailing bytes")
    return name, dims, a.reshape(dims)


def _read_table(src, name, shape):
    force = src.startswith("table!:")
    path = src.split(":", 1)[1]
    sealed, dims, a = read_llvqraw(path)
    if not force and TABLE_PARAM.get(sealed) != name:
        raise ValueError(f"{path} holds {sealed!r}, which is not {name!r} (table!: to force, negative control only)")
    if len(dims) != len(shape) or dims[1:] != list(shape[1:]) or dims[0] < shape[0]:
        raise ValueError(f"{path}: dims {dims} do not cover {name} {list(shape)}")
    return torch.from_numpy(np.ascontiguousarray(a[: shape[0]]))


def _read_file(src, shape):
    transposed = src.startswith("fileT:")
    path = src.split(":", 1)[1]
    a = np.fromfile(path, dtype=np.float32)
    if a.size != int(np.prod(shape)):
        raise ValueError(f"{path}: {a.size} values, parameter shape {list(shape)}")
    if transposed:
        assert len(shape) == 2
        a = np.ascontiguousarray(a.reshape(shape[1], shape[0]).T)
    return torch.from_numpy(a.reshape(shape))


def _read_dir_tensor(d, name, shape):
    for ext, dt in ((".f32", np.float32), (".f16", np.float16)):
        p = os.path.join(d, name + ext)
        if os.path.exists(p):
            a = np.fromfile(p, dtype=dt)
            if a.size != int(np.prod(shape)):
                raise ValueError(f"{p}: {a.size} values, parameter {name} has shape {list(shape)}")
            return torch.from_numpy(a.reshape(shape).astype(np.float32))
    raise FileNotFoundError(os.path.join(d, name) + ".{f32,f16}")


def tensor_names(model):
    """Every parameter and buffer name of the model (buffers: the routers' `e_score_correction_bias`)."""
    return [n for n, _ in model.named_parameters()] + [n for n, _ in model.named_buffers()]


def apply_tensor_plan(model, plan, log=print):
    """Overwrite the named parameters or buffers in place. Returns {name: {source, max_abs_change}}."""
    params = dict(model.named_parameters()) | dict(model.named_buffers())
    report = {}
    gg = {}
    for name, src in sorted(plan.items()):
        p = params[name]
        if src.startswith("dir:"):
            new = _read_dir_tensor(src[4:], name, tuple(p.shape))
        elif src.startswith(("table:", "table!:")):
            new = _read_table(src, name, tuple(p.shape))
        elif src.startswith(("file:", "fileT:")):
            new = _read_file(src, tuple(p.shape))
        elif src.startswith("gguf:"):
            import ggufmap

            path = src[5:]
            if path not in gg:
                gg[path] = ggufmap.verified_map(path)
            new = ggufmap.load_for_param(path, gg[path], name, tuple(p.shape))
        else:
            raise ValueError(src)
        new = new.to(device=p.device, dtype=p.dtype)
        change = float((new - p.data).abs().max())
        p.data.copy_(new)
        report[name] = {"source": src, "max_abs_change": change}
    if log and report:
        log(f"  substituted {len(report)} tensors")
    return report


def snapshot(model, names):
    """CPU copies of the named parameters or buffers, for `restore`."""
    params = dict(model.named_parameters()) | dict(model.named_buffers())
    return {n: params[n].detach().cpu().clone() for n in names}


def restore(model, snap):
    params = dict(model.named_parameters()) | dict(model.named_buffers())
    for n, t in snap.items():
        params[n].data.copy_(t.to(params[n].device))


class FastProvider(Provider):
    """`executor.Provider` whose `gguf:` layers are dequantized by `ggufx._load` on chunks of 8 experts in a thread
    pool (the same calls as `ggufx.load_gguf_experts`, so the same values); other sources as `Provider`."""

    def __init__(self, arm=None, device="cpu", workers=None):
        super().__init__(arm, device)
        self.workers = workers or max(1, _threads.THREADS)

    def get(self, layer):
        src = self.source(layer)
        if not src.startswith("gguf:"):
            return super().get(layer)
        from concurrent.futures import ThreadPoolExecutor

        import ggufx

        path = src[5:]
        w13 = np.empty((NUM_EXPERTS, 2 * INTER, HIDDEN), np.float32)
        w2 = np.empty((NUM_EXPERTS, HIDDEN, INTER), np.float32)

        def job(e0):
            w13[e0 : e0 + 8] = ggufx._load(path, f"blk.{layer}.ffn_gate_up_exps.weight", e0, e0 + 8)
            w2[e0 : e0 + 8] = ggufx._load(path, f"blk.{layer}.ffn_down_exps.weight", e0, e0 + 8)

        with ThreadPoolExecutor(self.workers) as ex:
            list(ex.map(job, range(0, NUM_EXPERTS, 8)))
        return torch.from_numpy(w13).to(self.device), torch.from_numpy(w2).to(self.device)


# ---------------------------------------------------------------- per-expert loading


class ExpertSource:
    """One (layer, expert) pair at a time, float32 on `device`, with an LRU bounded in bytes."""

    def __init__(self, arm=None, device="cpu", cache_gb=12.0):
        self.arm = arm or {}
        self.device = device
        self.full = Provider(self.arm, device)
        self.cap = int(cache_gb * 2**30)
        self.lru = collections.OrderedDict()
        self.bytes = 0
        self.stats = {"expert_hits": 0, "expert_misses": 0, "full_layer_loads": 0}
        self._st = None

    def source(self, layer):
        return self.arm.get(layer, "native")

    def layer(self, layer):
        self.stats["full_layer_loads"] += 1
        return self.full.get(layer)

    def get(self, layer):
        """The full-layer interface of `executor.Provider`, so a `Teacher` can use this object too."""
        return self.layer(layer)

    def _native(self, layer, e):
        from safetensors import safe_open
        from transformers.core_model_loading import Inkling2LoopDequantizeExperts, Interleave

        import nvfp4

        if self._st is None:
            self._st = safe_open(NATIVE_FILE, framework="pt")
        f = self._st
        keys = set(f.keys())
        shim = nvfp4._DtypeShim(torch.float32)
        out = []
        for mat in ("w13", "w2"):
            base = nvfp4.PREFIX.format(layer=layer) + f"{mat}_weight"
            names = [base, base + "_scale", base + "_amax", base + "_outer_scale_denominator"]
            t = {}
            for n in names:
                if n not in keys:
                    continue
                s = f.get_slice(n)
                # the denominator is one scalar for all experts; every other tensor has the expert axis first
                t[n] = s[:] if n.endswith("_outer_scale_denominator") else s[e : e + 1]
            res = Inkling2LoopDequantizeExperts().convert(
                t, source_patterns=names, target_patterns=["t"], model=shim, full_layer_name="t"
            )["t"]
            if mat == "w13":
                res = Interleave(dim=1).convert({"t": res}, source_patterns=["t"], target_patterns=["t"])["t"]
            out.append(res[0].contiguous())
        return out[0], out[1]

    def _dir(self, root, layer, e):
        d = os.path.join(root, f"L{layer:02d}")
        res = []
        for stem, shape in (("w13_hat", (2 * INTER, HIDDEN)), ("w2_hat", (HIDDEN, INTER))):
            for ext, dt in ((".f32", np.float32), (".f16", np.float16)):
                p = os.path.join(d, stem + ext)
                if os.path.exists(p):
                    n = shape[0] * shape[1]
                    a = np.fromfile(p, dtype=dt, count=n, offset=e * n * np.dtype(dt).itemsize)
                    res.append(torch.from_numpy(a.reshape(shape).astype(np.float32)))
                    break
            else:
                raise FileNotFoundError(os.path.join(d, stem) + ".{f32,f16}")
        return res[0], res[1]

    def _gguf(self, path, layer, e):
        import ggufx

        w13 = ggufx._load(path, f"blk.{layer}.ffn_gate_up_exps.weight", e, e + 1)[0]
        w2 = ggufx._load(path, f"blk.{layer}.ffn_down_exps.weight", e, e + 1)[0]
        return torch.from_numpy(np.ascontiguousarray(w13)), torch.from_numpy(np.ascontiguousarray(w2))

    def expert(self, layer, e):
        key = (layer, int(e))
        hit = self.lru.get(key)
        if hit is not None:
            self.lru.move_to_end(key)
            self.stats["expert_hits"] += 1
            return hit
        self.stats["expert_misses"] += 1
        src = self.source(layer)
        if src == "native":
            w13, w2 = self._native(layer, int(e))
        elif src.startswith("dir:"):
            w13, w2 = self._dir(src[4:], layer, int(e))
        elif src.startswith("gguf:"):
            w13, w2 = self._gguf(src[5:], layer, int(e))
        else:
            raise ValueError(src)
        assert tuple(w13.shape) == (2 * INTER, HIDDEN) and tuple(w2.shape) == (HIDDEN, INTER), (w13.shape, w2.shape)
        pair = (w13.float().contiguous().to(self.device), w2.float().contiguous().to(self.device))
        size = (pair[0].numel() + pair[1].numel()) * 4
        self.lru[key] = pair
        self.bytes += size
        while self.bytes > self.cap and len(self.lru) > 1:
            _, old = self.lru.popitem(last=False)
            self.bytes -= (old[0].numel() + old[1].numel()) * 4
        return pair


class LazyExperts(ProviderExperts):
    """`ProviderExperts` whose weights, when no full layer is set, come from an `ExpertSource`."""

    def forward(self, hidden_states, top_k_index, top_k_weights):
        if self.w13 is not None or not isinstance(self.provider, ExpertSource):
            return super().forward(hidden_states, top_k_index, top_k_weights)
        if self.recorder is not None:
            self.recorder(hidden_states, top_k_index, top_k_weights)
        full = None
        if top_k_index.numel() > FULL_LAYER_TOKENS:
            full = self.provider.layer(self.layer_idx)
        # verbatim: Inkling2LoopExperts.forward (modeling_inkling2_loop.py 787-807), weights fetched per expert
        final_hidden_states = torch.zeros_like(hidden_states, dtype=torch.float32)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            if full is not None:
                w13, w2 = full[0][expert_idx], full[1][expert_idx]
            else:
                w13, w2 = self.provider.expert(self.layer_idx, int(expert_idx))
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            current_hidden_states = self._apply_gate(F.linear(current_state, w13))
            current_hidden_states = F.linear(current_hidden_states, w2)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states.to(hidden_states.dtype)


def install_lazy(model, source):
    """Swap every layer's `ProviderExperts` for a `LazyExperts` fed by `source` (no weight is copied)."""
    for i, layer in enumerate(model.model.layers):
        old = layer.mlp.experts
        new = LazyExperts(model.config)
        new.act_fn = old.act_fn
        new.layer_idx = i
        new.provider = source
        layer.mlp.experts = new.to(next(model.parameters()).device)
    return model


# ---------------------------------------------------------------- self-test


def selftest(device="cpu", layers=(0, 23), experts=(0, 1, 127)):
    """Per-expert values equal the full-layer values bit for bit, for native layers (NVFP4 and MXFP8)."""
    src = ExpertSource({}, device="cpu", cache_gb=1.0)
    rows = []
    for L in layers:
        w13, w2 = src.layer(L)
        for e in experts:
            a13, a2 = src.expert(L, e)
            rows.append({"layer": L, "expert": e, "w13_equal": bool(torch.equal(a13, w13[e])),
                         "w2_equal": bool(torch.equal(a2, w2[e]))})
        del w13, w2
    ok = all(r["w13_equal"] and r["w2_equal"] for r in rows)
    print(json.dumps({"selftest": "per-expert == full layer", "ok": ok, "rows": rows}))
    return ok


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "selftest":
        sys.exit(0 if selftest() else 1)
    print(__doc__)
