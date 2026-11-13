# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "torch==2.5.1",
#   "safetensors==0.4.5",
#   "tokenizers==0.22.2",
#   "pyarrow==20.0.0",
#   "huggingface-hub==1.19.0",
# ]
# ///
"""The layer judge against the final-output judge, by projection group, on the sealed 4B.

Prereg: proofs/preregistration-layer-vs-output-4b-2026-11-13.md. Cost note:
docs/mesures/recherche-qualite-r2-chiffrage-2026-11-13.txt.

One process, two weight sets resident on the card: the FP16 checkpoint (W) and the
f16 export of the sealed file (W_hat, `llvq-llm --bin export`). An arm never writes a
weight: it names a base set and a set of projection names to take from the other one,
and the forward resolves every tensor through that rule. So the bases cannot drift,
and the job checks that they did not, by sha256 of every tensor before and after.

Per window of WikiText-2 test (4,096 tokens, cut as `bin/ppl` cuts them):
  T     the FP16 model, the teacher; its next-token log-probabilities are kept.
  S     the fully sealed export (projections, embedding, norms).
  M_g   mirror form: S with group g restored to FP16 (the form of the MMLU truth).
  D_g   direct form: T with only group g taken from the sealed export.
  T2    T again, after every arm: KL(T || T2) must be 0 exactly.
For each: KL(T || arm) per token, full vocabulary, and NLL.
Note 1 per matrix: sum ||(W - W_hat) x||^2 / sum ||W x||^2 over every token of the
window, x hooked (a) in T and (b) in S.

Modes:
  --selftest     tiny random Qwen3-shaped model built in-process on CPU, no download.
  --parity       the forward here against transformers' Qwen3ForCausalLM, tiny, CPU.
                 Needs transformers; run on the Mac, not in the job.
  --check-tokens the token stream and its fingerprint only (tokenizer and parquet).
  --run          the measurement. Card only.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Pinned inputs. Checked on 2026-11-13 (docs/mesures/layer-vs-output-4b-2026-11-13-brut/pre-launch/).
# ---------------------------------------------------------------------------
CKPT_REPO = "Qwen/Qwen3-4B"
CKPT_REV = "1cfa9a7208912126459214e8b04321603b3df60c"
CKPT_SHARDS = {
    "model-00001-of-00003.safetensors": "328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223",
    "model-00002-of-00003.safetensors": "6cd087b316306a68c562436b5492edbcf6e16c6dba3a1308279caa5a58e21ca5",
    "model-00003-of-00003.safetensors": "e4bf436957184f4eeb86a80e9db394503f1f56446b2e6b7edeac5b81470f4ca1",
}
TOKENIZER_SHA = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
EXPORT_SHA = "2a018cbdea37849c6fde29576e75aefb9125d9964e20e78bd3afb2a0948bba59"
EXPORT_BYTES = 8_044_981_648
WIKI_REPO = "Salesforce/wikitext"
WIKI_REV = "b08601e04326c79dfdd32d625aee71d232d685c3"
WIKI_FILE = "wikitext-2-raw-v1/test-00000-of-00001.parquet"
WIKI_SHA = "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91"
CTX = 4096
# `bin/ppl` on this corpus: 299,078 tokens, fingerprint of the first 12 windows.
WIKI_TOKENS = 299_078
FP12 = 0x3F1BACA9033BF251

# Rust dense path, f16, the same 12 windows (rule D). Per-window mean NLL.
# FP16: docs/mesures/a4-campagne-4b-ppl-BRUT-2026-08-06.txt lines 43-57, ppl 12.2369.
REF_NLL_FP16 = [2.507059152, 2.801779275, 2.309614908, 1.988569568, 2.600570198, 2.547732324,
                2.608475227, 2.557975141, 2.761579480, 2.208976028, 2.642599349, 2.518568519]
# Sealed 4B: docs/mesures/ppl-scelles-2026-10-04-brut/ppl-4b.txt lines 5-19, ppl 12.5834.
REF_NLL_SEALED = [2.528929144, 2.902055432, 2.263911306, 2.032169328, 2.639317193, 2.601170921,
                  2.603101152, 2.596139061, 2.733776423, 2.241730578, 2.668958429, 2.577238820]
REF_PPL_FP16, REF_PPL_SEALED = 12.2369, 12.5834
RULE_D_TOL = 0.005  # +-0.5 % on each perplexity; see the prereg section 4

BOOT_DRAWS = 10_000
BOOT_SEED = 20261113
Z95 = 1.959963984540054

# ---------------------------------------------------------------------------
# Groups, as the ablation of 2026-11-12 defined them.
# ---------------------------------------------------------------------------
SITE = {"q_proj": "self_attn", "k_proj": "self_attn", "v_proj": "self_attn", "o_proj": "self_attn",
        "gate_proj": "mlp", "up_proj": "mlp", "down_proj": "mlp"}
PROJ_TYPES = tuple(SITE)
GROUPS = {
    "qkv": ("q_proj", "k_proj", "v_proj"),
    "o": ("o_proj",),
    "gateup": ("gate_proj", "up_proj"),
    "down": ("down_proj",),
    "attn": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "mlp": ("gate_proj", "up_proj", "down_proj"),
    "all": PROJ_TYPES,
}
GROUP_ORDER = ("qkv", "o", "gateup", "down", "attn", "mlp", "all")
DISJOINT = ("gateup", "down", "qkv", "o")


def proj_name(layer: int, proj: str) -> str:
    return f"model.layers.{layer}.{SITE[proj]}.{proj}.weight"


def group_members(group: str, n_layers: int) -> frozenset[str]:
    return frozenset(proj_name(b, p) for b in range(n_layers) for p in GROUPS[group])


def all_projections(n_layers: int) -> list[str]:
    return [proj_name(b, p) for b in range(n_layers) for p in PROJ_TYPES]


# ---------------------------------------------------------------------------
# Small pure helpers.
# ---------------------------------------------------------------------------
def fnv_fingerprint(ids) -> int:
    """`llvq_llm::eval::token_fingerprint`: FNV-1a 64 over the u32 ids, little endian."""
    h = 0xCBF29CE484222325
    for i in ids:
        for b in int(i).to_bytes(4, "little"):
            h ^= b
            h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def cut_windows(ids: list[int], ctx: int) -> list[list[int]]:
    """Non-overlapping windows from token 0, the tail dropped, as `bin/ppl` cuts them."""
    return [ids[w * ctx:(w + 1) * ctx] for w in range(len(ids) // ctx)]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_tensors(d: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for k in sorted(d):
        h.update(k.encode())
        t = d[k].detach().contiguous().cpu()
        h.update(str(t.dtype).encode())
        h.update(t.view(torch.uint8).numpy().tobytes() if t.dim() else t.reshape(1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def mean_sd(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else float("nan")
    return m, sd


def normal_ci(xs: list[float]) -> tuple[float, float, float]:
    """Mean over windows and its normal 95 % interval, mean +- 1.96 sd / sqrt(n)."""
    m, sd = mean_sd(xs)
    half = Z95 * sd / math.sqrt(len(xs))
    return m, m - half, m + half


def bootstrap_ci(cols: list[list[float]], stat, draws: int = BOOT_DRAWS, seed: int = BOOT_SEED):
    """Percentile 95 % interval of stat(*resampled columns), windows resampled together."""
    n = len(cols[0])
    rng = random.Random(seed)
    vals = []
    for _ in range(draws):
        idx = [rng.randrange(n) for _ in range(n)]
        vals.append(stat(*[[c[i] for i in idx] for c in cols]))
    vals.sort()
    return vals[int(0.025 * draws)], vals[int(0.975 * draws) - 1]


def agrees(values: dict[str, float]) -> bool:
    """The ordering half of the AGREE rule: gateup first, down second, both above qkv and o.

    `values` maps each disjoint group to its note; larger means the group carries more.
    The resolution half (down against the better of qkv and o) is checked by the caller.
    """
    rest = max(values["qkv"], values["o"])
    return values["gateup"] > values["down"] > rest


# ---------------------------------------------------------------------------
# The Qwen3 forward. Same arithmetic as transformers' Qwen3 (checked by --parity):
# RMSNorm in f32 then cast, q/k norms before RoPE, rotate_half RoPE, GQA, SwiGLU.
# ---------------------------------------------------------------------------
class Cfg:
    def __init__(self, d: dict):
        self.hidden = d["hidden_size"]
        self.layers = d["num_hidden_layers"]
        self.heads = d["num_attention_heads"]
        self.kv_heads = d["num_key_value_heads"]
        self.head_dim = d.get("head_dim") or self.hidden // self.heads
        self.inter = d["intermediate_size"]
        self.vocab = d["vocab_size"]
        self.eps = d["rms_norm_eps"]
        self.theta = d["rope_theta"]
        self.tied = d.get("tie_word_embeddings", False)
        if d.get("rope_scaling"):
            raise SystemExit("refused: rope_scaling is set; this forward implements none")
        if d.get("use_sliding_window"):
            raise SystemExit("refused: sliding window attention is not implemented here")
        if d.get("attention_bias"):
            raise SystemExit("refused: attention bias is not implemented here")


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(x.dtype)


def rope_tables(cfg: Cfg, length: int, device, dtype):
    dim = cfg.head_dim
    inv = 1.0 / (cfg.theta ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim))
    pos = torch.arange(length, device=device).float()
    freqs = torch.outer(pos, inv)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class Arm:
    """A base weight set and the projection names taken from the other one."""

    def __init__(self, name: str, base: str, swapped: frozenset[str]):
        assert base in ("fp16", "sealed")
        self.name, self.base, self.swapped = name, base, swapped

    def resolver(self, sets: dict[str, dict[str, torch.Tensor]]):
        base = sets[self.base]
        other = sets["sealed" if self.base == "fp16" else "fp16"]
        swapped = self.swapped

        def get(name: str) -> torch.Tensor:
            src = other if name in swapped else base
            return src[name]

        return get


def forward(cfg: Cfg, get, ids: torch.Tensor, hook=None) -> torch.Tensor:
    """Logits (L, V) for one sequence of token ids (L,). `hook(name, x)` sees every projection input."""
    emb = get("model.embed_tokens.weight")
    dtype = emb.dtype
    length = ids.shape[0]
    h = emb[ids].unsqueeze(0)
    cos, sin = rope_tables(cfg, length, ids.device, dtype)

    def lin(name: str, x: torch.Tensor) -> torch.Tensor:
        if hook is not None:
            hook(name, x)
        return F.linear(x, get(name))

    for b in range(cfg.layers):
        p = f"model.layers.{b}."
        x = rms_norm(h, get(p + "input_layernorm.weight"), cfg.eps)
        q = lin(p + "self_attn.q_proj.weight", x).view(1, length, cfg.heads, cfg.head_dim)
        k = lin(p + "self_attn.k_proj.weight", x).view(1, length, cfg.kv_heads, cfg.head_dim)
        v = lin(p + "self_attn.v_proj.weight", x).view(1, length, cfg.kv_heads, cfg.head_dim)
        q = rms_norm(q, get(p + "self_attn.q_norm.weight"), cfg.eps).transpose(1, 2)
        k = rms_norm(k, get(p + "self_attn.k_norm.weight"), cfg.eps).transpose(1, 2)
        v = v.transpose(1, 2)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        rep = cfg.heads // cfg.kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=cfg.head_dim ** -0.5)
        a = a.transpose(1, 2).reshape(1, length, cfg.heads * cfg.head_dim)
        h = h + lin(p + "self_attn.o_proj.weight", a)
        x = rms_norm(h, get(p + "post_attention_layernorm.weight"), cfg.eps)
        g = lin(p + "mlp.gate_proj.weight", x)
        u = lin(p + "mlp.up_proj.weight", x)
        h = h + lin(p + "mlp.down_proj.weight", F.silu(g) * u)
    h = rms_norm(h, get("model.norm.weight"), cfg.eps)
    head = emb if cfg.tied else get("lm_head.weight")
    return F.linear(h, head)[0]


# ---------------------------------------------------------------------------
# Scores.
# ---------------------------------------------------------------------------
def teacher_logp(logits: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
    n = logits.shape[0] - 1
    out = torch.empty((n, logits.shape[1]), dtype=torch.float32, device=logits.device)
    for r in range(0, n, chunk):
        out[r:r + chunk] = F.log_softmax(logits[r:min(r + chunk, n)].float(), dim=-1)
    return out


def kl_nll(lp_t: torch.Tensor, logits: torch.Tensor, targets: torch.Tensor, chunk: int = 1024):
    """Sum over positions of KL(teacher || arm) and of the arm's NLL, in nats, and the count."""
    n = lp_t.shape[0]
    kl = torch.zeros((), dtype=torch.float64, device=logits.device)
    nll = torch.zeros((), dtype=torch.float64, device=logits.device)
    for r in range(0, n, chunk):
        e = min(r + chunk, n)
        lq = F.log_softmax(logits[r:e].float(), dim=-1)
        lp = lp_t[r:e]
        kl += (lp.exp() * (lp - lq)).sum().double()
        nll -= lq.gather(1, targets[r:e].unsqueeze(1)).sum().double()
    return float(kl), float(nll), n


class Note1:
    """Accumulates sum ||(W - W_hat) x||^2 and sum ||W x||^2 per projection."""

    def __init__(self, fp16: dict, sealed: dict, names: list[str], acc_dtype):
        self.fp16, self.sealed, self.acc = fp16, sealed, acc_dtype
        dev = next(iter(fp16.values())).device
        self.num = {n: torch.zeros((), dtype=torch.float64, device=dev) for n in names}
        self.den = {n: torch.zeros((), dtype=torch.float64, device=dev) for n in names}

    def __call__(self, name: str, x: torch.Tensor) -> None:
        if name not in self.num:
            return
        xa = x.reshape(-1, x.shape[-1]).to(self.acc)
        w = self.fp16[name].to(self.acc)
        w_hat = self.sealed[name].to(self.acc)
        y_err = xa @ (w - w_hat).T
        y_ref = xa @ w.T
        self.num[name] += y_err.pow(2).sum().double()
        self.den[name] += y_ref.pow(2).sum().double()

    def values(self) -> dict[str, tuple[float, float]]:
        return {n: (float(self.num[n]), float(self.den[n])) for n in self.num}


def group_note1(values: dict[str, tuple[float, float]], n_layers: int) -> dict[str, tuple[float, float]]:
    """Per group: pooled ratio (sum num / sum den) and plain mean of per-matrix ratios."""
    out = {}
    for g in GROUP_ORDER:
        mem = sorted(group_members(g, n_layers))
        num = sum(values[m][0] for m in mem)
        den = sum(values[m][1] for m in mem)
        ratios = [values[m][0] / values[m][1] for m in mem]
        out[g] = (num / den, sum(ratios) / len(ratios))
    return out


def build_arms(n_layers: int) -> list[Arm]:
    arms = [Arm("S", "sealed", frozenset())]
    arms += [Arm(f"M_{g}", "sealed", group_members(g, n_layers)) for g in GROUP_ORDER]
    arms += [Arm(f"D_{g}", "fp16", group_members(g, n_layers)) for g in GROUP_ORDER]
    return arms


def score_window(cfg, sets, ids_list, device, note_a=None, note_b=None):
    """Every arm on one window. Returns {arm: (kl_sum, nll_sum, n)} with T and T2 included."""
    ids = torch.tensor(ids_list, dtype=torch.long, device=device)
    targets = ids[1:]
    out = {}
    with torch.no_grad():
        t_get = Arm("T", "fp16", frozenset()).resolver(sets)
        z = forward(cfg, t_get, ids, hook=note_a)
        lp_t = teacher_logp(z)
        out["T"] = kl_nll(lp_t, z, targets)
        del z
        for arm in build_arms(cfg.layers):
            hook = note_b if arm.name == "S" else None
            z = forward(cfg, arm.resolver(sets), ids, hook=hook)
            out[arm.name] = kl_nll(lp_t, z, targets)
            del z
        z = forward(cfg, t_get, ids)
        out["T2"] = kl_nll(lp_t, z, targets)
        del z, lp_t
    return out


# ---------------------------------------------------------------------------
# Selftest: tiny random model, CPU, float64.
# ---------------------------------------------------------------------------
TINY = dict(hidden_size=16, num_hidden_layers=36, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            intermediate_size=24, vocab_size=40, rms_norm_eps=1e-6, rope_theta=1_000_000.0,
            tie_word_embeddings=True, rope_scaling=None, use_sliding_window=False, attention_bias=False)


def tiny_weights(cfg: Cfg, seed: int, dtype=torch.float64) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)

    def r(*shape, s=0.2):
        return (torch.randn(*shape, generator=g, dtype=torch.float64) * s).to(dtype)

    w = {"model.embed_tokens.weight": r(cfg.vocab, cfg.hidden, s=1.0),
         "model.norm.weight": 1.0 + r(cfg.hidden, s=0.05)}
    qd, kd = cfg.heads * cfg.head_dim, cfg.kv_heads * cfg.head_dim
    shapes = {"q_proj": (qd, cfg.hidden), "k_proj": (kd, cfg.hidden), "v_proj": (kd, cfg.hidden),
              "o_proj": (cfg.hidden, qd), "gate_proj": (cfg.inter, cfg.hidden),
              "up_proj": (cfg.inter, cfg.hidden), "down_proj": (cfg.hidden, cfg.inter)}
    for b in range(cfg.layers):
        p = f"model.layers.{b}."
        for proj, shp in shapes.items():
            w[proj_name(b, proj)] = r(*shp)
        w[p + "input_layernorm.weight"] = 1.0 + r(cfg.hidden, s=0.05)
        w[p + "post_attention_layernorm.weight"] = 1.0 + r(cfg.hidden, s=0.05)
        w[p + "self_attn.q_norm.weight"] = 1.0 + r(cfg.head_dim, s=0.05)
        w[p + "self_attn.k_norm.weight"] = 1.0 + r(cfg.head_dim, s=0.05)
    return w


def selftest() -> int:
    torch.manual_seed(0)
    cfg = Cfg(TINY)
    fails = []

    def check(ok: bool, what: str):
        print(f"  {'PASS' if ok else 'FAIL'}  {what}")
        if not ok:
            fails.append(what)

    print("selftest: tiny Qwen3-shaped model, 36 layers, CPU, float64")
    # 1. Group membership.
    counts = {g: len(group_members(g, 36)) for g in GROUP_ORDER}
    want = dict(qkv=108, o=36, gateup=72, down=36, attn=144, mlp=108, all=252)
    check(counts == want, f"group sizes {counts}")
    part = set().union(*(group_members(g, 36) for g in DISJOINT))
    check(len(part) == 252 and sum(counts[g] for g in DISJOINT) == 252, "the four disjoint groups partition 252")

    # 2. Arms on a sealed copy whose projections differ and whose embedding and norms do not.
    fp16 = tiny_weights(cfg, 1)
    noise = tiny_weights(cfg, 2)
    sealed = {k: (v + 0.05 * noise[k] if k in set(all_projections(36)) else v.clone()) for k, v in fp16.items()}
    sets = {"fp16": fp16, "sealed": sealed}
    h_fp, h_se = sha256_tensors(fp16), sha256_tensors(sealed)
    ids = torch.randint(0, cfg.vocab, (33,), generator=torch.Generator().manual_seed(3))
    t_get = Arm("T", "fp16", frozenset()).resolver(sets)
    z_t = forward(cfg, t_get, ids)
    z_s = forward(cfg, Arm("S", "sealed", frozenset()).resolver(sets), ids)
    all_m = group_members("all", 36)
    z_dall = forward(cfg, Arm("D_all", "fp16", all_m).resolver(sets), ids)
    z_mall = forward(cfg, Arm("M_all", "sealed", all_m).resolver(sets), ids)
    z_dnone = forward(cfg, Arm("D_none", "fp16", frozenset()).resolver(sets), ids)
    check(torch.equal(z_dall, z_s), "direct form with all 252 swapped reproduces the sealed logits exactly")
    check(torch.equal(z_mall, z_t), "mirror form with all 252 restored reproduces the FP16 logits exactly")
    lp = teacher_logp(z_t)
    kl0, _, n0 = kl_nll(lp, z_dnone, ids[1:])
    kls, _, _ = kl_nll(lp, z_s, ids[1:])
    check(kl0 == 0.0 and n0 == 32, f"swapping none gives KL exactly 0 over {n0} positions (got {kl0!r})")
    check(kls > 0.0, f"the sealed copy is at KL {kls:.3e} > 0")
    z_dq = forward(cfg, Arm("D_qkv", "fp16", group_members("qkv", 36)).resolver(sets), ids)
    check(not torch.equal(z_dq, z_t) and not torch.equal(z_dq, z_s), "a one-group swap is neither base")

    # 3. Every arm of the job, one at a time, the bases hashed after each; then the job's own window loop.
    moved = []
    for arm in build_arms(36):
        forward(cfg, arm.resolver(sets), ids)
        if sha256_tensors(fp16) != h_fp or sha256_tensors(sealed) != h_se:
            moved.append(arm.name)
    check(not moved, f"no arm of {len(build_arms(36))} changes a base tensor (sha256 after each){moved or ''}")
    res = score_window(cfg, sets, ids.tolist(), torch.device("cpu"))
    check(set(res) == {"T", "T2", "S"} | {f"M_{g}" for g in GROUP_ORDER} | {f"D_{g}" for g in GROUP_ORDER},
          f"score_window runs {len(res)} arms")
    check(res["T"][0] == 0.0 and res["T2"][0] == 0.0, "KL(T || T) and KL(T || T2) are exactly 0")
    check(res["D_all"][0] == res["S"][0] and res["M_all"][0] == 0.0, "D_all equals S and M_all equals T in KL")
    check(sha256_tensors(fp16) == h_fp and sha256_tensors(sealed) == h_se,
          "both bases unchanged by sha256 after every arm")

    # 4. Note 1.
    names = all_projections(36)
    note = Note1(fp16, {k: v.clone() for k, v in fp16.items()}, names, torch.float64)
    forward(cfg, t_get, ids, hook=note)
    v = note.values()
    check(all(num == 0.0 and den > 0 for num, den in v.values()), "note 1 is 0 for identical weights")
    eps = 0.03
    scaled = {k: (fp16[k] * (1 + eps) if k in set(names) else fp16[k]) for k in fp16}
    note = Note1(fp16, scaled, names, torch.float64)
    forward(cfg, t_get, ids, hook=note)
    worst = max(abs(num / den - eps * eps) / (eps * eps) for num, den in note.values().values())
    check(worst < 1e-9, f"note 1 equals eps^2 for W_hat = (1+eps) W, worst relative error {worst:.1e}")
    gv = group_note1(note.values(), 36)
    check(all(abs(p - eps * eps) < 1e-12 and abs(m - eps * eps) < 1e-12 for p, m in gv.values()),
          "group aggregates, pooled and mean, equal eps^2 too")

    # 5. Windows and fingerprint.
    toy = list(range(10))
    check(cut_windows(toy, 4) == [[0, 1, 2, 3], [4, 5, 6, 7]], "windows are non-overlapping and drop the tail")
    check(fnv_fingerprint([]) == 0xCBF29CE484222325 and fnv_fingerprint([1]) != fnv_fingerprint([256]),
          "fingerprint offset basis and byte order")

    # 6. Statistics and the decision rule.
    m, lo, hi = normal_ci([2.0, 2.0, 2.0, 2.0])
    check((m, lo, hi) == (2.0, 2.0, 2.0), "a constant difference has a zero-width normal interval")
    b1 = bootstrap_ci([[1.0, 2.0, 3.0, 4.0]], lambda c: sum(c) / len(c), draws=500)
    b2 = bootstrap_ci([[1.0, 2.0, 3.0, 4.0]], lambda c: sum(c) / len(c), draws=500)
    check(b1 == b2 and b1[0] <= 2.5 <= b1[1], f"bootstrap is seeded and brackets the mean {b1}")
    hits = 0
    for perm in itertools.permutations(DISJOINT):
        vals = {g: float(4 - perm.index(g)) for g in DISJOINT}
        hits += agrees(vals)
    check(hits == 2, f"the ordering rule accepts {hits} of the 24 orders of four groups (2 expected)")

    # 7. The report path end to end on 12 tiny windows: it runs, writes its three tables, names an outcome,
    #    and finds rule D red (the tiny model is not Qwen3-4B, so the perplexity check must refuse it).
    import tempfile

    na = Note1(fp16, sealed, names, torch.float64)
    nb = Note1(fp16, sealed, names, torch.float64)
    gen = torch.Generator().manual_seed(4)
    rows = [score_window(cfg, sets, torch.randint(0, cfg.vocab, (17,), generator=gen).tolist(),
                         torch.device("cpu"), na, nb) for _ in range(12)]
    with tempfile.TemporaryDirectory() as d:
        verdict = report_and_write(Path(d), cfg, rows, na, nb, 12, {"same_other": [], "others": ["x"]})
        n_groups = len((Path(d) / "groups.tsv").read_text().splitlines()) - 1
        n_mat = len((Path(d) / "matrices.tsv").read_text().splitlines()) - 1
        has_report = (Path(d) / "report.txt").stat().st_size > 0
    check(n_groups == 7 and n_mat == 252 and has_report, f"report writes 7 group rows and {n_mat} matrix rows")
    check(verdict["outcome"] in ("R-a", "R-b", "R-c", "R-d") and verdict["c1"],
          f"report names outcome {verdict['outcome']} with C1 green")
    check(not verdict["rule_d"], "rule D is red on a model that is not the 4B")

    print(f"selftest: {'GREEN' if not fails else 'RED'}, {len(fails)} failure(s)")
    return 0 if not fails else 1


def parity() -> int:
    """This forward against transformers' Qwen3ForCausalLM on the tiny config, float32."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    import transformers

    hf_cfg = Qwen3Config(**{k: v for k, v in TINY.items()}, max_position_embeddings=64)
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(hf_cfg).eval().float()
    # Every parameter drawn away from its init, norms included: a norm left at 1 would hide a norm bug.
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_((1.0 if "norm" in name else 0.0) + torch.randn_like(p) * (0.05 if "norm" in name else 0.2))
    sd = {k: v.detach() for k, v in model.state_dict().items()}
    cfg = Cfg(TINY)
    sets = {"fp16": sd, "sealed": sd}
    ids = torch.randint(0, cfg.vocab, (33,), generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        ref = model(ids.unsqueeze(0)).logits[0]
        ours = forward(cfg, Arm("T", "fp16", frozenset()).resolver(sets), ids)
    diff = float((ref - ours).abs().max())
    scale = float(ref.abs().max())
    ok = diff <= 1e-4 * max(1.0, scale)
    print(f"parity: transformers {transformers.__version__}, torch {torch.__version__}, "
          f"max |dlogit| {diff:.3e} over max |logit| {scale:.3e}: {'MATCH' if ok else 'MISMATCH'}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# The real inputs.
# ---------------------------------------------------------------------------
def fetch_tokens(cache: str | None):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    tok_path = Path(hf_hub_download(CKPT_REPO, "tokenizer.json", revision=CKPT_REV, cache_dir=cache))
    wiki = Path(hf_hub_download(WIKI_REPO, WIKI_FILE, repo_type="dataset", revision=WIKI_REV, cache_dir=cache))
    for path, want in ((tok_path, TOKENIZER_SHA), (wiki, WIKI_SHA)):
        got = sha256_file(path)
        if got != want:
            raise SystemExit(f"refused: {path} sha256 {got}, expected {want}")
    table = pq.read_table(wiki)
    if table.column_names != ["text"]:
        raise SystemExit(f"refused: wikitext columns {table.column_names}, `bin/ppl` reads one string column")
    text = "\n\n".join(table.column("text").to_pylist())
    ids = Tokenizer.from_file(str(tok_path)).encode(text, add_special_tokens=False).ids
    fp = fnv_fingerprint(ids[:12 * CTX])
    print(f"tokens: {len(ids)} (expected {WIKI_TOKENS}), {len(ids) // CTX} windows of {CTX}, "
          f"fingerprint of 12 windows {fp:016x} (expected {FP12:016x})")
    if len(ids) != WIKI_TOKENS or fp != FP12:
        raise SystemExit("refused: the token stream is not the one `bin/ppl` scored")
    return ids


def check_pins() -> None:
    import huggingface_hub
    import pyarrow
    import safetensors
    import tokenizers

    want = {"torch": "2.5.1", "safetensors": "0.4.5", "tokenizers": "0.22.2", "pyarrow": "20.0.0",
            "huggingface_hub": "1.19.0"}
    have = {"torch": torch.__version__.split("+")[0], "safetensors": safetensors.__version__,
            "tokenizers": tokenizers.__version__, "pyarrow": pyarrow.__version__,
            "huggingface_hub": huggingface_hub.__version__}
    print(f"versions: {have}")
    if have != want:
        raise SystemExit(f"refused: versions differ from the script header {want}")


def load_sets(export: Path, stage: Path, cache: str | None, device):
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    t0 = time.time()
    stage.mkdir(parents=True, exist_ok=True)
    local = stage / "model.safetensors"
    if export.stat().st_size != EXPORT_BYTES:
        raise SystemExit(f"refused: {export} is {export.stat().st_size} B, expected {EXPORT_BYTES}")
    shutil.copyfile(export, local)
    got = sha256_file(local)
    print(f"export: {export} staged to {local}, sha256 {got}")
    if got != EXPORT_SHA:
        raise SystemExit(f"refused: export sha256 {got}, expected {EXPORT_SHA}")
    cfg_path = Path(hf_hub_download(CKPT_REPO, "config.json", revision=CKPT_REV, cache_dir=cache))
    cfg_d = json.loads(cfg_path.read_text())
    exp_cfg = json.loads((export.parent / "config.json").read_text())
    for k in ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
              "intermediate_size", "vocab_size", "head_dim", "rope_theta", "rms_norm_eps", "tie_word_embeddings"):
        if exp_cfg.get(k) != cfg_d.get(k):
            raise SystemExit(f"refused: export config {k}={exp_cfg.get(k)} against checkpoint {cfg_d.get(k)}")
    fp16 = {}
    for shard, want in CKPT_SHARDS.items():
        p = Path(hf_hub_download(CKPT_REPO, shard, revision=CKPT_REV, cache_dir=cache))
        got = sha256_file(p)
        print(f"checkpoint: {shard} sha256 {got}")
        if got != want:
            raise SystemExit(f"refused: {shard} sha256 {got}, expected {want}")
        for k, v in load_file(str(p), device=str(device)).items():
            fp16[k] = v.to(torch.float16)
    sealed = {k: v.to(torch.float16) for k, v in load_file(str(local), device=str(device)).items()}
    if cfg_d.get("tie_word_embeddings"):
        fp16.pop("lm_head.weight", None)
        sealed.pop("lm_head.weight", None)
    if set(fp16) != set(sealed):
        raise SystemExit(f"refused: tensor names differ: {sorted(set(fp16) ^ set(sealed))[:8]}")
    for k in fp16:
        if fp16[k].shape != sealed[k].shape:
            raise SystemExit(f"refused: {k} is {tuple(sealed[k].shape)} in the export, {tuple(fp16[k].shape)} in W")
    cfg = Cfg(cfg_d)
    projs = set(all_projections(cfg.layers))
    same_proj = [k for k in projs if torch.equal(fp16[k], sealed[k])]
    others = sorted(set(fp16) - projs)
    same_other = [k for k in others if torch.equal(fp16[k], sealed[k])]
    print(f"tensors: {len(fp16)} each, {len(projs)} projections ({len(same_proj)} identical in W and W_hat), "
          f"{len(others)} others ({len(same_other)} identical: "
          f"{'all' if len(same_other) == len(others) else ', '.join(sorted(set(others) - set(same_other))[:6])})")
    if same_proj:
        raise SystemExit(f"refused: {len(same_proj)} projections are identical in W and W_hat")
    print(f"loaded in {time.time() - t0:.0f} s")
    return cfg, {"fp16": fp16, "sealed": sealed}, {"same_other": same_other, "others": others}


def report_and_write(out: Path, cfg, rows, note_a, note_b, n_windows, extra) -> dict:
    """Statistics of the prereg section 6, the TSVs and report.txt."""
    W = n_windows
    per = {arm: [rows[w][arm] for w in range(W)] for arm in rows[0]}
    kl = {a: [k / n for k, _, n in v] for a, v in per.items()}
    nll = {a: [s / n for _, s, n in v] for a, v in per.items()}
    lines = []
    say = lines.append

    # Rule D: perplexity of T and S on the first 12 windows against the Rust path.
    def ppl(arm, k=12):
        s = sum(per[arm][w][1] for w in range(k))
        n = sum(per[arm][w][2] for w in range(k))
        return math.exp(s / n)

    p_t, p_s = ppl("T"), ppl("S")
    d_t, d_s = p_t / REF_PPL_FP16 - 1, p_s / REF_PPL_SEALED - 1
    rule_d = abs(d_t) <= RULE_D_TOL and abs(d_s) <= RULE_D_TOL
    win_t = max(abs(nll["T"][w] - REF_NLL_FP16[w]) for w in range(12))
    win_s = max(abs(nll["S"][w] - REF_NLL_SEALED[w]) for w in range(12))
    t2_zero = sum(1 for w in range(W) if per["T2"][w][0] != 0.0)
    say("CONTROLS")
    say(f"  C1 KL(T || T2) exactly 0 on {W - t2_zero} of {W} windows: {'GREEN' if t2_zero == 0 else 'RED'}")
    say(f"  C2 ppl FP16, 12 windows: {p_t:.4f} against Rust {REF_PPL_FP16} ({d_t * 100:+.3f} %); "
        f"worst window |dnll| {win_t:.2e}")
    say(f"  C3 ppl sealed export, 12 windows: {p_s:.4f} against Rust {REF_PPL_SEALED} ({d_s * 100:+.3f} %); "
        f"worst window |dnll| {win_s:.2e}")
    say(f"  RULE D (each within +-{RULE_D_TOL * 100:.1f} %): {'GREEN' if rule_d else 'RED, THE RUN DECIDES NOTHING'}")
    say(f"  ratio sealed/FP16, 12 windows: {p_s / p_t:.5f} here, {REF_PPL_SEALED / REF_PPL_FP16:.5f} Rust")
    say(f"  non-projection tensors identical in W and W_hat: {len(extra['same_other'])} of {len(extra['others'])}")
    say(f"  KL(T || S), sealed: {normal_ci(kl['S'])[0]:.6f} nats/token; KL(T || D_all), FP16 embedding and norms "
        f"with the 252 sealed projections: {normal_ci(kl['D_all'])[0]:.6f}; KL(T || M_all): "
        f"{normal_ci(kl['M_all'])[0]:.6f}")

    # Note 2.
    groups = []
    for g in GROUP_ORDER:
        rec = [kl["S"][w] - kl[f"M_{g}"][w] for w in range(W)]
        m, lo, hi = normal_ci(rec)
        blo, bhi = bootstrap_ci([rec], lambda c: sum(c) / len(c))
        share = m / normal_ci(kl["S"])[0]
        slo, shi = bootstrap_ci([rec, kl["S"]], lambda r, s: sum(r) / sum(s))
        dm, dlo, dhi = normal_ci(kl[f"D_{g}"])
        dblo, dbhi = bootstrap_ci([kl[f"D_{g}"]], lambda c: sum(c) / len(c))
        nm = normal_ci([nll["S"][w] - nll[f"M_{g}"][w] for w in range(W)])
        nd = normal_ci([nll[f"D_{g}"][w] - nll["T"][w] for w in range(W)])
        groups.append(dict(group=g, mirror=m, mirror_lo=lo, mirror_hi=hi, mirror_blo=blo, mirror_bhi=bhi,
                           share=share, share_blo=slo, share_bhi=shi, direct=dm, direct_lo=dlo, direct_hi=dhi,
                           direct_blo=dblo, direct_bhi=dbhi, nll_mirror=nm[0], nll_mirror_lo=nm[1],
                           nll_mirror_hi=nm[2], nll_direct=nd[0], nll_direct_lo=nd[1], nll_direct_hi=nd[2]))
    G = {r["group"]: r for r in groups}

    # Note 1.
    va, vb = note_a.values(), note_b.values()
    ga, gb = group_note1(va, cfg.layers), group_note1(vb, cfg.layers)
    for r in groups:
        g = r["group"]
        r.update(n1_pooled_a=ga[g][0], n1_mean_a=ga[g][1], n1_pooled_b=gb[g][0], n1_mean_b=gb[g][1])

    # Decision readings (prereg section 6).
    def note2_reading(key: str, series) -> tuple[bool, str]:
        vals = {g: G[g][key] for g in DISJOINT}
        order = agrees(vals)
        best = "qkv" if vals["qkv"] >= vals["o"] else "o"
        diff = [series("down")[w] - series(best)[w] for w in range(W)]
        m, lo, hi = normal_ci(diff)
        blo, bhi = bootstrap_ci([diff], lambda c: sum(c) / len(c))
        ok = order and lo > 0
        return ok, (f"order {' > '.join(sorted(DISJOINT, key=lambda g: -vals[g]))}; down - {best} {m:+.6f} "
                    f"normal [{lo:+.6f}, {hi:+.6f}] bootstrap [{blo:+.6f}, {bhi:+.6f}]")

    mirror_series = lambda g: [kl["S"][w] - kl[f"M_{g}"][w] for w in range(W)]  # noqa: E731
    direct_series = lambda g: kl[f"D_{g}"]  # noqa: E731
    nllm_series = lambda g: [nll["S"][w] - nll[f"M_{g}"][w] for w in range(W)]  # noqa: E731
    nlld_series = lambda g: [nll[f"D_{g}"][w] - nll["T"][w] for w in range(W)]  # noqa: E731
    readings = {}
    readings["note2 mirror KL (primary)"] = note2_reading("mirror", mirror_series)
    readings["note2 direct KL"] = note2_reading("direct", direct_series)
    readings["note2 mirror NLL"] = note2_reading("nll_mirror", nllm_series)
    readings["note2 direct NLL"] = note2_reading("nll_direct", nlld_series)
    for agg in ("pooled", "mean"):
        a = {g: G[g][f"n1_{agg}_a"] for g in DISJOINT}
        b = {g: G[g][f"n1_{agg}_b"] for g in DISJOINT}
        tag = " (primary)" if agg == "pooled" else ""
        for name, v in (("FP16 x", a), ("sealed x", b)):
            readings[f"note1 {agg}, {name}"] = (
                agrees(v), "order " + " > ".join(sorted(DISJOINT, key=lambda g: -v[g])))
        readings[f"note1 {agg}, both x{tag}"] = (agrees(a) and agrees(b), "the ordering in both x variants")
    n2 = readings["note2 mirror KL (primary)"][0]
    n1 = readings["note1 pooled, both x (primary)"][0]
    outcome = {(True, False): "R-a", (True, True): "R-b", (False, False): "R-c", (False, True): "R-d"}[(n2, n1)]

    say("")
    say(f"NOTE 2 by group, {W} windows of {CTX} tokens, nats per token, normal and bootstrap 95 % over windows")
    say("  group   mirror KL recovered [normal] [bootstrap]   share of KL(S) [boot]   direct KL [normal]")
    for r in groups:
        say(f"  {r['group']:<7} {r['mirror']:+.6f} [{r['mirror_lo']:+.6f}, {r['mirror_hi']:+.6f}] "
            f"[{r['mirror_blo']:+.6f}, {r['mirror_bhi']:+.6f}]  {r['share']:.4f} [{r['share_blo']:.4f}, "
            f"{r['share_bhi']:.4f}]  {r['direct']:.6f} [{r['direct_lo']:.6f}, {r['direct_hi']:.6f}]")
    say("  group   mirror NLL recovered [normal]            direct NLL added [normal]")
    for r in groups:
        say(f"  {r['group']:<7} {r['nll_mirror']:+.6f} [{r['nll_mirror_lo']:+.6f}, {r['nll_mirror_hi']:+.6f}]   "
            f"{r['nll_direct']:+.6f} [{r['nll_direct_lo']:+.6f}, {r['nll_direct_hi']:+.6f}]")
    say("")
    say("NOTE 1 by group, sum ||(W - W_hat) x||^2 / sum ||W x||^2")
    say("  group   pooled FP16 x   mean FP16 x   pooled sealed x   mean sealed x")
    for r in groups:
        say(f"  {r['group']:<7} {r['n1_pooled_a']:.6e}   {r['n1_mean_a']:.6e}   {r['n1_pooled_b']:.6e}   "
            f"{r['n1_mean_b']:.6e}")
    say("")
    say("READINGS (prereg section 6; truth: gateup > down > max(qkv, o))")
    for k, (ok, why) in readings.items():
        say(f"  {k:<34} {'AGREES' if ok else 'does not agree'}: {why}")
    say(f"  OUTCOME {outcome}" + ("" if rule_d else "  (rule D is RED: the run decides nothing)"))
    say(f"  C1 {'GREEN' if t2_zero == 0 else 'RED'}")

    with open(out / "groups.tsv", "w") as f:
        keys = list(groups[0])
        f.write("\t".join(keys) + "\n")
        for r in groups:
            f.write("\t".join(f"{r[k]:.9g}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
    with open(out / "matrices.tsv", "w") as f:
        f.write("layer\tproj\tgroups\tnum_fp16x\tden_fp16x\tratio_fp16x\tnum_sealedx\tden_sealedx\tratio_sealedx\n")
        for b in range(cfg.layers):
            for p in PROJ_TYPES:
                n = proj_name(b, p)
                gs = ",".join(g for g in GROUP_ORDER if p in GROUPS[g])
                (na, da), (nb, db) = va[n], vb[n]
                f.write(f"{b}\t{p}\t{gs}\t{na:.9e}\t{da:.9e}\t{na / da:.9e}\t{nb:.9e}\t{db:.9e}\t{nb / db:.9e}\n")
    text = "\n".join(lines)
    (out / "report.txt").write_text(text + "\n")
    print(text)
    return dict(rule_d=rule_d, c1=t2_zero == 0, outcome=outcome)


def run(args) -> int:
    check_pins()
    if not torch.cuda.is_available():
        raise SystemExit("refused: --run is for the card; the 4B is never loaded on the Mac")
    device = torch.device("cuda:0")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ids = fetch_tokens(args.cache)
    windows = cut_windows(ids, CTX)
    cfg, sets, extra = load_sets(Path(args.export), Path(args.stage), args.cache, device)
    t0 = time.time()
    h0 = {k: sha256_tensors(v) for k, v in sets.items()}
    print(f"bases: fp16 {h0['fp16'][:16]}, sealed {h0['sealed'][:16]} ({time.time() - t0:.0f} s)")
    names = all_projections(cfg.layers)
    note_a = Note1(sets["fp16"], sets["sealed"], names, torch.float32)
    note_b = Note1(sets["fp16"], sets["sealed"], names, torch.float32)
    rows = []
    wtsv = open(out / "windows.tsv", "w")
    wtsv.write("window\tarm\tkl_sum\tnll_sum\tn\tkl_mean\tnll_mean\n")
    n_target = args.windows if args.windows else None
    t_loop = time.time()
    w = 0
    while True:
        if n_target is not None and w >= n_target:
            break
        if w >= len(windows):
            break
        tw = time.time()
        res = score_window(cfg, sets, windows[w], device, note_a, note_b)
        rows.append(res)
        for arm, (k, s, n) in res.items():
            wtsv.write(f"{w}\t{arm}\t{k:.12e}\t{s:.12e}\t{n}\t{k / n:.12e}\t{s / n:.12e}\n")
        wtsv.flush()
        dt = time.time() - tw
        print(f"  window {w + 1}: {dt:.1f} s, KL(S) {res['S'][0] / res['S'][2]:.5f}, "
              f"nll T {res['T'][1] / res['T'][2]:.6f}, nll S {res['S'][1] / res['S'][2]:.6f}", flush=True)
        if n_target is None:
            # The count is fixed once, from the first window's time, and printed.
            fit = int(args.budget_seconds // dt)
            n_target = min(len(windows), max(12, fit))
            print(f"windows: {n_target} of {len(windows)} (first window {dt:.1f} s, budget "
                  f"{args.budget_seconds} s; at least 12 for rule D)", flush=True)
        w += 1
    wtsv.close()
    if len(rows) < 12:
        raise SystemExit("refused: fewer than 12 windows, rule D cannot be read")
    print(f"window loop: {len(rows)} windows in {time.time() - t_loop:.0f} s")
    h1 = {k: sha256_tensors(v) for k, v in sets.items()}
    same = h0 == h1
    print(f"bases unchanged by sha256 after every arm: {'GREEN' if same else 'RED'}")
    verdict = report_and_write(out, cfg, rows, note_a, note_b, len(rows), extra)
    (out / "inputs.txt").write_text(
        f"checkpoint {CKPT_REPO}@{CKPT_REV}\nexport {args.export} sha256 {EXPORT_SHA}\n"
        f"wikitext {WIKI_REPO}@{WIKI_REV} {WIKI_FILE} sha256 {WIKI_SHA}\nwindows {len(rows)} of {len(windows)}\n"
        f"bases sha256 before {h0}\nbases sha256 after  {h1}\nverdict {verdict}\n")
    return 0 if (same and verdict["c1"]) else 3


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--selftest", action="store_true")
    mode.add_argument("--parity", action="store_true")
    mode.add_argument("--check-tokens", action="store_true")
    mode.add_argument("--run", action="store_true")
    ap.add_argument("--export", help="the sealed export's model.safetensors (bucket mount)")
    ap.add_argument("--stage", default="/tmp/lvo-export", help="local copy of the export")
    ap.add_argument("--out", help="output folder")
    ap.add_argument("--cache", default=os.environ.get("HF_HUB_CACHE"))
    ap.add_argument("--windows", type=int, default=0, help="fixed window count; 0 = from the budget")
    ap.add_argument("--budget-seconds", type=float, default=1380.0)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.parity:
        return parity()
    if a.check_tokens:
        fetch_tokens(a.cache)
        return 0
    if not (a.export and a.out):
        ap.error("--run needs --export and --out")
    if a.windows and a.windows < 12:
        ap.error("--windows must be at least 12 (rule D reads the first 12)")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
