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
"""The patch map: which part of FP16's residual stream, put into the sealed 4B's run, gives back its MMLU deficit.

Prereg (draft): proofs/preregistration-patch-map-4b-2026-11-15.md. Frame: docs/mesures/
recherche-qualite3-r1-cadrage-2026-11-15.txt, section 2 (H6-A to H6-D).

It extends contrast-map-4b (ops/contrast_map.py, imported, never edited): same two weight sets resident on the card
(FP16 Qwen/Qwen3-4B@1cfa9a72 and the f16 export of the sealed 4B, loaded by ops/layer_vs_output.py), same prompts
(the Rust harness's, ported in contrast_map), same 570 questions (the first 570 of the round robin, checked row for
row against contrast-map's questions.tsv), same lens (FP16's final norm weight and FP16's four answer rows), same rule
D against the two Rust dumps, same bands (eigendirections of FP16's uncentred second moment over every position of
the 570 prompts, here at depth 12 only).

Convention of depths, contrast_map's: h(0) is the embedding output, h(d) the stream after block d (blocks 0-based).
A patch at depth d replaces part of the sealed run's h(d) by FP16's h(d), then the sealed blocks d..35 run on. The
sealed run's final norm and head (the sealed q4 embedding, tied) read the output.

Two passes over the 570 prompts:
  pass 1  FP16 alone: C_ff at depth 12 over every position (the band basis), the massive positions of each prompt
          (FP16's depth-8 stream norms), and the 252 matrices' errors on FP16 inputs, by position mask.
  pass 2  FP16 again (its streams at depths 4..20 every position, its answer-row attention at layers 7..13), the
          sealed control (the matrices' errors on sealed inputs), then every arm, resumed from the control's own
          stream at the arm's depth. The first CHECK_Q prompts also run the exactness controls.

Arms (ARMS, ATTN_ARMS): d4..d20 whole stream every position; at depth 12 the three bands every position, the answer
position alone, every position but it, the massive positions alone, band [513, 2560] at the answer position; the
explicit attention path with no patch, and with FP16's answer-row attention probabilities at layers 7..13.

Readings per arm: accuracy at the output and through the lens at depth 25, paired against the control; the share of
the FP16-control gap recovered, with a paired bootstrap interval; the answer-position band slopes at depth 12.
Beside the arms: each matrix's relative error on answer-position inputs, all-position inputs and all but the massive
positions (H6-B, H6-C). The kill rules of the prereg are read last.

Modes:
  --selftest   tiny random models on CPU, float64; no download.
  --run        the measurement. Card only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contrast_map as cm  # noqa: E402
import layer_vs_output as lvo  # noqa: E402

# ---------------------------------------------------------------------------
# The measurement, fixed before the data.
# ---------------------------------------------------------------------------
N_Q = 570
PATCH_DEPTHS = (4, 8, 12, 16, 20)
BAND_DEPTH = 12
PATCH_BANDS = ((1, 64), (65, 512), (513, 2560))  # 1-based, inclusive, FP16's depth-12 eigendirections, descending
QUIET = 2  # PATCH_BANDS[QUIET] = [513, 2560]
LENS_DEPTH = 25
ROW_DEPTHS = (BAND_DEPTH, LENS_DEPTH)  # answer rows kept per arm
ATTN_LAYERS = tuple(range(7, 14))  # blocks 7..13, contrast_map's attention.tsv numbering
MASSIVE_DEPTH = 8
MASSIVE_K = 10.0  # a position is massive if its FP16 h(8) norm exceeds 10 times the prompt's median norm
MASSIVE_K_SHOWN = (5.0, 20.0, 50.0)  # printed beside, never used by an arm
MASKS = ("all", "answer", "nonmassive")
CHECK_Q = 57

# Arms: name, depth, positions, band index (None = the whole stream).
ARMS = (
    ("d4", 4, "all", None), ("d8", 8, "all", None), ("d12", 12, "all", None), ("d16", 16, "all", None),
    ("d20", 20, "all", None),
    ("d12_b1-64", 12, "all", 0), ("d12_b65-512", 12, "all", 1), ("d12_b513-2560", 12, "all", 2),
    ("d12_ans", 12, "ans", None), ("d12_notans", 12, "notans", None), ("d12_massive", 12, "massive", None),
    ("d12_b513-2560_ans", 12, "ans", 2),
)
ATTN_ARMS = (("attn_explicit", None), ("attn_fp16_7-13", "fp16"))
# H6-E: FP16's branch outputs in place of the sealed run's, every position, blocks 0..11.
BRANCH_LAYERS = tuple(range(0, 12))
BRANCH_ARMS = (("attn_0-11", ("attn",)), ("mlp_0-11", ("mlp",)))
H6E_MIN = 0.5

# The instrument check (--instrument): full-split dumps of the sealed 4B's restores, each against FP16, beside the
# control. sha256 of the committed files in docs/data/mmlu-dumps/.
INSTRUMENT_DIR = "docs/data/mmlu-dumps"
INSTRUMENT_ARMS = {
    "qkv": ("mmlu-4b-abl-qkv-FULL.csv", "f50850f2ae0d74021ea1feb7be1c38bc2f7d4a3770ba43d8df6f8858fa3d66d1"),
    "o": ("mmlu-4b-abl-o-FULL.csv", "ea5a95eef9a92a40760770d3fdbc6bbcb94824089b4672c6699534e0d3c0a17c"),
    "gateup": ("mmlu-4b-abl-gateup-FULL.csv", "78e1e2f2c17c1e4d38dc8a72d7fe9651920186bd8e376610ab81912711ff9110"),
    "down": ("mmlu-4b-abl-down-FULL.csv", "f63073b8cbabd3c2553c8ec8c6ec74b9e1b08607fd876db270d8c7033e0a259b"),
    "attn": ("mmlu-4b-abl-attn-FULL.csv", "9ba2c6b087190d7db9bd0e2fb84355e097ab4e1a80aae0e2079740f9958ceabe"),
    "mlp": ("mmlu-4b-abl-mlp-FULL.csv", "3ae92e6eb48ce902faf554072e1748fc3af463b64be15c4bc54705c268f92b35"),
    "all": ("mmlu-4b-abl-all-FULL.csv", "d27c39ace0c2a27e563a71c5ce0c11114d54577498c3027a0e40c77ee596351e"),
    "depth-late": ("mmlu-4b-depth-late-FULL.csv", "72039e0190e852add0a8e0beb040ef54b3e1b7c45e87ec3193351be596bba432"),
    "depth-early": ("mmlu-4b-depth-early-FULL.csv", "fa35e228e7ede069602c508a076dcd227bf1f087b0a09c9839b06ce31878f9d9"),
}
INSTRUMENT_RHO_MIN = 0.9

# The four matrices whose pooled relative error on FP16 inputs reads under 0.002 in contrast-map's matrices.tsv
# (gate_proj 1, 2, 3: 0.00121, 0.00173, 0.00128; down_proj 6: 0.00127). o_proj 0 (0.00204) and gate_proj 4 (0.00208)
# are above the line.
LOW_REL_MAX = 0.002
LOW_REL = tuple(lvo.proj_name(b, p) for b, p in ((1, "gate_proj"), (2, "gate_proj"), (3, "gate_proj"), (6, "down_proj")))

# Gates and rules.
PICK_AGREE_MIN = cm.PICK_AGREE_MIN  # 0.98, rule D's and the two tolerance controls'
REPRO_COUNT_TOL = 2  # questions, accuracy of FP16 and control against contrast-map's
REPRO_PICKS_MIN = 0.99
GAIN_REL_TOL = 1e-3
H6A_MIN, H6B_MAX, H6C_SHARE, H6C_ERR, H6D_MIN = 0.5, 1.5, 0.1, 2.0, 0.2
BOOT_DRAWS = 10_000
BOOT_SEED = 20261115
PASS1_CAP = 600.0  # projected pass-1 seconds after 57 prompts
LOOP_CAP = 1800.0  # projected pass 1 + pass 2 seconds after 57 prompts of pass 2

# contrast-map-4b's raw files, committed in docs/mesures/contrast-map-4b-2026-11-14-brut/.
REF_DIR = "docs/mesures/contrast-map-4b-2026-11-14-brut"
REF_FILES = {
    "control.tsv": "d0af90c432edba6a209bf717691dee477a9dfc03456ff7348fbafd15e4dfa5a2",
    "questions.tsv": "800cb9ccbb5f308b513eea31eafd3cfae302e6dc580a6207be5b1723b4460c1b",
    "matrices.tsv": "4e0e89db3aead9dd64fd42218ff0d29a0e6a18a4973757e1d8cead59d27498bc",
    "lens.tsv": "1c51d9cab46f3c0dd4a9d092a9242ac3b1915d369daa28fe6f8d7b49fd8af9d0",
    "stream_bands.tsv": "dc91372a93fa202b15bbd98ef2e63b2e05104ef180896ccaf08899e797978767",
}
REF_ACC = {"fp16": 419, "control": 378}  # of 570, contrast-map control.tsv script picks: 73.51 % and 66.32 %


# ---------------------------------------------------------------------------
# The forward, cut at block boundaries. Same operations, same order as contrast_map.forward_taps (bit for bit).
# ---------------------------------------------------------------------------
def getter(sets, model: str):
    return lvo.Arm(model, model, frozenset()).resolver(sets)


def block(cfg, get, b: int, h, cos, sin, proj_hook=None, attn_hook=None, attn_row=None, branch=None):
    """Block b. attn_hook(b, q, k, v, a) sees the attention output; attn_row(b, q, k, v, a) returns the one o_proj
    reads (the same tensor when it rewrites nothing). branch(b, kind, out) returns what the attention branch
    (kind "attn", the o_proj output) or the MLP branch ("mlp", the down_proj output) adds to the stream."""
    length = h.shape[1]
    p = f"model.layers.{b}."

    def lin(name, x):
        if proj_hook is not None:
            proj_hook(name, x)
        return F.linear(x, get(name))

    x = lvo.rms_norm(h, get(p + "input_layernorm.weight"), cfg.eps)
    q = lin(p + "self_attn.q_proj.weight", x).view(1, length, cfg.heads, cfg.head_dim)
    k = lin(p + "self_attn.k_proj.weight", x).view(1, length, cfg.kv_heads, cfg.head_dim)
    v = lin(p + "self_attn.v_proj.weight", x).view(1, length, cfg.kv_heads, cfg.head_dim)
    q = lvo.rms_norm(q, get(p + "self_attn.q_norm.weight"), cfg.eps).transpose(1, 2)
    k = lvo.rms_norm(k, get(p + "self_attn.k_norm.weight"), cfg.eps).transpose(1, 2)
    v = v.transpose(1, 2)
    q = q * cos + lvo.rotate_half(q) * sin
    k = k * cos + lvo.rotate_half(k) * sin
    rep = cfg.heads // cfg.kv_heads
    k = k.repeat_interleave(rep, dim=1)
    v = v.repeat_interleave(rep, dim=1)
    a = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=cfg.head_dim ** -0.5)
    if attn_hook is not None:
        attn_hook(b, q, k, v, a)
    if attn_row is not None:
        a = attn_row(b, q, k, v, a)
    a = a.transpose(1, 2).reshape(1, length, cfg.heads * cfg.head_dim)
    o = lin(p + "self_attn.o_proj.weight", a)
    if branch is not None:
        o = branch(b, "attn", o)
    h = h + o
    x = lvo.rms_norm(h, get(p + "post_attention_layernorm.weight"), cfg.eps)
    g = lin(p + "mlp.gate_proj.weight", x)
    u = lin(p + "mlp.up_proj.weight", x)
    m = lin(p + "mlp.down_proj.weight", F.silu(g) * u)
    if branch is not None:
        m = branch(b, "mlp", m)
    return h + m


def branch_replacer(store, layers, kinds):
    """The branch outputs of another run ({(block, kind): tensor}) in place of this run's, at `layers`, for `kinds`."""
    def f(b, kind, out):
        return store[(b, kind)] if b in layers and kind in kinds else out
    return f


def head(cfg, get, h):
    h = lvo.rms_norm(h, get("model.norm.weight"), cfg.eps)
    w = get("model.embed_tokens.weight") if cfg.tied else get("lm_head.weight")
    return F.linear(h, w)[0]


def run_from(cfg, get, h, start: int, cos, sin, proj_hook=None, stream_hook=None, attn_hook=None, attn_row=None,
             branch=None):
    """Blocks start..layers-1 on h = h(start), then the head. stream_hook(l, h) sees h(l), l = start+1..layers."""
    for b in range(start, cfg.layers):
        h = block(cfg, get, b, h, cos, sin, proj_hook, attn_hook, attn_row, branch)
        if stream_hook is not None:
            stream_hook(b + 1, h)
    return head(cfg, get, h)


def forward(cfg, get, ids, proj_hook=None, stream_hook=None, attn_hook=None, attn_row=None, branch=None):
    h = get("model.embed_tokens.weight")[ids].unsqueeze(0)
    cos, sin = lvo.rope_tables(cfg, ids.shape[0], ids.device, h.dtype)
    if stream_hook is not None:
        stream_hook(0, h)
    return run_from(cfg, get, h, 0, cos, sin, proj_hook, stream_hook, attn_hook, attn_row, branch)


def forward_hybrid(cfg, sets, ids, d: int, stream_hook=None):
    """The reference of the exactness control: FP16's embedding and blocks 0..d-1, then the sealed blocks d..35,
    the sealed final norm and head. Its own loop: it calls no patch and no resume."""
    gf, gs = getter(sets, "fp16"), getter(sets, "sealed")
    h = gf("model.embed_tokens.weight")[ids].unsqueeze(0)
    cos, sin = lvo.rope_tables(cfg, ids.shape[0], ids.device, h.dtype)
    if stream_hook is not None:
        stream_hook(0, h)
    for b in range(cfg.layers):
        h = block(cfg, gf if b < d else gs, b, h, cos, sin)
        if stream_hook is not None:
            stream_hook(b + 1, h)
    return head(cfg, gs, h)


# ---------------------------------------------------------------------------
# Patches, positions, attention rows.
# ---------------------------------------------------------------------------
def band_cols(lo: int, hi: int, d: int) -> slice:
    """1-based inclusive band [lo, hi] as columns of the descending eigenbasis."""
    return slice(lo - 1, min(hi, d))


def patch_stream(h_s, h_src, mask=None, basis=None):
    """h_s, h_src (1, L, d). mask: bool (L,) or None (every position). basis: (d, k) orthonormal columns or None.
    Whole stream: the masked rows of h_src replace those of h_s. Band: h_s + ((h_src - h_s) B) B^T on the masked
    rows, in float32 (float64 for float64 streams), then cast to the stream dtype."""
    if basis is None:
        if mask is None:
            return h_src.clone()
        out = h_s.clone()
        out[:, mask] = h_src[:, mask]
        return out
    acc = torch.float64 if h_s.dtype == torch.float64 else torch.float32
    bb = basis.to(device=h_s.device, dtype=acc)
    dlt = h_src.to(acc) - h_s.to(acc)
    if mask is None:
        return (h_s.to(acc) + (dlt @ bb) @ bb.T).to(h_s.dtype)
    out = h_s.clone()
    out[:, mask] = (h_s[:, mask].to(acc) + (dlt[:, mask] @ bb) @ bb.T).to(h_s.dtype)
    return out


def massive_positions(norms, pos: int, k: float = MASSIVE_K):
    """FP16's depth-8 norms (L,): positions whose norm exceeds k times the median over the prompt (torch.median, the
    lower middle value for an even count), the answer position never counted."""
    m = norms > k * norms.median()
    m[pos] = False
    return m


def position_mask(kind: str, length: int, pos: int, massive, device):
    if kind == "all":
        return None
    one = torch.zeros(length, dtype=torch.bool, device=device)
    one[pos] = True
    if kind == "ans":
        return one
    if kind == "notans":
        return ~one
    if kind == "massive":
        return massive.to(device)
    raise ValueError(kind)


def attn_rewriter(layers, pos: int, scale: float, probs=None):
    """At the blocks in `layers`, the answer row of the attention output is recomputed explicitly as P v, in float64,
    with P the given probabilities {block: (heads, pos+1)}, or the model's own softmax(q k^T scale) when probs is
    None. Every other row, and every other block, is the SDPA output untouched."""
    def f(b, q, k, v, a):
        if b not in layers:
            return a
        p = probs[b] if probs is not None else cm.last_attention(q, k, pos, scale)
        o = torch.einsum("hl,hld->hd", p.to(torch.float64), v[0, :, :pos + 1, :].double())
        a = a.clone()
        a[0, :, pos, :] = o.to(a.dtype)
        return a
    return f


class MaskedGain:
    """Per matrix and per token: sum_j y_tj^2, sum_j e_tj y_tj, sum_j e_tj^2, y = W x, e = (W_hat - W) x, products in
    `acc` as contrast_map.Gain, sums in float64. finish(masks) sums them over the tokens of each mask."""

    def __init__(self, fp16: dict, sealed: dict, names: list[str], acc):
        self.fp16, self.sealed, self.names, self.acc = fp16, sealed, names, acc
        self.known = set(names)
        self.cur: dict = {}

    def start(self):
        self.cur = {}

    def __call__(self, name: str, x):
        if name not in self.known:
            return
        xa = x.reshape(-1, x.shape[-1]).to(self.acc)
        w = self.fp16[name].to(self.acc)
        w_hat = self.sealed[name].to(self.acc)
        y = xa @ w.T
        e = xa @ (w_hat - w).T
        self.cur[name] = torch.stack(((y * y).sum(1, dtype=torch.float64), (e * y).sum(1, dtype=torch.float64),
                                      (e * e).sum(1, dtype=torch.float64)))

    def finish(self, masks):
        """masks (M, L) bool. Returns (n_names, M, 3) float64: S_yy, S_ey, S_ee."""
        t = torch.stack([self.cur[n] for n in self.names])
        return torch.einsum("nsl,ml->nms", t, masks.to(torch.float64))


def mask_stack(length: int, pos: int, massive, device):
    allm = torch.ones(length, dtype=torch.bool, device=device)
    ans = torch.zeros(length, dtype=torch.bool, device=device)
    ans[pos] = True
    return torch.stack((allm, ans, ~massive.to(device)))


# ---------------------------------------------------------------------------
# The run.
# ---------------------------------------------------------------------------
class PatchRun:
    def __init__(self, cfg, sets, names, acc, device, ans_ids, bands=PATCH_BANDS, arms=ARMS, massive_k=MASSIVE_K):
        self.cfg, self.sets, self.names, self.acc, self.device = cfg, sets, names, acc, device
        self.ans_ids, self.bands, self.arms, self.massive_k = ans_ids, bands, arms, massive_k
        self.gf, self.gs = getter(sets, "fp16"), getter(sets, "sealed")
        self.cff = torch.zeros((cfg.hidden, cfg.hidden), dtype=torch.float64, device=device)
        self.tokens = 0
        self.p1: list[dict] = []
        self.gain_f: list = []
        self.gain_s: list = []
        self.arm_names = (["fp16", "control"] + [a[0] for a in arms] + [a[0] for a in ATTN_ARMS]
                          + [a[0] for a in BRANCH_ARMS])
        self.out = {n: {"logits": [], **{l: [] for l in ROW_DEPTHS}} for n in self.arm_names}
        self.checks: list[dict] = []
        self.p1_mismatch = 0
        self.lam = self.v = self.bases = None

    # Pass 1: FP16 alone.
    def pass1(self, ids_list):
        cfg = self.cfg
        ids = torch.tensor(ids_list, dtype=torch.long, device=self.device)
        pos = cm.answer_pos(ids.shape[0])
        mg = MaskedGain(self.sets["fp16"], self.sets["sealed"], self.names, self.acc)
        mg.start()
        cap = {}

        def stream(l, h):
            if l == BAND_DEPTH:
                x = h[0].double()
                self.cff += x.T @ x
            if l == MASSIVE_DEPTH:
                cap["norms"] = h[0].float().norm(dim=-1)

        with torch.no_grad():
            z = forward(cfg, self.gf, ids, proj_hook=mg, stream_hook=stream)
            norms = cap["norms"]
            massive = massive_positions(norms, pos, self.massive_k)
            self.gain_f.append(mg.finish(mask_stack(len(ids_list), pos, massive, self.device)).cpu())
        med = float(norms.median())
        ratio = (norms / med).cpu()
        where = [int(i) for i in torch.nonzero(massive).flatten().tolist()]
        self.p1.append(dict(logits=z[pos, self.ans_ids].float().cpu(), massive=massive.cpu(), positions=where,
                            token_ids=[ids_list[i] for i in where], ratio=ratio, answer_ratio=float(ratio[pos]),
                            shown={kk: int(((ratio > kk).sum() - (1 if ratio[pos] > kk else 0))) for kk in MASSIVE_K_SHOWN}))
        self.tokens += len(ids_list)

    def set_basis(self):
        self.lam, self.v = cm.eig_desc(self.cff.cpu())
        d = self.cfg.hidden
        self.bases = [self.v[:, band_cols(lo, hi, d)].to(self.device) for lo, hi in self.bands]

    def arm_start(self, hs, hf, d, mask, basis):
        return patch_stream(hs[d], hf[d], mask, basis)

    # Pass 2: FP16, control, arms.
    def pass2(self, qi: int, ids_list, check: bool = False):
        cfg = self.cfg
        ids = torch.tensor(ids_list, dtype=torch.long, device=self.device)
        length = ids.shape[0]
        pos = cm.answer_pos(length)
        scale = cfg.head_dim ** -0.5
        massive = self.p1[qi]["massive"]
        keep_s = set(PATCH_DEPTHS) | {ATTN_LAYERS[0]}

        def rows_into(r):
            def f(l, h):
                if l in ROW_DEPTHS:
                    r[l] = h[0, pos].clone()
            return f

        with torch.no_grad():
            hf, rf, probs, fb = {}, {}, {}, {}

            def sf(l, h):
                if l in PATCH_DEPTHS or l == 0:
                    hf[l] = h
                if l in ROW_DEPTHS:
                    rf[l] = h[0, pos].clone()

            def af(b, q, k, v, a):
                if b in ATTN_LAYERS:
                    probs[b] = cm.last_attention(q, k, pos, scale)

            def bf(b, kind, out):
                if b in BRANCH_LAYERS:
                    fb[(b, kind)] = out
                return out

            z_f = forward(cfg, self.gf, ids, stream_hook=sf, attn_hook=af, branch=bf)
            lf = z_f[pos, self.ans_ids].float().cpu()
            if not torch.equal(lf, self.p1[qi]["logits"]):
                self.p1_mismatch += 1
            self._keep("fp16", lf, rf)
            del z_f

            hs, rs = {}, {}

            def ss(l, h):
                if l in keep_s:
                    hs[l] = h
                if l in ROW_DEPTHS:
                    rs[l] = h[0, pos].clone()

            mg = MaskedGain(self.sets["fp16"], self.sets["sealed"], self.names, self.acc)
            mg.start()
            explicit = []

            def ah(b, q, k, v, a):
                if b in ATTN_LAYERS:
                    p = cm.last_attention(q, k, pos, scale)
                    o = torch.einsum("hl,hld->hd", p, v[0, :, :pos + 1, :].double())
                    explicit.append(float((o - a[0, :, pos, :].double()).abs().max()))

            z_s = forward(cfg, self.gs, ids, proj_hook=mg, stream_hook=ss, attn_hook=ah if check else None)
            self.gain_s.append(mg.finish(mask_stack(length, pos, massive, self.device)).cpu())
            self._keep("control", z_s[pos, self.ans_ids].float().cpu(), rs)
            cos, sin = lvo.rope_tables(cfg, length, self.device, hs[BAND_DEPTH].dtype)

            for name, d, kind, band in self.arms:
                mask = position_mask(kind, length, pos, massive, self.device)
                basis = None if band is None else self.bases[band]
                h0 = self.arm_start(hs, hf, d, mask, basis)
                r = {d: h0[0, pos].clone()} if d in ROW_DEPTHS else {}
                z = run_from(cfg, self.gs, h0, d, cos, sin, stream_hook=rows_into(r))
                for l in ROW_DEPTHS:
                    if l < d:
                        r[l] = rs[l]
                self._keep(name, z[pos, self.ans_ids].float().cpu(), r)
                del z, h0

            start = ATTN_LAYERS[0]
            for name, src in ATTN_ARMS:
                rw = attn_rewriter(set(ATTN_LAYERS), pos, scale, probs if src == "fp16" else None)
                r = {}
                z = run_from(cfg, self.gs, hs[start], start, cos, sin, stream_hook=rows_into(r), attn_row=rw)
                self._keep(name, z[pos, self.ans_ids].float().cpu(), r)
                del z

            for name, kinds in BRANCH_ARMS:
                r = {}
                z = forward(cfg, self.gs, ids, stream_hook=rows_into(r),
                            branch=branch_replacer(fb, BRANCH_LAYERS, kinds))
                self._keep(name, z[pos, self.ans_ids].float().cpu(), r)
                del z

            if check:
                self._checks(qi, ids, pos, cos, sin, hf, hs, z_s, massive, explicit, fb)
            del z_s, hf, hs, fb

    def _keep(self, name, logits, rows):
        o = self.out[name]
        o["logits"].append(logits)
        for l in ROW_DEPTHS:
            o[l].append(rows[l].cpu())

    def _checks(self, qi, ids, pos, cos, sin, hf, hs, z_s, massive, explicit, fb):
        cfg, gs, length = self.cfg, self.gs, ids.shape[0]
        last = {n: self.out[n]["logits"][-1] for n in self.arm_names}
        add = self.checks.append
        one = position_mask("ans", length, pos, massive, self.device)
        mm = position_mask("massive", length, pos, massive, self.device)
        hd = hs[BAND_DEPTH]
        for label, mask, basis in (("whole, every position", None, None),
                                   ("band [513, 2560], every position", None, self.bases[QUIET]),
                                   ("whole, massive positions", mm, None),
                                   ("band [513, 2560], answer position", one, self.bases[QUIET])):
            h0 = patch_stream(hd, hd, mask, basis)
            z = run_from(cfg, gs, h0, BAND_DEPTH, cos, sin)
            add(dict(kind="self-patch", label=label, q=qi, ok=bool(torch.equal(h0, hd) and torch.equal(z, z_s)),
                     value=float((z - z_s).abs().max())))
        for d in PATCH_DEPTHS:
            cap_h, cap_a = {}, {}
            z_h = forward_hybrid(cfg, self.sets, ids, d, stream_hook=lambda l, h: cap_h.__setitem__(l, h))
            h0 = self.arm_start(hs, hf, d, None, None)
            z_a = run_from(cfg, gs, h0, d, cos, sin, stream_hook=lambda l, h: cap_a.__setitem__(l, h))
            streams = all(torch.equal(cap_a[l], cap_h[l]) for l in range(d + 1, cfg.layers + 1))
            ok = (torch.equal(h0, cap_h[d]) and streams and torch.equal(z_a, z_h)
                  and torch.equal(z_a[pos, self.ans_ids].float().cpu(), last[f"d{d}"]))
            add(dict(kind="hybrid", label=f"depth {d}", q=qi, ok=bool(ok), value=float((z_a - z_h).abs().max())))
            del cap_h, cap_a, z_h, z_a
        three = torch.cat(self.bases, 1)
        h3 = patch_stream(hd, hf[BAND_DEPTH], None, three)
        z3 = run_from(cfg, gs, h3, BAND_DEPTH, cos, sin)[pos, self.ans_ids].float().cpu()
        add(dict(kind="three-bands", label="bands 1-3 together against the whole stream, depth 12", q=qi,
                 ok=cm.pick_of(z3.tolist()) == cm.pick_of(last["d12"].tolist()),
                 value=float((z3 - last["d12"]).abs().max()),
                 stream=float((h3.double() - hf[BAND_DEPTH].double()).abs().max())))
        add(dict(kind="explicit-attention", label="explicit answer row against SDPA, layers 7-13", q=qi,
                 ok=True, value=max(explicit) if explicit else math.nan))
        cap_b = {}
        zb = run_from(cfg, gs, hf[0], 0, cos, sin, stream_hook=lambda l, h: cap_b.__setitem__(l, h),
                      branch=branch_replacer(fb, BRANCH_LAYERS, ("attn", "mlp")))
        okb = (torch.equal(cap_b[BAND_DEPTH], hf[BAND_DEPTH])
               and torch.equal(zb[pos, self.ans_ids].float().cpu(), last["d12"]))
        add(dict(kind="branch-exact", label="FP16 h(0) and both FP16 branches at 0-11 give FP16's h(12) and d12", q=qi,
                 ok=bool(okb), value=float((cap_b[BAND_DEPTH].double() - hf[BAND_DEPTH].double()).abs().max())))
        del cap_b, zb


# ---------------------------------------------------------------------------
# Statistics.
# ---------------------------------------------------------------------------
def mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def paired(ca, cb) -> dict:
    """a minus b on 0/1 vectors: b = a right and b wrong, c the reverse; 95 % interval d +- 1.96 sqrt(b + c -
    (b - c)^2 / n) / n, the paired normal interval; McNemar's exact two-sided p."""
    n = len(ca)
    b = int(((ca == 1) & (cb == 0)).sum())
    c = int(((ca == 0) & (cb == 1)).sum())
    d = (b - c) / n
    se = math.sqrt(max(b + c - (b - c) ** 2 / n, 0.0)) / n
    return dict(b=b, c=c, diff=d, lo=d - 1.96 * se, hi=d + 1.96 * se, p=mcnemar_p(b, c))


def boot_counts(n: int, draws: int = BOOT_DRAWS, seed: int = BOOT_SEED):
    """Resampling of the n questions with replacement, as counts (draws, n)."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, n, (draws, n), generator=g)
    return torch.zeros(draws, n, dtype=torch.float64).scatter_add_(1, idx, torch.ones(draws, n, dtype=torch.float64))


def safe_div(a, b):
    return torch.where(b != 0, a / torch.where(b != 0, b, torch.ones_like(b)), torch.full_like(a, math.nan))


def boot(cols, f, counts) -> dict:
    """cols (n, m): per-question columns; f maps column sums (k, m) to (k,). Point on all questions; 95 % percentile
    interval of the paired bootstrap; draws with a non-finite value counted and left out."""
    cols = cols.double()
    point = float(f(cols.sum(0, keepdim=True))[0])
    vals = f(counts @ cols)
    ok = torch.isfinite(vals)
    if int(ok.sum()) == 0:
        return dict(point=point, lo=math.nan, hi=math.nan, bad=int(vals.numel()))
    q = torch.quantile(vals[ok], torch.tensor([0.025, 0.975], dtype=torch.float64))
    return dict(point=point, lo=float(q[0]), hi=float(q[1]), bad=int((~ok).sum()))


def ratio_of_gains(s):
    """s columns: [x - ctl, y - ctl] summed. (x - ctl) / (y - ctl)."""
    return safe_div(s[:, 0], s[:, 1])


def ratio_of_rels(s):
    """s columns: [ee_a, yy_a, ee_b, yy_b] summed. (ee_a / yy_a) / (ee_b / yy_b)."""
    return safe_div(safe_div(s[:, 0], s[:, 1]), safe_div(s[:, 2], s[:, 3]))


def slope_recovery(s):
    """s columns: [a_arm, a_base, e] summed. (s_arm - s_base) / (1 - s_base), s = a / e."""
    return safe_div(s[:, 0] - s[:, 1], s[:, 2] - s[:, 1])


def verdict_at_least(lo: float, hi: float, thr: float) -> str:
    """Killed when the statistic is at least thr: the whole interval at or above thr kills, below thr spares."""
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return "NOT DECIDED"
    if lo >= thr:
        return "KILLED"
    if hi < thr:
        return "STANDS"
    return "NOT DECIDED"


def verdict_under(lo: float, hi: float, thr: float) -> str:
    """Killed when the statistic is under thr."""
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return "NOT DECIDED"
    if hi < thr:
        return "KILLED"
    if lo >= thr:
        return "STANDS"
    return "NOT DECIDED"


def guard_d12(verdict: str, s_d12: dict) -> str:
    """A ratio over s(d12) is read only when s(d12)'s own interval lies above 0."""
    if not (math.isfinite(s_d12["lo"]) and s_d12["lo"] > 0):
        return "NOT DECIDED (s(d12) reaches 0)"
    return verdict


def verdict_h6b(intervals: list[tuple[float, float]], k: float = H6B_MAX) -> str:
    """Killed if every type's interval lies inside [1/k, k]; stands if one lies wholly above k."""
    if any(math.isfinite(lo) and lo > k for lo, _ in intervals):
        return "STANDS"
    if all(math.isfinite(lo) and math.isfinite(hi) and lo >= 1 / k and hi <= k for lo, hi in intervals):
        return "KILLED"
    return "NOT DECIDED"


def verdict_h6c(share: tuple[float, float], err: tuple[float, float]) -> str:
    """Killed if the massive-position share is under 1/10 AND the low matrices' error moves by less than a factor 2
    (interval inside [1/2, 2]); stands if the share is at least 1/10 or the error moves by 2 or more."""
    s_lo, s_hi = share
    e_lo, e_hi = err
    fin = all(math.isfinite(x) for x in (s_lo, s_hi, e_lo, e_hi))
    if fin and (s_lo >= H6C_SHARE or e_lo >= H6C_ERR or e_hi <= 1 / H6C_ERR):
        return "STANDS"
    if fin and s_hi < H6C_SHARE and e_lo > 1 / H6C_ERR and e_hi < H6C_ERR:
        return "KILLED"
    return "NOT DECIDED"


def band_terms(x, y, basis):
    """Per question a_i = sum_k (y V)_ik (x V)_ik and e_i = sum_k (x V)_ik^2 over the band's directions: the
    energy-weighted band slope of contrast_map.band_table is sum a / sum e."""
    xv = x.double() @ basis.double()
    yv = y.double() @ basis.double()
    return (xv * yv).sum(1), (xv * xv).sum(1)


def shape_cos(x, f) -> tuple[float, bool]:
    """Cosine between two four-option logit vectors, each centred on its own mean over the four options. Pure Python,
    float64. Returns (c, degenerate): c = 0 and degenerate = True when a centred vector is all zeros."""
    mx, mf = sum(x) / len(x), sum(f) / len(f)
    xc = [v - mx for v in x]
    fc = [v - mf for v in f]
    nx = math.sqrt(sum(v * v for v in xc))
    nf = math.sqrt(sum(v * v for v in fc))
    if nx == 0.0 or nf == 0.0:
        return 0.0, True
    return sum(a * b for a, b in zip(xc, fc)) / (nx * nf), False


def log_softmax4(x) -> list[float]:
    m = max(x)
    s = math.log(sum(math.exp(v - m) for v in x)) + m
    return [v - s for v in x]


def kl4(f, x) -> float:
    """KL(softmax(f) || softmax(x)) over the four options, nats: from FP16 (f) to the arm (x)."""
    lf, lx = log_softmax4(f), log_softmax4(x)
    return sum(math.exp(a) * (a - b) for a, b in zip(lf, lx))


def logp_of(x, k: int) -> float:
    """The log-probability the arm gives option k over the four options."""
    return log_softmax4(x)[k]


def shape_share(c_x, c_ctl) -> float:
    """s(X) = (C(X) - C(control)) / (1 - C(control)), C the mean cosine."""
    cx, cc = sum(c_x) / len(c_x), sum(c_ctl) / len(c_ctl)
    return (cx - cc) / (1.0 - cc) if cc != 1.0 else math.nan


def ranks(v) -> list[float]:
    """Ranks 1..n, ties at their average rank."""
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b) -> float:
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else math.nan


def instrument_table(f16: dict, ctl: dict, arms: dict) -> dict:
    """The instrument check on dumps already read ({key: {"answer", "pick", "logits"}}): per arm the cosine share
    s(X) and the accuracy share (acc_X - acc_ctl) / (acc_fp16 - acc_ctl), every key of the FP16 dump; Spearman of
    the two columns over the arms; the rule of the prereg."""
    keys = sorted(f16)
    acc = lambda d: sum(d[k]["pick"] == d[k]["answer"] for k in keys) / len(keys)  # noqa: E731
    cos_of = lambda d: [shape_cos(d[k]["logits"], f16[k]["logits"]) for k in keys]  # noqa: E731
    cc = cos_of(ctl)
    c_ctl = [c for c, _ in cc]
    a_f, a_c = acc(f16), acc(ctl)
    rows = {}
    for name, d in arms.items():
        cx = cos_of(d)
        rows[name] = dict(cos_share=shape_share([c for c, _ in cx], c_ctl),
                          acc_share=(acc(d) - a_c) / (a_f - a_c) if a_f != a_c else math.nan,
                          mean_cos=sum(c for c, _ in cx) / len(cx), degenerate=sum(z for _, z in cx), acc=acc(d))
    names = list(rows)
    rho = spearman([rows[n]["cos_share"] for n in names], [rows[n]["acc_share"] for n in names])
    gap = max(abs(rows[n]["cos_share"] - rows[n]["acc_share"]) for n in names)
    return dict(rows=rows, rho=rho, max_abs_diff=gap, acc_fp16=a_f, acc_control=a_c,
                mean_cos_control=sum(c_ctl) / len(c_ctl), degenerate_control=sum(z for _, z in cc),
                primary="cosine" if (math.isfinite(rho) and rho >= INSTRUMENT_RHO_MIN) else "accuracy")


def instrument(args) -> int:
    """--instrument: reads the pinned dumps only; no model, no torch computation."""
    root = Path(args.dumps_dir or Path(__file__).resolve().parent.parent / INSTRUMENT_DIR)
    f16 = cm.read_dump(root / Path(cm.DUMP_F16[0]).name, cm.DUMP_F16[1])
    ctl = cm.read_dump(root / Path(cm.DUMP_SEALED[0]).name, cm.DUMP_SEALED[1])
    arms = {n: cm.read_dump(root / fn, sha) for n, (fn, sha) in INSTRUMENT_ARMS.items()}
    t = instrument_table(f16, ctl, arms)
    lines = [f"INSTRUMENT: {len(f16)} questions, each restore arm against {Path(cm.DUMP_F16[0]).name}, control "
             f"{Path(cm.DUMP_SEALED[0]).name}; accuracy FP16 {t['acc_fp16'] * 100:.2f}, control "
             f"{t['acc_control'] * 100:.2f}; control mean cosine {t['mean_cos_control']:.5f}, degenerate "
             f"{t['degenerate_control']}",
             "  arm           acc     mean cos   cosine share   accuracy share   |diff|   degenerate"]
    for n, r in t["rows"].items():
        lines.append(f"  {n:<12}  {r['acc'] * 100:6.2f}  {r['mean_cos']:.5f}    {r['cos_share']:+.4f}        "
                     f"{r['acc_share']:+.4f}          {abs(r['cos_share'] - r['acc_share']):.4f}   {r['degenerate']}")
    lines.append(f"  Spearman over the {len(t['rows'])} arms {t['rho']:.4f} (rule: at least {INSTRUMENT_RHO_MIN}); "
                 f"largest |difference| {t['max_abs_diff']:.4f}")
    lines.append(f"  PRIMARY: {t['primary'].upper()} SHARE"
                 + ("" if t["primary"] == "cosine" else "; H6-A and H6-C (i) are underpowered (prereg section 7)"))
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "instrument.txt").write_text(text + "\n")
    return 0


# ---------------------------------------------------------------------------
# References from contrast-map-4b.
# ---------------------------------------------------------------------------
def read_tsv(path: Path) -> list[dict]:
    lines = [ln for ln in path.read_text().splitlines() if ln and not ln.startswith("#")]
    cols = lines[0].split("\t")
    return [dict(zip(cols, ln.split("\t"))) for ln in lines[1:]]


def read_ref(d: Path) -> dict:
    for name, want in REF_FILES.items():
        got = lvo.sha256_file(d / name)
        if got != want:
            raise SystemExit(f"refused: {d / name} sha256 {got}, expected {want}")
    ctl = {(r["subject"], int(r["index"])): r for r in read_tsv(d / "control.tsv")}
    qs = [(r["subject"], int(r["index"]), int(r["qhash"], 16)) for r in read_tsv(d / "questions.tsv")]
    mats = {(int(r["layer"]), r["proj"]): r for r in read_tsv(d / "matrices.tsv")}
    lens25 = next(r for r in read_tsv(d / "lens.tsv") if r["depth"] == "25")
    bands12 = [r for r in read_tsv(d / "stream_bands.tsv") if r["depth"] == "12"]
    return dict(control=ctl, questions=qs, matrices=mats, lens25=lens25, bands12=bands12)


# ---------------------------------------------------------------------------
# Analysis and files.
# ---------------------------------------------------------------------------
def analyse(out: Path, run: PatchRun, keys, answers, qids, dump_f, dump_s, ref=None, extra_lines=(),
            primary: str = "cosine") -> dict:
    cfg = run.cfg
    N = len(keys)
    lines = list(extra_lines)
    say = lines.append
    names = run.arm_names
    ans = torch.tensor(answers)
    L = {n: torch.stack(run.out[n]["logits"]).double() for n in names}
    picks = {n: torch.tensor([cm.pick_of(r) for r in L[n].tolist()]) for n in names}
    corr = {n: (picks[n] == ans).double() for n in names}
    head_f = run.sets["fp16"]["model.embed_tokens.weight"] if cfg.tied else run.sets["fp16"]["lm_head.weight"]
    rows_f = head_f[run.ans_ids].cpu()
    norm_f = run.sets["fp16"]["model.norm.weight"].cpu()
    Z = {n: cm.lens(torch.stack(run.out[n][LENS_DEPTH]), norm_f, rows_f, cfg.eps) for n in names}
    lpick = {n: torch.tensor([cm.pick_of(r) for r in Z[n].tolist()]) for n in names}
    lcorr = {n: (lpick[n] == ans).double() for n in names}
    counts = boot_counts(N)
    res: dict = {}

    # Gates.
    gates = {}
    ctl = cm.control(keys, L["fp16"], L["control"], dump_f, dump_s)
    gates["rule D"] = ctl["green"]
    say(f"GATES, {N} questions")
    say(f"  rule D: picks equal to the Rust dumps {ctl['fp16']['picks_equal']:.4f} (FP16), "
        f"{ctl['sealed']['picks_equal']:.4f} (control); slope script {ctl['fit_script'][0]:.4f} dump "
        f"{ctl['fit_dump'][0]:.4f}: {'GREEN' if ctl['green'] else 'RED'}")
    sp = [c for c in run.checks if c["kind"] == "self-patch"]
    hy = [c for c in run.checks if c["kind"] == "hybrid"]
    tb = [c for c in run.checks if c["kind"] == "three-bands"]
    ex = [c for c in run.checks if c["kind"] == "explicit-attention"]
    nq = len({c["q"] for c in run.checks})
    gates["self-patch bit for bit"] = bool(sp) and all(c["ok"] for c in sp)
    gates["hybrid bit for bit"] = bool(hy) and all(c["ok"] for c in hy)
    gates["FP16 pass 2 = pass 1"] = run.p1_mismatch == 0
    say(f"  self-patch with the sealed stream, 4 forms, {nq} prompts: {sum(c['ok'] for c in sp)} of {len(sp)} bit for "
        f"bit (worst |dlogit| {max((c['value'] for c in sp), default=math.nan):.3e})")
    say(f"  whole-stream patch at depths {PATCH_DEPTHS} against FP16 blocks 0..d-1 then sealed, every later depth "
        f"and the output: {sum(c['ok'] for c in hy)} of {len(hy)} bit for bit")
    say(f"  FP16 logits, pass 2 against pass 1: {N - run.p1_mismatch} of {N} bit for bit")
    tb_agree = sum(c["ok"] for c in tb) / len(tb) if tb else math.nan
    gates["three bands = whole"] = bool(tb) and tb_agree >= PICK_AGREE_MIN
    say(f"  three bands together against the whole stream at depth 12: picks equal {tb_agree:.4f}, worst "
        f"|dlogit| {max((c['value'] for c in tb), default=math.nan):.3e}, worst |dstream| "
        f"{max((c['stream'] for c in tb), default=math.nan):.3e}")
    ex_agree = float((picks["attn_explicit"] == picks["control"]).double().mean())
    gates["explicit attention = SDPA"] = ex_agree >= PICK_AGREE_MIN
    say(f"  explicit attention row, no patch, against SDPA: picks equal to control {ex_agree:.4f} on {N}, worst "
        f"|dlogit| {float((L['attn_explicit'] - L['control']).abs().max()):.3e}, worst |d attention row| "
        f"{max((c['value'] for c in ex), default=math.nan):.3e} on {len(ex)} prompts")
    be = [c for c in run.checks if c["kind"] == "branch-exact"]
    gates["branches bit for bit"] = bool(be) and all(c["ok"] for c in be)
    say(f"  FP16 h(0) and FP16's two branches at blocks 0-11 in the sealed run give FP16's h(12) and d12's logits: "
        f"{sum(c['ok'] for c in be)} of {len(be)} bit for bit (worst |dh(12)| "
        f"{max((c['value'] for c in be), default=math.nan):.3e})")
    gm = {}
    for which, store in (("fp16", run.gain_f), ("sealed", run.gain_s)):
        gm[which] = torch.stack(store)  # (N, n_names, M, 3)
    tot = {w: gm[w].sum(0) for w in gm}  # (n_names, M, 3)
    if ref is not None:
        acc_n = {n: int(corr[n].sum()) for n in ("fp16", "control")}
        rp = {}
        for n, col in (("fp16", "pick_fp16_script"), ("control", "pick_sealed_script")):
            rp[n] = sum(int(picks[n][i]) == int(ref["control"][k][col]) for i, k in enumerate(keys)) / N
        ok = all(abs(acc_n[n] - REF_ACC[n]) <= REPRO_COUNT_TOL and rp[n] >= REPRO_PICKS_MIN for n in acc_n)
        gates["contrast-map reproduced"] = ok
        say(f"  against contrast-map-4b: FP16 {acc_n['fp16']} right ({REF_ACC['fp16']} there), control "
            f"{acc_n['control']} ({REF_ACC['control']}); picks equal {rp['fp16']:.4f} and {rp['control']:.4f}: "
            f"{'GREEN' if ok else 'RED'}")
        worst = 0.0
        for i, nm in enumerate(run.names):
            layer, proj = int(nm.split(".")[2]), nm.split(".")[4]
            for w, col in (("fp16", "rel_fp16"), ("sealed", "rel_sealed")):
                mine = float(tot[w][i, 0, 2] / tot[w][i, 0, 0])
                theirs = float(ref["matrices"][(layer, proj)][col])
                worst = max(worst, abs(mine - theirs) / theirs)
        gates["R4 relative error reproduced"] = worst <= GAIN_REL_TOL
        say(f"  all-position relative error of the 252 matrices against contrast-map's matrices.tsv, both inputs: "
            f"worst relative difference {worst:.2e} (at most {GAIN_REL_TOL:g})")
        lf25, ls25 = float(ref["lens25"]["lens_acc_fp16"]), float(ref["lens25"]["lens_acc_sealed"])
        say(f"  reported: lens accuracy at depth 25, FP16 {float(lcorr['fp16'].mean()):.4f} ({lf25:.4f} there), "
            f"control {float(lcorr['control'].mean()):.4f} ({ls25:.4f} there)")
        share = [float(run.lam[cm.band_slice(lo, hi, cfg.hidden)].sum() / run.lam.sum()) for lo, hi in cm.BANDS]
        theirs = [float(r["energy_share"]) for r in ref["bands12"] if r["positions"] == "all"]
        say("  reported: depth-12 energy shares of contrast-map's five bands, here / there: "
            + ", ".join(f"{a:.4e}/{b:.4e}" for a, b in zip(share, theirs)))
    green = all(gates.values())
    res["gates"], res["green"] = gates, green
    say(f"  GATES: {'GREEN' if green else 'RED, NOTHING BELOW IS READ'} "
        f"({', '.join(k for k, v in gates.items() if not v) or 'all pass'})")

    # Arms.
    say("")
    say("ARMS: accuracy at the output and through FP16's lens at depth 25; paired against the control; share of the "
        "FP16-control gap recovered, paired bootstrap 95 %")
    d = cfg.hidden
    x12 = torch.stack(run.out["fp16"][BAND_DEPTH])
    bases = [run.v[:, band_cols(lo, hi, d)] for lo, hi in run.bands]
    terms = {n: [band_terms(x12, torch.stack(run.out[n][BAND_DEPTH]), b) for b in bases] for n in names}
    rows = []
    gap = corr["fp16"] - corr["control"]
    lgap = lcorr["fp16"] - lcorr["control"]
    for n in names:
        pr = paired(corr[n], corr["control"])
        sh = boot(torch.stack((corr[n] - corr["control"], gap), 1), ratio_of_gains, counts)
        lsh = boot(torch.stack((lcorr[n] - lcorr["control"], lgap), 1), ratio_of_gains, counts)
        sl = [float(a.sum() / e.sum()) for a, e in terms[n]]
        rows.append(dict(arm=n, acc=float(corr[n].mean()), lens=float(lcorr[n].mean()), **{f"p_{k}": v for k, v in pr.items()},
                         share=sh, lshare=lsh, slopes=sl,
                         changed=float((picks[n] != picks["control"]).double().mean())))
    res["arms"] = {r["arm"]: r for r in rows}
    bnames = [f"[{lo}, {hi}]" for lo, hi in run.bands]
    say("  arm                  acc     lens25  minus control [95 %]        b/c      p        share of gap [95 %]     "
        "lens share [95 %]       slope d12 " + " ".join(bnames))
    for r in rows:
        say(f"  {r['arm']:<20} {r['acc'] * 100:6.2f}  {r['lens'] * 100:6.2f}  {r['p_diff'] * 100:+6.2f} "
            f"[{r['p_lo'] * 100:+6.2f}, {r['p_hi'] * 100:+6.2f}]  {r['p_b']:>3}/{r['p_c']:<3}  {r['p_p']:.2e}  "
            f"{r['share']['point']:+.3f} [{r['share']['lo']:+.3f}, {r['share']['hi']:+.3f}]  "
            f"{r['lshare']['point']:+.3f} [{r['lshare']['lo']:+.3f}, {r['lshare']['hi']:+.3f}]  "
            + " ".join(f"{s:.3f}" for s in r["slopes"]))
    with open(out / "arms.tsv", "w") as f:
        f.write("arm\tacc\tlens25_acc\tdiff_vs_control\tdiff_lo\tdiff_hi\tb\tc\tmcnemar_p\tshare\tshare_lo\tshare_hi\t"
                "share_bad_draws\tlens_share\tlens_share_lo\tlens_share_hi\tpicks_changed\t"
                + "\t".join(f"slope_d12_{lo}-{hi}" for lo, hi in run.bands) + "\n")
        for r in rows:
            f.write(f"{r['arm']}\t{r['acc']:.6f}\t{r['lens']:.6f}\t{r['p_diff']:.6f}\t{r['p_lo']:.6f}\t{r['p_hi']:.6f}\t"
                    f"{r['p_b']}\t{r['p_c']}\t{r['p_p']:.6g}\t{r['share']['point']:.6f}\t{r['share']['lo']:.6f}\t"
                    f"{r['share']['hi']:.6f}\t{r['share']['bad']}\t{r['lshare']['point']:.6f}\t"
                    f"{r['lshare']['lo']:.6f}\t{r['lshare']['hi']:.6f}\t{r['changed']:.6f}\t"
                    + "\t".join(f"{s:.6f}" for s in r["slopes"]) + "\n")

    # The primary statistic: the shape agreement of the four logits with FP16's (prereg section 6).
    say("")
    say("SHAPE: c = cosine of the centred four logits with FP16's; share s = (C - C_control) / (1 - C_control), "
        "output and lens 25, paired bootstrap 95 %; mean KL(FP16 || arm) over the four options; mean log-probability "
        "of FP16's pick")
    lf_ = L["fp16"].tolist()
    zf_ = Z["fp16"].tolist()
    pf_ = picks["fp16"].tolist()
    per = {}
    for n in names:
        lx, zx = L[n].tolist(), Z[n].tolist()
        co = [shape_cos(lx[i], lf_[i]) for i in range(N)]
        cl = [shape_cos(zx[i], zf_[i]) for i in range(N)]
        per[n] = dict(c=torch.tensor([c for c, _ in co], dtype=torch.float64),
                      cl=torch.tensor([c for c, _ in cl], dtype=torch.float64),
                      deg=sum(z for _, z in co), degl=sum(z for _, z in cl),
                      kl=torch.tensor([kl4(lf_[i], lx[i]) for i in range(N)], dtype=torch.float64),
                      lp=torch.tensor([logp_of(lx[i], pf_[i]) for i in range(N)], dtype=torch.float64))
    one = torch.ones(N, dtype=torch.float64)
    mean_of = lambda v: boot(torch.stack((v, one), 1), ratio_of_gains, counts)  # noqa: E731
    sh_rows = {}
    for n in names:
        p, q = per[n], per["control"]
        sh_rows[n] = dict(C=float(p["c"].mean()), Cl=float(p["cl"].mean()),
                          s=boot(torch.stack((p["c"] - q["c"], one - q["c"]), 1), ratio_of_gains, counts),
                          sl=boot(torch.stack((p["cl"] - q["cl"], one - q["cl"]), 1), ratio_of_gains, counts),
                          kl=mean_of(p["kl"]), lp=mean_of(p["lp"]), deg=p["deg"], degl=p["degl"])
    res["shape"] = sh_rows
    say("  arm                  C        s [95 %]                 C lens   s lens [95 %]            KL [95 %]"
        "                 log p(FP16 pick) [95 %]    degenerate")
    for n, r in sh_rows.items():
        say(f"  {n:<20} {r['C']:.5f}  {r['s']['point']:+.3f} [{r['s']['lo']:+.3f}, {r['s']['hi']:+.3f}]  "
            f"{r['Cl']:.5f}  {r['sl']['point']:+.3f} [{r['sl']['lo']:+.3f}, {r['sl']['hi']:+.3f}]  "
            f"{r['kl']['point']:.4f} [{r['kl']['lo']:.4f}, {r['kl']['hi']:.4f}]  "
            f"{r['lp']['point']:+.4f} [{r['lp']['lo']:+.4f}, {r['lp']['hi']:+.4f}]  {r['deg']}/{r['degl']}")
    with open(out / "shape.tsv", "w") as f:
        f.write("arm\tC\ts\ts_lo\ts_hi\tC_lens\ts_lens\ts_lens_lo\ts_lens_hi\tkl\tkl_lo\tkl_hi\tlogp_fp16_pick\tlogp_lo\t"
                "logp_hi\tdegenerate_out\tdegenerate_lens\n")
        for n, r in sh_rows.items():
            trip = lambda k: "\t".join(f"{r[k][x]:.6f}" for x in ("point", "lo", "hi"))  # noqa: E731
            f.write(f"{n}\t{r['C']:.7f}\t{trip('s')}\t{r['Cl']:.7f}\t{trip('sl')}\t{trip('kl')}\t{trip('lp')}\t"
                    f"{r['deg']}\t{r['degl']}\n")
    if ref is not None:
        xa = x12.double()
        ya = torch.stack(run.out["control"][BAND_DEPTH]).double()
        bt = cm.band_table(run.v, None, xa.T @ xa, ya.T @ xa, ya.T @ ya)
        say(f"  reported: control's answer-position slope of band [513, 2560] at depth 12, contrast_map.band_table "
            f"{bt['rows'][4]['slope_w']:.4f} (0.4697 in contrast-map), here {rows[1]['slopes'][QUIET]:.4f}")

    # H6-B and H6-C: the matrices by position.
    say("")
    say("MATRICES BY POSITION: relative error S_ee/S_yy pooled over the 36 layers of a type, answer position against "
        "every position, and every position but the massive ones; FP16 inputs (primary) and sealed inputs")
    tcols = {}
    say("  type        x       rel all      rel answer   answer/all [95 %]          rel nonmassive  nonmassive/all")
    h6b = []
    for p in lvo.PROJ_TYPES:
        idx = [i for i, nm in enumerate(run.names) if nm.split(".")[4] == p]
        for w in ("fp16", "sealed"):
            g = gm[w][:, idx].sum(1)  # (N, M, 3)
            cols = torch.stack((g[:, 1, 2], g[:, 1, 0], g[:, 0, 2], g[:, 0, 0]), 1)
            bt_ = boot(cols, ratio_of_rels, counts)
            cn = torch.stack((g[:, 2, 2], g[:, 2, 0], g[:, 0, 2], g[:, 0, 0]), 1)
            bn = boot(cn, ratio_of_rels, counts)
            s = g.sum(0)
            rel = [float(s[m, 2] / s[m, 0]) for m in range(len(MASKS))]
            tcols[(p, w)] = dict(rel=rel, ans=bt_, nm=bn)
            if w == "fp16":
                h6b.append((bt_["lo"], bt_["hi"]))
            say(f"  {p:<10}  {w:<6}  {rel[0]:.5e}  {rel[1]:.5e}  {bt_['point']:.3f} [{bt_['lo']:.3f}, {bt_['hi']:.3f}]"
                f"     {rel[2]:.5e}     {bn['point']:.3f}")
    with open(out / "types_pos.tsv", "w") as f:
        f.write("proj\tinputs\trel_all\trel_answer\trel_nonmassive\tanswer_over_all\tlo\thi\tnonmassive_over_all\tlo\thi\n")
        for (p, w), t in tcols.items():
            f.write(f"{p}\t{w}\t{t['rel'][0]:.9g}\t{t['rel'][1]:.9g}\t{t['rel'][2]:.9g}\t{t['ans']['point']:.6f}\t"
                    f"{t['ans']['lo']:.6f}\t{t['ans']['hi']:.6f}\t{t['nm']['point']:.6f}\t{t['nm']['lo']:.6f}\t"
                    f"{t['nm']['hi']:.6f}\n")
    with open(out / "matrices_pos.tsv", "w") as f:
        f.write("# per matrix, summed over the questions: S_yy, S_ey, S_ee for masks all / answer / nonmassive, x of the\n"
                "# FP16 pass (_fp16) and of the sealed control pass (_sealed); rel = S_ee/S_yy, a = 1 + S_ey/S_yy\n")
        f.write("layer\tproj\t" + "\t".join(f"{x}_{m}_{w}" for w in ("fp16", "sealed") for m in MASKS
                                           for x in ("rel", "a", "syy", "sey", "see")) + "\n")
        for i, nm in enumerate(run.names):
            vals = []
            for w in ("fp16", "sealed"):
                for m in range(len(MASKS)):
                    syy, sey, see = (float(v) for v in tot[w][i, m])
                    vals += [see / syy if syy else math.nan, 1 + sey / syy if syy else math.nan, syy, sey, see]
            f.write(f"{nm.split('.')[2]}\t{nm.split('.')[4]}\t" + "\t".join(f"{v:.9g}" for v in vals) + "\n")
    low = [run.names.index(n) for n in LOW_REL if n in run.names]
    if low:
        g = gm["fp16"][:, low].sum(1)
        h6c_err = boot(torch.stack((g[:, 2, 2], g[:, 2, 0], g[:, 0, 2], g[:, 0, 0]), 1), ratio_of_rels, counts)
        say(f"  the four low matrices ({', '.join(n.split('.')[2] + ' ' + n.split('.')[4] for n in LOW_REL)}), FP16 "
            f"inputs, pooled: nonmassive/all {h6c_err['point']:.3f} [{h6c_err['lo']:.3f}, {h6c_err['hi']:.3f}]; each: "
            + ", ".join(f"{float(tot['fp16'][i, 2, 2] / tot['fp16'][i, 2, 0]) / float(tot['fp16'][i, 0, 2] / tot['fp16'][i, 0, 0]):.3f}"
                        for i in low))
    else:
        h6c_err = dict(point=math.nan, lo=math.nan, hi=math.nan, bad=0)

    # Massive positions.
    say("")
    cnt = [len(p["positions"]) for p in run.p1]
    hist = {k: sum(1 for c in cnt if (c == k if k < 5 else c >= 5)) for k in range(6)}
    tok: dict[int, int] = {}
    for p in run.p1:
        for t in p["token_ids"]:
            tok[t] = tok.get(t, 0) + 1
    top = sorted(tok.items(), key=lambda kv: -kv[1])[:20]
    first = sum(1 for p in run.p1 if 0 in p["positions"])
    ansq = sum(1 for p in run.p1 if p["answer_ratio"] > run.massive_k)
    say(f"MASSIVE POSITIONS: FP16 h({MASSIVE_DEPTH}) norm over the prompt's median above {run.massive_k:g}, answer "
        f"position excluded. A prompt holds {min(cnt)} to {max(cnt)}, median {sorted(cnt)[len(cnt) // 2]}; prompts with "
        f"0, 1, 2, 3, 4, 5+: {', '.join(str(hist[k]) for k in range(6))}. Position 0 is massive in {first} of {N}; the "
        f"answer position would qualify in {ansq}.")
    say("  token ids (count): " + ", ".join(f"{t} ({c})" for t, c in top))
    say("  positions a prompt above the other multiples, mean: "
        + ", ".join(f"{kk:g}: {sum(p['shown'][kk] for p in run.p1) / N:.2f}" for kk in MASSIVE_K_SHOWN))
    with open(out / "massive.tsv", "w") as f:
        f.write("order\tsubject\tindex\tlength\tn_massive\tpositions\ttoken_ids\tmax_ratio_non_answer\tanswer_ratio\n")
        for i, (k, p) in enumerate(zip(keys, run.p1)):
            r = p["ratio"].clone()
            r[-1] = 0.0
            f.write(f"{i}\t{k[0]}\t{k[1]}\t{len(qids[i])}\t{len(p['positions'])}\t"
                    f"{','.join(map(str, p['positions']))}\t{','.join(map(str, p['token_ids']))}\t"
                    f"{float(r.max()):.4f}\t{p['answer_ratio']:.4f}\n")

    # Kills.
    say("")
    say(f"KILL RULES (prereg table; KILLED or STANDS only when the whole 95 % interval is on one side). Primary share: "
        f"{primary.upper()}, fixed by the instrument check before the job; the other is printed beside, no rule.")
    fam = {"cosine": ({n: per[n]["c"] for n in names}, {n: sh_rows[n]["s"] for n in names}),
           "accuracy": ({n: corr[n] for n in names}, {n: res["arms"][n]["share"] for n in names})}

    def over_d12(v, arm):
        return boot(torch.stack((v[arm] - v["control"], v["d12"] - v["control"]), 1), ratio_of_gains, counts)

    ratios = {}
    for fname, (v, sd) in fam.items():
        ratios[fname] = {a: over_d12(v, a) for a in ("d12_b513-2560", "d12_massive", "mlp_0-11", "attn_0-11")}
    sdp = fam[primary][1]
    rp = ratios[primary]
    aq, eq = terms["attn_fp16_7-13"][QUIET]
    a0, _ = terms["attn_explicit"][QUIET]
    h6d = boot(torch.stack((aq, a0, eq), 1), slope_recovery, counts)
    ac, _ = terms["control"][QUIET]
    h6d_ctl = boot(torch.stack((aq, ac, eq), 1), slope_recovery, counts)
    sd12 = sdp["d12"]
    h6a, h6cs, h6e = rp["d12_b513-2560"], rp["d12_massive"], rp["mlp_0-11"]
    kills = [
        ("H6-A", "s(d12_b513-2560) / s(d12)", h6a, guard_d12(verdict_at_least(h6a["lo"], h6a["hi"], H6A_MIN), sd12),
         f"killed if >= {H6A_MIN}"),
        ("H6-B", "answer-position over all-position relative error, per type, FP16 inputs (worst type shown)",
         max((dict(point=tcols[(p, 'fp16')]['ans']['point'], lo=lo, hi=hi) for p, (lo, hi) in zip(lvo.PROJ_TYPES, h6b)),
             key=lambda t: t["point"]),
         verdict_h6b(h6b), f"killed if every type in [1/{H6B_MAX}, {H6B_MAX}]"),
        ("H6-C", "s(d12_massive) / s(d12)", h6cs,
         guard_d12(verdict_h6c((h6cs["lo"], h6cs["hi"]), (h6c_err["lo"], h6c_err["hi"])), sd12),
         f"killed if < {H6C_SHARE} and the low matrices' error ratio in [1/{H6C_ERR}, {H6C_ERR}]"),
        ("H6-C err", "low matrices, error without the massive positions over with them", h6c_err, "(read with H6-C)",
         ""),
        ("H6-D", "share of the band [513, 2560] slope deficit at d12 recovered by FP16's attention at 7-13", h6d,
         verdict_under(h6d["lo"], h6d["hi"], H6D_MIN), f"killed if < {H6D_MIN}"),
        ("H6-E", "s(mlp_0-11) / s(d12)", h6e, guard_d12(verdict_under(h6e["lo"], h6e["hi"], H6E_MIN), sd12),
         f"killed if < {H6E_MIN}"),
    ]
    res["kills"] = {k[0]: dict(stat=k[2], verdict=k[3]) for k in kills}
    say(f"  s(d12), the denominator: {sd12['point']:+.3f} [{sd12['lo']:+.3f}, {sd12['hi']:+.3f}]")
    for name, what, st, vd, rule in kills:
        say(f"  {name:<9} {st['point']:+.3f} [{st['lo']:+.3f}, {st['hi']:+.3f}]  {vd:<12} {what}; {rule}")
    a0e = rp["attn_0-11"]
    say(f"  beside H6-E, no rule: s(attn_0-11) / s(d12) {a0e['point']:+.3f} [{a0e['lo']:+.3f}, {a0e['hi']:+.3f}]")
    say(f"  H6-D against the SDPA control instead of the explicit path: {h6d_ctl['point']:+.3f} "
        f"[{h6d_ctl['lo']:+.3f}, {h6d_ctl['hi']:+.3f}]")
    other = "accuracy" if primary == "cosine" else "cosine"
    say(f"  the same ratios on the {other} share, no rule"
        + (" (underpowered, about +-0.51 of the gap, prereg section 7)" if other == "accuracy" else "") + ": "
        + ", ".join(f"{a} {r['point']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]" for a, r in ratios[other].items()))
    with open(out / "kills.tsv", "w") as f:
        f.write("rule\tstatistic\tpoint\tlo\thi\tbad_draws\tverdict\tkill_condition\tprimary\n")
        for name, what, st, vd, rule in kills:
            f.write(f"{name}\t{what}\t{st['point']:.6f}\t{st['lo']:.6f}\t{st['hi']:.6f}\t{st.get('bad', 0)}\t{vd}\t"
                    f"{rule}\t{primary}\n")

    # Per question.
    with open(out / "picks.tsv", "w") as f:
        f.write("order\tsubject\tindex\tanswer\t" + "\t".join(f"pick_{n}\tlens25_{n}" for n in names) + "\n")
        for i, k in enumerate(keys):
            f.write(f"{i}\t{k[0]}\t{k[1]}\t{answers[i]}\t"
                    + "\t".join(f"{int(picks[n][i])}\t{int(lpick[n][i])}" for n in names) + "\n")
    with open(out / "logits.tsv", "w") as f:
        f.write("subject\tindex\tarm\t" + "\t".join(f"out_{x}" for x in "abcd") + "\t"
                + "\t".join(f"lens25_{x}" for x in "abcd") + "\n")
        for n in names:
            for i, k in enumerate(keys):
                f.write(f"{k[0]}\t{k[1]}\t{n}\t" + "\t".join(f"{v:.7g}" for v in L[n][i].tolist() + Z[n][i].tolist())
                        + "\n")
    with open(out / "checks.tsv", "w") as f:
        f.write("kind\tlabel\tquestion\tok\tvalue\n")
        for c in run.checks:
            f.write(f"{c['kind']}\t{c['label']}\t{c['q']}\t{c['ok']}\t{c['value']:.6e}\n")
    for w in gm:
        cm.write_raw(out / f"gains_{w}x.f64", gm[w], keys)
    cm.write_raw(out / "eigvals_d12.f64", run.lam, keys)
    say("")
    say("RAW: arms.tsv, shape.tsv, picks.tsv, logits.tsv, kills.tsv, types_pos.tsv, matrices_pos.tsv, massive.tsv, checks.tsv, "
        "gains_{fp16,sealed}x.f64 (N x 252 x 3 masks x [S_yy, S_ey, S_ee]), eigvals_d12.f64")
    text = "\n".join(lines)
    (out / "report.txt").write_text(text + "\n")
    print(text)
    return res


# ---------------------------------------------------------------------------
# Selftest: tiny random models, CPU, float64.
# ---------------------------------------------------------------------------
def selftest() -> int:
    import tempfile

    torch.manual_seed(0)
    fails: list[str] = []
    n_checks = [0]

    def check(ok, what: str):
        n_checks[0] += 1
        print(f"  {'PASS' if ok else 'FAIL'}  {what}")
        if not ok:
            fails.append(what)

    print("selftest: tiny Qwen3-shaped models (36 layers, d 16), CPU, float64")
    cfg = lvo.Cfg(lvo.TINY)
    cpu = torch.device("cpu")
    fp16 = lvo.tiny_weights(cfg, 1)
    PLANT = 5
    fp16["model.embed_tokens.weight"][PLANT] *= 60.0
    sealed = {k: v.clone() for k, v in fp16.items()}
    g0 = torch.Generator().manual_seed(3)
    names = lvo.all_projections(cfg.layers)
    for n in names:
        sealed[n] = fp16[n] + 0.08 * torch.randn(fp16[n].shape, generator=g0, dtype=torch.float64) * fp16[n].abs().mean()
    sealed["model.embed_tokens.weight"] = fp16["model.embed_tokens.weight"] * (
        1 + 0.03 * torch.randn(fp16["model.embed_tokens.weight"].shape, generator=g0, dtype=torch.float64))
    sets = {"fp16": fp16, "sealed": sealed}
    gf, gs = getter(sets, "fp16"), getter(sets, "sealed")
    ans_ids = [3, 7, 11, 13]
    gen = torch.Generator().manual_seed(5)
    plants = ([0, 4], [2], [], [0, 7, 9], [3], [0])
    prompts = []
    for n, pl in zip((19, 23, 17, 29, 21, 25), plants):
        p = torch.randint(6, cfg.vocab, (n,), generator=gen).tolist()
        for i in pl:
            p[i] = PLANT
        prompts.append(p)
    prompts[4][-1] = PLANT  # a planted answer position: never massive
    bands = ((1, 2), (3, 6), (7, 16))

    # 1. The cut forward against contrast_map.forward_taps and layer_vs_output.forward.
    ok_l = ok_s = True
    for p in prompts[:3]:
        ids = torch.tensor(p)
        for g in (gf, gs):
            sa, sb = [], []
            za = forward(cfg, g, ids, stream_hook=lambda l, h: sa.append(h))
            zb = cm.forward_taps(cfg, g, ids, stream_hook=lambda l, h: sb.append(h))
            ok_l &= torch.equal(za, zb) and torch.equal(za, lvo.forward(cfg, g, ids))
            ok_s &= len(sa) == 37 and all(torch.equal(x, y) for x, y in zip(sa, sb))
    check(ok_l, "forward gives contrast_map.forward_taps' and layer_vs_output.forward's logits bit for bit, both sets")
    check(ok_s, "and the same 37 streams")

    # 2. Resume: the sealed run cut at any depth and resumed gives the monolithic logits bit for bit.
    ok = True
    for p in prompts[:3]:
        ids = torch.tensor(p)
        cap = {}
        z = forward(cfg, gs, ids, stream_hook=lambda l, h: cap.__setitem__(l, h))
        cos, sin = lvo.rope_tables(cfg, len(p), cpu, torch.float64)
        for d in sorted(set(PATCH_DEPTHS) | {ATTN_LAYERS[0], 0, 35}):
            ok &= torch.equal(run_from(cfg, gs, cap[d], d, cos, sin), z)
    check(ok, "run_from(h(d), d) reproduces the sealed forward bit for bit, depths 0, 4, 7, 8, 12, 16, 20, 35")

    # 3. The run, both passes, every check.
    run = PatchRun(cfg, sets, names, torch.float64, cpu, ans_ids, bands=bands)
    for p in prompts:
        run.pass1(p)
    run.set_basis()
    for qi, p in enumerate(prompts):
        run.pass2(qi, p, check=True)
    sp = [c for c in run.checks if c["kind"] == "self-patch"]
    hy = [c for c in run.checks if c["kind"] == "hybrid"]
    tb = [c for c in run.checks if c["kind"] == "three-bands"]
    ex = [c for c in run.checks if c["kind"] == "explicit-attention"]
    check(len(sp) == 24 and all(c["ok"] for c in sp),
          f"self-patch with the sealed stream, 4 forms x 6 prompts: {sum(c['ok'] for c in sp)} of {len(sp)} bit for bit")
    check(len(hy) == 30 and all(c["ok"] for c in hy),
          f"whole-stream patch at depths {PATCH_DEPTHS} = FP16 blocks 0..d-1 then sealed, every later stream and "
          f"the output, bit for bit: {sum(c['ok'] for c in hy)} of {len(hy)}")
    check(run.p1_mismatch == 0, "FP16 logits of pass 2 equal pass 1 bit for bit")
    d12 = torch.stack(run.out["d12"]["logits"])
    ctl = torch.stack(run.out["control"]["logits"])
    f16 = torch.stack(run.out["fp16"]["logits"])
    check(float((d12 - ctl).abs().max()) > 1e-3 and float((d12 - f16).abs().max()) > 1e-3,
          "the depth-12 hybrid differs from both the control and FP16 (the plant is not trivial)")
    check(all(c["value"] < 1e-9 for c in tb) and all(c["stream"] < 1e-12 for c in tb),
          f"three bands together = the whole stream at depth 12: worst |dlogit| "
          f"{max(c['value'] for c in tb):.1e}, worst |dstream| {max(c['stream'] for c in tb):.1e}")
    check(all(c["value"] < 1e-12 for c in ex),
          f"explicit answer row of attention = SDPA at layers 7-13, no patch ({max(c['value'] for c in ex):.1e})")
    ae = torch.stack(run.out["attn_explicit"]["logits"])
    af = torch.stack(run.out["attn_fp16_7-13"]["logits"])
    check(float((ae - ctl).abs().max()) < 1e-6 and float((af - ctl).abs().max()) > 1e-4,
          f"explicit path, no patch, = control to {float((ae - ctl).abs().max()):.1e}; FP16's probabilities move the "
          f"output by {float((af - ctl).abs().max()):.1e}")
    be = [c for c in run.checks if c["kind"] == "branch-exact"]
    check(len(be) == 6 and all(c["ok"] for c in be),
          f"FP16 h(0) and FP16's two branches at blocks 0-11 in the sealed run give FP16's h(12) and d12's logits, bit "
          f"for bit: {sum(c['ok'] for c in be)} of {len(be)}")
    bm = torch.stack(run.out["mlp_0-11"]["logits"])
    ba = torch.stack(run.out["attn_0-11"]["logits"])
    check(float((bm - ctl).abs().max()) > 1e-4 and float((ba - ctl).abs().max()) > 1e-4
          and float((bm - d12).abs().max()) > 1e-4 and float((bm - ba).abs().max()) > 1e-4,
          "each branch arm moves the output, and neither equals d12 nor the other")
    capx, capy = {}, {}
    fbx = {}
    forward(cfg, gf, torch.tensor(prompts[0]), branch=lambda b, kind, o: fbx.__setitem__((b, kind), o) or o)
    zx_ = forward(cfg, gs, torch.tensor(prompts[0]), stream_hook=lambda l, h: capx.__setitem__(l, h),
                  branch=branch_replacer(fbx, BRANCH_LAYERS, ("mlp",)))
    ok_br = torch.equal(zx_[-1, ans_ids].float(), bm[0])
    forward(cfg, gs, torch.tensor(prompts[0]), stream_hook=lambda l, h: capy.__setitem__(l, h))
    ok_br &= not torch.equal(capx[1], capy[1]) and not torch.equal(capx[12], capy[12])
    check(ok_br and branch_replacer(fbx, BRANCH_LAYERS, ("mlp",))(12, "mlp", capy[1]) is capy[1]
          and branch_replacer(fbx, BRANCH_LAYERS, ("mlp",))(3, "attn", capy[1]) is capy[1],
          "the MLP arm is the sealed run with FP16's MLP outputs at blocks 0-11 only; block 12 and the attention "
          "branch are its own")

    # 4. The band projection.
    g4 = torch.Generator().manual_seed(9)
    hs_ = torch.randn(1, 11, 16, generator=g4, dtype=torch.float64)
    hf_ = torch.randn(1, 11, 16, generator=g4, dtype=torch.float64)
    vq, _ = torch.linalg.qr(torch.randn(16, 16, generator=g4, dtype=torch.float64))
    bs = [vq[:, band_cols(lo, hi, 16)] for lo, hi in bands]
    outs = [patch_stream(hs_, hf_, None, b) for b in bs]
    tot = sum(o - hs_ for o in outs)
    check(float((tot - (hf_ - hs_)).abs().max()) < 1e-12, "the three band patches' changes sum to the whole difference")
    inside = outside = 0.0
    for b_i, (lo, hi) in enumerate(bands):
        cols = band_cols(lo, hi, 16)
        dv = (outs[b_i] - hs_) @ vq
        want = (hf_ - hs_) @ vq
        rest = torch.ones(16, dtype=torch.bool)
        rest[cols] = False
        inside = max(inside, float((dv[..., cols] - want[..., cols]).abs().max()))
        outside = max(outside, float(dv[..., rest].abs().max()))
    check(inside < 1e-12 and outside < 1e-12,
          f"a band patch moves its own directions to FP16's and leaves the others ({inside:.1e}, {outside:.1e})")
    check(float((patch_stream(hs_, hf_, None, torch.cat(bs, 1)) - hf_).abs().max()) < 1e-12,
          "patching the three bands as one basis gives FP16's stream")
    full = [band_cols(lo, hi, 2560) for lo, hi in PATCH_BANDS]
    check(full[0] == slice(0, 64) and full[1] == slice(64, 512) and full[2] == slice(512, 2560),
          "production bands are columns 0:64, 64:512, 512:2560, a partition of 2,560")
    m3 = torch.zeros(11, dtype=torch.bool)
    m3[[2, 5]] = True
    pm = patch_stream(hs_, hf_, m3, bs[2])
    check(torch.equal(pm[:, ~m3], hs_[:, ~m3]) and float((pm[:, m3] - outs[2][:, m3]).abs().max()) < 1e-12,
          "a band patch on chosen positions changes those rows only, as the every-position patch does there")
    h16 = hs_.to(torch.float16)
    check(torch.equal(patch_stream(h16, h16, None, bs[2]), h16) and torch.equal(patch_stream(h16, h16, m3, bs[2]), h16),
          "an f16 stream patched with itself through the float32 band path is unchanged bit for bit")

    # 5. Positions.
    nrm = torch.ones(10, dtype=torch.float64)
    nrm[[0, 3, 9]] = 50.0
    nrm[5] = 10.0
    mp = massive_positions(nrm, 9, 10.0)
    check(mp.nonzero().flatten().tolist() == [0, 3],
          "massive rule: above 10 x median selected, exactly 10 x median not, the answer position never")
    oka = True
    for qi, p in enumerate(prompts):
        want = [i for i, t in enumerate(p[:-1]) if t == PLANT]
        oka &= run.p1[qi]["positions"] == want and run.p1[qi]["token_ids"] == [PLANT] * len(want)
    check(oka, "on the tiny run the massive positions are exactly the planted tokens, the planted answer excluded: "
          + "; ".join(str(p["positions"]) for p in run.p1))
    capm = {}
    forward(cfg, gf, torch.tensor(prompts[1]), stream_hook=lambda l, h: capm.__setitem__(l, h))
    n8 = capm[8][0].float().norm(dim=-1)
    check(torch.equal(run.p1[1]["ratio"], (n8 / n8.median()).cpu()),
          "the massive rule reads FP16's norms at depth 8, h(8) after block 8")
    ma = position_mask("ans", 8, 7, torch.zeros(8, dtype=torch.bool), cpu)
    mn = position_mask("notans", 8, 7, torch.zeros(8, dtype=torch.bool), cpu)
    check(ma.tolist() == [False] * 7 + [True] and torch.equal(mn, ~ma) and position_mask("all", 8, 7, None, cpu) is None,
          "answer position alone and every position but it partition the prompt")
    p0 = prompts[0]
    ids0 = torch.tensor(p0)
    capf, caps = {}, {}
    forward(cfg, gf, ids0, stream_hook=lambda l, h: capf.__setitem__(l, h))
    zc = forward(cfg, gs, ids0, stream_hook=lambda l, h: caps.__setitem__(l, h))
    cos, sin = lvo.rope_tables(cfg, len(p0), cpu, torch.float64)
    capa = {}
    m0 = position_mask("ans", len(p0), len(p0) - 1, None, cpu)
    za = run_from(cfg, gs, patch_stream(caps[12], capf[12], m0), 12, cos, sin,
                  stream_hook=lambda l, h: capa.__setitem__(l, h))
    early = all(torch.equal(capa[l][:, :-1], caps[l][:, :-1]) for l in range(13, 37))
    check(early and not torch.equal(za[-1], zc[-1]),
          "a patch at the answer position alone leaves every earlier position unchanged at every later depth, and "
          "moves the answer")

    # 6. The attention rewriter.
    g6 = torch.Generator().manual_seed(13)
    q6, k6, v6 = (torch.randn(1, 4, 9, 8, generator=g6, dtype=torch.float64) for _ in range(3))
    a6 = F.scaled_dot_product_attention(q6, k6, v6, is_causal=True, scale=8 ** -0.5)
    pr = torch.softmax(torch.randn(4, 9, generator=g6, dtype=torch.float64), -1)
    a7 = attn_rewriter({3}, 8, 8 ** -0.5, {3: pr})(3, q6, k6, v6, a6)
    check(float((a7[0, :, 8] - torch.einsum("hl,hld->hd", pr, v6[0])).abs().max()) < 1e-12
          and torch.equal(a7[0, :, :8], a6[0, :, :8]),
          "the rewritten answer row is P v exactly and every other row is SDPA's, bit for bit")
    check(attn_rewriter({3}, 8, 8 ** -0.5, {3: pr})(2, q6, k6, v6, a6) is a6, "a block outside the set is untouched")
    own = attn_rewriter({3}, 8, 8 ** -0.5, None)(3, q6, k6, v6, a6)
    check(float((own - a6).abs().max()) < 1e-12, "the model's own probabilities through the explicit path = SDPA")

    # 7. The lens read: the kept depth-25 row is h(25) at the answer position, and the lens is contrast_map's.
    pos0 = len(p0) - 1
    caph = {}
    forward_hybrid(cfg, sets, ids0, 12, stream_hook=lambda l, h: caph.__setitem__(l, h))
    kept = {n: run.out[n].get(25) for n in ("fp16", "control", "d12")}
    ok7 = (all(v is not None for v in kept.values()) and torch.equal(kept["fp16"][0], capf[25][0, pos0])
           and torch.equal(kept["control"][0], caps[25][0, pos0]) and torch.equal(kept["d12"][0], caph[25][0, pos0])
           and torch.equal(run.out["d12"][12][0], capf[12][0, pos0]))
    check(ok7, "rows kept at depth 25 (and 12) are the answer rows of h(25) (and h(12)) of FP16, control, hybrid")
    zl = cm.lens(capf[36][:, pos0], fp16["model.norm.weight"], fp16["model.embed_tokens.weight"][ans_ids], cfg.eps)
    check(float((zl[0] - f16[0].double()).abs().max()) < 1e-5,
          "the lens at depth 36 on FP16 is FP16's own answer logits (contrast_map.lens, the model's norm in float32)")

    # 8. Masked errors against contrast_map.Gain.
    mg = MaskedGain(fp16, sealed, names, torch.float64)
    mg.start()
    ga = cm.Gain(fp16, sealed, names, torch.float64)
    gl = cm.Gain(fp16, sealed, names, torch.float64)
    gn = cm.Gain(fp16, sealed, names, torch.float64)
    msv = run.p1[3]["massive"]
    ids3 = torch.tensor(prompts[3])
    pos3 = len(prompts[3]) - 1

    def both(nm, x):
        mg(nm, x)
        ga(nm, x)
        gl(nm, x[:, pos3:pos3 + 1])
        gn(nm, x[:, ~msv])

    forward(cfg, gf, ids3, proj_hook=both)
    t8 = mg.finish(mask_stack(len(prompts[3]), pos3, msv, cpu))
    worst = 0.0
    for i, nm in enumerate(names):
        for m, gg in enumerate((ga, gl, gn)):
            ref_ = torch.tensor(gg.totals(nm), dtype=torch.float64)
            worst = max(worst, float((t8[i, m] - ref_).abs().max() / ref_[0]))
    check(worst < 1e-12 and len(run.p1[3]["positions"]) == 3,
          f"masked sums = contrast_map.Gain on every position, the answer row alone, and all but the 3 massive ({worst:.1e})")

    # 9. Statistics.
    ca = torch.tensor([1.0] * 5 + [0.0] * 2 + [1.0] * 6 + [0.0] * 7)
    cb = torch.tensor([0.0] * 5 + [1.0] * 2 + [1.0] * 6 + [0.0] * 7)
    pr9 = paired(ca, cb)
    hw = 1.96 * math.sqrt(7 - 9 / 20) / 20
    check(pr9["b"] == 5 and pr9["c"] == 2 and abs(pr9["diff"] - 0.15) < 1e-12 and abs(pr9["hi"] - 0.15 - hw) < 1e-12
          and abs(pr9["p"] - 58 / 128) < 1e-12, "paired: b 5, c 2, +0.15, half width 1.96 sqrt(b+c-(b-c)^2/n)/n, McNemar p 0.453")
    cnt9 = boot_counts(40, 2000, 1)
    check(bool((cnt9.sum(1) == 40).all()), "bootstrap counts resample n questions per draw")
    cols9 = torch.stack((torch.full((40,), 0.3, dtype=torch.float64), torch.ones(40, dtype=torch.float64)), 1)
    b9 = boot(cols9, ratio_of_gains, cnt9)
    check(abs(b9["point"] - 0.3) < 1e-12 and abs(b9["lo"] - 0.3) < 1e-12 and abs(b9["hi"] - 0.3) < 1e-12,
          "bootstrap of a constant ratio is that ratio, point and interval")
    g9 = torch.Generator().manual_seed(2)
    x9 = torch.randn(300, 16, generator=g9, dtype=torch.float64)
    y9 = 0.5 * x9 + 0.3 * torch.randn(300, 16, generator=g9, dtype=torch.float64)
    a9, e9 = band_terms(x9, y9, vq[:, 6:16])
    bt9 = cm.band_table(vq, None, x9.T @ x9, y9.T @ x9, y9.T @ y9, bands=((1, 6), (7, None)))
    check(abs(float(a9.sum() / e9.sum()) - bt9["rows"][1]["slope_w"]) < 1e-12,
          "per-question band terms give contrast_map.band_table's energy-weighted slope")
    # 9b. The shape statistic and the instrument check.
    fz = [22.5, 24.1, 23.0, 21.7]
    c1, z1 = shape_cos([v * 0.4 + 3.0 for v in fz], fz)
    c2, _ = shape_cos([24.0, 22.4, 23.5, 24.8], fz)
    c0, z0 = shape_cos([5.0, 5.0, 5.0, 5.0], fz)
    check(abs(c1 - 1.0) < 1e-12 and not z1 and abs(c2 + 1.0) < 1e-12 and c0 == 0.0 and z0,
          "shape cosine: scale and offset free (1), mirrored shape -1, a flat vector 0 and counted")
    check(abs(kl4(fz, fz)) < 1e-15 and kl4(fz, [0.0] * 4) > 0
          and abs(logp_of([0.0, 0.0, 0.0, 0.0], 2) - math.log(0.25)) < 1e-15,
          "KL over four options is 0 on itself and positive otherwise; log p of a flat vector is log 1/4")
    check(ranks([3.0, 1.0, 3.0, 2.0]) == [3.5, 1.0, 3.5, 2.0] and abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1) < 1e-12
          and abs(spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1) < 1e-12, "ranks with ties, Spearman +1 and -1")
    gi = torch.Generator().manual_seed(17)
    nq_ = 3000
    fl = torch.randn(nq_, 4, generator=gi, dtype=torch.float64) * 1.5
    off = 22.0 + 2.0 * torch.rand(nq_, 1, generator=gi, dtype=torch.float64)
    mk = lambda z: {i: {"logits": z[i].tolist(), "pick": cm.pick_of(z[i].tolist()),  # noqa: E731
                        "answer": cm.pick_of(fl[i].tolist())} for i in range(nq_)}
    noise = lambda: torch.randn(nq_, 4, generator=gi, dtype=torch.float64) * 1.5  # noqa: E731
    alphas = (0.0, 0.2, 0.4, 0.6, 0.8, 0.95)
    plant = {f"a{a_}": mk(a_ * fl + (1 - a_) * noise() + off) for a_ in alphas[1:]}
    ti = instrument_table(mk(fl + off), mk(noise() + off), plant)
    cs = [ti["rows"][f"a{a_}"]["cos_share"] for a_ in alphas[1:]]
    check(all(x < y for x, y in zip(cs, cs[1:])) and cs[0] > 0 and cs[-1] < 1,
          "instrument: the cosine share of alpha * FP16 + noise grows with alpha: " + ", ".join(f"{x:.3f}" for x in cs))
    check(ti["rho"] >= 0.9 and ti["primary"] == "cosine",
          f"instrument: Spearman with the accuracy share {ti['rho']:.3f}, the rule picks the cosine share")
    ts = instrument_table(mk(fl + off), mk(noise() + off), {"self": mk(fl + off), "ctl": mk(noise() + off)})
    check(abs(ts["rows"]["self"]["cos_share"] - 1) < 1e-12 and abs(ts["rows"]["self"]["acc_share"] - 1) < 1e-12,
          "instrument: FP16 against itself reads both shares 1")
    check(abs(float(slope_recovery(torch.tensor([[0.8, 0.5, 1.0]], dtype=torch.float64))[0]) - 0.6) < 1e-12,
          "slope recovery (0.8 - 0.5) / (1 - 0.5) = 0.6")
    check(verdict_at_least(0.6, 0.9, 0.5) == "KILLED" and verdict_at_least(0.1, 0.4, 0.5) == "STANDS"
          and verdict_at_least(0.3, 0.7, 0.5) == "NOT DECIDED" and verdict_under(0.0, 0.1, 0.2) == "KILLED"
          and verdict_under(0.3, 0.5, 0.2) == "STANDS" and verdict_h6b([(0.9, 1.2), (0.8, 1.4)]) == "KILLED"
          and verdict_h6b([(0.9, 1.2), (1.6, 3.0)]) == "STANDS" and verdict_h6b([(0.9, 1.7)]) == "NOT DECIDED"
          and verdict_h6c((0.0, 0.05), (0.9, 1.5)) == "KILLED" and verdict_h6c((0.0, 0.05), (2.1, 3.0)) == "STANDS"
          and verdict_h6c((0.2, 0.4), (0.9, 1.5)) == "STANDS" and verdict_h6c((0.0, 0.2), (0.9, 1.5)) == "NOT DECIDED",
          "the three-way verdicts of H6-A to H6-D")
    check(guard_d12("KILLED", dict(lo=0.1)) == "KILLED" and guard_d12("KILLED", dict(lo=0.0)).startswith("NOT DECIDED")
          and guard_d12("STANDS", dict(lo=-0.2)).startswith("NOT DECIDED")
          and guard_d12("KILLED", dict(lo=math.nan)).startswith("NOT DECIDED"),
          "a ratio over s(d12) is NOT DECIDED when s(d12)'s interval reaches 0")

    # 10. End to end: analyse writes every file; gates green on dumps equal to the script's logits.
    keys = [("tiny", i) for i in range(len(prompts))]
    answers = [i % 4 for i in range(len(prompts))]
    df = {k: {"pick": cm.pick_of(f16[i].tolist()), "answer": answers[i], "logits": f16[i].tolist()}
          for i, k in enumerate(keys)}
    ds = {k: {"pick": cm.pick_of(ctl[i].tolist()), "answer": answers[i], "logits": ctl[i].tolist()}
          for i, k in enumerate(keys)}
    with tempfile.TemporaryDirectory() as dd:
        res = analyse(Path(dd), run, keys, answers, prompts, df, ds)
        files = {x.name for x in Path(dd).iterdir()}
        npk = len((Path(dd) / "picks.tsv").read_text().splitlines()) - 1
        nk = len((Path(dd) / "kills.tsv").read_text().splitlines()) - 1
        back = cm.read_raw(Path(dd) / "gains_fp16x.f64")
    want = {"report.txt", "arms.tsv", "picks.tsv", "logits.tsv", "kills.tsv", "types_pos.tsv", "matrices_pos.tsv",
            "massive.tsv", "checks.tsv", "gains_fp16x.f64", "gains_sealedx.f64", "shape.tsv"}
    check(want <= files and npk == len(prompts) and nk == 6, f"analyse writes its {len(want)} files, {npk} pick rows, {nk} kill rows")
    check(res["green"], "every gate green on the tiny run with self-consistent dumps")
    check(torch.equal(back, torch.stack(run.gain_f)), "the raw error sums read back bit for bit")
    check(set(res["arms"]) == set(run.arm_names) == set(res["shape"])
          and len(run.arm_names) == 2 + len(ARMS) + len(ATTN_ARMS) + len(BRANCH_ARMS),
          f"{len(run.arm_names)} arms read: fp16, control, {len(ARMS)} patches, {len(ATTN_ARMS)} attention arms, "
          f"{len(BRANCH_ARMS)} branch arms")
    sh = res["shape"]
    cvec = lambda n: [shape_cos(run.out[n]["logits"][i].tolist(), run.out["fp16"]["logits"][i].tolist())[0]  # noqa: E731
                      for i in range(len(prompts))]
    check(abs(sh["fp16"]["C"] - 1) < 1e-12 and abs(sh["fp16"]["s"]["point"] - 1) < 1e-12
          and abs(sh["control"]["s"]["point"]) < 1e-12
          and abs(sh["d12"]["s"]["point"] - shape_share(cvec("d12"), cvec("control"))) < 1e-12,
          "shape share: FP16 reads 1, the control 0, d12 the formula on its own cosines")

    print(f"selftest: {'GREEN' if not fails else 'RED'}, {n_checks[0]} checks, {len(fails)} failure(s)")
    for f_ in fails:
        print(f"  failed: {f_}")
    return 0 if not fails else 1


# ---------------------------------------------------------------------------
# The measurement.
# ---------------------------------------------------------------------------
def run(args) -> int:
    lvo.check_pins()
    if not torch.cuda.is_available():
        raise SystemExit("refused: --run is for the card; the 4B is never loaded on the Mac")
    device = torch.device("cuda:0")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    timeline = []

    def stamp(label):
        timeline.append((label, time.time() - t_start))
        print(f"== {time.time() - t_start:7.1f} s  {label}", flush=True)

    ref = read_ref(Path(args.ref_dir))
    dump_f = cm.read_dump(Path(args.dump_f16), cm.DUMP_F16[1])
    dump_s = cm.read_dump(Path(args.dump_sealed), cm.DUMP_SEALED[1])
    qs, ans_ids = cm.fetch_prompts(args.cache)
    bad = [(q["subject"], q["index"]) for q in qs for dmp in (dump_f, dump_s)
           if dmp[(q["subject"], q["index"])]["qhash"] != q["qhash"]]
    if bad:
        raise SystemExit(f"refused: {len(bad)} prompts differ from the dumps' token ids, first {bad[:3]}")
    order = cm.round_robin(qs)[:N_Q]
    same_q = [(q["subject"], q["index"], q["qhash"]) for q in order] == ref["questions"]
    print(f"questions: the first {N_Q} of the round robin equal contrast-map's questions.tsv row for row: {same_q}")
    if not same_q:
        raise SystemExit("refused: these are not contrast-map's 570 questions")
    stamp("inputs checked")
    cfg, sets, extra = lvo.load_sets(Path(args.export), Path(args.stage), args.cache, device)
    h0 = {k: cm.sha_set(v) for k, v in sets.items()}
    stamp(f"loaded, bases fp16 {h0['fp16'][:16]} sealed {h0['sealed'][:16]}")
    names = lvo.all_projections(cfg.layers)
    pr = PatchRun(cfg, sets, names, torch.float32, device, ans_ids)
    keys = [(q["subject"], q["index"]) for q in order]
    answers = [q["answer"] for q in order]
    qids = [q["ids"] for q in order]

    t1 = time.time()
    for i, ids in enumerate(qids):
        pr.pass1(ids)
        if i + 1 == 57:
            torch.cuda.synchronize()
            proj = (time.time() - t1) * N_Q / 57
            print(f"pass 1: 57 prompts in {time.time() - t1:.1f} s, projected {proj:.0f} s (cap {PASS1_CAP:.0f})", flush=True)
            if proj > PASS1_CAP:
                print("refused: pass 1 would exceed its cap; nothing is read", flush=True)
                return 5
    torch.cuda.synchronize()
    p1_s = time.time() - t1
    stamp(f"pass 1: {N_Q} prompts, {pr.tokens} tokens, {p1_s:.0f} s")
    pr.set_basis()
    stamp("band basis, depth 12")

    t2 = time.time()
    marks = {}
    for i, ids in enumerate(qids):
        pr.pass2(i, ids, check=i < CHECK_Q)
        if (i + 1) % 57 == 0:
            torch.cuda.synchronize()
            el = time.time() - t2
            marks[i + 1] = el
            print(f"  pass 2: {i + 1} prompts, {el:.0f} s", flush=True)
            if i + 1 == 2 * 57:
                # The first 57 prompts carry the exactness controls; the next 57 give the rate of the rest.
                proj = p1_s + el + (el - marks[57]) / 57 * (N_Q - 114)
                print(f"pass 2: projected pass 1 + pass 2 {proj:.0f} s (cap {LOOP_CAP:.0f})", flush=True)
                if proj > LOOP_CAP:
                    print("refused: the loop would exceed its cap; nothing is read", flush=True)
                    return 5
    torch.cuda.synchronize()
    stamp(f"pass 2: {N_Q} prompts, {time.time() - t2:.0f} s, {len(pr.checks)} checks on the first {CHECK_Q}")
    h1 = {k: cm.sha_set(v) for k, v in sets.items()}
    same = h0 == h1
    stamp(f"bases unchanged by sha256: {same}")
    head = [f"inputs: checkpoint {lvo.CKPT_REPO}@{lvo.CKPT_REV}; export sha256 {lvo.EXPORT_SHA}; "
            f"{cm.MMLU_REPO}@{cm.MMLU_REV}; run fingerprint {cm.RUN_FP:016x}; answer ids {ans_ids}",
            f"questions: {N_Q}, contrast-map's, {pr.tokens} tokens; pass 1 {p1_s:.0f} s, pass 2 {time.time() - t2:.0f} s",
            f"bases unchanged by sha256: {same}; non-projection tensors identical in W and W_hat: "
            f"{len(extra['same_other'])} of {len(extra['others'])}", ""]
    res = analyse(out, pr, keys, answers, qids, dump_f, dump_s, ref=ref, extra_lines=head, primary=args.primary)
    stamp("analysis written")
    (out / "timeline.txt").write_text("".join(f"{s:.1f}\t{lab}\n" for lab, s in timeline))
    (out / "inputs.txt").write_text(
        f"checkpoint {lvo.CKPT_REPO}@{lvo.CKPT_REV}\nexport {args.export} sha256 {lvo.EXPORT_SHA}\n"
        f"mmlu {cm.MMLU_REPO}@{cm.MMLU_REV}\ndumps {cm.DUMP_F16[1]} {cm.DUMP_SEALED[1]}\nref {json.dumps(REF_FILES)}\n"
        f"questions {N_Q}\nprimary {args.primary}\nbases sha256 before {h0}\nbases sha256 after  {h1}\ngates {json.dumps(res['gates'])}\n"
        f"total {time.time() - t_start:.0f} s\n")
    if not same:
        return 3
    g = res["gates"]
    if not (g["self-patch bit for bit"] and g["hybrid bit for bit"] and g["FP16 pass 2 = pass 1"]
            and g["branches bit for bit"]):
        return 3
    return 0 if res["green"] else 4


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--selftest", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--instrument", action="store_true",
                      help="the $0 instrument check on the dumps; run after the stamp, before the job")
    ap.add_argument("--export", help="the sealed export's model.safetensors (bucket mount)")
    ap.add_argument("--stage", default="/tmp/pm-export", help="local copy of the export")
    ap.add_argument("--out", help="output folder")
    ap.add_argument("--dump-f16")
    ap.add_argument("--dump-sealed")
    ap.add_argument("--ref-dir", help=f"contrast-map-4b's raw files ({REF_DIR})")
    ap.add_argument("--cache", default=os.environ.get("HF_HUB_CACHE"))
    ap.add_argument("--dumps-dir", help=f"--instrument: default {INSTRUMENT_DIR}")
    ap.add_argument("--primary", choices=("cosine", "accuracy"), default="cosine",
                    help="--run: the share the kill rules read, as the instrument check decided")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.instrument:
        return instrument(a)
    if not (a.export and a.out and a.dump_f16 and a.dump_sealed and a.ref_dir):
        ap.error("--run needs --export, --out, --dump-f16, --dump-sealed and --ref-dir")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
