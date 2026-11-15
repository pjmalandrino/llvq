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
"""The sensitivity map: is FP16's sensitivity to an error in the stream at depth 12 uneven over directions, in a basis
estimated in advance on other text, and does a band of that basis give back more than its share of the sealed 4B's
deviation energy?

Prereg (draft): proofs/preregistration-sensitivity-map-4b-2026-11-15-DRAFT.md. Frame: docs/mesures/
recherche-qualite3-r6-issue-2026-11-15.txt and the attack's last section (recherche-qualite3-r6-attaque-code).

It extends patch-map-4b (ops/patch_map.py, imported, never edited): same two weight sets on the card, same 570 scored
questions, same rule D, same band cut by rank ([1, 64], [65, 512], [513, 2560]), same patch at depth 12 at every
position, same primary statistic (the cosine share of the four centred answer logits, the instrument check of
patch-map having fixed it on these dumps), same bootstrap. Three bases of the 2,560-dimension stream at depth 12:
  variance     the eigenbasis of FP16's uncentred second moment of h(12) over the scored prompts (patch-map's own).
  fisher_text  the eigenbasis of sum g g^T, g the gradient with respect to FP16's h(12) of the log-likelihood of
               labels SAMPLED from FP16's own next-token distribution, every position of N_TEXT windows of WikiText-2
               (the token stream layer_vs_output.fetch_tokens pins). Sampled labels make it the true Fisher, whose
               cross-target terms vanish in expectation; observed labels would give the empirical Fisher.
  fisher_mmlu  the eigenbasis of sum_k p_k s_k s_k^T, s_k the gradient with respect to FP16's h(12), every position,
               of log p_k, the four-option log-probability at the answer position, p_k FP16's own: the exact Fisher of
               the four-way answer, on 570 questions that FOLLOW the scored ones in the round robin, rows
               whose (subject, index) or token hash repeats a scored or an earlier one skipped (MMLU repeats a few).
Both Fisher bases come from FP16 alone and never see a scored question (asserted on subject, index and token hash).

The gradient pass: FP16's h(12) in f16 under no_grad, bit for bit the stream the patch reads, cast to float32; blocks
12..35 and the head run in float32 (f16 weights cast per call, matmul precision "highest", no TF32), each block under
non-reentrant activation checkpointing, attention by the math SDPA backend. The gradient is taken with respect to
h(12) only; no weight requires a gradient.

Passes:
  pass 1  FP16 alone on the scored 570: C_ff at depth 12 (the variance basis) and its answer logits.
  text    N_TEXT WikiText-2 windows: the fisher_text accumulator, split by window parity.
  fisher  the 570 following questions: the fisher_mmlu accumulator, split by question parity; FP16's f16 picks beside
          the float32 path's.
  pass 2  FP16 again (equal to pass 1 bit for bit, after the gradient passes), the sealed control, the deviation
          h_s(12) - h_f(12) projected on every band of every basis, then every arm resumed from the control's h(12).
          The first CHECK_Q prompts also run the exactness controls.

Modes:
  --selftest   tiny random models on CPU, float64 (one float16 check); no download.
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
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contrast_map as cm  # noqa: E402
import layer_vs_output as lvo  # noqa: E402
import patch_map as pm  # noqa: E402

# ---------------------------------------------------------------------------
# The measurement, fixed before the data.
# ---------------------------------------------------------------------------
N_Q = 570
DEPTH = pm.BAND_DEPTH  # 12
BANDS = pm.PATCH_BANDS  # ((1, 64), (65, 512), (513, 2560))
TOP, BOT = 0, 2
N_TEXT = 256  # WikiText-2 windows, the first ones of the pinned test stream
TEXT_CTX = 512
FISHER_START = N_Q  # fisher_mmlu: the round robin walked from this row, skipping repeats, until N_Q rows are held
LABEL_SEED = 20261116  # sampled next tokens of the text pass
CHECK_Q = 57
CHECK_GRAD = 8  # windows and questions whose float32 logits are recomputed without autograd
BASES = ("variance", "fisher_text", "fisher_mmlu")
FISHERS = ("fisher_text", "fisher_mmlu")
SHORT = {"variance": "var", "fisher_text": "ft", "fisher_mmlu": "fm"}
# Deviation-energy conventions: every coordinate; direction 1 of the variance basis removed at position 0 only;
# direction 1 removed at every position (R6's "without direction 1"). The rule reads PRIMARY_CONV.
CONVS = ("all", "pos0", "dir1")
PRIMARY_CONV = "dir1"
OVERLAP_K = 64
UNEVEN_MIN = 3.0
FLAT_LO, FLAT_HI = 0.5, 2.0
VAR_CTRL = (0.5, 2.0)  # the variance basis's [1, 64] ratio, point, must fall here; computed expectation 1.03
# Its [1, 512] ratio, point: patch-map has no joint [1, 512] arm, so the expectation is a bracket, 0.35 (s(d12) -
# s(bottom), the complement bound) to 1.14 (s_1 + s_2, the additive bound), *computed* on patch-map's logits.
VAR_CTRL_W = (0.25, 2.0)
WIDE = (0, 1)  # bands [1, 64] and [65, 512] patched together: [1, 512]
FLAT_E_MIN = 0.15  # FLAT cannot fire if fisher_text's [1, 512] holds less of the deviation energy (dir1)
PM_SHARE_TOL = 0.01
PM_PICKS_MIN = 0.99
ORTHO_TOL = 1e-9
ENERGY_TOL = 1e-9
F32_PICKS_MIN = 0.98
PICK_AGREE_MIN = pm.PICK_AGREE_MIN
EIG_FLOOR = 1e-12
PRIMARY = "cosine"
CAP = dict(pass1=300.0, text=300.0, fisher=900.0)  # projected seconds of each loop, read early
LOOP_CAP = 1800.0  # projected seconds of every loop together, read after 114 prompts of pass 2

# patch-map-4b's raw files, committed in docs/mesures/patch-map-4b-2026-11-15-brut/.
PM_REF_DIR = "docs/mesures/patch-map-4b-2026-11-15-brut"
PM_REF_FILES = {
    "logits.tsv": "a3e6b4269b4162bee9d91e86f6c70244835ace924b0777d63a13e39bc5f8b3c8",
    "shape.tsv": "0a6b91c49a998236cc1e390bfb0d4e203aabff2f79755c3522caa8455a89638d",
}
PM_SAME = {"d12": "d12", "var_b1-64": "d12_b1-64", "var_b65-512": "d12_b65-512", "var_b513-2560": "d12_b513-2560"}


def arm_table(bands=BANDS):
    """name, basis (None = the whole stream), band indices patched together."""
    arms = [("d12", None, None)]
    for b in BASES:
        for i, (lo, hi) in enumerate(bands):
            arms.append((f"{SHORT[b]}_b{lo}-{hi}", b, (i,)))
    lo, hi = bands[TOP]
    for b in FISHERS:
        arms.append((f"{SHORT[b]}_not{lo}-{hi}", b, tuple(i for i in range(len(bands)) if i != TOP)))
    for b in BASES:
        arms.append((wide_name(b, bands), b, WIDE))
    return tuple(arms)


def wide_name(basis: str, bands=BANDS) -> str:
    return f"{SHORT[basis]}_b{bands[WIDE[0]][0]}-{bands[WIDE[-1]][1]}"


ARMS = arm_table()


# ---------------------------------------------------------------------------
# The forward pieces. Same operations as patch_map.forward (checked bit for bit).
# ---------------------------------------------------------------------------
def to_depth(cfg, get, ids, depth: int):
    """h(depth) of patch_map.forward, and the rope tables of the weights' dtype."""
    h = get("model.embed_tokens.weight")[ids].unsqueeze(0)
    cos, sin = lvo.rope_tables(cfg, ids.shape[0], ids.device, h.dtype)
    for b in range(depth):
        h = pm.block(cfg, get, b, h, cos, sin)
    return h


def cast(get, dtype):
    return lambda name: get(name).to(dtype)


def tail(cfg, get, h, start: int, cos, sin, ckpt: bool):
    for b in range(start, cfg.layers):
        if ckpt:
            h = checkpoint(pm.block, cfg, get, b, h, cos, sin, use_reentrant=False)
        else:
            h = pm.block(cfg, get, b, h, cos, sin)
    return h


def text_logits(cfg, get, x, start: int, ckpt: bool):
    """Logits (L, V) of blocks start..35 and the head on x = h(start), in x's dtype."""
    cos, sin = lvo.rope_tables(cfg, x.shape[1], x.device, x.dtype)
    return pm.head(cfg, get, tail(cfg, get, x, start, cos, sin, ckpt))


def answer_logits(cfg, get, x, start: int, pos: int, ans_ids, ckpt: bool):
    """The four answer logits at `pos` of blocks start..35 on x = h(start)."""
    cos, sin = lvo.rope_tables(cfg, x.shape[1], x.device, x.dtype)
    h = tail(cfg, get, x, start, cos, sin, ckpt)
    hn = lvo.rms_norm(h[:, pos], get("model.norm.weight"), cfg.eps)
    w = get("model.embed_tokens.weight") if cfg.tied else get("lm_head.weight")
    return F.linear(hn, w[ans_ids])[0]


def text_grad(cfg, get_acc, h, start: int, gen=None, labels=None, ckpt: bool = True):
    """Gradient (L, d) with respect to x = h(start) of sum_t log p(y_t), y_t sampled from the model's own softmax
    (gen) unless labels are given. Returns (g, logits, labels)."""
    x = h.detach().to(get_acc("model.norm.weight").dtype).requires_grad_(True)
    with sdpa_kernel(SDPBackend.MATH):
        z = text_logits(cfg, get_acc, x, start, ckpt)
        lp = F.log_softmax(z, -1)
        if labels is None:
            with torch.no_grad():
                labels = torch.multinomial(lp.detach().exp(), 1, generator=gen).squeeze(1)
        loss = lp.gather(1, labels[:, None]).sum()
        (g,) = torch.autograd.grad(loss, x)
    return g[0], z.detach(), labels


def answer_grads(cfg, get_acc, h, start: int, pos: int, ans_ids, ckpt: bool = True):
    """Scores s_k (4, L, d): the gradients with respect to x = h(start) of log p_k over the four options, and p."""
    x = h.detach().to(get_acc("model.norm.weight").dtype).requires_grad_(True)
    with sdpa_kernel(SDPBackend.MATH):
        z4 = answer_logits(cfg, get_acc, x, start, pos, ans_ids, ckpt)
        lp4 = F.log_softmax(z4, -1)
        s = [torch.autograd.grad(lp4[k], x, retain_graph=k < 3)[0][0] for k in range(4)]
    return torch.stack(s), lp4.detach().exp(), z4.detach()


class FisherAcc:
    """sum of G^T G in float64, split in two halves; n the gradient vectors fed."""

    def __init__(self, d: int, device):
        self.half = [torch.zeros((d, d), dtype=torch.float64, device=device) for _ in range(2)]
        self.n = [0, 0]

    def add(self, g, half: int):
        gd = g.reshape(-1, g.shape[-1]).double()
        self.half[half] += gd.T @ gd
        self.n[half] += gd.shape[0]

    def total(self):
        return self.half[0] + self.half[1]


def select_fisher(rows, start: int, n: int, scored):
    """rows: round-robin rows, each with "subject", "index", "qhash". Walks from `start`, skipping every row whose
    (subject, index) or token hash is among the scored ones or those already taken, until n rows are held. Returns
    (taken, skipped), skipped a list of (row number, subject, index, reason)."""
    seen_k = {(s, i) for s, i, _ in scored}
    seen_h = {h for _, _, h in scored}
    taken, skipped = [], []
    for r in range(start, len(rows)):
        if len(taken) == n:
            break
        q = rows[r]
        k = (q["subject"], q["index"])
        if k in seen_k or q["qhash"] in seen_h:
            skipped.append((r, q["subject"], q["index"], "question" if k in seen_k else "token hash"))
            continue
        taken.append(q)
        seen_k.add(k)
        seen_h.add(q["qhash"])
    if len(taken) != n:
        raise SystemExit(f"refused: only {len(taken)} Fisher rows after row {start}, {n} wanted")
    return taken, skipped


def disjoint_or_refuse(scored, other) -> tuple[int, int]:
    """scored, other: lists of (subject, index, qhash). Refuses any shared question or token hash."""
    a = {(s, i) for s, i, _ in scored}
    b = {(s, i) for s, i, _ in other}
    ha = {h for _, _, h in scored}
    hb = {h for _, _, h in other}
    if a & b or ha & hb:
        raise SystemExit(f"refused: the Fisher questions share {len(a & b)} questions and {len(ha & hb)} token "
                         f"hashes with the scored ones")
    return len(a), len(b)


def ortho_err(v) -> float:
    vd = v.double()
    return float((vd.T @ vd - torch.eye(vd.shape[1], dtype=torch.float64)).abs().max())


def overlap(va, vb, k: int) -> float:
    """||Va_k^T Vb_k||_F^2 / k for the first k columns: 1 for one subspace, k/d expected at random."""
    return float(((va[:, :k].double().T @ vb[:, :k].double()) ** 2).sum()) / k


def spectrum(lam, bands) -> dict:
    """Trace share of each band (eigenvalues clamped at 0), arithmetic over geometric mean of the eigenvalues above
    EIG_FLOOR times the largest (the transform-coding gain), and how many fell under the floor."""
    lc = lam.double().clamp_min(0.0)
    d = lam.shape[0]
    tr = float(lc.sum())
    shares = [float(lc[pm.band_cols(lo, hi, d)].sum()) / tr for lo, hi in bands]
    keep = lc > EIG_FLOOR * float(lc.max())
    lk = lc[keep]
    am, gm = float(lk.mean()), math.exp(float(torch.log(lk).mean()))
    return dict(shares=shares, am_gm=am / gm, gain_db=10 * math.log10(am / gm), floored=int((~keep).sum()), tr=tr,
                top=[float(x) for x in lc[:8]])


def give_back_ratio(s):
    """s columns: [c_top - c_ctl, c_bot - c_ctl, e_top, e_bot] summed. (s_top / E_top) / (s_bot / E_bot): the
    (1 - c_ctl) of the shares and the total of the energy shares cancel."""
    return pm.safe_div(s[:, 0] * s[:, 3], s[:, 1] * s[:, 2])


def cross_term(s):
    """s columns: [c_1 - c_ctl, c_2 - c_ctl, c_3 - c_ctl, c_d12 - c_ctl, 1 - c_ctl] summed. s_1 + s_2 + s_3 - s_d12."""
    return pm.safe_div(s[:, 0] + s[:, 1] + s[:, 2] - s[:, 3], s[:, 4])


def verdict_ratio(lo: float, hi: float) -> str:
    """UNEVEN if the whole interval lies above UNEVEN_MIN; FLAT if it lies inside [FLAT_LO, FLAT_HI]."""
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return "NOT DECIDED"
    if lo > UNEVEN_MIN:
        return "UNEVEN"
    if lo >= FLAT_LO and hi <= FLAT_HI:
        return "FLAT"
    return "NOT DECIDED"


def verdict_rule(r64: dict, r512: dict, e512: float) -> str:
    """The rule on fisher_text. UNEVEN if either ratio's whole interval lies above UNEVEN_MIN ([1, 64] or [1, 512]
    over [513, 2560]); FLAT if the [1, 512] ratio's interval lies inside [FLAT_LO, FLAT_HI], the [1, 64] ratio's does
    not lie above UNEVEN_MIN, and [1, 512] holds at least FLAT_E_MIN of the deviation energy; NOT DECIDED otherwise."""
    v64, v512 = verdict_ratio(r64["lo"], r64["hi"]), verdict_ratio(r512["lo"], r512["hi"])
    if v64 == "UNEVEN" or v512 == "UNEVEN":
        return "UNEVEN"
    if v512 == "FLAT" and math.isfinite(e512) and e512 >= FLAT_E_MIN:
        return "FLAT"
    return "NOT DECIDED"


def var_control_ok(point: float, lim=VAR_CTRL) -> bool:
    return math.isfinite(point) and lim[0] <= point <= lim[1]


# ---------------------------------------------------------------------------
# The run.
# ---------------------------------------------------------------------------
class SensRun:
    def __init__(self, cfg, sets, device, ans_ids, bands=BANDS, acc=torch.float32, ckpt=True, depth=DEPTH):
        self.cfg, self.sets, self.device, self.ans_ids = cfg, sets, device, ans_ids
        self.bands, self.acc, self.ckpt, self.depth = bands, acc, ckpt, depth
        self.arms = arm_table(bands)
        self.gf, self.gs = pm.getter(sets, "fp16"), pm.getter(sets, "sealed")
        self.gacc = cast(self.gf, acc)
        d = cfg.hidden
        self.cff = torch.zeros((d, d), dtype=torch.float64, device=device)
        self.fisher = {f: FisherAcc(d, device) for f in FISHERS}
        self.arm_names = ["fp16", "control"] + [a[0] for a in self.arms]
        self.out = {n: [] for n in self.arm_names}
        self.p1: list = []
        self.p1_mismatch = 0
        self.checks: list[dict] = []
        self.energy: list = []  # per question (bases, convs, bands + 1) float64, last = the direct total
        self.mis: list = []  # per question [u_0^2, sum_t u_t^2, |delta|^2], u = delta v_1
        self.g = dict(text_windows=0, text_positions=0, mmlu_q=0, mmlu_positions=0, backward=0, nonfinite=0)
        self.picks_mmlu: list = []  # (float32 pick, f16 pick, max |dz| of the centred logits)
        self.tokens = 0
        self.basis: dict = {}
        self.vdev: dict = {}

    # Pass 1: FP16 alone on the scored prompts.
    def pass1(self, ids_list):
        ids = torch.tensor(ids_list, dtype=torch.long, device=self.device)
        pos = cm.answer_pos(ids.shape[0])

        def stream(l, h):
            if l == self.depth:
                x = h[0].double()
                self.cff += x.T @ x

        with torch.no_grad():
            z = pm.forward(self.cfg, self.gf, ids, stream_hook=stream)
        self.p1.append(z[pos, self.ans_ids].float().cpu())
        self.tokens += len(ids_list)

    # The text pass: fisher_text.
    def text_window(self, w: int, ids_list, gen, check: bool = False):
        ids = torch.tensor(ids_list, dtype=torch.long, device=self.device)
        with torch.no_grad():
            h = to_depth(self.cfg, self.gf, ids, self.depth)
        g, z, _ = text_grad(self.cfg, self.gacc, h, self.depth, gen=gen, ckpt=self.ckpt)
        self.g["backward"] += 1
        if not bool(torch.isfinite(g).all()):
            self.g["nonfinite"] += 1
        self.fisher["fisher_text"].add(g, w % 2)
        self.g["text_windows"] += 1
        self.g["text_positions"] += g.shape[0]
        if check:
            with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
                z0 = text_logits(self.cfg, self.gacc, h.to(self.acc), self.depth, False)
            self.checks.append(dict(kind="grad-forward", label="text window, float32 logits with and without autograd",
                                    q=w, ok=bool(torch.equal(z0, z)), value=float((z0 - z).abs().max())))
        del g, z, h

    # The fisher pass: fisher_mmlu.
    def fisher_question(self, qi: int, ids_list, check: bool = False):
        cfg = self.cfg
        ids = torch.tensor(ids_list, dtype=torch.long, device=self.device)
        pos = cm.answer_pos(ids.shape[0])
        with torch.no_grad():
            h = to_depth(cfg, self.gf, ids, self.depth)
            cos, sin = lvo.rope_tables(cfg, ids.shape[0], self.device, h.dtype)
            z16 = pm.run_from(cfg, self.gf, h, self.depth, cos, sin)[pos, self.ans_ids].float().cpu()
        s, p, z4 = answer_grads(cfg, self.gacc, h, self.depth, pos, self.ans_ids, ckpt=self.ckpt)
        self.g["backward"] += 4
        if not bool(torch.isfinite(s).all()):
            self.g["nonfinite"] += 1
        self.fisher["fisher_mmlu"].add(s * p.sqrt().to(s.dtype)[:, None, None], qi % 2)
        self.g["mmlu_q"] += 1
        self.g["mmlu_positions"] += s.shape[1]
        z32 = z4.float().cpu()
        dz = float(((z32 - z32.mean()) - (z16 - z16.mean())).abs().max())
        self.picks_mmlu.append((cm.pick_of(z32.tolist()), cm.pick_of(z16.tolist()), dz))
        if check:
            with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
                z0 = answer_logits(cfg, self.gacc, h.to(self.acc), self.depth, pos, self.ans_ids, False)
            self.checks.append(dict(kind="grad-forward", label="Fisher question, float32 answer logits with and "
                                    "without autograd", q=qi, ok=bool(torch.equal(z0, z4)),
                                    value=float((z0 - z4).abs().max())))
        del s, h

    def set_bases(self):
        self.basis["variance"] = cm.eig_desc(self.cff.cpu())
        for f in FISHERS:
            self.basis[f] = cm.eig_desc(self.fisher[f].total().cpu())
        v1 = self.basis["variance"][1][:, 0]
        self.v1 = v1.to(self.device)
        for name, (_, v) in self.basis.items():
            self.vdev[name] = (v.to(self.device), (v1 @ v).to(self.device))

    def mat(self, basis: str, idx):
        v = self.basis[basis][1]
        d = self.cfg.hidden
        return torch.cat([v[:, pm.band_cols(*self.bands[i], d)] for i in idx], 1).to(self.device)

    def energies(self, hs, hf):
        """Per basis, convention and band: the energy of the deviation delta = h_s - h_f at depth 12 (every position)
        in that band, and the direct total of the convention's deviation in the canonical basis."""
        dl = hs[0].double() - hf[0].double()
        u = dl @ self.v1
        db = dl.clone()
        db[0] = db[0] - u[0] * self.v1
        dc = dl - u[:, None] * self.v1[None, :]
        tot = [float((dl * dl).sum()), float((db * db).sum()), float((dc * dc).sum())]
        d = self.cfg.hidden
        e = torch.zeros((len(BASES), len(CONVS), len(self.bands) + 1), dtype=torch.float64)
        for bi, name in enumerate(BASES):
            v, r = self.vdev[name]
            y = dl @ v
            yb = y.clone()
            yb[0] = y[0] - u[0] * r
            yc = y - u[:, None] * r[None, :]
            for ci, m in enumerate((y, yb, yc)):
                sq = (m * m).sum(0)
                for j, (lo, hi) in enumerate(self.bands):
                    e[bi, ci, j] = float(sq[pm.band_cols(lo, hi, d)].sum())
                e[bi, ci, -1] = tot[ci]
        self.energy.append(e)
        self.mis.append(torch.tensor([float(u[0] ** 2), float((u * u).sum()), tot[0]], dtype=torch.float64))

    # Pass 2: FP16, control, deviation energies, arms.
    def pass2(self, qi: int, ids_list, check: bool = False):
        cfg, D = self.cfg, self.depth
        ids = torch.tensor(ids_list, dtype=torch.long, device=self.device)
        length = ids.shape[0]
        pos = cm.answer_pos(length)
        with torch.no_grad():
            hf, hs = {}, {}

            def sf(l, h):
                if l in (0, D):
                    hf[l] = h

            def ss(l, h):
                if l == D:
                    hs[l] = h

            z_f = pm.forward(cfg, self.gf, ids, stream_hook=sf)
            lf = z_f[pos, self.ans_ids].float().cpu()
            if not torch.equal(lf, self.p1[qi]):
                self.p1_mismatch += 1
            self.out["fp16"].append(lf)
            del z_f
            z_s = pm.forward(cfg, self.gs, ids, stream_hook=ss)
            self.out["control"].append(z_s[pos, self.ans_ids].float().cpu())
            self.energies(hs[D], hf[D])
            cos, sin = lvo.rope_tables(cfg, length, self.device, hs[D].dtype)
            for name, basis, idx in self.arms:
                b = None if basis is None else self.mat(basis, idx)
                h0 = pm.patch_stream(hs[D], hf[D], None, b)
                z = pm.run_from(cfg, self.gs, h0, D, cos, sin)
                self.out[name].append(z[pos, self.ans_ids].float().cpu())
                del z, h0
            if check:
                self._checks(qi, ids, pos, cos, sin, hf, hs, z_s)
            del z_s, hf, hs

    def _checks(self, qi, ids, pos, cos, sin, hf, hs, z_s):
        cfg, gs, D = self.cfg, self.gs, self.depth
        hd = hs[D]
        last = {n: self.out[n][-1] for n in self.arm_names}
        add = self.checks.append
        for name in BASES:
            for i in (TOP, BOT):
                h0 = pm.patch_stream(hd, hd, None, self.mat(name, (i,)))
                z = pm.run_from(cfg, gs, h0, D, cos, sin)
                add(dict(kind="self-patch", label=f"{name} band {self.bands[i]}", q=qi,
                         ok=bool(torch.equal(h0, hd) and torch.equal(z, z_s)), value=float((z - z_s).abs().max())))
            three = self.mat(name, tuple(range(len(self.bands))))
            h3 = pm.patch_stream(hd, hf[D], None, three)
            z3 = pm.run_from(cfg, gs, h3, D, cos, sin)[pos, self.ans_ids].float().cpu()
            add(dict(kind="three-bands", label=f"{name}: three bands together against the whole stream", q=qi,
                     ok=cm.pick_of(z3.tolist()) == cm.pick_of(last["d12"].tolist()),
                     value=float((z3 - last["d12"]).abs().max()),
                     stream=float((h3.double() - hf[D].double()).abs().max())))
        z_h = pm.forward_hybrid(cfg, self.sets, ids, D)
        add(dict(kind="hybrid", label=f"d12 against FP16 blocks 0..{D - 1} then sealed", q=qi,
                 ok=bool(torch.equal(z_h[pos, self.ans_ids].float().cpu(), last["d12"])),
                 value=float((z_h[pos, self.ans_ids].float().cpu() - last["d12"]).abs().max())))


# ---------------------------------------------------------------------------
# References.
# ---------------------------------------------------------------------------
def read_pm_ref(d: Path) -> dict:
    for name, want in PM_REF_FILES.items():
        got = lvo.sha256_file(d / name)
        if got != want:
            raise SystemExit(f"refused: {d / name} sha256 {got}, expected {want}")
    logits: dict = {}
    for r in pm.read_tsv(d / "logits.tsv"):
        logits.setdefault(r["arm"], {})[(r["subject"], int(r["index"]))] = [float(r[f"out_{x}"]) for x in "abcd"]
    shape = {r["arm"]: float(r["s"]) for r in pm.read_tsv(d / "shape.tsv")}
    return dict(logits=logits, shape=shape)


# ---------------------------------------------------------------------------
# Analysis and files.
# ---------------------------------------------------------------------------
def fmt(t: dict, sign: bool = True) -> str:
    f = "+.3f" if sign else ".3f"
    return f"{t['point']:{f}} [{t['lo']:{f}}, {t['hi']:{f}}]"


def analyse(out: Path, run: SensRun, keys, answers, dump_f, dump_s, ref=None, pm_ref=None, extra_lines=(),
            fisher_keys=None) -> dict:
    cfg = run.cfg
    N = len(keys)
    lines = list(extra_lines)
    say = lines.append
    names = run.arm_names
    ans = torch.tensor(answers)
    L = {n: torch.stack(run.out[n]).double() for n in names}
    picks = {n: torch.tensor([cm.pick_of(r) for r in L[n].tolist()]) for n in names}
    corr = {n: (picks[n] == ans).double() for n in names}
    counts = pm.boot_counts(N)
    one = torch.ones(N, dtype=torch.float64)
    lf_ = L["fp16"].tolist()
    pf_ = picks["fp16"].tolist()
    cos_ = {n: torch.tensor([pm.shape_cos(L[n][i].tolist(), lf_[i])[0] for i in range(N)], dtype=torch.float64)
            for n in names}
    deg = {n: sum(pm.shape_cos(L[n][i].tolist(), lf_[i])[1] for i in range(N)) for n in names}
    E = torch.stack(run.energy)  # (N, bases, convs, bands + 1)
    MIS = torch.stack(run.mis)
    res: dict = {}

    # Gates.
    gates = {}
    say(f"GATES, {N} scored questions")
    ctl = cm.control(keys, L["fp16"], L["control"], dump_f, dump_s)
    gates["rule D"] = ctl["green"]
    say(f"  rule D: picks equal to the Rust dumps {ctl['fp16']['picks_equal']:.4f} (FP16), "
        f"{ctl['sealed']['picks_equal']:.4f} (control); slope script {ctl['fit_script'][0]:.4f} dump "
        f"{ctl['fit_dump'][0]:.4f}: {'GREEN' if ctl['green'] else 'RED'}")
    if fisher_keys is not None:
        gates["Fisher questions disjoint"] = True  # disjoint_or_refuse ran before any model loaded
        say(f"  Fisher questions: {fisher_keys[1]} questions, none among the {fisher_keys[0]} scored (subject, index "
            f"and token hash; checked before the load)")
    by = {k: [c for c in run.checks if c["kind"] == k] for k in ("self-patch", "three-bands", "hybrid", "grad-forward")}
    gates["self-patch bit for bit"] = bool(by["self-patch"]) and all(c["ok"] for c in by["self-patch"])
    say(f"  self-patch with the sealed stream, top and bottom band of each basis: "
        f"{sum(c['ok'] for c in by['self-patch'])} of {len(by['self-patch'])} bit for bit")
    gates["hybrid bit for bit"] = bool(by["hybrid"]) and all(c["ok"] for c in by["hybrid"])
    say(f"  d12 against FP16 blocks 0..{run.depth - 1} then sealed: {sum(c['ok'] for c in by['hybrid'])} of "
        f"{len(by['hybrid'])} bit for bit")
    gates["FP16 pass 2 = pass 1"] = run.p1_mismatch == 0
    say(f"  FP16 logits, pass 2 (after the gradient passes) against pass 1: {N - run.p1_mismatch} of {N} bit for bit")
    for name in BASES:
        tb = [c for c in by["three-bands"] if c["label"].startswith(name + ":")]
        agree = sum(c["ok"] for c in tb) / len(tb) if tb else math.nan
        gates[f"three bands = whole, {name}"] = bool(tb) and agree >= PICK_AGREE_MIN
        say(f"  {name}: three bands together against the whole stream, picks equal {agree:.4f}, worst |dlogit| "
            f"{max((c['value'] for c in tb), default=math.nan):.3e}, worst |dstream| "
            f"{max((c['stream'] for c in tb), default=math.nan):.3e}")
    oe = {name: ortho_err(run.basis[name][1]) for name in BASES}
    gates["bases orthonormal"] = all(v <= ORTHO_TOL for v in oe.values())
    say("  bases orthonormal, max |V^T V - I|: " + ", ".join(f"{k} {v:.1e}" for k, v in oe.items())
        + f" (at most {ORTHO_TOL:g})")
    gf_ = by["grad-forward"]
    gates["gradient forward bit for bit"] = bool(gf_) and all(c["ok"] for c in gf_)
    say(f"  float32 gradient pass, logits with and without autograd: {sum(c['ok'] for c in gf_)} of {len(gf_)} bit "
        f"for bit (worst {max((c['value'] for c in gf_), default=math.nan):.1e})")
    gates["gradients finite"] = run.g["nonfinite"] == 0
    say(f"  gradients: {run.g['nonfinite']} passes with a non-finite value, of {run.g['text_windows']} windows and "
        f"{run.g['mmlu_q']} questions")
    if run.picks_mmlu:
        agree32 = sum(a == b for a, b, _ in run.picks_mmlu) / len(run.picks_mmlu)
        gates["float32 path = f16 forward"] = agree32 >= F32_PICKS_MIN
        say(f"  Fisher questions, the float32 path's pick against FP16's f16 forward: {agree32:.4f} equal (at least "
            f"{F32_PICKS_MIN}); worst |d centred logit| {max(x for _, _, x in run.picks_mmlu):.3e}, median "
            f"{sorted(x for _, _, x in run.picks_mmlu)[len(run.picks_mmlu) // 2]:.3e}")
    worst_e = 0.0
    for bi in range(len(BASES)):
        for ci in range(len(CONVS)):
            s = E[:, bi, ci, :-1].sum(1)
            worst_e = max(worst_e, float(((s - E[:, bi, ci, -1]) / E[:, bi, ci, -1]).abs().max()))
    gates["energy shares sum to 1"] = worst_e <= ENERGY_TOL
    say(f"  deviation energy, three bands against the direct total in the canonical basis, every question, basis and "
        f"convention: worst relative difference {worst_e:.1e} (at most {ENERGY_TOL:g})")
    if ref is not None:
        acc_n = {n: int(corr[n].sum()) for n in ("fp16", "control")}
        rp = {}
        for n, col in (("fp16", "pick_fp16_script"), ("control", "pick_sealed_script")):
            rp[n] = sum(int(picks[n][i]) == int(ref["control"][k][col]) for i, k in enumerate(keys)) / N
        ok = all(abs(acc_n[n] - pm.REF_ACC[n]) <= pm.REPRO_COUNT_TOL and rp[n] >= pm.REPRO_PICKS_MIN for n in acc_n)
        gates["contrast-map reproduced"] = ok
        say(f"  against contrast-map-4b: FP16 {acc_n['fp16']} right ({pm.REF_ACC['fp16']} there), control "
            f"{acc_n['control']} ({pm.REF_ACC['control']}); picks equal {rp['fp16']:.4f} and {rp['control']:.4f}: "
            f"{'GREEN' if ok else 'RED'}")
    sh_point = {n: (float(cos_[n].sum()) - float(cos_["control"].sum())) / float((1 - cos_["control"]).sum())
                for n in names}
    if pm_ref is not None:
        okp = True
        parts = []
        for mine, theirs in PM_SAME.items():
            ref_l = torch.tensor([pm_ref["logits"][theirs][k] for k in keys], dtype=torch.float64)
            pe = float((torch.tensor([cm.pick_of(r) for r in ref_l.tolist()]) == picks[mine]).double().mean())
            ds = abs(sh_point[mine] - pm_ref["shape"][theirs])
            okp &= pe >= PM_PICKS_MIN and ds <= PM_SHARE_TOL
            parts.append(f"{mine} picks {pe:.4f}, s {sh_point[mine]:+.4f} against {pm_ref['shape'][theirs]:+.4f}, "
                         f"|dlogit| {float((ref_l - L[mine]).abs().max()):.2e}")
        gates["patch-map reproduced"] = okp
        say("  against patch-map-4b: " + "; ".join(parts) + f": {'GREEN' if okp else 'RED'}")

    # Arms.
    say("")
    say("ARMS: c = cosine of the centred four logits with FP16's; s = (C - C_control) / (1 - C_control), primary; "
        "accuracy, paired against the control; KL(FP16 || arm) and log p(FP16's pick) over the four options; 95 % "
        "paired bootstrap")
    rows = {}
    gap = corr["fp16"] - corr["control"]
    for n in names:
        pr = pm.paired(corr[n], corr["control"])
        rows[n] = dict(C=float(cos_[n].mean()),
                       s=pm.boot(torch.stack((cos_[n] - cos_["control"], one - cos_["control"]), 1),
                                 pm.ratio_of_gains, counts),
                       acc=float(corr[n].mean()), pr=pr,
                       acc_share=pm.boot(torch.stack((corr[n] - corr["control"], gap), 1), pm.ratio_of_gains, counts),
                       kl=pm.boot(torch.stack((torch.tensor([pm.kl4(lf_[i], L[n][i].tolist()) for i in range(N)],
                                                            dtype=torch.float64), one), 1), pm.ratio_of_gains, counts),
                       lp=pm.boot(torch.stack((torch.tensor([pm.logp_of(L[n][i].tolist(), pf_[i]) for i in range(N)],
                                                            dtype=torch.float64), one), 1), pm.ratio_of_gains, counts),
                       deg=deg[n])
    res["arms"] = rows
    say("  arm               C        s [95 %]                  acc     minus control [95 %]      b/c      "
        "acc share [95 %]          KL [95 %]")
    for n, r in rows.items():
        p = r["pr"]
        say(f"  {n:<16}  {r['C']:.5f}  {fmt(r['s'])}  {r['acc'] * 100:6.2f}  {p['diff'] * 100:+6.2f} "
            f"[{p['lo'] * 100:+6.2f}, {p['hi'] * 100:+6.2f}]  {p['b']:>3}/{p['c']:<3}  {fmt(r['acc_share'])}  "
            f"{fmt(r['kl'], False)}")
    with open(out / "arms.tsv", "w") as f:
        f.write("arm\tC\ts\ts_lo\ts_hi\tacc\tdiff\tdiff_lo\tdiff_hi\tb\tc\tmcnemar_p\tacc_share\tacc_share_lo\t"
                "acc_share_hi\tkl\tkl_lo\tkl_hi\tlogp_fp16_pick\tlogp_lo\tlogp_hi\tdegenerate\n")
        for n, r in rows.items():
            t = lambda k: "\t".join(f"{r[k][x]:.6f}" for x in ("point", "lo", "hi"))  # noqa: E731
            p = r["pr"]
            f.write(f"{n}\t{r['C']:.7f}\t{t('s')}\t{r['acc']:.6f}\t{p['diff']:.6f}\t{p['lo']:.6f}\t{p['hi']:.6f}\t"
                    f"{p['b']}\t{p['c']}\t{p['p']:.6g}\t{t('acc_share')}\t{t('kl')}\t{t('lp')}\t{r['deg']}\n")

    # Deviation energy by band.
    say("")
    tot_all = float(E[:, 0, 0, -1].sum())
    say(f"DEVIATION ENERGY at depth {run.depth}, h_s - h_f every position, share of the convention's total. Direction "
        f"1 of the variance basis at position 0 holds {float(MIS[:, 0].sum()) / tot_all:.4f} of all the deviation "
        f"energy, at every position {float(MIS[:, 1].sum()) / tot_all:.4f}. Conventions: all = every coordinate; "
        f"pos0 = direction 1 removed at position 0; dir1 = direction 1 removed at every position (primary)")
    eshare = {}
    say("  basis         conv   " + "   ".join(f"[{lo}, {hi}]" for lo, hi in run.bands)
        + f"   [{run.bands[WIDE[0]][0]}, {run.bands[WIDE[-1]][1]}]")
    for bi, name in enumerate(BASES):
        for ci, conv in enumerate(CONVS):
            tot = float(E[:, bi, ci, -1].sum())
            eshare[(name, conv)] = [float(E[:, bi, ci, j].sum()) / tot for j in range(len(run.bands))]
            ew = sum(eshare[(name, conv)][j] for j in WIDE)
            say(f"  {name:<12}  {conv:<5}  " + "   ".join(f"{x:.5f}" for x in eshare[(name, conv)]) + f"   {ew:.5f}")

    # The give-back a unit of deviation energy, by basis.
    say("")
    say("GIVE-BACK A UNIT OF DEVIATION ENERGY: s(band) / E(band); ratio top band over bottom band, paired bootstrap "
        "95 % (energies resampled with the questions); cross term s_1 + s_2 + s_3 - s(d12)")
    res["bases"] = {}
    for bi, name in enumerate(BASES):
        bn = [f"{SHORT[name]}_b{lo}-{hi}" for lo, hi in run.bands]
        out_b = dict(s=[rows[a]["s"] for a in bn])
        ratios = {}
        for ci, conv in enumerate(CONVS):
            cols = torch.stack((cos_[bn[TOP]] - cos_["control"], cos_[bn[BOT]] - cos_["control"],
                                E[:, bi, ci, TOP], E[:, bi, ci, BOT]), 1)
            ratios[conv] = pm.boot(cols, give_back_ratio, counts)
        acc_cols = torch.stack((corr[bn[TOP]] - corr["control"], corr[bn[BOT]] - corr["control"],
                                E[:, bi, CONVS.index(PRIMARY_CONV), TOP], E[:, bi, CONVS.index(PRIMARY_CONV), BOT]), 1)
        ratios["accuracy"] = pm.boot(acc_cols, give_back_ratio, counts)
        wn = wide_name(name, run.bands)
        ew_q = {ci: sum(E[:, bi, ci, j] for j in WIDE) for ci in range(len(CONVS))}
        for ci, conv in enumerate(CONVS):
            cols = torch.stack((cos_[wn] - cos_["control"], cos_[bn[BOT]] - cos_["control"], ew_q[ci],
                                E[:, bi, ci, BOT]), 1)
            ratios["w_" + conv] = pm.boot(cols, give_back_ratio, counts)
        pc = CONVS.index(PRIMARY_CONV)
        ratios["w_accuracy"] = pm.boot(torch.stack((corr[wn] - corr["control"], corr[bn[BOT]] - corr["control"],
                                                    ew_q[pc], E[:, bi, pc, BOT]), 1), give_back_ratio, counts)
        out_b["e_wide"] = float(ew_q[pc].sum() / E[:, bi, pc, -1].sum())
        out_b["s_wide"] = rows[wn]["s"]
        cx = pm.boot(torch.stack([cos_[a] - cos_["control"] for a in bn]
                                 + [cos_["d12"] - cos_["control"], one - cos_["control"]], 1), cross_term, counts)
        out_b.update(ratios=ratios, cross=cx)
        es = eshare[(name, PRIMARY_CONV)]
        say(f"  {name}: s " + ", ".join(fmt(t) for t in out_b["s"]) + "; per unit energy (" + PRIMARY_CONV + ") "
            + ", ".join(f"{t['point'] / e:.2f}" if e > 0 else "nan" for t, e in zip(out_b["s"], es)))
        say(f"    ratio top / bottom: {PRIMARY_CONV} {fmt(ratios[PRIMARY_CONV])} ({ratios[PRIMARY_CONV]['bad']} draws "
            f"left out); pos0 {fmt(ratios['pos0'])}; all {fmt(ratios['all'])}; on the accuracy share "
            f"{fmt(ratios['accuracy'])}")
        say(f"    [{run.bands[WIDE[0]][0]}, {run.bands[WIDE[-1]][1]}] together: s {fmt(out_b['s_wide'])}, E "
            f"{out_b['e_wide']:.5f} ({PRIMARY_CONV}); ratio over the bottom band: {PRIMARY_CONV} "
            f"{fmt(ratios['w_' + PRIMARY_CONV])}; pos0 {fmt(ratios['w_pos0'])}; all {fmt(ratios['w_all'])}; on the "
            f"accuracy share {fmt(ratios['w_accuracy'])}")
        say(f"    cross term s_1 + s_2 + s_3 - s(d12): {fmt(cx)}")
        if name in FISHERS:
            lo, hi = run.bands[TOP]
            comp = f"{SHORT[name]}_not{lo}-{hi}"
            marg = pm.boot(torch.stack((cos_["d12"] - cos_[comp], one - cos_["control"]), 1), pm.ratio_of_gains, counts)
            out_b.update(complement=rows[comp]["s"], marginal=marg)
            say(f"    every band but [{lo}, {hi}]: s {fmt(rows[comp]['s'])}; s(d12) minus it, the top band's marginal "
                f"give-back: {fmt(marg)}")
        res["bases"][name] = out_b

    # Spectra and overlaps.
    say("")
    say("SPECTRA (*computed* on FP16 alone): trace share of each band, arithmetic over geometric mean of the "
        "eigenvalues (transform-coding gain), split-half overlap of the top-64 subspaces")
    nsamp = {"variance": f"{run.tokens} positions of the {N} scored prompts",
             "fisher_text": f"{run.g['text_windows']} windows, {run.g['text_positions']} position gradients, one "
                            f"sampled label a position",
             "fisher_mmlu": f"{run.g['mmlu_q']} questions, {run.g['mmlu_positions']} positions x 4 scores, "
                            f"{4 * run.g['mmlu_q']} backward passes"}
    d = cfg.hidden
    k = min(OVERLAP_K, d)
    res["spectra"] = {}
    for name in BASES:
        sp = spectrum(run.basis[name][0], run.bands)
        line = (f"  {name:<12} " + ", ".join(f"{x:.4f}" for x in sp["shares"]) + f"; AM/GM {sp['am_gm']:.3e} "
                f"({sp['gain_db']:.1f} dB, {sp['floored']} under the floor); {nsamp[name]}")
        if name in FISHERS:
            fa = run.fisher[name]
            va, vb = cm.eig_desc(fa.half[0].cpu())[1], cm.eig_desc(fa.half[1].cpu())[1]
            sp["split_half"] = overlap(va, vb, k)
            line += f"; split-half top-{k} overlap {sp['split_half']:.3f} (halves of {fa.n[0]} and {fa.n[1]} vectors)"
        say(line)
        res["spectra"][name] = sp
    ov = {}
    for a, b in (("fisher_text", "fisher_mmlu"), ("fisher_text", "variance"), ("fisher_mmlu", "variance")):
        ov[(a, b)] = overlap(run.basis[a][1], run.basis[b][1], k)
    res["overlaps"] = ov
    say(f"  top-{k} overlaps (1 the same subspace, {k / d:.4f} at random): "
        + ", ".join(f"{a} / {b} {v:.3f}" for (a, b), v in ov.items()))

    # Kill rule.
    say("")
    sd12 = rows["d12"]["s"]
    rt = res["bases"]["fisher_text"]["ratios"][PRIMARY_CONV]
    rtw = res["bases"]["fisher_text"]["ratios"]["w_" + PRIMARY_CONV]
    et_w = res["bases"]["fisher_text"]["e_wide"]
    rm = res["bases"]["fisher_mmlu"]["ratios"][PRIMARY_CONV]
    rmw = res["bases"]["fisher_mmlu"]["ratios"]["w_" + PRIMARY_CONV]
    rv = res["bases"]["variance"]["ratios"][PRIMARY_CONV]
    rvw = res["bases"]["variance"]["ratios"]["w_" + PRIMARY_CONV]
    vctl = var_control_ok(rv["point"]) and var_control_ok(rvw["point"], VAR_CTRL_W)
    if pm_ref is not None:
        gates["variance control"] = vctl
    green = all(gates.values())
    res["gates"], res["green"] = gates, green
    verdict = verdict_rule(rt, rtw, et_w)
    if not (math.isfinite(sd12["lo"]) and sd12["lo"] > 0):
        verdict = "NOT DECIDED (s(d12) reaches 0)"
    if not green:
        verdict = "NOT READ (a gate is red)"
    res["verdict"] = verdict
    say(f"KILL RULE (prereg table). fisher_text, give-back a unit of deviation energy ({PRIMARY_CONV}) over the bottom "
        f"band's: [1, 64] {fmt(rt)}; [1, 512] {fmt(rtw)}, holding {et_w:.4f} of the deviation energy: {verdict}. "
        f"UNEVEN if either whole 95 % interval lies above {UNEVEN_MIN:g}; FLAT if the [1, 512] interval lies inside "
        f"[{FLAT_LO:g}, {FLAT_HI:g}], the [1, 64] one does not lie above {UNEVEN_MIN:g}, and [1, 512] holds at least "
        f"{FLAT_E_MIN:g} of the energy; NOT DECIDED otherwise. s(d12) {fmt(sd12)}.")
    say(f"  beside, no rule: fisher_mmlu [1, 64] {fmt(rm)}, [1, 512] {fmt(rmw)}; variance [1, 64] {fmt(rv)}, "
        f"[1, 512] {fmt(rvw)}, the control: points inside [{VAR_CTRL[0]:g}, {VAR_CTRL[1]:g}] (expected 1.03) and "
        f"[{VAR_CTRL_W[0]:g}, {VAR_CTRL_W[1]:g}] (expected 0.35 to 1.14): {'GREEN' if vctl else 'RED'}")
    say(f"  GATES: {'GREEN' if green else 'RED, NOTHING ABOVE IS READ'} "
        f"({', '.join(k for k, v in gates.items() if not v) or 'all pass'})")
    with open(out / "kills.tsv", "w") as f:
        f.write("basis\tconvention\tpoint\tlo\thi\tbad_draws\tverdict\n")
        for name in BASES:
            for conv, t in res["bases"][name]["ratios"].items():
                vd = verdict if (name == "fisher_text" and conv in (PRIMARY_CONV, "w_" + PRIMARY_CONV)) else "(no rule)"
                f.write(f"{name}\t{conv}\t{t['point']:.6f}\t{t['lo']:.6f}\t{t['hi']:.6f}\t{t['bad']}\t{vd}\n")

    # Files.
    with open(out / "logits.tsv", "w") as f:
        f.write("subject\tindex\tarm\t" + "\t".join(f"out_{x}" for x in "abcd") + "\n")
        for n in names:
            for i, kk in enumerate(keys):
                f.write(f"{kk[0]}\t{kk[1]}\t{n}\t" + "\t".join(f"{v:.7g}" for v in L[n][i].tolist()) + "\n")
    with open(out / "picks.tsv", "w") as f:
        f.write("order\tsubject\tindex\tanswer\t" + "\t".join(f"pick_{n}" for n in names) + "\n")
        for i, kk in enumerate(keys):
            f.write(f"{i}\t{kk[0]}\t{kk[1]}\t{answers[i]}\t" + "\t".join(str(int(picks[n][i])) for n in names) + "\n")
    with open(out / "checks.tsv", "w") as f:
        f.write("kind\tlabel\tquestion\tok\tvalue\n")
        for c in run.checks:
            f.write(f"{c['kind']}\t{c['label']}\t{c['q']}\t{c['ok']}\t{c['value']:.6e}\n")
    with open(out / "spectra.tsv", "w") as f:
        f.write("basis\t" + "\t".join(f"trace_share_{lo}-{hi}" for lo, hi in run.bands)
                + "\tam_gm\tgain_db\tfloored\tsplit_half_top64\n")
        for name, sp in res["spectra"].items():
            f.write(f"{name}\t" + "\t".join(f"{x:.6g}" for x in sp["shares"])
                    + f"\t{sp['am_gm']:.6g}\t{sp['gain_db']:.4f}\t{sp['floored']}\t{sp.get('split_half', math.nan):.6f}\n")
    cm.write_raw(out / "energies.f64", E, keys)
    cm.write_raw(out / "misscale.f64", MIS, keys)
    for name in BASES:
        cm.write_raw(out / f"eigvals_{name}.f64", run.basis[name][0], keys)
    for name in FISHERS:
        cm.write_raw(out / f"{name}.f64", run.fisher[name].total(), keys)
    say("")
    say("RAW: arms.tsv, kills.tsv, logits.tsv, picks.tsv, checks.tsv, spectra.tsv, energies.f64 (N x 3 bases x 3 "
        "conventions x [3 bands, direct total]), misscale.f64 (N x [u_0^2, sum u^2, |delta|^2]), eigvals_{basis}.f64, "
        "fisher_{text,mmlu}.f64 (the 2560 x 2560 sums)")
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
    D = DEPTH
    fp16 = lvo.tiny_weights(cfg, 1)
    sealed = {k: v.clone() for k, v in fp16.items()}
    g0 = torch.Generator().manual_seed(3)
    names = lvo.all_projections(cfg.layers)
    for n in names:
        sealed[n] = fp16[n] + 0.08 * torch.randn(fp16[n].shape, generator=g0, dtype=torch.float64) * fp16[n].abs().mean()
    sets = {"fp16": fp16, "sealed": sealed}
    gf = pm.getter(sets, "fp16")
    ans_ids = [3, 7, 11, 13]
    gen = torch.Generator().manual_seed(5)
    mk = lambda n: torch.randint(4, cfg.vocab, (n,), generator=gen).tolist()  # noqa: E731
    prompts = [mk(n) for n in (19, 23, 17, 29, 21, 25)]
    fprompts = [mk(n) for n in (18, 22, 27, 20, 24, 16)]
    windows = [mk(24) for _ in range(8)]
    bands = ((1, 2), (3, 6), (7, 16))

    # 1. The forward pieces.
    ok = ok2 = True
    for p in prompts[:3]:
        ids = torch.tensor(p)
        cap = {}
        z = pm.forward(cfg, gf, ids, stream_hook=lambda l, h: cap.__setitem__(l, h))
        h = to_depth(cfg, gf, ids, D)
        ok &= torch.equal(h, cap[D])
        ok &= torch.equal(text_logits(cfg, gf, h, D, False), z) and torch.equal(text_logits(cfg, gf, h, D, True), z)
        ok2 &= float((answer_logits(cfg, gf, h, D, len(p) - 1, ans_ids, True) - z[-1, ans_ids]).abs().max()) < 1e-12
    check(ok, "to_depth gives patch_map.forward's h(12) and the tail (checkpointed or not) its logits, bit for bit")
    check(ok2, "the four-row answer head equals the full head's answer row")

    # 2. Autograd leaves the forward bit for bit.
    h = to_depth(cfg, gf, torch.tensor(windows[0]), D)
    g_, z_, y_ = text_grad(cfg, gf, h, D, gen=torch.Generator().manual_seed(1))
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        z0 = text_logits(cfg, gf, h, D, False)
    hp = to_depth(cfg, gf, torch.tensor(prompts[0]), D)
    s_, p_, z4_ = answer_grads(cfg, gf, hp, D, len(prompts[0]) - 1, ans_ids)
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        z40 = answer_logits(cfg, gf, hp, D, len(prompts[0]) - 1, ans_ids, False)
    check(torch.equal(z0, z_) and torch.equal(z40, z4_), "logits with and without autograd are equal bit for bit")

    # 3. Gradients against finite differences. The forward's RMSNorm runs in float32 whatever the stream's dtype
    # (layer_vs_output.rms_norm), so a small step reads rounding: central differences at steps 1e-2 and 5e-3,
    # Richardson-extrapolated, agree to about 2e-4; a wrong gradient is off by order 1.
    gd = torch.Generator().manual_seed(7)
    FD_EPS, FD_TOL = 1e-2, 2e-3

    def richardson(f, x, u):
        d1 = (f(x + FD_EPS * u) - f(x - FD_EPS * u)) / (2 * FD_EPS)
        d2 = (f(x + FD_EPS / 2 * u) - f(x - FD_EPS / 2 * u)) / FD_EPS
        return (4 * d2 - d1) / 3

    def f_text(x):
        with torch.no_grad():
            return float(F.log_softmax(text_logits(cfg, gf, x, D, False), -1).gather(1, y_[:, None]).sum())
    worst = 0.0
    for _ in range(3):
        u = torch.randn(h.shape, generator=gd, dtype=torch.float64)
        an = float((g_ * u[0]).sum())
        worst = max(worst, abs(richardson(f_text, h, u) - an) / max(abs(an), 1e-12))
    check(worst < FD_TOL, f"text log-likelihood gradient against finite differences, 3 directions ({worst:.1e})")
    worst = 0.0
    pos0 = len(prompts[0]) - 1
    for kk in range(4):
        u = torch.randn(hp.shape, generator=gd, dtype=torch.float64)

        def f_ans(x):
            with torch.no_grad():
                return float(F.log_softmax(answer_logits(cfg, gf, x, D, pos0, ans_ids, False), -1)[kk])
        an = float((s_[kk] * u[0]).sum())
        worst = max(worst, abs(richardson(f_ans, hp, u) - an) / max(abs(an), 1e-12))
    check(worst < FD_TOL, f"four-option log p_k gradients against finite differences, each k ({worst:.1e})")
    check(abs(float(s_[0, 0, 0])) > 0 and float(s_[:, :-1].abs().sum()) > 0,
          "the answer gradient reaches the earlier positions (through attention)")

    # 4. The exact four-option Fisher (identities to float32 precision, the RMSNorm's).
    mean_s = (p_[:, None, None] * s_).sum(0)
    check(float(mean_s.abs().max()) < 1e-5 * float(s_.abs().max()),
          f"the four scores have mean 0 under p ({float(mean_s.abs().max()) / float(s_.abs().max()):.1e})")
    x4 = hp.detach().clone().requires_grad_(True)
    with sdpa_kernel(SDPBackend.MATH):
        zz = answer_logits(cfg, gf, x4, D, pos0, ans_ids, True)
        J = torch.stack([torch.autograd.grad(zz[kk], x4, retain_graph=kk < 3)[0][0] for kk in range(4)])
    Jf = J.reshape(4, -1)
    Ff = Jf.T @ (torch.diag(p_) - torch.outer(p_, p_)) @ Jf
    Sf = (s_ * p_.sqrt()[:, None, None]).reshape(4, -1)
    check(float((Sf.T @ Sf - Ff).abs().max()) < 1e-5 * float(Ff.abs().max()),
          "sum_k p_k s_k s_k^T = J^T (diag p - p p^T) J, the Fisher of the four-way answer")

    # 5. Sampled labels give the true Fisher: the cross-target terms vanish in expectation.
    wl = windows[1][:10]
    hw = to_depth(cfg, gf, torch.tensor(wl), D)
    exact = torch.zeros((hw.shape[1] * 16,) * 2, dtype=torch.float64)
    xw = hw.detach().clone().requires_grad_(True)
    with sdpa_kernel(SDPBackend.MATH):
        lpw = F.log_softmax(text_logits(cfg, gf, xw, D, False), -1)
    pw = lpw.detach().exp()
    for t in range(len(wl)):
        for y in range(cfg.vocab):
            gy = torch.autograd.grad(lpw[t, y], xw, retain_graph=True)[0][0].reshape(-1)
            exact += pw[t, y] * torch.outer(gy, gy)
    gs_ = torch.Generator().manual_seed(11)
    mc = torch.zeros_like(exact)
    draws = 600
    for _ in range(draws):
        gg, _, _ = text_grad(cfg, gf, hw, D, gen=gs_, ckpt=False)
        gg = gg.reshape(-1)
        mc += torch.outer(gg, gg) / draws
    rel = float((mc - exact).norm() / exact.norm())
    check(rel < 0.15, f"sampled-label G G^T over {draws} draws = the exact Fisher of the window, cross-position "
                      f"blocks included (relative Frobenius error {rel:.3f})")

    # 6. The float16 path: f16 weights, float32 gradient against float64 on the same f16 weights.
    try:
        f16w = {kk: v.to(torch.float16) for kk, v in fp16.items()}
        g16 = pm.getter({"fp16": f16w, "sealed": f16w}, "fp16")
        g64 = cast(g16, torch.float64)
        hh = to_depth(cfg, g16, torch.tensor(windows[2]), D)
        a32, _, yy = text_grad(cfg, cast(g16, torch.float32), hh, D, gen=torch.Generator().manual_seed(2))
        a64, _, _ = text_grad(cfg, g64, hh, D, labels=yy)
        r16 = float((a32.double() - a64).norm() / a64.norm())
        check(hh.dtype == torch.float16 and a32.dtype == torch.float32 and bool(torch.isfinite(a32).all())
              and r16 < 1e-4, f"f16 weights, f16 stream to depth 12, float32 gradient: finite, within {r16:.1e} of "
                              f"float64 on the same weights")
    except RuntimeError as e:  # CPU half kernels missing: say so, never pass silently
        check(False, f"the float16 path could not run on this CPU: {e}")

    # 7. The Fisher accumulator recovers a planted subspace.
    gq = torch.Generator().manual_seed(13)
    U, _ = torch.linalg.qr(torch.randn(16, 3, generator=gq, dtype=torch.float64))
    fa = FisherAcc(16, cpu)
    for i in range(40):
        gsyn = torch.randn(30, 3, generator=gq, dtype=torch.float64) * torch.tensor([3.0, 2.0, 1.5]) @ U.T
        gsyn += 0.05 * torch.randn(30, 16, generator=gq, dtype=torch.float64)
        fa.add(gsyn, i % 2)
    lam_f, v_f = cm.eig_desc(fa.total())
    o3 = overlap(v_f, U, 3)
    oh = overlap(cm.eig_desc(fa.half[0])[1], cm.eig_desc(fa.half[1])[1], 3)
    check(o3 > 0.99 and oh > 0.98 and fa.n == [600, 600] and torch.allclose(fa.total(), fa.total().T),
          f"planted rank-3 sensitivity: top-3 overlap {o3:.4f}, split halves {oh:.4f}, 600 vectors a half")
    vr, _ = torch.linalg.qr(torch.randn(16, 16, generator=gq, dtype=torch.float64))
    check(abs(overlap(vr, vr, 4) - 1) < 1e-12 and overlap(vr[:, :4], vr[:, 4:8], 4) < 1e-24,
          "overlap: one subspace 1, orthogonal 0")

    # 8. A model whose sensitivity at depth 12 is planted low rank: blocks 12-35 write nothing, the head is rank 2.
    tcfg = dict(lvo.TINY)
    tcfg["tie_word_embeddings"] = False
    cfg2 = lvo.Cfg(tcfg)
    w2 = lvo.tiny_weights(cfg2, 4)
    for b in range(D, cfg2.layers):
        w2[lvo.proj_name(b, "o_proj")].zero_()
        w2[lvo.proj_name(b, "down_proj")].zero_()
    gq2 = torch.Generator().manual_seed(17)
    A = torch.randn(cfg2.vocab, 2, generator=gq2, dtype=torch.float64) * 2.0
    Bm = torch.randn(16, 2, generator=gq2, dtype=torch.float64)
    plant = {}
    for kind, head in (("rank-2", A @ Bm.T), ("full", torch.randn(cfg2.vocab, 16, generator=gq2, dtype=torch.float64))):
        w2["lm_head.weight"] = head
        get2 = pm.getter({"fp16": w2, "sealed": w2}, "fp16")
        fa2 = FisherAcc(16, cpu)
        g2 = torch.Generator().manual_seed(19)
        cff2 = torch.zeros((16, 16), dtype=torch.float64)
        for i, wv in enumerate(windows):
            h2 = to_depth(cfg2, get2, torch.tensor(wv), D)
            cff2 += h2[0].T @ h2[0]
            for _ in range(4):
                gg, _, _ = text_grad(cfg2, get2, h2, D, gen=g2)
                fa2.add(gg, i % 2)
        span, _ = torch.linalg.qr(w2["model.norm.weight"][:, None] * Bm)
        plant[kind] = (overlap(cm.eig_desc(fa2.total())[1], span, 2), overlap(cm.eig_desc(cff2)[1], span, 2),
                       spectrum(cm.eig_desc(fa2.total())[0], bands))
    check(plant["rank-2"][0] > 0.9 and plant["full"][0] < 0.6 and plant["rank-2"][1] < 0.6,
          f"planted rank-2 head: the text Fisher's top-2 overlap with diag(g) B {plant['rank-2'][0]:.3f} (full-rank "
          f"head {plant['full'][0]:.3f}, the variance basis {plant['rank-2'][1]:.3f}, random 0.125)")
    check(plant["rank-2"][2]["am_gm"] > 10 * plant["full"][2]["am_gm"] and plant["rank-2"][2]["shares"][0] > 0.9,
          f"its spectrum: top-2 trace share {plant['rank-2'][2]['shares'][0]:.3f}, AM/GM {plant['rank-2'][2]['am_gm']:.2e}"
          f" against {plant['full'][2]['am_gm']:.2e} for the full-rank head")
    flat = spectrum(torch.full((16,), 2.0, dtype=torch.float64), bands)
    check(abs(flat["am_gm"] - 1) < 1e-12 and abs(sum(flat["shares"]) - 1) < 1e-12
          and abs(spectrum(torch.tensor([4.0, 1.0] + [0.0] * 14, dtype=torch.float64), bands)["am_gm"] - 1.25) < 1e-12,
          "AM/GM: equal eigenvalues 1, (4, 1) 1.25 with the zeros under the floor; band trace shares sum to 1")

    # 9. The run on the tiny pair, every pass and check.
    run = SensRun(cfg, sets, cpu, ans_ids, bands=bands, acc=torch.float64)
    for p in prompts:
        run.pass1(p)
    gt = torch.Generator().manual_seed(LABEL_SEED)
    for w, wv in enumerate(windows):
        run.text_window(w, wv, gt, check=w < 2)
    for qi, p in enumerate(fprompts):
        run.fisher_question(qi, p, check=qi < 2)
    run.set_bases()
    for qi, p in enumerate(prompts):
        run.pass2(qi, p, check=True)
    by = {kk: [c for c in run.checks if c["kind"] == kk] for kk in ("self-patch", "three-bands", "hybrid", "grad-forward")}
    check(len(by["self-patch"]) == 36 and all(c["ok"] for c in by["self-patch"]),
          f"self-patch, top and bottom band of each of the three bases, 6 prompts: "
          f"{sum(c['ok'] for c in by['self-patch'])} of {len(by['self-patch'])} bit for bit")
    check(len(by["three-bands"]) == 18 and all(c["value"] < 1e-9 and c["stream"] < 1e-12 for c in by["three-bands"]),
          f"three bands together = the whole stream in each basis: worst |dlogit| "
          f"{max(c['value'] for c in by['three-bands']):.1e}")
    check(len(by["hybrid"]) == 6 and all(c["ok"] for c in by["hybrid"]), "d12 = FP16 blocks 0-11 then sealed, bit for bit")
    check(len(by["grad-forward"]) == 4 and all(c["ok"] for c in by["grad-forward"]) and run.p1_mismatch == 0,
          "the run's gradient passes leave its float logits bit for bit, and FP16's pass 2 equals pass 1")
    check(all(ortho_err(run.basis[n][1]) < 1e-12 for n in BASES) and ortho_err(run.basis["variance"][1] + 1e-6) > 1e-7,
          "each basis is orthonormal; a perturbed one is caught")
    check(run.g["text_windows"] == 8 and run.g["mmlu_q"] == 6 and run.g["backward"] == 8 + 24
          and run.fisher["fisher_text"].n == [96, 96]
          and run.fisher["fisher_mmlu"].n == [sum(4 * len(p) for p in fprompts[0::2]), sum(4 * len(p) for p in fprompts[1::2])],
          "gradient counts: 8 windows of 24, 6 questions x 4 scores, positions split by parity")
    cff_ = torch.zeros((16, 16), dtype=torch.float64)
    for p in prompts:
        hh_ = to_depth(cfg, gf, torch.tensor(p), D)[0]
        cff_ += hh_.T @ hh_
    ft_ = torch.zeros_like(cff_)
    gt2 = torch.Generator().manual_seed(LABEL_SEED)
    for wv in windows:
        gg, _, _ = text_grad(cfg, gf, to_depth(cfg, gf, torch.tensor(wv), D), D, gen=gt2)
        ft_ += gg.T @ gg
    fm_ = torch.zeros_like(cff_)
    for p in fprompts:
        ss_, pp_, _ = answer_grads(cfg, gf, to_depth(cfg, gf, torch.tensor(p), D), D, len(p) - 1, ans_ids)
        for kk in range(4):
            fm_ += pp_[kk] * ss_[kk].T @ ss_[kk]
    rel_ = lambda a, b: float((a - b).abs().max() / b.abs().max())  # noqa: E731
    check(rel_(run.cff, cff_) < 1e-12 and rel_(run.fisher["fisher_text"].total(), ft_) < 1e-12
          and rel_(run.fisher["fisher_mmlu"].total(), fm_) < 1e-12,
          "the run's C_ff, text Fisher and four-way Fisher equal their recomputation from h(12): sum h h^T, "
          "sum g g^T on the same sampled labels, sum_k p_k s_k s_k^T")
    lam_t, v_t = run.basis["fisher_text"]
    check(torch.allclose(v_t @ torch.diag(lam_t) @ v_t.T, run.fisher["fisher_text"].total(), atol=1e-9)
          and bool((lam_t[:-1] >= lam_t[1:]).all()),
          "the fisher_text basis is the descending eigenbasis of the accumulated sum")

    # 10. Band patches in a basis that is not the variance order.
    hs_, hf_ = {}, {}
    pr = prompts[3]
    ids3 = torch.tensor(pr)
    pm.forward(cfg, gf, ids3, stream_hook=lambda l, h: hf_.__setitem__(l, h))
    pm.forward(cfg, pm.getter(sets, "sealed"), ids3, stream_hook=lambda l, h: hs_.__setitem__(l, h))
    tot_ = sum(pm.patch_stream(hs_[D], hf_[D], None, run.mat("fisher_mmlu", (i,))) - hs_[D] for i in range(3))
    check(float((tot_ - (hf_[D] - hs_[D])).abs().max()) < 1e-12,
          "fisher_mmlu: the three band patches' changes sum to the whole difference")
    pt = pm.patch_stream(hs_[D], hf_[D], None, run.mat("fisher_text", (TOP,)))
    dv = (pt - hs_[D])[0] @ run.basis["fisher_text"][1]
    want = (hf_[D] - hs_[D])[0] @ run.basis["fisher_text"][1]
    check(float((dv[:, :2] - want[:, :2]).abs().max()) < 1e-12 and float(dv[:, 2:].abs().max()) < 1e-12,
          "a fisher_text band patch moves its own coordinates to FP16's and leaves the others")
    cos3, sin3 = lvo.rope_tables(cfg, len(pr), cpu, torch.float64)
    gs_ = pm.getter(sets, "sealed")
    okar = True
    for arm_, vb_ in (("ft_b7-16", run.basis["fisher_text"][1][:, 6:16]), ("fm_not1-2", run.basis["fisher_mmlu"][1][:, 2:]),
                      ("var_b1-2", run.basis["variance"][1][:, :2])):
        za = pm.run_from(cfg, gs_, pm.patch_stream(hs_[D], hf_[D], None, vb_), D, cos3, sin3)[-1, ans_ids].float()
        okar &= torch.equal(za, run.out[arm_][3])
    check(okar, "three arms' logits equal the patch written out with their basis columns, bit for bit")
    comp = run.mat("fisher_text", (1, 2))
    names_ = [a[0] for a in run.arms]
    check(torch.equal(comp, run.basis["fisher_text"][1][:, 2:]) and len(run.arms) == 1 + 9 + 2 + 3
          and names_[10:12] == ["ft_not1-2", "fm_not1-2"] and names_[12:] == ["var_b1-6", "ft_b1-6", "fm_b1-6"]
          and torch.equal(run.mat("fisher_text", WIDE), run.basis["fisher_text"][1][:, :6]),
          "the complement arm patches every band but the top one; the wide arms the top two bands; 15 arms")
    full = [a[0] for a in ARMS]
    check(full[:4] == ["d12", "var_b1-64", "var_b65-512", "var_b513-2560"] and set(PM_SAME) <= set(full)
          and full[-3:] == ["var_b1-512", "ft_b1-512", "fm_b1-512"]
          and len(full) == 15, "production arms: d12, three bands of each basis, two complements; patch-map's names map")

    # 11. Deviation energies.
    E = torch.stack(run.energy)
    s_e = (E[..., :-1].sum(-1) - E[..., -1]).abs() / E[..., -1]
    check(float(s_e.max()) < 1e-12, f"the three band energies sum to the direct total, every basis and convention "
                                    f"({float(s_e.max()):.1e})")
    dl = (hs_[D][0] - hf_[D][0]).double()
    v1 = run.basis["variance"][1][:, 0]
    u = dl @ v1
    e3 = E[3]
    check(abs(float(e3[0, 0, -1]) - float((dl * dl).sum())) < 1e-12 * float((dl * dl).sum())
          and abs(float(e3[0, 1, -1]) - float((dl * dl).sum() - u[0] ** 2)) < 1e-10 * float((dl * dl).sum())
          and abs(float(e3[0, 2, -1]) - float((dl * dl).sum() - (u * u).sum())) < 1e-10 * float((dl * dl).sum()),
          "conventions: every coordinate; minus direction 1 at position 0; minus direction 1 everywhere")
    yv = dl @ run.basis["variance"][1]
    check(abs(float(e3[0, 2, 0]) - float((yv[:, 1] ** 2).sum())) < 1e-10 * float(e3[0, 2, 0])
          and abs(float(e3[0, 1, 0]) - float((yv[:, :2] ** 2).sum() - yv[0, 0] ** 2)) < 1e-10 * float(e3[0, 1, 0]),
          "in the variance basis, dir1 empties direction 1 and pos0 empties its position-0 coordinate only")
    yf = dl @ run.basis["fisher_mmlu"][1]
    check(abs(float(e3[2, 0, 2]) - float((yf[:, 6:] ** 2).sum())) < 1e-10 * float(e3[2, 0, 2]),
          "fisher_mmlu band [7, 16] energy is the deviation's energy on those eigenvectors")
    check(torch.equal(torch.stack(run.mis)[3], torch.tensor([float(u[0] ** 2), float((u * u).sum()),
                                                            float((dl * dl).sum())], dtype=torch.float64)),
          "the mis-scale record: u_0^2, sum u^2, |delta|^2")

    # 12. The ratio statistic on planted data.
    gr = torch.Generator().manual_seed(23)
    Nq = 570
    cnt = pm.boot_counts(Nq)

    def planted(k_top):
        cc = 0.6 + 0.3 * torch.rand(Nq, generator=gr, dtype=torch.float64)
        e = torch.rand(Nq, 3, generator=gr, dtype=torch.float64) * torch.tensor([0.3, 0.3, 0.4], dtype=torch.float64)
        k = torch.tensor([k_top, 1.0, 1.0], dtype=torch.float64)
        cb = cc[:, None] + (1 - cc[:, None]) * k * e + 0.01 * torch.randn(Nq, 3, generator=gr, dtype=torch.float64)
        return pm.boot(torch.stack((cb[:, 0] - cc, cb[:, 2] - cc, e[:, 0], e[:, 2]), 1), give_back_ratio, cnt)

    r_flat, r_low, r_mid = planted(1.0), planted(10.0), planted(2.5)
    check(verdict_ratio(r_flat["lo"], r_flat["hi"]) == "FLAT" and abs(r_flat["point"] - 1) < 0.15,
          f"isotropic sensitivity reads FLAT: {fmt(r_flat)}")
    check(verdict_ratio(r_low["lo"], r_low["hi"]) == "UNEVEN" and abs(r_low["point"] - 10) < 1.5,
          f"sensitivity planted on the top band, 10 times, reads UNEVEN: {fmt(r_low)}")
    check(verdict_ratio(r_mid["lo"], r_mid["hi"]) == "NOT DECIDED", f"2.5 times reads NOT DECIDED: {fmt(r_mid)}")
    # The two-ratio rule on planted data: [1, 64] noisy (unresolved), [1, 512] flat; [1, 512] planted 6 times;
    # [1, 64] planted 10 times with [1, 512] flat; the flat case with [1, 512] holding too little energy.
    r_noisy = dict(point=1.0, lo=-0.4, hi=2.4)
    check(verdict_rule(r_noisy, r_flat, 0.32) == "FLAT" and verdict_rule(r_noisy, r_low, 0.32) == "UNEVEN"
          and verdict_rule(r_low, r_flat, 0.32) == "UNEVEN" and verdict_rule(r_noisy, r_mid, 0.32) == "NOT DECIDED"
          and verdict_rule(r_noisy, r_flat, 0.10) == "NOT DECIDED"
          and verdict_rule(r_noisy, r_flat, math.nan) == "NOT DECIDED",
          "the rule: [1, 512] flat reads FLAT; either ratio planted reads UNEVEN, which outranks a flat [1, 512]; "
          "[1, 512] at 2.5 times, or holding under 0.15 of the energy, reads NOT DECIDED")
    check(verdict_ratio(3.01, 9.0) == "UNEVEN" and verdict_ratio(3.0, 9.0) == "NOT DECIDED"
          and verdict_ratio(0.5, 2.0) == "FLAT" and verdict_ratio(0.49, 1.5) == "NOT DECIDED"
          and verdict_ratio(0.6, 2.01) == "NOT DECIDED" and verdict_ratio(math.nan, 5.0) == "NOT DECIDED",
          "verdict boundaries: above 3 strictly; inside [0.5, 2] inclusive; a non-finite end decides nothing")
    check(abs(float(give_back_ratio(torch.tensor([[0.2, 0.1, 0.05, 0.5]], dtype=torch.float64))[0]) - 20.0) < 1e-12
          and abs(float(cross_term(torch.tensor([[1.0, 2.0, 3.0, 4.0, 10.0]], dtype=torch.float64))[0]) - 0.2) < 1e-12,
          "(0.2 / 0.05) / (0.1 / 0.5) = 20; (1 + 2 + 3 - 4) / 10 = 0.2")
    check(var_control_ok(1.03) and var_control_ok(0.5) and not var_control_ok(2.1) and not var_control_ok(math.nan)
          and var_control_ok(0.3, VAR_CTRL_W) and not var_control_ok(0.2, VAR_CTRL_W),
          "the variance control reads its points inside [0.5, 2] and [0.25, 2]")

    # 13. Disjoint question ids.
    sc = [("a", 0, 11), ("a", 1, 12), ("b", 0, 13)]
    check(disjoint_or_refuse(sc, [("a", 2, 14), ("b", 1, 15)]) == (3, 2), "disjoint Fisher questions pass")
    refused = []
    for other in ([("a", 1, 99)], [("c", 0, 13)]):
        try:
            disjoint_or_refuse(sc, other)
            refused.append(False)
        except SystemExit:
            refused.append(True)
    check(all(refused), "a shared (subject, index) is refused, and so is a shared token hash")
    cand = [dict(subject=s, index=i, qhash=h) for s, i, h in sc]
    cand += [dict(subject="a", index=2, qhash=14), dict(subject="c", index=0, qhash=13),
             dict(subject="b", index=1, qhash=15), dict(subject="d", index=0, qhash=15),
             dict(subject="d", index=1, qhash=16), dict(subject="d", index=2, qhash=17)]
    tk, sk = select_fisher(cand, 3, 4, sc)
    tk_ = [(q["subject"], q["index"], q["qhash"]) for q in tk]
    unfiltered = [(q["subject"], q["index"], q["qhash"]) for q in cand[3:7]]
    try:
        disjoint_or_refuse(sc, unfiltered)
        refuses_raw = False
    except SystemExit:
        refuses_raw = True
    check(len(tk) == 4 and [r for r, *_ in sk] == [4, 6] and [w for *_, w in sk] == ["token hash", "token hash"]
          and tk_ == [("a", 2, 14), ("b", 1, 15), ("d", 1, 16), ("d", 2, 17)]
          and disjoint_or_refuse(sc, tk_) == (3, 4) and refuses_raw,
          "Fisher rows: a scored token hash and a repeat of a taken one are skipped and replaced, the set holds its "
          "count and passes the guard; the same rows unfiltered are refused")

    # 14. End to end: analyse writes every file; gates green on self-consistent dumps.
    keys = [("tiny", i) for i in range(len(prompts))]
    answers = [i % 4 for i in range(len(prompts))]
    f16l = torch.stack(run.out["fp16"])
    ctll = torch.stack(run.out["control"])
    df = {kk: {"pick": cm.pick_of(f16l[i].tolist()), "answer": answers[i], "logits": f16l[i].tolist()}
          for i, kk in enumerate(keys)}
    ds = {kk: {"pick": cm.pick_of(ctll[i].tolist()), "answer": answers[i], "logits": ctll[i].tolist()}
          for i, kk in enumerate(keys)}
    with tempfile.TemporaryDirectory() as dd:
        res = analyse(Path(dd), run, keys, answers, df, ds, fisher_keys=(len(prompts), len(fprompts)))
        files = {x.name for x in Path(dd).iterdir()}
        back = cm.read_raw(Path(dd) / "energies.f64")
        nk = len((Path(dd) / "kills.tsv").read_text().splitlines()) - 1
    want = {"report.txt", "arms.tsv", "kills.tsv", "logits.tsv", "picks.tsv", "checks.tsv", "spectra.tsv",
            "energies.f64", "misscale.f64", "eigvals_variance.f64", "eigvals_fisher_text.f64",
            "eigvals_fisher_mmlu.f64", "fisher_text.f64", "fisher_mmlu.f64"}
    check(want <= files and nk == 24, f"analyse writes its {len(want)} files and 24 ratio rows")
    check(res["green"], "every gate green on the tiny run with self-consistent dumps: "
          + (", ".join(kk for kk, v in res["gates"].items() if not v) or "all pass"))
    check(torch.equal(back, E), "the energies read back bit for bit")
    sh = res["arms"]
    check(abs(sh["fp16"]["s"]["point"] - 1) < 1e-12 and abs(sh["control"]["s"]["point"]) < 1e-12
          and set(sh) == set(run.arm_names), "shape share: FP16 1, the control 0, every arm read")
    rb = res["bases"]["fisher_text"]
    es = [float(E[:, 1, 2, j].sum() / E[:, 1, 2, -1].sum()) for j in range(3)]
    check(abs(rb["ratios"]["dir1"]["point"] - (rb["s"][0]["point"] / es[0]) / (rb["s"][2]["point"] / es[2])) < 1e-9,
          "the ratio's point is (s_top / E_top) / (s_bot / E_bot) on the shares and energy shares as printed")

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
    torch.set_float32_matmul_precision("highest")
    if torch.backends.cuda.matmul.allow_tf32:
        raise SystemExit("refused: TF32 matmul is on; the gradient pass is float32")
    device = torch.device("cuda:0")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    timeline = []

    def stamp(label):
        timeline.append((label, time.time() - t_start))
        print(f"== {time.time() - t_start:7.1f} s  {label}", flush=True)

    ref = pm.read_ref(Path(args.ref_dir))
    pm_ref = read_pm_ref(Path(args.pm_ref_dir))
    dump_f = cm.read_dump(Path(args.dump_f16), cm.DUMP_F16[1])
    dump_s = cm.read_dump(Path(args.dump_sealed), cm.DUMP_SEALED[1])
    qs, ans_ids = cm.fetch_prompts(args.cache)
    bad = [(q["subject"], q["index"]) for q in qs for dmp in (dump_f, dump_s)
           if dmp[(q["subject"], q["index"])]["qhash"] != q["qhash"]]
    if bad:
        raise SystemExit(f"refused: {len(bad)} prompts differ from the dumps' token ids, first {bad[:3]}")
    rr = cm.round_robin(qs)
    order = rr[:N_Q]
    sc_ = [(q["subject"], q["index"], q["qhash"]) for q in order]
    forder, fskip = select_fisher(rr, FISHER_START, N_Q, sc_)
    if [(q["subject"], q["index"], q["qhash"]) for q in order] != ref["questions"]:
        raise SystemExit("refused: these are not contrast-map's 570 questions")
    fk = disjoint_or_refuse([(q["subject"], q["index"], q["qhash"]) for q in order],
                            [(q["subject"], q["index"], q["qhash"]) for q in forder])
    fdesc = (f"round robin from row {FISHER_START}, {len(fskip)} rows skipped"
             + (": " + "; ".join(f"row {r} {s}/{i} ({why})" for r, s, i, why in fskip) if fskip else ""))
    print(f"Fisher rows: {fdesc}", flush=True)
    print(f"questions: scored {fk[0]} (contrast-map's), Fisher {fk[1]} ({fdesc}), "
          f"disjoint; Fisher prompt lengths {min(len(q['ids']) for q in forder)}..{max(len(q['ids']) for q in forder)}")
    toks = lvo.fetch_tokens(args.cache)
    windows = lvo.cut_windows(toks, TEXT_CTX)[:N_TEXT]
    if len(windows) != N_TEXT:
        raise SystemExit(f"refused: {len(windows)} windows of {TEXT_CTX}, {N_TEXT} wanted")
    wfp = lvo.fnv_fingerprint([i for w in windows for i in w])
    print(f"text: WikiText-2 test, the first {N_TEXT} windows of {TEXT_CTX} tokens, fingerprint {wfp:016x}")
    stamp("inputs checked")
    cfg, sets, extra = lvo.load_sets(Path(args.export), Path(args.stage), args.cache, device)
    h0 = {k: cm.sha_set(v) for k, v in sets.items()}
    stamp(f"loaded, bases fp16 {h0['fp16'][:16]} sealed {h0['sealed'][:16]}")
    sr = SensRun(cfg, sets, device, ans_ids)
    keys = [(q["subject"], q["index"]) for q in order]
    answers = [q["answer"] for q in order]
    secs = {}

    def loop(label, items, fn, cap, every=57):
        t = time.time()
        for i, it in enumerate(items):
            fn(i, it)
            if i + 1 == min(every, len(items)):
                torch.cuda.synchronize()
                proj = (time.time() - t) * len(items) / (i + 1)
                print(f"{label}: {i + 1} in {time.time() - t:.1f} s, projected {proj:.0f} s (cap {cap:.0f})", flush=True)
                if proj > cap:
                    return False
        torch.cuda.synchronize()
        secs[label] = time.time() - t
        stamp(f"{label}: {len(items)} in {secs[label]:.0f} s, peak memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
        return True

    if not loop("pass 1", [q["ids"] for q in order], lambda i, ids: sr.pass1(ids), CAP["pass1"]):
        print("refused: pass 1 would exceed its cap; nothing is read", flush=True)
        return 5
    gen = torch.Generator(device=device).manual_seed(LABEL_SEED)
    if not loop("text", windows, lambda i, w: sr.text_window(i, w, gen, check=i < CHECK_GRAD), CAP["text"], every=32):
        print("refused: the text pass would exceed its cap; nothing is read", flush=True)
        return 5
    if not loop("fisher", [q["ids"] for q in forder],
                lambda i, ids: sr.fisher_question(i, ids, check=i < CHECK_GRAD), CAP["fisher"]):
        print("refused: the Fisher pass would exceed its cap; nothing is read", flush=True)
        return 5
    sr.set_bases()
    stamp("bases: variance, fisher_text, fisher_mmlu")
    t2 = time.time()
    marks = {}
    for i, q in enumerate(order):
        sr.pass2(i, q["ids"], check=i < CHECK_Q)
        if (i + 1) % 57 == 0:
            torch.cuda.synchronize()
            marks[i + 1] = time.time() - t2
            print(f"  pass 2: {i + 1} prompts, {marks[i + 1]:.0f} s", flush=True)
            if i + 1 == 114:
                proj = sum(secs.values()) + marks[114] + (marks[114] - marks[57]) / 57 * (N_Q - 114)
                print(f"pass 2: projected every loop together {proj:.0f} s (cap {LOOP_CAP:.0f})", flush=True)
                if proj > LOOP_CAP:
                    print("refused: the loops would exceed their cap; nothing is read", flush=True)
                    return 5
    torch.cuda.synchronize()
    secs["pass 2"] = time.time() - t2
    stamp(f"pass 2: {N_Q} prompts, {secs['pass 2']:.0f} s, {len(sr.checks)} checks, peak memory "
          f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    h1 = {k: cm.sha_set(v) for k, v in sets.items()}
    same = h0 == h1
    stamp(f"bases unchanged by sha256: {same}")
    head = [f"inputs: checkpoint {lvo.CKPT_REPO}@{lvo.CKPT_REV}; export sha256 {lvo.EXPORT_SHA}; "
            f"{cm.MMLU_REPO}@{cm.MMLU_REV}; {lvo.WIKI_REPO}@{lvo.WIKI_REV}; answer ids {ans_ids}",
            f"scored {N_Q} questions, {sr.tokens} tokens; Fisher questions {fdesc}; "
            f"text {N_TEXT} windows of {TEXT_CTX}, fingerprint {wfp:016x}, labels sampled with seed {LABEL_SEED}",
            "seconds: " + ", ".join(f"{k} {v:.0f}" for k, v in secs.items()),
            f"bases unchanged by sha256: {same}; non-projection tensors identical in W and W_hat: "
            f"{len(extra['same_other'])} of {len(extra['others'])}", ""]
    res = analyse(out, sr, keys, answers, dump_f, dump_s, ref=ref, pm_ref=pm_ref, extra_lines=head, fisher_keys=fk)
    stamp("analysis written")
    (out / "timeline.txt").write_text("".join(f"{s:.1f}\t{lab}\n" for lab, s in timeline))
    (out / "inputs.txt").write_text(
        f"checkpoint {lvo.CKPT_REPO}@{lvo.CKPT_REV}\nexport {args.export} sha256 {lvo.EXPORT_SHA}\n"
        f"mmlu {cm.MMLU_REPO}@{cm.MMLU_REV}\nwikitext {lvo.WIKI_REPO}@{lvo.WIKI_REV} {lvo.WIKI_FILE}\n"
        f"dumps {cm.DUMP_F16[1]} {cm.DUMP_SEALED[1]}\nref {json.dumps(pm.REF_FILES)}\npm_ref {json.dumps(PM_REF_FILES)}\n"
        f"bases sha256 before {h0}\nbases sha256 after  {h1}\ngates {json.dumps(res['gates'])}\n"
        f"verdict {res['verdict']}\ntotal {time.time() - t_start:.0f} s\n")
    if not same:
        return 3
    g = res["gates"]
    if not (g["self-patch bit for bit"] and g["hybrid bit for bit"] and g["FP16 pass 2 = pass 1"]
            and g["gradient forward bit for bit"]):
        return 3
    return 0 if res["green"] else 4


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--selftest", action="store_true")
    mode.add_argument("--run", action="store_true")
    ap.add_argument("--export", help="the sealed export's model.safetensors (bucket mount)")
    ap.add_argument("--stage", default="/tmp/sm-export", help="local copy of the export")
    ap.add_argument("--out", help="output folder")
    ap.add_argument("--dump-f16")
    ap.add_argument("--dump-sealed")
    ap.add_argument("--ref-dir", help=f"contrast-map-4b's raw files ({pm.REF_DIR})")
    ap.add_argument("--pm-ref-dir", help=f"patch-map-4b's raw files ({PM_REF_DIR})")
    ap.add_argument("--cache", default=os.environ.get("HF_HUB_CACHE"))
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not (a.export and a.out and a.dump_f16 and a.dump_sealed and a.ref_dir and a.pm_ref_dir):
        ap.error("--run needs --export, --out, --dump-f16, --dump-sealed, --ref-dir and --pm-ref-dir")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
