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
"""Where the contrast of the four MMLU answer logits goes, FP16 against the sealed 4B, side by side.

Prereg: proofs/preregistration-contrast-map-4b-2026-11-14.md. Question:
docs/mesures/recherche-qualite2-r0-contraste-2026-11-14.txt (slope 0.5817 of the sealed 4B's answer logits on
FP16's).

One process, both weight sets resident on the card, loaded by ops/layer_vs_output.py (imported, never edited):
the FP16 checkpoint (W) and the f16 export of the sealed file (W_hat). Each model reads the same N MMLU prompts on
its own activations. The forward here is layer_vs_output.forward with taps added; the selftest requires the two to
give equal logits, bit for bit.

The prompt is the Rust harness's (llvq-llm/src/bin/mmlu.rs, `block` lines 313-323, prefix lines 765-774, encode
lines 780-785, answer tokens lines 651-664, last row lines 806-808, pick lines 818-820). The port is proved on the
token ids: every prompt's FNV-1a fingerprint must equal the `qhash` column of the two dumps, and the 14,042 prompts
in census order must give the run fingerprint the dumps' trailer carries.

Readings, every accumulator float64 (the per-prompt GEMMs of R4 run in float32 as Note1 does on the card):
  control  rule D: own-head answer logits of each model against its Rust dump; decides whether anything is read.
  R1       lens slope by depth, one fixed readout (FP16 final norm weight, FP16 answer rows).
  R2       stream slope by variance band of FP16's uncentred second moment, at 10 depths.
  R3       share of the sealed stream's energy a linear map of the FP16 stream explains, against the diagonal share.
  R4       coherent gain, noise ratio and relative error of each of the 252 matrices, and per output row.
  R5       attention entropy and top weight of the answer position, per layer and head.

Modes:
  --selftest       tiny random model on CPU, float64, planted distortions; no download.
  --check-prompts  the prompt port against the dumps' qhash and trailer (tokenizer and parquet only, no model).
  --run            the measurement. Card only.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layer_vs_output as lvo  # noqa: E402

# ---------------------------------------------------------------------------
# Pinned inputs.
# ---------------------------------------------------------------------------
# cais/mmlu: the harness reads `main` (corpus.rs `dataset_revision`) and its dumps record no revision. This commit is
# the one in the local cache on 2026-11-14; --check-prompts proves its 14,042 prompts give the dumps' qhash and run
# fingerprint, token for token.
MMLU_REPO = "cais/mmlu"
MMLU_REV = "c30699e8356da336a370243923dbaf21066bb9fe"
MMLU_FILES = {
    "test": ("all/test-00000-of-00001.parquet", "74a41822ce7d3def56e1682f958469c04642a5336a5ce912fa375fdb90fb25d7"),
    "dev": ("all/dev-00000-of-00001.parquet", "2b19bde1ed8ca6b482fb283abc90e8e0d9d228947029c0b91795d64b28b3bc3f"),
}
N_TEST = 14_042
N_SUBJECTS = 57
RUN_FP = 0xA74A6D6213602979  # `# end fingerprint=` of both census dumps below
DUMP_F16 = ("docs/data/mmlu-dumps/mmlu-4b-f16-FULL.csv",
            "f022add79f69baeb59a751e6b84fa6e9ea0d5929851c9477e6adf1ee0af5e6bf")
DUMP_SEALED = ("docs/data/mmlu-dumps/mmlu-4b-abl-control-FULL.csv",
               "e2abc978add4e189d346f43d64638e9d16b778a7f3359fbb8c1fb53e5c1cfd74")

DEPTHS = 37  # h(0) the embedding output .. h(36) after the last block, before the final norm
R2_DEPTHS = (0, 4, 8, 12, 16, 20, 24, 28, 32, 36)
RAW_DEPTHS = (12, 24, 36)
BANDS = ((1, 1), (2, 8), (9, 64), (65, 512), (513, None))  # 1-based, inclusive; None = d
RIDGE_REL = 1e-6
N_MIN, N_MAX = 570, 2280
PICK_AGREE_MIN = 0.99
SLOPE_TOL = 0.02
LETTERS = ("A", "B", "C", "D")

# Rust `char::is_whitespace` (Unicode White_Space), what `str::trim` strips. Python's str.strip() also strips
# U+001C..U+001F, which Rust keeps, so the set is spelled out.
RUST_WS = ("\u0009\u000a\u000b\u000c\u000d \u0085         "
           "        　")


# ---------------------------------------------------------------------------
# The prompt, ported from llvq-llm/src/bin/mmlu.rs.
# ---------------------------------------------------------------------------
def rust_trim(s: str) -> str:
    return s.strip(RUST_WS)


def pretty(subject: str) -> str:
    """mmlu.rs `pretty`, line 115."""
    return subject.replace("_", " ")


def block(item: dict, answer: int | None) -> str:
    """mmlu.rs `block`, lines 313-323."""
    s = f"{rust_trim(item['question'])}\n"
    for i, c in enumerate(item["choices"]):
        s += f"{LETTERS[i]}. {rust_trim(c)}\n"
    s += "Answer:"
    if answer is not None:
        s += f" {LETTERS[answer]}\n\n"
    return s


def prefix_for(subject: str, shots: list[dict]) -> str:
    """mmlu.rs lines 765-774: the header, then every dev example of the subject in parquet order."""
    s = f"The following are multiple choice questions (with answers) about {pretty(subject)}.\n\n"
    for ex in shots:
        s += block(ex, ex["answer"])
    return s


def pick_of(options) -> int:
    """mmlu.rs lines 818-820: `max_by(total_cmp)`, which returns the LAST of equal maxima."""
    best = 0
    for i in range(1, 4):
        if float(options[i]) >= float(options[best]):
            best = i
    return best


def answer_pos(length: int) -> int:
    """The position whose logits the harness reads: the last one (mmlu.rs lines 806-808)."""
    return length - 1


def read_split(path: Path) -> list[dict]:
    """corpus.rs `mmlu_split`: columns read by name, four options, answer as an integer or a letter."""
    import pyarrow.parquet as pq

    t = pq.read_table(path)
    for c in ("question", "subject", "choices", "answer"):
        if c not in t.column_names:
            raise SystemExit(f"refused: {path} has no `{c}` column")
    out = []
    for q, s, ch, a in zip(*(t.column(c).to_pylist() for c in ("question", "subject", "choices", "answer"))):
        if isinstance(a, str):
            a = "ABCD".find(a.strip())
        ch = [x for x in ch if isinstance(x, str)]
        if len(ch) != 4 or not (0 <= int(a) < 4):
            raise SystemExit(f"refused: {path}: a row with {len(ch)} options or answer {a}")
        out.append({"subject": s, "question": q, "choices": ch, "answer": int(a)})
    return out


def build_questions(test: list[dict], dev: list[dict]) -> list[dict]:
    """Every test question in census order (subjects in byte order, parquet order inside), with its prompt."""
    shots: dict[str, list[dict]] = {}
    for it in dev:
        shots.setdefault(it["subject"], []).append(it)
    by_subject: dict[str, list[dict]] = {}
    for it in test:
        by_subject.setdefault(it["subject"], []).append(it)
    out = []
    for subject in sorted(by_subject):
        pre = prefix_for(subject, shots.get(subject, []))
        items = by_subject[subject]
        for index, it in enumerate(items):
            out.append({"subject": subject, "index": index, "population": len(items), "answer": it["answer"],
                        "prompt": pre + block(it, None)})
    return out


def round_robin(questions: list[dict]) -> list[dict]:
    """Subjects in turn, question index ascending inside each: every prefix is balanced over the subjects."""
    by_subject: dict[str, list[dict]] = {}
    for q in questions:
        by_subject.setdefault(q["subject"], []).append(q)
    subjects = sorted(by_subject)
    for s in subjects:
        by_subject[s].sort(key=lambda q: q["index"])
    deepest = max(len(v) for v in by_subject.values())
    return [by_subject[s][i] for i in range(deepest) for s in subjects if i < len(by_subject[s])]


def n_from_time(t_first: float, n_first: int, budget: float) -> int:
    fit = int(budget * n_first / t_first) if t_first > 0 else N_MAX
    return min(N_MAX, max(N_MIN, fit))


def fetch_prompts(cache: str | None):
    """Tokenizer and both splits at their pins; every prompt tokenized; the run fingerprint checked."""
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    tok_path = Path(hf_hub_download(lvo.CKPT_REPO, "tokenizer.json", revision=lvo.CKPT_REV, cache_dir=cache))
    got = lvo.sha256_file(tok_path)
    if got != lvo.TOKENIZER_SHA:
        raise SystemExit(f"refused: tokenizer sha256 {got}, expected {lvo.TOKENIZER_SHA}")
    splits = {}
    for split, (fname, want) in MMLU_FILES.items():
        p = Path(hf_hub_download(MMLU_REPO, fname, repo_type="dataset", revision=MMLU_REV, cache_dir=cache))
        got = lvo.sha256_file(p)
        if got != want:
            raise SystemExit(f"refused: {MMLU_REPO}@{MMLU_REV} {fname} sha256 {got}, expected {want}")
        splits[split] = read_split(p)
    tok = Tokenizer.from_file(str(tok_path))
    qs = build_questions(splits["test"], splits["dev"])
    t0 = time.time()
    encs = tok.encode_batch([q["prompt"] for q in qs], add_special_tokens=False)
    for q, e in zip(qs, encs):
        q["ids"] = e.ids
        q["qhash"] = lvo.fnv_fingerprint(e.ids)
    run_fp = lvo.fnv_fingerprint([i for q in qs for i in q["ids"]])
    print(f"prompts: {len(qs)} questions, {len({q['subject'] for q in qs})} subjects, {len(splits['dev'])} dev "
          f"examples, tokenized in {time.time() - t0:.1f} s; run fingerprint {run_fp:016x} (dumps {RUN_FP:016x})")
    if len(qs) != N_TEST or run_fp != RUN_FP:
        raise SystemExit("refused: the prompts are not the token stream the Rust harness scored")
    ans_ids = []
    for letter in LETTERS:
        ids = tok.encode(f" {letter}", add_special_tokens=False).ids
        if len(ids) != 1:
            raise SystemExit(f"refused: ' {letter}' tokenizes to {ids}, not one token (mmlu.rs lines 651-664)")
        ans_ids.append(ids[0])
    return qs, ans_ids


def read_dump(path: Path, want_sha: str) -> dict:
    """A census dump of `bin/mmlu`: header, columns by name, trailer. Returns {(subject, index): row}."""
    got = lvo.sha256_file(path)
    if got != want_sha:
        raise SystemExit(f"refused: {path} sha256 {got}, expected {want_sha}")
    lines = path.read_text().splitlines()
    if lines[0] != "# llvq-mmlu-dump v1" or "# dtype=f16" not in lines or "# limit=census" not in lines:
        raise SystemExit(f"refused: {path} is not an f16 census dump v1")
    if lines[-1] != f"# end fingerprint={RUN_FP:016x} questions={N_TEST}":
        raise SystemExit(f"refused: {path} trailer {lines[-1]!r}")
    cols = next(ln for ln in lines if not ln.startswith("#")).split(",")
    at = {c: cols.index(c) for c in ("subject", "index", "qhash", "answer", "pick",
                                     "logit_a", "logit_b", "logit_c", "logit_d")}
    rows = {}
    for ln in lines:
        if ln.startswith("#") or ln.startswith("subject,"):
            continue
        r = ln.split(",")
        rows[(r[at["subject"]], int(r[at["index"]]))] = {
            "qhash": int(r[at["qhash"]], 16), "answer": int(r[at["answer"]]), "pick": int(r[at["pick"]]),
            "logits": [float(r[at[k]]) for k in ("logit_a", "logit_b", "logit_c", "logit_d")]}
    if len(rows) != N_TEST:
        raise SystemExit(f"refused: {path} holds {len(rows)} questions, {N_TEST} expected")
    return rows


# ---------------------------------------------------------------------------
# The forward with taps. Same operations, same order as layer_vs_output.forward (checked bit for bit).
# ---------------------------------------------------------------------------
def forward_taps(cfg, get, ids: torch.Tensor, proj_hook=None, stream_hook=None, attn_hook=None) -> torch.Tensor:
    """Logits (L, V). stream_hook(l, h) sees h(l), l = 0..layers; attn_hook(b, q, k, v, a) sees the post-RoPE
    q, the repeated k and v, and the attention output, all (1, heads, L, head_dim)."""
    emb = get("model.embed_tokens.weight")
    dtype = emb.dtype
    length = ids.shape[0]
    h = emb[ids].unsqueeze(0)
    cos, sin = lvo.rope_tables(cfg, length, ids.device, dtype)
    if stream_hook is not None:
        stream_hook(0, h)

    def lin(name: str, x: torch.Tensor) -> torch.Tensor:
        if proj_hook is not None:
            proj_hook(name, x)
        return F.linear(x, get(name))

    for b in range(cfg.layers):
        p = f"model.layers.{b}."
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
        a = a.transpose(1, 2).reshape(1, length, cfg.heads * cfg.head_dim)
        h = h + lin(p + "self_attn.o_proj.weight", a)
        x = lvo.rms_norm(h, get(p + "post_attention_layernorm.weight"), cfg.eps)
        g = lin(p + "mlp.gate_proj.weight", x)
        u = lin(p + "mlp.up_proj.weight", x)
        h = h + lin(p + "mlp.down_proj.weight", F.silu(g) * u)
        if stream_hook is not None:
            stream_hook(b + 1, h)
    h = lvo.rms_norm(h, get("model.norm.weight"), cfg.eps)
    head = emb if cfg.tied else get("lm_head.weight")
    return F.linear(h, head)[0]


def last_attention(q: torch.Tensor, k: torch.Tensor, pos: int, scale: float) -> torch.Tensor:
    """Attention weights (heads, pos + 1) of position `pos` over the prompt, float64, causal."""
    qd = q[0, :, pos, :].double()
    kd = k[0, :, :pos + 1, :].double()
    return torch.softmax(torch.einsum("hd,hld->hl", qd, kd) * scale, dim=-1)


# ---------------------------------------------------------------------------
# Accumulators.
# ---------------------------------------------------------------------------
class Moments:
    """R2, all positions: C_ff = sum h_f h_f^T, C_sf = sum h_s h_f^T, C_ss = sum h_s h_s^T, float64."""

    def __init__(self, depths, d: int, device):
        z = lambda: torch.zeros((d, d), dtype=torch.float64, device=device)  # noqa: E731
        self.ff = {l: z() for l in depths}
        self.sf = {l: z() for l in depths}
        self.ss = {l: z() for l in depths}
        self.tokens = 0

    def add(self, l: int, hf: torch.Tensor, hs: torch.Tensor) -> None:
        x = hf.reshape(-1, hf.shape[-1]).double()
        y = hs.reshape(-1, hs.shape[-1]).double()
        self.ff[l] += x.T @ x
        self.sf[l] += y.T @ x
        self.ss[l] += y.T @ y


class Gain:
    """R4: per output row, S_yy = sum (W x)_i^2, S_ey = sum ((W_hat - W) x)_i (W x)_i, S_ee = sum ((W_hat - W) x)_i^2.

    The products run in `acc` (float32 on the card, as Note1), each prompt's column sums are taken in float64 and
    added to float64 accumulators.
    """

    def __init__(self, fp16: dict, sealed: dict, names: list[str], acc_dtype):
        self.fp16, self.sealed, self.acc = fp16, sealed, acc_dtype
        dev = next(iter(fp16.values())).device
        self.syy = {n: torch.zeros(fp16[n].shape[0], dtype=torch.float64, device=dev) for n in names}
        self.sey = {n: torch.zeros_like(self.syy[n]) for n in names}
        self.see = {n: torch.zeros_like(self.syy[n]) for n in names}

    def __call__(self, name: str, x: torch.Tensor) -> None:
        if name not in self.syy:
            return
        xa = x.reshape(-1, x.shape[-1]).to(self.acc)
        w = self.fp16[name].to(self.acc)
        w_hat = self.sealed[name].to(self.acc)
        y = xa @ w.T
        e = xa @ (w_hat - w).T
        self.syy[name] += (y * y).sum(0, dtype=torch.float64)
        self.sey[name] += (e * y).sum(0, dtype=torch.float64)
        self.see[name] += (e * e).sum(0, dtype=torch.float64)

    def totals(self, name: str) -> tuple[float, float, float]:
        return float(self.syy[name].sum()), float(self.sey[name].sum()), float(self.see[name].sum())


def gain_numbers(syy: float, sey: float, see: float) -> tuple[float, float, float]:
    """(coherent gain a, noise ratio, relative error)."""
    return 1.0 + sey / syy, (see - sey * sey / syy) / syy, see / syy


class Attn:
    """R5: sums over questions of the entropy and the top weight of the answer position, per layer and head."""

    def __init__(self, layers: int, heads: int, device):
        self.ent = torch.zeros((layers, heads), dtype=torch.float64, device=device)
        self.top = torch.zeros((layers, heads), dtype=torch.float64, device=device)
        self.n = 0

    def hook(self, pos: int, scale: float):
        def f(b, q, k, v, a):
            p = last_attention(q, k, pos, scale)
            ent = -(p * torch.log(p.clamp_min(1e-300))).sum(-1)
            self.ent[b] += ent
            self.top[b] += p.max(-1).values
        return f


class Run:
    """Everything the questions leave behind, for `analyse`."""

    def __init__(self, cfg, sets, names, acc_dtype, device):
        self.cfg = cfg
        self.keys, self.answers, self.lengths, self.qhash = [], [], [], []
        self.logits = {"fp16": [], "sealed": []}
        self.ans = {"fp16": [], "sealed": []}
        self.moments = Moments(R2_DEPTHS, cfg.hidden, device)
        self.gain = {m: Gain(sets["fp16"], sets["sealed"], names, acc_dtype) for m in ("fp16", "sealed")}
        self.attn = {m: Attn(cfg.layers, cfg.heads, device) for m in ("fp16", "sealed")}


def run_question(run: Run, sets, ids_list, ans_ids, device, extra_hooks=None) -> None:
    """Both models on one prompt, each on its own activations; every accumulator updated."""
    cfg = run.cfg
    ids = torch.tensor(ids_list, dtype=torch.long, device=device)
    pos = answer_pos(ids.shape[0])
    scale = cfg.head_dim ** -0.5
    keep: dict[int, torch.Tensor] = {}
    for model in ("fp16", "sealed"):
        get = lvo.Arm(model, model, frozenset()).resolver(sets)
        ans = torch.empty((DEPTHS, cfg.hidden), dtype=sets[model]["model.embed_tokens.weight"].dtype, device=device)

        def stream(l, h, model=model, ans=ans):
            ans[l] = h[0, pos]
            if l in run.moments.ff:
                if model == "fp16":
                    keep[l] = h[0]
                else:
                    run.moments.add(l, keep.pop(l), h[0])

        hooks = [run.gain[model]] + ([extra_hooks[model]] if extra_hooks else [])

        def proj(name, x, hooks=hooks):
            for hk in hooks:
                hk(name, x)

        with torch.no_grad():
            z = forward_taps(cfg, get, ids, proj_hook=proj, stream_hook=stream,
                             attn_hook=run.attn[model].hook(pos, scale))
            run.logits[model].append(z[pos, ans_ids].float().cpu())
            run.ans[model].append(ans.cpu())
        run.attn[model].n += 1
        del z
    run.moments.tokens += len(ids_list)


# ---------------------------------------------------------------------------
# Estimators.
# ---------------------------------------------------------------------------
def fit(x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    """ops/logit_snr.fidelity: the four values of a question centred, pooled, y = beta x + e.
    Returns (beta, noise), noise = sqrt(sum e^2 / 3N), the sd of e a logit before centring."""
    x = x.double()
    y = y.double()
    xc = (x - x.mean(1, keepdim=True)).reshape(-1)
    yc = (y - y.mean(1, keepdim=True)).reshape(-1)
    mx, my = xc.mean(), yc.mean()
    beta = float(((xc - mx) * (yc - my)).sum() / ((xc - mx) ** 2).sum())
    e = yc - my - beta * (xc - mx)
    return beta, math.sqrt(float((e * e).sum()) / (3 * x.shape[0]))


def lens(h: torch.Tensor, norm_w: torch.Tensor, rows: torch.Tensor, eps: float) -> torch.Tensor:
    """z = U_ans . final_norm(h), float64; h (N, d), rows (4, d)."""
    hd = h.double()
    hn = hd * torch.rsqrt(hd.pow(2).mean(-1, keepdim=True) + eps) * norm_w.double()
    return hn @ rows.double().T


def band_slice(lo: int, hi: int | None, d: int) -> slice:
    return slice(lo - 1, d if hi is None else hi)


def proj_diag(v: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """diag(V^T C V)."""
    return ((c @ v) * v).sum(0)


def eig_desc(c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    lam, v = torch.linalg.eigh(c)
    return lam.flip(0), v.flip(1)


def band_table(v: torch.Tensor, lam: torch.Tensor | None, cff, csf, css, bands=BANDS) -> dict:
    """R2 on one set of moments, projected on the FP16 all-position eigenbasis V.

    lam None: weights are diag(V^T C_ff V), the energy of these moments on V (the answer-position case).
    b_k = (V^T C_sf V)_kk / lam_k; n_k = ((V^T C_ss V)_kk - b_k^2 lam_k) / lam_k. Directions with lam_k <= 1e-12
    lam_max carry no slope and are left out of every band statistic.
    """
    d = cff.shape[0]
    lam = proj_diag(v, cff) if lam is None else lam
    s = proj_diag(v, csf)
    q = proj_diag(v, css)
    ok = lam > 1e-12 * float(lam.max())
    b = torch.where(ok, s / torch.where(ok, lam, 1.0), torch.nan)
    n = torch.where(ok, (q - b * b * lam) / torch.where(ok, lam, 1.0), torch.nan)
    tr = float(torch.trace(cff))
    rows = []
    for lo, hi in bands:
        sl = band_slice(lo, hi, d)
        m = ok[sl]
        if int(m.sum()) == 0:
            rows.append(dict(band=f"{lo}-{hi or d}", n=0, slope_w=math.nan, slope_med=math.nan, noise_w=math.nan,
                             share=math.nan))
            continue
        lb, bb, nb = lam[sl][m], b[sl][m], n[sl][m]
        rows.append(dict(band=f"{lo}-{hi or d}", n=int(m.sum()), slope_w=float((lb * bb).sum() / lb.sum()),
                         slope_med=float(bb.median()), noise_w=float((lb * nb).sum() / lb.sum()),
                         share=float(lam[sl].sum()) / tr))
    return dict(rows=rows, b=b, n=n, lam=lam, pooled=float(torch.trace(csf)) / tr,
                norm_ratio=math.sqrt(float(torch.trace(css)) / tr))


def linear_table(v: torch.Tensor, lam: torch.Tensor, cff, csf, css, bands=BANDS, ridge_rel=RIDGE_REL) -> dict:
    """R3. B = C_sf (C_ff + r I)^-1, r = ridge_rel tr(C_ff) / d, the least-squares map of h_f onto h_s.
    Explained energy B C_ff B^T; per band of FP16 directions v_k:
      e_lin = sum_k v_k^T B C_ff B^T v_k / sum_k v_k^T C_ss v_k,
      diag  = sum_k b_k^2 lam_k / sum_k v_k^T C_ss v_k, b_k = v_k^T C_sf v_k / lam_k."""
    d = cff.shape[0]
    r = ridge_rel * float(torch.trace(cff)) / d
    bt = torch.linalg.solve(cff + r * torch.eye(d, dtype=cff.dtype), csf.T)  # (C_ff + r I)^-1 C_fs = B^T
    e = proj_diag(v, bt.T @ cff @ bt)
    q = proj_diag(v, css)
    s = proj_diag(v, csf)
    ok = lam > 1e-12 * float(lam.max())
    dg = torch.where(ok, s * s / torch.where(ok, lam, 1.0), 0.0)
    rows = []
    for lo, hi in bands:
        sl = band_slice(lo, hi, d)
        qs = float(q[sl].sum()) if sl.start < d else 0.0
        if sl.start >= d or qs <= 0:
            rows.append(dict(band=f"{lo}-{hi or d}", e_lin=math.nan, diag=math.nan, ridge_over_lam_min=math.nan))
            continue
        lmin = float(lam[sl][ok[sl]].min()) if bool(ok[sl].any()) else math.nan
        rows.append(dict(band=f"{lo}-{hi or d}", e_lin=float(e[sl].sum()) / qs, diag=float(dg[sl].sum()) / qs,
                         ridge_over_lam_min=r / lmin))
    tot = float(q.sum())
    return dict(rows=rows, ridge=r, e_lin_all=float(e.sum()) / tot, diag_all=float(dg.sum()) / tot)


def control(keys, lf, ls, dump_f: dict, dump_s: dict) -> dict:
    """Rule D: picks, logit differences and the slope fit, script against dumps, on the same questions."""
    df = torch.tensor([dump_f[k]["logits"] for k in keys], dtype=torch.float64)
    ds = torch.tensor([dump_s[k]["logits"] for k in keys], dtype=torch.float64)
    out = {}
    for name, mine, ref, dump in (("fp16", lf, df, dump_f), ("sealed", ls, ds, dump_s)):
        picks = [pick_of(r) for r in mine.tolist()]
        same = sum(p == dump[k]["pick"] for p, k in zip(picks, keys)) / len(keys)
        diff = (mine.double() - ref).abs().reshape(-1)
        acc_me = sum(p == dump[k]["answer"] for p, k in zip(picks, keys)) / len(keys)
        acc_dump = sum(dump[k]["pick"] == dump[k]["answer"] for k in keys) / len(keys)
        out[name] = dict(picks_equal=same, max_abs=float(diff.max()), median_abs=float(diff.median()),
                         acc_script=acc_me, acc_dump=acc_dump)
    out["fit_script"] = fit(lf, ls)
    out["fit_dump"] = fit(df, ds)
    out["green"] = (out["fp16"]["picks_equal"] >= PICK_AGREE_MIN and out["sealed"]["picks_equal"] >= PICK_AGREE_MIN
                    and abs(out["fit_script"][0] - out["fit_dump"][0]) <= SLOPE_TOL)
    return out


def write_raw(path: Path, t: torch.Tensor, keys) -> None:
    """Raw little-endian row-major bytes and a JSON sidecar."""
    t = t.detach().contiguous().cpu()
    path.write_bytes(ctypes.string_at(t.data_ptr(), t.numel() * t.element_size()))
    Path(str(path) + ".json").write_text(json.dumps(
        {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", ""), "order": "C, little endian",
         "keys": [f"{s}/{i}" for s, i in keys]}) + "\n")


def read_raw(path: Path) -> torch.Tensor:
    meta = json.loads(Path(str(path) + ".json").read_text())
    dt = getattr(torch, meta["dtype"])
    return torch.frombuffer(bytearray(path.read_bytes()), dtype=dt).reshape(meta["shape"])


def sha_set(d: dict[str, torch.Tensor]) -> str:
    """sha256 over every tensor's name, dtype and bytes, without numpy."""
    h = hashlib.sha256()
    for k in sorted(d):
        t = d[k].detach().contiguous().cpu()
        h.update(k.encode())
        h.update(str(t.dtype).encode())
        h.update(ctypes.string_at(t.data_ptr(), t.numel() * t.element_size()))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Analysis and files.
# ---------------------------------------------------------------------------
def analyse(out: Path, run: Run, sets, ans_ids, dump_f, dump_s, extra_lines=()) -> dict:
    cfg = run.cfg
    keys = run.keys
    N = len(keys)
    lf = torch.stack(run.logits["fp16"]).double()
    ls = torch.stack(run.logits["sealed"]).double()
    A_f = torch.stack(run.ans["fp16"])  # (N, 37, d), the model dtype
    A_s = torch.stack(run.ans["sealed"])
    answers = run.answers
    lines = list(extra_lines)
    say = lines.append
    res = {}

    # Control.
    ctl = control(keys, lf, ls, dump_f, dump_s)
    res["control"] = ctl
    say(f"CONTROL (rule D), {N} questions, script against the Rust dense dumps")
    say("  model    picks equal   max |dlogit|   median |dlogit|   MMLU script   MMLU dump")
    for m in ("fp16", "sealed"):
        c = ctl[m]
        say(f"  {m:<8} {c['picks_equal']:.4f}        {c['max_abs']:.5f}        {c['median_abs']:.5f}         "
            f"{c['acc_script'] * 100:.2f}         {c['acc_dump'] * 100:.2f}")
    (bs, ns), (bd, nd) = ctl["fit_script"], ctl["fit_dump"]
    say(f"  fit sealed on FP16, script logits: slope {bs:.4f} noise {ns:.4f} slope/noise {bs / ns:.4f}")
    say(f"  fit sealed on FP16, dump logits:   slope {bd:.4f} noise {nd:.4f} slope/noise {bd / nd:.4f}")
    say(f"  slope difference {bs - bd:+.4f} (tolerance {SLOPE_TOL}); picks at least {PICK_AGREE_MIN} each")
    say(f"  RULE D: {'GREEN' if ctl['green'] else 'RED, NOTHING BELOW IS READ'}")
    with open(out / "control.tsv", "w") as f:
        f.write("subject\tindex\tanswer\tpick_fp16_script\tpick_fp16_dump\tpick_sealed_script\tpick_sealed_dump\t"
                + "\t".join(f"{m}_{src}_{x}" for m in ("fp16", "sealed") for src in ("script", "dump")
                            for x in "abcd") + "\n")
        for i, k in enumerate(keys):
            vals = (lf[i].tolist() + dump_f[k]["logits"] + ls[i].tolist() + dump_s[k]["logits"])
            f.write(f"{k[0]}\t{k[1]}\t{answers[i]}\t{pick_of(lf[i].tolist())}\t{dump_f[k]['pick']}\t"
                    f"{pick_of(ls[i].tolist())}\t{dump_s[k]['pick']}\t" + "\t".join(f"{v:.9g}" for v in vals)
                    + "\n")

    # R1, the lens with one fixed readout: FP16's final norm weight and FP16's answer rows, for both streams.
    norm_f = sets["fp16"]["model.norm.weight"]
    norm_s = sets["sealed"]["model.norm.weight"]
    head_f = sets["fp16"]["model.embed_tokens.weight"] if cfg.tied else sets["fp16"]["lm_head.weight"]
    rows = head_f[ans_ids].cpu()
    norm_f, norm_s = norm_f.cpu(), norm_s.cpu()
    lens_rows = []
    lens_logits = []
    for l in range(DEPTHS):
        zf = lens(A_f[:, l], norm_f, rows, cfg.eps)
        zs = lens(A_s[:, l], norm_f, rows, cfg.eps)
        b, n = fit(zf, zs)
        af = sum(pick_of(zf[i].tolist()) == answers[i] for i in range(N)) / N
        as_ = sum(pick_of(zs[i].tolist()) == answers[i] for i in range(N)) / N
        lens_rows.append(dict(depth=l, slope=b, noise=n, ratio=b / n if n > 0 else math.inf, acc_fp16=af,
                              acc_sealed=as_))
        lens_logits.append((zf, zs))
    res["lens"] = lens_rows
    res["lens_own_logits_maxdiff"] = float((lens_logits[-1][0] - lf).abs().max())
    res["lens_last_fp16"] = lens_logits[-1][0]
    with open(out / "lens.tsv", "w") as f:
        f.write("depth\tslope\tnoise\tslope_over_noise\tlens_acc_fp16\tlens_acc_sealed\n")
        for r in lens_rows:
            f.write(f"{r['depth']}\t{r['slope']:.6f}\t{r['noise']:.6f}\t{r['ratio']:.6f}\t{r['acc_fp16']:.6f}\t"
                    f"{r['acc_sealed']:.6f}\n")
    with open(out / "lens_logits.tsv", "w") as f:
        f.write("subject\tindex\tdepth\t" + "\t".join(f"fp16_{x}" for x in "abcd") + "\t"
                + "\t".join(f"sealed_{x}" for x in "abcd") + "\n")
        for l, (zf, zs) in enumerate(lens_logits):
            for i, k in enumerate(keys):
                f.write(f"{k[0]}\t{k[1]}\t{l}\t" + "\t".join(f"{v:.7g}" for v in zf[i].tolist() + zs[i].tolist())
                        + "\n")
    say("")
    say("R1 LENS, U_ans . final_norm_fp16(h(l)), FP16 readout for both streams; fit sealed on FP16")
    say(f"  final norm weights identical in the two sets: {torch.equal(norm_f, norm_s)}; "
        f"FP16 lens at depth {DEPTHS - 1} against its own logits, max |d| {res['lens_own_logits_maxdiff']:.3e}")
    say("  depth   slope    noise    slope/noise   lens acc FP16   lens acc sealed")
    for r in lens_rows:
        say(f"  {r['depth']:>5}   {r['slope']:.4f}   {r['noise']:.4f}   {r['ratio']:.4f}        "
            f"{r['acc_fp16']:.4f}          {r['acc_sealed']:.4f}")

    # R2 and R3, on CPU.
    t0 = time.time()
    r2, r3 = {}, {}
    for l in R2_DEPTHS:
        cff, csf, css = (run.moments.ff[l].cpu(), run.moments.sf[l].cpu(), run.moments.ss[l].cpu())
        lam, v = eig_desc(cff)
        xa, ya = A_f[:, l].double(), A_s[:, l].double()
        r2[l] = {"all": band_table(v, lam, cff, csf, css),
                 "answer": band_table(v, None, xa.T @ xa, ya.T @ xa, ya.T @ ya)}
        r3[l] = linear_table(v, lam, cff, csf, css)
    res["r2"], res["r3"] = r2, r3
    res["eig_seconds"] = time.time() - t0
    with open(out / "stream_bands.tsv", "w") as f:
        f.write("depth\tpositions\tband\tn_dirs\tslope_lambda_weighted\tslope_median\tnoise_lambda_weighted\t"
                "energy_share\tpooled_slope\tnorm_ratio\n")
        for l in R2_DEPTHS:
            for which in ("all", "answer"):
                t = r2[l][which]
                for r in t["rows"]:
                    f.write(f"{l}\t{which}\t{r['band']}\t{r['n']}\t{r['slope_w']:.6f}\t{r['slope_med']:.6f}\t"
                            f"{r['noise_w']:.6f}\t{r['share']:.6e}\t{t['pooled']:.6f}\t{t['norm_ratio']:.6f}\n")
    with open(out / "linear.tsv", "w") as f:
        f.write("# B = C_sf (C_ff + r I)^-1, r = 1e-6 tr(C_ff)/d, solved in float64: the least-squares linear map from\n"
                "# h_f to h_s over all positions (uncentred). Explained energy B C_ff B^T. Per band of FP16 eigen-\n"
                "# directions v_k (C_ff = V Lambda V^T, descending): e_lin = sum_k v_k^T B C_ff B^T v_k / sum_k\n"
                "# v_k^T C_ss v_k; diag = sum_k b_k^2 Lambda_k / sum_k v_k^T C_ss v_k, b_k = v_k^T C_sf v_k /\n"
                "# Lambda_k. e_lin - diag is energy a linear map explains beyond a per-direction gain (mixing);\n"
                "# 1 - e_lin is what no linear map of h_f explains. ridge_over_lambda_min = r / min Lambda_k in band.\n")
        f.write("depth\tband\te_lin\tdiag\tridge_over_lambda_min\te_lin_all\tdiag_all\n")
        for l in R2_DEPTHS:
            t = r3[l]
            for r in t["rows"]:
                f.write(f"{l}\t{r['band']}\t{r['e_lin']:.6f}\t{r['diag']:.6f}\t{r['ridge_over_lam_min']:.3e}\t"
                        f"{t['e_lin_all']:.6f}\t{t['diag_all']:.6f}\n")
    say("")
    say(f"R2 STREAM BY VARIANCE BAND of FP16's C_ff (all positions, {run.moments.tokens} tokens), "
        "Lambda-weighted slope / median slope / Lambda-weighted noise ratio / energy share")
    for which in ("all", "answer"):
        say(f"  positions: {which}")
        say("  depth  pooled  norm   " + "   ".join(f"[{r['band']}]" for r in r2[0][which]["rows"]))
        for l in R2_DEPTHS:
            t = r2[l][which]
            cells = "   ".join(f"{r['slope_w']:.3f}/{r['slope_med']:.3f}/{r['noise_w']:.3f}/{r['share']:.2e}"
                               for r in t["rows"])
            say(f"  {l:>5}  {t['pooled']:.4f} {t['norm_ratio']:.4f} {cells}")
    say("")
    say("R3 LINEAR PREDICTABILITY, all positions: e_lin / diagonal share, per band")
    say("  depth  all            " + "   ".join(f"[{r['band']}]" for r in r3[0]["rows"]))
    for l in R2_DEPTHS:
        t = r3[l]
        say(f"  {l:>5}  {t['e_lin_all']:.4f}/{t['diag_all']:.4f}  "
            + "   ".join(f"{r['e_lin']:.4f}/{r['diag']:.4f}" for r in t["rows"]))

    # R4.
    names = lvo.all_projections(cfg.layers)
    mats = []
    for nm in names:
        layer = int(nm.split(".")[2])
        proj = nm.split(".")[4]
        rec = dict(layer=layer, proj=proj, name=nm)
        for m in ("fp16", "sealed"):
            g = run.gain[m]
            syy, sey, see = g.totals(nm)
            a, noise, rel = gain_numbers(syy, sey, see)
            ok = g.syy[nm] > 0
            ra = (1.0 + g.sey[nm][ok] / g.syy[nm][ok]).cpu()
            qs = torch.quantile(ra, torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)).tolist()
            rec.update({f"syy_{m}": syy, f"sey_{m}": sey, f"see_{m}": see, f"a_{m}": a, f"noise_{m}": noise,
                        f"rel_{m}": rel, f"row_p10_{m}": qs[0], f"row_med_{m}": qs[1], f"row_p90_{m}": qs[2]})
        mats.append(rec)
    res["matrices"] = {r["name"]: r for r in mats}
    cols = ["layer", "proj"] + [f"{x}_{m}" for m in ("fp16", "sealed") for x in
                                ("a", "noise", "rel", "row_med", "row_p10", "row_p90", "syy", "sey", "see")]
    with open(out / "matrices.tsv", "w") as f:
        f.write("# x of the FP16 pass (_fp16) and of the sealed pass (_sealed). a = 1 + S_ey/S_yy, noise = (S_ee -\n"
                "# S_ey^2/S_yy)/S_yy, rel = S_ee/S_yy (note 1); row_* = percentiles of the per-row gains a_i\n")
        f.write("\t".join(cols) + "\n")
        for r in mats:
            f.write("\t".join(str(r[c]) if c in ("layer", "proj") else f"{r[c]:.9g}" for c in cols) + "\n")
    types = []
    for p in lvo.PROJ_TYPES:
        sel = [r for r in mats if r["proj"] == p]
        rec = dict(proj=p)
        for m in ("fp16", "sealed"):
            pooled = gain_numbers(*(sum(r[f"{s}_{m}"] for r in sel) for s in ("syy", "sey", "see")))
            for j, x in enumerate(("a", "noise", "rel")):
                rec[f"{x}_pooled_{m}"] = pooled[j]
                rec[f"{x}_mean_{m}"] = sum(r[f"{x}_{m}"] for r in sel) / len(sel)
        types.append(rec)
    res["types"] = {r["proj"]: r for r in types}
    tcols = ["proj"] + [f"{x}_{agg}_{m}" for m in ("fp16", "sealed") for x in ("a", "noise", "rel")
                        for agg in ("pooled", "mean")]
    with open(out / "types.tsv", "w") as f:
        f.write("\t".join(tcols) + "\n")
        for r in types:
            f.write("\t".join(r[c] if c == "proj" else f"{r[c]:.9g}" for c in tcols) + "\n")
    say("")
    say("R4 COHERENT GAIN BY TYPE, pooled over the 36 layers (plain mean in brackets)")
    say("  type        x     a                  noise ratio            rel error")
    for r in types:
        for m in ("fp16", "sealed"):
            say(f"  {r['proj']:<10}  {m:<6} {r[f'a_pooled_{m}']:.5f} [{r[f'a_mean_{m}']:.5f}]   "
                f"{r[f'noise_pooled_{m}']:.5e} [{r[f'noise_mean_{m}']:.5e}]   "
                f"{r[f'rel_pooled_{m}']:.5e} [{r[f'rel_mean_{m}']:.5e}]")
    lo = min(mats, key=lambda r: r["a_fp16"])
    hi = max(mats, key=lambda r: r["a_fp16"])
    say(f"  lowest a (FP16 x): layer {lo['layer']} {lo['proj']} {lo['a_fp16']:.5f}; highest: layer {hi['layer']} "
        f"{hi['proj']} {hi['a_fp16']:.5f}")

    # R5.
    with open(out / "attention.tsv", "w") as f:
        f.write("layer\thead\tentropy_fp16\ttop_fp16\tentropy_sealed\ttop_sealed\n")
        ef, tf, es, ts = (run.attn[m].__dict__[x].cpu() / run.attn[m].n
                          for m in ("fp16", "sealed") for x in ("ent", "top"))
        for b in range(cfg.layers):
            for hd in range(cfg.heads):
                f.write(f"{b}\t{hd}\t{ef[b, hd]:.6f}\t{tf[b, hd]:.6f}\t{es[b, hd]:.6f}\t{ts[b, hd]:.6f}\n")
    res["attn"] = (ef, tf, es, ts)
    say("")
    say("R5 ATTENTION OF THE ANSWER POSITION, mean over heads of the per-head means over questions (nats)")
    say("  layer  entropy FP16  entropy sealed  top FP16  top sealed  heads with higher entropy in sealed")
    for b in range(cfg.layers):
        say(f"  {b:>5}  {float(ef[b].mean()):.4f}        {float(es[b].mean()):.4f}          "
            f"{float(tf[b].mean()):.4f}    {float(ts[b].mean()):.4f}      {int((es[b] > ef[b]).sum())}/{cfg.heads}")

    # Raw dumps.
    for l in RAW_DEPTHS:
        write_raw(out / f"stream_d{l}_fp16.f16", A_f[:, l].to(torch.float16), keys)
        write_raw(out / f"stream_d{l}_sealed.f16", A_s[:, l].to(torch.float16), keys)
    with open(out / "questions.tsv", "w") as f:
        f.write("order\tsubject\tindex\tqhash\tlength\tanswer\n")
        for i, k in enumerate(keys):
            f.write(f"{i}\t{k[0]}\t{k[1]}\t{run.qhash[i]:016x}\t{run.lengths[i]}\t{answers[i]}\n")
    say("")
    say(f"RAW: stream_d{{{','.join(map(str, RAW_DEPTHS))}}}_{{fp16,sealed}}.f16 (N x d, f16, JSON sidecar), "
        f"lens_logits.tsv, control.tsv, questions.tsv; R2/R3 eigendecompositions and projections {res['eig_seconds']:.0f} s")
    text = "\n".join(lines)
    (out / "report.txt").write_text(text + "\n")
    print(text)
    return res


# ---------------------------------------------------------------------------
# Selftest: tiny random model, CPU, float64, planted distortions.
# ---------------------------------------------------------------------------
def synth_moments(lam: torch.Tensor, v: torch.Tensor, a_v: torch.Tensor, noise_v: torch.Tensor):
    """Population moments of h_f with C_ff = V diag(lam) V^T and h_s = A h_f + e, A = V a_v V^T and
    cov(e) = V noise_v V^T, e independent of h_f."""
    cff = v @ torch.diag(lam) @ v.T
    a = v @ a_v @ v.T
    csf = a @ cff
    css = a @ cff @ a.T + v @ noise_v @ v.T
    return cff, csf, css


def selftest() -> int:
    torch.manual_seed(0)
    fails = []
    n_checks = [0]

    def check(ok: bool, what: str):
        n_checks[0] += 1
        print(f"  {'PASS' if ok else 'FAIL'}  {what}")
        if not ok:
            fails.append(what)

    def close(x, y, tol):
        return abs(x - y) <= tol

    print("selftest: tiny Qwen3-shaped model (36 layers, d 16), CPU, float64; synthetic moments d 600")
    cfg = lvo.Cfg(lvo.TINY)
    fp16 = lvo.tiny_weights(cfg, 1)
    names = lvo.all_projections(cfg.layers)
    gen = torch.Generator().manual_seed(5)
    prompts = [torch.randint(0, cfg.vocab, (int(n),), generator=gen).tolist() for n in (19, 23, 17, 29, 21, 25)]
    ans_ids = [3, 7, 11, 13]
    cpu = torch.device("cpu")

    # 1. The forward with taps against layer_vs_output.forward.
    sets0 = {"fp16": fp16, "sealed": fp16}
    ids = torch.tensor(prompts[0])
    get = lvo.Arm("T", "fp16", frozenset()).resolver(sets0)
    z_ref = lvo.forward(cfg, get, ids)
    seen = []
    z_tap = forward_taps(cfg, get, ids, proj_hook=lambda n, x: None, stream_hook=lambda l, h: seen.append(l),
                         attn_hook=lambda *a: None)
    check(torch.equal(z_ref, z_tap), "forward_taps gives layer_vs_output.forward's logits bit for bit")
    check(seen == list(range(DEPTHS)), f"the stream is tapped at depths 0..{DEPTHS - 1}")
    f32 = {k: v.float() for k, v in fp16.items()}
    get32 = lvo.Arm("T", "fp16", frozenset()).resolver({"fp16": f32, "sealed": f32})
    z32, z32r = forward_taps(cfg, get32, ids), lvo.forward(cfg, get32, ids)
    check(torch.equal(z32, z32r), "and in float32")

    # 2. Attention of the last position from q and k reproduces the forward's attention output.
    worst = [0.0]

    def at(b, q, k, v, a):
        pos = answer_pos(q.shape[2])
        p = last_attention(q, k, pos, cfg.head_dim ** -0.5)
        o = torch.einsum("hl,hld->hd", p, v[0, :, :pos + 1, :].double())
        worst[0] = max(worst[0], float((o - a[0, :, pos, :].double()).abs().max()))
        check_sum.append(float((p.sum(-1) - 1).abs().max()))

    check_sum = []
    forward_taps(cfg, get, ids, attn_hook=at)
    check(worst[0] < 1e-12 and max(check_sum) < 1e-12,
          f"last-position attention weights times v reproduce the attention output, 36 layers ({worst[0]:.1e})")

    # 3. R4 planted: gate_proj of layer 20 = 0.9 W + Q W, Q antisymmetric (so <QWx, Wx> = 0 exactly), and
    #    q_proj of layer 7 = diag(g) W (row gains g); every other matrix untouched.
    gx = lvo.proj_name(20, "gate_proj")
    gy = lvo.proj_name(7, "q_proj")
    w = fp16[gx]
    qm = torch.randn(w.shape[0], w.shape[0], generator=gen, dtype=torch.float64) * 0.05
    qm = qm - qm.T
    g_rows = torch.linspace(0.5, 1.5, fp16[gy].shape[0], dtype=torch.float64)
    sealed = {k: v.clone() for k, v in fp16.items()}
    sealed[gx] = 0.9 * w + qm @ w
    sealed[gy] = g_rows[:, None] * fp16[gy]
    sealed["model.norm.weight"] = fp16["model.norm.weight"] * 1.3 + 0.1  # differs; streams do not depend on it
    sets = {"fp16": fp16, "sealed": sealed}
    run = Run(cfg, sets, names, torch.float64, cpu)
    n1 = {m: lvo.Note1(fp16, sealed, names, torch.float64) for m in ("fp16", "sealed")}
    xs_gx = []
    cap = {"fp16": lambda n, x: (n1["fp16"](n, x), xs_gx.append(x.reshape(-1, x.shape[-1])) if n == gx else None),
           "sealed": n1["sealed"]}
    for p in prompts:
        run.keys.append(("tiny", len(run.keys)))
        run.answers.append(len(run.keys) % 4)
        run.lengths.append(len(p))
        run.qhash.append(lvo.fnv_fingerprint(p))
        run_question(run, sets, p, ans_ids, cpu, extra_hooks=cap)
    a_x, noise_x, rel_x = gain_numbers(*run.gain["fp16"].totals(gx))
    check(close(a_x, 0.9, 1e-9), f"R4 recovers a = 0.9 for the planted matrix (got {a_x:.12f})")
    X = torch.cat(xs_gx)
    want_noise = float((X @ (qm @ w).T).pow(2).sum() / (X @ w.T).pow(2).sum())
    check(close(noise_x, want_noise, 1e-9 * max(1.0, want_noise)),
          f"R4 noise ratio = sum ||QWx||^2 / sum ||Wx||^2 ({noise_x:.6e} against {want_noise:.6e})")
    others = [n for n in names if n not in (gx, gy)]
    flat = all(gain_numbers(*run.gain[m].totals(n))[0] == 1.0 and gain_numbers(*run.gain[m].totals(n))[1] == 0.0
               for n in others for m in ("fp16", "sealed"))
    check(flat, f"R4 gives a = 1 and noise 0 exactly for the {len(others)} untouched matrices, both passes")
    a_s, _, _ = gain_numbers(*run.gain["sealed"].totals(gx))
    check(close(a_s, 0.9, 1e-9), f"and a = 0.9 on the sealed pass's inputs too (got {a_s:.12f})")
    worst = 0.0
    for m in ("fp16", "sealed"):
        nv = n1[m].values()
        for n in names:
            num, den = nv[n]
            rel = run.gain[m].totals(n)[2] / run.gain[m].totals(n)[0]
            worst = max(worst, abs(rel - num / den) / max(num / den, 1e-300) if num > 0 else abs(rel))
    check(worst < 1e-12, f"R4 relative error S_ee/S_yy equals Note1 on the same tokens, 252 x 2 ({worst:.1e})")
    ra = 1.0 + run.gain["fp16"].sey[gy] / run.gain["fp16"].syy[gy]
    check(float((ra - g_rows).abs().max()) < 1e-12, "R4 per-row gains recover diag(g) row for row")

    # 4. The answer position: the collected logits are the last row of layer_vs_output.forward, both models.
    ok = True
    for i, p in enumerate(prompts):
        for m in ("fp16", "sealed"):
            zr = lvo.forward(cfg, lvo.Arm(m, m, frozenset()).resolver(sets), torch.tensor(p))
            ok &= torch.equal(run.logits[m][i], zr[-1, ans_ids].float())
    check(ok, "the answer logits are the last position's, as the harness reads them, both models")

    # 5. End to end: analyse writes every file; control green against dumps equal to the script's logits, red
    #    when one pick in six is flipped; raw dumps read back.
    import tempfile

    dump_f = {k: {"pick": pick_of(run.logits["fp16"][i].tolist()), "answer": run.answers[i],
                  "logits": run.logits["fp16"][i].tolist(), "qhash": 0} for i, k in enumerate(run.keys)}
    dump_s = {k: {"pick": pick_of(run.logits["sealed"][i].tolist()), "answer": run.answers[i],
                  "logits": run.logits["sealed"][i].tolist(), "qhash": 0} for i, k in enumerate(run.keys)}
    with tempfile.TemporaryDirectory() as d:
        res = analyse(Path(d), run, sets, ans_ids, dump_f, dump_s)
        files = sorted(x.name for x in Path(d).iterdir())
        want = {"report.txt", "control.tsv", "lens.tsv", "lens_logits.tsv", "stream_bands.tsv", "linear.tsv",
                "matrices.tsv", "types.tsv", "attention.tsv", "questions.tsv"}
        check(want <= set(files), f"analyse writes {len(files)} files, the ten tables among them")
        n_mat = len([ln for ln in (Path(d) / "matrices.tsv").read_text().splitlines() if not ln.startswith("#")]) - 1
        n_att = len((Path(d) / "attention.tsv").read_text().splitlines()) - 1
        n_lens = len((Path(d) / "lens.tsv").read_text().splitlines()) - 1
        check(n_mat == 252 and n_att == 36 * cfg.heads and n_lens == DEPTHS,
              f"matrices {n_mat} rows, attention {n_att}, lens {n_lens}")
        back = read_raw(Path(d) / "stream_d24_sealed.f16")
        want_raw = torch.stack(run.ans["sealed"])[:, 24].to(torch.float16)
        check(torch.equal(back, want_raw), "raw stream dump reads back bit for bit")
    check(res["control"]["green"], "control is green against dumps equal to the script's logits")
    flip = dict(dump_f)
    k0 = run.keys[0]
    flip[k0] = dict(flip[k0], pick=(flip[k0]["pick"] + 1) % 4)
    ctl = control(run.keys, torch.stack(run.logits["fp16"]).double(), torch.stack(run.logits["sealed"]).double(),
                  flip, dump_s)
    check(not ctl["green"] and close(ctl["fp16"]["picks_equal"], 5 / 6, 1e-12),
          "control is red when one pick in six differs")
    t_ = res["types"]["gate_proj"]
    check(close(t_["a_pooled_fp16"], 1 - 0.1 * run.gain["fp16"].totals(gx)[0]
                / sum(run.gain["fp16"].totals(lvo.proj_name(b, "gate_proj"))[0] for b in range(36)), 1e-9),
          "types.tsv pooled gain of gate_proj = 1 - 0.1 S_yy(planted) / S_yy(all gate_proj)")

    # 6. R1 lens: sealed differs ONLY by its final norm weight, so every stream is equal and the fixed FP16 readout
    #    must give slope 1 and noise 0 at every depth; the FP16 lens at the last depth is FP16's own logits.
    sealed_n = {k: v.clone() for k, v in fp16.items()}
    sealed_n["model.norm.weight"] = fp16["model.norm.weight"] * 0.7 - 0.2
    sets_n = {"fp16": fp16, "sealed": sealed_n}
    run_n = Run(cfg, sets_n, names, torch.float64, cpu)
    for i, p in enumerate(prompts):
        run_n.keys.append(("tiny", i))
        run_n.answers.append(i % 4)
        run_n.lengths.append(len(p))
        run_n.qhash.append(0)
        run_question(run_n, sets_n, p, ans_ids, cpu)
    dn_f = {k: {"pick": pick_of(run_n.logits["fp16"][i].tolist()), "answer": run_n.answers[i],
                "logits": run_n.logits["fp16"][i].tolist()} for i, k in enumerate(run_n.keys)}
    dn_s = {k: {"pick": pick_of(run_n.logits["sealed"][i].tolist()), "answer": run_n.answers[i],
                "logits": run_n.logits["sealed"][i].tolist()} for i, k in enumerate(run_n.keys)}
    with tempfile.TemporaryDirectory() as d:
        rn = analyse(Path(d), run_n, sets_n, ans_ids, dn_f, dn_s)
    worst = max(max(abs(r["slope"] - 1.0), r["noise"]) for r in rn["lens"])
    check(worst < 1e-12, f"R1 lens reads slope 1 and noise 0 at all {DEPTHS} depths when only the final norm "
          f"differs (worst {worst:.1e})")
    own64 = torch.stack([lvo.forward(cfg, lvo.Arm("T", "fp16", frozenset()).resolver(sets_n),
                                     torch.tensor(p))[-1, ans_ids] for p in prompts])
    dl = float((rn["lens_last_fp16"] - own64).abs().max())
    # The model's RMSNorm runs in float32 even on float64 weights (layer_vs_output.rms_norm); the lens is float64.
    check(dl < 1e-5, f"R1 FP16 lens at depth 36 equals FP16's own last-position answer logits ({dl:.1e}; the "
          "model's norm is float32)")
    own = fit(torch.stack(run_n.logits["fp16"]), torch.stack(run_n.logits["sealed"]))
    check(abs(own[0] - 1.0) > 1e-3, f"while the own-head logits do differ (slope {own[0]:.4f})")

    # 7. The fit: planted sealed = 0.6 FP16 + noise.
    g2 = torch.Generator().manual_seed(7)
    x = torch.randn(40_000, 4, generator=g2, dtype=torch.float64) * 2.0
    y = 0.6 * x + 0.5 * torch.randn(40_000, 4, generator=g2, dtype=torch.float64)
    b, nz = fit(x, y)
    check(close(b, 0.6, 0.005) and close(nz, 0.5, 0.005), f"fit recovers slope 0.6 and noise 0.5 ({b:.4f}, {nz:.4f})")
    b0, n0 = fit(x, 0.6 * x + 3.0)
    check(close(b0, 0.6, 1e-12) and n0 < 1e-12, "fit: exact slope, zero noise, a per-question shift ignored")

    # 8. R2 / R3 on synthetic population moments, d = 600, the production bands.
    d = 600
    g3 = torch.Generator().manual_seed(11)
    v, _ = torch.linalg.qr(torch.randn(d, d, generator=g3, dtype=torch.float64))
    lam = torch.logspace(3, 0, d, dtype=torch.float64)
    beta_band = (0.95, 0.8, 0.6, 0.4, 0.2)
    sig_band = (0.01, 0.05, 0.1, 0.2, 0.3)
    bvec = torch.empty(d, dtype=torch.float64)
    svec = torch.empty(d, dtype=torch.float64)
    for (lo, hi), bb, ss in zip(BANDS, beta_band, sig_band):
        sl = slice(lo - 1, d if hi is None else hi)
        bvec[sl], svec[sl] = bb, ss
    cff, csf, css = synth_moments(lam, v, torch.diag(bvec), torch.diag(svec * lam))
    lam_e, v_e = eig_desc(cff)
    t2 = band_table(v_e, lam_e, cff, csf, css)
    ok_s = all(close(r["slope_w"], bb, 1e-9) and close(r["slope_med"], bb, 1e-9)
               for r, bb in zip(t2["rows"], beta_band))
    ok_n = all(close(r["noise_w"], ss, 1e-9) for r, ss in zip(t2["rows"], sig_band))
    check(ok_s, "R2 recovers the planted slope of each band, weighted and median: "
          + ", ".join(f"{r['slope_w']:.4f}" for r in t2["rows"]))
    check(ok_n, "R2 recovers the planted noise ratio of each band: "
          + ", ".join(f"{r['noise_w']:.4f}" for r in t2["rows"]))
    want_share = [float(lam[slice(lo - 1, d if hi is None else hi)].sum() / lam.sum()) for lo, hi in BANDS]
    check(all(close(r["share"], s, 1e-9) for r, s in zip(t2["rows"], want_share)), "R2 band energy shares")
    check(close(t2["pooled"], float((bvec * lam).sum() / lam.sum()), 1e-9)
          and close(t2["norm_ratio"], math.sqrt(float(((bvec ** 2 + svec) * lam).sum() / lam.sum())), 1e-9),
          "R2 pooled slope and norm ratio")
    t3 = linear_table(v_e, lam_e, cff, csf, css)
    want_e = [bb * bb / (bb * bb + ss) for bb, ss in zip(beta_band, sig_band)]
    check(all(close(r["e_lin"], we, 1e-4) and close(r["diag"], we, 1e-9) for r, we in zip(t3["rows"], want_e)),
          "R3 e_lin and the diagonal share both equal beta^2 / (beta^2 + sigma^2) for a diagonal map: "
          + ", ".join(f"{r['e_lin']:.4f}/{r['diag']:.4f}" for r in t3["rows"]))
    # Answer-position form: weights from the moments' own energy on V.
    lam_a = torch.linspace(5.0, 1.0, d, dtype=torch.float64)
    ca = synth_moments(lam_a, v, torch.diag(bvec * 0.5), torch.diag(svec * lam_a))
    ta = band_table(v_e, None, *ca)
    check(all(close(r["slope_w"], 0.5 * bb, 1e-9) for r, bb in zip(ta["rows"], beta_band)),
          "R2 answer-position form recovers half the slopes on the same V")
    # Pure rotation inside band [2, 8]: low diagonal slope, e_lin = 1 everywhere.
    rot = torch.eye(d, dtype=torch.float64)
    blk, _ = torch.linalg.qr(torch.randn(7, 7, generator=g3, dtype=torch.float64))
    rot[1:8, 1:8] = blk
    lam_r = lam.clone()
    lam_r[1:8] = 950.0  # one eigenvalue across the band (between lam[0] and lam[8]), so a rotation keeps C_ff
    cr = synth_moments(lam_r, v, rot, torch.zeros(d, d, dtype=torch.float64))
    # The degenerate band makes an eigensolver's V inside it arbitrary; read on the planted V, already descending.
    vr, lr = v, lam_r
    check(bool((lr[:-1] >= lr[1:]).all()), "the rotation case's eigenvalues are in descending order")
    tr2 = band_table(vr, lr, *cr)
    tr3 = linear_table(vr, lr, *cr)
    b2 = tr2["rows"][1]
    check(b2["slope_w"] < 0.9 and close(b2["slope_w"], float(torch.trace(blk)) / 7, 1e-9),
          f"pure rotation inside [2,8]: diagonal slope {b2['slope_w']:.4f} = tr(R)/7")
    check(all(close(r["e_lin"], 1.0, 1e-3) for r in tr3["rows"]) and tr3["rows"][1]["diag"] < 0.9,
          "pure rotation: e_lin = 1 in every band while the diagonal share of [2,8] is "
          f"{tr3['rows'][1]['diag']:.4f}")
    # C_ff exactly singular (ten dead coordinates): the ridge keeps the solve finite and e_lin right.
    vs = torch.zeros(d, d, dtype=torch.float64)
    q0, _ = torch.linalg.qr(torch.randn(d - 10, d - 10, generator=g3, dtype=torch.float64))
    vs[:d - 10, :d - 10] = q0
    vs[d - 10:, d - 10:] = torch.eye(10, dtype=torch.float64)
    lam_s = lam.clone()
    lam_s[d - 10:] = 0.0
    sv = svec.clone()
    sv[d - 10:] = 0.0
    cs = synth_moments(lam_s, vs, torch.diag(bvec), torch.diag(sv * lam_s))
    check(int((cs[0][d - 10:] != 0).sum()) == 0, "the singular case has ten exactly zero rows in C_ff")
    try:
        ls_, vs_ = eig_desc(cs[0])
        ts3 = linear_table(vs_, ls_, *cs)
        ts2 = band_table(vs_, ls_, *cs)
        good = (all(math.isfinite(r["e_lin"]) for r in ts3["rows"])
                and all(close(r["e_lin"], we, 1e-3) for r, we in zip(ts3["rows"][:4], want_e[:4]))
                and close(ts2["rows"][4]["slope_w"], beta_band[4], 1e-6))
        msg = ", ".join(f"{r['e_lin']:.4f}" for r in ts3["rows"])
    except Exception as e:  # noqa: BLE001
        good, msg = False, f"{type(e).__name__}: {e}"
    check(good, f"singular C_ff: the ridge keeps R3 finite and on the planted values ({msg})")

    # 9. The moment accumulator: C_sf = sum h_s h_f^T, not its transpose.
    mo = Moments((0,), 5, cpu)
    hf_ = torch.randn(13, 5, generator=g3, dtype=torch.float64)
    hs_ = torch.randn(13, 5, generator=g3, dtype=torch.float64)
    mo.add(0, hf_[:7], hs_[:7])
    mo.add(0, hf_[7:], hs_[7:])
    check(torch.allclose(mo.sf[0], hs_.T @ hf_, atol=1e-12) and torch.allclose(mo.ff[0], hf_.T @ hf_, atol=1e-12)
          and torch.allclose(mo.ss[0], hs_.T @ hs_, atol=1e-12) and not torch.allclose(mo.sf[0], mo.sf[0].T),
          "Moments accumulates C_ff, C_sf = sum h_s h_f^T, C_ss over calls")

    # 10. Prompt pieces, ordering, budget.
    it = {"question": "\u001c  What is 2+2? \n", "choices": [" 3", "4 ", " 5", "6　"], "answer": 1}
    check(block(it, None) == "\u001c  What is 2+2?\nA. 3\nB. 4\nC. 5\nD. 6\nAnswer:",
          "block: Rust trim (U+001C kept, U+00A0 and U+3000 stripped), no space after 'Answer:'")
    check(block(it, 1).endswith("Answer: B\n\n"), "a worked example ends 'Answer: B' and a blank line")
    check(prefix_for("high_school_us_history", []) ==
          "The following are multiple choice questions (with answers) about high school us history.\n\n",
          "header names the subject with underscores turned to spaces")
    check(pick_of([1.0, 3.0, 3.0, 2.0]) == 2 and pick_of([5.0, 5.0, 5.0, 5.0]) == 3 and pick_of([4, 1, 1, 1]) == 0,
          "tie rule: the last of equal maxima, as Rust max_by")
    qs = [{"subject": s, "index": i} for s, n in (("b", 3), ("a", 2), ("c", 4)) for i in reversed(range(n))]
    rr = round_robin(qs)
    check([(q["subject"], q["index"]) for q in rr] ==
          [("a", 0), ("b", 0), ("c", 0), ("a", 1), ("b", 1), ("c", 1), ("b", 2), ("c", 2), ("c", 3)],
          "round robin: subjects in byte order, index ascending, every prefix balanced")
    check(n_from_time(57.0, 57, 600.0) == 600 and n_from_time(600.0, 57, 600.0) == N_MIN
          and n_from_time(1.0, 57, 600.0) == N_MAX, "N = budget-fitted count clamped to [570, 2280]")
    check(answer_pos(10) == 9, "answer position is the last")

    print(f"selftest: {'GREEN' if not fails else 'RED'}, {n_checks[0]} checks, {len(fails)} failure(s)")
    for f_ in fails:
        print(f"  failed: {f_}")
    return 0 if not fails else 1


# ---------------------------------------------------------------------------
# Modes on real inputs.
# ---------------------------------------------------------------------------
def check_prompts(args) -> int:
    qs, ans_ids = fetch_prompts(args.cache)
    root = Path(__file__).resolve().parent.parent
    dumps = [read_dump(Path(args.dump_f16 or root / DUMP_F16[0]), DUMP_F16[1]),
             read_dump(Path(args.dump_sealed or root / DUMP_SEALED[0]), DUMP_SEALED[1])]
    bad = sum(1 for q in qs for dmp in dumps if dmp[(q["subject"], q["index"])]["qhash"] != q["qhash"])
    lens_ = sorted(len(q["ids"]) for q in qs)
    rr = round_robin(qs)
    first = [len(q["ids"]) for q in rr[:N_MAX]]
    print(f"qhash: {2 * len(qs) - bad} of {2 * len(qs)} equal to the two dumps; answer token ids {ans_ids}")
    print(f"prompt length, census: median {lens_[len(lens_) // 2]}, max {lens_[-1]}; round robin first 570 mean "
          f"{sum(first[:570]) / 570:.0f}, first {N_MAX} mean {sum(first) / len(first):.0f}, first 57 mean "
          f"{sum(first[:57]) / 57:.0f}, sum of squares first 570 {sum(x * x for x in first[:570]):.3e}")
    return 0 if bad == 0 else 2


def run(args) -> int:
    lvo.check_pins()
    if not torch.cuda.is_available():
        raise SystemExit("refused: --run is for the card; the 4B is never loaded on the Mac")
    device = torch.device("cuda:0")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    dump_f = read_dump(Path(args.dump_f16), DUMP_F16[1])
    dump_s = read_dump(Path(args.dump_sealed), DUMP_SEALED[1])
    qs, ans_ids = fetch_prompts(args.cache)
    bad = [(q["subject"], q["index"]) for q in qs for dmp in (dump_f, dump_s)
           if dmp[(q["subject"], q["index"])]["qhash"] != q["qhash"]]
    print(f"qhash: {2 * len(qs) - len(bad)} of {2 * len(qs)} equal to the dumps; answer ids {ans_ids}")
    if bad:
        raise SystemExit(f"refused: {len(bad)} prompts differ from the dumps' token ids, first {bad[:3]}")
    order = round_robin(qs)
    cfg, sets, extra = lvo.load_sets(Path(args.export), Path(args.stage), args.cache, device)
    t0 = time.time()
    h0 = {k: sha_set(v) for k, v in sets.items()}
    print(f"bases: fp16 {h0['fp16'][:16]}, sealed {h0['sealed'][:16]} ({time.time() - t0:.0f} s)", flush=True)
    names = lvo.all_projections(cfg.layers)

    # R4 against Note1 on the first question, both passes, before the timed loop.
    probe = Run(cfg, sets, names, torch.float32, device)
    n1 = {m: lvo.Note1(sets["fp16"], sets["sealed"], names, torch.float32) for m in ("fp16", "sealed")}
    run_question(probe, sets, order[0]["ids"], ans_ids, device, extra_hooks=n1)
    n1_worst = 0.0
    for m in ("fp16", "sealed"):
        nv = n1[m].values()
        for n in names:
            syy, _, see = probe.gain[m].totals(n)
            n1_worst = max(n1_worst, abs(see / syy - nv[n][0] / nv[n][1]) / (nv[n][0] / nv[n][1]))
    print(f"R4 against Note1 on question 0, 252 x 2: worst relative difference {n1_worst:.2e}", flush=True)
    del probe, n1
    torch.cuda.empty_cache()

    acc = Run(cfg, sets, names, torch.float32, device)
    n_target = None
    t_loop = time.time()
    i = 0
    while n_target is None or i < n_target:
        q = order[i]
        acc.keys.append((q["subject"], q["index"]))
        acc.answers.append(q["answer"])
        acc.lengths.append(len(q["ids"]))
        acc.qhash.append(q["qhash"])
        run_question(acc, sets, q["ids"], ans_ids, device)
        i += 1
        if i % 57 == 0:
            torch.cuda.synchronize()
            el = time.time() - t_loop
            print(f"  {i} questions, {el:.0f} s, {acc.moments.tokens} tokens", flush=True)
            if n_target is None:
                n_target = n_from_time(el, i, args.budget_seconds)
                print(f"questions: N = {n_target} (first {i} in {el:.1f} s, budget {args.budget_seconds} s, "
                      f"clamped to [{N_MIN}, {N_MAX}])", flush=True)
    torch.cuda.synchronize()
    loop_s = time.time() - t_loop
    print(f"loop: {i} questions, {acc.moments.tokens} tokens, {loop_s:.0f} s", flush=True)
    h1 = {k: sha_set(v) for k, v in sets.items()}
    same = h0 == h1
    print(f"bases unchanged by sha256: {'GREEN' if same else 'RED'}")
    head = [f"inputs: checkpoint {lvo.CKPT_REPO}@{lvo.CKPT_REV}; export sha256 {lvo.EXPORT_SHA}; "
            f"{MMLU_REPO}@{MMLU_REV}; run fingerprint {RUN_FP:016x}; answer ids {ans_ids}",
            f"questions: {i} (round robin over {N_SUBJECTS} subjects), {acc.moments.tokens} tokens, loop {loop_s:.0f} s",
            f"bases unchanged by sha256: {same}; R4 against Note1 on question 0: worst relative difference "
            f"{n1_worst:.2e}; non-projection tensors identical in W and W_hat: {len(extra['same_other'])} of "
            f"{len(extra['others'])}", ""]
    res = analyse(out, acc, sets, ans_ids, dump_f, dump_s, extra_lines=head)
    (out / "inputs.txt").write_text(
        f"checkpoint {lvo.CKPT_REPO}@{lvo.CKPT_REV}\nexport {args.export} sha256 {lvo.EXPORT_SHA}\n"
        f"mmlu {MMLU_REPO}@{MMLU_REV}\ndumps {DUMP_F16[1]} {DUMP_SEALED[1]}\nquestions {i}\n"
        f"bases sha256 before {h0}\nbases sha256 after  {h1}\nrule D green {res['control']['green']}\n"
        f"total {time.time() - t_start:.0f} s\n")
    if not same or n1_worst > 1e-4:
        return 3
    return 0 if res["control"]["green"] else 4


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--selftest", action="store_true")
    mode.add_argument("--check-prompts", action="store_true")
    mode.add_argument("--run", action="store_true")
    ap.add_argument("--export", help="the sealed export's model.safetensors (bucket mount)")
    ap.add_argument("--stage", default="/tmp/cm-export", help="local copy of the export")
    ap.add_argument("--out", help="output folder")
    ap.add_argument("--dump-f16", help=f"default {DUMP_F16[0]}")
    ap.add_argument("--dump-sealed", help=f"default {DUMP_SEALED[0]}")
    ap.add_argument("--cache", default=os.environ.get("HF_HUB_CACHE"))
    ap.add_argument("--budget-seconds", type=float, default=600.0)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.check_prompts:
        return check_prompts(a)
    if not (a.export and a.out and a.dump_f16 and a.dump_sealed):
        ap.error("--run needs --export, --out, --dump-f16 and --dump-sealed")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
