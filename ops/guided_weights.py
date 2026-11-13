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
"""Per-token gradient weights of gate_proj and up_proj on the 4B, for the guided encode (V1).

Cost note: docs/mesures/recherche-qualite-r2-chiffrage-encodeur-2026-11-13.txt (V1). Prereg:
proofs/preregistration-guided-gateup-4b-2026-11-13.md. Consumer: `llvq-llm/src/guided.rs`, which
reads the files written here under `LLVQ_GUIDED_WEIGHTS`.

For every calibration window of the encoder (the token ids `bin/calibtokens` writes, the very cut
`bin/smoke` calibrates on), one forward and one backward of the FP16 checkpoint
Qwen/Qwen3-4B@1cfa9a72, weights cast up to fp32, TF32 off, loss the next-token cross-entropy of the
window on its own tokens, summed over its positions. For each layer and each of gate_proj and
up_proj, the gradient of that loss with respect to the projection's output, g (tokens x 9,728), is
reduced to

    w[t, j] = sum over output rows r in [j * 608, (j + 1) * 608) of g[t, r]^2,   j = 0 .. 15,

and written as little-endian f32, shape [131,072, 16], one file per (layer, type): 72 files and a
manifest. `guided.rs` sums neighbouring columns into G in {1, 2, 4, 8, 16} groups and scales each
group to mean 1, so one job serves every G.

This is GuidedQuant's objective (Kim et al., arXiv 2505.07004, Eq. 4 and section 3.2): the group
Hessian H_k = X' Diag(sum over channels of the group of (dl/dz_j)^2) X. Departures, each stated:
  - GuidedQuant averages the squared gradients over the channels of a group (1/|J_k|); this sums
    them. The encoder rescales every group to mean 1, and the GPTQ path is invariant to the scale
    of H, so the two are the same encode.
  - GuidedQuant scales the gradients by 1e3 against underflow; here they are fp32 and unscaled.
  - GuidedQuant calibrates on 1,024 RedPajama sequences of 4,096; here the encoder's own 64 DCLM-edu
    windows of 2,048, because the weights multiply the encoder's own activations row by row.
  - The loss: the paper read here does not say which labels; this uses the window's next tokens
    (the operator's brief of 2026-11-13).

The forward. `layer_vs_output.forward` differentiates, but at 4B in fp32 its saved activations
(about 31 GB at 2,048 tokens, estimated) and the 16 GB of fp32 weights do not fit the L40S's 46 GB.
So `forward_probed` below repeats it op for op, layer by layer under activation checkpointing, and
adds a zero "probe" to each gate/up output whose gradient is dl/dz. The selftest holds it to
`layer_vs_output.forward` bit for bit on a tiny model, holds the probe gradients to central finite
differences, and holds them to output hooks placed on `layer_vs_output.forward` itself.

Modes:
  --selftest   tiny random model, CPU, float64, seconds; no download.
  --run        the capture. Card only: the 4B is never loaded on the Mac.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layer_vs_output as lvo  # noqa: E402  (the pins, the helpers and the reference forward)

FORMAT = "llvq-guided-weights-v1"
TOKENS_FORMAT = "llvq-calib-tokens-v1"
G_MAX = 16
TYPES = ("gate_proj", "up_proj")
WINDOWS, WINDOW_LEN = 64, 2048
# The shared round-trip vector (llvq-llm/tests/fixtures/guided/roundtrip-3x16.f32): 3 x 16 floats,
# value i * 0.25 for i = 0 .. 47, little-endian f32.
FIXTURE_SHA = "2986a643da0ea8a9d084d88bba9b749204f618f576dc3ae8ff25ac934d11d477"
FIXTURE_REL = "llvq-llm/tests/fixtures/guided/roundtrip-3x16.f32"


def file_name(layer: int, proj: str) -> str:
    """`guided::file_name`: w-003-gate_proj.f32."""
    return f"w-{layer:03d}-{proj}.f32"


def group_sq(g: torch.Tensor) -> torch.Tensor:
    """(tokens, d) gradient -> (tokens, 16): squares summed over 16 consecutive equal row chunks."""
    t, d = g.shape
    if d % G_MAX:
        raise SystemExit(f"refused: {d} output rows is not a multiple of {G_MAX}")
    return g.pow(2).view(t, G_MAX, d // G_MAX).sum(-1)


def f32_bytes(t: torch.Tensor) -> bytes:
    """Little-endian f32, row-major. Through `struct`, not numpy: the pinned environment has no numpy
    (pyarrow 20 does not pull it), and a value read back from f32 packs to the same four bytes."""
    v = t.detach().to(torch.float32).contiguous().cpu().reshape(-1).tolist()
    return struct.pack(f"<{len(v)}f", *v)


def read_weight_file(path: Path, tokens: int) -> torch.Tensor:
    raw = path.read_bytes()
    if len(raw) != tokens * G_MAX * 4:
        raise SystemExit(f"refused: {path} is {len(raw)} B, expected {tokens * G_MAX * 4}")
    return torch.tensor(struct.unpack(f"<{tokens * G_MAX}f", raw), dtype=torch.float32).view(tokens, G_MAX)


# ---------------------------------------------------------------------------
# The token file of `bin/calibtokens` (`guided::write_token_file`).
# ---------------------------------------------------------------------------
def read_tokens(path: Path, windows: int | None = WINDOWS, window_len: int | None = WINDOW_LEN):
    meta = json.loads(Path(f"{path}.json").read_text())
    if meta.get("format") != TOKENS_FORMAT:
        raise SystemExit(f"refused: {path}.json format {meta.get('format')!r}, expected {TOKENS_FORMAT!r}")
    raw = path.read_bytes()
    n = meta["tokens"]
    if len(raw) != 4 * n or meta["windows"] * meta["window_len"] != n:
        raise SystemExit(f"refused: {path} is {len(raw)} B for {n} tokens of "
                         f"{meta['windows']} x {meta['window_len']}")
    sha = hashlib.sha256(raw).hexdigest()
    if sha != meta["sha256"]:
        raise SystemExit(f"refused: {path} sha256 {sha}, its sidecar says {meta['sha256']}")
    ids = list(struct.unpack(f"<{n}I", raw))
    fp = f"{lvo.fnv_fingerprint(ids):016x}"
    if fp != meta["fingerprint"]:
        raise SystemExit(f"refused: {path} fingerprints to {fp}, its sidecar says {meta['fingerprint']}")
    if windows is not None and (meta["windows"], meta["window_len"]) != (windows, window_len):
        raise SystemExit(f"refused: {meta['windows']} x {meta['window_len']} windows, the encoder's are "
                         f"{windows} x {window_len}")
    return ids, meta


def write_tokens(path: Path, ids: list[int], windows: int, window_len: int) -> None:
    """`guided::write_token_file`'s layout, for the selftest."""
    raw = struct.pack(f"<{len(ids)}I", *ids)
    path.write_bytes(raw)
    Path(f"{path}.json").write_text(json.dumps({
        "format": TOKENS_FORMAT, "windows": windows, "window_len": window_len, "tokens": len(ids),
        "fingerprint": f"{lvo.fnv_fingerprint(ids):016x}", "sha256": hashlib.sha256(raw).hexdigest()}))


# ---------------------------------------------------------------------------
# The forward, with a probe on every gate/up output.
# ---------------------------------------------------------------------------
def forward_probed(cfg, get, ids: torch.Tensor, probes: dict | None = None, ckpt: bool = False) -> torch.Tensor:
    """`layer_vs_output.forward`, op for op, with `probes[(layer, proj)]` (1, L, inter) added to the
    gate/up outputs, and each layer optionally under activation checkpointing."""
    emb = get("model.embed_tokens.weight")
    dtype = emb.dtype
    length = ids.shape[0]
    h = emb[ids].unsqueeze(0)
    cos, sin = lvo.rope_tables(cfg, length, ids.device, dtype)

    def layer(b: int, h: torch.Tensor, pg, pu) -> torch.Tensor:
        p = f"model.layers.{b}."
        x = lvo.rms_norm(h, get(p + "input_layernorm.weight"), cfg.eps)
        q = F.linear(x, get(p + "self_attn.q_proj.weight")).view(1, length, cfg.heads, cfg.head_dim)
        k = F.linear(x, get(p + "self_attn.k_proj.weight")).view(1, length, cfg.kv_heads, cfg.head_dim)
        v = F.linear(x, get(p + "self_attn.v_proj.weight")).view(1, length, cfg.kv_heads, cfg.head_dim)
        q = lvo.rms_norm(q, get(p + "self_attn.q_norm.weight"), cfg.eps).transpose(1, 2)
        k = lvo.rms_norm(k, get(p + "self_attn.k_norm.weight"), cfg.eps).transpose(1, 2)
        v = v.transpose(1, 2)
        q = q * cos + lvo.rotate_half(q) * sin
        k = k * cos + lvo.rotate_half(k) * sin
        rep = cfg.heads // cfg.kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=cfg.head_dim ** -0.5)
        a = a.transpose(1, 2).reshape(1, length, cfg.heads * cfg.head_dim)
        h = h + F.linear(a, get(p + "self_attn.o_proj.weight"))
        x = lvo.rms_norm(h, get(p + "post_attention_layernorm.weight"), cfg.eps)
        g = F.linear(x, get(p + "mlp.gate_proj.weight"))
        u = F.linear(x, get(p + "mlp.up_proj.weight"))
        if pg is not None:
            g = g + pg
            u = u + pu
        return h + F.linear(F.silu(g) * u, get(p + "mlp.down_proj.weight"))

    for b in range(cfg.layers):
        pg = probes[(b, "gate_proj")] if probes else None
        pu = probes[(b, "up_proj")] if probes else None
        if ckpt:
            h = torch_checkpoint(layer, b, h, pg, pu, use_reentrant=False)
        else:
            h = layer(b, h, pg, pu)
    h = lvo.rms_norm(h, get("model.norm.weight"), cfg.eps)
    head = emb if cfg.tied else get("lm_head.weight")
    return F.linear(h, head)[0]


def seq_loss(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Next-token cross-entropy of the window on its own tokens, summed over its L - 1 positions."""
    return F.cross_entropy(logits[:-1].float() if logits.dtype != torch.float64 else logits[:-1],
                           ids[1:], reduction="sum")


def probe_grads(cfg, get, ids: torch.Tensor, ckpt: bool):
    """dl/dz for every gate/up output, (L, inter) each, and the summed loss."""
    dev, dt = ids.device, get("model.embed_tokens.weight").dtype
    probes = {(b, p): torch.zeros((1, ids.shape[0], cfg.inter), dtype=dt, device=dev, requires_grad=True)
              for b in range(cfg.layers) for p in TYPES}
    loss = seq_loss(forward_probed(cfg, get, ids, probes, ckpt), ids)
    grads = torch.autograd.grad(loss, list(probes.values()))
    return {k: g[0].detach() for k, g in zip(probes, grads)}, float(loss)


class _HookF:
    """`torch.nn.functional` with `linear` hooked: the gradient at the output of every watched weight
    is kept. Installed as `layer_vs_output.F` to read gradients off `layer_vs_output.forward` itself."""

    def __init__(self, watch: dict[int, tuple[int, str]]):
        self.watch, self.grads = watch, {}

    def __getattr__(self, name):
        return getattr(torch.nn.functional, name)

    def linear(self, x, w, b=None):
        out = torch.nn.functional.linear(x, w, b)
        key = self.watch.get(id(w))
        if key is not None and out.requires_grad:
            out.register_hook(lambda g, key=key: self.grads.__setitem__(key, g[0].detach().clone()))
        return out


def hooked_grads(cfg, weights: dict, ids: torch.Tensor):
    """The same gradients by output hooks on `layer_vs_output.forward`, independent of the probes."""
    w = dict(weights)
    norm0 = "model.layers.0.input_layernorm.weight"
    w[norm0] = weights[norm0].detach().clone().requires_grad_(True)  # makes every activation require grad
    watch = {id(w[lvo.proj_name(b, p)]): (b, p) for b in range(cfg.layers) for p in TYPES}
    shim = _HookF(watch)
    saved = lvo.F
    lvo.F = shim
    try:
        z = lvo.forward(cfg, w.__getitem__, ids)
        seq_loss(z, ids).backward()
    finally:
        lvo.F = saved
    return shim.grads


# ---------------------------------------------------------------------------
# The capture and its files.
# ---------------------------------------------------------------------------
def capture(cfg, get, windows: list[list[int]], device, ckpt: bool, log=print):
    """Per (layer, type): a (tokens, 16) f32 tensor on the CPU, windows in order. And per window:
    mean loss, seconds, zero weights."""
    n = sum(len(w) for w in windows)
    out = {(b, p): torch.empty((n, G_MAX), dtype=torch.float32) for b in range(cfg.layers) for p in TYPES}
    rows, at = [], 0
    for i, win in enumerate(windows):
        t0 = time.time()
        ids = torch.tensor(win, dtype=torch.long, device=device)
        grads, loss = probe_grads(cfg, get, ids, ckpt)
        zeros = 0
        for k, g in grads.items():
            w = group_sq(g).to(torch.float32).cpu()
            zeros += int((w == 0).sum())
            out[k][at:at + len(win)] = w
        del grads
        at += len(win)
        dt = time.time() - t0
        rows.append({"window": i, "loss_mean": loss / (len(win) - 1), "seconds": round(dt, 3), "zero_weights": zeros})
        log(f"  window {i + 1}/{len(windows)}: {dt:.1f} s, loss {loss / (len(win) - 1):.5f} nats/token, "
            f"{zeros} zero weights", flush=True)
    return out, rows


def write_dir(out: Path, arrays: dict, layers: int, tokens: int, meta: dict) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for b in range(layers):
        for p in TYPES:
            a = arrays[(b, p)]
            if tuple(a.shape) != (tokens, G_MAX):
                raise SystemExit(f"refused: layer {b} {p} is {tuple(a.shape)}")
            if not bool(torch.isfinite(a).all()) or bool((a < 0).any()):
                raise SystemExit(f"refused: layer {b} {p} holds a non-finite or negative weight")
            raw = f32_bytes(a)
            name = file_name(b, p)
            (out / name).write_bytes(raw)
            files[name] = hashlib.sha256(raw).hexdigest()
    manifest = {"format": FORMAT, "g_max": G_MAX, "tokens": tokens, "layers": layers, **meta, "files": files}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


# ---------------------------------------------------------------------------
# Selftest: tiny random model, CPU, float64.
# ---------------------------------------------------------------------------
TINY = dict(hidden_size=16, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            intermediate_size=32, vocab_size=40, rms_norm_eps=1e-6, rope_theta=1_000_000.0,
            tie_word_embeddings=True, rope_scaling=None, use_sliding_window=False, attention_bias=False)


def selftest() -> int:
    torch.manual_seed(0)
    cfg = lvo.Cfg(TINY)
    fails = []

    def check(ok: bool, what: str):
        print(f"  {'PASS' if ok else 'FAIL'}  {what}")
        if not ok:
            fails.append(what)

    print("selftest: tiny Qwen3-shaped model, 3 layers, intermediate 32, CPU, float64")
    w = lvo.tiny_weights(cfg, 1)
    get = w.__getitem__
    ids = torch.randint(0, cfg.vocab, (12,), generator=torch.Generator().manual_seed(5))

    # 1. The forward is layer_vs_output's, bit for bit, with and without probes and checkpointing.
    z_ref = lvo.forward(cfg, get, ids)
    check(torch.equal(forward_probed(cfg, get, ids), z_ref), "forward_probed equals layer_vs_output.forward exactly")
    zero = {(b, p): torch.zeros((1, 12, cfg.inter), dtype=torch.float64) for b in range(cfg.layers) for p in TYPES}
    check(torch.equal(forward_probed(cfg, get, ids, zero, ckpt=True), z_ref),
          "with zero probes and checkpointing, the logits are unchanged")

    # 2a. Probe gradients against central finite differences of the loss. `layer_vs_output.rms_norm`
    # computes in f32 as transformers does, which puts f32 rounding into the loss and drowns a
    # difference quotient; for this check only, both the gradients and the differences run with a
    # float64 RMSNorm, so the comparison is of autograd against the derivative, in float64.
    g, loss = probe_grads(cfg, get, ids, ckpt=False)
    g_ck, loss_ck = probe_grads(cfg, get, ids, ckpt=True)
    check(loss == loss_ck and all(torch.equal(g[k], g_ck[k]) for k in g),
          "checkpointing changes neither the loss nor any gradient")
    saved_norm = lvo.rms_norm
    lvo.rms_norm = lambda x, wn, e: wn * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + e))
    try:
        g64, _ = probe_grads(cfg, get, ids, ckpt=False)
        eps, worst = 1e-6, 0.0
        for (b, p, t, r) in [(0, "gate_proj", 3, 5), (1, "up_proj", 0, 31), (2, "gate_proj", 10, 16),
                             (2, "up_proj", 7, 0)]:
            def lossat(delta):
                pr = {k: v.clone() for k, v in zero.items()}
                pr[(b, p)][0, t, r] = delta
                return float(seq_loss(forward_probed(cfg, get, ids, pr), ids))
            fd = (lossat(eps) - lossat(-eps)) / (2 * eps)
            worst = max(worst, abs(fd - float(g64[(b, p)][t, r])) / max(1.0, abs(fd)))
    finally:
        lvo.rms_norm = saved_norm
    check(worst < 1e-7, f"probe gradients match finite differences, float64 RMSNorm (worst relative error {worst:.2e})")
    last = max(float(g[k][-1].abs().max()) for k in g)
    check(last == 0.0, "the last position, which has no label, has gradient exactly 0")

    # 2b. The same gradients by hooks on layer_vs_output.forward itself.
    hk = hooked_grads(cfg, w, ids)
    diff = max(float((hk[k] - g[k]).abs().max()) for k in g) if set(hk) == set(g) else math.inf
    check(diff < 1e-12, f"hooked gradients on layer_vs_output.forward equal the probe gradients (max {diff:.1e})")

    # 3. The reduction: squares summed over 16 consecutive chunks; non-negative; per-sequence totals.
    ok_sum, ok_tot, ok_neg = True, True, True
    chunk = cfg.inter // G_MAX
    for k, gk in g.items():
        red = group_sq(gk)
        ok_neg &= bool((red >= 0).all())
        for j in range(G_MAX):
            want = sum(float(gk[t, r]) ** 2 for t in range(12) for r in range(j * chunk, (j + 1) * chunk))
            got_cols = [sum(float(gk[t, r]) ** 2 for r in range(j * chunk, (j + 1) * chunk)) for t in range(12)]
            ok_sum &= all(abs(float(red[t, j]) - got_cols[t]) <= 1e-12 * max(1.0, got_cols[t]) for t in range(12))
            ok_tot &= abs(float(red[:, j].sum()) - want) <= 1e-12 * max(1.0, want)
    check(ok_sum, "column j is the sum of squared gradients over rows [j*d/16, (j+1)*d/16), token by token")
    check(ok_tot and ok_neg, "weights are non-negative and sum per sequence to the group's squared gradient norm")

    # 4. The file round-trip, against the vector the Rust reader test reads.
    vec = torch.arange(48, dtype=torch.float32).view(3, G_MAX) * 0.25
    raw = f32_bytes(vec)
    check(hashlib.sha256(raw).hexdigest() == FIXTURE_SHA, "the writer's bytes have the fixture's sha256")
    fix = Path(__file__).resolve().parent.parent / FIXTURE_REL
    if fix.exists():
        check(fix.read_bytes() == raw, f"the writer's bytes equal {FIXTURE_REL}")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "v.f32"
        p.write_bytes(raw)
        check(torch.equal(read_weight_file(p, 3), vec), "the vector reads back as written")

        # 5. The token file: read, and refused when its fingerprint or size lies.
        tp = Path(d) / "t.u32"
        toks = [int(x) for x in torch.randint(0, 151_936, (3 * 8,), generator=torch.Generator().manual_seed(9))]
        write_tokens(tp, toks, 3, 8)
        got, _ = read_tokens(tp, 3, 8)
        check(got == toks, "the token file reads back, fingerprint checked")
        side = json.loads(Path(f"{tp}.json").read_text())
        side["fingerprint"] = "0" * 16
        Path(f"{tp}.json").write_text(json.dumps(side))
        try:
            read_tokens(tp, 3, 8)
            check(False, "a wrong fingerprint is refused")
        except SystemExit as e:
            check("fingerprint" in str(e), "a wrong fingerprint is refused")
        write_tokens(tp, toks, 3, 8)
        tp.write_bytes(tp.read_bytes()[:-4])
        try:
            read_tokens(tp, 3, 8)
            check(False, "a truncated token file is refused")
        except SystemExit as e:
            check("refused" in str(e), "a truncated token file is refused")

        # 6. The capture end to end on two windows, its files and manifest.
        wins = [ids.tolist(), [int(x) for x in torch.randint(0, cfg.vocab, (12,), generator=torch.Generator().manual_seed(6))]]
        arrays, rows = capture(cfg, get, wins, torch.device("cpu"), ckpt=True, log=lambda *a, **k: None)
        out = Path(d) / "w"
        m = write_dir(out, arrays, cfg.layers, 24, {"token_fingerprint": "0" * 16})
        names = sorted(p.name for p in out.glob("w-*.f32"))
        check(len(names) == 2 * cfg.layers and names == sorted(m["files"]), f"{len(names)} files, one per (layer, type)")
        check(all(hashlib.sha256((out / n).read_bytes()).hexdigest() == s for n, s in m["files"].items()),
              "every file matches its manifest sha256")
        check(all(torch.equal(read_weight_file(out / file_name(b, p), 24), arrays[(b, p)])
                  for b in range(cfg.layers) for p in TYPES), "every file reads back to the captured array")
        first = arrays[(1, "up_proj")][:12]
        check(torch.equal(first, group_sq(g[(1, "up_proj")]).to(torch.float32)),
              "window 1 of the capture is the reduction of its probe gradients, in stream order")
        check(len(rows) == 2 and rows[0]["zero_weights"] == 2 * cfg.layers * G_MAX,
              "the only zero weights are the last position of each (layer, type)")

    print(f"selftest: {'GREEN' if not fails else 'RED'} ({len(fails)} failure(s))")
    return 0 if not fails else 1


# ---------------------------------------------------------------------------
# The run, on the card.
# ---------------------------------------------------------------------------
def load_fp32(cache: str | None, device):
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    cfg_path = Path(hf_hub_download(lvo.CKPT_REPO, "config.json", revision=lvo.CKPT_REV, cache_dir=cache))
    cfg = lvo.Cfg(json.loads(cfg_path.read_text()))
    w, shards = {}, {}
    for shard, want in lvo.CKPT_SHARDS.items():
        p = Path(hf_hub_download(lvo.CKPT_REPO, shard, revision=lvo.CKPT_REV, cache_dir=cache))
        got = lvo.sha256_file(p)
        print(f"checkpoint: {shard} sha256 {got}")
        if got != want:
            raise SystemExit(f"refused: {shard} sha256 {got}, expected {want}")
        shards[shard] = got
        for k, v in load_file(str(p), device=str(device)).items():
            w[k] = v.to(torch.float32)
    if cfg.tied:
        w.pop("lm_head.weight", None)
    if cfg.inter % G_MAX:
        raise SystemExit(f"refused: intermediate {cfg.inter} is not a multiple of {G_MAX}")
    return cfg, w, shards


def run(args) -> int:
    lvo.check_pins()
    if not torch.cuda.is_available():
        raise SystemExit("refused: --run is for the card; the 4B is never loaded on the Mac")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    t_all = time.time()
    ids, tmeta = read_tokens(Path(args.tokens))
    if tmeta.get("tokenizer_sha256") != lvo.TOKENIZER_SHA:
        raise SystemExit(f"refused: the token file was cut with tokenizer {tmeta.get('tokenizer_sha256')}, "
                         f"the pinned one is {lvo.TOKENIZER_SHA}")
    if tmeta.get("calibration") != "dclm-edu" or tmeta.get("calib_seed") != "prefix":
        raise SystemExit(f"refused: calibration {tmeta.get('calibration')}, seed {tmeta.get('calib_seed')}; "
                         "the encoder's is dclm-edu, prefix")
    print(f"tokens: {len(ids)}, fingerprint {tmeta['fingerprint']}, sha256 {tmeta['sha256']}")
    windows = [ids[i * WINDOW_LEN:(i + 1) * WINDOW_LEN] for i in range(WINDOWS)]
    cfg, w, shards = load_fp32(args.cache, device)
    print(f"loaded fp32, {sum(v.numel() for v in w.values()) / 1e9:.3f} G values, {time.time() - t_all:.0f} s; "
          f"cuda memory {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    t0 = time.time()
    arrays, rows = capture(cfg, w.__getitem__, windows, device, ckpt=True)
    print(f"capture: {len(rows)} windows in {time.time() - t0:.0f} s; peak cuda memory "
          f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    out = Path(args.out)
    meta = {
        "token_fingerprint": tmeta["fingerprint"],
        "token_file_sha256": tmeta["sha256"],
        "token_meta": tmeta,
        "checkpoint": f"{lvo.CKPT_REPO}@{lvo.CKPT_REV}",
        "checkpoint_shards_sha256": shards,
        "loss": "next-token cross-entropy on the window's own tokens, summed over its positions",
        "precision": "fp32 weights (cast up from the checkpoint), fp32 activations, TF32 off",
        "rows_per_column": cfg.inter // G_MAX,
        "types": list(TYPES),
        "windows": rows,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "script_sha256": lvo.sha256_file(Path(__file__)),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    m = write_dir(out, arrays, cfg.layers, len(ids), meta)
    mean_loss = sum(r["loss_mean"] for r in rows) / len(rows)
    zeros = sum(r["zero_weights"] for r in rows)
    print(f"wrote {len(m['files'])} files and manifest.json to {out}; mean loss {mean_loss:.5f} nats/token; "
          f"{zeros} zero weights (the last position of each window: {WINDOWS * cfg.layers * len(TYPES) * G_MAX} "
          f"expected); total {time.time() - t_all:.0f} s")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--selftest", action="store_true")
    mode.add_argument("--run", action="store_true")
    ap.add_argument("--tokens", help="the u32 token file of bin/calibtokens (its .json beside it)")
    ap.add_argument("--out", help="output folder")
    ap.add_argument("--cache", default=os.environ.get("HF_HUB_CACHE"))
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not (a.tokens and a.out):
        ap.error("--run needs --tokens and --out")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
