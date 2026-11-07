"""WikiText-2 perplexity of the MLX door, on M5's design, against M5's numbers.

M5 (`docs/mesures/metal-nuit-2026-11-06.txt`) read the sealed 4B through the
Rust Metal kernels, f16, expanded attention, by the decode path: the first 12
windows of 4,096 tokens of the WikiText-2 test split, each window from an empty
cache, one true token a step, 4,095 predictions a window. Pooled 12.580769.

This reads the same windows through the MLX door, by two paths:

* `decode`: one token a step from an empty cache, M5's path. It runs the one-row
  kernels, the path that writes every generated token.
* `prefill`: the window in chunks of 2,048 rows, as mlx-lm's `generate_step`
  reads a prompt. It runs the dense prefill, the path that reads every prompt.

The span is gated by its sha256 against M5's before any window runs: another
tokenizer, another split or another join would score other tokens. Each path is
compared to M5 window by window and pooled; the paired interval is a t over the
12 per-window log ratios (each window has 4,095 predictions, so the pooled
ratio is the exponential of their mean). The band is printed, not a decision.
"""
from __future__ import annotations

import math
import statistics
import time
from pathlib import Path

import mlx.core as mx

import llvq_mlx as L

# M5, the Rust Metal path (lutd, expanded, f16), decode path: raw/m5.txt of the journal.
M5_WINDOWS = [12.542946, 18.210249, 9.617751, 7.631275, 14.010765, 13.446688, 13.508824, 13.410889, 15.390806,
              9.403293, 14.426049, 13.160094]
M5_POOLED = 12.580769
M5_SPAN_SHA = "6b69a6ba6998ddda8d1f65406ac23b3e1bc09e0a693f8809d97f0ca7f4b38aab"
M5_SPLIT_TOKENS = 299078
M5_CTX, M5_NWIN = 4096, 12
BAND = 0.12  # percent, T8's band
PREFILL_CHUNK = 2048  # mlx-lm 0.28.3's prefill_step_size default
T_975 = {11: 2.200985}  # Student t, 0.975 quantile, by degrees of freedom


def span(packed: str, ctx: int, windows: int) -> tuple[list[int], int, str, int]:
    """The test split as `llvq_llm::corpus::wikitext2_test` builds it, tokenized as `metalrun ppl` does."""
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    text = "\n\n".join(pq.read_table(L.wiki_parquet()).column("text").to_pylist())
    ids = Tokenizer.from_file(str(Path(packed) / "tokenizer.json")).encode(text, add_special_tokens=False).ids
    nwin = min(len(ids) // ctx, windows)
    if nwin < 2 or ctx < 2:
        raise SystemExit(f"two windows of two tokens at least: {len(ids)} tokens")
    s = ids[: nwin * ctx]
    return s, nwin, L.sha_u32(s), len(ids)


def gate_span(digest: str, n_split: int, ctx: int, nwin: int) -> bool:
    """True when this is M5's span, so M5's numbers are the reference."""
    if (ctx, nwin) != (M5_CTX, M5_NWIN):
        print(f"design {nwin} x {ctx} is not M5's {M5_NWIN} x {M5_CTX}: no comparison to M5 is printed")
        return False
    if n_split != M5_SPLIT_TOKENS or digest != M5_SPAN_SHA:
        raise SystemExit(f"span refused: {n_split} tokens in the split, span sha256 {digest}; M5 read "
                         f"{M5_SPLIT_TOKENS} and {M5_SPAN_SHA}. Another tokenizer or split scores other tokens.")
    print(f"span gate: {n_split} tokens in the split, span sha256 {digest[:16]}, M5's")
    return True


def decode_window(model, win: list[int]) -> tuple[float, int, int, str]:
    """NLL over len(win) - 1 steps of one true token each, from an empty cache; top-1 hits; argmax digest."""
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    nll, hits, picks, pending = 0.0, 0, [], None

    def collect(p):
        nonlocal nll, hits
        loss, pick, target = p
        nll += float(loss.item())
        picks.append(int(pick.item()))
        hits += picks[-1] == target

    # The next input is the true token, never the output: step t + 1 is queued
    # before step t is read back, as generate_step overlaps its steps.
    for t in range(len(win) - 1):
        logits = model(mx.array([[win[t]]]), cache=cache)[0, -1].astype(mx.float32)
        out = (mx.logsumexp(logits) - logits[win[t + 1]], mx.argmax(logits), win[t + 1])
        mx.async_eval(out[0], out[1])
        if pending is not None:
            collect(pending)
        pending = out
    collect(pending)
    return nll, len(win) - 1, hits, L.sha_u32(picks)


def prefill_window(model, win: list[int]) -> tuple[float, int, int, str]:
    """NLL of the window read in chunks of PREFILL_CHUNK rows through one cache."""
    import numpy as np
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    nll, hits, picks = 0.0, 0, []
    n = len(win) - 1
    for s in range(0, n, PREFILL_CHUNK):
        e = min(s + PREFILL_CHUNK, n)
        logits = model(mx.array([win[s:e]]), cache=cache)[0].astype(mx.float32)
        tgt = mx.array(win[s + 1:e + 1])
        loss = mx.logsumexp(logits, axis=-1) - mx.take_along_axis(logits, tgt[:, None], axis=-1)[:, 0]
        pick = mx.argmax(logits, axis=-1)
        mx.eval(loss, pick)
        nll += float(np.asarray(loss, dtype=np.float64).sum())
        p = np.asarray(pick).tolist()
        picks += p
        hits += sum(a == b for a, b in zip(p, win[s + 1:e + 1]))
        del logits, loss
    return nll, n, hits, L.sha_u32(picks)


def run(model, packed: str, ctx: int, windows: int, paths: list[str]) -> None:
    s, nwin, digest, n_split = span(packed, ctx, windows)
    print(f"ppl: wikitext2 test, {nwin} windows of {ctx} from the start ({n_split} tokens in the split, "
          f"span sha256 {digest}); paths {', '.join(paths)}; a TOLERANCE reading against M5, never a served decision")
    against_m5 = gate_span(digest, n_split, ctx, nwin)
    for path in paths:
        score = decode_window if path == "decode" else prefill_window
        t0 = time.perf_counter()
        nlls, counts, hits = [], [], 0
        for w in range(nwin):
            nll, count, hit, picks = score(model, s[w * ctx:(w + 1) * ctx])
            nlls.append(nll), counts.append(count)
            hits += hit
            ppl = math.exp(nll / count)
            vs = f", MLX / M5 - 1 = {(ppl / M5_WINDOWS[w] - 1) * 100:+.4f} %" if against_m5 else ""
            print(f"  {path} window {w + 1:>2}: ppl {ppl:.6f}{vs}, nll {nll:.6f} over {count}, top-1 hits {hit}, "
                  f"argmax sha256 {picks[:16]}, {time.perf_counter() - t0:.0f} s elapsed", flush=True)
        pooled = math.exp(sum(nlls) / sum(counts))
        print(f"{path}: pooled ppl {pooled:.6f} over {nwin} windows, {sum(counts)} predictions, top-1 hits {hits}, "
              f"{time.perf_counter() - t0:.0f} s")
        if against_m5:
            verdict(path, nlls, counts, pooled)


def verdict(path: str, nlls: list[float], counts: list[int], pooled: float) -> None:
    logs = [n / c - math.log(m) for n, c, m in zip(nlls, counts, M5_WINDOWS)]
    k = len(logs)
    mean, sd = statistics.mean(logs), statistics.stdev(logs)
    half = T_975[k - 1] * sd / math.sqrt(k)
    pct = lambda v: (math.exp(v) - 1) * 100
    per = [pct(v) for v in logs]
    pooled_pct = (pooled / M5_POOLED - 1) * 100
    inside = abs(pooled_pct) <= BAND and all(abs(v) <= BAND for v in per)
    print(f"{path} against M5: pooled MLX {pooled:.6f} / M5 {M5_POOLED:.6f} - 1 = {pooled_pct:+.4f} %; paired t over "
          f"{k} windows {pct(mean):+.4f} % [{pct(mean - half):+.4f} ; {pct(mean + half):+.4f}]; per window "
          f"[{min(per):+.4f} ; {max(per):+.4f}] %, {sum(v > 0 for v in per)} of {k} above M5")
    print(f"{path} T8 band +-{BAND} %, pooled and every window: {'INSIDE' if inside else 'OUTSIDE'}"
          + ("" if inside else f" (windows outside: {[i + 1 for i, v in enumerate(per) if abs(v) > BAND]}, "
                                f"pooled {'inside' if abs(pooled_pct) <= BAND else 'outside'})"))
