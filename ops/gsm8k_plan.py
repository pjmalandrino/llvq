# /// script
# requires-python = ">=3.10"
# ///
"""Price the GSM8K campaign from the pilot's dumps and the L40S journals.

    uv run ops/gsm8k_plan.py docs/data/gsm8k-dumps/pilot-4b-f16-metal.jsonl \\
        docs/data/gsm8k-dumps/pilot-4b-sealed-metal.jsonl

Every number this prints is *computed* from two kinds of input, and the output
says which. *Measured*: the answer lengths of the pilot (Metal, 4B, 50
problems) and the L40S speeds of the journals named below. *Estimated*: the
prefill of the 8B and 14B through the kernel, the slowdown past 256 tokens of
context, the vLLM batched throughput, and the fixed cost of a job.

The lengths measured at 4B are applied to 8B and 14B. That is an assumption,
written on every line that uses it.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

USD_H = 1.80  # l40sx1, ops/run.py FLAVORS
N = 1319  # openai/gsm8k test split

# Served kernel, ms a generated token at 256 tokens, paper-table-2026-09-25 (measured).
DECODE_MS = {"4b": 1e3 / 113.8, "8b": 1e3 / 95.0, "14b": 1e3 / 57.2}
# Served kernel, ms a prompt token: 3.943 at 4B, f1e-census-2026-09-11 (measured);
# 8B and 14B scaled by the decode ratio (estimated).
PREFILL_MS = {s: 3.943 * DECODE_MS[s] / DECODE_MS["4b"] for s in DECODE_MS}
# Dense path at 4B, 43.0 tok/s, paper-table-2026-09-25 (measured). Prefill of
# the dense path: ~0.2 ms a token (estimated from the 4B MMLU census, 23 min).
DENSE_DECODE_MS_4B = 1e3 / 43.0
DENSE_PREFILL_MS = 0.2
# Decode slowdown past 256 tokens of context, never measured on the L40S
# (estimated): central and high.
CTX = {"central": 0.10, "high": 0.25}
# Fixed cost of a job in billed minutes: pull, load, oracle, checks (estimated
# on paper-served-*-r2, 5 to 16 min for two arms of 256 tokens).
FIXED_MIN = {"4b": 8, "8b": 8, "14b": 12}
# vLLM batched generation on one L40S, tokens a second (estimated, never
# measured in this repository), and the per-arm start (pull shared, load, graphs).
VLLM_TPS = {("4b", "f16"): 2500, ("4b", "awq"): 3500, ("8b", "f16"): 1500,
            ("8b", "awq"): 2500, ("14b", "f16"): 700, ("14b", "awq"): 1500}
VLLM_ARM_MIN = 3
VLLM_JOB_MIN = 6


def read(path: str) -> list[dict]:
    lines = [l for l in Path(path).read_text().splitlines() if l.strip()]
    rows = [json.loads(l) for l in lines[1:]]
    if not rows or "end" not in rows[-1]:
        sys.exit(f"{path}: no trailer, the run did not finish")
    return rows[:-1]


def usd(minutes: float) -> float:
    return minutes / 60 * USD_H


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        sys.exit(__doc__)
    f16, ours = read(argv[0]), read(argv[1])
    p = statistics.mean(r["n_prompt"] for r in ours)
    g_ours = statistics.mean(r["n_gen"] for r in ours)
    g_f16 = statistics.mean(r["n_gen"] for r in f16)
    cap_ours = sum(r["stop"] == "cap" for r in ours) / len(ours)
    print(f"pilot (measured, Metal, 4B, {len(ours)} problems): prompt {p:.1f} tokens, "
          f"answer FP16 {g_f16:.1f}, sealed {g_ours:.1f}, sealed at the cap {100 * cap_ours:.0f} %")
    print("applied to 8B and 14B unchanged (assumption)\n")

    total = {"central": 0.0, "high": 0.0}
    minutes = {"central": 0.0, "high": 0.0}
    print("our files, served kernel, batch 1")
    for s in ("4b", "8b", "14b"):
        line = []
        for k, pen in CTX.items():
            per = p * PREFILL_MS[s] + g_ours * DECODE_MS[s] * (1 + pen)
            m = N * per / 1e3 / 60 + FIXED_MIN[s]
            total[k] += usd(m)
            minutes[k] += m
            line.append(f"{k} {m:5.0f} min ${usd(m):5.2f}")
        cap = N * (p * PREFILL_MS[s] + 1024 * DECODE_MS[s] * (1 + CTX["high"])) / 1e3 / 60 + FIXED_MIN[s]
        print(f"  {s:>3}  " + " | ".join(line) + f" | every answer at the cap {cap:5.0f} min ${usd(cap):5.2f}")

    gate = {k: N * (p * DENSE_PREFILL_MS + g_f16 * DENSE_DECODE_MS_4B * (1 + pen)) / 1e3 / 60 + 10
            for k, pen in CTX.items()}
    print("\nengine gate: FP16 4B through our dense path, batch 1")
    print("  4b   " + " | ".join(f"{k} {m:5.0f} min ${usd(m):5.2f}" for k, m in gate.items()))
    for k in total:
        total[k] += usd(gate[k])
        minutes[k] += gate[k]

    print("\nreferences in vLLM, FP16 and AWQ, one job a size")
    for s in ("4b", "8b", "14b"):
        m = VLLM_JOB_MIN + sum(VLLM_ARM_MIN + N * g_f16 / VLLM_TPS[(s, a)] / 60 for a in ("f16", "awq"))
        print(f"  {s:>3}  {m:5.0f} min ${usd(m):5.2f}  (throughput estimated)")
        for k in total:
            total[k] += usd(m)
            minutes[k] += m

    print(f"\ncampaign: central {minutes['central'] / 60:.1f} card-hours ${total['central']:.2f}, "
          f"high {minutes['high'] / 60:.1f} card-hours ${total['high']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
