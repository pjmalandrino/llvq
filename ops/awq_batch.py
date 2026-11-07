#!/usr/bin/env python3
# No PEP 723 header, like ops/awq_speed.py: this file runs inside the pinned
# vllm/vllm-openai image, whose vLLM is the thing measured. Imports are stdlib
# and vllm only, and nothing from the repository: the file is copied into a
# bucket and executed alone.
"""Aggregate decode throughput of the official AWQ 4-bit Qwen3-4B in vLLM, B sequences at a time.

The batch-1 figure (200.5 tok/s, ops/awq_speed.py) answers one user. This file answers B users: B copies of the
door prompt, decoded together by vLLM's scheduler, 256 greedy tokens each, wall clock around one `generate`
call, prefill included. Aggregate rate = B * 256 / elapsed.

Protocol, shared with the Tetra side (`bin/batchrun`):
* prompt `"The capital of France is"` as raw ids [785, 6722, 315, 9625, 374], never re-tokenized;
* 256 tokens, greedy, ignore_eos, dtype float16, prefix caching off and read back;
* 2 discarded rounds, then 5 rounds; in each round every B runs once, in the order given;
* median and min-max over the 5 rounds, per B. No ratio is formed against another stack (rule 5): the
  comparable quantity is GB/s, the weight bytes on the card times steps per second, computed downstream.

Exit codes: 0 measured; 2 refused before loading; 4 a guard failed (token count, prefix caching, engine config).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

PROMPT_IDS = [785, 6722, 315, 9625, 374]
N_TOKENS = 256
WARMUPS = 2
ROUNDS = 5
# The revision ops/awq_dequant.py's structural controls ran against (ops/awq_speed.py SIZES["4b"]).
AWQ_REPO = "Qwen/Qwen3-4B-AWQ"
AWQ_REV = "74d4bd2bd4bff9cafc9345221320bffb08b406a3"
AWQ_GB = 2.67  # weights on the card, the published table's GB column (docs/ETAT.md section 3)


class Refused(Exception):
    pass


class Violation(Exception):
    pass


def dig(root, *paths):
    for path in paths:
        cur = root
        ok = True
        for part in path.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok:
            return cur
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="1,4,8,16,32")
    ap.add_argument("--json", required=True)
    ap.add_argument("--gpu-util", type=float, default=0.5)
    args = ap.parse_args()
    try:
        tag = os.environ.get("LLVQ_IMAGE_TAG", "").strip()
        if not tag or "latest" in tag:
            raise Refused(f"LLVQ_IMAGE_TAG={tag!r}: the launcher passes a pinned tag")
        batches = [int(b) for b in args.batches.split(",")]
        if any(b < 1 or b > 256 for b in batches) or len(set(batches)) != len(batches):
            raise Refused(f"--batches {args.batches!r}: distinct counts in 1..=256")
    except (Refused, ValueError) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    import vllm

    llm = LLM(
        model=AWQ_REPO,
        revision=AWQ_REV,
        tokenizer_revision=AWQ_REV,
        dtype="float16",
        enable_prefix_caching=False,
        gpu_memory_utilization=args.gpu_util,
        max_model_len=1024,
        max_num_seqs=max(256, max(batches)),
        tensor_parallel_size=1,
        seed=0,
        disable_log_stats=True,
    )
    prefix = dig(llm, "llm_engine.vllm_config.cache_config.enable_prefix_caching",
                 "llm_engine.cache_config.enable_prefix_caching")
    quant = dig(llm, "llm_engine.vllm_config.model_config.quantization",
                "llm_engine.model_config.quantization")
    print(f"vllm {vllm.__version__}, image {tag}, {AWQ_REPO}@{AWQ_REV[:12]}, quantization {quant}, "
          f"prefix caching {prefix}")
    if prefix is not False:
        raise Violation(f"prefix caching reads {prefix!r}, expected False")

    sp = SamplingParams(temperature=0.0, top_p=1.0, n=1, max_tokens=N_TOKENS, min_tokens=N_TOKENS,
                        ignore_eos=True)

    def one(b: int) -> tuple[float, list[list[int]]]:
        prompts = [TokensPrompt(prompt_token_ids=list(PROMPT_IDS)) for _ in range(b)]
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        toks = [list(o.outputs[0].token_ids) for o in outs]
        if len(toks) != b or any(len(t) != N_TOKENS for t in toks):
            raise Violation(f"B={b}: {[len(t) for t in toks]} tokens, expected {b} x {N_TOKENS}")
        return dt, toks

    for w in range(WARMUPS):
        for b in batches:
            one(b)
    rates: dict[int, list[float]] = {b: [] for b in batches}
    rows_equal: dict[int, bool] = {}
    first: dict[int, list[int]] = {}
    for r in range(ROUNDS):
        for b in batches:
            dt, toks = one(b)
            rates[b].append(b * N_TOKENS / dt)
            rows_equal[b] = rows_equal.get(b, True) and all(t == toks[0] for t in toks)
            first[b] = toks[0]
            print(f"  round {r + 1} B={b:>3}: {b * N_TOKENS / dt:8.1f} tok/s aggregate ({dt:.3f} s)")

    out = {"engine": f"vllm {vllm.__version__}", "image": tag, "repo": AWQ_REPO, "revision": AWQ_REV,
           "quantization": str(quant), "tokens": N_TOKENS, "prompt_ids": PROMPT_IDS, "weights_gb": AWQ_GB,
           "rounds": ROUNDS, "per_batch": {}}
    print(f"\nAWQ w4g128 (vLLM, {quant}), {N_TOKENS} tokens a sequence, prompt and decode, {ROUNDS} rounds")
    print(f"{'B':>4} {'tok/s aggregate':>18} {'range':>22} {'tok/s a sequence':>17} {'GB/s':>8}  rows equal")
    for b in batches:
        xs = rates[b]
        med = statistics.median(xs)
        gbs = AWQ_GB * med / b
        same_as_b1 = first[b] == first[batches[0]]
        out["per_batch"][str(b)] = {"rates": xs, "median": med, "min": min(xs), "max": max(xs),
                                    "gb_s": gbs, "rows_equal": rows_equal[b],
                                    "row0_equals_first_batch_row0": same_as_b1, "row0_tokens": first[b]}
        print(f"{b:>4} {med:>18.1f} [{min(xs):>9.1f}-{max(xs):>9.1f}] {med / b:>17.1f} {gbs:>8.1f}  "
              f"{rows_equal[b]} (row 0 = B={batches[0]} row 0: {same_as_b1})")
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Violation as e:
        print(f"VIOLATION: {e}", file=sys.stderr)
        sys.exit(4)
