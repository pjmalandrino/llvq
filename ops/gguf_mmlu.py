"""MMLU of a GGUF, scored the way `bin/mmlu` scores it, via llama-server and `n_probs`.

## Why not `llama-perplexity --multiple-choice`

Because it does not score the same thing. Its `multiple_choice_answers` compares
the likelihood of **whole continuations** (`tools/server/../perplexity.cpp:1331`,
"possible answers (continuations)"), while `llvq-llm/src/bin/mmlu.rs:420-432`
compares the **logits of four letter tokens** at the last position, after
"Answer: ". Two different decision rules give two different metrics, and the
published 70.32 is the one from `bin/mmlu`. Using the other one would produce a
number that compares to nothing in this dossier.

## What this file does instead

`llama-server` accepts `n_probs` on `/completion` and returns the probabilities
of the N most likely tokens at the generated position. We read those of
`' A' ' B' ' C' ' D'`, ids 362, 425, 356, 422, checked on 2026-08-30, and take
the argmax. Same decision rule as `bin/mmlu`, up to a monotone transform
(probabilities instead of logits), which does not move the argmax.

## WARNING: the trap, and it is handled rather than assumed

A top-N does not necessarily contain the four letters. A missing letter would
silently become a zero probability, so a wrong pick, another plausible and
false number. This file **counts** the questions where the four are not all
present and **refuses to return a score** if that count is not zero. That is the
lesson of §5: do not take an instrument at its word.

## The prompts are not rewritten

They come from `ops/vllm_score.py`, whose reconstruction was checked at
**2,280 / 2,280 qhash** against the reference dumps. A second implementation
would be a second chance to diverge.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("vs", HERE / "vllm_score.py")
vs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vs)

N_PROBS = 100


def ask(url: str, prompt: str) -> dict:
    body = json.dumps(
        {"prompt": prompt, "n_predict": 1, "temperature": 0.0, "n_probs": N_PROBS}
    ).encode()
    req = urllib.request.Request(
        url + "/completion", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def letter_logprobs(resp: dict, answer_ids: list[int]) -> list[float] | None:
    """The four logprobs, or None if one of the letters is missing from the top-N."""
    probs = resp.get("completion_probabilities") or []
    if not probs:
        return None
    # The field is `top_logprobs`, checked on a real response on 2026-08-30:
    #   completion_probabilities[0].top_logprobs = [{id, token, bytes, logprob}, …]
    # Guessing that name would have returned None everywhere. A loud failure,
    # not a silent one, but a failure all the same. We read it rather than
    # assume it.
    top = probs[0].get("top_logprobs") or []
    by_id = {int(e["id"]): float(e["logprob"]) for e in top if "id" in e}
    out = []
    for a in answer_ids:
        if a not in by_id:
            return None
        out.append(by_id[a])
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--label", required=True, help="what the dump will name it")
    ap.add_argument("--reference-dump", default="docs/data/mmlu-dumps/mmlu-4b-f16.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    answer_ids = [tok.encode(x, add_special_tokens=False)[0] for x in vs.ANSWER_STRINGS]
    print(f"  answer tokens         {answer_ids}")

    wanted, head = vs.read_reference_dump(Path(args.reference_dump))
    test = vs.load_mmlu("test", "main")
    dev = vs.load_mmlu("dev", "main")
    populations: dict[str, int] = {}
    for it in test:
        populations[it["subject"]] = populations.get(it["subject"], 0) + 1

    prompts, items, bad = vs.build_prompts(wanted, test, dev, tok)
    if bad:
        raise SystemExit(f"ALERT: {len(bad)} qhash diverge, nothing is scored")
    print(f"  qhash checked         {len(wanted)}/{len(wanted)} OK")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        resps = list(ex.map(lambda p: ask(args.url, p), prompts))

    scores, missing = [], 0
    for r in resps:
        lp = letter_logprobs(r, answer_ids)
        if lp is None:
            missing += 1
            lp = [0.0, 0.0, 0.0, 0.0]
        scores.append(lp)

    if missing:
        raise SystemExit(
            f"ALERT: {missing} questions out of {len(scores)} do not have the four "
            f"letters in the top-{N_PROBS}. A pick would be arbitrary. NOTHING SCORED."
        )
    print(f"  four letters in the top-{N_PROBS}: {len(scores)}/{len(scores)} OK")

    right, total, micro, macro = vs.emit_dump(
        Path(args.out), args.arm, args.label, wanted, items, scores,
        populations, None, "llama.cpp",
    )
    print(f"\n  MMLU micro            {micro:.2f} %   <-- the paper's metric")
    print(f"  MMLU macro            {macro:.2f} %")
    print(f"  raw                   {right}/{total}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
