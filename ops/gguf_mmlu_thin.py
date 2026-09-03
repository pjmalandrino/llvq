"""MMLU of a GGUF via llama-server, with NO tokenizer, NO dataset, stdlib only.

It exists for one precise reason: to measure the MMLU of IQ2_XXS **on CUDA**, in
a llama.cpp image that is C++ and has neither torch nor transformers. Installing
them for a tokenizer would cost more than the measurement.

## What makes this legitimate rather than convenient

The prompts are not rebuilt here: they are **shipped already checked**.
`mmlu-prompts.jsonl` is produced on the dev machine by the path that passed
2,280 / 2,280 qhash against the reference dumps, and it carries its `qhash` per
line. Rebuilding the prompts in a second environment would be a second chance to
diverge; shipping them guarantees they are **byte for byte** those of the Metal
run.

## The question this file answers, and that has never been asked here

Is the quality of a QUANTIZED arm invariant across backends, at identical file
and identical engine? §0 of the protocol says MMLU carries across, and that is
checked on four engines, but **all those checks were on f16**. It is the 2-bit
dequantization, whose Metal and CUDA kernels are distinct code, that could
diverge. IQ2_XXS is the only object in the dossier able to settle it: same file,
sha256 proven on both sides.
"""

from __future__ import annotations

import csv
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ANSWER_IDS = [362, 425, 356, 422]  # ' A' ' B' ' C' ' D', checked on 2026-08-30
LETTERS = "ABCD"
N_PROBS = 100


def ask(url: str, prompt: str) -> dict:
    body = json.dumps(
        {"prompt": prompt, "n_predict": 1, "temperature": 0.0, "n_probs": N_PROBS}
    ).encode()
    req = urllib.request.Request(
        url + "/completion", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def main(argv: list[str]) -> int:
    prompts_path, out_path = argv[0], argv[1]
    url = argv[2] if len(argv) > 2 else "http://127.0.0.1:8080"
    workers = int(argv[3]) if len(argv) > 3 else 4

    rows = [json.loads(l) for l in open(prompts_path, encoding="utf-8") if l.strip()]
    print(f"  questions             {len(rows)} (prompts shipped, qhash carried per line)")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        resps = list(ex.map(lambda r: ask(url, r["prompt"]), rows))

    # ALERT: same guard as on Metal. A letter missing from the top-N would make
    # an arbitrary pick. We count, and we refuse rather than return a dirty score.
    missing = 0
    scores = []
    for resp in resps:
        cp = resp.get("completion_probabilities") or []
        top = (cp[0].get("top_logprobs") or []) if cp else []
        by_id = {int(e["id"]): float(e["logprob"]) for e in top if "id" in e}
        if not all(a in by_id for a in ANSWER_IDS):
            missing += 1
            scores.append(None)
        else:
            scores.append([by_id[a] for a in ANSWER_IDS])
    if missing:
        print(f"ALERT: {missing}/{len(rows)} without the four letters in the top-{N_PROBS}")
        return 2
    print(f"  four letters in the top-{N_PROBS}: {len(rows)}/{len(rows)} OK")

    per: dict[str, list[int]] = {}
    pop: dict[str, int] = {}
    right = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("# llvq-mmlu-dump v1\n# scores=logprob\n# engine=llama.cpp/CUDA\n")
        fh.write(
            "subject,index,population,qhash,answer,pick,correct,"
            "logit_a,logit_b,logit_c,logit_d\n"
        )
        for r, sc in zip(rows, scores):
            pick = max(range(4), key=lambda k: sc[k])
            ok = int(pick == r["answer"])
            right += ok
            s = r["subject"]
            slot = per.setdefault(s, [0, 0])
            slot[0] += ok
            slot[1] += 1
            pop[s] = r["population"]
            fh.write(
                f"{s},{r['index']},{r['population']},{r['qhash']},{r['answer']},"
                f"{pick},{ok}," + ",".join(f"{v:.6f}" for v in sc) + "\n"
            )

    # MICRO, weighted by the test split population. Never Σright/Σtotal, which
    # is algebraically the MACRO when every subject carries 40 questions. That
    # defect cost $0.20 on 2026-08-30.
    micro = sum(v[0] / v[1] * pop[s] for s, v in per.items()) / sum(pop.values()) * 100
    macro = sum(v[0] / v[1] for v in per.values()) / len(per) * 100
    print(f"\n  MMLU micro            {micro:.2f} %   <-- the paper's metric")
    print(f"  MMLU macro            {macro:.2f} %")
    print(f"  raw                   {right}/{len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
