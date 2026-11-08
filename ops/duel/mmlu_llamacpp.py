"""MMLU census of a GGUF via llama-server, stdlib only, in a dump `bin/mmlupair` reads.

The scoring rule is `ops/gguf_mmlu_thin.py`'s, unchanged: one generated position,
the top-100 log-probabilities, the four letter tokens ' A' ' B' ' C' ' D'
(362, 425, 356, 422), argmax. The same rule as `bin/mmlu` up to a monotone map.

Two differences with the thin scorer, both for pairing:

* the prompt goes to the engine as the ids `ops/duel/inputs.py` checked against
  each line's qhash, so llama.cpp tokenizes nothing;
* the dump carries the census header lines and the `# end fingerprint=...
  questions=...` trailer `bin/mmlu` writes, the fingerprint being the FNV-1a of
  the concatenated ids (a74a6d6213602979 on the census), so `mmlupair` pairs it
  with our sealed dumps question by question.

Usage:
    python3 mmlu_llamacpp.py <mmlu-prompts-ids.jsonl> <out.csv> <model label> <url> <workers>
"""

from __future__ import annotations

import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ANSWER_IDS = [362, 425, 356, 422]
N_PROBS = 100
FNV_OFFSET, FNV_PRIME, U64 = 0xCBF29CE484222325, 0x00000100000001B3, (1 << 64) - 1
ALLOC = "# alloc=flat, 100..1534 per subject, 14042 questions"


def fnv_extend(h: int, ids) -> int:
    for tid in ids:
        for b in int(tid).to_bytes(4, "little"):
            h = ((h ^ b) * FNV_PRIME) & U64
    return h


def ask(url: str, ids: list[int]) -> dict:
    body = json.dumps({"prompt": ids, "n_predict": 1, "temperature": 0.0, "n_probs": N_PROBS,
                       "cache_prompt": False}).encode()
    req = urllib.request.Request(url + "/completion", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def main(argv: list[str]) -> int:
    prompts_path, out_path, label, url, workers = argv[0], argv[1], argv[2], argv[3], int(argv[4])
    rows = [json.loads(l) for l in open(prompts_path, encoding="utf-8").read().split("\n") if l.strip()]
    h = FNV_OFFSET
    for r in rows:
        h = fnv_extend(h, r["ids"])
    print(f"  questions {len(rows)}, run fingerprint {h:016x}, {workers} slots")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        resps = list(ex.map(lambda r: ask(url, r["ids"]), rows))

    scores, missing = [], 0
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
        print(f"REFUSED: {missing}/{len(rows)} without the four letters in the top-{N_PROBS}; nothing written")
        return 2
    print(f"  four letters in the top-{N_PROBS}: {len(rows)}/{len(rows)}")

    per: dict[str, list[int]] = {}
    pop: dict[str, int] = {}
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("# llvq-mmlu-dump v1\n")
        fh.write(f"# model={label}\n# dtype=gguf\n# limit=census\n{ALLOC}\n")
        fh.write("# config=none\n# arithmetic=llama.cpp\n# kv=f16\n# scores=logprob\n# engine=llama.cpp/CUDA\n")
        fh.write("subject,index,population,qhash,answer,pick,correct,logit_a,logit_b,logit_c,logit_d\n")
        for r, sc in zip(rows, scores):
            pick = max(range(4), key=lambda k: sc[k])
            ok = int(pick == r["answer"])
            s = r["subject"]
            slot = per.setdefault(s, [0, 0])
            slot[0] += ok
            slot[1] += 1
            pop[s] = r["population"]
            fh.write(f"{s},{r['index']},{r['population']},{r['qhash']},{r['answer']},{pick},{ok},"
                     + ",".join(f"{v:.6f}" for v in sc) + "\n")
        fh.write(f"# end fingerprint={h:016x} questions={len(rows)}\n")

    micro = sum(v[0] / v[1] * pop[s] for s, v in per.items()) / sum(pop.values()) * 100
    macro = sum(v[0] / v[1] for v in per.values()) / len(per) * 100
    print(f"  MMLU micro {micro:.2f} %, macro {macro:.2f} %")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
