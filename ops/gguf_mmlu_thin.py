"""MMLU d'un GGUF via llama-server — SANS tokenizer, SANS dataset, stdlib seule.

Existe pour une raison précise : mesurer le MMLU d'IQ2_XXS **sur CUDA**, dans une
image llama.cpp qui est du C++ et n'a ni torch ni transformers. Les installer
pour un tokenizer coûterait plus que la mesure.

## Ce qui rend ça légitime plutôt que commode

Les prompts ne sont pas reconstruits ici : ils sont **expédiés déjà vérifiés**.
`mmlu-prompts.jsonl` est produit sur la machine de dev par le chemin qui a passé
2 280 / 2 280 qhash contre les dumps de référence, et il porte son `qhash` par
ligne. Reconstruire les prompts dans un second environnement serait une seconde
occasion de diverger ; les expédier garantit qu'ils sont **au byte** ceux du run
Metal.

## La question à laquelle ce fichier répond, et qui n'a jamais été posée ici

La qualité d'un bras QUANTIFIÉ est-elle invariante par backend, à fichier
identique et moteur identique ? Le §0 du protocole dit que MMLU traverse, et
c'est vérifié sur quatre moteurs — mais **toutes ces vérifications portaient sur
le f16**. Or c'est la déquantisation 2 bits, dont les noyaux Metal et CUDA sont
du code distinct, qui pourrait diverger. IQ2_XXS est le seul objet du dossier
capable de trancher : même fichier, sha256 prouvé des deux côtés.
"""

from __future__ import annotations

import csv
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ANSWER_IDS = [362, 425, 356, 422]  # ' A' ' B' ' C' ' D', vérifiés le 2026-08-30
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
    print(f"  questions             {len(rows)} (prompts expédiés, qhash porté par ligne)")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        resps = list(ex.map(lambda r: ask(url, r["prompt"]), rows))

    # 🚨 Même garde que sur Metal : une lettre absente du top-N ferait un pick
    # arbitraire. On compte, et on refuse plutôt que de rendre un score sali.
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
        print(f"🚨 {missing}/{len(rows)} sans les quatre lettres au top-{N_PROBS}")
        return 2
    print(f"  quatre lettres au top-{N_PROBS} : {len(rows)}/{len(rows)} ✅")

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

    # MICRO, pondéré par la population du split test — jamais Σright/Σtotal, qui
    # est algébriquement la MACRO quand chaque matière porte 40 questions. Ce
    # défaut a coûté 0,20 $ le 2026-08-30.
    micro = sum(v[0] / v[1] * pop[s] for s, v in per.items()) / sum(pop.values()) * 100
    macro = sum(v[0] / v[1] for v in per.values()) / len(per) * 100
    print(f"\n  MMLU micro            {micro:.2f} %   <-- la métrique du papier")
    print(f"  MMLU macro            {macro:.2f} %")
    print(f"  brut                  {right}/{len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
