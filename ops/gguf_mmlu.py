"""MMLU d'un GGUF, scoré comme `bin/mmlu` — via llama-server et `n_probs`.

## Pourquoi pas `llama-perplexity --multiple-choice`

Parce qu'il ne score pas la même chose. Son `multiple_choice_answers` compare la
vraisemblance des **continuations entières** (`tools/server/../perplexity.cpp:1331`,
« possible answers (continuations) »), quand `llvq-llm/src/bin/mmlu.rs:420-432`
compare les **logits de quatre tokens-lettres** à la dernière position, après
« Answer: ». Deux règles de décision différentes donnent deux métriques
différentes, et le 70,32 publié est celle de `bin/mmlu`. Employer l'autre
produirait un nombre qui ne se compare à rien de ce dossier.

## Ce que ce fichier fait à la place

`llama-server` accepte `n_probs` sur `/completion` et rend les probabilités des
N tokens les plus probables à la position générée. On y lit celles de
`' A' ' B' ' C' ' D'` — ids 362, 425, 356, 422, vérifiés le 2026-08-30 — et on
prend l'argmax. Même règle de décision que `bin/mmlu`, à la transformation
monotone près (probabilités contre logits), qui ne déplace pas l'argmax.

## 🚨 Le piège, et il est traité plutôt que supposé

Un top-N ne contient pas forcément les quatre lettres. Une lettre absente
deviendrait silencieusement une probabilité nulle, donc un pick faux — encore un
nombre plausible et faux. Ce fichier **compte** les questions où les quatre ne
sont pas toutes présentes et **refuse de rendre un score** si le compte n'est pas
nul. C'est la leçon du §5 : ne pas croire un instrument sur parole.

## Les prompts ne sont pas réécrits

Ils viennent de `ops/vllm_score.py`, dont la reconstruction a été vérifiée à
**2 280 / 2 280 qhash** contre les dumps de référence. Une seconde
implémentation serait une seconde occasion de diverger.
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
    """Les quatre logprobs, ou None si l'une des lettres manque au top-N."""
    probs = resp.get("completion_probabilities") or []
    if not probs:
        return None
    # Le champ est `top_logprobs`, vérifié sur une réponse réelle le 2026-08-30 :
    #   completion_probabilities[0].top_logprobs = [{id, token, bytes, logprob}, …]
    # Deviner ce nom aurait rendu None partout — échec bruyant, pas silencieux,
    # mais échec quand même. On le lit plutôt qu'on ne le suppose.
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
    ap.add_argument("--label", required=True, help="ce que le dump nommera")
    ap.add_argument("--reference-dump", default="docs/data/mmlu-dumps/mmlu-4b-f16.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    answer_ids = [tok.encode(x, add_special_tokens=False)[0] for x in vs.ANSWER_STRINGS]
    print(f"  tokens de réponse     {answer_ids}")

    wanted, head = vs.read_reference_dump(Path(args.reference_dump))
    test = vs.load_mmlu("test", "main")
    dev = vs.load_mmlu("dev", "main")
    populations: dict[str, int] = {}
    for it in test:
        populations[it["subject"]] = populations.get(it["subject"], 0) + 1

    prompts, items, bad = vs.build_prompts(wanted, test, dev, tok)
    if bad:
        raise SystemExit(f"🚨 {len(bad)} qhash divergent — rien n'est scoré")
    print(f"  qhash vérifiés        {len(wanted)}/{len(wanted)} ✅")

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
            f"🚨 {missing} questions sur {len(scores)} n'ont pas les quatre lettres "
            f"dans le top-{N_PROBS}. Un pick y serait arbitraire. AUCUN SCORE RENDU."
        )
    print(f"  quatre lettres au top-{N_PROBS} : {len(scores)}/{len(scores)} ✅")

    right, total, micro, macro = vs.emit_dump(
        Path(args.out), args.arm, args.label, wanted, items, scores,
        populations, None, "llama.cpp",
    )
    print(f"\n  MMLU micro            {micro:.2f} %   <-- la métrique du papier")
    print(f"  MMLU macro            {macro:.2f} %")
    print(f"  brut                  {right}/{total}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
