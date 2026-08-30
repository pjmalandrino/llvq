# Runs inside the vLLM job image, with gptqmodel installed at job time.
"""Le test qui distingue « effondré » de « cassé » — et rien d'autre.

Le bras GPTQ 2 bits rend **24,74 % de MMLU**, le hasard à quatre choix, avec une
sortie dégénérée (68,9 % de « A », écart des logits divisé par 3,4). Deux
hypothèses produisent exactement cette forme :

  (a) le 2 bits GPTQ nu s'effondre RÉELLEMENT — la prémisse du champ entier ;
  (b) le CHARGEMENT est défectueux, et le chiffre ne mesure rien.

Une génération libre tranche : du texte fluide dit (a), du charabia dit (b).

## Le contrôle est gratuit parce que la réponse f16 est déjà publiée

`ops/awq_speed.py` épingle depuis le 2026-08-17 le prompt et la continuation que
le bras f16 du 4B produit :

    "The capital of France is" -> " Paris. (True or False?)\\nThe statement …"

On n'a donc pas besoin de charger un second modèle pour avoir un témoin.
"""

from __future__ import annotations

import sys

PROMPT = "The capital of France is"
REF_F16 = (
    ' Paris. (True or False?)\n'
    'The statement "The capital of France is Paris" is **False**.'
)


def main(argv: list[str]) -> int:
    path = argv[0] if argv else "/out/qwen3-4b-gptq2"
    import torch
    from gptqmodel import GPTQModel
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    wrapper = GPTQModel.load(path, device="cuda:0")
    ids = tok(PROMPT, return_tensors="pt").to("cuda:0")
    n = ids["input_ids"].shape[1]

    # 🚨 LES DEUX CHEMINS, DANS UN SEUL PROCESSUS.
    #
    # Le premier passage a généré du charabia en appelant `.generate()` sur
    # `GPTQModel.load(...).model` — le module HF DÉBALLÉ. Si c'est le wrapper qui
    # câble les noyaux quantifiés, ce déballage les court-circuite et le modèle
    # tourne sur des poids qui ne sont pas ceux qu'on croit.
    #
    # Le charabia NE DISTINGUE PAS « effondré » de « mal chargé » : un modèle
    # réellement détruit produit le même. Ce qui les distingue, c'est de faire
    # tourner LES DEUX chemins côte à côte :
    #   wrapper fluide + .model charabia  -> notre chemin de scoring est fautif,
    #                                        et le MMLU de 24,74 % est nul ;
    #   les deux en charabia              -> deux chemins concordent, la perte
    #                                        est réelle.
    outs = {}
    for name, obj in (("wrapper GPTQModel", wrapper), (".model (déballé)", wrapper.model)):
        try:
            with torch.inference_mode():
                o = obj.generate(**ids, max_new_tokens=60, do_sample=False)
            outs[name] = tok.decode(o[0][n:], skip_special_tokens=True)
        except Exception as e:
            outs[name] = f"<<ÉCHEC: {type(e).__name__}: {e}>>"

    print("=" * 78)
    print(f"prompt              {PROMPT!r}")
    print(f"f16 (publié)        {REF_F16!r}")
    for name, text in outs.items():
        print(f"{name:19s} {text!r}")
    print("=" * 78)
    print("\nLecture, posée AVANT de regarder :")
    print("  wrapper fluide + .model charabia -> notre scorer est fautif, MMLU nul")
    print("  les deux en charabia             -> perte RÉELLE, deux chemins d'accord")
    print("  wrapper en échec                 -> non concluant, à instruire")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
