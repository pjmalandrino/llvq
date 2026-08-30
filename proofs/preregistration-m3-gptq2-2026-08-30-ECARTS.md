# Écarts au pré-enregistrement M3 — journal tenu à chaud

> **Ce fichier existe parce que le pré-enregistrement ne s'édite pas.**
> [`preregistration-m3-gptq2-2026-08-30.md`](preregistration-m3-gptq2-2026-08-30.md)
> est horodaté (sha256 `87575cd00a967973a93852df29aab5c7192a287f5b2f3a9382041f4b9ec304d4`,
> 4 attestations en attente au 2026-08-30). Une ancre atteste des **octets à une
> date** : corriger le texte pour le rendre juste détruirait ce qu'elle prouve.
> Le §7 du `CLAUDE.md` l'énonce, et le dossier porte deux cas où la règle a été
> enfreinte — les préregs du 08-10 et du 08-11, dont la version attestée n'est
> **récupérable sous aucune révision**.
>
> 🕳️ **Et elle a failli l'être une troisième fois, le jour même de la pose.** Le
> §7 du pré-enregistrement porte une table « journal des écarts, tenu à chaud »
> — donc *à l'intérieur* du document tamponné. Y écrire le premier écart a été
> tenté, puis annulé : le fichier a été restauré depuis git et son sha256
> re-vérifié contre le tampon avant tout commit. **La table du §7 est
> inutilisable par construction ; ce fichier la remplace.** C'est un défaut de
> conception du pré-enregistrement, consigné ici plutôt que corrigé là-bas.

---

## Écart n°1 — le gate §2.4 ne porte que sur MMLU

**Date** : 2026-08-30, avant le premier job.

**L'écart.** Le §2.4 fait porter le gate de l'instrument sur **deux** métriques :
MMLU à ±1,5 pp et ppl à ±2 %, chacune contre une valeur connue. À ce jour seul
le volet **MMLU** est exécutable : le bras ppl d'`ops/vllm_score.py` **refuse de
tourner** — il n'est pas câblé au corpus wikitext-2 dans la forme de
`llvq-llm/src/corpus.rs`, et il lève plutôt que de scorer dans une convention de
fenêtrage approximative.

**Pourquoi ça n'a pas été résolu avant d'écrire le pré-enregistrement.** Ça
aurait dû l'être. Le §2.4 a été écrit en supposant les deux volets équivalents à
bâtir ; le volet MMLU est piloté par un dump existant qui porte les questions,
le volet ppl demande de reproduire une tokenisation de corpus et un découpage en
fenêtres non recouvrantes. Le second est plus de travail, et ça n'a pas été vu à
la signature.

**L'effet sur les conclusions, et il n'est pas nul.** Le gate est
**partiellement exercé** : un scorer juste en MMLU et faux en ppl le passerait.

⇒ **Conséquence posée maintenant, avant tout chiffre** : **aucune perplexité
mesurée dans vLLM ne se publie** tant que le volet ppl n'est pas câblé *et*
vert sur les deux bras connus. Le bras MMLU, lui, est gaté comme prévu.

**Ce qui reste vrai du §2.4** : son critère MMLU (±1,5 pp contre 70,32 et 70,04),
sa logique — un instrument neuf ne se croit pas avant d'avoir rendu une réponse
connue — et sa conséquence si rouge : le bras GPTQ n'est pas lu.

---

## Ce qui a été vérifié à 0 $ et n'est pas un écart

[`docs/mesures/m3-qhash-local-2026-08-30.txt`](../docs/mesures/m3-qhash-local-2026-08-30.txt) :
**2 280 / 2 280 qhash reproduits** sur la machine de dev, sans GPU ni job. La
reconstruction des prompts est identique au byte à celle de `bin/mmlu`, le
dataset `cais/mmlu` n'a pas bougé depuis les dumps du 08-13, et les quatre
chaînes de réponse se tokenisent chacune en un seul token (362, 425, 356, 422).

C'est **plus** que ce que le pré-enregistrement demandait à ce stade, pas moins.
