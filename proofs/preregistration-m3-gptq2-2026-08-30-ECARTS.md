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

## Écart n°2 — le gate a été lu sur la MACRO, et le micro est recalculé hors ligne

**Date** : 2026-08-30, job `6a93edf1984507d9db4ecbf1`, 0,20 $.

**L'écart.** `ops/vllm_score.py` imprimait `Σright/Σtotal` en l'appelant « MMLU
micro ». Avec exactement 40 questions par matière, cette division est
**algébriquement la macro** — le défaut que le §3ter du `CLAUDE.md` a identifié
le 2026-08-01 et corrigé dans `bin/mmlu`. Le gate a donc été évalué sur 72,85
contre une valeur connue qui est un micro, et il est passé au rouge.

**Ce que ça change au verdict.** Rien, une fois le dump lu. Le dump par question
a survécu au conteneur, et le micro s'en recalcule :

| | macro | micro |
|---|---|---|
| candle (référence 08-13) | 72,76 % | **70,32 %** |
| vLLM (ce job) | 72,85 % | **70,36 %** |

⇒ **|Δ| = 0,04 pp** contre un critère de 1,5 pp. Le volet f16 du gate §2.4 est
**vert**, sur la métrique que le §2.4 nomme.

🚨 **Et il faut être scrupuleux sur ce que « vert » veut dire ici.** Le job a
imprimé un nombre rouge ; le vert est **recalculé hors ligne** à partir des
lignes du dump. Ce n'est pas un sauvetage post-hoc : l'agrégation est une
fonction pure du dump, et le protocole exige un dump par question **précisément
pour que l'agrégation reste rejouable**. Mais le job, lui, a échoué, et son
journal le dit.

**Ce que le gate a réellement acheté.** Une erreur **réelle**, attrapée dès son
premier usage, avant qu'un seul chiffre GPTQ existe. Sans lui, la campagne aurait
comparé des MMLU **macro** côté vLLM à des MMLU **micro** côté maison — un
échange qui vaut ~1 pp sur la baseline mais **frappe plus fort le bras
quantifié** (§3ter), donc qui aurait faussé la comparaison **dans notre sens**.

**Fait non demandé, obtenu en prime.** Deux moteurs entièrement distincts —
candle + notre noyau CUDA, et vLLM — deux chemins numériques distincts (logits
f16 contre logprobs) s'accordent sur **2 272 / 2 280 picks, soit 99,65 %**, à
qhash identiques sur les 2 280.

**Correction.** `vllm_score.py` rend désormais le micro **et** la macro,
étiquetés ; le gate porte sur le micro.

**Reste dû** : le bras `awq_marlin` n'a jamais tourné — le job s'est arrêté au
premier gate rouge. La moitié du gate §2.4 est encore à passer.

---

## Ce qui a été vérifié à 0 $ et n'est pas un écart

[`docs/mesures/m3-qhash-local-2026-08-30.txt`](../docs/mesures/m3-qhash-local-2026-08-30.txt) :
**2 280 / 2 280 qhash reproduits** sur la machine de dev, sans GPU ni job. La
reconstruction des prompts est identique au byte à celle de `bin/mmlu`, le
dataset `cais/mmlu` n'a pas bougé depuis les dumps du 08-13, et les quatre
chaînes de réponse se tokenisent chacune en un seul token (362, 425, 356, 422).

C'est **plus** que ce que le pré-enregistrement demandait à ce stade, pas moins.
