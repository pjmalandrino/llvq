# Plan d'action — bilan de ce qui est implémentable (2026-08-05)

> Consolidation des deux notes d'exploration vérifiées
> ([`pistes-format-vram-2026-08-05.md`](pistes-format-vram-2026-08-05.md) et
> [`pistes-facteurs-cles-2026-08-05.md`](pistes-facteurs-cles-2026-08-05.md))
> et du chantier en cours (branche `noyau-cuda`). Hypothèse de départ : **on
> repart de `main` avec les dernières versions** — le lot A commence donc par
> commiter/merger le travail de la branche, et les deux notes de pistes (non
> suivies) avec.
>
> Échelle de complexité : 1 = paramètre/relance · 2 = quelques lignes ou
> outillage existant · 3 = chantier borné (transcodeur + noyau + tests) ·
> 4 = chantier profond (grade « ~25 mutants ») · 5 = plusieurs semaines.
> Garde-fous permanents : budget GPU annoncé avant et plafonné · go explicite
> avant tout lancement · une variable à la fois · **B1 (graines) est
> bloquant pour tout verdict qualité**.

## Vue d'ensemble

| # | sujet | complexité | apport (facteur → grandeur) | coût machine | dépend de |
|---|---|---|---|---|---|
| **A1** | Brancher le noyau fusé dans `bin/run`, 1er run carte | 2 | vitesse : **+19-23 %** bout-en-bout (42,8 → ~51-53 tok/s) | ~0,1-0,5 $ | — (plomberie écrite) |
| **A2** | Fix `repeat_kv().contiguous()` (cache recopié 4×/bloc/token) | 1 | vitesse : part de l'overhead, à mesurer | inclus A1 | A1 |
| **A3** | CUDA Graph (+ `new_stream`, contrôle 3 bras) | 2 | vitesse : attaque les **~11 ms/token** de dispatch (les 48 %) — fait sauter le plafond ~1,28× | ~0,1-0,3 $ | A1 |
| **A4** | Campagne 3 bras LLVQ / AWQ / f16 (prête, jamais lancée) | 1 | **le chiffre publiable** + l'adversaire enfin mesuré dans notre harnais | plafonné, à annoncer | A1 |
| **B1** | ⚠️ Gate S1 : 3 graines + damping (6 runs de 3 blocs) | 1 | méthode : la barre d'erreur sans laquelle aucun A/B qualité n'est interprétable | 0 $, ~1 h Mac | — |
| **B2** | Oracle calibration (calibrer sur wikitext-test) | 1 | qualité : le **plafond** de toute la famille calibration | 0 $, 16 min | — |
| **B3** | Courbe volume 131k→500k→2M (zéro code) | 1 | qualité : la pente du levier volume avant tout dollar | 0 $, ~30 min | B1 |
| **B4** | ppl+MMLU des artefacts e4/e8 existants | 1 | froid : décide **−365 / −559 Mo** (−21/−32 % du fichier scellé) | 0 $, ~2 h Mac | — |
| **B5** | Histogramme par classe dans `rtbits` | 2 | prérequis : tranche la fourchette E2 (2,92-3,05 vs 3,2-3,4) | 0 $, ~30 min | — |
| **B6** | Δppl du plafond L≤4 (swap transcodé sur le scellé) | 2 | qualité : convertit le « −0,26 pt gaussien » en vraie perplexité ; conditionne C2 | 0 $, Mac | — |
| **C1** | **E1a : 6ᵉ layout plans binaires AoS-14** + noyau + banc | 3 | VRAM : **5,51 → 4,80** b/poids (−13 %) **sans aucune perte** ; à 70B ~50 → 44 Go ; ET le test qui décide de toute l'échelle format | ~0,1-0,2 $ | A1 (même infra banc) |
| **C2** | E1b (12 o) + overlay épars batché | 3 | VRAM : **4,14-4,36** — passe sous le q4 (38,7 Go à 70B) | ~0,3 $ | C1, B6 |
| **C3** | E2 : étage Golay/XOR | 4 | VRAM : **~3,06** — 70B ~29 Go, sous les 32 Go | à chiffrer | C1, B5 |
| **D1** | Design C (rétraction libre + résolution close) | 4 | MMLU : **+1,9 à 3,3 pp** chiffrés Table 9 — le suspect n°1 du −4,8 pp | run complet ~8-10 $/point | B1 |
| **D2** | Run calibration ×100 (bf16, DCLM ~30 lignes) | 2 | qualité : inconnu, borné par B2 | **20-27 $** | B1, B2, B3 favorables |
| **D3** | leech2c11 (2 bits de gain, iso-débit 48 bits) | 1 | qualité : signal mixte au papier (wiki ↑, MMLU ↓) | ~4 h + MMLU | B1 ; basse priorité |
| **D4** | FT échelles par colonne (méthode du papier) | 5 | ppl : 17,05 → **9,26** au papier ; MMLU +2,1 pp seulement ; ⚠️ repositionne face à QTIP-FT | GPU, à chiffrer | plus tard, décision de positionnement |
| **E1** | Chemin d'exécution int8 embedding/lm_head (gather + matvec) | 3 | VRAM −0,39 Go + vitesse +2,6-3,1 % sur le 4B | ~0,1 $ | B4 favorable |
| **E2** | KV int8 (note produit 70B) | — | VRAM 70B : −1,3 Go à 8k ; hors périmètre 4B | — | plus tard |

## La séquence recommandée

**Semaine 1 — refermer, mesurer gratuit.**
1. **A1** : commiter la branche, brancher, premier run carte. C'est la
   priorité actée et tout le lot A/C en dépend.
2. En parallèle sur le Mac, **tout le lot B** (B1 d'abord) : six mesures,
   0 $, qui déverrouillent ou tuent la moitié du plan. B4 est le meilleur
   rapport information/coût de tout le tableau.
3. **A2, A3** dans la foulée du branchement, puis **A4** : la campagne rend
   le chiffre publiable ET l'adversaire mesuré.

**Semaine 2 — le point de décision.**
4. **C1** (E1a) : un seul essai discriminant, qualité strictement identique.
   S'il rend ≥ 0,95× Slot32 : toute l'échelle format s'ouvre (C2, puis C3) et
   le verdict « face au 4 bits » se renverse. S'il échoue : l'échelle
   s'arrête proprement, on aura dépensé 0,20 $.
5. Selon B1-B3 : engager **D1** (design C) — le seul levier MMLU chiffré —
   et/ou **D2** si l'oracle montre du plafond disponible.

**Ce qu'on ne fait pas** (fermé par les vérifications) : rotation de sortie,
codage entropique du froid, décodage spéculatif sur le 4B, lm_head Slot32,
et tout réglage noyau gelé par l'audit (bancs/padding, gather, table shared…).

## Lecture par facteur

| facteur | ce que le plan peut rendre | par quoi |
|---|---|---|
| volume à froid | −21 à −32 % du fichier scellé | B4 → embedding int8/int4 |
| VRAM | 5,51 → 4,80 (gratuit) → ~4,2 → ~3,4 → ~3,06 b/poids ; 70B : 50 → 44 → 39 → 34 → 29 Go | C1 → C2 → C3 |
| vitesse | +19-23 % (branchement), puis le dispatch (~48 %) via Graph, +3 % lm_head int8 | A1 → A3 → E1 |
| perplexité | plafond mesuré par B2 ; puis D1/D2 ; D4 en réserve (levier n°1 du papier) | B2 → D1/D2 → D4 |
| MMLU | +1,9-3,3 pp (design C) ; EoRA/Recover-LoRA en réserve documentée | D1, puis P15/P16 du dépôt |

## Rappels de méthode (payés cher, ne pas ré-apprendre)

- Aucun verdict MMLU sans run complet scellé (~4 h/point) — les A/B 3 blocs
  ne voient que la perplexité, et ils ont déjà inversé un signe (group_scales).
- Toute affirmation de vitesse se tranche sur carte (`LLVQ_KERNEL_DIR`,
  ~0,1 $), jamais sur un compte niveau source — il vient d'échouer d'un
  facteur 2.
- Une seule comptabilité par tableau (payload vs thesis vs canonique 70B),
  et la convention canonique (embedding f16 inclus) pour tout chiffre VRAM.
