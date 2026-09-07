# Pré-enregistrement — Q5 et le volume s'additionnent-ils ?

**Écrit, commité et TAMPONNÉ le 2026-09-07, AVANT le lancement.** Go de
l'opérateur du 2026-09-07.

Ce fichier ne s'édite plus. Ce qui le corrige va dans un `-ECARTS.md`.

**Coût : ~0,45 $** (*estimé*, deux bras d'évaluation sur `l40sx1`, plafond de
timeout 25 min soit 0,75 $ au pire). Aucun encodage : le fichier existe.
Total projet avant : 119,76 $.

---

## §1 — La question, et pourquoi elle vaut mieux qu'une projection

Deux gains sont mesurés séparément, sur deux bases différentes :

```
  Q5, v_proj en int4       +3,47 pp  IC95 [+1,42 ; +5,57]   base Tetra, 53,49
  volume ×32               +2,98 pp  IC95 [+0,15 ; +5,81]   base V1c,   52,31
```

Leur somme brute fait +6,45 pp. La sous-additivité mesurée du dossier va de
0,618 à 0,907 selon qu'on empile des bras isolés ou deux groupes, ce qui projette
la combinaison entre **57,5 et 59,3 de MMLU**. Cette fourchette de 1,8 point
repose sur un facteur **jamais mesuré entre un gain de calibration et un gain de
précision** — toutes les valeurs connues viennent de bras de restauration de
précision empilés entre eux.

Et il y a une raison mécanique de croire que la sous-additivité sera **forte
ici** : le volume améliore l'estimation de H, donc la compensation GPTQ, donc
`v_proj` est déjà mieux quantifié dans V32. Restaurer ce qui est déjà bon
rapporte moins.

**Ce banc remplace la projection par une mesure, pour 0,45 $ et quinze minutes.**

## §2 — Les bras, verbatim

Le fichier scellé de V32 existe :
`hf://buckets/Pier-Jean/jobs-artifacts/volume-2026-09-07/v32/qwen3-4b-v32.bin`,
encodé à 4 194 304 tokens de calibration C4, codebook `tetra`, ρ = 1.

```
export LLVQ_MODEL=Qwen/Qwen3-4B
LLVQ_MMLU_DUMP=$OUT/b0.csv                          mmlu $F cuda 40   # V32 nu
LLVQ_RESTORE_Q4=v_proj LLVQ_MMLU_DUMP=$OUT/b1.csv   mmlu $F cuda 40   # V32 + Q5
```

**Les deux bras tournent sur `l40sx1`**, la flavor de tous les chiffres publiés,
et non sur la `rtx-pro-6000x2` qui a produit le fichier. Le bras nu est donc
rejoué plutôt que repris : cela évite un appariement entre deux cartes, et cela
donne gratuitement un contrôle de transfert entre deux cartes NVIDIA.

## §3 — Les contrôles

1. Empreinte de tokens `65dcd53655e8bfa5`, 2 280 questions, sur les deux bras.
2. Le bras nu doit rejouer **55,29** à moins de 0,5 pp ; au-delà, le transfert
   entre cartes est en cause et aucun ΔQ5 ne se lit.
3. Le bras restauré imprime **36 matrices et 94 371 840 poids**.
4. Les deux dumps écrits avec leur trailer.
5. Les deux bras diffèrent au centième.

## §4 — La règle de décision, posée AVANT de voir le chiffre

Les deux bras lisent **le même fichier**, donc la barre est **0,43 pp** et non
2,92. C'est le régime où cette mesure est nette.

Soit **G = MMLU(V32 + Q5) − MMLU(V32 nu)**, apparié, à comparer au **+3,47 pp**
que Q5 rend sur la base Tetra.

| | conséquence |
|---|---|
| **G ≥ +2,5 pp** | **Les deux gains s'empilent presque sans perte.** La projection haute tient, la combinaison se lit au-dessus de 58, et la roadmap garde Q5 et le volume comme deux lignes indépendantes. |
| **+1,0 ≤ G < +2,5** | **Sous-additivité franche, comme attendu.** La combinaison vaut ~57 à 58. Les deux lignes restent, mais tout empilement futur se chiffre avec ce facteur mesuré et non avec 0,62–0,91 emprunté ailleurs. |
| **G < +1,0** | **Le volume a absorbé l'essentiel de ce que Q5 achetait.** Le +3,47 était en partie le symptôme d'une hessienne bruitée, et non une propriété de `v_proj`. Le chantier servi de Q5 — écrivain `kind = 2`, `tv_q4_h` sur carte, une à deux semaines — doit être **ré-argumenté avant d'être écrit**, puisque son gain se paierait deux fois. |
| **autrement** — bras nu hors de 55,29 ± 0,5 | Rien n'est décidé : le transfert entre cartes est en cause et il devient le sujet. |

## §5 — Divulgation datée, et la prédiction signée

Connu : Tetra 53,49 · V1c 52,31 · V32 55,29 · Q5 sur Tetra +3,47 · volume +2,98
· `Planes14` publié 55,59 · f16 70,32 · le papier 60,7 · sous-additivité mesurée
0,618 à 0,907 sur des bras de précision.

**Prédiction de l'auteur, opposable : G entre +1,2 et +2,6 pp, centre +1,9, donc
la DEUXIÈME ligne. La combinaison atterrit entre 56,5 et 57,9.**

Motif : l'argument mécanique du §1. Une meilleure hessienne quantifie déjà mieux
`v_proj`, donc le restaurer rapporte moins que sur une base mal calibrée. Je
n'attends pas l'effondrement complet, parce que 4 bits contre 2 reste un écart
de précision que nulle hessienne ne referme.

Quatorze prédictions signées sur ce dossier, sept fausses.

**Ce qui la rendrait fausse de façon instructive : G < +1,0.** Cela dirait que
Q5 mesurait une pathologie de calibration et non une propriété de `v_proj`, que
le chantier d'ingénierie de Q5 est à rouvrir avant d'être écrit, et que la
lecture de M2 tout entière — l'attribution par type de projection — a été faite
sur un artefact mal calibré et doit être refaite.

## §6 — Ce que ce banc ne peut pas être

Une mesure servie. `LLVQ_RESTORE_Q4` déquantifie en f16 avant le produit ;
aucun octet de 4 bits n'est lu par un noyau. Et il ne dit rien du 8B ni d'un
autre tirage de calibration.
