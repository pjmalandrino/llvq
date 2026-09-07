# Pré-enregistrement — le volume de calibration porte-t-il les 5,1 points ?

**Écrit, commité et TAMPONNÉ le 2026-09-07, AVANT le lancement.** Go de
l'opérateur du 2026-09-07. Ligne 6 de
[`docs/ROADMAP-QUALITY.md`](../docs/ROADMAP-QUALITY.md), promue par le chantier 3.

Ce fichier ne s'édite plus. Ce qui le corrige va dans un `-ECARTS.md`.

**Coût : ~19 $** (*estimé*), deux jobs `rtx-pro-6000x2` en parallèle, plafonds
de timeout à 180 et 100 min soit **25,7 $ au pire**. Aucun plafond de vague en
vigueur (opérateur, 2026-09-06). Total projet avant : 100,21 $.

---

## §1 — La question, et pourquoi le chantier 3 la désigne

Le chantier 3 a innocenté le codebook cette nuit : `leech0c13`, la configuration
LLM du papier, rend 19,61 de perplexité et 54,67 de MMLU chez nous, contre 16,16
et 53,49 pour Tetra, **à débit identique**. Les trois bras maison sont
indiscernables en MMLU pendant que leurs perplexités couvrent 21,5 %.

Il reste deux causes possibles aux 5,1 points d'écart au papier : **le volume de
calibration** et la composition du corpus. Ce banc mesure la première.

L'argument physique, et il est chiffré. La hessienne est `d_in × d_in`. Avec nos
131 072 tokens :

```
  q, k, v, o, gate, up   d_in = 2 560   →  51 échantillons par dimension
  down                   d_in = 9 728   →  13,5
```

**13,5 échantillons par dimension pour estimer une matrice 9 728 × 9 728**, et
`down_proj` fait 24,7 % des poids. À ×96 cela devient 1 295. Preuve indirecte au
dossier : M1 mesure que **rétrécir H rapporte −31 % de perplexité** au 0,6B, et
rétrécir H est exactement le correctif d'une hessienne mal estimée.

## §2 — Les bras, verbatim

Image `hf.co/spaces/Pier-Jean/llvq-runner-cuda`, dont le chemin d'encodage est
**identique au code qui a produit le Tetra du 2026-09-06** : aucun commit sur
`smoke.rs`, `calib.rs`, `tetra/`, `quantizer.rs` ni `llvq-artifact/src/` entre
le début de ce run et la construction de l'image (*vérifié*, git).

```
V96 :  smoke 6144 2048 12 4096 cuda nogs tetra 999 rot   # 12 582 912 tokens
V1c :  smoke   64 2048 12 4096 cuda nogs tetra 999 rot   #    131 072 tokens
```
`LLVQ_CALIB=c4` des deux côtés, puis `seal`, `ppl`, et `mmlu` avec
`LLVQ_MMLU_DUMP`. Deux jobs `rtx-pro-6000x2` en parallèle.

**V1c est le témoin de device, et il n'est pas un point de courbe.** Il porte
exactement les réglages du Tetra du 09-06 ; seul le device change. Sans lui,
l'effet de volume serait mélangé au confondant jamais mesuré du §5 du journal
du 8B : `calib.rs` accumule AᵀA **en f32 sur l'accélérateur**, et Metal n'est
pas CUDA.

Le shard de calibration C4 est **vérifié** : 45 576 documents, 97,9 M
caractères, environ 24,5 M tokens pour 12,58 demandés. Le bras ×96 ne relit
donc pas les mêmes tokens.

## §3 — Les contrôles, et si l'un tombe aucun chiffre n'est publié

1. Empreinte de tokens `65dcd53655e8bfa5`, 2 280 questions, sur les deux bras.
2. `verify_artifact` relit les 3 633 315 840 poids bit pour bit.
3. Débit effectif **2,0702 bits/poids** sur les deux bras, comme Tetra : le
   volume ne touche pas le format. Un écart ici invaliderait le banc.
4. Les dumps par question écrits avec leur trailer.
5. V1c doit rendre une perplexité et un MMLU **proches** du Tetra du 09-06
   (16,1569 et 53,49). C'est ce que le §5 lit comme effet de device.
6. La perplexité mesurée en cours d'encodage doit prédire celle de la carte à
   moins de 0,5 % — elle l'a fait trois fois (4B, 8B, `leech0c13` à 0,08 %).

## §4 — La règle de décision, posée AVANT de voir le chiffre

Ces bras **réencodent**, donc la barre est **2,92 pp de MMLU** et 5,2 % de
perplexité, et non 0,43. Un seul tirage par bras.

Soit **ΔV = MMLU(V96) − MMLU(V1c)**, apparié sur les mêmes questions, et
**ΔD = MMLU(V1c) − 53,49** l'effet de device.

| | conséquence |
|---|---|
| **ΔV ≥ +3,0 pp et IC95 > 0** | **Le volume porte une part large des 5,1 points.** La calibration prend la tête de la roadmap qualité, une courbe à trois ou quatre points devient rentable, et le volume servi passe de 131 072 tokens à ce que la courbe désigne. |
| **+1,0 ≤ ΔV < +3,0** | **Effet réel mais petit, ou bruit : ce banc ne les distingue pas.** Rien ne bouge sur un tirage. Une courbe à trois points et deux graines précède toute décision, et son coût se pose alors en connaissance de cause. |
| **ΔV < +1,0** | **Le volume ne porte pas l'écart à cette échelle.** La composition du corpus reste seule en lice, et la tête de la roadmap va aux finetunings, ligne 14 et suivantes. |
| **autrement** — `\|ΔD\| > 2,0 pp`, ou débit hors de 2,0702 | **Rien n'est décidé sur le volume.** Un effet de device de cette taille dominerait la lecture, et il deviendrait lui-même le sujet : tout écart Mac/carte du dossier serait à reprendre, à commencer par le 8B. |

## §5 — Divulgation datée, et la prédiction signée

Connu à la signature : Tetra 53,49 / 16,1569 · `Planes14` publié 55,59 / 16,9422
· `leech0c13` 54,67 / 19,6093 · f16 70,32 / 12,2369 · le papier 60,7 / 17,05 ·
au 0,6B sur 3 blocs, ×13 de volume rend −1,2 % de perplexité et l'oracle
−1,6 % · un bras se déplace de 13,9 % en ne changeant que le texte de
calibration · M1 : rétrécir H rend −31 % de perplexité au 0,6B.

**Prédiction de l'auteur, opposable : ΔV entre 0 et +4 pp, centre +1,5, donc la
DEUXIÈME ligne du §4. Perplexité −2 à −8 %. ΔD sous 1 pp en valeur absolue.**

Motif : l'argument des 13,5 échantillons par dimension est solide et devrait
produire quelque chose ; mais le chantier 3 vient de mesurer qu'un déplacement
de 21,5 % de perplexité produit **zéro** point de MMLU, et rien ne dit que
celui-ci fera mieux.

Ce que vaut cette prédiction : treize signées sur ce dossier, sept fausses.

**Ce qui la rendrait fausse de façon instructive : ΔV ≥ +3,0 pp.** Cela voudrait
dire que nous quantifions depuis le début sur une hessienne trop bruitée, que le
correctif est un paramètre déjà livré, et que les cinq points se rachètent en
tokens de calibration et non en ingénierie de format. Ce serait le résultat le
moins cher du dossier.

## §6 — Ce que ce banc ne peut pas être

Une courbe. Deux points ne donnent pas de forme, et l'opérateur a explicitement
reporté la courbe. Ce banc répond à « le volume produit-il un GROS effet », pas
à « combien exactement ».

Et il ne dit rien de la composition du corpus, qui reste l'autre cause en lice
et qui demande son propre bras.
