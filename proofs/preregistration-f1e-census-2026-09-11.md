# Pré-enregistrement — F1e : le recensement MMLU par le noyau servi

Écrit avant tout lancement, tamponné au commit qui le porte (le `.ots` fait
foi). Règle dure 2. Le brouillon `proofs/BROUILLON-preregistration-tetra-q5-servi.md`
est antérieur à `LLVQ_CONFIG`, au bras noyau de `mmlu` et à `MAX_PREFILL_ROWS` ;
il est remplacé par ce document et n'est pas tamponné.

## §0 — Les quatre décisions de l'opérateur, prises le 2026-09-11

1. **L'objet de comparaison** : le bras dense sur le **même fichier mixte**,
   dans le même job. Pas le dump 56,95 existant, qui a été mesuré sur le Tetra
   pur avec `v_proj` restaurées — un autre fichier, et une paire qui porterait
   trois confusions (fichier, embedding, arithmétique).
2. **La barre** : l'appariée à fichier constant, **0,79–1,44 pp** (*mesurée*,
   `docs/mesures/mmlupair-4b-8b-2026-08-13.txt`). Pas le 1,35 par bras.
3. **Le critère d'arrêt** : un écart apparié noyau − dense au-delà de cette
   barre est un **défaut du noyau**, pas un tie-break. Le chiffre ne se publie
   pas tant qu'il n'est pas expliqué.
4. **`regroup` (a)** : après le recensement, avec son propre préreg.

## §1 — L'objet

* `qwen3-4b-tetra-q5.bin`, **1 794 564 765 octets**, sha256
  `c084a47c00f0a0509783fdbe8c544f38709fdcec4feb5ac971ec1f91909027d2`
  (`docs/mesures/tetra-q5-encodage-2026-09-09.txt`), dans le bucket à
  `/out/tetra-q5-2026-09-09/`. 216 matrices Tetra + 36 `v_proj` int4 g128.
* `configs/qwen3-4b-tetra-q5.json`, dans l'image à
  `/usr/local/share/llvq/configs/`. C'est l'autorité sur les cinq choix.
* L'image `hf.co/spaces/Pier-Jean/llvq-runner-cuda` au commit qui suit
  celui-ci, imprimé par le job sur sa ligne `NVRTC source:` (sha256 de l'unité,
  cible `compute_89`).

**Ce fichier n'a jamais scoré MMLU, sur aucun bras.** Le noyau servi n'a
jamais scoré MMLU du tout. Ce sont deux premières mesures, pas deux
reproductions.

## §2 — Ce qui est mesuré

Un job `l40sx1`, deux bras, **les mêmes 2 280 questions** (`LLVQ_MMLU_ALLOC=flat`,
40 par sujet, 57 sujets, f16), les mêmes poids, seule l'arithmétique diffère :

* bras A, **dense** : `mmlu $F cuda 40` sans `LLVQ_CONFIG` — la reconstruction
  par candle, celle de toute barre publiée ;
* bras B, **noyau** : `LLVQ_CONFIG=$C mmlu $F cuda 40` — `tv_tetra48_h`,
  `tv_tetra48_rows_h`, `tv_q4_h`, `rot_apply_rows`, embedding q8.

Chacun écrit un dump (`LLVQ_MMLU_DUMP`) qui porte `# arithmetic=` ; `mmlupair`
les apparie. `oracle` d'abord (règle 10).

**Avant ce job, un job de fumée** à quelques centimes : `LLVQ_CONFIG=$C mmlu $F
cuda 1` (la porte s'ouvre sur carte) et `LLVQ_CONFIG=$C LLVQ_PREFILL_TOKENS=203
fusedrun $F` (le garde-fou sur un paquet de queue de trois rangées, jamais
exécuté sur carte). Si l'un des deux refuse, le recensement ne part pas.

## §3 — Les prédictions signées

| # | quantité | prédiction |
|---|---|---|
| P1 | la porte servie s'ouvre sur carte au premier essai | **oui, 70 %** |
| P2 | le garde-fou à 203 jetons passe (même argmax) | **oui, 80 %** |
| P3 | bras A, dense sur le fichier mixte, micro-précision | **56,5** [55,0 ; 58,0] |
| P4 | bras B − bras A, apparié | **0,0 pp** [−1,4 ; +1,4] |
| P5 | questions discordantes entre B et A, sur 2 280 | **3 %** [1 ; 6] |
| P6 | durée du bras B | **1,65 h** [1,4 ; 2,0] |
| P7 | quelque chose refuse au premier essai du recensement | **non, 60 %** |

**P3.** 56,95 a été mesuré sur le Tetra pur restauré ; le fichier mixte a été
réencodé séquentiellement avec les `v_proj` réinjectées et sa perplexité a
bougé de ×1,3203 à ×1,334 (+1 %). MMLU bouge moins que la perplexité en
général ; je retire un demi-point et je garde un intervalle large.

**P4 est le chiffre qui compte.** Mêmes poids, même prompt, même position :
la seule différence est l'ordre d'accumulation en f16 dans les matvecs et la
rotation. Un écart au-delà de la barre appariée ne peut pas venir de là.

**P5.** La paire f16/AWQ de 2026-08-13 avait 8,9 % de discordantes sur des
poids différents ; ici les poids sont identiques.

**P6.** 1,49 h *calculées* sur la pente mesurée (3,929 ms par jeton × 1 382 608
jetons) avec la tête projetée sur la dernière rangée seulement (corrigé le
2026-09-11 dans `bin/mmlu`), plus ~10 % d'attention en N² *estimés* aux
prompts longs.

## §4 — Ce qui n'est PAS décidé ni fait ici

* La confusion embedding reste : le bras dense n'a pas de mode q8. Un
  troisième bras noyau à `"embed": "f16"` (+2,69 $) l'isolerait ; il n'est
  pas lancé, et l'écart s'écrit avec cette réserve à côté.
* `regroup` (a), le noyau int4 par lots, `PREFILL_ROWS = 8` : après.
* La sortie de `LLVQ_PREFILL_TOKENS` imprime `max |Δlogit|` sans le borner.
  La borne s'écrira à partir de la mesure de fumée, pas avant.

## §5 — Ce qui tue

* |P4| > 1,44 pp → défaut du noyau. Pas de publication, investigation.
* P3 hors intervalle → le fichier mixte n'est pas l'objet qu'on croyait
  (56,95) ; son chiffre se publie sous son propre nom, jamais comme « l'objet
  servi à 56,95 ».
* P2 échoue → le paquet de queue est faux, le recensement ne part pas.

## §6 — Coût, annoncé avant

* Fumée : ~3 min, **~0,10 $**, plafond 15 min (0,45 $ au pire).
* Recensement : bras A ~26 min (*calculé*, f1e0 §0 ter) + bras B ~1,65 h
  (*estimé*, P6) + `oracle` ≈ **2,1 h, ~3,75 $** ; `--timeout 3h`, **5,40 $ au
  pire**. Cumul de la vague 3 avant : 4,41 $ sur 20 $.

Le recensement part sur un go explicite de l'opérateur, après la fumée.
