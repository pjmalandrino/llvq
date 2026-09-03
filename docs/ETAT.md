# État du projet au 2026-09-02

## 1. Le projet

LLVQ quantifie les poids d'un LLM à 2 bits sur le réseau de Leech, en Rust. Le but est de faire tenir de plus gros
modèles sur du matériel local. Le dépôt porte le quantifieur, le format de fichier et un noyau CUDA fusé qui décode et
multiplie sans repasser par f16. Trois Qwen3 sont scellés et servis : 4B, 8B, 14B. Le manuscrit est sur Zenodo (DOI
10.5281/zenodo.22133606). Le papier a été soumis à arXiv en sources le 2026-09-02 (commit `e721bc5`) ; l'opérateur le
rapporte publié, et le dépôt n'en porte pas encore l'identifiant.

## 2. Configuration servie v1

La configuration servie est `planes14` + `LLVQ_EMBED=q8` + `LLVQ_ROT_SHARE=1` + `LLVQ_FUSE=1`, aux trois tailles, sur
L40S.

| taille | tok/s [plage] | Go carte | b/param modèle entier | ppl wikitext (× f16) | MMLU micro (f16 → LLVQ) |
|---|---|---|---|---|---|
| 4B | 100,6 [99,9–100,7] | 2,57 | 5,162 | 16,94 (×1,3845) | 70,32 → 55,59 |
| 8B | 75,5 [75,5–75,6] | 5,41 | 5,322 | 10,97 (×1,2201) | 76,08 → 65,52 |
| 14B | 46,8 [46,7–46,8] | 9,40 | 5,106 | 9,49 (×1,1894) | 78,97 → 72,12 |

Débits et Go : *mesurés*, médianes de 5 rounds, Go en compte d'octets hôte
([vague2-fusion-8b-14b-2026-08-31.txt](mesures/vague2-fusion-8b-14b-2026-08-31.txt),
[d1-fusion-servie-2026-08-24.txt](mesures/d1-fusion-servie-2026-08-24.txt)). b/param : *calculés* sur octets mesurés,
embedding q8 à 8,5 b/param ([rtbits-14b-2026-08-17.txt](mesures/rtbits-14b-2026-08-17.txt)). Qualité : *mesurée* sur
le fichier scellé, empreintes `3f1baca9033bf251` et `65dcd53655e8bfa5`
([a4-campagne-2026-08-06.txt](mesures/a4-campagne-2026-08-06.txt),
[campagne-8b-qualite-2026-08-08.txt](mesures/campagne-8b-qualite-2026-08-08.txt),
[campagne-14b-qualite-2026-08-10.txt](mesures/campagne-14b-qualite-2026-08-10.txt)).

Le chemin dense f16 rend 43,5 / 26,4 / 17,0 tok/s dans 8,04 / 16,38 / 29,54 Go (*mesuré*,
[b2-fusedrun-plages-2026-08-18.txt](mesures/b2-fusedrun-plages-2026-08-18.txt)).

## 3. Concurrents au 4B

| bras | disque Go | b/param modèle entier | MMLU micro | ppl (× f16 de sa pile) |
|---|---|---|---|---|
| f16 | 8,04 | 16,0 | 70,32 | ×1 |
| AWQ w4 g128, officiel Qwen | 2,67 | 5,302 | 70,04 | ×1,105 |
| LLVQ 2 bits, `Planes14` + q8 | 1,77 (1,41 en int8) | 5,162 | 55,59 | ×1,3845 |
| IQ2_XXS, llama.cpp Metal | 1,25 (*mesuré*, 1 246 620 832 o) | 2,479 | 39,39 | ×2,6287 |

Disque : *mesuré*, ×4,54 sur f16 ([fiche-4b.md](fiche-4b.md)). Au 4B LLVQ gagne le disque et la mémoire, perd la
qualité. Ni le débit ni la mémoire de l'AWQ ne se lisent dans notre harnais : il y est déquantifié en f16. Son b/param
vaut dans son moteur. AWQ et f16 : *mesurés*, même harnais, même empreinte (a4-campagne). IQ2_XXS : *mesuré* dans sa pile
([m3-iq2-metal-2026-08-30.txt](mesures/m3-iq2-metal-2026-08-30.txt)) ; son MMLU traverse les moteurs à 0,52 pp près
(*mesuré*, m3-iq2-metal), pas sa perplexité. Écarts appariés : LLVQ perd 14,45 pp [11,60 ; 17,27] sur l'AWQ (*calculé*,
[mmlupair-4b-8b-2026-08-13.txt](mesures/mmlupair-4b-8b-2026-08-13.txt)) et gagne 16,20 pp [12,64 ; 19,72] sur IQ2_XXS
(*calculé*, m3-iq2-metal). L'écart à l'AWQ vaut 7,49 pp au 8B et 6,09 pp [3,62 ; 8,52] au 14B
([mmlupair-14b-2026-08-17.txt](mesures/mmlupair-14b-2026-08-17.txt)). En mémoire nous sommes sous l'AWQ officiel aux
trois tailles : −2,6 %, −10,6 %, −5,5 % (*calculé*, [rtbits-14b-2026-08-17.txt](mesures/rtbits-14b-2026-08-17.txt)).
Repère papier, Table 6, 4B sans fine-tuning : LLVQ 0 bit 17,05 ppl et 60,7 % MMLU, QTIP 17,04 et 57,4 (*mesurés* par le
papier, [llvq-paper-notes.md](llvq-paper-notes.md)). En excès de log-vraisemblance nous sommes 2,6 % pires que QTIP
(0,3254 contre 0,3171 nats, f16 sur le fichier scellé, *calculé*, [fiche-4b.md](fiche-4b.md) §3.1) ; le déficit de
5,1 pp au 60,7 du papier n'est pas expliqué. Contre nous : 131 k tokens de calibration contre 6 100 séquences, et la
rotation d'entrée seule.

## 4. Faits de structure

La recette servie est l'Algorithme 1 (shape-gain, reset de gain) plus une rotation d'incohérence en entrée. La
rétraction de l'Eq. 17 est un no-op sous un gain codé et l'Algorithme 3 (`group_scales`) est désactivé ; « Spherical
GPTQ » nomme le crate `llvq-quant`, pas la recette ([fiche-4b.md](fiche-4b.md) §2.3).

Le format déplie 4,804 b/poids en VRAM (*mesuré* au banc,
[e2-golay70-bench-2026-08-07.txt](mesures/e2-golay70-bench-2026-08-07.txt)) pour 2,1595 b/poids écrits dans le fichier
scellé, queue incluse, sur 3 633 315 840 poids de projections (*calculé*, `bin/seal`).

Le plancher `nullk` est celui de notre géométrie de lancement, pas de la carte. Il vaut 2,306 ms pour 252 projections
sans lire un poids, 4,77× f16 (*mesuré*, [f2-p3-qtip-banc-2026-08-21.txt](mesures/f2-p3-qtip-banc-2026-08-21.txt)).
QTIP finit les mêmes projections en 2,246 ms, à 4,89× [4,89–4,90], en lisant 0,91 Go. `Planes14` y met 5,103 ms pour
2,18 Go ; le rapport vaut 2,27× [2,27–2,28], proche du rapport de trafic 2,40×. La grandeur comparable est les Go/s
(405 contre 428). Ces × sont L40S : sur A100 aucun de nos bras ne bat f16 (*mesuré*,
[f4-a100-2026-08-18.txt](mesures/f4-a100-2026-08-18.txt)).

Le tirage des fenêtres de calibration vaut σ = 5,2 % de perplexité sur trois runs complets du 4B : 16,7425 / 15,8836 /
15,1027 (*mesuré*, [f5-graines-4b-2026-08-19.txt](mesures/f5-graines-4b-2026-08-19.txt)), étendue 10,3 % (*calculé*).
En MMLU il vaut 2,92 pp (étendue 5,83 pp ; *mesuré*,
[bruit-mmlu-graines-4b-2026-08-25.txt](mesures/bruit-mmlu-graines-4b-2026-08-25.txt)). La courbe d'échelle
compare des objets calibrés à l'identique : les artefacts 4B, 8B et 14B ont tous tourné sans graine, sur le même
préfixe contigu de 131 072 tokens du shard C4 00000 (*mesuré*, fiche-4b). Chaque niveau absolu est celui d'un seul
tirage. Une seconde graine au 8B et au 14B manque. Le fichier publié (autre shard) n'est pas un quatrième tirage. Tout
effet qui recalibre se lit contre ce σ. Un A/B à fichier constant a pour barre l'intervalle apparié, ±0,12 % en ppl et
0,43 pp en MMLU (*mesuré*, [kvq8-4b-2026-08-15.txt](mesures/kvq8-4b-2026-08-15.txt)).

À tête identique, le gain du noyau croît avec la taille : ×1,11, ×1,29, ×1,41 du 4B au 14B (*mesuré*,
b2-fusedrun-plages). La série brute (×2,00, ×2,57, ×2,55) n'a pas d'ordre ; elle date de ROT_SHARE=0/FUSE=0, non
rejouée sous v1.

## 5. Résultats du 2026-09-02

| lot | coût | ce qui est mesuré | résultat |
|---|---|---|---|
| M1 | 0 $, 12 runs Mac, 0,6B 28 blocs | shrink `H ← ρH + (1−ρ)diag H`, 3 graines | ρ=1 médiane 39,6042, étendue 4,6214 ; ρ=0,9 27,0812 / 3,1498 ; ρ=0,7 27,4944 / 0,6847 ; ρ=0,5 27,9506 / 2,9771 |
| M2 | ~2,17 $, 72,3 min | MMLU 4B, chaque type de projection rendu en f16, fichier constant, 11 bras | gate +5,18, up +4,94, v +4,48, down +2,96, o +2,35, k +2,09, q +1,85 pp ; attention +6,90, MLP +10,78, tout +14,73 |
| M2b | ~0,29 $, ~10 min | `v_proj` en int4 g128 déquantifié | MMLU 59,19, +3,60 pp [1,47 ; 5,79], McNemar 2,0e-4 ; 5,149 b/param |

Résultats *mesurés* : [m1-hessienne-shrink-2026-09-02.txt](mesures/m1-hessienne-shrink-2026-09-02.txt),
[m2-attribution-4b-2026-09-02.txt](mesures/m2-attribution-4b-2026-09-02.txt),
[m2b-v4bits-2026-09-02.txt](mesures/m2b-v4bits-2026-09-02.txt). Durées et coûts *calculés* sur les horodatages du
bucket (1,80 $/h). Préregs tamponnés avant chaque job, ancrage Bitcoin en attente. Vague 1 : 2,46 $ dépensés sur 5.

M2 désigne `v_proj` (2,6 % des poids, *calculé*, m2-attribution) ; le prior « k_proj et l'attention » est réfuté. Servir
`v_proj` en f16 coûterait +0,263 b/param (5,425, au-dessus de l'AWQ). En int4 g128 il en rend −0,013 (5,149) (*calculé*,
m2b-v4bits). Le Leech déplié pèse 4,804 b/poids là où int4 g128 en pèse 4,250 (*calculé*, m2b-v4bits). M2b garde 80,4 %
du gain f16 et ramène l'écart à l'AWQ de 14,45 à 10,85 pp (*calculé* sur M2 et M2b). M1 est vert : le kill prédit (ρ* =
1) est réfuté ; sur n = 3 le robuste est le signe et l'ordre de grandeur (médiane −12 ppl). Q1 adopte un ρ dans [0,5 ;
0,9], à ré-estimer au 4B (n/N 0,074 contre 0,023 au 0,6B, *calculé*, m1-hessienne-shrink). Boutons livrés :
`LLVQ_RESTORE_F16`, `LLVQ_RESTORE_Q4`, `LLVQ_H_SHRINK`.

## 6. Décisions ouvertes

- Ligne de lecture de M2b, opérateur. Aucune des trois lignes de la règle tamponnée ne s'applique : elle exige un IC >
  1,5 et la borne basse vaut 1,47
  ([preregistration-m2b-v4bits-2026-09-02-ECARTS.md](../proofs/preregistration-m2b-v4bits-2026-09-02-ECARTS.md)).
- Pousser `recherche/m1-m2-vague1` (15 commits non poussés ; `main` = `origin/main`), opérateur.
- Chantier suivant, opérateur : précision mixte sur `v_proj` (F3 sous F1, un codebook par matrice) ou F1. F1a doit
  compter la rétention exacte avant tout code : la projection rend 88,9 à 89,6 %, sous le kill de F1b à 90,3 %
  (*estimé* sur la branche recherche, [projection-gains-2026-09-01.md](archive/projection-gains-2026-09-01.md)).
- Triplet produit en vigueur (opérateur, 2026-08-16) : contexte 8k, marge 5 Go, unité 32 GiB, offload en référence
  seulement. Il laisse 27,93 Go aux poids, soit b_max = 3,00 b/poids noyau ; `Planes14` le dépasse de 60 %. La plus
  grande classe admise vaut 43,3 Md à 5,162 b/param (borne haute, embedding 9,7 %) et 45,8 Md à 4,878 (embedding
  ~2 %). Le 32B est l'objet servi, le 70B ne rentre pas (*calculé*,
  [note-produit-2026-08-13.md](archive/note-produit-2026-08-13.md) §B bis).
- Réplique de M2 sur la graine 3 de F5, préreg de Q1 au 4B, `ots upgrade` des trois tampons du jour après ancrage :
  opérateur.

## 7. Fermé sans idée neuve

- E1v sur le chemin servi : 0,25× f16 (*mesuré*, [e1v-cuda-2026-08-16.txt](mesures/e1v-cuda-2026-08-16.txt)).
- `Golay70` : v2 à 1,77× [1,76–1,78] sous le seuil tamponné de 2,0× (*mesuré*,
  [golay70-v2-sept-bras-2026-08-11.txt](mesures/golay70-v2-sept-bras-2026-08-11.txt)).
- E3 : 3,0444 b/poids noyau contre un critère de 2,60 (*calculé*,
  [radixstudy-x4-2026-08-12.txt](mesures/radixstudy-x4-2026-08-12.txt)).
- Volume de calibration : l'oracle rend −1,6 % de ppl, ×13 de tokens −1,2 % (*mesuré*,
  [verdicts-lot-b-2026-08-06.md](archive/verdicts-lot-b-2026-08-06.md)) ; l'échelle au 4B n'est pas partie, σ MMLU
  2,92 pp > 2,0.
- Embedding int4 g64 (`q4b-e4.llvq`, 1,211 Go, 2,4093 b/poids) : +1,52 % de ppl (*mesuré*,
  [verdicts-lot-b-2026-08-06.md](archive/verdicts-lot-b-2026-08-06.md) §B4) ; seul l'int8 (−0,02 %) est servi.
- Design C : ×1,99 de ppl à 28 blocs (*mesuré*, [verdicts-nuit-2026-08-07.md](archive/verdicts-nuit-2026-08-07.md)).
- `group_scales` : 44,66 → 53,60 de ppl à 28 blocs du 0,6B (*mesuré* au smoke du 2026-07-28, calibration 131 k tokens,
  sans journal).
- A2 servi (CUDA Graphs) : +12,6 % de débit contre +47 % de VRAM au 4B pour une fenêtre KV de 8k (*calculé*, jamais
  mesuré,
  [preregistration-a2-a3-geometrie-2026-08-31-ECARTS.md](../proofs/preregistration-a2-a3-geometrie-2026-08-31-ECARTS.md)
  §É7). Réouverture si le contexte servi descend à 2k (+12 % de mémoire pour +12,6 % de débit). Elle vaut aussi si le
  cache KV passe en q8 ou si la capture accepte un cache qui grandit. `KvStore::Cat` reste le défaut.
