# Pistes sur les cinq facteurs — verdicts d'exploration (2026-08-05)

> 🗓️ **BANDEAU D'ÉTAT — dernière revue le 2026-08-08. Les quatre expériences
> gratuites de §6 ont toutes été faites, et le suspect n°1 que cette note
> promeut a été réfuté.**
>
> - **§1 « ce qui est mort » tient toujours**, et s'allonge : la rotation de
>   sortie et le décodage spéculatif restent morts ; le **codage entropique**
>   est désormais clos une seconde fois, par la structure et non par zstd —
>   l'entropie de l'index vaut **46,6536 bits/bloc contre 47 payés**
>   ([`verdicts-lot-b-2026-08-06.md`](verdicts-lot-b-2026-08-06.md) §B5). Le
>   « lm_head int8 surdimensionné » a été mesuré et c'est l'inverse : le gain
>   bout-en-bout est **énorme** (48,7 → 88,5 tok/s), mais **pas pour la raison
>   annoncée** — le noyau q8 remplace **notre** chemin dense, qui appelle
>   `broadcast_matmul` et recopiait 778 Mo de vocabulaire par token, ce que ni
>   cette note ni le P5 du dépôt n'avaient vu
>   ([`mesures/phases-2026-08-07.txt`](mesures/phases-2026-08-07.txt)). ⚠️ Ce
>   n'est pas ce que fait candle : ses modèles passent par `Linear` et évitent
>   ce chemin ([candle#3871](https://github.com/huggingface/candle/issues/3871)).
> - **§2, suspect n°1 (design C) : ❌ RÉFUTÉ à pleine profondeur** — 0,6B,
>   28 blocs, une variable : **×1,99 de perplexité** contre le chemin publié.
>   Deuxième occurrence du motif « proxy local meilleur, composition
>   désastreuse » après `group_scales`
>   ([`verdicts-nuit-2026-08-07.md`](verdicts-nuit-2026-08-07.md) §M3).
>   Réserve maintenue : c'est *notre lecture* du design C qui est réfutée.
> - **§2, suspect n°2 (calibration) : ❌ plafonné.** L'oracle ne rend que
>   −1,6 %, la courbe de volume −1,2 % pour ×13. Le run ×100 est enterré
>   ([`verdicts-lot-b-2026-08-06.md`](verdicts-lot-b-2026-08-06.md) §B2/B3).
> - **§4, e4/e8 : ✅ mesurés.** int8 **gratuit** (−0,02 % de ppl, −365 Mo) et
>   **en production** depuis le 07 ; int4 **écarté sur le 4B** (+1,52 % de ppl).
>   Le « chaînon manquant » — un chemin d'exécution int8 — **existe** : deux
>   noyaux, un seul buffer, zéro spill (§M1 des verdicts de nuit).
> - **§6, l'ordre par dollar : les quatre premières lignes sont faites**, la
>   cinquième est tranchée (design C réfuté, calibration enterrée). Ce qui
>   reste des suspects du déficit MMLU : la config de gain, la composition du
>   corpus, la compensation post-hoc, le FT des échelles — et **l'axe
>   d'échelle**, qui est le seul à avoir bougé : à 8B le déficit tombe à
>   −10,56 pp et l'écart au 4 bits est divisé par deux
>   ([`echelle-4b-8b-2026-08-08.md`](echelle-4b-8b-2026-08-08.md)).
> - **§5, VRAM** : réglé ailleurs — `Planes14` en production à 4,804 b/poids,
>   `Planes12x` mesuré à 4,342 mais **non branché**, `Golay70` mesuré et
>   **écarté** à 1,31× ([`rapport-etat-2026-08-07.md`](rapport-etat-2026-08-07.md) §3).

> Deuxième exercice d'exploration (après
> [`pistes-format-vram-2026-08-05.md`](pistes-format-vram-2026-08-05.md) qui
> couvre le format VRAM). Méthode : 4 lecteurs + 3 vérificateurs adversariaux
> + 1 critique, sur l'arbre du 2026-08-05. **Rien n'a été modifié, rien n'a
> été lancé** — à une exception près, déclarée : un vérificateur a exécuté
> `zstd -19` en lecture seule sur le fichier scellé (hors dépôt, sortie en
> pipe, rien d'écrit) pour trancher la piste « codage entropique » par mesure.
>
> ⚠️ **Ce doc est une couche de vérification par-dessus
> [`pistes-battre-q4.md`](pistes-battre-q4.md)** (P1-P24, session
> antérieure), que cette exploration a redécouvert en route. Les numéros
> ci-dessous sont ceux du dépôt quand ils existent. Deux corrections à
> reporter dans ce doc-là quand le chantier le permettra : son P12 (rotation
> de sortie) tombe à ≈ 0, et son P22 porte un compte KV faux d'un facteur 2
> (déjà dénoncé par `fiche-4b.md:381` — le bon chiffre est 320 Kio/token,
> 2,68 Go à 8k).

## Le résultat en une phrase

Sur les cinq facteurs, l'exploration **ferme trois portes par mesure ou par
le papier lui-même**, **reclasse le suspect n°1 du déficit MMLU** (ce n'est
plus la calibration : c'est le chemin des magnitudes), et montre que **les
quatre prochaines expériences utiles coûtent 0 $** — elles se font sur le
Mac, sans code ou presque.

## 1. Ce qui est MORT — ne plus y revenir

| piste | verdict | preuve |
|---|---|---|
| **Rotation de sortie** (Input+Output) | effet ≈ 0 | Table 9 re-transcrite le 08-04 (`llvq-paper-notes.md:251-258`) : à Input fixé — notre config — l'étage Output vaut −1,7/+1,8/+1,2/−1,1 pp selon la famille, moyenne nulle. L'ancien « +5,6 pp » comparait *aucune* rotation à *toute* la rotation. C'était en plus le levier le plus cher à implémenter. |
| **Codage entropique du fichier froid** | clos par mesure | `zstd -19` sur la section matrices : **−0,89 %**. L'index de 6 octets/bloc est incompressible, comme `plan-de-test-papier.md:112` le prédisait. La marge structurelle n'était que 0,7 % (48 bits payés contre 47,66 d'entropie). |
| **Décodage spéculatif sur le 4B** | 0,95–1,3× aujourd'hui | Le draft 0.6B paie 78 % de l'overhead de lancement de la cible (28 blocs contre 36) : les lancements par token *augmentent* de ~50 %, ils sont déplacés, pas supprimés. Sous-multiplicatif avec le noyau fusé (1,37–1,65×). Le dépôt l'avait déjà écarté (`pistes-battre-q4.md:113` : « à ressortir à 70B » — là, le rapport draft/cible tombe à 1-2 % et la piste redevient bonne). |
| **lm_head int8 comme piste « nouvelle »** | duplicata, et surdimensionné | C'est le P5 du dépôt. Gain réel bout-en-bout : **+2,6 à 3,1 %**, pas +13 % (le lm_head tourne déjà à ~659 Go/s, quasi au pic — seule la réduction d'octets paie, et l'overhead de dispatch la dilue). Reste valable, à sa vraie taille. |

## 2. Perplexité + MMLU — le déficit de −4,8 pp, suspects reclassés

L'attribution à la calibration était « plausible, zéro mesure »
(`audit-publication-2026-08-03.md:372`) — et le suspect n°1 de cet audit
(rotation de sortie) vient de mourir ci-dessus. Classement corrigé :

**Suspect n°1 — le chemin des magnitudes (design C).** Le seul suspect
*chiffré* restant : la Table 9 promet **+1,9 à +3,3 pp de MMLU** pour
GPTQ→Spherical GPTQ, or notre config livrée a une rétraction no-op et le
raffinement désactivé (`retraction-et-gain.md:171-196`). Le débouché
documenté est le **design C** (rétraction libre + résolution close des
échelles en fin de couche) — jamais mesuré, et c'est un chantier, pas un
A/B. ⚠️ Le P14 du dépôt (prior de crête, fix de 2 lignes) n'en est qu'un
fragment, et un fragment **inscellable** (`gptq.rs:171-177` : aucun code de
bloc ne peut exprimer le résultat) — donc sans MMLU mesurable, donc pas
publiable seul. Et son protocole A/B 3 blocs est précisément celui qui a
donné le mauvais signe pour cette feature (gain sur 3 blocs, désastre sur
28).

**Suspect n°2 — la calibration (volume ET composition).** Toujours candidate,
toujours sans mesure. Avant tout run ×100 (~20-27 $ GPU, bf16 obligatoire,
DCLM à brancher ~30 lignes), **trois expériences gratuites bornent la
famille** :
- l'oracle (P3 du dépôt) : calibrer 3 blocs sur wikitext-2 test lui-même,
  2×8 min — donne le *plafond* de tout ce que la calibration peut rendre ;
- la courbe de volume 131k→500k→2M tokens sur 3 blocs : **zéro code** (le
  plafond `c4_calibration(8_000_000)` ≈ 2 M tokens suffit), ~30 min ;
- nouveau, par le mécanisme : le MMLU s'effondre sur le *raisonnement*
  (algèbre au hasard, histoire intacte) — faire varier la **composition**
  (dominante math/code) et pas seulement le volume. ~30 lignes par corpus.

**Suspect n°3 — la config 1 bit de gain.** Sans équivalent dans la Table 6
(écart 0↔2 bits : 1,4 pp). Le bon A/B est **leech2c11** (46+2 = 48
bits/bloc, iso-débit avec le fichier scellé), pas leech2c12 (49). ⚠️ Signal
mixte dans le papier : 2 bits de gain *améliore* wiki (15,54 vs 17,05) mais
*dégrade* MMLU (59,3 vs 60,7). Et sur gaussienne, 1 bit est l'optimum — notre
config est peut-être déjà la bonne.

**Le fine-tuning, à sa vraie place.** Les chiffres existent
(`llvq-paper-notes.md:96-100`) : le FT du papier — qui n'est qu'un
apprentissage des **échelles par colonne** (~760 k paramètres, ~52 M tokens,
acceptable côté souveraineté) — est le levier *perplexité* n°1 (17,05 →
**9,26**, sous la baseline FP16 !) mais ne rend que **+2,1 pp de MMLU**.
Pour le MMLU spécifiquement, les seuls gains publiés supérieurs sont EoRA
(+6,7-11,5 pp) et Recover-LoRA (+4-8 pp) — déjà aux P15/P16 du dépôt, avec
la réserve que le raisonnement est ce qui se récupère le moins. ⚠️ Tout FT
reclasse la comparaison : la ligne adverse devient QTIP-FT (9,61/59,5), pas
QTIP no-FT.

**⚠️ Aucun raccourci MMLU n'existe** : `bin/mmlu` exige un fichier scellé,
donc toute piste dont le verdict est MMLU coûte un run complet + scellement
+ 47 min de MMLU (~4 h locales ou ~8-10 $ GPU **par point**). Les A/B 3
blocs ne discriminent que la perplexité.

## 3. Vitesse — la hiérarchie est le chantier en cours, pas de détour

Vérifiée sur les mesures du dépôt : **branchement `bin/run` (+19-23 %) >
CUDA Graph (attaque les ~11 ms d'overhead) > fusion de la rotation dans les
producteurs > lm_head int8 (+3 %) > spéculatif (~0 sur le 4B)**. Cette note
n'y ajoute rien — c'est la priorité actée, et les pistes « contournantes »
examinées ici lui sont toutes inférieures.

Découverte au passage, pour le chantier lui-même :
`repeat_kv(...).contiguous()` **recopie le cache KV 4× par bloc et par
token** (`model.rs:422-423`) — un candidat de fix réel, à regarder pendant le
branchement. (Et la mémoire projet sur l'état du branchement est en retard :
la plomberie `new_with`/`ProjSource` est complète, non commitée, jamais
exécutée sur carte.)

## 4. Froid — le seul gisement restant est l'embedding, et il est mesurable gratuitement

Décomposition vérifiée à l'octet près : l'index+gain ne pèse que **51 % du
fichier scellé** et il est incompressible — tout levier froid hors embedding
est plafonné à cette assiette. L'embedding f16 = **43,9 %** du fichier
(777,9 Mo), et lui se compresse.

**Les artefacts existent déjà** : `~/q4b-e4.llvq` (int4, −559 Mo) et
`~/q4b-e8.llvq` (int8, −365 Mo), produits le 03-08, section matrices
bit-identique au scellé, **qualité jamais mesurée** (« interdit de
publication », `fiche-4b.md:385-390`). La chaîne de mesure est en place et
committée (LVQ3 lu par `bin/ppl` et `bin/mmlu`) : **ppl+MMLU des deux
fichiers = ~1 h 50 de Mac, 0 $, 0 ligne de code.** C'est l'expérience au
meilleur rapport information/coût de toute la note.

⚠️ **Le gain est à froid seulement, pour l'instant** : le chargeur
déquantifie l'embedding en f16 (`sealed.rs:105-113`), la VRAM ne bouge pas.
Convertir −365 Mo disque en −365 Mo VRAM exige un chemin d'exécution int8
(gather quantifié) qui n'existe nulle part — c'est le chaînon manquant
commun à cette piste et au lm_head (même tenseur lié : l'économie ne compte
qu'**une** fois).

## 5. VRAM — couvert ailleurs, deux compléments

Le gros de la VRAM est dans la note format
([`pistes-format-vram-2026-08-05.md`](pistes-format-vram-2026-08-05.md)).
S'y ajoutent : le chemin d'exécution int8 de l'embedding (§4, −0,39 Go sur
le 4B) et, pour la thèse 70B uniquement, la **quantification du cache KV**
(320 Kio/token f16, 2,68 Go à 8k ; int8 quasi gratuit en qualité d'après la
veille — TurboQuant descend à 3 bits). Personne n'a tranché, aucun code :
note produit, pas chantier immédiat.

## 6. L'ordre par dollar

| # | expérience | coût | tranche |
|---|---|---|---|
| 0 | **Gate S1 : 3 graines + damping** (6 runs de 3 blocs) | ~1 h Mac, 0 $ | ⚠️ **bloquant** : le projet n'a AUCUN chiffre de dispersion ; sans lui, tout écart de 2-6 % des A/B qualité est ininterprétable |
| 1 | Oracle calibration (P3 dépôt) | 2×8 min, 0 $ | le plafond de toute la famille calibration |
| 2 | Courbe volume 131k→2M sur 3 blocs | ~30 min, 0 $, zéro code | la pente du levier volume avant tout dollar GPU |
| 3 | **ppl+MMLU de e4/e8** | ~1 h 50 Mac, 0 $ | l'embedding int8/int4 est-il gratuit en qualité ? conditionne −365/−559 Mo de froid et le futur chemin VRAM |
| 4 | Mesurer l'adversaire (P1 dépôt, campagne CUDA prête) | — attend le branchement | le « ×1,386 contre ~1-2 % » enfin comparable |
| 5 | Ensuite, sur verdicts : design C (chantier), run calibration ×100 (~20-27 $), leech2c11 (~4 h + MMLU) | GPU, go explicite | les suspects n°1/2/3 du −4,8 pp |

## Correspondance des numérotations (obligatoire pour citer)

Cette note ↔ `pistes-battre-q4.md` : calibration ×100 = P10-dépôt · rotation
sortie = P12-dépôt (à réviser à ≈ 0) · 2 bits de gain = P13 · prior de crête
= P14 · FT = P17 · lm_head = P5 · KV = P22 (compte à corriger ×2) ·
allocation par sensibilité = P24 · oracle = P3 · mesurer l'adversaire = P1.
Le spéculatif et le codage entropique n'ont pas de numéro dépôt (le premier
y est écarté ligne 113, le second est clos ici par mesure).

## Risques et réserves à déclarer

- L'allocation de bits par couche (P24) : la granularité gratuite du format
  est **par matrice** (`shell_cap` par matrice), pas par couche ; la « carte
  de sensibilité » n'existe pas encore (`proxy_loss` est test-only,
  instrumentation bon marché à écrire) ; et le confondant `leech1c12L3` du
  8B (plafond L≤3 payé 48 bits, propagé par défaut par `ops/run.py:575`) est
  à purger d'abord.
- Le « 48 % hors matmuls » est une soustraction jamais confirmée par
  chronomètres — toute décision qui s'y adosse hérite de cette incertitude.
- Les A/B 3 blocs ont un précédent de signe inversé à pleine profondeur
  (group_scales) : un verdict 3 blocs ne suffit jamais à engager un run.
