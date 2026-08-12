# Battre le q4 — les pistes, notées (2026-08-03)

> 🗓️ **BANDEAU D'ÉTAT — dernière revue le 2026-08-08. Le « Rappel du score »
> ci-dessous est périmé sur trois axes sur quatre, et le tier 0 est intégralement
> mesuré.** Ce tableau remplace les statuts ; le corps est conservé pour la
> généalogie des décisions.
>
> | piste | statut | source |
> |---|---|---|
> | **P1 mesurer l'adversaire** | ✅ **fait le 06.** Et le pari de la correction n°2 est **perdu** : l'adversaire mesuré n'est pas du RTN MLX mais **l'AWQ officiel de Qwen**, qui rend **70,04 ± 1,25 de MMLU** — indiscernable du f16 (−0,28 pp), pas ~63. Notre écart réel est **−14,45 pp**, pas −7 | [`campagne-finale-2026-08-07.md`](../campagne-finale-2026-08-07.md), [`mesures/a4-campagne-2026-08-06.txt`](../mesures/a4-campagne-2026-08-06.txt) |
> | **P2 barre d'erreur + damping** | ✅ **fait.** σ(graines) ≈ **0,15 ppl ≈ 0,7 %** sur 3 blocs de 0,6B ; damping **nul** sur 3e-3..3e-2. ⚠️ Ce σ ne se transfère pas au chiffre publié du 4B — autre modèle, 3 blocs contre 36 | [`verdicts-lot-b-2026-08-06.md`](verdicts-lot-b-2026-08-06.md) §B1 |
> | **P3 oracle calibration** | ✅ **fait** : −1,6 % seulement, soit 29 % de l'écart. **La famille calibration est plafonnée** | idem §B2 |
> | **P4 profil GPU** | ✅ **fait sur CUDA** : ε = 3,63 µs/lancement × 252 = 0,915 ms, 15,8 % du bras LLVQ ; le CUDA Graph n'en récupère que 18 % | [`mesures/a3-graph-2026-08-06.txt`](../mesures/a3-graph-2026-08-06.txt) |
> | **P5 embedding/lm_head int8 puis int4** | ✅ **int8 EN PRODUCTION** (`LLVQ_EMBED=q8`), sans perte mesurable. ❌ **int4 mort sur le 4B** : +1,52 % de perplexité. Le 8B, têtes déliées, tourne aussi en q8 | [`verdicts-lot-b-2026-08-06.md`](verdicts-lot-b-2026-08-06.md) §B4, [`verdicts-nuit-2026-08-07.md`](verdicts-nuit-2026-08-07.md) §M1 |
> | **P6 plafond L≤4** | ❌ **MORT en qualité** : +4,75 % de ppl au swap mesuré, repasse au-dessus de QTIP. Remplacé par l'overlay épars, qui est exact | idem §B6 |
> | **P7 coupes ALU** | ⬜ non tentées telles quelles. Ce qui a payé à la place est le **changement de représentation** (one-hot → plans de bits), pas les coupes | — |
> | **P8 brancher le noyau** | ✅ **fait le 06**, sur CUDA, via `bin/fusedrun`. **×1,12 à tête identique**, ÷2,72 en mémoire | [`mesures/planes14-fusedrun-2026-08-06.txt`](../mesures/planes14-fusedrun-2026-08-06.txt) |
> | **P10 calibration ×100** | ❌ **enterré par P3** (~25 $ économisés) | §B2 |
> | **P12 rotation Input+Output** | ❌ morte : effet ≈ 0 à Input fixé (Table 9 relue) | [`pistes-facteurs-cles-2026-08-05.md`](pistes-facteurs-cles-2026-08-05.md) §1 |
> | **P14 design C** | ❌ **RÉFUTÉ à pleine profondeur** : ×1,99 de perplexité sur 28 blocs. Le suspect n°1 du déficit MMLU est mort tel qu'implémenté | [`verdicts-nuit-2026-08-07.md`](verdicts-nuit-2026-08-07.md) §M3 |
> | **P20 le point 8B** | ✅ **fait le 08.** ×1,220 de ppl et **−10,56 pp de MMLU** contre −14,73 au 4B : le déficit fond, et l'écart au 4 bits est **divisé par deux** (14,45 → 7,49 pp) | [`echelle-4b-8b-2026-08-08.md`](../echelle-4b-8b-2026-08-08.md) |
> | **P9 prefill hybride · P11 EoRA · P13 2 bits de gain · P15-P19 · P21-P24** | ⬜ non entamés | — |
> | *Écartées* : leech2c11, lm_head Slot32, mixte, transcodage à la volée, spéculatif, colonnes saillantes | ✅ toujours écartées ; le **codage entropique** l'est désormais aussi *par la structure* (46,6536 bits d'entropie contre 47 payés) | §B5 |
>
> **Le score, remesuré (4B, un seul harnais L40S)** : disque **gagné** (1,41–1,77
> contre 2,67 Go) · VRAM **gagnée depuis le 07** (5,15 contre 5,30 b/param) ·
> vitesse **non comparable** (deux moteurs) · qualité **perdue largement**
> (55,70 contre 70,04 de MMLU). L'« addition honnête » en fin de note visait
> « parité plausible » en qualité : elle est **infirmée** — l'adversaire ne perd
> rien à 4 bits sur un 4B.

> Produit par une exploration multi-agents (7 explorateurs par axe + 2 critiques
> adversariaux ; le 3ᵉ critique est tombé sur une limite de budget — la lentille
> « faisabilité vs code » a été refaite à la main sur les points porteurs).
> 47 pistes brutes, dédupliquées et corrigées ici en ~24. **Aucun run n'a été
> lancé** : tout ce qui suit est en attente de go.
>
> Notation : **A**pport (5 = change la donne sur un axe perdu face au q4),
> **C**omplexité (1 = heures, 3 = jours, 5 = semaines), **F**aisabilité
> (5 = quasi certain). Rappel du score : disque **gagné ×1,28** ; RAM perdue
> ×1,37 (3,28 vs 2,39 Go) ; débit perdu ×1,65 (78 projeté vs 129,8 mesuré) ;
> qualité perdue largement (ppl ×1,386, MMLU 56,09 vs 70,42).

## 0. Cinq corrections de cadrage — à lire avant les pistes

Elles sortent de la passe critique et invalident des raisonnements qui
circulaient dans le projet (y compris dans les pistes brutes elles-mêmes).

1. **Le noyau est borné ALU, pas mémoire — les octets ne rendent pas des
   millisecondes.** `format-noyau.md` le dit (240 Go/s tirés contre 336 au
   mur) : à décodeur inchangé, réduire les b/poids (L≤4, L≤3, Pack32) ne rend
   **aucune** ms — le nombre de blocs ne change pas, ~69 ps d'ALU par bloc non
   plus. Les formats compacts donnent la RAM ; la vitesse vient des coupes ALU
   (P7), qui sont le *multiplicateur* des formats. Ordre imposé par la
   physique : profil → coupes ALU → formats.
2. **La barre qualité de l'adversaire n'a jamais été mesurée.** Le « ~1-2 % »
   de `face-au-4-bits.md` est une supposition. MLX q4 g64 est du **RTN sans
   calibration** ; une étude externe (arXiv:2505.02214) donne ~**63 MMLU** pour
   du RTN 4 bits sur Qwen3-4B — pas ~69. Si ça se confirme localement, notre
   écart réel à l'adversaire est ~**−7 pp**, pas −14, et la parité qualité
   devient un objectif plausible au lieu d'un fantasme.
3. **L'artefact scellé est `leech1c12`** (vérifié : fin de
   `~/llvq-run-4b-artefact.log` — cap 12, 47 bits d'index + 1 gain = 48
   bits/bloc, 2,0702 b/poids effectifs). CLAUDE.md présente la restriction à
   Λ₂₄(12) comme une option future : elle est déjà dans le run publié. Trois
   chiffres coexistent pour un même objet (2,07 = comptabilité idéale, 2,17 =
   octets réels du fichier, « cap 13 » = une config qui n'est pas celle du
   fichier) — à réconcilier dans CLAUDE.md. Conséquence directe : la piste
   « leech2c11 pour gagner du disque » est **morte** (46+2 = 48 bits/bloc, Δ = 0).
4. **Le bruit n'est pas caractérisé.** Deux runs réputés identiques ont rendu
   14,27 et 15,29 de ppl (7 % de dispersion, `retraction-et-gain.md`) pendant
   que les pistes qualité promettent des effets de 2-6 %. Sans barre d'erreur
   (P2), chaque A/B futur est ininterprétable.
5. **Tous nos tok/s sont des projections poids-seuls ; les 129,8 de MLX sont
   du bout-en-bout** (attention, normes, KV, sampling — mesurés à ctx 256).
   Retrancher ~15-25 % de toute projection avant de comparer. Le seul chiffre
   opposable naîtra du branchement dans `bin/run` (P8).

## Tier 0 — Mesures préalables (2-3 jours, quasi zéro code)

Elles ne rapportent rien directement — elles empêchent de dépenser des
semaines au mauvais endroit.

| # | Piste | A | C | F | Ce que ça tranche |
|---|---|---|---|---|---|
| P1 | **Mesurer l'adversaire** : MMLU du q4 MLX local (mêmes 2 280 questions, micro), plus **q3** — l'adversaire iso-RAM jamais nommé (nos points de fonctionnement 3,35-4,67 b/poids encadrent le 3 bits, pas le 4) | 4 | 1 | 5 | La barre à battre (63 ? 66 ? 69 ?). Si LLVQ ≥ q3 en MMLU avec moins de RAM, un créneau « meilleure qualité qui tient dans X Go » existe dès le 4B |
| P2 | **Barre d'erreur + damping** : 6 runs de 3 blocs (`LLVQ_CALIB_SEED` {1,2,3}, `LLVQ_DAMPING` {3e-3,1e-2,3e-2}) — listés dans CLAUDE.md depuis des semaines | 2 | 1 | 5 | Le σ sans lequel aucun A/B qualité n'est lisible |
| P3 | **Calibration oracle** : calibrer 3 blocs sur wikitext-2 *test* lui-même (contamination délibérée), 2×8 min | 3 | 1 | 5 | Le **plafond** de toute la famille calibration (volume, corpus, longueur). Si l'oracle ne rend que 2-3 %, le suspect n°1 du projet est plafonné avant d'avoir payé le run ×100 |
| P4 | **Profil GPU** : Xcode GPU capture sur `bin/matvec` (jamais fait — le projet le note lui-même) | 2 | 1 | 5 | ALU vs latence vs occupancy — arbitre les coupes P7 avant de les écrire |

## Tier 1 — Les coups sûrs (RAM et débit, quelques jours chacun)

| # | Piste | A | C | F | Apport corrigé |
|---|---|---|---|---|---|
| P5 | **lm_head/embedding en int8 puis int4 g64 scalaire** (le format de l'adversaire, RTN + échelle/biais f16 par groupe). Noyau Metal trivial, memory-bound, zéro interaction avec le pipeline Leech. Le head f16 (0,78 Go) porte **88 % de l'écart RAM** avec le q4, et le risque qualité est borné par MLX lui-même (son fichier de 2,263 Go prouve que son embedding est en q4) | **5** | 2 | **5** | RAM 3,28 → 2,72 Go ; débit 2,32 → ~0,65 ms/token (78 → ~90 proj) ; disque 1,77 → 1,21-1,39 Go (avance ×1,63-1,87). Int8 d'abord (risque nul), int4 après A/B MMLU |
| P6 | **Plafond L≤4 du quantifieur** — déjà câblé (`with_level_cap`, `with_caps`, token `leech…L4` parsé) : zéro ligne de code, un A/B 3 blocs (8 min) puis un run 3,5 h. Stride Slot32 17 → 14 octets partout | 4 | **1** | 4 | RAM linéaires 2,50 → **2,12 Go** (4,667 b/poids — pas le « ~4,4 » optimiste). Avec P5 : **2,34 Go < 2,39** — la bascule RAM, à 2 % près (et calcul poids-seuls vs RSS mesurée : la vraie confirmation exige P8). **Aucun gain débit** (correction n°1). Décision sur MMLU, pas ppl |
| P7 | **Coupes ALU du décodeur Slot32** : masques précombinés par bloc (mA=m1\|m3, mB=m2\|m3 → 3 tests au lieu de 4), signe par XOR sur le bit 31, table des 384 classes en threadgroup (12,3 Ko), 2 blocs en vol par lane, sweep d'occupancy (256 jamais remis en cause) | 4 | 2 | 3 | Projections 10,46 → ~8,0-8,6 ms/token (270-310 Go/s). C'est **le multiplicateur** : sans lui, aucun format compact ne rend de débit. Avec P5 : ~8,2-9,3 ms → ~110-120 tok/s projetés |
| P8 | **Brancher le noyau dans `bin/run`** via un CustomOp candle (candle garde embedding/normes/attention/sampling ; les 252 Linear passent au matvec fusé + rotation d'activation). Le runner actuel décode en f16 (pic **7,3 Go — pire que MLX au chargement**) et **rejoue le préfixe à chaque token** (pas de cache KV) | **5** | 3 | 4 | Le seul chemin vers un chiffre bout-en-bout opposable aux 129,8. Pic RAM 7,3 → 3,3 Go — c'est lui qui rend vrai « tient sur la machine ». Prérequis explicite de toutes les pistes débit |
| P9 | **Prefill hybride** : déquant-par-tuile → GEMM f16 (scratch par couche, jamais le modèle entier). ~70 ms d'overhead fixe puis vitesse f16 | 3 | 2 | **5** | Sans lui, tout prompt > quelques centaines de tokens est perdu ×10-20 (25,6 s → ~2-3 s sur 2 000 tokens). Même mécanisme pour le régime batché |

## Tier 2 — La qualité (l'axe le plus perdu, et le seul sans gain mesuré chez nous)

Aucune piste qualité n'a aujourd'hui un gain mesuré **dans ce pipeline** — tout
est transposé du papier ou de la littérature. D'où l'ordre : P2/P3 d'abord,
puis les A/B 3 blocs, puis seulement les chantiers.

| # | Piste | A | C | F | Apport attendu (source) |
|---|---|---|---|---|---|
| P10 | **Calibration ×100 (~13 M tokens, DCLM-edu) sur GPU loué** (~20-30 $). Les passes avant sont embarrassingly parallel ; l'encodeur Leech (59 % du run, CPU) est *indépendant du volume* — profil GPU mesuré : la part qui scale est 0,2 % du run | 4 | 3 | 3 | Le suspect n°1 du −4,8 pp au papier (notre baseline le reproduit à 0,22 pp, le harnais est hors de cause). Cible : MMLU ~60-61. **Conditionné au verdict de P3 (oracle)** |
| P11 | **EoRA : compensation bas-rang fermée** dans l'espace propre des hessiennes **déjà calculées** — pas un fine-tuning, une SVD projetée. Rang 32 f16 sur les 252 matrices = 132 Mo (+0,29 b/poids) | 4 | 2 | 4 | +3-6 pp MMLU (gains publiés : 6,7-11,5 pp sur raisonnement à 3 bits ; à 2,07 bits le rang devra peut-être monter). À coupler à L≤4 pour rester sous les 4,50 b/poids du q4 |
| P12 | **Rotation Input+Output** (on n'a que la moitié du levier le plus fort mesuré : 21 % de ppl pour Input seule). H ne dépend que de x → machinerie GPTQ intacte, dé-rotation hors ligne | 3 | 2 | 4 | 1-3 % ppl, 1-3 pp MMLU (la Table 9 ne décompose pas). A/B 8 min avant le run |
| P13 | **2 bits de gain** (`gain_bits: 2`, Lloyd-Max déjà en place) : la meilleure ligne no-FT du papier (15,54) | 3 | 1 | 4 | ppl 16,94 → ~15,5-16 pour +0,042 b/poids disque. ⚠️ en MMLU le papier donne l'ordre **inverse** (59,3 < 60,7) : évaluer les deux métriques |
| P14 | **group_scales : prior vers 1** (`rhs[p] += ridge` — le fix identifié dans CLAUDE.md) puis le **design C** (rétraction libre + solve final + re-projection sur la grille de gain) — la config officiellement recommandée du papier | 3 | 2 | 3 | Récupère le gain local mesuré (−0,35 % sur 3 blocs) sans la dérive globale ; en 0 bit de gain, rend aussi 1 bit/bloc |
| P15 | **Audit Qualcomm, items B/C/E jamais repris** : vraie Hadamard de Paley H₂₀/H₇₆ (incohérence μ mesurée 3,4 → 1 sur down_proj), récupération d'échelle **par ligne** (Algorithme 3 à m=1 : 0 bit, 0 format), centroïdes de gain pondérés par row_scale², diagnostic gratuit de l'étalement de diag(U) | 3 | 2 | 4 | Qualité à format et débit strictement constants — le petit frère sans risque de group_scales |
| P16 | **Recover-LoRA / distillation légère** sur données synthétiques (recette publiée sur Qwen3-4B 2 bits exactement) : adaptateur bas-rang, poids quantifiés gelés → noyau et artefact intacts | **5** | 4 | 3 | +4-8 pp MMLU visés (80-95 % de récupération sur 9/12 benchmarks). ⚠️ le raisonnement — notre mode d'effondrement — est ce qui se récupère le **moins** (48 %) |
| P17 | **Fine-tuning des échelles par colonne** — le levier n°1 du papier, jamais implémenté (chez eux : ppl ×1,374 → ×0,746, MMLU −9,5 → −7,4 pp). ~760 k paramètres candle Var, passe avant maison déjà exacte au bit près | **5** | 4 | 3 | Même la moitié du gain ramènerait la ppl vers ~13-14 et 3-5 pp de MMLU. Risque : couverture des gradients Metal dans candle |
| P18 | **Corpus de calibration ciblé raisonnement** (math/code/symbolique) : une direction d'activation jamais excitée par X n'est pas protégée par GPTQ — le profil MMLU (algèbre à 25 %) est exactement cette signature | 4 | 2 | 2 | La seule piste qui attaque le **mécanisme** identifié. Risque somme-quasi-nulle : surveiller que les matières > 80 % ne bougent pas |

## Tier 3 — Structurel et produit

| # | Piste | A | C | F | Ce que ça change |
|---|---|---|---|---|---|
| P19 | **Pack32** : layout positionnel 2 bits/slot `[classe 9][gain 1][smask 24][codes 2×24]` = 82 bits. Le « positionnel » n'avait été enterré que face aux masques *imbriqués*, avant Slot32 dont les masques coûtent 24(L−1) bits. Offsets fixes, zéro divergence — toutes les propriétés de la brèche Slot32 | 4 | 3 | 4 | Corrigé par la critique : ~**4,33** b/poids sans cap (qualité bit-identique) ou **3,667** avec L≤4 — les deux promesses **ne se cumulent pas**. À 70B avec cap : ~32-33 Go vs 39,4. Vitesse non mesurée : le pari est le coût du select indexé vs 4 tests de masque — seul le banc froid tranche |
| P20 | **Le point 8B** (~20 $, infra ops déjà chiffrée) : le papier donne Qwen3-8B no-FT à **×1,13** là où le 4B fait ×1,37 — le déficit du 2 bits **fond avec l'échelle**, et LLVQ y bat déjà QTIP | 4 | 2 | 4 | LE point de la courbe qualité-vs-taille qui décide si le pari 32B/70B est fondé, pas juste une validation d'infra |
| P21 | **Benchmark métier d'extraction documentaire** — exigé par le plan (Phase 5, étape 5) et la veille, jamais construit. Le profil de casse dit : restitution conservée, raisonnement détruit | 4 | 2 | 4 | Si l'extraction — le métier réel de l'argument souveraineté — tient à 2 bits, le produit 4B est viable **aujourd'hui** sur son créneau malgré les −14,33 pp |
| P22 | **KV cache quantifié** — levier n°2 de la propre veille du dépôt, absent de tous les plans. 147 Ko/token f16 sur le 4B → ~0,60 Go à ctx 4096, l'ordre de **tout** l'écart RAM avec q4 | 3 | 3 | 3 | Troue la thèse 70B à contexte long : ~640 Ko/token → 8k = ~5,2 Go, qui fait repasser L≤3 (32,1 Go) au-dessus du working set d'un Mac 48 Go. Et les 129,8 de MLX sont mesurés à ctx 256 — la comparaison à contexte long n'existe pas |
| P23 | **Le créneau 70B** : L≤3 (32,1 Go) sur Mac 48 Go, là où q4 (39,4) **ne charge pas**. Encodage ~45 h sur M3 Max ou ~17 h loué | **5** | **5** | 2 | Le seul terrain où q4 est battu structurellement. Corrigé : sans les coupes ALU, ~5-7 tok/s (pas 7,5-10,5) — P7 est un prérequis. Conditionné à la qualité (P10-P17), au KV (P22) et au point 8B (P20). Barre exigeante et juste : battre le MMLU d'un q4-32B (18,4 Go) |
| P24 | **Allocation par matrice** : sonde de sensibilité (proxy erreur GPTQ, déjà loggée par couche) → caps/gain différenciés par matrice, sans changer le kernel (le format porte déjà des configs hétérogènes par matrice) | 3 | 2 | 3 | Dépenser L≤4 où ça raisonne, L≤3 où ça restitue — la version raffinée de P6, après P6 |

## Écartées — et pourquoi (ne pas re-creuser sans idée neuve)

- **`leech2c11` pour le disque** : économie fictive — l'artefact publié est déjà
  à 48 bits/bloc (`leech1c12`). L'A/B reste sensé comme pur échange
  index↔gain à bits constants, rien de plus.
- **lm_head en LLVQ Slot32** : dominé point par point par l'int4 g64 (268 Mo
  vs 219, 1,1 ms vs 0,65, plus de risque, plus de code).
- **Mixte Slot32/Grouped32 par matrice** : verdict négatif — en décodage,
  toutes les matrices sont chaudes à chaque token.
- **Transcodage à la volée / streaming de couches** : mort — 3,1 s de
  transcodage pour un modèle qu'un transformer dense touche entièrement à
  chaque token.
- **Décodage spéculatif draft 0.6B** : hors sujet sur un 4B ; à ressortir à 70B.
- **« L≤3 est plus petit ET plus rapide »** : la moitié débit est fausse
  (correction n°1 — noyau borné ALU). L≤3 reste l'option RAM du créneau 70B/48 Go.
- **Colonnes saillantes pleine précision (SpQR/OWQ)** : greffable via la queue,
  mais travaille *contre* la rotation d'incohérence — le levier n°1 mesuré.

## L'addition honnête, si tout marche

| axe | aujourd'hui | après P5+P6+P7+P8 | verdict |
|---|---|---|---|
| disque | 1,771 vs 2,263 Go | 1,21-1,39 Go | gagné ×1,6-1,9 |
| RAM | 3,28 vs 2,39 Go | ~2,34 Go (poids seuls) | **bascule, de justesse** — confirmation par RSS réelle exigée (P8) |
| débit | ~78 proj vs 129,8 | ~110-120 proj → **~95-105 bout-en-bout** | **pas gagné sur le 4B** — parité approchée au mieux |
| qualité | 56,09 vs ~63 ? (barre non mesurée) | +1-3 (calib) +3-6 (EoRA) → ~60-64 | parité **plausible** — contre une barre à mesurer d'abord (P1) |

Trois axes sur quatre peuvent basculer ou approcher la parité, mais **aucune
victoire nette sur le 4B au 4B** : le point de bascule structurel reste
l'échelle (P20 : ×1,13 à 8B chez eux ; P23 : 70B où q4 ne rentre pas) — et le
créneau immédiat, la tâche métier (P21) plutôt que MMLU.

## Ordre de bataille proposé (avec gates)

1. **S1 — mesurer avant de construire** (2-3 j, ~0 code) : P1 + P2 + P3 + P4.
   *Gate : la barre qualité réelle, le σ, le plafond calibration, le profil GPU.*
2. **S2 — la bascule RAM** (1 sem) : P5 (int8 → int4) + P6 (A/B puis run).
   *Gate : MMLU stable et RAM < 2,39 calculée.*
3. **S3 — le chiffre opposable** (1-2 sem) : P7 + P8 + P9.
   *Gate : tok/s bout-en-bout et RSS mesurés sur la même machine que les 129,8.*
4. **S4 — la qualité** (2-3 sem, conditionnée à S1) : A/B 3 blocs P12/P13/P14/P15,
   puis P10 si l'oracle le justifie, puis P11 ; P16/P17 seulement si l'écart
   résiduel l'exige. *Gate : MMLU vs la barre mesurée en S1.*
5. **S5 — le terrain** : P20 (8B) → décide P22 + P23 (70B). P21 en parallèle
   dès S2 (indépendant de tout).
