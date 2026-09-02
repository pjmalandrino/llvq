# Projection des gains — pistes de l'audit sur les critères fondamentaux

> Compagnon de [`audit-recherche-2026-09-01.md`](audit-recherche-2026-09-01.md)
> et de [`ROADMAP-RECHERCHE.md`](ROADMAP-RECHERCHE.md). Rédigé le 2026-09-01.
>
> 🚨 **Rien ici n'est mesuré.** Ce document projette ; le dépôt a payé trois
> fois la leçon « un compte d'opérations n'est pas une prédiction de temps »
> (×1,002 prédit, ×2,17 mesuré — `CLAUDE.md`). Chaque nombre porte donc son
> étiquette : **mesuré** (une ligne de `docs/mesures/` existe), **calculé**
> (arithmétique sur des nombres mesurés, reproductible à la main), **estimé**
> (le mien, avec l'ancre qui le fonde et une plage volontairement large).
> Les plages ne sont pas des intervalles de confiance : ce sont les bornes
> entre lesquelles je serais *surpris* que la mesure tombe, pas plus.
>
> Les critères fondamentaux sont ceux des quatre axes du dépôt (disque, VRAM,
> débit, qualité), plus trois grandeurs qui gouvernent la faisabilité :
> **classe de modèle chargeable**, **coût d'encodage**, **plancher de bruit**.

---

## 0. Le point de départ (Qwen3-4B, L40S — tout *mesuré*)

| critère | valeur | source |
|---|---|---|
| disque, b/poids effectifs | 2,0702 (1,771 Go ; 1,406 Go avec embedding int8) | `fiche-4b.md` |
| VRAM, b/param modèle entier | **5,162** → 2,57 Go carte | `rtbits-planes-8b`, `vague2-fusion` |
| débit servi (v1, batch 1) | **100,6** tok/s ; **112,5** avec A2 graphes (+13,45 %) | `vague2`, `a2-verdict` |
| banc, 144 lancements fusés (`Planes14`) | **4,504 ms** ; QTIP 2,246 ; plancher `nullk` 1,794 | `design-a3`, F2, A1 |
| perplexité | ×**1,3845** (16,94 / 12,24) | `a4-campagne` |
| MMLU micro | **55,59 %**, −14,73 pp f16, **écart AWQ 14,45 pp** | `a4-campagne`, `mmlupair` |
| σ de calibration (3 graines) | **5,2 %** ppl ; **2,92 pp** MMLU | F5, `bruit-mmlu-graines` |
| transcodage au chargement | 131 s (`Planes14`) | `format-noyau.md` |
| encodeur | 656 µs/bloc/cœur ; 4B ≈ 2,4 h sur M3 Max ; 7,11 $ sur GPU | §6 `CLAUDE.md`, F5 |
| A100 | `Planes14` 0,79× FP16 | F4 |

Références concurrentes *mesurées* dans le dépôt : AWQ 4 bits 5,302 b/param,
70,04 % MMLU ; IQ2_XXS 2,479 b/param, 39,39 % ; QTIP 2,000 b/poids noyau,
2,246 ms (qualité non mesurée chez nous ; 57,4 % dans le papier LLVQ).

---

## 1. Axe F — le format sans dépliage (F1 « Leech-3×E₈ »)

### 1.1 VRAM et classe de modèle (*calculé* sur les parts d'embedding mesurées)

Hypothèse de format : 48 bits/bloc + queue f32 (0,149 b/poids au 4B) +
échelles f32 (0,010) = **2,159 b/poids noyau** ; embedding q8 à 8,5 b/param.

| modèle | part embedding | v1 servi (mesuré) | **F1** | F1 + embedding **q4** (e4, +1,52 % ppl mesuré au lot B) |
|---|---|---|---|---|
| 4B | 9,7 % | 5,162 → 2,57 Go | **2,77 → 1,39 Go** | 2,39 → 1,20 Go |
| 8B (têtes déliées) | 15,2 % | 5,322 → 5,41 Go | **3,12 → 3,20 Go** | 2,52 |
| 14B | 10,5 % | 5,106 → 9,40 Go | **2,83 → 5,21 Go** | 2,40 |
| 70B dense (emb. ~1,5 %) | 1,5 % | ≈ 4,86 → **≈ 42,5 Go** | **≈ 2,25 → ≈ 19,7 Go** | ≈ 2,19 |

Comparaison : IQ2_XXS 2,479 au 4B. **F1 passe sous IQ2_XXS au 4B seulement
avec l'embedding q4** ; aux grandes tailles la part d'embedding fond et F1 y
passe dessous sans lui.

**Classe de modèle chargeable** (*calculé*, 85 % de la carte pour les poids,
15 % réservés au KV et aux activations) :

| carte | v1 (4,86 b/param) | AWQ (5,30) | **F1 (2,25)** |
|---|---|---|---|
| 24 Go | ~34B | ~31B | **~72B** |
| 48 Go | ~67B | ~62B | **~145B** |
| 80 Go | ~112B | ~103B | **~240B** |

C'est la seule ligne de ce document qui change la **thèse** du projet, et non
un chiffre : sous v1 un 70B exige 48 Go, sous F1 il tient sur 24.

### 1.2 Débit au banc (*estimé*, modèle additif « plancher + octets/débit »)

Le modèle : `t = t_plancher(géométrie) + octets / BW_eff`. Calibré sur le
mesuré : `4,504 = 1,794 + 2,182/BW` ⇒ BW_eff ≈ 805 Go/s (93 % du pic L40S —
cohérent avec un flux pur). F1 lit 0,981 Go (*calculé*).

| géométrie | `Planes14` (mesuré) | **F1** | gain au banc |
|---|---|---|---|
| la nôtre, 144 lancements | 4,504 ms | **≈ 3,0 ms** | **×1,5** |
| la nôtre, 252 lancements | 5,103 | ≈ 3,6 | ×1,4 |
| géométrie type QTIP / A3 (grille fixe, split-K) | — | **≈ 2,4 ms** (QTIP fait 2,246 pour 0,91 Go) | **×1,9** |

⚠️ Le modèle additif suppose que le plancher ne bouge pas quand les octets
tombent : c'est faux à la marge (moins de blocs par lane = moins d'itérations
vides), donc la borne basse est plutôt conservatrice. Il suppose aussi que le
décodage à trois lookups reste sous le trafic — c'est **l'hypothèse à tuer en
premier** : E1v l'a violée (0,25×) avec des marches binomiales, pas avec des
lookups.

### 1.3 Débit bout-en-bout, batch 1 (*estimé*, Amdahl)

Par token à 112,5 tok/s : 8,89 ms, dont ≈ 4,5 ms de matvec de projections
(*estimé* depuis le banc ; le reste : rotation, attention, `lm_head` q8,
normes, orchestration candle). F1 économise 1,5 ms (notre géométrie) à
2,1 ms (géométrie A3) :

| | tok/s 4B | ×  |
|---|---|---|
| aujourd'hui (A2) | 112,5 (mesuré) | 1,00 |
| F1, notre géométrie | **≈ 135** | ×1,20 |
| F1, géométrie A3 | **≈ 147** | ×1,31 |

**Ce que la loi d'Amdahl impose ensuite** : après F1, les projections ne font
plus que ~30 % du token ; le front bascule vers l'attention, la rotation et
l'orchestration — exactement ce que la comparaison vLLM f16 (83,1 tok/s contre
nos 43,6 en dense) pointait déjà. Le gain suivant n'est plus un format.

### 1.4 Qualité (*estimé*, en défaveur de F1)

La région de mise en forme change (produit de trois boules de dimension 8 au
lieu d'une boule de dimension 24) : perte de forme ≈ 0,3-0,4 dB ⇒ **+7-9 % de
MSE directionnelle** (*estimé* sur les gains de forme classiques : 0,65 dB en
dimension 8, ~1,0 en dimension 24). Transposé au premier ordre sur l'excès de
perplexité (0,3845 × 1,08) : **ppl ×1,3845 → ×1,41-1,42 ; MMLU −1 à −2 pp**.
Le trellis shaping entre sections peut en rendre une partie ; **F1b le
mesure à 0 $** avant toute décision. Kill posé : rétention gaussienne < 90,3 %.

### 1.5 Coûts d'ingénierie et de calcul

| grandeur | aujourd'hui | F1 | étiquette |
|---|---|---|---|
| transcodage au chargement | 131 s (`Planes14`), 1 340 s (`Planes12x`) | **0** — disque = VRAM | calculé |
| encodeur | 656 µs/bloc | **≤ 100 µs/bloc** (256 états × 3 décodages E₈) | estimé |
| quantification du 4B (Mac) | 2,4 h | **≈ 0,5 h** | estimé |
| quantification du 32B (GPU, encodeur 71,8 % du temps) | ≈ 62 $ | **≈ 25-30 $** | estimé |
| A100 | 0,79× | **non projetable** — QTIP n'a pas été mesuré sur A100 ; moins d'ALU par poids plaide pour F1, sans nombre | — |

---

## 2. Axe Q — perdre moins (Qwen3-4B, toutes valeurs *estimées*)

### 2.1 Une piste à la fois, avec son ancre

| piste | ancre (littérature ou dépôt) | transposition à notre excès de ppl (38,45 %) | Δ MMLU | Δ b/poids | ce qui la rend visible |
|---|---|---|---|---|---|
| **Q1** shrinkage de H | 2604.13806 : hors-diagonaux instables ; notre 13,5 échantillons/dim | **variance** : σ 5,2 % → **2-3 %** ; moyenne 0 à −3 % | σ MMLU 2,9 → ~1,5 pp ; moyenne +0 à +1 | 0 | rend tout le reste mesurable |
| **Q2** cible asymétrique + Hessienne de sortie | GPTAQ/KronQ : −13,5 % ppl à W2 sur Llama-2-7B (8,83 → 7,64), « l'essentiel vient de la dérive » ; D²Quant : DSQ +3,48 MMLU | **−10 à −20 % de l'excès** → ×1,31-1,35 | **+1,5 à +3 pp** | 0 | Q1 (sous σ actuel : invisible) |
| **Q3** GPTQ en faisceau | AQLM : faisceau vs glouton, ~2-4 % de ppl à 2 bits | −5 à −10 % de l'excès → ×1,35-1,37 | +0,5 à +1,5 pp | 0 | Q1 |
| **Q4a** équi-norme | Nagel 2019 (scalaire) ; ici n'agit que sur le terme de gain | 0 à −5 % de l'excès | 0 à +1 pp | 0 | Q1 |
| **Q4b** diagonale (FT d'échelles du papier) + biais radial | papier LLVQ : **+2,1 pp** MMLU, < 0,001 b/poids ; biais radial **+3,7 %** de sur-coût *mesuré* | −8 à −12 % de l'excès → ×1,34-1,35 | **+1,5 à +2,5 pp** | ≈ 0 | Q1 ; la version pleine attend Q6c |
| **Q5** précision mixte par fonction | APTQ (K à 4 bits), KVTuner, HyQuant ; **conditionnelle à M2** | si `k` porte ≥ 3 pp : la totalité de sa part | **+3 à +6 pp si M2 désigne `k`** ; 0 sinon | **+0,05** (`k`) à +0,26 (`q+k`) | M2 |
| **Q6a** distillation des paramètres libres du format | Norm Tweaking ; EoRA +6-11 pts à 3 bits ; ici ~18 M paramètres libres | **−15 à −25 % de l'excès** → ×1,29-1,33 | **+2 à +5 pp** | **0** | M3 (métrique STEM) |
| **Q6b** EoRA r ≤ 16 | gate déjà posé dans `BACKLOG` §3.3 | −10 à −15 % de l'excès | +2 à +4 pp | +0,13 à +0,25 b/param | Q6a rendu |
| **Q6c/d** relaxation différentiable + KL bout-en-bout | PV-tuning : **l'excès divisé par ~2** à 2 bits sur Llama-2-7B ; UPQ / Reasoning-QAT ; BCJR-QAT | **excès ÷ 1,7-2,2** → ×1,17-1,23 (le niveau **8B-14B** d'aujourd'hui) | **+5 à +9 pp** | 0 (poids) | go explicite, dizaines de $ |
| **Q7** corpus | oracle ppl −1,6 % (mesuré) ; non borné sur STEM | −0 à −4 % de l'excès | 0 à +2 pp (STEM) | 0 | M1, M3 |

⚠️ **Le fait le plus important de cette table n'est pas dans une cellule** :
prises une à une, Q2, Q3, Q4a et Q4b sont chacune **sous le plancher de bruit
actuel** (σ = 5,2 % ppl, 2,9 pp MMLU). Sans Q1, on ne pourrait ni les adopter
ni les enterrer — c'est pourquoi l'axe M précède tout.

### 2.2 Combinaisons (*estimées*, rendements décroissants appliqués : ×0,7 sur la somme)

Les effets ne s'additionnent pas : Q2 et Q3 optimisent le même proxy ; Q6a
récupère une partie de ce que Q2-Q4 auraient corrigé. Trois scénarios :

| scénario | contenu | ppl 4B | MMLU 4B | écart AWQ | repère |
|---|---|---|---|---|---|
| aujourd'hui | v1 | ×1,3845 | 55,6 % | 14,5 pp | — |
| **pessimiste** | Q1 + une seule des Q2/Q3/Q4 passe | ×1,34 | **57,5 %** | 12,5 pp | QTIP papier 57,4 |
| **central** | Q1 + Q2 + Q3 + Q4b + Q6a | ×**1,25** | **≈ 61 %** | **≈ 9 pp** | l'écart du **8B** aujourd'hui (7,5 pp) |
| **optimiste** | central + Q5 (si `k`) + Q6c/d | ×**1,18** | **≈ 66 %** | **≈ 4 pp** | mieux que le **14B** aujourd'hui (6,1 pp) |

Lecture : le scénario central ramène le 4B à 2 bits **au niveau de qualité
relative du 8B servi aujourd'hui**, à 0 bit de plus ; l'optimiste exige la
distillation (Q6d) et le franchissement du seuil est **la ligne « with FT »
du papier** (60,7 → 62,8 en MMLU pour eux, avec leur seul FT d'échelles).

---

## 3. Lecture par critère — les trois trajectoires côte à côte

| critère (4B, L40S) | aujourd'hui *(mesuré)* | **F1 seul** | **Q seul (central)** | **F1 + Q (central)** | concurrents *(mesurés)* |
|---|---|---|---|---|---|
| disque, b/poids | 2,07 | 2,07 | 2,07 | 2,07 | IQ2_XXS 2,06 |
| **VRAM, b/param** | **5,16** | **2,77** (2,39 avec e4) | 5,16-5,26 (Q5) | **2,8-2,9** | AWQ 5,30 · IQ2_XXS 2,48 |
| **70B en Go** | **42,5** | **19,7** | 42,5 | **≈ 20** | AWQ ≈ 46 · IQ2_XXS ≈ 21 |
| banc 144 lancements | 4,50 ms | 3,0 (2,4 sous A3) | 4,50 | 3,0 (2,4) | QTIP 2,25 |
| tok/s bout-en-bout | 112,5 | **135-147** | 112 | **135-147** | vLLM f16 83 *(autre pile)* |
| ppl (× f16) | ×1,385 | ×1,41-1,42 | ×**1,25** | ×**1,27-1,29** | AWQ ×1,105 · QTIP ×1,373 *(papier)* |
| MMLU micro | 55,6 % | 54-55,5 | **≈ 61** | **≈ 59-60** | AWQ 70,0 · QTIP 57,4 *(papier)* · IQ2_XXS 39,4 |
| écart AWQ | 14,5 pp | ~15,5 | ~9 | ~10,5 | — |
| transcodage | 131 s | **0** | 131 | **0** | — |
| encodeur 4B (Mac) | 2,4 h | **~0,5 h** | ×K (faisceau : ~10 h à K = 4) | ~2 h (F1 × K) | — |
| σ calibration | 5,2 % | 5,2 | **≤ 2,6** | **≤ 2,6** | — |

**Ce que cette table dit, en trois phrases.** (1) F1 est le seul chantier qui
change la **classe de modèle** — et il coûte probablement 1 à 2 pp de MMLU
qu'il faudra racheter par Q. (2) Q est le seul chantier qui change la
**qualité** — et il ne bouge ni la VRAM ni le débit. (3) Leur combinaison
donnerait un point **≈ 2,8 b/param, ≈ 60 % MMLU, ≈ 140 tok/s** : sous l'AWQ
de 1,9× en mémoire pour −10 pp, devant IQ2_XXS de +20 pp à mémoire
comparable, devant le QTIP du papier en qualité à 8 % de bits noyau en plus.
C'est un point qui **n'existe chez aucun concurrent mesuré** ; c'est aussi un
point *estimé* à quatre étages d'hypothèses, dont chaque étage a son gate.

---

## 4. Où la projection est la plus fragile (par ordre)

1. **Le décodage F1 reste-t-il sous le trafic ?** Tout le §1.2 en dépend ; E1v
   a montré qu'un décodeur plus « malin » peut coûter 17× — mais c'était de
   l'arithmétique par bloc, pas trois lookups. Tranché par **F1d** (~1 $).
2. **La perte de forme de F1** (§1.4) : entre 0 et 9 % de MSE selon ce que la
   contrainte de coset récupère. Tranché par **F1b** (0 $).
3. **La transposition ppl → MMLU** : le dépôt a mesuré que les deux métriques
   ne bougent pas ensemble (genou résolu en ppl, non résolu en MMLU). Les
   Δ MMLU du §2 sont les plus incertains du document.
4. **Les rendements décroissants** (×0,7) sont un choix, pas une mesure.
5. **L'Amdahl du §1.3** repose sur une part de 50 % des projections dans le
   token qui n'a jamais été profilée (« le profileur n'a jamais servi »).

---

## 5. Ce que la mesure devra remplacer, et quand

| projection | remplacée par | jalon |
|---|---|---|
| σ ≤ 2,6 % | M1 | D1 |
| Δ MMLU de Q5 | M2 | D1 |
| rétention F1 ≥ 91 % | F1b | D2 |
| Δppl de Q2/Q3/Q4 | gates 0,6B à 3 graines | D2 |
| t(F1) ≈ 3,0 / 2,4 ms | F1d | D3 |
| MMLU ≈ 61 (Q central) | un run 4B + `mmlupair` | D3 |
| 2,77 b/param, 135-147 tok/s | F1e | D4 |

Chaque ligne de ce document a vocation à être **barrée** par une ligne de
`docs/mesures/`. Une projection qui survit à sa mesure sans être barrée est
la dérive que ce dépôt documente depuis juillet.
