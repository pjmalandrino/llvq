# Roadmap recherche — LLVQ

> **Document vivant.** Dernière révision : 2026-09-01, depuis
> [`docs/audit-recherche-2026-09-01.md`](audit-recherche-2026-09-01.md) (le
> *pourquoi* de chaque piste vit là ; ici vit le *quoi, dans quel ordre, à quel
> prix, et ce qui tranche*). Il complète — sans les remplacer —
> [`BACKLOG.md`](BACKLOG.md) (RAF d'ingénierie et de publication) et
> [`PLAN.md`](PLAN.md) (les trois phases). Les gains attendus de chaque piste,
> étiquetés et ancrés, sont projetés dans
> [`projection-gains-2026-09-01.md`](projection-gains-2026-09-01.md). Il ne touche pas à la phase A en
> cours (A2 transfert 8B/14B tamponné, A3 dessiné) : décision d'opérateur
> `deaa449`, « A2 et A3 ont lieu quoi qu'il arrive ».
>
> **Règles, héritées du dépôt et non négociables ici** : (1) aucune mesure sans
> critère d'adoption *et* de kill écrits et tamponnés avant la première
> milliseconde ; (2) tout nombre est *mesuré* / *calculé* / *estimé* ; (3) tout
> ce qui recalibre passe sous σ = 5,2 % tant que M1 n'a pas rendu son verdict —
> donc **rien ne recalibre au 4B avant M1** ; (4) un lot n'est jamais justifié
> par « il pourrait renverser le verdict ».

---

## 0. La thèse de la roadmap en trois lignes

1. **Le front n'est plus le noyau, c'est le format** : aucun format déplié ne
   fait tenir un 70B sur 24 Go ; seul un format à index brut décodé
   séquentiellement le fait (audit §1.2, §3). → **Axe F.**
2. **La qualité ne se gagnera pas en cherchant mieux dans le même codebook**,
   mais en changeant l'altitude de l'objectif, la stabilité de l'estimateur et
   l'allocation (audit §4.1). → **Axe Q.**
3. **Rien de Q n'est mesurable avant d'avoir un plancher de bruit connu et
   une attribution par matrice** (audit §4.2, §4.6). → **Axe M, en premier.**

---

## 1. Vue d'ensemble

```
            sept. 2026            oct. 2026            nov. 2026            déc. 2026
Axe M   ████ M1 M2 M3 M4
Axe F   ████ F1a (papier) ██ F1b (banc gaussien) ████ F1c (format v2, 0,6B) ██ F1d (noyau) ██ F1e (4B)
Axe Q             ██ Q1 Q2 Q3 Q4a (0,6B, 3 graines)  ██ Q5 Q6a  ████ Q6c Q4b       ██ Q6d (si go)
Jalons  D0        D1                D2                 D3                     D4
```

| jalon | date visée | question tranchée | par quoi |
|---|---|---|---|
| **D0** | 2026-09-05 | la roadmap est-elle adoptée, et avec quel plafond ? | décision d'opérateur ; plafond proposé **30 $** hors Q6d et hors 32B |
| **D1** | fin sept. | le plancher de bruit est-il abaissable ? où vit la chute MMLU ? | M1 (étendue inter-graines sous shrinkage), M2 (attribution) |
| **D2** | mi-oct. | F1 est-il un codebook (rétention ≥ 91,0 %) ? lesquelles de Q1-Q4a passent à 0,6B ? | F1b, gates Q à 28 blocs |
| **D3** | mi-nov. | F1 est-il un noyau (t ≤ 1,15·t(QTIP)) ? Q gagne-t-il au 4B ? | F1d, un run 4B (7 $) |
| **D4** | déc. | le point 4B-F1 scellé bat-il le point servi sur les quatre axes ? faut-il Q6d ? | F1e + MMLU apparié |

---

## 2. Axe M — mesure et méthode (tout à ~0 $, tout en premier)

### M1 — Stabilité de l'estimateur de Hessienne

- **Hypothèse** : le σ = 5,2 % de F5 vient des termes **hors-diagonaux** de H,
  instables à 13,5 échantillons/dimension, par lesquels passe la rétroaction
  d'erreur (audit §4.1-2, littérature 2604.13806).
- **Protocole** : (a) sur le Mac, H pour trois matrices du 4B (`q/k/v`,
  `down`, `up`), 131 k tokens × 2 graines + 1 M tokens × 1 graine : corrélation
  des diagonales, corrélation des hors-diagonaux, spectre ; (b) shrinkage
  `H_ρ = ρ·H + (1−ρ)·diag(H)`, ρ ∈ {1, 0,9, 0,7, 0,5}, 0,6B / 28 blocs / 3
  graines ; (c) damping ∝ 1/√N sur le même banc.
- **Critère d'adoption** : l'**étendue inter-graines** à 28 blocs divisée par
  ≥ 2 pour un ρ* < 1, sans dégrader la ppl médiane de plus de l'étendue
  initiale. **Kill** : ρ* = 1.
- **Coût / durée** : 0 $ ; ~12 runs de 40 min sur Mac = 2 jours machine.
- **Livrable** : `docs/mesures/m1-hessienne-stabilite-<date>.txt`, préreg
  tamponné. **Ouvre** : Q1 (adoption du shrinkage), et rend Q2-Q7 lisibles.
- ⚠️ Ne rouvre **pas** le volume de calibration : on teste l'estimateur, pas
  la quantité de données.

### M2 — Attribution leave-one-out de la chute MMLU

- **Hypothèse** : la chute de −14,73 pp n'est pas uniforme par fonction ; la
  littérature 2025-2026 prédit `k_proj` et l'attention (audit §4.6).
- **Protocole** : sur le 4B scellé, restaurer **un type de matrice en f16**
  depuis le checkpoint (7 bras : `q`, `k`, `v`, `o`, `gate`, `up`, `down`),
  MMLU micro apparié, empreinte `65dcd53655e8bfa5`, dumps `qhash` par
  question, SE appariée attendue 0,43 pp.
- **Critère de lecture (pas d'adoption, c'est une mesure)** : un bras qui rend
  ≥ 3 pp apparié à lui seul désigne une cible de précision mixte (Q5) ; un
  profil plat (aucun bras > 1,5 pp) désigne la composition (Q6) plutôt que
  l'allocation.
- **Coût** : **≈ 1-2 $** (*estimé* depuis 0,49 $ mesuré pour trois bras).
- **Livrable** : `docs/mesures/m2-attribution-4b-<date>.txt` + table dans
  `docs/data/`. **Ouvre** : Q5.

### M3 — Métriques qui voient l'effondrement de calcul

- **Quoi** : (a) **entropie d'attention normalisée par couche**, f16 contre
  scellé, sur 64 fenêtres C4 (0 $, une passe) ; (b) **MMLU-STEM apparié**
  comme sous-score standard de tout gate de qualité (les matières qui tombent
  au hasard : algèbre, comptabilité, ML, physique).
- **Critère** : (a) est adopté comme diagnostic si son écart f16/scellé est
  > 3× son écart inter-fenêtres ; (b) devient la métrique de gate de Q6 et Q7.
- **Coût** : 0 $ (Mac) pour (a) ; (b) est un post-traitement des dumps.
- **Livrable** : `bin/attnent` (ou option de `bin/ppl`), une colonne STEM dans
  `mmlupair`.

### M4 — Outillage qui empêche la dérive (dette de l'audit §2.2)

- `ops/status.py` → `docs/ETAT.md` généré (compteurs `mesures/`, `jobs.csv`,
  `otsaudit`, config servie) + test CI qui échoue sur un compteur périmé dans
  les fichiers de reprise ;
- `[workspace.lints.rust] unsafe_code = "forbid"` + `[lints] workspace = true`
  sur les cinq crates du cœur ;
- compilation hôte des `.cuh` par `clang++` en CI pour les cinq décodeurs, sur
  le modèle de `host_e1v.cpp` ;
- entrée HISTORIQUE pour A2 (+13,45 %) ; `MACHINES.md:50-52` aligné sur v1.
- **Coût** : 0 $, ~2 jours. **Sans gate** : c'est de l'hygiène.

---

## 3. Axe F — le format sans dépliage

### F1 — « Leech-3×E₈ » : Λ₂₄ comme code de coset à trois sections

Fondement : Forney 1988 (treillis 3 sections / 256 états de Λ₂₄),
Lepowsky–Meurman 1982 (Λ₂₄ ⊂ E₈³) — audit §3.2-3.3. Format visé :
`[état : 8][s₁ : ~13][s₂ : ~13][s₃ : ~13][gain : 1]` = 48 bits, décodé par
trois lookups de type E8P et deux additions.

| étape | quoi | coût | critère d'adoption | kill |
|---|---|---|---|---|
| **F1a** papier | écrire la polarisation `(M, N)` de E₈, compter les états de Forney et l'alphabet de chaque section pour 47 bits, prouver la bijection par énumération (modèle : `classes_reproduce_theta_series`) | 0 $, ~1 sem. | le compte tient en 47 bits avec des tables **≤ 16 Kio** par section après factorisation bases × signes | états ou tables hors budget → passer à F2 |
| **F1b** banc gaussien | brancher le codebook dans `llvq-bench` (G4 : 20 000 blocs, graine figée, β-sweep) à 48 bits **empaquetés** | 0 $, ~1 sem. | rétention **≥ 91,0 %** (la boule Λ₂₄(12)+1 fait 92,14 ; la coquille 12 seule, 90,34, a été enterrée) | **< 90,3 %** |
| **F1c** format v2 + encodeur + 0,6B | encodeur = 256 états × 3 décodages E₈ ; format disque = format VRAM ; `codebook_fingerprint` v2 ; 0,6B / 28 blocs / même graine que `leech1c12` | 0 $ (Mac) | ppl à **±1 étendue inter-graines** (celle de M1) de `leech1c12` ; encodeur ≤ 656 µs/bloc | ppl hors bande sur 3 graines |
| **F1d** noyau | bras `tv_l3e8` dans `planesbench`, bras entrelacés, **QTIP témoin dans le même processus**, f64 à 1e-5 sur 1 105 920 lignes ; L40S puis A100 ; géométrie d'A3 si adoptée | **~1 $** | **t(F1) ≤ 1,15·t(QTIP)** ; b/poids noyau ≤ 2,20 | t > 1,5·t(QTIP) |
| **F1e** 4B scellé | quantification 4B en v2, `fusedrun`, MMLU apparié, b/param | **~8 $** (7,11 $ run + éval) | b/param modèle entier **≤ 2,6** ; MMLU ≥ 55,59 − 2·SE appariée ; tokens identiques au dense jusqu'au tie-break | MMLU < 53 % |

**Ce que F1 change s'il passe** : le transcodage disparaît (131 s → 0), le
disque et la VRAM sont le même objet, le débit devient flexible par bloc (F3),
et la thèse « 70B sur 24 Go » redevient une phrase mesurable (≈ 20 Go
*calculés*). **Ce qu'il coûte** : un format complet, et G5 rejoué.

### F2 — Treillis séquentiel + trellis shaping (repli conditionnel)

- **Déclencheur** : kill de F1a (états/tables hors budget) ou de F1b sur la
  mise en forme.
- **Quoi** : treillis minimal sur Λ₂₄/8ℤ²⁴, automate `(état, 2 bits) →
  (état', valeur)` de quelques Kio (mécanisme HYB de QTIP) ; mise en forme par
  trellis shaping (Forney 1992), Viterbi **à l'encodeur seulement**.
- **Contrainte posée d'avance** : ne se mesure **que** dans la géométrie
  d'A3 (grille fixe + split-K), jamais dans celle de `tv_planes` — un
  décodeur séquentiel à 24 pas met la latence des lookups sur le chemin
  critique de chaque lane.
- **Coût** : comme F1 ; **non budgété** tant que F1 vit.

### F3 — Débit flexible par rayon (conditionnel à F1c)

- **Quoi** : `cap` par matrice puis par ligne (44-50 bits/bloc) à budget total
  constant, guidé par M2 et par une saillance (Fisher ou gradient de perte).
- **Critère** : à b/param constant, MMLU apparié ≥ +2 pp sur l'allocation
  uniforme. **Kill** : < +1 pp.
- **Coût** : un run 4B (7 $) ; **après D3**.

### Écartées, pour mémoire (audit §3.6)

Table côté activations (≈ 5 Mo/token/matrice pour des codebooks de 2¹³) ·
dépliage par tensor cores (idée C, ~148 ops/bloc) · transcodage paresseux à
M = 1 (c'est E1v). Le transcodage paresseux **redevient exact à M ≥ 8** :
à noter pour le jour où le prefill (idée A) est servi — le format optimal
dépend de M.

---

## 4. Axe Q — perdre moins

Ordre imposé : **rien ici avant D1**. Tous les A/B de 0,6B suivent le
protocole du gate design C (28 blocs, même graine, puis 3 graines).

| id | piste | hypothèse | protocole | adoption | kill | coût | dépend de |
|---|---|---|---|---|---|---|---|
| **Q1** | shrinkage de H en production | M1 a rendu ρ* < 1 | rejouer `leech1c12` 0,6B à ρ*, 3 graines | étendue ÷ 2 tenue, ppl médiane ≤ +étendue | — | 0 $ | M1 |
| **Q2** | cible asymétrique (GPTAQ) + pondération de sortie (YAQA/KronQ) | le résidu vise `W·x̂` ; viser `W·x` (flux f16 parallèle) corrige la dérive accumulée | changement local à `calib.rs` (le flux f16 existe déjà en passe 1) ; puis poids par ligne depuis la covariance des gradients | Δppl ≥ 2× étendue (M1), stable 3 graines | < 1× étendue | 0 $ puis 7 $ | M1 |
| **Q3** | GPTQ en faisceau (K-best) | GPTQ = Babai glouton ; `shell_bests` rend 12 candidats | K ∈ {2, 4, 8} par bloc, faisceau sur la ligne, score `Tr(ΔW H ΔWᵀ)` partiel | Δppl ≥ 2× étendue, **σ inter-graines non augmenté**, encodeur ≤ K× | < 1× étendue | 0 $ (~10 h Mac à K = 4 pour le 4B) | M1 |
| **Q4a** | équi-norme inter-couches (Nagel–van Baalen, version VQ) | `down·diag(s)` et `diag(1/s)·{up,gate}` sont libres, `1/s` absorbé par l'échelle f16 de ligne déjà stockée ; idem `v`/`o` par dim de tête ; **pas** `q/k` (q_norm/k_norm) | test d'invariance **bit-exact** de la sortie du bloc à `s` quelconque, puis `s` égalisant les normes des blocs de 24 de `down` | Δppl ≥ 2× étendue | < 1× étendue | 0 $ | M1 |
| **Q4b** | cartes 24×24 apprises côté activations | `A_j` bloc-diagonale après la Hadamard : ~1 % de la matvec, ≈ 33 Mo au 4B, décodeur octet-identique ; la version diagonale = le FT d'échelles du papier, et absorbe le biais radial de +3,7 % | version **diagonale** d'abord (0 bit, un scalaire par position) ; puis pleine, apprise via Q6c | diagonale : Δppl ≥ 3 % (le biais radial mesuré) ; pleine : ≥ +2 pp MMLU apparié | — | 0 $ puis 7 $ | Q6c pour la pleine |
| **Q5** | précision mixte par fonction | M2 désigne une cible ; `k` seul = +0,05 b/poids à 4 bits, `q+k` = +0,26 | la cible en `cap` plus large (sous F1) ou en f16/4 bits (sous v1) ; MMLU apparié | ≥ +3 pp apparié pour ≤ +0,10 b/poids | < +1,5 pp | 7 $ | M2 |
| **Q6a** | distillation des **paramètres libres du format** | queue f16 (16,96 M), échelles de ligne, centroïdes, normes (≈ 0,2 M), embedding q8 : ~18 M paramètres, **0 bit de plus** (Norm Tweaking généralisé) | KL vers le f16 sur 2-5 M tokens C4+DCLM, une carte, LR petit, paramètres Leech gelés | ≥ +3 pp MMLU apparié, ppl ≤ | < +1,5 pp | **~3 $** | M3 |
| **Q6b** | EoRA / RILQ r ≤ 16 ou q8 | déjà chiffré (`BACKLOG` §3.3) : r = 32 f16 = +0,263 b/param | tel que `BACKLOG` §3.3 | ≥ +3 pp dans ≤ +0,25 b/param | — | ~3 $ | Q6a rendu |
| **Q6c** | relaxation différentiable de la recherche Leech | analogue Leech de BCJR-QAT : softmax de Boltzmann sur les K meilleurs points (`nearest_scaled` K-best), T → 0 rend le code dur | implémenter la relaxation, vérifier T→0 = `leech1c12` au bit ; apprendre Q4b et les échelles à 0,6B | gradient fini, T→0 bit-exact, Q4b pleine ≥ +2 pp | — | 0 $ puis 7 $ | Q3 (K-best) |
| **Q6d** | distillation KL bout-en-bout, Leech ré-affecté (PV-tuning) | UPQ / Reasoning-QAT : KL vers f16, PTQ comme init, domaine aligné | 4B, 20-50 M tokens, 1 carte | ≥ +6 pp apparié ; la ligne « with FT » du papier (17,05 → 9,26) comme repère | — | **dizaines de $, hors plafond**, go explicite | Q6a-c, D4 |
| **Q7** | composition du corpus (DCLM-edu, `BACKLOG` §3.2) | l'oracle ppl ne borne pas le raisonnement | tel que §3.2 **mais** gate = MMLU-STEM apparié + entropie d'attention (M3), 3 graines | ≥ +3 pp STEM apparié | < +1,5 pp | ~15 $ | M1, M3 |

---

## 5. Budget et calendrier (*estimés*)

| poste | $ | quand |
|---|---|---|
| Axe M (M1-M4) | ≈ 2 | sept. |
| F1a-c | 0 | sept.-oct. |
| Gates Q à 0,6B (Q1-Q4a, Q6c) | 0 | oct. |
| F1d noyau (L40S + A100) | ≈ 1 | nov. |
| Un run 4B pour la meilleure piste Q (Q2/Q3/Q4a) | ≈ 7 | nov. |
| Q6a distillation des paramètres libres | ≈ 3 | nov. |
| F1e 4B scellé + MMLU | ≈ 8 | déc. |
| Q5 ou F3 (le premier désigné par M2/D3) | ≈ 7 | déc. |
| **total roadmap** | **≈ 28** | **sous le plafond de 30 $** |
| hors plafond, go explicite : Q6d (dizaines), Q7 (15), 32B (≈ 62-80) | — | après D4 |

Rythme : ~1 journée machine Mac par semaine (encodeur K-best, H, 0,6B) ;
aucune carte avant novembre.

---

## 6. Ce que la roadmap **ne** fait pas

- Ne relance ni le volume de calibration, ni un format à ALU inchangée
  (`Golay70`, E1c, E3), ni la course de décodage sur la géométrie
  `tv_planes`, ni le 32B avant D4 — quatre interdits payés (audit §1.3).
- Ne réécrit pas le papier : la venue suivante et la restructuration
  (`revue-taco-2026-08-22.md`, option 2) sont un chantier de publication, pas
  de recherche ; F1e est le premier résultat qui justifierait un **second**
  papier (« le format sans dépliage »), pas une révision du premier.
- Ne préjuge pas de la phase A : A2 (8B/14B) et A3 suivent leur préreg. F1d
  **hérite** de la géométrie qu'A3 aura adoptée.

---

## 7. Décisions attendues de l'opérateur

| # | décision | échéance | défaut si silence |
|---|---|---|---|
| 1 | adopter la roadmap et son plafond (30 $) | D0 | rien ne démarre au-delà de M4 |
| 2 | tamponner les préregs de M1 et M2 (critères ci-dessus) | avant la première mesure | — |
| 3 | autoriser le format v2 (`codebook_fingerprint` change) | à F1b vert | F1 s'arrête au banc gaussien |
| 4 | choisir la première piste Q qui reçoit le run 4B (7 $) | D2 | celle au meilleur Δppl/étendue à 0,6B |
| 5 | go Q6d (hors plafond) | D4 | non |
| 6 | où s'écrit F1e s'il passe : second papier ou révision | D4 | second papier |

---

## 8. Journal des révisions

- **2026-09-01** — création, depuis l'audit du même jour. Aucune étape lancée ;
  aucun préreg tamponné.
- **2026-09-01** (soir) — pointeur vers la projection des gains par critère.
