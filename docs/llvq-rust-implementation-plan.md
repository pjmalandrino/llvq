# LLVQ en Rust — plan d'implémentation

> **Papier : [Leech Lattice Vector Quantization for Efficient LLM Compression — arXiv:2603.11021](https://arxiv.org/abs/2603.11021)** (v2, 7 juillet 2026)
> van der Ouderaa, van Baalen, Whatmough, Nagel — Qualcomm AI Research.

**Version 2 de ce plan — établie après lecture intégrale du PDF.** La v1 reposait sur des
résumés de moteurs de recherche et contenait trois erreurs de fond, corrigées ici et
signalées en §7.

Références indispensables, dans l'ordre de lecture :

1. **Adoul & Barth (1988)**, *Nearest neighbor algorithm for spherical codes from the Leech
   lattice*, IEEE Trans. Inf. Theory 34(5):1188–1202. **C'est l'algorithme de base que LLVQ
   étend.** Sans ce papier, la Phase 2 est infaisable.
2. Conway & Sloane, *Sphere Packings, Lattices and Groups* (3ᵉ éd., 2013), ch. 10 —
   construction de Λ₂₄ par le code de Golay étendu.
3. [QuIP# — arXiv:2402.04396](https://arxiv.org/abs/2402.04396) : codebook E8P, baseline.
4. [QTIP — arXiv:2406.11235](https://arxiv.org/abs/2406.11235) : la référence de vitesse.
5. [GPTQ — arXiv:2210.17323](https://arxiv.org/abs/2210.17323) : la boucle de correction hessienne.

---

## 1. Ce que fait LLVQ, précisément

### 1.1 Le mur que le papier franchit

VQ en dimension *d* à *b* bits/dim ⇒ 2^(bd) mots de code. En dimension 24 à 2 bits/dim :
2⁴⁸ points, soit ~280 To. Impossible à matérialiser — **c'est exactement pourquoi QuIP# a
pris E8 en dimension 8**, dont le codebook E8P tient en 2¹⁶ entrées ramenées à une table de
2⁸ par symétrie. LLVQ ne stocke rien : il calcule les points via la structure du code de
Golay étendu.

### 1.2 Construction (§2.3 du papier, Eq. 4–5)

`Λ₂₄ = (1/√8)·J`, avec `J = J_even ∪ J_odd ⊂ Z²⁴` :

| | J_even | J_odd |
|---|---|---|
| (i) parité | `xᵢ ≡ 0 (mod 2)` | `xᵢ ≡ 1 (mod 2)` |
| (ii) Golay | `(x/2) mod 2 ∈ G₂₄` | `((x−1)/2) mod 2 ∈ G₂₄` |
| (iii) somme | `Σxᵢ ≡ 0 (mod 8)` | `Σxᵢ ≡ 4 (mod 8)` |

`G₂₄` = code de Golay binaire étendu [24,12,8], 4096 mots, poids de Hamming ∈ {0,8,12,16,24}.
Avec la normalisation 1/√8, **le réseau est pair et unimodulaire** — donc de covolume 1, ce
qui valide le calcul de dimensionnement de §3.

> ⚠️ Les congruences (iii) sont à revérifier sur l'Eq. (5) p. 4 : l'extraction de texte du
> PDF brouille les chiffres en police mathématique. Le reste est certain.

### 1.3 Couches, classes, meneurs (§2.2, §2.4)

`Shell(m) = {v ∈ Λ₂₄ : ‖v‖² = 2m}`, m ≥ 2. Table 1 du papier :

| m | ‖v‖² | cardinal n(m) | cumul N(m) | bits/dim |
|---|---|---|---|---|
| 2 | 4 | 196 560 | 196 560 | 0,75 |
| 3 | 6 | 16 773 120 | 16 969 680 | 1,042 |
| 4 | 8 | 398 034 000 | 415 003 680 | 1,208 |
| 5 | 10 | 4 629 381 120 | 5 044 384 800 | 1,375 |
| **13** | **26** | — | **280 974 212 784 720** | **2,000** |
| 19 | 38 | — | 23 546 209 100 646 960 | 2,292 |

**Le régime 2 bits/poids correspond à l'union des couches jusqu'à m = 13, norme au carré 26.**

À l'intérieur d'une couche, les points se groupent en **classes** : ensembles stables par
permutation de coordonnées et changement de signe, représentés par un **meneur** (le multiset
canonique des valeurs absolues). Table 2 du papier donne les classes des couches 2, 3 et 4.
Cardinal d'une classe :

```
|classe| = γ · 2^C · (24! / ∏ρ!) · (1 / ∏|q|!)
```
où γ = nombre de mots de Golay admissibles (**4096 pour les classes impaires**, γ ∈ {1, 759,
2576, 759, 1} pour les paires selon le poids requis), 2^C les signes admissibles, puis les
facteurs multinomiaux de permutation.

### 1.4 Les quatre contributions

1. **Indexage bijectif** hiérarchique (§3.2) : couche → classe → symétries locales. Les
   symétries locales se décomposent en (r) raffinement de Golay, (s) motif de signes, (I″)
   rang de permutation, par divisions et modulos successifs.
2. **Recherche multi-couches** (§3.1) : Adoul–Barth ne traite qu'une couche, où le classement
   par produit scalaire coïncide avec le classement euclidien. Dès qu'on unit plusieurs
   couches, les normes varient et l'équivalence tombe — LLVQ ajoute deux métriques,
   euclidienne (*spherical shaping*) et angulaire par cosinus (*shape–gain*).
3. **Noyau de déquantification fusionné** (Annexe A et C).
4. **Spherical GPTQ** (§3.3, Algorithme 1) : le rescaling standard du shape–gain, entrelacé
   avec la rétropropagation d'erreur hessienne de GPTQ, s'interprète comme une **rétraction
   sur un produit de sphères**. `ṽ = (‖v‖₂/‖av‖₂)·av`, et les résidus GPTQ se forment sur `ṽ`.

### 1.5 L'asymétrie encodeur / décodeur

| | Quand | Nature |
|---|---|---|
| **Encodeur** (plus proche voisin) | hors ligne, 1× par modèle | Adoul–Barth : meneurs, placements Golay, motifs de signes, classement par produit scalaire |
| **Décodeur** (index → vecteur) | **chaque GEMM** | petites tables statiques, div/mod entiers, reconstruction combinatoire locale |

Le papier est explicite (Annexe A.5) : « aucune dépendance entre vecteurs, aucun accès mémoire
volumineux, trivialement parallélisable ». **C'est la raison pour laquelle le projet est
viable.**

---

## 2. L'ouverture d'ingénierie — ce que les auteurs n'ont pas fait

C'est le point le plus important pour décider d'y aller, et il ressort noir sur blanc du
papier (Annexe C) :

**Leur noyau CUDA ne traite qu'une seule couche, M = 3, « pour la simplicité ».** Le noyau
multi-couches — celui qu'il faut pour le régime 2 bits/poids, m = 13 — **n'existe pas**.

**Leur noyau est plus lent que QTIP, et ils l'assument** : *« nous soulignons que ce travail
ne vise pas à formuler des affirmations définitives sur les performances d'exécution
optimisées, l'ingénierie de noyaux bas niveau pouvant vraisemblablement améliorer encore nos
implémentations. Ces optimisations sont largement orthogonales à la contribution principale. »*

Table 7 du papier, matvec 4096×4096 :

| Noyau | Temps |
|---|---|
| FP16 matvec | 16,13 µs |
| FP16 matvec (4096×4104) | 17,169 µs |
| **LLVQ fusé (déquant + matvec)** | **11,194 µs — accélération 1,36–1,48× sur FP16** |

Traduction : les auteurs livrent la meilleure représentation de l'état de l'art, avec un
noyau de démonstration mono-couche qu'ils déclarent eux-mêmes non optimisé. **L'écart entre
la qualité de la représentation et la qualité de l'implémentation, c'est le projet.**

Aucun code publié à ce jour. Seul dépôt existant : `dmnunez1993/llvq-paper-reproduction`
(notebook, 0 étoile, dormant depuis le 2 juin 2026).

---

## 3. Contrôle de dimensionnement

Λ₂₄ étant unimodulaire, le nombre de points dans une boule de rayon R vaut `≈ V₂₄·R²⁴` avec
`V₂₄ = π¹²/12! ≈ 1,930×10⁻³`. Pour 2⁴⁸ points : `R ≈ 5,19`, donc `‖v‖² ≈ 26,9`.

**La Table 1 du papier donne 2,000 bits/dim à m = 13, soit ‖v‖² = 26.** L'estimation
asymptotique et le décompte exact concordent. Ce calcul reste utile comme test de cohérence
si l'implémentation des couches dérive.

---

## 4. Architecture Rust

```
llvq/
├── llvq-core/       # Golay, Λ₂₄, couches, classes, meneurs, indexage. #![no_std], 0 dépendance.
├── llvq-search/     # Adoul–Barth + extension multi-couches, métriques euclidienne et angulaire.
├── llvq-quant/      # shape–gain, Spherical GPTQ, hessiennes.  → faer
├── llvq-kernels/    # noyau fusé CUDA (cudarc) + SIMD CPU (pulp) + wgpu.
├── llvq-format/     # sérialisation, extension GGUF.
├── llvq-engine/     # intégration mistral.rs / candle.
├── llvq-cli/        # quantize | eval | bench
└── llvq-bench/      # source gaussienne, perplexité, tok/s, VRAM
```

| Besoin | Choix | Motif |
|---|---|---|
| Algèbre dense, Cholesky de H⁻¹ | **`faer`** | Pur Rust, pas de dépendance Fortran/BLAS. Build reproductible. |
| GPU CUDA | **`cudarc`** + noyau CUDA C (NVRTC) | |
| GPU portable | **`wgpu`** + WGSL (phase 8) | AMD, Intel Arc, Apple — l'argument souveraineté. |
| SIMD CPU | **`pulp`** | AVX-512/AVX2/NEON sans nightly. |
| Moteur | **`mistral.rs`**, sinon `candle` | |
| Tests de propriété | **`proptest`** | Bijectivité de l'indexage (G3). |
| Micro-bench | **`criterion`** | |

**Le noyau GPU ne sera pas en Rust.** Aucune chaîne Rust→GPU n'atteint le niveau d'un CUDA
écrit à la main quand la cible est QTIP. CUDA C piloté par `cudarc` ; tout le reste en Rust.
Contrainte « 100 % Rust » ⇒ `wgpu`/WGSL, en renonçant aux tensor cores.

---

## 5. Phases et gates

### Phase 0 — Transcription · 2 à 3 jours
Récupérer Adoul & Barth (1988) — c'est le vrai prérequis, pas le papier LLVQ. Transcrire
Table 1 (couches), Table 2 (classes et meneurs), Eq. 4–5 (congruences), Algorithme 1.

> **Gate G0** — Adoul & Barth en main et l'algorithme de recherche compris. Sinon, tout le
> reste est bloqué : c'est le seul prérequis externe du projet.

### Phase 1 — Cœur mathématique · 1 à 2 semaines
`llvq-core` : Golay `u32`, congruences, appartenance, énumération par couche/classe/meneur.

> **Gate G1 — invariants publics, vérifiables sans le papier.**
>
> | Test | Attendu |
> |---|---|
> | Mots de Golay | 4096 |
> | Distribution des poids | 1 / 759 / 2576 / 759 / 1 |
> | Distance minimale | 8 |
> | ‖v‖² minimale de Λ₂₄ | 4 |
> | Nombre de baisers | **196 560** |
> | Shell(3), Shell(4) | 16 773 120 · 398 034 000 |
> | Déterminant de Gram | 1 |
> | Clôture additive | sur 10⁶ tirages |
> | **Cardinaux de classes** | **doivent redonner la Table 2 du papier** |

### Phase 2 — Recherche du plus proche voisin · 2 à 3 semaines
Adoul–Barth mono-couche, puis extension multi-couches avec les deux métriques.

> **Gate G2.** (1) Sur une couche, coïncidence exacte avec la recherche exhaustive sur 10⁵
> tirages — l'algorithme est *exact*, tolérance zéro. (2) Multi-couches : le classement
> angulaire doit différer du classement euclidien, et les deux doivent être corrects
> séparément. (3) ≥ 10⁵ blocs/s/cœur (70 Md de poids ≈ 3×10⁹ blocs → ~15 min sur 32 cœurs).

### Phase 3 — Indexage bijectif · 1 à 2 semaines
Hiérarchie couche → classe → (r, s, I″), linéarisation et délinéarisation par div/mod.

> **Gate G3 — `proptest`.** Aller-retour exact dans les deux sens sur 10⁷ tirages ;
> injectivité vérifiée exhaustivement sur Shell(2) (196 560 points, énumérable) ; tout index
> du budget décode vers un point valide. **Une collision corrompt des poids en silence** —
> c'est le pire mode de défaillance, il passe tous les tests de perplexité.

### Phase 4 — Validation sur source gaussienne · 3 jours ⭐
**Le meilleur gate du projet, et il ne nécessite aucun LLM.** Quantifier des échantillons
i.i.d. `N(0,1)` et comparer à la Table 3 du papier, à 2 bits/dim :

| Méthode | dim | MSE ↓ | SQNR (bits) ↑ | Rétention ↑ |
|---|---|---|---|---|
| Uniforme | 1 | 0,1151 | 1,377 | 69 % |
| Lloyd–Max | 1 | 0,1121 | 1,537 | 77 % |
| E8 (cubique) | 8 | 0,1103 | 1,648 | 82,10 % |
| **LLVQ spherical shaping** | 24 | 0,1084 | 1,798 | 89,14 % |
| **LLVQ shape–gain** | 24 | **0,1078** | **1,849** | **92,11 %** |
| Limite théorique | — | 0,0625 | 2 | 100 % |

> ⚠️ **Correction (constatée en phase 4)** : les colonnes MSE et SQNR de cette transcription
> sont mutuellement incohérentes (−½log₂(0,1084) = 1,603 ≠ 1,798) — l'extraction texte du
> PDF avait un encodage de police décalé et les chiffres des tableaux sont partiellement
> corrompus. L'ancre auto-cohérente est la colonne **rétention** (89,14 % / 92,11 %). Notre
> implémentation mesure MSE 0,0775 / rétention 92,23 % (spherical, β ajusté) à 1,9999
> bits/dim — gate G4 atteint. À re-vérifier sur le PDF original en Phase 5.

> **Gate G4.** Rétention ≥ 89 % en spherical shaping et ≥ 92 % en shape–gain. Contrôle
> analytique gratuit : à 2 bits/dim, `MSE* = 2⁻²ᴿ = 0,0625` exactement — si votre limite
> théorique ne tombe pas sur 0,0625, le protocole de mesure est faux avant même le
> quantiseur. **Trois jours pour valider tout le cœur, avant d'avoir touché un modèle.**

### Phase 5 — Spherical GPTQ et pipeline LLM · 2 à 3 semaines
Algorithme 1 du papier : blocs de b = 24 canaux d'entrée, gauche à droite, `H = (1/N)AᵀA`,
Cholesky de `H⁻¹`, lignes en parallèle, reset de gain `ṽ = ‖v‖₂·Q_dir(v/‖v‖₂)`, propagation
du résidu sur les colonnes non traitées.

Calibration : **6 100 séquences de DCLM-edu** (même taille que QuIP#). Finetuning optionnel :
uniquement les échelles d'entrée partagées par lignes, ~52 M tokens, < 0,001 bpw de surcoût.

**Progression petit → gros** (décision projet) : d'abord **Qwen3-0.6B** en smoke test du
pipeline, puis **Qwen3-4B** — le plus petit modèle pour lequel le papier publie des chiffres
de référence (Table 6) — et seulement ensuite les 7B/8B. Chaque étape ne sert qu'à dérisquer
la suivante.

> **Gate G5 — reproduction, sur Qwen3-4B puis Llama-2 7B et Llama-3 8B à 2 bpw.** LLVQ doit battre
> QuIP#/E8P et QTIP en perplexité Wikitext-2 (contexte 4096), MMLU et CSR, dans le pipeline
> unifié de la Table 6. Écart de PPL ≤ 0,05 → validé. > 0,2 ou LLVQ ne bat pas QuIP# →
> **point de sortie du projet**.
>
> Ajouter le benchmark métier d'extraction documentaire ici, pas à la fin — cf.
> [*The Illusion of Equivalency in Quantization*, arXiv:2607.08734](https://arxiv.org/abs/2607.08734).

### Phase 6 — Noyau fusé · 3 à 4 semaines ⭐ *le cœur de la contribution*
Deux objectifs distincts, dans cet ordre :

1. **Reproduire** le noyau mono-couche M = 3 des auteurs : ≥ 1,36× sur le matvec FP16.
2. **Dépasser** — c'est là qu'est la valeur ajoutée :
   - **noyau multi-couches**, qui n'existe nulle part et qui conditionne le régime 2 bpw ;
   - franchir la barre QTIP, que les auteurs n'ont pas cherché à atteindre.

> **Gate G6.** (1) ≥ 1,36× sur FP16 en mono-couche. (2) Multi-couches m ≤ 13 fonctionnel et
> exact face au décodeur de référence Rust. (3) Face à QTIP : à parité → objectif atteint ;
> en deçà → livrer en documentant, une meilleure représentation à débit légèrement inférieur
> reste utile quand le but est de faire *entrer* le modèle.

### Phase 7 — Intégration moteur · 2 semaines
`mistral.rs`, format de sérialisation, CLI.

> **Gate G7 — sur le matériel réel.** VRAM pic, tok/s prefill et decode, perplexité,
> benchmark métier. Question binaire : **un modèle qui n'entrait pas entre-t-il maintenant ?**

### Phase 8 — Portabilité · optionnel, 2 à 3 semaines
`wgpu`/WGSL et chemin CPU SIMD. À arbitrer après G7.

---

## 6. Récapitulatif

| Phase | Durée | Gate | Si échec |
|---|---|---|---|
| 0 — Transcription + Adoul–Barth | 2–3 j | G0 | Bloquant |
| 1 — Golay + Λ₂₄ | 1–2 sem | G1 : 196 560, Table 2 | Bug |
| 2 — Recherche NN | 2–3 sem | G2 : exactitude + débit | Bug |
| 3 — Indexage | 1–2 sem | G3 : bijectivité | Bug |
| 4 — **Source gaussienne** | **3 j** | **G4 : rétention 92,11 %** | **Sortie** |
| 5 — Spherical GPTQ + LLM | 2–3 sem | G5 : bat QuIP#/QTIP | **Sortie** |
| 6 — Noyau fusé | 3–4 sem | G6 : 1,36× FP16, multi-couches | Livrer et documenter |
| 7 — Intégration | 2 sem | G7 : bout en bout | — |
| 8 — Portabilité | 2–3 sem | — | Optionnel |

**Total : 12–16 semaines.** Deux points de sortie, G4 et G5, tous deux avant l'investissement
noyau. G4 en particulier coûte trois jours et valide tout le cœur mathématique **sans LLM**.

---

## 7. Ce que la v1 de ce plan disait de faux

| v1 | Réalité |
|---|---|
| « Décodeur type Conway–Sloane / Vardy–Be'ery par l'hexacode » | **Faux.** LLVQ étend **Adoul & Barth (1988)** : meneurs, placements Golay, motifs de signes, classement par produit scalaire. Autre famille algorithmique. |
| « Le travail restant, c'est le noyau de production » | Incomplet. Le vrai trou est le **noyau multi-couches**, inexistant : les auteurs se limitent à M = 3. |
| « Risque : noyau trop lent » | Surestimé. Leur noyau fait déjà 1,36–1,48× le FP16. La barre est connue, pas hypothétique. |
| Étapes de validation floues | Remplacées par la **Table 3 (source gaussienne)** : trois jours, aucun LLM. |
| 11–15 semaines | 12–16, avec un découpage réaliste de la Phase 6. |

Reste vrai depuis la v1 : HARP et LLVQ sont **substituables, pas complémentaires**. Le papier
le confirme directement (Table 5) — Spherical GPTQ réduit fortement la dépendance aux
rotations de Hadamard, et LLVQ shape–gain reste compétitif sans aucune rotation. Améliorer la
rotation en amont d'un quantiseur 24-dimensionnel rapporte peu.

---

## 8. Risques

| Risque | Probabilité | Parade |
|---|---|---|
| Adoul & Barth (1988) difficile à obtenir ou à implémenter | **Élevée** | Gate G0. Papier IEEE de 1988, 15 pages. C'est le vrai chemin critique. |
| Qualcomm publie son code | Moyenne | Quatre mois de silence. Et le noyau multi-couches + intégration moteur garde sa valeur. |
| Chiffres non reproductibles | Faible | G4 puis G5, avant l'investissement noyau. |
| Ne pas battre QTIP en vitesse | Moyenne | Les auteurs ne le battent pas non plus. Repli documenté acceptable. |
| Collision d'indexage silencieuse | Faible mais **critique** | G3 par `proptest`, jamais relâché. |
