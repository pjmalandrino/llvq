# Notes de lecture — arXiv:2603.11021v2 (LLVQ)

> *Leech Lattice Vector Quantization for Efficient LLM Compression*
> van der Ouderaa, van Baalen, Whatmough, Nagel — Qualcomm AI Research, v2 du
> 7 juillet 2026, 26 pages.
>
> Relevé le 2026-07-28 **par rendu image** du PDF (voir « Comment lire le
> PDF » ci-dessous). Tous les chiffres de ce fichier ont été lus à l'écran,
> pas extraits par script.

## Comment lire le PDF (le piège, résolu)

L'extraction texte du PDF est corrompue — encodage de police décalé, et des
chiffres qui se dédoublent (`0.084` ressort en `0.1084`). **Ne jamais faire
confiance à `pdftotext` sur ce document.** Le rendu image, lui, est parfait :

```python
import fitz  # pymupdf
d = fitz.open("2603.11021v2.pdf")
p = d[6]                                   # page 7, index 0
clip = fitz.Rect(p.rect.x0+60, p.rect.y0+395, p.rect.x0+330, p.rect.y0+450)
p.get_pixmap(matrix=fitz.Matrix(9, 9), clip=clip).save("table3.png")
```

Un zoom ×9 sur la zone d'un tableau le rend parfaitement lisible.

## Plan du papier

| § | Contenu | Utile pour |
|---|---|---|
| 2 | Λ₂₄, coupes en boules, code de Golay étendu, classes et leaders | G1/G2, déjà fait |
| 3.1 | Recherche étendue à `Λ₂₄(m)`, métriques euclidienne et angulaire | G2b, déjà fait |
| 3.2 | Schéma d'indexage hiérarchique (coquille → classe → symétrie locale) | G3, déjà fait |
| 3.3 | Spherical GPTQ = rétraction sur un produit de sphères | **G5** |
| 4 | Source gaussienne, SQNR et rétention (Table 3) | G4, déjà fait |
| 5 | Résultats LLM (Tables 4, 5, 6) | **G5, cibles** |
| A | Déquantifieur (index → vecteur) | G3/G6 |
| **B** | **Algorithme 1** — shape–gain + corrections hessiennes | **G5** |
| C | Noyau CUDA fusé (Table 7) | G6 |
| D | Shape–gain et codes sphériques depuis un réseau | G5 |
| E | Schémas-blocs quantifieur/déquantifieur (Fig. 2–5) | G5 |
| F | Échelles optimales en forme close, corrections hessiennes | **G5** |
| G | Coquille unique vs union de coquilles (Fig. 6) | G4/G6 |
| H | Spherical shaping vs shape–gain (Table 8) | **G4** |
| **I** | **Spherical GPTQ, Algorithmes 2 et 3**, ablation Hadamard (Table 9) | **G5** |
| J | Llama-3.2 1B et 3B (Table 10) | G5 |

---

## Table 3 (§4) — rétention gaussienne à 2 bits/dim

| Méthode | Dim | MSE | SQNR (bits) | Ret (%) |
|---|---|---|---|---|
| Uniform | 1 | 0,15 | 1,37 | 69 |
| Lloyd-Max | 1 | 0,12 | 1,53 | 77 |
| E8 (cubic) | 8 | 0,103 | 1,64 | 82,0 |
| **LLVQ/Leech [spherical shaping]** | 24 | **0,084** | **1,79** | **89,4** |
| **LLVQ/Leech [shape-gain]** | 24 | **0,078** | **1,84** | **92,1** |
| Limite théorique | — | 0,0625 | 2 | 100 |

Auto-cohérent : −½log₂(0,084) = 1,7885 ; 1,79/2 = 89,4 %.

## Table 8 (Annexe H) — la même comparaison, détaillée

C'est **la** table à comparer à notre G4, parce qu'elle nomme le codebook.

| Méthode | Code | Bits/dim | MSE | SQNR | Ret (%) |
|---|---|---|---|---|---|
| Leech (spherical bounding) | `Λ₂₄(13)` | 2,0 | 0,084 | 1,787 | 89,37 |
| Leech (shape-gain) | `norm(Λ₂₄(13))` + 0 bit de gain | 2,00000 + 0 | 0,085 | 1,782 | 89,12 |
| **Leech (shape-gain)** | `norm(Λ₂₄(12))` + 1 bit de gain | 1,95833 + 0,04167 | **0,078** | **1,843** | **92,14** |
| Leech (shape-gain) | `norm(Λ₂₄(11))` + 2 bits de gain | 1,91667 + 0,08333 | 0,080 | 1,825 | 91,24 |
| Leech (shape-gain) | `norm(Λ₂₄(10))` + 4 bits de gain | 1,83333 + 0,16667 | 0,085 | 1,780 | 89,01 |

Conclusions du papier :
1. Le shape–gain bat le spherical shaping.
2. L'heuristique « 1/n des bits au gain » (soit 2 bits pour n = 24) est
   raisonnable mais **pas optimale** : l'optimum empirique est **1 bit**.
3. Recommandation : 1 ou 2 bits de gain par vecteur de 24 à 2 bits/dim.

## Table 6 (§5.3) — cibles LLM, **notre référence G5**

Wikitext-2 à 4096 de contexte, pipeline unifié des auteurs, 2 bits/poids.

**Qwen3-4B** (le plus petit modèle du papier avec des chiffres) :

| Méthode | FT | Wiki ↓ | MMLU ↑ | CSR ↑ |
|---|---|---|---|---|
| Baseline FP16 | — | 12,41 | 70,2 | 71,2 |
| GPTQ + Rotation (Quarot) | non | 280,7 | 26,3 | 43,6 |
| Quip#/E8P12 | non | 21,15 | 48,6 | 57,2 |
| QTIP (3INST) | non | 17,04 | 57,4 | 63,5 |
| LLVQ [spherical shaping] | non | **21,80** | 50,5 | 58,7 |
| LLVQ [shape-gain, 2 bit gain] | non | **15,54** | 59,3 | **64,1** |
| LLVQ [shape-gain, 0 bit gain] | non | 17,05 | **60,7** | 63,6 |
| Quip#/E8P12 | oui | 10,52 | 52,9 | 65,2 |
| QTIP (3INST) | oui | 9,61 | 59,5 | 66,9 |
| LLVQ [spherical shaping] | oui | 10,13 | 54,9 | 65,1 |
| LLVQ [shape-gain, 2 bit gain] | oui | 9,51 | 60,9 | 67,6 |
| LLVQ [shape-gain, 0 bit gain] | oui | **9,26** | **62,8** | 66,1 |

Qwen3-8B, sans FT : baseline 8,99 / QTIP 11,17 / LLVQ sg-2bit 10,82 /
LLVQ sg-0bit **10,19**. Avec FT : LLVQ sg-0bit **7,59**.

> ⚠️ **Le spherical shaping perd contre QTIP** (21,80 vs 17,04 sur Qwen3-4B
> sans fine-tuning). Seul le **shape–gain** gagne. Le « fine-tuning » ici
> n'est qu'un apprentissage des échelles par colonne (< 0,001 bit/poids,
> ~52 M tokens), pas un entraînement de bout en bout.

---

## Algorithme 1 (Annexe B) — shape–gain avec corrections hessiennes

```
Entrées : W ∈ R^{d_out × d_in}, X ∈ R^{N × d_in} (calibration),
          taille de bloc b = 24, quantifieur de direction Q_dir (Leech),
          quantifieur de gain optionnel Q_gain

 1  pour chaque couche l :
 2      H  ← (1/N)·Xᵀ X
 3      U  ← chol(H⁻¹)ᵀ                      # H⁻¹ = Uᵀ U, U triangulaire sup.
 4      W̃  ← W^(l)
 5      partitionner {1..d_in} en blocs Q_1..Q_m de taille b (le dernier peut être plus court)
 6      pour t = 1..m :
 7          Q ← Q_t,   R ← ∪_{u>t} Q_u
 8          pour chaque ligne i = 1..d_out (en parallèle) :
 9              v  ← W̃_{i,Q}
10              v̂  ← ‖v‖₂ · Q_dir(v/‖v‖₂)             # reset de gain
11              option : v̂ ← Q_gain(‖v‖₂) · Q_dir(v/‖v‖₂)
12              Ŵ_{i,Q} ← v̂
13          fin
14          E_{:,Q} ← W̃_{:,Q} − Ŵ_{:,Q}
15          W̃_{:,Q} ← Ŵ_{:,Q}
16          si R ≠ ∅ :
17              W̃_{:,R} ← W̃_{:,R} − (E_{:,Q} U_QQ⁻¹) U_QR
18          fin
19      fin
20      W^(l) ← W̃
21  fin
```

Points qui comptent :
- **Blocs de canaux d'entrée**, gauche→droite ; les `d_out` lignes sont
  indépendantes et se parallélisent.
- L'erreur `E` se calcule sur `W̃` **courant** (déjà compensé), pas sur `W`
  d'origine. Ligne 14 : c'est la convention GPTQ standard.
- La correction ligne 17 est un solve triangulaire à droite dans `U_QQ`,
  jamais une inversion explicite.

## Algorithme 3 (Annexe I.1) — Spherical GPTQ + échelles de groupe

C'est **la configuration recommandée** par le papier (0 bit de gain).

```
Entrées : W ∈ R^{k × d}, H SPD ∈ R^{d × d}, blocs Q_1..Q_m, quantifieur Q, amortissement λ ≥ 0

 1  U ← chol(H⁻¹)ᵀ
 2  W̃ ← W
 3  pour t = 1..m :
 4      Q ← Q_t,  R ← ∪_{u>t} Q_u
 5      W̃_{:,Q} ← (‖W̃_{:,Q}‖ / ‖Q(W̃_{:,Q})‖) · Q(W̃_{:,Q})       # rétraction
 6      E_{:,Q} ← W_{:,Q} − W̃_{:,Q}
 7      si R ≠ ∅ : W̃_{:,R} ← W̃_{:,R} − (E_{:,Q} U_QQ⁻¹) U_QR
10  fin
    # raffinement final des échelles par ligne, dans la métrique hessienne
11  pour i = 1..k :
12      M_i[p,q] ← W̃_{i,Q_p} H_{Q_p Q_q} W̃_{i,Q_q}ᵀ      ∀ p,q ∈ {1..m}
13      r_i[p]   ← W̃_{i,Q_p} H_{Q_p,:} W_{i,:}ᵀ           ∀ p
14      s_i      ← (M_i + λI)⁻¹ r_i
15      pour p = 1..m : W̃_{i,Q_p} ← s_i[p] · W̃_{i,Q_p}
18  fin
```

> ⚠️ **Deux ambiguïtés de notation à trancher à l'implémentation.**
> (a) La ligne 5 écrit la rétraction avec une norme de Frobenius sur tout le
> bloc de colonnes, alors que le texte de l'annexe I la définit **par ligne**
> (Éq. 17 : `Ŵ_{i,B} = (‖W_{i,B}‖₂ / ‖W̃_{i,B}‖₂)·W̃_{i,B}`). Le texte est
> explicite — « quantization is performed row-wise on each row-block […] and
> we apply the same retraction per row » — donc **implémenter par ligne**.
> (b) La ligne 6 écrit `E ← W − W̃` avec le `W` **d'origine**, alors que
> l'Algorithme 1 (ligne 14) et l'Algorithme 2 (ligne 6) utilisent le `W̃`
> **compensé**. La convention GPTQ standard est celle des Alg. 1/2 ;
> l'Algorithme 3 est vraisemblablement un raccourci d'écriture. **Suivre
> l'Alg. 1.**

## Échelles optimales en forme close (Annexe F.1)

Le quantifieur de forme est **invariant d'échelle** (`q(sw) = q(w)` pour
`s > 0`), donc pas de recherche linéaire sur β. L'échelle qui minimise
l'erreur de reconstruction dans l'espace des poids est la projection :

```
β* = argmin_β ‖w − β q‖²  =  (qᵀ w)/(qᵀ q)          avec q = q(w)
```

et par bloc `β*_i = q(w_i)ᵀ w_i / (q(w_i)ᵀ q(w_i))`.

Dans l'espace des activations (erreur de sortie locale), avec
`A := [q(w_1)x_1, …, q(w_G)x_G]` et `b := Wx`, les échelles optimales par
groupe sont les moindres carrés `β* = (AᵀA + λI)⁻¹ Aᵀ b`.

## Corrections hessiennes (Annexe F.2)

Objectif local standard : `L = E‖ΔW x‖² = Tr(ΔW H ΔWᵀ)` avec `H = E[xxᵀ]`,
approché par `Ĥ = (1/N) XᵀX`. En partitionnant en `Q` (quantifié) et `R`
(restant), la correction analytique est

```
Δw*_R = − H_RR⁻¹ H_RQ Δw_Q        ⟺        ΔW*_{:,R} = − ΔW_{:,Q} H_QQ⁻¹ H_QR
```

que l'on implémente sans inverse via le Cholesky de `H⁻¹` (forme GPTQ).

> Le papier note lui-même la limite : cet objectif local traite les couches
> comme découplées et ignore la propagation d'erreur inter-couches. Des
> substituts de courbure plus fidèles donneraient mieux mais coûtent des
> passes arrière. **Les gains d'une meilleure hessienne sont orthogonaux à
> LLVQ** — donc hors périmètre pour G5, où il faut comparer à iso-correction.

## Ablation Hadamard (Annexe I.2, Table 9 — Llama-2 7B, sans FT)

> ✅ **Re-transcrite intégralement par rendu image le 2026-08-04.** La version
> précédente, partielle, portait **six cellules fausses** (signalées ci-dessous)
> et omettait les lignes `Input`, dont l'absence a produit un contresens sur la
> rotation de sortie. Baseline Llama-2 7B : 5,12 / 45,7 / 70,4.

| Code | Correction | Hadamard | Wiki ↓ | MMLU ↑ | CSR ↑ |
|---|---|---|---|---|---|
| LLVQ [spherical shaping] | GPTQ | aucune | **191,90** ⚠️ | 24,0 | **53,5** ⚠️ |
| LLVQ [spherical shaping] | GPTQ | Input | 6,80 | 35,1 | 65,4 |
| LLVQ [spherical shaping] | GPTQ | Input+Output | 7,61 | 33,4 | 62,1 |
| LLVQ [sph. shaping] (forced ang.) | **Spherical GPTQ** | aucune | **6,90** | 37,4 | **63,8** ⚠️ |
| LLVQ [sph. shaping] (forced ang.) | Spherical GPTQ | Input | 6,70 | 35,1 | 65,4 |
| LLVQ [sph. shaping] (forced ang.) | Spherical GPTQ | Input+Output | 6,75 | 36,9 | 63,8 |
| LLVQ [shape-gain 2 bit] (forced eucl.) | GPTQ | aucune | 13,17 | **26,5** ⚠️ | **58,5** ⚠️ |
| LLVQ [shape-gain 2 bit] (forced eucl.) | GPTQ | Input | 7,28 | 34,1 | 62,8 |
| LLVQ [shape-gain 2 bit] (forced eucl.) | GPTQ | Input+Output | 7,31 | 35,3 | 62,8 |
| LLVQ [shape-gain 2 bit] | **Spherical GPTQ** | aucune | **7,27** | **29,8** ⚠️ | 61,5 |
| LLVQ [shape-gain 2 bit] | Spherical GPTQ | Input | 6,90 | 36,0 | 63,6 |
| LLVQ [shape-gain 2 bit] | Spherical GPTQ | Input+Output | 6,83 | 34,9 | 64,6 |

⚠️ = cellule corrigée le 2026-08-04. Anciennes valeurs fausses : Wiki 91,90
(un `1` initial perdu par l'extraction texte), CSR 37,7 · 65,9 · 56,5, MMLU
26,3 · 29,3.

**Le résultat le plus spectaculaire du papier** : sans rotation, le GPTQ
euclidien s'effondre (**191,90** de perplexité) alors que le Spherical GPTQ
tient (6,90). La dérive radiale est le mode de défaillance dominant, et la
préservation de norme l'élimine — d'où le PTQ *Hadamard-free*.

> 🕳️ **Ce que l'absence des lignes `Input` a coûté.** On a longtemps cité
> « la rotation Input+Output vaut +5,6 pp de MMLU » en comparant 29,3 (aucune
> rotation) à 34,9 (Input+Output). Ce n'est pas l'étage de sortie qu'on mesure
> ainsi, c'est **toute** la rotation. À `Input` fixé — notre configuration —
> l'étage de sortie vaut, sur les quatre familles :
> −1,7 · **+1,8** · **+1,2** · −1,1 pp. **Moyenne ≈ 0.**
> La rotation de sortie ne peut donc pas expliquer notre déficit de 4,8 pp sur
> MMLU. Retiré du README et de la carte HF le 2026-08-04.

Conclusions de l'annexe :
1. Le Spherical GPTQ améliore le GPTQ euclidien, sans toucher au codebook.
2. Il est **d'autant plus efficace que la distorsion angulaire du code est
   faible** — donc particulièrement pour LLVQ.
3. LLVQ gagne sous les deux corrections : l'avantage vient de la
   *représentation*, pas de l'heuristique de correction.
4. **Le budget de bits glisse vers les directions** : avec un code à faible
   distorsion angulaire, l'optimum passe de 2 bits de gain (GPTQ euclidien) à
   **0 bit de gain** (Spherical GPTQ) — toute la capacité aux directions, les
   magnitudes étant tenues par la contrainte sphérique en haute précision
   pendant GPTQ puis par la mise à jour close de l'Algorithme 3.

## Coquille unique vs union (Annexe G)

> ✅ **Relu par rendu image le 2026-08-04**, et il faut être précis ici :
> c'est la section sur laquelle on interroge les auteurs.

**Ce qu'ils mesurent** : la distance angulaire au plus proche voisin,
`D(x, q(x)) = arccos(xᵀq(x))/π`, sur une source **radialement uniforme**
(gaussienne normalisée), en fonction de `log₂(N)/d`. Figure 6, violons.

- **Key finding 1 — « Union of shells provide lowest angular distortion »** :
  l'union donne « a slightly better **Gaussian rate–distortion curves** »
  comparée aux coquilles individuelles. Citation exacte de la dernière phrase :
  > « We therefore adopt this approach **in our method** and recommend doing
  > the same. »
- **Key finding 2 — « Single shell provides a simpler algorithm »** : l'écart
  est **petit**, et « from a hardware perspective, using a single shell offers
  significant advantages. In particular, a constant norm implies a fixed
  scaling between dot products, eliminating the need to rescale intermediate
  dot product results before aggregation (as in group-wise or block
  quantization), along with its associated complications. »

> 🚨 **Deux erreurs de notre côté, corrigées le 2026-08-04.**
> 1. On citait la phrase **sans « in our method »**, et sans points de
>    suspension. À ne pas envoyer telle quelle à ses auteurs.
> 2. On répétait que « le papier mesure une distorsion angulaire, pas une
>    rétention MSE — donc deux métriques différentes ». **Le Key finding 1
>    nomme explicitement les courbes débit–distorsion gaussiennes.** C'est
>    notre métrique. Leur affirmation couvre donc bien ce qu'on mesurait, et
>    notre mesure corrigée (union gagnante à débit égal) **les confirme** au
>    lieu de les contredire.
>
> L'argument matériel n'est pas non plus « à eux de le découvrir » : le Key
> finding 2 le pose plus complètement qu'on ne le leur créditait. La seule
> question qui reste vraiment ouverte est donc : **ils nomment l'avantage
> matériel et adoptent quand même l'union — est-ce sur la seule courbe de
> distorsion, ou ont-ils mesuré le coût du rééchelonnage dans un noyau
> multi-coquilles ?** C'est ce qu'il faut leur demander.

## Table 7 (Annexe C) — noyau CUDA fusé

| Forme | Noyau | Temps |
|---|---|---|
| FP16 matvec | (4096×4096)·(4096×1) | 16,3 µs |
| FP16 matvec | (4096×4104)·(4104×1) | 17,69 µs |
| **LLVQ-FP16 (déquant + matvec fusés)** | (4096×4104)·(4104×1) | **11,94 µs** (1,36× / 1,48×) |

Le papier précise : noyau limité à **une seule coquille (M = 3)** « pour la
simplicité », **plus lent que QTIP**, et les auteurs déclarent l'optimisation
bas niveau « largement orthogonale » à leur contribution.

## Déquantifieur (Annexe A)

Index global → vecteur, en quatre étapes : (1) identifier la coquille par
recherche dans la table des cumuls `N(k) < I ≤ N(k+1)` ; (2) identifier la
classe par les offsets cumulés de la coquille ; (3) déplier les symétries
locales par `r = I_class mod A`, `I' = ⌊I_class/A⌋`, `s = I' mod 2^B`,
`I'' = ⌊I'/2^B⌋` — `r` sélectionne le raffinement Golay, `s` le motif de
signes, `I''` le rang de permutation ; (4) reconstruire depuis le leader.

Aucune dépendance entre vecteurs, aucun accès mémoire large : trivialement
parallélisable par blocs de 24 — c'est l'argument du noyau GPU.
