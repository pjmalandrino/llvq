# LLVQ — contexte projet (passation de session)

> Ce fichier est chargé automatiquement par Claude Code. Il contient tout ce
> qu'une nouvelle session doit savoir pour reprendre le travail sans relire
> l'historique.

## 1. Objectif

Réduire le coût d'inférence LLM pour de la **souveraineté** : faire tenir de
plus gros modèles sur du matériel local. Le seul levier qui change la classe
de modèle qu'on peut charger, c'est le nombre de bits par poids. À 2 bits, un
70B passe de 140 Go à 18 Go — il rentre sur une carte 24 Go.

On implémente en Rust le papier **LLVQ** : quantification vectorielle des
poids sur le réseau de Leech Λ₂₄, état de l'art à 2 bits/poids.

- **Papier** : [arXiv:2603.11021](https://arxiv.org/abs/2603.11021) —
  van der Ouderaa, van Baalen, Whatmough, Nagel (Qualcomm AI Research, 2026).
  *Le PDF est chez l'utilisateur* — le demander plutôt que de tenter arXiv.
- **Prérequis externe non résolu** : Adoul & Barth (1988), *Nearest neighbor
  algorithm for spherical codes from the Leech lattice*, IEEE Trans. Inf.
  Theory 34(5):1188–1202. On a re-dérivé ce qu'il fallait sans lui, mais
  l'avoir aiderait pour la Phase 2c (perf).
- Plan complet et gates : `docs/llvq-rust-implementation-plan.md`.
- Veille amont (pourquoi ce papier plutôt qu'un autre) :
  `docs/inference-cost-reduction-2026.md`.

⚠️ **Piège de transcription.** Le PDF a un encodage de police décalé (+1 par
glyphe : « K d d b g » = « Leech »). Les chiffres des tableaux extraits par
script sont **partiellement corrompus** — la colonne MSE de la Table 3
(0,1084 / 0,1078) est incohérente avec sa propre colonne SQNR
(−½log₂(0,1084) = 1,603 ≠ 1,798). L'ancre auto-cohérente est la **rétention**
(89,14 % spherical / 92,11 % shape–gain). Toujours revérifier un chiffre du
papier sur le PDF original avant de s'en servir comme cible.

## 2. Architecture

```
llvq-core/     Golay [24,12,8] + Λ₂₄ + couches. ZÉRO dépendance, forbid(unsafe).
llvq-search/   Recherche NN exacte, classes, moteur générique m≤13, indexage.
llvq-bench/    Évaluation débit-distorsion sur source gaussienne.
```

Commandes :
```bash
cargo test --release -- --include-ignored   # suite complète, ~45 s
cargo test                                   # suite rapide (les tests lourds sont ignored en debug)
cargo run --release -p llvq-bench            # tableau qualité
cargo clippy --all-targets                   # doit rester à zéro warning
```

## 3. État — 5 gates sur 7

| Gate | Contenu | Statut |
|---|---|---|
| G1 | Invariants Λ₂₄/Golay (nombre de baisers 196 560, Shell(3), série thêta) | ✅ |
| G2 | Recherche NN exacte m ≤ 3 vs force brute | ✅ |
| G2b | Moteur générique de classes m ≤ 13 | ✅ |
| G3 | Indexage bijectif 48 bits (format v1) | ✅ |
| G4 | Source gaussienne 2 bits/dim : **92,23 % de rétention** | ✅ |
| G5 | Spherical GPTQ + pipeline LLM | ❌ à faire |
| G6 | Noyau fusé (déquant + matvec) | ❌ à faire |

Résultat G4 mesuré (20 000 blocs, seed figée) :

| méthode | bits/dim | MSE | rétention |
|---|---|---|---|
| papier, spherical shaping | 2,000 | — | 89,14 % |
| papier, shape–gain | 2,000 | — | 92,11 % |
| **notre spherical shaping (β\* = 0,350)** | 1,9999 | 0,0775 | **92,23 %** |
| Shannon | 2,000 | 0,0625 | 100 % |

## 4. Dérivations à ne pas re-chercher

Ce sont les résultats non triviaux qui ont coûté du temps. Ils sont testés,
mais leur *raison* n'est pas évidente à la lecture du code seul.

**Construction de Λ₂₄** (Eq. 4–5 du papier), en coordonnées entières
`√8·Λ₂₄ ⊂ Z²⁴` :

| | coset pair | coset impair |
|---|---|---|
| parité | `xᵢ ≡ 0 (mod 2)` | `xᵢ ≡ 1 (mod 2)` |
| Golay | `{i : xᵢ ≡ 2 mod 4} ∈ G₂₄` | `{i : xᵢ ≡ 3 mod 4} ∈ G₂₄` |
| somme | `Σxᵢ ≡ 0 (mod 8)` | `Σxᵢ ≡ 4 (mod 8)` |

**Asymétrie encodeur/décodeur** — c'est ce qui rend le projet viable :
l'encodeur (plus proche voisin) tourne hors ligne une fois par modèle et peut
coûter des minutes ; le décodeur (index → vecteur) tourne à chaque GEMM et
n'est que du décalage/masquage. Ne jamais optimiser l'un en pensant à l'autre.

**Classes impaires : la condition de somme est au niveau classe.** Les signes
étant forcés par l'appartenance au codeword, leur contribution s'annule mod 2
et la condition mod 8 se réduit à `n₁ + n₇ + n₉ impair` (valeurs ≡ ±1 mod 8).
Conséquence : aucune contrainte résiduelle sur l'arrangement, donc le
maximiseur par classe est un **appariement trié** (exact par l'inégalité de
réarrangement), et les signes ne portent **aucun bit** dans l'index.

**Classes paires : la réparation de parité n'est pas un flip en place.**
Quand la parité des signes appariés diffère de celle requise, il faut
sacrifier une valeur — mais la placer au **plus petit |xᵢ| du support** et
*retasser* les autres, pas la flipper à son créneau glouton. Le flip en place
est sous-optimal dès que le gain de promotion dépasse la différence de
créneaux. Implémenté par scan de suffixes, validé contre une référence DP
exhaustive (`tests/g2b_generic.rs::even_repair_matches_dp_reference`).

**Le test qui verrouille tout** : la formule de cardinalité des classes doit
reproduire les coefficients thêta connus **et** la somme cumulée exacte
`N(13) = 280 974 212 784 720` (Table 1 du papier). C'est un verrou à 15
chiffres qu'aucune contrainte fausse ne peut franchir. Voir
`classes.rs::classes_reproduce_theta_series`.

**Format d'index v1 — contrat de stabilité.** Déterminé par : le générateur
Golay `0xC75` + l'ordre des codewords (weight-major, croissant dans un poids)
+ l'ordre d'énumération des classes + les ordres de composition mixed-radix.
Toute modification casse la compatibilité des fichiers quantifiés.

## 5. Leçon de méthode : les tests doivent être létaux

Un audit adversarial (mutation testing) a montré que la première suite G1
passait **entièrement** avec l'étage Golay du prédicat d'appartenance
supprimé — toutes les énumérations construisaient leurs mots à partir de vrais
codewords, donc le prédicat dégénérait en filtre de somme. Corrigé par
`golay_stage_is_load_bearing`, dont les sondes sont valides en parité *et* en
somme, et rejetées uniquement par Golay.

**Avant de déclarer un gate vert, muter le code et vérifier que la suite
échoue.** Un test qui passe sur du code cassé ne vaut rien.

## 6. Prochaines étapes, par ordre

### Phase 2c — performance de l'encodeur (préalable pratique)

Mesure actuelle : **~7 ms/bloc/cœur** pour le moteur générique m ≤ 13.
Sur 4 cœurs : 562 blocs/s. Ça donne, pour un modèle de N poids (N/24 blocs) :

| modèle | blocs | 4 cœurs | 12 cœurs (M-series) |
|---|---|---|---|
| Qwen3-0.6B (~0,5 Md) | 21 M | 10 h | 3,5 h |
| Qwen3-4B (~3,6 Md) | 150 M | 74 h | 25 h |
| Llama-3 8B (~7 Md) | 292 M | 144 h | 48 h |

C'est un coût **hors ligne, unique** — pas un coût d'inférence. Acceptable
pour le smoke test 0.6B, bloquant au-delà.

Goulot identifié, non mesuré au profileur : dans
`generic.rs::shell_bests`, la boucle des classes impaires fait un **tri de 24
éléments pour chacun des 4096 codewords**, à chaque bloc. Soit ~450 000
opérations par bloc rien que là. Pistes, par rapport gain/effort :
1. Trier les classes impaires par borne décroissante et sortir de la boucle
   des codewords dès que la borne passe sous le meilleur courant (l'élagage
   existe mais s'évalue *après* le tri).
2. Remplacer le tri complet par une sélection partielle : seules les
   premières positions comptent pour les classes à peu de valeurs lourdes.
3. Réutiliser le tri entre codewords proches (les codewords voisins en poids
   de Hamming partagent la majorité des `yᵢ`).
4. SIMD (`pulp`) en dernier.

### Phase 5 — Spherical GPTQ et premier LLM (le vrai jalon)

C'est ici qu'on sort du monde auto-vérifiable. Étapes :

1. Chargeur `safetensors` (première dépendance externe assumée).
2. Hessiennes par couche `H = AᵀA/N` sur corpus de calibration. Le papier
   utilise **6 100 séquences de DCLM-edu** (même taille que QuIP#).
3. **Algorithme 1 du papier** : blocs de b = 24 canaux d'entrée, gauche→droite,
   Cholesky de `H⁻¹`, lignes en parallèle, reset de gain
   `ṽ = ‖v‖₂ · Q_dir(v/‖v‖₂)`, propagation du résidu sur les colonnes non
   traitées. Le crate `faer` est le choix retenu pour l'algèbre (pur Rust,
   pas de dépendance Fortran/BLAS — build reproductible).
4. Progression **petit → gros** (consigne utilisateur) : Qwen3-0.6B en smoke
   test, puis **Qwen3-4B** qui est le plus petit modèle avec des chiffres de
   référence dans le papier (Table 6), puis 7B/8B.
5. Évaluation : perplexité WikiText-2 à 4096 de contexte, MMLU, CSR — **plus
   un benchmark métier d'extraction documentaire**. Cf.
   [arXiv:2607.08734](https://arxiv.org/abs/2607.08734) : perplexité et
   exactitude restent stables pendant que les réponses individuelles changent.

> **Gate G5 = point de sortie du projet.** Si LLVQ ne bat pas QuIP#/QTIP en
> perplexité sur Qwen3-4B, toute la thèse tombe et il faut le dire, pas
> optimiser un noyau pour une méthode qui ne tient pas ses promesses.

### Phase 6 — noyau fusé (déquant + matvec)

**C'est là qu'est la contribution d'ingénierie du projet.** Le papier dit
explicitement (Annexe C) : leur noyau CUDA ne traite qu'**une seule couche
(M = 3), « pour la simplicité »**, il est **plus lent que QTIP**, et les
auteurs déclarent que l'optimisation bas niveau est « largement orthogonale »
à leur contribution. Le noyau **multi-couches**, celui qu'exige le régime
2 bits/poids (m ≤ 13), **n'existe nulle part**.

Repères de la Table 7 (matvec 4096×4096) : FP16 = 16,13 µs ; leur LLVQ fusé =
11,194 µs, soit **1,36–1,48× le FP16**.

⚠️ Décision matérielle à trancher avec l'utilisateur avant d'écrire une ligne :
CUDA (cible serveur NVIDIA) ou Metal/`wgpu` (Mac de dev, portabilité AMD/Intel,
argument souveraineté) ? Le plan prévoyait CUDA via `cudarc`, mais si la
machine de développement est un MacBook, ça change tout.

## 7. Conventions

- `llvq-core` et `llvq-search` restent **sans dépendance** : le cœur
  mathématique doit rester auditable (contexte souveraineté).
- Zéro warning clippy.
- Les tests coûteux sont `#[cfg_attr(debug_assertions, ignore = "...")]` :
  rapides en debug, exhaustifs en release.
- Commentaires et docs en anglais dans le code, échanges en français.
