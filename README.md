# llvq — Leech Lattice Vector Quantization en Rust

> **Nouvelle session ?** Lire [`CLAUDE.md`](CLAUDE.md) — état, dérivations
> non triviales à ne pas re-chercher, pièges, et prochaines étapes.

Implémentation du papier **[LLVQ, arXiv:2603.11021](https://arxiv.org/abs/2603.11021)**
(van der Ouderaa, van Baalen, Whatmough, Nagel — Qualcomm AI Research, 2026) :
quantification vectorielle de poids de LLM sur le réseau de Leech Λ₂₄, état de
l'art à 2 bits/poids, sans codebook matérialisé.

Plan détaillé, gates de validation et provenance :
[`docs/llvq-rust-implementation-plan.md`](docs/llvq-rust-implementation-plan.md).

## État

| Phase | Contenu | Gate | Statut |
|---|---|---|---|
| 1 | `llvq-core` — Golay [24,12,8], Λ₂₄ (Eq. 4–5), couches | **G1** ✅ | fait |
| 2 | `llvq-search` — Adoul–Barth multi-couches (euclidien + angulaire) | **G2** ✅ | fait (m ≤ 3) |
| 2b | Moteur générique de classes (m ≤ 13, régime 2 bits/poids) | **G2b** ✅ | fait |
| 3 | Indexage bijectif hiérarchique (`index.rs`, format v1, 48 bits) | **G3** ✅ | fait |
| 4 | Validation source gaussienne (Table 3) | **G4** ✅ | **fait : 92,23 %** |
| 5 | Spherical GPTQ + pipeline LLM | G5 | à venir |
| 6 | Noyau CUDA fusé multi-couches | G6 | à venir |

Gate G1 (tout passe, `cargo test --release -- --include-ignored`, ~1,7 s) :
distribution des poids de Golay 1/759/2576/759/1, distance minimale 8,
auto-dualité, distinction des 4096 mots, **nombre de baisers 196 560** et
**|Shell(3)| = 16 773 120** reproduits par énumération exhaustive où chaque
vecteur compté est validé individuellement par le prédicat d'appartenance,
spot-checks Shell(4) (48 et 170 016), norme minimale 32, clôture additive.

La suite a été durcie par un audit adversarial multi-agents (mutation
testing) : le test `golay_stage_is_load_bearing` contient des sondes qui ne
sont rejetées **que** par l'étage Golay du prédicat — supprimer cet étage
fait échouer la suite (vérifié par mutation), ce qui n'était pas le cas de
la première version. Arithmétique exacte sur tout le domaine `[i32; 24]`
(norme en i128, add/neg vérifiés), `#![forbid(unsafe_code)]`.

Gate G2 (`llvq-search`) : recherche du plus proche voisin **exacte** sur
Shell(2), Shell(3) et leur union (métriques euclidienne *et* angulaire,
§3.1 du papier), validée contre la force brute (argmax sur les 196 560 puis
16,7 M points, énumérés par un troisième chemin de code lui-même épinglé à
la série thêta). Zéro énumération à la requête : maxima par classe en forme
close (réparation de parité par flip du min |xᵢ|), quantités par-codeword
en tables DP par chunks de 8 bits, élagage par borne supérieure. Débit
mesuré : **~7 300 requêtes/s/cœur** (au lieu de 507 en naïf).

⚠️ Ce chiffre ne vaut que pour le chemin rapide m ≤ 3. Le moteur générique
m ≤ 13, celui qu'exige le régime 2 bits/poids, tourne à **~140 blocs/s/cœur**
(voir `CLAUDE.md` §6 pour le goulot identifié et les pistes).

Gate G4 préliminaire (`llvq-bench`, `cargo run --release -p llvq-bench`) —
premier chiffre de qualité, sans LLM, protocole du §4 du papier sur source
gaussienne N(0,1), 20 000 blocs d'évaluation :

| méthode | bits/dim | MSE | rétention Shannon |
|---|---|---|---|
| Lloyd–Max scalaire 1 bit (analytique) | 1,0000 | 0,3634 | 73,0 % |
| **LLVQ spherical shaping (m ≤ 3)** | 1,0007 | 0,2865 | **90,1 %** |
| LLVQ shape–gain, gain 2 bits (m ≤ 3) | 1,0840 | 0,2733 | 86,3 % |
| Limite de Shannon | 1,0007 | 0,2498 | 100 % |

Gate G2b (moteur générique, `llvq-search/src/{classes,generic}.rs`) :
énumération programmatique des classes d'équivalence des couches 2..13 —
la formule de cardinalité reproduit les coefficients thêta connus **et la
somme cumulée exacte N(13) = 280 974 212 784 720** (Table 1 du papier), un
verrou à 15 chiffres. Deux résultats structurants dérivés et testés : la
condition de somme des classes impaires se réduit à « n₁+n₇+n₉ impair »
au niveau classe (signes forcés, maximiseur = appariement trié, exact par
réarrangement) ; la réparation de parité des classes paires est « sacrifier
une valeur au plus petit |x| du support et retasser », validée contre une
référence DP exhaustive, et le moteur coïncide avec le chemin rapide sur
les couches énumérables.

**Gate G4 complet — source gaussienne à 2 bits/dim (Table 3 du papier) :**

| méthode | bits/dim | MSE | rétention |
|---|---|---|---|
| papier, spherical shaping | 2,000 | — | 89,14 % |
| papier, shape–gain | 2,000 | — | 92,11 % |
| **LLVQ spherical shaping (β\* = 0,350)** | 1,9999 | 0,0775 | **92,23 %** |
| LLVQ shape–gain, gain 2 bits | 2,0832 | 0,0670 | 93,62 % |
| Shannon | 2,000 | 0,0625 | 100 % |

⚠️ La colonne MSE de la Table 3 telle que transcrite du PDF (0,1084/0,1078)
est incohérente avec sa propre colonne SQNR (1,798 ⇒ MSE ≈ 0,0845) — la
transcription des chiffres avait un encodage de police décalé. L'ancre
fiable est la **rétention** (89,14 % / 92,11 %), que notre implémentation
atteint et dépasse. La mesure ne peut pas surestimer la qualité : chaque
dot par-couche est atteint par un point matérialisé du codebook, vérifié
membre — un bug moteur ne pourrait que dégrader les chiffres. Le
dépassement s'explique vraisemblablement par le fit de l'échelle β.

Débit du moteur générique : ~560 blocs/s au total (~1,8 ms/bloc/cœur) —
suffisant pour G4 et Qwen3-0.6B, à optimiser (élagage, SIMD) avant les
modèles 4B+.

Gate G3 (`llvq-search/src/index.rs`) : bijection point ↔ entier ≤ N(13)
(48 bits par bloc de 24 poids, régime 2 bits/poids exact) sans codebook
matérialisé — hiérarchie couche → classe → (codeword, arrangement, signes)
linéarisée en mixed-radix, rangs de permutation de multiset en u128.
Vérifié : aller-retour exhaustif sur Shell(2) (196 560 points, zéro
collision), 2 M d'indices aléatoires à travers les 2⁴⁸ points
(decode → membre → re-encode identique), frontières de classes, gagnants
du moteur de recherche. Le format v1 (ordre des codewords, ordre
d'énumération des classes, ordre de composition) est documenté comme
contrat de stabilité.

## Stratégie de test LLM (phases 4+)

Du petit vers le gros, chaque étape ne servant qu'à dérisquer la suivante :

1. **Source gaussienne** — aucun modèle, cibles chiffrées de la Table 3.
2. **Qwen3-0.6B** — smoke test du pipeline (pas de chiffres de référence).
3. **Qwen3-4B** — premier modèle avec chiffres de référence dans le papier
   (Table 6) : c'est le juge de paix « petit modèle ».
4. **Llama-2 7B / Llama-3 8B** — comparaison finale aux tables du papier.

## Commandes

```bash
cargo test                                        # suite rapide (debug)
cargo test --release -- --include-ignored         # + Shell(3), ~25M vérifications
cargo clippy --all-targets
```

`llvq-core` n'a **aucune dépendance** : le cœur mathématique doit rester
auditable et reproductible (contexte souveraineté).
