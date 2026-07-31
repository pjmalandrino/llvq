# LLVQ — contexte projet (passation de session)

> Ce fichier est chargé automatiquement par Claude Code. Il contient tout ce
> qu'une nouvelle session doit savoir pour reprendre le travail sans relire
> l'historique.

> 🧭 **Reprise de session** : [`docs/passation-2026-07-31.md`](docs/passation-2026-07-31.md)
> — où on en est, quoi faire ensuite, et les pièges de mesure GPU chèrement
> acquis. Le modèle est publié et démarre seul ; le noyau est le chantier
> ouvert.

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

⚠️ **Piège de transcription — résolu le 2026-07-28.** L'extraction *texte*
du PDF est corrompue (police décalée de +1 par glyphe, et des chiffres qui se
dédoublent : `0,084` ressortait en `0,1084`). **Le rendu image, lui, est
parfait.** Recette :

```python
import fitz  # pymupdf
p = fitz.open("2603.11021v2.pdf")[6]           # page 7
clip = fitz.Rect(p.rect.x0+60, p.rect.y0+395, p.rect.x0+330, p.rect.y0+450)
p.get_pixmap(matrix=fitz.Matrix(9, 9), clip=clip).save("t.png")
```

Le papier a été relu intégralement par ce moyen : **tous** ses chiffres,
l'Algorithme 1, l'Algorithme 3 et les tables 3/6/7/8/9 sont transcrits dans
[`docs/llvq-paper-notes.md`](docs/llvq-paper-notes.md). Ne plus relire le PDF
sans raison, et **ne jamais** faire confiance à `pdftotext` dessus.

## 2. Architecture

```
llvq-core/     Golay [24,12,8] + Λ₂₄ + couches. ZÉRO dépendance, forbid(unsafe).
llvq-search/   Recherche NN exacte, classes, moteur générique m≤13, indexage, packing.
llvq-quant/    Spherical GPTQ : algèbre dense, boucle par blocs, quantifieurs.
llvq-artifact/ Le format .llvq : writer, reader, décodeur. ZÉRO dépendance.
llvq-metal/    Micro-bancs GPU (macOS) : plomberie Metal, coût du décodage.
llvq-llm/      Côté modèle : passe avant observable, corpus, perplexité. (candle)
llvq-bench/    Débit-distorsion, débit encodeur, coût du décodage.
```

**`llvq-core`, `llvq-search`, `llvq-quant`, `llvq-artifact` et `llvq-bench`
restent sans dépendance externe.** Lire un modèle quantifié ne doit pas exiger
un runtime de tenseurs : l'arbre complet de `llvq-artifact` fait 3 crates,
contre 690 pour `llvq-llm`. Seul `llvq-llm` en a — candle, tokenizers, hf-hub,
parquet — parce qu'il faut bien charger et exécuter un modèle. C'est aussi le
seul crate où `unsafe` est autorisé (mapper un safetensors de plusieurs Go
l'est par construction dans candle). Le plan prévoyait
`faer` pour l'algèbre de la Phase 5 ; on ne l'a pas pris. Ce dont
l'Algorithme 1 a besoin — Cholesky, inverse triangulaire, produit
triangulaire — tient en ~150 lignes verrouillées par des identités exactes
(`llvq-quant/src/linalg.rs`), et l'API de `faer` bouge beaucoup d'une version
à l'autre. Si le débit du Cholesky devient limitant sur des couches
2560×2560, `faer` se glisse derrière `GptqFactor::new` sans toucher un
appelant.

Commandes :
```bash
cargo test --release -- --include-ignored   # suite complète, ~45 s
cargo test                                   # suite rapide (les tests lourds sont ignored en debug)
cargo run --release -p llvq-bench --bin llvq-bench   # tableau qualité
cargo run --release -p llvq-bench --bin encbench      # débit encodeur, 1 cœur
cargo run --release -p llvq-bench --bin betasweep     # sensibilité de β (G4)
cargo run --release -p llvq-bench --bin decbench      # coût du décodage (G6)

# côté modèle (Metal recommandé : ~7× le CPU sur M3 Max)
cargo run --release -p llvq-llm --features metal --bin oracle
cargo run --release -p llvq-llm --features metal --bin ppl -- 4096 999 metal
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
| 2c | Encodeur : 639 µs/bloc/cœur (5,5× le départ) | ✅ |
| G5 | Spherical GPTQ + pipeline LLM | ✅ **Wiki 16,9617 à 2,1696 bits pesés** sur Qwen3-4B (QTIP : 17,04 à 2,000). Vert avec réserve : on passe de 0,08 point, à 8,5 % de bits en plus |
| G6 | Noyau fusé (déquant + matvec) | 🟡 **verrou levé** — décodage à masques mesuré sur GPU à **0,11 ns/bloc** (1,43× le coût de ne rien décoder). Le noyau lui-même reste à écrire. Voir [`docs/format-noyau.md`](docs/format-noyau.md) et [`docs/passation-2026-07-31.md`](docs/passation-2026-07-31.md) |

Résultat G4 mesuré (20 000 blocs, seed figée), face aux chiffres du papier
relus sur le PDF (Table 8, annexe H — celle qui nomme le codebook) :

| méthode | codebook | bits/dim | MSE | rétention |
|---|---|---|---|---|
| papier, spherical shaping | `Λ₂₄(13)` | 2,000 | 0,084 | 89,37 % |
| papier, shape–gain 0 bit de gain | `norm(Λ₂₄(13))` | 2,000 | 0,085 | 89,12 % |
| papier, shape–gain 1 bit de gain | `norm(Λ₂₄(12))` | 2,000 | 0,078 | 92,14 % |
| **notre shape–gain 0 bit de gain** | `norm(Λ₂₄(13))` | 1,9999 | 0,0840 | **89,36 %** |
| **notre spherical shaping (β\* = 0,350)** | `Λ₂₄(13)` | 1,9999 | 0,0775 | **92,23 %** |
| Shannon | — | 2,000 | 0,0625 | 100 % |

> 🔎 **L'écart sur le spherical shaping s'explique par β, pas par le
> codebook — ne pas le revendiquer comme une victoire.** Notre shape–gain
> 0 bit reproduit le papier au millième (89,36 vs 89,12), donc protocole et
> codebook sont bons. Mais notre spherical shaping le dépasse de presque
> 3 points (92,23 vs 89,37). Le balayage
> (`cargo run --release -p llvq-bench --bin betasweep`) montre un optimum
> **étroit** :
>
> | β | 0,300 | 0,325 | **0,350** | 0,375 | 0,400 |
> |---|---|---|---|---|---|
> | Ret (%) | 87,00 | 91,09 | **92,24** | 90,95 | 88,07 |
>
> Les 89,37 % du papier correspondent à un β désaccordé d'environ ±0,04
> (≈ 11 %). Nous ajustons β\* sur un jeu d'entraînement séparé ; leur rayon
> de boule ne semble pas optimisé. **Conclusion : à iso-réglage, on reproduit
> le papier ; notre avance est un artefact de réglage, pas un meilleur code.**

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

**Classes paires : la réparation de parité est un seul flip.** Quand la
parité des signes appariés diffère de celle requise, il faut sacrifier une
valeur. Sacrifier la valeur de type `u` à son dernier créneau `j` puis
retasser les suivantes vaut `base − u·A_j + D_j − u·A_{w−1}` avec
`D_j = Σ_{i>j} Vᵢ(A_{i−1} − Aᵢ)` ; en développant la somme télescopique,

```
score(j) − score(w−1) = Σ_{t=j}^{w−2} (V_{t+1} − V_t)·A_t + (V_{w−1} − V_j)·A_{w−1}
```

et **chaque terme est ≤ 0** (`V` décroissant, `A ≥ 0`). Donc aucun retassage
ne bat le flip simple de la **plus petite valeur de mot**, que l'appariement
trié a déjà posée sur le plus petit `|xᵢ|` du support.

> ⚠️ Correction (2026-07-28) d'une note antérieure de ce fichier, qui
> affirmait le contraire (« le flip en place est sous-optimal dès que le gain
> de promotion dépasse la différence de créneaux »). La formule générale
> multi-types était *correcte* mais son maximum est toujours atteint en
> `j = w−1` : le balayage de suffixes était du code mort. Découvert par
> mutation testing — neutraliser l'accumulateur `D` ne faisait échouer aucun
> test, y compris la référence DP. Vérifié algébriquement, numériquement
> (10⁶ instances aléatoires + à égalités) et par la référence naïve, qui
> essaie encore *tous* les types à *tous* les créneaux.

Validé par `tests/g2b_generic.rs::even_repair_matches_dp_reference` (DP
exhaustive) et par `generic_ref` de bout en bout.

**Appariement trié ⇒ somme télescopique par runs.** Tous les maximiseurs de
classe sont de la forme `Σᵢ Vᵢ·Aᵢ` avec `V` (valeurs de la classe) constant
par morceaux — au plus 3 types de mot, 2 types libres, 5 types impairs — et
`A` (la requête) trié. Avec `P` les sommes préfixes de `A` :

```
Σᵢ Vᵢ·Aᵢ = Σ_runs (v_r − v_{r+1}) · P[fin_r]        (v_{dernier+1} := 0)
```

Une multiplication-addition **par run**, plus aucun travail par coordonnée.
C'est ce qui a fait passer l'évaluation d'une classe de 24 opérations à ≤ 5.

**Asymétrie des deux API de recherche.** `shell_bests` répond à douze
questions indépendantes (le maximum sur chaque coquille) et porte donc douze
seuils d'élagage indépendants : c'est ce dont a besoin le balayage de β en
G4, qui reclasse une même recherche à plusieurs échelles. Un quantifieur n'en
pose qu'une seule : `nearest_scaled(β)` classe toutes les classes par
l'objectif unique `key = 2β⟨x,v⟩ − β²‖v‖²`, ce qui permet un seuil global,
donc un élagage *entre* coquilles et l'abandon de sections entières (les 4096
codewords impairs d'un coup). ≈ 1,6× plus rapide que le passage par
coquilles, et c'est le chemin que la Phase 5 appellera.

**Le moteur générique n'a pas besoin de `Workspace`.** Il ne consommait des
tables DP par sous-ensembles qu'une seule quantité, la parité des signes sur
le support — soit `popcount(c & neg_mask) mod 2`, deux instructions. Les
tables restent utilisées par le chemin rapide m ≤ 3 de `lib.rs`.

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

La Phase 2c a produit une deuxième prise : après réécriture du moteur, deux
mutants survivaient. L'un (« prune trop agressif de 1e-6 ») a été tué en
portant `g2c_reference` de 8 à 40 requêtes. L'autre (« accumulateur de
report de réparation mis à zéro ») **ne pouvait pas** être tué — parce que le
code muté était mathématiquement équivalent au code d'origine. C'est comme ça
qu'on a trouvé que le balayage multi-types était du code mort (cf. §4).

> Un mutant qui survit dit soit « le test est faible », soit « le code est
> mort ». Les deux méritent qu'on s'arrête.

La Phase 5 a produit la troisième, et la plus coûteuse : `group_scales`
activé détruisait le modèle — **perplexité 1 327 613 contre 19,5 de
baseline**. Cause : dans `refine_group_scales`, la crête `λ` était ajoutée
**en absolu** sur la diagonale de `M`, alors que l'amortissement de la
hessienne, lui, est relatif à `mean(diag H)`. Or `M[p,q] = g_p H g_q` varie
comme le **carré** de l'amplitude des poids ; sur des poids réels (~1e-2),
`λ = 1e-2` écrase `M`, le système dégénère en `s ≈ r/λ` et tous les blocs
sont ramenés vers zéro.

Le test existant ne pouvait pas l'attraper : **il tournait à `λ = 0`**, donc
il n'exerçait jamais la crête. Corrigé par
`group_scale_ridge_is_scale_invariant`, qui teste la propriété qui doit tenir
plutôt qu'une valeur : multiplier les poids par une constante ne doit rien
changer à ce que le raffinement décide, **pour tout λ**.

> **Le motif commun aux trois prises** : une assertion qui n'exerce pas le
> paramètre qu'elle est censée couvrir. Golay neutralisé et jamais vu ; une
> monotonie non stricte qui accepte un no-op ; un λ à zéro qui ne teste
> aucune crête. Quand un paramètre a une valeur « neutre », un test qui ne
> l'utilise qu'à cette valeur ne teste rien.

**Avant de déclarer un gate vert, muter le code et vérifier que la suite
échoue.** Un test qui passe sur du code cassé ne vaut rien. Le moteur
générique est actuellement tenu par 8 mutants tués (coefficient télescopique,
côtés de la partition branchless, comparaison de fusion, seuil d'élagage,
amplitude du flip, créneau du flip, matérialisation, abandon de section).

## 6. Prochaines étapes, par ordre

### Phase 2c — performance de l'encodeur ✅ (fait le 2026-07-28)

Mesuré sur Apple M3 Max (12 cœurs perf), un seul cœur, via
`cargo run --release -p llvq-bench --bin encbench` :

| chemin | avant | après | gain |
|---|---|---|---|
| `shell_bests` (12 coquilles, échelles mixtes) | 3 532 µs/bloc | 932 µs/bloc | 3,8× |
| `nearest_scaled(β)` — encodeur euclidien, N(0,1) | — | 656 µs/bloc | 5,4× vs départ |
| `nearest_angular()` — **`Q_dir`, le chemin de la Phase 5** | — | 680 µs/bloc | 5,2× vs départ |

Soit **1 469 blocs/s/cœur** pour `nearest_angular`, celui que la Phase 5
appellera. Projection sur un modèle de N poids (N/24 blocs), en coût **hors
ligne unique** :

| modèle | blocs | 4 cœurs | 12 cœurs |
|---|---|---|---|
| Qwen3-0.6B (~0,5 Md) | 21 M | 1,0 h | **20 min** |
| Qwen3-4B (~3,6 Md) | 150 M | 7,1 h | **2,4 h** |
| Llama-3 8B (~7 Md) | 292 M | 13,8 h | **4,6 h** |

Ce n'est plus bloquant pour la Phase 5, y compris sur 8B.

Ce qui a payé, dans l'ordre (détails et démonstrations en §4) :

1. **Sommes télescopiques par runs** — évaluation d'une classe de 24
   opérations à ≤ 5. Le plus gros gain unitaire.
2. **Tables plates par coquille**, physiquement retriées par borne
   décroissante à chaque requête : la boucle de classes devient un balayage
   contigu qui `break`, sans indirection (avant : `Vec<usize>` → `Vec<Rt>` →
   `Vec<Run>`, trois sauts de pointeur par classe).
3. **Suppression des tris par codeword** : la partition support /
   hors-support tombe d'un seul passage sur l'ordre global de `|x|`, et
   l'ordre de `y` est une **fusion** à deux pointeurs (24 comparaisons) au
   lieu d'un tri de 24 éléments.
4. **Partitions branchless** : le bit d'appartenance est un pile-ou-face sur
   24 coordonnées × 8 191 codewords ; le mauvais store redondant coûte bien
   moins cher que la moitié des branches mal prédites. ≈ 1,4×.
5. **`nearest_scaled` à objectif global** (cf. §4) : ≈ 1,6× de plus que
   `shell_bests`.
6. **Réparation de parité réduite à un flip** (cf. §4) : le balayage
   multi-types était du code mort.

Ce qui a été **mesuré et écarté** — à ne pas retenter sans idée neuve :

- *Pré-amorçage de l'incumbent.* Testé avec un oracle (le vrai optimum injecté
  comme seuil de départ) : plafond de 1,37× côté pair et **1,07× côté
  impair**. La borne support-aveugle est trop lâche pour que l'ordre de
  découverte compte. Toute amélioration passe par une **borne plus fine**,
  pas par un meilleur ordre.
- *Élagage au niveau codeword.* Après l'étape 2, on n'évalue déjà que ~3,5
  classes par (codeword, coquille) : un test de borne par codeword coûterait
  presque ce qu'il économise.

Pistes restantes, si jamais il faut encore du débit : casser les chaînes de
dépendance des sommes préfixes (elles sont sérielles, ~3 cycles/élément),
réutiliser la partition d'un octade pour son complément de poids 16 (les
codewords de Golay vont par paires complémentaires, ≈ 50 % des partitions
paires en moins), et SIMD (`pulp`) en dernier. Aucune n'est nécessaire pour
la Phase 5.

⚠️ **Le profileur n'a jamais été utilisé** : tout ci-dessus vient de compteurs
instrumentés et de mesures avant/après. Si on repart là-dessus, commencer par
un vrai profil.

### Phase 5 — Spherical GPTQ et premier LLM (le vrai jalon)

C'est ici qu'on sort du monde auto-vérifiable. **Le papier a été relu
intégralement le 2026-07-28** ; tout ce qui suit est transcrit dans
[`docs/llvq-paper-notes.md`](docs/llvq-paper-notes.md) — Algorithme 1,
Algorithme 3, échelles closes, tables de référence. Ne pas rouvrir le PDF
sans raison.

> ⚠️ **La cible a changé.** Le plan initial visait le *spherical shaping*.
> La Table 6 du papier dit qu'il **perd contre QTIP** sur Qwen3-4B sans
> fine-tuning (21,80 vs 17,04 de perplexité Wikitext-2). Ce qui gagne, c'est
> le **shape–gain** : 15,54 avec 2 bits de gain, 17,05 avec 0 bit. Et
> l'annexe I recommande **0 bit de gain sous Spherical GPTQ** — avec un code
> à faible distorsion angulaire, la capacité est mieux dépensée entièrement
> en directions, les magnitudes étant tenues par la contrainte de norme
> pendant GPTQ puis par une résolution close en fin de couche.
>
> Conséquence concrète : le quantifieur appelé en boucle est `Q_dir`, la
> **recherche angulaire** — d'où `BallSearcher::nearest_angular`, ajouté le
> 2026-07-28 (680 µs/bloc/cœur, cf. §6 Phase 2c).

**État au 2026-07-28 : chaîne complète en place, smoke test Qwen3-0.6B en
cours.**

### Références mesurées

| | valeur |
|---|---|
| Qwen3-0.6B FP32, wikitext-2 test, ctx 4096, 73 fenêtres | **ppl = 19,1481** |
| Passe avant maison vs `candle_transformers::qwen3` | **max \|Δhidden\| = 0** |
| Metal vs CPU (M3 Max) sur la perplexité | **~7×** (2,5 s vs 17 s/fenêtre) |

L'écart nul avec candle n'est pas « proche », c'est exact. C'était le risque
principal : une passe avant écrite à la main qui rate le RMSNorm **par tête**
sur q et k (spécifique à Qwen3) ou une convention RoPE produirait des
hessiennes silencieusement fausses, et l'erreur ne se verrait que bien plus
tard sous forme d'une perplexité inexpliquée. `bin/oracle` verrouille ça.

### Décisions prises

- **Queues de bloc → `TailPolicy::KeepExact`.** Les colonnes non alignées sur
  24 restent en pleine précision. Attention au contresens : elles **reçoivent
  quand même** la rétroaction d'erreur des blocs précédents, et c'est
  souhaitable — n'étant pas quantifiées, elles absorbent cette compensation
  exactement. Elles ne font qu'arrêter la boucle, sans jamais produire
  d'erreur propre. Coût : ~0,22 bit/poids sur une couche en 1024, nul en 3072.
- **4 hessiennes par bloc, pas 7.** `q/k/v` consomment le même tenseur,
  `gate/up` aussi. Une factorisation par *activation*, réutilisée par les
  matrices qui la partagent.
- **Quantification parallélisée par ligne.** Toutes les étapes de GPTQ sont
  par ligne — quantification, résidu, solve triangulaire, propagation,
  échelles de groupe. Le découpage est donc **exact**, pas approché, et
  `parallel_matches_serial_exactly` l'exige au bit près. Sans ça le 0.6B
  demandait 3,4 h d'encodage mono-thread.
- **Calibration séquentielle** : le bloc *t* est quantifié contre les
  activations qui l'atteignent réellement, c'est-à-dire après traversée des
  blocs 0..*t*−1 **déjà quantifiés**. D'où deux passes avant par bloc.

### Résultats quantifiés — Qwen3-0.6B (smoke, 2026-07-28)

Protocole : shape–gain 0 bit de gain, rétraction sphérique, `TailPolicy::KeepExact`,
amortissement 1e-2, **sans rotation Hadamard**. Évaluation wikitext-2 test,
ctx 2048, 12 fenêtres. Baseline FP32 = **19,5038**.

| calibration | éch. groupe | rotation | blocs | ppl | dégradation |
|---|---|---|---|---|---|
| — (identité, contrôle) | — | off | 3 | **19,5038** | **×1,000 exact** |
| — (identité, contrôle) | — | **on** | 3 | **19,5038** | **×1,000 exact** |
| 33 k tokens | off | off | 3 | 21,2412 | ×1,089 |
| 33 k tokens | on | off | 3 | 21,1677 | ×1,085 |
| 33 k tokens | off | **on** | 3 | 20,3012 | ×1,041 |
| 33 k tokens | off | off | 28 | 45,9787 | ×2,357 |
| 131 k tokens | off | off | 28 | 44,6644 | ×2,290 |
| 131 k tokens | on | off | 28 | 53,6043 | ×2,748 |
| **131 k tokens** | **off** | **on** | **28** | **35,3252** | **×1,811** ← meilleur |

Coût : ~2 500 s pour 28 blocs sur M3 Max (Cholesky dominant, encodage Leech
parallélisé sur 12 cœurs).

**Ce que ces chiffres établissent :**

1. **La chaîne est saine.** Le contrôle identité rend la baseline au chiffre
   près — hessiennes, boucle GPTQ, réécriture des poids, conversions, passe
   séquentielle : tout est correct. C'est *le* test à relancer en premier si
   un résultat futur paraît absurde.
2. **Plus de calibration aide**, comme attendu (45,98 → 44,66).
3. **`group_scales` aide localement et nuit globalement** : gain sur 3 blocs
   (21,24 → 21,17), perte franche sur 28 (44,66 → 53,60). Le raffinement
   optimise le proxy **local** de chaque couche ; 28 optima locaux décalés se
   composent. C'est cohérent avec la thèse du papier — la dérive radiale est
   le mode de défaillance dominant, et ce raffinement réintroduit
   délibérément des variations de magnitude que la rétraction sphérique
   venait d'éliminer. **Le laisser désactivé** jusqu'à comprendre.
   - 🔎 Piste de formulation : l'Algorithme 3 écrit `s = (M + λI)⁻¹ r`, donc
     la crête tire les échelles vers **zéro**. Pour un facteur censé valoir
     ≈1, le prior naturel serait `(M + λI)s = r + λ·1`. Implémenté à la
     lettre du papier ; à tester si on veut récupérer le gain local.

4. **La rotation d'incohérence est le plus gros levier mesuré** : ×2,290 →
   ×1,811 d'un seul coup, soit 21 % de perplexité. C'était bien l'écart de
   configuration qui manquait, pas un défaut de la chaîne.

**On est dans le régime du papier.** Repère : leur Llama-3.2 1B sans
fine-tuning est à **×1,76** (Table 10) ; on est à **×1,811** sur un modèle
**40 % plus petit**, avec **beaucoup moins** de calibration, et avec la
rotation d'entrée seule là où ils utilisent « Input + Output ».

⚠️ Ce n'est **pas** un chiffre comparable — modèle, famille, corpus de
calibration et contexte diffèrent tous. Ce qu'on peut dire, c'est que la
méthode se comporte comme annoncé, ce qui n'était pas établi avant.

⚠️ **Calibration sur wikitext-2 *train*, pas DCLM-edu**, et sur bien moins de
tokens que le papier. Calibrer et évaluer dans le même domaine flatte le
résultat : validation de chaîne, **pas** un chiffre comparable à la Table 6.

### Leçon de méthode (expérimentale, cette fois)

Entre le run 1 et le run 3, j'ai changé **deux variables à la fois**
(échelles de groupe *et* volume de calibration) : 45 minutes de machine pour
un résultat ininterprétable, et une fausse piste sur « plus de calibration
dégrade ». Les A/B se font désormais **sur 3 blocs** — 8 minutes au lieu de
45 — avant tout run complet, et sur une seule variable.

### Qwen3-4B — le gate G5 (2026-07-29/30)

> 🚨 **Les chiffres de cette section sont périmés (2026-07-31).** La rétraction
> sphérique annulait le code de gain : la magnitude stockée était un flottant
> libre par bloc, 16 bits que la comptabilité ne facturait pas. Les 2,1117
> bits/poids valaient en réalité **2,7338**, et les 14,9104 décrivaient donc un
> modèle à 2,73 bits, pas à 2,11.
>
> **Le chiffre honnête, mesuré sur un fichier de 981 Mo et vérifié bit pour bit :
> 16,9617 de perplexité à 2,1696 bits/poids** (×1,386). Juste sous QTIP (17,04),
> à 8,5 % de bits en plus, et 9 % au-dessus de la meilleure config du papier.
>
> Diagnostic complet, les trois défauts et ce qui reste à décider :
> [`docs/retraction-et-gain.md`](docs/retraction-et-gain.md).

Protocole : shape–gain, rétraction sphérique, rotation d'entrée, `faer`,
131 k tokens de calibration, `TailPolicy::KeepExact`. Évaluation wikitext-2
test, ctx 4096, 12 fenêtres. Baseline FP32 **12,2336** (papier : 12,41 — 1,4 %
d'écart, expliqué par 12 fenêtres contre 73 ; **le harnais est validé**).

| calibration | bits/poids | wiki | × | C4 | × |
|---|---|---|---|---|---|
| wikitext, magnitude libre (⚠️ pas 2 bits) | 2,7289 | 14,2684 | ×1,166 | 26,0866 | ×1,296 |
| wikitext, 1 bit de gain | 2,1117 | 15,2909 | ×1,250 | 28,2024 | ×1,401 |
| **C4 (protocole du papier), 1 bit de gain** | **2,1117** | **14,9104** | **×1,219** | — | — |

3,45 h pour 252 matrices et 3,63 Md de poids (contre 6,3 h avant `faer`).

**Face au papier**, à 2 bits, sans fine-tuning : Quip# 21,15 · **QTIP 17,04**
(le seuil) · LLVQ 0 bit 17,05 · LLVQ 2 bits de gain 15,54 (leur meilleur).

**Trois choses que ces chiffres établissent, et une qu'ils n'établissent pas.**

1. **Quantifier le gain ne coûte presque rien.** A/B sur 3 blocs, config
   identique : magnitude libre 20,3012 à 2,73 bits, 1 bit de gain 20,3102 à
   2,21 bits. **0,04 % de perplexité pour 0,52 bit/poids.** Les 0,73 bit
   dépensés en magnitude libre ne servaient à rien — exactement ce
   qu'annonce la Table 8.
2. **Il n'y a pas de pénalité de domaine — j'avais mal raisonné.** Le modèle
   calibré sur wikitext donne ×1,250 sur wikitext et ×1,401 sur C4, et j'en
   avais conclu que calibrer en domaine flattait de ~12 %. **Faux** : cet
   écart mesure la *difficulté du corpus C4*, pas un avantage de calibration.
   Le bon test change le corpus de **calibration** en gardant l'évaluation
   fixe — et il dit l'inverse : calibrer sur C4 donne **14,91 contre 15,29**,
   donc *mieux*, C4 étant plus divers que wikitext-2 train.
   > ⚠️ **Ne pas confondre « corpus d'évaluation plus difficile » et « biais de
   > domaine ».** Seul le second est un biais, et il ne se mesure qu'en
   > faisant varier la calibration à évaluation constante.
3. **Le contrôle identité rend ×1,000 exact** avec et sans rotation, et
   reporte 16,01 bits/poids — la chaîne *et* la comptabilité sont saines.

**Ce qui est défendable** : avec le protocole de calibration du papier
(hors domaine), on atterrit à **14,9104 (×1,219)**, au niveau de leur
meilleure configuration sans fine-tuning (15,54, ×1,252), et nettement sous
QTIP et sous leur LLVQ 0 bit (tous deux ×1,37).

**Ce qui ne l'est pas** : « on les bat ». Il reste **2,1117 bits/poids contre
2,000**, soit 5,6 % de bits en plus. Dont ~0,1 bit vient de la politique de
queue — que le papier **ne spécifie jamais**, donc une partie de l'écart est
un détail non renseigné chez eux plutôt qu'un avantage chez nous.

Deux différences jouent en revanche **contre** nous : ~131 k tokens de
calibration contre leurs 6 100 séquences (~100× moins), et la rotation
d'entrée seule là où ils utilisent « Input + Output ».

**Pour un chiffre à débit strictement égal** : restreindre la recherche
angulaire à `Λ₂₄(12)` fait tomber l'index à 47 bits, plus 1 bit de gain =
48 bits par bloc — littéralement la meilleure ligne de leur Table 8
(`norm(Λ₂₄(12))` + 1 bit de gain). Petit changement dans le quantifieur,
run de 3,5 h.

> 🚨 **L'erreur de comptabilité, et ce qu'elle apprend.** Le premier run 4B a
> été annoncé à 2,0653 bits/poids. Faux : `LeechDirection` stocke la
> direction (48 bits) **et la norme du bloc en pleine précision**, soit
> 2,7289 bits/poids réels. « Zéro bit de gain » veut dire **une constante**
> pour tout le tenseur, pas une norme flottante libre par bloc.
>
> Aucun test ne pouvait l'attraper : ils portaient tous sur la **qualité** de
> la reconstruction, jamais sur son **coût**. C'est le même motif que les
> trois défauts précédents — une propriété qu'on croit tenue parce qu'on ne
> l'a jamais énoncée. `the_gain_is_actually_quantized` l'énonce maintenant :
> la magnitude reconstruite doit être **l'un des niveaux finis du code**.
>
> Découvert en préparant l'artefact. Tant qu'on simule, on peut se raconter
> ce qu'on veut sur ce que ça coûterait ; dès qu'il faut écrire des octets,
> il faut les compter.

### Compression réelle sur le 4B (1 bit de gain)

| | |
|---|---|
| linéaires quantifiés | 3 633,3 M @ **2,1117** bits |
| embedding | 389,0 M @ 16 bits (**9,7 %** du modèle) |
| **artefact** | **1,74 Go** contre 8,04 Go en FP16 → **×4,63** |

Extrapolation 70B (embedding ~1,5 %) : **×6,9** — 140 Go → ~20 Go, qui tient
dans la mémoire unifiée d'un Mac. C'est la thèse du projet.

⚠️ **1,74 Go est un chiffre calculé, pas un fichier.** Ce qu'on écrit fait
6,8 Go : des reconstructions en f16. Produire le vrai fichier demande de
brancher l'indexeur 48 bits (G3, écrit et testé) et un décodeur.

### ⚠️ Le piège du « x bits/poids » sur un petit modèle

2,1531 bits/poids ne concerne que les **196 matrices linéaires**. Sur
Qwen3-0.6B, la matrice d'embedding (liée au `lm_head`) pèse 155,6 M poids,
soit **26 % du modèle**, et reste en pleine précision :

| | |
|---|---|
| linéaires quantifiés | 440,4 M poids @ 2,1531 bits |
| embedding liée | 155,6 M poids @ 16 bits |
| **artefact** | **430 Mo** contre 1 192 Mo en FP16 → **×2,77** |

Donc ×2,77 de compression réelle, pas ×7,4. Ce n'est pas un problème de la
méthode : sur un 70B l'embedding pèse ~1 % et le ratio tend vers le nominal.
Mais **il ne faut jamais annoncer le bits/poids des linéaires comme le taux
de compression du modèle** — sur les petits modèles c'est faux d'un facteur
2,7.

---

**Le noyau de quantification lui-même est écrit et verrouillé.**

`llvq-quant` implémente l'Algorithme 1, l'Algorithme 3 et les échelles closes
de l'annexe F. 12 tests, **8 mutants tués** (signe de propagation, signe du
résidu, solve triangulaire court-circuité, rétraction neutralisée, `U` non
transposée, amortissement absolu au lieu de relatif, second membre des
échelles de groupe, terme de mise à jour du Cholesky). Le test porteur est
`correction_is_the_analytic_minimizer` : il reconstruit
`−ΔW_Q H_QR H_RR⁻¹` avec un Gauss–Jordan indépendant écrit dans le fichier de
test, et exige que le chemin par facteur de Cholesky produise exactement ça.

Tout est vérifiable **sans modèle** : la boucle GPTQ est testée contre des
quantifieurs qui n'ont rien à voir avec Λ₂₄ (identité, arrondi scalaire), ce
qui découple la validation de la rétroaction d'erreur de celle du codebook.

> 🕳️ **Le mutant qui a failli passer.** « Second membre des échelles de
> groupe construit sur `W` quantifié au lieu de `W` d'origine » survivait :
> le mutant rend le raffinement *inopérant* (`s = 1` exactement), ce que
> l'assertion « ne doit pas augmenter la perte » acceptait. Corrigé en
> exigeant une décroissance **stricte**. Même leçon qu'en §5 : une assertion
> de monotonie non stricte ne teste rien contre un no-op.

**Deux points ouverts, délibérément non tranchés dans le code :**

1. **Blocs de queue.** Un bloc fait 24 canaux d'entrée, mais les couches
   réelles ne sont pas des multiples de 24 (Qwen3-4B : 2560 = 24·106 + 16).
   Le papier dit seulement « le dernier peut être plus court » et ne dit
   **jamais** ce qui le quantifie. `quantize_layer` lève une assertion
   explicite sur une queue incompatible, plutôt que de dégrader une couche en
   silence. À trancher : padding, quantifieur scalaire de secours, ou choix
   des dimensions de blocs par couche.
2. **D'où vient `H`.** Calculer `H = AᵀA/N` exige une passe avant sur le
   corpus de calibration, donc un runtime de modèle. Deux voies : une passe
   avant en Rust (`candle`, grosse dépendance), ou un script externe qui
   décharge les hessiennes en `safetensors` et laisse la quantification à
   Rust. La contribution du projet est le quantifieur, pas le harnais de
   calibration — mais c'est un arbitrage à faire explicitement, et il engage
   l'argument « build reproductible ».

Étapes restantes :

1. Chargeur `safetensors` (première dépendance externe assumée).
2. Hessiennes par couche `H = AᵀA/N` sur corpus de calibration. Le papier
   utilise **6 100 séquences de DCLM-edu** (même taille que QuIP#).
3. **Algorithme 1** (annexe B du papier) : blocs de b = 24 canaux d'entrée,
   gauche→droite, Cholesky de `H⁻¹`, lignes en parallèle, reset de gain
   `ṽ = ‖v‖₂ · Q_dir(v/‖v‖₂)`, propagation du résidu sur les colonnes non
   traitées. Puis l'**Algorithme 3** (annexe I.1) pour la variante sphérique
   avec raffinement final des échelles par ligne dans la métrique hessienne.
   Le crate `faer` est le choix retenu pour l'algèbre (pur Rust, pas de
   dépendance Fortran/BLAS — build reproductible).
   - ⚠️ Deux ambiguïtés de notation de l'Algorithme 3 sont relevées et
     tranchées dans les notes de lecture (rétraction **par ligne**, résidu
     formé sur le `W̃` **compensé**). Les lire avant d'implémenter.
4. Progression **petit → gros** (consigne utilisateur) : Qwen3-0.6B en smoke
   test, puis **Qwen3-4B** qui est le plus petit modèle avec des chiffres de
   référence dans le papier (Table 6), puis 7B/8B.
5. Évaluation : perplexité WikiText-2 à 4096 de contexte, MMLU, CSR — **plus
   un benchmark métier d'extraction documentaire**. Cf.
   [arXiv:2607.08734](https://arxiv.org/abs/2607.08734) : perplexité et
   exactitude restent stables pendant que les réponses individuelles changent.

**Cibles chiffrées sur Qwen3-4B, 2 bits, sans fine-tuning** (Table 6) :

| | Wiki ↓ | MMLU ↑ | CSR ↑ |
|---|---|---|---|
| Baseline FP16 | 12,41 | 70,2 | 71,2 |
| QTIP (3INST) — **le concurrent à battre** | 17,04 | 57,4 | 63,5 |
| Quip#/E8P12 | 21,15 | 48,6 | 57,2 |
| LLVQ shape–gain 2 bits de gain | 15,54 | 59,3 | 64,1 |
| LLVQ shape–gain 0 bit de gain | 17,05 | 60,7 | 63,6 |

Ordre de bataille suggéré : d'abord reproduire **LLVQ shape–gain 0 bit +
Spherical GPTQ sans rotation Hadamard**. C'est le résultat le plus
spectaculaire du papier (Table 9, Llama-2 7B : GPTQ euclidien sans rotation
s'effondre à 91,90 de perplexité, le Spherical GPTQ tient à 6,90) et c'est
aussi le moins coûteux à implémenter — pas de transformée Hadamard en ligne,
pas de quantifieur de gain.

> **Gate G5 = point de sortie du projet.** Si LLVQ ne bat pas QuIP#/QTIP en
> perplexité sur Qwen3-4B, toute la thèse tombe et il faut le dire, pas
> optimiser un noyau pour une méthode qui ne tient pas ses promesses.
> Le critère est maintenant précis : **Wiki < 17,04** sur Qwen3-4B, 2 bits,
> sans fine-tuning.

### Phase 6 — noyau fusé (déquant + matvec)

**C'est là qu'est la contribution d'ingénierie du projet.** Le papier dit
explicitement (Annexe C) : leur noyau CUDA ne traite qu'**une seule couche
(M = 3), « pour la simplicité »**, il est **plus lent que QTIP**, et les
auteurs déclarent que l'optimisation bas niveau est « largement orthogonale »
à leur contribution. Le noyau **multi-couches**, celui qu'exige le régime
2 bits/poids (m ≤ 13), **n'existe nulle part**.

Repères de la Table 7, relus sur le PDF : FP16 matvec (4096×4096) =
**16,3 µs** ; FP16 matvec (4096×4104) = **17,69 µs** ; leur LLVQ fusé
(4096×4104) = **11,94 µs**, soit **1,36×/1,48× le FP16**. (Les valeurs
« 16,13 » et « 11,194 » d'une version antérieure de ce fichier venaient de
l'extraction texte corrompue.)

Argument matériel à garder de l'annexe G : une **coquille unique** implique
une norme constante, donc un facteur d'échelle fixe entre produits scalaires,
ce qui supprime le rééchelonnage des accumulations intermédiaires. L'union de
coquilles gagne en distorsion angulaire mais coûte ce rééchelonnage.

#### 🔎 Piste à évaluer avant d'écrire le noyau : une seule coquille

Le papier adopte l'**union** de coquilles, pour une uniformité angulaire par
bit « légèrement meilleure » (annexe G, ses mots), tout en notant lui-même le
contre-argument matériel : une coquille unique a une **norme constante**, donc
un facteur d'échelle fixe entre produits scalaires, donc pas de rééchelonnage
des accumulations intermédiaires dans un noyau fusé.

Mesuré chez nous (20 000 blocs, seed figée, centroïdes de gain ajustés sur le
train — `cargo run --release -p llvq-bench --bin llvq-bench`) :

| code | bits/dim | MSE | rétention | classes |
|---|---|---|---|---|
| papier, union `norm(Λ₂₄(12))` + 1 bit de gain | 2,0000 | 0,078 | 92,14 % | 383 |
| notre union `norm(Λ₂₄(13))` + 0 bit | 1,9999 | 0,0840 | 89,36 % | 383 |
| **coquille 12 seule + 1 bit de gain** | **1,9584** | 0,0805 | **92,81 %** | **79** |
| **coquille 13 seule + 1 bit de gain** | 2,0113 | 0,0751 | **92,83 %** | **82** |

La coquille 12 seule bat la meilleure configuration union du papier **à la
fois en débit et en rétention**, avec **4,8× moins de classes** et une norme
constante. Structure des coquilles (vérifiée par la même formule de
cardinalité que la série thêta) :

| m | \|Shell(m)\| | bits/dim | classes |
|---|---|---|---|
| 12 | 70 486 236 999 360 | 1,917 | 79 |
| 13 | 169 931 095 326 720 | 1,970 | 82 |
| 14 | 384 163 586 352 000 | 2,019 | 115 |
| union m ≤ 13 | 280 974 212 784 720 | 2,000 | 383 |

⚠️ **Ne pas surinterpréter.** C'est une source gaussienne i.i.d., un seul
harnais, une seule seed. Le papier mesurait une *distance angulaire au plus
proche voisin* sur une source radialement uniforme, pas une rétention MSE
après quantifieur de gain — deux métriques peuvent classer différemment, donc
ce n'est pas une contradiction frontale. Mais ça contredit la conclusion
pratique qu'il en tire (« we therefore adopt this approach and recommend
doing the same »). **À revérifier sur de vrais poids après la boucle GPTQ
avant d'engager un noyau dessus** — les poids ne sont pas gaussiens et le
GPTQ déforme leur distribution.

Si ça tient, ça simplifie énormément le noyau G6 *et* accélère l'encodeur
(4,8× moins de classes à balayer). Le format d'index v1 et le moteur de
recherche sont en revanche épinglés sur la boule m ≤ 13 : passer à une
coquille unique casse la compatibilité des fichiers quantifiés (§4).

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
