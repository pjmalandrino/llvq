# Passation — session d'exécution P1→P7 (après le lot du 2026-08-13)

> **Pour la session qui exécute.** Ce document est autonome : tout ce qu'il
> faut savoir pour travailler est ici ou pointé d'ici. Le plan est validé par
> l'opérateur dans son principe (« on fonctionnera par P1, P2… comme
> proposé ») ; **chaque dépense de carte reste soumise à un go explicite,
> chantier par chantier** (mémoire projet : annoncer le coût avant, le cumul
> après).
>
> ⚠️ **Ce plan rouvre de fait l'axe noyau**, qui était formellement arrêté
> (`docs/PLAN.md`). La validation du plan par l'opérateur en est l'arbitrage.
> La clause de profondeur ≤ 24 du spec X4, elle, n'est PAS encore rouverte —
> c'est une décision distincte, à prendre en P5 (voir sa fiche).

## 1. Où on en est (le lot du 2026-08-13, en huit lignes)

- **E3 est mort une seconde fois, plus durement** : en ordre-fichier réel, le
  meilleur point *admissible* (shift-only ET profondeur ≤ 24) vaut
  **2,9650 b/poids noyau** contre le critère de 2,60 — mesuré-largeurs,
  [`docs/mesures/e1v-ordre-fichier-4b-2026-08-13.txt`](../mesures/e1v-ordre-fichier-4b-2026-08-13.txt).
- **Mais deux leviers neufs sont chiffrés** : l'adressage **warp-scan**
  (chaque bloc paie sa largeur, offsets par somme préfixe — jamais prixé
  avant, −9,5 à −27 bits/bloc selon layout) et **`e1v-séparé`** : 53,332
  bits/bloc → **2,3709 b/poids** sous warp-scan, à ~2 bits du plancher
  théorique absolu de toute variante à en-tête.
- ⚠️ `e1v-séparé` = `radix2` **au bit près** (test épinglé). Son vert est une
  re-qualification : la largeur était déjà publiée le 08-12 sous 2,60 et
  classée ❌ sur le seul booléen shift-only. **Ce qui manque n'est pas la
  place, c'est le décodeur** : la re-bijection CNS n'a zéro ligne de code.
- `golay_signs` mort comme layout (4,13 mesuré vs seuil 3,45) — survit comme
  brique d'E1v. Le MoE chaud/froid **en VRAM** mort (mélange 3,405 > 3,20,
  robuste à 5 tentatives de renversement) — l'héritière est le tier froid
  **en RAM hôte** (miss = copie PCIe ~0,35 ms, ~25× moins cher qu'un
  décodage).
- Dérive de distribution 4B→8B : **≲ 0,003 bit/bloc** (mais test non létal —
  l'histogramme est une propriété du codebook, pas du modèle).
- Tout est à 0 $, aucun job GPU lancé.

**Décisions produit de l'opérateur, à respecter** : le barreau 40 Go dense
via Golay70 v2 **ne l'intéresse pas** (« si c'est la VRAM du q4 avec une
qualité moindre, c'est inutile ») — ne pas le remonter sans la comparaison
70B@2bits vs 32B@q4. Le cas d'usage qui l'intéresse : **gros input, sortie
courte** (extraction JSON) — c'est le package B.

## 2. L'état de l'arbre de travail (à régler en étape 0)

Branche `claude/html-mini-cours-planes-w9prvm`, en avance de 19 commits sur
origin. **Non commité** :

| fichier | quoi | urgence |
|---|---|---|
| `proofs/preregistration-2026-08-13.md` | le pré-enr. qui juge le lot du 13 | 🚨 **son antériorité ne tient qu'à un mtime** — commit + `ots stamp` (opérateur) |
| `llvq-bench/src/bin/radixstudy.rs` | 639 → ~2000 lignes : modes FileOrder/WarpScan, variantes e1v/golay_signs, verdict en conjonction, 28 tests dont 2 prouvés létaux par mutation | commit avec les journaux |
| `docs/mesures/e1v-ordre-fichier-4b-2026-08-13.txt`, `e1v-8b-2026-08-13.txt`, `moe-ciseau-2026-08-13.txt` | les mesures du lot | commit |
| `ops/moe_ciseau.py` | le script du ciseau (stdlib seule, en-tête PEP-723) | commit |
| `docs/note-produit-2026-08-13.md` | **BROUILLON à faire arbitrer** (cases §A) | relecture opérateur = P0 |
| `docs/HISTORIQUE.md`, `docs/PLAN.md`, `docs/data/jobs.csv`, `docs/data/mmlu-dumps/`, `docs/mesures/mmlupair-4b-8b-2026-08-13.txt` | modifs antérieures à ce lot, en vol | **ne pas toucher sans comprendre** — vérifier avec l'opérateur |

**Étape 0 de la session d'exec** : (a) `git fetch` d'abord — deux copies du
dépôt poussent sur le même origin (mémoire projet) ; (b) proposer le commit
du lot du 13 (pré-enr. + code + mesures) et demander à l'opérateur le
`ots stamp` ; (c) obtenir l'arbitrage de la note produit — **sans le triplet
coché, S_alt n'a aucun statut et P4 n'a pas de critère d'admission complet**.

## 3. Les trois packages (ce vers quoi tout converge)

| | A — l'agent local | B — l'usine à extraction | C — le 70B de poche |
|---|---|---|---|
| modèle | MoE ~120B | 70B dense (validé au 8B d'abord) | 70B dense |
| carte | 32 Go + 64 Go RAM hôte | 24-32 Go | 32 Go |
| b/poids noyau | mélange ~2,9-3,2 ᶜ | 2,15 ᶜ (l'archive telle quelle) | 2,37 ᶜ (E1v + warp-scan) |
| verrou d'entrée | qualité 2 bits sur MoE : jamais mesurée | vitesse du décodeur + chemin GEMM à écrire | le décodeur E1v n'existe pas |
| chantiers | P2 → P6 | P1 → P3 → P4 | P1 → P4 → P5 |

Qualité de référence (les layouts ne changent pas le contenu décodé) :
8B mesuré MMLU 65,52 (−10,56 pp), ppl ×1,220. Rien au-delà du 8B, rien sur
MoE.

## 4. Les chantiers, fiche par fiche

**Règle transversale : pré-enregistrer avant de mesurer.** Chaque chantier
qui produit un chiffre décisif écrit d'abord son pré-enregistrement dans
`proofs/` (seuils ci-dessous = propositions issues de la contre-expertise du
13, à figer ou amender AVANT la première mesure), le commit, et demande le
`ots stamp` à l'opérateur.

### P1 — Le banc décodeur Metal (0 $, ~1 semaine) — sert B et C

**Objectif** : trancher si un décodage du rang (cascade uniformisée, marche
binomiale) peut être assez rapide, sur le Mac, avant d'écrire une ligne de
CUDA.

- **Harnais existant** : `llvq-metal/src/bin/decode.rs` — trois bras mesurés
  (sol 0,08 · masques 0,11 · rang-cascade 8,27 ns/bloc). ⚠️ Ses réserves
  connues (`docs/format-noyau.md:142-148`) : codes synthétiques à 4
  magnitudes, divergence inter-lanes non testée. **Le banc P1 doit tourner
  sur la distribution réelle de classes** (tirer les blocs du 4B scellé), et
  re-mesurer les ancres dans le même run — jamais contre les 0,11/8,27
  historiques.
- **Bras à ajouter** : (1) **cascade uniformisée** — 24 itérations identiques
  pour toutes les classes, réciproques magiques par (classe, étage) en table,
  candidats en ILP, sélection branchless, zéro indexation dynamique ;
  (2) **marche binomiale** (le décodeur d'E1v) — unranking par table C(n≤24,
  k≤12) en L1, pas à compte fixe, zéro division ; (3) **bras étalon
  cascade-archive** tel quel (l'étage « 1bis » : si la tolérance d'un cadre
  capacity-first accepte la cascade, E1v est mort-né — l'archive fait 2,19 et
  existe déjà).
- **V0 avant V1** : exactitude d'abord — chaque décodeur vérifié bloc à bloc
  contre `FastDecoder::decode` (`llvq-search/src/fastdec.rs`) sur un
  échantillon large, puis le sweep intégral des 150 681 600 blocs (harnais du
  sweep E1c, `llvq-artifact/tests/`). Tout écart enterre sans banc.
- **Seuils proposés à pré-enregistrer** : marche binomiale kill si
  > **1,5 ns/bloc** ; cascade uniformisée kill si > **2,0 ns/bloc** ; le bras
  CUDA (P4) n'est autorisé que si ≤ **0,45 ns** (le derating ×2 que le
  précédent Golay70 impose — un compte niveau source a déjà été faux d'un
  facteur 2 sur ce noyau).
- **Kill aval** : > 2,0 ns ⇒ le package C meurt, le B se réduit au prefill
  pur (où le décodage s'amortit par le nombre de tokens du lot).

### P2 — MoE : le dump temporel et le choix du modèle (0 $, ~2 jours) — sert A

- **Re-run du banc de routage en conservant l'ordre temporel**
  (`ops/moe_routing.py`, ~438 s MPS sur le 20b — le dump actuel
  `docs/data/moe-routing-gptoss20b-2026-08-12.json` est agrégé, donc aveugle
  au LRU). Produire les courbes de hit statique ET LRU à α ∈ {0,27 ; 0,45 ;
  0,5}.
- **Seuil proposé** : hit (statique ou LRU) < **96 %** à α = 0,45 ⇒ le
  package A recule d'un cran (le budget de miss RAM-hôte est ~3-5/token à
  0,35-0,75 ms le miss — recalculer le budget exact dans la préreg avec les
  chiffres de `ops/moe_ciseau.py`).
- **Choisir le modèle MoE quantifiable** : gpt-oss est MXFP4 natif **sans
  référence f16** et la politique « expert mort » manque au pipeline — deux
  bloquants connus. Passer en revue l'étude MoE du dépôt (grep « mémoire
  extrême », commits bd053a5/f8840b3) et documenter le choix (candidat à
  référence f16 propre, taille quantifiable en heures de Mac ou ~10 $ de
  carte).
- ⚠️ Réserve à porter partout : le dump est un 20b/32 experts, la cible un
  ~120b/128 — un vert ne se transporte pas, un rouge si.

### P3 — KV q8 (0 $, ~2-3 jours) — sert A et B, et les seuils de la note

- Quantifier le cache KV en int8 (échelle par tête ou par groupe) dans
  `llvq-llm`. Le levier : ~1,35 Go au lieu de 2,7 à 8k sur un 70B, et c'est
  lui qui fait passer le seuil du barreau 32 Go de 2,58 à 3,09 (note produit
  §B).
- **Validation** : ppl et MMLU sur le 4B scellé, Metal, **empreinte de tokens
  identique des deux bras** (`3f1baca9033bf251` / `65dcd53655e8bfa5`), même
  dtype. Kill si perte > 1σ (ppl : 0,7 % ; MMLU : σ McNemar 0,4-0,6 pp).
- Précédent rassurant : l'embedding q8 avait coûté 0 (−0,02 % ppl, MMLU dans
  le σ). Mais le KV traverse l'attention à chaque token — ne pas présumer.

### P4 — Le job carte mutualisé (~0,3-0,5 $ — **GO REQUIS**) — sert B et C

Un seul job L40S, une seule image, mêmes rounds :

- **Bras k-colonnes** : planesbench étendu — {cuBLAS f16 n=k (dénominateur
  publié) ; matvec-k f16 maison (contrôle, jamais publié seul — le piège
  `broadcast_matmul` est documenté) ; Planes14 ; Planes12x ; Golay70 v2} ×
  k ∈ {1, 2, 4, 8}, 7 rounds dont 2 jetés, rapports round par round.
- **Bras cascade/marche CUDA** — seulement si P1 l'a autorisé (≤ 0,45 ns).
- **X3 au passage** : le banc E1c gelé (~0,2 $) profite de la même carte.
- **Seuils proposés (préreg avant le job)** : K1 — si Golay70 v2 < 0,95×
  Planes12x **au même k** pour tout k ≤ 4, la famille des décodeurs lourds se
  referme (jamais recycler le 2,0× de n=1 : à k=4 Planes12x monte aussi, un
  seuil absolu serait un déplacement de poteaux). K2 — à k=8,
  T(Golay70) ≤ 0,60 × T(k=1), sinon le modèle d'amortissement est faux.
  K3 — le contrat zéro-spill (`fused_cuda.rs` refuse `local_bytes ≠ 0`) doit
  tenir avec k accumulateurs, sinon k plafonne à 4.
- Annoncer le coût avant le lancement, le cumul après (règle budget HF).

### P5 — La bijection E1v (0 $ dev, ~1 semaine) — **seulement si P1 vert** — sert C

- **D'abord la décision de passation** : rouvrir la clause « profondeur ≤ 24 »
  du spec X4 (`docs/archive/spec-memoire-extreme-2026-08-12.md:185-188`).
  Le binaire classe déjà e1v en « réouverture, décision de passation, pas un
  résultat de banc » — cette passation-ci est l'endroit où l'écrire, avec le
  précédent E2→v2 comme forme (idée neuve + nouveau critère pré-enregistré).
- Écrire la **re-bijection CNS** au transcodage : même cardinalité par classe
  (les largeurs sont prouvées suffisantes — c'est le résultat T2), rang
  d'arrangement re-numéroté en système combinatoire, champs séparés par étage
  (la sous-variante sans division, 53,332 bits/bloc, qui fait foi).
- **Preuve** : sweep intégral des 150 681 600 blocs contre le décodeur
  d'archive (harnais E1c). Tout écart enterre. **Chronométrer le
  transcodage** : précédent Planes12x = 404 s (recherche réseau par bloc) vs
  Planes14 = 84 s ; E1v est un re-rangement sans recherche réseau — attendu
  côté 84 s, à vérifier, pas à affirmer.
- Test létal à garder vert : `e1v_is_never_narrower_than_the_class_rank`
  (une largeur sous ⌈log₂|classe|⌉ = bijection impossible = bug de comptage).

### P6 — Gate X5-MoE (~25-55 $ — **GO REQUIS**) — **seulement si P2 vert** — sert A

- La case vide qui commande tout le package A : **aucun MoE n'a jamais été
  quantifié à 2 bits** dans ce projet. Quantifier le modèle choisi en P2,
  mesurer ppl + MMLU contre sa référence, décider si le déficit 2 bits suit
  les paramètres **totaux** ou **actifs** — c'est la question posée depuis la
  contre-expertise, et elle conditionne aussi le rôle du tier froid.
- Préreg avec seuils de qualité AVANT le job. La politique « expert mort »
  doit être conçue avant (bloquant identifié).

### P7 — Quantification 70B (~150-200 $ estimé — **GO REQUIS**) — **seulement si B ou C validé au 8B**

- Le 32B avait été sous-estimé de 25 % (621 s/bloc mesurés contre ~500
  prédits) : passer par `uv run ops/run.py estimate` et prendre la marge.
  C3 (chargement bf16) est acquis ; `oracle` d'abord sur l'image, toujours.
- Ne se lance que quand un package a démontré sa chaîne complète au 8B.

## 5. Les règles de la maison (rappel opposable)

1. **Aucun run payant, aucun arrêt de run, sans go explicite** (mémoire
   projet). Annoncer le coût avant, le cumul après.
2. **Pré-enregistrement avant mesure** pour tout chiffre qui décide
   (`proofs/`, `ots stamp` par l'opérateur). Un pré-enregistrement horodaté
   est en lecture seule pour toujours.
3. `git fetch` en début de session — **deux copies du dépôt** poussent sur le
   même origin.
4. `oracle` à chaque nouveau backend ; `--features fast-linalg` partout où
   l'on paie ; clippy à zéro ; boucle `cargo test` debug (la suite
   `--include-ignored` = dizaines de minutes).
5. **Étiqueter chaque nombre** (mesuré/calculé/estimé) et sa comptabilité ;
   b/param modèle entier embedding compris (q8 = 8,5) ; rapports round par
   round, jamais quotient de minima ; tout effet < ~1,5 % sur 3 blocs est du
   bruit.
6. Tests **létaux** : muter avant de déclarer vert. Deux leurres ont encore
   été attrapés dans ce lot (fermeture de groupe no-op, verdict sur
   `shift_only` seul).
7. Commentaires et code en anglais, échanges et docs en français.

## 6. Annexe — les chiffres clés du lot du 13 (pour ne pas les rechercher)

| point | valeur | provenance |
|---|---|---|
| meilleur admissible (conjonction spec), ordre réel 4B | 2,9650 b/p (golay_tight FO/scan) | calculé-largeurs / ordre mesuré |
| e1v-séparé FO/scan | 53,332 bits/bloc → 2,3709 b/p | idem |
| e1v-séparé FO/max (sans warp-scan) | 57,000 → 2,5230 | idem |
| plancher de toute variante à en-tête | ≈ 52,5 bits → ~2,34 | calculé (entropie 40,98 + 10 + adressage) |
| archive (le fichier) | 48 bits → 2,1912 grp32 ; ~509 ops, 8,27 ns/bloc | calculé / mesuré Metal |
| dérive 4B→8B (e1v-séparé) | ≲ 0,003 bit/bloc | dérivé (test non létal) |
| ciseau MoE VRAM | mélange 3,4053 > 3,20 à α_min 0,8659 | calculé sur dump mesuré |
| miss RAM hôte (cellule ~11,2 Mo) | ~0,35 ms gen4 / ~0,22 gen5 | estimé (PCIe jamais mesuré ici) |
| seuils barreau 32 Go | 2,58 (8k, KV f16, marge 5) ↔ 3,09 (8k, KV q8, marge 2) | calculés — **triplet à arbitrer, note produit §A** |
