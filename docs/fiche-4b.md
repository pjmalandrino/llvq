# FICHE 4B — Qwen3-4B-LLVQ-2bit et son noyau fusé

> 🗓️ **BANDEAU D'ÉTAT — dernière revue le 2026-08-08.** Cette fiche est
> **exacte sur le fichier** (identité, octets, débits, provenance des chiffres
> de qualité) et le reste : rien n'a bougé dans l'artefact publié. Elle est en
> revanche **périmée sur l'environnement d'exécution**, qui a changé trois fois
> depuis le 03. Ce qui est faux, et où :
>
> | ce que la fiche dit | ce qui est vrai au 08-08 | source |
> |---|---|---|
> | §1.5 « en tirer la moindre vitesse aujourd'hui : le noyau **n'est pas branché** » et §6.10 « **pourquoi** le noyau n'est pas branché » | **Branché sur CUDA le 06** (`fused_cuda` + `bin/fusedrun`). 48,7 tok/s dans 2,96 Go contre 43,6 dans 8,04. Toujours **rien sur Metal** — la fiche reste juste pour le Mac qu'elle décrit | [`mesures/planes14-fusedrun-2026-08-06.txt`](mesures/planes14-fusedrun-2026-08-06.txt) |
> | §6.10 « une implémentation GPU de `Rotation` … **qui n'existe pas** » | **Elle existe sur CUDA** : `rot_apply`, vérifiée contre une référence f64 sur 8 formes (pire rel. 9,5e-8), 8,05 µs à n = 2560 en isolation, et exécutée dans le chemin fusé | [`mesures/rotation-cuda-2026-08-05.txt`](mesures/rotation-cuda-2026-08-05.txt) |
> | §6.7 « **Grouped32 et Flat32 n'ont jamais tourné sur le modèle entier** » — que §6.7 annonçait déjà comme réparable | **Réparé le 05** : ils ont tourné dans le même processus que `Slot32`, une seule comptabilité, sept bras entrelacés. Flat32 **5,256 b/poids / 0,91×**, Grouped32 **3,498 / 0,69×** | [`mesures/k1-metal-2026-08-05.txt`](mesures/k1-metal-2026-08-05.txt) |
> | §5.4 « 2,2 – 7,6 tok/s … **sans cache KV** » | `bin/run` **a un cache KV** depuis le commit `9c24d26`, épinglé par un test qui exige les mêmes tokens que le chemin non caché. Sur L40S le fichier scellé rend **42,7 tok/s** | [`mesures/mini-2026-08-05.txt`](mesures/mini-2026-08-05.txt) |
> | §5.5 « La colonne qualité : **vide, pas faible** » | **Remplie le 06** — et l'adversaire retenu n'est pas le q4 MLX mais **l'AWQ officiel de Qwen** : **70,04 ± 1,25 de MMLU** contre 55,59 chez nous, ppl ×1,105 contre ×1,384. La conclusion de §5.5 est confirmée dans le pire sens : le 4 bits ne perd **rien** | [`mesures/a4-campagne-2026-08-06.txt`](mesures/a4-campagne-2026-08-06.txt) |
> | §5.6 « e4/e8 : qualité **ABSENT**, interdit de publication » | **Mesurée le 06** : e8 (int8) **gratuit** (−0,02 % de ppl, MMLU sous le σ), e4 (int4) **+1,52 %**. L'interdit est **levé pour e8**, maintenu pour e4 | [`verdicts-lot-b-2026-08-06.md`](archive/verdicts-lot-b-2026-08-06.md) §B4 |
> | §6.4 « 2,09× [2,05–2,11] » comme chiffre du noyau | Le layout de référence n'est plus `Slot32` mais **`Planes14`** : 4,804 b/poids, 2,14–2,16× sur L40S, **1,14× plus rapide que `Slot32` à contenu décodé identique** | [`mesures/c1-planesbench-2026-08-06.txt`](mesures/c1-planesbench-2026-08-06.txt) |
>
> ⚠️ Ce qui **n'a pas** changé et reste la référence du dépôt : la note de
> provenance des trois comptabilités de b/poids, le statut de chaque chiffre de
> qualité, et le §7 des manques. La **méthode** de cette fiche est ce qu'il faut
> garder ; ses statuts d'environnement, non.

**Source de vérité unique.** Établie le 2026-08-03 depuis les octets du fichier, le code aux commits concernés, les quatre logs de run et l'historique git. Les documents du dépôt (README, LAUNCH_ME, CLAUDE.md, docs/\*) n'ont servi que de pistes ; là où ils divergent de l'objet, l'objet gagne et la divergence est signalée.

**Statuts employés :** `MESURE` (une trace existe) · `CALCULE` (dérivé arithmétiquement de mesures nommées) · `SUPPOSE` (personne ne l'a mesuré) · `ABSENT` (n'existe pas) · `RETRACTE` (chiffre faux encore publié quelque part).

**Machine de référence** (`system_profiler`, 2026-08-03) : MacBook Pro Mac15,8, Apple M3 Max, 16 cœurs CPU (12P+4E), **40 cœurs GPU**, `hw.memsize` = 68 719 476 736 o (= 64 GiB = 68,72 Go décimaux — le « 69 Go » de CLAUDE.md est correct en décimal, ne pas le « corriger »). La bande passante crête de 400 Go/s de cette variante est une **spec constructeur, `SUPPOSE`** : rien dans le dépôt ne la mesure.

---

## 1. L'objet

### 1.1 Identité

| | valeur | statut |
|---|---|---|
| chemin | `/Users/pjmalandrino/qwen3-4b-llvq.bin` | MESURE |
| taille | **1 770 527 533 octets** (1,771 Go) | MESURE |
| sha256 | `9db213ef9fa9d7d7000789a8a529ce9459ce9ba6002ef5a72fd5a1c05c1c84b0` | MESURE |
| mtime | 2026-07-31 17:56 | MESURE |
| = HF | `x-linked-etag` et `content-length` de `Pier-Jean/Qwen3-4B-LLVQ-2bit` identiques ; dépôt HF au commit `f00daa7bc1dd12a720304a4483f2219d10f15c96` | MESURE |

Reproduction : `shasum -a 256 /Users/pjmalandrino/qwen3-4b-llvq.bin` ; `curl -sIL https://huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-2bit/resolve/main/qwen3-4b-llvq.bin`.

Le dépôt HF ne contient que `.gitattributes`, `LICENSE`, `README.md`, `qwen3-4b-llvq.bin`.

### 1.2 Contenu, lu octet par octet

Magic `LVQ2`, trois sections, et le parse atterrit **exactement** sur la taille du fichier (écart 0) :

| section | octets | contenu |
|---|---|---|
| matrices | 980 790 202 | 252 matrices quantifiées |
| tenseurs bruts | 778 313 898 | 146 tenseurs f16 |
| blobs | 11 423 433 | `config.json` (726 o) + `tokenizer.json` (11 422 654 o) |
| **total** | **1 770 527 533** | = taille du fichier |

Décomposition de la section matrices, qui boucle **à l'octet** :

```
payload  980 770 752  (= 7 846 166 016 bits, cf. §3.3)
framing      19 450   (en-tête 8 + noms 10 370 + métadonnées 252×28 = 7 056 + préfixes 252×8 = 2 016)
             -------
             980 790 202
```
> Le relevé initial annonçait 33 804 o de framing ; **19 450** est reconstruit analytiquement et vérifié. `CALCULE` sur entrées `MESURE`.

**Tenseurs portés** : `model.embed_tokens.weight` [151936, 2560] = **388 956 160** valeurs, + 36×(`input_layernorm` 2560, `post_attention_layernorm` 2560, `q_norm` 128, `k_norm` 128) + `model.norm` 2560 = 196 096. **Total porté 389 152 256**. Pas de `lm_head` : `tie_word_embeddings: true`.
Les 146 tenseurs sont **bit pour bit égaux à f16(bf16 du checkpoint)**, 146/146 vérifiés (`MESURE`).

**Total paramètres du modèle : 4 022 468 096** = 3 633 315 840 (projections) + 389 152 256 (portés).

> ⚠️ La constante `389_070_848` codée en dur dans `llvq-metal/src/bin/thesis.rs:432` **ne correspond à aucun tenseur** (ni 388 956 160, ni 389 152 256). Effet sur les tok/s : +0,03 %. À corriger, et à ne jamais citer.

**Blobs** : `config.json` et `tokenizer.json` sont des copies **octet pour octet** du checkpoint (`sha256(tokenizer.json)` = `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`, qui est le nom du blob HF amont). Conséquence exploitable : **les flux de tokens et les prompts MMLU sont identiques entre bras baseline et bras scellé par construction**, pas par chance.

`config.json` : hidden 2560, intermediate 9728, 36 couches, head_dim 128, 32 têtes / 8 KV, vocab 151936, `tie_word_embeddings` true, `torch_dtype` **bfloat16**.

### 1.3 Comptes de poids (relus dans les en-têtes de matrice)

| grandeur | valeur | statut |
|---|---|---|
| matrices | 252 (36 blocs × 7 projections) | MESURE |
| poids de projection | 3 633 315 840 | MESURE (fichier **et** log de run) |
| dont quantifiés | 3 616 358 400 | MESURE |
| dont queue `KeepExact` | 16 957 440 (0,4667 %) | MESURE |
| blocs de 24 | 150 681 600 | MESURE (relu, plus déduit) |
| lignes de sortie | 1 105 920 | MESURE — **exactement** les « 1 105 920 lignes vérifiées » du banc noyau |
| centroïdes de gain | 504 (2 × 252) | MESURE |

Formes par bloc : q 4096×2560 · k 1024×2560 · v 1024×2560 · o 2560×4096 · gate 9728×2560 · up 9728×2560 · down 2560×9728. Queues : `2560 % 24 = 16`, `4096 % 24 = 16`, `9728 % 24 = 8` → 471 040 poids de queue par couche × 36.

### 1.4 Relation avec `~/llvq-q4b.llvq`

`/Users/pjmalandrino/llvq-q4b.llvq` (980 790 202 o, magic `LVQ1`, sha256 `94f60e86…`) : ses octets `[8, 980 790 202)` sont **identiques** à ceux du fichier scellé — `sha256 = 5acd89c07afc143ce12ab5a04a4a24ba38f8bd7f0601d049e14e734715725a6b` des deux côtés. Seul le magic diffère. Le `.llvq` s'arrête là (le writer de l'époque n'écrivait pas les deux `u32` nuls de `finish()`).

Deux conséquences fortes :
1. **`bin/seal` fait un aller-retour décode → ré-encode bit-identique** sur les 150 681 600 index réels. Preuve gratuite, jamais revendiquée.
2. La preuve `verify_artifact` du log (« ✓ 3633315840 weights identical, bit for bit ») **couvre donc les octets publiés** sans rien relancer. Et comme le narrowing f16 est déterministe, `f16(decode(fichier)) == f16(poids_évalués)` : le bras MMLU est bien le narrowing du modèle f32 prouvé.

### 1.5 Ce qu'on peut / ne peut pas en faire

**On peut** : le charger sans réseau, sans cache HF, sans checkpoint (`bin/run`, `bin/ppl`, `bin/mmlu` acceptent `.bin`), et le passer aux trois bancs GPU (`thesis`, `matvec`, `decreal`).

**On ne peut pas** :
- l'ouvrir avec `transformers`, `vLLM`, `llama.cpp` ou MLX — format propriétaire ;
- le servir par le widget d'inférence HF — `config.json` et `tokenizer.json` sont **dans** le `.bin`, pas à la racine du dépôt ;
- ~~en tirer la moindre vitesse aujourd'hui : le noyau fusé **n'est pas branché** dans `bin/run` (§6.5)~~ — **vrai sur Mac, faux depuis le 2026-08-06 sur CUDA** : `bin/fusedrun` (linux + `--features cuda`) le fait tourner encodé, **48,7 tok/s dans 2,96 Go contre 43,6 dans 8,04** sur ces mêmes octets. `bin/run`, lui, décode toujours en mémoire sur tous les backends ;
- le reproduire à l'octet depuis HEAD (§2.4).

---

## 2. La configuration exacte

### 2.1 Réglage par réglage

| réglage | valeur livrée | ce que le code fait réellement | statut / preuve |
|---|---|---|---|
| codebook | `leech1c12` | `LeechShapeGain::with_caps(centroids, cap=12, level_cap=MAX_LEVELS_ANY=5)` — recherche angulaire plafonnée à Λ₂₄(12) | MESURE : `shell_cap = 12` sur 252/252 dans le fichier |
| plafond de niveaux | **aucun** | `MAX_LEVELS_ANY = 5` = le maximum structurel (3 types de mot, 2 libres, 5 impairs) → n'exclut rien | MESURE : le jeton `L<n>` **n'existait pas** au run (commit `fabab22`, 2026-08-01 19:22, soit 25 h après le scellement) |
| index | 47 bits | `index_bits(12) = ⌈log₂ N(12)⌉`, `N(12) = 111 043 117 458 000` | MESURE : longueur de flux = `nblocs × 6 o` sur 252/252 ; index max observé sur tout le fichier = **111 043 117 450 038** (7 962 sous N(12) : l'espace est saturé) |
| gain | **1 bit**, 2 centroïdes/matrice | Lloyd–Max, 40 itérations, sur les normes de bloc relatives de *cette* matrice, poids déjà tournés | MESURE : 2 centroïdes sur 252/252 |
| bits/bloc | **47 + 1 = 48** = 6 octets pile, MSB-first, sans bourrage | → **2,000000 b/poids exactement** pour le code de réseau | CALCULE, exact |
| échelles de ligne | 1 105 920 en **f64** | `row_scale = sqrt(Σ row²/(d_in/24))`, calculée une fois sur les poids tournés **avant** la boucle, figée | MESURE. **0 sur 1 105 920 n'est représentable en f32** (ni 0/504 centroïdes) : le f64 est le prix de la preuve bit-pour-bit, pas une négligence |
| queue | `TailPolicy::KeepExact`, **f32 sur disque** | les colonnes non alignées sur 24 reçoivent la rétroaction d'erreur puis arrêtent la boucle ; elles ne produisent aucune erreur propre | MESURE |
| rotation | **entrée seule**, graine de base `0x110FEED` | `Q = (Q_odd ⊗ H_m) D` ; graine effective `base ^ (bloc<<32) ^ (act<<16)` ; **144 graines distinctes** = 36 blocs × 4 activations (q/k/v partagent, o, gate/up partagent, down) | MESURE : les 252 graines du fichier reproduisent la formule **sans une exception**. Aucun `rotate_weight_cols` n'existe dans le code |
| `group_scales` | **off** | double verrou : arg 5 = `nogs` ≠ `gs`, **et** `ensure!(!cfg.group_scales)` interdit d'écrire un artefact avec | MESURE, déjà au commit du run |
| rétraction | `true` — **et c'est un no-op** | `retraction_target()` renvoie `None` quand `retract_to_level`, donc le rescale de `gptq.rs` est intégralement sauté | MESURE (§2.3) |
| amortissement | `1e-2`, **relatif** à `mean(diag H)` | codé en dur au commit du run ; `LLVQ_DAMPING` (défaut identique) arrive le 2026-08-01 | MESURE. **Jamais balayé** |
| dtype | **f32** partout | `ck.var_builder(DType::F32, …)` littéral ; `LLVQ_DTYPE` n'existe que depuis `80191d2` (2026-08-01 21:59) et n'atteint `smoke` que le 2026-08-03 | MESURE |
| calibration | C4 validation **shard 00000**, 64 fenêtres × 2048 = **131 072 tokens**, préfixe contigu depuis le token 0, **sans graine** | `LLVQ_CALIB_SEED` n'existait pas | MESURE |
| threads d'encodage | 16 | valeur **résolue** imprimée par le binaire | MESURE (log ligne 1) |
| portée | 36 blocs / 36, 252 matrices | | MESURE |

### 2.2 La ligne de commande qui a produit l'objet

```bash
LLVQ_MODEL=Qwen/Qwen3-4B LLVQ_CALIB=c4 LLVQ_ARTIFACT=/Users/pjmalandrino/llvq-q4b.llvq \
cargo run --release -p llvq-llm --features metal,fast-linalg --bin smoke -- \
  64 2048 12 4096 metal nogs leech1c12 999 rot
```
Puis le scellement :
```bash
LLVQ_MODEL=Qwen/Qwen3-4B cargo run --release -p llvq-llm --bin seal -- \
  /Users/pjmalandrino/llvq-q4b.llvq /Users/pjmalandrino/qwen3-4b-llvq.bin
```

Sémantique des positionnels (lue au commit du run, `51d7c55`) : `0` n_calib · `1` calib_len · `2` n_eval · `3` eval_ctx · `4` device · `5` **teste `== "gs"`** (donc `nogs` n'est pas un mot-clé, juste une chaîne différente) · `6` codebook (`leech` retiré → `1c12` → pas de suffixe `f` → magnitude non libre ; split sur `c` → gain_bits=1, max_shell=12) · `7` limit (toute valeur ≥ 36 équivalente) · `8` `rot` → seed `Some(0x11_0FEED)` · `9` absent → pas de dump safetensors.

**Deux incertitudes résiduelles.** (a) `--features fast-linalg` n'est **pas traçable** : le garde-fou qui l'imprime est postérieur, le log ne porte pas de profil de phases. La feature existait bien à `51d7c55` (donc la commande est *valide*), et la durée est compatible — mais rien ne le prouve. (b) L'argument 7 (`999`) n'est pas distinguable de toute valeur ≥ 36.

**Provenance du binaire** : commit `51d7c55` (2026-07-31 12:36:19), surdéterminé par quatre indices — mtime du `.llvq` (16:41) moins 14 447 s → démarrage ~12:40 ; absence des lignes « model dtype », « hessian damping », « phases » ; et le format du compteur de progression `block N/405` (voir §2.3).

### 2.3 Là où la recette ne correspond pas à son nom — **quatre pièges**

1. **Ce n'est pas du « Spherical GPTQ ».** `LeechShapeGain::retraction_target()` renvoie `None` dès que `retract_to_level` (le défaut, et la valeur du run), et dans `gptq.rs` tout le rééchelonnage est sous `if let Some(target) = …`. La rétraction est **intégralement sautée**. Et ce n'est pas un interrupteur mal placé : dès qu'un code de gain a un codebook fini, `quantize` a déjà posé le bloc sur la sphère du niveau le plus proche, donc l'Eq. 17 n'a plus rien à faire — **même à `gain_bits=0`**, `fit_gain_centroids` rendant alors un seul niveau. Le second étage (`refine_group_scales`, Algorithme 3) est en plus doublement désactivé.
   → **Nom correct de la recette livrée : « Algorithme 1 (shape–gain, reset de gain) + rotation d'incohérence en entrée ».** À corriger dans README.md:117, LAUNCH_ME, CLAUDE.md.

2. **Les logs mentent sur la configuration.** La chaîne `(shape–gain, 0 gain bits, spherical retraction, group scales …, input rotation …)` est un **littéral codé en dur** dans l'`eprintln` de `smoke.rs` — encore à HEAD. Elle ne reflète ni `gain_bits` (qui vaut **1**) ni l'état de la rétraction (no-op). Elle est présente dans **les trois logs de run**, y compris celui dont le bras B est explicitement à magnitude libre. Seule la ligne de résultat (`leech1c12`) est fiable.

3. **`block 1/405` ne désigne pas 405 blocs.** `405 = 9728/24`, le nombre de blocs de 24 colonnes de `down_proj` : une variable `nblocks` shadowée dans la boucle par matrice. Le log compte bien 36 lignes. Corollaire utile : ce format **date le binaire** — `/36` avant `51d7c55`, `/405` après.

4. **Le bit de gain est porteur, contrairement à ce que le log affirme** : niveau 0 = 72 008 871 blocs (47,79 %), niveau 1 = 78 672 729 (52,21 %). **Aucun bloc n'est codé à l'origine** (0 sur 150 681 600). Par matrice, la fraction au niveau 1 va de 0,4660 à 0,7604 (médiane 0,5143) — l'image « équilibrée » est trop lisse. Centroïdes : niveau 0 moyenne 0,8723, niveau 1 moyenne 1,1146, strictement croissants sur les 252, rapport moyen 1,2791. `MESURE`, fait neuf.

### 2.4 Reproductibilité : **deux blocages indépendants**

1. **Corpus.** Le commit `aba3989` (2026-08-01 22:03) déplace `LLVQ_CALIB=c4` du shard `00000` au shard `00001`. Le run publié calibrait sur le **shard 0**, qui est aussi celui que `bin/ppl … c4` lit pour **évaluer**. Deux conséquences : relancer la commande publiée à HEAD calibre sur un autre texte ; et **aucune perplexité C4 de cet objet ne peut être produite par la commande standard sans être contaminée**.
2. **Conteneur.** À `51d7c55` l'écrivain vivait dans `llvq-llm/src/artifact2.rs` avec `MAGIC = b"LVQ1"` et un `finish()` qui n'écrivait rien. À HEAD c'est `llvq_artifact::ArtifactWriter`, magic `LVQ2` (`LVQ3` dans l'arbre de travail) et `finish()` écrit deux `u32` nuls. Un re-run à HEAD produirait **980 790 210** octets avec un autre magic. Les enregistrements de matrice, eux, restent comparables.

S'ajoute un troisième écart, cosmétique mais parlant : le binaire de HEAD imprime le dtype, l'amortissement et une progression française par bloc de transformer ; le log publié n'a **aucune** de ces lignes.

> **Formulation honnête** : la commande publiée reproduit la **méthode**, pas les **octets**. Ni README.md:210 ni LAUNCH_ME.md:147-151 ne le disent.

---

## 3. Les chiffres, avec leur provenance

### 3.1 Perplexité

| valeur | statut | commande | objet | dtype | protocole | trace | caveat |
|---|---|---|---|---|---|---|---|
| **LLVQ 16,9617** / baseline **12,2336** → **×1,3865** | MESURE | la commande `smoke` de §2.2 (boucle `ppl` interne, avant/après quantification) | **modèle en mémoire**, pas le fichier | f32 | wikitext-2 test, ctx 4096, 12 fenêtres non chevauchantes, préfixe contigu | `~/llvq-run-4b-artefact.log`, bloc final | Jamais produit par `bin/ppl`. **Iso-conditions garantie par construction** : un seul `test_ids`, une seule fermeture `ppl`, un seul objet modèle dont seules les 252 projections ont changé — c'est plus fort qu'une empreinte, et c'est pourquoi ce couple n'en porte pas. `verify_artifact` rattache ce modèle aux octets publiés (§1.4). Aucune commande du dépôt ne le rejoue tel quel |
| **LLVQ 16,9415** / baseline **12,2361** → **×1,3845**, empreinte `3f1baca9033bf251` des deux côtés | **MESURE — rejoué et loggué le 2026-08-04** (`~/ppl-scelle-f16-2026-08-04.log`, `~/ppl-base-f16-2026-08-04.log`) : les deux valeurs reproduisent au dix-millième et les deux empreintes sont identiques. Le manque n° 1 de §7 est comblé | `LLVQ_DTYPE=f16 … --bin ppl -- 4096 12 metal /Users/pjmalandrino/qwen3-4b-llvq.bin` et `LLVQ_MODEL=Qwen/Qwen3-4B LLVQ_DTYPE=f16 … --bin ppl -- 4096 12 metal` | **les octets publiés** (bras 1) et le checkpoint (bras 2) | f16 | idem | **corps du commit `8c17eff`** (2026-08-02 07:23), qui porte le tableau verbatim, l'empreinte et les deux ratios | **Le seul chiffre de qualité mesuré sur les octets livrés.** Aucun stdout conservé : argv et lignes par fenêtre irrécupérables. Un corps de commit est horodaté et immuable — au-dessus d'une phrase de CLAUDE.md, en dessous d'un log. **Arbitrage** : le relevé initial le classait `SUSPECT` faute de log ; la contre-expertise a exhibé la trace git, plus primaire → statut relevé à `MESURE`, avec la réserve explicite ci-dessus |
| **15,3272** (run de nuit) | **RETRACTE comme point de variance** | idem `smoke`, sans `LLVQ_ARTIFACT`, 10ᵉ arg de sauvegarde | overlay `~/llvq-q4b-c12.safetensors` | f32 | idem | `~/llvq-run-nuit.log` | Ce run affiche **la même ligne de configuration** `[leech1c12, 36 blocks, rot on, calib c4]` et le même `2,0702`, mais tournait sur un binaire **antérieur** au correctif `60068db` (2026-07-31 01:12) : magnitude f16 libre + `row_scale` recalculé par bloc → **2,6923 b/poids réels**. Le commit `db84454`, écrit 1 min après la fin du run, s'intitule littéralement « Resultat du run leech1c12 : 15,3272 a 2,6923 bits reels ». **Ce n'est pas 10,7 % de dispersion, ce sont deux quantifieurs.** (Le « 2,78 » qui a circulé est le débit du Qwen3-0.6B de `llvq-ab-retraction.log`, pas celui-ci.) |
| 14,2684 / 15,2909 / 14,9104 · 2,1117 b/poids | **RETRACTE** | — | modèles en mémoire, cap 13 | f32 | idem | CLAUDE.md:657-659, encore publié | Perplexités réelles, **débits faux** : 2,1117 annoncé valait 2,7338. Et 14,2684 vs 15,2909 est le couple de **dispersion** (§3.6), pas un effet de codebook |

**Surcoût de log-vraisemblance normalisé sur sa propre baseline** (`CALCULE`, la seule comparaison inter-papier qui tienne) :

| | Δ nats/token | vs QTIP |
|---|---|---|
| nous, f32 | ln(16,9617) − ln(12,2336) = **0,326772** | **+3,06 %** |
| nous, f16 sur le fichier | ln(16,9415) − ln(12,2361) = **0,325376** | +2,62 % |
| QTIP (17,04 / 12,41) | 0,317061 | — |
| LLVQ 0 bit du papier (17,05 / 12,41) | 0,317648 | +0,19 % |

> **Nous sommes au-dessus de QTIP et de leur config 0 bit**, avant même de payer les 8,5 % de bits en plus. La phrase « just under QTIP » du README dit **l'inverse de sa propre table**.

### 3.2 MMLU

| valeur | statut | commande | objet | dtype | protocole | trace |
|---|---|---|---|---|---|---|
| baseline **micro 70,42 ± 1,28** · macro 72,85 | MESURE | `cargo run --release -p llvq-llm --features metal --bin mmlu -- Qwen/Qwen3-4B metal 40` | checkpoint | f16 | Hendrycks 5-shot, dev de la même matière, logits des tokens uniques `" A".." D"` comparés en f32, 1 passe avant/question | `docs/mmlu-micro-2026-08-02.log` (2 620 s) |
| LLVQ **micro 56,09 ± 1,36** · macro 57,59 | MESURE | `… --bin mmlu -- /Users/pjmalandrino/qwen3-4b-llvq.bin metal 40` | **les octets publiés** | f16 | identique | même log (2 805 s), profil par matière complet |
| chute **−14,33 pp** (79,65 % retenus) ; macro −15,26 pp | CALCULE | — | — | — | — | papier : −9,5 pp (60,7 / 70,2) |

**Ce que « micro » veut dire ici — à écrire correctement, c'est l'axe que l'audit vient de refermer.** Ce **n'est pas** un recensement des 2 280 questions : sous `limit`, chaque matière est tirée à 40, donc un ratio poolé vaudrait algébriquement le macro. `micro()` est un **estimateur stratifié** repondérant les 57 taux échantillonnés par la population **réelle** de chaque matière (`professional_law` pèse 1 534/14 042 = 10,9 % du chiffre publié et est estimé sur 40 tirages). Le `±` est une **erreur-type stratifiée avec correction de population finie**, donc **1 σ**, pas un IC 95 %, et elle **exclut** toute variance de modèle, de prompt et de graine.

Tirage : Fisher–Yates seedé `SplitMix64(0x6_11B0 ^ subject.len())` — la graine ne dépend que de la **longueur du nom** de matière, donc elle est identique aux deux bras par construction. 2 280 questions sur 14 042 (16,2 %).

Profil du bras quantifié : `abstract_algebra` 10/40, `professional_accounting` 10/40, `machine_learning` 12/40 ; à l'opposé `high_school_european_history` et `international_law` 33/40, `high_school_psychology` 32/40. Le mécanisme (le 2 bits abîme le raisonnement, pas la restitution) est visible ; **le « exactement 25 %, le hasard » ne doit pas être présenté comme une égalité** — la barre par matière est de ±7 pp.

**Validation du harnais** : 70,42 contre 70,2 au papier = **+0,22 pp, 0,17 σ**. C'est l'argument le plus fort du dossier et il est sous-employé.

> ⚠️ README.md publie encore la ligne **macro** (72,85 / 57,59 / −15,3 pp) sans jamais écrire le mot « macro », et annonce « ±1 pp » là où les barres valent ±1,28 et ±1,36.

### 3.3 Taille et débit

Payload = 150 681 600×48 + 1 105 920×64 + 504×64 + 16 957 440×32 = **7 846 166 016 bits = 980 770 752 octets**.

| dénominateur | b/poids | statut | qui l'imprime | où il est publié |
|---|---|---|---|---|
| 3 616 358 400 (quantifiés seuls) | **2,169632** | CALCULE sur comptages MESURE | `bin/smoke` (ligne « artifact: ») | README, CLAUDE.md |
| 3 633 315 840 (projections, queue incluse) | **2,159506** | idem | `bin/seal` | carte HF |
| — (comptabilité idéale : queue f16, échelles f16) | **2,070226** | CALCULE | `Report::bits_per_weight`, ligne « effective rate » | **À NE JAMAIS CITER pour ce fichier** : il décrit un fichier qui n'a pas été écrit |

**Lequel afficher.** Le numérateur **inclut** les 542 638 080 bits de queue ; le diviser par un dénominateur qui **exclut** la queue (2,1696) impute son coût à des poids qui n'en font pas partie — c'est un ratio mixte. **2,1595 est le chiffre homogène**, et c'est celui de la carte HF ; 2,1696 est le conservateur. Les deux sont arithmétiquement exacts. Recommandation : afficher **2,1595 b/poids sur 3 633 315 840 poids de projection, en nommant le dénominateur**, et donner 2,1696 comme variante étiquetée.

**Décomposition (dénominateur 3 616 358 400)**, qui boucle au 7ᵉ chiffre :

| poste | b/poids |
|---|---|
| code de réseau (48 b/bloc) | **2,000000** exactement |
| queue en **f32** | 0,150051 |
| échelles de ligne en **f64** | 0,019572 |
| centroïdes f64 | 0,0000089 |
| **total** | **2,169632** (+8,48 % sur 2,000) |
| *si queue et échelles étaient f16* | *2,0799* (+4,0 %) |

> ⚠️ La décomposition du README (`+0.075 / +0.015 / ~0.05`) mélange **deux surcoûts vs f16** (0,075026 pour la queue, 0,014679 pour les échelles — tous deux **exacts sous cette convention**) avec un poste absolu erroné : le troisième terme devrait être 0,075026 (queue à f16) et il en manque un quatrième (échelles à f16, 0,004893). Dire « les chiffres sont faux » ferait rater que deux d'entre eux sont justes. **Choisir une convention et s'y tenir.**
>
> Gisement réel : la **queue f32 → f16** vaut 0,075 b/poids, deux fois plus que les échelles — mais alors la preuve bit-pour-bit porterait sur des poids f16. Les **échelles f64 ne sont pas réductibles sans casser la preuve** : 0 sur 1 105 920 n'est représentable en f32.
>
> Note pour un relecteur : `format.rs` documente « 0.0146 bits/weight » pour les échelles (le **surcoût** f64 sur f16) ; README dit 0,020 (le **coût total**). Aucun n'est faux, ils ne comptent pas la même chose. Ne pas « harmoniser ».

**Compression** :

| | valeur | statut |
|---|---|---|
| fichier | 1 770 527 533 o | MESURE |
| FP16 équivalent | 4 022 468 096 × 2 = 8 044 936 192 o | CALCULE |
| **ratio** | **×4,5438** | **MESURE** — imprimé par `bin/run` et par `bin/seal` |
| débit du modèle **entier** | **3,5213 b/paramètre** | CALCULE — le seul chiffre comparable au 4,50 du q4 |

> **`×4,63` est RETRACTE** (CLAUDE.md:724-733) : calculé sur une hypothèse d'artefact de 1,74 Go qui n'a jamais existé.

**Un chiffre à ne pas oublier** : sur les **projections seules**, 2,1595 contre 4,5000 pour le q4 = **×2,084**. Le ×1,278 disque est ce ×2,084 dilué par un embedding que MLX quantifie et pas nous.

### 3.4 Coût de production

| | valeur | statut | trace |
|---|---|---|---|
| durée du run publié | **14 447 s = 4,013 h** | MESURE | `~/llvq-run-4b-artefact.log` : « quantized 252 matrices, 3633315840 weights in 14447s » |
| par couche | min 328 s, max 592 s, moyenne 401 s | MESURE | 36 lignes horodatées |
| threads | 16 | MESURE | ligne 1 du log |
| coût | 0 $ (M3 Max local) | MESURE | — |
| profil par phase | **ABSENT** | — | l'instrumentation date du 2026-08-02 18:04, deux jours après le run : il n'a **jamais pu** exister |

> **RETRACTE** : « ~3,5 h » (README:129/205, LAUNCH_ME:143) est le **run de nuit** (12 715 s), pas celui qui a produit le fichier ; « 3,45 h » (CLAUDE.md) décrit la config cap 13. Le seul chiffre correct est **4,01 h**. L'écart de +13,6 % entre les deux runs n'est pas une variance machine : le run publié encode les 150 681 600 index **dans** la boucle et écrit en flux.

### 3.5 Preuve d'aller-retour

`✓ 3633315840 weights identical, bit for bit` — `MESURE`, `~/llvq-run-4b-artefact.log`. Comparaison `to_bits() == to_bits()` sur les 252 matrices, **f32**, queue comprise, après `decode_matrix`. Se transporte aux octets publiés (§1.4). **Ne couvre pas** un run natif f16 (le dépôt le dit lui-même dans `eval.rs`) — mais le narrowing étant déterministe, la propriété se transporte au bras f16 de MMLU.

### 3.6 Barre d'erreur — l'état réel

**Le projet possède exactement UNE observation de dispersion**, et elle n'est pas expliquée :

| | valeur | statut |
|---|---|---|
| couple | **14,2684 vs 15,2909** (36 blocs, calib wikitext, cap 13) | MESURE (lignes de tableau, aucun log de run) |
| écart | **~7,2 %** | CALCULE |
| pourquoi c'est une dispersion | le test `under_the_old_retraction_shape_gain_was_direction_only` **démontre que les deux bras étaient le même quantifieur** (écart relatif max **7,1·10⁻¹⁵**) | MESURE |
| cause | **non tranchée** : divergence numérique amplifiée sur 36 blocs séquentiels, **ou** différence de configuration non consignée | — |

> **Arbitrage explicite.** Le relevé « bras-llvq » a correctement réfuté le couple (15,3272 ; 16,9617) comme mesure de variance — mais en a conclu que « le dépôt n'a aucune barre d'erreur » et que « 7 % ne doit jamais être écrit ». **C'est une surcorrection** : le 7 % qui circule repose sur l'*autre* couple, adossé à un test nommé. **Ne pas supprimer la seule donnée de variance du projet.** Formulation juste : « une observation, n=2, cause non tranchée, aucun σ ».

**Conséquence directe** : la marge de 0,08 point sous QTIP (16,9617 vs 17,04) est **~90× sous la seule dispersion observée**. Elle n'est pas défendable. Le +3,06 % de surcoût de nats (§3.1) est du même ordre que cette dispersion — donc lui non plus n'est pas résolu.

### 3.7 L'A/B oublié : ce que coûte réellement le bit de gain

`~/llvq-ab-retraction.log` (2026-07-31 12:23) — **le seul A/B conservé du projet, deux bras dans un même fichier**, et il n'est cité sur aucune surface :

| bras | codebook | b/poids | ppl (Qwen3-0.6B, 3 blocs, ctx 2048, 12 fen., baseline 19,5038) |
|---|---|---|---|
| A | `leech1c12` — gain porté (47+1 = 48 b/bloc) | 2,1656 | **21,4157** (×1,098) |
| B | `leech1c12f` — magnitude f16 libre (47+16 = 63 b/bloc) | 2,7838 | **20,7582** (×1,064) |

→ **Coder le gain coûte +3,17 % de perplexité pour −0,618 bit/poids.** `MESURE`.

> ⚠️ Cela **contredit** la ligne encore publiée « quantifier le gain ne coûte presque rien : 0,04 % pour 0,52 bit » — laquelle provenait d'un A/B où les deux bras étaient le **même** quantifieur. Réserves à porter : 3 blocs, un 0.6B, et le suffixe `f` ne restaure que la magnitude libre (pas le second défaut du `row_scale` dérivant), ce qui explique que l'écart 4B (10,7 %) soit plus grand que l'écart 0.6B (3,2 %).

---

## 4. Face au FP16

### 4.1 Rigoureusement iso-conditions

**Perplexité (12,2336 / 16,9617)** — iso **garantie par construction**, pas vérifiée a posteriori : `bin/smoke` tokenise une fois, construit un seul `test_ids`, définit une seule fermeture `ppl` et l'appelle deux fois sur **le même objet modèle**, avant et après réécriture des 252 projections. Il n'existe pas deux flux de tokens à comparer. C'est plus fort qu'une empreinte — et c'est pourquoi ce couple n'en porte pas. 4 095 prédictions × 12 fenêtres = **49 140 tokens notés**, identique aux deux bras ; logits remontés en f32 avant `log_softmax` des deux côtés.

**Perplexité f16 (12,2361 / 16,9415)** — iso par l'empreinte `3f1baca9033bf251` annoncée identique. Et l'égalité est **attendue par construction** puisque le tokenizer scellé est byte-identique à celui du checkpoint (§1.2) : ce que le rejeu vérifiera réellement, c'est le nombre de fenêtres et le contexte.

**MMLU (70,42 / 56,09)** — iso sur tous les axes vérifiables : même binaire (« Finished release in 0.42s » au second bras), même session, exécutions consécutives dans un seul log, dtype f16 imprimé des deux côtés, mêmes 2 280 questions (graine déterministe), même tokenizer (prouvé). **Un seul manque** : `bin/mmlu` n'imprime **aucune empreinte** — l'identité des questions est établie par lecture de code, pas relue sur une sortie.

### 4.2 Écart de protocole **non contrôlé**

**Toute comparaison f32 entre le FICHIER SCELLÉ et le CHECKPOINT.** Le checkpoint est **bf16** ; `seal` écrit les tenseurs portés en **f16**. Mesuré : sur les 388 956 160 valeurs de l'embedding, **77 045 changent** (1,98·10⁻⁴), dont **451** tombent à zéro ; toutes les valeurs touchées sont sous 7,600·10⁻⁶ (zone subnormale f16), aucun débordement (max |v| = 0,250), erreur absolue max 2,98·10⁻⁸. Comme `tie_word_embeddings = true`, cet embedding **est** le `lm_head` : l'écart entre directement dans les logits.

→ **À f16 les deux bras convergent** (bf16→f16 des deux côtés) : MMLU et la ppl f16 sont propres, et le couple 16,9415/12,2361 n'a **aucun confondant d'embedding**.
→ **À f32 ils ne convergent pas** : baseline bf16→f32 (exact) contre scellé f16→f32. **Ne jamais comparer une ppl f32 du fichier scellé à la baseline f32 du checkpoint sans le dire.** Le couple publié 12,2336/16,9617 n'est pas concerné (il tourne en mémoire, embedding bf16→f32 des deux côtés). Personne n'a scoré le fichier en f32.

**Résidu non chiffré** : le bras baseline matérialise ses tenseurs par `VarBuilder::from_mmaped_safetensors`, le bras scellé par `Tensor::from_vec(f32).to_dtype(…)`. Même résultat arithmétique attendu, ordre conversion/copie différent. Impact `SUPPOSE` négligeable.

**Le « FP16 » du banc GPU n'est pas ce FP16.** Voir §6.2.

---

## 5. Face au 4 bits

### 5.1 L'adversaire, tel qu'il est sur le disque

| | valeur | statut |
|---|---|---|
| chemin | `/Users/pjmalandrino/qwen3-4b-mlx-q4/` (mtime 2026-08-01 16:06) | MESURE |
| `model.safetensors` | 2 263 022 417 o ; répertoire complet 2 274 510 217 o | MESURE |
| recette | `mlx_lm.convert --hf-path Qwen/Qwen3-4B -q --q-bits 4 --q-group-size 64` | MESURE (config.json : `{group_size: 64, bits: 4}`) |
| structure | 904 tenseurs = 253 poids U32 + 253 `.scales` BF16 + 253 `.biases` BF16 + 145 normes | MESURE |
| **MLX quantifie AUSSI l'embedding** | 253 tenseurs quantifiés = 252 projections + `model.embed_tokens` | MESURE — asymétrie jamais dite |
| débit | **4,500000 b/poids** sur les poids quantifiés ; **4,500561** tous poids (les 196 096 normes bf16) | CALCULE, exact |
| mêmes totaux | 4 022 468 096 poids des deux côtés | CALCULE — la base de comparaison est saine |

### 5.2 Axe par axe

| axe | nous | q4 | verdict |
|---|---|---|---|
| **disque** | 1 770 527 533 o · 3,5213 b/param | 2 263 022 417 o · 4,5006 b/param | **×1,2782 pour nous** — `MESURE` des deux côtés, **la seule ligne honnête du tableau**. Sur les projections seules : **×2,084** |
| **RAM** | voir 5.3 | 2,39 Go **`SUSPECT`** | comparaison actuellement invalide |
| **débit** | voir 5.4 | 129,8 tok/s **`SUSPECT`** | aucun chiffre commun |
| **qualité** | 56,09 ± 1,36 micro, tracé | **`ABSENT`** | la case qui décide est vide |

### 5.3 RAM — trois quantités différentes sous le même mot

| quantité | valeur | statut |
|---|---|---|
| MLX q4, « 2,39 Go » | pic de l'**allocateur MLX** (`mx.get_peak_memory()`, poids + KV + activations), **aucune trace** — ni log, ni script, ni historique shell ; prompt élidé dans le doc | **SUSPECT** |
| nous, « 3,28 Go » | arithmétique poids-seuls du format **Slot32**, que **`bin/run` ne charge jamais** | CALCULE, hors sujet pour le runner |
| nous, modèle **résident** de `bin/run` | 4 022 468 096 × 2 = **8 044 936 192 o** (8,045 Go), imprimé par le binaire | CALCULE, exact par construction |
| nous, **pic RSS mesuré** | **CPU 9,79 Go** (`cpu 12`, la commande publiée) · **Metal 17,41 Go** (reproductible à 0,0006 % sur 4 lancements) | **MESURE**, cette session |

Commandes : `/usr/bin/time -l ./target/release/run /Users/pjmalandrino/qwen3-4b-llvq.bin metal 1` et `… cpu 12`.
Mécanisme du pic Metal : **inconnu** (double résidence hôte/buffer en mémoire unifiée ? pool de tampons candle-metal ?). Ne pas l'écrire comme établi.

> **Les trois chiffres publiés sont tous faux** : « ~3,3 Go chargés » (README, LAUNCH_ME) décrit le noyau ; « 7,3 Go de pic » (`pistes-battre-q4.md:69`) est le `7.3 GB (FP16)` de `thesis.rs:11` réétiqueté ; « ~8 Go de RAM libre » (LAUNCH_ME:55) est **dépassé de 1,8 Go par la commande que le document propose en premier**.

À convention identique **poids seuls** : q4 4,5006 contre nous (Slot32 + lm_head f16) **6,5245 b/poids** = **×1,45** contre nous — pas le ×1,37 publié, qui n'est ni poids-contre-poids ni pic-contre-pic.

### 5.4 Débit — aucun chiffre commun

Ce que chaque nombre inclut, lu dans le code des deux côtés :

| chiffre | inclut | exclut |
|---|---|---|
| MLX **129,8 tok/s** (`SUSPECT`, aucune trace) | 253 matmuls quantifiés, attention, RMSNorm, RoPE, cache KV, lm_head, échantillonnage, détokenisation | prefill (compté à part en `prompt_tps` — `tic` est réinitialisé après le prompt), chargement, tokenisation. Mesuré à `--max-tokens 256`, prompt inconnu |
| nous **10,46 ms** | 252 matvec fusés, un token, mémoire froide | attention, normes, RoPE, KV, lm_head, échantillonnage, chargement, transcodage, **et la rotation d'entrée** |
| nous **78,2 tok/s** | ci-dessus + lm_head **modélisé** | idem ; le lm_head n'est jamais exécuté |
| nous **2,2 – 7,6 tok/s** | **tout**, bout en bout | rien — mais **sans cache KV et sans noyau fusé** |

Les 2,2–7,6 tok/s sont `MESURE` (cette session, `bin/run` Metal, f16, 4 prompts) : n_new=1 → 154,84 s ; n_new=13 → 161,15 s (7,60 tok/s sur la pente) ; n_new=49 → 225,32 s (2,24 tok/s sur la pente). Le débit **décroît** parce que `Qwen3::generate` rejoue tout le préfixe à chaque pas (le code le documente : « No KV cache … That is quadratic »). C'est un **plancher**, pas un régime permanent.

→ Face aux 129,8 tok/s : **entre ×17 et ×58 contre nous**, pas ×1,65. Le doc avait le bon sens et se trompait d'un ordre de grandeur.

### 5.5 La colonne qualité : **vide, pas faible**

Aucune perplexité, aucun MMLU, aucune tâche n'a jamais été passée sur le q4 — ni dans le dépôt, ni sur le disque. Le « ~1-2 % » de `face-au-4-bits.md` n'a été produit par rien, et le document l'admet. La seule alternative chiffrée qui circule (« ~63 MMLU pour du RTN 4 bits », arXiv:2505.02214) est une citation de littérature **non vérifiée**, attribuée à **deux réglages différents à deux endroits du dépôt**. Ne pas la citer sans relire la source.

**Verrou technique** : `bin/ppl` accepte un **overlay `.safetensors`** des 252 projections → une perplexité du q4 ne demande **aucune ligne de Rust**. `bin/mmlu` n'a que deux branches (chemin scellé ou repo HF) → un MMLU du q4 demande ~15 lignes copiées de `ppl.rs`.

### 5.6 Dans quel régime le 2 bits gagne réellement

**Démontré : un seul axe, le disque, ×1,278** — et c'est le moins précieux.

**Le seul créneau structurel** est la fenêtre mémoire où le q4 ne rentre pas et nous si. Sa largeur : 4,50/3,727 = **×1,21** avec Grouped32, 4,50/4,034 = **×1,12** avec L≤3 — soit **12 à 21 %**, pas les 18 % annoncés sur des chiffres 70B optimistes.

Recalcul 70B (Llama-3.1-70B, 70,554 Md, embedding + lm_head **non liés** = 2 101 346 304 = 2,978 %, laissés en f16 comme dans notre artefact) :

| | q4 | Slot32 | L≤3 | Grouped32 | disque |
|---|---|---|---|---|---|
| **recalculé** | 39,69 Go | **51,35** | **35,58** | **32,87** | **22,77** |
| publié dans `face-au-4-bits.md` | 39,4 | 48,2 | 32,1 | 29,3 | 19,0 |

Les chiffres publiés appliquent un débit **projections seules** à **tous** les poids : exact pour le q4 (qui quantifie son embedding), **optimiste de 6 à 12 % pour nous**, systématiquement dans le même sens.

Cette fenêtre repose sur **quatre inconnues** : le débit Grouped32 du modèle entier (la vitesse 0,68× est mesurée sur **une couche**), la qualité à 70B (**aucun 70B n'a jamais été quantifié**), le cache KV (**320 Kio/token** en f16 = 2,68 Go à 8k, **jamais budgété** — et le `~640 Ko/token` de `pistes-battre-q4.md:97` est **faux d'un facteur 2**, ce qui invalide la phrase qui suit), et le fait que **le format qui va vite est plus gros que du 4 bits**.

**Le levier qui déciderait de l'axe disque est déjà exécuté** — et il est interdit de publication :

| fichier | taille | b/poids | vs q4 | qualité |
|---|---|---|---|---|
| `~/q4b-e4.llvq` (embedding int4 g64) | 1 211 403 653 o | 2,4093 | **×1,868** | **ABSENT** |
| `~/q4b-e8.llvq` (embedding int8) | 1 405 881 733 o | 2,7961 | ×1,610 | **ABSENT** |

Tous deux au format `LVQ3` (arbre de travail non commité), section matrices bit-identique au fichier publié. L'outil qui les produit écrit lui-même « score the OUTPUT file (ppl + mmlu) before believing anything ». **Les publier sans mesure reproduirait exactement l'erreur de comptabilité que le README dénonce.**

---

## 6. Le noyau fusé

### 6.1 Le protocole exact de `bin/thesis`

1. compile deux pipelines Metal (`tv_f16`, `tv_slot`) ;
2. construit la table de 384 classes (un buffer constant **partagé**) ;
3. construit **une** activation : `SplitMix64(0x6_7451)`, 16 384 f32 gaussiens, **un seul buffer** partagé par les deux bras et les 252 matrices ;
4. boucle streaming sur les 252 matrices : `read_matrix_raw` → `transcode(Slot32)` → reconstruction f64 → arrondi f16 → références `y_ref`/`y16_ref` en f64 → upload de 6 buffers LLVQ + 1 buffer FP16 ;
5. **vérification** (§6.3), avant toute mesure ;
6. **mesure** : un seul command buffer par bras, 252 encoders, `d_out×32` threads en threadgroups de 256, tuilage identique (128 blocs = 3072 colonnes = 12 Ko de threadgroup memory des deux côtés) ; chrono **après** l'encodage, autour de `commit()` + `wait_until_completed()` seuls ; **7 passes, reps 0 et 1 jetées, minimum des 5 restantes** ; FP16 mesuré d'abord.

Sérialisation par le hazard write-write sur le buffer de sortie unique.

**Symétrie** : cinq asymétries trouvées, **toutes contre le bras LLVQ ou négligeables** — (i) FP16 mesuré d'abord (la dérive thermique pénalise LLVQ) ; (ii) surcoût de soumission non soustrait, ce qui comprime le rapport ; (iii) le bras LLVQ lit sa queue en f32 là où le FP16 la lit en f16 ; (iv) 9 binds contre 4 (hors chrono) ; (v) 12 Ko de table de classes non comptés dans ses 2,50 Go.

> **Corollaire non exploité** : tout terme additif **commun** (surcoût de soumission, 252 coûts d'encoder, lm_head) donne (T₁₆+c)/(T_slot+c) < T₁₆/T_slot. **Le 2,07× est donc un MINORANT du rapport ALU/mémoire pur.**

### 6.2 Le bras « FP16 » n'est pas le checkpoint FP16

`w16 = f16_bits(w)` où `w` est la reconstruction **f64 des blocs LLVQ**, dans la **base tournée**. Le bras « FP16 » lit les mêmes valeurs que le bras LLVQ, à l'arrondi près. Sans effet sur le **temps** (mêmes formes, mêmes octets) — mais c'est un baseline de **coût**, pas de qualité, et aucune surface publique ne le dit.

Ce baseline n'a par ailleurs **jamais été confronté à MPS, MLX ou Accelerate** : le 2,07× est un rapport contre un noyau écrit par le même auteur. C'est l'angle hostile restant, non adressé.

### 6.3 Vérification numérique

| | valeur | statut |
|---|---|---|
| lignes vérifiées | **1 105 920**, les 252 matrices, **les deux bras** | MESURE |
| métrique | max sur les lignes de \|got − want\| / max(Σ\|wᵢxᵢ\|, 1e-12), référence **f64** | MESURE |
| seuil dur | `assert!(e < 1e-3)`, identique aux deux bras, **avant** toute mesure de temps | MESURE |
| pire erreur LLVQ | **3,4·10⁻⁸** · Σ\|w·x\| | MESURE, **identique au chiffre près entre les deux exécutions connues et entre les deux fichiers** |
| pire erreur FP16 | 2,8·10⁻⁸ | MESURE |

Trois réserves : granularité **ligne**, pas bloc (des erreurs qui se compensent dans une ligne passeraient — la vérification par bloc vit dans `bin/decreal`) ; le seuil 1e-3 est 5 ordres au-dessus de l'erreur réelle, c'est un détecteur d'incendie, la **preuve est le pire-erreur imprimé** ; la référence LLVQ est construite par `rt.decode_block`, donc `thesis` **ne re-vérifie pas** le transcodage contre `Indexer::decode` (ce verrou vit dans `llvq-artifact/tests/runtime_format.rs`, sur des index synthétiques).

### 6.4 Les chiffres et leur dispersion

> 🏷️ **Section historique depuis le 2026-08-05 — ne plus y puiser un chiffre à
> publier.** Les trois lignes ci-dessous sont les runs du **banc à deux bras**
> (2026-08-01 et 2026-08-03), dont aucun n'a laissé de journal. Le chiffre
> courant vient du run archivé à sept bras : **`Slot32` 5,510 b/poids,
> 2,03× [2,03–2,10]** sur les mêmes 252 projections, journal
> [`docs/mesures/k1-metal-2026-08-05.txt`](mesures/k1-metal-2026-08-05.txt).
> Ce rapport-là est la **médiane du rapport formé round par round**, avec sa
> plage sur les 5 rounds gardés — pas le quotient de deux minima, qui mêlerait
> deux rounds n'ayant jamais coexisté. Et **les millisecondes dérivent d'un run
> à l'autre** (c'est précisément ce que cette section a établi) là où les
> b/poids et les octets reproduisent au chiffre : citer le b/poids et le
> rapport avec sa plage, renvoyer au journal pour les ms.

| | FP16 | LLVQ Slot32 | rapport |
|---|---|---|---|
| 2026-08-01 | **21,691 ms** · 7,27 Go · 335,0 Go/s | **10,460 ms** · 2,50 Go · 239,2 Go/s | **2,0737×** |
| 2026-08-01, 2ᵉ passe | — | — | 2,08× |
| 2026-08-03 (rejeu, fichier scellé) | 22,675 ms | 11,021 ms | 2,0574× |

`SUSPECT` sur les décimales, `MESURE` sur l'ordre de grandeur. **Aucun fichier de log n'existe pour ces trois lignes-là** : leurs valeurs ne vivent que recopiées dans README.md:147-155, CLAUDE.md:121 et la table de tête de `docs/format-noyau.md` § « La thèse, sur le modèle entier ». *(Le reproche est levé pour le banc courant : `docs/mesures/` existe depuis le 2026-08-05 et archive le run à sept bras ainsi que les trois invocations témoins du banc à deux bras.)* Écarts entre exécutions : FP16 +4,5 %, LLVQ +5,4 %, **rapport −0,8 %**.

> **Ce qui se publie aujourd'hui** vient du run archivé du 2026-08-05, pas de cette table : `Slot32` **5,510 b/poids**, **2,03× [2,03–2,10]** (médiane du rapport formé round par round sur les 5 rounds gardés), les 1 105 920 lignes, la pire erreur 3,4·10⁻⁸, les 7,27 et 2,50 Go. Millisecondes : `docs/mesures/k1-metal-2026-08-05.txt`.
> **Ce qui ne se publie plus** : le « 2,06–2,08× (n=2) » de cette section. Deux raisons cumulées — il agrège deux invocations distinctes du binaire, ce que §4.6bis de `docs/archive/portage-noyau-cuda.md` disqualifie ; et sa fourchette est plus étroite que la dispersion réellement mesurée depuis, [2,029 ; 2,080] sur trois runs témoins.
> **Les trois décimales de « 21,691 » ne survivent pas au rejeu et ne doivent pas être publiées.**

**Reproduction** : `cargo run --release -p llvq-metal --bin thesis -- /Users/pjmalandrino/qwen3-4b-llvq.bin`
⚠️ Le **défaut** des trois bancs GPU (`thesis.rs:191`, `matvec.rs:503`, `decreal.rs:139`) est `~/llvq-q4b.llvq`, **non publié**. Les commandes de README.md:222-231 échouent chez un tiers. Le fichier scellé fonctionne, et ses codes sont **bit-identiques** (§1.4) : la correction est purement documentaire. LAUNCH_ME.md:100-103 nomme d'ailleurs explicitement le fichier absent — c'est un **paragraphe** à réécrire.

### 6.5 Ce que le 2,07× **exclut**

(1) l'attention entière (QKᵀ, softmax, AV, RoPE, cache KV) ; (2) les 145 RMSNorm, dont les `q_norm`/`k_norm` **par tête** spécifiques à Qwen3 ; (3) la SwiGLU ; (4) les résiduels ; (5) **la rotation d'incohérence appliquée à x — 144 par token, payée par le seul bras LLVQ** ; (6) le `lm_head` lié (rajouté **analytiquement**, jamais exécuté) ; (7) l'échantillonnage sur 151 936 logits ; (8) tout le prefill (un seul token, batch 1) ; (9) le transcodage au chargement.

**Coût de la rotation** — arithmétiquement **0,206 %** des projections (1,499·10⁷ ops/token contre 7,267·10⁹ flops), `CALCULE` depuis `rotation.rs`. **En latence : inconnu, et c'est là qu'est le risque.** 144 transformées minuscules et sérielles ; le plancher de soumission documenté par ce crate est ~0,15 ms/dispatch, donc 144 dispatches naïfs coûteraient ~21,6 ms et **effaceraient toute l'avance**. Fusionnées, bien moins — mais **aucune implémentation GPU de `Rotation` n'existe** (ni WHT ni Hadamard dans `llvq-metal`). **Ne jamais avancer de chiffre ici.**

### 6.6 Les tok/s : ce que le code fait vraiment

`thesis.rs:433-435` : `head_bytes = 389_070_848 × 2` ; `bw = f16_bytes / t16` (335,0 Go/s) ; `head_s = 2,3228 ms`. Puis, **pour les deux bras** : `total = t + head_s`. → FP16 24,014 ms = **41,64 tok/s** ; LLVQ 12,783 ms = **78,23 tok/s** ; **rapport 1,879**.

> **Arbitrage important, contre l'audit et contre le relevé « bras-q4 ».** Le grief « `bw` est le débit du bras FP16 alors que le bras LLVQ n'atteint que 239 Go/s » est **faux et ne doit pas partir dans un mail** : le `lm_head` est un tenseur f16 **non quantifié, identique dans les deux bras**, donc le débit f16 est le bon débit pour ce terme, et `head_s` est calculé **une fois** puis **ajouté aux deux** (vérifié dans la boucle du code). Ajouter une même constante **comprime** le rapport (2,07 → 1,88) : le traitement est **conservateur pour LLVQ**, pas optimiste.
>
> Les **vrais** défauts sont : (a) le lm_head n'est **jamais exécuté** — 2,32 ms est une extrapolation depuis un débit agrégé sur 252 dispatches de petites formes, appliquée à un unique dispatch de 778 Mo ; (b) tout le reste d'un pas de décodage est exclu ; (c) la constante `389_070_848` ne correspond à rien.
>
> Comme chaque terme exclu est ≥ 0 et qu'au moins un (les 144 rotations) n'est payé que par LLVQ : **41,6 et 78,2 sont des BORNES SUPÉRIEURES, et 1,88× est un MAJORANT du rapport de bout en bout.**

**Formulation défendable** : « projections seules, 2,07× ; en ajoutant analytiquement le lm_head f16 non quantifié (778 Mo au débit f16 de la machine, 2,32 ms — calculé, jamais exécuté), le rapport plafonne à **1,88×** ; 78,2 tok/s n'est le débit mesuré de rien. » Le README titre 2,07× sans jamais poser le 1,88× à côté, alors que `format-noyau.md` § « La thèse, sur le modèle entier » l'écrit correctement, dans le paragraphe qui suit sa table de tête.

### 6.7 L'échelle bits ↔ vitesse — **le point le plus attaquable, et il se répare**

| layout | b/poids (métrique **étroite**) | b/poids (métrique **large**) | vitesse | objet de la vitesse |
|---|---|---|---|---|
| Grouped32 (imbriqué) | **3,3548** — `bin/rtbits`, **exhaustif sur les 150 681 600 blocs du 4B publié** (6,5 s), corroboré par `decreal` sur 16,8 M blocs | 3,4982 → **1,589 Go** | 0,68× | **gate_proj seul** |
| Flat32 | 4,54 — *sur gate_proj* | 4,6779 → **2,125 Go** | 0,90× | **gate_proj seul** |
| Sorted32 | 4,75 | — | 1,04× | gate_proj seul |
| Fixed96 | 4,000 (structurel) | — | — | jamais en matvec |
| **Slot32** | **5,376** (modèle) / 5,375 (gate_proj) | **5,51** (modèle) / 5,554 (gate_proj) → **2,50 Go** | **2,07×** | **modèle entier** |

> **Correction majeure.** L'écart 5,51 vs 5,375 n'est **pas** un écart d'objet, c'est un écart de **métrique** : `RuntimeBlocks::bits_per_weight()` (payload + adressage / poids quantifiés) contre le calcul de `thesis` (payload + adressage + **queue f32** + **échelles de ligne f32** / **tous** les poids). Converti : le modèle entier vaut **5,376** contre 5,375 sur gate_proj — **identiques à 0,02 % près**.
>
> ✅ **Le reproche est soldé, et il reste écrit ici pour la généalogie.** L'explication publiée dans `format-noyau.md` § « Le prix en RAM, et que c'est un cadran » — « les autres formes ont d'autres distributions de classes et d'autres arrondis de stride » — **était** fausse. Elle **a été corrigée le 2026-08-05 par le lot K-1(a)** : cette section porte désormais le même diagnostic de comptabilité que le paragraphe ci-dessus, et l'appuie sur un recoupement par deux chemins de code indépendants — `bin/rtbits` rend **5,3756** sur le modèle **entier**, `bin/matvec` rend **5,375** sur **gate_proj seule**. Si l'écart tenait à la forme des matrices, ces deux-là ne coïncideraient pas.
>
> **Ce qui mélange réellement deux protocoles, c'est la colonne VITESSE** : 0,68× et 0,90× sont mesurés sur `gate_proj` par `bin/matvec` (32 dispatches, R=4 copies rotatives pour forcer le froid, surcoût **soustrait**, best-of-15) ; 2,07× est le modèle entier par `bin/thesis`. `thesis` ne compile que deux kernels — **Grouped32 et Flat32 n'ont jamais tourné sur le modèle entier.**
>
> Les 1,52 et 2,06 Go publiés omettent purement la queue f32 et les échelles f32 (+0,159 b/poids) → **1,589** et **2,125 Go**.

**Gisement immédiat** : la queue est chargée en **f32** par le noyau alors que le bras FP16 lit les mêmes colonnes en f16 — 67 829 760 o sur 2 502 446 285, soit **2,71 % du trafic LLVQ mesuré**. Queue en f16 → Slot32 tombe à **5,435 b/poids** et 2,435 Go. ~20 lignes.

### 6.8 « 335 Go/s ≈ 93 % du pic » — **RETRACTE**

Le « 93 % » appartient à un **autre banc** : `format-noyau.md` § « Le matvec fusé, et le layout qu'il a imposé » l'attache aux **370 Go/s de gate_proj** (`bin/matvec`), dans sa table et dans la phrase « le baseline est honnête » qui la suit, et 370/400 = 92,5 %. Il a été transplanté sur le 335-336 Go/s du modèle entier, où il vaut **83,8 %** — contre un pic constructeur de 400 Go/s qui est lui-même `SUPPOSE`. Conséquence : l'argument « le FP16 est au mur, il n'y a rien à en tirer » est plus faible qu'annoncé.

### 6.9 Transcodage au chargement

| chiffre | ce qu'il est |
|---|---|
| « ~3 s pour un 4B » (README:136) | **SUPPOSE** : 150 681 600 × 243 ns / 12 cœurs, en supposant une parallélisation **qui n'existe pas** (`llvq-artifact` est sans dépendance, `transcode()` est mono-thread, `thesis` ne lance aucun thread). Mono-cœur : **~37 s** (`CALCULE`) |
| « 128 s » (`load_s` de `thesis`) | **MESURE mal étiquetée** : le timer couvre aussi le dépaquetage de 981 Mo, la reconstruction f64 de 3,63 Md de poids, 3,63 Md de conversions f16, la **référence CPU f64** (~2,5·10¹⁰ opérations) et 7 uploads Metal par matrice |
| coût du transcodage **seul** | **`bin/decreal` le chronomètre**, sur 16 777 216 blocs réels et **deux** layouts (Fixed96 + Grouped32). Facteur d'échelle vers le modèle entier : ×8,98 **puis ÷2**. Pour Slot32 : **ABSENT** |

### 6.10 ~~Pourquoi le noyau n'est pas branché~~ — les deux obstacles, et comment ils sont tombés

> ✅ **Levés le 2026-08-05/06, sur CUDA.** Le cache KV existe (`bin/run`, commit `9c24d26`, épinglé par un test qui exige les mêmes tokens que le chemin non caché), et l'implémentation GPU de `Rotation` existe (`rot_apply`, vérifiée contre une référence f64 sur 8 formes, pire relatif 9,5e-8, 8,05 µs à n = 2560 en isolation — [`mesures/rotation-cuda-2026-08-05.txt`](mesures/rotation-cuda-2026-08-05.txt)). Le chemin fusé fait donc **deux lancements par projection** : rotation de l'activation, puis matvec fusé. Le diagnostic ci-dessous était juste — c'étaient bien les deux obstacles, et ce n'était pas de la plomberie. **Sur Metal ils sont tous les deux encore là** : aucun noyau de rotation, et `llvq-metal` n'a pas de runner.

Par ordre de dureté :
1. **`bin/run` n'a pas de cache KV.** `Qwen3::generate` re-exécute tout le préfixe à chaque token (le code le documente). Or `tv_slot` est un **matvec** ; le runner ne fait jamais de matvec, il fait toujours un GEMM sur tout le préfixe. **Le noyau n'a littéralement pas d'appelant.**
2. **La rotation.** Les codes vivent dans la base tournée ; le runner s'en sort en dé-tournant les **poids** au chargement (`decode_matrix`), ce qu'un noyau fusé interdit par construction. Il faudrait une implémentation GPU de `Rotation` (WHT + bloc dense k×k, k ∈ {1,5,19}), qui n'existe pas, et **144 applications par token**.
3. **Le prefill.** Il faut de toute façon un chemin dense pour seq > 1 : soit garder les poids déquantifiés (ce qui annule le gain RAM), soit écrire une variante GEMM du décodeur.

**Surface d'API à écrire** (`CALCULE`, estimation d'ingénierie) : (a) enum `Proj { Dense(Linear), Fused(LlvqLinear) }` en remplacement des 7 `Linear` de `Block` ; (b) `LlvqLinear` détenant les 9 buffers + pipeline + table de classes partagés ; (c) `CustomOp1::metal_fwd` encodant dans le **command buffer de candle** (pas une queue à soi, sinon l'ordonnancement casse) ; (d) `sealed::load` transcodant au lieu de décoder ; (e) la rotation GPU ; (f) le cache KV. **(a)–(d) ~2-3 jours, (f) ~1-2 jours, (e) non borné** — il faut d'abord décider si Q se fuse dans le noyau, se replie sur la norme précédente, ou se fait en dispatch séparé. À vérifier avant de citer : que candle 0.9 expose bien `MetalDevice::command_buffer()` et `CustomOp1::metal_fwd`.

**Un mono-verrou non signalé** : le shader `slot_dot` hard-code `uint gain = hdr >> 9` — **1 bit de gain**. `decreal` l'assert (`assert_eq!(m.centroids.len(), 2)`), **`thesis` non**. Un artefact à 2 bits de gain passerait en silence (l'assert 1e-3 le rattraperait — c'est un filet, pas une garde).

---

## 7. Ce qui manque

**Tout ce qui suit tourne en local et coûte 0 $**, sauf mention contraire. Ordonné par impact/coût.

| # | mesure | ce qu'elle changerait | commande | temps |
|---|---|---|---|---|
| **1** | **Le stdout des deux perplexités f16** | Le seul couple reproductible sur les octets publiés n'a pas de sortie relisible ; on ne peut pas relire l'empreinte qui prouve l'iso-conditions. C'est la commande que l'audit veut mettre dans LAUNCH_ME | `LLVQ_DTYPE=f16 cargo run --release -p llvq-llm --features metal --bin ppl -- 4096 12 metal /Users/pjmalandrino/qwen3-4b-llvq.bin 2>&1 \| tee ~/ppl-scelle-f16.log` puis `LLVQ_MODEL=Qwen/Qwen3-4B LLVQ_DTYPE=f16 … --bin ppl -- 4096 12 metal 2>&1 \| tee ~/ppl-base-f16.log`. **Critère : même empreinte `3f1baca9033bf251` des deux côtés.** Attendu 16,9415 et 12,2361 | **~10-15 min** (chargement du scellé < 150 s, mesuré) |
| **2** | **Le log de `bin/thesis`** | Le chiffre phare du gate G6 n'a **aucune** trace de rang supérieur à un `.md` | `cargo run --release -p llvq-metal --bin thesis -- /Users/pjmalandrino/qwen3-4b-llvq.bin 2>&1 \| tee ~/thesis-2026-08-03.log` | **~2-4 min** |
| **3** | **Infrastructure + corrections d'étiquette** | `Cargo.lock` commité, CI, URL de dépôt corrigée (`Cargo.toml:8` pointe sur `pjmalandrino/pjmalandrino`), table des 7 variables `LLVQ_*`, specs machine, `CITATION.cff`, et **les 6 renommages** : « Spherical GPTQ » → « Algorithme 1 + rotation d'entrée » ; le `+5,6 pp` de la rotation Output (§8) ; « 93 % du pic » ; « ~3,5 h » → 4,01 h ; « ±1 pp » → ±1,28/±1,36 et le mot « macro » ; la commande `ppl` cassée de LAUNCH_ME | — | **~3 h d'humain, 0 machine** |
| **4** | **Perplexité du q4 dans notre harnais** | Remplit la case décisive de la comparaison, **sans toucher au dépôt** (`bin/ppl` accepte un overlay) | déquantifier les 252 projections MLX vers un `.safetensors` aux noms `model.layers.{b}.{proj}.weight` (~25 lignes Python), puis `LLVQ_MODEL=Qwen/Qwen3-4B LLVQ_DTYPE=f16 … --bin ppl -- 4096 12 metal /chemin/q4-dequant.safetensors`. **Réserve à écrire** : l'overlay mesure « q4 sur les linéaires + embedding f16 », le bon comparateur de notre artefact — **pas** le fichier MLX dont l'embedding est quantifié | **~1,5 h** |
| **5** | **Corriger la commande de reproduction** (shard C4 + conteneur) | Un tiers qui relance obtient un autre fichier et une autre perplexité, sans avertissement. Deux sorties : documenter « la commande reproduit la méthode, pas les octets », ou ajouter un `LLVQ_CALIB_SHARD` | mesure de l'écart : 2 runs 3 blocs, un par shard (le shard n'est pas paramétrable aujourd'hui → ~10 lignes dans `corpus.rs`, ou checkout de `51d7c55` pour le bras shard 0) | **15 min** en documentaire seul ; **~50 min** avec la mesure |
| **6** | **`bin/seal` rejoué avec sa sortie** | Le 2,1595 de la carte HF n'est reconstitué qu'arithmétiquement ; le rejeu teste **aussi** la reproductibilité du build | `LLVQ_MODEL=Qwen/Qwen3-4B cargo run --release -p llvq-llm --bin seal -- /Users/pjmalandrino/llvq-q4b.llvq /tmp/reseal.bin` — attendu 2,1595 b/poids, 1,771 Go, ×4,54 ; **le fichier ne sera pas identique** (LVQ2 + deux u32) | **~10 min** |
| **7** | **`k=1` dans le banc de rétention** | La ligne de comparaison la plus importante du tableau adressé aux auteurs (« union + 1 bit de gain ») **n'est jamais mesurée chez nous** : elle est recopiée de leur Table 8. `main.rs:109` boucle sur `[0, 2]` | ajouter `1` à la boucle, puis `cargo run --release -p llvq-bench --bin llvq-bench 2>&1 \| tee ~/bench-retention.log` | **1 ligne + 4 s** |
| **8** | **MMLU du q4** | Le seul point de comparaison en capacité | ajouter à `mmlu.rs` une branche « chemin `.safetensors` », copiée de `ppl.rs` | ~30 min de code + **~47 min** de run |
| **9** | **Contrôle de déterminisme du pipeline** | Tranche la moitié de la question de la barre d'erreur : si deux runs identiques rendent le même chiffre, les 7 % sont une différence de configuration non consignée, pas du bruit numérique | relancer **deux fois** la commande de §2.2 **sans** `LLVQ_ARTIFACT` et comparer | **~8,5 h** (nuit) |
| **10** | **Barre d'erreur, à la profondeur qui compte** | Sans σ, la marge sur QTIP et le +3,06 % de nats sont illisibles. ⚠️ **Le protocole habituel (3 graines × 3 blocs) ne répond PAS à la question** : `LLVQ_CALIB_SEED` ne bouge que les offsets de fenêtre, et le mécanisme suspect vaut 0,04 % à 3 blocs contre 7 % à 36. Un σ mesuré sur 3 blocs serait **faux pour l'objet publié** | `for s in 1 2 3; do LLVQ_MODEL=Qwen/Qwen3-4B LLVQ_CALIB=c4 LLVQ_CALIB_SEED=$s cargo run --release -p llvq-llm --features metal,fast-linalg --bin smoke -- 64 2048 12 4096 metal nogs leech1c12 999 rot 2>&1 \| tee ~/seed-$s.log; done`. Sondage bon marché : le même sur Qwen3-0.6B à 28 blocs | **~12,6 h** locales · sondage 0.6B **~2,3 h** |
| **11** | **Débit du q4 et sa RSS, rejoués et loggués** | Convertit deux `SUSPECT` en `MESURE` et rend la ligne RAM opposable aux 17,41 Go mesurés | `/usr/bin/time -l mlx_lm.generate --model /Users/pjmalandrino/qwen3-4b-mlx-q4 --prompt "The capital of France is" --max-tokens 256 --temp 0 2>&1 \| tee ~/q4-generate.log` (×3) | **~2 min** (mlx_lm 0.24.0 déjà installé) |
| **12** | **A/B du plafond `L≤4`** | Le seul levier RAM **déjà entièrement câblé** (`smoke.rs` parse `L<n>`, `with_caps` l'applique) : 2,50 → 2,12 Go, et couplé à un lm_head quantifié, 2,34 Go contre les 2,39 de MLX. Gain **en RAM seulement** (le fichier paie quand même 48 bits). À décider sur MMLU, pas sur la perplexité | `… --bin smoke -- 64 2048 12 4096 metal nogs leech1c12L4 3 rot` contre le même sans `L4` | **~25 min** (A/B) ; **~5,5 h** (chaîne complète) |
| **13** | **Balayage d'amortissement** | Seul paramètre jamais touché, dans un dépôt qui balaye β au millième. Résultat nul attendu — **et un résultat nul est publiable**. À faire **après** le n° 10 | `for d in 3e-3 1e-2 3e-2; do … LLVQ_DAMPING=$d … --bin smoke -- 64 2048 12 4096 metal nogs leech1c12 3 rot; done` | **~1,3 h** |
| **14** | **Grouped32 / Flat32 sur le modèle entier** | L'échelle bits↔vitesse n'a jamais eu sa colonne vitesse mesurée sur un même objet | ajouter deux kernels tuilés à `thesis` (les shaders existent dans `matvec.rs`) | ~1 h de dev + 3 min |
| **15** | **Coût de la rotation d'activation en ligne** | Première question qu'un auteur posera ; seul terme exclu **asymétrique** | implémenter `Rotation` en Metal, puis 144 applications/token dans le bras LLVQ | **2-3 jours**, non chiffrable tant que le choix de fusion n'est pas fait |
| **16** | **Noyau branché + cache KV** | Seul chemin vers un tok/s opposable aux 129,8 de MLX. **Ne pas engager avant le n° 4** — la qualité du q4 peut rendre la course inutile | §6.10 | **semaines** ; attente honnête : ~110-120 tok/s projetés, **encore sous 129,8** |
| **17** | **CSR** | Troisième colonne de la Table 6. Métrique la plus **flatteuse** et la moins discriminante pour nous. **Bloqué en amont** : les notes de lecture ne transcrivent pas *quelles tâches* composent leur agrégat | étape 0 : relire la définition du CSR par **rendu image** (recette pymupdf) ; puis chargeurs ARC/PIQA/HellaSwag/WinoGrande + un `bin/csr` à scoring par **continuation** (mécanique différente de MMLU) | **1-2 jours** + runs |

**Ce qui exigerait un GPU loué** : rien dans cette liste. L'estimateur local `ops/run.py estimate Qwen/Qwen3-4B` donne ~2,8 h et **7,67 $/run** sur `rtx-pro-6000` (≈23 $ pour les trois graines du n° 10) — mais il ne modélise pas les passes avant, et le 4B tient largement en local. **Aucune dépense n'est nécessaire pour ce périmètre.**

> Le n° 1 seul, s'il ne faut faire qu'une chose : 15 minutes, et il transforme le point le plus inconfortable du dossier.

---

## 8. Points non tranchés

1. **La cause des 7 % de dispersion** (14,2684 vs 15,2909). Deux suspects laissés ouverts par le dépôt : divergence numérique amplifiée par la calibration séquentielle sur 36 blocs, ou différence de configuration non consignée. Le contrôle de déterminisme (§7 n° 9) en tranche la moitié.

2. **`--features fast-linalg` au run publié.** Non traçable : le garde-fou est postérieur, le log n'a pas de profil de phases. La feature existait au commit, la durée est compatible, rien ne le prouve. Aucune commande ne le récupère a posteriori.

3. **Le mécanisme du pic RAM Metal à 17,41 Go.** Le nombre est mesuré et reproductible ; la cause (double résidence hôte/buffer en mémoire unifiée, ou pool de tampons candle-metal) est `SUPPOSE`. Réserve honnête : en mémoire unifiée, la RSS peut ne pas comptabiliser les `MTLBuffer` de la même façon selon le chemin.

4. **L'heure de fin du run de nuit** : mtime du log 02:30, `docs/archive/retraction-et-gain.md:85` dit 04:26. Sans conséquence — dans les deux cas le démarrage précède le correctif `60068db` (01:12).

5. **`format-noyau.md` § « Le matvec fusé, et le layout qu'il a imposé » ne boucle pas**, dans le paragraphe sur `Layout::Flat32` : « 4,54 b/poids sur cette couche (16,5 Mo …) ». 4,54 en métrique large donne **14,8 Mo** ; 16,5 Mo implique 5,12 b/poids étroits. L'un des deux est périmé (la phrase et la table datent de sessions différentes). À trancher avant qu'un lecteur le fasse.

6. **Le gain incrémental de la rotation de sortie.** La Table 9 du papier chiffre **29,3 → 34,9 = aucune rotation → Input+Output**, pas *Input seule → Input+Output*. Notre configuration a **déjà** l'étage Input. Le gain incrémental de l'étage Output **n'est chiffré nulle part dans le papier**, et la seule ligne « Input seule » disponible (spherical shaping, 24,0 → 35,1) suggère que l'étage Input capte l'essentiel. **La phrase « +5,6 pp, plus que notre déficit de 4,8 pp » doit être retirée du README et de l'audit §Q4.** Le levier reste plausible ; son ampleur est inconnue. Note technique rassurante : l'objectif GPTQ est invariant par mélange orthogonal des lignes, donc la machinerie hessienne ne bouge pas — seule l'isotropie vue par un code 24-dimensionnel change. Côté code, **rien n'existe** (`rotation.rs` n'a que `rotate_weight_rows`), et l'ajouter impose un bump de MAGIC, ce qui entre en collision avec le travail LVQ3 non commité.

7. **Ce que coûterait `gain_bits = 0`** — la vraie ligne du papier (une constante par tenseur). L'A/B `~/llvq-ab-retraction.log` mesure `1 bit` contre `magnitude libre`, pas contre `0 bit`. Inconnu.

8. **Comptage des tests.** 129 fonctions `#[test]` à HEAD ; 142 dans l'arbre de travail ; LAUNCH_ME annonce 106 ; l'audit rapporte 128/128 **cas exécutés** en release (les `#[cfg_attr(debug_assertions, ignore)]` font diverger fonctions et cas). Les trois sont différents et aucun n'est faux — il manque la définition. Piège : un `grep -r` naïf depuis la racine en compte 249 parce qu'il descend dans `target/`.

9. **Rétention 92,01 vs 92,14 — RÉSOLU, mais la correction doit être appliquée.** `retention_pct(mse, rate) = 100·(−½·log₂ mse)/rate`. Le banc reçoit la MSE **arrondie** que le papier affiche (0,078) → 92,0096 → il imprime **92,01**. Le **92,14** est la valeur imprimée par le **papier**, calculée sur sa MSE non arrondie (≈0,077718, SQNR 1,843). **Les deux sont justes.** À faire : citer 92,14 comme chiffre du papier **avec son SQNR 1,843**, et cesser de le recalculer depuis 0,078.

10. **Aucun chiffre de qualité pour `q4b-e4.llvq` / `q4b-e8.llvq`** — ni ppl, ni MMLU. Et ces fichiers sont en `LVQ3`, format que le dépôt à HEAD ne lit pas.

11. **Un pré-requis à la reproduction par un tiers, non mesuré** : `calib.rs` accumule `AᵀA` en **f32 sur l'accélérateur**. Un tiers sur CUDA n'obtiendra donc **pas** les mêmes poids, là où l'encodeur Leech est exactement déterministe. Aucune surface ne le dit, et l'écart n'est pas chiffré.

12. **La constante `389_070_848`** (`thesis.rs:432`) : origine inconnue, ne se factorise par aucune dimension du modèle (= 2¹⁴ × 23 747, 23 747 premier). Effet 0,03 %. À corriger, pas à expliquer.

---

### Annexe — le seul paragraphe qu'on peut écrire sans réserve dans un mail

> Qwen3-4B, 252 projections quantifiées sur le réseau de Leech Λ₂₄ plafonné à la coquille 12 : **47 bits d'index + 1 bit de gain = 48 bits par bloc de 24 poids, soit exactement 2,000 bit/poids de code**. Le fichier livré pèse **2,1595 b/poids** sur ses 3 633 315 840 poids de projection (2,1696 hors queue au dénominateur), l'excès de 8 % étant entièrement de la sérialisation : queue `KeepExact` en f32 (+0,150) et échelles de ligne en f64 (+0,020, dont aucune n'est représentable en f32 — c'est le prix d'une preuve de décodage bit-pour-bit sur les 3 633 315 840 poids). Modèle entier, embedding f16 compris : **1,771 Go contre 8,045 Go en FP16, ×4,54**.
> **Perplexité** wikitext-2, ctx 4096, 12 fenêtres, f16, sur les octets publiés : **16,9415 contre une baseline f16 de 12,2361, ×1,385** — soit **+3,1 % de surcoût de log-vraisemblance par rapport à QTIP** sur sa propre baseline. **MMLU** 5-shot, micro pondéré par population, 2 280 questions sur 14 042 : **56,09 ± 1,36 contre 70,42 ± 1,28**, une chute de **14,3 pp** là où le papier annonce 9,5. Le harnais reproduit la baseline du papier à **0,22 pp (0,17 σ)**, donc le déficit n'est pas imputable au harnais ; la cause la plus probable reste le volume de calibration (131 072 tokens contre ~100× plus).
> **Le projet n'a pas de barre d'erreur sur ce chiffre-là.** Une dispersion existe depuis le 2026-08-06, mais sur un autre objet : trois graines de calibration sur un run de 3 blocs de Qwen3-0.6B donnent **σ ≈ 0,15 de perplexité (≈ 0,7 %)** autour de ~20,66. Elle ne se transfère pas au 16,9415 — autre modèle, 3 blocs contre 36, autre échelle — et **aucun σ n'a jamais été mesuré sur le chiffre publié**. L'observation plus ancienne tient toujours : ~7 % sur deux configurations dont un test démontre qu'elles étaient le même quantifieur, cause non tranchée. Aucune marge inférieure à cela n'est revendiquée.
> 🆕 **AMENDÉ le 2026-08-17, et il faut lire exactement CE QUI a reçu une barre.** Ce n'est pas la perplexité publiée, c'est la **dégradation** : sur la campagne 4B du 2026-08-06 (f16 des deux côtés, empreinte `3f1baca9033bf251`, 12 fenêtres), l'excès de LLVQ sur f16 vaut **+38,45 %, IC95 [+33,62 ; +43,45]**, intervalle t **apparié fenêtre par fenêtre** — 12 fenêtres sur 12 dans le même sens ([`mesures/ppl-appariee-4b-2026-08-17.txt`](mesures/ppl-appariee-4b-2026-08-17.txt), *calculé* sur des NLL déjà mesurées, 0 $). ⚠️ **Cela ne contredit pas le paragraphe ci-dessus, cela en précise la portée** : cette barre ne porte **que** l'échantillonnage du corpus d'évaluation. Le **tirage de calibration** en est toujours absent, et l'y ajouter en empruntant le σ de 0,7 % serait fabriquer un nombre. **« Aucun σ sur le chiffre publié » reste donc vrai au sens qui compte** — le fichier livré est *un* tirage d'un processus à graine, et ce que vaudrait un autre tirage n'est pas mesuré.
> **Noyau fusé**, 252 projections, un token, batch 1, M3 Max, mémoire froide par construction : `Slot32` à **5,510 b/poids en RAM**, **2,03× à 2,09× le FP16 selon l'invocation** — 7 rounds dont 2 jetés, les sept bras dispatchés à chaque round dans le même ordre, le rapport formé **round par round** puis résumé par sa médiane et sa plage (ce n'est pas un quotient de deux minima ; les millisecondes sont dans le journal `docs/mesures/k1-metal-2026-08-05.txt` et dérivent d'un run à l'autre, là où les b/poids reproduisent au chiffre ; trois invocations du banc non modifié rendent 2,03× · 2,06× · 2,09×, cf. `docs/mesures/thesis-temoin-2026-08-04.txt` — **une valeur ponctuelle n'a pas de contenu ici**). **1 105 920 lignes de sortie vérifiées contre une référence CPU f64 avant toute mesure, pire erreur 3,4·10⁻⁸·Σ|w·x|**. Hors attention, normes, lm_head, échantillonnage. En y ajoutant analytiquement le lm_head f16, le rapport de bout en bout plafonne à **1,88×**.
> **Depuis le 2026-08-06, le noyau est branché — sur CUDA.** Le layout de référence n'est plus `Slot32` mais **`Planes14`** (plans de bits binaires, 4,804 b/poids, **1,14× plus rapide que `Slot32` à contenu décodé identique**, 2,14× le FP16 sur L40S). Dans le modèle, sur les octets publiés, 128 tokens : **48,7 tok/s dans 2,96 Go de carte contre 43,6 dans 8,04**, mêmes tokens gloutons jusqu'à un tie-break au token 89. Avec l'embedding int8 au chargement : **88,4-88,5 tok/s dans 2,60 Go**. ⚠️ **Le ×2,03 de ce dernier chiffre n'est pas le noyau Leech** — ~25 ms/token viennent du remplacement de **notre propre** chemin `lm_head` dense (`Head::project` → `broadcast_matmul`), qui recopie 778 Mo de vocabulaire par token ; les modèles de `candle_transformers` passent par `Linear` et ne paient pas cette copie ([candle#3871](https://github.com/huggingface/candle/issues/3871)). **Le rapport à tête identique est ×1,12**, et c'est celui qui mesure le noyau. Les deux se citent ensemble. La **rotation d'entrée**, que seul le bras quantifié paie (144 transformées par token), n'est plus « latence non mesurée » : `rot_apply` existe sur CUDA, est vérifiée contre une référence f64 sur 8 formes, et le bout-en-bout ci-dessus la paie déjà. **Sur Metal, rien n'est branché** : `bin/run` décode toujours en mémoire, sur CPU comme sur Metal.
> **La recette livrée est l'Algorithme 1 (shape–gain, reset de gain) plus une rotation d'incohérence en entrée** — pas du Spherical GPTQ : avec un code de gain fini, la rétraction de l'Eq. 17 est un no-op, et le raffinement d'échelles de l'Algorithme 3 est désactivé.
