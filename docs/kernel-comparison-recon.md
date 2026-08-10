# Lot 0 — reconnaissance des noyaux concurrents (2026-08-10)

> Note de reconnaissance pour la campagne de comparaison kernel. Un verdict par
> concurrent, ses sources, ce qu'il impose, et ce qu'il coûte. **Aucune mesure
> n'a été faite ici** — c'est une étude de faisabilité, et son seul rôle est de
> décider quels lots existent.
>
> ⚠️ Le nom du fichier est en anglais et neutre à dessein (contrainte
> d'anonymat de la campagne). Le contenu suit la convention du dépôt : docs en
> français, code et commentaires en anglais.

## 0. Ce que cette reconnaissance a changé au plan

Trois découvertes, chacune invalidant une hypothèse de la spec de campagne.

1. **Le bras mono-shell n'est pas iso-débit.** `|Shell(3)| = 16 773 120`, soit
   **exactement 1,000 bit/poids** et **4 classes** contre nos 301. La table des
   issues pré-enregistrées de la spec supposait une course à débit apparié ;
   elle n'en est pas une. → §1.
2. 🚨 **Les chiffres publiés du banc dépendent de son jeu de bras.** Trois
   invocations de `planesbench`, même carte et même modèle, donnent pour
   `Planes14` des plages **disjointes** entre 4 et 5 bras. Ajouter un
   concurrent n'est donc pas gratuit : ça re-dérive les cinq lignes de la
   Table 1. → §4.
3. **L'axe horizontal de la Figure 1 est mal étiqueté.**
   `docs/data/echelle-formats.csv` nomme sa colonne `bpw_payload` et
   `paper/sections/layouts.tex:36` dit « Payload rates » ; les valeurs sont la
   comptabilité **noyau**. → §5.

---

## 1. Concurrent n°1 — le noyau mono-shell (M = 3) des auteurs de LLVQ

**Verdict : faisable, mais uniquement en réimplémentation clean-room.**

### Le code n'est pas publié

Recherche refaite depuis un réseau non bloqué (la précédente, notée
`README.md:411-425`, portait la réserve d'un réseau qui bloquait arXiv et HF).
Résultat identique, et la réserve peut être retirée :

| Endroit vérifié | Résultat |
|---|---|
| `arxiv.org/abs/2603.11021` | aucun lien code, aucun « Code available at » |
| `arxiv.org/html/2603.11021v1` | ni « our code », ni « will be released », ni URL GitHub ⚠️ le HTML v1 ne contient pas l'Annexe C ; nos notes viennent de la **v2** par rendu image |
| page ICML `icml.cc/virtual/2026/75287` | un seul lien sortant, vers OpenReview |
| `github.com/orgs/Qualcomm-AI-research/repositories` | 48 dépôts énumérés, **rien** en leech / llvq / lattice ; les plus proches par sujet sont `gptvq` et `fastforward` |
| recherches larges (`"llvq" github`, `"leech lattice" quantization CUDA kernel`, `"2603.11021" code`) | seulement arXiv / ICML / agrégateurs |

**Un seul angle mort subsiste** : un supplément éventuellement attaché à
`openreview.net/forum?id=HFk5TQ2ILj`, derrière un mur de vérification de
navigateur. Non vérifiable par outil.

> 💡 Le chemin le moins cher vers l'Annexe C reste `docs/mail-qualcomm-draft.md`,
> **rédigé et jamais envoyé**, qui pose déjà la question. Décision et action de
> l'utilisateur.

### Ce que « M = 3 » veut dire, chiffré

Dans le plongement entier `√8·Λ₂₄ ⊂ Z²⁴` du dépôt, la coquille `m` a
`‖x‖² = 16m`.

| | Shell(3) | boule Λ₂₄(12), notre format |
|---|---|---|
| cardinal | **16 773 120** (`llvq-core/src/leech.rs:44`) | 111 043 117 458 000 |
| bits d'index par bloc de 24 poids | **24,0** | 47 |
| **b/poids de payload** | **1,000** | 2,000 (47 + 1 de gain) |
| classes de permutation | **4** | **301** |
| niveaux distincts max (`L`) | **3** | 5 |

Les quatre classes de Shell(3) sont exhaustives, pas estimées :
`(±2¹²)` sur dodécades · `(±4, ±2⁸)` sur octade · `(±5, ±1²³)` ·
`(±3³, ±1²¹)`, et `(±4³)`, `(±4², ±2⁴)`, `(±6, ±2³)` sont **prouvées vides**
(`llvq-core/tests/g1_invariants.rs:445-566`).

Leur Table 7 (Annexe C) donne 11,94 µs contre 16,3 / 17,69 µs FP16, soit
1,36×/1,48× — **à 1 bit/poids**, sur un GPU datacenter d'une autre génération.

### Pourquoi la course brute ne veut rien dire, et quelle est la bonne question

Sur un noyau memory-bound, celui qui lit moins d'octets gagne. Un bras `M = 3`
bat `Planes14` par construction, et le perdre n'apprendrait rien.

La question bien posée est celle que la Figure 1 pose déjà : **le décodage
multi-coquilles atteint-il les mêmes Go/s ?** On connaît la forme de la
réponse — `Slot32` → `Planes14` tient 428 → 425 Go/s, `Golay70` s'écroule à
195. Et le papier **affirme déjà la réponse sans la mesurer**
(`paper/sections/decoder.tex:66-71`) : le rééchelonnage inter-coquilles que
l'Annexe G des auteurs présente comme le coût matériel du multi-shell « se
réduit à une multiplication par bloc ». Le code le confirme —
`llvq-llm/src/fused_cuda.rs:236-252` replie `1/√(16·shell)` dans les valeurs
flottantes de la table de classes **au téléversement**, donc zéro instruction
à l'exécution.

**D'où deux bras, pas un :**

| bras | index | classes | taux VRAM | ce qu'il mesure |
|---|---|---|---|---|
| `MonoShell3` | 24 bits | 4 | **3,33 à 3,67 b/poids** *(calculé)* | « leur noyau tel que publié », un point de plus sur la courbe débit↔taux |
| `Shell12` | 47 + 1 bits | **79** | **identique à `Planes14`** | ce que coûtent les 222 classes supplémentaires, à taux apparié |

- `MonoShell3` : ses 4 classes ont `L ≤ 3`, donc **2 plans binaires** suffisent.
  Record = `classe + 1 gain + 24 signes + 2×24 plans` = **10 à 11 o** selon
  qu'on garde le champ classe à 9 bits ou qu'on le réduit à 2.
  ⚠️ **Pas 1,0 b/poids en VRAM** — le payload disque et le taux noyau sont deux
  comptabilités (§5). Il atterrit **juste à gauche de `Golay70`** (3,589).
- `Shell12` : le record `Planes14` fait `9 + 1 + 24 + 3×24 = 106 bits = 14 o`,
  et ses trois plans nomment jusqu'à 8 niveaux. Un jeu de classes à `L ≤ 5` y
  entre **sans changer un octet**. Taux VRAM identique **par construction**.
  Qualité déjà mesurée, gratuitement : **90,34 % de rétention contre 92,14 %**
  pour la boule (CLAUDE.md §6).

### Ce que ça coûte, et le piège

⚠️ **`Planes12x` et `Golay70` étaient bon marché parce que ce sont des
bijections du même contenu** — mêmes classes, mêmes gains, mêmes signes,
re-disposés. Un bras mono-shell est un **contenu différent** : il faut
**ré-encoder** chaque bloc.

Les primitives existent :

- `Searcher::nearest_in_shell3` (`llvq-search/src/lib.rs:418`) — exactement
  Shell(3), déjà écrit ;
- `BallSearcher::shell_bests` (`llvq-search/src/generic.rs:537`) rend déjà le
  meilleur point **par coquille** ;
- **la seule primitive manquante** : un *plancher* de coquille.
  `set_shell_cap` (`generic.rs:520-533`) est une **boule** `2..=cap`.

→ Le ré-encodage se fait **localement sur le M3 Max**, pas sur la carte à
0,03 $/min : ~30 min par bras (4 ou 79 classes au lieu de 301). Les flux
transcodés partent au bucket une fois.

**Licence** : rien à vérifier, il n'y a rien à réutiliser. Le précédent du
dépôt est Adoul & Barth (1988), re-dérivé faute d'être obtenu (CLAUDE.md §1).

---

## 2. Concurrent n°2 — QTIP

**Verdict : faisable en stratégie A (même process, mêmes rounds, même
horloge), avec quatre réserves à déclarer avant de mesurer.**

Dépôt `github.com/Cornell-RelaxML/qtip`, HEAD `e90c668` (2025-06-21).

### La stratégie A est prouvée possible par leur propre dépôt

Le noyau fusé est **un seul fichier de 472 lignes**,
`qtip-kernels/src/inference.cu` : décodage par LUT en mémoire partagée
(`:339-359`) alimentant directement un `mma.sync.aligned.m16n8k16` (`:364-373`),
**aucun poids matérialisé** — structurellement la même revendication que
`Planes14`.

Le couplage torch est **isolé** dans deux autres fichiers (`qtip_torch.cu`,
`wrapper.cpp`), et surtout : **un pilote sans torch est livré en amont** —
`qtip-kernels/src/test.cu` appelle `decompress_matvec_ptr<...>` avec des
pointeurs `cudaMalloc` et un stream `NULL`, et le `Makefile:8-13` le construit
en `nvcc` nu, `-gencode arch=compute_89` inclus.

Notre côté a déjà les motifs d'intégration : `gpu.rs` compile en `compute_89`
et relit la version binaire (`:177-184`), la mémoire partagée dynamique est
déjà utilisée (`:627-630`), et `matvec.cu:14-19` montre le motif « l'hôte
injecte un `#define` » qui sert à dé-templater.

⚠️ **Le vrai travail** : nous compilons par **NVRTC**, eux par **nvcc**.
`inference.cu` tire `<mma.h>` et `<cuda/pipeline>`, absents de notre image sans
paquet `-dev` — le même mur que `matvec.cu:22-30` documente pour
`__half2float`, franchi par du PTX en ligne. Il faut retirer les includes,
dé-templater les sept paramètres, et ajouter les instanciations des formes
Qwen3. Mécanique et borné, mais ce n'est pas « brancher un fichier ».

### Les quatre réserves

1. 🚨 **Le noyau ne couvre pas la configuration dont notre papier cite la
   qualité.** La seule famille implémentée en CUDA est **HYB
   (`quantlut_sym`)** ; 3INST et 1MAD n'existent qu'en Python
   (`lib/utils/kernel_check.py:1-15`), alors que ce sont la nouveauté affichée
   du papier. Or `paper/sections/evaluation.tex:127` cite « QTIP (3INST) ».
   **Vitesse et qualité viendraient de deux configurations différentes** — à
   écrire en toutes lettres, sinon c'est le mélange de comptabilités que le §7
   de CLAUDE.md interdit.
2. **C'est un GEMV, `N = 1` en dur** (`inference.cu:462`,
   `static_assert(N == 1)`) ; côté Python, `bitshift.py:443` ne prend le chemin
   noyau que `if bs == 1`. QTIP **ne peut pas entrer dans un balayage de
   batch**. Symétrique avec notre matvec, donc loyal à M = 1 — mais la table du
   balayage aura une case vide et sa légende devra dire pourquoi.
3. **Chaque forme est une instanciation de template écrite à la main** : 66 en
   amont (`lib/codebook/__init__.py:4-66`), **aucune ne correspond à Qwen3**.
   Les cinq formes du 4B satisfont les contraintes en dur
   (`M % 32 == 0` via `inference.cu:194`, `K % 64 == 0` via `:202`) et
   compileraient ; il faut les instancier.
4. **Licence GPL-3.** Mesurer n'oblige à rien — les obligations se déclenchent
   à la **distribution**. Vendoriser leur `.cu` contaminerait un workspace
   `MIT OR Apache-2.0` (`Cargo.toml:16`). **Motif sûr, et le mécanisme existe
   déjà** : garder le `.cu` patché **hors du dépôt** et le charger par
   `LLVQ_KERNEL_DIR` (`llvq-cuda/src/lib.rs:127+`), en ne commitant qu'un
   script de récupération et le patch. ⚠️ Ça sacrifie la propriété
   « reproductible depuis le binaire seul » que le `include_str!` par défaut
   garantit. C'est une décision, pas une formalité.

### La transformée de Hadamard

QTIP l'applique **à l'exécution**, en entrée et en sortie, par token et par
couche (`lib/codebook/bitshift.py:429-441` et `:465-471`).

Notre banc n'applique **aucune** rotation : la nôtre vit dans le modèle, 144
lancements par token (`llvq-cuda/src/bin/rotbench.rs:1-18`). **L'exclusion est
donc symétrique** et défendable — à condition de l'écrire.

*(Détail utile ailleurs : `lib/utils/matmul_had.py:59-61` assert `is_pow2(n)`
hors d'une petite liste, et `9728 = 2⁹·19` n'y est pas. Ça interdirait un bras
QTIP **bout en bout** sur Qwen3-4B, mais pas un banc matvec.)*

### Checkpoints

**Tous les checkpoints QTIP publics sont des Llama** (collection `relaxml`,
27 modèles). Un seul Qwen3 tiers existe, `Hschen335/Qwen3-1.7b-QTIP-4Bit`,
tagué MIT — étiquette douteuse pour un dérivé QTIP, à ne pas utiliser.

→ **La qualité de QTIP se cite, elle ne se remesure pas** : `tab:lit`
(`evaluation.tex:127`) porte déjà 17,04 (×1,37) au 4B et 11,17 (×1,24) au 8B,
sous la discipline rapport-à-sa-propre-baseline.

Note : QTIP n'est **pas** dans vLLM ; la demande
`vllm-project/vllm#11416` est close *not planned*. Le successeur
`Cornell-RelaxML/yaqa-quantization` embarque le **même** `qtip-kernels/`.

---

## 3. Concurrent n°3 — AWQ 4 bits et la famille Marlin

**Verdict : faisable en stratégie A. Mais ce n'est pas un bras, c'en est deux,
et leur place est dans un balayage de batch.**

### Quel noyau, et ce n'est plus une supposition

La chaîne a été lue dans le source de vLLM, pas devinée — c'est ce que
`docs/portage-noyau-cuda.md:721` et `docs/experience-mesure.md:229-233`
exigeaient tous les deux :

- `auto_awq.py:260-284` : toute config HF `quant_method == "awq"` est prise en
  charge par `AutoAWQConfig` ;
- `auto_awq.py:306-331` : `use_marlin` si `check_marlin_supported(uint4,
  group_size, zero_point)` ;
- `marlin_utils.py:80-82` : avec `has_zp = True`, le type supporté est
  exactement `uint4` — **les zero-points AWQ ne poussent pas hors de Marlin** ;
- `linear/__init__.py:449-458` : l'ordre de priorité CUDA est
  `CutlassW4A8 (cap 90) → Machete (90) → AllSpark (80, refuse les zero-points)
  → **Marlin (75)** → …`.

→ **Sur SM89 avec `zero_point: true`, le premier noyau éligible est Marlin.**
Machete et Cutlass sont Hopper-only par construction (`wgmma`).

### Contraintes, et la L40S passe

| contrainte | valeur | verdict Qwen3 |
|---|---|---|
| capacité minimale | `≥ 75` (`marlin.cu:405`), `CMakeLists.txt` nomme `8.9` | ✅ L40S = SM89 |
| tailles de groupe | `[-1, 32, 64, 128]` (`marlin_utils.py:36`) | ✅ AWQ Qwen3 = 128 |
| formes | `N % 64 == 0`, `K % 128 == 0` (`marlin_utils.py:173-205`) | ✅ **sans padding** : 4B 2560/9728 · 8B 4096/12288 · 14B 5120/17408 |
| dtypes activations | f16 ou bf16 (`auto_awq.py:226-227`) | ✅ |

### 🚨 Le point décisif : Marlin est une GEMM, sans chemin gemv

`marlin.cu:423-438` : la plus petite tuile en M est **8**
(`m_block_size_8` pour `M ≤ 8` en activations 16 bits). Il n'y a pas de gemv.

À `M = 1`, tous les noyaux 4 bits convergent vers la même borne de bande
passante — là où vit `Planes14`. **Un point unique à M = 1 ne peut pas les
séparer**, et le citer seul serait la faute « un point, pas une plage ». Leur
propre README le concède : le gain quasi idéal tient « jusqu'à des batchs de
16-32 tokens, **contrairement aux 1-2 tokens des travaux antérieurs à gain
comparable** » — c'est-à-dire qu'à M = 1 les noyaux antérieurs font aussi bien.

**Conséquences :**

1. La place honnête du 4 bits est le **balayage de batch**, pas une ligne à
   M = 1. Opposer notre matvec à une GEMM à M = 32 sans le dire serait la même
   erreur de catégorie que « 5,51 contre 4,50 ».
2. **Le vrai bras M = 1 n'est pas Marlin**, c'est `llm-awq/gemv_cuda.cu`
   (MIT, `mit-han-lab/llm-awq`, dépôt vivant, dernier push 2025-07-17). Deux
   bras 4 bits, donc.

### Licences et sources

| artefact | licence | vendorisable ? |
|---|---|---|
| `IST-DASLab/marlin` — `marlin_cuda_kernel.cu` | Apache-2.0 | ✅ un seul `.cu`, « no dependencies beyond base-CUDA » — mais **symétrique, sans zero-point** : il ne lit pas un checkpoint AWQ |
| fork vLLM (`csrc/.../marlin/marlin.cu`) | Apache-2.0, en-tête composite Neural Magic / Elias Frantar | ⚠️ le seul qui gère les zero-points, mais couplé à torch |
| `mit-han-lab/llm-awq` — `gemv_cuda.cu`, `gemm_cuda_gen.cu` | MIT | ✅ sans friction |
| `AutoAWQ` / `AutoAWQ_kernels` | MIT, **archivés** (2025-05-11 / 2024-11-26) | ❌ ne pas vendoriser du code mort — c'est déjà le constat de `ops/awq_dequant.py:30-32` |

Notre workspace est `MIT OR Apache-2.0` (`Cargo.toml:16`) : Apache-2.0 est
compatible, à condition que l'attribution voyage avec le fichier.

Checkpoints publics : `Qwen/Qwen3-{4B,8B,14B,32B}-AWQ`.

### Ne pas confondre avec ce qui existe déjà

`llvq-llm/src/bin/gbench.rs` produit déjà un débit pour « les poids AWQ
**déquantifiés** dans notre moteur ». Ce **n'est pas** le noyau AWQ, et ça ne
comble pas le trou de `paper/sections/limitations.tex:63-65`.

---

## 4. 🚨 Le risque qui domine tous les autres : le jeu de bras déplace les chiffres publiés

Trois invocations de `planesbench` existent — **même carte, même fichier
modèle, même protocole** — et seul le nombre de bras change :

| journal | bras | source NVRTC | Slot32 | **Planes14** | Planes12x |
|---|---|---|---|---|---|
| `mesures/c1-planesbench-2026-08-06.txt:23-29` | 3 | 28 710 o | 1,89× [1,89–1,89] | **2,16× [2,16–2,16]** | — |
| `mesures/nuit-planes12x-q8-2026-08-07.txt:25-32` | 4 | 40 826 o | 1,89× [1,89–1,89] | **2,16× [2,16–2,16]** | 2,01× [2,01–2,01] |
| `mesures/e2-golay70-bench-2026-08-07.txt:27-36` **(publié)** | 5 | 55 949 o | 1,87× [1,86–1,88] | **2,14× [2,11–2,15]** | 1,98× [1,95–1,99] |

**Les plages sont DISJOINTES** entre 4 et 5 bras : [2,16–2,16] contre
[2,11–2,15]. Les temps absolus bougent aussi (FP16 médian 10,990 → 10,996 →
**11,025 ms**). Les plages imprimées sont des dispersions **intra-run** ; elles
ne sont pas une enveloppe d'incertitude du nombre.

**Deux précisions, parce qu'un effet mal nommé est pire qu'un effet ignoré :**

- **Ce n'est pas monotone en nombre de bras.** Le passage de 3 à 4 ne déplace
  *rien*. C'est celui de 4 à 5 qui déplace tout. Ce qui est établi, c'est que
  le **jeu** de bras change le résultat au-delà des plages imprimées — pas
  qu'il existe une loi en N.
- **Le run à 5 bras est surtout plus bruyant** : la dispersion intra-run de
  `Planes14` passe de 0,005 ms (5,091–5,096) à **0,107 ms** (5,129–5,236), ×20.
  L'hypothèse la moins chère est la pression VRAM — le banc annonce lui-même
  ~15 Go pour cinq bras (`planesbench.rs:44-47`). À vérifier, pas à supposer.
- ✅ **Les octets, eux, ne bougent pas d'un chiffre** : 16,000 / 5,510 / 4,804
  dans les trois runs.

⚠️ **Le détecteur sur lequel on aurait compté échoue en silence** : le rapport
registres/spill est **identique** dans les trois runs (`tv_slot 40`,
`tv_f16 42`, `tv_planes 40`, 0 o locaux). Zéro spill, et les chiffres bougent.

C'est la règle n°2 de CLAUDE.md §7 (« une plage, pas un point »), déjà payée
une fois sur Metal (2,029 / 2,050 / 2,080 sur le *même binaire non modifié*) —
mais personne n'avait vérifié que la table CUDA y était exposée. Elle l'est.

**Ce que la campagne doit en faire :**

1. Le 2,14× de `paper/main.tex:46-49`, `layouts.tex:29` et
   `docs/data/echelle-formats.csv:4` est **périmé à l'instant où un sixième
   bras entre**. Le coût d'un bras n'est pas 0,05 $ de transcodage, c'est une
   re-publication de la Table 1, de la Figure 1 et de l'abstract. → à faire
   **une seule fois, au bout**, depuis le run final portant tous les bras.
2. **Un verdict de parité n'est pas décidable à cette résolution** sans seuil
   posé d'avance : un décalage de 1 à 1,5 % lié au jeu de bras tient dans
   toute marge de parité plausible.
3. **La parade est bon marché et se pré-enregistre** : dans **le même job**,
   lancer le banc à N bras **et** un contrôle à 5 bras, publier le delta sur
   les bras incumbents, et ne citer le rapport d'un concurrent **que contre le
   jeu de bras qui l'a produit**.

> 🔎 C'est aussi un résultat de méthode gratuit. `limitations.tex:67-70` dit
> déjà « Speedups carry ranges, not third decimals » en s'appuyant sur la
> dispersion inter-processus ; ceci en fait un effet **nommé et mesuré à trois
> points**.

---

## 5. Comptabilités d'octets : quatre divergences à solder avant d'ajouter un bras

La spec pose que *« toute divergence de convention invalide l'axe horizontal de
la Figure 1 »*. Il est **déjà divergent** :

1. **Le bras FP16 ne facture que `d_out·d_in·2`** (`planesbench.rs:754`) : sa
   queue est à 16 bits pendant que tous les bras LLVQ facturent la leur à 32
   (`:662-664, 686, 716, 749`). Ce **n'est pas un bug** — les noyaux de banc
   lisent vraiment une queue f32 (`gpu.rs:498-506`, « deux résidences, deux
   comptabilités ») — et ça joue **contre nous**. Mais ce n'est écrit nulle
   part. → documenter en légende.
2. **`Golay70` facture son padding de flux** (`llvq-artifact/src/runtime.rs:1546`)
   là où les trois autres le refusent explicitement (`planesbench.rs:676-682`).
3. **Les tables de classes ne sont facturées par personne** : `d_tab` (12 Kio,
   partagée) et les `d_gtab`/`d_cw` de Golay70 (16 Kio de codewords,
   `planesbench.rs:505-527`). Négligeable en valeur, mais Golay70 lit deux
   tables que les autres n'ont pas. → documenter.
4. 🚨 **La colonne du CSV publié est mal étiquetée, et le papier reprend
   l'étiquette.** `docs/data/echelle-formats.csv` nomme sa colonne
   `bpw_payload` et `layouts.tex:36` dit « Payload rates » — or 5,510 / 4,804 /
   4,342 / 3,589 sont la comptabilité **noyau**. `llvq-bench/src/bin/rtbits.rs:63-85`
   sépare les deux en toutes lettres, et donne le vrai payload : **5,3756 /
   4,6667 / 4,2029** (`mesures/rtbits-planes-8b-2026-08-09.txt:99-110`).
   → **à corriger** : renommer la colonne `bpw_kernel`, corriger la légende.
   Aucune valeur ne bouge, 0 $ de GPU.

⚠️ **Et deux des cinq taux publiés ne sont épinglés par aucun test.**
`rtbits.rs:1119-1146` ne fixe que **4,804**, **4,342** et le taux d'exception
de 3,3824 % ; **5,510 et 3,589 n'apparaissent que dans des commentaires**.

---

## 6. Machine, argent, et ce qui coûte vraiment

`l40sx1` — 8 vCPU / 62 Go / 48 Go VRAM / **1,80 $/h** / SM 89
(`ops/run.py:84`) — est le **seul** membre de `BENCH_FLAVORS` (`:883`), et
c'est la carte de tous les chiffres publiés.

**La facture d'un banc est le transcodage CPU, pas la mesure** :

| run | bras | transcodage | facturé |
|---|---|---|---|
| `c1-planesbench` | 3 | 150 s | **0,08 $** |
| `nuit-planes12x-q8` | 4 (+ un `fusedrun`) | 1 459 s | 0,90 $ |
| `e2-golay70-bench` | 5 | 1 468 s | **0,74 $** |

Le chronométrage lui-même vaut ~0,4 s. Passer de 4 à 5 bras a coûté **+9 s**.
→ **Ajouter un bras est quasi gratuit en dollars.** Le facteur limitant de
cette campagne est le **jour d'ingénierie**.

Deux réserves de comptabilité :

- **`docs/data/jobs.csv` s'arrête au 2026-08-08 et ne contient pas le 14B.**
  Sa somme fait exactement 19,82 $ — le chiffre que le papier cite à **quatre
  endroits** (`main.tex:70`, `intro.tex:79`, `evaluation.tex:193`,
  `conclusion.tex:26`), tous écrits à la main puisque `make_figures.py` n'ouvre
  jamais ce fichier. La dépense réelle est ~47,5 $ une fois les ~27,69 $ du 14B
  ajoutés (`docs/reprise-14b-2026-08-09.md:23`). Les quatre sites et
  `docs/data/README.md:13` devront bouger **ensemble**.
- **L'image de run ne contient que des binaires Rust**
  (`ops/Dockerfile.cuda:73-77` = `ca-certificates` + `libssl3`). **Aucun moteur
  Python n'y tourne** : une stratégie B demanderait une image neuve ou la route
  `run_uv_job` (`ops/run.py:1012-1044`).

Et une bonne nouvelle pour le balayage de batch : **cuBLAS et cuBLASLt sont
déjà des features actives de cudarc** (`llvq-cuda/Cargo.toml:54-55`). La
baseline GEMM honnête est un appel, pas un noyau à écrire.

---

## 7. Récapitulatif des verdicts

| concurrent | verdict | stratégie | ce qui le rend cher |
|---|---|---|---|
| **mono-shell LLVQ** (`MonoShell3` + `Shell12`) | ✅ faisable | A — réimplémentation clean-room, aucun code amont n'existe | le ré-encodage (contenu différent, pas une bijection) — mais local et gratuit |
| **QTIP** | ✅ faisable | A — prouvée par leur propre `test.cu` sans torch | NVRTC vs nvcc, dé-templatage, 5 instanciations, et la décision GPL |
| **AWQ / Marlin** | ✅ faisable | A — Marlin d'origine est un `.cu` autonome Apache-2.0 | deux bras au lieu d'un, et il faut un balayage de batch pour que ça ait un sens |

**Aucun des trois n'est infaisable.** Ce qui borne la campagne, ce sont les
jours et le risque du §4, pas la disponibilité des noyaux.
