# Audit de recherche — LLVQ au 2026-09-01

> Objet : audit complet du dépôt, puis **pistes de recherche fondamentale** sur
> les deux problèmes ouverts — le **dépliage en VRAM** du codebook Leech, et la
> **perte de qualité** à 2 bits. Rédigé depuis l'état du dépôt à `f0b86e7`
> (branche `claude/repo-audit-optimization-20r2pz`), trois dossiers techniques
> internes (noyau/format, quantifieur/qualité, état/roadmap) et la littérature
> 2025-2026 citée au §6.
>
> Conventions du dépôt respectées : chaque nombre est étiqueté *mesuré* (il
> existe un journal dans `docs/mesures/`), *calculé* (dérivé de nombres mesurés)
> ou *estimé* (le mien, à ne pas citer comme un fait). Les citations de code
> sont `fichier:ligne` sur l'arbre courant. Ce document **propose** ; il ne
> tranche rien qui relève de l'opérateur, et il ne rouvre rien de ce que le §1.3
> liste comme fermé.

---

## 0. Résumé exécutif

**Ce que le dépôt établit.** Un quantifieur Leech Λ₂₄ + Spherical GPTQ complet
en Rust, 526 tests, culture de mutation testing et de pré-enregistrement
tamponné, quatre modèles scellés (0,6B/4B/8B/14B), un noyau CUDA fusé qui
décode **301 classes sans divergence** et sert le 4B à **100,6 tok/s dans
2,57 Go** (*mesuré*, config v1 gelée le 08-31), et — fait neuf non consigné
dans `HISTORIQUE.md` — les **CUDA Graphs (A2) rendent +13,45 % [13,36–13,58]**
bout-en-bout (*mesuré*, `docs/mesures/a2-verdict-2026-09-01.txt`).

**Le point dur, en une phrase.** Le format servi lit **4,80 b/poids en VRAM
pour 2,00 b/poids d'information** (*mesuré*, `paper/sections/layouts.tex:33-68`).
Cela coûte au projet **sa propre thèse** : à 5,16 b/param modèle entier, un 70B
occuperait **~45 Go** (*calculé*), pas les 18-24 Go annoncés au §1 de
`CLAUDE.md`. Sur l'axe VRAM le point servi est dominé par l'AWQ 4 bits (5,30
b/param, −0,28 pp MMLU) et par IQ2_XXS (2,48 b/param) ; sur l'axe qualité il
domine IQ2_XXS de +16,2 pp mais perd 14,7 pp sur le f16. **Le dépliage n'est
pas un détail d'implémentation : c'est ce qui décide de la classe de modèle
qu'on peut charger.**

**Diagnostic fondamental du dépliage (§3).** Le codebook (1,1·10¹⁴ points) est
un **ensemble non-produit** : un code coordonnée-par-coordonnée (les plans de
bits) le paie **112 bits pour 48**, un décodage arithmétique de l'index le
paie en calcul (E1v : 0,25× FP16), et une table est impossible. Il ne reste
qu'une voie, celle que QTIP a prise : **un décodage séquentiel à petit état**.
Or la théorie des codes l'a déjà écrite pour Λ₂₄ : Forney (1988) donne au
réseau de Leech un **treillis à 3 sections et 256 états**, qui est la
construction de Turyn/Lepowsky–Meurman **Λ₂₄ ⊂ E₈ × E₈ × E₈**. **Piste F1** :
un format 48 bits = 8 bits d'état + 3 index de section, décodé par **trois
lookups de type E8P** (tables de l'ordre du Kio) et deux additions — le
dépliage disparaît, le transcodage aussi, et l'encodeur devient *plus rapide*
(256 × 3 décodages E₈ au lieu d'une recherche à 301 classes).

**Diagnostic fondamental de la qualité (§4).** Le codebook n'est pas le
coupable : sur source gaussienne il reproduit le papier à 0,2 point et l'oracle
de calibration ne rend que −1,6 %. La perte vient de **l'altitude de
l'objectif** (le proxy local `Tr(ΔW H ΔWᵀ)` a été trois fois meilleur pendant
que la composition était pire : `group_scales`, design C, `gptq2`), de
**l'instabilité de l'estimateur de Hessienne** (σ = 5,2 % entre trois tirages
de 131 k tokens ; 13,5 échantillons par dimension sur `down_proj`), et d'une
**allocation uniforme** là où la littérature 2026 montre que l'attention
s'effondre la première à 2 bits. Sept pistes, dont trois à 0 $ : **stabiliser
H** (shrinkage hors-diagonale, pour rendre toute autre expérience mesurable),
**GPTQ en faisceau** (le GPTQ est un Babai glouton ; K-best sur les candidats
que `shell_bests` rend déjà), et **l'attribution leave-one-out** de la chute
MMLU par type de matrice (décide la précision mixte, ~1-2 $).

**Ordre proposé (§5).** (i) trois expériences à 0 $ sur la stabilité de H ;
(ii) l'attribution leave-one-out ; (iii) le compte d'états et d'alphabets de la
piste F1 sur papier, puis sa distorsion dans `llvq-bench` ; (iv) seulement
alors une carte. Tout ce qui recalibre passe sous σ = 5,2 % tant que (i) n'est
pas fait.

---

## 1. Ce que le dépôt établit, et où il se situe

### 1.1 Les quatre axes du point servi (Qwen3-4B, L40S)

| axe | LLVQ servi (v1) | concurrent | verdict | source |
|---|---|---|---|---|
| disque | 1,77 Go (1,41 avec embedding int8) ; 2,0702 b/poids | AWQ 2,67 Go · IQ2_XXS 2,0625 bpw | ✅ gagné sur l'AWQ ; à parité avec IQ2_XXS | `README.md`, `HISTORIQUE.md:1877-1882` |
| VRAM, b/param modèle entier | **5,162** | AWQ **5,302** · IQ2_XXS **2,479** | ⚠️ sous l'AWQ de 2,6 %, **2,08× au-dessus d'IQ2_XXS** | `docs/mesures/rtbits-planes-8b-2026-08-09.txt`, `HISTORIQUE.md:61-66` |
| débit intra-pile | ×1,11 à tête identique (le noyau) ; 100,6 tok/s servi | ×2,41 pour l'AWQ dans vLLM (autre pile) | ⚠️ ne se compare pas inter-piles | `b2-fusedrun-plages`, `awq-vllm-4b` |
| qualité MMLU micro | 55,59 % (−14,73 pp) | AWQ 70,04 · IQ2_XXS 39,39 (+16,20 pp [12,64 ; 19,72] pour nous) | ❌ le 4 bits domine ; ✅ devant le 2 bits tiers | `a4-campagne`, `HISTORIQUE.md:61-66` |

Tous *mesurés*. La dernière colonne du tableau `layouts.tex:33-68` ajoute le
cinquième point qui recadre tout : **QTIP 2 bits, dans notre banc, même
processus : 2,000 b/poids noyau, 2,246 ms contre 5,103 pour `Planes14`,
r = 2,27× [2,27–2,28]** (*mesuré*, F2).

### 1.2 La thèse VRAM n'est pas tenue par le format servi

Le §1 de `CLAUDE.md` dit : « À 2 bits, un 70B passe de 140 Go à 18 Go — il
rentre sur une carte 24 Go. » C'est vrai du **disque**. En VRAM (*calculé* sur
les b/param mesurés, embedding ~1,5 % d'un 70B à 8,5 b/param) :

| format VRAM | b/param 70B | Go pour 70B | tient sur 24 Go ? |
|---|---|---|---|
| `Planes14` + q8 (servi) | ≈ 4,86 | **≈ 42,5** | non |
| `Planes12x` + q8 | ≈ 4,40 | ≈ 38,5 | non |
| `Golay70` (écarté, 1,31×) | ≈ 3,66 | ≈ 32 | non |
| E3 (borne, 3,04) | ≈ 3,12 | ≈ 27 | non |
| **un format à 48 bits/bloc + queue + échelles** | **≈ 2,3** | **≈ 20** | **oui** |

Aucun format écrit, mesuré ou borné dans le dépôt ne passe sous 24 Go pour un
70B. Seul un format qui **ne déplie pas** l'index le fait. C'est l'argument
qui fait du §3 la priorité de recherche, et pas un raffinement du noyau.

### 1.3 Fermé — ne pas rouvrir sans idée neuve (tout est *mesuré*)

- **Format à ALU inchangée** : `Golay70` 1,31× puis v2 1,77× (< 2,0
  pré-enregistré) ; E1v 0,25× ; E3 borné à 3,04 b/poids contre 2,60 ;
  `E1c14` plus gros que `Planes14` au 4B une fois aligné warp.
- **`nullk` comme plancher machine** : faux, c'est le plancher de *notre*
  géométrie de lancement (QTIP passe dessous).
- **Optimiser le témoin f16** : à 1,5–2,4 % de cuBLAS (F1).
- **Conflit de bancs / stride 28** : nul (K-1).
- **Idée C (bit-serial int8 sur les plans)** : algèbre exacte mais ~148 ops/bloc
  contre 96 ; conditionnée à un format à masques ≥ 32 coordonnées.
- **Volume de calibration** : oracle −1,6 %, ×13 de tokens → −1,2 % (lot B,
  *mais* à 0,6B/3 blocs, sous le σ d'époque — cf. §4.2 pour ce que ça ne dit pas).
- **Design C** (magnitude libre + résolution close) : ×1,99 à profondeur.
- **`group_scales`** : mieux local, catastrophe globale.
- **Rotation de sortie** : moyenne ≈ 0 dans l'ablation du papier.
- **Codage entropique de l'index** : 46,65 bits d'entropie pour 47 payés.
- **Bits de gain 0/1/2/4** : le bruit inter-graines (13,9 %) dépasse le signal
  inter-bras (10,6 %) ; rien n'est adopté.
- **Coquille unique** : perd 1,8 point de rétention à 48 bits contre la boule.

---

## 2. Audit du dépôt

### 2.1 Ce qui est solide (mesuré par l'agent d'audit, `wc`/`grep`)

- **157 fichiers Rust, 67 678 lignes, 526 `#[test]`, 0 TODO/FIXME**, 35 tests
  `#[ignore]` déclarés ; cinq crates du cœur en `#![forbid(unsafe_code)]` et
  sans dépendance (garde `cargo tree` en CI).
- Sweeps intégraux du 4B scellé (150 681 600 blocs) pour chaque layout,
  transcodeurs bit-exacts, ~25 mutants tués ; le texte des noyaux CUDA exécuté
  contre Rust (`host_e1v.cpp`, `*_decoder_matches_rust.rs`).
- Discipline de mesure rare : médianes à plage, rapports formés round par
  round, empreintes de tokens, pré-enregistrements `.ots` (28), registre des
  coûts (`docs/data/jobs.csv` : **98 jobs, 91,85 $**).

### 2.2 Dettes concrètes, chacune avec sa correction

| dette | fait | correction proposée | coût |
|---|---|---|---|
| **Le résultat le plus important de la semaine n'est pas dans le fil** | A2 = +13,45 % vit dans `a2-verdict-2026-09-01.txt` et `jobs.csv`, pas dans `HISTORIQUE.md` (dernière entrée : le préreg) | entrée HISTORIQUE + mise à jour des trois surfaces de reprise | 0 $, 30 min |
| **Compteurs qui dérivent** — le dépôt documente sa propre dérive ≥ 10 fois | `ARTIFACT-EVALUATION.md` annonce 69 journaux / 73 jobs / 87,36 $ / « aucun `.ots` upgradé » ; réel : 92 / 98 / 91,85 $ / 16 ancrés. `HISTORIQUE.md:72-79` : 86 journaux, réel 92 | **un `ops/status.py` qui génère `docs/ETAT.md`** (compteurs, coût, état `.ots` via `otsaudit`, config servie) et un test CI qui échoue si un fichier de reprise cite un compteur périmé | 0 $, ½ journée |
| **Masse documentaire** | 6 fichiers de reprise = **9 475 lignes**, `CLAUDE.md` seul **2 932** ; les bannières datées s'empilent sans se remplacer | `CLAUDE.md` réduit à une carte d'une page pointant `ETAT.md` généré ; le reste en `docs/archive/` avec date | 0 $ |
| **Trou `unsafe` des tests d'intégration** (connu, `CLAUDE.md` §7) | `[workspace.lints]` absent de `Cargo.toml` | `[workspace.lints.rust] unsafe_code = "forbid"` + `[lints] workspace = true` sur les cinq crates du cœur ; les trois crates matériels gardent leur permission | 0 $, 20 min |
| **La CI ne compile aucun noyau** | `ci.yml` exclut `llvq-cuda` et `llvq-metal` (pas de nvcc) | compiler le **texte** des `.cuh` par `clang++ -x c++` avec un prélude d'émulation (le dépôt le fait déjà pour E1v) pour les cinq décodeurs, et exécuter les `*_matches_rust` sur un échantillon | 0 $, 1 jour |
| **Le 2×2 du gel v1 est incomplet** | `planes12x + FUSE=1` refusé par `check_fuse` (`llvq-llm/src/fused.rs:563-576`) | étendre la fusion à `planes12x` (même stride 12, records alignés mot) ou déclarer l'arbitrage | ~0,3 $ |
| **Instruction périmée active** | `docs/exp-piles-isolees-2026-08-30/MACHINES.md:50-52` : « Ne pas activer la fusion » | remplacer par la config v1 | 0 $ |
| **Pas de prefill servi** | `MAX_ROWS = 256` (`llvq-llm/src/model.rs:500`), une matvec par token en prompt | déjà « bloqueur produit » (idée A) ; cf. §3.6 pour ce que ça change au choix de format | — |
| **Toolchain** | `rust-toolchain.toml` 1.95.0 épingle vers l'avant ; le compilateur du run publié est inconnaissable | écrire `rustc -V` dans l'en-tête de chaque artefact `.llvq` (format v5) | 0 $ |

---

## 3. Le dépliage — analyse fondamentale et pistes

### 3.1 Le théorème informel qui gouverne le problème

Le noyau doit rendre `w·x` depuis un index de 47 bits désignant un point de
Λ₂₄(12). Trois manières d'y arriver, et le dépôt a **mesuré** deux d'entre
elles jusqu'au bout :

| voie | mécanisme | ce que ça coûte | où c'est mesuré |
|---|---|---|---|
| **table** | LUT index → 24 valeurs | 1,1·10¹⁴ entrées : impossible | `layouts.tex:190-202` |
| **parallèle** (coordonnée par coordonnée) | plans de bits, une décision par slot | **112 bits pour 48** : la redondance d'un code produit sur un ensemble non-produit | `Planes14`, 4,80 b/poids, 2,15× |
| **arithmétique** (dé-ranger l'index) | marche binomiale, Golay par XOR, parité | borné en calcul sur notre géométrie : 869 ns/bloc côté archive, **0,25× FP16** sur carte | E1v ; `Golay70` v2 1,77× |

La quatrième voie est celle de QTIP : **séquentielle à petit état** —
`(état, bits) → (état', valeur)` par une table de 2 Kio, ~2-3 instructions
par poids, *aucune* arithmétique dépendante des données. Elle est absente du
dépôt parce que l'index v1 (rang mixed-radix sur 301 classes) n'a pas cette
structure. **Mais le réseau de Leech, lui, l'a.**

### 3.2 Le fait de structure que le dépôt n'exploite pas

Forney, *Coset codes II* (1988) : le réseau de Leech admet un **treillis à
3 sections et 256 états**, régulier, obtenu par la « cubing construction »
qui donne aussi le code de Golay. C'est la construction de Turyn, formalisée
par Tits (1980) et Lepowsky–Meurman (1982) : avec `L = E₈` et une polarisation
`(M, N)` de `E₈`,

```
Λ₂₄ ≅ { (m + a,  m + b,  m + c) : m ∈ M ; a, b, c ∈ N ; a + b + c ∈ 2L }
```

(formule à vérifier sur Lepowsky–Meurman 1982 avant toute ligne de code — je
la cite de seconde main). Lecture pour nous : **un point de Leech est trois
points de E₈ liés par une contrainte de coset à 8 bits**. Le décodage n'a
plus besoin ni de 301 classes ni d'un rang combinatoire : il a besoin de
**trois lookups dans un codebook E₈** — exactement l'objet que QuIP# sert
depuis 2024 sous le nom E8P (2¹⁶ points, stockés comme 256 vecteurs de base ×
motifs de signes, dans quelques Kio de mémoire partagée).

Ce n'est pas une spéculation isolée : un décodeur de Leech « Turyn-based »
existe (Corlay et al., WCC 2019), et la littérature 2025-2026 sur la
quantification par réseaux emboîtés (Kaplan–Ordentlich, ISIT 2025 ;
NestQuant ; HyperQuant, qui quantifie sur E₈/D₄ derrière une Hadamard)
converge vers « réseau de dimension 4-8 + table » précisément parce que la
table tient en cache. **Λ₂₄ en trois sections met le gain de codage de Leech
dans ce régime.**

### 3.3 Piste F1 — « Leech-3×E₈ » : un format 48 bits sans dépliage

**Format VRAM proposé (par bloc de 24 poids)** :

```
[ état : 8 ][ section 1 : ~13 ][ section 2 : ~13 ][ section 3 : ~13 ][ gain : 1 ] = 48 bits
```

- l'**état** (8 bits) est le coset de Forney qui lie les trois sections ;
- chaque **section** indexe un point de E₈ dans le coset désigné par l'état,
  à l'intérieur d'une boule de ~2¹³ points (*estimé* : 40 bits pour trois
  sections, à ajuster au compte exact des cosets) ;
- **décodage** : 3 lookups `(état, idx) → 8 valeurs` + 2 additions vectorielles
  + le signe/gain. À la QuIP#, chaque table se factorise en bases × signes
  pour tenir en quelques Kio par section (*estimé*, à compter).

**Ce qu'on gagne, si ça tient** (tout *estimé*, la règle du dépôt étant qu'un
compte d'opérations n'est pas une prédiction de temps) :

| grandeur | `Planes14` (mesuré) | F1 (borne) |
|---|---|---|
| b/poids noyau (payload + queue f32 + échelles) | 4,804 | **≈ 2,16** (2,000 + 0,149 + 0,010) ; 2,08 en comptabilité inférence |
| Go lus, 252 matrices du 4B | 2,18 | **≈ 0,98** |
| ALU par poids | ~4 (arbre prédiqué) | ~1 lookup / 8 poids + 2 add/8 ; **moins que QTIP** (2-3 instr/poids) |
| transcodage au chargement | 131 s | **0** — le format disque *est* le format VRAM |
| encodeur | recherche à 301 classes, 656 µs/bloc | 256 états × 3 décodages E₈ (E₈ se décode en ~100 ops) : *plus rapide* |

**Ce que ça coûte, et c'est là que vit le risque** :

1. **Le codebook change.** L'ensemble indexé n'est plus la boule Λ₂₄(12) mais
   `Λ₂₄ ∩ (B₈ × B₈ × B₈)` dans les coordonnées de Lepowsky–Meurman : même
   réseau (même gain de codage), **autre région de mise en forme**. La boule
   de dimension 24 a ~1,0 dB de gain de forme, le produit de trois boules de
   dimension 8 en garde ~0,65 dB (*estimé* d'après les valeurs classiques) :
   perte attendue de l'ordre de **0,3-0,4 dB ≈ +7-9 % de MSE** sur la
   direction. La contrainte de coset entre sections peut en récupérer une
   partie (c'est la définition du *trellis shaping* de Forney 1992 : ~1 dB
   avec 4 états, 1,36 dB revendiqué avec plus).
2. **G5 doit être rejoué** : banc gaussien, puis 0,6B/28 blocs (0 $), puis 4B
   (7,11 $ mesuré par run).
3. **Un format v2 et un encodeur** : l'index v1 est un contrat de stabilité
   (`codebook_fingerprint`) ; F1 en est un autre. L'écriture est modeste —
   E₈ est un réseau trivial à décoder — mais c'est un format complet.

**Critères posés d'avance (à tamponner avant la première mesure)** :
sur source gaussienne, rétention ≥ **91,0 %** à 48 bits (la boule fait 92,14,
la coquille 12 seule 90,34 et a été écartée) ; sur carte, dans le banc à bras
entrelacés avec QTIP comme témoin, `t(F1) ≤ 1,15 · t(QTIP)` ; kill si la
rétention gaussienne est < 90,3 % (le point où la coquille unique a été
enterrée).

**Première étape, 0 $, une semaine de papier** : écrire la polarisation
`(M, N)` de E₈ explicitement, compter les états de Forney et l'alphabet de
chaque section pour un budget de 47 bits, vérifier la bijection sur
`|Λ₂₄(≤ r) ∩ tri-boule|`, et brancher le compte dans `llvq-bench` contre la
table G4 existante. La bijection se prouve par énumération exhaustive comme le
fait `classes_reproduce_theta_series`.

### 3.4 Piste F2 — la variante séquentielle : treillis à 24 pas + trellis shaping

Si le compte d'états de F1 ne tient pas dans les tables qu'un SM offre, la
même structure se déroule **coordonnée par coordonnée** : un treillis minimal
sur Λ₂₄/8ℤ²⁴ (|Λ₂₄/8ℤ²⁴| = 8¹² = 2³⁶ cosets, *calculé* sur les déterminants),
état ≤ quelques centaines, table `(état, 2 bits) → (état', valeur)` de
quelques Kio, exactement le mécanisme HYB de QTIP. La mise en forme se fait
par **trellis shaping** (Forney 1992) sur les bits de poids fort — un Viterbi
**à l'encodeur seulement**, le décodeur restant un automate. L'asymétrie
encodeur/décodeur est celle que le §4 de `CLAUDE.md` pose en principe.

Réserve : un décodeur séquentiel à 24 pas par bloc met la latence de la
chaîne de lookups sur le chemin critique de chaque lane ; QTIP montre que
c'est absorbé à 2 bits/poids sur L40S, mais **notre géométrie « un warp par
ligne » ne l'est pas** (c'est la leçon de F2/`nullk`). F2 ne se mesure donc
que dans la géométrie d'A3 (grille fixe + split-K), pas dans celle de
`tv_planes`.

### 3.5 Piste F3 — l'allocation de débit par rayon, rendue possible par F1/F2

Un format à index brut est **flexible en débit** : un bloc à 44, 47 ou 50 bits
se lit sans changer de noyau (là où les plans de bits imposent un nombre de
niveaux). Cela ouvre la **précision mixte par rayon de boule** : `cap` par
matrice ou par ligne, guidé par une saillance (Fisher ou gradient de perte,
cf. §4.6), à budget total constant. La littérature 2025-2026 (FAMPWQ, APTQ,
KVTuner, HyQuant) mesure que **les projections K et l'attention en général
sont les plus sensibles** ; le §4.6 propose l'attribution qui décide où
dépenser.

### 3.6 Écartées sur papier, avec le chiffre

- **Table côté activations** (T-MAC / LUT-GEMM / LUT Tensor Core) :
  précalculer `⟨x_j, c_k⟩` par position de bloc. Avec un codebook par section
  de 2¹³ points : 3 × 8 192 × (2560/24) × 2 o ≈ **5 Mo par token et par
  matrice** contre 1,6 Mo de poids (*calculé*) — le régime LUT-GEMM exige des
  codebooks ≤ 2⁸. Mort à cette taille de codebook.
- **Dépliage par tensor cores** : c'est l'idée C, algèbre exacte, coût ALU
  dégradé (~148 ops/bloc). Elle ne redevient vivante que si F1 échoue *et*
  qu'on redessine des masques ≥ 32 coordonnées.
- **Transcodage paresseux par couche** (garder 48 bits en VRAM, déplier une
  couche dans un tampon avant sa matvec) : à M = 1 c'est refaire le dépliage à
  chaque token, donc payer en ALU ce qu'on ne paie plus en octets — c'est E1v.
  **Redevient exact à M ≥ 8** (prefill, batch) : le dépliage se paie une fois
  par tuile puis se réutilise sur M lignes — c'est ce que QuIP#/QTIP font en
  déquantifiant vers f16 pour les grands M. Conséquence : **le format optimal
  dépend de M**, et le bloqueur produit « pas de prefill » (idée A) est aussi
  une question de format.

### 3.7 Plan de validation

| étape | coût | ce qui décide |
|---|---|---|
| F1 sur papier : polarisation, états, alphabets, bijection | 0 $, ~1 sem. | le compte tient-il dans 47 bits avec des tables ≤ ~16 Kio ? |
| F1 dans `llvq-bench` : rétention gaussienne à 48 bits, `β`-sweep | 0 $, ~1 sem. | ≥ 91,0 % → continuer ; < 90,3 % → kill |
| F1 encodeur + format v2 + 0,6B/28 blocs | 0 $ (Mac) | ppl à ±σ de `leech1c12` sur même graine |
| noyau F1, bras dans `planesbench` avec QTIP témoin, L40S puis A100 | ~1 $ | `t(F1) ≤ 1,15·t(QTIP)` ; f16 f64 à 1e-5 |
| 4B scellé F1, `fusedrun`, MMLU | ~8 $ | b/param ≤ 2,6 ; MMLU ≥ 55,59 − 2·0,43 pp apparié |

---

## 4. La qualité — analyse et pistes

### 4.1 Où est la perte (faits mesurés, puis lecture)

**Ce qui n'est pas coupable** : le codebook. Notre shape–gain 0 bit reproduit
la Table 8 du papier à 0,2 point (88,90 contre 89,12) ; l'écart du spherical
shaping s'explique par β ; et l'oracle (calibrer sur le test lui-même) ne
referme que 29 % de l'écart de quantification.

**Ce qui l'est, par ordre de preuve** :

1. **L'altitude de l'objectif.** Trois fois, un proxy local meilleur a donné
   une composition pire : `group_scales` (21,24→21,17 local ; 44,66→53,60
   global), design C (test vert ; ×1,99 à 28 blocs), et le bras `gptq2`
   (pertes par module médiane 0,0021 ; modèle mort, MMLU = hasard). Le
   dépôt en tire « la rigidité de norme est porteuse à profondeur ». La
   lecture plus générale est celle de BRECQ/OmniQuant/PV-tuning/GPTAQ : **un
   objectif par couche ne voit pas la propagation**, et toute la littérature
   qui gagne à 2 bits remonte l'objectif d'un cran (bloc, ou modèle).
2. **L'estimateur de Hessienne.** σ = 5,2 % entre trois tirages de 131 k
   tokens sur le même 4B (*mesuré*, F5), avec **13,5 échantillons par
   dimension** sur `down_proj` (9728², *calculé*). La littérature 2025-2026
   (« Stable Diagonal Curvature », arXiv 2604.13806) mesure exactement ce
   motif : **les termes diagonaux se stabilisent vite, les hors-diagonaux
   restent instables même à 2 048 séquences**, et c'est par eux que passe la
   rétroaction d'erreur de GPTQ.
3. **L'allocation.** Sept projections, une Hessienne par activation, un même
   `cap` partout, aucune attribution par matrice n'existe dans le dépôt
   (`docs/mesures` ne contient aucun budget d'erreur par couche). Or le profil
   MMLU est le profil d'un **effondrement de calcul**, pas d'une dégradation
   de signal : algèbre abstraite et comptabilité au hasard, histoire et droit
   tenus. Le papier « From Signal Degradation to Computation Collapse » (arXiv
   2604.19884) nomme ces deux modes, mesure qu'à 2 bits **l'entropie
   d'attention normalisée dépasse 0,80 dans les couches médianes**, et que les
   réparations sans entraînement soignent le premier mode, pas le second.
4. **Le biais radial** : +3,7 % de sur-coût géométrique sur la config servie,
   parce que le gain quantifie ‖w‖ là où l'optimum à direction fixée est
   ⟨w, û⟩ (*mesuré*, `cosdiag-biais-radial`). Gratuit, non traité.

### 4.2 Piste Q1 — stabiliser H d'abord (méthodologie avant tout)

Tant que σ = 5,2 %, aucune piste de qualité qui recalibre n'est mesurable
sans trois graines (21,45 $ par triplet au 4B). **La première expérience de
qualité est donc une expérience sur H**, et elle coûte 0 $ :

1. Sur le Mac, pour trois matrices du 4B (`q/k/v`, `down`, `up`), calculer H
   à 131 k tokens pour deux graines et 1 M tokens pour une : corrélation des
   diagonales, corrélation des hors-diagonaux, spectre. Si la diagonale est
   stable et le hors-diagonal ne l'est pas, le mécanisme de F5 est nommé.
2. **Shrinkage linéaire hors-diagonal** `H_ρ = ρ·H + (1−ρ)·diag(H)` (la
   famille de 2604.13806), balayé sur 0,6B/28 blocs à trois graines : la
   grandeur à publier n'est pas la ppl médiane mais **l'étendue
   inter-graines**. Kill si ρ* = 1 (aucun shrinkage n'aide).
3. Damping ∝ 1/√N plutôt que constant — même banc.

Ce que ça achète : un plancher de bruit divisé par 2-3 rendrait mesurables
les effets de 1-3 % que toutes les autres pistes promettent. ⚠️ Cela ne
rouvre pas le volume de calibration : ça teste **l'estimateur**, pas la
quantité de données.

### 4.3 Piste Q2 — l'objectif asymétrique et la pondération de sortie

Aujourd'hui (`calib.rs:549-563`, `:808-815`) : H est collectée sur des
activations **déjà passées par les blocs quantifiés**, et le résidu vise
`W·x̂`. C'est le GPTQ « symétrique sur entrées dérivées ». **GPTAQ** (ICML 2025,
arXiv 2504.02692) ajoute le terme qui vise la **sortie du modèle f16**
`W·x` : « 20 lignes de plus que GPTQ », gains concentrés à 2-3 bits, et
**KronQ** (COLM 2026, arXiv 2607.07964) mesure que la plus grande part de son
gain à W2 vient de cette correction de dérive. Pour nous : garder le flux
résiduel f16 en parallèle du flux quantifié (deux passes existent déjà), et
former la cible sur `W·x` — un changement local à `calib.rs`.

Second cran : la **Hessienne de sortie**. YAQA (arXiv 2505.22988) et KronQ
utilisent le facteur de Kronecker côté sortie (covariance des gradients) pour
pondérer les lignes. Spherical GPTQ est par ligne : une pondération par ligne
s'insère sans toucher au codebook, comme un poids sur l'objectif de rétraction.

Coût : 0 $ à 0,6B (gate à 28 blocs, protocole du design C), 7 $ au 4B.
Critère : sous Q1, une amélioration ≥ 2× l'étendue inter-graines.

### 4.4 Piste Q3 — GPTQ en faisceau (le Babai glouton n'est pas l'optimum)

Un résultat 2025 (arXiv 2508.01077) montre que **GPTQ est l'algorithme du plan
le plus proche de Babai** sur le réseau engendré par les données : un CVP
résolu gloutonnement, bloc après bloc. L'amélioration classique de Babai est
la **recherche en faisceau / sphere decoding** : garder K candidats par bloc
et propager l'erreur pour chacun. Le dépôt a déjà la pièce manquante :
`shell_bests` rend **douze candidats** (un par coquille) au lieu d'un.

- K ∈ {2, 4, 8} candidats par bloc de 24, faisceau de largeur K sur la ligne,
  score = `Tr(ΔW H ΔWᵀ)` partiel ; l'encodeur (90 % du temps) coûte ~K× —
  2,4 h → ~10 h pour le 4B sur Mac à K = 4, 0 $.
- AQLM fait exactement cela (beam search sur ses codes) et le donne comme un
  de ses gains principaux.
- Critère : ppl à 28 blocs du 0,6B améliorée d'au moins l'étendue inter-graines,
  **et** stable sur trois graines (c'est un changement de recherche, pas
  d'estimateur : il ne devrait pas augmenter σ).

### 4.5 Piste Q4 — reparamétrisation d'équi-norme et cartes 24×24 apprises

Deux idées de la même famille : utiliser les **invariances du réseau** pour
rendre les blocs de 24 plus faciles à coder, sans un bit de plus en VRAM.

**(a) Équilibrage entre couches (Nagel–van Baalen 2019, version VQ).** Toute
échelle diagonale `s` sur la dimension intermédiaire est libre :
`down·diag(s)` et `diag(1/s)·up`, `diag(1/s)·gate`. Or les lignes de `up` et
`gate` sont **déjà** munies d'une échelle f16 par ligne (le format la stocke),
donc `1/s` y est absorbé gratuitement, et `s` peut être choisi pour **égaliser
les normes des blocs de 24 de `down_proj`** — la matrice la plus large, à 13,5
échantillons par dimension. Même liberté entre `v_proj` et `o_proj` par
dimension de tête. ⚠️ Pas libre pour `q/k` sur Qwen3 : `q_norm`/`k_norm`
renormalisent avant RoPE. Coût 0 $, un test d'invariance bit-exact (sortie du
bloc inchangée à s quelconque) avant toute mesure.

**(b) Cartes 24×24 apprises côté activations.** La rotation est appliquée aux
activations une fois par token (`rot_apply`, hissée par `ROT_SHARE`). Une
carte **bloc-diagonale** `A_j` (24×24 par position de bloc), composée après la
Hadamard, coûte 576 MAC × 107 blocs ≈ 62 k MAC par token et par activation —
~1 % de la matvec (*calculé*), ≈ 33 Mo de tables pour le 4B (*calculé* sur
les 790 blocs d'entrée par couche × 36 couches, ≈ 0,07 b/poids). Le décodeur est **octet-identique** ; le codebook effectif devient
`A_j⁻ᵀ·Λ₂₄` par position — c'est la direction de GLVQ (NeurIPS 2025,
arXiv 2510.20984 : générateur de réseau appris par groupe, arrondi de Babai
pour la différentiabilité), avec Leech comme base au lieu d'un réseau libre.
La version **diagonale** de `A_j` est exactement le « fine-tuning des échelles
par colonne » du papier (< 0,001 b/poids, la seule forme de FT qu'il
pratique), jamais fait ici ; et un scalaire par bloc absorbe le biais radial
de +3,7 %. L'apprentissage passe par Q6(c).

### 4.6 Piste Q5 — quantifier la fonction : attribution, puis précision mixte

**L'expérience qui manque est la moins chère du dossier.** Sur le 4B scellé,
restaurer **un type de matrice en f16** (depuis le checkpoint) et mesurer MMLU
apparié : sept bras (`q`, `k`, `v`, `o`, `gate`, `up`, `down`), même empreinte
`65dcd53655e8bfa5`, SE appariée 0,43 pp. Coût : sept évaluations MMLU du 4B,
≈ 1-2 $ (*estimé* depuis les 0,49 $ mesurés pour trois bras, `jobs.csv`). Résultat : **l'attribution des
−14,73 pp par fonction** — le budget d'erreur qui n'existe pas encore.

Ce que ça décide, avec l'arithmétique déjà prête (*calculée* sur les formes du
4B, 100,9 M poids par bloc) :

| si le coupable est… | poids | coût de le porter à 4 bits | b/poids noyau résultant |
|---|---|---|---|
| `k_proj` seul | 2,6 % | +0,05 | 4,85 |
| `q + k` | 13 % | +0,26 | 5,06 |
| tout l'attention (`q,k,v,o`) | 26 % | +0,52 | 5,32 (= l'AWQ) |
| `down_proj` | 25 % | +0,49 | 5,30 |

La littérature 2025-2026 prédit `k` et l'attention (APTQ : 4 bits sur K ;
« l'attention à 2 bits s'effondre, à 4 bits elle sauve le modèle » dans le
cadre MoE ; KVTuner). Si l'attribution le confirme, **+0,05 b/poids pour
récupérer plusieurs points de MMLU** serait le meilleur rapport du dossier —
et sous F1, où le débit est flexible par bloc, il se paie en `cap` plutôt
qu'en changement de format.

**Cran suivant, la métrique.** Remplacer la ppl comme gate de calibration par
deux grandeurs qui voient l'effondrement de calcul : **l'entropie d'attention
normalisée par couche** (0 $, une passe) et le **sous-ensemble MMLU-STEM**
apparié. Objectif *attention-aware* pour `q/k` : préserver la distribution
`softmax(qᵀk)` (KL par ligne) plutôt que la MSE pré-softmax — les deux
matrices n'agissent que par leur forme bilinéaire, et c'est elle qui se
dérègle.

### 4.7 Piste Q6 — la récupération, dans sa forme non évidente

Le post-training est la piste « évidente » ; ce qui ne l'est pas, c'est
qu'**une partie de l'adaptateur existe déjà dans le format**.

**(a) Les paramètres libres du format, 0 bit de plus.** Queue `KeepExact` :
**16,96 M poids f16** (0,47 % des projections), échelles de ligne (1,1 M),
centroïdes de gain, `RMSNorm`/`q_norm`/`k_norm` (≈ 0,2 M), embedding q8. Norm
Tweaking (AAAI 2024) montre que régler les seules normes par SGD pour
rapprocher les distributions d'activations du f16 rend beaucoup à 2 bits ;
ici l'ensemble libre est ~18 M paramètres. Une distillation KL légère de ces
seuls paramètres (quelques M tokens, une carte, ~1-3 h) est la première chose
à faire — **avant** EoRA, parce qu'elle ne coûte pas un octet.

**(b) EoRA / RILQ / SERQ** avec budget déclaré : le dossier a déjà l'arithmétique
(r = 32 f16 → +0,263 b/param pousse le 4B au-dessus de l'AWQ ; r ≤ 16 ou
adaptateurs q8). Gate déjà posé : ≥ +3 pp MMLU apparié.

**(c) Une relaxation différentiable de la recherche Leech.** BCJR-QAT (arXiv
2605.10655, mai 2026) remplace l'argmax de Viterbi de QTIP par une espérance
de Boltzmann à température T, différentiable, qui retombe sur le code dur à
T→0, et ouvre le QAT/la distillation aux codes à treillis. **L'analogue Leech
est immédiat** : `shell_bests` rend 12 candidats, `nearest_scaled` peut en
rendre K ; une **softmax sur les K meilleurs points du réseau** est une
relaxation différentiable du quantifieur, sans changer ni le codebook ni le
décodeur. Elle sert à apprendre Q4(b), les échelles, et — à la PV-tuning —
à ré-affecter des blocs sous un objectif global. C'est le chemin vers la
ligne « with FT » du papier (0 bit : 17,05 → 9,26) que le dépôt n'a jamais
empruntée.

**(d) Distillation KL bout-en-bout.** UPQ/Distill-QAT et « Reasoning-QAT »
(arXiv 2601.14888) mesurent que la KL vers le f16 est l'objectif robuste pour
les modèles de raisonnement à 2 bits, que le PTQ est la bonne initialisation,
et qu'**aligner le domaine de calibration PTQ sur le domaine QAT** accélère.
Coût réel au 4B : dizaines de dollars, pas dizaines de milliers — mais
seulement après (a)-(c), et sous un σ connu.

### 4.8 Piste Q7 — composition du corpus, avec la bonne métrique

`BACKLOG.md` §3.2 (DCLM-edu, ~15 $) existe. Ce que cet audit ajoute : l'oracle
en perplexité **ne borne pas** l'effet sur le raisonnement (le dépôt le sait),
donc le gate de §3.2 doit être MMLU-STEM apparié et l'entropie d'attention,
pas la ppl — et il ne se lance qu'après Q1, sinon il mesure une graine.

---

## 5. Programme proposé

| ordre | quoi | coût | ce qui décide | ce que ça ouvre |
|---|---|---|---|---|
| 1 | **Q1** stabilité de H (diag/hors-diag, shrinkage, 3 graines à 0,6B) | 0 $ | l'étendue inter-graines baisse-t-elle ? | rend Q2-Q7 mesurables |
| 2 | **Q5** attribution leave-one-out f16 sur le 4B scellé | ~1-2 $ | quelle fonction porte les −14,73 pp | précision mixte, cible de Q6 |
| 3 | **F1 sur papier** : polarisation E₈, états, alphabets, bijection | 0 $ | tient-il en 47 bits et ≤ 16 Kio de tables | tout l'axe VRAM |
| 4 | **F1 dans `llvq-bench`** (rétention gaussienne) | 0 $ | ≥ 91,0 % | encodeur + format v2 |
| 5 | **Q3** GPTQ en faisceau, **Q2** cible asymétrique, **Q4(a)** équi-norme — trois A/B à 0,6B sous le protocole design C | 0 $ | gain ≥ 2× l'étendue de Q1 | run 4B (7 $) |
| 6 | **Q6(a)** distillation des paramètres libres du format | ~3 $ | ≥ +3 pp apparié | Q6(b)(c) |
| 7 | **F1 noyau** dans `planesbench` avec QTIP témoin | ~1 $ | `t ≤ 1,15·t(QTIP)` | 4B scellé F1 (~8 $) |
| 8 | 2×2 du gel v1 (`planes12x+FUSE`), `ETAT.md` généré, lints workspace, CI des `.cuh` | 0-0,3 $ | — | hygiène |

Budget total des étapes 1-7 : **≈ 25 $** (*estimé*), dont ~15 $ de runs 4B.
Aucune étape n'exige une carte avant la cinquième.

**Ce que ce programme ne fait pas, à dessein** : il ne relance ni le volume
de calibration, ni un format à ALU inchangée, ni la course de décodage sur
notre géométrie, ni le 32B avant que la qualité soit décidée — quatre
interdits que le dépôt a payés.

---

## 6. Sources

**Dépôt (dossiers d'audit, tous `file:line` vérifiés sur `f0b86e7`)** :
`paper/sections/layouts.tex:190-202`, `paper/sections/decoder.tex:8-45`,
`llvq-cuda/kernels/llvq_planes.cuh:8-127`, `llvq-search/src/classes.rs:1-45`,
`llvq-search/src/index.rs:26-178`, `llvq-quant/src/gptq.rs:183-520`,
`llvq-quant/src/quantizer.rs:413-712`, `llvq-quant/src/rotation.rs:1-66`,
`llvq-llm/src/calib.rs:41-73, 549-563, 808-815`, `llvq-llm/src/fused.rs:563-576, 736-812`,
`docs/format-noyau.md`, `docs/idees-optimisation-2026-08-29.md`,
`docs/design-a3-occupation-2026-09-01.md`, `docs/BACKLOG.md:97-171, 285-411`,
`docs/mesures/f2-p3-qtip-banc-2026-08-21.txt`, `f5-graines-4b-2026-08-19.txt`,
`cosdiag-biais-radial-0.6b-2026-08-25.txt`, `gain-ab-gate-0.6b-2026-08-25.txt`,
`m3-gptq2-mmlu-2026-08-30.txt`, `a2-verdict-2026-09-01.txt`, `docs/data/jobs.csv`.

**Théorie des codes (à relire en source)** :
Forney, *Coset codes I & II*, IEEE Trans. IT 34(5), 1988 (treillis 3 sections /
256 états de Λ₂₄, cubing construction) · Forney, *Trellis shaping*, IEEE Trans.
IT 38(2), 1992 · Lepowsky–Meurman, *An E₈-approach to the Leech lattice and the
Conway group*, J. Algebra 77, 1982 · Tits, *Four presentations of Leech's
lattice*, 1980 · Corlay et al., *A Turyn-based neural Leech decoder*, WCC 2019.

**Quantification, 2024-2026 (identifiants arXiv tels que rendus par la
recherche ; résumés lus en extraits, pas en texte intégral — l'accès direct à
arxiv.org est bloqué depuis cette machine)** :
QTIP 2406.11235 (HYB/1MAD/3INST) · BCJR-QAT 2605.10655 · GPTAQ 2504.02692 ·
YAQA 2505.22988 · KronQ 2607.07964 · GuidedQuant (2025) · D²Quant 2602.02546
(Qwen3-8B 2 bits : ppl 20,24 → 14,10) · *From Signal Degradation to Computation
Collapse* 2604.19884 · *Stable Diagonal Curvature Estimate* 2604.13806 ·
*GPTQ = Babai* 2508.01077 · GLVQ 2510.20984 · Kaplan–Ordentlich (nested
lattices, small LUTs) 2505.13164 · NestQuant 2502.09720 · HyperQuant 2606.23406
· PolarQuant 2603.29078 · EoRA 2410.21271 · RILQ 2412.01129 · SERQ 2603.08185 ·
Norm Tweaking 2309.02784 · ParetoQ 2502.02631 · UPQ 2506.09104 · Reasoning-QAT
2601.14888 · PV-Tuning 2405.14852 · LC-QAT 2606.10531 · ReSpinQuant 2604.11080
· HARP 2605.29843 · FAMPWQ 2608.24945 · HyQuant 2608.27875 · FLUTE 2407.10960 ·
any4 2507.04610 · T-MAC 2407.00088 · LUT Tensor Core 2408.06003 · Mirage MPK
(megakernel, 2025) · Nagel et al., *Data-Free Quantization through Weight
Equalization*, ICCV 2019.
