# Passation — 2026-08-16 : la ligne des formats se referme, et le plancher prend sa place

> **Pour la session qui reprend.** Ce document est autonome. Il **périme le §2
> de [`passation-e1v-2026-08-15.md`](passation-e1v-2026-08-15.md)**, qui donne
> E1v comme la branche à suivre — c'est fermé depuis.
>
> 12 commits, `3a5a6b7..c52ecdf`, **1,62 $** de carte (deux jobs L40S), cinq
> constructions d'image dont trois échouées.

## 1. Ce que la session a rendu, en six lignes

- **P1c mesuré** : le vrai flux E1v décodé sur Metal rend **0,6795 ns/bloc**,
  et son adressage à largeur variable ne coûte que **+1,2 %**. Vert contre le
  kill de 1,50.
- **E1v est FERMÉ pour le chemin servi** : **0,25× FP16, 25 Go/s** sur L40S,
  contre un plancher de 1,60× posé d'avance. Le format tient exactement sa
  promesse mémoire ; c'est son décodeur en ligne qui est borné en calcul.
- **La coupe alignée ligne existe et est pesée** : 2,3877 → **2,3983 b/poids
  noyau**, **+0,478 %**, sur octets écrits — le chiffre était calculé, il est
  mesuré.
- **`e1c12` a enfin un verdict** : il **survit** à l'alignement (4,2880 contre
  4,3424 pour `Planes12x`, −1,3 %), et sa question cesse d'être une question de
  bits.
- 🆕 **Le PLANCHER est mesuré** : **2,305 ms contre 5,102 pour `Planes14`**, soit
  **45,2 %** d'une passe qui ne touche aucun poids. Il plafonne tout travail de
  format à **4,77×**.
- **Deux casses préexistantes du chemin CUDA trouvées et corrigées**, l'image
  étant incompilable depuis le 2026-08-15 sans que personne puisse le savoir.

## 2. Le résultat de fond, et il vaut plus que le verdict d'E1v

Quatre routes sous `Planes14` ont été tentées : E3 (papier), `Golay70` v2
(1,77×), `e1c14` (papier), E1v (0,25×). **Toutes bornées en calcul, aucune en
octets.** Le plancher explique pourquoi c'était le mauvais front :

| | |
|---|---|
| plafond absolu de tout travail de **format** | **4,77×** FP16 (= FP16 / plancher) |
| où `Planes14` en est | **2,16×** |
| ce que le format achète **net** du plancher | **3,11×** (8,691 ms de trafic contre 2,797) |
| coût du décodage de `Planes14` | **~7 %** du temps de trafic (779 Go/s net contre 836) |
| part du temps qu'AUCUN format ne touche | **45,2 %** |

Le format se dispute au plus 55 % du temps, `Planes14` en capture déjà
l'essentiel, et le poste majoritaire n'a jamais été attaqué. **C'est ce que la
famille `k` de P4 §2.6 existe pour amortir, et elle n'est pas écrite.**

⚠️ Ce 45 % **n'est pas** les « 39 % » de l'attribution du 2026-08-05 : celle-ci
découpe 2,04 ms par **token**, normes et attention comprises, contre 252
projections ici. **Deux dénominateurs.** Le rapprochement demande de refaire
l'attribution, pas de reporter un nombre — posé avant la mesure, tenu après.

## 3. La leçon d'ingénierie, payée trois fois en deux jours

**Tout ce qui vit sous `cfg(target_os = "linux")` n'a aucun filet**, et la seule
chose qui l'exerce est une construction d'image que personne ne lance par
routine. Trois occurrences, toutes découvertes le même soir :

| cause | depuis | trouvée par |
|---|---|---|
| `fused_cuda.rs` sans le `KvMode` de `6fcc366` | 2026-08-15, KV q8 | build 1 |
| deux tables `[_; N_ARMS]` à 7 littéraux, `N_ARMS` porté à 15 par P4 | 2026-08-15 15h44 | build 2 |
| mon propre alias `const DISPLAY: [usize; 8]` | 2026-08-16 | build 4 |

Le troisième est le plus instructif : il **reproduit le deuxième à trois lignes
de sa propre correction**, parce qu'un alias `const` doit annoncer un type,
donc restater une longueur. Il n'y a pas de bonne façon d'aliaser ces tables ;
il y a une façon de ne pas les aliaser.

**Ce qui a été fait pour que ça ne recommence pas** : les tables vivent dans
`arms.rs`, qui compile et se teste sur le Mac, et un test y exige qu'aucun bras
exécutable n'ait de ligne manquante ni de ligne en double. **Ce qui reste
ouvert** : `planesbench.rs` (2 100 lignes) et `fused_cuda.rs` n'ont toujours
aucun typage local.

## 4. Ce qui a été construit, et ce que chaque pièce vaut

| pièce | état |
|---|---|
| `llvq-metal/shaders/e1v_flux.metal` | mesuré (P1c). Corps de décodage **byte-identique** à `binomial_block.metal`, deux tests l'exigent sur le texte |
| `llvq_artifact::e1v::transcode_e1v_rows` | la coupe servable, aller-retour prouvé sur 150 681 600 blocs |
| `llvq_artifact::blockrec` | foyer partagé de la table de records — `p1host` la ré-exporte |
| `llvq-cuda/kernels/llvq_e1v.cuh` | **exact sur carte** (2,4e-8 sur 1 105 920 lignes), 79 registres, 0 spill — et trop lent |
| `llvq-cuda/kernels/nullk.cu` | mesuré : le plancher |
| `bin/cuhcheck` + `tests/host_shim.h` étendu | **14 unités CUDA parsent sur le Mac** ; le dépôt déclarait cette classe d'erreur inattrapable |
| `tests/host_e1v.cpp` + `e1v_decoder_matches_rust.rs` | le **texte** du noyau, compilé par clang++ et **exécuté** contre Rust |
| `tests/e1v_cuda_mirror.rs` | la même arithmétique re-dérivée en Rust, scan compris |

🚨 **Le défaut que le miroir a attrapé, et qui justifie tout le dispositif** :
`e1v_peek` est transcrit du `peek` Metal scellé, dont le commentaire garantit
« `off` est toujours au moins 10 ». **Faux pour E1v**, dont l'en-tête vit dans le
préfixe du groupe : son premier champ est à l'offset zéro, et `hi << (64 − 0)`
est un décalage de 64 — indéfini en C++, exécuté par NVIDIA comme un décalage de
rien. L'index Golay de presque tous les blocs serait revenu corrompu.

> **Porter un corps d'un langage à l'autre porte ses gardes ; ça ne porte pas
> les *raisons* pour lesquelles ils suffisaient.**

## 5. L'état des phases

| | état | RAF |
|---|---|---|
| **P1**, **P1b**, **P1c** | ✅ clos | — |
| **P5** | ✅ clos, 4/4 | le format survit, son décodeur en ligne non |
| **E1v** | ❌ **fermé pour le chemin servi** | le format reste disponible **hors boucle** (disque, transport) |
| **`e1c12`** | ⏳ verdict de bits rendu | sa question est désormais la **vitesse** ; noyau CUDA inexistant |
| **`e1c14`**, **E2**, **E3** | ❌ enterrés | — |
| **P4** | 🔒 bloqué sur A2/A4/A6 | `nullk` est écrit et mesuré **parce qu'il ne candidate pas** ; `cublasf16`, `mvkf16` et la famille `k` restent |
| **P2**, **P6** | ⏸ en pause (opérateur) | modèle tranché Qwen3-30B-A3B, run ~1,4 $ décide P6 |
| **P7** | non ouvert | gaté sur un package validé au 8B |
| **papier** | 🔒 bloqué | le **point 14B** (Phase 1.1 de `PLAN.md`) |

## 6. Ce qui revient à l'opérateur

| | pourquoi ça bloque |
|---|---|
| **A2** (contexte), **A4** (marge), **A6** (offload) | sans elles P4 n'a pas de critère d'admission, et six de ses bras ne peuvent pas rendre de verdict |
| **le point 14B** | il bloque le papier, et rien de cette session ne le touche |
| **budget** | 1,62 $ dépensés ; la famille `k` demandera un job de plus |
| **le tampon** | abandonné sur le pré-enregistrement E1v-CUDA, par décision explicite. Consigné dans le document avec ce que ça coûte |

## 7. Les dettes déclarées, à ne pas redécouvrir

- **`proofs/preregistration-e1v-cuda-2026-08-15.md` n'est PAS horodaté.** Son
  antériorité ne repose que sur la date de commit. Les seuils, eux, sont ceux
  d'X3 publiés le 2026-08-12, antérieurs par un chemin indépendant.
- **`planesbench.rs` n'est typé par aucune machine de développement.** Le
  câblage `e1v` et `nullk` a été écrit sans compilateur ; il compile, mais rien
  ne le protège d'une prochaine dérive.
- **Deux bras du contrôle sortent de leur plage publiée** par le haut de 1 à
  1,5 % (`Planes12x` 2,01 contre [1,95–1,99], `Golay70 v1` 1,34 contre
  [1,29–1,32]). Dérive **inter-run**, l'intra-run étant de 0,13 %. Aucun verdict
  n'en dépend ici ; un run publiant à 1 % près devrait l'expliquer.
- **La ligne V0 de `nullk` imprimait `0.0e0`**, ce qui se lit comme un accord
  parfait alors que ce bras n'est pas comparé. Corrigé après le run ; le journal
  de CE run porte la note.
- **`planesbench` nu inclut désormais `e1v` et `nullk`** — un transcodage et
  ~1 Go de flux de plus. Le plan de phases passe par `LLVQ_BENCH_ARMS` et n'est
  pas concerné ; un run nu, si.

## 8. Ce que la session apprend, au-delà d'elle

**Un compte d'octets ne prédit pas un temps, et un plancher non mesuré fausse
toute stratégie.** Quatre tentatives ont visé le tiers du budget que le format
peut atteindre, en ignorant les 45 % qu'il ne peut pas. Le chiffre qui l'aurait
dit coûtait 0,77 $ et un noyau de trente lignes.

**Et la règle qui sort de la nuit, pour tout portage futur** : une transcription
porte les gardes de son original sans porter les hypothèses qui les rendaient
suffisantes. Le seul dispositif qui l'attrape est de faire **exécuter le texte
lui-même** contre une référence indépendante — `tests/host_e1v.cpp` le fait
maintenant pour E1v, et les quatre autres `*_decoder_matches_rust.rs` le
faisaient déjà pour les layouts servis.
