# C1 prêt à tirer — layout Planes14, préparation du 2026-08-06

> État : **tout le code de C1 est écrit, testé et passé en revue adversariale
> sur le Mac ; il ne manque que le run sur carte (~0,2 $)**. Ce doc est la
> passation pour la session qui le lancera. Spec, verdicts et généalogie :
> [`pistes-format-vram-2026-08-05.md`](pistes-format-vram-2026-08-05.md)
> (barreau E1a) et [`verdicts-lot-b-2026-08-06.md`](verdicts-lot-b-2026-08-06.md).

## Ce qui existe (fichiers neufs, non commités, aucun conflit avec le lot A)

| fichier | rôle |
|---|---|
| `llvq-artifact/src/runtime.rs` (+167, ajouts purs) | `PlanesBlocks`, `transcode_planes14`, décodeur CPU de référence |
| `llvq-artifact/tests/planes14_format.rs` | 8 tests, dont le **sweep intégral des 150 681 600 blocs** du scellé (Planes14 == Slot32 au bit, ~50 s, relancé par le relecteur) et les tests d'offsets gelés/canonicité |
| `llvq-cuda/kernels/llvq_planes.cuh` | décodeur GPU : fenêtre 4 mots, sh ∈ {0,16} (aucun shift dégénéré, borne structurelle), arbre de sélection sans indexation dynamique |
| `llvq-cuda/kernels/planes.cu` | `tv_planes` — grille/staging/accumulation identiques à `tv_slot` |
| `llvq-cuda/src/bin/planesbench.rs` (+ `src/planes14_host.rs`, `include!`-é) | banc **3 bras** slot/planes/f16, protocole de `bin/matvec` reproduit point par point (7 rounds/2 jetés, bras entrelacés, médiane des rapports round par round, vérif ligne à ligne f64 à 1e-5, refus de spill, comptabilité par bras) |
| `llvq-cuda/tests/planes_decoder_matches_rust.rs` + `tests/host_planes.cpp` | le texte kernel exact compilé par clang++ sur le Mac et diffé champ par champ contre la référence Rust — les fautes de syntaxe/type sont attrapées ici, pas sur carte |

Vérifié en revue : mutants tous tués (6 posés, 4 rejoués indépendamment —
dont M6, le bug *cohérent* writer+reader que seul le test d'offsets gelés
attrape) ; sonde adversariale indépendante empaquetée à la main depuis la
spec ; clippy 0 warning sur darwin **et** x86_64-linux ; `lib.rs`/`gpu.rs`/
`matvec.rs` intacts ; `planesbench` compile sans le delta de `runtime.rs`
(découplage vérifié contre `git show HEAD:`).

## Comment tirer (session lot A, après son branchement)

1. **Commiter ces fichiers AVANT le `publish` de l'image** : `planesbench`
   est du code hôte — l'inclure dans le rebuild déjà nécessaire au lot A
   évite un second rebuild de 40-70 min. (Les `.cu`/`.cuh`, eux, restent
   itérables par `LLVQ_KERNEL_DIR` sans rebuild.)
2. Point de raccord unique (2 lignes) : dans `build()` de `planesbench.rs`,
   remplacer `planes14_from_slot32(&rt, &table)` par `transcode_planes14`
   de `llvq-artifact` et supprimer `src/planes14_host.rs` (sa copie de
   travail). Les deux implémentations sont prouvées équivalentes — le swap
   est cosmétique et peut attendre.
3. Lancer `planesbench` sur L40S sur le modèle publié (~0,2 $, go préalable).

## Grille de lecture du run

- **Critère d'acceptation (plan d'action)** : `tv_planes` ≥ 0,95× `tv_slot`
  à −0,71 b/poids thesis (4,80 contre 5,51) et qualité **identique par
  construction** (bijection prouvée). Au-dessus de 1,0× : les plans binaires
  dominent Slot32 strictement et deviennent le layout de référence.
- Sous 0,95× : lire l'écart avant de conclure — les suspects sont les
  fenêtres sh=16 (un bloc sur deux) et la perte du gather 5 mots ; l'échec
  franc tue l'échelle E1 proprement pour 0,2 $.
- Risques carte-seulement listés par l'implémenteur et validés en revue :
  spill de `tv_planes` (le banc refuse au démarrage), dérive de registres
  des bras témoins dans la nouvelle unité de traduction (rapport imprimé au
  démarrage — ne jamais comparer ces ms à celles de `bin/matvec`),
  coalescing réel des fenêtres, padding de fin de flux, anti-cache.
