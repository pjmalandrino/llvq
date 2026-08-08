# LLVQ — contexte projet (passation de session)

> Ce fichier est chargé automatiquement par Claude Code. Il contient tout ce
> qu'une nouvelle session doit savoir pour reprendre le travail sans relire
> l'historique.

> 🧭 **Reprise de session** :
> [`docs/rapport-etat-2026-08-07.md`](docs/rapport-etat-2026-08-07.md) — la
> photographie d'ensemble. Le noyau fusé **tourne dans le modèle** : le
> Qwen3-4B publié rend **88,4–88,5 tok/s dans 2,60 Go** contre 43,5 tok/s dans
> 8,04 Go au chemin dense, à qualité identique au bit près (deux invocations,
> `campagne-finale-bras4` et `nuit-planes12x-q8` — une plage, pas un point,
> cf. §3). Puis
> [`docs/echelle-4b-8b-2026-08-08.md`](docs/echelle-4b-8b-2026-08-08.md) — le
> point d'échelle, la question centrale du projet : **le déficit du 2 bits
> fond quand le modèle grossit** (ppl ×1,385 → ×1,220, MMLU −14,73 → −10,56 pp
> du 4B au 8B), et l'écart au 4 bits fond deux fois plus vite. Puis
> [`docs/verdicts-nuit-2026-08-07.md`](docs/verdicts-nuit-2026-08-07.md) —
> embedding q8 en production, overlay `Planes12x` validé au banc, design C
> réfuté. Puis [`docs/format-noyau.md`](docs/format-noyau.md) — l'état du
> noyau, les quatre pièges de mesure GPU chèrement acquis, et l'échelle
> bits↔vitesse.
> Historique : [`docs/passation-lot-a-2026-08-06.md`](docs/passation-lot-a-2026-08-06.md)
> · [`docs/passation-2026-08-05.md`](docs/passation-2026-08-05.md)
> · [`docs/passation-2026-07-31.md`](docs/passation-2026-07-31.md).
>
> ⚠️ **Deux phrases de ce fichier ont survécu à leur propre démenti, et il
> faut savoir ce qu'elles disaient** — c'est le motif récurrent du projet :
> une affirmation vraie le jour où on l'écrit, laissée en place pendant que
> la mesure la retourne trois lignes plus bas.
>
> 1. « **Le noyau n'est pas encore branché** ». Faux depuis le 2026-08-06. Le
>    noyau *est* branché, et il est mesuré dans le modèle par
>    **`bin/fusedrun`**, qui charge le fichier scellé par `fused_cuda::load`
>    (`llvq-llm/src/bin/fusedrun.rs:115`) et exige les mêmes tokens que le
>    chemin dense. Ce qui reste dense, c'est **`bin/run`** — la démo de
>    génération, encore sur `sealed::load` (`llvq-llm/src/bin/run.rs:45`) :
>    un binaire non porté, pas un chantier de noyau. Dire « pas branché »
>    parce que `bin/run` ne l'appelle pas, c'est confondre la démo et le
>    produit.
> 2. « **Le point de décision suivant est C1** ». C1 a été gagné, branché et
>    mesuré le même jour : le layout de production est **`Planes14`**, défaut
>    de `LLVQ_FUSED_LAYOUT` (`llvq-llm/src/fused.rs:68`). Le chantier ouvert
>    n'est plus le format — l'échelle des formats est close, cf. §3 — mais
>    **la qualité** (−14,73 pp de MMLU au 4B, §3ter) et un arbitrage produit
>    entre `Planes14` (vitesse) et `Planes12x` (bits).

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

**Huit crates** (`ls */Cargo.toml`, et les huit `members` de `Cargo.toml`) :

```
llvq-core/     Golay [24,12,8] + Λ₂₄ + couches. ZÉRO dépendance, forbid(unsafe).
llvq-search/   Recherche NN exacte, classes, moteur générique m≤13, indexage, packing.
llvq-quant/    Spherical GPTQ : algèbre dense, boucle par blocs, quantifieurs.
llvq-artifact/ Le format .llvq : writer, reader, décodeur. ZÉRO dépendance.
llvq-metal/    Micro-bancs GPU (macOS) : plomberie Metal, coût du décodage.
llvq-cuda/     Le noyau fusé sur NVIDIA : source CUDA compilée par NVRTC au
               démarrage, bancs matvec/planes/rotation. (cudarc, cfg(linux) seul)
llvq-llm/      Côté modèle : passe avant observable, corpus, perplexité,
               chemin fusé dans le modèle. (candle)
llvq-bench/    Débit-distorsion, débit encodeur, coût du décodage.
```

> 🕳️ **`llvq-cuda` manquait à cette liste depuis sa création** — sept crates
> annoncés pour huit réels. Ce n'est pas cosmétique : c'est le crate où vit le
> noyau qui porte la thèse d'ingénierie du projet, et une session qui lit
> cette liste conclut qu'il n'existe pas.

**`llvq-core`, `llvq-search`, `llvq-quant`, `llvq-artifact` et `llvq-bench`
restent sans dépendance externe.** Lire un modèle quantifié ne doit pas exiger
un runtime de tenseurs : l'arbre complet de `llvq-artifact` fait 3 crates,
contre **261 paquets distincts** pour `llvq-llm` — 291 avec
`metal,fast-linalg`. `llvq-llm` en a — candle, tokenizers, hf-hub, parquet —
parce qu'il faut bien charger et exécuter un modèle ; `llvq-cuda` en a une,
`cudarc`, invisible au résolveur ailleurs que sur Linux (voir le long
commentaire de `llvq-cuda/Cargo.toml`, qui documente les trois murs de son
`build.rs` sur macOS).

> 🕳️ **Le « 690 » qu'affichait ce fichier était un nombre de lignes de
> `cargo tree`, pas un compte de paquets.** Corrigé au README le 2026-08-03
> (`docs/audit-publication-2026-08-03.md`, ligne « Corriger `README.md:122-124` » ;
> `README.md:277-280` porte les bons chiffres depuis). Même motif que
> partout ailleurs : un chiffre repris d'une sortie d'outil sans vérifier ce
> que l'outil comptait.

⚠️ **`unsafe` n'est plus l'exclusivité de `llvq-llm`, et l'affirmation
contraire dormait ici.** C'était vrai quand `llvq-llm` était le seul crate à
parler à un accélérateur ; `llvq-metal` puis `llvq-cuda` l'ont périmée.
Compté le 2026-08-08 (`grep -rn unsafe --include='*.rs'`, crate par crate) :

| crate | `unsafe` | statut |
|---|---|---|
| `llvq-core`, `llvq-search`, `llvq-quant`, `llvq-bench` | 0 | `#![forbid(unsafe_code)]` |
| `llvq-artifact` | 0 | `#![forbid(unsafe_code)]`, **posé le 2026-08-08** — le crate n'en contenait aucun mais rien ne l'interdisait, alors que c'est celui dont la doc de module plaide qu'on doit pouvoir le lire de bout en bout |
| `llvq-metal` | 12 mentions | `read<T>` d'un buffer GPU typé (`lib.rs:106`) + ses appelants |
| `llvq-cuda` | 13 mentions | `b.launch(cfg)` de cudarc (`gpu.rs`), `disable_event_tracking` |
| `llvq-llm` | 11 mentions | mmap safetensors (`loader.rs:62`), `alloc`/`launch` (`fused_cuda.rs`) |

**Ce qui est faux, c'est l'exclusivité — pas la permission ni l'intention.**
Le cœur mathématique reste `forbid(unsafe_code)`, et c'est ça qu'il fallait
protéger. Les trois crates qui en contiennent le font tous pour la même
raison — **franchir une frontière que le compilateur ne peut pas prouver**
(mmap, lancement de noyau, lecture d'un buffer device) — et aucun n'en met
dans du code mathématique. La règle à écrire est donc : *`unsafe` est
autorisé aux frontières matérielles, interdit partout ailleurs.*

🚨 **`faer` : « on ne l'a pas pris » était faux, et faux d'une manière qui
compte.** Il est déclaré à `llvq-quant/Cargo.toml:11` (version 0.24,
`optional = true`) derrière la feature `fast-linalg`
(`llvq-quant/Cargo.toml:19`), et **c'est lui qui a factorisé les modèles
publiés** : la ligne de commande de production porte
`--features metal,fast-linalg` (`README.md:492`, `docs/run-de-nuit.md:8`), et
l'image CUDA des runs 8B/32B l'a figée dedans (`docs/spec-lot-a-2026-08-05.md:26`).
Un lecteur du paragraphe précédent en déduisait que les chiffres du projet
sortent d'une algèbre maison ; ils sortent de `faer`.

> ⚠️ **Une nuance de provenance, et elle est due** : pour le run 4B scellé
> lui-même, la présence de `fast-linalg` n'est **pas traçable a posteriori**
> — le garde-fou qui l'imprime est postérieur au run et le log ne porte pas
> de profil de phases. La feature existait au commit et la durée est
> compatible, mais rien ne le *prouve*, et aucune commande ne le récupère
> (`docs/fiche-4b.md:556`). Sans effet sur les chiffres — les deux chemins
> sont bit-identiques, c'est tout l'objet de `both_factorizations_agree` —
> mais on ne dit pas « prouvé » quand on veut dire « très probable ».

L'argument d'origine survit **intact**, mais il porte sur le *défaut*, pas
sur l'existence : ce dont l'Algorithme 1 a besoin — Cholesky, inverse
triangulaire, produit triangulaire — tient en ~150 lignes verrouillées par
des identités exactes (`llvq-quant/src/linalg.rs`), l'API de `faer` bouge
beaucoup d'une version à l'autre, et `llvq-quant` **compile et passe sans
lui**. Ce que la mesure a ajouté, c'est que le chemin sans dépendance n'est
pas utilisable en production : ~1 G multiply-add/s, soit **~40× plus lent**
pour un résultat bit-identique (28,4 s contre 0,7 s sur un bloc de
Qwen3-0.6B — `llvq-llm/src/bin/smoke.rs:474-486`, qui l'avertit
bruyamment). D'où la forme actuelle, qui est la bonne :

- `faer` derrière un drapeau, donc l'audit du cœur reste possible ;
- le chemin maison gardé comme **référence de vérification**, pas comme
  folklore — `both_factorizations_agree`
  (`llvq-quant/tests/g5_gptq.rs:824`) exige que les deux produisent le
  **même facteur**, ce qu'aucune identité seule n'attrape (un facteur
  *valide* de la mauvaise chose passerait) ;
- et le drapeau **allumé partout où on mesure ou où on paie du matériel**.

> ⚠️ Conséquence pratique, à ne pas redécouvrir : lancer un run sans
> `--features fast-linalg` ne donne pas un résultat faux, il donne le bon
> résultat 40× trop tard. Sur GPU loué, ça se paie en dollars.

Commandes :
```bash
# ⚠️ DEUX BOUCLES, DEUX ORDRES DE GRANDEUR — ne pas les confondre.
cargo test                                   # boucle rapide : les tests lourds sont
                                             #   #[cfg_attr(debug_assertions, ignore)]
cargo test --release -- --include-ignored    # suite complète : DIZAINES DE MINUTES
cargo run --release -p llvq-bench --bin llvq-bench   # tableau qualité
cargo run --release -p llvq-bench --bin encbench      # débit encodeur, 1 cœur
cargo run --release -p llvq-bench --bin betasweep     # sensibilité de β (G4)
cargo run --release -p llvq-bench --bin decbench      # coût du décodage (G6)
cargo run --release -p llvq-bench --bin classhist     # histogramme par classe (lot B)

# GPU (macOS) — le noyau et la thèse
cargo run --release -p llvq-metal --bin thesis        # un token, 252 matrices, LLVQ vs FP16
cargo run --release -p llvq-metal --bin matvec        # une couche, tous les layouts
cargo run --release -p llvq-metal --bin decreal       # coût du décodage seul, blocs réels

# GPU (Linux + CUDA, donc carte louée) — c'est là que tournent les chiffres publiés.
# llvq-cuda est gaté par cfg(target_os = "linux"), PAS par une feature : sur Mac
# ses bins n'existent tout simplement pas (cudarc ne compile pas, cf. son Cargo.toml).
cargo run --release -p llvq-cuda --bin planesbench -- <model.llvq>  # banc N bras, tous les layouts
cargo run --release -p llvq-cuda --bin preflight                    # SM, registres, spill, L2
cargo run --release -p llvq-llm  --features cuda --bin fusedrun     # LE noyau DANS le modèle
#   LLVQ_FUSED_LAYOUT=planes14|slot32   (défaut planes14)
#   LLVQ_EMBED=f16|q8                   (défaut f16 ; q8 = l'embedding int8 en prod)
#   LLVQ_TIME_PHASES=1                  (profil par phase, hors protocole publié)

# côté modèle (Metal recommandé : ~7× le CPU sur M3 Max)
cargo run --release -p llvq-llm --features metal --bin oracle
cargo run --release -p llvq-llm --features metal --bin ppl -- 4096 999 metal
# ⚠️ ppl tourne en f32 par défaut, mmlu et run en f16. Chaque métrique est
# cohérente en interne ; les COMPARER exige le même dtype des deux côtés :
#   LLVQ_DTYPE=f16 cargo run … --bin ppl -- 4096 12 metal
# ppl sait scorer le FICHIER SCELLÉ (4ᵉ arg), donc exactement l'objet que mmlu
# score. Les deux bras ne sont comparables que si l'EMPREINTE DE TOKENS
# imprimée sur la ligne de résultat est la même :
#   LLVQ_DTYPE=f16 cargo run … --bin ppl -- 4096 12 metal ~/qwen3-4b-llvq.bin

# ⚠️ Les deux A/B que ce bloc réclamait ONT ÉTÉ LANCÉS (lot B, nuit du 05
# au 06 — docs/verdicts-lot-b-2026-08-06.md §B1). Ne pas les redemander.
#   LLVQ_CALIB_SEED={1,2,3} → 20,6239 / 20,4709 / 20,7687 sur 3 blocs :
#     σ ≈ 0,15 ppl ≈ 0,7 %. C'est la première barre d'erreur du projet.
#   LLVQ_DAMPING={3e-3,1e-2,3e-2} → 20,6740 / 20,6643 / 20,6014 : écart
#     0,35 %, SOUS 1σ. Effet nul — exactement ce que le code prédisait.
cargo clippy --all-targets                   # doit rester à zéro warning
```

> 🚨 **« Suite complète, ~45 s » était faux d'un ordre de grandeur, et ça se
> paie en temps perdu.** Chronométré le 2026-08-08 sur le M3 Max de dev :
> après **17 minutes**, `cargo test --release -- --include-ignored` n'avait
> pas fini le **premier crate des huit** (`llvq-artifact`) — il en était au
> 5ᵉ de ses binaires de test, avec
> `the_sealed_artifact_planes12x_overlay_is_exact` qui tenait la suite depuis
> près de dix minutes. La bonne échelle est donc **la dizaine de minutes au
> moins, pas la dizaine de secondes**. Chiffre volontairement donné en ordre de
> grandeur : la suite grossit à chaque layout — trois sweeps intégraux sont
> arrivés en une semaine — et un nombre précis ici serait périmé au prochain
> commit. *(Provenance honnête : la machine portait un second `cargo test`
> concurrent pendant la mesure, donc ce temps est majoré. Mais il faudrait un
> facteur ~25 de contention pour ramener la suite aux 45 s annoncées, ce que
> deux processus ne produisent pas.)*
>
> **La raison est structurelle, et il faut la connaître avant de râler.** Les
> tests les plus lourds ne sont pas lents par négligence : ils **balayent le
> fichier scellé de 981 Mo en entier** — 150,7 M blocs, bijection et overlay
> prouvés bloc par bloc pour `Planes14`, `Planes12x` et `Golay70`. C'est
> exactement la létalité que le §5 réclame : un sweep intégral attrape ce
> qu'un échantillon laisse passer. Ils se **sautent proprement** quand
> `~/llvq-q4b.llvq` n'est pas sur la machine
> (`llvq-artifact/tests/planes12x_format.rs:305-315`, `SKIP:` sur stderr) —
> donc la même commande est rapide sur une machine nue et longue ici. ⚠️ Ne
> jamais lire un « tout vert » rapide comme une preuve : vérifier d'abord
> qu'il n'est pas rapide *parce que* l'artefact manque.
>
> **En pratique** : `cargo test` (debug) pour la boucle de développement, la
> suite complète avant un commit qui touche un format ou l'indexage.

> ✅ **La règle opérationnelle qui sort du lot B, et qui vaut pour tout run
> futur : sur un run unique de 3 blocs, tout effet sous ~1,5 % (2σ) est du
> BRUIT.** Le protocole d'A/B à 3 blocs, adopté en §6 pour ne plus changer
> deux variables à la fois, avait un angle mort — il donnait un Δ sans
> savoir ce qu'un Δ vaut. Il le sait maintenant. Ne plus publier ni décider
> sur un écart de 3 blocs sans le confronter à ce seuil.
>
> ⚠️ Trois réserves, parce qu'un σ mal cité est pire que pas de σ. (1) Il est
> estimé sur **3 points** — c'est un ordre de grandeur, pas un écart-type de
> précision. (2) Il vaut pour **3 blocs de Qwen3-0.6B en perplexité** ;
> `LLVQ_CALIB_SEED` ne déplace que les offsets de fenêtre, et un mécanisme
> qui pèse 0,04 % à 3 blocs peut peser 7 % à 36 (`docs/fiche-4b.md:537`) —
> **ce σ n'est pas la barre d'erreur de l'objet publié.** (3) L'évaluation,
> elle, est déterministe à empreinte de tokens identique : un Δ mesuré entre
> deux fichiers sur la même empreinte n'a pas ce bruit-là (c'est pourquoi le
> +4,75 % du swap L≤4 est réel, cf. §3).

## 3. État — les 7 gates sont verts

> 🕳️ Ce titre disait « 5 gates sur 7 » alors que les huit lignes du tableau
> ci-dessous portent ✅ depuis le 2026-08-06. Un en-tête qui contredit son
> propre tableau, dans le fichier chargé au démarrage de chaque session :
> exactement le genre de décalage qui oriente une session entière vers un
> chantier déjà clos.

| Gate | Contenu | Statut |
|---|---|---|
| G1 | Invariants Λ₂₄/Golay (nombre de baisers 196 560, Shell(3), série thêta) | ✅ |
| G2 | Recherche NN exacte m ≤ 3 vs force brute | ✅ |
| G2b | Moteur générique de classes m ≤ 13 | ✅ |
| G3 | Indexage bijectif 48 bits (format v1) | ✅ |
| G4 | Source gaussienne 2 bits/dim : **92,23 % de rétention** | ✅ |
| 2c | Encodeur : 639 µs/bloc/cœur (5,5× le départ) | ✅ |
| G5 | Spherical GPTQ + pipeline LLM | ✅ **Wiki 16,9617 à 2,1696 bits pesés** sur Qwen3-4B (QTIP : 17,04 à 2,000), fichier scellé **`leech1c12`** — cap 12, 47 bits d'index + 1 de gain = **48 bits/bloc**, 2,0702 b/poids effectifs (note de provenance dans la section G5). Vert avec réserve : on passe de 0,08 point, à 8,5 % de bits en plus |
| G6 | Noyau fusé (déquant + matvec) | ✅ **la thèse est mesurée sur le modèle entier : 2,03–2,09×** — `bin/thesis`, un token des 252 projections, un command buffer par format, froid par construction, **1 105 920 lignes vérifiées** contre référence f64 : FP16 21,69 ms contre **10,46 ms** ; 41,6 → **78,2 tok/s** avec le lm_head f16. ⚠️ **une plage, pas un point.** Le 2,07× publié le 2026-08-01 est le haut d'une plage [2,029 ; 2,080] mesurée sur trois invocations du banc à deux bras ; le banc à sept bras en rend 2,03× · 2,06× · 2,09× sur trois autres. Les octets, les b/poids et les pires erreurs sont identiques au chiffre à chacune — seuls les temps bougent, et ils bougent ensemble sur les deux bras. Une troisième décimale sur ce rapport n'a donc pas de contenu ([`docs/mesures/thesis-temoin-2026-08-04.txt`](docs/mesures/thesis-temoin-2026-08-04.txt)), et ces deux temps sont ceux du **run publié le 2026-08-01** : le run à sept bras du 2026-08-05 cité plus bas rend 21,728 / 10,496 ms **sur les mêmes deux bras**, soit **2,03× [2,03–2,10]**. Deux invocations du même objet, pas deux mesures contradictoires — ne pas les soustraire. Sur une couche isolée (`bin/matvec`, protocole froid à 4 copies) : **2,2×**. Le layout est `Slot32` — offsets fixes `[classe 9][gain 1][smask 24][m₁..m₄@24]`, zéro divergence. **Échelle bits↔vitesse, un seul protocole et une seule comptabilité d'octets** (`bin/thesis` du 2026-08-05, 7 rounds dont 2 jetés, tous les bras dispatchés à chaque round dans le même ordre ; payload + bases + queue f32 + échelles de ligne f32 ; le « vs FP16 » est la **médiane du rapport formé round par round** avec sa plage sur les 5 rounds gardés — surtout pas un quotient de deux minima, qui mêlerait deux rounds n'ayant jamais coexisté ; millisecondes dans [`docs/mesures/k1-metal-2026-08-05.txt`](docs/mesures/k1-metal-2026-08-05.txt)) : FP16 16,000 b/poids, 1,00× · **`Slot32` 5,510, 2,03× [2,03–2,10]** · `Flat32` 5,256, 0,91× [0,91–0,91] · `Grouped32` 3,498, 0,69× [0,68–0,69]. (L'ancienne échelle « 3,35 nested = 0,68× ; 4,54 Flat32 = 0,90× ; 5,51 Slot32 = 2,07× » mélangeait plusieurs comptabilités d'octets dans une même liste — la faute que ce run supprime.) Transcodeur 5 layouts bit-exacts, ~25 mutants tués. ✅ **BRANCHÉ ET MESURÉ le 2026-08-06** (lot A, [`docs/passation-lot-a-2026-08-06.md`](docs/passation-lot-a-2026-08-06.md)) : `bin/fusedrun` sur L40S rend **47,0 tok/s contre 43,5 dense, 3,28 Go contre 8,04 (÷2,45)**, 88 tokens gloutons identiques avant un tie-break. Et la campagne à quatre bras a tranché : **sur un 4B le 4 bits nous domine partout sauf le disque** — MMLU 70,04 % pour l'AWQ officiel contre 55,59 % pour nous, ppl ×1,105 contre ×1,384 ([`docs/mesures/a4-campagne-2026-08-06.txt`](docs/mesures/a4-campagne-2026-08-06.txt)). ⚠️ La comparaison VRAM se dit en **b/param modèle entier, embedding compris** — jamais « 5,51 contre 4,50 » (deux dénominateurs, et deux quatre-bits confondus : l'AWQ mesuré pèse **5,30** dans son moteur, le 4,50 est le MLX q4 absent de la campagne) — [`docs/errata-rapport-lot-a-2026-08-06.md`](docs/errata-rapport-lot-a-2026-08-06.md). ✅ **C1 GAGNÉ ET BRANCHÉ le même jour** : le layout VRAM de référence est **`Planes14`** — plans de bits binaires au lieu du one-hot, stride uniforme 14 o, sans bases — **1,14× [1,14–1,15] plus rapide que `Slot32` à contenu décodé identique, 4,804 b/poids contre 5,510** au banc 3 bras (2,16× vs FP16 ; Go/s constants : le temps tombe exactement comme les octets ; [`docs/mesures/c1-planesbench-2026-08-06.txt`](docs/mesures/c1-planesbench-2026-08-06.txt)), et dans le modèle : **48,7 tok/s / 2,96 Go (÷2,72)**, bascule `LLVQ_FUSED_LAYOUT`, contrôle slot32 reproduit à l'identique — 47,0/3,28, divergence au même token 89 ([`docs/mesures/planes14-fusedrun-2026-08-06.txt`](docs/mesures/planes14-fusedrun-2026-08-06.txt)). En modèle entier : **5,88 b/param projeté avec l'embedding f16, 5,11 avec l'embedding int8** (errata du lot A) — puis **5,15 mesuré en production** une fois `LLVQ_EMBED=q8` livré, **sous les 5,30 de l'AWQ réel** dans les deux comptabilités. ⚠️ Le plafond L ≤ 4 sec est **mort en qualité** (swap mesuré : **+4,75 % de ppl**, repasse au-dessus de QTIP — lot B) ; il est remplacé par l'**overlay épars**, qui atteint le même point de bits à **qualité exacte**. **L'échelle des formats est close depuis le 2026-08-07 — voir le tableau juste sous celui-ci.** Détail : [`docs/format-noyau.md`](docs/format-noyau.md), [`docs/pistes-format-vram-2026-08-05.md`](docs/pistes-format-vram-2026-08-05.md), [`docs/verdicts-nuit-2026-08-07.md`](docs/verdicts-nuit-2026-08-07.md) |

### L'échelle des formats runtime — quatre points mesurés, un écarté (2026-08-07)

Le format que le noyau lit en VRAM n'est pas celui du fichier. Quatre layouts
ont été portés sur CUDA et mesurés **dans un seul processus, un seul
protocole, une seule comptabilité d'octets** (banc à 5 bras, 7 rounds dont 2
jetés, rapports formés round par round — jamais un quotient de deux minima) ;
chacun est vérifié ligne à ligne contre une référence f64 sur les **1 105 920
lignes** du modèle publié. Journal :
[`docs/mesures/e2-golay70-bench-2026-08-07.txt`](docs/mesures/e2-golay70-bench-2026-08-07.txt).

| layout | b/poids payload | vs FP16 | Go/s | statut |
|---|---|---|---|---|
| `Slot32` (one-hot) | 5,510 | 1,87× [1,86–1,88] | 428 | remplacé, gardé en repli (`LLVQ_FUSED_LAYOUT=slot32`) |
| **`Planes14`** (plans de bits) | **4,804** | **2,14× [2,11–2,15]** | 425 | ✅ **en production, défaut** |
| **`Planes12x`** (overlay épars) | **4,342** | 1,98× [1,95–1,99] | 356 | ✅ validé au banc, qualité **exacte** — ⚠️ **pas dans le modèle** (voir ci-dessous) |
| `Golay70` (E2) | 3,589 | **1,31× [1,29–1,32]** | 195 | ❌ **écarté** — sous le critère de 1,6× posé d'avance |

⚠️ **« Validé au banc » n'est pas « livré », et la différence est vérifiable
en une commande.** `Planes12x` et `Golay70` ont leur transcodeur
(`llvq-artifact/src/runtime.rs`) et leur noyau (`llvq-cuda`), mais l'énum
`FusedLayout` que lit le modèle n'expose que **`Planes14` et `Slot32`**
(`llvq-llm/src/fused.rs:58-73` — et elle **refuse** toute autre valeur plutôt
que de retomber en silence sur le défaut, parce qu'un A/B qui se trompe de
bras en silence est pire qu'un A/B qui échoue). `grep -rn planes12x llvq-llm`
rend zéro. Passer `Planes12x` en production est donc du travail restant, pas
un réglage.

**Trois choses que cette échelle établit.**

1. **Le one-hot ne payait rien, il coûtait.** Le recodage binaire de
   `Planes14` rend le format **plus petit ET plus rapide** — les Go/s sont
   constants (428 → 425), donc le temps tombe exactement comme les octets.
   C'est la découverte de la semaine, et elle contredit l'intuition qui avait
   fait choisir `Slot32` : on croyait acheter de la vitesse avec des bits.
2. **Le dilemme central du §3bis est levé.** « Le format qui va vite ne rentre
   pas mieux que du 4 bits ; le format qui rentre mieux ne va pas vite » était
   vrai sur `Slot32`/`Flat32`/`Grouped32`. `Planes14` et `Planes12x` sont
   devant `Slot32` **sur les deux axes à la fois**.
3. **Mais la courbe finit par se retourner, et E2 le prouve.** `Golay70`
   descend bien à 3,589 b/poids avec reconstruction exacte (le résultat de
   *format* tient, prouvé sur les 150,7 M blocs), et il s'écroule à 1,31× :
   le décodage à double coset borne le noyau **en calcul**, 195 Go/s
   effectifs. Le critère de 1,6× avait été posé **avant** la mesure — c'est ce
   qui permet de l'écarter sans discussion. ⚠️ Ne pas rouvrir E2 sans idée
   neuve sur le coût ALU (pistes notées, non poursuivies : spécialiser les
   warps par coset, payer le XOR seulement côté pair).

En **b/param modèle entier** — la seule comptabilité dans laquelle une
comparaison mémoire a un sens, cf. l'errata cité en G6 :

| point | b/param, embedding compris |
|---|---|
| `Slot32` + embedding f16 (avant C1) | 6,52 |
| `Planes14` + embedding f16 | 5,88 |
| **`Planes14` + embedding q8** | **≈ 5,15 mesuré** — sous les 5,30 de l'AWQ réel |
| `Planes12x` + embedding q8 | 4,69 — sous l'AWQ, à 4 % du MLX q4 |
| repères : AWQ w4 g128 réel · MLX q4 g64 | 5,30 · 4,50 |

**L'embedding int8 est en production** (`LLVQ_EMBED=q8`,
`llvq-llm/src/fused.rs:106`) : validé sans perte au lot B — ppl **16,9379**
contre 16,9415, soit −0,02 %, et MMLU dans le σ (55,44 contre 56,09) — puis
livré la nuit du 06 au 07. ⚠️ **Deux mécanismes, un seul quantifieur, et il
ne faut pas les confondre** : `bin/embedq` **écrit** un fichier à embedding
int8 (`q4b-e8.llvq`, **1,406 Go contre 1,771 — −365 Mo, −21 % à froid**),
tandis que `LLVQ_EMBED=q8` quantifie **au chargement**, par la même
fonction, et fait tomber le modèle sur la carte de 2,96 à **2,60 Go**. Le
gain disque et le gain VRAM sont donc réels tous les deux, mais ils ne
s'obtiennent pas par le même chemin — et un job peut très bien scorer l'un et
chronométrer l'autre (c'est le cas de la campagne finale du 4B : ppl et MMLU
sur `q4b-e8.llvq`, `fusedrun` sur le fichier à embedding f16).

> ⚠️ **Le saut de débit qui l'accompagne n'est pas ce qu'il paraît, et il ne
> faut pas en faire un titre.** Passer à q8 fait bondir le modèle de 48,7 à
> 88,4–88,5 tok/s, très au-delà du trafic du `lm_head` (~0,6 ms attendus). Le
> mécanisme est attribué : le noyau q8 remplace un chemin candle qui recopiait
> les **778 Mo du vocabulaire à chaque token** (~26 ms/token, le `TODO` est
> dans le code de candle et la copie est chronométrée,
> [`docs/mesures/phases-2026-08-07.txt`](docs/mesures/phases-2026-08-07.txt)).
> **Donc deux formulations, et il faut donner les deux** : ×2,03 contre le
> moteur de référence tel quel, **~×1,4 contre ce moteur corrigé**. Seuls
> ~2,9 ms/token viennent réellement du noyau Leech.

Résultat G4 mesuré (20 000 blocs, seed figée), face aux chiffres du papier
relus sur le PDF (Table 8, annexe H — celle qui nomme le codebook) :

| méthode | codebook | bits/dim | MSE | rétention |
|---|---|---|---|---|
| papier, spherical shaping | `Λ₂₄(13)` | 2,000 | 0,084 | 89,37 % |
| papier, shape–gain 0 bit de gain | `norm(Λ₂₄(13))` | 2,000 | 0,085 | 89,12 % |
| papier, shape–gain 1 bit de gain | `norm(Λ₂₄(12))` | 2,000 | 0,078 | 92,14 % |
| **notre shape–gain 0 bit de gain** | `norm(Λ₂₄(13))` | 1,9999 | 0,0850 | **88,90 %** |
| **notre spherical shaping (β\* = 0,350)** | `Λ₂₄(13)` | 1,9999 | 0,0775 | **92,23 %** |
| Shannon | — | 2,000 | 0,0625 | 100 % |

> ⚠️ **Les lignes shape–gain ont bougé le 2026-08-01 (§A5).** Le banc codait
> le gain sur la **projection** `⟨x,v̂⟩` — l'optimum à direction fixée — alors
> que `LeechShapeGain` code la **norme** du bloc. Même reconstruction, même
> formule d'erreur, scalaire différent : le banc mesurait un quantifieur
> strictement meilleur que celui qui a produit l'artefact, de `2/(1+cos θ)`
> par bloc. Les chiffres ci-dessus sont ceux du quantifieur **livré** ; la
> borne à gain optimal reste imprimée en dessous de chaque ligne par le banc.
> `tests/bench_matches_production.rs` épingle désormais le banc sur
> `LeechShapeGain::quantize`, bloc par bloc. Le spherical shaping n'a pas de
> code de gain, donc ses 92,23 % sont inchangés.
>
> 🔎 **L'écart sur le spherical shaping s'explique par β, pas par le
> codebook — ne pas le revendiquer comme une victoire.** Notre shape–gain
> 0 bit reproduit le papier à 0,2 point près (88,90 vs 89,12), donc protocole
> et codebook sont bons. Mais notre spherical shaping le dépasse de presque
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

## 3ter. MMLU — ce que la perplexité cachait (2026-08-01, remesuré le 08-02, puis sur carte le 08-06/07/08)

> 🚨 **Les deux chiffres de cette section sont des moyennes MACRO ; le papier
> rapporte du MICRO. Ils ne sont pas comparables et il faut les remesurer.**
> Corrigé dans le harnais le 2026-08-01 (§A1).
>
> `bin/mmlu` tirait `limit` questions **par matière** puis divisait
> `Σright / Σtotal` globalement. Avec 40 par matière et 57 matières, chaque
> terme pèse pareil : la division est algébriquement la moyenne **non
> pondérée** des 57 taux. Or le split MMLU est très déséquilibré —
> `professional_law` 1 534 questions, `abstract_algebra` 100 — donc le macro
> sur-pondère les petites matières STEM d'un facteur ~2,5.
>
> **C'est exactement là que le 2 bits fait ses dégâts** (voir le profil par
> matière ci-dessous), donc le biais frappe beaucoup plus fort le bras
> quantifié que la baseline. Conséquence directe : l'argument « les deux
> écarts pointent en sens opposés, donc ce n'est pas un décalage de protocole
> qui s'annulerait » **est faux** — un échange macro/micro produit
> précisément cette signature-là.
>
> ✅ **Le second facteur de confusion suspecté — ppl en F32, MMLU en F16 — a
> été mesuré et il est nul** (§A2, 0,1 % d'écart). L'agrégation était donc bien
> la seule piste restante.

**Remesuré en micro le 2026-08-02** (§A1), f16 des deux côtés, mêmes 2 280
questions et même graine que le run publié. Log par matière conservé cette
fois : [`docs/mmlu-micro-2026-08-02.log`](docs/mmlu-micro-2026-08-02.log).

Harnais maison (`bin/mmlu`), **dans notre pipeline, sur le fichier scellé** —
pas un checkpoint déquantifié dans le moteur d'un tiers, parce que MLX et
notre `bin/run` divergent au 5ᵉ token sur les mêmes poids.

| | micro (= papier) | macro | papier |
|---|---|---|---|
| FP16 | **70,42 ± 1,28** | 72,85 | 70,2 |
| LLVQ 2 bits | **56,09 ± 1,36** | 57,59 | 60,7 |
| **chute** | **−14,33 pp** (79,7 % retenus) | −15,26 pp | −9,5 pp (86,5 %) |

> ✅ **Le contrôle passe sur les deux bras** : les macros ressortent à 72,85 et
> 57,59, *exactement* les chiffres publiés la veille. La seule différence entre
> les deux colonnes est l'agrégation, sans autre variable.

> 🚨 **Ces deux valeurs sont celles de l'ère Metal ; le chiffre de référence
> du 4B est aujourd'hui `70,32 / 55,59`.** Le raisonnement de cette section
> est intact — c'est *lui* qui compte, et il est reproduit à la ligne près par
> la remesure — mais les **valeurs** ont bougé quand tout est passé sur carte
> louée, dans un seul harnais, avec l'empreinte de tokens imprimée des deux
> côtés (`65dcd53655e8bfa5`). Les trois mesures qui font foi :
>
> | | FP16 | LLVQ 2 bits | chute | source |
> |---|---|---|---|---|
> | 4B, Metal, 08-02 | 70,42 ± 1,28 | 56,09 ± 1,36 | −14,33 pp | `docs/mmlu-micro-2026-08-02.log` |
> | **4B, L40S, 08-06** | **70,32 ± 1,28** | **55,59 ± 1,35** | **−14,73 pp** | [`mesures/a4-campagne-2026-08-06.txt`](docs/mesures/a4-campagne-2026-08-06.txt) |
> | 4B, L40S, embedding q8, 08-07 | — | 55,70 ± 1,35 | — | [`mesures/campagne-finale-bras4-2026-08-07.txt`](docs/mesures/campagne-finale-bras4-2026-08-07.txt) |
> | **8B, L40S, 08-08** | **76,08 ± 1,21** | **65,52 ± 1,31** | **−10,56 pp** | [`mesures/campagne-8b-qualite-2026-08-08.txt`](docs/mesures/campagne-8b-qualite-2026-08-08.txt) |
>
> **Trois lectures, dans l'ordre d'importance.**
>
> 1. **La baseline ne bouge pas** : 70,42 → 70,32, soit 0,10 pp, 0,08 σ. Le
>    harnais traverse un changement de backend, de carte et de dtype sans
>    dériver — c'est le contrôle qui rend les deux autres lignes lisibles.
> 2. **L'embedding q8 ne coûte rien en capacités** : 55,70 contre 55,59, dans
>    le bruit. C'est ce qui autorise à le mettre en production (cf. §3).
> 3. **Le déficit fond avec l'échelle** : −14,73 pp au 4B, **−10,56 pp au
>    8B** — 79,1 % de MMLU retenu contre 86,1 %. C'est le premier signal de
>    capacités, et pas seulement de perplexité, sur l'axe d'échelle
>    ([`docs/echelle-4b-8b-2026-08-08.md`](docs/echelle-4b-8b-2026-08-08.md)).
>
> ⚠️ **Un point non expliqué, et il faut le laisser visible.** Le bras
> quantifié perd 0,50 pp entre le run Metal du 08-02 (56,09) et le run CUDA du
> 08-06 (55,59) — **5× le glissement de la baseline**, sur ce qui devrait être
> le même fichier. L'errata du lot A le relève et constate qu'il n'est pas
> vérifiable par empreinte : le log du 08-02 est **antérieur** à l'impression
> des empreintes de tokens. Les *deltas* restent cohérents, donc aucune
> conclusion n'en dépend — mais c'est une dette de provenance, pas un
> non-sujet ([`docs/errata-rapport-lot-a-2026-08-06.md`](docs/errata-rapport-lot-a-2026-08-06.md),
> mineur n°2).
>
> ⚠️ Et le même errata (mineur n°3) corrige une justification qu'on a écrite
> plusieurs fois : les questions étant **appariées**, l'écart-type pertinent
> pour une *différence* est celui des paires discordantes (McNemar), pas le
> ± d'échantillonnage imprimé sur chaque ligne. À 3-8 % de discordance,
> σ ≈ 0,4-0,6 pp. Les écarts de 4 à 15 pp dépassent toute correction
> plausible ; les petits (−0,28 pp du 4 bits au 4B) ne doivent pas être
> sur-interprétés.

**Ce que la correction change, et ce qu'elle ne change pas.**

1. **Sur la baseline, elle explique tout.** 72,85 → **70,42 contre 70,2 au
   papier : +0,22 pp, soit 0,17 σ.** L'écart de +2,65 pp qu'on mettait sur le
   compte de vagues différences de protocole était *entièrement* l'échange
   macro/micro. **Le harnais maison est validé** à un niveau jamais atteint.
2. **Sur le bras quantifié, elle ne change presque rien.** La chute passe de
   −15,26 à −14,33 pp : l'agrégation n'en valait que **0,93 pp**. Face aux
   −9,5 pp du papier il reste **−4,8 pp**, et notre 56,09 est à 3,4 σ de leur
   60,7.

> ⚠️ **La prédiction de l'audit était fausse sur ce point, et il faut le dire.**
> Il annonçait « chute ~−10 pp, c'est-à-dire au niveau du papier ». Le
> *diagnostic* du bug était juste ; l'*ordre de grandeur* de son effet ne
> l'était pas. Le macro/micro pesait ~1 pp, pas ~6.

**Donc la conclusion de fond tient, et elle est mieux fondée qu'avant** :
notre quantification perd nettement plus en capacités que la leur. L'argument
est même plus propre — puisque la baseline reproduit le papier à 0,22 pp, le
déficit du bras quantifié ne peut plus être imputé au harnais.

> 🚨 **La cause « la plus probable » écrite ici — le volume de calibration —
> a été mesurée et elle est PLAFONNÉE.** C'est le résultat le plus utile du
> lot B, parce qu'il ferme une piste sur laquelle on allait dépenser 25 $ de
> GPU. Deux mesures, une seule variable chacune
> ([`docs/verdicts-lot-b-2026-08-06.md`](docs/verdicts-lot-b-2026-08-06.md)) :
>
> - **L'oracle** — calibrer sur wikitext-2 *test* lui-même, la triche
>   maximale, donc le plafond absolu de ce que volume, corpus et longueur
>   peuvent rendre — ne gagne que **−1,6 %** de perplexité (20,3289 contre
>   20,6643), soit 2,3σ, et ne referme que **29 %** de l'écart de
>   quantification.
> - **La courbe de volume** : 131 k → 500 k → 1,73 M tokens rend **−1,2 %
>   pour ×13 de volume** (~1,7σ).
>
> Le critère avait été écrit **avant** la mesure (« si l'oracle ne rend que
> 2-3 %, le suspect est plafonné », `docs/pistes-battre-q4.md`) : il est
> atteint par le bas. **Le run de calibration ×100 est enterré.**
>
> Et le suspect qui l'avait remplacé — le **design C**, chemin des
> magnitudes — a été **réfuté à pleine profondeur** la nuit du 06 au 07 :
> ×1,99 de perplexité sur 28 blocs de Qwen3-0.6B, gate automatique, 0 $ de
> GPU (§6, [`docs/verdicts-nuit-2026-08-07.md`](docs/verdicts-nuit-2026-08-07.md)).
> Restent, par ordre : la config 1 bit de gain (écart 0↔2 bits : 1,4 pp au
> papier), la **composition** du corpus (mécanisme raisonnement, que l'oracle
> en perplexité ne borne pas), la compensation post-hoc (EoRA/Recover-LoRA,
> +4-11 pp publiés), le fine-tuning des échelles (+2,1 pp) — et l'issue
> stratégique, **l'axe d'échelle**, qui est la seule à avoir produit un
> chiffre depuis : −10,56 pp au 8B contre −14,73 au 4B.

⚠️ Ce qui **est** mort : l'argument « les deux écarts pointent en sens opposés,
donc ce n'est pas un décalage de protocole qui s'annulerait ». Il ne reste
qu'un seul écart, du côté quantifié.

> 🔎 **Le profil par matière montre le mécanisme** : algèbre abstraite et
> comptabilité tombent à **25 %, exactement le hasard**, pendant qu'histoire,
> droit et psychologie tiennent au-dessus de 80 %. Le 2 bits abîme le
> **raisonnement** bien plus que la **restitution** — et c'est la restitution
> que mesure surtout un corpus de perplexité. Ne plus jamais présenter la
> perplexité seule comme preuve de qualité.

## 3bis. ⚠️ Face au 4 bits — la comparaison qui recadre tout (2026-08-01, remesurée le 08-06/07, échelle le 08-08)

Tout le reste de ce fichier compare LLVQ au **FP16**. C'est la mauvaise
référence : personne ne choisit entre 2 bits et 16 bits, on choisit entre
2 bits et 4 bits.

> 🕳️ **La première version de cette section était hétérogène, et il faut
> savoir en quoi** — c'est le défaut le plus coûteux du dossier. Elle
> alignait un MLX q4 produit localement, une RAM mesurée d'un côté et
> *projetée* de l'autre, un débit mesuré contre un débit extrapolé, et
> surtout un « **5,51 b/poids contre les 4,50 du 4 bits** » qui **mêle deux
> dénominateurs et confond deux quatre-bits**. La campagne du 2026-08-06 l'a
> refaite proprement : **une seule carte (L40S), un seul harnais, une seule
> empreinte de tokens des deux côtés**. Les conclusions de fond ont survécu ;
> deux des quatre lignes se sont depuis renversées.

🚨 **La comparaison mémoire se dit en b/param MODÈLE ENTIER, embedding
compris — et jamais « 5,51 contre 4,50 ».** Deux erreurs empilées dans une
seule phrase : (a) 5,51 compte les **projections seules, hors embedding**
pendant que 4,50 est un **modèle entier, embedding quantifié compris** — deux
dénominateurs ; (b) le 4,50 est le **MLX q4 g64, un artefact absent de la
campagne**, alors que le quatre-bits réellement mesuré est l'**AWQ officiel
de Qwen**, qui pèse **5,30 b/param dans son propre moteur**. Interdit posé
d'avance (`docs/portage-noyau-cuda.md:31`), enfreint quand même, relevé comme
**erreur grave** par l'errata :
[`docs/errata-rapport-lot-a-2026-08-06.md`](docs/errata-rapport-lot-a-2026-08-06.md).

**Qwen3-4B, tout mesuré sur L40S** (sauf indication), empreintes
`3f1baca9033bf251` (ppl) et `65dcd53655e8bfa5` (MMLU) :

| | f16 | AWQ 4 bits (officiel Qwen) | **LLVQ 2 bits** |
|---|---|---|---|
| disque | 8,04 Go | 2,67 Go | **1,77 Go** — **1,41 avec l'embedding int8** |
| VRAM, **b/param modèle entier** | 16,0 | 5,30 ¹ | **5,15** (`Planes14` + embedding q8) |
| débit | 43,5 tok/s | jamais mesuré chez nous ¹ | **88,4–88,5 tok/s** |
| ppl wikitext | 12,2369 | 13,5207 (×1,105) | 16,9422 (×1,385) ² |
| MMLU micro | 70,32 % | 70,04 % (−0,28 pp) | 55,59 % (−14,73 pp) ² |

¹ Dans son propre moteur. **Notre harnais charge l'AWQ déquantifié en f16**,
donc les octets qu'il occupe chez nous ne veulent rien dire — réserve qui joue
*contre* nous, pas pour.
² Sur le fichier scellé à embedding f16. La variante à embedding int8 rend
16,9358 et 55,70 % — dans le bruit des deux côtés, donc les lignes disque et
VRAM peuvent citer le q8 sans casser la colonne qualité.

**Où on en est, ligne par ligne, et ce qui a changé depuis le 2026-08-01.**

1. **Disque** : notre avantage, inchangé (1,77 contre 2,67 Go).
2. **Mémoire : renversé.** 5,15 contre 5,30 — on est passé **sous l'AWQ
   réel**, et `Planes12x` descendrait à 4,69 (cf. l'échelle des formats en
   §3). Ce que le §3bis d'origine décrivait comme structurellement perdu ne
   l'était pas : c'était le one-hot de `Slot32`, pas la méthode.
3. **Débit : renversé contre notre propre chemin dense** (×2,03), mais
   ⚠️ **à formuler deux fois** : l'essentiel du gain vient d'un défaut du
   moteur de référence (candle recopie 778 Mo de vocabulaire par token), donc
   ~×1,4 contre ce moteur corrigé (§3). Contre l'AWQ **dans son moteur à
   lui**, on n'a toujours aucune mesure.
4. **Qualité : pas renversé, et c'est le point dur.** −14,73 pp de MMLU au 4B
   contre −0,28 pour le 4 bits. Sur un 4B, **le 4 bits domine sans
   discussion**.

> 🕳️ **Le dilemme central de cette section est levé.** « Le format qui va vite
> ne rentre pas mieux que du 4 bits ; le format qui rentre mieux ne va pas
> vite » était vrai sur `Slot32` / `Flat32` / `Grouped32`, et il a orienté
> tout le travail de format. `Planes14` le contredit : **plus petit ET plus
> rapide** que `Slot32`, à Go/s constants. Analyse d'origine, conservée pour
> la généalogie : [`docs/face-au-4-bits.md`](docs/face-au-4-bits.md).

**Et l'axe d'échelle a produit son premier chiffre** (2026-08-08,
[`docs/echelle-4b-8b-2026-08-08.md`](docs/echelle-4b-8b-2026-08-08.md)) : au
8B, notre chute MMLU tombe à −10,56 pp pendant que celle du 4 bits monte à
−3,07 pp. **L'écart au 4 bits passe de 14,45 pp à 7,49 pp — il est divisé par
deux en doublant la taille.** À 4B le choix ne se discute pas ; à 8B c'est un
arbitrage réel — 7,5 points de MMLU contre 25 % de mémoire en moins. ⚠️ Deux
points ne font pas une loi d'échelle, et extrapoler à 70B serait exactement le
raccourci que ce dossier refuse.

Ce qui n'est pas réfuté : le noyau lui-même (un décodeur Leech multi-coquilles
fusé qui bat le FP16 de 2,14× sur les projections, tourne dans un vrai modèle
et rend les mêmes tokens, n'existe nulle part — papier compris). Ce qui est
réfuté, c'est le *produit* sur un 4B.

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

> ✅ **État du code aujourd'hui, parce que ce paragraphe est au passé et se
> relit mal** : la crête **est relative** depuis la correction —
> `let ridge = cfg.lambda * mean_diag;` puis `mat[p*m+p] += ridge;`
> (`llvq-quant/src/gptq.rs:448-451`), `mean_diag` étant la moyenne de la
> diagonale de `M`. Il n'y a plus de λ absolu nulle part. Ce qui reste ouvert
> n'est pas la crête mais le **second membre** — voir la note du §6 sur le
> prior des échelles de groupe.

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
   - 🔎 **Piste de formulation — précisée, chiffrée, et NON POURSUIVIE.**
     L'Algorithme 3 écrit `s = (M + λI)⁻¹ r` : la crête tire les échelles
     vers **zéro**, alors que le facteur est censé valoir ≈ 1. Pour un tel
     facteur, le prior naturel serait `(M + λ̃I)s = r + λ̃·1`.
     ⚠️ **Ce qui manque n'est PAS la relativité de la crête** — une version
     antérieure de cette note laissait croire le contraire. Le code applique
     déjà `λ̃ = λ·mean(diag M)` (`llvq-quant/src/gptq.rs:448-451`, cf. §5) ;
     ce qui manque, c'est uniquement le `+ λ̃·1` au **second membre**, `rhs`
     restant `gᵀH·w_orig` (`gptq.rs:436`). Un terme, pas une refonte.
     **Verdict, et c'est pourquoi elle n'est pas au programme** : le gain
     local qu'elle chercherait à récupérer vaut **0,35 %** (21,2412 →
     21,1677, ligne « 33 k / on / off » du tableau ci-dessus — la seule
     mesure qu'on en ait). Le σ inter-graines mesuré depuis est de **0,7 %**
     sur exactement ce protocole (§2, lot B). **Le gain espéré est sous 1σ :
     l'expérience ne peut pas le distinguer du bruit**, et son bras global
     est de toute façon rouge (44,66 → 53,60). Piste close, pas en attente.

4. **La rotation d'incohérence est le plus gros levier mesuré** : ×2,290 →
   ×1,811 d'un seul coup, soit 21 % de perplexité. C'était bien l'écart de
   configuration qui manquait, pas un défaut de la chaîne.

**On est dans le régime du papier.** Repère : leur Llama-3.2 1B sans
fine-tuning est à **×1,76** (Table 10, vérifiée par rendu image le 2026-08-04 :
21,36 sur une baseline de 12,14, **variante shape–gain 0 bit + Spherical
GPTQ**, c'est-à-dire leur meilleure ligne 1B sans FT — leur spherical shaping
y est à ×1,96) ; on est à **×1,811** sur un modèle
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

> 🔎 **Le fait de méthode qui a émergé depuis, et qui vaut plus que les deux
> anecdotes qui le portent : dans ce pipeline, un proxy local meilleur
> prédit une composition pire.** Deux occurrences indépendantes, à un an
> d'écart d'intuition :
>
> | | proxy local | 28 blocs |
> |---|---|---|
> | `group_scales` | mieux (21,24 → 21,17) | **catastrophe** (44,66 → 53,60) |
> | **design C** (2026-08-07) | strictement décroissant, test vert | **×1,99** (35,98 → 71,42) |
>
> Le **design C** — rétraction libre plus résolution close des magnitudes —
> était le suspect n°1 du déficit MMLU. Il a été implémenté fidèlement au
> doc, passé en revue adversariale point par point, son solve est l'algèbre
> verrouillée du crate, un défaut de signe trouvé en revue a été corrigé
> avant le run — et il rend **×1,99 de perplexité à pleine profondeur**. Un
> gate automatique a bloqué le run 4B de 4 h qui devait suivre : 0 $ de GPU
> pour un rouge net ([`docs/verdicts-nuit-2026-08-07.md`](docs/verdicts-nuit-2026-08-07.md)).
>
> **Ce qu'on en retient** : *la rigidité de norme de la rétraction sphérique
> est porteuse à profondeur*. Tout raffinement qui la relâche pour améliorer
> une couche paie plus cher à la composition. Réserve honnête : c'est **notre
> lecture** du design C qui est réfutée — le papier n'en donne pas le
> pseudo-code. **Conséquence pratique : un A/B à 3 blocs ne suffit pas à
> valider un mécanisme qui touche aux magnitudes.** Il faut le gate à
> profondeur, et il faut l'automatiser avant de payer une carte.

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
> **Ce fichier est `leech1c12`** (fin de `~/llvq-run-4b-artefact.log` :
> « leech1c12, 36 blocks, rot on, calib c4 ») : recherche angulaire plafonnée
> à la boule Λ₂₄(12), soit **47 bits d'index + 1 bit de gain = 48 bits/bloc**.
> Le paragraphe « débit strictement égal » plus bas, qui présente cette
> restriction comme une option future, décrit donc **le run déjà publié**.
>
> 🔎 **Note de provenance — trois chiffres, un seul objet** (même logique que
> celle de [`docs/format-noyau.md`](docs/format-noyau.md)) :
> **2,0702** = l'« effective rate » imprimé par `bin/smoke`
> (`calib.rs::bits_per_weight`) — la comptabilité idéale du payload : 48 bits
> par bloc, queue `KeepExact` à 16 bits, une échelle f16 par ligne de sortie,
> le tout rapporté aux **3 633,3 M poids des linéaires, queue comprise**.
> **2,1696** = les bits réellement écrits dans le fichier de 981 Mo —
> en-têtes, échelles de ligne et centroïdes en f64, queue en pleine
> précision — rapportés aux **3 616,4 M poids quantifiés seuls**. Deux
> numérateurs *et* deux dénominateurs, pas deux mesures contradictoires :
> le fichier pesé est cohérent avec les deux.
> Les **2,1117** du tableau ci-dessous relèvent enfin d'une **autre config** —
> cap 13, 48 bits d'index + 1 de gain = **49 bits/bloc** — et, comme dit
> ci-dessus, valaient en réalité 2,7338 : rien à voir avec le fichier scellé.
>
> Diagnostic complet, les trois défauts et ce qui reste à décider :
> [`docs/retraction-et-gain.md`](docs/retraction-et-gain.md).

> ✅ **Le 16,9617 est confirmé sur le fichier lui-même (2026-08-01, §A2).**
> Il avait été mesuré par la boucle interne de `smoke`, en F32, sur le modèle
> encore en mémoire. `bin/ppl` sait désormais charger l'artefact scellé, donc
> on peut scorer les octets livrés plutôt qu'une reconstruction :
>
> | bras | dtype | source | ppl |
> |---|---|---|---|
> | baseline | f32 | checkpoint | 12,2336 |
> | baseline | **f16** | checkpoint | **12,2361** |
> | LLVQ 2 bits | f32 | modèle en mémoire | 16,9617 |
> | **LLVQ 2 bits** | **f16** | **`~/qwen3-4b-llvq.bin` décodé** | **16,9415** |
>
> Dégradation **×1,3846** en f16 contre ×1,3865 en f32. **Le confondant
> F32/MMLU-F16 est donc nul à 0,1 % près** : il ne reste que l'agrégation
> macro/micro (§3ter) pour expliquer l'écart au papier sur MMLU.
>
> Les deux bras impriment la même empreinte de tokens `3f1baca9033bf251` — 12
> fenêtres de 4096 identiques des deux côtés, donc le rapport a un sens. C'est
> la condition qu'on supposait sans la vérifier.

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
(`norm(Λ₂₄(12))` + 1 bit de gain). **✅ Fait : c'est exactement la config du
fichier scellé `leech1c12`** (2,0702 b/poids effectifs — cf. la note de
provenance en tête de section). Ce paragraphe précède le run et n'est
conservé que pour la généalogie de la décision.

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
*(Depuis : fait — le fichier scellé `leech1c12` existe : 981 Mo de
projections, 1,771 Go avec l'embedding f16, cf. §3bis.)*

### Qwen3-8B — le premier point d'échelle (2026-08-02, sur GPU loué)

Premier run hors du Mac : HF Jobs, `rtx-pro-6000` (23 vCPU, 96 Go), CUDA,
`leech1c12L3`, calibration C4 131 k tokens, 36 blocs. **4,18 h facturées,
11,48 $.** 399 s/bloc, stable au dixième sur les 36 — aucune dérive.
`verify_artifact` repasse : 6 945 767 424 poids identiques bit pour bit.

| | Qwen3-4B | **Qwen3-8B** |
|---|---|---|
| bits/poids | 2,1696 | **2,0436** |
| baseline (ctx 4096, 12 fen.) | 12,2336 | 8,9893 |
| LLVQ 2 bits | 16,9617 | 11,3934 |
| **dégradation** | ×1,386 | **×1,267** |

**Le 8B se dégrade moins que le 4B, à moins de bits.** C'est le signal
d'échelle qu'on cherchait, et il va dans le bon sens pour le 32B.

> ✅ **Ce signal a été confirmé, élargi aux capacités, et refait à une seule
> variable le 2026-08-08.** Le run ci-dessus était `leech1c12L3` ; la campagne
> d'échelle a requantifié le 8B en `leech1c12` — **même codebook, même corpus
> (C4 131 k), même rotation, même harnais, même carte (L40S), mêmes empreintes
> de tokens des deux côtés** — pour que la taille du modèle soit la seule
> chose qui change. C'est ce que le run du 08-02 ne permettait pas de dire.
>
> | | Qwen3-4B | **Qwen3-8B** | tendance |
> |---|---|---|---|
> | ppl, dégradation | ×1,3845 | **×1,2201** | **−42 % de l'excès** |
> | MMLU micro, f16 → LLVQ | 70,32 → 55,59 | **76,08 → 65,52** | — |
> | chute MMLU | −14,73 pp | **−10,56 pp** | +7,0 pp de rétention |
> | **écart LLVQ ↔ AWQ 4 bits, MMLU** | **14,45 pp** | **7,49 pp** | **divisé par deux** |
>
> **Le déficit fond sur les deux axes, et l'écart au 4 bits fond deux fois
> plus vite que le déficit lui-même** — parce que le 4 bits, lui, commence à
> payer : indiscernable du f16 à 4B (−0,28 pp), il perd 3,07 pp à 8B.
> Sources : [`docs/echelle-4b-8b-2026-08-08.md`](docs/echelle-4b-8b-2026-08-08.md),
> [`mesures/campagne-8b-qualite-2026-08-08.txt`](docs/mesures/campagne-8b-qualite-2026-08-08.txt).
>
> **Le bras vitesse, mesuré séparément** — et il illustre exactement le piège
> d'embedding du ⚠️ ci-dessous : le 8B **délie ses têtes**, donc deux tables à
> porter au lieu d'une.
>
> | | dense f16 | fusé `Planes14`, embedding f16 | fusé `Planes14`, embedding **q8** |
> |---|---|---|---|
> | tok/s | 26,5 | 34,4 (×1,30) | **69,3 (×2,61)** |
> | Go carte | 16,38 | 6,62 (÷2,48) | **5,45 (÷3,01)** |
>
> 128 tokens identiques au bras dense dans les deux cas
> ([`mesures/campagne-8b-vitesse-2026-08-08.txt`](docs/mesures/campagne-8b-vitesse-2026-08-08.txt),
> [`mesures/campagne-8b-q8-2026-08-08.txt`](docs/mesures/campagne-8b-q8-2026-08-08.txt)).
> ⚠️ **Sans q8, le 8B ne renverse rien** — 6,62 Go dont 2,49 de tables portées
> en f16. Le q8 n'est pas un raffinement à cette échelle, c'est ce qui rend le
> verdict VRAM favorable.
>
> ⚠️ **Ce que ces deux points ne prouvent pas : une loi d'échelle.** Deux
> points ne font pas une courbe. *Si* la tendance se poursuit, l'écart au
> 4 bits se referme vers 16-32B — le point 32B coûterait ~60 $ et une nuit, et
> c'est lui qui trancherait. Repère externe : le papier donne le 8B sans
> fine-tuning à **×1,13** ; nous sommes à ×1,220 avec ~100× moins de tokens de
> calibration et la rotation d'entrée seule. Au 4B nous étions à parité
> (×1,385 contre leur ×1,374) — l'écart qui *apparaît* à 8B est cohérent avec
> un déficit de calibration qui pèse plus lourd quand le modèle grossit. Une
> piste, pas une conclusion.

Le débit plus bas n'est pas un progrès de méthode, c'est un alignement :
`intermediate_size = 12288 = 24 × 512` exactement, donc `down_proj` — un tiers
des poids — n'a **aucune queue**. Sur le 4B, `9728 = 24 × 405 + 8` en a une.

⚠️ **Ne pas publier le ratio de compression du 8B — ou alors avec son
mécanisme.** `tie_word_embeddings` y est `false` avec un `hidden` de 4096
seulement : l'embedding pèse 15,2 % des poids et **57 % de l'artefact
scellé**, pour un ratio de ×3,7 — moins bon que le 4B (×4,63) à méthode
identique. L'artefact écrit ici fait 1,823 Go, mais c'est un fichier
**projections seules** (format v1) ; scellé il ferait ~4,3 Go.

> ✅ **Le « ~4,3 Go » projeté est confirmé par la mesure du 2026-08-08 :
> 4,32 Go scellé** avec l'embedding f16, et **3,157 Go** avec l'embedding
> int8. Le q8 reprend donc à lui seul **1,16 Go** sur ce modèle — parce qu'il
> attaque les **deux** tables déliées. C'est la même leçon que le §3ter du 4B,
> mais amplifiée : sur un modèle à têtes déliées, l'embedding n'est pas un
> détail de comptabilité, c'est la moitié du fichier.

**Profil par phase à cette échelle** (GPU) :

| phase | s | % |
|---|---|---|
| quantification (encodeur) | 12 951 | **90,3 %** |
| factorisation | 783 | 5,5 % |
| écriture artefact | 255 | 1,8 % |
| capture (passe 1) | 174 | 1,2 % |
| transfert f64 | 118 | 0,8 % |
| advance (passe 2) | 66 | 0,5 % |

L'accélérateur sert à évacuer les passes avant (1,2 %) ; tout le reste est
l'encodeur, qui est CPU. Donc **sur GPU, seuls comptent le nombre de vCPU et
la vitesse de l'encodeur.** La factorisation remonte de 1,6 % (0,6B) à 5,5 %
— le `n³` commence à se voir, et il pèsera plus à 25 600.

**Projection 32B** sur base mesurée (4,77·10⁻⁵ cœur-s/poids) : **≈ 18 h**.
`rtx-pro-6000` **49 $** — mais 131 Go en f32 ne tiennent pas dans 96 Go de
VRAM, donc **C3 (chargement bf16) est un prérequis**, pas une optimisation.
Sans lui il faut un `h200x2` à 10 $/h, soit **~180 $**. C3 vaut donc ~130 $
sur ce seul run.

### Qwen3-32B — dé-risqué sur 4 blocs, pas encore lancé (2026-08-03)

4 blocs sur 64, `rtx-pro-6000x2`, **bf16** (C3), 59 min, **5,43 $**. Le but
n'était pas un chiffre de qualité — 4 blocs sur 64 donnent ×1,002, ça
n'apprend rien — mais de lever trois inconnues avant d'engager 11 h.

| inconnue | verdict |
|---|---|
| pic mémoire de `faer` à n=25600 | ✅ 70,6 Go hôte / 512, et 77,4 Go VRAM / 97 |
| bf16 à cette échelle | ✅ `verify_artifact` repasse, 1 950 351 360 poids bit pour bit |
| s/bloc réel | ⚠️ **621 s**, contre ~500 prédits |

**L'estimation était 25 % basse.** Le run complet fait **~11,4 h et ~62 $**,
pas 9 h et 49 $. Le dé-risquage a coûté 5,43 $ et corrigé une erreur de 13 $
avant engagement.

**Le profil par phase explique l'écart, et il bouge avec la largeur :**

| phase | 0,6B | 8B | **32B** |
|---|---|---|---|
| quantification (encodeur) | 97,6 % | 90,3 % | **71,8 %** |
| **factorisation** | 1,6 % | 5,5 % | **16,5 %** |

Le terme en `n³` remonte exactement comme `cholbench` le prédisait (~1,9 h sur
le run complet). Conséquence méthodologique : **le coût par poids n'est pas
linéaire** — 4,77·10⁻⁵ cœur-s à 8B, **6,36·10⁻⁵ à 32B**. Ne plus extrapoler
une largeur depuis une autre sans marge.

Reste à décider avant de payer les 62 $ : l'encodeur pèse encore 71,8 %, et
§2c liste deux optimisations jamais tentées (complément d'octade, SIMD `pulp`).
Un facteur 1,5 ramènerait le run à ~40 $, et ça composerait sur tous les runs
suivants.

### Faire tourner ailleurs — [`ops/`](ops/README.md)

Le Mac de dev fait 69 Go ; Qwen3-32B en pèse 65,5 en bf16. Tout ce qui dépasse
le 8B tourne sur **HF Jobs**, piloté par `ops/run.py` (Python, hors du
workspace Rust qui reste sans dépendance).

```bash
uv run ops/run.py estimate Qwen/Qwen3-32B --dtype bf16   # cœur-heures et coût
uv run ops/run.py selftest                                # l'estimateur vs le run 4B réel
uv run ops/run.py publish <user>/llvq-runner-cuda --cuda  # HF construit l'image
uv run ops/run.py oracle --image hf.co/spaces/…           # ⚠️ le verrou, à chaque backend
uv run ops/run.py launch --model … --flavor … --bucket auto
uv run ops/run.py monitor <job_id> --flavor …             # coût facturé + logs
```

Quatre choses apprises en s'en servant, qui coûtent cher à redécouvrir :

- **`oracle` d'abord, toujours.** Sur CUDA il rend `max |Δhidden| = 0.000e0`,
  exactement comme en Metal. 42 s et ~1 centime pour savoir si les hessiennes
  construites sur ce backend valent quelque chose.
- **`fast-linalg` n'est pas optionnel en pratique.** Sans lui la factorisation
  est **40× plus lente** pour une perplexité bit-identique. `smoke` avertit
  bruyamment quand la feature manque.
- **Le builder d'un Space n'a pas de GPU**, donc `CUDA_COMPUTE_CAP` doit être
  figée dans l'image (89 = Ada). Et le profil `lto = "thin"` +
  `codegen-units = 1` tue le build par OOM : `ops/Dockerfile.cuda` relève les
  codegen units et limite les jobs cargo.
- **Sans C5, le conteneur retélécharge le checkpoint** : 26 min sur 65,5 Go,
  soit 45 % d'un run court.

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
s'effondre à 191,90 de perplexité, le Spherical GPTQ tient à 6,90) et c'est
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

**Lot K-1 (2026-08-05) — trois jalons locaux, avant d'engager une carte louée.**

1. **Échelle bits↔vitesse, un seul protocole et une seule comptabilité**
   (`bin/thesis`, 7 bras, 7 rounds dont 2 jetés, tous dispatchés à chaque
   round dans le même ordre ; journal
   [`docs/mesures/k1-metal-2026-08-05.txt`](docs/mesures/k1-metal-2026-08-05.txt)) :
   FP16 16,000 b/poids → 1,00× ; `Slot32` 5,510 → **2,03× [2,03–2,10]** ;
   `Flat32` 5,256 → 0,91× [0,91–0,91] ; `Grouped32` 3,498 → 0,69×
   [0,69–0,69]. ⚠️ Chaque rapport est la **médiane du rapport formé round par
   round**, avec sa plage sur les 5 rounds gardés : ce n'est pas le quotient
   de deux minima, qui mêlerait deux rounds n'ayant jamais coexisté. Les
   millisecondes dérivent d'un run à l'autre — c'est le fait même que ce lot
   a établi — là où les b/poids sont exacts et se reproduisent au chiffre.
   **La courbe est brutalement non linéaire** : `Flat32` n'économise que
   0,254 b/poids sur `Slot32` et coûte 2,27× le temps ; `Grouped32` économise
   2,012 b/poids et coûte 3,01×. *(Différences et quotients de valeurs du même
   run, du même processus et de la même comptabilité — grandeurs dérivées,
   pas des mesures séparées.)* Donc **reprendre des bits se fait
   *dans* `Slot32`** (plafond
   L ≤ 4), jamais en changeant de layout — c'est ce qui oriente le port CUDA.
2. **Le plafond L ≤ 4 vaut ≤ 4,7083 b/poids** (comptabilité `bin/rtbits`, où
   `Slot32` pèse 5,3756 aujourd'hui : 0,667 b/poids, 12,4 %). C'est un
   **majorant inconditionnel**, pas une simulation : L ≤ 4 impose
   `9 + 1 + 24·4 = 106 bits = 14 octets`, donc un stride ≤ 14 o pour tout
   groupe, et il est atteint dès qu'un groupe porte un bloc à 4 niveaux —
   **4 708 799 groupes sur 4 708 800** en portent un. Un compte, pas une
   probabilité.
3. **Le conflit de bancs prédit en §3.2 de `docs/portage-noyau-cuda.md`
   n'existe pas ici.** Le pas de 28 flottants ne gagne rien (10,081 contre
   10,126, soit 0,4 % *plus lent*, et sa plage de rapport [2,06–2,17] recouvre
   par le haut celle du dense [2,12–2,19] : rien ne le distingue). Ce qui paie
   est la **largeur de chargement** : `float4` rend 3,5 % sur LLVQ et 5,1 % sur
   FP16 — des deux côtés, donc le rapport ne bouge pas (2,04× float4/float4
   contre 2,09× scalaire/scalaire, chacun dans la dispersion de l'autre).
   *(Ces pourcentages et ces quotients sont dérivés de la table du journal, pas
   lus dedans.)* Le modèle NVIDIA (32 bancs de 4 octets)
   n'est pas transposable à Apple. Les deux variantes `float4` de `Slot32`
   sont bit-exactes contre le noyau scalaire ; celle du bras FP16 ne l'est
   pas (3,1e-8, somme non écrite en `fma` explicites) — confondant déclaré.
4. **Dispersion inter-processus** : trois invocations consécutives du banc à
   deux bras **non modifié** donnent 2,029× puis 2,050× puis 2,080×, octets
   et erreurs identiques. Le **2,07×** publié est le haut de cette plage ; un
   effet de quelques pour cent ne se tranche pas en comparant deux
   invocations distinctes du binaire.

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

> 🚨 **RENVERSÉ le 2026-08-04. La coquille unique ne bat pas l'union : elle
> perd.** Ce qui suit est la version corrigée ; l'ancienne table est conservée
> plus bas pour la généalogie. Table publiable :
> [`README.md`](README.md#an-open-question-for-the-authors).

| code | bits/**bloc** | MSE | rétention | classes |
|---|---|---|---|---|
| papier, union `norm(Λ₂₄(12))` + 1 bit de gain | 48 | 0,078 | **92,14 %** (le sien, MSE non arrondie) | **301** |
| **coquille 12 seule + 1 bit de gain** | **48** | 0,0817 | **90,34 %** | **79** |
| coquille 13 seule + 1 bit de gain | 49 | 0,0762 | 90,96 % | 82 |
| notre union `norm(Λ₂₄(13))` + 1 bit de gain | 49 | **0,0725** | **92,72 %** | 383 |
| notre union `norm(Λ₂₄(13))` + 0 bit | 48 | 0,0850 | 88,90 % | 383 |

**À débit empaqueté identique, l'union gagne.** `ceil(log₂ 70 486 236 999 360)
= 47` pour la coquille et `ceil(log₂ 111 043 117 458 000) = 47` pour la boule du
papier : les deux coûtent 48 bits/bloc. Les lignes 3 et 4 le montrent dans un
seul harnais, à 49 bits des deux côtés : 0,0725 contre 0,0762.

> 🕳️ **Le défaut, et il est instructif.** La table publiée jusqu'ici donnait
> 92,24 % à la coquille 12 contre 92,14 % au papier. Elle divisait la MSE par le
> débit **fractionnaire** `log₂|Shell(12)|/24 = 1,9584` — qu'aucun fichier ne
> paie — et comparait le résultat à un chiffre du papier cité à 2,000. La
> colonne **débit** avait été corrigée le 2026-08-03 ; la colonne **rétention**,
> calculée à partir d'elle, ne l'avait pas été. Une correction partielle est
> pire qu'aucune : elle donnait au tableau l'air d'avoir été vérifié.
>
> Signature repérable à l'œil, sans recalcul : le papier y avait à la fois une
> **meilleure MSE** (0,078 contre 0,0817) et une **rétention pire** — impossible
> à débit fixé, la rétention étant une fonction monotone de la MSE.
>
> Second défaut, indépendant : les **383 classes** attribuées au codebook du
> papier sont le compte de `Λ₂₄(13)`. `Λ₂₄(12)` en a **301** (180 paires + 121
> impaires, `enumerate_classes`). Le gain structurel est ×3,8, pas ×4,8.

**Ce qui survit** n'est pas un résultat de distorsion mais un arbitrage
d'ingénierie : **79 classes contre 301, et une norme constante** — l'argument
matériel que l'annexe G soulève elle-même sans le mesurer. Et le fichier livré
utilise la **boule** `Λ₂₄(12)`, c'est-à-dire le codebook du papier, pas une
coquille unique.

<details><summary>Table périmée (avant le 2026-08-04)</summary>

| code | bits/dim | MSE | rétention | classes |
|---|---|---|---|---|
| papier, union `norm(Λ₂₄(12))` + 1 bit de gain | 2,0000 | 0,078 | 92,14 % | 383 |
| notre union `norm(Λ₂₄(13))` + 0 bit | 1,9999 | 0,0850 | 88,90 % | 383 |
| **coquille 12 seule + 1 bit de gain** | **1,9584** | 0,0817 | **92,24 %** | **79** |
| **coquille 13 seule + 1 bit de gain** | 2,0113 | 0,0762 | **92,33 %** | **82** |

Chiffres déjà révisés une fois le 2026-08-01 (§A5, le banc codait le gain sur
la projection) : 12 seule 92,81 → 92,24, 13 seule 92,83 → 92,33.

</details>

Structure des coquilles (vérifiée par la même formule de cardinalité que la
série thêta) :

| m | \|Shell(m)\| | bits/dim | classes |
|---|---|---|---|
| 12 | 70 486 236 999 360 | 1,917 | 79 |
| 13 | 169 931 095 326 720 | 1,970 | 82 |
| 14 | 384 163 586 352 000 | 2,019 | 115 |
| union m ≤ 13 | 280 974 212 784 720 | 2,000 | 383 |

⚠️ **Ne rien surinterpréter dans l'autre sens non plus.** C'est une source
gaussienne i.i.d., un seul harnais, une seule seed. Le papier mesurait une
*distance angulaire au plus proche voisin* sur une source radialement
uniforme, pas une rétention MSE après quantifieur de gain — donc nos chiffres
ne le confirment pas plus qu'ils ne le contredisaient. Ce qui est acquis,
c'est qu'**on ne conteste plus sa conclusion sur la distorsion** : à débit
égal l'union est le meilleur code, dans notre propre harnais. **À revérifier
sur de vrais poids après la boucle GPTQ** — les poids ne sont pas gaussiens et
le GPTQ déforme leur distribution.

L'intérêt d'une coquille unique n'est donc plus la qualité mais la structure :
**79 classes contre 301** (encodeur plus rapide, noyau sans rééchelonnage entre
coquilles), au prix de ~5 % de MSE. Le format d'index v1 et le moteur de
recherche sont épinglés sur la boule m ≤ 13 : passer à une coquille unique
casse la compatibilité des fichiers quantifiés (§4).

> ✅ **La décision matérielle est prise, et elle a été prise deux fois.**
> Cette ligne demandait de trancher « CUDA (cible serveur NVIDIA) ou
> Metal/`wgpu` (Mac de dev, portabilité, argument souveraineté) ? » avant
> d'écrire une ligne. Réponse livrée : **les deux, dans cet ordre**. `Metal`
> d'abord parce que la machine de dev est un MacBook et qu'un banc gratuit
> vaut mieux qu'une carte louée pour trouver les pièges de mesure
> (`llvq-metal`, tous les chiffres jusqu'au 2026-08-05) ; **CUDA ensuite**
> parce que c'est la seule cible où le résultat devient reproductible par un
> tiers (`llvq-cuda`, source compilée par NVRTC au démarrage — voir le
> commentaire de `llvq-cuda/Cargo.toml` sur pourquoi `cudarc` est
> `cfg(target_os = "linux")` et pas derrière une feature). Tous les chiffres
> publiés depuis le 2026-08-06 sont CUDA, sur L40S ou `rtx-pro-6000`.
> `wgpu` n'a jamais été écrit et ne l'est pas au programme.

## 7. Conventions

- `llvq-core` et `llvq-search` restent **sans dépendance** : le cœur
  mathématique doit rester auditable (contexte souveraineté).
- **`unsafe` est autorisé aux frontières matérielles, interdit partout
  ailleurs.** Concrètement : mmap, lancement de noyau, lecture d'un buffer
  device — c'est-à-dire `llvq-metal`, `llvq-cuda`, `llvq-llm`. Les quatre
  crates du cœur portent `#![forbid(unsafe_code)]` et doivent le garder
  (cf. §2 pour le compte exact, crate par crate).
- Zéro warning clippy.
- Les tests coûteux sont `#[cfg_attr(debug_assertions, ignore = "...")]` :
  rapides en debug, exhaustifs en release. ⚠️ La suite complète se compte en
  dizaines de minutes, pas en secondes (cf. §2).
- Commentaires et docs en anglais dans le code, échanges en français.

**Trois règles de chiffres, chacune payée par une erreur publiée :**

1. **Toute comparaison mémoire se dit en b/param MODÈLE ENTIER, embedding
   compris.** Jamais un b/poids de projections contre un b/param de modèle
   entier — c'est la faute grave de l'errata du lot A, et elle est facile à
   refaire parce que les deux chiffres se ressemblent.
2. **Une plage, pas un point.** Les millisecondes dérivent d'une invocation à
   l'autre (2,029× / 2,050× / 2,080× sur le *même binaire non modifié*) là où
   les octets, les b/poids et les pires erreurs se reproduisent au chiffre.
   Un rapport se forme **round par round**, jamais comme quotient de deux
   minima issus de rounds n'ayant jamais coexisté.
3. **Étiqueter la provenance de chaque nombre** : *mesuré* / *calculé* /
   *estimé*, et dans quelle comptabilité. Trois chiffres différents peuvent
   décrire le même objet sans se contredire (2,0702 · 2,1696 · 2,1117, §6) —
   ce qui se contredit, c'est de les comparer entre eux.
