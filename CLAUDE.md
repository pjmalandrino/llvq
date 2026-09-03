# LLVQ, carte du dépôt

Chargé au démarrage de chaque session : où reprendre et ce qu'on ne fait pas.
Les chiffres de résultat sont dans `docs/ETAT.md`,
l'histoire dans `docs/HISTORIQUE.md`.

## Objectif

Réduire le coût d'inférence des LLM pour de la souveraineté : faire tenir de
plus gros modèles sur du matériel local. Le levier est le nombre de bits par
poids ; à 2 bits, un 70B passe de 140 Go à 18 Go sur disque (*calculé*). En VRAM
le format servi déplie l'index à 4,804 b/poids (*mesuré*,
`docs/mesures/e2-golay70-bench-2026-08-07.txt`). Sous le triplet produit en
vigueur, la plus grande classe admise vaut 43,3 Md à 5,162 b/param ; le 32B est
l'objet servi, le 70B ne rentre pas (*calculé*, `docs/ETAT.md` §6). On
implémente en Rust le papier LLVQ,
quantification vectorielle sur le réseau de Leech Λ₂₄
([arXiv:2603.11021](https://arxiv.org/abs/2603.11021)). La contribution
d'ingénierie est le noyau fusé multi-coquilles, déquantification et matvec
dans un seul noyau CUDA.

## Reprise

Lire dans cet ordre, chaque document se suffit à son niveau.

1. `docs/ETAT.md` : config servie, chiffres de tête, décisions ouvertes.
2. `docs/ROADMAP.md` : la suite, avec ses gates et ses coûts.
3. `docs/HISTORIQUE.md` : le fil chronologique, une entrée par période.
4. `docs/METHODE.md` : les règles du labo, préregs, étiquettes, rétention.
5. `docs/STYLE.md` : les règles d'écriture de tout document vivant.

Le papier est transcrit en entier dans `docs/llvq-paper-notes.md` ; ne pas
rouvrir le PDF, et ne jamais lui appliquer `pdftotext` (extraction corrompue).
Les journaux sont dans `docs/mesures/`, les préregs dans `proofs/`, le
registre des jobs dans `docs/data/jobs.csv`. `docs/archive/` ne s'édite pas.
Données : `docs/data/mmlu-appariee.csv` (neuf paires, 3 tailles × 3 bras),
`docs/data/mmlu-dumps/`, `docs/data/ppl-genou.csv`, `docs/data/echelle-formats.csv`,
`docs/data/awq-speed-4b-2026-08-17.json` ; `docs/data/README.md` donne la provenance
des montants. `docs/fiche-4b.md` décrit l'objet publié, `docs/format-noyau.md` le
noyau et ses pièges de mesure, `docs/echelle-4b-8b-2026-08-08.md` l'échelle.

## Architecture

Huit crates, membres de `Cargo.toml` (*mesuré*, `Cargo.toml:3-12`).

| crate | rôle | dépendances externes | `unsafe` |
|---|---|---|---|
| `llvq-core` | Golay [24,12,8], Λ₂₄, coquilles | aucune | forbid |
| `llvq-search` | recherche NN exacte, classes m ≤ 13, indexage, packing, `rankdec` | aucune | forbid |
| `llvq-quant` | Spherical GPTQ, algèbre dense, boucle par blocs | `faer` 0.24, optionnel, feature `fast-linalg` | forbid |
| `llvq-artifact` | format `.llvq` : writer, reader, décodeur | aucune | forbid |
| `llvq-bench` | débit-distorsion, débit encodeur, coût du décodage | aucune | forbid |
| `llvq-metal` | micro-bancs GPU macOS, shaders MSL, `rankbench` | `metal` | autorisé |
| `llvq-cuda` | noyau fusé NVIDIA compilé par NVRTC, bancs | `cudarc`, `cfg(target_os = "linux")` | autorisé |
| `llvq-llm` | passe avant, corpus, perplexité, MMLU, chemin fusé dans le modèle | `candle`, `tokenizers`, `hf-hub`, `parquet` | autorisé |

L'arbre complet de `llvq-artifact` fait 3 crates ; celui de `llvq-llm` fait
261 paquets, 291 avec `metal,fast-linalg` (*mesuré*, `docs/archive/audit-publication-2026-08-03.md`).
`unsafe` n'est autorisé qu'aux frontières matérielles : mmap, lancement de
noyau, lecture d'un buffer device. Réserve : `#![forbid(unsafe_code)]` dans
un `lib.rs` ne couvre pas les tests d'intégration, crates séparés ; fermer ce
trou demande `[workspace.lints]`, décision d'opérateur en attente.
Sans `--features fast-linalg`, la factorisation est ~40× plus lente pour un
résultat bit-identique (*mesuré*, `llvq-llm/src/bin/smoke.rs:1253-1265`).
Le chemin d'algèbre maison (`llvq-quant/src/linalg.rs`, 246 lignes) reste la
référence de vérification : `both_factorizations_agree`
(`llvq-quant/tests/g5_gptq.rs:825`) exige le même facteur que `faer`. Ne pas le
supprimer. L'encodeur (plus proche voisin) tourne hors ligne une fois par modèle ;
le décodeur (index → vecteur) tourne à chaque GEMM, en décalage et masquage. Ne
jamais optimiser l'un en pensant à l'autre. Les dérivations sont verrouillées
par `classes_reproduce_theta_series` et `even_repair_matches_dp_reference` :
cosets de Λ₂₄, flip de parité unique, sommes télescopiques, objectif global
de `nearest_scaled` plus rapide que `shell_bests`. Leur raison est dans
`docs/HISTORIQUE.md`, entrée « Fondations, G1 à G4 ».
`bin/fusedrun` est le noyau dans le modèle ; `bin/run` est la démo dense,
non portée.

## Commandes

Deux boucles de test, deux ordres de grandeur. `cargo test` en debug ignore
les tests lourds (`cfg_attr(debug_assertions, ignore)`) et tourne en minutes.
`cargo test --release -- --include-ignored` se compte en dizaines de minutes
(*mesuré* le 2026-08-08 : 17 min sans finir `llvq-artifact`, sans journal).
Les tests d'archive scellée portent un `#[ignore]` inconditionnel, exigent
`~/llvq-q4b.llvq` et échouent en le nommant s'il manque.

```bash
cargo test                                            # boucle rapide
cargo test --release -- --include-ignored             # suite complète, avant tout commit de format
cargo clippy --all-targets                            # zéro warning
cargo run --release -p llvq-bench --bin llvq-bench    # aussi : encbench, betasweep, decbench, classhist
cargo run --release -p llvq-metal --bin thesis        # macOS ; aussi : matvec, decreal, mslcheck, p1v0, rankbench
cargo run --release -p llvq-cuda --bin planesbench -- <model.llvq>   # Linux + CUDA ; aussi : preflight
LLVQ_MODEL=Qwen/Qwen3-4B LLVQ_CALIB=c4 LLVQ_ARTIFACT=q4b.llvq cargo run --release -p llvq-llm --features metal,fast-linalg --bin smoke -- 64 2048 12 4096 metal nogs leech1c12 999 rot   # requantifier le 4B (4,01 h = 14 447 s sur M3 Max, mesuré, docs/fiche-4b.md §3.4)
#   positionnels : n_calib · calib_len · n_eval · eval_ctx · device · gs/nogs · codebook (suffixe f = magnitude libre, L<n> = plafond) · limit · rot
LLVQ_MODEL=Qwen/Qwen3-4B cargo run --release -p llvq-llm --bin seal -- q4b.llvq qwen3-4b-llvq.bin   # scellé attendu 1,771 Go (mesuré, docs/fiche-4b.md)
cargo run --release -p llvq-llm --features metal --bin mmlu -- <checkpoint|scellé> metal 40
cargo run --release -p llvq-cuda --bin nullkbench                     # le plancher ; tout bin de l'image va dans `cargo build --bin` ET le COPY de ops/Dockerfile.cuda
uv run ops/awq_speed.py … | uv run ops/awq_dequant.py check           # AWQ : révision non épinglée refusée ; verrous L1/L2/L4
cargo run --release -p llvq-llm --features metal --bin oracle              # passe avant vs candle, à chaque backend
LLVQ_DTYPE=f16 cargo run --release -p llvq-llm --features metal --bin ppl -- 4096 12 metal <scellé>
cargo run --release -p llvq-llm --features cuda --bin fusedrun             # le noyau dans le modèle
uv run ops/run.py estimate|selftest|publish|oracle|launch|monitor         # HF Jobs
uv run --with opentimestamps ops/otsaudit.py                              # état des ancres .ots
```

Autres binaires de `llvq-llm` : `mmlu`, `mmlupair`, `embedq`, `seal`. Ceux de
`llvq-bench` comprennent aussi `rtbits` (comptabilité b/param) et `radixstudy`
(E3). `rankbench` refuse de démarrer sans
`proofs/preregistration-p1-2026-08-13.md.ots`.

### Variables d'environnement

| variable | valeurs | effet |
|---|---|---|
| `LLVQ_FUSED_LAYOUT` | `planes14` (défaut), `planes12x`, `slot32`, `golay70` | layout VRAM du noyau fusé ; toute autre valeur est refusée |
| `LLVQ_EMBED` | `f16` (défaut), `q8` | embedding quantifié au chargement ; `q8` est la config servie |
| `LLVQ_KV` | `f16` (défaut), `q8` | cache KV int8, livré, pas défaut (contexte court seulement) |
| `LLVQ_ROT_SHARE` | `0`, `1` | une rotation par groupe de projections ; servi = `1` |
| `LLVQ_FUSE` | `0`, `1` | fusion q+k+v et gate+up ; servi = `1` ; `FUSE=1` avec `ROT_SHARE=0` refusé |
| `LLVQ_FUSE_AB` | `1` | `fusedrun` : les deux bras de la fusion dans un seul processus, la forme de D1 |
| `LLVQ_TIME_PHASES` | `1` | `fusedrun` : profil par phase, hors protocole publié |
| `LLVQ_DTYPE` | `f32` (défaut de `ppl`), `f16` | dtype d'évaluation ; comparer ppl et MMLU exige le même des deux côtés |
| `LLVQ_CALIB` | `wikitext2` (défaut), `c4`, `wikitext2-test` | `smoke` : corpus de calibration ; `c4` est le protocole du papier |
| `LLVQ_ARTIFACT` | chemin | `smoke` : écrit l'artefact compressé, index empaquetés ; absent, rien n'est écrit |
| `LLVQ_RESUME` | chemin d'un shard | `smoke` : reprise depuis ce shard ; exige `LLVQ_ARTIFACT` |
| `LLVQ_SEALED_ARTIFACT` | chemin | tests d'archive de `llvq-artifact` : déplace la recherche du fichier scellé (`tests/common/mod.rs`) |
| `LLVQ_CALIB_SEED` | entier | fenêtres de calibration tirées au hasard au lieu du préfixe (`smoke`) |
| `LLVQ_DAMPING` | flottant | amortissement relatif de la hessienne (`smoke`) |
| `LLVQ_H_SHRINK` | ρ dans [0, 1], défaut `1` | `H ← ρ·H + (1−ρ)·diag(H)` avant rotation (`smoke`, bouton M1) |
| `LLVQ_RESTORE_F16` | types de projection séparés par virgule, ou `all` | `mmlu`, `ppl` : ces types repris du checkpoint en f16, le reste tel que servi |
| `LLVQ_RESTORE_Q4` | même liste | idem en int4 g128 ; poser les deux est refusé |
| `LLVQ_MODEL` | dépôt HF ou répertoire local | checkpoint ; exigé par `RESTORE_*`, jamais défaut dans `mmlu` |
| `LLVQ_THREADS` | entier | plafond du pool d'encodage (`smoke`) ; ncpu−4 et `nice` sur une machine partagée |
| `LLVQ_NVRTC_ARCH` | `compute_NN`, défaut `compute_89` | cible NVRTC ; `compute_80` pour A100 ; autre forme refusée |
| `LLVQ_TIME_EVENTS` | `1` | span device par events CUDA (`planesbench`), hors protocole publié |
| `LLVQ_BENCH_ARMS` | phases séparées par `;` | bras de `planesbench` ; nom inconnu refusé |
| `LLVQ_QTIP_DIR` | répertoire | noyau QTIP amont, GPL v3, non redistribué (`docs/qtip-provenance.md`) |

`LLVQ_KV_PREALLOC`, `LLVQ_GRAPH_AB` et `LLVQ_SEG_ARMS` sont des modes de
mesure, jamais une config servie.

## Interdits

Le détail et les raisons sont dans `docs/METHODE.md`.

1. Ne lance ni n'arrête aucun run et ne prends aucune décision structurante sans go explicite ; annonce le coût avant, le cumul après.
2. Tamponne le préreg avant la première mesure ; n'édite jamais un préreg tamponné, écris l'écart à côté.
3. N'implémente pas A2 (CUDA Graphs) dans le cœur ; ne rouvre ni A2, ni E1v, ni Golay70 hors des conditions écrites dans `docs/ETAT.md` §7 (A2 : `proofs/preregistration-a2-a3-geometrie-2026-08-31-ECARTS.md` §É7).
4. Ne publie jamais le rapport brut seul ; donne toujours le rapport à tête identique.
5. Ne divise ni × inter-cartes (L40S, A100) ni × inter-piles (vLLM, nous) ; compare AWQ et QTIP en Go/s.
6. Dis toute comparaison mémoire en b/param modèle entier, embedding compris.
7. Publie des médianes à plage formées round par round, jamais un quotient de deux minima.
8. Étiquette chaque chiffre *mesuré*, *calculé* ou *estimé*, avec sa comptabilité et son journal.
9. Avant de chiffrer un rejeu, épuise `hf buckets ls`, `hf jobs logs`, `hf jobs inspect` ; garde la sortie brute.
10. Un test qui saute faute d'archive échoue en la nommant ; mute le code avant de déclarer un gate vert ; `oracle` d'abord sur chaque backend.

## Conventions

- Commentaires et docs en anglais dans le code ; échanges en français.
- Zéro warning à `cargo clippy --all-targets`.
- `docs/STYLE.md` pour tout document vivant ; un fait qui change se remplace, l'ancien va dans `HISTORIQUE.md` avec sa date.
- Les cinq crates du cœur restent sans dépendance externe et en `forbid(unsafe_code)`.
- Deux paliers d'`ignore` : `cfg_attr(debug_assertions, ignore)` pour le calcul, `#[ignore]` inconditionnel pour les archives scellées.
- Suite complète avant tout commit qui touche un format ou l'indexage ; toute modification de la carte d'index casse le format v1 (`codebook_fingerprint` l'épingle).
- En cas de désaccord entre documents, l'objet gagne : le fichier scellé, le journal. `docs/fiche-4b.md` fait foi sur le fichier publié, `paper/` sur ce qui est soumis ; README et fiche priment sur cette carte.
- Tout run payant passe par `ops/run.py` avec `--features fast-linalg`, et le job va dans `docs/data/jobs.csv`.
