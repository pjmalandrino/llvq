# Passation — Golay70 v2, branche `claude/golay70-memory-performance-ksbbzs`

**2026-08-11.** Ce document fait foi sur **l'état** du chantier Golay70 et
sur **ce qui reste** ; il ne recopie aucun chiffre ni aucune analyse. Les
références, dans l'ordre de lecture d'une session neuve :

1. [`spec-apres-awq-2026-08-10.md`](spec-apres-awq-2026-08-10.md) — pourquoi
   E2 est rouvert, le critère pré-enregistré (§3), les lots. Ses §4 et §6
   portent des notes datées du 08-11 qui renvoient ici — les lire, elles
   corrigent des pistes périmées.
2. [`projections-golay70-2026-08-11.md`](projections-golay70-2026-08-11.md) —
   **la référence pour tous les chiffres** : projections b/param 8B→70B
   (§2), audit du goulot (§3.1), algèbre de la v2 (§3.2), fourchettes (§3.3).
3. Ce fichier — statut, commandes, pièges, ordre de bataille.

---

## 1. État de la branche

| commit | contenu |
|---|---|
| `e66e4ad` | `docs/projections-golay70-2026-08-11.md` — projections + analyse |
| `b41a476` | **la v2** : `llvq-cuda/kernels/llvq_golay.cuh` (prologue par bloc `hw`/`aw`/`nw`, slot à 3 masques), `llvq-cuda/tests/host_golay70.cpp` (le harnais exécute le même texte), note ✅ dans le doc de projections |
| `07eb969` | ce document |
| `9402e4e` | **lot A** : `proofs/preregistration-2026-08-11.md` + recensement E2 du 8B |
| `f56ae30` | ancrage OpenTimestamps du pré-enregistrement |
| `38ff87c` et suivants | **lot B** : sélecteur `LLVQ_BENCH_ARMS`, bras témoin `golay70_v1.cu`, corrections de la revue adversariale |

**Ce que la v2 est** : une réécriture du décodage seul. Zéro octet de format
changé — `golay70_fields`, le flux 9 o, les exceptions, `GolayClassRec`, la
signature de `tv_golay70` et la concaténation NVRTC sont intacts. Un
transcodage existant reste valable ; aucun artefact à refaire.

## 2. Ce qui est prouvé, et par quoi

Exécuté sur la machine du port (Linux sans CUDA — voir piège n°2),
`cargo test -p llvq-cuda --release --test golay70_decoder_matches_rust` :

- reconstruction exacte contre Slot32 — 20 767 blocs, 2 425 exceptions,
  toutes les classes traversées (violantes E2 comprises) ✅
- lancement émulé contre la référence f64 — pire erreur 2,6e-8·Σ|w·x| ✅
- records hand-packed conformes aux offsets figés, 4 décalages de fenêtre ✅
- probe compilé `clang++ -Wall -Wextra -Werror` (donc `tv_golay70`
  type-checke) ✅
- **3 mutants tués sur le prologue** — XOR du mot de signe supprimé, mux de
  flags croisé (m1↔m2), mot haut croisé (`hw = odd ? cw : bm`) : chacun fait
  échouer `the_golay70_kernel_decides_what_the_rust_decoder_decides`, le
  code restauré repasse. Ne pas les redemander ; en ajouter si le prologue
  bouge.

## 3. Ce qui N'est PAS prouvé — l'ordre de bataille

Dans l'ordre, chaque étape débloquant la suivante :

1. **Le sweep de l'artefact scellé a SAUTÉ** (`SKIP:` — `~/llvq-q4b.llvq`
   absent de la machine du port). À passer sur le Mac de dev, où le fichier
   est :
   ```bash
   cargo test -p llvq-cuda --release --test golay70_decoder_matches_rust
   # attendu : the_sealed_artifact_decodes_identically_through_golay70
   # balaye 150 681 600 blocs, ~7,44 % d'exceptions — plusieurs minutes
   ```
   ⚠️ Un « ok » en ~1 s = le fichier manque, pas une preuve (le piège du §2
   de `CLAUDE.md`, à la lettre).
   > ✅ **Passé le 2026-08-11 sur le Mac de dev, preuve positive à l'appui**
   > (relancé en `--nocapture` pour lire le compte, pas seulement le vert) :
   > `sealed sweep: 150681600 blocks verified identical in Golay70 and
   > Slot32, 11204181 exceptions (7.4357 %), payload 3.4461 b/w`, en ~44 s —
   > le « plusieurs minutes » ci-dessus était une estimation prudente, le
   > balayage est parallélisé. Les trois chiffres recoupent le doc de
   > projections au chiffre près.
2. **Lot A — pré-enregistrer le critère dans `proofs/`**, avant tout job.
   Le §3 du spec, plus les deux corrections du doc de projections : la
   comparaison mémoire à l'embedding q8 se facture à **8,5** b/param (pas
   8,0 — le seuil « ≤ 4,1 » du 4B correspond à **4,065**, pas 4,016), et
   l'invariant transposable est la **marge ≥ 20 % vs l'AWQ déployé du même
   modèle**, pas le 4,1 absolu (le 8B projeté le viole à 4,29 avec la
   meilleure marge du tableau). Signer et horodater comme le précédent.
   > ✅ **Fait le 2026-08-11** : `proofs/preregistration-2026-08-11.md`,
   > horodaté OpenTimestamps (`.ots` commité), puis **corrigé et
   > ré-horodaté le jour même** après revue adversariale (quatre défauts de
   > présentation des nombres, consignés dans son §7bis É0 — aucun verdict
   > ne bouge) ; la signature GPG reste à l'opérateur. Il acte aussi que la
   > condition MÉMOIRE est déjà un compte (4,065 contre une borne de
   > 4,2416) : le job du lot C ne tranche que la vitesse. Et la réserve §2.4 des projections est close le même jour par
   > le recensement E2 du 8B — 7,4116 % contre 7,4357 % au 4B, « pair
   > violant » 4,0394 % contre 4,05
   > ([`mesures/classhist-e2-8b-2026-08-11.txt`](mesures/classhist-e2-8b-2026-08-11.txt)).
3. **Lot B — le sélecteur de bras** (`planesbench`). Dette É1 du
   pré-enregistrement du 08-10, toujours ouverte : sans lui, le job v2
   rejouera l'entorse « pas de contrôle dans le même processus ». Un bras
   écarté ne doit pas construire ses tampons.
   > ✅ **Fait le 2026-08-11** : `LLVQ_BENCH_ARMS` — phases monotones dans un
   > même processus (contrôle puis table), bras écarté sans transcode ni
   > tampon device, TU NVRTC invariante à la sélection, noms inconnus
   > refusés (« golay70 » nu est refusé comme ambigu : nommer `golay70v1`
   > ou `golay70v2`), ordre de dispatch inchangé, `Δ_contrôle` imprimé.
   > Parseur portable testé sur le Mac (`llvq-cuda/src/arms.rs`, 11 tests) ;
   > le module linux entier est type-checké et clippé à zéro warning depuis
   > le Mac via `CUDARC_CUDA_VERSION=12040 cargo clippy -p llvq-cuda
   > --target x86_64-unknown-linux-gnu --all-targets`.
   > **Et le bras témoin v1 est câblé** : `kernels/golay70_v1.cu`, copie
   > gelée du décodeur publié (trois symboles renommés, tout le reste
   > partagé), tampons partagés avec la v2 (zéro VRAM ajoutée), prouvé
   > **bit à bit égal à la v2** par le harnais hôte sur toutes les classes —
   > le rapport v2/v1 du lot C se formera dans les mêmes rounds, comme le
   > §4 du pré-enregistrement l'exige.
4. **Lot C, moitié carte — LE job**, sept bras avec contrôle, L40S, même
   protocole que `six-arm-awq` (7 rounds, 2 jetés, rapports round par
   round). À lire au démarrage : registres et `local_size_bytes == 0` pour
   `tv_golay70` — le prologue ajoute des registres vifs (v1 : 40, 0 o
   local) ; un spill invaliderait l'estimation avant même les rounds.
   Verdict par le critère du lot A, pas par enthousiasme : la fourchette
   estimée est 1,9–2,4× pour un seuil à 2,0× — ça peut échouer, c'est prévu
   pour.
   > 🧭 **Prêt à lancer depuis le 2026-08-11 — il ne manque que le go** (et
   > le go est à l'opérateur : coût ~1 $, campagne kernel à 1,56 $ dépensés
   > sur 15 $ de plafond). La sélection exacte du job, protocole du
   > pré-enregistrement du 08-11 §4 :
   > ```
   > LLVQ_BENCH_ARMS="slot32,planes14,planes12x,golay70v1,fp16,awq;slot32,planes14,planes12x,golay70v1,fp16,awq,golay70v2"
   > ```
   > phase 1 = le jeu du run publié du 08-10 (contrôle), phase 2 = + la v2.
   > À l'arrivée : consigner le coût dans `docs/data/jobs.csv` et ouvrir
   > `ops/manifest.jsonl` (piège n°5).
5. **Si adopté** : `fusedrun` avec un layout `golay70` (câblage
   `LLVQ_FUSED_LAYOUT` **non écrit** à ce jour — `fused.rs` n'admet que
   `planes14|planes12x|slot32`), tokens gloutons contre le bras dense, et
   la mesure tok/s qui tranchera l'additivité (piège n°4).
   > ✅ **Le câblage est écrit le 2026-08-11, en avance sur le verdict** —
   > être sélectionnable est ce qui rend mesurable, et le critère décide de
   > SERVIR, pas de câbler. `LLVQ_FUSED_LAYOUT=golay70` : noyau modèle
   > `tv_golay70_h` (`llvq-llm/kernels/tv_golay70_h.cu`, le motif row_exc de
   > `tv_planes12x_h` — pas de memset, pas d'atomic, corrections par tranche
   > de ligne ; en PLUS simple : rien à soustraire, le flux principal porte
   > l'origine aux exceptions), transcodage par `transcode_golay70` de
   > `llvq_artifact::runtime` (celui que le sweep scellé prouve), tables GPU
   > construites de la même dérivation (`fused::golay70_gpu_class_table`).
   > Verrous locaux : harnais `tests/host_golay70_h.cpp` (le décodeur v2 et
   > la correction EXÉCUTÉS contre une référence f64 depuis les indices
   > sources — reconstruction exacte, pas d'algèbre d'overlay) + test de
   > l'unité de traduction (golay70 prolonge planes12x ; ni `tv_golay70` de
   > banc ni `tv_golay70_v1` dans l'unité d'inférence). ⚠️ Le noyau lui-même
   > reste **compile-only** localement : grille, barrières, `warp_sum`, store
   > f16 ne s'établissent que sur carte — le premier `fusedrun` diffe ses
   > tokens gloutons contre le bras dense, comme pour `Planes12x`. À noter
   > pour le 8B : `down_proj` y porte ~38 exceptions par ligne, au-delà des
   > 32 voies — une seconde passe série, pas un mur, mais le nombre à
   > surveiller (commentaire d'en-tête du noyau).

## 4. Les pièges de reprise

1. **Ne pas réinventer les pistes du spec §4.** « Warps par coset » est
   subsumée par la v2 (après hissage, le chemin par slot est aveugle au
   coset — il ne reste ~10 ops de prologue par bloc à spécialiser) ; « XOR
   côté pair » est caduque (le noyau lit `cwtab`, aucun ré-encodage
   n'existe). C'est écrit dans le spec en note datée, mais l'ancienne prose
   est toujours au-dessus.
2. **Machine sans CUDA** : le `build.rs` de cudarc exige `nvcc` et les
   `.so`. Contournement **d'environnement seulement, rien à commiter** :
   ```bash
   CUDARC_CUDA_VERSION=12040 \
   CUDA_HOME=<répertoire avec lib64/lib{cuda,nvrtc,cublas,cublasLt}.so vides> \
   RUSTFLAGS="-C link-arg=-Wl,--unresolved-symbols=ignore-all" \
   cargo test -p llvq-cuda --release --test golay70_decoder_matches_rust
   ```
   (`CUDARC_CUDA_VERSION` est lu **avant** la sonde `nvcc` ; les stubs se
   créent par `gcc -shared -o lib$l.so -x c /dev/null`-équivalent.) Sur le
   Mac de dev comme dans l'image CUDA, la commande nue suffit.
3. **Aucune milliseconde n'existe pour la v2.** Les 1,9–2,4× sont un compte
   d'instructions, pas un profil ni une mesure. Ne rien publier, ne rien
   décider là-dessus — c'est le job du lot C.
4. **Les tok/s de l'arbitrage produit sont contestés entre deux documents** :
   le spec §7 dit ~55 tok/s pour Golay70-v1 au 4B (report du rapport de
   banc), les projections §1 disent ~69 (additivité, 5,1 ms de linéaires
   dans un token de 11,3). Aucun des deux n'est mesuré. Le premier
   `fusedrun` tranchera ; d'ici là, citer les deux ou aucun.
5. **`ops/manifest.jsonl` n'existe toujours pas** (dette du spec §5). Le job
   du lot C est une bonne première entrée.

## 5. Vérification rapide d'arrivée

```bash
git log --oneline main..HEAD   # e66e4ad → lot B, cf. la table du §1
cargo test -p llvq-cuda --release --test golay70_decoder_matches_rust   # Mac dev
cargo test -p llvq-cuda --lib                                            # parseur LLVQ_BENCH_ARMS
```

*(La première version de ce bloc figeait « 2 commits, 4 fichiers » — périmé
dès le commit suivant, exactement le motif que ce dépôt documente. Un compte
de commits ne se met pas dans un document de passation ; la table du §1
liste les contenus, `git log` donne le compte du jour.)*
