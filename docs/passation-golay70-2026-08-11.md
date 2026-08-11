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

## 1. État de la branche (2 commits au-dessus de `main`)

| commit | contenu |
|---|---|
| `e66e4ad` | `docs/projections-golay70-2026-08-11.md` — projections + analyse |
| `b41a476` | **la v2** : `llvq-cuda/kernels/llvq_golay.cuh` (prologue par bloc `hw`/`aw`/`nw`, slot à 3 masques), `llvq-cuda/tests/host_golay70.cpp` (le harnais exécute le même texte), note ✅ dans le doc de projections |

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
2. **Lot A — pré-enregistrer le critère dans `proofs/`**, avant tout job.
   Le §3 du spec, plus les deux corrections du doc de projections : la
   comparaison mémoire à l'embedding q8 se facture à **8,5** b/param (pas
   8,0 — le seuil « ≤ 4,1 » du 4B correspond à **4,065**, pas 4,016), et
   l'invariant transposable est la **marge ≥ 20 % vs l'AWQ déployé du même
   modèle**, pas le 4,1 absolu (le 8B projeté le viole à 4,29 avec la
   meilleure marge du tableau). Signer et horodater comme le précédent.
3. **Lot B — le sélecteur de bras** (`planesbench`). Dette É1 du
   pré-enregistrement du 08-10, toujours ouverte : sans lui, le job v2
   rejouera l'entorse « pas de contrôle dans le même processus ». Un bras
   écarté ne doit pas construire ses tampons.
4. **Lot C, moitié carte — LE job**, sept bras avec contrôle, L40S, même
   protocole que `six-arm-awq` (7 rounds, 2 jetés, rapports round par
   round). À lire au démarrage : registres et `local_size_bytes == 0` pour
   `tv_golay70` — le prologue ajoute des registres vifs (v1 : 40, 0 o
   local) ; un spill invaliderait l'estimation avant même les rounds.
   Verdict par le critère du lot A, pas par enthousiasme : la fourchette
   estimée est 1,9–2,4× pour un seuil à 2,0× — ça peut échouer, c'est prévu
   pour.
5. **Si adopté** : `fusedrun` avec un layout `golay70` (câblage
   `LLVQ_FUSED_LAYOUT` **non écrit** à ce jour — `fused.rs` n'admet que
   `planes14|planes12x|slot32`), tokens gloutons contre le bras dense, et
   la mesure tok/s qui tranchera l'additivité (piège n°4).

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
git log --oneline -3        # b41a476, e66e4ad au-dessus de main
git diff main --stat        # 4 fichiers : .cuh, .cpp, 2 docs
cargo test -p llvq-cuda --release --test golay70_decoder_matches_rust   # Mac dev
```
