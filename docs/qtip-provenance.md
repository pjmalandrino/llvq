# QTIP : pourquoi ce noyau n'est pas dans le dépôt (2026-08-20)

Le bras de comparaison `qtip` du banc mesure le noyau matvec 2 bits publié par
Cornell-RelaxML. Contrairement au bras AWQ, **son code n'est pas commité ici** :
il est récupéré au moment du job par [`ops/fetch-qtip.sh`](../ops/fetch-qtip.sh).

## La raison, et elle n'est pas symétrique du bras AWQ

| | AWQ | QTIP |
|---|---|---|
| licence amont | MIT | **GPL v3** |
| dans le dépôt | oui — `llvq-cuda/kernels/awq_gemv.cu`, `include_str!` | **non** |
| chemin de chargement | embarqué, `LLVQ_KERNEL_DIR` en surcharge | `LLVQ_KERNEL_DIR` **seul** |

Le dépôt est sous MIT OU Apache-2.0. Redistribuer un fichier GPL v3 dedans
obligerait à replacer l'ensemble sous GPL v3 — ce n'est pas le choix du projet.
La récupération job-time évite la question sans rien perdre de la mesure.

⚠️ **Ce que la GPL contraint est la DISTRIBUTION, pas l'usage.** Exécuter un
logiciel GPL pour produire une mesure, le patcher pour le compiler, le
chronométrer, publier ses temps : rien de tout cela n'est restreint. 🕳️ Le plan
du matin ([`plan-f2-qtip-2026-08-20.md`](plan-f2-qtip-2026-08-20.md)) présentait
« sans exécuter une ligne de leur Python » comme une contrainte juridique ;
c'était une confusion. C'était un choix de simplicité — et il se trouve qu'on
n'en a pas eu besoin, le format ayant été dérivé du CUDA puis validé contre une
transcription de leur code.

## Ce que le script fait, et ce qu'il refuse de faire

1. Refuse d'écrire dans un répertoire non vide.
2. Télécharge `inference.cu` et `inference.h` au commit **épinglé**
   `e90c6688c8dfae326a3a81b5eb032db7c6680ec0`.
3. **Vérifie les deux sha256** et échoue bruyamment sinon. C'est le point qui
   compte : un fichier amont qui changerait en silence déplacerait un chiffre
   publié sans que rien ne le dise.
4. Retire **quatre lignes mortes** — `#include <cuda/pipeline>`, `#include
   <mma.h>`, `#include <c10/cuda/CUDAStream.h>`, `using namespace nvcuda;`.
   Chacune a été vérifiée à **zéro usage** dans le fichier le 2026-08-20 (le MMA
   passe par de l'asm PTX inline, pas par l'API `wmma`), donc le code device
   généré est inchangé : elles tombent parce que NVRTC ne porte ni torch ni
   libcu++. Les macros `CHECK_CUDA`/`CHECK_CONTIGUOUS` contiennent `TORCH_CHECK`
   mais ne sont jamais développées dans ce fichier — elles sont **laissées
   telles quelles** plutôt qu'éditées.
5. **Prouve** le patch après coup (re-grep des quatre lignes et de quatre
   jetons résiduels) au lieu de faire confiance au filtre.
6. Écrit un `PROVENANCE.txt` à côté : URL, commit, sha256 avant patch, lignes
   retirées, date, et la mention de licence.

## La limite honnête, à déclarer dans le papier

**Notre dépôt seul ne rejoue pas ce bras.** Il faut le réseau et le dépôt amont
vivant au commit épinglé. Les autres bras du banc se rejouent hors ligne ; celui
-ci non. Si l'amont disparaît, les sha256 de `PROVENANCE.txt` permettent encore
d'authentifier une copie retrouvée ailleurs, mais ils ne la fabriquent pas.

C'est le prix de la licence, il est déclaré, et il ne se contourne pas en
copiant le fichier ici.
