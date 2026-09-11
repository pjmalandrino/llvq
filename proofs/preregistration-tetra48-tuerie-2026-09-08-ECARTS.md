# Écarts — pré-enregistrement de la tuerie tetra48 du 2026-09-08

Le prereg est tamponné (`d60850bd6ed1c5fe28872a5660d82d1a4a215870b8fb71071d2a0120c96b7af9`,
quatre calendriers) et ne s'édite plus. Tout ce qui s'en écarte est ici.

**Aucun écart ne touche §4 (la règle de décision) ni §5 (la prédiction
signée).** Les trois ci-dessous sont de la plomberie de lancement, et deux
d'entre eux ont brûlé de la carte.

---

## É1 — Le premier lancement est mort en `exit 127` : le CLI a mangé le `-lc`

**Job `6aa0117132d5d0c22c5ad753`, `l40sx1`, mort ~20 s après `RUNNING`.**

J'ai lancé par `hf jobs run … bash -lc "…"`. Le parseur d'options du CLI a
consommé `-lc`, et `bash` a reçu la chaîne entière comme un nom de fichier.
Vérifié, pas déduit — `inspect_job` rend :

```
command: ['bash', 'mkdir -p /out/tetra48-tuerie-2026-09-08 && f1rankfloor …']
```

et le log :

```
bash: mkdir -p /out/… && f1rankfloor …: No such file or directory
```

Corrigé en passant par `huggingface_hub.run_job()`, qui prend la commande
comme une **liste** et ne parse rien — la route que `ops/run.py cmd_launch`
utilise depuis le début. Je ne l'ai pas prise d'emblée et c'est le seul motif
de cet écart.

Même classe que le `hf: command not found` du 2026-09-06 : de la plomberie de
lancement, jamais du calcul.

## É2 — Trente minutes perdues à attendre le mauvais état de la Space

**Coût 0 $, coût en horloge une demi-heure.**

La Space de construction d'image passe par :

```
[    0s] BUILDING
[  868s] APP_STARTING
[ 2727s] RUNTIME_ERROR
```

**L'image existe et est utilisable dès `APP_STARTING`, à 868 s.** Le
`RUNTIME_ERROR` de 2 727 s est le health-check qui expire — la Space ne sert
rien, son `CMD` est `sleep infinity` (`ops/Dockerfile.cuda:148`). J'ai attendu
l'état terminal.

Le dossier connaissait déjà ce piège d'une session précédente ; il n'était
écrit nulle part. Il l'est maintenant : **le signal de fin de construction est
`APP_STARTING`.**

## É3 — Un `#include` non gardé : vert sur le Mac, catastrophique sur la carte

**Job `6aa01601900620b5c77e257c`, `l40sx1`, mort ~20 s après `RUNNING`.**

`llvq_tetra48.cuh` portait un `#include "llvq_f1rank_v3.cuh"` **inconditionnel**
là où tout le reste de l'arbre écrit :

```c
#ifndef LLVQ_F1RANK_V3_CUH
#include "llvq_f1rank_v3.cuh"
#endif
```

Sortie de la carte :

```
NVRTC source: 117172 bytes, sha256 6bc4e771a34e1a9abe3cd11f597c38b2a8ab3c948045fe4d2397799a6a1ba2b2
NVRTC_ERROR_COMPILATION
  default_program(2043): catastrophic error: cannot open source file "llvq_f1rank_v3.cuh"
```

**Ce qui compte ici n'est pas la faute, c'est que rien ne pouvait la voir.**
Le fichier a passé : `cuhcheck` (parse propre), `cargo clippy --all-targets`
sur l'hôte **et** sur `x86_64-unknown-linux-gnu`, et les cinq tests
d'exactitude bit à bit. Parce que `cuhcheck` parse avec `clang++`, **qui a un
système de fichiers et résout l'include sans un mot**. NVRTC n'en a pas : le
concaténateur d'hôte est la seule source, et un include non gardé est une
erreur catastrophique — après que la carte est louée et l'image tirée.

Le correctif est de classe, pas d'instance : `cuhcheck` balaie désormais les
33 sources `.cu`/`.cuh` et refuse tout `#include "…"` sans `#ifndef` dans les
trois lignes non vides au-dessus. Mutant vérifié — retirer la garde relève le
drapeau par nom et par ligne.

Commit `e21c8eb`.

---

## Coût des écarts

| job | ce qui s'est passé | facturé (*estimé*) |
|---|---|---|
| `6aa0117132d5d0c22c5ad753` | É1, `exit 127` en ~20 s | ~0,01 $ |
| `6aa01601900620b5c77e257c` | É3, NVRTC en ~20 s | ~0,01 $ |

**~0,02 $ de plomberie**, à réconcilier sur le registre. Le banc lui-même n'a
pas encore tourné, et sa prédiction signée reste intacte et non lue.

## É4 — La garde d'architecture testait une égalité, et a refusé un run correct

**Job `6aa0397b32d5d0c22c5addbf`, `rtx-pro-6000`, mort ~20 s après `RUNNING`.**

```
tv_nullk: compiled for sm_120, not sm_89.
```

`gpu.rs:415` exigeait `binary_version == arch_binary_version()`. Or NVRTC émet
du PTX et le PTX est compatible **vers l'avant** : le pilote JIT un module
`compute_89` pour n'importe quel sm ≥ 89 — la phrase même avec laquelle
`ops/run.py` épingle `MIN_COMPUTE_CAP`. Une RTX PRO 6000 rend 120 parce que le
mécanisme **fonctionne**.

`compute_120` n'était pas une issue : l'image est CUDA 12.4 et sm_120 est
arrivé en 12.8, donc son NVRTC ne connaît pas cette architecture.

L'égalité tenait partout où elle avait été exercée — `compute_89` sur L40S,
`compute_80` sur A100 en F4 — parce que chaque run nommait le sm de sa propre
carte. La première carte plus récente que son `LLVQ_NVRTC_ARCH` l'a cassée.

Corrigé en ordre : `binary >= compiled_for`. Rien n'est perdu — le repli
silencieux vers `compute_75` que l'ancien message invoquait n'était déjà pas
attrapé par l'égalité, puisque sur une carte de sm ≥ la demande
`binary_version` rend le sm du **device** quel que soit le PTX d'origine.

Le prédicat est sorti du module `cfg(linux)` vers `crate::arch_binary_ok`, pour
la raison que `lib.rs` donne déjà à propos de `f16_bits` : la machine de dev
n'a pas de CUDA, et un prédicat que personne ne peut muter sur la machine de
dev est un prédicat que personne ne vérifie. Trois mutants tués — retour à
l'égalité (le bug d'origine), garde désactivée, sens inversé. Commit `c878c2e`.

## É5 — Le banc a tourné sur RTX PRO 6000, pas sur la L40S de §5

**Écart de protocole, déclaré avant d'être exploité.**

Le job `6aa01d85900620b5c77e27bd` sur `l40sx1` a attendu **99 minutes** sans
jamais démarrer, au-delà du maximum historique de 50 min
(`f1-rang-variantes-2026-09-05`). Sur go de l'opérateur, un job parallèle a été
lancé sur `rtx-pro-6000` ; il a rendu en 46 s et le L40S a été annulé **en
file, sans jamais démarrer — 0 $, jamais facturé**.

Conséquence, écrite d'avance dans le prereg §5 et tenue :

- les **absolus** de la prédiction signée (nullk 2,18 ; f1r_v3 3,88) sont
  ancrés L40S et **ne sont pas lus** ;
- **R est lu**, parce qu'il est intra-processus : les deux bras tournent dans
  le même processus, sur les mêmes tours, sur la même horloge, et la règle 5
  n'est pas sollicitée.

Ce qui reste ouvert et que ce banc ne tranche pas : le verdict dépend-il de
l'architecture. Indice mesuré — `t(f1r)/t(nullk)` vaut **2,4584** ici contre
**2,21** en référence L40S, donc le décodage F1 est relativement plus cher sur
Blackwell. Le job `6aa10ba8900620b5c77e5a7d` rejoue le même banc sur `l40sx1`
pour répondre.
