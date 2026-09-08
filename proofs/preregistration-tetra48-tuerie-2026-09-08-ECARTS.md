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
