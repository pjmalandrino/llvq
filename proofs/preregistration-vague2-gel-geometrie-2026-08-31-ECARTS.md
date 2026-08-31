# Écarts au pré-enregistrement vague 2 (2026-08-31)

> Fichier nommé d'avance par le préreg lui-même
> (`preregistration-vague2-gel-geometrie-2026-08-31.md`, l.9-10 : « Ce document
> ne s'édite jamais. Écarts → … nommé ici d'avance »). Le préreg n'est pas
> touché ; ce qui a dévié s'écrit ici, avec sa date et son mécanisme.

## É1 — A1 est mort au probe : le binaire n'était pas dans l'image (2026-08-31, 0,01 $)

Job `6a954da821c5aa7c83649476` (l40sx1, 15 s facturées, *mesuré*,
`docs/data/jobs.csv:90`) : `which nullkbench` échoue, le job s'arrête avant
toute mesure — exactement le rôle du probe (gate G2 du protocole
piles-isolées v2, réemployé ici). Cause : `nullkbench` absent des **deux**
listes explicites de `ops/Dockerfile.cuda` (`cargo build --bin` et `COPY`).
Corrigé par `c6642e4`, image republiée.

⚠️ La ligne du registre dit « nullkbench compile mais absent des DEUX
listes » — le « compile » était un constat **macOS**, donc vide : le corps
entier du bin est sous `#[cfg(target_os = "linux")]` et le Mac ne compile
qu'un stub. Voir É2.

## É2 — Première compile Linux : `BUILD_ERROR` du Space (2026-08-31 10:07:28 UTC, 0 $)

La republication de É1 a fait compiler `nullkbench` sur Linux **pour la
première fois de son existence**, et le build est mort :
`error[E0599] no method named 'arg' … trait PushKernelArg … not in scope`
(`nullkbench.rs:139`, log de build du Space, *mesuré*). Le trait qui fournit
`.arg()` sur `LaunchArgs` n'était pas importé — `planesbench.rs:80` l'importe,
même idiome. Corrigé par `970d27d` (une ligne).

Conséquence pour la passation de session : « rebuild en cours, finit tout
seul » était faux — le Space est resté en `BUILD_ERROR` ~9 h 40, jusqu'à la
reprise du soir.

## É3 — Le job aurait eu une seconde mort, sur carte : l'unité NVRTC n'embarquait pas `llvq_slot.cuh` (2026-08-31, trouvé et tué à 0 $)

`nullkbench` assemblait `defines + matvec.cu + nullk.cu` — seul bin du crate
à ne pas préfixer `llvq_slot.cuh`. Or `matvec.cu:11-13` garde son
`#include "llvq_slot.cuh"` derrière `#ifndef LLVQ_SLOT_CUH` : sans le header
prépendu, le garde ne tient pas, NVRTC (sans système de fichiers) évalue
l'include et **refuse la source** — à `Cuda::new`, donc après le probe
`which` et après le début de la facturation. `bin/cuhcheck` ne pouvait pas le
voir : il compile avec `-I` sur le répertoire des kernels, l'include se
résout depuis le disque.

Reproduit à 0 $ **avant** relance, sur l'unité exacte :
`clang -E -x c++ -nostdinc` rend `fatal error: 'llvq_slot.cuh' file not
found` sur l'assemblage actuel, et passe (353 lignes, `tv_nullk` présent) sur
l'assemblage corrigé. Corrigé par `3815eda` (assemblage de `planesbench` :
`llvq_slot.cuh + matvec.cu + nullk.cu` en un seul `load_sources_many`).

## Ce que É2+É3 laissent au dépôt : l'instrument qui manquait

`CUDARC_CUDA_VERSION=12040 cargo check --target x86_64-unknown-linux-gnu
-p llvq-cuda --all-targets` type-checke le crate CUDA **depuis le Mac**
(le mur `nvcc` du build.rs de cudarc tombe avec la variable ; la cible
`rust-std` x86_64-linux suffit, aucun lien). Il aurait vu É2 avant tout
build. Clippy passe au même standard depuis `3815eda`. É3, lui, ne se voit
qu'en exécutant le **texte** du noyau (leçon §5 de CLAUDE.md) — le
`clang -E` ci-dessus est la forme 0 $ de cette exécution.

Aucun de ces trois écarts ne touche les **mesures** de la vague 2 : les jobs
0.1 (8B, 14B) ont tourné avant, leurs chiffres sont au journal, et A1 n'a
encore produit aucun nombre.

## É4 — Quatrième mort d'A1, celle-ci dans le LANCEUR (2026-08-31 soir, ~0,01 $)

Job `6a95e0780718b0f6d890a159`, mort en ~1 s : la relance est passée par le
**CLI** (`hf jobs run … bash -lc '<script>'`) là où le job d'origine avait
été créé par l'**API**. Le parseur du CLI (click) a consommé `-lc` comme
`-l c` — son option `--label` — et bash a reçu le script entier comme un
**nom de fichier** : « bash: set -euo pipefail… : No such file or directory ».
Aucune milliseconde mesurée, aucun octet écrit au bucket.

Relance corrigée dans la minute par l'API (`huggingface_hub.run_job`), avec
un **assert d'identité** : le tableau `['bash', '-lc', …]` est vérifié égal à
l'octet près à celui du job d'origine (`hf jobs inspect 6a954da8…`) avant
l'envoi — job `6a95e11b21c5aa7c8364a122`. Règle qui en sort : **une commande
de job se relance par le canal qui l'a créée, ou par un canal dont on a
vérifié qu'il reproduit le même tableau d'arguments** — un wrapper shell
autour d'un CLI à options courtes ne le garantit pas.

## É5 — A4 : deux jobs au lieu d'un, deux traductions, un bras en plus (2026-08-31 soir, 0,83 $)

Trois déviations d'exécution, aucune de lecture. (1) **Deux jobs** : l'image
publiée fige `CUDA_COMPUTE_CAP=89` et `fusedrun` ne peut pas charger ses
noyaux candle sur sm_80 — le banc a tourné sur l'image standard (NVRTC,
précédent F4), `fusedrun` sur une image jumelle sm80 née le soir même
(`publish --compute-cap 80`). (2) **Vocabulaire** : le §A4 écrit « planes14,
planes14_seg, nullk, f16, cublasf16 » ; la sélection réelle est
`slot32,planes14,fp16,cublasf16,nullk` — `fp16` est le nom d'arme, la
section seg n'est pas une arme nommable ET exige slot32+planes14 dans la
sélection (`planesbench.rs:2740`), piège trouvé à la lecture AVANT le
lancement. (3) **Un bras en plus** : `nullkbench` a tourné dans le job banc
— le r de A1 sur la seconde architecture, non promis par le préreg, 0 $ de
plus. Résultat : r = 0,8198 contre 0,8158 sur L40S. Coût total 0,83 $, sous
le devis (0,9-1,0 $). Journal : `docs/mesures/a4-a100-2026-08-31.txt`.
