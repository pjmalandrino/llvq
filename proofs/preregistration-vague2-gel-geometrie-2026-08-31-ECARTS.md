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
