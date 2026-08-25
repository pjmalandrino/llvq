#!/usr/bin/env bash
# A/B du partage des 48 bits entre direction et gain — ÉTAGE 1, gate 0.6B.
# Pré-enregistrement : proofs/preregistration-bits-de-gain-2026-08-25.md
#                      sha256 428f17d4273173996a1be65ebe27bab747dd27700a28bc96ea46a03efbeb486d
#
# Trois bras, une seule variable : le 7e positionnel (le codebook).
# Tout le reste est celui de m3-queue.sh du 2026-08-07, à une exception
# DÉCLARÉE au §3 du préreg : le binaire est reconstruit avec `fast-linalg`.
#
# ⚠️ Ce script NE CHAÎNE AUCUN RUN 4B. Le script hérité le faisait ; l'étage 2
#    exige son propre go de l'opérateur.
set -u
cd /Users/pjmalandrino/Documents/Pro/workspace/poc/llvq
LOG=/Users/pjmalandrino/llvq-nuit-b
SMOKE=target/release/smoke

note() { echo "$*"; echo "$*" >> "$LOG/journal.txt"; }

step() {                       # step <nom> <cmd...>
  local name="$1"; shift
  note ""
  note "=== $name — $(date '+%Y-%m-%d %H:%M:%S')"
  local t0=$SECONDS
  "$@" > "$LOG/$name.log" 2>&1
  local rc=$?
  local dt=$(( (SECONDS - t0 + 30) / 60 ))
  if [ $rc -eq 0 ]; then note "    OK en $dt min"; else note "    ECHEC (code $rc) apres $dt min"; fi
  return $rc
}

note ""
note "##### A/B partage des 48 bits — etage 1 (gate 0.6B) — depart $(date '+%Y-%m-%d %H:%M:%S') #####"
note "  preregistrement sha256 428f17d4...b486d, tamponne avant ce depart"

# Ordre du §3 du pré-enregistrement, tel qu'écrit.
step gain-ab-0c13 "$SMOKE" 64 2048 12 2048 metal nogs leech0c13 999 rot
step gain-ab-1c12 "$SMOKE" 64 2048 12 2048 metal nogs leech1c12 999 rot
step gain-ab-2c11 "$SMOKE" 64 2048 12 2048 metal nogs leech2c11 999 rot

note ""
note "##### les trois bras sont rendus — aucun verdict n'est calcule ici #####"
note "  les quatre controles du §4 se lisent a la main avant toute perplexite."
