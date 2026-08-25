#!/usr/bin/env bash
# JOB B du pré-enregistrement proofs/preregistration-bits-de-gain-suite-2026-08-25.md
#   sha256 99d30d6f71431121bef691ef266d7e900026b1ecc85fb2fac9fc795065cf3085
# Les QUATRE bras à LLVQ_CALIB_SEED=1 — seconde graine de calibration.
# Une seule chose change par rapport à l'étage 1 : la graine.
# ⚠️ Ne chaîne AUCUN run 4B.
set -u
cd /Users/pjmalandrino/Documents/Pro/workspace/poc/llvq
LOG=/Users/pjmalandrino/llvq-nuit-b
SMOKE=target/release/smoke
note() { echo "$*"; echo "$*" >> "$LOG/journal.txt"; }
step() {
  local name="$1"; shift
  note ""; note "=== $name — $(date '+%Y-%m-%d %H:%M:%S')"
  local t0=$SECONDS
  "$@" > "$LOG/$name.log" 2>&1
  local rc=$?
  local dt=$(( (SECONDS - t0 + 30) / 60 ))
  if [ $rc -eq 0 ]; then note "    OK en $dt min"; else note "    ECHEC (code $rc) apres $dt min"; fi
}
note ""
note "##### JOB B — quatre bras a LLVQ_CALIB_SEED=1 — depart $(date '+%Y-%m-%d %H:%M:%S') #####"
note "  preregistrement sha256 99d30d6f...3085, tamponne avant ce depart"
export LLVQ_CALIB_SEED=1
step gain-ab-s1-0c13 "$SMOKE" 64 2048 12 2048 metal nogs leech0c13 999 rot
step gain-ab-s1-1c12 "$SMOKE" 64 2048 12 2048 metal nogs leech1c12 999 rot
step gain-ab-s1-2c11 "$SMOKE" 64 2048 12 2048 metal nogs leech2c11 999 rot
step gain-ab-s1-4c10 "$SMOKE" 64 2048 12 2048 metal nogs leech4c10 999 rot
note ""
note "##### JOB B rendu — aucun verdict n'est calcule ici #####"
