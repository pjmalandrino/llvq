#!/usr/bin/env bash
# M1 — stabilité de l'estimateur de Hessienne : shrinkage hors-diagonale.
# Pré-enregistrement : proofs/preregistration-m1-hessienne-shrink-2026-09-02.md
#
# Douze runs, une seule variable par bras (LLVQ_H_SHRINK = ρ), la graine étant
# un facteur de réplication. Tout le reste est le protocole du gate design C du
# 2026-08-25 (gain-ab-queue.sh) : 0,6B, 28 blocs, wikitext-2, Metal, f32,
# leech1c12, rotation on, nogs. Le binaire DOIT être reconstruit avec
# `--features metal,fast-linalg` au commit qui porte le bouton.
#
# ⚠️ Ce script NE CHAÎNE AUCUN run 4B, et il REFUSE de démarrer sans le tampon
#    du pré-enregistrement — la règle de `rankbench`, portée au shell : une
#    règle qui ne vit que dans la prose se saute le soir où on veut un chiffre.
set -u
cd /Users/pjmalandrino/Documents/Pro/workspace/poc/llvq
PREREG=proofs/preregistration-m1-hessienne-shrink-2026-09-02.md
if [ ! -f "$PREREG.ots" ]; then
  echo "refus : $PREREG.ots absent — tamponner le pré-enregistrement AVANT le premier run" >&2
  exit 2
fi
LOG=/Users/pjmalandrino/llvq-nuit-b
mkdir -p "$LOG"
SMOKE=target/release/smoke
if ! "$SMOKE" --help >/dev/null 2>&1 && [ ! -x "$SMOKE" ]; then
  echo "refus : $SMOKE introuvable — cargo build --release -p llvq-llm --features metal,fast-linalg --bin smoke" >&2
  exit 2
fi

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
note "##### M1 shrinkage de H — 12 runs 0.6B/28 blocs — depart $(date '+%Y-%m-%d %H:%M:%S') #####"
note "  preregistrement $(shasum -a 256 "$PREREG" | cut -c1-16)..., tampon $PREREG.ots present"
note "  binaire $(shasum -a 256 "$SMOKE" | cut -c1-16)..., commit $(git rev-parse --short HEAD)"

# §2 du pré-enregistrement : (ρ = 1, s = 1) d'abord — c'est le contrôle du §3,
# qui doit rejouer 38,4507 — puis ρ-majeur, tel qu'écrit.
for rho in 1 0.9 0.7 0.5; do
  for s in 1 2 3; do
    LLVQ_CALIB_SEED=$s LLVQ_H_SHRINK=$rho \
      step "m1-r${rho}-s${s}" "$SMOKE" 64 2048 12 2048 metal nogs leech1c12 999 rot
  done
done

note ""
note "##### M1 rendu — aucun verdict n'est calcule ici #####"
note "  les quatre controles du §3 se lisent a la main avant toute etendue."
