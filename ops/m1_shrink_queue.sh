#!/usr/bin/env bash
# M1, stability of the Hessian estimator: off-diagonal shrinkage.
# Preregistration: proofs/preregistration-m1-hessienne-shrink-2026-09-02.md
#
# Twelve runs, a single variable per arm (LLVQ_H_SHRINK = ρ). The seed is a
# replication factor. Everything else is the protocol of the design C gate of
# 2026-08-25 (gain-ab-queue.sh): 0.6B, 28 blocks, wikitext-2, Metal, f32,
# leech1c12, rotation on, nogs. The binary MUST be rebuilt with
# `--features metal,fast-linalg` at the commit that carries the knob.
#
# WARNING: this script CHAINS NO 4B run, and it REFUSES to start without the
#    timestamp of the preregistration. That is the `rankbench` rule carried to
#    the shell: a rule that lives only in prose gets skipped the evening you
#    want a number.
set -u
cd /Users/pjmalandrino/Documents/Pro/workspace/poc/llvq
PREREG=proofs/preregistration-m1-hessienne-shrink-2026-09-02.md
if [ ! -f "$PREREG.ots" ]; then
  echo "refused: $PREREG.ots missing, timestamp the preregistration BEFORE the first run" >&2
  exit 2
fi
LOG=/Users/pjmalandrino/llvq-nuit-b
mkdir -p "$LOG"
SMOKE=target/release/smoke

# WARNING: the Mac is the operator's WORKSTATION, not a compute node. Measured
# on 2026-09-02 during this very queue: a `smoke` with no cap takes ~1,470% of
# CPU, that is ~15 cores out of 16, and makes the machine painful to use for
# five hours. RAM is not the cause (1.22 GB of RSS out of 64).
#
# The two guards below cost ~20% of compute time and make the machine usable.
# They move NO number: the quantization is split by row and exact, and
# `parallel_matches_serial_exactly` requires that to the bit. Only the
# durations in the journal move.
#
# The fix was applied at the 5th measurement out of 12 with a `renice` (see
# ...-ECARTS.md §É1) because it was not set at the start. It is set now.
: "${LLVQ_THREADS:=$(( $(sysctl -n hw.ncpu) - 4 ))}"
export LLVQ_THREADS
renice 10 -p $$ >/dev/null 2>&1   # child arms inherit it
if ! "$SMOKE" --help >/dev/null 2>&1 && [ ! -x "$SMOKE" ]; then
  echo "refused: $SMOKE not found, cargo build --release -p llvq-llm --features metal,fast-linalg --bin smoke" >&2
  exit 2
fi

note() { echo "$*"; echo "$*" >> "$LOG/journal.txt"; }

step() {                       # step <name> <cmd...>
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
note "  LLVQ_THREADS=$LLVQ_THREADS sur $(sysctl -n hw.ncpu) coeurs, nice $(ps -o nice= -p $$ | tr -d ' ')"

# §2 of the preregistration: (ρ = 1, s = 1) first, the control of §3, which
# must replay 38.4507. Then ρ-major, as written.
for rho in 1 0.9 0.7 0.5; do
  for s in 1 2 3; do
    LLVQ_CALIB_SEED=$s LLVQ_H_SHRINK=$rho \
      step "m1-r${rho}-s${s}" "$SMOKE" 64 2048 12 2048 metal nogs leech1c12 999 rot
  done
done

note ""
note "##### M1 rendu — aucun verdict n'est calcule ici #####"
note "  les quatre controles du §3 se lisent a la main avant toute etendue."
