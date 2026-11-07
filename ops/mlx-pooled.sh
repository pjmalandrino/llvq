#!/usr/bin/env bash
# The MLX door on WikiText-2 windows 13 to 24, against the Rust dense f16 reading of the same windows, under a gate on
# the paired interval. Prereg: proofs/preregistration-mlx-pooled-2026-11-07.md. About 25 minutes of the Mac's GPU.
#
#   ops/mlx-pooled.sh
#
# Refuses before the prereg is stamped, and with any LLVQ_* variable set. Every step's output goes to $OUT (default
# ~/llvq-mlx-pooled-<date>). A failed gate stops the run before the MLX reading.
set -euo pipefail
cd "$(dirname "$0")/.."
PREREG=proofs/preregistration-mlx-pooled-2026-11-07.md
CONFIG=configs/mlx/qwen3-4b-tetra-e4-mlx.json
PACKED="${PACKED:-$HOME/q4b-hf-2026-09-28}"
SEALED="${SEALED:-$HOME/q4b-sealed-2026-09-23/qwen3-4b-sealed.bin}"
PPL="${PPL:-$(git rev-parse --git-common-dir)/../.claude/worktrees/validation-vague1/target-local/release/ppl}"
S1=docs/mesures/s1-f32-metal-2026-11-04-brut/ppl-f16.txt
OUT="${OUT:-$HOME/llvq-mlx-pooled-$(date +%Y%m%d-%H%M)}"
[ -f "$PREREG.ots" ] || { echo "refused: $PREREG is not stamped; stamp it first (ots stamp $PREREG)"; exit 1; }
if env | grep -q '^LLVQ_'; then echo "refused: LLVQ_* set: $(env | grep '^LLVQ_' | cut -d= -f1 | tr '\n' ' ')"; exit 1; fi
[ -f "$PACKED/model.safetensors" ] || { echo "no packed model at $PACKED (set PACKED=)"; exit 1; }
sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
[ "$(sha "$PPL")" = 5f3ff087c2dab878e4b5bf46d6214fb3bb16f7c20c8cfcccb0d2be9ed45a0697 ] || { echo "GATE FAILED: $PPL is not S1's ppl binary"; exit 1; }
[ "$(sha "$SEALED")" = 886391a8c03f66dc269cc65c3598c6627dbdcd259180aff36604ef10d37371b8 ] || { echo "GATE FAILED: $SEALED"; exit 1; }
mkdir -p "$OUT"
{
  echo "start $(date)"
  echo "commit $(git rev-parse HEAD), $(git status --short | wc -l | tr -d ' ') paths changed"
  shasum -a 256 "$PREREG" "$PREREG.ots" "$CONFIG" ops/mlx_tetra/{serve,served,llvq_mlx,ppl,check,test_served}.py ops/mlx-pooled.sh "$PPL" "$SEALED"
  echo "packed $PACKED"; shasum -a 256 "$PACKED/config.json"
  echo "uv $(uv --version)"
  echo "machine $(sysctl -n machdep.cpu.brand_string) | $(sw_vers -productName) $(sw_vers -productVersion)"
  pmset -g therm 2>/dev/null | tr '\n' ' '; echo
  echo "load: other processes $(ps -A -o %cpu= | awk '{s+=$1} END {print s}') % CPU"
} | tee "$OUT/env.txt"
step() {
  local name="$1"; shift
  echo "=== $name: $* ($(date +%H:%M:%S))"
  if ! nice -n 10 "$@" > "$OUT/$name.txt" 2>&1; then
    tail -5 "$OUT/$name.txt"; echo "GATE FAILED at $name; no MLX reading. Raw: $OUT"; exit 1
  fi
  tail -2 "$OUT/$name.txt"
}
step 1-reference env LLVQ_DTYPE=f16 "$PPL" 4096 24 metal "$SEALED"
want=$(grep -oE 'window +[0-9]+/12 +nll [0-9.]+' "$S1" | awk '{print $NF}')
got=$(grep -oE 'window +[0-9]+/24 +nll [0-9.]+' "$OUT/1-reference.txt" | head -12 | awk '{print $NF}')
[ -n "$want" ] && [ "$want" = "$got" ] || { echo "GATE FAILED: reference windows 1-12 differ from S1's"; diff <(echo "$want") <(echo "$got") || true; exit 1; }
echo "gate: reference windows 1-12 equal S1's f16 NLLs to the last digit"
step 2-served uv run ops/mlx_tetra/test_served.py
step 3-check-mini uv run ops/mlx_tetra/check.py llvq-tetra/tests/fixtures/mini
step 4-check-4b uv run ops/mlx_tetra/check.py "$PACKED" 0,12
echo "=== 5-ppl ($(date +%H:%M:%S))"
nice -n 10 uv run ops/mlx_tetra/serve.py "$CONFIG" "$PACKED" ppl 4096 12 decode,prefill 12 "$OUT/1-reference.txt" 2>&1 | tee "$OUT/5-ppl.txt"
echo "end $(date)" | tee -a "$OUT/env.txt"
