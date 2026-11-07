#!/usr/bin/env bash
# The T8 reading of the MLX door: WikiText-2 perplexity of the sealed Qwen3-4B through MLX against M5's 12.580769.
#
#   ops/mlx-t8.sh            # gates, then decode and prefill over 12 windows of 4,096; 15 to 25 min estimated
#
# Prereg: proofs/preregistration-mlx-t8-2026-11-07.md. The script refuses to start before that file is stamped
# (ots stamp), and with any LLVQ_* variable set. The Mac's GPU is busy for the whole run: start it on the operator's
# go, on a quiet machine. Every step's output and the environment go to $OUT (default ~/llvq-mlx-t8-<date>).
# A failed gate stops the run before any perplexity is read.
set -euo pipefail
cd "$(dirname "$0")/.."
PREREG=proofs/preregistration-mlx-t8-2026-11-07.md
CONFIG=configs/mlx/qwen3-4b-tetra-e4-mlx.json
PACKED="${PACKED:-$HOME/q4b-hf-2026-09-28}"
OUT="${OUT:-$HOME/llvq-mlx-t8-$(date +%Y%m%d-%H%M)}"
[ -f "$PREREG.ots" ] || { echo "refused: $PREREG is not stamped; stamp it first (ots stamp $PREREG)"; exit 1; }
if env | grep -q '^LLVQ_'; then echo "refused: LLVQ_* set: $(env | grep '^LLVQ_' | cut -d= -f1 | tr '\n' ' ')"; exit 1; fi
[ -f "$PACKED/model.safetensors" ] || { echo "no packed model at $PACKED (set PACKED=)"; exit 1; }
mkdir -p "$OUT"
{
  echo "start $(date)"
  echo "commit $(git rev-parse HEAD), $(git status --short | wc -l | tr -d ' ') paths changed"
  shasum -a 256 "$PREREG" "$PREREG.ots" "$CONFIG" ops/mlx_tetra/{serve,served,llvq_mlx,ppl,check,test_served}.py ops/mlx-t8.sh
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
    tail -5 "$OUT/$name.txt"; echo "GATE FAILED at $name; no perplexity read. Raw: $OUT"; exit 1
  fi
  tail -2 "$OUT/$name.txt"
}
step 1-served uv run ops/mlx_tetra/test_served.py
step 2-check-mini uv run ops/mlx_tetra/check.py llvq-tetra/tests/fixtures/mini
step 3-hf-check uv run ops/llvq_hf_check.py "$PACKED"
step 4-check-4b uv run ops/mlx_tetra/check.py "$PACKED" 0,12
step 5-span uv run ops/mlx_tetra/serve.py "$CONFIG" "$PACKED" span
echo "=== 6-ppl ($(date +%H:%M:%S))"
nice -n 10 uv run ops/mlx_tetra/serve.py "$CONFIG" "$PACKED" ppl 4096 12 decode,prefill 2>&1 | tee "$OUT/6-ppl.txt"
echo "end $(date)" | tee -a "$OUT/env.txt"
