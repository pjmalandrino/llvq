#!/bin/bash
# Full attention capture of Inkling 2 Mini (item 1a) in resumable slices, one heavy job a slice.
#
#   ops/inkling/run_attn_capture.sh [OUT] [N] [LEN] [PER]
#   defaults: OUT=~/llvq-inkling-mini/attn/full_64x2048  N=64  LEN=2048  PER=8
#
# Each slice of PER sequences runs through ~/.llvq-guard/heavy.sh: capture, partial written under a temporary name
# then renamed, folded into the accumulator in the same job. Between slices the script sleeps 45 s so the encode can
# take the lock back. A slice already folded or with a complete partial is skipped, so the script can be rerun after
# any kill. After the last slice, `merge` refuses any gap and writes OUT/L{xx}_{stream}.f64, OUT/manifest.json and
# the inkwh manifests OUT/wh/*.json. Slice size: 8 x 2,048 = 16,384 tokens, 356 s at the 46 tok/s MPS probe, under 8
# minutes with the load and the fold. Disk: 4.87 GB at the end, about 9.8 GB at the peak of a fold.
set -euo pipefail
OUT="${1:-$HOME/llvq-inkling-mini/attn/full_64x2048}"
N="${2:-64}"; LEN="${3:-2048}"; PER="${4:-8}"
cd "$(dirname "$0")"
export INKLING_THREADS="${INKLING_THREADS:-4}"
for ((a = 0; a < N; a += PER)); do
  b=$((a + PER)); [ "$b" -gt "$N" ] && b=$N
  if INKLING_THREADS=2 uv run -q attn_slices.py done --out "$OUT" --seqs "$a..$b"; then
    echo "$(date '+%T') slice $a..$b already done"; continue
  fi
  echo "$(date '+%T') slice $a..$b"
  ~/.llvq-guard/heavy.sh "lot1-A: attention capture slice $a..$b of $N x $LEN" \
    uv run -q attn_slices.py slice --out "$OUT" --n "$N" --len "$LEN" --seqs "$a..$b" --fold
  sleep 45
done
~/.llvq-guard/heavy.sh "lot1-A: attention capture merge ($N x $LEN)" uv run -q attn_slices.py merge --out "$OUT"
echo "$(date '+%T') done: $OUT"
