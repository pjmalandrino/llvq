#!/usr/bin/env bash
# plan-300 step 3 (docs/plan-300.md §6b): 4E v9 (v8 with the int4 rows on tiled words, K11, row form 7) against v8
# and v6, and v6s (v6's module asking for v8's shared bytes: the same code and bits under v8's L1/shared carve-out,
# the diagnostic of v8's slower rows), at 5 and 8,192 prompt tokens (fusedrun LLVQ_PERS_4E=64), each owing the eager
# tokens, rates and ratios round by round, one stamped launch each.
# Preregistered: proofs/preregistration-step3-4e-2026-11-07.md, to stamp before this.
#
#   DRY_RUN=1 bash ops/jobs/step3-4e.sh                 parses the job script, launches nothing
#   IMAGE_SHA=<space sha> bash ops/jobs/step3-4e.sh      launches, after the operator's go
#
# Every LLVQ_* variable is set on its own command line. The image must carry this branch (a Space rebuild first,
# ops/run.py publish, no card). Cost, estimated (l40sx1 $1.80/h, prereg section 8): about 12 min, $0.36. Timeout 25m:
# $0.75 at worst.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-step3-4e-2026-11-07.md
IMAGE_SHA=${IMAGE_SHA:-}
BUCKET=Pier-Jean/jobs-artifacts
O=/out/step3-4e-2026-11-07${RETRY:+-r$RETRY}
TIMEOUT=25m
DRY=${DRY_RUN:-0}

SEALED="sealed-4b-2026-09-23/qwen3-4b-sealed.bin:1418224685"
SEALED_SHA8=886391a8

if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  path=${SEALED%%:*}; bytes=${SEALED##*:}
  REMOTE=$(hf buckets ls "hf://buckets/$BUCKET/${path%/*}/" 2>/dev/null \
    | awk -v n="/${path##*/}" 'substr($NF, length($NF) - length(n) + 1) == n {print $1}' || true)
  [ "$REMOTE" = "$bytes" ] || { echo "refused: $path is ${REMOTE:-absent} B in the bucket, expected $bytes" >&2; exit 1; }
  if hf buckets ls "hf://buckets/$BUCKET/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already exists" >&2; exit 1
  fi
  IMAGE_SHA=${IMAGE_SHA:?IMAGE_SHA=<the Space sha of the rebuild that carries this branch>}
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    'from huggingface_hub import HfApi; print(HfApi().space_info("Pier-Jean/llvq-runner-cuda").sha)')
  [ "$NOW" = "$IMAGE_SHA" ] || { echo "refused: the Space is at $NOW, expected $IMAGE_SHA" >&2; exit 1; }
  echo "bucket copy verified by size; image $NOW"
fi

PRE=$(printf 'O=%q\nS=%q\nSB=%q\nSSHA8=%q' "$O" "/out/${SEALED%%:*}" "${SEALED##*:}" "$SEALED_SHA8")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,clocks.max.sm --format=csv,noheader | tee "$O/gpu.txt"
test "$(stat -c %s "$S")" = "$SB"
sha256sum "$S" | tee "$O/files.sha256"
test "$(sha256sum "$S" | cut -c1-8)" = "$SSHA8"
echo '== gate 1: oracle =='
timeout 5m oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle.txt" | tail -2
grep -q MATCH "$O/oracle.txt"

echo '== 4E v4, v6, v7, v8, v9 and v6s against the eager path, 5 and 8,192 prompt tokens =='
date -u
rc=0
LLVQ_FUSED_LAYOUT=tetra48 LLVQ_EMBED=q4 LLVQ_ROT_SHARE=1 LLVQ_FUSE=0 LLVQ_PERS_4E=64 \
  timeout 20m fusedrun "$S" 64 > "$O/pers-4e.txt" 2>&1 || rc=$?
date -u
grep -aE '^4E|cudarc event|context |tokens equal|DIVERGES|not run|served door|tok/s|round by round|phase stamps|stamped token|inside q/k/v|^ +(q/k/v|norm|attention|o |MLP|head|token|embedding)' "$O/pers-4e.txt" | tail -200
ls -la "$O"
[ "$rc" = 0 ] || { echo "fusedrun exited $rc (a divergence voids the run)"; exit 1; }
grep -q 'cudarc event tracking: off (one stream, ordered by issue)' "$O/pers-4e.txt"
grep -q '^4E: RoPE forms equal to candle' "$O/pers-4e.txt"
# v4 and v6 are the references and gate the run; v7, v8, v9 and v6s are read (prereg §4)
for f in v4 v6; do
  n=$(grep -c "4E $f at 512 threads: tokens equal the eager path" "$O/pers-4e.txt" || true)
  echo "4E $f identity lines: $n of 2"
  test "$n" = 2
done
for f in v7 v8 v9 v6s; do
  n=$(grep -c "4E $f at 512 threads: tokens equal the eager path" "$O/pers-4e.txt" || true)
  echo "4E $f identity lines: $n of 2 (read, not gated)"
done
JOB

if [ "$DRY" = 1 ]; then
  JOBSCRIPT="set -euo pipefail
$PRE
$BODY"
  printf '%s\n' "$JOBSCRIPT" | bash -n && echo "DRY_RUN: job script parses; nothing launched" >&2
  exit 0
fi

uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor l40sx1 --timeout "$TIMEOUT" \
  --bucket "$BUCKET" --out-mount /out \
  --name "step3-4e${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
