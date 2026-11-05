#!/usr/bin/env bash
# 4E v3 on the card: the Tetra row stream, the mixed q/k/v split and the short-context attention staging, against
# the eager path on the sealed 4B at 5 and 8,192 prompt tokens, with the phase table. One reading, nothing else.
# Preregistered: proofs/preregistration-pers-4e-v3-2026-11-05.md, to stamp before this.
#
#   DRY_RUN=1 bash ops/jobs/pers-4e-v3.sh                 parses the job script, launches nothing
#   IMAGE_SHA=<space sha> bash ops/jobs/pers-4e-v3.sh      launches, after the operator's go
#
# One job, l40sx1, two processes after the oracle: fusedrun LLVQ_PERS_4E=64 on the sealed 4B; fusedrun through the
# served door for its NVRTC digest and threshold line. Cost, estimated (l40sx1 $1.80/h): about 9 min, $0.27 (v2's
# fusedrun part and served door, without its planesbench). Timeout 30m: $0.90 at worst.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-pers-4e-v3-2026-11-05.md
IMAGE_SHA=${IMAGE_SHA:-}
BUCKET=Pier-Jean/jobs-artifacts
O=/out/pers-4e-v3-2026-11-05${RETRY:+-r$RETRY}
TIMEOUT=30m
DRY=${DRY_RUN:-0}

SEALED="sealed-4b-2026-09-23/qwen3-4b-sealed.bin:1418224685"
SEALED_SHA8=886391a8
CONF_LOCAL=configs/qwen3-4b-tetra-e4-kernel-from1280.json
CONF=$(cat "$CONF_LOCAL")
CONF_SHA=$(printf '%s\n' "$CONF" | shasum -a 256 | cut -d' ' -f1)
[ "$CONF_SHA" = "$(shasum -a 256 < "$CONF_LOCAL" | cut -d' ' -f1)" ] \
  || { echo "refused: $CONF_LOCAL does not end in exactly one newline" >&2; exit 1; }
# The served unit's digest (llvq-cuda/src/pers.rs::the_served_unit_is_the_published_one): 4E v2 touches only the
# persistent module, never the served unit.
SERVED_SHA=6193094310bc18acdb0a0c554807ff45f6df09c1f6ab49e0577032874f107812

if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  for row in "$SEALED"; do
    path=${row%%:*}; bytes=${row##*:}
    DIR=${path%/*}; NAME=${path##*/}
    REMOTE=$(hf buckets ls "hf://buckets/$BUCKET/$DIR/" 2>/dev/null \
      | awk -v n="/$NAME" 'substr($NF, length($NF) - length(n) + 1) == n {print $1}' || true)
    [ "$REMOTE" = "$bytes" ] || { echo "refused: $path is ${REMOTE:-absent} B in the bucket, expected $bytes" >&2; exit 1; }
  done
  if hf buckets ls "hf://buckets/$BUCKET/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already exists" >&2; exit 1
  fi
  IMAGE_SHA=${IMAGE_SHA:?IMAGE_SHA=<the Space sha of the rebuild that carries this branch>}
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    'from huggingface_hub import HfApi; print(HfApi().space_info("Pier-Jean/llvq-runner-cuda").sha)')
  [ "$NOW" = "$IMAGE_SHA" ] || { echo "refused: the Space is at $NOW, expected $IMAGE_SHA" >&2; exit 1; }
  echo "the sealed 4B verified by size; config $CONF_SHA; image $NOW"
fi

PRE=$(printf 'O=%q\nS=%q\nSB=%q\nSSHA8=%q\nCK=%q\nCONF=%q\nCONF_SHA=%q\nSERVED_SHA=%q' \
  "$O" "/out/${SEALED%%:*}" "${SEALED##*:}" "$SEALED_SHA8" "$O/qwen3-4b-tetra-e4-kernel-from1280.json" "$CONF" \
  "$CONF_SHA" "$SERVED_SHA")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,clocks.max.sm --format=csv,noheader | tee "$O/gpu.txt"
test "$(stat -c %s "$S")" = "$SB"
sha256sum "$S" | tee "$O/files.sha256"
test "$(sha256sum "$S" | cut -c1-8)" = "$SSHA8"
printf '%s\n' "$CONF" > "$CK"
test "$(sha256sum "$CK" | cut -d' ' -f1)" = "$CONF_SHA"

echo '== gate 1: oracle =='
oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle.txt" | tail -3
grep -q 'MATCH' "$O/oracle.txt"

echo '== process 1: 4E v3 against the eager path, 5 and 8,192 prompt tokens =='
date -u
rc=0
LLVQ_FUSED_LAYOUT=tetra48 LLVQ_EMBED=q4 LLVQ_ROT_SHARE=1 LLVQ_FUSE=0 LLVQ_PERS_4E=64 \
  fusedrun "$S" 64 > "$O/pers-4e.txt" 2>&1 || rc=$?
date -u
echo "fusedrun exit $rc"
grep -aE '^4E|cudarc event|context |tokens equal|served door|tok/s|4E / eager|phase stamps|stamped token|^ +(q/k/v|norm|attention|o |MLP|head|token|embedding)' "$O/pers-4e.txt" | tail -80
[ "$rc" = 0 ]
# gates 3 and 4: each line from where it is printed
grep -q 'cudarc event tracking: off (one stream, ordered by issue)' "$O/pers-4e.txt"
grep -q '^4E: RoPE forms equal to candle' "$O/pers-4e.txt"
test "$(grep -c '4E at 512 threads: tokens equal the eager path' "$O/pers-4e.txt")" = 2
grep -q 'the served config: every decode row at or past position 1280, inside the T8' "$O/pers-4e.txt"

echo '== process 2: the served door, for its digest =='
LLVQ_CONFIG="$CK" fusedrun "$S" 16 > "$O/served-16.txt" 2>&1
tail -5 "$O/served-16.txt"
grep -q 'attention kernel from position 1280 on grow(256)' "$O/served-16.txt"

D_SERVED=$(grep -o 'NVRTC source: [0-9]* bytes, sha256 [0-9a-f]*' "$O/served-16.txt" | head -1)
echo "served door: $D_SERVED" | tee "$O/digests.txt"
case "$D_SERVED" in *"$SERVED_SHA"*) ;; *) echo "gate 5: the served unit moved from $SERVED_SHA" >&2; exit 1;; esac
ls -l "$O"
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
  --name "pers-4e-v3${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
