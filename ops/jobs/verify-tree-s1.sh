#!/usr/bin/env bash
# Stage S1 of the tree verify (docs/plan-verify-tree.md, prereg proofs/preregistration-verify-tree-s1-2026-11-09.md):
# on one L40S, the sealed 4B under its served v9 config, one `vtreebench <sealed>` process:
#   the 4E v9 install and the tree pass beside it (pers_4et.cu, its own module), its registers and local bytes;
#   at the chat and rag contexts: vt1 against 4E's greedy tokens on 64 positions; identity of every row across
#   R = 1, 2, 4, 8, 16, 32 and two shuffled slot orders; c_R = t(vt R) / t(4e), 2 discarded and 5 timed rounds,
#   arms interleaved; the phases of vt1 and vt16.
# Stamped before this: the prereg above and its .ots.
#
#   DRY_RUN=1 bash ops/jobs/verify-tree-s1.sh             parses the job script, launches nothing
#   IMAGE_SHA=<space sha> bash ops/jobs/verify-tree-s1.sh  launches
#
# Cost, estimated (l40sx1 $1.80/h, prereg "Cost"): about 10 min, $0.30; timeout 20m, $0.60, the cap in force.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-verify-tree-s1-2026-11-09.md
IMAGE_SHA=${IMAGE_SHA:-}
BUCKET=Pier-Jean/jobs-artifacts
O=/out/verify-tree-s1-2026-11-09${RETRY:+-r$RETRY}
DRY=${DRY_RUN:-0}
P4E_V9_SHA=1b422b62328be6752df4add43280565fd2a49cfa20ee26953b301abe7a15c530
TIMEOUT=${TIMEOUT:-20m}
SEALED="sealed-4b-2026-09-23/qwen3-4b-sealed.bin:1418224685"
SEALED_SHA=886391a8c03f66dc269cc65c3598c6627dbdcd259180aff36604ef10d37371b8
C=configs/qwen3-4b-tetra-e4-4e-v9.json
grep -q '^  "rows": "v9",$' "$C" || { echo "refused: $C does not name rows v9" >&2; exit 1; }
[ "$(printf '%s\n' "$(cat "$C")" | shasum -a 256 | cut -d' ' -f1)" = "$(shasum -a 256 < "$C" | cut -d' ' -f1)" ] \
  || { echo "refused: $C does not end in exactly one newline" >&2; exit 1; }
CFG_SHA=$(shasum -a 256 < "$C" | cut -d' ' -f1)
CFG_B64=$(base64 < "$C" | tr -d '\n')

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

PRE=$(printf 'O=%q\nSEALED=%q\nSEALED_SHA=%q\nCFG_SHA=%q\nCFG_B64=%q\nP4E_V9_SHA=%q\nexport HF_HOME=/tmp/hf' \
  "$O" "$SEALED" "$SEALED_SHA" "$CFG_SHA" "$CFG_B64" "$P4E_V9_SHA")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,clocks.max.sm --format=csv,noheader | tee "$O/gpu.txt"
echo '== control 1: oracle =='
timeout 5m oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle.txt" | tail -2
grep -q MATCH "$O/oracle.txt"
echo '== control 1: the object and its config =='
S=/out/${SEALED%%:*}; SB=${SEALED##*:}
test "$(stat -c %s "$S")" = "$SB"
sha256sum "$S" | tee "$O/files.sha256"
test "$(sha256sum "$S" | cut -d' ' -f1)" = "$SEALED_SHA"
PT=$O/qwen3-4b-tetra-e4-4e-v9.json
printf '%s' "$CFG_B64" | base64 -d > "$PT"
test "$(sha256sum "$PT" | cut -d' ' -f1)" = "$CFG_SHA"
echo '== vtreebench =='
date -u
set +e
LLVQ_CONFIG="$PT" timeout 14m vtreebench "$S" > "$O/vtree-4b.txt" 2>&1
rc=$?
set -e
date -u
echo "vtreebench exit $rc" | tee "$O/summary.txt"
grep -E '^4E: persistent module|^4E tree|^== tree|^  (vt1|identity|4E, one|timed passes|an extra row|rows| +[0-9]+ +[0-9.]+ \[)|^    (rows| +[0-9]+ )|^c_16' "$O/vtree-4b.txt" | tee -a "$O/summary.txt"
tail -3 "$O/vtree-4b.txt"
# control 2: 4E is the served v9 module (its one-token launches are checked against its greedy tokens inside)
grep -qF "4E: persistent module 252757 bytes, sha256 $P4E_V9_SHA, row form 7," "$O/vtree-4b.txt" \
  && echo "control 2: the served 4E v9 module" | tee -a "$O/summary.txt" \
  || echo "VOID: control 2, the 4E module is not the served v9 module" | tee -a "$O/summary.txt"
grep -E '^4E tree: pers_4et_rows_f[0-9]_t512: [0-9]+ registers, [0-9]+ B local' "$O/vtree-4b.txt" | tee -a "$O/summary.txt" || true
ls -la "$O"
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
  --name "verify-tree-s1${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
