#!/usr/bin/env bash
# Job E1 of the large-4E budget (prereg proofs/preregistration-large-4e-e1-2026-11-11.md,
# docs/mesures/large-4e-budget-2026-11-11.txt cut 1): on one L40S, the sealed 4B, 4E v9 against two arms that move
# only its L2 prefetches (pers.rs::p4e_e1_edits): v9np, no prefetch of the o and gate/up rows in the attention phase;
# v9mv, layer l + 1's q/k/v and o rows asked of L2 during layer l's gate/up rows. Each call is its own process; each
# step writes its own files and is non-fatal for the next:
#   (0) the served v9 config at the door, `LLVQ_CONFIG=<v9> LLVQ_CONTEXTS=5 fusedrun <sealed> 64`, under the 100 ms
#       nvidia-smi sampler: control 2 (the served v9 module) and the served short-context peak;
#   (a) `LLVQ_PERS_4E=64:v9,v9np,v9mv fusedrun <sealed> 64` (the served flags on the command line, as step3-4e): the three E1 modules in one process, at a 5-token prompt
#       and at 8,192, 2 + 5 interleaved rounds (eager and every arm each round), medians and round-by-round ratios,
#       the 64 tokens against the eager path, one stamped launch an arm (phase walls), one logits launch an arm
#       (64 positions x 151,936 binary16 logits against e1:v9's, bit for bit);
#   (b) `LLVQ_PERS_4E=64:<arm> fusedrun <sealed> 64`, one process an arm (v9, v9np, v9mv), each under the sampler: the
#       peak VRAM of each arm, and its logits fingerprints for a second, cross-process comparison.
# l40sx1 only: no twin.
#
#   DRY_RUN=1 bash ops/jobs/large-4e-e1.sh             parses the job script, launches nothing
#   IMAGE_SHA=<space sha> bash ops/jobs/large-4e-e1.sh  launches
#
# Cost, estimated (l40sx1 $1.80/h): about 18 min, $0.54; timeout 29m, $0.87.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-large-4e-e1-2026-11-11.md
OUT_NAME=large-4e-e1-2026-11-11
IMAGE_SHA=${IMAGE_SHA:-}
BUCKET=Pier-Jean/jobs-artifacts
O=/out/$OUT_NAME${RETRY:+-r$RETRY}
DRY=${DRY_RUN:-0}
P4E_V9_SHA=1b422b62328be6752df4add43280565fd2a49cfa20ee26953b301abe7a15c530
TIMEOUT=${TIMEOUT:-29m}
NTOK=64
FLAVOR=${FLAVOR:-l40sx1}
case "$FLAVOR" in
  l40sx1) FLAVOR_ARGS=() ;;
  *) echo "refused: FLAVOR=$FLAVOR (E1 runs on l40sx1 only, no twin)" >&2; exit 1 ;;
esac
SEALED="sealed-4b-2026-09-23/qwen3-4b-sealed.bin:1418224685"
SEALED_SHA=886391a8c03f66dc269cc65c3598c6627dbdcd259180aff36604ef10d37371b8
C=configs/qwen3-4b-tetra-e4-4e-v9.json
grep -q '^  "rows": "v9",$' "$C" || { echo "refused: $C does not name rows v9" >&2; exit 1; }
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

PRE=$(printf 'O=%q\nSEALED=%q\nSEALED_SHA=%q\nCFG_SHA=%q\nCFG_B64=%q\nP4E_V9_SHA=%q\nNTOK=%q\nexport HF_HOME=/tmp/hf' \
  "$O" "$SEALED" "$SEALED_SHA" "$CFG_SHA" "$CFG_B64" "$P4E_V9_SHA" "$NTOK")

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
# VRAM sampler (ops/jobs/duel-iq2.sh): memory.used every 100 ms around one command; idle and peak in MiB.
vram() {
  local tag=$1; shift
  local idle; idle=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 100 > "$O/vram-$tag.csv" &
  local sp=$!
  local rc=0
  "$@" || rc=$?
  kill "$sp"; wait "$sp" 2>/dev/null || true
  local peak; peak=$(sort -n "$O/vram-$tag.csv" | tail -1)
  echo "vram $tag idle $idle MiB peak $peak MiB" | tee -a "$O/vram.txt" >&2
  return $rc
}
set +e
echo "== (0) the served v9 config at the door, 5-token prompt =="
date -u
vram served-v9 env LLVQ_CONFIG="$PT" LLVQ_CONTEXTS=5 timeout 5m fusedrun "$S" "$NTOK" > "$O/served-v9.txt" 2>&1
echo "fusedrun served v9 exit $?" >> "$O/served-v9.txt"
echo "== (a) the three E1 modules, interleaved =="
date -u
vram e1-all env LLVQ_FUSED_LAYOUT=tetra48 LLVQ_EMBED=q4 LLVQ_ROT_SHARE=1 LLVQ_FUSE=0 LLVQ_PERS_4E="$NTOK:v9,v9np,v9mv" \
  timeout 9m fusedrun "$S" "$NTOK" > "$O/e1-all.txt" 2>&1
echo "fusedrun e1 all exit $?" >> "$O/e1-all.txt"
for arm in v9 v9np v9mv; do
  echo "== (b) $arm alone, under the sampler =="
  date -u
  vram "e1-$arm" env LLVQ_FUSED_LAYOUT=tetra48 LLVQ_EMBED=q4 LLVQ_ROT_SHARE=1 LLVQ_FUSE=0 LLVQ_PERS_4E="$NTOK:$arm" \
    timeout 5m fusedrun "$S" "$NTOK" > "$O/e1-$arm.txt" 2>&1
  echo "fusedrun e1 $arm exit $?" >> "$O/e1-$arm.txt"
done
date -u
set -e
peak_gb() { awk -v t="$1" '$1 == "vram" && $2 == t { printf "%.3f", $7 * 1048576 / 1e9 }' "$O/vram.txt"; }
{
  echo "peak VRAM by nvidia-smi (MiB x 1,048,576 / 1e9 GB):"
  for t in served-v9 e1-all e1-v9 e1-v9np e1-v9mv; do echo "  $t $(peak_gb "$t") GB"; done
  echo "the served door (v9 config, 5-token prompt):"
  grep -E '^ +5 +[0-9.]+ \[|4E tokens equal|exit ' "$O/served-v9.txt" | sed 's/^/  /'
  for f in e1-all e1-v9 e1-v9np e1-v9mv; do
    echo "== $f =="
    grep -aE '^4E: persistent module|^4E: pers_4e_(tokens|stamps|logits)_|^4E e1|^4E: RoPE|cudarc event|loaded and checked|^  context|tokens equal|DIVERGES|not run|served door|tok/s|round by round|^  logits|stamped token|exit |^Error|^error' "$O/$f.txt" || true
  done
  echo "== phase stamps (e1-all) =="
  awk '/phase stamps, 4E/ {p = 1} /stamped token/ {print; p = 0} p' "$O/e1-all.txt" || true
} > "$O/summary.txt" 2>&1
cat "$O/summary.txt"
grep -qF "4E: persistent module 252757 bytes, sha256 $P4E_V9_SHA, row form 7," "$O/served-v9.txt" \
  && echo "control 2: the served door ran the served 4E v9 module" | tee -a "$O/summary.txt" \
  || echo "VOID: control 2, the served door's 4E module is not the served v9 module" | tee -a "$O/summary.txt"
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
  --flavor "$FLAVOR" ${FLAVOR_ARGS[@]+"${FLAVOR_ARGS[@]}"} --timeout "$TIMEOUT" \
  --bucket "$BUCKET" --out-mount /out \
  --name "large-4e-e1" \
  "$PRE" "$BODY"
