#!/usr/bin/env bash
# Decode-speed research phase (CERVEAU O11), step R6: the two row bodies of the third rowbench run INSIDE the token.
# On one L40S, the sealed 4B, 4E v9 against v9ep (every Tetra row phase on rowbench's `endpass`: the served row, its row
# end moved to a second pass; owes the bits) and v9fl (every Tetra row phase on rowbench's `flat`: a thread a run of 8
# blocks on aligned 128-bit loads; owes a bound of 1e-5 on rows, not the bits), through E1's mechanism
# (pers.rs::p4e_e1_edits, kernels/pers_4e_rc.cuh; the served v9 module does not move). Each call is its own process;
# each step writes its own files and is non-fatal for the next:
#   (0) the served v9 config at the door, `LLVQ_CONFIG=<v9> LLVQ_CONTEXTS=5 fusedrun <sealed> 64`, under the 100 ms
#       nvidia-smi sampler: control 2 (the served v9 module) and the served short-context peak;
#   (a) `LLVQ_PERS_4E=256:v9,v9ep,v9fl,v9 fusedrun <sealed> 256` (the served flags on the command line, as E1): the
#       three modules in one process, at a 5-token prompt and at 8,192; 2 + 5 rounds, each round v9, v9ep, v9fl, v9
#       AGAIN (rule F1's control, the same runtime), then the eager path; medians and round-by-round ratios, F1 a
#       context, a SCORE line an arm (F2); 256 tokens an arm against the eager path's (v9 and v9ep owe them, v9fl's are
#       counted); one stamped launch an arm (phase walls); one logits launch an arm (256 positions x 151,936 binary16
#       logits against v9's: bit for bit for v9ep, for v9fl the values that differ and the largest |dlogit| on the
#       positions whose input tokens are v9's);
#   (b) `LLVQ_PERS_4E=256:<arm>` one process an arm (v9, v9fl), under the sampler: the peak VRAM of each, and its
#       logits fingerprints for a second, cross-process comparison.
# Control 3: the E1 reference module (v9 and the logits symbols, d55a097b) and the two arms' modules (v9ep f4166ccf,
# v9fl 9925b4ad) at the card's tile define, as pers.rs builds them on the Mac (dump_tree_modules).
# l40sx1 only: no twin. Nothing is added to the image: the bodies are text in the binary (include_str!), fusedrun is
# already in both lists of ops/Dockerfile.cuda.
# Preregistered: proofs/preregistration-recherche-debit-jeton-2026-11-13.md, to stamp before this.
#
#   DRY_RUN=1 bash ops/jobs/recherche-debit-jeton.sh               parses the job script, launches nothing
#   IMAGE_SHA=<space sha> bash ops/jobs/recherche-debit-jeton.sh    launches, after the operator's go
#
# The image must carry the commit with the arms: a Space rebuild first (ops/run.py publish, no card).
# Cost, estimated (l40sx1 $1.80/h) from the two E1-mechanism jobs (docs/data/jobs.csv: large-4e-e1 6af46f1d, 19 min
# billed for 7 module loads in 5 processes, $0.57; large-attn-ft-4e 6af4f42b, 16 min for 5 loads in 4 processes,
# $0.48): here 6 module loads in 4 processes, about 100 s a load (ft's three-arm process: 351 s for its loads); the
# decode is 4 times ft's tokens on half its contexts with two more 4E timings a round, about 4 min in (a) and 2 min
# in each of (b). About 20 min, $0.60. Timeout 30m: $0.90 at worst, the cap.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-recherche-debit-jeton-2026-11-13.md
OUT_NAME=recherche-debit-jeton-2026-11-13
IMAGE_SHA=${IMAGE_SHA:-}
BUCKET=Pier-Jean/jobs-artifacts
O=/out/$OUT_NAME${RETRY:+-r$RETRY}
DRY=${DRY_RUN:-0}
# the served v9 module (pers.rs::the_4e_v9_module_is_the_one_the_doors_gate_on)
P4E_V9_SHA=1b422b62328be6752df4add43280565fd2a49cfa20ee26953b301abe7a15c530
# the E1 reference module and the two arms', at the card's tile define (pers.rs dump_tree_modules,
# the_rc_modules_keep_their_bytes)
E1_V9_SHA=d55a097b146e8b962829045dfea6975470b408b942c1e0c22c038ad4d9d00860
V9EP_SHA=f4166ccf9d857b30cae8f7691e9e4f83c2726218cb48c29dc512cb42b1a668dd
V9FL_SHA=9925b4ad769c791d2372d52a14b68e650a549ce15edd05eecae0dececfa4f03d
TIMEOUT=${TIMEOUT:-30m}
NTOK=256
FLAVOR=${FLAVOR:-l40sx1}
case "$FLAVOR" in
  l40sx1) FLAVOR_ARGS=() ;;
  *) echo "refused: FLAVOR=$FLAVOR (this job runs on l40sx1 only, no twin)" >&2; exit 1 ;;
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

PRE=$(printf 'O=%q\nSEALED=%q\nSEALED_SHA=%q\nCFG_SHA=%q\nCFG_B64=%q\nP4E_V9_SHA=%q\nE1_V9_SHA=%q\nV9EP_SHA=%q\nV9FL_SHA=%q\nNTOK=%q\nexport HF_HOME=/tmp/hf' \
  "$O" "$SEALED" "$SEALED_SHA" "$CFG_SHA" "$CFG_B64" "$P4E_V9_SHA" "$E1_V9_SHA" "$V9EP_SHA" "$V9FL_SHA" "$NTOK")

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
FLAGS=(LLVQ_FUSED_LAYOUT=tetra48 LLVQ_EMBED=q4 LLVQ_ROT_SHARE=1 LLVQ_FUSE=0)
set +e
echo "== (0) the served v9 config at the door, 5-token prompt =="
date -u
vram served-v9 env LLVQ_CONFIG="$PT" LLVQ_CONTEXTS=5 timeout 5m fusedrun "$S" 64 > "$O/served-v9.txt" 2>&1
echo "fusedrun served v9 exit $?" >> "$O/served-v9.txt"
echo "== (a) v9, v9ep, v9fl, v9 again, interleaved, at 5 and 8,192 =="
date -u
vram jeton-all env "${FLAGS[@]}" LLVQ_PERS_4E="$NTOK:v9,v9ep,v9fl,v9" \
  timeout 14m fusedrun "$S" "$NTOK" > "$O/jeton-all.txt" 2>&1
echo "fusedrun jeton all exit $?" >> "$O/jeton-all.txt"
for arm in v9 v9fl; do
  echo "== (b) $arm alone, under the sampler =="
  date -u
  vram "jeton-$arm" env "${FLAGS[@]}" LLVQ_PERS_4E="$NTOK:$arm" timeout 5m fusedrun "$S" "$NTOK" > "$O/jeton-$arm.txt" 2>&1
  echo "fusedrun jeton $arm exit $?" >> "$O/jeton-$arm.txt"
done
date -u
set -e
peak_gb() { awk -v t="$1" '$1 == "vram" && $2 == t { printf "%.3f", $7 * 1048576 / 1e9 }' "$O/vram.txt"; }
{
  echo "peak VRAM by nvidia-smi (MiB x 1,048,576 / 1e9 GB):"
  for t in served-v9 jeton-all jeton-v9 jeton-v9fl; do echo "  $t $(peak_gb "$t") GB"; done
  echo "the served door (v9 config, 5-token prompt):"
  grep -E '^ +5 +[0-9.]+ \[|4E tokens equal|exit ' "$O/served-v9.txt" | sed 's/^/  /'
  for f in jeton-all jeton-v9 jeton-v9fl; do
    echo "== $f =="
    grep -aE '^4E: persistent module|^4E: pers_4e_(tokens|stamps|logits)_|^4E e1|^4E: RoPE|cudarc event|loaded and checked|^  context|tokens equal|owes a bound|DIVERGES|not run|served door|tok/s|round by round|^  F1 |^  SCORE |^  logits|stamped token|exit |^Error|^error' "$O/$f.txt" || true
  done
  echo "== phase stamps (jeton-all) =="
  awk '/phase stamps, 4E/ {p = 1} /stamped token/ {print; p = 0} p' "$O/jeton-all.txt" || true
} > "$O/summary.txt" 2>&1
cat "$O/summary.txt"
grep -aE '^  F1 |^  SCORE ' "$O/jeton-all.txt" > "$O/rule.txt" || true
grep -qF "4E: persistent module 252757 bytes, sha256 $P4E_V9_SHA, row form 7," "$O/served-v9.txt" \
  && echo "control 2: the served door ran the served 4E v9 module" | tee -a "$O/summary.txt" \
  || echo "VOID: control 2, the served door's 4E module is not the served v9 module" | tee -a "$O/summary.txt"
# control 3: the three modules are the ones pers.rs builds on the Mac
for sha in "$E1_V9_SHA" "$V9EP_SHA" "$V9FL_SHA"; do
  grep -aq "sha256 $sha" "$O/jeton-all.txt" \
    && echo "control 3: module sha256 $sha loaded" | tee -a "$O/summary.txt" \
    || echo "VOID: control 3, module sha256 $sha not loaded" | tee -a "$O/summary.txt"
done
# control 4: the F1 control ran at both contexts (two F1 lines, two SCORE lines an arm)
[ "$(grep -c '^  F1 ' "$O/rule.txt")" = 2 ] && [ "$(grep -c '^  SCORE ' "$O/rule.txt")" = 4 ] \
  && echo "control 4: F1 and the four SCORE lines printed" | tee -a "$O/summary.txt" \
  || echo "VOID: control 4, the rule's lines are missing (see jeton-all.txt)" | tee -a "$O/summary.txt"
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
  --name "recherche-debit-jeton${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
