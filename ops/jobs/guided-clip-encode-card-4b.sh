#!/usr/bin/env bash
# The one encode of the clipped guided gate/up retry ON A CARD: the 2026-11-13 candidate with its per-token weights
# clipped at 10 times each row group's mean (LLVQ_GUIDED_CLIP=10, llvq-llm/src/guided.rs). Go given by the operator
# for one retry after the 2026-11-13 loss (docs/mesures/guided-gateup-4b-2026-11-13.txt, "POST HOC DIAGNOSTIC": the
# weights sat on about 2 % of the calibration tokens). Preregistered: proofs/preregistration-guided-clip-gateup-4b-
# 2026-11-14.md, which must be stamped, and must name this script's sha256, before the arm launches.
#
# Derived from ops/jobs/guided-encode-card-4b.sh, left untouched (a stamped prereg cites its sha256). Everything is
# the 2026-11-13 candidate's: the recipe (the 2026-09-18 DCLM base's `tetra1`, rotation 0x110feed, nogs, damping 1e-2,
# h_shrink 1, gain_scale 1, LLVQ_INT4_TYPES=v_proj, dclm-edu 64 x 2048 from the prefix, f32, every block, eval
# wikitext-2 12 x 4096, device cuda, 23 encoder threads), the weights (guided-weights-4b-2026-11-13/ in the bucket,
# manifest sha256 e63dad83...), G = 4, the flavor. The ONLY difference from that candidate is LLVQ_GUIDED_CLIP=10. The
# reference arm is NOT re-encoded: `compare` reads the 2026-11-13 baseline already in the bucket.
#
# The image: the private Space Pier-Jean/llvq-runner-cuda-qualite, republished from this branch so that `smoke` knows
# LLVQ_GUIDED_CLIP. The job refuses before the encode if the `smoke` binary does not name the variable: an old smoke
# would ignore it and encode the 2026-11-13 candidate again. The operator publishes and pins IMAGE_SHA below.
#
# Cost, estimated from the 2026-11-13 card encodes (journal, "billed": 77.5 and 80.2 min, $3.55 for the first): the
# clip adds one pass over each weight vector, nothing measurable. 80 min, about $3.70 at $2.75/h. Timeout 105m:
# ceiling $4.81. A cut leaves the partial .llvq and its .state on /out.
#
#   DRY_RUN=1 bash ops/jobs/guided-clip-encode-card-4b.sh run   prints the job script, parses it, launches nothing
#   bash ops/jobs/guided-clip-encode-card-4b.sh run             launches the arm
#   bash ops/jobs/guided-clip-encode-card-4b.sh compare         local, light: fetches the new arm and the 2026-11-13
#                                                               baseline and candidate, refuses unless the new state
#                                                               differs from the baseline's by the guided line alone;
#                                                               writes COMPARE-OK
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

CMD=${1:-}
ARM=candidate-clip10
PREREG=proofs/preregistration-guided-clip-gateup-4b-2026-11-14.md
BUCKET=Pier-Jean/jobs-artifacts
SPACE=Pier-Jean/llvq-runner-cuda-qualite
IMAGE_SHA=${IMAGE_SHA:-c0fb594fe50371bc0171de58167dfe7e1ed6f9e6}   # published 2026-11-14T09:27Z from 8107e19
FLAVOR=rtx-pro-6000
THREADS=23                                   # the flavor's vCPU (ops/run.py FLAVORS), as encode-14b.sh
TIMEOUT=${TIMEOUT:-105m}
MODEL=Qwen/Qwen3-4B
MODEL_SHA=1cfa9a7208912126459214e8b04321603b3df60c
DCLM_SHA=dbad8ad71224482740cd9c9d353591adbf62fe04
FP=45fb781f65574d4d                          # the calibration token fingerprint of the weights job (calib-tokens-4b.u32)
GROUPS_G=4
CLIP=10
MANIFEST_1113=e63dad8346c33773bed7a9dc23f30859714bfeed6bd677de0465bc5633286f22   # the 2026-11-13 candidate's state
SEALED_BYTES=1794564765                      # both 2026-11-13 arms (COMPARE-OK of that run)
OUT=guided-clip-4b-2026-11-14
REF=guided-encode-4b-2026-11-13              # the 2026-11-13 baseline and candidate, already in the bucket
WSRC=guided-weights-4b-2026-11-13
D=${OUT_DIR:-$HOME/llvq-guided-clip-2026-11-14}/card
DRY=${DRY_RUN:-0}

refuse() { echo "refused: $*" >&2; exit 1; }

if [ "$CMD" = compare ]; then
  mkdir -p "$D/$ARM" "$D/baseline" "$D/candidate"
  for f in "$ARM.llvq.state" smoke.txt files.sha256 sizes.txt gpu.txt DONE; do
    hf buckets cp "hf://buckets/$BUCKET/$OUT/$ARM/$f" "$D/$ARM/$f" >/dev/null || refuse "$OUT/$ARM/$f absent from the bucket"
  done
  for f in baseline.llvq.state smoke.txt files.sha256 sizes.txt gpu.txt; do
    hf buckets cp "hf://buckets/$BUCKET/$REF/baseline/$f" "$D/baseline/$f" >/dev/null || refuse "$REF/baseline/$f absent from the bucket"
  done
  for f in candidate.llvq.state files.sha256; do
    hf buckets cp "hf://buckets/$BUCKET/$REF/candidate/$f" "$D/candidate/$f" >/dev/null || refuse "$REF/candidate/$f absent from the bucket"
  done
  B=$D/baseline; C=$D/$ARM; O=$D/candidate
  FB=$(grep -o 'calibration token fingerprint [0-9a-f]*' "$B/smoke.txt"); FC=$(grep -o 'calibration token fingerprint [0-9a-f]*' "$C/smoke.txt")
  [ "$FB" = "calibration token fingerprint $FP" ] && [ "$FC" = "$FB" ] || refuse "token fingerprints: '$FB', '$FC', expected $FP"
  grep -q '^guided' "$B/baseline.llvq.state" && refuse "the baseline's state names guided weights"
  grep -qx "guided = gate_proj,up_proj G=$GROUPS_G manifest_sha256=$MANIFEST_1113 clip=$CLIP" "$C/$ARM.llvq.state" \
    || refuse "the new arm's state does not name the 2026-11-13 weights at G=$GROUPS_G with clip=$CLIP"
  if ! diff <(grep -v '^guided' "$B/baseline.llvq.state") <(grep -v '^guided' "$C/$ARM.llvq.state") > "$D/state-diff.txt"; then
    cat "$D/state-diff.txt" >&2
    refuse "the new state and the baseline's differ beyond the guided line (diff above, kept in $D/state-diff.txt)"
  fi
  # The guided line against the 2026-11-13 candidate's: only ` clip=10` may be added.
  diff <(grep '^guided' "$O/candidate.llvq.state") <(grep '^guided' "$C/$ARM.llvq.state") > "$D/guided-line-diff.txt" || true
  echo '== guided line, 2026-11-13 candidate (<) against this arm (>) =='; cat "$D/guided-line-diff.txt"
  [ "$(grep '^guided' "$C/$ARM.llvq.state")" = "$(grep '^guided' "$O/candidate.llvq.state") clip=$CLIP" ] \
    || refuse "the guided line differs from the 2026-11-13 candidate's by more than ' clip=$CLIP'"
  # Not a refusal, a statement (prereg section 4.3): the encode is CPU and f64 on the host for everything the code
  # depends on, so another card is recorded and the result still read. The GPU line (name, compute capability,
  # memory, driver) of each, and whether they agree.
  GB=$(head -1 "$B/gpu.txt"); GC=$(head -1 "$C/gpu.txt")
  if [ "$GB" = "$GC" ]; then GPU="same GPU line: $GB"; else GPU="DIFFERENT GPU lines: baseline '$GB', $ARM '$GC'"; fi
  echo "$GPU"
  SB=$(awk '$2 ~ /baseline\.bin$/ {print $1}' "$B/sizes.txt"); SC=$(awk -v a="$ARM.bin" '$2 ~ a"$" {print $1}' "$C/sizes.txt")
  [ "$SB" = "$SEALED_BYTES" ] && [ "$SC" = "$SB" ] || refuse "sealed sizes: baseline '$SB', $ARM '$SC', $SEALED_BYTES expected"
  HB=$(awk '$2 == "baseline.bin" {print $1}' "$B/files.sha256"); HC=$(awk -v a="$ARM.bin" '$2 == a {print $1}' "$C/files.sha256")
  HO=$(awk '$2 == "candidate.bin" {print $1}' "$O/files.sha256")
  [ -n "$HB" ] && [ -n "$HC" ] && [ -n "$HO" ] || refuse "a sealed sha256 is missing: '$HB', '$HC', '$HO'"
  [ "$HC" != "$HB" ] || refuse "$ARM.bin equals the baseline's sealed file"
  [ "$HC" != "$HO" ] || refuse "$ARM.bin equals the 2026-11-13 candidate's sealed file: the clip changed nothing"
  grep -h '^exact-ppl' "$B/smoke.txt" "$C/smoke.txt" || true
  { echo "$FB, baseline and $ARM"; echo "states equal but for the guided line"; echo "guided line = the 2026-11-13 candidate's + clip=$CLIP"
    echo "sealed $SC B both"; echo "baseline.bin $HB"; echo "candidate.bin $HO"; echo "$ARM.bin $HC"
    echo "$GPU"; date -u +%FT%TZ; } | tee "$D/COMPARE-OK"
  exit 0
fi

[ "$CMD" = run ] || refuse "usage: $0 run|compare"

if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || refuse "$PREREG(.ots) missing"
  # The anchor is the proof (departure E1 of the 2026-11-13 prereg): the banner a stamped prereg keeps is not read.
  ots info "$PREREG.ots" 2>/dev/null | grep -q "$(shasum -a 256 "$PREREG" | cut -d' ' -f1)" \
    || refuse "$PREREG.ots does not anchor the sha256 of $PREREG"
  grep -q "$(shasum -a 256 "$0" | cut -d' ' -f1)" "$PREREG" || refuse "$PREREG does not name this script's sha256"
  git diff --quiet HEAD -- "$0" || refuse "$0 differs from HEAD"
  git ls-files --error-unmatch "$0" >/dev/null 2>&1 || refuse "$0 is not committed"
  case "$IMAGE_SHA" in PIN_ME_AFTER_PUBLISH|TO-BE-SET*|'') refuse "IMAGE_SHA is not pinned: the Space sha the publish returned" ;; esac
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    "from huggingface_hub import HfApi; print(HfApi().space_info('$SPACE').sha)")
  [ "$NOW" = "$IMAGE_SHA" ] || refuse "the Space $SPACE moved to $NOW, expected $IMAGE_SHA"
  if hf buckets ls "hf://buckets/$BUCKET/$OUT/$ARM/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    refuse "$OUT/$ARM/ already exists in the bucket"
  fi
  LIST=$(hf buckets ls "hf://buckets/$BUCKET/$WSRC/" 2>/dev/null | awk '{print $NF}' || true)
  N=$(printf '%s\n' "$LIST" | grep -cE "^$WSRC/w-[0-9]{3}-(gate|up)_proj\.f32$" || true)
  [ "$N" = 72 ] || refuse "$WSRC/ holds $N weight files in the bucket, 72 expected"
  for f in manifest.json files.sha256; do
    printf '%s\n' "$LIST" | grep -qx "$WSRC/$f" || refuse "$WSRC/$f absent from the bucket"
  done
  echo "arm $ARM; image $SPACE@$NOW; $FLAVOR, timeout $TIMEOUT; out $OUT/$ARM"
fi

GUIDED_ENV="LLVQ_GUIDED_WEIGHTS=/scratch/weights LLVQ_GUIDED_GROUPS=$GROUPS_G LLVQ_GUIDED_CLIP=$CLIP"
PRE=$(printf 'ARM=%q\nTHREADS=%q\nMODEL=%q\nMODEL_SHA=%q\nDCLM_SHA=%q\nFP=%q\nGROUPS_G=%q\nCLIP=%q\nMANIFEST_1113=%q\nW=%q\nWSRC=%q\nGUIDED_ENV=%q' \
  "$ARM" "$THREADS" "$MODEL" "$MODEL_SHA" "$DCLM_SHA" "$FP" "$GROUPS_G" "$CLIP" "$MANIFEST_1113" "/out/$OUT/$ARM" "/out/$WSRC" "$GUIDED_ENV")

IFS= read -r -d '' BODY <<'JOB' || true
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
for b in oracle smoke seal; do
  type -P "$b" >/dev/null || { echo "refused: $b is not in this image" >&2; exit 1; }
done
# An image built before the clip would ignore LLVQ_GUIDED_CLIP and encode the 2026-11-13 candidate again.
grep -qa 'LLVQ_GUIDED_CLIP' "$(type -P smoke)" || { echo 'refused: this smoke does not know LLVQ_GUIDED_CLIP' >&2; exit 1; }
mkdir -p "$W" /scratch
stamp() { echo "$1 $(date -u +%FT%TZ) $(date +%s)" | tee -a "$W/timing.txt"; }
stamp job-start
{ nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader
  echo "nproc $(nproc)"; grep -m1 'model name' /proc/cpuinfo || true; grep '^MemTotal:' /proc/meminfo || true
  df -h /scratch "$HOME" 2>&1 | tail -n +2 || true; } | tee "$W/gpu.txt"
mem_sampler() {
  echo 'utc,gpu_used_mib,gpu_total_mib,gpu_util_pct,host_avail_kb,proc,vmrss_kb,vmhwm_kb'
  while :; do
    g=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ') || g=',,'
    [ -n "$g" ] || g=',,'
    a=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null) || a=
    p=; c=; r=; h=
    for d in /proc/[0-9]*; do
      n=$(cat "$d/comm" 2>/dev/null) || continue
      case "$n" in smoke|seal|oracle) p=${d#/proc/}; c=$n; break ;; esac
    done
    if [ -n "$p" ]; then
      r=$(awk '/^VmRSS:/ {print $2}' "/proc/$p/status" 2>/dev/null) || r=
      h=$(awk '/^VmHWM:/ {print $2}' "/proc/$p/status" 2>/dev/null) || h=
    fi
    echo "$(date -u +%FT%TZ),$g,$a,$c,$r,$h"
    sleep 30
  done
}
mem_sampler > "$W/mem.csv" 2>/dev/null &
SAMPLER=$!
trap 'kill "$SAMPLER" 2>/dev/null || true' EXIT
need()  { grep -qF -- "$2" "$1" || { echo "refused: $1 lacks: $2" >&2; exit 1; }; }
needx() { grep -qxF -- "$2" "$1" || { echo "refused: $1 lacks the line: $2" >&2; exit 1; }; }
never() { if grep -qF -- "$2" "$1"; then echo "refused: $1 holds: $2" >&2; exit 1; fi; }

echo '== oracle (hard rule 10) =='
oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$W/oracle-cuda.txt" | tail -2
need "$W/oracle-cuda.txt" MATCH
stamp oracle-done

echo '== the weights: bucket mount to local disk, sha256 checked, the 2026-11-13 manifest =='
mkdir -p /scratch/weights
cp "$WSRC/manifest.json" "$WSRC/files.sha256" /scratch/weights/
cp "$WSRC"/w-*.f32 /scratch/weights/
test "$(ls /scratch/weights/w-*.f32 | wc -l)" = 72
( cd /scratch/weights && sha256sum -c files.sha256 ) > "$W/weights-check.txt"
test "$(grep -c ': OK$' "$W/weights-check.txt")" = 73
MANIFEST_SHA=$(sha256sum /scratch/weights/manifest.json | cut -d' ' -f1)
echo "manifest sha256 $MANIFEST_SHA" | tee -a "$W/weights-check.txt"
[ "$MANIFEST_SHA" = "$MANIFEST_1113" ] || { echo "refused: manifest $MANIFEST_SHA, the 2026-11-13 candidate read $MANIFEST_1113" >&2; exit 1; }
stamp weights-copied

echo "== smoke, arm $ARM =="
stamp smoke-start
# shellcheck disable=SC2086
env LLVQ_MODEL="$MODEL" LLVQ_CALIB=dclm-edu LLVQ_INT4_TYPES=v_proj LLVQ_THREADS="$THREADS" \
    LLVQ_ARTIFACT="$W/$ARM.llvq" $GUIDED_ENV \
  smoke 64 2048 12 4096 cuda nogs tetra1 999 rot 2>&1 | tee "$W/smoke.txt"
stamp smoke-end

# The resolved configuration (smoke.rs header), as encode-14b.sh checks it.
need "$W/smoke.txt" 'codebook     tetra1 → Tetra word map'
need "$W/smoke.txt" 'calibration  dclm-edu, 64 × 2048 = 131072 tokens requested, contiguous prefix from token 0'
need "$W/smoke.txt" "revision $DCLM_SHA"
need "$W/smoke.txt" 'rotation     on (seed 0x110feed)'
need "$W/smoke.txt" '  mode         nogs (group scales off, design C off'
needx "$W/smoke.txt" '  dtype        f32'
need "$W/smoke.txt" ", $THREADS encoder threads"
need "$W/smoke.txt" '  damping      1e-2 (relative to mean(diag H))'
need "$W/smoke.txt" '  h_shrink     1 (H as is, published path)'
need "$W/smoke.txt" '  gain_scale   1 (centroids as fitted, published path)'
need "$W/smoke.txt" 'weights identical, bit for bit (at f32)'
never "$W/smoke.txt" 'WITHOUT `fast-linalg`'
need "$W/smoke.txt" "calibration token fingerprint $FP"
needx "$W/$ARM.llvq.state" 'device = cuda'
needx "$W/$ARM.llvq.state" "model = $MODEL"
need "$W/smoke.txt" "gate_proj, up_proj against gradient-weighted Hessians, G = $GROUPS_G"
need "$W/smoke.txt" ", weights clipped at $CLIP x the group mean"
need "$W/smoke.txt" " clip=$CLIP"
needx "$W/$ARM.llvq.state" "guided = gate_proj,up_proj G=$GROUPS_G manifest_sha256=$MANIFEST_SHA clip=$CLIP"
# The checkpoint smoke resolved: `refs/main` of the hf-hub cache under $HOME (hf-hub ignores HF_HOME).
f="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-4B/refs/main"
if [ -s "$f" ]; then
  echo "refs/main $(tr -d ' \n' < "$f")" | tee "$W/model-revision.txt"
  [ "$(tr -d ' \n' < "$f")" = "$MODEL_SHA" ] || { echo "refused: the checkpoint resolved is not $MODEL_SHA" >&2; exit 1; }
else
  echo "no refs/main under $HOME/.cache/huggingface/hub" | tee "$W/model-revision.txt"
fi

echo '== seal to /scratch, copy to /out, size and sha256 read back =='
stamp seal-start
LLVQ_MODEL="$MODEL" seal "$W/$ARM.llvq" "/scratch/$ARM.bin" 2>&1 | tee "$W/seal.txt"
SHA=$(sha256sum "/scratch/$ARM.bin" | cut -d' ' -f1)
cp "/scratch/$ARM.bin" "$W/$ARM.bin"
sync "$W/$ARM.bin" 2>/dev/null || true
test "$(stat -c %s "$W/$ARM.bin")" = "$(stat -c %s "/scratch/$ARM.bin")"
test "$(sha256sum "$W/$ARM.bin" | cut -d' ' -f1)" = "$SHA"
stamp seal-end

( cd "$W" && sha256sum "$ARM.llvq" "$ARM.llvq.state" "$ARM.bin" ) | tee "$W/files.sha256"
stat -c '%s %n' "$W/$ARM.llvq" "$W/$ARM.bin" | tee "$W/sizes.txt"
sed -n '/^phases:/,/^[^ ]/p' "$W/smoke.txt" | tee "$W/phases.txt"
grep '^exact-ppl' "$W/smoke.txt" | tee "$W/ppl.txt" || true
awk -F, 'NR > 1 && $2 != "" {if ($2 + 0 > g) g = $2 + 0} NR > 1 && $8 != "" {if ($8 + 0 > h) h = $8 + 0}
         END {printf "GPU peak %d MiB sampled; host VmHWM peak %.1f GB\n", g, h / 1e6}' "$W/mem.csv" | tee "$W/peaks.txt"
stamp job-end
date -u > "$W/DONE"
echo '== what landed ==' ; ls -la "$W"
JOB

if [ "$DRY" = 1 ]; then
  JOBSCRIPT="set -euo pipefail
$PRE
$BODY"
  printf '%s\n' "$JOBSCRIPT"
  printf '%s\n' "$JOBSCRIPT" | bash -n && echo "DRY_RUN: arm $ARM parses; $FLAVOR, timeout $TIMEOUT, image $SPACE@$IMAGE_SHA; nothing launched" >&2
  exit 0
fi

uv run ops/run.py bench \
  --image "hf.co/spaces/$SPACE" \
  --flavor "$FLAVOR" --any-flavor --timeout "$TIMEOUT" \
  --bucket "$BUCKET" --out-mount /out \
  --name "guided-clip-encode-4b-$ARM" \
  "$PRE" "$BODY"
