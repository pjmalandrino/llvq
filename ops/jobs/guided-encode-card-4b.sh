#!/usr/bin/env bash
# The two encodes of the guided gate/up attempt (V1) ON A CARD, one job per arm, by the operator's decision of
# 2026-11-13 (CERVEAU O12, commit 49b714d). Preregistered: proofs/preregistration-guided-gateup-4b-2026-11-13.md;
# this route is departure E2 of its -ECARTS file, which must be stamped before either arm launches.
#
# The recipe is the Mac script's (ops/jobs/guided-encode-4b.sh): the 2026-09-18 DCLM base's `tetra1`, rotation
# 0x110feed, nogs, damping 1e-2, h_shrink 1, gain_scale 1, LLVQ_INT4_TYPES=v_proj, dclm-edu 64 x 2048 from the prefix,
# f32, every block, eval wikitext-2 12 x 4096, with two changes taken from ops/jobs/encode-14b.sh: device `cuda`
# instead of `metal`, and 23 encoder threads, the flavor's vCPU (ops/run.py FLAVORS). The ONLY difference between the
# arms is LLVQ_GUIDED_WEIGHTS and LLVQ_GUIDED_GROUPS=4. Both arms run on the same flavor and the same image, pinned
# by its Space sha; a card encode is a different object from a Mac encode (jobs.csv line 120), so neither arm is
# compared with the 2026-09-18 Mac base.
#
# The image: the private Space Pier-Jean/llvq-runner-cuda-qualite, published from this branch so that the shared
# Space the speed phase pins does not move. It holds `smoke` with the guided path, `seal` and `oracle`.
#
# Cost per arm, estimated (no 4B Tetra encode has run on a card): the 14B's Tetra quantize rate on this flavor,
# 556 us a block and thread over 23 vCPU (innovation-cout-encodage-2026-11-12 lines 58-60), on the 4B's 1.475e8
# lattice blocks, is 59 min of quantize phase; the Mac's other phases (factor 770.7 s, capture 337.2 s, rest 57.7 s,
# dclm-4b-2026-09-18 lines 46-49) are taken as an upper bound for 23 threads and a card: 20 min; image pull and start
# 3, oracle 1, checkpoint download and f32 load 4, two 12-window perplexities 2, verify 1, seal and copies 3:
# 85 to 95 min, $3.90 to $4.35 at $2.75/h. The candidate adds the weights copy (604 MB, 1 min) and +13 % of the
# factor phase (+2 min). Timeout 100m by the operator's decision: ceiling $4.58 an arm. The margin is thin: the
# estimate's top is 5 min under the timeout. A cut leaves the partial .llvq and its .state on /out.
#
#   DRY_RUN=1 bash ops/jobs/guided-encode-card-4b.sh baseline|candidate   prints the job script, parses it
#   IMAGE_SHA=<sha> bash ops/jobs/guided-encode-card-4b.sh baseline       launches one arm (both can run at once)
#   bash ops/jobs/guided-encode-card-4b.sh compare                        local, light: fetches the two states and
#                                                                         logs, refuses unless they differ by the
#                                                                         guided line alone; writes COMPARE-OK
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

ARM=${1:-}
PREREG=proofs/preregistration-guided-gateup-4b-2026-11-13.md
ECARTS=proofs/preregistration-guided-gateup-4b-2026-11-13-ECARTS.md
BUCKET=Pier-Jean/jobs-artifacts
SPACE=Pier-Jean/llvq-runner-cuda-qualite
IMAGE_SHA=${IMAGE_SHA:-TO-BE-SET-AFTER-PUBLISH}
FLAVOR=rtx-pro-6000
THREADS=23                                   # the flavor's vCPU (ops/run.py FLAVORS), as encode-14b.sh
TIMEOUT=${TIMEOUT:-100m}
MODEL=Qwen/Qwen3-4B
MODEL_SHA=1cfa9a7208912126459214e8b04321603b3df60c
DCLM_SHA=dbad8ad71224482740cd9c9d353591adbf62fe04
FP=45fb781f65574d4d                          # the calibration token fingerprint of the weights job (calib-tokens-4b.u32)
GROUPS_G=4
OUT=guided-encode-4b-2026-11-13
WSRC=guided-weights-4b-2026-11-13
D=${OUT_DIR:-$HOME/llvq-guided-2026-11-13}/card
DRY=${DRY_RUN:-0}

refuse() { echo "refused: $*" >&2; exit 1; }

if [ "$ARM" = compare ]; then
  mkdir -p "$D"
  for a in baseline candidate; do
    mkdir -p "$D/$a"
    for f in "$a.llvq.state" smoke.txt files.sha256 sizes.txt gpu.txt DONE; do
      hf buckets cp "hf://buckets/$BUCKET/$OUT/$a/$f" "$D/$a/$f" >/dev/null || refuse "$OUT/$a/$f absent from the bucket"
    done
  done
  B=$D/baseline; C=$D/candidate
  FB=$(grep -o 'calibration token fingerprint [0-9a-f]*' "$B/smoke.txt"); FC=$(grep -o 'calibration token fingerprint [0-9a-f]*' "$C/smoke.txt")
  [ "$FB" = "calibration token fingerprint $FP" ] && [ "$FC" = "$FB" ] || refuse "token fingerprints: '$FB', '$FC', expected $FP"
  grep -q '^guided' "$B/baseline.llvq.state" && refuse "the baseline's state names guided weights"
  grep -q "^guided = gate_proj,up_proj G=$GROUPS_G manifest_sha256=" "$C/candidate.llvq.state" || refuse "the candidate's state does not name its guided weights"
  if ! diff <(grep -v '^guided' "$B/baseline.llvq.state") <(grep -v '^guided' "$C/candidate.llvq.state") > "$D/state-diff.txt"; then
    cat "$D/state-diff.txt" >&2
    refuse "the two run states differ beyond the guided line (diff above, kept in $D/state-diff.txt)"
  fi
  SB=$(awk '$2 ~ /baseline\.bin$/ {print $1}' "$B/sizes.txt"); SC=$(awk '$2 ~ /candidate\.bin$/ {print $1}' "$C/sizes.txt")
  [ -n "$SB" ] && [ "$SB" = "$SC" ] || refuse "sealed sizes differ or are missing: '$SB' against '$SC'"
  HB=$(awk '$2 == "baseline.bin" {print $1}' "$B/files.sha256"); HC=$(awk '$2 == "candidate.bin" {print $1}' "$C/files.sha256")
  [ -n "$HB" ] && [ -n "$HC" ] && [ "$HB" != "$HC" ] || refuse "sealed sha256 missing or equal: '$HB', '$HC'"
  grep -h '^exact-ppl' "$B/smoke.txt" "$C/smoke.txt" || true
  # Not a refusal, a statement: two jobs may land on two machines. The GPU line (name, compute capability,
  # memory, driver) of each arm, and whether they agree; a different driver can choose other cuBLAS kernels.
  GB=$(head -1 "$B/gpu.txt"); GC=$(head -1 "$C/gpu.txt")
  if [ "$GB" = "$GC" ]; then GPU="same GPU line: $GB"; else GPU="DIFFERENT GPU lines: baseline '$GB', candidate '$GC'"; fi
  echo "$GPU"
  { echo "$FB, both arms"; echo "states equal but for the guided line"; echo "sealed $SB B both arms"
    echo "baseline.bin $HB"; echo "candidate.bin $HC"; echo "$GPU"; date -u +%FT%TZ; } | tee "$D/COMPARE-OK"
  exit 0
fi

case "$ARM" in baseline|candidate) ;; *) refuse "usage: $0 baseline|candidate|compare" ;; esac

if [ "$DRY" != 1 ]; then
  for f in "$PREREG" "$ECARTS"; do
    { test -f "$f" && test -f "$f.ots"; } || refuse "$f(.ots) missing"
    ots info "$f.ots" 2>/dev/null | grep -q "$(shasum -a 256 "$f" | cut -d' ' -f1)" \
      || refuse "$f.ots does not anchor the sha256 of $f"
  done
  grep -q '^## E2' "$ECARTS" || refuse "$ECARTS has no departure E2: the card route is not declared"
  grep -q "$(shasum -a 256 "$0" | cut -d' ' -f1)" "$ECARTS" || refuse "$ECARTS does not name this script's sha256"
  git diff --quiet HEAD -- "$0" || refuse "$0 differs from HEAD"
  git ls-files --error-unmatch "$0" >/dev/null 2>&1 || refuse "$0 is not committed"
  case "$IMAGE_SHA" in TO-BE-SET*) refuse "IMAGE_SHA is not set: the Space sha the publish returned" ;; esac
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    "from huggingface_hub import HfApi; print(HfApi().space_info('$SPACE').sha)")
  [ "$NOW" = "$IMAGE_SHA" ] || refuse "the Space $SPACE moved to $NOW, expected $IMAGE_SHA"
  if hf buckets ls "hf://buckets/$BUCKET/$OUT/$ARM/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    refuse "$OUT/$ARM/ already exists in the bucket"
  fi
  if [ "$ARM" = candidate ]; then
    LIST=$(hf buckets ls "hf://buckets/$BUCKET/$WSRC/" 2>/dev/null | awk '{print $NF}' || true)
    N=$(printf '%s\n' "$LIST" | grep -cE "^$WSRC/w-[0-9]{3}-(gate|up)_proj\.f32$" || true)
    [ "$N" = 72 ] || refuse "$WSRC/ holds $N weight files in the bucket, 72 expected"
    for f in manifest.json files.sha256; do
      printf '%s\n' "$LIST" | grep -qx "$WSRC/$f" || refuse "$WSRC/$f absent from the bucket"
    done
  fi
  echo "arm $ARM; image $SPACE@$NOW; $FLAVOR, timeout $TIMEOUT; out $OUT/$ARM"
fi

GUIDED_ENV=
[ "$ARM" = candidate ] && GUIDED_ENV="LLVQ_GUIDED_WEIGHTS=/scratch/weights LLVQ_GUIDED_GROUPS=$GROUPS_G"
PRE=$(printf 'ARM=%q\nTHREADS=%q\nMODEL=%q\nMODEL_SHA=%q\nDCLM_SHA=%q\nFP=%q\nGROUPS_G=%q\nW=%q\nWSRC=%q\nGUIDED_ENV=%q' \
  "$ARM" "$THREADS" "$MODEL" "$MODEL_SHA" "$DCLM_SHA" "$FP" "$GROUPS_G" "/out/$OUT/$ARM" "/out/$WSRC" "$GUIDED_ENV")

IFS= read -r -d '' BODY <<'JOB' || true
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
for b in oracle smoke seal; do
  type -P "$b" >/dev/null || { echo "refused: $b is not in this image" >&2; exit 1; }
done
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

if [ "$ARM" = candidate ]; then
  echo '== the weights: bucket mount to local disk, sha256 checked =='
  mkdir -p /scratch/weights
  cp "$WSRC/manifest.json" "$WSRC/files.sha256" /scratch/weights/
  cp "$WSRC"/w-*.f32 /scratch/weights/
  test "$(ls /scratch/weights/w-*.f32 | wc -l)" = 72
  ( cd /scratch/weights && sha256sum -c files.sha256 ) > "$W/weights-check.txt"
  test "$(grep -c ': OK$' "$W/weights-check.txt")" = 73
  MANIFEST_SHA=$(sha256sum /scratch/weights/manifest.json | cut -d' ' -f1)
  echo "manifest sha256 $MANIFEST_SHA" | tee -a "$W/weights-check.txt"
  stamp weights-copied
fi

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
if [ "$ARM" = baseline ]; then
  need "$W/smoke.txt" '  guided       off'
  never "$W/smoke.txt" 'guided weights:'
  if grep -q '^guided' "$W/$ARM.llvq.state"; then echo "refused: the baseline's state names guided weights" >&2; exit 1; fi
else
  need "$W/smoke.txt" "gate_proj, up_proj against gradient-weighted Hessians, G = $GROUPS_G"
  needx "$W/$ARM.llvq.state" "guided = gate_proj,up_proj G=$GROUPS_G manifest_sha256=$MANIFEST_SHA"
fi
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
  --name "guided-encode-4b-$ARM" \
  "$PRE" "$BODY"
