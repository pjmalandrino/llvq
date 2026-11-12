#!/usr/bin/env bash
# Attribution of the sealed 4B's MMLU loss by projection group, full split.
# Preregistered: proofs/preregistration-ablation-qualite-4b-2026-11-12.md,
# stamped (ots stamp) before this runs.
#
# M2's design (m2-attribution-4b-2026-09-02) replayed on the sealed, served
# object: `qwen3-4b-sealed.bin` (63.37 against 70.14 for FP16, 14,042
# questions). Each arm takes the named projection types from the checkpoint at
# f16 (`LLVQ_RESTORE_F16`), every layer at once, and scores the rest of the
# SAME file as sealed. Constant file, so the bar is the paired interval, not
# the calibration sigma.
#
# Path, every arm: dense reconstruction, dtype f16, kv f16, `config=none`.
# That is the path of the published 63.37 (embed-q4-swap-2026-09-23) and of
# the FP16 census dump (f16-full-2026-09-18). The restore arms cannot run on
# the kernel (`mmlu` refuses `LLVQ_RESTORE_*` beside `LLVQ_CONFIG`), so the
# served-control arm runs dense too: the pairs are like for like.
#
# What the sealed file holds (int4swap, dclm-rowscales journals): lattice
# codes for q, k, gate, up (36 each) and down at layers 0-11 and 24-35; int4
# g128 records for v (36), o (36) and down at layers 12-23; the embedding
# (tied head) in int4 g64. A restore arm replaces lattice and int4 matrices
# alike; the embedding is not a projection type and is never restored.
#
# No image republication: the Space holds a03a2bad, built from commit 36230ff
# (2026-11-11), an ancestor of main with `mmlu.rs`, `sealed.rs`, `loader.rs`,
# `model.rs`, `eval.rs`, `corpus.rs`, `llvq-artifact` and `ops/Dockerfile.cuda`
# identical to main's. `mmlu` and `oracle` are built with cuda and copied in.
#
# Cost, estimated: ~23 min a census arm on l40sx1 (f16-full 6aad0dd9: 23 min;
# sealed-rownorms-mmlu 6ab42404: 23 min with oracle; errmap 6aaa4e4e: 45 min
# for two arms; embed-q4-swap 6ab38781: 29 min for one restore arm plus a
# limit-40 arm). Restore arms ~24 min (checkpoint fetch on the first one).
# Seven arms ~169 min, ~$5.07; with WITH_ALL=1 eight arms ~193 min, ~$5.79.
# Timeout 200m = $6.00 ceiling at $1.80/h.
#
#   DRY_RUN=1 bash ops/jobs/ablation-qualite-4b.sh   prints the job script, parses it, launches nothing
#   bash ops/jobs/ablation-qualite-4b.sh             launches (seven arms)
#   WITH_ALL=1 bash ops/jobs/ablation-qualite-4b.sh  launches with the optional eighth arm, `all`
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-ablation-qualite-4b-2026-11-12.md
IMAGE_SHA=${IMAGE_SHA:-a03a2badec0dc876b2231f30a65abe220d80fa28}   # Space revision; its COMMIT file names 36230ff
BUCKET=Pier-Jean/jobs-artifacts
F=/out/sealed-4b-2026-09-23/qwen3-4b-sealed.bin
SHA=886391a8c03f66dc269cc65c3598c6627dbdcd259180aff36604ef10d37371b8
BYTES=1418224685
REF=/out/embed-q4-swap-2026-09-23/mmlu-4b-embed-q4-swap-FULL.csv   # the published 63.37 dump
O=/out/ablation-qualite-4b-2026-11-12
FP=a74a6d6213602979
CKPT=Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c       # the snapshot int4swap read
TIMEOUT=${TIMEOUT:-200m}
DRY=${DRY_RUN:-0}

# arm name | LLVQ_RESTORE_F16 value | matrices | weights (computed from the 4B's shapes:
# 36 layers, hidden 2560, q/o 4096, k/v 1024, intermediate 9728). Order = what a
# timeout cuts last: the two group arms are sums checks, `all` is optional.
ARMS='control||0|0
qkv|q_proj,k_proj,v_proj|108|566231040
o|o_proj|36|377487360
gateup|gate_proj,up_proj|72|1793064960
down|down_proj|36|896532480
attn|q_proj,k_proj,v_proj,o_proj|144|943718400
mlp|gate_proj,up_proj,down_proj|108|2689597440'
if [ "${WITH_ALL:-0}" = 1 ]; then
  ARMS="$ARMS
all|all|252|3633315840"
fi

if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  if grep -q 'TO BE FILLED' "$PREREG"; then echo "refused: $PREREG still has an empty prediction block" >&2; exit 1; fi
  if grep -qF '[operator:' "$PREREG"; then echo "refused: $PREREG does not say whether the all arm is included" >&2; exit 1; fi
  if grep -qxF 'ALL ARM: INCLUDED' "$PREREG"; then [ "${WITH_ALL:-0}" = 1 ] || { echo "refused: the prereg includes the all arm, set WITH_ALL=1" >&2; exit 1; }
  elif grep -qxF 'ALL ARM: NOT INCLUDED' "$PREREG"; then [ "${WITH_ALL:-0}" != 1 ] || { echo "refused: the prereg excludes the all arm" >&2; exit 1; }
  else echo "refused: $PREREG has no 'ALL ARM: INCLUDED' or 'ALL ARM: NOT INCLUDED' line" >&2; exit 1; fi
  REMOTE=$(hf buckets ls "hf://buckets/$BUCKET/${F#/out/}" 2>/dev/null | awk '$NF ~ /qwen3-4b-sealed\.bin$/ {print $1}' || true)
  [ "$REMOTE" = "$BYTES" ] || { echo "refused: bucket copy of the sealed file is ${REMOTE:-absent} B" >&2; exit 1; }
  RREF=$(hf buckets ls "hf://buckets/$BUCKET/${REF#/out/}" 2>/dev/null | awk '$NF ~ /mmlu-4b-embed-q4-swap-FULL\.csv$/ {print $1}' || true)
  [ -n "$RREF" ] || { echo "refused: reference dump ${REF#/out/} absent from the bucket" >&2; exit 1; }
  if hf buckets ls "hf://buckets/$BUCKET/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already exists" >&2; exit 1
  fi
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    'from huggingface_hub import HfApi; print(HfApi().space_info("Pier-Jean/llvq-runner-cuda").sha)')
  [ "$NOW" = "$IMAGE_SHA" ] || { echo "refused: the Space moved to $NOW, expected $IMAGE_SHA" >&2; exit 1; }
  echo "sealed 4B: $BYTES B, sha256 $SHA; image $NOW; timeout $TIMEOUT"
fi

PRE=$(printf 'O=%q\nF=%q\nSHA=%q\nBYTES=%q\nREF=%q\nFP=%q\nCKPT=%q\nARMS=%q' \
  "$O" "$F" "$SHA" "$BYTES" "$REF" "$FP" "$CKPT" "$ARMS")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee "$O/gpu.txt" || true
echo '== oracle (hard rule 10) =='
oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle-cuda.txt" | tail -2
grep -q MATCH "$O/oracle-cuda.txt"
echo '== the sealed file on the mount: bytes and sha256 =='
test "$(stat -c %s "$F")" = "$BYTES"
sha256sum "$F" | tee "$O/files.sha256"
test "$(cut -d' ' -f1 "$O/files.sha256")" = "$SHA"
test -s "$REF"

# micro on a census dump: every question of every subject is scored, so the
# stratified micro equals correct / questions.
micro() { awk -F, '$0 !~ /^#/ && $1 != "subject" { n++; c += $7 } END { printf "%d %d %.4f\n", c, n, 100 * c / n }' "$1"; }
# identical picks, joined on subject, index and qhash
picks() { awk -F, 'FNR == NR { if ($0 !~ /^#/ && $1 != "subject") r[$1 "," $2 "," $4] = $6; next }
  $0 !~ /^#/ && $1 != "subject" { n++; k = $1 "," $2 "," $4; if (k in r) { j++; if (r[k] == $6) s++ } }
  END { printf "%d %d %d\n", n, j, s }' "$1" "$2"; }

S="$O/summary.tsv"
printf 'arm\trestore\tmatrices\tweights\tcorrect\tquestions\tmicro_pct\tgain_vs_control_pp\tstart_utc\tend_utc\n' > "$S"
CTRL_MICRO=
while IFS='|' read -r ARM R M W; do
  D="$O/mmlu-$ARM-FULL.csv"
  T="$O/out-$ARM.txt"
  echo "== arm $ARM: restore '${R:-none}', dense f16, FULL split ==" ; T0=$(date -u +%FT%TZ) ; echo "$T0"
  if [ -z "$R" ]; then
    LLVQ_MMLU_ALLOC=flat LLVQ_MMLU_DUMP="$D" mmlu "$F" cuda </dev/null 2>&1 | tee "$T" | tail -6
  else
    LLVQ_MODEL="$CKPT" LLVQ_RESTORE_F16="$R" LLVQ_MMLU_ALLOC=flat LLVQ_MMLU_DUMP="$D" \
      mmlu "$F" cuda </dev/null 2>&1 | tee "$T" | tail -6
  fi
  T1=$(date -u +%FT%TZ) ; echo "$T1"
  grep -qxF '# dtype=f16' "$D"
  grep -qxF '# limit=census' "$D"
  grep -qxF '# alloc=flat, 100..1534 per subject, 14042 questions' "$D"
  grep -qxF '# config=none' "$D"
  grep -qxF '# arithmetic=dense reconstruction' "$D"
  grep -qxF '# kv=f16' "$D"
  test "$(tail -n 1 "$D")" = "# end fingerprint=$FP questions=14042"
  if [ -n "$R" ]; then
    grep -qF "($M matrices, $W weights)" "$T" || { echo "FAIL: arm $ARM did not restore $M matrices / $W weights" >&2; exit 1; }
  else
    if grep -q 'restored' "$D"; then echo 'FAIL: the control arm carries a restoration' >&2; exit 1; fi
  fi
  read -r C N MIC < <(micro "$D")
  if [ "$ARM" = control ]; then
    CTRL_MICRO=$MIC
    read -r PN PJ PS < <(picks "$REF" "$D")
    echo "control against the published 63.37 dump: $PS identical picks of $PJ joined, $PN questions" | tee "$O/control-picks.txt"
    # Control 3 of the prereg: at least 99.5 % identical picks (13,972 of 14,042),
    # else nothing more is paid for.
    [ "$PJ" = 14042 ] && [ "$PS" -ge 13972 ] || { echo 'FAIL: the control arm does not reproduce the published picks; stopping' >&2; exit 1; }
  fi
  G=$(awk -v a="$MIC" -v b="$CTRL_MICRO" 'BEGIN { printf "%+.4f", a - b }')
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$ARM" "${R:-none}" "$M" "$W" "$C" "$N" "$MIC" "$G" "$T0" "$T1" >> "$S"
  echo "dump ok: $D  micro $MIC  gain $G pp"
done <<< "$ARMS"
echo '== summary ==' ; cat "$S"
wc -l "$O"/*.csv ; ls -la "$O"
JOB

if [ "$DRY" = 1 ]; then
  JOBSCRIPT="set -euo pipefail
$PRE
$BODY"
  printf '%s\n' "$JOBSCRIPT"
  printf '%s\n' "$JOBSCRIPT" | bash -n && echo "DRY_RUN: job script parses; nothing launched" >&2
  exit 0
fi

uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor l40sx1 --timeout "$TIMEOUT" \
  --bucket "$BUCKET" --out-mount /out \
  --name ablation-qualite-4b \
  "$PRE" "$BODY"
