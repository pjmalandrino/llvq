#!/usr/bin/env bash
# The sealed 4B with the blocks after the answer write restored to f16, then with the blocks up to it, full split.
# Preregistered: proofs/preregistration-depth-restore-4b-2026-11-14.md, stamped (ots stamp) before this runs.
#
# The contrast map (docs/mesures/contrast-map-4b-2026-11-14.txt, CERVEAU Q25) reads the MMLU loss as complete when
# the answer is written, after block index 24 of 0..35. Two arms test it on the served object with the probe of the
# 2026-11-12 ablation (ops/jobs/ablation-qualite-4b.sh, whose job body this script reuses): every projection type
# taken from the checkpoint at f16 inside a window of layers (`LLVQ_RESTORE_F16`, `type@first-last`, inclusive,
# 0-based), the rest of the SAME file as sealed.
#   late  : layers 25 to 35, 11 blocks, 77 matrices, 1,110,179,840 weights
#   early : layers 0 to 24, 25 blocks, 175 matrices, 2,523,136,000 weights
# Path, both arms: dense reconstruction, dtype f16, kv f16, `config=none`, the path of the published 63.37 and of
# the ablation. No control arm runs: the pairs are formed against the committed control dump of the ablation,
# docs/data/mmlu-dumps/mmlu-4b-abl-control-FULL.csv (63.3742, picks equal to the published dump on 14,042 of 14,042).
#
# Image: the private Space Pier-Jean/llvq-runner-cuda-qualite at c0fb594f, built from 8107e19; `mmlu.rs` and
# `sealed.rs` are those of the ablation's image but for the guided encoder path, which `mmlu` does not reach.
#
# Cost, estimated: 22 to 24 min an arm on l40sx1 (the seven restore arms of 2026-11-12 ran 21.7 to 23.3 min), oracle
# and checkpoint fetch 2 min: 46 to 50 min, $1.38 to $1.50. Timeout 64m, ceiling $1.92. The late arm runs first.
#
#   DRY_RUN=1 bash ops/jobs/depth-restore-4b.sh   prints the job script, parses it, launches nothing
#   bash ops/jobs/depth-restore-4b.sh             launches the two arms
#   bash ops/jobs/depth-restore-4b.sh pair        local, light: fetches the two dumps and pairs each with the control
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-depth-restore-4b-2026-11-14.md
SPACE=Pier-Jean/llvq-runner-cuda-qualite
IMAGE_SHA=${IMAGE_SHA:-c0fb594fe50371bc0171de58167dfe7e1ed6f9e6}   # published 2026-11-14T09:27Z from 8107e19
BUCKET=Pier-Jean/jobs-artifacts
F=/out/sealed-4b-2026-09-23/qwen3-4b-sealed.bin
SHA=886391a8c03f66dc269cc65c3598c6627dbdcd259180aff36604ef10d37371b8
BYTES=1418224685
O=/out/depth-restore-4b-2026-11-14
FP=a74a6d6213602979
CKPT=Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c
CTRL=docs/data/mmlu-dumps/mmlu-4b-abl-control-FULL.csv
CTRL_MICRO=63.3742                                                 # 8,899 of 14,042, the committed control dump
TIMEOUT=${TIMEOUT:-64m}
DRY=${DRY_RUN:-0}
T=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
win() { echo "$T" | sed "s/,/@$1,/g; s/\$/@$1/"; }
ARMS="late|$(win 25-35)|77|1110179840
early|$(win 0-24)|175|2523136000"

if [ "${1:-}" = pair ]; then
  P=${OUT_DIR:-$HOME/llvq-depth-restore-2026-11-14}; mkdir -p "$P"
  test -x target/release/mmlupair || { echo "refused: target/release/mmlupair not built" >&2; exit 1; }
  for a in late early; do
    hf buckets cp "hf://buckets/$BUCKET/${O#/out/}/mmlu-$a-FULL.csv" "$P/mmlu-$a-FULL.csv"
    # mmlupair prints delta = first minus second: the restore arm goes first.
    target/release/mmlupair "$P/mmlu-$a-FULL.csv" "$CTRL" --no-fpc 2>&1 | tee "$P/mmlupair-$a-vs-control.txt"
  done
  exit 0
fi

if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  ots info "$PREREG.ots" 2>/dev/null | grep -q "$(shasum -a 256 "$PREREG" | cut -d' ' -f1)" \
    || { echo "refused: $PREREG.ots does not anchor the sha256 of $PREREG" >&2; exit 1; }
  grep -q "$(shasum -a 256 "$0" | cut -d' ' -f1)" "$PREREG" || { echo "refused: $PREREG does not name this script's sha256" >&2; exit 1; }
  git diff --quiet HEAD -- "$0" || { echo "refused: this script differs from HEAD" >&2; exit 1; }
  REMOTE=$(hf buckets ls "hf://buckets/$BUCKET/${F#/out/}" 2>/dev/null | awk '$NF ~ /qwen3-4b-sealed\.bin$/ {print $1}' || true)
  [ "$REMOTE" = "$BYTES" ] || { echo "refused: bucket copy of the sealed file is ${REMOTE:-absent} B" >&2; exit 1; }
  if hf buckets ls "hf://buckets/$BUCKET/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already exists" >&2; exit 1
  fi
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    "from huggingface_hub import HfApi; print(HfApi().space_info('$SPACE').sha)")
  [ "$NOW" = "$IMAGE_SHA" ] || { echo "refused: the Space $SPACE moved to $NOW, expected $IMAGE_SHA" >&2; exit 1; }
  echo "sealed 4B: $BYTES B, sha256 $SHA; image $SPACE@$NOW; timeout $TIMEOUT"
fi

PRE=$(printf 'O=%q\nF=%q\nSHA=%q\nBYTES=%q\nFP=%q\nCKPT=%q\nCTRL_MICRO=%q\nARMS=%q' \
  "$O" "$F" "$SHA" "$BYTES" "$FP" "$CKPT" "$CTRL_MICRO" "$ARMS")

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

# micro on a census dump: every question of every subject is scored, so the
# stratified micro equals correct / questions.
micro() { awk -F, '$0 !~ /^#/ && $1 != "subject" { n++; c += $7 } END { printf "%d %d %.4f\n", c, n, 100 * c / n }' "$1"; }
# identical picks, joined on subject, index and qhash
picks() { awk -F, 'FNR == NR { if ($0 !~ /^#/ && $1 != "subject") r[$1 "," $2 "," $4] = $6; next }
  $0 !~ /^#/ && $1 != "subject" { n++; k = $1 "," $2 "," $4; if (k in r) { j++; if (r[k] == $6) s++ } }
  END { printf "%d %d %d\n", n, j, s }' "$1" "$2"; }

S="$O/summary.tsv"
printf 'arm\trestore\tmatrices\tweights\tcorrect\tquestions\tmicro_pct\tgain_vs_control_pp\tstart_utc\tend_utc\n' > "$S"
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
  printf '%s\n' "$JOBSCRIPT" | bash -n && echo "DRY_RUN: job script parses; timeout $TIMEOUT; nothing launched" >&2
  exit 0
fi

uv run ops/run.py bench \
  --image "hf.co/spaces/$SPACE" \
  --flavor l40sx1 --timeout "$TIMEOUT" \
  --bucket "$BUCKET" --out-mount /out \
  --name depth-restore-4b-2026-11-14 \
  "$PRE" "$BODY"
