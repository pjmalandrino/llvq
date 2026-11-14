#!/usr/bin/env bash
# The full-split MMLU arm of the clipped guided gate/up retry, then its two pairings.
# Preregistered: proofs/preregistration-guided-clip-gateup-4b-2026-11-14.md, stamped before this.
#
# Derived from ops/jobs/guided-mmlu-4b.sh, left untouched (a stamped prereg cites its sha256). ONE arm,
# candidate-clip10, on the path of both 2026-11-13 arms: the sealed file, dense reconstruction, dtype f16, kv f16,
# config none, flat allocation, 14,042 questions, a dump per question, plan fingerprint a74a6d6213602979. The
# two 2026-11-13 dumps are not rescored: they are committed (docs/data/mmlu-dumps/mmlu-guided-baseline-FULL.csv
# and mmlu-guided-candidate-FULL.csv) and `pair` reads them.
#
# The arm is encoded on the card by ops/jobs/guided-clip-encode-card-4b.sh and sealed there; this job reads it from
# the bucket, guided-clip-4b-2026-11-14/candidate-clip10/candidate-clip10.bin, and only after
# `guided-clip-encode-card-4b.sh compare` has written COMPARE-OK with its sha256 and size. The image is the
# attempt's private Space, Pier-Jean/llvq-runner-cuda-qualite, at the sha the operator pins below; the script
# refuses while it is the placeholder, and refuses if that Space moved.
#
# Cost, estimated: 23 to 24 min on l40sx1, about $0.70 (the two 2026-11-13 arms, job 6af75129: 2,707 s billed, $1.35;
# f16-full 6aad0dd9: 23 min). Timeout 34m: ceiling $1.02 at $1.80/h.
#
#   DRY_RUN=1 bash ops/jobs/guided-clip-mmlu-4b.sh   prints the job script, parses it, launches nothing
#   bash ops/jobs/guided-clip-mmlu-4b.sh             launches the arm
#   bash ops/jobs/guided-clip-mmlu-4b.sh pair        local: fetches the new dump, runs mmlupair --no-fpc against
#                                                    the 2026-11-13 baseline (primary) and candidate (secondary)
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-guided-clip-gateup-4b-2026-11-14.md
SPACE=${SPACE:-Pier-Jean/llvq-runner-cuda-qualite}
IMAGE_SHA=${IMAGE_SHA:-c0fb594fe50371bc0171de58167dfe7e1ed6f9e6}   # published 2026-11-14T09:27Z from 8107e19
BUCKET=Pier-Jean/jobs-artifacts
ARM=candidate-clip10
IN=guided-clip-4b-2026-11-14
O=/out/guided-clip-mmlu-4b-2026-11-14
D=${OUT_DIR:-$HOME/llvq-guided-clip-2026-11-14}/card
BASE_DUMP=docs/data/mmlu-dumps/mmlu-guided-baseline-FULL.csv
CAND_DUMP=docs/data/mmlu-dumps/mmlu-guided-candidate-FULL.csv
FP=a74a6d6213602979
TIMEOUT=${TIMEOUT:-34m}
DRY=${DRY_RUN:-0}

if [ "${1:-}" = pair ]; then
  P=$D/pair; mkdir -p "$P"
  hf buckets cp "hf://buckets/$BUCKET/${O#/out/}/mmlu-$ARM-FULL.csv" "$P/mmlu-$ARM-FULL.csv"
  test -x target/release/mmlupair || { echo "refused: target/release/mmlupair not built" >&2; exit 1; }
  for f in "$BASE_DUMP" "$CAND_DUMP"; do test -s "$f" || { echo "refused: $f missing" >&2; exit 1; }; done
  # mmlupair prints delta = first minus second; the statistic is the new arm minus the reference, so the new
  # arm goes first both times.
  echo '== primary: candidate-clip10 against the 2026-11-13 baseline =='
  target/release/mmlupair "$P/mmlu-$ARM-FULL.csv" "$BASE_DUMP" --no-fpc 2>&1 | tee "$P/mmlupair-vs-baseline.txt"
  echo '== secondary: candidate-clip10 against the 2026-11-13 unclipped candidate =='
  target/release/mmlupair "$P/mmlu-$ARM-FULL.csv" "$CAND_DUMP" --no-fpc 2>&1 | tee "$P/mmlupair-vs-candidate.txt"
  exit 0
fi

if [ "$DRY" = 1 ] && [ ! -s "$D/COMPARE-OK" ]; then
  # Before the encode exists, a dry run prints the job with placeholders and checks nothing about the arm.
  echo "DRY_RUN: no COMPARE-OK yet in $D; the job below carries placeholders for the sha256 and the size" >&2
  SHA=DRY-$ARM-sha256; BYTES=0
else
  test -s "$D/COMPARE-OK" || { echo "refused: $D/COMPARE-OK missing; run guided-clip-encode-card-4b.sh compare first" >&2; exit 1; }
  SHA=$(awk -v a="$ARM.bin" '$1 == a {print $2}' "$D/COMPARE-OK")
  BASE_SHA=$(awk '$1 == "baseline.bin" {print $2}' "$D/COMPARE-OK")
  BYTES=$(awk '$1 == "sealed" {print $2}' "$D/COMPARE-OK")
  [ -n "$SHA" ] && [ -n "$BASE_SHA" ] && [ -n "$BYTES" ] || { echo "refused: $D/COMPARE-OK lacks a sha256 or the size" >&2; exit 1; }
  [ "$SHA" != "$BASE_SHA" ] || { echo "refused: the new sealed file is the baseline's" >&2; exit 1; }
fi

if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  # The stamped prereg keeps its "NOT YET STAMPED" banner, as every prereg here does: the anchor is the proof.
  ots info "$PREREG.ots" 2>/dev/null | grep -q "$(shasum -a 256 "$PREREG" | cut -d' ' -f1)" \
    || { echo "refused: $PREREG.ots does not anchor the sha256 of $PREREG" >&2; exit 1; }
  if grep -q 'TO BE FILLED' "$PREREG"; then echo "refused: $PREREG still has an empty prediction block" >&2; exit 1; fi
  grep -q "$(shasum -a 256 "$0" | cut -d' ' -f1)" "$PREREG" || { echo "refused: $PREREG does not name this script's sha256" >&2; exit 1; }
  R=$(hf buckets ls "hf://buckets/$BUCKET/$IN/$ARM/" 2>/dev/null | awk -v n="$IN/$ARM/$ARM.bin" '$NF == n {print $1}' || true)
  [ "$R" = "$BYTES" ] || { echo "refused: $IN/$ARM/$ARM.bin is ${R:-absent} B in the bucket, $BYTES expected" >&2; exit 1; }
  if hf buckets ls "hf://buckets/$BUCKET/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already exists" >&2; exit 1
  fi
  case "$IMAGE_SHA" in PIN_ME_AFTER_PUBLISH|TO-BE-SET*|'') echo "refused: IMAGE_SHA is not pinned" >&2; exit 1 ;; esac
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    "from huggingface_hub import HfApi; print(HfApi().space_info('$SPACE').sha)")
  [ "$NOW" = "$IMAGE_SHA" ] || { echo "refused: the Space moved to $NOW, expected $IMAGE_SHA" >&2; exit 1; }
  echo "arm: $ARM $SHA, $BYTES B; image $NOW; timeout $TIMEOUT"
fi

ARMS="$ARM|/out/$IN/$ARM/$ARM.bin|$SHA"
PRE=$(printf 'O=%q\nBYTES=%q\nFP=%q\nARMS=%q' "$O" "$BYTES" "$FP" "$ARMS")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee "$O/gpu.txt" || true
echo '== oracle (hard rule 10) =='
oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle-cuda.txt" | tail -2
grep -q MATCH "$O/oracle-cuda.txt"
micro() { awk -F, '$0 !~ /^#/ && $1 != "subject" { n++; c += $7 } END { printf "%d %d %.4f\n", c, n, 100 * c / n }' "$1"; }
S="$O/summary.tsv"
printf 'arm\tfile\tsha256\tcorrect\tquestions\tmicro_pct\tstart_utc\tend_utc\n' > "$S"
while IFS='|' read -r ARM F SHA; do
  echo "== arm $ARM: $F, dense f16, FULL split ==" ; T0=$(date -u +%FT%TZ)
  test "$(stat -c %s "$F")" = "$BYTES"
  test "$(sha256sum "$F" | cut -d' ' -f1)" = "$SHA"
  echo "$SHA  $F" >> "$O/files.sha256"
  D="$O/mmlu-$ARM-FULL.csv"
  LLVQ_MMLU_ALLOC=flat LLVQ_MMLU_DUMP="$D" mmlu "$F" cuda </dev/null 2>&1 | tee "$O/out-$ARM.txt" | tail -6
  T1=$(date -u +%FT%TZ)
  grep -qxF '# dtype=f16' "$D"
  grep -qxF '# limit=census' "$D"
  grep -qxF '# alloc=flat, 100..1534 per subject, 14042 questions' "$D"
  grep -qxF '# config=none' "$D"
  grep -qxF '# arithmetic=dense reconstruction' "$D"
  grep -qxF '# kv=f16' "$D"
  test "$(tail -n 1 "$D")" = "# end fingerprint=$FP questions=14042"
  if grep -q 'restored' "$D"; then echo "FAIL: arm $ARM carries a restoration" >&2; exit 1; fi
  read -r C N MIC < <(micro "$D")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$ARM" "$F" "$SHA" "$C" "$N" "$MIC" "$T0" "$T1" >> "$S"
  echo "dump ok: $D  micro $MIC"
done <<< "$ARMS"
echo '== summary ==' ; cat "$S"
ls -la "$O"
JOB

if [ "$DRY" = 1 ]; then
  JOBSCRIPT="set -euo pipefail
$PRE
$BODY"
  printf '%s\n' "$JOBSCRIPT"
  printf '%s\n' "$JOBSCRIPT" | bash -n && echo "DRY_RUN: job script parses; image $SPACE@$IMAGE_SHA; nothing launched" >&2
  exit 0
fi

uv run ops/run.py bench \
  --image "hf.co/spaces/$SPACE" \
  --flavor l40sx1 --timeout "$TIMEOUT" \
  --bucket "$BUCKET" --out-mount /out \
  --name guided-clip-mmlu-4b-2026-11-14 \
  "$PRE" "$BODY"
