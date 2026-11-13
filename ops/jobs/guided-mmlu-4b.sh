#!/usr/bin/env bash
# The two full-split MMLU arms of the guided gate/up attempt (V1), then their pairing.
# Preregistered: proofs/preregistration-guided-gateup-4b-2026-11-13.md, stamped before this.
#
# Both arms on the path of the 2026-09-18 DCLM base (out-dclm.txt) and of the ablation of 2026-11-12:
# the sealed file, dense reconstruction, dtype f16, kv f16, config none, flat allocation, 14,042
# questions, a dump per question. The baseline first: if its dump fails a check, the candidate is not
# paid for.
#
# The image (departure E2). The arms are encoded on the card by ops/jobs/guided-encode-card-4b.sh and
# sealed there; this job reads them from the bucket, guided-encode-4b-2026-11-13/<arm>/<arm>.bin, and only
# after `guided-encode-card-4b.sh compare` has written COMPARE-OK with their sha256 and size.
# The shared Space this script first pinned, bb28db3b (commit e71e5918), has moved since (655ed135 at
# 16:09Z on 2026-11-13, commit f23a109), and it moves with the speed phase. So the scoring runs by default
# on the attempt's own private Space, Pier-Jean/llvq-runner-cuda-qualite, which also encoded the arms.
# Its mmlu.rs, sealed.rs, loader.rs, model.rs, eval.rs, artifact2.rs, oracle.rs, seal.rs and llvq-artifact
# are identical to 36230ff's, the commit of the image that scored the ablation and the f16 and DCLM-base
# dumps (git diff --stat 36230ff HEAD, empty, read on 2026-11-13); so are f23a109's. SPACE and IMAGE_SHA
# name the image; the script refuses if that Space moved.
#
# Cost, estimated: 23 to 24 min an arm on l40sx1 (f16-full 6aad0dd9: 23 min; sealed-rownorms-mmlu
# 6ab42404: 23 min with oracle; dclm-4b 6aad610d: 23 min), two arms 46 to 48 min, $1.38 to $1.44.
# Timeout 64m: ceiling $1.92 at $1.80/h.
#
#   DRY_RUN=1 bash ops/jobs/guided-mmlu-4b.sh   prints the job script, parses it, launches nothing
#   bash ops/jobs/guided-mmlu-4b.sh             launches the two arms
#   bash ops/jobs/guided-mmlu-4b.sh pair        local: fetches the two dumps, runs mmlupair --no-fpc
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-guided-gateup-4b-2026-11-13.md
SPACE=${SPACE:-Pier-Jean/llvq-runner-cuda-qualite}
IMAGE_SHA=${IMAGE_SHA:-TO-BE-SET-AFTER-PUBLISH}     # the Space revision the publish returned
BUCKET=Pier-Jean/jobs-artifacts
IN=guided-encode-4b-2026-11-13
O=/out/guided-mmlu-4b-2026-11-13
D=${OUT_DIR:-$HOME/llvq-guided-2026-11-13}/card
FP=a74a6d6213602979
TIMEOUT=${TIMEOUT:-64m}
DRY=${DRY_RUN:-0}

if [ "${1:-}" = pair ]; then
  P=$D/pair; mkdir -p "$P"
  for a in baseline candidate; do
    hf buckets cp "hf://buckets/$BUCKET/${O#/out/}/mmlu-$a-FULL.csv" "$P/mmlu-$a-FULL.csv"
  done
  test -x target/release/mmlupair || { echo "refused: target/release/mmlupair not built" >&2; exit 1; }
  # mmlupair prints delta = first minus second; the prereg's statistic is candidate minus baseline,
  # so the candidate goes first.
  target/release/mmlupair "$P/mmlu-candidate-FULL.csv" "$P/mmlu-baseline-FULL.csv" --no-fpc 2>&1 | tee "$P/mmlupair.txt"
  exit 0
fi

if [ "$DRY" = 1 ] && [ ! -s "$D/COMPARE-OK" ]; then
  # Before the encodes exist, a dry run prints the job with placeholders and checks nothing about the arms.
  echo "DRY_RUN: no COMPARE-OK yet in $D; the job below carries placeholders for the two sha256 and the size" >&2
  BASE_SHA=DRY-baseline-sha256; CAND_SHA=DRY-candidate-sha256; BYTES=0
else
  test -s "$D/COMPARE-OK" || { echo "refused: $D/COMPARE-OK missing; run guided-encode-card-4b.sh compare first" >&2; exit 1; }
  BASE_SHA=$(awk '$1 == "baseline.bin" {print $2}' "$D/COMPARE-OK")
  CAND_SHA=$(awk '$1 == "candidate.bin" {print $2}' "$D/COMPARE-OK")
  BYTES=$(awk '$1 == "sealed" {print $2}' "$D/COMPARE-OK")
  [ -n "$BASE_SHA" ] && [ -n "$CAND_SHA" ] && [ -n "$BYTES" ] || { echo "refused: $D/COMPARE-OK lacks a sha256 or the size" >&2; exit 1; }
  [ "$BASE_SHA" != "$CAND_SHA" ] || { echo "refused: the two sealed files are identical" >&2; exit 1; }
fi

if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  # The stamped prereg keeps its "NOT YET STAMPED" banner, as every prereg here does: the anchor is the proof.
  ots info "$PREREG.ots" 2>/dev/null | grep -q "$(shasum -a 256 "$PREREG" | cut -d' ' -f1)" \
    || { echo "refused: $PREREG.ots does not anchor the sha256 of $PREREG" >&2; exit 1; }
  if grep -q 'TO BE FILLED' "$PREREG"; then echo "refused: $PREREG still has an empty prediction block" >&2; exit 1; fi
  for a in baseline candidate; do
    R=$(hf buckets ls "hf://buckets/$BUCKET/$IN/$a/" 2>/dev/null | awk -v n="$IN/$a/$a.bin" '$NF == n {print $1}' || true)
    [ "$R" = "$BYTES" ] || { echo "refused: $IN/$a/$a.bin is ${R:-absent} B in the bucket, $BYTES expected" >&2; exit 1; }
  done
  if hf buckets ls "hf://buckets/$BUCKET/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already exists" >&2; exit 1
  fi
  ECARTS=proofs/preregistration-guided-gateup-4b-2026-11-13-ECARTS.md
  { test -f "$ECARTS" && test -f "$ECARTS.ots"; } || { echo "refused: $ECARTS(.ots) missing" >&2; exit 1; }
  ots info "$ECARTS.ots" 2>/dev/null | grep -q "$(shasum -a 256 "$ECARTS" | cut -d' ' -f1)" \
    || { echo "refused: $ECARTS.ots does not anchor the sha256 of $ECARTS" >&2; exit 1; }
  grep -q '^## E2' "$ECARTS" || { echo "refused: $ECARTS has no departure E2 (the card route)" >&2; exit 1; }
  case "$IMAGE_SHA" in TO-BE-SET*) echo "refused: IMAGE_SHA is not set" >&2; exit 1 ;; esac
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    "from huggingface_hub import HfApi; print(HfApi().space_info('$SPACE').sha)")
  [ "$NOW" = "$IMAGE_SHA" ] || { echo "refused: the Space moved to $NOW, expected $IMAGE_SHA" >&2; exit 1; }
  echo "arms: baseline $BASE_SHA, candidate $CAND_SHA, $BYTES B each; image $NOW; timeout $TIMEOUT"
fi

ARMS="baseline|/out/$IN/baseline/baseline.bin|$BASE_SHA
candidate|/out/$IN/candidate/candidate.bin|$CAND_SHA"
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
  printf '%s\n' "$JOBSCRIPT" | bash -n && echo "DRY_RUN: job script parses; nothing launched" >&2
  exit 0
fi

uv run ops/run.py bench \
  --image "hf.co/spaces/$SPACE" \
  --flavor l40sx1 --timeout "$TIMEOUT" \
  --bucket "$BUCKET" --out-mount /out \
  --name guided-mmlu-4b-2026-11-13 \
  "$PRE" "$BODY"
