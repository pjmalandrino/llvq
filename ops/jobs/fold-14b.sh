#!/usr/bin/env bash
# 14B paper-2 chain, the Mac stage between the h200 training and the FT jobs:
#   FETCH   sealed base + encode sums + trained sigma.json (+ journal) from the bucket
#   FOLD    all-ones control, then sigma.json folded into $LJ/qwen3-14b-dclm-ft.bin
#   UPLOAD  the FT file to dclm-14b-ft-<OBJ_DATE>/, size checked by `hf buckets ls`
# One stage per invocation, each on an explicit go:
#   bash fold-14b.sh fetch <rowscales-bucket-dir> | fold | upload
#   DRY_RUN=1 bash fold-14b.sh <stage> [dir]   checks what is local, prints the stage, runs nothing
# Every stage is $0: Mac CPU for seconds, and bandwidth. Nothing here launches an HF job.
#
# Why the Mac and not a cpu-xl job: rowscale streams one record at a time and copies
# the tail byte for byte (rowscale.rs:167-172): 0.99 GB and 8.66 s at 8B (*measured*,
# dclm-8b-rowscales-2026-09-21-brut/fold.txt), ~1-2 GB and ~15 s at 14B (*estimated*).
# It needs no card and no image: the image a963a020 carries no rowscale, and a
# rebuilt one is not needed for this. The cost is bandwidth: 6.56 GB down, 6.56 GB up
# (the 8B's 16.4 GB export went up in ~6 min, 4.36 GB FT in minutes; 9.8 MB/s is the
# slowest rate seen, 11 min a way at 14B). The local FT copy is also what
# dclm-14b-ft-mmlu.sh and served-14b.sh check their sha256 against.
#
# Files, all under $LJ = ~/q14b-dclm-<OBJ_DATE>, the directory `encode-14b.sh fetch`
# also uses:
#   qwen3-14b-dclm.bin      the base; `encode-14b.sh fetch` puts it here, fetch reuses it
#                           if its size matches the bucket, and downloads it otherwise
#   encode-files.sha256     the encode job's sums, fetched fresh ($LJ/files.sha256 is
#                           `encode-14b.sh fetch`'s copy and is left alone)
#   ft.sha256               written here: the base's line (checked) and the FT file's;
#                           dclm-14b-ft-mmlu.sh and served-14b.sh read it
#   qwen3-14b-dclm-ft.bin   the folded file
#   ft/                     sigma.json, the training's journal and probe, the fold logs
#
# The fold binary is the 8B's, not a new build: ~/q8b-dclm-2026-09-21/bin-ft/rowscale,
# sha256 d25136b9... (dclm-8b-rowscales-2026-09-21-brut/bin-ft.sha256), built at
# 5d36d52. rowscale.rs and the format crates have not changed since 1190736, and the
# same bytes fold both sizes. BIN= and ROWSCALE_SHA= override, for a fresh build.
#
# Expected at 14B (*computed* from Qwen/Qwen3-14B's config at 40c06982):
#   240 Tetra matrices (6 x 40 layers), 40 int4 v_proj passed through;
#   2,048,000 row scales = 40 x (5120 q + 1024 k + 5120 o + 17408 gate + 17408 up + 5120 down);
#   the tail after the last record >= 3,123,922,582 B = embed + lm_head f16
#   3,111,649,280 + norms 849,920 + tokenizer 11,422,654 + config 728 (framing extra),
#   so every byte the fold changes sits before SIZE - TAIL.
# Memory ~1-2 GB; disk ~20 GB at peak (base, ones control, FT), ~13 GB after.
set -euo pipefail

OBJ_DATE=${OBJ_DATE:-2026-09-22}
HERE=$(cd "$(dirname "$0")" && pwd)
LJ=${LJ:-$HOME/q14b-dclm-$OBJ_DATE}                 # the 14B chain's journal dir
BK=hf://buckets/Pier-Jean/jobs-artifacts
OBJ_DIR=${OBJ_DIR:-dclm-14b-$OBJ_DATE}
SUMS_REMOTE=${SUMS_REMOTE:-$OBJ_DIR/files.sha256}
FT_DIR=${FT_DIR:-dclm-14b-ft-$OBJ_DATE}
BASE=${BASE:-$LJ/qwen3-14b-dclm.bin}
FT=${FT:-$LJ/qwen3-14b-dclm-ft.bin}
SIG=${SIG:-$LJ/ft/sigma.json}
SUMS=$LJ/ft.sha256
BIN=${BIN:-$HOME/q8b-dclm-2026-09-21/bin-ft/rowscale}
ROWSCALE_SHA=${ROWSCALE_SHA:-d25136b97f21b17ac5025422afd8577639825195aac6a93985c035d987679b68}
CHECK=${CHECK:-$HERE/export-sigma-check.py}
WANT_M=240
WANT_ROWS=2048000
TAIL=3123922582
DRY=${DRY_RUN:-0}

STAGE=${1:-}
case "$STAGE" in fetch|fold|upload) ;; *) sed -n '2,8p' "$0"; exit 2 ;; esac
if [ "$STAGE" = fetch ]; then R=$BK/${2:?the training job output dir in the bucket, e.g. dclm-14b-rowscales-2026-09-22}; fi

# What every stage needs locally: the pinned binary and the sigma checker.
test -x "$BIN" || { echo "refused: $BIN missing (BIN=<rowscale> to point at a fresh build)" >&2; exit 1; }
[ "$(shasum -a 256 "$BIN" | cut -d' ' -f1)" = "$ROWSCALE_SHA" ] || { echo "refused: $BIN is not the pinned rowscale" >&2; exit 1; }
test -f "$CHECK" || { echo "refused: $CHECK missing" >&2; exit 1; }
FREE=$(df -k "$HOME" | awk 'NR==2 {print int($4/1e6)}')   # GB, roughly
echo "rowscale $BIN (pinned); checker $CHECK; ~$FREE GB free under \$HOME"

stage_fetch() {
  mkdir -p "$LJ/ft"
  # The base: from the bucket, then its sha256 against the encode job's line.
  hf buckets ls "$BK/$OBJ_DIR/" | tee "$LJ/ft/base-bucket-ls.txt"
  RB=$(awk '$NF ~ /\/qwen3-14b-dclm\.bin$/ {print $1}' "$LJ/ft/base-bucket-ls.txt")
  [ -n "$RB" ] || { echo "no qwen3-14b-dclm.bin in $OBJ_DIR/"; exit 1; }
  if [ ! -f "$BASE" ] || [ "$(stat -f %z "$BASE")" != "$RB" ]; then
    [ "$FREE" -ge 25 ] || { echo "under 25 GB free; the fold needs ~20"; exit 1; }
    hf buckets cp "$BK/$OBJ_DIR/qwen3-14b-dclm.bin" "$BASE"
  fi
  test "$(stat -f %z "$BASE")" = "$RB"
  hf buckets cp "$BK/$SUMS_REMOTE" "$LJ/encode-files.sha256"
  WANT=$(awk '{p = "/" $2} substr(p, length(p) - 18) == "/qwen3-14b-dclm.bin" {print $1}' "$LJ/encode-files.sha256")
  GOT=$(shasum -a 256 "$BASE" | cut -d' ' -f1)
  [ ${#WANT} -eq 64 ] && [ "$GOT" = "$WANT" ] || { echo "base sha256 $GOT, the encode job wrote ${WANT:-nothing}"; exit 1; }
  # $SUMS is what the FT and served launchers read: the base first.
  grep -vF "  $BASE" "$SUMS" 2>/dev/null > "$SUMS.tmp" || true
  echo "$GOT  $BASE" >> "$SUMS.tmp"; mv "$SUMS.tmp" "$SUMS"
  echo "base: $RB B, sha256 $GOT, as the encode job wrote"
  # The training's outputs. `hf buckets ls` sizes are authoritative.
  hf buckets ls "$R/" | tee "$LJ/ft/sigma-bucket-ls.txt"
  hf buckets cp "$R/sigma.json" "$SIG"
  for f in journal.jsonl probe.jsonl; do hf buckets cp "$R/$f" "$LJ/ft/$f"; done
  test "$(awk '$NF ~ /\/sigma\.json$/ {print $1}' "$LJ/ft/sigma-bucket-ls.txt")" = "$(stat -f %z "$SIG")"
  shasum -a 256 "$SIG" | tee "$LJ/ft/sigma.sha256"
}

stage_fold() {
  test -s "$SIG" && test -f "$BASE" || { echo "run fetch first"; exit 1; }
  mkdir -p "$LJ/ft"
  [ "$FREE" -ge 14 ] || { echo "under 14 GB free: the ones control and the FT file need 13"; exit 1; }
  python3 "$CHECK" "$SIG" "$WANT_M" "$WANT_ROWS" | tee "$LJ/ft/sigma-summary.txt"
  # The base must be the file the export (and so sigma) was made from.
  grep -F "  $BASE" "$SUMS" | shasum -a 256 -c -
  # Control, free (~15 s + 6.6 GB of disk): all-ones must reproduce the base byte for byte.
  python3 "$CHECK" ones "$SIG" "$LJ/ft/sigma-ones.json"
  "$BIN" "$BASE" "$LJ/ft/ones.bin" "$LJ/ft/sigma-ones.json" | tee "$LJ/ft/fold-ones.txt"
  grep -q "^0 scaled, $WANT_M untouched, 40 int4 passed through" "$LJ/ft/fold-ones.txt"
  cmp "$BASE" "$LJ/ft/ones.bin"                   # exits the script on any difference
  echo "ones control: BYTE-IDENTICAL" | tee -a "$LJ/ft/fold-ones.txt"
  rm -f "$LJ/ft/ones.bin"
  # The fold. 8B: 8.66 s, 0.99 GB.
  /usr/bin/time -l "$BIN" "$BASE" "$FT" "$SIG" 2>&1 | tee "$LJ/ft/fold.txt"
  grep -q "^$WANT_M scaled, 0 untouched, 40 int4 passed through" "$LJ/ft/fold.txt"
  grep -q "^$WANT_ROWS row scales read" "$LJ/ft/fold.txt"
  test "$(stat -f %z "$FT")" -eq "$(stat -f %z "$BASE")"
  # Only row_scales move, so every differing byte sits in the records region.
  SIZE=$(stat -f %z "$BASE")
  { cmp -l "$BASE" "$FT" || [ $? -eq 1 ]; } | awk -v lim=$((SIZE - TAIL)) -v cap=$((8 * WANT_ROWS)) \
      '{n++; if ($1>m) m=$1} END {printf "%d bytes differ, last at %d, records end before %d\n", n, m, lim;
       exit (n>0 && n<=cap && m<lim)?0:1}' | tee -a "$LJ/ft/fold.txt"
  GOT=$(shasum -a 256 "$FT" | cut -d' ' -f1)
  grep -vF "  $FT" "$SUMS" > "$SUMS.tmp" || true
  echo "$GOT  $FT" >> "$SUMS.tmp"; mv "$SUMS.tmp" "$SUMS"
  cat "$SUMS"
}

stage_upload() {
  test -f "$FT" && grep -qF "  $FT" "$SUMS" || { echo "run fold first"; exit 1; }
  grep -F "  $FT" "$SUMS" | shasum -a 256 -c -
  if hf buckets ls "$BK/$FT_DIR/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "$FT_DIR/ already holds files; FT_DIR=<another> or look first"; exit 1
  fi
  hf buckets cp "$FT" "$BK/$FT_DIR/qwen3-14b-dclm-ft.bin"
  hf buckets ls "$BK/$FT_DIR/" | tee "$LJ/ft/ft-bucket-ls.txt"
  UP=$(awk '$NF ~ /\/qwen3-14b-dclm-ft\.bin$/ {print $1}' "$LJ/ft/ft-bucket-ls.txt")
  [ "$UP" = "$(stat -f %z "$FT")" ] || { echo "the bucket copy is ${UP:-absent} B, local $(stat -f %z "$FT") B"; exit 1; }
  echo "FT for dclm-14b-ft-mmlu.sh and served-14b.sh: FT_DIR=$FT_DIR"
}

if [ "$DRY" = 1 ]; then
  declare -f "stage_$STAGE"
  echo "DRY_RUN: stage $STAGE checked and printed; nothing fetched, folded or uploaded" >&2
  exit 0
fi
"stage_$STAGE"
