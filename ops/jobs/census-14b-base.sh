#!/usr/bin/env bash
# The 14B DCLM base on a card: arm A of the census, the served smoke and the
# kernel bench, as three stages of one l40sx1 job (step 4 of the card-mode plan),
# and a fourth, the harness control, when the Space moved since census-14b-ref.
#
#   A      MMLU census of the sealed base, dense reconstruction, full split,
#          14,042 questions. The arm the trained file pairs against, and the
#          gate before the h200 of dclm-14b-rowscales.sh is paid.
#   smoke  the served path on the base, through LLVQ_CONFIG: the prefill gate at
#          203 tokens, then 32 decode tokens. The first card run of
#          rot_apply_rows at 17,408 (rot_apply itself ran exact at 17,408 under
#          Planes14 on 2026-08-17 and 2026-08-31, through the shared-memory opt-in).
#   bench  planesbench, ball FIRST, Tetra SECOND, arms fp16,planes14,nullk,tetra48,
#          the served tile.
#   harness  (HARNESS=1, or auto) the 8B base, scored FULL by census-8b on a963a020,
#          scored again at limit=40 on this image: arm A pairs with the reference arms
#          B and C of census-14b-ref.sh, and if the Space was rebuilt in between, that
#          pair crosses images (preregistration-census-14b-ref-2026-09-22.md, "What it
#          will not establish"). ~8 min, ~$0.24 (*estimated*: 253 s of scoring at 8B
#          limit=40, campagne-8b-qualite, + load and sha256). The picks are joined on
#          the Mac against docs/data/mmlu-dumps/mmlu-8b-dclm-FULL.csv, $0.
#          auto: on when the Space is not REF_IMAGE_SHA (a963a020, census-14b-ref's default).
#
# Preregistered: proofs/preregistration-served-14b-2026-09-22.md (PREREG), all four
# stages; census-14b-ref's prereg scores B and C only and leaves arm A to its own. The
# launcher refuses until the .ots exists (DRY_RUN=1 excepted; hard rule 2).
#
# Why one job and not three:
#   - every job of 2026-09-22 queued 1 h 12 to 3 h 06 before it ran (*measured*,
#     jobs.csv): two queues fewer is 2.4 to 6.2 h of calendar, and two container
#     starts and oracles fewer, ~$0.10 each (*estimated*, bench-8b ran 378 s whole);
#   - ~70 min of running together is not long: the 8B census ran 99 min in one job;
#   - each stage runs in its own shell. A failure is written to stages.txt and does
#     not skip the next stage; the job exits 1 at the end if any stage failed;
#   - the order is what the chain needs next: A first (it gates $10-15 of h200), the
#     smoke second (it gates served-14b.sh), the bench and the harness last (they gate
#     nothing). A timeout can only cut those two.
#   Not folded here: arms B and C (they need no object and run earlier, under their
#   own launcher) and the served decode (it runs on the trained file).
#
# Inputs, written by ops/jobs/encode-14b.sh seg2 (its prereg, not this one):
#   /out/dclm-14b-<OBJ_DATE>/qwen3-14b-dclm.bin   sealed base, 6,563,782,117 B (*computed*,
#   preregistration-encode-14b-2026-09-22.md)
#   /out/<SUMS_REMOTE>, default dclm-14b-<OBJ_DATE>/files.sha256: the job's sha256sum
#   lines, exactly one naming qwen3-14b-dclm.bin (hashed on /scratch before the copy).
#   The preflight fetches that file to $LJ/encode-files.sha256 (not $LJ/files.sha256,
#   which `encode-14b.sh fetch` writes), and the job checks the base's bytes and
#   sha256 on the mount before anything reads it. The Mac needs no copy of the base.
#   ball /out/qwen3-14b-c12-3f21abde/qwen3-14b-llvq.bin, the only 14B Planes14 file:
#   6,506,354,741 B, sha256 9df4d475... (rtbits-14b-2026-08-17.txt:15-20).
#
# Image: whatever the Space holds at launch. Its sha goes to $LJ/image.sha, and
# dclm-14b-ft-mmlu.sh and served-14b.sh refuse any other: the trained arm pairs
# against arm A on the same harness. IMAGE_SHA=<sha> makes this launcher refuse a
# Space that moved from that one.
#
# The served config: configs/qwen3-14b-tetra-q5.json (CONF_LOCAL overrides). The
# job writes its bytes into $O from this command line and checks their sha256,
# as served-8b.sh does, so the image need not carry the file.
#
# planesbench.rs:3537,3554 hard-code the 4B's head, 389,070,848 weights; the 14B
# head is 777,912,320. If the log prints "(389 M weights", the two "f16 lm_head"
# lines are struck from the journal, as at 8B (served-8b-2026-09-21.txt); the job
# writes bench-head-note.txt. The table, the ratios against FP16 and the FLOOR
# REMOVED line do not use them. An image rebuilt with the head.rs fix prints 778 M.
#
# ARMS: awq is refused here. Its buffers would put the device at 44.9 of 48.3 GB in
# one phase with the other four (*computed*, the card-mode costing); run it alone.
#
# Cost (l40sx1 $1.80/h), *estimated*:
#   pull 2.5 + oracle 0.5 + sha256 of the base over the mount 2.3        =  5 min
#   A      2,824 s of scoring: the 8B's arm A, 1,786 s, x 1.58, the 14B/8B
#          ratio of the sealed-arm MMLU samples (400/253 s, 2026-08-10 and
#          08-08), + load ~1.5 min                                       = 49 min
#   smoke  Tetra load ~0.7 min (8B 23.0 s x 1.9), gate 203 + 32 tokens   =  4 min
#   bench  sha256 of the ball 2.3 + build 9.0 (8B 284 s x 1.9, the weight
#          ratio) + proofs and 7 rounds 1.5                              = 13 min
#   total ~71 min, $2.13, range [63, 86] min = [$1.89, $2.58]
#   timeout 110m, ceiling $3.30. With the harness stage ~79 min, $2.37, timeout 120m,
#   ceiling $3.60. TILES="unset 128" adds a planesbench at 128, ~11 min.
#
# One writer at a time on the bucket: a second job mounting it dies at the mount,
# unbilled (ops/README.md, mount failure 1). Launch after the job before has ended.
#
# DRY_RUN=1 bash ops/jobs/census-14b-base.sh   prints the job script, parses it, launches nothing.
set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
OBJ_DATE=${OBJ_DATE:-2026-09-22}                  # the encode's date; names the object dirs
D=${RUN_DATE:-$(date -u +%F)}                     # launch date; names the output dir
PREREG=${PREREG:-proofs/preregistration-served-14b-2026-09-22.md}
BUCKET=Pier-Jean/jobs-artifacts
OBJ_DIR=${OBJ_DIR:-dclm-14b-$OBJ_DATE}
SUMS_REMOTE=${SUMS_REMOTE:-$OBJ_DIR/files.sha256}
LJ=${LJ:-$HOME/q14b-dclm-$OBJ_DATE}               # the 14B chain's journal dir on the Mac
CONF_LOCAL=${CONF_LOCAL:-$REPO/configs/qwen3-14b-tetra-q5.json}
ARMS=${ARMS:-fp16,planes14,nullk,tetra48}
TILES=${TILES:-unset}
IMAGE_SHA=${IMAGE_SHA:-}
REF_IMAGE_SHA=${REF_IMAGE_SHA:-a963a02010cec2d3c342ec52dd38f2dded06f2a1}   # census-14b-ref's
HARNESS=${HARNESS:-auto}
DRY=${DRY_RUN:-0}

NAME=qwen3-14b-dclm.bin
F=/out/$OBJ_DIR/$NAME
O=/out/census-14b-base-$D
C=$O/qwen3-14b-tetra-q5.json
FP=a74a6d6213602979                               # census plan fingerprint, every FULL dump
B=/out/qwen3-14b-c12-3f21abde/qwen3-14b-llvq.bin
B_BYTES=6506354741
B_SHA=9df4d475d83698f0fd7ac0cc04dd91c0ab45445a970fca31e5a7c0e4d707334d
H=/out/dclm-8b-2026-09-21/qwen3-8b-dclm.bin       # the harness reference, census-8b arm A
H_BYTES=4364205777
H_SHA=bcea0d5a7de2fbbfd244ecf1793dd74d94808eadc4a1103533d539829f41bc04
case "$HARNESS" in auto|0|1) ;; *) echo "refused: HARNESS=$HARNESS, expected auto, 0 or 1" >&2; exit 1 ;; esac

case ",$ARMS," in *,fp16,*) ;; *) echo "refused: ARMS without fp16 (the witness)" >&2; exit 1 ;; esac
case ",$ARMS," in *,tetra48,*) ;; *) echo "refused: ARMS without tetra48, nothing to measure" >&2; exit 1 ;; esac
case ",$ARMS," in *,awq,*) echo "refused: awq beside the four others is 44.9 of 48.3 GB at 14B; run it alone" >&2; exit 1 ;; esac
for t in $TILES; do
  case "$t" in unset|32|64|128|256) ;; *) echo "refused: tile $t" >&2; exit 1 ;; esac
done
case " $TILES " in *" unset "*) ;; *) echo "refused: TILES without unset, the served tile" >&2; exit 1 ;; esac

cd "$REPO"
test -f "$CONF_LOCAL" || { echo "refused: $CONF_LOCAL missing (CONF_LOCAL=<path> to point at a draft)" >&2; exit 1; }
CONF=$(cat "$CONF_LOCAL")
# The job writes printf '%s\n' "$CONF": the same bytes as the file only if it
# ends in exactly one newline. Checked, not assumed.
CONF_SHA=$(printf '%s\n' "$CONF" | shasum -a 256 | cut -d' ' -f1)
[ "$CONF_SHA" = "$(shasum -a 256 < "$CONF_LOCAL" | cut -d' ' -f1)" ] \
  || { echo "refused: $CONF_LOCAL does not end in exactly one newline" >&2; exit 1; }
python3 -c 'import json,sys; d=json.load(open(sys.argv[1]));
k={"layout","embed","rot_share","fuse","kv","note"}; assert set(d)==k, set(d)^k;
assert (d["layout"],d["embed"],d["rot_share"],d["fuse"],d["kv"])==("tetra48","q8","1","0","f16"), d;
assert "Qwen3-14B" in d["note"] and "40 v_proj" in d["note"], d["note"]' "$CONF_LOCAL"

# ---- preflight on the Mac, $0 -------------------------------------------------------
if [ "$DRY" = 1 ]; then
  SHA=0000000000000000000000000000000000000000000000000000000000000000
  BYTES=6563782117                                # the encode prereg's computed size, dry run only
  [ "$HARNESS" = auto ] && HARNESS=1                # parse the optional stage too
else
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  mkdir -p "$LJ"
  L=$(hf buckets ls "hf://buckets/$BUCKET/$OBJ_DIR/" 2>/dev/null || true)
  printf '%s\n' "$L"
  BYTES=$(printf '%s\n' "$L" | awk -v n="/$NAME" 'substr($NF, length($NF) - length(n) + 1) == n {print $1}')
  [ -n "$BYTES" ] || { echo "refused: no $NAME in $OBJ_DIR/" >&2; exit 1; }
  hf buckets cp "hf://buckets/$BUCKET/$SUMS_REMOTE" "$LJ/encode-files.sha256"
  SHA=$(awk -v n="/$NAME" '{p = "/" $2} substr(p, length(p) - length(n) + 1) == n {print $1}' "$LJ/encode-files.sha256")
  [ "$(printf '%s\n' "$SHA" | grep -c .)" = 1 ] && [ ${#SHA} -eq 64 ] \
    || { echo "refused: want exactly one sha256 for $NAME in $SUMS_REMOTE" >&2; exit 1; }
  RB=$(hf buckets ls "hf://buckets/$BUCKET/qwen3-14b-c12-3f21abde/" 2>/dev/null | awk '$NF ~ /\/qwen3-14b-llvq\.bin$/ {print $1}' || true)
  [ "$RB" = "$B_BYTES" ] || { echo "refused: ball file is ${RB:-absent} B, expected $B_BYTES" >&2; exit 1; }
  if hf buckets ls "hf://buckets/$BUCKET/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already holds files; pick another RUN_DATE" >&2; exit 1
  fi
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    'from huggingface_hub import HfApi; print(HfApi().space_info("Pier-Jean/llvq-runner-cuda").sha)')
  if [ -n "$IMAGE_SHA" ] && [ "$NOW" != "$IMAGE_SHA" ]; then
    echo "refused: the Space is at $NOW, IMAGE_SHA says $IMAGE_SHA" >&2; exit 1
  fi
  if [ "$HARNESS" = auto ]; then
    if [ "$NOW" = "$REF_IMAGE_SHA" ]; then HARNESS=0; else HARNESS=1; fi
  fi
  if [ "$HARNESS" = 1 ]; then
    RH=$(hf buckets ls "hf://buckets/$BUCKET/dclm-8b-2026-09-21/" 2>/dev/null | awk '$NF ~ /\/qwen3-8b-dclm\.bin$/ {print $1}' || true)
    [ "$RH" = "$H_BYTES" ] || { echo "refused: harness file is ${RH:-absent} B, expected $H_BYTES" >&2; exit 1; }
  fi
  printf '%s\n' "$NOW" > "$LJ/image.sha"
  { echo "image  Pier-Jean/llvq-runner-cuda $NOW (census-14b-ref default $REF_IMAGE_SHA; harness stage $HARNESS)"
    echo "base   $F $BYTES B sha256 $SHA (from $SUMS_REMOTE)"
    echo "ball   $B $B_BYTES B sha256 $B_SHA"
    echo "config $CONF_LOCAL sha256 $CONF_SHA"
  } | tee "$LJ/census-base-provenance.txt"
fi

# ---- the job -------------------------------------------------------------------------
# Each stage is a quoted heredoc: nothing in it expands on the Mac. `read -d ''`
# rather than $(cat <<'X'): the Mac's /bin/bash is 3.2, which mis-parses a heredoc
# inside $( ). The stages travel as variables and run in their own `bash`, so a
# failing stage stops itself and not the next one.
IFS= read -r -d '' S_A <<'JOB' || true
check() {
  grep -qxF '# dtype=f16' "$1"
  grep -qxF '# limit=census' "$1"
  grep -qxF '# alloc=flat, 100..1534 per subject, 14042 questions' "$1"
  grep -qxF '# config=none' "$1"
  grep -qxF '# arithmetic=dense reconstruction' "$1"
  grep -qxF '# kv=f16' "$1"
  test "$(tail -n 1 "$1")" = "# end fingerprint=$FP questions=14042"
  echo "dump ok: $1"
}
echo '== arm A: DCLM base 14B, sealed, dense reconstruction, FULL split =='
LLVQ_MMLU_ALLOC=flat LLVQ_MMLU_DUMP="$O/mmlu-14b-dclm-FULL.csv" mmlu "$F" cuda 2>&1 | tee "$O/out-dclm.txt" | tail -10
check "$O/mmlu-14b-dclm-FULL.csv"
JOB

IFS= read -r -d '' S_SMOKE <<'JOB' || true
echo '== the served config, written from the launch command =='
printf '%s\n' "$CONF" > "$C"
sha256sum "$C" | tee -a "$O/files.sha256"
test "$(sha256sum "$C" | cut -d' ' -f1)" = "$CONF_SHA"
cat "$C"
echo '== prefill gate, 203 tokens: 50 chunks of four and one of three (configs/README.md) =='
LLVQ_CONFIG="$C" LLVQ_PREFILL_TOKENS=203 fusedrun "$F" 2>&1 | tee "$O/prefill-203.txt" | tail -14
# The served tile and the 14B object, or nothing further in this stage runs.
grep -qF 'tile 64 (served: measured optimum for sm_89)' "$O/prefill-203.txt"
grep -qF 'Qwen3-14B' "$O/prefill-203.txt"
grep -qF '160 rot_launches/token for 280 projections' "$O/prefill-203.txt"
grep -qF '(0 groups + 240 lone + 40 int4)' "$O/prefill-203.txt"
echo '== the served path, one arm, 32 tokens: the decode runs (no dense arm by design) =='
LLVQ_CONFIG="$C" fusedrun "$F" 32 2>&1 | tee "$O/served-32.txt" | tail -8
JOB

IFS= read -r -d '' S_BENCH <<'JOB' || true
echo '== the ball on the mount: bytes and sha256 =='
test "$(stat -c %s "$B")" = "$B_BYTES"
sha256sum "$B" | tee "$O/ball.sha256"
test "$(cut -d' ' -f1 "$O/ball.sha256")" = "$B_SHA"
for t in $TILES; do
  echo "== tile $t: planesbench, ball FIRST, Tetra SECOND, arms $ARMS ==" ; date -u
  if [ "$t" = unset ]; then
    LLVQ_BENCH_ARMS="$ARMS" planesbench "$B" "$F" 2>&1 | tee "$O/bench-tile-served.txt" | tail -45
    # The served tile, or this was not the served-path measurement.
    grep -qF 'tile 64 (served: measured optimum for sm_89)' "$O/bench-tile-served.txt"
  else
    LLVQ_TILE_BLOCKS="$t" LLVQ_BENCH_ARMS="$ARMS" planesbench "$B" "$F" 2>&1 | tee "$O/bench-tile-$t.txt" | tail -45
  fi
  date -u
done
# The coverage the ratios are read on: 240 of 280 at 14B (216 of 252 at 4B and 8B).
grep -hF 'matched by name' "$O"/bench-tile-*.txt
grep -qF '240 of 280 matrices matched by name' "$O/bench-tile-served.txt"
if grep -qF '(389 M weights' "$O"/bench-tile-*.txt; then
  echo 'the two "f16 lm_head" lines carry the 4B head (planesbench.rs:3537,3554): STRUCK' | tee "$O/bench-head-note.txt"
fi
JOB

IFS= read -r -d '' S_HARNESS <<'JOB' || true
echo '== harness control: the 8B base, scored FULL by census-8b on a963a020, at limit=40 here =='
test "$(stat -c %s "$H")" = "$H_BYTES"
sha256sum "$H" | tee "$O/harness.sha256"
test "$(cut -d' ' -f1 "$O/harness.sha256")" = "$H_SHA"
LLVQ_MMLU_ALLOC=flat LLVQ_MMLU_DUMP="$O/mmlu-8b-dclm-l40-harness.csv" mmlu "$H" cuda 40 2>&1 | tee "$O/out-harness.txt" | tail -6
grep -qxF '# arithmetic=dense reconstruction' "$O/mmlu-8b-dclm-l40-harness.csv"
test "$(tail -n 1 "$O/mmlu-8b-dclm-l40-harness.csv")" = "# end fingerprint=65dcd53655e8bfa5 questions=2280"
JOB
[ "$HARNESS" = 1 ] || S_HARNESS=''

PRE=$(printf 'O=%q\nF=%q\nSHA=%q\nBYTES=%q\nFP=%q\nB=%q\nB_BYTES=%q\nB_SHA=%q\nARMS=%q\nTILES=%q\nC=%q\nCONF=%q\nCONF_SHA=%q\nH=%q\nH_BYTES=%q\nH_SHA=%q\nS_A=%q\nS_SMOKE=%q\nS_BENCH=%q\nS_HARNESS=%q' \
  "$O" "$F" "$SHA" "$BYTES" "$FP" "$B" "$B_BYTES" "$B_SHA" "$ARMS" "$TILES" "$C" "$CONF" "$CONF_SHA" \
  "$H" "$H_BYTES" "$H_SHA" "$S_A" "$S_SMOKE" "$S_BENCH" "$S_HARNESS")

# The head: the refusal of an inherited LLVQ_*, the card, oracle (hard rule 10),
# the base by bytes and sha256; then the three stages in order.
IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
export O F SHA BYTES FP B B_BYTES B_SHA ARMS TILES C CONF CONF_SHA H H_BYTES H_SHA
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader | tee "$O/gpu.txt"
echo '== oracle (hard rule 10) =='
oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle-cuda.txt" | tail -2
grep -q MATCH "$O/oracle-cuda.txt"
echo '== the base on the mount: bytes and sha256 against the encode job =='
test "$(stat -c %s "$F")" = "$BYTES"
sha256sum "$F" | tee "$O/files.sha256"
test "$(cut -d' ' -f1 "$O/files.sha256")" = "$SHA"
FAILED=
stage() {  # $1 name, $2 body, run in its own bash under -euo pipefail
  echo "== stage $1 ==" ; date -u
  if bash -euo pipefail -c "$2"; then echo "$1 ok" | tee -a "$O/stages.txt"
  else echo "$1 FAILED rc=$?" | tee -a "$O/stages.txt"; FAILED="$FAILED $1"; fi
  date -u
}
stage A "$S_A"
stage smoke "$S_SMOKE"
stage bench "$S_BENCH"
[ -z "$S_HARNESS" ] || stage harness "$S_HARNESS"
echo '== raw kept ==' ; ls -la "$O" ; wc -l "$O"/*.txt ; cat "$O/stages.txt"
[ -z "$FAILED" ] || { echo "failed stages:$FAILED" >&2; exit 1; }
JOB

if [ "$DRY" = 1 ]; then
  # What ops/run.py:1091 hands to ["bash", "-lc", ...]: printed and parsed, not run.
  JOBSCRIPT="set -euo pipefail
$PRE
$BODY"
  printf '%s\n' "$JOBSCRIPT"
  printf '%s\n' "$JOBSCRIPT" | bash -n
  # The stages are strings to the parse above; parse each one as the job will run it.
  for s in "$S_A" "$S_SMOKE" "$S_BENCH" "$S_HARNESS"; do printf '%s\n' "$s" | bash -n; done
  echo "DRY_RUN: job script and its stages parse (harness $HARNESS); nothing launched" >&2
  exit 0
fi

if [ "$HARNESS" = 1 ]; then TIMEOUT=120m; else TIMEOUT=110m; fi
uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor l40sx1 --timeout "$TIMEOUT" \
  --bucket "$BUCKET" --out-mount /out \
  --name census-14b-base \
  "$PRE" "$BODY"
