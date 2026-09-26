#!/usr/bin/env bash
# DRAFT, not launched. The 14B paper-2 base object, encoded ON A CARD, in two segments.
# The 4B and 8B were encoded on the Mac. The operator's go for the 14B is "allé lance moi le
# 14B"; card mode and the $52 cap for the whole chain are the card plan's proposals, which the
# prereg records verbatim before stamping. The device change is a declared confound, not a
# detail: see the prereg, "The card, declared first".
#
# Preregistered: proofs/preregistration-encode-14b-2026-09-22.md (both segments, one prereg).
#   Hard rule 2. seg1 and seg2 refuse to launch until the .ots exists (DRY_RUN=1 excepted).
#
# The object: Qwen/Qwen3-14B (rev 40c06982), `tetra1`, `v_proj` in int4 g128
# (LLVQ_INT4_TYPES=v_proj), dclm-edu (rev dbad8ad7) 64 x 2048 from the prefix, rotation on
# (seed 0x110feed), nogs, damping 1e-2, f32, eval wikitext-2 12 x 4096. The 8B recipe
# (preregistration-dclm-8b-2026-09-21.md, run.sh) with two changes: device cuda instead of
# metal, 23 encoder threads instead of 12.
#
#   seg1  one job, rtx-pro-6000 x1: oracle 0.6B (hard rule 10); the GATE, the exact recipe on
#         Qwen3-0.6B, block 0 then resumed to block 1 (tetra1, int4 records and LLVQ_RESUME
#         have never run on a card; a failure here costs ~8 min, $0.37, not a segment);
#         then Qwen3-14B blocks 0..19 into /out/dclm-14b-seg1-$D/.
#   seg2  one job, same flavor, same image: oracle; the shard copied to /scratch and checked;
#         LLVQ_RESUME, blocks 20..39, the whole .llvq on /out (smoke verifies it bit for bit);
#         seal to /scratch, copied to /out, size and sha256 read back; ppl f16 of the sealed
#         file on cuda, gated at 1 % of the encoding perplexity; export to /scratch, copied to
#         /out/dclm-14b-export-$D/ with sizes and sha256; rtbits only if the image has it.
#   check $0, Mac: every byte count in the bucket against the prereg (hf buckets ls is the
#         authority, ops/README.md "Bucket mounting fails silently").
#   fetch $0, Mac: the sealed file to $HOME, sha256 against the job's, rtbits (not in the image).
#
# Why two segments (check of the card plan, 2026-09-22): the longest card job ever billed is
# 302 min (jobs.csv:27), a one-piece x1 encode is ~430 min, and no encode was ever resumed
# after a cut inside a job. Two segments stay inside the measured envelope; the resume path is
# the one tests/resume.rs proves byte-identical on CPU. On cuda the hidden states are
# recomputed, not restored, so the two-segment file is not GUARANTEED to be the one-piece file
# (smoke.rs:62-70); the result line says so, and the prereg declares it.
#
# How `blocks` works (smoke.rs:55-66, 1063-1066): an ABSOLUTE bound, not a count. seg1 passes
# 20: n_target = 20, the header announces 140 records, the writer is finished and flushed
# normally (FileSink::finish, format.rs finish()), verify_artifact reads the 140 back, and the
# .state beside it says blocks_done = 20. seg2 passes 40: the restart block is read off the
# shard, never passed. seg2's output must differ from its input (smoke.rs:801-806).
#
# Where files live. The .llvq of each segment is written straight to /out, as every card encode
# did (v1c, dclm-x32, the 14B c12: 3,382,420,994 B written for 5 h on the mount, intact): if the
# job is cut, a partial shard with its .state survives and the resume reads it (smoke.rs:937-966).
# Sealed file and export go to /scratch first and are COPIED: the mount has truncated a 4 GB
# write with no error (ops/README.md, 2026-08-10). No file is mmapped from /out: llvq-artifact
# reads through BufReader (sealed.rs:495, seal.rs, export.rs), and the checkpoint smoke mmaps
# sits in the hf-hub cache on the container's disk, `$HOME/.cache/huggingface/hub` (hf-hub
# ignores HF_HOME, see revision_check below), never on /out: the SIGBUS of 2026-08-10 cannot
# recur here.
#
# The image. a963a020 (afaed1e) has no `export` and no `rowscale`: both launches refuse it by
# name, and every job checks `type -P` for its binaries before it pays for anything. Both
# segments must run on ONE image (IMAGE_SHA, checked against the live Space and, for seg2,
# against what seg1 recorded): a resume across two builds is two encoders. `rtbits` is in
# llvq-bench, which ops/Dockerfile.cuda does not build: `fetch` runs it on the Mac.
#
# Cost, estimated from the card plan (tasks/wghg5aptb, check 1): the loop is 381.7 min for 40
# blocks on x1 (x2 CPU 181.7 min x 2 + ~1,100 s of capture, transfer and write), 190.9 min a
# segment, ~573 s a block; range [154, 226] min a segment (the check's [353, 497]-min job
# less its 45 fixed minutes). rtx-pro-6000 at $2.75/h (`hf jobs hardware`, 2026-09-22).
#   seg1  pull+start 3, oracle 1, gate 4, 14B download 29.5 GB + f32 load + baseline ppl 12,
#         loop 191, verify + partial ppl + sha256 5            = 216 min [179, 251]
#         $9.90 [$8.20, $11.50]; timeout 255m, ceiling $11.69
#   seg2  start+oracle 4, download+load+baseline 12, shard copy+sha 2, reload 4, replay of
#         20 blocks 3, loop 191, verify+ppl+sha 7, seal+copy+read-back 8, sealed ppl 3,
#         export 6, export copy+sha 10                         = 250 min [213, 285]
#         $11.46 [$9.76, $13.06]; timeout 285m, ceiling $13.06
#   total 466 min, $21.36 central [$17.97, $24.57], ceiling $24.75 (= the one-job 540m of the
#   plan). The platform has billed +18 and +28 min past a timeout (jobs.csv): +$1.28 a job.
#   Caveat: seg2's range touches its timeout. The order is chosen for that: within the range, a
#   cut can only land in the export or its copy (the last 16 min); the .llvq, the sealed file
#   and its perplexity are on /out before, and the export alone is a small rerun. A loop more
#   than 16 min over the top of its range cuts earlier. seg1's range tops at 251 of its 255
#   min, 4 min of margin: a cut there leaves a partial shard and its .state on /out but no
#   DONE, and both launches refuse it (resuming it is a new go).
#
# Predicted bytes, computed from the writer layout (format.rs put_record_head / write_codes,
# sealed.rs write_raw / write_blob; the same arithmetic returns the 4B .llvq 1,004,826,858 B,
# the 8B .llvq 1,862,836,458 B and the 8B sealed 4,364,205,777 B to the byte):
#   seg1 shard 1,719,924,170 B; .llvq 3,439,848,370 B; sealed 6,563,782,117 B (163 carried
#   tensors, 1,556,249,600 weights); export model.safetensors 29,536,665,800 B (443 tensors,
#   header 51,392 B), config.json 728, tokenizer.json 11,422,654, tokenizer_config.json 64.
#
# Usage (from any checkout; REPO is this script's own):
#   DRY_RUN=1 bash ops/jobs/encode-14b.sh seg1|seg2   prints the job script, parses it, launches nothing
#   IMAGE_SHA=<rebuilt Space sha> bash ops/jobs/encode-14b.sh seg1
#   IMAGE_SHA=<same sha>          bash ops/jobs/encode-14b.sh seg2
#   bash ops/jobs/encode-14b.sh check | fetch
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
STAGE=${1:-}
D=${OBJ_DATE:-2026-09-22}                          # the object's encoding date: names every dir
PREREG=${PREREG:-proofs/preregistration-encode-14b-2026-09-22.md}
BUCKET=Pier-Jean/jobs-artifacts
FLAVOR=rtx-pro-6000
THREADS=23                                         # the flavor's vCPU (ops/run.py:107)
MODEL=Qwen/Qwen3-14B
MODEL_SHA=40c069824f4251a91eefaf281ebe4c544efd3e18
DCLM_SHA=dbad8ad71224482740cd9c9d353591adbf62fe04
OLD_IMAGE=a963a02010cec2d3c342ec52dd38f2dded06f2a1  # afaed1e: no export, no rowscale
LOG=${LOG:-$HOME/q14b-dclm-$D}                     # Mac-side provenance and fetched files
DRY=${DRY_RUN:-0}

S1_DIR=dclm-14b-seg1-$D
OBJ_DIR=dclm-14b-$D
EXP_DIR=dclm-14b-export-$D
S1_NAME=qwen3-14b-dclm-seg1.llvq
A_NAME=qwen3-14b-dclm.llvq
SB_NAME=qwen3-14b-dclm.bin

# Computed (header above). Used by `check` and by seg2's preflight on the shard; the jobs
# themselves gate on source-against-copy equality, never on a prediction.
SEG1_BYTES=${SEG1_BYTES:-1719924170}
LLVQ_BYTES=3439848370
SEALED_BYTES=6563782117
ST_BYTES=29536665800
R_STOP=1.4502          # the prereg's two-draw bound: seg1's partial ratio above it stops seg2

refuse() { echo "refused: $*" >&2; exit 1; }
cd "$REPO"

# `hf buckets ls` prints `(empty)` and exits 0 for a directory that does not exist (the false
# positive of served-8b ECARTS E2); a failed listing refuses rather than reading as empty. Call
# it as an assignment, `L=$(bucket_ls d)`, so that set -e sees the failure.
bucket_ls() {
  local out
  out=$(hf buckets ls "hf://buckets/$BUCKET/$1/" 2>&1) || { echo "refused: hf buckets ls $1/ failed: $out" >&2; return 1; }
  printf '%s\n' "$out" | grep -v '^(empty)$' || true
}
size_of() { awk -v n="$2" '$NF ~ ("/" n "$") {print $1}' <<<"$1"; }

hub_revisions() {
  uv run --quiet --with huggingface_hub python -c '
from huggingface_hub import HfApi
a = HfApi()
print("image", a.space_info("Pier-Jean/llvq-runner-cuda").sha)
print("model", a.model_info("Qwen/Qwen3-14B").sha)
print("dclm", a.dataset_info("HuggingFaceTB/dclm-edu").sha)
print("gate", a.model_info("Qwen/Qwen3-0.6B").sha)
'
}

# ---- the Mac-only stages, $0 -----------------------------------------------------------
case "$STAGE" in
check)
  # Byte counts as the bucket reports them, against the prereg. Exit 1 on a missing file or on
  # a deviation above 0.01 %; equality is printed, not assumed.
  bad=0
  row() {  # dir name predicted
    local L got
    L=$(bucket_ls "$1")
    got=$(size_of "$L" "$2")
    if [ -z "$got" ]; then echo "MISSING  $1/$2"; bad=1; return; fi
    awk -v g="$got" -v p="$3" -v f="$1/$2" 'BEGIN {
      d = (g - p) / p; printf "%-8s %s  %d B, predicted %d B, %+.6f %%\n",
      (g == p ? "EXACT" : (d < 1e-4 && d > -1e-4 ? "within" : "OUTSIDE")), f, g, p, 100 * d
      exit (d < 1e-4 && d > -1e-4) ? 0 : 1 }' || bad=1
  }
  row "$S1_DIR" "$S1_NAME" "$SEG1_BYTES"
  row "$OBJ_DIR" "$A_NAME" "$LLVQ_BYTES"
  row "$OBJ_DIR" "$SB_NAME" "$SEALED_BYTES"
  row "$EXP_DIR" model.safetensors "$ST_BYTES"
  row "$EXP_DIR" config.json 728
  row "$EXP_DIR" tokenizer.json 11422654
  row "$EXP_DIR" tokenizer_config.json 64
  exit "$bad"
  ;;
fetch)
  # The sealed file to the Mac (the fold needs it anyway), its sha256 against the job's, and
  # rtbits, which the image does not carry. 6.56 GB of disk.
  [ "$DRY" = 1 ] && { echo "DRY_RUN: would fetch $OBJ_DIR/$SB_NAME to $LOG and run rtbits"; exit 0; }
  mkdir -p "$LOG"
  hf buckets cp "hf://buckets/$BUCKET/$OBJ_DIR/files.sha256" "$LOG/files.sha256"
  hf buckets cp "hf://buckets/$BUCKET/$OBJ_DIR/$SB_NAME" "$LOG/$SB_NAME"
  # seg2 hashes from inside the dir (`cd "$W" && sha256sum ...`): the names are bare, no `/`.
  WANT=$(awk -v n="/$SB_NAME" '{p = "/" $2} substr(p, length(p) - length(n) + 1) == n {print $1}' "$LOG/files.sha256" | tail -1)
  [ ${#WANT} -eq 64 ] || refuse "no sha256 for $SB_NAME in the job's files.sha256"
  [ "$(shasum -a 256 "$LOG/$SB_NAME" | cut -d' ' -f1)" = "$WANT" ] || refuse "the fetched file differs from the job's"
  echo "fetched $LOG/$SB_NAME, sha256 $WANT"
  cargo run --release -p llvq-bench --bin rtbits -- "$LOG/$SB_NAME" 2>&1 | tee "$LOG/rtbits.txt"
  exit 0
  ;;
seg1|seg2) ;;
*) sed -n '2,40p' "$0"; exit 2 ;;
esac

# ---- preflight on the Mac, $0 ----------------------------------------------------------
if [ "$DRY" = 1 ]; then
  IMAGE_SHA=${IMAGE_SHA:-DRYRUN}
else
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || refuse "$PREREG(.ots) missing"
  [ -n "${IMAGE_SHA:-}" ] || refuse "IMAGE_SHA=<the rebuilt Space sha> is required: both segments run on one image"
  [ "$IMAGE_SHA" != "$OLD_IMAGE" ] || refuse "IMAGE_SHA is a963a020 (afaed1e), which has no export: rebuild the image first"
  REV=$(hub_revisions)
  printf '%s\n' "$REV"
  [ "$(awk '$1=="image" {print $2}' <<<"$REV")" = "$IMAGE_SHA" ] || refuse "the Space is not at IMAGE_SHA=$IMAGE_SHA"
  [ "$(awk '$1=="model" {print $2}' <<<"$REV")" = "$MODEL_SHA" ] || refuse "$MODEL main moved off $MODEL_SHA"
  [ "$(awk '$1=="dclm" {print $2}' <<<"$REV")" = "$DCLM_SHA" ] || refuse "dclm-edu main moved off $DCLM_SHA"
  mkdir -p "$LOG"
  if [ "$STAGE" = seg1 ]; then
    for d in "$S1_DIR" "$OBJ_DIR" "$EXP_DIR"; do
      L=$(bucket_ls "$d")
      [ -z "$L" ] || refuse "$d/ already holds files; pick another OBJ_DATE"
    done
  else
    P1="$LOG/provenance-seg1.txt"
    test -s "$P1" || refuse "$P1 missing: seg2 checks its image against seg1's"
    [ "$(awk '$1=="image" {print $2}' "$P1")" = "$IMAGE_SHA" ] || refuse "seg1 ran on another image ($P1)"
    L1=$(bucket_ls "$S1_DIR")
    printf '%s\n' "$L1"
    [ "$(size_of "$L1" "$S1_NAME")" = "$SEG1_BYTES" ] \
      || refuse "the shard is $(size_of "$L1" "$S1_NAME") B, want $SEG1_BYTES (a cut seg1 is an operator decision: SEG1_BYTES=<n>)"
    [ -n "$(size_of "$L1" "$S1_NAME.state")" ] || refuse "$S1_NAME.state missing"
    [ -n "$(size_of "$L1" DONE)" ] || refuse "seg1 wrote no DONE marker: it did not reach its end"
    hf buckets cp "hf://buckets/$BUCKET/$S1_DIR/smoke.txt" "$LOG/smoke-seg1.txt"
    R1=$(awk '/^exact-ppl/ {printf "%.4f", $5 / $3}' "$LOG/smoke-seg1.txt")
    [ -n "$R1" ] || refuse "no exact-ppl line in seg1's smoke.txt"
    echo "seg1 partial ratio (blocks 0..19 quantized, 20..39 dense): x$R1"
    if awk -v r="$R1" -v s="$R_STOP" 'BEGIN {exit !(r > s)}'; then
      [ "${FORCE_SEG2:-0}" = 1 ] || refuse "partial ratio x$R1 > x$R_STOP, the prereg's stop signal; the operator decides (FORCE_SEG2=1)"
    fi
    for d in "$OBJ_DIR" "$EXP_DIR"; do
      L=$(bucket_ls "$d")
      [ -z "$L" ] || refuse "$d/ already holds files"
    done
  fi
  { printf '%s\n' "$REV"
    echo "stage $STAGE $(date -u +%FT%TZ)"
    echo "launcher $(shasum -a 256 "$0" | cut -d' ' -f1)"
    echo "prereg $(shasum -a 256 "$PREREG" | cut -d' ' -f1)"
    echo "head $(git rev-parse HEAD)"; } | tee "$LOG/provenance-$STAGE.txt"
fi

# ---- the job ----------------------------------------------------------------------------
PRE=$(printf 'THREADS=%q\nMODEL=%q\nMODEL_SHA=%q\nDCLM_SHA=%q\nS1=%q\nO=%q\nEO=%q\nS1_NAME=%q\nA_NAME=%q\nSB_NAME=%q' \
  "$THREADS" "$MODEL" "$MODEL_SHA" "$DCLM_SHA" "/out/$S1_DIR" "/out/$OBJ_DIR" "/out/$EXP_DIR" \
  "$S1_NAME" "$A_NAME" "$SB_NAME")

# Common head. Quoted heredoc: nothing expands on the Mac. `read -d ''` rather than
# $(cat <<'JOB'): the Mac's /bin/bash is 3.2. ops/run.py prepends `set -euo pipefail` and runs
# the whole as ['bash', '-lc', script] (ops/run.py:1088-1100).
IFS= read -r -d '' HEAD <<'JOB' || true
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
# `type -P`, not `command -v`: `export` is a bash builtin, so `command -v export` always
# succeeds and a bare `export a b` runs the builtin. The binary is called by its path.
for b in oracle smoke seal ppl export; do
  type -P "$b" >/dev/null || { echo "refused: $b is not in this image" >&2; exit 1; }
done
EXPORT_BIN=$(type -P export)
mkdir -p "$W"
{ nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader
  echo "nproc $(nproc)"; grep -m1 'model name' /proc/cpuinfo || true; grep '^MemTotal:' /proc/meminfo || true
  df -h /scratch "$HOME" 2>&1 | tail -n +2 || true; } | tee "$W/gpu.txt"   # $HOME: the hf-hub cache
# Memory, every 30 s: GPU used/total MiB and utilization, host MemAvailable, and the resident
# set and high-water mark of whichever of our binaries runs. /proc, not ps: procps is not in
# the runtime image. A failed sample is a blank field, never a dead job.
mem_sampler() {
  echo 'utc,gpu_used_mib,gpu_total_mib,gpu_util_pct,host_avail_kb,proc,vmrss_kb,vmhwm_kb'
  while :; do
    g=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ') || g=',,'
    [ -n "$g" ] || g=',,'
    a=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null) || a=
    p=; c=; r=; h=
    for d in /proc/[0-9]*; do
      n=$(cat "$d/comm" 2>/dev/null) || continue
      case "$n" in smoke|seal|ppl|export|oracle) p=${d#/proc/}; c=$n; break ;; esac
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
peaks() {  # the sampled peaks, for the journal; the prereg signs the GPU one
  awk -F, 'NR > 1 && $2 != "" {if ($2 + 0 > g) g = $2 + 0}
           NR > 1 && $8 != "" {if ($8 + 0 > h[$6]) h[$6] = $8 + 0}
           END {printf "GPU peak %d MiB sampled\n", g; for (k in h) printf "  %s VmHWM %.1f GB\n", k, h[k] / 1e6}' "$W/mem.csv" | tee "$W/peaks.txt"
}
# A gate that fails names the line it wanted: `need <file> <fixed string>`, `needx` for a
# whole line, `never` for a string that must be absent.
need()  { grep -qF -- "$2" "$1" || { echo "refused: $1 lacks: $2" >&2; exit 1; }; }
needx() { grep -qxF -- "$2" "$1" || { echo "refused: $1 lacks the line: $2" >&2; exit 1; }; }
never() { if grep -qF -- "$2" "$1"; then echo "refused: $1 holds: $2" >&2; exit 1; fi; }
# The resolved configuration a log must print, and what it must not (smoke.rs:840-905).
recipe_ok() {
  need "$1" 'codebook     tetra1 → Tetra word map'
  need "$1" 'calibration  dclm-edu, 64 × 2048 = 131072 tokens requested, contiguous prefix from token 0'
  need "$1" "revision $DCLM_SHA"
  need "$1" 'rotation     on (seed 0x110feed)'
  need "$1" '  mode         nogs (group scales off, design C off'
  needx "$1" '  dtype        f32'
  need "$1" ", $THREADS encoder threads"
  need "$1" '  damping      1e-2 (relative to mean(diag H))'
  need "$1" '  h_shrink     1 (H as is, published path)'
  need "$1" '  gain_scale   1 (centroids as fitted, published path)'
  need "$1" 'weights identical, bit for bit (at f32)'
  never "$1" 'spherical feedback on'
  never "$1" 'WITHOUT `fast-linalg`'
  echo "recipe ok: $1"
}
ratio() { awk '/^exact-ppl/ {printf "%.4f", $5 / $3}' "$1"; }
# The checkpoint revision smoke resolved: `refs/main` of the hf-hub cache, NOT under HF_HOME.
# `hf_hub::api::sync::Api::new()` (loader.rs from_hub) builds `Cache::default()`, that is
# `$HOME/.cache/huggingface/hub`; only `ApiBuilder::from_env()` reads HF_HOME (hf-hub 0.4.3,
# api/sync.rs:229-245, lib.rs:196-203). HF_HOME=/scratch/hf serves the Python tools. Both
# places are read. A ref found and not MODEL_SHA refuses; none found is recorded as such, and
# the Mac's check before each launch is then the only guard.
revision_check() {  # $1 the record file
  local c f
  for c in "$HOME/.cache/huggingface/hub" "${HF_HOME:-/scratch/hf}/hub"; do
    f="$c/models--Qwen--Qwen3-14B/refs/main"
    if [ -s "$f" ]; then echo "refs/main $(tr -d ' \n' < "$f") $f"; fi
  done > "$1"
  [ -s "$1" ] || echo 'no refs/main under $HOME/.cache/huggingface/hub nor $HF_HOME/hub' > "$1"
  cat "$1"
  if awk -v s="$MODEL_SHA" '$1 == "refs/main" && $2 != s {bad = 1} END {exit !bad}' "$1"; then
    echo "refused: the checkpoint resolved is not $MODEL_SHA ($1)" >&2; exit 1
  fi
}
echo '== oracle (hard rule 10) =='
oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$W/oracle-cuda.txt" | tail -2
need "$W/oracle-cuda.txt" MATCH
JOB

IFS= read -r -d '' SEG1 <<'JOB' || true
echo '== gate: the exact recipe on Qwen3-0.6B, block 0, then resumed to block 1 =='
G=/scratch/gate; mkdir -p "$G"
date -u
env LLVQ_MODEL=Qwen/Qwen3-0.6B LLVQ_CALIB=dclm-edu LLVQ_INT4_TYPES=v_proj LLVQ_THREADS="$THREADS" \
    LLVQ_ARTIFACT="$G/a.llvq" \
  smoke 64 2048 12 4096 cuda nogs tetra1 1 rot 2>&1 | tee "$W/gate-a.txt" | tail -14
env LLVQ_MODEL=Qwen/Qwen3-0.6B LLVQ_CALIB=dclm-edu LLVQ_INT4_TYPES=v_proj LLVQ_THREADS="$THREADS" \
    LLVQ_RESUME="$G/a.llvq" LLVQ_ARTIFACT="$G/b.llvq" \
  smoke 64 2048 12 4096 cuda nogs tetra1 2 rot 2>&1 | tee "$W/gate-b.txt" | tail -14
date -u
recipe_ok "$W/gate-a.txt"
recipe_ok "$W/gate-b.txt"
need "$W/gate-a.txt" 'verifying 7 matrices against the evaluated model (v5, kinds Tetra+Int4G128)'
need "$W/gate-a.txt" 'and 1 matrices, 1048576 weights in int4 g128'
need "$W/gate-b.txt" 'of which 7 matrices resumed from the shard'
need "$W/gate-b.txt" 'verifying 14 matrices against the evaluated model (v5, kinds Tetra+Int4G128)'
sha256sum "$G/a.llvq" "$G/b.llvq" | tee "$W/gate.sha256"
echo "gate ratios: block 0 x$(ratio "$W/gate-a.txt"), blocks 0..1 x$(ratio "$W/gate-b.txt")"

echo '== Qwen3-14B, blocks 0..19, the shard straight to /out =='
date -u
env LLVQ_MODEL="$MODEL" LLVQ_CALIB=dclm-edu LLVQ_INT4_TYPES=v_proj LLVQ_THREADS="$THREADS" \
    LLVQ_ARTIFACT="$W/$S1_NAME" \
  smoke 64 2048 12 4096 cuda nogs tetra1 20 rot 2>&1 | tee "$W/smoke.txt"
date -u
recipe_ok "$W/smoke.txt"
need "$W/smoke.txt" '  blocks       20 at most'
need "$W/smoke.txt" "quantizing blocks 0..19 of 40 of $MODEL"
need "$W/smoke.txt" 'verifying 140 matrices against the evaluated model (v5, kinds Tetra+Int4G128)'
need "$W/smoke.txt" '✓ 6606028800 weights identical, bit for bit (at f32)'
need "$W/smoke.txt" 'and 20 matrices, 104857600 weights in int4 g128'
needx "$W/$S1_NAME.state" 'blocks_done = 20'
needx "$W/$S1_NAME.state" 'device = cuda'
needx "$W/$S1_NAME.state" "model = $MODEL"
revision_check "$W/model-revision.txt"       # before DONE: a shard of another revision is not resumed
echo "partial ratio, blocks 0..19 quantized, 20..39 dense: x$(ratio "$W/smoke.txt")" | tee "$W/R-partial.txt"
( cd "$W" && sha256sum "$S1_NAME" "$S1_NAME.state" ) | tee "$W/files.sha256"
stat -c '%s %n' "$W/$S1_NAME" "$W/$S1_NAME.state" | tee "$W/sizes.txt"
peaks
date -u > "$W/DONE"
echo '== raw kept ==' ; ls -la "$W"
JOB

IFS= read -r -d '' SEG2 <<'JOB' || true
echo '== the shard: DONE, copied to /scratch, sha256 against seg1 =='
[ -s "$S1/DONE" ] || { echo "refused: $S1/DONE missing, seg1 did not reach its end" >&2; exit 1; }
R=/scratch/seg1; mkdir -p "$R"
cp "$S1/$S1_NAME" "$S1/$S1_NAME.state" "$R/"
( cd "$R" && sha256sum -c "$S1/files.sha256" ) | tee "$W/shard-check.txt" \
  || { echo 'refused: the shard copied to /scratch is not the one seg1 hashed' >&2; exit 1; }
cp "$S1/model-revision.txt" "$W/model-revision-seg1.txt"

echo '== Qwen3-14B, blocks 20..39, resumed; the whole .llvq straight to /out =='
date -u
env LLVQ_MODEL="$MODEL" LLVQ_CALIB=dclm-edu LLVQ_INT4_TYPES=v_proj LLVQ_THREADS="$THREADS" \
    LLVQ_RESUME="$R/$S1_NAME" LLVQ_ARTIFACT="$W/$A_NAME" \
  smoke 64 2048 12 4096 cuda nogs tetra1 40 rot 2>&1 | tee "$W/smoke.txt"
date -u
recipe_ok "$W/smoke.txt"
need "$W/smoke.txt" "resume from $R/$S1_NAME: 140 matrices, blocks 0..19 already quantized"
need "$W/smoke.txt" '✓ 140 matrices copied and reloaded (6606028800 weights, 20 blocks)'
need "$W/smoke.txt" "quantizing blocks 20..39 of 40 of $MODEL"
need "$W/smoke.txt" 'verifying 280 matrices against the evaluated model (v5, kinds Tetra+Int4G128)'
need "$W/smoke.txt" '✓ 13212057600 weights identical, bit for bit (at f32)'
need "$W/smoke.txt" 'and 40 matrices, 209715200 weights in int4 g128'
need "$W/smoke.txt" '280 matrices in the file, of which 140 matrices resumed from the shard'
# The declared caveat, on the result line (smoke.rs:1375-1380): the prereg says it is printed.
need "$W/smoke.txt" "segments                 = resumed at block 20 from $R/$S1_NAME (hidden states recomputed, not restored)"
needx "$W/$A_NAME.state" 'blocks_done = 40'
# A hybrid of two checkpoint revisions is refused before anything is sealed from it: both
# segments must have resolved MODEL_SHA, whenever the ref is found.
revision_check "$W/model-revision.txt"
echo "R = x$(ratio "$W/smoke.txt")" | tee "$W/R.txt"
( cd "$W" && sha256sum "$A_NAME" "$A_NAME.state" ) | tee "$W/files.sha256"

echo '== seal to /scratch, copy to /out, size and sha256 read back =='
date -u
SB=/scratch/$SB_NAME
LLVQ_MODEL="$MODEL" seal "$W/$A_NAME" "$SB" 2>&1 | tee "$W/seal.txt"
date -u
need "$W/seal.txt" 'format v5, 280 quantized matrices, kinds Tetra+Int4G128'
need "$W/seal.txt" 'carrying 163 tensors, 1556249600 weights'
SHA=$(sha256sum "$SB" | cut -d' ' -f1)
cp "$SB" "$W/$SB_NAME"
sync "$W/$SB_NAME" 2>/dev/null || true
test "$(stat -c %s "$W/$SB_NAME")" = "$(stat -c %s "$SB")"
test "$(sha256sum "$W/$SB_NAME" | cut -d' ' -f1)" = "$SHA"
echo "$SHA  $SB_NAME" | tee -a "$W/files.sha256"
stat -c '%s %n' "$W/$A_NAME" "$W/$SB_NAME" | tee "$W/sizes.txt"

echo '== ppl of the sealed file, f16, cuda, the same 12 windows =='
date -u
LLVQ_DTYPE=f16 ppl 4096 12 cuda "$SB" 2>&1 | tee "$W/ppl-sealed-f16.txt" | tail -3
date -u
need "$W/ppl-sealed-f16.txt" 'dtype f16, kv f16, tokens 3f1baca9033bf251'
# Control 6: the sealed f16 perplexity within 1 % of the encoding's. Outside, the object is
# suspect and the export is not paid for.
Q=$(awk '/^exact-ppl/ {print $5}' "$W/smoke.txt")
P=$(awk '/^ppl = / {print $3}' "$W/ppl-sealed-f16.txt" | tail -1)
awk -v p="$P" -v q="$Q" 'BEGIN {d = p / q - 1; printf "sealed f16 %.4f against encoding %.4f: %+.3f %%\n", p, q, 100 * d
  exit (d <= 0.01 && d >= -0.01) ? 0 : 1}' | tee "$W/ppl-control.txt" \
  || { echo 'refused: control 6, the sealed perplexity is outside 1 % of the encoding; no export' >&2; exit 1; }

echo '== export of the sealed file to /scratch, copied to the export dir =='
date -u
X=/scratch/export
"$EXPORT_BIN" "$SB" "$X" 2>&1 | tee "$W/export.txt"
date -u
need "$W/export.txt" '443 tensors identical bit for bit'
need "$W/export.txt" '(40 from Int4G128)'
mkdir -p "$EO"
( cd "$X" && sha256sum config.json tokenizer.json tokenizer_config.json model.safetensors ) | tee "$X/files.sha256"
( cd "$X" && stat -c '%s %n' config.json tokenizer.json tokenizer_config.json model.safetensors ) | tee "$X/sizes.txt"
for f in config.json tokenizer.json tokenizer_config.json model.safetensors files.sha256 sizes.txt; do
  cp "$X/$f" "$EO/$f"
  sync "$EO/$f" 2>/dev/null || true
  test "$(stat -c %s "$EO/$f")" = "$(stat -c %s "$X/$f")"
done
# The three small files are read back; model.safetensors is not: 29.5 GB through the mount
# would bill ~10 min for a read the page cache may serve. `check` on the Mac reads its size
# from the bucket, which is the authority (ops/README.md).
( cd "$EO" && sha256sum -c <(grep -v model.safetensors "$X/files.sha256") ) | tee "$W/export-check.txt"

echo '== rtbits =='
if type -P rtbits >/dev/null; then
  rtbits "$SB" 2>&1 | tee "$W/rtbits.txt"
else
  echo 'rtbits is not in this image (llvq-bench is not built by ops/Dockerfile.cuda): encode-14b.sh fetch runs it on the Mac' | tee "$W/rtbits.txt"
fi
peaks
date -u > "$W/DONE"
echo '== raw kept ==' ; ls -la "$W" "$EO"
JOB

case "$STAGE" in
  seg1) PRE="$PRE
W=/out/$S1_DIR"; BODY="$HEAD
$SEG1"; TIMEOUT=255m ;;
  seg2) PRE="$PRE
W=/out/$OBJ_DIR"; BODY="$HEAD
$SEG2"; TIMEOUT=285m ;;
esac

if [ "$DRY" = 1 ]; then
  # What ops/run.py hands to ['bash', '-lc', ...]: printed and parsed, not run.
  JOBSCRIPT="set -euo pipefail
$PRE
$BODY"
  printf '%s\n' "$JOBSCRIPT"
  printf '%s\n' "$JOBSCRIPT" | bash -n && echo "DRY_RUN: $STAGE job script parses; $FLAVOR, timeout $TIMEOUT; nothing launched" >&2
  exit 0
fi

uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor "$FLAVOR" --any-flavor --timeout "$TIMEOUT" \
  --bucket "$BUCKET" --out-mount /out \
  --name "encode-14b-$STAGE" \
  "$PRE" "$BODY"
