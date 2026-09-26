#!/usr/bin/env bash
# DRAFT, not launched. The 14B export, landed in the bucket in PARTS of at most 1 GB.
#
# Why this job exists. `encode-14b.sh seg2` finished COMPLETED and its export was correct
# *inside the container*: `443 tensors identical bit for bit`, the four files hashed on
# /scratch, copied to /out/dclm-14b-export-2026-09-22/ and each read back at the right size —
# the job's own `ls` at 18:43 UTC shows `model.safetensors` there at 29,536,665,800 B. The
# bucket says otherwise, and `hf buckets ls` is the authority (ops/README.md, "Bucket mounting
# fails silently"). Listed on 2026-09-22 from the Mac:
#
#   dclm-14b-export-2026-09-22/   config.json, tokenizer.json, tokenizer_config.json  — and
#                                 NOT model.safetensors, files.sha256, sizes.txt
#   dclm-14b-2026-09-22/          everything up to export.txt (18:41:32) — and NOT DONE,
#                                 mem.csv, peaks.txt, export-check.txt, rtbits.txt
#   dclm-14b-seg1-2026-09-22/     COMPLETE, DONE and mem.csv and peaks.txt included
#
# Read those three lines together and the mechanism is not "the tail of a job is lost". seg1
# lost nothing, and its largest single write was the 1.72 GB shard. seg2 lost **everything it
# wrote from the 29.5 GB `cp` onward, in BOTH directories at once** — the copy loop's order is
# config, tokenizer, tokenizer_config, model.safetensors, files.sha256, sizes.txt, and the cut
# falls exactly there. One oversized write to the mount takes the rest of the job with it. The
# in-job read-back did not catch it because `stat` on the mount is served by the page cache.
# This is the same layer as the truncated 4 GB write of ops/README.md:246-250, which is why
# `dequant` defaults to 1 GB shards here and to 4 GB on a local disk.
#
# ⚠️ Read encode-14b.sh:447-451 before trusting anything below. seg2 ALREADY did, for all six
# files and `model.safetensors` among them: `cp`, then `sync "$EO/$f" 2>/dev/null || true`, then
# `test "$(stat -c %s "$EO/$f")" = "$(stat -c %s "$X/$f")"`. All three passed, and the job's own
# closing `ls` printed `29536665800  model.safetensors` in that directory
# (`hf jobs logs 6ab2a33251992417dfcd390e`). So the per-file `sync` and the size read-back
# repeated below are NOT what is new here and are not evidence of anything: they are the exact
# pair that already reported success on a file the bucket never received. **The only
# load-bearing change in this job is the size of a single write.** Everything else is
# bookkeeping, and the two things that can actually prove the landing are outside the job:
# `export-14b.sh check` against `hf buckets ls` on the Mac, and the whole-file sha256 that
# `ops/jobs/dclm-14b-rowscales.sh` recomputes on the training card after reassembly.
#
# So: re-export, and write the result in parts of at most 1,000,000,000 B — under every size
# ever observed to fail, and the size the bucket already carries 33 of. Precedents, measured:
# the 8B export's 16.4 GB `model.safetensors` landed whole (`dclm-8b-export-2026-09-21/`), the
# sealed 14B's 6,563,782,117 B landed whole tonight in the very job that lost the 29.5 GB file,
# and the AWQ 14B reconstruction sits in the bucket as 33 shards of ≤ 1 GB and was read back
# whole tonight (29,552,586,033 B in 240 s).
#
# 🕳️ Consequence for the preflight: `dclm-14b-2026-09-22/DONE` IS NOT IN THE BUCKET, although
# seg2 wrote it. Nothing here may gate on it. What proves seg2 reached its end is the set that
# did land — the sealed file at its exact byte count, `files.sha256`, `sizes.txt`,
# `ppl-control.txt`, `export.txt` — and that set is a stronger gate than a marker anyway.
#
# Preregistered: proofs/preregistration-encode-14b-2026-09-22.md, ALREADY STAMPED. This rerun is
# its **control 8** ("The export: 443 tensors identical bit for bit, 40 from Int4G128, four
# files with sizes and sha256. A failed export does not void 1 to 7: it is rerun alone").
# Controls 1 to 7 passed and stand; R = ×1.1709 is published from seg2 and nothing here can
# change it. Hard rule 2: the launcher refuses until the `.ots` exists (DRY_RUN=1 excepted).
# What this job adds beyond the prereg's control 8 — parts, a manifest, a Mac-side re-check —
# is a change of *transport*, not of object, and goes in the deviations file beside the prereg.
#
# The object is not re-encoded and not re-sealed. The sealed file
# `dclm-14b-2026-09-22/qwen3-14b-dclm.bin` (6,563,782,117 B, sha256 f8e975b8…) is read from the
# mount, copied to local disk and checked against the sha256 seg2 wrote, then exported again.
#
# ## The gate that makes this cheap to believe
#
# The export is byte-deterministic for a fixed sealed file and a fixed image. `decode_matrix`
# rebuilds in f64 and narrows once (`export.rs:100-120`), and `safetensors::prepare` sorts its
# entries by descending dtype then by name (`safetensors-0.7.0/src/tensor.rs:231-235`) — NOT by
# HashMap iteration order, which is randomized per process and would otherwise move the header
# bytes. So the fresh `model.safetensors` must hash to what seg2 hashed:
#
#   07e82f7c239dec48e49f79ddfed3accfbf24f6e50e669ca45bc63cb76b5046f1
#
# The job refuses if it does not. That single equality proves, in one line and before a single
# part is written, that the mount read the sealed file intact and that this image reproduces
# seg2's export. It is checked against the image seg2 ran on: `IMAGE_SHA` is required and is
# compared with `provenance-seg2.txt`. `ST_SHA=<other>` overrides it, and then this job has
# stopped being control 8 and the deviation says so.
#
# No `oracle`: `cpu-xl` has no GPU and this image's forward pass never runs here (hard rule 10
# is about backends, and there is no backend to prove). What stands in its place is the export's
# own read-back — it re-reads the file it wrote and compares 443 tensors bit for bit — and the
# sha256 equality above.
#
# ## Flavor, and the one real risk
#
# `cpu-xl`: 16 vCPU, 124 GB RAM, 1000 GB of storage, $1.00/h (`hf jobs hardware`, 2026-09-22).
# `export` hard-codes `Device::Cpu` and is built WITHOUT the `cuda` feature exactly so it can
# run here (ops/Dockerfile.cuda:67-77).
#
# 🚨 **RAM is the risk and it is not comfortable.** `export` measured VmHWM **86.5 GB** in seg2
# (`peaks.txt`), on a 256 GB host. Here the ceiling is 124 GB — 37.5 GB of headroom, and the
# peak already includes the mmap of the 29.5 GB file it re-reads to verify, so the shape of the
# peak is known, not guessed. Page cache from writing 29.5 GB is reclaimable and should not add
# to it. If the job dies without a line, that is an OOM kill: rerun with
# `FLAVOR=cpu-performance` (32 vCPU, 256 GB, $1.90/h), which costs $0.41 more at the same wall
# clock and removes the question. Nothing else in this script changes. The job refuses at its
# first line if MemTotal is under 100 GB, so a mistyped `FLAVOR` dies in seconds rather than
# after ten billed minutes of export.
#
# Disk: 6.56 (sealed) + 29.54 (export) + 29.54 (parts) = 65.6 GB of 1000. The export's local
# copy is deliberately NOT deleted before the split: a part that fails to copy is then retried
# from local disk rather than from a second export.
#
# ## Cost, estimated from rates measured tonight
#
# Bucket-mount READ 123 MB/s (29,552,586,033 B in 240 s, `census-14b-ref-2026-09-22-brut/
# stage-awq.txt`); local sha256 284 MB/s (29.5 GB in 104 s, same file); `export` itself 411 s
# on the encode host (18:34:38 → 18:41:29 UTC, `hf jobs logs 6ab2a33251992417dfcd390e`), where
# the decode loop is serial — `llvq-artifact` carries no rayon — so 16 vCPU instead of 23 costs
# little. The mount WRITE rate is NOT measured; it is assumed no better than the read.
#
#   pull + start                                            3 min
#   sealed file: 6.56 GB off the mount + local sha256       1.3
#   export                                                  9    [7, 12]
#   split 29.5 GB, local → local                            4    [3, 6]
#   sha256 of the 30 parts, local                           2
#   copy 30 parts to /out + size read-back                  5    [4, 12]
#   small files, manifest, listing                          1
#   settling tail, SETTLE=120 s                             2
#                                                    total 27.3 min [23.3, 39.3]
#   $0.46 central [$0.39, $0.66]; timeout 90m, ceiling $1.50.
#
# The total is the sum of the column above it, and the two bounds are the sums of the two
# bounds — never a wider round number picked by hand. A row with one figure has no range
# because nothing in it varies: `pull + start` and `settle` are fixed, and the sha256 of 30 GB
# at a measured 284 MB/s is a rate with no spread worth writing down.
#
# The settling tail is insurance, not a fix: seg1 wrote its `DONE` and its `mem.csv` in its
# last seconds and both landed, so nothing measured says the sync needs time. It buys two
# quiet minutes after the last write for $0.033, after a bare `sync`, with the sampler stopped
# and — unlike seg2 — no file of ours ever opened on the mount for the length of the job.
# `SETTLE=0` removes it.
#
# 14B chain spent before this job: $18.17 measured over three rows of docs/data/jobs.csv
# (census-14b-ref $3.14, encode-14b-seg1 $6.95, encode-14b-seg2 $8.08). After, at the ceiling:
# $19.67 of the $52 the encode prereg records.
#
# ## What lands, and in what order
#
# /out/dclm-14b-export-2026-09-22/
#   parts/model.safetensors.part-000 … part-029   29 × 1,000,000,000 B + 1 × 536,665,800 B
#   config.json  tokenizer.json  tokenizer_config.json      (again; they already landed)
#   parts.sha256      `sha256sum -c`-able, paths `parts/…`
#   files.sha256      the four files as seg2 wrote them, model.safetensors WHOLE
#   sizes.txt         `stat -c '%s %n'` of the four
#   parts.manifest    `whole` / `count` / `part` / `small` records, one per line
#   DONE              written LAST
#
# The `parts/` level is not a gamble: `hf buckets ls` lists a nested prefix in this very bucket
# (`volume-2026-09-07/v1c/` holds a 980,791,242 B `q4b.llvq`, checked 2026-09-22), and a
# subdirectory shows in its parent's listing as a row with no size column, which none of the
# name lookups here can match. Were it ever to go wrong, the objects would still be in the
# bucket under their own keys and `hf buckets cp` would still fetch them by name; only the
# listing would be inconvenient, and the fix is to drop the `parts/` level.
#
# The parts go first and `DONE` last on purpose: what seg2 lost was everything written from the
# big file onward, so a dropped tail shows up as a missing `DONE` rather than as a silently
# short directory. The job's own read-back of each part's size is kept because it is free, and
# is believed only as far as a page cache allows — `export-14b.sh check`, on the Mac, against
# `hf buckets ls`, is the authority, and `ops/jobs/dclm-14b-rowscales.sh` re-hashes the whole
# reassembled file on the training card before it trains a step. Run `check` after the job
# reports COMPLETED, not while it runs.
#
# One writer at a time on the bucket (ops/README.md, mount failure 1: the second job dies on
# `Volume mount failed: init container exhausted retries`, before starting, so without billing —
# but it dies). Nothing else of ours may be writing to `Pier-Jean/jobs-artifacts` while this
# runs, and this job is the one that must not be disturbed.
#
# Usage (from any checkout; REPO is this script's own):
#   DRY_RUN=1 bash ops/jobs/export-14b.sh run     prints the job script, parses it, launches nothing
#   DRY_RUN=1 bash ops/jobs/export-14b.sh check   says what it would check, reads nothing
#   IMAGE_SHA=af90741654a5b28c51f6993c24c94993813220bc bash ops/jobs/export-14b.sh run
#   bash ops/jobs/export-14b.sh check
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
STAGE=${1:-}
D=${OBJ_DATE:-2026-09-22}                          # the object's encoding date: names every dir
PREREG=${PREREG:-proofs/preregistration-encode-14b-2026-09-22.md}
BUCKET=Pier-Jean/jobs-artifacts
FLAVOR=${FLAVOR:-cpu-xl}
OLD_IMAGE=a963a02010cec2d3c342ec52dd38f2dded06f2a1  # afaed1e: no export, no rowscale
SEG2_IMAGE=af90741654a5b28c51f6993c24c94993813220bc # the image seg2 ran on; the sha256 gate needs it
LOG=${LOG:-$HOME/q14b-dclm-$D}                     # Mac-side provenance, same dir as the encode
TIMEOUT=${TIMEOUT:-90m}
SETTLE=${SETTLE:-120}                                # quiet seconds after the last write; 0 removes it
DRY=${DRY_RUN:-0}

OBJ_DIR=dclm-14b-$D
EXP_DIR=dclm-14b-export-$D
W_DIR=export-14b-$D                                # this job's own logs, beside the payload
SB_NAME=qwen3-14b-dclm.bin
ST_NAME=model.safetensors

# Measured in seg2 (`hf jobs logs 6ab2a33251992417dfcd390e`), not predicted.
SB_BYTES=6563782117
ST_BYTES=29536665800
ST_SHA=${ST_SHA:-07e82f7c239dec48e49f79ddfed3accfbf24f6e50e669ca45bc63cb76b5046f1}
PART_BYTES=1000000000                              # the `dequant --shard-gb` default, decimal GB
# 29 × 1e9 + 536,665,800. Computed here so the job refuses a split that produced another count.
N_PARTS=30
TAIL_BYTES=536665800

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
size_of() { awk -v n="$2" '$NF ~ ("/" n "$") || $NF == n {print $1}' <<<"$1"; }

# ---- the Mac-only stage, $0 ------------------------------------------------------------
case "$STAGE" in
check)
  # The authority (ops/README.md, "Bucket mounting fails silently"). Everything the job claims
  # to have written, as the BUCKET reports it: the three small files at their exact sizes, the
  # bookkeeping files present, every manifest part listed at its manifest size, nothing extra,
  # and the unforgiving sum — the parts must add up to the whole file to the byte.
  if [ "$DRY" = 1 ]; then
    echo "DRY_RUN: would list $EXP_DIR/ and $EXP_DIR/parts/, fetch parts.manifest to $LOG,"
    echo "         and demand $N_PARTS parts summing to $ST_BYTES B; reads nothing"
    exit 0
  fi
  bad=0
  L=$(bucket_ls "$EXP_DIR")
  printf '%s\n' "$L"
  row() {  # name predicted
    local got
    got=$(size_of "$L" "$1")
    if [ -z "$got" ]; then echo "MISSING  $EXP_DIR/$1"; bad=1; return; fi
    if [ "$got" = "$2" ]; then echo "EXACT    $EXP_DIR/$1  $got B"; else
      echo "OUTSIDE  $EXP_DIR/$1  $got B, predicted $2 B"; bad=1; fi
  }
  row config.json 728
  row tokenizer.json 11422654
  row tokenizer_config.json 64
  for f in parts.manifest parts.sha256 files.sha256 sizes.txt DONE; do
    if [ -z "$(size_of "$L" "$f")" ]; then echo "MISSING  $EXP_DIR/$f"; bad=1; else
      echo "present  $EXP_DIR/$f"; fi
  done
  WHOLE=$(size_of "$L" "$ST_NAME")
  [ -z "$WHOLE" ] || echo "note     $EXP_DIR/$ST_NAME is ALSO in the bucket, $WHOLE B (the launchers prefer it)"

  mkdir -p "$LOG"
  hf buckets cp "hf://buckets/$BUCKET/$EXP_DIR/parts.manifest" "$LOG/parts.manifest"
  awk -v b="$ST_BYTES" -v s="$ST_SHA" -v n="$ST_NAME" '
    $1 == "whole" && $2 == n { seen = 1
                               if ($3 == b && $4 == s) { printf "EXACT    whole %s %s B, sha256 matches seg2\n", n, $3; ok = 1 }
                               else printf "OUTSIDE  whole %s: %s B / %s in the manifest, want %s B / %s\n", n, $3, $4, b, s }
    END { if (!seen) printf "MISSING  the manifest has no whole-file line for %s\n", n
          exit ok ? 0 : 1 }' "$LOG/parts.manifest" || bad=1

  PL=$(bucket_ls "$EXP_DIR/parts")
  awk -v total="$ST_BYTES" -v want_n="$N_PARTS" '
    FNR == NR { if ($1 == "part") { want[$2] = $3; n++ } ; next }
    $1 ~ /^[0-9]+$/ { b = $NF; sub(/.*\//, "", b); got[b] = $1; m++ }
    END {
      bad = 0
      for (k in want) {
        if (!(k in got))       { printf "MISSING  parts/%s\n", k; bad = 1; continue }
        if (got[k] != want[k]) { printf "OUTSIDE  parts/%s: %d B in the bucket, %d in the manifest\n", k, got[k], want[k]; bad = 1; continue }
        s += got[k]; ok++
      }
      for (k in got) if (!(k in want)) { printf "EXTRA    parts/%s, %d B\n", k, got[k]; bad = 1 }
      printf "%d of %d parts listed at their manifest size (%d in the listing), sum %d B, predicted %d\n", ok, n, m, s, total
      if (n != want_n) { printf "the manifest holds %d parts, predicted %d\n", n, want_n; bad = 1 }
      if (s != total)  { print  "the parts do not sum to the whole file"; bad = 1 }
      exit bad
    }' "$LOG/parts.manifest" <(printf '%s\n' "$PL") || bad=1
  exit "$bad"
  ;;
run) ;;
*) sed -n '2,40p' "$0"; exit 2 ;;
esac

# ---- preflight on the Mac, $0 ----------------------------------------------------------
if [ "$DRY" = 1 ]; then
  IMAGE_SHA=${IMAGE_SHA:-DRYRUN}
else
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || refuse "$PREREG(.ots) missing"
  [ -n "${IMAGE_SHA:-}" ] || refuse "IMAGE_SHA=<the Space sha seg2 ran on> is required: the sha256 gate compares this export with seg2's"
  [ "$IMAGE_SHA" != "$OLD_IMAGE" ] || refuse "IMAGE_SHA is a963a020 (afaed1e), which has no export"
  if [ "$IMAGE_SHA" != "$SEG2_IMAGE" ] && [ "${FORCE_IMAGE:-0}" != 1 ]; then
    refuse "IMAGE_SHA is not seg2's image ($SEG2_IMAGE): the export sha256 gate is only sound on the same build (FORCE_IMAGE=1, and then set ST_SHA too)"
  fi
  P2="$LOG/provenance-seg2.txt"
  if [ -s "$P2" ]; then
    [ "$(awk '$1=="image" {print $2}' "$P2")" = "$IMAGE_SHA" ] || refuse "seg2 ran on another image ($P2)"
  else
    echo "note: $P2 absent; IMAGE_SHA is checked against the constant only" >&2
  fi
  SPACE=$(uv run --quiet --with huggingface_hub python -c \
    "from huggingface_hub import HfApi; print(HfApi().space_info('Pier-Jean/llvq-runner-cuda').sha)")
  echo "image $SPACE"
  [ "$SPACE" = "$IMAGE_SHA" ] || refuse "the Space is at $SPACE, not IMAGE_SHA=$IMAGE_SHA"

  # The sealed file must be in the bucket at its byte count, with the rest of seg2's end-state
  # beside it: this job re-exports, it does not re-seal, and nothing else can produce that file.
  # NOT `DONE` — seg2 wrote one and the bucket does not have it (see the header). The set below
  # is what did land, and it says more: a sealed file at the predicted byte count, hashed, its
  # size recorded, its perplexity control passed and an export attempted.
  LO=$(bucket_ls "$OBJ_DIR")
  printf '%s\n' "$LO"
  [ "$(size_of "$LO" "$SB_NAME")" = "$SB_BYTES" ] \
    || refuse "$OBJ_DIR/$SB_NAME is $(size_of "$LO" "$SB_NAME") B, want $SB_BYTES"
  for f in files.sha256 sizes.txt ppl-control.txt export.txt; do
    [ -n "$(size_of "$LO" "$f")" ] || refuse "$OBJ_DIR/$f missing: seg2 did not reach its end"
  done

  # The export dir is NOT required to be empty — three small files landed there and this job
  # rewrites them. What must be empty is parts/: a half-written previous attempt is an operator
  # decision, not something to overwrite in place.
  LE=$(bucket_ls "$EXP_DIR")
  printf '%s\n' "$LE"
  LP=$(bucket_ls "$EXP_DIR/parts")
  [ -z "$LP" ] || refuse "$EXP_DIR/parts/ already holds files; clear it or pick another OBJ_DATE"
  [ -z "$(size_of "$LE" DONE)" ] || refuse "$EXP_DIR/DONE already there: this export already landed"
  if [ "$(size_of "$LE" "$ST_NAME")" = "$ST_BYTES" ]; then
    refuse "$EXP_DIR/$ST_NAME is already in the bucket at $ST_BYTES B: nothing to redo"
  fi
  LW=$(bucket_ls "$W_DIR")
  [ -z "$LW" ] || refuse "$W_DIR/ already holds files; pick another OBJ_DATE"

  mkdir -p "$LOG"
  { echo "image $IMAGE_SHA"
    echo "stage $STAGE $(date -u +%FT%TZ)"
    echo "launcher $(shasum -a 256 "$0" | cut -d' ' -f1)"
    echo "prereg $(shasum -a 256 "$PREREG" | cut -d' ' -f1)"
    echo "sealed $SB_BYTES $OBJ_DIR/$SB_NAME"
    echo "expect $ST_BYTES $ST_SHA"
    echo "head $(git rev-parse HEAD)"; } | tee "$LOG/provenance-export.txt"
fi

# ---- the job ----------------------------------------------------------------------------
PRE=$(printf 'O=%q\nEO=%q\nW=%q\nSB_NAME=%q\nST_NAME=%q\nSB_BYTES=%q\nST_BYTES=%q\nST_SHA=%q\nPART_BYTES=%q\nN_PARTS=%q\nTAIL_BYTES=%q\nSETTLE=%q' \
  "/out/$OBJ_DIR" "/out/$EXP_DIR" "/out/$W_DIR" \
  "$SB_NAME" "$ST_NAME" "$SB_BYTES" "$ST_BYTES" "$ST_SHA" "$PART_BYTES" "$N_PARTS" "$TAIL_BYTES" "$SETTLE")

# Quoted heredoc: nothing expands on the Mac. `read -d ''` rather than $(cat <<'JOB'): the
# Mac's /bin/bash is 3.2. ops/run.py prepends `set -euo pipefail` and runs the whole as
# ['bash', '-lc', script] (ops/run.py:1088-1100).
IFS= read -r -d '' BODY <<'JOB' || true
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
# `type -P`, not `command -v`: `export` is a bash builtin, so `command -v export` always
# succeeds and a bare `export a b` runs the builtin. The binary is called by its path.
# `split` and the coreutils below are checked here, before the job pays for an export it could
# not then cut up — the `nullkbench` lesson, applied to a tool rather than to a binary of ours.
for b in export split sha256sum stat sync cat; do
  type -P "$b" >/dev/null || { echo "refused: $b is not in this image" >&2; exit 1; }
done
EXPORT_BIN=$(type -P export)
mkdir -p "$W" /scratch
{ echo "nproc $(nproc)"; grep -m1 'model name' /proc/cpuinfo || true; grep '^MemTotal:' /proc/meminfo || true
  df -h /scratch / 2>&1 | tail -n +2 || true; } | tee "$W/host.txt"

# The flavor, before anything is paid for. `--any-flavor` lets `ops/run.py bench` take any
# name, and `FLAVOR` is an environment variable, so a typo buys a machine that cannot finish:
# `export` measured VmHWM 86.5 GB, and the local disk must hold 6.56 (sealed) + 29.54 (export)
# + 29.54 (parts) = 65.6 GB. Both are checked in the first second, not ten billed minutes in.
# Both reads are `|| VAR=` so that an unreadable /proc or a `df` that fails refuses with the
# sentence below rather than with awk's own error: `set -e` would otherwise kill the job on a
# line that names neither the flavor nor the reason.
MEMKB=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo 2>/dev/null) || MEMKB=
[ "${MEMKB:-0}" -ge 100000000 ] \
  || { echo "refused: MemTotal reads '$MEMKB' kB; export peaked at 86.5 GB in seg2 and this is not cpu-xl" >&2; exit 1; }
FREEKB=$(df -Pk /scratch 2>/dev/null | awk 'NR == 2 {print $4}') || FREEKB=
[ "${FREEKB:-0}" -ge 80000000 ] \
  || { echo "refused: /scratch reads '$FREEKB' kB free; this job writes 65.6 GB there" >&2; exit 1; }
echo "host ok: MemTotal $MEMKB kB, /scratch $FREEKB kB free"

# Memory every 30 s. No nvidia-smi arm: cpu-xl has no GPU. The number this exists for is
# `export`'s VmHWM against the flavor's 124 GB; seg2 measured 86.5 GB on a 256 GB host.
mem_sampler() {
  echo 'utc,host_avail_kb,proc,vmrss_kb,vmhwm_kb'
  while :; do
    a=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null) || a=
    p=; c=; r=; h=
    for d in /proc/[0-9]*; do
      n=$(cat "$d/comm" 2>/dev/null) || continue
      case "$n" in export|split|sha256sum|cp) p=${d#/proc/}; c=$n; break ;; esac
    done
    if [ -n "$p" ]; then
      r=$(awk '/^VmRSS:/ {print $2}' "/proc/$p/status" 2>/dev/null) || r=
      h=$(awk '/^VmHWM:/ {print $2}' "/proc/$p/status" 2>/dev/null) || h=
    fi
    echo "$(date -u +%FT%TZ),$a,$c,$r,$h"
    sleep 30
  done
}
# To /scratch, not to "$W": this is the one file that would otherwise stay OPEN on the mount
# for the whole job, and seg2's `mem.csv` is one of the files the bucket did not receive. It is
# copied to "$W" at the end like every other artifact, so it goes through the same size
# read-back as the rest instead of depending on a handle closing cleanly at job exit.
mem_sampler > /scratch/mem.csv 2>/dev/null &
SAMPLER=$!
trap 'kill "$SAMPLER" 2>/dev/null || true' EXIT
peaks() {
  awk -F, 'NR > 1 && $2 != "" {if (n == 0 || $2 + 0 < lo) lo = $2 + 0; n++}
           NR > 1 && $5 != "" {if ($5 + 0 > h[$3]) h[$3] = $5 + 0}
           END {printf "host MemAvailable low-water %.1f GB over %d samples\n", lo / 1e6, n
                for (k in h) printf "  %s VmHWM %.1f GB\n", k, h[k] / 1e6}' /scratch/mem.csv | tee /scratch/peaks.txt
}
need() { grep -qF -- "$2" "$1" || { echo "refused: $1 lacks: $2" >&2; exit 1; }; }

echo '== the sealed file: off the mount to local disk, sha256 against seg2 =='
date -u
# Not `$O/DONE`: seg2 wrote one and the bucket does not carry it (the header explains why).
# The sha256 below is the real gate on this file, and it is checked two lines down.
[ -s "$O/files.sha256" ] || { echo "refused: $O/files.sha256 missing, seg2 did not reach its end" >&2; exit 1; }
SB=/scratch/$SB_NAME
cp "$O/$SB_NAME" "$SB"
[ "$(stat -c %s "$SB")" = "$SB_BYTES" ] \
  || { echo "refused: the sealed copy is $(stat -c %s "$SB") B, want $SB_BYTES" >&2; exit 1; }
# seg2 hashed from inside its directory (`cd "$W" && sha256sum ...`), so the names are bare.
awk -v n="$SB_NAME" '$2 == n' "$O/files.sha256" > /scratch/sealed.sha256
[ "$(grep -c . /scratch/sealed.sha256)" = 1 ] \
  || { echo "refused: want exactly one sha256 line for $SB_NAME in $O/files.sha256" >&2; exit 1; }
( cd /scratch && sha256sum -c sealed.sha256 ) | tee "$W/sealed-check.txt"
date -u

echo '== export to LOCAL disk (never the mount: export re-reads by mmap to verify) =='
date -u
X=/scratch/export
"$EXPORT_BIN" "$SB" "$X" 2>&1 | tee "$W/export.txt"
date -u
need "$W/export.txt" '443 tensors identical bit for bit'
need "$W/export.txt" '(40 from Int4G128)'
[ "$(stat -c %s "$X/$ST_NAME")" = "$ST_BYTES" ] \
  || { echo "refused: the export is $(stat -c %s "$X/$ST_NAME") B, want $ST_BYTES" >&2; exit 1; }
# The gate. The export is byte-deterministic for a fixed sealed file and a fixed image
# (safetensors sorts by dtype then name, never by HashMap order), so this equality proves in
# one line that the mount read the sealed file intact and that this build reproduces seg2's.
ST_GOT=$(sha256sum "$X/$ST_NAME" | cut -d' ' -f1)
echo "$ST_GOT  $ST_NAME (fresh)" | tee "$W/export-sha256.txt"
[ "$ST_GOT" = "$ST_SHA" ] \
  || { echo "refused: the fresh export hashes $ST_GOT, seg2's hashed $ST_SHA" >&2; exit 1; }
echo "the fresh export is seg2's export, byte for byte" | tee -a "$W/export-sha256.txt"

echo "== split into parts of at most $PART_BYTES B =="
date -u
P=/scratch/parts
mkdir -p "$P"
# -d -a 3: numeric suffixes 000..029, fixed width, so the shell glob and `sort` agree and the
# reassembly order is the split order whatever reads it.
split -b "$PART_BYTES" -d -a 3 "$X/$ST_NAME" "$P/$ST_NAME.part-"
GOT_N=$(ls "$P" | grep -c .)
[ "$GOT_N" = "$N_PARTS" ] || { echo "refused: split produced $GOT_N parts, want $N_PARTS" >&2; exit 1; }
[ "$(stat -c %s "$P/$ST_NAME.part-$(printf '%03d' $((N_PARTS - 1)))")" = "$TAIL_BYTES" ] \
  || { echo "refused: the last part is not $TAIL_BYTES B" >&2; exit 1; }
( cd /scratch && sha256sum parts/* ) > /scratch/parts.sha256
date -u

echo '== the manifest, written from local disk =='
{ echo "# $ST_NAME of the 14B paper-2 base, split into parts of at most $PART_BYTES B"
  echo "# reassemble: cat parts/$ST_NAME.part-* > $ST_NAME   (the glob order IS the split order)"
  echo "# fields: whole NAME SIZE SHA256 | count N | part NAME SIZE SHA256 | small NAME SIZE SHA256"
  echo "whole $ST_NAME $(stat -c %s "$X/$ST_NAME") $ST_GOT"
  echo "count $GOT_N"
  for f in "$P"/*; do
    echo "part $(basename "$f") $(stat -c %s "$f") $(sha256sum "$f" | cut -d' ' -f1)"
  done
  for f in config.json tokenizer.json tokenizer_config.json; do
    echo "small $f $(stat -c %s "$X/$f") $(sha256sum "$X/$f" | cut -d' ' -f1)"
  done; } > /scratch/parts.manifest
cat /scratch/parts.manifest
( cd "$X" && sha256sum config.json tokenizer.json tokenizer_config.json "$ST_NAME" ) > /scratch/files.sha256
( cd "$X" && stat -c '%s %n' config.json tokenizer.json tokenizer_config.json "$ST_NAME" ) > /scratch/sizes.txt

echo '== the parts to the bucket, one by one, each size read back from /out =='
date -u
mkdir -p "$EO/parts"
# No pipe around this loop: a `| tee` would put it in a subshell, where `exit 1` exits the
# subshell and the pipeline reports tee's status. The file is appended and printed after.
: > "$W/parts-readback.txt"
for f in "$P"/*; do
  b=$(basename "$f")
  cp "$f" "$EO/parts/$b"
  sync "$EO/parts/$b" 2>/dev/null || true
  g=$(stat -c %s "$EO/parts/$b"); w=$(stat -c %s "$f")
  [ "$g" = "$w" ] || { echo "refused: parts/$b is $g B on /out, $w local" >&2; exit 1; }
  echo "$b $g" >> "$W/parts-readback.txt"
done
cat "$W/parts-readback.txt"
awk -v t="$ST_BYTES" '{s += $2} END {printf "%d parts on /out, sum %d B, whole file %d\n", NR, s, t
  exit (s == t) ? 0 : 1}' "$W/parts-readback.txt" | tee "$W/parts-sum.txt" \
  || { echo 'refused: the parts on /out do not sum to the whole file' >&2; exit 1; }
date -u

echo '== the small files and the bookkeeping, then DONE last =='
for f in config.json tokenizer.json tokenizer_config.json; do
  cp "$X/$f" "$EO/$f"
  sync "$EO/$f" 2>/dev/null || true
  [ "$(stat -c %s "$EO/$f")" = "$(stat -c %s "$X/$f")" ] \
    || { echo "refused: $f is short on /out" >&2; exit 1; }
done
for f in parts.sha256 files.sha256 sizes.txt parts.manifest; do
  cp "/scratch/$f" "$EO/$f"
  sync "$EO/$f" 2>/dev/null || true
  [ "$(stat -c %s "$EO/$f")" = "$(stat -c %s "/scratch/$f")" ] \
    || { echo "refused: $f is short on /out" >&2; exit 1; }
done
# The three small files re-read from the mount. `model.safetensors` is not there as one file,
# so its line is dropped; the parts were read back by size above, and `export-14b.sh check` on
# the Mac is what actually settles it (a stat inside the job can be served by the page cache —
# that is exactly how seg2's loss went unseen).
( cd "$EO" && sha256sum -c <(grep -vF "$ST_NAME" /scratch/files.sha256) ) | tee "$W/small-check.txt"
# The sampler is stopped BEFORE its file is copied, so that nothing of ours is still writing
# anywhere when the last bytes go to the mount. It samples to /scratch (see above), so this is
# ordering hygiene rather than the rescue of an open handle.
kill "$SAMPLER" 2>/dev/null || true
wait "$SAMPLER" 2>/dev/null || true
peaks
for f in mem.csv peaks.txt; do
  cp "/scratch/$f" "$W/$f"
  [ "$(stat -c %s "$W/$f")" = "$(stat -c %s "/scratch/$f")" ] \
    || { echo "refused: $f is short on /out" >&2; exit 1; }
done
date -u > "$EO/DONE"
sync "$EO/DONE" 2>/dev/null || true
date -u > "$W/DONE"
# A bare `sync`, once, after the last write. Every `sync FILE` above is `2>/dev/null || true`,
# so on a mount whose fsync is a no-op — or absent — those calls return success having flushed
# nothing, and the argument-less form is the only one that asks the kernel to flush everything
# it still holds. It costs about a second and covers the case the per-file form cannot.
sync 2>/dev/null || true
echo '== what the container sees (NOT the authority: run `export-14b.sh check` on the Mac) =='
ls -la "$EO" "$EO/parts" "$W"
# Two quiet minutes after the last write, with no file open on the mount. Insurance against a
# sync that needs a window; `SETTLE=0` removes it. See the cost table in the header.
if [ "$SETTLE" -gt 0 ]; then
  echo "== settling $SETTLE s, nothing open on /out =="
  sleep "$SETTLE"
  date -u
fi
JOB

if [ "$DRY" = 1 ]; then
  # What ops/run.py hands to ['bash', '-lc', ...]: printed and parsed, not run.
  JOBSCRIPT="set -euo pipefail
$PRE
$BODY"
  printf '%s\n' "$JOBSCRIPT"
  printf '%s\n' "$JOBSCRIPT" | bash -n && echo "DRY_RUN: run job script parses; $FLAVOR, timeout $TIMEOUT; nothing launched" >&2
  exit 0
fi

uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor "$FLAVOR" --any-flavor --timeout "$TIMEOUT" \
  --bucket "$BUCKET" --out-mount /out \
  --name "export-14b" \
  "$PRE" "$BODY"
