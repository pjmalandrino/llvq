#!/usr/bin/env bash
# INTEGRATOR COPY of census3.sh: only the timeout changes (160m -> 135m, $4.80 -> $4.05).
# DRAFT. Three-arm MMLU census at Qwen3-8B, full split, one job, one card, one image.
# Destined for ops/jobs/census-8b.sh once the operator gives the go.
#
# Preregistered: proofs/preregistration-census-8b-${D}.md   <-- NOT WRITTEN YET.
#   Hard rule 2 and preregistration-dclm-8b-2026-09-21.md:15 ("each of its jobs needs
#   its own prereg"). This launcher refuses to run until the .ots exists.
#
# Arms, all f16, dense reconstruction, LLVQ_MMLU_ALLOC=flat, no limit = census, 14,042 q:
#   A  DCLM base    /out/dclm-8b-2026-09-21/qwen3-8b-dclm.bin  sealed: Tetra + 36 v_proj int4 g128
#   B  f16          Qwen/Qwen3-8B                             Hub, public
#   C  AWQ deq      Pier-Jean/qwen3-8b-awq-deq                Hub, public
# Order: A first, because the trained arm pairs against it; C last, the least needed if
# the job dies at its timeout.
#
# Why B and C are needed: the only 8B f16 and AWQ dumps are limit=40, fingerprint
# 65dcd53655e8bfa5 (docs/data/mmlu-dumps/mmlu-8b-{f16,awq}.csv). mmlupair pairs them with
# a census dump (a74a6d6213602979) only under --intersect, on the 2,280 shared questions
# (mmlupair.rs:326-365), never on the full split, and the 8B sample read 2.24 pp low
# (vod-8b-2026-09-18.txt:53-55). No 8B f16 or AWQ census exists, in the repo or in the
# bucket (checked 2026-09-21 by `hf buckets ls -R`).
# Free by-product: B and C against those two limit=40 dumps under --intersect are a
# harness control across the device port (85a7ec9, 01c5c66, e1d2c9e changed model.rs
# after every committed FULL dump): same weights, same questions, only the image differs.
#
# SIGBUS: B and C are downloaded from the Hub into HF_HOME=/scratch/hf, the container's
# local disk (ops/Dockerfile.cuda:161), then mmapped from there. The 14B SIGBUS
# (campagne-14b-qualite-2026-08-10.txt:5-8, 16) came from mmapping safetensors ON THE
# BUCKET MOUNT; nothing here mmaps from /out. A reads the sealed file, which the
# llvq-artifact reader does not mmap (same journal, lines 7-8).
#
# Cost, estimated from vod-8b (job 6aad4666, `hf jobs inspect`):
#   arm A segment there: 32.6 min = container start + oracle + 4.32 GB sealed load
#     + 1,768 s of scoring (out-tetra.txt:64)
#   arm B segment there: 36.2 min = Qwen3-8B Hub download + int4 restore + 1,764 s
#   here: A 32.7 + sha256 over the mount ~1.5 + B 36.2 + C 36.2 = ~107 min
#   => ~$3.20 on l40sx1 at $1.80/h, range [100, 115] min = [$3.00, $3.45]
#   timeout 135m (integrator, to hold the $20 cap): +17 % over the top of the range;
#   a cut can only land in arm C, which runs last => worst case $4.05
# Dense f16 scoring does not depend on where the weights came from: 1,258 s (4B f16
# checkpoint) against 1,253 s (4B sealed DCLM), 248 s against 248 s at 8B limit=40.
#
# DRY_RUN=1 bash census3.sh   assembles the job script, syntax-checks it, launches nothing.
set -euo pipefail

REPO=$HOME/Documents/Pro/workspace/poc/llvq
D=${CENSUS_DATE:-2026-09-21}                      # launch date; names prereg and output dir
PREREG=proofs/preregistration-census-8b-$D.md
BUCKET=Pier-Jean/jobs-artifacts
OBJ_DIR=dclm-8b-2026-09-21                        # the object keeps its encoding date
LOCAL=$HOME/qwen3-8b-dclm.bin
SUMS=$HOME/q8b-dclm-2026-09-21/files.sha256
O=/out/census-8b-$D
F=/out/$OBJ_DIR/qwen3-8b-dclm.bin
FP=a74a6d6213602979                               # census plan fingerprint, every FULL dump
DRY=${DRY_RUN:-0}

cd "$REPO"

# ---- preflight on the Mac, $0 ----------------------------------------------------------
if [ "$DRY" = 1 ]; then
  SHA=0000000000000000000000000000000000000000000000000000000000000000
  BYTES=4364205777                                # the prereg's predicted size, dry run only
else
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  test -s "$SUMS" || { echo "refused: $SUMS missing, the seal step did not finish" >&2; exit 1; }
  SHA=$(awk '$2 ~ /qwen3-8b-dclm\.bin$/ {print $1}' "$SUMS")
  [ ${#SHA} -eq 64 ] || { echo "refused: no sha256 for qwen3-8b-dclm.bin in $SUMS" >&2; exit 1; }
  BYTES=$(stat -f %z "$LOCAL")
  REMOTE=$(hf buckets ls "hf://buckets/$BUCKET/$OBJ_DIR/" 2>/dev/null | awk '$NF ~ /qwen3-8b-dclm\.bin$/ {print $1}' || true)
  [ "$REMOTE" = "$BYTES" ] || { echo "refused: bucket copy is ${REMOTE:-absent} B, local $BYTES B" >&2; exit 1; }
  if hf buckets ls "hf://buckets/$BUCKET/census-8b-$D/" 2>/dev/null | grep -q 'mmlu-8b-'; then
    echo "refused: census-8b-$D/ already holds dumps; pick another CENSUS_DATE" >&2; exit 1
  fi
  echo "sealed base: $BYTES B, sha256 $SHA, bucket copy $REMOTE B"
  # Provenance the image cannot print (COMMIT stays in the build stage): the Space
  # revision and the Hub revisions the unpinned B and C arms will resolve to.
  uv run --with huggingface_hub python -c '
from huggingface_hub import HfApi
a = HfApi()
s = a.space_info("Pier-Jean/llvq-runner-cuda")
print("image  Pier-Jean/llvq-runner-cuda", s.sha, s.last_modified)
for r in ("Qwen/Qwen3-8B", "Pier-Jean/qwen3-8b-awq-deq"):
    i = a.model_info(r)
    print("model ", r, i.sha, "private" if i.private else "public")
' | tee "/tmp/census-8b-$D-provenance.txt"
fi

# ---- the job ----------------------------------------------------------------------------
# Values fixed on the Mac, quoted for the job shell.
PRE=$(printf 'O=%q\nF=%q\nSHA=%q\nBYTES=%q\nFP=%q' "$O" "$F" "$SHA" "$BYTES" "$FP")

# Quoted heredoc: nothing below expands on the Mac. `read -d ''` rather than
# $(cat <<'JOB'): the Mac's /bin/bash is 3.2, which mis-parses a heredoc inside $( ).
# ops/run.py prepends `set -euo pipefail` and runs the whole as ['bash', '-lc', script]
# through the API (ops/run.py:1091-1098).
IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
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
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee "$O/gpu.txt" || true
echo '== oracle (hard rule 10) =='
oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle-cuda.txt" | tail -2
echo '== the sealed base on the mount: bytes and sha256 against the Mac =='
test "$(stat -c %s "$F")" = "$BYTES"
sha256sum "$F" | tee "$O/files.sha256"
test "$(cut -d' ' -f1 "$O/files.sha256")" = "$SHA"
echo '== arm A: DCLM base 8B, sealed, dense reconstruction, FULL split ==' ; date -u
LLVQ_MMLU_ALLOC=flat LLVQ_MMLU_DUMP="$O/mmlu-8b-dclm-FULL.csv" mmlu "$F" cuda 2>&1 | tee "$O/out-dclm.txt" | tail -10
date -u
check "$O/mmlu-8b-dclm-FULL.csv"
echo '== arm B: Qwen3-8B f16 checkpoint, FULL split ==' ; date -u
LLVQ_MMLU_ALLOC=flat LLVQ_MMLU_DUMP="$O/mmlu-8b-f16-FULL.csv" mmlu Qwen/Qwen3-8B cuda 2>&1 | tee "$O/out-f16.txt" | tail -10
date -u
check "$O/mmlu-8b-f16-FULL.csv"
echo '== arm C: AWQ w4 g128 dequantized to f16, FULL split ==' ; date -u
LLVQ_MMLU_ALLOC=flat LLVQ_MMLU_DUMP="$O/mmlu-8b-awq-FULL.csv" mmlu Pier-Jean/qwen3-8b-awq-deq cuda 2>&1 | tee "$O/out-awq.txt" | tail -10
date -u
check "$O/mmlu-8b-awq-FULL.csv"
echo '== dumps ==' ; wc -l "$O"/mmlu-8b-*-FULL.csv ; ls -la "$O"
JOB

if [ "$DRY" = 1 ]; then
  # What ops/run.py:1091 will hand to ['bash', '-lc', ...]: printed and parsed, not run.
  JOBSCRIPT="set -euo pipefail
$PRE
$BODY"
  printf '%s\n' "$JOBSCRIPT"
  printf '%s\n' "$JOBSCRIPT" | bash -n && echo "DRY_RUN: job script parses; nothing launched" >&2
  exit 0
fi

uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor l40sx1 --timeout 135m \
  --bucket "$BUCKET" --out-mount /out \
  --name census-8b \
  "$PRE" "$BODY"
