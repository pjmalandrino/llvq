#!/usr/bin/env bash
# DRAFT, never run. Job script sketch for prereg-attention-1b (llama.cpp depth sweep on the L40S),
# an OPTIONAL complement to the FlashInfer anchor. Modelled on ops/jobs/paper-iq2-census.sh. Not
# repository code; it would go to ops/jobs/attn-depth-anchor.sh only after the operator's go, with
# the prereg stamped first.
#
#   DRY_RUN=1 bash ops/jobs/attn-depth-anchor.sh   prints and parses the job script
#   bash ops/jobs/attn-depth-anchor.sh             launches (needs the stamped prereg)
#
# The image tag `full-cuda` floats. At stamp time, read its digest without pulling it ($0, read-only):
#   TOKEN=$(curl -s 'https://ghcr.io/token?scope=repository:ggml-org/llama.cpp:pull' | jq -r .token)
#   curl -sI -H "Authorization: Bearer $TOKEN" \
#     -H 'Accept: application/vnd.oci.image.index.v1+json' \
#     https://ghcr.io/v2/ggml-org/llama.cpp/manifests/full-cuda | grep -i docker-content-digest
# and write it into DIGEST below and into the prereg. The job passes IMAGE@DIGEST; if HF Jobs
# refuses a digest reference, the prereg's fallback applies (tag, and the commit printed by
# `llama-bench --version` decides whether the "same kernel" reading holds).
#
# Known infra failure: this image failed twice at container start on 2026-09-25 (OCI runtime create
# failed, MOUNT_ATTR_IDMAP on /usr/bin/nvidia-cuda-mps-control, $0 each, docs/data/jobs.csv rows
# paper-iq2-census echec infra 1 and 2) and passed an hour later. Such a failure bills nothing and
# is retried later, not immediately.
#
# Inputs already in the bucket (hf buckets ls, 2026-11-02): m4-iq2-cuda-1b57b7d3/
#   qwen3-4b-iq2xxs.gguf  1,246,620,832 B  sha256 19a8ed49...
#   qwen3-4b-f16.gguf     8,051,284,928 B  sha256 259393f9...
# Cost, estimated: 3 to 5 min billed on l40sx1 at $1.80/h, $0.09 to $0.15; timeout 20 min, $0.60.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
PREREG=proofs/preregistration-attn-depth-anchor-YYYY-MM-DD.md
BUCKET=Pier-Jean/jobs-artifacts
DIR=attn-depth-anchor-YYYY-MM-DD
IMAGE=ghcr.io/ggml-org/llama.cpp
DIGEST=sha256:TO-BE-READ-AT-STAMP-TIME
M=/out/$DIR
GI=/out/m4-iq2-cuda-1b57b7d3/qwen3-4b-iq2xxs.gguf
GF=/out/m4-iq2-cuda-1b57b7d3/qwen3-4b-f16.gguf
SI=19a8ed4946353b6fdc5d19ba9766ffe903923cb7aefe848dcbcd5dba1de27605
SF=259393f93c7a55515161c1db0609118d8c1e94f78c01ad75f5d8ee0471cc0c3f
DRY=${DRY_RUN:-0}
if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  case "$DIGEST" in sha256:TO-BE-*) echo "refused: DIGEST not read at stamp time" >&2; exit 1 ;; esac
  if hf buckets ls "hf://buckets/$BUCKET/$DIR/" 2>/dev/null | grep -q 'depth-f16kv.jsonl'; then
    echo "refused: $DIR already holds a result" >&2; exit 1
  fi
fi
PRE=$(printf 'M=%q\nGI=%q\nGF=%q\nSI=%q\nSF=%q' "$M" "$GI" "$GF" "$SI" "$SF")
IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$M"
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,memory.total --format=csv | tee "$M/gpu.txt"
grep -q 'L40S' "$M/gpu.txt"
sha256sum "$GI" "$GF" | tee "$M/files.sha256"
test "$(sha256sum "$GI" | cut -d' ' -f1)" = "$SI"
test "$(sha256sum "$GF" | cut -d' ' -f1)" = "$SF"
B=$(command -v llama-bench || echo /app/llama-bench)
"$B" --version 2>&1 | tee "$M/version.txt"
grep -Eq 'version: [0-9]+ \([0-9a-f]{7,}\)' "$M/version.txt"
echo '== A: decode at depth, f16 KV, both GGUFs ==' ; date -u
"$B" -m "$GI" -m "$GF" -ngl 99 -fa 1 -p 0 -n 128 -d 0,2048,4096,8192 -r 5 -o jsonl \
  | tee "$M/depth-f16kv.jsonl"
echo '== B: decode at depth, q8_0 KV, IQ2_XXS ==' ; date -u
"$B" -m "$GI" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 0 -n 128 -d 0,2048,4096,8192 -r 5 -o jsonl \
  | tee "$M/depth-q8kv.jsonl"
echo '== C: prefill, f16 KV, both GGUFs ==' ; date -u
"$B" -m "$GI" -m "$GF" -ngl 99 -fa 1 -p 2048,8192 -n 0 -r 3 -o jsonl \
  | tee "$M/prefill.jsonl"
date -u
test "$(grep -c '"n_gen": 128' "$M/depth-f16kv.jsonl")" = 8
test "$(grep -c '"n_gen": 128' "$M/depth-q8kv.jsonl")" = 4
echo "anchor ok: $M"
JOB
if [ "$DRY" = 1 ]; then
  JOBSCRIPT="set -euo pipefail
$PRE
$BODY"
  printf '%s\n' "$JOBSCRIPT" | bash -n && echo "DRY_RUN: job script parses; nothing launched" >&2
  exit 0
fi
uv run ops/run.py bench \
  --image "$IMAGE@$DIGEST" \
  --flavor l40sx1 --timeout 20m \
  --bucket "$BUCKET" --out-mount /out \
  --name attn-depth-anchor \
  "$PRE" "$BODY"
# After the job: the slope of 1/tg against depth gives the KV read rate,
#   GB/s = bytes per position / slope, 147,456 B (f16 KV) or 78,336 B (q8_0 KV) at Qwen3-4B
#   (36 layers x 8 KV heads x 128 x 2; q8_0 is 34 B per 32 values, 8.5 bits per value).
# The JSONL rows carry per-repetition samples (samples_ns), from which ranges are formed.
# Then, at $0: read ggml/src/ggml-cuda/fattn.cu at the commit version.txt prints, and check that the
# Ada vector-kernel rule (cd8427d, lines 656-665) is unchanged before the "same kernel" reading.
