#!/usr/bin/env bash
# DRAFT, never compiled or run. Skeleton of ops/jobs/prefill-ab.sh for prereg-prefill-1.md,
# modelled on ops/jobs/served-grouped.sh and gqa-ab.sh (read-only, HEAD 06e1602). Not launched:
# a launch needs the operator's go, a stamped prereg and a cap (CLAUDE.md rule 1).
#
# Revision 2 (2026-11-02): the object is read where jobs read it (/out, the bucket mount, as
# served-grouped.sh:36), not /mnt/bucket; step 2b replays the SERVED door under the grouped config,
# the path the AttnRows dispatch touches; the base commit is the operator's choice (prereg §3).
#
# Gates of the prereg (§4), in order; any failure exits non-zero and names the line it wanted.
set -euo pipefail
O=${O:-/out/prefill-ab-4b-$(date -u +%F)${RETRY:+-r$RETRY}}
mkdir -p "$O"
F=/out/sealed-4b-2026-09-23/qwen3-4b-sealed.bin
SHA=886391a8...            # the full digest is pinned in the stamped prereg, as served-grouped.sh
BYTES=1418224685
CG=$O/qwen3-4b-tetra-e4-grouped.json   # written by the launcher from configs/, sha256 2455fe85...
REF_LINE_SHA=8784ffc423bfbeeaeb9a92abd426573cc45f9a083ba56e61e52d9ff40e3b018f   # bench door, paper-table run 1
SERVED_TEXT_SHA=0597cee8b20befecdf92c2f3f2f1938d8216bd244542faab42d2637920a3c12a # served door, whole text
# (docs/mesures/served-grouped-2026-11-02-brut/lines.sha256: the same bytes under both configs)
TILE='tile 64'
PARITY='int4: tv_q4_hh equals tv_q4_h bit for bit'
need()  { grep -qF -- "$2" "$1" || { echo "refused: $1 lacks: $2" >&2; exit 1; }; }

nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader | tee "$O/gpu.txt"
# Second memory channel (the first is the pool high-water printed by the binary): 100 ms samples.
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 100 > "$O/mem.csv" &
SAMPLER=$!
trap 'kill "$SAMPLER" 2>/dev/null || true' EXIT

# Gate 2: object bytes and digest; the config at its digest.
test "$(stat -c %s "$F")" = "$BYTES" || { echo "refused: object size" >&2; exit 1; }
sha256sum "$F" "$CG" | tee "$O/files.sha256"   # compared against the pinned digests, as served-grouped.sh:94-101

# Gate 1: oracle first, rule 10.
oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle-cuda.txt" | tail -2
need "$O/oracle-cuda.txt" "MATCH"

# Gate 4: the bench door replayed, dense arm resident, reference line.
env LLVQ_FUSED_LAYOUT=tetra48 LLVQ_EMBED=q4 LLVQ_ROT_SHARE=1 LLVQ_FUSE=0 \
    fusedrun "$F" 256 2>&1 | tee "$O/bench-q4-256.txt"
need "$O/bench-q4-256.txt" 'fused F0: 256 tokens identical to the dense arm'
need "$O/bench-q4-256.txt" "$PARITY"
test "$(grep -cF '  fused F0:  ' "$O/bench-q4-256.txt")" = 1
test "$(grep -F '  fused F0:  ' "$O/bench-q4-256.txt" | sha256sum | cut -d' ' -f1)" = "$REF_LINE_SHA"

# Gate 4b: the served door under the grouped config, the path whose attention gains the dispatch
# (model.rs:1648-1657, unexpanded branch). Its whole decoded text must be the measured bytes.
env LLVQ_CONFIG="$CG" fusedrun "$F" 256 2>&1 | tee "$O/served-grouped-256.txt" | tail -20
need "$O/served-grouped-256.txt" "$TILE"
need "$O/served-grouped-256.txt" "$PARITY"
grep -E '^served config: ' "$O/served-grouped-256.txt" | grep -qF 'attention grouped on grow(256)'
need "$O/served-grouped-256.txt" 'No dense arm on this path'
test "$(grep -cE '^served : ' "$O/served-grouped-256.txt")" = 1
awk '/^served : /{f=1;next} /No dense arm on this path/{f=0} f' "$O/served-grouped-256.txt" > "$O/line-grouped.txt"
test "$(sha256sum < "$O/line-grouped.txt" | cut -d' ' -f1)" = "$SERVED_TEXT_SHA" \
  || { echo "refused: the served grouped text moved" >&2; exit 1; }

# Step 3: the measurement mode, served flags spelled out, nothing else in the environment.
env LLVQ_FUSED_LAYOUT=tetra48 LLVQ_EMBED=q4 LLVQ_ROT_SHARE=1 LLVQ_FUSE=0 \
    LLVQ_PREFILL_AB=2048,4096,8192 \
    fusedrun "$F" 256 2>&1 | tee "$O/prefill-ab.txt"
need "$O/prefill-ab.txt" "cudarc event tracking: off (one stream, ordered by issue)"
need "$O/prefill-ab.txt" "comparator flags a planted one-ulp change"      # gate 6
need "$O/prefill-ab.txt" "identity grid: seed "                             # the grid's seed is printed
need "$O/prefill-ab.txt" "$TILE"                                            # gate 7
need "$O/prefill-ab.txt" "fp64 probe: "                                     # step 3e printed

kill "$SAMPLER" 2>/dev/null || true
awk -F, '{gsub(/ /,"",$2); if ($2+0 > m) m = $2+0} END {printf "nvidia-smi peak %d MiB\n", m}' "$O/mem.csv" | tee "$O/peaks.txt"
sha256sum "$O"/*.txt > "$O/outputs.sha256"
# The bucket keeps $O; the journal and the jobs.csv row are written after the run, with the
# running total of spend (rule 1).
#
# RIDE-ALONG variant (V3-lite, prefill.md §5): if the operator folds steps 3a, 3c, 3d and 3e into
# his next scheduled card job for audit steps 2-4, this script is not launched; that job appends
# `LLVQ_PREFILL_AB=micro` (3a + 3c + 3d + 3e only, no timed prefill), about 3 min of card time.
