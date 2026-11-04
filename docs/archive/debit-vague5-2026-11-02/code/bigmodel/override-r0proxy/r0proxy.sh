#!/usr/bin/env bash
# DRAFT, never run. Written read-only on 2026-11-02 for lead "bigmodel"
# (wave5-deep/bigmodel.md §5, step V0d). Not part of the repository. A PAID
# JOB: it needs the operator's cap and go (CLAUDE.md rule 1), and a stamped
# prereg if any number from it is published. Informative only: two processes.
#
# Runs on TODAY's image (rotbench honours LLVQ_KERNEL_DIR, lib.rs:315-335),
# so no rebuild and no commit beyond this script. The job command carries the
# two files base64-encoded; the launcher checks, before encoding, that
# llvq_rot.cuh is byte-identical to the repository's (its sha256 goes in the
# journal) and the job prints both sha256 again from the card.
#
#   DRY_RUN=1 bash r0proxy.sh      prints the job script
#   bash r0proxy.sh                launches through ops/run.py bench, after the go
#
# Estimated (l40sx1, $1.80/h): setup 3 min, six rotbench runs of 20-40 s each,
# so 5-7 min, $0.15-0.21. Timeout 20 min, $0.60 at worst.
set -euo pipefail
REPO=/Users/pjmalandrino/Documents/Pro/workspace/poc/llvq
HERE=$(cd "$(dirname "$0")" && pwd)
CUH=$REPO/llvq-cuda/kernels/llvq_rot.cuh
ROT=$HERE/rotate.cu
CUH_SHA=$(shasum -a 256 "$CUH" | awk '{print $1}')
CUH_B64=$(base64 < "$CUH" | tr -d '\n')
ROT_B64=$(base64 < "$ROT" | tr -d '\n')
O=/out/r0proxy-$(date +%Y-%m-%d)

JOB=$(cat <<SCRIPT
set -euo pipefail
mkdir -p /tmp/r0proxy "$O"
echo "$CUH_B64" | base64 -d > /tmp/r0proxy/llvq_rot.cuh
echo "$ROT_B64" | base64 -d > /tmp/r0proxy/rotate.cu
sha256sum /tmp/r0proxy/* | tee "$O/sources.txt"
grep -q "^$CUH_SHA " "$O/sources.txt" || { echo "llvq_rot.cuh is not the repository's" >&2; exit 1; }
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem --format=csv | tee "$O/gpu.txt"
for r in 1 2 3; do
  rotbench > "$O/stock-\$r.txt" 2>&1
  LLVQ_KERNEL_DIR=/tmp/r0proxy rotbench > "$O/proxy-\$r.txt" 2>&1
  grep -q "SOURCES OVERRIDDEN" "$O/proxy-\$r.txt" || { echo "proxy run \$r not overridden" >&2; exit 1; }
  ! grep -q "SOURCES OVERRIDDEN" "$O/stock-\$r.txt" || { echo "stock run \$r overridden" >&2; exit 1; }
done
SCRIPT
)
if [ "${DRY_RUN:-0}" = 1 ]; then printf '%s\n' "$JOB"; exit 0; fi
echo "launch through ops/run.py bench with this script, l40sx1, timeout 20m, after the operator's go" >&2
exit 2   # deliberately not wired: the house launcher and its argument shape are the operator's

# Reading, done locally on the six files (pairs r = 1..3):
#   * identity: per width, the "rel" and "max coord" columns of proxy-r equal
#     stock-r's to the printed digit (same bits in, same arithmetic);
#   * ratio: per width, proxy-r µs / stock-r µs, then median and range over the
#     three pairs (rule 7); never a quotient of medians;
#   * read at 9,728 and 12,288 only; 17,408 is printed with the carveout caveat.
# Rule (proposal): proxy/stock <= 1.5 at 9,728 -> R0 stays the first wall-lift
# step; > 2.0 -> R1w goes first for the 32B; otherwise no change, V2 decides.
