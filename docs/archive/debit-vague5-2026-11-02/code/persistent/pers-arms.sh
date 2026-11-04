#!/usr/bin/env bash
# DRAFT, never run. Written read-only on 2026-11-02 for the persistent-kernel lead
# (wave5-deep/persistent.md, step L2; wave5-deep/prereg-persistent-1.md), after the
# adversarial review split the old combined job in two: pers-combined.sh is the cheap
# first step (L1); this one times the persistent arm set, and only runs if L1's
# reading recommends it and the operator lifts the frozen settings (decision 8).
#
#   DRY_RUN=1 bash ops/jobs/pers-arms.sh                     prints and parses the job script
#   IMAGE_SHA=<space sha> bash ops/jobs/pers-arms.sh          launches, after the operator's go
#
# One job, l40sx1, one process. Cost, estimated: about 12 min, $0.36. Timeout 90m
# (five table arms, METHODE section 6): $2.70 at worst.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-pers-arms-2026-11-XX.md      # name fixed when stamped
BUCKET=Pier-Jean/jobs-artifacts
B=/out/ball-ref-2026-09-20/qwen3-4b-llvq.bin
F=/out/tetra-4b-2026-09-06/qwen3-4b-tetra.bin
O=/out/pers-arms-2026-11-XX${RETRY:+-r$RETRY}
TIMEOUT=90m
DRY=${DRY_RUN:-0}
ARMS=pnull,pword,pl1,psmem,psmem_s25,psmem_t28,psmem_t28_u2,pt28_512,pt28_1024,pchain_w,pchain_w2,pchain_wp

if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  IMAGE_SHA=${IMAGE_SHA:?IMAGE_SHA=<the Space sha of the rebuild that carries the persistent arms>}
fi

read -r -d '' SCRIPT <<EOS || true
set -euo pipefail
mkdir -p $O $O/cubin
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,clocks.max.sm --format=csv,noheader | tee $O/gpu.txt
sha256sum $B $F | tee $O/files.sha256
oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee $O/oracle.txt | tail -3
grep -q 'MATCH' $O/oracle.txt || { echo 'gate 1: oracle did not print MATCH' >&2; exit 1; }
# The table (control), A4 with the Tetra fusion as the route-D reference, the
# served unit's tv_tetra48_h as the identity and time reference (tetra48h, its
# digest printed), and the twelve persistent arms in registration order.
LLVQ_TILE_BLOCKS=64 \
LLVQ_BENCH_ARMS=slot32,planes14,fp16,nullk,tetra48 \
LLVQ_SEG_TETRA=1 LLVQ_PERS_ARMS=$ARMS LLVQ_DUMP_CUBIN=$O/cubin \
  planesbench $B $F 2>&1 | tee $O/bench.txt | tail -100
grep -q 'tetra48: 252 of 252 matrices matched by name' $O/bench.txt || { echo 'gate 3: Tetra match line missing' >&2; exit 1; }
grep -q '^tile 64' $O/bench.txt || { echo 'gate 3: tile 64 line missing' >&2; exit 1; }
grep -q 'served unit: 101625 bytes, sha256 6193094310bc' $O/bench.txt || { echo 'gate 3: tetra48h is not the served unit' >&2; exit 1; }
ls -l $O $O/cubin
EOS

if [ "$DRY" = 1 ]; then printf '%s\n' "$SCRIPT" | bash -n && echo "DRY_RUN: parses; nothing launched" >&2; printf '%s\n' "$SCRIPT"; exit 0; fi
echo "sketch only: the launch line is deliberately left out (copy banc-252.sh's)" >&2; exit 1
# Note: the served-unit digest above is the one of HEAD 06e1602. If the operator's
# branch changes the served unit before this job, the prereg names the new digest,
# recomputed offline from the tree with ptx_probe.py's recipe, before the stamp.
