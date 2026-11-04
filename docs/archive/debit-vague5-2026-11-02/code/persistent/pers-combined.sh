#!/usr/bin/env bash
# DRAFT, never run. Written read-only on 2026-11-02 for the persistent-kernel lead
# (wave5-deep/persistent.md, step L1; wave5-deep/prereg-persistent-0.md), revised
# after the adversarial review: this script is now the FIRST paid step, the cheap
# one (persall-Tetra, the fixed Tetra fusion check, the microbenchmarks, the
# boundary A/B, the driver cubins). The persistent arm set moved to pers-arms.sh
# (step L2, prereg-persistent-1.md). Sketch of ops/jobs/pers-combined.sh, on the
# model of ops/jobs/banc-252.sh and served-grouped.sh.
#
#   DRY_RUN=1 bash ops/jobs/pers-combined.sh                 prints and parses the job script
#   IMAGE_SHA=<space sha> bash ops/jobs/pers-combined.sh      launches, after the operator's go
#
# One job, l40sx1, two processes. Cost, estimated (l40sx1 $1.80/h, median of 107
# priced jobs of 5 min or more in docs/data/jobs.csv): about 15 min, $0.45.
# Timeout 90m: METHODE section 6 asks it of any planesbench with five arms or more,
# and this one times five (slot32,planes14,fp16,nullk,tetra48). Worst case $2.70.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-pers-l1-2026-11-XX.md        # name fixed when stamped
BUCKET=Pier-Jean/jobs-artifacts
B=/out/ball-ref-2026-09-20/qwen3-4b-llvq.bin                 # 1,770,527,533 B in the bucket listing
F=/out/tetra-4b-2026-09-06/qwen3-4b-tetra.bin                # 1,770,529,149 B, the bare 252-Tetra encode
S=/out/sealed-4b-2026-09-23/qwen3-4b-sealed.bin              # 1,418,224,685 B, sha256 886391a8...
CONF_GRP_LOCAL=configs/qwen3-4b-tetra-e4-grouped.json        # the served door, for its digest line
CONF_GRP_SHA=$(shasum -a 256 "$CONF_GRP_LOCAL" | cut -d' ' -f1)
O=/out/pers-l1-2026-11-XX${RETRY:+-r$RETRY}
CG=$O/qwen3-4b-tetra-e4-grouped.json                         # written from the launch command, served-grouped.sh:82-101
TIMEOUT=90m
DRY=${DRY_RUN:-0}

if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  IMAGE_SHA=${IMAGE_SHA:?IMAGE_SHA=<the Space sha of the rebuild that carries the L1 code>}
fi

read -r -d '' SCRIPT <<EOS || true
set -euo pipefail
mkdir -p $O $O/cubin
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,clocks.max.sm --format=csv,noheader | tee $O/gpu.txt
sha256sum $B $F $S | tee $O/files.sha256
# Gate 1: rule 10, oracle first.
oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee $O/oracle.txt | tail -3
grep -q 'MATCH' $O/oracle.txt || { echo 'gate 1: oracle did not print MATCH' >&2; exit 1; }

# Process 1: the bench. Five table arms (A4 needs slot32 AND planes14,
# planesbench.rs:3861), the Tetra fusion with its stream built from the Tetra
# file and a non-fatal check, the persall port, the microbenchmarks, the cubins.
LLVQ_TILE_BLOCKS=64 \
LLVQ_BENCH_ARMS=slot32,planes14,fp16,nullk,tetra48 \
LLVQ_SEG_TETRA=1 LLVQ_TETRA_OCC=1 LLVQ_PERS_MICRO=1 LLVQ_DUMP_CUBIN=$O/cubin \
  planesbench $B $F 2>&1 | tee $O/bench.txt | tail -80
grep -q 'tetra48: 252 of 252 matrices matched by name' $O/bench.txt || { echo 'gate 3: Tetra match line missing' >&2; exit 1; }
grep -q '^tile 64' $O/bench.txt || { echo 'gate 3: tile 64 line missing' >&2; exit 1; }

# Process 2a: the served door, 16 tokens, for the digest line it prints. The config
# reaches the job inside the launch command (served-grouped.sh:82-101 passes it
# through printf %q) and is checked by sha256 before use.
test "\$(sha256sum $CG | cut -d' ' -f1)" = $CONF_GRP_SHA || { echo 'refused: config bytes' >&2; exit 1; }
LLVQ_CONFIG=$CG fusedrun $S 16 2>&1 | tee $O/served-16.txt | tail -5
# Process 2b: the boundary A/B on the bench door. The served flags spelled out;
# the mode turns the events off and sets the grouped form itself.
LLVQ_FUSED_LAYOUT=tetra48 LLVQ_EMBED=q4 LLVQ_ROT_SHARE=1 LLVQ_FUSE=0 \
LLVQ_BOUNDARY_AB=1 \
  fusedrun $S 256 2>&1 | tee $O/boundary.txt | tail -30
# Gate 4, the four lines, each from where it is printed.
grep -q 'int4: tv_q4_hh equals tv_q4_h bit for bit' $O/boundary.txt || { echo 'gate 4: int4 parity line missing' >&2; exit 1; }
grep -q 'cudarc event tracking: off (one stream, ordered by issue)' $O/boundary.txt || { echo 'gate 4: events not off' >&2; exit 1; }
grep -q '^attention grouped on grow(256)' $O/boundary.txt || { echo 'gate 4: attention line missing' >&2; exit 1; }
D_SERVED=\$(grep -o 'NVRTC source: [0-9]* bytes, sha256 [0-9a-f]*' $O/served-16.txt | head -1)
D_BENCH=\$(grep -o 'NVRTC source: [0-9]* bytes, sha256 [0-9a-f]*' $O/boundary.txt | head -1)
echo "served door: \$D_SERVED" | tee $O/digests.txt
echo "bench door : \$D_BENCH"  | tee -a $O/digests.txt
[ -n "\$D_SERVED" ] && [ "\$D_SERVED" = "\$D_BENCH" ] || { echo 'gate 4: the two doors compiled different units' >&2; exit 1; }
ls -l $O $O/cubin
EOS

if [ "$DRY" = 1 ]; then printf '%s\n' "$SCRIPT" | bash -n && echo "DRY_RUN: parses; nothing launched" >&2; printf '%s\n' "$SCRIPT"; exit 0; fi
# Launch line to be copied from ops/jobs/banc-252.sh (lines 85-90 today), through
# the API with ['bash', '-lc', script] and the identity assert against
# `hf jobs inspect` (METHODE section 6):
#   uv run ops/run.py bench --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
#     --flavor l40sx1 --timeout "$TIMEOUT" --bucket "$BUCKET" --out-mount /out ...
# with the Space sha checked against IMAGE_SHA first, as served-grouped.sh does.
# The config file is uploaded beside the job as served-grouped.sh does with CE/CG.
echo "sketch only: the launch line is deliberately left out" >&2; exit 1
