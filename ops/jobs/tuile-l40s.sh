#!/usr/bin/env bash
# The tile on the SERVED path, on an L40S. Preregistered:
#   proofs/preregistration-tuile-l40s-2026-09-20.md, sha256 13474e574b939f6533f9d964...
#
# tile.rs ships the mechanism and refuses the policy: its sm_89 row comes from
# f1rankfloor, a synthetic bench, and the module names promoting a synthetic
# optimum to a served default as "the class of error this repository keeps
# catching". F1d swept the real path on sm_120 only. This is sm_89.
#
# Cost: about $0.30 on l40sx1, timeout 1 h.
set -euo pipefail
O=/out/tuile-l40s-2026-09-20
B=/out/ball-ref-2026-09-20/qwen3-4b-llvq.bin
F=/out/dclm-ft-2026-09-19/qwen3-4b-dclm-ft.bin
ARMS=fp16,planes14,nullk,tetra48
uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor l40sx1 --timeout 1h \
  --bucket Pier-Jean/jobs-artifacts --out-mount /out \
  --name tuile-l40s \
  "mkdir -p $O" \
  'nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader' \
  "for T in 128 64 32 ; do \
     echo \"== tile \$T ==\" ; \
     LLVQ_TILE_BLOCKS=\$T LLVQ_BENCH_ARMS=$ARMS planesbench $B $F 2>&1 \
       | tee -a $O/sweep.txt | grep -aE 'tile |Tetra48|Planes14|floor \(nullk\)' ; \
   done" \
  "echo '== raw kept ==' ; wc -l $O/sweep.txt"
