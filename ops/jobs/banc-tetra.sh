#!/usr/bin/env bash
# The ten-arm bench with Tetra in it, one process. Preregistered:
#   proofs/preregistration-banc-tetra-2026-09-20.md, sha256 95fb0166d0736b1b9d284185...
#
# The WHOLE table is re-measured, not just the new row: docs/data/README.md
# records that adding a row measured in another process is forbidden, and that
# is why the August run re-measured everything when QTIP arrived.
#
# QTIP is NOT here: arms.rs carries HAS_KERNEL[qtip] = false, so naming it is
# refused. Stated in section 2 of the prereg with its consequence.
#
# Cost: a few minutes on l40sx1, about $0.30, timeout 1 h.
set -euo pipefail
O=/out/banc-tetra-fuse-2026-09-20
B=/out/ball-ref-2026-09-20/qwen3-4b-llvq.bin   # ball arms
F=/out/dclm-ft-2026-09-19/qwen3-4b-dclm-ft.bin  # tetra48, SECOND argument
P1=slot32,planes14,planes12x,golay70v1,fp16,awq,golay70v2,cublasf16,nullk
uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor l40sx1 --timeout 1h \
  --bucket Pier-Jean/jobs-artifacts --out-mount /out \
  --name banc-tetra-fuse \
  "mkdir -p $O" \
  'nvidia-smi --query-gpu=name,memory.total --format=csv' \
  "echo '== the ten arms, phase 2 adds tetra48 ==' ; date ; \
   LLVQ_BENCH_ARMS='$P1;$P1,tetra48' \
   planesbench $B $F 2>&1 | tee $O/banc.txt | tail -40 ; date" \
  "echo '== raw kept ==' ; wc -l $O/banc.txt"
