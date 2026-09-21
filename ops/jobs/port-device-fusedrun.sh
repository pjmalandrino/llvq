#!/usr/bin/env bash
# What the device port cost on a card, two arms in one process.
#
# Prereg: proofs/preregistration-port-device-2026-09-21.md, stamped
# sha256 9d4d6b99401c5183..., before this ran.
#
# The reference is docs/mesures/dclm-ft-fusedrun-2026-09-20.txt: 98.3 tok/s
# [97.5, 98.6] in 1.39 GB, job 6aaf855552d0dbd7f1d72e3b, l40sx1. That run did
# NOT pin the tile and its journal records 128. The served default has since
# moved to the measured row, which is 64 on sm_89, so re-running the reference
# script verbatim would move for two reasons at once.
#
#   arm A  LLVQ_TILE_BLOCKS=128   comparable trait for trait to 98.3
#   arm B  unset, so 64           the number to publish after this lot
#
# Everything else is held: same image, same file, same flags. `oracle` first,
# per hard rule 10.
set -euo pipefail
O=/out/port-device-2026-09-21
F=/out/dclm-ft-2026-09-19/qwen3-4b-dclm-ft.bin
FLAGS="LLVQ_FUSED_LAYOUT=tetra48 LLVQ_EMBED=q8 LLVQ_ROT_SHARE=1 LLVQ_FUSE=0 LLVQ_KV=f16"

uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor l40sx1 --timeout 1h \
  --bucket Pier-Jean/jobs-artifacts --out-mount /out \
  --name port-device-fusedrun \
  "mkdir -p $O" \
  'echo "== oracle (hard rule 10) ==" ; oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tail -2' \
  "echo '== the binary carries the port ==' ; \
   strings \$(command -v fusedrun) | grep -c 'device projection' || true" \
  "echo '== ARM A: tile pinned at 128, the reference tile ==' ; date ; \
   LLVQ_TILE_BLOCKS=128 $FLAGS \
   fusedrun $F 2>&1 | tee $O/arm-a-tile128.txt | tail -30 ; date" \
  "echo '== ARM B: tile unset, so the measured row for sm_89 ==' ; date ; \
   $FLAGS \
   fusedrun $F 2>&1 | tee $O/arm-b-default.txt | tail -30 ; date" \
  "echo '== raw kept ==' ; wc -l $O/arm-a-tile128.txt $O/arm-b-default.txt"
