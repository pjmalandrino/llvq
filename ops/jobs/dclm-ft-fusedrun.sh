#!/usr/bin/env bash
# The fine-tuned object on a card, through the served kernel, against its own
# dense arm in the same process. This is the check every quality cell of
# docs/table-formats-2026-09-20.md has been carrying as a caveat.
#
# Flags are the served config spelled out (configs/qwen3-4b-tetra-q5.json:
# layout tetra48, embed q8, rot_share 1, fuse 0, kv f16) rather than
# LLVQ_CONFIG, because the config puts fusedrun on a one-arm path with no dense
# reference and the dense reference is the point.
#
# Reference, same binary and same card, on the 2026-09-10 object (F1e section 0):
#   fused ROT_SHARE=1  100.8 tok/s, 1.39 GB · dense f16 43.0 · 256 tokens identical.
#
# Cost: a few minutes on l40sx1, about $0.30, timeout 1 h.
set -euo pipefail
O=/out/dclm-ft-fusedrun-2026-09-20
F=/out/dclm-ft-2026-09-19/qwen3-4b-dclm-ft.bin
uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor l40sx1 --timeout 1h \
  --bucket Pier-Jean/jobs-artifacts --out-mount /out \
  --name dclm-ft-fusedrun \
  "mkdir -p $O" \
  'echo "== oracle (hard rule 10) ==" ; oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tail -2' \
  "echo '== fused against dense, same file, served flags ==' ; date ; \
   LLVQ_FUSED_LAYOUT=tetra48 LLVQ_EMBED=q8 LLVQ_ROT_SHARE=1 LLVQ_FUSE=0 LLVQ_KV=f16 \
   fusedrun $F 2>&1 | tee $O/fusedrun.txt | tail -30 ; date" \
  "echo '== raw kept ==' ; wc -l $O/fusedrun.txt"
