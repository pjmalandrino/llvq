#!/usr/bin/env bash
# The paper's calibration corpus at our volume, preregistered:
#   proofs/preregistration-dclm-4b-2026-09-18.md
#   sha256 fdef055c2605fb355c6014e6e9ccd92041c4cf17f14d403cfc709fa5164a6fd2
#
# One variable: C4 -> DCLM-edu. Same tetra1, same rotation seed, same nogs,
# same h_shrink 1, same 131,072 tokens. The encoding is done (1 h 47 of Mac).
#
# One arm. This is a RE-ENCODING comparison, where the measured noise is
# 2.92 pp against 0.43 at constant file, so one draw resolves only 8.1 pp. The
# prereg registers no significance claim, only a point and three readings.
#
# Cost announced before the go: about 24 min on l40sx1, $0.72, timeout 1 h.
set -euo pipefail
O=/out/dclm-4b-2026-09-18
F=$O/qwen3-4b-dclm.bin
uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor l40sx1 --timeout 1h \
  --bucket Pier-Jean/jobs-artifacts --out-mount /out \
  --name dclm-4b \
  'echo "== oracle (hard rule 10) ==" ; oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tail -2' \
  "echo '== dclm-calibrated Q5 recipe, FULL split ==' ; date ; \
   LLVQ_MMLU_ALLOC=flat LLVQ_MMLU_DUMP=$O/mmlu-4b-dclm-FULL.csv \
   mmlu $F cuda 2>&1 | tee $O/out-dclm.txt | tail -10 ; date" \
  "echo '== dump ==' ; wc -l $O/mmlu-4b-dclm-FULL.csv"
