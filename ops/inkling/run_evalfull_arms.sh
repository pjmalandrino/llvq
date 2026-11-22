#!/bin/bash
# The whole-model arms of Inkling 2 Mini against the native teacher, one heavy job per (arm, window range).
#
#   ops/inkling/run_evalfull_arms.sh ATTN_DIR [ARMS...]
#   ATTN_DIR: where B's inkwh wrote the attention reconstructions (searched recursively for
#             model.layers.{L}.attn.{wq_du|wk_dv|wv_dv|wo_ud|gate_attn|wr_du}.weight.f32)
#   ARMS default: experts24 experts24_attn whole rival_iq2xxs rival_q2kxl
#
# Each job goes through ~/.llvq-guard/heavy.sh, 45 s apart so the other users can take the lock. Windows already
# done are skipped, so the script can be rerun after a kill. An arm whose inputs are incomplete is skipped with its
# reason (evalfull_arms.py check). Our arms run windows 0-3 and 4-7 as two jobs; a rival arm runs 0-7 in one job,
# because its cost is the per-visit GGUF dequantization, which a split would pay twice. Ends with the report.
set -uo pipefail
ATTN="${1:?usage: run_evalfull_arms.sh ATTN_DIR [ARMS...]}"; shift
ARMS=("$@"); [ ${#ARMS[@]} -eq 0 ] && ARMS=(experts24 experts24_attn whole rival_iq2xxs rival_q2kxl)
cd "$(dirname "$0")"
export INKLING_THREADS="${INKLING_THREADS:-4}"
for arm in "${ARMS[@]}"; do
  if [[ "$arm" == rival_* ]]; then ranges=(0-7); else ranges=(0-3 4-7); fi
  for w in "${ranges[@]}"; do
    echo "$(date '+%T') $arm windows $w"
    ~/.llvq-guard/heavy.sh "lot1-A: evalfull arm $arm windows $w (MPS)" \
      uv run -q evalfull_arms.py run "$arm" --windows "$w" --attn "$ATTN" || echo "$(date '+%T') $arm $w: not run or failed"
    sleep 45
  done
done
INKLING_THREADS=2 uv run -q evalfull_arms.py report
