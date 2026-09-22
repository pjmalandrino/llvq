#!/usr/bin/env bash
# Trains the row scales (or, with MODE=row_norms, the row scales and the norms)
# on a card, sizing itself from a measured rate.
#
# Preregistered: proofs/preregistration-tetranu-rowscales-2026-09-19.md
# Deviations:    same name, -ECARTS.md
#
# The step count is NOT passed in. On 2026-09-19 a run was priced from a
# cross-entropy measurement and came out three times slower than planned, so
# this script measures six steps on the card it actually got and derives the
# count from that. The wall budget is the input; the token count is the output.
set -euo pipefail

EXPORT=${EXPORT:?the exported f16 checkpoint}
OUT=${OUT:?where sigma and the journal go}
TEACHER=${TEACHER:-Qwen/Qwen3-4B}
BUDGET=${BUDGET:-7200}          # seconds of training, loading excluded
SEQ=${SEQ:-1024}
BATCH=${BATCH:-2}
LR=${LR:-3e-4}
PROBE=${PROBE:-6}
MODE=${MODE:-row_scales}        # row_norms adds the RMSNorm weights, 0 bits

mkdir -p "$OUT"
COMMON=(--student "$EXPORT" --teacher "$TEACHER"
        --mode "$MODE" --objective kl
        --seq-len "$SEQ" --batch-size "$BATCH"
        --device cuda --dtype bf16 --seed 0)

echo "== probe: $PROBE steps, to read this card's rate =="
python -m llvqtune "${COMMON[@]}" \
  --steps "$PROBE" --lr 1e-8 --warmup 1 \
  --journal "$OUT/probe.jsonl" --out "$OUT/probe.json" || true

RATE=$(python - "$OUT/probe.jsonl" <<'PY'
import json, sys
rate = None
for line in open(sys.argv[1]):
    r = json.loads(line)
    if r.get("event") == "closed":
        rate = r.get("seconds_per_step")
print(rate if rate else "")
PY
)
if [ -z "$RATE" ]; then
  echo "the probe wrote no rate; refusing to guess a step count"
  exit 2
fi

STEPS=$(python -c "print(max(1, int($BUDGET / $RATE)))")
TOKENS=$(python -c "print($STEPS * $SEQ * $BATCH)")
echo "== measured $RATE s a step, so $STEPS steps and $TOKENS tokens in $BUDGET s =="

echo "== training =="
date
python -m llvqtune "${COMMON[@]}" \
  --steps "$STEPS" --lr "$LR" --warmup 100 \
  --checkpoint-every 200 \
  --journal "$OUT/journal.jsonl" --out "$OUT/sigma.json"
date
ls -la "$OUT"
