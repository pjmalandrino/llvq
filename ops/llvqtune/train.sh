#!/usr/bin/env bash
# Trains the row scales (or, with MODE=row_norms, the row scales and the norms)
# on a card, sizing itself from a measured rate.
#
# Preregistered: proofs/preregistration-tetranu-rowscales-2026-09-19.md
# Deviations:    same name, -ECARTS.md
#
# By default the step count is NOT passed in. On 2026-09-19 a run was priced
# from a cross-entropy measurement and came out three times slower than
# planned, so this script measures six steps on the card it actually got and
# derives the count from that. The wall budget is the input; the token count
# is the output.
#
# STEPS overrides that, for a run that must match another run's token count
# (the 8B against the 4B's 9,507 steps). The probe still runs: it is what
# reads the rate, the first KL and the peak memory, and a probe that did not
# close refuses the run whatever STEPS says.
#
# Optional guards, all off when unset:
#   STAGE              copy EXPORT there first and train from the copy. A
#                      bucket mount is not mmapped: SIGBUS at 14B,
#                      docs/mesures/campagne-14b-qualite-2026-08-10.txt:5
#   MAX_TRAIN_SECONDS  refuse when STEPS x the probe's rate exceeds it, rather
#                      than be killed by the job timeout at step 9,000
#   MAX_FIRST_KL       refuse when the probe's first loss exceeds it; the
#                      manual control of the 4B prereg (section 5), automated
set -euo pipefail

EXPORT=${EXPORT:?the exported f16 checkpoint}
OUT=${OUT:?where sigma and the journal go}
# No default. It was Qwen/Qwen3-4B, and an 8B student under a 4B teacher
# yields logits of the same shape: the run would have gone to the end.
TEACHER=${TEACHER:?the dense twin of the student, e.g. Qwen/Qwen3-8B}
BUDGET=${BUDGET:-7200}          # seconds of training, loading excluded
SEQ=${SEQ:-1024}
BATCH=${BATCH:-2}
LR=${LR:-3e-4}
PROBE=${PROBE:-6}
MODE=${MODE:-row_scales}        # row_norms adds the RMSNorm weights, 0 bits
STEPS_SET=${STEPS:-}
# The training text. `dclm` is what every published arm ran on; `mmlu-aux`
# is the task-format arm, `mix` interleaves the two at MIX_RATIO. The probe
# reads the same corpus as the run: a rate measured on other text is the
# error of 2026-09-19 in another costume.
CORPUS=${CORPUS:-dclm}
MIX_RATIO=${MIX_RATIO:-0.5}

# Checked before anything is loaded: a typo must not cost a probe.
positive_int() {
  case "$2" in
    ''|*[!0-9]*) echo "$1=$2 is not a positive integer"; exit 2 ;;
  esac
  if [ "$2" -le 0 ]; then echo "$1=$2 is not a positive integer"; exit 2; fi
}
[ -z "$STEPS_SET" ] || positive_int STEPS "$STEPS_SET"
[ -z "${MAX_TRAIN_SECONDS:-}" ] || positive_int MAX_TRAIN_SECONDS "$MAX_TRAIN_SECONDS"

mkdir -p "$OUT"

if [ -n "${STAGE:-}" ]; then
  echo "== staging $EXPORT to $STAGE, local disk, before anything mmaps it =="
  mkdir -p "$STAGE"
  cp -v "$EXPORT"/* "$STAGE"/
  python - "$EXPORT" "$STAGE" <<'PY'
import os, sys
src, dst = sys.argv[1], sys.argv[2]
names = sorted(n for n in os.listdir(src) if os.path.isfile(os.path.join(src, n)))
if not names:
    sys.exit(f"{src} holds no file")
for n in names:
    a = os.path.getsize(os.path.join(src, n))
    b = os.path.getsize(os.path.join(dst, n))
    if a != b:
        sys.exit(f"{n}: {a} bytes on the source, {b} staged; refusing")
    print(f"  {n}: {b} bytes staged")
PY
  EXPORT=$STAGE
fi

COMMON=(--student "$EXPORT" --teacher "$TEACHER"
        --mode "$MODE" --objective kl
        --seq-len "$SEQ" --batch-size "$BATCH"
        --corpus "$CORPUS" --mix-ratio "$MIX_RATIO"
        --device cuda --dtype bf16 --seed 0)

echo "== probe: $PROBE steps on corpus $CORPUS, to read this card's rate =="
set +e
python -m llvqtune "${COMMON[@]}" \
  --steps "$PROBE" --lr 1e-8 --warmup 1 \
  --journal "$OUT/probe.jsonl" --out "$OUT/probe.json"
PROBE_RC=$?
set -e
# 1 is the probe's normal exit: six steps at 1e-8 do not improve anything.
# 2 is a refused wiring, the teacher pairing among them.
if [ "$PROBE_RC" -eq 2 ]; then
  echo "the probe refused its wiring (exit 2); nothing is trained"
  exit 2
fi

READ=$(python - "$OUT/probe.jsonl" <<'PY'
import json, sys
closed = None
try:
    for line in open(sys.argv[1]):
        r = json.loads(line)
        if r.get("event") == "closed":
            closed = r
except FileNotFoundError:
    pass
if closed and closed.get("seconds_per_step"):
    print(closed["seconds_per_step"])
    print(closed.get("first_loss", ""))
    print(json.dumps(closed.get("gauge"), separators=(",", ":")))
PY
)
RATE=$(sed -n 1p <<<"$READ")
FIRST_KL=$(sed -n 2p <<<"$READ")
GAUGE=$(sed -n 3p <<<"$READ")
if [ -z "$RATE" ]; then
  echo "the probe wrote no rate; refusing to guess a step count"
  exit 2
fi
echo "== probe: $RATE s a step, first loss $FIRST_KL, gauge $GAUGE =="
# The device is cuda, so a probe with no gauge means the wiring dropped it,
# and the peak memory this run exists to record would be lost for two hours.
if [ -z "$GAUGE" ] || [ "$GAUGE" = "null" ]; then
  echo "the probe recorded no device memory; refusing a run that would not either"
  exit 5
fi

if [ -n "${MAX_FIRST_KL:-}" ]; then
  if python -c "import sys; sys.exit(0 if float('$FIRST_KL') > float('$MAX_FIRST_KL') else 1)"; then
    echo "first loss $FIRST_KL is above MAX_FIRST_KL=$MAX_FIRST_KL: the export is suspect; nothing is trained"
    exit 4
  fi
fi

# The routing check, automatic when FIRST_LOSS_BAND="lo hi" is set: the probe's
# first KL is the model as exported, on the first batch, before any update. A
# value outside the preregistered band means the wrong model, the wrong
# routing or the wrong batch, and the run stops here rather than bill 2 h.
if [ -n "${FIRST_LOSS_BAND:-}" ]; then
  python - "$OUT/probe.jsonl" $FIRST_LOSS_BAND <<'PY'
import json, sys
path, lo, hi = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
first = None
for line in open(path):
    r = json.loads(line)
    if r.get("event") == "closed":
        first = r.get("first_loss")
print(f"first_loss {first} against the band [{lo}, {hi}]")
sys.exit(0 if first is not None and lo <= first <= hi else 3)
PY
fi

if [ -n "$STEPS_SET" ]; then
  STEPS=$STEPS_SET
  echo "== STEPS=$STEPS set by the caller; BUDGET is not read =="
else
  STEPS=$(python -c "print(max(1, int($BUDGET / $RATE)))")
fi
TOKENS=$(python -c "print($STEPS * $SEQ * $BATCH)")
PROJECTED=$(python -c "print(int($STEPS * $RATE))")
echo "== measured $RATE s a step, so $STEPS steps and $TOKENS tokens, ~$PROJECTED s at the probe's rate =="

if [ -n "${MAX_TRAIN_SECONDS:-}" ] && [ "$PROJECTED" -gt "$MAX_TRAIN_SECONDS" ]; then
  echo "projected $PROJECTED s exceeds MAX_TRAIN_SECONDS=$MAX_TRAIN_SECONDS; refusing rather than dying at the timeout"
  exit 3
fi

echo "== training =="
date
python -m llvqtune "${COMMON[@]}" \
  --steps "$STEPS" --lr "$LR" --warmup 100 \
  --checkpoint-every 200 \
  --journal "$OUT/journal.jsonl" --out "$OUT/sigma.json"
date
ls -la "$OUT"
