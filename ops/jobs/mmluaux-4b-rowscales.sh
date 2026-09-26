#!/usr/bin/env bash
# DRAFT, not launched. Trains the 4B row scales on MMLU-FORMAT text instead of
# generic text. Everything else is the run of 2026-09-19 (job 6aaeeddc, 130 min,
# $4.00): same export, same teacher, same 9,507 steps, same seed, same
# hyper-parameters. One flag moves: --corpus mmlu-aux.
#
# WHY: the DCLM arm brought perplexity to 1.0074 times f16 and left MMLU nine
# points under it (dclm-rowscales-2026-09-20.txt). The generic-text objective
# has given what it can. This asks whether distilling the same teacher on text
# shaped like the task moves the task, at the same zero bits and the same
# 1,794,564,765 bytes.
#
# NOT CONTAMINATION, and it is measured, not argued: `uv run
# ops/mmlu_aux_overlap.py` reads 0 items of `test`, `dev` or `validation`
# present in `auxiliary_train` on question AND choices (2026-09-23). The
# adapter refuses the three scored splits by name.
#
# THE CORPUS IS FINITE, unlike DCLM's shard. auxiliary_train holds about
# 24.75 M tokens (*measured*, 247.9 a question over 99,842) against the
# 19,470,336 this run consumes: 0.79 of a pass at seed 0, no repetition, 27 %
# of margin. `check()` refuses the run if the plan outruns the split, before a
# weight is loaded.
#
# MAX_FIRST_KL IS NOT SET, deliberately. The 0.50 the 8B run carried is a
# threshold on DCLM text; the first KL of this run is read on other text and
# the two are not comparable. The export is the same object the 2026-09-19 run
# already validated, so the guard it provided is spent. MAX_TRAIN_SECONDS still
# refuses a run that would die at the timeout.
#
# Cost: l40sx1 at ~$1.85/h. The reference run billed 130 min for $4.00 and this
# one asks for the same 9,507 steps; ceiling 135m = $4.16.
#
# DRY_RUN=1 bash ops/jobs/mmluaux-4b-rowscales.sh   prints, parses, launches nothing.
set -euo pipefail
REPO=$HOME/Documents/Pro/workspace/poc/llvq
cd "$REPO"

D=${D:-2026-09-23}
TGZ_DATE=${TGZ_DATE:-$D}
PREREG=${PREREG:-proofs/preregistration-mmluaux-4b-rowscales-$D.md}
BK=Pier-Jean/jobs-artifacts
O=/out/mmluaux-4b-rowscales-$D
E=/out/dclm-export-2026-09-19          # the base the 61.11 object was trained from
S=/out/llvqtune-$TGZ_DATE.tgz
EXPORT_BYTES=8044981648                # measured, hf buckets ls, 2026-09-23
MAX_TRAIN_SECONDS=${MAX_TRAIN_SECONDS:-7600}
TIMEOUT=${TIMEOUT:-135m}
DRY=${DRY_RUN:-0}

if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  L=$(hf buckets ls "hf://buckets/$BK/${E#/out/}/")
  printf '%s\n' "$L"
  got() { printf '%s\n' "$L" | awk -v n="$1" '$NF ~ ("/" n "$") {print $1}'; }
  [ "$(got model.safetensors)" = "$EXPORT_BYTES" ] || { echo "refused: model.safetensors is $(got model.safetensors) B, want $EXPORT_BYTES" >&2; exit 1; }
  [ "$(got config.json)" = 726 ] && [ "$(got tokenizer.json)" = 11422654 ] && [ "$(got tokenizer_config.json)" = 64 ] \
    || { echo "refused: config/tokenizer sizes differ from the export" >&2; exit 1; }
  hf buckets ls "hf://buckets/$BK/" | grep -q "llvqtune-$TGZ_DATE\.tgz$" || { echo "refused: $S not in the bucket" >&2; exit 1; }
  if hf buckets ls "hf://buckets/$BK/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already holds files" >&2; exit 1
  fi
  # The contamination gate, on the Mac, before a cent is spent.
  uv run ops/mmlu_aux_overlap.py || { echo "refused: the training split overlaps a scored split" >&2; exit 1; }
fi

CMDS=(
  'nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader'
  'df -h /tmp | tail -1 ; free -g | head -2'
  'pip install --quiet --no-input transformers==5.17.0 safetensors==0.8.0 pyarrow==25.0.1 huggingface_hub==1.32.0'
  "mkdir -p /tmp/src && tar xzf $S -C /tmp/src && ls /tmp/src"
  "python -c 'import torch, transformers; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), \"transformers\", transformers.__version__)'"
  "test \"\$(stat -c %s $E/model.safetensors)\" = $EXPORT_BYTES || { echo 'the export on the mount is not the uploaded size'; exit 2; }"
  # The journal must say which corpus ran, or the arm is unreadable afterwards.
  "cd /tmp/src && PYTHONPATH=/tmp/src EXPORT=$E OUT=$O TEACHER=Qwen/Qwen3-4B CORPUS=mmlu-aux STEPS=9507 MAX_TRAIN_SECONDS=$MAX_TRAIN_SECONDS bash train.sh"
  "echo '== what landed ==' ; ls -la $O"
  "python -c \"import json; h=json.loads(open('$O/journal.jsonl').readline()); assert h['corpus']=='mmlu-aux', h; print(h)\""
  "python -c \"import json; r=json.loads(open('$O/journal.jsonl').readlines()[-1]); r.pop('losses', None); print(r)\""
)

if [ "$DRY" = 1 ]; then
  { echo 'set -euo pipefail'; printf '%s\n' "${CMDS[@]}"; } | tee /dev/stderr | bash -n && echo "DRY_RUN: parses; nothing launched" >&2
  exit 0
fi

uv run ops/run.py bench \
  --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime \
  --flavor l40sx1 --timeout "$TIMEOUT" \
  --bucket "$BK" --out-mount /out \
  --name mmluaux-4b-rowscales \
  "${CMDS[@]}"
