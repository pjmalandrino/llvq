#!/usr/bin/env bash
# The row scales of the 14B DCLM base, trained on an h200: ops/jobs/dclm-8b-rowscales.sh
# at 14B. Same recipe (KL at T = 1 against the dense bf16 teacher, seq 1024 x batch 2,
# AdamW 3e-4, warmup 100, cosine, seed 0, 9,507 steps), same tarball, same image.
#
# Preregistered: proofs/preregistration-dclm-14b-rowscales-2026-09-22.md, which also
# covers the fold (fold-14b.sh, Mac) and the FT census (dclm-14b-ft-mmlu.sh). The
# launcher refuses until its .ots exists (DRY_RUN=1 excepted; hard rule 2).
#
# What changes against the 8B launcher, and why:
#   1. TEACHER=Qwen/Qwen3-14B, and the Mac refuses if the Hub moved from the
#      revision the encode read (40c06982): the teacher resolves `main` in the job.
#   2. PROBE=20 instead of train.sh's 6. At 8B the 6-step probe read 0.596 s/step
#      against 0.3994 in the loop, 49 % slow (dclm-8b-rowscales ECARTS E1): about
#      1.2 s of fixed cost spread over 6 steps. At 14B, PROBE=6 projects 8,900 to
#      11,700 s and MAX_TRAIN_SECONDS would refuse a good run after ~8 billed min;
#      PROBE=20 projects 7,550 to 8,400 s (*computed*, the card-mode costing).
#   3. MAX_TRAIN_SECONDS=9500: it admits a probe rate up to 0.999 s/step. The loop
#      is predicted at 0.734 s/step [0.67, 0.85] (*computed*: 0.3994 measured at 8B
#      x 1.838, the 14B/8B FLOP ratio), 6,981 s. Worst admitted: 9,500 s of loop +
#      ~900 s before it (8B: 247 s, scaled to the 14B's bytes ~370 s, pessimistic 740,
#      + this launcher's staging) = 10,400 s, under the 10,800 s timeout.
#   4. The export is staged by this launcher, not by train.sh's STAGE: the four
#      files are copied to local disk, and model.safetensors is checked by sha256
#      against the line the encode job wrote (EXP_SUMS, default
#      dclm-14b-export-<OBJ_DATE>/files.sha256: encode-14b.sh seg2 hashes the four
#      files on /scratch before the copy, and reads back the three small ones only)
#      before the probe loads anything. This is the first read-back of
#      model.safetensors from the bucket. train.sh's STAGE checked byte counts only; a 29.5 GB
#      file written through a bucket mount has never been tried (the costing's
#      risk), and a 4 GB write there was once truncated with no error (ops/README.md).
#      train.sh then runs with EXPORT=/tmp/export and no STAGE: the same path the
#      8B trained from.
#   5. EXPORT_BYTES=29536665800 (*computed*: export-predict.py's header emulation,
#      51,392-byte header, 443 tensors; the same emulation gives the 8B's
#      16,381,516,776 to the byte). config.json 728 B and tokenizer.json 11,422,654 B
#      are Qwen/Qwen3-14B's at 40c06982, the sealed file carries them byte for byte
#      (seal.rs:121-129, export.rs:146-153); tokenizer_config.json is export's
#      minimal 64 B. If the real export differs, EXPORT_BYTES=<its size> after
#      reading the encode job's `export-predict.py check` line.
#   6. The tarball's sha256 is checked on the mount before it is unpacked:
#      llvqtune-2026-09-21.tgz, 94,659 B, sha256 0f5d9498... (the 8B's, unchanged).
#   7. Timeout 180m, ceiling $15.00.
#
# Memory, *computed* (the component model fitted to the 8B's 72.65 GB within 0.7 %):
# 119.5 GB allocated [112.1, 122.9] of 150.1. It does not fit a 96 or 80 GB card as
# written (--grad-checkpoint is inert: the model stays in eval()); h200 only.
#
# Cost: h200 at $5.00/h (`hf jobs hardware`, 2026-09-22). Before the loop ~7.5 min
# (pip, tarball, staging 29.5 GB + sha256, teacher download 29.5 GB, two model
# loads); probe ~0.5 min; loop ~116 min. ~125 min, $10.42, range [114, 150] min =
# [$9.50, $12.50] (*estimated*); timeout 180m, ceiling $15.00.
#
# No `oracle` here, as in the 4B and 8B training jobs: this image is torch, not
# ours, and our forward pass does not run in it. What stands in for it is the
# teacher pairing refusal and the probe's first KL (MAX_FIRST_KL), which compares
# the staged export's forward with the teacher's before a step is trained.
#
# One writer at a time on the bucket (ops/README.md, mount failure 1).
#
# DRY_RUN=1 bash ops/jobs/dclm-14b-rowscales.sh   prints the job script, parses it, launches nothing.
set -euo pipefail
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
cd "$REPO"

OBJ_DATE=${OBJ_DATE:-2026-09-22}
TGZ_DATE=${TGZ_DATE:-2026-09-21}                  # the 8B's tarball, unchanged
PREREG=${PREREG:-proofs/preregistration-dclm-14b-rowscales-2026-09-22.md}
BK=Pier-Jean/jobs-artifacts
LJ=${LJ:-$HOME/q14b-dclm-$OBJ_DATE}
O=/out/dclm-14b-rowscales-$OBJ_DATE
E=/out/${EXPORT_DIR:-dclm-14b-export-$OBJ_DATE}
EXP_SUMS=${EXP_SUMS:-${E#/out/}/files.sha256}
S=/out/llvqtune-$TGZ_DATE.tgz
TGZ_BYTES=94659
TGZ_SHA=0f5d9498001d029ae4a6a6c105c135875c0882732054651485b2d58c137c368b
EXPORT_BYTES=${EXPORT_BYTES:-29536665800}         # computed; see note 5
TEACHER=Qwen/Qwen3-14B
TEACHER_REV=40c069824f4251a91eefaf281ebe4c544efd3e18
STEPS=9507
PROBE=20
MAX_TRAIN_SECONDS=${MAX_TRAIN_SECONDS:-9500}
MAX_FIRST_KL=${MAX_FIRST_KL:-0.50}                # the prereg fixes this value
TIMEOUT=${TIMEOUT:-180m}
DRY=${DRY_RUN:-0}

if [ "$DRY" = 1 ]; then
  EXP_SHA=0000000000000000000000000000000000000000000000000000000000000000
else
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  mkdir -p "$LJ"
  L=$(hf buckets ls "hf://buckets/$BK/${E#/out/}/")
  printf '%s\n' "$L"
  got() { printf '%s\n' "$L" | awk -v n="/$1" 'substr($NF, length($NF) - length(n) + 1) == n {print $1}'; }
  [ "$(got model.safetensors)" = "$EXPORT_BYTES" ] || { echo "refused: model.safetensors is $(got model.safetensors) B, want $EXPORT_BYTES" >&2; exit 1; }
  [ "$(got config.json)" = 728 ] && [ "$(got tokenizer.json)" = 11422654 ] && [ "$(got tokenizer_config.json)" = 64 ] \
    || { echo "refused: config/tokenizer sizes differ from the export" >&2; exit 1; }
  hf buckets cp "hf://buckets/$BK/$EXP_SUMS" "$LJ/export-files.sha256"
  EXP_SHA=$(awk '{p = "/" $2} substr(p, length(p) - 17) == "/model.safetensors" {print $1}' "$LJ/export-files.sha256")
  [ "$(printf '%s\n' "$EXP_SHA" | grep -c .)" = 1 ] && [ ${#EXP_SHA} -eq 64 ] \
    || { echo "refused: want exactly one sha256 for model.safetensors in $EXP_SUMS" >&2; exit 1; }
  [ "$(hf buckets ls "hf://buckets/$BK/" | awk -v n="llvqtune-$TGZ_DATE.tgz" '$NF == n {print $1}')" = "$TGZ_BYTES" ] \
    || { echo "refused: $S is not in the bucket at $TGZ_BYTES B" >&2; exit 1; }
  if hf buckets ls "hf://buckets/$BK/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already holds files" >&2; exit 1
  fi
  REV=$(uv run --quiet --with huggingface_hub python -c \
    "from huggingface_hub import HfApi; print(HfApi().model_info('$TEACHER').sha)")
  [ "$REV" = "$TEACHER_REV" ] || { echo "refused: $TEACHER moved to $REV; the encode read $TEACHER_REV" >&2; exit 1; }
  echo "export $E: $EXPORT_BYTES B, sha256 $EXP_SHA; teacher $TEACHER $REV" | tee "$LJ/rowscales-provenance.txt"
  # No HF_TOKEN is exported: the teacher is public and the export and tarball come
  # from the bucket mount. run.py passes one only if the caller set it.
fi

# Every check is its own command, joined by `;` and never by `&&`: under set -e a
# failure in the middle of an && list does not stop the shell, only the last one does.
CMDS=(
  'nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader'
  'df -h /tmp | tail -1 ; free -g | head -2'
  "mkdir -p $O"
  "test \"\$(stat -c %s $S)\" = $TGZ_BYTES; sha256sum $S | tee $O/tarball.sha256; test \"\$(cut -d' ' -f1 $O/tarball.sha256)\" = $TGZ_SHA"
  'pip install --quiet --no-input transformers==5.17.0 safetensors==0.8.0 pyarrow==25.0.1 huggingface_hub==1.32.0'
  "mkdir -p /tmp/src; tar xzf $S -C /tmp/src; ls /tmp/src"
  "python -c 'import torch, transformers; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), torch.cuda.get_arch_list(), \"transformers\", transformers.__version__)'"
  "test \"\$(stat -c %s $E/model.safetensors)\" = $EXPORT_BYTES || { echo 'the export on the mount is not the uploaded size'; exit 2; }"
  "echo '== staging the export to local disk, then sha256 against the encode job =='; mkdir -p /tmp/export; date -u; for f in config.json tokenizer.json tokenizer_config.json model.safetensors; do cp -v $E/\$f /tmp/export/; test \"\$(stat -c %s /tmp/export/\$f)\" = \"\$(stat -c %s $E/\$f)\"; done; date -u"
  "sha256sum /tmp/export/* | tee $O/export-staged.sha256; test \"\$(awk '\$2 ~ /model[.]safetensors\$/ {print \$1}' $O/export-staged.sha256)\" = $EXP_SHA; echo 'staged export: sha256 matches the encode job'; date -u"
  "cd /tmp/src; PYTHONPATH=/tmp/src EXPORT=/tmp/export OUT=$O TEACHER=$TEACHER STEPS=$STEPS PROBE=$PROBE MAX_TRAIN_SECONDS=$MAX_TRAIN_SECONDS MAX_FIRST_KL=$MAX_FIRST_KL bash train.sh"
  "echo '== what landed ==' ; ls -la $O"
  "python -c \"import json; r=json.loads(open('$O/journal.jsonl').readlines()[-1]); r.pop('losses', None); print(r)\""
)

if [ "$DRY" = 1 ]; then
  { echo 'set -euo pipefail'; printf '%s\n' "${CMDS[@]}"; } | tee /dev/stderr | bash -n && echo "DRY_RUN: parses; nothing launched" >&2
  exit 0
fi

uv run ops/run.py bench \
  --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime \
  --flavor h200 --any-flavor --timeout "$TIMEOUT" \
  --bucket "$BK" --out-mount /out \
  --name dclm-14b-rowscales \
  "${CMDS[@]}"
