#!/usr/bin/env bash
# The speculative decode's gate on one L40S: the sealed 0.6B drafts, each served target (4B, 8B, 14B under their v9
# configs) checks with 4E's verify pass, and every token must be the target's greedy token. One job, the three sizes
# in turn, each its own `specrun` process (`llvq-llm/src/bin/specrun.rs`):
#   the install (the target's 4E v9, the drafter's 4E v9, the verify pass) and the card memory before and after;
#   four prompts (chat, code, summary, the served door's house prompt), 256 tokens, greedy then k = 1, 2, 3,
#   1 + 5 rounds, the tokens of every arm gated against the greedy reference in every round;
#   the 50 GSM8K pilot problems, greedy then k = 1, 2, 3, every completion gated against the greedy one.
# A size whose identity fails is reported and the job goes on to the next size.
# Preregistered: proofs/preregistration-spec-gate-2026-11-07.md, to stamp before this.
#
#   DRY_RUN=1 bash ops/jobs/spec-gate.sh             parses the job script, launches nothing
#   bash ops/jobs/spec-gate.sh upload                 copies the drafter into the bucket
#   IMAGE_SHA=<space sha> bash ops/jobs/spec-gate.sh  launches (SIZES="4b 8b 14b" by default)
#
# The image must be built from this branch: specrun is in the cargo build list and the runtime COPY of
# ops/Dockerfile.cuda; configs/** is in UPLOAD_ALLOW of ops/run.py.
#
# Cost, estimated (l40sx1 $1.80/h, prereg section 7): about 50 min, $1.50; timeout 90m, $2.70.
#
# The registry row (docs/data/jobs.csv):
# <date>,<job_id>,spec-gate,l40sx1,<min>,<cost>,spec-gate-2026-11-07.txt,"DECODAGE SPECULATIF 0.6B : jetons = greedy <oui|non> aux 3 tailles ; meilleur decodage / greedy 4B x<a> (k=<k>), 8B x<b>, 14B x<c> ; acceptation chat/code/resume/GSM8K <...> ; memoire carte + <g> Go ; prereg tamponne avant"
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-spec-gate-2026-11-07.md
IMAGE_SHA=${IMAGE_SHA:-}
BUCKET=Pier-Jean/jobs-artifacts
O=/out/spec-gate-2026-11-07${RETRY:+-r$RETRY}
DRY=${DRY_RUN:-0}
SIZES=${SIZES:-4b 8b 14b}
DATASET_REV=740312add88f781978c0658806c59bc2815b9866
DRAFT_DIR=draft-0p6b-2026-11-07
DRAFT_LOCAL=${DRAFT_LOCAL:-$HOME/q06b-tetra-2026-11-07/qwen3-0p6b-sealed.bin}
DRAFT="$DRAFT_DIR/qwen3-0p6b-sealed.bin:339261863"
DRAFT_SHA=5d62875ea1eee5f8079ee522d307e9050645d6f630ad4ec233f397f16ba48494
P4E_V9_SHA=1b422b62328be6752df4add43280565fd2a49cfa20ee26953b301abe7a15c530
P4V_SHA=ab162e7fca4a1b5130170931eda22194cc7514c15fa64a52829f33a728ddd2fd
CD_LOCAL=configs/qwen3-0p6b-tetra-e4-4e-v9.json

if [ "${1:-}" = upload ]; then
  [ "$(stat -f %z "$DRAFT_LOCAL")" = "${DRAFT##*:}" ] || { echo "refused: $DRAFT_LOCAL is not ${DRAFT##*:} B" >&2; exit 1; }
  [ "$(shasum -a 256 "$DRAFT_LOCAL" | cut -d' ' -f1)" = "$DRAFT_SHA" ] || { echo "refused: $DRAFT_LOCAL sha256 differs" >&2; exit 1; }
  hf buckets cp "$DRAFT_LOCAL" "hf://buckets/$BUCKET/${DRAFT%%:*}"
  hf buckets ls "hf://buckets/$BUCKET/$DRAFT_DIR/"
  exit 0
fi

# size -> sealed path:bytes, sha256, the served v9 door's 256-token fingerprint (served-v9*-2026-11-07)
row() {
  case "$1" in
    4b) echo "sealed-4b-2026-09-23/qwen3-4b-sealed.bin:1418224685 886391a8c03f66dc269cc65c3598c6627dbdcd259180aff36604ef10d37371b8 61d0654532d35120 20m" ;;
    8b) echo "sealed-8b27-2026-09-24/qwen3-8b-sealed-B.bin:2815098745 7bdb9a5503518081b1f652a93215250898caae5cc1b6a2f33ee9fb6094565c13 c817aeaf2f5d1116 25m" ;;
    14b) echo "sealed-14b-2026-09-23/qwen3-14b-sealed.bin:5087000541 61db37fe7ce8e6a57c5c59f257319c796574981ecb5e450a02c73dd3c17e2ca0 dc4f9ab9aa24c4d3 35m" ;;
    *) echo "refused: size $1" >&2; exit 1 ;;
  esac
}
ROWS=""
CFGS=""
for s in $SIZES; do
  ROWS="$ROWS$s $(row "$s");"
  c=configs/qwen3-$s-tetra-e4-4e-v9.json
  grep -q '^  "rows": "v9",$' "$c" || { echo "refused: $c does not name rows v9" >&2; exit 1; }
  CFGS="$CFGS$s $(shasum -a 256 < "$c" | cut -d' ' -f1) $(base64 < "$c" | tr -d '\n');"
done
for c in "$CD_LOCAL" configs/qwen3-{4b,8b,14b}-tetra-e4-4e-v9.json; do
  [ "$(printf '%s\n' "$(cat "$c")" | shasum -a 256 | cut -d' ' -f1)" = "$(shasum -a 256 < "$c" | cut -d' ' -f1)" ] \
    || { echo "refused: $c does not end in exactly one newline" >&2; exit 1; }
done
CD=$(cat "$CD_LOCAL"); CD_SHA=$(shasum -a 256 < "$CD_LOCAL" | cut -d' ' -f1)

if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  for entry in "$DRAFT" $(for s in $SIZES; do row "$s" | cut -d' ' -f1; done); do
    path=${entry%%:*}; bytes=${entry##*:}
    REMOTE=$(hf buckets ls "hf://buckets/$BUCKET/${path%/*}/" 2>/dev/null \
      | awk -v n="/${path##*/}" 'substr($NF, length($NF) - length(n) + 1) == n {print $1}' || true)
    [ "$REMOTE" = "$bytes" ] || { echo "refused: $path is ${REMOTE:-absent} B in the bucket, expected $bytes" >&2; exit 1; }
  done
  if hf buckets ls "hf://buckets/$BUCKET/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already exists" >&2; exit 1
  fi
  IMAGE_SHA=${IMAGE_SHA:?IMAGE_SHA=<the Space sha of the rebuild that carries this branch>}
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    'from huggingface_hub import HfApi; print(HfApi().space_info("Pier-Jean/llvq-runner-cuda").sha)')
  [ "$NOW" = "$IMAGE_SHA" ] || { echo "refused: the Space is at $NOW, expected $IMAGE_SHA" >&2; exit 1; }
  echo "bucket copies verified by size; drafter config $CD_SHA; image $NOW; sizes $SIZES"
fi

PRE=$(printf 'O=%q\nROWS=%q\nCFGS=%q\nCD=%q\nCD_SHA=%q\nDRAFT=%q\nDRAFT_SHA=%q\nP4E_V9_SHA=%q\nP4V_SHA=%q\nDATASET_REV=%q\nexport HF_HOME=/tmp/hf' \
  "$O" "$ROWS" "$CFGS" "$CD" "$CD_SHA" "/out/${DRAFT%%:*}" "$DRAFT_SHA" "$P4E_V9_SHA" "$P4V_SHA" "$DATASET_REV")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,clocks.max.sm --format=csv,noheader | tee "$O/gpu.txt"
echo '== gate 1: oracle =='
timeout 5m oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle.txt" | tail -2
grep -q MATCH "$O/oracle.txt"
echo '== gate 2: the drafter and its config =='
sha256sum "$DRAFT" | tee "$O/files.sha256"
test "$(cut -d' ' -f1 "$O/files.sha256")" = "$DRAFT_SHA"
PD=$O/qwen3-0p6b-tetra-e4-4e-v9.json
printf '%s\n' "$CD" > "$PD"
test "$(sha256sum "$PD" | cut -d' ' -f1)" = "$CD_SHA"
# the specrun gates of one size's output; prints PASS lines, returns non-zero on the first failure
gates() {
  f=$1; fp=$2
  grep -qF "4E: persistent module 252757 bytes, sha256 $P4E_V9_SHA, row form 7," "$f" || { echo "gate: the target's 4E v9 module"; return 1; }
  grep -qF "drafter 4E: persistent module 252757 bytes, sha256 $P4E_V9_SHA, row form 7," "$f" || { echo "gate: the drafter's 4E v9 module"; return 1; }
  grep -qF "4E verify: module 287744 bytes, sha256 $P4V_SHA," "$f" || { echo "gate: the verify module"; return 1; }
  grep -qE '^4E verify: pers_4ev_rows_f[0-9](_t512)?: [0-9]+ registers, [0-9]+ B local' "$f" || { echo "gate: the verify symbol line"; return 1; }
  grep -qE "^== door: 5 prompt tokens, greedy reference 256 tokens, fingerprint $fp\$" "$f" || { echo "gate: the greedy door is not the served v9 door ($fp)"; return 1; }
  test "$(grep -cE '^(chat|code|summary|door): every arm made the greedy tokens in every round' "$f")" = 4 || { echo "gate: identity at the four prompts"; return 1; }
  test "$(grep -cE '^  gsm8k k = [0-9]: .*completions identical to greedy 50 of 50' "$f")" = 3 || { echo "gate: identity on the 50 GSM8K problems"; return 1; }
  echo "PASS: modules, the served door's greedy tokens, identity at the four prompts and on the 50 GSM8K problems"
}
IFS=';' read -ra RR <<< "$ROWS"
for r in "${RR[@]}"; do
  [ -n "$r" ] || continue
  read -r SIZE ENTRY SSHA FP TMO <<< "$r"
  S=/out/${ENTRY%%:*}; SB=${ENTRY##*:}
  echo "== $SIZE: the object and its config =="
  test "$(stat -c %s "$S")" = "$SB"
  sha256sum "$S" | tee -a "$O/files.sha256"
  test "$(sha256sum "$S" | cut -d' ' -f1)" = "$SSHA"
  C=$(printf '%s' "$CFGS" | tr ';' '\n' | awk -v s="$SIZE" '$1 == s {print $3}')
  CS=$(printf '%s' "$CFGS" | tr ';' '\n' | awk -v s="$SIZE" '$1 == s {print $2}')
  PT=$O/qwen3-$SIZE-tetra-e4-4e-v9.json
  printf '%s' "$C" | base64 -d > "$PT"
  test "$(sha256sum "$PT" | cut -d' ' -f1)" = "$CS"
  echo "== $SIZE: specrun =="
  date -u
  set +e
  LLVQ_CONFIG="$PT" LLVQ_DATASET_REV="$DATASET_REV" timeout "$TMO" specrun "$S" "$PD" "$DRAFT" 1,2,3 all > "$O/spec-$SIZE.txt" 2>&1
  rc=$?
  set -e
  date -u
  echo "specrun $SIZE exit $rc" | tee -a "$O/summary.txt"
  grep -E '^card memory|^weights|^4E verify|^== |: every arm|^  (chat|code|summary|door|gsm8k) ' "$O/spec-$SIZE.txt" | tee -a "$O/summary.txt"
  tail -3 "$O/spec-$SIZE.txt"
  gates "$O/spec-$SIZE.txt" "$FP" 2>&1 | sed "s/^/$SIZE /" | tee -a "$O/summary.txt" || true
done
ls -la "$O"
JOB

if [ "$DRY" = 1 ]; then
  JOBSCRIPT="set -euo pipefail
$PRE
$BODY"
  printf '%s\n' "$JOBSCRIPT" | bash -n && echo "DRY_RUN: job script parses; nothing launched; drafter config $CD_SHA; sizes $SIZES" >&2
  exit 0
fi

uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor l40sx1 --timeout 90m \
  --bucket "$BUCKET" --out-mount /out \
  --name "spec-gate${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
