#!/usr/bin/env bash
# The speculative route's last card job (2026-11-08 night, prereg spec-final): on one L40S, for each served target
# (4B, 8B, 14B under their v9 configs), one `specrun <sealed> lookup-n3p1 3 all` process:
#   the install (the target's 4E v9, the verify pass with its phase stamps) and the card memory;
#   phases: at the chat and rag contexts, one-token 4E launches, 4E's stamped token, then the verify pass at 1 to 4
#     rows, stamped and plain, every launch's tokens gated against the greedy ones: where an extra row's time goes;
#   six prompts, 256 tokens, greedy then prompt lookup at k = 3 under the policy n-grams 3 only, one draft a round
#     until one was kept (rounds without a draft on 4E's token launch), 1 + 5 rounds, door and decode tok/s, every
#     round's tokens gated against the greedy reference;
#   the 50 GSM8K pilot problems, greedy then the same arm, every completion gated against the greedy one.
# The job stops after the 4B if any 4B gate fails.
# Preregistered: proofs/preregistration-spec-final-2026-11-08.md, stamped before this.
#
#   DRY_RUN=1 bash ops/jobs/spec-final.sh             parses the job script, launches nothing
#   IMAGE_SHA=<space sha> bash ops/jobs/spec-final.sh  launches (SIZES="4b 8b 14b" by default)
#
# Cost, estimated (l40sx1 $1.80/h, prereg section 7): about 42 min, $1.26; timeout 75m, $2.25.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-spec-final-2026-11-08.md
IMAGE_SHA=${IMAGE_SHA:-}
BUCKET=Pier-Jean/jobs-artifacts
O=/out/spec-final-2026-11-08${RETRY:+-r$RETRY}
DRY=${DRY_RUN:-0}
SIZES=${SIZES:-4b 8b 14b}
DATASET_REV=740312add88f781978c0658806c59bc2815b9866
P4E_V9_SHA=1b422b62328be6752df4add43280565fd2a49cfa20ee26953b301abe7a15c530
P4V_SHA=d7686d86b990b079c47a754ccb5675e090c0b9d19e1b3710558967614c3944e5
TIMEOUT=${TIMEOUT:-75m}

# size -> sealed path:bytes, sha256, the served v9 door's 256-token fingerprint (served-v9*-2026-11-07)
row() {
  case "$1" in
    4b) echo "sealed-4b-2026-09-23/qwen3-4b-sealed.bin:1418224685 886391a8c03f66dc269cc65c3598c6627dbdcd259180aff36604ef10d37371b8 61d0654532d35120 14m" ;;
    8b) echo "sealed-8b27-2026-09-24/qwen3-8b-sealed-B.bin:2815098745 7bdb9a5503518081b1f652a93215250898caae5cc1b6a2f33ee9fb6094565c13 c817aeaf2f5d1116 22m" ;;
    14b) echo "sealed-14b-2026-09-23/qwen3-14b-sealed.bin:5087000541 61db37fe7ce8e6a57c5c59f257319c796574981ecb5e450a02c73dd3c17e2ca0 dc4f9ab9aa24c4d3 33m" ;;
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
for c in configs/qwen3-{4b,8b,14b}-tetra-e4-4e-v9.json; do
  [ "$(printf '%s\n' "$(cat "$c")" | shasum -a 256 | cut -d' ' -f1)" = "$(shasum -a 256 < "$c" | cut -d' ' -f1)" ] \
    || { echo "refused: $c does not end in exactly one newline" >&2; exit 1; }
done

if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  for entry in $(for s in $SIZES; do row "$s" | cut -d' ' -f1; done); do
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
  echo "bucket copies verified by size; image $NOW; sizes $SIZES"
fi

PRE=$(printf 'O=%q\nROWS=%q\nCFGS=%q\nP4E_V9_SHA=%q\nP4V_SHA=%q\nDATASET_REV=%q\nexport HF_HOME=/tmp/hf' \
  "$O" "$ROWS" "$CFGS" "$P4E_V9_SHA" "$P4V_SHA" "$DATASET_REV")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,clocks.max.sm --format=csv,noheader | tee "$O/gpu.txt"
echo '== gate 1: oracle =='
timeout 5m oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle.txt" | tail -2
grep -q MATCH "$O/oracle.txt"
: > "$O/files.sha256"
# the specrun gates of one size's output; prints PASS lines, returns non-zero on the first failure
gates() {
  f=$1; fp=$2
  grep -qF "4E: persistent module 252757 bytes, sha256 $P4E_V9_SHA, row form 7," "$f" || { echo "gate: the target's 4E v9 module"; return 1; }
  grep -qF "drafter: prompt lookup, n-grams 3 to 3 of the prompt and the text so far; no model; one draft a round until 1 kept in a row; a round with no draft on 4E's token launch" "$f" || { echo "gate: the drafter is not the preregistered lookup policy"; return 1; }
  grep -qF "4E verify: module 288366 bytes, sha256 $P4V_SHA," "$f" || { echo "gate: the verify module"; return 1; }
  test "$(grep -cE '^== phases, (chat|rag): context [0-9]+' "$f")" = 2 || { echo "gate: the phases at both contexts"; return 1; }
  test "$(grep -cE '^ +[1-4] +([0-9.]+ +){8}' "$f")" = 8 || { echo "gate: the verify phases at 1 to 4 rows, both contexts"; return 1; }
  grep -qE '^4E verify: pers_4ev_rows_f[0-9](_t512)?: [0-9]+ registers, [0-9]+ B local' "$f" || { echo "gate: the verify symbol line"; return 1; }
  grep -qE "^== door: 5 prompt tokens, greedy reference 256 tokens, fingerprint $fp\$" "$f" || { echo "gate: the greedy door is not the served v9 door ($fp)"; return 1; }
  test "$(grep -cE '^(chat|code|summary|edit|rag|door): every arm made the greedy tokens in every round' "$f")" = 6 || { echo "gate: identity at the six prompts"; return 1; }
  test "$(grep -cE '^  gsm8k k = 3: .*completions identical to greedy 50 of 50' "$f")" = 1 || { echo "gate: identity on the 50 GSM8K problems"; return 1; }
  echo "PASS: modules, phases, the served door's greedy tokens, identity at the six prompts and on the 50 GSM8K problems"
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
  LLVQ_CONFIG="$PT" LLVQ_DATASET_REV="$DATASET_REV" timeout "$TMO" specrun "$S" lookup-n3p1 3 all > "$O/spec-$SIZE.txt" 2>&1
  rc=$?
  set -e
  date -u
  echo "specrun $SIZE exit $rc" | tee -a "$O/summary.txt"
  grep -E '^card memory|^weights|^4E verify|^== |: every arm|^  (chat|code|summary|edit|rag|door|gsm8k) |^  4E|^  verify|^ +[1-4] ' "$O/spec-$SIZE.txt" | tee -a "$O/summary.txt"
  tail -3 "$O/spec-$SIZE.txt"
  if gates "$O/spec-$SIZE.txt" "$FP" > "$O/gates-$SIZE.txt" 2>&1; then ok=1; else ok=0; fi
  sed "s/^/$SIZE /" "$O/gates-$SIZE.txt" | tee -a "$O/summary.txt"
  if [ "$SIZE" = 4b ] && [ "$ok" = 0 ]; then echo "STOP: a 4B gate failed, the 8B and 14B are not run" | tee -a "$O/summary.txt"; break; fi
done
ls -la "$O"
JOB

if [ "$DRY" = 1 ]; then
  JOBSCRIPT="set -euo pipefail
$PRE
$BODY"
  printf '%s\n' "$JOBSCRIPT" | bash -n && echo "DRY_RUN: job script parses; nothing launched; sizes $SIZES" >&2
  exit 0
fi

uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor l40sx1 --timeout "$TIMEOUT" \
  --bucket "$BUCKET" --out-mount /out \
  --name "spec-final${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
