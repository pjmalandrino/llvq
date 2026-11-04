#!/usr/bin/env bash
# The 4B under the grouped config: GSM8K through the served kernel, paired with the expanded census of
# 2026-09-26, and the 57-question MMLU door under both configs. Preregistered:
# proofs/preregistration-gsm8k-grouped-2026-11-04.md, stamped before this.
#
#   DRY_RUN=1 bash ops/jobs/gsm8k-grouped.sh                 prints and parses the job script
#   IMAGE_SHA=<space sha> bash ops/jobs/gsm8k-grouped.sh      launches, after the operator's go
#
# Four steps in one job: oracle; a replay of run 0's 50 problems under the expanded config, gated on a
# digest of run 0's rows, which ties today's image to the census of 2026-09-26; the grouped census on
# 1,319 problems; the MMLU door under each config. A separate script rather than an arm of
# gsm8k-served.sh, so the three arms of the 2026-09-26 campaign keep their bytes.
#
# Every LLVQ_* variable is set on its own command line, never exported: the job refuses to start with
# one in its environment. The dumps are flushed per problem, so a job killed at its timeout keeps what
# it paid for, without its trailer.
#
# Cost, estimated (l40sx1 $1.80/h): about 71 min, $2.1. Ceiling 150m, $4.50.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

OBJ_DIR=sealed-4b-2026-09-23
NAME=qwen3-4b-sealed.bin
LOCAL=$HOME/q4b-sealed-2026-09-23/$NAME
TIMEOUT=150m
PREREG=proofs/preregistration-gsm8k-grouped-2026-11-04.md
BUCKET=Pier-Jean/jobs-artifacts
DATASET_REV=740312add88f781978c0658806c59bc2815b9866
CONF_EXP_LOCAL=configs/qwen3-4b-tetra-e4.json
CONF_GRP_LOCAL=configs/qwen3-4b-tetra-e4-grouped.json
# Run 0 of 2026-09-26, the pilot's 50 problems under the expanded config. The digest drops the header
# and the two timing fields, writes each row and the trailer with sorted keys, joins them by newlines
# and hashes the UTF-8 bytes (the prereg's gate 4).
REPLAY_REF=docs/mesures/gsm8k-wave1-2026-09-26-brut/smoke/gsm8k-4b-served.jsonl
REPLAY_SHA=8300f75e771ef30d4c2ee4a10968261cfaa99b1ba90f8fa072e9f36446940805
REPLAY_FP=94c7b98876288901
CENSUS_FP=bfa9135ce4b55c75
O=/out/gsm8k-4b-grouped-2026-11-04${RETRY:+-r$RETRY}
F=/out/$OBJ_DIR/$NAME
CE=$O/qwen3-4b-tetra-e4.json
CG=$O/qwen3-4b-tetra-e4-grouped.json
DRY=${DRY_RUN:-0}

# The digest rule, one copy, run on the Mac now and in the job later.
DIGEST_PY='import hashlib,json,sys
rows=[l for l in open(sys.argv[1],encoding="utf-8").read().split("\n") if l][1:]
out=[]
for l in rows:
    d=json.loads(l); d.pop("prefill_s",None); d.pop("decode_s",None)
    out.append(json.dumps(d,sort_keys=True,ensure_ascii=False))
print(hashlib.sha256("\n".join(out).encode("utf-8")).hexdigest())'

CONF_EXP=$(cat "$CONF_EXP_LOCAL")
CONF_GRP=$(cat "$CONF_GRP_LOCAL")
CONF_EXP_SHA=$(printf '%s\n' "$CONF_EXP" | shasum -a 256 | cut -d' ' -f1)
CONF_GRP_SHA=$(printf '%s\n' "$CONF_GRP" | shasum -a 256 | cut -d' ' -f1)
for pair in "$CONF_EXP_LOCAL:$CONF_EXP_SHA" "$CONF_GRP_LOCAL:$CONF_GRP_SHA"; do
  f=${pair%:*}; want=${pair#*:}
  [ "$want" = "$(shasum -a 256 < "$f" | cut -d' ' -f1)" ] \
    || { echo "refused: $f does not end in exactly one newline" >&2; exit 1; }
done
# The two configs must differ in the attention field and in nothing else.
python3 -c 'import json,sys
e=json.load(open(sys.argv[1])); g=json.load(open(sys.argv[2]))
k=("layout","embed","rot_share","fuse","kv")
assert tuple(e[x] for x in k)==("tetra48","q4","1","0","f16"), e
assert tuple(g[x] for x in k)==tuple(e[x] for x in k), (e,g)
assert e.get("attention") is None, e
assert g.get("attention")=="grouped", g' "$CONF_EXP_LOCAL" "$CONF_GRP_LOCAL"
[ "$(python3 -c "$DIGEST_PY" "$REPLAY_REF")" = "$REPLAY_SHA" ] \
  || { echo "refused: $REPLAY_REF does not digest to $REPLAY_SHA" >&2; exit 1; }
tail -1 "$REPLAY_REF" | grep -qF "\"fingerprint\":\"$REPLAY_FP\"" \
  || { echo "refused: $REPLAY_REF does not end on fingerprint $REPLAY_FP" >&2; exit 1; }

if [ "$DRY" = 1 ]; then
  SHA=0000000000000000000000000000000000000000000000000000000000000000
  BYTES=0
else
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  test -f "$LOCAL" || { echo "refused: $LOCAL missing" >&2; exit 1; }
  SHA=$(shasum -a 256 "$LOCAL" | cut -d' ' -f1)
  BYTES=$(stat -f %z "$LOCAL")
  REMOTE=$(hf buckets ls "hf://buckets/$BUCKET/$OBJ_DIR/" 2>/dev/null | awk -v n="/$NAME" 'substr($NF, length($NF) - length(n) + 1) == n {print $1}' || true)
  [ "$REMOTE" = "$BYTES" ] || { echo "refused: bucket copy is ${REMOTE:-absent} B, local $BYTES B" >&2; exit 1; }
  if hf buckets ls "hf://buckets/$BUCKET/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already exists" >&2; exit 1
  fi
  IMAGE_SHA=${IMAGE_SHA:?IMAGE_SHA=<the Space sha the prereg names>}
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    'from huggingface_hub import HfApi; print(HfApi().space_info("Pier-Jean/llvq-runner-cuda").sha)')
  [ "$NOW" = "$IMAGE_SHA" ] || { echo "refused: the Space is at $NOW, expected $IMAGE_SHA" >&2; exit 1; }
  echo "4b sealed: $BYTES B, sha256 $SHA; configs $CONF_EXP_SHA / $CONF_GRP_SHA; image $NOW"
fi

PRE=$(printf 'O=%q\nF=%q\nCE=%q\nCG=%q\nSHA=%q\nBYTES=%q\nCONF_EXP=%q\nCONF_GRP=%q\nCONF_EXP_SHA=%q\nCONF_GRP_SHA=%q\nDATASET_REV=%q\nREPLAY_SHA=%q\nREPLAY_FP=%q\nCENSUS_FP=%q\nDIGEST_PY=%q\nexport HF_HOME=/tmp/hf' \
  "$O" "$F" "$CE" "$CG" "$SHA" "$BYTES" "$CONF_EXP" "$CONF_GRP" "$CONF_EXP_SHA" "$CONF_GRP_SHA" \
  "$DATASET_REV" "$REPLAY_SHA" "$REPLAY_FP" "$CENSUS_FP" "$DIGEST_PY")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader | tee "$O/gpu.txt"
echo '== oracle (hard rule 10) =='
oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle-cuda.txt" | tail -2
grep -q MATCH "$O/oracle-cuda.txt"
echo '== the object on the mount: bytes and sha256 against the Mac =='
test "$(stat -c %s "$F")" = "$BYTES"
sha256sum "$F" | tee "$O/files.sha256"
test "$(cut -d' ' -f1 "$O/files.sha256")" = "$SHA"
echo '== the two served configs, written from the launch command =='
printf '%s\n' "$CONF_EXP" > "$CE"
printf '%s\n' "$CONF_GRP" > "$CG"
sha256sum "$CE" "$CG" | tee -a "$O/files.sha256"
test "$(sha256sum "$CE" | cut -d' ' -f1)" = "$CONF_EXP_SHA"
test "$(sha256sum "$CG" | cut -d' ' -f1)" = "$CONF_GRP_SHA"
TILE='tile 64 (served: measured optimum for sm_89)'
PARITY='int4: tv_q4_hh equals tv_q4_h bit for bit'
provenance() {  # log, wanted attention label
  grep -E '^served config: ' "$1" | grep -qF "$2"
  grep -qF "$TILE" "$1"
  grep -qF "$PARITY" "$1"
}

echo '== STEP 2: the expanded replay, run 0 of 2026-09-26 on today'"'"'s image =='
date -u
LLVQ_DATASET_REV="$DATASET_REV" LLVQ_CONFIG="$CE" LLVQ_GSM8K_DUMP="$O/gsm8k-4b-expanded-n50.jsonl" \
  gsm8k "$F" cuda 50 2>&1 | tee "$O/gsm8k-4b-expanded-n50.txt" | grep -vE '^  \[ *[0-9]+/' | tail -16
date -u
provenance "$O/gsm8k-4b-expanded-n50.txt" 'attention expanded on cat'
tail -1 "$O/gsm8k-4b-expanded-n50.jsonl" | grep -q '"end":true'
tail -1 "$O/gsm8k-4b-expanded-n50.jsonl" | grep -qF "\"fingerprint\":\"$REPLAY_FP\""
GOT=$(python3 -c "$DIGEST_PY" "$O/gsm8k-4b-expanded-n50.jsonl")
echo "replay digest $GOT, run 0 $REPLAY_SHA"
if [ "$GOT" != "$REPLAY_SHA" ]; then
  echo 'GATE 4 FAILED: the expanded replay differs from run 0; the census is not run' >&2
  exit 3
fi
echo 'gate 4: the replay writes run 0'"'"'s 50 rows'

echo '== STEP 3: GSM8K under the grouped config, 1,319 problems =='
date -u
LLVQ_DATASET_REV="$DATASET_REV" LLVQ_CONFIG="$CG" LLVQ_GSM8K_DUMP="$O/gsm8k-4b-grouped.jsonl" \
  gsm8k "$F" cuda 2>&1 | tee "$O/gsm8k-4b-grouped.txt" | grep -vE '^  \[ *[0-9]+/' | tail -16
date -u
provenance "$O/gsm8k-4b-grouped.txt" 'attention grouped on grow(256)'
tail -1 "$O/gsm8k-4b-grouped.jsonl" | grep -q '"end":true'
tail -1 "$O/gsm8k-4b-grouped.jsonl" | grep -qF "\"fingerprint\":\"$CENSUS_FP\""
echo 'gate 5: the census ends on the fingerprint of run 1'

echo '== STEP 4: the MMLU door, 57 questions, under each config =='
for arm in expanded grouped; do
  case $arm in
    expanded) C=$CE; WANT='attention expanded on cat' ;;
    grouped)  C=$CG; WANT='attention grouped on grow(256)' ;;
  esac
  date -u
  LLVQ_CONFIG="$C" LLVQ_MMLU_ALLOC=flat LLVQ_MMLU_DUMP="$O/mmlu-4b-$arm-door.csv" \
    mmlu "$F" cuda 1 2>&1 | tee "$O/mmlu-4b-$arm-door.txt" | tail -12
  date -u
  provenance "$O/mmlu-4b-$arm-door.txt" "$WANT"
  grep -qF "# attention=$arm" "$O/mmlu-4b-$arm-door.csv"
  tail -1 "$O/mmlu-4b-$arm-door.csv" | grep -qE '^# end fingerprint=[0-9a-f]{16} questions=57$'
done
test "$(tail -1 "$O/mmlu-4b-expanded-door.csv")" = "$(tail -1 "$O/mmlu-4b-grouped-door.csv")"
echo 'gate 6: both MMLU dumps end on the same plan'
echo '== raw kept ==' ; ls -la "$O"
JOB

if [ "$DRY" = 1 ]; then
  JOBSCRIPT="set -euo pipefail
$PRE
$BODY"
  printf '%s\n' "$JOBSCRIPT" | bash -n && echo "DRY_RUN: job script parses; nothing launched" >&2
  exit 0
fi

uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor l40sx1 --timeout "$TIMEOUT" \
  --bucket "$BUCKET" --out-mount /out \
  --name "gsm8k-4b-grouped${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
