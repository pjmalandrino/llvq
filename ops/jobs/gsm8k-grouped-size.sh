#!/usr/bin/env bash
# The 8B or the 14B under the grouped config: GSM8K through the served kernel, paired with the expanded census
# of 2026-09-26 at that size. Preregistered: proofs/preregistration-gsm8k-grouped-size-2026-11-04.md, stamped
# before this. The 4B run of the same question is ops/jobs/gsm8k-grouped.sh, which keeps its bytes.
#
#   SIZE=8b|14b DRY_RUN=1 bash ops/jobs/gsm8k-grouped-size.sh               prints and parses the job script
#   SIZE=8b|14b IMAGE_SHA=<space sha> bash ops/jobs/gsm8k-grouped-size.sh    launches, after the operator's go
#
# Three steps in one job: oracle; a replay of the pilot's 50 problems under the expanded config, gated on the
# digest of the same 50 rows of that size's census of 2026-09-26, which ties today's image to the census; then
# the grouped census on 1,319 problems. No MMLU step: the light MMLU of both sizes ran on 2026-11-04.
#
# Every LLVQ_* variable is set on its own command line, never exported: the job refuses to start with one in
# its environment. The dumps are flushed per problem, so a job killed at its timeout keeps what it paid for,
# without its trailer.
#
# Cost, estimated (l40sx1 $1.80/h): 8B about 75 min, $2.25, ceiling 100m, $3.00; 14B about 137 min, $4.11,
# ceiling 170m, $5.10.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

SIZE=${SIZE:?SIZE=8b or 14b}
case "$SIZE" in
  8b)  OBJ_DIR=sealed-8b27-2026-09-24; NAME=qwen3-8b-sealed-B.bin; TIMEOUT=100m
       OBJ_SHA=7bdb9a5503518081b1f652a93215250898caae5cc1b6a2f33ee9fb6094565c13
       CENSUS_REF=docs/mesures/gsm8k-wave2-2026-09-26-brut/run2-served-8b/gsm8k-8b-served.jsonl
       CENSUS_REF_SHA=23bbac8be887e0ea328f8f211604fe84d30905b8be3da794a5c331d60fcb6932
       CONF_REF=docs/mesures/gsm8k-wave2-2026-09-26-brut/run2-served-8b/qwen3-8b-tetra-e4.json
       REPLAY_ROWS_SHA=3e5666620e204b3aff244c6398963771dcae18f67e8a9fd87c7737d49b766ecd ;;
  14b) OBJ_DIR=sealed-14b-2026-09-23; NAME=qwen3-14b-sealed.bin; TIMEOUT=170m
       OBJ_SHA=61db37fe7ce8e6a57c5c59f257319c796574981ecb5e450a02c73dd3c17e2ca0
       CENSUS_REF=docs/mesures/gsm8k-wave2-2026-09-26-brut/run3-served-14b/gsm8k-14b-served.jsonl
       CENSUS_REF_SHA=05acf95362b32567c8b24c61c11cdf1b5f5bb1ac448dae486cd63c4962c7e678
       CONF_REF=docs/mesures/gsm8k-wave2-2026-09-26-brut/run3-served-14b/qwen3-14b-tetra-e4.json
       REPLAY_ROWS_SHA=14631e2b231f47cb2131e7d310e56f65c4e8207d35e648d45a463b9c46ad1f72 ;;
  *) echo "refused: SIZE=$SIZE (8b or 14b; the 4B is ops/jobs/gsm8k-grouped.sh)" >&2; exit 1 ;;
esac
LOCAL=$HOME/q8b-14b-sealed-2026-09-23/$NAME
PREREG=proofs/preregistration-gsm8k-grouped-size-2026-11-04.md
BUCKET=Pier-Jean/jobs-artifacts
DATASET_REV=740312add88f781978c0658806c59bc2815b9866
CONF_EXP_LOCAL=configs/qwen3-$SIZE-tetra-e4.json
CONF_GRP_LOCAL=configs/qwen3-$SIZE-tetra-e4-grouped.json
# The pilot's 50 problems (gsm8k::select, seed 0x65736d386b): the indices run 0 of 2026-09-26 holds.
PILOT=docs/mesures/gsm8k-wave1-2026-09-26-brut/smoke/gsm8k-4b-served.jsonl
REPLAY_FP=94c7b98876288901
CENSUS_FP=bfa9135ce4b55c75
O=/out/gsm8k-$SIZE-grouped-2026-11-04${RETRY:+-r$RETRY}
F=/out/$OBJ_DIR/$NAME
CE=$O/qwen3-$SIZE-tetra-e4.json
CG=$O/qwen3-$SIZE-tetra-e4-grouped.json
DRY=${DRY_RUN:-0}

# The row digest, one copy, run on the Mac now and in the job later. The header line and the trailer are
# dropped, and `prefill_s` and `decode_s` from each row; each row is written with sorted keys, the rows are
# joined by newlines, and the UTF-8 bytes hashed. A second argument keeps only the rows whose index that
# dump also holds.
ROWS_PY='import hashlib,json,sys
def rows(p):
    ls=[json.loads(l) for l in open(p,encoding="utf-8").read().split("\n") if l][1:]
    return [d for d in ls if not d.get("end")]
r=rows(sys.argv[1])
if len(sys.argv)>2:
    keep={d["index"] for d in rows(sys.argv[2])}
    r=[d for d in r if d["index"] in keep]
out=[]
for d in r:
    d.pop("prefill_s",None); d.pop("decode_s",None)
    out.append(json.dumps(d,sort_keys=True,ensure_ascii=False))
print(len(out), hashlib.sha256("\n".join(out).encode("utf-8")).hexdigest())'

CONF_EXP=$(cat "$CONF_EXP_LOCAL")
CONF_GRP=$(cat "$CONF_GRP_LOCAL")
CONF_EXP_SHA=$(printf '%s\n' "$CONF_EXP" | shasum -a 256 | cut -d' ' -f1)
CONF_GRP_SHA=$(printf '%s\n' "$CONF_GRP" | shasum -a 256 | cut -d' ' -f1)
for pair in "$CONF_EXP_LOCAL:$CONF_EXP_SHA" "$CONF_GRP_LOCAL:$CONF_GRP_SHA"; do
  f=${pair%:*}; want=${pair#*:}
  [ "$want" = "$(shasum -a 256 < "$f" | cut -d' ' -f1)" ] \
    || { echo "refused: $f does not end in exactly one newline" >&2; exit 1; }
done
# The two configs must differ in the attention field (and the note) and in nothing else.
python3 -c 'import json,sys
e=json.load(open(sys.argv[1])); g=json.load(open(sys.argv[2]))
k=("layout","embed","rot_share","fuse","kv")
assert tuple(e[x] for x in k)==("tetra48","q4","1","0","f16"), e
assert tuple(g[x] for x in k)==tuple(e[x] for x in k), (e,g)
assert e.get("attention") is None, e
assert g.get("attention")=="grouped", g' "$CONF_EXP_LOCAL" "$CONF_GRP_LOCAL"
# The expanded config is the one the census of 2026-09-26 ran, byte for byte.
cmp -s "$CONF_EXP_LOCAL" "$CONF_REF" || { echo "refused: $CONF_EXP_LOCAL is not $CONF_REF" >&2; exit 1; }
[ "$(shasum -a 256 < "$CENSUS_REF" | cut -d' ' -f1)" = "$CENSUS_REF_SHA" ] \
  || { echo "refused: $CENSUS_REF is not $CENSUS_REF_SHA" >&2; exit 1; }
tail -1 "$CENSUS_REF" | grep -qF "\"fingerprint\":\"$CENSUS_FP\"" \
  || { echo "refused: $CENSUS_REF does not end on fingerprint $CENSUS_FP" >&2; exit 1; }
[ "$(python3 -c "$ROWS_PY" "$CENSUS_REF" "$PILOT")" = "50 $REPLAY_ROWS_SHA" ] \
  || { echo "refused: the census rows at the pilot's indices do not digest to $REPLAY_ROWS_SHA" >&2; exit 1; }

if [ "$DRY" = 1 ]; then
  SHA=0000000000000000000000000000000000000000000000000000000000000000
  BYTES=0
else
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  test -f "$LOCAL" || { echo "refused: $LOCAL missing" >&2; exit 1; }
  SHA=$(shasum -a 256 "$LOCAL" | cut -d' ' -f1)
  [ "$SHA" = "$OBJ_SHA" ] || { echo "refused: $LOCAL is $SHA, the census ran $OBJ_SHA" >&2; exit 1; }
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
  echo "$SIZE sealed: $BYTES B, sha256 $SHA; configs $CONF_EXP_SHA / $CONF_GRP_SHA; image $NOW"
fi

PRE=$(printf 'SIZE=%q\nO=%q\nF=%q\nCE=%q\nCG=%q\nSHA=%q\nBYTES=%q\nCONF_EXP=%q\nCONF_GRP=%q\nCONF_EXP_SHA=%q\nCONF_GRP_SHA=%q\nDATASET_REV=%q\nREPLAY_ROWS_SHA=%q\nREPLAY_FP=%q\nCENSUS_FP=%q\nROWS_PY=%q\nexport HF_HOME=/tmp/hf' \
  "$SIZE" "$O" "$F" "$CE" "$CG" "$SHA" "$BYTES" "$CONF_EXP" "$CONF_GRP" "$CONF_EXP_SHA" "$CONF_GRP_SHA" \
  "$DATASET_REV" "$REPLAY_ROWS_SHA" "$REPLAY_FP" "$CENSUS_FP" "$ROWS_PY")

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

echo "== STEP 2: the expanded replay, the pilot's 50 problems of the $SIZE census on today's image =="
date -u
LLVQ_DATASET_REV="$DATASET_REV" LLVQ_CONFIG="$CE" LLVQ_GSM8K_DUMP="$O/gsm8k-$SIZE-expanded-n50.jsonl" \
  gsm8k "$F" cuda 50 2>&1 | tee "$O/gsm8k-$SIZE-expanded-n50.txt" | grep -vE '^  \[ *[0-9]+/' | tail -16
date -u
provenance "$O/gsm8k-$SIZE-expanded-n50.txt" 'attention expanded on cat'
tail -1 "$O/gsm8k-$SIZE-expanded-n50.jsonl" | grep -q '"end":true'
tail -1 "$O/gsm8k-$SIZE-expanded-n50.jsonl" | grep -qF "\"fingerprint\":\"$REPLAY_FP\""
GOT=$(python3 -c "$ROWS_PY" "$O/gsm8k-$SIZE-expanded-n50.jsonl")
echo "replay rows: $GOT; the census of 2026-09-26 at the same indices: 50 $REPLAY_ROWS_SHA"
if [ "$GOT" != "50 $REPLAY_ROWS_SHA" ]; then
  echo 'GATE 4 FAILED: the expanded replay differs from the census rows; the grouped census is not run' >&2
  exit 3
fi
echo 'gate 4: the replay writes the census rows'

echo "== STEP 3: GSM8K under the grouped config, 1,319 problems =="
date -u
LLVQ_DATASET_REV="$DATASET_REV" LLVQ_CONFIG="$CG" LLVQ_GSM8K_DUMP="$O/gsm8k-$SIZE-grouped.jsonl" \
  gsm8k "$F" cuda 2>&1 | tee "$O/gsm8k-$SIZE-grouped.txt" | grep -vE '^  \[ *[0-9]+/' | tail -16
date -u
provenance "$O/gsm8k-$SIZE-grouped.txt" 'attention grouped on grow(256)'
tail -1 "$O/gsm8k-$SIZE-grouped.jsonl" | grep -q '"end":true'
tail -1 "$O/gsm8k-$SIZE-grouped.jsonl" | grep -qF "\"fingerprint\":\"$CENSUS_FP\""
echo 'gate 5: the census ends on the census fingerprint'
echo '== raw kept ==' ; ls -la "$O"
JOB

# EMIT=<file>: write the job script to <file> and launch nothing.
if [ -n "${EMIT:-}" ]; then
  printf 'set -euo pipefail\n%s\n%s\n' "$PRE" "$BODY" > "$EMIT"
  bash -n "$EMIT" && echo "EMIT: job script written to $EMIT; nothing launched" >&2
  exit 0
fi

if [ "$DRY" = 1 ]; then
  JOBSCRIPT="set -euo pipefail
$PRE
$BODY"
  printf '%s\n' "$JOBSCRIPT" | bash -n && echo "DRY_RUN: job script parses; nothing launched ($SIZE, timeout $TIMEOUT)" >&2
  exit 0
fi

uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor l40sx1 --timeout "$TIMEOUT" \
  --bucket "$BUCKET" --out-mount /out \
  --name "gsm8k-$SIZE-grouped${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
