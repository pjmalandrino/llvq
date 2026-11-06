#!/usr/bin/env bash
# The served door of the sealed 4B under configs/qwen3-4b-tetra-e4-4e-v4.json, the 4B's served config since
# 2026-11-05 (4E with the v4 row loop). One job, three parts:
#   (1) the plain door: 4E against the eager path under the config, the identity gate, both rates round by round;
#   (2) the context sweep at 5, 2,048, 4,096 and 8,192, the identity gated at every length;
#   (3) the pilot's 50 GSM8K problems through the config.
# The tokens of (1) and (2) and the completions of (3) are compared with the v3 config's, read by served-4e
# (docs/mesures/served-4e-2026-11-05-brut/): the same arithmetic owes the same text.
# Preregistered: proofs/preregistration-served-v4-2026-11-05.md, to stamp before this.
#
#   DRY_RUN=1 bash ops/jobs/served-v4.sh                 parses the job script, launches nothing
#   IMAGE_SHA=<space sha> bash ops/jobs/served-v4.sh      launches, after the operator's go
#
# Every LLVQ_* variable is set on its own command line, never exported. Cost, estimated (l40sx1 $1.80/h, prereg
# section 8): about 12 min, $0.36. Timeout 25m: $0.75 at worst.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-served-v4-2026-11-05.md
IMAGE_SHA=${IMAGE_SHA:-}
BUCKET=Pier-Jean/jobs-artifacts
O=/out/served-v4-2026-11-05${RETRY:+-r$RETRY}
TIMEOUT=25m
DRY=${DRY_RUN:-0}
DATASET_REV=740312add88f781978c0658806c59bc2815b9866
CONTEXTS=5,2048,4096,8192

SEALED="sealed-4b-2026-09-23/qwen3-4b-sealed.bin:1418224685"
SEALED_SHA=886391a8c03f66dc269cc65c3598c6627dbdcd259180aff36604ef10d37371b8
CV4_LOCAL=configs/qwen3-4b-tetra-e4-4e-v4.json
SERVED_SHA=6193094310bc18acdb0a0c554807ff45f6df09c1f6ab49e0577032874f107812
REPLAY_FP=94c7b98876288901
# The v3 config's readings (served-4e-2026-11-05): door and sweep token fingerprints, and its pilot dump.
V3_DOOR=61d0654532d35120
V3_SWEEP=5:61d0654532d35120,2048:a4645c00f879ae80,4096:cc9b3414c4d67ba6,8192:9aa830caa48cb332
V3_PILOT=docs/mesures/served-4e-2026-11-05-brut/gsm8k-4b-4e-n50.jsonl
V3_PILOT_SHA=1512a90348240e4cf0664eeaf4b410c3cf74e497e85cf3155b357ff308467354
REF_PY='import hashlib,json,sys
ls=[json.loads(l) for l in open(sys.argv[1],encoding="utf-8").read().split("\n") if l][1:]
print(",".join("%d:%d:%s"%(d["index"],int(d["correct"]),hashlib.sha256(d["completion"].encode()).hexdigest()[:12]) for d in ls if not d.get("end")))'

[ "$(printf '%s\n' "$(cat "$CV4_LOCAL")" | shasum -a 256 | cut -d' ' -f1)" = "$(shasum -a 256 < "$CV4_LOCAL" | cut -d' ' -f1)" ] \
  || { echo "refused: $CV4_LOCAL does not end in exactly one newline" >&2; exit 1; }
CV4=$(cat "$CV4_LOCAL"); CV4_SHA=$(shasum -a 256 < "$CV4_LOCAL" | cut -d' ' -f1)
[ "$(shasum -a 256 < "$V3_PILOT" | cut -d' ' -f1)" = "$V3_PILOT_SHA" ] || { echo "refused: $V3_PILOT is not $V3_PILOT_SHA" >&2; exit 1; }
PILOT_REF=$(python3 -c "$REF_PY" "$V3_PILOT")

if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  path=${SEALED%%:*}; bytes=${SEALED##*:}
  REMOTE=$(hf buckets ls "hf://buckets/$BUCKET/${path%/*}/" 2>/dev/null \
    | awk -v n="/${path##*/}" 'substr($NF, length($NF) - length(n) + 1) == n {print $1}' || true)
  [ "$REMOTE" = "$bytes" ] || { echo "refused: $path is ${REMOTE:-absent} B in the bucket, expected $bytes" >&2; exit 1; }
  if hf buckets ls "hf://buckets/$BUCKET/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already exists" >&2; exit 1
  fi
  IMAGE_SHA=${IMAGE_SHA:?IMAGE_SHA=<the Space sha of the rebuild that carries this branch>}
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    'from huggingface_hub import HfApi; print(HfApi().space_info("Pier-Jean/llvq-runner-cuda").sha)')
  [ "$NOW" = "$IMAGE_SHA" ] || { echo "refused: the Space is at $NOW, expected $IMAGE_SHA" >&2; exit 1; }
  echo "bucket copy verified by size; config $CV4_SHA; image $NOW"
fi

CMP_PY='import re,sys,hashlib,json
mode=sys.argv[1]
if mode=="door":
    t=open(sys.argv[2],encoding="utf-8").read()
    h=re.search(r"^  token ids: ([0-9a-f]{16}) ",t,re.M).group(1)
    print("door tokens: %s against the v3 config %s: %s"%(h,sys.argv[3],"identical" if h==sys.argv[3] else "DIFFERENT"))
elif mode=="sweep":
    want=dict(x.split(":") for x in sys.argv[3].split(","))
    got={m.group(1):m.group(2) for m in re.finditer(r"^  tokens after (\d+): ([0-9a-f]{16}) ",open(sys.argv[2],encoding="utf-8").read(),re.M)}
    for n,h in want.items():
        print("sweep tokens after %s: %s against the v3 config %s: %s"%(n,got.get(n),h,"identical" if got.get(n)==h else "DIFFERENT"))
else:
    ref={int(i):(int(c),h) for i,c,h in (x.split(":") for x in sys.argv[3].split(","))}
    ls=[json.loads(l) for l in open(sys.argv[2],encoding="utf-8").read().split("\n") if l][1:]
    rows=[d for d in ls if not d.get("end")]
    same=sum(hashlib.sha256(d["completion"].encode()).hexdigest()[:12]==ref.get(d["index"],(0,""))[1] for d in rows)
    a=sum(int(d["correct"]) for d in rows); g=sum(c for c,_ in ref.values())
    print("GSM8K pilot, %d problems: v4 config %d correct, v3 config %d; completions identical %d/%d"%(len(rows),a,g,same,len(rows)))'

PRE=$(printf 'O=%q\nS=%q\nSB=%q\nSSHA=%q\nCV4=%q\nCV4_SHA=%q\nSERVED_SHA=%q\nDATASET_REV=%q\nCONTEXTS=%q\nREPLAY_FP=%q\nPILOT_REF=%q\nV3_DOOR=%q\nV3_SWEEP=%q\nCMP_PY=%q\nexport HF_HOME=/tmp/hf' \
  "$O" "/out/${SEALED%%:*}" "${SEALED##*:}" "$SEALED_SHA" "$CV4" "$CV4_SHA" "$SERVED_SHA" \
  "$DATASET_REV" "$CONTEXTS" "$REPLAY_FP" "$PILOT_REF" "$V3_DOOR" "$V3_SWEEP" "$CMP_PY")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,clocks.max.sm --format=csv,noheader | tee "$O/gpu.txt"
echo '== gate 1: oracle =='
timeout 5m oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle.txt" | tail -2
grep -q MATCH "$O/oracle.txt"
echo '== gate 2: the object and the config =='
test "$(stat -c %s "$S")" = "$SB"
sha256sum "$S" | tee "$O/files.sha256"
test "$(cut -d' ' -f1 "$O/files.sha256")" = "$SSHA"
P4=$O/qwen3-4b-tetra-e4-4e-v4.json
printf '%s\n' "$CV4" > "$P4"
sha256sum "$P4" | tee -a "$O/files.sha256"
test "$(sha256sum "$P4" | cut -d' ' -f1)" = "$CV4_SHA"
WANT='attention kernel from position 0 on grow(256), decode persistent (4E rows v4, one launch a token run)'
served_unit() { grep -qF "sha256 $SERVED_SHA" "$1" || { echo "gate: $1 did not compile the served unit $SERVED_SHA" >&2; exit 1; }; }
v4_loaded() {  # the 4E install lines: row form 2, RoPE on the card, the 512-thread symbol with no local memory
  grep -qE '^4E: persistent module [0-9]+ bytes, sha256 [0-9a-f]+, row form 2,' "$1"
  grep -q '^4E: RoPE forms equal to candle' "$1"
  grep -qE '^4E: pers_4e_tokens_f[0-9]_t512: [0-9]+ registers, 0 B local' "$1"
}

echo '== PART 1: the door, 256 tokens =='
date -u
LLVQ_CONFIG="$P4" timeout 8m fusedrun "$S" 256 > "$O/door.txt" 2>&1
date -u
grep -E '^served config|^4E:|^served door|^served :|^  (4E|eager)' "$O/door.txt"
grep -E '^served config: ' "$O/door.txt" | grep -qF "$WANT"
served_unit "$O/door.txt"; v4_loaded "$O/door.txt"
grep -q '^served door (decode persistent): 4E tokens equal the eager path.s under this config, 256 of 256' "$O/door.txt"
echo 'gate 3: the v4 door decoded the eager path tokens'

echo "== PART 2: the sweep, LLVQ_CONTEXTS=$CONTEXTS =="
date -u
LLVQ_CONFIG="$P4" LLVQ_CONTEXTS="$CONTEXTS" timeout 14m fusedrun "$S" 256 > "$O/sweep.txt" 2>&1
date -u
grep -v '^  tokens after ' "$O/sweep.txt" | tail -12
grep -E '^served config: ' "$O/sweep.txt" | grep -qF "$WANT"
v4_loaded "$O/sweep.txt"
grep -q '^decode engine: persistent (4E)' "$O/sweep.txt"
test "$(grep -cE '^  prompt [0-9]+: 4E tokens equal the eager path.s under this config, 256 of 256' "$O/sweep.txt")" = 4
test "$(grep -cE '^ +[0-9]+ +[0-9.]+ \[' "$O/sweep.txt")" = 4
echo 'gate 4: the v4 sweep decoded the eager path tokens at every length'
python3 -c "$CMP_PY" door "$O/door.txt" "$V3_DOOR" | tee "$O/compare.txt"
python3 -c "$CMP_PY" sweep "$O/sweep.txt" "$V3_SWEEP" | tee -a "$O/compare.txt"

echo '== PART 3: the pilot 50 GSM8K problems =='
date -u
LLVQ_DATASET_REV="$DATASET_REV" LLVQ_CONFIG="$P4" LLVQ_GSM8K_DUMP="$O/gsm8k-4b-v4-n50.jsonl" \
  timeout 8m gsm8k "$S" cuda 50 > "$O/gsm8k-4b-v4-n50.txt" 2>&1
date -u
grep -vE '^  \[ *[0-9]+/' "$O/gsm8k-4b-v4-n50.txt" | tail -8
tail -1 "$O/gsm8k-4b-v4-n50.jsonl" | grep -q '"end":true'
tail -1 "$O/gsm8k-4b-v4-n50.jsonl" | grep -qF "\"fingerprint\":\"$REPLAY_FP\""
grep -E '^served config: ' "$O/gsm8k-4b-v4-n50.txt" | grep -qF "$WANT"
v4_loaded "$O/gsm8k-4b-v4-n50.txt"
python3 -c "$CMP_PY" gsm8k "$O/gsm8k-4b-v4-n50.jsonl" "$PILOT_REF" | tee -a "$O/compare.txt"
ls -la "$O"
JOB

if [ "$DRY" = 1 ]; then
  JOBSCRIPT="set -euo pipefail
$PRE
$BODY"
  printf '%s\n' "$JOBSCRIPT" | bash -n && echo "DRY_RUN: job script parses; nothing launched; pilot reference of $(echo "$PILOT_REF" | tr ',' '\n' | wc -l | tr -d ' ') rows" >&2
  exit 0
fi

uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor l40sx1 --timeout "$TIMEOUT" \
  --bucket "$BUCKET" --out-mount /out \
  --name "served-v4${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
