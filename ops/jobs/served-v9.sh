#!/usr/bin/env bash
# The served door of the sealed 4B under configs/qwen3-4b-tetra-e4-4e-v9.json, the v6 config with the v9 row loop
# (4E row form 7: v8, with the int4 rows on tiled words), the int4 words tiled at load and held once on the card.
# One job, four parts:
#   (0) the v6 config's door, the card memory it holds (one copy, the file's order), and its tokens against its door;
#   (1) the v9 door: 4E against the eager path under the config, the identity gate, both rates round by round, the
#       card memory, and the load-time checks of the tiled readers;
#   (2) the context sweep at 5, 2,048, 4,096 and 8,192, the identity gated at every length;
#   (3) the pilot's 50 GSM8K problems through the config.
# The tokens of (1) and (2) and the completions of (3) are compared with the v6 config's, read by served-v6
# (docs/mesures/served-v6-2026-11-07-brut/): the same arithmetic owes the same text.
# Preregistered: proofs/preregistration-served-v9-2026-11-07.md, to stamp before this.
#
#   DRY_RUN=1 bash ops/jobs/served-v9.sh                 parses the job script, launches nothing
#   IMAGE_SHA=<space sha> bash ops/jobs/served-v9.sh      launches, after the operator's go
#
# The image must be rebuilt from the commit that carries this file: an image built before it refuses "rows": "v9"
# in served.rs and has no tiled int4 reader. No new binary: oracle, fusedrun and gsm8k are already in the build list
# and the runtime COPY of ops/Dockerfile.cuda; configs/** and llvq-*/** (tv_q4_tile.cu) are already in UPLOAD_ALLOW
# of ops/run.py, and the Dockerfile tests the v9 config's presence.
#
# Every LLVQ_* variable is set on its own command line, never exported. Cost, estimated (l40sx1 $1.80/h, prereg
# section 8): about 16 min, $0.48. Timeout 30m: $0.90 at worst.
#
# The registry row, once the job id and the billed minutes are known (docs/data/jobs.csv):
# <date>,<job_id>,served-v9,l40sx1,<min>,<cost>,served-v9-2026-11-07.txt,"PORTE SERVIE SOUS 4E V9, INT4 UNE COPIE : <door> tok/s a la porte, decodage <x>, x<r> sur eager ; balayage <a> / <b> / <c> / <d> a 5 / 2048 / 4096 / 8192 ; memoire carte v9 <g> Go contre v6 <h> Go ; jetons et GSM8K pilote contre la config v6 : <identiques|DIFFERENTS> ; prereg tamponne avant"
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-served-v9-2026-11-07.md
IMAGE_SHA=${IMAGE_SHA:-}
BUCKET=Pier-Jean/jobs-artifacts
O=/out/served-v9-2026-11-07${RETRY:+-r$RETRY}
TIMEOUT=30m
DRY=${DRY_RUN:-0}
DATASET_REV=740312add88f781978c0658806c59bc2815b9866
CONTEXTS=5,2048,4096,8192

SEALED="sealed-4b-2026-09-23/qwen3-4b-sealed.bin:1418224685"
SEALED_SHA=886391a8c03f66dc269cc65c3598c6627dbdcd259180aff36604ef10d37371b8
CV9_LOCAL=configs/qwen3-4b-tetra-e4-4e-v9.json
CV6_LOCAL=configs/qwen3-4b-tetra-e4-4e-v6.json
CV6_SHA_WANT=ffbeb4157d43e5811eef038581621bea43522495489841bf474ff00198307887
# The served unit under v9: the unit every other config compiles, then tv_q4_tile.cu (12 parts). Pinned by
# llvq-cuda's the_tiled_served_unit_is_the_served_unit_and_the_tile_readers. The v6 door keeps the 11-part unit.
SERVED_SHA_V9=e269ea5d11b7e40ba0682f536f52ed9f09abfaa9fd6f3185eb306edca5c60e53
SERVED_SHA_V6=6193094310bc18acdb0a0c554807ff45f6df09c1f6ab49e0577032874f107812
# 4E's row form 7 module: step3-4e's (d9da09aa, 250,057 B) with the head's own pointers and the Tetra o phase added
# for the 8B and the 14B (proofs/preregistration-served-v9-2026-11-07-ECARTS.md, deviation 1). Pinned by llvq-cuda's
# the_4e_v9_module_is_the_one_the_doors_gate_on.
P4E_V9_SHA=1b422b62328be6752df4add43280565fd2a49cfa20ee26953b301abe7a15c530
REPLAY_FP=94c7b98876288901
# The 4B's vector prologue inputs from v7 on: act, x_a, x_b, ctx, then ln1 and ln2 of each of the 36 layers.
V9_INPUTS=76
# The int4 records of the 4B (36 v, 36 o, 12 downs) and their words, 0.385 GB (step3-4e's relay line).
INT4_RECORDS=84
INT4_WORDS_GB=0.385
# The v6 config's readings (served-v6-2026-11-07): door and sweep token fingerprints, and its pilot dump.
V6_DOOR=61d0654532d35120
V6_SWEEP=5:61d0654532d35120,2048:a4645c00f879ae80,4096:cc9b3414c4d67ba6,8192:9aa830caa48cb332
V6_PILOT=docs/mesures/served-v6-2026-11-07-brut/gsm8k-4b-v6-n50.jsonl
V6_PILOT_SHA=fc2791030e8ef09c7e526983b11dc394a9da5628f2184bc4e2e11ad283e087a8
REF_PY='import hashlib,json,sys
ls=[json.loads(l) for l in open(sys.argv[1],encoding="utf-8").read().split("\n") if l][1:]
print(",".join("%d:%d:%s"%(d["index"],int(d["correct"]),hashlib.sha256(d["completion"].encode()).hexdigest()[:12]) for d in ls if not d.get("end")))'

for c in "$CV9_LOCAL" "$CV6_LOCAL"; do
  [ "$(printf '%s\n' "$(cat "$c")" | shasum -a 256 | cut -d' ' -f1)" = "$(shasum -a 256 < "$c" | cut -d' ' -f1)" ] \
    || { echo "refused: $c does not end in exactly one newline" >&2; exit 1; }
done
grep -q '^  "rows": "v9",$' "$CV9_LOCAL" || { echo "refused: $CV9_LOCAL does not name rows v9" >&2; exit 1; }
CV9=$(cat "$CV9_LOCAL"); CV9_SHA=$(shasum -a 256 < "$CV9_LOCAL" | cut -d' ' -f1)
CV6=$(cat "$CV6_LOCAL"); CV6_SHA=$(shasum -a 256 < "$CV6_LOCAL" | cut -d' ' -f1)
[ "$CV6_SHA" = "$CV6_SHA_WANT" ] || { echo "refused: $CV6_LOCAL is not the config served-v6 read ($CV6_SHA_WANT)" >&2; exit 1; }
[ "$(shasum -a 256 < "$V6_PILOT" | cut -d' ' -f1)" = "$V6_PILOT_SHA" ] || { echo "refused: $V6_PILOT is not $V6_PILOT_SHA" >&2; exit 1; }
PILOT_REF=$(python3 -c "$REF_PY" "$V6_PILOT")

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
  echo "bucket copy verified by size; config v9 $CV9_SHA, v6 $CV6_SHA; image $NOW"
fi

CMP_PY='import re,sys,hashlib,json
mode=sys.argv[1]
if mode=="door":
    t=open(sys.argv[2],encoding="utf-8").read()
    h=re.search(r"^  token ids: ([0-9a-f]{16}) ",t,re.M).group(1)
    print("%s door tokens: %s against the v6 config %s: %s"%(sys.argv[4],h,sys.argv[3],"identical" if h==sys.argv[3] else "DIFFERENT"))
elif mode=="sweep":
    want=dict(x.split(":") for x in sys.argv[3].split(","))
    got={m.group(1):m.group(2) for m in re.finditer(r"^  tokens after (\d+): ([0-9a-f]{16}) ",open(sys.argv[2],encoding="utf-8").read(),re.M)}
    for n,h in want.items():
        print("sweep tokens after %s: %s against the v6 config %s: %s"%(n,got.get(n),h,"identical" if got.get(n)==h else "DIFFERENT"))
elif mode=="memory":
    def used(p):
        return float(re.search(r"^card memory: ([0-9.]+) GB used of ([0-9.]+) GB",open(p,encoding="utf-8").read(),re.M).group(1))
    a,b=used(sys.argv[2]),used(sys.argv[3])
    print("card memory after the install: v9 %.3f GB, v6 %.3f GB, v9 - v6 %+.3f GB (the measurement form held %s GB more)"%(a,b,a-b,sys.argv[4]))
else:
    ref={int(i):(int(c),h) for i,c,h in (x.split(":") for x in sys.argv[3].split(","))}
    ls=[json.loads(l) for l in open(sys.argv[2],encoding="utf-8").read().split("\n") if l]
    head=ls[0]; rows=[d for d in ls[1:] if not d.get("end")]
    same=sum(hashlib.sha256(d["completion"].encode()).hexdigest()[:12]==ref.get(d["index"],(0,""))[1] for d in rows)
    a=sum(int(d["correct"]) for d in rows); g=sum(c for c,_ in ref.values())
    print("GSM8K pilot, %d problems: v9 config %d correct, v6 config %d; completions identical %d/%d; dump header rows %s"%(len(rows),a,g,same,len(rows),head.get("rows")))'

PRE=$(printf 'O=%q\nS=%q\nSB=%q\nSSHA=%q\nCV9=%q\nCV9_SHA=%q\nCV6=%q\nCV6_SHA=%q\nSERVED_SHA_V9=%q\nSERVED_SHA_V6=%q\nP4E_V9_SHA=%q\nDATASET_REV=%q\nCONTEXTS=%q\nREPLAY_FP=%q\nPILOT_REF=%q\nV6_DOOR=%q\nV6_SWEEP=%q\nV9_INPUTS=%q\nINT4_RECORDS=%q\nINT4_WORDS_GB=%q\nCMP_PY=%q\nexport HF_HOME=/tmp/hf' \
  "$O" "/out/${SEALED%%:*}" "${SEALED##*:}" "$SEALED_SHA" "$CV9" "$CV9_SHA" "$CV6" "$CV6_SHA" "$SERVED_SHA_V9" \
  "$SERVED_SHA_V6" "$P4E_V9_SHA" "$DATASET_REV" "$CONTEXTS" "$REPLAY_FP" "$PILOT_REF" "$V6_DOOR" "$V6_SWEEP" \
  "$V9_INPUTS" "$INT4_RECORDS" "$INT4_WORDS_GB" "$CMP_PY")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,clocks.max.sm --format=csv,noheader | tee "$O/gpu.txt"
echo '== gate 1: oracle =='
timeout 5m oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle.txt" | tail -2
grep -q MATCH "$O/oracle.txt"
echo '== gate 2: the object and the configs =='
test "$(stat -c %s "$S")" = "$SB"
sha256sum "$S" | tee "$O/files.sha256"
test "$(cut -d' ' -f1 "$O/files.sha256")" = "$SSHA"
P9=$O/qwen3-4b-tetra-e4-4e-v9.json
P6=$O/qwen3-4b-tetra-e4-4e-v6.json
printf '%s\n' "$CV9" > "$P9"
printf '%s\n' "$CV6" > "$P6"
sha256sum "$P9" "$P6" | tee -a "$O/files.sha256"
test "$(sha256sum "$P9" | cut -d' ' -f1)" = "$CV9_SHA"
test "$(sha256sum "$P6" | cut -d' ' -f1)" = "$CV6_SHA"
WANT='attention kernel from position 0 on grow(256), decode persistent (4E rows v9, one launch a token run)'
unit_is() { grep -qF "sha256 $2" "$1" || { echo "gate: $1 did not compile the served unit $2" >&2; exit 1; }; }
v9_loaded() {  # the load and 4E install lines: the tiled int4 words held once, row form 7, every input aligned, RoPE, 512 threads
  grep -qF "int4 layout: tiled (LLVQ_CONFIG), $INT4_RECORDS records, words $INT4_WORDS_GB GB on the card, held once" "$1" \
    || { echo "gate: $1 did not load the int4 words tiled" >&2; exit 1; }
  grep -qE '^int4: tv_q4_hht equals tv_q4_ht bit for bit on ' "$1" || { echo "gate: $1 has no tiled f16/f32 parity line" >&2; exit 1; }
  test "$(grep -cE "^int4: tv_q4_hht on the tiled words equals tv_q4_hh on the file's order bit for bit on .* \(([0-9]+) rows, d_in (2560|4096|9728)\); the file-order copy is freed" "$1")" = 3 \
    || { echo "gate: $1 did not check the tiled reader on the three int4 widths" >&2; exit 1; }
  grep -qF "4E: persistent module 252757 bytes, sha256 $P4E_V9_SHA, row form 7," "$1" || { echo "gate: $1 did not install the row form 7 module $P4E_V9_SHA" >&2; exit 1; }
  grep -qF "4E: head tied to the embedding's table (151936 rows)" "$1" || { echo "gate: $1 did not read the tied head" >&2; exit 1; }
  grep -qF "4E v9: the $V9_INPUTS prologue inputs are 16-byte aligned; the vector path runs on every layer" "$1" \
    || { echo "gate: $1 did not run the vector prologues on the $V9_INPUTS inputs" >&2; exit 1; }
  grep -qF "4E v9: the int4 records are read in place, tiled at load and held once on the card (36 layers)" "$1" \
    || { echo "gate: $1 did not read the int4 records in place" >&2; exit 1; }
  if grep -q 'relaid into tiled words' "$1"; then echo "gate: $1 relaid a second copy" >&2; exit 1; fi
  grep -q '^4E: RoPE forms equal to candle' "$1" || { echo "gate: $1 has no RoPE line" >&2; exit 1; }
  grep -qE '^4E: pers_4e_tokens_f[0-9]_t512: [0-9]+ registers, 0 B local' "$1" || { echo "gate: $1 spills or is not the 512-thread symbol" >&2; exit 1; }
}

echo '== PART 0: the v6 config door, its card memory =='
date -u
LLVQ_CONFIG="$P6" timeout 8m fusedrun "$S" 256 > "$O/door-v6.txt" 2>&1
date -u
grep -E '^served config|^4E v6|^int4|^card memory|^served door|^served :|^  (4E|eager)' "$O/door-v6.txt"
unit_is "$O/door-v6.txt" "$SERVED_SHA_V6"
if grep -q '^int4 layout: tiled' "$O/door-v6.txt"; then echo 'gate: the v6 config loaded tiled words' >&2; exit 1; fi
grep -q '^card memory: ' "$O/door-v6.txt"
grep -q '^served door (decode persistent): 4E tokens equal the eager path.s under this config, 256 of 256' "$O/door-v6.txt"
python3 -c "$CMP_PY" door "$O/door-v6.txt" "$V6_DOOR" v6 | tee "$O/compare.txt"
echo 'gate 3a: the v6 door decoded the eager path tokens'

echo '== PART 1: the v9 door, 256 tokens =='
date -u
LLVQ_CONFIG="$P9" timeout 8m fusedrun "$S" 256 > "$O/door.txt" 2>&1
date -u
grep -E '^served config|^int4|^4E:|^4E v9|^card memory|^served door|^served :|^  (4E|eager)' "$O/door.txt"
grep -E '^served config: ' "$O/door.txt" | grep -qF "$WANT"
unit_is "$O/door.txt" "$SERVED_SHA_V9"; v9_loaded "$O/door.txt"
grep -q '^served door (decode persistent): 4E tokens equal the eager path.s under this config, 256 of 256' "$O/door.txt"
echo 'gate 3: the v9 door decoded the eager path tokens'
python3 -c "$CMP_PY" door "$O/door.txt" "$V6_DOOR" v9 | tee -a "$O/compare.txt"
python3 -c "$CMP_PY" memory "$O/door.txt" "$O/door-v6.txt" "$INT4_WORDS_GB" | tee -a "$O/compare.txt"

echo "== PART 2: the sweep, LLVQ_CONTEXTS=$CONTEXTS =="
date -u
LLVQ_CONFIG="$P9" LLVQ_CONTEXTS="$CONTEXTS" timeout 14m fusedrun "$S" 256 > "$O/sweep.txt" 2>&1
date -u
grep -v '^  tokens after ' "$O/sweep.txt" | tail -12
grep -E '^served config: ' "$O/sweep.txt" | grep -qF "$WANT"
v9_loaded "$O/sweep.txt"
grep -q '^decode engine: persistent (4E)' "$O/sweep.txt"
test "$(grep -cE '^  prompt [0-9]+: 4E tokens equal the eager path.s under this config, 256 of 256' "$O/sweep.txt")" = 4
test "$(grep -cE '^ +[0-9]+ +[0-9.]+ \[' "$O/sweep.txt")" = 4
echo 'gate 4: the v9 sweep decoded the eager path tokens at every length'
python3 -c "$CMP_PY" sweep "$O/sweep.txt" "$V6_SWEEP" | tee -a "$O/compare.txt"

echo '== PART 3: the pilot 50 GSM8K problems =='
date -u
LLVQ_DATASET_REV="$DATASET_REV" LLVQ_CONFIG="$P9" LLVQ_GSM8K_DUMP="$O/gsm8k-4b-v9-n50.jsonl" \
  timeout 8m gsm8k "$S" cuda 50 > "$O/gsm8k-4b-v9-n50.txt" 2>&1
date -u
grep -vE '^  \[ *[0-9]+/' "$O/gsm8k-4b-v9-n50.txt" | tail -8
tail -1 "$O/gsm8k-4b-v9-n50.jsonl" | grep -q '"end":true'
tail -1 "$O/gsm8k-4b-v9-n50.jsonl" | grep -qF "\"fingerprint\":\"$REPLAY_FP\""
head -1 "$O/gsm8k-4b-v9-n50.jsonl" | grep -qF '"rows":"v9"'
grep -E '^served config: ' "$O/gsm8k-4b-v9-n50.txt" | grep -qF "$WANT"
v9_loaded "$O/gsm8k-4b-v9-n50.txt"
python3 -c "$CMP_PY" gsm8k "$O/gsm8k-4b-v9-n50.jsonl" "$PILOT_REF" | tee -a "$O/compare.txt"
ls -la "$O"
JOB

if [ "$DRY" = 1 ]; then
  JOBSCRIPT="set -euo pipefail
$PRE
$BODY"
  printf '%s\n' "$JOBSCRIPT" | bash -n && echo "DRY_RUN: job script parses; nothing launched; config v9 $CV9_SHA, v6 $CV6_SHA; pilot reference of $(echo "$PILOT_REF" | tr ',' '\n' | wc -l | tr -d ' ') rows" >&2
  exit 0
fi

uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor l40sx1 --timeout "$TIMEOUT" \
  --bucket "$BUCKET" --out-mount /out \
  --name "served-v9${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
