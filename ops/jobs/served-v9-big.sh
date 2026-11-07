#!/usr/bin/env bash
# The served door of the sealed 8B or 14B under its v9 config (4E v9, the int4 words tiled and held once), one job a
# size. Five parts:
#   (0) the size's served grouped config at the door: its rate, its tokens, its card memory;
#   (1) the v9 door: 4E against the eager path under the v9 config (the identity gate), the card memory;
#   (2) the context sweep at 5, 2,048, 4,096 and 8,192, the identity gated at every length;
#   (3) the pilot's 50 GSM8K problems through the v9 config, read against the grouped census rows of the same problems;
#   (4) the minimal quality check CLAUDE.md asks of a new attention form: the kernel form on every decode row from
#       position 0 against the served grouped form, paired WikiText-2 decode-path perplexity, inside +-0.12 %.
# 4E owes the eager path's tokens under the v9 config (both run the kernel form). The v9 config does not owe the
# grouped config's tokens: the attention form differs, which is why part 4 exists.
# Preregistered: proofs/preregistration-served-v9-8b-14b-2026-11-07.md, to stamp before this.
#
#   SIZE=8b|14b DRY_RUN=1 bash ops/jobs/served-v9-big.sh                 parses the job script, launches nothing
#   SIZE=8b|14b IMAGE_SHA=<space sha> bash ops/jobs/served-v9-big.sh      launches
#
# The image must carry the 8B/14B 4E work (the head's own pointers, the Tetra o phase). Binaries: oracle, fusedrun,
# gsm8k, already in the build list and the runtime COPY of ops/Dockerfile.cuda; configs/** in UPLOAD_ALLOW.
#
# Cost, estimated (l40sx1 $1.80/h, prereg section 8): 8B about 35 min, $1.05, timeout 60m ($1.80); 14B about 60 min,
# $1.80, timeout 95m ($2.85).
#
# The registry row (docs/data/jobs.csv):
# <date>,<job_id>,served-v9-<size>,l40sx1,<min>,<cost>,served-v9-<size>-2026-11-07.txt,"PORTE SERVIE SOUS 4E V9 AU <SIZE> : <door> tok/s contre <grouped> en groupee ; jetons 4E = eager <oui|non> ; balayage <a>/<b>/<c>/<d> ; pilote GSM8K <x>/50 contre <y>/50 au recensement groupe ; ppl noyau des 0 contre groupee <d> pc [<lo> ; <hi>] ; prereg tamponne avant"
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

SIZE=${SIZE:?SIZE=8b or SIZE=14b}
case "$SIZE" in
  8b)
    SEALED="sealed-8b27-2026-09-24/qwen3-8b-sealed-B.bin:2815098745"
    SEALED_SHA=7bdb9a5503518081b1f652a93215250898caae5cc1b6a2f33ee9fb6094565c13
    LAYERS=36; INT4_RECORDS=53; INT4_WIDTHS=2; V9_INPUTS=76; TETRA_O='4E: 36 Tetra o_proj read under their own rotation, 0 int4'
    CENSUS=docs/mesures/gsm8k-grouped-8b-2026-11-04-brut/gsm8k-8b-grouped.jsonl
    CENSUS_SHA=$(shasum -a 256 < "$CENSUS" | cut -d' ' -f1)
    TIMEOUT=60m; DOOR_T=10m; SWEEP_T=20m; GSM_T=12m; PPL_T=20m ;;
  14b)
    SEALED="sealed-14b-2026-09-23/qwen3-14b-sealed.bin:5087000541"
    SEALED_SHA=61db37fe7ce8e6a57c5c59f257319c796574981ecb5e450a02c73dd3c17e2ca0
    LAYERS=40; INT4_RECORDS=99; INT4_WIDTHS=2; V9_INPUTS=84; TETRA_O=''
    CENSUS=docs/mesures/gsm8k-grouped-14b-2026-11-04-brut/gsm8k-14b-grouped.jsonl
    CENSUS_SHA=$(shasum -a 256 < "$CENSUS" | cut -d' ' -f1)
    TIMEOUT=95m; DOOR_T=14m; SWEEP_T=30m; GSM_T=18m; PPL_T=30m ;;
  *) echo "refused: SIZE=$SIZE, the sizes are 8b and 14b" >&2; exit 1 ;;
esac

PREREG=proofs/preregistration-served-v9-8b-14b-2026-11-07.md
IMAGE_SHA=${IMAGE_SHA:-}
BUCKET=Pier-Jean/jobs-artifacts
O=/out/served-v9-$SIZE-2026-11-07${RETRY:+-r$RETRY}
DRY=${DRY_RUN:-0}
DATASET_REV=740312add88f781978c0658806c59bc2815b9866
CONTEXTS=5,2048,4096,8192
CV9_LOCAL=configs/qwen3-$SIZE-tetra-e4-4e-v9.json
CG_LOCAL=configs/qwen3-$SIZE-tetra-e4-grouped.json
SERVED_SHA_V9=e269ea5d11b7e40ba0682f536f52ed9f09abfaa9fd6f3185eb306edca5c60e53
P4E_V9_SHA=1b422b62328be6752df4add43280565fd2a49cfa20ee26953b301abe7a15c530
REPLAY_FP=94c7b98876288901
FLAGS='LLVQ_FUSED_LAYOUT=tetra48 LLVQ_ROT_SHARE=1 LLVQ_FUSE=0 LLVQ_KV=f16 LLVQ_EMBED=q4'
# The band: 6 windows of 4,096 from window index 60 (tokens 245,760 to 270,335), every position under the kernel form.
PPL_SPEC=4096,6,60

for c in "$CV9_LOCAL" "$CG_LOCAL"; do
  [ "$(printf '%s\n' "$(cat "$c")" | shasum -a 256 | cut -d' ' -f1)" = "$(shasum -a 256 < "$c" | cut -d' ' -f1)" ] \
    || { echo "refused: $c does not end in exactly one newline" >&2; exit 1; }
done
grep -q '^  "rows": "v9",$' "$CV9_LOCAL" || { echo "refused: $CV9_LOCAL does not name rows v9" >&2; exit 1; }
CV9=$(cat "$CV9_LOCAL"); CV9_SHA=$(shasum -a 256 < "$CV9_LOCAL" | cut -d' ' -f1)
CG=$(cat "$CG_LOCAL"); CG_SHA=$(shasum -a 256 < "$CG_LOCAL" | cut -d' ' -f1)
# The census rows of the 50 pilot problems (the indices gsm8k's 50-problem draw takes, as the 4B pilots ran them).
PILOT_IDX=33,46,75,160,162,185,222,231,422,453,461,477,486,515,547,557,604,617,657,660,661,703,717,750,758,760,783,801,809,810,811,826,834,843,888,895,915,955,965,1042,1072,1108,1140,1169,1170,1217,1219,1258,1260,1266
REF_PY='import hashlib,json,sys
want=set(int(x) for x in sys.argv[2].split(","))
ls=[json.loads(l) for l in open(sys.argv[1],encoding="utf-8").read().split("\n") if l][1:]
rows=[d for d in ls if not d.get("end") and d["index"] in want]
assert len(rows)==len(want), "the census lacks pilot rows"
print(",".join("%d:%d:%s"%(d["index"],int(d["correct"]),hashlib.sha256(d["completion"].encode()).hexdigest()[:12]) for d in rows))'
PILOT_REF=$(python3 -c "$REF_PY" "$CENSUS" "$PILOT_IDX")

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
  echo "bucket copy verified by size; config v9 $CV9_SHA, grouped $CG_SHA; census $CENSUS_SHA; image $NOW"
fi

CMP_PY='import re,sys,hashlib,json
mode=sys.argv[1]
def toks(p):
    return re.search(r"^  token ids: ([0-9a-f]{16}) ",open(p,encoding="utf-8").read(),re.M).group(1)
def rate(p):
    m=re.search(r"^served : loaded in +[0-9.]+ s, +([0-9.]+) tok/s \[([0-9.]+).([0-9.]+), ",open(p,encoding="utf-8").read(),re.M)
    return m.group(1)
def mem(p):
    return float(re.search(r"^card memory: ([0-9.]+) GB used",open(p,encoding="utf-8").read(),re.M).group(1))
if mode=="door":
    v,g=sys.argv[2],sys.argv[3]
    print("door: v9 %s tok/s, grouped %s tok/s (two processes, one job); tokens v9 %s, grouped %s (not owed: the attention form differs); card memory v9 %.3f GB, grouped %.3f GB"%(rate(v),rate(g),toks(v),toks(g),mem(v),mem(g)))
else:
    ref={int(i):(int(c),h) for i,c,h in (x.split(":") for x in sys.argv[3].split(","))}
    ls=[json.loads(l) for l in open(sys.argv[2],encoding="utf-8").read().split("\n") if l]
    head=ls[0]; rows=[d for d in ls[1:] if not d.get("end")]
    same=sum(hashlib.sha256(d["completion"].encode()).hexdigest()[:12]==ref.get(d["index"],(0,""))[1] for d in rows)
    a=sum(int(d["correct"]) for d in rows); g=sum(ref[d["index"]][0] for d in rows if d["index"] in ref)
    print("GSM8K pilot, %d problems: v9 config %d correct, grouped census on the same problems %d; completions identical %d/%d (not owed); dump header rows %s"%(len(rows),a,g,same,len(rows),head.get("rows")))'

BAND_PY='import re,sys
p=sys.argv[1]
ls=[l for l in open(p,encoding="utf-8").read().splitlines() if re.match(r"^  from 0 +[\d.]+ ",l)]
m=re.search(r"([+-][\d.]+) % \[([+-][\d.]+) ; ([+-][\d.]+)\]",ls[0]) if len(ls)==1 else None
if not m: print("%s: no single from-0 summary line"%p); sys.exit(1)
mid,lo,hi=(float(x) for x in m.groups())
ok=lo>=-0.12 and hi<=0.12
print("perplexity, kernel from position 0 against grouped: %+.4f %% [%+.4f ; %+.4f], %s the +-0.12 %% band"%(mid,lo,hi,"inside" if ok else "NOT inside"))
print("band: inside" if ok else "band: not inside")'

PRE=$(printf 'O=%q\nS=%q\nSB=%q\nSSHA=%q\nSIZE=%q\nCV9=%q\nCV9_SHA=%q\nCG=%q\nCG_SHA=%q\nSERVED_SHA_V9=%q\nP4E_V9_SHA=%q\nDATASET_REV=%q\nCONTEXTS=%q\nREPLAY_FP=%q\nPILOT_REF=%q\nLAYERS=%q\nINT4_RECORDS=%q\nINT4_WIDTHS=%q\nV9_INPUTS=%q\nTETRA_O=%q\nFLAGS=%q\nPPL_SPEC=%q\nDOOR_T=%q\nSWEEP_T=%q\nGSM_T=%q\nPPL_T=%q\nCMP_PY=%q\nBAND_PY=%q\nexport HF_HOME=/tmp/hf' \
  "$O" "/out/${SEALED%%:*}" "${SEALED##*:}" "$SEALED_SHA" "$SIZE" "$CV9" "$CV9_SHA" "$CG" "$CG_SHA" "$SERVED_SHA_V9" \
  "$P4E_V9_SHA" "$DATASET_REV" "$CONTEXTS" "$REPLAY_FP" "$PILOT_REF" "$LAYERS" "$INT4_RECORDS" "$INT4_WIDTHS" \
  "$V9_INPUTS" "$TETRA_O" "$FLAGS" "$PPL_SPEC" "$DOOR_T" "$SWEEP_T" "$GSM_T" "$PPL_T" "$CMP_PY" "$BAND_PY")

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
P9=$O/qwen3-$SIZE-tetra-e4-4e-v9.json
PG=$O/qwen3-$SIZE-tetra-e4-grouped.json
printf '%s\n' "$CV9" > "$P9"
printf '%s\n' "$CG" > "$PG"
sha256sum "$P9" "$PG" | tee -a "$O/files.sha256"
test "$(sha256sum "$P9" | cut -d' ' -f1)" = "$CV9_SHA"
test "$(sha256sum "$PG" | cut -d' ' -f1)" = "$CG_SHA"
WANT='attention kernel from position 0 on grow(256), decode persistent (4E rows v9, one launch a token run)'
v9_loaded() {
  grep -qF "sha256 $SERVED_SHA_V9" "$1" || { echo "gate: $1 did not compile the v9 served unit" >&2; exit 1; }
  grep -qE "^int4 layout: tiled \(LLVQ_CONFIG\), $INT4_RECORDS records, words [0-9.]+ GB on the card, held once" "$1" \
    || { echo "gate: $1 did not load the $INT4_RECORDS int4 records tiled" >&2; exit 1; }
  grep -qE '^int4: tv_q4_hht equals tv_q4_ht bit for bit on ' "$1" || { echo "gate: $1 has no tiled f16/f32 parity line" >&2; exit 1; }
  test "$(grep -cE "^int4: tv_q4_hht on the tiled words equals tv_q4_hh on the file's order bit for bit on " "$1")" = "$INT4_WIDTHS" \
    || { echo "gate: $1 did not check the tiled reader on its $INT4_WIDTHS int4 widths" >&2; exit 1; }
  grep -qF "4E: persistent module 252757 bytes, sha256 $P4E_V9_SHA, row form 7," "$1" || { echo "gate: $1 did not install the v9 module" >&2; exit 1; }
  grep -qF "4E: head on a table of its own (untied) (151936 rows)" "$1" || { echo "gate: $1 did not read the untied head" >&2; exit 1; }
  if [ -n "$TETRA_O" ]; then grep -qF "$TETRA_O" "$1" || { echo "gate: $1 did not read the Tetra o_proj" >&2; exit 1; }; fi
  grep -qF "4E v9: the $V9_INPUTS prologue inputs are 16-byte aligned; the vector path runs on every layer" "$1" \
    || { echo "gate: $1 did not run the vector prologues on the $V9_INPUTS inputs" >&2; exit 1; }
  grep -qF "4E v9: the int4 records are read in place, tiled at load and held once on the card ($LAYERS layers)" "$1" \
    || { echo "gate: $1 did not read the int4 records in place" >&2; exit 1; }
  if grep -q 'relaid into tiled words' "$1"; then echo "gate: $1 relaid a second copy" >&2; exit 1; fi
  grep -q '^4E: RoPE forms equal to candle' "$1" || { echo "gate: $1 has no RoPE line" >&2; exit 1; }
  grep -qE '^4E: pers_4e_tokens_f[0-9]_t512: [0-9]+ registers, 0 B local' "$1" || { echo "gate: $1 spills or is not the 512-thread symbol" >&2; exit 1; }
}

echo '== PART 0: the served grouped config at the door =='
date -u
LLVQ_CONFIG="$PG" timeout "$DOOR_T" fusedrun "$S" 256 > "$O/door-grouped.txt" 2>&1
date -u
grep -E '^served config|^card memory|^served :' "$O/door-grouped.txt"
grep -q '^card memory: ' "$O/door-grouped.txt"
grep -qE '^  token ids: [0-9a-f]{16} ' "$O/door-grouped.txt"

echo '== PART 1: the v9 door, 256 tokens =='
date -u
LLVQ_CONFIG="$P9" timeout "$DOOR_T" fusedrun "$S" 256 > "$O/door.txt" 2>&1
date -u
grep -E '^served config|^int4|^4E:|^4E v9|^card memory|^served door|^served :|^  (4E|eager)' "$O/door.txt"
grep -E '^served config: ' "$O/door.txt" | grep -qF "$WANT"
v9_loaded "$O/door.txt"
grep -q '^served door (decode persistent): 4E tokens equal the eager path.s under this config, 256 of 256' "$O/door.txt"
echo 'gate 3: the v9 door decoded the eager path tokens'
python3 -c "$CMP_PY" door "$O/door.txt" "$O/door-grouped.txt" | tee "$O/compare.txt"

echo "== PART 2: the sweep, LLVQ_CONTEXTS=$CONTEXTS =="
date -u
LLVQ_CONFIG="$P9" LLVQ_CONTEXTS="$CONTEXTS" timeout "$SWEEP_T" fusedrun "$S" 256 > "$O/sweep.txt" 2>&1
date -u
grep -v '^  tokens after ' "$O/sweep.txt" | tail -12
grep -E '^served config: ' "$O/sweep.txt" | grep -qF "$WANT"
v9_loaded "$O/sweep.txt"
grep -q '^decode engine: persistent (4E)' "$O/sweep.txt"
test "$(grep -cE '^  prompt [0-9]+: 4E tokens equal the eager path.s under this config, 256 of 256' "$O/sweep.txt")" = 4
test "$(grep -cE '^ +[0-9]+ +[0-9.]+ \[' "$O/sweep.txt")" = 4
echo 'gate 4: the v9 sweep decoded the eager path tokens at every length'

echo '== PART 3: the pilot 50 GSM8K problems =='
date -u
LLVQ_DATASET_REV="$DATASET_REV" LLVQ_CONFIG="$P9" LLVQ_GSM8K_DUMP="$O/gsm8k-$SIZE-v9-n50.jsonl" \
  timeout "$GSM_T" gsm8k "$S" cuda 50 > "$O/gsm8k-$SIZE-v9-n50.txt" 2>&1
date -u
grep -vE '^  \[ *[0-9]+/' "$O/gsm8k-$SIZE-v9-n50.txt" | tail -8
tail -1 "$O/gsm8k-$SIZE-v9-n50.jsonl" | grep -q '"end":true'
tail -1 "$O/gsm8k-$SIZE-v9-n50.jsonl" | grep -qF "\"fingerprint\":\"$REPLAY_FP\""
head -1 "$O/gsm8k-$SIZE-v9-n50.jsonl" | grep -qF '"rows":"v9"'
grep -E '^served config: ' "$O/gsm8k-$SIZE-v9-n50.txt" | grep -qF "$WANT"
v9_loaded "$O/gsm8k-$SIZE-v9-n50.txt"
python3 -c "$CMP_PY" gsm8k "$O/gsm8k-$SIZE-v9-n50.jsonl" "$PILOT_REF" | tee -a "$O/compare.txt"

echo "== PART 4: the kernel form from position 0 against the grouped form, perplexity ($PPL_SPEC) =="
date -u
env $FLAGS LLVQ_GQA_PPL="$PPL_SPEC,kernel0" timeout "$PPL_T" fusedrun "$S" > "$O/ppl.txt" 2>&1
date -u
tail -9 "$O/ppl.txt"
grep -qE '^attn_dec_g[45]_u64: the card equals the CPU replay bit for bit on [0-9]+ calls' "$O/ppl.txt"
grep -qF '  the kernel form takes positions 0 and later' "$O/ppl.txt"
test "$(grep -cxF '  window   1: prefill path bit for bit between the two arms' "$O/ppl.txt")" = 1
python3 -c "$BAND_PY" "$O/ppl.txt" | tee "$O/band.txt"
cat "$O/band.txt" >> "$O/compare.txt"
ls -la "$O"
JOB

if [ "$DRY" = 1 ]; then
  JOBSCRIPT="set -euo pipefail
$PRE
$BODY"
  printf '%s\n' "$JOBSCRIPT" | bash -n && echo "DRY_RUN $SIZE: job script parses; nothing launched; config v9 $CV9_SHA, grouped $CG_SHA; pilot reference of $(echo "$PILOT_REF" | tr ',' '\n' | wc -l | tr -d ' ') rows" >&2
  exit 0
fi

uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor l40sx1 --timeout "$TIMEOUT" \
  --bucket "$BUCKET" --out-mount /out \
  --name "served-v9-$SIZE${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
