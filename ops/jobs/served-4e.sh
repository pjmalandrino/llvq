#!/usr/bin/env bash
# The served door of the sealed 4B under configs/qwen3-4b-tetra-e4-4e.json (the kernel attention form on every
# decode row from position 0, decoded by 4E), against the served config (configs/qwen3-4b-tetra-e4-kernel-from1280.json).
# One job, four parts:
#   (1) the plain door under the 4E config: 4E against the eager path under the SAME config, the identity gate, and
#       both rates round by round;
#   (2) the context sweep at 5, 2,048, 4,096 and 8,192 under the 4E config, the identity gated at every length;
#   (3) the door and the sweep under the served config, for the speed beside it and the token agreement, reported;
#   (4) the pilot's 50 GSM8K problems through the 4E config, compared with the grouped census rows, reported.
# Preregistered: proofs/preregistration-served-4e-2026-11-05.md, to stamp before this.
#
#   DRY_RUN=1 bash ops/jobs/served-4e.sh                 parses the job script, launches nothing
#   IMAGE_SHA=<space sha> bash ops/jobs/served-4e.sh      launches, after the operator's go
#
# Every LLVQ_* variable is set on its own command line, never exported. Cost, estimated (l40sx1 $1.80/h, prereg
# section 8): about 23 min, $0.69. Timeout 45m: $1.35 at worst.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-served-4e-2026-11-05.md
IMAGE_SHA=${IMAGE_SHA:-}
BUCKET=Pier-Jean/jobs-artifacts
O=/out/served-4e-2026-11-05${RETRY:+-r$RETRY}
TIMEOUT=45m
DRY=${DRY_RUN:-0}
DATASET_REV=740312add88f781978c0658806c59bc2815b9866
CONTEXTS=5,2048,4096,8192

SEALED="sealed-4b-2026-09-23/qwen3-4b-sealed.bin:1418224685"
SEALED_SHA=886391a8c03f66dc269cc65c3598c6627dbdcd259180aff36604ef10d37371b8
C4E_LOCAL=configs/qwen3-4b-tetra-e4-4e.json
CK_LOCAL=configs/qwen3-4b-tetra-e4-kernel-from1280.json
# The served unit's digest (llvq-cuda/src/pers.rs::the_served_unit_is_the_published_one): the eager path's kernels.
SERVED_SHA=6193094310bc18acdb0a0c554807ff45f6df09c1f6ab49e0577032874f107812
# The pilot's 50 problems (ops/jobs/attn-t8-r2.sh): their prompt fingerprint, and the grouped census rows at their
# indices as index:correct:sha256(completion)[:12], from
# docs/mesures/gsm8k-grouped-2026-11-04-brut/gsm8k-4b-grouped.jsonl (sha256 eda98402...), 40 of 50 correct.
REPLAY_FP=94c7b98876288901
GRP_CENSUS=docs/mesures/gsm8k-grouped-2026-11-04-brut/gsm8k-4b-grouped.jsonl
GRP_CENSUS_SHA=eda98402a34de849e88800106acb189d6b5d5667bea39db5b3170a1eddca3969
PILOT=docs/mesures/gsm8k-wave1-2026-09-26-brut/smoke/gsm8k-4b-served.jsonl
REF_PY='import hashlib,json,sys
def rows(p):
    ls=[json.loads(l) for l in open(p,encoding="utf-8").read().split("\n") if l][1:]
    return [d for d in ls if not d.get("end")]
g={d["index"]:d for d in rows(sys.argv[1])}
print(",".join("%d:%d:%s"%(i,int(g[i]["correct"]),hashlib.sha256(g[i]["completion"].encode()).hexdigest()[:12]) for i in [d["index"] for d in rows(sys.argv[2])]))'

for f in "$C4E_LOCAL" "$CK_LOCAL"; do
  [ "$(printf '%s\n' "$(cat "$f")" | shasum -a 256 | cut -d' ' -f1)" = "$(shasum -a 256 < "$f" | cut -d' ' -f1)" ] \
    || { echo "refused: $f does not end in exactly one newline" >&2; exit 1; }
done
C4E=$(cat "$C4E_LOCAL"); C4E_SHA=$(shasum -a 256 < "$C4E_LOCAL" | cut -d' ' -f1)
CKC=$(cat "$CK_LOCAL"); CK_SHA=$(shasum -a 256 < "$CK_LOCAL" | cut -d' ' -f1)
[ "$(shasum -a 256 < "$GRP_CENSUS" | cut -d' ' -f1)" = "$GRP_CENSUS_SHA" ] \
  || { echo "refused: $GRP_CENSUS is not $GRP_CENSUS_SHA" >&2; exit 1; }
PILOT_REF=$(python3 -c "$REF_PY" "$GRP_CENSUS" "$PILOT")

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
  echo "bucket copy verified by size; configs $C4E_SHA (4E) / $CK_SHA (served); image $NOW"
fi

# The comparisons, read from the logs after the runs. Never gated beyond what the prereg's gates name.
CMP_PY='import re,sys
mode=sys.argv[1]
def door(p):
    t=open(p,encoding="utf-8").read()
    ids=re.search(r"^  token ids: ([0-9a-f]{16}) ([\d,]+)$",t,re.M)
    rate=re.search(r"^served : loaded in +[\d.]+ s, +([\d.]+) tok/s \[([\d.]+).([\d.]+)",t,re.M)
    return [int(x) for x in ids.group(2).split(",")], tuple(float(x) for x in rate.groups())
def sweep(p,lengths):
    rate,tok={},{}
    for l in open(p,encoding="utf-8").read().splitlines():
        m=re.match(r"^ +(\d+) +[\d.]+ \[[^\]]*\] +([\d.]+) \[([\d.]+)\D+([\d.]+)\] +[\d.]+$",l)
        if m: rate[int(m.group(1))]=tuple(float(m.group(i)) for i in (2,3,4))
        m=re.match(r"^  tokens after (\d+): ([0-9a-f]{16}) ([\d,]+)$",l)
        if m: tok[int(m.group(1))]=[int(x) for x in m.group(3).split(",")]
    assert sorted(rate)==lengths and sorted(tok)==lengths, (p,sorted(rate),sorted(tok))
    return rate,tok
def agree(a,b):
    if a==b: return "identical (%d)"%len(a)
    i=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
    return "first divergence at token %d of %d"%(i,len(a))
if mode=="door":
    (a,ra),(b,rb)=door(sys.argv[2]),door(sys.argv[3])
    print("door: 4E config %.1f tok/s [%.1f-%.1f], served config %.1f tok/s [%.1f-%.1f]; medians 4E/served %.4f (two processes, not paired round by round); tokens: %s"%(ra+rb+(ra[0]/rb[0],agree(a,b))))
elif mode=="sweep":
    L=[int(x) for x in sys.argv[4].split(",")]
    (r4,t4),(rk,tk)=sweep(sys.argv[2],L),sweep(sys.argv[3],L)
    print("  prompt   4E config tok/s [range]   served config tok/s [range]   medians 4E/served   tokens")
    for n in L:
        print("  %6d   %6.1f [%.1f-%.1f]       %6.1f [%.1f-%.1f]          %.4f            %s"%((n,)+r4[n]+rk[n]+(r4[n][0]/rk[n][0],agree(t4[n],tk[n]))))
else:
    import hashlib,json
    ref={int(i):(int(c),h) for i,c,h in (x.split(":") for x in sys.argv[3].split(","))}
    ls=[json.loads(l) for l in open(sys.argv[2],encoding="utf-8").read().split("\n") if l][1:]
    rows=[d for d in ls if not d.get("end")]
    assert sorted(d["index"] for d in rows)==sorted(ref), "the 4E dump does not hold the pilot indices"
    same=sum(hashlib.sha256(d["completion"].encode()).hexdigest()[:12]==ref[d["index"]][1] for d in rows)
    a=sum(int(d["correct"]) for d in rows); g=sum(c for c,_ in ref.values())
    b10=sum(1 for d in rows if int(d["correct"])==1 and ref[d["index"]][0]==0)
    b01=sum(1 for d in rows if int(d["correct"])==0 and ref[d["index"]][0]==1)
    from math import comb
    n=b10+b01; k=min(b10,b01)
    p=min(1.0,2*sum(comb(n,i) for i in range(k+1))/2**n) if n else 1.0
    print("GSM8K pilot, 50 problems: 4E config %d/50 correct, grouped census %d/50; completions identical %d/50; discordant %d (4E right %d, census right %d), McNemar exact p = %.3f; delta %+.1f pp"%(a,g,same,n,b10,b01,p,2.0*(a-g)))'

PRE=$(printf 'O=%q\nS=%q\nSB=%q\nSSHA=%q\nC4E=%q\nCKC=%q\nC4E_SHA=%q\nCK_SHA=%q\nSERVED_SHA=%q\nDATASET_REV=%q\nCONTEXTS=%q\nREPLAY_FP=%q\nPILOT_REF=%q\nCMP_PY=%q\nexport HF_HOME=/tmp/hf' \
  "$O" "/out/${SEALED%%:*}" "${SEALED##*:}" "$SEALED_SHA" "$C4E" "$CKC" "$C4E_SHA" "$CK_SHA" "$SERVED_SHA" \
  "$DATASET_REV" "$CONTEXTS" "$REPLAY_FP" "$PILOT_REF" "$CMP_PY")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,clocks.max.sm --format=csv,noheader | tee "$O/gpu.txt"
echo '== gate 1: oracle =='
timeout 5m oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle.txt" | tail -2
grep -q MATCH "$O/oracle.txt"
echo '== gate 2: the object and both configs =='
test "$(stat -c %s "$S")" = "$SB"
sha256sum "$S" | tee "$O/files.sha256"
test "$(cut -d' ' -f1 "$O/files.sha256")" = "$SSHA"
P4=$O/qwen3-4b-tetra-e4-4e.json; PK=$O/qwen3-4b-tetra-e4-kernel-from1280.json
printf '%s\n' "$C4E" > "$P4"; printf '%s\n' "$CKC" > "$PK"
sha256sum "$P4" "$PK" | tee -a "$O/files.sha256"
test "$(sha256sum "$P4" | cut -d' ' -f1)" = "$C4E_SHA"
test "$(sha256sum "$PK" | cut -d' ' -f1)" = "$CK_SHA"
WANT_4E='attention kernel from position 0 on grow(256), decode persistent'
WANT_K='attention kernel from position 1280 on grow(256))'
served_unit() { grep -qF "sha256 $SERVED_SHA" "$1" || { echo "gate: $1 did not compile the served unit $SERVED_SHA" >&2; exit 1; }; }
four_e_loaded() {  # the 4E install lines: RoPE on the card, the 512-thread symbol with no local memory
  grep -q '^4E: RoPE forms equal to candle' "$1"
  grep -qE '^4E: pers_4e_tokens_f[0-9]_t512: [0-9]+ registers, 0 B local' "$1"
}

echo '== PART 1: the door under the 4E config, 256 tokens =='
date -u
LLVQ_CONFIG="$P4" timeout 8m fusedrun "$S" 256 > "$O/door-4e.txt" 2>&1
date -u
grep -E '^served config|^4E:|^served door|^served :|^  (4E|eager)' "$O/door-4e.txt"
grep -E '^served config: ' "$O/door-4e.txt" | grep -qF "$WANT_4E"
served_unit "$O/door-4e.txt"; four_e_loaded "$O/door-4e.txt"
grep -q '^served door (decode persistent): 4E tokens equal the eager path.s under this config, 256 of 256' "$O/door-4e.txt"
echo 'gate 3: the 4E door decoded the eager path tokens under its config'

echo "== PART 2: the sweep under the 4E config, LLVQ_CONTEXTS=$CONTEXTS =="
date -u
LLVQ_CONFIG="$P4" LLVQ_CONTEXTS="$CONTEXTS" timeout 14m fusedrun "$S" 256 > "$O/sweep-4e.txt" 2>&1
date -u
grep -v '^  tokens after ' "$O/sweep-4e.txt" | tail -12
grep -E '^served config: ' "$O/sweep-4e.txt" | grep -qF "$WANT_4E"
four_e_loaded "$O/sweep-4e.txt"
grep -q '^decode engine: persistent (4E)' "$O/sweep-4e.txt"
test "$(grep -cE '^  prompt [0-9]+: 4E tokens equal the eager path.s under this config, 256 of 256' "$O/sweep-4e.txt")" = 4
test "$(grep -cE '^ +[0-9]+ +[0-9.]+ \[' "$O/sweep-4e.txt")" = 4
echo 'gate 4: the 4E sweep decoded the eager path tokens at every length'

echo '== PART 3: the door and the sweep under the served config =='
date -u
LLVQ_CONFIG="$PK" timeout 6m fusedrun "$S" 256 > "$O/door-served.txt" 2>&1
LLVQ_CONFIG="$PK" LLVQ_CONTEXTS="$CONTEXTS" timeout 12m fusedrun "$S" 256 > "$O/sweep-served.txt" 2>&1
date -u
grep -E '^served config: ' "$O/door-served.txt" | grep -qF "$WANT_K"
grep -E '^served config: ' "$O/sweep-served.txt" | grep -qF "$WANT_K"
served_unit "$O/door-served.txt"
python3 -c "$CMP_PY" door "$O/door-4e.txt" "$O/door-served.txt" | tee "$O/compare.txt"
python3 -c "$CMP_PY" sweep "$O/sweep-4e.txt" "$O/sweep-served.txt" "$CONTEXTS" | tee -a "$O/compare.txt"

echo '== PART 4: the pilot 50 GSM8K problems through the 4E config =='
date -u
LLVQ_DATASET_REV="$DATASET_REV" LLVQ_CONFIG="$P4" LLVQ_GSM8K_DUMP="$O/gsm8k-4b-4e-n50.jsonl" \
  timeout 8m gsm8k "$S" cuda 50 > "$O/gsm8k-4b-4e-n50.txt" 2>&1
date -u
grep -vE '^  \[ *[0-9]+/' "$O/gsm8k-4b-4e-n50.txt" | tail -8
tail -1 "$O/gsm8k-4b-4e-n50.jsonl" | grep -q '"end":true'
tail -1 "$O/gsm8k-4b-4e-n50.jsonl" | grep -qF "\"fingerprint\":\"$REPLAY_FP\""
grep -E '^served config: ' "$O/gsm8k-4b-4e-n50.txt" | grep -qF "$WANT_4E"
four_e_loaded "$O/gsm8k-4b-4e-n50.txt"
python3 -c "$CMP_PY" gsm8k "$O/gsm8k-4b-4e-n50.jsonl" "$PILOT_REF" | tee -a "$O/compare.txt"
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
  --name "served-4e${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
