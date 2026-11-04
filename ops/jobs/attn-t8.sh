#!/usr/bin/env bash
# The T8 control of the split-KV attention kernel on the sealed 4B, against the served grouped form: paired
# decode-path perplexity at 4,096 and 8,192, then GSM8K through the served door under the kernel config, paired
# with the grouped census of 2026-11-04. Preregistered: proofs/preregistration-attn-t8-2026-11-04.md, stamped
# before this.
#
#   DRY_RUN=1 bash ops/jobs/attn-t8.sh                          prints and parses the job script
#   IMAGE_SHA=<sha> EMIT=<file> bash ops/jobs/attn-t8.sh         writes the job script for ops/jobs/lot1.sh
#   IMAGE_SHA=<space sha> bash ops/jobs/attn-t8.sh               launches it alone, after the operator's go
#
# CONDITIONAL. The job first reads the two sweep logs of the attention bench r2 on the mount ($R2, written by
# ops/jobs/attn-bench-r2.sh earlier in the same lot job, or by its own job) and runs nothing more unless the
# lead's stop line stays silent (prereg section 3). It then gates each step on the one before: the GSM8K steps
# run only when both perplexity intervals lie inside the band of +-0.12 %, as the lead orders V7 after V6.
#
# Every LLVQ_* variable is set on its own command line, never exported. Each step has its own ceiling
# (`timeout`), so a hang voids that step and stops the job instead of eating the lot's time.
#
# Cost, estimated (l40sx1 $1.80/h): 1 min if the condition fails; 46 min, $1.38, if the band fails; 109 min,
# $3.27, in full. Ceiling 160m alone, $4.80.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

OBJ_DIR=sealed-4b-2026-09-23
NAME=qwen3-4b-sealed.bin
LOCAL=$HOME/q4b-sealed-2026-09-23/$NAME
TIMEOUT=160m
PREREG=proofs/preregistration-attn-t8-2026-11-04.md
BUCKET=Pier-Jean/jobs-artifacts
DATASET_REV=740312add88f781978c0658806c59bc2815b9866
CONF_GRP_LOCAL=configs/qwen3-4b-tetra-e4-grouped.json
# The kernel config of both attention benches, byte for byte (`attn-bench-2026-11-04-brut/`).
CONF_KER_WANT=d74ea29abee6b585348f56d849aa09f1f27db1e09eab4bbd23bf2fe38aeef3fa
FLAGS='LLVQ_FUSED_LAYOUT=tetra48 LLVQ_ROT_SHARE=1 LLVQ_FUSE=0 LLVQ_KV=f16 LLVQ_EMBED=q4'
# The grouped census of 2026-11-04 (job 6aeb1041, image f77f803e), the B arm of the GSM8K pair.
GRP_CENSUS=docs/mesures/gsm8k-grouped-2026-11-04-brut/gsm8k-4b-grouped.jsonl
GRP_CENSUS_SHA=eda98402a34de849e88800106acb189d6b5d5667bea39db5b3170a1eddca3969
# The pilot's 50 problems (gsm8k::select, seed 0x65736d386b), whose indices run 0 of 2026-09-26 holds.
PILOT=docs/mesures/gsm8k-wave1-2026-09-26-brut/smoke/gsm8k-4b-served.jsonl
# Digest of the grouped census rows at the pilot's indices, by ROWS_PY (gate 5).
REPLAY_ROWS_SHA=f11a0b2386062f6a1b306784d6ddd7247e0da635fc0d77e921a759610cde2f01
REPLAY_FP=94c7b98876288901
CENSUS_FP=bfa9135ce4b55c75
# RETRY=2 names a relaunch; R2_RETRY=2 points at a relaunched r2.
O=/out/attn-t8-4b-2026-11-04${RETRY:+-r$RETRY}
R2=/out/attn-bench-r2-2026-11-04${R2_RETRY:+-r$R2_RETRY}
F=/out/$OBJ_DIR/$NAME
CG=$O/qwen3-4b-tetra-e4-grouped.json
CK=$O/qwen3-4b-tetra-e4-kernel.json
DRY=${DRY_RUN:-0}
SRC_SHA=$(shasum -a 256 < llvq-cuda/kernels/attn_dec.cu | cut -d' ' -f1)

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

# The condition (prereg section 3), read in the job from r2's two sweep logs. It prints one verdict line.
COND_PY='import os,re,sys
d=sys.argv[1]
def sweep(arm,want):
    p=os.path.join(d,"sweep-%s.txt"%arm)
    t=open(p,encoding="utf-8").read().splitlines()
    sc=[l for l in t if l.startswith("served config: ")]
    if len(sc)!=1 or want not in sc[0]: raise ValueError("%s: no served config line naming %s"%(p,want))
    r={}
    for l in t:
        m=re.match(r"^ +(\d+) +[\d.]+ \[[^\]]*\] +([\d.]+) \[([\d.]+)\D+([\d.]+)\] +[\d.]+$",l)
        if m: r[int(m.group(1))]=tuple(float(m.group(i)) for i in (2,3,4))
    if sorted(r)!=[5,256,2048,4096,8192]: raise ValueError("%s: lengths %s"%(p,sorted(r)))
    return t,r
try:
    tk,k=sweep("kernel","attention kernel on grow(256)")
    tg,g=sweep("grouped","attention grouped on grow(256)")
    if not any("the card equals the CPU replay bit for bit" in l for l in tk):
        raise ValueError("the kernel arm printed no load-time identity line")
except (OSError,ValueError) as e:
    print("r2 unreadable: %s"%e); print("T8 condition: not met (r2 void or absent)"); sys.exit(0)
print("  prompt   kernel tok/s [range]     grouped tok/s [range]    medians k/g   in situ [min k/max g, max k/min g]")
for L in (5,256,2048,4096,8192):
    (km,kl,kh),(gm,gl,gh)=k[L],g[L]
    print("  %6d   %6.1f [%.1f-%.1f]   %6.1f [%.1f-%.1f]   %.4f        [%.4f, %.4f]"%(L,km,kl,kh,gm,gl,gh,km/gm,kl/gh,kh/gl))
r5=k[5][0]/g[5][0]; u8=k[8192][2]/g[8192][1]
c5=r5>=0.97; c8=u8>=1.0
print("stop line, 5 tokens: medians kernel/grouped %.4f, %s 0.97"%(r5,">=" if c5 else "<"))
print("stop line, 8,192: upper end of the in-situ range %.4f, %s 1.00"%(u8,">=" if c8 else "<"))
print("T8 condition: met" if c5 and c8 else "T8 condition: not met (the lead stop line)")'

# The band (prereg section 7, K1), read from the two perplexity logs.
BAND_PY='import re,sys
inside=True
for p in sys.argv[1:]:
    ls=[l for l in open(p,encoding="utf-8").read().splitlines() if re.match(r"^  decode +[\d.]+ ",l)]
    m=re.search(r"([+-][\d.]+) % \[([+-][\d.]+) ; ([+-][\d.]+)\]",ls[0]) if len(ls)==1 else None
    if not m: print("%s: no single decode summary line"%p); sys.exit(1)
    mid,lo,hi=(float(x) for x in m.groups())
    ok=lo>=-0.12 and hi<=0.12
    inside=inside and ok
    print("%s: kernel / grouped - 1 = %+.4f %% [%+.4f ; %+.4f], %s the band"%(p,mid,lo,hi,"inside" if ok else "NOT inside"))
print("band: both inside" if inside else "band: not both inside")'

CONF_GRP=$(cat "$CONF_GRP_LOCAL")
CONF_GRP_SHA=$(printf '%s\n' "$CONF_GRP" | shasum -a 256 | cut -d' ' -f1)
[ "$CONF_GRP_SHA" = "$(shasum -a 256 < "$CONF_GRP_LOCAL" | cut -d' ' -f1)" ] \
  || { echo "refused: $CONF_GRP_LOCAL does not end in exactly one newline" >&2; exit 1; }
# The kernel config, built as ops/jobs/attn-bench-r2.sh builds it, and held to the bytes both benches ran.
CONF_KER=$(python3 -c 'import json,sys
g=json.load(open(sys.argv[1]))
assert (g["layout"],g["embed"],g["rot_share"],g["fuse"],g["kv"],g["attention"])==("tetra48","q4","1","0","f16","grouped"), g
g["attention"]="kernel"
g["note"]="MEASUREMENT ONLY, written by ops/jobs/attn-bench.sh: the qwen3-4b-tetra-e4 object with decode rows sent to the split-KV attention kernel. Not licensed: the kernel is not bit for bit with the grouped form, and its T8 decision is the operator'"'"'s."
print(json.dumps(g, indent=2))' "$CONF_GRP_LOCAL")
CONF_KER_SHA=$(printf '%s\n' "$CONF_KER" | shasum -a 256 | cut -d' ' -f1)
[ "$CONF_KER_SHA" = "$CONF_KER_WANT" ] \
  || { echo "refused: the kernel config is $CONF_KER_SHA, the benches ran $CONF_KER_WANT" >&2; exit 1; }
[ "$(shasum -a 256 < "$GRP_CENSUS" | cut -d' ' -f1)" = "$GRP_CENSUS_SHA" ] \
  || { echo "refused: $GRP_CENSUS is not $GRP_CENSUS_SHA" >&2; exit 1; }
tail -1 "$GRP_CENSUS" | grep -qF "\"fingerprint\":\"$CENSUS_FP\"" \
  || { echo "refused: $GRP_CENSUS does not end on fingerprint $CENSUS_FP" >&2; exit 1; }
[ "$(python3 -c "$ROWS_PY" "$GRP_CENSUS" "$PILOT")" = "50 $REPLAY_ROWS_SHA" ] \
  || { echo "refused: the grouped census rows at the pilot's indices do not digest to $REPLAY_ROWS_SHA" >&2; exit 1; }

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
  IMAGE_SHA=${IMAGE_SHA:?IMAGE_SHA=<the Space sha the lot runs on>}
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    'from huggingface_hub import HfApi; print(HfApi().space_info("Pier-Jean/llvq-runner-cuda").sha)')
  [ "$NOW" = "$IMAGE_SHA" ] || { echo "refused: the Space is at $NOW, expected $IMAGE_SHA" >&2; exit 1; }
  echo "attn_dec.cu sha256 $SRC_SHA (file); 4b sealed: $BYTES B, sha256 $SHA; configs $CONF_KER_SHA (kernel) / $CONF_GRP_SHA (grouped); image $NOW"
fi

PRE=$(printf 'O=%q\nR2=%q\nF=%q\nCG=%q\nCK=%q\nSHA=%q\nBYTES=%q\nCONF_GRP=%q\nCONF_KER=%q\nCONF_GRP_SHA=%q\nCONF_KER_SHA=%q\nSRC_SHA=%q\nFLAGS=%q\nDATASET_REV=%q\nREPLAY_ROWS_SHA=%q\nREPLAY_FP=%q\nCENSUS_FP=%q\nROWS_PY=%q\nCOND_PY=%q\nBAND_PY=%q\nexport HF_HOME=/tmp/hf' \
  "$O" "$R2" "$F" "$CG" "$CK" "$SHA" "$BYTES" "$CONF_GRP" "$CONF_KER" "$CONF_GRP_SHA" "$CONF_KER_SHA" "$SRC_SHA" \
  "$FLAGS" "$DATASET_REV" "$REPLAY_ROWS_SHA" "$REPLAY_FP" "$CENSUS_FP" "$ROWS_PY" "$COND_PY" "$BAND_PY")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
echo '== STEP 0: the condition, read from the r2 sweep logs =='
python3 -c "$COND_PY" "$R2" | tee "$O/condition.txt"
if ! grep -qx 'T8 condition: met' "$O/condition.txt"; then
  echo 'T8 control NOT RUN: the condition of its prereg section 3 does not hold (outcome C0, not a failure)'
  exit 0
fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader | tee "$O/gpu.txt"
echo '== gate 1: oracle (hard rule 10) =='
oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle-cuda.txt" | tail -2
grep -q MATCH "$O/oracle-cuda.txt"
echo '== gate 2: the object and both configs on the mount =='
test "$(stat -c %s "$F")" = "$BYTES"
sha256sum "$F" | tee "$O/files.sha256"
test "$(cut -d' ' -f1 "$O/files.sha256")" = "$SHA"
printf '%s\n' "$CONF_KER" > "$CK"
printf '%s\n' "$CONF_GRP" > "$CG"
sha256sum "$CK" "$CG" | tee -a "$O/files.sha256"
test "$(sha256sum "$CK" | cut -d' ' -f1)" = "$CONF_KER_SHA"
test "$(sha256sum "$CG" | cut -d' ' -f1)" = "$CONF_GRP_SHA"
TILE='tile 64 (served: measured optimum for sm_89)'
PARITY='int4: tv_q4_hh equals tv_q4_h bit for bit'
kernel_loaded() {  # log: the kernel text, built and checked against the CPU replay on the card
  grep -qF "attn_dec.cu: embedded, sha256 $SRC_SHA" "$1"
  grep -qE '^attn_dec_g4_u64: the card equals the CPU replay bit for bit on [0-9]+ calls' "$1"
}
for spec in 4096:12:60:30m 8192:6:6:35m; do
  IFS=: read -r C W FIRST CEIL <<< "$spec"
  P=$O/ppl-$C.txt
  echo "== STEP: decode-path perplexity, $W windows of $C from window index $FIRST, kernel against grouped =="
  date -u
  env $FLAGS LLVQ_GQA_PPL="$C,$W,$FIRST,kernel" timeout "$CEIL" fusedrun "$F" 2>&1 | tee "$P" | tail -8
  date -u
  grep -qF "$TILE" "$P"
  grep -qF "$PARITY" "$P"
  grep -qF 'cudarc event tracking: off (one stream, ordered by issue)' "$P"
  kernel_loaded "$P"
  grep -E '^GQA perplexity: ' "$P" | grep -qF "wikitext2, $W windows of $C from window $FIRST (tokens "
  grep -E '^GQA perplexity: ' "$P" | grep -qF 'against grow(256)+grouped, decode path in lockstep'
  test "$(grep -cE "^  window +[0-9]+/$W  decode nll " "$P")" = "$W"
  if [ "$C" = 4096 ]; then
    grep -E '^GQA perplexity: ' "$P" | grep -qF 'prefill path on 1 window(s), checked bit for bit'
    test "$(grep -cxF '  window   1: prefill path bit for bit between the two arms' "$P")" = 1
  else
    grep -E '^GQA perplexity: ' "$P" | grep -qF 'prefill path on 0 window(s)'
  fi
  grep -qF "kernel / grouped − 1, paired 95 % t interval over $W windows" "$P"
  grep -qE '^  decode top-1 agreement [0-9]+/[0-9]+ steps' "$P"
done
echo 'gates 3 and 4: both perplexity runs printed every line'
python3 -c "$BAND_PY" "$O/ppl-4096.txt" "$O/ppl-8192.txt" | tee "$O/band.txt"
if ! grep -qx 'band: both inside' "$O/band.txt"; then
  echo 'GSM8K NOT RUN: a perplexity interval leaves the band (outcome K2, read in the prereg)'
  echo '== raw kept ==' ; ls -la "$O"
  exit 0
fi
provenance() {  # log, wanted attention label
  grep -E '^served config: ' "$1" | grep -qF "$2"
  grep -qF "$TILE" "$1"
  grep -qF "$PARITY" "$1"
}
echo '== STEP: the grouped replay, the pilot'"'"'s 50 problems on this image =='
date -u
LLVQ_DATASET_REV="$DATASET_REV" LLVQ_CONFIG="$CG" LLVQ_GSM8K_DUMP="$O/gsm8k-4b-grouped-n50.jsonl" \
  timeout 10m gsm8k "$F" cuda 50 2>&1 | tee "$O/gsm8k-4b-grouped-n50.txt" | grep -vE '^  \[ *[0-9]+/' | tail -12
date -u
provenance "$O/gsm8k-4b-grouped-n50.txt" 'attention grouped on grow(256)'
tail -1 "$O/gsm8k-4b-grouped-n50.jsonl" | grep -q '"end":true'
tail -1 "$O/gsm8k-4b-grouped-n50.jsonl" | grep -qF "\"fingerprint\":\"$REPLAY_FP\""
GOT=$(python3 -c "$ROWS_PY" "$O/gsm8k-4b-grouped-n50.jsonl")
echo "replay rows: $GOT; the grouped census at the same indices: 50 $REPLAY_ROWS_SHA"
if [ "$GOT" != "50 $REPLAY_ROWS_SHA" ]; then
  echo 'GATE 5 FAILED: the grouped replay differs from the grouped census; the kernel census is not run' >&2
  exit 3
fi
echo 'gate 5: this image writes the grouped census rows'
echo '== STEP: GSM8K under the kernel config, 1,319 problems =='
date -u
LLVQ_DATASET_REV="$DATASET_REV" LLVQ_CONFIG="$CK" LLVQ_GSM8K_DUMP="$O/gsm8k-4b-kernel.jsonl" \
  timeout 80m gsm8k "$F" cuda 2>&1 | tee "$O/gsm8k-4b-kernel.txt" | grep -vE '^  \[ *[0-9]+/' | tail -16
date -u
provenance "$O/gsm8k-4b-kernel.txt" 'attention kernel on grow(256)'
kernel_loaded "$O/gsm8k-4b-kernel.txt"
tail -1 "$O/gsm8k-4b-kernel.jsonl" | grep -q '"end":true'
tail -1 "$O/gsm8k-4b-kernel.jsonl" | grep -qF "\"fingerprint\":\"$CENSUS_FP\""
echo 'gate 6: the kernel census ends on the census fingerprint'
echo '== raw kept ==' ; ls -la "$O"
JOB

# EMIT=<file>: write the job script to <file> and launch nothing (ops/jobs/lot1.sh).
if [ -n "${EMIT:-}" ]; then
  printf 'set -euo pipefail\n%s\n%s\n' "$PRE" "$BODY" > "$EMIT"
  bash -n "$EMIT" && echo "EMIT: job script written to $EMIT; nothing launched" >&2
  exit 0
fi

if [ "$DRY" = 1 ]; then
  JOBSCRIPT="set -euo pipefail
$PRE
$BODY"
  printf '%s\n' "$JOBSCRIPT" | bash -n && echo "DRY_RUN: job script parses; nothing launched" >&2
  exit 0
fi

# Alone, this job reads r2's logs from the bucket: refuse before billing if they are not there.
for f in sweep-kernel.txt sweep-grouped.txt; do
  hf buckets ls "hf://buckets/$BUCKET/${R2#/out/}/" 2>/dev/null | grep -qF "$f" \
    || { echo "refused: ${R2#/out/}/$f is not in the bucket" >&2; exit 1; }
done
uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor l40sx1 --timeout "$TIMEOUT" \
  --bucket "$BUCKET" --out-mount /out \
  --name "attn-t8-4b${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
