#!/usr/bin/env bash
# The T8 control of the attention kernel form WITH its threshold (grouped below position 1,280, the kernel from
# it, llvq_llm::kvq::KERNEL_FROM), on the sealed 4B, against the served grouped form. One job, three parts:
# (c) the pilot's 50 GSM8K problems under the kernel config, which must write the grouped census rows; (a) the
# served sweep, kernel config then grouped config, with the tokens at 5, 256 and 1,024 held identical; (b) the
# paired decode-path perplexity at 4,096 then 8,192, the band read on the positions the kernel takes.
# Preregistered: proofs/preregistration-attn-t8-r2-2026-11-04.md, stamped before this. It supersedes
# proofs/preregistration-attn-t8-2026-11-04.md, which ran no step (lot 1, outcome C0).
#
#   DRY_RUN=1 bash ops/jobs/attn-t8-r2.sh                        prints and parses the job script
#   IMAGE_SHA=<sha> EMIT=<file> bash ops/jobs/attn-t8-r2.sh       writes the job script, launches nothing
#   IMAGE_SHA=<space sha> bash ops/jobs/attn-t8-r2.sh             launches it, after the operator's go
#
# The parts run in that order, cheapest gate first, and the job stops as soon as the outcome is settled:
# a replay that differs (N1), or a 4,096 interval outside the band (N2), ends it before the rest is billed.
# Every LLVQ_* variable is set on its own command line, never exported. Each step has its own ceiling.
#
# Cost, estimated (l40sx1 $1.80/h, prereg section 8): 54 min, $1.62, in full. Ceiling 80m, $2.40.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

OBJ_DIR=sealed-4b-2026-09-23
NAME=qwen3-4b-sealed.bin
LOCAL=$HOME/q4b-sealed-2026-09-23/$NAME
TIMEOUT=80m
PREREG=proofs/preregistration-attn-t8-r2-2026-11-04.md
BUCKET=Pier-Jean/jobs-artifacts
DATASET_REV=740312add88f781978c0658806c59bc2815b9866
CONTEXTS=5,256,1024,1280,2048,4096,8192
# The kernel threshold the image must carry, and the lengths whose whole decode lies below it (gate 6).
FROM=1280
SAME_TOKENS=5,256,1024
CONF_GRP_LOCAL=configs/qwen3-4b-tetra-e4-grouped.json
# The kernel config of this run: the grouped one with the attention field and the note changed.
CONF_KER_WANT=e01326fc54039505831715a44c98e686c8105f1a597d1e520a995944581b913b
FLAGS='LLVQ_FUSED_LAYOUT=tetra48 LLVQ_ROT_SHARE=1 LLVQ_FUSE=0 LLVQ_KV=f16 LLVQ_EMBED=q4'
# The grouped census of 2026-11-04 (job 6aeb1041, image f77f803e), and the pilot's 50 problems in it.
GRP_CENSUS=docs/mesures/gsm8k-grouped-2026-11-04-brut/gsm8k-4b-grouped.jsonl
GRP_CENSUS_SHA=eda98402a34de849e88800106acb189d6b5d5667bea39db5b3170a1eddca3969
PILOT=docs/mesures/gsm8k-wave1-2026-09-26-brut/smoke/gsm8k-4b-served.jsonl
REPLAY_ROWS_SHA=f11a0b2386062f6a1b306784d6ddd7247e0da635fc0d77e921a759610cde2f01
REPLAY_FP=94c7b98876288901
CENSUS_FP=bfa9135ce4b55c75
# RETRY=2 names a relaunch after a failed attempt left its directory behind.
O=/out/attn-t8-r2-4b-2026-11-04${RETRY:+-r$RETRY}
F=/out/$OBJ_DIR/$NAME
CG=$O/qwen3-4b-tetra-e4-grouped.json
CK=$O/qwen3-4b-tetra-e4-kernel-from1280.json
DRY=${DRY_RUN:-0}
SRC_SHA=$(shasum -a 256 < llvq-cuda/kernels/attn_dec.cu | cut -d' ' -f1)

# The row digest of ops/jobs/attn-t8.sh, unchanged: header and trailer dropped, `prefill_s` and `decode_s`
# dropped from each row, rows written with sorted keys, joined by newlines, sha256 of the UTF-8 bytes. A second
# argument keeps only the rows whose index that dump also holds.
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

# The sweep (part a): the two logs' tables and token lines. Gate 6 is the token identity at SAME_TOKENS;
# the rest is read and printed, never gated.
SWEEP_PY='import re,sys
ker,grp,lengths,same=sys.argv[1],sys.argv[2],[int(x) for x in sys.argv[3].split(",")],[int(x) for x in sys.argv[4].split(",")]
def read(p):
    t=open(p,encoding="utf-8").read().splitlines()
    rate,tok={},{}
    for l in t:
        m=re.match(r"^ +(\d+) +[\d.]+ \[[^\]]*\] +([\d.]+) \[([\d.]+)\D+([\d.]+)\] +[\d.]+$",l)
        if m: rate[int(m.group(1))]=tuple(float(m.group(i)) for i in (2,3,4))
        m=re.match(r"^  tokens after (\d+): ([0-9a-f]{16}) ([\d,]+)$",l)
        if m: tok[int(m.group(1))]=(m.group(2),[int(x) for x in m.group(3).split(",")])
    if sorted(rate)!=lengths or sorted(tok)!=lengths:
        print("%s: lengths %s / tokens %s, expected %s"%(p,sorted(rate),sorted(tok),lengths)); sys.exit(1)
    return rate,tok
(kr,kt),(gr,gt)=read(ker),read(grp)
print("  prompt   kernel tok/s [range]     grouped tok/s [range]    medians k/g   in situ [min k/max g, max k/min g]   tokens")
ok=True
for L in lengths:
    (km,kl,kh),(gm,gl,gh)=kr[L],gr[L]
    a,b=kt[L][1],gt[L][1]
    if a==b: tk="identical (%d)"%len(a)
    else:
        i=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
        tk="first divergence at token %d of %d"%(i,len(a))
    if L in same and kt[L]!=gt[L]: ok=False
    print("  %6d   %6.1f [%.1f-%.1f]   %6.1f [%.1f-%.1f]   %.4f        [%.4f, %.4f]          %s"%(L,km,kl,kh,gm,gl,gh,km/gm,kl/gh,kh/gl,tk))
print("gate 6: the tokens at %s are identical between the two configs"%",".join(map(str,same)) if ok else "GATE 6 FAILED: the tokens differ at a length whose decode lies wholly below the threshold")
sys.exit(0 if ok else 3)'

# The band (part b), read on the `from <FROM>` summary line of one perplexity log.
BAND_PY='import re,sys
p,frm=sys.argv[1],sys.argv[2]
ls=[l for l in open(p,encoding="utf-8").read().splitlines() if re.match(r"^  from %s +[\d.]+ "%frm,l)]
m=re.search(r"([+-][\d.]+) % \[([+-][\d.]+) ; ([+-][\d.]+)\]",ls[0]) if len(ls)==1 else None
if not m: print("%s: no single from-%s summary line"%(p,frm)); sys.exit(1)
mid,lo,hi=(float(x) for x in m.groups())
ok=lo>=-0.12 and hi<=0.12
print("%s: from position %s, kernel / grouped - 1 = %+.4f %% [%+.4f ; %+.4f], %s the band"%(p,frm,mid,lo,hi,"inside" if ok else "NOT inside"))
print("band: inside" if ok else "band: not inside")'

CONF_GRP=$(cat "$CONF_GRP_LOCAL")
CONF_GRP_SHA=$(printf '%s\n' "$CONF_GRP" | shasum -a 256 | cut -d' ' -f1)
[ "$CONF_GRP_SHA" = "$(shasum -a 256 < "$CONF_GRP_LOCAL" | cut -d' ' -f1)" ] \
  || { echo "refused: $CONF_GRP_LOCAL does not end in exactly one newline" >&2; exit 1; }
CONF_KER=$(python3 -c 'import json,sys
g=json.load(open(sys.argv[1]))
assert (g["layout"],g["embed"],g["rot_share"],g["fuse"],g["kv"],g["attention"])==("tetra48","q4","1","0","f16","grouped"), g
g["attention"]="kernel"
g["note"]="MEASUREMENT ONLY, written by ops/jobs/attn-t8-r2.sh: the qwen3-4b-tetra-e4 object with decode rows at position 1,280 and later sent to the split-KV attention kernel, the rows below on the grouped path. Not licensed: past 1,280 the kernel is not bit for bit with the grouped form, and its T8 decision is the operator'"'"'s."
print(json.dumps(g, indent=2))' "$CONF_GRP_LOCAL")
CONF_KER_SHA=$(printf '%s\n' "$CONF_KER" | shasum -a 256 | cut -d' ' -f1)
[ "$CONF_KER_SHA" = "$CONF_KER_WANT" ] \
  || { echo "refused: the kernel config is $CONF_KER_SHA, the prereg names $CONF_KER_WANT" >&2; exit 1; }
# The threshold the code carries is the one this run reads.
grep -qx "pub const KERNEL_FROM: usize = $FROM;" llvq-llm/src/kvq.rs \
  || { echo "refused: llvq-llm/src/kvq.rs does not set KERNEL_FROM to $FROM" >&2; exit 1; }
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
  IMAGE_SHA=${IMAGE_SHA:?IMAGE_SHA=<the Space sha built from the commit that carries the prereg>}
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    'from huggingface_hub import HfApi; print(HfApi().space_info("Pier-Jean/llvq-runner-cuda").sha)')
  [ "$NOW" = "$IMAGE_SHA" ] || { echo "refused: the Space is at $NOW, expected $IMAGE_SHA" >&2; exit 1; }
  echo "attn_dec.cu sha256 $SRC_SHA (file); 4b sealed: $BYTES B, sha256 $SHA; configs $CONF_KER_SHA (kernel from $FROM) / $CONF_GRP_SHA (grouped); image $NOW"
fi

PRE=$(printf 'O=%q\nF=%q\nCG=%q\nCK=%q\nSHA=%q\nBYTES=%q\nCONF_GRP=%q\nCONF_KER=%q\nCONF_GRP_SHA=%q\nCONF_KER_SHA=%q\nSRC_SHA=%q\nFLAGS=%q\nDATASET_REV=%q\nCONTEXTS=%q\nFROM=%q\nSAME_TOKENS=%q\nREPLAY_ROWS_SHA=%q\nREPLAY_FP=%q\nROWS_PY=%q\nSWEEP_PY=%q\nBAND_PY=%q\nexport HF_HOME=/tmp/hf' \
  "$O" "$F" "$CG" "$CK" "$SHA" "$BYTES" "$CONF_GRP" "$CONF_KER" "$CONF_GRP_SHA" "$CONF_KER_SHA" "$SRC_SHA" \
  "$FLAGS" "$DATASET_REV" "$CONTEXTS" "$FROM" "$SAME_TOKENS" "$REPLAY_ROWS_SHA" "$REPLAY_FP" \
  "$ROWS_PY" "$SWEEP_PY" "$BAND_PY")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
outcome() { echo "OUTCOME $1" | tee "$O/outcome.txt"; echo '== raw kept =='; ls -la "$O"; }
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader | tee "$O/gpu.txt"
echo '== gate 1: oracle (hard rule 10) =='
timeout 5m oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle-cuda.txt" | tail -2
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
WANT_K="attention kernel from position $FROM on grow(256)"
WANT_G='attention grouped on grow(256)'
kernel_loaded() {  # log: the kernel text, built and checked against the CPU replay on the card
  grep -qF "attn_dec.cu: embedded, sha256 $SRC_SHA" "$1"
  grep -qE '^attn_dec_g4_u64: the card equals the CPU replay bit for bit on [0-9]+ calls' "$1"
}
provenance() {  # log, wanted attention words
  grep -E '^served config: ' "$1" | grep -qF "$2"
  grep -qF "$TILE" "$1"
  grep -qF "$PARITY" "$1"
}
replay() {  # name, config: the pilot's 50 problems through the served door
  date -u
  LLVQ_DATASET_REV="$DATASET_REV" LLVQ_CONFIG="$2" LLVQ_GSM8K_DUMP="$O/gsm8k-4b-$1-n50.jsonl" \
    timeout 8m gsm8k "$F" cuda 50 2>&1 | tee "$O/gsm8k-4b-$1-n50.txt" | grep -vE '^  \[ *[0-9]+/' | tail -12
  date -u
  tail -1 "$O/gsm8k-4b-$1-n50.jsonl" | grep -q '"end":true'
  tail -1 "$O/gsm8k-4b-$1-n50.jsonl" | grep -qF "\"fingerprint\":\"$REPLAY_FP\""
}

echo '== PART c: the pilot'"'"'s 50 GSM8K problems under the kernel config =='
replay kernel "$CK"
provenance "$O/gsm8k-4b-kernel-n50.txt" "$WANT_K"
kernel_loaded "$O/gsm8k-4b-kernel-n50.txt"
echo 'gates 3 and 4: the replay ran the kernel config, ended on the pilot fingerprint'
GOT=$(python3 -c "$ROWS_PY" "$O/gsm8k-4b-kernel-n50.jsonl")
echo "kernel replay rows: $GOT; the grouped census at the same indices: 50 $REPLAY_ROWS_SHA" | tee "$O/replay.txt"
if [ "$GOT" != "50 $REPLAY_ROWS_SHA" ]; then
  echo 'the kernel replay differs from the grouped census: the grouped replay runs to attribute it (diagnostic)'
  replay grouped "$CG"
  provenance "$O/gsm8k-4b-grouped-n50.txt" "$WANT_G"
  echo "grouped replay rows: $(python3 -c "$ROWS_PY" "$O/gsm8k-4b-grouped-n50.jsonl")" | tee -a "$O/replay.txt"
  outcome 'N1: the kernel replay differs from the grouped census; not proposed'
  exit 0
fi

echo "== PART a: LLVQ_CONTEXTS=$CONTEXTS beside each config, 256 new tokens, kernel config first =="
for arm in kernel grouped; do
  case $arm in
    kernel)  C=$CK; WANT=$WANT_K ;;
    grouped) C=$CG; WANT=$WANT_G ;;
  esac
  date -u
  env LLVQ_CONFIG="$C" LLVQ_CONTEXTS="$CONTEXTS" timeout 13m fusedrun "$F" 256 2>&1 \
    | tee "$O/sweep-$arm.txt" | grep -v '^  tokens after ' | tail -12
  date -u
  provenance "$O/sweep-$arm.txt" "$WANT"
  test "$(grep -cE '^ +[0-9]+ +[0-9.]+ \[' "$O/sweep-$arm.txt")" = 7
  test "$(grep -cE '^  tokens after [0-9]+: [0-9a-f]{16} ' "$O/sweep-$arm.txt")" = 7
  if [ "$arm" = kernel ]; then kernel_loaded "$O/sweep-$arm.txt"; fi
done
echo 'gate 5: both sweeps printed their config, tile, parity, seven lengths and seven token lines'
python3 -c "$SWEEP_PY" "$O/sweep-kernel.txt" "$O/sweep-grouped.txt" "$CONTEXTS" "$SAME_TOKENS" | tee "$O/sweep.txt"

echo '== PART b: decode-path perplexity, kernel against grouped =='
for spec in 4096:12:60:24m 8192:6:6:26m; do
  IFS=: read -r C W FIRST CEIL <<< "$spec"
  P=$O/ppl-$C.txt
  echo "== $W windows of $C from window index $FIRST =="
  date -u
  env $FLAGS LLVQ_GQA_PPL="$C,$W,$FIRST,kernel" timeout "$CEIL" fusedrun "$F" 2>&1 | tee "$P" | tail -9
  date -u
  grep -qF "$TILE" "$P"
  grep -qF "$PARITY" "$P"
  grep -qF 'cudarc event tracking: off (one stream, ordered by issue)' "$P"
  kernel_loaded "$P"
  grep -E '^GQA perplexity: ' "$P" | grep -qF "wikitext2, $W windows of $C from window $FIRST (tokens "
  grep -E '^GQA perplexity: ' "$P" | grep -qF 'against grow(256)+grouped, decode path in lockstep'
  grep -qF "  the kernel form takes positions $FROM and later: positions 0..$FROM checked bit for bit" "$P"
  test "$(grep -cE "^  window +[0-9]+/$W  positions < $FROM: bit for bit  decode nll " "$P")" = "$W"
  if [ "$C" = 4096 ]; then
    grep -E '^GQA perplexity: ' "$P" | grep -qF 'prefill path on 1 window(s), checked bit for bit'
    test "$(grep -cxF '  window   1: prefill path bit for bit between the two arms' "$P")" = 1
  else
    grep -E '^GQA perplexity: ' "$P" | grep -qF 'prefill path on 0 window(s)'
  fi
  grep -qF "kernel / grouped − 1, paired 95 % t interval over $W windows" "$P"
  grep -qE '^  decode top-1 agreement [0-9]+/[0-9]+ steps' "$P"
  grep -qE "^  from $FROM: top-1 agreement [0-9]+/[0-9]+ steps .*below $FROM: bit for bit in all $W windows$" "$P"
  echo "gates 7 and 8 at $C: every line printed, every step below $FROM bit for bit"
  python3 -c "$BAND_PY" "$P" "$FROM" | tee "$O/band-$C.txt"
  if ! grep -qx 'band: inside' "$O/band-$C.txt"; then
    if [ "$C" = 4096 ]; then outcome 'N2: the 4,096 interval leaves the band; not proposed'
    else outcome 'N3: the 8,192 interval leaves the band; not proposed'; fi
    exit 0
  fi
done
outcome 'P: replay identical, both intervals inside the band; the T8 of the kernel form with its threshold is proposed to the operator'
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
  printf '%s\n' "$JOBSCRIPT" | bash -n && echo "DRY_RUN: job script parses; nothing launched" >&2
  exit 0
fi

uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor l40sx1 --timeout "$TIMEOUT" \
  --bucket "$BUCKET" --out-mount /out \
  --name "attn-t8-r2-4b${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
