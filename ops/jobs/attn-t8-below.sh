#!/usr/bin/env bash
# The T8 extension of the attention kernel form BELOW its threshold, on the sealed 4B: every decode row on the
# split-KV kernel from position 0 (the measurement form 4E runs at a short prompt), against the grouped form, by the
# paired decode-path perplexity. Two parts: (1) windows of 1,280, so every scored position lies below the served
# threshold (llvq_llm::kvq::KERNEL_FROM); (2) windows of 4,096, the whole object under the extended form, on the
# same 12 windows as the threshold's own T8 (attn-t8-r2).
# Preregistered: proofs/preregistration-attn-t8-below-2026-11-05.md, stamped before this.
#
#   DRY_RUN=1 bash ops/jobs/attn-t8-below.sh                       prints and parses the job script
#   IMAGE_SHA=<sha> EMIT=<file> bash ops/jobs/attn-t8-below.sh      writes the job script, launches nothing
#   IMAGE_SHA=<space sha> bash ops/jobs/attn-t8-below.sh            launches it, after the operator's go
#
# Part 1 runs first and ends the job if its interval leaves the band (N1). Every LLVQ_* variable is set on its own
# command line, never exported. Each step has its own ceiling.
#
# Cost, estimated (l40sx1 $1.80/h, prereg section 8): 28 min, $0.84. Ceiling 50m, $1.50.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

OBJ_DIR=sealed-4b-2026-09-23
NAME=qwen3-4b-sealed.bin
LOCAL=$HOME/q4b-sealed-2026-09-23/$NAME
TIMEOUT=50m
PREREG=proofs/preregistration-attn-t8-below-2026-11-05.md
BUCKET=Pier-Jean/jobs-artifacts
FROM=1280
FLAGS='LLVQ_FUSED_LAYOUT=tetra48 LLVQ_ROT_SHARE=1 LLVQ_FUSE=0 LLVQ_KV=f16 LLVQ_EMBED=q4'
# windows: ctx:count:first index:ceiling. 1,280 x 32 from index 77 is tokens 98,560 to 139,519, used by no earlier
# perplexity run; 4,096 x 12 from index 60 is tokens 245,760 to 294,911, attn-t8-r2's windows.
SPECS='1280:32:77:18m 4096:12:60:24m'
O=/out/attn-t8-below-4b-2026-11-05${RETRY:+-r$RETRY}
F=/out/$OBJ_DIR/$NAME
DRY=${DRY_RUN:-0}
SRC_SHA=$(shasum -a 256 < llvq-cuda/kernels/attn_dec.cu | cut -d' ' -f1)

# The band, read on the `from 0` summary line of one perplexity log: in this form every position is the kernel's.
BAND_PY='import re,sys
p,frm=sys.argv[1],sys.argv[2]
ls=[l for l in open(p,encoding="utf-8").read().splitlines() if re.match(r"^  from %s +[\d.]+ "%frm,l)]
m=re.search(r"([+-][\d.]+) % \[([+-][\d.]+) ; ([+-][\d.]+)\]",ls[0]) if len(ls)==1 else None
if not m: print("%s: no single from-%s summary line"%(p,frm)); sys.exit(1)
mid,lo,hi=(float(x) for x in m.groups())
ok=lo>=-0.12 and hi<=0.12
print("%s: from position %s, kernel / grouped - 1 = %+.4f %% [%+.4f ; %+.4f], %s the band"%(p,frm,mid,lo,hi,"inside" if ok else "NOT inside"))
print("band: inside" if ok else "band: not inside")'

# The threshold the code carries, and the mode this run needs.
grep -qx "pub const KERNEL_FROM: usize = $FROM;" llvq-llm/src/kvq.rs \
  || { echo "refused: llvq-llm/src/kvq.rs does not set KERNEL_FROM to $FROM" >&2; exit 1; }
grep -qF '[c, w, f, "kernel0"]' llvq-llm/src/bin/fusedrun.rs \
  || { echo "refused: this fusedrun has no kernel0 arm" >&2; exit 1; }

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
  echo "attn_dec.cu sha256 $SRC_SHA (file); 4b sealed: $BYTES B, sha256 $SHA; image $NOW"
fi

PRE=$(printf 'O=%q\nF=%q\nSHA=%q\nBYTES=%q\nSRC_SHA=%q\nFLAGS=%q\nFROM=%q\nSPECS=%q\nBAND_PY=%q\nexport HF_HOME=/tmp/hf' \
  "$O" "$F" "$SHA" "$BYTES" "$SRC_SHA" "$FLAGS" "$FROM" "$SPECS" "$BAND_PY")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
outcome() { echo "OUTCOME $1" | tee "$O/outcome.txt"; echo '== raw kept =='; ls -la "$O"; }
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader | tee "$O/gpu.txt"
echo '== gate 1: oracle (hard rule 10) =='
timeout 5m oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle-cuda.txt" | tail -2
grep -q MATCH "$O/oracle-cuda.txt"
echo '== gate 2: the object on the mount =='
test "$(stat -c %s "$F")" = "$BYTES"
sha256sum "$F" | tee "$O/files.sha256"
test "$(cut -d' ' -f1 "$O/files.sha256")" = "$SHA"
TILE='tile 64 (served: measured optimum for sm_89)'
PARITY='int4: tv_q4_hh equals tv_q4_h bit for bit'
part=0
for spec in $SPECS; do
  part=$((part + 1))
  IFS=: read -r C W FIRST CEIL <<< "$spec"
  P=$O/ppl-$C.txt
  echo "== part $part: $W windows of $C from window index $FIRST, every decode row on the kernel =="
  date -u
  env $FLAGS LLVQ_GQA_PPL="$C,$W,$FIRST,kernel0" timeout "$CEIL" fusedrun "$F" 2>&1 | tee "$P" | tail -9
  date -u
  grep -qF "$TILE" "$P"
  grep -qF "$PARITY" "$P"
  grep -qF 'cudarc event tracking: off (one stream, ordered by issue)' "$P"
  grep -qF "attn_dec.cu: embedded, sha256 $SRC_SHA" "$P"
  grep -qE '^attn_dec_g4_u64: the card equals the CPU replay bit for bit on [0-9]+ calls' "$P"
  grep -qF "measurement form: every decode row on the kernel, from position 0 (the served config runs the grouped path below $FROM)" "$P"
  grep -E '^GQA perplexity: ' "$P" | grep -qF "wikitext2, $W windows of $C from window $FIRST (tokens "
  grep -E '^GQA perplexity: ' "$P" | grep -qF 'against grow(256)+grouped, decode path in lockstep'
  grep -E '^GQA perplexity: ' "$P" | grep -qF 'prefill path on 1 window(s), checked bit for bit'
  grep -qF '  the kernel form takes positions 0 and later' "$P"
  test "$(grep -cE "^  window +[0-9]+/$W  every position on the kernel  decode nll " "$P")" = "$W"
  test "$(grep -cxF '  window   1: prefill path bit for bit between the two arms' "$P")" = 1
  grep -qF "kernel / grouped − 1, paired 95 % t interval over $W windows" "$P"
  grep -qE '^  decode top-1 agreement [0-9]+/[0-9]+ steps' "$P"
  grep -qE "^  from 0: top-1 agreement [0-9]+/[0-9]+ steps " "$P"
  echo "gates 3 and 4 at $C: the kernel checked on the card, every line printed"
  python3 -c "$BAND_PY" "$P" 0 | tee "$O/band-$C.txt"
  if ! grep -qx 'band: inside' "$O/band-$C.txt"; then
    if [ "$part" = 1 ]; then outcome 'N1: below 1,280 the interval leaves the band; not proposed'
    else outcome 'N2: the 4,096 interval leaves the band; not proposed'; fi
    exit 0
  fi
done
outcome 'P: both intervals inside the band; the T8 extension of the kernel form below 1,280 is proposed to the operator'
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
  --name "attn-t8-below-4b${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
