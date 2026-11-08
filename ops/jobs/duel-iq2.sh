#!/usr/bin/env bash
# The llama.cpp duel, one job a size: IQ2_XXS made by us, the llama.cpp rung at our bits, and unsloth's public
# UD-IQ2_XXS, each timed, sized and scored on the same L40S class as our served files.
# Preregistered: proofs/preregistration-duel-iq2-2026-11-08.md, to stamp before any launch.
#
#   bash ops/jobs/duel-iq2.sh upload                     inputs and scripts into the bucket (run once, before the go)
#   SIZE=4b|8b|14b DRY_RUN=1 bash ops/jobs/duel-iq2.sh   parses the job script, launches nothing
#   SIZE=4b|8b|14b bash ops/jobs/duel-iq2.sh             launches (only after the operator's go)
#   SIZE=.. RETRY=1 REUSE=1 bash ops/jobs/duel-iq2.sh    a retry that takes the first attempt's f16 gate, imatrix,
#                                                        rung and quantized files from $DIR/$SIZE/, by sha256
#
# What the job does, in order:
#   0. card, CPU, disk; the shipped inputs at their sha256;
#   1. llama.cpp at the pinned commit, built in the job for sm_89 (GGML_NATIVE for the host CPU), and the
#      window scorer ops/duel/ppl_windows.cpp linked against it;
#   2. the base checkpoint at the revision our sealed file was encoded from, converted to a f16 GGUF;
#   3. gate: the f16 GGUF on our 12 WikiText-2 windows against our own f16 reading of that checkpoint;
#   4. the imatrix on our calibration text (dclm-edu, the 131,072 tokens of the sealed files, chunks of 2,048);
#   5. the rung at our bits: llama-quantize --dry-run over IQ2_XS, IQ2_S, IQ2_M, IQ3_XXS; the one nearest our
#      file's whole-model b/param is quantized beside IQ2_XXS;
#   6. per arm: perplexity on our windows; decode speed, llama-bench tg256 after depth 5 and 8,192, 5 reps, each
#      under a 100 ms VRAM sampler; at 4B, llama-batched-bench at B = 1, 8, 32; MMLU census and GSM8K census
#      through llama-server, 16 slots.
#
# Cost, estimated at $1.80/h (prereg section 7): 4B ~80 min $2.40, timeout 2h ($3.60); 8B ~130 min $3.90, timeout
# 3h ($5.40); 14B ~220 min $6.60, timeout 5h ($9.00).
#
# The registry row (docs/data/jobs.csv):
# <date>,<job_id>,duel-iq2-<size>,l40sx1,<min>,<cost>,duel-iq2-2026-11-08.txt,"DUEL LLAMA.CPP <SIZE> : ..."
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-duel-iq2-2026-11-08.md
BUCKET=Pier-Jean/jobs-artifacts
DIR=duel-iq2-2026-11-08
DRY=${DRY_RUN:-0}
LCPP_SHA=cd959bae99ff4bed22bef2604f45241c09bbb8c8   # llama.cpp release v0.9.0, 2026-11-04
IMAGE=nvidia/cuda:12.8.1-devel-ubuntu24.04
# The shipped inputs (ops/duel/inputs.py) and the three scripts, by sha256.
INPUTS_SHA="6b69a6ba6998ddda8d1f65406ac23b3e1bc09e0a693f8809d97f0ca7f4b38aab  ppl-tokens.u32
03b5730bcc5f8c1cf55a2c1dc5070ec1942af0ab0ca7e193993e9e721fd424a4  calib-dclm.txt
eaaddf30b24c5368c716ca88b086e8be2995ff4254ad3171731c38e199cddbe8  gsm8k-prompts.jsonl
a3d875ba7265678dcb91e06643a666e78dee9f43f78a492b0f646fb1b0ad34b9  mmlu-prompts-ids.jsonl"

if [ "${1:-}" = upload ]; then
  SRC=${INPUTS_DIR:?INPUTS_DIR=<the directory ops/duel/inputs.py wrote>}
  (cd "$SRC" && printf '%s\n' "$INPUTS_SHA" | shasum -a 256 -c -) || { echo "refused: $SRC is not the prereg's inputs" >&2; exit 1; }
  for f in ppl-tokens.u32 calib-dclm.txt gsm8k-prompts.jsonl mmlu-prompts-ids.jsonl; do
    hf buckets cp "$SRC/$f" "hf://buckets/$BUCKET/$DIR/inputs/$f"
  done
  for f in ops/duel/ppl_windows.cpp ops/duel/gsm8k_llamacpp.py ops/duel/mmlu_llamacpp.py; do
    hf buckets cp "$f" "hf://buckets/$BUCKET/$DIR/inputs/${f##*/}"
  done
  hf buckets ls "hf://buckets/$BUCKET/$DIR/inputs/"
  exit 0
fi

SIZE=${SIZE:?SIZE=4b, 8b or 14b}
case "$SIZE" in
  4b)  BASE=Qwen/Qwen3-4B;  BASE_REV=1cfa9a7208912126459214e8b04321603b3df60c; PARAMS=4022468096
       OUR_BYTES=1418224685; F16_REF=12.2361; PUB_REPO=unsloth/Qwen3-4B-GGUF; PUB_REV=22c9fc8a8c7700b76a1789366280a6a5a1ad1120
       PUB_FILE=Qwen3-4B-UD-IQ2_XXS.gguf; PUB_BYTES=1255345952; BATCHED=1; TIMEOUT=2h ;;
  8b)  BASE=Qwen/Qwen3-8B;  BASE_REV=b968826d9c46dd6066d109eabc6255188de91218; PARAMS=8190735360
       OUR_BYTES=2815098745; F16_REF=8.9899; PUB_REPO=unsloth/Qwen3-8B-GGUF; PUB_REV=a6adef130ffb23ddaf1a62fec9dced968c9bc482
       PUB_FILE=Qwen3-8B-UD-IQ2_XXS.gguf; PUB_BYTES=2605221696; BATCHED=0; TIMEOUT=3h ;;
  14b) BASE=Qwen/Qwen3-14B; BASE_REV=40c069824f4251a91eefaf281ebe4c544efd3e18; PARAMS=14768307200
       OUR_BYTES=5087000541; F16_REF=7.9820; PUB_REPO=unsloth/Qwen3-14B-GGUF; PUB_REV=a04a82c4739b3ef5fa6da7d10261db2c67dd1985
       PUB_FILE=Qwen3-14B-UD-IQ2_XXS.gguf; PUB_BYTES=4470113664; BATCHED=0; TIMEOUT=5h ;;
  *) echo "refused: SIZE=$SIZE, the sizes are 4b, 8b and 14b" >&2; exit 1 ;;
esac
O=/out/$DIR/$SIZE${RETRY:+-r$RETRY}
# a REUSE retry skips the f16, the imatrix and the quantization: its ceilings, 2h / 2h / 200m, keep the duel under its cap
if [ -n "${REUSE:-}" ]; then case "$SIZE" in 4b|8b) TIMEOUT=2h ;; 14b) TIMEOUT=200m ;; esac; fi
REUSE=${REUSE:-}
KEEP=${KEEP:-}   # comma-separated arm names to measure; empty = all three
if [ -n "$REUSE" ] && [ -z "${RETRY:-}" ]; then echo "refused: REUSE=1 reads the first attempt, so it needs RETRY" >&2; exit 1; fi

SCRIPTS_SHA=$(shasum -a 256 ops/duel/ppl_windows.cpp ops/duel/gsm8k_llamacpp.py ops/duel/mmlu_llamacpp.py \
  | sed 's#  ops/duel/#  #')

if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  for f in ppl-tokens.u32 calib-dclm.txt gsm8k-prompts.jsonl mmlu-prompts-ids.jsonl ppl_windows.cpp \
           gsm8k_llamacpp.py mmlu_llamacpp.py; do
    hf buckets ls "hf://buckets/$BUCKET/$DIR/inputs/" | grep -q "$f" || { echo "refused: $f not uploaded; run 'upload'" >&2; exit 1; }
  done
  if [ -n "$REUSE" ]; then
    hf buckets ls "hf://buckets/$BUCKET/$DIR/$SIZE/" | grep -q 'gguf.sha256' \
      || { echo "refused: REUSE but $DIR/$SIZE/gguf.sha256 is missing" >&2; exit 1; }
  fi
  if hf buckets ls "hf://buckets/$BUCKET/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already exists" >&2; exit 1
  fi
fi

PRE=$(printf 'O=%q\nDIR=%q\nI=%q\nSIZE=%q\nBASE=%q\nBASE_REV=%q\nPARAMS=%q\nOUR_BYTES=%q\nF16_REF=%q\nPUB_REPO=%q\nPUB_REV=%q\nPUB_FILE=%q\nPUB_BYTES=%q\nBATCHED=%q\nREUSE=%q\nKEEP=%q\nLCPP_SHA=%q\nINPUTS_SHA=%q\nSCRIPTS_SHA=%q' \
  "$O" "$DIR" "/out/$DIR/inputs" "$SIZE" "$BASE" "$BASE_REV" "$PARAMS" "$OUR_BYTES" "$F16_REF" "$PUB_REPO" "$PUB_REV" \
  "$PUB_FILE" "$PUB_BYTES" "$BATCHED" "$REUSE" "$KEEP" "$LCPP_SHA" "$INPUTS_SHA" "$SCRIPTS_SHA")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
date -u | tee "$O/start.txt"
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,clocks.max.sm --format=csv,noheader | tee "$O/gpu.txt"
grep -m1 'model name' /proc/cpuinfo | tee "$O/cpu.txt"; nproc | tee -a "$O/cpu.txt"; df -h / | tee "$O/disk.txt"
echo '== 0. inputs =='
(cd "$I" && printf '%s\n' "$INPUTS_SHA" | sha256sum -c - && printf '%s\n' "$SCRIPTS_SHA" | sha256sum -c -) | tee "$O/inputs-check.txt"

echo '== 1. llama.cpp at the pinned commit, sm_89 =='
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git cmake build-essential python3-venv python3-dev curl ca-certificates > /dev/null
git clone -q https://github.com/ggml-org/llama.cpp /tmp/llama.cpp
cd /tmp/llama.cpp && git checkout -q "$LCPP_SHA"
test "$(git rev-parse HEAD)" = "$LCPP_SHA"
date -u
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DGGML_NATIVE=ON -DLLAMA_OPENSSL=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release > "$O/cmake.txt" 2>&1
cmake --build build -j"$(nproc)" --target llama-quantize llama-imatrix llama-bench llama-server llama-batched-bench \
  > "$O/build.txt" 2>&1 || { tail -40 "$O/build.txt"; exit 1; }
B=/tmp/llama.cpp/build/bin
g++ -O2 -std=c++17 "$I/ppl_windows.cpp" -Iinclude -Iggml/include -L"$B" -lllama -lggml -lggml-base \
  -Wl,-rpath,"$B" -o "$B/llama-ppl-windows"
date -u
sha256sum "$B"/llama-* | tee "$O/binaries.sha256"
{ "$B/llama-bench" --list-devices 2>&1 || true; } | tee "$O/devices.txt" | tail -3

echo '== 2. the base checkpoint, f16 GGUF =='
python3 -m venv /opt/v
. /opt/v/bin/activate
pip install -q --extra-index-url https://download.pytorch.org/whl/cpu \
  -r requirements/requirements-convert_hf_to_gguf.txt ./gguf-py "huggingface_hub[cli]>=0.34" tokenizers
pip freeze > "$O/pip-freeze.txt"

# VRAM sampler: memory.used every 100 ms around one command; idle and peak in MiB.
vram() {
  local tag=$1; shift
  local idle; idle=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 100 > "$O/vram-$tag.csv" &
  local sp=$!
  local rc=0
  "$@" || rc=$?
  kill "$sp"; wait "$sp" 2>/dev/null || true
  local peak; peak=$(sort -n "$O/vram-$tag.csv" | tail -1)
  # to stderr: the caller redirects the command's stdout into its own file (llama-bench's JSON)
  echo "vram $tag idle $idle MiB peak $peak MiB" | tee -a "$O/vram.txt" >&2
  return $rc
}

if [ -n "$REUSE" ]; then
P0=/out/$DIR/$SIZE
echo "== 2-5. reused from $P0: its f16 gate, imatrix, rung and quantized files, checked by sha256 =="
hf download "$BASE" tokenizer.json --revision "$BASE_REV" --local-dir /tmp/base > /dev/null
cp "$P0/gate-f16.txt" "$P0/rungs.txt" "$O/"
cat "$O/gate-f16.txt" "$O/rungs.txt"
grep -q ' inside ' "$O/gate-f16.txt"
RSTAR=$(sed -nE 's/^rung at our bits: ([A-Z0-9_]+) .*/\1/p' "$O/rungs.txt")
test -n "$RSTAR"
mkdir -p /tmp/q
for T in IQ2_XXS "$RSTAR"; do
  t=$(echo "$T" | tr 'A-Z' 'a-z')
  cp "$P0/qwen3-$SIZE-$t.gguf" /tmp/q/
done
(cd /tmp/q && for g in *.gguf; do grep " $P0/$g\$" "$P0/gguf.sha256" | sed "s# $P0/# #"; done | sha256sum -c -) \
  | tee "$O/reuse-check.txt"
test "$(grep -c ': OK$' "$O/reuse-check.txt")" = 2
QD=/tmp/q
date -u
else
hf download "$BASE" --revision "$BASE_REV" --local-dir /tmp/base > /dev/null
python3 convert_hf_to_gguf.py /tmp/base --outtype f16 --outfile /tmp/f16.gguf > "$O/convert.txt" 2>&1
ls -l /tmp/f16.gguf | tee "$O/f16-size.txt"
date -u

echo '== 3. gate: the f16 GGUF on our windows =='
"$B/llama-ppl-windows" /tmp/f16.gguf "$I/ppl-tokens.u32" 4096 12 > "$O/ppl-f16.txt" 2> "$O/ppl-f16.err"
cat "$O/ppl-f16.txt"; grep -E '^model|^tokens' "$O/ppl-f16.err"
grep -q 'fingerprint 3f1baca9033bf251' "$O/ppl-f16.err"
python3 -c "
import re,sys
p=float(re.search(r'ppl = ([0-9.]+)',open('$O/ppl-f16.txt').read()).group(1)); r=$F16_REF
d=(p/r-1)*100
print('gate f16: llama.cpp %.4f against our %.4f, %+.3f %%, %s' % (p, r, d, 'inside' if abs(d) <= 0.5 else 'OUTSIDE'), ' +-0.5 %')
" | tee "$O/gate-f16.txt"

echo '== 4. imatrix on our calibration text =='
date -u
"$B/llama-imatrix" -m /tmp/f16.gguf -f "$I/calib-dclm.txt" -o "$O/imatrix-dclm.gguf" -c 2048 -ngl 99 \
  > "$O/imatrix.txt" 2>&1 || { tail -30 "$O/imatrix.txt"; exit 1; }
grep -iE 'tokenizing|chunks|tokens' "$O/imatrix.txt" | head -5
date -u

echo '== 5. the rung at our bits =='
OUR_BPP=$(python3 -c "print('%.4f' % ($OUR_BYTES*8/$PARAMS))")
echo "our file: $OUR_BYTES B, $OUR_BPP b/param over $PARAMS" | tee "$O/rungs.txt"
for T in IQ2_XXS IQ2_XS IQ2_S IQ2_M IQ3_XXS; do
  "$B/llama-quantize" --dry-run --imatrix "$O/imatrix-dclm.gguf" /tmp/f16.gguf "$T" > "$O/dry-$T.txt" 2>&1
  MIB=$(sed -nE 's/.*quant size *= *([0-9.]+) MiB.*/\1/p' "$O/dry-$T.txt" | tail -1)
  test -n "$MIB"
  python3 -c "print('dry-run %-8s %10.2f MiB  %.4f b/param' % ('$T', $MIB, $MIB*1048576*8/$PARAMS))" | tee -a "$O/rungs.txt"
done
RSTAR=$(python3 -c "
import re
ours=$OUR_BPP
rows=[l.split() for l in open('$O/rungs.txt') if l.startswith('dry-run')]
up=[(abs(float(r[4])-ours),r[1]) for r in rows if r[1]!='IQ2_XXS']
print(min(up)[1] if up else 'NONE')")
echo "rung at our bits: $RSTAR (of IQ2_XS, IQ2_S, IQ2_M, IQ3_XXS, the nearest $OUR_BPP)" | tee -a "$O/rungs.txt"
test "$RSTAR" != NONE

echo '== quantize =='
for T in IQ2_XXS "$RSTAR"; do
  t=$(echo "$T" | tr 'A-Z' 'a-z')
  date -u
  "$B/llama-quantize" --imatrix "$O/imatrix-dclm.gguf" /tmp/f16.gguf "$O/qwen3-$SIZE-$t.gguf" "$T" "$(nproc)" \
    > "$O/quantize-$T.txt" 2>&1 || { tail -20 "$O/quantize-$T.txt"; exit 1; }
  grep -E 'quant size|model size' "$O/quantize-$T.txt"
done
QD=$O
date -u
fi
hf download "$PUB_REPO" "$PUB_FILE" --revision "$PUB_REV" --local-dir /tmp/pub > /dev/null
test "$(stat -c %s "/tmp/pub/$PUB_FILE")" = "$PUB_BYTES"
rm -f /tmp/f16.gguf
r=$(echo "$RSTAR" | tr 'A-Z' 'a-z')
ARMS="own-iq2_xxs:$QD/qwen3-$SIZE-iq2_xxs.gguf own-$r:$QD/qwen3-$SIZE-$r.gguf unsloth-ud-iq2_xxs:/tmp/pub/$PUB_FILE"
for a in $ARMS; do
  f=${a#*:}
  python3 -c "import os; b=os.path.getsize('$f'); print('size ${a%%:*} %d B  %.4f b/param  %.3f GB' % (b, b*8/$PARAMS, b/1e9))" | tee -a "$O/sizes.txt"
done
sha256sum "$QD"/qwen3-*.gguf "/tmp/pub/$PUB_FILE" | tee "$O/gguf.sha256"
if [ -n "$KEEP" ]; then
  K=""; for a in $ARMS; do case ",$KEEP," in *",${a%%:*},"*) K="$K $a" ;; esac; done
  ARMS=${K# }
  echo "arms measured: $ARMS (KEEP=$KEEP)" | tee "$O/arms.txt"
  test -n "$ARMS"
fi

echo '== 6. per arm: perplexity, speed, VRAM, MMLU, GSM8K =='
for a in $ARMS; do
  n=${a%%:*}; f=${a#*:}
  echo "---- $n ----"; date -u
  "$B/llama-ppl-windows" "$f" "$I/ppl-tokens.u32" 4096 12 > "$O/ppl-$n.txt" 2> "$O/ppl-$n.err"
  grep 'fingerprint 3f1baca9033bf251' "$O/ppl-$n.err" > /dev/null
  tail -1 "$O/ppl-$n.txt"
  for d in 5 8192; do
    vram "bench-$n-d$d" "$B/llama-bench" -m "$f" -ngl 99 -fa on -p 0 -n 256 -d "$d" -r 5 -o json \
      > "$O/bench-$n-d$d.json" 2> "$O/bench-$n-d$d.err"
    python3 -c "
import json,statistics as s
t=json.load(open('$O/bench-$n-d$d.json'))[0]; ts=t['samples_ts']
print('speed $n depth $d: tg256 %.1f tok/s [%.1f-%.1f] over %d reps, fa %s, build %s' % (s.median(ts), min(ts), max(ts), len(ts), t.get('flash_attn'), t.get('build_commit')))
" | tee -a "$O/speed.txt"
  done
  if [ "$BATCHED" = 1 ] && [ "$n" != unsloth-ud-iq2_xxs ]; then
    vram "batched-$n" "$B/llama-batched-bench" -m "$f" -ngl 99 -fa on -c 8352 -b 2048 -ub 512 \
      -npp 5 -ntg 256 -npl 1,8,32 --output-format jsonl > "$O/batched-$n.jsonl" 2> "$O/batched-$n.err"
    python3 -c "
import json
for l in open('$O/batched-$n.jsonl'):
    l=l.strip()
    if not l.startswith('{'): continue
    d=json.loads(l)
    print('batched $n B=%d: %.1f tok/s aggregate, generated tokens over prompt+decode time (t_pp %.3f s, t_tg %.3f s)' % (d['pl'], d['pl']*d['tg']/(d['t_pp']+d['t_tg']), d['t_pp'], d['t_tg']))
" | tee -a "$O/batched.txt"
  fi
  "$B/llama-server" -m "$f" -ngl 99 -fa on -np 16 -c 53248 --host 127.0.0.1 --port 8080 > "$O/server-$n.log" 2>&1 &
  SRV=$!
  ok=0
  for i in $(seq 1 200); do
    if curl -sf http://127.0.0.1:8080/health | grep -q ok; then ok=1; break; fi
    kill -0 $SRV 2>/dev/null || break
    sleep 3
  done
  [ "$ok" = 1 ] || { echo "server $n never became healthy" >&2; tail -20 "$O/server-$n.log" >&2; exit 1; }
  date -u
  python3 "$I/mmlu_llamacpp.py" "$I/mmlu-prompts-ids.jsonl" "$O/mmlu-$SIZE-$n-FULL.csv" "$f [llama.cpp $n]" \
    http://127.0.0.1:8080 16 2>&1 | tee "$O/mmlu-$n.txt"
  tail -1 "$O/mmlu-$SIZE-$n-FULL.csv" | grep -q '^# end fingerprint=a74a6d6213602979 questions=14042$'
  date -u
  python3 "$I/gsm8k_llamacpp.py" "$I/gsm8k-prompts.jsonl" /tmp/base/tokenizer.json "$O/gsm8k-$SIZE-$n.jsonl" \
    "$f [llama.cpp $n]" http://127.0.0.1:8080 16 2>&1 | tee "$O/gsm8k-$n.txt"
  tail -1 "$O/gsm8k-$SIZE-$n.jsonl" | grep -q '"fingerprint": "bfa9135ce4b55c75"'
  date -u
  kill $SRV; wait $SRV 2>/dev/null || true
done

echo '== summary =='
cat "$O/gate-f16.txt" "$O/rungs.txt" "$O/sizes.txt" "$O/speed.txt" "$O/vram.txt"
[ "$BATCHED" = 1 ] && cat "$O/batched.txt"
grep -h 'MMLU micro' "$O"/mmlu-*.txt
date -u | tee "$O/end.txt"
ls -la "$O"
JOB

if [ "$DRY" = 1 ]; then
  JOBSCRIPT="set -euo pipefail
$PRE
$BODY"
  printf '%s\n' "$JOBSCRIPT" | bash -n && echo "DRY_RUN $SIZE: job script parses; nothing launched; out $O" >&2
  exit 0
fi

uv run ops/run.py bench \
  --image "$IMAGE" --kernels-cap 89 \
  --flavor l40sx1 --timeout "$TIMEOUT" \
  --bucket "$BUCKET" --out-mount /out \
  --name "duel-iq2-$SIZE${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
