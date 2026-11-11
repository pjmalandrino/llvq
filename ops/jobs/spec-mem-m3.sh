#!/usr/bin/env bash
# Job M3 of the speculative loop's memory axis (prereg proofs/preregistration-spec-mem-m3-2026-11-11.md): on one L40S,
# the sealed 4B under its served v9 config, the verify pass a12 against a12ip (ff3bb32: the tree pass on the served
# words read in place, no tree-tiled copy), against base Tetra (the served 4E object alone). Each call is its own
# process under a 100 ms nvidia-smi sampler (the duel's, ops/jobs/duel-iq2.sh), each step writes its own files and is
# non-fatal for the next:
#   (0) base: `LLVQ_CONFIG=<v9> LLVQ_CONTEXTS=5 fusedrun <sealed> 256`, then the same at LLVQ_CONTEXTS=8192 (the
#       pic-vram-8k form): no drafter, no tree pass; base peak VRAM and decode tok/s on the same card and image;
#   (a) house 44, 256 tokens, `--vocab q4 --readout device`, drafter dflash:tree f16: once a12@8.10.8.20.6, once
#       a12ip@8.10.8.20.6;
#   (b) GSM8K-200 at 768 tokens with a12ip, dump gsm200-ip.jsonl, compared with SP8's dump (in the bucket at
#       spec-loop-sp8-2026-11-11/gsm200.jsonl, checked against the journal's copy by sha256);
#   (c) one 8,191-token prompt (ops/data/spec-long-8k-v1.jsonl) with a12ip, 256 tokens.
# No dflashcheck step: SP8's gate on f16 is not rerun. Inputs and controls as SP8. l40sx1 only: no twin.
#
#   DRY_RUN=1 bash ops/jobs/spec-mem-m3.sh             parses the job script, launches nothing
#   IMAGE_SHA=<space sha> bash ops/jobs/spec-mem-m3.sh  launches
#
# Cost, estimated (l40sx1 $1.80/h, prereg deviation E1): about 58 min, $1.74; timeout 85m, $2.55.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-spec-mem-m3-2026-11-11.md
OUT_NAME=spec-mem-m3-2026-11-11
IMAGE_SHA=${IMAGE_SHA:-}
BUCKET=Pier-Jean/jobs-artifacts
O=/out/$OUT_NAME${RETRY:+-r$RETRY}
DRY=${DRY_RUN:-0}
P4E_V9_SHA=1b422b62328be6752df4add43280565fd2a49cfa20ee26953b301abe7a15c530
TIMEOUT=${TIMEOUT:-85m}
ARM_A='a12@8.10.8.20.6'
ARM_B='a12ip@8.10.8.20.6'
DRAFTERS='dflash:tree'
DREPO=z-lab/Qwen3-4B-DFlash-b16
DREV=b74e3a329c4d963783143b1e970d95b002be72bd
DCFG_SHA=2de97cdf48f429628074d229d024753f1ea02b575732983d6ed2bc882bbd63c8
DMODEL_SHA=a21d0ffe5970c84e9a0710ccb12087d1a4bcbedf979e623476c92c9d005a9cd4
NTOK=256
NTOK_GSM=768
PROMPTS=/usr/local/share/llvq/spec-prompts-v1.jsonl
PROMPTS_SHA=$(shasum -a 256 < ops/data/spec-prompts-v1.jsonl | cut -d' ' -f1)
[ "$PROMPTS_SHA" = f749eb359c2cf5c8a648ca99fc0f856e620f8adb3f1d233f4c477ff322257f1a ] \
  || { echo "refused: ops/data/spec-prompts-v1.jsonl is not at its pinned sha256" >&2; exit 1; }
GSM=/usr/local/share/llvq/spec-gsm8k-200.jsonl
GSM_SHA=$(shasum -a 256 < ops/data/spec-gsm8k-200.jsonl | cut -d' ' -f1)
[ "$GSM_SHA" = 9e778f58f8a9e6c093aad13abac87ff279c967ab3eb39549910eb47b343c6a48 ] \
  || { echo "refused: ops/data/spec-gsm8k-200.jsonl is not at its pinned sha256" >&2; exit 1; }
LONG=/usr/local/share/llvq/spec-long-8k-v1.jsonl
LONG_SHA=$(shasum -a 256 < ops/data/spec-long-8k-v1.jsonl | cut -d' ' -f1)
[ "$LONG_SHA" = da4edbd6f12184cf8ff938f057500f3e30e35d8c7a351d2e925316d225d62ebc ] \
  || { echo "refused: ops/data/spec-long-8k-v1.jsonl is not at its pinned sha256" >&2; exit 1; }
REF=spec-loop-sp8-2026-11-11/gsm200.jsonl
REF_LOCAL=docs/mesures/spec-loop-sp8-2026-11-11-brut/gsm200.jsonl
REF_SHA=$(shasum -a 256 < "$REF_LOCAL" | cut -d' ' -f1)
FLAVOR=${FLAVOR:-l40sx1}
case "$FLAVOR" in
  l40sx1) FLAVOR_ARGS=(); DEV_PRE='' ;;
  *) echo "refused: FLAVOR=$FLAVOR (M3 runs on l40sx1 only, no twin)" >&2; exit 1 ;;
esac
SEALED="sealed-4b-2026-09-23/qwen3-4b-sealed.bin:1418224685"
SEALED_SHA=886391a8c03f66dc269cc65c3598c6627dbdcd259180aff36604ef10d37371b8
C=configs/qwen3-4b-tetra-e4-4e-v9.json
grep -q '^  "rows": "v9",$' "$C" || { echo "refused: $C does not name rows v9" >&2; exit 1; }
[ "$(printf '%s\n' "$(cat "$C")" | shasum -a 256 | cut -d' ' -f1)" = "$(shasum -a 256 < "$C" | cut -d' ' -f1)" ] \
  || { echo "refused: $C does not end in exactly one newline" >&2; exit 1; }
CFG_SHA=$(shasum -a 256 < "$C" | cut -d' ' -f1)
CFG_B64=$(base64 < "$C" | tr -d '\n')

if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  path=${SEALED%%:*}; bytes=${SEALED##*:}
  REMOTE=$(hf buckets ls "hf://buckets/$BUCKET/${path%/*}/" 2>/dev/null \
    | awk -v n="/${path##*/}" 'substr($NF, length($NF) - length(n) + 1) == n {print $1}' || true)
  [ "$REMOTE" = "$bytes" ] || { echo "refused: $path is ${REMOTE:-absent} B in the bucket, expected $bytes" >&2; exit 1; }
  rbytes=$(wc -c < "$REF_LOCAL" | tr -d ' ')
  RREMOTE=$(hf buckets ls "hf://buckets/$BUCKET/${REF%/*}/" 2>/dev/null \
    | awk -v n="/${REF##*/}" 'substr($NF, length($NF) - length(n) + 1) == n {print $1}' || true)
  [ "$RREMOTE" = "$rbytes" ] || { echo "refused: $REF is ${RREMOTE:-absent} B in the bucket, expected $rbytes" >&2; exit 1; }
  if hf buckets ls "hf://buckets/$BUCKET/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already exists" >&2; exit 1
  fi
  IMAGE_SHA=${IMAGE_SHA:?IMAGE_SHA=<the Space sha of the rebuild that carries this branch>}
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    'from huggingface_hub import HfApi; print(HfApi().space_info("Pier-Jean/llvq-runner-cuda").sha)')
  [ "$NOW" = "$IMAGE_SHA" ] || { echo "refused: the Space is at $NOW, expected $IMAGE_SHA" >&2; exit 1; }
  echo "bucket copies verified by size; image $NOW"
fi

PRE=$(printf 'O=%q\nSEALED=%q\nSEALED_SHA=%q\nCFG_SHA=%q\nCFG_B64=%q\nP4E_V9_SHA=%q\nARM_A=%q\nARM_B=%q\nDRAFTERS=%q\nNTOK=%q\nNTOK_GSM=%q\nPROMPTS=%q\nPROMPTS_SHA=%q\nGSM=%q\nGSM_SHA=%q\nLONG=%q\nLONG_SHA=%q\nREF=%q\nREF_SHA=%q\nDREPO=%q\nDREV=%q\nDCFG_SHA=%q\nDMODEL_SHA=%q\nexport HF_HOME=/tmp/hf' \
  "$O" "$SEALED" "$SEALED_SHA" "$CFG_SHA" "$CFG_B64" "$P4E_V9_SHA" "$ARM_A" "$ARM_B" "$DRAFTERS" "$NTOK" "$NTOK_GSM" "$PROMPTS" "$PROMPTS_SHA" "$GSM" "$GSM_SHA" "$LONG" "$LONG_SHA" "$REF" "$REF_SHA" "$DREPO" "$DREV" "$DCFG_SHA" "$DMODEL_SHA")$DEV_PRE

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,clocks.max.sm --format=csv,noheader | tee "$O/gpu.txt"
echo '== control 1: oracle =='
timeout 5m oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle.txt" | tail -2
grep -q MATCH "$O/oracle.txt"
echo '== control 1: the object and its config =='
S=/out/${SEALED%%:*}; SB=${SEALED##*:}
test "$(stat -c %s "$S")" = "$SB"
sha256sum "$S" | tee "$O/files.sha256"
test "$(sha256sum "$S" | cut -d' ' -f1)" = "$SEALED_SHA"
PT=$O/qwen3-4b-tetra-e4-4e-v9.json
printf '%s' "$CFG_B64" | base64 -d > "$PT"
test "$(sha256sum "$PT" | cut -d' ' -f1)" = "$CFG_SHA"
echo '== inputs: the drafter, the prompt sets, SP8 dump =='
DD=/tmp/dflash-$DREV
mkdir -p "$DD"
for f in config.json model.safetensors; do
  curl -fsSL --retry 3 -o "$DD/$f" "https://huggingface.co/$DREPO/resolve/$DREV/$f"
done
sha256sum "$DD/config.json" "$DD/model.safetensors" | tee "$O/drafter.sha256" | tee -a "$O/files.sha256"
test "$(sha256sum "$DD/config.json" | cut -d' ' -f1)" = "$DCFG_SHA"
test "$(sha256sum "$DD/model.safetensors" | cut -d' ' -f1)" = "$DMODEL_SHA"
for pair in "$PROMPTS:$PROMPTS_SHA" "$GSM:$GSM_SHA" "$LONG:$LONG_SHA" "/out/$REF:$REF_SHA"; do
  f=${pair%:*}; h=${pair##*:}
  sha256sum "$f" | tee -a "$O/files.sha256"
  test "$(sha256sum "$f" | cut -d' ' -f1)" = "$h"
done
cp "/out/$REF" /tmp/gsm200-sp8.jsonl
# VRAM sampler (ops/jobs/duel-iq2.sh): memory.used every 100 ms around one command; idle and peak in MiB.
vram() {
  local tag=$1; shift
  local idle; idle=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 100 > "$O/vram-$tag.csv" &
  local sp=$!
  local rc=0
  "$@" || rc=$?
  kill "$sp"; wait "$sp" 2>/dev/null || true
  local peak; peak=$(sort -n "$O/vram-$tag.csv" | tail -1)
  echo "vram $tag idle $idle MiB peak $peak MiB" | tee -a "$O/vram.txt" >&2
  return $rc
}
set +e
echo "== (0) base: the served config alone, fusedrun, 5-token prompt =="
date -u
vram base-short env LLVQ_CONFIG="$PT" LLVQ_CONTEXTS=5 timeout 12m fusedrun "$S" 256 > "$O/base-short.txt" 2>&1
echo "fusedrun base short exit $?" >> "$O/base-short.txt"
echo "== (0) base: the served config alone, fusedrun, 8,192-token prompt =="
date -u
vram base-8k env LLVQ_CONFIG="$PT" LLVQ_CONTEXTS=8192 timeout 15m fusedrun "$S" 256 > "$O/base-8k.txt" 2>&1
echo "fusedrun base 8k exit $?" >> "$O/base-8k.txt"
echo "== (a) specbench, house 44, $ARM_A =="
date -u
vram house-a12 env LLVQ_CONFIG="$PT" timeout 16m specbench --prompts "$PROMPTS" --vocab q4 --readout device "$S" "$ARM_A" "$DRAFTERS" "$NTOK" "$DD" f16 > "$O/spec-house-a12.txt" 2>&1
echo "specbench house a12 exit $?" >> "$O/spec-house-a12.txt"
echo "== (a) specbench, house 44, $ARM_B =="
date -u
vram house-a12ip env LLVQ_CONFIG="$PT" timeout 16m specbench --prompts "$PROMPTS" --vocab q4 --readout device "$S" "$ARM_B" "$DRAFTERS" "$NTOK" "$DD" f16 > "$O/spec-house-a12ip.txt" 2>&1
echo "specbench house a12ip exit $?" >> "$O/spec-house-a12ip.txt"
echo "== (b) specbench, gsm8k 200, $ARM_B, dump =="
date -u
vram gsm200-a12ip env LLVQ_CONFIG="$PT" timeout 30m specbench --prompts "$GSM" --vocab q4 --readout device --dump "$O/gsm200-ip.jsonl" "$S" "$ARM_B" "$DRAFTERS" "$NTOK_GSM" "$DD" f16 > "$O/spec-gsm200-ip.txt" 2>&1
echo "specbench gsm200 a12ip exit $?" >> "$O/spec-gsm200-ip.txt"
{
  echo "dump against SP8's ($REF, sha256 $REF_SHA):"
  if cmp /tmp/gsm200-sp8.jsonl "$O/gsm200-ip.jsonl"; then echo "cmp: byte for byte identical"; else echo "cmp: DIFFER"; fi
  awk 'NR == FNR { if (match($0, /"decoder":"[^"]*"/)) d = substr($0, RSTART + 11, RLENGTH - 12)
                   if (match($0, /"id":"[^"]*"/)) i = substr($0, RSTART + 6, RLENGTH - 7)
                   ref[i SUBSEP d] = $0; next }
       { if (match($0, /"decoder":"[^"]*"/)) d = substr($0, RSTART + 11, RLENGTH - 12)
         if (match($0, /"id":"[^"]*"/)) i = substr($0, RSTART + 6, RLENGTH - 7)
         tot[d]++; ids[i] = 1
         if ((i SUBSEP d) in ref && ref[i SUBSEP d] == $0) same[d]++; else bad[i] = 1 }
       END { for (d in tot) printf "decoder %s: %d of %d prompts identical to SP8\n", d, same[d] + 0, tot[d]
             n = 0; m = 0; for (i in ids) { m++; if (!(i in bad)) n++ }
             printf "prompts identical to SP8 on every decoder: %d of %d\n", n, m }' \
      /tmp/gsm200-sp8.jsonl "$O/gsm200-ip.jsonl"
} > "$O/gsm200-vs-sp8.txt" 2>&1
cat "$O/gsm200-vs-sp8.txt"
echo "== (c) specbench, one 8,191-token prompt, $ARM_B =="
date -u
vram long8k-a12ip env LLVQ_CONFIG="$PT" timeout 16m specbench --prompts "$LONG" --vocab q4 --readout device "$S" "$ARM_B" "$DRAFTERS" "$NTOK" "$DD" f16 > "$O/spec-long8k-a12ip.txt" 2>&1
echo "specbench long8k a12ip exit $?" >> "$O/spec-long8k-a12ip.txt"
date -u
# (peak VRAM, tok/s) a speculative arm beside the base pair. Peak: nvidia-smi, MiB x 1,048,576 / 1e9 GB. Base tok/s:
# fusedrun's decode tok/s after the prompt (LLVQ_CONTEXTS, median of 5). Arm tok/s: specbench's dflash:tree "all"
# line, total tokens over total wall; its "x greedy" is against specbench's own 4E greedy on the same text.
peak_gb() { awk -v t="$1" '$1 == "vram" && $2 == t { printf "%.3f", $7 * 1048576 / 1e9 }' "$O/vram.txt"; }
base_tps() { sed -nE "s/^ +$1 +[0-9.]+ \[[^]]*\] +([0-9.]+) \[.*/\1/p" "$2" | head -1; }
spec_tps() { grep -E 'dflash:tree +(all|rag):' "$1" | tail -1 | sed -nE 's/.* ([0-9.]+) tok\/s, greedy ([0-9.]+), x([0-9.]+);.*/\1 \2 \3/p'; }
BS_P=$(peak_gb base-short); BS_T=$(base_tps 5 "$O/base-short.txt")
BL_P=$(peak_gb base-8k); BL_T=$(base_tps 8192 "$O/base-8k.txt")
{
  echo "pairs (peak VRAM GB by nvidia-smi, tok/s) against base Tetra, the served 4E config alone (fusedrun):"
  echo "  base short (5-token prompt): peak ${BS_P:-?} GB, decode ${BS_T:-?} tok/s"
  echo "  base 8k (8,192-token prompt): peak ${BL_P:-?} GB, decode ${BL_T:-?} tok/s"
  for row in "house-a12:spec-house-a12:short" "house-a12ip:spec-house-a12ip:short" "gsm200-a12ip:spec-gsm200-ip:short" "long8k-a12ip:spec-long8k-a12ip:8k"; do
    tag=${row%%:*}; rest=${row#*:}; f=${rest%%:*}; b=${rest##*:}
    if [ "$b" = short ]; then bp=$BS_P; bt=$BS_T; else bp=$BL_P; bt=$BL_T; fi
    p=$(peak_gb "$tag"); read -r t g x <<< "$(spec_tps "$O/$f.txt")"
    awk -v tag="$tag" -v b="$b" -v p="${p:-nan}" -v t="${t:-nan}" -v g="${g:-nan}" -v x="${x:-nan}" -v bp="${bp:-nan}" -v bt="${bt:-nan}" \
      'BEGIN { printf "  %-13s peak %s GB, %s tok/s | base %s: %+.3f GB, x%.3f tok/s | against specbench 4E greedy on the same text (%s tok/s): x%s\n", tag, p, t, b, p - bp, t / bt, g, x }'
  done
} > "$O/pairs.txt" 2>&1
cat "$O/pairs.txt"
set -e
{
  cat "$O/vram.txt"
  cat "$O/pairs.txt"
  cat "$O/gsm200-vs-sp8.txt"
  for f in base-short base-8k; do
    grep -E '^ +(5|8192) +[0-9.]+ \[|^  prompt |exit ' "$O/$f.txt" | sed "s/^/$f: /" || true
  done
  for f in spec-house-a12 spec-house-a12ip spec-gsm200-ip spec-long8k-a12ip; do
    grep -E '^(card memory|4E tree|loaded in|arm |drafter )' "$O/$f.txt" | sed "s/^/$f: /" || true
    grep -vE '^(target|cudarc|shared rotation|projection fusion|fused layout|NVRTC|int4|embedding|total expected|attn_dec|4E|batch|tv_|attn_|card memory|loaded in|arm |drafter |== |  4E agreement)' "$O/$f.txt" | sed "s/^/$f: /" || true
  done
} > "$O/summary.txt"
tail -80 "$O/summary.txt"
for f in base-short base-8k spec-house-a12 spec-house-a12ip spec-gsm200-ip spec-long8k-a12ip; do
  grep -qF "4E: persistent module 252757 bytes, sha256 $P4E_V9_SHA, row form 7," "$O/$f.txt" \
    && echo "control 2 ($f): the served 4E v9 module" | tee -a "$O/summary.txt" \
    || echo "VOID: control 2 ($f), the 4E module is not the served v9 module" | tee -a "$O/summary.txt"
done
ls -la "$O"
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
  --flavor "$FLAVOR" ${FLAVOR_ARGS[@]+"${FLAVOR_ARGS[@]}"} --timeout "$TIMEOUT" \
  --bucket "$BUCKET" --out-mount /out \
  --name "spec-mem-m3" \
  "$PRE" "$BODY"
