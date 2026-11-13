#!/usr/bin/env bash
# The two Mac encodes of the guided gate/up attempt (V1), baseline then candidate, then their seals.
# Preregistered: proofs/preregistration-guided-gateup-4b-2026-11-13.md. LOCAL: launched by the lead
# on the operator's go, never by an agent, never beside another heavy job.
#
# The recipe is the 2026-09-18 DCLM base's (prereg dclm-4b-2026-09-18 section 1: tetra1, rotation
# 0x110feed, nogs, h_shrink 1, gain_scale 1, LLVQ_INT4_TYPES=v_proj, dclm-edu 64 x 2048 from the
# prefix), on the command line of ops/jobs/encode-14b.sh lines 355-357 with the Mac device and every
# block. The 2026-09-18 command line itself is not journalled; this one is reconstructed from those
# two sources. The ONLY difference between the arms is LLVQ_GUIDED_WEIGHTS (and LLVQ_GUIDED_GROUPS).
#
# Expected, per arm (from the journals, *estimated* beyond them): the base encode took 1 h 47 on this
# Mac at 12 threads (dclm-4b-2026-09-18 line 4; phases 6,408.6 s, innovation-cout line 33). The
# candidate adds 8 weighted accumulations per window and 8 factorizations of 2,560 per block, about
# +13 % of the 770.7 s factor phase and a few percent of capture (*computed* from n^3 and GEMM counts):
# 1 h 50 to 2 h. Peak memory of a 4B Mac encode is not journalled; the 8B's footprint was 114.5 GB for
# 23.2 GB resident (innovation-cout line 147), so the 4B is expected at 40 to 60 GB on this 64 GiB Mac
# (*estimated*, run nothing else beside it). Seal: minutes each. Both arms: about 4 h of wall clock.
#
#   bash ops/jobs/guided-encode-4b.sh tokens [--force]  writes the calibration token file (tokenizer and
#                                                      corpus only, no weights) for the card job; refuses
#                                                      to overwrite an existing one without --force
#   DRY_RUN=1 bash ops/jobs/guided-encode-4b.sh         prints what it would run and checks the inputs
#   bash ops/jobs/guided-encode-4b.sh                   runs both encodes and both seals, one at a time
#
# Needs, built beforehand (release, `--features metal,fast-linalg`): target/release/{smoke,seal,calibtokens}.
# Needs the weights of the card job in $WEIGHTS (default ~/llvq-guided-2026-11-13/weights), fetched with
#   hf buckets cp -r hf://buckets/Pier-Jean/jobs-artifacts/guided-weights-4b-2026-11-13 <dir>
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

D=${OUT_DIR:-$HOME/llvq-guided-2026-11-13}
WEIGHTS=${WEIGHTS:-$D/weights}
GROUPS_G=${GROUPS_G:-4}
MODEL=Qwen/Qwen3-4B
MODEL_SHA=1cfa9a7208912126459214e8b04321603b3df60c
HFHUB=$HOME/.cache/huggingface/hub
NCPU=$(sysctl -n hw.ncpu)
THREADS=$((NCPU - 4))
BIN=$REPO/target/release
NEED_GB=12          # two .llvq of ~1.8 GB, two sealed of ~1.8 GB, logs, margin
DRY=${DRY_RUN:-0}
SMOKE_ARGS="64 2048 12 4096 metal nogs tetra1 999 rot"
COMMON="LLVQ_MODEL=$MODEL LLVQ_CALIB=dclm-edu LLVQ_INT4_TYPES=v_proj LLVQ_THREADS=$THREADS"

refuse() { echo "refused: $*" >&2; exit 1; }
need()  { grep -qF -- "$2" "$1" || refuse "$1 lacks: $2"; }
never() { if grep -qF -- "$2" "$1"; then refuse "$1 holds: $2"; fi; }

if [ "${1:-}" = tokens ]; then
  mkdir -p "$D"
  test -x "$BIN/calibtokens" || refuse "$BIN/calibtokens not built"
  if [ -e "$D/calib-tokens-4b.u32" ] || [ -e "$D/calib-tokens-4b.u32.json" ]; then
    [ "${2:-}" = --force ] || refuse "$D/calib-tokens-4b.u32 exists; pass --force to overwrite it"
  fi
  env LLVQ_MODEL=$MODEL LLVQ_CALIB=dclm-edu "$BIN/calibtokens" 64 2048 "$D/calib-tokens-4b.u32" 2>&1 | tee "$D/calibtokens.txt"
  exit 0
fi

# ---- inputs, checked without loading anything ----
inputs() {
  local ref; ref=$(tr -d ' \n' < "$HFHUB/models--Qwen--Qwen3-4B/refs/main" 2>/dev/null || true)
  [ "$ref" = "$MODEL_SHA" ] || refuse "refs/main of $MODEL is '${ref:-absent}', expected $MODEL_SHA"
  local snap=$HFHUB/models--Qwen--Qwen3-4B/snapshots/$MODEL_SHA
  for f in config.json tokenizer.json model.safetensors.index.json model-00001-of-00003.safetensors \
           model-00002-of-00003.safetensors model-00003-of-00003.safetensors; do
    test -e "$snap/$f" || refuse "$snap/$f missing from the HF cache"
  done
  ls -d "$HFHUB/datasets--HuggingFaceTB--dclm-edu" >/dev/null 2>&1 || refuse "the dclm-edu corpus is not in the HF cache"
  for b in smoke seal; do test -x "$BIN/$b" || refuse "$BIN/$b not built (release, --features metal,fast-linalg)"; done
  local newer; newer=$(find llvq-llm/src llvq-quant/src llvq-search/src llvq-core/src llvq-artifact/src -name '*.rs' -newer "$BIN/smoke" | head -3)
  [ -z "$newer" ] || refuse "sources newer than $BIN/smoke: $newer"
  test -f "$WEIGHTS/manifest.json" || refuse "$WEIGHTS/manifest.json missing"
  local n; n=$(ls "$WEIGHTS"/w-*-gate_proj.f32 "$WEIGHTS"/w-*-up_proj.f32 2>/dev/null | wc -l | tr -d ' ')
  [ "$n" = 72 ] || refuse "$WEIGHTS holds $n weight files, 72 expected"
  ( cd "$WEIGHTS" && shasum -a 256 -c files.sha256 >/dev/null ) || refuse "$WEIGHTS/files.sha256 does not check"
  local free; free=$(df -g "$HOME" | awk 'NR == 2 {print $4}')
  [ "$free" -ge "$NEED_GB" ] || refuse "$free GB free under $HOME, $NEED_GB needed"
  [ ! -e "$D/baseline.llvq" ] && [ ! -e "$D/candidate.llvq" ] || refuse "$D already holds an arm; move it away first"
  echo "inputs ok: checkpoint $MODEL@$MODEL_SHA cached, dclm-edu cached, binaries built after the sources," \
       "72 weight files checked, $free GB free, $THREADS encoder threads"
}

BASE_CMD="env $COMMON LLVQ_ARTIFACT=$D/baseline.llvq $BIN/smoke $SMOKE_ARGS"
CAND_CMD="env $COMMON LLVQ_GUIDED_WEIGHTS=$WEIGHTS LLVQ_GUIDED_GROUPS=$GROUPS_G LLVQ_ARTIFACT=$D/candidate.llvq $BIN/smoke $SMOKE_ARGS"
echo "baseline : $BASE_CMD"
echo "candidate: $CAND_CMD"
echo "seal     : env LLVQ_MODEL=$MODEL $BIN/seal <arm>.llvq <arm>.bin"
echo "guard    : ~/.llvq-guard/heavy.sh (one heavy job at a time, nice 15), memory_pressure -Q before each arm"
inputs

if [ "$DRY" = 1 ]; then
  echo "DRY_RUN: nothing launched"
  exit 0
fi

# The resolved configuration both logs must print (smoke.rs header), as encode-14b.sh checks it.
recipe_ok() {
  need "$1" 'codebook     tetra1 → Tetra word map'
  need "$1" 'calibration  dclm-edu, 64 × 2048 = 131072 tokens requested, contiguous prefix from token 0'
  need "$1" 'rotation     on (seed 0x110feed)'
  need "$1" '  mode         nogs (group scales off, design C off'
  need "$1" "  device       "
  need "$1" ", $THREADS encoder threads"
  need "$1" '  damping      1e-2 (relative to mean(diag H))'
  need "$1" '  h_shrink     1 (H as is, published path)'
  need "$1" '  gain_scale   1 (centroids as fitted, published path)'
  need "$1" 'weights identical, bit for bit (at f32)'
  never "$1" 'WITHOUT `fast-linalg`'
}

arm() {  # $1 name, $2 command
  local name=$1 cmd=$2
  echo "== $name: $(date '+%F %T') ==" ; memory_pressure -Q | tail -1
  # shellcheck disable=SC2086
  ~/.llvq-guard/heavy.sh "recherche-qualite: guided encode $name" $cmd > "$D/$name-smoke.txt" 2>&1
  recipe_ok "$D/$name-smoke.txt"
  echo "== seal $name: $(date '+%F %T') =="
  ~/.llvq-guard/heavy.sh "recherche-qualite: seal $name" env LLVQ_MODEL=$MODEL "$BIN/seal" "$D/$name.llvq" "$D/$name.bin" \
    > "$D/$name-seal.txt" 2>&1
  ( cd "$D" && shasum -a 256 "$name.llvq" "$name.llvq.state" "$name.bin" ) | tee "$D/$name.sha256"
  stat -f '%z %N' "$D/$name.llvq" "$D/$name.bin" | tee -a "$D/$name.sizes"
}

mkdir -p "$D"
arm baseline "$BASE_CMD"
# Informational only, decides nothing: the baseline's sealed file against the 2026-09-18 DCLM base,
# 1,794,564,765 B, sha256 471f39883b0baabc42b90c83... (prereg dclm-4b-2026-09-18 lines 20-21). The
# repository records only those first 24 hex digits of its hash; the comparison reads that prefix.
BASE0918_BYTES=1794564765
BASE0918_SHA_PREFIX=471f39883b0baabc42b90c83
BB=$(stat -f %z "$D/baseline.bin")
BH=$(awk '$2 == "baseline.bin" {print $1}' "$D/baseline.sha256")
if [ "$BB" = "$BASE0918_BYTES" ] && [ "${BH:0:24}" = "$BASE0918_SHA_PREFIX" ]; then V=EQUAL; else V=DIFFERENT; fi
echo "baseline against the 2026-09-18 base: $V ($BB B against $BASE0918_BYTES; sha256 $BH against" \
     "$BASE0918_SHA_PREFIX...)" | tee "$D/against-0918.txt"
never "$D/baseline-smoke.txt" 'guided weights:'
need "$D/baseline-smoke.txt" '  guided       off'
if grep -q '^guided' "$D/baseline.llvq.state"; then refuse "the baseline's state names guided weights"; fi
arm candidate "$CAND_CMD"
need "$D/candidate-smoke.txt" "gate_proj, up_proj against gradient-weighted Hessians, G = $GROUPS_G"
grep -q "^guided = gate_proj,up_proj G=$GROUPS_G manifest_sha256=" "$D/candidate.llvq.state" \
  || refuse "the candidate's state does not name its guided weights"

# Control: same format, same record counts, so the same byte count; the 2026-09-18 base sealed to
# 1,794,564,765 B (prereg dclm-4b-2026-09-18 line 20).
SB=$(stat -f %z "$D/baseline.bin"); SC=$(stat -f %z "$D/candidate.bin")
[ "$SB" = "$SC" ] || refuse "sealed sizes differ: $SB against $SC"
echo "sealed size $SB B both arms (2026-09-18 base: 1794564765)" | tee "$D/size.txt"
# Control: the two arms calibrated on the same tokens.
FB=$(grep -o 'calibration token fingerprint [0-9a-f]*' "$D/baseline-smoke.txt")
FC=$(grep -o 'calibration token fingerprint [0-9a-f]*' "$D/candidate-smoke.txt")
[ -n "$FB" ] && [ "$FB" = "$FC" ] || refuse "token fingerprints differ: '$FB' against '$FC'"
echo "$FB, both arms" | tee "$D/fingerprint.txt"
# Control: the state files differ by the guided line and nothing else. Refused otherwise, before
# anything is uploaded or scored (prereg section 4, control 3).
if ! diff <(grep -v '^guided' "$D/baseline.llvq.state") <(grep -v '^guided' "$D/candidate.llvq.state") > "$D/state-diff.txt"; then
  cat "$D/state-diff.txt" >&2
  refuse "the two run states differ beyond the guided line (diff above, kept in $D/state-diff.txt)"
fi
echo "states equal but for the guided line" | tee -a "$D/fingerprint.txt"
grep -h '^exact-ppl\|baseline ppl' "$D"/baseline-smoke.txt "$D"/candidate-smoke.txt | tee "$D/ppl.txt" || true
echo "done: $(date '+%F %T'); upload with:"
echo "  hf buckets cp $D/baseline.bin hf://buckets/Pier-Jean/jobs-artifacts/guided-encode-4b-2026-11-13/baseline.bin"
echo "  hf buckets cp $D/candidate.bin hf://buckets/Pier-Jean/jobs-artifacts/guided-encode-4b-2026-11-13/candidate.bin"
