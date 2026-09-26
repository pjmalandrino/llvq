#!/usr/bin/env bash
# The SERVED checks of the 14B paper-2 object on a card: does the Tetra kernel
# serve a Qwen3-14B file, give the dense arm's tokens, and at what speed.
# ops/jobs/served-8b.sh at 14B.
#
# Preregistered: proofs/preregistration-served-14b-2026-09-22.md (PREREG), which also
# covers the smoke and bench stages of census-14b-base.sh. The launcher refuses until
# its .ots exists (DRY_RUN=1 excepted; hard rule 2).
#
# Two phases, one script:
#
#   PHASE=full   (the default) on the TRAINED file, after the fold. The published
#                protocol, in the order it runs: the prefill gate at 203; the served
#                door (57 MMLU questions through the kernel, a dump written; DOOR=1
#                by default, since the base job's smoke has no door); then fusedrun
#                256 tokens against the dense arm of the same process at the served
#                flags (q8), then the same at LLVQ_EMBED=f16, the same-head arm hard
#                rule 4 wants beside the raw ratio. 5 rounds an arm (fusedrun.rs:68).
#   PHASE=smoke  on the BASE file: the prefill gate, the door, 32 served tokens.
#                A fallback only: census-14b-base.sh runs the gate and the 32
#                tokens as its smoke stage. Use it if that stage failed and must
#                be rerun alone.
#
# Why the fused arms spell the flags out instead of LLVQ_CONFIG: the config puts
# fusedrun on a one-arm path with no dense reference (fusedrun.rs:396-464,
# configs/README.md), and the dense arm is the point.
#
# The config: configs/qwen3-14b-tetra-q5.json (CONF_LOCAL overrides). The job
# writes its bytes into $O from this command line (recorded by `hf jobs inspect`)
# and checks their sha256, so the image need not carry the file. The 4B's or the
# 8B's config is not borrowed: its path would land on every dump's `# config=` line
# and its note on the provenance line.
#
# The tile is left unset on purpose: the served policy reads 64 on sm_89, and
# LLVQ_TILE_BLOCKS beside LLVQ_CONFIG is refused anyway (served.rs:93). The job
# fails early if the log does not say "tile 64 (served: measured optimum for sm_89)".
#
# Image: the one census-14b-base.sh ran on ($LJ/image.sha, or IMAGE_SHA): the
# refusal holds the served numbers and arm A on one harness.
#
# Card memory, *computed*: the q8 arm ~5.06 GB (projections ~3.41 at ~2.06 b/weight
# + two q8 tables 1.65); the dense arm 29.54 GB (*measured*, fusedrun-14b-2026-08-17);
# fusedrun loads one arm at a time, on a 46 GB L40S.
#
# Cost, *estimated* (l40sx1 $1.80/h), scaled from served-8b-full (755 s running,
# DOOR=1): ~3.5 min of container, oracle and sha256 at 8B, ~4.5 at 14B; the door
# and the gate x ~1.9 (weights); each dense load ~265 s (Planes14 14B loaded its
# dense arm in 748.7 s, fusedrun-14b-2026-08-17.txt, x 117/330, the Tetra/Planes14
# dense-load ratio at 8B), twice, one per fusedrun process:
#   full, DOOR=1:  ~22 min, $0.66, range [18, 30] min = [$0.54, $0.90];
#                  timeout 45m, ceiling $1.35
#   smoke:         ~12 min, $0.36, range [9, 18]; timeout 40m, ceiling $1.20
#
# DRY_RUN=1 bash ops/jobs/served-14b.sh   prints the job script, parses it, launches nothing.
set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
PHASE=${PHASE:-full}
OBJ_DATE=${OBJ_DATE:-2026-09-22}
D=${RUN_DATE:-$(date -u +%F)}                      # launch date; names the output dir
PREREG=${PREREG:-proofs/preregistration-served-14b-2026-09-22.md}
BUCKET=Pier-Jean/jobs-artifacts
LJ=${LJ:-$HOME/q14b-dclm-$OBJ_DATE}
IMAGE_SHA=${IMAGE_SHA:-$(cat "$LJ/image.sha" 2>/dev/null || true)}
CONF_LOCAL=${CONF_LOCAL:-$REPO/configs/qwen3-14b-tetra-q5.json}
DOOR=${DOOR:-1}
DRY=${DRY_RUN:-0}

case "$PHASE" in
  smoke)
    OBJ_DIR=${BASE_DIR:-dclm-14b-$OBJ_DATE}         # the encode job's
    NAME=qwen3-14b-dclm.bin
    SUMS=${SUMS:-$LJ/encode-files.sha256}         # fetched by census-14b-base.sh or fold-14b.sh
    TIMEOUT=40m ;;
  full)
    OBJ_DIR=${FT_DIR:-dclm-14b-ft-$OBJ_DATE}        # fold-14b.sh upload
    NAME=qwen3-14b-dclm-ft.bin
    SUMS=${SUMS:-$LJ/ft.sha256}                   # fold-14b.sh writes the FT sha
    TIMEOUT=45m ;;
  *) echo "refused: PHASE=$PHASE, expected smoke or full" >&2; exit 1 ;;
esac
LOCAL=${LOCAL:-$LJ/$NAME}                         # optional: checked when present

O=/out/served-14b-$PHASE-$D
F=/out/$OBJ_DIR/$NAME
C=$O/qwen3-14b-tetra-q5.json

cd "$REPO"
test -f "$CONF_LOCAL" || { echo "refused: $CONF_LOCAL missing (CONF_LOCAL=<path> to point at a draft)" >&2; exit 1; }
CONF=$(cat "$CONF_LOCAL")
# The job writes printf '%s\n' "$CONF": the same bytes as the file only if it
# ends in exactly one newline. Checked, not assumed.
CONF_SHA=$(printf '%s\n' "$CONF" | shasum -a 256 | cut -d' ' -f1)
[ "$CONF_SHA" = "$(shasum -a 256 < "$CONF_LOCAL" | cut -d' ' -f1)" ] \
  || { echo "refused: $CONF_LOCAL does not end in exactly one newline" >&2; exit 1; }
python3 -c 'import json,sys; d=json.load(open(sys.argv[1]));
k={"layout","embed","rot_share","fuse","kv","note"}; assert set(d)==k, set(d)^k;
assert (d["layout"],d["embed"],d["rot_share"],d["fuse"],d["kv"])==("tetra48","q8","1","0","f16"), d;
assert "Qwen3-14B" in d["note"] and "40 v_proj" in d["note"], d["note"]' "$CONF_LOCAL"

# ---- preflight on the Mac, $0 -------------------------------------------------------
if [ "$DRY" = 1 ]; then
  SHA=0000000000000000000000000000000000000000000000000000000000000000
  BYTES=6563782117                                 # the costing's computed size, dry run only
else
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  test -s "$SUMS" || { echo "refused: $SUMS missing" >&2; exit 1; }
  SHA=$(awk -v n="/$NAME" '{p = "/" $2} substr(p, length(p) - length(n) + 1) == n {print $1}' "$SUMS" | tail -1)
  [ ${#SHA} -eq 64 ] || { echo "refused: no sha256 for $NAME in $SUMS" >&2; exit 1; }
  BYTES=$(hf buckets ls "hf://buckets/$BUCKET/$OBJ_DIR/" 2>/dev/null | awk -v n="/$NAME" 'substr($NF, length($NF) - length(n) + 1) == n {print $1}' || true)
  [ -n "$BYTES" ] || { echo "refused: no $NAME in $OBJ_DIR/" >&2; exit 1; }
  if [ -f "$LOCAL" ]; then
    [ "$SHA" = "$(shasum -a 256 "$LOCAL" | cut -d' ' -f1)" ] || { echo "refused: $LOCAL does not match $SUMS" >&2; exit 1; }
    [ "$BYTES" = "$(stat -f %z "$LOCAL")" ] || { echo "refused: bucket copy is $BYTES B, local $(stat -f %z "$LOCAL") B" >&2; exit 1; }
  fi
  if hf buckets ls "hf://buckets/$BUCKET/served-14b-$PHASE-$D/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: served-14b-$PHASE-$D/ already exists; pick another RUN_DATE" >&2; exit 1
  fi
  [ ${#IMAGE_SHA} -eq 40 ] || { echo "refused: no IMAGE_SHA, and $LJ/image.sha is absent (census-14b-base.sh writes it)" >&2; exit 1; }
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    'from huggingface_hub import HfApi; print(HfApi().space_info("Pier-Jean/llvq-runner-cuda").sha)')
  [ "$NOW" = "$IMAGE_SHA" ] || { echo "refused: the Space moved to $NOW; arm A ran on $IMAGE_SHA" >&2; exit 1; }
  echo "object: $F, $BYTES B, sha256 $SHA; config $CONF_LOCAL sha256 $CONF_SHA; image $NOW"
fi

PRE=$(printf 'O=%q\nF=%q\nC=%q\nSHA=%q\nBYTES=%q\nCONF=%q\nCONF_SHA=%q' \
  "$O" "$F" "$C" "$SHA" "$BYTES" "$CONF" "$CONF_SHA")

# Common head: the refusal of an inherited LLVQ_*, the card, oracle (hard rule
# 10), the object by bytes and sha256, the config written from this command line,
# and the prefill gate.
IFS= read -r -d '' HEAD <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader | tee "$O/gpu.txt"
echo '== oracle (hard rule 10) =='
oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle-cuda.txt" | tail -2
grep -q MATCH "$O/oracle-cuda.txt"
echo '== the object on the mount: bytes and sha256 against the Mac =='
test "$(stat -c %s "$F")" = "$BYTES"
sha256sum "$F" | tee "$O/files.sha256"
test "$(cut -d' ' -f1 "$O/files.sha256")" = "$SHA"
echo '== the served config, written from the launch command =='
printf '%s\n' "$CONF" > "$C"
sha256sum "$C" | tee -a "$O/files.sha256"
test "$(sha256sum "$C" | cut -d' ' -f1)" = "$CONF_SHA"
cat "$C"
echo '== prefill gate, 203 tokens: 50 chunks of four and one of three (configs/README.md) =='
date -u
LLVQ_CONFIG="$C" LLVQ_PREFILL_TOKENS=203 fusedrun "$F" 2>&1 | tee "$O/prefill-203.txt" | tail -14
date -u
# The served tile and the 14B object, or nothing further runs: every number below is read at 64.
grep -qF 'tile 64 (served: measured optimum for sm_89)' "$O/prefill-203.txt"
grep -qF 'Qwen3-14B' "$O/prefill-203.txt"
grep -qF '160 rot_launches/token for 280 projections' "$O/prefill-203.txt"
grep -qF '(0 groups + 240 lone + 40 int4)' "$O/prefill-203.txt"
JOB

IFS= read -r -d '' SMOKE <<'JOB' || true
echo '== the served door: 57 questions through the kernel, a dump written (configs/README.md) =='
date -u
LLVQ_CONFIG="$C" LLVQ_MMLU_ALLOC=flat LLVQ_MMLU_DUMP="$O/mmlu-14b-served-door.csv" \
  mmlu "$F" cuda 1 2>&1 | tee "$O/mmlu-door.txt" | tail -12
date -u
grep -qxF '# arithmetic=served kernel' "$O/mmlu-14b-served-door.csv"
grep -qxF "# config=$C" "$O/mmlu-14b-served-door.csv"
grep -qxF '# layout=tetra48' "$O/mmlu-14b-served-door.csv"
JOB

IFS= read -r -d '' SERVED32 <<'JOB' || true
echo '== the served path, one arm, 32 tokens: the decode runs (no dense arm by design) =='
LLVQ_CONFIG="$C" fusedrun "$F" 32 2>&1 | tee "$O/served-32.txt" | tail -8
JOB

IFS= read -r -d '' FULL <<'JOB' || true
FLAGS='LLVQ_FUSED_LAYOUT=tetra48 LLVQ_ROT_SHARE=1 LLVQ_FUSE=0 LLVQ_KV=f16'
echo '== ARM q8: the served flags spelled out, 256 tokens against the dense arm, same process =='
date -u
env $FLAGS LLVQ_EMBED=q8 fusedrun "$F" 256 2>&1 | tee "$O/fused-q8-256.txt" | tail -30
date -u
grep -qF 'tile 64 (served: measured optimum for sm_89)' "$O/fused-q8-256.txt"
grep -qF '2 tables (model.embed_tokens.weight + lm_head.weight)' "$O/fused-q8-256.txt"
grep -qF '(0 groups + 240 lone + 40 int4)' "$O/fused-q8-256.txt"
# Identity is READ, not gated: the position of the first divergence is the
# datum (fusedrun.rs:21-28). The prereg says where a divergence is a defect.
grep -E 'tokens identical to the dense arm|divergence at token' "$O/fused-q8-256.txt" || true
echo '== ARM f16: the same-head arm, LLVQ_EMBED=f16 (hard rule 4), 256 tokens =='
date -u
env $FLAGS LLVQ_EMBED=f16 fusedrun "$F" 256 2>&1 | tee "$O/fused-f16-256.txt" | tail -30
date -u
grep -E 'tokens identical to the dense arm|divergence at token' "$O/fused-f16-256.txt" || true
JOB

TAIL='echo "== raw kept ==" ; ls -la "$O" ; wc -l "$O"/*.txt'

case "$PHASE" in
  smoke) BODY="$HEAD
$SMOKE
$SERVED32
$TAIL" ;;
  full)  if [ "$DOOR" = 1 ]; then BODY="$HEAD
$SMOKE
$FULL
$TAIL"; else BODY="$HEAD
$FULL
$TAIL"; fi ;;
esac

if [ "$DRY" = 1 ]; then
  # What ops/run.py:1091 hands to ["bash", "-lc", ...]: printed and parsed, not run.
  JOBSCRIPT="set -euo pipefail
$PRE
$BODY"
  printf '%s\n' "$JOBSCRIPT"
  printf '%s\n' "$JOBSCRIPT" | bash -n && echo "DRY_RUN: job script parses; nothing launched" >&2
  exit 0
fi

uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor l40sx1 --timeout "$TIMEOUT" \
  --bucket "$BUCKET" --out-mount /out \
  --name "served-14b-$PHASE" \
  "$PRE" "$BODY"
