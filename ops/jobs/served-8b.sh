#!/usr/bin/env bash
# INTEGRATOR COPY of fused8b.sh: PHASE=full timeout 60m -> 45m ($1.35), run with DOOR=1 (~22 min).
# DRAFT. The SERVED checks of the 8B paper-2 object on a card: does the Tetra
# kernel serve a Qwen3-8B file, give the dense arm's tokens, and at what speed.
# Destined for ops/jobs/served-8b.sh once the operator gives the go.
#
# Preregistered: proofs/preregistration-served-8b-${D}.md   <-- NOT WRITTEN YET.
#   Hard rule 2, and preregistration-dclm-8b-2026-09-21.md:15 ("each of its
#   jobs needs its own prereg"). This launcher refuses to run until the .ots
#   exists (DRY_RUN=1 excepted).
#
# Two phases, one script:
#
#   PHASE=smoke  on the BASE file /out/dclm-8b-2026-09-21/qwen3-8b-dclm.bin, as
#                soon as it is in the bucket. De-risks the kernel at 8B shapes
#                while the row scales train: F1e §0 found three load failures
#                at 2-3 cents each before the 8 $ census (f1e0-2026-09-10.txt
#                :54-76), and no Tetra file above 4B has ever been served
#                (tetra-8b-2026-09-06.txt:136-138). Steps: the served door
#                (57 MMLU questions through the kernel, a dump written), the
#                prefill gate at 203, the served one-arm decode at 32 tokens.
#   PHASE=full   on the FT file, after the fold. The published protocol:
#                prefill gate 203, then fusedrun 256 tokens against the dense
#                arm of the same process at the served flags (q8), then the
#                same at LLVQ_EMBED=f16, the same-head arm hard rule 4 wants
#                beside the raw ratio (METHODE.md:93-95, B2 at 8B:
#                b2-fusedrun-plages-2026-08-18.txt:27-28). 5 rounds per arm are
#                built in (fusedrun.rs:68). DOOR=1 adds the served-door MMLU
#                smoke if PHASE=smoke was skipped.
#
# Why the fused arms spell the flags out instead of LLVQ_CONFIG: the config
# puts fusedrun on a one-arm path with no dense reference (fusedrun.rs:396-464,
# configs/README.md:84-87), and the dense arm is the point. Same shape as
# ops/jobs/dclm-ft-fusedrun.sh:6-9 and port-device-fusedrun.sh:21.
#
# The config: configs/qwen3-8b-tetra-q5.json (draft: chain8b/qwen3-8b-tetra-q5.json,
# code-changes.patch adds it with its test). The image does NOT carry it: the
# Space holds commit afaed1e (Space commit cd5dbbde, 2026-09-21 01:12 UTC) and
# ops/Dockerfile.cuda:155,160 copies configs/, where only the 4B file exists.
# No rebuild is needed for it: LLVQ_CONFIG takes any path (served.rs:166-183),
# so the job writes the file's bytes into $O from this command line (recorded
# by `hf jobs inspect`) and checks their sha256 against the local file. The
# 4B config is NOT borrowed: its path would land on every dump's `# config=`
# line (mmlu.rs:724-731) and its note on the provenance line (served.rs:219-233).
#
# The tile is left unset on purpose: the served policy since 2026-09-20 reads
# 64 on sm_89 (tile.rs:239-241), and LLVQ_TILE_BLOCKS beside LLVQ_CONFIG is
# refused anyway (served.rs:93). The job fails early if the log does not say
# "tile 64 (served: measured optimum for sm_89)" (tile.rs:360-362).
#
# Cost, estimated (l40sx1, $1.80/h, ops/run.py:105), scaled from:
#   f1e-smoke 4B  (6aa40f3a) door 57 q + gate 203: 4 billed min, $0.10
#   dclm-ft-fusedrun 4B (6aaf8555): fused load 12.1 s, dense load 81.8 s
#   b2-8b (6a84b26b): two 8B fusedrun invocations, dense load 330 s, 26.5 tok/s,
#     21 billed min, $0.63 (planes14, whose load is 250 s where Tetra's is ~10x less)
#   f1e census 4B kernel arm: ~88 min / 2,280 q = 2.3 s a question; x1.9 at 8B
#   smoke: pull 2.5 + oracle 0.5 + sha 1 + door 4.8 + gate 1 + served 32 tok 1
#          = ~11 min, $0.33 [$0.27, $0.42]; timeout 40m, worst $1.20
#   full:  pull 2.5 + oracle 0.5 + sha 1 + gate 1 + q8 arm 4.2-7.3 + f16 arm 4.6-7.7
#          = ~17 min, $0.50 [$0.41, $0.60]; DOOR=1 adds ~5 min, $0.15;
#          timeout 60m, worst $1.80
#
# DRY_RUN=1 PHASE=smoke bash fused8b.sh   prints the job script, parses it, launches nothing.
set -euo pipefail

REPO=$HOME/Documents/Pro/workspace/poc/llvq
SCRATCH=$(cd "$(dirname "$0")" && pwd)
PHASE=${PHASE:-full}
D=${RUN_DATE:-2026-09-21}                          # launch date; names prereg and output dir
PREREG=${PREREG:-proofs/preregistration-served-8b-$D.md}
BUCKET=Pier-Jean/jobs-artifacts
DOOR=${DOOR:-0}
DRY=${DRY_RUN:-0}

# The committed config if the patch has landed, the scratchpad draft otherwise.
CONF_LOCAL=${CONF_LOCAL:-$REPO/configs/qwen3-8b-tetra-q5.json}
[ -f "$CONF_LOCAL" ] || CONF_LOCAL=$SCRATCH/qwen3-8b-tetra-q5.json

case "$PHASE" in
  smoke)
    OBJ_DIR=dclm-8b-2026-09-21                     # the base keeps its encoding date (census3.sh)
    NAME=qwen3-8b-dclm.bin
    LOCAL=$HOME/qwen3-8b-dclm.bin
    SUMS=$HOME/q8b-dclm-2026-09-21/files.sha256
    TIMEOUT=40m ;;
  full)
    OBJ_DIR=${FT_DIR:-}                            # e.g. dclm-8b-ft-2026-09-22, set at the fold
    NAME=qwen3-8b-dclm-ft.bin
    LOCAL=${FT_LOCAL:-$HOME/qwen3-8b-dclm-ft.bin}
    SUMS=${FT_SUMS:-}                              # empty: sha256 computed here from $LOCAL
    TIMEOUT=45m ;;
  *) echo "refused: PHASE=$PHASE, expected smoke or full" >&2; exit 1 ;;
esac
[ "$DRY" = 1 ] && OBJ_DIR=${OBJ_DIR:-dclm-8b-ft-DRYRUN}
[ -n "$OBJ_DIR" ] || { echo "refused: PHASE=full needs FT_DIR=<bucket dir of the FT file>" >&2; exit 1; }

O=/out/served-8b-$PHASE-$D
F=/out/$OBJ_DIR/$NAME
C=$O/qwen3-8b-tetra-q5.json

cd "$REPO"
CONF=$(cat "$CONF_LOCAL")
# The job writes printf '%s\n' "$CONF": the same bytes as the file only if it
# ends in exactly one newline. Checked, not assumed.
CONF_SHA=$(printf '%s\n' "$CONF" | shasum -a 256 | cut -d' ' -f1)
[ "$CONF_SHA" = "$(shasum -a 256 < "$CONF_LOCAL" | cut -d' ' -f1)" ] \
  || { echo "refused: $CONF_LOCAL does not end in exactly one newline" >&2; exit 1; }
python3 -c 'import json,sys; d=json.load(open(sys.argv[1]));
k={"layout","embed","rot_share","fuse","kv","note"}; assert set(d)==k, set(d)^k;
assert (d["layout"],d["embed"],d["rot_share"],d["fuse"],d["kv"])==("tetra48","q8","1","0","f16"), d;
assert "Qwen3-8B" in d["note"], d["note"]' "$CONF_LOCAL"

# ---- preflight on the Mac, $0 -------------------------------------------------------
if [ "$DRY" = 1 ]; then
  SHA=0000000000000000000000000000000000000000000000000000000000000000
  BYTES=4364205777                                 # prereg dclm-8b line 133, dry run only
else
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  test -f "$LOCAL" || { echo "refused: $LOCAL missing" >&2; exit 1; }
  if [ -n "$SUMS" ]; then
    SHA=$(awk -v n="$NAME" '$2 ~ ("/" n "$") {print $1}' "$SUMS")
    [ "$SHA" = "$(shasum -a 256 "$LOCAL" | cut -d' ' -f1)" ] \
      || { echo "refused: $LOCAL does not match $SUMS" >&2; exit 1; }
  else
    SHA=$(shasum -a 256 "$LOCAL" | cut -d' ' -f1)
  fi
  [ ${#SHA} -eq 64 ] || { echo "refused: no sha256 for $NAME" >&2; exit 1; }
  BYTES=$(stat -f %z "$LOCAL")
  REMOTE=$(hf buckets ls "hf://buckets/$BUCKET/$OBJ_DIR/" 2>/dev/null | awk -v n="$NAME" '$NF ~ (n "$") {print $1}' || true)
  [ "$REMOTE" = "$BYTES" ] || { echo "refused: bucket copy is ${REMOTE:-absent} B, local $BYTES B" >&2; exit 1; }
  if hf buckets ls "hf://buckets/$BUCKET/served-8b-$PHASE-$D/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: served-8b-$PHASE-$D/ already exists; pick another RUN_DATE" >&2; exit 1
  fi
  echo "object: $F, $BYTES B, sha256 $SHA; config $CONF_LOCAL sha256 $CONF_SHA"
fi

PRE=$(printf 'O=%q\nF=%q\nC=%q\nSHA=%q\nBYTES=%q\nCONF=%q\nCONF_SHA=%q' \
  "$O" "$F" "$C" "$SHA" "$BYTES" "$CONF" "$CONF_SHA")

# Common head: the refusal of an inherited LLVQ_*, the card, oracle (hard rule
# 10, the precedent of every 8B card job: vod-8b.sh:24, b2, vague2), the object
# by bytes and sha256, and the config written from this command line.
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
echo '== prefill gate, 203 tokens: 50 chunks of four and one of three (configs/README.md:79-83) =='
date -u
LLVQ_CONFIG="$C" LLVQ_PREFILL_TOKENS=203 fusedrun "$F" 2>&1 | tee "$O/prefill-203.txt" | tail -14
date -u
# The served tile, or nothing further runs: every number below is read at 64.
grep -qF 'tile 64 (served: measured optimum for sm_89)' "$O/prefill-203.txt"
grep -qF '36 int4' "$O/prefill-203.txt"
JOB

IFS= read -r -d '' SMOKE <<'JOB' || true
echo '== the served door: 57 questions through the kernel, a dump written (configs/README.md:76-78) =='
date -u
LLVQ_CONFIG="$C" LLVQ_MMLU_ALLOC=flat LLVQ_MMLU_DUMP="$O/mmlu-8b-served-door.csv" \
  mmlu "$F" cuda 1 2>&1 | tee "$O/mmlu-door.txt" | tail -12
date -u
grep -qxF '# arithmetic=served kernel' "$O/mmlu-8b-served-door.csv"
grep -qxF "# config=$C" "$O/mmlu-8b-served-door.csv"
grep -qxF '# layout=tetra48' "$O/mmlu-8b-served-door.csv"
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
  # What ops/run.py:1088 hands to ["bash", "-lc", ...] (ops/run.py:1098): printed and parsed, not run.
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
  --name "served-8b-$PHASE" \
  "$PRE" "$BODY"
