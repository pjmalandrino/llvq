#!/usr/bin/env bash
# Decode-speed research phase (CERVEAU O11), step R6d: R6c found that the gate/up call site alone brings the 8 B of
# local memory (docs/mesures/recherche-debit-sondes-2026-11-14.txt). Three arms leave that site out or rewrite it:
# v9fn (flat on q/k, down, o; v9's gate/up), v9fg (flat, gate/up in two calls), v9eg (endpass, gate/up in two calls).
# The token measurement of the jeton prereg (proofs/preregistration-recherche-debit-jeton-2026-11-13.md, its score and
# its rule F1 to F2), in jeton2's shape (ops/jobs/recherche-debit-jeton2.sh):
#   (1) the LOAD CHECK first (`LLVQ_PERS_4E=check:v9,v9fg,v9fn,v9eg`, no model, 9 min); the four digests checked
#       against the Mac's; then the selection: the flat family's arm is the first of v9fg, v9fn that LOADS, the
#       endpass family's is v9eg if it LOADS; if none loads, the job STOPS there (exit 0);
#   (2) the served door, as before (control 2);
#   (3) `LLVQ_PERS_4E=256:v9,<selected>,v9`: at a 5-token prompt and at 8,192, 2 + 5 rounds, each v9, the arms, v9
#       again, then eager; F1 a context, a SCORE line an arm and a context (F2); tokens and logits against v9;
#   (4) one-arm process of the selected flat-family arm, if any (peak VRAM, logits fingerprints).
# The candidates are fixed here, not by the launcher. Output recherche-debit-jeton3-2026-11-14.
# Preregistered: proofs/preregistration-recherche-debit-jeton3-2026-11-14.md, to stamp before this.
#
#   DRY_RUN=1 bash ops/jobs/recherche-debit-jeton3.sh                     parses, launches nothing
#   IMAGE_SHA=<space sha> bash ops/jobs/recherche-debit-jeton3.sh         launches, after the operator's go
#
# Cost (l40sx1 $1.80/h), estimated from measured times. The void jeton run (job 6af70da3, 923 s running) held what
# this job holds beside the check: the oracle, the door, a main process loading three modules (v9 and two arms,
# 339 s with the model) and timing v9 and its F1 control at two contexts, and two one-arm processes. The check, from
# the charge2 and sondes runs (jobs 6af82af3, 6af837d1): v9 101.6 s, a flat arm 125.9 to 127.2 s, an endpass arm
# 102.4 to 104.0 s, one after the other: about 102 + 127 + 127 + 104 = 460 s. Two arms timed add about 4 min of
# decode (jeton2's estimate); one one-arm process fewer saves about 2 min. 923 + 460 + 240 - 120 = about 1,500 s:
# about 25 min, $0.75. Timeout 32m: $0.96 at worst.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-recherche-debit-jeton3-2026-11-14.md
OUT_NAME=recherche-debit-jeton3-2026-11-14
IMAGE_SHA=${IMAGE_SHA:-}
# the candidates, fixed by the prereg: the flat family in its order, then the endpass family
ARMS=v9fg,v9fn,v9eg
BUCKET=Pier-Jean/jobs-artifacts
O=/out/$OUT_NAME${RETRY:+-r$RETRY}
DRY=${DRY_RUN:-0}
P4E_V9_SHA=1b422b62328be6752df4add43280565fd2a49cfa20ee26953b301abe7a15c530
TIMEOUT=${TIMEOUT:-32m}
NTOK=256
# the modules' sha256 at the card's tile define (pers.rs::the_rc_modules_keep_their_bytes, dump_tree_modules)
SHAS="v9:d55a097b146e8b962829045dfea6975470b408b942c1e0c22c038ad4d9d00860 v9fg:36b617b0811d24cb8aa475fa9c3f0cd49ced1138c918a0fbdd7556ceac1c67bb v9fn:ce902a755173d05c4d50353e68d7bf62a326e13566fd4f7b84bf97e9933b6f4f v9eg:1971b46969e4579c64327852a47c49887bd2acb0ffc5be802d2bed614117c29d"

SEALED="sealed-4b-2026-09-23/qwen3-4b-sealed.bin:1418224685"
SEALED_SHA=886391a8c03f66dc269cc65c3598c6627dbdcd259180aff36604ef10d37371b8
C=configs/qwen3-4b-tetra-e4-4e-v9.json
grep -q '^  "rows": "v9",$' "$C" || { echo "refused: $C does not name rows v9" >&2; exit 1; }
CFG_SHA=$(shasum -a 256 < "$C" | cut -d' ' -f1)
CFG_B64=$(base64 < "$C" | tr -d '\n')

if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  for a in ${ARMS//,/ }; do
    case " $SHAS " in *" $a:TO_FILL "*) echo "refused: $a's module sha256 is not filled in" >&2; exit 1 ;; esac
  done
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
  echo "bucket copy verified by size; image $NOW; arms $ARMS"
fi

PRE=$(printf 'O=%q\nSEALED=%q\nSEALED_SHA=%q\nCFG_SHA=%q\nCFG_B64=%q\nP4E_V9_SHA=%q\nSHAS=%q\nARMS=%q\nNTOK=%q\nexport HF_HOME=/tmp/hf' \
  "$O" "$SEALED" "$SEALED_SHA" "$CFG_SHA" "$CFG_B64" "$P4E_V9_SHA" "$SHAS" "$ARMS" "$NTOK")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,clocks.max.sm --format=csv,noheader | tee "$O/gpu.txt"
echo '== control 1: oracle =='
timeout 4m oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle.txt" | tail -2
grep -q MATCH "$O/oracle.txt"
echo '== control 1: the object and its config =='
S=/out/${SEALED%%:*}; SB=${SEALED##*:}
test "$(stat -c %s "$S")" = "$SB"
sha256sum "$S" | tee "$O/files.sha256"
test "$(sha256sum "$S" | cut -d' ' -f1)" = "$SEALED_SHA"
PT=$O/qwen3-4b-tetra-e4-4e-v9.json
printf '%s' "$CFG_B64" | base64 -d > "$PT"
test "$(sha256sum "$PT" | cut -d' ' -f1)" = "$CFG_SHA"
FLAGS=(LLVQ_FUSED_LAYOUT=tetra48 LLVQ_EMBED=q4 LLVQ_ROT_SHARE=1 LLVQ_FUSE=0)
echo "== (1) the load check: v9,$ARMS =="
date -u
rc=0
env "${FLAGS[@]}" LLVQ_PERS_4E="check:v9,$ARMS" timeout 9m fusedrun "$S" 2 > "$O/check.txt" 2>&1 || rc=$?
echo "fusedrun check exit $rc" | tee -a "$O/check.txt"
grep -aE '^4E check|^CHECK' "$O/check.txt" | tee "$O/check-summary.txt"
for pair in $SHAS; do
  arm=${pair%%:*}; sha=${pair#*:}
  case ",v9,$ARMS," in *",$arm,"*) ;; *) continue ;; esac
  grep -aq "^4E check $arm: module [0-9]* bytes, sha256 $sha\$" "$O/check.txt" \
    || { echo "STOP: control 3, $arm's module is not the Mac's $sha" | tee -a "$O/check-summary.txt"; exit 1; }
done
grep -q '^CHECK ARM v9 LOADS' "$O/check.txt" || { echo "STOP: v9 does not load" | tee -a "$O/check-summary.txt"; exit 1; }
# the selection (prereg): the flat family's arm is the first of v9fg, v9fn that loads; the endpass family's is v9eg
# if it loads; an arm not selected is not timed
RUN=""; FL=""
for a in v9fg v9fn; do
  if grep -q "^CHECK ARM $a LOADS" "$O/check.txt"; then
    FL=$a; RUN="$RUN,$a"; break
  fi
  echo "arm $a does not load" | tee -a "$O/check-summary.txt"
done
if grep -q '^CHECK ARM v9eg LOADS' "$O/check.txt"; then
  RUN="$RUN,v9eg"
else
  echo "arm v9eg does not load" | tee -a "$O/check-summary.txt"
fi
[ -n "$RUN" ] || { echo "STOP: none of $ARMS loads; nothing to time (prereg: the job ends here)" | tee -a "$O/check-summary.txt"; exit 0; }
echo "arms timed: ${RUN#,}" | tee -a "$O/check-summary.txt"
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
echo "== (2) the served v9 config at the door, 5-token prompt =="
date -u
vram served-v9 env LLVQ_CONFIG="$PT" LLVQ_CONTEXTS=5 timeout 5m fusedrun "$S" 64 > "$O/served-v9.txt" 2>&1
echo "fusedrun served v9 exit $?" >> "$O/served-v9.txt"
echo "== (3) v9$RUN, v9 again, interleaved, at 5 and 8,192 =="
date -u
vram jeton-all env "${FLAGS[@]}" LLVQ_PERS_4E="$NTOK:v9$RUN,v9" timeout 14m fusedrun "$S" "$NTOK" > "$O/jeton-all.txt" 2>&1
echo "fusedrun jeton all exit $?" >> "$O/jeton-all.txt"
if [ -n "$FL" ]; then
  echo "== (4) $FL alone, under the sampler =="
  date -u
  vram "jeton-$FL" env "${FLAGS[@]}" LLVQ_PERS_4E="$NTOK:$FL" timeout 4m fusedrun "$S" "$NTOK" > "$O/jeton-$FL.txt" 2>&1
  echo "fusedrun jeton $FL exit $?" >> "$O/jeton-$FL.txt"
fi
date -u
set -e
peak_gb() { awk -v t="$1" '$1 == "vram" && $2 == t { printf "%.3f", $7 * 1048576 / 1e9 }' "$O/vram.txt"; }
{
  echo "peak VRAM by nvidia-smi (MiB x 1,048,576 / 1e9 GB):"
  for t in served-v9 jeton-all ${FL:+jeton-$FL}; do echo "  $t $(peak_gb "$t") GB"; done
  echo "the served door (v9 config, 5-token prompt):"
  grep -E '^ +5 +[0-9.]+ \[|4E tokens equal|exit ' "$O/served-v9.txt" | sed 's/^/  /'
  for f in jeton-all ${FL:+jeton-$FL}; do
    echo "== $f =="
    grep -aE '^4E: persistent module|^4E: pers_4e_(tokens|stamps|logits)_|^4E e1|^4E rc|^4E: RoPE|loaded and checked|^  context|tokens equal|owes a bound|DIVERGES|not run|served door|tok/s|round by round|^  F1 |^  SCORE |^  logits|stamped token|exit |^Error|^error' "$O/$f.txt" || true
  done
  echo "== phase stamps (jeton-all) =="
  awk '/phase stamps, 4E/ {p = 1} /stamped token/ {print; p = 0} p' "$O/jeton-all.txt" || true
} > "$O/summary.txt" 2>&1
cat "$O/summary.txt"
grep -aE '^  F1 |^  SCORE ' "$O/jeton-all.txt" > "$O/rule.txt" || true
grep -qF "4E: persistent module 252757 bytes, sha256 $P4E_V9_SHA, row form 7," "$O/served-v9.txt" \
  && echo "control 2: the served door ran the served 4E v9 module" | tee -a "$O/summary.txt" \
  || echo "VOID: control 2, the served door's 4E module is not the served v9 module" | tee -a "$O/summary.txt"
n_arms=$(printf '%s' "${RUN#,}" | tr ',' '\n' | grep -c .)
[ "$(grep -c '^  F1 ' "$O/rule.txt")" = 2 ] && [ "$(grep -c '^  SCORE ' "$O/rule.txt")" = $((2 * n_arms)) ] \
  && echo "control 4: F1 and the $((2 * n_arms)) SCORE lines printed" | tee -a "$O/summary.txt" \
  || echo "VOID: control 4, the rule's lines are missing (see jeton-all.txt)" | tee -a "$O/summary.txt"
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
  --flavor l40sx1 --timeout "$TIMEOUT" \
  --bucket "$BUCKET" --out-mount /out \
  --name "recherche-debit-jeton3${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
