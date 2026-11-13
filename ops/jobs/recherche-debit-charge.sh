#!/usr/bin/env bash
# Decode-speed research phase (CERVEAU O11), step R6b: the LOAD CHECK of the research arms on one L40S, nothing timed.
# The first token run (docs/mesures/recherche-debit-jeton-2026-11-13.txt) was void: under the driver's ptxas
# (580.178.04) the timed token symbol of v9ep and of v9fl held 8 B of local memory, and the runtime refuses that. This
# job asks the card which arms load, before any token measurement:
#   - oracle on Qwen3-0.6B (hard rule 10);
#   - `LLVQ_PERS_4E=check:<arms> fusedrun <sealed> 2`: no model loaded; each arm's module compiled as install_e1
#     compiles it (pers.rs::p4e_e1_unit_text at the card's tile define, NVRTC, the driver's JIT), one thread an arm;
#     for each, the token symbol of the three RoPE forms at 512 threads and the stamp and logits symbols of form 1:
#     registers, local bytes, shared bytes at the sealed 4B's shape (85,952 B), CTAs a SM; the rc layout at the card's
#     grid; and `CHECK ARM <arm> LOADS` or `CHECK ARM <arm> REFUSED: <why>` (the runtime's rule, on form 1's token
#     symbol, the form the card has picked in every 4B run).
# The arms: v9 (E1's reference: must load, 0 B, as in three jobs), v9ep and v9fl (the refused pair: must read 8 B
# again, or the check does not reproduce the runtime), and the lighter variants v9ep2, v9ep3, v9fl2, v9fl3
# (kernels/pers_4e_rc2.cuh).
# Preregistered: proofs/preregistration-recherche-debit-charge-jeton2-2026-11-13.md, to stamp before this.
#
#   DRY_RUN=1 bash ops/jobs/recherche-debit-charge.sh               parses the job script, launches nothing
#   IMAGE_SHA=<space sha> bash ops/jobs/recherche-debit-charge.sh    launches, after the operator's go
#
# The image must carry the commit with the variants and the check: a Space rebuild first (ops/run.py publish, no
# card). fusedrun is already in both lists of ops/Dockerfile.cuda; the kernels are text in the binary; nothing new.
# Cost, estimated (l40sx1 $1.80/h) from the void run (job 6af70da3): its oracle and the start of its first process took
# about 2 min; its main process spent 339 s before the refusal, for the model (about 70 s, E1's one-arm process) and
# three modules compiled one after the other (about 90 s each). Here no model and seven modules on parallel threads:
# about 2 to 4 min for the check. About 6 min, $0.18. Timeout 7m: $0.21 at worst.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-recherche-debit-charge-jeton2-2026-11-13.md
OUT_NAME=recherche-debit-charge-2026-11-13
IMAGE_SHA=${IMAGE_SHA:-}
BUCKET=Pier-Jean/jobs-artifacts
O=/out/$OUT_NAME${RETRY:+-r$RETRY}
DRY=${DRY_RUN:-0}
TIMEOUT=${TIMEOUT:-7m}
ARMS=v9,v9ep,v9fl,v9ep2,v9ep3,v9fl2,v9fl3
# The modules' sha256 at the card's tile define, as pers.rs builds them on the Mac (dump_tree_modules,
# the_rc_modules_keep_their_bytes). v9, v9ep and v9fl are the jeton run's; the four variants' are filled in when the
# Mac gate has run (not yet: see the prereg's list of what has not run).
SHA_V9=d55a097b146e8b962829045dfea6975470b408b942c1e0c22c038ad4d9d00860
SHA_V9EP=f4166ccf9d857b30cae8f7691e9e4f83c2726218cb48c29dc512cb42b1a668dd
SHA_V9FL=9925b4ad769c791d2372d52a14b68e650a549ce15edd05eecae0dececfa4f03d
SHA_V9EP2=TO_FILL
SHA_V9EP3=TO_FILL
SHA_V9FL2=TO_FILL
SHA_V9FL3=TO_FILL
SEALED="sealed-4b-2026-09-23/qwen3-4b-sealed.bin:1418224685"

if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  for v in "$SHA_V9EP2" "$SHA_V9EP3" "$SHA_V9FL2" "$SHA_V9FL3"; do
    [ "$v" != TO_FILL ] || { echo "refused: a variant's module sha256 is not filled in" >&2; exit 1; }
  done
  if hf buckets ls "hf://buckets/$BUCKET/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already exists" >&2; exit 1
  fi
  IMAGE_SHA=${IMAGE_SHA:?IMAGE_SHA=<the Space sha of the rebuild that carries this branch>}
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    'from huggingface_hub import HfApi; print(HfApi().space_info("Pier-Jean/llvq-runner-cuda").sha)')
  [ "$NOW" = "$IMAGE_SHA" ] || { echo "refused: the Space is at $NOW, expected $IMAGE_SHA" >&2; exit 1; }
  echo "image $NOW"
fi

PRE=$(printf 'O=%q\nS=%q\nARMS=%q\nSHAS=%q\nexport HF_HOME=/tmp/hf' "$O" "/out/${SEALED%%:*}" "$ARMS" \
  "v9:$SHA_V9 v9ep:$SHA_V9EP v9fl:$SHA_V9FL v9ep2:$SHA_V9EP2 v9ep3:$SHA_V9EP3 v9fl2:$SHA_V9FL2 v9fl3:$SHA_V9FL3")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,clocks.max.sm --format=csv,noheader | tee "$O/gpu.txt"
echo '== gate 1: oracle =='
timeout 4m oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle.txt" | tail -2
grep -q MATCH "$O/oracle.txt"
echo '== the load check =='
date -u
rc=0
env LLVQ_FUSED_LAYOUT=tetra48 LLVQ_EMBED=q4 LLVQ_ROT_SHARE=1 LLVQ_FUSE=0 LLVQ_PERS_4E="check:$ARMS" \
  timeout 5m fusedrun "$S" 2 > "$O/check.txt" 2>&1 || rc=$?
date -u
echo "fusedrun check exit $rc" | tee -a "$O/check.txt"
grep -aE '^4E check|^CHECK' "$O/check.txt" | tee "$O/summary.txt"
# gate 2: every module is the Mac's
for pair in $SHAS; do
  arm=${pair%%:*}; sha=${pair#*:}
  grep -aq "^4E check $arm: module [0-9]* bytes, sha256 $sha\$" "$O/check.txt" \
    && echo "gate 2: $arm module sha256 $sha" | tee -a "$O/summary.txt" \
    || echo "VOID: gate 2, $arm's module is not the Mac's $sha" | tee -a "$O/summary.txt"
done
# gate 3: the check reproduces the runtime: v9 loads, v9ep and v9fl are refused for local memory
grep -q '^CHECK ARM v9 LOADS' "$O/check.txt" && grep -q '^CHECK ARM v9ep REFUSED: .*B local' "$O/check.txt" \
  && grep -q '^CHECK ARM v9fl REFUSED: .*B local' "$O/check.txt" \
  && echo "gate 3: v9 loads, v9ep and v9fl refused as in the jeton run" | tee -a "$O/summary.txt" \
  || echo "VOID: gate 3, the check does not reproduce the jeton run's loads" | tee -a "$O/summary.txt"
grep -a '^CHECK ARM' "$O/check.txt" | sed 's/ (.*s)$//' > "$O/verdicts.txt" || true
cat "$O/verdicts.txt"
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
  --name "recherche-debit-charge${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
