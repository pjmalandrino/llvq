#!/usr/bin/env bash
# Decode-speed research phase (CERVEAU O11), step R6c: the LOAD PROBES on one L40S, nothing timed, nothing launched.
# The six whole research arms are each refused for the same 8 B of local memory on 4E's timed token symbol, whatever
# their body sheds (docs/mesures/recherche-debit-charge2-2026-11-14.txt). The probes ask where the 8 B come from:
# E1's v9 with pers_4e_rc.cuh inserted and the endpass body on ONE call site, or on none (pers.rs::P4E_RC_PROBES):
#   v9pe0 no site (the text alone), v9peq q/k, v9peg gate/up, v9ped the Tetra down, v9peo the Tetra o.
#   - oracle on Qwen3-0.6B (hard rule 10);
#   - `LLVQ_PERS_4E=check:<probes> fusedrun <sealed> 2`: no model loaded, the arms one after the other, each line
#     printed as it is known (persist4e_cuda::check_e1_arms, as job 6af82af3 ran it).
# v9 (LOADS) and v9ep (REFUSED, 8 B) are not rerun: their texts are unchanged and were read by 6af82af3 under this
# driver. The check's own change since then is the parser alone.
# Preregistered: proofs/preregistration-recherche-debit-sondes-2026-11-14.md, to stamp before this.
#
#   DRY_RUN=1 bash ops/jobs/recherche-debit-sondes.sh               parses the job script, launches nothing
#   IMAGE_SHA=<space sha> bash ops/jobs/recherche-debit-sondes.sh    launches, after the operator's go
#
# Cost, estimated (l40sx1 $1.80/h) from job 6af82af3: an endpass arm took 103 to 104 s in the check, the oracle and
# the start about 40 s. Five probes: about 9.5 min, $0.29. Timeout 12m: $0.36 at worst.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-recherche-debit-sondes-2026-11-14.md
OUT_NAME=recherche-debit-sondes-2026-11-14
IMAGE_SHA=${IMAGE_SHA:-}
BUCKET=Pier-Jean/jobs-artifacts
O=/out/$OUT_NAME${RETRY:+-r$RETRY}
DRY=${DRY_RUN:-0}
TIMEOUT=${TIMEOUT:-12m}
ARMS=v9pe0,v9peg,v9peq,v9ped,v9peo
# The modules' sha256 at the card's tile define, as pers.rs builds them on the Mac (dump_tree_modules).
SHAS="v9pe0:9db6055b82a1a946d83a87631a47c1c65b54f6e59ab608b80dc786c15a75f7e4 v9peq:c9bb1820159a926a727e77849d08a39d1ddfa1c5d3827dea860b50e4e808de02 v9peg:e7ad915e4231de1eb1762da8e839cc9374cfefcc3f4dfe0de2cee9579d7f7d1c v9ped:89f470f26e2181218a99983f265ce046d30d397a57edc1a35c8f3d538481121c v9peo:154f41ac69ba9058eb5293ba77eca53372e229ea7f860c27d6e73bc40f7e2577"
SEALED="sealed-4b-2026-09-23/qwen3-4b-sealed.bin:1418224685"

if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  case "$SHAS" in *TO_FILL*) echo "refused: a probe's module sha256 is not filled in" >&2; exit 1 ;; esac
  if hf buckets ls "hf://buckets/$BUCKET/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already exists" >&2; exit 1
  fi
  IMAGE_SHA=${IMAGE_SHA:?IMAGE_SHA=<the Space sha of the rebuild that carries this branch>}
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    'from huggingface_hub import HfApi; print(HfApi().space_info("Pier-Jean/llvq-runner-cuda").sha)')
  [ "$NOW" = "$IMAGE_SHA" ] || { echo "refused: the Space is at $NOW, expected $IMAGE_SHA" >&2; exit 1; }
  echo "image $NOW"
fi

PRE=$(printf 'O=%q\nS=%q\nARMS=%q\nSHAS=%q\nexport HF_HOME=/tmp/hf' "$O" "/out/${SEALED%%:*}" "$ARMS" "$SHAS")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,clocks.max.sm --format=csv,noheader | tee "$O/gpu.txt"
echo '== gate 1: oracle =='
timeout 4m oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle.txt" | tail -2
grep -q MATCH "$O/oracle.txt"
echo '== the load probes =='
date -u
rc=0
env LLVQ_FUSED_LAYOUT=tetra48 LLVQ_EMBED=q4 LLVQ_ROT_SHARE=1 LLVQ_FUSE=0 LLVQ_PERS_4E="check:$ARMS" \
  timeout 10m fusedrun "$S" 2 > "$O/check.txt" 2>&1 || rc=$?
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
  --name "recherche-debit-sondes${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
