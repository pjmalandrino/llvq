#!/usr/bin/env bash
# Decode-speed research phase (MODE RECHERCHE, step R2), stage 2 of the verifier: one card run of `rowbench`
# (llvq-cuda/src/bin/rowbench.rs) on the SEALED 4B and, as the control, the bare all-Tetra encode of pers-rowloop and
# step1-micro. One process after the oracle:
#   - every registered row program of llvq-cuda/kernels/rowcand.cu (today: served, served_np, acc2, pword) on each
#     d_in of the sealed file's Tetra records apart, on all of them, and at 4E's gate/up geometry (tv_rowc_gu_*),
#     checked on the card against the served tv_tetra48_h before it is timed; an arm that fails is refused, no rate;
#   - the journals' row-loop arms (v2, v3, ilp2, v2u, l1w, bcast, xbcast, v2_s, v2_o2, nc, nc256, cpa, tile, bal) on
#     the same sets, as controls;
#   - K11's int4 arms (q4hh, q4, q4nc, q4t, q4read) on the sealed file's own int4 records, each d_in apart and all;
#   - the bare read probe: a 2 GiB buffer of hashed words, read and summed by three kernels at 16 configurations, the
#     sum checked every round against the host's;
#   - the fidelity rule of the prereg (§6): the served row on the bare encode inside the journals' window, the known
#     arms at their journals' speeds.
# Preregistered: proofs/preregistration-recherche-debit-banc-2026-11-12.md, to stamp before this.
#
#   DRY_RUN=1 bash ops/jobs/recherche-debit-banc.sh                 parses the job script, launches nothing
#   IMAGE_SHA=<space sha> bash ops/jobs/recherche-debit-banc.sh      launches, after the operator's go
#
# The image must carry the commit with rowbench (a new binary in BOTH lists of ops/Dockerfile.cuda): a Space rebuild
# first (ops/run.py publish, no card).
# Cost, estimated (l40sx1 $1.80/h) from the two jobs of the same bench family in docs/data/jobs.csv: pers-rowloop
# (6aec7403, 9 min billed, 497 s running, $0.25) and step1-micro (6aef03fa, 8 min billed, $0.25), each an oracle and one
# bench process over the same 1.77 GB file. This one reads two files (1.42 + 1.77 GB) and compiles the row-loop module
# plus the candidates twice (once for the dump); its rounds take about 0.3 s each (computed from the bytes at the
# journals' rates). About 11 min billed, $0.33. Timeout 25m: $0.75 at worst, the cap.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-recherche-debit-banc-2026-11-12.md
IMAGE_SHA=${IMAGE_SHA:-}
BUCKET=Pier-Jean/jobs-artifacts
O=/out/recherche-debit-banc-2026-11-12${RETRY:+-r$RETRY}
TIMEOUT=25m
ROUNDS=15
DRY=${DRY_RUN:-0}

SEALED="sealed-4b-2026-09-23/qwen3-4b-sealed.bin:1418224685"
SEALED_SHA=886391a8c03f66dc269cc65c3598c6627dbdcd259180aff36604ef10d37371b8
TETRA="tetra-4b-2026-09-06/qwen3-4b-tetra.bin:1770529149"
TETRA_SHA8=0adb7cfd
# The served unit's digest (llvq-cuda/src/pers.rs::the_served_unit_is_the_published_one): the reference rows are the
# served tv_tetra48_h and tv_q4_hh, compiled from this unit.
SERVED_SHA=6193094310bc18acdb0a0c554807ff45f6df09c1f6ab49e0577032874f107812
# The bench module at the card's tile define, as the Mac builds it (rowcand.rs::tests::dump_rowcand_module).
ROWBENCH_SHA=5dab2bcac212cd1158809f4aecf7977bc7607f9db267ab905297ce486b8c2aaf

if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  for row in "$SEALED" "$TETRA"; do
    path=${row%%:*}; bytes=${row##*:}
    REMOTE=$(hf buckets ls "hf://buckets/$BUCKET/${path%/*}/" 2>/dev/null \
      | awk -v n="/${path##*/}" 'substr($NF, length($NF) - length(n) + 1) == n {print $1}' || true)
    [ "$REMOTE" = "$bytes" ] || { echo "refused: $path is ${REMOTE:-absent} B in the bucket, expected $bytes" >&2; exit 1; }
  done
  if hf buckets ls "hf://buckets/$BUCKET/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already exists" >&2; exit 1
  fi
  IMAGE_SHA=${IMAGE_SHA:?IMAGE_SHA=<the Space sha of the rebuild that carries this branch>}
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    'from huggingface_hub import HfApi; print(HfApi().space_info("Pier-Jean/llvq-runner-cuda").sha)')
  [ "$NOW" = "$IMAGE_SHA" ] || { echo "refused: the Space is at $NOW, expected $IMAGE_SHA" >&2; exit 1; }
  echo "two bucket copies verified by size; image $NOW"
fi

PRE=$(printf 'O=%q\nS=%q\nF=%q\nSB=%q\nFB=%q\nSSHA=%q\nFSHA8=%q\nSERVED_SHA=%q\nROWBENCH_SHA=%q\nROUNDS=%q' \
  "$O" "/out/${SEALED%%:*}" "/out/${TETRA%%:*}" "${SEALED##*:}" "${TETRA##*:}" "$SEALED_SHA" "$TETRA_SHA8" \
  "$SERVED_SHA" "$ROWBENCH_SHA" "$ROUNDS")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O" "$O/cubin"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,clocks.max.sm,clocks.max.mem --format=csv,noheader | tee "$O/gpu.txt"
test "$(stat -c %s "$S")" = "$SB"
test "$(stat -c %s "$F")" = "$FB"
sha256sum "$S" "$F" | tee "$O/files.sha256"
test "$(sha256sum "$S" | cut -d' ' -f1)" = "$SSHA"
test "$(sha256sum "$F" | cut -c1-8)" = "$FSHA8"

echo '== gate 1: oracle =='
timeout 5m oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle.txt" | tail -3
grep -q 'MATCH' "$O/oracle.txt"

echo '== process 1: rowbench on the sealed 4B and the bare encode =='
date -u
rc=0
timeout 18m rowbench "$S" "$F" --rounds "$ROUNDS" --dump "$O/cubin" > "$O/bench.txt" 2>&1 || rc=$?
date -u
echo "rowbench exit $rc"
grep -aE '^(sealed|bare|card|rowbench module|reference:|  sealed|  bare|set |  [a-z_0-9]+ +.*(BIT FOR BIT|REFUSED|PASS|diagnostic|equal)|read probe|FIDELITY|ORDER)' "$O/bench.txt" | tail -200
grep -a '^TSV' "$O/bench.txt" > "$O/summary.tsv" || true
grep -a '^ROUND' "$O/bench.txt" > "$O/rounds.tsv" || true
[ "$rc" = 0 ]
# gate 2: the references are the served unit, the module the one the Mac built
grep -q "reference: the served unit, 101625 bytes, sha256 $SERVED_SHA" "$O/bench.txt" \
  || { echo "gate 2: the reference is not the served unit $SERVED_SHA" >&2; exit 1; }
grep -q "rowbench module: [0-9]* bytes, sha256 $ROWBENCH_SHA" "$O/bench.txt" \
  || { echo "gate 2: the bench module is not the Mac's $ROWBENCH_SHA" >&2; exit 1; }
# gate 3: the served arm is the served row on every set, rows and gate/up
for s in sealed:all sealed:gu4e bare:all; do
  awk -v s="set $s:" 'index($0, s) == 1 {p = 1; next} /^set / {p = 0} p' "$O/bench.txt" \
    | grep -qE '^  (gu_)?served +[0-9]+ rows identical BIT FOR BIT to tv_tetra48_h' \
    || { echo "gate 3: the served arm is not bit for bit on $s" >&2; exit 1; }
done
# gate 4: every probe configuration that was timed summed what the host summed (a miss is a refusal, printed)
if grep -q '^  probe .* REFUSED: round' "$O/bench.txt"; then echo "gate 4: a probe sum missed" >&2; fi
# rule D, the fidelity of the stand-in: printed by the bench, copied here
{ grep -a '^FIDELITY' "$O/bench.txt"; grep -a '^ORDER' "$O/bench.txt"; } | tee "$O/fidelity.txt"
if grep -qP '^FIDELITY\tserved\tPASS' "$O/fidelity.txt" && grep -qP '^ORDER\tall\tPASS' "$O/fidelity.txt"; then
  echo "rule D: PASS, the stand-in reproduces the journals" | tee -a "$O/fidelity.txt"
else
  echo "rule D: FAIL, no candidate rate of this run counts (prereg §6)" | tee -a "$O/fidelity.txt"
fi
ls -l "$O" "$O/cubin"
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
  --name "recherche-debit-banc${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
