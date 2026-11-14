#!/usr/bin/env bash
# Decode-speed research phase (MODE RECHERCHE, R3), the third card run of `rowbench` (llvq-cuda/src/bin/rowbench.rs):
# the first population of WHOLE BODIES, candidates that compute y = W x for a list of sites from the Tetra words on a
# flat, wide read (docs/mesures/recherche-debit-r3-espace-2026-11-12.txt). One process after the oracle, on the SEALED
# 4B and the bare all-Tetra encode of pers-rowloop and step1-micro:
#   - every registered candidate of llvq-cuda/kernels/rowcand.cu (ROWC_ARMS: served, acc2, pword, tr32, tr32p,
#     endpass, wq, flat, tr32r, flats) on each d_in of the sealed file's Tetra records apart, on all of them, at 4E's
#     gate/up geometry and on the bare encode whole, checked on the card against the served tv_tetra48_h before it is
#     timed (the bits, or the 1e-5 bound); an arm that misses, spills or does not fit is refused, no rate;
#   - the served arm timed twice a round, first and last of each set (F1); pw_all, read-checked after every launch;
#   - on bare:all, the journal controls F3 needs: v2, v3, v2u, nc, tile;
#   - q4, q4t and q4read on the sealed int4 records; the bare read probe at its 16 configurations;
#   - the rule of the R3 note: F1 on sealed:all, F3 on bare:all, and a SCORE line a candidate and a set (F2).
# Preregistered: proofs/preregistration-recherche-debit-corps-2026-11-13.md, to stamp before this.
#
#   DRY_RUN=1 bash ops/jobs/recherche-debit-corps.sh                 parses the job script, launches nothing
#   IMAGE_SHA=<space sha> bash ops/jobs/recherche-debit-corps.sh      launches, after the operator's go
#
# The image must carry the commit with the bodies: a Space rebuild first (ops/run.py publish, no card). rowbench is
# already in both lists of ops/Dockerfile.cuda; no new binary.
# Cost, estimated (l40sx1 $1.80/h) from the last two runs of this bench (jobs 6af5ce28 and 6af5e9e7): 203 s running each,
# $0.10, oracle included. This run times fewer controls and fewer diagnostics, and adds 7 candidates whose rates are
# not known; a thread-a-row body may read 5 to 10 times slower than the served row (estimated: 69 rows a CTA on 142
# SMs), about 20 ms a set a round, under 2 s over 17 rounds. About 4 min running, $0.12. Timeout 25m: $0.75 at worst,
# the cap.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-recherche-debit-corps-2026-11-13.md
IMAGE_SHA=${IMAGE_SHA:-}
BUCKET=Pier-Jean/jobs-artifacts
O=/out/recherche-debit-corps-2026-11-13${RETRY:+-r$RETRY}
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
ROWBENCH_SHA=587ddd4857e92ea821d624f11dce4c67407999ddc80467841aa5eb10ac1049b6

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
grep -aE '^(sealed|bare|card|rowbench module|reference:|  sealed|  bare|set |  [a-z_0-9]+ +.*(BIT FOR BIT|REFUSED|PASS|diagnostic|equal|reads every word)|read probe)' "$O/bench.txt" | tail -300
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
# gate 5: pw_all read every word on every set, before timing and after every launch
if grep -aqE '^  (gu_)?pw_all +REFUSED|^  [a-z0-9:=_]+ (gu_)?pw_all REFUSED: round' "$O/bench.txt"; then
  echo "gate 5: pw_all missed its check (its line carries no rate)" >&2
fi
# gate 6: every candidate's check on the card, copied (a refusal is a result, not a failure of the job)
grep -aE '^  (gu_)?(acc2|tr32|tr32p|endpass|wq|flat|tr32r|flats) +(owes|REFUSED|tv_rowc)' "$O/bench.txt" | tee "$O/candidates.txt"
# the rule of the R3 note: F1 on sealed:all, F3 on bare:all, then F2 a candidate and a set
{ grep -a '^F1' "$O/bench.txt"; grep -a '^F3' "$O/bench.txt"; grep -a '^RULE' "$O/bench.txt"; } | tee "$O/rule.txt"
grep -a '^SCORE' "$O/bench.txt" | tee "$O/score.tsv"
{ grep -a '^PROBE' "$O/bench.txt"; grep -a '^Q4READ' "$O/bench.txt"; } | tee "$O/card.txt"
if grep -qP '^F1\tsealed:all\tPASS' "$O/rule.txt" && [ "$(grep -cP '^F3\t[a-z0-9]+\tPASS' "$O/rule.txt")" = 4 ] \
   && grep -qP '^RULE\tPASS' "$O/rule.txt"; then
  echo "rule F: PASS, the SCORE lines on sealed:all decide (prereg §5)" | tee -a "$O/rule.txt"
else
  echo "rule F: FAIL, every SCORE line is a description (prereg §5)" | tee -a "$O/rule.txt"
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
  --name "recherche-debit-corps${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
