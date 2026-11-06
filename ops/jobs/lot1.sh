#!/usr/bin/env bash
# Lot 1: every card measurement of 2026-11-04 under one operator go, on one image, in three jobs.
#
#   bash ops/jobs/lot1.sh          checks, every script's DRY_RUN, the 4B lot job composed and parsed, the cost
#                                  table; launches nothing
#   GO=1 bash ops/jobs/lot1.sh     the same, then every script's launch checks, then the three jobs
#
# The jobs, all on the Space image IMAGE_SHA below, built from BASE (integration/lot1):
#   lot1-4b   one card, the sealed 4B, five preregs' worth of steps in four parts, in this order:
#             gestures (gesture-ab.sh), 32B walls (bigmodel-identity.sh), attention r2 (attn-bench-r2.sh), and
#             the attention T8 control (attn-t8.sh), which runs only if r2 passed its gates and then reads its own
#             condition from r2's sweep logs. Each part is the job script its own launcher writes (EMIT), run
#             as its own bash process under its own ceiling, so a failed or hung part voids that part only.
#   gsm8k-8b-grouped, gsm8k-14b-grouped   ops/jobs/gsm8k-grouped-size.sh, launched by the script itself.
# The three jobs run in parallel; nothing in one reads another.
#
# Nothing here commits, stamps or publishes. The operator stamps the preregs, commits, rebuilds the Space and
# gives the go; this script refuses GO=1 until each of those is visible.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

IMAGE_SHA=142216fc72326a98b94316a7f6cd2e989f0efc50
BASE=02ec809323eb4a458725157573e2ac22905d002f
SPACE=Pier-Jean/llvq-runner-cuda
BUCKET=Pier-Jean/jobs-artifacts
USD_H=1.80
GO=${GO:-0}
LOT_TIMEOUT=180m
L=/out/lot1-4b-2026-11-04${RETRY:+-r$RETRY}
# In-job ceilings of the four parts, each at most its prereg's timeout less the container start.
CEIL_GESTURES=18m
CEIL_BIGMODEL=18m
CEIL_ATTN_R2=31m
CEIL_ATTN_T8=160m
TMPROOT=${TMPDIR:-/tmp}
WORK=$(mktemp -d "${TMPROOT%/}/lot1.XXXXXX")
FAIL=0
note() { printf '%s\n' "$*"; }
miss() { note "  MISSING: $*"; FAIL=1; }

git rev-parse --verify --quiet "$BASE^{commit}" >/dev/null \
  || { echo "refused: the base commit $BASE is not in this repository" >&2; exit 1; }

note "== 1. preregistrations, stamps and deviation files =="
for p in gestures-2026-11-04 attn-bench-r2-2026-11-04 bigmodel-2-2026-11-04 attn-t8-2026-11-04 \
         gsm8k-grouped-size-2026-11-04; do
  f=proofs/preregistration-$p.md
  if [ -f "$f" ] && [ -f "$f.ots" ]; then note "  ok       $f (+ .ots)"
  elif [ -f "$f" ]; then miss "$f.ots (stamp it before GO)"
  else miss "$f"; fi
done
for p in attn-bench-r2-2026-11-04 bigmodel-2-2026-11-04; do
  f=proofs/preregistration-$p-ECARTS.md
  if [ -f "$f" ] && grep -qF "$IMAGE_SHA" "$f"; then note "  ok       $f (names the image)"
  else miss "$f naming $IMAGE_SHA"; fi
done
grep -qF "**Space sha:** \`$IMAGE_SHA\`" proofs/preregistration-gestures-2026-11-04.md \
  && note "  ok       the gestures prereg names the image" || miss "the Space sha line of the gestures prereg"

note "== 2. the code the image was built from =="
if git merge-base --is-ancestor "$BASE" HEAD; then note "  ok       HEAD $(git rev-parse --short HEAD) descends from $(git rev-parse --short "$BASE")"
else miss "HEAD does not descend from $BASE"; fi
# Everything the Space build reads (UPLOAD_ALLOW of ops/run.py, and the Dockerfile) must be BASE's, or the
# image is not the code the scripts check against (the attention kernel's sha256, the configs).
if git diff --quiet "$BASE" HEAD -- 'llvq-*' configs Cargo.toml Cargo.lock rust-toolchain.toml ops/fetch-qtip.sh \
     ops/Dockerfile.cuda && git diff --quiet HEAD -- 'llvq-*' configs; then
  note "  ok       the build perimeter is byte for byte $(git rev-parse --short "$BASE")'s, committed and in the tree"
else miss "the build perimeter differs from $BASE (git diff $BASE -- llvq-* configs ...)"; fi
DIRTY=$(git status --porcelain -- ops/jobs proofs)
if [ -n "$DIRTY" ]; then
  note "  the lot's scripts, preregs or stamps are not committed:"; printf '%s\n' "$DIRTY" | sed 's/^/    /'
  [ "$GO" = 1 ] && FAIL=1
else note "  ok       scripts and preregs committed"; fi
for b in oracle fusedrun rotbench attnbench gsm8k; do
  if grep -qE -- "--bin $b( |$)" ops/Dockerfile.cuda && grep -qE "^ +/src/target/release/$b( |$)" ops/Dockerfile.cuda
  then note "  ok       $b: in the cargo build and in the runtime COPY of ops/Dockerfile.cuda"
  else miss "binary $b in both lists of ops/Dockerfile.cuda"; fi
done

note "== 3. the Space =="
NOW=$(uv run --quiet --with huggingface_hub python -c \
  "from huggingface_hub import HfApi; print(HfApi().space_info('$SPACE').sha)" 2>/dev/null || echo unknown)
if [ "$NOW" = "$IMAGE_SHA" ]; then note "  ok       $SPACE at $NOW"
else
  note "  the Space reads $NOW, the lot needs $IMAGE_SHA"
  [ "$GO" = 1 ] && FAIL=1
fi

note "== 4. every script's DRY_RUN =="
dry() { local out; out=$(env "$@" 2>&1) || { note "  FAILED   ${*: -1}: $out"; FAIL=1; return; }; note "  $out"; }
dry DRY_RUN=1 bash ops/jobs/gesture-ab.sh
dry DRY_RUN=1 bash ops/jobs/bigmodel-identity.sh
dry DRY_RUN=1 bash ops/jobs/attn-bench-r2.sh
dry DRY_RUN=1 bash ops/jobs/attn-t8.sh
dry SIZE=8b DRY_RUN=1 bash ops/jobs/gsm8k-grouped-size.sh
dry SIZE=14b DRY_RUN=1 bash ops/jobs/gsm8k-grouped-size.sh

note "== 5. the 4B lot job, composed from the parts' own job scripts =="
# DRY: the parts written with placeholder object digests. GO: with every launch check of every part.
if [ "$GO" = 1 ]; then MODE="IMAGE_SHA=$IMAGE_SHA"; else MODE="DRY_RUN=1"; fi
emit() { env "$MODE" EMIT="$WORK/part-$1.sh" bash "ops/jobs/$2" 2>&1 | sed 's/^/  /' || FAIL=1; test -s "$WORK/part-$1.sh" || FAIL=1; }
emit gestures gesture-ab.sh
emit bigmodel bigmodel-identity.sh
emit attn-r2 attn-bench-r2.sh
emit attn-t8 attn-t8.sh
if [ "$GO" = 1 ]; then
  for s in 8b 14b; do
    env SIZE=$s IMAGE_SHA="$IMAGE_SHA" EMIT="$WORK/gsm8k-$s.sh" bash ops/jobs/gsm8k-grouped-size.sh 2>&1 | sed 's/^/  /' \
      || FAIL=1
  done
fi
PRE=$(printf 'L=%q\nCEIL_GESTURES=%q\nCEIL_BIGMODEL=%q\nCEIL_ATTN_R2=%q\nCEIL_ATTN_T8=%q' \
  "$L" "$CEIL_GESTURES" "$CEIL_BIGMODEL" "$CEIL_ATTN_R2" "$CEIL_ATTN_T8")
{
  printf '%s\n' 'mkdir -p "$L"'
  printf '%s\n' "if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi"
  printf '%s\n' 'nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader | tee "$L/gpu.txt"'
  for p in gestures bigmodel attn-r2 attn-t8; do
    printf 'cat > "$L/part-%s.sh" <<'"'"'LOT1_PART_EOF'"'"'\n' "$p"
    cat "$WORK/part-$p.sh"
    printf 'LOT1_PART_EOF\n'
  done
  cat <<'DRIVER'
sha256sum "$L"/part-*.sh | tee "$L/parts.sha256"
# One part: its own bash process, so its `set -e` holds; its own ceiling; its exit code kept, never fatal here.
part() {
  echo "== PART $1, ceiling $2 ==" ; date -u
  set +e
  timeout --kill-after=60s "$2" bash "$L/part-$1.sh" 2>&1 | tee "$L/part-$1.log"
  local rc=${PIPESTATUS[0]}
  set -e
  date -u
  echo "PART $1: exit $rc" | tee -a "$L/parts.txt"
}
part gestures "$CEIL_GESTURES"
part bigmodel "$CEIL_BIGMODEL"
part attn-r2 "$CEIL_ATTN_R2"
if grep -qx 'PART attn-r2: exit 0' "$L/parts.txt"; then
  part attn-t8 "$CEIL_ATTN_T8"
else
  echo 'PART attn-t8: not run (attention r2 did not pass its gates)' | tee -a "$L/parts.txt"
fi
echo '== lot 1, 4B: parts ==' ; cat "$L/parts.txt"
# The job reads as failed when a part failed; every part's raw output is on the mount either way.
if grep -qE '^PART .*: exit [1-9]' "$L/parts.txt"; then exit 1; fi
DRIVER
} > "$WORK/lot1-4b.body"
LOT_JOB="$WORK/lot1-4b.job"
printf 'set -euo pipefail\n%s\n' "$PRE" > "$LOT_JOB"
cat "$WORK/lot1-4b.body" >> "$LOT_JOB"
if bash -n "$LOT_JOB"; then note "  ok       $LOT_JOB parses"; else note "  FAILED   $LOT_JOB does not parse"; FAIL=1; fi
for p in gestures bigmodel attn-r2 attn-t8; do
  test "$(grep -c "^cat > \"\$L/part-$p.sh\"" "$LOT_JOB")" = 1 || { note "  FAILED   part $p not exactly once"; FAIL=1; }
done
test "$(grep -cx 'LOT1_PART_EOF' "$LOT_JOB")" = 4 || { note "  FAILED   a part holds its own delimiter"; FAIL=1; }
SIZE_B=$(wc -c < "$LOT_JOB" | tr -d ' ')
# The job goes to the container as one argv string of `bash -lc`; Linux caps one argument at 131,072 bytes.
if [ "$SIZE_B" -lt 120000 ]; then note "  ok       $SIZE_B bytes, under the 131,072-byte cap on one argument"
else note "  FAILED   $SIZE_B bytes: too large for one argument"; FAIL=1; fi

note "== 6. cost, l40sx1 at \$$USD_H/h (estimated; worst = the job timeout, computed) =="
python3 - "$USD_H" <<'PY'
import sys
r = float(sys.argv[1]) / 60
rows = [
    ("lot1-4b, T8 condition fails", 33.5, 180),
    ("lot1-4b, T8 band fails", 79.5, None),
    ("lot1-4b, T8 in full", 142.6, None),
    ("gsm8k-8b-grouped", 75.0, 100),
    ("gsm8k-14b-grouped", 137.0, 170),
]
print("  %-30s %9s %9s %10s %9s" % ("job", "exp. min", "exp. $", "ceiling", "worst $"))
for n, m, c in rows:
    print("  %-30s %9.1f %9.2f %10s %9s" % (n, m, m * r, "%dm" % c if c else "", "%.2f" % (c * r) if c else ""))
lo = 33.5 + 75 + 137; hi = 142.6 + 75 + 137; w = 180 + 100 + 170
print("  %-30s %9.1f %9.2f %10s %9.2f" % ("total, T8 condition fails", lo, lo * r, "%dm" % w, w * r))
print("  %-30s %9.1f %9.2f" % ("total, T8 in full", hi, hi * r))
import csv
rows = list(csv.DictReader(open("docs/data/jobs.csv")))
spent = sum(float(x["cost_usd"] or 0) for x in rows)
print("  registry before the lot: $%.2f over %d rows (measured, docs/data/jobs.csv); after it, $%.2f to $%.2f "
      "expected, $%.2f at worst (computed)" % (spent, len(rows), spent + lo * r, spent + hi * r, spent + w * r))
PY

if [ "$FAIL" != 0 ]; then
  note "== not ready: see MISSING or FAILED above =="
  [ "$GO" = 1 ] && { echo "refused: GO=1 with the lot not ready; nothing launched" >&2; exit 1; }
  note "(nothing launched; work files in $WORK)"
  exit 0
fi
if [ "$GO" != 1 ]; then
  note "== ready: GO=1 bash ops/jobs/lot1.sh launches the three jobs (work files in $WORK) =="
  exit 0
fi

note "== 7. launch: three jobs, in parallel =="
LOG=$WORK/launch.log
uv run ops/run.py bench \
  --image "hf.co/spaces/$SPACE" \
  --flavor l40sx1 --timeout "$LOT_TIMEOUT" \
  --bucket "$BUCKET" --out-mount /out \
  --name "lot1-4b${RETRY:+-r$RETRY}" \
  "$PRE" "$(cat "$WORK/lot1-4b.body")" 2>&1 | tee -a "$LOG"
for s in 8b 14b; do
  SIZE=$s IMAGE_SHA="$IMAGE_SHA" bash ops/jobs/gsm8k-grouped-size.sh 2>&1 | tee -a "$LOG"
done
note "== job ids =="
grep -E '^  id ' "$LOG" || true
note "(launch log: $LOG; register each job in docs/data/jobs.csv when it ends)"
