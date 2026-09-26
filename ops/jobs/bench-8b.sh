#!/usr/bin/env bash
# INTEGRATOR COPY of bench8b.sh: oracle added (hard rule 10), timeout 45m -> 30m ($0.90).
# DRAFT. The Tetra kernel at 8B shapes, out of the model: planesbench, one
# token's worth of the 252 projections, arms interleaved round by round.
# Destined for ops/jobs/bench-8b.sh once the operator gives the go.
#
# Preregistered: proofs/preregistration-served-8b-${D}.md   <-- NOT WRITTEN YET
#   (the same prereg as fused8b.sh can carry both jobs; hard rule 2).
#
# TWO FILES, ball FIRST, Tetra SECOND. planesbench reads the Tetra file from
# args().nth(2) (planesbench.rs:1280; jobs.csv still says 1242, the file grew); the 4B's first attempt passed one file
# and was refused after the NVRTC compile, $0.10 (jobs.csv, 6aaf7b68). A swap
# is refused by name (planesbench.rs:1864-1870).
#   ball  /out/qwen3-8b-c12-77e76284/qwen3-8b-llvq.bin  the B3 reseal, the only
#         8B ball file: 4,324,243,889 B, sha256 01670ebf... (b3-8b-reseal-2026-08-18.txt
#         :63-64; used by b2-8b and vague2-fusion-8b). tetra-8b-2026-09-06/
#         qwen3-8b-planes14.bin has the same size and date and is not used.
#   tetra the FT file by default (OBJ=ft), or the base (OBJ=base). The fold
#         changes row scales only: "no index byte changes, the rate is
#         untouched and the decoder stays byte-identical" (rowscale.rs:17-18),
#         so the base gives the same stream to time and can run as soon as it
#         is uploaded. The prereg must say which file it names.
#
# ARMS, default fp16,planes14,nullk,tetra48: the tuile-l40s set
# (tuile-l40s.sh:15), the latest 4B served-path reference at tile 64. fp16
# cannot be dropped (the tuile-l40s first attempt, 6aaf86db, $0.05). nullk
# gives the FLOOR REMOVED ratio (planesbench.rs:3591-3612); planes14 is the
# ball control. ARMS=fp16,planes14,awq,nullk,tetra48 adds the AWQ kernel's
# GB/s, the only comparable quantity for a competitor (hard rule 5).
#
# The full ten-arm table of banc-tetra.sh is NOT the default: 1,597 s running
# at 4B (hf jobs inspect 6aaf7c10), so ~51 min at 8B, and its buffers, 18.4 GB
# at 4B plus the A4 fused streams, scale to ~43 GB at 8B against 48 GB.
#
# TILES, default "unset": the served policy, 64 on sm_89 (tile.rs:239-241,
# resolved at planesbench.rs:1372). TILES="unset 128" adds a process at 128,
# the tile banc-tetra-2026-09-20 was measured at, one process a tile as
# tuile-l40s did.
#
# ⚠️ Without code-changes.patch, the two "f16 lm_head" lines are WRONG at 8B:
# planesbench.rs:3537 adds a 389,070,848-weight head, the 4B's (and a count
# that matches no tensor, fiche-4b.md:48-49), where the 8B reads 622,329,856.
# The table, the per-arm ratios against FP16 and the FLOOR REMOVED line do not
# use it. Either rebuild the image with the patch first, or strike those two
# lines from the journal.
#
# Cost, estimated (l40sx1 $1.80/h): tuile-l40s ran 3 processes of these four
# arms in 464 s at 4B, build 146-149 s each (sweep.txt:34,111,188). The build
# scales with the weights, x1.91 (6,945,767,424 / 3,633,315,840): ~290 s a
# process, ~300 s with verification and 7 rounds.
#   TILES=unset      pull 2.5 + sha 1.5 + 5.0 = ~9 min, $0.27 [$0.21, $0.36]
#   TILES="unset 128"                  ~14 min, $0.42 [$0.33, $0.54]
#   timeout 45m, worst $1.35
#
# DRY_RUN=1 bash bench8b.sh   prints the job script, parses it, launches nothing.
set -euo pipefail

REPO=$HOME/Documents/Pro/workspace/poc/llvq
D=${RUN_DATE:-2026-09-21}
PREREG=${PREREG:-proofs/preregistration-served-8b-$D.md}
BUCKET=Pier-Jean/jobs-artifacts
OBJ=${OBJ:-ft}
ARMS=${ARMS:-fp16,planes14,nullk,tetra48}
TILES=${TILES:-unset}
DRY=${DRY_RUN:-0}

B=/out/qwen3-8b-c12-77e76284/qwen3-8b-llvq.bin
B_BYTES=4324243889
B_SHA=01670ebf2a2aed8a4dcbb96118a3579101626040185b699130cd8f9b97dbf8e7
case "$OBJ" in
  base) T_DIR=dclm-8b-2026-09-21; T_NAME=qwen3-8b-dclm.bin; T_LOCAL=$HOME/qwen3-8b-dclm.bin ;;
  ft)   T_DIR=${FT_DIR:-}; T_NAME=qwen3-8b-dclm-ft.bin; T_LOCAL=${FT_LOCAL:-$HOME/qwen3-8b-dclm-ft.bin} ;;
  *) echo "refused: OBJ=$OBJ, expected base or ft" >&2; exit 1 ;;
esac
[ "$DRY" = 1 ] && T_DIR=${T_DIR:-dclm-8b-ft-DRYRUN}
[ -n "$T_DIR" ] || { echo "refused: OBJ=ft needs FT_DIR=<bucket dir of the FT file>" >&2; exit 1; }
case ",$ARMS," in *,fp16,*) ;; *) echo "refused: ARMS without fp16 (the witness)" >&2; exit 1 ;; esac
case ",$ARMS," in *,tetra48,*) ;; *) echo "refused: ARMS without tetra48, nothing to measure" >&2; exit 1 ;; esac
for t in $TILES; do
  case "$t" in unset|32|64|128|256) ;; *) echo "refused: tile $t" >&2; exit 1 ;; esac
done

T=/out/$T_DIR/$T_NAME
O=/out/bench-8b-$OBJ-$D
cd "$REPO"

if [ "$DRY" = 1 ]; then
  T_SHA=0000000000000000000000000000000000000000000000000000000000000000
  T_BYTES=4364205777
else
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  test -f "$T_LOCAL" || { echo "refused: $T_LOCAL missing" >&2; exit 1; }
  T_SHA=$(shasum -a 256 "$T_LOCAL" | cut -d' ' -f1)
  T_BYTES=$(stat -f %z "$T_LOCAL")
  REMOTE=$(hf buckets ls "hf://buckets/$BUCKET/$T_DIR/" 2>/dev/null | awk -v n="$T_NAME" '$NF ~ (n "$") {print $1}' || true)
  [ "$REMOTE" = "$T_BYTES" ] || { echo "refused: bucket copy is ${REMOTE:-absent} B, local $T_BYTES B" >&2; exit 1; }
  RB=$(hf buckets ls "hf://buckets/$BUCKET/qwen3-8b-c12-77e76284/" 2>/dev/null | awk '$NF ~ /qwen3-8b-llvq\.bin$/ {print $1}' || true)
  [ "$RB" = "$B_BYTES" ] || { echo "refused: ball file is ${RB:-absent} B, expected $B_BYTES" >&2; exit 1; }
  if hf buckets ls "hf://buckets/$BUCKET/bench-8b-$OBJ-$D/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: bench-8b-$OBJ-$D/ already exists; pick another RUN_DATE" >&2; exit 1
  fi
  echo "ball $B ($B_BYTES B); tetra $T ($T_BYTES B, sha256 $T_SHA); arms $ARMS; tiles $TILES"
fi

PRE=$(printf 'O=%q\nB=%q\nT=%q\nB_BYTES=%q\nB_SHA=%q\nT_BYTES=%q\nT_SHA=%q\nARMS=%q\nTILES=%q' \
  "$O" "$B" "$T" "$B_BYTES" "$B_SHA" "$T_BYTES" "$T_SHA" "$ARMS" "$TILES")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader | tee "$O/gpu.txt"
echo '== oracle (hard rule 10) =='
oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle-cuda.txt" | tail -2
grep -q MATCH "$O/oracle-cuda.txt"
echo '== both files on the mount: bytes and sha256 =='
test "$(stat -c %s "$B")" = "$B_BYTES"
test "$(stat -c %s "$T")" = "$T_BYTES"
sha256sum "$B" "$T" | tee "$O/files.sha256"
test "$(awk 'NR==1 {print $1}' "$O/files.sha256")" = "$B_SHA"
test "$(awk 'NR==2 {print $1}' "$O/files.sha256")" = "$T_SHA"
for t in $TILES; do
  echo "== tile $t: planesbench, ball FIRST, Tetra SECOND, arms $ARMS ==" ; date -u
  if [ "$t" = unset ]; then
    LLVQ_BENCH_ARMS="$ARMS" planesbench "$B" "$T" 2>&1 | tee "$O/bench-tile-served.txt" | tail -45
    # The served tile, or this was not the served-path measurement.
    grep -qF 'tile 64 (served: measured optimum for sm_89)' "$O/bench-tile-served.txt"
  else
    LLVQ_TILE_BLOCKS="$t" LLVQ_BENCH_ARMS="$ARMS" planesbench "$B" "$T" 2>&1 | tee "$O/bench-tile-$t.txt" | tail -45
  fi
  date -u
done
# The coverage the ratios are read on: 216 of 252 at 8B as at 4B.
grep -hF 'matched by name' "$O"/bench-tile-*.txt
echo '== raw kept ==' ; ls -la "$O" ; wc -l "$O"/*.txt
JOB

if [ "$DRY" = 1 ]; then
  JOBSCRIPT="set -euo pipefail
$PRE
$BODY"
  printf '%s\n' "$JOBSCRIPT"
  printf '%s\n' "$JOBSCRIPT" | bash -n && echo "DRY_RUN: job script parses; nothing launched" >&2
  exit 0
fi

uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor l40sx1 --timeout 30m \
  --bucket "$BUCKET" --out-mount /out \
  --name "bench-8b-$OBJ" \
  "$PRE" "$BODY"
