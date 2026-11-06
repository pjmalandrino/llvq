#!/usr/bin/env bash
# DRAFT, never parsed and never run (wave-5 deep dive "gestures", 2026-11-02, revised after review).
# Not part of the repository. Proposed path: ops/jobs/gqa-size.sh.
# Wave-6 check, 2026-11-02: `bash -n` passes, and the DRY_RUN path ran read-only against HEAD 06e1602 for
# SIZE=8b and SIZE=14b (both job scripts parse; SIZE=4b is refused). Nothing was launched.
#
# Built line for line from ops/jobs/gqa-ab.sh and ops/jobs/gqa-ppl.sh at HEAD 06e1602: the same refusal block,
# the same PRE printf (every variable the body reads is in it), the same config write and sha256, the same
# DRY_RUN branch with `bash -n`, the same run.py call. Only the SIZE case is new, in the shape of
# ops/jobs/paper-served.sh; both hashes and the 14B line re-derive from the paper-table brut files below.
# One departure from the templates: neither sealed file is on the Mac any more (~/q8b-14b-sealed-2026-09-23/
# holds only text files, read on 2026-11-02), so the object's bytes and sha256 are pinned from committed
# measurements instead of hashed from a local copy, as ops/jobs/ppl-scelles.sh pinned them on 2026-10-04.
#
# The grouped attention at 8B or 14B, one job a size: the published arm replayed, the speed A/B
# (LLVQ_GQA_AB, three arms, 5/2048/4096/8192) and the quality control of the tolerance gate
# (LLVQ_GQA_PPL=4096,12), all through the served kernel on the bench door, with the size's served flags
# spelled out. Preregistered: proofs/preregistration-gqa-size-YYYY-MM-DD.md, stamped before this.
#
#   SIZE=8b|14b DRY_RUN=1 bash ops/jobs/gqa-size.sh                 prints and parses the job script
#   SIZE=8b|14b IMAGE_SHA=<space sha> bash ops/jobs/gqa-size.sh      launches, after the operator's go
#
# The 14B runs only after the 8B landed in R1 or R2 of the prereg (§7). The launcher cannot read a verdict,
# so SIZE=14b asks for GQA_8B_VERDICT=R1 or R2, copied by hand from the 8B journal; the operator's go decides.
#
# Four steps in one job: oracle; the published arm replayed (paper-table run 2 or 3, flags spelled out,
# against the dense arm, decoded line checked by sha256 against that run), which checks that the default
# path did not move; LLVQ_GQA_AB on the same flags; LLVQ_GQA_PPL on the same flags. The config is written by
# the job from this command line and checked by sha256.
#
# The image: built from a commit whose engine code, tests aside, is the served one, f808856 (the served image
# f77f803e) or HEAD 06e1602, which differ by one refusal message on a path this job does not take. The
# group-of-5 test is a host test and changes no binary. An image that carries the audit's fusions (norm and
# rotation, tv_tetra48_seg) is another object, and gate 4 would void the run at step 2.
#
# Cost, estimated (l40sx1 $1.80/h): 8B about 46 min, $1.38, ceiling 90m, $2.70; 14B about 77 min, $2.31,
# ceiling 120m, $3.60.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

SIZE=${SIZE:?SIZE=8b or 14b}
# BYTES as measured on the bucket copy (sealed-8b-27-2026-09-24.txt:12, sealed-8b-14b-2026-09-23.txt:12) and
# again on the mount (ppl-scelles-2026-10-04.txt:23).
case "$SIZE" in
  8b)  OBJ_DIR=sealed-8b27-2026-09-24;  NAME=qwen3-8b-sealed-B.bin
       BYTES=2815098745; TIMEOUT=90m
       # The decoded line of run 2 of the paper table, `grep -F '  fused F0:  '` of $REF below.
       REF_LINE_SHA=323cabeb8520cba9cede5ea6c1a611a1ceb5322c0aae473fac63ed911d61e498
       DENSE_LINE='fused F0: 256 tokens identical to the dense arm' ;;
  14b) OBJ_DIR=sealed-14b-2026-09-23;  NAME=qwen3-14b-sealed.bin
       BYTES=5087000541; TIMEOUT=120m
       # Run 3: the fused arm leaves the dense one at token 78, so the gate is that same position.
       REF_LINE_SHA=186974aa69fef4b5611d3fd7598b3f517bb1e35ae78142da836ee75db3cfeabe
       DENSE_LINE='WARNING: fused F0: divergence at token 78 out of 256' ;;
  *) echo "refused: SIZE=$SIZE, expected 8b or 14b" >&2; exit 1 ;;
esac
TABLES='2 tables (model.embed_tokens.weight + lm_head.weight)'
PREREG=proofs/preregistration-gqa-size-YYYY-MM-DD.md
BUCKET=Pier-Jean/jobs-artifacts
CONF_LOCAL=configs/qwen3-$SIZE-tetra-e4.json
REF=docs/mesures/paper-table-2026-09-25-brut/paper-served-$SIZE-2026-09-25-r2/fused-q4-256.txt
LENGTHS=5,2048,4096,8192
PPL_SPEC=4096,12
# The scored stream of ppl-scelles-2026-10-04 and gqa-ppl-2026-11-01: wikitext-2 test, 12 windows of 4,096.
# `ppl` printed this fingerprint for all three sealed files, each read with the tokenizer it carries
# (ppl-scelles-2026-10-04.txt:27, measured), and fusedrun reads that same tokenizer (fused_cuda.rs:2903).
TOKENS=3f1baca9033bf251
# RETRY=2 names a relaunch after a failed attempt left its directory behind.
O=/out/gqa-size-$SIZE-YYYY-MM-DD${RETRY:+-r$RETRY}
F=/out/$OBJ_DIR/$NAME
C=$O/qwen3-$SIZE-tetra-e4.json
DRY=${DRY_RUN:-0}

CONF=$(cat "$CONF_LOCAL")
CONF_SHA=$(printf '%s\n' "$CONF" | shasum -a 256 | cut -d' ' -f1)
[ "$CONF_SHA" = "$(shasum -a 256 < "$CONF_LOCAL" | cut -d' ' -f1)" ] \
  || { echo "refused: $CONF_LOCAL does not end in exactly one newline" >&2; exit 1; }
# The published arm is the expanded form: a config that already names an attention form is not the object
# whose replay this job gates on.
python3 -c 'import json,sys; d=json.load(open(sys.argv[1]));
assert (d["layout"],d["embed"],d["rot_share"],d["fuse"],d["kv"])==("tetra48","q4","1","0","f16"), d
assert d.get("attention") is None, d' "$CONF_LOCAL"
[ "$(grep -F '  fused F0:  ' "$REF" | shasum -a 256 | cut -d' ' -f1)" = "$REF_LINE_SHA" ] \
  || { echo "refused: the reference line of $REF does not hash to $REF_LINE_SHA" >&2; exit 1; }
grep -qF "$DENSE_LINE" "$REF" \
  || { echo "refused: $REF does not carry the line gate 4 expects" >&2; exit 1; }
# The object's sha256 as paper-table run 2 or 3 read it on the mount, from that run's committed files.sha256.
SHA=$(awk -v p="$F" '$2 == p {print $1}' "${REF%/*}/files.sha256")
[ "${#SHA}" = 64 ] || { echo "refused: ${REF%/*}/files.sha256 holds no single sha256 for $F" >&2; exit 1; }

if [ "$DRY" != 1 ]; then
  # Launch path only, so `SIZE=14b DRY_RUN=1` still parses without it.
  if [ "$SIZE" = 14b ]; then
    case "${GQA_8B_VERDICT:-}" in
      R1|R2) ;;
      *) echo "refused: the 14B runs after the 8B landed in R1 or R2 (prereg §7); GQA_8B_VERDICT=${GQA_8B_VERDICT:-unset}" >&2
         exit 1 ;;
    esac
  fi
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  REMOTE=$(hf buckets ls "hf://buckets/$BUCKET/$OBJ_DIR/" 2>/dev/null | awk -v n="/$NAME" 'substr($NF, length($NF) - length(n) + 1) == n {print $1}' || true)
  [ "$REMOTE" = "$BYTES" ] || { echo "refused: bucket copy is ${REMOTE:-absent} B, expected $BYTES B" >&2; exit 1; }
  if hf buckets ls "hf://buckets/$BUCKET/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already exists" >&2; exit 1
  fi
  IMAGE_SHA=${IMAGE_SHA:?IMAGE_SHA=<the Space sha of an image built on the served engine, see the header>}
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    'from huggingface_hub import HfApi; print(HfApi().space_info("Pier-Jean/llvq-runner-cuda").sha)')
  [ "$NOW" = "$IMAGE_SHA" ] || { echo "refused: the Space is at $NOW, expected $IMAGE_SHA" >&2; exit 1; }
  echo "$SIZE sealed: $BYTES B, sha256 $SHA (paper-table digest); config $CONF_LOCAL sha256 $CONF_SHA; image $NOW"
fi

PRE=$(printf 'O=%q\nF=%q\nC=%q\nSHA=%q\nBYTES=%q\nCONF=%q\nCONF_SHA=%q\nTABLES=%q\nREF_LINE_SHA=%q\nDENSE_LINE=%q\nLENGTHS=%q\nPPL_SPEC=%q\nTOKENS=%q' \
  "$O" "$F" "$C" "$SHA" "$BYTES" "$CONF" "$CONF_SHA" "$TABLES" "$REF_LINE_SHA" "$DENSE_LINE" "$LENGTHS" "$PPL_SPEC" "$TOKENS")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader | tee "$O/gpu.txt"
echo '== oracle (hard rule 10) =='
oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle-cuda.txt" | tail -2
grep -q MATCH "$O/oracle-cuda.txt"
echo '== the object on the mount: bytes and sha256 against the paper-table run =='
test "$(stat -c %s "$F")" = "$BYTES"
sha256sum "$F" | tee "$O/files.sha256"
test "$(cut -d' ' -f1 "$O/files.sha256")" = "$SHA"
echo '== the served config, written from the launch command =='
printf '%s\n' "$CONF" > "$C"
sha256sum "$C" | tee -a "$O/files.sha256"
test "$(sha256sum "$C" | cut -d' ' -f1)" = "$CONF_SHA"
cat "$C"
PARITY='int4: tv_q4_hh equals tv_q4_h bit for bit'
TILE='tile 64 (served: measured optimum for sm_89)'
FLAGS='LLVQ_FUSED_LAYOUT=tetra48 LLVQ_ROT_SHARE=1 LLVQ_FUSE=0 LLVQ_KV=f16 LLVQ_EMBED=q4'
echo '== STEP 2: the published arm replayed, 256 tokens against the dense arm, same process =='
date -u
env $FLAGS fusedrun "$F" 256 2>&1 | tee "$O/published-q4-256.txt" | tail -30
date -u
grep -qF "$TILE" "$O/published-q4-256.txt"
grep -qF "$PARITY" "$O/published-q4-256.txt"
grep -F 'embedding: q4 g64 (LLVQ_EMBED)' "$O/published-q4-256.txt" | grep -qF "$TABLES"
grep -qF "$DENSE_LINE" "$O/published-q4-256.txt"
test "$(grep -cF '  fused F0:  ' "$O/published-q4-256.txt")" = 1
test "$(grep -F '  fused F0:  ' "$O/published-q4-256.txt" | sha256sum | cut -d' ' -f1)" = "$REF_LINE_SHA"
echo 'gate 4: decoded line identical to the paper-table run of this size'
echo "== STEP 3: LLVQ_GQA_AB=$LENGTHS, three arms, one model, 256 tokens =="
date -u
env $FLAGS LLVQ_GQA_AB="$LENGTHS" fusedrun "$F" 256 2>&1 | tee "$O/gqa-ab-256.txt" | tail -40
date -u
grep -qF "$TILE" "$O/gqa-ab-256.txt"
grep -qF "$PARITY" "$O/gqa-ab-256.txt"
grep -qF 'cudarc event tracking: off (one stream, ordered by issue)' "$O/gqa-ab-256.txt"
test "$(grep -cE '^prompt [0-9]+$' "$O/gqa-ab-256.txt")" = 4
test "$(grep -cE '^  cat\+expanded +[0-9.]+ \[' "$O/gqa-ab-256.txt")" = 4
for arm in shared grouped; do
  test "$(grep -cE "^  grow\(256\)\+$arm +logits against cat\+expanded: " "$O/gqa-ab-256.txt")" = 4
  test "$(grep -cE "^  grow\(256\)\+$arm +[0-9.]+ \[.*(tokens identical|differs from token)" "$O/gqa-ab-256.txt")" = 4
done
echo "== STEP 4: LLVQ_GQA_PPL=$PPL_SPEC, two arms, prefill and decode paths =="
date -u
env $FLAGS LLVQ_GQA_PPL="$PPL_SPEC" fusedrun "$F" 2>&1 | tee "$O/gqa-ppl.txt" | tail -24
date -u
grep -qF "$TILE" "$O/gqa-ppl.txt"
grep -qF "$PARITY" "$O/gqa-ppl.txt"
grep -qF 'cudarc event tracking: off (one stream, ordered by issue)' "$O/gqa-ppl.txt"
grep -qF "(tokens $TOKENS)" "$O/gqa-ppl.txt"
test "$(grep -cE '^  window +[0-9]+/12  prefill nll ' "$O/gqa-ppl.txt")" = 12
test "$(grep -cE '^  (prefill|decode) +[0-9.]+ +[0-9.]+ +[-+][0-9.]+ % \[' "$O/gqa-ppl.txt")" = 2
# The harness: both paths score the same tokens, so on the published arm they agree to rounding.
awk '/^  harness: decode over prefill/ {r = $NF; ok = (r >= 0.99 && r <= 1.01)} END {exit !ok}' "$O/gqa-ppl.txt"
echo 'gate: decode and prefill agree on the published arm'
echo '== raw kept ==' ; ls -la "$O" ; wc -l "$O"/*.txt
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
  --name "gqa-size-$SIZE${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
