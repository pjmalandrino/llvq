#!/usr/bin/env bash
# DRAFT, not launched. The two REFERENCE arms of the 14B MMLU census, full split,
# one job, one card: f16 (B) and AWQ w4 g128 dequantized (C). The object's own arms
# (A, the DCLM base, and FT, the row-scale-trained file) do not exist yet and are
# scored later under their own preregs; B and C do not wait for them.
#
# Preregistered: proofs/preregistration-census-14b-ref-2026-09-22.md, DRAFT, NOT
# STAMPED. Hard rule 2: this launcher refuses to run until the .ots exists.
#
# Arms, both f16, dense reconstruction, LLVQ_MMLU_ALLOC=flat, no limit = census, 14,042 q:
#   B  f16      Qwen/Qwen3-14B@40c069824f42...   Hub, public, PINNED (the 8B census
#               printed its revision; here it is part of the command)
#   C  AWQ deq  /scratch/awq-14b                 copied from the bucket, see below
# Order: C is staged first (a broken copy is known in minutes), B is scored, C last,
# so a timeout can only cut C. The staging is bounded (20 min, 600 s a file) so that
# a stalled mount cannot hold B back either.
#
# Why B and C are needed: the only 14B f16 and AWQ dumps are limit=40, fingerprint
# 65dcd53655e8bfa5 (docs/data/mmlu-dumps/mmlu-14b-{f16,awq}.csv, job 6a7971c3,
# 2026-08-10). mmlupair pairs them with a census dump only under --intersect, on the
# 2,280 shared questions (mmlupair.rs:326-380). No 14B census exists, in the repo or
# in the bucket (`hf buckets ls -R | grep -i mmlu | grep -i 14b`, 2026-09-22: the
# three campagne-14b-qualite/ dumps only).
# Free by-product, as at 8B: B and C against those two dumps under --intersect are a
# harness control across the device port of 2026-09-20: same bytes, same questions.
#
# C IS NOT ON THE HUB. Pier-Jean/qwen3-14b-awq-deq returns 404; the dequantized
# checkpoint lives only in the bucket, at qwen3-14b-awq-deq-1g/, with three stray
# `.shard-0003{0,1,2}.safetensors` that duplicate model-00031..33 (the silent
# os.rename failure, ops/README.md:251-254). Two rules follow:
#   - SIGBUS: mmapping safetensors ON THE BUCKET MOUNT killed job 6a796e08
#     (campagne-14b-qualite-2026-08-10.txt:5-8). The job copies to /scratch, the
#     container's local disk (`ENV HF_HOME=/scratch/hf` and `WORKDIR /scratch`,
#     ops/Dockerfile.cuda:161-162 at afaed1e, the image's commit), with `cp`, the
#     read that worked on 2026-08-10 (28 GB into /tmp/awq, job 6a7971c3), and mmaps
#     from there.
#   - EXPLICIT LIST: the 40 files of MANIFEST below, never a glob, so the dotfiles
#     stay out by construction, and LICENSE (11,544 B) with them: the directory holds
#     44 entries. MANIFEST is the bucket listing of 2026-09-22 (`hf buckets ls`,
#     sizes authoritative, ops/README.md:270-274); the Mac refuses to launch unless
#     the live listing still matches it byte for byte, and prints what it leaves out.
#     Read the same day from the bucket (`hf buckets cp <file> -`, $0): index.json
#     total_size 29,536,614,400, 443 tensors over exactly the 33 model-000NN shards;
#     config.json with the 14B shapes, torch_dtype float16, no quantization_config.
# Before C is scored, the job proves the local copy (python3, stdlib only; the image
# carries full python3 since d958070): each file at its listed size; each shard's
# header lands exactly on its length; the headers' tensor bytes equal index.json
# `total_size` (tensor bytes, NOT file bytes: ops/awq_dequant.py:602) and
# 2 x 14,768,307,200 weights; 443 tensors; the index maps to the 33 listed shards and
# no other; config.json has the 14B shapes and no quantization_config; tokenizer.json
# has Qwen3-14B's sha256. If any of it fails, C is not scored, B still is, and the job
# exits non-zero at the end (a COMPLETED job with a missing arm would read as a pass).
#
# Bucket: this job mounts Pier-Jean/jobs-artifacts writable. Two jobs writing the same
# bucket do not mount (ops/README.md:241-243): the 14B encode job is launched after
# this one ends, or into another bucket.
#
# Image: the Space must sit at IMAGE_SHA (a963a020 = afaed1e, 2026-09-21 01:12 UTC).
# If it is rebuilt first (export, rowscale), set IMAGE_SHA to the new sha: the dense
# scoring path (llvq-llm/src/{bin/mmlu.rs,loader.rs,model.rs,eval.rs}) is unchanged
# from afaed1e to 01dae9a (`git diff --stat`), and control 4 is the check on the card.
#
# Cost, estimated. Scoring time per arm scaled from the 8B census (1,744 s f16,
# 1,740 s AWQ; census-8b-2026-09-21-brut/out-*.txt) by the 14B/8B ratio of the old
# sample runs: 387/248 = 1.56 for the f16 and AWQ arms, 400/253 = 1.58 for the sealed
# arm (campagne-14b-qualite-2026-08-10.txt:99,206,312; campagne-8b-qualite-2026-08-08
# .txt:121,203,285). B 2,722 s + C 2,715 s = 90.6 min. Overhead ~14 min: start and
# oracle 2.5, staging C 4.5 (cp of 29.54 GB + sha256), B's Hub download of 29.54 GB and
# load 6, C's load 1. => ~105 min, ~$3.15 on l40sx1 at $1.80/h, range [95, 120] min =
# [$2.85, $3.60]. If the ported harness scales like the non-embedding FLOPs instead
# (x1.90), ~124 min: still inside the timeout.
#   timeout 135m: +12.5 % over the top of the range; a cut can only land in arm C,
#   which starts near minute 60 => worst case $4.05 at the timeout, $4.89 if the
#   platform bills 28 min past it, as it did once (volume-v32, timeout 120 min, 148 billed, jobs.csv:121).
# VRAM: 29.54 GB of f16 weights on 46,068 MiB; the 14B f16 ran ppl at ctx 4096 on this
# card on 2026-08-10, before the port. The longest census prompt at 14B on the ported
# harness is not measured.
#
# DRY_RUN=1 bash ops/jobs/census-14b-ref.sh   assembles the job script, syntax-checks
# it, launches nothing, touches no network.
# PREFLIGHT_ONLY=1 bash ops/jobs/census-14b-ref.sh   runs the Mac checks (bucket listing,
# output dir, Space sha, Hub revision and tokenizer), then stops; $0, launches nothing.
set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
D=${CENSUS_DATE:-$(date -u +%F)}                  # launch date (UTC); names the output dir
PREREG=${PREREG:-proofs/preregistration-census-14b-ref-2026-09-22.md}
IMAGE_SHA=${IMAGE_SHA:-a963a02010cec2d3c342ec52dd38f2dded06f2a1}
BUCKET=Pier-Jean/jobs-artifacts
AWQ_DIR=qwen3-14b-awq-deq-1g
REV=40c069824f4251a91eefaf281ebe4c544efd3e18      # Qwen/Qwen3-14B, main since 2025-07-26
TOK_SHA=aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4  # its tokenizer.json (= 4B, 8B)
O=/out/census-14b-ref-$D
SRC=/out/$AWQ_DIR
AWQ=/scratch/awq-14b
FP=a74a6d6213602979                               # census plan fingerprint, every FULL dump
DATA_BYTES=29536614400                            # 2 x 14,768,307,200 weights
N_PARAMS=14768307200
N_TENSORS=443                                     # 40 x 11 + embed + norm + lm_head
SHARD_SUM=29536663384                             # the 33 shards, bytes, listing 2026-09-22
DRY=${DRY_RUN:-0}
PREFLIGHT=${PREFLIGHT_ONLY:-0}                    # 1: the Mac checks below, then stop, $0
PROV=${PROV:-/tmp/census-14b-ref-$D-provenance.txt}

# name and byte count of every file C needs; `hf buckets ls`, 2026-09-22
MANIFEST='config.json 728
generation_config.json 239
merges.txt 1671853
model-00001-of-00033.safetensors 964691280
model-00002-of-00033.safetensors 838861704
model-00003-of-00033.safetensors 964691280
model-00004-of-00033.safetensors 838861704
model-00005-of-00033.safetensors 838861704
model-00006-of-00033.safetensors 964691280
model-00007-of-00033.safetensors 838861704
model-00008-of-00033.safetensors 838861712
model-00009-of-00033.safetensors 964691296
model-00010-of-00033.safetensors 838861712
model-00011-of-00033.safetensors 838861712
model-00012-of-00033.safetensors 964691296
model-00013-of-00033.safetensors 838861712
model-00014-of-00033.safetensors 838861712
model-00015-of-00033.safetensors 964691296
model-00016-of-00033.safetensors 838861712
model-00017-of-00033.safetensors 838861712
model-00018-of-00033.safetensors 964691296
model-00019-of-00033.safetensors 838861712
model-00020-of-00033.safetensors 838861712
model-00021-of-00033.safetensors 964691296
model-00022-of-00033.safetensors 838861712
model-00023-of-00033.safetensors 838861712
model-00024-of-00033.safetensors 964691296
model-00025-of-00033.safetensors 838861712
model-00026-of-00033.safetensors 838861712
model-00027-of-00033.safetensors 964691296
model-00028-of-00033.safetensors 838861712
model-00029-of-00033.safetensors 838861712
model-00030-of-00033.safetensors 838861712
model-00031-of-00033.safetensors 1555824736
model-00032-of-00033.safetensors 1555824752
model-00033-of-00033.safetensors 866776
model.safetensors.index.json 36514
tokenizer.json 11422654
tokenizer_config.json 9732
vocab.json 2776833'

cd "$REPO"

# ---- preflight on the Mac, $0 ----------------------------------------------------------
# The constant block against itself first: a typo in MANIFEST must not reach the card.
GOT=$(printf '%s\n' "$MANIFEST" | awk '$1 ~ /^model-000[0-9][0-9]-of-00033\.safetensors$/ {n++; s+=$2} END {printf "%d %d", n, s}')
[ "$GOT" = "33 $SHARD_SUM" ] || { echo "refused: MANIFEST holds '$GOT' (shards, bytes), expected '33 $SHARD_SUM'" >&2; exit 1; }
[ "$(printf '%s\n' "$MANIFEST" | wc -l | tr -d ' ')" = 40 ] || { echo "refused: MANIFEST is not 40 lines" >&2; exit 1; }
[ $((SHARD_SUM - DATA_BYTES)) -gt 0 ] && [ $((SHARD_SUM - DATA_BYTES)) -lt 330000 ] \
  || { echo "refused: shard bytes minus tensor bytes = $((SHARD_SUM - DATA_BYTES)), not a few headers" >&2; exit 1; }

if [ "$DRY" != 1 ]; then
  if ! { test -f "$PREREG" && test -f "$PREREG.ots"; }; then
    [ "$PREFLIGHT" = 1 ] || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
    echo "PREFLIGHT_ONLY: $PREREG(.ots) missing; a launch would be refused here" >&2
  fi
  if hf buckets ls "hf://buckets/$BUCKET/census-14b-ref-$D/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: census-14b-ref-$D/ already exists; pick another CENSUS_DATE" >&2; exit 1
  fi
  # The live listing against MANIFEST, file by file. `(empty)` is what `hf buckets ls`
  # prints for a directory that does not exist (served-8b ECARTS E2); it matches no line.
  LIVE=$(hf buckets ls "hf://buckets/$BUCKET/$AWQ_DIR/" 2>/dev/null | awk '{n=$NF; sub(/^.*\//, "", n); print n, $1}' || true)
  while read -r name size; do
    printf '%s\n' "$LIVE" | grep -qxF "$name $size" \
      || { echo "refused: $AWQ_DIR/$name is not $size B in the bucket (listing changed?)" >&2; exit 1; }
  done <<EOF
$MANIFEST
EOF
  # Every live name MANIFEST does not hold (2026-09-22: the three dotfiles and
  # LICENSE). ENVIRON, not -v: the Mac's awk refuses a newline in a -v value.
  echo "AWQ-deq 14B: 40 files as listed, 33 shards = $SHARD_SUM B; left out:" \
       "$(printf '%s\n' "$LIVE" | M="$MANIFEST" awk 'BEGIN {n = split(ENVIRON["M"], l, "\n"); for (i = 1; i <= n; i++) {split(l[i], f, " "); keep[f[1]] = 1}} NF && !($1 in keep) {printf "%s ", $1}')"
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    'from huggingface_hub import HfApi; print(HfApi().space_info("Pier-Jean/llvq-runner-cuda").sha)')
  [ "$NOW" = "$IMAGE_SHA" ] || { echo "refused: the Space is at $NOW, IMAGE_SHA says $IMAGE_SHA" >&2; exit 1; }
  # Provenance the image cannot print, and the tokenizer the fingerprint depends on.
  uv run --quiet --with huggingface_hub python - "$REV" "$TOK_SHA" <<'PY' | tee "$PROV"
import sys
from huggingface_hub import HfApi
rev, tok = sys.argv[1], sys.argv[2]
a = HfApi()
s = a.space_info("Pier-Jean/llvq-runner-cuda")
print("image  Pier-Jean/llvq-runner-cuda", s.sha, s.last_modified)
i = a.model_info("Qwen/Qwen3-14B", revision=rev, files_metadata=True)
assert i.sha == rev, (i.sha, rev)
t = {x.rfilename: x for x in i.siblings}["tokenizer.json"]
assert t.lfs and t.lfs.sha256 == tok, t
n = sum(x.size for x in i.siblings if x.rfilename.endswith(".safetensors"))
print("model  Qwen/Qwen3-14B", i.sha, "public" if not i.private else "private", n, "B of safetensors")
print("model  Qwen/Qwen3-14B main ->", a.model_info("Qwen/Qwen3-14B").sha)
print("tokenizer.json", t.lfs.sha256)
PY
  if [ "$PREFLIGHT" = 1 ]; then echo "PREFLIGHT_ONLY: the Mac checks passed; nothing launched" >&2; exit 0; fi
fi

# ---- the job ----------------------------------------------------------------------------
# Values fixed on the Mac, quoted for the job shell.
PRE=$(printf 'O=%q\nSRC=%q\nAWQ=%q\nREV=%q\nTOK_SHA=%q\nFP=%q\nDATA_BYTES=%q\nN_PARAMS=%q\nN_TENSORS=%q\nMANIFEST=%q' \
  "$O" "$SRC" "$AWQ" "$REV" "$TOK_SHA" "$FP" "$DATA_BYTES" "$N_PARAMS" "$N_TENSORS" "$MANIFEST")

# Quoted heredoc: nothing below expands on the Mac. `read -d ''` rather than
# $(cat <<'JOB'): the Mac's /bin/bash is 3.2, which mis-parses a heredoc inside $( ).
# ops/run.py prepends `set -euo pipefail` and runs the whole as ['bash', '-lc', script].
IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
check() {  # check <dump> <model as mmlu was given it>
  grep -qxF "# model=$2 [reference checkpoint]" "$1"
  grep -qxF '# dtype=f16' "$1"
  grep -qxF '# limit=census' "$1"
  grep -qxF '# alloc=flat, 100..1534 per subject, 14042 questions' "$1"
  grep -qxF '# config=none' "$1"
  grep -qxF '# arithmetic=dense reconstruction' "$1"
  grep -qxF '# kv=f16' "$1"
  test "$(tail -n 1 "$1")" = "# end fingerprint=$FP questions=14042"
  echo "dump ok: $1"
}
# Runs as an `if` condition, where `set -e` is off inside the function: every step
# returns explicitly.
stage_c() {
  mkdir -p "$AWQ" || return 1
  printf '%s\n' "$MANIFEST" > "$AWQ-manifest.txt" || return 1
  t0=$(date -u +%s)
  # Bounded, so that a stalled read on the FUSE mount costs C and not B: 600 s a
  # file (the largest is 1.56 GB; 28 GB took ~3.5 min on 2026-08-10) and 20 min in
  # all. Without the bound the copy would hang to the job timeout and B, which is
  # scored after it, would never run.
  while read -r name size; do
    [ $(($(date -u +%s) - t0)) -lt 1200 ] || { echo "STAGE FAIL: the copy passed 20 min, at $name"; return 1; }
    timeout -k 30 600 cp "$SRC/$name" "$AWQ/$name" || { echo "STAGE FAIL: cp $name failed or passed 600 s"; return 1; }
    got=$(stat -c %s "$AWQ/$name") || return 1
    [ "$got" = "$size" ] || { echo "STAGE FAIL: $name is $got B on local disk, $size B listed"; return 1; }
  done < "$AWQ-manifest.txt"
  t1=$(date -u +%s)
  echo "copied 40 files, $(du -sb "$AWQ" | cut -f1) B, in $((t1 - t0)) s"
  python3 - "$AWQ" "$AWQ-manifest.txt" "$DATA_BYTES" "$N_PARAMS" "$N_TENSORS" "$TOK_SHA" <<'PY' || return 1
# stdlib only, no pathlib (python3-minimal lacked ntpath once, d958070)
import hashlib, json, os.path, struct, sys
d, manifest = sys.argv[1], sys.argv[2]
want_data, want_params, want_tensors = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
want_tok = sys.argv[6]
fail = []
listed = {}
with open(manifest) as f:
    for line in f:
        if line.strip():
            name, size = line.split()
            listed[name] = int(size)
for name, size in sorted(listed.items()):
    got = os.path.getsize(os.path.join(d, name))
    if got != size:
        fail.append("%s: %d B on disk, %d B listed" % (name, got, size))
shards = sorted(n for n in listed if n.startswith("model-") and n.endswith(".safetensors"))
with open(os.path.join(d, "model.safetensors.index.json")) as f:
    idx = json.load(f)
total = idx["metadata"]["total_size"]
wm = idx["weight_map"]
if sorted(set(wm.values())) != shards:
    fail.append("index.json maps tensors to %d shard names, the explicit list holds %d"
                % (len(set(wm.values())), len(shards)))
data = params = 0
seen = {}
dups = []
for s in shards:
    p = os.path.join(d, s)
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    end = 0
    for k, v in h.items():
        a, b = v["data_offsets"]
        end = max(end, b)
        data += b - a
        c = 1
        for x in v["shape"]:
            c *= x
        params += c
        if k in seen:
            dups.append("%s in %s and %s" % (k, seen[k], s))
        seen[k] = s
    if os.path.getsize(p) != 8 + n + end:
        fail.append("%s: %d B on disk, its header ends the data at %d"
                    % (s, os.path.getsize(p), 8 + n + end))
if dups:
    fail.append("%d tensors stored twice, first: %s" % (len(dups), dups[0]))
moved = [k for k in set(seen) | set(wm) if seen.get(k) != wm.get(k)]
if moved:
    fail.append("shard headers and index.json place %d tensors differently, first: %s"
                % (len(moved), sorted(moved)[0]))
if not (total == data == want_data):
    fail.append("tensor bytes: index.json %d, headers %d, expected %d" % (total, data, want_data))
if params != want_params:
    fail.append("weights: %d in the headers, expected %d" % (params, want_params))
if len(seen) != want_tensors:
    fail.append("tensors: %d in the headers, expected %d" % (len(seen), want_tensors))
with open(os.path.join(d, "config.json")) as f:
    cfg = json.load(f)
want_cfg = {"hidden_size": 5120, "intermediate_size": 17408, "num_hidden_layers": 40,
            "num_attention_heads": 40, "num_key_value_heads": 8, "head_dim": 128,
            "vocab_size": 151936, "tie_word_embeddings": False}
for k in sorted(want_cfg):
    if cfg.get(k) != want_cfg[k]:
        fail.append("config.json %s = %r, expected %r" % (k, cfg.get(k), want_cfg[k]))
if "quantization_config" in cfg:
    fail.append("config.json carries a quantization_config: not a dequantized checkpoint")
with open(os.path.join(d, "tokenizer.json"), "rb") as f:
    tok = hashlib.sha256(f.read()).hexdigest()
if tok != want_tok:
    fail.append("tokenizer.json sha256 %s, expected %s" % (tok, want_tok))
print("staged: %d files, %d shards, %d tensors, %d weights, %d tensor bytes (index.json %d),"
      " %d B of shard headers" % (len(listed), len(shards), len(seen), params, data, total,
                                  sum(listed[s] for s in shards) - data))
for m in fail:
    print("STAGE FAIL: " + m)
sys.exit(1 if fail else 0)
PY
  # A record, not a control: no reference hash exists (the listing gives sizes, and
  # the 2026-08-10 job recorded none). It fixes which bytes arm C scored.
  ( cd "$AWQ" && sha256sum model-000*-of-00033.safetensors ) > "$AWQ.sha256" || return 1
  cp "$AWQ.sha256" "$AWQ-manifest.txt" "$O/" || return 1
  echo "sha256 of the 33 local shards in $(($(date -u +%s) - t1)) s: $O/$(basename "$AWQ").sha256"
}
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee "$O/gpu.txt" || true
df -h / /scratch | tee "$O/disk.txt" || true
echo '== oracle (hard rule 10) =='
oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle-cuda.txt" | tail -2
grep -q '^MATCH ' "$O/oracle-cuda.txt"                # oracle.rs:83; DIVERGENCE also exits 1
echo '== arm C staging: 40 files by explicit list, bucket mount -> local disk, then proved ==' ; date -u
if stage_c 2>&1 | tee "$O/stage-awq.txt"; then C_OK=1; else C_OK=0; echo 'arm C staging FAILED; B still runs' >&2; fi
date -u
echo '== arm B: Qwen3-14B f16 checkpoint, pinned revision, FULL split ==' ; date -u
LLVQ_MMLU_ALLOC=flat LLVQ_MMLU_DUMP="$O/mmlu-14b-f16-FULL.csv" mmlu "Qwen/Qwen3-14B@$REV" cuda 2>&1 | tee "$O/out-f16.txt" | tail -10
date -u
check "$O/mmlu-14b-f16-FULL.csv" "Qwen/Qwen3-14B@$REV"
if [ "$C_OK" = 1 ]; then
  echo '== arm C: AWQ w4 g128 dequantized to f16, local copy, FULL split ==' ; date -u
  LLVQ_MMLU_ALLOC=flat LLVQ_MMLU_DUMP="$O/mmlu-14b-awq-FULL.csv" mmlu "$AWQ" cuda 2>&1 | tee "$O/out-awq.txt" | tail -10
  date -u
  check "$O/mmlu-14b-awq-FULL.csv" "$AWQ"
fi
df -h / /scratch | tee -a "$O/disk.txt" || true
echo '== dumps ==' ; wc -l "$O"/mmlu-14b-*-FULL.csv ; ls -la "$O"
[ "$C_OK" = 1 ] || { echo 'arm C was not scored: staging failed, see stage-awq.txt' >&2; exit 1; }
JOB

if [ "$DRY" = 1 ]; then
  # What ops/run.py will hand to ['bash', '-lc', ...]: printed and parsed, not run.
  JOBSCRIPT="set -euo pipefail
$PRE
$BODY"
  printf '%s\n' "$JOBSCRIPT"
  printf '%s\n' "$JOBSCRIPT" | bash -n && echo "DRY_RUN: job script parses; nothing launched" >&2
  exit 0
fi

uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor l40sx1 --timeout 135m \
  --bucket "$BUCKET" --out-mount /out \
  --name census-14b-ref \
  "$PRE" "$BODY"
