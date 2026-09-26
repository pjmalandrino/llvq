#!/usr/bin/env bash
# DRAFT, NOT RUN. The post-publish smoke of the rebuilt CUDA image, for the 14B paper-2 chain.
# Destined for ops/jobs/image-smoke-14b.sh once the operator gives the go.
#
# What it proves, before any 14B job is billed on the new image:
#   1. the Space holds the commit it says (COMMIT file, first line = EXPECT_COMMIT, default
#      local HEAD), and the build log OF THAT SPACE SHA shows the four new steps and the push
#      (space-build-log.py, beside this script; checked on the Mac, $0);
#   2. the forward pass on this image and this card (hard rule 10: oracle first);
#   3. every binary of the two Dockerfile lists is on the target, `export` and `rowscale`
#      included, and both run (no-argument refusal, exit 1, their own message);
#   4. the three served configs are in the image, the 14B one byte for byte the committed file;
#   5. FUNCTIONAL=1 (default): `export` and `rowscale` do their job in the image, on the 4B DCLM
#      base (dclm-4b-2026-09-18/qwen3-4b-dclm.bin, 1,794,564,765 B, measured, bucket):
#        - export: "tensors identical bit for bit", "(36 from Int4G128)", and model.safetensors
#          at 8,044,981,648 B, the size of both 4B exports already in the bucket
#          (dclm-export-2026-09-19/, tetranu-export-2026-09-19/, measured, `hf buckets ls`);
#        - rowscale: the idempotence control of rowscale-controles-2026-09-19.txt, sigma 1.0 on
#          q_proj + gate_proj of layer 0 (4,096 + 9,728 = 13,824 rows), "0 scaled, 216
#          untouched, 36 int4 passed through", then `cmp` BYTE-IDENTICAL against the input.
#      That is the same control the 14B fold runs, on the same code, one size down.
#
# Not a measurement, so no prereg: it is the smoke of configs/README.md steps 2-3, a gate.
#
# 🚨 `export` is ALWAYS called as /usr/local/bin/export. ops/run.py runs the job as
# ['bash', '-lc', script], where `export` is the shell builtin (checked 2026-09-22).
#
# Cost, estimated (l40sx1, $1.80/h, `hf jobs hardware`):
#   pull ~2.5 min + oracle 0.6B ~0.5 + checks ~0.2                        = ~3.5 min minimal
#   + cp 1.79 GB off the mount ~0.5 + export 4B ~2-3 (Mac: 49 s at 4B, 80 s at 8B, measured;
#     8 vCPU here) + two sha256 of 8 GB / 1.8 GB ~1 + one fold + cmp ~1     = ~9 min functional
#   FUNCTIONAL=1: ~$0.27 [$0.21; $0.36], timeout 25m, worst $0.75
#   FUNCTIONAL=0: ~$0.11 [$0.09; $0.15], timeout 15m, worst $0.45
#   Peak RAM of the 4B export ~25 GB (half the 8B's 51.2 GB footprint, measured on the Mac,
#   dclm-8b-rowscales-2026-09-21-brut/export.txt), 62 GB on l40sx1.
#
# Writes to the bucket (/out): two jobs writing into the same bucket do not mount
# (ops/README.md, "Bucket mounting fails silently"), so this runs when no other job writes there.
#
# DRY_RUN=1 bash image-smoke-14b.sh   assembles the job script, parses it, launches nothing.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)                                 # before the cd below
BUILDLOG=${BUILDLOG:-$HERE/space-build-log.py}   # moves to ops/jobs/ WITH this script
REPO=${REPO:-$HOME/Documents/Pro/workspace/poc/llvq-commit-8b}   # the tree the image was published from
D=${RUN_DATE:-2026-09-22}
BUCKET=Pier-Jean/jobs-artifacts
SPACE=Pier-Jean/llvq-runner-cuda
OLD_SHA=a963a02010cec2d3c342ec52dd38f2dded06f2a1                   # the image before the rebuild
FUNCTIONAL=${FUNCTIONAL:-1}
O=/out/image-smoke-14b-$D
B=/out/dclm-4b-2026-09-18/qwen3-4b-dclm.bin
B_BYTES=1794564765
E_BYTES=8044981648
DRY=${DRY_RUN:-0}
if [ "$FUNCTIONAL" = 1 ]; then TIMEOUT=25m; else TIMEOUT=15m; fi

cd "$REPO"
# The commit the image was published from. Defaults to HEAD; set it to the published sha
# when the branch has moved on since `publish` (docs, launchers), or this refuses a good image.
EXPECT_COMMIT=${EXPECT_COMMIT:-$(git rev-parse HEAD)}
# The 14B config's sha256 as COMMITTED at that sha, not as the working copy holds it: the
# image carries the committed bytes, and a local edit would fail a good image in the job.
C14=configs/qwen3-14b-tetra-q5.json
if git cat-file -e "$EXPECT_COMMIT:$C14" 2>/dev/null; then
  CONF14_SHA=$(git show "$EXPECT_COMMIT:$C14" | shasum -a 256 | cut -d' ' -f1)
elif [ "$DRY" = 1 ]; then
  CONF14_SHA=$(shasum -a 256 "$C14" | cut -d' ' -f1)   # dry run only: not committed yet
else
  echo "refused: $C14 is not in commit $EXPECT_COMMIT" >&2; exit 1
fi

# ---- preflight on the Mac, $0 ----------------------------------------------------------
if [ "$DRY" != 1 ]; then
  # The build log of the Space's CURRENT sha, all five markers DONE. Right after `publish`
  # the sha moves at once while the stage can still read the previous build's
  # RUNTIME_ERROR, which the stage test below lets through; the log's own sha does not.
  test -f "$BUILDLOG" || { echo "refused: $BUILDLOG missing" >&2; exit 1; }
  uv run --quiet "$BUILDLOG" "/tmp/image-smoke-14b-$D-build.txt" \
    || { echo "refused: the build log does not show the rebuilt image (see above)" >&2; exit 1; }
  uv run --quiet --with huggingface_hub python - "$SPACE" "$OLD_SHA" "$EXPECT_COMMIT" <<'PY' \
    | tee "/tmp/image-smoke-14b-$D-provenance.txt"
import sys
from huggingface_hub import HfApi, hf_hub_download
space, old, want = sys.argv[1:4]
s = HfApi().space_info(space)
stage = s.runtime.stage if s.runtime else None
print("image", space, s.sha, s.last_modified, "stage", stage)
if s.sha == old:
    sys.exit(f"refused: the Space is still {old}, the image before the rebuild")
if stage in ("BUILDING", "BUILD_ERROR", "CONFIG_ERROR", "NO_APP_FILE"):
    sys.exit(f"refused: Space stage {stage}; the image is not built")
note = open(hf_hub_download(space, "COMMIT", repo_type="space", revision=s.sha)).read()
print(note, end="")
if note.splitlines()[0].strip() != want:
    sys.exit(f"refused: the Space COMMIT names {note.splitlines()[0]!r}, EXPECT_COMMIT is {want}")
if "Uploaded perimeter clean at upload time." not in note:
    sys.exit("refused: the Space COMMIT says the uploaded tree was dirty")
PY
  if [ "$FUNCTIONAL" = 1 ]; then
    REMOTE=$(hf buckets ls "hf://buckets/$BUCKET/dclm-4b-2026-09-18/" 2>/dev/null \
             | grep -v '^(empty)$' | awk '$NF ~ /qwen3-4b-dclm\.bin$/ {print $1}' || true)
    [ "$REMOTE" = "$B_BYTES" ] || { echo "refused: 4B base is ${REMOTE:-absent} B in the bucket" >&2; exit 1; }
  fi
  if hf buckets ls "hf://buckets/$BUCKET/image-smoke-14b-$D/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: image-smoke-14b-$D/ is not empty; pick another RUN_DATE" >&2; exit 1
  fi
fi

# ---- the job ----------------------------------------------------------------------------
PRE=$(printf 'O=%q\nB=%q\nB_BYTES=%q\nE_BYTES=%q\nCONF14_SHA=%q\nFUNCTIONAL=%q' \
      "$O" "$B" "$B_BYTES" "$E_BYTES" "$CONF14_SHA" "$FUNCTIONAL")

# Quoted heredoc, `read -d ''`: the Mac's /bin/bash is 3.2 (see ops/jobs/census-8b.sh).
IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee "$O/gpu.txt" || true
echo '== oracle (hard rule 10) =='
oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle-cuda.txt" | tail -2
echo '== both Dockerfile lists, on the target =='
for b in smoke ppl oracle mmlu gbench fusedrun chat seal export rowscale preflight matvec \
         rotbench graphbench planesbench nullkbench f1floorbench f1rankfloor; do
  test -x "/usr/local/bin/$b" || { echo "missing: /usr/local/bin/$b" >&2; exit 1; }
done
sha256sum /usr/local/bin/export /usr/local/bin/rowscale /usr/local/bin/seal | tee "$O/bin.sha256"
echo '== export and rowscale run: no argument, their own refusal, exit 1 =='
rc=0; /usr/local/bin/export > "$O/export-noarg.txt" 2>&1 || rc=$?
cat "$O/export-noarg.txt"; test "$rc" = 1
grep -qF 'give the path to a sealed .llvq model' "$O/export-noarg.txt"
rc=0; /usr/local/bin/rowscale > "$O/rowscale-noarg.txt" 2>&1 || rc=$?
cat "$O/rowscale-noarg.txt"; test "$rc" = 1
grep -qF 'usage: rowscale <in.bin|in.llvq> <out> <sigma.json>' "$O/rowscale-noarg.txt"
echo '== the served configs =='
for c in qwen3-4b-tetra-q5 qwen3-8b-tetra-q5 qwen3-14b-tetra-q5; do
  test -f "/usr/local/share/llvq/configs/$c.json"
done
sha256sum /usr/local/share/llvq/configs/*.json | tee "$O/configs.sha256"
test "$(sha256sum < /usr/local/share/llvq/configs/qwen3-14b-tetra-q5.json | cut -d' ' -f1)" = "$CONF14_SHA"
if [ "$FUNCTIONAL" = 1 ]; then
  echo '== export, 4B DCLM base, on local disk (never mmap from the mount) ==' ; date -u
  test "$(stat -c %s "$B")" = "$B_BYTES"
  cp "$B" /scratch/base.bin
  sha256sum /scratch/base.bin | tee "$O/base.sha256"
  t0=$(date +%s)
  /usr/local/bin/export /scratch/base.bin /scratch/e4b 2>&1 | tee "$O/export-4b.txt" | tail -12
  echo "export wall: $(( $(date +%s) - t0 )) s" | tee -a "$O/export-4b.txt"
  grep -q 'tensors identical bit for bit' "$O/export-4b.txt"
  grep -qF '(36 from Int4G128)' "$O/export-4b.txt"
  test "$(stat -c %s /scratch/e4b/model.safetensors)" = "$E_BYTES"
  sha256sum /scratch/e4b/model.safetensors | tee "$O/export-4b.sha256"
  ls -la /scratch/e4b | tee -a "$O/export-4b.txt"
  echo '== rowscale, the idempotence control: all-ones sigma on 13,824 rows ==' ; date -u
  python3 -c 'import json; print(json.dumps({"kind": "row_scales", "sigma": {
      "model.layers.0.self_attn.q_proj": [1.0] * 4096,
      "model.layers.0.mlp.gate_proj": [1.0] * 9728}}))' > /scratch/sigma-ones.json
  /usr/local/bin/rowscale /scratch/base.bin /scratch/ones.bin /scratch/sigma-ones.json 2>&1 | tee "$O/fold-ones.txt"
  grep -qxF '0 scaled, 216 untouched, 36 int4 passed through' "$O/fold-ones.txt"
  grep -q '^13824 row scales read' "$O/fold-ones.txt"
  cmp /scratch/base.bin /scratch/ones.bin
  echo 'ones control: BYTE-IDENTICAL' | tee -a "$O/fold-ones.txt"
fi
echo '== image smoke: PASS ==' | tee "$O/verdict.txt"; date -u
ls -la "$O"
JOB

if [ "$DRY" = 1 ]; then
  JOBSCRIPT="set -euo pipefail
$PRE
$BODY"
  printf '%s\n' "$JOBSCRIPT"
  printf '%s\n' "$JOBSCRIPT" | bash -n && echo "DRY_RUN: job script parses; nothing launched (timeout $TIMEOUT)" >&2
  exit 0
fi

uv run ops/run.py bench \
  --image "hf.co/spaces/$SPACE" \
  --flavor l40sx1 --timeout "$TIMEOUT" \
  --bucket "$BUCKET" --out-mount /out \
  --name image-smoke-14b \
  "$PRE" "$BODY"
