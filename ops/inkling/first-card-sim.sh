#!/usr/bin/env bash
# Run the body of ops/jobs/inkling-first-card.sh on the Mac, on fake inputs, with stand-ins for the card's tools.
# $0, no network, no GPU, a few MB of disk. It catches the packaging errors a first run dies on: a path the body
# reads that the staging does not write, an archive that does not extract to the layout inkrun reads, a manifest
# that does not check, a count that is wrong, a summary that does not parse inkrun's report.
#
#   bash ops/inkling/first-card-sim.sh [report.json]          green: the body runs to its end and exits 0
#   MUTANT=drop-code bash ops/inkling/first-card-sim.sh       one codes file missing from L05.tar: must fail
#   MUTANT=layers-23 bash ops/inkling/first-card-sim.sh       L23 without VERIFIED: inkrun's stand-in must refuse
#
# [report.json] is the report a real `inkrun` wrote (the Mac's Metal run); the stand-in copies it as the card's, so the
# summary is exercised on the real schema. Without it a minimal report of the same keys is used.
# What it does not check: GNU tar and sha256sum on the image (bsdtar and shasum stand in), the bucket mount, the card.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
SIM=${SIM_DIR:-$HOME/llvq-inkling-mini/lot2-u/sim}
REPORT=${1:-}
MUTANT=${MUTANT:-}
rm -rf "$SIM"
mkdir -p "$SIM/bin" "$SIM/out/inkling2-mini-inputs-2026-11-22/layers" "$SIM/scratch" "$SIM/src"
IN=$SIM/out/inkling2-mini-inputs-2026-11-22
D=${INKLING_DATA:-$HOME/llvq-inkling-mini}

# fake inputs, written as ops/inkling/stage-first-card.sh writes the real ones
cp "$D/textonly/config.json" "$IN/config.json"
head -c 4096 /dev/zero > "$IN/model.safetensors"
cp "$D/lot2-u/prompt32.json" "$IN/tokens.json"
for i in $(seq 0 23); do
  l=$(printf L%02d "$i")
  mkdir -p "$SIM/src/$l/codes"
  [ "$MUTANT" = layers-23 ] && [ "$l" = L23 ] || echo "2026-11-22 sim" > "$SIM/src/$l/VERIFIED"
  for e in $(seq 0 127); do printf 'codes %s %03d\n' "$l" "$e" > "$SIM/src/$l/codes/$(printf e%03d "$e").llvqrec"; done
done
( cd "$SIM/src" && for i in $(seq 0 23); do l=$(printf L%02d "$i"); shasum -a 256 "$l/VERIFIED" "$l"/codes/e*.llvqrec 2>/dev/null || true; done ) > "$IN/codes.sha256"
for i in $(seq 0 23); do
  l=$(printf L%02d "$i")
  files=("$l/codes")
  [ -f "$SIM/src/$l/VERIFIED" ] && files=("$l/VERIFIED" "${files[@]}")
  if [ "$MUTANT" = drop-code ] && [ "$l" = L05 ]; then rm "$SIM/src/$l/codes/e077.llvqrec"; fi
  COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata --format ustar -cf "$IN/layers/$l.tar" -C "$SIM/src" "${files[@]}"
done
sha() { shasum -a 256 "$1" | cut -d' ' -f1; }

# stand-ins for the card's tools
cat > "$SIM/bin/nvidia-smi" <<'EOF'
#!/bin/bash
echo "SIM L4, 8.9, 23034 MiB, 0 MiB, 550.00"
EOF
cat > "$SIM/bin/oracle" <<'EOF'
#!/bin/bash
echo "oracle (sim) $*"; echo "MATCH"
EOF
cat > "$SIM/bin/timeout" <<'EOF'
#!/bin/bash
shift; exec "$@"
EOF
cat > "$SIM/bin/sha256sum" <<'EOF'
#!/bin/bash
exec shasum -a 256 "$@"
EOF
cat > "$SIM/bin/stat" <<'EOF'
#!/bin/bash
if [ "$1" = -c ] && [ "$2" = %s ]; then exec /usr/bin/stat -f %z "$3"; fi
exec /usr/bin/stat "$@"
EOF
cat > "$SIM/bin/inkrun" <<'EOF'
#!/bin/bash
# inkrun's argument contract, and its --require-layers refusal, on the extracted layout
set -euo pipefail
CONFIG= CK= LAYERS= TOKENS= DEV= N= REQ= OUT=
while [ $# -gt 0 ]; do
  case $1 in
    --config) CONFIG=$2;; --checkpoint) CK=$2;; --layers) LAYERS=$2;; --tokens) TOKENS=$2;; --device) DEV=$2;;
    --n) N=$2;; --require-layers) REQ=$2;; --out) OUT=$2;; *) echo "inkrun (sim): unknown option $1" >&2; exit 2;;
  esac
  shift 2
done
for v in CONFIG CK LAYERS TOKENS DEV N REQ OUT; do eval "[ -n \"\$$v\" ]" || { echo "inkrun (sim): $v missing" >&2; exit 2; }; done
[ "$DEV" = cuda ] || { echo "inkrun (sim): --device $DEV" >&2; exit 2; }
for f in "$CONFIG" "$CK" "$TOKENS"; do test -f "$f" || { echo "inkrun (sim): no $f" >&2; exit 1; }; done
n=0
for i in $(seq 0 23); do
  l=$(printf L%02d "$i")
  if [ -f "$LAYERS/$l/VERIFIED" ]; then
    c=$(ls "$LAYERS/$l/codes" | wc -l | tr -d ' ')
    [ "$c" = 128 ] || { echo "inkrun (sim): $l has $c codes files" >&2; exit 1; }
    n=$((n + 1))
  fi
done
[ "$n" -ge "$REQ" ] || { echo "Error: $n layers resident, --require-layers $REQ" >&2; exit 1; }
if [ -n "${SIM_REPORT:-}" ]; then cp "$SIM_REPORT" "$OUT"; else
python3 -I - "$OUT" <<'PY'
import json, sys
m = {"kind": "sim", "used": 1}
json.dump({"bin": "inkrun", "device": "cuda", "prompt_tokens": 32, "new_tokens": 32, "resident_layers": list(range(24)),
  "native_layers": [], "upload": [], "memory": {"before_load": m, "after_dense": m, "after_resident": m, "end": m},
  "seconds_dense_load": 1.0, "seconds_resident_load": 1.0, "seconds_prefill": 1.0, "step_seconds": [0.1] * 31,
  "step_seconds_median": 0.1, "tokens_per_second_median_step": 10.0, "lattice_launches_prefill": 1,
  "lattice_launches_per_step": [1920] * 31, "tokens": [1] * 32, "margins": [1.0] * 32, "min_margin": 1.0,
  "reference": [1] * 32, "equal_prefix": 32, "first_mismatch": None, "all_equal": True}, open(sys.argv[1], "w"))
PY
fi
echo "inkrun (sim): $n layers, report $OUT" >&2
EOF
chmod +x "$SIM/bin/"*

# the job script exactly as the launcher would send it, with /out moved to the sim directory
JOB=$(cd "$REPO" && PRINT_JOB=1 DRY_RUN=1 bash ops/jobs/inkling-first-card.sh)
JOB=${JOB//\/out\//$SIM/out/}
JOB=$(printf '%s\n' "$JOB" | sed -e "s|^TEXTONLY_BYTES=.*|TEXTONLY_BYTES=4096|" \
  -e "s|^TEXTONLY_SHA=.*|TEXTONLY_SHA=$(sha "$IN/model.safetensors")|" \
  -e "s|^CONFIG_SHA=.*|CONFIG_SHA=$(sha "$IN/config.json")|" \
  -e "s|^TOKENS_SHA=.*|TOKENS_SHA=$(sha "$IN/tokens.json")|" \
  -e "s|^MANIFEST_SHA=.*|MANIFEST_SHA=$(sha "$IN/codes.sha256")|")
set +e
env -i HOME="$HOME" PATH="$SIM/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin" SCRATCH="$SIM/scratch" SIM_REPORT="$REPORT" \
  bash -c "$JOB" > "$SIM/job.log" 2>&1
RC=$?
set -e
tail -15 "$SIM/job.log"
echo "sim: job body exit $RC (mutant: ${MUTANT:-none})"
exit $RC
