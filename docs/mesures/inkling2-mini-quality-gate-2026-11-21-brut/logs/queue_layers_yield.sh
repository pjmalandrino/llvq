#!/bin/bash
# Stage C of 2026-11-22: the layers outside the loop, encoded while other sessions may need the guard's lock.
# Same steps as queue_layers.sh, with three differences:
#   the encode of a layer runs in slices of experts (extq is resumable an expert), with a pause after every heavy
#   step so that a job waiting on the lock (heavy.sh polls every 30 s) can take it;
#   threads come from THREADS (default 8);
#   HATS=f16 stores the reconstructions as float16 once the codes are verified against the float32 ones.
# usage: queue_layers_yield.sh "<layers>" ["<set spec for the arms>" ["<rungs>"]]   (no set spec = no arms)
set -euo pipefail
LAYERS="$1"; SET="${2:-}"; RUNGS="${3:-IQ1_S IQ1_M IQ2_XXS Q2_K_XL}"
THREADS="${THREADS:-8}"; SLICE="${SLICE:-32}"; YIELD="${YIELD:-45}"; HATS="${HATS:-f32}"
W=/Users/pjmalandrino/Documents/Pro/workspace/poc/llvq/.claude/worktrees/inkling-mini
D=$HOME/llvq-inkling-mini
H=$HOME/.llvq-guard/heavy.sh
X=$W/target/release/extq
export INKLING_THREADS=$THREADS LLVQ_THREADS=$THREADS RAYON_NUM_THREADS=$THREADS
cd "$W/ops/inkling"

for L in $LAYERS; do
  LD=$(printf "%s/layers/L%02d" "$D" "$L")
  if [ -f "$LD/VERIFIED" ]; then echo "== layer $L already verified, skipped"; continue; fi
  echo "== $(date '+%F %T') layer $L: Hessians"
  "$H" "inkling: prep layer $L" uv run prep_layer.py "$L" > "$D/logs/prep_L$L.log" 2>&1 || { echo "prep failed"; tail -15 "$D/logs/prep_L$L.log"; exit 3; }
  tail -1 "$D/logs/prep_L$L.log"; sleep "$YIELD"
  : > "$D/logs/extq_L$L.log"
  for a in $(seq 0 "$SLICE" 127); do
    b=$((a + SLICE)); [ $b -gt 128 ] && b=128
    echo "== $(date '+%F %T') layer $L: encode experts $a..$b"
    "$H" "inkling-extq: layer $L experts $a..$b" "$X" "$LD" --layer "$L" --experts "$a..$b" >> "$D/logs/extq_L$L.log" 2>&1 || { echo "extq failed"; tail -15 "$D/logs/extq_L$L.log"; exit 4; }
    sleep "$YIELD"
  done
  tail -1 "$D/logs/extq_L$L.log"
  echo "== $(date '+%F %T') layer $L: verify codes against reconstructions"
  "$H" "inkling-extq: verify layer $L" "$X" verify "$LD" > "$D/logs/verify_L$L.log" 2>&1 || { echo "verify failed"; tail -15 "$D/logs/verify_L$L.log"; exit 5; }
  tail -1 "$D/logs/verify_L$L.log"
  grep -q "128 experts, 0 differing weights" "$D/logs/verify_L$L.log" || { echo "verify did not cover 128 experts"; exit 5; }
  date '+%F %T' > "$LD/VERIFIED"
  rm -f "$LD/h13.f64" "$LD/h2.f64" "$LD/w13.f32" "$LD/w2.f32"
  if [ "$HATS" = "f16" ]; then
    uv run hats_f16.py "$LD" --drop-f32 > "$D/logs/f16_L$L.log" 2>&1 || { echo "f16 failed"; tail -5 "$D/logs/f16_L$L.log"; exit 8; }
    tail -1 "$D/logs/f16_L$L.log"
  fi
  echo "== $(date '+%F %T') layer $L done; inputs removed"; df -h /Users/pjmalandrino | tail -1 | awk '{print "   disk free:", $4}'
  sleep "$YIELD"
done

[ -z "$SET" ] && { echo "== $(date '+%F %T') stage done, no arms asked"; exit 0; }
echo "== $(date '+%F %T') arms on {$SET}"
"$H" "inkling: arm tetra $SET" uv run evalkl.py arm "$SET=dir:$D/layers" --name "tetra_$SET" --device mps > "$D/logs/arm_tetra_$SET.log" 2>&1 || { echo "tetra arm failed"; tail -15 "$D/logs/arm_tetra_$SET.log"; exit 6; }
tail -1 "$D/logs/arm_tetra_$SET.log"; sleep "$YIELD"
for R in $RUNGS; do
  n=$(echo "$R" | tr 'A-Z' 'a-z' | tr -d '_')
  "$H" "inkling: arm $R $SET" uv run evalkl.py arm "$SET=gguf:$D/gguf/Inkling-2-Mini-UD-$R.gguf" --name "${n}_$SET" --device mps > "$D/logs/arm_${n}_$SET.log" 2>&1 || { echo "arm $R failed"; tail -15 "$D/logs/arm_${n}_$SET.log"; exit 7; }
  tail -1 "$D/logs/arm_${n}_$SET.log"; sleep "$YIELD"
done
echo "== $(date '+%F %T') stage done for {$SET}"
