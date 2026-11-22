#!/bin/bash
# Stage B of the night of 2026-11-21: for each layer, Hessians, lattice encode, code check, clean-up;
# then the evaluation arms on a cumulative layer set.
# usage: queue_layers.sh "<layers to encode>" "<set spec for the arms, e.g. 6-8>" ["<rungs, default all four>"]
# Every heavy step goes through the guard's lock, one at a time. Fails fast and is resumable:
# a layer whose reconstructions are verified is skipped.
set -euo pipefail
LAYERS="$1"; SET="$2"; RUNGS="${3:-IQ1_S IQ1_M IQ2_XXS Q2_K_XL}"
W=/Users/pjmalandrino/Documents/Pro/workspace/poc/llvq/.claude/worktrees/inkling-mini
D=$HOME/llvq-inkling-mini
H=$HOME/.llvq-guard/heavy.sh
X=$W/target/release/extq
export INKLING_THREADS=10 LLVQ_THREADS=10 RAYON_NUM_THREADS=10
cd "$W/ops/inkling"

for L in $LAYERS; do
  LD=$(printf "%s/layers/L%02d" "$D" "$L")
  if [ -f "$LD/VERIFIED" ]; then echo "== layer $L already verified, skipped"; continue; fi
  echo "== $(date '+%F %T') layer $L: Hessians"
  "$H" "inkling: prep layer $L" uv run prep_layer.py "$L" > "$D/logs/prep_L$L.log" 2>&1 || { echo "prep failed"; tail -15 "$D/logs/prep_L$L.log"; exit 3; }
  tail -2 "$D/logs/prep_L$L.log"
  echo "== $(date '+%F %T') layer $L: encode"
  "$H" "inkling-extq: layer $L" "$X" "$LD" --layer "$L" > "$D/logs/extq_L$L.log" 2>&1 || { echo "extq failed"; tail -15 "$D/logs/extq_L$L.log"; exit 4; }
  tail -3 "$D/logs/extq_L$L.log"
  echo "== $(date '+%F %T') layer $L: verify codes against reconstructions"
  "$H" "inkling-extq: verify layer $L" "$X" verify "$LD" > "$D/logs/verify_L$L.log" 2>&1 || { echo "verify failed"; tail -15 "$D/logs/verify_L$L.log"; exit 5; }
  tail -2 "$D/logs/verify_L$L.log"
  date '+%F %T' > "$LD/VERIFIED"
  rm -f "$LD/h13.f64" "$LD/h2.f64" "$LD/w13.f32" "$LD/w2.f32"
  echo "== $(date '+%F %T') layer $L done; inputs removed"; df -h /Users/pjmalandrino | tail -1 | awk '{print "   disk free:", $4}'
done

echo "== $(date '+%F %T') arms on {$SET}"
"$H" "inkling: arm tetra $SET" uv run evalkl.py arm "$SET=dir:$D/layers" --name "tetra_$SET" --device mps > "$D/logs/arm_tetra_$SET.log" 2>&1 || { echo "tetra arm failed"; tail -15 "$D/logs/arm_tetra_$SET.log"; exit 6; }
tail -1 "$D/logs/arm_tetra_$SET.log"
for R in $RUNGS; do
  n=$(echo "$R" | tr 'A-Z' 'a-z' | tr -d '_')
  "$H" "inkling: arm $R $SET" uv run evalkl.py arm "$SET=gguf:$D/gguf/Inkling-2-Mini-UD-$R.gguf" --name "${n}_$SET" --device mps > "$D/logs/arm_${n}_$SET.log" 2>&1 || { echo "arm $R failed"; tail -15 "$D/logs/arm_${n}_$SET.log"; exit 7; }
  tail -1 "$D/logs/arm_${n}_$SET.log"
done
echo "== $(date '+%F %T') stage done for {$SET}"
