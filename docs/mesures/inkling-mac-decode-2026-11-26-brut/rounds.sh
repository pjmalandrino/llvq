#!/bin/bash
# Five alternated rounds of prereg inkling-mac-decode-2026-11-26: reference (the current default) then variant.
# usage: rounds.sh <prefix> <n rounds> <variant args for decode_arm.sh...>     (from the worktree root)
set -u
O=docs/mesures/inkling-mac-decode-2026-11-26-brut
p=$1; n=$2; shift 2
for r in $(seq 1 $n); do
  $O/decode_arm.sh $p-ref$r || exit 1
  $O/decode_arm.sh $p-var$r "$@" || exit 1
done
echo "rounds $p: all done"
