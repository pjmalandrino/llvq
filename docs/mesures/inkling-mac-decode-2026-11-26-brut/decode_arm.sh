#!/bin/bash
# One decode arm of prereg inkling-mac-decode-2026-11-26: 32-token prompt, 128 timed steps (--n 129), the served
# stack, plus the arm's flags. ENV=<VAR=value> before the label sets an environment variable for the run only.
# usage: decode_arm.sh <label> [ENV=VAR=value] <extra inkrun flags...>      (from the worktree root)
set -u
O=docs/mesures/inkling-mac-decode-2026-11-26-brut; R=$O/run_quiet.sh
D=$HOME/llvq-inkling-mini
T=$D/lot1-b/tables/unembed.q4g64.llvqraw
STACK="--glue fused --attn fused+scores --head int4 --head-table $T"
label=$1; shift
envs=()
while [ $# -gt 0 ] && [ "${1#ENV=}" != "$1" ]; do envs+=("${1#ENV=}"); shift; done
while :; do
  env "${envs[@]+"${envs[@]}"}" $R $O "$label" --n 129 --tokens $D/lot4-coord/tokens.json $STACK "$@"; rc=$?
  [ $rc = 4 ] || break; echo "redo $label"
done
[ $rc = 0 ] && echo "${envs[*]+"${envs[*]}"} $*" > "$O/$label.arm"
exit $rc
