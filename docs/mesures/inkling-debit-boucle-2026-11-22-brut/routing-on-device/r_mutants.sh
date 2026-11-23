#!/bin/bash
# Mutants of the routing kept on the device (glue.rs of inkling-w3a), each against tests/inkling_wave2_moe.rs.
W=$WORKTREE
F=llvq-llm/src/inkling/glue.rs; M=$HOME/llvq-inkling-mini/lot4-coord/night/mutant1.sh
$M $W $F rw1_weights_reversed '    out[k + i] = as_type<uint>(top[e + i]);' '    out[k + i] = as_type<uint>(top[e + (k - 1u - i)]);' inkling_wave2_moe
$M $W $F rw2_ids_off_by_one '    out[i] = uint(top[e + k + 1u + i]);' '    out[i] = uint(top[e + k + i]);' inkling_wave2_moe
echo "end $(date '+%T')"
