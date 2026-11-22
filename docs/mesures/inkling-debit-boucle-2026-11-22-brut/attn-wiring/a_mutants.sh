#!/bin/bash
# Mutants of the attention wiring (layers.rs of inkling-w3a), each against tests/inkling_wave3_attn.rs.
W=$WORKTREE
F=llvq-llm/src/inkling/layers.rs; M=$HOME/llvq-inkling-mini/lot4-coord/night/mutant1.sh
$M $W $F aw1_kv_states_swapped 'AttnSlot::new(d, k.as_ref(), v.as_ref())?' 'AttnSlot::new(d, v.as_ref(), k.as_ref())?' inkling_wave3_attn
$M $W $F aw2_position_late 'stage_in(&d.w, &xq, &xk, &xv, offset, rings)?' 'stage_in(&d.w, &xq, &xk, &xv, offset + 1, rings)?' inkling_wave3_attn
$M $W $F aw3_prompt_reads_stale_states '            cache.settle_attn()?;
            let m = d.widen()?;' '            let m = d.widen()?;' inkling_wave3_attn
$M $W $F aw4_rollback_keeps_rings '            self.fused = None;
            self.attn = None;' '            self.fused = None;' inkling_wave3_attn
$M $W $F aw5_rings_not_filled 'AttnSlot::new(d, k.as_ref(), v.as_ref())?' 'AttnSlot::new(d, None, None)?' inkling_wave3_attn
echo "end $(date '+%T')"
