#!/bin/bash
# Mutants of the score-kernel wiring (layers.rs of inkling-w3a), each against tests/inkling_wave3_attn.rs.
W=$WORKTREE
F=llvq-llm/src/inkling/layers.rs; M=$HOME/llvq-inkling-mini/lot4-coord/night/mutant1.sh
$M $W $F sw1_row_position_early '                    pos: offset,' '                    pos: offset.saturating_sub(1),' inkling_wave3_attn
$M $W $F sw2_window_dropped '                let window = if self.sliding { Some(cfg.sliding_window_size) } else { None };
                let keys = kt.transpose(1, 2)?;' '                let window = None::<usize>;
                let keys = kt.transpose(1, 2)?;' inkling_wave3_attn
$M $W $F sw3_values_for_keys '                    keys: &keys,' '                    keys: &v,' inkling_wave3_attn
echo "end $(date '+%T')"
