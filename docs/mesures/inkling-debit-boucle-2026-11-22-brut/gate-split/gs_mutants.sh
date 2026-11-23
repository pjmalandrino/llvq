#!/bin/bash
# Mutants of the gate and router split (glue.rs of inkling-w3a), each against tests/inkling_glue.rs.
W=$WORKTREE
G=llvq-llm/src/inkling/glue.rs; M=$HOME/llvq-inkling-mini/lot4-coord/night/mutant1.sh
$M $W $G gm1_message_factor_in_down '    for (uint c = lane; c < h; c += 32u) { float v = m[c] * im; acc = acc + row[h + c] * v; }
    acc = simd_sum(acc);
    if (lane == 0) { out[s] = acc; }' '    for (uint c = lane; c < h; c += 32u) { float v = m[c] * ix; acc = acc + row[h + c] * v; }
    acc = simd_sum(acc);
    if (lane == 0) { out[s] = acc; }' inkling_glue
$M $W $G gm2_every_gate_reads_the_first_projection '    device const float *G = g + j * rank;' '    device const float *G = g;' inkling_glue
$M $W $G gm3_one_norm_for_every_row '    device const float *X = x + j * h;
    uint ni = a.n_norms == 3u ? j : 0u;' '    device const float *X = x + j * h;
    uint ni = 0u;' inkling_glue
$M $W $G gm4_router_reads_shifted_input '    for (uint c = lane; c < h; c += 32u) { acc = acc + row[c] * xn[c]; }' '    for (uint c = lane; c < h; c += 32u) { acc = acc + row[c] * xn[(c + 32u) % h]; }' inkling_glue
$M $W $G gm5_stage_a_hands_the_row_factor_twice '            out[3u * h + 2u * j + 1u] = im;' '            out[3u * h + 2u * j + 1u] = ix;' inkling_glue
$M $W $G gm6_post_stage_a_hands_the_row_factor_twice '            out[2u * h + 1u] = im;' '            out[2u * h + 1u] = ix;' inkling_glue
echo "end $(date '+%T')"
