#!/bin/bash
# Mutants of the message kernel (glue.rs) and of the activation kernel (moe_act.rs), worktree inkling-w3a.
W=$WORKTREE
M=$HOME/llvq-inkling-mini/lot4-coord/night/mutant1.sh
G=llvq-llm/src/inkling/glue.rs; A=llvq-llm/src/inkling/moe_act.rs
$M $W $G nm1_every_query_the_first '    bar_one(queries + j * h, src, a.n_states + 1u, h, a.inv_h, a.bar_eps, a.bar_scale, X, scr, tid, lane, sg);
    for (uint c = tid; c < h; c += GLUE_TG) { out[j * h + c] = X[c]; }' '    bar_one(queries, src, a.n_states + 1u, h, a.inv_h, a.bar_eps, a.bar_scale, X, scr, tid, lane, sg);
    for (uint c = tid; c < h; c += GLUE_TG) { out[j * h + c] = X[c]; }' inkling_glue
$M $W $G nm2_partial_replaced_by_a_state '    src[a.n_states] = partial;' '    src[a.n_states] = s0;' inkling_glue
$M $W $A am1_product_order '    float p = sg * s;
    float act = p * su;' '    float p = s * su;
    float act = sg * p;' inkling_moe_act metal --include-ignored
$M $W $A am2_caps_swapped '    float su = softcap(u, a.inv_cap_u, a.cap_u, a.zero);' '    float su = softcap(u, a.inv_cap_g, a.cap_g, a.zero);' inkling_moe_act metal --include-ignored
$M $W $A am3_precise_exp '    float e = fast::exp(-x);' '    float e = precise::exp(-x);' inkling_moe_act metal --include-ignored
$M $W $A am4_ieee_division '    return fast::divide(1.0f, d);' '    return 1.0f / d;' inkling_moe_act metal --include-ignored
$M $W $A am5_scaling_by_division '    float a = fma(x, inv_cap, zero);' '    float a = x / cap;' inkling_moe_act metal --include-ignored
echo "end $(date '+%T')"
