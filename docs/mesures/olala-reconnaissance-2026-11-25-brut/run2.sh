#!/bin/bash
cd "$(dirname "$0")"
run(){ venv/bin/python fwd.py "$@" > log_$3.txt 2>&1; echo "$3 $(tail -1 log_$3.txt)"; }
DENSE_RE='mixer\.in_proj' run 4 512 d_inproj 0 0.085 1
DENSE_RE='mixer_proj' run 4 512 d_mixerproj 0 0.085 1
DENSE_RE='shared_experts' run 4 512 d_shared 0 0.085 1
DENSE_RE='c_q|gate_proj|W_B' run 4 512 d_attn 0 0.085 1
DENSE_RE='down_proj|up_proj' run 4 512 d_latent 0 0.085 1
DENSE_RE='moe_gate' run 4 512 d_router 0 0.085 1
DENSE_RE='moe_gate' run 4 512 d_router010 0 0.010 1
SEED=77 run 4 512 exp085_s77 0.085 0 1
for off in 2048 4096 6144; do run 4 512 base_o$off 0 0 1 $off; run 4 512 exp085_o$off 0.085 0 1 $off; run 4 512 C_o$off 0.085 0.010 1 $off; done
