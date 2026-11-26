#!/bin/bash
cd "$(dirname "$0")"
while [ ! -f run3.done ]; do sleep 10; done
run(){ venv/bin/python fwd.py "$@" > log_$3.txt 2>&1; echo "$3 $(tail -1 log_$3.txt)"; }
run 4 512 Cnew_o0 0.085 0.010 1 0
DENSE_RE='mixer\.in_proj|mixer_proj|shared_experts|c_q|gate_proj|W_B' run 4 512 Cbig_o2048 0.085 0.010 1 2048
DENSE_RE='moe_gate' run 4 512 router010_o2048 0 0.010 1 2048
DENSE_RE='down_proj|up_proj' run 4 512 latent010_o2048 0 0.010 1 2048
SEED=77 run 4 512 C_s77_o2048 0.085 0.010 1 2048
echo done > run4.done
