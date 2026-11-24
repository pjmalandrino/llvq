# Prereg: Inkling 2 Mini, GSM8K-200 through the fused CUDA stack (2026-11-24)

Written before the run. Not edited after its timestamp; deviations go in `-ECARTS.md` beside it.

## Question

The fused stack decodes at 72.2 tok/s on an L40S and 36.8 on an L4 (*measured*, 32 tokens after a 32-token prompt),
and its perplexity sits +0.413 % [+0.181, +0.646] above the path with no switch when the int4 head is read as stored.
Does it keep the GSM8K count of the sealed values, 191 of 200 (job `6b0347de130a00b5aa8100d2`, the path with no
switch), and what does it decode at over a whole benchmark?

## Object and method

The values of the run of 2026-11-22: experts of the 24 layers in lattice codes, the two int4 g64 tables of the sealed
file, every other tensor from the text-only checkpoint. `inkgsm run --stack fused`: experts in four launches with the
one-launch activation, fused glue, fused attention products and scores, the output head read as int4 (not decoded);
the embedding table is decoded to f32 at load, so the memory is not the sealed object's. Same 200 problems, same
template at effort `none`, greedy, cap 1,536, a completion cut by the cap counted wrong. Job
`ops/jobs/inkling-gsm8k-card.sh` with `STACK=fused RETRY=fused`, on an `l40sx1`, or an `l4x1` if no L40S comes in 45
minutes. The prompt fingerprint must be `6d4d8fa2207c0146`.

The job's control 3 runs `inkrun` with `--moe-act candle`: the 32 reference tokens were recorded with candle's
activation.

## Decision rule

The count is read under the cap rule and paired, problem by problem, with the dump of 2026-11-22.
KEEPS: the exact two-sided McNemar test on the discordant pairs gives p > 0.05.
LOSES: p <= 0.05 with fewer right. The speed is the median step over the 200 problems, given with the card.
One run, one tirage: a difference of a few problems is not a finding either way.

## Signed prediction (the coordinator, before the run)

| quantity | predicted |
|---|---|
| right of 200 | 187 to 194 |
| discordant pairs against the dump of 2026-11-22 | 2 to 10 |
| verdict | KEEPS, probability 0.85 |
| median step, L40S | 0.0139 to 0.0155 s |
| the job ends without a fault on a first launch of `--stack fused` | probability 0.7 |

## Cost (*estimated*)

About 70,000 generated tokens. L40S: 35 minutes with setup, $1.05. L4: 50 minutes, $0.67. Deadline 3,000 s.
