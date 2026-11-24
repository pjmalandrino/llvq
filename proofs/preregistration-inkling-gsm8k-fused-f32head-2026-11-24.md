# Prereg: Inkling 2 Mini, GSM8K-200 through the fused CUDA stack with the head decoded to f32 (2026-11-24)

Written before the run. Not edited after its timestamp; deviations go in `-ECARTS.md` beside it.

## Question

The fused stack with the int4 head read as stored gave 186 of 200 (job `6b04efaaca8b51043c3ae619`), against 191 of 200
through candle with the head decoded to f32 (job `6b0347de130a00b5aa8100d2`): 7 lost, 2 gained, exact McNemar p = 0.180,
not separated. The int4 head in the kernel is the one arm with a measured perplexity cost (+0.413 % against +0.027 %).
Does the fused stack with the head of the candle arm return to the candle count?

## Object and method

`inkgsm run --stack fused-f32head`: the fused stack of the run above (experts in four launches with the one-launch
activation, fused glue, fused attention products and scores, routing on the device), with the output head decoded to
f32 at load as the run of 2026-11-22 read it. Same values, same 200 problems, same template at effort `none`, greedy,
cap 1,536, a completion cut by the cap counted wrong. Job `ops/jobs/inkling-gsm8k-card.sh` with
`STACK=fused-f32head RETRY=f32head`, on an `l40sx1`, or an `l4x1` if no L40S comes in 45 minutes. The prompt
fingerprint must be `6d4d8fa2207c0146`.

## Decision rule

The count is read under the cap rule and paired, problem by problem, with the two dumps.
HEAD CLEARED: the exact two-sided McNemar test against the candle dump gives p > 0.05 and against the fused int4
dump gives p > 0.05. That is the expected reading and it places nothing: 200 problems do not separate five.
HEAD NAMED: p <= 0.05 against the fused int4 dump with more right here.
STACK NAMED: p <= 0.05 against the candle dump with fewer right here.
One run, one draw. A count between 186 and 191 is reported as such, with its two tables of discordant pairs.

## Signed prediction (the coordinator, before the run)

| quantity | predicted |
|---|---|
| right of 200 | 186 to 192 |
| discordant pairs against the candle dump | 2 to 9 |
| discordant pairs against the fused int4 dump | 3 to 10 |
| reading | HEAD CLEARED and nothing placed, probability 0.8 |
| median step, L40S | 0.0145 to 0.0175 s (the f32 head is candle's matmul) |

## Cost (*estimated*)

About 75,000 generated tokens. L40S: 35 minutes with setup, $1.05. L4: 55 minutes, $0.75. Deadline 3,000 s.
