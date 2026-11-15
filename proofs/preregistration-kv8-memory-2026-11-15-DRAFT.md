# Prereg kv8-memory: the int8 decode cache with the binary16 one released, 4B (DRAFT, 2026-11-15)

Status: DRAFT, not timestamped. The lead rereads it, signs the predictions below and stamps it before any run.
Cost: about $1.66 (*estimated*), hard cap 75 min, $2.25. Budget: the operator's, to announce with the go.
Measured code: branch `dev/trois-gains-2026-11-15`, commit of gain 2 (to fill at the stamp). Job script
`ops/jobs/kv8-memory.sh`. Config `configs/qwen3-4b-tetra-e4-4e-v9-kv8-NOT-SERVED.json` (K8), against the served
`configs/qwen3-4b-tetra-e4-4e-v9.json` (V9).

A timestamped prereg is no longer edited. A deviation is written in `<name>-ECARTS.md`.

## Question

With the binary16 caches released at the hand-over, how much card memory does the int8 decode cache save after an
8,192-token prompt, at what decode speed and with what texts, under the eager and the chain prefill?
In the research arm the binary16 caches stayed allocated beside the int8 ones, so no saving was measured
(`docs/mesures/recherche-debit-kv8-local-2026-11-15.txt`). This informs the operator's memory lead; nothing is served.

## Setup

One L40S, the sealed 4B (sha256 886391a8). Each arm in its own process, under both samplers of
`ops/jobs/large-prefill-ft.sh` (process peak every 100 ms; epoch-stamped peak inside each `mark time-<l>` window).

```
(1) LLVQ_FUSED_LAYOUT=tetra48 LLVQ_EMBED=q4 LLVQ_ROT_SHARE=1 LLVQ_FUSE=0 LLVQ_PERS_4E=check:v9,v9k8 fusedrun S 2
(2) LLVQ_CONFIG=V9 prefillbench --form eager --long L --lengths 32,1024,4096,8191 --prompts P --dump f16-eager.jsonl S
(3) LLVQ_CONFIG=K8 the same, --dump int8-eager.jsonl
(4) LLVQ_CONFIG=V9 prefillbench --form chain --arm a12ipg32ft@8.10.8.20.6 --r 32 (same lengths, prompts) --dump f16-chain.jsonl S
(5) LLVQ_CONFIG=K8 the same, --dump int8-chain.jsonl
(6) LLVQ_DATASET_REV=740312ad... LLVQ_CONFIG=K8 gsm8k S cuda 200, then V9, then gsm8kpair
```

32 tokens is the shortest cut of the long prompt with its chat template; the 5-token figure stays the served door's.

## Controls

1. oracle MATCH; the sealed file and both configs at their sha256; no `LLVQ_*` in the job environment.
2. Step (1) stops the job unless v9 loads, v9k8's module is 8b53b1cd and its token symbol line is printed.
3. Every K8 process prints `MEASUREMENT ONLY` and the kv8 report line `each released after its layer's fill`.
4. Every V9 process runs the served 4E v9 module (252,757 bytes, sha256 1b422b62).
5. Tokens identical across rounds in every process (prefillbench fails otherwise).

## What gets published, and what does not get compared

Published: per arm and length, phase peak and the card memory after the rounds (`cuMemGetInfo`, the steady state);
time to first token and decode tok/s after it, medians and ranges, and K8 over V9 as a quotient of medians with its
envelope; texts identical of 47 for int8 against binary16 under each prefill, with each first divergence; GSM8K-200
correct counts, discordant pairs, McNemar p. Not compared: across jobs or cards; the margins at a divergence (not
read by this job).

## Decision rule

| reading | verdict |
|---|---|
| controls hold; steady state at 8,191 K8 <= V9 - 0.45 GB under both prefills; K8 phase peak <= V9's + 0.05 GB under both; decode K8/V9 >= 0.95 at every length; GSM8K-200 p >= 0.05 | SAVES: reported to the operator as a memory lead, with the local-memory decision |
| a control fails | VOID |
| controls hold; GSM8K p < 0.05 with K8 below | FAILS on quality |
| controls hold, any other bound missed | NOT SETTLED: the operator decides with the numbers |
| otherwise | not settled, operator decision |

## Signed prediction

To be signed by the lead before the stamp. Each: expected value, interval, reason in two lines, known flaw.

- Steady state at 8,191, eager: V9 ... GB, K8 ... GB.
- Steady state at 8,191, chain: V9 ... GB, K8 ... GB.
- Phase peak at 8,191, each arm: ...
- Decode K8/V9 at 32 and 8,191: ...
- Texts identical of 47, each prefill: ...
- GSM8K-200 discordant pairs: ...

## Reading table (lead's, filled after the run)

| quantity | predicted | measured | inside |
|---|---|---|---|
| | | | |

## Cost, from measured times (*estimated*)

oracle 60 s; (1) 210.9 s (job 6af8df0a); (2) and (3) about 420 s each: load 100.6 s, rounds of 3.95 + 17.64 + 37.04 s
of eager prefill four times (large-prefill), 44 house prompts; (4) and (5) about 700 s each: two tree modules through
NVRTC about 545 s (large-prefill-ft), rounds of 0.42 + 2.39 + 6.49 s four times, the R gates, the prompts; (6) about
800 s, 63,441 tokens at 218.8 tok/s a GSM8K-200 arm (spec-loop-sp8) plus two loads. Total about 3,310 s, 55 min, $1.66.

## Open operator decision

The arm's token symbol holds 8 B of local memory at 128 registers on the L40S. 4E's load rule refuses it, and this
job waives it as the `+local` measurement mode does. Whether a served module may hold local memory is the operator's
structural decision. The tree modules of the chain prefill hold 720 to 744 B each (large-prefill-ft): the same
decision covers them.
