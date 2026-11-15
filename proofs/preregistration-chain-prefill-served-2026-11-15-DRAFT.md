# Prereg chain-prefill-served: the chain prefill at the served door, 4B (DRAFT, 2026-11-15)

Status: DRAFT, not timestamped. The lead rereads it, signs the predictions below and stamps it before any run.
Cost: about $1.28 (*estimated*), hard cap 60 min, $1.80. Budget: the operator's, to announce with the go.
Measured code: branch `dev/trois-gains-2026-11-15`, commit of gain 1 (to fill at the stamp). Job script
`ops/jobs/chain-prefill-served.sh`.

A timestamped prereg is no longer edited. A deviation is written in `<name>-ECARTS.md`.

## Question

Does `configs/qwen3-4b-tetra-e4-4e-v9-chain.json` give the first token sooner than the served v9 config at the served
door, with the same decode speed, a lower peak, and GSM8K-200 not separated?
The chain was measured only inside `prefillbench`, its own launch loop, with NVRTC at every start
(`docs/mesures/large-prefill-ft-2026-11-11.txt`). No door called it, and no GSM8K pairing exists. This decides
whether the operator is shown a candidate for serving.

## Setup

One L40S. The sealed 4B (`qwen3-4b-sealed.bin`, sha256 886391a8). Two configs, each in its own processes:
`V9` = `configs/qwen3-4b-tetra-e4-4e-v9.json` (served), `CH` = `configs/qwen3-4b-tetra-e4-4e-v9-chain.json`.

```
(1) LLVQ_CONFIG=CH LLVQ_MODULE_CACHE=$O/modcache fusedrun S 8                     # load check, cache empty: miss x2
(2) the same command again                                                        # cache hit x2, load time
(3) LLVQ_CONFIG=CH fusedrun S 256                                                 # served door, 5-token prompt
(4) LLVQ_CONFIG=CH LLVQ_CONTEXTS=5,1024,4096,8192 fusedrun S 256                  # under both samplers
(5) LLVQ_CONFIG=V9 LLVQ_CONTEXTS=5,1024,4096,8192 fusedrun S 256                  # under both samplers
(6) LLVQ_DATASET_REV=740312ad... LLVQ_CONFIG=CH LLVQ_GSM8K_DUMP=ch.jsonl gsm8k S cuda 200
(7) the same with LLVQ_CONFIG=V9, then gsm8kpair v9.jsonl ch.jsonl
```

Steps (2) to (7) set the same `LLVQ_MODULE_CACHE`. The epoch sampler cuts each `mark ctx-<n>` window.

## Controls

1. oracle MATCH; the sealed file and both configs at their sha256; no `LLVQ_*` in the job environment.
2. Every process prints `4E: persistent module 252757 bytes, sha256 1b422b62...`: the served v9 module.
3. Step (1) stops the job unless it prints two `module cache: miss` lines, the `chain prefill: arm a12ipg32ft` line,
   the a12ip load check line (`253 projections read in place`) and exits 0.
4. Step (2) prints two `module cache: hit` lines and the same token ids as step (1).
5. Under CH, at every length and at the door, 4E's tokens equal the eager decode's after the same chain prefill
   (the binary fails otherwise). This is the bit-for-bit gate of the decode. The tokens against V9's are reported.
6. The tree modules' text sha256: base 306,060 bytes ce1c354d, a12ipg32ft 457,150 bytes 19b734a3 (as in
   large-prefill-ft).

## What gets published, and what does not get compared

Published: time to first token and decode tok/s after it at 5 / 1,024 / 4,096 / 8,192, medians and round ranges, per
config; CH over V9 time to first token as a quotient of medians with its envelope (two processes, METHODE §2); phase
peaks per length and process peaks; load time cold and cached; GSM8K-200 correct counts, discordant pairs, McNemar p.
Not compared: CH against `prefillbench` figures (another loop, another job); any figure across cards.

## Decision rule

| reading | verdict |
|---|---|
| controls 1 to 6 hold; TTFT CH/V9 at 8,192 <= 0.25; decode tok/s CH/V9 in [0.98, 1.02] at every length; CH phase peak at 8,192 <= V9's; cached load <= 150 s; GSM8K-200 McNemar p >= 0.05 | PASSES: shown to the operator as a candidate |
| a control fails | VOID: nothing published but the failure |
| controls hold, GSM8K p < 0.05 with CH below V9 | FAILS on quality |
| controls hold, any other bound missed | NOT SETTLED: the operator decides with the numbers |
| otherwise | not settled, operator decision |

## Signed prediction

To be signed by the lead before the stamp. Each: expected value, interval, reason in two lines, known flaw.

- TTFT at 8,192, CH: ...
- TTFT at 8,192, V9: ...
- Decode tok/s at 8,192, CH over V9: ...
- Phase peak at 8,192, CH and V9: ...
- Load, cold and cached: ...
- GSM8K-200, discordant pairs: ...

## Reading table (lead's, filled after the run)

| quantity | predicted | measured | inside |
|---|---|---|---|
| | | | |

## Cost, from measured times (*estimated*)

oracle 60 s; (1) 100.6 s eager load (large-prefill) plus the base tree module about 165 s and a12ipg32ft about 380 s
(large-prefill-ft header, 1,127.1 s for four modules) = about 645 s; (2) and (3) about 110 and 140 s; (4) about 310 s,
the 8,192 identity runs carrying two eager prefills of 37 s; (5) about 480 s; (6) and (7) about 400 s each, 63,441
tokens at 218.8 tok/s for GSM8K-200 under 4E (spec-loop-sp8). Total about 2,550 s, 43 min, $1.28 at $1.80/h.
