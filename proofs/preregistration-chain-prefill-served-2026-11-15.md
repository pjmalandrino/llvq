# Prereg chain-prefill-served: the chain prefill at the served door, 4B (2026-11-15)

Written by an executing agent, reread and signed by the lead, timestamped before any run. The operator gave the go
for this job on 2026-11-15, its cost announced: about $1.28 (*estimated*), hard cap 60 min, $1.80.
Measured code: branch `dev/trois-gains-2026-11-15`, commits `80cc6cb` and `5acc9f5` and the commit that carries this
file; the image is rebuilt from the commit that carries its timestamp. Job script `ops/jobs/chain-prefill-served.sh`.
This is development work on the operator's order, outside the research phase's budget.

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

Signed by the lead (main session) on 2026-11-15, before the stamp.

- The controls all hold on the first launch: 50 %. The door path, the module cache and the cubin load have never run
  on a card; the image has never been built from this branch.
- TTFT at 8,192, CH: 6.7 s [6.0, 8.0]. The bench read 6.485 s at 8,191; the door adds a hand-over launch. Known flaw:
  the bench loaded its modules from PTX, the door loads a cubin.
- TTFT at 8,192, V9: 35.5 s [33, 39]. The served door read 35.05 s on 2026-11-07.
- TTFT, CH over V9 at 8,192: 0.19 [0.16, 0.24]; at or under 0.25: 85 %. At 1,024: 0.12 [0.08, 0.20].
- Decode tok/s, CH over V9: inside [0.98, 1.02] at every length: 70 %. The decode is 4E v9 in both; the caches live
  in other buffers after the chain.
- Phase peak at 8,192: CH 3.7 GB [3.5, 4.3], V9 5.95 GB [5.8, 6.1]. CH at or under V9: 90 %.
- Load: cold 650 s [500, 800]; cached 110 s [95, 150]; cached at or under 150 s: 75 %.
- GSM8K-200: V9 166 right [163, 169]; CH within 3 of V9; 4 discordant pairs [0, 12]; McNemar p at or above 0.05:
  90 %. The prompts are short, so the chain feeds few rows; a near tie can flip.
- The verdict reads PASSES: 35 %. NOT SETTLED is the likeliest row if the controls hold, on the cached load or on a
  decode ratio just outside its band.

## Reading table (lead's, filled after the run)

| quantity | predicted | measured | inside |
|---|---|---|---|
| | | | |

## Cost, from measured times (*estimated*)

oracle 60 s; (1) 100.6 s eager load (large-prefill) plus the base tree module about 165 s and a12ipg32ft about 380 s
(large-prefill-ft header, 1,127.1 s for four modules) = about 645 s; (2) and (3) about 110 and 140 s; (4) about 310 s,
the 8,192 identity runs carrying two eager prefills of 37 s; (5) about 480 s; (6) and (7) about 400 s each, 63,441
tokens at 218.8 tok/s for GSM8K-200 under 4E (spec-loop-sp8). Total about 2,550 s, 43 min, $1.28 at $1.80/h.
