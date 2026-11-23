# Prereg inkling-first-card: Inkling 2 Mini generates on one L4 with its experts resident in lattice codes (2026-11-22)

Status: written by agent U (lot 2) as a draft, completed and signed by the coordinator on 2026-11-22, timestamped by
the `.ots` beside it before any run on a card.
Cost: about $0.15 (*estimated*, §Cost); hard cap 30 min on `l4x1` at $0.80/h, $0.40. Card budget left: $25 (operator,
2026-11-22), nothing spent on cards for this model.
Measured code: commit `f05dc6eb934129692b61e54c44139e2e38f7ced2`, from which the image is built. It carries
`llvq-llm/src/inkling/` (the forward pass, the Metal resident experts, `expert_codes.rs`, `resident_cuda.rs`),
`llvq-llm/src/bin/inkrun.rs` and the additive section of `llvq-llm/src/fused_cuda.rs`. Job script
`ops/jobs/inkling-first-card.sh`; inputs staged by `ops/inkling/stage-first-card.sh`.

A timestamped prereg is no longer edited. A deviation is written in `<name>-ECARTS.md`.

## Question

Does this engine generate Inkling 2 Mini on an NVIDIA card with its 24 layers of routed experts resident in Tetra
codes, the same tokens as the Mac's resident run, in how many GB, at what first speed?
The forward pass and the resident experts have run on CPU and Metal only. CUDA has never run a line of
`inkling/`, and `resident_cuda.rs` was type-checked, not executed. This run decides whether GSM8K on the same card is
worth launching, and its step time prices it.

## Setup

One `l4x1` (L4, 24 GB, sm_89), the image `hf.co/spaces/Pier-Jean/llvq-runner-cuda` at Space sha
`b959fa385a7713baba9befc7571e6454d4deb8cc`, if its build passes; a rebuild at another sha is a deviation written
beside this file before the run. Inputs from `hf://buckets/Pier-Jean/jobs-artifacts/inkling2-mini-inputs-2026-11-22/`,
copied to the container's disk and checked at their sha256 before the run: the text-only file (646 non-expert
tensors, 3,218,968,280 B), its config, the 24 layer archives (codes of 128 experts x 3 records and `VERIFIED` each,
6.23 GB), the manifest of every codes file, and the tokens file, sha256
`0f09e3e902b72965d8b2e146cda116b901273861a56ea13e2aeb5f886c216cdf`: the 32-token prompt of `lot2-d/ref_full.json` and,
as `reference`, the 32 tokens of the Mac's resident run on that same commit, 24 layers resident, Metal
(`docs/mesures/inkling-first-card-2026-11-22-brut/mac-reference/inkrun_metal_24.json`).

```
oracle Qwen/Qwen3-0.6B 64 cuda
inkrun --config config.json --checkpoint model.safetensors --layers layers --tokens tokens.json \
       --device cuda --n 32 --require-layers 24 --out inkrun.json
```

No variable: the job refuses any `LLVQ_*` in its environment. The tile is the card's measured row (64 on sm_89).

## Controls

1. `oracle` MATCH on this backend.
2. Every input at its pinned sha256; 24 x 129 manifest lines checked by `sha256sum -c`.
3. `inkrun` reports 24 resident layers and no native layer (it refuses below 24 before any token).
4. The report is complete: tokens, four memory readings, step times, launches.

## What gets published, and what does not get compared

Published (*measured*): the 32 tokens and their comparison with the Mac's resident run (equal prefix, first mismatch
and the top-2 margin there); GB the driver reports before load, after the dense load, after the experts and at the
end; dense and expert load seconds; prefill seconds for 32 tokens; median step seconds with range over 31 steps, and
its inverse as a first tok/s; lattice launches a token. The VRAM series of `ops/run.py monitor --metrics-out`.
Not compared: this tok/s with any Qwen3 figure, with the Mac's, or with another stack (rule 5: no ratio across cards
or stacks); it is an eager path whose dense half runs in f32, a first reading and not a served speed. Not published as
whole-model b/param: the dense half is f32 here, not the sealed object's encoding.

## Decision rule

| reading | verdict |
|---|---|
| controls hold; 32 of 32 tokens equal the Mac's; GB after the experts at most 14.0 | PASSES: GSM8K on this card may be prepared |
| controls hold; tokens part at a step whose top-2 margin is under 0.05 in the Mac's run and in the card's | PASSES WITH A NEAR-TIE: reported with the step and the margins; GSM8K may be prepared |
| controls hold; tokens part at a step whose margin is 0.05 or more on either device | FAILS: a CUDA defect to find on the Mac first, no rerun before |
| controls hold; tokens as in one of the two rows above; GB after the experts above 14.0 | NOT SETTLED on memory: the tokens' verdict stands, the GB is reported, operator decision |
| a control fails, or `inkrun` errors | VOID: the cause is read in the logs; a rerun is the operator's decision |
| otherwise | not settled, operator decision |

## Signed prediction

Signed by the coordinator before the run.

- The job runs to its end on the first launch: probability 0.55. It is the first execution of this code on CUDA.
- GB after the experts: 11.5 to 12.5. The Mac allocated 5.09 GB of dense weights and 11.37 GB after the experts
  (*measured*, the reference run); the CUDA context adds a few hundred MB.
- Median step: 0.10 to 0.30 s, 3 to 10 tok/s. The Mac's reference run reads 0.205 s on Metal with the same 1,920
  lattice launches a token.
- 32 of 32 tokens equal to the Mac's: probability 0.6. The Mac's run has two steps with a top-2 margin under 0.05
  (0.0094 at step 4, 0.0113 at step 22) and nine under 0.2; the CUDA and Metal kernels sum a row in different orders.
- Verdict: PASSES or PASSES WITH A NEAR-TIE, probability 0.5 together; VOID 0.4; FAILS 0.1.

Agent U's inputs for it (*estimated*): GB after the experts about 11.5 to 12.5 (5.09 GB of dense weights as Metal
allocated them, 6.20 GB of expert codes as the Mac measured them, 258.3 MB a layer, plus the CUDA context), about
0.4 GB more at the end; median step 0.1 to 0.3 s, set by about 1,920 lattice launches and several thousand candle
launches a token and 48 host reads of the routing.

## Reading table (filled after the run)

| quantity | predicted | measured | inside |
|---|---|---|---|
| | | | |

## Cost, from estimated times (*estimated*)

Image pull and start about 3 min; oracle about 1 min (it downloads Qwen3-0.6B); 9.45 GB copied from the bucket mount
and checked, about 4 min; dense load and 3,072 experts uploaded, about 2 min; prefill and 31 steps under 1 min. About
11 min, $0.15. Timeout 30 min, $0.40.
