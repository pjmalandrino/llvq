# Prereg: Inkling 2 Mini on CUDA, the perplexity of the fused stack against no switch (2026-11-23)

Written before the run, and before any result of the stack job was read. Not edited after its timestamp; deviations
go in `-ECARTS.md` beside it.

## Question

On CUDA, does the fused stack (experts in four launches with the one-launch activation, fused glue, attention
products and scores in their kernels) stay inside the tolerance of +-0.12 % in perplexity against the path with no
switch? What does the int4 g64 output head add to that?

## Object and method

The sealed Inkling 2 Mini file, 24 layers resident in lattice codes, through `inkrun --device cuda`. Three WikiText-2
test windows of 2,048 tokens, the ones the Mac's reading of 2026-11-22 used, staged in the bucket as
`windows/w{0,1,2}.json` (sha256 in the job script). 2,047 teacher-forced positions a window through the one-row path
(`--score-from 1`), paired by position. Arms `ref` (no switch), `all`, `head` (`all` with the int4 head). Job
`ops/jobs/inkling-cuda-ppl.sh` on one `l40sx1`.

The interval is the 95 % interval of the mean paired difference of the negative log-likelihoods, the three windows
pooled (6,141 positions), as the Mac's reading formed it.

## Decision rule

`all`: PASS if the pooled point is inside +-0.12 %. If the interval leaves the band while the point is inside, the
reading is "not separated", and more windows are owed before the switches are served. FAIL if the point is outside:
the switches are not served on CUDA until the cause is named.

`head`: no verdict here. The head reads other values than the checkpoint's, so its ratio is a quality cost of the
sealed table, to be stated beside the speed it buys.

## Signed prediction (the coordinator, before the run)

| quantity | predicted |
|---|---|
| `all`, pooled point | inside +-0.05 % |
| `all`, pooled interval inside +-0.12 % | probability 0.7 |
| `head`, pooled point | between -0.3 % and +0.6 % |
| every arm ends | probability 0.8 |

## Cost (*estimated*)

About 25 minutes on `l40sx1`: $0.75. Timeout 50 minutes, $1.50 at most.
