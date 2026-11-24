# Prereg: Inkling 2 Mini on CUDA, the read of a prompt by part (2026-11-24)

Written before the run. Not edited after its timestamp; deviations go in `-ECARTS.md` beside it.

## Question

`--moe-rows batch` reads 2,000 tokens in 7.36 s against 16.56 s, byte for byte the same logits (*measured*, job
`6b054af5ca8b51043c3b5983`, L40S), and 6.87 s at 8 rows a decode. The lattice launches fell from 970,955 to 192. Where
do the remaining 6.9 s go: the experts' kernels, candle's attention over several rows, the glue, the head?

## Object and method

`inkrun --prefill-phases on` (commit of this prereg) drains the device at each mark of `layers::phases` during the
prompt's read and reports the seconds of each part. Job `ops/jobs/inkling-cuda-ttft.sh` with
`ARMS="phchunk phbatch8 batch4x64 batch8x32" RETRY=parts`, same prompt, same stack, an `l40sx1` or an `l4x1`.
`batch4x64` and `batch8x32` read the tile's effect at a row count: 16 rows at tile 32 was slower than 8 at 64.

## Decision rule

No threshold. The part with the most seconds in `phbatch8` names the next step. Every arm must hold the logits of the
first, byte for byte, or it is a fault to find.

## Signed prediction (the coordinator, before the run)

| quantity | predicted |
|---|---|
| `phbatch8`, share of the experts (`moe`) | 35 to 60 % |
| `phbatch8`, share of attention (products and scores) | 20 to 45 % |
| `phchunk`, share of the experts | 70 to 85 % |
| `batch4x64` against `batch` (4 rows at tile 128, 7.36 s) | within 0.5 s |
| `batch8x32` against `batch8` (6.87 s) | 0.3 to 2 s slower |

## Cost (*estimated*)

About 9 minutes on an `l40sx1`, $0.27. 25-minute timeout.
