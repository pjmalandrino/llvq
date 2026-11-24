# Prereg: Inkling 2 Mini on CUDA, a prompt's rows through the experts in one launch a step (2026-11-24)

Written before the run. Not edited after its timestamp; deviations go in `-ECARTS.md` beside it.

## Question

Reading a prompt of 2,000 tokens takes 16.64 s on an L40S under the whole stack (*measured*, job
`inkling-cuda-context`, 2026-11-23), with 970,955 lattice launches: the resident experts take the rows one expert and
one chunk at a time (`resident_cuda.rs`, `forward`). How much of that time is the launches and the host work around
them, and how much is the decode of the weight stream?

## Object and method

`inkrun --moe-rows batch` (commit of this prereg): every (token, expert) pair of a layer is one row, and a layer is
four launches: `moe_rot` on the pairs, `moe_tetra_rows` for gate and up, `moe_rot`, `moe_tetra_rows` for down. The
kernel body is `tv_tetra48_rows_h`'s, the rotation's is `rot_apply_rows`'s, the activation is candle's, and each
token's weighted rows are added in ascending expert order from zero. `--moe-rows chunk` is the path so far.
Job `ops/jobs/inkling-cuda-ttft.sh` on an `l40sx1`, an `l4x1` if no L40S comes: 2,000 tokens of WikiText-2 test
window 0, 32 greedy tokens after them, the whole stack (`--moe four --glue fused --attn fused+scores --head int4`).
Five arms: `chunk`, `batch`, `batch8` and `batch16` (`LLVQ_PREFILL=8:64` and `16:32`), `chunk16`. The served pair of
`LLVQ_PREFILL` is 4 rows at tile 128.

## Decision rule

IDENTITY: the logits file of an arm (the prompt's last position and the 32 steps) equals `chunk`'s byte for byte.
An arm that is not identical is not a speed result; it is a fault to find, and it is reported as such.
The speed is `seconds_prefill` of each arm, one run each, with the launches counted. No threshold: the run attributes
the time. An arm keeps its place in the stack only if it is identical.

## Signed prediction (the coordinator, before the run)

| quantity | predicted |
|---|---|
| the unit compiles and every arm ends on a first launch | probability 0.6 |
| `batch`, `batch8`, `batch16` identical to `chunk` | probability 0.75 each, given they run |
| `chunk16` identical to `chunk` | probability 0.9 |
| `chunk`, seconds of prefill, L40S | 15.5 to 17.5 |
| `batch`, seconds of prefill, L40S | 6 to 11 |
| `batch8` | 4 to 9 |
| `batch16` | 3 to 8 |
| `chunk16` | 8 to 14 |
| lattice launches in the prefill, `batch` | 192 |
| GB at the end, `batch` | 13 to 16 |

The coordinator's last four speed predictions on this model were too high. These ranges are wide on purpose.

## Cost (*estimated*)

About 13 minutes on an `l40sx1`, $0.40; $0.20 on an `l4x1`. 25-minute timeout.
