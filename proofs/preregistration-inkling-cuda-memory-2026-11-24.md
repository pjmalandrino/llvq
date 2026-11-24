# Prereg: Inkling 2 Mini on CUDA, memory and the read of a prompt at 2,000 and 9,000 tokens (2026-11-24)

Written before the run. Not edited after its timestamp; deviations go in `-ECARTS.md` beside it.

## Question

The card holds 13.997 GB at the end of a run after a 2,000-token prompt and 16.043 GB with the batched read, against
8.387 GB after the load (*measured*, L40S). One cause is read in the code: the convolution state kept after a prompt
was a view into the whole `[T + 3, C]` context, four times a slot, 2.56 GB at 2,000 tokens (*computed*). Does the
copy that replaces it give that memory back, what is the peak, and what do 9,000 tokens cost in seconds and in GB?

## Object and method

Commit of this prereg: `conv_context` keeps its `k - 1` rows in a buffer of their own (`layers.rs`), and
`inkrun --peak-probe on` reads the driver's total minus free on CUDA after every layer visit, before and after a
drain, and once after the prompt. Job `ops/jobs/inkling-cuda-ttft.sh` with
`ARMS="pk2000c pk2000 t9000 pk9000" RETRY=memory`, an `l40sx1`. `pk2000c`: the chunk path at 2,000 tokens with the
probe. `pk2000`: the batched read at 8 rows and tile 64. `t9000`, `pk9000`: the batched read of 9,000 tokens, without
and with the probe. The 9,000 tokens are the head of windows 0, 1, 2, 0, 1 of the staged WikiText-2 test windows: a
repeat, a prompt for time and memory and not for quality.
The driver's figure counts what the stream's pool still holds; the probe reads at visit boundaries and can miss a
peak inside a visit. Both are said with the result.

## Decision rule

IDENTITY: the logits of `pk2000c` have the sha256 of the chunk arm of job `6b054af5ca8b51043c3b5983`,
`9c1624ad...` in its `summary.txt`, and `pk2000` holds the same bytes. The copy changes no value or it is a fault.
The memory and the seconds are reported as read. No threshold.

## Signed prediction (the coordinator, before the run)

| quantity | predicted |
|---|---|
| `pk2000c`, GB at the end | 10.5 to 12.0 (13.997 before) |
| `pk2000`, GB at the end | 11.5 to 14.0 (16.043 before) |
| `pk2000`, peak of the probe during the prompt | 12 to 16 |
| `t9000`, seconds of the prompt's read | 33 to 62 |
| `pk9000`, peak of the probe during the prompt | 17 to 30 |
| `pk9000`, GB at the end | 13 to 20 |
| the 9,000-token arms end without a fault | probability 0.7 |

## Cost (*estimated*)

About 14 minutes on an `l40sx1`, $0.42. 25-minute timeout.
