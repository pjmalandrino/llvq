# Prereg: the read of a long prompt by Inkling 2 Mini on the Mac (2026-11-26)

Written before the first timed run. Not edited after its timestamp; deviations go in `-ECARTS.md` beside it.

## Question

On this Mac (M3 Max, 64 GB, Metal), arm B of prereg `inkling-mac-serve-2026-11-26` (fused stack, int4 head, prompt in
pieces of 512, scores by 128 rows, sliding slots bounded with a slack of 1,024, score bias on the host) reads a
2,000-token prompt in 23.49 s and a 9,000-token prompt in 148.66 s, at an 11.42 GB peak (*measured*,
`docs/mesures/inkling-mac-serve-2026-11-26.txt`). The use aimed at is a coding agent whose prompt holds about 9,000
tokens. Where does the time of that read go on Metal, and how much of it can be removed without changing the file or
the quality?

On CUDA the batched read of the experts and the bias built on the card cut the read with the same bytes
(`docs/mesures/inkling-cuda-ttft-2026-11-24.txt`), but there the gain came mostly from the launch count. On Metal a
chained candle launch costs about 1.66 us and the experts are already read 8 rows a decode, so the time may be
elsewhere. The first step is therefore a profile.

## Object and method

Branch `debit/inkling-mac-2026-11-26`, from commit dc46537. Release build, `--features metal`, of `inkrun` (crate
`llvq-llm`) and `llvq` (crate `llvq-runtime`). Every run goes through `~/.llvq-guard/heavy.sh`, one at a time, 45 s
apart, with the load sentinel running. Same files and prompts as the prereg of the morning: the text checkpoint, the
24 verified layers, the int4 table `unembed.q4g64.llvqraw`, the first 2,000 ids of WikiText-2 window 0, the 9,000-id
file of the CUDA reading.

Stack of every timed arm: `--glue fused --attn fused+scores --head int4 --head-table <table> --n 32
--require-layers 24 --device metal --prefill-chunk 512 --attn-chunk 128 --kv-window bounded=1024`, plus the flag of
the step under test.

1. Profile. `inkrun --prefill-phases on` on arm B at 2,000 then 9,000 tokens. The device is drained at every mark, so
   the shares are inflated by the number of marks a part carries; the profile is read for its order, and the shares
   are given with that bias.
2. Steps. The largest part first, then the next, while the measured gain of a step exceeds 5 % of the 9,000-token
   read. Each step is a reference arm and the new arm run in the same window of time, never against a morning figure
   alone. Each new or changed kernel has a twin test against the path it replaces and a mutant seen red.
3. Final reading, the commands of the morning: 2,000 and 9,000 tokens (seconds of the read, decode after it, the 32
   tokens, logits sha256), a second run with `--peak-probe on` for the peak; then the server through
   `ops/inkling/serve-mac.sh`: the 13 checks and the time to the first token of the 8,791-token prompt.

Machine. Load at one minute and GPU utilisation are written before each timed run. Above load 4 or GPU 60 %, wait up
to 10 minutes, then run and mark the line "machine busy".

## Decision rule

A change of class A gives the logits of `b2000.logits.f32` and `b9000.logits.f32` of the morning bit for bit (same
sha256). A change that does not is of class B: it becomes a default only if the perplexity paired by position against
arm B of the morning has its point inside +-0.12 % over 3 WikiText-2 windows (`inkrun --score-prompt on`, as the CUDA
journal), and the journal says so. A step is kept when its gain on the 9,000-token read exceeds 5 %.

Stop rules. The sentinel's critical alert, a load or memory alert, or free disk under 60 GiB stops everything.

## Signed predictions (the main session, Claude Fable 5.2, 2026-11-26)

| | quantity | predicted |
|---|---|---|
| Q1 | profile of arm B, 2,000 tokens: experts | 50 to 75 % of the time |
| Q1 | 2,000 tokens: scores and bias together | 5 to 20 % |
| Q1 | 9,000 tokens: experts | 30 to 55 % |
| Q1 | 9,000 tokens: scores and bias together | 30 to 55 % |
| Q2 | end of this lot: 9,000-token prompt read in | 70 to 110 s (148.66 today); probability 0.5 of being under 110 s |
| Q3 | end of this lot: 2,000-token prompt read in | 13 to 19 s (23.49 today) |
| Q4 | peak Metal memory at 9,000 tokens | 12.5 GB at most |

## Cost

$0. Local only, on the operator's Mac, which stays in use.
