# Prereg: Inkling 2 Mini served on the Mac, with the long-prompt settings (2026-11-26)

Written before the first timed run. Not edited after its timestamp; deviations go in `-ECARTS.md` beside it.

## Question

The runtime `llvq` has served Inkling 2 Mini on CUDA cards only. On this Mac (M3 Max, 64 GB, Metal) only the bench
`inkrun` has run it: 27.70 tok/s after a 32-token prompt, a 2,000-token prompt read in 24.7 s with 12.87 GB held
after it (*measured*, `docs/mesures/inkling-debit-boucle-2026-11-22.txt`). A 9,000-token prompt has never been read
on the Mac. The runtime's long-prompt settings (prompt in pieces of 512 tokens, scores in chunks of 128 rows, sliding
slots bounded with a slack of 1,024) were gated to CUDA; the branch of this prereg opens them to Metal, the score
bias staying on the host (`attn_bias` is a CUDA kernel). This is a reading of the starting point: does `llvq` serve
the model on the Mac, and what do the settings do to a long prompt there?

## Object and method

Branch `debit/inkling-mac-2026-11-26`, commit of this prereg. Release build, `--features metal`, of `llvq`
(crate `llvq-runtime`) and `inkrun` (crate `llvq-llm`). Every run goes through `~/.llvq-guard/heavy.sh`, one at a
time, 45 s apart, with the load sentinel running.

Files. `inkrun`: the text checkpoint `~/llvq-inkling-mini/textonly/` (config sha256 as read in the raw folder), the
24 verified layers `~/llvq-inkling-mini/layers`, the int4 output table `unembed.q4g64.llvqraw` sha256 `cb26ce0c...`,
the files of the 2026-11-23 bench. `llvq`: the store copy of `inkling2:mini` (Hub revision `697661e3`, the slim file
set), loaded by `llvq serve --local` on its `inkling.json`, with `LLVQ_HOME` in a scratch directory.

Prompts. The 32-token prompt and its reference tokens: `~/llvq-inkling-mini/lot4-coord/tokens.json` (sha256
`0f09e3e9...`). The 2,000-token prompt: the first 2,000 ids of WikiText-2 test window 0
(`~/llvq-inkling-mini/lot2-d/wikitext_w0_2048.json`). The 9,000-token prompt: the first 9,000 ids of windows
0, 1, 2, 0, 1 joined, the file of the CUDA reading (sha256 `776675c5...`).

Stack of every timed `inkrun` arm: `--glue fused --attn fused+scores --head int4 --head-table <table> --n 32
--require-layers 24 --device metal`.

1. Correctness gate, not timed. `inkrun` on the 32-token prompt in the configuration that produced
   `tokens.json` (no switch, the checkpoint's head): the 32 greedy tokens equal the reference. Then the whole stack
   with the checkpoint's head (also equal on 2026-11-23). Then `llvq serve --local` on the real model: load, a short
   answer to a simple prompt, and the 13 checks of `ops/inkling/serve_checks.py` as the card ran them (the backend
   it expects becomes `metal`; nothing else changes).
2. Decode, 32-token prompt: three rounds in separate processes, the median step of each, then the median of three.
3. 2,000-token prompt, one run each: arm A (main as it is: one call, full slots, scores in chunks of 512) and arm B
   (`--prefill-chunk 512 --attn-chunk 128 --kv-window bounded=1024`). Read: seconds of the prompt, device memory
   after it (Metal `currentAllocatedSize`), decode speed after it, the 32 tokens, the logits. Then one run of each
   arm with `--peak-probe on` for the peak (the probe drains the device at every visit, so its seconds are not read).
   The sliding window is 1,024 positions, so a bounded slot keeps 2,048: at 2,000 tokens it drops nothing, and arm
   B differs from A by the pieces and the score chunks. The CUDA reading bounded with `inkrun`'s default slack of
   256; this one uses the runtime's 1,024.
4. 9,000-token prompt, arm B only, one run unprobed and one with `--peak-probe on`. Arm A is not run at 9,000 tokens
   (memory risk on the operator's machine).
5. The runtime: time to first token of the long prompt of check (e) (about 8,791 tokens) and the generation speed
   the server reports (`eval_count` over `eval_duration`).

Machine. Load at one minute and GPU utilisation are written before each timed run. Above load 4 or GPU 60 %, wait up
to 10 minutes, then run and mark the line "machine busy".

## Decision rule

A reading, with one gate: the correctness gate above. The `inkrun` tokens must equal the reference; the server's 13
correctness checks are reported as read, PASS or FAIL each. The timed figures are reported as read.

Stop rules. If arm B at 2,000 tokens holds more than 20 GB (peak, or memory after the prompt when no peak is read),
the 9,000-token run is not launched. The sentinel's critical alert stops everything.

## Signed predictions (the main session, Claude Fable 5.2, 2026-11-26)

| | quantity | predicted |
|---|---|---|
| P1 | `llvq` serves Inkling on the Mac at the first complete try | probability 0.6 |
| P2 | decode after the 32-token prompt, fused stack and int4 head, by `inkrun` | 26 to 29 tok/s on a quiet machine |
| P3 | 2,000-token prompt: arm A read in | 22 to 27 s |
| P3 | arm B against arm A | within +-15 % |
| P3 | the same 32 greedy tokens after the prompt in A and B | probability 0.8 |
| P4 | 9,000-token prompt, arm B: read in | 90 to 200 s |
| P4 | peak Metal memory | 12 to 18 GB |
| P4 | decode after it | 21 to 26 tok/s |

## Cost

$0. Local only, on the operator's Mac, which stays in use.
