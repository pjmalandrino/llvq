# Prereg: the grouped attention of the Qwen3 files on the CPU backend, read and not served (2026-11-27)

Written before the first measurement of this lot. Not edited after its timestamp; deviations go in `-ECARTS.md`
beside it. Operator's order: autonomy for the night of 2026-11-26, local work at 0 $. Cost: 0 $, this Mac only
(M3 Max, 64 GB, `LLVQ_THREADS` 6). Predictions signed by the main session. This lot serves nothing: serving the
grouped form on the CPU is the operator's decision.

## Question

The runtime's CPU backend serves the three Qwen3 files under `configs/qwen3-*-tetra-e4.json`, the expanded attention
on a `Cat` store. At every token the expanded form copies each cached key and value head to its query heads
(`repeat_kv`) and the store concatenates the cache. The grouped form reads the cached heads with no copy. It is
served on the card under a tolerance and is bit for bit on Metal (`metal-nuit-2026-11-06`, M5). It has never run on
the CPU backend.

What does the grouped form give on the CPU, in bits and in decode speed, at a short and at a long prompt?

## Design

The 4B sealed file through `cpurun` (or the binary that times the CPU backend as `llvq` serves it), two configs that
differ by the `attention` field alone: the served `qwen3-4b-tetra-e4.json` and a measurement config that names
`grouped`. If the CPU path refuses the grouped form or cannot name it, the smallest change that lets a measurement
config through is made, and the served CPU configs of `library.json` do not change. No format, no sealed file, no
default is touched.

## Gates

- G-i. Grouped against expanded on the decode path in lockstep on WikiText-2 test, one window of 2,048 tokens, the 4B:
  max |dlogit| at every step and the count of steps whose argmax differs. Zero everywhere makes the form class A on
  the CPU. Otherwise the journal gives the largest |dlogit| against the largest |logit|, and the form is class B.
- G-ii. Same generated tokens, grouped against expanded, at a 5-token prompt (256 new) and at a 2,048-token prompt
  (64 new).
- G-iii. Speed, arms alternated in one window: 1 discarded + 3 timed rounds, decode tok/s after the 5-token prompt
  and after the 2,048-token prompt, and the time to the first token at 2,048. Medians with ranges, ratios formed
  round by round. Peak footprint of each arm, one process an arm.
- G-iv, only if G-i to G-iii are handed in before 04:30: the same three gates on the 8B, 1 + 2 rounds.
- Reading, not a rule: nothing is served by this lot. If the form is class A and gains at 2,048 tokens without
  losing at 5 tokens, the journal says a served CPU config would need only the operator's word. If it is class B, the
  journal says which tolerance run the decision would need and what it would cost on this Mac.
- Mutant, if code changes: the measurement config's `attention` field ignored on the CPU, seen red by a test that
  reads the form the CPU load reports.

`cargo test` in debug on the touched crates and `cargo clippy --all-targets` at zero warnings if code changes.

## Predictions

- G-i: bit for bit on the CPU, confidence 50 %. The form stacks the query heads of a key head as rows of one matrix,
  and the CPU product may block the two shapes differently. If the bits move: max |dlogit| at most 1e-3 and no step
  whose argmax differs, confidence 85 %.
- G-ii: same tokens at both lengths, confidence 85 %.
- Decode after a 5-token prompt: x1.00 [0.97 ; 1.04].
- Decode after a 2,048-token prompt: x1.5 [1.15 ; 2.5]. Computed from the bytes: the expanded form copies about
  2.4 GB of cache a token at 2,048 positions on the 4B if the cache is f32, the grouped form none.
- First token at 2,048 tokens: x1.1 [1.0 ; 1.4].
- Peak footprint at 2,048 tokens: lower by 0.1 to 1.0 GB.

## What this lot does not do

No card, no paid run, no x86. No served config changes and nothing is added under `cpu` in `library.json`. No
perplexity run: a tolerance gate on the CPU is a lot of its own if the form is class B.
