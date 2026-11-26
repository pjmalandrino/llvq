# Prereg: serve what is already measured bit for bit, with no card (2026-11-26)

Written before the first measurement of this lot. Not edited after its timestamp; deviations go in `-ECARTS.md`
beside it. Operator's go: 2026-11-26 evening, after the throughput audit of the three backends. Cost: 0 $, this Mac
only (M3 Max, 64 GB). Predictions signed by the main session.

## Question

The audit of 2026-11-26 found gains that are measured or visible in the code and that `llvq` does not serve. This lot
takes the ones that move no bit and need no card. Three parts.

- A. Qwen3 on the Mac. `llvq` serves the 4B, 8B and 14B on Metal under `configs/qwen3-*-tetra-e4.json`: the expanded
  attention and `MetalTuning::default()` (kernel `lut`, one prompt row a launch). `metal-nuit-2026-11-06` measured on
  the 4B the grouped attention bit for bit over 49,140 decode steps (M5) and x2.38 in decode at 4,096 tokens (M4), the
  kernel `lutd` at x1.025 (M2), `+h` as a wash that moves no bit (M8), and the prefill kernels `+p` and `+pq` taking
  the first token at 4,096 tokens from 59.73 s to 31.53 s (M9, M10), every one behind a bit gate of 253 matrices of
  253. Serving the grouped form on Metal was the operator's decision; the go of this lot takes it, for the sizes that
  pass the identity gate below.
- B. Inkling 2 Mini on the CPU. The prefix cache of this morning (`llvq-runtime/src/prefix.rs`) is refused on the CPU
  because `long_prompt` answers `Off` there, while `experts_cpu::load_served` already sets the pieces of 512, the
  score rows of 128 and the slack of 1,024.
- C. A prompt that the cache holds whole. `Worker` reads its last segment again to get the logits of its last
  position (`llvq-runtime/src/inkling.rs`, "The whole prompt is held"). Keeping those logits when a prompt ends would
  answer a regeneration with no read.

## Design

A. One served config a size for Metal, in `configs/`, named in `library.json` under `metal` and compiled in through
`CONFIGS`. It carries `attention: grouped` and names the Metal kernels it serves, so the served object stays a file.
`check_device` accepts the grouped form on Metal; the kernel form stays refused. `MetalTuning::default()` keeps its
values, so Inkling's Metal stack and every bench keep their bits and their speed. `LLVQ_CONFIG` with one of the six
older configs still serves the expanded form with the default kernels.

B. `long_prompt` answers for the CPU what the CPU load already does, and the cache follows. The key carries the
backend, so a CPU file is never loaded on Metal.

C. A held segment that ended a prompt keeps the logits of its last position (200,058 f32, 0.8 MB). When a request's
whole prompt is held and its last segment has them, the server goes back to that segment's mark and samples from
them. Otherwise it reads the last segment again, as today.

The `.llvq` format, the indexing and the sealed files are not touched.

## Gates

Identity comes first in every part. A part that fails its identity gate is not served, and the reading goes to the
operator.

A, for each of the 4B, 8B and 14B sealed files:
- A-i. The matrix bit gate of `metalrun`, new kernels against `lut`: every matrix.
- A-ii. Grouped against expanded on the decode path in lockstep on WikiText-2 test, max |dlogit| = 0 at every step:
  the 4B has M5 (12 windows of 4,096) and is not run again; the 8B and the 14B on one window of 2,048.
- A-iii. Same generated tokens, new config against the served one, at a 5-token prompt (256 new) and at a long
  prompt (4,096 tokens for the 4B, 2,048 for the 8B and the 14B; 64 new).
- A-iv. Speed through the same binary, arms alternated: 1 discarded + 5 timed rounds for the 4B, 1 + 3 for the 8B
  and the 14B. Decode tok/s and first token, medians with ranges, ratios formed round by round.
- A-v. Through `llvq serve` and `llvq run` with the store's 4B: the answer at temperature 0 is the text the old
  config gives, and the Inkling checks of `ops/inkling/serve-mac.sh` still pass (13 of 13).
- Rule: a size is served under its new config when A-i, A-ii and A-iii hold and its decode at 5 tokens is not under
  x0.97 by the median. A size that fails stays on its expanded config.
- Mutants, each seen red with its site counted: `check_device` accepting the kernel form on Metal; the new config's
  kernel name not applied; the grouped field dropped from the new config.

B:
- B-i. Tiny model on the CPU stack: read `0..b`, save, load into a new cache, read the rest; every logit equal byte
  for byte to the read without the cut.
- B-ii. Real model on this Mac's CPU: a prompt saved and loaded at `b` = 2,048, logits sha256 equal to the straight
  read's.
- B-iii. Server on the CPU, request shaped like a coding agent with a static block just over 2,048 tokens: cold
  first request, then the server restarted and another user message. `cached` at least 2,048, the answer at
  temperature 0 is the text a server with the cache off gives, a damaged file is refused by name.
- Rule: on by default on the CPU when B-i to B-iii hold and the warm first token is at most a quarter of the cold
  one. Otherwise it stays off by default and can be turned on.
- Mutants: the backend removed from the key; the CPU branch of `long_prompt` answering `Off`.

C:
- C-i. Unit tests on the stub model: the logits handed back are those of the first read, byte for byte; a segment
  without kept logits is read again; a rollback refused falls back to the full read.
- C-ii. Server on Metal, real model: the same request sent twice at temperature 0 gives the same text twice, and
  the text of a server before this change.
- Mutants: the kept logits of another segment handed back; the rollback to the mark skipped.

Every part: `cargo test` in debug on the touched crates, `cargo clippy --all-targets` at zero warnings, the Linux
half type-checked by `ops/check-linux-cuda.sh clippy` on `llvq-llm` and `llvq-runtime`.

## Predictions

A, 4B (served config against the new one):
- Identity: A-i 253 of 253, A-iii same tokens at both lengths. Confidence high.
- Decode after a 5-token prompt: x1.03 [1.00 ; 1.07].
- Decode after 4,096 tokens: x2.4 [2.1 ; 2.7], from 15 to 17 tok/s to 36 to 42.
- First token at 4,096 tokens: from 57 to 66 s to 31 s [27 ; 36], x1.9 [1.6 ; 2.2].
- Peak memory at 4,096: not higher than today.

A, 8B and 14B (no long-context number on Metal exists for them):
- Identity: A-i every matrix, A-ii max |dlogit| = 0, A-iii same tokens. Confidence about 85 %: the form is the
  4B's code, and the 14B has another head count (40 against 32).
- Decode after 2,048 tokens: 8B x1.35 [1.15 ; 1.7], 14B x1.3 [1.1 ; 1.7].
- First token at 2,048 tokens: x1.5 [1.2 ; 2.1] for both.
- Decode after 5 tokens: x1.02 [0.98 ; 1.06] for both.

B, this Mac's CPU (no Inkling number on this CPU exists):
- Identity: B-i and B-ii hold.
- Cold first token on a static block just over 2,048 tokens: 250 s [120 ; 600].
- Warm first token after a restart: 20 s [6 ; 45], at least x8 under the cold one.
- File at `b` = 2,048: 0.6 GB [0.35 ; 0.9].

C, Metal, real model:
- Identity: C-i and C-ii hold.
- Same request sent again, agent-shaped prompt of about 8,800 tokens, answer under 1,024 tokens: first token today
  3 to 5 s through the prefix cache; after, at most 0.5 s.
- Short chat prompt of about 40 tokens sent again: first token from 0.3 to 1.5 s to at most 0.15 s.

## What this lot does not do

No card, no paid run. No change of default for `LLVQ_CPU_WEIGHTS`, no grouped attention on the CPU, no prefix cache
on CUDA: each is a decision or a card run of its own. No measurement on x86: the CPU numbers of part B are this Mac's.
