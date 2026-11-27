# Prereg: the sliding layers of Inkling 2 Mini score the window only, on the Mac (2026-11-27)

Written before the first measurement of this lot. Not edited after its timestamp; deviations go in `-ECARTS.md`
beside it. Operator's order: autonomy for the night of 2026-11-26, local work at 0 $. Cost: 0 $, this Mac only
(M3 Max, 64 GB). Predictions signed by the main session. The arm is built off by default: turning it on is the
operator's decision, whatever this lot reads.

## Question

Inkling 2 Mini has 24 layers. Twenty are sliding layers with a window of 1,024 keys, four are full layers (5, 11, 17,
23). On the long-prompt path a sliding slot holds its window, its slack of 1,024 and the piece being read: 2,560 to
3,072 keys. `Attention::scores` (`llvq-llm/src/inkling/layers.rs`) computes the score of every query row against
every key the slot holds, then the bias sets the keys outside the window to minus infinity. A chunk of 128 query rows
can see 1,151 keys at most. The profile of `inkling-mac-prompt-2026-11-26` charges 34.7 % of a 9,000-token read to
scores, bias and softmax and 4.5 % to the cache copies (*measured* with the device drained at every mark).

Does reading only the keys a chunk can see shorten the read of a long prompt, and what does it do to the bits?

## Design

D, the prompt path (`forward_mats`, `scores`). For a sliding layer, each chunk of query rows reads the keys and the
values of its visible range only, from the first key its first row can see to its last row's own position. The bias is
built for that range. The full layers are not touched. The arm is a switch of `inkrun`, off by default; the runtime
keeps today's path.

E, the decode path, second and only if D is handed in before 04:30. If a sliding slot's ring holds more than the
window after a long prompt, the score kernel of a decode step reads the window only, behind the same kind of switch.
If the ring already holds the window only, E is void and the journal says so.

The keys outside the window have a probability of exactly zero today (the bias is minus infinity). Dropping them
removes zero terms from the softmax sum and from the value product. The value of each row is the same in exact
arithmetic; the order of the additions in the device's products may change, so this lot does not assume class A.

The `.llvq` format, the sealed files, the runtime's defaults and the prefix cache files are not touched.

## Gates

- D-i. Tiny model, arm on against arm off, at prompts that cross the window, the slack and a piece boundary, and with
  a slot whose first key held is not position 0: every logit within 1e-4 of the reference scaled by the largest
  logit, and the same argmax at every row.
- D-ii. Real model, 2,000 and 9,000 tokens (the prompts of `inkling-mac-prompt-2026-11-26`): the logits file of the
  arm against the reference digests `6c05fabd...` and `01407333...`. Equal digests make the arm class A. Otherwise the
  journal gives max |dlogit|, the largest logit, the argmax of every row and the 32 tokens after the prompt.
- D-iii. Speed of the read through `inkrun`, arms alternated in one window: 1 discarded + 4 timed pairs at 9,000
  tokens, 1 + 4 at 2,000. Medians with ranges, ratios formed pair by pair. Peak footprint of each arm.
- E-i to E-iii, if E runs: the same three gates on the decode path, 256 steps after the 9,000-token prompt and after
  a 5-token prompt, 1 + 4 pairs.
- Rule: the arm stays in the code, off, when its identity gate holds (class A, or D-i and the same argmax and the same
  32 tokens in D-ii) and its median pair ratio at 9,000 tokens gains 5 % or more. Otherwise it is removed and the
  reading stays in the journal. Nothing becomes a default in either case.
- Mutants, each seen red with its site counted: the visible range starting one key late; the bias built for the
  unshifted range; the full layers narrowed as if they were sliding.

`cargo test` in debug on the touched crates, `cargo clippy --all-targets` at zero warnings, the Linux half
type-checked by `ops/check-linux-cuda.sh clippy -p llvq-llm`.

## Predictions

D, the read on Metal:
- Bits: the digests differ (confidence 65 %). Max |dlogit| at most 2e-3, the same argmax at every compared row and
  the same 32 tokens at both lengths (confidence 85 %).
- Read of 9,000 tokens: x0.87 [0.80 ; 0.94] of the reference, from about 140 s to about 122 s. Computed from the key
  counts: the score work of the sliding layers falls from about 2,400 keys a row to about 1,100, and the four full
  layers keep theirs.
- Read of 2,000 tokens: x0.95 [0.91 ; 1.00].
- Peak footprint: not higher, 0 to 0.3 GB lower.

E, the decode after a long prompt, if the ring holds more than the window:
- Decode after 9,000 tokens: from 22.7 tok/s to 25.0 [23.5 ; 27.0].
- Decode after 5 tokens: within 1 % of the reference.

## What this lot does not do

No card, no paid run. No default changes. Nothing on CUDA or on the CPU, where the same statement exists: each is a
measurement of its own. The spread of four reference reads at 9,000 tokens was 11 % on 2026-11-26, so a gain under
5 % is not claimed.
