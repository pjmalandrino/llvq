# Prereg: the window arm of Inkling 2 Mini behind a switch of the runtime, off (2026-11-27)

Written before the first measurement of this lot. Not edited after its timestamp; deviations go in `-ECARTS.md`
beside it. Operator's order: autonomy for the night of 2026-11-26, local work at 0 $. Cost: 0 $, this Mac only
(M3 Max, 64 GB). Predictions signed by the main session. The switch is off by default and hidden from the help:
turning the arm on, or showing the switch, is the operator's decision.

## Question

The window arm reads a 9,000-token prompt x0.855 [0.854, 0.856] faster through `inkrun`
([inkling-mac-fenetre](../docs/mesures/inkling-mac-fenetre-2026-11-27.txt)) and its paired perplexity is inside the
+-0.12 % band, +0.022 % [-0.045, +0.089]
([inkling-mac-fenetre-ppl](../docs/mesures/inkling-mac-fenetre-ppl-2026-11-27.txt)). Both journals say the runtime is
unchanged. Does the gain reach the first token `llvq serve` gives on a long prompt, and can the arm sit in the
runtime without moving one byte of what is served today?

## Object

`llvq serve` gains one switch, `--attn-window full|narrow`, default `full`, hidden from the help. `narrow` calls
`layers::set_window_scores(true)` at the load of Inkling 2 Mini on Metal under the fused stack. It is refused by name
on every other load (the CPU, CUDA, the candle stack, a Qwen3 file): the arm has been read on Metal only.

The prefix cache's key names the arm, so a file written under one arm is never loaded under the other. A key without
the field reads as `full`.

No default changes. `library.json`, the served configs and `llvq-llm` are not touched, except a test if one is needed.

## Gates

- H-i, the default is untouched. With the switch absent or `full`, the server's answer to the long prompt of
  `inkling-mac-serve-2026-11-26` (8,791 tokens) and to a short one is the text the committed build gives, token for
  token, and the load's lines are the same. The key of the default serializes to the bytes it had before the field,
  or the journal says which bytes moved and why.
- H-ii, the arm through the server. With `narrow`: the same prompts answered; the tokens compared to `full`'s.
  Equal tokens are expected and not required (the arm is class B); a difference is reported with its first position.
- H-iii, the first token. Cold reads of the 8,791-token prompt, prefix cache off, a fresh server each, `full` then
  `narrow` alternated, one discarded pair then three timed, one guarded job. Medians and ranges round by round.
- H-iv, the cache does not cross. A prefix file written under `full` is not loaded under `narrow`, and the reverse:
  a unit test on the key, and one live check in a working `LLVQ_HOME` (the server's line names the miss).
- Refusals: `narrow` on the CPU backend and on a Qwen3 file fails naming the switch.
- Mutants, each an in-place replacement on a committed file, the sites counted, the file restored: the key's field
  dropped from the key (H-iv's test red); the switch parsed and not applied (a test red).
- `cargo clippy --all-targets` on the Metal build and the Linux check at 0 warnings; `cargo test -p llvq-runtime`
  and `-p llvq-llm --features metal` in debug, counted by their `test result:` lines.

## Decision rule

The switch is committed, off and hidden, when H-i, H-iv, the refusals, the mutants, the lints and the tests hold.
H-iii is reported whatever it reads and decides nothing in the code. If a gate does not hold before 06:15, nothing
is committed in `llvq-runtime`: the diff is saved as a patch beside the raw files, the tree is restored, and the
journal says where it stopped.

## Signed predictions (the main session, before the run)

| quantity | predicted |
|---|---|
| H-i, same tokens and same lines under the default | probability 0.95 |
| H-ii, `narrow` gives the tokens of `full` on both prompts | probability 0.8 |
| H-iii, first token at 8,791 tokens, ratio narrow / full | x0.87 [0.84 ; 0.92] |
| H-iii, first token under `full` | 105 to 135 s |
| H-iv, a file of one arm not loaded under the other | holds, probability 0.9 at the first try |
| the lot ends committed before 06:15 | probability 0.6 |

Basis: `inkrun` and the server call the same `forward_pieces` with the same three settings, so the server's read is
`inkrun`'s plus the template and the HTTP path; the fixed part of a request dilutes the ratio a little.

## What this lot does not do

No card, no paid run. No reading of the arm on the CPU or on CUDA. No perplexity through the server. The arm is not
served and the README, the site and the Hub do not change.
