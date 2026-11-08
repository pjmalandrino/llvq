# Speculative decoding on the served Qwen3 family

A small model of the same family proposes tokens, the served target checks them in one pass, and the output is the
target's greedy output token for token. The drafter is Qwen3-0.6B in Tetra, which shares the tokenizer of every
served Qwen3. The operator's go: "ça vaut le coup pour notre famille de tester avec le 0.6B" (2026-11-07). It lifts
the hold of `plan-debit-noyau.md` K9 on neural drafters for this test.

## What exists

- **The drafter.** `qwen3-0p6b-sealed.bin`, 0.339 GB, sealed on the Mac with the 14B's recipe plus `o_proj` in int4
  (*measured*, `mesures/draft-0p6b-2026-11-07.txt`). The file carries an unread duplicate head table: 0.252 GB and
  3.10 b/param over what the model reads (*computed*). Perplexity 40.30 against 18.93 for the f16 checkpoint: Tetra
  costs the 0.6B a factor two, which only lowers the acceptance rate. Config
  `configs/qwen3-0p6b-tetra-e4-4e-v9.json`; it decodes through 4E v9 like the targets.
- **The verify pass.** `llvq-cuda/kernels/pers_4ev.cu`: n rows of the decode in one cooperative launch, n up to 4
  (`pers::P4V_KMAX`), row r the token at position `pos0 + r`. Each Tetra and int4 row phase decodes a weight once for
  every row of a group; the norms, rotations, RoPE, the append and attention run row by row, attention's jobs of every
  row dealt over the grid. A group is as many rows as the staging area holds at the phase's width: on the L40S's
  101,376 B, 4 rows at the 4B except the down (2), 4 at the 8B except the ln phases (3) and the down (1), 3 at the 14B
  except the down (1). Its own NVRTC module beside the served one (`Runtime4e::install_verify`), so the served text
  does not move.
- **The proof on the host.** Each row equals the served chain run token by token: tokens, KV caches, residual, at
  the 4B's widths, the 8B's Tetra o and untied head, rows across the first attention split, every staging size
  (`llvq-cuda/tests/pers_matches_served.rs`, `the_4e_verify_*`, 11 mutants killed).
- **The loop.** `llvq-llm/src/speculative.rs`: greedy acceptance, the target's caches rewound to the kept prefix, the
  drafter's rewound or fed its missing token when every draft was kept. Tested against the plain greedy loop on toy
  models whose output depends on the whole cache. `Decoder::continue_speculative` runs it on two 4E engines.
- **The gate.** `bin/specrun`, `ops/jobs/spec-gate.sh`, `proofs/preregistration-spec-gate-2026-11-07.md`.

## What the card read (2026-11-07)

[spec-gate](mesures/spec-gate-2026-11-07.txt), one L40S, *measured*. Identity holds: at the 4B, 8B and 14B, k = 1 to
3, every speculative arm made the greedy tokens on chat, code, summary and the house prompt in every round, and the
greedy completion on each of the 50 GSM8K pilot problems. The speed does not: k = 1 is best everywhere, and the decode
reads x0.569 to x0.676 of greedy on chat, code and summary, x0.68 to x0.74 on the looping house prompt, x0.68 to
x0.70 on GSM8K. Acceptance at k = 1: 0.43 to 0.69 on chat, code and summary, 0.75 on GSM8K. Card memory +0.237,
+0.271 and +0.237 GB.

| size | greedy decode, chat | k = 1 | ratio | 2-row verify, in greedy tokens | drafter, a launch at k = 1 |
|---|---|---|---|---|---|
| 4B | 221.8 | 126.3 | x0.569 | 1.95 | 2.5 ms |
| 8B | 142.3 | 84.1 | x0.591 | 2.05 | 2.5 ms |
| 14B | 88.6 | 51.3 | x0.579 | 2.25 | 2.5 ms |

## What decides the speed

A round of k drafts costs the drafter's k tokens plus one verify pass of k + 1 rows, and emits the kept drafts plus
one token. The verify pass shares the decode of every weight, but not the prologues, the attention, or the down where
the staging area cannot hold the rows. The card gate reads each term.

## Later

- A drafter on CPU, in parallel with the GPU, once the CPU Tetra branch has a matvec.
- The down's rows shared beyond the staging area (the rotated input held in L2), and the verify prologues shared.
- Feeding the drafter's missing token inside its next launch instead of a catch-up launch.
