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

## Gate 1, the integer-dot verify rows (2026-11-08)

[spec-micro](mesures/spec-micro-2026-11-08.txt), one L40S, the 4B's 252 Tetra matrices at 4E's geometry. An extra
verify row with B1's integer dot costs f = 0.187 of 4E's Tetra row (0.173, 0.187 and 0.201 at K = 2, 3 and 4), against
0.381 for a second f32 row (*measured*). That is BETWEEN on the prereg: PASS needed 0.15. In situ, the Tetra part of an
extra row is about 0.34 ms at 4B (*estimated*, f x 1.81 ms). The integer dot keeps 91.6 % of the rows identical to the
served rows and the rest one binary16 ulp away (*measured*), so it changes tokens: the single-token path must carry it
too, under the ±0.12 % perplexity band, for the verify to keep the greedy tokens. The prereg's gate 4 was checked on one
activation of four (deviation 3).

Doors a speculative round would reach on these terms, batch 1, L40S, best k ≤ 4 (*estimated*). δ is the extra row:
its Tetra part f x the Tetra rows (0.34, 0.68, 1.16 ms), plus the rest of the row (int4, head, attention, norms) at
0.30 ms at 4B, scaled with the door; the bracket puts that rest at 0.15 and 0.45. Greedy door T1 = 4.51, 7.02 and
11.59 ms. A draft step costs 0.02 ms for prompt lookup and 0.18, 0.29 and 0.40 ms for a one-layer head.

| drafter | α a token | 4B | 8B | 14B |
|---|---|---|---|---|
| greedy today (*measured*, served-v9) | | 221.5 | 142.5 | 86.3 |
| prompt lookup | 0.3 | 251 [244-259] | 159 [154-163] | 96 [93-99] |
| prompt lookup | 0.5 | 300 [285-316] | 187 [178-197] | 113 [108-119] |
| EAGLE-3 head | 0.5 | 284 [274-299] | 177 [173-186] | 108 [105-113] |
| EAGLE-3 head | 0.6 | 319 [304-335] | 198 [189-208] | 121 [115-127] |
| EAGLE-3 head | 0.7 | 363 [341-388] | 223 [211-238] | 136 [129-145] |

- A 2-row pass costs 1.14, 1.16 and 1.17 greedy tokens at the central rest (*estimated*), at the study's 1.15 gate
  and under ETAT §6's 1.4.
- Prompt lookup's α is the workload's: 0.3 to 0.5 where the text repeats (code edits, retrieval, summaries), near 0
  on free chat, where a round verifies one row and the door stays greedy's. It has not been measured on our prompts.
- The table assumes B1 in the verify only. B1 in the single-token path as well would lower T1 too (i16 reads x1.143
  over v2 alone); two stacked estimates are not printed.
- What it costs to get there: B1's integer dot ported into `pers_4ev.cu` and into 4E's single-token row, the
  perplexity gate, then a prompt mix on the card. Weeks of kernel work (the study's 3 to 4 weeks for lookup).

## Prompt lookup (2026-11-08)

The drafter is the context itself: the tokens that followed the latest earlier occurrence of the context's last
n-gram, n from 3 down to 1 (`speculative::lookup_draft`). No model, no memory, no training. A round with no match
verifies one row, which is the greedy token. `speculative::lookup_greedy` is the loop, tested against the greedy loop;
`Decoder::continue_lookup` runs it on one 4E engine and its verify pass; `specrun <sealed> lookup [ks] [what]` times
it on chat, code, summary, a code edit and a retrieval answer. It has not run on a card. On today's verify pass (0.95
of a token an extra row) it cannot pay: it needs the integer-dot verify above first.

## Published EAGLE heads for Qwen3 (2026-11-08)

A one-layer head that reads the target's hidden states is the drafter the study ranks first (`recherche-debit-maths.md`
§11.3). Three published heads fit our family with a clean license. Nothing here was run; revisions pinned on
2026-11-08.

| size | repo | revision | license | file | head params | reported acceptance |
|---|---|---|---|---|---|---|
| 4B | `AngelSlim/Qwen3-4B_eagle3` | `fd331e59626c8e95c392381a16ee59d518727fbb` | Apache-2.0 | 436,899,680 B, BF16, draft vocab 32k | 218 M | 2.08 tokens a round at k = 2 |
| 8B | `RedHatAI/Qwen3-8B-speculator.eagle3` | `08610ffa01dd9f16731fe8f627b85905b6aa51c4` | Apache-2.0 | 2,044,116,968 B, BF16, its own copy of the embedding | 400 M | 2.17 to 2.81 at T = 0.6, k = 2 to 7, thinking off |
| 14B | `RedHatAI/Qwen3-14B-speculator.eagle3` | `2d489f5a8e490507e4fba701a5515e291f211fdc` | Apache-2.0 | 2,775,244,776 B, BF16, its own copy of the embedding | 610 M | about 2.2 to 2.76 at T = 0.6 |

- Head params: the file's bytes over two, less the embedding the RedHat heads copy from the target (*computed*).
- Acceptance: as the repositories report it, on BF16 targets, at their temperature (*reported*, not ours). The
  4B's figure is at k = 2; at a per-token α it reads about 0.55. The RedHat range reads α 0.55 to 0.7.
- Stronger heads exist without a license (DCore, greedy τ 3.2 to 4.9) or under CC-BY-NC-SA (WhiFlash); MIT heads of
  five layers over the full vocabulary are too heavy to be a one-layer drafter. None is retained.

**Memory, whole model.** As published, in BF16: +0.87, +0.78 and +0.66 b/param at 4B, 8B and 14B, the embedding read
from the target (*computed*). That is a third of the served model's bytes, so a head runs in int8 (+0.45, +0.40, +0.34)
or int4 g64 (+0.24, +0.22, +0.19 b/param, 123, 225 and 343 MB) (*computed*). Int4 g64 matches the study's +0.21 /
0.20 / 0.17. The head's quantization is a second unknown on the acceptance.

**Expected acceptance on Tetra.** Not measured anywhere at 2.7 b/param. The 0.6B drafter on our targets accepted 0.43
to 0.69 a token at k = 1 on chat, code and summary (spec-gate, *measured*); a head trained on the BF16 target's
features reads Tetra's features, a shift of the same order as the target's own quantization. Range α 0.5 to 0.7
(*estimated*): 1.88 to 2.53 tokens a round at k = 3. Evidence on quantized targets is mixed: rotations matter where
they change the features the head reads (vllm-ascend #6914). Ours are applied inside each projection, so the residual
stream the head reads should be the unrotated one; this has to be checked on the code before any port.

**What a port needs.** 4E would emit the three hidden states EAGLE-3 reads (low, middle and high layers), run the
head's layer and its 32k head as a second pass a draft step, and map the draft vocabulary to the target's. Weeks of
kernel work, $0 of training. Training our own heads on Tetra features is the alternative: about one to two GPU-days a
size (*estimated*), never without the operator's go.
