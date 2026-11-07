# Preregistration. The speculative decode on the card: the 0.6B drafts, the served 4B, 8B and 14B check

**Written on 2026-11-07.** Stamped with OpenTimestamps before the first measurement, then never edited; a deviation
goes beside it in `proofs/preregistration-spec-gate-2026-11-07-ECARTS.md`.

Operator go: "ça vaut le coup pour notre famille de tester avec le 0.6B" (2026-11-07), and, relayed by the session
that started branch `claude/spec-0p6b-96ba73`, "Annonce le coût une fois. Lance sans redemander tant que le total de
la branche reste sous 5 $." Cost: **$1.50 expected, $2.70 at the timeout** on l40sx1 (*estimated*, section 7). Branch
total before this run: $0. Registry on main: $277.11 (*computed*, `docs/data/jobs.csv` at 529100a).

**Base commit:** the commit of `claude/spec-0p6b-96ba73` that carries this file and `ops/jobs/spec-gate.sh`.
**Image:** the Space rebuilt from it, passed as `IMAGE_SHA`; the launcher refuses another.

## 1. Why this run exists

- The served targets decode greedily through 4E v9: 221.5, 142.5 and 86.3 tok/s at the door (*measured*,
  `served-v9*-2026-11-07`). At the bit, the 4B's route tops out near 235-258 (`docs/plan-300.md`).
- `pers_4ev.cu` runs n rows of the decode in one cooperative launch (n up to 4), and on the host each row equals the
  served chain run token by token: tokens, KV caches, residual, at the 4B's widths, the 8B's Tetra o and untied
  head, rows across the first attention split; 11 mutants killed (`llvq-cuda/tests/pers_matches_served.rs`,
  `the_4e_verify_*`). The card has never run it.
- `speculative_greedy` keeps the leading drafts the target agrees with, then the target's own token, and rewinds both
  caches; on the host it equals the plain greedy loop on toy models whose output depends on the whole cache
  (`llvq-llm/src/speculative.rs`), 3 of 4 mutants killed, the fourth equivalent (it only drafts past the end).
- The drafter is the sealed Qwen3-0.6B, 0.339 GB, perplexity 40.30 against 18.93 for its f16 checkpoint (*measured*,
  `docs/mesures/draft-0p6b-2026-11-07.txt`).
- What no run has read: the verify pass on the card, the drafter on the card, the acceptance rate, the speed.

## 2. What runs

One job, l40sx1, `ops/jobs/spec-gate.sh`, timeout 90 min. Per size, 4B then 8B then 14B, one process:
`LLVQ_CONFIG=<size's v9 config> specrun <sealed target> configs/qwen3-0p6b-tetra-e4-4e-v9.json <sealed 0.6B> 1,2,3 all`.

1. Load the target under its v9 config, its 4E; the card memory (`cuMemGetInfo`). Load the drafter under its config,
   its 4E; install the target's verify pass; the card memory again.
2. Four prompts, 256 new tokens, cut at the chat's stop tokens: chat, code and summary (one user turn through the
   chat template, reasoning off, texts in `specrun.rs`), then the served door's house prompt, raw. For each: the
   greedy 4E reference, then 1 discarded and 5 timed rounds of greedy, k = 1, k = 2, k = 3 in turn.
3. The 50 GSM8K pilot problems (`gsm8k::select` at 50, the served pilots'), reasoning off, 1,024 new tokens at most:
   greedy, then k = 1, 2, 3.

## 3. Gates

Each one voids the size it concerns; a size that fails is reported and the job goes on.

1. The image builds; `oracle` prints MATCH.
2. Every object and config at its bytes and sha256.
3. Target and drafter install the 4E v9 module `1b422b62…` (252,757 B), the verify module is `ab162e7f…`
   (287,744 B), and its symbol line prints.
4. The greedy reference at the house prompt is the served v9 door's 256 tokens: `61d0654532d35120` (4B),
   `c817aeaf2f5d1116` (8B), `dc4f9ab9aa24c4d3` (14B).
5. **Identity**: every speculative arm makes the greedy tokens in every round at the four prompts, and the greedy
   completion on each of the 50 GSM8K problems, at k = 1, 2 and 3.
6. No `LLVQ_*` variable in the job environment.

Gate 5 is the result this run exists for. If it fails, the verify pass does not keep the bits on the card, and the
speed figures of that size are not reported as speculative decoding.

## 4. Outcomes reported, never gated

- Per size, prompt and k: door and decode tok/s, median and range over 5 rounds; the ratio to greedy formed round by
  round; acceptance (kept over proposed drafts); tokens a verify pass; catch-ups; the mean verify pass and drafter
  launch in ms, and the verify pass in greedy tokens.
- Per size and k: GSM8K decode tok/s over the 50 problems, correct count, acceptance.
- The best k per size, by the median decode ratio on chat, code and summary pooled as their geometric mean.
- The card memory added by the drafter and the verify pass, in GB.

## 5. Signed predictions

*Estimated* from the v8/v9 phase table at 4B (`step2-4e-2026-11-07`, `step3-4e-2026-11-07`) with the k2 cost of a
row (0.378 a pass, shaker-4e) applied to every row phase, the prologues and attention costed per row:

- Identity (gate 5): holds at the three sizes, every prompt, every k.
- The verify pass of k + 1 rows in greedy tokens at 4B: x1.4 [1.2, 1.7] at k = 1, x1.8 [1.5, 2.2] at k = 2,
  x2.2 [1.8, 2.7] at k = 3. At 8B and 14B no higher.
- The drafter, a 4E launch of the 0.6B: 1.0 ms a token [0.6, 1.6].
- Acceptance at k = 1: chat 0.55 [0.35, 0.75], code 0.65 [0.45, 0.85], summary 0.60 [0.40, 0.80], GSM8K 0.65
  [0.45, 0.85], the house prompt 0.8 [0.5, 0.97].
- Decode ratio to greedy at the best k, chat, code and summary pooled: 4B x1.00 [0.80, 1.20], 8B x1.08
  [0.85, 1.30], 14B x1.15 [0.90, 1.40]. Best k: 1 at 4B, 1 or 2 at 8B and 14B.
- Card memory added: 0.5 GB [0.3, 0.9].

## 6. Decision rules

- Gate 5 holds at a size and its pooled ratio has its whole range above x1.00: the speculative decode is proposed
  for that size, under a config field the operator decides on. Otherwise it stays a measurement.
- Gate 5 fails: the failing row phase is found on the host first (the card's contraction or ordering), no rerun
  before it is.

## 7. Cost

l40sx1 at $1.80/h. Image pull and start 5 min; per size the load and installs 1 to 2 min, the four prompts
(25 generates of 256 tokens each) 3, 4 and 7 min, GSM8K (4 x 50 problems) 5, 8 and 13 min: about 50 min, $1.50
(*estimated*). Timeout 90 min, $2.70. The platform has billed up to 28 min past a timeout: $3.54 at worst, under
the branch's $5.

## 8. What this does not establish

A drafter on CPU (the CPU Tetra branch, `claude/hopeful-swirles-b4155f`, had no commit when this was written), a long context (the prompts stay under 600
positions), a batch above 1, any card but the L40S, and the drafter's own quality beyond the acceptance rate.
