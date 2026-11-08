# Preregistration. Prompt lookup on the served 4B, 8B and 14B, through 4E's verify pass of today

Written on 2026-11-08 on branch `claude/spec-4e-0058c6`. Timestamped before the first measurement, then never edited;
a departure goes to `preregistration-spec-lookup-2026-11-08-ECARTS.md`.

The operator gave the go for step 2 of the speculative route on 2026-11-08 ("Go on", relayed by the main session in
answer to "on lance l'étape 2 ?"), starting with the first card run of prompt lookup. The go names no cap; the cost
below is announced before the launch and the launch waits for a cap that covers its worst case.

## 1. Why this run exists

`docs/plan-speculation.md` projects the doors prompt lookup would reach with the integer-dot verify: 251 to 300 tok/s
at 4B for α 0.3 to 0.5 a token (*estimated*). α is assumed, not measured on our targets or our prompts. This run
measures it by workload, through the code that would serve it, and checks that every token stays the greedy token.

Today's verify pass costs 0.95 of a token an extra row (*measured*, spec-gate). Prompt lookup cannot speed the door
on it; the speed it reads here is today's, and the integer-dot verify is not in this image. The number this run
carries forward is the tokens a verify, by prompt and by k.

## 2. What runs

Image: the Space `Pier-Jean/llvq-runner-cuda` at sha `c25e611da92d21a5442fd3ad8fb9f7e5697217ab`, built from commit
1dd0973 of this branch, which carries `specrun <sealed> lookup` (c535044). One job, l40sx1, `ops/jobs/spec-lookup.sh`.

1. Oracle on Qwen3-0.6B (rule 10).
2. For each of 4B, 8B and 14B: the sealed file at its bytes and sha256, the served v9 config at its sha256, then
   `LLVQ_CONFIG=<v9 config> specrun <sealed> lookup 1,2,3 all`:
   - the target's 4E v9 and the verify pass installed, card memory before and after;
   - six prompts, 256 new tokens: chat, code, summary, edit (a function to return renamed, most of it copied), rag (a
     question answered from a passage, quoting it), and the served door's house prompt; greedy then k = 1, 2, 3, one
     discarded and five timed rounds, the tokens of every arm checked against the greedy reference in every round;
   - the 50 GSM8K pilot problems, zero-shot, reasoning off, 1,024 new tokens at most; greedy then k = 1, 2, 3, every
     completion checked against the greedy one.

The drafter: the tokens that followed the latest earlier occurrence of the context's last n-gram, n from 3 down to 1,
over the prompt and the text so far (`speculative::lookup_draft`). A round with no match verifies one row.

## 3. Gates

Each voids its size:

1. `oracle` prints MATCH (voids the job).
2. The sealed file at its bytes and sha256; the config at its sha256.
3. The target's 4E module 252,757 bytes at `1b422b62…`, row form 7; the verify module 287,744 bytes at `ab162e7f…`;
   the drafter line names prompt lookup.
4. The greedy door's 256 tokens carry the served v9 door's fingerprint (`61d0654532d35120`, `c817aeaf2f5d1116`,
   `dc4f9ab9aa24c4d3`).
5. Identity: every arm makes the greedy tokens at the six prompts in every round, and the greedy completion on each
   of the 50 GSM8K problems.
6. No `LLVQ_*` variable in the job environment.

## 4. Outcomes reported

By size, prompt and k: tokens a verify (`tokens_per_round`), acceptance of the drafted tokens, door and decode tok/s
with their ratio to greedy round by round, the verify pass in greedy tokens. GSM8K: the same summed over 50 problems,
and the correct count. The card memory the verify pass adds.

From the tokens a verify t at k, α is read as the per-token acceptance whose geometric round gives t:
(1 - α^(k+1)) / (1 - α) = t (*computed*). That α replaces the assumed one in the plan's table.

## 5. Decision rules

- **A, identity.** Any divergence voids its size and stops the route: no further speculative job until its cause is
  found.
- **B, acceptance.** Reported by workload, never gated. The plan's prompt-lookup rows are rewritten with the measured
  α; rows whose α is under 0.2 are marked as falling back to greedy.
- **C, nothing else follows from this run.** The integer-dot port into 4E is outside the operator's go of 2026-11-08.

## 6. Signed predictions

*Estimated*, nothing fitted to a number of this run. Tokens a verify at k = 3, the same at the three sizes within 0.2:

| prompt | tokens a verify, k = 3 | decode / greedy at the best k |
|---|---|---|
| chat | 1.2 [1.0, 1.5] | x0.75 [0.55, 0.95] |
| code | 1.4 [1.1, 1.8] | x0.75 [0.55, 1.0] |
| summary | 1.6 [1.2, 2.2] | x0.8 [0.6, 1.1] |
| edit | 2.6 [1.8, 3.4] | x1.0 [0.7, 1.4] |
| rag | 2.0 [1.4, 2.8] | x0.9 [0.65, 1.2] |
| door (house prompt, loops) | 3.2 [2.4, 3.9] | x1.1 [0.8, 1.5] |
| GSM8K | 1.5 [1.2, 2.0] | x0.75 [0.55, 1.0] |

Identity holds everywhere. The card memory added by the verify pass stays within 0.05 GB of spec-gate's (+0.237,
+0.271, +0.237 GB less the drafter, whose share spec-gate did not split).

## 7. Cost

spec-gate's second run took 69 min for the same three sizes with four prompts and a 0.6B drafter (*measured*). This
job has six prompts, no drafter: about 17, 23 and 35 min at 4B, 8B and 14B, about 75 min with the oracle,
**$2.25** (*estimated*, l40sx1 at $1.80/h). Timeout 110 min: **$3.30** at worst. Per-size timeouts 25, 35 and 50 min.
Spent before it: $280.53 over the rows of `docs/data/jobs.csv` on this branch (*computed*).

## 8. What this run does not establish

Any speed-up of prompt lookup on the integer-dot verify; any perplexity under B1; EAGLE; any card but the L40S;
prompts beyond these six and GSM8K.
