# Preregistration. The speculative route's last card job: where a verify row's time goes, and prompt lookup under a policy at the three sizes

Written on 2026-11-08 on branch `claude/spec-4e-0058c6`, once the code it describes existed and passed its host
checks. Timestamped before the first measurement, then never edited; a departure goes to
`preregistration-spec-final-2026-11-08-ECARTS.md`.

The operator gave the go on 2026-11-08 for the night: "Terminer le speculatif et tout les essais asoscié pour ça ut as
15$ de test max, pas de test inutile". The cap is $15 for every job of the night, over a registry of $304.05 across
branches (*computed* by the audit, PR #29). This is the night's first job. Cost: about **$1.26** on l40sx1
(*estimated*, §7), timeout 75 min, **$2.25** at worst (*computed* at $1.80/h).

## 1. Why this run exists

Every speculative arm measured so far loses to greedy because the verify pass is dear: two, three and four rows cost
1.95, 2.61 and 3.39 greedy tokens at 4B, 2.05 / 2.81 / 3.79 at 8B, 2.25 / 3.10 / 4.11 at 14B (*measured*,
`docs/mesures/spec-gate-2026-11-07.txt`). The row phases alone would add 0.378 of a token a row (*measured*,
shaker-4e). Nobody has measured where the other 0.3 to 0.9 of a token goes. The integer-dot rows (B1) would cut the
Tetra part, but they change the logits, so they cannot serve a greedy-identical decode without a new served object.

A replay at $0 shows that prompt lookup cannot pass greedy on free text with today's verify, whatever the policy.
The replay takes the greedy texts of spec-lookup, retokenized, and the verify costs above. Its best policy reads x0.95
to x0.98 on chat, code and summary, and x1.13 on retrieval at 4B (*computed*,
`ops/spec_lookup_replay.py`; it reproduces the old plain policy at x0.718 against x0.722 measured on chat).

So this job answers two questions of `docs/plan-speculation.md` and no other:
1. **Where does an extra verify row's time go**, phase by phase, at the three sizes? That decides whether an f32
   verify can get near 0.378 a row (a restructure of the kernel) or not (then only B1 or a new object moves it).
2. **What does prompt lookup give on the served object, decode and end to end**, with rounds without a draft on 4E's
   token launch and a policy that drafts only from repeated trigrams? That is the step-2 number the operator asked
   for, both ways, and the verdict on serving it.

## 2. What the code holds at the commit that carries this file

- `pers_4ev.cu`: `Pers4vLaunch` carries an optional stamps pointer (136 B). With it, thread 0 of each CTA writes the
  grid clock (`%globaltimer`) at its start and after each of the pass's 2 + 6 x layers grid waits. Nothing a row
  computes reads it. The last row of a group takes the vector ln path when the area holds one more row (the same
  values as the scalar path; host-checked). Verify module: 288,366 bytes, sha256 `d7686d86...` (from 287,744,
  `ab162e7f...`).
- `pers::p4v_phases`: event e's time is the median over CTAs; phase e lasts from event e - 1 to e; phases summed over
  layers as embed, qkv, rope (q/k-norm, RoPE, KV append), attn, o, gateup, down, head.
- `served_decode::continue_lookup`: a round with no draft calls 4E's token launch (the served symbol), not a verify
  of one row. `speculative::LookupPolicy { min_n: 3, probe: 1 }`: n-grams of length 3 only; a round drafts one token
  until one drafted token was kept, then up to k; a lost draft goes back to one.
- `specrun <sealed> lookup-n3p1 3 all`: install; `phases`; the six prompts; GSM8K.

### Host checks at $0

- `the_4e_verify_rows_are_the_served_chain`: 1, 2, 3 and 4 rows, stamped and plain, equal the served chain token by
  token (tokens, KV caches, residual, barrier count).
- `the_4e_verify_mutants_are_killed`: 14 mutants, among them each stamp store and the last row's scratch.
- `the_verify_launch_mirror_matches_the_kernel`: the 136-byte struct.
- `the_verify_phases_sum_their_events`, `the_lookup_loop_is_the_greedy_loop` (three policies),
  `the_lookup_policy_parses_and_probes`.
- What the host cannot check: any time; the driver ptxas; the clock's grain on the card.

## 3. What runs

One image built from the commit that carries this file. One job, l40sx1, `ops/jobs/spec-final.sh`.
1. Oracle on Qwen3-0.6B (rule 10).
2. For 4B, then 8B, then 14B: the sealed file by size and sha256, its v9 config by sha256, then
   `LLVQ_CONFIG=<config> specrun <sealed> lookup-n3p1 3 all`:
   - **phases**, at the chat prompt's context and the rag prompt's: 2 discarded and 5 timed one-token 4E launches;
     one stamped 4E launch of 9 tokens (`p4e_phase_report`); for n = 1..4 rows (the greedy tokens), 2 discarded and
     5 timed pairs of a stamped and a plain verify launch, each gated against the greedy tokens;
   - **door**: chat, code, summary, edit, rag, the house prompt; 256 tokens; greedy then the lookup arm; 1 + 5
     rounds; door tok/s (prefill and decode) and decode tok/s, ratios round by round;
   - **gsm8k**: the 50 pilot problems, greedy then the lookup arm.
3. If a 4B gate fails, the job stops after the 4B.

## 4. Gates

Each voids its size:
1. The image builds; `oracle` prints MATCH. No `LLVQ_*` in the job environment.
2. The sealed file and the config at their sha256.
3. The 4E module 252,757 B at `1b422b62...`, row form 7; the verify module 288,366 B at `d7686d86...`.
4. The drafter line names n-grams 3 to 3 and the probe of 1.
5. The greedy door at the served v9 fingerprint (61d0654532d35120, c817aeaf2f5d1116, dc4f9ab9aa24c4d3).
6. Phases printed at both contexts, 1 to 4 rows each; every phase launch's tokens are the greedy ones (specrun
   refuses otherwise).
7. Identity at the six prompts in every round, and on the 50 GSM8K completions.

## 5. Outcomes reported, never gated

- Per size and context: the one-token 4E launch (us); 4E's stamped phases; the verify at n rows: each phase (us),
  the stamped span, the plain wall, the plain wall over a one-token 4E launch, an extra row's cost in us and in
  one-token launches, and **its split by phase** (the increment of each phase from n = 1 to n = 4, over 3).
- Per size and prompt: door and decode tok/s, greedy and lookup, ratios round by round; tokens a verify, acceptance.
- Both tok/s are published: decode alone and end to end (the door, prefill included), each labeled.
- GSM8K correct counts.

## 6. Decision rules

- **A, identity.** A size where any arm moves a token is void and reported.
- **B, lookup.** Served as an opt-in only if, at all three sizes, the decode reads at least x1.00 of greedy on chat,
  code, summary and GSM8K (each) and at least x1.10 on edit or rag. Then it is proposed to the operator; nothing
  is wired into the runtime tonight. Otherwise prompt lookup closes on today's verify, with its numbers.
- **C, the verify.** Of an extra row's increment (n = 1 to 4, over 3, plain wall):
  - attn + rope + head + embed at 40 % or more: a restructured f32 verify (attention and the per-row prologues
    shared or spread over the grid) is the next lever, its gain estimated from these numbers;
  - qkv + o + gateup + down at 70 % or more: the cost is in the row phases themselves (the K-row dot and each row's
    norm and rotation inside them). Only fewer bytes or cheaper arithmetic a row moves it: B1, which needs a new
    served object.
  Either way, kernel work for a later go, written into the plan with its estimate.
- **D, EAGLE-3 on the card (the night's conditional job).** It runs only if the acceptance measured on the Mac at
  $0, with this job's verify costs, projects at least x1.10 of greedy decode at 4B on chat and code (*computed* by
  the plan's model). Otherwise it does not run.

## 7. Signed predictions

*Estimated* from spec-gate, spec-lookup, shaker-4e and the replay; nothing fitted to a number of this run.

| quantity | point | interval |
|---|---|---|
| identity, every size, prompt and phase launch | **holds** | |
| an extra verify row at 4B, chat context, in one-token 4E launches | **0.85** | [0.55, 1.2] |
| the same at 14B | **1.0** | [0.65, 1.4] |
| share of the increment in qkv + o + gateup + down, 4B chat | **75 %** | [55, 90] |
| share in attn + rope, 4B chat | **10 %** | [3, 25] |
| share in head, 4B | **12 %** | [5, 20] |
| lookup decode / greedy at 4B: chat, code, summary, GSM8K | **x0.95** each | [0.85, 1.02] |
| lookup decode / greedy at 4B: edit | **x0.95** | [0.8, 1.1] |
| lookup decode / greedy at 4B: rag | **x1.10** | [0.95, 1.25] |
| lookup decode / greedy at 14B: rag | **x0.92** | [0.8, 1.05] |
| lookup door / greedy (end to end), 4B rag | **x1.04** | [0.95, 1.12] |
| rule B (lookup served) | **not met** | |

## 8. Cost

spec-lookup ran greedy and three lookup arms in 16 min (4B) and 59 min (8B, 14B) billed (*measured*). This job runs
greedy and one arm, plus the phases (about 90 launches a context). About 7, 12 and 19 min of specrun, plus the oracle
and the image pull: **about 42 min, $1.26** (*estimated*). Per-size timeouts 14, 22 and 33 min; job timeout 75 min,
**$2.25** at worst (*computed*).

## 9. What this run does not establish

- Any integer-dot verify, any EAGLE acceptance, any kernel restructure: the phases say where the time is, not what a
  rewrite would save.
- Phase stamps split the pass at its grid waits only: inside a phase, a row's prologue and its dot are not apart.
- Long contexts (both contexts are under 500 positions), batches, any card but the L40S.
