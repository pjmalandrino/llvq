# Prereg: D1, the cache read once for a group of rows in the chain prefill, sealed 4B

Written on 2026-11-11 on branch `claude/arbre-prefill-2026-11-11`, once the code it describes existed. Timestamped
before the first measurement, then never edited; a departure goes to
`preregistration-large-prefill-d1-2026-11-11-ECARTS.md`.

Go: the operator, 2026-11-11, card spending inside a **$25** cap; about **$11.9** spent or committed before this job
(relayed by the orchestrating session). Cost: about **$0.57** on l40sx1 (*estimated*, "Cost"), timeout 29 min,
**$0.87** at worst.

Measured code: the commit that holds this file. The image is published from a clean detached checkout of the commit
that holds this file's stamp. D1 of `docs/mesures/large-prefill-contexte-2026-11-11.txt` section 3, after the first
job (`large-prefill-2026-11-11`): at 8,191 tokens the attention phase is 77.9 % of a 16-row chain pass, 34.4 of
44.4 ms (*measured*, its `chain.txt`), and each attention job serves one row, so 16 rows read the cache 16 times.

## The arm

`a12ipg` (`llvq-cuda/kernels/pers_4et_a12g.cu`, `TreeForm::A12ipg`, selected only by its name): a12ip's pass, its
body copied from a12's with two edits (the area check, the attention call), a host test pinning that copy. a12 and
a12ip keep their module text. Only the attention phase differs:

- A split that ends at or below pos0, (s + 1) x 256 <= pos0, is read whole from the cache by every row of the launch,
  and every quantity of a12ip's job on it (units, tiles a step, tiles, steps, each tile's count) is the same for every
  row. There are pos0 / 256 such splits.
- Those splits are served by shared jobs (group of 4 rows, split, KV head): each step of K and V is staged once, then
  each row of the group runs a12ip's tasks on its own q and unit states, `pers_attn_tile` on the same staged tile, in
  the same step and tile order. Each state receives the same operations in the same order as in a12ip's job.
- The other splits (the one that holds pos0 and the tree's positions, any later one) keep a12ip's job, one row each.
- Each row of a job then merges its units, writes its split state and draws its ticket on its own counter, as a12ip.
  The fold runs in split order whichever job draws the last ticket.
- The ring holds 2 stages, not 3 (copies only, no bit moves). Shared memory: 93,568 B a CTA against a12ip's 78,848 B,
  under the 101,376 B an sm_89 CTA may have. Group of 5 at 2 stages needs 103,936 B; group of 4 at 3 stages 109,696 B.
- No new arithmetic, no copy, no buffer on the card grows: b/param of the file (2.73) and the row buffers do not move.

Expected reads at 8,191, R = 16: 4 groups instead of 16 rows read the 31 shared splits, and each of the 16 rows reads
the last split, about 5.3 GB a pass against 19.3 GB (*computed*: 1.208 GB a row's whole cache, x 4 x 31/32, plus
16 x 1/32). The note's 4.8 GB counted no per-row tail.

## Mac gates (before the stamp)

Host twin (`llvq-cuda/tests/tree_verify_host.rs`, the 4B's widths, two layers, random caches; release, one thread):

- `a12ipg_rows_equal_a12ip_rows_at_short_and_long_caches`: chains of R = 1, 5, 16 at pos0 9 (no shared split),
  500 (one shared split, the rows' lengths crossing 512 inside a group) and 1,000 (three shared splits), and a
  branching tree of 16 at 1,000, with the 4B's parts and with the 8B's (Tetra o, untied head): every row's argmax,
  logits, residual, K, V and taps of a12ipg equal a12ip's bit for bit.
- `a12ipg_chain_prefill_equals_the_one_row_path`: 37 tokens after 600 cached positions, a12ipg at R = 16 and R = 5
  against a12ip's R = 1 path, every row and the hand-over row.
- `a12ipg_mutants_are_killed`: a row scoring with its group neighbour's staged q, and one split too many read as
  shared, each applied at exactly one site and killed.
- The earlier a12, a12ip, chain and loop twin gates; `the_a12ipg_pass_is_a12ip_with_the_grouped_attention` in
  `llvq-cuda/src/pers.rs`; clippy for the Linux target, zero new warnings.

Results: `docs/mesures/large-prefill-d1-2026-11-11-brut/mac-gates.txt`.

## Setup

One image, one job, `ops/jobs/large-prefill-d1.sh`, on l40sx1 only (no twin). Output `large-prefill-d1-2026-11-11/`
in `hf://buckets/Pier-Jean/jobs-artifacts/`. Inputs as the first job: the sealed 4B (sha256 886391a8),
`configs/qwen3-4b-tetra-e4-4e-v9.json`, `ops/data/spec-prompts-v1.jsonl` (sha256 f749eb35, 44 prompts),
`ops/data/spec-long-8k-v1.jsonl` (sha256 da4edbd6, 8,191 tokens), cut to 1,024 and 4,096 the same way.

One `prefillbench` process, `--form chain --arm a12ipg@8.10.8.20.6 --against a12ip@8.10.8.20.6 --r 16`, both arms
installed beside 4E (the same words read in place, no copy; the a12ip module adds about 0.4 MB of code), under the
two samplers of the first job. The phase sampler's tags carry the arm and the length (`time-<l>-<arm>`, one window a
round). In order:

1. Timing at 1,024, 4,096 and 8,191 tokens: 1 discarded + 3 timed rounds, each round a12ip then a12ipg, prefill then
   256 tokens of 4E greedy with no stop, tokens identical across rounds or refused. Time to the first token, prefill
   to the hand-over, decode tok/s after the first token, median [range] per arm; the ratio a12ip over a12ipg round by
   round; whether the two arms' 256 tokens are identical.
2. Identity of the arms at the three lengths, R = 16: argmax, last row's logits bit for bit, every layer's K and V over
   the prompt bit for bit (FNV-1a a layer).
3. Phase split of each arm: the 8,191 prompt prefilled once more, the last three passes stamped
   (`pers::p4t_phases_a9`). Pass = the sum of the phases, attention = phase `attn`, median of the three.
4. The 47 texts (the three long prompts' first rounds, the 44 house prompts) through both arms, paired in the process:
   the count of identical texts, and each first divergence.

No eager arm and no perplexity: a12ipg is held bit for bit to a12ip, which the first job gated against the eager
prefill (identity of the chain against the one-row path, texts, paired perplexity). If the identity holds, those
readings carry over; if it fails, this job says so and settles nothing on quality.

## Controls

Each voids the run.

1. `oracle` prints MATCH; the file, the config and both prompt sets at their sha256; no `LLVQ_*` in the job environment.
2. 4E's module is the served v9 module (252,757 bytes, sha256 1b422b62).
3. If the image fails to build, nothing is launched. If the a12ipg module fails to build or to fit a CTA on a SM, the
   process refuses (arm void) and the journal says so.

## Signed prediction

Signed by the orchestrating session, verbatim:

- the new arm equals a12ip bit for bit on the card (K, V, logits) at 1,024 and 4,096; 47 of 47 texts identical.
- a12ip in this job reproduces the first job: first token 0.70 s [0.66, 0.76] at 1,024; 4.31 s [4.1, 4.6] at 4,096; 13.4 s [12.7, 14.2] at 8,191.
- new arm: a pass at 8,191 takes 19 ms [15, 27] against 44.4; its attention phase 9.5 ms [6, 17] against 34.4; first token 6.2 s [4.8, 8.6] at 8,191; 2.7 s [2.1, 3.5] at 4,096; 0.63 s [0.52, 0.72] at 1,024.
- peak VRAM of the new arm within +0.10 GB of a12ip's at each length.
- Decision rule: PASSES if bit-identical and the first token at 8,191 comes at least 1.5 times sooner than a12ip's chain; otherwise the reading "repeated cache reads" is refuted or the design missed, and the journal says which from the phase split.

Operational reading, fixed here: "bit-identical" is the identity of step 2 holding at 1,024 and 4,096 and the 47 texts
identical; the 8,191 identity line is published beside it. "At least 1.5 times sooner" is the ratio of the two arms'
medians of the time to the first token at 8,191 (the round-by-round ratios published beside it). "A pass at 8,191"
and "its attention phase" are the medians of the three stamped passes of step 3 (the sum of the phases, and phase
`attn`). "Peak VRAM at a length" is the phase peak of that arm's timing tag at that length. When the speed rule fails,
the journal reads the phase split: an attention phase that fell to about 4 times less of its share and a pass still
over 29.6 ms (44.4 / 1.5) points at the rest of the pass; an attention phase that did not fall points at the design
(the shared jobs, their balance, the 2-stage ring) or refutes the repeated-read reading.

## Decision rule

| result | outcome |
|---|---|
| identity holds at 1,024 and 4,096, 47 of 47 texts, first token at 8,191 at least 1.5x sooner | PASSES |
| identity holds, first token at 8,191 less than 1.5x sooner | FAILS for speed; the phase split says whether the reads were the cost or the design missed |
| identity fails at any length, or a text differs | FAILS: a12ipg is not a12ip on the card; no speed reading carries the first job's quality readings |

## What gets published, and what does not get compared

Published: the loader's lines (registers, local bytes, shared bytes of both arms), the `card memory:` lines, both
samplers' files, the phase peaks, the timing table and the round-by-round ratios, the identity lines, both phase splits,
both text dumps and the pairing.

Not established: a served path (no engine or chat door calls the chain prefill); R above 16; D2 (tensor-core
attention); groups of rows in 4E's or the spec loop's verify pass (a12ipg serves any tree, but this job times chains
only); 8B, 14B; any card but the L40S.

## Cost

The first job's chain process loaded in 419.5 s with two tree modules (*measured*); a third module about 2.5 min more
(*estimated*): load about 10 min. Timing about 2.5 min (a12ip 22.5 s a round, a12ipg about 14 s, four rounds),
identity about 0.5 min, phases about 0.5 min, 88 texts about 2 min, oracle and controls about 1.5 min, container start
about 2 min. About 19 min, **$0.57** at $1.80 an hour (*estimated*). Timeout 29 min: **$0.87** at worst; about $12.8
of $25 at worst after it.
