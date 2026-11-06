# Preregistration. 4E v3 on the card: the Tetra row stream, its tokens and decode rate at 5 and 8,192 prompt tokens with the phase table

**NOT STAMPED. NOT LAUNCHED.** Written on 2026-11-05 on branch `claude/serene-bohr-1edf69`, once the code it describes
existed. To be timestamped before the first measurement, then never edited; a departure goes to
`preregistration-pers-4e-v3-2026-11-05-ECARTS.md`.

The launch is the operator's, through the integration session. Cost: about **$0.27** on l40sx1 (*estimated*, §8).
Timeout 30 min: **$0.90** at worst (*computed* at $1.80/h). Spent before it: see `docs/data/jobs.csv` at launch.

## 1. Why this run exists

The target is the sealed 4B at 300 tok/s or more after a short prompt on the L40S. llama.cpp IQ2_XXS reaches 312.85 on
this card (*measured*, 2026-08-30). No ratio is formed across stacks (rule 5).

4E v2 (`docs/mesures/pers-4e-v2-2026-11-05.txt`, *measured*) gave the eager path's tokens:
- 178.7 tok/s at 5 tokens and 131.8 at 8,192;
- the rows' walls 4.75 ms of a 5.60 ms token;
- the rows' mean CTA work at 317 GB/s.

The host analysis of its phase table and of the card's own SASS (`docs/mesures/pers-4e-v3-host-2026-11-05.txt`) splits
that rate:
- the int4 rows (o, the head) streamed at 587-588 GB/s;
- the Tetra rows (q/k, gate/up, the Tetra downs) at about 260 GB/s (*computed*).

A Tetra block iteration costs 241 instructions on the card, like the one-launch pass's 243. But a Tetra row of d_in
2,560 has 4 iterations a lane and pays its fixed round trips every row: its first words, lane 0's rscale and tail,
and in gate/up the SiLU table. v3 removes those round trips from the row's critical path, keeping every row's
summation order (§2).

This run asks the card one thing, in one process: v3's tokens, rate and phase table against the eager path.

## 2. What the code holds at the commit that carries this file

Off by default; no served path moves (the served unit's digest `6193094310bc…` is a gate). The 512-thread symbols:

- **The Tetra row stream** (`llvq_pers.cuh::pers_tetra_stream`):
  - a warp takes a static list of rows;
  - the lane's blocks of the whole list form one sequence, with 4 items always in flight across row boundaries;
  - lane 0 loads a row's rscale, tail (and a down row's residual) when the row's first item is consumed;
  - per lane and per row the blocks are summed in the served order, then warp_sum, then the epilogue as before.
  - It runs on q/k, gate/up (the SiLU table lookup one pair late) and the Tetra downs.
- **q/k/v**: each CTA takes its share of the q | k rows and of the v rows. v2's contiguous slice left the int4 v rows
  to the last 24 CTAs.
- **Attention at a short context**: a step stages 4 or 2 tiles of each non-empty unit when a split holds 1 or 2
  units of positions, so a row of 64 positions takes 2 steps instead of 8. A CTA that runs an attention job issues its
  L2 prefetches after the job. v2: 8 CTAs carried the phase, 737 us against a mean of 260.
- Unchanged from v2:
  - the attention mapping at long contexts, the ticket fold, the norm spread, the int4 rows' prefetch, the tables
    staged once;
  - the threshold: positions below 1,280 are refused outside a measurement;
  - the 768-thread symbols (no stream).

Host checks at $0:
- 27 tests of the kernel text against the served text: the token loop on both variants, positions past 256, the
  attention at 9 lengths including 3 non-empty units;
- 21 mutants added by v2 and v3, all caught; v3's 9 are 8 on the stream and 1 on the step staging.

Under the devel ptxas 12.4 (indicative): 512 threads 120 registers, 0 B local. Never run on a card.

## 3. What runs

One image built from the commit that carries this file. One job, l40sx1, `ops/jobs/pers-4e-v3.sh`, the sealed 4B
(`sealed-4b-2026-09-23/qwen3-4b-sealed.bin`, sha256 `886391a8…`) checked by size then sha256.

1. **Oracle** on Qwen3-0.6B (rule 10).
2. `LLVQ_FUSED_LAYOUT=tetra48 LLVQ_EMBED=q4 LLVQ_ROT_SHARE=1 LLVQ_FUSE=0 LLVQ_PERS_4E=64 fusedrun <sealed> 64`:
   - 64 tokens a run, at the 5-token prompt (measurement form, kernel on every row) and at 8,192 tokens (the served
     config, inside the T8);
   - 2 discarded and 5 timed rounds of eager and 4E 512, 4E 768 where it loads;
   - one stamped launch a geometry.
3. `LLVQ_CONFIG=<kernel-from1280 config> fusedrun <sealed> 16`: the served door's digest and threshold line.

## 4. Gates

Each voids the run:

1. The image builds; `oracle` prints MATCH.
2. The sealed file at its bytes and sha256; the config is the committed one.
3. `cudarc event tracking: off (one stream, ordered by issue)`; `4E: RoPE forms equal to candle's rope on this card`
   with at least one form.
4. Identity: `4E at 512 threads: tokens equal the eager path's` at both contexts, no round moving a token; the 8,192
   context prints the served config inside the T8.
5. The served door prints `attention kernel from position 1280 on grow(256)` and the NVRTC digest `6193094310bc…`.
6. No `LLVQ_*` variable in the job environment.

A 768-thread geometry that does not load, or whose tokens differ, voids only its own numbers. Stamps whose launch
moved its tokens are void.

## 5. Outcomes reported, never gated

- Each loaded 4E symbol's registers and local bytes (the driver's ptxas).
- Per context: eager and 4E decode tok/s, median and range; 4E / eager round by round (*measured*, same stack).
- The phase table at each context.
- The rows' rate: 1.377 GB over the mean CTA's work summed over q/k/v, o, MLP and head (*computed* from the table).
- The Tetra rows' rate (*computed*, the rule of `ops/pers_4e_v3_estimate.py`):
  - from q/k/v, take out 36 × 3 us of fixed work and v's 50.1 MB at 560 GB/s;
  - from MLP, take out 36 × 7 us and the int4 downs' 158.8 MB at 560 GB/s;
  - divide the 748.9 MB of Tetra by what remains.

## 6. Signed predictions

*Estimated* by `ops/pers_4e_v3_estimate.py` from the v2 card table; nothing fitted to a v3 number.

| quantity | point | interval |
|---|---|---|
| 4E 512 tokens against eager, both contexts | **identical** | |
| 4E 512 loads without local memory (driver ptxas) | **yes** | |
| Tetra rows' rate on the mean CTA's work, GB/s | **480** | [330, 650] |
| rows' rate on the mean CTA's work, GB/s | **450** | [350, 560] |
| attention wall, 5 tokens, us a token | **300** | [200, 450] |
| attention wall, 8,192, us a token | **2,680** | [2,400, 2,900] |
| q/k/v wall over its mean CTA work, 5 tokens | **1.05** | [1.00, 1.15] |
| 4E 512 decode, 5 tokens, tok/s | **273** | [214, 333] |
| 4E 512 decode, 8,192, tok/s | **166** | [145, 182] |
| eager, kernel form, 5 tokens, tok/s | **132** | [128, 136] |
| eager, served config, 8,192, tok/s | **103.5** | [100, 107] |

300 tok/s at 5 tokens needs the Tetra rows at about 610 GB/s on the mean CTA's work, with the middle case's other terms
(*computed* by the script).

## 7. Decision rules

- **A, identity.** If 4E 512's tokens are identical at both contexts, v3 keeps the bits on this card. Otherwise the run
  is void, and the first divergence is found on the host with the card's dumps.
- **B, the stream.**
  - Tetra rows at 450 GB/s or more: the per-row round trips were the bound, and the remaining gap is the iteration's
    issue (241 instructions a block).
  - Under 330: the stream did not remove them, or the iteration binds. The next step is the card's SASS of the
    stream loop at $0 (a planesbench cubin dump in a later job, or the driver's register report).
  - In between: a curve point.
- **C, attention at 5 tokens.** 450 us or less is fixed. Otherwise the slowest CTA against the mean names the next
  host test.
- **D, q/k/v balance.** Wall over mean CTA work at 1.15 or less: the mixed split holds.
- **F, the target.** 4E at 300 tok/s or more over its whole range at 5 tokens is reported as the target reached in the
  measurement form only. Serving it needs the operator's license of the kernel form below 1,280: 4E does not compute
  the grouped arithmetic the served config runs there.
- **G.** A miss is reported as a miss, with the term of §6 it refutes. No threshold moves after the clock.

## 8. Cost

*Estimated* on l40sx1 at $1.80/h:

| step | time |
|---|---|
| container and image | 3 min |
| oracle | 1 to 2 min |
| fusedrun, v2's part of 536 s with planesbench's 270 s out | about 3.5 min |
| the served door | 1 min |

About 9 min, **$0.27**. Timeout 30 min: **$0.90** at worst.

## 9. What this run does not establish

- Serving 4E: below 1,280 a license 4E's arithmetic does not have; past it, the A2 reading and the operator.
- 4E at 8B or 14B.
- The one-launch pass, which this run does not time.
