# Preregistration. plan-300 step 2: 4E v7 (K2) and v8 (K3a's latency half), against v6

**NOT STAMPED. NOT LAUNCHED.** Written on 2026-11-07 on branch `claude/route-300-toks-de97fe`, once the code it
describes existed. To be timestamped before the first measurement, then never edited; a departure goes to
`preregistration-step2-4e-2026-11-07-ECARTS.md`.

The operator's go for step 2 alone was relayed on 2026-11-07, about $1.1, to be asked again above $1.50. Cost: about
**$0.36** on l40sx1 (*estimated*, §8). Timeout 25 min: **$0.75** at worst (*computed* at $1.80/h). Spent before it:
$273.96 (*computed*: `docs/data/jobs.csv`'s $273.55 plus the served-v6 door's $0.41, not yet in this branch's
registry).

## 1. Why this run exists

Step 1 (job 6aef03fa, `docs/mesures/step1-micro-2026-11-07.txt`) found no Tetra row lever, so 300 tok/s at the bit
is out. plan-300 continues toward 279 to 290 tok/s at most, on the leads that cut non-row work.

v6's stamps at 5 prompt tokens, mean CTA work, us a token (*measured*, pers-4e-v6):
- MLP down input 254.6, against a gate of 200 that v6 missed;
- ln1+rotation 318.9 and ln2+rotation 157.8: the same work, with ln1 paying more post-barrier reads;
- attention: wall 643.0, slowest CTA 621.8, mean CTA 238.9, so slowest over mean 2.60.

K2 reads the remaining prologue cost as L2 round trips after each grid barrier: the argument struct's fields, the
CTA's row slices through a second dependent load, the sign words. K3a reads attention's wall as latency: at 5
positions a row holds at most 69 positions, attention runs 8 jobs of 4 serial steps, and only the next step is in
flight. This run measures both.

## 2. What the code holds at the commit that carries this file

`llvq_pers.cuh` and `pers_4e.cu`, commit b7c8406, plus the driver change below.

- **v7, `PERS_4E_ROWS` 5.** v6's rows and prologues, plus:
  - `pers_grid_wait_gap` (§7c): arrive at the barrier, run a staging step, then spin.
  - At launch, the arguments are copied once into a v7 area in shared memory (header word `PERS_HDR_V7`).
  - During each layer's last barrier: the next layer's struct, its row slices, ln1's signs and `small`.
  - During the o barrier: ln2's signs and `small`. During the inner wait: the down input's signs and `small`.
  - The o phase widens its input from 16-byte vectors (8 serial loads in v6). The mix reads `small` as float4.
  - Shared memory 65,344 B at the 4B (*computed*).
- **v8, `PERS_4E_ROWS` 6.** v7, plus attention's steps staged by `cp.async.cg` into a ring of three stages
  (`PERS_ATTN_RING` 3), zero-filled past the row's end. Tiles, units and folds are unchanged. Shared memory 85,952 B
  at the 4B (*computed*), under the 101,376 B opt-in.
- **Install.** `persist4e_cuda::install_rows` takes forms 5 and 6 with v6's alignment check, adds `ctx`, and refuses
  sign widths past the area's 576 words.
- **Driver.** `pers::P4E_BENCH_FORMS` is `[v4, v6, v7, v8]`. `fusedrun LLVQ_PERS_4E` installs the four modules at 512
  threads. A module that does not install prints `not run` and the others go on. Changed by this file's commit:
  - each form's rate is divided by every earlier form's, round by round, so `4E v7 / 4E v6` and `4E v8 / 4E v6`
    print;
  - a v4 or v6 divergence voids the run as before. A v7 or v8 divergence prints `DIVERGES`, voids that form's
    numbers, drops it, and the others are timed.

Host checks at $0 (`llvq-cuda/tests/pers_matches_served.rs`, under nice 10):
- `the_4e_v7_staged_constants_are_the_served_chain`: identity on every served rotation width, the -0 column, the
  q/k/v and o phases and the token loop (four geometries, past position 256, three consecutive launches, stamps).
- `the_4e_v7_mutants_are_killed`: 16 mutants caught.
- `the_4e_v8_attention_ring_is_the_served_chain`: attention equal to `attn_ref` on nine row lengths at three
  geometries, and the token loop.
- `the_4e_v8_mutants_are_killed`: 7 mutants caught on the ring's indexing and reader barrier.
- The host's `cp.async` copies at once, so v8's waits are held only by this run's identity check on the card.

## 3. What runs

One image built from the commit that carries this file. One job, l40sx1, `ops/jobs/step2-4e.sh`. The sealed 4B
(`886391a8…`) is checked by size, then sha256.

1. **Oracle** on Qwen3-0.6B (rule 10).
2. `LLVQ_FUSED_LAYOUT=tetra48 LLVQ_EMBED=q4 LLVQ_ROT_SHARE=1 LLVQ_FUSE=0 LLVQ_PERS_4E=64 fusedrun <sealed> 64`.
   - Four runtimes at 512 threads, v4, v6, v7, v8, each its own module.
   - At 5 tokens (measurement form, kernel on every row) and at 8,192 (the served config, inside the T8).
   - Per context: each form's tokens against the eager path's, 2 + 5 interleaved rounds, then one stamped launch a
     form.

## 4. Gates

Each voids the run:

1. The image builds; `oracle` prints MATCH.
2. The sealed file at its bytes and sha256.
3. Event tracking off; a RoPE form equal to candle's.
4. **v4 and v6** tokens equal to the eager path's at both contexts.
5. No `LLVQ_*` variable in the job environment.

Each voids one form's numbers, not the run:

6. **v7 or v8** tokens differ from the eager path's at a context, or its module does not install. The first divergent
   token goes back to the host.

## 5. Outcomes reported, never gated

- Each form's registers, local bytes and shared bytes under the driver's ptxas.
- Per context:
  - each form's tok/s, median and range, and its ratio to the eager path round by round;
  - v7 over v6, v8 over v6, v8 over v7, round by round.
- Each form's phase table and its 7 inner parts (CTA work, slowest and mean).

## 6. Signed predictions

*Estimated*. The rates take plan-300 §4's K2 and K3a arithmetic at the midpoint of its high and low (41 % realised)
bounds; v8 counts only the latency half of K3a's 0.51 ms. Nothing fitted to a number of this run.

| quantity | point | interval |
|---|---|---|
| v4, v6, v7, v8 tokens against the eager path, both contexts | **identical** | |
| v7 and v8 load without local memory (driver ptxas) | **yes** | |
| v4 at 5 tokens, tok/s | **190.2** | [186, 194] |
| v6 at 5 tokens, tok/s | **212.6** | [208, 217] |
| **MLP down input, v7, us a token (mean CTA, 5 tokens)** | **110** | [60, 220] |
| q/k/v ln1+rotation, v7 | **170** | [120, 320] |
| MLP ln2+rotation, v7 | **140** | [90, 165] |
| v7 at 5 tokens, tok/s | **223** | [210, 229] |
| v7 over v6, 5 tokens, round by round | **x1.05** | [0.99, 1.08] |
| v7 at 8,192, tok/s | **157** | [151, 161] |
| v7 over v6, 8,192, round by round | **x1.035** | [0.99, 1.06] |
| **attention slowest over mean CTA work, v8, 5 tokens** | **2.3** | [1.6, 2.8] |
| attention wall, v8, 5 tokens, us | **450** | [300, 660] |
| v8 at 5 tokens, tok/s | **233** | [215, 250] |
| v8 over v7, 5 tokens, round by round | **x1.045** | [0.99, 1.10] |
| v8 over v6, 5 tokens, round by round | **x1.10** | [1.00, 1.16] |
| attention wall, v8, 8,192, us | **2,380** | [1,900, 2,600] |
| v8 at 8,192, tok/s | **161** | [150, 175] |
| v8 over v6, 8,192, round by round | **x1.06** | [0.98, 1.14] |

The point for slowest over mean predicts that K3a's gate fails under v8: the ring shortens every CTA's steps, and the
job count per CTA, which sets the ratio, does not change.

## 7. Decision rules

- **Gate of K2 (plan-300 §6, the operator's).** v7's MLP down input, mean CTA work at 5 tokens:
  - at most 200 us a token: the lever holds;
  - above 200: it fails, and the post-barrier reads were not the remaining cost.
- **Gate of K3a.** v8's attention, slowest CTA work over mean CTA work at 5 tokens:
  - under 1.5: the gate holds;
  - at or above 1.5 with the wall down: the latency half worked, and the imbalance left is the job spread, K3a's
    other half, which v8 does not contain;
  - at or above 1.5 with the wall not down by 50 us: the steps were not latency-bound as read.
- **A, identity.** §4. A v7 or v8 divergence is reported with its token and voids only that form.
- **B, serving.** For v7 and v8 each, the ratio over v6 round by round at both contexts.
  - With the whole range above x1.00 at both contexts and its gate held, the form is proposed for a door reading: a
    config whose `rows` names it, a separate job.
  - `served.rs` refuses v7 and v8 today. Serving, and the door job, are the operator's call.
  - Otherwise the form stays a measurement.
- **C, the route.** Step 3 (K3b, K5) is not chained by this run. Its high bound is re-estimated from this run's walls
  before it is costed.
- **G.** A miss is reported as a miss, with the term of §6 it refutes. No threshold moves after the clock.

## 8. Cost

*Estimated* on l40sx1 at $1.80/h:

| step | time |
|---|---|
| container and image | 3 min |
| oracle | 1 to 2 min |
| fusedrun: four 4E modules built, 2 contexts x 7 rounds x 5 decodes, 8 stamped launches (pers-4e-v6 ran two modules and 3 decodes a round in about 4 min) | 7 min |

About 12 min, **$0.36**. Timeout 25 min: **$0.75** at worst.

## 9. What this run does not establish

- The served door. `fusedrun LLVQ_PERS_4E` checks each form's tokens against the eager path's at 5 tokens and under
  the served config at 8,192; a door reading of v7 or v8 needs a config that names it, which `served.rs` refuses
  until this reading exists.
- K3a's job spread over more CTAs, and K7 at 8,192.
- The final norm before the head; the 8B and the 14B; any card but the L40S.
