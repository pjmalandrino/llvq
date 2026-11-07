# Route to 300 tok/s on the served 4B

**300 tok/s at the served door is out of reach at the bit: step 1 found no row lever (*measured*,
[step1-micro](mesures/step1-micro-2026-11-07.txt), 2026-11-07).** The five arms on the Tetra row loop read ×0.810 to
×1.002 over v2, under the ×1.15 bar. The exact leads that remain reach 279 to 290 tok/s on their high bound, or 284
to 296 if the int4 tile layout keeps inside 4E the ×1.338 it read alone (*estimated*). Adding K10, which moves
quality inside the T8 band, gives 289 to 301. 300 then holds only on the proportional door conversion, with every
lead at its full arithmetic, where 4E's versions delivered 41 %. The low bound stays 229 tok/s at the bit
(*estimated*). At 8,192 positions only an int4 KV cache reaches 300, on its high bound, and it moves quality, GB in
memory and the model class. Step 2 measured v8 at 227.9 tok/s in the measurement form, ×1.076 over v6, bit for bit
(*measured*, [step2-4e](mesures/step2-4e-2026-11-07.txt)).

State as of 2026-11-07. This is a study at $0: no run, no job, no measurement. It starts from 4E v6 served, which the
session `serve-4e-v6` is putting in place. Its base is [`plan-debit-noyau.md`](plan-debit-noyau.md) (K1 to K10, M1 to
M5), which it does not repeat. Every lead needs the operator's go and its cost announced (`CLAUDE.md` rule 1).

## 1. Starting point and target

| object | 5 positions | 8,192 positions | label, journal |
|---|---|---|---|
| v4, served door | 178.0 tok/s, 5.618 ms | 139.5 | *measured*, [served-v4](mesures/served-v4-2026-11-05.txt) |
| v4, measurement form | 190.2, 5.258 ms | 138.9 | *measured*, [pers-4e-v6](mesures/pers-4e-v6-2026-11-05.txt) |
| v6, measurement form | 212.6, 4.704 ms | 152.0, 6.579 ms | *measured*, pers-4e-v6 |
| v6, served door | 197.7 to 199.2 | 152.6 | *estimated* below |
| target, served door | 300, 3.333 ms | 300, 3.333 ms | operator, 2026-11-07 |

The door and the measurement form differ by ×0.937 at 5 positions (*computed*, 178.0 over v4's 190.0 in two jobs).
At 8,192 they agree within 1 %. The door's 5-token prefill costs 0.084 ms a token over its 256 tokens (178.0
against 180.7 decode alone). The other 0.271 ms, between the door's decode and the measurement form, has no known cause.

So 300 at the door asks the measurement-form token to reach 3.12 ms if the gap is proportional, 2.98 ms if it is a
fixed 0.355 ms. v6 must lose 1.58 to 1.73 ms, a third of its token. Every figure below quotes both cases as
"prop / add". The v6 door reading that `serve-4e-v6` takes replaces both estimates.

The weight floor is 1.594 ms, 628 tok/s (*computed*, 1.377 GB at 864 GB/s). Whole-token, v6 reads 293 GB/s; 300 at
the door asks 413 to 424 GB/s (*computed*). llama.cpp IQ2_XXS reads 391 GB/s whole-token on the same card (*computed*
from 312.85 tok/s on 1.25 GB, [m4-iq2-cuda](mesures/m4-iq2-cuda-2026-08-30.txt)). 300 asks our Leech decode to beat
llama.cpp's E8-grid decode on this card by 6 to 8 %, whole-token.

## 2. The v6 token, stage by stage

Phase stamps of v6, µs a token, median over 62 tokens; the stamped token costs 34 µs more than the timed one
(*measured*, pers-4e-v6 raw `pers-4e.txt`). Rows are mean CTA work. Non-row is mean CTA work outside rows. Imbalance
is each phase's wall above its mean, plus the MLP's inner wait.

| stage | rows | non-row | imbalance | 8,192: rows / non-row / imbalance | lead |
|---|---|---|---|---|---|
| ln1 + rotation | | 318.9 | | 325.4 | K2 |
| q/k/v rows | 408.3 | | 70.9 | 420.1 / 71.6 | K4 |
| norm, RoPE, KV append | | 26.4 | 86.8 | 27.5 / 83.7 | K3b |
| attention | | 238.9 | 404.1 | 1,850.7 / 681.8 | K3a, K8 |
| o rows (int4) | 345.8 | | 127.9 | 320.2 / 122.1 | K11, K3b |
| ln2 + rotation | | 157.8 | | 165.7 | K2 |
| gate/up rows | 1,203.3 | | | 1,178.7 | K4 |
| inner wait | | | 192.9 | 199.0 | none (true dependency) |
| down input | | 254.6 | | 256.7 | K2 |
| down rows | 452.8 | | 82.9 (MLP wall) | 459.9 / 91.8 | K4, K11 |
| head | 359.5 | | 0.3 | 362.5 / 0.4 | K5 |
| **sum** | **2,770** | **997** | **967** | 2,741 / 2,626 / 1,252 | |

The rows stream at 497 GB/s on mean work (*computed*), past the 490 that the
[v2 prereg](../proofs/preregistration-pers-4e-v2-2026-11-05.md) set as 300's need. What stands
between v6 and 300 is the 1.96 ms of non-row work and imbalance, plus rows that no measured arm has yet sped up.

With non-row work and imbalance both at zero, the token is 2.77 ms of rows: 361 tok/s,
338 at the door (*computed*, prop). Every lead short of that is a fraction of 1.96 ms.

## 3. Why the low bound keeps 41 %

| version | predicted | realised | share of the predicted ms | journal |
|---|---|---|---|---|
| v2 | 240 tok/s from 132.4 | 178.7 | 0.58 | [pers-4e-v2](mesures/pers-4e-v2-2026-11-05.txt) |
| v3 | 273 from 178.7 | 179.2 | 0.01 | [pers-4e-v3](mesures/pers-4e-v3-2026-11-05.txt) |
| v4 | ×1.03 from 179.0 | ×1.060 | 1.95 | [shaker-4e](mesures/shaker-4e-2026-11-05.txt) |
| v5 | ×1.06 from 189.8 | ×1.001 | 0.02 | shaker-4e |
| v6 | 243 from 190.2 | 212.6 | 0.48 | pers-4e-v6 |

Pooled, 2.85 of 6.92 predicted ms were realised: 41 % (*computed*). The low bound of each lead below keeps 41 % of its
arithmetic. Two versions of five realised nothing, so any single lead can give zero.

## 4. Leads

Gains are ms a token in the measurement form, high / low. High is the arithmetic; low keeps 41 %. "Identity" means
the served tokens stay bit for bit, checked at the door. Prices come from `docs/data/jobs.csv`.

| lead | what | 5 pos. high / low | 8,192 high / low | cost | gate | criterion moved |
|---|---|---|---|---|---|---|
| K2 | v7: down input to 100 µs, ln1 to ln2's 158 | 0.32 / 0.13 | 0.32 / 0.13 | 2-4 days, $0.24 | down input under 200 µs on the stamps | none, identity |
| K3a | attention phase wall to its floor (with K7 at 8,192) | 0.51 / 0.21 | 0.78 / 0.32 | 1-2 weeks, $0.42 + $0.4 bench | slowest over mean under 1.5 at 5 pos. | none if each head keeps its order |
| K3b | counters for the 218 grid barriers, the other walls | 0.37 / 0.15 | 0.37 / 0.15 | 1-2 weeks, $0.1 micro + $0.24 | counter wait under a barrier by 0.3 µs | none, identity |
| K5 | certified head by bit planes | 0.16 / 0.07 | 0.16 / 0.07 | 1-2 weeks, $0.2 + $0.24 | head under 250 µs, same argmax | +0.6 MB on card, +0.001 b/param |
| K4 | Tetra row loop | 0.32 / 0 | 0.32 / 0 | microbench $0.25; port 1 week, $0.24 | an arm at ×1.15 over v2 alone | none, identity |
| K11 | int4 rows at a streaming kernel's rate | 0.11 / 0.05 | 0.11 / 0.05 | microbench $0.15; port 3-5 days, $0.24 | int4 rows at ×1.10 alone | none if lanes keep their order |
| K10 | prefill by GEMM | 0.06 / 0.02 at the door | 0 | 4-7 days, $1.40-1.67 | T8 | quality (T8 band) |
| Q1 | the 84 int4 records re-encoded in Tetra | 0.15 / 0.06 | 0.15 / 0.06 | 1 week, $7-9 | full MMLU census | quality, b/param, GB, encoding |
| K8 | KV store int8 / int4 | 0 | 0.84 / 0.34, 1.26 / 0.52 | 2-4 weeks, $3-5 | its own prereg | quality, GB, class, noise |
| K9 | prompt-lookup speculation | workload | workload | 2-3 weeks, $1.00 | rulings R1, R8 | throughput becomes workload-dependent |

All gains and future costs *estimated*. The leads already written in plan-debit-noyau keep their text there (K2,
K5, K7, K8, K9, K10).
What changes here:

- **K3 splits in two.** K3a is the attention phase. At 5 positions its wall is 643 µs for 239 of mean work, 17.9 µs a
  layer over at most 261 cached positions. FlashInfer's decode reads 2,048 positions in 14.6 µs a layer on this card
  (*measured*, [fi-decode-anchor](mesures/fi-decode-anchor-2026-11-04.txt)). A floor of 130 µs is two grid barriers
  and a small split a layer (*estimated*). At 8,192 the same lever is K7: the wall reads the cache at η 0.55, the mean
  work at 0.76; η 0.80 is FlashInfer's level. K3b takes the other walls over mean (369 µs), where counters in the
  style of Mirage MPK replace grid barriers at 0.79 to 1.13 µs each (*measured*,
  [pers-l1](mesures/pers-l1-2026-11-04.txt)).
  The MLP's inner wait (193 µs) stays: the down input needs the whole gate/up vector.
- **K4 has no measured basis on the loop 4E runs.** 4E v4 and v6 run v2's loop (row form 2, `llvq_pers.cuh`). With
  every word served from L1 (arm `l1w`) the loop reads 2.516 ms against v2's 2.499
  ([pers-rowloop](mesures/pers-rowloop-2026-11-05.txt)): the word path costs v2 nothing measurable. The ×1.214 the
  plan cites is over `v2u`. The one suspect still standing is the warp balance, 0.63 at d_in 4,096 and 9,728 against
  0.81 at 2,560 (same journal). High 0.32 is a ×1.21 on the Tetra rows' share, 1.81 ms (*computed* from the solo
  rates); low is 0. No 4E variant is built before the microbench passes, as
  [pers-4e-rowloop-host](mesures/pers-4e-rowloop-host-2026-11-05.txt) requires.
- **K11 is new.** The int4 rows stream about 587 GB/s (*measured*, pers-4e-v3). FlashInfer streams 700 GB/s on the
  same card, and llama.cpp's f16 decode 682 whole-token (*computed*, m4-iq2-cuda). At 700, the 0.409 GB of int4
  records save 0.11 ms (*computed*). `tv_q4_hh` has never been timed alone; the microbench does that first.
- **Q1 is listed to be rejected.** Tetra streams slower than int4, so halving those bytes saves 0.15 ms at most
  (*computed*: 0.697 ms at 587 GB/s against 0.546 at 379). It undoes part of the step that took the 4B from 61.11 to
  63.37 MMLU and moves b/param from 2.73 to about 2.33 (*computed*, −0.20 GB). Not proposed.
- **R0, the door residue.** 0.271 ms a token between the door's decode and the measurement form has no cause. The
  phase stamps printed over the door's 256 tokens would name it. It rides on any door job, at $0 of extra card.

Closed leads stay closed (`ETAT.md` §6). v5 bought 160 µs of mean gate/up work and lost it in the down rows and the
wait (shaker-4e); a gate/up-only form would be a reopening, the operator's call.

## 5. Elite kernels, read against our measurements

Three techniques give the route a lead (K3a, K3b, K11); the others bound it. The table holds only what
[`plan-debit-noyau.md`](plan-debit-noyau.md) §3 does not already cover.

| technique | what our measurement says | what it gives the route |
|---|---|---|
| Hazy "No Bubbles": weights prefetched during barrier waits | words resident in L1 or L2 do not speed v2's Tetra loop (`l1w` ×0.99); the one-launch pass with an L2 prefetch read ×0.958 of the plain pass (pers-4e-v2) | nothing for Tetra rows; int4 rows and head unknown, measured inside K11 |
| Mirage MPK: event counters instead of grid barriers | 218 barriers at 0.79-1.13 µs, 0.17-0.25 ms a token; walls over mean 369 µs outside attention | K3b |
| FlashInfer and FlashDecoding at short context: few splits, merged once | our attention phase costs more a layer at ≤ 261 positions than FlashInfer at 2,048 | K3a, the largest single exact lead |
| Marlin: 128-bit loads from an offline layout, multi-stage `cp.async` | our int4 rows 587 GB/s, the AWQ kernel in our bench 584 ([f2-p3-qtip-banc](mesures/f2-p3-qtip-banc-2026-08-21.txt)), a stream on this card 700 | K11; Tetra side folded into K4's arms |
| llama.cpp CUDA decode, IQ2_XXS | 391 GB/s whole-token on our card, through graphs and fused small ops | the whole-token bar 300 must clear by 6-8 % |
| QTIP, computed trellis decode | 405 GB/s through 252 launches in our bench, where Tetra reads 259 through the same launches ([pers-l1](mesures/pers-l1-2026-11-04.txt)) | a computed decode can stream faster than ours; K4's ceiling is unknown, not capped |
| Kernel-side speculation, greedy verify | the second row costs 0.378 of a row pass, rows bit for bit (shaker-4e); attention and head unchecked | K9; the house prompt loops, so the door cannot read it |

## 6. Sequences, gates, cost

**Low bound: no sequence passes 300.** Every exact lead at its low bound reaches 229 / 225 tok/s at the door, and
adding K10 and Q1 gives 233 / 229 (*estimated*). Passing 300 on the low bound would need 3.86 ms of arithmetic,
more than the 1.96 ms of non-row work and imbalance that exist.

**High bound at the bit: five or six leads.** K3a, K3b, K2, K4 and K5 together give 1.68 ms: 310 / 296 at the door.
Adding K11 gives 1.79 ms: 322 / 306. Without K4 the other five give 290 / 279. No criterion moves.

**High bound touching quality.** K10 and Q1 add 0.21 ms: 347 / 327. They shorten no sequence. At a short prompt,
quality buys little, and 300 at the bit depends on the same row lever either way.

| step | leads | card | running total | gate, and what follows from it |
|---|---|---|---|---|
| 0 | R0 on `serve-4e-v6`'s door; host replays for K3a's split and K3b's counters; preregs of step 1 | $0 | $0 | the real v6 door fixes prop or add |
| 1 | one job: row loop arms (K4), int4 rows alone (K11), counter microbench (K3b) | about $0.5 | $0.5 | **if no K4 arm reaches ×1.15, 300 is out at the bit; the route continues toward 279-290 at most** |
| 2 | K2, then K3a, one 4E version | about $1.1 | $1.6 | stamps: down input under 200 µs, attention slowest over mean under 1.5; door identity |
| 3 | K3b, K5 | about $0.7 | $2.3 | identity at the door, same argmax |
| 4 | K4 and K11 ports, if step 1 passed | about $0.5 | $2.8 | identity; a door reading, the 50 GSM8K pilot |
| 5 | doors after steps 2 to 4, and a full GSM8K census under the final 4E | about $3.3 | $6.1 | the census is an open decision of `ETAT.md` §5 |
| 6 | K8, after the kernel (operator's order of 2026-11-05) | $3-5 | $9.1-11.1 | its prereg: paired ppl, light MMLU, a task past 1,280 |
| 7 | 8B and 14B (§8) | about $7 | $16.1-18.1 | T8 per size first |

Days, exact route: 26 to 46 working days, 5 to 9 weeks, before K8 and the larger sizes (*estimated*, the sum of the
table's ranges). The project has spent $273.30 (*computed*, `ETAT.md` §5). No spend cap is in force; one is owed
before the job of step 1.

### 6b. What is coded (2026-11-07)

The operator asked on 2026-11-07 for the package toward 300 without waiting for the v6 door. Steps 1 and 2 are coded
and identical to the served chain on the host. Nothing has run on the card.

**Step 1, run on the operator's go of 2026-11-07.** Job 6aef03fa, 8 min 24 s, $0.25 (*computed*), prereg stamped
before ([step1-micro](mesures/step1-micro-2026-11-07.txt)). Every arm that owes the bits kept them. *Measured*:
- K4 is dead. nc ×0.810, nc256 ×0.845, cpa ×0.960, tile ×1.002, bal ×0.934 over v2, all under ×1.10. The K4 port of
  step 4 is dropped.
- K11 is a lever. q4t, the int4 rows in 4-step tiles, reads ×1.338 over 4E's int4 row alone, 548 against 410 GB/s.
  It licenses the int4 port of step 4, up to 0.18 ms a token if the ratio holds in situ (*estimated*).
- K3b's counter form is not licensed. ctr8 waits 0.274 µs less than the barrier's 0.771, under the 0.3 bar; tight
  and rel gain 0 and 0.09. K3b's high bound is to be re-estimated from the walls over mean alone. Splitting arrival
  from wait hides 0.415 µs of the hop.

**Step 2, two new row forms beside v6.** Both are refused by a served config (`served.rs`) until measured. `fusedrun
LLVQ_PERS_4E` now times v4, v6, v7 and v8 in one process (`pers::P4E_BENCH_FORMS`).

- **v7, row form 5, K2.** After each grid barrier v6 still reads from L2, on lines all 142 CTAs ask for at once: the
  fields of the layer's argument struct, then the CTA's row slices through a second dependent load, then the sign
  words that the norm reads again after its scale. ln1 pays more of these than ln2, which fits ln1's 318.9 µs against
  ln2's 157.8 (pers-4e-v6). v7 copies these constants into shared memory between the CTA's arrival at the previous
  barrier and its release (`pers_grid_wait_gap`, `llvq_pers.cuh` §7c): the launch's arguments once, each layer's
  struct, slices and ln1 inputs during the layer's last barrier, ln2's during the o barrier, the down input's signs
  and `small` during the inner wait. It also widens the o phase's input from 16-byte vectors (8 serial loads in v6)
  and reads `small` as float4 in the mix. Shared memory: 65,344 B at the 4B (*computed*). Host: identity on every
  served rotation width, the -0 column, the q/k/v and o phases and the token loop (four geometries, past 256, three
  consecutive launches, stamps); 16 mutants killed.
- **v8, row form 6, K3a.** At 5 positions a row holds at most 69 positions, so attention runs 8 jobs of 4 steps, and
  only the next step's loads are in flight while a step runs. v8 stages the steps by `cp.async` into a ring of three
  stages, so three steps are in flight. Tiles, units and folds are unchanged, so the bits are too. This is the latency
  half of K3a. The other half, a job spread over more CTAs, is not in v8. Shared memory: 85,952 B at the 4B, 96,064 at
  the 14B, under the 101,376 opt-in (*computed*). Host: attention equal to `attn_ref` on nine row lengths at three
  geometries, the token loop; 7 mutants killed on the ring's indexing and reader barrier. The host's `cp.async` copies
  at once, so the waits are only held by the card's identity gate.

**Step 2, run on the operator's go of 2026-11-07.** Job 6aef3f69, 7 min, $0.22 (*computed*), prereg stamped before
([step2-4e](mesures/step2-4e-2026-11-07.txt)). *Measured*:
- v8 keeps the eager tokens at both contexts: 227.9 tok/s against v6's 211.8 at 5 positions (×1.076 round by round),
  155.6 against 151.9 at 8,192 (×1.025).
- v7 did not install: at 512 threads its module spills or fails its occupancy under the card's ptxas. K2's gate is
  read on v8, which holds all of v7's staging (departure E1 beside the prereg).
- K2 holds. Down input 254.8 to 150.2 µs, ln1 319.9 to 227.0, ln2 158.3 to 135.5.
- K3a fails its gate. Attention's slowest over mean reads 4.40. The ring halved the mean work (237 to 125 µs) and took
  79 µs off the wall; the job spread is what remains. At 8,192 the ring costs attention 141 µs.
- v8's rows run 175 µs slower at 5 positions (gate/up +94, down +52, q/k/v +28); the cause is not established.
- By the prereg's rule v8 stays a measurement. Step 3 is not chained.

## 7. At 8,192 positions

At 8,192 positions v6 takes 6.58 ms, where 300 asks 3.33 ms. The six exact leads at their high bound reach 221
tok/s; low, 171. K8 adds the cache: int8 gives 272 high and 181 low; int4 gives 307 high and 187 low (*estimated*,
1.208 GB of f16 cache read at η 0.80 after K3a). With Q1, 322 high. 300 at 8,192 therefore needs K8 in int4 at its
full arithmetic. K8 changes stored values and moves the 8k cache from 1.21 GB to about 0.34 GB. The product
triplet's b_max already counts a q8 cache at 8k that the engine does not store, so K8 touches the class too. It moves
quality, GB, class, and the noise floor it is read against.

## 8. Propagation to the 8B and the 14B

At a short prompt, 4E and the route take the 8B from 105.8 to 133-221 tok/s and the 14B from 62.5 to 73-104 at the
door (*estimated*, low / high). Each size owes a T8 of the kernel form and a light MMLU first.

| size | today, door | 4E v6 shape-general | after the route, low / high | weight floor | 8,192: today / kernel form / 4E / route |
|---|---|---|---|---|---|
| 8B | 105.8 | 122 to 144 | 133 / 221 | 359 | 67.1 / 86 / 99-117 / 108-171 |
| 14B | 62.5 | 68 to 76 | 73 / 104 | 188 | 45.3 / 55 / 60-67 / 64-90 |

Today's numbers *measured* ([served-grouped-8b](mesures/served-grouped-8b-2026-11-04.txt),
[served-grouped-14b](mesures/served-grouped-14b-2026-11-04.txt), [gqa-14b](mesures/gqa-14b-2026-11-04.txt)); the rest
*estimated*. 4E v6 saves 2.48 ms at the 4B's door over its grouped door, 133.3 tok/s
([served-grouped](mesures/served-grouped-2026-11-02.txt)), against the 199.2 of §1. The table moves 50 to 100 % of
it, scaled by layers (36, 40). The route's leads scale by width: rotations by d, the head by d, K4 and K11 by each size's Tetra and
int4 bytes. Floors take the bytes a token reads, the untied embedding left out (*computed*, about 2.41 and 4.60 GB).
At 8,192 the kernel form saves what it saved the 4B: 3.34 ms, from the grouped 77.5 to the eager kernel form's
104.6. The cache is the same size at 8B and 11 % larger at 14B.

What each size owes before it serves:

- **T8 of the kernel form from position 0.** 4E runs attention through `attn_dec.cu`, which neither size serves. The
  4B's reading was 32 windows of 1,280 and 12 of 4,096 at $0.81
  ([attn-t8-below](mesures/attn-t8-below-4b-2026-11-05.txt)).
  About $1.5 at 8B and $2.3 at 14B (*estimated*, scaled by decode time). The 14B runs G = 5, never read on the card
  in this kernel.
- **A light MMLU, 285 questions, under the new config**, with the door and the sweep in the same job: $1.25 at 8B and
  $1.81 at 14B, the price of the grouped doors.
- **4E identity at the door**, as on the 4B. The exact leads of the route owe only this.
- **Code.** 4E reads the 4B's placement: v and o int4. The 8B seals only part of down in int4, the 14B o and part of
  down. The 14B's head is untied. Its down input stages 17,408 f32 values: v6's 62,080 B of shared memory becomes
  at least 92,800 B, under sm_89's 101,376 B opt-in (*computed*). 1 to 3 weeks (*estimated*, never costed against code).
- K8, if served, needs its own T8, light MMLU and long task per size.

## 9. What this route does not establish

- Any gain above: the parts come from one job's stamps, and the rows' model is uncalibrated.
- The cause of the door residue of 0.271 ms, which decides between prop and add.
- Whether the warp balance binds the Tetra loop: no arm has tested it alone.
- Any card but the L40S, quality under 4E beyond a 50-problem pilot, and 4E above the 4B's shape.
