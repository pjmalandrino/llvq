# Preregistration. 4E v6: ln1, ln2 and the MLP's down input in one round trip a thread, against v4

**NOT STAMPED. NOT LAUNCHED.** Written on 2026-11-05 on branch `claude/serene-bohr-1edf69`, once the code it describes
existed. To be timestamped before the first measurement, then never edited; a departure goes to
`preregistration-pers-4e-v6-2026-11-05-ECARTS.md`.

The launch is the operator's, through the integration session. Cost: about **$0.30** on l40sx1 (*estimated*, §8).
Timeout 20 min: **$0.60** at worst (*computed* at $1.80/h). Spent before it: see `docs/data/jobs.csv` at launch.

## 1. Why this run exists

The shaker (job 6aec9a1d) stamped 4E v4's inner parts at a 5-token prompt. On the mean CTA, us a token, *measured*:
- MLP down input 799.0;
- q/k/v ln1+rotation 406.6;
- MLP ln2+rotation 261.6.

That is 1,467 us of a 5,269 us token, on work every CTA repeats on the whole vector.

`docs/mesures/shaker-4e-rule-c-host-2026-11-05.txt` reads these three parts as serial round trips to memory:
- in the SASS, every loop of these parts loads one element and uses it before the next load issues;
- that makes 15 round trips a thread a ln, 38 a Tetra down input and 19 an int4 one;
- the measured parts put each round trip at 0.48 to 0.75 us;
- in each round trip, the grid's 142 CTAs read the same lines at once.

v6 does the same work in one or two round trips. This run measures what that buys.

## 2. What the code holds at the commit that carries this file

`llvq_pers.cuh`, `PERS_4E_ROWS` 4: 4E v6. Its rows are form 2's (v4, the served form since 2026-11-05). Its prologues
are section 7b's. These are used in 4E only, and when the input is 16-byte aligned and its length a multiple of 8:
- **Loads.** Each binary16 vector comes as 16-byte loads, at most 3 a thread a batch: one batch at 512 threads for
  the 4B's 9,728. The sign words and the k rows of the rotation's `small` come in the same batch. Every load is issued
  before the first use.
- **The mix.** `small` is staged in a 4 KB shared scratch after the row vector (`PERS_4E_SCRATCH`). The mix reads it
  from there. At k = 5 and k = 19 (the 4B's two odd factors) it is unrolled, each column chain kept as written: from
  +0, t ascending, then + 0.
- **ln1 and ln2.**
  - x and alpha are staged widened in shared memory.
  - candle's RMS tree is replayed from there (`exact_rms_scale_by`, the same body as `exact_rms_scale`, which now
    calls it).
  - The apply is the same two roundings on the same values.
- **The int4 down input** is the same vector load with no signs.
- **Install.** `persist4e_cuda::install_rows` refuses form 4 when one of its inputs is not 16-byte aligned (act, x_a,
  x_b, every layer's ln1 and ln2 weights; `pers::p4e_v6_inputs`): the kernel would take the scalar path and time as
  v4. It prints `4E v6: the 75 prologue inputs are 16-byte aligned` otherwise.
- **Driver.** `pers::P4E_BENCH_FORMS` is `[v4, v6]`. `fusedrun LLVQ_PERS_4E` installs those two modules at 512 threads.
  It prints `4E v6 / 4E v4, round by round` beside each form's rate.

Host checks at $0 (`llvq-cuda/tests/pers_matches_served.rs`, `cargo test` under nice 10):
- `the_4e_v6_prologues_are_the_served_chain`:
  - the rotation at the six served widths, at 64 and 512 threads, equal to `rot_apply`;
  - the -0 column at k = 3, 5, 19;
  - the token loop at four geometries (512 threads among them), at positions 270-271, and through the stamp symbol,
    equal to the served chain.
- `the_4e_v6_mutants_are_killed`: 14 mutants caught.
  - 11 in section 7b: half order, sign word, ln sign shift, `small` store, `small` loop, mix order, the mix's + 0,
    apply order, alpha vector, ln1's natural copy, RMS loader.
  - 3 at the segment's call sites: ln2's signs, the down's k, the int4 widening's signs. These prove the token loop
    runs the vector path.
- The full llvq-cuda suite; `cargo clippy --all-targets` on the Mac and on the Linux half.
- The devel ptxas 12.4 registers of `pers_4e_tokens_f1_t512` and its stamp symbol under forms 2 and 4
  (`docs/mesures/pers-4e-v6-host-2026-11-05.txt`).

## 3. What runs

One image built from the commit that carries this file. One job, l40sx1, `ops/jobs/pers-4e-v6.sh`. The sealed 4B
(`886391a8…`) is checked by size, then sha256.

1. **Oracle** on Qwen3-0.6B (rule 10).
2. `LLVQ_FUSED_LAYOUT=tetra48 LLVQ_EMBED=q4 LLVQ_ROT_SHARE=1 LLVQ_FUSE=0 LLVQ_PERS_4E=64 fusedrun <sealed> 64`.
   - Two runtimes at 512 threads, v4 (form 2) and v6 (form 4), each its own module.
   - At 5 tokens (measurement form, kernel on every row) and at 8,192 (the served config, inside the T8).
   - Per context: each form's tokens against the eager path's, 2 + 5 interleaved rounds, then one stamped launch a
     form.

## 4. Gates

Each voids the run:

1. The image builds; `oracle` prints MATCH.
2. The sealed file at its bytes and sha256.
3. Event tracking off; a RoPE form equal to candle's.
4. **v4 and v6** tokens equal to the eager path's at both contexts.
   - `fusedrun` stops at the first form that diverges, so a v6 divergence voids the run.
   - Its first divergence goes back to the host.
5. No `LLVQ_*` variable in the job environment.

## 5. Outcomes reported, never gated

- Each form's registers and local bytes under the driver's ptxas.
- Per context:
  - each form's tok/s, median and range, and its ratio to the eager path round by round;
  - v6 over v4, round by round.
- Each form's phase table and its 7 inner parts (CTA work, slowest and mean).

## 6. Signed predictions

*Estimated*, ops/pers_4e_insitu.py `rule_c`, from the shaker's measured parts and the SASS counts; nothing fitted to a
number of this run.

| quantity | point | interval |
|---|---|---|
| v4 and v6 tokens against the eager path, both contexts | **identical** | |
| v6 loads without local memory (driver ptxas) | **yes** | |
| v4 at 5 tokens, tok/s (the shaker's reading again) | **189.8** | [186, 194] |
| **MLP down input, v6, us a token (mean CTA)** | **91** | [50, 200] |
| q/k/v ln1+rotation, v6 | **90** | [60, 300] |
| MLP ln2+rotation, v6 | **71** | [40, 200] |
| v6 at 5 tokens, tok/s | **243** | [192, 247] |
| v6 at 8,192, tok/s | **167** | [141, 169] |
| v6 over v4, 5 tokens, round by round | **x1.28** | [1.01, 1.30] |
| v6 over v4, 8,192, round by round | **x1.19** | [1.01, 1.21] |
| MLP down rows, v6 against v4 (int4 DRAM fill exposed) | **+0 to +60 us** | |

## 7. Decision rules

- **Gate of the lever (the operator's).** v6's MLP down input, mean CTA work at 5 tokens:
  - at most 200 us a token: the lever holds;
  - above 200: it fails, and the round trips were not the cost they were read as;
  - under 100: the round trips were the cost, as rule C read them.
- **A, identity.** v6's tokens equal the eager path's at both contexts, or the run is void (§4).
- **B, serving.** v6 over v4, round by round.
  - With its whole range above x1.00 and the gate held, v6 is proposed for serving: the served config's rows would
    name it.
  - The served engine changes only on the operator's yes. That yes needs this reading and a door reading of v6
    under the served config.
  - Otherwise v6 stays a measurement.
- **C, the parts.** Each of ln1+rotation, ln2+rotation and the down input is read against its §6 interval.
  - A part that falls but stays above its interval names what the round trips do not cover: the mix, the WHT, the
    barriers.
  - The down rows rising more than 60 us says the int4 down's DRAM fill is no longer hidden.
- **G.** A miss is reported as a miss, with the term of §6 it refutes. No threshold moves after the clock.

## 8. Cost

*Estimated* on l40sx1 at $1.80/h:

| step | time |
|---|---|
| container and image | 3 min |
| oracle | 1 to 2 min |
| fusedrun: two 4E modules built, 2 contexts x 7 rounds x 3 decodes, 4 stamped launches (the shaker's part A ran three forms) | 5 min |

About 10 min, **$0.30**. Timeout 20 min: **$0.60** at worst.

## 9. What this run does not establish

- Serving v6: the served config's 4E stays v4 (row form 2); a form is served only on the operator's yes.
- The final norm before the head keeps the scalar path (one per token); it is not in v6.
- The 8B and the 14B, whose rotations have other k (the generic mix from shared memory); any card but the L40S.
