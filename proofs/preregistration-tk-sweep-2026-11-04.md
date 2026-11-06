# Preregistration P2. t(k) of the served 4B, k = 1 to 8, and what one verification row costs (2026-11-04)

**Written on 2026-11-04, to be committed and TIMESTAMPED (`ots stamp`) BEFORE the run.** Operator go given on
2026-11-04 in chat for the first wave of the validation roadmap, under a **$10** HF cap for the wave. P2 is in that
wave. Cost: about **$0.11** on l40sx1 (*estimated*, §8). Timeout 15 min, so **$0.45** at worst (*computed*). Spent
before it: $248.84 over 237 rows of `docs/data/jobs.csv` (*measured*), plus $3.19 of five billed jobs of 2026-09-19
to 2026-09-21 missing from the registry (*computed* on their measured durations), so **$252.03**. At worst $252.48
after it. No engine code changes: an existing measurement mode on an existing image, and `ops/jobs/tk-sweep.sh`.

The speculative lead was closed for the 4B on 2026-08-05, to come back at 70B
(`docs/archive/pistes-battre-q4.md:142`). The go opens this measurement only, and the verdict stands. No build
follows from this run without a further go. Every route below that leads to code also waits on the hold "Batch above
1, and prefill" of `docs/ROADMAP.md` §4. This run serves nothing, so the hold does not bind it.

## 1. Question

What does one extra row cost in a k-row forward of the served 4B on the L40S, for k = 1 to 8?

Exact speculative decoding drafts k tokens and verifies them in one forward of k rows. Its gain rests on β, the cost
of one extra row in decode steps, and β was never measured on this engine. It decides whether B1 is worth building
alone. B1 is the verify itself: one k-row forward with a head and an argmax on every row, 1 to 2 weeks of code
(*estimated*). B2 is the cut of the per-row costs inside that forward. G0 is the acceptance pilot on the Mac, which
measures c, the share of rounds with a draft, and τ_d, the tokens accepted a drafted round.

## 2. What the run can measure, and what it cannot

`fusedrun` with `LLVQ_PREFILL_TOKENS=k` takes one length a process. With `LLVQ_CONFIG` it loads the served object and
applies the config's store and attention form. It times 6 passes and discards the first. Each pass starts from fresh
caches at position 0: one `hidden_cached` over k rows, the head on the last row only, a device synchronize. It prints
the median and range at 0.1 ms, and the median per token at 0.001 ms. A guard then compares k rows in one call with k
one-row calls, and exits non-zero if the two argmaxes differ, after the timing lines.

`LLVQ_CONTEXTS=5,2048` times the one-token decode step after a 5-token and a 2,048-token prompt, 256 new tokens, 1
discarded round and 5 timed, with the time to the first token. It runs alone: `LLVQ_PREFILL_TOKENS` beside it is
refused.

**No mode times k rows at an offset.** So t(k) at 2,048 positions is not measured by this run, and the rule cannot
read β at 2,048. It reads Δ4 = (t_pf(4) − t_pf(1))/3 instead, the ms of one extra row at position 0. β at 2,048 is
derived from Δ4, *estimated*, and the thresholds of §6 are that derivation inverted once, before the run.

**The path timed is the batched prefill, not an exact verify.** Rows go to `group_forward` in packets of 4
(`prefill 4x128`), the head runs on the last row only, and one causal mask is built. An exact verify adds per row a
head projection, 0.32 ms (*computed*, 218.8 MB of table at about 690 GB/s). It adds 216 attention launches, 0.76 ms
at 3.5 µs each (*estimated*), and a KV read of 0.668 µs a position. That slope is (12.90 − 7.43)/8,187 ms
(*computed* on the grouped sweep, *measured*, `docs/mesures/served-grouped-2026-11-02.txt`). It skips the k-row mask,
0.23 ms (*computed* from 119.0 against 122.4 tok/s, `docs/mesures/trims-contexts-2026-10-30.txt`, and taken as the
k-row mask, *estimated*). So an exact t(4) at short context should read about 2.3-2.5 decode steps where the batched
path reads about 2.1 (*estimated*). The thresholds of §6 carry that gap as constants.

## 3. Setup

Image: Space `Pier-Jean/llvq-runner-cuda` at sha `f77f803e`, built from `f808856`, the served-grouped image. From
`f808856` to `06e1602` the code differs in `fusedrun.rs` only, in a comment and in one error string of the
`LLVQ_CONTEXTS` block that fires under `LLVQ_KV_PREALLOC` alone. Neither mode used here moves. Another image needs
`proofs/preregistration-tk-sweep-2026-11-04-ECARTS.md` before the run, and the script refuses without it.

Objects: `sealed-4b-2026-09-23/qwen3-4b-sealed.bin`, 1,418,224,685 B, sha256 `886391a8…`, on the bucket mount.
`configs/qwen3-4b-tetra-e4-grouped.json`, sha256 `2455fe85…`, written into the job from the launch command.

```
oracle Qwen/Qwen3-0.6B 64 cuda
for k in 1 2 3 4 5 6 7 8, then 8 7 6 5 4 3 2 1:
  rc=0; env LLVQ_CONFIG=<grouped> LLVQ_PREFILL_TOKENS=$k fusedrun <sealed> > pf-$k-<order>.txt 2>&1 || rc=$?
env LLVQ_CONFIG=<grouped> LLVQ_CONTEXTS=5,2048 fusedrun <sealed> 256
```

The reverse order exposes drift across the job. Output in `/out/tk-sweep-4b-2026-11-04/`.

## 4. Controls, each one voids the run

1. The Space is at `f77f803e`, and `oracle` prints MATCH.
2. The sealed file at its bytes and sha256 on the mount; the config at sha256 `2455fe85`.
3. All 17 processes print `tile 64 (served: measured optimum for sm_89)`, `prefill 4x128 (served pair)`,
   `int4: tv_q4_hh equals tv_q4_h bit for bit`, the int4 g64 embedding on 1 table, and a `served config:` line with
   `attention grouped on grow(256)`. Each prefill process prints `prefill: k tokens, layout tetra48, attention grouped
   on grow(256)`. `LLVQ_PREFILL` moves the packet pair and nothing refuses it, so the printed pair is checked.
4. No `LLVQ_*` variable in the job environment.

A guard refusal or a crash after those lines is no gate: its exit code is kept and the sweep goes on.

## 5. Reported, never gated

- Per k and per order: median ms, range, ms a token, guard line, exit code. t_pf(k) is read as the per-token line
  times k.
- t_pf(k)/t_pf(1) within each order, as a quotient of medians from two processes with its envelope
  [min_k/max_1, max_k/min_1] (METHODE §2). Δ4 with the same envelope.
- t_dec(5), t_dec(2,048) and the first token at 2,048 from the last step; t_pf(1) beside t_dec(5).
- Derived, *estimated*: β_ex(p) = (Δ4 − 0.078 + 0.32 + 0.756 + 0.000668·p)/t_dec(p), in ms, with 0.078 the mask
  over 3 rows. At p = 5 and 2,048 on this job's t_dec. At 8,192 on t_dec = 12.90 of served-grouped, a cross-job reading, reported and not published.
- Derived, *estimated*: r_ex(k, p) = 1 + [t_pf(k) − t_pf(1) − 0.23 + (k − 1)(1.076 + 0.000668·p)]/t_dec(p), for
  k = 2, 3, 4. Never 1 + β(k − 1) at k = 2 or 3: a packet computes 4 rows whatever its row count.
- The staircase: t_pf(5) − t_pf(4) against t_pf(4) − t_pf(3), since k = 5 opens a second packet.

## 6. Decision rule

Primary quantity: Δ4 from the forward-order medians, in ms. A β band at 2,048 maps to Δ4 = β × 8.78 − 2.367 ms.
8.78 ms is t_dec(2,048) of served-grouped (*measured*). 2.367 is 0.32 + 0.756 + 1.368 − 0.078 (*computed* on the
estimated constants of §2). So β = 0.40 reads Δ4 = 1.15 ms, and β = 0.60 reads 2.90 ms. Launches at 2.2 or 4.8 µs
would move both by 0.28 ms (*computed*); that is reported, not used. The first row that applies decides.

| route | condition | consequence |
|---|---|---|
| No route | A timing line for k = 1 or 4 is missing in either order. Or the two orders' t_pf(1), or t_pf(4), differ by more than 5 %. Or the guard refuses at k = 4, where a wrong row offset would be faster and wrong at once | Every line reported. A rerun needs a deviation and a go |
| Split | Forward and reverse Δ4 fall in different bands below | The costlier band of the two applies, reported as split |
| R-a | Δ4 ≤ 1.15 ms | The rows path already costs what B2 plans. B1 alone may pay on extraction; G0 decides with r_ex(k, p) |
| R-b | 1.15 < Δ4 ≤ 2.90 ms | B1 alone is plumbing or close to it: ×1.14 on extraction decode at β = 0.40, ×1.04 at 0.60, best k (*computed* at c = 0.45, τ_d(4) = 2.6, both *estimated*). Proposal: B1 with B2, or wait |
| R-c | Δ4 > 2.90 ms | The rows path costs more than planned. B2's per-row estimate is redone from these numbers before any code |
| Otherwise | Any case the rows above do not name | Not settled, operator decision |

R-a, R-b and R-c cover every real Δ4. A negative Δ4 falls in R-a and is reported as suspect. A guard refusal at
k = 2 or 3 removes that k's r_ex(k, p), and is reported. At k = 5 to 8 it is reported only. Every route is a
proposal; the operator decides.

## 7. Signed predictions

| quantity | point | interval |
|---|---|---|
| t_pf(1), ms | 7.4 | [7.0, 8.0] |
| t_pf(2)/t_pf(1) | 1.75 | [1.45, 2.10] |
| t_pf(3)/t_pf(1) | 1.98 | [1.60, 2.40] |
| t_pf(4)/t_pf(1) | 2.15 | [1.8, 2.7] |
| t_pf(5)/t_pf(1) | 2.7 | [2.2, 3.3] |
| t_pf(8)/t_pf(1) | 3.8 | [3.0, 4.8] |
| Δ4, ms | 2.8 | [2.0, 4.2] |
| route | R-b or R-c | |
| t_pf(5) − t_pf(4) above t_pf(4) − t_pf(3) | yes | |
| t_pf(2) − t_pf(1) above 2 × (t_pf(4) − t_pf(3)) | yes | |
| guard argmax identical at every k | yes | |
| the two orders within 2 % at every k | yes | |
| t_dec(5), t_dec(2,048), ms | 7.43, 8.78 | [7.2, 7.7], [8.5, 9.1] |

All *estimated*. A packet of 2 rows already pays a full packet's Tetra work, 3 × 0.93 ms, and `regroup` copies
nothing at k = 1 only. That gives t_pf(4) ≈ 16.3 ms. The 203-token prefill took 780.7 ms for 51 packets (*measured*,
`docs/mesures/paper-table-2026-09-25-brut/paper-served-4b-2026-09-25-r2/prefill-203.txt`), so 15.7-16.1 ms a 4-row
pass, on the older expanded path. The known flaw: the per-row constants of §2 are not measured, and at 2,048 they
are 48 % of the derived row cost (*computed*).

## 8. Cost

*Estimated* on l40sx1 at $1.80/h, billed by the second. Served-grouped reached its first `fusedrun` 23 s after it
started, and loaded the 4B in 4.0-4.4 s (*measured*, `docs/mesures/served-grouped-2026-11-02-brut/job.log`). Here:
16 prefill processes of 4.7-6.5 s, 75-104 s. The context step: 6 rounds of 8.2 s of prefill and 2.2 s of decode at
2,048, and 1.9 s at 5, about 80 s. In all 190-215 s, **$0.10-0.11**. Timeout 15 min, **$0.45** at worst
(*computed*).

## 9. What this run does not establish

The exact verify's own t(k), so β_ex stays *estimated*. Any k-row cost at an offset. Row-invariance: the guard
compares a batched path with one-row calls, to f16 only. The acceptance side, which G0 measures. The 8B, the 14B,
another card, Metal.

Any departure from this file goes in `proofs/preregistration-tk-sweep-2026-11-04-ECARTS.md`, beside it.
