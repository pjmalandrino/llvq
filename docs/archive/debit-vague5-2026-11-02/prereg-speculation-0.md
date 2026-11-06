DRAFT, not stamped, not committed.

# Preregistration. L1a: what a k-row forward of the served 4B costs, k = 1 to 8, and what an exact verify row would cost

**Written on 2026-11-02 as a draft. To be committed and TIMESTAMPED (`ots stamp`) BEFORE the run.** No operator go
yet. No engine code: an existing measurement mode on an existing image, plus one job script to write. Cost: about
**$0.08**, **$0.12** with margin, on l40sx1 (*estimated*, §8). Timeout 20 min, so **$0.60** at worst (*computed*).
Cap asked: the wave cap of the validation roadmap (its decision 1); this job alone stops at its timeout. Spent
before it: $248.84 over 237 lines of `docs/data/jobs.csv` (*measured*). The registry misses five billed jobs of
2026-09-19 to 2026-09-21, $3.19 (*computed* on their measured durations), so $252.03 in all.

Ruling R1 comes first (`speculation.md` §7). The speculative lead was closed on 2026-08-05, to come back "at 70B"
(`docs/archive/pistes-facteurs-cles-2026-08-05.md:75`, `docs/archive/pistes-battre-q4.md:142`). It has no row in
`docs/ETAT.md` §6, so only the operator's ruling can open this run. Every route of §7 that leads to B1 or B2 also
waits on R8, the reading of the ROADMAP §4 hold "Batch above 1, and prefill" (decision 9 of the wave-2 report). This
run serves nothing and changes no code, so the hold does not bind the run itself.

## 1. Why this run exists

Every speculative estimate on today's engine rests on one input that was never measured: what an extra row costs in
a k-row forward of the served object. The first version of the speculation dive took t(4)/t(1) ≈ 2.1 from the
prefill path. The exact verify adds work the prefill path does not do: one attention call per row, one head launch
per row, and a KV read per row. So its β was re-estimated at 0.58 at 2,048 positions (`speculation.md` §3.3). That
estimate decides whether build step B1 is worth 1-2 weeks on its own. This run measures the rows path's part of it
at no code cost.

`fusedrun` with `LLVQ_PREFILL_TOKENS=k` is one of the two measurement modes the served door accepts
(`fusedrun.rs:669-686`, `:947-1118`). With `LLVQ_CONFIG` it loads the served object and applies the config's store
and attention form. It then times 5 passes of k tokens, after a discarded one, each pass starting from fresh caches
at position 0: one `hidden_cached` over k rows, then the head on the last row, then a device synchronize
(`:1001-1014`). It prints the median and range at 0.1 ms, and the median per token at 0.001 ms. It then compares k
tokens in one call with k one-token calls (max |Δlogit|, scale, both argmaxes), and refuses if the argmaxes differ.

No existing mode times k rows at an offset. The other mode, `LLVQ_CONTEXTS`, times a prompt and then one-token
decode, and refuses `LLVQ_PREFILL_TOKENS` beside it (`:828-834`). So this run measures k rows from position 0 only.
β at 2,048 and 8,192 is derived in §5 with three per-row terms this run does not measure. At 2,048 they are 48 % of
the derived row cost (*computed* on the central values of §5).

The full-packet cost is already measured. The 203-token prefill of the sealed 4B took 780.7 ms [779.1, 781.4]: 50
full packets and one 3-row packet, which does a full packet's work (*measured*,
`docs/mesures/paper-table-2026-09-25-brut/paper-served-4b-2026-09-25-r2/prefill-203.txt`). That is about 15.3 ms a
4-row packet (*computed*), on the expanded form and before the trims of 2026-10-30. It puts t_pf(4)/t_pf(1) near
2.1-2.2 and β_ex(2,048) near 0.55-0.60 before this run (*estimated*). This run adds the same quantity on today's
grouped path, and the shape of t(k) at k = 2, 3 and 5 to 8, which D3 reads one k at a time (§5).

**This run carries a measurement, not a gate** (METHODE §1). Its rule routes build work; it adopts nothing.

## 2. What the code holds

No change. Image: Space sha `f77f803e` (built from `f808856`, the served-grouped image), if the Space still holds it.
Between `f808856` and `06e1602` the code differs only in a comment and an error message of the `LLVQ_CONTEXTS`
block (`fusedrun.rs:824-827`, `:836-840`). Otherwise the image must come from a commit at which `git diff 06e1602 <commit> --
Cargo.toml Cargo.lock ops/Dockerfile.cuda llvq-core llvq-search llvq-quant llvq-artifact llvq-bench llvq-metal
llvq-cuda llvq-llm` is empty. The `LLVQ_PREFILL_TOKENS` block alone is not enough: it calls `group_forward`, the
rows kernel, `regroup` and the attention, which the audit-steps branch changes. Any other image needs a deviation
written beside this prereg before the run. The launch script checks `IMAGE_SHA`, as `ops/jobs/gqa-ab.sh:60-63` does.

Objects: `sealed-4b-2026-09-23/qwen3-4b-sealed.bin` (sha256 `886391a8…`) on the bucket mount;
`configs/qwen3-4b-tetra-e4-grouped.json` (sha256 `2455fe85…`), written into the job from the launch command and
checked by sha256, as served-grouped did.

## 3. What runs

One job, l40sx1, the script modelled on `ops/jobs/served-grouped.sh`:
1. `oracle Qwen/Qwen3-0.6B 64 cuda` (rule 10).
2. For k = 1, 2, 3, 4, 5, 6, 7, 8, in that order, one process each:
   `env LLVQ_CONFIG="$CG" LLVQ_PREFILL_TOKENS=$k fusedrun "$F"`, the form of `ops/jobs/sealed-4b.sh:87` and four
   other job scripts. Each output goes to its own file. A non-zero exit
   from the guard's refusal is recorded with its code, and the sweep continues: the refusal prints after the timing
   lines (`fusedrun.rs:1018-1045`, then `:1103-1116`). `ops/run.py bench` runs the script under `set -euo pipefail`
   (`run.py:1053-1055`), so a bare `| tee` pipeline would end the job at the first refusal. Each command is therefore
   written `rc=0; env … fusedrun "$F" > "$O/pf-$k-$order.txt" 2>&1 || rc=$?`, and `k`, the order and `rc` are echoed.
3. The same eight commands a second time in reverse order, k = 8 down to 1, to expose drift across the job.

## 4. Gates, each one voids the run

1. The image is the one named, and `oracle` prints MATCH.
2. The sealed file at its bytes and sha256 on the mount; the config at sha256 `2455fe85`.
3. Every one of the 16 runs prints `prefill: k tokens, layout tetra48, attention grouped on grow(256)` (the labels of
   `fusedrun.rs:992-997`), `tile 64 (served: measured optimum for sm_89)`, `prefill 4x128 (served pair)`, and
   `int4: tv_q4_hh equals tv_q4_h bit for bit`. The packet pair decides where t(k) steps. `LLVQ_PREFILL` changes it
   and is refused neither by the served door (`served.rs:111`) nor by `fusedrun.rs:677-679`, so the printed pair is
   checked as well as gate 4.
4. No `LLVQ_*` variable in the job environment. Only `LLVQ_CONFIG` and `LLVQ_PREFILL_TOKENS` appear, on each command
   line.

## 5. Outcomes reported, never gated

- Per k and per pass order: the median ms of a k-token pass with its range, and the ms per token.
- t_pf(k)/t_pf(1), formed within each pass order, then both orders side by side. Each k runs in its own process, so
  each ratio is a quotient of medians with its envelope [min_k/max_1, max_k/min_1] (METHODE §2).
- Per k: the guard line (max |Δlogit|, scale, argmax pair) and whether it refused.
- Derived, *estimated* (measured t_pf, *computed* and *estimated* constants): the exact-verify per-row cost and β_ex
  at 5, 2,048 and 8,192 positions.

  per_row_exact(p) = (t_pf(4) − t_pf(1) − 0.23 ms)/3 + 0.32 ms + 216 × 3.5 µs + 0.668 µs × p

  The terms: 0.23 ms is the causal mask a k-row pass builds and the exact path does not. It is *computed* from the
  trim of 2026-10-30 (119.0 against 122.4 tok/s, `trims-contexts-2026-10-30.txt`), measured on the expanded form
  at decode, and taken here as the k-row mask (*estimated*). 0.32 ms is one head launch, *computed*. 216 launches
  at 3.5 µs is the per-row attention, *estimated*, also printed at 2.2 and 4.8 µs. 0.668 µs per position is the
  grouped door's *computed* slope.

  β_ex(p) = per_row_exact(p)/t1(p), with t1 = 7.43, 8.78 and 12.90 ms (*measured*, the "ms a decode token" column
  of `served-grouped-2026-11-02-brut/sweep-grouped.txt`). That divides a number of this job by a number of another
  job: a cross-job reading, reported and not published (METHODE §2). The two grouped sweeps of 2026-11-01 and
  2026-11-02 agree within 0.6 % (*measured*, `jobs.csv` 235 and 238), which moves β_ex by under 0.005 (*computed*).
- Per-k exact ratios for D3, *estimated* on the same constants, for k = 2, 3 and 4 and p = 5, 2,048, 8,192:

  r_ex(k, p) = 1 + [t_pf(k) − t_pf(1) − 0.23 ms + (k − 1)(0.32 ms + 216 × 3.5 µs + 0.668 µs × p)]/t1(p)

  r_ex(4, p) = 1 + 3 β_ex(p). D3 takes r_ex(k, p) for k = 2 and 3 as printed, never 1 + β_ex(p)(k − 1). Within a
  packet t(k) is not linear in k (§6), so the linear form overstates S at k = 2 and 3.
- The packet staircase: t_pf(5) − t_pf(4) against t_pf(4) − t_pf(3). At k = 5 the fifth row runs the one-row path,
  a second pass over the weights. And t_pf(2) − t_pf(1) against t_pf(4) − t_pf(3): the fold of §6.

## 6. Signed predictions

| quantity | point | interval | label |
|---|---|---|---|
| t_pf(1), ms | 7.4 | [7.0, 8.0] | *estimated*: a decode step at position 0, fresh caches included |
| t_pf(2)/t_pf(1) | 1.75 | [1.45, 2.10] | *estimated*: the fold and the first `regroup` copies land at k = 2 |
| t_pf(3)/t_pf(1) | 1.98 | [1.60, 2.40] | *estimated* |
| t_pf(4)/t_pf(1) | 2.15 | [1.8, 2.7] | *estimated*: 2.12-2.18 from the 203-token prefill (§1), 2.21 itemized below |
| t_pf(5)/t_pf(1) | 2.7 | [2.2, 3.3] | *estimated*: one more weight pass |
| t_pf(8)/t_pf(1) | 3.8 | [3.0, 4.8] | *estimated* |
| t_pf(5) − t_pf(4) larger than t_pf(4) − t_pf(3) | yes | | *estimated* |
| t_pf(2) − t_pf(1) larger than 2 × (t_pf(4) − t_pf(3)) | yes | | *estimated*: the fold |
| β_ex at 2,048 (central launch cost) | 0.59 | [0.53, 0.64] | *estimated*: the 203-token prefill pins the measured term |
| route of §7 | R-b or R-c | | *estimated*: R-a needs t_pf(4) ≤ 10.8 ms at t_pf(1) = 7.4 (*computed*) |
| guard argmax identical at every k | yes | | *estimated*; ten earlier guards were (*measured*, `jobs.csv` 144-146) |
| forward and reverse pass orders within 2 % at every k | yes | | *estimated* |

Reasoning, *estimated*. Two costs land whole at k = 2 instead of spreading over the packet. The rows kernel stages
and computes all 4 rows of its packet whatever `n_rows` is (`tv_tetra48_h.cu:198-205`, `llvq_tetra48.cuh:191-204`).
A packet of 2 or 3 rows therefore costs a full packet's Tetra work, 3 × 0.93 ms above the one-row kernel.
`regroup` copies nothing at k = 1, because `Tensor::cat` of one tensor returns it (`model.rs:900-901`). At k ≥ 2 it
copies 252·k rows (`model.rs:1102-1106`), 0.875 ms per 252. The int4 fan-out adds 0.8 ms a row (84 records re-read,
plus their casts), and the mask 0.23 ms once a pass. That gives t_pf(2) ≈ 7.4 + 2.79 + 0.8 + 1.75 + 0.23 = 13.0 ms,
t_pf(3) ≈ 14.6 ms and t_pf(4) ≈ 16.3 ms, ratios 1.75, 1.98 and 2.21. The 203-token prefill puts t_pf(4) at about
15.7-16.1 ms on its older path (§1). The F1e sweep's fit on 606-token prompts, a + b/R with a = 3.06 and b = 3.18 ms
per token (*computed* in plan B), puts the per-row cost at about 3 ms with attention and the per-packet weight pass
at about 3.2 ms. Both agree within the intervals.

## 7. Decision rule

Primary quantity: β_ex(2,048) at the central launch cost, from the forward-order medians. The lines below are read
in order, and the first that applies decides.
- **No route.** A timing line for k = 1 or k = 4 is missing in either order. Or the two orders' medians of t_pf(1),
  or of t_pf(4), differ by more than 5 %. Or the guard refuses at k = 4: its own comment says a wrong row offset
  makes the pass "faster and wrong at once" (`fusedrun.rs:1047-1052`). Every line is reported.
- **R-a.** β_ex(2,048) ≤ 0.40: today's rows path already costs what B2 plans, and B1 alone may pay on extraction.
  G0's rule D3 decides, with the r_ex(k, p) of §5 and G0's c and τ_d.
- **R-b.** 0.40 < β_ex(2,048) ≤ 0.60: B1 alone is plumbing or close to it. At the central c and τ_d, prompt lookup
  gives ×1.14 on extraction decode at β = 0.40 and ×1.04 at 0.60, best k (*computed* on estimated inputs, the linear
  model of `speculation.md` §3.3). The fold of §6 lowers both at k = 2 and 3. Proposal before G0: build B1 with B2,
  or wait. Once G0 has run, D3 decides with r_ex(k, p).
- **R-c.** β_ex(2,048) > 0.60: the rows path costs more than planned. B2's per-row estimate of 1.31 ms for
  projections and head is revisited from these numbers before any build. It is the 2.93 ms the first two terms of
  per_row_exact take at the central inputs, less D3's 1.615 ms of B2 savings (*computed* on estimated inputs). D3
  still runs with r_ex(k, p).

Past the first line, the three bands cover every value of β_ex(2,048). The predicted interval of §6 straddles 0.60.
R-b and R-c differ in what gets revisited, and neither builds B1 alone. Every route that leads to B1 or B2 waits on
R1 and R8.

A guard refusal at k = 2 or 3 removes that k's r_ex from D3, and is reported. At k = 5 to 8 it is reported only. The
guard lines also inform U3.

Any stop is a proposal; the operator decides.

## 8. Cost

*Estimated* on l40sx1 at $1.80/h, billed by the second. The served-grouped job reached its first `fusedrun` 23 s
after it started, oracle, sha256 of the sealed file and config writes included (*measured*,
`served-grouped-2026-11-02.txt:5` and `-brut/job.log:28`). Its served-door processes loaded the 4B in 4.0-4.1 s and
spent 0.4-1.9 s outside load and decode (*computed* from its `date -u` lines). Here: 16 processes of 4.7-6.5 s
each (a 4 s load, six passes of at most 60 ms, a guard of at most 8 one-token calls). That is 110-140 s billed,
**$0.06-0.07**, about **$0.08** with the closing listing. The roadmap carries 4 billed minutes, **$0.12**, as the
margin figure. Timeout 20 min, **$0.60** at worst (*computed*). Other anchors in `docs/data/jobs.csv`: `f1e-tail`
(3 prompt lengths, 2 billed min, $0.03) and `f1e-tail3` (4 lengths, 2 min, $0.04).

## 9. What this run does not establish

- The exact verify's own t(k): its per-row attention, head and KV terms are *estimated* constants here, and E0
  (`prereg-speculation-2.md`) measures them. They are 48 % of the derived row cost at 2,048, so β_ex stays
  *estimated* after this run. The per-row argmax read of a verify (`argmax_last`, `model.rs:2390-2392`) is left
  out of the formula: a reduction and a synchronizing 4-byte read a row, small against 5 ms (*estimated*).
- Row-invariance of anything: the guard compares a batched path, attention included, with one-row calls.
- Any long-context cost: every pass starts at position 0.
- The 8B and the 14B.
