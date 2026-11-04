# Preregistration. The split-KV attention kernel after its fixed-cost revision: alone, then in the served 4B

**Finalized on 2026-11-04, timestamped with OpenTimestamps before the first measurement (rule 2).** After the
stamp this file is never edited; a deviation goes beside it in
`proofs/preregistration-attn-bench-r2-2026-11-04-ECARTS.md`. Operator go: "ok pour les 3" (2026-11-04, relayed
by the session of the throughput audit, for three card runs, this one second), to be confirmed in this session
before the launch. Cost: **about $0.48**, *estimated* in §8, timeout 33 min, so **$0.99 at worst**.

It repeats `proofs/preregistration-attn-bench-2026-11-04.md` (job 6aeb53b2, journal
`docs/mesures/attn-bench-2026-11-04.txt`) on the revised kernel, with one short length added to each step.

## 1. Why this run exists

The first card run found the kernel exact on the card and slow where it should not be: 42.3 / 54.5 / 95.7 us a
launch at 2,048 / 4,096 / 8,192 (G = 4), η 0.230 / 0.356 / 0.406, and in the served 4B ×0.93 against the grouped
form after 5 tokens, ×1.23 at 8,192 (*measured*). The excess over the DRAM floor grew with splits x heads, about
0.4 us each (*computed*), which points at the last CTA's split fold: dependent volatile loads, two L2 round trips
per (split, head), in series. That is a hypothesis read off timings, with no profiler.

The revision (commit c8a01d3, operator: "ok pour le cout fixe") changes no output bit: the fold loads a chunk of 8
splits at once, then folds it in the same order; a grid of one split writes its output directly; the in-CTA unit
fold is unrolled over heads. This run tests the hypothesis (did the excess go?) and measures what the revision
does to the served decode. It is a bench plus an informative in-situ reading: it gates nothing (METHODE §1).

## 2. What the code holds, and what the Mac showed

*Measured* on the dev Mac (`docs/mesures/attn-kernel-host-2026-11-04.txt`, section "Revision of the fixed cost"):
the kernel text, executed by clang++ on `llvq-cuda/tests/host_cta.h`, equals the CPU replay bit for bit on every
launch of the first run's tests plus 8 that cross the 8-split chunk (rows of 8, 9, 10 and 17 splits, a verify call
straddling 2,304, a wide grid in a permuted order, fresh rows), and under UBSan. Six new mutants killed (chunk
folded in reverse, maxima in the wrong layout, no barrier after a chunk, the one-split path at two splits, the
chunk's first accumulators for all, an fma in the chunked merge); the earlier kernel set rerun, all killed but the
known equivalent K5. Clippy at zero warnings on the Mac and for Linux. Unchanged: the replay, the wiring, the
served defaults.

## 3. What runs

One image built from the commit that carries this file. One job, l40sx1: `ops/jobs/attn-bench-r2.sh`.

1. **Oracle** on Qwen3-0.6B, rule 10.
2. **`attnbench 256,2048,4096,8192`**: the four entry points' driver reports; identity of every variant against
   the replay (the load-time check, then each L and k in {1, 2, 4, 8}, eager and wide grids, verify rows alone);
   timing on 36 (G = 4) or 40 (G = 5) per-layer KV sets, 2 discarded and 7 timed rounds, as in the first run. L =
   256 is new: one split, an eager grid of one CTA a head, the fixed cost alone.
3. **The served 4B**, `LLVQ_CONFIG=<c> LLVQ_CONTEXTS=5,256,2048,4096,8192 fusedrun <sealed 4B> 256`, kernel arm
   then grouped arm, the two configs of the first run at their bytes (sha256 d74ea29a… and 2455fe85…), 1
   discarded + 5 timed rounds each. 256 is new.

Derived, *computed*: per-launch time t / sets; η_k(L) = sets x 4,096 x L / (t x 864 GB/s); r_k = t(k) / t(1),
round by round; the in-situ ratio as [min kernel / max grouped, max kernel / min grouped].

## 4. Gates, each one voids the run

1. The image builds, and `oracle` prints MATCH.
2. The source line says `embedded` with the sha256 of `attn_dec.cu` at the commit that carries this file, and
   every entry point prints its report.
3. The identity lines and the timing table (64 rows) are printed in full.
4. No `LLVQ_*` variable is in the job environment.
5. Each sweep arm prints its served config line, the int4 parity line, the served tile and five lengths; the
   kernel arm prints its load-time identity line. A kernel arm that fails to load voids step 3 only.

## 5. Signed predictions

| quantity | point | interval |
|---|---|---|
| identity, every variant, every (L, k, G) | **pass** | |
| spill, V-64, G = 4 and 5 | **0** | |
| registers, V-64, G = 4 | **165** | [150, 210] |
| per launch at L = 256, V-64, G = 4 | **8 us** | [4, 20] |
| η_k(2,048), V-64, G = 4 | **0.31** | [0.25, 0.45] |
| η_k(4,096), V-64, G = 4 | **0.60** | [0.40, 0.75] |
| η_k(8,192), V-64, G = 4 | **0.77** | [0.55, 0.88] |
| η_k(8,192) at G = 5 over G = 4, V-64 | **0.95** | [0.80, 1.05] |
| r_8(8,192), V-64, G = 4 | **6.5** | [4.0, 8.0] |
| served kernel / grouped at 5 | **0.94** | [0.90, 1.00] |
| served kernel / grouped at 256 | **0.97** | [0.92, 1.03] |
| served kernel / grouped at 2,048 | **1.08** | [1.02, 1.15] |
| served kernel / grouped at 4,096 | **1.25** | [1.12, 1.35] |
| served kernel / grouped at 8,192 | **1.45** | [1.25, 1.60] |
| grouped arm, against the first run (136.2 / 115.2 / 98.9 / 78.0) | **same** | each within ±2 % |

Reasoning, *estimated*. Removing 0.38 us per (split, head) and adding 1.5 to 3 us of chunked fold leaves about
32 / 32 / 50 us a launch at 2,048 / 4,096 / 8,192. In the model the kernel arm then saves about 0.37 / 0.8 / 1.6 ms
a token at those lengths, from 8.43 / 8.88 / 10.44 ms. At 5 tokens the fold held one split: only the fence,
atomic and reload of the one-split path go, a few us a layer, so most of the short-context loss stays. Verify
re-stages per row, unchanged by this revision. The known flaw: the attribution is a fit of three lengths.

## 6. Decision rules

Readings, evaluated in this order. None is a gate, none pronounces a kill (METHODE §1).

1. **Identity.** A variant that fails one identity check has a bug, excluded and fixed; if V-64 fails, no timing
   of this run is read.
2. **The hypothesis.** η_k(8,192), V-64, G = 4, at least 0.60: the serial fold was the excess, confirmed; the
   in-situ A/B (`prereg-attention-2`) is finalized on this kernel. Between 0.45 and 0.60: partly; the remaining
   excess is attributed with the L = 256 time before any further revision. Below 0.45: the fold was not the
   excess, the hypothesis is refuted, and the next step is a profile (Nsight Compute on one launch) rather than
   another guess. Otherwise (identity failed): rule 1.
3. **Short context.** Served kernel / grouped at 5 tokens, lower end of its interval at least 0.97: the short-context
   loss is gone. Otherwise it remains, and is diagnosed in the model (the CPU work around the launch, the q copy,
   the cache writes candle launches) before the in-situ A/B is stamped; the lead's stop line (5 tokens < 0.97)
   is reported to the operator as it reads, a proposal, the decision his.
4. **Served 4B.** Informative. No served default changes and no T8 starts on this run.

## 7. Deviations

Any departure after the stamp goes in `proofs/preregistration-attn-bench-r2-2026-11-04-ECARTS.md`.

## 8. Cost

*Estimated* on l40sx1 at $1.80/h, from the first run (`attn-bench-2026-11-04`: 15.5 min running, $0.46): the
same steps plus L = 256 in the bench (seconds) and 256 in each sweep (a 256-token prompt and 6 decodes of 256
tokens, about 15 s an arm). About 16 min, **$0.48**. Timeout 33 min, **$0.99** at worst.

## 9. What this run does not establish

The kernel against the grouped form round by round (the arms run in sequence; that is the in-situ A/B's), quality
(the kernel is not bit for bit with the grouped form; its licence is the operator's T8 decision, not granted), the
cause of the excess beyond the timings (no profiler), the 8B and the 14B in situ.
