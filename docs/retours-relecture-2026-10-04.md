# The review of paper 2, priced

Answering this review in full costs about **$285 and 10 to 13 weeks of work**, against the $241.88 the project has
spent since 2026-08-05 (*measured*, `data/jobs.csv`). This document prices each ask. The operator retained lines 1, 2
and 3 of section 4 on 2026-10-04, for **$4 of card time**. Rule 1 holds for the rest.

Source: a review of *Tetra: Serving Leech-Lattice Quantized LLMs at 2.7 Bits per Parameter*, submitted 2026-09-30 to
an automated review service, returned 2026-10-04. Its verdict is positive and its overall line is "strong potential
for publication at a top-tier venue". It is not a venue decision.

## 1. What the review does not do

**It contests no number.** No table, no interval, no claim of ours is called wrong. The only correction it carries is
in its own summary, which compresses our rate to "an effective 2 bits per weight" where the paper says 2.148 kernel
b/weight and 2.70 to 2.73 b/param.

**Seven of its nine experimental gaps are already in `limitations.tex`**: one card, one calibration draw, MMLU on the
dense reconstruction, int4 selection on test questions, GSM8K too easy, no perplexity for the sealed files, batch 1
and short context. The review asks us to close the list the paper wrote itself. Section 2 prices that closing.

**One of its asks is already in the paper.** Question 7 asks about incoherence processing, HIP or QuaRot. The pipeline
rotates its input and cites QuaRot at `paper2/sections/tetra.tex:20`. One sentence of cross-reference, $0.

## 2. The paper's own limitations, priced

Unit costs below are *measured* from `data/jobs.csv`: l40sx1 at $1.80/h, a100-large at $2.87/h, one full MMLU split
arm at $0.70 at 4B, $0.98 at 8B and $1.59 at 14B, one row-scale training arm at $4.00, $5.66 and $9.78, the 14B
encode at $15.03. The 4B and 8B encodes run on the Mac at $0.

| # | Ask | Route | Cost | Dev |
|---|---|---|---|---|
| L1 | A second kind of GPU | served decode at three sizes on a100-large | $2 *estimated* | none |
| L2 | Calibration variance for Tetra | 2 extra draws at 4B: 2 Mac encodes, 2 trainings, 2 MMLU, 2 ppl | $9.88 *computed* | none |
| L3 | MMLU through the served kernel | mixed route of ROADMAP §2.1 | $20 *estimated* | 1 day, `ppl` must read `LLVQ_CONFIG` |
| L4 | Selection bias quantified | see N6 below | $15 *estimated* | 2 to 3 days |
| L5 | A harder reasoning test | thinking mode on GSM8K at 4B, plus its two references | $3 to $9 *estimated* | none for thinking mode |
| L6 | Perplexity for the sealed files | dense path, one arm a size | $3 *estimated* | none |
| L7 | Batch above 1 and 32k context | a resident-model benchmark | $3 *estimated* | 1 to 2 weeks |

L2 at the other two sizes costs $13.28 at 8B and $52.80 at 14B (*computed*), the 14B encode dominating. The review
asks for three to five draws; three at 4B is what the 2026-08-25 noise figure already used, on a `Planes14` base.

L3's full census through the kernel at three sizes costs $70 instead of $20 (*estimated*, ROADMAP §2.1).

L5's price moved. ROADMAP §4 carries $3 to $5 for this row, priced before the token budget. GSM8K at 4B cost $2.13
for 316 generated tokens a problem on average (*measured*, `data/jobs.csv`); thinking mode writes longer chains, so
one arm near three times the tokens lands near $6, with the vLLM references at $0.16 to $0.31. MATH or AIME is a new
harness, 3 to 5 days, before any run.

Also in scope and already on the roadmap: **the kernel bench does not time equal work**, 216 matrices for Tetra
against 252 for every other arm. The review's complaint that our GB/s is low rests on those passes. About $1 for the
run, and 1 to 2 days of code: the 36 int4 `v_proj` must be timed, and `planesbench` refuses `d_in` 17408, the 14B
`down_proj` (`planesbench.rs:1917`).

## 3. The asks that are new

### N1. Hardware counters. Blocked on this platform, not undone.

Nsight Compute was tried on 2026-08-19. It installed, it attached to `planesbench`, and the device refused the
counters: `ERR_NVGPUCTRPERM`. The prereg closed the lead on the spot, for $0.86 (*measured*,
[f3-events](mesures/f3-events-2026-08-19.txt)). `paper2/sections/kernel.tex:112` already states that our rented
platform exposes no performance counters.

Answering question 1 needs a card where we hold counter rights, so a provider we do not use today. About $5 to $15
*estimated*, plus an operator decision on a new vendor.

### N2. Table placement. Untried, and the cheapest kernel lead.

`llvq-cuda` contains zero `__constant__` and zero `__ldg`. The decoder tables sit in global memory and reach the SM
through L1; shared memory holds the activation tile alone (`extern __shared__ float xs[]`). Constant memory,
per-CTA replication and const-cache hints have never been measured.

The mechanism is already measured: across the tile sweep the Tetra arm swings 19.1% where `Planes14` swings 1.5%
(*measured*, [tuile-l40s](mesures/tuile-l40s-2026-09-20.txt)). That is the paper's own evidence that the tile steals
L1 from the decoder table, and it is what makes this lead worth more than the others.

Cost: 2 to 4 days of dev, about $2 for the run.

### N3. Launch fusion and persistent CTAs.

`LLVQ_FUSE=0` is served because `Tetra48` carries no segmented kernel. The floor these ideas attack is measured:
`nullk` finishes 252 projections in 2.306 ms without reading a weight, where QTIP finishes the same in 2.246
(*measured*, [f2-p3-qtip-banc](mesures/f2-p3-qtip-banc-2026-08-21.txt)).

Cost: 1 to 2 weeks of dev an idea, about $2 a run. Caveat: A2, CUDA Graphs, bought +12.6% of throughput for +47% of
VRAM and is closed by rule 3. Fusion is a different mechanism and does not reopen it.

### N4. Rank-table ablation, question 5.

Every variant of the 2,048 kept rank rows or of the N0/N1 balance changes `codebook_fingerprint` and needs a full
re-encode. At 4B that is 2 h 27 on the Mac and $0, then $0.70 for the MMLU arm and about $0.20 for the bench. Three
variants cost about $3 and 8 Mac hours, after 2 to 3 days on the table generator.

This is also the format-level test of N2's hypothesis. Halving the table to 8 KiB halves what the activation tile
competes with in L1, at a known cost in rate-distortion.

### N5. AWQ with a 4-bit embedding, question 4. Buildable with what exists.

`ops/awq_dequant.py` already rebuilds `Qwen3-4B-AWQ` as a dense f16 checkpoint our harness loads, and `embedq.rs`
carries MLX's exact int4 g64 scheme. Half a day of Python passes the embedding tensor through the round trip offline,
with no change to the Rust harness.

Cost: one MMLU arm a size, $0.70 + $0.98 + $1.59 = $3.27 (*computed*), doubled to $6.54 with the FP16 control.

Our own anchor: going from an 8-bit to a 4-bit embedding cost 0.35 pp at 4B, under the 0.43 pp bar, so undetected
rather than null (*measured*, ROADMAP-QUALITY row 4, on a `Planes14` base).

### N6. A validation split for the int4 selection, question 8 and the selection-bias gap.

The contamination gate exists: `ops/mmlu_aux_overlap.py` reports 0 scored items in `cais/mmlu` `auxiliary_train`.
Redoing the selection there and reading the test split once costs about $5 a size, so $15 for three (*estimated*).
Re-sealing a size whose selection changes is $0 at 4B and 8B and $15.03 at 14B.

N7 below says this route cannot be replaced by a free diagnostic.

### N7. The a priori int4 heuristic, question 8. Measured 2026-10-04, and it fails.

The Hessian ratio does not predict where int4 buys MMLU. `hratio`'s dump of 2026-09-18 was read against the six
measured int4 arms of the same 4B file, and no variant of the statistic reproduces their ranking (*measured*,
[hratio-4b](mesures/hratio-4b-2026-09-18.txt)):

| statistic against | Spearman over six types | exact two-sided p |
|---|---|---|
| MMLU delta | +0.5429 | 0.297 |
| MMLU gain per b/param | −0.7143 | 0.136 |

The sign flips on the quantity an int4 budget actually spends. This is the seventh case in the repository where a
better local proxy ranks against the model.

Two facts survive. Both the proxy and MMLU put `down_proj` first, 4.95 and +3.79 pp. And `o_proj` sits in the served
4B file for a budget reason, not a ranking error: `down_proj` whole costs +0.4641 b/param and takes the model to
3.2286, over b_max of 3.00, where `o_proj` at +0.1954 fits at 2.9599 (*measured*,
[q5-alloc-int4](mesures/q5-alloc-int4-2026-09-16.txt)). ROADMAP §2.3 should carry that reason.

What stays open is per-matrix selection. The ratio spreads by a factor of 850 inside `down_proj` alone, and the 8B
winner picked 17 of its 36 matrices. This reading tested type medians only.

### N8. The four baselines exist, and two of them compete with us directly.

Verified 2026-10-04 for $0. None of the four is in paper 2's 24 references.

| work | venue | id | code | why it matters here |
|---|---|---|---|---|
| GLVQ, grouped lattice VQ | NeurIPS 2025 | [arXiv:2510.20984](https://arxiv.org/abs/2510.20984) | [xzhang9308/GLVQ](https://github.com/xzhang9308/GLVQ) | learned per-group lattices plus companding, claims lower perplexity than QTIP and QuIP# at 2 b/w at similar throughput |
| LiftQuant | ICML 2026 | [arXiv:2606.04050](https://arxiv.org/abs/2606.04050) | not checked | fractional bit widths such as 2.4 b/w with standard kernels, so it competes on the served-bits axis |
| KronQ | COLM 2026 | [arXiv:2607.07964](https://arxiv.org/abs/2607.07964) | not checked | Kronecker-factored Hessian; uses the gradient covariance for mixed-precision allocation, which is the problem N7 just failed to solve |
| Qronos | ICLR 2026 | [arXiv:2505.11695](https://arxiv.org/abs/2505.11695) | not checked | corrects the error accumulated by previously quantized layers, which is ROADMAP-QUALITY row C, our own intra-block sequencing defect, in published form |

**GLVQ is the finding.** It is a learned lattice at 2 bits with throughput claims, published before our submission,
and we do not cite it. Positioning against it is writing, so $0 and about half a day, and it is owed whatever we
decide about benchmarking. Our distinct claim is served VRAM bits per weight at a fixed codebook, which no learned
codebook gives for free, and that claim has to be made explicitly rather than by omission.

Benchmarking them is a different budget. QTIP end to end on Qwen3 needs their quantization pipeline, since the
released QTIP models are not Qwen3: about $30 to $60 at 4B plus 3 to 5 days of dev (*estimated*). GLVQ has released
code, so the same order applies, with the risk that their pipeline has never seen a Qwen3.

### N9. Presentation. $0.

Decoding pseudocode for the 48-bit word, one table of every constant and field, and one memory-accounting table per
component. Every number exists (`rtbits`, `data/echelle-formats.csv`). About 1.5 days of writing.

## 4. The order, and what is retained

Lines 1, 2 and 3 were retained on 2026-10-04. The rest is unordered and unfunded.

| # | item | cost | running total | dev | state |
|---|---|---|---|---|---|
| 1 | Bundle A: N7, N8, N9, the QuaRot cross-reference | $0 | $0 | 3 days | N7 and N8 done; N9 and the QuaRot line not written |
| 2 | Perplexity of the three sealed files | $3 | $3 | none | **done for $0.33**, [ppl-scelles](mesures/ppl-scelles-2026-10-04.txt) |
| 3 | Equal work in the kernel bench, 216 to 252 | $1 | $4 | 1 to 2 days | the `d_in` 17408 refusal is lifted; the int4 arm is not written |
| 4 | AWQ and FP16 with a 4-bit embedding | $6.54 | $10.54 | half a day | not retained |
| 5 | A second kind of GPU | $2 | $12.54 | none | not retained |
| 6 | MMLU through the kernel, mixed route | $20 | $32.54 | 1 day | not retained |
| 7 | Table placement in constant or shared memory | $2 | $34.54 | 2 to 4 days | not retained |
| 8 | Launch fusion, rank-table ablation | $7 | $41.54 | 3 to 5 weeks | not retained |
| 9 | Calibration variance, 4B only | $9.88 | $51.42 | none | not retained, last by decision |

Full compliance adds L2 at 8B and 14B, the $70 census, L4, L5, L7, N1 and the QTIP arm, for about $285 in total. That
exceeds the project's entire spend to date.

Perplexity is the best buy on the list. Three dollars, no code, and it is the number a quantization paper is read
for. `sealed::load` already serves `ppl`, so the three files read as they are on the dense path.

## 4 bis. Where to resume, state of 2026-10-05

Spent: **$0.33** of the $9 cap, so $8.67 left. The ledger reads $242.21 over 215 priced jobs.
Branch `retours-relecture-ia`, seven commits, not pushed.

Three things are open, in this order.

**The device-clean cost of sealing, deviation É2 of the perplexity prereg.** The Metal route is
abandoned: `hf buckets cp` has no timeout and no resume, it stalled three times out of four
between 430 and 851 MB, and killing it restarts that file from zero. The 4B came down whole and
sits in `~/scelles-2026-10-04/qwen3-4b-sealed.bin`, sha256 `886391a8`, so that size is ready for
a Metal run; the 8B and the 14B are not worth the lottery.

The better route costs about **$0.25** and transfers nothing: read the two **trained bases** on a
card, `dclm-ft-2026-09-19/qwen3-4b-dclm-ft.bin` (1.79 GB) and
`dclm-8b-ft-2026-09-21/qwen3-8b-dclm-ft.bin` (4.36 GB), both already in the bucket. That gives the
device effect paired on a quantized file at 8B, since the Metal base's twelve per-window NLLs are
on disk, and the cost of sealing device-clean at 4B and 8B. It needs a deviation on
`preregistration-ppl-metal-2026-10-04.md`, which measured nothing, and a new prereg.

**The int4 arm of the bench.** Design settled, not written: a new arm `tetra48q4` rather than a
redefinition of `tetra48`, because redefining would take the arm from 216 to 252 matrices under
the same name and make the published 4.078, 3.423 and 3.553 ms incomparable to everything that
cites them. `tv_q4_h.cu` enters the bench by a cross-crate `include_str!` at the end of the NVRTC
unit, since `llvq-cuda` cannot depend on `llvq-llm` and appending is the only placement that moves
no published arm's fragment.

**The living documents and the paper.** `ETAT.md` §5, `ROADMAP.md` §2.4 and
`paper2/sections/limitations.tex` still say no sealed file has a perplexity. One question is the
operator's: the three numbers weaken the paper's position rather than strengthen it, because
perplexity names the 4B our best object against AWQ and MMLU names the 14B.

Two smaller debts found on the way. `llvq-llm/src/fused.rs:698` says `tv_q4_h.cu` "has never run
on a GPU", which `jobs.csv` contradicts since 2026-09-25. And `hratio`'s 4B dump had sat
unjournalled since 2026-09-18; it now has one.

## 5. Decisions owed

Rule 1 applies to every line. No cap is in force; one is owed before lines 2 and 3 run (`ETAT.md` §5).

| decision | default if silent |
|---|---|
| a cap for lines 2 and 3, which cost $4 together | no paid job |
| lines 4 to 9 | not done |
| whether to cite GLVQ, LiftQuant, KronQ and Qronos without benchmarking them | the paper ships without them, which the next reviewer will also catch |
| a new vendor for hardware counters | question 1 stays answered by a platform refusal |
| whether this review changes the venue decision | preprint only |
