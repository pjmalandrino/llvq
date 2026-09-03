# docs/data, the clean measurement data (2026-08-05 → 07)

CSV files ready to plot, decimal points, units in the headers. Every value
comes from a file in `docs/mesures/` (source column, or the file's README);
nothing is smoothed.

Two columns are derived, and the net recomputes both.

1. `echelle-formats.csv::pct_byte_bound` exists in no journal. It is
   `round(gbps / 661 × 100)`, where 661 GB/s is the FP16 arm of the same run.
   It gives the fraction of its byte bound that a kernel turns into time. It is
   stable under the choice of round: recomputed on the **medians** rather than
   on the benchmark minima, it yields the same integers (100/65/65/54/30/40/88).
2. `echelle-4b-8b.csv::vram_margin_vs_awq_pct` (−2.6 / −10.6 / −5.5) is
   `round((llvq2 / awq4 − 1) × 100, 1)` over the two `vram_bits_per_param` of
   the same size. No journal writes it in that form. It is the margin the prose
   cites, and it must follow the two rates or one of the three is wrong.

`paper/scripts/check_tables.py` recomputes both at every `make` and refuses the
build if the CSV and the paper's table diverge.

These CSVs must be RECTANGULAR. An unescaped comma in a free-text field
(`notes`, `what`) cuts the row: `csv.DictReader` throws everything that follows
into the `None` key, and the note reads truncated mid-sentence. Nothing fails
on its own, precisely because no checked table reads those columns. Convention:
in a free-text field, separate clauses with `;` or ` - `, **never with a
comma**, and `check_tables.py::check_csv_shape` enforces it.

| file | contents | source |
|---|---|---|
| `campagne-finale.csv` | the 4 arms × 5 factors table (disk, VRAM, speed, ppl, MMLU) | a4-campagne + campagne-finale-bras4 |
| `echelle-formats.csv` | the **10 arms** on the benchmark (b/weight **kernel**, ms, GB/s, % of the byte bound, ratio vs FP16 with range) | **f2-p3-qtip-banc-2026-08-21 (phase 2)**, re-sourced on 2026-08-21 from the 10-arm run. Adding the QTIP row to the seven-arm file it came from (`golay70-v2-sept-bras`) would have put rounds from two processes side by side, which the paper's methodology forbids, so the **whole** table was re-measured with every arm present. The incumbents reproduce within the spread (`Planes14` ×2.15 on both runs, `Golay70` v2 ×1.77 then ×1.78) |
| `phases.csv` | the per-phase time of a token, 4 profiles (fenced: attribution, not a total) | phases-2026-08-07 |
| `progression.csv` | the arc of the week: VRAM/throughput/b-param at each step | mini, a1, planes14-fusedrun, nuit |
| `echelle-4b-8b.csv` | the model scale, 3 arms × 3 sizes (ppl, ratio, MMLU micro, **b/param whole model**) | a4-campagne + campagne-finale-bras4 (4B), campagne-8b-qualite (8B), campagne-14b-qualite (14B); `params_total` and the `vram_*` columns come from elsewhere: rtbits-planes-8b (4B, 8B) and rtbits-14b-2026-08-17 (14B) |
| **`mmlu-appariee.csv`** | the **PAIRED** MMLU gaps: 3 sizes × 3 pairs, point + CI95 + SE + McNemar + discordance rate | mmlupair-4b-8b-2026-08-13 (4B, 8B), campagne-14b-qualite (14B `f16 −` …) and **mmlupair-14b-2026-08-17** (14B `awq4 − llvq2`) |
| **`ppl-appariee.csv`** | the **PAIRED perplexity intervals**: 3 sizes × 3 pairs, excess in % + CI95 + ratio + mean and SE of the NLL differences + t | ppl-appariee-4b-2026-08-17 (4B), ppl-appariee-8b-14b-2026-08-17 (8B and 14B) |
| **`ppl-genou.csv`** | the **slowdown IN PERPLEXITY**: 2 steps × 2 references, plus the 2 knee tests (the difference of the two steps) | ppl-appariee-4b-2026-08-17 |
| `mmlu-dumps/` | the MMLU dumps **question by question**, 3 sizes × 3 arms, the raw material of the pairs above | 4B/8B/14B campaigns; the three 14B files committed on 2026-08-17 (see the bucket lesson below) |
| `jobs.csv` | every GPU job: id, duration, cost, what it measured | ops/run.py monitor |
| **`campagne-14b-vitesse-2/`** | the **raw output** of the four steps of the served 14B, as the job wrote it into the mounted bucket | job `6a83121be55292eada79b611`; summary in [`mesures/fusedrun-14b-2026-08-17.txt`](../mesures/fusedrun-14b-2026-08-17.txt) |

### Why the SERVED 14B cells go into no CSV (2026-08-17)

The 14B was served for the first time on the evening of 2026-08-17,
**42.9 tok/s and 9.39 GB on the card**, against 17.0 and 29.54 for the dense
arm. Those two cells are added to **no** CSV, and the decision is argued file by
file rather than by principle:

1. **`campagne-finale.csv` is a 4B table**, indexed by *arm* and pinned column
   by column onto `tab:campaign` by `check_campaign_table`. A 14B row has no
   column to land in, and forcing it there would break the net, which is the
   correct signal.
2. **`tableau-8b.csv` is the 8B analogue.** The per-size pattern would therefore
   call for a `tableau-14b.csv`. It is not created: the 14B has only **two**
   throughput arms (dense f16 and served LLVQ), no AWQ measurement in its
   engine, and no per-arm disk or VRAM beyond what `echelle-4b-8b.csv` already
   carries. The schema would promise four arms and carry two, a table whose
   shape lies about what was measured.
3. **`echelle-4b-8b.csv` has neither a throughput column nor a GB-on-card
   column**: its grain is *(model, arm)* over **quality** and **b/param whole
   model**. Adding one would cost two things. (a) The 4B and 8B throughputs
   already live in `campagne-finale.csv` and `tableau-8b.csv`; copying them here
   would create a **second home** for the same number, exactly what the
   "physical separation" argument below exists to prevent. (b) The three `awq4`
   cells would have to be decided, and their only existing number,
   **200.49 tok/s in vLLM**, comes from **another stack** and divides with none
   of ours; the cell is left empty **everywhere else in the record**, and a new
   column is not the place to fill it.
4. **`progression.csv` is the dated arc of the kernel work at 4B**
   (`tab:progression`, 08-05 → 08-07), indexed by step. A 14B point is not a
   step in that arc.

What this choice leaves out of the CSVs must be known: the only two served 14B
cells live in their journal and in the `what` of `jobs.csv`. No net checks them.
Creating their home is a structural decision that belongs to the operator, and it
should be taken only on the day the 14B has as many arms as the other two sizes,
otherwise a table gets built for two numbers.

One result of this run needs no column: the "GB on card" are a **third route** to
b/param whole model, the one `rtbits-14b-2026-08-17.txt` declared missing at
14B. It yields **5.0866 b/param** (*computed*: 9.39 GB × 8 / 14,768,307,200)
against **5.106** measured by `rtbits` on the exact bytes, a gap of **−0.38%**,
inside the ±0.5% band set **before the run** (prereg §2). And the dense arm
yields **16.0018** against 16.000 exact by construction, 0.011%: the 14B
`params_total` gets its third route at the same time. The **published** figure
stays the 5.106 from `rtbits`. This is a **cross-check**, and it does not
replace anything; it is computed on a card display rounded to the hundredth of a
GB, the very route by which the "≈ 5.15" of the 4B fell **below** the right
value.

### Why TWO perplexity files, and not one more column (2026-08-17)

The nine perplexity intervals could have been written into
`echelle-4b-8b.csv`. **They are not there, and the device is the same as for
MMLU: physical separation.**

1. **The grain is not the same.** `echelle-4b-8b.csv` is indexed by
   *(model, arm)*, one row per artifact. A paired interval bears on a
   *(model, PAIR)*: it belongs to neither of the two arms, it is the comparison.
   Housing it in a per-arm file would force either duplicating it on both sides
   or inventing hybrid arm-pair rows, exactly the confusion that the "no paired
   column" note below exists to prevent.
2. **Two columns that are not in the same file cannot be subtracted by
   accident.** That is the literal argument of the warning on
   `mmlu-appariee.csv`, and it holds word for word here: subtracting two `ppl`
   values from `echelle-4b-8b.csv` does not produce a tested gap, and the knee
   is precisely the number one would be tempted to manufacture that way.
3. **The knee has yet another grain**, *(step, reference)* rather than
   *(model, pair)*, hence the second file rather than an extra column or a union
   key. `ppl-genou.csv` carries the two steps on each of the two references,
   plus the two knee tests, all in the same shape.

No table in the paper consumes them today: those numbers appear there in prose
(§ "Perplexity gets error bars"). The net checks them anyway, in three
independent ways: literal pinning against the journals (`PPL_PAIRED`,
`PPL_KNEE`), **exact internal derivations** (ratio = exp(mean difference),
excess = ratio − 1, t = mean / SE, additivity of the three pairs of a size, and
the knee = difference of the two steps), and a **cross-check** against
`echelle-4b-8b.csv`, whose `ppl_ratio_vs_f16` is the same quantity reached by
another route. 18 mutants killed out of 18 when the guard was written. A guard
written before the table is the goal: the table that comes will inherit it
instead of asking for one.

One column deliberately escapes derivation: `ppl-appariee.csv::ratio` for the
`llvq2_over_awq4` pair. It is **1.1458** at 14B (exp of the mean difference,
exact) while the quotient of the two already-rounded ratios of
`echelle-4b-8b.csv` gives **1.1457**. Both are right in their own accounting;
deriving that one would impose a rounding artifact instead of verifying an
agreement. It is pinned and held by the additivity identity, not by the
quotient.

`echelle-4b-8b.csv::params_total` is filled on the three 14B rows since
2026-08-17. The passage of `rtbits` over a sealed 14B is
[`mesures/rtbits-14b-2026-08-17.txt`](../mesures/rtbits-14b-2026-08-17.txt).
The artifact had never been brought back after the 08-10 campaign; it was
sleeping in the bucket `Pier-Jean/jobs-artifacts`
(`qwen3-14b-c12-3f21abde/qwen3-14b-llvq.bin`, 6,506,354,741 bytes), from where
it was read back for $0, bandwidth alone, no GPU. The count is therefore
**exact like the other two**: 4B 4,022,468,096 · 8B 8,190,735,360 ·
**14B 14,768,307,200**, the first two read in
[`mesures/rtbits-planes-8b-2026-08-09.txt`](../mesures/rtbits-planes-8b-2026-08-09.txt)
(l. 114 and 275), the third in the 08-17 journal, and cross-checked there by a
second route, the arithmetic of the architecture (§3 of the journal: the eight
integers, including the 163 carried tensors that the resumption note had set in
advance as the sealing criterion).

**The trap stays whole, and it must be kept**: the only 14B count then in
circulation was **13,212,057,600 quantized weights**
([`archive/reprise-14b-2026-08-09.md`](../archive/reprise-14b-2026-08-09.md), l. 38).
That is the numerator of the **projections**, **not** a whole-model total; it is
missing 1,555,824,640 of embedding and 424,960 of norms. Putting it in that
column would have been exactly the denominator confusion that the batch A errata
calls GRAVE, and it is off by 10.6%. The empty cell was the right choice while
the true count was missing; the count changed, the rule did not.

**And the 14B memory row, which that hole blocked, now exists**: `Planes14` +
q8 embedding weighs **5.106 b/param whole model** against **5.404** for
`Qwen/Qwen3-14B-AWQ` (safetensors bytes of the official repo read through the
Hub API, ÷ `params_total`), **5.5% under AWQ**, as at 4B (−2.6%) and 8B
(−10.6%). The margin **is not monotone** and tells no trend: it follows the
share of the embedding (9.7% · 15.2% · 10.5%), which AWQ leaves in f16 and which
we take to q8. Detail and provenance labels in §2 of the journal.

**`echelle-4b-8b.csv` still carries NO paired column, and that is deliberate:
subtracting two of its `mmlu_micro_pct` values does not produce a tested
gap.** The trap is structural: `awq4 − llvq2` reads off the page, and that is
where the "6.09 pp" of the 14B came from (78.21 − 72.12), a bare difference. The
nine paired gaps live in a separate file,
[`mmlu-appariee.csv`](mmlu-appariee.csv), 3 sizes × 3 pairs, each with its
point, its CI95, its SE, its McNemar, its discordance rate and its journal.
Physical separation is the device: two columns that are not in the same file
cannot be subtracted by accident.

**A rounding divergence to know before believing a 0.01 pp gap**:
`echelle-4b-8b.csv::mmlu_delta_pp` is −10.56 at 8B (that is 65.52 − 76.08, two
already-rounded micros) while `mmlu-appariee.csv::f16_minus_llvq2` is 10.57 (the
stratified Δ computed over the questions, before rounding). Both are right in
their own accounting and neither corrects the other; the paper cites 10.57,
which is the paired number.

The AWQ − LLVQ pair at 14B is **+6.09 pp, CI95 [+3.62; +8.52], SE 1.25 pp,
McNemar p = 1.143e-11**
([`mesures/mmlupair-14b-2026-08-17.txt`](../mesures/mmlupair-14b-2026-08-17.txt)),
and it was computed for $0. **The mechanism of the error that had declared it
impossible is worth keeping, because it is reproducible.** The search concluded
"verified on 2026-08-16: no trace left **on the machine**", a search in ONE
place. The campaign job wrote into the **mounted bucket**, which exists
precisely so that outputs survive the container: the three dumps had been
sleeping there since 2026-08-10
(`hf://buckets/Pier-Jean/jobs-artifacts/campagne-14b-qualite/`). Real cost of
the "impossible correction": **579 kB of bandwidth**.

> ### Permanent rule: inventory the bucket BEFORE budgeting a re-run
>
> **Any output declared lost deserves an `hf buckets ls` before its
> reproduction is costed.** The bucket `hf://buckets/Pier-Jean/jobs-artifacts/`
> holds **69 files and 46.7 GB**, and **nobody has inventoried it since its
> creation on 2026-08-02**. It is the device that `ops/run.py --bucket` exists
> to feed ("without `--bucket`, nothing the job writes survives the
> container"): a search "on the machine" does not see it, by construction.
>
> **Two catches the same day, for two budgets avoided**: the three 14B MMLU
> dumps (579 kB, against a re-budgeted MMLU campaign) and the **sealed 14B
> artifact** itself (`qwen3-14b-c12-3f21abde/qwen3-14b-llvq.bin`,
> 6,506,354,741 bytes, ~9 min of bandwidth, against the **$27.67 and 302 min**
> its quantization cost).
>
> **And the rule is not a guarantee, or it would lie**: the **sealed 8B** was
> searched in both places and it is lost. The machine does not have it, and the
> bucket hosts only its *projections-only* version. `hf buckets ls` changes what
> is known, not what exists.

The three dumps are **now committed** in [`mmlu-dumps/`](mmlu-dumps/)
(`mmlu-14b-{f16,awq,llvq}.csv`), so the loss can no longer repeat. Their
authenticity is established before use: the three stratified micros replay
78.97 / 78.21 / 72.12, and `f16 − LLVQ` replays its four published quantities
(+6.85 [+4.52; +9.12], SE 1.16, McNemar 8.666e-16).

**Consequence: the series "14.45 → 7.49 → 6.09" no longer mixes two species of
number**, all three terms are paired and carry a CI. It loses something else,
and that is more awkward: **the "knee" between 8B and 14B is not resolved ON
MMLU.** The fall of the gap is 6.96 pp from 4B to 8B (SE 1.82, p = 1e-4,
**resolved**) and only 1.40 pp from 8B to 14B (SE 1.68, p = 0.40, **NOT
resolved**; SEs composed in quadrature, *computed*). The sentences in the record
that make the slowdown a result, "there is a knee", "the decay slows", therefore
rest, **on this metric**, on estimated points that the bars do not separate. One
claim stays tested: the gap is much smaller at 14B than at 4B (8.36 pp,
p ≈ 1e-5). And p = 0.40 does not prove equality either: on that step the data
are silent, not conclusive.

> **AMENDED ON 2026-08-17 (evening), AND IT IS THE MOST IMPORTANT AMENDMENT IN
> THIS FILE.** The paragraph above **stays true, for MMLU.** It was then the
> only verdict available, because perplexity had no bar at 4B and the 4B→8B step
> was therefore not testable.
> **It is testable now** ([`ppl-genou.csv`](ppl-genou.csv),
> [`mesures/ppl-appariee-4b-2026-08-17.txt`](../mesures/ppl-appariee-4b-2026-08-17.txt)),
> and it answers the opposite:
>
> | metric | 4B→8B step | 8B→14B step | the slowdown |
> |---|---|---|---|
> | **perplexity** *(paired, 12 windows, same text at all three sizes)* | ×0.881211 [0.856; 0.907] | ×0.974855 [0.959; 0.991] | **RESOLVED**: step1 − step2 = −0.1010 [−0.1377; −0.0643], t = −6.06 |
> | **MMLU gap at 4 bits** *(unpaired across sizes, SEs in quadrature)* | −6.96 pp, p = 1e-4 | −1.40 pp, p = 0.40 | **NOT RESOLVED** on the second step |
>
> **THIS IS NOT A CONTRADICTION, IT IS INFORMATION**: two metrics, two
> verdicts. Perplexity is paired *across sizes* (same window, same text, common
> fingerprint) and therefore tests with far more power; MMLU composes two
> independent campaigns. And the two do not measure the same thing: §3ter of the
> record has established since 2026-08-02 that 2 bits damages **reasoning** far
> more than **recall**, and recall is mostly what a perplexity corpus measures.
>
> **DRAFTING RULE, MANDATORY: every sentence about the knee must NAME ITS
> METRIC.** "The knee holds" bare is half false; "the knee does not hold" bare is
> false on the other half. The right form: *the slowdown is resolved in
> perplexity and is not resolved on the MMLU gap at 4 bits.*

**`jobs.csv` covers FIVE campaigns since 2026-08-17, and the sum of the column
is no longer the figure the paper claims for itself.**

| campaign | rows | sum | in the paper's total? |
|---|---|---|---|
| paper 4B + 8B | up to 2026-08-08 inclusive | $19.82 | yes |
| **kernel** (5-, 6- and 7-arm benchmarks) | marked `[kernel]` | **$2.33** | yes, **since batch D (2026-08-11)** |
| 14B | marked `[14B]` | $30.20 | yes |
| **`[phase 1.2]`** (paired MMLU replay) | marked `[phase 1.2]` | **$1.30** | yes |
| **`[plancher]`** (E1v on card + `nullk`) | marked `[plancher]` | **$1.62** | no, **deliberately** |
| **`[vitesse]`** (throughput batch of 08-17) | marked `[vitesse]` | **$1.59** | no, **deliberately** |

**Why `[vitesse]` and not `[14B]` for the `campagne-14b-vitesse` job.** The tag
names the campaign that pays, not the model measured, which is already the
convention of the `paliers-4b-128` row. Here the choice has an arithmetic
consequence worth stating: the paper cites **two** 14B subtotals ($31.46,
"everything billed under the 14B tag", and $30.20, "the same minus one 4B
measurement"), and filing this job under `[14B]` would have moved both by $0.24,
while it died on a guard without producing a token and no cell of the paper
depends on it. The two 14B subtotals are therefore **unchanged**, verified after
writing.

**The column total goes from $55.59 to $57.21 on 2026-08-17**, settling a debt:
`jobs.csv` stopped at 2026-08-13 and **was missing the two jobs of 08-16**,
`6a814ba31f5885ae605bcb55` (llvq-e1v, l40sx1, 28 min, $0.85) and
`6a81b2b71f5885ae605bdcc9` (llvq-nullk, l40sx1, 26 min, $0.77). Both durations
and both amounts are **read in the header of their own journal**
([`e1v-cuda-2026-08-16`](../mesures/e1v-cuda-2026-08-16.txt),
[`nullk-plancher-2026-08-16`](../mesures/nullk-plancher-2026-08-16.txt)), and
cross-checked by the l40sx1 rate implied by the rows already present
(≈ $0.030/min: 0.0304 and 0.0296 here), *computed*, not a second measurement.

**And the $55.59 did not move for all that: it is now a subtotal.** No cell of
the paper rests on the two jobs of 08-16, since E1v and the `nullk` floor appear
nowhere in it, so folding them into "the cost of this evidence" would have
inflated the figure without adding anything to what it pays for. The paper now
says both: **$58.80 in the register, of which $55.59 stands behind its own
numbers.** The $58.80 includes the landing of the 14B speed job, below. The
$55.59 did not move with it: no cell of the paper rests on that job either.

**The total moved a SECOND time the same day: $57.21 → $57.56**, settling the
**four jobs of 08-17** that `jobs.csv` did not yet have (the file stopped at
08-16):

| job | name | flavor | duration | cost |
|---|---|---|---|---|
| `6a82f40ce55292eada79b526` | campagne-14b-vitesse (failed the shared-memory guard) | l40sx1 | 488 s | $0.24 |
| `6a830ce8cd3824960fcbb26a` | sonde-entrypoint-vllm | cpu-upgrade | not logged | $0.00 |
| `6a8311e8cd3824960fcbb2ff` | sonde-image-llvq | cpu-upgrade | not logged | $0.00 |
| `6a830d53e55292eada79b600` | awq-speed-4b | l40sx1 | 226 s | $0.11 |

Durations and amounts **reported by the job monitor** (*measured* on the
platform side, *cited* here); cross-checked by the l40sx1 rate of $1.80/h,
226 s giving $0.113 and 488 s giving $0.244, *computed*, not a second
measurement. The `billed_min` column carries the rounded minute (4 and 8) and
the exact seconds live in `what`: rounding then re-multiplying by the rate does
not close to the cent, and that is intended. A visible rounded minute beats a
duration invented to the second.

A `jobs.csv` row is laid down when the job has landed, never in anticipation.
The 14B speed job has landed, so it has its row:

| job | name | flavor | duration | cost |
|---|---|---|---|---|
| `6a83121be55292eada79b611` | campagne-14b-vitesse-2 | l40sx1 | 2,472 s | **$1.24** |

Duration and amount **reported by the platform** (*measured* on the platform
side, *cited* here); cross-checked by the l40sx1 rate of $1.80/h, 2,472 s giving
$1.236, *computed*, not a second measurement. `billed_min` carries 41, the
rounded minute; the exact 2,472 s live in `what`, as for the four rows above.

**Arithmetic consequences, all verified after writing**: `[vitesse]`
**$0.35 → $1.59** · column total **$57.56 → $58.80** · the **two 14B subtotals
of the paper stay $31.46 and $30.20**, because the `[vitesse]` tag names the
campaign that pays and not the model measured, the same convention as for
`campagne-14b-vitesse` and `paliers-4b-128`, and here it holds for a job that
did measure 14B. Journal:
[`mesures/fusedrun-14b-2026-08-17.txt`](../mesures/fusedrun-14b-2026-08-17.txt).

**And the raw output of the job is committed**:
[`campagne-14b-vitesse-2/`](campagne-14b-vitesse-2/), the four files the job
wrote into the mounted bucket (`preflight.txt`, `rotbench.txt`,
`fusedrun-q8.txt`, `phases-q8.txt`), taken as they are. That is the lesson paid
twice in the week: a summary journal is an irreversible loss as soon as the
retention channel expires, and the bucket is not a guarantee.

**The total lives in TWO sites, verified on 2026-08-17**: `paper/main.tex`
(abstract, l. 74-75) and `paper/sections/evaluation.tex` ("Cost of evidence").
`sections/intro.tex` mentions the practice, "every claim traces to a dated GPU
job with its billed cost", **without citing an amount**, and
`sections/conclusion.tex` cites none either. The total is regenerated by no
script: `paper/scripts/make_figures.py` never opens this file and
`check_tables.py` does not check that sentence.

So: never re-sum the whole column, and **do not confuse the two Golay70 jobs**.
`e2-golay70-bench` ($0.74, 08-07, the discovery of the negative result) is in
the 19.82; `golay70-v2-sept-bras` ($0.77, 08-11, the repair attempt) is in
the 2.33. The paper adds them to $1.51 **and says so**, because they belong to two
campaigns. The total is regenerated by no script:
`paper/scripts/make_figures.py` never opens this file.

Conventions: VRAM in b/param = whole model including embedding (never payload
alone, see errata-rapport-lot-a); speed ratios = median of the ratios formed
round by round, with range; MMLU micro = the paper's protocol, ± = sampling
error alone; the phases are bounded by synchronization (they attribute, their
sum does not make a tok/s).

## Provenance labels: what is MEASURED, what is COMPUTED

The `vram_bits_per_param` column puts **two different provenances** side by
side. Which row is which is written in the `notes` of each row concerned, and
summarized here:

| quantity | status | how |
|---|---|---|
| **our** b/param (5.162 · 5.322 · 5.106) | **MEASURED** | `rtbits` over the real bytes of the sealed file; the embedding is *modelled* there at 8.5 b/param, a model validated on 08-09 against a real q8 file (`q8b-e8.llvq`, carried measured 8.502) |
| **AWQ** b/param (5.302 · 5.956 · 5.404) | **COMPUTED** | safetensors bytes of the official repo (READ through the Hub API) ÷ `params_total`. The 8B was obtained by the **rate** route (5.956); the **bytes** route gives 5.9566. Indistinguishable at the thousandth, two routes all the same |
| **f16** disks (8.04 · 16.382 GB) | **COMPUTED** | 2 bytes × parameters. No f16 file is weighed |
| **AWQ** disks (2.67 · 6.099 GB) | **READ** at the repo | 2,666,027,672 B and 6,098,581,864 B |
| our disks (1.77 · 1.41 · 4.324 · 3.157 GB) | **READ** on the file | `qwen3-4b-llvq.bin`, `q4b-e8.llvq`, `qwen3-8b-llvq.bin`, `q8b-e8.llvq` |
| `params_total` | **MEASURED** then **CROSS-CHECKED** | read in the sealed file; the 14B is additionally cross-checked by the arithmetic of the architecture (rtbits-14b §3, eight integers) |

**Two values that look like measurements and are not.**

1. **The `15.999` of `tableau-8b.csv` is a ROUNDING ARTIFACT**, not a
   measurement: it is `16.38 GB displayed × 8 ÷ params_total`. The construction
   is worth **exactly 16.000** (two bytes per parameter), and that is what
   `echelle-4b-8b.csv` carries. `check_tables.py` pins the 16.000 so the 15.999
   does not migrate.
2. **The `5.323` and `6.461` of `tableau-8b.csv`** are not `rtbits` verdicts
   either: they are the **engine's VRAM ratios** (5.45 and 6.62 GB on card
   ÷ `params_total`). They cross-check `rtbits` to **0.001**, so they count as a
   **third instrument**, which is valuable, and the figure to publish is still
   the one from `rtbits` (5.322 and 6.462), which is what `echelle-4b-8b.csv`
   carries. Same mechanism as the `5.15` of the 4B, which was the division of
   the "2.60 GB" card display and was withdrawn; the difference is that here the
   two routes agree.

**Check passed on 2026-08-17: no living surface any longer sets a projections
b/weight against a whole-model b/param**, the ban set in advance
(`docs/archive/portage-noyau-cuda.md:31`) and broken once (batch A errata, "GRAVE
error"). A single violation remained, `LAUNCH_ME.md` ("5.51 b/weight, more than
the 4.50 of an ordinary 4 bits"): corrected in place, with a mention of what it
said. The other occurrences of "5.51" and "4.50" are either quotations of the
ban itself (CLAUDE.md, cheatsheet, note-produit, HISTORIQUE), or archive, or, in
`README.md` and `paper/sections/intro.tex`, **licit** b/weight to b/weight
comparisons (5.510 against the 4.179 of the AWQ kernel, `echelle-formats.csv`).
`docs/fiche-4b.md:392` does compare 4.50 to 3.727/4.034, which are **whole
model** b/param (verified: they re-derive from its 70B table).
