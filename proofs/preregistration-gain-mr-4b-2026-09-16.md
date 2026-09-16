# Prereg gain-mr: fitting the two stored gain levels against the dense model (2026-09-16)

Status: **DRAFT, not timestamped.** Nothing measured under it yet.
Cost: stages 1 and 2 are $0 on the Mac, ceiling **3 h**; stage 3 is paid and
carries its own go. Wave budget left: not established — see `budget-hf-plafonne`.
Measured code: to be named at stamping; the binary does not exist yet.

A timestamped prereg is no longer edited. A deviation goes in `<name>-ECARTS.md`.

## Question

Can the gain parameters the Tetra format **already stores** — exactly two
centroids per matrix — be moved so the model's output distribution gets closer
to the dense checkpoint's? And does the second degree of freedom, the ratio
between the two levels, earn its plumbing over the first?

The pilot decides **whether M or MR deserves a paid functional confirmation**.
It does not try to replace that confirmation with a free gauge. Nothing here
claims an MMLU gain, and no free quantity is given the authority to.

It has no answer today because every fit so far used a different objective. The
map of 2026-09-15 minimized the model's own NLL: it won 17.4 % of perplexity and
lost 3.06 pp of MMLU. Stage 0 removed the obvious reason to expect the same from
a KL objective — a single global temperature is worth 2.49 % of the gap on
held-out wikitext and 0.93 % on C4
(`docs/mesures/kl-temperature-4b-2026-09-16.txt`).

## The parameterization, exactly

For a record's two positive centroids `c0 < c1`, with `m > 0` and `r > 0`:

```text
c0' = m · c0 / sqrt(r)
c1' = m · c1 · sqrt(r)
```

`m` moves their common scale, `r` their ratio: `c1'/c0' = r · c1/c0`, and the
geometric mean is `m` times the original. At `(m, r) = (1, 1)` both are
unchanged.

Constraints: `m, r > 0`, and `|log m| <= 0.04`, `|log r| <= 0.04` — the trust
half-width the 2026-09-15 map was read at. Positivity of both levels is
preserved for any admissible `(m, r)`.

**Nothing else moves.** Lattice indices, gain bits, row scales, rotation seeds,
shell caps and tails are copied through byte for byte. The rate is untouched:
the correction is 2 numbers per matrix that the file already holds.

Special values, read off the file rather than assumed (*measured*, `artstat` on
`llvq-4b-tetra.llvq`): **0 records with `c0 = 0`, 0 with `c0 = c1`, 0 with a
negative level**. So no record degenerates under this map, and no special case
is needed. If a future artifact carries one, the writer refuses rather than
silently making `r` a no-op on it.

Worth recording, and not acted on here: `c1/c0` is nearly constant across the
model — min 1.2596, median 1.2691, max 2.0106 over 252 matrices. The fitted
ratio is close to a property of the format rather than of each matrix. If the
optimal `r` turns out equally uniform, one global number would do the work of
252, and that is a cheaper pilot than this one — a lead, not a claim.

## Setup

Three stages. Stage 3 does not start without a separate operator go.

### Stage 1 — a descriptive control on a known-bad arm ($0)

Point KL and top-1 agreement at the gain-rescaled artifact that lost 3.06 pp of
MMLU, and record what they read.

```text
LLVQ_MODEL=Qwen/Qwen3-4B LLVQ_KL_ARTIFACT=~/llvq-4b-corrige.llvq \
  cargo run --release -p llvq-llm --features metal --bin kltemp -- 4 4 1024 metal
LLVQ_KL_VAL=c4 (same, second corpus)
```

**This is description, not calibration.** An earlier draft used it to set a
tolerance; that is withdrawn. A threshold built on one degradation, measured
once, would be a number with no standing. An indicator can catch this failure
and miss the next; and failing to catch this one would not make KL useless,
since KL measures fidelity to the dense model and not accuracy. So stage 1
informs the reading of stage 2 and gates nothing.

### Stage 2 — the fit ($0, Mac)

Per matrix, central differences of `eps = 0.04` on `log m` and `log r`, four
evaluations, shared between the two arms:

| arm | parameters | per matrix |
|---|---|---|
| reference | none | the artifact as shipped |
| M | `m` only | 1 |
| MR | `m` and `r` | 2 |

**Three splits, and they do different jobs.** All start past 131,072 tokens of
wikitext-2 train, so none overlaps the windows GPTQ calibrated on.

| split | windows | what it is allowed to do |
|---|---|---|
| probe | 4 × 1024 wikitext-2 | fit the surrogate |
| selection | 4 × 1024 wikitext-2, disjoint | choose M against MR, and the amplitude |
| reserved | 4 × 1024 wikitext-2 + 4 × 1024 C4 | report the retained arm, once |

The reserved split is read **after** the arm is chosen, and its number is the
one published. Anything the selection split decides is tuning.

f32, Metal. Reference artifact `~/llvq-4b-tetra.llvq`, sha256
`eadc9ef3c3f6478cf58751865069f313a757a62bf68d1e9f97c1bc61c7da409d`; stage 1 also
reads `~/llvq-4b-corrige.llvq`, sha256
`aec6761635c6dfb19f0e0cd2c0368f144bf0b820df150036599e350682d0fec3`.

Teacher distributions for all 16 windows are computed **before** the artifact is
loaded over the projections — one model object is resident at a time, and the
dense side cannot be recomputed afterwards. Peak resident is expected near
27 GB: 16 GB of model, 9.9 GB of stored teacher probabilities, the rest
transient.

### Stage 3 — MMLU ($, separate go)

Paired MMLU against the reference, dumps kept per question.

## The approximation, named

Four probes give the two first derivatives and the two pure second derivatives.
They do **not** give the interaction `d²L/dm dr`. The surrogate is therefore a
**diagonal quadratic in (m, r)**, and it additionally assumes matrices perturb
the loss independently — 252 simultaneous moves, no cross terms of any kind.

Both assumptions are assumptions, not theorems. So the combined correction's
**real** loss is measured on the selection split before any arm is retained, and
the surrogate's prediction is published beside it. A measured loss outside the
prediction is the size of what the diagonal model cannot see, and it is reported
rather than smoothed.

## Controls

If one fails, no number from that stage gets published.

1. **The probe applies what the fit will apply.** `errmap` probed by scaling the
   reconstructed tensor, which moves the tail; centroids do not. All 252 records
   carry a tail (*measured*, `artstat`), so those are different operations. This
   pilot probes centroids.
2. **Null identity.** Writing the artifact at `(m, r) = (1, 1)` produces a
   **byte-identical** file.
3. **Non-null decode.** For a non-trivial `(m, r)`, the written artifact is
   reloaded and 1,000 blocks drawn across 10 matrices are checked against
   `m · c_g^(r) · row_scale · u` computed independently. Byte-identity at the
   null point cannot catch a writer/reader disagreement that only shows when
   something actually moved.
4. **q/k invariance is re-tested, not inherited.** Under a common multiplier
   they are scale-invariant through QK-norm, which is why `errmap` dropped them.
   `r` changes a row's direction, not only its length. All 252 are probed and
   the verdict is read off `g_r`, `h_r`.
5. **A KL gain that is only a rescaling stays visible.** KL is reported at
   `T = 1` and at the fitted temperature, before and after.
6. **The harness refuses to contradict itself.** `kltemp` aborts if a positive
   temperature moves any argmax.

## What gets published, and what does not get compared

Published: KL on the reserved split for the retained arm, and on the selection
split for both arms; top-1 agreement with the dense model; perplexity; how many
matrices moved in `r` and by how much; the q/k verdict; the surrogate residual;
peak memory and Mac hours.

Not compared: KL against perplexity as though they rank the same thing — they
demonstrably do not. And arm MR against the map of 2026-09-15, which scaled the
tail as well and is a different operation on a different object.

## Decision rule

Read on the **selection** split for the choice, then reported once on the
**reserved** split.

| selection result | action |
|---|---|
| MR's KL reduction beats M's by more than 10 % relative | MR is the arm; measure it on reserved |
| both reduce KL, MR within 10 % of M | M is the arm, for parsimony; the ratio did not earn its plumbing |
| only one arm reduces KL | that arm; the other is dropped |
| neither reduces KL on selection | the fit failed; stop, nothing goes to reserved |
| otherwise | not settled, operator decision |

Then, on reserved:

| reserved result | action |
|---|---|
| KL falls out of calibration | the arm is a **candidate for functional evaluation**; the operator decides whether to spend stage 3 |
| KL does not fall | the gain did not survive reservation; stop |

**No row of either table claims an accuracy gain.** A KL improvement out of
calibration authorizes a functional evaluation and nothing more — not even if
top-1 agreement improves, which is reported as description. Adoption is a
quality-axis decision on MMLU, and it is the operator's.

## Budget

Ceiling **3 h of Mac, $0**. The estimate covers the whole stage, not the probes
alone:

| item | estimate |
|---|---|
| teacher pass, 16 windows | 25 s |
| reference load and evaluation | 50 s |
| 1,008 probes at ~5 s | 84 min |
| writing the M and MR artifacts | 2 min |
| loading and evaluating both arms on selection and reserved | 4 min |
| stage 1, two `kltemp` runs | 4 min |
| **total** | **~1 h 40, slack to 3 h** |

The 5 s per probe comes from `kltemp`'s measured 9 s for 8 windows of 1024
(*measured*, `docs/mesures/kl-temperature-4b-2026-09-16.txt`), halved for 4
windows. It is an extrapolation until the run confirms it.

**The rate is confirmed on 12 matrices spread through the model** — every 21st
target, so all 7 projection types and blocks 0 through 35 are represented — and
reported before the full pass starts. The first 12 would all sit in block 0 and
say nothing about the rest. If the projection exceeds the 3 h ceiling, the pass
does not start and the plan comes back for revision.

## Signed prediction

**Stage 1.** The corrected arm shows **higher** KL to the dense model than the
uncorrected one. Reasoning: stage 0 found the uncorrected arm already slightly
over-confident (T* = 1.07), and the corrected arm collapses its option logits a
further 32.5 %, which overshoots.

Known flaw, and it is the one I got wrong a week ago: that 32.5 % was measured on
four MMLU option logits, not on prose. The magnitude does not transfer, and the
sign may not either.

On top-1 agreement in stage 1: **no defensible quantitative prediction.** The
mechanism behind the MMLU loss — a letter prior gaining relative weight as the
evidence shrank — has no counterpart in next-token prediction on prose, so I
have no basis for a number and will not invent an interval.

**Stage 2.** Arm MR reduces KL on the selection split by **3 to 15 %** relative.
Reasoning: `r` is a direction the 2026-09-15 family could not reach, both gain
levels are populated (52.9 % of 150,681,600 blocks on level 1), and the levels
are separated by about 27 % so the knob is not degenerate.

Known flaw: the diagonal surrogate ignores every cross term, and on 252
simultaneous moves that is an assumption. The residual control exists to size it.

On MR against M, and on top-1 agreement after the fit: **no defensible
quantitative prediction.** Nothing measured so far constrains either. That is
the experiment's question, and an interval I could not defend would be worse
than saying so.
