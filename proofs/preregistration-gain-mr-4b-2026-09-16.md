# Prereg gain-mr: fitting the two stored gain levels against the dense model (2026-09-16)

Status: **DRAFT, not timestamped.** Nothing measured under it yet.
Cost: stages 1 and 2 are $0 on the Mac; stage 3 is paid and carries its own go.
Wave budget left: not established — see `budget-hf-plafonne`.
Measured code: to be named at stamping; the binary does not exist yet.

A timestamped prereg is no longer edited. A deviation goes in `<name>-ECARTS.md`.

## Question

Can the gain parameters the Tetra format **already stores** — exactly two
centroids per matrix — be moved so the model's output distribution gets closer
to the dense checkpoint's, and does that improve decisions rather than only the
divergence?

It has no answer today because every fit so far used a different objective. The
map of 2026-09-15 minimized the model's own NLL: it won 17.4 % of perplexity
and lost 3.06 pp of MMLU. Stage 0 of 2026-09-16 removed the obvious reason to
expect the same from a KL objective — a single global temperature is worth
2.49 % of the gap on held-out wikitext and 0.93 % on C4, so the objective is
not mostly scale (`docs/mesures/kl-temperature-4b-2026-09-16.txt`).

It decides whether a distillation pilot on this family is worth building
further, and whether the **second** gain parameter earns its plumbing.

## Setup

Three stages. Stage 3 does not start without a separate operator go.

### Stage 1 — earn the gate before using it ($0)

Stage 2 needs a free quantity that can refuse a candidate without a paid MMLU.
Two are available: KL to the dense model, and top-1 agreement with it. Neither
is entitled to that role. So stage 1 points both of them at an arm **already
known to be bad** — the gain-rescaled artifact that lost 3.06 pp — and asks
whether they saw it.

```text
LLVQ_MODEL=Qwen/Qwen3-4B LLVQ_KL_ARTIFACT=~/llvq-4b-corrige.llvq \
  cargo run --release -p llvq-llm --features metal --bin kltemp -- 4 4 1024 metal
LLVQ_KL_VAL=c4 (same, second corpus)
```

Against the reference numbers already measured on `~/llvq-4b-tetra.llvq`.
One variable: which artifact. Everything else is the stage 0 command verbatim.

### Stage 2 — the fit ($0, Mac, overnight)

For each of the 252 matrices, two directions on the stored centroids:

```text
m : both centroids multiplied together      (the direction the 2026-09-15 map used)
r : the ratio between the two levels        (c0 /= sqrt(r), c1 *= sqrt(r))
```

At `(m, r) = (1, 1)` the artifact is unchanged. Central differences of
`eps = 0.04` on `log m` and `log r` give a diagonal quadratic surrogate
(`llvq_llm::errmodel`), and its trust-region optimum at `|delta| <= 0.04` is the
candidate. **Four evaluations per matrix serve both arms**, since the `m`
probes are shared:

| arm | parameters | per matrix |
|---|---|---|
| reference | none | the artifact as shipped |
| M | `m` only | 1 |
| MR | `m` and `r` | 2 |

Probes read 4 x 1024 calibration windows past 131,072 tokens of wikitext-2
train. Evaluation reads 4 x 1024 disjoint windows of the same text and 4 x 1024
of C4. f32, Metal. The reference artifact is
`~/llvq-4b-tetra.llvq`, sha256 `eadc9ef3c3f6478cf58751865069f313a757a62bf68d1e9f97c1bc61c7da409d`;
stage 1 also reads `~/llvq-4b-corrige.llvq`, sha256
`aec6761635c6dfb19f0e0cd2c0368f144bf0b820df150036599e350682d0fec3`.

Before the full pass: the per-probe rate is measured on the first 5 matrices and
reported. If the projection exceeds 12 h of Mac, the pass does not start and the
plan is revised.

### Stage 3 — MMLU ($, separate go)

Only if stage 2 passes its gate. Paired MMLU against the reference, dumps kept.

## Controls

If one fails, no number from that stage gets published.

1. **The probe applies what the fit will apply.** `errmap` probed by scaling the
   reconstructed tensor, which moves the tail; `artscale` moves centroids, which
   does not. Every one of the 252 records carries a tail (*measured*, `artstat`),
   so the two are different operations. This pilot probes centroids, tail
   untouched, and proves it: writing the artifact at `(m, r) = (1, 1)` must
   produce a **byte-identical** file.
2. **The surrogate is checked against the thing it predicts.** The applied
   combination's measured loss goes beside the surrogate's prediction
   (`errmodel::Residual`). A measured loss outside the prediction means the
   diagonal model is insufficient, and it is reported, not smoothed.
3. **q/k invariance is re-tested, not inherited.** Under a common multiplier,
   `q_proj` and `k_proj` are scale-invariant through QK-norm, and `errmap`
   dropped them for that reason. The `r` direction changes a row's **direction**,
   not only its length, so the invariance is not expected to hold. All 252
   matrices are probed, and the verdict is read off `g_r`, `h_r` rather than
   assumed either way.
4. **A KL gain that is only a rescaling stays visible.** KL is reported at
   `T = 1` and at the fitted temperature, before and after. Stage 0 bounds this
   at 2.49 %, but the fit could create scale that was not there.
5. **No window fits and scores.** Probe, validation and C4 windows are disjoint;
   a map fitted on its own evaluation set measured a transfer of -0.008 once
   already.
6. **The harness refuses to contradict itself.** `kltemp` aborts if a positive
   temperature moves any argmax.

## What gets published, and what does not get compared

Published: held-out KL on both corpora, top-1 agreement with the dense model,
perplexity, how many matrices moved in `r` and by how much, the q/k verdict, the
surrogate residual, and the Mac hours.

Not compared: KL against perplexity as though they rank the same thing — they
demonstrably do not. And arm MR against the map of 2026-09-15: that map scaled
the tail as well, so it is a different operation on a different object, not a
weaker version of this one.

## Decision rule

Let **A** be the top-1 agreement drop stage 1 measures on the known-bad
corrected arm. The rows are read on held-out wikitext, with C4 as the tiebreak.

| stage 1 | stage 2 result | action |
|---|---|---|
| A >= 0.5 pp | KL falls, agreement loses less than A/3 | candidate stands; propose stage 3, operator decides |
| A >= 0.5 pp | KL falls, agreement loses A/3 or more | same failure mode as 2026-09-15; stop, and record that KL is not a safe objective for this family |
| A >= 0.5 pp | KL does not fall on held-out | the fit failed; stop |
| A < 0.5 pp | any | agreement is NOT a usable gate. Stage 2 still runs and reports, but carries no free gate: nothing is concluded without paid MMLU, which is then the operator's call |
| otherwise | — | not settled, operator decision |

Separately, and not a gate: if arm MR's KL reduction is within 10 % of arm M's,
the ratio buys nothing and the second gain parameter is not worth plumbing into
anything downstream.

Nothing here adopts anything. Adoption is a quality-axis decision on MMLU, and
it is the operator's.

## Signed prediction

**Stage 1.** The corrected arm shows **higher** KL to the dense model than the
uncorrected one, and top-1 agreement **drops by 0.5 to 3 pp**. Reasoning: it
lost 3.08 pp of MMLU accuracy through a mechanism — the letter prior gaining
relative weight as the evidence shrank — that should also push prose
predictions toward high-prior tokens; and stage 0 showed the uncorrected arm is
already slightly over-confident (T* = 1.07), so a further 32.5 % collapse
overshoots.

Known flaw, and it is the one I got wrong a week ago: the 32.5 % was measured on
four MMLU option logits, not on prose, and a four-way choice with a letter prior
has structure that next-token prediction does not. Agreement may be flat. If it
is, the fourth row of the table fires and this pilot cannot conclude for free.

**Stage 2.** Arm MR reduces held-out KL by **3 to 15 %** relative, and beats arm
M by a factor **1.3 to 3**. Reasoning: `r` is a direction the 2026-09-15 family
could not reach at all, and both gain levels are populated — 52.9 % of
150,681,600 blocks sit on level 1 — so the knob is live rather than degenerate.

Known flaw: the two centroids may be close enough in value that `r` is nearly
collinear with `m`, which nobody has measured; and the diagonal surrogate
ignores cross terms, which on 252 simultaneous moves is an assumption, not a
theorem. Control 2 exists to catch exactly that.

**Top-1 agreement after the fit: within +/- 1 pp of the reference.** I have no
strong reason for this one, and say so — it is the experiment's actual question,
and a prediction I would defend would mean the experiment was not worth running.
