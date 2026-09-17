# The Tetra gain rule (2026-09-12)

**Question.** The shipped encoder rounds the block norm to one of two gain levels, where the
euclidean optimum is the projection `t = ‖x‖·cos θ`. Is the difference worth anything?

**Answer.** No, once the shrink it carries is controlled. The projection rule is worth 0.70 pp
of retention, and the shipped rule with its two centroids multiplied by 0.96 is worth 0.69 pp of
the same. Holding the reconstructed norm fixed, the rule change is worth 0.02 pp.

## Setup

- Object: `Tetra`, the production encoder (`llvq_search::tetra::Encoder`), Gaussian blocks
- Single variable: which of the two legal gain levels the encoder writes
- Controls: same blocks, same directions, same 48-bit word, same file. A, B, C and E write the
  same format and differ only in the encoder's decision
- Cost: $0, 4 seconds on 4 cores of the session container, two seeds

## Result

At 20,000 evaluation blocks, 4,000 training blocks, rate 2.000 b/dim (*measured*,
[gainrule-tetra-2026-09-12](mesures/gainrule-tetra-2026-09-12.txt)). `cos θ` between a block and
its Tetra direction is 0.9608 with an sd of 0.0077.

| arm | centroids fitted on | level chosen by | MSE | retention | paired gap against A | mean norm |
|---|---|---|---|---|---|---|
| A, shipped | norms | the norm | 0.085418 | 88.73% | — | 4.8525 |
| B | norms | `t` | 0.084383 | 89.17% | −0.00103 ± 0.00003 | 4.7298 |
| C | projections | `t` | 0.083772 | 89.43% | −0.00165 ± 0.00005 | 4.6588 |
| E | C rescaled to A's mean norm | `t` | 0.085374 | 88.75% | −0.00004 ± 0.00002 | 4.8525 |
| D, bound | free gain | the continuous optimum | 0.076929 | 92.51% | −0.00849 ± 0.00009 | 4.6589 |

The control that reads the result: keep the shipped rule, multiply its two centroids by a
constant `k`, and sweep `k`. The curve peaks at `k = 0.96` and 89.42% of retention, where arm C
reads 89.43%. The best constant shrink and the full rule change land on the same number, and
`0.96` is `cos θ`. Seed 777 replays it: 89.51% against 89.53%.

## Controls

| control | expected | obtained | verdict |
|---|---|---|---|
| A reproduces the retention of the Tetra journals | 88.9 to 89.1% | 88.73% at 20,000 blocks | passes |
| `cos θ` near the 0.96 `lib.rs` records for ball-13 | ≈ 0.96 | 0.9608, sd 0.0077 | passes |
| D above every rule | strictly | 92.51% against 89.43% | passes |
| second seed | same ordering | same, 89.51 against 89.53 | passes |
| clippy on the new example | zero warnings | zero | passes |

## What this does not establish

- The source is Gaussian. A GPTQ residue is not, and a local win has composed worse three times
  in this file (design C, `group_scales`, gptq2). The structural half of the argument does
  transport: the minimizer of `‖x − g·v̂‖²` is `g = t = ‖x‖ cos θ`, so any projection rule is a
  shrink by `cos θ`, and `cos θ` here has an sd of 0.008.
- Nothing here measures perplexity or MMLU. The shrink is the failure mode the spherical
  retraction exists to kill (paper Table 9, 91.90 to 6.90), and
  [mecanismes-perte-qualite](mecanismes-perte-qualite-2026-09-12.md) measures the served 4B
  already attenuating its logits at `beta = 0.466`. Both point the same way and neither is a
  measurement of this arm on a model.
- The conditional metric of [hypothese-metrique-tetra](hypothese-metrique-tetra-2026-09-08.md)
  is a different arm and is untested: it needs an H, which no bench here has. The lesson that
  transports to it is the control, not the verdict. Any gain rule scored without holding the
  reconstructed norm fixed is measuring a shrink.

## Decision

None taken. This closes nothing by itself: a kill is written by the operator on a fundamental
criterion (`METHODE.md` §1), and the euclidean gain arm never had a gate. What it says is that
arm B of the metric audit costs one line of encoder and buys 0.02 pp of retention on a Gaussian
source, and that the 0.70 pp it appears to buy is a scalar shrink available without touching the
rule.

## Provenance

- prereg: none. A $0 bench on a Gaussian source, decisional for nothing, run after
  `docs/mesures/logit-snr-4b-2026-09-12.txt` raised the shrink question
- code: `llvq-bench/examples/gainrule.rs`
- raw: `docs/mesures/gainrule-tetra-2026-09-12.txt`, two seeds in one file
