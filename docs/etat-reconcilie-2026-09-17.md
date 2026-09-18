# Reconciled state, 2026-09-10 to 2026-09-17

This Mac holds the current work. HEAD is `eab780c` of 2026-09-17, the tree is clean, and the
branch `claude/tetra-quality-map-plan-t8ojq2` sits 57 commits ahead of `main`. A container
consulted a HEAD of 2026-09-11 and concluded that several experiments were missing. They are
not missing. They are in these 57 commits, with their journals, their preregs and their dumps.

Scope: `main..HEAD`, which is 2026-09-10 to 2026-09-17. Operator decision of 2026-09-17: no
merge into `main`, and the 38 leads of `claude/fine-tuning-cost-mmlu-gain-kic9ua` stay out of
this inventory. This document is written before the next experiment, so that no measurement
repeats a settled one or compares two different bases without saying so.

Cost of producing it: $0, no card, no job, no run. Reading, arithmetic and `artstat`.

## 1. Branches

| branch | tip | state |
|---|---|---|
| `claude/tetra-quality-map-plan-t8ojq2` | `eab780c` | current, pushed, 57 commits ahead of `main` |
| `claude/fine-tuning-cost-mmlu-gain-kic9ua` | `803d8d8` | merged into the current branch |
| `origin/claude/analyse-comportements-couteux-85ypbr` | `5728820` | merged into the current branch |
| `origin/claude/llvq-concepts-presentation-j9o1ts` | `36238df` | merged into the current branch |
| `recherche/tetra-schur-2026-09-14` | `09e0f65` | merged into the current branch |
| `main` | `7d62cff` | 2026-09-11, deliberately behind |

No stash. The worktree `scratchpad/enc` was dead and is pruned; its commit `07eb114` is
reachable from four branches, so nothing was lost. One copy of the repository exists on this
Mac (*measured*, `mdfind`, 2026-09-17), not two.

## 2. The four rho

Four distinct quantities are called rho in this repository. Confusing two of them would make
the next factorial design meaningless, so each is given with its site and its served value.

| name | definition | site | served value |
|---|---|---|---|
| rho_H | `H[i][j] *= rho` for `j != i`, diagonal left exact | `calib.rs:893`, applied `calib.rs:1052` and `calib.rs:827` | 1.0 |
| rho_gain | `<w_i, r_i> / <r_i, r_i>`, multiplies `row_scales` | `llvq-bench/examples/rhoapply.rs` | 0.929433 on the served file, 0.929234 on bare Tetra |
| `gain_scale` | multiplies the fitted gain centroids of a matrix | `LLVQ_GAIN_SCALE`, `calib.rs` `RunConfig` | 1.0 |
| rho_rank | index of a value in the progression `o + 4Z` | `llvq-bench/src/f1/rank.rs:119` | naming collision, unrelated |

Three facts about rho_H decide the design of any sweep over it (*measured* in source,
2026-09-17).

1. It is applied to the estimate, in the natural basis, before the rotation. The code says
   why: shrinking after would target `diag(Q H Q')`, which a Hadamard flattens, and that is a
   large relative damping under another name.
2. Selection and compensation are inseparable under it. One `GptqFactor` per block and
   activation is built from the shrunk, rotated `H`, and the loop takes both from that single
   factor.
3. Row scales do not move with it. `row_scales` is fixed before the block loop from the
   original weights by `quantizer::row_scale`, which never reads `H` (`gptq.rs:246`).

In the setup line of a prereg, "rho = 1" means rho_H. The negative conditional rollout of
2026-09-16 ran at rho_H = 1.0. That value is not a knob of that experiment: `tetraalt.rs`
contains no rho at all, and the value is inherited from the Schur pilot bundles, which record
`"h_shrink": 1.0`.

## 3. The int4 budget, redone in kernel b/weight

Every projection type fits under b_max natively, and the "over budget" verdicts of
2026-09-16 were a unit error. The allocation journal compared whole-model b/param against a
b_max defined in kernel b/weight. The o_proj journal of 2026-09-17 caught this for its own
arm (deviation E4) and redid one row. The other five rows were never redone.

Accounting (*computed*): 3,633,315,840 projection weights, `Tetra` 2.1498, int4 g128 4.250,
f16 16.000. The weight count reproduces from the architecture, and the four figures the
o_proj journal published (2.2044, 2.4226, 2.5095, 3.9485) reproduce to the fourth decimal.

| arm | native int4 | dequantized to f16 |
|---|---|---|
| served Q5, v_proj alone | 2.2044 | 2.5095 |
| v + k_proj | 2.2589 | 2.8693 |
| v + q_proj | 2.4226 | 3.9485 |
| v + o_proj | 2.4226 | 3.9485 |
| v + gate_proj | 2.7226 | 5.9271 |
| v + up_proj | 2.7226 | 5.9271 |
| v + down_proj | 2.7226 | 5.9271 |
| v + o + down | 2.9408 | 7.3661 |

What this does not establish. The condition is a native int4 kernel for the shape in
question. That kernel has run on v_proj's 1024 x 2560 and on no other shape. The quality of
all six arms was measured by dense reconstruction of a dequantized int4 tensor, which is the
right way to price four bits of information and is not the served path. So `down_proj` is a
candidate again on the memory axis, not a decided arm.

## 4. Measured

| result | figure | journal |
|---|---|---|
| o_proj at int4, confirmed | +1.55 pp, CI95 [+0.91; +2.20], 11,762 held-out questions, McNemar p = 1.5e-6 | [oproj-int4-full-2026-09-17](mesures/oproj-int4-full-2026-09-17.txt) |
| v + o at int4, census | 58.07 micro, against 56.37 for the shipped file | same |
| the earlier 58s, paired | +0.16 pp [-2.97; +2.73] against T3, not resolved | same |
| six types at int4, exploration | down_proj +3.79, o_proj +2.77, q_proj +1.81, k_proj +1.30, up_proj +1.88, gate_proj +0.97; individual intervals, no multiplicity correction | [q5-alloc-int4-2026-09-16](mesures/q5-alloc-int4-2026-09-16.txt) |
| conditional selector, fixed state | +2.393 % of conditional cost at K = 6, 29.96 % of blocks changed | [tetra-directions-2026-09-16](mesures/tetra-directions-2026-09-16.txt) |
| conditional selector, chained | +0.911 % on the loop's own objective, **-8.763 %** on reserved activations | same |
| sensitivity map at the 4B | -14.93 % of perplexity at T = 0.06, 15.7575 to 13.4044 | [errmap-4b-2026-09-15](mesures/errmap-4b-2026-09-15.txt) |
| that map scored on MMLU | -3.06 pp micro, 17.7 % discordant, p = 6.2e-18, on 14,042 questions | [errmap-mmlu-4b-2026-09-16](mesures/errmap-mmlu-4b-2026-09-16.txt) |
| one output temperature | T* = 1.07216 removes 2.49 % of KL on held-out wikitext, 0.93 % on C4 | [kl-temperature-4b-2026-09-16](mesures/kl-temperature-4b-2026-09-16.txt) |
| Tetra gain rule, synthetic | MSE 0.085418 to 0.083772 for the projection rule, free gain bound 0.076929 | [gainrule-tetra-2026-09-12](mesures/gainrule-tetra-2026-09-12.txt) |
| the Euclidean gain rule, end to end | costs 4.0 % of perplexity at the 0.6B | [tetrapost-ppl-0.6b-2026-09-15](mesures/tetrapost-ppl-0.6b-2026-09-15.txt) |
| centroid scale sweep | shallow minimum at 1.02, -0.836 % of perplexity, then a cliff | [gain-scale-0.6b-2026-09-15](mesures/gain-scale-0.6b-2026-09-15.txt) |
| logit fidelity | beta 0.498 on the served file against 0.960 for AWQ, noise share 44.1 % | [logit-snr-4b-2026-09-12](mesures/logit-snr-4b-2026-09-12.txt) |
| rho_gain on the served object | +1.23 pp [-0.50; +3.06], and +0.13 pp at p = 0.8944 unweighted | [l01-bras-servi-2026-09-13](mesures/l01-bras-servi-2026-09-13.txt) |
| T3 against the served object | +1.52 pp, not resolved on the published statistic | [t3-genou-2026-09-12](mesures/t3-genou-2026-09-12.txt) |

## 5. Implemented, not measured

| item | state | what is missing |
|---|---|---|
| (m, r) gain pilot | prereg `preregistration-gain-mr-4b-2026-09-16.md` written and revised | no `.ots`, no timestamp, no go, no announced cost |
| rho_H at the 4B | `LLVQ_H_SHRINK` shipped, swept at the 0.6B only | the 4B run. The served 4B `Tetra` was encoded at rho_H = 1 |
| L36 capture pass | capture built, ten tests, five mutants dead | the reduce. `hstats` serves three of eight rows; the five needing `ΔW` have no tool |

## 6. Hypothesis, with no artifact in the repository

`tetrahist` has zero occurrences in any `.rs`, `.md`, `.txt` or `.py` file (*measured*,
2026-09-17). Three independent E8 codebooks do not exist here either, and the
only E8 figure in the record is published, QuIP# E8P12 at 21.15 perplexity and 48.6 MMLU. A
multi-shell Leech comparison has parts: `llvq-core` carries the shells, and `leech1c12` and
`leech0c13` were measured. Q6a is row 17 of `ROADMAP-QUALITY`, at +2 to +5 pp *estimated* for
about 18 M free parameters and zero bits, and no training loop exists in this repository.

## 7. Closed or paused in this window

| lead | verdict |
|---|---|
| K-best beam, row 12 | paused, not refuted. One selector failed on 48 rows of one pilot |
| gain centroid rescaling, the operator's PGMR | closed by the MMLU loss; the T sweep was cancelled before billing |
| Euclidean gain rule | closed, 4 % of perplexity |
| centroid scalar | not a small lever, not a lever |
| Tetra gain rule as a served change | +0.02 pp once its shrink is held fixed |

PGMR is the operator's name for the gain centroid rescaling axis (operator, 2026-09-18). It
appears under that name in no file, which is why an earlier reading filed it as a lead with no
artifact. It has two members, and both are closed. The per-matrix map bought 17.4 % of
perplexity at the 4B and lost 3.06 pp of MMLU on 14,042 questions. The single global scalar
bought 0.836 % of perplexity at the 0.6B, and applied post hoc it was worse than the sweep
that found it. The axis is the sharpest case in this repository of a better proxy composing
worse, and the two metrics move in opposite directions with both resolved.

## 8. The two references, by fingerprint

| file | sha256 | bytes | content |
|---|---|---|---|
| `~/llvq-4b-tetra.llvq` | `eadc9ef3c3f6478c` | 980,791,242 | bare `Tetra`, 252 lattice records, no correction |
| `~/llvq-4b-corrige.llvq` | `aec6761635c6dfb1` | 980,791,234 | the same codes with the map's centroids at T = 0.06 |

Both carry 252 lattice records and 0 int4 (*measured*, `artstat`). Neither is the served Q5
object, which holds 216 lattice records and 36 int4. No mixed file exists on this Mac: the
four other local artifacts are 252-lattice files too. The served Q5 lives where the card runs
read it, and the diagnostic reference for the codebook is therefore the bare `Tetra` file.
The eight-byte size difference is trailing slack, and it is worth knowing before `artscale`
is used as an instrument. The bare file ends with eight zero bytes after its last record; the
corrected file ends at the record (*measured*, `cmp` and `xxd`, 2026-09-17). The forty bytes
before that point are identical in both. `artscale` rewrites a header and then walks records
(`artscale.rs:96-119`), so it emits no trailing slack. A byte-identical round trip through it
is therefore impossible on this file, and L37's record transplant needs its own idempotence
control before it relies on that path.

## 9. Data provenance

`jobs.csv` is current: the three billed jobs of 2026-09-16 and 2026-09-17 carry their id,
flavor, duration, cost and journal. The convention of `docs/data/README.md` is to declare
directories rather than individual dumps, so the eleven dumps written in this window need no
per-file row. What they do need is their sampling plan, because `mmlupair` refuses two plans
and a wrong assumption there costs a full run. That table is added to `docs/data/README.md`
with this document.

Ten files of `docs/data/` predate this window and are declared nowhere:
`attribution-slot32.csv`, `awq-speed-4b-2026-08-17.json`, `bruit-mmlu-graines`,
`echelle-formats-a100.csv`, `f5-nll`, `knee-seeds.csv`, `m2-attribution`, `m2b-graine3`,
`m2rep-graine3`, `moe-routing-gptoss20b-2026-08-12.json`. They are outside this scope and
recorded here as debt.

## 10. What this document does not establish

It reads code, journals and file headers. It re-measures nothing about the model. The one new
number it carries is the budget table of section 3, labelled *computed*, and the arithmetic
that produced it is validated only by reproducing four figures of an existing journal.
