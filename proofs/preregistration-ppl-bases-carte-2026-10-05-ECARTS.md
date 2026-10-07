# Deviations from the card-route prereg of 2026-10-05

The prereg is `preregistration-ppl-bases-carte-2026-10-05.md`, sha256 `f1abf54c`, stamped before
the run. It is not edited. One deviation, in the attribution and not in what ran.

Job `6ac355cf404719ba376533cb`, l40sx1, 6 min, $0.17 against the $0.25 announced. The five
controls all pass, including control 3, whose full digests came back `f8c1c903b753fe34...` and
`783cef6700e0efa9...`, so the bucket directories held the trained files and not their untrained
twins of identical size.

## É1. The decision rule credits the embedding alone, and the measurement cannot

Row 3 of section 5 reads: "the paired sealing cost excludes zero at both sizes: **the int4
embedding** has a measured perplexity cost on a served object, at two sizes." It excludes zero at
both sizes. The attribution is wrong.

Sealing changed each file in **two** ways, not one. It wrote the embedding as int4 g64, and it
moved matrices into int4. The counts, read off the files:

| size | base, int4 records | sealed, int4 records | source |
|---|---|---|---|
| 4B | 36 (*presumed* `v_proj`, no `rtbits` dump of this file exists) | **84** | `artstat` and `rtbits` on the local copy, 2026-10-05 |
| 8B | 36, `v_proj` | 53 (`v_proj` plus `down_proj`@10-26) | pre-seal `rtbits`, dclm-8b-2026-09-21-brut; sealed-8b-27-2026-09-24 |

So +2.084% at 4B and +1.396% at 8B are the cost of **sealing**, which is what section 4 says is
published, and not the cost of the embedding, which is what the decision rule names. The two
components are not separable by this run and no number here attributes between them.

Separating them is cheap and not done: one `LLVQ_EMBED` arm on an otherwise unchanged sealed file
would isolate the embedding, since `embedq` rewrites a carried tensor without touching a matrix
record.

## What stands

Row 1 fires, and with ten times the margin it asked for. The paired device effect at 8B is
+0.000% over twelve windows, CI95 [−0.009%, +0.010%], against the ±0.12% the row named. The
cross-device perplexity comparisons already published in this repository stand, and the 4B's
ratios to f16 and AWQ in `ppl-scelles-2026-10-04` need no correction.

Row 3's interval reading stands: the cost of sealing excludes zero at 4B and 8B, 12 of 12 windows
worse in both cases. Only its causal clause is struck.
