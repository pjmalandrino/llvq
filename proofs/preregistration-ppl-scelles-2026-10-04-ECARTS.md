# Deviations from the perplexity prereg of 2026-10-04

The prereg is `preregistration-ppl-scelles-2026-10-04.md`, sha256 `e4fa1e5d`, stamped before the
run. It is not edited. Two deviations, both in the reading and neither in what ran.

Job `6ac2b234fbc85ba6823a1fdb`, l40sx1, 11 min, $0.33 against the $1.35 announced. The five
controls of section 3 all pass.

## É1. The 14B landed below its trained base, and the prereg's rule for that is wrong

Section 5 says: "any file below its trained base: a control is wrong, since no step of sealing
improves perplexity; nothing is published until it is found." The 14B read **8.4440 against the
base's 8.4622**, so that row fires on the point estimate.

No control is wrong. The rule assumed a point difference means a difference. Both arms printed a
per-window NLL over the same twelve windows of the same token stream on the same kind of card, so
they pair, and the paired reading was not preregistered:

| quantity | value |
|---|---|
| paired mean ΔNLL, sealed minus base | −0.002151 nats |
| CI95, paired t over 12 windows | [−0.008815, +0.004512] |
| as a perplexity ratio | [−0.878%, +0.452%] |
| windows where the sealed file is worse | 4 of 12 |

The interval contains zero. Sealing the 14B costs nothing this measurement can resolve, and the
premise "no step of sealing improves perplexity" is not contradicted because nothing is resolved.

Consequence for the prereg's own form: a decision rule that compares two point estimates without
naming the interval it reads them against will fire on noise. The three intervals of section 6
were written on the sealed numbers and none on the differences.

## É2. Only the 14B has a device-clean comparison to its base

The prereg assumed each sealed file could be read against its trained base. At 4B and 8B it
cannot, and the prereg did not check where those bases were measured.

| size | base ppl | device of the base | paired to this run? |
|---|---|---|---|
| 4B | 12.3267 | Metal, Mac, and derived from a ratio with no window log on disk | no |
| 8B | 9.5725 | Metal, Mac (`dclm-8b-rowscales-2026-09-21-brut/ppl-ft-f16.txt`) | no |
| 14B | 8.4622 | Cuda, L40S (`dclm-14b-rowscales-2026-09-22-brut/ppl-ft-f16.txt`) | yes |

A paired reading at 8B does exist arithmetically, +1.396% [+1.054%, +1.740%] with 12 of 12
windows worse, and it is **not published as the cost of sealing**: it crosses Metal and Cuda, so
it mixes sealing with a change of arithmetic. The 4B has no window log at all.

What this costs: the cost of sealing is measured at one size out of three, and at that size it is
not resolved. Closing it needs the three sealed files read on Metal, which pairs them with the
Metal bases, for $0 on the Mac plus the download of 9.3 GB from the bucket.

## What is published

The three perplexities and their ratios to f16 and to AWQ, with the three predictions marked
inside their intervals. The cost of sealing is published for the 14B only, as unresolved.
