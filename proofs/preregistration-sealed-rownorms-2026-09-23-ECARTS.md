# Deviations from the sealed-rownorms preregistration

> The preregistration
> [`preregistration-sealed-rownorms-2026-09-23.md`](preregistration-sealed-rownorms-2026-09-23.md)
> is timestamped. It is not edited.

## É1. The training job reads ERROR, and the fold went ahead

**What happened.** Job `6ab3ddb451992417dfcd7bc8` trained its 9,333 steps, closed the
journal and wrote `sigma.json`, then `llvqtune` returned exit code 1:
`__main__.py` returns `0 if outcome.improved else 1`, and `improved` compares the last third
of the logged losses to the first. Here 0.2095 against 0.2031. `train.sh` runs under
`set -e`, so the job ends in ERROR after the file is complete.

**Why it is not a stop.** `domain/loop.py` says of that gauge that it is "only meaningful when
the corpus repeats a fixed batch"; on the streaming corpus two losses differ by the text as
much as by the parameters. The prereg's stop conditions (§4) are the probe band, the inputs'
sha256, the fold's idempotence and counts, and the scoring controls; `improved` is not one
of them. The 14B arm met the same exit on 2026-09-22 and the operator folded it anyway
(`preregistration-dclm-14b-rowscales-2026-09-22-ECARTS.md` §E3). The operator's go of
2026-09-23 covers the chain for this arm.

**What the journal shows** (*measured*, `docs/mesures/sealed-rownorms-2026-09-23-brut/`):
first KL 0.2622 on the card (band [0.25, 0.27], passed); mean loss by eighths 0.2011, 0.2038,
0.2068, 0.2098, 0.2050, 0.2060, 0.2093, 0.2094. The drop from the first batch happens inside
the first eighth; the rest is flat.

## É2. Cost

Training billed 6,188 s, **$3.09**, against ~$4.00 announced. The step count is fixed from
the probe's rate, 0.771 s a step, so 9,333 steps for 7,200 s; the run then went at 0.646 s a
step and trained for 6,032 s. Same tokens as planned, 19,113,984, fewer seconds.
