# Departures from the prereg inkling2-mini-quality-gate (2026-11-21)

The prereg `preregistration-inkling2-mini-quality-gate-2026-11-21.md` (sha256 e947b7c8, stamped 22:01) and its annex
on the encoder (sha256 2f5a02d2, stamped 22:41) are not edited. This file records what was done that they do not
provide for. The journal is `docs/mesures/inkling2-mini-quality-gate-2026-11-21.txt`.

## E1. The layer manifest was completed after a first launch that encoded nothing

`ops/inkling/prep_layer.py` wrote a manifest without the keys `extq` reads. The first launch of layer 6 stopped after
one minute on "no per-expert token counts", with no expert encoded and nothing measured. The manifest now carries
`num_experts` and `experts: [{tokens, weight_sum}]`. `weight_sum` is the sum of the squared routing weights over the
layer's visits, which are the weights the Hessian carries. The annex names a weight sum without saying which; the
squared one is the one that makes `H / weight_sum` a weighted mean. `extq` divides the Hessian by it, a positive scale
the annex already calls neutral up to rounding, and falls back to `H = I` when it or the token count is zero. No
expert of the night took that fallback.

## E2. A night copy of the sentinel replaced the operator's

The operator's sentinel exits at once on a swap level that predates the night. A copy with a swap-growth criterion
(2 GB over its own start) ran instead, `~/llvq-inkling-mini/logs/sentinel_night.sh`. The operator's files under
`~/.llvq-guard` were not touched. Two memory incidents came before any measurement, at 22:50 and 22:52: a mutant test
of the executor ran with autograd on and took 20 GB of swap. Both runs were killed and the loader now turns autograd
off. No alert fired after that.

## E3. A census prediction was missed

Signed: under 5 % of the (expert) cells of layers 6 to 17 see fewer than 2,304 tokens after the sum over three
passes. Measured: 11.2 %. The other census prediction held (27.96 % in one visit, signed 20 to 50 %). The census
carries no criterion.

## E4. Readings on the two smaller sets were taken on the way

The prereg lists the sets {6, 7, 8}, {6..11} and {6..17} in order and reads the criterion on the largest one the
night completes. The two smaller sets were read as they completed and are in the journal. They do not enter the
verdict.

## E5. The layers outside the loop are encoded after the reading, with two changes of storage

The gate was read at 06:25 on {6..17}. Layers 0 to 5 and 18 to 23, which the prereg allows "if the night allows",
are encoded after it, during the day of 2026-11-22. Their reading on the 24 layers is not the gate's.

- Their dense reconstructions are kept in float16, for disk space. Measured first on layers 6 to 8: KL 0.009614 in
  float16 against 0.009611 in float32. The codes are checked against the float32 reconstructions before the
  conversion, and the reconstructions of layers 6 to 17 stay in float32.
- They are encoded in slices of 32 experts, where layers 6 to 17 ran at 10 threads in one piece. Experts 0 to 95 of
  layer 0 ran at 8 threads. The guard's sentinel then warned on the load (13.25 at 07:12, with two builds of lot 1
  running), and everything after runs at 6 threads, on the Hessians already built. The encoder is resumable an
  expert and the settings are the annex's. Whether its output is bit-identical across thread counts was not checked.

The calibration captures of layers 6 to 17 (36 visit directories, 21 GB) were deleted at 06:30 to make room. They are
rebuilt by `ops/inkling/capture.py` in about 18 minutes from the same tokens (`calib/tokens.u32` is kept).
