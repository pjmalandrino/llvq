# Deviations from the CUDA prereg of 2026-09-30

The prereg is stamped and is not edited. Every departure from it is written here, with its
date and its reason.

## 1. The card is an `l4x1`, not the `l40sx1` the prereg names (2026-09-30, 20:0x UTC)

**What the prereg says.** Line 5: "$0.45 estimated, hard cap $1.20, which is the 40 minute
timeout at l40sx1's $1.80 an hour."

**What happened.** The fourth launch, `6abd3428fbc85ba68235bcd4`, sat in `SCHEDULING`,
"Waiting for requested hardware to become available", for **3 h 47** with no card. The
longest queue in `docs/data/jobs.csv` before today is 70 minutes. It was canceled at
**$0 billed**: billing starts at `RUNNING`.

**The deviation.** The default flavour of `ops/jobs/hf-cuda-4b.sh` becomes `l4x1`, and
`FLAVOR=` overrides it. `run.py bench` refuses any card outside `BENCH_FLAVORS`, so the
launch carries `--any-flavor`, whose stated duty is to name the card in every figure that
comes out of the run.

**Why it does not touch the result.** The four arms are a build, a per-row arithmetic
identity, 64 generated tokens against a reference dump, and a count of resident bytes. Not
one of them is a throughput, so rule 5 has nothing to divide. Two facts carry the substance:

- **Same architecture.** `ops/run.py` records `cap=89` for both flavours. The L4 and the
  L40S are Ada, sm_89, so `cpp_extension` compiles the same device code for the same target,
  which is precisely what arm 1 tests.
- **Enough memory.** 24 GB against 48. The object is the packed 4B, 2.8 GB on the Metal
  path, and the reference arm of the check materializes nothing dense.

**What it costs.** The same 40 minute timeout at $0.80 an hour is a hard cap of **$0.53**
instead of $1.20, and the estimate falls from $0.45 to about **$0.25**.

**What it forbids.** No tok/s, no GB/s and no × from this run may be compared to any figure
of this repository, which were all taken on an `l40sx1` or on the Mac. If a throughput is
ever wanted through this loader, it is a separate run on an `l40sx1`, preregistered as such.
