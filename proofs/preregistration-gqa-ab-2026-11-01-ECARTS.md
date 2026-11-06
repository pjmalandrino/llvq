# Deviations from the GQA A/B prereg of 2026-11-01

The prereg is `preregistration-gqa-ab-2026-11-01.md`, sha256 `a9bf7a28`, stamped before any run. It is not edited.
Written before the job launched.

## É1. The first image did not build, and the image comes from the next commit

The prereg says the image is built from the commit that carries it (`3e5fdb8`, anchored in `de65c9b`). That build,
Space sha `77bd2904`, stopped at `BUILD_ERROR` after 9.5 min: two `match` on `KvStore` in the cfg-gated half of
`bin/fusedrun`, in the `LLVQ_GRAPH_AB` and `LLVQ_KV_AB` modes, did not cover the new `Grow` variant (E0004). That
half does not compile on the Mac, so the image build was its first compiler, as the commit message said.

The fix makes both arms refuse `Grow` as they refuse `Cat`. `KvStore::from_env` never returns `Grow`, so neither mode
changes behaviour, and neither runs in this job. The image is rebuilt from the commit that carries this file. No
code path the job measures moves. The failed build cost $0: Space builds are not billed as jobs.
