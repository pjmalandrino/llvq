# Departures from the prereg of the decode-speed bench (`preregistration-recherche-debit-banc-2026-11-12.md`)

## E1. The image is built from a later commit than the bench commit (2026-11-12, before any measurement)

The prereg says one image built from the commit that carries the bench, `2794bbc`. The Space build of that tree was
cut twice at 1,800 s by the builder's limit, with no compile error (*measured*, build logs: first RUN 1,391 s,
runtime 236 s, `seal` 112 s, cut at the start of the `llvq-cuda` RUN). The operator chose to raise
`CARGO_BUILD_JOBS` from 2 to 3 in `ops/Dockerfile.cuda`. The image is therefore built from the commit that carries
this file. Between `2794bbc` and it, the only changes are the prereg and its anchor, this file, and that one line of
the recipe. No source file, kernel text or job script changes; the job's two digest gates (served unit, bench module)
are unchanged and still void the run if they miss. No job had been launched and nothing had been billed.
