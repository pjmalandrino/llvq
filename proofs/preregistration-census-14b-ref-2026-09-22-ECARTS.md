# Deviations. Preregistration census-14b-ref-2026-09-22

The prereg is stamped (sha256 `4c6beb59fc67a844...`) and not edited.

## E1. The encode job is launched while this census is still queued

The prereg says the 14B encode job is launched after this one ends, or into another bucket,
because `ops/README.md:241-243` (2026-08-10) records that two jobs writing the same bucket do not
mount. The 8B chain contradicts that note three times in one night: census-8b with bench-8b-base,
census-8b with dclm-8b-rowscales, and dclm-8b-ft-mmlu with served-8b-full all mounted
`Pier-Jean/jobs-artifacts` read-write together and completed (`jobs.csv`, 2026-09-22 rows;
`hf jobs inspect`). A mount refusal happens before billing. `seg1` of the encode was therefore
launched at 15:1x on 2026-09-22 while this census was still in `SCHEDULING`, into the same
bucket and different directories. If either job dies on the mount, it is relaunched and the
journal says so.
