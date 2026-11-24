# Deviations from `preregistration-inkling-audit-identity-2026-11-24.md`

## E1, a first reading on a partial tree (2026-11-24)

The prereg names main after the three audit sets. The two jobs were first run on main with the code audit of
2026-11-18 alone (image f0aef998, jobs `6b05af09d0582cfbbefc8945` and `6b05b0f0d0582cfbbefc8c71`), without waiting
for the port of pull request 23 and of the llvq-tetra audits. Both pass. The reading the prereg names, on the whole
tree, is run after it and is the one the merge waits for.
