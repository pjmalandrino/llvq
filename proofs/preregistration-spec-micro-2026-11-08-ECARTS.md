# Deviations from proofs/preregistration-spec-micro-2026-11-08.md

1. **Host harness command letter (2026-11-08, after the stamp, before any measurement).** The verify rows' host
   command took the letter `'V'`, which `host_pers.cpp` already used further down its chain for another command. Six
   older tests of `pers_matches_served.rs` then fed that command to the new block and failed (`bad_alloc`); the three
   verify-row tests passed because they spoke to the new block. The command is now `'D'`, in `host_pers.cpp` and the
   test's `vrows_cmd`. The full suite passes: 62 of 62. No kernel, no bench code and nothing in the image changed:
   the Space build of commit 3476562 carries the same `pers_vrows.cu`, `persbench.rs` and `planesbench.rs`.
