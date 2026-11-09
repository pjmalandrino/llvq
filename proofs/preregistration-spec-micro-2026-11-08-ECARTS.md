# Deviations from proofs/preregistration-spec-micro-2026-11-08.md

1. **Host harness command letter (2026-11-08, after the stamp, before any measurement).** The verify rows' host
   command took the letter `'V'`, which `host_pers.cpp` already used further down its chain for another command. Six
   older tests of `pers_matches_served.rs` then fed that command to the new block and failed (`bad_alloc`); the three
   verify-row tests passed because they spoke to the new block. The command is now `'D'`, in `host_pers.cpp` and the
   test's `vrows_cmd`. The full suite passes: 62 of 62. No kernel, no bench code and nothing in the image changed:
   the Space build of commit 3476562 carries the same `pers_vrows.cu`, `persbench.rs` and `planesbench.rs`.

2. **The first run panicked before any verify-row number (2026-11-08).** Job `6af07cbfe6320c66fad755a7`, image
   `99b161e9`, 6 min 50 s billed, $0.21 (*computed* at $1.80/h). Gates 1 and 2 passed: oracle MATCH, the files at
   their bytes and sha256, `tetra48: 252 of 252`, `tile 64`. A4's arms ran and kept their bits. Then `planesbench`
   panicked at `planesbench.rs:4573` ("built when a Tetra section is on"): the line that builds the f64 codec listed
   `LLVQ_TETRA_OCC`, `LLVQ_PERS_ARMS` and `LLVQ_TROWS` but not `LLVQ_VROWS`, and step1-micro had always set
   `LLVQ_TROWS`. No verify-row kernel launched. The codec line now lists `LLVQ_VROWS`; nothing else moves. The rerun is
   the same command on a rebuilt image, output `spec-micro-2026-11-08-r1/`, timeout 13 min so that both runs together
   stay under the operator's $0.60 cap ($0.21 + $0.39 at worst). The raw output of the first run is in
   `docs/mesures/spec-micro-2026-11-08-brut/r0/`.

3. **Gate 4 was checked on one activation of four (2026-11-08, read on the card).** §4 asks v2 bit for bit to
   `tv_tetra48_h` on each of the four activations. `persbench::vrows_setup` checks the arms that owe the served bits
   (v2, lean, rot) on the bench activation only, and k2 on activations 0 and 1; it times i16 against the served rows
   on all four. The host test `the_verify_row_arms_that_owe_the_bits_keep_them` does check v2 on four activations, but
   the bench does not repeat it. The job script's gate 4 found no line for activation 1 and exited 1 after the report
   was written. On the card: v2, lean and rot bit for bit on activation 0; k2, whose second row is v2's f32 dot on
   activation 1, bit for bit; i16 within tolerance of the served rows on activations 1 to 3, 91.3 to 91.6 % identical.
   The numbers are reported under this reserve in `docs/mesures/spec-micro-2026-11-08.txt`; the operator decides
   whether the gate voids them. Not rerun: a third run would pass the $0.60 cap.
