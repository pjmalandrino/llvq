# Departures from preregistration-step2-4e-2026-11-07.md

Written on 2026-11-07, after job 6aef3f69 and before its journal was committed. The prereg itself is not edited.

## E1. K2's gate read on v8, because v7 did not install

- **What happened.** `install_rows` refused v7 at 512 threads: "every block size asked for spills or fails its
  occupancy". By §4, gate 6, v7's numbers are void. The registers and local bytes were not printed, because the line
  is pushed before the check and printed only on success.
- **What the prereg said.** §7 reads K2's gate on v7's MLP down input.
- **What was done.** The gate is read on v8. v8 is v7 plus the attention ring (`PERS_4E_ROWS` 6 takes every `>= 5`
  branch), and the ring does not touch the MLP. The threshold, 200 us, does not move. v8's stamp symbol spills 48 B,
  which can only add time to the stamped parts, so a reading under 200 on it holds a fortiori.
- **What it costs.** v7 over v6 and v8 over v7 have no reading. The share of v8's gain due to the ring alone is not
  separated from K2's.
