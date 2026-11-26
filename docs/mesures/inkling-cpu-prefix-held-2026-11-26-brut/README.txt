Raw files of docs/mesures/inkling-cpu-prefix-held-2026-11-26.txt (parts B and C of prereg servi-au-bit-2026-11-26).
  b-ii/            gate B-ii: run.log, b-ii.json, logits.sha256, machine.txt (the two .logits.f32, 7.2 MB each, stay out of git)
  cpu-serve/       gate B-iii: summary.txt, checks.jsonl, driver.txt, machine.txt, per request .json/.raw.txt/.footprint.txt,
                   per server .server.log and .mem-*.txt; the store home/ (the 0.81 GB prefix file) stays out of git
  held-before/     gate C-ii, a05d99a; held-after/ gate C-ii, c00ce5e; c-ii-compare.txt joins them; their home/ stay out of git
  mutants.py, mutants.log   the four mutants; clippy*.log, tests-*.log   the final checks
