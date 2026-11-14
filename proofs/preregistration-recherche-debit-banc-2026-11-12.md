# Preregistration. Decode-speed research phase, R2: stage 2 of the verifier, one card run of `rowbench`

Written on 2026-11-12 on branch `recherche/debit-2026-11-12`, once the code it describes existed. Timestamped before
the first measurement, then never edited; a departure goes to
`preregistration-recherche-debit-banc-2026-11-12-ECARTS.md`.

The launch is the operator's. Cost: about **$0.33** on l40sx1 (*estimated*, §9), after a Space rebuild that bills no
card. Cap: the job's timeout of 25 min, **$0.75** at worst (*computed* at $1.80/h). Spent before it: $19.22 of the $25
cap, $0 of the phase's $5 (`docs/mesures/recherche-debit-r1-cadrage-2026-11-12.txt` line 77); see
`docs/data/jobs.csv` at launch.

## 1. The question

The phase's number is the stream rate of the Tetra rows, in GB/s of packed words, on the L40S (R1). Its search route
needs a verifier whose stage 2 scores candidates on the card. Before any candidate counts, this run must show four
things the journals do not hold (R2 condition of `docs/MODES.md`):

1. it reproduces the served row on the old all-Tetra encode at the journals' rate, and the known arms in the
   journals' order (rule D, §6);
2. it reports each shape apart (no journal has a rate at one d_in);
3. it measures the card's read rate with a bare probe (no journal has one: 864 GB/s is a datasheet figure);
4. it times the sealed file's Tetra rows at 4E's gate/up geometry with no other traffic on the card.

This run scores and describes. It projects no gain.

## 2. What the code holds at the commit that carries the bench

The commit is `2794bbc`.

Nothing served moves: no `configs/` change, no served kernel text changes a byte (the served unit's pin
`pers.rs::the_served_unit_is_the_published_one`, 101,625 bytes, `6193094310bc…`, and 4E's pins stay green). New:

- `llvq-cuda/kernels/rowcand.cu`: the row programs, registered one line each in `ROWC_ARMS`, and three bodies at 4E's
  geometry (one CTA of 512 threads a SM, 4E's shared layout, sites split over CTAs as `[d c / g, d (c + 1) / g)`,
  rows over warps statically): `tv_rowc_<arm>` over a list of sites, `tv_rowc_gu_<arm>` as 4E's gate/up phase (pairs,
  gate then up, one staged activation a layer), `tv_rowc_relay_<arm>` (a load-time rewrite in the served stride).
- `llvq-cuda/kernels/bwprobe.cu`: the read probe.
- `llvq-cuda/src/rowcand.rs`: the registry read from the kernel text, the site words, the error judge, the probe's
  expected sum. `llvq-cuda/src/bin/rowbench.rs`: the bench. In both lists of `ops/Dockerfile.cuda`.
- `llvq-cuda/tests/rowcand_gate.rs` and `tests/host_pers_rowcand.inc`: stage 1, the Mac gate.

### Stage 1, run on the Mac at $0 (*measured* on the host twin, `cargo test -p llvq-cuda --test rowcand_gate`)

The served reference is the host's `served_ref` (`tv_tetra48_h` with NVVM's three fusions, pinned to the served text
by `the_served_replay_is_the_served_text`). 5,280 rows on recorded words: coverage sites at d_in 2,560, 4,096 and
9,728; the six lattice records of `llvq-tetra/tests/fixtures/mini` at their shapes, and relaid at 2,560 and 9,728;
4,096 seeded random words; the gate/up body on two pairs; three grids each.

| arm | owes | y equal | f32 equal | max rel err | q99 | verdict |
|---|---|---|---|---|---|---|
| served | the bits | 5,280 / 5,280 | 5,280 / 5,280 | 0 | 0 | pass |
| served_np | the bits | 5,280 | 5,280 | 0 | 0 | pass |
| acc2 | bound 1e-5 | 5,277 | 4,725 | 3.753e-7 | 2.024e-7 | pass |
| pword | nothing | 0 | 0 | 7.2e2 | 4.8e2 | diagnostic |

Relative error: `|f32 - served f32|` over the rms of the site's served f32 rows.

Coverage of the recorded words (*computed*, asserted by `coverage_of_the_recorded_words`): all 16 combinations of
parity, section-1 class, mixed split and gain bit; 128 of 128 prefixes (s8, b1), 1,024 of 1,024 branches (s8, b2),
128 of 128 suffixes (s16, b3); every table row of each section in each class (4,096 + 2,048 + 4,096); shells 0 and 2
to 25 (2 and 3 by enumeration of Λ₂₄'s shells, every class pattern); shell 26 (the codebook's maximum,
`tetrashell.rs`) is not reached; shell 1 does not exist. The fixture's 4,520 real blocks reach shells 7 to 23, both
gain levels.

Mutants of the served row, each applied exactly once to a copy (`mutant` asserts one site and a changed text), each
refused by the gate on the served arm:

| mutant | site | rows moved of 5,280 |
|---|---|---|
| table_index: section 3's row in the other class when p = 1 | `llvq_f1rank_v3.cuh` | 4,860 |
| dropped_sign: every coordinate's magnitude | `llvq_f1rank_v3.cuh` | 5,280 |
| swapped_lane: lanes 0 and 1 trade their blocks | `llvq_pers.cuh` | 813 (max rel err 2.0e-7) |
| gain_dropped: every block at centroid 0 | `llvq_pers.cuh` | 5,115 |
| mixed_split: `i2 < N0` to `i2 <= N0` (only i2 = 1,240 sees it) | `llvq_f1rank_v3.cuh` | 102 |

One equivalent mutant is checked as such: lanes 2k and 2k + 1 swapped is an automorphism of `warp_sum`'s xor
butterfly, and the gate reports it bit for bit. The swapped-lane mutant moves bits by 2.0e-7 of the rms: a bound arm at
1e-5 would not see a reassociation of that size, which is the meaning of declaring a bound.

Gates of the commit, on the Mac (*measured*, 2026-11-12): `cargo test -p llvq-cuda`, 34 suites, 287 passed, 0 failed;
`cargo clippy -p llvq-cuda --all-targets` clean on the Mac and on `x86_64-unknown-linux-gnu`; the bench module
(320,451 bytes, sha256 `5dab2bca…`) compiles under devel nvcc 12.4 for sm_89. Its block loops count 226 SASS
instructions a block for served, 218 for served_np, 224 for acc2 and 24 for pword (`ops/rowcand-sass.sh`, devel
count, to compare between arms only).

## 3. What runs

One image built from the commit that carries the bench. One job, l40sx1, `ops/jobs/recherche-debit-banc.sh`:

1. **Oracle** on Qwen3-0.6B (hard rule 10).
2. The sealed 4B (`sealed-4b-2026-09-23/qwen3-4b-sealed.bin`, 1,418,224,685 B, sha256 `886391a8…`) and the bare encode
   (`tetra-4b-2026-09-06/qwen3-4b-tetra.bin`, 1,770,529,149 B, sha256 `0adb7cfd…`) checked by size and sha256. Both are
   in the bucket (`hf buckets ls`, read on 2026-11-12).
3. `rowbench <sealed> <bare> --rounds 15 --dump <out>/cubin`: no `LLVQ_*` variable; the tile is the card's row (64 on
   sm_89). 2 warmup rounds, 15 timed; every valid run once a round, in a fixed order; times on the host around launch
   and sync, as the row-loop bench took them.

## 4. The sets, the arms, the controls

Sets (bytes *computed* from the shapes; words with their row stride, then with binary16 tails and f32 row scales):

| set | records | rows | words GB | with tails and scales GB |
|---|---|---|---|---|
| sealed:d_in=2560 | q, k, gate, up x 36 | 884,736 | 0.5662 | 0.5981 |
| sealed:d_in=9728 | down x 24 (layers 0-11, 24-35) | 61,440 | 0.1494 | 0.1506 |
| sealed:all | the 168 Tetra records, file order | 946,176 | 0.7157 | 0.7487 |
| sealed:gu4e | gate, up of 36 layers, as pairs | 700,416 | 0.4483 | 0.4735 |
| bare:d_in=2560 / 4096 / 9728 | the bare encode by shape | 921,600 / 92,160 / 92,160 | 0.5898 / 0.0944 / 0.2241 | 0.6230 / 0.0977 / 0.2260 |
| bare:all | the 252 records, the served schedule's order | 1,105,920 | 0.9083 | 0.9467 |
| sealed:int4:d_in=2560 / 4096 / 9728 / all | v x 36 / o x 36 / down x 12 / 84 | | 0.0501 / 0.2005 / 0.1588 / 0.4094 (words, scales, biases) | |

The bench prints the shapes it finds; a sealed file whose Tetra records are not these is reported as it is.

Arms, every one on every row set:

| arm | what | owes |
|---|---|---|
| served | the registry's reference: 4E v9's row (`pers_tetra_row<24, false, true>`, as `pers_tetra_rows_static` runs it) | the bits; its rows must equal `tv_tetra48_h`'s on every set, or no arm of the set is timed |
| served_np | the same row with no word in flight | the bits |
| acc2 | two accumulators a lane, blocks alternating, summed before the butterfly | bound 1e-5 |
| pword | the words read and folded by xor, nothing decoded: the Tetra stream's own read at this geometry | nothing (diagnostic) |
| gu_<arm> | each of the above at 4E's gate/up geometry, on sealed:gu4e | as its arm |

Controls (the journals' arms, `pers_trows.cu`, unchanged): v2, v3, ilp2, v2u, l1w, bcast, xbcast, v2_s, v2_o2, nc,
nc256, cpa, tile, bal, on every row set (v2_s and v2_o2 on d_in up to 4,096, as in the journals). The int4 controls on
the sealed file's own int4 records: the served eager `tv_q4_hh` (q4hh, one launch a record), q4, q4nc, q4t, q4read.

Every arm is checked on the card before it is timed: y against `tv_tetra48_h` (or `tv_q4_hh`), its f32 values against
the served arm's. An arm that owes the bits and moves one, or misses its bound, spills, or does not fit, is REFUSED:
its line says why and carries no rate.

## 5. The probe

`tv_bw_fill` writes word i of a 2^29-word buffer (2.147 GB, 22 times the L40S's 96 MB L2, so a read of it is a DRAM
read whatever L2 kept from the previous round) as a 32-bit finaliser of i: no constant, no zero page. Three kernels
read and sum it (one u32 a load; one 128-bit load; four 128-bit loads in flight a thread), at 4E's geometry (512
threads, one CTA a SM) with and without a `prefetch.global.L2` two grid strides ahead, then the 128-bit unrolled kernel
at 256, 512 and 1,024 threads and 1, 2 and 4 CTAs a SM where they fit: 16 configurations. Each CTA writes its u64
partial; the host adds them after every timed launch and compares with its own sum of the same hash. A miss refuses
the configuration. Reported: GB/s of the buffer, median and range over the rounds, each configuration. The highest
median is the phase's measured read rate on this card; the probe and `pword` together bound the Tetra stream at this
geometry from the read side.

## 6. Rule D, the fidelity of the stand-in

The stand-in counts only if both hold:

- **D1.** The served arm on bare:all, in the journals' accounting (0.9467 GB), reads a median inside **[368, 383] GB/s**:
  pers-rowloop 379 [368-383], step1-micro 374 [368-378]. The control v2 is printed beside it on the same rule.
- **D2.** Each known arm's median speed over its base, round by round, on bare:all, lies within **±0.03** of its
  journal's median: v3 0.849, ilp2 0.928, v2u 0.825 (over v2); l1w 1.214, bcast 0.987, xbcast 0.995 (over v2u); v2_o2
  1.076 (over v2_s); nc 0.810, nc256 0.845, cpa 0.960, tile 1.002, bal 0.934 (over v2). This keeps the journals' order
  where their arms were separated (v2 and tile at the top, nc at the bottom).

If D1 or D2 fails, no rate of this run is a candidate's score: the run is reported as a description of the sealed
file's rows, and the verifier goes back to R2.

## 7. Gates

Each voids the run:

1. The image builds; `oracle` prints MATCH.
2. Both files at their bytes and sha256. The references print the served unit, 101,625 bytes, sha256
   `6193094310bc…`; the bench module prints the digest the Mac builds (`rowcand.rs::tests::dump_rowcand_module`).
3. The served arm prints every row bit for bit to `tv_tetra48_h` on sealed:all, sealed:gu4e and bare:all.
4. No `LLVQ_*` variable in the job environment.

An arm or a probe configuration that is refused voids only its own line.

## 8. What the run reports

- `ROUND` lines: round, set, arm, ms, GB/s of words, GB/s with tails and scales.
- A median [round-by-round range] a run, in both accountings, and the speed of each arm over its base (registry arms
  over `served`, `gu_` arms over `gu_served`, known arms over v2, v2u or v2_s, int4 arms over q4), formed round by round,
  never a quotient of two minima.
- `FIDELITY` and `ORDER` lines (rule D), one `TSV` line a run (set, arm, kind, owes, status, bytes, ms and GB/s with
  ranges, rows equal, max and q99 error).
- The driver's cubin of the module (`cubin/rowbench.cubin`) and its text, for the offline SASS count
  (`ops/rowcand-sass.sh`, Docker on the Mac).

Labels: every rate *measured*; bytes *computed*.

## 9. Cost

*Estimated* from the two jobs of this bench family (`docs/data/jobs.csv`): pers-rowloop (6aec7403, 9 min billed, 497 s
running, $0.25) and step1-micro (6aef03fa, 8 min billed, $0.25), each an oracle and one bench process over the 1.77 GB
bare file. This run reads two files (3.19 GB), decodes their records, compiles the module twice (once for the dump);
its rounds take about 0.3 s each (*computed* from the bytes at the journals' rates). About 11 min billed, **$0.33**.
Cap: 25 min, **$0.75**.

## 10. Signed predictions (main session)

Signed on 2026-11-12, before the stamp, from the journals alone. Rates are medians in GB/s in the accounting with tails
and scales; ratios are formed round by round. Each line is scored in the journal as in or out.

| # | quantity | point | interval | why |
|---|---|---|---|---|
| 1 | served on bare:all (rule D1) | 376 | [368, 383] | two journals, same card, same bytes |
| 2 | rule D2, all twelve ratios inside ±0.03 | holds | 60 % | twelve tests at once; l1w and nc are the likeliest to leave |
| 3 | probe, highest median of the 16 configurations | 760 | [700, 830] | 864 is the datasheet; a summing read pays a little |
| 4 | probe at 4E's geometry (512 threads, one CTA a SM), best of its rows | 690 | [640, 740] | q4read reads 671 at that geometry |
| 5 | gu_served on sealed:gu4e, no other traffic | 375 | [355, 395] | the stamps give 359 and 385 in the token |
| 6 | served by shape on the sealed file | 2,560 < 9,728 | 2,560 in [340, 375], 9,728 in [385, 430] | longer rows amortise the row's fixed cost |
| 7 | served_np over served, sealed:all | 0.94 | [0.90, 0.98] | v9 without its prefetch reads 0.94 on gate/up in the token |
| 8 | acc2 over served, sealed:all | 1.00 | [0.97, 1.04] | the row is bound by its decode, not by its sum |
| 9 | pword on sealed:all | 640 | [560, 700] | a read of the same words with nothing decoded |
| 10 | q4t and q4read on sealed:int4:all | 548 and 671 | ±25 each | step1-micro |

What would surprise: a probe under 700 (the card's read rate would then explain the int4 rows, and the Tetra rows
would sit at more than half of it); pword under 450 (the geometry, not the decode, would hold the Tetra rows); acc2
above 1.05 (the sum would matter).

## 11. Not established by this run

- Any tok/s. A candidate that wins stage 2 is an idea only after a token-level measurement (R6).
- Any card but the L40S. Hardware counters (none on HF Jobs). The SASS count, which is offline.
- The coverage of shell 26 on the Mac (§2).
- The relay form (`tv_rowc_relay_<arm>`): no registered arm uses it, and it has never run.
