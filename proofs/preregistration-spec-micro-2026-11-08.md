# Preregistration. The verify rows of a speculative round with B1's integer dot, and B1's levers, alone on the card

Written on 2026-11-08 on branch `claude/spec-4e-0058c6`, once the code it describes existed and passed its host
checks. Timestamped before the first measurement, then never edited; a departure goes to
`preregistration-spec-micro-2026-11-08-ECARTS.md`.

The operator gave the go on 2026-11-08 ("Ok go dans une sous session", for a microbench of about $0.30 under a cap of
$0.60). Cost: about **$0.30** on l40sx1 (*estimated*, §8), after a Space rebuild that bills no card. Timeout 20 min:
**$0.60** at worst (*computed* at $1.80/h). Spent before it: $280.04 over the rows of `docs/data/jobs.csv` on this
branch (*computed*); other branches carry their own rows.

## 1. Why this run exists

The served doors decode 221.5, 142.5 and 86.3 tok/s at 4B, 8B and 14B (*measured*, served-v9). One token cannot go
past about 260, 174 and 105 tok/s on this engine (*estimated*, `docs/recherche-debit-maths.md` §2). Only reading each
weight once for several tokens moves that limit: a speculative round checks k drafted tokens in one pass of k + 1 rows.

The 0.6B drafter failed at ×0.60 of greedy because two rows of 4E's verify pass cost 1.95 greedy tokens (*measured*,
`docs/mesures/spec-gate-2026-11-07.txt`). The study attributes that to the kernel, not to the format, and projects an
extra row at about 0.30 ms at 4B with B1's integer dot (*estimated*, §11.2). Of those 0.30 ms, the Tetra rows are the
only part this format decides. This job times them alone: the cost of one more row in 4E's Tetra row loop, with the
relabeled word, the activation rounded to int16 and the dot in `dp2a`. It also times B1's three levers one by one, as
§8 of the study asks.

`docs/ETAT.md` §6 reopens speculative decoding when a 2-row verify pass costs under 1.4 greedy tokens and a draft
token under 1 ms. This job cannot reopen it: it times one part of the pass. It decides whether that part leaves the
rest of the pass room to meet the study's gate of 1.15 greedy tokens.

## 2. What the code holds at the commit that carries this file

Off by default; no served path moves. `planesbench` with `LLVQ_VROWS=1` builds its own NVRTC module: the row-loop
unit of `pers_trows.cu`, then `llvq-cuda/kernels/pers_vrows.cu`. 4E's module and the row loops' keep their bytes.

The geometry is 4E's and the row loops': one CTA of 512 threads a SM (142 on the L40S), the decoder tables and the
activation in shared memory, each site's rows split over CTAs as `[d c / g, d (c + 1) / g)` and over warps
statically, a CTA barrier between sites, no grid barrier. The sites are the 252 matrices of the bare Tetra file
(`qwen3-4b-tetra.bin`, sha256 `0adb7cfd…`) in served order. The reference rows are the served `tv_tetra48_h`'s, from
the served unit (digest `6193094310bc…`, a gate), on four activations: the bench's, and three seeded gaussians of
9,728 values (seeds `0x7e72`, `0x7e73`, `0x7e74`).

| arm | what it is | K | owes |
|---|---|---|---|
| v2 | `pers_tetra_row<24, false, true>`, 4E's row. The base of every ratio | 1 | the served bits |
| k2 | `trows_row_k2`: one decode, two f32 activations. Replicates shaker-4e's second row | 2 | the served bits on both |
| pword | v2's loads and prefetch, words folded by XOR, nothing decoded: the stream rate of Tetra words here | 1 | nothing |
| lean | the relabeled word: a bijection of the 48 bits made once at setup (`tv_vrows_lean_relayout`, same bytes, same place), the first rank row's index one 12-bit field, the three Golay pattern bytes as `GA[s8] ^ GB[b1, b3, b2]` from two tables of 64 and 128 entries (`tv_vrows_lean_tables`, which counts the paths where the XOR is not the trellis). Same point, same f32 dot and fold as v2 | 1 | the served bits |
| rot | v2's decode and dot with lane rotation: a warp's rows as one stream of items, item t on lane t mod 32. Rows under 32 blocks run v2's row | 1 | the served bits |
| i16 | lean's decode, the activation rounded to int16 a block of 24 at staging (scale = max / 32,767 rounded up to bf16), the dot in exact integers through twelve `dp2a`, one fold a block | 1 | within one binary16 ulp of the served rows, or 1e-3 of the site's rms |
| i16k2, i16k3, i16k4 | i16 with one decode for K activations: a verify pass of K rows | 2, 3, 4 | each activation's rows equal to i16's on that activation, bit for bit |

The relabeled words are a second copy made for the bench; the setup prints nothing new about memory because the copy
has the served stream's bytes (rule 6: 0 b/param). No file and no served layout changes.

Two facts the host checks established on the way, by construction and by test:
- **The Golay map is linear.** GA ^ GB reproduces the trellis on the 4,096 paths; the bench recounts it on the card.
- **Lane rotation needs no shuffle back.** `warp_sum`'s XOR butterfly pairs residue classes (c, c + 2^k) mod 2^(k+1)
  at every level, a set a cyclic shift of the lanes maps onto itself; the rotated sum is the same tree up to
  commutation. A mutant that shifted the rotation by one lane survived, which is how this was found.

Shared bytes: 101,120 for i16k4 at d_in 9,728, under the 101,376 a CTA may opt into on sm_89. Registers under the
devel ptxas 12.4 for sm_89 (docker, on the Mac): v2 72, k2 87, pword 35, lean 73, rot 78, i16 63, i16k2 80, i16k3
84, i16k4 103; zero bytes of spill on every symbol.

### Host checks at $0

- `the_verify_row_arms_that_owe_the_bits_keep_them`: v2, lean and rot give the served rows and their f32 values before
  `f2h` on 7 sites (d_in 128, 800, 2,560, 2,568, 4,096 twice, 9,728: odd and even block counts, a row of 33 blocks, a
  row under 32, a tail of 0), at 4 grids from 1 to 4 warps a CTA; k2 on both its activations; pword moves rows.
- `the_integer_dot_verify_rows_are_the_one_row_rows`: i16 sits within 1e-3 of each site's rms of the served f32
  values on four activations, and is not equal to them; i16k2, i16k3 and i16k4 give i16's rows on each activation,
  bit for bit, binary16 and f32, at 4 grids.
- `verify_row_mutants_are_killed`: 16 mutants, each applied exactly once and killed (lean's field masks, its word,
  the relayout's odd block, rotation's row end, next row and accumulator choice, the `dp2a` halves, the scale, the
  trio order of the staging, its round-up, the K activations' loads, areas and tails).
- `the_i16_check_reads_ulps_and_the_rms_bound`, `the_verify_row_shared_bytes_fit_one_cta`.
- What the host cannot check: any time; the driver ptxas's registers; the memory ordering between SMs.

## 3. What runs

One image built from the commit that carries this file. One job, l40sx1, `ops/jobs/spec-micro.sh`.

1. **Oracle** on Qwen3-0.6B (rule 10).
2. The two banc-252 files checked by size, the Tetra file by sha256. Then `LLVQ_TILE_BLOCKS=64
   LLVQ_BENCH_ARMS=slot32,planes14,fp16,tetra48 LLVQ_SEG_TETRA=1 LLVQ_VROWS=1 LLVQ_DUMP_CUBIN=<dir> planesbench <ball>
   <tetra>`: the row loops' command of step1-micro with `LLVQ_VROWS` in place of `LLVQ_TROWS` and `LLVQ_TETRA_OCC`.
   The arms run in A4's 7 rounds, 2 discarded, interleaved with A4's.

## 4. Gates

Each voids the run:

1. The image builds; `oracle` prints MATCH.
2. The files at their bytes, the Tetra file at its sha256; `tetra48: 252 of 252 matrices matched by name`; `tile 64`.
3. The reference prints the served unit, 101,625 bytes, sha256 `6193094310bc…`.
4. v2 prints every row identical bit for bit to `tv_tetra48_h` on each of the four activations.
5. The lean tables print 0 of 4,096 paths wrong.
6. No `LLVQ_*` variable in the job environment.

Another arm that spills, does not fit, moves a row it owes or misses its tolerance voids only its own numbers. If i16
or any i16kK is void, the verify gate (§6, rule B) is void.

## 5. Outcomes reported, never gated

- Each symbol's registers and local bytes under the driver's ptxas; the cubins.
- Per arm ms and GB/s, median and range, on words at the row stride, binary16 tails and row scales (pword: words only);
  speed over v2 round by round.
- k2: its time over v2 less 1, round by round.
- i16kK: an extra row's cost, f_K = (t(i16kK) - t(i16)) / ((K - 1) t(v2)), and (t(i16kK) - t(i16)) / (K - 1) in ms,
  round by round; f = the median of f_2, f_3, f_4.
- i16 against the served rows: rows identical, rows one ulp away, the worst |dy| over the site's rms, by activation.
- In situ, *estimated*: f × 1.81 ms, the 4B's Tetra rows inside 4E (plan-300 §4).

## 6. Decision rules

- **A, identity.** v2, k2, lean and rot keep the served bits; i16 its tolerance; i16kK i16's bits. An arm that fails
  is reported and not timed.
- **B, the verify gate.** f ≤ **0.15**: PASS. The Tetra part of an extra row is at most 0.27 ms in situ, which leaves
  0.36 to 0.41 ms for the rest of the row (int4 rows, head, attention, norms) under the study's gate of a 2-row pass
  at 1.15 greedy tokens (0.63 to 0.68 ms at 4B). Step 2, the prompt-lookup port into 4E, is then proposed to the
  operator with its cost. 0.15 < f ≤ **0.26**: BETWEEN, reported; the operator decides. f > 0.26: FAIL. The Tetra rows
  alone would take the 1.15-token pass out of reach; the mission stops and reports.
- **C, B1's levers.** lean at ×1.05 over v2, rot at ×1.05, i16 at ×1.15 (the study's §8 bars): each licenses its 4E
  port as a single-token lever. Reported; not this mission's gate.

## 7. Signed predictions

*Estimated* from step1-micro, shaker-4e, sass-tetra-lean and the study's time model; nothing fitted to a number of
this run.

| quantity | point | interval |
|---|---|---|
| every arm that owes the bits, against its reference | **identical** | |
| i16 against the served rows, rows out of tolerance | **0** | |
| i16 against the served rows, rows identical | **95 %** | [85, 99.9] |
| every symbol loads without local memory (driver ptxas) | **yes** | |
| v2, GB/s | **374** | [355, 395] |
| k2, second f32 row over v2 | **0.38** | [0.32, 0.45] |
| pword, GB/s on words | **600** | [450, 760] |
| lean over v2 | **×1.06** | [0.98, 1.15] |
| rot over v2 | **×1.06** | [0.97, 1.17] |
| i16 over v2 | **×1.20** | [1.05, 1.40] |
| f, an extra i16 row over v2 | **0.10** | [0.04, 0.20] |
| verify gate | **PASS** | |

## 8. Cost

step1-micro ran the same command with 18 row-loop arms, 5 int4 arms, 6 counters and the one-launch pass in 8 min 24
s billed, $0.25 (*measured*). This job has 9 arms, the references on four activations (1,008 launches of
`tv_tetra48_h`, under 2 s) and the relayout of 0.95 GB of words (under 1 s). About 9 min, **$0.27** (*estimated*).
Timeout 20 min: **$0.60** at worst, the operator's cap.

## 9. What this run does not establish

- Any verify pass inside 4E: the int4 rows, the head, attention and the norms of an extra row, the spills and
  prologues spec-gate measured, and the grid barriers.
- Any tok/s at a door, any acceptance rate, any perplexity of B1's rounding.
- The 8B and 14B. Their `down` sites stage wider activations: at d_in 12,288 (8B) K = 3 fits one CTA's shared memory
  and K = 4 does not; at 17,408 (14B) K = 2 fits and K = 3 does not (*computed*, `pers::vrows_area_bytes`). A verify
  pass of more rows there needs the activation split over d_in, int8 rows, or rows a pass capped by size.
- Any card but the L40S.
