# Preregistration. Decode-speed research phase, R6: rowbench's two winning bodies inside the token, one card run of `fusedrun LLVQ_PERS_4E`

Written on 2026-11-13 on branch `recherche/debit-2026-11-12`, once the code it describes existed. Timestamped before
the first measurement, then never edited; a departure goes to
`preregistration-recherche-debit-jeton-2026-11-13-ECARTS.md`.

The launch is the operator's. Cost: about **$0.60** on l40sx1 (*estimated*, §9), after a Space rebuild that bills no
card. Cap: the job's timeout of 30 min, **$0.90** at worst (*computed* at $1.80/h). Spent before it: $0.30 of the
phase's $5 and $19.52 of the $25 cap (`docs/mesures/recherche-debit-corps-2026-11-13.txt`, last line); see
`docs/data/jobs.csv` at launch.

## 1. The question

The third rowbench run (`docs/mesures/recherche-debit-corps-2026-11-13.txt`, job 6af6f074, L40S) scored two bodies
faster than the served row on the sealed 4B's Tetra rows ALONE: `flat` x1.185 [1.179-1.190] (a bound of 1e-5 on rows),
`endpass` x1.053 [1.050-1.056] (the bits). As a ceiling only (*computed* there): x1.07 and x1.02 on the token if the
gains carried to the token's Tetra bytes. K5 and K10 of `docs/CERVEAU.md` record that time freed in 4E's rows has
reappeared downstream before, and that the gate/up rows already lose 10 to 16 % in the token against alone.

This run asks: inside 4E v9's token, on the sealed 4B, at a 5-token prompt and at 8,192, does either body, put on
every Tetra row phase, decode faster than v9 by more than the job's own noise floor (rule F1 to F2, §5)? And for
`flat`, what do its tokens and logits do against v9's?

## 2. What the code holds at the commit that carries the arms

The commit is `0768aa6`. Nothing served moves: no `configs/` change; the served unit keeps its pin
(`pers.rs::the_served_unit_is_the_published_one`, 101,625 bytes, `6193094310bc…`), the served 4E v9 module keeps its
pin (`the_4e_v9_module_is_the_one_the_doors_gate_on`, 252,757 bytes, `1b422b62…`), E1's v9 reference and v9ft keep
theirs (`d55a097b…`, `22ff8320…`). Changed since `d71da2f`:

- `llvq-cuda/kernels/pers_4e_rc.cuh` (new): the two bodies as 4E row forms (§2.1). Never in a served module.
- `llvq-cuda/src/pers.rs`: two E1 arms, `v9ep` and `v9fl`, in `P4E_E1_ARMS`, through E1's mechanism
  (`p4e_e1_edits`: v9's module text with edits applied as text, the pattern of v9np, v9mv and v9ft); `p4e_rc_edits`,
  the edits; `p4e_e1_owes_bits` (every arm but v9fl); `p4e_rc_layout`, the shared-memory check of §2.2; tests that
  the arms are E1's v9 with the edits and nothing else, that the layout is as §2.2 says, and the two modules' sha256.
- `llvq-llm/src/persist4e_cuda.rs`: an rc arm's install runs `p4e_rc_layout` and refuses a shape that does not fit,
  by name; it prints the regions and how many Tetra matrices are off 16 bytes.
- `llvq-llm/src/bin/fusedrun.rs` (`LLVQ_PERS_4E`, the measurement mode, refused beside `LLVQ_CONFIG` as before):
  the first arm named again as the last of three or more is rule F1's control (§3); an arm that owes a bound is
  timed on its own tokens and its tokens and logits are compared with the reference's and reported (§7).
- Tests: `tests/host_pers_rc4e.inc` (command 'e' of the host twin, the row gate), `tests/host_pers.cpp`,
  `tests/host_pers_4e.inc` (+ 256: a smooth SiLU table), `tests/rowcand_gate.rs`, `tests/pers_matches_served.rs`.

The pattern followed: an E1 arm. A form is a module built from v9's text with edits (`p4e_e1_unit_text`), selected
only by `LLVQ_PERS_4E=<count>:<arms>`, a measurement mode that `fusedrun` refuses beside a served `LLVQ_CONFIG` and
beside every other mode (`runs_alone`). No served config can name an E1 arm.

### 2.1 The two arms

Both replace, on 4E's 512-thread symbols, the call of `pers_4e_rows` on each warp's list of rows by a call of
`pers_rc_rows<PERS_RC_ARM>` on the CTA's whole slice, at the four Tetra row phases: q/k (one list, q then k), gate/up,
the Tetra down, and the 8B's Tetra o (not on the 4B). The 768-thread symbols keep v9's rows; the runtime takes 512.

| arm | body | owes | what it does |
|---|---|---|---|
| `v9ep` | `endpass` | the bits | v9's row (`pers_tetra_row`, a warp a row, the next block in flight, row i on warp i mod 16 as v9 deals them); lane 0 writes the row's sum to a shared slot; after a barrier a thread a row does the tail, the row scale and the phase's store (`pers_epilogue`, then f2h, the residual add or the gate/up value). Chunks of 2,048 rows |
| `v9fl` | `flat` | a bound of 1e-5 on rows | the activation restaged with one float of padding every 8 blocks; unit u = run u mod G of row u / G (a run is 8 blocks, 48 bytes, three aligned 128-bit loads), a thread a unit, each run's fold to its own slot; after a barrier a thread a row sums its G slots in run order and does the row end. Chunks of 2,048 / G rows. rowbench's `flat`, the same association |

The gate/up phase. v9 gives each warp pairs (gate row r, then up row r) and lane 0 folds each pair's product under
the next pair. Both arms take the CTA's gate rows THEN its up rows as one list (rowbench's gate/up form), write the
binary16 values to shared memory, and form the products `exact_hmul(silu[g], u)` after the body's last barrier. The
products are v9's values; what changes against v9 is the order of the rows over the CTA's warps (endpass) or threads
(flat), and one CTA barrier and one pass over the pairs where v9 has none. The pair's activation is staged once in 4E
already (the rotated ln2 output in the row vector), and both bodies read it from there.

What the residual and the store read. v9's down and o rows have lane 0 load the residual at the row's start; the arms
load it in the second pass, `__ldcg`, the same value (the residual is not written in the phase).

### 2.2 Shared memory and VRAM

No byte is added to 4E's launch: the arms use the part of the dynamic buffer the row phases leave free, after the
widest row-phase vector and before the v7 area (form 5's launch constants), which the attention phase's stage covers
between row phases. q/k and gate/up: from 2 x hidden floats on (after xr and xn); flat's padded copy there, the
slots after, then gate/up's binary16 values. Down and o: flat restages in place (from the top down, in batches of 4 x
blockDim with a barrier between reads and writes); the slots after the padded vector. `pers_rc_rows` traps if its
region would pass the v7 area; `p4e_rc_layout` refuses such a shape at install. The last float used (from the row
vector), *computed* (`pers.rs::the_rc_layout_at_the_three_sealed_shapes`); the v7 area starts at 14,880 floats on the
4B and the 8B (59,520 B), 17,408 on the 14B:

| shape | arm | q/k | gate/up | down | o | fits |
|---|---|---|---|---|---|---|
| 4B (2,560 / 9,728, 4 groups, 142 CTAs) | endpass | 7,168 | 7,237 | 11,828 | int4 | yes |
| 4B | flat | 9,744 | 9,813 | 11,828 | int4 | yes |
| 8B (4,096 / 12,288, Tetra o) | endpass | 10,240 | 10,327 | 14,400 | 6,168 | yes |
| 8B | flat | 14,360 | 14,447 | 14,400 | 6,168 | yes, 433 floats to spare |
| 14B (5,120 / 17,408, 5 groups) | endpass | 12,288 | 12,411 | 19,548 | int4 | NO: refused by name at the down phase |
| 14B | flat | 17,436 | … | … | int4 | NO: refused by name at the q/k phase |

4E v9's dynamic shared memory stays 85,952 B a CTA on the 4B and the 8B (96,064 B on the 14B), under sm_89's 101,376.
VRAM: no device buffer is added; every buffer the arms read is one v9 holds. This run's bench runtime relays the int4
records beside the file's words as every `LLVQ_PERS_4E` run does (0.385 GB, large-4e-e1); the served door does not.

### 2.3 Stage 1, run on the Mac at $0 (*measured* on the host twin, 2026-11-13)

The row gate (`tests/rowcand_gate.rs::the_rc_bodies_keep_what_they_owe`, command 'e'): `pers_rc_rows`, the function
the arms call, on the recorded words of the corps gate (coverage, fixture and random sites at 2,560, 4,096, 9,728 and
their own shapes, and the stride-6 site whose odd rows take the u32 reader), in three placements (the padded copy
beside the staging, in place, gate/up pairs as one list), three grids (3 x 96, 2 x 64, 5 x 32): 9,168 rows a slot
count, at 2,048 slots and at 96 slots (several chunks a slice).

| body | slots | rows | y equal | f32 equal | max rel err | q99 | verdict |
|---|---|---|---|---|---|---|---|
| endpass | 2,048 | 9,168 | 9,168 | 9,168 | 0 | 0 | the bits |
| endpass | 96 | 9,168 | 9,168 | 9,168 | 0 | 0 | the bits |
| flat | 2,048 | 9,168 | 9,153 | 6,693 | 5.915e-7 | 2.438e-7 | within 1e-5 |
| flat | 96 | 9,168 | 9,153 | 6,693 | 5.915e-7 | 2.438e-7 | within 1e-5 |

flat's figures are rowbench's flat on the same words (5.915e-7, 2.438e-7): the same association.

The token chain (`tests/pers_matches_served.rs`, the host twin of 4E: two layers at the 4B's widths, a Tetra then an
int4 down). `v9ep` is in `the_4e_e1_arms_are_the_served_chain` with every arm that owes the bits: tokens, residual and
both layers' KV caches equal the served chain byte for byte on five grid and RoPE configurations (three of them on
the 512-thread symbols that carry the bodies, 3 x 512 included; two on the 768-thread symbols, v9's rows), from
position 270 across the split at 256, over two launches through one state, and the stamped launch equal to the token
launch. `v9fl` (`the_4e_rc_flat_arm_stays_near_the_served_chain`), on six commands
under a smooth SiLU table (the twin's random table maps a one-ULP change of a gate value to an unrelated activation):
tokens equal 20 of 20; binary16 state values differing 8,469 of 1,522,688; largest |difference| over the state's
rms 5.0e-2 (the run from position 270), 1.1e-2 to 1.2e-2 on the five others; the test holds it under 0.1; the
stamped launch equals the token launch.

Mutants, each applied at exactly one site (asserted), each killed:

| mutant | where | killed on | rows or chain moved |
|---|---|---|---|
| `rc_run_last_block` | the run fold drops block 7 of every run | flat (row gate) | 3,003 of 9,168 rows not bit equal, max 1.08 |
| `rc_quad_last` | the 128-bit reader drops the last quad of every row | flat | 9,009, max 0.77 |
| `rc_u32_tail` | the u32 reader drops the last two words of a row | flat | 2,634, max 0.51 |
| `rc_gather_boundary` | the gather takes each run's slot from the next | flat | 9,168, max 1.37 |
| `rc_flat_tail_at` | flat reads the tail at the natural position | flat | 3,003, max 2.5 |
| `rc_pad_stride` | the restaging pads every 16 blocks | flat | 3,003, max 3.5 |
| `rc_repad_one_pass` | the in-place restaging in one pass | flat | 2,700, max 3.2 |
| `rc_slots_on_copy` | flat's slots on its padded copy | flat | 6,494, max 25.9 |
| `rc_endpass_last_row` | the second pass skips a chunk's last row | endpass | 290, max 11.5 |
| `rc_endpass_skip_tail` | the row end drops the binary16 tail | endpass | 9,168, max 3.3 |
| `gu_product` | gate/up: the product's operands swapped | v9ep (chain), v9fl (0 of 3 tokens) | chain |
| `gu_matrix` | gate/up: the up rows read from the gate matrix | v9ep, v9fl (0 of 3) | chain |
| `down_residual` | down: the CTA's first residual on every row | v9ep, v9fl (0 of 3) | chain |
| `qk_region_on_xn` | q/k: the region over xn, which the v rows read | v9ep, v9fl (0 of 3) | chain |
| `down_not_in_place` | down: flat's copy after the staging, its slots past the v7 area | v9fl: the guard traps | process |
| `o_residual` | the 8B's Tetra o: the CTA's first residual on every row | v9ep, v9fl (0 of 3) | chain |
| `gu_last_pair` | gate/up: the CTA's last pair dropped | v9ep | chain |

The rows-moved counts exclude nothing: rows of fewer than 8 blocks have no block 7, so a fold mutant cannot move
them. `gu_last_pair` SURVIVES the flat chain's distance (3 of 3 tokens equal, 5.3e-2 of rms): a defect of two rows a
CTA in one phase is under the flat chain's own noise in a two-layer twin. The call sites are one text in both arms
(the edits differ by the define only), so it is killed on v9ep. The flat chain gate is coarse; the row gate is
where flat's bound is checked.

Gates of the commit, on the Mac (*measured*, 2026-11-13): `cargo test -p llvq-cuda --no-fail-fast`, 34 suites, 298 passed, 0 failed, 66 ignored;
`cargo test -p llvq-llm --no-fail-fast`, 86 suites, 541 passed, 0 failed, 22 ignored; `cargo clippy --all-targets` clean on the Mac; `CUDARC_CUDA_VERSION=12040 cargo clippy
--target x86_64-unknown-linux-gnu -p llvq-cuda --all-targets` clean; `ops/check-linux-on-mac.sh` (clippy of
llvq-llm `--features cuda` for Linux: `fusedrun`, `persist4e_cuda.rs`) clean.

The modules at the card's tile define (`dump_tree_modules`): E1's v9 253,722 B `d55a097b…`; v9ep 264,761 B
`f4166ccf9d857b30cae8f7691e9e4f83c2726218cb48c29dc512cb42b1a668dd`; v9fl 264,761 B
`9925b4ad769c791d2372d52a14b68e650a549ce15edd05eecae0dececfa4f03d`. Under the devel nvcc 12.4 for sm_89
(`ops/tree-sass-probe.sh`), registers, stack and spill:

| module | tokens_t512 (timed) | logits_t512 | stamps_t512 | 768-thread tokens | 768-thread stamps |
|---|---|---|---|---|---|
| E1 v9 | 128, 0 B, 0 B | 128, 0, 0 | 128, 32 B, 32/44 B | 80, 120 B, 208/452 B | 80, 208 B, 332/804 B |
| v9ep | 128, 0 B, 0 B | 128, 0, 0 | 128, 40 B, 36/56 B | 80, 128 B, 244/484 B | 80, 224 B, 336/780 B |
| v9fl | 128, 0 B, 0 B | 128, 0, 0 | 128, 56 B, 44/68 B | 80, 120 B, 236/504 B | 80, 208 B, 332/812 B |

(registers; stack frame; spill stores/loads.) The timed and logits symbols do not spill. The stamp symbols spill a
little more than E1's v9 (they run outside the timed rounds, as in E1). The 768-thread symbols spill in all three, as
in v9; the runtime refuses a symbol with local memory and takes 512 threads. The card's driver ptxas compiles NVRTC's
PTX and decides; the install prints its counts.

## 3. What runs

One image built from the commit that carries the arms. One job, l40sx1, `ops/jobs/recherche-debit-jeton.sh`, output
`/out/recherche-debit-jeton-2026-11-13`, each step its own process:

1. **Oracle** on Qwen3-0.6B (hard rule 10). The sealed 4B (`sealed-4b-2026-09-23/qwen3-4b-sealed.bin`, 1,418,224,685 B,
   sha256 `886391a8…`) and the v9 config checked by size and sha256.
2. (0) The served door: `LLVQ_CONFIG=<v9 config> LLVQ_CONTEXTS=5 fusedrun <sealed> 64`, under the 100 ms sampler.
3. (a) `LLVQ_PERS_4E=256:v9,v9ep,v9fl,v9 fusedrun <sealed> 256` with the served flags on the command line
   (`LLVQ_FUSED_LAYOUT=tetra48 LLVQ_EMBED=q4 LLVQ_ROT_SHARE=1 LLVQ_FUSE=0`), under the sampler. Three modules in one
   process. At the 5-token seed prompt, then at 8,192: one eager prefill; the eager path's 256 tokens; each arm's 256
   tokens; 2 discarded rounds and 5 timed, each round in this order: v9, v9ep, v9fl, v9 AGAIN (the same runtime), then
   the eager path. A rate is (256 - 1) / the decode's host wall, launch to sync (`fusedrun`'s 4E rate, decode only,
   not the door's prefill). Then one stamped launch an arm and one logits launch an arm.
4. (b) `LLVQ_PERS_4E=256:v9` and `LLVQ_PERS_4E=256:v9fl`, one process each, under the sampler: peak VRAM, and the
   logits fingerprints of a second process.

## 4. The score

An arm's score at a context is its speed over v9 timed FIRST, round by round (t_v9 / t_arm), median [range], in the
same process. Both contexts decide, each for itself: a body may help at one context and not the other.

## 5. The rule F1 to F2 (the R3 note, on tok/s)

- **F1.** At each context, v9 timed last over v9 timed first, round by round, must have a median inside
  [0.98, 1.02]. Its round-by-round range [lo, hi] is that context's noise floor.
- **F2.** An arm is FASTER at a context only if its whole round-by-round range over v9 lies above hi; SLOWER if its
  whole range lies under lo; NOT SEPARATED otherwise. If F1 fails at a context, that context's SCORE lines describe.
- There is no F3 here: the token bench has no set of journal arms of known order. The F1 control, the eager path in
  every round and the v9 module's rate against E1's earlier readings are what this run has.

## 6. Gates

Each voids the run:

1. The image builds; `oracle` prints MATCH; the file and the config at their bytes and sha256; no `LLVQ_*` variable in
   the job environment.
2. The served door runs the served 4E v9 module (252,757 bytes, `1b422b62…`).
3. The three modules loaded are the Mac's: `d55a097b…`, `f4166ccf…`, `9925b4ad…`.
4. v9 and v9ep give the eager path's 256 tokens at both contexts (`fusedrun` voids the run otherwise: v9 is the
   reference of every ratio, and v9ep owes the bits).
5. Every arm keeps its own tokens in every round (`fusedrun` voids the run otherwise).
6. F1 and four SCORE lines are printed (two contexts, two arms).

Each voids only its own line: an arm that does not load (`not run`, its reason printed: a spill on the card's ptxas,
or the layout refused); v9ep's logits that are not v9's bit for bit void v9ep's SCORE lines (it owes the bits).

## 7. What the run reports

- Per context: each arm's tok/s median [range], over eager and over every earlier arm round by round, v9 again.
- `F1` a context, `SCORE` an arm and a context: speed over v9, median [range], what it owes, FASTER / NOT SEPARATED /
  SLOWER against the floor.
- v9fl's tokens: how many of 256 equal the eager path's and v9's, and the first divergence.
- Logits (256 positions x 151,936): v9ep against v9, bit for bit or not. v9fl against v9 on the positions whose input
  tokens are v9's (all 256 if its tokens are v9's): the values that differ and the largest |dlogit| (binary16 read as
  f32), with its position.
- One phase table an arm a context (the stamps), the install lines (registers, local bytes, shared bytes, residency,
  the rc regions, the Tetra matrices off 16 bytes), the peaks of the sampler.

Labels: every rate *measured*; the ceiling of §1 and the layout of §2.2 *computed*.

## 8. Paired perplexity for v9fl: not in this run (item 4 of the step)

The mandate's tolerance for an off-bit form is +-0.12 % of paired perplexity (docs/ETAT.md §4, K9). The path that
scored the grouped and the kernel attention (`fusedrun LLVQ_GQA_PPL=<ctx>,<windows>[,kernel]`) runs the EAGER model
twice in lockstep, teacher-forced (`Qwen3::decode_nll_split`), and pairs the windows' NLL. It never runs 4E: the kernel
form's score holds for 4E only because 4E owes the eager kernel path's bits. v9fl does not; its rows exist only in
its module. So the path cannot score v9fl as it is.

The smallest addition (*estimated*, not built): (1) one more edit of the rc arms' modules, under the logits symbol
only: the next token read from the ring the host pre-fills (`tok = L.ring[t + 1]`, the forced tokens; the ring's slot
t + 1 is written only after token t + 1's barriers, so the read is ordered), no struct change; (2)
`Runtime4e::decode_logits_forced`; (3) a mode `LLVQ_PERS_4E_PPL=<ctx>,<windows>:<arms>` that runs each window from
its first token, v9's logits module and v9fl's in lockstep, NLL from the binary16 logits in f64 on the host, and
pairs the windows with `paired_ppl_pct`; (4) a twin test: the forced launch equals the served chain fed the same
tokens (the twin's verify run, variant 32, already forces tokens on the served side). About 200 to 300 lines with
tests. On the card, at K9's 4B setting (32 windows of 1,280): about 82,000 4E tokens over two arms at 4.2 to 4.6 ms,
about 6 min, plus two module loads (about 3 min), the logits' transfer (about 12 GB an arm) and the host's
log-softmax over 151,936 rows a position (under 1 min): about 12 min, **$0.36** (*estimated*), $0.60 at a 20-minute
cap. K9's 4B interval at that setting was 0.09 % wide, under the 0.24 % of the band.

## 9. Cost

*Estimated* from the two jobs that ran E1's mechanism (`docs/data/jobs.csv`): large-4e-e1 (6af46f1d), 19 min billed for
7 module loads in 5 processes, $0.57; large-attn-ft-4e (6af4f42b), 16 min for 5 loads in 4 processes, $0.48 (its
three-arm process: 457 s, 351 s of them the loads). Here 6 loads in 4 processes, about 100 s a load. The decode: 256
tokens an arm (4 times theirs) on 2 contexts (half of ft's), 4 timings of 4E and the eager path a round, 7 rounds:
about 4 min in (a) with its two prefills at 8,192, and about 2 min in each one-arm process of (b). About 20 min,
**$0.60**. Cap: 30 min, **$0.90**.

## 10. Signed predictions (main session)

Signed on 2026-11-13, before the stamp, from the corps journal, K5, K10 and the arms' text; the hand-back's row
gate, chain tests and pins were rerun by the main session first (the same figures). Scores are speeds over v9 timed
first, round by round, median. Each line is scored in the journal as in or out.

| # | quantity | point | interval | why |
|---|---|---|---|---|
| 1 | F1 passes at both contexts; the floor's width at 5 tokens | passes, 0.01 | 80 %; width under 0.03 | one runtime twice a round; 5 rounds only |
| 2 | v9 at 5 tokens, tok/s | 222 | [215, 228] | E1's readings of the same module |
| 3 | both arms load on the card (no local memory under the driver's ptxas, the layout accepted) | holds | 80 % | the devel count has 0 B on the timed symbols |
| 4 | v9ep's 256 tokens are v9's and its logits bit for bit, both contexts | holds | 92 % | the twin's chain is byte for byte |
| 5 | v9ep at 5 tokens | 1.01 | [0.98, 1.04]; FASTER under F2: 35 % | x1.053 alone, on the short rows only; one more barrier and pass on gate/up; K5 |
| 6 | v9fl at 5 tokens | 1.03 | [0.97, 1.08]; FASTER under F2: 50 % | x1.185 alone is x1.07 on the token at most; the padded restaging, the barriers and K5 take from it |
| 7 | v9fl at 8,192 | 1.02 | [0.98, 1.06] | the same rows in a longer token |
| 8 | v9fl over v9ep at 5 tokens (ratio of medians) | above 1 | 70 % | the order alone |
| 9 | v9fl's tokens equal v9's, 256 of 256, at the 5-token prompt | holds | 65 % | errors of 1e-6 on rows; a greedy near-tie in 256 steps is the risk |
| 10 | v9fl's largest |dlogit| against v9 | 0.06 | [0.015, 0.5] | a few binary16 steps at the logits' magnitude |
| 11 | peak VRAM of the v9fl process against the v9 process | equal to the sampler's step | 90 % | no device buffer added |
| 12 | in v9fl's stamps, the gate/up wall is under v9's | holds | 70 % | the body that won alone |

What would surprise: an arm SLOWER than v9 by more than 3 % (the body would cost the token more than it gives the
rows); v9fl above 1.08 (more than the ceiling computed from the rows alone); v9fl's tokens leaving v9's within the
first 32.

## 11. Not established by this run

- v9fl's paired perplexity, hence whether it is inside the mandate's +-0.12 % (§8). Its tokens and logits against
  v9's are a description, not that tolerance.
- Any card but the L40S; any model but the 4B (the 8B fits the layout and is not run; the 14B is refused).
- The door's tok/s with prefill (the rate here is the decode), and the served memory (the bench runtime holds the int4
  bytes twice).
- Why an arm is fast or slow beyond the phase walls of its stamped launch.
- Whether the gate/up order (gate rows then up rows) or the body is the cause of a gate/up difference: the run changes
  both at once.
