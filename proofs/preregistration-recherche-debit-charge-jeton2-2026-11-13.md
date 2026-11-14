# Preregistration. Decode-speed research phase, R6b: which lighter body loads on the card, then the token measurement of R6 on the arms that load

Written on 2026-11-13 on branch `recherche/debit-2026-11-12`, once the code it describes existed and its Mac gates
had run (§3). Timestamped before the first measurement, then never edited; a departure goes to
`preregistration-recherche-debit-charge-jeton2-2026-11-13-ECARTS.md`.

Two jobs, each launched by the operator: `ops/jobs/recherche-debit-charge.sh`, then, depending on its result,
`ops/jobs/recherche-debit-jeton2.sh`. Cost: about $0.18, then about $0.63 (*estimated*, §7). Caps: $0.21 and $0.84
(the timeouts, *computed* at $1.80/h). Spent before them: $0.76 of the phase's $5 and $19.98 of the $25 cap
(`docs/mesures/recherche-debit-jeton-2026-11-13.txt`, last line).

## 1. The void run and its cause

The R6 job (6af70da3, prereg `preregistration-recherche-debit-jeton-2026-11-13.md`, sha256 d8da874a) timed no
candidate. Under the driver's ptxas (580.178.04), 4E's timed token symbol `pers_4e_tokens_f1_t512` held 128 registers
and 8 B of local memory in v9ep and in v9fl. v9 holds 128 registers and 0 B. The runtime refuses a symbol with local
memory, and that rule stays. On the Mac, the devel nvcc 12.4 gave 0 B to the same texts. So the Mac's count could not
see the problem.

The symbol is at the cap: 512 threads with one CTA a SM leave 128 registers a thread. v9 fits exactly. Each body needs
a few more live values in its phase, and the driver's allocator puts the excess on the stack.

The job then ran 923 s for a question it could no longer answer: the refusal came 339 s in, after the model load and
three NVRTC compiles. That is a flaw of the job script, corrected here: the load check comes first and stops the job.

The run did establish two facts in the token (*measured*):
- F1 passes with a floor 0.06 % wide at 5 tokens and 0.05 % at 8,192.
- v9 decodes 225.8 and 160.5 tok/s.

## 2. What the code holds

Commit: `62aadfb` (the code at `59f1900`; gates, pins and digests at `62aadfb`) (after 4358021, the void run's journal). Nothing served moves. The served
unit and the served v9 module keep their pins. E1's v9 reference, v9ep and v9fl keep their texts and sha256: the edits
of bodies 1 and 2 are unchanged (`pers.rs::p4e_rc_edits_12`).

### 2.1 The variants (`llvq-cuda/kernels/pers_4e_rc2.cuh`, four more E1 arms)

Each variant sheds live values from the timed symbol's row phases in a different way. Each is inserted after
pers_4e_rc.cuh's text, with the same four call sites renamed `pers_rc2_rows` (q/k, gate/up gate rows then up rows,
the Tetra down, the 8B's Tetra o).

| arm | body | what it changes against its body | owes |
|---|---|---|---|
| `v9ep2` | ep2 | endpass's first pass with NO block in flight (`pers_tetra_row<24, false, false>`): two words a lane fewer live across the block loop | the bits (the same sum, term for term) |
| `v9ep3` | ep3 | endpass in ONE chunk of slots: no chunk loop, so neither its bounds nor the slot offset live across the row loop; the next block in flight as v9 | the bits |
| `v9fl2` | fl2 | flat in ONE chunk (the slice's rows x runs fit 2,048 slots): no chunk loop | flat's rows bit for bit, so flat's bound |
| `v9fl3` | fl3 | fl2, with each run's three quads loaded as its blocks consume them (blocks 0-1 from quad 0, 2-4 once quad 1 is in, 5-7 once quad 2 is; an empty asm with a memory clobber between): at most five words of a run live, against twelve | flat's rows bit for bit, so flat's bound |

How each body holds what it owes:
- ep2 and ep3 run v9's row (`pers_tetra_row`), with or without the next block in flight, and v9's epilogue on its value.
- fl2 and fl3 fold every block of a run from 0.0f in block order, and every run of a row in run order: flat's
  association. The row gate also compares their rows with flat's byte for byte (§3).

The one-chunk bodies need room in the slots:
- ep3, fl2 and fl3 trap if the CTA's slice does not fit one chunk.
- `pers.rs::p4e_rc_layout` refuses such a shape at install, by phase and by number.
- The sealed 4B on 142 CTAs fits. The largest slice is flat's gate/up: 138 rows x 14 runs = 1,932 slots of 2,048
  (*computed*).
- The 8B's flat gate/up does not fit (174 x 22 = 3,828), so fl2 and fl3 are refused there. ep3 fits the 8B.
- The twin's usual grids of 3 to 5 CTAs do not fit either. The one-chunk arms are twin-tested on 142 CTAs of one warp
  (§3): from 134 CTAs flat's gate/up slice fits one chunk, and 142 gives the card's own largest slice.

Shared memory and VRAM are as in the jeton prereg's §2.2:
- Each variant uses its family's region, with the same last float at every phase (`the_rc_layout_at_the_three_sealed_shapes`).
- 85,952 B a CTA on the 4B, unchanged.
- No device buffer added.

What was not tried, and why:
- The second pass as a `__noinline__` function: a call keeps a frame, and its saved registers are local memory, which
  is the refusal itself. Not built.
- The call sites' own live values (gate/up's value pointers): they are the same in every variant, so the variants
  isolate the bodies.

### 2.2 The load check (`fusedrun LLVQ_PERS_4E=check:<arms>`)

A value of the existing measurement mode `LLVQ_PERS_4E`. It is dispatched before any model is loaded and refused
beside `LLVQ_CONFIG` and beside every other mode, by the mode's own rule (`runs_alone`). No new binary.

What `persist4e_cuda::check_e1_arms` does for each arm:
- compiles the module as `install_e1` compiles it: the same text and sha256 at the card's tile define, NVRTC for the
  same arch, then the driver's JIT;
- uses one thread an arm;
- reads, at the sealed 4B's shared bytes (85,952 B): the token symbol of each RoPE form at 512 threads, and the stamp
  and logits symbols of RoPE form 1, each with registers, local bytes and CTAs a SM;
- checks an rc arm's layout at the card's grid.

The verdict is the runtime's rule, applied to RoPE form 1's token symbol: `CHECK ARM <arm> LOADS`, or
`CHECK ARM <arm> REFUSED: <why>`. Form 1 is the one the card picked in every 4B run on the L40S. The other two forms
are printed beside it.

What made the 339 s, and what the check avoids:
- the model load, about 70 s (E1's one-arm process);
- three modules compiled one after the other, about 90 s each.

The check loads no model and compiles in parallel. How long NVRTC and the JIT take each is printed for each arm. It
has never run on a card: only its Linux build is type-checked (`ops/check-linux-on-mac.sh`), and its parser tested.

## 3. Stage 1 on the Mac at $0 (*measured*, 2026-11-13, under the operator's limits: one job at a time through the guard, 3 test threads)

The row gate (`tests/rowcand_gate.rs`, command 'e' of the host twin, the corps gate's recorded words, three placements,
three grids, 9,168 rows a body):

| body | slots | y equal | f32 equal | max rel err | q99 | owes | verdict |
|---|---|---|---|---|---|---|---|
| endpass | 2,048 and 96 | 9,168 | 9,168 | 0 | 0 | the bits | PASS |
| flat | 2,048 and 96 | 9,153 | 6,693 | 5.915e-7 | 2.438e-7 | 1e-5 | PASS |
| ep2 | 2,048 and 96 | 9,168 | 9,168 | 0 | 0 | the bits | PASS |
| ep3 | 2,048 | 9,168 | 9,168 | 0 | 0 | the bits | PASS |
| fl2 | 2,048 | 9,153 | 6,693 | 5.915e-7 | 2.438e-7 | 1e-5 | PASS |
| fl3 | 2,048 | 9,153 | 6,693 | 5.915e-7 | 2.438e-7 | 1e-5 | PASS |

The one-chunk bodies are not run at 96 slots: they trap there by design. fl2 and fl3 equal flat byte for byte on the
nine runs (`the_flat_variants_are_flat_bit_for_bit`).

Mutants, each applied at exactly one site (asserted), each killed (rows not bit equal to the served row, of 9,168):

| mutant | body | rows moved, max rel err |
|---|---|---|
| `rc2_ep2_slot`, the slot one row on | ep2 | 9,168, 3.6 |
| `rc2_ep3_last_row`, the second pass skips the last row | ep3 | 290, 11.5 |
| `rc2_fl2_fold_end`, the fold one block short | fl2 | 9,168, 0.76 |
| `rc2_seq_word`, block 2 reads the wrong word | fl3 | 9,009, 0.97 |
| `rc2_seq_block4`, block 4 dropped | fl3 | 8,853, 1.26 |
| `rc2_seq_quad2`, quad 2 read from quad 1's place | fl3 | 3,003, 2.6 |
| `rc2_gather`, the gather reads the next run's slot | fl2, fl3 | 9,168 each, 1.37 |

The ten R6 body mutants are killed again with the same counts. One edit tried first is EQUIVALENT, not a survivor:
fl3's quad-2 guard moved to 3 g + 3 < nq bites only where a row's last run ends at block 5 exactly, which no recorded
shape has; it was replaced by the quad-2 mutant above.

The token chain (`tests/pers_matches_served.rs`):
- `the_4e_e1_arms_are_the_served_chain`, which now includes v9ep2: tokens, residual and both layers' caches equal the
  served chain byte for byte, on the same commands as R6.
- `the_4e_rc_one_chunk_arms_on_the_cards_grid`, 142 CTAs of 32 threads, from position 5 and from 270, 2 tokens each:
  v9ep2 and v9ep3 equal the served chain byte for byte; v9fl2 and v9fl3 equal v9fl's chain byte for byte; the
  stamped launch of v9ep3 and v9fl3 equals its token launch. v9fl against the served chain there: tokens 2 of 2 and
  2 of 2; 1,607 of 68,096 and 3,387 of 1,182,208 binary16 state values differ; the largest |difference| over the
  state's rms is 1.7e-2 and 5.5e-2. The run took 225 s and peaked at 0.61 GB.
- The grid change. 64 threads a CTA was written first. One warp a CTA halves the fibers (4,544 of 64 KB of stack,
  about 0.3 GB) and still covers every body. The grid could not shrink: one chunk needs 134 CTAs or more.
- The R6 tests pass again: `the_4e_rc_flat_arm_stays_near_the_served_chain` (tokens 20 of 20, max 5.0e-2 of rms) and
  `the_4e_rc_call_site_mutants_are_killed` (the same verdicts; `gu_last_pair` still survives the flat chain distance,
  5.3e-2, and is killed on v9ep).

Unit tests (`cargo test -p llvq-cuda --lib`): `the_rc_arms_are_v9_with_the_body_on_every_tetra_phase` (six arms),
`the_rc_layout_at_the_three_sealed_shapes`, `the_rc_modules_keep_their_bytes` (v9ep and v9fl keep f4166ccf and
9925b4ad; the four variants pinned), the served unit's and the served v9 module's pins: pass.

The modules at the card's tile define:

| arm | bytes | sha256 |
|---|---|---|
| v9ep2 | 274,808 | `a7d9a04d4bb963aac4ed831f1d7b41f6473e939dbc6c7c047d3b9e90b9801aa9` |
| v9ep3 | 274,808 | `60dfa277ef82cf5c94b5a4b0b5745610324abac432c8ef4725ea96ba150714f7` |
| v9fl2 | 274,808 | `d863288607ba961223b3ac8501b99f2e4c97dfc082cd237b80300df8b33dbd11` |
| v9fl3 | 274,808 | `713ed5877aef63da831b8d149d31418889e42ac2580589294fa303e408ccf3fe` |

The register proxy (`ops/rc-regprobe.sh`, the devel ptxas 12.4 on each module's PTX, the timed entry alone) says
NOTHING. All seven modules (E1's v9, v9ep, v9fl and the four variants) read 128 registers, 0 B of stack and 0 B of
spill at every cap tried: none, 120, 112, 104, 96. `--maxrregcount` does not act under the kernel's
`__launch_bounds__(512, 1)`. The probe therefore ranks nothing, and the devel allocator gives v9ep and v9fl the same
0 B that the card's did not. No other measure on the Mac separates the variants from v9. The card's load check
decides alone. A probe that rewrote the PTX's `.maxntid` to free the cap, or a CUDA 13 ptxas, would be the next
proxy; neither was run, and the second needs an image download the operator decides.

Gates of the commit: `cargo clippy --all-targets` clean on the Mac; `CUDARC_CUDA_VERSION=12040 cargo clippy --target
x86_64-unknown-linux-gnu -p llvq-cuda --all-targets` clean; `ops/check-linux-on-mac.sh` (llvq-llm `--features cuda`
for Linux, which compiles `check_e1_arms` and the check's dispatch) clean. `cargo test -p llvq-cuda --no-fail-fast`
34 suites, 301 passed, 0 failed, 66 ignored; `cargo test -p llvq-llm --no-fail-fast` 86 suites, 541 passed, 0 failed, 22 ignored.

## 4. What runs

**Job 1, `ops/jobs/recherche-debit-charge.sh`** (l40sx1, output `/out/recherche-debit-charge-2026-11-13`):
1. the oracle;
2. `LLVQ_PERS_4E=check:v9,v9ep,v9fl,v9ep2,v9ep3,v9fl2,v9fl3 fusedrun <sealed> 2`.

Nothing is timed. Its gates:
- the seven digests are the Mac's;
- the check reproduces the jeton run: v9 LOADS; v9ep and v9fl are REFUSED for local memory. Otherwise the check does
  not describe the runtime, and the job is void.

**Job 2, `ops/jobs/recherche-debit-jeton2.sh`**: `ARMS` set by the launcher from job 1 (§5), at most one arm a body.
1. The oracle, the file and the config.
2. The load check of `v9,$ARMS`. An arm that does not load leaves the run. If none loads, the job stops and prints
   STOP.
3. The served door (control 2).
4. `LLVQ_PERS_4E=256:v9,<arms that load>,v9`, as the jeton prereg's §3 step (a): two contexts, 2 + 5 rounds, v9
   first, the arms, v9 again, then eager.
5. The one-arm process of the flat arm only, if one loads. The one-arm v9 process is dropped.

## 5. The decision rule between the jobs

The order of each body follows how close each arm is to the body rowbench measured.

| family | order |
|---|---|
| endpass | `v9ep`, then `v9ep3` (rowbench's endpass ran one chunk on the 4B), then `v9ep2` (it drops the block in flight, which rowbench never timed) |
| flat | `v9fl`, then `v9fl2` (rowbench's flat in one chunk), then `v9fl3` (it reorders the loads) |

How the rule reads job 1:
- For each family, the first arm whose line reads `CHECK ARM <arm> LOADS` goes into `ARMS`.
- A family with no arm that loads contributes none.
- If neither family has one, job 2 is not launched.
- A variant that loads on the card is the arm, whatever the Mac's proxy said.

## 6. The score and the rule

As in the jeton prereg, §4 and §5: the score, and F1 to F2 at each context. An arm's score at a context is its speed
over v9 timed first, round by round. F1: v9 last over v9 first, median in [0.98, 1.02], its range the floor. F2:
FASTER only if the whole range lies above the floor. Its gates: §6 there. Its reports: §7 there.

Two changes against the jeton prereg:
- the load check comes first and stops the job;
- the SCORE lines number twice the arms that load.

## 7. Cost

*Estimated* from the void run's own timings (job 6af70da3, 923 s running, $0.46):
- Its oracle and the start of its first process took about 2 min.
- Its main process spent 339 s on loads (the model about 70 s, three modules about 90 s each), then about 8 min of
  decode, stamps and logits at two contexts.
- Its one-arm processes ran about 2 min each.

| job | what it holds | estimate | cap (timeout) |
|---|---|---|---|
| charge | the oracle, then seven modules on parallel threads with no model, about 2 to 4 min | about 6 min, $0.18 | 7 min, $0.21 |
| jeton2 | the check (about 3 min), the door, the main process with up to two arms timed (about 4 min more decode, two more module loads), at most one one-arm process | about 21 min, $0.63 | 28 min, $0.84 |

## 8. Signed predictions (main session)

Written on 2026-11-13 after rereading the hand-back against the repository, before the stamp. Nothing on the Mac
separates the variants (§3), so the load predictions rest on one reading only: 8 B of local memory is one spilled
64-bit value, so the excess over v9 is small.

Job 1, the load check:
1. The gate passes: v9 LOADS, v9ep and v9fl REFUSED with 8 B of local memory each. 90 %.
2. The job runs 240 s [150, 420]. It hits its 7 min timeout at 15 %.
3. `v9ep2` loads at 50 %, `v9ep3` at 45 %, `v9fl2` at 35 %, `v9fl3` at 55 %.
4. At least one of the four loads: 60 %. None loads, and the phase closes on "4E has no register room at 512
   threads": 40 %.
5. A variant that is refused shows 8 B of local memory, not more: 60 %.

Job 2, if it is launched:
6. F1 passes at both contexts, with a floor under 0.2 % wide. 90 %.
7. v9 timed first: 225.8 tok/s [224.5, 227.0] at 5 tokens, 160.5 [159.0, 162.0] at 8,192.
8. The endpass arm at 5 tokens: x1.005 [0.990, 1.020] if it is `v9ep3`, x0.990 [0.960, 1.010] if it is `v9ep2`.
   FASTER at 35 %.
9. The flat arm at 5 tokens: x1.03 [0.99, 1.07]. FASTER at 60 %. At 8,192: x1.02 [0.99, 1.05].
10. No arm reaches x1.05 at 5 tokens: 75 %.
11. The flat arm's 256 tokens are all v9's: 50 %.
12. The job runs 1,150 s [900, 1,500].

## 9. Not established by these runs

- Any tok/s if no variant loads. Whether a variant that loads keeps the speed its body had alone: ep2 drops the block
  in flight, and fl3 reorders flat's loads. Rowbench never timed either.
- The register proxy's agreement with the driver's allocator (§3).
- v9fl's paired perplexity: the jeton prereg's §8, unchanged.
- Any card but the L40S, any model but the 4B.
