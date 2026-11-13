# Preregistration. Decode-speed research phase, R6b: which lighter body loads on the card, then the token measurement of R6 on the arms that load

DRAFT, not yet ready to stamp: the Mac gates of §3 have not run (the operator's order of 2026-11-13: no heavy work on
the Mac until he says so). §3 lists what is missing. Once the gates run, their figures replace the NOT RUN lines and
the four module digests are filled in, in this file and in both job scripts. Then the main session signs §8. Only
then is it timestamped and never edited; a departure goes to
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

Commit: NOT YET (on the working tree of `recherche/debit-2026-11-12`, after 4358021). Nothing served moves. The served
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
- The twin's usual grids of 3 to 5 CTAs do not fit either. The one-chunk arms are twin-tested on 142 CTAs (§3).

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
has never run on a card.

## 3. The Mac gates: NOT RUN

Nothing below has run. The operator's order came while the first probe was compiling; that probe was stopped (the
`ops/rc-regprobe.sh` run of 4e-v9e1, 4e-v9ep and 4e-v9fl, in Docker, before any result).

The Rust and kernel text of R6b has not been compiled. Each line becomes a figure before this prereg is stamped.

1. `cargo test -p llvq-cuda --lib`: `the_rc_arms_are_v9_with_the_body_on_every_tetra_phase` (now six arms),
   `the_rc_layout_at_the_three_sealed_shapes` (now the one-chunk rules), `the_rc_modules_keep_their_bytes` (v9ep and
   v9fl must keep f4166ccf and 9925b4ad). The four variants' digests are to be added to this test and to the scripts.
2. The row gate, `tests/rowcand_gate.rs`:
   - `the_rc_bodies_keep_what_they_owe`: six bodies at 2,048 slots, bodies 1 to 3 at 96 slots, 9,168 rows each.
     Expected: ep2 and ep3 bit for bit; fl2 and fl3 at flat's 5.915e-7.
   - `the_flat_variants_are_flat_bit_for_bit`
   - `the_rc2_body_mutants_are_killed`, seven mutants:

     | mutant | body | the edit |
     |---|---|---|
     | `rc2_ep2_slot` | ep2 | the slot one row on |
     | `rc2_ep3_last_row` | ep3 | the second pass skips the last row |
     | `rc2_fl2_fold_end` | fl2 | the fold one block short |
     | `rc2_seq_word` | fl3 | block 2 reads the wrong word |
     | `rc2_seq_block4` | fl3 | block 4 dropped |
     | `rc2_seq_quad2` | fl3 | a run's last quad not loaded |
     | `rc2_gather` | fl2, fl3 | the gather reads the next run's slot |

   - The ten mutants of `the_rc_body_mutants_are_killed`, unchanged.
3. The token chain, `tests/pers_matches_served.rs`:
   - `the_4e_e1_arms_are_the_served_chain`, which now includes v9ep2;
   - `the_4e_rc_one_chunk_arms_on_the_cards_grid`: 142 CTAs x 64 threads. v9ep2 and v9ep3 must be the served chain
     byte for byte; v9fl2 and v9fl3 must be v9fl's chain byte for byte. Twice, from 5 and from 270, and the stamped
     launch. It has never run; the twin has never run 142 CTAs, so its run time is not known;
   - the R6 tests `the_4e_rc_flat_arm_stays_near_the_served_chain` and `the_4e_rc_call_site_mutants_are_killed`, as
     before.
4. `cargo test -p llvq-cuda --no-fail-fast`, `cargo test -p llvq-llm --no-fail-fast` (a test of the check's parser
   added), `cargo clippy --all-targets`, `CUDARC_CUDA_VERSION=12040 cargo clippy --target x86_64-unknown-linux-gnu
   -p llvq-cuda --all-targets`, and `ops/check-linux-on-mac.sh`. The last one is the only compiler that sees
   `check_e1_arms` and the check's dispatch in `fusedrun`.
5. The register proxy, devel nvcc 12.4: `cargo test -p llvq-cuda --lib -- --ignored dump_tree_modules` (to extend
   with the four variants), then `ops/rc-regprobe.sh 4e-v9e1 4e-v9ep 4e-v9fl 4e-v9ep2 4e-v9ep3 4e-v9fl2 4e-v9fl3`.
   - What it prints: ptxas on each module's PTX, `pers_4e_tokens_f1_t512` alone, at its own cap of 128 and at
     `--maxrregcount` 120, 112, 104 and 96.
   - How it ranks: by the spill each module shows as the cap tightens. Below v9's spill means less pressure than v9.
   - Not yet established: whether ptxas honours `--maxrregcount` under the kernel's `__launch_bounds__`. If it does
     not, the probe reads the same at every cap and says nothing; it will say so, and the card's check decides alone.
   - The proxy is the devel ptxas 12.4. The card's is a later allocator. A CUDA 13 devel image would give the
     driver's own allocator, and it is not on the Mac.
   - Pulling that image is a download the operator decides. It is not used.

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

## 9. Not established by these runs

- Any tok/s if no variant loads. Whether a variant that loads keeps the speed its body had alone: ep2 drops the block
  in flight, and fl3 reorders flat's loads. Rowbench never timed either.
- The register proxy's agreement with the driver's allocator (§3, item 5).
- v9fl's paired perplexity: the jeton prereg's §8, unchanged.
- Any card but the L40S, any model but the 4B.
