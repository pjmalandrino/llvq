# Preregistration. Decode-speed research phase, R6d: three arms without the one-call gate/up site, their load check, then the token measurement of R6 on the arms that load

Written on 2026-11-14 on branch `recherche/debit-2026-11-12`, once the code it describes existed and its Mac gates
had run (§3). Timestamped before the first measurement, then never edited; a departure goes to
`preregistration-recherche-debit-jeton3-2026-11-14-ECARTS.md`.

One job, launched by the operator: `ops/jobs/recherche-debit-jeton3.sh`. Cost: about $0.75 (*estimated*, §7). Cap:
$0.96 (the 32 min timeout, *computed* at $1.80/h). Spent before it: $1.61 of the phase's $5 and $20.83 of the $25 cap
(`docs/mesures/recherche-debit-sondes-2026-11-14.txt`, last line).

## 1. The question

R6c put the endpass body on one call site at a time (job 6af837d1, *measured*). The gate/up site alone brings the 8 B
of local memory on the timed symbol `pers_4e_tokens_f1_t512`. The q/k, the Tetra down and the Tetra o sites each load
at 0 B. Whether the three load together was not checked.

The gate/up site of the six refused arms is the only one that selects between two matrices inside `row_at`. It is
also the only one that keeps values of its own across the body: the binary16 values of the pair, and `np`. That this
is what spills is a hypothesis, not a measurement.

The question has two parts:
1. Does an arm with the body on q/k, down and o, and no one-call gate/up site, load on the card?
2. If one loads, is it faster than v9 in the token, by the score and the rule of R6?

## 2. What the code holds

Commit: the code, its tests and pins at `e6892d7`; this prereg and the job script in the commit after it. Nothing served moves. The served unit, the served v9 module, E1's v9
reference, the six research arms of R6 and R6b and the five load probes of R6c keep their texts and sha256 (§3).

### 2.1 The three arms (`pers.rs::P4E_RC_GU_ARMS`, three more E1 arms)

Each arm is E1's v9 with edits taken from `p4e_rc_edits` of its body. Edit 0 inserts `pers_4e_rc.cuh` with its
define; edits 1, 3 and 4 put the body on q/k, the Tetra down and the Tetra o, as in v9ep and v9fl. Edit 2, the
gate/up site, differs.

| arm | body | gate/up site | owes |
|---|---|---|---|
| `v9fn` | flat (2) | v9's own text: edit 2 dropped | flat's bound of 1e-5 on the rows of q/k, down and o; v9's rows at gate/up |
| `v9fg` | flat (2) | two calls, one matrix a call | flat's rows everywhere: its chain equals v9fl's byte for byte |
| `v9eg` | endpass (1) | two calls, one matrix a call | the bits of the served chain |

### 2.2 The two-call site (`pers.rs::P4E_RC_GU_TWO_CALLS`)

It replaces the same v9 text as edit 2. As committed:

```
        // rc, two calls: the CTA's gate rows, silu of each parked in act; after the body's last barrier its up
        // rows, each product exact_hmul(silu[g], u), the stream's own. One matrix a call, no value kept in shared.
        const u32 glo = PERS_SEG_GU(0u), ghi = PERS_SEG_GU(1u);
        const u32 nb = A->gate.nblocks, tw = A->gate.tail_w;
        {
            const auto row_at = [&](u32 idx) {
                const u32 r = glo + idx;
                return PersTRow{A->gate.words + r * A->gate.stride_u32, A->gate.gscale, A->gate.rscale, A->gate.tail, r};
            };
            const auto epi = [&](u32 idx, float y) { A->act[glo + idx] = A->silu[f2h(y)]; };
            pers_rc_rows<PERS_RC_ARM>(ghi - glo, nb, tw, tab, t_inv, v, row_at, epi, v + 2u * H, false);
        }
        {
            const auto row_at = [&](u32 idx) {
                const u32 r = glo + idx;
                return PersTRow{A->up.words + r * A->up.stride_u32, A->up.gscale, A->up.rscale, A->up.tail, r};
            };
            const auto epi = [&](u32 idx, float y) { A->act[glo + idx] = exact_hmul(A->act[glo + idx], f2h(y)); };
            pers_rc_rows<PERS_RC_ARM>(ghi - glo, nb, tw, tab, t_inv, v, row_at, epi, v + 2u * H, false);
        }
```

Why it holds what each arm owes:
- `act` is the global binary16 scratch of the gated activation (`SegArgs::act`). A CTA writes only its rows
  [glo, ghi). No CTA reads `act` before the grid barrier that ends the MLP's gate/up part; the down phase reads it
  after that barrier. v9's own stream rests on the same two facts.
- Each body ends its chunk loop on a `__syncthreads()` after its last epilogue. So call 2 reads what call 1 wrote. In
  both bodies the thread that writes row i in call 1 is the thread that reads it in call 2.
- Gate and up share `nblocks` and `tail_w`: both read the hidden vector, and v9's stream and the one-call rc site
  already pass the gate's values for both.
- `v` is intact after call 1. Endpass never writes its input. Flat with `in_place == false` restages into
  `v + 2 hidden`, past the input's `hidden` floats, by the disjoint path of `pers_rc_repad`, which reads `v` only.
- The value is v9's: silu[f2h(g)], then exact_hmul(silu, f2h(u)). Endpass's row is v9's row, value for value
  (R6's row gate). Flat's row does not depend on its list or its chunk: a row's runs fold from 0.0f in run order.

### 2.3 Shared memory

`pers.rs::p4e_rc_layout` is unchanged. Its gate/up end adds the one-call site's binary16 values after the body's
region. The three arms keep no value there: v9fn runs v9's site, which uses no region, and the two calls end at the
body's region. The bound is conservative for them. 85,952 B a CTA on the 4B, unchanged. No device buffer added.

### 2.4 The check and the timed run

The check is R6b's (`fusedrun LLVQ_PERS_4E=check:<arms>`, `persist4e_cuda::check_e1_arms`), unchanged. The three
arms are E1 arms: the check and a timed run both name them. `p4e_rc_body` gives body 2 to v9fn and v9fg and body 1
to v9eg, so every consumer reads them as their family: what an arm owes (`p4e_e1_owes_bits`), the rc layout line, the
install checks.

## 3. Stage 1 on the Mac at $0 (*measured*, 2026-11-14, one job at a time through the guard, 3 test threads)

The text (`cargo test --release -p llvq-cuda --lib`, 9 tests by name, pass):
- `the_rc_arms_are_v9_with_the_body_on_every_tetra_phase` now covers the three arms. Each arm's edits are its body's
  edits 0, 1, 3 and 4, and at gate/up nothing (v9fn) or the two calls (v9fg, v9eg). Undoing them gives E1's v9 byte
  for byte. v9fn holds v9's gate/up text once, 3 rc calls and v9's row loop on gate/up; v9fg and v9eg hold the
  two-call site once and 5 rc calls. No module holds the one-call site's shared values.
- `the_rc_modules_keep_their_bytes` pins the three arms beside the six.
- `the_rc_layout_at_the_three_sealed_shapes`, the served unit's, the served v9 module's and the verify module's pins:
  pass.

The token chain (`tests/pers_matches_served.rs`):
- `the_4e_e1_arms_are_the_served_chain` now includes v9eg, on every command there (each RoPE form, both row
  variants, grids of 2 to 5 CTAs, 64 to 512 threads, positions past 256, two launches, the stamped launch): tokens,
  residual and both layers' caches equal the served chain byte for byte. Pass, 445 s.
- `the_4e_rc_flat_arm_stays_near_the_served_chain` now runs v9fl, v9fn and v9fg on its six commands. Pass, 158 s.
  v9fg equals v9fl's chain byte for byte on the six. Over the six, all 20 tokens equal the served chain's in each arm:

| arm | state values differing | max abs difference over rms |
|---|---|---|
| v9fl | 8,469 of 1,522,688 | 5.02e-2 |
| v9fn | 6,749 of 1,522,688 | 5.02e-2 |
| v9fg | 8,469 of 1,522,688 | 5.02e-2 |

  v9fn reads 3.8e-7 and 3.9e-6 of rms on the 4x96 grid and on the two launches, where v9fl reads 1.2e-2. A reading,
  not a measurement: on those commands the gate/up rows would carry most of flat's distance.

Mutants of the two-call site (`the_4e_rc_two_call_mutants_are_killed`, on v9eg against the served chain bit for
bit, three commands: the smooth SiLU table, the 8B's Tetra o, the random table). Each twist is asserted to match
exactly one site of the arm's replacement texts. Pass, 79 s.

| mutant | sites | verdict |
|---|---|---|
| `gu2_silu_on_up`, silu applied on the up call's value, not the gate call's | 2 | killed: the chain moved |
| `gu2_up_reads_gate`, the second call reads `A->gate` | 1 | killed: the chain moved |
| `gu2_gate_short`, the gate call on `ghi - glo - 1u` rows | 1 | killed: the chain moved |
| `gu2_up_short`, the up call on `ghi - glo - 1u` rows | 1 | killed: the chain moved |
| `gu2_no_product`, the product replaced by `f2h(y)` | 1 | killed: the chain moved |

The parser (`cargo test -p llvq-llm --bin fusedrun pers_4e_takes_a_count_and_optional_e1_arms`): `check:` takes
`v9,v9fg,v9fn,v9eg`, and the timed parser takes `256:v9,v9fg,v9eg,v9` and `256:v9,v9fn,v9`. Pass.

`cargo clippy -p llvq-cuda -p llvq-llm --all-targets` clean on the Mac; `ops/check-linux-on-mac.sh` clean.

The modules at the card's tile define (`dump_tree_modules`):

| arm | bytes | sha256 |
|---|---|---|
| v9fn | 264,720 | `ce902a755173d05c4d50353e68d7bf62a326e13566fd4f7b84bf97e9933b6f4f` |
| v9fg | 264,835 | `36b617b0811d24cb8aa475fa9c3f0cd49ced1138c918a0fbdd7556ceac1c67bb` |
| v9eg | 264,835 | `1971b46969e4579c64327852a47c49887bd2acb0ffc5be802d2bed614117c29d` |

The same dump reads every earlier module at its earlier digest: the served v9 module 1b422b62 (252,757 B), E1's v9
d55a097b, v9ep f4166ccf, v9fl 9925b4ad, the four R6b variants, the five R6c probes.

Not verified on the Mac: whether any arm loads. The Mac's ptxas gives 0 B of local memory to every module, the
refused ones included (R6b §3). No CUDA compiler ran on the three texts.

## 4. What runs

`ops/jobs/recherche-debit-jeton3.sh` (l40sx1, output `/out/recherche-debit-jeton3-2026-11-14`). The candidates are
fixed in the script, not set by the launcher.
1. The oracle, the file and the config.
2. `LLVQ_PERS_4E=check:v9,v9fg,v9fn,v9eg fusedrun <sealed> 2`, timeout 9 min. Its gate: the four digests are the
   Mac's (§3), and v9 LOADS. Otherwise the job stops.
3. The selection, by the check's lines `CHECK ARM <arm> LOADS`:
   - the flat family's arm is the first of `v9fg`, `v9fn` that loads;
   - the endpass family's arm is `v9eg` if it loads;
   - if none loads, the job stops there and prints STOP (exit 0).
4. The served door (control 2).
5. `LLVQ_PERS_4E=256:v9,<the selected arms>,v9`, as the jeton prereg's §3 step (a): two contexts, 2 + 5 rounds, v9
   first, the arms, v9 again, then eager.
6. The one-arm process of the selected flat-family arm, if there is one.

The order v9fg before v9fn: v9fg keeps flat on every Tetra phase, gate/up included, the largest share of the Tetra
bytes. v9fn is the fallback that leaves gate/up on v9.

## 5. The reading, fixed before the run

The load verdict of an arm is the check's: RoPE form 1's token symbol, `LOADS` at 0 B of local memory, `REFUSED`
otherwise.

| what the check says | what it means | what follows |
|---|---|---|
| v9fg loads | the two-call site removes the 8 B for flat | v9fg is timed |
| v9fg refused, v9fn loads | the 8 B come with any rc gate/up site, or with the three other sites together | v9fn is timed; the gate/up gain stays out |
| v9fn refused | the three sites refused together, though each loaded alone | the phase's register room is read as spent for flat |
| v9eg loads | the two-call site removes the 8 B for endpass | v9eg is timed |
| none loads | no arm to time | the job stops; the operator decides on the phase |

The score and the rule are those of the jeton prereg (`preregistration-recherche-debit-jeton-2026-11-13.md`), §4
and §5, unchanged: the SCORE lines, and F1 to F2 at each context. Its gates are §6 there and its reports §7 there.
The SCORE lines number twice the arms timed, as in R6b (`preregistration-recherche-debit-charge-jeton2-2026-11-13.md`,
§6).

An arm that loads and is FASTER is not servable by this run (§8).

## 6. What the job script changes against jeton2

- The candidates are fixed: `v9fg,v9fn,v9eg`. No `ARMS` from the launcher.
- The check names v9 and the three, with a 9 min timeout (6 min in jeton2).
- At most one arm a family, by the order of §4.
- The job's timeout is 32 min (28 in jeton2).
- `ops/Dockerfile.cuda` and `UPLOAD_ALLOW` of `ops/run.py` need nothing new: no binary and no file is added; the
  kernels are text inside `llvq-cuda`.

## 7. Cost

*Estimated* from measured times:
- The void jeton run (job 6af70da3, 923 s running) held what this job holds beside the check: the oracle, the door, a
  main process that loaded three modules (339 s with the model) and timed v9 and its F1 control at two contexts, and
  two one-arm processes.
- The check, from jobs 6af82af3 and 6af837d1, one arm after the other: v9 101.6 s, a flat arm 125.9 to 127.2 s, an
  endpass arm 102.4 to 104.0 s. Four arms: about 102 + 127 + 127 + 104 = 460 s. Its 9 min timeout leaves 80 s.
- Two arms timed add about 4 min of decode (jeton2's estimate). One one-arm process fewer saves about 2 min.

| what | estimate | cap |
|---|---|---|
| 923 + 460 + 240 - 120 s | about 1,500 s, 25 min, $0.75 | 32 min, $0.96 |

If no arm loads, the job ends after the check: about 10 min, $0.30.

## 8. Not established by this run

- Any card but the L40S, any driver but 580.178.04.
- The 8B and the 14B. The two-call site was not checked on their shapes on the card.
- Whether an arm that loads is servable: the served path, the 8B's Tetra o, the paired perplexity of a flat arm (the
  jeton prereg's §8) are not run.
- Why the one-call site spills. The arms test a rewrite, not the cause.

## 9. Signed predictions (main session)

Written before the stamp. The three earlier sets of this phase: R6b's were wrong on the main point (60 % that a
variant loads, none did), R6c's six were in.

1. v9 loads. 97 %.
2. v9fg loads: 40 %. v9fn loads: 55 %. v9eg loads: 45 %. None of the three loads: 30 %.
3. A refused arm shows 8 B of local memory. 70 %.
4. In the check a flat arm takes 127 s [110, 150], an endpass arm 103 s [90, 125].
5. If v9fg is timed, its score at a 5-token prompt is x1.03 [1.00, 1.06] of v9. It reaches x1.05 at 20 %.
6. If v9fn is timed: x1.01 [0.99, 1.03]. If v9eg is timed: x1.00 [0.98, 1.02].
7. Over the whole job, an arm reaches x1.05 at a 5-token prompt: 8 %. An arm is separated above v9 from F1's floor
   at a 5-token prompt: 30 %.
8. v9eg, if timed, gives v9's tokens at both contexts: 95 %. A flat arm, if timed, gives v9's tokens at both
   contexts: 80 %.
9. The job runs 1,500 s [1,100, 1,800] if an arm is timed, 600 s [520, 700] otherwise.

The ceilings these rest on are computed, not forecasts: x1.07 on the token for flat on every Tetra phase, x1.02 for
endpass (`docs/mesures/recherche-debit-corps-2026-11-13.txt`). The two-call site adds barriers and one restaging
that the rows-alone bench did not pay.
