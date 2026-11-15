# Preregistration. Decode-speed research phase O13: the int8 KV cache read by 4E (arm v9k8), its load check, then the token at 5 and 8,192

Written on 2026-11-14 on branch `recherche/debit-kv-2026-11-14`, once the code it describes existed and its Mac gates
had run (§3). Timestamped before the first measurement, then never edited; a departure goes to
`preregistration-recherche-debit-kv8-2026-11-14-ECARTS.md`.

One job, launched by the operator: `ops/jobs/recherche-debit-kv8.sh`. Cost: about $0.52 (*estimated*, §7). Cap: $0.78
(the 26 min timeout, *computed* at $1.80/h). Spent before it: $5.31 of the phase's $10; registry $346.95
(`docs/mesures/recherche-debit-kv-ladder-2026-11-14.txt`, last lines).

## 1. The question

The served 4B decodes 160.5 tok/s after an 8,192-token prompt. Its token takes 6,211 µs there, of which the attention
phase takes 2,646 µs (*measured*, `large-4e-hors-poids-etude-2026-11-12.txt`). That phase reads the KV cache in
binary16, 147,456 B a position.

An int8 cache with a scale and a bias a group of 64 holds 78,336 B a position, 0.531 of binary16. As a round trip on
the decode path it stays inside the ±0.12 % tolerance at 8,192: +0.0126 % [−0.0485 ; +0.0737] (*measured*, job
6af8a0bd). No kernel read it.

The question has two parts:
1. Does a 4E module whose attention reads the int8 cache load on the card, at 128 registers and 0 B of local memory?
2. If it loads, is its token faster than v9's at 8,192, and not slower at 5, by the rule F1 to F2 of
   `preregistration-recherche-debit-jeton-2026-11-13.md` §5?

That attention time follows the bytes read is a conjecture (CERVEAU P3). This run tests it on one tile.

## 2. What the code holds

Commits `5fbb8f5`, `07aa7c5`, `a899a36`, `45b5acd`; this prereg and the job script in the commit after them. Nothing
served moves: `llvq_pers.cuh`, `pers_4e.cu`, the served v9 module and E1's v9 reference keep their texts and sha256.

The arm is `v9k8`, one more E1 arm (`pers::P4E_E1_ARMS`), named inside `LLVQ_PERS_4E`. Its module is E1's v9 with
three edits (`pers::p4e_k8_edits`): the text of `kernels/pers_4e_k8.cuh` inserted, v9's append call and v9's attention
call replaced by the int8 ones. Module at the card's tile: 275,313 bytes, sha256 `8b53b1cd…c224f9`.

- **Layout.** A KV head holds one block of 1,088 B for each 8 positions: 8 rows of 128 codes, then 8 rows of
  (scale 0, bias 0, scale 1, bias 1) in binary16. 136 B a position and head.
- **Quantizer.** `kvq`'s operations in its order, each a round to nearest even. One device function writes every
  byte, in the append of a decode step and in the fill after a prefill. The codes are those of the measured `k8v8`
  round trip, bit for bit.
- **Value read.** `fma(scale, code, bias)` in f32, not rounded back to binary16. The measured round trip rounded it
  to binary16. The two differ by at most half a binary16 step a value.
- **Attention.** Section 10's steps, units, splits, folds and tickets, op for op. A step stages 544 chunks of 16 B
  where v9 stages 1,024, in the same ring.
- **The bench process.** The eager prefill writes the binary16 caches. `pers_kv8_fill` quantizes them into the arm's
  own int8 caches before the first launch of a context, outside the timed rounds. The binary16 caches stay allocated
  for the eager arm of every round.

Consequence for memory: in this job the arm's peak VRAM is ABOVE v9's, by about 0.66 GB at 8,192 (*computed*). A form
that frees the binary16 cache after the fill is not written. This job establishes no VRAM gain.

## 3. Stage 1 on the Mac at $0 (*measured*, 2026-11-14, one job at a time through the guard)

- `kv8_ref` (11 tests) and `kvq::the_k8_arms_codes_are_kvqs`: the reference's codes are candle's q8 codes.
- `pers_matches_served`, 74 of 74: the fill and the append write the reference's bytes; the int8 attention phase
  gives the reference's rows at 9 lengths and 3 grids, groups of 4 and 5; the arm's whole token loop equals a served
  chain on the int8 cache, at the three RoPE forms and past position 256.
- `pers::the_k8_module_is_v9_with_the_int8_cache`, `the_k8_attention_is_section_10_with_the_int8_reader`,
  `the_k8_launch_rule`, `the_k8_module_keeps_its_bytes`: pass, rerun by the main session.
- 19 mutants applied by the tests, each at exactly one site, all killed; 4 more on committed files, all killed.
- `cargo clippy` clean for llvq-cuda and llvq-llm on the Mac and through `ops/check-linux-on-mac.sh`.

Not verified on the Mac: that the module compiles as CUDA, its register count, its local memory. The Mac's ptxas gave
0 B to seven modules the card later refused (O11).

## 4. What runs

`ops/jobs/recherche-debit-kv8.sh` (l40sx1, sealed 4B `886391a8…`, output `/out/recherche-debit-kv8-2026-11-14`):
1. the oracle; the file and the config at their bytes and sha256;
2. the load check `LLVQ_PERS_4E=check:v9,v9k8`; the job stops here if v9k8 is refused;
3. the served door (the served v9 config, 5-token prompt, 64 tokens);
4. `LLVQ_PERS_4E=256:v9,v9k8,v9` at a 5-token prompt and at 8,192: 2 + 5 rounds, each v9, v9k8, v9 again, then
   eager, 256 tokens; the phase stamps of v9 and of v9k8;
5. v9k8 alone under the VRAM sampler.

## 5. Gates and the reading, fixed before the run

Each voids the run: the oracle does not match; a file or the config is not at its sha256; a module's digest is not the
Mac's (`d55a097b…` for v9, `8b53b1cd…` for v9k8); v9 does not load; the served door does not run the served v9 module
(252,757 bytes, `1b422b62…`); v9 does not give the eager path's 256 tokens at a context; v9k8 does not keep its own
tokens in every round; F1 or a SCORE line is missing.

| what the card says | what it means | what follows |
|---|---|---|
| v9k8 REFUSED | the int8 tile as written does not fit 128 registers at 0 B | the job ends; a lighter tile may be proposed under its own prereg, or the lever closes; the operator decides |
| loads; FASTER at 8,192 and not SLOWER at 5 | reading fewer KV bytes buys token time in 4E | v9k8 is a candidate: the quality controls through the arm and the form that frees binary16 come next, each under its own prereg and go |
| loads; NOT SEPARATED at 8,192 | the attention phase is not bound by the bytes read on this tile | no speed is claimed; what int8 keeps is the cache's bytes; the operator decides |
| loads; SLOWER at 8,192, or SLOWER at 5 | the int8 read costs more than it saves | the arm is not proposed as written |

Reported and not a gate: v9k8's tokens against the eager path's and v9's (how many of 256, the first divergence); its
logits against v9's on the positions whose input tokens are v9's; the attention phase's µs for v9 and v9k8 at both
contexts; the peaks of the sampler. The arm owes the int8 cache's values, not v9's bits: a divergence of tokens is
expected and voids nothing.

No quality is established by this job. The `k8v8` ladder figure was read on values rounded to binary16; the arm's
values are not rounded. No harness drives 4E teacher-forced, so no paired perplexity through the arm exists. Before
any served format: the 47 near-tie texts and GSM8K-200 through the arm, and the prefill path.

## 6. Labels

Every rate, phase time and peak of this job: *measured*. The cache bytes of §1 and §2: *computed*.

## 7. Cost

*Estimated* from measured times. The void jeton run (job 6af70da3, 923 s) held the oracle, the door, a main process
of three modules timing two arms at two contexts, and two one-arm processes. This job loads two modules (about
−110 s), times one arm more (about +120 s), holds one one-arm process (about −120 s) and adds the check (v9 105.1 s
and an arm 109.5 s in job 6af85b14, about 215 s). About 1,030 s: 17 min, $0.52. Timeout 26 min: $0.78 at worst. A
refusal at the check ends the job near 5 min, $0.15.

The image is rebuilt from the stamped commit before the launch (the Space builds for free).

## 8. Signed predictions (main session)

1. v9k8 loads at 0 B of local memory: 55 %. If refused, the check shows 8 B: 60 %.
2. v9k8 takes 110 s [95, 135] in the check. The job runs 1,030 s [850, 1,350] if the arm loads.
3. F1 holds at both contexts: 85 %.
4. At 8,192, v9k8 over v9: 1.11 [1.03, 1.20] in tok/s, read FASTER at 70 %. The attention phase goes from about
   2,650 µs to 1,950 µs [1,500, 2,500].
5. At 5 tokens, v9k8 over v9: 0.995 [0.98, 1.005], read NOT SEPARATED at 75 %, SLOWER at 20 %.
6. v9k8's 256 tokens differ from v9's somewhere at 8,192: 85 %. At 5 tokens: 50 %.
7. The one-arm peak of v9k8 is above 6.0 GB (the binary16 caches beside the int8 ones).

## 9. Not established by this run

- Any quality figure through the arm. Any VRAM gain.
- The prefill path, the 8B and the 14B, any text but the bench prompt.
- Any card but the L40S, any driver but 580.178.04.
- Whether int8 with a different tile (wider positions a step, codes read as words) would do better.
