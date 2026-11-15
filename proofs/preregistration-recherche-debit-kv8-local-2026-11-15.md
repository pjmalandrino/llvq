# Preregistration. Decode-speed research phase O13: the int8 KV arm of 4E (v9k8) timed with its 8 B of local memory

Written on 2026-11-15 on branch `recherche/debit-kv-2026-11-14`, once the code it describes existed and its Mac gates
had run (§3). Timestamped before the first measurement, then never edited; a departure goes to
`preregistration-recherche-debit-kv8-local-2026-11-15-ECARTS.md`.

One job, launched by the operator: `ops/jobs/recherche-debit-kv8-local.sh`. Cost: about $0.46 (*estimated*, §7). Cap:
$0.72 (the 24 min timeout, *computed* at $1.80/h). Spent before it: $5.44 of the phase's $10; registry $347.08
(`docs/mesures/recherche-debit-kv8-2026-11-14.txt`, last lines).

This is a prereg of its own, not a departure of `preregistration-recherche-debit-kv8-2026-11-14.md`. That prereg's
reading of a refusal names two outcomes, a lighter tile or the lever closed. The operator chose a third on 2026-11-15:
time the arm as it is.

## 1. The question

Arm v9k8 reads the KV cache as int8, 0.531 of binary16's bytes. On the L40S its token symbol holds 8 B of local memory
at 128 registers, at the three RoPE forms, and the load rule refused it (*measured*, job 6af8df0a). v9 holds 0 B in
the same check.

The load rule is "no local memory and one CTA a SM, or refused". Nine arms and one probe of O11 were refused under it, then
v9k8. None was timed, since the rule refuses before the first launch. What 8 B of local memory cost a token has no figure.

Two questions, one job:
1. With its 8 B of local memory, is v9k8's token faster than v9's at 8,192, by the rule F1 to F2 of
   `preregistration-recherche-debit-jeton-2026-11-13.md` §5?
2. At a 5-token prompt, where the cache holds almost nothing, what does the arm cost against v9? That reading bounds
   what the local memory and the int8 append cost together on a token.

That attention time follows the bytes read stays a conjecture (CERVEAU P3). This run tests it on one tile, with a
handicap whose size is unknown.

## 2. What the code holds

Commit `a7793cd`; this prereg and the job script in the commit after it. No kernel text moves: v9's module and
v9k8's keep their bytes and sha256 (`d55a097b…` and `8b53b1cd…`), and nothing served moves.

- **The rule is one function.** `pers::p4e_token_symbol_loads(local_bytes, resident_a_sm, spill)`: one CTA a SM, and
  no local memory unless `spill`. The load check and every install call it; only one caller passes `spill`.
- **The mark.** An arm named `<arm>+local` in `LLVQ_PERS_4E` is installed by `persist4e_cuda::install_e1_local`,
  which passes `spill`. The mark is refused on the first arm, the reference of every ratio. The load check does not
  know it and keeps printing REFUSED. `LLVQ_PERS_4E` is refused beside `LLVQ_CONFIG`, so no served path reaches it.
- **What the run prints.** A line `4E: MEASUREMENT ONLY: <symbol> holds N B of local memory …` at install, and the
  arm's label carries `N B local` on every rate, F1 and SCORE line.

The arm is timed with whatever local memory the card gives its token symbol that day. The check records the figure
first. If it is not 8 B, the run is read at the figure printed and the journal says so.

Everything else is the kv8 prereg's §2: the layout, the quantizer, the value read, the attention, the fill outside the
timed rounds, the binary16 caches left allocated beside the int8 ones. This job establishes no VRAM gain and reads no
VRAM figure for the arm.

## 3. Stage 1 on the Mac at $0 (*measured*, 2026-11-15, one job at a time through the guard)

- `pers::the_load_rule_waives_local_memory_only_when_asked`: passes. Four mutants of the rule, each at one site, all
  killed.
- `fusedrun::pers_4e_takes_a_count_and_optional_e1_arms`, extended to the mark: passes. Four mutants of the split,
  each at one site, all killed.
- `pers::the_k8_module_keeps_its_bytes`: passes after the change.
- `cargo clippy` clean for llvq-cuda and llvq-llm on the Mac and through `ops/check-linux-on-mac.sh`.

Not verified on the Mac: that a module holding local memory launches and keeps its tokens on the card. v9's stamp
symbol has run with 32 B of local memory outside the timed rounds; no token symbol has.

## 4. What runs

`ops/jobs/recherche-debit-kv8-local.sh` (l40sx1, sealed 4B `886391a8…`, output
`/out/recherche-debit-kv8-local-2026-11-15`):
1. the oracle; the file and the config at their bytes and sha256;
2. the load check `LLVQ_PERS_4E=check:v9,v9k8`: the digests, v9 loads, v9k8's figures recorded;
3. the served door (the served v9 config, 5-token prompt, 64 tokens);
4. `LLVQ_PERS_4E=256:v9,v9k8+local,v9` at a 5-token prompt and at 8,192: 2 + 5 rounds, each v9, v9k8, v9 again, then
   eager, 256 tokens; the phase stamps of v9 and of v9k8.

## 5. Gates and the reading, fixed before the run

Each voids the run: the oracle does not match; a file or the config is not at its sha256; a module's digest is not the
Mac's; v9 does not load; the check prints no token symbol for v9k8; the served door does not run the served v9 module
(252,757 bytes, `1b422b62…`); v9 does not give the eager path's 256 tokens at a context; v9k8 does not keep its own
tokens in every round; the MEASUREMENT ONLY line is missing for v9k8 while the check gave it local memory; F1 or a
SCORE line is missing.

| what the card says | what it means | what follows |
|---|---|---|
| FASTER at 8,192, not SLOWER at 5 | fewer KV bytes buy token time in 4E, local memory included | v9k8 is a candidate as it is. Whether a served module may hold local memory is a structural decision, the operator's. Quality controls through the arm and the form that frees binary16 come next, each under its own prereg and go |
| FASTER at 8,192, SLOWER at 5 | the bytes pay at long context and the arm costs at short | the size of each is published; a config would have to choose by context; the operator decides |
| NOT SEPARATED at 8,192 | no gain is shown; the local memory's cost and the bytes' gain are not separated by this run | no speed is claimed; the reading at 5 and the attention stamps say which way it leans; the operator decides |
| SLOWER at 8,192 | the arm as written loses where it should win most | the arm is not proposed as written |

The reading at 5 tokens stands by itself: v9k8 over v9 there is the cost of the local memory and of the int8 append
together, the cache being almost empty. NOT SEPARATED at 5 means that cost is below the run's floor. It informs the
nine refused arms of O11 and decides nothing about them.

Reported and not a gate: v9k8's tokens against the eager path's and v9's (how many of 256, the first divergence); its
logits against v9's on the positions whose input tokens are v9's; the attention phase's µs for v9 and v9k8 at both
contexts. The arm owes the int8 cache's values, not v9's bits: a divergence of tokens is expected and voids nothing.

No quality is established by this job, for the reasons of the kv8 prereg §5.

## 6. Labels

Every rate and phase time of this job: *measured*. The cache bytes of §1: *computed*.

## 7. Cost

*Estimated* from measured times. The kv8 job was costed at about 1,030 s had the arm loaded; its check ran 210.9 s
where 215 s was planned (job 6af8df0a). This job is that one without the one-arm process (about −120 s): about 910 s,
15 min, $0.46. Timeout 24 min: $0.72 at worst.

The image is rebuilt from the stamped commit before the launch (the Space builds for free).

## 8. Signed predictions (main session)

1. The check gives v9k8 8 B of local memory again: 95 %. The arm launches and keeps its own tokens in every round:
   85 %.
2. The job runs 910 s [760, 1,200].
3. F1 holds at both contexts: 85 %.
4. At 8,192, v9k8 over v9: 1.07 [0.97, 1.18] in tok/s. Read FASTER: 55 %. NOT SEPARATED: 25 %. SLOWER: 20 %.
5. At 5 tokens, v9k8 over v9: 0.985 [0.94, 1.005]. Read NOT SEPARATED: 55 %. SLOWER: 40 %.
6. The attention phase at 8,192 goes from about 2,650 µs to 2,100 µs [1,500, 2,900].
7. v9k8's 256 tokens differ from v9's somewhere at 8,192: 85 %. At 5 tokens: 50 %.

The kv8 prereg put v9k8 at 1.11 at 8,192 had it loaded at 0 B. Prediction 4 is lower because the local memory is in
the timed symbol. I have no measured basis for its size: the interval is wide on purpose.

## 9. Not established by this run

- Any quality figure through the arm. Any VRAM figure for the arm.
- Which value the compiler spills, and whether a module at 0 B would be faster.
- What local memory costs the nine arms of O11: their code is not this arm's.
- The prefill path, the 8B and the 14B, any text but the bench prompt.
- Any card but the L40S, any driver but 580.178.04.
