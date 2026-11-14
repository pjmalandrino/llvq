# Preregistration. Decode-speed research phase, R6c: which call site brings the 8 B of local memory

Written on 2026-11-14 on branch `recherche/debit-2026-11-12`, once the probes existed and their Mac gates had run
(§3). Timestamped before the first measurement, then never edited; a departure goes to
`preregistration-recherche-debit-sondes-2026-11-14-ECARTS.md`.

One job, launched by the operator: `ops/jobs/recherche-debit-sondes.sh`. Cost: about $0.29 (*estimated*, §6). Cap:
$0.36 (the 12 min timeout, *computed* at $1.80/h). Spent before it: $1.34 of the phase's $5 and $20.56 of the $25 cap
(`docs/mesures/recherche-debit-charge2-2026-11-14.txt`, last line).

## 1. The question

Six research arms put a body on every Tetra row phase of 4E's token. Each is refused by the runtime for 8 B of local
memory at 128 registers on the timed symbol `pers_4e_tokens_f1_t512`; v9 holds 0 B (job 6af82af3, *measured*). Four
of the six shed live values from the body in four ways, and the 8 B did not move. So the 8 B do not come from what
those variants removed. Where they come from is not established.

The question: does one call site bring the 8 B, or do they need several?

It is a diagnosis. No speed is measured and nothing is launched on the card but compiles. No candidate becomes
servable by this job.

## 2. What the code holds

Five load probes (`pers.rs::P4E_RC_PROBES`). Each is E1's v9 with `pers_4e_rc.cuh` inserted and the endpass body
(`PERS_RC_ARM 1`) on one call site, or on none. Every other site keeps v9's row loop.

| probe | the endpass body on | edit of `p4e_rc_edits(1)` kept |
|---|---|---|
| `v9pe0` | no site: the text alone | 0 |
| `v9peq` | q/k | 0 and 1 |
| `v9peg` | gate/up | 0 and 2 |
| `v9ped` | the Tetra down | 0 and 3 |
| `v9peo` | the Tetra o (the 8B's; compiled in every module) | 0 and 4 |

The probes are not E1 arms. `LLVQ_PERS_4E=check:` names them; a timed run refuses them by name
(`parse_pers_4e`, `install_e1`). No chain test covers a module with the body on one site, and none is needed: a probe
is compiled and read, never launched.

The check is the one job 6af82af3 ran (`persist4e_cuda::check_e1_arms`: the arms one after the other, each line
printed as it is known). Its parser alone changed, to accept the probes. A probe has no rc layout line.

Why endpass and not flat: both families are refused for the same 8 B, and endpass's sites are the simpler text. Flat
is not probed. If the answer differs for flat, this job does not see it.

Why v9 and v9ep are not rerun: their texts and sha256 are unchanged, and 6af82af3 read them under this driver
(580.178.04): v9 LOADS at 0 B, v9ep REFUSED at 8 B. The image is rebuilt, the kernels are text compiled on the card.

## 3. Stage 1 on the Mac at $0 (*measured*, 2026-11-14, one job at a time through the guard)

- `the_rc_arms_are_v9_with_the_body_on_every_tetra_phase` now covers the probes: each text is E1's v9 with the
  inserted text and exactly its one site edited, nothing else; `pers_4e_rows` keeps its callers on the other sites;
  no probe is an E1 arm. Pass.
- `pers_4e_takes_a_count_and_optional_e1_arms`: the check's parser names the five probes; the timed parser refuses
  each. Pass.
- `the_rc_modules_keep_their_bytes` and `the_rc_layout_at_the_three_sealed_shapes`: pass, the six arms' digests
  unchanged.
- `cargo clippy` clean for llvq-cuda and llvq-llm on the Mac and through `ops/check-linux-on-mac.sh`.

The probes' modules at the card's tile define (`dump_tree_modules`):

| probe | bytes | sha256 |
|---|---|---|
| v9pe0 | 264,669 | `9db6055b82a1a946d83a87631a47c1c65b54f6e59ab608b80dc786c15a75f7e4` |
| v9peq | 264,741 | `c9bb1820159a926a727e77849d08a39d1ddfa1c5d3827dea860b50e4e808de02` |
| v9peg | 264,710 | `e7ad915e4231de1eb1762da8e839cc9374cfefcc3f4dfe0de2cee9579d7f7d1c` |
| v9ped | 264,692 | `89f470f26e2181218a99983f265ce046d30d397a57edc1a35c8f3d538481121c` |
| v9peo | 264,625 | `154f41ac69ba9058eb5293ba77eca53372e229ea7f860c27d6e73bc40f7e2577` |

Not verified on the Mac: that each probe compiles as CUDA. No compiler ran on a probe's text. Each site's edit
compiles inside v9ep, and the sites are independent texts, so a failure is not expected; a probe that does not
compile prints `REFUSED` with the compiler's message and is read as void for that probe, not as a refusal for local
memory.

## 4. What runs

`ops/jobs/recherche-debit-sondes.sh` (l40sx1, output `/out/recherche-debit-sondes-2026-11-14`):
1. the oracle;
2. `LLVQ_PERS_4E=check:v9pe0,v9peg,v9peq,v9ped,v9peo fusedrun <sealed> 2`.

Gates: the oracle matches; the five digests are the Mac's.

## 5. The reading, fixed before the run

The verdict of a probe is the check's: RoPE form 1's token symbol, `LOADS` at 0 B of local memory, `REFUSED`
otherwise.

| what the card says | what it means | what follows |
|---|---|---|
| `v9pe0` refused | the inserted text alone moves the allocation; the sites are not read | the job is void for its question |
| `v9pe0` loads, exactly one site refused | that site brings the 8 B | a rewrite of that one site may be proposed, under its own prereg and go |
| `v9pe0` loads, two sites or more refused | the 8 B come with the body on any such site: the cause is in what the sites share, the body function or its barrier and second pass | no site rewrite is proposed; the phase closes on the register room unless another idea is written |
| `v9pe0` loads, no site refused | no single site is enough; the 8 B need several | the phase closes, or one round of pairs is proposed, the operator decides |

A probe that loads says nothing of its speed. A site that is cleared here is cleared for endpass only.

## 6. Cost

*Estimated* from job 6af82af3: an endpass arm took 103 to 104 s in the check; the oracle and the start took about
40 s. Five probes: about 9.5 min, $0.29. Timeout 12 min, $0.36 at worst; the check's own timeout is 10 min.

## 7. Signed predictions (main session)

1. `v9pe0` loads. 90 %.
2. `v9peg` (gate/up) is refused at 60 %, `v9peq` at 35 %, `v9ped` at 35 %, `v9peo` at 25 %.
3. Exactly one site is refused: 45 %. Two or more: 30 %. None: 25 %.
4. A refused probe shows 8 B. 70 %.
5. A probe fails to compile: 10 %.
6. One probe takes 104 s [90, 125]. The job runs 570 s [480, 690].

## 8. Not established by this run

- Any tok/s. Whether endpass or flat keeps its rows-alone gain in the token.
- What a rewritten site would cost or whether it would load.
- Flat's sites. Any card but the L40S, any driver but 580.178.04.
