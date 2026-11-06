# Preregistration. Three trims of the decode step, and the served speed at the product's context

**Written, committed and TIMESTAMPED on 2026-10-30, BEFORE the run.** Operator go to launch given
2026-10-30 ("Go, plafond 2 $"). Cost: about **$0.60** on l40sx1, *estimated* below (§7), timeout 45 min, so
$1.35 at worst. Cap for this wave: **$2**, which covers one retry. Spent before it: $246.75 over 227 priced
jobs (*measured*, `docs/data/jobs.csv`).

## 1. Why this run exists

The throughput audit of 2026-10-29 (`docs/archive/audit-debit-2026-10-29.md`, not committed) counted 1,410
kernel launches a decode token on the sealed 4B, of which 350 are ours. It also computed that the served
KV path re-copies the whole history at every step, 2.06 MB × L a token, 17 GB at 8,192 positions against
1.38 GB of weights. Two facts follow and neither is measured:

1. Three pieces of that overhead can go without changing an output bit. They are coded, uncommitted, and
   have never run on a card:
   - **mask**: the all-zero causal mask of a `Cat` decode step is no longer built, uploaded, cast and added
     in 36 layers (`Qwen3::step_mask`);
   - **cast**: the int4 activation is read in f16 by a new entry point `tv_q4_hh`, which widens it while
     staging, instead of a cast launch before `tv_q4_h`, 84 a token;
   - **events**: on the served door only, cudarc's two CUDA events per allocation are turned off. One
     stream orders every access by issue.
2. Every published tok/s was measured after a 5-token prompt, so at most 261 cache positions. The product
   triplet promises 8,192. The speed there, and the time to the first token, have never been measured.

`fused_cuda.rs` does not compile on the Mac: five dependencies build C for the Linux target. So the image
build of this run is also the first compiler of that file's changes.

Rule 9: nothing in `docs/mesures/` times a decode past 261 positions on the CUDA served path, nor any of the
three trims. The closest is the prefill sweep of 2026-09-11 (3.861 ms a prompt token at 606 tokens, on an
older object with 36 int4 projections instead of 84).

## 2. What runs

One image built from the commit that carries this file. One job, l40sx1, the sealed 4B
(`sealed-4b-2026-09-23/qwen3-4b-sealed.bin`, sha256 `886391a8…`), config `configs/qwen3-4b-tetra-e4.json`.

1. **Oracle** on Qwen3-0.6B, rule 10.
2. **The published arm, replayed.** `fusedrun <file> 256` with the served flags spelled out and
   `LLVQ_EMBED=q4`, against the dense arm of the same process, exactly as run 1 of the paper table
   (`ops/jobs/paper-served.sh`). The mask and cast trims are active on this door; the events trim is not.
3. **The trim A/B.** The same flags plus `LLVQ_TRIM_AB=1`, 256 tokens. One model, five arms in every round,
   same order: `trimmed`, `+mask`, `+cast`, `+events` (each one trim undone), `untrimmed` (all three undone,
   the path the published numbers ran). Two rounds discarded, five timed, each ratio
   `trimmed / arm` formed round by round.
4. **The context sweep.** Through the served door, `LLVQ_CONFIG=<config> LLVQ_CONTEXTS=5,2048,4096,8192`,
   256 new tokens after each prompt. For each length: the time to the first token, and the decode speed
   `255 / (t_last − t_first)`. One discarded generation, five timed. The prompt repeats the 5 tokens of
   `fusedrun`'s prompt to length; no kernel branches on the ids. Prompts above 4,096 go through in chunks of
   4,096 (`MAX_PREFILL_ROWS`), each attending over the cache the earlier ones filled.

## 3. Gates, each one voids the run

1. The image builds, and `oracle` prints MATCH.
2. The sealed file and the config at their bytes and sha256 on the mount.
3. Every load that carries int4 records prints `int4: tv_q4_hh equals tv_q4_h bit for bit`. The check runs
   on the card at load, on the first int4 record, with a pseudo-random activation; a differing row stops
   the load by construction.
4. Step 2 prints `256 tokens identical to the dense arm`, and its decoded line has the sha256 of the same
   line in run 1 of the paper table, `8784ffc4…` (`paper-table-2026-09-25-brut/paper-served-4b-2026-09-25-r2/fused-q4-256.txt`).
5. Step 3 prints its table, which `run_trim_ab` refuses to do if any arm or round returns a different token.
6. Step 4 prints `cudarc event tracking: off (one stream, ordered by issue)`, and every length returns the
   same tokens in all six generations (`run_context_sweep` refuses otherwise).
7. The header of every step reads `tile 64`, and no `LLVQ_*` variable is in the job environment.

## 4. Signed predictions

| quantity | point | interval |
|---|---|---|
| step 2, fused tok/s (mask and cast trims on, events on) | **117** | [110, 124] |
| step 3, `trimmed / +mask` | **1.02** | [1.005, 1.04] |
| step 3, `trimmed / +cast` | **1.03** | [1.01, 1.05] |
| step 3, `trimmed / +events` | **1.04** | [1.00, 1.12] |
| step 3, `trimmed / untrimmed` | **1.09** | [1.03, 1.20] |
| step 4, decode tok/s at a 5-token prompt | **120** | [110, 130] |
| step 4, decode tok/s at 2,048 | **60** | [40, 85] |
| step 4, decode tok/s at 4,096 | **42** | [28, 60] |
| step 4, decode tok/s at 8,192 | **27** | [15, 42] |
| step 4, time to the first token at 2,048, s | **10** | [6, 16] |
| step 4, time to the first token at 8,192, s | **45** | [25, 90] |

Reasoning, *computed*. Mask: 37 launches and 37 small uploads at about 4 µs, 0.15 to 0.3 ms of 8.8. Cast:
84 launches at 3.76 µs (*measured* per launch, `a1-nullk-252-144-2026-08-31`), 0.32 ms. Events: host time
only, about 1,450 allocations a token at six driver calls each; it pays only where the host, not the card,
sets the pace, which the graph A/B suggests at 13.45 % (`a2-verdict-2026-09-01`), hence the wide interval.
Contexts: the `Cat` path moves 2.06 MB × L a token; at an effective 600 GB/s that is 3.4 ms per thousand
positions, on top of the 8.4 ms of a short-context step. The first token: 4 to 6 ms a prompt token, the
84 int4 projections running one row a launch in prefill.

## 5. What is published, whatever the numbers

The step 3 table whole, the step 4 table whole, and step 2's tok/s beside the paper table's 113.8 as a
cross-job reading, never as a ratio (rule 7).

## 6. Decision rules

For each trim, on its `trimmed / arm` range:

- the whole range at or above 1.000: the trim stays, and its gain is published as measured;
- the range contains 1.000: the trim stays, since it is bit-identical, and it is reported as not separated
  from zero;
- otherwise, the whole range below 1.000: the trim is reverted before any commit to `main`.

For the context sweep, a measurement and not a gate. It informs the operator's ordering of the audit's
next steps, and does not decide it:

- decode at 8,192 at or below half the decode at 5 tokens: the KV path (GQA attention, preallocated
  unexpanded cache) is the first lever;
- otherwise: the launch count is the first lever, and the KV path second.

## 7. Cost

*Estimated* on l40sx1 at $1.80/h: container and image pull 3 min, oracle 2 min, step 2 about 3 min (the
dense arm loads in 73 s), step 3 under 2 min (35 generations of 256 tokens), step 4 about 10 min (six
generations a length, dominated by the 8,192 prefill). About 20 min, **$0.60**. Timeout 45 min, **$1.35**
at worst.

## 8. What this run does not establish

- Where the rest of the 1,410 launches spend their time: that needs a kernel timeline (`nsys`), which the
  runtime image does not carry. Adding it is an image decision, left out of this run.
- Anything at 8B or 14B, on another card, or on Metal.
- Quality: the trims are bit-identical by gate 4 and 5; the context sweep scores nothing.
