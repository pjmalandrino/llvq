# DRAFT, not stamped, not committed

# Preregistration. The GEMM route of the prefill: its dequant against the dense arm, its first token from 2,048 to 8,192 tokens, and the T8 control on fresh windows under the served grouped attention

**Draft revision 2, written on 2026-11-02 for the branch that builds step 5 of
`docs/archive/audit-debit-2026-10-29.md`. To be committed and TIMESTAMPED BEFORE the run.** It assumes the arithmetic
C-dense and a threshold R* = 64; the branch replaces both, and every other mark ⟨…⟩, before stamping. Operator go:
**not given**. A T8 decision for the route is **not given** either: this run is the control such a decision would
rest on, not the decision. Cost: about **$1.40** with two arms and **$1.67** with the optional third, *estimated*
below (§8), timeout 90 min, so $2.70 at worst. Cap asked: **$3**. No cap is in force (ETAT §5), and one is owed before
the next paid job. Spent before it: $248.84 over 237 rows of the registry (*measured*, `docs/data/jobs.csv`), plus
prereg-prefill-1 if it ran.

## 1. Why this run exists

The first token of an 8,192-token prompt takes 35.02 s on the served 4B (*measured*, `served-grouped-2026-11-02.txt`),
84 % of an 8,192 + 512 request (*computed*). The rows kernel reads the whole weight stream once per 4 prompt rows. It
also spends about 486 row-granular operations per prompt token (*computed*).

The GEMM route dequantizes one matrix at a time into an f16 buffer and multiplies the whole chunk through cuBLAS. Under
⟨C-dense⟩ the buffer holds the dense arm's own weights: `llvq_artifact::decode_matrix` (f64 decode and un-rotation,
then f32) cast to f16 (`llvq-llm/src/sealed.rs:519-531`). The call is the dense arm's own
`candle_nn::Linear::forward`. So the route's hidden states should equal the dense arm's bit for bit under the same
attention form, and its numerics are those the published MMLU and perplexity were read with. Against today's fused
prefill it is a numerical change, which is a T8 event.

What is known at $0, and why it does not decide:
- The dense reconstruction and the fused prefill pair at +0.0095 % [-0.0268 %, +0.0458 %] of perplexity over 12
  windows of 4,096 (*computed* from `ppl-scelles-2026-10-04-brut/ppl-4b.txt` and
  `gqa-ppl-2026-11-01-brut/gqa-ppl.txt`). The two arms ran in two jobs a month apart, so this reading is reported and
  not published (METHODE §2). It indicates a small cost; it bounds nothing.
- Those windows already served a decision.
- The head differs between the two arms of that pairing, and the route keeps the fused head.
- No pairing covers a prompt prefilled one way and continued by the fused decode.

## 2. What the code will hold before the stamp

To be written by the branch, with its host results filled in here:

- A dequant kernel per record type:
  - Tetra48 ⟨`deq_tetra48_f64_h`⟩: f64 operations through `__dmul_rn` and `__dadd_rn`, `__dsqrt_rn` and `__ddiv_rn`,
    stored by `f2h(__double2float_rn(x))`.
  - int4: `q4_deq`, then `f2h`.
  - A check-only variant of both kernels that stores the f32 values before the cast, for the digest comparison of
    step 3b. It is never on a served path.
  - The f64 row scales, the f32 tail and the f64 rotation tables come from the file and stay in pinned host memory,
    uploaded per pass. Resident VRAM does not move.
- A branch in `model::group_forward` for rows ≥ ⟨64⟩: dequantize, then `Linear::new(w, None).forward(x)`. No rotation
  of the activation, since the weights are in the natural basis.
- A config field ⟨`"prefill": "gemm"`⟩; absent means `"rows"`, every published path. A measurement mode
  ⟨`LLVQ_PREFILL_ROUTE_AB`⟩ on the bench door, refused beside `LLVQ_CONFIG`, that switches the route on one loaded
  model. It spells out the served flags and sets the grouped heads on `Grow(256)` itself, as `LLVQ_GQA_AB` does.
- The harness pieces (`wave5-deep/code/prefill/gemm_route.rs`):
  - `window_nll_split(win, 2048)`: one forward of the whole window, returning the whole-window NLL over 4,095
    predictions and the tail NLL over the predictions of tokens 2,048 to 4,095;
  - the prompt-then-decode path: tokens 0 to 2,047 through `prefill_chunked` under the arm, then tokens 2,048 to 4,094
    one at a time through `step_logits` (`model.rs:2207`), scoring tokens 2,048 to 4,095.
- ⟨The third arm⟩: `RowsTrim(1024)` from `prereg-prefill-1.md`, on top of the route, if that run's R2 selected it.
  Otherwise its rows below go before the stamp.
- Host evidence, Mac CPU, *results to be filled*:
  - the device functions replayed by clang++ with `-ffp-contract=off` against `decode_matrix` on fixture records,
    bitwise;
  - mutants killed: contracted FMA, single-step f64 → f16, reversed mix order, WHT scale before the stages, int4
    dequant in f64;
  - every new matmul operand through `cuda_gemm_layout_check`, the replica of candle's CUDA layout rules;
  - host code in `llvq-cuda` type-checked from the Mac (`CUDARC_CUDA_VERSION=12040 cargo clippy --target
    x86_64-unknown-linux-gnu -p llvq-cuda --all-targets`); the `cfg(linux)` side of `llvq-llm` is compiled first by
    the image, so it stays thin.

## 3. What runs

One image built from the branch commit that carries this file. One job, l40sx1, the sealed 4B
(`sealed-4b-2026-09-23/qwen3-4b-sealed.bin`, sha256 `886391a8…`), the served flags of
`configs/qwen3-4b-tetra-e4-grouped.json` (sha256 `2455fe85…`) spelled out. If the branch's own commits moved either
reference line of gate 4, the stamped prereg names the branch's measured lines instead.

1. **Oracle** on Qwen3-0.6B, rule 10.
2. **The bench door, replayed**, with the dense arm resident: gate 4.
   - **2b. The served door, replayed** under the grouped config, no `prefill` field: gate 4b.
3. **The route against the dense arm, on the card.** In the process of step 2, with the dense arm still resident:
   - (a) for every one of the 252 records, the route's f16 buffer against the dense arm's f16 tensor on the device:
     the number of differing elements;
   - (b) for one record per shape class (q, k, gate, down, and one int4), the route's f32 values before the cast,
     hashed on the host, against the record's entry in the dense digest. The digest is the bucket copy
     `/out/hf-cuda-4b-2026-09-30/model/llvq-dense-digest.json`, 28,987 B (`wave5-deep/bucket-listing.txt:473`). The job
     checks its sha256 against ⟨the value pinned here, read from the Mac with `hf buckets cp` before the stamp⟩ and
     its `source.sha256` against the sealed file's `886391a8…`. `hfdense` is in neither list of `ops/Dockerfile.cuda`,
     and no 4B dense digest is committed. If the bucket copy names another source, the fallback is to add `hfdense` to
     both lists of the Dockerfile and to `UPLOAD_ALLOW`, a recipe change that needs a go; the stamped prereg says
     which reference it uses;
   - (c) the final hidden state of the last prompt position after 2,048 tokens. The route and the dense arm both run
     the expanded form on `Cat`, which is the dense arm's form, and the bits are compared.
4. **The first token.** Arms `rows` (today) and `gemm`, on one model and in one process, under the grouped form on
   `Grow(256)`. Prompts of 2,048, 4,096 and 8,192 tokens in chunks of 4,096, built as `LLVQ_CONTEXTS` builds them.
   1 discarded and 5 timed rounds, both arms in each round, in alternating order.
   - The ratio `t_rows / t_gemm` is formed round by round.
   - In round 0, the 256 greedy tokens after each prompt, both arms (the first divergence).
   - The pool high-water of each prefill (`CU_MEMPOOL_ATTR_USED_MEM_HIGH`).
   - One full dequant pass, timed as a burst of 3 passes between two fences.
5. **The T8 control, under the served grouped attention.** The cached paths run the grouped heads on `Grow(256)`; the
   uncached forward of `window_nll_split` runs the grouped heads on a fresh `Cat` with `rows_contiguous`, as gqa-ppl's
   grouped arm did. WikiText-2 test, windows **13 to 36** of 4,096 tokens: 24 windows that never served a decision. The
   token fingerprint is printed. Per window, for each arm (`rows`, `gemm`, ⟨`gemm + RowsTrim(1024)`⟩):
   - the **prefill path**: `window_nll_split(win, 2048)`, the whole window through the prefill, the route engaged for
     its 4,096 rows: the whole-window NLL, and the tail NLL over tokens 2,048 to 4,095;
   - the **prompt-then-decode path**: tokens 0 to 2,047 through the prefill under the arm, then tokens 2,048 to 4,094
     one at a time through the fused decode, which is the same code for every arm. It scores tokens 2,048 to 4,095;
   - per path, a paired 95 % t interval over the 24 windows of `ppl_gemm / ppl_rows − 1`: the whole window for the
     prefill path, the 2,048 scored positions for the prompt-then-decode path. Top-1 agreement and max |Δlogit| on the
     prompt-then-decode path;
   - for the third arm, the same intervals against `gemm`.

## 4. Gates, each one voids the run

1. The image builds, and `oracle` prints MATCH.
2. The sealed file and the config at their bytes and sha256 on the mount.
3. Every load prints `int4: tv_q4_hh equals tv_q4_h bit for bit`.
4. Step 2 prints `256 tokens identical to the dense arm`, and its decoded line hashes to `8784ffc4…`. The 5-token
   prompt is under R*, so the route cannot reach it.
   - **4b.** Step 2b's provenance line names `attention grouped on grow(256)`, and its whole decoded text hashes to
     `0597cee8…` (`docs/mesures/served-grouped-2026-11-02-brut/lines.sha256`).
5. **The dequant is the dense arm's.** Step 3a counts 0 differing elements on all 252 records. In 3b the digest file
   matches its pinned sha256, names the sealed file as its source, and matches all five records. 3c is bit for bit. A
   route that is not the dense arm has no reference for this control.
6. **The instruments discriminate.**
   - The comparator of 3a flags a planted one-ulp change.
   - On the `rows` arm, the prompt-then-decode path's perplexity over the prefill path's tail perplexity, on the same
     2,048 positions, lies in [0.99, 1.01]. gqa-ppl's gate 6 made the same check over 4,095 positions and read 0.99994
     (*measured*). A misaligned target would read far outside.
7. The header of every step reads `tile 64`, and no `LLVQ_*` variable is in the job environment.

## 5. Outcomes reported, never gated

- Step 4: for each arm and length, the time to the first token, median and range; the ratio, median and range; the
  first token divergence; the high-water. The time of one dequant pass.
- Step 5: per window, each arm's whole-window NLL, tail NLL and prompt-then-decode NLL; the intervals; top-1 agreement;
  max |Δlogit|; the harness ratio of gate 6.

## 6. Signed predictions

| quantity | point | interval |
|---|---|---|
| step 3a, differing elements, 252 records | **0** | |
| step 3c, hidden state | **bit for bit** | |
| `rows` first token at 2,048 / 4,096 / 8,192, s | **⟨8.20 / 17.03 / 35.02⟩** | ±3 % each; re-signed from the branch's own sweep if its step-2 fusions moved them |
| `gemm` first token at 2,048, s | **0.5** | [0.35, 0.8] |
| `gemm` first token at 4,096, s | **1.2** | [1.0, 1.6] |
| `gemm` first token at 8,192, s | **3.25** | [2.8, 3.9] |
| `t_rows / t_gemm` at 8,192 | **10.8** | [9.0, 12.5] |
| one full dequant pass, s | **0.18** | [0.11, 0.30] |
| high-water at 8,192, `gemm` minus `rows`, GB | **+0.1** | [0.0, +0.4] |
| 256 tokens after the 8,192 prompt, `gemm` against `rows` | **diverge** (different KV) | first divergence after token 20 |
| gate 6 harness ratio, `rows` arm | **1.000** | [0.998, 1.002] |
| prefill path, `gemm / rows − 1` | **+0.01 %** | [-0.04 %, +0.06 %] |
| prompt-then-decode path, `gemm / rows − 1` | **+0.005 %** | [-0.03 %, +0.04 %] |
| both paired intervals inside [-0.12 %, +0.12 %] | **yes** | |
| prompt-then-decode top-1 agreement | **99.8 %** | [99.0 %, 99.95 %] |
| third arm, either path, `(gemm + RowsTrim) / gemm − 1` | **0.000 %** | [-0.02 %, +0.02 %] |

Reasoning, *estimated* unless marked:
- **First token.** Three dense windows of one job put the attention of a 4,096² pass at 0.74-0.85 s and the linear
  work of 4,096 rows at 0.155-0.25 s (`wave5-deep/prefill.md` §3.2). Tied, that gives C-dense 0.44-0.57, 1.15-1.28 and
  3.05-3.40 s at the three lengths. The grouped form may add up to 0.2 s at 8,192 (*measured* +0.18 s on the rows
  path, served-grouped). Prereg-prefill-1's measured attention times, if available, replace this paragraph before the
  stamp.
- **Quality.** The cross-job pairing above indicates about +0.01 %; the head difference is removed here.
- **Prompt-then-decode.** Half the positions of the window, all scored after a shared decode, which dilutes the effect.
- **Third arm.** `RowsTrim` changes the rounding of attention products only. gqa-ppl's grouped-against-expanded change,
  of the same kind, read +0.0000 % by prefill and +0.0095 % by decode (*measured*).

## 7. Decision rules

- **T1, eligible.** Gates pass, both intervals lie inside ±0.12 %, and the `gemm` first token at 8,192 lies at 4 s or
  under. The route is then eligible to serve behind its config field, with the reference lines unchanged (R* > 5).
  - Serving it needs the operator's T8 decision, with this run as its input.
  - It also needs his reading of the ROADMAP §4 hold, which says serving the prefill "reopens the layout" (wave
    report, decision 9).
  - The census through the served object (ROADMAP §2.1) becomes the next cheap measurement: under the expanded config
    it inherits the dense census, and under the grouped config it reads the served object.
- **T2, quality fails.** An interval leaves the band on one side of zero. The route is not served, today's prefill
  stays, and the bit-exact rows path (`prefill.md` §3.3) becomes the first-token route.
- **T3, no power.** An interval is wider than the band but contains zero. The operator chooses between more windows and
  stopping.
- **T4, quality passes but the clock misses.** Both intervals lie inside the band, the first token at 8,192 is above
  4 s, and the prediction is false. Before any serving, the split between dequant, GEMM and attention is measured.
  The operator weighs the gain, which still lies near 10× by this run's own numbers.
- **T5, the third arm**, if it ran. Both of its intervals inside ±0.12 %: `RowsTrim(1024)` becomes eligible on top of
  the route, under T1's conditions. An interval leaving the band on one side of zero: it is not served. Wider than the
  band and containing zero: T3's choice. Its windows stop at 4,096 tokens, so the control covers L up to 4,096; the
  L = 8,192 shapes rest on prereg-prefill-1's grid report.
- **Otherwise**, for example a gate failing: the run is reported and nothing is proposed. A failing gate 5 means the
  kernel is not the dense arm, and it is fixed on the host replay first. A failing gate 6 means the harness is wrong,
  and the quality results are void.

## 8. Cost

*Estimated* on l40sx1 at $1.80/h (`ops/run.py:105`):

| item | minutes |
|---|---|
| container and image pull | 3 |
| oracle | 2 |
| step 2, with the dense arm's load | 4 |
| step 2b | 1.5 |
| step 3, 252 device comparisons, 5 host hashes and one 2,048 prefill per arm | 1.5 |
| step 4: `rows` 60.3 s and `gemm` about 5 s of prefill a round, 6 rounds, round-0 decodes, the dequant burst | 7 |
| step 5, two arms, 24 windows of 69.2 s: `rows` window 17.0 s plus about 1.0 s of all-position head, `gemm` window 1.3 s plus 1.0 s, prompts of 2,048 at 8.20 s (*measured*) and about 0.55 s, two decodes of 2,047 steps at 9.8 ms | 27.7 |
| step 5, third arm: 24 windows of 22.8 s (window 2.2 s, prompt 0.55 s, one decode of 2,047 steps) | 9.1 |

In all about 47 min, **$1.40**, with two arms, and 56 min, **$1.67**, with three. The decode step costs 8.78 ms at
2,048 and 10.20 ms at 4,096 under the grouped form (*measured*, served-grouped), plus about 0.3 ms of scoring. Under
the expanded form, at 12.09-16.28 ms, two arms would take 54 min. The anchor is gqa-ppl-r2: 123.5 s a window for two
prefill paths of 4,096 and 4,095 lockstep steps of one expanded and one grouped decode, 21.4 ms a pair (*computed* from
its printed clock), which matches the per-step costs above. Timeout 90 min, **$2.70** at worst.

## 9. Deviations

Any departure goes in `proofs/preregistration-prefill-gemm-<date>-ECARTS.md`, beside this file, never inside it.

## 10. What this run does not establish

- Task accuracy under the route: MMLU through the served object and GSM8K through the kernel are the debt that
  follows.
- Quality at 8,192 of context: the windows stop at 4,096, because no harness scores past it.
- C-rot16, d1 or d2; `RowsTrim` at L above 4,096.
- 8B, 14B, another card, Metal (which has no f64, so C-dense does not exist there).
