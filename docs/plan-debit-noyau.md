# Plan: decode throughput of the Tetra kernel

State as of 2026-11-06. This is a study at $0: no run, no build on a card, no job. Every target below is
*estimated*, and no lead starts without the operator's go, with its cost announced (CLAUDE.md rule 1). Where things
stand is in [`ETAT.md`](ETAT.md), the queue in [`ROADMAP.md`](ROADMAP.md), the Metal detail in
[`plan-metal-debit.md`](plan-metal-debit.md). The route to 300 tok/s from 4E v6 is [`plan-300.md`](plan-300.md).

## 1. Starting point

| size, card | config | short prompt | 8,192 positions | GB | journal |
|---|---|---|---|---|---|
| 4B, L40S | `qwen3-4b-tetra-e4-4e-v4.json` | 178.0 door, 180.8 decode | 139.5 | 1.38 | [served-v4](mesures/served-v4-2026-11-05.txt) |
| 8B, L40S | `qwen3-8b-tetra-e4-grouped.json` | 105.8 door | 67.1 | 2.76 | [served-grouped-8b](mesures/served-grouped-8b-2026-11-04.txt) |
| 14B, L40S | `qwen3-14b-tetra-e4-grouped.json` | 62.5 door | 45.3, bench door | 5.04 | [served-grouped-14b](mesures/served-grouped-14b-2026-11-04.txt), [gqa-14b](mesures/gqa-14b-2026-11-04.txt) |
| 4B, M3 Max | nothing served; kernel `lut`, expanded form, `metalrun` | 50.9 | 15.8 at 4,096 (`lutd`) | | [metal-nuit](mesures/metal-nuit-2026-11-06.txt) M2, M4 |

All *measured*, tok/s, batch 1. Three faster paths are measured and not served: 4E v6 at 212.6 and 152.0 in the
measurement form ([pers-4e-v6](mesures/pers-4e-v6-2026-11-05.txt)), the bit-exact Rust Metal set at 65.5 (M14), and
the MLX path at 93.9, 70.3 at 4,096 positions (M16).

The 4B's weights are 1.377 GB a token. At the L40S's 864 GB/s that read takes 1.59 ms, a ceiling of 628 tok/s
(*computed*). The 4E v6 token takes 4.70 ms, so two thirds of it is above the weight floor.

## 2. Where a 4B token goes under 4E

| part, 5 positions | µs a token, mean CTA work | journal |
|---|---|---|
| ln1 + rotation | 318.9 | pers-4e-v6 |
| q/k/v rows | 404 | [shaker-4e](mesures/shaker-4e-2026-11-05.txt), v4 |
| norm, RoPE, KV append | 27 (wall 111) | [pers-4e-v3](mesures/pers-4e-v3-2026-11-05.txt) |
| attention | 248 (wall 644) | pers-4e-v3 |
| o rows | 360 (wall 476) | pers-4e-v3 |
| ln2 + rotation | 157.8 | pers-4e-v6 |
| gate/up rows | 1,184 | shaker-4e, v4 |
| wait before the down input | 189 (slowest CTA 348) | shaker-4e, v4 |
| down input | 254.6 | pers-4e-v6 |
| down rows | 452.8 | pers-4e-v6 |
| head | 360 | pers-4e-v3 |

The row parts sum to 2.76 ms for the whole 1.377 GB (*computed*). The Tetra row loop alone streams 379 GB/s
([pers-rowloop](mesures/pers-rowloop-2026-11-05.txt)) and the int4 rows about 587 (pers-4e-v3), both *measured*. At
those solo rates the 0.769 GB of Tetra rows and the 0.628 GB of int4 rows and head would take 3.1 ms (*computed*,
bytes from [revue-debit](archive/revue-debit-2026-11-05.md) §2.2). The rows in 4E are no slower than alone. The other
1.94 ms of the 4.70 ms token splits into 1.20 ms of non-row work and about 0.75 ms of the slowest CTA waiting above
the mean (*computed*; parts from three jobs, so the split is approximate). 300 tok/s is 3.33 ms, 1.37 ms under v6.

Two earlier 4E estimates overshot. v6 realised 42 % of its predicted gain and v3 none (*computed* from the predictions
in pers-4e-v6 and pers-4e-v3). Each target below is therefore a range: the high end is the arithmetic, the low end
keeps half of the gain. The measurement form converts to the door by ×0.937 at 5 positions (*computed*, 178.0 over
v4's 190.0, two jobs). At 8,192 the two forms agree within 1 % (served-v4, shaker-4e).

## 3. Techniques surveyed

| technique | how it works | what it means for Tetra | tried here | verdict |
|---|---|---|---|---|
| persistent megakernel: Hazy "No Bubbles", Mirage MPK | one launch a token, dependencies by counters | 4E is this form, with 218 grid barriers a token | 4E v4 served, ×1.404 over eager | K3 swaps barriers for counters |
| CUDA Graphs | replay a captured launch list | 4E already runs host-free | A2, closed in `ETAT.md` §6 | not proposed |
| programmatic dependent launch | a kernel starts before its producer ends | sm_90 and up, absent on sm_89 | no | eager path on sm_120 only; not proposed |
| Marlin | multi-stage `cp.async` loads, offline layout for 128-bit loads, decode hidden behind loads | Tetra words go through generic `LD.E`, never `.nc` ([pers-sass](mesures/pers-sass-2026-11-05.txt)); a load-time re-layout keeps the file | prefetch ×0.994, tables in shared ×0.948 ([pers-4e-v2](mesures/pers-4e-v2-2026-11-05.txt)); words held in L1 ×1.214 over `v2u` but ×0.99 against v2, the loop 4E runs (pers-rowloop) | K4, microbench first |
| Machete | Marlin on `wgmma` | Hopper only | no | does not apply |
| FLUTE, LUT-GEMM, CodeGEMM, AQLM GEMV | table of codebook × activation, reused across rows | Tetra pair tables pay above about 1,024 rows an activation block; a 4E CTA sees 7 to 68 rows ([recherche-decodage](archive/recherche-decodage-kernel-2026-09-08.md) §4) | analysed | not at batch 1 |
| QuIP# E8P, QTIP trellis | small codebook LUT, or a code computed from a bit window | the Leech ball holds 1.1·10¹⁴ points, no LUT; Tetra is already a computed decode | QTIP arm in the bench | QTIP beats our `nullk` floor, a geometry lesson |
| ExLlamaV2/V3, llama.cpp MMVQ | activation quantized to int8 once, `dp4a` on integer weights | Tetra values are integers, \|y\| ≤ 10: block dots are exact against an int16 activation | the brick lead, designed, not built ([brick](archive/debit-vague5-2026-11-02/brick.md)) | a T8 event; pays only for a flat multi-row verify (K9) |
| llama.cpp IQ2_XXS | E8 grid table plus sign tables, graphs | 312.9 tok/s on our L40S, about 391 GB/s | reference arm | a target in GB/s |
| MLX `qmv`, llama.cpp Metal `mul_mv` | rows per simdgroup, wide loads, few dispatches | our MLX Tetra kernel reads 129 GB/s where mlx-lm 4-bit reads about 290 | M15, M16 | M4 |
| FlashDecoding, FlashInfer split-KV | split the cache over CTAs, merge by log-sum-exp | `attn_dec.cu` is split-KV at η 0.545, FlashInfer 0.810 on the same card | measured | K7 |
| split-K on the matvec | split d_in over CTAs | | −1.87 % on `Planes14` ([a3-occupation](mesures/a3-occupation-banc-2026-09-01.txt)) | not proposed |
| speculation: prompt lookup, SuffixDecoding; EAGLE-3, DFlash | draft tokens, verify k rows in one pass | 2-row Tetra rows equal 1-row rows bit for bit on the card; the second row costs 0.378 of a pass (shaker-4e) | analysed ([speculation](archive/debit-vague5-2026-11-02/speculation.md)) | K9, prompt lookup only |
| KV quantization: KIVI, QServe KV4, fp8 KV | store K and V in 8 or 4 bits, read in place | `KvMode::Q8` keeps the reconstructed f16 and saves nothing | q8 green at short context only | K8 |
| tensor cores at batch 1 | MMA on 16-row tiles | 15 of 16 rows wasted at batch 1 | no | prefill only (K10) |

## 4. Leads

Criteria are those of [`METHODE.md`](METHODE.md) §1. "Identity" means the served tokens stay bit for bit, checked at
the door. The T8 band is ±0.12 % of paired perplexity ([`ETAT.md`](ETAT.md) §4). Prices come from
`docs/data/jobs.csv`.

### K1. Serve 4E v6

- **Target.** 4B: 199 tok/s short, 152.6 at 8,192 (*estimated*: v6 over v4 round by round, ×1.118 and ×1.094,
  applied to 178.0 and 139.5).
- **Criteria.** Throughput only. v6 returns the eager tokens at both contexts, 64 of 64 (*measured*). No stored bit
  moves.
- **License.** A door reading under a v6 config: 256 tokens equal to v4's, the 50 GSM8K pilot completions identical.
- **Cost.** Half a day of code (a `"rows": "v6"` value), one l40sx1 job at about $0.42 (served-v4's price).
- **Gate.** The operator, against the missed gate of v6's prereg ([`ETAT.md`](ETAT.md) §5).

### K2. v7: the rest of the down input and ln1's rotation

- **Target.** Saves 0.15 to 0.31 ms: down input from 254.6 to the prereg's 100 µs, ln1 + rotation from 318.9 to
  ln2's 160. 4B door 206 to 213 short, 156 to 160 at 8,192 (*estimated*, on K1).
- **Criteria.** Throughput only, if the Walsh-Hadamard stage order holds. Identity gate.
- **License.** Inner stamps under 200 µs for the down input, eager tokens at both contexts.
- **Cost.** 2 to 4 days, one reading at $0.24 (pers-4e-v6's price). This is [`ROADMAP.md`](ROADMAP.md) §2.6.
- **Gate.** v6's own: down input under 200 µs by the inner stamps.

### K3. The slowest CTA: attention at short context, counters for barriers

- **Target.** Recovers 0.3 to 0.6 ms of the 0.75 ms the slowest CTA spends above the mean. At 5 positions the
  attention phase runs 644 µs on the wall for 248 of mean work; the wait before the down input reaches 348 µs on the
  slowest CTA. 218 grid barriers at 0.79 to 1.13 µs cost 0.17 to 0.25 ms (*computed*, barrier cost measured in
  pers-l1). 4B door 220 to 247 short; at 8,192 the attention wall exceeds its mean by 0.66 ms too, so 163 to 176
  (*estimated*, on K2).
- **Criteria.** Throughput only, if each head keeps its reduction order. A repartition that moves the order is a T8
  event and needs the band.
- **License.** Identity at the door, at 5 and 8,192.
- **Cost.** 1 to 2 weeks, one reading at $0.24 to $0.42.
- **Gate.** The slowest-over-mean ratio of the attention phase under 1.5 at 5 positions, eager tokens kept. Spreading
  a CTA's last rows over its warps is 4E v5, closed in `ETAT.md` §6, and is not this lead.

### K4. The word stream: asynchronous staging and a load-time layout

- **Target.** Saves 0 to 0.32 ms of the 2.76 ms of rows (*estimated*). 4E v4 and v6 run v2's loop, and with every word
  held in L1 (arm `l1w`) that loop reads 2.516 ms against v2's 2.499: the word path costs v2 nothing measurable. The
  ×1.214 of `l1w` is over `v2u`. The upper end is a ×1.21 on the Tetra rows' share, by whatever arm; the suspect left
  standing is the warp balance, 0.63 at d_in 4,096 and 9,728. 4B door up to 284 short, 193 at 8,192 (on K3); the
  targets below that chain on K4 take its upper end.
- **Criteria.** Throughput. The file and its fingerprint stay as they are. A tile-major layout in VRAM may pad: GB on
  card stated before the run (rule 6). Identity gate.
- **License.** First the row-loop microbench (`pers_trows.cu`) with three arms: `.nc` loads of the words, a
  double-buffered `cp.async` stage, a 128-bit tile-major word layout. An arm under ×1.10 over v2 on the loop kills
  it. Only an arm over that bar goes into 4E.
- **Cost.** 3 to 5 days, $0.25 for the microbench (pers-rowloop's price); a week and $0.24 more for the 4E port.
- **Gate.** The microbench bar above. The model of this loop is uncalibrated: the only prefetch tried read ×0.994,
  and two static models are refuted, so no 4E variant runs before the microbench.

### K5. A certified head

- **Target.** The head reads 360 µs. Reading the int4 g64 table by bit planes, and rescoring only the rows whose upper
  bound beats the leader, saves 0.08 to 0.16 ms. 4B door 238 to 298 short, 172 to 199 at 8,192 (*estimated*, on K4).
- **Criteria.** Throughput. Same argmax by construction. The planes are the same bits rearranged at load; per-row
  bounds add about 0.6 MB, +0.001 b/param (*computed*, 151,936 rows of 4 bytes).
- **License.** Argmax equal on every door token, a host proof of the bound, mutants killed.
- **Cost.** 1 to 2 weeks, a bench arm at about $0.2, then the K-series door.
- **Gate.** Head under 250 µs, same tokens.

### K6. Kernel attention form and 4E at 8B and 14B

- **Target.** 4E removed 2.27 ms a token at 4B (126.8 to 178.0 at the door). The 8B has the same 36 layers, the 14B
  40. 8B: 118 to 140 short, 82 to 110 at 8,192. 14B: 67 to 74 short, 54 to 68 at 8,192 (*estimated*: the kernel form
  at 8,192 from the review's 88 and 58, revue-debit §3.2 item 3, then 1.0 to 2.3 ms of 4E's fixed saving, 1.15 to
  2.5 at 14B).
- **Criteria.** Throughput. The kernel form changes the attention arithmetic: T8 per size, then a light MMLU, as the
  grouped form had. 4E then owes the eager tokens. The 14B's down input stages 17,408 f32 values, 69,632 B, past the
  default 49,152 and under the 101,376 B opt-in ([`format-noyau.md`](format-noyau.md) §8).
- **License.** Paired perplexity in the band per size, 285-question MMLU unchanged, door identity of 4E.
- **Cost.** T8 $1.5 at 8B and $2.5 at 14B (*estimated*, same review); 1 to 3 weeks to make 4E shape-general, never
  costed before; door readings about $1.25 and $1.81 (the grouped doors' prices).
- **Gate.** The T8 at each size first. 4E stays on the 4B's shape until it passes.

### K7. The attention kernel's memory pipeline

- **Target.** η from 0.545 to about 0.8, FlashInfer's level on the same card
  ([fi-decode-anchor](mesures/fi-decode-anchor-2026-11-04.txt)). The 8,192 attention wall drops by 0.4 to 0.8 ms. 4B
  185 to 237 at 8,192 on K5 (*estimated*). No effect at short context.
- **Criteria.** Throughput, if the reduction tree stays. Otherwise T8.
- **License.** The attention bench in situ, the canonical replay's bits, then door identity.
- **Cost.** About a week, $0.3 to $0.5 (revue-debit §3.2 item 2).
- **Gate.** η ≥ 0.70 in the bench at 8,192.

### K8. A real KV store, int8 then int4

- **Target.** After K7, the 8,192 attention reads about 1.25 ms of mean work. int8 saves about 0.6 ms and int4 about
  0.9: 4B 8,192 up to 202 to 301 at the end of the series (*estimated*). The 8k cache goes from 1.21 GB to 0.64 or
  0.34 GB (*computed*, revue-debit).
- **Criteria.** Quality: the stored values change. GB in memory. Model class, because the b_max of the product
  triplet already counts a q8 cache at 8k that the engine does not store. Noise: read against the paired bar.
- **License.** Its own prereg: paired perplexity in the band, a light MMLU, a long task past 1,280 positions.
  GSM8K never reaches 1,280. A Hadamard on q and k, which the repository already computes, is the first design to
  test for int4 keys.
- **Cost.** 2 to 4 weeks of code, $3 to $5 of card.
- **Gate.** The operator's order: kernel first, KV second (2026-11-05).

### K9. Exact speculation by prompt lookup, inside 4E

- **Target.** Workload-dependent. A 2-row verify costs about 1.25 to 1.30 tokens (*estimated* from the 0.378 of
  shaker-4e plus attention). One drafted token accepted with probability p gives (1 + p) / 1.3: about ×1.0 on chat,
  where gating skips the draft, and ×1.15 to ×1.30 on extraction and code edits at p = 0.5 to 0.7 (*estimated*). The door with the house prompt cannot read
  it: that prompt loops and accepts almost every draft.
- **Criteria.** Throughput. Greedy verify keeps the tokens if every k-row part keeps its bits on the card; the rows do
  (*measured*), attention and the head are unchecked. GB unchanged: prompt lookup carries no drafter. Neural drafters
  stay excluded by the operator.
- **License.** Full-step identity of the 2-row verify on the card, then a neutral prompt set.
- **Cost.** 2 to 3 weeks; an end-to-end job at about $1.00 (speculation.md, L4).
- **Gate.** The operator's rulings R1 (the speculative verdict of 2026-08-05) and R8 (the hold on the multi-row path).

### K10. Beside the decode: the prefill by GEMM

- **Target.** Time to the first token at 8,192 from 35.0 s to 2.85 to 3.4 s (*estimated*,
  [prefill](archive/debit-vague5-2026-11-02/prefill.md)). Decode tok/s do not move. A chat at 8k waits 35 s today
  before any token.
- **Criteria.** Latency. GEMM rounding: T8.
- **Cost.** 4 to 7 days, $1.40 to $1.67.

### M1. The runtime on Metal at the engine's rate

- **Target.** The runtime decodes 21 to 24 tok/s in chats under 300 positions (*measured* by the operator with `curl`,
  2026-11-06, no journal), where `metalrun` reads 50.9 after a 5-token prompt (M2). Context does not explain it: the
  expanded form falls to 23.2 tok/s only near 2,048 positions (*measured*, M4). The cause is in the runtime's own
  loop (*estimated*). Suspects, cheapest first: the fused kernel not loaded (its provenance line), the prompt's
  prefill counted in the rate, a host sync or a full logits copy each token, the detokenizer and the stream flush.
- **Criteria.** None. A diagnosis.
- **License.** The same prompt at the same position through `metalrun` and through the runtime, each token's time
  split into kernel, host and stream.
- **Cost.** Half a day, $0, light enough for a working Mac.

### M2. Serve the bit-exact Metal set

- **Target.** Short: 50.9 to 65.5, the `lutd+h+grp+pq+r` arm (*measured* in M14, another session than M2, so the
  ratio is *estimated*). At 2,048: 23.2 to 43.3, and at 4,096: 15.8 to 37.6 (*measured*, ×1.750 and ×2.380 round by
  round, M4). This is the runtime's chat at length.
- **Criteria.** None beyond throughput: every kernel returns the served bits, and the grouped form the expanded
  logits over 49,140 decode steps (M5).
- **License.** Already measured. A door reading on a cool, idle Mac.
- **Cost.** A line in `served.rs` and a config, $0.
- **Gate.** Rulings 2 and 4 of [`plan-metal-debit.md`](plan-metal-debit.md) §3.

### M3. Serve the MLX path

- **Target.** 93.9 short and 70.3 at 4,096 (*measured*, M16, unserved).
- **Criteria.** Quality: its tokens leave the Rust path's at token 243 after 5 positions. T8 band on Metal. Peak
  memory 3.20 GB at 4,096 positions (M16).
- **License.** Perplexity on the 12-window design against 12.580769 (M5), in the band pooled and per window.
- **Cost.** One hour of the Mac's GPU, $0. The Mac is unusable during it: the operator picks the hour.
- **Gate.** Ruling 1 of plan-metal-debit §3: tolerance or bit for bit on Metal.

### M4. The MLX Tetra kernel

- **Target.** The Tetra matvec takes about 4.4 ms of a 10.65 ms token; a kernel at the same grid with no work costs a
  third of it (*measured*, ablation before M16). At 2.4 to 3.0 ms: 100 to 116 tok/s short (*estimated*). Leads in
  MLX's `qmv` order: one dispatch for gate and up, two blocks a lane interleaved, a threadgroup table of decoded byte
  quads.
- **Criteria.** Throughput. A load-time re-encoding would change GB: stated in b/param before (rule 6). Kernel
  arithmetic under M3's band.
- **Cost.** Days, $0, on a cool Mac.
- **Gate.** M3's reading first.

### M5. MLX at long context

- **Target.** From 5 to 4,096 positions the MLX path loses 4.6 ms a token and mlx-lm 1.9, with the same attention
  kernel (M15). At mlx-lm's loss: 70.3 to about 80 at 4,096 (*estimated*).
- **Cost.** A day, $0.

## 5. Order

| step | leads | card cost | running total | why here |
|---|---|---|---|---|
| 0 | M1, then the operator's rulings for M2 and K1 | $0 | $0 | $0; M1 says whether the runtime's loop or the engine binds its chat |
| 1 | K1 door; K4 microbench in the same job | about $0.67 | $0.67 | the cheapest served gain; K4 calibrates the loop before any 4E variant |
| 2 | K2, then K3 | about $0.70 | $1.37 | the 1.94 ms around the rows is the largest block above the weight floor |
| 3 | K6 T8 at 8B and 14B, then 4E port and doors | about $7.1 | $8.47 | the two larger sizes gain most per dollar |
| 4 | K7, then K4 in 4E if its bar held, then K5 | about $1.4 | $9.87 | 8k first, then the short-prompt residue |
| 5 | K8 | $3 to $5 | $12.9 to $14.9 | after the kernel, the operator's order |
| 6 | K9, K10 | about $2.6 | $15.5 to $17.5 | rulings first; both need weeks of code |
| Mac | M3 when the operator frees an hour, then M4, M5 | $0 | | no job on a Mac in use |

The project has spent $273.30 (*computed*, `ETAT.md` §5). No spend cap is in force, and one is owed before the next
paid job. End of the series, 4B on the L40S: 238 to 298 tok/s short and 202 to 301 at 8,192 (*estimated*, low ends at
half the arithmetic gain).

## 6. What this plan does not establish

- Any target above. The parts of §2 come from three jobs, and the 4E model of the rows is uncalibrated.
- Any card but the L40S. On the A100 the served files lost to vLLM FP16 ([`ETAT.md`](ETAT.md) §4), and no consumer
  card has been timed.
- Quality under any served form beyond what `ETAT.md` §2 records: no full task score under 4E.
- The cause of the runtime's Metal rate: M1 lists suspects, none measured.
