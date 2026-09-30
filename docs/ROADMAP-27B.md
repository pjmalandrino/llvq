# Roadmap 27B: Qwen3.8-27B through the Qwen3.5 test beds

Draft of 2026-09-28, not adopted. It plans the code, the Mac runs and the card runs that take the repository from
Qwen3 to `Qwen/Qwen3.8-27B`. Every step is tested first on smaller models of the same architecture. Nothing in it is
launched.

Rules 1 and 2 apply to every row. No run starts without the operator's go and its announced cost. Every row that
measures runs under a prereg stamped before its first measurement. The code backlog is in
[ROADMAP-27B-DEV](ROADMAP-27B-DEV.md). The raw Hub reads behind the numbers are in
[qwen35-recon](mesures/qwen35-recon-2026-09-28.txt).

## 1. What a 27B at 2.7 b/param would claim

A 4-bit 27B already fits the product triplet. RedHatAI's INT4 file weighs 19.45 GB, and 17.68 GB without its BF16
vision tower and MTP head, 5.26 b/param over the text model (*computed*). With 0.54 GB of f16 KV cache at 8k and
0.15 GB of DeltaNet state, it needs 18.4 GB of the 29.4 GB a 32 GiB card leaves after its 5 GB margin (*computed*).
Our 27B is therefore not the triplet's product point. `Tetra` admits 81 to 101 G there ([ETAT](ETAT.md) §4).

It becomes one on a 16 GiB card. That card leaves 12.2 GB after the margin. Our 27B needs about 9.9 GB, and a 4-bit
27B does not fit (*computed*). There the rivals are a 2-bit GGUF of the same 27B and the 9B at Q8_0, 9.53 GB.

The 2-bit rival already exists. ISTA's GSQ-RCO IQ2_S weighs 9.26 GB at 2.751 b/param over the text model (*computed*
from its header). Its card at `d562806d` reads wikitext2 perplexity 7.39 against 7.05 in BF16, ×1.048. Our trained
14B read ×1.060 on our harness ([dclm-14b-rowscales](mesures/dclm-14b-rowscales-2026-09-22.txt)). The two harnesses
differ, so this informs and does not compare. Our sealed 14B sits 3.22 MMLU points below FP16 ([ETAT](ETAT.md) §3).

So the first measurement is K of §4, on the Mac for $0. It needs one development item first, the GGUF letter ids of
P3, under half a day. It asks how much MMLU the rival loses at 27B, and whether the 9B at Q8_0 matches it at equal
memory.

## 2. Target and test beds

No Qwen3.8 exists below 27B. The 27B keeps the Qwen3.5 architecture: its text config equals Qwen3.6-27B's field for
field, and its 1,199 tensors carry the names and shapes of Qwen3.5-27B. Only the weights are new.

| model | revision | layers, DeltaNet + full | DeltaNet V/K heads | KV heads | tables, % of text | quantized weights | role |
|---|---|---|---|---|---|---|---|
| tiny random | written by the dumper | 4, 3 + 1 | 6/2, ratio 3 | 2 | untied | small | ratio 3, mixer widths that differ |
| Qwen3.5-0.8B | `2fc06364` | 24, 18 + 6 | 16/16, ratio 1 | 2 | tied, 34% | 0.497 G | code, oracle, full-depth reference |
| Qwen3.5-2B | `15852e8c` | 24, 18 + 6 | 16/16, ratio 1 | 2 | tied, about 27% | 1.372 G | the vLLM engine gate of C7 only |
| Qwen3.5-4B | `851bf6e8` | 32, 24 + 8 | 32/16, ratio 2 | 4 | tied, 15% | 3.565 G | noise, attribution, harness check with the 4B-Base `1001bb4d` |
| Qwen3.5-9B | `c2022362` | 32, 24 + 8 | 32/16, ratio 2 | 4 | untied, 23% | 6.912 G | the whole chain, untied like the 27B |
| Qwen3.8-27B, 8 layers | `1d4bf0f2`, 5 of 18 shards | 8, 6 + 2 | 48/16, ratio 3 | 4 | untied | 3.04 G | real 27B weights on the Mac |
| Qwen3.8-27B | `1d4bf0f2` | 64, 48 + 16 | 48/16, ratio 3 | 4 | untied, 9.45% | 24.327 G | the target |

Counts are *computed* from the safetensors headers, and from `config.json` for the 2B and 4B. The 27B text model
holds 26,895,998,464 parameters. The vision tower (0.461 G) and the MTP head (0.425 G) bring the checkpoint to
27,781,427,952. The 2B is never a noise bed.

The small beds test code. They do not predict the 27B's level. They carry Qwen3.5 weights, and their tables weigh 15
to 34% of the text parameters against 9.45%. On Qwen3 our gap to AWQ moved with size, from 4.76 points at 4B to 2.46
at 14B ([ETAT](ETAT.md) §3). None of them has the 27B's head ratio of 3; the tiny model and the truncated 27B cover
it. The 3.5 beds store `A_log` and the DeltaNet norm in F32, the 3.8 in BF16.

## 3. What the new block changes

48 of the 64 layers are Gated DeltaNet: a causal depthwise conv of width 4, then a delta-rule recurrence on an f32
state of 48 × 128 × 128 a layer. The 16 others are full attention with an output gate. Nothing in `llvq-llm` reads
`layer_types`, and candle 0.9.2 has no `qwen3_5` model. The oracle reference therefore becomes transformers. Eight
changes each give a finite, fluent and wrong model if missed:

1. RMSNorm is zero-centred, `x̂·(1+w)` in f32, on the input, post-attention, final, `q_norm` and `k_norm` norms.
2. The DeltaNet output norm uses a plain `w`, then multiplies by `silu(z)`.
3. `q_proj` has 12,288 rows, 256 of query then 256 of gate per head. The context is multiplied by `sigmoid(gate)`
   before `o_proj`. The 3.6 and 3.8 configs carry `output_gate_type: swish`, which transformers and vLLM ignore.
4. RoPE covers the first 64 of 256 dims, theta 1e7. mrope reduces to 1-D RoPE for text.
5. V head `h` reads K head `h / 3` (`repeat_interleave`). llama.cpp permutes its weights to a tiled order, so a
   kernel ported from ggml is wrong on HF weights.
6. `g = −exp(A_log)·softplus(a + dt_bias)` in f32; `beta = sigmoid(b)` in the model dtype.
7. The conv output passes through SiLU before the q/k/v split, and q is scaled by 1/√128 after its l2norm.
8. Prefill runs the chunked rule (chunk 64, decay masked before the exp), decode the recurrent rule. Only the 16
   full-attention layers carry a KV cache.

Quantized, 400 matrices: `in_proj_qkv` [10240, 5120], `in_proj_z` [6144, 5120], `out_proj` [5120, 6144], `q_proj`
[12288, 5120], `k/v_proj` [1024, 5120], `o_proj` [5120, 6144], and the MLP, whose shapes equal the Qwen3-14B's.
Carried: `in_proj_a/b` [48, 5120], `conv1d`, `A_log`, `dt_bias` and every norm, 26.2 M parameters (*computed*),
about 0.016 b/param in f16. The GGUF, AWQ, GPTQ and FP8 builds we read keep them at 8 bits or more.

The `rot_apply` wall of [format-noyau](format-noyau.md) §8 does not block the 27B. Its widest activation is 17,408,
the 14B's, which stages 69,632 B and passes with the opt-in (*computed*). The wall stays on the 32B.

## 4. Sequence and gates

The order puts the Mac before the cards and the served path last. MMLU is read on the dense reconstruction, so no
quality number waits for the kernel.

1. M0: free the Mac's disk. The GGUF letter ids of P3, then K on the Mac.
2. P0, P1, then G1 on the tiny model, the 0.8B, 2B, 4B and 9B, and the truncated 27B.
3. P2 without the export, then G2 on the 0.8B.
4. The rest of P3, then G3, the noise (M6) and the attribution (M7) on the 4B.
5. The 9B chain on the Mac (M8, M9), then G5. The 27B quote on the Mac (M10, G6).
6. The operator sets the cap of the first cards, and decides the 27B spend and its cap, with K, G5, G6 and the noise
   in hand.
7. Cards, after P2 rows 9 and 10 (`launch` calls the estimate): image smoke and first contact (C0, C1), the 0.8B card
   gate (C2), the 27B oracle (C9, C10), C14 if G6 failed, the encode (C15, G7), the base census (C16).
8. P4, P5 and the baselines are built while the 27B encodes. Then the 9B cards (C3 to C8), decision 7, the 27B
   references (C11 to C13), the row scales and trained census (C17, C18), and G8. G4 and the served runs (C19 to C21)
   come last.

| gate | kind | passes when | otherwise |
|---|---|---|---|
| K | measurement | informs the operator: the IQ2_S loss below Q8_0 at 27B, and IQ2_S against the 9B at Q8_0 | the operator re-scopes: a lower rate, or the 27B as a scale point only |
| G0 | gate | `tetra_lock` and `oracle Qwen/Qwen3-0.6B` pass on the Mac before each merge | the change does not merge |
| G1 | gate | per-layer error under the bound the oracle prereg stamps, top-1 identical on 16 decode steps, CPU and Metal; every mutant of the backlog turns it red | fix the forward; no encode |
| G2 | gate | the 0.8B resumes byte-identical, its sealed ppl sits within 1% of its encoding, and the three-seed R of M4 (encoding perplexity over the unquantized model's) is stamped as the Qwen3.5 reference | fix the chain; no card |
| G3 | gate | the chat-wrapped control lands within 1.5 pp of raw 5-shot, p ≥ 0.05, paired on 2,280 questions, at 4B and 4B-Base | raw 5-shot is not the harness on Qwen3.5; operator decision |
| G5 | measurement | informs the operator: our 4B and 9B base arms against unsloth UD-IQ2_XXS, after the llama.cpp engine gate | the operator decides the 27B spend in every case |
| G6 | gate | M10's s/block lands within 30% of its Mac quote, 745 s/block (*computed* at 1.96 µs a weight); its peak is recorded and does not quote a card | C14 on the card, then requote |
| G7 | gate | the 27B sealed ppl sits within 1% of its encoding, and its bytes match the prediction | no census |
| G8 | criterion | the sealed 27B reads under 3.00 kernel b/weight; its MMLU gaps to FP16, the 4-bit baseline and the 2-bit GGUF are published with paired CIs on the full split | publication is the operator's decision; no 9B projection pronounces it |
| G4 | gate | CUDA oracle green; the served arm first diverges from dense at token 32 or later, and the position is published | before token 32: no served number until explained |

G5 checks that the method survives the new architecture, and little more. We beat our own imatrix IQ2_XXS build of
Qwen3-4B by 23.6 points ([ETAT](ETAT.md) §3), so a win at 9B is expected. No unsloth file has been scored yet. ISTA
publishes no 9B, so the 27B rival is read at 27B, in K.

No Qwen3 noise bar is inherited. A comparison at constant file is read on its own paired CI from `mmlupair`. A
comparison that recalibrates is read against the σ of M6, and none is read before M6 lands. If that σ exceeds the
2.92 pp of Qwen3-4B (*measured*, [bruit-mmlu](mesures/bruit-mmlu-graines-4b-2026-08-25.txt)), the operator re-reads
G5 before any 27B spend.

Twelve preregs are planned: the oracle tolerance (before M1, and for C1, C9, C10), K, the 0.8B chain (M3, M4, C2),
the harness (M5, G3), the noise (M6), the attribution (M7), the 9B chain and G5 (M8, M9), the 9B cards (C3 to C8), the
27B references (C11 to C13), the 27B encode (M10, C14, C15), the 27B censuses and row scales (C16 to C18), and the
27B served runs (C19 to C21). Each quote is checked against `docs/data/jobs.csv` after its job.

The critical path to step 6 is about 26 to 41 development days (*computed* from the backlog sizes). K lands a few
days after M0 and the letter ids.

## 5. Mac runs

All $0, under `nice 10` with `LLVQ_THREADS=12`. Durations are *estimated* from the Mac rate of 1.96 µs a weight
([dclm-8b](mesures/dclm-8b-2026-09-21.txt)) and the 0.8 h a 2,280-question arm takes at Qwen3-4B (*estimated* in
`bin/mmlu.rs`).

| step | model | what | duration |
|---|---|---|---|
| M0 | none | free at least 150 GB. 12 GiB were free on 2026-09-28; the Hub cache holds 58 GB and the home directory 52.6 GB of `.llvq` and `.bin`. The operator chooses what goes | minutes |
| K1 | 27B, 9B | llama.cpp b10050 on Metal, the 2,280-question plan through `gguf_mmlu`: unsloth 27B Q8_0 (29.05 GB), ISTA IQ2_S (9.26 GB), unsloth 9B Q8_0 (9.53 GB); bartowski IQ2_XS (2.629 b/param) optional. The Q8_0 file carries the MTP head as a 65th block, so its load is checked first | 3 to 6 h an arm |
| M1 | tiny, 0.8B, 2B | dumps; oracle on CPU f32, Metal f32 and f16; cache identity | about 1 h |
| M2 | 4B, 9B, 27B at 8 layers | same oracle; downloads 9.3, 19.3 and 16.9 GB; the 9B never runs beside its dumper | about 2 h |
| M3 | 0.8B | encode: block 0, resume to 1, all 24; seal, `embedq q4`. Transformers parity waits for the export (P2 row 8) | about 1 h |
| M4 | 0.8B | full-depth reference: the recipe at 24 blocks, same seed rerun, then three calibration seeds | about 3 h |
| M5 | 0.8B, 4B, 4B-Base | BF16 anchors on the 2,280 questions; raw against chat-wrapped (G3); the BF16 GGUF of the 4B through `gguf_mmlu` (engine gate) | 1 to 3 h an arm |
| M6 | 4B | noise: three calibration draws, perplexity and the 2,280 questions | 3 to 4 h a draw |
| M7 | 4B | attribution at constant file: `LLVQ_RESTORE_F16` for each of the ten types on two of M6's draws, then `LLVQ_RESTORE_Q4` on the types that carry the loss; the int4 allocation ranked on the 1,531 validation questions, then measured once on test | about 16 h for the first 20 arms |
| M8 | 4B, 9B | unsloth UD-IQ2_XXS at 2.871 and 2.841 b/param; ISTA's 4B GSQ at 3.671 as context only | 1 to 2 h an arm |
| M9 | 9B | full encode and seal, then its base arm on the 2,280 questions (G5) | 5 to 6 h; the 8B peaked at 114.5 GB with swap |
| M10 | 27B at 8 layers | encode in f32: s/block, peak footprint, DeltaNet calibration share | about 1.7 h |
| M11 | 27B | optional full-depth f32 dump with a streaming dumper, which replaces the cpu-performance dump of C10 | 55.6 GB download, hours |

The 9B is encoded on the Mac and the 27B on a card. At 4B, bare `Tetra` calibrated on C4 read +9.35% perplexity and
−1.18 MMLU points on a card against Metal, one pair, cause not established (*measured*,
[encode-14b prereg](../proofs/preregistration-encode-14b-2026-09-22.md)). Every 9B-to-27B reading carries that term,
declared and not bridged. If M0 frees less than 80 GB, M9 moves to a card for about $7 (*estimated*).

## 6. Card runs and costs

Spent to date: $241.88 over 214 priced jobs (*measured*, `docs/data/jobs.csv`). No cap is in force; one is owed
before C0 ([ETAT](ETAT.md) §5). Costs below are *estimated* from rows of `jobs.csv` and the prices of
`ops/run.py:92-110`, unless the basis says otherwise. Add 10% for reruns. Rule 9 runs before any rerun is costed. One
job writes to the bucket at a time. The `.llvq`, the sealed file and the export go to `/scratch`, then to the bucket
in parts of at most 1 GB, checked by `hf buckets ls`. Queues on the L40S have run 5 h 43 before start (*measured*,
`jobs.csv`, census-14b-base).

**Small models, $18 to $25.** C0 to C2 run in step 7 of §4, C3 to C8 in step 8.

| step | flavor | what | cost | basis |
|---|---|---|---|---|
| C0 | l40sx1 | image smoke after each Space rebuild, four to five rebuilds | $0.5 to $1.4 | `image-smoke-14b.sh` header estimate, $0.11 to $0.27; no row in `jobs.csv` |
| C1 | l4x1, rtx-pro-6000, h200 | oracle on the tiny model and the 0.8B, then first contact on each card that runs candle; a Qwen3-4B MMLU sample on the h200 to time its census rate | $0.7 to $0.9 | oracle rows at $0.01 on l4x1 |
| C2 | the card of decision 3 | 0.8B encode gate: `tetra1`, int4, resume | $0.4 to $0.85 on rtx-pro-6000, $1.3 to $1.7 on h200 | the 14B in-job gate, $0.37 estimated |
| C3 | l40sx1 | served 0.8B and 9B: prefill gate, divergence position, median of five rounds with range | $0.45 to $0.85 | `paper-served-4b/8b` |
| C4 | l40sx1 | DeltaNet kernel against the candle-op arm, one process | $0.2 to $0.4 | same |
| C5 | h200 | 9B row scales; logs the probe s/step and the peak that C17 needs | $6 to $8 | 8B $5.66 at 0.40 s/step |
| C6 | l40sx1 | 9B MMLU census: FP16, base, trained, sealed | about $4 | 8B arms at $0.97 |
| C7 | l40sx1 | 9B FP16 and 4-bit references in vLLM; engine gate against our dense path on the 2B | $2.5 to $4.5 | 4B gate $4.71 at 23.4 ms a token |
| C8 | l40sx1 | 9B GSM8K through the served kernel | $3.1 to $3.6 | 8B $2.37 at 10.9 ms a token, plus DeltaNet launches |

**27B, $71 to $157** at the ends of the rows, with one sealed arm, one h200 and no second card (*computed* from the
rows). Two more sealed arms add up to $16. An h200x2 training adds up to $14. C20 adds $2.8 if decision 10 adds the
card.

| step | flavor | what | cost | basis |
|---|---|---|---|---|
| C9 | l40sx1 | CUDA oracle on the 8-layer 27B against the Mac dump | $0.3 to $0.45 | 16.9 GB download |
| C10 | cpu-performance, rtx-pro-6000 | full-depth f32 dump, unless M11 made it; bf16 oracle on the rtx-pro-6000 in every case, about 10 min. Its absmax decides whether f16 serves the 27B | $0.5 to $4 | 55.6 GB download |
| C11 | l40sx1, rtx-pro-6000 | vLLM: 4-bit speed on the L40S beside C19; FP16 smoke and GSM8K on the rtx-pro-6000, FP16 speed there only under decision 10; one KV dtype for every arm | $2 to $4 | 14B at $0.31 and $0.20, ×1.84 |
| C12 | h200 | MMLU references, FP16 and 4-bit, our harness, one job per arm | $8 to $16 | `census-14b-ref` 105 min ×1.84; h200 at 1× to 2× the L40S rate |
| C13 | Mac or l40sx1 | GGUF arms on the full split: IQ2_S 2.751, IQ2_XS 2.629, UD-Q2_K_XL 2.816 b/param | $0 to $9 | IQ2 4B census $0.43, ×6.7 |
| C14 | the card of decision 3 | 4-block quote, only if G6 fails | $0 to $5.5 | 405 to 412 s/block *measured* at 14B, ×1.15 |
| C15 | h200 (f32) or rtx-pro-6000 (bf16) | encode in three segments on one image, seal, sealed ppl, export in parts | $28 to $54 | 1.226 µs a weight at 14B ([encode-14b](mesures/encode-14b-2026-09-22.txt)) × 24.33 G = 8.3 h of loop |
| C16 | h200 | base census | $5 to $10 | 14B 65 min ×1.84, same card terms |
| C17 | h200 | row scales with working gradient checkpointing, 2,500 or 9,507 steps | $7.5 to $24 | 14B 0.69 s/step ×1.82 ×1.33; about 120 GB of 150 |
| C18 | h200 | trained census, then one sealed arm | $9 to $18 | 65 and 53 min ×1.84 |
| C19 | l40sx1 | served speed with q4 tables and head-f16 arm; an 8,192-token prompt with KV f16 and q8 and the VRAM peak; kernel bench | $1.5 to $2 | 14B served $0.47; 8B bench $0.19 |
| C20 | rtx-pro-6000 | only under decision 10: same-card table, dense, same-head and `Tetra` arms interleaved in one `fusedrun`; vLLM FP16 and 4-bit in one process each; × within a stack, GB/s across (rules 4, 5, 7) | about $2.8 | 14B rows ×1.85 |
| C21 | l40sx1 | GSM8K through the served kernel, two jobs of 155 to 170 min | $9.2 to $10.2 | 14B 141 min at 17.9 ms a token |

The served 27B should weigh about 9.2 GB on card, with 537 MB of KV cache at 8k and 151 MB of DeltaNet state
(*computed*). Speed on the L40S should land at 23 to 26 tok/s with candle-op DeltaNet, and at 31 at most by bytes
from the 14B's 57.2 (*estimated*).

## 7. Risks

- The rival. ISTA's IQ2_S already sits at our rate with a small perplexity loss. If K shows it loses less MMLU at 27B
  than our 14B lost, the 27B must beat our own record to publish a win.
- Launch count. Composed from candle ops, DeltaNet adds about 2,160 launches a token at 27B, estimated at +7 to 11 ms.
  CUDA Graphs stay closed, so the step kernel is the only lever.
- Perplexity can lie here. In a W4A4 build, a forget gate mis-scaled by vLLM's fused `in_proj_b+a` GEMM moved 32k
  perplexity from 10.84 to 6.86 while AIME fell from 86.7 to 80.8 (arXiv 2609.04098 §6). MMLU and the gate check of
  P3 carry the verdict.
- `out_proj` is the most sensitive DeltaNet projection: 12.7% layer-output error at NVFP4 W4A4, against 2.1% for
  `in_proj_a` (same paper, Table 3, one projection at a time). Expect it in int4.
- The thinking-by-default model. A raw-completion harness read MMLU-Pro 66.3 against 80.4 (same paper). Our harness
  scores logits. It stays unvalidated on this model until G3 passes.
- f16 overflow. The served path is f16 end to end. A full-depth absmax above about 3.2e4 would need a bf16 kernel
  path, sized L and not costed here.
- No Qwen card publishes a 5-shot MMLU or a GSM8K. Third parties publish GSM8K 95.5 for the BF16 27B with thinking
  off (arXiv 2609.04098, Table 1), so our FP16 GSM8K should land near it. The MMLU anchors are ours alone.
- Bucket writes of 4 GB and 29.5 GB have vanished without an error (*measured*, `jobs.csv`, 2026-08-10 and
  2026-09-22). The 27B `.llvq`, sealed file and export weigh about 6.5, 11 and 54 GB (*computed*).

## 8. Decisions for the operator

| decision | options | default if silent |
|---|---|---|
| 1. adopt the plan, its place and its rank | two files beside ROADMAP-QUALITY; ROADMAP §4 "a model above 14B" moves to §2, before or after §2.1 | draft, not adopted; ROADMAP order unchanged |
| 2. M0: what leaves the Mac's disk | Hub cache (58 GB), home `.llvq` and `.bin` (52.6 GB), elsewhere | nothing deleted; K and M2 cannot run |
| 3. 27B encode memory | f32 on h200 (about $52, recipe intact); bf16 on rtx-pro-6000 (about $28, declared confound and a control); block-streaming `smoke` (L, then f32 at about $28 or on the Mac in about 13 h) | f32 on h200 |
| 4. b/param denominator | text only, 26,895,998,464, for every arm, MTP and vision removed from every numerator, declared beside each figure as a departure from rule 6; or the checkpoint's 27,781,427,952 | text only |
| 5. 4-bit baseline | `RedHatAI/Qwen3.8-27B-INT4@91bd022d` (AWQ smoothing then GPTQ, sym g128, weights of 2026-09-06, published evals); `mattbucci/Qwen3.8-27B-AWQ@2cc4cffb` (GPTQ repacked as AutoAWQ, 45% image and video calibration, one individual); our own AWQ. None is plain AWQ as in the Qwen3 tables, so the column becomes "4-bit g128" | RedHatAI, pinned and mirrored |
| 6. 2-bit arms in the tables | ISTA IQ2_S (2.751), bartowski IQ2_XS (2.629), unsloth UD-Q2_K_XL (2.816), turboderp EXL3 | the first three |
| 7. row scales at 27B | one h200 with gradient checkpointing; h200x2; none. The gain was +3.15, +3.29 and +1.67 at 4B, 8B and 14B (*measured*, [ETAT](ETAT.md) §4) | decided after C5 and C6 measure it at 9B |
| 8. MMLU scoring at 27B | dense reconstruction on an h200, or through the kernel, about 35 h a census without a prefill kernel (*estimated* from the 4B kernel census rate of 2026-09-11). Dense repeats the limitation ROADMAP §2.1 exists to close | dense on the h200 |
| 9. DeltaNet CUDA kernel before 27B numbers | yes, or quality first | quality first |
| 10. a second card | the rtx-pro-6000 for the same-card table of C20, which is ROADMAP §2.6 | L40S absolute numbers only |
| 11. MTP head | serve it for speculative decoding later, or drop it | dropped |
| 12. what the 27B claims | a scale point for the paper, or the product point of a 16 GiB card, a new triplet | a scale point |
| 13. F32 carry on the 3.5 beds | a raw F32 tag in the format, or f16 narrowing measured by the oracle | f16 narrowing, measured |

## 9. Adoption edits

On adoption: CLAUDE.md lines 15-16 (the wall), line 23 (the new pointer) and the `smoke` example at line 72, which
shows the August `leech1c12` recipe and not `tetra1`. Also the ROADMAP header, §2 and §4; ETAT §4 and §5; the docs
README table and its count of fourteen; README line 102; a HISTORIQUE entry dated 2026-09-28.
