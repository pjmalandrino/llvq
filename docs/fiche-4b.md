# 4B datasheet

Provenance register of `Qwen3-4B-LLVQ-2bit` and its fused kernel: every number of the published file has its row here,
with source, instrument, dtype, protocol and label (*measured*, *computed*, *estimated*). Where a document diverges from
the object, the object wins. Current state: [ETAT.md](ETAT.md); dated verdicts: [HISTORIQUE.md](HISTORIQUE.md);
measurement rules: [METHODE.md](METHODE.md); CUDA layouts: [format-noyau.md](format-noyau.md). Metal measurements on a
MacBook Pro Mac15,8, M3 Max, 16 CPU cores, 40 GPU cores, 68,719,476,736 bytes (*measured*, `system_profiler`), peak
400 GB/s (*estimated*, spec); CUDA on L40S.

## 1. Identity
| field | value | label, source |
|---|---|---|
| name | `Pier-Jean/Qwen3-4B-LLVQ-2bit`, file `qwen3-4b-llvq.bin` | HF repo at commit `f00daa7bc1dd12a720304a4483f2219d10f15c96` |
| size | 1,770,527,533 B (1.771 GB) | *measured*, `shasum`; HF `content-length` identical (2026-08-03) |
| sha256 | `9db213ef9fa9d7d7000789a8a529ce9459ce9ba6002ef5a72fd5a1c05c1c84b0` | *measured*, identical to the HF `x-linked-etag` |
| magic | `LVQ2`, three sections, parses to the exact byte | *measured* |
| sealing | 2026-07-31 17:56 (mtime) | *measured* |
| binary | commit `51d7c55` (2026-07-31 12:36:19) | *measured*, four clues (§4) |
| local copy | `/Users/pjmalandrino/qwen3-4b-llvq.bin` | same sha256 |

The HF repo holds `.gitattributes`, `LICENSE`, `README.md` and the `.bin`; reproduce with `shasum -a 256` and
`curl -sIL https://huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-2bit/resolve/main/qwen3-4b-llvq.bin`. The file loads with no
network and no checkpoint (`bin/run`, `bin/ppl`, `bin/mmlu`, `bin/fusedrun`, benchmarks `thesis`, `matvec`, `decreal`) and
opens in no other engine. The HF widget does not serve it: `config.json` and `tokenizer.json` live inside the `.bin`.

## 2. Byte-by-byte content
| section | bytes | content | label |
|---|---|---|---|
| matrices | 980,790,202 | 252 quantized matrices | *measured* |
| raw tensors | 778,313,898 | 146 f16 tensors | *measured* |
| blobs | 11,423,433 | `config.json` 726 B, `tokenizer.json` 11,422,654 B | *measured* |
| total | 1,770,527,533 | = file size, difference 0 | *measured* |

The matrix section splits into payload 980,770,752 bytes (7,846,166,016 bits, §5.3) and framing 19,450 bytes (*computed*
on measured inputs, byte-exact loop). The framing counts the 8-byte header, 10,370 for the names, 252 × 28 of metadata
and 252 × 8 of prefixes.

| carried tensors | values | label |
|---|---|---|
| `model.embed_tokens.weight` [151936, 2560] | 388,956,160 | *measured* |
| norms: 36 × (2560 + 2560 + 128 + 128) + `model.norm` 2560 | 196,096 | *measured* |
| total carried, 146 tensors | 389,152,256 | *measured* |

No `lm_head`: `tie_word_embeddings` is `true`. The 146 tensors are equal bit for bit to f16(checkpoint bf16), 146/146
verified (*measured*). The blobs are byte-for-byte copies of the checkpoint: sha256 of `tokenizer.json` =
`aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`, the upstream HF blob. Tokens and MMLU prompts are
therefore identical between the two arms by construction. `config.json`: hidden 2560, intermediate 9728, 36 layers,
head_dim 128, 32 heads, 8 KV, vocab 151,936, bfloat16 (*measured*). The `389_070_848` constant in `thesis.rs:432`
matches no tensor (2¹⁴ × 23,747), effect +0.03% on the tok/s: to be fixed, never to be quoted.

## 3. Weight counts
Read back from the matrix headers (*measured*, 2026-08-03).

| quantity | value | note |
|---|---|---|
| matrices | 252 | 36 blocks × 7 projections |
| projection weights | 3,633,315,840 | file and `~/llvq-run-4b-artefact.log` |
| of which quantized | 3,616,358,400 | |
| of which `KeepExact` tail | 16,957,440 (0.4667%) | 471,040 per layer |
| blocks of 24 | 150,681,600 | read back, not derived |
| output rows | 1,105,920 | the rows verified by the kernel benchmark; also the number of row scales |
| gain centroids | 504 | 2 × 252 |
| blocks at gain level 0 | 72,008,871 (47.79%) | mean centroid 0.8723 |
| blocks at gain level 1 | 78,672,729 (52.21%) | mean centroid 1.1146; 0 blocks coded at the origin |
| total model parameters | 4,022,468,096 | 3,633,315,840 + 389,152,256 (*computed*) |

Shapes: q 4096 × 2560, k 1024 × 2560, v 1024 × 2560, o 2560 × 4096, gate and up 9728 × 2560, down 2560 × 9728. Tails:
2560 % 24 = 16, 4096 % 24 = 16, 9728 % 24 = 8. Per matrix, the fraction of blocks at level 1 runs from 0.4660 to 0.7604,
median 0.5143; the centroids are strictly increasing over the 252, mean ratio 1.2791. `~/llvq-q4b.llvq`
(980,790,202 bytes, magic `LVQ1`) carries over bytes [8, 980,790,202) the same sha256 as the sealed file,
`5acd89c07afc143ce12ab5a04a4a24ba38f8bd7f0601d049e14e734715725a6b` (*measured*): `bin/seal` re-encodes bit-identical.
Its own sha256 is `94f60e86…`; it is the default file of the three Metal benchmarks (`thesis.rs:191`, `matvec.rs:503`,
`decreal.rs:139`), hence the object of the 08-01 `thesis` runs.

## 4. Configuration
| setting | shipped value | what the code does | proof |
|---|---|---|---|
| codebook | `leech1c12` | `LeechShapeGain::with_caps(centroids, cap = 12, level_cap = 5)` | *measured*: `shell_cap = 12` on 252/252 |
| level cap | none | `MAX_LEVELS_ANY = 5`, the structural maximum | *measured*: the `L<n>` token dates from commit `fabab22`, 25 h after sealing |
| index | 47 bits | `⌈log₂ N(12)⌉`, N(12) = 111,043,117,458,000 | *measured*: stream = nblocks × 6 B; max index observed 111,043,117,450,038 |
| gain | 1 bit, 2 centroids per matrix | Lloyd-Max, 40 iterations, relative block norms, rotated weights | *measured*: 2 centroids on 252/252 |
| bits per block | 47 + 1 = 48, 6 bytes, MSB-first, no padding | 2.000000 b/weight of code | *computed*, exact |
| row scales | 1,105,920 in f64 | `row_scale = sqrt(Σ row² / (d_in/24))`, frozen before the loop | *measured*: 0 of 1,105,920 representable in f32, 0/504 centroids |
| tail | `TailPolicy::KeepExact`, f32 on disk | receives the error feedback, produces no error of its own | *measured* |
| rotation | input only, seed `0x110FEED` | `Q = (Q_odd ⊗ H_m) D`, seed `base ^ (block<<32) ^ (act<<16)`, 144 distinct seeds | *measured*: 252 seeds reproduced without exception; no `rotate_weight_cols` |
| `group_scales` | off | arg 5 = `nogs`, and `ensure!(!cfg.group_scales)` at write time | *measured* |
| retraction | `true`, no-op | `retraction_target()` returns `None` under `retract_to_level` | *measured* |
| damping | 1e-2, relative to `mean(diag H)` | hard-coded at run time | *measured*; swept in batch B, no effect ([verdicts](archive/verdicts-lot-b-2026-08-06.md)) |
| dtype | f32 everywhere | literal `var_builder(DType::F32)`; `LLVQ_DTYPE` came later | *measured* |
| calibration | C4 validation shard 00000, 64 × 2048 = 131,072 tokens, contiguous prefix | `LLVQ_CALIB_SEED` did not exist | *measured* |
| encoding threads | 16 | resolved value, line 1 of the log | *measured* |
| scope | 36 blocks out of 36, 252 matrices | | *measured* |

Command line that produced the object, then sealing:
```bash
LLVQ_MODEL=Qwen/Qwen3-4B LLVQ_CALIB=c4 LLVQ_ARTIFACT=/Users/pjmalandrino/llvq-q4b.llvq \
cargo run --release -p llvq-llm --features metal,fast-linalg --bin smoke -- \
  64 2048 12 4096 metal nogs leech1c12 999 rot
LLVQ_MODEL=Qwen/Qwen3-4B cargo run --release -p llvq-llm --bin seal -- \
  /Users/pjmalandrino/llvq-q4b.llvq /Users/pjmalandrino/qwen3-4b-llvq.bin
```
Positionals at commit `51d7c55`: 0 n_calib, 1 calib_len, 2 n_eval, 3 eval_ctx, 4 device, 5 tests `== "gs"`, 6 codebook,
7 limit, 8 `rot`, 9 absent. The `1c12` codebook means 1 gain bit, shell 12, no `f` suffix; any limit value ≥ 36 is
equivalent. Four clues date the binary. The `.llvq` mtime minus 14,447 s puts the start around 12:40; the "model dtype",
"hessian damping" and "phases" lines are absent; the counter reads `block N/405`. `--features fast-linalg` is not
traceable: the guard that prints it came after the run. Facts about the recipe:
- The shipped recipe is Algorithm 1 (shape-gain, gain reset) plus an incoherence rotation on the input;
  "Spherical GPTQ" names the crate, not the recipe.
- The configuration line in `smoke`'s logs is a hard-coded literal ("0 gain bits, spherical retraction");
  only the `leech1c12` result line is reliable.
- The Eq. 17 retraction is a no-op under a coded gain: `quantize` has already put the block on the level's sphere.
- Algorithm 3 (`refine_group_scales`) is disabled twice over.
- `block N/405` counts the 405 column blocks of `down_proj` (9728 / 24), not layers; `/36` before `51d7c55`.

The published command reproduces the method without reproducing the bytes. Two blockers (*measured*, git):

| blocker | fact | consequence |
|---|---|---|
| corpus | commit `aba3989` (2026-08-01) moves `LLVQ_CALIB=c4` from shard 00000 to shard 00001 | the published command calibrates at HEAD on different text; no C4 ppl of the object can be produced without contamination |
| container | at `51d7c55` the writer was `artifact2.rs`, magic `LVQ1`, empty `finish()`; at HEAD `ArtifactWriter`, magic `LVQ2`, two zero `u32` | a re-run yields 980,790,210 B with a different magic; the matrix records stay comparable |

A third party on CUDA will not get the same weights: `calib.rs` accumulates AᵀA in f32 on the accelerator, difference
not quantified.

## 5. The numbers
### 5.1 Perplexity
Wikitext-2 test, ctx 4096, 12 non-overlapping windows, 49,140 scored tokens (4,095 × 12), f32 logits before `log_softmax`.

| LLVQ | baseline | × | object | dtype | instrument | trace | label |
|---|---|---|---|---|---|---|---|
| 16.9617 | 12.2336 | 1.3865 | in-memory model, before and after rewriting the 252 projections | f32 | `ppl` loop of `smoke` | `~/llvq-run-4b-artefact.log` | *measured*; iso-conditions by construction, `verify_artifact` ties this model to the published bytes |
| 16.9415 | 12.2361 | 1.3845 | published bytes against checkpoint, fingerprint `3f1baca9033bf251` on both sides | f16 | `bin/ppl`, Metal | body of commit `8c17eff`; replayed `~/ppl-scelle-f16-2026-08-04.log`, `~/ppl-base-f16-2026-08-04.log` | *measured*, reproduced to the ten-thousandth |
| 16.9422 | 12.2369 | 1.385 | published bytes, L40S, same fingerprint | f16 | `bin/ppl`, CUDA | [a4-campagne](mesures/a4-campagne-2026-08-06.txt) | *measured*; per-window NLL in [the raw log](mesures/a4-campagne-4b-ppl-BRUT-2026-08-06.txt) |
| 16.9358 | 12.2369 | 1.384 | q8 embedding variant (`q4b-e8.llvq`) | f16 | `bin/ppl`, CUDA | LLVQ: [final campaign arm 4](mesures/campagne-finale-bras4-2026-08-07.txt); baseline: a4-campagne, same fingerprint | *measured* on both sides; the × is *computed* between the two journals |
| 15.3272 | | | overlay `~/llvq-q4b-c12.safetensors`, night run, binary predating the `60068db` fix, 2.6923 b/weight actual, quantizer of commit `db84454` | f32 | `smoke` | `~/llvq-run-nuit.log` | *measured*, not a reference |
| 14.2684 / 15.2909 / 14.9104 | 12.2336 | | in-memory models, cap 13, announced at 2.1117 b/weight, actual rate 2.7338 | f32 | `smoke` | table rows, no log | *measured*, not a reference; the 14.2684 / 15.2909 pair is the spread observation of §5.7 |

Invocations of the f16 pair, criterion = same fingerprint `3f1baca9033bf251`: `LLVQ_DTYPE=f16 cargo run --release -p
llvq-llm --features metal --bin ppl -- 4096 12 metal /Users/pjmalandrino/qwen3-4b-llvq.bin` (sealed) and
`LLVQ_MODEL=Qwen/Qwen3-4B LLVQ_DTYPE=f16 … --bin ppl -- 4096 12 metal` (checkpoint); expected 16.9415 / 12.2361.

Log-likelihood overhead over its own baseline (*computed*), the only cross-paper comparison that holds:

| arm | Δ nats/token | vs QTIP |
|---|---|---|
| us, f32 | ln(16.9617) − ln(12.2336) = 0.326772 | +3.06% |
| us, f16 on the file | ln(16.9415) − ln(12.2361) = 0.325376 | +2.62% |
| QTIP (17.04 / 12.41) | 0.317061 | |
| 0-bit LLVQ of the paper (17.05 / 12.41) | 0.317648 | +0.19% |

We sit above QTIP and above the paper's 0-bit config before paying 8.5% more bits ([notes](llvq-paper-notes.md)).

### 5.2 MMLU
Hendrycks 5-shot, dev split of the same subject, logits of the `" A".." D"` tokens compared in f32, one forward pass per
question. 2,280 questions out of 14,042 (40 per subject, 57 subjects), seeded draw
`SplitMix64(0x6_11B0 ^ subject.len())` on both arms.

| arm | micro | macro | card | dtype | trace | label |
|---|---|---|---|---|---|---|
| f16 checkpoint | 70.32 ± 1.28 | | L40S | f16 | [a4-campagne](mesures/a4-campagne-2026-08-06.txt), fingerprint `65dcd53655e8bfa5` | *measured*, reference |
| LLVQ, published bytes | 55.59 ± 1.35 | | L40S | f16 | same | *measured*, reference |
| LLVQ, q8 embedding | 55.70 ± 1.35 | | L40S | f16 | [final campaign arm 4](mesures/campagne-finale-bras4-2026-08-07.txt) | *measured*; within the noise |
| drop f16 → LLVQ | −14.73 pp, paired 95% CI [+11.98; +17.47] | | | | [mmlupair 4B/8B](mesures/mmlupair-4b-8b-2026-08-13.txt) | *computed*; paper −9.5 pp (60.7 / 70.2) |
| f16 checkpoint | 70.42 ± 1.28 | 72.85 | Metal | f16 | [mmlu-micro-2026-08-02.log](mmlu-micro-2026-08-02.log), 2,620 s | *measured*, not a reference |
| LLVQ, published bytes | 56.09 ± 1.36 | 57.59 | Metal | f16 | same log, 2,805 s, per-subject profile | *measured*, not a reference |
| drop, Metal | −14.33 pp | −15.26 pp | | | | *computed*, not a reference |

"Micro" is a stratified estimator: the 57 rates are reweighted by the real population of each subject.
`professional_law` weighs 1,534 / 14,042 = 10.9%, estimated on 40 draws. The ± is a stratified standard error at 1 σ,
not a 95% CI; it excludes model, prompt and seed. The bar on a difference is the paired one (McNemar): 0.43 pp at
constant file, 0.79 to 1.44 pp between models (*measured*, [KV q8](mesures/kvq8-4b-2026-08-15.txt), mmlupair 4B/8B).
Harness validation: 70.42 against 70.2 in the paper, +0.22 pp, 0.17 σ (*computed*). The quantized arm loses 0.50 pp
between Metal and L40S on the same file, a difference that cannot be verified (the Metal log predates the fingerprints).
Profile (Metal): `abstract_algebra` 10/40, `professional_accounting` 10/40, `machine_learning` 12/40;
`international_law` 33/40; per-subject bar ±7 pp. Metal invocations:
`cargo run --release -p llvq-llm --features metal --bin mmlu -- Qwen/Qwen3-4B metal 40` and
`… --bin mmlu -- /Users/pjmalandrino/qwen3-4b-llvq.bin metal 40`; the `40` is the per-subject limit that defines the
2,280 questions, and the positional `.bin` proves that the quantized arm scores the published bytes.

### 5.3 Size and rate
Payload: 7,846,166,016 bits = 980,770,752 bytes (*computed*). Sum: 150,681,600 × 48 + 1,105,920 × 64 + 504 × 64 +
16,957,440 × 32.

| denominator | b/weight | what prints it | where it is published | use |
|---|---|---|---|---|
| 3,633,315,840 (projections, tail included) | 2.159506 | `bin/seal` | HF card | the consistent number, to be shown with its denominator named |
| 3,616,358,400 (quantized only) | 2.169632 | `bin/smoke`, "artifact:" line | README, CLAUDE.md | conservative variant, mixed ratio |
| ideal accounting (f16 tail, f16 scales) | 2.070226 | `Report::bits_per_weight`, "effective rate" | | never quoted for this file: it describes a file that was never written |

Breakdown over 3,616,358,400 (*computed*, loop to the seventh digit):

| item | b/weight |
|---|---|
| lattice code, 48 b/block | 2.000000 |
| tail in f32 | 0.150051 |
| row scales in f64 | 0.019572 |
| f64 centroids | 0.0000089 |
| total | 2.169632 (+8.48% over 2.000) |
| if tail and scales were f16 | 2.0799 (+4.0%) |

The headroom is the f32 → f16 tail, 0.075026 b/weight; the f64 scales cannot be reduced without breaking the
bit-for-bit proof. `format.rs` documents 0.0146 (f64 overhead over f16), the README 0.020 (total cost): two conventions,
neither false.

| compression | value | label |
|---|---|---|
| file | 1,770,527,533 B | *measured* |
| FP16 equivalent | 4,022,468,096 × 2 = 8,044,936,192 B | *computed* |
| ratio | ×4.5438 | *measured*, printed by `bin/run` and `bin/seal` |
| whole-model rate, f16 embedding | 3.5213 b/param | *computed* |
| `q4b-e8.llvq`, int8 embedding | 1,405,881,733 B, 2.7961 b/param | *measured*; −0.02% ppl, in production; matrix section bit-identical to the sealed file (then-current `LVQ3` format), which ties 16.9358 and 55.70 to the same 252 projections as 16.9415 and 55.59 |
| `q4b-e4.llvq`, int4 g64 embedding | 1,211,403,653 B, 2.4093 b/param | *measured*; +1.52% ppl, not published; matrix section bit-identical to the sealed file ([batch B verdicts](archive/verdicts-lot-b-2026-08-06.md) §B4) |

### 5.4 Production cost
| quantity | value | trace | label |
|---|---|---|---|
| run duration | 14,447 s = 4.013 h | `~/llvq-run-4b-artefact.log`, "quantized 252 matrices … in 14447s" | *measured* |
| per layer | min 328 s, max 592 s, mean 401 s | 36 timestamped lines | *measured* |
| threads | 16 | line 1 of the log | *measured* |
| cost | $0, local M3 Max | | *measured* |
| per-phase profile | none | the instrumentation dates from 2026-08-02 | not measured |

Two withdrawn durations refer to two other runs. "~3.5 h" is the night run: 12,715 s, `~/llvq-run-nuit.log`,
`leech1c12` on a binary predating `60068db`, 2.6923 b/weight actual. "3.45 h" is the cap 13 run: 2.1117 b/weight
announced, 2.7338 actual. Only 14,447 s = 4.013 h is the published run; it costs +13.6% over the night run (*computed*)
because it encodes the indices inside the loop and writes as a stream.

### 5.5 Round-trip proof
`3633315840 weights identical, bit for bit` (*measured*, `~/llvq-run-4b-artefact.log`): `to_bits()` comparison over the
252 matrices, f32, tail included, after `decode_matrix`; carried to the published bytes by the identity of the sections
(§3). It does not cover a native f16 run; narrowing is deterministic, so `f16(decode(file))` is the MMLU arm.

### 5.6 Served throughput on these bytes
| configuration | tok/s [range] | GB on card | card | source | label |
|---|---|---|---|---|---|
| served v1: `planes14` + q8 + `ROT_SHARE=1` + `FUSE=1` | 100.6 [99.9–100.7] | 2.57 | L40S | [D1](mesures/d1-fusion-servie-2026-08-24.txt), [wave 2](mesures/vague2-fusion-8b-14b-2026-08-31.txt) | *measured*, median of 5 rounds |
| `planes14` + q8, `ROT_SHARE=1/FUSE=0` (hoisting only) | 94.9 [94.1–95.2] | | L40S | D1 | *measured* |
| `planes14` + q8, `ROT_SHARE=0/FUSE=0` | 87.0 [86.8–87.0] | 2.56 | L40S | [B2](mesures/b2-fusedrun-plages-2026-08-18.txt) | *measured* |
| `planes12x` + q8 | 85.0 [84.7–85.1] | 2.36 | L40S | [G3](mesures/g-horloges-planes12x-2026-08-23.txt) | *measured*; −2.3% throughput for −0.20 GB |
| `planes14`, f16 embedding (same-head) | 48.3 [48.1–48.3] | 2.93 | L40S | B2 | *measured* |
| dense f16, our path | 43.5 [43.4–43.5] | 8.04 | L40S | B2 | *measured* |
| `bin/run` with KV cache, decoded in memory | 42.7 | | L40S | [mini](mesures/mini-2026-08-05.txt) | *measured* |

The same-head ratio, ×1.11 [1.11–1.11], measures the kernel. The raw ×2.00 [1.99–2.00] is never published alone: the
dense arm re-copies 778 MB of vocabulary per token (*measured*, [phases](mesures/phases-2026-08-07.txt)). Fusion:
×1.061 [1.050–1.069] within one job (*measured*, D1); divergence from the dense arm at token 89 of 128 under each fused
arm, and 128 identical tokens between the two fused arms (F1 and F0).

### 5.7 Error bars
| bar | value | what it covers | source | label |
|---|---|---|---|---|
| calibration σ, ppl | 5.2% (0.8202 ppl), range 10.3% over 16.7425 / 15.8836 / 15.1027 | three full 4B runs, seeds 1/2/3, $21.45 | [F5](mesures/f5-graines-4b-2026-08-19.txt) | *measured*; the three paired comparisons resolved (t +4.54 / +10.92 / +7.68) |
| calibration σ, MMLU | 2.92 pp, range 5.83 pp over 58.02 / 52.19 / 55.17 | same three artifacts | [MMLU noise](mesures/bruit-mmlu-graines-4b-2026-08-25.txt) | *measured* |
| LLVQ excess over f16, ppl | +38.45% [+33.62; +43.45] | corpus sampling, paired window by window, 12/12 | [paired ppl](mesures/ppl-appariee-4b-2026-08-17.txt) | *computed* on measured NLLs |
| A/B at constant file | ±0.12% ppl, SE 0.43 pp MMLU | paired interval; does not carry the calibration σ | [KV q8](mesures/kvq8-4b-2026-08-15.txt) | *measured* |
| n = 2 observation | 14.2684 against 15.2909, gap 7.2% | same quantizer (gap 7.1e-15, test `under_the_old_retraction_shape_gain_was_direction_only`), cause undecided | table rows, no log | *measured* |
| σ of 0.7% (0.15 ppl) | 3 blocks of Qwen3-0.6B, batch B | not the published size: a factor 7 below F5 | [batch B verdicts](archive/verdicts-lot-b-2026-08-06.md) | *measured* |

The published file is one draw from a seeded process, calibrated on the earlier shard 00000; it is not a fourth draw.
The 0.08 point below QTIP (16.9617 against 17.04) sits under the measured spread and is not claimed.

### 5.8 Cost of the gain bit
Coding the gain costs +3.17% perplexity for −0.618 b/weight (*measured*, `~/llvq-ab-retraction.log`, 2026-07-31). A/B:
Qwen3-0.6B, 3 blocks, ctx 2048, 12 windows, baseline 19.5038.

| arm | codebook | b/weight | ppl | × |
|---|---|---|---|---|
| A | `leech1c12`, carried gain (47 + 1 = 48 b/block) | 2.1656 | 21.4157 | 1.098 |
| B | `leech1c12f`, free f16 magnitude (47 + 16 = 63 b/block) | 2.7838 | 20.7582 | 1.064 |

Caveats: 3 blocks of a 0.6B, and the `f` suffix restores only the free magnitude. Hence a 4B gap (10.7%, *computed* on
16.9617 against 15.3272, §5.1) larger than the 0.6B gap (3.2%, *computed* on the table above). The 28-block gate on the
0.6B returns, at 2.1656 b/weight, `leech0c13` 39.3309, `leech2c11` 39.5350, `leech1c12` 43.4865, `leech4c10`
47.1537 (seed 0) (*measured*, [gain gate](mesures/gain-ab-gate-0.6b-2026-08-25.txt)). Seed 1 reverses the ranking.

### 5.9 Gaps and undecided points
Of the seventeen gaps recorded on 2026-08-03, ten have been measured since (dates in [HISTORIQUE.md](HISTORIQUE.md)),
five remain (table below), one was documentary (`Cargo.toml`, README, ppl command: 0 machine time) and one has never
been done (MLX q4 throughput and RSS replayed and logged, ~2 min, secondary since AWQ).
Measured: stdout of the two f16 ppl runs; `thesis` journal ([control](mesures/thesis-temoin-2026-08-04.txt), K1); ppl
and MMLU of the 4-bit (AWQ, §7); σ at the published depth (F5). Also measured: the L ≤ 4 cap (+4.75% ppl, *measured*,
[batch B verdicts](archive/verdicts-lot-b-2026-08-06.md)); damping (20.6740 / 20.6643 / 20.6014, batch B); `Grouped32`
and `Flat32` on the whole model (K1); GPU rotation and the kernel wired up on CUDA. Remaining:

| point | state |
|---|---|
| ppl gap between calibration shard 0 and shard 1 | not measured; 2 runs of 3 blocks, ~50 min (*estimated*) |
| `bin/seal` replayed on its own output | expected 2.1595, 1.771 GB, file not identical (`LVQ2` + two `u32`), ~10 min |
| `k = 1` in `llvq-bench` (`main.rs:109` loops over `[0, 2]`) | the "union + 1 gain bit" row of the table sent to the authors is copied from Table 8 |
| retention in the table sent to the authors | `retention_pct(mse, rate) = 100·(−½·log₂ mse)/rate`: on the rounded MSE 0.078 the benchmark prints 92.01, the paper 92.14 on its unrounded MSE (≈ 0.077718, SQNR 1.843); quote 92.14, never recompute it from 0.078 |
| determinism control, two identical runs | decides whether the 7.2% of the n = 2 observation is numerical noise or an unrecorded configuration; ~8.5 h (*estimated*) |
| CSR | task definition not transcribed; 1 to 2 days (*estimated*) |
| mechanism of the 17.41 GB Metal RSS peak | *estimated*: dual host/buffer residency, or a candle-metal buffer pool |
| incremental gain of the output rotation | Table 9 of the paper quantifies "none → Input+Output" (29.3 → 34.9), not "Input → Input+Output"; nothing in the code, a MAGIC bump required |
| cost of `gain_bits = 0` on the 4B | not measured; the A/B of §5.8 compares 1 bit against a free magnitude |
| CUDA against Metal difference in the weights produced | AᵀA accumulated in f32 on the accelerator; not quantified |

## 6. Against FP16
| pair | iso-conditions | how |
|---|---|---|
| f32 ppl, 12.2336 / 16.9617 | guaranteed by construction | `smoke` tokenizes once, a single `test_ids`, a single `ppl` closure, a single model object before and after rewriting; no fingerprint to compare |
| f16 ppl, 12.2361 / 16.9415 | identical fingerprint `3f1baca9033bf251` | expected by construction: the sealed tokenizer is byte-identical to the checkpoint |
| MMLU, 70.32 / 55.59 | same binary, same session, same printed dtype, same 2,280 questions, same tokenizer, fingerprint `65dcd53655e8bfa5` | the 08-02 Metal log predates the printing of fingerprints |

Uncontrolled protocol difference in f32 (*measured*): the checkpoint is bf16, `seal` writes the carried tensors in f16.
Of the 388,956,160 embedding values, 77,045 change (1.98·10⁻⁴), 451 fall to zero, all below 7.600·10⁻⁶; max |v| 0.250,
max absolute error 2.98·10⁻⁸. The embedding is the `lm_head`, so the difference enters the logits. In f16 the two
arms converge, MMLU and f16 ppl are clean. In f32, a ppl of the sealed file is not compared to the checkpoint baseline
without saying so. The 12.2336 / 16.9617 pair runs in memory and is unaffected. Residual `from_mmaped_safetensors`
against `from_vec(f32).to_dtype()`: *estimated* negligible. The "FP16" of the benchmark is not this FP16 (§8.2).

## 7. Against 4-bit
The chosen opponent is Qwen's official AWQ (decision of 2026-08-06), measured in the same harness at the same
fingerprint ([a4-campagne](mesures/a4-campagne-2026-08-06.txt)). MLX q4 remains the object of the local disk
comparison; IQ2_XXS: [ETAT.md](ETAT.md) §3.

### 7.1 On disk
| object | value | label |
|---|---|---|
| MLX q4, `/Users/pjmalandrino/qwen3-4b-mlx-q4/` | `model.safetensors` 2,263,022,417 B, directory 2,274,510,217 B | *measured* |
| recipe | `mlx_lm.convert --hf-path Qwen/Qwen3-4B -q --q-bits 4 --q-group-size 64` | *measured*, `config.json` |
| structure | 904 tensors: 253 U32, 253 `.scales`, 253 bf16 `.biases`, 145 norms | *measured* |
| embedding | quantized too (253 = 252 projections + `embed_tokens`) | *measured*; we carry it in f16 or in q8 |
| rate | 4.500000 b/weight on the quantized weights, 4.500561 on all weights | *computed*, exact |
| total | 4,022,468,096 weights on both sides | *computed* |
| official AWQ w4 g128 | 2.67 GB, 5.302 b/param in its own engine | *measured* / *computed* ([rtbits](mesures/rtbits-planes-8b-2026-08-09.txt)) |

### 7.2 Axis by axis
| axis | LLVQ | 4-bit | verdict | label |
|---|---|---|---|---|
| disk | 1,770,527,533 B, 3.5213 b/param (1.41 GB in q8) | MLX q4 2,263,022,417 B, 4.5006; AWQ 2.67 GB | ×1.2782 for us; projections alone 2.1595 against 4.5000, ×2.084 | *measured* on both sides |
| VRAM, b/param whole model | 5.162 (`Planes14` + q8); 4.745 (`Planes12x` + q8) | AWQ 5.302 in its own engine; MLX q4 4.50 | 2.6% below the real AWQ | *computed* on measured bytes (rtbits) |
| throughput | ×1.11 [1.11–1.11] same-head on our side | ×2.413 [2.412; 2.414] for AWQ in vLLM (200.49 tok/s against 83.09 f16) | two stacks, two controls, no legitimate quotient | *measured*, [vLLM](mesures/awq-vllm-4b-2026-08-17.txt) |
| wikitext ppl | ×1.385 | AWQ ×1.105 (13.5207 / 12.2369) | excess 0.385 against 0.105, ratio 3.7 (*computed*) | *measured*, a4-campagne |
| MMLU micro | 55.59 (−14.73 pp) | AWQ 70.04 ± 1.25 (−0.28 pp, unresolved [−1.63; +2.13]) | paired gap 14.45 pp [+11.60; +17.27] | *measured*, [mmlupair](mesures/mmlupair-4b-8b-2026-08-13.txt) |

On a 4B, 4-bit dominates everywhere except disk; the MMLU gap to AWQ is 7.49 pp at 8B and 6.09 pp at 14B
([ETAT.md](ETAT.md) §3). Our harness loads AWQ dequantized in f16: neither its throughput nor its memory can be read
on our side.

### 7.3 The three RAM figures
| quantity | value | label |
|---|---|---|
| MLX q4, "2.39 GB" | peak of the MLX allocator (weights + KV + activations), prompt unknown | *estimated*, no trace |
| us, "3.28 GB" | weights-only arithmetic of the `Slot32` format, which `bin/run` never loads | *computed*, irrelevant for the runner |
| us, resident model of `bin/run` | 4,022,468,096 × 2 = 8,044,936,192 B | *computed*, exact by construction |
| us, RSS peak of `bin/run` | CPU 9.79 GB (`cpu 12`); Metal 17.41 GB, reproducible to 0.0006% over 4 launches, mechanism unknown | *measured*, `/usr/bin/time -l` |
| us, L40S card under `fusedrun` | 2.57 GB (v1), 2.56 (`Planes14` + q8), 2.36 (`Planes12x` + q8), host byte count | *measured*; D1 and wave 2 (2.57), B2 (2.56), G3 (2.36) |

Under the weights-only convention, `Slot32` + f16 `lm_head` is 6.5245 b/weight against 4.5006 for q4, ×1.45 against us
(*computed*).

### 7.4 Throughput
| number | includes | excludes | label |
|---|---|---|---|
| MLX 129.8 tok/s | 253 matmuls, attention, norms, RoPE, KV, lm_head, sampling | prefill, loading; `--max-tokens 256`, prompt unknown | *estimated*, no trace |
| AWQ 200.49 tok/s [200.39; 200.61], vLLM 0.26.0, L40S, batch 1, 128 tokens | everything, prefill included | another stack: no legitimate quotient with anything of ours | *measured* |
| `thesis` 10.46 ms | 252 fused matvecs, one token, cold memory | attention, norms, RoPE, KV, lm_head, rotation, transcoding | *measured*, Metal, 2 arms of 2026-08-01; the ratio to publish is K1's (§8.4) |
| `thesis` 78.2 tok/s | the above + modelled lm_head | same; never executed | *computed*, upper bound |
| `bin/run` 2.2 to 7.6 tok/s, Metal f16 | everything, end to end | KV cache, fused kernel | before the KV cache (commit `9c24d26`); with the cache, 42.7 tok/s on L40S (§5.6) |
| `fusedrun` v1 100.6 tok/s [99.9–100.7] | everything, end to end; divergence from the dense arm at token 89 of 128 | | *measured*, L40S, D1 |

### 7.5 Regime where 2-bit wins
One axis alone is demonstrated: disk, ×1.278. The structural niche is the memory window where 4-bit does not fit. It is
worth 12 to 21% (*computed*): 4.50 / 3.727 = ×1.21 with `Grouped32`, 4.50 / 4.034 = ×1.12 with L ≤ 3. Recomputation for
70B, Llama-3.1-70B, 70.554 B params, embedding and `lm_head` untied (2.978%) left in f16 (*computed*):

| | q4 | `Slot32` | L ≤ 3 | `Grouped32` | disk |
|---|---|---|---|---|---|
| recomputed | 39.69 GB | 51.35 | 35.58 | 32.87 | 22.77 |
| [face-au-4-bits.md](archive/face-au-4-bits.md) | 39.4 | 48.2 | 32.1 | 29.3 | 19.0 |

The archive numbers applied a projections-only rate to all weights, optimistic by 6 to 12% for us. Four unknowns remain.
Quality at 70B: no 70B has been quantized. The KV cache: 320 KiB/token in f16, 2.68 GB at 8k (*computed*). The served
`Grouped32` throughput. The fast format, `Planes14` at 4.804 b/weight in the kernel, bigger than 4-bit. The product
triplet (8k, 5 GB, 32 GiB) bounds b_max to 3.00 b/weight in the kernel ([ETAT.md](ETAT.md) §6). The L ≤ 4 cap is dead
on quality (+4.75% ppl, §5.9).

## 8. The fused kernel
### 8.1 Protocol of `bin/thesis`
Metal, one token, batch 1, 252 projections, cold memory by construction. Two pipelines (`tv_f16`, `tv_slot`); a table of
384 classes in a shared constant buffer (12 kB); one activation `SplitMix64(0x6_7451)`, 16,384 Gaussian f32, a single
buffer for both arms. Per matrix: `read_matrix_raw`, `transcode(Slot32)`, f64 reconstruction, f16 rounding, references
`y_ref` / `y16_ref` in f64, upload of 6 LLVQ buffers and 1 FP16 buffer; verification (§8.3) before any measurement.
Measurement: one command buffer per arm, 252 encoders, `d_out × 32` threads in groups of 256, identical tiling
(128 blocks, 3,072 columns, 12 kB of threadgroup memory). Clock around `commit()` and `wait_until_completed()`;
7 passes, reps 0 and 1 discarded, minimum of the remaining 5 for the two-arm runs: their ratio is a quotient of two
minima, which disqualifies it against K1. The seven-arm benchmark (K1) dispatches every arm each round in the same order and
forms the ratio round by round, median and range. Five asymmetries, all against LLVQ or negligible. FP16 measured first;
submission not subtracted; tail read in f32 against f16; 9 binds against 4; 12 kB of table not counted. Any common
additive term compresses the ratio: the 2.07× is a lower bound on the pure ALU/memory ratio.

### 8.2 The FP16 arm
`w16 = f16_bits(w)`, where `w` is the f64 reconstruction of the LLVQ blocks in the rotated basis. The FP16 arm reads the
same values up to rounding: it measures a cost, and says nothing about quality. On CUDA, `r = t(tv_f16) ÷ t(cuBLAS)` is
1.024 (2 arms) and 1.015 (5 arms) on L40S (*measured*, [F1](mesures/f1-cublasf16-2026-08-18.txt)). On the A100 the same
control sits at 1.14× cuBLAS (*measured*, [F4](mesures/f4-a100-2026-08-18.txt)). On Metal it has never been put against
MPS or MLX.

### 8.3 Numerical verification
| quantity | value | label |
|---|---|---|
| rows verified | 1,105,920, 252 matrices, both arms | *measured* |
| metric | max over the rows of \|got − want\| / max(Σ\|wᵢxᵢ\|, 1e-12), f64 reference | |
| hard threshold | `assert!(e < 1e-3)`, before any time measurement | |
| worst LLVQ error | 3.4·10⁻⁸ · Σ\|w·x\|, identical across runs and files | *measured* |
| worst FP16 error | 2.8·10⁻⁸ | *measured* |

Caveats: row granularity (the block lives in `bin/decreal`); the threshold is five orders above the error, the proof is
the printed worst error. `thesis` does not re-check the transcoding against `Indexer::decode` (that lock lives in
`llvq-artifact/tests/runtime_format.rs`). `slot_dot` hard-codes `gain = hdr >> 9` (1 bit); `decreal` asserts it,
`thesis` does not.

### 8.4 Numbers and spread
| benchmark | FP16 | LLVQ `Slot32` | ratio | source | label |
|---|---|---|---|---|---|
| 7 arms, 2026-08-05 | 21.728 ms, 7.27 GB | 10.496 ms, 2.50 GB, 5.510 b/weight | 2.03× [2.03–2.10], median round by round | [K1](mesures/k1-metal-2026-08-05.txt) | *measured*, the number to publish |
| 2 arms, three control invocations | | | [2.029; 2.080]; 7 arms: 2.03× · 2.06× · 2.09× | [control](mesures/thesis-temoin-2026-08-04.txt) | *measured*; a single point value has no content |
| 2 arms, 2026-08-01 | 21.691 ms, 335.0 GB/s | 10.460 ms, 239.2 GB/s | 2.0737×; 2nd pass 2.08× | README of the time, no log | *measured* on the order of magnitude, suspect on the decimals; superseded by K1, the milliseconds are not published |
| 2 arms, 2026-08-03, sealed file | 22.675 ms | 11.021 ms | 2.0574× | no log | *measured* on the order of magnitude, suspect on the decimals; superseded by K1; gaps to the 08-01 row +4.5% / +5.4% / −0.8% |

The milliseconds drift from one invocation to the next; bytes, b/weight and worst errors reproduce to the digit.
Reproduction: `cargo run --release -p llvq-metal --bin thesis -- <sealed>`; the default of the three benchmarks
(`thesis.rs:191`, `matvec.rs:503`, `decreal.rs:139`) is `~/llvq-q4b.llvq`, not published.

### 8.5 What the ratio excludes
The ratio excludes the whole of attention (QKᵀ, softmax, AV, RoPE, KV cache); the 145 RMSNorms, including per-head
`q_norm` / `k_norm`; the SwiGLU; the residuals. It also excludes the incoherence rotation on x, 144 per token, paid by
the LLVQ arm alone, and the tied `lm_head`, added analytically. Also outside the ratio: sampling over 151,936 logits,
the prefill and the transcoding. The rotation costs 0.206% of the projections in arithmetic (1.499·10⁷ ops against
7.267·10⁹ flops, *computed*, `rotation.rs`). On CUDA, `rot_apply` returns 9.5e-8 against f64 on 8 shapes and 8.05 µs at
n = 2560 (*measured*, [CUDA rotation](mesures/rotation-cuda-2026-08-05.txt)), and the end-to-end of §5.6 pays for it.
On Metal, no rotation kernel exists and `bin/run` decodes in memory.

### 8.6 The benchmark tok/s
Projections alone: 2.07×. With the modelled f16 `lm_head`: 1.88× at most. 78.2 tok/s rests on no measurement.
`thesis.rs:433-435`: `head_bytes = 389_070_848 × 2`, `bw = f16_bytes / t16` (335.0 GB/s), `head_s = 2.3228 ms` on both
arms (*computed*).

| arm | total | tok/s | label |
|---|---|---|---|
| FP16 | 24.014 ms | 41.64 | *computed*, upper bound |
| LLVQ | 12.783 ms | 78.23 | *computed*, upper bound |
| ratio | 1.879 | | upper bound on the Metal end-to-end |

Adding a common constant compresses the ratio (2.07 → 1.88): the treatment is conservative for LLVQ. The real defects:
the `lm_head` is never executed, the rest of the decode step is excluded, the constant matches nothing.

### 8.7 The bits-against-speed scale
One accounting (payload + addressing + f32 tail + f32 scales, over all weights), one process, seven interleaved arms,
7 rounds of which 2 discarded (*measured*, K1, Metal):

| layout | b/weight, narrow metric | b/weight, kernel metric | vs FP16 | object | label |
|---|---|---|---|---|---|
| `Grouped32` | 3.3548 (`rtbits`, 150,681,600 blocks, 6.5 s) | 3.498 → 1.589 GB | 0.69× [0.68–0.69] | whole model | *measured* |
| `Flat32` | 4.54 (gate_proj) | 5.256 → 2.39 GB | 0.91× [0.91–0.91] | whole model | *measured* |
| `Sorted32` | 4.75 | | 1.04× | gate_proj alone, `bin/matvec` | *measured* |
| `Fixed96` | 4.000, structural | | | never in matvec | *computed* |
| `Slot32` | 5.376 model / 5.375 gate_proj (`rtbits` 5.3756) | 5.510 → 2.50 GB | 2.03× [2.03–2.10] | whole model | *measured* |
| FP16 | 16.000 | | 1.00× | | |

The 5.51 against 5.375 gap is a gap of metric, not of object: `rtbits` (whole model) and `matvec` (gate_proj) agree to
0.02%. The curve is non-linear: `Flat32` saves 0.254 b/weight over `Slot32` for 2.27× the time, `Grouped32` 2.012 for
3.01× (*computed*). `float4` returns 3.5% on LLVQ and 5.1% on FP16, ratio unchanged (*measured*, K1). The f32 tail
is 2.71% of LLVQ traffic (67,829,760 bytes of 2,502,446,285); in f16, `Slot32` drops to 5.435 b/weight (*computed*).
The 335 GB/s of the FP16 arm are 83.8% of an assumed 400 GB/s peak; the "93%" is that of gate_proj (370 GB/s). On CUDA,
the served layout is `Planes14`: 4.804 b/weight in the kernel, 2.14× [2.11–2.15] on L40S (*measured*,
[C1](mesures/c1-planesbench-2026-08-06.txt), [E2](mesures/e2-golay70-bench-2026-08-07.txt)). It returns 1.14×
[1.14–1.15] over `Slot32` at identical content, and 0.79× on the A100 (*measured*, F4).

### 8.8 Transcoding at load
| number | what it is | label |
|---|---|---|
| "~3 s for a 4B" | 150,681,600 × 243 ns / 12 cores, with a parallelization that does not exist (`transcode()` is single-threaded) | ~37 s single-core (*computed*) |
| 128 s, `load_s` of `thesis` | covers unpacking 981 MB, the f64 reconstruction, 3.63 billion f16 conversions, the f64 CPU reference and 7 uploads per matrix | *measured*, mislabelled |
| transcoding alone | `bin/decreal` on 16,777,216 real blocks, `Fixed96` and `Grouped32`; factor ×8.98 then ÷2 towards the whole model; `Slot32` absent | *measured* |
| `Planes14` / `Planes12x`, M3 Max, 16 threads | 84 s / 404 s (×4.8, 5-level lattice search per block) | *measured*, 2026-08-09 |
| `Planes12x` on a rented card | 1,340 s | *measured*, [G3](mesures/g-horloges-planes12x-2026-08-23.txt) |
