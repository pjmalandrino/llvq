# The served configurations

One file per shipped object. It carries the choices that decide how a `.llvq`
is read, and it is the authority on them: `LLVQ_CONFIG=<file>` puts a runner on
the served path, and an environment variable that contradicts the file is
refused rather than silently preferred.

There is no built-in served default. A runner with no `LLVQ_CONFIG` is a runner
in measurement mode, reading the environment variables it always read, which
is what every A/B in `docs/mesures/` depends on.

This README is the authority on every file here. Each file's sha256 is pinned
by the journals that ran it, so its bytes do not move.
The `note` field inside an older file is frozen with it and may be stale: it
can call a file served that is served no more, or quote a speed a later run
replaced. Where a note and this table disagree, the table wins.

## At a glance

| state | files |
|---|---|
| served on CUDA since 2026-11-07 | `qwen3-4b-tetra-e4-4e-v9.json`, `qwen3-8b-tetra-e4-4e-v9.json`, `qwen3-14b-tetra-e4-4e-v9.json` |
| served on the Mac since 2026-11-07 | `mlx/qwen3-4b-tetra-e4-mlx.json` (MLX door) |
| the paper 2 objects, expanded form, every published score | `qwen3-{4b,8b,14b}-tetra-e4.json` |
| served before, kept and still loading | `qwen3-4b-tetra-e4-grouped.json`, `-kernel-from1280.json`, `-4e.json`, `-4e-v4.json`, `qwen3-{8b,14b}-tetra-e4-grouped.json` |
| measured, never served | `qwen3-4b-tetra-e4-4e-v6.json`, the two `*-metal-*-NOT-SERVED.json` |
| written, never served, card gate not run | `qwen3-{4b,8b,14b}-tetra-e4-4e-v9-kprefill.json` |
| earlier objects | `qwen3-{4b,8b,14b}-tetra-q5.json` |

The runtime `llvq` compiles in the v9 configs for CUDA and the `*-e4.json` configs for Metal (`llvq-runtime/library.json`).

## Every file

| file | object | b/param | MMLU |
|---|---|---|---|
| `qwen3-4b-tetra-q5.json` | Qwen3-4B, `Tetra` + 36 `v_proj` int4 g128, `qwen3-4b-tetra-q5.bin` (1,794,564,765 B, sha256 `ae31087a4b72494d52394f2cf2070da37a845c0a7d4aded65d5bdb0bd18b2263`, *measured*, [references-comptabilite-2026-09-18](../docs/mesures/references-comptabilite-2026-09-18.txt)). The only copy is in the bucket at `tetra-q5-2026-09-09/` | **2.7475** as served (*measured*, whole model, q8 embedding, tail f16, same journal). The f32 tail accounting every published figure used gives 2.8126, and the card has held the tail in f16 since 2026-08-09, so 2.7475 is the width that runs | **56.37 micro, 58.42 macro** on the full 14,042-question split, dense reconstruction (*measured*, [f1e-census-2026-09-11](../docs/mesures/f1e-census-2026-09-11.txt)). On the 2,280-question plan the same file reads 55.52 dense and **55.66 through the served kernel**, 3 discordant questions, all three f16 tie-breaks. The 56.95 the record also carries is a different file: the pure `Tetra` `qwen3-4b-tetra.bin` with `v_proj` restored to int4 at load by `LLVQ_RESTORE_Q4` (`q5-tetra-2026-09-06.txt` T2), so it is not this object's score |
| `qwen3-4b-tetra-e4.json` | Qwen3-4B paper-2 object: `Tetra` with dclm-edu calibration and trained row scales, `v_proj` + `o_proj` + `down_proj@12-23` as int4 g128 records, the tied embedding stored int4 g64, `qwen3-4b-sealed.bin` (1,418,224,685 B, sha256 `886391a8c03f66dc269cc65c3598c6627dbdcd259180aff36604ef10d37371b8`). Built on 2026-09-23 from the 61.11 object by `int4swap` then `embedq q4` ([embed-q4-swap-2026-09-23](../docs/mesures/embed-q4-swap-2026-09-23.txt)) | **2.7320** as served (*computed* by `rtbits` over the file, tail f16, embedding q4); 2.5420 kernel b/weight | **63.37 micro** for the same weights on the dense path (*measured*, the restore arm of the same journal); the file itself is not scored yet |
| `qwen3-4b-tetra-e4-grouped.json` | **The 4B's served configuration from 2026-11-02 to 2026-11-05** (operator decision), replaced by the row below. The SAME object as the row above, `qwen3-4b-sealed.bin`, served with the grouped attention: `"attention": "grouped"` and nothing else differs. Added 2026-11-02 | **2.7320** as served, the row above's number: the form changes no stored bit | Not scored. **Perplexity** is what licensed it: WikiText-2 paired against the expanded form through the kernel, +0.0000 % by prefill (bit for bit) and +0.0095 % [-0.0048 ; +0.0238] by decode, inside the +-0.12 % band of `docs/ETAT.md` section 4 (*measured*, [gqa-ppl-2026-11-01](../docs/mesures/gqa-ppl-2026-11-01.txt)). Through the served door it decodes at **133.3 tok/s** after a 5-token prompt against 121.5, and by the context sweep **134.5 / 113.8 / 98.0 / 77.5** against 122.5 / 82.7 / 61.4 / 33.2 at 5 / 2,048 / 4,096 / 8,192, with the same 256-token text (*measured*, [served-grouped-2026-11-02](../docs/mesures/served-grouped-2026-11-02.txt)). **GSM8K** through the kernel, all 1,319 problems: **81.96 %** against 82.49 % for the expanded census, paired -0.53 pp [-1.07 ; +0.00], McNemar p = 0.092; on the 57-question MMLU door both configs write byte-identical rows (*measured*, [gsm8k-grouped-2026-11-04](../docs/mesures/gsm8k-grouped-2026-11-04.txt)). No full MMLU under this form |
| `qwen3-4b-tetra-e4-kernel-from1280.json` | **The 4B's served configuration on 2026-11-05, until the v4 config below** (operator decision, T8 of the kernel form with its threshold). The SAME object, `qwen3-4b-sealed.bin`, with `"attention": "kernel"` and nothing else different: decode rows at position 1,280 and later go to the split-KV kernel, rows below run the grouped arithmetic bit for bit. The threshold is `llvq_llm::kvq::KERNEL_FROM` in the code, not a field. Refused on Metal. Added 2026-11-05 | **2.7320** as served: the form changes no stored bit | Not scored. **Perplexity** licensed it: WikiText-2 paired against the grouped form on the positions the kernel takes, -0.0013 % [-0.0074 ; +0.0047] in 12 windows of 4,096 and +0.0001 % [-0.0076 ; +0.0077] in 6 windows of 8,192, inside +-0.12 %; every step below 1,280 bit for bit. Served sweep **134.7 / 120.5 / 117.6 / 104.0** at 5 / 2,048 / 4,096 / 8,192 against 135.0 / 114.1 / 97.9 / 77.5 for the grouped form, same 256 tokens at every length; the 50-problem GSM8K replay wrote the grouped census's rows (*measured*, [attn-t8-r2-2026-11-04](../docs/mesures/attn-t8-r2-2026-11-04.txt)). No task score past 1,280 of context |
| `qwen3-4b-tetra-e4-4e.json` | **Not served**: the v3 row loop, read at the served door; the served 4E config is the next row. The SAME object, `qwen3-4b-sealed.bin`, with `"attention": "kernel"` and `"decode": "persistent"`: every decode row goes to the split-KV kernel from position 0, and 4E decodes it in one cooperative launch a token run (512 threads). The kernel form below 1,280 is licensed by the operator's T8 extension of 2026-11-05 on this object only. Refused on Metal, on any shape but the 4B's, and beside any measurement-mode variable. Added 2026-11-05 | **2.7320** as served: the engine changes no stored bit | Not scored. **Perplexity** licensed the form below 1,280: paired WikiText-2 decode path against the grouped form, +0.0318 % [-0.0151 ; +0.0786] in 32 windows of 1,280 and +0.0018 % [-0.0053 ; +0.0088] in 12 windows of 4,096, inside +-0.12 % (*measured*, [attn-t8-below-4b-2026-11-05](../docs/mesures/attn-t8-below-4b-2026-11-05.txt)). 4E decodes the eager path's tokens under this form, **179.2 tok/s** against 133.0 after a 5-token prompt and 135.4 against 104.6 at 8,192, in the measurement form of `LLVQ_PERS_4E` (*measured*, [pers-4e-v3-2026-11-05](../docs/mesures/pers-4e-v3-2026-11-05.txt)). At the served door: **169.0 tok/s** against 133.7 for the kernel config, sweep **171.2 / 162.0 / 154.5 / 134.1** at 5 / 2,048 / 4,096 / 8,192 against 134.8 / 120.6 / 117.3 / 103.8, same 256 tokens as the kernel config at every length; GSM8K pilot 40/50 against 40/50 for the grouped census, McNemar p = 1 (*measured*, [served-4e-2026-11-05](../docs/mesures/served-4e-2026-11-05.txt)) |
| `qwen3-4b-tetra-e4-4e-v4.json` | **The 4B's served configuration from 2026-11-05 to 2026-11-07**, replaced by the v9 config (operator decision, "Oui, sers 4E v4"). The 4E config above with `"rows": "v4"` and nothing else different: 4E runs the v2 row loop inside the v3 structure (row form 2). Same arithmetic, so the same tokens as the v3 config and as the eager path. Refused where the 4E config is. Added 2026-11-05 | **2.7320** as served: the row loop changes no stored bit | Not scored; the arithmetic and the tokens are the v3 config's, scored in the row above. v4 decodes **189.8 tok/s** after a 5-token prompt against 179.0 for v3 and 134.2 for the eager path (x1.060 over v3), and **140.1** at 8,192 against 135.4, same 64 tokens as the eager path at both contexts, in the measurement form of `LLVQ_PERS_4E` (*measured*, [shaker-4e-2026-11-05](../docs/mesures/shaker-4e-2026-11-05.txt)). At the served door: **178.0 tok/s**, decode 180.7, x1.404 over the eager path; sweep **180.8 / 169.5 / 162.1 / 139.5** at 5 / 2,048 / 4,096 / 8,192; the v3 config's tokens at the door and every length, and its 50 pilot completions (*measured*, [served-v4-2026-11-05](../docs/mesures/served-v4-2026-11-05.txt)) |
| `qwen3-4b-tetra-e4-4e-v6.json` | **Never served**: written 2026-11-07, its door read the same day with every gate held; the v9 config was served in its place. The v4 config above with `"rows": "v6"` and nothing else different: 4E runs v4's rows with ln1, ln2 and the down's input in one round trip a thread (row form 4). The install refuses it when an input of a vector prologue is not 16-byte aligned, so it never runs v4's scalar path under v6's name. Refused where the 4E config is | **2.7320** as served: the row loop changes no stored bit | Not scored; the arithmetic is v4's. v6 decodes **212.6 tok/s** after a 5-token prompt against 190.2 for v4 (x1.118 [1.117-1.118] round by round) and **152.0** at 8,192 against 138.9 (x1.094), same 64 tokens as the eager path at both contexts, in the measurement form of `LLVQ_PERS_4E` (*measured*, [pers-4e-v6-2026-11-05](../docs/mesures/pers-4e-v6-2026-11-05.txt)). At the served door: **197.2 tok/s** against 178.0 for v4, decode 200.6, x1.568 over the eager path; sweep **201.8 / 189.0 / 178.8 / 150.9** at 5 / 2,048 / 4,096 / 8,192; the v4 config's tokens at the door and every length, and its 50 pilot completions (*measured*, [served-v6-2026-11-07](../docs/mesures/served-v6-2026-11-07.txt)) |
| `qwen3-4b-tetra-e4-4e-v9.json` | **The 4B's served configuration since 2026-11-07** (operator decision, "Tu me fait 4,8 et 14 en V9"). The v6 config above with `"rows": "v9"` and nothing else different: 4E runs v8 (v6, the launch's constants staged during the barriers, attention's steps in a cp.async ring) with the int4 rows v, o and the int4 downs read from tiled words (row form 7). The rows field also lays the int4 words tiled on the card, written by the loader and held once: the eager path, which prefills, reads them through `tv_q4_hht`, checked at every load against `tv_q4_hh` on a file-order copy of one record of every width, then freed. Every other config keeps the file's order, and 4E v3 to v8 refuse tiled words by name. Refused where the 4E config is | **2.7320** as served: the tiling is a permutation inside each row, so neither the file nor a byte of the count moves (*computed*); the measurement form of step3-4e held the int4 words twice, 0.385 GB more on the card | Not scored; the arithmetic is v6's. v9 decodes **238.6 tok/s** after a 5-token prompt against 213.3 for v6 (x1.118 [1.118-1.119] round by round) and **163.5** at 8,192 against 154.2 (x1.060), same 64 tokens as the eager path at both contexts, in the measurement form of `LLVQ_PERS_4E` with the int4 bytes twice on the card (*measured*, [step3-4e-2026-11-07](../docs/mesures/step3-4e-2026-11-07.txt)). At the served door: **221.5 tok/s** against 197.4 for v6 in the same job, decode 225.8, x1.717 over the eager path; sweep **226.2 / 205.8 / 194.2 / 161.5** at 5 / 2,048 / 4,096 / 8,192; the v6 config's tokens at the door and every length, its 50 pilot completions, and 1.980 GB on the card under both (*measured*, [served-v9-2026-11-07](../docs/mesures/served-v9-2026-11-07.txt)) |
| `qwen3-8b-tetra-e4-4e-v9.json` | **The 8B's served configuration since 2026-11-07** (operator decision, same message). The 4B v9 config's keys on the sealed 8B (`qwen3-8b-sealed-B.bin`, sha256 `7bdb9a55…`): 4E v9 with the head on its own table and `o_proj` in Tetra under its own rotation, the int4 words tiled and held once. The attention form is the kernel form from position 0, not the grouped form the 8B serves; its door reads the paired perplexity band | the 8B object's **2.6953**: no stored bit moves (*computed*) | Not scored. 4E against the served chain byte for byte on the host with a Tetra o and an untied head (*measured*, `llvq-cuda/tests/pers_matches_served.rs`). At the served door: **142.5 tok/s** against 105.9 for the grouped config in the same job, x1.384 over the eager path, 256 of 256 tokens equal; sweep **144.5 / 135.2 / 129.3 / 110.0**; pilot 46/50 as the grouped census; the kernel form from position 0 against grouped, perplexity +0.0043 % [-0.0141 ; +0.0227] (*measured*, [served-v9-8b-2026-11-07](../docs/mesures/served-v9-8b-2026-11-07.txt)) |
| `qwen3-14b-tetra-e4-4e-v9.json` | **The 14B's served configuration since 2026-11-07** (operator decision, same message). The 4B v9 config's keys on the sealed 14B (`qwen3-14b-sealed.bin`, sha256 `61db37fe…`): 4E v9 with the head on its own table, `o_proj` int4, the int4 words tiled and held once; v9 asks 96,064 B of shared memory a CTA (*computed*). The attention form is the kernel form from position 0 (groups of 5), not the grouped form the 14B serves; its door reads the paired perplexity band | the 14B object's **2.7305**: no stored bit moves (*computed*) | Not scored. Same host proof as the 8B row. At the served door: **86.3 tok/s** against 62.8 for the grouped config in the same job, x1.375 over the eager path, 256 of 256 tokens equal; sweep **87.3 / 83.7 / 80.9 / 71.5**; pilot 46/50 as the grouped census; perplexity against grouped -0.0108 % [-0.0281 ; +0.0065] (*measured*, [served-v9-14b-2026-11-07](../docs/mesures/served-v9-14b-2026-11-07.txt)) |
| `qwen3-{4b,8b,14b}-tetra-e4-4e-v9-kprefill.json` | **Not served**: written 2026-11-08, card gate not run (`proofs/preregistration-pic-vram-8k-2026-11-08.md`). The v9 config of each size with `"prefill_attention": "kernel"` and nothing else different: a prefill chunk goes to the decode-attention kernel 8 rows a call instead of the grouped matmuls, so no heads x 4,096 x 8,192 score tensor is made. Every prompt row gets the bits a decode step at its position gets; it is not bit for bit with the grouped prefill of the served v9 configs | the v9 row of its size: the prefill form changes no stored bit | Not scored. Peak card memory after an 8,192-token prompt **3.47 / 4.91 / 7.37 GB** (*computed*) against 5.65 / 7.09 / 10.08 for the served configs since the exact fixes of 2026-11-08 (*computed*) and 11.48 / 12.66 / 17.09 measured before them; on the host its logits do not move with the chunking (*measured*, [pic-vram-8k-2026-11-08](../docs/mesures/pic-vram-8k-2026-11-08.txt)) |
| `qwen3-4b-tetra-e4-metal-grouped-NOT-SERVED.json` | **Not served**: a Metal measurement config for the operator, written 2026-11-06. The SAME object, `qwen3-4b-sealed.bin`, with `"attention": "grouped"`. `served.rs` refuses the form on Metal by name, so only `bin/metalrun` loads it, through `Served::apply_measurement`, and says so on its first line | **2.7320**: the form changes no stored bit | Not scored. On the M3 Max the grouped form returned the expanded form's logits bit for bit over 49,140 decode steps of 12 WikiText-2 windows of 4,096, and decoded x1.750 [1.599-1.997] faster at 2,048 tokens of context and x2.380 [2.362-2.396] at 4,096, same tokens (*measured*, [metal-nuit-2026-11-06](../docs/mesures/metal-nuit-2026-11-06.txt), M4 and M5) |
| `qwen3-4b-tetra-e4-metal-kernel-NOT-SERVED.json` | **Not served**: a Metal measurement config for the operator, written 2026-11-06. The SAME object, `qwen3-4b-sealed.bin`, with `"attention": "kernel"` on the eager engine: decode rows at position 1,280 and later go to `kernels/attn_dec.metal`, rows below run the grouped arithmetic, bit for bit the expanded form's on Metal. Loaded only by `bin/metalrun`, through `Served::apply_measurement` | **2.7320**: the form changes no stored bit | Not scored. The kernel returns the canonical replay's bits (`tests/attn_metal.rs`); from position 0 against the grouped form it read +0.0051 % of WikiText-2 perplexity over 12 windows of 4,096, every window within [-0.0268 ; +0.0490] %, top-1 agreement 99.772 %, and decoded x0.864 [0.851-0.879] at 5 tokens, x0.974 [0.848-1.030] at 2,048 and x1.037 [0.996-1.093] at 4,096, same tokens (*measured*, [metal-nuit-2026-11-06](../docs/mesures/metal-nuit-2026-11-06.txt), M6). The threshold of 1,280 keeps the short contexts off the kernel; this config itself was not timed |
| `qwen3-8b-tetra-e4-grouped.json` | **The 8B's served configuration from 2026-11-04 to 2026-11-07**, replaced by its v9 config (operator decision, T8 granted on condition of a light MMLU check, which passed). The SAME object as `qwen3-8b-tetra-e4.json`, `qwen3-8b-sealed-B.bin`, with `"attention": "grouped"` and nothing else different. Added 2026-11-04 | the row of `qwen3-8b-tetra-e4.json`: the form changes no stored bit | **Perplexity** licensed it: WikiText-2 paired against the expanded form through the kernel, +0.0000 % by prefill and -0.0048 % [-0.0117 ; +0.0021] by decode, inside +-0.12 % (*measured*, [gqa-8b-2026-11-04](../docs/mesures/gqa-8b-2026-11-04.txt)). **MMLU**, 285 questions (5 a subject): the same 285 picks under both configs, 65.55 % each (*measured*, [served-grouped-8b-2026-11-04](../docs/mesures/served-grouped-8b-2026-11-04.txt)). Served door **105.8 tok/s** against 98.3, same 256-token text; sweep **106.9 / 93.1 / 82.1 / 67.1** at 5 / 2,048 / 4,096 / 8,192 (*measured*, same journal). **GSM8K**, all 1,319 problems: **88.55 %** against 88.63 % expanded, paired -0.08 pp [-0.61 ; +0.46], p = 1.000 (*measured*, [gsm8k-grouped-8b-2026-11-04](../docs/mesures/gsm8k-grouped-8b-2026-11-04.txt)) |
| `qwen3-14b-tetra-e4-grouped.json` | **The 14B's served configuration from 2026-11-04 to 2026-11-07**, replaced by its v9 config (operator decision, T8 granted at 14B after its control and a light MMLU check both passed). The SAME object as `qwen3-14b-tetra-e4.json`, `qwen3-14b-sealed.bin`, with `"attention": "grouped"` (groups of 5) and nothing else different. Added 2026-11-04 | the row of `qwen3-14b-tetra-e4.json`: the form changes no stored bit | **Perplexity**: WikiText-2 paired against the expanded form through the kernel, +0.0000 % by prefill and +0.0023 % [-0.0062 ; +0.0109] by decode, inside +-0.12 % (*measured*, [gqa-14b-2026-11-04](../docs/mesures/gqa-14b-2026-11-04.txt)). **MMLU**, 285 questions: the same 285 picks under both configs, 74.13 % each (*measured*, [served-grouped-14b-2026-11-04](../docs/mesures/served-grouped-14b-2026-11-04.txt)). Served door **62.5 tok/s** against 58.6, same 256-token text; bench sweep **63.1 / 57.7 / 52.6 / 45.3** at 5 / 2,048 / 4,096 / 8,192 against 59.2 / 42.7 / 34.7 / 21.2 (*measured*, gqa-14b). **GSM8K**, all 1,319 problems: **91.96 %** against 92.04 % expanded, paired -0.08 pp [-0.22 ; +0.07], p = 1.000 (*measured*, [gsm8k-grouped-14b-2026-11-04](../docs/mesures/gsm8k-grouped-14b-2026-11-04.txt)) |
| `qwen3-8b-tetra-q5.json` | Qwen3-8B, `Tetra` + 36 `v_proj` int4 g128, calibrated on `dclm-edu`: `qwen3-8b-dclm.bin` (base, 4,364,205,777 B, sha256 `bcea0d5a7de2fbbfd244ecf1793dd74d94808eadc4a1103533d539829f41bc04`) and `qwen3-8b-dclm-ft.bin` (row scales trained and folded, same size, sha256 `783cef6700e0efa9bca1bf225861932879ce49f5509ec7a22c5c0b3a712ebdad`), in the bucket at `dclm-8b-2026-09-21/` and `dclm-8b-ft-2026-09-21/` ([dclm-8b-rowscales-2026-09-21](../docs/mesures/dclm-8b-rowscales-2026-09-21.txt)). The five values are the 4B's; only the object differs. The head is untied, so `q8` quantizes two tables | **3.0683** as served (*computed* by `rtbits` on the file's bytes, whole model, q8 embedding, tail f16; 3.1064 at tail f32; [dclm-8b-2026-09-21](../docs/mesures/dclm-8b-2026-09-21.txt)) | **68.16 micro, 69.93 macro** for the trained file on the full 14,042-question split, dense reconstruction with the f16 embedding, 4.2080 b/param (*measured*, [dclm-8b-rowscales-2026-09-21](../docs/mesures/dclm-8b-rowscales-2026-09-21.txt)); the base reads 64.87 / 66.58 ([census-8b-2026-09-21](../docs/mesures/census-8b-2026-09-21.txt)). No census through the served kernel: only the 57-question door ([served-8b-2026-09-21](../docs/mesures/served-8b-2026-09-21.txt)) |
| `qwen3-8b-tetra-e4.json` | Qwen3-8B paper-2 object, sealed 2026-09-24: `Tetra` with dclm-edu calibration and trained row scales, `v_proj` + `down_proj@10-26` as int4 g128 records, both untied tables stored int4 g64, `qwen3-8b-sealed-B.bin` (2,815,098,745 B, sha256 `7bdb9a55...`, *measured*, [sealed-8b-27-2026-09-24](../docs/mesures/sealed-8b-27-2026-09-24.txt)). Variant A, `o_proj` + `down_proj@15-20` (2,819,588,161 B, sha256 `0c6b08b3...`), scored 68.81 at 2.7047 and was declined: B beats it by 0.77 pp [+0.26 ; +1.28], p = 0.004, with fewer bytes | **2.6953** as served (*computed* by `rtbits` over the file's bytes, tail f16, both tables q4) | **69.58 micro** on the full 14,042-question split, dense reconstruction, +1.42 pp [+0.93 ; +1.92] over the 68.16 base (*measured*, same journal). Not scored through the kernel |
| `qwen3-14b-tetra-q5.json` | Qwen3-14B, `Tetra` + 40 `v_proj` int4 g128, calibrated on `dclm-edu`: `qwen3-14b-dclm.bin` (base, 6,563,782,117 B, sha256 `f8e975b8...`) and `qwen3-14b-dclm-ft.bin` (row scales trained and folded, same size, sha256 `c825cb92...`), written on the Mac under `~/q14b-dclm-2026-09-22/`, both digests read back on the card ([encode-14b-2026-09-22](../docs/mesures/encode-14b-2026-09-22.txt), [dclm-14b-rowscales-2026-09-22](../docs/mesures/dclm-14b-rowscales-2026-09-22.txt)). The five values are the 4B's; only the object differs. The head is untied, so `q8` quantizes two tables | **2.7371** as served (*computed* by `rtbits` on the file's bytes, whole model, q8 embedding, tail f16; 3.5272 at the f16 tables the census read) | **74.20 micro, 75.64 macro** for the trained file on the full 14,042-question split, dense reconstruction with the f16 tables (*measured*, dclm-14b-rowscales); the base reads 72.53 / 74.07 ([census-14b-base-2026-09-22](../docs/mesures/census-14b-base-2026-09-22.txt)). No census through the served kernel |
| `qwen3-14b-tetra-e4.json` | Qwen3-14B paper-2 object, sealed 2026-09-23: `Tetra` with dclm-edu calibration and trained row scales, `v_proj` + `o_proj` + `down_proj@10-28` as int4 g128 records, both untied tables stored int4 g64, `qwen3-14b-sealed.bin` (5,087,000,541 B, sha256 `61db37fe...`, *measured*, [sealed-8b-14b-2026-09-23](../docs/mesures/sealed-8b-14b-2026-09-23.txt)) | **2.7305** as served (*computed* by `rtbits` over the file's bytes); 2.5217 kernel b/weight | **75.66 micro** on the full 14,042-question split, dense reconstruction, +1.46 pp [+0.99 ; +1.94] over the 74.20 base (*measured*, same journal). Not scored through the kernel |

No field is model-specific. The loader reads every shape from the sealed
file's own `config.json`, and the values name runtime choices: five required
fields and four optional ones. Each object has a file of its own anyway, for the two things a run prints
from it: the note on the `served config:` line, and the path on every dump's
`# config=` line. Borrowing another size's file would record the wrong object on
every line of the run.

## `mlx/`: the MLX door on the Mac

`mlx/qwen3-4b-tetra-e4-mlx.json` is the config of the MLX door,
`ops/mlx_tetra/serve.py`. It is **the 4B's served configuration on the Mac since
2026-11-07** (operator decision: « oiui c'est bon on gare », corrected by « garde * »,
and the option "Garder MLX"), under a gate on the paired interval of its perplexity. Its schema is its own, so it lives in a subdirectory:
`every_shipped_config_resolves` reads the `*.json` of this directory with
`deny_unknown_fields`, and does not descend.

Both perplexity readings ran the file at sha256 `d7bd5b5c…` (commit `fce41da`).
Its `note` then named the first reading's gate, every window inside +-0.12 %.
On 2026-11-07 the note alone was rewritten to name the gate that passed, the
paired interval of `preregistration-mlx-pooled-2026-11-07.md`: sha256
`8542b7f9…`, every other key unchanged.

| file | object | b/param | quality |
|---|---|---|---|
| `mlx/qwen3-4b-tetra-e4-mlx.json` | The SAME object as `qwen3-4b-tetra-e4.json`, `qwen3-4b-sealed.bin` (sha256 `886391a8…`), read from its hfpack by mlx-lm's Qwen3 with the 168 Tetra records as custom Metal kernels, f16 activations, mlx-lm's attention and f16 KV cache. Written 2026-11-07 on the operator's ruling « Sur métal, on a validé mlx donc implem le aussi »: on Metal, a tolerance rather than bit for bit | **2.7320**, the file's (*computed* by `rtbits`): the door reads the stored bits. Resident, it holds q and k twice, alone for the prefill and stacked for the decode, about 0.12 GB more (*computed* from the record shapes) | Not scored on a task. **Perplexity** licensed it: WikiText-2 windows 13 to 24 of 4,096 against the Rust dense reconstruction in f16 on Metal, **-0.0103 %** [-0.0847 ; +0.0642] by decode and **-0.0099 %** [-0.0842 ; +0.0644] by prefill, paired, inside +-0.12 % (*measured*, [mlx-pooled-2026-11-07](../docs/mesures/mlx-pooled-2026-11-07.txt)). A first reading on windows 1 to 12 against the Rust Metal path's 12.580769 read +0.0078 % [-0.0554 ; +0.0710] pooled, with windows 6 and 7 at +0.27 % and -0.13 %, outside the every-window rule it had then (*measured*, [mlx-t8-2026-11-07](../docs/mesures/mlx-t8-2026-11-07.txt)). Not bit for bit: its tokens leave the Rust path's at token 243 after a 5-token prompt (*measured*, [metal-nuit-2026-11-06](../docs/mesures/metal-nuit-2026-11-06.txt), M15). The engine decoded 93.9 tok/s after a 5-token prompt and 70.3 after 4,096 at the prototype's door (*measured*, [metal-nuit-2026-11-06](../docs/mesures/metal-nuit-2026-11-06.txt), M16); the served door is not timed |

Every key has one accepted value. `ops/mlx_tetra/served.py` refuses any other
by name, refuses the Rust keys `fuse`, `decode` and `rows`, any other object,
another mlx or mlx-lm than the pinned 0.32.4 and 0.28.3, and any `LLVQ_*`
variable, since the door reads none. `ops/mlx_tetra/test_served.py` holds each
refusal.

## Why `fuse` is `0` on the served object

`Tetra48` carries no segmented kernel: `planes_source_names` gives it a list of
its own and `tv_planes_seg_h.cu` is not in it. `LLVQ_FUSE=1` on that layout
would compile and then fail at the first group. The value is a fact about the
layout, not a preference, and it is written here so a reader does not conclude
that projection fusion was measured and declined.

## The `attention` field, and why it is optional

Five fields are required and have no defaults. Three are optional:
`attention`, added on 2026-11-02, then `decode` and `rows`, added on
2026-11-05. An absent `attention` means `expanded`, the form every published
number ran. That is what makes the six files written before it keep serving
exactly what they served, down to the logits, while a file that SAYS
`grouped` or `kernel` changes path.

| value | what attention does | where it is measured |
|---|---|---|
| absent, or `"expanded"` | `repeat_kv` copies each key and value head to the query heads that read it, the whole history at every step, on the `cat` store | every published number of this repository |
| `"grouped"` | the query heads of one key head are stacked as rows of one matmul, which reads each cached byte once and copies no history, on a `grow(256)` store | the L40S at 4B, 8B and 14B (`gqa-ppl-2026-11-01`, `gqa-8b-2026-11-04`, `gqa-14b-2026-11-04`); the M3 Max at 4B, bit for bit with the expanded form (`metal-nuit-2026-11-06`) |
| `"kernel"` | decode rows go to the split-KV attention kernel (`attn_dec.cu`, `attn_dec.metal` on the Mac) from position 1,280, or from position 0 beside `"decode": "persistent"`; rows below the threshold run the grouped arithmetic bit for bit | the L40S at 4B only (`attn-t8-r2-2026-11-04`, `attn-t8-below-4b-2026-11-05`); the M3 Max at 4B from position 0 (`metal-nuit-2026-11-06`, M6) |

`"shared"` is refused by name. It exists in the A/B modes of `bin/fusedrun`,
which is where it was shown slower than `grouped` at every length and not bit
for bit on the card either, so no object is served under it.

## The `decode` field (2026-11-05)

The seventh field, optional like `attention`. Absent or `"eager"`, a token is
decoded by the eager path, as every earlier file. `"persistent"` decodes it in
4E's one cooperative launch (`llvq_llm::served_decode`). It changes the engine,
never the arithmetic: under it every decode row goes to the attention kernel
from position 0, on 4E and on the eager path alike (the prefill, and any binary
that does not drive 4E), and 4E owes the eager path's tokens bit for bit. The
served door and `gsm8k` decode through 4E; `mmlu` scores prompts and decodes
nothing. It is accepted only with `"attention": "kernel"` and the served 4B's
other values, and refused on Metal, on any shape but the Qwen3-4B's, and beside
any measurement-mode variable
(`llvq_llm::served::MEASUREMENT_MODES`). It is accepted on the three sealed shapes,
4B, 8B and 14B: at 8B and 14B 4E reads the head on its own table (both are untied),
and at 8B the Tetra `o_proj` under its own rotation. The kernel form from position
0 passed the perplexity band against the grouped form at both sizes
(`served-v9-8b-2026-11-07`, `served-v9-14b-2026-11-07`).

Beside `"decode": "persistent"`, an optional `rows` field names 4E's row
loop: absent or `"v3"` is row form 1, the loop the served-4e door ran; `"v4"`
is row form 2, x1.060 faster after a 5-token prompt with the same tokens
(`shaker-4e-2026-11-05`). `"v6"` is row form 4, v4 with the vector prologues,
197.2 tok/s at the served door against 178.0 with the same tokens
(`served-v6-2026-11-07`). `"v9"` is row form 7, v8 with the int4 rows on tiled
words, x1.118 over v6 after a 5-token prompt with the same tokens
(`step3-4e-2026-11-07`, measurement form). It is the one value that also changes
how the card holds the int4 words: tiled at load and held once, read by 4E v9
and by the eager path's `tv_q4_ht` / `tv_q4_hht` (`Rows4e::int4_layout`).
`"v5"`, `"v7"` and `"v8"` are refused by name: v5 measured x1.001 over v4 and
stayed a measurement, v7's module spills and does not install, v8 is v9's base.
Beside the eager decode the key itself is refused.

The cache store is NOT a field beside the form: it is read from the form
(`AttnHeads::store`). `expanded` has always run on `cat`, and the unexpanded
form exists to read the history where it lies, which `cat` cannot offer because
its buffer moves at every step. Two separate fields could name a pair no
measurement ever ran.

**`grouped` and `kernel` are refused on Metal, by name, and the Rust door serves
nothing there.** Both are measured on the M3 Max: the grouped form returned the
expanded form's logits bit for bit over 49,140 decode steps and decoded x2.380
faster at 4,096 positions; the kernel form from position 0 read +0.0051 % of
perplexity against grouped, inside +-0.12 % (*measured*,
[metal-nuit-2026-11-06](../docs/mesures/metal-nuit-2026-11-06.txt)). No
operator decision serves either on Metal, so `served.rs` keeps refusing them.
The Mac's served path is the MLX door, `mlx/` above.
The two `*-metal-*-NOT-SERVED.json` files reach them only through
`bin/metalrun`, which says so on its first line. `decode: "persistent"` is
refused on Metal even there: 4E has no Metal kernel.

**Not bit for bit.** Past five tokens of context on the card the grouped form
moves the logits: cuBLAS rounds a batch of 8 key heads differently from 32
expanded ones. It is served under the tolerance of `docs/ETAT.md` section 4,
not under identity, and that is a change of gate the operator took on
2026-11-01. The kernel form is served under the same tolerance. A run
that must reproduce a published line to the bit uses the expanded config.

## Why every value is a string

They go through the same `parse` the environment variables go through:
`FusedLayout::parse`, `EmbedMode::parse`, `RotShare::parse`, `FuseMode::parse`,
`KvMode::parse`, `AttnHeads::parse`, `Decode::parse`, `Rows4e::parse`. One
vocabulary, defined once. A second
spelling would be a second thing to keep in step.

## Where they are, on a card

`ops/Dockerfile.cuda` copies this directory to
`/usr/local/share/llvq/configs/`, so inside a job the served run is

```
LLVQ_CONFIG=/usr/local/share/llvq/configs/qwen3-4b-tetra-q5.json \
  mmlu /out/tetra-q5-2026-09-09/qwen3-4b-tetra-q5.bin cuda 40
```

and the same variable puts `fusedrun` on its one-arm path, with no dense
reference loaded beside the 1.39 GB it serves in.

## What the two binaries do differently with it

| binary | with `LLVQ_CONFIG` | without |
|---|---|---|
| `fusedrun` | one arm, no dense reference, no ratio | the bench: two fuse arms and a dense arm, unchanged |
| `mmlu` | scores **through the kernel** | scores a dense reconstruction, unchanged |

`mmlu` without it is what produced every published bar, and that is why the
switch is a named file and never an inference from the artifact: a binary that
silently changed which arithmetic it scored would make the next number
incomparable to all of them while looking like a bug fix.

## The runbook, end to end

Every step is a job on `l40sx1` through `ops/run.py bench`, with the bucket
mounted at `/out` and `--timeout` given by hand (it has no default). `$F` is
the sealed file, `$C` the config inside the image:

```
F=/out/tetra-q5-2026-09-09/qwen3-4b-tetra-q5.bin
C=/usr/local/share/llvq/configs/qwen3-4b-tetra-q5.json
```

1. **`oracle`**, hard rule 10, on every job: `oracle Qwen/Qwen3-0.6B 64 cuda`.
2. **The served door opens** (cents): `LLVQ_CONFIG=$C mmlu $F cuda 1`, one
   question a subject, 57 in all. It proves the config is in the image, the
   served arm loads, and a dump writes. Nothing about the score.
3. **The prefill gate on a tail chunk** (cents): `LLVQ_CONFIG=$C
   LLVQ_PREFILL_TOKENS=203 fusedrun $F`, 203 is 50 chunks of four and one
   of three; the census ends half its prompts on such a chunk and the gate had
   only ever run at 200 and 800. The gate compares N tokens in one call
   against N calls of one token and refuses on a different argmax.
4. **The served protocol** (~0.15 $): `LLVQ_ROT_SHARE=1 LLVQ_EMBED=q8
   LLVQ_FUSED_LAYOUT=tetra48 fusedrun $F 256`, WITHOUT `LLVQ_CONFIG`, so the
   dense arm loads and the 256 tokens have something to be identical to. This
   is the bench; the served path has no dense reference by design.
5. **The census, two arms, one job** (~1.9 h, ~3.5 $ *computed* on the
   measured slope; `--timeout 3h`): first the dense reconstruction of the
   SAME mixed file, `LLVQ_MMLU_DUMP=/out/<run>/mmlu-4b-tetra-q5-file-dense.csv
   mmlu $F cuda 40` (~26 min), then the kernel, `LLVQ_CONFIG=$C
   LLVQ_MMLU_DUMP=/out/<run>/mmlu-4b-tetra-q5-file-kernel.csv mmlu $F cuda 40`
   (~1.5 h). Same 2,280 questions, same weights, only the arithmetic differs.
   The dumps say which is which on their `# arithmetic=` line.
6. **The pair**: `hf buckets cp` both dumps, then `mmlupair <dense> <kernel>`.
   The bar is the paired one at constant file, 0.79–1.44 pp (*measured*,
   `docs/mesures/mmlupair-4b-8b-2026-08-13.txt`). A paired delta beyond it is
   a kernel defect and the number is not published until it is explained.

Steps 2–3 are the smoke test; steps 4–6 need a stamped prereg first (hard
rule 2), and the census needs the operator's go with the cost announced.

The runbook ran on 2026-09-11 on the q5 object (`ops/jobs/f1e-census.sh`,
[f1e-census-2026-09-11](../docs/mesures/f1e-census-2026-09-11.txt)).
