# The served configurations

One file per shipped object. It carries the choices that decide how a `.llvq`
is read, and it is the authority on them: `LLVQ_CONFIG=<file>` puts a runner on
the served path, and an environment variable that contradicts the file is
refused rather than silently preferred.

There is no built-in served default. A runner with no `LLVQ_CONFIG` is a runner
in measurement mode, reading the environment variables it always read, which
is what every A/B in `docs/mesures/` depends on.

| file | object | b/param | MMLU |
|---|---|---|---|
| `qwen3-4b-tetra-q5.json` | Qwen3-4B, `Tetra` + 36 `v_proj` int4 g128, `qwen3-4b-tetra-q5.bin` (1,794,564,765 B, sha256 `ae31087a4b72494d52394f2cf2070da37a845c0a7d4aded65d5bdb0bd18b2263`, *measured*, [references-comptabilite-2026-09-18](../docs/mesures/references-comptabilite-2026-09-18.txt)). The only copy is in the bucket at `tetra-q5-2026-09-09/` | **2.7475** as served (*measured*, whole model, q8 embedding, tail f16, same journal). The f32 tail accounting every published figure used gives 2.8126, and the card has held the tail in f16 since 2026-08-09, so 2.7475 is the width that runs | **56.37 micro, 58.42 macro** on the full 14,042-question split, dense reconstruction (*measured*, [f1e-census-2026-09-11](../docs/mesures/f1e-census-2026-09-11.txt)). On the 2,280-question plan the same file reads 55.52 dense and **55.66 through the served kernel**, 3 discordant questions, all three f16 tie-breaks. The 56.95 the record also carries is a different file: the pure `Tetra` `qwen3-4b-tetra.bin` with `v_proj` restored to int4 at load by `LLVQ_RESTORE_Q4` (`q5-tetra-2026-09-06.txt` T2), so it is not this object's score |
| `qwen3-4b-tetra-e4.json` | Qwen3-4B paper-2 object: `Tetra` with dclm-edu calibration and trained row scales, `v_proj` + `o_proj` + `down_proj@12-23` as int4 g128 records, the tied embedding stored int4 g64, `qwen3-4b-sealed.bin` (1,418,224,685 B, sha256 `886391a8c03f66dc269cc65c3598c6627dbdcd259180aff36604ef10d37371b8`). Built on 2026-09-23 from the 61.11 object by `int4swap` then `embedq q4` ([embed-q4-swap-2026-09-23](../docs/mesures/embed-q4-swap-2026-09-23.txt)) | **2.7320** as served (*computed* by `rtbits` over the file, tail f16, embedding q4); 2.5420 kernel b/weight | **63.37 micro** for the same weights on the dense path (*measured*, the restore arm of the same journal); the file itself is not scored yet |
| `qwen3-4b-tetra-e4-grouped.json` | **The 4B's served configuration since 2026-11-02** (operator decision). The SAME object as the row above, `qwen3-4b-sealed.bin`, served with the grouped attention: `"attention": "grouped"` and nothing else differs. Added 2026-11-02 | **2.7320** as served, the row above's number: the form changes no stored bit | Not scored. **Perplexity** is what licensed it: WikiText-2 paired against the expanded form through the kernel, +0.0000 % by prefill (bit for bit) and +0.0095 % [-0.0048 ; +0.0238] by decode, inside the +-0.12 % band of `docs/ETAT.md` section 4 (*measured*, [gqa-ppl-2026-11-01](../docs/mesures/gqa-ppl-2026-11-01.txt)). Through the served door it decodes at **133.3 tok/s** after a 5-token prompt against 121.5, and by the context sweep **134.5 / 113.8 / 98.0 / 77.5** against 122.5 / 82.7 / 61.4 / 33.2 at 5 / 2,048 / 4,096 / 8,192, with the same 256-token text (*measured*, [served-grouped-2026-11-02](../docs/mesures/served-grouped-2026-11-02.txt)). No task score has run under this form |
| `qwen3-8b-tetra-q5.json` | Qwen3-8B, `Tetra` + 36 `v_proj` int4 g128, calibrated on `dclm-edu`: `qwen3-8b-dclm.bin` (base, 4,364,205,777 B, sha256 `bcea0d5a7de2fbbfd244ecf1793dd74d94808eadc4a1103533d539829f41bc04`) and `qwen3-8b-dclm-ft.bin` (row scales trained and folded, same size, sha256 `783cef6700e0efa9bca1bf225861932879ce49f5509ec7a22c5c0b3a712ebdad`), in the bucket at `dclm-8b-2026-09-21/` and `dclm-8b-ft-2026-09-21/` ([dclm-8b-rowscales-2026-09-21](../docs/mesures/dclm-8b-rowscales-2026-09-21.txt)). The five values are the 4B's; only the object differs. The head is untied, so `q8` quantizes two tables | **3.0683** as served (*computed* by `rtbits` on the file's bytes, whole model, q8 embedding, tail f16; 3.1064 at tail f32; [dclm-8b-2026-09-21](../docs/mesures/dclm-8b-2026-09-21.txt)) | **68.16 micro, 69.93 macro** for the trained file on the full 14,042-question split, dense reconstruction with the f16 embedding, 4.2080 b/param (*measured*, [dclm-8b-rowscales-2026-09-21](../docs/mesures/dclm-8b-rowscales-2026-09-21.txt)); the base reads 64.87 / 66.58 ([census-8b-2026-09-21](../docs/mesures/census-8b-2026-09-21.txt)). No census through the served kernel: only the 57-question door ([served-8b-2026-09-21](../docs/mesures/served-8b-2026-09-21.txt)) |
| `qwen3-8b-tetra-e4.json` | Qwen3-8B paper-2 object, sealed 2026-09-24: `Tetra` with dclm-edu calibration and trained row scales, `v_proj` + `down_proj@10-26` as int4 g128 records, both untied tables stored int4 g64, `qwen3-8b-sealed-B.bin` (2,815,098,745 B, sha256 `7bdb9a55...`, *measured*, [sealed-8b-27-2026-09-24](../docs/mesures/sealed-8b-27-2026-09-24.txt)). Variant A, `o_proj` + `down_proj@15-20` (2,819,588,161 B, sha256 `0c6b08b3...`), scored 68.81 at 2.7047 and was declined: B beats it by 0.77 pp [+0.26 ; +1.28], p = 0.004, with fewer bytes | **2.6953** as served (*computed* by `rtbits` over the file's bytes, tail f16, both tables q4) | **69.58 micro** on the full 14,042-question split, dense reconstruction, +1.42 pp [+0.93 ; +1.92] over the 68.16 base (*measured*, same journal). Not scored through the kernel |
| `qwen3-14b-tetra-q5.json` | Qwen3-14B, `Tetra` + 40 `v_proj` int4 g128, calibrated on `dclm-edu`: `qwen3-14b-dclm.bin` (base, 6,563,782,117 B, sha256 `f8e975b8...`) and `qwen3-14b-dclm-ft.bin` (row scales trained and folded, same size, sha256 `c825cb92...`), written on the Mac under `~/q14b-dclm-2026-09-22/`, both digests read back on the card ([encode-14b-2026-09-22](../docs/mesures/encode-14b-2026-09-22.txt), [dclm-14b-rowscales-2026-09-22](../docs/mesures/dclm-14b-rowscales-2026-09-22.txt)). The five values are the 4B's; only the object differs. The head is untied, so `q8` quantizes two tables | **2.7371** as served (*computed* by `rtbits` on the file's bytes, whole model, q8 embedding, tail f16; 3.5272 at the f16 tables the census read) | **74.20 micro, 75.64 macro** for the trained file on the full 14,042-question split, dense reconstruction with the f16 tables (*measured*, dclm-14b-rowscales); the base reads 72.53 / 74.07 ([census-14b-base-2026-09-22](../docs/mesures/census-14b-base-2026-09-22.txt)). No census through the served kernel |
| `qwen3-14b-tetra-e4.json` | Qwen3-14B paper-2 object, sealed 2026-09-23: `Tetra` with dclm-edu calibration and trained row scales, `v_proj` + `o_proj` + `down_proj@10-28` as int4 g128 records, both untied tables stored int4 g64, `qwen3-14b-sealed.bin` (5,087,000,541 B, sha256 `61db37fe...`, *measured*, [sealed-8b-14b-2026-09-23](../docs/mesures/sealed-8b-14b-2026-09-23.txt)) | **2.7305** as served (*computed* by `rtbits` over the file's bytes); 2.5217 kernel b/weight | **75.66 micro** on the full 14,042-question split, dense reconstruction, +1.46 pp [+0.99 ; +1.94] over the 74.20 base (*measured*, same journal). Not scored through the kernel |

No field is model-specific. The loader reads every shape from the sealed
file's own `config.json`, and the six values name runtime choices. Each object has a file of its own anyway, for the two things a run prints
from it: the note on the `served config:` line, and the path on every dump's
`# config=` line. Borrowing another size's file would record the wrong object on
every line of the run.

## Why `fuse` is `0` on the served object

`Tetra48` carries no segmented kernel: `planes_source_names` gives it a list of
its own and `tv_planes_seg_h.cu` is not in it. `LLVQ_FUSE=1` on that layout
would compile and then fail at the first group. The value is a fact about the
layout, not a preference, and it is written here so a reader does not conclude
that projection fusion was measured and declined.

## The `attention` field, and why it is optional

Five fields are required and have no defaults. `attention` is the sixth, added
on 2026-11-02, and it is the one field that may be absent: an absent
`attention` means `expanded`, the form every published number ran. That is what
makes the six files written before it keep serving exactly what they served,
down to the logits, while a file that SAYS `grouped` changes path.

| value | what attention does | where it is measured |
|---|---|---|
| absent, or `"expanded"` | `repeat_kv` copies each key and value head to the query heads that read it, the whole history at every step, on the `cat` store | every published number of this repository |
| `"grouped"` | the query heads of one key head are stacked as rows of one matmul, which reads each cached byte once and copies no history, on a `grow(256)` store | the L40S, on the 4B object only (`gqa-ab-2026-11-01`, `gqa-ppl-2026-11-01`) |

`"shared"` is refused by name. It exists in the A/B modes of `bin/fusedrun`,
which is where it was shown slower than `grouped` at every length and not bit
for bit on the card either, so no object is served under it.

The cache store is NOT a field beside the form: it is read from the form
(`AttnHeads::store`). `expanded` has always run on `cat`, and the unexpanded
form exists to read the history where it lies, which `cat` cannot offer because
its buffer moves at every step. Two separate fields could name a pair no
measurement ever ran.

**`grouped` is refused on Metal, by name.** The form is backend-independent
code in `model.rs`, which is exactly why a silent acceptance would be easy and
wrong: no throughput, no perplexity and no reading of Metal's own rounding
exists for it, and the tolerance it passed was passed on an NVIDIA card. The
refusal names the measurement that would lift it.

**Not bit for bit.** Past five tokens of context the grouped form moves the
logits: cuBLAS rounds a batch of 8 key heads differently from 32 expanded ones.
It is served under the tolerance of `docs/ETAT.md` section 4, not under
identity, and that is a change of gate the operator took on 2026-11-01. A run
that must reproduce a published line to the bit uses the expanded config.

## Why every value is a string

They go through the same `parse` the environment variables go through:
`FusedLayout::parse`, `EmbedMode::parse`, `RotShare::parse`, `FuseMode::parse`,
`KvMode::parse`, `AttnHeads::parse`. One vocabulary, defined once. A second
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

Steps 2–3 ran green on 2026-09-11 (jobs `6aa40f3a`, `6aa4108b`, `6aa41263`,
0.17 $). Step 5 is `ops/jobs/f1e-census.sh`, one command, prereg stamped,
not launched, by the operator's choice.
