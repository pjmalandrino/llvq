# The served configurations

One file per shipped object. It carries the choices that decide how a `.llvq`
is read, and it is the authority on them: `LLVQ_CONFIG=<file>` puts a runner on
the served path, and an environment variable that contradicts the file is
refused rather than silently preferred.

There is no built-in served default. A runner with no `LLVQ_CONFIG` is a runner
in measurement mode, reading the environment variables it always read — which
is what every A/B in `docs/mesures/` depends on.

| file | object | b/param | MMLU |
|---|---|---|---|
| `qwen3-4b-tetra-q5.json` | Qwen3-4B, `Tetra` + 36 `v_proj` int4 g128, `qwen3-4b-tetra-q5.bin` (1,794,564,765 B, sha256 `c084a47c…27d2`) | 2.8138 (*computed*, whole model, q8 embedding, `docs/mesures/q5-tetra-2026-09-06.txt` accounting) | 56.95 (*measured* — **on a different file**: the PURE Tetra `qwen3-4b-tetra.bin` with `v_proj` restored to int4 at load by `LLVQ_RESTORE_Q4`, dense candle arm, L40S, `q5-tetra-2026-09-06.txt` T2). The shipped mixed file has scored MMLU on **no arm yet**, and the served kernel has scored MMLU **never**. The first served-arm score is a first measurement, not a regression against 56.95. |

## Why `fuse` is `0` on the served object

`Tetra48` carries no segmented kernel: `planes_source_names` gives it a list of
its own and `tv_planes_seg_h.cu` is not in it. `LLVQ_FUSE=1` on that layout
would compile and then fail at the first group. The value is a fact about the
layout, not a preference, and it is written here so a reader does not conclude
that projection fusion was measured and declined.

## Why every value is a string

They go through the same `parse` the environment variables go through —
`FusedLayout::parse`, `EmbedMode::parse`, `RotShare::parse`, `FuseMode::parse`,
`KvMode::parse`. One vocabulary, defined once. A second spelling would be a
second thing to keep in step.

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
2. **The served door opens** (cents): `LLVQ_CONFIG=$C mmlu $F cuda 1` — one
   question a subject, 57 in all. It proves the config is in the image, the
   served arm loads, and a dump writes. Nothing about the score.
3. **The prefill gate on a tail chunk** (cents): `LLVQ_CONFIG=$C
   LLVQ_PREFILL_TOKENS=203 fusedrun $F` — 203 is 50 chunks of four and one
   of three; the census ends half its prompts on such a chunk and the gate had
   only ever run at 200 and 800. The gate compares N tokens in one call
   against N calls of one token and refuses on a different argmax.
4. **The served protocol** (~0.15 $): `LLVQ_ROT_SHARE=1 LLVQ_EMBED=q8
   LLVQ_FUSED_LAYOUT=tetra48 fusedrun $F 256` — WITHOUT `LLVQ_CONFIG`, so the
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
0.17 $). Step 5 is `ops/jobs/f1e-census.sh`, one command, prereg stamped —
not launched, by the operator's choice.
