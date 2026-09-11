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
| `qwen3-4b-tetra-q5.json` | Qwen3-4B, `Tetra` + 36 `v_proj` int4 g128 | 2.8138 | 56.95 |

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
