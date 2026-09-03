# `ops/`: running the quantization on Hugging Face Jobs

The dev machine is a Mac with **69 GB** of unified memory. Qwen3-32B weighs
**65.5 GB** in bf16: the model alone fills the machine, and the factorization
peak needs ~26 GB more. The 32B will not run here, no matter how good the code
is.

This directory is the off-machine way out. It is Python because that is the
native HF Jobs API, and **outside the Rust workspace** because the crates are
deliberately auditable and dependency-free. The orchestration is not.

## What the measurement changed

`CLAUDE.md` says "Cholesky dominant". That was true before `faer`, and it is
not any more. Measured with `bin/cholbench` on M3 Max:

| n | G mult-add | s | G mult-add/s |
|---|---|---|---|
| 1024 | 0.72 | 0.02 | 33 |
| 2048 | 5.73 | 0.09 | 62 |
| 3072 | 19.33 | 0.22 | 87 |
| 4096 | 45.81 | 0.43 | 106 |

Breakdown of the run that produced the 16.94 (Qwen3-4B, 14,447 s):

| | time | share |
|---|---|---|
| Leech encoding | ~8,600 s | **59%** |
| forward passes, f64 conversions, writing | ~5,600 s | 39% |
| **Cholesky** | **223 s** | **1.5%** |

That is exactly what explains the drop from 6.3 h to 3.45 h credited to `faer`.

**Consequence: the run is massively CPU-bound, and the GPU does only ~10
minutes of real work** (2 forward passes over 131k tokens). The only number
that decides becomes the **core-hour cost**, and an H200 at $5/h is the worst
on the list.

## Estimate before launching

```bash
uv run ops/run.py estimate Qwen/Qwen3-32B
uv run ops/run.py selftest        # checks the estimator against the real 4B run
```

`selftest` is not decorative: it demands that the weight count land **exactly**
on the 3,633,315,840 of the published run, and that encoding account for 40% to
80% of the 14,447 s measured. An estimator nobody has confronted with a real
run is a spreadsheet.

Output for the 32B:

```
  poids quantifiés     31.21 Md
  poids portés 16 b     1.56 Md   (4.7 % des poids, 27 % de l'artefact)
  checkpoint bf16       65.5 Go
  artefact projeté      11.6 Go   ×5.7
  encodage Leech       245.9 cœur-h
  Cholesky               1.9 cœur-h   (0.8 %)

  cpu-performance          7.7    14.71
  rtx-pro-6000            10.8    29.62
  h200                    10.8    53.86
```

> Caveat: `tie_word_embeddings: false` on the 32B. Unlike the 4B,
> `embed_tokens` and `lm_head` are two tensors. They are 4.7% of the weights but
> **27% of the artifact**, and the ratio falls to ×5.7 instead of the nominal
> ×7.4. This is the "x bits/weight" trap of `CLAUDE.md` §3, which we thought was
> reserved for small models.

## The undecided trade-off

`cpu-performance` and `rtx-pro-6000` come out at about the same price, for
opposite reasons: the CPU is 2× cheaper per core-hour, but has to do the forward
passes by hand. On GPU that is ~10 min; on CPU, several hours, and my
uncertainty on candle's f32 `gemm` throughput on 32 vCPU is a factor of 3. **So
the estimator does not model the forward passes.** Dressing a guess up as an
estimate would be worse than leaving it out.

Step 8B is what settles it, for ~$20 across both flavors.

## Progress

| step | flavor | cost | status |
|---|---|---|---|
| 0. `oracle` on CUDA | `l4x1` | $0.01 | passes, `max \|Δhidden\| = 0.000e0` |
| 1. 0.6B, 3 blocks, CPU then CUDA | `cpu-upgrade`, `l4x1` | $0.11 | passes, chain validated, `verify_artifact` OK |
| 2. **full 8B** | `rtx-pro-6000` | **$11.48** | passes, **×1.267 at 2.0436 b/weight** |
| 3. 32B, 4 blocks (de-risking) | `rtx-pro-6000x2` | $5.43 | passes, memory and bf16 OK, **621 s/block** |
| 4. **full 14B** | `rtx-pro-6000x2` | **$27.67 / 5.03 h** | passes, **×1.1894 at 2.0481 b/weight**, sealed 6.506 GB (×4.54) |
| 5. **full 32B** | `rtx-pro-6000x2` | **~$62 / ~11.4 h** | pending |

> Step 4 cost four more jobs than planned, all of them on the writing side.
> Quantization and sealing went fine: 163 tensors carried, both criteria set in
> advance met. What resisted was **getting 29.5 GB out of a job**: three bucket
> failures, then a SIGBUS. See the section "Bucket mounting fails silently"
> below; it exists because those four jobs cost $0.86 for zero measurement.

**De-risking paid.** It predicted 9 h / $49 by extrapolation from the 8B; the
measurement at `d_in = 25600` gives **621 s/block against the ~500 predicted**,
so 11.4 h and ~$62. $5.43 to correct a $13 error before committing.

**Why the extrapolation missed**: the cost per weight is not independent of the
width. The `n³` factorization goes from 1.6% of a run (0.6B) to 5.5% (8B) to
**16.5%** (32B). The estimator now uses the largest of the measured constants,
so that it errs high rather than low.

Do not skip to step 3: launching a run on a CUDA path that has never been
executed is the kind of job that dies in its 10th hour on a backend divergence
that `bin/oracle` would have caught for $7.

> Step 2 validates the pipeline, it does not produce a publishable number.
> Qwen3-8B also has `tie_word_embeddings: false`, but with a `hidden` of only
> 4096: the embedding there weighs **15.2% of the weights and 57% of the
> artifact**, and the ratio falls to **×3.7**. The 8B is a bad showcase, worse
> than the 4B (×4.63) *and* the 32B (×5.7), while the method is identical.
> Never put it out as a compression result.

**Account prerequisite**: Jobs require a positive prepaid credit balance.
Without one, `launch` reports a `402 Pre-paid credit balance is insufficient`
after passing the cost guard.

## What is still missing on the Rust side

The skeleton is complete; the binary it launches is not.

| item | what it blocks | status |
|---|---|---|
| **C1** `cuda` feature | `--device cuda`. `llvq-llm/Cargo.toml:29` declares `cuda` (and `cudnn` on line 30), `eval.rs:52` routes to `Device::new_cuda`, `ops/Dockerfile.cuda` builds it | passes |
| **C2** `bin/oracle` on CUDA | the proof **has been redone on this backend**: step 0 above (`l4x1`, $0.01) returns `max \|Δhidden\| = 0.000e0`, as on Metal. It still has to be replayed at every backend change, which is what `cmd_oracle` (`ops/run.py:505`) exists for | passes |
| **C4** resume from checkpoint | long runs. A Job's default timeout is **30 min**; the maximum duration is not documented. **Done 2026-08-09**: `LLVQ_RESUME=<shard>` resumes behind a segment, and the last segment's file **is** the complete artifact: it copies its shard forward, so there is no merge step. Proved, not asserted: on a **complete Qwen3-0.6B (28 blocks, CPU)**, a single-piece run and a run cut 14/14 return the **same file by SHA-256** (`6c8ba465…`, 130,934,618 bytes) and the same perplexity to the ten-thousandth (58.4879). And it was **free within the noise**: 1,747 s in one piece against 1,730 s for the two segments end to end. `llvq-llm/tests/resume.rs` demands the same property continuously on a miniature model, in 0.9 s | passes |
| **C5** local path for `LLVQ_MODEL` | the `Volume(type="model")` mount. **Done 2026-08-08**: `Checkpoint::fetch` decides *syntactically* between directory and repo (leading `/`, `./`, `../`, `~/`), so `--mount-model` works, without a line of Python | passes |
| **C6** quantizer memory | 12.4 GB of factors coexist at 32B, 6.2 of them never read when `group_scales` is off. **Done 2026-08-08**, but the statement was misleading: those 6.2 GB are on the **plateau**, not on the peak. Measured on 0.6B, −166 MiB of floor and **peak unchanged**. At 32B, ~0.96 GB of host peak (1.4%), and not "70.6 → 64.4". The real gain is elsewhere, found along the way: the f32 accumulators freed as soon as `to_f64()` runs, worth **3.10 GB of VRAM** at 32B, on the resource that sat at 80% during de-risking | passes |
| C3 bf16 loading | *became optional*: `cpu-performance` has 256 GB of RAM, the model fits in f32 | n/a |

**C4 was the one that mattered most, and it is the one that was done.**
Calibration is sequential. Block *t* is quantized against the activations that
passed through blocks 0..*t*−1 **already quantized**, so resuming means
reloading the base checkpoint, re-applying the matrices already written into
the artifact, and starting again at block *k*.

What makes that possible without serializing anything more: the shard **is** the
state. `verify_artifact` demands at every run that `decode_matrix` return the
evaluated weights **bit for bit** (6,945,767,424 weights on the 8B), so the
matrices already written are a lossless snapshot of everything the loop changed.
Only the hidden states are left, and we **recompute** them by replaying the
forward passes over the inherited blocks.

```bash
# segment 1: blocks 0..31
uv run ops/run.py launch --model Qwen/Qwen3-32B --blocks 32 \
    --bucket auto --name qwen3-32b-s1 …
# segment 2: blocks 32..63. "blocks" is an ABSOLUTE bound, not a count.
uv run ops/run.py launch --model Qwen/Qwen3-32B --blocks 64 \
    --resume /out/qwen3-32b-s1.llvq --bucket auto --name qwen3-32b-s2 …
```

The resume block is **not** passed in: `smoke` reads it from the shard. A single
source of truth. A run that can be told two different things produces an
artifact with a hole or an overlap, and nothing downstream detects either one.

What is refused before a single byte is downloaded: a shard written under
another codebook, another rotation, another model, in another record order,
and, through the `<artifact>.state` file written beside it, under another
calibration, another damping, another dtype or another backend. **An artifact
whose two halves do not share the same configuration is valid in appearance and
wrong**; that is the only failure mode this path can have, so it is the one it
is built against.

How many segments: `uv run ops/run.py estimate <model> --segments N` prices the
plan. The extra cost is **linear in (N−1)**, since with equal shares each added
cut point makes it re-read **half the model** on average, while the maximum loss
only falls as 1/N. Two segments is where the trade is best: the loss goes from
11.4 h to ~6 h against re-reading half a model.

Caveat: the hidden states are recomputed, not restored. On CPU and Metal the
equality is exact and the test suite demands it. On a backend that is not
deterministic from one process to the next (cuBLAS picks its algorithms per
launch), two segments are not *guaranteed* bit-identical to a single run. The
gap is far below the ±0.7% seed-to-seed spread already measured here, but it is
real, it declares itself, and the result line prints it.

## Building the image

From the **repository root**, not from `ops/`:

```bash
docker build -f ops/Dockerfile -t <user>/llvq:cpu .
```

```bash
docker push <user>/llvq:cpu
```

The CUDA variant has its own recipe, `ops/Dockerfile.cuda`. A Space builds its
Dockerfile **as is**: `--build-arg` never reaches the Hub builder, so the
CPU/CUDA choice is made by *which file* you upload, and `publish --cuda` is what
does it (`cmd_publish` in `ops/run.py`):

```bash
uv run ops/run.py publish <user>/llvq-runner-cuda --cuda
```

Locally, the same image builds directly, still from the root:

```bash
docker build -f ops/Dockerfile.cuda -t <user>/llvq:cuda .
```

The compute capability is pinned there to `89` (Ada): a Space builder has no
GPU, so `candle`'s `nvidia-smi` detection fails and the build dies without it.
Change the line and rebuild to target an H200.

> `Cargo.lock` is tracked by git, `.gitignore` is only two lines, `target/` and
> `__pycache__/`, and both Dockerfiles build with `--locked`. The image is
> therefore built from the committed tree, and a number it produces can be tied
> to a commit.

## Launching

```bash
uv run ops/run.py launch --model Qwen/Qwen3-8B --flavor cpu-performance --image <user>/llvq:cpu --bucket <org>/llvq-runs --name qwen3-8b-c12L3
```

`launch` **refuses** above `--max-usd` ($60 by default) without `--yes`, and
sets a `timeout` computed at 1.5× the estimate. The 30-minute default would kill
every real run.

Two volumes handle the download and the output:

- `Volume(type="model", …, read_only=True)` mounts the checkpoint repo, so
  65 GB are not re-downloaded at every relaunch (**C5**, done 2026-08-08);
- `Volume(type="bucket", …, read_only=False)` receives the artifact, and will
  receive the **C4** resume checkpoints.

```bash
uv run ops/run.py watch <job_id>
```

## Bucket mounting fails silently, three ways, measured 2026-08-10

Three attempts at the same 14B AWQ reconstruction, three failures, **none in
the reconstruction code**: its five checks were green every time (L2 per matrix,
worst L4 margin ×658 for a ×100 criterion, L1 0.998, embedding reproduced value
for value, 163 tensors). All three are in the writing layer, and **none of the
three raises an error at the moment it happens**.

1. **Two jobs writing into the same bucket do not mount.** The second dies on
   `Volume mount failed: init container exhausted retries`, before starting, so
   without billing. **Serialize the jobs that write to the same bucket.**
   Sealing and dequantization cannot be parallelized, contrary to what the 14B
   resume note announced.
2. **A 4 GB write can be truncated with no error.** The 7th shard of 8 came out
   at 2,642,414,752 bytes where its peers were 3,963,622,136, and the internal
   `rename` of `safetensors.save_file` found nothing left to rename. Hence the
   **1 GB** default of `dequant --shard-gb` here, against the 4 GB of
   `awq_dequant.py`, which are still fine for a local disk.
3. **`os.rename` can return success without renaming.** With 1 GB shards, the 33
   files were written, `finish()` ran its 33 renames, **raised nothing**, wrote
   its `index.json`, and three files stayed under their temporary name. This is
   not latency: ten minutes later they were still there.

**What works, and it is the workaround**: `hf cp` bucket-to-bucket is a
server-side copy, instant (2 s for 1.5 GB) and reliable. Repairing therefore
costs a few seconds once you know which file is missing.

**How to know the data is complete**, with an unforgiving sum: add up the sizes
of the weight files and subtract the tensor byte count the job announces. The
remainder must be worth a few hundred to a few thousand bytes per file, the
`safetensors` JSON headers. One truncated file and the gap becomes enormous.

```bash
hf buckets ls hf://buckets/<user>/<bucket>/<dir>/ \
 | awk '$1 ~ /^[0-9]+$/ && $NF ~ /model-000/ {s+=$1; n++} \
        END {printf "%d fichiers, somme %d\n", n, s}'
```

Caveat: `get_bucket_file_metadata` does NOT return the content size. It returns
~1100 bytes for every file, including a 36,514-byte `index.json` downloaded and
parsed successfully: that is the size of the Xet pointer. **The `hf buckets ls`
sizes are authoritative.**

## On splitting across machines

`cpu-upgrade` comes out at **$0.93** for the 32B's 246 core-hours, against
$14.71 on `cpu-performance`, 16× cheaper per core-hour. It is the biggest
theoretical lever on the list, and it is probably unusable.

The **rows** of a matrix are independent (`parallel_matches_serial_exactly`
demands it to the bit), so a matrix can be split across workers. But every
worker needs the `U` factor, which is **5.24 GB** for `down_proj`: distributing
it 64 times costs more than it saves on a single run.

The **blocks** are not parallelizable at all, because calibration is sequential.
Decoupling them would mean computing every Hessian on the FP model, which is
what QuIP# does (and it still reaches 17.04). That is a trade-off of method, not
of infrastructure, and it is not taken to save $14.

**The right split is temporal, not spatial**: C4, plus per-minute billing, lets
you kill a run that goes wrong without losing anything.
