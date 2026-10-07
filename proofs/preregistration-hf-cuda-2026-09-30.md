# Prereg. Stage 4: the served CUDA Tetra kernel reached from PyTorch

Status: written, committed and TIMESTAMPED on 2026-09-30, BEFORE the launch.
Operator go given 2026-09-30, explicitly, for this job.
Cost: **$0.45 estimated**, hard cap **$1.20**, which is the 40 minute timeout at l40sx1's
$1.80 an hour. No cap on the wave is in force; this is the first paid job since 2026-09-26 and the
running total before it is $241.88 over 214 jobs (`docs/data/jobs.csv`).

A timestamped prereg is no longer edited. Any departure goes in
`proofs/preregistration-hf-cuda-2026-09-30-ECARTS.md`.

## 1. Question

Can `tv_tetra48_h`, the kernel that serves the three sealed files today, be compiled by `nvcc` into
a torch extension and give the same answers?

On Metal the same question was answered yes on 2026-09-30, with the weights resident compressed and
256 greedy ids of 256 (`docs/mesures/hf-metal-m1-4b-2026-09-30.txt`). Nothing about that transports:
the CUDA kernel is a different file, its output is f16 where Metal's is f32, its launch geometry is
the host's, and it is compiled by NVRTC today rather than by `nvcc`.

## 2. The four arms, in this order

A failure has to say which thing failed, so the arms run in increasing cost and the first failure
stops the rest.

1. **The build.** `nvcc` over `llvq-hf/llvqhf/csrc/tetra_cuda.cu`, which `#include`s the served
   `llvq-llm/kernels/tv_tetra48_h.cu` rather than a copy of it. A failure here costs two minutes and
   is a complete answer.
2. **The arithmetic, per row.** One matrix of each shape at layer 0, against the dense
   reconstruction of the same record, on random activations. The bar is **1e-2 relative**, loose
   because the kernel writes f16 where Metal writes f32.
3. **The tokens.** 64 greedy ids on the four prompts of `bin/run`, against the dump `bin/run`
   produced on the Mac at f32 on the CPU, the same reference the Metal arm was gated against.
4. **The memory**, measured on the loaded model.

Arm 2 is the sensitive one and it is deliberately before arm 3. M1 measured why: dropping the tail
entirely moves arm 2 by 8.79 % and leaves arm 3 untouched on two prompts of four.

## 3. What the reference costs, said plainly

The token reference was produced on another machine. An identity may cross machines where a ratio may
not, and no ratio is formed anywhere in this job. The price is diagnostic, not soundness: if the ids
differ I cannot tell the card from the kernel from this job alone, and the follow-up would be a second
job in our own image, which carries `bin/run` but no `nvcc`, so it cannot be the same job.

## 4. Setup

```bash
bash ops/jobs/hf-cuda-4b.sh upload     # sources and reference into the bucket
DRY_RUN=1 bash ops/jobs/hf-cuda-4b.sh  # parses, launches nothing
bash ops/jobs/hf-cuda-4b.sh            # launches
```

Image `nvidia/cuda:12.4.1-devel-ubuntu22.04`, the base our own build stage uses, because our runtime
image carries no `nvcc`. torch arrives by pip, about three minutes. The packed 4B is read from the
bucket, `hf-cuda-4b-2026-09-30/model/`, `model.safetensors` sha256 `a28348cafa82d23d...`, packed at
commit `0ea902d` from `qwen3-4b-sealed.bin` whose sha256 is `886391a8c03f66dc...`.

## 5. Controls

1. The check script is verified in the job against the sha256 of the file that was uploaded.
2. `nvidia-smi` and `nvcc --version` are recorded before anything is measured.
3. Arm 2 refuses to hand over to arm 3 if any shape is past the bar.
4. The kernel is the repository's, included and not copied, which the tarball's shape enforces:
   `tv_tetra48_h.cu` includes `../../llvq-cuda/kernels/`, so a flat copy would not build at all.
5. `d_out` not filling whole blocks, a stream of the wrong length, a tail that is not f16: refused by
   name in the binding, the same guards the Metal one carries.

## 6. Signed predictions

**The build passes.** The binding is a copy of the Metal one whose differences are the two the kernel
declares, an f16 output and the host's launch geometry. If it fails it fails on an include path, not
on the kernel.

**Arm 2 lands between 1e-4 and 3e-3 relative.** f16 carries 2⁻¹¹, about 4.9e-4, and the output is
written through `f2h`, so a maximum over up to 9,728 rows should sit a few times that. Metal reads
2.6e-5 with an f32 output, and the difference between the two numbers is the f16 store and nothing
else.

**Arm 3 matches on all four prompts.** Weaker than the Metal prediction was: every matvec rounds to
f16 here, where the Metal arm kept f32, and the reference is a dense f32 path on another machine.
What supports it is that the served engine is f16 throughout and its 4B matched its own dense path
over 256 tokens.

**The job bills between 12 and 20 minutes.** Image pull and apt two, pip three, the build two, arm 2
one, arm 3 three, overhead the rest.

I have been wrong on signed predictions six times in this repository, twice on 2026-09-30. These are
scored the same way.

## 7. What this job cannot establish

- Nothing about throughput. The path dispatches once a token a projection and is not the served
  engine; no tok/s from it is divided against anything (rule 5).
- Nothing about the int4 records or the embedding, which stay dense as they do on Metal.
- Nothing about the 8B and the 14B.
- Nothing about a second kind of card: this is one L40S.
- Nothing about `bin/run` on the card, which this image cannot carry.
