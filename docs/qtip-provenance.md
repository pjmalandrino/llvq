# QTIP: why this kernel is not in the repository (2026-08-20)

The `qtip` comparison arm of the benchmark measures the 2-bit matvec kernel
published by Cornell-RelaxML. Unlike the AWQ arm, **its code is not committed
here**: it is fetched at job time by [`ops/fetch-qtip.sh`](../ops/fetch-qtip.sh).

## The reason, and it is not symmetric with the AWQ arm

| | AWQ | QTIP |
|---|---|---|
| upstream licence | MIT | **GPL v3** |
| in the repository | yes: `llvq-cuda/kernels/awq_gemv.cu`, `include_str!` | **no** |
| load path | embedded, `LLVQ_KERNEL_DIR` as override | `LLVQ_QTIP_DIR` (its own variable) |

The repository is under MIT OR Apache-2.0. Redistributing a GPL v3 file inside
it would put the whole repository under GPL v3, which is not the project's choice.
Fetching at job time avoids the question without losing any of the measurement.

What the GPL constrains is DISTRIBUTION, not use. Running GPL software to
produce a measurement, patching it so it compiles, timing it, publishing its
times: none of that is restricted. The morning plan
([`plan-f2-qtip-2026-08-20.md`](archive/plan-f2-qtip-2026-08-20.md)) presented
"without running a line of their Python" as a legal constraint; that was a
confusion. Skipping their Python was a simplicity choice, and it turned out we
did not need it: the format was derived from the CUDA, then validated against a
transcription of their code.

## What the script does, and what it refuses to do

1. Refuses to write into a non-empty directory.
2. Downloads `inference.cu` and `inference.h` at the **pinned** commit
   `e90c6688c8dfae326a3a81b5eb032db7c6680ec0`.
3. **Checks both sha256** and fails loudly otherwise. That is the point that
   matters: an upstream file changing in silence would move a published number
   and nothing would say so.
4. Removes **four dead lines**: `#include <cuda/pipeline>`, `#include
   <mma.h>`, `#include <c10/cuda/CUDAStream.h>`, `using namespace nvcuda;`.
   Each was verified to have **zero uses** in the file on 2026-08-20 (the MMA
   goes through inline PTX asm, not the `wmma` API), so the generated device
   code is unchanged: they are dropped because NVRTC carries neither torch nor
   libcu++. The `CHECK_CUDA`/`CHECK_CONTIGUOUS` macros contain `TORCH_CHECK` but
   are never expanded in this file, so they are **left as they are** rather than
   edited.
5. **Proves** the patch afterwards (re-grep of the four lines and four
   residual tokens) instead of trusting the filter.
6. Writes a `PROVENANCE.txt` beside it: URL, commit, sha256 before patch, lines
   removed, date, and the licence notice.

## Its own variable, and why that is not cosmetic

The QTIP kernel arrives through **`LLVQ_QTIP_DIR`**, never through
`LLVQ_KERNEL_DIR`. The latter means *"override ALL kernels from this
directory"*: every loader of the benchmark looks there for **its** files and
**fails hard** if it does not find one (`load_sources_many`,
`llvq-cuda/src/lib.rs:230`). Pointing it at the output of `fetch-qtip.sh` would
therefore break the whole benchmark on a `matvec.cu: No such file or directory`.
QTIP is an **addition**, not an override: two distinct variables that compose.

Corollary: only the **device half** comes from that directory. The `extern "C"`
shims that give the kernel a name are ours, committed, and embedded in the
binary, so they are always present and locked to the version that launches them.

## The honest limit, to be declared in the paper

Our repository alone does not replay this arm. It needs the network and the
upstream repository alive at the pinned commit. The other arms of the benchmark
replay offline; this one does not. If upstream disappears, the sha256 digests in
`PROVENANCE.txt` can still authenticate a copy found elsewhere, but they cannot
produce one.

That is the price of the licence. It is declared, and copying the file here does
not work around it.
