#!/usr/bin/env bash
# Fetch the QTIP 2-bit inference kernel for the `qtip` comparison arm.
#
# WHY THIS EXISTS INSTEAD OF A VENDORED COPY. The AWQ arm ships as
# `llvq-cuda/kernels/awq_gemv.cu`, embedded with `include_str!`, because AWQ is
# MIT. QTIP is **GPL v3**, which this repository (MIT OR Apache-2.0) does not
# redistribute. So the kernel is fetched at job time and never committed. Using
# GPL software to produce a measurement is unrestricted; only distribution is.
#
# The bench reads this directory through `LLVQ_QTIP_DIR` — deliberately NOT the
# `LLVQ_KERNEL_DIR` that overrides our own kernels. That one means "override
# every kernel source from here", and every loader fails hard on a file it does
# not find, so pointing it at this directory would break the whole bench. QTIP
# is an addition, not an override, and the two variables compose.
#
# Only the device half is read from here. The `extern "C"` shims that name the
# kernel are ours, committed, and embedded in the binary.
set -euo pipefail

COMMIT=e90c6688c8dfae326a3a81b5eb032db7c6680ec0
RAW="https://raw.githubusercontent.com/Cornell-RelaxML/qtip/${COMMIT}/qtip-kernels/src"
# sha256 of the pristine upstream files, verified 2026-08-20. A silent upstream
# change must stop the job, not quietly move a published number.
SHA_CU=dcb0d9dd7b26953fefa854e0c9e9454fc0c6bcab8a92f1621dffdf287d229774
SHA_H=98ba023d37e340cb4a1f5d7af1074eb01ca1d81d08029d8c99eec32b8f20f838

DEST=${1:?usage: fetch-qtip.sh <destination-dir>}

if [ -e "$DEST" ] && [ -n "$(ls -A "$DEST" 2>/dev/null)" ]; then
  echo "fetch-qtip: $DEST exists and is not empty — refusing to overwrite." >&2
  exit 1
fi
mkdir -p "$DEST"

# `sha256sum` (coreutils) is present on every Linux image; `shasum` is Perl and
# is missing from the runtime image. We prefer the first and fall back to the
# second, which is what exists on the dev Mac.
sha_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}
expect() {  # expect <file> <wanted-sha>
  got=$(sha_of "$1")
  if [ "$got" != "$2" ]; then
    echo "fetch-qtip: sha256 mismatch for $1" >&2
    echo "  expected $2" >&2
    echo "  got      $got" >&2
    echo "  Upstream changed under a pinned commit, or the download is corrupt." >&2
    exit 1
  fi
}

for f in inference.cu inference.h; do
  curl -sSL --fail -o "$DEST/$f" "$RAW/$f"
done
expect "$DEST/inference.cu" "$SHA_CU"
expect "$DEST/inference.h"  "$SHA_H"
echo "fetch-qtip: both files fetched at $COMMIT, sha256 verified."

# ---------------------------------------------------------------------------
# The patch: four DEAD lines. Each was verified to have zero uses in the file
# on 2026-08-20 (the MMA goes through inline PTX asm, not the wmma API), so
# removing them changes no generated code — it only frees the translation unit
# from torch and from libcu++, which NVRTC does not carry.
#
# CHECK_CUDA / CHECK_CONTIGUOUS expand to TORCH_CHECK but are never invoked in
# this file. An unexpanded macro costs nothing, so they are deliberately left
# alone rather than edited.
# ---------------------------------------------------------------------------
DEAD=(
  '#include <cuda/pipeline>'
  '#include <mma.h>'
  '#include <c10/cuda/CUDAStream.h>'
  'using namespace nvcuda;'
)
for line in "${DEAD[@]}"; do
  # Idempotent: deleting an already-absent line is a no-op.
  grep -vxF "$line" "$DEST/inference.cu" > "$DEST/.tmp" && mv "$DEST/.tmp" "$DEST/inference.cu"
done
# Prove the patch, rather than trust the sed.
for line in "${DEAD[@]}"; do
  if grep -qxF "$line" "$DEST/inference.cu"; then
    echo "fetch-qtip: patch failed to remove: $line" >&2
    exit 1
  fi
done
for tok in 'cuda::pipeline' 'wmma' 'nvcuda' 'c10'; do
  n=$(grep -c "$tok" "$DEST/inference.cu" || true)
  if [ "$n" -ne 0 ]; then
    echo "fetch-qtip: '$tok' still present $n time(s) — the file is not the one this script was written against." >&2
    exit 1
  fi
done
echo "fetch-qtip: patch applied and verified (4 dead lines removed, 0 residual tokens)."

# ---------------------------------------------------------------------------
# The device-only extract.
#
# NVRTC carries `cuda_fp16.h` and nothing else of what this file includes —
# no <cstdio>, no <cassert>, no <climits>, no <cuda_runtime.h>. So the kernel
# cannot be handed to NVRTC as it stands. What NVRTC needs is the device half
# alone: the unions, the load helpers, and the kernel template. The host
# launcher `decompress_matvec_ptr` stays out — it calls
# cudaGetDeviceProperties and cudaFuncSetAttribute, and we launch from Rust
# through cudarc anyway.
#
# One transformation, and only one. cudarc resolves kernels BY NAME
# (`Cuda::func`, llvq-cuda/src/gpu.rs:357), and a `__global__ static` template
# has no reachable name. The template therefore becomes `__device__`, and our
# own glue (llvq-cuda/kernels/qtip_glue.cu, which contains no upstream code)
# wraps it in `extern "C" __global__` shims that DO have names. The
# `__launch_bounds__` migrates to those shims.
#
# The extraction is anchored on text, not on line numbers, and it is PROVEN
# afterwards rather than trusted: an extractor that silently mis-cuts would
# produce a kernel that compiles and computes something else.
# ---------------------------------------------------------------------------
python3 - "$DEST" <<'PYEXTRACT'
# `os.path` + `open` only: `python3-minimal` does not carry `ntpath`, which
# `pathlib` imports, and job 6a878b14 died on it. This script therefore depends
# only on what a stripped interpreter still provides.
import os.path, sys
dest = sys.argv[1]
with open(os.path.join(dest, "inference.h")) as f:
    hdr = f.readlines()
with open(os.path.join(dest, "inference.cu")) as f:
    cu = f.readlines()

# (1) the three `ditto` unions — the device code needs them; gpuAssert and the
#     host launcher's declaration do not come along.
i0 = next(i for i, l in enumerate(hdr) if l.startswith("typedef union ditto {"))
i1 = next(i for i, l in enumerate(hdr) if l.startswith("} ditto4;"))
unions = hdr[i0 : i1 + 1]

# (2) from the first #define through the end of the kernel template, stopping
#     at the comment block that introduces the host launcher.
j0 = next(i for i, l in enumerate(cu) if l.rstrip("\n") == "#define BLOCKS_PER_SM 1")
j1 = next(i for i, l in enumerate(cu) if l.rstrip("\n") == "// L: shift register bit-width")
body = "".join(cu[j0:j1])

# (3a) C99 designated initializers. `nvcc` accepts `{.field = expr}` on a union
#      as an extension; NVRTC refuses it below C++20, and raising the standard
#      is not an option — the NVRTC options are shared with OUR kernels, and
#      every published number depends on how those compile. So the two
#      occurrences are rewritten into a declaration plus an assignment, which
#      is the same object in any dialect.
import re
desig = re.compile(r"^(\s*)(\w+)\s+(\w+)\s*=\s*\{\.(\w+)\s*=\s*(.+)\};\s*$", re.M)
body, n_desig = desig.subn(r"\1\2 \3; \3.\4 = \5;", body)

# (3) the one transformation.
before = "__global__ static void\n__launch_bounds__(BLOCK_SIZE, 1)\n"
after = "__device__ static void\n"
if body.count(before) != 1:
    sys.exit(f"fetch-qtip: expected exactly one kernel declaration to rewrite, found {body.count(before)}")
body = body.replace(before, after)

out = (
    "// DERIVED FILE — device-only extract of QTIP's inference.cu, GPL v3.\n"
    "// Produced by ops/fetch-qtip.sh. NOT redistributed by this repository.\n"
    "// See PROVENANCE.txt beside it.\n"
    "#pragma once\n#define QTIP_DEVICE_CUH 1\n\n" + "".join(unions) + "\n" + body
)
with open(os.path.join(dest, "qtip_device.cuh"), "w") as f:
    f.write(out)
PYEXTRACT

# Prove the extract rather than trust it.
CUH="$DEST/qtip_device.cuh"
[ -s "$CUH" ] || { echo "fetch-qtip: qtip_device.cuh was not produced." >&2; exit 1; }
check_count() {  # check_count <pattern> <wanted> <why>
  n=$(grep -c -- "$1" "$CUH" || true)
  if [ "$n" -ne "$2" ]; then
    echo "fetch-qtip: qtip_device.cuh has $n occurrence(s) of '$1', expected $2 ($3)." >&2
    exit 1
  fi
}
check_count '__global__' 0 'the template must have become __device__'
check_count '#include' 0 'NVRTC carries none of the headers this file used'
check_count 'cudaGetDeviceProperties' 0 'the host launcher must stay out'
check_count 'cudaFuncSetAttribute' 0 'the host launcher must stay out'
check_count 'kernel_decompress_matvec(' 1 'exactly one kernel template'
check_count '__device__ static void' 1 'the rewritten declaration'
# PITFALL: the naive pattern '= {.' also counted the '= {};' lines. In basic
# regex the '.' matches any character, so the guard fired on perfectly valid
# empty initializations. A check that raises false positives wears out the
# trust placed in it; this one names the field.
if grep -qE '=[[:space:]]*\{\.[a-zA-Z_]' "$CUH"; then
  echo "fetch-qtip: C99 designated initializers remain, NVRTC refuses them." >&2
  grep -nE '=[[:space:]]*\{\.[a-zA-Z_]' "$CUH" >&2
  exit 1
fi
echo "fetch-qtip: qtip_device.cuh extracted and verified ($(wc -l < "$CUH" | tr -d ' ') lines)."

cat > "$DEST/PROVENANCE.txt" <<EOF
QTIP inference kernel — fetched, patched, NOT redistributed
===========================================================
Fetched     : $(date -u +%Y-%m-%dT%H:%M:%SZ)
Upstream    : https://github.com/Cornell-RelaxML/qtip
Commit      : $COMMIT
Files       : qtip-kernels/src/inference.cu, inference.h
sha256 (pristine, before patch)
  inference.cu $SHA_CU
  inference.h  $SHA_H

Licence: GNU GPL v3. This code is NOT part of the LLVQ repository and is NOT
redistributed by it. It is fetched here to be measured. Running GPL software
to produce a measurement is unrestricted by the licence, which governs
distribution.

Lines removed from inference.cu (each verified to have zero uses upstream, so
the generated device code is unchanged; they are removed because NVRTC carries
neither torch nor libcu++):
$(printf '  %s\n' "${DEAD[@]}")

Derived file: qtip_device.cuh — the device half alone (the three ditto unions,
the load helpers, the kernel template), with the kernel template rewritten from
\`__global__ static\` to \`__device__ static\` so that our own extern "C" shims
can carry a name cudarc can resolve. The host launcher is excluded. This file
is derived from GPL v3 sources and is likewise NOT redistributed.
EOF

echo
echo "fetch-qtip: done. Pass this to the bench:"
echo "  export LLVQ_QTIP_DIR=$(cd "$DEST" && pwd)"
