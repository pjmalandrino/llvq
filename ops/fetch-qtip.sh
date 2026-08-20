#!/usr/bin/env bash
# Fetch the QTIP 2-bit inference kernel for the `qtip` comparison arm.
#
# WHY THIS EXISTS INSTEAD OF A VENDORED COPY. The AWQ arm ships as
# `llvq-cuda/kernels/awq_gemv.cu`, embedded with `include_str!`, because AWQ is
# MIT. QTIP is **GPL v3**, which this repository (MIT OR Apache-2.0) does not
# redistribute. So the kernel is fetched at job time and never committed. Using
# GPL software to produce a measurement is unrestricted; only distribution is.
#
# The `LLVQ_KERNEL_DIR` override this feeds already exists — see
# `llvq-cuda/src/bin/planesbench.rs:218` (`load_awq_source`), where it is an
# *override* for AWQ. For QTIP it is the only path: there is no embedded
# fallback to silently succeed with.
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

sha_of() { shasum -a 256 "$1" | cut -d' ' -f1; }
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
EOF

echo
echo "fetch-qtip: done. Pass this to the bench:"
echo "  export LLVQ_KERNEL_DIR=$(cd "$DEST" && pwd)"
