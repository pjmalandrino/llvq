#!/usr/bin/env bash
# Build the Linux CUDA `llvq` archive: the binary, the CUDA run-time libraries it loads beside it in lib/, found
# through an rpath of $ORIGIN/lib. Runs INSIDE nvidia/cuda:12.4.1-devel-ubuntu22.04, the base of ops/Dockerfile.cuda,
# which has built `llvq --features cuda` for the card jobs since 2026-11:
#
#   docker run --rm -v "$PWD":/src -w /src nvidia/cuda:12.4.1-devel-ubuntu22.04 bash ops/release/build-linux-cuda.sh
#
# Output: dist/llvq-<version>-linux-x86_64-cuda.tar.gz and its .sha256, for llvq-runtime/install.sh.
# No GPU is needed to build. The archive needs, on the machine that runs it: x86_64, glibc 2.35 or newer and libssl3
# (Ubuntu 22.04, Debian 12 and later), an NVIDIA driver 550 or newer, a card of compute capability 8.9 or newer.
#
# Why the card floor: candle's own kernels (candle-kernels) are compiled here by nvcc for CUDA_COMPUTE_CAP, as PTX
# for compute_89 and a library for sm_89; a driver JIT-compiles that PTX for 8.9 and newer, never for older cards.
# Our kernels are compiled at run time by NVRTC (llvq-cuda/src/gpu.rs, compute_89 unless LLVQ_NVRTC_ARCH).
# Why the driver floor: the PTX of CUDA 12.4 (ISA 8.4) is JIT-compiled by the driver, which must know that ISA:
# R550 is the driver of CUDA 12.4.
set -euo pipefail
cd "$(dirname "$0")/../.."

CAP=${CUDA_COMPUTE_CAP:-89}
export CUDA_COMPUTE_CAP=$CAP
export DEBIAN_FRONTEND=noninteractive
FLAVOR=linux-x86_64-cuda

command -v nvcc > /dev/null || export PATH=/usr/local/cuda/bin:$PATH
command -v nvcc > /dev/null || { echo "refused: no nvcc; run this in nvidia/cuda:12.4.1-devel-ubuntu22.04" >&2; exit 1; }
NVCC=$(nvcc --version | sed -n 's/.*release \([0-9.]*\),.*/\1/p')
echo "nvcc $NVCC, compute capability $CAP"

if ! command -v cargo > /dev/null; then
  apt-get update
  apt-get install -y --no-install-recommends curl ca-certificates build-essential pkg-config libssl-dev git
  # rustup reads rust-toolchain.toml at the root and installs the pinned toolchain on first use.
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain none
  export PATH=$HOME/.cargo/bin:$PATH
fi
# Inside a container the checkout belongs to another user: git refuses it without this, and the commit in VERSION
# would read "unknown".
git config --global --add safe.directory "$PWD" 2> /dev/null || true
# The toolchain of rust-toolchain.toml, installed explicitly: recent rustup no longer installs it on first use.
if command -v rustup > /dev/null; then rustup toolchain install; fi
cargo --version
rustc --version

# DT_RPATH, not DT_RUNPATH (--disable-new-dtags): an rpath is searched before LD_LIBRARY_PATH, so a CUDA toolkit
# the user installed (12.x or 13.x) cannot shadow the libraries we ship, and it also covers the libraries those
# libraries load, libnvrtc-builtins among them.
export RUSTFLAGS="${RUSTFLAGS:-} -C link-arg=-Wl,-rpath,\$ORIGIN/lib -C link-arg=-Wl,--disable-new-dtags"
# The same profile overrides as ops/Dockerfile.cuda: thin LTO in one codegen unit is too much for a small builder.
export CARGO_PROFILE_RELEASE_CODEGEN_UNITS=${CARGO_PROFILE_RELEASE_CODEGEN_UNITS:-16}
cargo build --release --locked -p llvq-runtime --features cuda --bin llvq

VERSION=$(sed -n 's/^version = "\(.*\)"/\1/p' llvq-runtime/Cargo.toml | head -n 1)
COMMIT=$(git rev-parse --short=12 HEAD 2> /dev/null || echo unknown)
OUT=dist/llvq
rm -rf dist
mkdir -p "$OUT/lib" "$OUT/licenses"
cp target/release/llvq "$OUT/llvq"
strip "$OUT/llvq" || true
echo "$FLAVOR" > "$OUT/FLAVOR"
printf 'llvq %s (%s), %s, CUDA %s, compute capability %s\n' "$VERSION" "$COMMIT" "$FLAVOR" "$NVCC" "$CAP" > "$OUT/VERSION"

# Every CUDA library the binary names, from the toolkit, as real files under their sonames. libcuda.so.1 is the
# driver's and is never shipped.
CUDA_LIB=/usr/local/cuda/lib64
NEEDED=$(readelf -d "$OUT/llvq" | sed -n 's/.*Shared library: \[\(.*\)\]/\1/p')
echo "NEEDED:"; echo "$NEEDED"
for so in $NEEDED; do
  case "$so" in
    libcuda.so.*) ;;
    libcudart.so.*|libcublas.so.*|libcublasLt.so.*|libcurand.so.*|libnvrtc.so.*)
      cp -L "$CUDA_LIB/$so" "$OUT/lib/$so" ;;
    libnv*|libcu*) echo "refused: $so is a CUDA library this script does not know; add it to the list" >&2; exit 1 ;;
  esac
done
# NVRTC loads its builtins with dlopen, so they are not in NEEDED.
cp -L "$CUDA_LIB"/libnvrtc-builtins.so.12* "$OUT/lib/"
for want in libcublas.so.12 libcublasLt.so.12 libnvrtc.so.12; do
  [ -f "$OUT/lib/$want" ] || { echo "refused: $want missing from lib/" >&2; exit 1; }
done

# The rpath is there, and every CUDA library resolves into lib/ (libcuda.so.1 cannot resolve in a container
# without the driver).
readelf -d "$OUT/llvq" | grep -q 'RPATH.*\$ORIGIN/lib' || { echo "refused: no RPATH \$ORIGIN/lib" >&2; exit 1; }
LDD=$(env -u LD_LIBRARY_PATH ldd "$OUT/llvq" || true)
echo "$LDD"
for f in "$OUT"/lib/*; do
  so=$(basename "$f")
  case "$so" in libnvrtc-builtins*) continue ;; esac
  echo "$LDD" | grep -q "$so => .*/dist/llvq/lib/$so" || { echo "refused: $so does not resolve into lib/" >&2; exit 1; }
done
echo "$LDD" | grep 'not found' | grep -v 'libcuda.so.1' && { echo "refused: a library is missing" >&2; exit 1; }

cp LICENSE-MIT LICENSE-APACHE "$OUT/licenses/"
cat > "$OUT/licenses/NVIDIA-CUDA.txt" << EOF
The files in lib/ are NVIDIA CUDA run-time libraries from the CUDA Toolkit $NVCC, redistributed unmodified
under the NVIDIA CUDA Toolkit End User License Agreement, Attachment A
(https://docs.nvidia.com/cuda/eula/index.html). They are for use by llvq only.
EOF

NAME=llvq-$VERSION-$FLAVOR
tar -czf "dist/$NAME.tar.gz" -C dist llvq
(cd dist && sha256sum "$NAME.tar.gz" > "$NAME.tar.gz.sha256")
ls -l dist
cat "dist/$NAME.tar.gz.sha256"
