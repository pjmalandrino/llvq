#!/usr/bin/env bash
# Build a Linux CPU `llvq` archive: the binary alone, no GPU feature. Runs INSIDE ubuntu:22.04, on x86_64 or arm64,
# whichever the host is:
#
#   docker run --rm -v "$PWD":/src -w /src ubuntu:22.04 bash ops/release/build-linux-cpu.sh
#
# Output: dist/llvq-<version>-linux-<arch>-cpu.tar.gz and its .sha256, <arch> = x86_64 or arm64, for
# llvq-runtime/install.sh. The archive needs, on the machine that runs it: that architecture, glibc 2.35 or newer and
# libssl3 (Ubuntu 22.04, Debian 12 and later). No GPU, no driver.
#
# The binary serves the Qwen3 models on the CPU path (llvq_llm::fused_cpu): the Tetra codes read at every matvec on
# the SIMD path the CPU reports at run time (AVX2 + FMA or scalar on x86_64, NEON on arm64). It must run on ANY CPU of
# its architecture, so nothing here may raise the compile-time baseline: no target-cpu, no target-feature. The
# script refuses RUSTFLAGS that do, and on x86_64 starts the binary under qemu-user, whose CPU (qemu 6.2 of Ubuntu
# 22.04) has no AVX at all: a binary built for a newer baseline dies there on its first AVX instruction.
set -euo pipefail
cd "$(dirname "$0")/../.."

case "$(uname -m)" in
  x86_64) ARCH=x86_64 ;;
  aarch64|arm64) ARCH=arm64 ;;
  *) echo "refused: no CPU archive for $(uname -m)" >&2; exit 1 ;;
esac
FLAVOR=linux-$ARCH-cpu
export DEBIAN_FRONTEND=noninteractive

case " ${RUSTFLAGS:-} ${CARGO_ENCODED_RUSTFLAGS:-} " in
  *target-cpu*|*target-feature*) echo "refused: RUSTFLAGS raise the CPU baseline: ${RUSTFLAGS:-}" >&2; exit 1 ;;
esac
for f in .cargo/config.toml .cargo/config "$HOME/.cargo/config.toml" "$HOME/.cargo/config"; do
  if [ -f "$f" ] && grep -q 'target-cpu\|target-feature' "$f"; then
    echo "refused: $f raises the CPU baseline" >&2; exit 1
  fi
done

if ! command -v cargo > /dev/null; then
  apt-get update
  apt-get install -y --no-install-recommends curl ca-certificates build-essential pkg-config libssl-dev git binutils
  if [ "$ARCH" = x86_64 ]; then apt-get install -y --no-install-recommends qemu-user; fi
  # rustup reads rust-toolchain.toml at the root and installs the pinned toolchain.
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain none
  export PATH=$HOME/.cargo/bin:$PATH
fi
# Inside a container the checkout belongs to another user: git refuses it without this, and the commit in VERSION
# would read "unknown".
git config --global --add safe.directory "$PWD" 2> /dev/null || true
if command -v rustup > /dev/null; then rustup toolchain install; fi
cargo --version
rustc --version

# The same profile override as the CUDA archive: thin LTO in one codegen unit is too much for a small builder.
export CARGO_PROFILE_RELEASE_CODEGEN_UNITS=${CARGO_PROFILE_RELEASE_CODEGEN_UNITS:-16}
cargo build --release --locked -p llvq-runtime --bin llvq

VERSION=$(sed -n 's/^version = "\(.*\)"/\1/p' llvq-runtime/Cargo.toml | head -n 1)
COMMIT=$(git rev-parse --short=12 HEAD 2> /dev/null || echo unknown)
OUT=dist/llvq
rm -rf dist
mkdir -p "$OUT/licenses"
cp target/release/llvq "$OUT/llvq"
strip "$OUT/llvq" || true
echo "$FLAVOR" > "$OUT/FLAVOR"
printf 'llvq %s (%s), %s\n' "$VERSION" "$COMMIT" "$FLAVOR" > "$OUT/VERSION"

# No GPU library, and nothing the system does not resolve.
NEEDED=$(readelf -d "$OUT/llvq" | sed -n 's/.*Shared library: \[\(.*\)\]/\1/p')
echo "NEEDED:"; echo "$NEEDED"
if echo "$NEEDED" | grep -E '^(libcu|libnv)'; then echo "refused: a CPU build names a CUDA library" >&2; exit 1; fi
LDD=$(env -u LD_LIBRARY_PATH ldd "$OUT/llvq" || true)
echo "$LDD"
if echo "$LDD" | grep 'not found'; then echo "refused: a library is missing" >&2; exit 1; fi

"$OUT/llvq" -v
if [ "$ARCH" = x86_64 ]; then
  if command -v qemu-x86_64 > /dev/null; then
    qemu-x86_64 --version | head -n 1
    qemu-x86_64 "$OUT/llvq" -v || { echo "refused: the binary does not start on a CPU without AVX" >&2; exit 1; }
  else
    echo "warning: no qemu-x86_64, the start on a CPU without AVX is not checked" >&2
  fi
fi

cp LICENSE-MIT LICENSE-APACHE "$OUT/licenses/"
NAME=llvq-$VERSION-$FLAVOR
tar -czf "dist/$NAME.tar.gz" -C dist llvq
(cd dist && sha256sum "$NAME.tar.gz" > "$NAME.tar.gz.sha256")
ls -l dist
cat "dist/$NAME.tar.gz.sha256"
