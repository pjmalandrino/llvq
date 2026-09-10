#!/usr/bin/env bash
# Type-check the half of this workspace the Mac cannot compile.
#
#   ops/check-cuda.sh                 # cargo check, all targets
#   ops/check-cuda.sh clippy          # clippy instead
#   ops/check-cuda.sh check -p llvq-cuda --features …   # anything else
#
# First run builds the image and the dependency graph (~5 min). Every run
# after that is INCREMENTAL: 6 s wall for a one-line edit, measured 2026-09-10.
# Proof it is not decorative: a deliberate `u32 + &str` in `fused_cuda.rs` is
# invisible to `cargo clippy --all-targets` on the host and caught here.
#
# It is a TYPE check and nothing more. The two failures of 2026-09-10 were a
# refusal at load and a symbol the driver did not have; neither is a type
# error. Portable tests are what catch those.
set -euo pipefail
cd "$(dirname "$0")/.."
docker image inspect llvq-check >/dev/null 2>&1 \
  || docker build --platform linux/arm64 -t llvq-check -f ops/Dockerfile.check .
exec docker run --rm --platform linux/arm64 \
  -v "$PWD":/src \
  -v llvq-ctarget:/ctarget \
  -v llvq-cargo-registry:/root/.cargo/registry \
  llvq-check cargo "${@:-check}" ${@:+} \
  $([ $# -eq 0 ] && echo "--locked -p llvq-llm --features cuda --all-targets")
