#!/usr/bin/env bash
# Run the host-side tests of llvq-cuda (its library unit tests and tests/*) on a Linux machine with no GPU and no
# CUDA toolkit, as CI does:
#
#   ops/run-cuda-host-tests.sh test --locked -p llvq-cuda
#   ops/run-cuda-host-tests.sh test --locked --profile ci -p llvq-cuda
#
# What it proves: the tests ran, on the host. The pinned SHA-256 digests of the served kernel units were computed on
# the kernel text in the tree, the source-reading tests read the Rust in the tree, and the integration tests compiled
# the .cu text as host C++ with clang++ and compared it with the Rust reference. What it does not prove: nothing ran
# on a GPU and no kernel text met NVRTC. No test of the package reaches the driver (2026-11-28, every target, dev and
# ci profiles), so nothing is filtered out here; a test that would reach it dies on a stand-in, as described below.
#
# How it links. cudarc is built with `dynamic-linking`, so every binary of the package asks the linker for -lcuda,
# -lnvrtc, -lcublas and -lcublasLt. This script writes stand-ins for those four libraries in a directory of its own
# and points the linker (RUSTFLAGS -L) and the loader (LD_LIBRARY_PATH) at them. The stand-ins export the symbols
# the binaries reference: the script links against empty libraries, reads the undefined symbols from the linker's
# errors, repeats until the package links, then adds those of binaries cargo did not relink, read from their dynamic
# symbol tables (58 symbols on 2026-11-28, none in libcublasLt). Every exported function writes its own name to
# stderr and aborts, so a test that reaches the driver fails loudly and never reads a made-up answer. Nothing in the
# release build changes: the stand-ins exist only in that directory and only for this cargo invocation. The
# type-checking counterpart is ops/check-cuda-types.sh.
set -euo pipefail
case "$(uname -s)" in
  Linux) ;;
  *) echo "refused: this script is for Linux; on a Mac, cargo test -p llvq-cuda runs these tests without cudarc" >&2; exit 1 ;;
esac
command -v cc > /dev/null || { echo "refused: no C compiler (cc) to build the stand-in libraries" >&2; exit 1; }
command -v clang++ > /dev/null || { echo "refused: the integration tests compile the kernel text with clang++, which is not on PATH" >&2; exit 1; }
REPO=$(cd "$(dirname "$0")/.." && pwd)
W=${LLVQ_CUDA_HOST_DIR:-${TMPDIR:-/tmp}/llvq-cuda-host}
LIBS="cuda nvrtc cublas cublasLt"
mkdir -p "$W/lib" "$W/src"
# attn_dec_matches_rust builds the kernel text under -fsanitize=undefined, whose runtime Ubuntu ships apart from
# clang (libclang-rt-18-dev on 24.04). Without it that test fails at link time, so say it once here.
echo 'int main() { return 0; }' > "$W/src/ubsan.cpp"
clang++ -fsanitize=undefined -o "$W/src/ubsan" "$W/src/ubsan.cpp" 2> /dev/null \
  || { echo "refused: clang++ cannot link -fsanitize=undefined (on Ubuntu, install libclang-rt-<version>-dev)" >&2; exit 1; }
: > "$W/symbols"
# The library a symbol belongs to, from its prefix; anything else is not a CUDA symbol and stops the script.
lib_of() {
  case "$1" in
    cublasLt*) echo cublasLt ;;
    cublas*) echo cublas ;;
    nvrtc*) echo nvrtc ;;
    cu[A-Z]*) echo cuda ;;
    *) return 1 ;;
  esac
}
build_stubs() {
  for l in $LIBS; do
    {
      echo '#include <stdlib.h>'
      echo '#include <unistd.h>'
      { grep -x "$l .*" "$W/symbols" || true; } | while read -r _ s; do
        m="stand-in lib$l: $s was called; this test needs a GPU"
        printf 'void %s(void) { static const char m[] = "%s\\n"; write(2, m, sizeof m - 1); abort(); }\n' "$s" "$m"
      done
    } > "$W/src/lib$l.c"
    cc -shared -fPIC -o "$W/lib/lib$l.so" "$W/src/lib$l.c"
  done
}
build_stubs
export CUDARC_CUDA_VERSION=12040
export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }-L $W/lib -C link-arg=-Wl,--error-limit=0"
export LD_LIBRARY_PATH="$W/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$REPO"
# Link until nothing is missing. cargo test stops at the first binary that fails to link, so each round may add the
# symbols of one more binary; a round that adds none means the failure is something else, and its log is shown.
# Only the arguments before a `--` reach the build.
build_args=()
for a in "$@"; do [ "$a" = "--" ] && break; build_args+=("$a"); done
round=0
until CARGO_TERM_COLOR=never cargo "${build_args[@]}" --no-run > "$W/link.log" 2>&1; do
  round=$((round + 1))
  new=$(grep -oE "undefined (symbol: |reference to \`)[A-Za-z_][A-Za-z0-9_]*" "$W/link.log" \
        | sed -E 's/.*(symbol: |`)//' | sort -u || true)
  added=0
  for s in $new; do
    l=$(lib_of "$s") || { cat "$W/link.log" >&2; echo "refused: $s is undefined and is not a CUDA symbol" >&2; exit 1; }
    grep -qx "$l $s" "$W/symbols" || { echo "$l $s" >> "$W/symbols"; added=$((added + 1)); }
  done
  if [ "$added" -eq 0 ] || [ "$round" -gt 50 ]; then cat "$W/link.log" >&2; exit 1; fi
  build_stubs
done
# A binary cargo did not relink in this run (a warm target directory, a CI cache) still needs its symbols at load
# time, and the round above never saw them: read them from the dynamic symbol table of every executable cargo left.
T=$(cargo metadata --locked --format-version 1 --no-deps | grep -o '"target_directory":"[^"]*"' | cut -d'"' -f4)
added=0
for s in $(find "$T" -mindepth 1 -maxdepth 3 -path "$T/*/build" -prune -o -type f -perm -u+x -print 2> /dev/null \
           | xargs -r nm -D --undefined-only 2> /dev/null | awk '{print $NF}' | sed 's/@.*//' | sort -u); do
  l=$(lib_of "$s") || continue
  grep -qx "$l $s" "$W/symbols" || { echo "$l $s" >> "$W/symbols"; added=$((added + 1)); }
done
[ "$added" -eq 0 ] || build_stubs
echo "stand-ins: $(wc -l < "$W/symbols") symbols in $W/lib ($(for l in $LIBS; do printf 'lib%s %s, ' "$l" "$(grep -c "^$l " "$W/symbols" || true)"; done | sed 's/, $//'))" >&2
exec cargo "$@"
