#!/usr/bin/env bash
# Builds a self-contained Mac bundle of the Metal engine: one prebuilt binary, the configs it reads, a launcher and a
# README. It runs outside the repository with no cargo and no network (the tokenizer is inside the sealed file); only
# `bench --long` reads WikiText-2, from the Hugging Face cache or the network the first time.
#
#   ops/metal-package.sh            # writes dist/llvq-metal-<date>-<commit>-macos-arm64.tar.gz
#
# The sealed file is not inside (1.4 GB): the launcher looks for it at $SEALED, then at the path below.
set -euo pipefail
cd "$(dirname "$0")/.."
[ "$(uname -s)-$(uname -m)" = "Darwin-arm64" ] || { echo "Apple silicon only"; exit 1; }
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "uncommitted changes: commit first, the bundle names its commit"; exit 1; }
COMMIT=$(git rev-parse --short=8 HEAD)
NAME="llvq-metal-$(date +%Y-%m-%d)-$COMMIT-macos-arm64"
OUT="dist/$NAME"
cargo build --release -p llvq-llm --features metal,fast-linalg --bin metalrun
rm -rf "$OUT" "$OUT.tar.gz"
mkdir -p "$OUT/bin" "$OUT/configs"
cp target/release/metalrun "$OUT/bin/"
cp configs/qwen3-4b-tetra-e4.json configs/qwen3-4b-tetra-e4-metal-grouped-NOT-SERVED.json \
   configs/qwen3-4b-tetra-e4-metal-kernel-NOT-SERVED.json "$OUT/configs/"
cp ops/metal-package/llvq-metal "$OUT/"
cp ops/metal-package/README.md "$OUT/"
chmod +x "$OUT/llvq-metal" "$OUT/bin/metalrun"
{
  echo "commit $(git rev-parse HEAD) on $(git rev-parse --abbrev-ref HEAD)"
  echo "built  $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(sysctl -n machdep.cpu.brand_string), $(sw_vers -productName) $(sw_vers -productVersion)"
  (cd "$OUT" && shasum -a 256 bin/metalrun configs/*.json llvq-metal README.md)
} > "$OUT/MANIFEST.txt"
# Only system libraries: a bundle that needs Homebrew on the other side is not a bundle.
if otool -L "$OUT/bin/metalrun" | tail -n +2 | grep -v -e '^\s*/System/' -e '^\s*/usr/lib/'; then
  echo "metalrun links a library outside /System and /usr/lib"; exit 1
fi
tar -C dist -czf "$OUT.tar.gz" "$NAME"
echo "bundle  $OUT.tar.gz ($(du -h "$OUT.tar.gz" | cut -f1)), sha256 $(shasum -a 256 "$OUT.tar.gz" | cut -d' ' -f1)"
echo "try it  tar -xzf $OUT.tar.gz -C ~ && ~/$NAME/llvq-metal chat"
