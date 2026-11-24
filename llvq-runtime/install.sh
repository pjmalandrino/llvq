#!/bin/sh
# Install llvq from a prebuilt archive, in the manner of Ollama's install.sh.
#
#   curl -fsSL https://github.com/pjmalandrino/llvq-tetra/releases/latest/download/install.sh | sh
#
# What it does:
#   1. reads the system: Linux x86_64 with an NVIDIA card, or macOS on Apple Silicon;
#   2. on Linux, refuses a missing driver, a driver older than LLVQ_MIN_DRIVER, or a card older than Ada (sm_89);
#   3. downloads the archive of release v$LLVQ_VERSION for this system, or the one LLVQ_URL names, and checks its
#      sha256 (LLVQ_SHA256, or the file <url>.sha256);
#   4. unpacks it into <prefix>/lib/llvq and links <prefix>/bin/llvq to it.
#
# The prefix is LLVQ_PREFIX, else /usr/local as root, else ~/.local. No root is needed for ~/.local.
# The archive carries the CUDA run-time libraries the binary loads (lib/, found through an rpath of $ORIGIN/lib).
# The NVIDIA driver is the one thing it cannot carry: it must be installed already.
#
# Variables:
#   LLVQ_VERSION     the release to install, default the one this script shipped with.
#   LLVQ_URL         another archive (.tar.gz), https:// or file://, instead of the release's.
#   LLVQ_SHA256      its sha256. When unset, read from "$LLVQ_URL.sha256".
#   LLVQ_PREFIX      where to install.
#   LLVQ_MIN_DRIVER  oldest NVIDIA driver accepted, default 550 (the archive's CUDA 12.4 run-time and its PTX need it).
#   LLVQ_SKIP_GPU_CHECK=1  install on Linux without checking the driver or the card (a build machine, a container
#                    image built without a GPU). The binary still needs the driver to start.
set -eu

say() { printf 'llvq install: %s\n' "$*" >&2; }
die() { printf 'llvq install: %s\n' "$*" >&2; exit 1; }

need() { command -v "$1" > /dev/null 2>&1 || die "this script needs $1"; }

need uname
need tar
need mkdir
need ln

OS=$(uname -s)
ARCH=$(uname -m)
case "$OS/$ARCH" in
  Linux/x86_64) FLAVOR=linux-x86_64-cuda ;;
  Darwin/arm64) FLAVOR=macos-arm64-metal ;;
  Linux/*) die "Linux on $ARCH: only x86_64 has an archive" ;;
  Darwin/*) die "macOS on $ARCH: only Apple Silicon (arm64) has an archive" ;;
  *) die "$OS is not served: Linux x86_64 with an NVIDIA card, or macOS on Apple Silicon" ;;
esac
say "system: $OS $ARCH, archive flavor $FLAVOR"

# --- the driver and the card, Linux only ---------------------------------------------------------------------------
MIN_DRIVER=${LLVQ_MIN_DRIVER:-550}
if [ "$FLAVOR" = linux-x86_64-cuda ] && [ "${LLVQ_SKIP_GPU_CHECK:-0}" != 1 ]; then
  if ! command -v nvidia-smi > /dev/null 2>&1; then
    die "no nvidia-smi: the NVIDIA driver is not installed. Install driver $MIN_DRIVER or newer from your distribution \
(Ubuntu: sudo ubuntu-drivers install), reboot, then run this again. The CUDA toolkit is NOT needed."
  fi
  ROW=$(nvidia-smi --query-gpu=driver_version,compute_cap,name --format=csv,noheader 2>/dev/null | head -n 1) ||
    die "nvidia-smi failed: the driver is installed but does not answer (a reboot after installing it?)"
  [ -n "$ROW" ] || die "nvidia-smi lists no GPU"
  DRIVER=$(printf '%s' "$ROW" | cut -d, -f1 | tr -d ' ')
  CAP=$(printf '%s' "$ROW" | cut -d, -f2 | tr -d ' ')
  NAME=$(printf '%s' "$ROW" | cut -d, -f3- | sed 's/^ *//')
  MAJOR=${DRIVER%%.*}
  case "$MAJOR" in
    ''|*[!0-9]*) die "cannot read the driver version from nvidia-smi: '$DRIVER'" ;;
  esac
  [ "$MAJOR" -ge "$MIN_DRIVER" ] ||
    die "NVIDIA driver $DRIVER is older than $MIN_DRIVER, which this archive needs. Update the driver, then run this again."
  CAP_MAJOR=${CAP%%.*}
  CAP_MINOR=${CAP#*.}
  case "$CAP_MAJOR$CAP_MINOR" in
    ''|*[!0-9]*) die "cannot read the compute capability from nvidia-smi: '$CAP'" ;;
  esac
  if [ "$CAP_MAJOR" -lt 8 ] || { [ "$CAP_MAJOR" -eq 8 ] && [ "$CAP_MINOR" -lt 9 ]; }; then
    die "$NAME is compute capability $CAP. This archive runs on 8.9 and newer (RTX 40xx, L4, L40S, H100, RTX 50xx): \
its kernels are built for sm_89. Build from source with CUDA_COMPUTE_CAP set for this card."
  fi
  say "GPU: $NAME, compute capability $CAP, driver $DRIVER"
fi

# --- the archive ---------------------------------------------------------------------------------------------------
VERSION=${LLVQ_VERSION:-0.1.1}
URL=${LLVQ_URL:-https://github.com/pjmalandrino/llvq-tetra/releases/download/v$VERSION/llvq-$VERSION-$FLAVOR.tar.gz}
need curl
if command -v sha256sum > /dev/null 2>&1; then
  SUM="sha256sum"
elif command -v shasum > /dev/null 2>&1; then
  SUM="shasum -a 256"
else
  die "this script needs sha256sum or shasum"
fi

TMP=$(mktemp -d 2> /dev/null || mktemp -d -t llvq)
trap 'rm -rf "$TMP"' EXIT INT TERM
say "downloading $URL"
curl -fL --proto '=https,file' --progress-bar -o "$TMP/llvq.tar.gz" "$URL" || die "download failed: $URL"

WANT=${LLVQ_SHA256:-}
if [ -z "$WANT" ]; then
  curl -fsSL --proto '=https,file' -o "$TMP/llvq.tar.gz.sha256" "$URL.sha256" ||
    die "no LLVQ_SHA256 and no $URL.sha256: refusing an archive whose digest is unknown"
  WANT=$(cut -d' ' -f1 < "$TMP/llvq.tar.gz.sha256")
fi
GOT=$($SUM "$TMP/llvq.tar.gz" | cut -d' ' -f1)
[ "$GOT" = "$WANT" ] || die "sha256 mismatch: the archive is $GOT, expected $WANT. Nothing was installed."
say "sha256 $GOT"

mkdir -p "$TMP/x"
tar -xzf "$TMP/llvq.tar.gz" -C "$TMP/x"
[ -f "$TMP/x/llvq/llvq" ] || die "the archive has no llvq/llvq"
if [ -f "$TMP/x/llvq/FLAVOR" ]; then
  GOTF=$(cat "$TMP/x/llvq/FLAVOR")
  [ "$GOTF" = "$FLAVOR" ] || die "the archive is built for $GOTF, this machine needs $FLAVOR"
fi

# --- install -------------------------------------------------------------------------------------------------------
if [ -n "${LLVQ_PREFIX:-}" ]; then
  PREFIX=$LLVQ_PREFIX
elif [ "$(id -u)" = 0 ]; then
  PREFIX=/usr/local
else
  PREFIX=$HOME/.local
fi
DEST=$PREFIX/lib/llvq
mkdir -p "$PREFIX/bin" "$PREFIX/lib" || die "cannot write under $PREFIX: set LLVQ_PREFIX, or run as root"
rm -rf "$DEST.new"
cp -R "$TMP/x/llvq" "$DEST.new"
chmod 755 "$DEST.new/llvq"
rm -rf "$DEST.old"
if [ -e "$DEST" ]; then mv "$DEST" "$DEST.old"; fi
mv "$DEST.new" "$DEST"
rm -rf "$DEST.old"
ln -sf "$DEST/llvq" "$PREFIX/bin/llvq"
say "installed $DEST, linked $PREFIX/bin/llvq"

if [ "$FLAVOR" = linux-x86_64-cuda ] && command -v ldd > /dev/null 2>&1; then
  MISSING=$(ldd "$DEST/llvq" 2>/dev/null | grep 'not found' || true)
  if [ -n "$MISSING" ]; then
    say "libraries not found (libcuda.so.1 comes with the NVIDIA driver):"
    printf '%s\n' "$MISSING" >&2
    exit 1
  fi
fi
if [ "${LLVQ_SKIP_GPU_CHECK:-0}" != 1 ] || [ "$FLAVOR" != linux-x86_64-cuda ]; then
  "$DEST/llvq" -v || die "the installed binary does not start"
fi
case ":$PATH:" in
  *":$PREFIX/bin:"*) ;;
  *) say "add $PREFIX/bin to PATH, e.g. echo 'export PATH=\"$PREFIX/bin:\$PATH\"' >> ~/.profile" ;;
esac
# The mark of docs/brand/LOGO.md, its outer chevron in terracotta on a colour terminal.
if [ -t 2 ] && [ -z "${NO_COLOR:-}" ] && [ "${TERM:-}" != dumb ]; then A=$(printf '\033[38;2;180;80;45m'); R=$(printf '\033[0m'); else A=; R=; fi
printf '%s\n' "   $A/$R$A\\$R" "  $A/$R/\\$A\\$R" " $A/$R//\\\\$A\\$R    tetra" "$A/$R///\\\\\\$A\\$R" >&2
say "done. Next: llvq run inkling2:mini (8 GB to download), or llvq run qwen3:4b (1.4 GB)"
