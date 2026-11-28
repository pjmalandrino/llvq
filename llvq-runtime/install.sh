#!/bin/sh
# Install llvq from a prebuilt archive, in the manner of Ollama's install.sh.
#
#   curl -fsSL https://github.com/pjmalandrino/llvq-tetra/releases/latest/download/install.sh | sh
#
# What it does:
#   1. reads the system and picks the archive flavor:
#        Linux x86_64 with an NVIDIA card       linux-x86_64-cuda
#        Linux x86_64 without one               linux-x86_64-cpu
#        Linux arm64 (aarch64)                  linux-arm64-cpu
#        macOS on Apple Silicon                 macos-arm64-metal
#        macOS on Intel                         macos-x86_64-cpu
#      LLVQ_FLAVOR names one instead; it must be a flavor of this system;
#   2. for the CUDA flavor, refuses a missing driver, a driver older than LLVQ_MIN_DRIVER, or a card older than Ada
#      (sm_89). When the flavor was not named, such a machine gets the CPU flavor instead, and the reason is printed;
#   3. downloads the archive of release v$LLVQ_VERSION for this flavor, or the one LLVQ_URL names, and checks its
#      sha256 (LLVQ_SHA256, or the file <url>.sha256);
#   4. unpacks it into <prefix>/lib/llvq and links <prefix>/bin/llvq to it.
#
# The prefix is LLVQ_PREFIX, else /usr/local as root, else ~/.local. No root is needed for ~/.local.
# The CUDA archive carries the CUDA run-time libraries the binary loads (lib/, found through an rpath of $ORIGIN/lib).
# The NVIDIA driver is the one thing it cannot carry: it must be installed already.
# A CPU archive carries the binary alone. It reads the Tetra codes at every matvec on the SIMD path the CPU reports
# at run time (AVX2 + FMA or scalar on x86_64, NEON on arm64) and serves the Qwen3 models. Inkling 2 is refused on
# the CPU: it needs CUDA or Metal.
#
# Variables:
#   LLVQ_VERSION     the release to install, default the one this script shipped with.
#   LLVQ_FLAVOR      the archive flavor, instead of the one this script picks.
#   LLVQ_URL         another archive (.tar.gz), https:// or file://, instead of the release's.
#   LLVQ_SHA256      its sha256. When unset, read from "$LLVQ_URL.sha256".
#   LLVQ_PREFIX      where to install.
#   LLVQ_MIN_DRIVER  oldest NVIDIA driver accepted, default 550 (the archive's CUDA 12.4 run-time and its PTX need it).
#   LLVQ_SKIP_GPU_CHECK=1  install the CUDA flavor on Linux x86_64 without checking the driver or the card (a build
#                    machine, a container image built without a GPU). The binary still needs the driver to start.
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
  Linux/x86_64|Linux/amd64) SYSTEM=linux-x86_64; FLAVORS="linux-x86_64-cuda linux-x86_64-cpu" ;;
  Linux/aarch64|Linux/arm64) SYSTEM=linux-arm64; FLAVORS="linux-arm64-cpu" ;;
  Darwin/arm64) SYSTEM=macos-arm64; FLAVORS="macos-arm64-metal" ;;
  Darwin/x86_64) SYSTEM=macos-x86_64; FLAVORS="macos-x86_64-cpu" ;;
  Linux/*) die "Linux on $ARCH: archives exist for x86_64 and arm64 only; build from source (llvq-runtime/README.md)" ;;
  Darwin/*) die "macOS on $ARCH: archives exist for arm64 and x86_64 only" ;;
  *) die "$OS is not served: Linux x86_64 or arm64, or macOS; build from source (llvq-runtime/README.md)" ;;
esac

# --- the driver and the card, for the CUDA flavor --------------------------------------------------------------------
# Returns 1 with WHY set when this machine cannot run the CUDA archive.
MIN_DRIVER=${LLVQ_MIN_DRIVER:-550}
cuda_ok() {
  WHY=
  if ! command -v nvidia-smi > /dev/null 2>&1; then
    WHY="no nvidia-smi: the NVIDIA driver is not installed. Install driver $MIN_DRIVER or newer from your \
distribution (Ubuntu: sudo ubuntu-drivers install), reboot, then run this again. The CUDA toolkit is NOT needed."
    return 1
  fi
  if ! ROW=$(nvidia-smi --query-gpu=driver_version,compute_cap,name --format=csv,noheader 2> /dev/null | head -n 1); then
    WHY="nvidia-smi failed: the driver is installed but does not answer (a reboot after installing it?)"
    return 1
  fi
  if [ -z "$ROW" ]; then
    WHY="nvidia-smi lists no GPU"
    return 1
  fi
  DRIVER=$(printf '%s' "$ROW" | cut -d, -f1 | tr -d ' ')
  CAP=$(printf '%s' "$ROW" | cut -d, -f2 | tr -d ' ')
  NAME=$(printf '%s' "$ROW" | cut -d, -f3- | sed 's/^ *//')
  MAJOR=${DRIVER%%.*}
  case "$MAJOR" in
    ''|*[!0-9]*) WHY="cannot read the driver version from nvidia-smi: '$DRIVER'"; return 1 ;;
  esac
  if [ "$MAJOR" -lt "$MIN_DRIVER" ]; then
    WHY="NVIDIA driver $DRIVER is older than $MIN_DRIVER, which the CUDA archive needs. Update the driver, then run \
this again."
    return 1
  fi
  CAP_MAJOR=${CAP%%.*}
  CAP_MINOR=${CAP#*.}
  case "$CAP_MAJOR$CAP_MINOR" in
    ''|*[!0-9]*) WHY="cannot read the compute capability from nvidia-smi: '$CAP'"; return 1 ;;
  esac
  if [ "$CAP_MAJOR" -lt 8 ] || { [ "$CAP_MAJOR" -eq 8 ] && [ "$CAP_MINOR" -lt 9 ]; }; then
    WHY="$NAME is compute capability $CAP. The CUDA archive runs on 8.9 and newer (RTX 40xx, L4, L40S, H100, \
RTX 50xx): its kernels are built for sm_89. Build from source with CUDA_COMPUTE_CAP set for this card."
    return 1
  fi
  say "GPU: $NAME, compute capability $CAP, driver $DRIVER"
}

if [ -n "${LLVQ_FLAVOR:-}" ]; then
  FLAVOR=$LLVQ_FLAVOR
  case " $FLAVORS " in
    *" $FLAVOR "*) ;;
    *) die "LLVQ_FLAVOR=$FLAVOR is not a flavor of $OS $ARCH, which has: $FLAVORS" ;;
  esac
  if [ "$FLAVOR" = linux-x86_64-cuda ] && [ "${LLVQ_SKIP_GPU_CHECK:-0}" != 1 ]; then
    cuda_ok || die "$WHY"
  fi
elif [ "$SYSTEM" = linux-x86_64 ]; then
  if [ "${LLVQ_SKIP_GPU_CHECK:-0}" = 1 ]; then
    FLAVOR=linux-x86_64-cuda
  elif cuda_ok; then
    FLAVOR=linux-x86_64-cuda
  else
    say "$WHY"
    say "installing the CPU flavor instead. For the CUDA one, fix the above and run this again; \
LLVQ_FLAVOR=linux-x86_64-cuda refuses with the reason instead of falling back."
    FLAVOR=linux-x86_64-cpu
  fi
else
  FLAVOR=${FLAVORS%% *}
fi
say "system: $OS $ARCH, archive flavor $FLAVOR"

# --- the archive ---------------------------------------------------------------------------------------------------
VERSION=${LLVQ_VERSION:-0.1.3}
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
  [ "$GOTF" = "$FLAVOR" ] || die "the archive is built for $GOTF, this machine was given $FLAVOR (LLVQ_FLAVOR picks another)"
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

# The new binary must start BEFORE it replaces anything: a failed check removes $DEST.new and leaves $DEST as it was.
if [ -e "$DEST" ]; then KEPT="the previous install in $DEST is untouched"; else KEPT="nothing was installed"; fi
refuse() { rm -rf "$DEST.new"; die "$1; $KEPT"; }
if [ "$OS" = Linux ] && command -v ldd > /dev/null 2>&1; then
  MISSING=$(ldd "$DEST.new/llvq" 2> /dev/null | grep 'not found' || true)
  if [ -n "$MISSING" ]; then
    # glibc says "version `GLIBC_2.35' not found"; a missing library says "libssl.so.3 => not found".
    OLDLIBC=$(printf '%s\n' "$MISSING" | grep 'version .GLIBC_' || true)
    NOLIB=$(printf '%s\n' "$MISSING" | grep -v 'version .GLIBC_' || true)
    if [ -n "$OLDLIBC" ]; then
      say "this system's glibc is older than the archive needs (glibc 2.35 or newer: Ubuntu 22.04, Debian 12 and later):"
      printf '%s\n' "$OLDLIBC" >&2
    fi
    if [ -n "$NOLIB" ]; then
      case "$FLAVOR" in
        *-cuda) say "libraries not found (libcuda.so.1 comes with the NVIDIA driver):" ;;
        *) say "libraries not found (libssl.so.3 and libcrypto.so.3: the libssl3 package):" ;;
      esac
      printf '%s\n' "$NOLIB" >&2
    fi
    refuse "the new binary cannot load its libraries"
  fi
fi
if [ "$FLAVOR" != linux-x86_64-cuda ] || [ "${LLVQ_SKIP_GPU_CHECK:-0}" != 1 ]; then
  "$DEST.new/llvq" -v || refuse "the new binary does not start"
fi

rm -rf "$DEST.old"
if [ -e "$DEST" ]; then mv "$DEST" "$DEST.old"; fi
mv "$DEST.new" "$DEST"
rm -rf "$DEST.old"
ln -sf "$DEST/llvq" "$PREFIX/bin/llvq"
say "installed $DEST, linked $PREFIX/bin/llvq"
case ":$PATH:" in
  *":$PREFIX/bin:"*) ;;
  *) say "add $PREFIX/bin to PATH, e.g. echo 'export PATH=\"$PREFIX/bin:\$PATH\"' >> ~/.profile" ;;
esac
# The mark of docs/brand/LOGO.md, its outer chevron in terracotta on a colour terminal.
if [ -t 2 ] && [ -z "${NO_COLOR:-}" ] && [ "${TERM:-}" != dumb ]; then A=$(printf '\033[38;2;180;80;45m'); R=$(printf '\033[0m'); else A=; R=; fi
printf '%s\n' "   $A/$R$A\\$R" "  $A/$R/\\$A\\$R" " $A/$R//\\\\$A\\$R    tetra" "$A/$R///\\\\\\$A\\$R" >&2
case "$FLAVOR" in
  *-cpu) say "done. Next: llvq run qwen3:4b (1.4 GB to download). Inkling 2 needs CUDA or Metal and is refused on the CPU" ;;
  *) say "done. Next: llvq run inkling2:mini (8 GB to download), or llvq run qwen3:4b (1.4 GB)" ;;
esac
