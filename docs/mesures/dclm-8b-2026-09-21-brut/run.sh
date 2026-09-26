#!/usr/bin/env bash
set -euo pipefail
REPO=$HOME/Documents/Pro/workspace/poc/llvq; OUT=$HOME/q8b-dclm-2026-09-21
caffeinate -i -w $$ &                      # pmset sleep = 1 min: no idle sleep while this runs
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* inherited' >&2; exit 1; fi
cd "$REPO"; git diff --quiet 5d36d52 -- '*.rs' '*.toml' Cargo.lock '*.metal'
cargo build --release -p llvq-bench --bin rtbits          # the one binary dated before 5d36d52
mkdir -p "$OUT/bin"; cp target/release/{oracle,smoke,seal,ppl,rtbits} "$OUT/bin/"
BIN=$OUT/bin; shasum -a 256 "$BIN"/* | tee "$OUT/bin.sha256"   # the checkout is free from here
sysctl vm.swapusage | tee "$OUT/swap.txt"

nice -n 10 "$BIN/oracle" Qwen/Qwen3-8B 64 metal 2>&1 | tee "$OUT/oracle-metal.txt"
grep -q MATCH "$OUT/oracle-metal.txt"

export LLVQ_MODEL=Qwen/Qwen3-8B LLVQ_CALIB=dclm-edu LLVQ_INT4_TYPES=v_proj
export LLVQ_ARTIFACT=$HOME/q8b-dclm-2026-09-21.llvq LLVQ_THREADS=12
env | grep '^LLVQ_' | sort | tee "$OUT/env.txt"          # exactly these five
/usr/bin/time -l nice -n 10 "$BIN/smoke" 64 2048 12 4096 metal nogs tetra1 999 rot 2>&1 | tee "$OUT/smoke.txt"
sysctl vm.swapusage | tee -a "$OUT/swap.txt"

/usr/bin/time -l nice -n 10 "$BIN/seal" "$LLVQ_ARTIFACT" "$HOME/qwen3-8b-dclm.bin" 2>&1 | tee "$OUT/seal.txt"
shasum -a 256 "$LLVQ_ARTIFACT" "$HOME/qwen3-8b-dclm.bin" | tee "$OUT/files.sha256"
LLVQ_DTYPE=f16 nice -n 10 "$BIN/ppl" 4096 12 metal "$HOME/qwen3-8b-dclm.bin" 2>&1 | tee "$OUT/ppl-sealed-f16.txt"
"$BIN/rtbits" "$HOME/qwen3-8b-dclm.bin" 2>&1 | tee "$OUT/rtbits.txt"
