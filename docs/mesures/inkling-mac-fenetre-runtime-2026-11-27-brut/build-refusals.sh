#!/bin/bash
# Lot H, part 1, from the worktree root, under the guard: the release build of the committed tree copied to
# target/loth/llvq, a CPU build in target/loth-cpu, then the refusals that need no model read:
#   the switch's parse, the help, narrow on the CPU backend (Inkling 2, refused before a weight is read),
#   narrow on a Qwen3 of the store (Metal build; the operator's blob linked read-only into a working home).
set -u
B=docs/mesures/inkling-mac-fenetre-runtime-2026-11-27-brut
W=$B/work; mkdir -p $W
M=$HOME/.llvq/models/sha256-1e9314dba89332cb1af444619dc5d31a2a1ed6a50a1112e0ca2b8fc55bb6bfd3/inkling.json
lim() { perl -e 'alarm shift; exec @ARGV or die' "$@"; }
echo "build: start $(date '+%F %T') at $(git rev-parse HEAD); tree: [$(git status --short llvq-llm llvq-runtime configs | tr '\n' ' ')]"
lim 1500 cargo build --release -p llvq-runtime --features metal --bin llvq > $B/build-after.log 2>&1 || { echo "STOP build"; exit 1; }
cp target/release/llvq target/loth/llvq && shasum -a 256 target/loth/llvq | tee $B/llvq.sha256
target/loth/llvq -v
CARGO_TARGET_DIR=target/loth-cpu lim 1500 cargo build --release -p llvq-runtime --bin llvq > $B/build-cpu.log 2>&1 || { echo "STOP cpu build"; exit 1; }
shasum -a 256 target/loth-cpu/release/llvq | tee $B/llvq-cpu.sha256
echo "== the parse =="
for v in wide Narrow ""; do
  lim 30 target/loth/llvq serve --attn-window "$v" < /dev/null > $W/parse.out 2>&1; echo "--attn-window '$v': rc=$? $(tr '\n' ' ' < $W/parse.out)"
done
echo "== the help (hidden) =="
lim 30 target/loth/llvq serve --help < /dev/null > $B/help-after.txt 2>&1
lim 30 target/loth/llvq-before serve --help < /dev/null > $B/help-before.txt 2>&1
cmp $B/help-before.txt $B/help-after.txt && echo "help: identical to the build before"; grep -c attn $B/help-after.txt
# serve_one <label> <binary> <home> <port> <model> [flags]: start, ask one empty /api/chat, print the answer, stop
serve_one() {
  local label=$1 bin=$2 home=$3 port=$4 model=$5; shift 5
  mkdir -p "$home"
  env LLVQ_HOME="$home" LLVQ_HOST=127.0.0.1:$port LLVQ_KEEP_ALIVE=-1 "$bin" serve "$@" < /dev/null > $B/$label.server.log 2>&1 &
  local srv=$! up=0
  for _ in $(seq 1 60); do curl -sS --max-time 2 http://127.0.0.1:$port/api/version > /dev/null 2>&1 && { up=1; break; }; kill -0 $srv 2>/dev/null || break; sleep 1; done
  echo "$label: up=$up"
  printf '{"model": "%s", "messages": [], "stream": false, "keep_alive": -1}' "$model" > $W/$label.req.json
  curl -sS --max-time 300 -w '\nHTTP %{http_code} %{time_total} s\n' -H 'Content-Type: application/json' --data-binary @$W/$label.req.json \
    http://127.0.0.1:$port/api/chat > $B/$label.answer.txt 2>&1
  cat $B/$label.answer.txt
  kill $srv 2>/dev/null; wait $srv 2>/dev/null
  echo "$label server log:"; grep -v '^ *[/\\]' $B/$label.server.log | grep -v tetra
}
echo "== narrow on the CPU backend, Inkling 2 =="
serve_one ref-cpu-inkling target/loth-cpu/release/llvq $W/home-cpu 11438 inkling2:mini --local $M --attn-window narrow
echo "== narrow on a Qwen3 of the store, Metal build =="
H=$W/home-qwen; mkdir -p $H/manifests/qwen3 $H/blobs
cp $HOME/.llvq/manifests/qwen3/4b.json $H/manifests/qwen3/4b.json
ln -s $HOME/.llvq/blobs/sha256-886391a8c03f66dc269cc65c3598c6627dbdcd259180aff36604ef10d37371b8 $H/blobs/
serve_one ref-metal-qwen3 target/loth/llvq $H 11438 qwen3:4b --attn-window narrow
echo "build: done $(date '+%F %T')"
