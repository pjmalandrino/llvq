#!/bin/bash
# usage: tlim.sh <seconds> <command> [args...]
# Runs the command with stdin closed in its own process group and kills the
# whole group after <seconds> (TERM, then KILL 10 s later).
set -u
secs=$1; shift
set -m
"$@" < /dev/null &
c=$!
( sleep "$secs"; echo "[tlim] $(date '+%T') time limit ${secs}s reached, killing group $c" >&2
  kill -TERM -- -"$c" 2>/dev/null; sleep 10; kill -KILL -- -"$c" 2>/dev/null ) < /dev/null &
w=$!
wait "$c"; rc=$?
pkill -P "$w" 2>/dev/null; kill "$w" 2>/dev/null
exit $rc
