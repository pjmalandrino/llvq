# Deviations from the prereg "served bit for bit" (2026-11-26), part A

The prereg is `proofs/preregistration-servi-au-bit-2026-11-26.md`, timestamped and not edited. Parts B and C are not
recorded here; the agent that ran them writes its own entries below theirs if it uses this file.

## A-E1. The baseline `llvq` binary was built from a tree under edit

`llvq-before` (sha256 `9a35c3cb…`), the binary that gives the old config's answer in A-v, was built from commit
`a05d99ad` while the agent of parts B and C had started editing `llvq-runtime/src/inkling.rs` and
`llvq-runtime/src/prefix.rs`. Those files serve Inkling only; the Qwen3 path of the binary is `a05d99ad`'s. The
`metalrun` of every timed reading was built from commit `1e35fa1b` alone (`llvq-llm` does not depend on
`llvq-runtime`).

## A-E2. A-ii ran the new config's kernels

The prereg does not name the kernels of A-ii. It ran `lutd+h+pq`, the kernels the new configs serve; M3 and M5 ran
`lutd`. The decode path of A-ii is one row a step, so the prefill kernels of `+pq` do not run in it; A-iii covers them.

## A-E3. Peak memory came from separate processes

`metalrun time` holds both arms in one process, so its footprint is the sum of two models. The peak at the long prompt
was read from one process a config, one generation each, with `/usr/bin/time -l` ("peak memory footprint"), after the
timed runs.

# Parts B and C

## BC1. Files outside the lot's list

Gate B-i is a test added to `llvq-llm/tests/inkling_cpu.rs`, which holds the tiny served file set and its CPU load.
The help of `llvq serve --prefix-cache` and the head of `llvq-runtime/src/main.rs` said "off on the CPU"; their two
lines now say "on for Metal and the CPU".

## BC2. Gate B-ii does not go through `inkrun`

`inkrun` loads native experts on the CPU, not the served CPU stack. B-ii is an ignored test,
`llvq-runtime/tests/inkling_cpu_prefix_real.rs`: `experts_cpu::load_served`, the first 2,600 ids of the lot 1 prompt
(`docs/mesures/inkling-mac-prefix-2026-11-26-brut/long.json`), the state written by `InklingCache::write_state` to a
file and read back by `Inkling::read_cache`, as the server's `save_state` and `load_state` do, without the server's
header and digest (B-iii covers those). The logits compared are the prompt's last row and 8 greedy steps.

## BC3. Gate B-iii's static block

12 sections and 3 tools of `prefix_checks.py` (`PREFIX_SECTIONS`, `PREFIX_TOOLS`): prompts of 2,316 tokens, save point
2,048. A block of 10 sections, tried first on Metal, ended under 2,048 (prompt 2,035, save point 1,536). The reference
answer, from a server with the cache off, is of the warm request (u5); the cold request (u4) that wrote the file has
no reference of its own.

## BC4. Gate B-iii ran with the cache turned on by its switch

The CPU build was of commit c00ce5e, whose CPU default was still off; every server of B-iii but the reference had
`--prefix-cache on`. The default changed after the reading, by the rule.

## BC5. The binaries of gate C-ii

"Before" is the `llvq` binary built at a05d99ad by the session of part A (Metal, sha256 9a35c3cb...). It was copied
before the first edit of parts B and C, and its behaviour shows the old code (a prompt held whole is read again through
the prefix file). "After" is c00ce5e, built with the uncommitted edits of part A in the same worktree (`engine.rs`,
`library.rs`, `library.json`: the Qwen3 path, which an Inkling request does not reach).

## BC6. Gate C-ii's "before" server read one calibration request first

One request of the shorter block of BC3, a conversation of its own: the agent request after it matched nothing held
and was read from an empty cache, as on the "after" server.
