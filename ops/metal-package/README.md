# llvq-metal

The sealed Qwen3-4B (Tetra, 2.7 b/param, 1.38 GB on the device) runs compressed on this Mac's GPU. One binary, no
cargo, no repository, no network: the tokenizer is inside the sealed file.

You need Apple silicon and the sealed file `qwen3-4b-sealed.bin` (1,418,224,685 bytes, sha256 `886391a8…`). The
launcher looks for it at `$SEALED`, then at `~/q4b-sealed-2026-09-23/qwen3-4b-sealed.bin`, and checks its size and
digest before every run.

## Talk to it

```bash
./llvq-metal chat              # the new path: fused kernels, lutd, grouped attention
./llvq-metal chat fused        # the served Metal kernels, expanded attention
./llvq-metal chat dense        # the dense reconstruction, the only Metal path before 2026-11-06
./llvq-metal chat new 0 256    # greedy, 256 new tokens a turn at most
```

Arguments after the path: temperature (default 0.7, `0` is greedy) and the cap on new tokens a turn (default 512).
`/reset` drops the conversation, `/bye` leaves. The tok/s printed after each answer is the REPL's own rate, not a
measurement. `new`, `fused` and `dense` read the same weights. `new` and `fused` give the same tokens bit for bit; the
dense path gave the same 256 greedy tokens in the night's bench.

`./llvq-metal chat kernel` uses the split-KV attention kernel from position 0. It returns the canonical replay's bits,
but its tokens are not bit for bit with the other paths by design: online softmax in f32, in another order.

## Measure it

```bash
./llvq-metal bench          # about 9 minutes: lut, lutd and dense, 256 tokens after a 5-token prompt
./llvq-metal bench --long   # about 25 minutes: expanded, grouped, kernel, 64 tokens after 4,096 tokens of WikiText-2
./llvq-metal gate           # every path's 253 records against the served kernels, bit for bit
```

Each bench prints a bit gate, each arm's median tok/s with its range over rounds, each arm's ratio to the first one
round by round, and whether the tokens are the same. Close other GPU applications and wait for the machine to cool
after a build for a reading worth quoting. `bench --long` reads WikiText-2 from the Hugging Face cache, or downloads it
once.

What the night of 2026-11-06 read on an M3 Max, *measured*, same tokens on every arm:

| context | before | after |
|---|---|---|
| 5 tokens | dense 7.8 tok/s | fused 50.9, fused with `lutd` 52.1 (×1.025) |
| 4,096 tokens | expanded 15.8 tok/s | grouped 37.6 (×2.38), kernel 39.4 |

## What is not here

None of these Metal paths is a served config. `configs/` holds the 4B's served config, which the paths start from, and
two configs marked NOT SERVED that name the grouped and kernel forms. The engine refuses both forms at its served door
on Metal; this binary reaches them through its measurement door and says so on one line. `MANIFEST.txt` names the
commit and the digest of every file.
