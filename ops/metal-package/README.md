# llvq-metal

The sealed Qwen3-4B (Tetra, 2.7 b/param, 1.38 GB on the device) runs compressed on this Mac's GPU. One binary, no
cargo, no repository, no network: the tokenizer is inside the sealed file.

You need Apple silicon and the sealed file `qwen3-4b-sealed.bin` (1,418,224,685 bytes, sha256 `886391a8…`). The
launcher looks for it at `$SEALED`, then at `~/q4b-sealed-2026-09-23/qwen3-4b-sealed.bin`, and checks its size and
digest before every run.

## Talk to it

```bash
./llvq-metal chat              # the default: expanded attention, lutd, half-io, prefill kernels
./llvq-metal chat fused        # the served Metal kernels, expanded attention
./llvq-metal chat grouped      # grouped attention: bit for bit on Metal (M5), not served
./llvq-metal chat kernel       # the split-KV attention kernel from position 0: not bit for bit, inside T8 (M6)
./llvq-metal chat dense        # the dense reconstruction, the only Metal path before 2026-11-06
./llvq-metal chat expanded 0 256   # the default, greedy, 256 new tokens a turn at most
```

Arguments after the path: temperature (default 0.7, `0` is greedy) and the cap on new tokens a turn (default 512).
`/reset` drops the conversation, `/bye` leaves. The tok/s printed after each answer is the REPL's own rate, not a
measurement. Every path reads the same weights.

The default is the form the served door allows on Metal, expanded attention, and gives the tokens of `fused` bit for
bit. `grouped` returned the expanded form's logits bit for bit over 49,140 decode steps on this Mac (M5), but the
served door refuses it on Metal: this binary reaches it through its measurement door and says so on one line.
`kernel` returns the canonical replay's bits, but its tokens are not bit for bit with the other paths by design
(online softmax in f32, in another order). Its perplexity stays inside the card's T8 band of +-0.12 %, +0.0051 %
pooled over 12 windows of 4,096 (M6), and the served door refuses it on Metal too. The dense path gave the same 256
greedy tokens in the night's bench.

## Measure it

```bash
./llvq-metal bench          # about 9 minutes: fused, the chat default and dense, 256 tokens after a 5-token prompt
./llvq-metal bench --long   # about 20 minutes: grouped, grouped with half-io and prefill kernels, kernel,
                            # 64 tokens after 4,096 tokens of WikiText-2
./llvq-metal gate           # every path's 253 records against the served kernels, bit for bit
```

Each bench prints a bit gate, each arm's median tok/s with its range over rounds, each arm's ratio to the first one
round by round, and whether the tokens are the same. Close other GPU applications and wait for the machine to cool
after a build for a reading worth quoting. `bench --long` reads WikiText-2 from the Hugging Face cache, or downloads it
once.

What the night of 2026-11-06 read on an M3 Max, *measured*, same tokens on every arm
(M2, M4 and M6 of the repository's journal `docs/mesures/metal-nuit-2026-11-06.txt`):

| context | before | after |
|---|---|---|
| 5 tokens | dense 7.8 tok/s | fused 50.9, fused with `lutd` 52.1 (×1.025) |
| 4,096 tokens | expanded 15.8 tok/s | grouped 37.6 (×2.38), kernel 39.4 |

Fused against dense, x6.5 (dense over fused read x0.154 round by round, M2), is a raw ratio. No same-head arm exists
on Metal: the dense path copies its f16 head table every token, so part of its gap is that copy, not the kernel.

And on the afternoon of 2026-11-06, same Mac, *measured*, same tokens on every arm:

| what | before | after |
|---|---|---|
| decode, 5-token prompt | `lutd` 54.1 tok/s | with half-io 54.4 (×1.010, a wash) |
| reading a 4,096-token prompt | 59.7 s (68 tokens/s) | Tetra prefill kernel 37.2 s (×1.60), then int4 too: ×1.09 more, 31.5 s (130 tokens/s) |

The prefill kernels decode each weight block once for 8 prompt tokens instead of once a token. Every token they
return is the bit-identical f16 of the one-token path. A chat turn with a long pasted prompt starts answering about
twice as soon. Decode speed does not move.

## What is not here

None of these Metal paths is a served config. `configs/` holds the 4B's served config, which the paths start from, and
two configs marked NOT SERVED that name the grouped and kernel forms. The engine refuses both forms at its served door
on Metal; this binary reaches them through its measurement door and says so on one line. `MANIFEST.txt` names the
commit and the digest of every file.
