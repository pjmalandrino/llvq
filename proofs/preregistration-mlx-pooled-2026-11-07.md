# Preregistration. The MLX door on fresh windows, under a gate on the paired interval

Written on 2026-11-07 on branch `claude/serve-mlx-metal-855462`, after the first T8 reading of the MLX door
(`docs/mesures/mlx-t8-2026-11-07.txt`) and before any reading of the windows below. To be timestamped with
`ots stamp` before its first measurement, then never edited; departures go to
`preregistration-mlx-pooled-2026-11-07-ECARTS.md`. Cost **$0**, local, on the operator's M3 Max, about 25 minutes of
its GPU (*estimated*).

## 1. Why

The first reading kept the MLX door's pooled perplexity within +0.0078 % [-0.0554 ; +0.0710] of the Rust Metal path
by decode, and missed the per-window half of its rule on windows 6 and 7 (+0.27 %, -0.13 %). The same two windows move
by about 0.2 % between two Rust engines (Metal f16 against f32, the card against Metal). After reading that journal,
the operator chose on 2026-11-07 to keep MLX and change the gate. Verbatim: « oiui c'est bon on gare », corrected by
« garde * »; then, asked which, the option "Garder MLX": a gate on the paired pooled interval, stamped before a new
reading. The first reading's windows are
not reread under the new gate: this one scores windows it has never seen.

## 2. What runs

```bash
ops/mlx-pooled.sh
```

It refuses to start without this file's `.ots`, or with any `LLVQ_*` variable in its environment. It records the
environment as `ops/mlx-t8.sh` does, then, under nice 10:

1. **Reference.** The Rust `ppl` binary of `s1-f32-metal-2026-11-04`, sha256 `5f3ff087…`
   (`.claude/worktrees/validation-vague1/target-local/release/ppl`), `LLVQ_DTYPE=f16` on its command line only:
   `ppl 4096 24 metal ~/q4b-sealed-2026-09-23/qwen3-4b-sealed.bin`. The dense reconstruction of the sealed 4B, f16, on
   Metal, the first 24 windows of 4,096. On windows 1 to 12 this reading agreed with M5 (the fused Rust path) within
   [-0.029 ; +0.058] % a window (*computed*).
2. `uv run ops/mlx_tetra/test_served.py`, then `check.py` on the mini fixture and on the sealed 4B (layers 0, 12).
3. `uv run ops/mlx_tetra/serve.py <config> $PACKED ppl 4096 12 decode,prefill 12 <reference>`: windows 13 to 24,
   decode and prefill as in the first reading, each window against the reference's window of the same number.

## 3. Gates

Any gate that fails stops the run before step 3, and is journaled.

1. The binary's sha256 is `5f3ff087…` and the sealed file's `886391a8…`.
2. The reference's windows 1 to 12 equal the NLLs of `s1-f32-metal-2026-11-04-brut/ppl-f16.txt` to the last printed
   digit: same binary, same machine, same windows.
3. Step 2 as in the first reading: 9 of 9, 41 of 41, 66 of 66.
4. In step 3, the first 12 windows of the MLX span hash to M5's `6b69a6ba…`, and the split holds 299,078 tokens.

## 4. Rule

For each path, over windows 13 to 24: the 95 % paired t interval of `MLX / Rust - 1`, formed on the 12 per-window
log ratios, lies inside [-0.12 %, +0.12 %].

| decode | prefill | consequence |
|---|---|---|
| inside | inside | The MLX door is the served Metal path of the 4B, on the operator's choice of §1. `configs/README.md`, `docs/ETAT.md`, `docs/plan-metal-debit.md` and `CLAUDE.md` say so |
| outside | either, or the reverse | Not served. The journal names the path |

Reported, not gated: each window's ratio, the pooled ratio, the top-1 hits, the elapsed time.

## 5. Predictions, signed

*Estimated*.

| quantity | prediction | why |
|---|---|---|
| decode interval | inside, about 80 % | the first reading's was [-0.0554 ; +0.0710], 0.13 % wide and centred near zero |
| decode pooled | inside ±0.05 % | the first reading read +0.0078 % |
| widest window ratio | 0.05 % to 0.30 % | windows 1 to 12 reached 0.27 % |
| decode against prefill, every window | within 0.03 % | the first reading's pair was within [-0.0197 ; +0.0123] % |
| reference wall time | 7 to 10 min | S1 read 12 windows in 219 s |
| MLX decode wall time | 12 to 20 min | the first reading took 886 s |

## 6. What this does not establish

Any task score through the MLX door, context past 4,096, the 8B and the 14B. A reference other than the dense
reconstruction: the fused Rust path was not run on windows 13 to 24.
