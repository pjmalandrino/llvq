# Preregistration. MLX T8: the MLX door's WikiText-2 perplexity against the Rust Metal path

Written on 2026-11-07 on branch `claude/serve-mlx-metal-855462`, once the code it describes existed (the commit that
carries this file). To be timestamped with `ots stamp` before its first measurement, then never edited; departures go
to `preregistration-mlx-t8-2026-11-07-ECARTS.md`. Cost **$0**, local, on the operator's M3 Max. The Mac's GPU is
taken for the length of the run, so it waits for the operator's go and hour.

## 1. Why

The operator chose MLX as the engine on the Mac (decision 2, 2026-11-06), then wrote on 2026-11-07: « Sur métal, on a
validé mlx donc implem le aussi ». That sentence is the operator's ruling on open decision 1 of
`docs/plan-metal-debit.md` §3: on Metal, a fused path is served under a tolerance, not only bit for bit.

The MLX door (`ops/mlx_tetra/serve.py`, config `configs/mlx/qwen3-4b-tetra-e4-mlx.json`) runs the sealed Qwen3-4B with
f16 activations, other kernels than the Rust path for everything but the Tetra matvec, and another rotation
arithmetic. Its tokens leave the Rust Metal path's at token 243 after a 5-token prompt (M15 of
`docs/mesures/metal-nuit-2026-11-06.txt`). It has never been scored. The tolerance is T8's band, the one that
licensed the grouped and kernel forms on the L40S and the kernel form on Metal (M6): ±0.12 % of WikiText-2
perplexity, pooled and every window.

The reference is M5: the sealed 4B through the Rust Metal kernels (`lutd`, expanded attention, f16), decode path, the
first 12 WikiText-2 test windows of 4,096, pooled **12.580769** (*measured*, raw `m5.txt` of the journal). Its 12
window perplexities are in `ops/mlx_tetra/ppl.py` (`M5_WINDOWS`), copied from that raw file.

## 2. What runs

One command, from the repository root, on a quiet Mac (no build, no other GPU application):

```bash
ops/mlx-t8.sh
```

It refuses to start without `proofs/preregistration-mlx-t8-2026-11-07.md.ots`, or with any `LLVQ_*` variable set. It
writes every step's output to `$OUT` (default `~/llvq-mlx-t8-<date>`), and records first: date, git commit and status,
the sha256 of this file, of the config and of the six Python files of `ops/mlx_tetra/` it runs, uv and mlx versions,
machine model, macOS version, thermal state and load. Then, by path, each under nice 10:

1. `uv run ops/mlx_tetra/test_served.py`: the door's refusals.
2. `uv run ops/mlx_tetra/check.py llvq-tetra/tests/fixtures/mini`: sections A to D on the fixture.
3. `uv run ops/llvq_hf_check.py $PACKED`: every field of the packed directory against the sealed file's hashes.
4. `uv run ops/mlx_tetra/check.py $PACKED 0,12`: sections A to C on the sealed 4B, layers 0 and 12.
5. `uv run ops/mlx_tetra/serve.py <config> $PACKED span`: the span gate alone.
6. `uv run ops/mlx_tetra/serve.py <config> $PACKED ppl 4096 12 decode,prefill`: the reading.

`$PACKED` is `~/q4b-hf-2026-09-28`, the hfpack of `qwen3-4b-sealed.bin` (sha256 `886391a8…`).

The two paths of step 6, each window from an empty cache:
- **decode**, M5's path: one true token a step, 4,095 predictions a window, through the one-row kernels.
- **prefill**: the window in chunks of 2,048 rows, as mlx-lm 0.28.3's `generate_step` reads a prompt, through the
  dense prefill.

## 3. Gates

Any gate that fails stops the run before step 6. It is journaled; no perplexity is read.

1. Step 1: 9 of 9. Step 2: 41 of 41 comparisons within their bounds. Step 3: exit 0.
2. Step 4: every comparison within the bounds written in `check.py` at the stamp (rotation 1e-5, kernels 2e-3,
   dense prefill 4e-3, embedding 2e-3). These bounds were set on the fixture, where the worst of those reads
   6.1e-4; they have never run on the sealed 4B. A bound that the 4B exceeds is not widened inside this run: the run stops, and a
   new bound is a new prereg.
3. Step 5: 299,078 tokens in the split, span sha256 `6b69a6ba6998ddda8d1f65406ac23b3e1bc09e0a693f8809d97f0ca7f4b38aab`,
   M5's. Checked on 2026-11-07 before this prereg was written, without loading the model.
4. The door's own refusals pass: no `LLVQ_*` variable, the packed object `886391a8…`, mlx 0.32.4, mlx-lm 0.28.3,
   the GPU.

## 4. Rules

For each path, against M5 window by window (`ppl.py`, `verdict`):

- **T8 band**: the pooled `MLX / M5 - 1` and each of the 12 window ratios inside [-0.12 %, +0.12 %].
- **Decode** is the primary path: it is M5's path, and it writes every generated token.
- **Prefill** is held to the same band: it reads every prompt, and its cache feeds every later step.

| decode | prefill | consequence |
|---|---|---|
| inside | inside | The reading licenses the MLX door as the served Metal path of the 4B, under the operator's ruling of §1. `configs/README.md`, `docs/ETAT.md` and `docs/plan-metal-debit.md` record it served once the operator has read the journal |
| inside | outside | Not served. The dense prefill is named as the path outside the band; a fix is a new prereg |
| outside | either | Not served. The one-row path is named; a fix is a new prereg |

A pooled ratio inside with one window outside is outside, as written. Reported and not gated: the paired t interval
over the 12 window log ratios, the top-1 hit counts, the argmax digests, the elapsed time.

## 5. Predictions, signed

*Estimated*.

| quantity | prediction | why |
|---|---|---|
| decode, pooled `MLX / M5 - 1` | inside ±0.05 % | on Metal, f16 against f32 everywhere moved the pooled perplexity by +0.0225 % [-0.0353 ; +0.0803] (`s1-f32-metal-2026-11-04`); here both sides are f16 and differ by kernel and order only |
| decode, every window inside ±0.12 % | yes, with about 60 % confidence | the same reading saw one window move by 0.19 % (+0.00189 nats, window 7) from f16 alone; per-window scatter is the risk of this rule |
| decode, widest window ratio | 0.03 % to 0.20 % | as above |
| prefill, pooled `MLX / M5 - 1` | inside ±0.05 % | on the card, decode against prefill of one form differs by -2.8e-4 to +3.6e-4 nats a window (`gqa-ppl-2026-11-01`) |
| decode path, wall time | 8 to 20 min | 49,140 steps at the 70 to 94 tok/s of M16, plus one read-back a step |
| prefill path, wall time | 1 to 3 min | 24 chunks of 2,048 rows; M16 read the first token after 4,096 tokens in 4.11 s |

## 6. What this does not establish

- Any task score (MMLU, GSM8K) through the MLX door. T8 on the card has a GSM8K criterion too; it does not run here.
- Speed at the served door: M16 timed the prototype's door, which runs the same engine. A `serve.py time` reading is
  not part of this run.
- Context past 4,096, the 8B and the 14B, and any Mac other than this M3 Max.
