# Plan: the CUDA throughput work, ported to Metal

The sealed 4B now runs fused on the Mac, at 50.9 tok/s against 7.8 for the dense path, and 52.1 with the one ported
row loop that pays, all with the same 256 tokens (*measured*, M2 of the journal below). The fused path was refused at
load before this branch. The other ports of the CUDA row loops lose on Metal or move nothing measurable. The grouped
attention form is bit for bit with the expanded one on Metal over 49,140 decode steps, and decodes x1.75 faster at
2,048 tokens of context and x2.38 at 4,096, with the same tokens (*measured*, M4 and M5). The split-KV attention
kernel now exists in MSL, bit for bit with the canonical replay, and reads x0.86 the grouped form at a short context
and a wash at 2,048 and 4,096 (*measured*, M6). Both forms stay refused on the served Metal door: lifting that is
the operator's call. Two NOT SERVED Metal configs are ready for it: `configs/qwen3-4b-tetra-e4-metal-grouped-NOT-SERVED.json`, bit for bit
with the published expanded form, and `configs/qwen3-4b-tetra-e4-metal-kernel-NOT-SERVED.json`, the kernel from 1,280 tokens, inside the
card's ±0.12 % perplexity band (M6).

Branch `claude/metal-nuit-3c4257`, cut from `integration/lot1` at `8c0673a`. Night of 2026-11-05 to 11-06, $0, all
on the operator's M3 Max. Journal: [`mesures/metal-nuit-2026-11-06.txt`](mesures/metal-nuit-2026-11-06.txt).
Prereg: [`proofs/preregistration-metal-nuit-2026-11-06.md`](../proofs/preregistration-metal-nuit-2026-11-06.md).

## Test on waking

One command, from this worktree, on a quiet Mac (no build running, no other GPU application):

```bash
cd /Users/pjmalandrino/Documents/Pro/workspace/poc/llvq/.claude/worktrees/metal-nuit-3c4257 && ops/metal-bench.sh
```

The sealed file is `~/q4b-sealed-2026-09-23/qwen3-4b-sealed.bin` (1,418,224,685 B, sha256 `886391a8…`); the script
checks both before anything else, and takes another path as its first argument. The build takes seconds in this
worktree, about 9 minutes in a fresh checkout. The run takes about 9 minutes and 13 GB of unified memory.

What it must print, in order:
- `metal: tetra matvec … | int4 84 records, 12 through tv_q4_metal_tiled at 2048 columns a tile, 72 untiled`: the
  wall is lifted. Before this branch the same load stopped on `staging d_in=9728 wants 38912 B`.
- `bit gate: arm lutd against lut: 253 of 253 records equal bit for bit`.
- One line a round and an arm, then the medians. Last night's readings, M2 of the journal:
  - `arm lut` (before, fused with the served Metal kernels) about 51 tok/s;
  - `arm lutd` (after) about 52 tok/s;
  - `arm D` (before, the only way the file decoded on Metal until now) about 7.8 tok/s.
- `ratio lutd / lut: … decode x1.025 [1.013-1.034]` or near it, and `ratio D / lut` near x0.154.
- `tokens lutd against lut: the same 256 tokens` and `tokens D against lut: the same 256 tokens`.

Second command, about 25 minutes: the long context, where the attention work of the card pays. Three arms after the
first 4,096 tokens of WikiText-2: expanded (`lutd`, before), grouped (`lutd+grp`, after) and the split-KV kernel
(`lutd+ker`). The last two are measurement arms: `served.rs` refuses both forms on Metal by name.

```bash
cd /Users/pjmalandrino/Documents/Pro/workspace/poc/llvq/.claude/worktrees/metal-nuit-3c4257 && ops/metal-bench.sh --long
```

Last night's readings: `lutd` 15.8 tok/s, `lutd+grp` 37.5 to 37.6 (x2.380 [2.362-2.396] over `lutd`), `lutd+ker` 39.4
(x1.037 [0.996-1.093] over `lutd+grp`), and `the same 64 tokens` on every arm (M4 and M6 of the journal). The prefill
of 4,096 tokens takes about 62 s an arm: Metal prefills one row a launch.

The NOT SERVED Metal config can be loaded the same way, with every arm then under the grouped form:
`target/release/metalrun time ~/q4b-sealed-2026-09-23/qwen3-4b-sealed.bin configs/qwen3-4b-tetra-e4-metal-grouped-NOT-SERVED.json lut,lutd 256 7`
prints `NOT SERVED on Metal, measured through Served::apply_measurement` first.

The absolute rates move with the machine's thermals: M1 read F0 at 47.2 and M2 at 50.9 tok/s on the same binary an
hour apart. The ratios formed round by round are what carries over. `metalrun` refuses to start if any `LLVQ_*` or
`CANDLE_*` variable is set in the shell.

## 1. Where the night started

No sealed file ran fused on Metal. The loader refused the 12 int4 `down_proj` records of the 4B: d_in 9,728 stages
38,912 B against 32,768 B of threadgroup memory (`fused_metal.rs`, before this branch). The sealed 4B decoded on the
Mac only through the dense reconstruction in candle.

On the L40S the served 4B went from 114 to 178 tok/s at the door between 2026-10-29 and 11-05. Four pieces did it:
grouped attention, the split-KV attention kernel, 4E (one launch a token) and its v4 row loop. Metal had none of them,
and refuses the served config by name.

## 2. Each CUDA piece, and what it becomes on Metal

| CUDA piece | on Metal | status after the night |
|---|---|---|
| int4 wall (not a CUDA piece: CUDA stages 99 KiB) | `tv_q4_metal_tiled` wired, 2,048 columns a tile | done: the sealed 4B loads and decodes fused; F0 at 50.85 tok/s [45.18-51.45], x6.5 the dense arm, same tokens |
| 4E v4 row loop: next block's words in flight | `tv_tetra48_metal_lutp` | refused: x0.929 [0.912-0.953], bit for bit |
| 4E v6: loads issued before first use | `_lutd` (activation from device, no staging), `_lutdp` (both) | `_lutd` retained: x1.055 [1.006-1.087] in M1, x1.025 [1.013-1.034] in M2; `_lutdp` a wash, x1.004 |
| 4E v6 on the rotation and the norms | not ported. The rotation staged in threadgroup memory (`rot_apply_tg_metal`, already in the file) measured 7.5 % slower in situ on 2026-09-21 (`31a5acb`); the norms are candle's own `rmsnorm` ops | left as is, the prudent choice: the matvecs and the attention carry the step |
| 4E's int4 row loop with prefetch | `tv_q4_metal_p`, `tv_q4_metal_tiled_p` | wash: x1.003 [0.891-1.061] |
| (Metal only) R rows a simdgroup, lane order kept | `_lutd2`, `_lutd4` | `_lutd2` a wash, x1.023 [0.972-1.107]; `_lutd4` refused, x0.916 |
| grouped attention | the backend-free `grouped_attention`; `metalrun` arm `+grp`; `configs/qwen3-4b-tetra-e4-metal-grouped-NOT-SERVED.json` | bit for bit with the expanded form on Metal over 49,140 decode steps of 12 windows of 4,096 (M5); x0.999 at 5 tokens, x1.750 [1.599-1.997] at 2,048, x2.380 [2.362-2.396] at 4,096, same tokens (M4); still refused on the served Metal door |
| 4E itself: one launch a token | no Metal equivalent: no grid-wide barrier, no forward-progress guarantee | a candle launch costs 1.66 us in a dependent chain (*measured*): about 2 ms of a 19.7 ms token at 1,200 to 1,400 launches (*estimated* from the forward's ops) |
| split-KV decode-attention kernel | `kernels/attn_dec.metal` + `attn_metal.rs`, two dispatches a layer, safe math; `metalrun` arm `+ker`, from position 0 | bit for bit with the canonical replay (`attn_ref.rs`), hence the card kernel, up to 4,096 positions; against the grouped form: x0.864 [0.851-0.879] at 5 tokens, x0.974 [0.848-1.030] at 2,048, x1.037 [0.996-1.093] at 4,096, same tokens (M6); T8 band held: +0.0051 % pooled over 12 windows of 4,096, every window inside ±0.12 %, top-1 99.772 % (M6); `configs/qwen3-4b-tetra-e4-metal-kernel-NOT-SERVED.json` (kernel from 1,280) |

## 3. What was built, and how it is held

Every kernel below returns the served kernel's bits. None changes a served config: the served door passes no tuning
and loads what it loaded before, plus the tiled route for the records it used to refuse.

| piece | where | how it is held |
|---|---|---|
| split-KV attention kernel | `kernels/attn_dec.metal`, `src/attn_metal.rs`, `attn::kernel_for` on Metal | `llvq-llm/tests/attn_metal.rs` (6 tests: every boundary to 4,096, verify rows, another geometry, repetition); a self-check against the replay every time it is built; 10 mutants, 10 killed |
| measurement door | `Served::apply_measurement`, used by `metalrun` only | `served_config` and `served_unit` tests; `Served::apply` and its refusals unchanged |
| tiled int4 route | `fused_metal.rs`: `Q4_TILE_COLS`, `int4_tile_for`, `upload_int4_tiled`, `Int4Op::metal_fwd` | `llvq-metal/tests/q4_tiled_matches_served.rs` (8 tests: tiled = served, = host reference at 9,728, 12,288, 17,408, fast math either way, 16 repetitions); `llvq-llm/tests/q4_metal_candle.rs` (binding at six widths, offset activation, refusals) |
| Tetra row-loop twins `_lutp`, `_lutd`, `_lutdp`, `_lutd2`, `_lutd4` | `kernels/llvq_tetra48_rows.metal`, compiled after the served unit as its own library | `llvq-llm/tests/tetra48_metal_rows.rs` (6 tests, against the served kernel and the CPU oracle at the 4B's shapes and the loop edges) |
| int4 prefetching twins | `kernels/tv_q4_pf.metal`, same rule | in `q4_tiled_matches_served.rs` and `q4_metal_candle.rs` |
| `MetalTuning`, `load_resolved_tuned`, `compiled_pipelines` | `fused_metal.rs` | the tests above; `compiled_pipelines` proves which kernel ran, since outputs cannot |
| `metalrun` | `llvq-llm/src/bin/metalrun.rs` | its bit gate: every arm's 252 projections and head against the reference arm's on the real file, 253 of 253 for all nine fused arms |

Mutation runs: 37 mutants, 37 killed (27 in the matvec lot, among them the four barrier removals; 10 in the attention
kernel). `cargo clippy --all-targets`: zero
warnings. The Linux half was not type-checked: Docker was not running, and the cross `cargo check` stops in
`openssl-sys`, a dependency of `hf-hub`. Every line this branch adds outside a test is under `cfg(target_os =
"macos")` or in `metalrun`, whose non-Metal stub compiles on the Mac without the `metal` feature.

`tv_q4_h.metal` and `llvq_tetra48.metal` are byte-identical to `integration/lot1`: the llvq-tetra wheel ships and
checks copies of the served shaders.

## 4. What remains after the night

Ordered by what a step buys on a fundamental criterion (served tok/s on the Mac), then by cost. Times are
*estimated* working time, machine time apart.

| # | step | cost | gate | what it buys |
|---|---|---|---|---|
| 1 | Operator ruling: does a Mac reading answer ETAT §5's "consumer GPU" item, and is a Metal config served | 0 | operator | the wall lift becomes a served path on the Mac |
| 2 | Operator ruling on the grouped form on Metal: bit identity on 12 windows is measured (M5), the config is written NOT SERVED. Lifting the refusal is a one-line change in `served.rs` `check_device` plus renaming the config | 0 | operator | x1.75 at 2,048 and x2.38 at 4,096 (M4), the largest Metal gain measured, at zero bit cost |
| 3 | Rows kernel for the prefill (`mac.md` §3.3 b2): 256-row cap lifted, TTFT ×2-3 | 2-3 days | G-D bit for bit on 252 records, prereg for TTFT | long prompts and chat turns past 256 tokens on the fused Metal path; every long-context measurement below needs it |
| 4 | The attention kernel's speed on Metal: it loses 14 % at a short context. A one-dispatch form (the combine fused into the split pass by an ordered fold, or a split count of one under 256 positions) and the CUDA kernel's 32-position staging | 1 day | bit for bit against `attn_ref.rs` (the arithmetic does not move), then the M6 arms | the kernel at least level with the grouped form at every length; at 8,192 untested on Metal |
| 5 | f16 in and out of the matvecs: 336 conversion launches a token removed | 0.5 day | bit for bit (`half()` is RNE in MSL and in candle's cast) | probably little: 290 such launches removed on the q5 object changed nothing measurable (`5a75338`) |
| 6 | Same-head ratio on Metal: an f16-head fused arm, and the dense head's per-token table copy (`model.rs` `broadcast_matmul`) | 0.5 day plus an operator ruling | rule 4 | the raw F0/D ratio gets its same-head companion |
| 7 | The 8B and the 14B on the Mac | download 7.9 GB (operator), then 1 h | `metalrun gate`, then a prereg | the other two sealed files; the tiled int4 kernel already passes their d_in in the tests |

4E itself has no Metal port on this plan. Metal offers no barrier across threadgroups and no documented
forward-progress guarantee between them, so a persistent kernel that spins on other threadgroups can deadlock. The
analog would be one command buffer a token with indirect command buffers, which is A2's mechanism and closed in the
core by rule 3. A launch measured 1.66 us through candle in a dependent chain, so removing every launch but the
matvecs would save about 10 % of the step (*estimated*). The 36 blocks take 18.0 to 19.8 ms of a 19.2 to 19.7 ms
token (*measured*, fenced, off protocol), and the whole step reads 70 GB/s of weights (*computed*) where the M3 Max
offers about 340. At a short context the next Metal lever is the Tetra kernel's decode, not the launch count.

## 5. What this plan does not decide

- Serving anything on Metal. No served config was changed. Which Metal arm becomes a served config is the operator's
  decision, as is whether a Mac reading answers the "consumer GPU" item of `ETAT.md` §5.
- Lifting the Metal refusal of the grouped form or of the kernel form. M5 read the grouped form bit for bit on the
  L40S's 12-window design; M6 read the kernel's band. The ruling is step 2.
- GSM8K under either form on Metal, the card T8's second criterion: 1,319 problems did not fit in the night.
