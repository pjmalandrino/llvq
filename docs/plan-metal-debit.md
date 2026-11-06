# Plan: throughput on Metal

Nothing is served on Metal. The sealed 4B runs on the Mac by two unserved paths. The Rust path (`metalrun`) decodes
at 65.5 tok/s after a 5-token prompt, every kernel bit for bit with the served Metal kernels (*measured*, M14). The
MLX prototype (`ops/mlx_tetra/llvq_mlx.py`) decodes at 93.9 tok/s, and at 70.3 after 4,096 tokens of context, but is
not bit-exact and has no perplexity reading (*measured*, M16). mlx-lm's own Qwen3-4B 4-bit reads 129.5 and 98.4 on
the same Mac, at about 4.5 b/param against our 2.7 (*measured*, M16). Rule 5: no ratio is formed across the two
stacks.

Every M number is from the journal [`mesures/metal-nuit-2026-11-06.txt`](mesures/metal-nuit-2026-11-06.txt), on the
operator's M3 Max, $0, no job. Preregs: `proofs/preregistration-metal-*-2026-11-06.md`, one a reading. The design
written after M13 is archived as [`archive/plan-metal-v2-2026-11-06.md`](archive/plan-metal-v2-2026-11-06.md); the
M14 and M15 preregs cite it as `docs/plan-metal-v2.md`.

## Run it

From the repository root, on a quiet Mac (no build running, no other GPU application):

```bash
ops/metal-bench.sh          # about 9 minutes: lut, lutd, D, 256 tokens after a 5-token prompt
ops/metal-bench.sh --long   # about 25 minutes: lutd, lutd+grp, lutd+ker, 64 tokens after 4,096 tokens of WikiText-2
ops/llvq-mlx bench          # the MLX prototype, same prompt; needs PACKED= (hfpack of the sealed file)
```

The sealed file is `~/q4b-sealed-2026-09-23/qwen3-4b-sealed.bin` (1,418,224,685 B, sha256 `886391a8…`); the script
checks both, and takes another path as its last argument. `metal-bench.sh` prints the bit gate (253 of 253 records),
each arm's median tok/s with its range, each arm over the first round by round, and whether the tokens match. Absolute
rates move with the machine's thermals (M1 read `lut` at 47.2 and M2 at 50.9 an hour apart): the ratios formed round
by round carry over. `ops/metal-package.sh` builds the same engine as a bundle with no cargo (`ops/metal-package/`).

## 1. Where it stands

The Rust path, every arm the same file in one process, *measured*:

| reading | result | journal |
|---|---|---|
| int4 wall lifted: the 12 int4 `down_proj` records past d_in 8,192 go through `tv_q4_metal_tiled` | the sealed 4B loads fused; `lut` 50.9 tok/s [45.2-51.5] | M2 |
| dense reconstruction `D`, the only Metal path before | 7.8 tok/s; `lut` over `D` x6.5 raw. No same-head arm exists on Metal: `D` copies its f16 head table every token | M2 |
| `lutd`: the Tetra matvec reads its activation unstaged | x1.025 [1.013-1.034] over `lut`, same 256 tokens | M2 |
| other ported row loops (`lutp`, `lutdp`, `lutd2`, `lutd4`, `+q4p`, tile 8,192) | refused or a wash | M1 |
| grouped attention | bit for bit over 49,140 decode steps of 12 windows of 4,096; x1.750 at 2,048, x2.380 at 4,096 | M5, M4 |
| split-KV attention kernel `+ker` | x0.864 at 5 tokens, a wash at 2,048 and 4,096; +0.0051 % ppl pooled, every window inside T8's +-0.12 % | M6 |
| half-io twins `+h` | x1.010, a wash, kept since it moves no bit | M8 |
| Tetra prefill kernel `+p`, then int4 `+pq` | first token after 4,096 tokens x1.595, then x1.090 more: 31.5 s, 130 prompt tokens/s | M9, M10 |
| `lutd+h+grp+pq`, cool machine | 65.9 tok/s; host 5.4 ms of a 15.2 ms token, GPU-bound | M12 |
| `lutd+h+grp+pq+r`, the bar run | 65.5 tok/s [65.3-66.5], 90 GB/s of weights | M14 |

A short-context token splits as Tetra matvec 9.0 ms (58 %), int4 2.4, rotations 1.95, attention 1.25, the rest 0.87
(*computed*, M7b and M13). The Tetra classes read 74 to 90 GB/s of weights alone against about 400 nominal: the kernel
is decode-bound. At 4,096 positions the grouped attention takes 7.9 ms, 76 GB/s of KV (*computed*, M13).

The MLX path is mlx-lm's Qwen3 with the 168 Tetra records as custom Metal kernels. It reads the sealed codes from the
packed directory without re-quantizing them, and runs f16 activations. *Measured*, M15 then M16 after tuning:

| | 5-token prompt | after 4,096 tokens | first token at 4,096 | peak | weights read |
|---|---|---|---|---|---|
| MLX path, M15 | 81.6 tok/s [80.0-81.8] | 59.4 [56.2-66.1] | 3.49 s | 1.50 / 2.99 GB | 1.377 GB, 113 GB/s |
| MLX path, M16 | 93.9 tok/s [88.4-94.7] | 70.3 [69.7-71.4] | 4.11 s | 1.63 / 3.20 GB | 1.377 GB, 129 GB/s |
| mlx-lm 4-bit, M16 | 129.5 tok/s [129.3-129.5] | 98.4 [97.1-104.2] | 3.96 s | 2.39 / 3.67 GB | 2.26 GB, about 290 GB/s |

GB/s are *computed*. b/param for the whole model with its tables: ours 2.7, the rival about 4.5. After the 5-token
prompt the MLX path's 256 greedy tokens differ from the Rust path's: it leaves the repeated answer at token 243, the
Rust path at another token. After 4,096 tokens of context its 64 tokens are the Rust path's (M15). No perplexity has
been measured on it.

## 2. What is served on Metal

Nothing. No served config changed. The served Metal door accepts the expanded config only, and refuses the grouped
and kernel forms by name (`served.rs`). Two configs name those forms and are marked NOT SERVED:
`configs/qwen3-4b-tetra-e4-metal-grouped-NOT-SERVED.json` and `configs/qwen3-4b-tetra-e4-metal-kernel-NOT-SERVED.json`.
`metalrun` reaches them through `Served::apply_measurement` and says so on one line.

The Metal door's behaviour did change once. Commit `31e49c1` routes the int4 records past d_in 8,192 to
`tv_q4_metal_tiled`, bit-identical to the served kernel (`llvq-metal/tests/q4_tiled_matches_served.rs`). Before it,
the fused Metal door refused every sealed file at load (`staging d_in=9728 wants 38912 B`). It now loads them. The
operator has not acknowledged that change yet.

## 3. Decisions

Taken, 2026-11-06:
- The engine on the Mac is MLX. The journal, M14: "the operator chose MLX as the engine (decision 2, 2026-11-06), so
  the prototype below was built." Our own command-buffer engine (layer B of the archived design) is set aside.
- The rival's 2.28 GB download, approved before M14 (its prereg).

Open:
1. Fused kernels change roundings: serve under the M6 tolerance (+-0.12 % ppl, as the grouped attention is on CUDA),
   or only what stays bit for bit? No ruling is recorded. The MLX path was built ahead of it and is not bit-exact.
   The M15 prereg writes "served under the M6 tolerance" as if this were ruled; no journal line records that ruling.
2. Acknowledge the int4 wall lift on the Metal door (section 2).
3. Serve a Metal config at all, and whether a Mac reading answers the "consumer GPU" item of `ETAT.md` §5.
4. Lift the Metal refusal of the grouped form (bit for bit, M5: a one-line change in `served.rs` `check_device`) or of
   the kernel form (inside T8, M6).
5. The bundle's form for the MLX path: M15's verdict makes it the bundle's Metal path, its form waits for the operator.
   `ops/metal-package/` still ships the Rust path; its `chat` defaults to the expanded fused arm `lutd+h+pq`.

## 4. What comes next

Ordered by what a step buys on a fundamental criterion (served tok/s on the Mac, quality), then by cost. All local,
$0. Times are *estimated* working time.

| # | step | cost | gate | what it buys |
|---|---|---|---|---|
| 1 | Rulings 1 to 5 above | 0 | operator | a served Metal path, or a clear refusal |
| 2 | MLX path perplexity on the 12-window design, against the Rust expanded form's 12.580769 (M5) | 1 h, a prereg | T8 band, pooled and every window | the quality reading the MLX path lacks; decision 1 needs it |
| 3 | MLX path, the Tetra kernel: about 4.4 ms a token in the off-protocol ablation before M16. Leads: two blocks a lane interleaved, a table of decoded byte quads, a load-time re-encoding (b/param stated, rule 6) | days | microbench, then the M16 arms | the largest cost left at short context |
| 4 | MLX path at long context: it loses 4.6 ms a token from 5 to 4,096 positions where the rival loses 1.9 with the same attention kernel (M15), not explained | 1 day | the M16 L2 arm | decode at 4,096 |
| 5 | Same-head ratio on Metal: an f16-head fused arm | 0.5 day | rule 4 | the raw `lut`/`D` ratio gets its companion |
| 6 | The 8B and the 14B on the Mac; the tiled int4 kernel passes their d_in in the tests | download 7.9 GB (operator), then 1 h | `metalrun gate`, then a prereg | the other two sealed files |
| 7 | GSM8K under the grouped or kernel form on Metal, T8's second criterion | more than a night | prereg | the kernel form's full T8 |
| 8 | Speculative decoding: a block's decode is paid once for 8 rows, so verifying drafts is cheap. Prompt lookup first | days | prereg | short-context decode |

Done from the previous list: the prefill rows kernel (M9, M10) and f16 in and out of the matvecs (M8, a wash). The
Rust attention kernel's speed at short context (x0.864, M6) is not pursued: MLX brings its own attention. `+r`, the
rotation staged in threadgroup memory, is wired with the same bits (`abca2b3`) and has no timed verdict; M14's O1
ran with it.

4E has no Metal port. Metal offers no barrier across threadgroups and no forward-progress guarantee between them, so
a persistent kernel can deadlock. A candle launch costs 1.66 us in a dependent chain (*measured*, M1), and 336 cast
launches a token cost about 0.1 ms (M8): on Metal the launch count is not the wall.

## 5. How the Rust path is held

Every Metal kernel below returns the served kernel's bits, except the attention kernel, which returns the canonical
replay's. None changes a served config.

| piece | where | how it is held |
|---|---|---|
| tiled int4 route | `fused_metal.rs`: `Q4_TILE_COLS`, `int4_tile_for`, `upload_int4_tiled` | `llvq-metal/tests/q4_tiled_matches_served.rs` (8 tests), `llvq-llm/tests/q4_metal_candle.rs` |
| Tetra row-loop twins `_lutp`, `_lutd`, `_lutdp`, `_lutd2`, `_lutd4`, `_lutdt` | `kernels/llvq_tetra48_rows.metal`, compiled after the served unit | `llvq-llm/tests/tetra48_metal_rows.rs` |
| half-io twins (`+h`) | Tetra and int4 kernels storing f16 | the same two test files; 13 mutants, 13 killed |
| prefill kernels `tv_tetra48_metal_lutdr` (`+p`), `tv_q4_metal_rows` (`+pq`) | the same units | 11 mutants: 10 killed, P4 after a test fix; H1 survives by construction (the binding offset is always 0) |
| split-KV attention kernel | `kernels/attn_dec.metal`, `src/attn_metal.rs` | `llvq-llm/tests/attn_metal.rs` (6 tests, to 4,096 positions), a self-check at build; 10 mutants, 10 killed |
| measurement door | `Served::apply_measurement`, used by `metalrun` only | `served_config`, `served_unit`; `Served::apply` and its refusals unchanged |
| `metalrun` | `llvq-llm/src/bin/metalrun.rs` | its bit gate: every arm's 252 projections and head against the reference arm's on the real file |

The night's 37 mutants were all killed (27 matvec, 10 attention). `tv_q4_h.metal` and `llvq_tetra48.metal` stay
byte-identical: the llvq-tetra wheel ships and checks copies of the served shaders. The MLX path is held only by
`ops/mlx_tetra/check.py`: rotation within 1e-5 of four MLX ops, the 8-row and dense prefill within 2e-2 of the
one-row kernel. It has had no mutation run.
