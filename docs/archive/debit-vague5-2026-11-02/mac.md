# Lead "mac": the fused Metal path for the sealed files

Wave 5 deep dive, 2026-11-02, revised the same day against an adversarial review (§10). Read-only on the repository,
$0, no run. Every number carries *measured*, *computed* or *estimated* once. Arithmetic: `code/mac/mac_bounds.py`, a
record of expressions evaluated with `python3 -I`; it reads nothing and writes nothing.

## 1. Verdict

None of the three sealed files can run fused on the Mac today, and the fix is a host-only change that keeps every bit.
`fused_metal.rs` never names `tv_q4_metal_tiled`, so the 12, 17 and 19 int4 `down_proj` records of the 4B, 8B and 14B
are refused at load (`fused_metal.rs:970-977`). Wired, the sealed files should decode at about 45-61, 28-47 and 15-28
tok/s on the M3 Max, 62-131 GB/s of weights a token (*estimated*). They would hold 1.38, 2.76 and 5.04 GB on the device
(*computed*: the loader's host byte count printed on the L40S, `paper-table-2026-09-25.txt`). That is 2.74, 2.70 and
2.73 b/param over the whole model, tables included, against 16.0 for the dense arm (*computed*). The roofline at the
measured 340 GB/s is 247, 141 and 74 tok/s (*computed*). llama.cpp IQ2_XXS reads 142 GB/s on the same Mac (*computed*
from *measured* 114.02 tok/s).

At 4B the fused file should run ×4.2-8.5 faster than our dense Metal arm raw (point ×6.1) and ×1.4-4.3 same-head (point
×2.8) (*estimated*). The dense arm reads 7.5-9.6 tok/s, low-grade *measured*: two commit messages and one journal on
the paper-1 file. The same-head ratio is low for a mechanical reason. Both arms with an f16 head copy the whole table
every token: `model.rs:1395` calls `broadcast_matmul`, which materializes the transposed 778 MB table (candle-core 0.9.2
`tensor.rs:1545-1556`). The dense arm is also slow on this Mac: 60-77 GB/s of weights (*computed*). The same f16
weights run at 282-298 GB/s through torch on mps and 346.8 GB/s through llama.cpp (*computed*). So the raw ratio
measures candle's dense Metal route as much as the format, and it never appears without its same-head companion and
that comparison in GB/s.

What decides the rest has three parts. First, the in-situ rate of the int4 kernels: the 75-250 GB/s bracket alone spans
45-61 tok/s at 4B. Second, whether the e4 objects pass, on Metal, a per-record check against the host oracles and a
logits check against the dense arm. Third, what the 2.4-3.6 ms of non-weight time a token is (*estimated*: the q5
step's ~5.2 ms rest minus its q8 head). Beyond the wall, attention that reads the cached heads is worth little at short
context and ×2-4 at 8k (*estimated*). It is a Metal tolerance event, and candle 0.9.2 already ships a Metal SDPA with
native GQA. A Rust rows kernel would cut prefill about ×2-3 (*estimated*) and lift the 256-row cap that blocks MMLU and
long chat turns. It is new MSL text gated by G-D against the served `_lut` kernel, not a port of `llvq-tetra`'s bodies.

The Mac is a second local target. Whether a Mac reading answers ETAT §5's "consumer GPU" item is the operator's ruling,
not this lead's. Only the expanded config can run here: the 4B's served configuration is the grouped one since
2026-11-02 (`docs/ETAT.md:20`), which Metal refuses by name (`served.rs:275-289`).

The first steps cost $0 and minutes. Kernel tests of the tiled path at six widths, with fast math on and off. A binding
test through `MetalRuntime::upload_int4` and `matvec_int4`. An isolated int4 bench at the served shapes. Then one local
run on a quiet machine (`prereg-mac-1.md`): the oracle, a per-record gate, a logits gate, and the sealed 4B fused and
dense in one process. That gives the first journaled Mac speed of a sealed file.

## 2. What exists today

| where | what it does | status |
|---|---|---|
| `llvq-llm/src/fused_metal.rs:80` | `THREADGROUP_LIMIT = 32_768`, read once from `maxThreadgroupMemoryLength` on 2026-09-20 | constant, by design |
| `fused_metal.rs:93-102` | `source_of`: entry point to MSL file. `"tv_q4_metal"` is there, `"tv_q4_metal_tiled"` is not | the host cannot compile the tiled kernel |
| `fused_metal.rs:397` | `new_library_with_source(&src, None)`: Apple's default options, fast math on | the shipped compile |
| `fused_metal.rs:949-994` | `upload_int4`; `:970-977` refuses `d_in * 4 > THREADGROUP_LIMIT` | the wall |
| `fused_metal.rs:997-1003`, `:1016-1018` | `matvec_int4` widens `x` to f32; `Int4Op::cpu_fwd` bails | no CPU arm for int4 |
| `fused_metal.rs:1020-1057` | `Int4Op::metal_fwd`: `tv_q4_metal`, threadgroup length `d_in * 4`, `d_out * 32` threads, groups of 256 | works for d_in ≤ 8,192 |
| `fused_metal.rs:1539-1553` | the int4 upload loop of `load_resolved`, where the sealed files die | runs before `finish()`, so no e4 q4 head has run on Metal |
| `fused_metal.rs:1225-1285` | `HeadOp`: stages `d * 4` ≤ 32,768 B, one launch a row | 10,240 / 16,384 / 20,480 B at 4B / 8B / 14B: passes |
| `fused_metal.rs:257-296`, `:1385-1413` | `MetalRuntime::new(.., keep_host)` keeps host copies for `cpu_fwd`; `from_tables`, the served load, sets `host: None` | the served load cannot run the oracle |
| `fused_metal.rs:777-784, 889-891` | rotation scratch in device memory | no 32 KB wall for any size |
| `llvq-llm/kernels/llvq_rot.metal:104` | `LLVQ_ROT_KMAX 32` | covers k = 5, 19 (4B), 1, 3 (8B), 5, 17 (14B) and 25 (32B) |
| `fused_metal.rs:1346-1348`, `:1702` | `rows_per_launch() = 1`, `Prefill { rows: 1 }` | no Metal rows kernel in Rust |
| `fused_metal.rs:1498-1502` | tile 64 hard-coded, "no measured row for Apple" | tile moves 1-4 % on Metal (`5a75338`) |
| `llvq-llm/kernels/tv_q4_h.metal:176-223` | `tv_q4_metal_tiled`; barriers at `:202` and `:206`; argument `:152-170` | gated only from Python |
| `tv_q4_h.metal:162`, `:45-63` | cites `tests/q4_tiled_matches_served.rs`, which does not exist; "no Metal host yet" | stale |
| `llvq-llm/src/model.rs:1393-1396` | `Head::Dense` projects by `broadcast_matmul(&t.t())` | candle copies the table every token |
| `model.rs:85-88`, `:1609-1707` | `Capture`: four activation points a layer (q/k/v, o, gate/up, down inputs) | the hook a per-record gate needs |
| `model.rs:2316` | `generate_phased`: embed, blocks, head, argmax, each closed by `Device::synchronize` | backend-independent, runs on Metal today |
| `llvq-tetra/llvq_tetra/fused.py:229-307`, `mps.py:283-317` | `Int4FusedLinear` calls `mps.q4_matvec`, whose default `rows=True` runs the hand-written `tv_q4_rows_cs` even for one row (tile 1,024). `tv_q4_tiled_cs`, the checked edit of the served tiled kernel, runs only with `rows=False`, the single-row side of G-D | the 4B runs fused on mps |
| `llvq-tetra/llvq_tetra/mps.py:90-164`, `:222-279` | `TETRA_ROWS_KERNEL`, `Q4_ROWS_KERNEL`: hand-written bodies in Python strings. Only `tetra48_dot_q` (`:172-184`) and the single-row `_cs` variants are checked edits | gated by G-D 252/252 (`567f6fe`, deviation É1 of `hf-metal-bloc`) |
| `llvq-tetra/llvq_tetra/integration.py:25-28` | `INT4_TILE_COLUMNS = 2048`, "the values the kernels were validated with" | the validated tile |
| `llvq-llm/src/served.rs:275-289` | `check_device` refuses `grouped` and `shared` on Metal | the Mac runs the expanded config |
| `llvq-llm/src/bin/chat.rs:183-205` | loads through `served::load_resolved`, never calls `Served::apply` | prints the config's attention, does not apply it |
| `llvq-llm/src/bin/fusedrun.rs:649-653` | CUDA only | no Metal decode bench with a protocol exists |
| candle-metal-kernels 0.9.2 `kernel.rs:185`, `metal/commands.rs:60` | `CANDLE_METAL_ENABLE_FAST_MATH` sets the math mode of every candle kernel; `CANDLE_METAL_COMPUTE_PER_BUFFER` moved the q5 step from 33.9 to 27.7 tok/s (`41e6628`) | uncontrolled today |
| candle-core 0.9.2 `metal_backend/device.rs:203-270` | uploads are exact; op outputs are rounded to the next power of two and pooled until a flush | host byte counts are not device allocations |

What works: the q5 objects (36 `v_proj` int4, all under the wall) run fused on Metal. Their REPL rate went from 33.6 to
48.1-49.8 tok/s, with 60 tokens identical to the dense arm (*measured*, commit messages only: `976549b`, `5a75338`,
`31a5acb`, `5ec72b0`). What is missing: the wiring, a Metal bench binary, any journal of a Metal speed, and the 8B and
14B sealed files on this Mac. Only `~/q4b-sealed-2026-09-23/qwen3-4b-sealed.bin` (1,418,224,685 B) is local. The Hub
cache holds the Qwen3-4B checkpoint (snapshot `1cfa9a72`), the 8B, 14B and 32B checkpoints and the WikiText-2 test
parquet, and no Qwen3-0.6B.

## 3. Design

### 3.1 The wall: what the branch needs to keep bits

The audit's step 4 is being implemented in another branch. This section gives it the bit-keeping shape, the traps and
the gates, not a second plan.

Route a record to the tiled kernel only when it does not fit. Every q5 record keeps its exact code path.

```rust
// fused_metal.rs (sketch; full draft in code/mac/fused_metal_wall.rs)
pub const Q4_TILE_COLS: usize = 2048; // llvq-tetra's validated value; 8,192 is the A/B's other arm

"tv_q4_metal" | "tv_q4_metal_tiled" => Ok(SRC_Q4),        // source_of, :97

pub struct MetalTuning { pub q4_tile_cols: usize, pub keep_host: bool, pub null_kernels: bool }
// validate() at the start of the load: 0, not a multiple of 256, or x4 over the limit is refused by name,
// whether or not a record needs the tile.

pub struct MetalInt4Proj { /* ... */ tile_cols: Option<usize> /* None: whole activation */ }

// upload_int4_tiled(.., tile), replacing :970-977; upload_int4 calls it with the runtime's tile
let tile_cols = (d_in * 4 > THREADGROUP_LIMIT).then_some(tile);

// Int4Op::metal_fwd, replacing :1031 and :1045
let (kname, cols) = match p.tile_cols { None => ("tv_q4_metal", p.d_in), Some(c) => ("tv_q4_metal_tiled", c) };
if let Some(c) = p.tile_cols { set_param(&enc, 7, c as u32); }
enc.set_threadgroup_memory_length(0, cols * 4);
```

Geometry: `d_out * 32` threads in groups of 256, so 8 rows a group and one simdgroup a row, as today. Threadgroup
memory is `tile_cols * 4` B, 8 KB at 2,048. At 2,048 columns the 4B, 8B and 14B `down_proj` take 5, 6 and 9 tiles; at
8,192 they take 2, 2 and 3. Registers do not change. Two barriers a tile.

The default is 2,048, and the reason is residency. A tile of 8,192 columns asks 32,768 B, the whole limit, so likely
one such threadgroup fits a core (*estimated*). This Mac has already paid for that mechanism: pinning 18.7 KB of
threadgroup memory capped the groups resident on a core and cost +8 to +23 % (`82c029c`). `llvq-tetra` ships 2,048 as
the value its kernels were validated with. `prereg-mac-1` times 8,192 as the other arm and moves the default only on a
ratio range wholly on one side. Bits do not move either way.

Exactness, by construction. A tile of a multiple of 256 columns is a multiple of 32 words, so lane l still walks words
l, l+32, l+64 in order, and the butterfly is unchanged (`tv_q4_h.metal:152-170`). The f32 store narrows in candle by
round-to-nearest-even, the rule of the CUDA `f2h` (`fused_metal.rs:1333-1342`). The tiled path should therefore equal
the CUDA `tv_q4_h` record for record, as the untiled one claims to.

Traps, each seen before or readable in the code:
- A zero `tile_cols`, or buffer 7 left unbound, makes `c0 += 0` loop forever: a GPU hang, not an error. Validate the
  tile when the load starts and always bind 7 on the tiled path. This one is guarded structurally, never planted as a
  mutant on the operator's machine.
- A forgotten threadgroup length writes zeros (`fused_metal.rs:24-25`). The tiled path sets `tile * 4`, never
  `d_in * 4`, which is above the limit.
- Do not touch `tv_q4_h.metal` in this lot. The `llvq-tetra` wheel ships a copy and checks its sha256
  (`llvq_tetra/metal.py:62-79`, `hfpack.rs:750-770`). Both copies are byte-identical today (`cmp`). The stale header and
  the phantom test name go in a separate lot with `bin/tetratables`.
- Mutants of the tiled kernel are partitioned at the next entry point. `tv_q4_metal_tiled` comes first in the file, so
  a split on the name hits both kernels and voids the mutant (`hf-metal-m2-4b-2026-09-30.txt`, controls).
- Race mutants die only above the machine's concurrency. The int4 barrier mutant of the untiled kernel failed five of
  seven GPU tests and passed two (`llvq-metal/tests/q4_matches_host.rs` header). The rotation's dropped per-row offset
  lived up to 40 rows and died from 48 (`a10c92c`). So both barriers of the tile loop get a mutant. The first one
  (`:202`) is new with the tiled kernel: it stops the next fill from racing a simdgroup still reading the previous
  tile. Its fixtures need many tiles and many threadgroups (16 tiles over 320 groups, 9 tiles over 640), repeated 16
  times, with the kill rate recorded.
- Fast math. The shipped path compiles with Apple's defaults, fast math on (`fused_metal.rs:397`). The gate must hold
  either way: `q4_matches_host.rs` carries that test for the untiled kernel, and the tiled file adds its twin. A bench
  compiled with fast math off over-predicted in situ (`31a5acb`).
- Bank conflicts. Lanes read `xs` at a stride of 8 floats. K-1 measured no bank-conflict effect on Apple: a stride of 28
  floats ran 0.4 % slower than 24 (`docs/format-noyau.md` §9). The risk is small (*estimated*).

The gates (L2), equality of every f32, never a tolerance:
1. `llvq-metal/tests/q4_tiled_matches_served.rs` (`code/mac/q4_tiled_matches_served.rs`), the file the kernel header
   promises. Tiled equals untiled at d_in 2,560, 4,096 and 5,120, tiles 256, 2,048 and 8,192. Tiled equals a host
   lane-order reference from `Int4Matrix::to_f32` at 9,728, 12,288 and 17,408. Fast math on and off give the same bits.
   Sixteen repetitions at many tiles give the same bits. This is stronger than `llvq-tetra`'s check past the wall,
   which was 2.90e-07 relative against the dense reconstruction.
2. `llvq-llm/tests/q4_metal_candle.rs` (`code/mac/q4_metal_candle.rs`), the binding, which no test covers today for the
   int4 arm, tiled or not. `MetalRuntime::upload_int4_tiled` and `matvec_int4` at the six widths against the same
   reference, an activation bound at a non-zero offset, and the refusals by name.

### 3.2 The speed the wall gives, and its bound

Bytes a token (*computed*, wave-1 U1 for the totals, the configs for the int4 records at 4.25 b/weight). "Weights a
token" is the projections plus the head table, the bytes a decode step reads. GB/s in this lead is that figure times
tok/s. Device bytes are a memory figure only: at 8B and 14B they include the untied input table, of which a step reads
one row.

| size | weights a token | Tetra | int4 | head | int4 past the wall | device bytes |
|---|---|---|---|---|---|---|
| 4B | 1.377 GB | 0.749 (54 %) | 0.409 (30 %) | 0.219 | 0.159 GB, 12 records | 1.38 GB |
| 8B | 2.414 GB | 1.529 (63 %) | 0.535 (22 %) | 0.350 | 0.455 GB, 17 records | 2.76 GB, +14 % |
| 14B | 4.606 GB | 2.600 (56 %) | 1.568 (34 %) | 0.438 | 0.900 GB, 19 records | 5.04 GB, +9 % |

The q5 step anchors the rates. At 48.5 tok/s it is 20.6 ms: matvec 12.66 ms over 252 launches (projections only),
rotation 2.72 ms over 144 launches, and a rest of about 5.2 ms (commit `5a75338`). The q8 head (0.413 GB) sits inside
that rest. At 150-250 GB/s it takes 1.65-2.76 ms, which leaves 2.4-3.6 ms of non-weight time (*estimated*). `5ec72b0`
later read 49.8 tok/s with the same kernels, inside the ~5 % spread `5a75338` reports for single readings. The 216
Tetra matrices carry 0.926 GB, so Tetra runs at 74-77 GB/s (*computed*), 22 % of the 340 GB/s. The kernel header calls
that a latency bound: 3.3 blocks a lane at d_in 2,560 and a three-deep dependent load chain
(`llvq_tetra48.metal:1013-1024`). A rotation fits t(n) ≈ 3.9 + 0.0032·n µs (*estimated*: 12 µs at 2,560 from
`31a5acb`, total from `5a75338`).

The int4 kernels have never been timed in situ. The bracket is 75 GB/s (as slow as Tetra) to 250 GB/s (*estimated*).
The upper end is what hand-written matvec kernels with a trivial decode reached on this Mac: Slot32 238-248, FP16
334-353 GB/s (*measured*, `k1-metal-2026-08-05.txt`). L2b brackets it at $0 before `prereg-mac-1` is stamped.

| size | step after the wall | tok/s | GB/s of weights | roofline at 340 GB/s | at llama.cpp's 142 GB/s |
|---|---|---|---|---|---|
| 4B | 16.4-22.3 ms | 45-61 | 62-84 | 247 | 103 |
| 8B | 21.4-36.0 ms | 28-47 | 67-113 | 141 | 59 |
| 14B | 35.3-65.3 ms | 15-28 | 71-131 | 74 | 31 |

Step and tok/s are *estimated*: Tetra at 74-77 GB/s at 4B and 75-120 at the 8B and 14B shapes, int4 at 75-250, the q4
head at 150-250, rotations by the fit at 96, 127 and 101 launches a token (*measured* on the card,
`paper-table-2026-09-25-brut/paper-served-*-r2/fused-q4-256.txt:3`), and the non-weight rest of 2.4-3.6 ms scaled by
layers. The last two columns are *computed* bounds and anchors, not forecasts. The tiled records alone take 0.6-2.1 ms
at 4B and 3.6-12.0 ms at 14B. Their rate matters most where the product lives.

**The f16 head, and why the same-head ratio is low.** Arms Fh and D project through `Head::Dense`. Its
`broadcast_matmul` turns the `[d, vocab]` view into a broadcast `[1, d, vocab]` and calls `contiguous()`, so every token
copies the transposed table (candle-core 0.9.2 `tensor.rs:1545-1556`). At 4B that is 0.778 GB read and written, then read
again by the gemm: 2.33 GB of traffic (*computed*). On the L40S it costs 9.4 ms a token over the q4 head (*computed*
from `paper-table`: 55.0 against 113.8 tok/s), 248 GB/s or 29 % of 864. At 15-40 % of 340 GB/s it would take 17-46 ms on
the M3 Max (*estimated*). So Fh should read 15-31 tok/s and Fh/D ×1.6-4.1 (*estimated*). The operator's ML audit of
2026-11-02 found the same copy compressing paper 2's same-head ratios. Removing it is one line (a 2-D `matmul`, as
`candle_nn::Linear` does), but it moves the D and f16-table arms that published ratios rest on. That is an operator
decision outside this lead. Step 5 of `prereg-mac-1` measures the head's share on Metal.

Rule 5 holds throughout. llama.cpp and torch are other stacks, compared in GB/s only. A ratio is formed only against our
own dense Rust arm in the same process on the same Mac. The dense arm reads 60-77 GB/s of weights, 72-92 counting the
head copy (*computed*), against 282-298 for torch on mps (`hf-chat-4b-2026-10-04.txt:56`, "not a benchmark") and 346.8
for llama.cpp f16 (`m3-iq2-metal-2026-08-30.txt`) on the same f16 weights.

### 3.3 Beyond the wall

**(b1) Attention that reads the cached heads.** The expanded form moves about 16.9 GB of cache a token at 8,192
positions on the 4B (*computed*, `preregistration-gqa-ab-2026-11-01.md` §1). Weights plus that traffic bound the Mac at
18.6 tok/s there (*computed*: 18.3 GB at 340 GB/s). Reading the heads in place leaves 1.38 + 1.21 GB, a 131 tok/s bound
(*computed*). Two forms can do it on Metal:
- `grouped_attention` (`model.rs:571-596`), plain candle ops on a `Grow(256)` store. candle 0.9.2's MLX gemm accepts the
  transposed key view with head stride W·hd and takes the batch stride from `stride[-3]` (`mlx_gemm.rs:297-322,
  360-375`), so no copy is expected (*computed* from the source, never run).
- candle-nn 0.9.2's Metal `sdpa` (`ops.rs:1040-1106, 1219-1257`). It runs a vector kernel for 1 to 8 query rows with
  native GQA, two passes from 1,024 keys, and is handed the kv-head stride (`sdpa.rs:244-270`). That is one launch a
  layer in decode instead of about six. A new `AttnFn` of the existing signature wraps it.

Neither form is bit for bit. Both change matmul shapes or move to an online softmax, so each is a tolerance event.
`served.rs:262-271` already says why the L40S tolerance does not transfer. Value: under 5 % or negative at short
context, since the preallocated store lost 3.7 % at 120 tokens on Metal (46.5 against 48.3 tok/s, `5a75338`). At 8k it
is ×2-4 (*estimated*). A microbench of the three forms on synthetic caches (L7a, minutes) says whether the tolerance run
is worth its hour. The paired measurement needs a Metal host for `decode_nll_paired`, which is library code hosted only
by `fusedrun` on CUDA.

**(b2) A rows kernel.** The prefill fans out one launch a row, and one call is capped at 256 rows (`model.rs:699,
746-753`). Chat feeds a whole turn in one call (`chat.rs:290-296`), so a user turn above 256 tokens errors on the fused
Metal path (*computed* from the code, not run). An 8,192-token prompt would take about 2-2.3 min at 15-17 ms a row
(*estimated* from the q5 matvec and rotation share). The design:
- New MSL text in a new file, `kernels/llvq_rows.metal`, compiled as the served unit's text followed by the rows text
  (`concat!` of the two `include_str!`), under new `source_of` entries (`tv_tetra48_rows_lut`, `tv_q4_rows`). The served
  single-row pipelines still compile from their own files, so their register allocation cannot move. The wheel's
  checked copies stay byte-identical (`llvq_tetra/metal.py:62-79`).
- Built around `_lut`'s decode. The rows kernel calls the served helpers (`f1r_load`, the 64-float table, `warp_sum`,
  `q4_deq`) and owns only its loops. A direct port of `llvq-tetra`'s bodies would inherit the faithful `tetra48_dot`,
  which the step-level A/B put ×1.480 [1.436, 1.490] behind `_lut` (`5f439df`).
- Exactness by gate, not by construction: the loops are new code, as `llvq-tetra`'s deviation É1 says of its own. G-D
  compares each output row with `tv_tetra48_metal_lut`, `tv_q4_metal` or `tv_q4_metal_tiled`, bit for bit, on all 252
  records of the 4B with captured activations, each kernel with a planted mutant.
- Budget: 4 × 64 blocks × 24 × 4 B = 24,576 B for Tetra at tile 64, above the 18.7 KB that capped residency in
  `82c029c`, so tile 32 (12,288 B) is the other arm. Int4: 4 × 1,024 × 4 = 16,384 B. Registers `acc[4]` and `q[6]`.
- Integration: `MetalLattice::rows_per_launch` returns 4 and `matvec_rows` gets a `RowsOp`. `prepare_rows` already
  rotates a chunk in one launch. `Int4Proj` has no rows method (`device.rs:106-123`), and int4 is 30 % of the 4B's
  bytes, so the trait gains one with a default of 1.
- Value: prefill about ×2-3 (*estimated*). The cap becomes `MAX_PREFILL_ROWS = 4096`, so MMLU prompts (up to 3,096
  tokens) and long chat turns run through the Metal kernel. It is also the brick of any Mac speculation.

**(b3) The non-weight time.** About 2.4-3.6 ms a token is neither weights nor rotation (*estimated*, §3.2). Removing
290 of the 688 launches a token changed nothing measurable (`5a75338`), and larger command buffers made things slower
(33.9 to 27.7 tok/s, `41e6628`). Instruments is not available: this Mac has the Command Line Tools only, and `xcrun
--find xctrace` fails (checked). Two $0 instruments, in order:
1. `Qwen3::generate_phased` on F, Fh and D inside `metalrun`, off protocol (`prereg-mac-1` step 5). It splits embed,
   blocks, head and argmax with `Device::synchronize` fences. It settles the head's share. Fences serialize the work,
   so this pass cannot see idle gaps between dispatches.
2. A null-kernel arm N, the analog of `nullk` (`docs/format-noyau.md` §6). The same dispatches with the same bindings
   and empty bodies, unfenced. N's step prices host encoding, launches and candle's own ops with their overlap intact,
   and F − N is the weight kernels' bodies in situ. Design: a new unit `kernels/llvq_null.metal` with one entry point
   for each served signature, writing `y = 0` from lane 0 of each row; `MetalTuning { null_kernels: true }` maps each pipeline
   name to its twin; the rotation stays real. Tokens become id 0 at every step, which moves no timing, since nothing
   branches on data. Never served.

Installing Xcode for Instruments is an operator decision.

### 3.4 What Metal lacks, and the brick on Metal

| CUDA feature the wave-2 plans lean on | on this Mac |
|---|---|
| persistent kernel with spin waits across CTAs | no documented forward-progress guarantee across threadgroups. `MTLForwardProgressUsage` (automatic, weak, simdGroupParallel) appears in the macOS 27 SDK with undocumented cases (developer.apple.com). This Mac runs macOS 26.6.2 |
| PRMT byte permute | absent; emulating it cost about 240 instructions a block, removing it gave ×1.245 (`82c029c`, `5f439df`) |
| dp4a, int8 MMA | no int8 dot in MSL (UNVERIFIED, plan C says the same) |
| 99 KiB of shared memory, clusters, PDL | 32 KiB a threadgroup, no clusters, no PDL |
| CUDA Graphs | indirect command buffers are the analog. A2 is closed in the core (rule 3), and its reason (a fixed KV window) carries over. Not planned |
| tensor cores | `simdgroup_matrix` 8×8 tiles (candle's own gemm and SDPA use them). The M5 has a Neural Accelerator in each GPU core, programmable through Metal 4 Tensor APIs (Apple newsroom, 2025-10-15). The M3 Max has none |

The brick's integer argument holds on Metal, with one hardware fact unverified. Tetra coordinates satisfy |v| ≤ 22,
since a shell m ≤ 31 has norm² 16m ≤ 496 (`fused_metal.rs:670-675`). So a block has Σ|v| ≤ √(24·496) ≈ 109
(*computed*). An activation plane of 11 bits is exact in f16 (|a| ≤ 2,047). A per-block, per-plane dot is then an
integer of magnitude at most 109 × 2,047 = 223,123 < 2²⁴ (*computed*), and products stay below 2¹⁶. Every partial
sum is exact in f32, so the result does not depend on the order in which `simdgroup_multiply_accumulate` adds. The
per-block scales (gscale × invnorm) and the recombination of the two planes then round, and a canonical order makes
that replayable on the host. Three limits:
- Whether Apple's 8×8 multiply-accumulate is exact on exact integer inputs is implementation behaviour, UNVERIFIED. A
  micro-test settles it at $0, compiled at run time with no Xcode (L9, runnable at any time).
- Quantizing the activation into planes changes the numbers, so a brick is a tolerance event (T8).
- At k = 1 an 8×8 tile wastes 7 rows of 8, so a brick pays only for rows, verify or prefill at k ≥ 8. The M3's f16
  rate equals its f32 rate (UNVERIFIED), so the gain is operand reuse, not FLOPs.

## 4. Deciding unknowns

A kill is the operator's decision in every row. Each row ends with "otherwise: journaled, the operator decides".

| # | unknown | best value now | decision rule |
|---|---|---|---|
| U1 | int4 kernels' in-situ rate on the M3 Max | 75-250 GB/s (*estimated*); never timed | F's median ≥ 50 tok/s: the int4 kernels are not the next lever. 30 ≤ F < 50: the phased split and the L2b bench attribute the gap before any 8B or 14B timing. F < 30: journaled, no sentence beyond the journal, L6 first. Otherwise: journaled, the operator decides |
| U2 | the e4 objects fused on Metal against oracles and the dense arm | q5: 60 tokens identical (`976549b`); e4 never loaded | Record gate 253 of 253 and G-L ≥ 248 of 256: the path is right at kernel and model level. Record gate fails: the record named is the defect, fixed under a deviation. Record gate passes and G-L fails: a wiring defect (routing, rotation keys, head), diagnosed before any speed is quoted. Both pass and greedy tokens diverge before 128: the position is reported, not a defect by itself. Otherwise: journaled, the operator decides |
| U3 | what the 2.4-3.6 ms a token is | after the q8 head (*estimated*, `5a75338`) | Phased pass: the head and the blocks split it. Null arm N at ≥ 25 % of F's step: host encoding and candle's ops are a lever (launch count, f16 kernels, and an operator reading of rule 3 before any replay). N < 25 %: kernel work first (§6 item 7). Otherwise: journaled, the operator decides |
| U4 | 8B and 14B sealed files on this Mac | absent (`ls ~/q8b-14b-sealed-2026-09-23`) | Downloaded after `prereg-mac-1` reaches R2, checked by sha256: proceed to L5. A digest mismatch: deleted and fetched again. Stalls: the Hub client resumes (`hf buckets cp` stalled 3 times in 4 in October, `5c6f5e2`). Otherwise: journaled, the operator decides. L2's synthetic fixtures at d_in 12,288 and 17,408 already cover the 8B and 14B kernel shapes |
| U5 | grouped or sdpa on Metal: tolerance and value | refused by name; no Metal reading | A form inside ±0.12 % with its speed range wholly above 1.000 at 2,048: the operator may lift the refusal for it. Outside the band: that form stays refused. Inside the band with a speed range straddling or below 1.000: it stays refused, for want of a reason. Otherwise: journaled, the operator decides |
| U6 | rows kernel on Metal: G-D and prefill | G-D 252/252 on torch-mps (`567f6fe`); no Rust kernel | G-D in Rust on 252 records and a TTFT range wholly above 1.0: rows 4 on Metal. G-D fails: rows stay 1 and the defect is journaled. G-D passes and the TTFT range straddles or falls below 1.0: rows stay 1. Otherwise: journaled, the operator decides |
| U7 | `simdgroup_matrix` exact on integer f16 inputs | UNVERIFIED | Exact on every fixture, sums near 2²⁴ included: the premise holds, and any brick stays a T8 and an operator decision. Any inexact output: no brick on Metal. Otherwise (the test cannot be built): journaled, the operator decides |

## 5. Validation ladder

Nothing here is a paid job. Every local step needs the operator's go. Builds run under `nice -n 15` with `cargo -j 4`
(the handover's limits, `passation-debit-trims-2026-11-01.md:192`). A measurement runs the built binary by path, its
sha256 printed, never `cargo run`, under `nice -n 10` (METHODE §6). Timing steps need a quiet machine: no build
anywhere, no other GPU application, the browser closed, no other agent session. They wait 30 minutes after the last
build, because the same binary read 49.7 tok/s cold and 23.1 after three hours of compiling (`5f439df`). `LLVQ_THREADS`
is not used: only `smoke` and `errmap` read it.

| step | settles | method | class | prereg | decision rule |
|---|---|---|---|---|---|
| L0 | what exists, what is missing | this document | $0 | no | done |
| L1 | the wiring compiles and refuses by name | branch: §3.1 code, the two test files, `metalrun` | $0, code | no | n/a |
| L2 | the tiled kernel and its binding are exact | `cargo test` of `q4_tiled_matches_served.rs` and `q4_metal_candle.rs`, then the mutants of §6 item 1 | local, minutes, light GPU | no (gates) | every equality holds and every mutant fails a test: green. A race mutant that never fails in 16 repetitions: widen the fixtures. A hang or a mismatch: stop and fix the binding. Otherwise: journaled, the operator decides |
| L2b | the int4 kernels' isolated rate and the tile's residency effect | `q4bench` (`code/mac/q4bench.rs`): `tv_q4_metal` at 1024×2560 and 2560×4096, `tv_q4_metal_tiled` at 2560×9728, 4096×12288, 5120×17408 for tiles 1,024, 2,048 and 8,192; fast math on as shipped; cold buffers rotated past the 48 MB SLC; one process, same order every round, 2 discarded + 7 timed | local, minutes | a one-page note stamped before it runs | informs only: isolated benches over-predicted in situ by 9-18 % (`31a5acb`) and 15 % against 5 % (`5ec72b0`). Its range replaces the 75-250 bracket in `prereg-mac-1`'s predictions before that file is stamped. Nothing is decided on it |
| L3 | first load of a sealed file fused on Metal: oracles, logits, tokens, speed | `prereg-mac-1.md`: oracle; a check process (record gate, logits gate against D); then F (tile 2,048), F' (8,192), Fh (f16 head), D (dense) in one process, 2 discarded + 7 timed rounds of 128 tokens, then the phased pass | local, about 20 min of machine time after a 30-min wait; 13 GB of unified memory in the timed process, about 32 GB in the oracle's own | yes, stamped | in the prereg, §8 |
| L3b | task-level agreement (optional) | `gsm8k` with `LLVQ_CONFIG=e4` on the 50 pilot problems (prefill chunked at 256, `gsm8k.rs:284-296`), paired by `gsm8kpair` with `docs/data/gsm8k-dumps/pilot-4b-sealed-metal.jsonl` | local, about 10 min (*estimated*) | yes | ≥ 45 of 50 correctness labels shared and a paired interval containing 0: consistent. Otherwise: per-record diagnosis, and the operator decides |
| L4 | the 8B and 14B files on the Mac | after L3 reaches R2. `Pier-Jean/Qwen3-8B-LLVQ-Tetra-sealed` and `-14B-`, 2.82 + 5.09 GB, sha256 against the digests written on this Mac (`~/q8b-14b-sealed-2026-09-23/sha256-*.txt`) | $0, network | no | U4 |
| L5 | 8B and 14B fused speeds and oracles | `prereg-mac-2.md`, its predictions re-derived from L3's measured F before stamping | local, about 25 min, 45-55 with the optional dense and f16-head arms (*estimated*) | yes | in the prereg |
| L6 | the non-weight time | the null-kernel arm N against F in `metalrun null`, off protocol, after L3's phased pass | local, minutes | a one-page note | U3 |
| L7a | whether the attention forms are worth L7 | microbench of expanded, `grouped_attention` and candle's `sdpa` on synthetic caches, 8 kv heads, hd 128, L = 5 to 8,192, rotated over 36 layer caches so the SLC does not hold them. An `#[ignore]` test inside `model.rs`, so the private forms need no new API | local, minutes | no | informs only. A form at least ×1.3 faster than expanded at 2,048 and 8,192: L7 is worth its hour. Otherwise L7 waits, and the operator decides |
| L7 | attention forms on Metal: tolerance and speed | expanded, grouped, sdpa: paired WikiText-2 decode ppl and top-1 agreement on 4 windows of 4,096, the prefill path chunked at 256 rows, and speed at 5 and 2,048 | local, about 1 h for 4 windows (*estimated*); the card's 12 windows would take 2.3-3.3 h | yes | U5. A 4-window interval that crosses a band edge: the 12-window design under a new prereg. Otherwise U5 decides |
| L8 | rows kernel | G-D in Rust on 252 records, then TTFT at 256 and 1,024 tokens with both arms through `generate_timed` at the same chunk of 256 rows, rows 1 and 4 interleaved | local, about 30 min | yes for TTFT | U6 |
| L9 | brick premise | `simdgroup_multiply_accumulate` on integer f16 fixtures against exact integers. Independent of every other step, runnable at any time | local, minutes | no | U7 |

## 6. Build plan after validation

1. **Wiring and gates**, half a day to a day. The code of §3.1, `q4_tiled_matches_served.rs`, `q4_metal_candle.rs`, and
   refusals by name for tile 0, a tile not a multiple of 256, a tile × 4 over the limit, and a `gpr` mismatch. Mutants
   to kill: lane stride 31, the `- c0` dropped from the `xs` index, `chi` not clamped, the first barrier removed, the
   second barrier removed, every tile restarting at word 0, buffer 7 bound to half the tile, the threadgroup length set
   to `d_in * 4` on the tiled path. The two barrier mutants are races: 16 repetitions, kill rate recorded.
2. **`metalrun`**, about a day (`code/mac/metalrun.rs`). Subcommands `check`, `time` and `null`. Every `LLVQ_*`,
   `CANDLE_*`, `MTL_*` and `METAL_*` variable refused, with an empty allowlist printed in the header. Arms named by the
   binary from the config file. Same order every round, 2 discarded and 7 timed, ratios formed round by round. Host
   copies for the record gate. The logits gate on a WikiText-2 window. The device's own allocation beside the host
   count. Load samples between rounds. The phased pass. It stays out of the CUDA image, so the two-lists trap does not
   apply. It must still build on Linux as a stub, like `fusedrun.rs:649-653`, and pass the Linux type-check of
   METHODE §6.
3. **Journal**, half a day: the journal of L3, the configs README's Metal paragraph, CLAUDE.md's commands. ETAT §5 is
   not edited from a Mac run before the operator rules on what a Mac reading answers. Fixing the shader header goes in
   its own lot, with `bin/tetratables` and a new wheel digest.
4. **Chat**, half a day, 0 bits, operator ruling: feed a turn in chunks of the row cap, and apply or refuse the config's
   attention instead of printing it.
5. **Rows kernel**, 2-3 days, gate L8.
6. **Attention form on Metal**, 1-2 days plus L7, behind the existing refusal until the run passes.
7. **Bit-identical kernel work**, each an interleaved A/B with identical tokens required: float4 reads of `xs` in the
   int4 kernels; an unstaged int4 kernel reading `x` from device memory in a new `.metal` file (no wall at any width);
   f16 input and output in the kernels, which removes 336 conversion launches a token on the e4 4B (*computed*: 168 + 84
   + 84) and needs the exhaustive `f2h` gate the headers ask for, though 290 such launches removed on q5 changed
   nothing measurable; two rows a simdgroup with 32 lanes each for Tetra, which doubles independent chains without
   touching a row's order. The `_ilp` precedent (+5 % in situ against +15 % on the bench, `5ec72b0`) caps what to expect.

## 7. Rules map

- **T8.** The wall is not a T8 event: bits are kept by construction. The rows kernel is not one if G-D holds. Grouped
  or sdpa attention on Metal, a lane-count change (the `_ilp` precedent), a brick and a GEMM prefill on Metal are each
  a tolerance event for the operator.
- **A2.** Indirect command buffers are the Metal replay mechanism. Rule 3 closes CUDA Graphs in the core, and the ETAT
  §6 reason (a fixed KV window) carries over. Not proposed. Whether Metal falls under the closure is an operator reading.
- **ETAT §6, "a mixed Tetra plus int4 bench arm"** (operator, 2026-10-06): no such arm here. The served files are timed
  end to end.
- **Rule 4.** Raw against the dense arm and same-head (Fh, fused with the dense arm's f16 head), always together, with
  the head copy named while it stands.
- **Rule 5.** No ratio across stacks (llama.cpp, torch) or cards (L40S, A100). GB/s only. No sentence sets a Mac figure
  beside the card's 133.3 tok/s without naming both attention forms and both machines.
- **Rule 6.** Memory in b/param for the whole model, tables included: 2.7320, 2.6953 and 2.7305 by `rtbits` over the
  files (*computed*, `configs/README.md`), 2.74, 2.70 and 2.73 by the loader's host count, and the device's own
  allocation beside it.
- **Rule 7.** Medians with ranges formed round by round, never a quotient of minima.
- **Rules 1 and 2.** A go for every local run and for the download. Stamp before the first measurement.
- **Rule 10.** `oracle` on Metal first, on the cached Qwen3-4B or on a Qwen3-0.6B whose download the operator allows.
- **Operator rulings owed.** Does a Mac reading answer ETAT §5's local-hardware item? Lift the Metal refusal of a form
  after L7? Fix chat's missing `apply`? Allow the 7.9 GB download? Set the Apple tile row for `tile_cols`? Make the
  per-record check a gate in its own right (open since `hf-metal-m2-4b-2026-09-30.txt`)? Remove the f16 head copy,
  which moves published same-head ratios? Does an MMLU through the Metal kernel count as "through the served kernel"?

## 8. Checked now at $0

1. The refusal is `fused_metal.rs:970-977`, the limit `:80`, and `source_of` (`:93-102`) has no entry for the tiled
   kernel. Wiring touches both the map and the op.
2. `tv_q4_metal_tiled` is `tv_q4_h.metal:176-223`, its barriers `:202` and `:206`, its argument `:152-170`. `:162` cites
   `tests/q4_tiled_matches_served.rs`, which `find` locates nowhere outside comments.
3. The int4 records come from the config notes: 4B v + o + down@12-23 = 84, 8B v + down@10-26 = 53, 14B v + o +
   down@10-28 = 99. Only `down_proj` crosses the wall: 12, 17, 19 records, 0.159, 0.455 and 0.900 GB a token (*computed*).
4. `llvq-tetra` on mps: tiled equals untiled at tiles 256, 2,048 and 8,192 on the v and o shapes, 2.90e-07 against dense
   at 9,728, 256 of 256 ids, 2.750 GB (`hf-metal-m2-4b-2026-09-30.txt`). Its rows kernels are hand-written around one
   checked helper (`mps.py:90-164`, `:172-184`, `:222-279`; deviation É1). No `llvq-tetra` speed is journaled.
5. This Mac: Apple M3 Max, 40 GPU cores, Metal 4, 64 GiB, 16 CPUs, macOS 26.6.2, Command Line Tools only, no `xctrace`
   and no offline `metal` compiler. `pmset -g therm` and `ioreg -c IOAccelerator` ("Device Utilization %") read without
   privilege. During this review the other processes summed about 320 % CPU and the GPU read 30 % busy, which is what a
   working session looks like and why the timing steps need a quiet machine.
6. Ceilings: 340 GB/s and 12.5 TFLOPS (`5ec72b0`). Hand-written FP16 matvecs at 334-353 GB/s (`k1-metal-2026-08-05.txt`).
7. The q5 Metal step: 33.6 → 43.1 → 48.5 tok/s, then 48.1-49.8; at 48.5, matvec 12.66 ms, rotation 2.72 ms over 144
   launches of one threadgroup each, rest about 5.2 ms. 290 conversions removed for no measurable change. Tile 1-4 %.
   `_ilp` +5 %, refused at token 57. The LUT is ×1.480 [1.436, 1.490] over the faithful port. Thermal drift 49.7 → 23.1
   (`976549b`, `82c029c`, `5a75338`, `31a5acb`, `5ec72b0`, `5f439df`, commit messages only).
8. Dense Rust on Metal: 7.5 tok/s in 8.04 GB (`976549b`), 7.9 (`a10c92c`), 9.6 on the paper-1 file (`kvq8-4b`), 347 and
   368 ms a token shared with the user's browser (`gsm8k-pilot-2026-09-26.txt:53-54`).
9. llama.cpp on this Mac: IQ2_XXS 114.02 ± 0.62 tok/s on 1,246,620,832 B, f16 43.08 on 8,051,284,928 B
   (`m3-iq2-metal-2026-08-30.txt`), so 142.1 and 346.8 GB/s (*computed*).
10. The 4B sealed file is local at its README size. The 8B and 14B sealed files and the 14B q5 file are not. The q5 4B
    file lives only in the bucket (`configs/README.md`).
11. Metal rotation: device scratch, KMAX 32, and `fused.rs:1247` refuses only k > 32. At 25,600 the odd factor is 25,
    so the 32B rotation would pass on Metal (*computed*). `llvq_rot.metal:97-98` repeats the wrong 512·50 factorization.
12. `chat.rs:290-296` feeds a turn in one call. `row_cap(1) = 256` (`model.rs:746-753`), so a long turn errors on the
    fused Metal path (*computed*). `gsm8k` chunks its prefill at `MAX_ROWS` (`gsm8k.rs:284-296`).
13. `chat` never calls `Served::apply`. `fusedrun` (`:862, :978, :1153`), `mmlu` (`:579`) and `gsm8k` (`:187`) do.
14. `fusedrun` is CUDA-only (`:649-653`), and chat disclaims its own tok/s (`chat.rs:36-38`).
15. candle 0.9.2 on Metal: the gemm layout rules (`mlx_gemm.rs:297-322, 360-375`), the SDPA dispatch rules
    (`ops.rs:1040-1106`, `sdpa.rs:244-270`), `broadcast_matmul`'s copy (`tensor.rs:1545-1556`), the buffer pool and its
    power-of-two rounding (`metal_backend/device.rs:203-270`), and the variables `CANDLE_METAL_ENABLE_FAST_MATH`,
    `CANDLE_METAL_COMPUTE_PER_BUFFER` and `CANDLE_METAL_COMMAND_POOL_SIZE`.
16. `LLVQ_THREADS` is read by `smoke.rs:669` and `errmap.rs:314` only. `LLVQ_METAL_KERNEL` picks the Metal matvec
    (`fused_metal.rs:231-246`) and falls back with a warning on an unknown value.
17. The two shader copies the wheel ships are byte-identical to the served ones (`cmp`).
18. The GSM8K Metal pilot dumps are in the repository (`docs/data/gsm8k-dumps/`, 50 problems, dense).
19. On the L40S the fused files give the dense arm's 256 tokens at 4B and 8B, and the 14B diverges at token 78
    (`paper-table-2026-09-25.txt`). The 4B's greedy line after the reference prompt is the loop "Paris. The capital of
    France is Paris." (`paper-served-4b-2026-09-25-r2/fused-q4-256.txt:16-17`), whose `fused F0` line hashes to
    `8784ffc4…` (`ops/jobs/served-grouped.sh:29-31`). A token gate on that prompt is weak: the tail mutant of M1
    survived 8 ids on four prompts at an 8.79e-2 per-row error (`hf-metal-m2-4b-2026-09-30.txt`).
20. `fusedrun`'s 203-token prefill prompt is the 5-token prompt repeated to length (`fusedrun.rs:981-991`), so it is not
    a natural text. WikiText-2's test parquet is in the local Hub cache, read by `corpus::wikitext2_test()`.
21. Rotation launches a token on the card: 96 (4B), 127 (8B), 101 (14B, 280 projections) (`fused-q4-256.txt:3` of each
    paper-table run).
22. The oracle defaults to `Qwen/Qwen3-0.6B` on the CPU (`oracle.rs:26-29`); `Checkpoint::fetch` takes a local
    directory and never falls back to the Hub (`loader.rs:217-235`).

## 9. Not established

- Any speed of a sealed file on Metal, fused or dense, under a protocol. Every Mac rate above is a commit message or an
  estimate.
- The int4 kernels' rate in situ, and the tile's residency effect.
- Token agreement of the e4 objects on Metal, and whether the Rust q4 head kernel, which no journal shows run in situ,
  is right end to end.
- What the f16 head copy costs on Metal, and so where the same-head ratio lands.
- What the device holds for the dense and f16-table arms. candle rounds op outputs to the next power of two and keeps
  freed buffers until a flush, so the host counts 8.04 and 1.94 GB may understate the allocation.
- Anything at 8B or 14B on Metal. No timing exists, and the files are not here.
- What the 2.4-3.6 ms is.
- Whether candle's Metal gemm and SDPA behave as their source reads on the `Grow` views, and what either form rounds to.
- Whether `simdgroup_matrix` is exact on integer inputs, and the M3's f16 rate.
- Normative Metal text on forward progress: the MSL specification was not located, and the Apple enum page defines
  nothing.
- The value of a Metal 32B: no file exists, and only the code says the walls are absent on Metal.

## 10. Revision against the review of 2026-11-02

Every high and medium finding is applied. The two places where the text departs from the fix as written:
- The review proposed "the 203-token prompt of prefill-203" as the non-degenerate prompt of the logits gate. That
  prompt is the reference prompt repeated to length (`fusedrun.rs:981-991`). The gate uses the first 256 tokens of the
  WikiText-2 test split instead, which is cached and natural text.
- The review asked for the same-head estimate ×4.7 [×3.0, ×7.0] beside every raw figure. Checking the f16 head path for
  it showed the per-token table copy of §3.2, which that estimate had missed. The same-head estimate is now ×2.8
  [×1.4, ×4.3], and Fh's predicted rate moved from 40 to 24 tok/s.

| finding | where it is applied |
|---|---|
| quiet machine, load samples, binary by path, niced label, days and thermal states | `prereg-mac-1` §3, §8 R2b and R7; §5 here |
| rule 4: same-head beside raw, the dense arm as a slow baseline in GB/s | §1, §3.2; `prereg-mac-1` §7 |
| rows kernels are hand-written in `mps.py` | §2, §3.3 (b2), §8 item 4 |
| arm F is the expanded config, not the served one | §1; `prereg-mac-1` §3 and R7 |
| 127 rotations at 8B | `prereg-mac-2` §5; `mac_bounds.py` |
| environment gate: every `LLVQ_*` and `CANDLE_*` refused, `LLVQ_THREADS` dropped, build limits | `metalrun.rs`; `prereg-mac-1` §3 and gate 3; §5 |
| an independent reference for the e4 path | record gate and G-L in `prereg-mac-1` §4-5; `metalrun check` |
| tile default 2,048 with residency | §3.1; `fused_metal_wall.rs`; `prereg-mac-1` §7 and R3 |
| memory: host count labelled computed, device allocation measured, b/param | §1, §3.2, §7; `prereg-mac-1` §6-7 |
| fast math either way, binding test drafted | §3.1; `q4_tiled_matches_served.rs`; `q4_metal_candle.rs` |
| barrier evidence and the first-barrier mutant | §3.1, §6 item 1; `q4_tiled_matches_served.rs` |
| arm order: METHODE's same order, 2 discarded + 7 timed | `prereg-mac-1` §4; `prereg-mac-2` §3; `metalrun.rs` |
| oracle without an undeclared download | `prereg-mac-1` step 1; §7 rule 10 |
| every U row ends with "otherwise" | §4 |
| L7 priced at 1 h for 4 windows, 2.3-3.3 h for 12 | §5 |
| L8 at the same chunk of 256 in both arms | §5 |
| prereg-mac-2 predictions re-derived before the stamp | `prereg-mac-2` §5 |
| the 14B dense arm's true cost | `prereg-mac-2` §3 step 6 and §8 |
| one GB/s definition | §3.2; `metalrun.rs`; both preregs |
| U3 is 2.4-3.6 ms; phased pass first, null kernels for host encoding | §1, §3.2, §3.3 (b3), §4, §5 |
| the Mac as a second local target, ETAT §5 not edited | §1, §6 item 3, §7; `prereg-mac-1` §1 |
| cheaper experiments | L2b, L6, L7a, L9 in §5; `prereg-mac-1` steps 2, 4 and 5 |
