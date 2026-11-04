DRAFT, not stamped, not committed

# Preregistration. The sealed 4B on the fused Metal path: oracles, logits, tokens and decode speed on the M3 Max

**Draft written on 2026-11-02 by the wave-5 deep dive (lead "mac"), revised the same day against an adversarial
review. To be reviewed by the operator, committed and TIMESTAMPED BEFORE the run.** Operator go to run: **not given**.
Cost: **$0**, the operator's Mac, no paid job, no download. About 15-20 min of machine time after a 30-minute wait
(*estimated*, §9). Spend before it: $248.84 over 237 rows of the registry (*measured*, `docs/ETAT.md:189`), unchanged by
this run.

## 1. Why this run exists

No sealed file has ever run fused on Metal. `MetalRuntime::upload_int4` refuses any int4 record whose activation does
not fit in 32,768 B of threadgroup memory (`llvq-llm/src/fused_metal.rs:970-977`). The 12 `down_proj` records of the
4B (d_in 9,728) are past it, so the load fails before `finish()`. The tiled kernel that lifts the wall exists
(`llvq-llm/kernels/tv_q4_h.metal:176-223`). It runs on mps in `llvq-tetra` (`docs/mesures/hf-metal-m2-4b-2026-09-30.txt`),
and the Rust host does not route to it.

No speed of a sealed file on the Mac is journaled. The documented Mac route is the dense reconstruction, 7.5 to 9.6
tok/s in 8.04 GB (commit messages `976549b`, `a10c92c`, and `kvq8-4b-2026-08-15.txt` on the paper-1 file). The only
fused Metal rates are the q5 object's, 48.1-49.8 tok/s, in commit messages without a journal (`31a5acb`, `5ec72b0`).

The Mac is a second local target beside the consumer GPU of `docs/ETAT.md` §5. Whether a Mac reading answers that item
is the operator's ruling, and this run does not presume it.

## 2. What the code must hold before the run

The branch implementing the audit's step 4 delivers, and the commit that carries this file names:

1. Routing: an int4 record with `d_in * 4 > 32,768` goes to `tv_q4_metal_tiled`, every other record keeps
   `tv_q4_metal`. The column tile is a load-time parameter (`MetalTuning`), default 2,048, the value `llvq-tetra` was
   validated with (`llvq_tetra/integration.py:25-28`). It is refused by name when the load starts if it is 0, not a
   multiple of 256, or above 8,192.
2. The load prints one routing line: records, how many tiled at which tile, how many untiled.
3. `llvq-metal/tests/q4_tiled_matches_served.rs` and `llvq-llm/tests/q4_metal_candle.rs`, green, fast math on and off,
   with their planted mutants failing, the two barrier mutants over 16 repetitions with their kill rate (wave-5 `mac.md`
   §6 item 1).
4. `MetalTuning { keep_host: true }` keeps the host copies that `TetraMatvec::cpu_fwd` needs (the code path of
   `MetalRuntime::new(.., keep_host)`, `fused_metal.rs:257-296`, which `from_tables` does not take today).
5. The bench binary `metalrun` with the protocol below (`check` and `time`), built with `nice -n 15 cargo build --release
   -p llvq-llm --features metal --bin metalrun -j 4` and the `oracle` binary built the same way.

If any of this differs on the day, the deviation file says how before the first measurement.

## 3. Machine, environment, object

**A quiet machine.** No build anywhere, no other GPU application, the browser closed, no other agent session, on AC
power. The last build ends at least 30 minutes before step 1, because the same binary measured 49.7 tok/s cold and 23.1
after three hours of compiling (`5f439df`). The repository's own records say contention dominates absolute rates on this
Mac: per-prompt walls swung from 62 to 255 s under contention ("A throughput figure needs a quiet machine and its own
prereg", `hf-metal-m2-4b-2026-09-30.txt:118-122`), and the dense arm read 347-368 ms a token shared with a browser
(`gsm8k-pilot-2026-09-26.txt:53-54`).

**Load samples**, taken outside every timed window and printed:
- the GPU's "Device Utilization %" from `ioreg -r -d 1 -c IOAccelerator`, before the first load and 5 s after the last
  round;
- the other processes' summed `%CPU` from `ps -A -o pid=,%cpu=`, this process excluded, before the first load and after
  every arm of every round;
- `pmset -g therm` before step 1 and after step 5.

Thresholds: GPU at most 10 % before the first load, others' CPU at most 100 % (one core of 16) at every sample. Before
stamping, the operator takes one dry reading of both on the quiet machine and may raise the GPU bar to that reading plus
5 points. A sample past a threshold triggers R2b.

**Binaries by path.** `target/release/oracle` and `target/release/metalrun`, each sha256 printed in the journal header
(`metalrun` prints its own), never `cargo run`. Both run under `nice -n 10` (METHODE §6). The absolute rates are
therefore labelled "niced, on a quiet machine".

**Environment.** `metalrun` reads every variable at start and refuses any `LLVQ_*`, `CANDLE_*`, `MTL_*` or `METAL_*`
name, with an explicit and empty allowlist, and prints that check. `CANDLE_METAL_ENABLE_FAST_MATH` sets the math mode of
every candle kernel (candle-metal-kernels 0.9.2 `kernel.rs:185`), and `CANDLE_METAL_COMPUTE_PER_BUFFER` moved the q5 step
from 33.9 to 27.7 tok/s (`41e6628`). `LLVQ_THREADS` is not used: only `smoke` and `errmap` read it. `RAYON_NUM_THREADS`
is printed if set.

**Object.** `~/q4b-sealed-2026-09-23/qwen3-4b-sealed.bin`, 1,418,224,685 B, sha256
`886391a8c03f66dc269cc65c3598c6627dbdcd259180aff36604ef10d37371b8`. Config: `configs/qwen3-4b-tetra-e4.json`, sha256
`098fe49c…`, the expanded form. That is the Mac's default under the open decision of `docs/ROADMAP.md` §5 ("refused, and
the Mac serves the expanded form"). The 4B's served configuration is `qwen3-4b-tetra-e4-grouped.json` since 2026-11-02
(`docs/ETAT.md:20`), which Metal refuses by name (`served.rs:275-289`). No arm of this run is the served configuration.

## 4. What runs

1. **Oracle** on Metal, rule 10, on the cached checkpoint, no download:
   `nice -n 10 target/release/oracle ~/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c 64 metal`.
   `Checkpoint::fetch` reads a local directory and never falls back to the Hub (`loader.rs:217-235`). Two f32 copies of
   the 4B, about 32 GB of unified memory in its own process (*estimated*). The alternative the repository used so far,
   `oracle Qwen/Qwen3-0.6B 64 metal`, downloads about 1.5 GB (*estimated*) and needs the operator's go for it. If the
   4B oracle cannot allocate, the deviation file switches to the 0.6B, after that go.
2. **Check process**, `nice -n 10 target/release/metalrun check <file> <config>`:
   - F loaded with host copies, tile 2,048, through `Served::apply` (expanded, `cat`).
   - The window: the first 256 tokens of the WikiText-2 test split as `corpus::wikitext2_test()` returns it and the
     file's tokenizer encodes it, no special tokens. The parquet is in the local Hub cache.
   - **Record gate.** F prefills the window, then decodes one teacher-forced step on token 257 with a `Capture` that
     keeps each projection's input (`model.rs:85-88`). Each of the 168 Tetra records' device output is compared with
     its CPU arm on the host copies (`cpu_fwd`), both at the adapter's output. Each of the 84 int4 records' device
     output is compared with the host lane-order reference built from the record (`fused::load_with`), narrowed to f16
     by round-to-nearest-even. The q4 head's logits are compared with the host lane-order reference of
     `llvq-metal/tests/q4e_matches_host.rs`. Bit equality everywhere.
   - **Logits gate G-L.** D is loaded (`sealed::load`, f16). F and D score the window: logits at all 256 positions.
3. **Timed process**, `nice -n 10 target/release/metalrun time <file> <config> 128 7`, four arms loaded in this order:

   | arm | what | role |
   |---|---|---|
   | F | the expanded config, q4 tile 2,048 | the object as the Mac can run it |
   | F' | the same, q4 tile 8,192 | exactness control, and the tile choice |
   | Fh | fused projections, `EmbedMode::F16`: the dense arm's f16 head | same-head companion, rule 4 |
   | D | `sealed::load`, the dense reconstruction | the route the Mac documents today |

   Every arm goes through `Served::apply`, so all four run the expanded attention on `cat`. Then the window gate: F and
   F' logits on the 256 positions. Then the rounds: prompt `"The capital of France is"` (fusedrun's `PROMPT`), 128 new
   tokens, greedy. Two discarded rounds, then 7 timed, the arms in the same order F, F', Fh, D at every round (METHODE
   §2, `docs/format-noyau.md` §9). tok/s = 128 / t over a whole `generate`, prefill included, the served door's reading.
   Ratios are formed round by round.
4. **Untimed, after the rounds.** F decodes 256 tokens once, printed as `fusedrun` prints its F0 line
   (`"  fused F0: {}"`, `fusedrun.rs:1781-1786`). `grep -F '  fused F0:  ' | shasum -a 256` is compared with `8784ffc4…`,
   the card's reference line (`ops/jobs/served-grouped.sh:29-31`), and `cmp` against that line gives the first
   differing byte if they differ.
5. **Off protocol, after step 4.** `Qwen3::generate_phased` (`model.rs:2316`) on F, Fh and D, 32 tokens each: embed,
   blocks, head and argmax, each closed by a device fence. The fences serialize the work, so these totals are never set
   beside the timed rates.
6. **Optional arm L**, after step 5, as a separate process: `llama-bench -p 0 -n 128 -r 5` on the IQ2_XXS GGUF of
   2026-08-30. Its GB/s is tok/s × 1.2466 GB and is never divided against an arm of ours (rule 5). It runs only if
   that GGUF and build are still on the Mac, sha256 recorded before stamping. Otherwise the 142.1 GB/s of 2026-08-30 is
   cited with its date.

## 5. Gates, each one voids the run

1. `oracle` prints MATCH.
2. The file and the config at their bytes and sha256; both binaries' sha256 in the header.
3. Both `metalrun` processes print the environment check: no `LLVQ_*`, `CANDLE_*`, `MTL_*` or `METAL_*` variable.
4. F, F' and Fh each print `int4: 84 records, 12 through tv_q4_metal_tiled at <tile> columns a tile, 72 through
   tv_q4_metal`, with their own tile. This proves the routing reached the op, not only the parser.
5. **Record gate:** 252 of 252 projections and the head equal their host oracles, bit for bit.
6. **G-L:** F and D put the same token on top at no fewer than 248 of the 256 positions.
7. F and F' give the same window logits, every f32 at every position, and the same 128 tokens in every round. The two
   tiles walk the same words in the same order, so a difference means the tiled path is not exact as implemented.
8. Each arm returns the same 128 tokens in all its rounds.
9. Every arm prints `attention expanded on cat`, read from the model's getters after `apply`.
10. F's host byte count reads 1.38 GB. This is the card's own accounting (`fused_metal.rs:1645-1650`), true by
    construction, so it checks the load and predicts nothing.

## 6. Outcomes reported, never gated

- Per arm: load time; the host byte count (*computed*); the device's own allocation, `MTLDevice.currentAllocatedSize`,
  before and after its load and again after the discarded rounds (*measured*); both in b/param over 4,022,468,096
  parameters (`docs/fiche-4b.md:66`), against 16.0 for the dense model's f16 weights.
- Per arm: tok/s median and range, ms a token, and GB/s of weights (*computed*: weights a token × tok/s, with 1.377 GB
  for F and F', 1.936 for Fh and 8.045 for D). Device bytes are never a GB/s denominator.
- Ratios, round by round: raw F/D and same-head Fh/D, published together; F'/F for the tile.
- First divergence of F against D and against Fh over the 128 tokens, or "identical through 128".
- G-L: the agreement count, the largest |Δlogit| over the largest |logit| of D, and each arm's NLL over the 255
  predictions with their ratio.
- F's 256-token line: its sha256, whether it equals `8784ffc4…`, the first differing byte if not.
- Drift: F's first and last timed rounds.
- Load samples, `recommendedMaxWorkingSetSize`, `pmset -g therm` before and after.
- The phased medians of F, Fh and D, labelled fenced.
- Arm L's GB/s beside F's, no ratio.

## 7. Signed predictions

| quantity | point | interval |
|---|---|---|
| F (tile 2,048), tok/s | **52** | [40, 64] |
| F'/F, tile 8,192 over 2,048 | **0.98** | [0.94, 1.01] |
| Fh, tok/s | **24** | [14, 32] |
| D, tok/s | **8.5** | [6.5, 10.5] |
| F/D, raw | **×6.1** | [×4.2, ×8.5] |
| Fh/D, same head | **×2.8** | [×1.4, ×4.3] |
| F, GB/s of weights | **72** | [55, 88] |
| F, device allocation after its load | **1.39 GB**, 2.76 b/param | [1.37, 1.45] GB |
| G-L, positions with the same top token | **254 of 256** | [249, 256] |
| G-L, NLL of F over D, minus 1 | **0.00 %** | [-0.30, +0.30] % |
| phased head, Fh minus F | **22 ms** | [14, 46] |
| F against D, greedy tokens | **identical through 128** | a divergence, if any, after token 32 |
| F's 256-token line | **hashes to `8784ffc4…`** | a divergence, if any, after token 128 |
| record gate | **253 of 253** | |
| arm L | **142 GB/s** | [130, 150] |

Reasoning, from wave-5 `mac.md` §3.2 and `code/mac/mac_bounds.py`.

F. The q5 step was 20.6 ms at 48.5 tok/s: matvec 12.66 ms, rotation 2.72 ms over 144 launches, a rest of about 5.2 ms
(`5a75338`). Taking the q8 head out of that rest leaves 2.4-3.6 ms, and Tetra ran at 74-77 GB/s (*computed*). The e4 4B
moves 0.749 GB of Tetra, 0.409 of int4 and 0.219 of q4 head, with 96 rotations instead of 144. With int4 between 75 and
250 GB/s the step is 16.4-22.3 ms, so 45-61 tok/s (*estimated*), widened for the never-timed int4 kernels and the
never-run q4 head. If the L2b bench runs before this file is stamped, its range divided by 1.09-1.18 (the in-situ
over-prediction of `31a5acb`) replaces the 75-250 bracket and the points are recomputed. Nothing moves after the stamp.

Fh and the same head. Fh projects through `Head::Dense`, whose `broadcast_matmul` copies the transposed 778 MB table
every token (candle-core 0.9.2 `tensor.rs:1545-1556`): 2.33 GB of traffic (*computed*). The L40S paid 9.4 ms for it, 29 %
of its bandwidth (*computed* from `paper-table-2026-09-25.txt`, 55.0 against 113.8 tok/s). At 15-40 % of 340 GB/s that
is 17-46 ms on this Mac (*estimated*), so Fh reads 15-31 tok/s and Fh/D ×1.6-4.1, widened here. D pays the same copy
inside its measured 7.5-9.6.

The tile. The 12 tiled records carry 0.159 GB, 0.6-2.1 ms of a 16-22 ms step. A tile of 8,192 asks the whole 32 KB, so
likely one threadgroup fits a core, against several at 2,048 (*estimated*); 18.7 KB already capped residency on this Mac
at +8 to +23 % (`82c029c`). Even 30 % slower tiled records move the step by 1-3 %, hence 0.98 and an interval that
reaches just above 1.

Memory. `new_buffer_with_data` uploads at exact size (candle-core 0.9.2 `metal_backend/device.rs:203-217`) and every
F table goes through it, so F's allocation should sit just above its host count. The f16-table and dense arms convert
tensors on the device, whose outputs candle rounds to the next power of two (`device.rs:228-270`), so their allocations
are reported and not predicted.

G-L and tokens. D multiplies f16-rounded weights in a candle gemm; F accumulates the exact decoded values in f32. Both
score the same weights, so they disagree only at near-ties. On the card, two forms of one arithmetic disagreed on top-1
at 0.21 % of decode steps (`gqa-ppl-2026-11-01.txt`). The greedy prediction follows the q5 precedent, 60 tokens
identical (`976549b`), and the L40S, where the 4B gave the dense arm's 256 tokens (`paper-table-2026-09-25.txt`). The
line hash: the int4 records equal the CUDA ones by construction and the Tetra records equal the same host reference,
but attention, norms and narrowing run on other kernels, so a late divergence stays possible.

The phased head: 17-46 ms for the f16 head against about 1 ms for the q4 head (*estimated*, the same derivation).

## 8. Decision rules

- **R1.** A gate fails: the run is void and the failure is journaled. A deviation names the fix, then the run repeats.
- **R2.** Every gate passes and every load sample is within its threshold: the tiled routing becomes the Metal path for
  the sealed files, on the operator's go to merge. The journal carries the first Mac speed of a sealed file, labelled
  "niced, on a quiet machine", raw and same-head ratios together, GB/s beside llama.cpp's with no ratio.
- **R2b.** Every gate passes and a load sample breaks a threshold: the ratios are journaled, and the absolute rates are
  void outside the journal. The timed process may repeat once on a quiet machine under a deviation.
- **R3, the tile.** F'/F range wholly above 1.000: 8,192 becomes the Apple row. Wholly below, or straddling 1.000:
  2,048 stays. Bits are identical either way (gate 7).
- **R4.** F's median under 30 tok/s: the number is journaled and no sentence beyond it leaves the journal. The
  attribution pass (`mac.md` L6) comes before any 8B or 14B timing.
- **R5.** Gates 5 and 6 pass and F leaves D's greedy line before token 32: the position is reported with the G-L
  figures. Two arithmetics of the same weights diverge at near-ties, so this is not a defect by itself.
- **R6.** F's 256-token line differs from `8784ffc4…`: the first differing byte is reported. Cross-device equality is
  not guaranteed by construction, and nothing follows from it alone.
- **R7, framing.** No ratio is formed between this run and the q5 readings of 2026-09-21 or any card figure. A sentence
  that sets a Mac figure beside the card's 133.3 tok/s names both attention forms, expanded here and grouped there, and
  both machines. A sentence that sets this F beside the q5 readings says they cross days, builds and thermal states.
- **Otherwise**, not settled: the operator decides. No outcome of this run kills a lead; a kill is the operator's.

## 9. Cost

*Estimated*. $0: no card, no job, no download. Machine time: oracle 2-4 min. Check process 4-6 min: F with host copies
and D (35-80 s, `976549b` and `chat.rs:167-168`), a 256-row fused prefill at 15-17 ms a row, 253 comparisons. Timed
process 6-7 min: loads of D 35-80 s and of F, F', Fh 3-8 s each; the window gate about 10 s; nine rounds of F 2.5 s +
F' 2.6 s + Fh 5.3 s + D 15 s, about 3.8 min; steps 4 and 5 about 30 s. Arm L about 1 min. In all about 15-20 min of
machine time after the 30-minute wait. Unified memory: about 32 GB in the oracle's process, 9.4 GB of host count plus
about 0.8 GB of host copies in the check process, 12.7 GB of host count in the timed process (*computed* from the card's
counts 1.38 + 1.38 + 1.94 + 8.04), more on the device for Fh and D if candle's rounding applies, all under the working
set the run prints first. One host thread, niced.

## 10. Deviations

Any departure goes in `proofs/preregistration-mac-1-<date>-ECARTS.md`, beside this file once committed, never inside
it.

## 11. What this run does not establish

- Anything at 8B or 14B. Their files are not on the Mac (`prereg-mac-2.md`, after the operator's go to download).
- Quality beyond one window of 256 tokens. No score runs here. A GSM8K pairing with the Metal pilot dump is a separate,
  optional run.
- Long context, prefill speed, the grouped attention (refused on Metal), a rows kernel.
- The int4 kernels' rate in isolation, and where the non-weight time goes beyond the fenced split.
- Whether Mac readings answer the local-hardware item of ETAT §5: an operator ruling.
- Any other Mac. Only the M3 Max is measured, and the M5's Neural Accelerators are not exercised.
