# Throughput review, 2026-11-05

Read-only review at $0. No job launched, no config changed, no branch touched but this one. Inputs: `ETAT.md`,
`ROADMAP.md`, `format-noyau.md`, the archive `debit-vague5-2026-11-02/`, the uncommitted audit
`docs/archive/audit-debit-2026-10-29.md` (main checkout only), every journal from 2026-10-30 to 2026-11-05,
`docs/data/jobs.csv`, the sibling hand-over notes, and the served decode path in code. Served object throughout: the
sealed 4B (`886391a8`) under `configs/qwen3-4b-tetra-e4-kernel-from1280.json`, on one L40S.

## 1. Summary

The served 4B decodes at 134.7 tok/s after a 5-token prompt and 104.0 at 8,192 (*measured*,
[attn-t8-r2](../mesures/attn-t8-r2-2026-11-04.txt)). That is 7.42 and 9.62 ms a token. 300 tok/s is 3.33 ms.

- **No eager route reaches 300.** The host takes 4.11 ms to enqueue one token (*measured*, `H` in
  [lot1](../mesures/lot1-4b-2026-11-04.txt)). Today it hides behind 7.24 ms of device work. Every eager gain listed
  below together reaches about 210 tok/s (*estimated*), and past about 244 tok/s the host binds unless its 929 launches
  shrink or get cheaper.
- **300 is a persistent-kernel target, so it rides on 4E v2.** On the rule-H form (`rows + fixed`), 300 needs the
  1.377 GB weight stream at 490 GB/s with 0.45 ms of fixed cost, or 520 GB/s with 0.70 ms (*computed*). 4E streams
  at 290 GB/s today ([pers-4e-r3](../mesures/pers-4e-r3-2026-11-05.txt)). The tables-in-shared model points at a DRAM
  floor near 676 GB/s, untested on a card ([pers-sass](../mesures/pers-sass-2026-11-05.txt)).
- **Three items nobody owns are worth more than any eager fusion.** The prefill takes 35 s to the first token at 8,192
  and a GEMM route takes it to about 3 s. The attention kernel reads the cache at η 0.545 where FlashInfer reads 0.810
  on the same card, worth +10 tok/s at 8k and shared by 4E. The kernel form has never run at 8B or 14B.
- **Several measured or built gains sit unserved**: the gestures (×1.012, measured), `attend_writing` (written, unused),
  `tv_tetra48_seg` (bit-exact on the bench, −20 % of the matmul pass).
- **The living documents lag the object by three days.** `ETAT.md`, `ROADMAP.md`, `HISTORIQUE.md` and `main` do not
  carry the served 4B config, the 8B and 14B grouped runs, or the spend since 2026-11-02 (§6).

## 2. Where a decode token goes

### 2.1 Launches, read in code

A token below position 1,280 runs 929 kernels; from 1,280 on it runs 821, plus 2 memcpys and 1 host sync. The three
exact counts match the served log line "96 rot_launches/token, 252 matvec_launches/token (0 groups + 168 lone + 84
int4)" (`fused_cuda.rs:3716-3741`). The candle counts are reconstructed from `model.rs:1777-1927` and candle 0.9.2.

| family | launches a token | how it runs |
|---|---|---|
| `tv_tetra48_h`: q, k, gate, up ×36, down ×24 | 168 | one warp a row, 256 threads, f16 out |
| `tv_q4_hh`: v ×36, o ×36, down ×12 (layers 12-23) | 84 | int4 g128, f16 in, staged f32 |
| `rot_apply` | 96 | **grid of one CTA**, 1,024 threads (`fused_cuda.rs:2639`) |
| RMSNorm: ln1, ln2, q_norm, k_norm ×36, final | 145 | candle `rms_norm`, one row |
| RoPE, q and k | 72 | candle `rope` |
| KV append into `Grow(256)` | 72 | candle `copy2d` |
| attention below 1,280: two cuBLAS gemms, affine, softmax | 144 | `grouped_attention`, `model.rs:662-690` |
| attention from 1,280: split-KV kernel | 36 | grid (ceil(pos/256), 8) of 128 threads |
| SiLU, mul, two residual adds | 144 | candle elementwise |
| head: gather, `tv_emb_q4_h`, f32 cast, argmax | 4 | tied int4 g64 table, 18,992 CTAs |

Host work a token: about 505 candle launches each call `cuModuleGetFunction` through `get_or_load_func`, about 860
`cuMemAllocAsync` with their frees, one blocking `to_scalar`, one pageable upload of the next id. Event tracking is
off on the served door before the first upload (`fused_cuda.rs:3684-3709`).

### 2.2 Short prompt, 7.42 ms

| line | ms | label and source |
|---|---|---|
| 168 Tetra matvecs, 0.769 GB | 2.3-2.8 | *estimated*: 252 all-Tetra launches take 3.788 ms for 0.98 GB, *measured* ([pers-l1](../mesures/pers-l1-2026-11-04.txt)); `nullk` is 2.294 of 3.734 ([banc-252](../mesures/banc-252-2026-10-06.txt)); scaled by launches and bytes |
| 84 int4 matvecs, 0.409 GB | 0.8-1.2 | *estimated*; `tv_q4_hh` has never been timed alone |
| head, 0.219 GB, one launch | 0.32-0.40 | *estimated*; 4E's head phase reads the same table in 377 µs, *measured* in another geometry (pers-4e-r3) |
| 96 single-CTA rotations | 0.9-1.25 | 1.00 *computed* from isolated medians, 7.82 µs at 2,560 and 18.40 at 9,728, *measured* (lot1); never timed in situ |
| about 577 small candle and cuBLAS launches | 1.3-1.8 | *estimated* at 2.2-3 µs each; one tiny candle launch adds 2.03 µs, *measured* (pers-l1 boundary A/B) |
| argmax, cast, gather, final norm | 0.07 | candle argmax 53.16 µs, *measured* (lot1 P3); rest *estimated* |
| token boundary | 0.06-0.10 | 0.062 ms recovered by the on-device loop, *measured* (lot1) |
| **sum** | **5.8-7.6** | against 7.42 *measured* |
| host enqueue, hidden | 4.11 [3.90-6.68] | *measured*: H 4.106, D 7.238, slack 3.146 (lot1) |

The weights are 1.398 GB at the formats' rates, a 1.62 ms read at 864 GB/s (*computed*). About 78 % of the token is
something else.

### 2.3 At 8,192 positions, 9.62 ms

The token adds 2.20 ms over the short one (*computed* from the two medians). One attention-kernel pass over the 1.208
GB f16 cache takes 2.57 ms at η 0.545 (*computed* from [lot1](../mesures/lot1-4b-2026-11-04.txt) r2). It replaces
144 grouped-attention launches worth about 0.35-0.45 ms (*estimated*). The two agree. The `Grow(256)` reallocation
copies the prefix every 256 positions, about 4.7 MB a token amortized at 8k (*computed*), which is negligible.

### 2.4 What is unknown, and the reading that fills it

The split between matvecs, rotations and small kernels **in situ** is unknown. Every row of §2.2 except the total is
an estimate or a measurement taken in another geometry. `ncu` is refused on HF Jobs (`ERR_NVGPUCTRPERM`,
[f3-events](../mesures/f3-events-2026-08-19.txt)), and `nsys` is not in the image (decision 13 of the wave-2 report,
default no).

The reading: `nsys profile --trace=cuda` on the served door, 64 tokens at 5 and 8,192 positions. Kernel activity
tracing goes through CUPTI and should not need the counter permission that blocked `ncu` (*estimated*, unverified on
HF). Cost: one `l40sx1` job of about 5 minutes, about $0.15, plus an image rebuild at $0 that adds `nsys`. Fallback
if CUPTI is refused too: `LLVQ_TIME_EVENTS` ported to `fusedrun`, code at $0, same job cost. It turns every row of §2.2
into a measurement and decides the order of §3.

## 3. Remaining optimizations

Gains are against 7.42 and 9.62 ms, *estimated* unless marked. "4E" is the 4E v2 session on
`claude/serene-bohr-1edf69`. "Bits" says whether the served tokens can stay identical.

### 3.1 Owned by 4E v2, listed for completeness

| item | short | 8k | note |
|---|---|---|---|
| 4E attention phase: grouped below 1,280, kernel above | from 132 to about 180-190 | from 54 to about 104+ | 2,355 µs a token at 5 tokens, 13,273 at 8k, *measured* (pers-4e-r3) |
| prefetch in the row loop, 768 threads, cost-weighted q/k/v slices | stream from 290 toward 410 GB/s | same | pers-4e-loop |
| decoder tables in shared, kept across layers | stream toward 490-676 GB/s | same | the lever that decides 300 (pers-sass) |
| norm/RoPE/append spread over the grid | 220 µs wall for 5.7 µs of mean work | same | pers-4e-r3 |

### 3.2 Not owned by anyone

| # | item | short tok/s | 8k tok/s | effort | bits, quality, memory | card reading | if 4E v2 succeeds |
|---|---|---|---|---|---|---|---|
| 1 | prefill GEMM route: dequantize a tile to f16, then cuBLAS | 0 | 0; first token at 8k from 35.0 s to 2.85-3.4 s (prefill.md) | 4-7 days | T8 (rounding of the GEMM); 0 bits | V4 $1.40-1.67 | still matters: 4E prefills through the same eager rows path |
| 2 | attention kernel memory pipeline, η 0.545 to about 0.8; smaller splits at mid length | 0 | +10 (104 to 114); 4k +8.6 (117.6 to 126.2) | about 1 week | same reduction tree keeps the kernel form's bits | attnbench in situ, $0.3-0.5 | still matters: 4E runs the same arithmetic |
| 3 | kernel attention form at 8B (G 4) and 14B (G 5) | 0 | 8B 67.1 to about 88; 14B 45.3 to about 58, applying the 4B's ×1.34 and the G 5 η ratio 0.874 | configs and preregs only | T8 per size, like the 4B's | about $1.5 (8B) and $2.5 (14B) | still matters: 4E has never run above 4B |
| 4 | real KV store, int8 then int4, read in place by the split-KV kernel | 0 | int8 +15 (to 119), +21 with item 2; int4 +29 (to 133) at η 0.8 | 2-4 weeks | quality T8; KV memory at 8k from 1.21 to 0.64 or 0.34 GB | ppl, light MMLU, a long task: $3-5 | operator's next step; composes with 4E |
| 5 | eager row loop: port 4E v2's prefetch form into `tv_tetra48_h` and `tv_q4_hh` | +12 (0.6 ms) | +7 | 2-3 days once 4E v2 has it | keeps each lane's order, so exact | planesbench arm, $0.2 | moot; cheap hedge if 4E stalls |
| 6 | rotation in the matvec staging for eager grids | +15 (0.75 ms net of a 1-2.5 µs prologue a CTA) | +9 | 1-2 weeks | exact if the WHT order is kept; f32 activation at 9,728 is 39 KB of shared, against the tile-64 L1 budget | planesbench + served door, $0.3 | moot (4E already rotates per CTA) |
| 7 | fast single-CTA rotation R1 (register WHT) | +11 (0.56 ms, bigmodel.md) | +6 | 3-5 days | exact by construction | V0c $0.24-0.30 | partly: 4E repeats three rotations a layer in every CTA |
| 8 | serve `tv_tetra48_seg` on the served object: q+k and gate+up | +10 (0.51 ms, 72 launches at the bench's 7.1 µs each) | +6 | 3-5 days: an f16-output twin, partial groups beside int4 v, `SegPlan` | kernel bit-exact on 72/72 groups ([seg-tetra](../mesures/seg-tetra-2026-11-04.txt)); the twin needs its own proof | identity + served door, $0.3-0.5 | moot |
| 9 | residual adds in the o and down epilogues; q/k-norm + RoPE + KV write in one kernel | +9 (0.45 ms, about 250 launches) | +5 | 1-2 weeks | exact with the load-time form check 4E already has (`pers_4e_norm_rope_append`) | served door, $0.3 | moot |
| 10 | ln1+rot, ln2+rot, silu·mul+rot fused | +6 (0.30 ms, 120 launches) | +3 | 1 week | exact if RMSNorm is emulated and checked at load | served door, $0.3 | moot |
| 11 | serve the gestures: f16 argmax by key kernel, token kept on the card | +1.6 (×1.012, *measured*, lot1) | +1.0 | code exists; a serving decision | exact, 9 of 9 card mutants killed | none | moot (4E has its own head and argmax) |
| 12 | `attend_writing` from 1,280: KV append folded into attention | 0 | +1.6 (72 `copy2d`) | 1 day | same values written | rides any job | 4E already appends in its own phase |
| 13 | progressive head read by bit planes, argmax certified then rescored | +3 (0.17 ms) | +2 | 1-2 weeks | same argmax by construction; 0 bits | planesbench arm, $0.2 | **matters more**: 377 µs is 11 % of a 3.33 ms token |
| 14 | host diet: outputs written in place (b1), cached function handles (b2) | 0 today | 0 | b1 2-3 days; b2 a vendored candle, operator's call | 0 bits | lot1's clocks again, $0.3 | moot; mandatory for any eager route past about 244 |
| 15 | `tv_q4_hh` timed alone, then the tiled `tv_q4_ht` at 14B | unknown at 4B | unknown | a bench arm | exact | $0.1-0.2 | 14B wall: a one-CTA int4 down at 17,408 is suspected in a 2.2 ms gap (bigmodel.md) |
| 16 | per-shape `LLVQ_TILE_BLOCKS` | 0-5 % | same | a sweep | zero bits, exact | $0.2 | moot |

Measurement traps for these items:

- The house prompt decodes "The capital of France is Paris." on a loop. Any n-gram speculation measured on it would
  accept almost every draft. A neutral prompt set comes first.
- GSM8K never reaches position 1,280: its largest position is 1,200. No task score exists for the kernel form past
  its threshold.

### 3.3 Leads looked at and left aside

- **n-gram and prompt-lookup speculation.** The operator excluded only a neural drafter. One extra row costs 3.93 ms
  today ([tk-sweep](../mesures/tk-sweep-4b-2026-11-04.txt)), so prompt lookup buys ×1.00-1.05. It pays only after a flat
  multi-row verify, the brick lead, 12-20 weeks.
- **Layer-skip self-speculation** was set aside because published gains use full-precision targets. Nothing measured
  refutes it.
- **CUDA Graphs (A2)** stay closed and out of the core. The 47 % memory figure behind its row came from an expanded
  8k window; the grouped `Grow` store changes that premise. Reopening still needs the condition of its row, and 4E is
  host-free anyway.
- **Bytes.** int4 records and the head carry 46 % of the bytes (wave-2 report). Re-encoding them in Tetra would cut
  about 0.4 GB of int4 traffic to 0.2. It changes the sealed object and loses the MMLU the int4 step bought: a quality
  decision, not a throughput one.

## 4. Ranking

**In parallel with 4E v2, independent of its outcome:**

1. The `nsys` timeline of §2.4, about $0.15. It is the only thing that turns §2.2 into measurements.
2. The prefill GEMM route (item 1). It is the largest latency on the product at 8k, about 31 s, and 4E inherits it.
3. The attention kernel pipeline (item 2). It is +10 tok/s at 8k for the eager path and for 4E alike.
4. The kernel form at 8B and 14B (item 3), with `tv_q4_hh` timed alone (item 15) in the same job.
5. Code of the real KV store (item 4), on the operator's order of kernel first, KV second. Writing it now costs no card
   time.
6. The progressive head (item 13), which grows in value as 4E shrinks the rest.

**If 4E v2 plateaus near 260 tok/s, 3.85 ms:**

- Inside 4E, the head read, the norm/RoPE/append wall and R1's faster per-CTA rotation are worth about 0.3-0.45 ms
  together (*estimated*). That gives about 280-295 tok/s.
- On the eager side, items 5 to 12 together reach about 210 tok/s (*estimated*). They cannot beat a 260 4E.
- The last 5-10 % to 300 then needs either bytes (§3.3, a quality decision) or a flat multi-row verify for prompt
  lookup (weeks).

**Is 300 reachable without the excluded levers?** Only through a persistent path whose weight stream reaches about
490-520 GB/s at 0.45-0.70 ms of fixed cost. The pers-sass model says the stream is bound by the L1 pipe, and tables in
shared memory move its floor to DRAM's (*estimated*). That is a model fitted at one point, not a measurement. L2's arms
(`pdec_smem`, `psmem_t28`) are the card reading that decides it, about $0.33. No eager route gets there.

## 5. What this review does not establish

- Any in-situ split of the token: §2.2 is an estimate bracketing a measured total.
- Any gain in §3.2: each is *estimated* from isolated measurements or launch counts. The 3.76 µs a launch of
  `a1-nullk` and the 2.03 µs of `pers-l1` do not convert one for one into token time.
- Anything on a card other than the L40S, or on Metal.

## 6. Accounting errors and contradictions

1. **`ETAT.md` and `ROADMAP.md` stop at 2026-11-02.** They say the 4B serves the grouped config and the 8B and 14B never
   ran it. Since then: the 4B serves `kernel-from1280` (operator T8, 2026-11-05); the 8B and 14B ran grouped with
   paired perplexity, light MMLU and GSM8K ([gqa-8b](../mesures/gqa-8b-2026-11-04.txt),
   [served-grouped-8b](../mesures/served-grouped-8b-2026-11-04.txt), [gqa-14b](../mesures/gqa-14b-2026-11-04.txt),
   [served-grouped-14b](../mesures/served-grouped-14b-2026-11-04.txt)). The 8B door reads 105.8 tok/s and the 14B 62.5,
   against the pre-trim 95.0 and 57.2 still in the table.
2. **Spend.** `ETAT.md` gives $248.84 over 237 rows and $2.09 for the throughput campaign. The registry reads $270.06
   over 254 priced rows of 259, and $20.12 over 22 jobs since 2026-10-30 (*computed*, `docs/data/jobs.csv`).
3. **Two quantities named tok/s.** `ETAT.md`'s 133.3 is the door, 256 tokens over a whole generate, prefill included.
   The config note's 134.7 is the sweep, 255 over decode time. The kernel config has no door measurement, so the
   ETAT-comparable figure for the served object does not exist yet.
4. **`main` carries none of this.** `origin/main` stops at `ddecad3` of 2026-10-07. The grouped and kernel configs live
   on `integration/lot1`, `debit-trims` and the session branches only.
5. **`format-noyau.md` §5 bis is stale.** It says no kernel reads the int4 kind and `tv_q4_h` never ran on a card.
   Every card job since 2026-10-30 checks `tv_q4_hh` against `tv_q4_h` on the card at load (gate 3 of each journal).
6. **`format-noyau.md` §4 cites `ETAT.md` §7 for A2.** `ETAT.md` has six sections; A2 is in §6.
7. **`HISTORIQUE.md`** has no entry for 2026-11-04 beyond the 32B wall, and none for 2026-11-05.
8. **`CLAUDE.md`** says `LLVQ_FUSE=0` because Tetra48 "carries no segmented kernel". The kernel exists and is bit-exact
   on the bench. What is missing is its f16-output twin and partial-group wiring.
9. **Launch count.** 1,410 (audit, trims journal, `brick.md`) is the expanded `Cat` path. Today's grouped path is 929
   and the kernel path past 1,280 is 821.
10. **The trims' events arm undercounts.** cudarc records events on every slice created while tracking was on, and
    `LLVQ_TRIM_AB` turned tracking off after the load (gestures hand-over note). Its ×1.0115 is a lower bound. The served
    door turns tracking off before the load and is not affected.
11. **Rule H fed 4E a rate it did not have.** The 222-258 tok/s projection used 410 GB/s from `tpersall`, 32 warps a SM
    in its own grid. 4E runs 16 warps a SM and streams 290 (pers-l1, pers-4e-r3).
12. **4E's "about 190 tok/s without its attention phase"** treats short-context attention as free. The eager path spends
    about 0.35-0.45 ms on it (§2.3), so the figure is nearer 180-190.
13. **GSM8K under the grouped form at 4B** reads −0.53 pp [−1.07; +0.00], p = 0.092
    ([gsm8k-grouped](../mesures/gsm8k-grouped-2026-11-04.txt)). It passed rule G1, but its interval touches zero.
