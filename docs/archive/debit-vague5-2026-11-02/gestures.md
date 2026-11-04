# Gestures: cheap bit-exact levers, and the grouped form at 8B and 14B

Deep dive of 2026-11-02, revised the same day after an adversarial review (§10 lists every finding and what changed). Written read-only: no file of the repository changed, no build, no run, $0 spent. Default object: the sealed 4B on the L40S, batch 1, greedy, grouped config, 133.3 tok/s at the served door, 7.50 ms a token (*measured*, `served-grouped-2026-11-02.txt`). Labels: *measured* (named journal or source), *computed* (arithmetic on measured or code inputs), *estimated* (judgment, or at least one estimated input). Code sketches: `code/gestures/`, never compiled, parsed or run. Preregs: `prereg-gestures-0.md` (the 4B's task debt), `prereg-gestures-2.md` (8B and 14B), `prereg-gestures-1.md` (the gestures). Verification arithmetic: `calc/gestures-check.py`.

## 1. Verdict

The lead holds one large lever and three small ones, and a fourth whose value depends on a design the operator's branch has not fixed yet.

- **The large lever is the grouped attention at 8B and 14B.** The code serves any group count, the two modes that measured it at 4B run on any sealed file today, and the expected effect at 8,192 positions is ×2.1 on both sizes: 31 to 68 tok/s at 8B, 21.5 to 44-46 at 14B, with +8 % and +6 % after a 5-token prompt (*estimated*: the 4B's measured slopes composed with the 8B and 14B launch counts, a sum METHODE §3 says must be measured). It is also the precondition of an O(1) speculative rollback, since `Grow` and `Prealloc` truncate by moving a length, and `Grow` reserves no window.
- **The 4B's task debt comes first.** ETAT §5 calls GSM8K and MMLU under the grouped form "the first thing to settle on this axis". GSM8K through the kernel under the grouped config costs about $2.0 (*estimated* on the expanded arm's $2.13, *measured*) and needs a job-script arm, no engine code. A loss outside the engine-gate criterion suspends the R1 proposals at 8B and 14B (`prereg-gestures-0.md`, `-2.md` §7). MMLU should not move at all: it scores by the prefill path, which read bit for bit at 4B; a 57-question smoke checks it.
- **The paid sequence.** 4B debt about $2.0, then the 8B control about $1.38, then the 14B about $2.31 only if the 8B lands in R1 or R2, then the gesture A/B about $0.35 once its code exists: about $6.0 in all, $12.15 at the ceilings (*estimated*, anchors in §5). Each step needs a go, a stamped prereg and the spending cap ETAT §5 says is owed.
- **The three small levers are bit-exact by construction**, worth together +0.5-1.3 % at the 4B's 7.50 ms step (*estimated*): a1, the argmax on the f16 logits, about 0.03-0.05 %; a2, a 75-CTA key argmax with candle's tie order, 0.15-0.6 %; a3, the token kept on the card, 0.3-0.6 %. The A/B now isolates each with its own ratio, and two direct probes settle a1's and a2's kernel times and a3's mechanism without leaning on a 0.4 % end-to-end ratio.
- **The side stream (c) is scoped to today's graph.** There it is worth +1.7-3.6 % (*estimated*), because v overlaps the one-CTA rotation. The operator's branch is fusing norm and rotation into a one-block kernel (audit step 2). If that kernel also writes v's input (design i), the fork moves after it and v overlaps q and k only: 0 to +2.4 %, or −0.5 to +1.5 % with the segmented q+k kernel. If v's int4 kernel takes the norm in its prologue (design ii), the window survives. So (c) is measured only on the commit that carries the fusion. Its fallback with cudarc's events on costs an estimated 1.7-3.2 %, as much as its gain.
- **The host's share.** Two facts suggest the host is off the critical path: the graph replay of 2026-09-01 saved the same 1.19-1.24 ms a token at three sizes while the eager step doubled, and the events trim removed thousands of driver calls for 0.0945 ms. That reading is *estimated* and rests on two assumptions (a host cost of at least 0.1 µs a removed call, and a 2026-09-01 step that had about 1,040 launches against 929 today). The clocks decide, and they are now read on the device-token step, after a3, because base's exposure is mostly the boundary a3 removes.

## 2. What exists today

| piece | where | state |
|---|---|---|
| greedy loop | `model.rs:2119-2155` | per token: head, f32 cast (`:2141`), candle argmax, `to_scalar` (`:2142`), `Tensor::from_slice` of the id (`:2147`) |
| same boundary elsewhere | `model.rs:2067-2091` (stepped), `:2216-2232` (`decode_greedy`, `argmax_last` `:2390`), `:2316-2350` (phased), `bin/gsm8k.rs:304-317` (EOS stop) | unchanged copies of the boundary |
| candle argmax | `candle-kernels-0.9.2/src/reduce.cu:478-523`, f16 instance `:673`, f32 `:688`; dispatch `candle-core-0.9.2/src/cuda_backend/mod.rs:321-365` | one CTA of 1,024 threads; uploads dims and strides from a `Vec` every call (`:333`); each element runs `get_strided_index` with a 64-bit `%` and `/` on `size_t` dims (`cuda_utils.cuh:25-38`) |
| candle's kernels from outside | `get_or_load_func` is public (`device.rs:227-245`); `kernels` re-exports `candle_kernels` (`cuda_backend/mod.rs:6`) | a probe can launch `fast_argmax_f16` itself with its info uploaded once |
| `to_scalar` | `tensor.rs:661-669` | rank 0 only; copies the storage, then indexes |
| the readback | cudarc 0.19.8 `memcpy_dtoh` into a `Vec` (`core.rs:1644-1653`), `SyncOnDrop::Sync(None)` for a `Vec` (`:1363-1380`) | pageable D2H: the driver makes the host wait |
| pinned memory | `alloc_pinned` (`core.rs:1404-1427`), `as_slice` (`:1451-1478`) | exists; write-combined, event with `BLOCKING_SYNC` |
| candle function lookup | `get_or_load_func` calls `load_function` (`core.rs:2215-2222`) | a `CString` and a `cuModuleGetFunction` on every candle launch |
| candle's stream | `CudaDevice::new` (`device.rs:279`) | the legacy NULL stream |
| our kernels | `fused_cuda.rs:389-444` | cached `CudaFunction` handles; each launch allocates its output (`:1419`, `:1572`, `:1765`) |
| gather by device id | `EmbedOp::cuda_fwd` (`fused_cuda.rs:1240-1271`) | honours the layout start offset, so a view into a ring works |
| events off | `fused_cuda.rs:2821-2844` | served door only, and only while one stream exists |
| multi-stream tracking | cudarc `core.rs:466-467`, `launch.rs:100-131`, `device_ptr` `core.rs:1166-1176`, `Drop` `:800-818` | with a second stream and tracking on: a stream wait and an event record per slice argument of every launch; 3.67 against 3.63 µs a launch on a near-empty chain (*measured*, `a3-graph-2026-08-06.txt`) |
| tracking flipped between arms | `fusedrun.rs:905-925` | the `LLVQ_TRIM_AB` template, checked single-stream there |
| kernel text override | `LLVQ_KERNEL_DIR`: `fused.rs:916-960`, `fused_cuda.rs:63-75`; refused beside `LLVQ_CONFIG` (`served.rs:111`) | all or nothing over the 10-file served unit, 101,541 bytes, which the image does not carry (wave-3 digest E7) |
| launch geometry | `launch_q4_h` (`fused_cuda.rs:2269-2297`), Tetra48 grid `d_out × 32 / 256` (`:2393`, `THREADS` `:51`) | at 4B: q 512 CTAs, k 128, int4 v 128; order rot, q, k, v (`rotplan.rs:519-545`) |
| A/B templates | `run_trim_ab` (`fusedrun.rs:187-241`), `LLVQ_GQA_AB` (`:263-436`, `:775-815`), `LLVQ_GQA_PPL` (`:535-620`, `:708-766`) | TRIM_AB refuses any store but `Cat`, so a new mode sets the grouped state itself |
| hook per layer | `Capture` (`model.rs:85-97`), called after ln1 at `:1609`, threaded through `hidden_cached` (`:1913-1919`) | usable as a timing hook with no change to `model.rs` |
| configs | `configs/qwen3-4b-tetra-e4-grouped.json` | the only file with `"attention": "grouped"`; 8B and 14B expanded |
| attention field | `served.rs:29-37`, `:218-229`, `apply` `:297-307`, Metal refusal `:262-288` | optional; sets store and form together, for `fusedrun`, `mmlu` and `gsm8k` |
| grouped attention | `grouped_attention` (`model.rs:571-593`), `group_view` (`:509-517`) | any divisor of the head counts |
| stores | `KvStore` (`kvq.rs:414-470`), `truncate` (`model.rs:421-446`) | `Grow(256)` and `Prealloc` move a length, `Cat` copies the kept prefix; `Prealloc` reserves its expanded window, 4.83 GB at 8k at 4B (*computed*, handover of 2026-11-01) |
| tests | `attention_forms_agree` (`model.rs:2790`), `stepped_decode_matches_generate_on_all_stores` (`:2724`) | groups of 2 and 4 only: no test runs a group of 5, the 14B's shape |
| job templates | `ops/jobs/gqa-ab.sh`, `gqa-ppl.sh`, `served-grouped.sh`, `gsm8k-served.sh` | `gqa-size.sh` is rebuilt from the first two; the served text is extracted at `served-grouped.sh:133-137` |

Works: the grouped form at 4B, served since 2026-11-02; both GQA modes, which read the file, the tokenizer and wikitext-2 and nothing 4B-specific; candle's argmax already runs on the device. Missing: every line of code for (a) to (c); the 8B and 14B grouped configs and controls; a group-of-5 test; any task score under the grouped form; any clock of the host span; any journal of the one-CTA argmax time or of the v kernel in situ.

## 3. Design

### 3.1 (a) The token stays on the card

Four loop kinds sit beside `generate`, each one mechanism from its neighbour, so each A/B ratio isolates one gesture (`code/gestures/decode_on_device.rs`):

| kind | per token | ratio | gesture |
|---|---|---|---|
| `Base` | `generate` unchanged | | today |
| `F16` | Base without the cast | `F16 / Base` | a1 |
| `Key` | F16 with `tok_argmax_h` into a one-slot ring, `to_scalar` of its rank-0 view, `from_slice` | `Key / F16` | a2 |
| `OnDevKey` | Key without the per-token readback and upload: the next step embeds `ring[pos]` | `OnDevKey / Key` | a3 |
| `OnDevCandle` | F16 without the per-token readback and upload: candle's argmax output is the next input, no copy | `OnDevCandle / F16` | a3 with candle's kernel |

The first draft routed the f16 arm through a ring copy, which added a launch to the cast it removed, and left the key arm's readback undefined; `to_scalar` also refuses a rank-1 view (`tensor.rs:662`). `OnDev / Base` moves three mechanisms and is reported only.

**a1, argmax on the logits as stored.** A `logits_last_native` without the f32 cast serves the greedy loops; `logits_last` keeps the cast for callers that sample. Exactness: `fast_argmax_f16` and `fast_argmax_f32` are one template with the same block of 1,024, and f16 to f32 widening is exact and monotone, so every `>` gives the same answer; signed zeros tie in both, NaN compares false in both. It removes one launch and one 608 KB allocation a token, about 2-4 µs (*estimated* on about 2.2-2.3 µs a cast launch in situ, *computed* from `trims-contexts-2026-10-30.txt`).

**a2, a key argmax.** `tok_argmax_h` (`code/gestures/tok_argmax.cu`) reads the f16 logits and writes `ring[pos]`.
- Candle's tie rule, from the code: thread t scans i ≡ t (mod 1,024) with a strict `>` from −inf, so the earliest index wins inside a thread; the tree keeps the left operand on ties from s = 512 down to 1, so among tied threads the smallest `bitrev10(t)` wins.
- Key: `ord16(v) << 32 | (1023 − bitrev10(i & 1023)) << 22 | (0x3FFFFF − (i >> 10))`, with −0 mapped to +0 before `ord16`. A max over unique keys is order-free, so any grid returns candle's index.
- Geometry: 256 threads, 75 CTAs for 151,936 logits, one 16-byte load a thread, one wave on 142 SMs. Shared 384 B. State: one u64 slot, one u32 ticket, one u32 flag; the last CTA (ticket pattern after `__threadfence`) reads and resets the slot, resets the ticket, decodes the index.
- Integration: a `KeyKernel` trait in `model.rs`, implemented by the kernel as a candle `InplaceOp2` on the ring in `fused_cuda.rs` (no allocation, no upload), and by the host reference in CPU tests.
- NaN: candle's result then depends on tree position, which no commutative key reproduces. The kernel keys a NaN 0, raises a sticky flag and writes token 0, an in-bounds id; the loop refuses the generation. That is a stricter behaviour on broken logits, the operator's to accept.
- Cost: 4-6 µs (*estimated*), against 15-50 µs for candle's kernel, whose 151,936 iterations each run a 64-bit modulo and division on one SM (*estimated* from the code; the fenced boundary measured 0.078-0.101 ms with it, `phases-2026-08-07.txt`).
- Checks (`code/gestures/argmax_key_ref.rs`): a host replay of `fast_argmax` against the key on 8 adversarial cases, 300 sparse tie sets and one dense set, plus the NaN flag. The sparse sets put the maximum at 2 to 6 random positions over four background levels with signed zeros, as the wave-2 simulation did. The first draft's random sets drew every element from six levels, so thread 0 held the maximum with probability 1 − 2.3e-9 and the cross-thread order was never exercised (*computed*). Its signed-zero case also had the −0 at the losing position, so it could not kill mutant m1 (*computed*, `calc/gestures-check.py`).
- Card check at load: 41 calls in one sequence through one slot, ticket and flag (8 adversarial, the first 32 sparse sets, the NaN case last), against candle's own `fast_argmax_f16` on the card. The order kills the state mutants: case 1 holds the largest key, so a slot that is not reset poisons later answers.
- Card mutants (rule 10): the A/B mode applies m1-m8 to the embedded kernel text in memory, counts each replaced string exactly once, compiles each mutant as its own NVRTC module, runs the 41-call sequence, and requires 8 of 8 dead. m5 to m7 (ticket reset, `atomicExch`, `atomicMax`) exist only on the card. `LLVQ_KERNEL_DIR` would need 8 complete copies of the 10-file served unit shipped to the bucket, since the image carries no source; the in-memory route needs none.
- Evidence now: 0 mismatches on 300 integer sparse sets (wave-2 simulation) and on 12 f16 sparse sets with signed zeros (*computed*, `calc/gestures-check.py`).

**a3, no per-token wait.** `generate_loop` with `OnDevKey` keeps a `[max_new]` u32 ring on the device. Step t+1 embeds the view `ring.narrow(0, t, 1)`; one pageable readback ends the decode.
- Why it should matter: the CUDA runtime documents that an async copy to or from pageable memory "might be synchronous with respect to host", and the step holds two such uploads, the token and candle's argmax info. Small pageable uploads are usually staged without waiting, so the expected gain is the boundary bubble, not a sync. The pageable-copy probe of §3.5 says which.
- Run-ahead is bounded by the driver's launch queue only. Memory stays bounded: every temporary is freed stream-ordered on the same stream.
- EOS callers (`bin/gsm8k`, `bin/chat`) take a lag-1 form: copy `ring[t]` into a pinned slot, test token t−1 while step t+1 runs, and on EOS drop one wasted step with `truncate`, O(1) on `Grow` and `Prealloc`.
- Exactness: the same kernels in the same order on one stream; only host waits move. A host test pins every kind on `Cat`, `Grow` and `Prealloc`, under the three forms, in f32 and f16, with groups of 2, 4 and 5; the A/B pins it on the card.
- Value: the boundary bubble (wake-up, readback, upload, the first ops starving the device) is about 20-45 µs (*estimated*). With a1 and a2: 0.04-0.10 ms, +0.5-1.3 % at 7.50 ms (*estimated*).

### 3.2 (b) Host diet, and host exposure without nsys

Per 4B token on the grouped path (*computed* from the code and U1): 929 launches, about 859 `cuMemAllocAsync` with as many frees (350 from our ops, about 507 from candle's), 507 `get_or_load_func` (a `CString` and a driver lookup each), 2 pageable uploads, 1 pageable readback.

**What two facts suggest, and why they do not decide.** The graph replay saved 1.192, 1.204 and 1.242 ms a token at 4B, 8B and 14B while the eager step went from 10.08 to 13.12 to 21.32 ms (*computed* from `a2-verdict`, `a2-transfert-8b`, `a2-transfert-14b`). An exposed host should be hidden more by a longer device step, and that saving did not shrink. The events trim removed on the order of 10⁴ driver calls a token (audit of 2026-10-29) and saved 0.0945 ms (*computed* from `trims-contexts-2026-10-30.txt`). The reading that the host is hidden is *estimated*, on two assumptions:
- each removed driver call cost the host well over 0.1 µs, which no journal measures, and `cuStreamWaitEvent` also puts commands on the device, so part of the 0.0945 ms may be device time;
- the 2026-09-01 step stands for today's: it ran about 1,040 launches (≈750 candle, 288 ours, `a2-verdict`) on a Prealloc(256) store, a fresh stream and events off, in 10.08 ms; today's is 7.44 ms with 929 launches, about 0.74 of it (*computed*), so the margin the reading rests on is smaller.
If it holds, the host enqueues a step in about 2.5-5.5 ms against about 7.4 ms of device work, and the graph's 1.2 ms was mostly device-side launch cost, which only fewer launches (route D or P) recover; the device-side cost of a launch is 3.76 µs on the bench even with the host ahead (*computed*, `a1-nullk-252-144-2026-08-31.txt`).

Diet items, built only if the clocks show exposure after a3: b1, our 350 outputs written in place into per-site buffers (`InplaceOp1/2`, same kernels, 2-3 days); b2, a cache of candle's function handles, which means a vendored `candle-core` patch, a dependency change for the operator; b3 and b4 are a2 and a3.

**The clocks, read on the device-token step** (`code/gestures/gesture_ab.rs`). The first draft timed the base step behind two spins. Two defects: candle's argmax inside the timed window uploads from pageable memory, which may synchronize and void every step; and base's exposure is mostly the per-token boundary, which a3 removes, so a diet decided on it could prescribe work a3 already made useless.
- `E_base`: the base pass's wall time between consecutive readbacks, decode steps only, 5 × 64 tokens. `E_ondev`: the `OnDevKey` pass, synchronized after the prefill, from the first decode enqueue to the final readback, over 63 steps, 5 passes.
- H and D: the `OnDevKey` loop, 5 × 64 steps, six segments a step, each behind `spin_ns(5 ms)` on candle's stream, launched at the step start and by the capture hook after ln1 of layers 6, 12, 18, 24 and 30. About 155 launches a segment: the host launch queue depth is undocumented, one NVIDIA forum report shows launches blocking past about 1,000, and each launch here also brings an allocation and a free.
- A step is valid only if every segment was enqueued at least 1 ms before its spin could end. No pageable copy happens inside a step: the token comes from the ring view and the key kernel writes the ring.
- H is the host enqueue span, D the device time, both summed over segments. Reported: slack = D − H, X_ondev = E_ondev − D, and E_base − E_ondev, the bubble a1 + a2 + a3 recover.
- Rule (prereg-gestures-1 §7, B0-B4): no diet if slack ≥ 0.10 ms and X_ondev ≤ 0.10 ms; b1 if −0.40 < slack < 0.10 ms; the diet as the next gesture and b2 to the operator if slack ≤ −0.40 ms; a kernel timeline if slack ≥ 0.10 ms but X_ondev > 0.10 ms; nothing if fewer than 90 % of steps are valid.

### 3.3 (c) The int4 v_proj on a side stream

**Today's graph.** In all three sealed files `v_proj` is int4 and reads the unrotated ln1 output, while q and k wait for the shared rotation, one CTA for 8.05 µs at n = 2,560 (*measured*, `rotation-cuda-2026-08-05.txt`). The single stream runs ln1, rot, q, k, then v. Forked after ln1, v's 128 CTAs run on the 141 SMs the rotation leaves idle: 36 × (v time − join cost) = 36 × (3.5-7.5 µs) = 0.13-0.27 ms, +1.7-3.6 % at 4B (*estimated*; v in situ 5-9 µs and a join 1-2 µs, both *estimated*).

**The graph the operator's branch is building.** Step 2 of the audit fuses norm and rotation, two operations on one vector that one block can hold (audit T1), under the tolerance gate. sm_89 has no programmatic dependent launch, so a consumer cannot start before its producer kernel ends.
- Design (i): the fused kernel also writes v's unrotated input. The fork moves after it, and v overlaps q (512 CTAs) and k (128 CTAs) only. k occupies about 15 % of the card's thread slots (*computed*: 128 × 256 threads of 142 × 1,536), so part of v can still hide there, and v's own launch gap hides. Value: 0 to +2.4 % without the segmented kernel; with `tv_tetra48_seg` serving q+k as one 640-CTA launch, −0.5 to +1.5 % (*estimated*).
- Design (ii): v's int4 kernel takes ln1 in its prologue, reading the residual and the 2,560 ln1 weights; each of its 128 CTAs reduces the row redundantly, about 655 KB of L2 reads (*computed*). The fork sits after the residual add, and v overlaps the one-block fused kernel, at least as long as today's rotation: the window survives, +1.7-3.6 % or more (*estimated*).
- What the branch needs from this lead: under design (i), (c) is probably not worth its code; under design (ii) it is the cheapest gesture left. Either way the fused kernel's numerics are under T8 already, so design (ii) costs no new gate.

**So the A/B of (c) runs only on a commit that carries the fusion** (prereg-gestures-1 §3, Part C), and that commit is named in the prereg. Otherwise (c) stays out of the A/B and out of the bundle.

**Design** (`code/gestures/vstream.rs`):
- One non-blocking side stream per model (cudarc `new_stream`, so no implicit sync with candle's legacy NULL stream), created at load after events were turned off, so cudarc also skips its context synchronize (`core.rs:674-679`).
- Fork: allocate v's output on the main stream, record an event on main, the side stream waits on it, the v kernel runs on the side stream, an event is recorded after it. Main runs the rotated sites as today and waits on that event before `qkv[2]` is read (`model.rs:1626`).
- Lifetimes: the forked call holds an `Arc` clone of v's input until the join; dropped earlier, it would be freed stream-ordered on main and its bytes could serve the next allocation while v still reads them. v's output is freed on main, like every tensor.
- Generic hook: `Proj::forward_forked` returns `None` off CUDA and when no side stream is armed, so the published path is the same code with the hook off.
- Exactness: the same kernel, input and output buffer; nothing touches that buffer before the join.
- Race pre-check, on the card, before Part C's timed rounds: with `spin_ns(50 µs)` on the side stream before each v, 16 tokens must equal base; the no-join mutant with the same delay must diverge within 16 tokens, or Part C is void, since a delay that a missing join survives proves nothing; the context is synchronized after it.

**Events.** Off on two streams revisits the argument written at `fused_cuda.rs:2821-2844`; the operator rules. The fallback keeps tracking on, and its cost is not measured. It is above the 1.15 % the events trim bought (*measured* on one stream, `trims-contexts-2026-10-30.txt`, arm `+events` 1.0115 [1.0080-1.0143]), because one stream only creates, waits on and destroys two events per allocation. With a second stream, `is_managing_stream_synchronization()` is true (`core.rs:466-467`), and every launch also pushes a stream wait and an event record per slice argument (`launch.rs:100-131`), for candle's ops and ours. The one measurement of that part is 3.67 against 3.63 µs a launch on a near-empty chain (*measured*, `a3-graph-2026-08-06.txt`). With one to four slice arguments a launch, the fallback comes to about 1.7-3.2 % in all (*computed* from one measured and one estimated input), the size of (c)'s whole gain, plus two driver calls per slice argument on the host. The A/B therefore carries the fallback as two arms, `BaseEv` and `VStreamEv`, which flip tracking around their allocations as `LLVQ_TRIM_AB` does, so rule C can be applied under either ruling.

### 3.4 (d) The grouped form at 8B and 14B

No CUDA code changes: `grouped_attention` takes any divisor, 32/8 = 4 at 8B and 40/8 = 5 at 14B, and `Grow` is size-free. Four things are missing.
- The 4B's task debt, first (`prereg-gestures-0.md`): GSM8K through the kernel under the grouped config, paired with the expanded census (1,088 of 1,319, 82.49 %, *measured*, `gsm8k-wave1-2026-09-26.txt`), under the engine-gate criterion of 2026-09-26 (|Δ| ≤ 1.5 pp and McNemar exact p ≥ 0.05), plus the 57-question MMLU smoke under both configs. About $2.0 (*estimated*), a job-script arm, no engine code (`bin/gsm8k` applies the config's attention, `bin/gsm8k.rs:184-188`).
- A host test with a group of 5 (`tiny_model_heads(dtype, 10, 2)`) in `attention_forms_agree` and in the token-loop test.
- The control, through the existing modes on the bench door (`code/gestures/gqa-size.sh`, rebuilt line for line from `gqa-ab.sh` and `gqa-ppl.sh`): oracle; the published arm replayed against the dense one, gate 4 on the decoded line (`323cabeb…` at 8B with 256 identical tokens, `186974aa…` at 14B with its divergence at token 78, *computed* from the paper-table brut files); `LLVQ_GQA_AB=5,2048,4096,8192`; `LLVQ_GQA_PPL=4096,12`. The 14B runs only after the 8B landed in R1 or R2: the 8B is the cheaper probe, $1.38 against $2.31 (*estimated*).
- After a pass and a ruling: `configs/qwen3-8b-tetra-e4-grouped.json` and `configs/qwen3-14b-tetra-e4-grouped.json`, the e4 files plus `"attention": "grouped"`, then a served-door sweep, as at 4B (control, config, served door).

An option for the operator, before the stamp of `prereg-gestures-2.md`: half a day of engine code in `run_gqa_ppl` would also print the published arm's own decode-against-prefill drift, the reference the 4B's 13.13 logit spike lacks. Default: not taken, and the control needs no engine code.

Expected (*estimated*; 4B slopes 0.67 and 2.69 ms a thousand positions *computed* in `gqa-ab-2026-11-01.txt`; 8B attention shapes identical to the 4B's; 14B slopes scaled by bytes a position, ×1.35 expanded and ×1.09-1.25 grouped; trims and launch removals scaled by layer, int4 and launch counts):

| decode tok/s | expanded after trims, 5 tokens | grouped, 5 tokens | expanded, 8,192 | grouped, 8,192 |
|---|---|---|---|---|
| 8B (served untrimmed: 95.0, *measured*) | 99 | 107 | 31.1 | 67.6 |
| 14B (served untrimmed: 57.2, *measured*) | 59 | 63 | 21.5 | 43.9-45.8 |

On the L40S at batch 1, the 14B's grouped 8k figure is about 2.2 times the product's 20 tok/s chat floor, against about 1.07 times expanded (*estimated*). That floor is a batch-1 chat floor defined for the target card, a 5090, not the L40S, so the comparison is an order of magnitude only (`note-produit-2026-08-13.md` A5; its batch of 8 belongs to the extraction floor, 60 s a document).

Memory: `Grow` holds at most 255 spare positions, 37.6 MB at 8B and 41.8 MB at 14B, 0.037 and 0.023 b/param over the whole model (*computed*). `Cat` doubles the history transiently at every step instead. Rollback: on `Cat` a rejected speculative round copies the kept prefix, 1.31 GB at 8,192 on the 14B (*computed*: 160 KB a position); on `Grow` and `Prealloc` it moves a length, and of those two only `Grow` reserves no window.

### 3.5 (e) The A/B mode

`LLVQ_GESTURE_AB=1` (`code/gestures/gesture_ab.rs`): one model in the served 4B state (`Grow(256)` + grouped), events off before any allocation. In order, each step gating the next:
- P1, the key's load check (41 calls); P2, the eight card mutants. Either failure voids the run.
- P3, the kernel-time probe, on the real f16 logits, 5 × 100 back-to-back launches each, event-timed: candle's `fast_argmax_f16` and `fast_argmax_f32` launched directly with their info uploaded once, the logits cast, `tok_argmax_h`, and candle's full `argmax` call. It settles U4 and a1's 2-4 µs directly.
- P4, the pageable-copy probe, five times each behind `spin_ns(5 ms)`: host time of `Tensor::from_slice` of one id, of candle's `argmax` on the logits vector, and of `to_scalar` (the control, which must wait). Microseconds mean fire-and-forget, about 5 ms means a sync. It settles U5's mechanism before any arm is read.
- Part C only: the race pre-check with its no-join arm.
- The arms, every arm in every round, 2 rounds discarded and 5 timed: Part A `Base`, `F16`, `Key`, `OnDevKey`, `OnDevCandle`; Part C `VStream`, `BaseEv`, `VStreamEv`, `Bundle`. Any token that differs from base in any arm or round voids the run.
- The clocks, then Part C's v-in-situ pass.
- Refused beside `LLVQ_CONFIG`, beside any other mode, and with `LLVQ_KV_PREALLOC`.
- No new binary, so no change to `ops/Dockerfile.cuda` or `UPLOAD_ALLOW`. Kernels ship through `include_str!`, as `emb_q8.cu` does (`fused_cuda.rs:61`).

## 4. Deciding unknowns

| unknown | best value now | confirms | kills (a proposal; the kill is the operator's) |
|---|---|---|---|
| U1 paired ppl, grouped vs published, 8B and 14B | 4B: +0.0000 % prefill, +0.0095 % [−0.0048; +0.0238] decode (*measured*) | both intervals inside ±0.12 % at a size | an interval leaves the band: that size stays expanded |
| U2 grouped/expanded speed, 8B and 14B | ×2.17 and ×2.04-2.13 at 8,192; ×1.08 and ×1.06 at 5 (*estimated*) | ranges above 1.000 at all four lengths | a range not entirely above 1.000 at a length: not served at that size, a timeline first |
| U3 (c)'s window in the served graph | today +1.7-3.6 %; design (i) 0 to +2.4 %, −0.5 to +1.5 % with the segmented kernel; design (ii) as today or more (*estimated*) | `VStream/Base` range above 1.005 on the fused commit | range below 1.000, or identity fails |
| U4 candle's one-CTA argmax | 15-50 µs (*estimated* from the code) | P3 reads it above `tok_argmax_h`, and `Key/F16` not below 1.000 | P3 reads it at or under the key kernel: candle's argmax stays, the ring stays |
| U5 the boundary bubble, and its mechanism | 20-45 µs; small pageable uploads fire-and-forget (*estimated*) | P4 under 0.5 ms and `OnDevKey/Key` above 1.000 | P4 over 4 ms: candle's upload syncs, a3 needs the key kernel; `OnDevKey/Key` below 1.000: a hidden sync to find |
| U6 host exposure after a3: slack = D − H | 3.5 ms [0.8, 5.5] (*estimated*) | slack ≥ 0.10 ms and X_ondev ≤ 0.10 ms: no diet | slack < 0.10 ms: b1; ≤ −0.40 ms: the diet first, b2 to the operator |
| U7 launch queue depth under spin | undocumented; one forum report near 1,000 | ≥ 90 % of spun steps valid at six segments | most steps invalid: H and D not measured, no ruling on (b) |
| U8 key = candle on the card | 0 mismatches on 300 integer and 12 f16 sparse sets (*computed*) | the 41-call load check passes and 8 of 8 mutants die | any mismatch or survivor stops the load |
| U9 the fallback's cost, tracking on with two streams | 1.7-3.2 % (*estimated*: 1.15 % *measured* on one stream, plus per-argument waits) | `BaseEv/Base` range above 0.990 | range below 0.970: the fallback eats (c)'s gain |
| U10 the 4B's task debt under the grouped form | GSM8K Δ +0.0 pp [−1.0, +1.0]; MMLU picks identical by the prefill argument (*estimated*) | G1 and M1 of `prereg-gestures-0.md` | G2: the 8B and 14B R1 proposals are suspended |
| U11 the fusion branch's design | not fixed yet | design (ii): the window survives | design (i) with the segmented kernel: (c) near zero |

## 5. Validation ladder

Each paid step needs the operator's go, a cap (none in force, ETAT §5), its prereg stamped before the first millisecond, and its cost announced before and the running total after.

| # | settles | method | class, cost | prereg | decision rule | needs |
|---|---|---|---|---|---|---|
| 0 | exactness of a1, a2, a3, c; the scope of c; the host reading's assumptions | this document and its review | zero-cost, $0 | no | done | nothing |
| 1 | the plumbing, the key, groups of 5 | host tests: every loop kind = `generate` on three stores, three forms, two dtypes, groups of 2, 4, 5; the key against the candle replay (8 adversarial, 32 sparse in debug, 300 in release, one dense, NaN); `attention_forms_agree` with 10/2 heads; host mutants | local run, minutes of `cargo test` on the operator's machine | no | all green after mutation, else fix before any image | routine |
| 1b | the job script | `SIZE=8b DRY_RUN=1` and `SIZE=14b DRY_RUN=1 bash ops/jobs/gqa-size.sh` | zero-cost, seconds, local | no | both parse and the refusals fire, else fix | the script committed |
| 2 | U10 | GSM8K through the kernel under the grouped config, 1,319 problems; the 57-question MMLU smoke under both configs | paid, ~65 min, ~$2.0 (anchor `gsm8k-4b-served` 71 min $2.13); ceiling $4.50 | yes, `prereg-gestures-0.md` | G1/G2/G3 and M1/M2 of that prereg | a job-script arm, go, cap |
| 3 | U1, U2 at 8B | `gqa-size.sh SIZE=8b` | paid, ~46 min, ~$1.38 (anchors `gqa-ab-4b` 10 min $0.29, `gqa-ppl-4b-r2` 28 min $0.82); ceiling $2.70 | yes, `prereg-gestures-2.md` | R1/R2/R3 of that prereg, R1 suspended under G2 | go, cap |
| 4 | U1, U2 at 14B | `gqa-size.sh SIZE=14b`, after an 8B R1 or R2 | paid, ~77 min, ~$2.31; ceiling $3.60 | same prereg | same rule | go, cap, the 8B verdict |
| 5 | U4 to U8, and U3, U9 if the fusion has landed | `LLVQ_GESTURE_AB` on the 4B, Part A (Part C on a fused commit) | paid, ~12 min, ~$0.35, Part C +$0.04 (anchor `trims-contexts-4b` 12 min $0.35); ceiling $1.35 | yes, `prereg-gestures-1.md` | A1, A2, A3, B (C) of that prereg | code (§6), go, cap, the base commit named |
| 6 | U3, U9 on the fused graph, if step 5 ran without Part C | the same mode, Part C, on the fusion commit | paid, ~12 min, ~$0.39; ceiling $1.35 | yes, `prereg-gestures-1.md` Part C | C | the operator's fusion commit, the events ruling |
| 7 | the served object at 8B, 14B | served-door sweeps with the new configs | paid, ~$0.75 and ~$1.05 (*estimated* on `served-grouped-4b` $0.54) | yes | as `served-grouped-2026-11-02` §6 | T8 ruling per size |
| 8 | task debt at 8B, 14B | GSM8K through the kernel, grouped | paid, ~$2.2 and ~$3.9 (*estimated* on $2.37 and $4.24, jobs.csv) | yes, on the model of `prereg-gestures-0.md` | its G rules | operator |

Steps 2 to 5 cost about $6.0 (*estimated*: 2.0 + 1.38 + 2.31 + 0.35), $12.15 at the ceilings; the 14B's $2.31 is spent only after an 8B R1 or R2. Steps 2 to 4 need no engine code, so they come first in time as well as in rank.

## 6. Build plan after validation

1. **(d), on a passing step 3 or 4, G1 at step 2, and the ruling, 0.5 day.** Config file per size; `configs/README.md`; the served-door sweep (step 7); ETAT §2 and §4 on the result. Tests: `served.rs` parse tests for the new files; the group-of-5 host test stays.
2. **(a), 2-3 days.** `logits_last_native`, `KeyKernel`, `generate_loop` with its four kinds; the key kernel, its `InplaceOp2`, its 41-call load check sharing the case list with the host test. The served door switches to the chosen loop as an engine default, like the trims, with no config field because the tokens are the same; that is the operator's call. Then the lag-1 form for `bin/gsm8k`. Tests: the host test of §3.1; the host replay; the load check on the card; host mutants in the loop (wrong view, no `offset += 1`, off-by-one write, early break, `OnDevCandle` embedding the first pick).
3. **(e), 1.5-2.5 days.** The mode in `fusedrun`, its refusals, P1 to P4 with the in-memory mutants and their counts, the clocks. Type-check on the Mac what builds there; `fused_cuda.rs` is first compiled by the image, 16-18 min a build.
4. **(c), 2-3 days, after the fusion lands and the events ruling.** Side stream at load, `forward_forked` at the fork site of the fusion's design, the join, the `-ev` arms, the race pre-check and its no-join arm. Mutants: no join (the pre-check), join after `qkv[2]` is read, output allocated on the side stream, `keep_x` dropped before the join (caught by ownership, not by the delay).
5. **(b), only under B2 or B3.** b1 first (bit-identical); b2 waits for a ruling on a vendored candle.

Gates: identity of tokens in every arm and round (a, c); the load check and the card mutants (a2); the paired band per size (d); the task criterion at 4B before any R1 proposal. Benches inform, the end-to-end gate decides. Effort about 6.5-11 days for (a) + (c) + (e) (*estimated*, before the factor-2 optimism the verifiers found in every plan).

## 7. Rules map

- **T8.** (d) is a new T8 event at each size: the grant of 2026-11-01 covers the 4B only. (a), (b) and (c) claim bit identity against their base commit, and a failed identity voids the run instead of opening a T8. The fusion's own T8 belongs to the operator's branch.
- **METHODE §1, one mechanism.** The arms form chains in which each ratio moves one mechanism; ratios that move several (`OnDevKey/Base`, `VStreamEv/Base`) are reported, and rule C reads `VStreamEv/Base` only when the operator's refusal of events off binds tracking to the fork.
- **A2 and rule 3.** No gesture captures a graph. The device-token loop still launches every kernel from the host and promises no window, so it is not A2. 4E, which loops N tokens inside a kernel, is a different lead.
- **Rule 6.** Ring 1 KB, key state 16 B, pinned host memory outside VRAM. `Grow` slack at most 0.037 b/param at 8B and 0.023 at 14B over the whole model (*computed*).
- **Rules 4, 5, 7.** Every arm reads the same int4 head, so the raw ratio is the same-head ratio; one card; medians with round-by-round ranges.
- **Rule 10.** Oracle first in every job; the kernel mutants die on the card, the loop mutants on the host; the race pre-check carries its own mutant.
- **Closed leads.** ETAT §6 binds A2 only here, untouched. The speculative verdict of 2026-08-05 (docs/archive) is not reopened: (d) stands on its 8k speed, and its rollback role is noted, not used.
- **Decision 12 of `rapport-debit-vague2.md` §11, candle ops.** The key argmax replaces candle's argmax on the condition that decision's default sets: same index, checked at load. b2 would patch candle itself, a separate ruling.
- **Operator rulings.** A cap; a go per job; the 4B debt before the 8B and 14B controls; the 14B after the 8B; T8 per size; the optional drift reference in the control; events off with two streams; the fusion design, which decides (c); the new loop as the served default; NaN refusal; b2 if ever needed.

## 8. Checked now at $0

1. The loop pays a cast, a blocking readback and an upload a token: `model.rs:2141`, `:2142`, `:2147`.
2. `fast_argmax` starts at −inf, keeps the earliest index in a thread and the left operand on ties, block 1,024 for 151,936 logits: `reduce.cu:478-523`, `cuda_backend/mod.rs:324`.
3. The f16 and f32 argmax are one template: `reduce.cu:526-538`, instances `:673` and `:688`.
4. Every element of that scan runs a 64-bit `%` and `/` (`unsigned int` against `size_t`): `cuda_utils.cuh:25-38`.
5. The argmax uploads dims and strides from a `Vec` on every call: `cuda_backend/mod.rs:333`.
6. `get_or_load_func` is public and `candle_kernels` is re-exported, so a probe can launch candle's kernel with its info uploaded once: `device.rs:227-245`, `cuda_backend/mod.rs:6`.
7. `to_scalar` refuses any rank but 0: `tensor.rs:661-669`.
8. cudarc does not itself synchronize after a copy from a `Vec`: `SyncOnDrop::Sync(None)` (`core.rs:1363-1380`, `:1130-1145`).
9. The CUDA runtime allows a pageable async copy to synchronize: "might be synchronous with respect to host" (docs.nvidia.com, API synchronization behavior).
10. Candle looks a kernel up by name on every launch: `device.rs:227-245`, `core.rs:2215-2222`.
11. Candle runs on the legacy NULL stream; cudarc's new streams are non-blocking, switch the context to multi-stream mode, and synchronize the context only when tracking is on: `device.rs:279`, `core.rs:670-685`.
12. In multi-stream mode with tracking on, every slice argument of every launch adds a stream wait and an event record: `core.rs:466-467`, `launch.rs:100-131`, `core.rs:1166-1176`.
13. That per-launch part was measured once: 3.67 against 3.63 µs a launch on a near-empty chain, `a3-graph-2026-08-06.txt`.
14. The events trim's 1.15 % was measured on one stream: arm `+events` 1.0115 [1.0080-1.0143], `trims-contexts-2026-10-30.txt`; `fused_cuda.rs:2834-2843` turns tracking off only with one stream; `LLVQ_TRIM_AB` flips it between arms (`fusedrun.rs:905-925`).
15. The gather reads the id at the view's offset: `fused_cuda.rs:1240-1271`.
16. v is int4 in every layer of the three sealed files and reads the unrotated input; the order is rot, q, k, v: the three e4 configs, `rotplan.rs:519-545`, `model.rs:1132-1208`.
17. At 4B q runs 512 CTAs, k and the int4 v 128 each: grid `d_out × 32 / 256`, `fused_cuda.rs:2283`, `:2393`, `THREADS` at `:51`.
18. The audit's step 2 fuses norm and rotation, a one-block kernel, under the tolerance gate: `audit-debit-2026-10-29.md` T1 and §5.
19. The graph replay saved 1.192, 1.204, 1.242 ms a token at 4B, 8B, 14B: 99.2→112.5, 76.2→83.9, 46.9→49.8 tok/s in the three A2 journals (*computed*). The 4B eager step there: 10.08 ms (*computed*), ≈750 candle and 288 of our launches, a Prealloc(256) base, a fresh stream with tracking off (`a2-verdict`, `a2-e1b`, `gpu.rs:300-330`).
20. The events trim saved 0.0945 ms a token: 121.0 against 122.4 tok/s (*computed*).
21. The fenced boundary costs 0.078-0.101 ms: `phases-2026-08-07.txt` (*measured*).
22. Both GQA modes are size-free: `fusedrun.rs:337-436`, `:535-620`.
23. `attention_forms_agree` covers groups of 2 and 4 only: `model.rs:2790-2800`, `tiny_model_heads` `:3065-3085`.
24. `Grow` and `Prealloc` truncate by moving a length, `Cat` copies the kept prefix: `model.rs:421-446`.
25. The 8B and 14B reference lines hash to `323cabeb…` and `186974aa…`, each the only `  fused F0:  ` line of its file, and the 14B diverges from dense at token 78: the paper-table brut files.
26. The served text of both configs on 2026-11-02 hashes to `0597cee8…`, extracted by `served-grouped.sh:133-134`; the bench line hashes to `8784ffc4…`: `served-grouped-2026-11-02-brut/lines.sha256`.
27. `gqa-size.sh` keeps every line of both template bodies except the parameterized dense line, its echo and the step number (a line-by-line comparison against `gqa-ab.sh` and `gqa-ppl.sh`).
28. The chat floor of 20 tok/s is the chat/agent floor, the batch of 8 belongs to the extraction floor, and the target card is the RTX 5090: `note-produit-2026-08-13.md:6`, `:131-137`; `PLAN.md:1004-1007`.
29. ETAT §5 ranks the grouped form's task debt "the first thing to settle on this axis".
30. GSM8K at 4B, expanded: 1,088 of 1,319, 82.49 %, 8 at the cap, 316.1 tokens, 4,251 s, $2.13: `gsm8k-wave1-2026-09-26.txt`, `docs/data/jobs.csv`.
31. `bin/gsm8k` applies the config's store and attention: `bin/gsm8k.rs:184-188`; the grouped prefill read bit for bit and the decode top-1 agreed on 49,036 of 49,140 steps: `gqa-ppl-2026-11-01.txt`.
32. `LLVQ_KERNEL_DIR` is all or nothing and the image carries no source: `fused.rs:916-960`; wave-3 digest E7 (10 files, 101,541 bytes).
33. The first draft's random tie test kept the maximum in thread 0 with probability 1 − 2.3e-9, its signed-zero case did not kill m1, and the swapped case does; the key matches the candle replay on 12 f16 sparse sets: `calc/gestures-check.py` (*computed*).
34. Cost anchors: 237 rows, $248.84; `gqa-ab-4b` 10 min $0.29; `gqa-ppl-4b-r2` 28 min $0.82; `served-grouped-4b` 18 min $0.54; `trims-contexts-4b` 12 min $0.35; GSM8K $2.13 (4B), $2.37 (8B), $4.24 (14B): `docs/data/jobs.csv`.
35. The A/B mode needs no Dockerfile change: kernels are embedded with `include_str!` (`fused_cuda.rs:61`), and no binary is added.

## 9. Not established

- Any speed or quality at 8B or 14B under the grouped form; the 8B and 14B after the trims at all; quality past 4,096 positions at any size; any task score under the grouped form at any size.
- The host span, the device-only step time and their difference: the host-hidden reading rests on two indirect facts and two assumptions.
- The time of candle's argmax kernel, of the int4 v kernel in situ, and of a fork/join on sm_89; whether small pageable uploads synchronize on this driver.
- The host launch queue depth, so the validity of the spun clocks.
- Whether the key kernel matches candle on the card: designed and simulated, never compiled.
- That events off on two streams is safe in every path of the engine: argued for one buffer pair, to be reviewed.
- The fusion branch's design, so the value of (c) on the graph that will be served.
- Whether the 14B's group of 5 rounds differently in cuBLAS than the 4B's group of 4.
- Every figure above marked *estimated*, and the effort estimates.

## 10. Review of 2026-11-02, and what changed

| finding (severity) | change |
|---|---|
| `gqa-size.sh` could not run and missed gates 2, 3, 6, 7 (high) | rebuilt line for line from both templates: refusal block, `PRE` with every variable, config written and hashed, `DRY_RUN` with `bash -n`, the `run.py` call, every grep kept, `(tokens 3f1baca9033bf251)` included; the 14B launch asks for the 8B's verdict |
| prereg-1 gate 5 compared texts in two formats (high) | gate 5 now hashes the served text extracted as `served-grouped.sh` does, to `0597cee8…`; gate 4 keeps `8784ffc4…` |
| the fallback's "−1.15 % measured" belonged to one stream (high) | rewritten: not measured, above 1.15 %, per-argument waits and records in multi-stream mode; estimated 1.7-3.2 %; `BaseEv` and `VStreamEv` arms added so rule C applies under either ruling |
| "a 5090 at batch 8" (high) | the floor is a batch-1 chat floor for a 5090; the batch of 8 is the extraction floor |
| (c)'s window is what the fusion removes (medium) | (c) scoped to today's graph; re-derived for designs (i) and (ii); measured only on a fused commit named in prereg-1 §3 |
| the D clock could be voided by candle's upload (medium) | the clocks time the device-token step, which holds no pageable copy; P4 probes the copies directly |
| rule B decided on base's exposure (medium) | rule B reads slack = D − H and X_ondev after a3; E_base − E_ondev is reported only |
| "two measured facts say" (medium) | "suggest", *estimated*, with its two assumptions written out |
| arms were not single-mechanism (medium) | five loop kinds, each one mechanism from its neighbour; a1, a2, a3 decided on their own ratios, `OnDevKey/Base` reported only |
| the 4B's task debt ranked first by ETAT §5 (medium) | `prereg-gestures-0.md` added as the first paid step; prereg-2 §7 suspends R1 under its G2 |
| card mutants unplanned; no-join arm missing (medium) | m1-m8 applied in memory, counted, compiled and killed on the card in the A/B; a no-join arm that must diverge |
| the random tie test never reached the cross-thread order; wrong m4 note (medium) | sparse sets with 2-6 maxima, one dense set, 32 sparse cases on the card; m4's note corrected; the signed-zero case swapped so it kills m1 (found while fixing this one) |
| "only Grow truncates by moving a length" (medium) | `Grow` and `Prealloc` both do; `Grow` reserves no window |

Two points where this revision keeps a nuance against the review, with evidence:
- The review gave the 2026-09-01 step "about 1,410 launches". The A2 journal itself says about 750 of candle's and 288 of ours (`a2-verdict-2026-09-01.txt`, "~750 lancements candle … nos 288"); 1,410 is the audit's count for the `Cat` path of 2026-10-29. The finding's point stands either way: that step was longer and ran under other settings.
- The review proposed `LLVQ_KERNEL_DIR` copies for the card mutants. That variable replaces every kernel family at once and the image carries no source, so each of the eight runs would need a full 10-file, 101,541-byte unit shipped to the bucket. The mode applies the mutants to the embedded text in memory instead, counts each replaced string, and compiles each as its own module: the same kill, nothing to ship.
