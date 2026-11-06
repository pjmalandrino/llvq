// DRAFT, never compiled or run. Wave-5 deep dive "gestures", 2026-11-02, revised after review.
// Not part of the repository; a sketch for the operator to review.
//
// `LLVQ_GESTURE_AB=1` in bin/fusedrun: the cheap gestures, one mechanism per
// arm, one model, every arm in every round, 2 rounds discarded then
// ROUNDS_TIMED, ratios formed round by round (the `run_trim_ab` template,
// fusedrun.rs:187-241). A measurement mode on the bench door: refused beside
// LLVQ_CONFIG (add it to the list at fusedrun.rs:677-679), refused beside any
// other mode, refused with LLVQ_KV_PREALLOC set. The model is put in the served
// 4B state by hand: Grow(256) + Grouped, as `GqaArm::Grouped.apply` does, with
// cudarc's event tracking off before any allocation (fusedrun.rs:702-756).
// Raw ratio = same-head ratio here: every arm reads the same int4 head.
//
// Order inside the mode, each step gating the next:
//   P1  key load check, 41 calls in sequence          (gate)
//   P2  card mutants m1-m8 of tok_argmax.cu            (gate)
//   P3  kernel-time probe                             (reported)
//   P4  pageable-copy probe                           (reported)
//   R   race pre-check, Part C only                   (gate for Part C)
//   A   the timed arms, interleaved rounds            (identity gate, then ratios)
//   K   the clocks E_base, E_ondev, then H and D      (reported; rule B reads them)
//   V   v in situ, Part C only                        (reported)

// ---------------------------------------------------------------- arms ----
//
// Part A always. Part C only when the base commit carries the operator's
// norm+rotation fusion (prereg-gestures-1 §3); otherwise its arms are not
// compiled into the mode (a `cfg` or a const set from the commit), so the
// side stream is never created and the context stays single-stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GestureArm {
    // Part A: the token boundary (decode_on_device.rs, TokenLoop)
    Base,        // `generate` unchanged
    F16,         // a1: no cast                         F16 / Base
    Key,         // a2: key kernel, readback per token  Key / F16
    OnDevKey,    // a3: no readback, no upload          OnDevKey / Key
    OnDevCandle, // a3 with candle's argmax            OnDevCandle / F16
    // Part C: the side stream, tracking off unless named -ev
    VStream,     // c: base loop, v forked              VStream / Base
    BaseEv,      // the fallback's price: base, tracking on (two streams)   BaseEv / Base
    VStreamEv,   // the fork under tracking            VStreamEv / BaseEv
    Bundle,      // OnDevKey + VStream, reported only
}

const PART_A: [GestureArm; 5] = [GestureArm::Base, GestureArm::F16, GestureArm::Key,
                                 GestureArm::OnDevKey, GestureArm::OnDevCandle];
const PART_C: [GestureArm; 4] = [GestureArm::VStream, GestureArm::BaseEv,
                                 GestureArm::VStreamEv, GestureArm::Bundle];

/// The ratios the decision rules read, all formed round by round. Each one
/// moves one mechanism; OnDevKey / Base and VStreamEv / Base are reported.
const RATIOS: [(GestureArm, GestureArm, &str); 7] = [
    (GestureArm::F16, GestureArm::Base, "a1 (cast)"),
    (GestureArm::Key, GestureArm::F16, "a2 (key kernel)"),
    (GestureArm::OnDevKey, GestureArm::Key, "a3 (readback + upload), key"),
    (GestureArm::OnDevCandle, GestureArm::F16, "a3 (readback + upload), candle"),
    (GestureArm::VStream, GestureArm::Base, "c, tracking off"),
    (GestureArm::BaseEv, GestureArm::Base, "tracking on, two streams"),
    (GestureArm::VStreamEv, GestureArm::BaseEv, "c, tracking on"),
];

fn generate_arm(m: &Qwen3, ids: &[u32], n_new: usize, arm: GestureArm, key: &KeyArgmax,
                side: Option<&SideStream>, ctx: &CudaContext) -> Result<Vec<u32>> {
    // Tracking is flipped BEFORE the arm's allocations and back AFTER, as
    // LLVQ_TRIM_AB does (fusedrun.rs:905-925); Part C only.
    let ev = matches!(arm, GestureArm::BaseEv | GestureArm::VStreamEv);
    if ev { unsafe { ctx.enable_event_tracking() } }
    set_vstream(m, side, matches!(arm, GestureArm::VStream | GestureArm::VStreamEv | GestureArm::Bundle));
    let out = match arm {
        GestureArm::Base | GestureArm::VStream | GestureArm::BaseEv | GestureArm::VStreamEv =>
            m.generate(ids, n_new, &mut NoCapture),
        GestureArm::F16 => m.generate_loop(ids, n_new, &TokenLoop::F16),
        GestureArm::Key => m.generate_loop(ids, n_new, &TokenLoop::Key(key)),
        GestureArm::OnDevKey | GestureArm::Bundle => m.generate_loop(ids, n_new, &TokenLoop::OnDevKey(key)),
        GestureArm::OnDevCandle => m.generate_loop(ids, n_new, &TokenLoop::OnDevCandle),
    };
    set_vstream(m, side, false);
    if ev { unsafe { ctx.disable_event_tracking() } }
    out
}

fn run_arms(m: &mut Qwen3, ids: &[u32], n_new: usize, arms: &[GestureArm], key: &KeyArgmax,
            side: Option<&SideStream>, ctx: &CudaContext) -> anyhow::Result<()> {
    const WARMUP: usize = 2;
    let mut reference: Option<Vec<u32>> = None;
    let mut rates = vec![Vec::new(); arms.len()];
    for round in 0..WARMUP + ROUNDS_TIMED {
        for (k, &arm) in arms.iter().enumerate() {
            let t = Instant::now();
            let out = generate_arm(m, ids, n_new, arm, key, side, ctx)?;
            let rate = n_new as f64 / t.elapsed().as_secs_f64();
            match &reference {
                None => reference = Some(out),
                Some(r) => {
                    if let Some(i) = first_divergence(r, &out).map_err(|e| anyhow::anyhow!("{e}"))? {
                        anyhow::bail!("round {round}, arm {arm:?}: divergence at token {i}; \
                                       every gesture claims bit identity, and the run is void");
                    }
                }
            }
            if round >= WARMUP {
                rates[k].push(rate);
            }
        }
    }
    // Table: tok/s [range] per arm; then every RATIOS pair present in `arms`,
    // median [range] round by round; then OnDevKey / Base, labelled
    // "three mechanisms, reported only".
    print_table(arms, &rates, &RATIOS);
    Ok(())
}

// --------------------------------------------------------------- probes ---
//
// P3, kernel-time probe (U4 and a1, about 10 ms of device time). On the real
// f16 logits of the prompt's last position, event-timed, 5 repetitions of 100
// back-to-back launches each, median per launch:
//   (a) candle's fast_argmax_f16 launched DIRECTLY: the function from
//       `dev.get_or_load_func("fast_argmax_f16", &candle_core::cuda_backend::kernels::REDUCE)`
//       (device.rs:227-245; `kernels` re-exports candle_kernels,
//       cuda_backend/mod.rs:6), grid 1, block 1024, args (numel, numel, 1,
//       info, src, out) as cuda_backend/mod.rs:321-355 builds them, with the
//       info buffer uploaded ONCE: the kernel without its per-call upload;
//   (b) the same for fast_argmax_f32 on the f32 cast of the logits;
//   (c) the f16 -> f32 cast of the logits through candle (`to_dtype`), one
//       launch and one 608 KB allocation per call, which is the mechanism a1
//       removes;
//   (d) tok_argmax_h into a scratch slot;
//   (e) candle's full `argmax` call on the f16 logits (upload + alloc + kernel):
//       (e) - (a) is what the per-call pageable upload costs on the device
//       timeline. If that upload synchronizes, the host waits for the previous
//       call each time and the gap shows here.
// Print: "kernel time, us a launch: argmax_f16 X, argmax_f32 X, cast X, key X,
// candle argmax call X".
//
// P4, pageable-copy probe (U5's mechanism, seconds). Five times each, on
// candle's stream, behind `spin_ns(5 ms)`:
//   (i)   host time of `Tensor::from_slice(&[id], (1, 1), dev)`;
//   (ii)  host time of candle's `argmax` on the f16 logits vector (its 16-byte
//         dims/strides upload, cuda_backend/mod.rs:333);
//   (iii) host time of `to_scalar` on a one-element u32 tensor: the control,
//         a pageable D2H that must wait (expected about 5 ms).
// Reading, medians: under 0.5 ms, fire-and-forget; over 4 ms, it synchronizes
// with the stream; between, reported as ambiguous. If (iii) is not over 4 ms
// the probe has no power and its readings are not used.

// --------------------------------------------------------------- clocks ---
//
// The first draft read the host exposure on the BASE step behind two spins.
// Two defects (review of 2026-11-02): candle's argmax inside the timed window
// uploads from pageable memory, which may synchronize and void every step; and
// base's exposure is mostly the per-token boundary, which a3 itself removes, so
// a diet decided on it could prescribe work a3 already made useless. The clocks
// now time the ONDEV step, where nothing in the step waits on the device.
//
// E_base:  unspun base pass, wall time between consecutive readbacks, decode
//          steps only. 5 passes of 64 tokens.
// E_ondev: unspun OnDevKey pass: synchronize after the prefill, then the wall
//          time from the first decode enqueue to the return of the final
//          readback, over the 63 decode steps. 5 passes of 64 tokens.
// H, D:    spun OnDevKey loop, 5 passes of 64 steps, 6 segments a step: a
//          `spin_ns(S)` with S = 5 ms before the step's first launch and, through
//          the capture hook (called after ln1 of every layer, model.rs:1609), at
//          layers 6, 12, 18, 24 and 30. About 155 launches a segment: the host
//          launch queue depth is undocumented, one forum report shows launches
//          blocking past about 1,000, and each launch here also brings an
//          allocation and a free.
//            Segment k: host records t_k right after launching its spin, enqueues
//            the segment, and records t'_k when done; events ev_start_k right
//            after the spin and ev_end_k at the segment's end.
//            VALID only if t'_k - t_k < S - 1 ms for all six segments: then the
//            host finished each segment before the spin in front of it could end,
//            so the device never starved inside the step.
//            H = sum_k (t'_k - t_k); D = sum_k (ev_end_k - ev_start_k).
//          No pageable copy happens inside a step: the token comes from the ring
//          view and tok_argmax_h writes the ring; one synchronize after each step
//          reads its events (outside every window).
// Reported: E_base, E_ondev, H, D (medians over valid steps), the count of valid
// steps, slack = D - H, X_ondev = E_ondev - D, and E_base - E_ondev, "the bubble
// a1 + a2 + a3 recover". Rule B (prereg-gestures-1 §7) reads slack and X_ondev.

struct SpinCapture<'a> {
    stream: &'a CudaStream,
    spin: &'a CudaFunction,
    ns: u64,
    at_layers: [usize; 5],          // 6, 12, 18, 24, 30
    ev_start: Vec<CudaEvent>,       // timing events: ctx.new_event(Some(CU_EVENT_DEFAULT))
    ev_end: Vec<CudaEvent>,
    t_spin: Vec<Instant>,           // t_k
    t_done: Vec<Instant>,           // t'_k
}

impl llvq_llm::model::Capture for SpinCapture<'_> {
    fn on_activation(&mut self, layer: usize, act: Act, _x: &Tensor) -> candle_core::Result<()> {
        if act == Act::Attn {
            if let Some(k) = self.at_layers.iter().position(|&l| l == layer) {
                // close segment k, open segment k + 1
                self.ev_end[k].record(self.stream).map_err(candle_core::Error::msg)?;
                self.t_done[k] = Instant::now();
                launch_spin(self.stream, self.spin, self.ns)?;
                self.t_spin[k + 1] = Instant::now();
                self.ev_start[k + 1].record(self.stream).map_err(candle_core::Error::msg)?;
            }
        }
        Ok(())
    }
}
// The step driver (in fusedrun, from the public hidden_cached and
// logits_last_native, as decode_greedy does, model.rs:2216-2232): launch the
// first spin, t_spin[0], ev_start[0]; hidden_cached(ring view, offset, caches,
// &mut spin_capture); head; tok_argmax_h into ring[pos]; ev_end[5], t_done[5];
// then synchronize and read the twelve events. Tokens of every pass must equal
// the base arm's.

// ------------------------------------------------------------ Part C extras --
//
// R, race pre-check, before the timed rounds (vstream.rs, "THE RACE PRE-CHECK"):
// joined arm with spin_ns(50 us) on the side stream before v, 16 tokens, equal
// to base; nojoin arm, same delay, must diverge within 16 tokens, then the
// context is synchronized. Print one line with both results. Either failure
// voids Part C.
// V, v in situ: one VStream pass of 64 tokens with timing events around the
// forked v on the side stream; median over 36 layers x 64 tokens.
