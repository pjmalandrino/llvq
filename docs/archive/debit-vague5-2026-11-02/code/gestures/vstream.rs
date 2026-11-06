// DRAFT, never compiled or run. Wave-5 deep dive "gestures", 2026-11-02, revised after review.
// Not part of the repository; a sketch for the operator to review.
//
// The int4 v_proj on a side stream.
//
// SCOPE. The window this gesture uses exists in TODAY's graph: in every layer
// of the three sealed files, v_proj is an int4 record read on the UNROTATED ln1
// output, while q and k wait for the shared rotation, a one-CTA kernel (8.05 us
// at n = 2,560, measured, rotation-cuda-2026-08-05). The order on candle's
// single stream is ln1 -> rot -> q -> k -> v (rotplan.rs:519-545, ROT_SHARE=1).
// Forked right after ln1, v's 128 CTAs (fused_cuda.rs:2281-2286, d_out 1,024)
// run on the 141 SMs the rotation leaves idle.
//
// The operator's branch is implementing step 2 of the audit of 2026-10-29
// ("norme + rotation" in one kernel, a one-block kernel on one vector, under
// the tolerance gate T8) and serving tv_tetra48_seg (q+k in one launch). That
// moves the window, and sm_89 has no programmatic dependent launch, so a
// consumer cannot start before its producer kernel ends:
//   design (i)  the fused norm+rot kernel also writes v's unrotated input.
//               The fork moves after it; v overlaps q and k only (or the one
//               q+k segmented launch). Re-derived value, estimated: 0 to +2.4 %
//               at 4B without the segmented kernel, -0.5 to +1.5 % with it.
//   design (ii) v's int4 kernel takes ln1 in its prologue (it reads the
//               residual and the 2,560 ln1 weights; each of its 128 CTAs reduces
//               the row redundantly, about 655 KB of L2 reads). The fork sits
//               after the residual add, and v overlaps the one-block fused
//               kernel, which is at least as long as today's rotation: the
//               window survives, +1.7 to +3.6 % or more, estimated.
// Which design the branch picks decides whether (c) exists at all. For that
// reason the A/B of (c) runs only on a commit that carries the fusion
// (prereg-gestures-1 §3, "Part C"), never on today's graph alone.
//
// Exactness: the same kernel on the same input writes the same output buffer;
// nothing else reads or writes that buffer before the join. No T8 beyond the
// one the fusion itself carries.
//
// Where it would live: fused_cuda.rs (the side stream, the async launch) and
// model.rs::group_forward (the generic fork/join, through an optional Proj hook).

use std::sync::Arc;
use candle_core::cuda_backend::cudarc::driver::{CudaEvent, CudaStream};

/// One side stream per model, created at load AFTER the served door turned
/// cudarc's event tracking off: cudarc's `new_stream` then skips its implicit
/// context synchronize (core.rs:674-679), and every cross-stream access below
/// is ordered by an explicit event instead.
///
/// EVENTS. fused_cuda.rs:2821-2844 turns tracking off only while one stream
/// exists, and says a second stream must revisit that argument. This is the
/// revisit, and it is the operator's to accept. If the operator refuses events
/// off with two streams, the FALLBACK keeps tracking on, and its cost is NOT
/// measured. It is larger than the 1.15 % the events trim bought
/// (trims-contexts-2026-10-30, arm "+events", 1.0115 [1.0080-1.0143]), because
/// that figure was measured on ONE stream, where cudarc only creates, waits on
/// and destroys two events per allocation. Once a second stream exists the
/// context is in multi-stream mode, `is_managing_stream_synchronization()` is
/// true (cudarc 0.19.8 core.rs:466-467), and every launch also pushes a stream
/// wait and an event record per CudaSlice argument (launch.rs:100-125), for
/// candle's ops and ours, plus a wait in every device_ptr (core.rs:1166-1176).
/// The one measurement of that per-launch part is a near-empty kernel chain:
/// 3.67 against 3.63 us a launch (a3-graph-2026-08-06, measured). Scaled to the
/// served step, 929 launches with one to four slice arguments each, the
/// fallback is estimated at 1.7 to 3.2 % in all (computed in
/// calc/gestures-check.py from one measured and one estimated input), the size
/// of (c)'s whole expected gain, plus two driver calls per slice argument on the
/// host. The A/B therefore carries the fallback as arms: `base-ev` and
/// `vstream-ev`, tracking re-enabled before the arm's allocations and disabled
/// after, as LLVQ_TRIM_AB flips it (fusedrun.rs:905-925). Safe here because
/// every cross-stream access is also ordered by the explicit fork and join
/// events; slices created under either setting are freed on main in order.
pub struct SideStream {
    pub stream: Arc<CudaStream>, // cudarc new_stream(): CU_STREAM_NON_BLOCKING, so no
                                 // implicit sync with candle's legacy NULL stream
                                 // (candle device.rs:279 uses default_stream()).
}

/// What a forked launch hands back: the output tensor, valid on the main
/// stream only after `join`, and the input it read, held alive until then.
pub struct Pending {
    pub y: candle_core::Tensor,       // allocated on MAIN before the fork event
    pub keep_x: candle_core::Tensor,  // an Arc clone of v's input: dropping it on the
                                      // host before the join would cuMemFreeAsync it
                                      // on MAIN, and the pool could hand its bytes to
                                      // the next allocation while v still reads them
    pub done: CudaEvent,              // recorded on SIDE after the v kernel
}

// FusedInt4Proj, sketch of the forked forward (decode, one row):
//
//   fn forward_forked(&self, x: &Tensor, side: &SideStream) -> Result<Pending> {
//       let main = self.rt.device.cuda_stream();
//       let mut y = unsafe { main.alloc::<f16>(self.proj.d_out) }?;   // on MAIN
//       let fork = main.record_event(None)?;      // after x was produced and y allocated
//       side.stream.wait(&fork)?;
//       launch_q4_h_on(&side.stream, self.f_int4_half, &self.proj, x_slice, &mut y, THREADS)?;
//       let done = side.stream.record_event(None)?;
//       Ok(Pending { y: wrap(y), keep_x: x.clone(), done })
//   }
//
//   fn join(p: Pending, main: &CudaStream, mode: JoinMode) -> Result<Tensor> {
//       if mode != JoinMode::NoJoinMutant {        // reachable from the A/B pre-check only
//           main.wait(&p.done)?;                   // before any read of y, any free of x
//       }
//       drop(p.keep_x);                            // safe once MAIN is ordered after v
//       Ok(p.y)                                    // freed later on MAIN, as every tensor
//   }
//
// model.rs::group_forward, decode branch (rows == 1), sketch:
//
//   // 1. fork every site with no rotation key that can run forked (int4 v today)
//   let pending: Vec<(usize, Pending)> = projs.iter().enumerate()
//       .filter(|(_, p)| p.rot_key().is_none())
//       .filter_map(|(s, p)| p.forward_forked(x).transpose().map(|r| r.map(|q| (s, q))))
//       .collect::<Result<_>>()?;
//   // 2. the rotated sites on the main stream, exactly as today (drive_rows_batched),
//   //    skipping the forked ones
//   // 3. join each pending before returning, in site order
//
//   `forward_forked` returns None when no side stream is armed (CPU, Metal, the
//   dense path, the bench door without the flag), so the published path is the
//   same code with the hook off. Under design (i) the fork site is after the
//   fused norm+rot launch; under design (ii) it is before it, with x = the
//   residual. The segmented q+k launch, if served, takes the same fork before it.
//
// Where the join must sit: before `qkv[2].reshape` (model.rs:1626). The KV
// append that reads v comes after q_norm, k_norm and RoPE.
//
// THE RACE PRE-CHECK (A/B mode, before any timed round of Part C), on the card:
//   joined: `spin_ns(50_000)` on the SIDE stream before v, 16 tokens; the
//           tokens must equal the base arm's.
//   nojoin: the same delay with `JoinMode::NoJoinMutant`, 16 tokens; the tokens
//           must DIVERGE from base within the 16, or Part C is void: a delay that
//           a missing join survives proves nothing about the join. Main then
//           reads y before v wrote it (stale pool bytes, very likely a previous
//           layer's v), so a divergence at the first token is expected. After
//           this arm the context is synchronized before anything else runs, so
//           no late v write lands in another arm's memory.
// The mutant "drop keep_x before the join" is NOT caught by a delay alone, so
// it is caught by construction: `Pending` owns `keep_x`, and `join` is the only
// place that drops it; review rejects any other drop.
//
// Stream priority: equal to main. Under today's graph and design (ii), the
// overlapped kernel is one CTA, so v needs no priority to start at once. Under
// design (i), v competes with q and k for SMs; a priority arm is not planned.
