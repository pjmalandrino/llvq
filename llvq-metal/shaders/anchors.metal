// =============================================================================
// anchors.metal — the two floors of the P1 bench
//
// Appended to `llvq_metal::PAYLOAD_MSL`, which already defines DIM, Cursor,
// ClassRec and decode_payload — the same source `matvec` and `decreal` use.
// This file is NOT self-contained and must not be compiled alone;
// `bin/mslcheck` prepends PAYLOAD_MSL exactly as `bin/rankbench` does.
//
// `floor96` and `decode_f96` are copied unchanged from `decreal.rs:41-78`.
// They are anchors: an anchor that drifted would make the run unreadable, and
// §1.4 of the pre-registration forbids reading a threshold against another
// run's anchors precisely because they are supposed to be the same object.
//
// `floor_rank` is amendment É3(a), added 2026-08-15 before the first
// millisecond: the rank stream's floor. `floor96` reads 12 bytes and is the
// floor of `masques` alone; the rank arms read 8, so without this kernel the
// journal has to carry a prose caveat where a measured arm belongs. It is
// `floor96` with the three 4-byte loads replaced by one 8-byte load and
// nothing else changed, so the gap between the two anchors is traffic and not
// arithmetic.
// =============================================================================
kernel void floor96(device const uint*  words [[buffer(0)]],
                    device const float* x     [[buffer(1)]],
                    device float*       out   [[buffer(2)]],
                    uint gid [[thread_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float xs[DIM];
    if (tid < DIM) xs[tid] = x[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint acc = words[3*gid] ^ words[3*gid + 1] ^ words[3*gid + 2];
    float dot = 0.0f;
    for (uint i = 0; i < DIM; ++i) {
        dot += float(char((acc >> (i & 7)) & 0x7f)) * xs[i];
    }
    out[gid] = dot;
}

// The rank stream's floor: ONE aligned 8-byte load, then the same 24 FMAs.
// Deliberately `floor96` with the loads changed and nothing else, so the gap
// between the two anchors is the traffic and not the arithmetic (É3(a)).
kernel void floor_rank(device const ulong* words [[buffer(0)]],
                       device const float* x     [[buffer(1)]],
                       device float*       out   [[buffer(2)]],
                       uint gid [[thread_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float xs[DIM];
    if (tid < DIM) xs[tid] = x[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    ulong acc = words[gid];
    float dot = 0.0f;
    for (uint i = 0; i < DIM; ++i) {
        dot += float(char((acc >> (i & 7)) & 0x7f)) * xs[i];
    }
    out[gid] = dot;
}

kernel void decode_f96(device const uint*     words  [[buffer(0)]],
                       constant ClassRec*     tab    [[buffer(1)]],
                       constant float*        gscale [[buffer(2)]],
                       device const float*    x      [[buffer(3)]],
                       device float*          out    [[buffer(4)]],
                       uint gid [[thread_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float xs[DIM];
    if (tid < DIM) xs[tid] = x[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint w0 = words[3*gid], w1 = words[3*gid + 1], w2 = words[3*gid + 2];
    Cursor c = { (ulong(w1) << 32) | ulong(w0), w2 };
    out[gid] = decode_payload(c, tab, gscale, xs);
}
