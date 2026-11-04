// DRAFT, never compiled or run. Written read-only on 2026-11-02 for lead
// "bigmodel" (wave5-deep/bigmodel.md §5, step V0c). Not part of the repository.
//
// The null rotation: the SAME launch as rot_apply and rot_apply_rows (same
// parameter list, same grid, same 1,024-thread block, the same n * 4 B of
// dynamic shared memory requested by gpu.rs:813 and :861, and the same opt-in
// attribute posed through func_dynamic_shared when the model needs one), and
// NO work. Swapped in for rot_apply by the measurement mode LLVQ_ROT_NULL_AB,
// it measures what the rotation's GPU work costs the step in situ: the ceiling
// of any rotation speedup that keeps the launch, R1 included (R1 keeps every
// launch and is estimated to keep about a quarter of the GPU work).
//
// The tokens of the null arm are garbage by design (the matvec after it reads
// whatever the output buffer holds) and are never compared. Its time is still
// the time of the same work everywhere else: `generate` decodes exactly
// max_new tokens with no stop token (llvq-llm/src/model.rs:2119-2150), every
// shape is fixed, and no kernel of the step branches on the data's values in
// a way that changes its cost (the Tetra and int4 matvecs, the norms and the
// attention do the same arithmetic on any finite or non-finite input). That
// last point is an assumption, not a measurement, which is why the null arm
// is read only as a bound and never as a speed of anything served.
//
// A unit of its own, standalone (no header needed: plain unsigned int), loaded
// only under LLVQ_ROT_NULL_AB, so the served unit's text and printed sha256 do
// not move. The empty bodies still declare every parameter so the launch
// builder's argument list (gpu.rs launch_rot / launch_rot_rows) is unchanged.

extern "C" __global__ void rot_null(const unsigned short* __restrict__ xin,
                                    const unsigned int* __restrict__ signbits,
                                    const float* __restrict__ small,
                                    float* __restrict__ xout,
                                    unsigned int n,
                                    unsigned int m,
                                    unsigned int k,
                                    float inv,
                                    unsigned int x_off)
{
}

extern "C" __global__ void rot_null_rows(const unsigned short* __restrict__ xin,
                                         const unsigned int* __restrict__ signbits,
                                         const float* __restrict__ small,
                                         float* __restrict__ xout,
                                         unsigned int n,
                                         unsigned int m,
                                         unsigned int k,
                                         float inv,
                                         unsigned int x_off,
                                         unsigned int row_stride)
{
}
