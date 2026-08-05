// The first thing this port asks of a rented card.
//
// Two kernels, deliberately small. Their job is not speed — nothing here is
// timed for publication — but to answer, in one billed job of a few minutes,
// every question `docs/portage-noyau-cuda.md` §6.2 lists as "answered at the
// first job, for a few cents":
//
//   * does NVRTC exist in the *runtime* image, and does it compile this?
//   * is the PTX actually built for sm_89, or silently for the compute_75
//     default? (`binary_version()`, read host-side)
//   * how many registers does `slot_dot` take, and does it spill?
//   * does the decoder produce, bit for bit, the same lattice point the Rust
//     decoder produces — on a real GPU, at every byte alignment?
//
// The last one is the one that cannot be faked. Everything else is plumbing.

#include "llvq_slot.cuh"

// Decode probe: one thread per block, writes what the decoder *decided*.
//
// Integers, not floats. `slot_dot` returns a float, and a float comparison
// across two compilers can only ever be "close"; the level index and the sign
// bit are exact, and they are where the mask arithmetic lives. Packing them as
// `level | sign << 3` makes the host comparison a plain integer diff over
// 24 slots — a flipped sign or an off-by-one mask shows up immediately and
// unambiguously, which a relative error on a dot product does not.
//
// `cls_out` also captures the class id and the gain bit, so a header misread
// is distinguished from a payload misread.
extern "C" __global__ void decode_probe(const u32* __restrict__ words,
                                        const u32* __restrict__ bases,
                                        const ClassRec* __restrict__ tab,
                                        u32 nblocks,
                                        u32* __restrict__ slots_out,
                                        u32* __restrict__ cls_out)
{
    u32 b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;

    u32 len;
    SlotFields f = slot_fields(words, tab, slot_byte(bases, b), &len);
    for (u32 j = 0; j < LLVQ_DIM; ++j) {
        u32 lev  = slot_level(f, j);
        u32 sign = (f.smask >> j) & 1u;
        slots_out[b * LLVQ_DIM + j] = lev | (sign << 3);
    }
    cls_out[b * 2u + 0u] = f.id;
    cls_out[b * 2u + 1u] = f.gain;
}

// Dot probe: the production path, one thread per block.
//
// This is not the fused matvec — there is no tiling, no warp reduction, no
// shared staging. It exists so that `slot_dot` itself is instantiated and its
// register count and spill size can be read from the loaded module, and so
// that the value the host will later compare against an f64 reference comes
// out of the same code the real kernel will call.
//
// The activation is read straight from global memory. Slower, and irrelevant:
// nothing here is a performance claim.
extern "C" __global__ void dot_probe(const u32* __restrict__ words,
                                     const u32* __restrict__ bases,
                                     const ClassRec* __restrict__ tab,
                                     const float* __restrict__ gscale,
                                     const float* __restrict__ x,
                                     u32 nblocks,
                                     float* __restrict__ out)
{
    u32 b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    out[b] = slot_dot(words, bases, tab, gscale, b, x + b * LLVQ_DIM);
}

// The floor: same five-word read, no decode.
//
// `compute-sanitizer` is absent from the runtime image and `ncu` is very
// likely refused inside a container, so the only instrument left for
// separating "what the reads cost" from "what the decode costs" is a kernel
// that performs the reads and nothing else — the method already used on
// Metal. The XOR is there so the loads cannot be optimised away.
extern "C" __global__ void floor_probe(const u32* __restrict__ words,
                                       const u32* __restrict__ bases,
                                       u32 nblocks,
                                       float* __restrict__ out)
{
    u32 b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    u32 w = slot_byte(bases, b) >> 2;
    u32 acc = words[w] ^ words[w + 1] ^ words[w + 2] ^ words[w + 3] ^ words[w + 4];
    out[b] = (float)(acc & 0xffu);
}
