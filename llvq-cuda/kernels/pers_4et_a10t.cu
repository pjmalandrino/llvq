// pers_4et_a10t.cu: arms a10t384 and a10t256 of the tree verify, a10 at 12
// and 8 warps a CTA. Appended after a10's files, which keep their bytes. A
// module with P4T_A10T 384 or 256 emits pers_4et_a10_f*_t384 or _t256: a10's
// pass (pers_4et_a10_body, unchanged) under __launch_bounds__(THREADS, 1),
// so one CTA a SM gets up to 170 or 255 registers a thread. The pass reads
// its warp count from blockDim: a unit is 12 or 8 tiles, the area, the slice
// cap and the deal follow it, and the host gives the launch the area and the
// plan computed at that warp count. The tree-tiled copies are the same (one
// 16-row tile a warp). Same slices, same fold order: a10's bits.

#ifndef P4T_A10T
#define P4T_A10T 0
#endif

#define PERS_4ET_A10T(sym, FORM_, THREADS_)                                                             \
    extern "C" __global__ void __launch_bounds__(THREADS_, 1) sym(const Pers4eArgs* __restrict__ A,     \
                                                                   const Pers4tLaunch L)                \
    {                                                                                                   \
        pers_4et_a10_body<FORM_>(A, L);                                                                 \
    }

#if P4T_A10T == 384 || defined(LLVQ_HOST_BUILD)
PERS_4ET_A10T(pers_4et_a10_f0_t384, 0, 384)
PERS_4ET_A10T(pers_4et_a10_f1_t384, 1, 384)
PERS_4ET_A10T(pers_4et_a10_f2_t384, 2, 384)
#endif

#if P4T_A10T == 256 || defined(LLVQ_HOST_BUILD)
PERS_4ET_A10T(pers_4et_a10_f0_t256, 0, 256)
PERS_4ET_A10T(pers_4et_a10_f1_t256, 1, 256)
PERS_4ET_A10T(pers_4et_a10_f2_t256, 2, 256)
#endif
