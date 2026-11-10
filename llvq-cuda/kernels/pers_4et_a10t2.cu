// pers_4et_a10t2.cu: arms a10t448, a10t320 and a10t384r16 of the tree
// verify. Appended after a10's files (a10t448, a10t320) or after the row-cap
// file (a10t384r16), which keep their bytes. A module defines P4T_A10T2 as
// 448, 320 or 38416 and gets that arm's symbols only:
//   * pers_4et_a10_f*_t448 and _t320: a10's pass (pers_4et_a10_body) under
//     __launch_bounds__(448 or 320, 1), 14 or 10 warps a CTA, as
//     pers_4et_a10t.cu does at 384 and 256;
//   * pers_4et_a10r16_f*_t384: the 16-row-cap pass of pers_4et_a10rc.cu
//     (pers_4et_rc_body<FORM, 2>) at 384 threads.
// The pass reads its warp count from blockDim; the host gives each arm its
// area and plan at that warp count. Same slices, same fold order: a10's bits.

#ifndef P4T_A10T2
#define P4T_A10T2 0
#endif

#define PERS_4ET_A10T2(sym, FORM_, THREADS_)                                                            \
    extern "C" __global__ void __launch_bounds__(THREADS_, 1) sym(const Pers4eArgs* __restrict__ A,     \
                                                                   const Pers4tLaunch L)                \
    {                                                                                                   \
        pers_4et_a10_body<FORM_>(A, L);                                                                 \
    }

#define PERS_4ET_A10T2RC(sym, FORM_, THREADS_)                                                          \
    extern "C" __global__ void __launch_bounds__(THREADS_, 1) sym(const Pers4eArgs* __restrict__ A,     \
                                                                   const Pers4tLaunch L)                \
    {                                                                                                   \
        pers_4et_rc_body<FORM_, 2u>(A, L);                                                              \
    }

#if P4T_A10T2 == 448 || defined(LLVQ_HOST_BUILD)
PERS_4ET_A10T2(pers_4et_a10_f0_t448, 0, 448)
PERS_4ET_A10T2(pers_4et_a10_f1_t448, 1, 448)
PERS_4ET_A10T2(pers_4et_a10_f2_t448, 2, 448)
#endif

#if P4T_A10T2 == 320 || defined(LLVQ_HOST_BUILD)
PERS_4ET_A10T2(pers_4et_a10_f0_t320, 0, 320)
PERS_4ET_A10T2(pers_4et_a10_f1_t320, 1, 320)
PERS_4ET_A10T2(pers_4et_a10_f2_t320, 2, 320)
#endif

#if P4T_A10T2 == 38416 || defined(LLVQ_HOST_BUILD)
PERS_4ET_A10T2RC(pers_4et_a10r16_f0_t384, 0, 384)
PERS_4ET_A10T2RC(pers_4et_a10r16_f1_t384, 1, 384)
PERS_4ET_A10T2RC(pers_4et_a10r16_f2_t384, 2, 384)
#endif
