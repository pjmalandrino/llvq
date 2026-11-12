// pers_4et_a12g32.cu: arm a12ipg32, design A of
// docs/mesures/large-prefill-rangees-etude-2026-11-11.txt section 3: a12ipg's
// pass (pers_4et_a12g_body, the words read in place, the grouped attention)
// compiled at RC = 4, so one launch carries up to 32 rows where a12ipg's RC = 2
// carries 16. Appended after a12ipg's text, which keeps its bytes.
//
// Nothing else changes. RC sets the n8 tiles of each lane's accumulators in
// p4t_unit_rc and p4t_store_rc and the row guard of the body (R > 8 RC
// aborts). Row n of a projection is column n of the MMA's B and D: no other
// row enters it. The slices are the plan's, never R's, and each tile folds its
// slices in order. The attention's jobs are functions of each row's length,
// and a group job gives a row job's bits (D1). Every row buffer on the card is
// already sized for P4T_RMAX = 32 rows. The area is a12ipg's, which stages 32
// rows already (p4g_area_floats at 8 ceil(R / 8) rows).

#ifndef P4T_A12IPG32
#define P4T_A12IPG32 0
#endif

#if P4T_A12IPG32 || defined(LLVQ_HOST_BUILD)

#define PERS_4ET_A12IPG32(sym, FORM_, THREADS_)                                                         \
    extern "C" __global__ void __launch_bounds__(THREADS_, 1) sym(const Pers4eArgs* __restrict__ A,     \
                                                                   const Pers4tLaunch L)                \
    {                                                                                                   \
        pers_4et_a12g_body<FORM_, 4u, true>(A, L);                                                      \
    }

PERS_4ET_A12IPG32(pers_4et_a12ipg32_f0_t384, 0, 384)
PERS_4ET_A12IPG32(pers_4et_a12ipg32_f1_t384, 1, 384)
PERS_4ET_A12IPG32(pers_4et_a12ipg32_f2_t384, 2, 384)

#endif
