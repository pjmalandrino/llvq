// pers_4et_a4.cu: arm a4 of the tree verify, the stream unit (a3) with two
// fixes, both bit for bit with a3.
//
// Appended after pers_4et.cu and pers_4et_stream.cu, which keep their bytes
// (the a0..a3 module texts do not move). A module that defines P4T_A4 1
// (pers.rs, p4t_a4_unit_text) emits the pers_4et_a4_* symbols, on a3's
// tree-tiled copies and a3's plan.
//
//   F1 (P4T_A4_FOLD, default 1): the slice fold. a3's folding warp walks a
//      lane's elements one after the other and loads each element's S
//      partials one by one, an add between two loads: up to (16 R / 32) S
//      dependent round trips a lane. Here a lane takes two elements at once
//      and loads 8 slices of each into registers with independent __ldcg
//      before any add, then adds in slice order 0 .. S - 1 as before (the
//      first partial assigned, not added to zero): the same sums.
//   F2 (P4T_A4_PRO, default 1): the prologues (ln1, ln2, the final norm,
//      the rounding of a row into xin, the down's input). 4E v9's vector path
//      (llvq_pers.cuh section 7b, pers_load_1rt / pers_ln_1rt): binary16
//      rows read and written as 16-byte vectors of 8 halves, a thread's
//      vectors of a batch (PERS_1RT_V) issued before their first use. The
//      element arithmetic is p4t_ln's, p4t_out's and the scalar down input's:
//      candle's scale replayed from the widened row in shared memory (the
//      values __ldcg gave), exact_rms_apply on the same floats,
//      exact_silu_mul on the same halves. Each falls back to the scalar code
//      when a row or a weight vector is not 16-byte aligned or its width not
//      a multiple of 8 (pers_1rt_ok).
//
//   F3 (stamps only): a stamped launch (L.stamps non-null) also times, on
//      thread 0 of each CTA, four parts of every row phase: the setup up to
//      its first unit, the staging (unit start to the barrier after the
//      wait), the decode-and-MMA loops (to the unit's last barrier), the
//      store and fold. Each CTA's sums over the layers go after the phase
//      events, at [1 + 2 n_bar + 4 phase + part][grid] (phases q/k/v, o,
//      gate/up, down, head). An unstamped launch only tests the pointer.
//
// The unit and the pass are pers_4et_stream.cu's with the substitutions
// pers.rs's test the_a4_body_is_the_stream_body pins; the deal is
// p4t_phase_stream's with the F3 clocks.

#ifndef P4T_A4
#define P4T_A4 0
#endif
#ifndef P4T_A4_FOLD
#define P4T_A4_FOLD 1
#endif
#ifndef P4T_A4_PRO
#define P4T_A4_PRO 1
#endif

#if P4T_A4 || defined(LLVQ_HOST_BUILD)

// F3: thread 0's clock at mark k of a unit, a stamped launch only.
#define P4T_A4_MARK(k)                                                          \
    do {                                                                        \
        if (L.stamps != nullptr && threadIdx.x == 0u) tm[k] = pers_now();       \
    } while (0)

// p4t_store, the fold's loads batched (F1).
__device__ __forceinline__ void p4t_store_a4(const P4tSeg& g, const Pers4tLaunch& L, const Pers4eArgs* A, u32 sl,
                                             const float (&acc)[4][4], u32 wrow, u32 lane, unsigned long long* cmax)
{
#if !P4T_A4_FOLD
    p4t_store(g, L, A, sl, acc, wrow, lane, cmax);
#else
    const u32 R = L.n_rows, NT = (R + 7u) >> 3, gid = lane >> 2, tig = lane & 3u;
    if (g.S == 1u) {
        p4t_store(g, L, A, sl, acc, wrow, lane, cmax);
        return;
    }
#pragma unroll
    for (u32 nt = 0; nt < 4u; ++nt)
#pragma unroll
        for (u32 e = 0; e < 4u; ++e) {
            const u32 row = wrow + gid + (e >> 1) * 8u, n = nt * 8u + 2u * tig + (e & 1u);
            if (nt < NT && n < R) g.part[((unsigned long long)sl * R + n) * g.d_out + row] = acc[nt][e];
        }
    __threadfence();  // each lane's partials before the warp's ticket
    __syncwarp();
    u32 last = 0u;
    if (lane == 0u) last = atomicAdd(g.ticket + (wrow >> 4), 1u) == g.S - 1u ? 1u : 0u;
    last = __shfl_sync(0xffffffffu, last, 0);
    if (last == 0u) return;
    __threadfence();
    const u32 NE = 16u * R;
    for (u32 i0 = lane; i0 < NE; i0 += 64u) {
        float v[2] = {0.0f, 0.0f};
        for (u32 k0 = 0; k0 < g.S; k0 += 8u) {
            float pb[2][8];
#pragma unroll
            for (u32 a = 0; a < 2u; ++a) {
                const u32 i = i0 + 32u * a, n = i >> 4, row = wrow + (i & 15u);
#pragma unroll
                for (u32 j = 0; j < 8u; ++j) {
                    const bool ok = i < NE && k0 + j < g.S;
                    pb[a][j] = ok ? __ldcg(g.part + ((unsigned long long)(k0 + j) * R + n) * g.d_out + row) : 0.0f;
                }
            }
#pragma unroll
            for (u32 a = 0; a < 2u; ++a) {
#pragma unroll
                for (u32 jj = 0; jj < 8u; ++jj) {
                    const u32 j = jj;
                    if (k0 + j < g.S) v[a] = k0 + j == 0u ? pb[a][j] : __fadd_rn(v[a], pb[a][j]);
                }
            }
        }
#pragma unroll
        for (u32 a = 0; a < 2u; ++a) {
            const u32 i = i0 + 32u * a;
            if (i < NE) p4t_epi(g, L, A, i >> 4, wrow + (i & 15u), v[a], cmax);
        }
    }
    __syncwarp();
    if (lane == 0u) g.ticket[wrow >> 4] = 0u;
#endif
}

// The binary16 of 8 floats, in memory order, as one 16-byte vector.
__device__ __forceinline__ uint4 p4t_pack8h(const unsigned short h[8])
{
    uint4 w;
    w.x = (u32)h[0] | ((u32)h[1] << 16);
    w.y = (u32)h[2] | ((u32)h[3] << 16);
    w.z = (u32)h[4] | ((u32)h[5] << 16);
    w.w = (u32)h[6] | ((u32)h[7] << 16);
    return w;
}

// The 8 halves of a 16-byte vector, in memory order.
__device__ __forceinline__ void p4t_unpack8h(const uint4 w, unsigned short h[8])
{
    h[0] = (unsigned short)(w.x & 0xffffu);
    h[1] = (unsigned short)(w.x >> 16);
    h[2] = (unsigned short)(w.y & 0xffffu);
    h[3] = (unsigned short)(w.y >> 16);
    h[4] = (unsigned short)(w.z & 0xffffu);
    h[5] = (unsigned short)(w.z >> 16);
    h[6] = (unsigned short)(w.w & 0xffffu);
    h[7] = (unsigned short)(w.w >> 16);
}

// p4t_ln on vectors (F2): the row widened into S by 16-byte loads, candle's
// scale from S, then by the thread that loaded them the rmsnorm of its
// elements, alpha as vectors, nat as vectors. Ends with a barrier.
__device__ __forceinline__ void p4t_ln_a4(const unsigned short* x, const unsigned short* alpha, const u32* signs,
                                          float eps, u32 w, float* S, unsigned short* nat)
{
#if P4T_A4_PRO
    if (pers_1rt_ok(x, w) && pers_1rt_ok(alpha, w) && (nat == nullptr || pers_1rt_ok(nat, w))) {
        const uint4* xv = reinterpret_cast<const uint4*>(x);
        const uint4* av = reinterpret_cast<const uint4*>(alpha);
        const u32 nv = w >> 3, B = blockDim.x;
        for (u32 q0 = 0; q0 < nv; q0 += PERS_1RT_V * B) {
            uint4 xw[PERS_1RT_V];
#pragma unroll
            for (u32 u = 0; u < PERS_1RT_V; ++u) {
                const u32 q = q0 + threadIdx.x + u * B;
                if (q < nv) xw[u] = __ldcg(xv + q);
            }
#pragma unroll
            for (u32 u = 0; u < PERS_1RT_V; ++u) {
                const u32 q = q0 + threadIdx.x + u * B;
                if (q < nv) {
                    float xf[8];
                    pers_unpack8(xw[u], xf);
#pragma unroll
                    for (u32 e = 0; e < 8u; ++e) S[8u * q + e] = xf[e];
                }
            }
        }
        const float sc = exact_rms_scale_by<0>([&](u32 col) { return S[col]; }, w, eps);
        for (u32 q0 = 0; q0 < nv; q0 += PERS_1RT_V * B) {
            uint4 aw[PERS_1RT_V];
#pragma unroll
            for (u32 u = 0; u < PERS_1RT_V; ++u) {
                const u32 q = q0 + threadIdx.x + u * B;
                if (q < nv) aw[u] = av[q];
            }
#pragma unroll
            for (u32 u = 0; u < PERS_1RT_V; ++u) {
                const u32 q = q0 + threadIdx.x + u * B;
                if (q < nv) {
                    float af[8];
                    unsigned short hs[8];
                    pers_unpack8(aw[u], af);
#pragma unroll
                    for (u32 e = 0; e < 8u; ++e) {
                        const u32 i = 8u * q + e;
                        hs[e] = exact_rms_apply_f(sc, S[i], af[e]);
                        const float f = h2f(hs[e]);
                        S[i] = (signs != nullptr && ((signs[i >> 5] >> (i & 31u)) & 1u)) ? -f : f;
                    }
                    if (nat != nullptr) reinterpret_cast<uint4*>(nat)[q] = p4t_pack8h(hs);
                }
            }
        }
        __syncthreads();
        return;
    }
#endif
    p4t_ln(x, alpha, signs, eps, w, S, nat);
}

// p4t_out on vectors (F2): S rounded to binary16, 8 halves a store. Ends
// with a barrier.
__device__ __forceinline__ void p4t_out_a4(const float* S, u32 w, unsigned short* dst)
{
#if P4T_A4_PRO
    if (pers_1rt_ok(dst, w)) {
        uint4* dv = reinterpret_cast<uint4*>(dst);
        for (u32 q = threadIdx.x; q < (w >> 3); q += blockDim.x) {
            unsigned short ho[8];
#pragma unroll
            for (u32 e = 0; e < 8u; ++e) ho[e] = f2h(S[8u * q + e]);
            dv[q] = p4t_pack8h(ho);
        }
        __syncthreads();
        return;
    }
#endif
    p4t_out(S, w, dst);
}

// The down's input of one row (F2): silu(g) u from the gate and up rows as
// 16-byte vectors, a thread's vectors of a batch issued first; signed into S
// for a Tetra down, rounded into dst as vectors for an int4 one. No barrier
// (the caller's follows).
__device__ __forceinline__ void p4t_dn_in_a4(const SegArgs& M, bool tetra, const unsigned short* gr,
                                             const unsigned short* ur, unsigned short* dst, float* S, u32 I)
{
#if P4T_A4_PRO
    if (pers_1rt_ok(gr, I) && pers_1rt_ok(ur, I) && pers_1rt_ok(dst, I)) {
        const uint4* gv = reinterpret_cast<const uint4*>(gr);
        const uint4* uv = reinterpret_cast<const uint4*>(ur);
        const u32 nv = I >> 3, B = blockDim.x;
        for (u32 q0 = 0; q0 < nv; q0 += PERS_1RT_V * B) {
            uint4 gb[PERS_1RT_V], ub[PERS_1RT_V];
#pragma unroll
            for (u32 u = 0; u < PERS_1RT_V; ++u) {
                const u32 q = q0 + threadIdx.x + u * B;
                if (q < nv) {
                    gb[u] = __ldcg(gv + q);
                    ub[u] = __ldcg(uv + q);
                }
            }
#pragma unroll
            for (u32 u = 0; u < PERS_1RT_V; ++u) {
                const u32 q = q0 + threadIdx.x + u * B;
                if (q < nv) {
                    unsigned short gh[8], uh[8], ah[8];
                    p4t_unpack8h(gb[u], gh);
                    p4t_unpack8h(ub[u], uh);
#pragma unroll
                    for (u32 e = 0; e < 8u; ++e) {
                        const u32 i = 8u * q + e;
                        ah[e] = exact_silu_mul(M.silu, gh[e], uh[e]);
                        if (tetra) {
                            const float f = h2f(ah[e]);
                            S[i] = ((M.signs_i[i >> 5] >> (i & 31u)) & 1u) ? -f : f;
                        }
                    }
                    if (!tetra) reinterpret_cast<uint4*>(dst)[q] = p4t_pack8h(ah);
                }
            }
        }
        return;
    }
#endif
    for (u32 i = threadIdx.x; i < I; i += blockDim.x) {
        const unsigned short a = exact_silu_mul(M.silu, __ldcg(gr + i), __ldcg(ur + i));
        if (tetra) {
            const float f = h2f(a);
            S[i] = ((M.signs_i[i >> 5] >> (i & 31u)) & 1u) ? -f : f;
        } else {
            dst[i] = a;
        }
    }
}

// p4t_unit_a4 is defined below (pers_4et_stream.cu's unit with the F3 marks).
__device__ __forceinline__ void p4t_unit_a4(const P4tSeg& g, u32 sl, u32 blk, const Pers4tLaunch& L,
                                            const Pers4eArgs* A, const F1rTables& tab, const float* t_inv,
                                            u32* area, unsigned long long* cmax, unsigned long long* tm);

// p4t_phase_stream's deal on the a4 unit; a stamped launch adds this CTA's
// four parts of the phase to its F3 slots.
__device__ __forceinline__ void p4t_phase_a4(const P4tSeg& s0, const P4tSeg& s1, const P4tSeg& s2, u32 nseg,
                                             const Pers4tLaunch& L, const Pers4eArgs* A, const F1rTables& tab,
                                             const float* t_inv, u32* area, unsigned long long* cmax)
{
    const u32 NW = blockDim.x >> 5;
    const u32 n0 = p4t_nblk(s0, NW) * s0.S;
    const u32 n1 = nseg > 1u ? p4t_nblk(s1, NW) * s1.S : 0u;
    const u32 n2 = nseg > 2u ? p4t_nblk(s2, NW) * s2.S : 0u;
    const bool st = L.stamps != nullptr && threadIdx.x == 0u;
    unsigned long long tm[4] = {0ull, 0ull, 0ull, 0ull}, part[4] = {0ull, 0ull, 0ull, 0ull};
    const unsigned long long t_start = st ? pers_now() : 0ull;
    bool first = true;
    for (u32 u = blockIdx.x; u < n0 + n1 + n2; u += gridDim.x) {
        const u32 i = u < n0 ? 0u : (u < n0 + n1 ? 1u : 2u);
        const u32 lu = i == 0u ? u : (i == 1u ? u - n0 : u - n0 - n1);
        const P4tSeg& g = i == 0u ? s0 : (i == 1u ? s1 : s2);
        p4t_unit_a4(g, lu % g.S, lu / g.S, L, A, tab, t_inv, area, cmax, tm);
        if (st) {
            if (first) part[0] += tm[0] - t_start;
            part[1] += tm[1] - tm[0];
            part[2] += tm[2] - tm[1];
            part[3] += tm[3] - tm[2];
            first = false;
        }
    }
    if (st) {
        u32 n_ot = 0u;
        for (u32 l = 0; l < A->n_layers; ++l) n_ot += A->layers[l].ot != nullptr ? 1u : 0u;
        const u32 ph = nseg == 3u ? 0u
                       : nseg == 2u ? 2u
                       : s0.epi == P4T_HEAD ? 4u
                       : s0.d_in == A->layers[0].mlp.inter ? 3u
                                                            : 1u;
        unsigned long long* base = L.stamps + (size_t)(1u + 2u * n_bar_4et(A->n_layers, n_ot)) * gridDim.x;
        for (u32 p = 0; p < 4u; ++p) base[(size_t)(4u * ph + p) * gridDim.x + blockIdx.x] += part[p];
    }
}

// ---- the units and the pass: pers_4et_stream.cu's, the fold and the prologues above ----

// One unit: tiles [blk nw, blk nw + nw) of g (one a warp; a warp past the
// last tile stages with the others and computes nothing), slice sl, staged
// once and streamed by each warp alone.
__device__ __forceinline__ void p4t_unit_a4(const P4tSeg& g, u32 sl, u32 blk, const Pers4tLaunch& L,
                                                const Pers4eArgs* A, const F1rTables& tab, const float* t_inv,
                                                u32* area, unsigned long long* cmax, unsigned long long* tm)
{
    const u32 t = threadIdx.x, lane = t & 31u, warp = t >> 5, NW = blockDim.x >> 5, gid = lane >> 2, tig = lane & 3u;
    const u32 R = L.n_rows, NT = (R + 7u) >> 3, RS = 8u * NT;
    const u32 nsub = (g.d_in + P4T_SUB - 1u) / P4T_SUB;
    const u32 sub_lo = sl * nsub / g.S, sub_hi = (sl + 1u) * nsub / g.S, ns = sub_hi - sub_lo;
    if (ns > p4t_stream_cap(L.area_floats, NW)) {
        // a plan the area cannot stage: refused, every CTA alike
        if (t == 0u) atomicExch(A->abort_flag, 1u);
        return;
    }
    const u32 XS = 24u * ns + 4u;
    u32* aw = area + warp * 16u * P4T_AS;
    u32* xs = area + NW * 16u * P4T_AS;
    const u32 tile = blk * NW + warp;
    const bool has = tile < (g.d_out >> 4);
    const u32 wrow = tile * 16u;
    float acc[4][4];
#pragma unroll
    for (u32 nt = 0; nt < 4u; ++nt)
#pragma unroll
        for (u32 e = 0; e < 4u; ++e) acc[nt][e] = 0.0f;
    P4T_A4_MARK(0);
    p4t_stream_stage(g, xs, sub_lo, ns, R, RS, XS);
    p4t_cp_wait<0u>();
    __syncthreads();
    P4T_A4_MARK(1);
    if (has && ns > 0u) {
        const u32 SW = g.kind == 0u ? P4T_TSW : P4T_TSQ;
        const u32 r = lane >> 1, h = lane & 1u;
        const u32* wp = g.w + ((unsigned long long)tile * nsub + sub_lo) * SW + (g.kind == 0u ? 3u * r + h : 6u * r + 3u * h);
        u32 c0, c1, c2;
        p4t_stream_load(g.kind, wp, c0, c1, c2);
        for (u32 s = 0; s < ns; ++s) {
            u32 n0 = 0u, n1 = 0u, n2 = 0u;
            if (s + 1u < ns) p4t_stream_load(g.kind, wp + SW, n0, n1, n2);
            p4t_decode_stream(g, tab, t_inv, c0, c1, c2, aw, sub_lo + s, wrow, lane);
            __syncwarp();
#pragma unroll
            for (u32 kk = 0; kk < 3u; ++kk) {
                const u32* a0p = aw + gid * P4T_AS + kk * 8u + tig;
                const u32 af[4] = {a0p[0], a0p[8u * P4T_AS], a0p[4], a0p[8u * P4T_AS + 4u]};
                const u32 kw = 24u * s + 8u * kk + tig;
#pragma unroll
                for (u32 nt = 0; nt < 4u; ++nt) {
                    if (nt < NT) {
                        const u32* xr = xs + (nt * 8u + gid) * XS + kw;
                        p4t_mma(acc[nt], af, xr[0], xr[4]);
                    }
                }
            }
            __syncwarp();
            wp += SW;
            c0 = n0;
            c1 = n1;
            c2 = n2;
        }
    }
    __syncthreads();
    P4T_A4_MARK(2);
    if (has) p4t_store_a4(g, L, A, sl, acc, wrow, lane, cmax);
    P4T_A4_MARK(3);
}

template <int FORM>
__device__ __forceinline__ void pers_4et_a4_body(const Pers4eArgs* __restrict__ A, const Pers4tLaunch L)
{
    unsigned char* base = PERS_SMEM_BASE();
    u32* t_rows = reinterpret_cast<u32*>(base + PERS_SMEM_HEADER);
    unsigned char* t_pref = reinterpret_cast<unsigned char*>(t_rows + 4096);
    unsigned short* t_br = reinterpret_cast<unsigned short*>(t_pref + 128);
    unsigned char* t_suf = reinterpret_cast<unsigned char*>(t_br + 1024);
    float* t_inv = reinterpret_cast<float*>(t_suf + 128);
    float* S = t_inv + 32;  // the area, L.area_floats floats
    u32* area = reinterpret_cast<u32*>(S);
    unsigned long long* cmax = reinterpret_cast<unsigned long long*>(S + (L.area_floats - 2u * P4T_RMAX));
    const u32 H = A->hidden, G = gridDim.x, c0 = blockIdx.x, R = L.n_rows, NL = A->n_layers;
    const u32 lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const u32 dq = A->n_heads * 128u, dkv = A->n_kv * 128u;
    // Refused before any wait, by every CTA alike: a row count out of range,
    // an area smaller than this launch needs.
    {
        const u32 I = A->layers[0].mlp.inter;
        u32 widest = H > I ? H : I;
        if (dq > widest) widest = dq;
        if (R == 0u || R > P4T_RMAX || (L.area_floats & 1u) != 0u ||
            L.area_floats < p4t_area_floats(nw, 8u * ((R + 7u) >> 3), widest, A->groups)) {
            if (c0 == 0u && threadIdx.x == 0u) atomicExch(A->abort_flag, 1u);
            return;
        }
    }
    for (u32 i = threadIdx.x; i < 4096u; i += blockDim.x) t_rows[i] = A->rows[i];
    for (u32 i = threadIdx.x; i < 128u; i += blockDim.x) {
        t_pref[i] = A->pref[i];
        t_suf[i] = A->suf[i];
    }
    for (u32 i = threadIdx.x; i < 1024u; i += blockDim.x) t_br[i] = A->br[i];
    for (u32 i = threadIdx.x; i < 32u; i += blockDim.x) t_inv[i] = A->invnorm[i];
    __syncthreads();
    const F1rTables tab{t_rows, t_pref, t_br, t_suf};
    u32 nb = 0u;
#define P4T_WAIT()                                                                                   \
    do {                                                                                             \
        if (L.stamps != nullptr) {                                                                   \
            __syncthreads();                                                                         \
            if (threadIdx.x == 0u) L.stamps[(size_t)(2u * nb + 1u) * G + c0] = pers_now();           \
        }                                                                                            \
        if (!pers_grid_wait(A->bar, L.bar_base + (++nb) * G, A->abort_flag)) return;                 \
        if (L.stamps != nullptr && threadIdx.x == 0u) L.stamps[(size_t)(2u * nb) * G + c0] = pers_now(); \
    } while (0)
    // Phase stamps (a measurement launch, L.stamps non-null): event 0 at the
    // CTA's start; for its e-th wait, event 2e - 1 once the whole CTA reached
    // it (its work of the phase done) and event 2e on leaving it, thread 0 of
    // each CTA. Nothing a row computes reads the clock.
    if (L.stamps != nullptr && threadIdx.x == 0u) L.stamps[c0] = pers_now();
    // The argmax keys and every ticket start at zero; the first wait's fence
    // orders these stores before any CTA's atomics.
    if (c0 == 0u) {
        for (u32 i = threadIdx.x; i < P4T_RMAX; i += blockDim.x) L.slots[i] = 0ull;
        for (u32 i = threadIdx.x; i < P4T_RMAX * A->n_kv; i += blockDim.x) L.tickets[i] = 0u;
    }
    // The previous launch's accepted rows into the cache, every layer.
    for (u32 it = c0; it < L.n_move * NL; it += G) {
        const u32 i = it / NL, l = it - i * NL, pos = L.move_pos0 + i;
        const Pers4eLayer& Y = A->layers[l];
        const unsigned short* sk = L.tk + ((unsigned long long)l * P4T_RMAX + L.mv[i]) * dkv;
        const unsigned short* sv = L.tv + ((unsigned long long)l * P4T_RMAX + L.mv[i]) * dkv;
        for (u32 e = threadIdx.x; e < dkv; e += blockDim.x) {
            const unsigned long long at = ((unsigned long long)(e >> 7) * A->cap + pos) * 128u + (e & 127u);
            Y.kc[at] = __ldcg(sk + e);
            Y.vc[at] = __ldcg(sv + e);
        }
    }
    for (u32 r = 0; r < R; ++r)
        pers_embed(A->emb_w, A->emb_s, A->emb_b, L.toks[r], H, A->emb_gpr, L.x_a + (size_t)r * H, H * c0 / G,
                   H * (c0 + 1u) / G);
    P4T_WAIT();
    for (u32 l = 0; l < NL; ++l) {
        const Pers4eLayer& Y = A->layers[l];
        const Pers4eQkv& N = Y.qkv;
        const SegArgs& M = Y.mlp;
        const u32 I = M.inter;
        unsigned short* tk_l = L.tk + (unsigned long long)l * P4T_RMAX * dkv;
        unsigned short* tv_l = L.tv + (unsigned long long)l * P4T_RMAX * dkv;
        // ln1: the natural values for the int4 v rows, the signed and
        // rotated ones for the Tetra q and k rows
        for (u32 r = c0; r < R; r += G) {
            p4t_ln_a4(L.x_a + (size_t)r * H, Y.ln1, Y.signs_a, A->eps, H, S, L.xn + (size_t)r * H);
            pers_rot_finish(S, Y.small_a, H, Y.m_a, Y.k_a, Y.inv_a);
            p4t_out_a4(S, H, L.xin + (size_t)r * L.ldx);
        }
        P4T_WAIT();
        {
            P4tSeg sq = p4t_seg_tetra(N.qw, N.qgs, N.qrs, N.qtl, N.q_stride, N.nblocks, N.tail_w, N.q_dout);
            P4tSeg sk = p4t_seg_tetra(N.kw, N.kgs, N.krs, N.ktl, N.k_stride, N.nblocks, N.tail_w, N.k_dout);
            P4tSeg sv = p4t_seg_q4(N.vw, N.vsc, N.vbi, N.d_in, N.v_dout, 7u, 1u);
            p4t_io(sq, L.xin, L.ldx, L.q_raw, dq, nullptr, P4T_STORE, L.split_qkv);
            p4t_io(sk, L.xin, L.ldx, L.k_raw, dkv, nullptr, P4T_STORE, L.split_qkv);
            p4t_io(sv, L.xn, H, L.v_raw, dkv, nullptr, P4T_STORE, L.split_qkv);
            unsigned long long po = 0ull;
            u32 to = 0u;
            p4t_place(sq, L, po, to);
            p4t_place(sk, L, po, to);
            p4t_place(sv, L, po, to);
            p4t_phase_a4(sq, sk, sv, 3u, L, A, tab, t_inv, area, cmax);
        }
        P4T_WAIT();
        // q/k-norm and RoPE at the row's position; K and V into its slot
        {
            const u32 per = A->n_heads + 2u * A->n_kv;
            for (u32 w = c0 + G * warp; w < R * per; w += G * nw) {
                const u32 r = w / per, hh = w - r * per, pos = L.pos0 + L.depth[r];
                const unsigned short* cr = A->cosr + pos * 64u;
                const unsigned short* sr = A->sinr + pos * 64u;
                if (hh < A->n_heads) {
                    pers_head_norm_rope<FORM>(L.q_raw + (size_t)r * dq + hh * 128u, Y.q_norm, cr, sr, A->eps,
                                              L.q_out + (size_t)r * dq + hh * 128u, lane);
                } else if (hh < A->n_heads + A->n_kv) {
                    const u32 h = hh - A->n_heads;
                    pers_head_norm_rope<FORM>(L.k_raw + (size_t)r * dkv + h * 128u, Y.k_norm, cr, sr, A->eps,
                                              tk_l + (size_t)r * dkv + h * 128u, lane);
                } else {
                    const u32 h = hh - A->n_heads - A->n_kv;
                    for (u32 i = lane; i < 128u; i += 32u)
                        tv_l[(size_t)r * dkv + h * 128u + i] = __ldcg(L.v_raw + (size_t)r * dkv + h * 128u + i);
                }
            }
        }
        P4T_WAIT();
        p4t_attention(A, L, Y.kc, Y.vc, tk_l, tv_l, reinterpret_cast<unsigned char*>(S));
        P4T_WAIT();
        // o and the residual: x_b = x_a + o(ctx)
        {
            P4tSeg so;
            if (Y.ot != nullptr) {
                const Pers4eOT& O = *Y.ot;
                for (u32 r = c0; r < R; r += G) {
                    __syncthreads();
                    pers_rot_load(L.ctx + (size_t)r * dq, O.signs, S, dq);
                    pers_rot_finish(S, O.small, dq, O.m, O.k, O.inv);
                    p4t_out_a4(S, dq, L.xin + (size_t)r * L.ldx);
                }
                P4T_WAIT();
                so = p4t_seg_tetra(O.t.words, O.t.gscale, O.t.rscale, O.t.tail, O.t.stride_u32, O.t.nblocks,
                                   O.t.tail_w, O.t.d_out);
                p4t_io(so, L.xin, L.ldx, L.x_b, H, L.x_a, P4T_RESID, L.split_o);
            } else {
                so = p4t_seg_q4(Y.ow, Y.osc, Y.obi, dq, H, 7u, 1u);
                p4t_io(so, L.ctx, dq, L.x_b, H, L.x_a, P4T_RESID, L.split_o);
            }
            unsigned long long po = 0ull;
            u32 to = 0u;
            p4t_place(so, L, po, to);
            p4t_phase_a4(so, so, so, 1u, L, A, tab, t_inv, area, cmax);
        }
        P4T_WAIT();
        // ln2, signed and rotated: the gate and up rows' input
        for (u32 r = c0; r < R; r += G) {
            p4t_ln_a4(L.x_b + (size_t)r * H, M.alpha, M.signs_h, M.eps, H, S, nullptr);
            pers_rot_finish(S, M.small_h, H, M.m_h, M.k_h, M.inv_h);
            p4t_out_a4(S, H, L.xin + (size_t)r * L.ldx);
        }
        P4T_WAIT();
        {
            P4tSeg sg = p4t_seg_tetra(M.gate.words, M.gate.gscale, M.gate.rscale, M.gate.tail, M.gate.stride_u32,
                                      M.gate.nblocks, M.gate.tail_w, M.gate.d_out);
            P4tSeg su = p4t_seg_tetra(M.up.words, M.up.gscale, M.up.rscale, M.up.tail, M.up.stride_u32, M.up.nblocks,
                                      M.up.tail_w, M.up.d_out);
            p4t_io(sg, L.xin, L.ldx, L.g_raw, I, nullptr, P4T_STORE, L.split_gu);
            p4t_io(su, L.xin, L.ldx, L.u_raw, I, nullptr, P4T_STORE, L.split_gu);
            unsigned long long po = 0ull;
            u32 to = 0u;
            p4t_place(sg, L, po, to);
            p4t_place(su, L, po, to);
            p4t_phase_a4(sg, su, su, 2u, L, A, tab, t_inv, area, cmax);
        }
        P4T_WAIT();
        // the down's input: silu(g) u, signed and rotated for a Tetra down,
        // natural for an int4 one
        for (u32 r = c0; r < R; r += G) {
            const unsigned short* gr = L.g_raw + (size_t)r * I;
            const unsigned short* ur = L.u_raw + (size_t)r * I;
            unsigned short* dst = L.xin + (size_t)r * L.ldx;
            p4t_dn_in_a4(M, Y.tetra_down, gr, ur, dst, S, I);
            if (Y.tetra_down) {
                pers_rot_finish(S, M.small_i, I, M.m_i, M.k_i, M.inv_i);
                p4t_out_a4(S, I, dst);
            } else {
                __syncthreads();
            }
        }
        P4T_WAIT();
        // the down and the residual: x_a = x_b + down
        {
            P4tSeg sd = Y.tetra_down
                            ? p4t_seg_tetra(M.down.words, M.down.gscale, M.down.rscale, M.down.tail,
                                            M.down.stride_u32, M.down.nblocks, M.down.tail_w, M.down.d_out)
                            : p4t_seg_q4(M.q4w, M.q4s, M.q4b, I, H, 7u, 1u);
            p4t_io(sd, L.xin, L.ldx, L.x_a, H, L.x_b, P4T_RESID, L.split_dn);
            unsigned long long po = 0ull;
            u32 to = 0u;
            p4t_place(sd, L, po, to);
            p4t_phase_a4(sd, sd, sd, 1u, L, A, tab, t_inv, area, cmax);
        }
        P4T_WAIT();
    }
    // the final norm, natural: the head's input
    for (u32 r = c0; r < R; r += G)
        p4t_ln_a4(L.x_a + (size_t)r * H, A->final_norm, nullptr, A->eps, H, S, L.xin + (size_t)r * L.ldx);
    P4T_WAIT();
    // the head: each row's argmax key through the CTA's maxima
    {
        for (u32 i = threadIdx.x; i < P4T_RMAX; i += blockDim.x) cmax[i] = 0ull;
        __syncthreads();
        P4tSeg sh = p4t_seg_q4(A->head_w, A->head_s, A->head_b, H, A->vocab, 6u, 0u);
        p4t_io(sh, L.xin, L.ldx, nullptr, 0u, nullptr, P4T_HEAD, L.split_head);
        unsigned long long po = 0ull;
        u32 to = 0u;
        p4t_place(sh, L, po, to);
        p4t_phase_a4(sh, sh, sh, 1u, L, A, tab, t_inv, area, cmax);
        __syncthreads();
        if (threadIdx.x < R && cmax[threadIdx.x] != 0ull) atomicMax(L.slots + threadIdx.x, cmax[threadIdx.x]);
    }
    P4T_WAIT();
    if (c0 == 0u && threadIdx.x < R) {
        if (pers_ld_acquire(A->nan_flag) != 0u) return;  // the host reads the flag
        const unsigned long long k = __ldcg(L.slots + threadIdx.x);
        const u32 br = 1023u - (u32)((k >> 22) & 0x3FFu);
        L.ring[threadIdx.x] = ((0x3FFFFFu - (u32)(k & 0x3FFFFFu)) << 10) | (__brev(br) >> 22);
    }
#undef P4T_WAIT
}

#define PERS_4ET_A4(sym, FORM_, THREADS_)                                                               \
    extern "C" __global__ void __launch_bounds__(THREADS_, 1) sym(const Pers4eArgs* __restrict__ A,     \
                                                                   const Pers4tLaunch L)                \
    {                                                                                                   \
        pers_4et_a4_body<FORM_>(A, L);                                                                  \
    }

PERS_4ET_A4(pers_4et_a4_f0_t512, 0, 512)
PERS_4ET_A4(pers_4et_a4_f1_t512, 1, 512)
PERS_4ET_A4(pers_4et_a4_f2_t512, 2, 512)

#endif
