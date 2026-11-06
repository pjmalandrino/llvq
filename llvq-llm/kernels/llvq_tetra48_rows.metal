
// ===========================================================================
// llvq_tetra48_rows.metal: row-loop twins of `tv_tetra48_metal_lut`.
//
// This text is never compiled alone. The host compiles it appended to
// `llvq_tetra48.metal` (`SRC_TETRA_ROWS` in `fused_metal.rs`), as a library
// of its own, so the served pipelines, which compile from the served unit
// alone, keep their bytes and their register allocation. The pragma
// `fp contract(off)` of the served unit covers this text too.
//
// ## What these kernels port
//
// The CUDA 4E row loops (`llvq-cuda/kernels/llvq_pers.cuh`): v4 keeps block
// j + 32's words in flight while block j decodes (`pers_tetra_row`,
// PREFETCH), and v6 issues each thread's loads before their first use. Both
// keep the served lane order, so both keep every bit. Here:
//
//   - `_lutp`: the served loop (activation staged by tiles of
//     LLVQ_TILE_BLOCKS), with the next block's two words loaded before the
//     current block decodes. The tile is a power of two of at least 32, so a
//     lane's blocks j, j + 32, ... run on across tile boundaries in the
//     served order.
//   - `_lutd`: no staging. The activation is read from device memory at the
//     block's own offset, so there is no tile, no barrier and no threadgroup
//     memory. The eight rows of a threadgroup read the same activation, which
//     the caches serve.
//   - `_lutdp`: `_lutd` with the prefetch of `_lutp`.
//   - `_lutd2`, `_lutd4`: `_lutd` with 2 or 4 rows a simdgroup, the lane order
//     of each row unchanged (section at the end of this file).
//
// ## Why the answer does not move
//
// Lane `l` accumulates `acc += tetra48_dot_lut(block j)` for j = l, l + 32,
// ... in ascending order, then the same butterfly and the same epilogue. The
// per-block dot is the served function's text; `_lutd`'s twin differs only in
// the address space of the activation pointer, and the activation values are
// the same floats. Gated, never assumed:
// `llvq-llm/tests/tetra48_metal_rows.rs` demands every f32 equal to the served
// kernel's, fast math on, at the served shapes and past them.
// ===========================================================================

/// `tetra48_dot_lut` reading its 24 activations from device memory.
/// The same text as the served function, line for line, but the pointer's
/// address space.
inline float tetra48_dot_lut_dev(uint lo,
                                 uint hi16,
                                 F1rTables t,
                                 const device float* xb,
                                 const device float* gscale,
                                 const device float* invnorm)
{
    F1rV3Block b = f1r_v3_fetch(lo, hi16, t);
    uint cs[3] = { b.c1, b.c2, b.c3 };
    uint rs[3] = { b.row1, b.row2, b.row3 };
    float acc = 0.0f;
    float n2 = 0.0f;
    for (uint sec = 0u; sec < 3u; ++sec) {
        uint c = cs[sec];
        uint row = rs[sec];
        for (uint j = 0u; j < 8u; ++j) {
            uint o = b.p + 2u * ((c >> j) & 1u);
            uint rho = (row >> (4u * j)) & 15u;
            float v = F1R_VAL[(o << 4) | rho];
            n2 = fma(v, v, n2);
            acc = fma(v, xb[TETRA48_ORDER[sec * 8u + j]], acc);
        }
    }
    uint m = (uint(n2) >> 4) & (TETRA48_SHELLS - 1u);
    uint g = (hi16 >> 15) & 1u;
    return acc * gscale[g] * invnorm[m];
}

/// The served epilogue: the f16 tail, then `acc * rscale + tail`, lane 0.
inline void tetra48_rows_epilogue(float acc,
                                  uint row,
                                  uint lane,
                                  const device float* rscale,
                                  const device half* tail,
                                  const device float* x,
                                  device float* y,
                                  uint nblocks,
                                  uint tail_w)
{
    acc = warp_sum(acc);
    if (lane == 0u) {
        float tv = 0.0f;
        const device float* xt = x + nblocks * 24u;
        for (uint i = 0u; i < tail_w; ++i) {
            tv = fma(float(tail[row * tail_w + i]), xt[i], tv);
        }
        y[row] = fma(acc, rscale[row], tv);
    }
}

kernel void tv_tetra48_metal_lutp(const device uint*   words          [[buffer(0)]],
                                  constant uint&       row_stride_u32 [[buffer(1)]],
                                  const device uint*   rows           [[buffer(2)]],
                                  const device uchar*  prefixes       [[buffer(3)]],
                                  const device ushort* branches       [[buffer(4)]],
                                  const device uchar*  suffixes       [[buffer(5)]],
                                  const device float*  gscale         [[buffer(6)]],
                                  const device float*  invnorm        [[buffer(7)]],
                                  const device float*  rscale         [[buffer(8)]],
                                  const device half*   tail           [[buffer(9)]],
                                  const device float*  x              [[buffer(10)]],
                                  device float*        y              [[buffer(11)]],
                                  constant uint&       nblocks        [[buffer(12)]],
                                  constant uint&       tail_w         [[buffer(13)]],
                                  threadgroup float*   xs             [[threadgroup(0)]],
                                  uint tid  [[thread_position_in_threadgroup]],
                                  uint gid  [[thread_position_in_grid]],
                                  uint tgs  [[threads_per_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]])
{
    uint row = gid >> 5;
    const device uint* wrow = words + row * row_stride_u32;
    F1rTables tab = { rows, prefixes, branches, suffixes };
    float acc = 0.0f;

    // The lane's first block, before the first fill.
    uint lo = 0u, hi16 = 0u;
    if (lane < nblocks) {
        f1r_load(wrow, lane, lo, hi16);
    }
    uint ntiles = (nblocks + LLVQ_TILE_BLOCKS - 1u) / LLVQ_TILE_BLOCKS;
    for (uint t = 0u; t < ntiles; ++t) {
        uint jlo = t * LLVQ_TILE_BLOCKS;
        uint jhi = min(jlo + LLVQ_TILE_BLOCKS, nblocks);
        uint n = (jhi - jlo) * 24u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < n; i += tgs) {
            xs[i] = x[jlo * 24u + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint j = jlo + lane; j < jhi; j += 32u) {
            // Block j + 32's words in flight while block j decodes. It may
            // sit in the next tile: the words are device memory, not staged.
            uint lo2 = 0u, hi2 = 0u;
            if (j + 32u < nblocks) {
                f1r_load(wrow, j + 32u, lo2, hi2);
            }
            acc += tetra48_dot_lut(lo, hi16, tab, xs + (j - jlo) * 24u, gscale, invnorm);
            lo = lo2;
            hi16 = hi2;
        }
    }
    tetra48_rows_epilogue(acc, row, lane, rscale, tail, x, y, nblocks, tail_w);
}

kernel void tv_tetra48_metal_lutd(const device uint*   words          [[buffer(0)]],
                                  constant uint&       row_stride_u32 [[buffer(1)]],
                                  const device uint*   rows           [[buffer(2)]],
                                  const device uchar*  prefixes       [[buffer(3)]],
                                  const device ushort* branches       [[buffer(4)]],
                                  const device uchar*  suffixes       [[buffer(5)]],
                                  const device float*  gscale         [[buffer(6)]],
                                  const device float*  invnorm        [[buffer(7)]],
                                  const device float*  rscale         [[buffer(8)]],
                                  const device half*   tail           [[buffer(9)]],
                                  const device float*  x              [[buffer(10)]],
                                  device float*        y              [[buffer(11)]],
                                  constant uint&       nblocks        [[buffer(12)]],
                                  constant uint&       tail_w         [[buffer(13)]],
                                  uint gid  [[thread_position_in_grid]],
                                  uint lane [[thread_index_in_simdgroup]])
{
    uint row = gid >> 5;
    const device uint* wrow = words + row * row_stride_u32;
    F1rTables tab = { rows, prefixes, branches, suffixes };
    float acc = 0.0f;
    for (uint j = lane; j < nblocks; j += 32u) {
        uint lo, hi16;
        f1r_load(wrow, j, lo, hi16);
        acc += tetra48_dot_lut_dev(lo, hi16, tab, x + j * 24u, gscale, invnorm);
    }
    tetra48_rows_epilogue(acc, row, lane, rscale, tail, x, y, nblocks, tail_w);
}

kernel void tv_tetra48_metal_lutdp(const device uint*   words          [[buffer(0)]],
                                   constant uint&       row_stride_u32 [[buffer(1)]],
                                   const device uint*   rows           [[buffer(2)]],
                                   const device uchar*  prefixes       [[buffer(3)]],
                                   const device ushort* branches       [[buffer(4)]],
                                   const device uchar*  suffixes       [[buffer(5)]],
                                   const device float*  gscale         [[buffer(6)]],
                                   const device float*  invnorm        [[buffer(7)]],
                                   const device float*  rscale         [[buffer(8)]],
                                   const device half*   tail           [[buffer(9)]],
                                   const device float*  x              [[buffer(10)]],
                                   device float*        y              [[buffer(11)]],
                                   constant uint&       nblocks        [[buffer(12)]],
                                   constant uint&       tail_w         [[buffer(13)]],
                                   uint gid  [[thread_position_in_grid]],
                                   uint lane [[thread_index_in_simdgroup]])
{
    uint row = gid >> 5;
    const device uint* wrow = words + row * row_stride_u32;
    F1rTables tab = { rows, prefixes, branches, suffixes };
    float acc = 0.0f;
    uint lo = 0u, hi16 = 0u;
    if (lane < nblocks) {
        f1r_load(wrow, lane, lo, hi16);
    }
    for (uint j = lane; j < nblocks; j += 32u) {
        uint lo2 = 0u, hi2 = 0u;
        if (j + 32u < nblocks) {
            f1r_load(wrow, j + 32u, lo2, hi2);
        }
        acc += tetra48_dot_lut_dev(lo, hi16, tab, x + j * 24u, gscale, invnorm);
        lo = lo2;
        hi16 = hi2;
    }
    tetra48_rows_epilogue(acc, row, lane, rscale, tail, x, y, nblocks, tail_w);
}

// ---------------------------------------------------------------------------
// R rows a simdgroup: `_lutd` with each lane carrying R rows.
//
// The 32-lane kernels give a lane 3.3 blocks at d_in 2,560, each a chain of
// three dependent loads, and the matvec reads 22 % of the bandwidth: the
// header of the served kernel calls that latency. `_ilp` gave a row 8 lanes
// instead of 32 and measured +5 % in situ, but a lane count is part of the
// arithmetic and its tokens broke at 57. These keep 32 lanes a row and the
// served lane order, and give each lane R rows instead: R independent chains
// a thread, and the activation read once for R rows. Row r's accumulator sees
// blocks lane, lane + 32, ... in ascending order, as in `_lut`, so the bits
// are the served kernel's. Gated in `tests/tetra48_metal_rows.rs`.
//
// The host dispatches d_out / R simdgroups and refuses a d_out that is not a
// multiple of 8 R (eight simdgroups a threadgroup, no row guard).
// ---------------------------------------------------------------------------

template <uint R>
inline void tetra48_lutd_rows(const device uint*   words,
                              uint                 row_stride_u32,
                              F1rTables            tab,
                              const device float*  gscale,
                              const device float*  invnorm,
                              const device float*  rscale,
                              const device half*   tail,
                              const device float*  x,
                              device float*        y,
                              uint                 nblocks,
                              uint                 tail_w,
                              uint                 row0,
                              uint                 lane)
{
    float acc[R];
    for (uint r = 0u; r < R; ++r) {
        acc[r] = 0.0f;
    }
    for (uint j = lane; j < nblocks; j += 32u) {
        uint lo[R], hi16[R];
        for (uint r = 0u; r < R; ++r) {
            f1r_load(words + (row0 + r) * row_stride_u32, j, lo[r], hi16[r]);
        }
        for (uint r = 0u; r < R; ++r) {
            acc[r] += tetra48_dot_lut_dev(lo[r], hi16[r], tab, x + j * 24u, gscale, invnorm);
        }
    }
    for (uint r = 0u; r < R; ++r) {
        tetra48_rows_epilogue(acc[r], row0 + r, lane, rscale, tail, x, y, nblocks, tail_w);
    }
}

kernel void tv_tetra48_metal_lutd2(const device uint*   words          [[buffer(0)]],
                                   constant uint&       row_stride_u32 [[buffer(1)]],
                                   const device uint*   rows           [[buffer(2)]],
                                   const device uchar*  prefixes       [[buffer(3)]],
                                   const device ushort* branches       [[buffer(4)]],
                                   const device uchar*  suffixes       [[buffer(5)]],
                                   const device float*  gscale         [[buffer(6)]],
                                   const device float*  invnorm        [[buffer(7)]],
                                   const device float*  rscale         [[buffer(8)]],
                                   const device half*   tail           [[buffer(9)]],
                                   const device float*  x              [[buffer(10)]],
                                   device float*        y              [[buffer(11)]],
                                   constant uint&       nblocks        [[buffer(12)]],
                                   constant uint&       tail_w         [[buffer(13)]],
                                   uint gid  [[thread_position_in_grid]],
                                   uint lane [[thread_index_in_simdgroup]])
{
    F1rTables tab = { rows, prefixes, branches, suffixes };
    tetra48_lutd_rows<2u>(words, row_stride_u32, tab, gscale, invnorm, rscale, tail, x, y, nblocks, tail_w,
                          (gid >> 5) * 2u, lane);
}

kernel void tv_tetra48_metal_lutd4(const device uint*   words          [[buffer(0)]],
                                   constant uint&       row_stride_u32 [[buffer(1)]],
                                   const device uint*   rows           [[buffer(2)]],
                                   const device uchar*  prefixes       [[buffer(3)]],
                                   const device ushort* branches       [[buffer(4)]],
                                   const device uchar*  suffixes       [[buffer(5)]],
                                   const device float*  gscale         [[buffer(6)]],
                                   const device float*  invnorm        [[buffer(7)]],
                                   const device float*  rscale         [[buffer(8)]],
                                   const device half*   tail           [[buffer(9)]],
                                   const device float*  x              [[buffer(10)]],
                                   device float*        y              [[buffer(11)]],
                                   constant uint&       nblocks        [[buffer(12)]],
                                   constant uint&       tail_w         [[buffer(13)]],
                                   uint gid  [[thread_position_in_grid]],
                                   uint lane [[thread_index_in_simdgroup]])
{
    F1rTables tab = { rows, prefixes, branches, suffixes };
    tetra48_lutd_rows<4u>(words, row_stride_u32, tab, gscale, invnorm, rscale, tail, x, y, nblocks, tail_w,
                          (gid >> 5) * 4u, lane);
}

// ===========================================================================
// Half-store twins: `_luth` (the served loop) and `_lutdh` (`_lutd`'s).
//
// The served Tetra matvec stores f32 and the adapter narrows the result with
// a candle cast, one launch a projection: 168 a token on the sealed 4B. These
// narrow in the kernel instead, `half(fma(acc, rscale, tail))`, which is the
// rounding the cast applied (round to nearest even, candle's
// `static_cast<half>`). The loop, the dot and the epilogue's arithmetic are
// the twins' text. Gated by `llvq-llm/tests/tetra48_metal_rows.rs` against
// the served kernel plus the cast, every f16 equal.
// ===========================================================================

/// The served epilogue, narrowed in the kernel.
inline void tetra48_rows_epilogue_h(float acc,
                                    uint row,
                                    uint lane,
                                    const device float* rscale,
                                    const device half* tail,
                                    const device float* x,
                                    device half* y,
                                    uint nblocks,
                                    uint tail_w)
{
    acc = warp_sum(acc);
    if (lane == 0u) {
        float tv = 0.0f;
        const device float* xt = x + nblocks * 24u;
        for (uint i = 0u; i < tail_w; ++i) {
            tv = fma(float(tail[row * tail_w + i]), xt[i], tv);
        }
        y[row] = half(fma(acc, rscale[row], tv));
    }
}

kernel void tv_tetra48_metal_luth(const device uint*   words          [[buffer(0)]],
                                  constant uint&       row_stride_u32 [[buffer(1)]],
                                  const device uint*   rows           [[buffer(2)]],
                                  const device uchar*  prefixes       [[buffer(3)]],
                                  const device ushort* branches       [[buffer(4)]],
                                  const device uchar*  suffixes       [[buffer(5)]],
                                  const device float*  gscale         [[buffer(6)]],
                                  const device float*  invnorm        [[buffer(7)]],
                                  const device float*  rscale         [[buffer(8)]],
                                  const device half*   tail           [[buffer(9)]],
                                  const device float*  x              [[buffer(10)]],
                                  device half*         y              [[buffer(11)]],
                                  constant uint&       nblocks        [[buffer(12)]],
                                  constant uint&       tail_w         [[buffer(13)]],
                                  threadgroup float*   xs             [[threadgroup(0)]],
                                  uint tid  [[thread_position_in_threadgroup]],
                                  uint gid  [[thread_position_in_grid]],
                                  uint tgs  [[threads_per_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]])
{
    uint row = gid >> 5;
    const device uint* wrow = words + row * row_stride_u32;
    F1rTables tab = { rows, prefixes, branches, suffixes };
    float acc = 0.0f;

    uint ntiles = (nblocks + LLVQ_TILE_BLOCKS - 1u) / LLVQ_TILE_BLOCKS;
    for (uint t = 0u; t < ntiles; ++t) {
        uint jlo = t * LLVQ_TILE_BLOCKS;
        uint jhi = min(jlo + LLVQ_TILE_BLOCKS, nblocks);
        uint n = (jhi - jlo) * 24u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < n; i += tgs) {
            xs[i] = x[jlo * 24u + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint j = jlo + lane; j < jhi; j += 32u) {
            uint lo, hi16;
            f1r_load(wrow, j, lo, hi16);
            acc += tetra48_dot_lut(lo, hi16, tab, xs + (j - jlo) * 24u, gscale, invnorm);
        }
    }
    tetra48_rows_epilogue_h(acc, row, lane, rscale, tail, x, y, nblocks, tail_w);
}

kernel void tv_tetra48_metal_lutdh(const device uint*   words          [[buffer(0)]],
                                   constant uint&       row_stride_u32 [[buffer(1)]],
                                   const device uint*   rows           [[buffer(2)]],
                                   const device uchar*  prefixes       [[buffer(3)]],
                                   const device ushort* branches       [[buffer(4)]],
                                   const device uchar*  suffixes       [[buffer(5)]],
                                   const device float*  gscale         [[buffer(6)]],
                                   const device float*  invnorm        [[buffer(7)]],
                                   const device float*  rscale         [[buffer(8)]],
                                   const device half*   tail           [[buffer(9)]],
                                   const device float*  x              [[buffer(10)]],
                                   device half*         y              [[buffer(11)]],
                                   constant uint&       nblocks        [[buffer(12)]],
                                   constant uint&       tail_w         [[buffer(13)]],
                                   uint gid  [[thread_position_in_grid]],
                                   uint lane [[thread_index_in_simdgroup]])
{
    uint row = gid >> 5;
    const device uint* wrow = words + row * row_stride_u32;
    F1rTables tab = { rows, prefixes, branches, suffixes };
    float acc = 0.0f;
    for (uint j = lane; j < nblocks; j += 32u) {
        uint lo, hi16;
        f1r_load(wrow, j, lo, hi16);
        acc += tetra48_dot_lut_dev(lo, hi16, tab, x + j * 24u, gscale, invnorm);
    }
    tetra48_rows_epilogue_h(acc, row, lane, rscale, tail, x, y, nblocks, tail_w);
}

// ---------------------------------------------------------------------------
// The prefill twin: `_lutdh` over up to TETRA48_PREFILL_MAX activation rows a
// launch.
//
// A lane decodes each of its blocks ONCE and applies the 24 decoded values to
// every row of the chunk, so a chunk of 8 prompt positions reads and decodes
// the weights once instead of 8 times. Per row the arithmetic is
// `tetra48_dot_lut_dev`'s, in its order: from zero, acc = fma(v, x[ORDER[k]],
// acc) for k = 0..23, then `acc * gscale[g] * invnorm[m]`, then the lane's
// running sum in ascending block order, the same butterfly and the half
// epilogue. `n2`, and so the shell `m`, depend on the block alone. So row r of
// a chunk is the f16 the decode kernel narrows for that row alone. Gated,
// never assumed: `tests/tetra48_metal_rows.rs`.
//
// The row loop runs to the constant bound with a guard, so the accumulators
// are indexed by constants and stay in registers; `nrows` is uniform across
// the grid, so every lane reaches every butterfly.
// ---------------------------------------------------------------------------
constant uint TETRA48_PREFILL_MAX = 8u;

kernel void tv_tetra48_metal_lutdr(const device uint*   words          [[buffer(0)]],
                                   constant uint&       row_stride_u32 [[buffer(1)]],
                                   const device uint*   rows           [[buffer(2)]],
                                   const device uchar*  prefixes       [[buffer(3)]],
                                   const device ushort* branches       [[buffer(4)]],
                                   const device uchar*  suffixes       [[buffer(5)]],
                                   const device float*  gscale         [[buffer(6)]],
                                   const device float*  invnorm        [[buffer(7)]],
                                   const device float*  rscale         [[buffer(8)]],
                                   const device half*   tail           [[buffer(9)]],
                                   const device float*  x              [[buffer(10)]],
                                   device half*         y              [[buffer(11)]],
                                   constant uint&       nblocks        [[buffer(12)]],
                                   constant uint&       tail_w         [[buffer(13)]],
                                   constant uint&       nrows          [[buffer(15)]],
                                   constant uint&       d_out          [[buffer(16)]],
                                   uint gid  [[thread_position_in_grid]],
                                   uint lane [[thread_index_in_simdgroup]])
{
    uint row = gid >> 5;
    const device uint* wrow = words + row * row_stride_u32;
    F1rTables tab = { rows, prefixes, branches, suffixes };
    uint d_in = nblocks * 24u + tail_w;
    float acc[TETRA48_PREFILL_MAX];
    for (uint r = 0u; r < TETRA48_PREFILL_MAX; ++r) {
        acc[r] = 0.0f;
    }
    for (uint j = lane; j < nblocks; j += 32u) {
        uint lo, hi16;
        f1r_load(wrow, j, lo, hi16);
        F1rV3Block b = f1r_v3_fetch(lo, hi16, tab);
        uint cs[3] = { b.c1, b.c2, b.c3 };
        uint rs[3] = { b.row1, b.row2, b.row3 };
        float v[24];
        float n2 = 0.0f;
        for (uint sec = 0u; sec < 3u; ++sec) {
            uint c = cs[sec];
            uint rw = rs[sec];
            for (uint k = 0u; k < 8u; ++k) {
                uint o = b.p + 2u * ((c >> k) & 1u);
                uint rho = (rw >> (4u * k)) & 15u;
                float vv = F1R_VAL[(o << 4) | rho];
                v[sec * 8u + k] = vv;
                n2 = fma(vv, vv, n2);
            }
        }
        uint m = (uint(n2) >> 4) & (TETRA48_SHELLS - 1u);
        uint g = (hi16 >> 15) & 1u;
        float gs = gscale[g];
        float inv = invnorm[m];
        for (uint r = 0u; r < TETRA48_PREFILL_MAX; ++r) {
            if (r < nrows) {
                const device float* xb = x + r * d_in + j * 24u;
                float a = 0.0f;
                for (uint i = 0u; i < 24u; ++i) {
                    a = fma(v[i], xb[TETRA48_ORDER[i]], a);
                }
                acc[r] += a * gs * inv;
            }
        }
    }
    for (uint r = 0u; r < TETRA48_PREFILL_MAX; ++r) {
        if (r < nrows) {
            tetra48_rows_epilogue_h(acc[r], row, lane, rscale, tail, x + r * d_in, y + r * d_out,
                                    nblocks, tail_w);
        }
    }
}

// ---------------------------------------------------------------------------
// `_lutdt` and its half twin `_lutdth`: `_lutd` with the decoder's tables in
// threadgroup memory, and few threadgroups that loop over the rows.
//
// The served kernels read the four rank tables (18,688 B) and the 64 values of
// `F1R_VAL` from device and constant memory: three dependent gathers a block
// whose index varies by lane, which `constant` serves worst and `device`
// serves through the cache. Here each threadgroup copies them once into
// threadgroup memory, then its simdgroups take rows row = tg * nsg + sg,
// stepping by the grid's simdgroup count, so one copy serves many rows instead
// of the eight a served threadgroup decodes. `tv_tetra48_metal_tg` (the
// faithful decoder) asked the same question without the row loop, so it paid
// one copy for eight rows.
//
// Why the answer does not move: the fetch below is `f1r_v3_fetch`'s text with
// the word `threadgroup`, the dot is `tetra48_dot_lut_dev`'s text reading the
// same 64 floats from threadgroup memory, and a row's lane order, butterfly
// and epilogue are `_lutd`'s. Gated: `tests/tetra48_metal_rows.rs`.
// ---------------------------------------------------------------------------

inline F1rV3Block f1r_v3_fetch_tgt(uint lo, uint hi16, F1rTablesTG t)
{
    uint p  = lo & 1u;
    uint r  = (lo >> 1) & 1u;
    uint s8 = (lo >> 2) & 63u;
    uint b1 = (lo >> 8) & 1u;
    uint i1 = (lo >> 9) & 0x7ffu;
    uint b2 = (lo >> 20) & 15u;
    uint i2 = ((lo >> 24) | ((hi16 & 7u) << 8)) & 0x7ffu;
    uint b3 = (hi16 >> 3) & 1u;
    uint i3 = (hi16 >> 4) & 0x7ffu;

    F1rV3Block b;
    b.p  = p;
    b.c1 = t.prefixes[2u * s8 + b1];
    uint br = t.branches[16u * s8 + b2];
    b.c2 = br & 0xffu;
    uint s16 = (br >> 8) & 63u;
    b.c3 = t.suffixes[2u * s16 + b3];

    b.row1 = t.rows[2048u * r + i1];
    bool mid = i2 < F1R_N0_MIXED;
    uint idx2 = mid ? i2 : (2048u - F1R_N0_MIXED) + i2;
    b.row2 = t.rows[idx2];
    uint delta = mid ? 0u : 1u;
    uint r3 = (p ^ r ^ delta) & 1u;
    b.row3 = t.rows[2048u * r3 + i3];
    return b;
}

inline float tetra48_dot_lut_tgt(uint lo,
                                 uint hi16,
                                 F1rTablesTG t,
                                 const threadgroup float* val,
                                 const device float* xb,
                                 const device float* gscale,
                                 const device float* invnorm)
{
    F1rV3Block b = f1r_v3_fetch_tgt(lo, hi16, t);
    uint cs[3] = { b.c1, b.c2, b.c3 };
    uint rs[3] = { b.row1, b.row2, b.row3 };
    float acc = 0.0f;
    float n2 = 0.0f;
    for (uint sec = 0u; sec < 3u; ++sec) {
        uint c = cs[sec];
        uint row = rs[sec];
        for (uint j = 0u; j < 8u; ++j) {
            uint o = b.p + 2u * ((c >> j) & 1u);
            uint rho = (row >> (4u * j)) & 15u;
            float v = val[(o << 4) | rho];
            n2 = fma(v, v, n2);
            acc = fma(v, xb[TETRA48_ORDER[sec * 8u + j]], acc);
        }
    }
    uint m = (uint(n2) >> 4) & (TETRA48_SHELLS - 1u);
    uint g = (hi16 >> 15) & 1u;
    return acc * gscale[g] * invnorm[m];
}

/// The copy, once a threadgroup, before anything reads a table.
inline F1rTablesTG tetra48_stage_tables(const device uint*   rows,
                                        const device uchar*  prefixes,
                                        const device ushort* branches,
                                        const device uchar*  suffixes,
                                        threadgroup uint*    t_rows,
                                        threadgroup ushort*  t_bran,
                                        threadgroup uchar*   t_pref,
                                        threadgroup uchar*   t_suff,
                                        threadgroup float*   t_val,
                                        uint tid,
                                        uint tgs)
{
    for (uint i = tid; i < 4096u; i += tgs) t_rows[i] = rows[i];
    for (uint i = tid; i < 1024u; i += tgs) t_bran[i] = branches[i];
    for (uint i = tid; i < 128u; i += tgs) { t_pref[i] = prefixes[i]; t_suff[i] = suffixes[i]; }
    for (uint i = tid; i < 64u; i += tgs) t_val[i] = F1R_VAL[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    F1rTablesTG tab = { t_rows, t_pref, t_bran, t_suff };
    return tab;
}

#define TETRA48_LUTDT_ARGS                                                         \
    const device uint*   words          [[buffer(0)]],                             \
    constant uint&       row_stride_u32 [[buffer(1)]],                             \
    const device uint*   rows           [[buffer(2)]],                             \
    const device uchar*  prefixes       [[buffer(3)]],                             \
    const device ushort* branches       [[buffer(4)]],                             \
    const device uchar*  suffixes       [[buffer(5)]],                             \
    const device float*  gscale         [[buffer(6)]],                             \
    const device float*  invnorm        [[buffer(7)]],                             \
    const device float*  rscale         [[buffer(8)]],                             \
    const device half*   tail           [[buffer(9)]],                             \
    const device float*  x              [[buffer(10)]],                            \
    constant uint&       nblocks        [[buffer(12)]],                            \
    constant uint&       tail_w         [[buffer(13)]],                            \
    constant uint&       d_out          [[buffer(16)]],                            \
    threadgroup uint*    t_rows         [[threadgroup(0)]],                        \
    threadgroup ushort*  t_bran         [[threadgroup(1)]],                        \
    threadgroup uchar*   t_pref         [[threadgroup(2)]],                        \
    threadgroup uchar*   t_suff         [[threadgroup(3)]],                        \
    threadgroup float*   t_val          [[threadgroup(4)]],                        \
    uint tid  [[thread_position_in_threadgroup]],                                  \
    uint tgs  [[threads_per_threadgroup]],                                         \
    uint tgid [[threadgroup_position_in_grid]],                                    \
    uint ntg  [[threadgroups_per_grid]],                                           \
    uint sg   [[simdgroup_index_in_threadgroup]],                                  \
    uint nsg  [[simdgroups_per_threadgroup]],                                      \
    uint lane [[thread_index_in_simdgroup]]

kernel void tv_tetra48_metal_lutdt(TETRA48_LUTDT_ARGS, device float* y [[buffer(11)]])
{
    F1rTablesTG tab = tetra48_stage_tables(rows, prefixes, branches, suffixes,
                                           t_rows, t_bran, t_pref, t_suff, t_val, tid, tgs);
    for (uint row = tgid * nsg + sg; row < d_out; row += ntg * nsg) {
        const device uint* wrow = words + row * row_stride_u32;
        float acc = 0.0f;
        for (uint j = lane; j < nblocks; j += 32u) {
            uint lo, hi16;
            f1r_load(wrow, j, lo, hi16);
            acc += tetra48_dot_lut_tgt(lo, hi16, tab, t_val, x + j * 24u, gscale, invnorm);
        }
        tetra48_rows_epilogue(acc, row, lane, rscale, tail, x, y, nblocks, tail_w);
    }
}

kernel void tv_tetra48_metal_lutdth(TETRA48_LUTDT_ARGS, device half* y [[buffer(11)]])
{
    F1rTablesTG tab = tetra48_stage_tables(rows, prefixes, branches, suffixes,
                                           t_rows, t_bran, t_pref, t_suff, t_val, tid, tgs);
    for (uint row = tgid * nsg + sg; row < d_out; row += ntg * nsg) {
        const device uint* wrow = words + row * row_stride_u32;
        float acc = 0.0f;
        for (uint j = lane; j < nblocks; j += 32u) {
            uint lo, hi16;
            f1r_load(wrow, j, lo, hi16);
            acc += tetra48_dot_lut_tgt(lo, hi16, tab, t_val, x + j * 24u, gscale, invnorm);
        }
        tetra48_rows_epilogue_h(acc, row, lane, rscale, tail, x, y, nblocks, tail_w);
    }
}
