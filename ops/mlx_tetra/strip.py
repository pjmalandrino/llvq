# /// script
# requires-python = ">=3.10"
# dependencies = ["mlx-lm==0.28.3", "mlx==0.32.4", "numpy==2.5.4", "safetensors==0.8.0", "tokenizers==0.23.3",
#                 "pyarrow==26.0.1", "transformers==5.22.1", "llvq-tetra"]
# [tool.uv.sources]
# llvq-tetra = { path = "../../llvq-tetra" }
# ///
"""Strip the Tetra decode kernel stage by stage on one matrix, off protocol.

Variants (only `full` computes the right numbers; the others time what is left):
  full    the served lutd loop
  nolut   F1R_VAL lookups replaced by arithmetic on the same bits
  nofetch the rank-table fetch skipped too (the words stand in for the rows)
  stream  load the words and fma them into the activation: the memory floor
Chains of N calls, each fed from the last, median of rounds.
"""
import statistics, sys, time
import mlx.core as mx
import llvq_mlx as L

HEAD = """
    uint row = thread_position_in_grid.x >> 5;
    uint lane = thread_index_in_simdgroup;
    const device uint* wrow = words + row * STRIDE;
    F1rTables tab = { rows, prefixes, branches, suffixes };
    float acc = 0.0f;
    for (uint j = lane; j < uint(NB); j += 32u) {
        uint lo, hi16;
        f1r_load(wrow, j, lo, hi16);
        const device float4* x4 = (const device float4*)(x + j * 24u);
        float xv[24];
        for (uint q = 0u; q < 6u; ++q) { float4 t4 = x4[q]; xv[4u*q]=t4.x; xv[4u*q+1u]=t4.y; xv[4u*q+2u]=t4.z; xv[4u*q+3u]=t4.w; }
"""
TAIL = """
    }
    tetra48_rows_epilogue_h(acc, row, lane, rscale, tail, x, y, uint(NB), uint(TW));
"""
DEC = {
"full": """
        F1rV3Block b = f1r_v3_fetch(lo, hi16, tab);
        uint cs[3] = { b.c1, b.c2, b.c3 }; uint rs[3] = { b.row1, b.row2, b.row3 };
        float a = 0.0f, n2 = 0.0f;
        for (uint sec = 0u; sec < 3u; ++sec) for (uint k = 0u; k < 8u; ++k) {
            uint o = b.p + 2u * ((cs[sec] >> k) & 1u); uint rho = (rs[sec] >> (4u * k)) & 15u;
            float v = F1R_VAL[(o << 4) | rho]; n2 = fma(v, v, n2); a = fma(v, xv[sec * 8u + k], a); }
        uint m = (uint(n2) >> 4) & (TETRA48_SHELLS - 1u); uint g = (hi16 >> 15) & 1u;
        acc += a * gscale[g] * invnorm[m];
""",
"nolut": """
        F1rV3Block b = f1r_v3_fetch(lo, hi16, tab);
        uint cs[3] = { b.c1, b.c2, b.c3 }; uint rs[3] = { b.row1, b.row2, b.row3 };
        float a = 0.0f, n2 = 0.0f;
        for (uint sec = 0u; sec < 3u; ++sec) for (uint k = 0u; k < 8u; ++k) {
            uint o = b.p + 2u * ((cs[sec] >> k) & 1u); uint rho = (rs[sec] >> (4u * k)) & 15u;
            float v = float(int(rho) - 8) * 2.0f + float(o); n2 = fma(v, v, n2); a = fma(v, xv[sec * 8u + k], a); }
        uint m = (uint(n2) >> 4) & (TETRA48_SHELLS - 1u); uint g = (hi16 >> 15) & 1u;
        acc += a * gscale[g] * invnorm[m];
""",
"nofetch": """
        uint cs[3] = { lo & 255u, (lo >> 8) & 255u, (lo >> 16) & 255u }; uint rs[3] = { lo, hi16 | (lo << 16), lo ^ hi16 };
        uint p = lo & 1u;
        float a = 0.0f, n2 = 0.0f;
        for (uint sec = 0u; sec < 3u; ++sec) for (uint k = 0u; k < 8u; ++k) {
            uint o = p + 2u * ((cs[sec] >> k) & 1u); uint rho = (rs[sec] >> (4u * k)) & 15u;
            float v = float(int(rho) - 8) * 2.0f + float(o); n2 = fma(v, v, n2); a = fma(v, xv[sec * 8u + k], a); }
        uint m = (uint(n2) >> 4) & (TETRA48_SHELLS - 1u); uint g = (hi16 >> 15) & 1u;
        acc += a * gscale[g] * invnorm[m];
""",
"stream": """
        float a = 0.0f;
        for (uint k = 0u; k < 24u; ++k) a = fma(float((k < 16u ? lo >> (2u*k) : hi16 >> (k-16u)) & 3u), xv[k], a);
        acc += a;
""",
}

# R rows a simdgroup: a lane loads its activation block once and applies it to
# the R rows' blocks, so the activation traffic drops by R.
HEAD_R = """
    uint row0 = (thread_position_in_grid.x >> 5) * RR;
    uint lane = thread_index_in_simdgroup;
    F1rTables tab = { rows, prefixes, branches, suffixes };
    float acc[RR];
    for (uint r = 0u; r < RR; ++r) acc[r] = 0.0f;
    for (uint j = lane; j < uint(NB); j += 32u) {
        const device float4* x4 = (const device float4*)(x + j * 24u);
        float xv[24];
        for (uint q = 0u; q < 6u; ++q) { float4 t4 = x4[q]; xv[4u*q]=t4.x; xv[4u*q+1u]=t4.y; xv[4u*q+2u]=t4.z; xv[4u*q+3u]=t4.w; }
        for (uint r = 0u; r < RR; ++r) {
            uint lo, hi16;
            f1r_load(words + (row0 + r) * STRIDE, j, lo, hi16);
"""
TAIL_R = """
            acc[r] += a_r;
        }
    }
    for (uint r = 0u; r < RR; ++r)
        tetra48_rows_epilogue_h(acc[r], row0 + r, lane, rscale, tail, x, y, uint(NB), uint(TW));
"""
DEC_R = {
"stream": """
            float a_r = 0.0f;
            for (uint k = 0u; k < 24u; ++k) a_r = fma(float((k < 16u ? lo >> (2u*k) : hi16 >> (k-16u)) & 3u), xv[k], a_r);
""",
"full": """
            F1rV3Block b = f1r_v3_fetch(lo, hi16, tab);
            uint cs[3] = { b.c1, b.c2, b.c3 }; uint rs[3] = { b.row1, b.row2, b.row3 };
            float a = 0.0f, n2 = 0.0f;
            for (uint sec = 0u; sec < 3u; ++sec) for (uint k = 0u; k < 8u; ++k) {
                uint o = b.p + 2u * ((cs[sec] >> k) & 1u); uint rho = (rs[sec] >> (4u * k)) & 15u;
                float v = F1R_VAL[(o << 4) | rho]; n2 = fma(v, v, n2); a = fma(v, xv[sec * 8u + k], a); }
            uint m = (uint(n2) >> 4) & (TETRA48_SHELLS - 1u); uint g = (hi16 >> 15) & 1u;
            float a_r = a * gscale[g] * invnorm[m];
""",
}

# The epilogue without the tail loop (warp_sum, scale, store), and an empty
# kernel at the same grid: the per-row and per-call floors.
TAIL_NT = """
    }
    acc = warp_sum(acc);
    if (lane == 0u) y[row] = half(acc * rscale[row]);
"""
EMPTY = """
    uint row = thread_position_in_grid.x >> 5;
    if (thread_index_in_simdgroup == 0u) y[row] = half(x[row % 64u]);
"""

N, ROUNDS = 200, 5

def main():
    model, pm, _ = L.build(sys.argv[1])
    lin = model.model.layers[0].mlp.gate_proj
    s = lin.shared
    kerns = {k: (mx.fast.metal_kernel(name=f"strip_{k}", input_names=L.INPUTS, output_names=["y"],
                                      source=HEAD + v + TAIL, header=L.HEADER), 1) for k, v in DEC.items()}
    for k in ("full", "stream"):
        kerns[f"{k}_notail"] = (mx.fast.metal_kernel(name=f"strip_{k}_nt", input_names=L.INPUTS, output_names=["y"],
                                                      source=HEAD + DEC[k] + TAIL_NT, header=L.HEADER), 1)
    kerns["empty"] = (mx.fast.metal_kernel(name="strip_empty", input_names=L.INPUTS, output_names=["y"],
                                           source=EMPTY, header=L.HEADER), 1)
    for rr in (2,):
        for k, v in DEC_R.items():
            kerns[f"{k}x{rr}"] = (mx.fast.metal_kernel(name=f"strip_{k}_r", input_names=L.INPUTS,
                                                       output_names=["y"], source=HEAD_R + v + TAIL_R,
                                                       header=L.HEADER), rr)
    x0 = mx.random.normal((1, lin.d_in)).astype(mx.float32); mx.eval(x0)
    for name, (k, rr) in kerns.items():
        def f(xr, k=k, rr=rr):
            return k(inputs=[lin._words, s.rows, s.prefixes, s.branches, s.suffixes, lin._gscale, s.invnorm,
                             lin._rscale, lin._tail, xr],
                     template=[("NB", lin.nb), ("TW", lin.tw), ("STRIDE", lin.stride)] + ([("RR", rr)] if rr > 1 else []),
                     grid=(lin.d_out // rr * 32, 1, 1), threadgroup=(256, 1, 1),
                     output_shapes=[(1, lin.d_out)], output_dtypes=[mx.float16])[0]
        def once():
            x = x0
            for _ in range(N):
                x = f(x)[:, :lin.d_in].astype(mx.float32)
            mx.eval(x)
        once()
        ts = []
        for _ in range(ROUNDS):
            t = time.perf_counter(); once(); ts.append((time.perf_counter() - t) * 1e6 / N)
        m = statistics.median(ts)
        print(f"{name:<8} {m:7.1f} us a call [{min(ts):.1f}-{max(ts):.1f}]  {lin.device_bytes() / (m * 1e-6) / 1e9:6.1f} GB/s")

if __name__ == "__main__":
    main()
