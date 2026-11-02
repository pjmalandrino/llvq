// The served CUDA kernels, reached from PyTorch.
//
// The twin of `tetra_decode.mm`, and the same discipline: a binding, no
// arithmetic. Every kernel here is one `llvq_llm::fused_cuda` launches on the
// card today, included from the repository rather than copied:
//
//   * `tv_tetra48_h`, the Tetra matvec;
//   * `tv_q4_h`, the int4 g128 matvec of the `v_proj`, `o_proj` and some
//     `down_proj` records;
//   * `tv_emb_q4_h`, the int4 g64 tied output layer, which reads the same
//     table as the embedding;
//   * `rot_apply_rows`, the rotation the lattice codes were written under,
//     applied to many activation rows in one launch.
//
// Differences from the Metal binding, all of them the kernels':
//
//   * every matvec writes `y` in f16, through `f2h`, where the Metal kernels
//     write f32. The served path reads f16 back, so these return half tensors
//     and the caller widens if it wants to;
//   * the launch geometry is `fused_cuda`'s: one warp a row, `threads / 32`
//     rows a block, `grid = d_out · 32 / threads`, and the shared bytes the
//     kernel stages. `rot_apply_rows` is one block a row instead, with
//     `threads = next_power_of_two(n)` clamped to [32, 1024];
//   * a kernel that stages more than the 48 KiB default asks for the opt-in
//     here, once per call, as `fused_cuda` poses it once per function. The 14B
//     `down_proj` stages 69,632 bytes and the 4B never passes 38,912.
//
// `LLVQ_TILE_BLOCKS` is a compile-time define of the Tetra kernel, as it is on
// Metal, so the caller sets it through `extra_cuda_cflags` and the shared size
// in the same breath.

#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

// The served translation unit, in the order `llvq_llm::fused::planes_source_names`
// gives for `Tetra48` and not in any other: `tv_tetra48_h.cu` uses `F1rTables` and
// `f1r_load`, which `llvq_f1rank.cuh` defines, and its own `#ifndef` guards do not
// reach for that one. Including it alone cost $0.06 and twelve
// `identifier "F1rTables" is undefined`.
//
// `the_cuda_binding_includes_the_served_list` holds this list against the Rust
// one, so the two cannot drift. `TILE_BLOCKS` arrives as a compile flag, which
// `matvec.cu` `#error`s without.
// The served sources, as `bin/tetratables` copies them into `data/kernels/`,
// keeping the repository's two-level shape so that `tv_tetra48_h.cu`'s own
// `#include "../../llvq-cuda/kernels/..."` resolve. Relative paths and no `-I`:
// the same lines then compile from a checkout and from an installed wheel,
// and the bytes included are byte for byte the ones the repository serves,
// which `the_shipped_shader_is_the_repositorys` holds.
#include "../data/kernels/llvq-cuda/kernels/llvq_f1rank.cuh"
#include "../data/kernels/llvq-cuda/kernels/llvq_f1rank_v3.cuh"
#include "../data/kernels/llvq-cuda/kernels/llvq_tetra48.cuh"
#include "../data/kernels/llvq-llm/kernels/tv_tetra48_h.cu"
// After the Tetra unit, so `matvec.cu` (h2f, f2h, warp_sum) and `llvq_slot.cuh`
// (u32) are already in: each of these three guards its own include of them, and
// the guards are then taken. `rotate.cu` reaches `llvq_rot.cuh` by its own
// relative include, which resolves beside it under `data/kernels/llvq-cuda/`.
#include "../data/kernels/llvq-llm/kernels/tv_q4_h.cu"
#include "../data/kernels/llvq-llm/kernels/emb_q4.cu"
#include "../data/kernels/llvq-cuda/kernels/rotate.cu"

namespace {

// The default dynamic shared allowance of a block on every card here.
constexpr int64_t kSharedDefault = 48 * 1024;

// Ask for `bytes` of dynamic shared memory on `fn` when it passes the default,
// refusing past the card's opt-in ceiling rather than letting the launch fail
// with an unnamed error.
template <typename F>
void pose_shared(F fn, int64_t bytes, const char *name) {
    if (bytes <= kSharedDefault) {
        return;
    }
    int dev = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    int optin = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    TORCH_CHECK(bytes <= optin, name, " stages ", bytes, " bytes of shared memory, past the ",
                optin, " this card offers on request");
    C10_CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(bytes)));
}

// The kernels read these blobs as `u32*` and `ushort*`. A fresh allocation is
// aligned far past that; a view at an odd byte offset is not, and would read a
// shifted word without any error.
void require_aligned(const torch::Tensor &t, int64_t bytes, const char *name) {
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % bytes == 0, name,
                " is not ", bytes, "-byte aligned; pass a tensor of its own, not a view");
}

void require_rows(int64_t d_out, int64_t threads) {
    TORCH_CHECK(threads % 32 == 0 && threads > 0, "threads must be a positive multiple of 32");
    TORCH_CHECK(d_out % (threads / 32) == 0, "d_out ", d_out,
                " does not fill whole blocks of ", threads / 32,
                " rows; the kernel carries no row guard");
}

}  // namespace

static torch::Tensor
tv_tetra48_cuda(const torch::Tensor &words, const torch::Tensor &rows,
                const torch::Tensor &prefixes, const torch::Tensor &branches,
                const torch::Tensor &suffixes, const torch::Tensor &gscale,
                const torch::Tensor &invnorm, const torch::Tensor &rscale,
                const torch::Tensor &tail, const torch::Tensor &x, int64_t d_out,
                int64_t nblocks, int64_t tail_w, int64_t row_stride_u32,
                int64_t tile, int64_t threads) {
    require_rows(d_out, threads);
    for (const auto &t : {words, rows, prefixes, branches, suffixes}) {
        TORCH_CHECK(t.dtype() == torch::kUInt8 && t.is_contiguous() && t.is_cuda(),
                    "the stream and the tables are contiguous byte blobs on cuda");
    }
    for (const auto &t : {gscale, invnorm, rscale, x}) {
        TORCH_CHECK(t.dtype() == torch::kFloat32 && t.is_contiguous() && t.is_cuda(),
                    "gscale, invnorm, rscale and x are contiguous f32 on cuda");
    }
    TORCH_CHECK(tail.dtype() == torch::kFloat16, "the kernel reads the tail as half");
    TORCH_CHECK(rscale.numel() == d_out, "one row scale a row");
    TORCH_CHECK(gscale.numel() == 2, "a Tetra record has two gain centroids");
    TORCH_CHECK(x.numel() == nblocks * 24 + tail_w, "x is the whole activation");
    const int64_t want = d_out * row_stride_u32 * 4;
    TORCH_CHECK(words.numel() == want || words.numel() == want + 4,
                "the stream holds ", words.numel(), " bytes, which wants ", want,
                " or one guard word more");

    auto y = torch::empty({d_out}, x.options().dtype(torch::kFloat16));
    const dim3 grid(static_cast<unsigned>(d_out * 32 / threads), 1, 1);
    const dim3 block(static_cast<unsigned>(threads), 1, 1);
    const unsigned shared = static_cast<unsigned>(tile * 24 * 4);
    tv_tetra48_h<<<grid, block, shared, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const unsigned *>(words.data_ptr()),
        static_cast<unsigned>(row_stride_u32),
        reinterpret_cast<const unsigned *>(rows.data_ptr()),
        reinterpret_cast<const unsigned char *>(prefixes.data_ptr()),
        reinterpret_cast<const unsigned short *>(branches.data_ptr()),
        reinterpret_cast<const unsigned char *>(suffixes.data_ptr()),
        gscale.data_ptr<float>(), invnorm.data_ptr<float>(), rscale.data_ptr<float>(),
        reinterpret_cast<const unsigned short *>(tail.data_ptr()),
        x.data_ptr<float>(),
        reinterpret_cast<unsigned short *>(y.data_ptr()),
        static_cast<unsigned>(nblocks), static_cast<unsigned>(tail_w));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

// `tv_q4_h(wq, scales, biases, x, y, d_in, gpr)`, as `fused_cuda::launch_q4_h`
// launches it: `x` is the whole activation in f32 and is staged whole in shared
// memory, `d_in · 4` bytes, so a `down_proj` asks for the opt-in at 8B and 14B.
//
// `wq` is the record's `qweight` as the file stores it, nibbles low first at the
// global flat index; `scales` and `biases` its f16 pairs, one a group of 128.
// The kernel reads `wq` through a u32 pointer, which is row aligned exactly
// when `d_in % 8 == 0`.
static torch::Tensor
tv_q4_cuda(const torch::Tensor &wq, const torch::Tensor &scales, const torch::Tensor &biases,
           const torch::Tensor &x, int64_t d_out, int64_t d_in, int64_t gpr, int64_t threads) {
    require_rows(d_out, threads);
    TORCH_CHECK(d_in % 8 == 0, "d_in ", d_in, " is not a multiple of 8, so a row of nibbles "
                "does not start on a u32");
    TORCH_CHECK(gpr == (d_in + 127) / 128, "gpr ", gpr, " is not ceil(", d_in, " / 128)");
    TORCH_CHECK(wq.dtype() == torch::kUInt8 && wq.is_contiguous() && wq.is_cuda(),
                "wq is a contiguous byte blob on cuda");
    TORCH_CHECK(wq.numel() == d_out * d_in / 2, "wq holds ", wq.numel(), " bytes for ", d_out,
                " rows of ", d_in, " nibbles");
    for (const auto &t : {scales, biases}) {
        TORCH_CHECK(t.dtype() == torch::kFloat16 && t.is_contiguous() && t.is_cuda(),
                    "scales and biases are contiguous f16 on cuda");
        TORCH_CHECK(t.numel() == d_out * gpr, "one scale and one bias a group of 128");
    }
    TORCH_CHECK(x.dtype() == torch::kFloat32 && x.is_contiguous() && x.is_cuda(),
                "x is contiguous f32 on cuda");
    TORCH_CHECK(x.numel() == d_in, "x is the whole activation");
    require_aligned(wq, 4, "wq");
    require_aligned(scales, 2, "scales");
    require_aligned(biases, 2, "biases");

    const int64_t shared = d_in * 4;
    pose_shared(tv_q4_h, shared, "tv_q4_h");
    auto y = torch::empty({d_out}, x.options().dtype(torch::kFloat16));
    const dim3 grid(static_cast<unsigned>(d_out * 32 / threads), 1, 1);
    const dim3 block(static_cast<unsigned>(threads), 1, 1);
    tv_q4_h<<<grid, block, static_cast<unsigned>(shared), c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const unsigned *>(wq.data_ptr()),
        reinterpret_cast<const unsigned short *>(scales.data_ptr()),
        reinterpret_cast<const unsigned short *>(biases.data_ptr()),
        x.data_ptr<float>(),
        reinterpret_cast<unsigned short *>(y.data_ptr()),
        static_cast<unsigned>(d_in), static_cast<unsigned>(gpr));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

// `tv_emb_q4_h(wq, scales, biases, x, y, d, gpr, x_off, y_off)`: the logits of
// one hidden state against the int4 g64 table the embedding also reads. `x` is
// f16 as the model carries it and is widened once into shared, `d · 4` bytes.
// Both offsets are 0: the binding hands over whole tensors.
static torch::Tensor
tv_emb_q4_cuda(const torch::Tensor &wq, const torch::Tensor &scales,
               const torch::Tensor &biases, const torch::Tensor &x, int64_t vocab, int64_t d,
               int64_t gpr, int64_t threads) {
    require_rows(vocab, threads);
    TORCH_CHECK(d % 8 == 0, "d ", d, " is not a multiple of 8, so a row does not start on a u32");
    TORCH_CHECK(gpr == (d + 63) / 64, "gpr ", gpr, " is not ceil(", d, " / 64)");
    TORCH_CHECK(wq.dtype() == torch::kUInt8 && wq.is_contiguous() && wq.is_cuda(),
                "wq is a contiguous byte blob on cuda");
    TORCH_CHECK(wq.numel() == vocab * d / 2, "wq holds ", wq.numel(), " bytes for ", vocab,
                " rows of ", d, " nibbles");
    for (const auto &t : {scales, biases}) {
        TORCH_CHECK(t.dtype() == torch::kFloat16 && t.is_contiguous() && t.is_cuda(),
                    "scales and biases are contiguous f16 on cuda");
        TORCH_CHECK(t.numel() == vocab * gpr, "one scale and one bias a group of 64");
    }
    TORCH_CHECK(x.dtype() == torch::kFloat16 && x.is_contiguous() && x.is_cuda(),
                "x is the hidden state, contiguous f16 on cuda");
    TORCH_CHECK(x.numel() == d, "x is one hidden state");
    require_aligned(wq, 4, "wq");
    require_aligned(scales, 2, "scales");
    require_aligned(biases, 2, "biases");

    const int64_t shared = d * 4;
    pose_shared(tv_emb_q4_h, shared, "tv_emb_q4_h");
    auto y = torch::empty({vocab}, x.options());
    const dim3 grid(static_cast<unsigned>(vocab * 32 / threads), 1, 1);
    const dim3 block(static_cast<unsigned>(threads), 1, 1);
    tv_emb_q4_h<<<grid, block, static_cast<unsigned>(shared),
                  c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const unsigned *>(wq.data_ptr()),
        reinterpret_cast<const unsigned short *>(scales.data_ptr()),
        reinterpret_cast<const unsigned short *>(biases.data_ptr()),
        reinterpret_cast<const unsigned short *>(x.data_ptr()),
        reinterpret_cast<unsigned short *>(y.data_ptr()),
        static_cast<unsigned>(d), static_cast<unsigned>(gpr), 0u, 0u);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

// `rot_apply_rows(xin, signbits, small, xout, n, m, k, inv, x_off, row_stride)`:
// `[n_rows, n]` f16 in, the same shape in f32 out, rotated, one block a row.
//
// What the kernel owes nothing for and the host must assert, from `rotate.cu`:
// `n = k · m` with `m` a power of two and `k ≤ LLVQ_ROT_KMAX`, `small` zero
// padded to `KMAX × KMAX` row-major, `inv = 1/√m` computed in f64 and narrowed
// by the caller, and `n · 4` bytes of shared memory. `signbits` is one bit a
// coordinate, set when the sign is negative, read as u32.
static torch::Tensor
rot_apply_rows_cuda(const torch::Tensor &x, const torch::Tensor &signbits,
                    const torch::Tensor &small, int64_t n, int64_t m, int64_t k, double inv,
                    int64_t threads) {
    TORCH_CHECK(m > 0 && (m & (m - 1)) == 0, "m ", m, " is not a power of two");
    TORCH_CHECK(k >= 1 && k <= static_cast<int64_t>(LLVQ_ROT_KMAX), "the odd factor ", k,
                " is outside [1, ", LLVQ_ROT_KMAX, "], which the kernel handles");
    TORCH_CHECK(n == k * m, "n ", n, " is not k · m = ", k * m);
    TORCH_CHECK(threads >= 32 && threads <= 1024 && threads % 32 == 0,
                "one block a row, threads in [32, 1024] and a whole number of warps");
    TORCH_CHECK(x.dtype() == torch::kFloat16 && x.is_contiguous() && x.is_cuda(),
                "x is contiguous f16 on cuda, as the model carries it");
    TORCH_CHECK(x.dim() == 2 && x.size(1) == n, "x is [n_rows, ", n, "]");
    TORCH_CHECK(signbits.dtype() == torch::kInt32 && signbits.is_contiguous() && signbits.is_cuda(),
                "signbits are contiguous int32 on cuda, read as u32");
    TORCH_CHECK(signbits.numel() == (n + 31) / 32, "one sign bit a coordinate");
    TORCH_CHECK(small.dtype() == torch::kFloat32 && small.is_contiguous() && small.is_cuda(),
                "small is contiguous f32 on cuda");
    TORCH_CHECK(small.numel() == static_cast<int64_t>(LLVQ_ROT_KMAX * LLVQ_ROT_KMAX),
                "small is padded to KMAX × KMAX");

    const int64_t n_rows = x.size(0);
    auto y = torch::empty({n_rows, n}, x.options().dtype(torch::kFloat32));
    if (n_rows == 0) {
        return y;
    }
    const int64_t shared = n * 4;
    pose_shared(rot_apply_rows, shared, "rot_apply_rows");
    const dim3 grid(static_cast<unsigned>(n_rows), 1, 1);
    const dim3 block(static_cast<unsigned>(threads), 1, 1);
    rot_apply_rows<<<grid, block, static_cast<unsigned>(shared),
                     c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const unsigned short *>(x.data_ptr()),
        reinterpret_cast<const unsigned *>(signbits.data_ptr()),
        small.data_ptr<float>(), y.data_ptr<float>(), static_cast<unsigned>(n),
        static_cast<unsigned>(m), static_cast<unsigned>(k), static_cast<float>(inv), 0u,
        static_cast<unsigned>(n));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

TORCH_LIBRARY(llvq_cuda, m) {
    m.def("tv_tetra48(Tensor words, Tensor rows, Tensor prefixes, Tensor branches, "
          "Tensor suffixes, Tensor gscale, Tensor invnorm, Tensor rscale, Tensor tail, "
          "Tensor x, int d_out, int nblocks, int tail_w, int row_stride_u32, int tile, "
          "int threads) -> Tensor");
    m.def("tv_q4(Tensor wq, Tensor scales, Tensor biases, Tensor x, int d_out, int d_in, "
          "int gpr, int threads) -> Tensor");
    m.def("tv_emb_q4(Tensor wq, Tensor scales, Tensor biases, Tensor x, int vocab, int d, "
          "int gpr, int threads) -> Tensor");
    m.def("rot_apply_rows(Tensor x, Tensor signbits, Tensor small, int n, int m, int k, "
          "float inv, int threads) -> Tensor");
}

TORCH_LIBRARY_IMPL(llvq_cuda, CUDA, m) {
    m.impl("tv_tetra48", &tv_tetra48_cuda);
    m.impl("tv_q4", &tv_q4_cuda);
    m.impl("tv_emb_q4", &tv_emb_q4_cuda);
    m.impl("rot_apply_rows", &rot_apply_rows_cuda);
}
