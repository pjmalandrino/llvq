// The served CUDA Tetra matvec, reached from PyTorch.
//
// The twin of `tetra_decode.mm`, and the same discipline: a binding, no
// arithmetic. The kernel is `tv_tetra48_h`, the one `llvq_llm::fused_cuda`
// launches on the card today, included from the repository rather than copied.
//
// Two differences from the Metal binding, both the kernel's:
//
//   * `y` is f16, written through `f2h`, where the Metal kernel writes f32. The
//     served path reads f16 back, so this returns a half tensor and the caller
//     widens if it wants to.
//   * the launch geometry comes from `fused_cuda::launch_tetra48`: one warp a
//     row, `threads / 32` rows a block, `grid = d_out · 32 / threads`, and the
//     shared bytes the caller passes, `tile · 24 · 4`.
//
// `LLVQ_TILE_BLOCKS` is a compile-time define of the kernel, as it is on Metal,
// so the caller sets it through `extra_cuda_cflags` and the shared size in the
// same breath.

#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include "tv_tetra48_h.cu"

static torch::Tensor
tv_tetra48_cuda(const torch::Tensor &words, const torch::Tensor &rows,
                const torch::Tensor &prefixes, const torch::Tensor &branches,
                const torch::Tensor &suffixes, const torch::Tensor &gscale,
                const torch::Tensor &invnorm, const torch::Tensor &rscale,
                const torch::Tensor &tail, const torch::Tensor &x, int64_t d_out,
                int64_t nblocks, int64_t tail_w, int64_t row_stride_u32,
                int64_t tile, int64_t threads) {
    TORCH_CHECK(threads % 32 == 0 && threads > 0, "threads must be a positive multiple of 32");
    TORCH_CHECK(d_out % (threads / 32) == 0,
                "d_out ", d_out, " does not fill whole blocks of ", threads / 32,
                " rows; the kernel carries no row guard");
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

TORCH_LIBRARY(llvq_cuda, m) {
    m.def("tv_tetra48(Tensor words, Tensor rows, Tensor prefixes, Tensor branches, "
          "Tensor suffixes, Tensor gscale, Tensor invnorm, Tensor rscale, Tensor tail, "
          "Tensor x, int d_out, int nblocks, int tail_w, int row_stride_u32, int tile, "
          "int threads) -> Tensor");
}

TORCH_LIBRARY_IMPL(llvq_cuda, CUDA, m) {
    m.impl("tv_tetra48", &tv_tetra48_cuda);
}
