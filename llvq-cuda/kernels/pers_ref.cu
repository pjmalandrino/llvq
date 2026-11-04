// pers_ref.cu: the L2 bench's reference for the norm prologue of
// `pchain_wp` (prereg-persistent-1 gate 7).
//
// candle-kernels 0.9.2's `rmsnorm` (reduce.cu:187-228) transcribed for one
// row, line for line, in candle's own shape: one CTA of `block_size`
// threads (1,024 from 1,024 columns, 32 below, candle-nn ops.rs:545), the
// strided `tmp += xi * xi`, the warp butterfly, the 32 partials in shared
// memory, the butterfly again, `tmp / ncols`, `rsqrtf(mean + eps)`, and
// `scale * x * a`. It is an independent transcription, not the persistent
// code's virtual-warp mapping, and it calls the same `rsqrtf` instruction
// candle does, which no host replay can. It is not candle: route P's
// load-time check holds the emulation against candle's kernel itself.
//
// Launched with blockDim.x == block_size, one CTA. Needs warp_sum, h2f, f2h
// (matvec.cu).

extern "C" __global__ void pers_ref_rmsnorm(const unsigned short* __restrict__ x,
                                            unsigned short* __restrict__ dst,
                                            const unsigned short* __restrict__ alpha,
                                            int ncols,
                                            int block_size,
                                            float eps)
{
    const int tid = threadIdx.x;
    float tmp = 0.0f;
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = h2f(x[col]);
        tmp += xi * xi;
    }
    tmp = warp_sum(tmp);
    if (block_size > 32) {
        __shared__ float s_sum[32];
        const int warp_id = threadIdx.x / 32;
        const int lane_id = threadIdx.x % 32;
        if (lane_id == 0) s_sum[warp_id] = tmp;
        __syncthreads();
        tmp = s_sum[lane_id];
        tmp = warp_sum(tmp);
    }
    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);
    for (int col = tid; col < ncols; col += block_size) {
        const float a = h2f(alpha[col]);
        dst[col] = f2h(scale * h2f(x[col]) * a);
    }
}
