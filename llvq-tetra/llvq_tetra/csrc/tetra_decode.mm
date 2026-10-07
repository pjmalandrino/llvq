// The served Tetra shader, reached from PyTorch on Metal.
//
// One op, `tetra48_probe`, which is the shader's own decoder entry point: one
// thread a block, no tile, no reduction. It exists in
// `llvq-llm/kernels/llvq_tetra48.metal` so the decode can be judged on its own,
// and `llvq-metal/tests/tetra48_matches_rust.rs` already compares what it writes
// against `llvq_search::tetra::Tetra`, exactly.
//
// ## What this file is and is not
//
// It is a binding: compile the source it is handed, bind five buffers, dispatch,
// commit. It contains no arithmetic, on purpose. Anything computed here would be
// a second implementation with no reference.
//
// The tables arrive as **byte blobs**. Metal reads `rows` as `uint*` and
// `branches` as `ushort*`, and torch's unsigned integer dtypes are a moving
// target; bytes have no such question, and the layout is decided once, in
// `llvq_llm::hfpack::tetra_tables`.
//
// ## Fast math
//
// Off, as `llvq-metal/src/lib.rs` compiles it. The shader also carries
// `#pragma clang fp contract(off)` for the contraction fast math does not
// cover. Neither matters for a decode, which converts integers and multiplies
// nothing; they are set because the gate is equality and a compile option is
// not the place to differ from the served path.

#include <torch/extension.h>
#include <torch/mps.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

static inline id<MTLBuffer> mtl(const torch::Tensor &t) {
    return __builtin_bit_cast(id<MTLBuffer>, t.storage().data());
}

static inline NSUInteger offset_of(const torch::Tensor &t) {
    return static_cast<NSUInteger>(t.storage_offset() * t.element_size());
}

static id<MTLComputePipelineState> pipeline_for(const std::string &source,
                                                const std::string &entry) {
    static NSMutableDictionary<NSString *, id> *cache = [NSMutableDictionary new];
    static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    TORCH_CHECK(device != nil, "no Metal device");

    NSString *src = [NSString stringWithUTF8String:source.c_str()];
    NSString *name = [NSString stringWithUTF8String:entry.c_str()];
    NSString *key = [NSString stringWithFormat:@"%lu/%@", (unsigned long)[src hash], name];
    id<MTLComputePipelineState> cached = cache[key];
    if (cached != nil) {
        return cached;
    }
    NSError *error = nil;
    MTLCompileOptions *options = [MTLCompileOptions new];
    options.fastMathEnabled = NO;
    id<MTLLibrary> library = [device newLibraryWithSource:src options:options error:&error];
    TORCH_CHECK(library != nil, "Metal compile failed: ",
                error ? error.localizedDescription.UTF8String : "no error given");
    id<MTLFunction> function = [library newFunctionWithName:name];
    TORCH_CHECK(function != nil, "the shader has no function named ", entry);
    id<MTLComputePipelineState> state =
        [device newComputePipelineStateWithFunction:function error:&error];
    TORCH_CHECK(state != nil, "pipeline failed: ",
                error ? error.localizedDescription.UTF8String : "no error given");
    cache[key] = state;
    return state;
}

// `tetra48_probe`: one block a thread, the served row-strided little-endian
// words in, 24 coordinates and the shell out.
static std::tuple<torch::Tensor, torch::Tensor>
tetra_decode(const torch::Tensor &words, const torch::Tensor &rows,
             const torch::Tensor &prefixes, const torch::Tensor &branches,
             const torch::Tensor &suffixes, int64_t d_out, int64_t nblocks,
             int64_t row_stride_u32, const std::string &source) {
    TORCH_CHECK(words.device().is_mps(), "words must be on mps, got ", words.device());
    for (const auto &t : {words, rows, prefixes, branches, suffixes}) {
        TORCH_CHECK(t.dtype() == torch::kUInt8, "the buffers are byte blobs, got ", t.dtype());
        TORCH_CHECK(t.is_contiguous(), "a non-contiguous buffer would be read as if it were not");
        TORCH_CHECK(t.device().is_mps(), "every buffer must be on mps");
    }
    TORCH_CHECK(d_out > 0 && nblocks > 0, "d_out and nblocks must be positive");
    TORCH_CHECK(words.numel() == d_out * row_stride_u32 * 4,
                "words holds ", words.numel(), " bytes for ", d_out, " rows of ",
                row_stride_u32, " u32");

    const int64_t n = d_out * nblocks;
    auto options = torch::TensorOptions().device(words.device());
    torch::Tensor points = torch::empty({n, 24}, options.dtype(torch::kFloat32));
    torch::Tensor shell = torch::empty({n}, options.dtype(torch::kInt32));

    id<MTLComputePipelineState> state = pipeline_for(source, "tetra48_probe");
    dispatch_queue_t queue = torch::mps::get_dispatch_queue();
    dispatch_sync(queue, ^() {
        id<MTLCommandBuffer> buffer = torch::mps::get_command_buffer();
        TORCH_CHECK(buffer != nil, "no MPS command buffer");
        id<MTLComputeCommandEncoder> encoder = [buffer computeCommandEncoder];
        [encoder setComputePipelineState:state];
        [encoder setBuffer:mtl(words) offset:offset_of(words) atIndex:0];
        [encoder setBuffer:mtl(rows) offset:offset_of(rows) atIndex:1];
        [encoder setBuffer:mtl(prefixes) offset:offset_of(prefixes) atIndex:2];
        [encoder setBuffer:mtl(branches) offset:offset_of(branches) atIndex:3];
        [encoder setBuffer:mtl(suffixes) offset:offset_of(suffixes) atIndex:4];
        [encoder setBuffer:mtl(points) offset:offset_of(points) atIndex:5];
        [encoder setBuffer:mtl(shell) offset:offset_of(shell) atIndex:6];
        uint32_t stride = static_cast<uint32_t>(row_stride_u32);
        uint32_t blocks = static_cast<uint32_t>(nblocks);
        [encoder setBytes:&stride length:sizeof(stride) atIndex:7];
        [encoder setBytes:&blocks length:sizeof(blocks) atIndex:8];

        NSUInteger width = state.maxTotalThreadsPerThreadgroup;
        if (width > static_cast<NSUInteger>(n)) {
            width = static_cast<NSUInteger>(n);
        }
        [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(n), 1, 1)
          threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
        [encoder endEncoding];
        torch::mps::commit();
    });
    return {points, shell};
}

// `tv_tetra48_metal`: one projection against one activation, the weights never
// materialized. The served kernel, bound and not reimplemented.
//
// `d_out` threads times 32, a threadgroup of 256, and the tile of activation
// blocks staged in threadgroup memory at index 0. The caller prepends
// `#define LLVQ_TILE_BLOCKS` to the source, so the tile and the buffer length
// cannot disagree: one number, passed once.
static torch::Tensor
tv_tetra48(const torch::Tensor &words, const torch::Tensor &rows,
           const torch::Tensor &prefixes, const torch::Tensor &branches,
           const torch::Tensor &suffixes, const torch::Tensor &gscale,
           const torch::Tensor &invnorm, const torch::Tensor &rscale,
           const torch::Tensor &tail, const torch::Tensor &x, int64_t d_out,
           int64_t nblocks, int64_t tail_w, int64_t row_stride_u32, int64_t tile,
           const std::string &source) {
    TORCH_CHECK(d_out % 8 == 0,
                "d_out ", d_out, " is not a multiple of 8. Thirty-two threads take a row and the "
                "threadgroup is 256, and the kernel carries no row guard: a partial group would "
                "store past the output");
    TORCH_CHECK(tile > 0 && tile <= 4096, "the tile must be positive and sane, got ", tile);
    for (const auto &t : {words, rows, prefixes, branches, suffixes}) {
        TORCH_CHECK(t.dtype() == torch::kUInt8 && t.is_contiguous() && t.device().is_mps(),
                    "the word stream and the tables are contiguous byte blobs on mps");
    }
    for (const auto &t : {gscale, invnorm, rscale, x}) {
        TORCH_CHECK(t.dtype() == torch::kFloat32 && t.is_contiguous() && t.device().is_mps(),
                    "gscale, invnorm, rscale and x are contiguous f32 on mps, got ", t.dtype());
    }
    TORCH_CHECK(tail.dtype() == torch::kFloat16 && tail.is_contiguous(),
                "the kernel reads the tail as half, got ", tail.dtype());
    TORCH_CHECK(rscale.numel() == d_out, rscale.numel(), " row scales for ", d_out, " rows");
    TORCH_CHECK(gscale.numel() == 2, "a Tetra record has two gain centroids");
    TORCH_CHECK(x.numel() == nblocks * 24 + tail_w,
                "x holds ", x.numel(), " values for ", nblocks, " blocks and a tail of ", tail_w);
    TORCH_CHECK(tail.numel() == std::max<int64_t>(1, d_out * tail_w),
                "the tail holds ", tail.numel(), " values for ", d_out, " by ", tail_w);
    const int64_t want = d_out * row_stride_u32 * 4;
    TORCH_CHECK(words.numel() == want || words.numel() == want + 4,
                "the stream holds ", words.numel(), " bytes for ", d_out, " rows of ",
                row_stride_u32, " words, which wants ", want, " or one guard word more");

    torch::Tensor y = torch::empty({d_out}, x.options());
    id<MTLComputePipelineState> state = pipeline_for(source, "tv_tetra48_metal");
    dispatch_queue_t queue = torch::mps::get_dispatch_queue();
    dispatch_sync(queue, ^() {
        id<MTLCommandBuffer> buffer = torch::mps::get_command_buffer();
        TORCH_CHECK(buffer != nil, "no MPS command buffer");
        id<MTLComputeCommandEncoder> encoder = [buffer computeCommandEncoder];
        [encoder setComputePipelineState:state];
        uint32_t stride = static_cast<uint32_t>(row_stride_u32);
        uint32_t blocks = static_cast<uint32_t>(nblocks);
        uint32_t tw = static_cast<uint32_t>(tail_w);
        [encoder setBuffer:mtl(words) offset:offset_of(words) atIndex:0];
        [encoder setBytes:&stride length:sizeof(stride) atIndex:1];
        [encoder setBuffer:mtl(rows) offset:offset_of(rows) atIndex:2];
        [encoder setBuffer:mtl(prefixes) offset:offset_of(prefixes) atIndex:3];
        [encoder setBuffer:mtl(branches) offset:offset_of(branches) atIndex:4];
        [encoder setBuffer:mtl(suffixes) offset:offset_of(suffixes) atIndex:5];
        [encoder setBuffer:mtl(gscale) offset:offset_of(gscale) atIndex:6];
        [encoder setBuffer:mtl(invnorm) offset:offset_of(invnorm) atIndex:7];
        [encoder setBuffer:mtl(rscale) offset:offset_of(rscale) atIndex:8];
        [encoder setBuffer:mtl(tail) offset:offset_of(tail) atIndex:9];
        [encoder setBuffer:mtl(x) offset:offset_of(x) atIndex:10];
        [encoder setBuffer:mtl(y) offset:offset_of(y) atIndex:11];
        [encoder setBytes:&blocks length:sizeof(blocks) atIndex:12];
        [encoder setBytes:&tw length:sizeof(tw) atIndex:13];
        [encoder setThreadgroupMemoryLength:static_cast<NSUInteger>(tile * 24 * 4) atIndex:0];
        [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(d_out) * 32, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        torch::mps::commit();
    });
    return y;
}

// The int4 g128 matvec, both entry points: the served one and the tiled one.
//
// Two ops and not one with a flag, because the gate is that they agree: a flag
// would let a caller think it had compared them when it had run one twice.
//
// `tile_cols` a multiple of 256 is the whole of the tiled kernel's bit-identity
// argument, so it is refused here rather than rounded up. Rounding would change
// the arithmetic of a served number silently.
static torch::Tensor
tv_q4(const torch::Tensor &wq, const torch::Tensor &scales, const torch::Tensor &biases,
      const torch::Tensor &x, int64_t d_out, int64_t d_in, int64_t gpr, int64_t tile_cols,
      const std::string &source) {
    const bool tiled = tile_cols > 0;
    TORCH_CHECK(d_in % 8 == 0,
                "d_in ", d_in, " is not a multiple of 8. Eight nibbles a word is what makes a "
                "row start on a word, and without it every row after the first reads at a "
                "shifted nibble");
    TORCH_CHECK(d_out % 8 == 0, "d_out ", d_out, " is not a multiple of 8");
    const int64_t staged = tiled ? tile_cols : d_in;
    TORCH_CHECK(staged * 4 <= 32768,
                "staging ", staged, " columns wants ", staged * 4,
                " B of threadgroup memory against 32768. That is the wall the tiled entry "
                "point exists for");
    if (tiled) {
        TORCH_CHECK(tile_cols % 256 == 0,
                    "tile_cols ", tile_cols, " is not a multiple of 256. A tile of tile_cols/8 "
                    "words must be a multiple of 32 or the lanes interleave differently and the "
                    "sum is a different f32");
    }
    for (const auto &t : {wq, scales, biases}) {
        TORCH_CHECK(t.dtype() == torch::kUInt8 && t.is_contiguous() && t.device().is_mps(),
                    "the packed weights and the scale pairs are contiguous byte blobs on mps");
    }
    TORCH_CHECK(x.dtype() == torch::kFloat32 && x.is_contiguous() && x.device().is_mps(),
                "x is contiguous f32 on mps");
    TORCH_CHECK(x.numel() == d_in, "x holds ", x.numel(), " values for d_in ", d_in);
    TORCH_CHECK(wq.numel() == d_out * d_in / 2,
                "the packed weights hold ", wq.numel(), " bytes for ", d_out, " by ", d_in);
    TORCH_CHECK(scales.numel() == d_out * gpr * 2 && biases.numel() == scales.numel(),
                "one f16 scale and one f16 bias a group, ", d_out * gpr, " groups");

    torch::Tensor y = torch::empty({d_out}, x.options());
    id<MTLComputePipelineState> state =
        pipeline_for(source, tiled ? "tv_q4_metal_tiled" : "tv_q4_metal");
    dispatch_queue_t queue = torch::mps::get_dispatch_queue();
    dispatch_sync(queue, ^() {
        id<MTLCommandBuffer> buffer = torch::mps::get_command_buffer();
        TORCH_CHECK(buffer != nil, "no MPS command buffer");
        id<MTLComputeCommandEncoder> encoder = [buffer computeCommandEncoder];
        [encoder setComputePipelineState:state];
        uint32_t din = static_cast<uint32_t>(d_in);
        uint32_t groups = static_cast<uint32_t>(gpr);
        uint32_t tile = static_cast<uint32_t>(tile_cols);
        [encoder setBuffer:mtl(wq) offset:offset_of(wq) atIndex:0];
        [encoder setBuffer:mtl(scales) offset:offset_of(scales) atIndex:1];
        [encoder setBuffer:mtl(biases) offset:offset_of(biases) atIndex:2];
        [encoder setBuffer:mtl(x) offset:offset_of(x) atIndex:3];
        [encoder setBuffer:mtl(y) offset:offset_of(y) atIndex:4];
        [encoder setBytes:&din length:sizeof(din) atIndex:5];
        [encoder setBytes:&groups length:sizeof(groups) atIndex:6];
        if (tiled) {
            [encoder setBytes:&tile length:sizeof(tile) atIndex:7];
        }
        [encoder setThreadgroupMemoryLength:static_cast<NSUInteger>(staged * 4) atIndex:0];
        [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(d_out) * 32, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        torch::mps::commit();
    });
    return y;
}

TORCH_LIBRARY(llvq, m) {
    m.def("tetra_decode(Tensor words, Tensor rows, Tensor prefixes, Tensor branches, "
          "Tensor suffixes, int d_out, int nblocks, int row_stride_u32, str source) "
          "-> (Tensor, Tensor)");
    m.def("tv_tetra48(Tensor words, Tensor rows, Tensor prefixes, Tensor branches, "
          "Tensor suffixes, Tensor gscale, Tensor invnorm, Tensor rscale, Tensor tail, "
          "Tensor x, int d_out, int nblocks, int tail_w, int row_stride_u32, int tile, "
          "str source) -> Tensor");
    // `tile_cols` of 0 selects the served entry point, which stages the whole
    // activation; anything else selects the tiled one.
    m.def("tv_q4(Tensor wq, Tensor scales, Tensor biases, Tensor x, int d_out, int d_in, "
          "int gpr, int tile_cols, str source) -> Tensor");
}

TORCH_LIBRARY_IMPL(llvq, MPS, m) {
    m.impl("tetra_decode", &tetra_decode);
    m.impl("tv_tetra48", &tv_tetra48);
    m.impl("tv_q4", &tv_q4);
}
