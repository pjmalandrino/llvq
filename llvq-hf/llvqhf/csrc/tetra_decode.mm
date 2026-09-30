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

TORCH_LIBRARY(llvq, m) {
    m.def("tetra_decode(Tensor words, Tensor rows, Tensor prefixes, Tensor branches, "
          "Tensor suffixes, int d_out, int nblocks, int row_stride_u32, str source) "
          "-> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(llvq, MPS, m) {
    m.impl("tetra_decode", &tetra_decode);
}
