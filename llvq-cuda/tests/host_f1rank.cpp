// Drives the F1 universal-table decoder of `llvq_f1rank.cuh` on the CPU, one
// word at a time.
//
// Reads a fixture on stdin, writes results on stdout, both little-endian
// binary. The Rust side builds the fixture and checks the answers against
// `llvq_bench::f1::rank::decode_word`, so this file holds no expectations — it
// is a harness, not a reference.
//
//   in : u32 n
//        u32[4096] rows, u8[128] prefixes, u16[1024] branches, u8[128] suffixes
//        u64[n] words — bits 0..47 the label; bits 48..63 whatever the Rust
//                       side put there, since `f1r_decode` is told to ignore
//                       the upper half of `hi16`
//   out: i8[n*24] the decoded coordinates, trio order
//
// `f1r_decode` is scalar register arithmetic — no warp primitive, no shared
// memory — so it is *executed* here, exactly as host_probe.cpp executes
// slot_dot. The kernels of f1rank.cu are not included: the tile loop is the
// card's business and `bin/cuhcheck` parses it.

#include "host_shim.h"
// In the order the bench concatenates for NVRTC: llvq_slot.cuh gives `u32`
// and LLVQ_DIM, the decoder header comes after it.
#include "../kernels/llvq_slot.cuh"
#include "../kernels/llvq_f1rank.cuh"

#include <cstdio>
#include <cstdlib>
#include <vector>

// Nothing here launches, but the shim declares these `extern` and a header
// that so much as names a block index needs them defined.
Dim3 blockIdx{0, 0, 0};
Dim3 threadIdx{0, 0, 0};
Dim3 blockDim{1, 1, 1};

template <typename T>
static std::vector<T> read_n(std::FILE* f, std::size_t n, const char* what) {
    std::vector<T> v(n);
    if (n && std::fread(v.data(), sizeof(T), n, f) != n) {
        std::fprintf(stderr, "fixture tronquee: %s\n", what);
        std::exit(2);
    }
    return v;
}

int main() {
    std::FILE* in = stdin;
    const unsigned n = read_n<unsigned>(in, 1, "n")[0];
    const auto rows = read_n<unsigned>(in, 4096, "rows");
    const auto prefixes = read_n<unsigned char>(in, 128, "prefixes");
    const auto branches = read_n<unsigned short>(in, 1024, "branches");
    const auto suffixes = read_n<unsigned char>(in, 128, "suffixes");
    const auto words = read_n<unsigned long long>(in, n, "words");

    const F1rTables t{rows.data(), prefixes.data(), branches.data(), suffixes.data()};
    std::vector<signed char> y(static_cast<std::size_t>(n) * LLVQ_DIM, 0);
    for (unsigned i = 0; i < n; ++i) {
        const unsigned long long w = words[i];
        f1r_decode(static_cast<u32>(w), static_cast<u32>(w >> 32), t,
                   y.data() + static_cast<std::size_t>(i) * LLVQ_DIM);
    }
    std::fwrite(y.data(), 1, y.size(), stdout);
    return 0;
}
