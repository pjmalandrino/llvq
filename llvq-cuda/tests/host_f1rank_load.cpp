// Drives `f1r_load` of `llvq_f1rank.cuh` on the CPU: the two aligned u32 a
// lane reads to assemble its 6-byte word from a row of the stream.
//
// Separate from host_f1rank.cpp on purpose: the decode is the contract the
// spec fixes, the load is a helper whose signature the same spec asks for —
// if the helper is missing or renamed, only this harness fails to compile and
// the decode check still stands.
//
//   in : u32 nblocks, u32 nrow            (nrow = row stride in u32, stride
//        u32[nrow] row                     in bytes = round_up(6·nblocks, 8))
//   out: u32[nblocks*2] (lo, hi16) per block

#include "host_shim.h"
#include "../kernels/llvq_slot.cuh"
#include "../kernels/llvq_f1rank.cuh"

#include <cstdio>
#include <cstdlib>
#include <vector>

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
    const auto hdr = read_n<unsigned>(in, 2, "en-tete");
    const unsigned nblocks = hdr[0], nrow = hdr[1];
    const auto row = read_n<unsigned>(in, nrow, "row");

    std::vector<unsigned> out(static_cast<std::size_t>(nblocks) * 2, 0);
    for (unsigned j = 0; j < nblocks; ++j) {
        u32 lo = 0, hi16 = 0;
        f1r_load(row.data(), j, lo, hi16);
        out[static_cast<std::size_t>(j) * 2] = lo;
        out[static_cast<std::size_t>(j) * 2 + 1] = hi16;
    }
    std::fwrite(out.data(), sizeof(unsigned), out.size(), stdout);
    return 0;
}
