// Drives the CUDA kernels on the CPU, one "thread" at a time.
//
// Reads a fixture on stdin, writes results on stdout, both little-endian
// binary. The Rust side builds the fixture from `RuntimeBlocks` and checks the
// answer against `decode_block`, so this file holds no expectations of its
// own — it is a harness, not a reference.
//
//   in : u32 nblocks, u32 nwords, u32 ntab, u32 nx
//        u32[nwords] words, u32[nbases] bases (nbases = ceil(nblocks/32) + 1),
//        u32[ntab] tab, f32[2] gscale, f32[nx] x
//   out: u32[nblocks*24] slots, u32[nblocks*2] cls, f32[nblocks] dot,
//        f32[nblocks] floor

#include "host_shim.h"
// Deliberately in the order NVRTC sees them, and with the header first, so
// that the `#ifndef` guard in the `.cu` takes the same branch here as it does
// on the device. Including only the `.cu` would let clang resolve the include
// itself and leave the concatenated form — the one that actually ships —
// untested. It cost one failed job to learn that.
#include "../kernels/llvq_slot.cuh"
#include "../kernels/preflight.cu"

#include <cstdio>
#include <cstdlib>
#include <vector>

Dim3 blockIdx{0, 0, 0};
Dim3 threadIdx{0, 0, 0};
Dim3 blockDim{256, 1, 1};

template <typename T>
static std::vector<T> read_n(std::FILE* f, std::size_t n, const char* what) {
    std::vector<T> v(n);
    if (n && std::fread(v.data(), sizeof(T), n, f) != n) {
        std::fprintf(stderr, "fixture tronquee: %s\n", what);
        std::exit(2);
    }
    return v;
}

// One "launch": walk the grid the way the driver would, so the guard
// `if (b >= nblocks) return;` is exercised on the ragged last block too.
template <typename F>
static void launch(unsigned n, F body) {
    unsigned nb = (n + blockDim.x - 1) / blockDim.x;
    for (unsigned bi = 0; bi < nb; ++bi) {
        blockIdx.x = bi;
        for (unsigned ti = 0; ti < blockDim.x; ++ti) {
            threadIdx.x = ti;
            body();
        }
    }
}

int main() {
    std::FILE* in = stdin;
    auto hdr = read_n<unsigned>(in, 4, "en-tete");
    unsigned nblocks = hdr[0], nwords = hdr[1], ntab = hdr[2], nx = hdr[3];
    // ceil, not floor. The last group is partial whenever nblocks is not a
    // multiple of 32 — which it is not, for any real layer — and `slot_dot`
    // reads `bases[g + 1]` to get the stride, so one entry short is an
    // out-of-bounds read of the last group on the device.
    unsigned nbases = (nblocks + 31u) / 32u + 1u;

    auto words = read_n<unsigned>(in, nwords, "words");
    auto bases = read_n<unsigned>(in, nbases, "bases");
    auto tab_raw = read_n<unsigned>(in, ntab, "tab");
    auto gscale = read_n<float>(in, 2, "gscale");
    auto x = read_n<float>(in, nx, "x");

    const ClassRec* tab = reinterpret_cast<const ClassRec*>(tab_raw.data());

    std::vector<unsigned> slots(static_cast<std::size_t>(nblocks) * LLVQ_DIM, 0);
    std::vector<unsigned> cls(static_cast<std::size_t>(nblocks) * 2, 0);
    std::vector<float> dot(nblocks, 0.0f);
    std::vector<float> flr(nblocks, 0.0f);

    launch(nblocks, [&] {
        decode_probe(words.data(), bases.data(), tab, nblocks, slots.data(), cls.data());
    });
    launch(nblocks, [&] {
        dot_probe(words.data(), bases.data(), tab, gscale.data(), x.data(), nblocks,
                  dot.data());
    });
    launch(nblocks, [&] {
        floor_probe(words.data(), bases.data(), nblocks, flr.data());
    });

    std::fwrite(slots.data(), sizeof(unsigned), slots.size(), stdout);
    std::fwrite(cls.data(), sizeof(unsigned), cls.size(), stdout);
    std::fwrite(dot.data(), sizeof(float), dot.size(), stdout);
    std::fwrite(flr.data(), sizeof(float), flr.size(), stdout);
    return 0;
}
