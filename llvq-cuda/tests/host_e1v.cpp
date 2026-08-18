// Drives the E1v decoder on the CPU, one block at a time.
//
// Reads a fixture on stdin, writes results on stdout, both little-endian
// binary. The Rust side builds the fixture with `transcode_e1v_rows` and checks
// the answer against `E1vBlocks::decode_block`, so this file holds no
// expectations of its own — it is a harness, not a reference.
//
//   in : u32 nblocks, u32 nwords, u32 nrec, u32 npay
//        u32[nwords]  data          the row-aligned stream, two words padded
//        u32[nrec*27] tab           E1vRec[nrec], 108 bytes each
//        u32[npay]    pay           payload bits per header id
//        u32[625]     binom         C(p,c) transposed
//        u32[4096]    golay         the codewords, flat
//        f32[2]       gscale
//        f32[24]      x
//        per block:   u32 gbit_lo, u32 gbit_hi, u32 len, u32 off, u32 lane
//   out: f32[nblocks] dot, u32[nblocks] id, u32[nblocks] gain
//
// 🚨 **What is NOT driven here, and why.** `e1v_dot` and `e1v_scan_exclusive`
// are compile-only: `__shfl_up_sync` has no host semantics, and `host_shim.h`
// says the same of every shuffle in this crate. So the fixture carries each
// lane's `off` — the value the scan produces — and this harness exercises
// everything downstream of it, which is where the bit arithmetic lives. The
// scan's own value is checked in Rust, by `llvq-artifact/tests/e1v_cuda_mirror.rs`.
//
// What it DOES drive that the mirror cannot: this text. The mirror is a
// transcription and could drift; these are the bytes NVRTC will compile.

#include "host_shim.h"
// Deliberately in the order NVRTC sees them, with the header first, so the
// `#ifndef` guard takes the same branch here as on the device.
#include "../kernels/llvq_slot.cuh"
#include "../kernels/llvq_e1v.cuh"

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

int main() {
    std::FILE* in = stdin;
    auto hdr = read_n<unsigned>(in, 4, "en-tete");
    unsigned nblocks = hdr[0], nwords = hdr[1], nrec = hdr[2], npay = hdr[3];

    auto data  = read_n<unsigned>(in, nwords, "data");
    auto tabw  = read_n<unsigned>(in, nrec * (sizeof(E1vRec) / 4), "tab");
    auto pay   = read_n<unsigned>(in, npay, "pay");
    auto binom = read_n<unsigned>(in, 625, "binom");
    auto golay = read_n<unsigned>(in, 4096, "golay");
    auto gs    = read_n<float>(in, 2, "gscale");
    auto x     = read_n<float>(in, 24, "x");
    auto addr  = read_n<unsigned>(in, nblocks * 5, "adresses");

    const E1vRec* tab = reinterpret_cast<const E1vRec*>(tabw.data());

    std::vector<float> dot(nblocks);
    std::vector<unsigned> ids(nblocks), gains(nblocks);
    for (unsigned b = 0; b < nblocks; ++b) {
        unsigned long long gbit =
            (unsigned long long)addr[5 * b] | ((unsigned long long)addr[5 * b + 1] << 32);
        unsigned len  = addr[5 * b + 2];
        unsigned off  = addr[5 * b + 3];
        // ⚠️ The lane is given, not computed as `b % 32`: with the row-aligned
        // cut a group ends where its ROW ends, so a block's lane is
        // `(b mod row_blocks) mod 32` and the naive form is wrong on every row
        // after the first.
        unsigned lane = addr[5 * b + 4];
        unsigned h = e1v_read_small(data.data(), gbit + LLVQ_E1V_HDR_BITS * lane,
                                    LLVQ_E1V_HDR_BITS);
        ids[b]   = h & 0x1ffu;
        gains[b] = h >> 9;

        dot[b] = e1v_decode_at(data.data(), tab, binom.data(), golay.data(), gs.data(),
                               gbit, len, ids[b], gains[b], off, x.data());
    }

    std::fwrite(dot.data(), sizeof(float), nblocks, stdout);
    std::fwrite(ids.data(), sizeof(unsigned), nblocks, stdout);
    std::fwrite(gains.data(), sizeof(unsigned), nblocks, stdout);
    return 0;
}
