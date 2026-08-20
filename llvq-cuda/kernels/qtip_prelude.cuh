// Types the upstream QTIP kernel names, which the NVRTC translation unit does
// not otherwise have. Ours, committed, concatenated BEFORE the fetched device
// half.
//
// ============================================================================
// WHY THIS FILE EXISTS.
//
// NVRTC carries no `<cstdint>`, and this image ships no CUDA `-dev` package, so
// neither `uint32_t` nor `half2` reaches the compiler on its own. The
// repository has already met both halves of this and written them down:
// `awq_gemv.cu:45` records that porting AWQ meant rewriting `uint32_t` to
// `unsigned int` for exactly this reason, and `matvec.cu:25` records that
// `__half2float` is unreachable because the runtime image has no headers.
//
// The AWQ answer — rewrite the names in the source — is not available here:
// QTIP is GPL v3 and is fetched, not vendored, so the less we transform their
// text the better. Declaring the names instead leaves their file untouched.
//
// ⚠️ UNVERIFIED UNTIL A DEVICE COMPILE. Everything below is a reading of what
// the kernel needs, not something a compiler has confirmed. It is precisely
// what job P2 exists to answer, and it is the likeliest thing to be wrong.
// ============================================================================

#pragma once
#define QTIP_PRELUDE_CUH 1

// Redefining a typedef to the SAME type is legal, so these are safe even if a
// future NVRTC injects them. The widths are the LP64 ones the device uses.
typedef unsigned short   uint16_t;
typedef unsigned int     uint32_t;
typedef unsigned long long uint64_t;

// `half2` is used by the kernel as a 32-bit CONTAINER and nothing else: it
// moves them between global, shared and registers, and every arithmetic op
// goes through inline PTX on `uint32_t` registers instead. The only place a
// half is ever read as a number is a commented-out `printf`. A plain 32-bit
// POD therefore has the semantics the kernel actually uses, without needing
// the half intrinsics this image cannot provide.
//
// 🚨 If that reading is wrong — if any path does half arithmetic — this will
// fail to compile rather than compute silently, which is the failure mode to
// prefer.
#ifndef __CUDA_FP16_H__
struct __align__(4) half2 {
    unsigned short x;
    unsigned short y;
};
#endif
