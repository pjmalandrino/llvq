"""DRAFT, never run. Written read-only on 2026-11-02 for the persistent-kernel
lead (wave5-deep/persistent.md, section 5, step L1a), after the adversarial
review asked to read the PTX before any card time.

What it settles, at $0 and with no card:
  * the load widths of the staged-activation reads in the served Tetra kernel
    (`ld.shared.v4.f32` against `ld.shared.f32`), which decide whether a
    stride-24 layout caps a persistent Tetra stream near 358 GB/s (24 scalar
    loads, 8-way conflicts) or not at all (6 LDS.128, 2-way);
  * what NVVM made of the three contractible sites (persistent.md 3.7):
    `fma.rn.f32` is final (ptxas never splits it); bare `mul.f32` + `add.f32`
    is left to the DRIVER's ptxas at load time, which only the card settles
    (the cubin dump of step L1, culink_dump.rs).

What it does not settle: the card's SASS. The card JIT-compiles this PTX with
driver 580.178.04's ptxas (a3-gpu.txt), not with the 12.4 devel ptxas.

How to run it (operator, after a go; light, local, no card):
  docker run --rm --platform linux/amd64 -v <repo>:/repo:ro -v <out>:/out python:3.12-slim \
    sh -c "pip install nvidia-cuda-nvrtc-cu12==12.4.127 && python3 -I /out/ptx_probe.py /repo /out"
NVRTC 12.4.127 is the library of nvidia/cuda:12.4.1, the image's base
(ops/Dockerfile.cuda:21, :100). The probe refuses to go on unless its
assembly of the served unit has the digest the served door prints:
101,625 bytes, sha256 6193094310bc18ac... (served-grouped-2026-11-02), a
digest reproduced from the working tree on 2026-11-02 with the recipe below.
"""
import ctypes
import glob
import hashlib
import os
import re
import sys

SERVED_SHA = "6193094310bc18acdb0a0c554807ff45f6df09c1f6ab49e0577032874f107812"
SERVED_DEFINES = "#define TILE_BLOCKS 64u\n#define TETRA48_ROWS 4u\n#define TETRA48_TILE 128u\n"
SERVED_FILES = [  # fused_cuda.rs:493 + fused.rs:548-553 + emb_q4 + int4, joined by "\n" (gpu.rs:177-181)
    "llvq-cuda/kernels/llvq_slot.cuh", "llvq-cuda/kernels/matvec.cu",
    "llvq-cuda/kernels/llvq_rot.cuh", "llvq-cuda/kernels/rotate.cu",
    "llvq-cuda/kernels/llvq_f1rank.cuh", "llvq-cuda/kernels/llvq_f1rank_v3.cuh",
    "llvq-cuda/kernels/llvq_tetra48.cuh", "llvq-llm/kernels/tv_tetra48_h.cu",
    "llvq-llm/kernels/emb_q4.cu", "llvq-llm/kernels/tv_q4_h.cu",
]
# Indicative only: the bench unit carries ~20 more kernels, so its exact text
# differs; a kernel's own PTX is not expected to depend on its neighbours.
BENCH_DEFINES = "#define TILE_BLOCKS 64u\n"
BENCH_FILES = [
    "llvq-cuda/kernels/llvq_slot.cuh", "llvq-cuda/kernels/matvec.cu",
    "llvq-cuda/kernels/llvq_f1rank.cuh", "llvq-cuda/kernels/llvq_f1rank_v3.cuh",
    "llvq-cuda/kernels/llvq_tetra48.cuh", "llvq-cuda/kernels/tetra48_v3g.cu",
    "llvq-cuda/kernels/tetra48_seg.cu",
]
TARGETS = {
    "served": ["tv_tetra48_h", "tv_tetra48_rows_h"],
    "bench": ["tv_f1r_v3g", "tv_tetra48_seg"],
}
OPCODES = [
    "ld.shared.f32", "ld.shared.v2.f32", "ld.shared.v4.f32",
    "fma.rn.f32", "mul.f32", "mul.rn.f32", "add.f32", "add.rn.f32",
    "cvt.rn.f16.f32", "shfl.sync.bfly.b32",
]


def assemble(root, defines, files):
    parts = [defines] + [open(os.path.join(root, f), encoding="utf-8").read() for f in files]
    return "\n".join(parts)


def nvrtc():
    libs = glob.glob(os.path.join(sys.prefix, "lib", "python3*", "site-packages",
                                  "nvidia", "cuda_nvrtc", "lib", "libnvrtc.so.12"))
    if not libs:
        sys.exit("libnvrtc.so.12 not found: pip install nvidia-cuda-nvrtc-cu12==12.4.127")
    return ctypes.CDLL(libs[0])


def compile_ptx(lib, src, name):
    prog = ctypes.c_void_p()
    ok = lib.nvrtcCreateProgram(ctypes.byref(prog), src.encode(), name.encode(), 0, None, None)
    assert ok == 0, f"nvrtcCreateProgram {ok}"
    # The only option cudarc passes (gpu.rs:270-277): NVRTC's defaults do the
    # rest, --fmad=true included.
    opts = (ctypes.c_char_p * 1)(b"--gpu-architecture=compute_89")
    ok = lib.nvrtcCompileProgram(prog, 1, opts)
    size = ctypes.c_size_t()
    lib.nvrtcGetProgramLogSize(prog, ctypes.byref(size))
    log = ctypes.create_string_buffer(size.value)
    lib.nvrtcGetProgramLog(prog, log)
    assert ok == 0, f"NVRTC refused {name}:\n{log.value.decode()}"
    lib.nvrtcGetPTXSize(prog, ctypes.byref(size))
    ptx = ctypes.create_string_buffer(size.value)
    lib.nvrtcGetPTX(prog, ptx)
    return ptx.value.decode()


def entries(ptx):
    out, cur, buf = {}, None, []
    for line in ptx.splitlines():
        m = re.search(r"\.entry\s+(\w+)", line)
        if m:
            if cur:
                out[cur] = "\n".join(buf)
            cur, buf = m.group(1), []
        if cur:
            buf.append(line)
    if cur:
        out[cur] = "\n".join(buf)
    return out


def report(name, body):
    counts = {op: len(re.findall(r"\b" + re.escape(op) + r"\b", body)) for op in OPCODES}
    print(f"  {name}: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    # The epilogue: from the last butterfly shuffle to the f16 store. The tail
    # loop and acc*rscale + tv live here; read it by eye.
    tail = body[body.rfind("shfl.sync.bfly"):]
    print("    epilogue float ops: " + " ".join(re.findall(r"\b(fma\.rn\.f32|mul(?:\.rn)?\.f32|add(?:\.rn)?\.f32)\b", tail)))


def main(root, out):
    lib = nvrtc()
    served = assemble(root, SERVED_DEFINES, SERVED_FILES)
    sha = hashlib.sha256(served.encode()).hexdigest()
    print(f"served unit: {len(served.encode())} bytes, sha256 {sha}")
    if sha != SERVED_SHA:
        sys.exit("refused: the assembly is not the served unit of the digest named above")
    bench = assemble(root, BENCH_DEFINES, BENCH_FILES)
    for tag, src in (("served", served), ("bench", bench)):
        ptx = compile_ptx(lib, src, tag + ".cu")
        with open(os.path.join(out, tag + ".ptx"), "w") as f:
            f.write(ptx)
        ent = entries(ptx)
        print(f"{tag}: {len(ent)} entries")
        for k in TARGETS[tag]:
            report(k, ent[k])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
