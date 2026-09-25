# Compile the served Tetra translation unit to a sm_89 cubin with NVRTC, no card.
#
#   python3 nvrtc_cubin.py <libnvrtc.so.12> <repo root> <arch> <tile> <out dir>
#   e.g. python3 nvrtc_cubin.py wheels/nvrtc/nvidia/cuda_nvrtc/lib/libnvrtc.so.12 . sm_89 64 /tmp/sass
#
# The parts and their order are those of `llvq-llm/src/fused_cuda.rs` for the
# Tetra48 layout: the host defines, then llvq_slot.cuh, matvec.cu, llvq_rot.cuh,
# rotate.cu, then `fused::planes_source_names(Tetra48)`, joined with "\n". The
# emb_q8 and tv_q4_h parts are left out: they are other kernels and nothing of
# them enters tv_tetra48_h. `sm_89` asks NVRTC for SASS; the served path asks
# for `compute_89` PTX and lets the driver JIT it on the card.
#
# Tools used on 2026-09-25: NVRTC 12.4.127 (PyPI wheel nvidia-cuda-nvrtc-cu12),
# cuobjdump and nvdisasm 12.4.127 (conda channel nvidia, label cuda-12.4.1):
#   cuobjdump -res-usage <cubin>
#   PATH=<nvdisasm dir>:$PATH cuobjdump -sass -fun tv_tetra48_h <cubin>
import ctypes, hashlib, sys

lib_path, repo, arch, tile, out = sys.argv[1:6]
lib = ctypes.CDLL(lib_path)
K = repo + "/llvq-cuda/kernels/"
L = repo + "/llvq-llm/kernels/"
files = [K + "llvq_slot.cuh", K + "matvec.cu", K + "llvq_rot.cuh", K + "rotate.cu",
         K + "llvq_f1rank.cuh", K + "llvq_f1rank_v3.cuh", K + "llvq_tetra48.cuh", L + "tv_tetra48_h.cu"]
# TILE_BLOCKS: the served row for sm_89 is 64 (llvq-cuda/src/tile.rs, TILE_BY_SM);
# TETRA48_ROWS / TETRA48_TILE: Prefill::SERVED, 4 rows over 128 blocks.
defines = f"#define TILE_BLOCKS {tile}u\n" + "#define TETRA48_ROWS 4u\n#define TETRA48_TILE 128u\n"
src = "\n".join([defines] + [open(f).read() for f in files]).encode()
print("source bytes", len(src), "sha256", hashlib.sha256(src).hexdigest()[:16])

prog = ctypes.c_void_p()
assert lib.nvrtcCreateProgram(ctypes.byref(prog), src, b"llvq.cu", 0, None, None) == 0
opts = [f"--gpu-architecture={arch}".encode()]
rc = lib.nvrtcCompileProgram(prog, len(opts), (ctypes.c_char_p * len(opts))(*opts))
n = ctypes.c_size_t()
lib.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
log = ctypes.create_string_buffer(n.value)
lib.nvrtcGetProgramLog(prog, log)
if log.value.strip():
    print("LOG:", log.value.decode()[:3000])
assert rc == 0, f"compile failed {rc}"

lib.nvrtcGetCUBINSize(prog, ctypes.byref(n))
buf = ctypes.create_string_buffer(n.value)
lib.nvrtcGetCUBIN(prog, buf)
open(f"{out}/tetra_{arch}_t{tile}.cubin", "wb").write(buf.raw)
print("cubin bytes", n.value)
major, minor = ctypes.c_int(), ctypes.c_int()
lib.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor))
print("nvrtc", major.value, minor.value)
