"""Shared knobs of the Inkling 2 Mini gate. Import this module FIRST, before numpy or torch.

`INKLING_THREADS` (default 4) sets the BLAS/OpenMP thread count through the environment, which BLAS reads at
load time, then `torch.set_num_threads` is applied by `apply_torch()` once torch is imported.

Paths: `INKLING_DATA` (default ~/llvq-inkling-mini) is the data directory D.
"""

import os

THREADS = int(os.environ.get("INKLING_THREADS", "4"))
for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_v] = str(THREADS)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

DATA = os.path.expanduser(os.environ.get("INKLING_DATA", "~/llvq-inkling-mini"))
NATIVE = os.path.join(DATA, "native")
NATIVE_FILE = os.path.join(NATIVE, "model-00001-of-00001.safetensors")
TEXTONLY = os.path.join(DATA, "textonly")

NUM_EXPERTS = 128
HIDDEN = 2304
INTER = 1152
TOPK = 8


def apply_torch():
    import torch

    torch.set_num_threads(THREADS)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return torch
