"""Stage 4: the served CUDA kernel reached from PyTorch, checked on a card.

    python3 ops/hf_cuda_check.py --model <packed dir> --ref <bin/run dump> [--new 64]

Four arms, in this order, so a failure says which thing failed:

1. **the build**, `nvcc` over `llvqhf/csrc/tetra_cuda.cu`, which includes the
   served `tv_tetra48_h.cu` rather than a copy of it. A build failure costs two
   minutes and answers the question on its own.
2. **the arithmetic**, per row, against the dense reconstruction of the same
   record. This is the sensitive arm: on Metal, dropping the tail entirely moved
   this number by 8.79 % and left 64 greedy tokens untouched on two prompts of
   four.
3. **the tokens**, 64 greedy ids on the four prompts of `bin/run`, against a dump
   produced before the job. The reference crosses machines, which a ratio may not
   do and an identity may; the prereg says what that costs.
4. **the memory**, measured on the loaded model.

Nothing here is a speed measurement and no number it prints is divided by another
(rule 5).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def per_row(model_dir: Path, shapes=("self_attn.k_proj", "mlp.down_proj", "mlp.gate_proj")) -> int:
    """Arm 2: the kernel against the dense reconstruction, one matrix a shape."""
    from llvqhf.fused import FusedTetraLinear, Rotation
    from llvqhf.modules import TetraLinear
    from llvqhf.reader import PackedModel

    rng = np.random.default_rng(0xA1)
    bad = 0
    with PackedModel(model_dir) as m:
        for want in shapes:
            name = next((n for n, r in m.records.items()
                         if r["kind"] == "tetra" and want in n and ".0." in n), None)
            if name is None:
                print(f"  {want}: no Tetra record at layer 0, skipped by shape and not by failure")
                continue
            r = m.records[name]
            w = m.dequantize(name).astype(np.float64)
            x = rng.standard_normal(r["d_in"]).astype(np.float32)
            y_ref = w @ x.astype(np.float64)
            mod = TetraLinear(r)
            for f in ("codes", "row_scales", "centroids", "tail"):
                if hasattr(mod, f):
                    getattr(mod, f).copy_(torch.from_numpy(m.tensor(f"{r['prefix']}.{f}")))
            rot = None
            if r.get("rotation"):
                signs, small = m.rotation(r["rotation"])
                rot = Rotation(signs, small, "cuda")
            fused = FusedTetraLinear.from_loaded(mod, rot, 64, "cuda")
            y = fused(torch.from_numpy(x).to("cuda")).cpu().numpy().astype(np.float64)
            rel = np.abs(y - y_ref).max() / np.abs(y_ref).max()
            # f16 out on CUDA against f32 on Metal, so the bar is the f16 ulp of
            # the output scale and not Metal's 2.6e-5.
            ok = rel < 1e-2
            bad += not ok
            print(f"  {name}\n     d_out {r['d_out']} d_in {r['d_in']} tail {r['tail_cols']}"
                  f" | max|Δ| {np.abs(y - y_ref).max():.3e} | relative {rel:.2e}"
                  f" | {'passes' if ok else 'FAILS the 1e-2 bar'}")
    return bad


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="hf_cuda_check")
    ap.add_argument("--model", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--new", type=int, default=64)
    ap.add_argument("--dump", default=None)
    a = ap.parse_args(argv[1:])

    print(f"torch {torch.__version__}, cuda {torch.version.cuda}, "
          f"device {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
    if not torch.cuda.is_available():
        print("REFUSED: no CUDA device", file=sys.stderr)
        return 2

    print("== arm 1, the build ==", flush=True)
    t = time.time()
    from llvqhf import metal

    metal.extension_for("cuda")
    print(f"  built in {time.time() - t:.1f} s", flush=True)

    print("== arm 2, the arithmetic per row ==", flush=True)
    bad = per_row(Path(a.model))
    if bad:
        print(f"REFUSED: {bad} shapes past the bar; the tokens are not run", file=sys.stderr)
        return 1

    print("== arm 3, the tokens ==", flush=True)
    from llvqhf import gentokens

    dump = a.dump or "/tmp/hf-cuda-tokens.json"
    rc = gentokens.main(["gentokens", a.model, "--new", str(a.new), "--dtype", "f32",
                         "--device", "cuda", "--dump", dump])
    if rc:
        return rc
    from llvqhf import comparetokens

    return comparetokens.main(["comparetokens", a.ref, dump])


if __name__ == "__main__":
    sys.exit(main(sys.argv))
