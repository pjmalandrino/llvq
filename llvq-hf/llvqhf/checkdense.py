"""Gate A of stage 1: the dequantized weights against the Rust decoder.

    uv run python -m llvqhf.checkdense <packed directory>

`bin/hfdense` writes one SHA-256 per record of the f32 values
`llvq_artifact::decode_matrix` produces. This rebuilds every one of them from the
safetensors and compares. Exactness is the point: a tolerance here would mean the
weights a reader outside Rust gets are not the weights the artifact defines.

Exit 0 and one line per section when all 253 agree. Exit 1 naming the first that
does not, with the two hashes.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from .reader import PackedModel

DENSE_DIGEST_FILE = "llvq-dense-digest.json"


def sha_f32(a: np.ndarray) -> str:
    if a.dtype != np.dtype("<f4"):
        raise ValueError(f"dtype {a.dtype} where little-endian f32 was expected")
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[2].strip(), file=sys.stderr)
        return 2
    d = Path(argv[1])
    want = json.loads((d / DENSE_DIGEST_FILE).read_text())
    bad = 0
    t0 = time.time()
    with PackedModel(d) as m:
        if want["artifact"]["sha256"] != m.qc["artifact_sha256"]:
            print("MISMATCH: the dense digest and config.json name two different artifacts",
                  file=sys.stderr)
            return 1
        names = list(m.records)
        if set(names) != set(want["records"]):
            missing = sorted(set(want["records"]) - set(names))
            extra = sorted(set(names) - set(want["records"]))
            print(f"MISMATCH: records differ, missing {missing[:3]}, extra {extra[:3]}",
                  file=sys.stderr)
            return 1
        for i, name in enumerate(names):
            got = sha_f32(m.dequantize(name))
            if got != want["records"][name]:
                print(f"MISMATCH: {name}\n  rust   {want['records'][name]}\n  python {got}",
                      file=sys.stderr)
                bad += 1
                if bad == 1:
                    return 1
            if i % 36 == 0:
                print(f"  record {i:>3}/{len(names)}  {time.time() - t0:6.1f} s", flush=True)
        for name in want["raw"]:
            got = sha_f32(m.dequantize_raw(name))
            if got != want["raw"][name]:
                print(f"MISMATCH: {name}\n  rust   {want['raw'][name]}\n  python {got}",
                      file=sys.stderr)
                return 1
        total = len(names) + len(want["raw"])
    print(f"{d}")
    print(f"  {len(names)} records and {len(want['raw'])} carried tensors dequantized")
    print(f"  {total} digests identical to llvq_artifact::decode_matrix, in {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
