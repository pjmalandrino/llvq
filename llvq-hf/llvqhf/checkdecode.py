"""Gate of stage 2: the Metal op against the numpy decode, every block.

    uv run python -m llvqhf.checkdecode <packed directory>

Both sides are already pinned to `llvq_search::tetra`: the shader by
`llvq-metal/tests/tetra48_matches_rust.rs`, the numpy decode by control 2 of stage
1, 100,000 labels with no difference. So this closes a triangle rather than
trusting one side, over the 118,665,216 blocks of the served 4B.

Integer values: equality, and no tolerance is defined. Exit 1 naming the first
record and block that differ.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from .metal import tetra_decode
from .reader import PackedModel
from .tetra import split_stream


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m llvqhf.checkdecode <packed directory>", file=sys.stderr)
        return 2
    d = Path(argv[1])
    t0 = time.time()
    blocks = 0
    with PackedModel(d) as m:
        names = [n for n, r in m.records.items() if r["kind"] == "tetra"]
        for i, name in enumerate(names):
            r = m.records[name]
            codes = m.tensor(f"{r['prefix']}.codes")
            n = r["d_out"] * r["nblocks"]
            labels, _ = split_stream(codes, n, r["index_bits"], r["gain_bits"])
            want = m.tables.decode(labels).astype(np.int8)
            got = tetra_decode(codes, r["d_out"], r["nblocks"],
                               r["index_bits"], r["gain_bits"]).cpu().numpy()
            if got.shape != want.shape:
                print(f"MISMATCH: {name} shape {got.shape} against {want.shape}", file=sys.stderr)
                return 1
            bad = np.flatnonzero((got != want).any(axis=1))
            if bad.size:
                b = int(bad[0])
                print(f"MISMATCH: {name}, block {b} of {n}, {bad.size} blocks differ\n"
                      f"  numpy {want[b]}\n  metal {got[b]}", file=sys.stderr)
                return 1
            blocks += n
            if i % 24 == 0:
                print(f"  record {i:>3}/{len(names)}  {blocks / 1e6:6.1f} M blocks"
                      f"  {time.time() - t0:6.1f} s", flush=True)
    print(f"{d}")
    print(f"  {len(names)} Tetra records, {blocks} blocks, {blocks * 24} coordinates")
    print(f"  every point identical to the numpy decode, in {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
