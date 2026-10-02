"""Gate of M2: the tiled int4 kernel against the served one, bit for bit.

    uv run python -m llvq_tetra.checkq4 <packed directory>

`tv_q4_metal` stages the whole activation and runs only for `d_in` up to 8,192;
`tv_q4_metal_tiled` stages slices and runs for any width. Where both run they must
give the **same f32**, not a close one: a tile of a multiple of 256 columns is a
multiple of 32 words, so each lane walks the same words in the same order, and that
is a claim about indices rather than an approximation.

For the records the served kernel cannot take, the tiled one is checked against the
dense reconstruction instead, at 1e-2 relative.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from . import metal
from .reader import PackedModel

TILES = (256, 2048, 8192)
LIMIT = 8192


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m llvq_tetra.checkq4 <packed directory>", file=sys.stderr)
        return 2
    metal.extension_for("mps")
    src = metal.q4_source()
    rng = np.random.default_rng(0x4)
    d = Path(argv[1])
    compared = dense_checked = 0
    with PackedModel(d) as m:
        int4 = [(n, r) for n, r in m.records.items() if r["kind"] == "int4g128"]
        shapes = {}
        for n, r in int4:
            shapes.setdefault((r["d_out"], r["d_in"]), n)
        print(f"{len(int4)} int4 records, {len(shapes)} distinct shapes")
        for (d_out, d_in), name in sorted(shapes.items()):
            r = m.records[name]
            p = r["prefix"]
            blobs = [torch.from_numpy(
                np.ascontiguousarray(m.tensor(f"{p}.{f}")).view(np.uint8).reshape(-1).copy()
            ).to("mps") for f in ("qweight", "scales", "biases")]
            x = torch.from_numpy(rng.standard_normal(d_in).astype(np.float32)).to("mps")
            args = (*blobs, x, d_out, d_in, r["groups_per_row"])
            if d_in <= LIMIT:
                served = torch.ops.llvq.tv_q4(*args, 0, src).cpu().numpy()
                for tile in TILES:
                    # A tile wider than `d_in` is legal: the kernel takes one pass
                    # with a short tile, and the prereg names all three.
                    got = torch.ops.llvq.tv_q4(*args, tile, src).cpu().numpy()
                    if not np.array_equal(served, got):
                        bad = int(np.flatnonzero(served != got)[0])
                        print(f"MISMATCH: {name} at tile {tile}, row {bad}: "
                              f"served {served[bad]!r} tiled {got[bad]!r}", file=sys.stderr)
                        return 1
                    compared += 1
                print(f"  {name}  {d_out} by {d_in}: identical to the served kernel at "
                      f"{', '.join(str(t) for t in TILES)}")
            else:
                # Past the served kernel's wall, so the reference is the dense one.
                w = m.dequantize(name).astype(np.float64)
                y_ref = w @ x.cpu().numpy().astype(np.float64)
                got = torch.ops.llvq.tv_q4(*args, 2048, src).cpu().numpy().astype(np.float64)
                rel = np.abs(got - y_ref).max() / np.abs(y_ref).max()
                dense_checked += 1
                ok = rel < 1e-2
                print(f"  {name}  {d_out} by {d_in}: past the served wall, against the dense "
                      f"reconstruction {rel:.2e} relative, {'passes' if ok else 'FAILS'}")
                if not ok:
                    return 1
    print(f"  {compared} comparisons identical, {dense_checked} shapes checked against the dense "
          f"reconstruction")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
