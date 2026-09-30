"""The guards of the tiled int4 kernel: its refusals, and its gate under mutation.

    uv run python -m llvqhf.checkq4guards <packed directory>

`checkq4` says the tiled kernel agrees with the served one. This says the
agreement is informative: that the gate fails when the kernel is wrong, and that
the illegal calls are refused by name rather than run.

Two families. The refusals, controls 1 and 2 of the M2 prereg: a tile that is not
a multiple of 256, a `d_in` that is not a multiple of 8, and a staging past
32,768 bytes. The mutants, control 4: the group index shifted by one, the nibble
order reversed, and the lane stride moved off 32.

The mutation is applied to `tv_q4_metal_tiled` only, never to the served
`tv_q4_metal` the comparison uses as its reference, or both sides would move
together and the mutant would be void.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from llvqhf import metal
from llvqhf.reader import PackedModel

TILES = (256, 2048, 8192)
SPLIT = "kernel void tv_q4_metal_tiled"


def mutate(src: str, old: str, new: str, want: int) -> str:
    """Rewrite `old` inside the tiled kernel only, asserting the site count.

    `tv_q4_metal_tiled` comes FIRST in the file and the served `tv_q4_metal`
    after it, so the region has to be closed at the next entry point. Without
    that the mutation lands on both kernels, the reference moves with the
    subject, and the mutant reads as survived while testing nothing.
    """
    head, _, rest = src.partition(SPLIT)
    assert rest, "the tiled entry point is not in the shader"
    cut = rest.index("\nkernel void")
    body, foot = rest[:cut], rest[cut:]
    assert "tv_q4_metal(" not in body, "the region still contains the served kernel"
    n = body.count(old)
    assert n == want, f"{n} sites for {old!r} in the tiled kernel, not {want}"
    return head + SPLIT + body.replace(old, new) + foot


def main(argv: list[str]) -> int:
    metal.extension_for("mps")
    src = metal.q4_source()
    rng = np.random.default_rng(0x4)
    with PackedModel(Path(argv[1])) as m:
        name = next(n for n, r in m.records.items()
                    if r["kind"] == "int4g128" and r["d_in"] <= 8192)
        r = m.records[name]
        p, d_out, d_in = r["prefix"], r["d_out"], r["d_in"]
        blobs = [torch.from_numpy(
            np.ascontiguousarray(m.tensor(f"{p}.{f}")).view(np.uint8).reshape(-1).copy()
        ).to("mps") for f in ("qweight", "scales", "biases")]
        x = torch.from_numpy(rng.standard_normal(d_in).astype(np.float32)).to("mps")
        args = (*blobs, x, d_out, d_in, r["groups_per_row"])
        print(f"subject {name}  {d_out} by {d_in}, groups per row {r['groups_per_row']}")

        served = torch.ops.llvq.tv_q4(*args, 0, src).cpu().numpy()

        print("\n== the refusals (controls 1 and 2) ==")
        probes = [
            ("a tile of 128, not a multiple of 256", (*args, 128, src), "256"),
            ("a tile of 300, not a multiple of 256", (*args, 300, src), "256"),
            ("d_in 2555, not a multiple of 8",
             (*blobs, x, d_out, 2555, r["groups_per_row"], 2048, src), "multiple of 8"),
            ("a tile of 16384, staging past 32768 bytes",
             (*args, 16384, src), "32768"),
        ]
        for what, a, needle in probes:
            try:
                torch.ops.llvq.tv_q4(*a)
            except Exception as e:  # noqa: BLE001 - the message is the evidence
                first = str(e).strip().splitlines()[0]
                ok = needle in str(e)
                print(f"  {'REFUSED' if ok else 'wrong message'}: {what}\n"
                      f"      {first[:140]}")
                if not ok:
                    return 1
            else:
                print(f"  ACCEPTED, and should not have been: {what}", file=sys.stderr)
                return 1

        print("\n== the mutants (control 4) ==")
        mutants = [
            ("the group index shifted by one",
             mutate(src, "uint g = s0 + c / LLVQ_Q4_GROUP;",
                    "uint g = s0 + c / LLVQ_Q4_GROUP + 1u;", 1)),
            ("the nibble order reversed",
             mutate(src, "(p >> (4u * k))", "(p >> (4u * (7u - k)))", 1)),
            ("the tile stride read as 32 words instead of the lane's stride",
             mutate(src, "wi += 32u", "wi += 31u", 1)),
        ]
        for what, msrc in mutants:
            caught = []
            for tile in TILES:
                got = torch.ops.llvq.tv_q4(*args, tile, msrc).cpu().numpy()
                if not np.array_equal(served, got):
                    bad = int(np.flatnonzero(served != got)[0])
                    caught.append(f"tile {tile} row {bad} {served[bad]:.6g} vs {got[bad]:.6g}")
            if not caught:
                print(f"  SURVIVED: {what}", file=sys.stderr)
                return 1
            print(f"  caught at {len(caught)} of {len(TILES)} tiles: {what}\n"
                  f"      {caught[0]}")

        print("\nevery refusal fired and every mutant was caught")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
