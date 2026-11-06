"""The gates of the Mac block, `proofs/preregistration-hf-metal-bloc-2026-10-09.md` section 5.

    python -m llvq_tetra.checkmps <model> [--mutant NAME]

Loads the model compressed on Metal and runs G-A, G-D, G-B1, G-B2 and G-C. G-A compares with the
Objective-C++ dispatch of 0.1.0, so it builds that extension, which needs ninja on the PATH; the
other gates do not. Exit status 1 if a gate fails.

`--mutant NAME` edits one shader text in memory before anything compiles, and expects the gate
it targets to fail: `caught` or `SURVIVED`. No file is touched, so a mutant cannot outlive its run.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from . import metal, mps

SEED = 20261009

# The references of G-A read these, saved before any mutant is applied: a reference that moved with
# its subject would compare a mutant with itself.
_TILED_SOURCE = metal.tiled_source
_Q4_SOURCE = metal.q4_source

# name: (gate, function patched, served text, mutated text)
MUTANTS = {
    "single-unfused-epilogue": ("G-A", "tetra", "y[row] = fma(acc, rscale[row], tv);", "y[row] = acc * rscale[row] + tv;"),
    "rows-unfused-epilogue": ("G-D", "rows", "y[r * d_out + row] = fma(a, rscale[row], tv);", "y[r * d_out + row] = a * rscale[row] + tv;"),
    "gather-wrong-group": ("G-B1", "vocab", "uint g = s0 + (c >> 6);", "uint g = s0 + (c >> 5);"),
    "head-reads-half": ("G-B2", "head", "const device float* x ", "const device half* x "),
    "rotation-unsigned": ("G-C", "rot", "? -v : v;", "? v : v;"),
}


def apply_mutant(name: str) -> str:
    gate, where, old, new = MUTANTS[name]
    if where == "tetra":
        base = _TILED_SOURCE

        def patched(tile):
            src = base(tile)
            kernel = mps.kernel_text(src, "tv_tetra48_metal")
            assert kernel.count(old) == 1, f"{name}: the served text matches {kernel.count(old)} times"
            return src.replace(kernel, kernel.replace(old, new))

        metal.tiled_source = patched
    elif where == "rows":
        assert mps.TETRA_ROWS_KERNEL.count(old) == 1
        mps.TETRA_ROWS_KERNEL = mps.TETRA_ROWS_KERNEL.replace(old, new)
    elif where in ("vocab", "head", "rot"):
        file = {"vocab": "emb_q4.metal", "head": "emb_q4.metal", "rot": "llvq_rot.metal"}[where]
        base = metal.shader_source

        def patched(n="llvq_tetra48.metal"):
            src = base(n)
            if n != file:
                return src
            if where == "head":
                # The f32 variant is made by `mps.vocab_library`; undo it after the fact.
                return src
            # The edit stays inside the one function it targets: the served file repeats some
            # lines in its threadgroup variants.
            scope = (mps.helper_text(src, "inline void rot_load(const device ushort* xin,")
                     if where == "rot" else mps.kernel_text(src, "emb_q4_gather_metal"))
            assert scope.count(old) == 1, f"{name}: the served text matches {scope.count(old)} times"
            return src.replace(scope, scope.replace(old, new))

        metal.shader_source = patched
        if where == "head":
            original = mps.edit

            def edit(text, what, edits):
                if what == "tv_emb_q4_metal":
                    edits = [(o, n) for o, n in edits if not (o == old and n == new)]
                    edits = [(o, n) if o != "const device half* x " else (o, o) for o, n in edits]
                return original(text, what, edits)

            mps.edit = edit
    return gate


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    """Per row, the largest difference over the largest reference value; the worst row."""
    a, b = a.double(), b.double()
    num = (a - b).abs().amax(dim=-1)
    den = b.abs().amax(dim=-1).clamp_min(1e-30)
    return float((num / den).max())


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="checkmps")
    ap.add_argument("model")
    ap.add_argument("--mutant", choices=sorted(MUTANTS))
    ap.add_argument("--gates", default="A,D,B1,B2,C")
    a = ap.parse_args(argv[1:])
    target = apply_mutant(a.mutant) if a.mutant else None
    gates = set(a.gates.split(","))

    from transformers import AutoModelForCausalLM

    from .dequant import dequantize_int4
    from .fused import FusedTetraLinear, Int4FusedLinear, Rotation
    from .vocab import Int4TiedHead, Int4Vocab

    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32, device_map="mps")
    gen = torch.Generator().manual_seed(SEED)
    verdict = {}

    tetra = [(n, m) for n, m in model.named_modules() if isinstance(m, FusedTetraLinear)]
    q4 = [(n, m) for n, m in model.named_modules() if isinstance(m, Int4FusedLinear)]
    print(f"{len(tetra)} Tetra records, {len(q4)} int4 records")

    if "A" in gates:
        metal._extension()  # the 0.1.0 dispatch, the reference of G-A
        tables, invnorm = metal.shared_tables("mps"), metal.invnorm_on("mps")
        bad = 0
        for name, m in tetra:
            d = m.desc
            x = torch.randn(1, d["d_in"], generator=gen).to("mps")
            ref = torch.ops.llvq.tv_tetra48(
                m._words, *tables, m._gscale, invnorm, m._rscale, m._tail, x[0],
                d["d_out"], d["nblocks"], d["tail_cols"], metal.stride_u32(d["nblocks"]), m._tile,
                _TILED_SOURCE(m._tile),
            )
            got = mps.tetra_matvec(m._words, m._gscale, m._rscale, m._tail, x, d, m._tile, rows=False)[0]
            bad += not torch.equal(ref.cpu(), got.cpu())
        for name, m in q4:
            d = m.desc
            x = torch.randn(1, d["d_in"], generator=gen).to("mps")
            ref = torch.ops.llvq.tv_q4(m._wq, m._scales, m._biases, x[0], d["d_out"], d["d_in"],
                                       d["groups_per_row"], m._tile, _Q4_SOURCE())
            got = mps.q4_matvec(m._wq, m._scales, m._biases, x, d, m._tile, rows=False)[0]
            bad += not torch.equal(ref.cpu(), got.cpu())
        verdict["G-A"] = bad == 0
        print(f"G-A  {len(tetra) + len(q4) - bad} of {len(tetra) + len(q4)} records bit-identical to the 0.1.0 dispatch")

    if "D" in gates:
        bad = 0
        n_rows = mps.ROWS + 1  # one full dispatch and one partial
        for name, m in tetra:
            d = m.desc
            x = torch.randn(n_rows, d["d_in"], generator=gen).to("mps")
            one = mps.tetra_matvec(m._words, m._gscale, m._rscale, m._tail, x, d, m._tile, rows=False)
            many = mps.tetra_matvec(m._words, m._gscale, m._rscale, m._tail, x, d, m._tile, rows=True)
            bad += not torch.equal(one.cpu(), many.cpu())
        for name, m in q4:
            d = m.desc
            x = torch.randn(n_rows, d["d_in"], generator=gen).to("mps")
            one = mps.q4_matvec(m._wq, m._scales, m._biases, x, d, m._tile, rows=False)
            many = mps.q4_matvec(m._wq, m._scales, m._biases, x, d, m._tile, rows=True)
            bad += not torch.equal(one.cpu(), many.cpu())
        verdict["G-D"] = bad == 0
        print(f"G-D  {len(tetra) + len(q4) - bad} of {len(tetra) + len(q4)} records: {n_rows} rows in one go equal row by row, bit for bit")

    vocab = next((m for m in model.modules() if isinstance(m, Int4Vocab)), None)
    head = next((m for m in model.modules() if isinstance(m, Int4TiedHead)), None)
    if vocab is not None and ({"B1", "B2"} & gates):
        rows, d = vocab.num_embeddings, vocab.embedding_dim
        q, s, b = (t.cpu().numpy() for t in (vocab.qweight, vocab.scales, vocab.biases))
        dense = torch.from_numpy(dequantize_int4(q, s, b, rows, d, vocab.desc["group"]).reshape(rows, d))
        if "B1" in gates:
            bad = 0
            for lo in range(0, rows, 16384):
                ids = torch.arange(lo, min(lo + 16384, rows))
                bad += int((vocab(ids.to("mps")).cpu() != dense[lo : lo + ids.numel()]).any(dim=-1).sum())
            verdict["G-B1"] = bad == 0
            print(f"G-B1 {rows - bad} of {rows} vocabulary rows bit-identical to the dense dequantization")
        if "B2" in gates and head is not None:
            h = torch.randn(64, d, generator=gen)
            got = head(h.to("mps")).cpu()
            ref = h @ dense.T
            err = rel_err(got, ref)
            verdict["G-B2"] = err < 1e-4
            print(f"G-B2 output layer against an f32 dense matmul: worst row {err:.2e} (bar 1e-4)")

    if "C" in gates:
        worst, n = 0.0, 0
        seen = set()
        for name, m in tetra:
            rot = m.rotation
            if rot is None or id(rot) in seen:
                continue
            seen.add(id(rot))
            # The torch rotation of 0.1.0, rebuilt from the kernel's own tables.
            signs = np.where(
                (rot.signbits.cpu().numpy()[np.arange(rot.n) >> 5] >> (np.arange(rot.n) & 31)) & 1, -1.0, 1.0
            )
            small = rot.small.cpu().numpy().reshape(mps.ROT_KMAX, mps.ROT_KMAX)[: rot.k, : rot.k]
            ref_rot = Rotation(signs, small, "mps")
            x = torch.randn(8, rot.n, generator=gen).to("mps")
            worst = max(worst, rel_err(rot(x).cpu(), ref_rot(x).cpu()))
            n += 1
        verdict["G-C"] = worst < 1e-5
        print(f"G-C  {n} rotations against the torch rotation: worst row {worst:.2e} (bar 1e-5)")

    failed = [g for g, ok in verdict.items() if not ok]
    if target:
        status = "caught" if target in failed else "SURVIVED"
        print(f"mutant {a.mutant} on {target}: {status}")
        return 0 if status == "caught" else 1
    print("all gates green" if not failed else f"FAILED: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
