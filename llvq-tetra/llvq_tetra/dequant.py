"""Rebuilding a matrix exactly as `llvq_artifact::decode_matrix` does.

Every step is reproduced operation by operation rather than handed to a library,
because the claim of stage 1 is bit-exactness and not closeness:

* the shape-gain reconstruction is one multiply, one division and one multiply
  per coordinate (`llvq_quant::quantizer::reconstruct_shape_gain`);
* the Walsh-Hadamard transform is a butterfly of single additions in a fixed
  pairing order, then one global scale;
* the `k` by `k` mix accumulates over `t` from zero, in order. A matrix product
  would be free to reassociate that sum and change the last bits.

The order of the whole is the quantizer's own: decode in the rotated basis,
restore the tail, un-rotate, and only then narrow to f32. Narrowing earlier, or
un-rotating in f32, moves the last bits, and the format's claim is that it does
not.

The int4 arm is the opposite discipline: `w = scale·q + bias` is computed in
**f32**, because that is what the Rust reader does
(`llvq_artifact::Int4Matrix::to_f32`). Doing it in f64 and narrowing afterwards
rounds differently.
"""

from __future__ import annotations

import numpy as np


def reconstruct_blocks(points: np.ndarray, gains: np.ndarray, centroids: np.ndarray,
                       row_scales: np.ndarray, d_out: int, nblocks: int) -> np.ndarray:
    """`(d_out, nblocks·24)` f64, the rotated basis.

    `points` is `(d_out·nblocks, 24)` int32, row-major by output row then block,
    which is the order the code stream stores.
    """
    dim = points.shape[1]
    n2 = (points.astype(np.int64) ** 2).sum(axis=1)
    m = n2 // 16
    # `Leech::shell_index` is None unless the norm is a multiple of 16, and the
    # origin block reconstructs as zero rather than through the scale.
    live = (n2 % 16 == 0) & (m > 0)
    picked = centroids[gains] * np.repeat(row_scales, nblocks)
    scale = np.zeros(points.shape[0], dtype=np.float64)
    np.divide(picked, np.sqrt((16 * m).astype(np.float64)), out=scale, where=live)
    w = points.astype(np.float64) * scale[:, None]
    w[~live] = 0.0
    return w.reshape(d_out, nblocks * dim)


def wht_rows(v: np.ndarray) -> None:
    """In-place scaled Walsh-Hadamard transform along the last axis.

    The pairing order of `llvq_quant::rotation::wht`: `len` doubles from 1, and
    within a stage every output is one addition or one subtraction of the same
    two f64 values Rust adds. Vectorizing a stage changes no single operation.
    """
    n = v.shape[-1]
    if n & (n - 1):
        raise ValueError(f"the Walsh-Hadamard transform needs a power of two, got {n}")
    length = 1
    while length < n:
        x = v.reshape(*v.shape[:-1], n // (2 * length), 2, length)
        a = x[..., 0, :].copy()
        b = x[..., 1, :].copy()
        x[..., 0, :] = a + b
        x[..., 1, :] = a - b
        length <<= 1
    v *= 1.0 / np.sqrt(float(n))


def unrotate_rows(w: np.ndarray, signs: np.ndarray, small: np.ndarray) -> np.ndarray:
    """`W ← W Q`, the inverse of the quantizer's `W ← W Qᵀ`.

    `Rotation::apply_transpose`: the mix by `Qᵀ`, then the transform per group,
    then the sign flip. `Q = (Q_odd ⊗ H_m) D` with `m` the largest power of two
    dividing `n`, so the groups are the `k = n / m` slices of a row.
    """
    n = w.shape[1]
    if signs.shape != (n,):
        raise ValueError(f"{signs.shape[0]} signs for {n} columns")
    m = n & (-n)  # the largest power of two dividing n
    k = n // m
    if small.shape != (k, k):
        raise ValueError(f"the mix is {small.shape}, want {k} by {k}")
    if k > 1:
        v = w.reshape(w.shape[0], k, m)
        acc = np.zeros_like(v)
        for g in range(k):
            for t in range(k):
                # Row-major `small`: Qᵀ[g][t] = Q[t][g], accumulated over t from
                # zero, which is the order Rust's inner loop takes.
                acc[:, g, :] += small[t, g] * v[:, t, :]
        w = acc.reshape(w.shape[0], n)
    else:
        w = np.ascontiguousarray(w)
    view = w.reshape(w.shape[0], k, m)
    wht_rows(view)
    w = view.reshape(w.shape[0], n)
    w *= signs
    return w


def dequantize_lattice(points: np.ndarray, gains: np.ndarray, centroids: np.ndarray,
                       row_scales: np.ndarray, tail: np.ndarray | None, d_out: int,
                       d_in: int, rotation: tuple[np.ndarray, np.ndarray] | None,
                       dim: int = 24) -> np.ndarray:
    """The whole chain, `(d_out, d_in)` f32 in the natural basis."""
    nblocks = d_in // dim
    tail_cols = d_in % dim
    w = np.zeros((d_out, d_in), dtype=np.float64)
    if nblocks:
        w[:, : nblocks * dim] = reconstruct_blocks(
            points, gains, centroids, row_scales, d_out, nblocks
        )
    if tail_cols:
        if tail is None or tail.shape != (d_out, tail_cols):
            raise ValueError(f"a {d_out} by {tail_cols} tail is missing")
        # The stream stores the tail in f32 and the decoder widens it, exactly.
        w[:, nblocks * dim :] = tail.astype(np.float64)
    elif tail is not None and tail.size:
        raise ValueError("a tail on a record whose d_in is a multiple of 24")
    if rotation is not None:
        w = unrotate_rows(w, *rotation)
    return w.astype(np.float32)


def dequantize_int4(packed: np.ndarray, scales: np.ndarray, biases: np.ndarray,
                    d_out: int, d_in: int, group: int) -> np.ndarray:
    """`w = scale·q + bias`, in f32, groups along the innermost dimension.

    Nibbles are stored **low first**, which is the opposite convention from the
    lattice stream's MSB-first packing. Two orders in one file, and a reader that
    assumed one for both would produce plausible, wrong weights.
    """
    if d_in % 2:
        raise ValueError(f"d_in {d_in} is odd, so its nibbles do not split by row")
    gpr = -(-d_in // group)
    if packed.shape != (d_out, d_in // 2):
        raise ValueError(f"packed is {packed.shape}, want {(d_out, d_in // 2)}")
    if scales.shape != (d_out, gpr) or biases.shape != (d_out, gpr):
        raise ValueError(f"scales {scales.shape} and biases {biases.shape} for {gpr} groups")
    q = np.empty((d_out, d_in), dtype=np.uint8)
    q[:, 0::2] = packed & 0x0F
    q[:, 1::2] = packed >> 4
    # One f16 pair per group, widened to f32: the multiply and the add are f32,
    # as they are in `Int4Matrix::to_f32`.
    s = np.repeat(scales.astype(np.float32), group, axis=1)[:, :d_in]
    b = np.repeat(biases.astype(np.float32), group, axis=1)[:, :d_in]
    return s * q.astype(np.float32) + b
