"""The device buffers a layer serves from, built from the tensors the file holds.

Three transforms, each held on a Mac against the numpy reference of `llvq_tetra`:

* `tetra48_words` turns a record's disk stream into the served `tetra48` layout,
  on the device, so a 1.4 GB load never goes back through the host. The rule is
  `llvq_tetra.metal.to_tetra48`'s: the gain bit moves from the bottom of the
  big-endian disk value to bit 47 of the little-endian word.
* `rotation_buffers` builds what `rot_apply_rows` reads: one sign bit a
  coordinate, `Q_odd` zero padded to 32 by 32 in f32, `1/√m` narrowed from f64.
  `llvq_llm::fused::RotationTables::build` is the specification.
* `tetra_dequant` rebuilds `W'`, the weight in the rotated basis, from the served
  words. It is the dequantize half of the M > 1 path, written in torch so the
  same code runs on a CPU in the tests and on the card. Its arithmetic is
  `tetra48_decode_f`'s, `val · (gscale[g] · invnorm[m])`, then the row scale.

The decode is integer and gated exactly against `TetraTables.decode`. The
scaling is f32 and is not the kernel's association, so the dense path is held to
a tolerance against the f64 reconstruction, never to bit identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

from llvq_tetra.fused import invnorm_table
from llvq_tetra.metal import stride_u32
from llvq_tetra.tetra import TetraTables

ROT_KMAX = 32
WORD_BYTES = 6
LABEL_MASK = (1 << 47) - 1


def tetra48_words(codes: torch.Tensor, d_out: int, nblocks: int) -> torch.Tensor:
    """The served `tetra48` stream, `d_out · stride_u32 · 4` bytes, from the disk bytes.

    `codes` is the record's `codes` tensor, six bytes a block, MSB first: read
    big-endian they give `label << 1 | gain`. The served word is `label | gain
    << 47`, little-endian, at byte `6 j` of a row padded to `stride_u32` words.
    """
    n = d_out * nblocks
    if codes.dtype != torch.uint8 or codes.numel() != n * WORD_BYTES:
        raise ValueError(f"{codes.numel()} code bytes of {codes.dtype} for {n} blocks of six")
    b = codes.reshape(n, WORD_BYTES).to(torch.int64)
    value = torch.zeros(n, dtype=torch.int64, device=codes.device)
    for i in range(WORD_BYTES):  # most significant byte first
        value = (value << 8) | b[:, i]
    word = (value >> 1) | ((value & 1) << 47)
    little = torch.stack([(word >> (8 * i)) & 0xFF for i in range(WORD_BYTES)], dim=1)
    stride = stride_u32(nblocks)
    out = torch.zeros(d_out, stride * 4, dtype=torch.uint8, device=codes.device)
    out[:, : nblocks * WORD_BYTES] = little.to(torch.uint8).reshape(d_out, nblocks * WORD_BYTES)
    return out.reshape(-1)


def words_to_labels(words: torch.Tensor, d_out: int, nblocks: int):
    """`(labels, gains)`, int64, out of a served stream: the inverse of the layout."""
    stride = stride_u32(nblocks)
    if words.numel() < d_out * stride * 4:
        raise ValueError(f"{words.numel()} bytes for {d_out} rows of {stride} words")
    rows = words[: d_out * stride * 4].reshape(d_out, stride * 4)[:, : nblocks * WORD_BYTES]
    b = rows.reshape(d_out * nblocks, WORD_BYTES).to(torch.int64)
    word = torch.zeros(b.shape[0], dtype=torch.int64, device=words.device)
    for i in range(WORD_BYTES):  # little-endian
        word = word | (b[:, i] << (8 * i))
    return word & LABEL_MASK, word >> 47


@dataclass(frozen=True)
class RotationBuffers:
    """What `rot_apply_rows` reads for one rotation, on one device."""

    key: str
    signbits: torch.Tensor  # int32 [ceil(n / 32)], read as u32
    small: torch.Tensor  # f32 [ROT_KMAX · ROT_KMAX], Q_odd zero padded, row-major
    n: int
    m: int
    k: int
    inv: float

    @property
    def threads(self) -> int:
        """One block a row, the widest the driver allows, at least a warp."""
        return min(max(1 << (self.n - 1).bit_length(), 32), 1024)


def rotation_buffers(key: str, signs: np.ndarray, small: np.ndarray, device) -> RotationBuffers:
    """`RotationTables::build` from the file's two tables instead of the seed.

    The file carries `signs` and `small` in f64, and stage 0 rebuilt them bit for
    bit against the Rust rotation, so this is the same object without a second
    derivation of the generator.
    """
    signs = np.asarray(signs, dtype=np.float64).reshape(-1)
    n = signs.shape[0]
    m = n & (-n)
    k = n // m
    small = np.asarray(small, dtype=np.float64)
    if small.shape != (k, k):
        raise ValueError(f"rotation {key}: the mix is {small.shape}, want {k} by {k}")
    if k > ROT_KMAX:
        raise ValueError(f"rotation {key}: odd factor {k} above KMAX={ROT_KMAX}, the rotation "
                         "kernel does not handle it (docs/format-noyau.md §8)")
    bits = np.zeros(-(-n // 32), dtype=np.uint32)
    neg = np.flatnonzero(signs < 0)
    np.bitwise_or.at(bits, neg >> 5, np.left_shift(np.uint32(1), (neg & 31).astype(np.uint32)))
    padded = np.zeros((ROT_KMAX, ROT_KMAX), dtype=np.float32)
    padded[:k, :k] = small.astype(np.float32)
    return RotationBuffers(
        key=key,
        signbits=torch.from_numpy(bits.view(np.int32).copy()).to(device),
        small=torch.from_numpy(padded.reshape(-1)).to(device),
        n=n,
        m=m,
        k=k,
        inv=float(np.float32(1.0 / np.sqrt(np.float64(m)))),
    )


def rotate_reference(x: torch.Tensor, rb: RotationBuffers) -> torch.Tensor:
    """`rot_apply_rows` in torch: `[M, n]` in any float, `[M, n]` f32 out.

    The kernel's three phases in order: widen and flip by the sign bits, the
    unscaled butterflies over each group of `m`, then the scale alone when
    `k = 1` or the mix of the groups by `Q_odd` with the scale folded in. The
    mix's sum is a matrix product here and an ordered FMA chain on the card, so
    the two agree to f32 rounding and not to the bit.
    """
    M, n = x.shape
    if n != rb.n:
        raise ValueError(f"rotation {rb.key} is {rb.n} wide, the activation {n}")
    idx = torch.arange(n, device=x.device)
    bits = rb.signbits.to(x.device).to(torch.int64) & 0xFFFFFFFF
    neg = ((bits[idx >> 5] >> (idx & 31)) & 1).bool()
    s = torch.where(neg, -x.to(torch.float32), x.to(torch.float32))
    v = s.reshape(M, rb.k, rb.m)
    length = 1
    while length < rb.m:
        w = v.reshape(M, rb.k, rb.m // (2 * length), 2, length)
        a, b = w[..., 0, :], w[..., 1, :]
        v = torch.stack((a + b, a - b), dim=-2).reshape(M, rb.k, rb.m)
        length <<= 1
    col = v * rb.inv
    if rb.k == 1:
        return col.reshape(M, n)
    small = rb.small.to(x.device).reshape(ROT_KMAX, ROT_KMAX)[: rb.k, : rb.k]
    return torch.einsum("gt,btm->bgm", small, col).reshape(M, n)


@dataclass(frozen=True)
class DecodeTables:
    """`TetraTables` as int64 tensors on one device, for the torch decode."""

    order_inv: torch.Tensor
    prefixes: torch.Tensor
    suffixes: torch.Tensor
    branch_c2: torch.Tensor
    branch_s16: torch.Tensor
    rows: torch.Tensor
    values: torch.Tensor
    invnorm: torch.Tensor
    class_rows: int
    n0_mixed: int
    section: int
    fields: dict


@lru_cache(maxsize=4)
def decode_tables(device: str) -> DecodeTables:
    t = TetraTables.load()
    as64 = lambda a: torch.from_numpy(np.asarray(a).astype(np.int64)).to(device)  # noqa: E731
    order_inv = np.empty_like(t.order)
    order_inv[t.order] = np.arange(t.order.size)
    return DecodeTables(
        order_inv=as64(order_inv),
        prefixes=as64(t.prefixes),
        suffixes=as64(t.suffixes),
        branch_c2=as64(t.branch_c2),
        branch_s16=as64(t.branch_s16),
        rows=as64(t.rows),
        values=as64(t.values),
        invnorm=torch.from_numpy(invnorm_table()).to(device),
        class_rows=t.class_rows,
        n0_mixed=t.n0_mixed,
        section=t.section,
        fields=t.fields,
    )


def decode_points(labels: torch.Tensor, dt: DecodeTables) -> torch.Tensor:
    """`(n, 24)` int64 points in natural order: `TetraTables.decode`, in torch.

    Four table lookups and a nibble read per section, the eight coordinates of
    a section at once. Exact: every step is an integer gather or a mask.
    """
    f = {name: (labels >> lo) & ((1 << width) - 1) for name, (lo, width) in dt.fields.items()}
    p, r = f["p"], f["r"]
    c1 = dt.prefixes[f["s8"], f["b1"]]
    c2 = dt.branch_c2[f["s8"], f["b2"]]
    s16 = dt.branch_s16[f["s8"], f["b2"]]
    c3 = dt.suffixes[s16, f["b3"]]
    cr, n0 = dt.class_rows, dt.n0_mixed
    row1 = dt.rows[cr * r + f["i1"]]
    mixed = f["i2"] >= n0
    row2 = torch.where(mixed, dt.rows[cr + f["i2"] - n0], dt.rows[f["i2"]])
    r3 = (p ^ r ^ mixed.to(torch.int64)) & 1
    row3 = dt.rows[cr * r3 + f["i3"]]
    js = torch.arange(dt.section, device=labels.device)
    parts = []
    for c, row in ((c1, row1), (c2, row2), (c3, row3)):
        o = p[:, None] + 2 * ((c[:, None] >> js) & 1)
        rho = (row[:, None] >> (4 * js)) & 15
        parts.append(dt.values[o, rho])
    return torch.cat(parts, dim=1).index_select(1, dt.order_inv)


def tetra_dequant(words: torch.Tensor, gscale: torch.Tensor, rscale: torch.Tensor,
                  tail: torch.Tensor, d_out: int, nblocks: int, tail_w: int,
                  dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """`W'`, `(d_out, nblocks·24 + tail_w)`, in the rotated basis, from the served buffers.

    `tetra48_decode_f`'s arithmetic in f32: `val · (gscale[g] · invnorm[m])`,
    then `· rscale[row]`, then one narrowing to `dtype`. The tail is the f16
    the kernel reads, widened.
    """
    dt = decode_tables(str(words.device))
    labels, gains = words_to_labels(words, d_out, nblocks)
    pts = decode_points(labels, dt)
    m = ((pts * pts).sum(dim=1) >> 4) & 31
    scale = gscale.to(torch.float32)[gains] * dt.invnorm[m]
    w = (pts.to(torch.float32) * scale[:, None]).reshape(d_out, nblocks * 24)
    w = w * rscale.to(torch.float32)[:, None]
    if tail_w:
        w = torch.cat([w, tail.reshape(d_out, tail_w).to(torch.float32)], dim=1)
    return w.to(dtype)


def int4_dequant(wq: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor, d_out: int,
                 d_in: int, group: int, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """`w = scale·q + bias` in f32, nibbles low first, then one narrowing.

    `llvq_tetra.dequant.dequantize_int4` in torch, for the device. The product
    of an f16 scale and a 4-bit `q` is exact in f32, so contracting the multiply
    and the add into an fma cannot move a bit either.
    """
    packed = wq.reshape(d_out, d_in // 2).to(torch.int32)
    q = torch.stack((packed & 0x0F, packed >> 4), dim=2).reshape(d_out, d_in)
    gpr = -(-d_in // group)
    s = scales.reshape(d_out, gpr).to(torch.float32).repeat_interleave(group, dim=1)[:, :d_in]
    b = biases.reshape(d_out, gpr).to(torch.float32).repeat_interleave(group, dim=1)[:, :d_in]
    return (s * q.to(torch.float32) + b).to(dtype)


def int4_rows(wq: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor, ids: torch.Tensor,
              d: int, group: int, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Rows `ids` of an int4 table, dequantized: `emb_q4_gather`'s output, in torch.

    The kernel narrows `scale·q + bias`, computed in f32, to f16 once, which is
    exactly what this does.
    """
    gpr = -(-d // group)
    rows = wq.reshape(-1, d // 2).index_select(0, ids)
    s = scales.reshape(-1, gpr).index_select(0, ids)
    b = biases.reshape(-1, gpr).index_select(0, ids)
    return int4_dequant(rows, s, b, ids.numel(), d, group, dtype)
