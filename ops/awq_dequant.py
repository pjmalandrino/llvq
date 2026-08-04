# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "safetensors>=0.4", "huggingface-hub>=1.26"]
# ///
"""Rebuild dense f16 weights from `Qwen/Qwen3-4B-AWQ` into a loadable checkpoint.

Arm B0 of the measurement campaign (`docs/plan-de-test-v2-cuda.md`) is the 4-bit
quantization published by the model's own author. Our harness cannot read AWQ,
so the only way to score it under *our* forward pass, *our* tokenizer and *our*
perplexity definition is to reconstruct the weights and write a checkpoint that
`Checkpoint::fetch` accepts — entry path R2 of §2.5, zero lines of Rust.

## Why a full checkpoint and not an overlay of the 252 projections

AWQ folds its per-input-channel salience scales into the RMSNorm that precedes
each projection. Measured here, tensor by tensor: of the 146 carried tensors,
**72 differ** from the base checkpoint and 74 are byte-identical. Replacing only
the projections would leave the projections at scale `s` and the norms without
the compensating `1/s` — a model that is not wrong by a little, but wrong. So
every non-quantized tensor is copied **from the AWQ repository**, never from
the base one.

## The one thing this file exists to prevent

Publishing plausible, wrong weights. AWQ's GEMM layout interleaves the eight
nibbles of a word by `AWQ_ORDER`; forget to undo it and you get a permutation of
output channels **in packets of 8**, which loads, runs, and produces numbers.

> Formally forbidden: `gptqmodel.utils.model_dequant.convert_awq_file`. Its
> `unpack_cols` never applies `AWQ_REVERSE_ORDER` (§2.6 of the plan). AutoAWQ,
> the only other reference, has been an archived repository since 2025-05-11.
> There is no library to trust here, so we do not trust one.

## What each control actually closes — and what it does not

* **L2 (repack)** — go from our float output back to `qweight`/`qzeros` and
  demand byte equality with the downloaded file. It closes the group size, the
  zero-point convention (no `-1`), the nibble arithmetic, and any *asymmetry*
  between unpack and repack. It does **not** close the permutation: unpacking
  and repacking with the same wrong order round-trips perfectly. Verified, not
  assumed — a no-permutation reconstruction passes L2 unchanged.
* **L1 (base cross-check)** — the control that does close it. AWQ's own method
  guarantees `W_awq[out, in] ~= W_base[out, in] * s[in]`, so fitting one scalar
  per input channel must leave a residual equal to the arm's own quantization
  noise. Measured on this file: 0.105 when the order is right, **0.97** without
  `AWQ_REVERSE_ORDER`, 0.33 with the spurious `-1` on the zeros, and a plain
  broadcast failure on a wrong group size. The wrong-order failure even has a
  signature: cosine stays at 0.99 exactly for output rows `= 0` and `= 7 (mod
  8)`, the two fixed points of the permutation, and collapses to ~0 for the
  other six.
* **L4 (narrowing budget)** — that we publish AWQ's error and not our own f16
  cast. Measured margin x557 to x577, criterion x100.
* **Iso-perimeter** — 74 identical / 72 different. Any other split means the
  upstream repository moved and the campaign's premises need re-reading.

The quantization error used as L4's yardstick is derived from the scales alone,
`sqrt(gs)*||scales||_F / sqrt(12) / ||W||_F`, so it needs no second
implementation. It agrees with the residual measured against the base
checkpoint to four significant figures (0.1052 against 0.10506 on
`layers.0.self_attn.k_proj`) — two independent routes to the same number.

Arithmetic: everything in float32, a **single** cast to float16 on output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

# --- the AWQ GEMM layout ----------------------------------------------------
#
# Packing (AutoAWQ `packing_utils.pack`): bit field `i` of `packed[:, c]` holds
# `intweight[:, 8c + AWQ_ORDER[i]]`.
# Unpacking is therefore a shift by `4i` followed by the inverse gather:
# `out[:, 8g + t] = pre[:, 8g + AWQ_REVERSE_ORDER[t]]`.
AWQ_ORDER = np.array([0, 2, 4, 6, 1, 3, 5, 7])
AWQ_REVERSE_ORDER = np.array([0, 4, 1, 5, 2, 6, 3, 7])

# --- the objects, as read on the Hub ----------------------------------------

AWQ_REPO = "Qwen/Qwen3-4B-AWQ"
AWQ_REVISION = "74d4bd2bd4bff9cafc9345221320bffb08b406a3"
BASE_REPO = "Qwen/Qwen3-4B"
BASE_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"

# The seven projections a transformer block carries, in the order
# `llvq_llm::artifact::key()` builds them.
PROJECTIONS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)

# Copied verbatim into the output repository. `config.json` is rewritten, not
# copied, because `quantization_config` has to go.
ANNEX_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "generation_config.json",
    "LICENSE",
)

# Structural expectations, per repository. Without an entry the structural
# controls cannot run, and a silently degraded run is worse than no run.
EXPECTED: dict[str, dict] = {
    AWQ_REPO: dict(
        # `model.safetensors`, content-length on the Hub.
        bytes=2_666_027_672,
        tensors=902,
        # 252 x (qweight, qzeros, scales) + 146 carried.
        by_suffix={
            ("qweight", "I32"): 252,
            ("qzeros", "I32"): 252,
            ("scales", "F16"): 252,
            ("weight", "BF16"): 146,
        },
        # Iso-perimeter against the base checkpoint: 36 x (input_layernorm,
        # post_attention_layernorm) carry the folded salience scales, nothing
        # else does.
        carried=146,
        carried_same=74,
        carried_diff=72,
        # sha256 of `tokenizer.json`, identical in the two repositories **and**
        # in the blob sealed inside `~/qwen3-4b-llvq.bin` (fiche-4b §1.2). This
        # is what makes the token fingerprint identical across the three arms by
        # construction rather than by luck.
        tokenizer_sha256=(
            "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
        ),
        # bf16 -> f16 on the embedding, as measured in `docs/fiche-4b.md` §4.2
        # on the base checkpoint — whose embedding bytes this file verifies to
        # be identical to AWQ's. bf16 has 8 mantissa bits against f16's 11, so
        # the narrowing is **exact** over f16's normal range; only the subnormal
        # zone loses anything, which is why the count is this small.
        embed_narrowed=77_045,
        embed_flushed=451,
    ),
}

# L1's thresholds, calibrated on five projections spread across the model
# (`layers.{0,5,17,30,35}`, four projection kinds), each measured correct and
# against three mutants — the table every number below comes from:
#
#   statistic       correct          no-REV         zeros-1        REV<->ORD
#   cosine floor    0.958 – 0.999    -0.014 – 0.00  0.958 – 0.999  -0.014 – 0.00
#   cosine spread   0.0013 – 0.003   ~1.0           ~ as correct   ~1.0
#   ratio           0.999 – 1.441    9.51 – 10.45   3.60 – 3.75    9.53 – 10.45
#
# The two statistics are complementary, and that is the point: the cosine
# signature catches a permutation and is blind to a zero-point shift, the ratio
# catches a zero-point shift. Neither alone would pass this table.
#
# L1's sharp criterion: a row permutation leaves two populations of cosines,
# ~1 at slots 0 and 7 and ~0 at the other six. Anything short of that split is
# not a permutation, whatever the residual says.
L1_COSINE_FLOOR = 0.80   # worst slot measured on a correct read: 0.958
L1_COSINE_SPREAD = 0.15  # worst spread measured on a correct read: 0.003
# The blunt net behind it. Loose on purpose: AutoAWQ does not only rescale, it
# also searches a per-group **clipping** factor, so the residual legitimately
# exceeds pure round-off — most on the layers whose scales spread widest.
# Measured up to 1.441 on `layers.17.mlp.up_proj`, which is a correct
# reconstruction. The ceiling sits 1.7x above that and 1.4x below the closest
# mutant; an earlier 1.60, calibrated on the 1.18 of `layers.35.mlp.down_proj`
# alone, left 11 % of headroom and would have failed a correct full run.
L1_RATIO_CEILING = 2.5
L4_MIN_MARGIN = 100.0  # narrowing must be this far below the arm's own error

HTTP_TIMEOUT = 120
MAX_HEADER_BYTES = 100_000_000

NPDT = {
    "F64": np.dtype("<f8"),
    "F32": np.dtype("<f4"),
    "F16": np.dtype("<f2"),
    "BF16": np.dtype("<u2"),  # no numpy bfloat16; widened on read
    "I64": np.dtype("<i8"),
    "I32": np.dtype("<i4"),
    "I16": np.dtype("<i2"),
    "I8": np.dtype("<i1"),
    "U8": np.dtype("<u1"),
    "BOOL": np.dtype("?"),
}


class Fatal(RuntimeError):
    """An error no retry can fix."""


def jsonable(obj):
    """Non-finite floats become `null`: `json.dumps` would emit `Infinity`,
    which no strict JSON reader accepts, and this report is meant to be read
    by the campaign manifest."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return jsonable(obj.item())
    return obj


# --- transport --------------------------------------------------------------


def _content_range_total(value: str | None) -> int | None:
    """Total size out of a `Content-Range: bytes a-b/total` header."""
    if not value or "/" not in value:
        return None
    tail = value.rsplit("/", 1)[1].strip()
    return int(tail) if tail.isdigit() else None


def http_range(url: str, off: int, n: int, retries: int = 3) -> tuple[bytes, int | None]:
    """Exactly `n` bytes at `off`, and the object's total size if advertised.

    A server that ignores `Range` answers 200 with the whole body. On a 2.7 GB
    file that is not a slow success, it is a silent 2.7 GB download followed by
    a parse of the wrong bytes — so a non-206 status is fatal and never retried.
    """
    last: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(
            url, headers={"Range": f"bytes={off}-{off + n - 1}"}
        )
        try:
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:  # noqa: S310
                if r.status != 206:
                    raise Fatal(
                        f"le serveur a ignoré l'en-tête Range (statut {r.status}) "
                        f"sur {url} : la lecture partielle est impossible"
                    )
                total = _content_range_total(r.headers.get("Content-Range"))
                buf = r.read()
            if len(buf) != n:
                raise OSError(f"{len(buf)} octets reçus, {n} demandés")
            return buf, total
        except Fatal:
            raise
        except (OSError, urllib.error.URLError, TimeoutError) as e:
            last = e
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    raise Fatal(f"lecture Range impossible sur {url} : {last}")


def hub_url(repo: str, filename: str, revision: str = "main") -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"


def hub_json(repo: str, filename: str, revision: str = "main") -> dict:
    """A small text file straight from the Hub — no token, no cache, no download."""
    url = f"https://huggingface.co/{repo}/raw/{revision}/{filename}"
    with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT) as r:  # noqa: S310
        return json.load(r)


def hub_bytes(repo: str, filename: str, revision: str = "main") -> bytes:
    with urllib.request.urlopen(  # noqa: S310
        hub_url(repo, filename, revision), timeout=HTTP_TIMEOUT
    ) as r:
        return r.read()


# --- safetensors ------------------------------------------------------------


class SafeTensors:
    """Random access to one safetensors file, local or over HTTP Range.

    One parser, two transports, deliberately: `check` reads the Hub without
    downloading anything and `dequant` reads the file it pulled, so the cheap
    command exercises the exact code the expensive one runs.
    """

    def __init__(self, source: str | Path):
        s = str(source)
        self.remote = s.startswith("http://") or s.startswith("https://")
        self.source = s
        self._fh = None
        if self.remote:
            self.total: int | None = None
        else:
            self.total = Path(s).stat().st_size
            self._fh = Path(s).open("rb")

        n = struct.unpack("<Q", self.read_at(0, 8))[0]
        if not 0 < n <= MAX_HEADER_BYTES:
            raise Fatal(f"en-tête safetensors invalide : {n} octets annoncés")
        self.header: dict = json.loads(self.read_at(8, n))
        self.metadata = self.header.pop("__metadata__", None)
        self.header_bytes = n
        self.data_start = 8 + n

        # The parse has to land exactly on the file, the way `fiche-4b` §1.2
        # checks the sealed artifact. A header that stops short of the bytes is
        # a header read at the wrong offset.
        end = max((e["data_offsets"][1] for e in self.header.values()), default=0)
        if self.total is not None and self.data_start + end != self.total:
            raise Fatal(
                f"le parse tombe sur {self.data_start + end} octets, "
                f"le fichier en fait {self.total}"
            )

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> SafeTensors:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def read_at(self, off: int, n: int) -> bytes:
        if n <= 0:
            return b""
        if self.remote:
            buf, total = http_range(self.source, off, n)
            if self.total is None:
                self.total = total
            return buf
        self._fh.seek(off)
        buf = self._fh.read(n)
        if len(buf) != n:
            raise Fatal(f"{len(buf)} octets lus, {n} attendus à {off}")
        return buf

    def entry(self, name: str) -> dict:
        try:
            return self.header[name]
        except KeyError:
            raise Fatal(f"tenseur absent : {name}") from None

    def span(self, name: str) -> tuple[int, int]:
        """`(absolute offset, length)` of a tensor's payload."""
        o0, o1 = self.entry(name)["data_offsets"]
        return self.data_start + o0, o1 - o0

    def declared_bytes(self, name: str) -> int:
        """Payload size the header's own shape and dtype imply."""
        e = self.entry(name)
        return NPDT[e["dtype"]].itemsize * int(np.prod(e["shape"], dtype=np.int64))

    def raw(self, name: str) -> bytes:
        e = self.entry(name)
        off, n = self.span(name)
        want = self.declared_bytes(name)
        if n != want:
            raise Fatal(
                f"{name}: {n} octets pour une forme {e['shape']} "
                f"en {e['dtype']} qui en demande {want}"
            )
        return self.read_at(off, n)

    def raw_many(self, names, budget: int = 64_000_000) -> dict[str, bytes]:
        """Payloads for several tensors, coalesced into one read when they are
        close together.

        The 145 small carried tensors of a Qwen3 checkpoint span 392 KB in
        total, so fetching them one by one is 145 TLS handshakes for less than
        half a megabyte.
        """
        names = list(names)
        if not names:
            return {}
        spans = {}
        for n in names:
            off, ln = self.span(n)
            if ln != self.declared_bytes(n):
                raise Fatal(f"{n}: {ln} octets, {self.declared_bytes(n)} déclarés")
            spans[n] = (off, ln)
        lo = min(o for o, _ in spans.values())
        hi = max(o + ln for o, ln in spans.values())
        if hi - lo > budget:
            return {n: self.raw(n) for n in names}
        blob = self.read_at(lo, hi - lo)
        return {n: blob[o - lo : o - lo + ln] for n, (o, ln) in spans.items()}

    def tensor(self, name: str) -> np.ndarray:
        """The tensor as numpy. `BF16` is widened to float32, exactly."""
        e = self.entry(name)
        a = np.frombuffer(self.raw(name), dtype=NPDT[e["dtype"]])
        if e["dtype"] == "BF16":
            a = (a.astype(np.uint32) << 16).view(np.float32)
        return a.reshape(e["shape"])


class ShardWriter:
    """Stream tensors into sharded safetensors, one shard resident at a time.

    Accumulating all 8 GB before writing would need 8 GB of RAM for a file that
    is read back by mmap anyway. Shards are written under temporary names
    because HF's convention (`model-00001-of-00003`) embeds a count nobody knows
    until the last tensor has arrived.
    """

    def __init__(self, out: Path, shard_bytes: int):
        self.out = out
        self.limit = max(1, shard_bytes)
        self._pending: dict[str, np.ndarray] = {}
        self._acc = 0
        self._parts: list[tuple[Path, list[str]]] = []
        self.total = 0
        self.weights = 0

    def add(self, name: str, arr: np.ndarray) -> None:
        if self._pending and self._acc + arr.nbytes > self.limit:
            self._flush()
        self._pending[name] = np.ascontiguousarray(arr)
        self._acc += arr.nbytes
        self.total += arr.nbytes
        self.weights += int(np.prod(arr.shape, dtype=np.int64))

    def _flush(self) -> None:
        from safetensors.numpy import save_file

        path = self.out / f".shard-{len(self._parts):05d}.safetensors"
        save_file(self._pending, str(path))
        self._parts.append((path, sorted(self._pending)))
        self._pending.clear()
        self._acc = 0

    def finish(self) -> tuple[list[str], dict]:
        if self._pending:
            self._flush()
        n = len(self._parts)
        names = (
            ["model.safetensors"]
            if n == 1
            else [f"model-{i + 1:05d}-of-{n:05d}.safetensors" for i in range(n)]
        )
        weight_map: dict[str, str] = {}
        for (tmp, keys), final in zip(self._parts, names):
            tmp.rename(self.out / final)
            for k in keys:
                weight_map[k] = final
        index = {"metadata": {"total_size": self.total}, "weight_map": weight_map}
        # A single-file checkpoint gets no index: `Checkpoint::fetch` tries
        # `model.safetensors` first and only falls back to the index.
        if n > 1:
            (self.out / "model.safetensors.index.json").write_text(
                json.dumps(index, indent=2) + "\n", encoding="utf-8"
            )
        return names, index


# --- the reconstruction -----------------------------------------------------


def unpack_nibbles(packed: np.ndarray) -> np.ndarray:
    """Steps 1 and 2: unfold eight contiguous nibbles, then undo `AWQ_ORDER`.

    `packed` is `[rows, cols]` int32, the result `[rows, cols * 8]` int32 in
    `[0, 15]`.
    """
    if packed.ndim != 2:
        raise Fatal(f"forme empaquetée inattendue : {packed.shape}")
    rows, cols = packed.shape
    p = np.ascontiguousarray(packed, dtype=np.int32).view(np.uint32)
    shifts = np.arange(0, 32, 4, dtype=np.uint32)
    v = ((p[:, :, None] >> shifts[None, None, :]) & np.uint32(0xF)).reshape(
        rows, cols * 8
    )
    gather = np.arange(cols * 8).reshape(-1, 8)[:, AWQ_REVERSE_ORDER].reshape(-1)
    return v[:, gather].astype(np.int32)


def repack_nibbles(iweights: np.ndarray) -> np.ndarray:
    """The exact inverse of `unpack_nibbles`, for control L2."""
    if iweights.ndim != 2 or iweights.shape[1] % 8:
        raise Fatal(f"forme dépliée inattendue : {iweights.shape}")
    if iweights.size and (iweights.min() < 0 or iweights.max() > 15):
        raise Fatal(f"quartets hors de [0, 15] : [{iweights.min()}, {iweights.max()}]")
    rows, n = iweights.shape
    g = iweights.reshape(rows, n // 8, 8)[:, :, AWQ_ORDER].astype(np.uint32)
    out = np.zeros((rows, n // 8), dtype=np.uint32)
    for i in range(8):
        out |= (g[:, :, i] & np.uint32(0xF)) << np.uint32(4 * i)
    return out.view(np.int32)


class Projection:
    """One AWQ-packed projection, unpacked once and shared by every control."""

    def __init__(self, st: SafeTensors, prefix: str, group_size: int):
        self.prefix = prefix
        self.group_size = group_size
        self.qweight = st.tensor(prefix + ".qweight")  # [d_in, d_out/8]  I32
        self.qzeros = st.tensor(prefix + ".qzeros")  # [groups, d_out/8] I32
        self.scales = st.tensor(prefix + ".scales").astype(np.float32)  # [groups,d_out]

        d_in, packed = self.qweight.shape
        groups, d_out = self.scales.shape
        if packed * 8 != d_out:
            raise Fatal(
                f"{prefix}: qweight annonce {packed * 8} sorties, scales {d_out}"
            )
        if self.qzeros.shape != (groups, packed):
            raise Fatal(
                f"{prefix}: qzeros {self.qzeros.shape}, attendu {(groups, packed)}"
            )
        if groups * group_size != d_in:
            raise Fatal(
                f"{prefix}: {groups} groupes x {group_size} != {d_in} canaux d'entrée"
            )
        self.d_in, self.d_out, self.groups = d_in, d_out, groups

        self.iweights = unpack_nibbles(self.qweight)  # [d_in, d_out]
        self.izeros = unpack_nibbles(self.qzeros)  # [groups, d_out]

        # Step 3, in float32, with **no** `-1` on the zeros: AutoAWQ's `-1`
        # lives only in its exllama repack path. Grouping by reshape rather
        # than `np.repeat` keeps two full-size temporaries off the heap and
        # makes the group structure explicit.
        w = (
            self.iweights.reshape(groups, group_size, d_out) - self.izeros[:, None, :]
        ).astype(np.float32)
        w *= self.scales[:, None, :]
        self.w_in_major = w.reshape(d_in, d_out)  # [d_in, d_out]
        # Step 4: [d_out, d_in], contiguous, the layout candle expects.
        self.weight = np.ascontiguousarray(self.w_in_major.T)
        self._f16: np.ndarray | None = None

    @property
    def f16(self) -> np.ndarray:
        """The output tensor. One cast, computed once, reused by L4."""
        if self._f16 is None:
            self._f16 = self.weight.astype(np.float16)
        return self._f16

    def quantization_error(self) -> float:
        """The arm's own relative error, from the scales alone.

        Round-off is uniform on +/- half an LSB and the LSB is the group's
        scale, so the expected Frobenius error is `sqrt(gs)*||scales||/sqrt(12)`.
        Independent of any reconstruction, which is what makes it usable as
        L4's yardstick — and it lands on the residual measured against the base
        checkpoint to four figures.
        """
        num = float(np.sqrt(self.group_size) * np.linalg.norm(self.scales))
        den = float(np.linalg.norm(self.weight))
        return num / np.sqrt(12.0) / den if den > 0 else float("inf")


# --- controls ---------------------------------------------------------------


def selftest_packing() -> list[str]:
    """Pure arithmetic, no bytes: the pack/unpack pair against a naive reference.

    Two claims a round-trip alone would not separate: that the two orders are
    mutual inverses, and that ours is the order AutoAWQ actually writes. The
    second is pinned by re-deriving the packing one bit field at a time, the way
    `generic_ref` pins the search engine.
    """
    fail: list[str] = []
    if not np.array_equal(AWQ_REVERSE_ORDER[AWQ_ORDER], np.arange(8)):
        fail.append("AWQ_REVERSE_ORDER o AWQ_ORDER n'est pas l'identité")
    if not np.array_equal(AWQ_ORDER[AWQ_REVERSE_ORDER], np.arange(8)):
        fail.append("AWQ_ORDER o AWQ_REVERSE_ORDER n'est pas l'identité")

    rng = np.random.default_rng(0x11_0FEED)
    iw = rng.integers(0, 16, size=(37, 8 * 5)).astype(np.int32)

    # Literal transcription of AutoAWQ's `pack`, one bit field at a time. The
    # order is written out **here** rather than read from `AWQ_ORDER`: a
    # reference built from the constant it is meant to check tests nothing, and
    # mutation testing caught exactly that — swapping the two module constants
    # kept them mutually inverse, so a round-trip-only assertion let it through.
    order_map = [0, 2, 4, 6, 1, 3, 5, 7]
    rows, n = iw.shape
    ref = np.zeros((rows, n // 8), dtype=np.uint32)
    for col in range(n // 8):
        for i in range(8):
            ref[:, col] |= iw[:, 8 * col + order_map[i]].astype(np.uint32) << (4 * i)
    if not np.array_equal(repack_nibbles(iw), ref.view(np.int32)):
        fail.append("repack_nibbles ne reproduit pas la boucle de référence")
    if not np.array_equal(unpack_nibbles(ref.view(np.int32)), iw):
        fail.append("unpack_nibbles n'inverse pas la boucle de référence")

    # The top nibble must survive int32's sign bit.
    hi = np.full((2, 8), 15, dtype=np.int32)
    if not np.array_equal(unpack_nibbles(repack_nibbles(hi)), hi):
        fail.append("le quartet de poids fort est perdu (bit de signe int32)")

    # A permutation applied on only one side must be caught: this is exactly the
    # asymmetry L2 detects on real data, and the reason L2 is not vacuous.
    plain = iw.reshape(rows, n // 8, 8).astype(np.uint32)
    naive = np.zeros((rows, n // 8), dtype=np.uint32)
    for i in range(8):
        naive |= plain[:, :, i] << np.uint32(4 * i)
    if np.array_equal(unpack_nibbles(naive.view(np.int32)), iw):
        fail.append("unpack_nibbles accepte un empaquetage sans AWQ_ORDER")
    return fail


def control_l2(p: Projection) -> dict:
    """Go back to the packed integers from our float output and compare bytes.

    Closes the group size, the zero-point convention and the nibble arithmetic.
    Does **not** close the permutation — see the module docstring, and L1.
    """
    sc = p.scales[:, None, :]
    grouped = p.w_in_major.reshape(p.groups, p.group_size, p.d_out)
    nz = sc != 0.0
    q = np.where(nz, grouped / np.where(nz, sc, np.float32(1.0)), np.float32(0.0))
    rec = (np.rint(q).astype(np.int32) + p.izeros[:, None, :]).reshape(p.d_in, p.d_out)

    in_range = bool(rec.min() >= 0 and rec.max() <= 15)
    bad = int(np.count_nonzero(rec != p.iweights))
    zero_scales = int(np.count_nonzero(~nz))

    same_w = same_z = False
    if in_range and bad == 0:
        same_w = bool(np.array_equal(repack_nibbles(rec), p.qweight))
        same_z = bool(np.array_equal(repack_nibbles(p.izeros), p.qzeros))
    return dict(
        ok=bool(in_range and bad == 0 and same_w and same_z),
        integers_recovered=bad == 0,
        in_range=in_range,
        mismatched=bad,
        zero_scales=zero_scales,
        qweight_identical=same_w,
        qzeros_identical=same_z,
    )


def control_l1_base(p: Projection, base_weight: np.ndarray) -> dict:
    """Pin the OUTPUT-channel order against the unquantized checkpoint.

    AWQ scales input channels and folds the inverse into the preceding norm, so
    `W_awq[out, in] ~= W_base[out, in] * s[in]` up to quantization. A
    permutation of output rows destroys that alignment, and destroys it with a
    signature that no other defect produces: rows `= 0` and `= 7 (mod 8)` — the
    fixed points of `AWQ_REVERSE_ORDER` — keep a cosine near 1 while the other
    six collapse toward 0.

    **The verdict rests on that signature, not on the residual size**, and the
    distinction matters. An earlier version demanded the residual equal the
    pure rounding error to within 15 %. It failed on `layers.35.mlp.down_proj`
    at ratio 1.18 while every cosine sat at a uniform 0.958 — i.e. it reported
    a permutation that demonstrably had not happened. The model was incomplete:
    AutoAWQ does not only rescale, it also searches a per-group **clipping**
    factor, so the residual legitimately exceeds pure round-off, and exceeds it
    most on the layers whose scales spread widest (that one spans 0.19 to 5.01
    against 0.66 to 1.53 on the best-behaved). Asserting on a proxy whose model
    omits a term is how a correct reconstruction gets rejected.

    The residual is still reported, and still bounded — loosely, at a level a
    real formula error could not survive — because it costs nothing to keep a
    second, blunter net behind the sharp one.
    """
    if base_weight.shape != p.weight.shape:
        raise Fatal(f"{p.prefix}: base {base_weight.shape} contre AWQ {p.weight.shape}")
    w = p.weight
    b = np.asarray(base_weight, dtype=np.float32)
    den = (b * b).sum(axis=0)
    s = np.divide((w * b).sum(axis=0), den, out=np.zeros_like(den), where=den > 0)
    residual = float(np.linalg.norm(w - b * s) / np.linalg.norm(w))
    q_err = p.quantization_error()
    ratio = residual / q_err if 0 < q_err < math.inf else float("inf")

    num = (w * b).sum(axis=1)
    dn = np.sqrt((w * w).sum(axis=1) * (b * b).sum(axis=1))
    cos = np.divide(num, dn, out=np.zeros_like(num), where=dn > 0)
    by_slot = [float(x) for x in cos.reshape(-1, 8).mean(axis=0)]
    lo, hi = min(by_slot), max(by_slot)
    # A row permutation splits the slots into two populations; nothing else
    # does. Both halves of this are load-bearing: the floor alone would pass a
    # uniformly-degraded fit, the spread alone would pass a uniformly-broken
    # one.
    permuted = lo < L1_COSINE_FLOOR or (hi - lo) > L1_COSINE_SPREAD

    return dict(
        ok=bool(not permuted and s.min() > 0.0 and ratio <= L1_RATIO_CEILING),
        permuted=permuted,
        residual=residual,
        quantization_error=q_err,
        ratio=ratio,
        cosine_floor=lo,
        cosine_spread=hi - lo,
        scale_min=float(s.min()),
        scale_max=float(s.max()),
        cosine_by_slot=[round(x, 4) for x in by_slot],
    )


def control_l4(p: Projection) -> dict:
    """What the single cast to f16 costs, against what the arm already costs."""
    w = p.weight
    h = p.f16
    back = h.astype(np.float32)
    nrm = float(np.linalg.norm(w))
    rel = float(np.linalg.norm(back - w) / nrm) if nrm > 0 else 0.0
    bits = h.view(np.uint16)
    subnormal = int(np.count_nonzero(((bits & 0x7C00) == 0) & ((bits & 0x03FF) != 0)))
    flushed = int(np.count_nonzero((h == 0) & (w != 0)))
    overflow = int(np.count_nonzero(~np.isfinite(back) & np.isfinite(w)))
    q_err = p.quantization_error()
    margin = q_err / rel if rel > 0 else float("inf")
    return dict(
        ok=bool(margin >= L4_MIN_MARGIN and overflow == 0),
        relative=rel,
        quantization_error=q_err,
        margin=margin,
        subnormals=subnormal,
        flushed_to_zero=flushed,
        overflow=overflow,
    )


class BaseCheckpoint:
    """The unquantized checkpoint, read tensor by tensor over HTTP Range."""

    def __init__(self, repo: str = BASE_REPO, revision: str = "main"):
        self.repo, self.revision = repo, revision
        try:
            idx = hub_json(repo, "model.safetensors.index.json", revision)
            self.weight_map: dict[str, str] = idx["weight_map"]
        except urllib.error.HTTPError:
            self.weight_map = {}
        self._shards: dict[str, SafeTensors] = {}
        self._names: set[str] | None = None

    def _shard(self, name: str) -> SafeTensors:
        f = self.weight_map.get(name, "model.safetensors")
        if f not in self._shards:
            self._shards[f] = SafeTensors(hub_url(self.repo, f, self.revision))
        return self._shards[f]

    def names(self) -> set[str]:
        if self._names is None:
            self._names = (
                set(self.weight_map)
                if self.weight_map
                else set(self._shard("model.safetensors").header)
            )
        return self._names

    def has(self, name: str) -> bool:
        return name in self.names()

    def raw(self, name: str) -> bytes:
        return self._shard(name).raw(name)

    def raw_many(self, names, budget: int = 64_000_000) -> dict[str, bytes]:
        by_shard: dict[str, list[str]] = {}
        for n in names:
            by_shard.setdefault(
                self.weight_map.get(n, "model.safetensors"), []
            ).append(n)
        out: dict[str, bytes] = {}
        for group in by_shard.values():
            out.update(self._shard(group[0]).raw_many(group, budget))
        return out

    def tensor(self, name: str) -> np.ndarray:
        return self._shard(name).tensor(name)

    def slice_bytes(self, name: str, off: int, n: int) -> bytes:
        st = self._shard(name)
        base, _ = st.span(name)
        return st.read_at(base + off, n)


def control_perimeter(
    awq: SafeTensors,
    base: BaseCheckpoint,
    *,
    big_threshold: int = 64_000_000,
    probe_bytes: int = 1_000_000,
    full: bool = False,
) -> dict:
    """Which carried tensors AWQ modified — expected 74 identical / 72 different.

    Small tensors are compared byte for byte. The embedding (778 MB) is probed
    at head, middle and tail unless `full`, because downloading it twice to
    confirm what three probes already say is 1.5 GB for no new information.
    Any split other than the expected one means the upstream repository moved.
    """
    carried = sorted(
        n for n in awq.header if n.endswith(".weight") and ".qweight" not in n
    )
    missing = set(n for n in carried if not base.has(n))
    small = [
        n
        for n in carried
        if n not in missing and (full or awq.span(n)[1] <= big_threshold)
    ]
    big = [n for n in carried if n not in missing and n not in small]

    same: list[str] = []
    diff: list[str] = []
    probed: list[str] = []
    a_small, b_small = awq.raw_many(small), base.raw_many(small)
    for n in small:
        (same if a_small[n] == b_small[n] else diff).append(n)
    for n in big:
        off0, span = awq.span(n)
        ok = True
        for frac in (0.0, 0.5, 1.0):
            off = min(int(span * frac), max(0, span - probe_bytes))
            off -= off % 2
            n_read = min(probe_bytes, span - off)
            ok &= awq.read_at(off0 + off, n_read) == base.slice_bytes(n, off, n_read)
        probed.append(n)
        (same if ok else diff).append(n)
    return dict(
        carried=len(carried),
        identical=len(same),
        different=len(diff),
        missing_from_base=sorted(missing),
        sampled=probed,
        different_names=diff,
    )


def control_embedding_narrowing(embed_f32: np.ndarray) -> dict:
    """bf16 -> f16 on the tied embedding, which is also the `lm_head`.

    Reported because it is the one place the reconstruction is not exact, and
    because `docs/fiche-4b.md` §4.2 measured it independently on the base
    checkpoint — whose embedding bytes this file verifies to be identical to
    AWQ's. Reproducing its counts ties our cast to a number nobody here chose.
    """
    h = embed_f32.astype(np.float16)
    back = h.astype(np.float32)
    changed = back != embed_f32
    n_changed = int(np.count_nonzero(changed))
    return dict(
        values=int(embed_f32.size),
        changed=n_changed,
        flushed_to_zero=int(np.count_nonzero((h == 0) & (embed_f32 != 0))),
        touched_max=float(np.abs(embed_f32[changed]).max()) if n_changed else 0.0,
        abs_error_max=float(np.abs(back - embed_f32).max()),
        max_abs=float(np.abs(embed_f32).max()),
    )


# --- plumbing ---------------------------------------------------------------


def projection_names(n_layers: int) -> list[str]:
    return [f"model.layers.{b}.{p}" for b in range(n_layers) for p in PROJECTIONS]


def expected_weight_count(cfg: dict) -> int:
    """Every parameter the written checkpoint must carry, from `config.json`.

    Same arithmetic as `ops/run.py:weight_counts`, kept here so the two can
    disagree loudly rather than share a bug. On Qwen3-4B it is 4 022 468 096 —
    the number `docs/fiche-4b.md` §1.2 reads off the sealed artifact.
    """
    layers = cfg["num_hidden_layers"]
    hidden = cfg["hidden_size"]
    inter = cfg["intermediate_size"]
    attn_out = cfg["head_dim"] * cfg["num_attention_heads"]
    kv = cfg["head_dim"] * cfg["num_key_value_heads"]
    per_layer = (
        hidden * attn_out  # q_proj
        + 2 * hidden * kv  # k_proj, v_proj
        + attn_out * hidden  # o_proj
        + 3 * hidden * inter  # gate_proj, up_proj, down_proj
        + 2 * hidden  # input_layernorm, post_attention_layernorm
        + 2 * cfg["head_dim"]  # q_norm, k_norm
    )
    embed = cfg["vocab_size"] * hidden
    carried = embed if cfg.get("tie_word_embeddings", False) else 2 * embed
    return layers * per_layer + hidden + carried


def quant_config(cfg: dict) -> tuple[int, int]:
    """`(bits, group_size)`, refusing anything this file was not written for."""
    q = cfg.get("quantization_config")
    if not q:
        raise Fatal("config.json sans quantization_config : ce n'est pas un dépôt AWQ")
    if q.get("quant_method") != "awq" or q.get("version") != "gemm":
        raise Fatal(
            f"format non géré : quant_method={q.get('quant_method')}, "
            f"version={q.get('version')} (attendu awq/gemm)"
        )
    if q.get("bits") != 4:
        raise Fatal(f"{q.get('bits')} bits : seul le 4 bits est implémenté")
    if not q.get("zero_point", False):
        raise Fatal("zero_point=false : la formule de ce fichier suppose l'asymétrique")
    if q.get("modules_to_not_convert"):
        raise Fatal(
            f"modules_to_not_convert={q['modules_to_not_convert']} : "
            "le périmètre quantifié n'est plus les 252 projections"
        )
    return q["bits"], q["group_size"]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while buf := f.read(chunk):
            h.update(buf)
    return h.hexdigest()


def inventory(awq: SafeTensors) -> dict:
    counts: dict[str, int] = {}
    for name, e in awq.header.items():
        k = f"{name.rsplit('.', 1)[-1]}/{e['dtype']}"
        counts[k] = counts.get(k, 0) + 1
    return dict(
        tensors=len(awq.header),
        header_bytes=awq.header_bytes,
        bytes=awq.total,
        by_suffix=dict(sorted(counts.items())),
    )


def check_structure(awq: SafeTensors, exp: dict | None, say) -> bool:
    inv = inventory(awq)
    say(f"  {inv['tensors']} tenseurs · en-tête {inv['header_bytes']} o "
        f"· fichier {inv['bytes']} o")
    for k, v in inv["by_suffix"].items():
        say(f"      {k:<16} {v}")
    if exp is None:
        return True
    ok = True
    if inv["bytes"] != exp["bytes"]:
        say(f"  FAIL  {inv['bytes']} octets, {exp['bytes']} attendus")
        ok = False
    if inv["tensors"] != exp["tensors"]:
        say(f"  FAIL  {inv['tensors']} tenseurs, {exp['tensors']} attendus")
        ok = False
    want = {f"{s}/{d}": n for (s, d), n in exp["by_suffix"].items()}
    if inv["by_suffix"] != want:
        say(f"  FAIL  inventaire {inv['by_suffix']}, attendu {want}")
        ok = False
    if ok:
        say("  ok    structure conforme (octets, comptes, dtypes)")
    return ok


def check_tokenizer(repo: str, revision: str, exp: dict | None, say) -> bool:
    blob = hub_bytes(repo, "tokenizer.json", revision)
    got = hashlib.sha256(blob).hexdigest()
    say(f"  tokenizer.json  {len(blob)} o  sha256 {got}")
    if exp is None:
        return True
    if got != exp["tokenizer_sha256"]:
        say(f"  FAIL  sha256 attendu {exp['tokenizer_sha256']}")
        say("        l'empreinte de tokens n'est plus commune aux trois bras")
        return False
    say("  ok    identique au checkpoint de base et au blob du .bin scellé")
    return True


def report_perimeter(res: dict, exp: dict | None, say) -> bool:
    say(f"  {res['carried']} tenseurs portés : {res['identical']} identiques, "
        f"{res['different']} différents"
        + (f"  (dont {len(res['sampled'])} échantillonné(s))" if res["sampled"] else ""))
    if res["missing_from_base"]:
        say(f"  FAIL  absents du checkpoint de base : {res['missing_from_base'][:5]}")
        return False
    if exp is None:
        return True
    if (
        res["carried"] == exp["carried"]
        and res["identical"] == exp["carried_same"]
        and res["different"] == exp["carried_diff"]
    ):
        say(f"  ok    {exp['carried_same']}/{exp['carried_diff']} comme attendu — "
            "AWQ replie ses échelles dans les RMSNorm")
        return True
    say(f"  FAIL  attendu {exp['carried']} portés, {exp['carried_same']} identiques, "
        f"{exp['carried_diff']} différents. Le dépôt amont a bougé : relire §2.4 "
        "du plan avant d'aller plus loin")
    say(f"        premiers différents : {res['different_names'][:4]}")
    return False


def report_embedding(res: dict, exp: dict | None, say) -> bool:
    say(f"  embedding bf16->f16 : {res['changed']} valeurs sur {res['values']} "
        f"({res['changed'] / res['values']:.3e}), {res['flushed_to_zero']} à zéro")
    say(f"      |v| touché max {res['touched_max']:.3e} · erreur abs max "
        f"{res['abs_error_max']:.3e} · |v| max {res['max_abs']:.4f}")
    if exp is None:
        return True
    if (
        res["changed"] == exp["embed_narrowed"]
        and res["flushed_to_zero"] == exp["embed_flushed"]
    ):
        say("  ok    reproduit fiche-4b §4.2 au comptage près")
        return True
    say(f"  FAIL  fiche-4b §4.2 mesure {exp['embed_narrowed']} changées et "
        f"{exp['embed_flushed']} à zéro.")
    say("        Soit le dépôt amont a bougé, soit la référence est à revoir : "
        "un humain tranche, ce script ne devine pas.")
    return False


def sample_indices(total: int, k: int) -> list[int]:
    """`k` evenly spaced indices, ends included — deterministic, so a `check`
    that passed once is the same `check` next week."""
    k = max(1, min(k, total))
    if k == 1:
        return [0]
    return sorted({round(i * (total - 1) / (k - 1)) for i in range(k)})


def run_selftest(say) -> bool:
    say("\n[0] arithmétique d'empaquetage (aucun octet lu)")
    failures = selftest_packing()
    for line in failures:
        say(f"  FAIL  {line}")
    if not failures:
        say("  ok    ordres mutuellement inverses, pack/unpack contre référence "
            "naïve, empaquetage non permuté rejeté")
    return not failures


def resolve_expectations(repo: str, allow_unknown: bool, say) -> tuple[dict | None, bool]:
    exp = EXPECTED.get(repo)
    if exp is not None:
        return exp, True
    say(f"\n⚠️  aucune attente enregistrée pour {repo} : les contrôles de "
        "structure, d'iso-périmètre et de tokenizer sont désactivés")
    if not allow_unknown:
        say("    relance avec --allow-unknown-repo pour l'accepter explicitement")
    return None, allow_unknown


# --- commands ---------------------------------------------------------------


def cmd_check(args) -> int:
    """L2, L1 and L4 on a sample, plus the structural controls — no download."""
    say = print
    ok = run_selftest(say)

    cfg = hub_json(args.awq_repo, "config.json", args.awq_revision)
    bits, gs = quant_config(cfg)
    n_layers = cfg["num_hidden_layers"]
    exp, allowed = resolve_expectations(args.awq_repo, args.allow_unknown_repo, say)
    if not allowed:
        return 2

    say(f"\n[1] structure de {args.awq_repo}@{args.awq_revision[:8]} "
        f"({bits} bits, group_size {gs}, {n_layers} couches)")
    awq = SafeTensors(hub_url(args.awq_repo, "model.safetensors", args.awq_revision))
    ok &= check_structure(awq, exp, say)

    say("\n[2] tokenizer")
    ok &= check_tokenizer(args.awq_repo, args.awq_revision, exp, say)

    base = BaseCheckpoint(args.base_repo, args.base_revision)
    names = projection_names(n_layers)
    missing = [n for n in names if n + ".qweight" not in awq.header]
    if missing:
        say(f"\n  FAIL  {len(missing)} projections absentes, ex. {missing[0]}")
        ok = False
    picks = [names[i] for i in sample_indices(len(names), args.samples)]

    say(f"\n[3] L2 / L1 / L4 sur {len(picks)} projections sur {len(names)}")
    tested_l1 = 0
    for prefix in picks:
        p = Projection(awq, prefix, gs)
        l2 = control_l2(p)
        l4 = control_l4(p)
        say(f"  {prefix}  [{p.d_out}x{p.d_in}]")
        say(f"      L2  ré-empaquetage "
            f"{'IDENTIQUE' if l2['ok'] else 'DIFFÉRENT'}"
            f"  (entiers récupérés {l2['integers_recovered']},"
            f" quartets != {l2['mismatched']},"
            f" échelles nulles {l2['zero_scales']})")
        ok &= l2["ok"]
        if base.has(prefix + ".weight"):
            l1 = control_l1_base(p, base.tensor(prefix + ".weight"))
            tested_l1 += 1
            say(f"      L1  cosinus min {l1['cosine_floor']:.4f}, écart entre "
                f"positions {l1['cosine_spread']:.4f}  {'ok' if l1['ok'] else 'FAIL'}")
            say(f"          résidu {l1['residual']:.5f} contre arrondi pur "
                f"{l1['quantization_error']:.5f}, ratio {l1['ratio']:.3f}"
                f"  (> 1 attendu : AWQ écrête)")
            say(f"          s dans [{l1['scale_min']:.4f}, {l1['scale_max']:.4f}]"
                f"  cosinus par position mod 8 {l1['cosine_by_slot']}")
            if l1["permuted"]:
                say("          les positions se séparent en deux populations : "
                    "c'est la signature\n          d'un AWQ_REVERSE_ORDER non "
                    "appliqué, et rien d'autre ne la produit")
            elif not l1["ok"]:
                say(f"          résidu au-delà du plafond {L1_RATIO_CEILING:.2f} "
                    "sans signature de permutation —\n          la lecture du "
                    "format est probablement juste (voir L2), mais quelque chose\n"
                    "          d'autre a bougé chez l'adversaire : à examiner "
                    "avant de mesurer")
            ok &= l1["ok"]
        else:
            say("      L1  ignoré : tenseur absent du checkpoint de base")
        say(f"      L4  {l4['relative']:.3e} relatif, marge x{l4['margin']:,.0f}"
            f"  subnormaux {l4['subnormals']}  ->0 {l4['flushed_to_zero']}"
            f"  {'ok' if l4['ok'] else 'FAIL'}")
        ok &= l4["ok"]
        del p

    if tested_l1 == 0:
        say("  FAIL  L1 n'a tourné sur aucun tenseur : l'ordre des quartets "
            "n'est pinné par rien")
        ok = False

    say("\n[4] iso-périmètre des tenseurs portés")
    ok &= report_perimeter(
        control_perimeter(awq, base, full=args.full_embed), exp, say
    )

    awq.close()
    say(f"\n{'TOUS LES CONTRÔLES PASSENT' if ok else 'AU MOINS UN CONTRÔLE ÉCHOUE'}")
    return 0 if ok else 1


def cmd_dequant(args) -> int:
    """Full reconstruction to `--out`, controls printed and recorded."""
    from safetensors.numpy import save_file  # noqa: F401  — fail early, not at [6]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    say = print
    report: dict = dict(
        source=dict(repo=args.awq_repo, revision=args.awq_revision),
        base=dict(repo=args.base_repo, revision=args.base_revision),
        controls={},
    )

    ok = run_selftest(say)
    report["controls"]["packing"] = dict(ok=ok, failures=selftest_packing())
    if not ok:
        say("\nabandon : la couche d'empaquetage est fausse, rien d'autre n'a de sens")
        return 1

    cfg = hub_json(args.awq_repo, "config.json", args.awq_revision)
    bits, gs = quant_config(cfg)
    n_layers = cfg["num_hidden_layers"]
    exp, allowed = resolve_expectations(args.awq_repo, args.allow_unknown_repo, say)
    if not allowed:
        return 2

    if args.awq_file:
        src = Path(args.awq_file)
    else:
        from huggingface_hub import hf_hub_download

        say(f"\n[1] téléchargement de {args.awq_repo}/model.safetensors "
            f"(~{(exp or {}).get('bytes', 0) / 1e9:.2f} Go)")
        src = Path(
            hf_hub_download(
                repo_id=args.awq_repo,
                filename="model.safetensors",
                revision=args.awq_revision,
            )
        )
    awq = SafeTensors(src)
    say(f"\n[1] structure ({bits} bits, group_size {gs}, {n_layers} couches)")
    st_ok = check_structure(awq, exp, say)
    ok &= st_ok
    report["controls"]["structure"] = dict(ok=st_ok, **inventory(awq))

    say("\n[2] tokenizer")
    tk_ok = check_tokenizer(args.awq_repo, args.awq_revision, exp, say)
    ok &= tk_ok
    report["controls"]["tokenizer"] = dict(ok=tk_ok)

    base = BaseCheckpoint(args.base_repo, args.base_revision)
    say("\n[3] iso-périmètre des tenseurs portés")
    perim = control_perimeter(awq, base, full=args.full_embed)
    p_ok = report_perimeter(perim, exp, say)
    ok &= p_ok
    report["controls"]["perimeter"] = dict(ok=p_ok, **perim)

    names = projection_names(n_layers)
    picks = {names[i] for i in sample_indices(len(names), args.l1_samples)}
    writer = ShardWriter(out, int(args.shard_gb * 1e9))

    say(f"\n[4] reconstruction des {len(names)} projections "
        f"— L2 sur toutes, L1 sur {len(picks)}, L4 sur toutes")
    l2_all, l4_all, l1_all = [], [], []
    worst_margin = float("inf")
    for prefix in names:
        p = Projection(awq, prefix, gs)
        l2 = control_l2(p)
        l4 = control_l4(p)
        l2_all.append(dict(name=prefix, **l2))
        l4_all.append(dict(name=prefix, **l4))
        worst_margin = min(worst_margin, l4["margin"])
        line = (
            f"  {prefix:<44} [{p.d_out:>5}x{p.d_in:<5}]"
            f"  L2 {'ok  ' if l2['ok'] else 'FAIL'}"
            f"  L4 {l4['relative']:.3e} x{l4['margin']:>6,.0f}"
            f"  sub {l4['subnormals']:>6}  ->0 {l4['flushed_to_zero']:>6}"
        )
        if prefix in picks and base.has(prefix + ".weight"):
            l1 = control_l1_base(p, base.tensor(prefix + ".weight"))
            l1_all.append(dict(name=prefix, **l1))
            line += f"  L1 {l1['ratio']:.3f} {'ok' if l1['ok'] else 'FAIL'}"
            ok &= l1["ok"]
            say(line)
            if not l1["ok"]:
                say(f"      cosinus par position mod 8 : {l1['cosine_by_slot']}")
        else:
            say(line)
        ok &= l2["ok"] and l4["ok"]
        writer.add(prefix + ".weight", p.f16)  # the single cast, via Projection.f16
        del p

    if not l1_all:
        say("  FAIL  L1 n'a tourné sur aucun tenseur : l'ordre des quartets "
            "n'est pinné par rien")
        ok = False
    say(f"  marge L4 la pire : x{worst_margin:,.0f} (critère x{L4_MIN_MARGIN:,.0f})")
    report["controls"]["l2"] = dict(
        ok=all(r["ok"] for r in l2_all), tested=len(l2_all), detail=l2_all
    )
    report["controls"]["l1"] = dict(
        ok=bool(l1_all) and all(r["ok"] for r in l1_all),
        tested=len(l1_all),
        detail=l1_all,
    )
    report["controls"]["l4"] = dict(
        ok=all(r["ok"] for r in l4_all),
        tested=len(l4_all),
        worst_margin=worst_margin,
        detail=l4_all,
    )

    say("\n[5] tenseurs portés, copiés du dépôt AWQ")
    carried = sorted(
        n for n in awq.header if n.endswith(".weight") and ".qweight" not in n
    )
    for n in carried:
        a = awq.tensor(n)  # BF16 widened to float32, exactly
        if n == "model.embed_tokens.weight":
            emb = control_embedding_narrowing(a)
            e_ok = report_embedding(emb, exp, say)
            ok &= e_ok
            report["controls"]["embedding_narrowing"] = dict(ok=e_ok, **emb)
        writer.add(n, a.astype(np.float16))
    say(f"  {len(carried)} tenseurs, bf16 -> f16")
    say("  ⚠️  le checkpoint écrit est f16 partout, donc les 74 tenseurs qui")
    say("      étaient bit-identiques au checkpoint de base ne le sont plus en")
    say("      tant qu'octets. Chargés en f16 — ce que la campagne impose —")
    say("      les valeurs sont identiques ; en f32 elles ne le sont pas.")
    say("      Même écart que pour le .bin scellé, cf. fiche-4b §4.2.")

    say(f"\n[6] écriture dans {out}")
    files, index = writer.finish()
    want_weights = expected_weight_count(cfg)
    say(f"  {len(index['weight_map'])} tenseurs · {writer.weights} poids "
        f"· {writer.total} octets · {len(files)} fichier(s)")
    if writer.weights != want_weights:
        say(f"  FAIL  {writer.weights} poids écrits, {want_weights} déduits de "
            "config.json — un tenseur manque ou une forme est fausse")
        ok = False
    else:
        say(f"  ok    {want_weights} poids, exactement ce que config.json décrit")

    # `quantization_config` has to go: what is written is no longer quantized,
    # and leaving it would tell every other loader to expect packed nibbles.
    out_cfg = dict(cfg)
    out_cfg.pop("quantization_config", None)
    out_cfg["torch_dtype"] = "float16"
    (out / "config.json").write_text(
        json.dumps(out_cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    copied = ["config.json"]
    for name in ANNEX_FILES:
        try:
            (out / name).write_bytes(hub_bytes(args.awq_repo, name, args.awq_revision))
            copied.append(name)
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
            say(f"  (absent du dépôt source, non copié : {name})")
    say(f"  annexes : {', '.join(copied)}")

    # Re-read what we wrote with our own parser, which re-runs the "the parse
    # lands exactly on the file" check on every shard.
    say("\n[7] relecture des fichiers écrits")
    seen: set[str] = set()
    for f in files:
        with SafeTensors(out / f) as st:
            seen |= set(st.header)
    reread_ok = seen == set(index["weight_map"])
    say(f"  {'ok   ' if reread_ok else 'FAIL '} {len(seen)} tenseurs relus sur "
        f"{len(index['weight_map'])} annoncés par l'index")
    ok &= reread_ok
    report["controls"]["reread"] = dict(ok=reread_ok, tensors=len(seen))

    report["ok"] = bool(ok)
    report["output"] = dict(
        files=[
            dict(name=f.name, bytes=f.stat().st_size, sha256=sha256_file(f))
            for f in sorted(out.iterdir())
            if f.is_file() and f.name != "RECONSTRUCTION.json"
        ],
        tensors=len(index["weight_map"]),
        weights=writer.weights,
        total_size=writer.total,
    )
    (out / "RECONSTRUCTION.json").write_text(
        json.dumps(jsonable(report), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    awq.close()

    if ok:
        say(f"\nTOUS LES CONTRÔLES PASSENT — rapport dans "
            f"{out / 'RECONSTRUCTION.json'}")
        say("  publier : uv run ops/awq_dequant.py push <user>/<repo> "
            f"--out {out} --public --yes")
    else:
        say("\n🚨 AU MOINS UN CONTRÔLE ÉCHOUE — ces poids ne doivent être ni "
            "publiés ni scorés. `push` les refusera.")
    return 0 if ok else 1


MODEL_CARD = """---
license: apache-2.0
base_model: {awq_repo}
tags:
  - measurement-artifact
  - not-for-use
---

> **This is a measurement reconstruction, not a model to use.** It is the dense
> f16 dequantization of [`{awq_repo}`]({awq_url}), published only so that a
> quantization benchmark can be replayed by a third party. Use the original.
>
> **Reconstruction de mesure, pas un modèle à utiliser.**

# {repo}

## Pourquoi ce dépôt existe

Le projet LLVQ compare une quantification 2 bits sur le réseau de Leech à la
quantification 4 bits AWQ publiée par l'auteur du modèle, sur Qwen3-4B. Les
trois bras doivent passer par **le même moteur, le même tokenizer et la même
définition de la perplexité** ; or ce harnais ne sait pas lire le format AWQ
empaqueté.

Ce dépôt contient donc les poids AWQ **reconstruits en f16 dense**. Il ne
compresse rien, il ne va pas plus vite, et il pèse plus lourd que l'original :
il n'a d'autre intérêt que de rendre la mesure rejouable.

## Pourquoi un checkpoint complet et pas seulement les projections

AWQ replie ses échelles de saillance par canal dans les RMSNorm qui précèdent
les projections. Sur les {n_carried} tenseurs non quantifiés, **{diff} sont
modifiés** par rapport au checkpoint de base et **{same} sont bit pour bit
identiques**. Ne remplacer que les {n_proj} projections produirait un modèle
mathématiquement faux — projections à l'échelle `s`, normes sans le `1/s`
compensatoire.

Tout ce qui n'est pas quantifié est donc copié **depuis le dépôt AWQ**, jamais
depuis le checkpoint de base.

## Provenance

| | |
|---|---|
| source | `{awq_repo}` révision `{awq_revision}` |
| checkpoint de référence | `{base_repo}` révision `{base_revision}` |
| produit par | `ops/awq_dequant.py dequant` |
| dtype | f16 partout |
| tenseurs | {n_tensors} ({n_proj} projections reconstruites, {n_carried} portés) |
| poids | {n_weights} |

`quantization_config` est **retiré** du `config.json` : ce qui est écrit ici
n'est plus quantifié.

## Contrôles passés avant publication

* **L2 — ré-empaquetage.** Nos flottants sont re-convertis en `qweight` /
  `qzeros` et comparés **octet pour octet** au fichier source, sur les {n_proj}
  projections. Ferme le group size, la convention de zéro (pas de `-1`) et
  l'arithmétique des quartets.
* **L1 — recoupement avec le checkpoint non quantifié.** AWQ garantit
  `W_awq[out, in] ~= W_base[out, in] * s[in]` ; ajuster un scalaire par canal
  d'entrée doit laisser exactement le bruit de quantification du bras. C'est ce
  contrôle qui épingle l'ordre `AWQ_REVERSE_ORDER` des quartets, que L2 seul ne
  peut pas voir.
* **L4 — budget de narrowing.** L'erreur du cast f16 est au moins {margin} fois
  sous l'erreur de quantification du bras lui-même.
* **Iso-périmètre.** {same} tenseurs portés identiques, {diff} différents.

Le rapport complet, avec le sha256 de chaque fichier, est dans
`RECONSTRUCTION.json`.

## Limite déclarée

On mesure la **reconstruction**, pas l'arithmétique fusionnée du noyau AWQ. Un
noyau fusé accumule dans un autre ordre ; l'écart est borné, pas nul.
"""


def cmd_push(args) -> int:
    """Publish the reconstruction — refuses anything whose controls did not pass."""
    from huggingface_hub import create_repo, upload_folder

    out = Path(args.out)
    rpath = out / "RECONSTRUCTION.json"
    if not rpath.exists():
        print(f"aucun rapport dans {rpath} : lance `dequant` d'abord", file=sys.stderr)
        return 2
    report = json.loads(rpath.read_text(encoding="utf-8"))
    if not report.get("ok"):
        failed = [k for k, v in report.get("controls", {}).items() if not v.get("ok")]
        print(f"refus : les contrôles {failed} n'ont pas passé."
              "\nUn objet qu'on ne sait pas juste ne se publie pas.", file=sys.stderr)
        return 1

    margin = report["controls"]["l4"]["worst_margin"]
    card = MODEL_CARD.format(
        repo=args.repo,
        awq_repo=report["source"]["repo"],
        awq_url=f"https://huggingface.co/{report['source']['repo']}",
        awq_revision=report["source"]["revision"],
        base_repo=report["base"]["repo"],
        base_revision=report["base"]["revision"],
        n_tensors=report["output"]["tensors"],
        n_proj=report["controls"]["l2"]["tested"],
        n_carried=report["controls"]["perimeter"]["carried"],
        n_weights=report["output"]["weights"],
        same=report["controls"]["perimeter"]["identical"],
        diff=report["controls"]["perimeter"]["different"],
        margin=f"{margin:,.0f}" if isinstance(margin, (int, float)) else "n/d",
    )
    (out / "README.md").write_text(card, encoding="utf-8")

    total = sum(f["bytes"] for f in report["output"]["files"])
    print(f"\nà publier : {out} -> {args.repo} "
          f"({'public' if args.public else 'privé'})")
    print(f"  {len(report['output']['files'])} fichiers, {total / 1e9:.2f} Go")
    print("  carte : « reconstruction de mesure, pas un modèle à utiliser »")
    if not args.public:
        print("\n  ⚠️  un dépôt PRIVÉ n'est pas scorable : côté Rust, `hf-hub` ne lit"
              "\n      jamais HF_TOKEN (plan §2.5). Pour la campagne, --public.")
    if not args.yes:
        sys.stdout.flush()
        print("\nrefus : publier engage le compte. Relance avec --yes.",
              file=sys.stderr)
        return 1

    create_repo(args.repo, repo_type="model", private=not args.public, exist_ok=True)
    upload_folder(
        repo_id=args.repo,
        repo_type="model",
        folder_path=str(out),
        commit_message=f"Reconstruction f16 dense de {report['source']['repo']}",
    )
    print(f"\npublié : https://huggingface.co/{args.repo}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--awq-repo", default=AWQ_REPO)
        sp.add_argument("--awq-revision", default=AWQ_REVISION)
        sp.add_argument("--base-repo", default=BASE_REPO)
        sp.add_argument("--base-revision", default=BASE_REVISION)
        sp.add_argument("--full-embed", action="store_true",
                        help="comparer l'embedding en entier (778 Mo) au lieu de "
                             "trois sondages")
        sp.add_argument("--allow-unknown-repo", action="store_true",
                        help="accepter un dépôt sans attentes enregistrées")

    c = sub.add_parser("check", help="L2/L1/L4 par requêtes Range, sans téléchargement")
    common(c)
    c.add_argument("--samples", type=int, default=4,
                   help="projections échantillonnées (~10 Mo AWQ + 25 Mo base pièce)")
    c.set_defaults(fn=cmd_check)

    d = sub.add_parser("dequant", help="reconstruction complète vers un répertoire")
    common(d)
    d.add_argument("--out", required=True, help="répertoire de sortie")
    d.add_argument("--awq-file", default=None,
                   help="model.safetensors déjà local, au lieu de le retélécharger")
    d.add_argument("--l1-samples", type=int, default=6,
                   help="projections recoupées avec le checkpoint de base")
    d.add_argument("--shard-gb", type=float, default=4.0,
                   help="taille max d'un shard, en Go décimaux")
    d.set_defaults(fn=cmd_dequant)

    u = sub.add_parser("push", help="publier la reconstruction sur le Hub")
    u.add_argument("repo", help="ex. Pier-Jean/qwen3-4b-awq-deq")
    u.add_argument("--out", required=True, help="répertoire produit par `dequant`")
    u.add_argument("--public", action="store_true",
                   help="requis pour que bin/ppl puisse le charger")
    u.add_argument("--yes", action="store_true", help="confirmer la publication")
    u.set_defaults(fn=cmd_push)

    args = p.parse_args()
    try:
        return args.fn(args)
    except Fatal as e:
        print(f"\nerreur : {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrompu", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
