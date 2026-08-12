# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "safetensors>=0.4", "huggingface-hub>=1.26"]
# ///
"""Rebuild dense f16 weights from `Qwen/Qwen3-4B-AWQ` into a loadable checkpoint.

Arm B0 of the measurement campaign (`docs/archive/plan-de-test-v2-cuda.md`) is the 4-bit
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
  signature: cosine stays near 1 exactly for output rows `= 0` and `= 7 (mod
  8)`, the two fixed points of the permutation, and collapses to ~0 for the
  other six. L1 also closes something L2 traverses without noticing: `scales`
  are stored **unpermuted** while `qweight` and `qzeros` are permuted, and
  re-packing divides by the same scales it multiplied by.
* **L4 (narrowing budget)** — that we publish AWQ's error and not our own f16
  cast. Measured margin x557 to x577, criterion x100.
* **Iso-perimeter** — 74 identical / 72 different. Any other split means the
  upstream repository moved and the campaign's premises need re-reading.
* **`selftest`** — all of the above, offline, in a second, against a synthetic
  projection whose ground truth we chose. It is the only place where the
  controls are ever *seen to fail*: seven named mutants, each of which has to
  die at a named net. A control that has never rejected anything is a control
  whose failure mode is unknown.

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
#
# Each entry also carries the two **revisions** it was measured at. A revision
# is a fact about a repository, not about an invocation: pinning the 4B's SHA as
# the command line's default made every other repository 404 before reading a
# byte — see `resolve_revision`.
EXPECTED: dict[str, dict] = {
    AWQ_REPO: dict(
        awq_revision=AWQ_REVISION,
        base_repo=BASE_REPO,
        base_revision=BASE_REVISION,
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
        # bf16 -> f16 on the embedding. `docs/fiche-4b.md` §4.2 measured these
        # on the **base** checkpoint; they have since been recounted on **AWQ's
        # own 777 912 320 bytes**, streamed by Range: 77 045 changed
        # (1.981e-4), 451 flushed, largest touched value 7.5996e-6, largest
        # absolute error 2.9802e-8 = 2^-25, max |v| 0.250000. Every figure
        # identical, which does more than confirm the constants: it says the two
        # embeddings agree far beyond the three probes the perimeter control
        # affords.
        #
        # bf16 carries 8 significand bits against f16's 11, so the narrowing is
        # **exact** wherever f16 is normal; a bf16 at 2^e is a multiple of
        # 2^(e-7) and f16's subnormal grid is 2^-24, so everything at or above
        # 2^-17 survives untouched. That is why the count is this small, and why
        # the largest touched value sits just under 2^-17 = 7.6294e-6.
        embed_narrowed=77_045,
        embed_flushed=451,
    ),
    # --- the 14B, measured 2026-08-09 by `check --allow-unknown-repo` ---------
    #
    # Every figure below was read off the tool itself at those two revisions —
    # `check` for the structure, the tokenizer and the perimeter, and a Range
    # stream through `control_embedding_narrowing` for the last two. Nothing
    # here is a 4B constant rescaled by 40/36.
    "Qwen/Qwen3-14B-AWQ": dict(
        awq_revision="31c69efc29464b6bb0aee1398b5a7b50a99340c3",
        base_repo="Qwen/Qwen3-14B",
        base_revision="40c069824f4251a91eefaf281ebe4c544efd3e18",
        # **Sharded**, unlike the 4B: `model-00001-of-00002` (4 988 339 832) +
        # `model-00002-of-00002` (4 988 350 408), which is the sum
        # `ShardedSafeTensors.total` forms and therefore what `check_structure`
        # compares against. The field needs no generalisation — the class
        # already presents a sharded export as one header and one size.
        #
        # ⚠️ This is **not** `metadata.total_size` of `model.safetensors.index
        # .json`, which announces 9 989 683 200 — 13 107 200 bytes above the
        # payload the shards actually carry. The served bytes are the ones the
        # parser proves: `SafeTensors.__init__` checks, per shard, that the
        # header lands exactly on that shard's own length. Sourcing this field
        # from the index instead would fail the control on a correct repository.
        bytes=9_976_690_240,
        tensors=1_003,
        # 280 x (qweight, qzeros, scales) + 163 carried, and 280 = 40 layers x 7
        # projections where the 4B has 36 x 7 = 252.
        #
        # ⚠️ `scales` are **BF16** here, F16 on the 4B — same publisher, same
        # `quant_method`/`version`, different storage dtype. Read off the
        # aggregated header, not deduced. It is not cosmetic: `Projection` reads
        # them through `SafeTensors.tensor`, which widens BF16 to float32
        # exactly, so the reconstruction is unaffected — but L4's margins are.
        # A bf16 significand is 8 bits and `q - z` spans [-15, 15], i.e. 4 bits,
        # so the product needs at most 12 significant bits against f16's 11;
        # with F16 scales it needs up to 15. Measured: the narrowing costs
        # 1.6e-5 to 2.3e-5 relative here (margins x4 486 to x6 361) against
        # 1.7e-4 to 1.9e-4 on the 4B (x538 to x625) — the ~8x = 2^3 the three
        # extra significand bits predict.
        by_suffix={
            ("qweight", "I32"): 280,
            ("qzeros", "I32"): 280,
            ("scales", "BF16"): 280,
            ("weight", "BF16"): 163,
        },
        # Iso-perimeter against the base checkpoint.
        # 163 = 40 x (input_layernorm, post_attention_layernorm, q_norm, k_norm)
        #       + model.norm + model.embed_tokens + lm_head.
        # The 80 that differ were checked **by name**, not by count:
        # {'input_layernorm': 40, 'post_attention_layernorm': 40} — exactly the
        # two RMSNorm of each block, the same AWQ fold the 4B shows. The 83
        # identical are the 80 q_norm/k_norm plus those three carried tensors.
        #
        # ⚠️ `tie_word_embeddings` is **false** here and true on the 4B, so the
        # head is a carried tensor in its own right: two 1.556 GB bf16 tables
        # instead of one, both copied and cast by `dequant`. That is why 163 is
        # 40*4 + **3** and not 40*4 + 2, and it is the same untied shape that
        # made `LLVQ_EMBED=q8` decisive on the 8B (CLAUDE.md §3ter).
        carried=163,
        carried_same=83,
        carried_diff=80,
        # sha256 of `tokenizer.json`, measured on all three repositories:
        # **byte-identical** to `Qwen/Qwen3-14B`'s and to the 4B entry's above,
        # 11 422 654 bytes each. So the token fingerprint is common to the 4B
        # arms and the 14B arms by construction, not by luck — the property the
        # campaign assumes when it compares a ppl across models, and the one
        # whose failure would have invalidated the comparison outright.
        tokenizer_sha256=(
            "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
        ),
        # bf16 -> f16 on `model.embed_tokens.weight`, which is the only tensor
        # `control_embedding_narrowing` is ever called on. Counted by streaming
        # that tensor's 1 555 824 640 bytes by Range in 64 MB chunks **through
        # that same function**, never touching the disk: 777 912 320 values,
        # 212 296 changed (2.729e-4), 1 228 flushed, largest touched value
        # 7.5996e-6, largest absolute error 2.9802e-8 = 2^-25, max |v| 1.015625.
        #
        # The mechanism is the 4B's and the numbers obey it: bf16 carries 8
        # significand bits against f16's 11, so the narrowing is exact wherever
        # f16 is normal, and only values under 2^-17 = 7.6294e-6 can move —
        # which is where `touched_max` lands. The **rate** differs from the 4B's
        # (2.729e-4 against 1.981e-4), which is what tells you this was measured
        # and not copied.
        #
        # ⚠️ Two reservations, both of which weaken this lock relative to the
        # 4B's, and neither of which is a reason to leave it out.
        #  1. It has **no independent reference**. `docs/fiche-4b.md` §4.2
        #     measured the 4B's embedding on the base checkpoint, so reproducing
        #     it there tied the cast to a number nobody in this file chose.
        #     Nothing measured this one. What it pins is that the upstream bytes
        #     have not moved — with full coverage, where the perimeter control
        #     only probes 3 MB of the 1.556 GB. It does not corroborate the cast.
        #  2. `lm_head.weight` gets **no** count, because the control looks at
        #     `model.embed_tokens.weight` alone. On this untied model that
        #     leaves half the large carried mass — another 1.556 GB — covered by
        #     three 1 MB probes and nothing else.
        embed_narrowed=212_296,
        embed_flushed=1_228,
    ),
}

# L1's thresholds. The cosine is measured against the **fitted** reference
# `W_base * s`, not against `W_base` — see `control_l1_base` for why, and
# `selftest_pipeline` for the mutant that forced the change.
#
# Calibrated on nine projections spread across the model (`layers.{0,4,9,13,18,
# 22,26,31,35}`, three projection kinds, salience spreads from 1.0 to 4 421)
# plus ten synthetic ones, each measured correct and against three mutants.
# Every number below comes from that table (`selftest_pipeline` rebuilds the
# three mutant columns on every run):
#
#   statistic      correct real   correct synth   no-REV   scales perm  zeros-1
#   cosine floor   0.9930-0.9950  0.9934-0.9941   0.0099   0.9866       0.9262
#   cosine spread  0.0001-0.0007  0.0007-0.0015   0.8936   0.0021       0.0126
#   ratio          0.998 -1.441   0.972 -0.994    8.787    5.534        3.550
#   L2             pass           pass            pass     pass         FAIL
#
# Read the mutant columns downward, because each one dies at a different net and
# no net catches all three:
#   * **no-REV** — the permutation L2 provably cannot see — dies at the
#     **spread**, 89x over threshold. Two populations, ~1 at slots 0 and 7 (the
#     fixed points of `AWQ_REVERSE_ORDER`) and ~0 at the other six.
#   * **scales permuted** dies at the **ratio**, and only there: it leaves no
#     split (spread 0.0021) and barely dents the floor (0.9866). L2 round-trips
#     it, because both sides divide by the same wrong scales.
#   * **zeros-1** dies at **L2**, outright.
# The floor is the net that survives a corrupted yardstick: `ratio` divides by
# `quantization_error`, which is itself computed from `scales`, so a defect that
# corrupts the scales corrupts the denominator too. The floor has no denominator.
#
# ⚠️ **The floor is the one threshold no mutant in the suite exercises**, and
# that is stated rather than hidden: every mutant built so far dies at the
# spread, at the ratio or at L2 first, so widening the floor to -1 fails
# nothing. It is kept as the denominator-free backstop described above, not
# claimed as a lock. Anyone tightening it should build the mutant first.
L1_COSINE_FLOOR = 0.95   # worst correct: 0.9930. no-REV sits at 0.0099
L1_COSINE_SPREAD = 0.010  # worst correct: 0.0015. no-REV sits at 0.8936
# The blunt net behind them. Loose on purpose: AutoAWQ does not only rescale, it
# also searches a per-group **clipping** factor, so the residual legitimately
# exceeds pure round-off — most on the layers whose scales spread widest.
# Measured up to 1.441 on `layers.17.mlp.up_proj`, which is a correct
# reconstruction. The ceiling sits 1.7x above that and 1.4x below the closest
# mutant; an earlier 1.60, calibrated on the 1.18 of `layers.35.mlp.down_proj`
# alone, left 11 % of headroom and would have failed a correct full run.
#
# ⚠️ The ceiling and the floor bound the same quantity from two sides: a
# residual `r` puts the fitted cosine near `sqrt(1 - r^2)`. At `q_err ~ 0.105`,
# ratio 2.5 means `r ~ 0.26`, i.e. a cosine of ~0.965. The floor is set *below*
# that on purpose, so the ratio net fires first and the report never blames a
# permutation for what is only a large residual.
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

    🕳️ **No single field below can be killed by mutation, and it is worth
    knowing why** (CLAUDE.md §5: a surviving mutant means either a weak test or
    dead code — here it means neither, it means *equivalent* code).
    `selftest_packing` proves `repack o unpack = id`, and from that:

    * `qweight_identical` and `mismatched == 0` are the same statement. Given
      `rec` in range, `repack(rec) == qweight` iff `rec == unpack(qweight) ==
      iweights`. Neutralise either and the other still fails the run — but
      neutralise **both** and `selftest_pipeline`'s displaced-nibble mutant
      dies here, which is what proves the pair is not decoration.
    * `qzeros_identical` is a **tautology**: `izeros` *is* `unpack(qzeros)`, so
      re-packing it always returns `qzeros`. It can never fire. It is reported
      because it is a true and cheap statement about the round-trip, not
      because it guards anything — do not count it as a control.
    * `in_range` is implied by the pair whenever the run is correct, and on its
      own it is weaker than it looks: it catches the spurious `-1` only because
      AWQ's min/max grid uses the full 0..15 range, so some nibble overflows.
      A defect that stays inside the range would walk straight through it.

    What actually guards the zero-point convention is `mismatched`: a spurious
    `-1` shifts every recovered integer by one, and `selftest_pipeline` kills
    that mutant here.
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

    🕳️ **The cosine is taken against `W_base * s`, not against `W_base`, and
    the first version of this file got that wrong.** Comparing raw rows makes
    the statistic a function of how widely `s` spreads inside the layer, which
    is a property of the *layer*, not of our reading: measured on the real file,
    the floor tracks the spread from 0.9950 at spread 1.0 to 0.9583 at spread
    26.5, and `layers.9.self_attn.q_proj` spreads by **4 421**. On a synthetic
    projection with that much spread a *correct* reconstruction floors at
    **0.4307** — the old threshold of 0.80 would have called it permuted. The
    same defect the docstring above describes, one level up: a criterion whose
    model omits the very term the method is built on. Dividing `s` back out
    makes the statistic invariant (0.9915 – 0.9950 measured across spreads from
    1.0 to 4 421) and sharpens the permutation gap from ~75x to ~1 400x.
    Pinned offline by `selftest_pipeline`.
    """
    if base_weight.shape != p.weight.shape:
        raise Fatal(f"{p.prefix}: base {base_weight.shape} contre AWQ {p.weight.shape}")
    w = p.weight
    b = np.asarray(base_weight, dtype=np.float32)
    den = (b * b).sum(axis=0)
    s = np.divide((w * b).sum(axis=0), den, out=np.zeros_like(den), where=den > 0)
    fit = b * s
    residual = float(np.linalg.norm(w - fit) / np.linalg.norm(w))
    q_err = p.quantization_error()
    ratio = residual / q_err if 0 < q_err < math.inf else float("inf")

    num = (w * fit).sum(axis=1)
    dn = np.sqrt((w * w).sum(axis=1) * (fit * fit).sum(axis=1))
    cos = np.divide(num, dn, out=np.zeros_like(num), where=dn > 0)
    by_slot = [float(x) for x in cos.reshape(-1, 8).mean(axis=0)]
    lo, hi = min(by_slot), max(by_slot)
    # Two verdicts, deliberately separate, because they accuse different things.
    # A nibble permutation splits the slots into two populations and nothing
    # else does — that is the spread, and it is the *only* thing entitled to
    # print the word "permutation". A uniform loss of alignment across all eight
    # slots is a different crime with different suspects — that is the floor.
    # Folding them into one flag is how a run whose scales moved gets reported
    # as an `AWQ_REVERSE_ORDER` bug and sends the reader to the wrong file.
    permuted = (hi - lo) > L1_COSINE_SPREAD
    degraded = lo < L1_COSINE_FLOOR
    smin = float(s.min())

    return dict(
        ok=bool(not permuted and not degraded and smin > 0.0
                and ratio <= L1_RATIO_CEILING),
        permuted=permuted,
        degraded=degraded,
        residual=residual,
        quantization_error=q_err,
        ratio=ratio,
        cosine_floor=lo,
        cosine_spread=hi - lo,
        scale_min=smin,
        scale_max=float(s.max()),
        salience_spread=float(s.max() / smin) if smin > 0 else float("inf"),
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



class ShardedSafeTensors:
    """`SafeTensors`'s interface over a sharded export, one parser underneath.

    The 4B AWQ ships a single `model.safetensors`; the 8B ships
    `model-0000X-of-0000Y.safetensors` plus an index. This class only routes
    each tensor to its shard, so every downstream lock (structure, L2 repack,
    L1 against the base) exercises the exact bytes it always did.

    `read_at` is deliberately NOT provided: absolute offsets are meaningless
    across shards, and a caller that wants raw ranges must route through
    `shard_for(name)` — a mistake fails loudly instead of reading the wrong
    file.
    """

    def __init__(self, shards: "dict[str, SafeTensors]", weight_map: "dict[str, str]"):
        self._shards = shards
        self._map = weight_map
        self.header: dict = {}
        self.metadata = None
        for st in shards.values():
            self.header.update(st.header)
            if self.metadata is None:
                self.metadata = st.metadata
        self.header_bytes = sum(st.header_bytes for st in shards.values())
        self.total = sum((st.total or 0) for st in shards.values()) or None

    def shard_for(self, name: str) -> "SafeTensors":
        try:
            return self._shards[self._map[name]]
        except KeyError:
            raise Fatal(f"tenseur absent de l'index : {name}") from None

    def entry(self, name: str) -> dict:
        return self.shard_for(name).entry(name)

    def span(self, name: str) -> tuple[int, int]:
        return self.shard_for(name).span(name)

    def declared_bytes(self, name: str) -> int:
        return self.shard_for(name).declared_bytes(name)

    def tensor(self, name: str):
        return self.shard_for(name).tensor(name)

    def raw(self, name: str) -> bytes:
        return self.shard_for(name).raw(name)

    def raw_many(self, names, budget: int = 64_000_000) -> dict[str, bytes]:
        by_shard: dict[str, list] = {}
        for n in names:
            by_shard.setdefault(self._map[n], []).append(n)
        out: dict[str, bytes] = {}
        for group in by_shard.values():
            out.update(self.shard_for(group[0]).raw_many(group, budget))
        return out

    def close(self) -> None:
        for st in self._shards.values():
            st.close()


def open_awq_remote(repo: str, revision: str):
    """The AWQ file over HTTP Range — single-file first, sharded fallback."""
    try:
        return SafeTensors(hub_url(repo, "model.safetensors", revision))
    except Exception as e:
        if "404" not in str(e):
            raise
    idx = hub_json(repo, "model.safetensors.index.json", revision)
    wmap = idx["weight_map"]
    shards = {
        f: SafeTensors(hub_url(repo, f, revision))
        for f in sorted(set(wmap.values()))
    }
    return ShardedSafeTensors(shards, wmap)


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
        st = awq.shard_for(n) if hasattr(awq, "shard_for") else awq
        off0, span = st.span(n)
        ok = True
        for frac in (0.0, 0.5, 1.0):
            off = min(int(span * frac), max(0, span - probe_bytes))
            off -= off % 2
            n_read = min(probe_bytes, span - off)
            ok &= st.read_at(off0 + off, n_read) == base.slice_bytes(n, off, n_read)
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
        say("  ok    reproduit fiche-4b §4.2 au comptage près, sur les octets "
            "d'AWQ")
        return True
    say(f"  FAIL  attendu {exp['embed_narrowed']} changées et "
        f"{exp['embed_flushed']} à zéro (fiche-4b §4.2, recompté sur les octets "
        "d'AWQ).")
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


# --- the offline lock -------------------------------------------------------
#
# `selftest_packing` pins the bit arithmetic and nothing else. Everything the
# reconstruction actually rests on — the group reshape, the zero-point
# convention, the transpose, the fact that `scales` are stored **unpermuted**
# while `qweight` and `qzeros` are permuted — was, until this section existed,
# asserted only by a comment and by controls nobody had ever seen fail.
#
# A control that has never rejected anything is a control whose failure mode is
# unknown (CLAUDE.md §5). So: synthesize a projection whose ground truth we
# chose, run the real controls on the real code, and demand that each named
# mutant dies at the named control. No network, no model, deterministic.


class _MemoryTensors:
    """A `SafeTensors` stand-in over arrays already in memory.

    `Projection` touches its source through `tensor(name)` and nothing else, so
    the selftest exercises the production class rather than a copy of it.
    """

    def __init__(self, tensors: dict[str, np.ndarray]):
        self._t = tensors

    def tensor(self, name: str) -> np.ndarray:
        return self._t[name]


def _pack_with(mat: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Pack nibbles under an arbitrary intra-word order, for building mutants."""
    rows, n = mat.shape
    g = mat.reshape(rows, n // 8, 8)[:, :, order].astype(np.uint32)
    out = np.zeros((rows, n // 8), dtype=np.uint32)
    for i in range(8):
        out |= (g[:, :, i] & np.uint32(0xF)) << np.uint32(4 * i)
    return out.view(np.int32)


def _synthetic_awq(
    d_out: int = 64,
    d_in: int = 512,
    group_size: int = 128,
    seed: int = 7,
    *,
    order: np.ndarray | None = None,
    scale_order: np.ndarray | None = None,
) -> tuple[_MemoryTensors, np.ndarray]:
    """AWQ tensors for a projection whose unquantized weight we know.

    Reproduces what AutoAWQ writes: scale the input channels by a salience
    profile `s`, quantize `(W*s)^T` asymmetrically per group of `group_size`
    input channels, then pack `qweight`/`qzeros` under `AWQ_ORDER` while leaving
    `scales` in natural order.

    The salience spread is deliberately brutal — ~3 000x end to end — because
    the real file carries a layer at **4 421** (`layers.9.self_attn.q_proj`) and
    a threshold that survives only mild spreads is a threshold that fails in
    production, not in the selftest.

    It is built per **group**, with mild jitter inside, and that shape is not
    cosmetic. A salience that swings by 3 000x *within* one group of 128 input
    channels lets the loud channel set the group's scale and quantizes the quiet
    ones to a constant; the least-squares `s` of the quiet channels then comes
    back exactly 0, and L1 rejects a reconstruction that is in fact correct.
    Real salience comes from activation magnitudes and is channel-structured,
    which is why the measured `scale_min` is 0.19 and 0.66, not 0. Keeping the
    global spread and dropping the intra-group one reproduces what stresses the
    statistic without manufacturing a degeneracy the file will never see.

    Output rows are given a magnitude of their own, and that is not cosmetic
    either: it is what a permutation of the **scales** damages. Rows of iid
    gaussians all have the same dynamic range, so swapping the scales of eight
    neighbouring output channels costs almost nothing and that mutant survives
    with a cosine spread of 0.0099 against a threshold of 0.010 — passing by
    luck. Measured on the real checkpoint, row norms spread by 7.6x on
    `layers.0.self_attn.k_proj` and 9.7x on `layers.18.self_attn.q_proj`; the
    `exp(U(-1.1, 1.1))` below reproduces that 9x.

    `order` / `scale_order` are the mutation handles: they change what the
    *file* contains, not what our reader does, which is the only honest way to
    ask "would we notice?".
    """
    order = AWQ_ORDER if order is None else order
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((d_out, d_in)).astype(np.float32)
    base *= np.exp(rng.uniform(-1.1, 1.1, size=(d_out, 1))).astype(np.float32)
    groups_of_s = np.repeat(
        rng.uniform(-4.0, 4.0, size=d_in // group_size), group_size
    )
    s = np.exp(groups_of_s + rng.uniform(-0.35, 0.35, size=d_in)).astype(np.float32)
    target = np.ascontiguousarray((base * s).T)  # [d_in, d_out], what AWQ sees
    groups = d_in // group_size
    g = target.reshape(groups, group_size, d_out)
    lo, hi = g.min(axis=1), g.max(axis=1)
    # f16 round-trip on the scales, as the file stores them.
    sc = ((hi - lo) / 15.0).astype(np.float16).astype(np.float32)
    izeros = np.rint(-lo / sc).astype(np.int32).clip(0, 15)
    iweights = (
        np.rint(g / sc[:, None, :]).astype(np.int32) + izeros[:, None, :]
    ).clip(0, 15).reshape(d_in, d_out)

    scales = sc.astype(np.float16)
    if scale_order is not None:
        scales = scales.reshape(groups, d_out // 8, 8)[:, :, scale_order].reshape(
            groups, d_out
        )
    return _MemoryTensors({
        "t.qweight": _pack_with(iweights, order),
        "t.qzeros": _pack_with(izeros, order),
        "t.scales": scales,
    }), base


def selftest_pipeline() -> list[str]:
    """The reconstruction and its controls, against four named mutants."""
    fail: list[str] = []
    gs = 128

    src, base = _synthetic_awq(group_size=gs)
    p = Projection(src, "t", gs)
    # The reconstruction must be the transpose, [d_out, d_in]. Checked first:
    # every control below would raise on a mismatched shape, and a stack trace
    # is a worse report than a sentence.
    if p.weight.shape != base.shape:
        return [f"forme reconstruite {p.weight.shape}, attendu {base.shape} — "
                "la transposition finale a sauté"]
    l2, l4 = control_l2(p), control_l4(p)
    l1 = control_l1_base(p, base)
    if not l2["ok"]:
        fail.append(f"L2 rejette une reconstruction correcte : {l2}")
    if not l4["ok"]:
        fail.append(f"L4 rejette une reconstruction correcte : marge {l4['margin']}")
    if not l1["ok"]:
        fail.append(
            f"L1 rejette une reconstruction correcte (plancher {l1['cosine_floor']:.4f},"
            f" écart {l1['cosine_spread']:.4f}, ratio {l1['ratio']:.3f}) — c'est la"
            f" faute que le seuil brut avait déjà commise, cf. control_l1_base"
        )
    # The reconstruction must be the transpose: [d_out, d_in], not [d_in, d_out].
    if p.weight.shape != base.shape:
        fail.append(f"forme reconstruite {p.weight.shape}, attendu {base.shape}")

    # Mutant 1 — the file packed WITHOUT `AWQ_ORDER`, i.e. our reverse gather is
    # spurious. This is the exact failure the module docstring forbids, and the
    # exact one L2 cannot see: it must round-trip, and L1 must catch it.
    src2, base2 = _synthetic_awq(group_size=gs, order=np.arange(8))
    p2 = Projection(src2, "t", gs)
    if not control_l2(p2)["ok"]:
        fail.append("L2 voit la permutation de quartets — l'énoncé du fichier, "
                    "qui affirme qu'il ne peut pas la voir, est faux")
    m = control_l1_base(p2, base2)
    if m["ok"] or not m["permuted"]:
        fail.append(
            f"L1 laisse passer un AWQ_REVERSE_ORDER non appliqué "
            f"(plancher {m['cosine_floor']:.4f}, écart {m['cosine_spread']:.4f})"
        )

    # Mutant 2 — `scales` permuted like `qweight`. Nothing else pins the fact
    # that AutoAWQ's `apply_order` skips the scales: L2 round-trips it, because
    # both sides divide by the same wrong scales. It must be caught, and it must
    # NOT be reported as a nibble permutation — it leaves no split (0.0021).
    src3, base3 = _synthetic_awq(group_size=gs, scale_order=AWQ_ORDER)
    p3 = Projection(src3, "t", gs)
    if not control_l2(p3)["ok"]:
        fail.append("L2 voit des échelles permutées — l'énoncé selon lequel il "
                    "les traverse est faux, et le rôle du ratio est à revoir")
    m = control_l1_base(p3, base3)
    if m["ok"]:
        fail.append(f"L1 laisse passer des échelles permutées : ratio {m['ratio']:.3f}")
    if m["permuted"]:
        fail.append("L1 impute des échelles permutées à AWQ_REVERSE_ORDER : le "
                    "rapport enverrait le lecteur dans le mauvais fichier")

    # Mutant 3 — the spurious `-1` on the zeros: `(iw - (iz-1))*s = w + s`.
    p4 = Projection(_synthetic_awq(group_size=gs)[0], "t", gs)
    p4.w_in_major = p4.w_in_major + np.repeat(p4.scales, gs, axis=0)
    p4.weight = np.ascontiguousarray(p4.w_in_major.T)
    p4._f16 = None
    if control_l2(p4)["ok"]:
        fail.append("L2 laisse passer le -1 parasite sur les zéros")

    # Mutant 3b — two input rows of the same group swapped. Nibbles all legal,
    # simply not the ones the file holds.
    #
    # 🕳️ Without this one, `mismatched` and `qweight_identical` are BOTH
    # unkillable: neutralise the pair and mutant 3 still dies, because a `-1`
    # on the zeros pushes some nibble past 15 and `in_range` catches it. That
    # is luck of the data, not a proof — AWQ's min/max grid happens to use the
    # full 0..15 range. A defect that stays inside the range would walk through
    # an L2 whose only live field is `in_range`.
    p4b = Projection(_synthetic_awq(group_size=gs)[0], "t", gs)
    p4b.w_in_major = p4b.w_in_major.copy()
    p4b.w_in_major[[0, 1]] = p4b.w_in_major[[1, 0]]
    p4b.weight = np.ascontiguousarray(p4b.w_in_major.T)
    p4b._f16 = None
    r = control_l2(p4b)
    if r["ok"]:
        fail.append("L2 laisse passer des quartets déplacés dans [0, 15]")
    if not r["in_range"]:
        fail.append("le mutant de quartets déplacés sort de [0, 15] : il ne "
                    "teste plus `mismatched`, il teste `in_range`")

    # The output dtype is a campaign requirement, not a preference (plan §2.6:
    # "calculer en float32, un seul .astype(float16) en sortie"). Nothing else
    # in this file would notice an f32 output: L4 would report a relative error
    # of 0 and an infinite margin, the weight count is a count of weights, and
    # step [7] re-reads names. The checkpoint would simply be twice the size and
    # perfectly plausible.
    if p.f16.dtype != np.float16:
        fail.append(f"la sortie est en {p.f16.dtype}, f16 imposé par le protocole")

    # Mutant 4 — an f16 overflow, and mutant 5 — one rounding too many (a bf16
    # detour). Without these two, `L4_MIN_MARGIN` and the overflow test are
    # decoration: every correct projection clears them by x500, so loosening
    # either threshold to zero fails nothing.
    p6 = Projection(_synthetic_awq(group_size=gs)[0], "t", gs)
    p6.weight = p6.weight * np.float32(1e5)  # past f16's 65 504
    p6.scales = p6.scales * np.float32(1e5)  # keep the yardstick consistent
    p6._f16 = None
    # The overflow is the point of this mutant, so numpy's warning about it is
    # not news. Silenced here and nowhere else: a real run must still shout.
    with np.errstate(over="ignore", invalid="ignore"):
        r = control_l4(p6)
    if r["ok"] or r["overflow"] == 0:
        fail.append(f"L4 accepte un débordement f16 : {r['overflow']} valeurs")
    p7 = Projection(_synthetic_awq(group_size=gs)[0], "t", gs)
    p7._f16 = (
        (p7.weight.view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32)
    ).astype(np.float16)
    r = control_l4(p7)
    if r["ok"]:
        fail.append(f"L4 accepte un arrondi supplémentaire : marge {r['margin']:.1f}")

    # Mutant 6 — the transpose dropped, and mutant 7 — the wrong group size.
    # Both must be structural refusals, not silent degradations.
    p5 = Projection(_synthetic_awq(group_size=gs)[0], "t", gs)
    p5.weight = p5.w_in_major
    try:
        control_l1_base(p5, base)
        fail.append("L1 accepte une reconstruction non transposée")
    except Fatal:
        pass
    try:
        Projection(_synthetic_awq(group_size=gs)[0], "t", gs // 2)
        fail.append("Projection accepte un group_size faux")
    except Fatal:
        pass
    return fail


def selftest_narrowing() -> list[str]:
    """bf16 -> f16: exact above 2^-17, and rounding below it.

    `control_embedding_narrowing` reports a count the campaign compares to a
    reference. If the cast were exact everywhere the count would be 0 and the
    comparison would look like agreement, so the mechanism is pinned here.
    """
    fail: list[str] = []
    # Every bf16 (8-bit significand) at or above 2^-17 lands on f16's grid.
    e = np.arange(-17, 8)
    k = np.arange(128)
    v = ((128 + k[None, :]) * np.float64(2.0) ** (e[:, None] - 7)).astype(np.float32)
    r = control_embedding_narrowing(v.reshape(-1))
    if r["changed"] != 0:
        fail.append(f"{r['changed']} valeurs bf16 >= 2^-17 changent au cast f16")
    # One exponent lower, the odd significands need a bit f16 does not have.
    v2 = ((128 + k) * np.float64(2.0) ** -25).astype(np.float32)
    r2 = control_embedding_narrowing(v2)
    if r2["changed"] != 64:
        fail.append(f"{r2['changed']} valeurs changent à 2^-18, 64 attendues")
    if abs(r2["abs_error_max"] - 2.0**-25) > 1e-12:
        fail.append(f"erreur max {r2['abs_error_max']}, 2^-25 attendu")
    if control_embedding_narrowing(np.zeros(4, np.float32))["flushed_to_zero"]:
        fail.append("des zéros exacts sont comptés comme écrasés à zéro")
    return fail


def selftest_weight_count() -> list[str]:
    """`expected_weight_count` against the number the campaign publishes."""
    fail: list[str] = []
    qwen3_4b = dict(
        num_hidden_layers=36, hidden_size=2560, intermediate_size=9728,
        head_dim=128, num_attention_heads=32, num_key_value_heads=8,
        vocab_size=151936, tie_word_embeddings=True,
    )
    got = expected_weight_count(qwen3_4b)
    if got != 4_022_468_096:
        fail.append(f"Qwen3-4B compté à {got}, 4 022 468 096 attendus (plan §2.1)")
    # The 252 projections alone, the homogeneous denominator of the same table.
    if got - 151936 * 2560 - 2560 - 36 * (2 * 2560 + 2 * 128) != 3_633_315_840:
        fail.append("les projections seules ne retombent pas sur 3 633 315 840")
    untied = expected_weight_count(dict(qwen3_4b, tie_word_embeddings=False))
    if untied - got != 151936 * 2560:
        fail.append("un lm_head délié ne coûte pas une embedding de plus")
    return fail


def selftest_writer() -> list[str]:
    """`ShardWriter` round-trip: shard split, index, and values re-read."""
    import shutil
    import tempfile

    fail: list[str] = []
    tmp = Path(tempfile.mkdtemp(prefix="llvq-awq-selftest-"))
    try:
        rng = np.random.default_rng(3)
        want = {f"t{i}": rng.standard_normal((64, 32)).astype(np.float16)
                for i in range(5)}
        # 8 KB each, so a 20 KB limit must produce more than one shard.
        w = ShardWriter(tmp, 20_000)
        for k, v in want.items():
            w.add(k, v)
        files, index = w.finish()
        if len(files) < 2:
            fail.append(f"{len(files)} shard(s) pour 5 tenseurs sous une limite "
                        "de 20 000 o : le découpage ne se déclenche pas")
        if w.weights != sum(v.size for v in want.values()):
            fail.append(f"{w.weights} poids comptés, {sum(v.size for v in want.values())} écrits")
        if set(index["weight_map"]) != set(want):
            fail.append("l'index n'annonce pas les tenseurs écrits")
        if len(files) > 1 and not (tmp / "model.safetensors.index.json").exists():
            fail.append("index.json absent d'un checkpoint multi-shard")
        seen: dict[str, np.ndarray] = {}
        for f in files:
            if not (tmp / f).exists():
                fail.append(f"shard annoncé mais absent : {f}")
                continue
            with SafeTensors(tmp / f) as st:
                for n in st.header:
                    seen[n] = st.tensor(n)
        for k, v in want.items():
            if k not in seen or not np.array_equal(seen[k], v):
                fail.append(f"{k} ne se relit pas à l'identique")
        if any(p.name.startswith(".shard-") for p in tmp.iterdir()):
            fail.append("des shards temporaires survivent à finish()")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return fail


SELFTESTS = (
    ("arithmétique d'empaquetage", selftest_packing,
     "ordres mutuellement inverses, pack/unpack contre référence naïve, "
     "empaquetage non permuté rejeté"),
    ("reconstruction et contrôles", selftest_pipeline,
     "L2/L1/L4 acceptent une projection correcte à saillance extrême et tuent "
     "7 mutants (quartets, échelles, zéros, débordement f16, arrondi de trop, "
     "transposée, group_size) ; la sortie est bien en f16"),
    ("narrowing bf16 -> f16", selftest_narrowing,
     "exact au-dessus de 2^-17, arrondi de 2^-25 en dessous"),
    ("comptabilité des poids", selftest_weight_count,
     "Qwen3-4B retombe sur 4 022 468 096 et 3 633 315 840"),
    ("écriture des shards", selftest_writer,
     "découpage, index et relecture bit pour bit"),
)


def run_selftest(say) -> bool:
    """Every offline lock. No network, no model, no bytes read."""
    say("\n[0] verrous hors ligne (aucun octet lu)")
    ok = True
    for name, fn, blurb in SELFTESTS:
        failures = fn()
        for line in failures:
            say(f"  FAIL  {name} : {line}")
        if not failures:
            say(f"  ok    {name} — {blurb}")
        ok &= not failures
    return ok


def selftest_failures() -> dict[str, list[str]]:
    """The same locks, as data for the reconstruction report."""
    return {name: fn() for name, fn, _ in SELFTESTS}


def resolve_revision(args) -> list[str]:
    """Fill in the revisions the command line left out, from `EXPECTED`.

    A revision belongs to a repository, not to an invocation. While the 4B's
    SHA was the argparse default, `check --awq-repo Qwen/Qwen3-14B-AWQ` asked
    the Hub for a commit that repository has never had and died on a bare 404 —
    before printing a single control. That failure has a bad shape: the obvious
    reaction is to reach for `--allow-unknown-repo`, which turns *off* the
    structure, tokenizer and perimeter controls. So the pin travels with the
    entry, and the fallback for a repository we know nothing about is `main`,
    reported rather than assumed.
    """
    exp = EXPECTED.get(args.awq_repo)
    unpinned: list[str] = []
    if args.awq_revision is None:
        args.awq_revision = exp["awq_revision"] if exp else "main"
        if exp is None:
            unpinned.append("--awq-revision")
    if args.base_revision is None:
        # Only when the entry names *this* base repository: an entry pins a
        # pair, and lending the 14B's base SHA to `Qwen/Qwen3-4B` would resolve
        # to a commit of the wrong model.
        if exp is not None and args.base_repo == exp["base_repo"]:
            args.base_revision = exp["base_revision"]
        else:
            args.base_revision = "main"
            unpinned.append("--base-revision")
    return unpinned


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


def cmd_selftest(args) -> int:
    """Every offline lock, in about a second. No network, no model, no bytes.

    Same role as `ops/run.py selftest`: the thing you run before believing any
    other output of this file, and the thing to run after touching it.
    """
    ok = run_selftest(print)
    print("\n" + ("TOUS LES VERROUS TIENNENT" if ok else "AU MOINS UN VERROU CÈDE"))
    return 0 if ok else 1


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
    awq = open_awq_remote(args.awq_repo, args.awq_revision)
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
                f" (étendue x{l1['salience_spread']:,.0f})"
                f"  cosinus par position mod 8 {l1['cosine_by_slot']}")
            if l1["permuted"]:
                say("          les positions se séparent en deux populations : "
                    "c'est la signature\n          d'un AWQ_REVERSE_ORDER non "
                    "appliqué, et rien d'autre ne la produit")
            elif not l1["ok"]:
                say("          pas de signature de permutation : les huit "
                    "positions bougent ensemble.\n          La lecture des "
                    "quartets est donc probablement juste (voir L2), mais\n"
                    "          l'alignement avec le checkpoint de base ne tient "
                    "plus — échelles\n          déplacées, révision amont "
                    "différente. À examiner avant de mesurer.")
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
    report["controls"]["selftest"] = dict(ok=ok, failures=selftest_failures())
    if not ok:
        say("\nabandon : un verrou hors ligne cède, rien de ce qui suit n'a de sens")
        return 1

    cfg = hub_json(args.awq_repo, "config.json", args.awq_revision)
    bits, gs = quant_config(cfg)
    n_layers = cfg["num_hidden_layers"]
    exp, allowed = resolve_expectations(args.awq_repo, args.allow_unknown_repo, say)
    if not allowed:
        return 2

    # Everything `config.json` alone can settle is settled now, before an hour
    # of arithmetic. `expected_weight_count` reads keys this file does not
    # otherwise touch; discovering at step [6] that one is missing would throw
    # away the whole run.
    want_weights = expected_weight_count(cfg)
    say(f"\n  cible : {want_weights} poids d'après config.json")

    # Stale outputs are the quiet way to publish the wrong model, and there are
    # two of them.
    #
    # 1. A previous run that produced a single `model.safetensors` leaves it
    #    behind when this one produces three shards. `Checkpoint::fetch` tries
    #    `model.safetensors` first and only then the index, so the campaign
    #    would score the *old* weights while every control printed below passed
    #    on the new ones — and `upload_folder` would push both.
    # 2. A `RECONSTRUCTION.json` from a run that passed outlives a run that
    #    crashes halfway. `push` reads that file and nothing else, so it would
    #    green-light a directory of half-written shards.
    #
    # Both are cleared here rather than at the end, because the failure mode is
    # a run that never reaches the end.
    stale = sorted(
        p
        for p in out.iterdir()
        if p.is_file()
        and (
            (p.name.startswith("model") and p.suffix in (".safetensors", ".json"))
            or p.name.startswith(".shard-")
            or p.name in ("RECONSTRUCTION.json", "README.md")
        )
    )
    for p in stale:
        p.unlink()
    if stale:
        say(f"  {len(stale)} fichier(s) d'un run précédent effacé(s) : "
            f"{', '.join(p.name for p in stale[:4])}"
            + (" …" if len(stale) > 4 else ""))

    if args.awq_file:
        awq = SafeTensors(Path(args.awq_file))
    else:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError

        say(f"\n[1] téléchargement de {args.awq_repo} "
            f"(~{(exp or {}).get('bytes', 0) / 1e9:.2f} Go)")
        try:
            awq = SafeTensors(Path(hf_hub_download(
                repo_id=args.awq_repo,
                filename="model.safetensors",
                revision=args.awq_revision,
            )))
        except EntryNotFoundError:
            # Sharded export (the 8B): pull the index, then every shard, and
            # route through the same parser — the locks below do the proving.
            idx = hub_json(args.awq_repo, "model.safetensors.index.json",
                           args.awq_revision)
            wmap = idx["weight_map"]
            shards = {
                f: SafeTensors(Path(hf_hub_download(
                    repo_id=args.awq_repo, filename=f,
                    revision=args.awq_revision,
                )))
                for f in sorted(set(wmap.values()))
            }
            awq = ShardedSafeTensors(shards, wmap)
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
    waive_l1 = {n for n in (args.waive_l1 or "").split(",") if n}
    # A waiver names a tensor the sampler might not draw: force-draw it, so
    # the derogation is measured and written down rather than silently unused.
    picks |= waive_l1
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
            waived = (not l1["ok"]) and prefix in waive_l1 and l2["ok"]
            l1_all.append(dict(name=prefix, waived=waived, **l1))
            line += (
                f"  L1 {l1['ratio']:.3f} "
                + ("DÉROGÉ (--waive-l1)" if waived else ("ok" if l1["ok"] else "FAIL"))
            )
            ok &= l1["ok"] or waived
            say(line)
            if not l1["ok"]:
                say(f"      cosinus par position mod 8 : {l1['cosine_by_slot']}")
                say("      deux populations = AWQ_REVERSE_ORDER non appliqué"
                    if l1["permuted"]
                    else "      les huit positions bougent ensemble : ce n'est pas "
                         "une permutation de quartets, mais l'alignement avec le "
                         "checkpoint de base ne tient plus")
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
        ok=bool(l1_all) and all(r["ok"] or r.get("waived") for r in l1_all),
        tested=len(l1_all),
        waived=[r["name"] for r in l1_all if r.get("waived")],
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
    dtypes: dict[str, int] = {}
    for f in files:
        with SafeTensors(out / f) as st:
            seen |= set(st.header)
            for n, e in st.header.items():
                dtypes[e["dtype"]] = dtypes.get(e["dtype"], 0) + 1
    names_ok = seen == set(index["weight_map"])
    # f16 is a protocol requirement (plan §2.6), and it is the one property no
    # control upstream can see: an f32 output would give L4 a relative error of
    # 0 and an infinite margin, and the weight count counts weights. Two bytes
    # per weight, over the whole checkpoint, says it in one line.
    dtype_ok = set(dtypes) == {"F16"}
    size_ok = writer.total == 2 * writer.weights
    reread_ok = names_ok and dtype_ok and size_ok
    say(f"  {'ok   ' if names_ok else 'FAIL '} {len(seen)} tenseurs relus sur "
        f"{len(index['weight_map'])} annoncés par l'index")
    say(f"  {'ok   ' if dtype_ok else 'FAIL '} dtypes écrits : "
        f"{dict(sorted(dtypes.items()))} (F16 seul exigé)")
    say(f"  {'ok   ' if size_ok else 'FAIL '} {writer.total} octets pour "
        f"{writer.weights} poids, soit {writer.total / max(writer.weights, 1):.3f} "
        "o/poids (2,000 exigés)")
    ok &= reread_ok
    report["controls"]["reread"] = dict(
        ok=reread_ok, tensors=len(seen), dtypes=dict(sorted(dtypes.items())),
        bytes_per_weight=writer.total / max(writer.weights, 1),
    )

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

    # The report certifies bytes, not a directory. Between `dequant` and here a
    # shard can have been re-written, truncated by a full disk, or replaced by a
    # run against another repository — and every one of those leaves a green
    # report sitting next to different weights. Re-hash before uploading: the
    # controls above are only worth what the files still are.
    waived = report.get("controls", {}).get("l1", {}).get("waived", [])
    if waived:
        print("\n⚠️  DÉROGATION L1 — à reproduire dans toute publication de ces poids :")
        for w in waived:
            row = next((r for r in report["controls"]["l1"]["detail"]
                        if r["name"] == w), {})
            print(f"    {w} : ratio {row.get('ratio')}, L2 identique au fichier AWQ —"
                  "\n    l'alignement au checkpoint de base ne tient pas sur ce tenseur"
                  " (propriété de l'export amont).")
    print(f"\nvérification des octets contre {rpath.name}")
    drift: list[str] = []
    listed = report.get("output", {}).get("files", [])
    if not listed:
        drift.append("le rapport ne liste aucun fichier")
    for entry in listed:
        f = out / entry["name"]
        if not f.exists():
            drift.append(f"{entry['name']} : absent")
            continue
        if f.stat().st_size != entry["bytes"]:
            drift.append(
                f"{entry['name']} : {f.stat().st_size} o, {entry['bytes']} certifiés"
            )
            continue
        if sha256_file(f) != entry["sha256"]:
            drift.append(f"{entry['name']} : sha256 différent")
    extra = sorted(
        p.name
        for p in out.iterdir()
        if p.is_file()
        and p.suffix == ".safetensors"
        and p.name not in {e["name"] for e in listed}
    )
    if extra:
        drift.append(f"poids non certifiés dans le répertoire : {extra}")
    if drift:
        print("refus : le répertoire ne correspond plus au rapport."
              "\n  " + "\n  ".join(drift)
              + "\nRelance `dequant` : on ne publie pas des octets qu'aucun "
                "contrôle n'a vus.", file=sys.stderr)
        return 1
    print(f"  ok    {len(listed)} fichiers, sha256 et tailles conformes")

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
        sp.add_argument("--awq-revision", default=None,
                        help="défaut : la révision épinglée dans EXPECTED pour "
                             "ce dépôt, `main` s'il n'y en a pas")
        sp.add_argument("--base-repo", default=BASE_REPO)
        sp.add_argument("--base-revision", default=None,
                        help="défaut : la révision de base épinglée par la même "
                             "entrée, `main` si elle ne nomme pas ce dépôt")
        sp.add_argument("--full-embed", action="store_true",
                        help="comparer l'embedding en entier (778 Mo) au lieu de "
                             "trois sondages")
        sp.add_argument("--allow-unknown-repo", action="store_true",
                        help="accepter un dépôt sans attentes enregistrées")

    s = sub.add_parser("selftest", help="les verrous hors ligne, ~1 s, sans réseau")
    s.set_defaults(fn=cmd_selftest)

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
    d.add_argument("--waive-l1", default="",
                   help="tenseurs (séparés par des virgules) dont un échec L1 "
                        "est DÉROGÉ : le L2 (octets du fichier AWQ) doit rester "
                        "identique, la dérogation est écrite dans le rapport et "
                        "criée par `push` — pour un export amont dont "
                        "l'alignement au checkpoint de base ne tient pas sur un "
                        "tenseur isolé")
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
    if hasattr(args, "awq_repo"):  # `push` reads its revisions from the report
        for flag in resolve_revision(args):
            print(f"⚠️  {flag} non épinglé : lecture sur `main`, donc ce run "
                  "n'est pas reproductible dans le temps")
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
