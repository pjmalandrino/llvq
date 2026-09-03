#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""LRU simulator over a temporal routing trace. P2, §2.4 and V0.4.

Answers a question the aggregated dump cannot ask: **the temporal locality
of MoE routing**. The 2026-08-12 dump is aggregated per cell (layer,
expert), so it only bounds a *static oracle*, the best choice of a frozen
set of hot cells. An LRU can beat that oracle if locality exists, and the
scissor says so itself
(`ops/moe_ciseau.py:445-450`: "settling it would take a TEMPORAL dump, which
does not exist").

**What this script measures**: `hit_LRU(decode order)`, the rate of cells
already resident when the layer needs them, in single-sequence decode
(batch 1), one global cache over the (layer, expert) cells.

**What it does not measure**: time. A hit is not free, a miss costs a PCIe
copy, and neither of the two is timed here. The script counts events. Section
4.0 of the preregistration is what converts them into a budget.

Stdlib only: no GPU, no network, no dependency. It runs on the trace, not on
the model.

Usage:
    python3 ops/moe_lru.py --trace ~/llvq-moe/trace-....u8
    python3 ops/moe_lru.py --selftest        # the five V0.4 cases, no trace
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import OrderedDict


# --------------------------------------------------------------------------
# The simulator
# --------------------------------------------------------------------------


class Lru:
    """True LRU: admission on miss, **recency refresh on hit**, eviction of the
    least recently used.

    The refresh on hit is the load-bearing parameter. It is the only mechanism
    by which an LRU can beat a static oracle. A cache that orders only by
    insertion date is a FIFO, and V0.4 cases 1 to 3 do not tell the two apart.
    Only case 4 does.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 0:
            raise ValueError(f"negative capacity: {capacity}")
        self.capacity = capacity
        self.slots: OrderedDict[int, None] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.compulsory = 0  # first accesses: unavoidable whatever the size
        self._seen: set[int] = set()

    def touch(self, cell: int) -> bool:
        """Access `cell`. Returns True if it was resident."""
        if cell in self.slots:
            self.slots.move_to_end(cell)  # ← the refresh, see the docstring
            self.hits += 1
            return True
        self.misses += 1
        if cell not in self._seen:
            self._seen.add(cell)
            self.compulsory += 1
        if self.capacity == 0:
            return False
        self.slots[cell] = None
        if len(self.slots) > self.capacity:
            self.slots.popitem(last=False)
        return False

    @property
    def total(self) -> int:
        return self.hits + self.misses

    def rate(self) -> float:
        return self.hits / self.total if self.total else 0.0


def simulate(stream, capacity: int) -> Lru:
    c = Lru(capacity)
    for cell in stream:
        c.touch(cell)
    return c


def decode_order(trace: bytes, n_tokens: int, n_layers: int, top_k: int, n_experts: int):
    """The cells in DECODE order: `for t, for ℓ, the K`.

    This is not the execution order of the dump. A prefill processes every
    token of a window at layer 0, then at layer 1. This is the order a cache
    would work in during generation, the regime of package A. The argument that
    allows the transposition (causal router, so the same decisions per token)
    is an ARGUMENT, not a measurement. It is declared in §2.4 of the
    preregistration, and not verified.

    A cell is `ℓ * n_experts + e`, the binding unit, never the expert alone
    (the aggregation trap of §1).
    """
    stride = n_layers * top_k
    for t in range(n_tokens):
        base = t * stride
        for l in range(n_layers):
            off = base + l * top_k
            for k in range(top_k):
                yield l * n_experts + trace[off + k]


# --------------------------------------------------------------------------
# V0.4: the five cases, to pass BEFORE reading a line of trace
# --------------------------------------------------------------------------


def selftest() -> int:
    """The five cases of §V0.4. Returns 0 if all pass."""
    fails = []

    def check(name, got, want, why):
        ok = abs(got - want) < 1e-9 if isinstance(want, float) else got == want
        print(f"  [{'ok ' if ok else 'FAIL'}] {name}: {got} (expected {want})")
        if not ok:
            fails.append(f"{name}: {why}")

    print("V0.4: the simulator, on synthetic streams with a known answer\n")

    # 1. working set <= size ⇒ 100 % outside the compulsory misses.
    c = simulate([0, 1, 2] * 100, capacity=3)
    check("1. working set fits in cache",
          c.hits, 300 - 3,
          "a cache that holds everything must keep everything")
    check("1bis. compulsory misses counted", c.compulsory, 3,
          "the first accesses are unavoidable and must be counted apart")

    # 2. round robin over size + 1 ⇒ 0 hit. The worst case of an LRU.
    c = simulate([0, 1, 2, 3] * 50, capacity=3)
    check("2. round robin over size+1", c.hits, 0,
          "each cell is evicted just before it is asked for again")

    # 3. flat i.i.d. ⇒ hit ≈ size / cells. Deterministic generator.
    n_cells, cap, n = 100, 25, 200_000
    x = 12345
    flat = []
    for _ in range(n):
        x = (1103515245 * x + 12345) & 0x7FFFFFFF
        flat.append(x % n_cells)
    c = simulate(flat, capacity=cap)
    got, want = c.rate(), cap / n_cells
    ok3 = abs(got - want) < 0.02
    print(f"  [{'ok ' if ok3 else 'FAIL'}] 3. flat i.i.d.: {got:.4f} (expected ≈ {want:.4f})")
    if not ok3:
        fails.append("3. flat i.i.d.: on a source with no locality an LRU is worth its relative size")

    # 4. THE case that kills the FIFO. A B A C A, size 2.
    #    LRU : A(miss) B(miss) A(HIT, refreshes A) C(miss, evicts B) A(HIT) = 2.
    #    FIFO: A(miss) B(miss) A(HIT)  C(miss, evicts A) A(miss)            = 1.
    c = simulate([0, 1, 0, 2, 0], capacity=2)
    check("4. A B A C A, size 2 (kills the FIFO)", c.hits, 2,
          "with no refresh on hit this is a FIFO, which would return 1, and then "
          "no LRU could beat the static oracle, so the whole of §4.2 falls")

    # 5. THE case that exercises the order of the K. size 2, K = 2.
    #    Descending: X Y | Z X  →  X(miss) Y(miss) Z(miss, evicts X) X(miss) = 0.
    #    Reversed  : Y X | X Z  →  Y(miss) X(miss) X(HIT) Z(miss)            = 1.
    dec = simulate([10, 11, 12, 10], capacity=2)
    inv = simulate([11, 10, 10, 12], capacity=2)
    check("5. order of the K, descending score", dec.hits, 0,
          "the insertion order of the K changes the verdict: it must be frozen (§2.1)")
    check("5bis. order of the K, reversed", inv.hits, 1,
          "the contrast with 5 is what proves the order is exercised")

    print()
    if fails:
        print("V0.4 RED: no line of trace may be read:")
        for f in fails:
            print(f"  · {f}")
        return 1
    print("V0.4 GREEN: the five cases pass, reading the trace is allowed.")
    return 0


# --------------------------------------------------------------------------
# Reading a real trace
# --------------------------------------------------------------------------


def read_trace(path: str):
    """One-line JSON header, then the flat array. A missing file FAILS, it
    does not skip (house convention, the shape of
    `ops/moe_ciseau.py:206-208`)."""
    if not os.path.exists(path):
        sys.exit(
            f"trace not found: {path}\n"
            "  It is produced by:\n"
            "    uv run ops/moe_routing.py --model openai/gpt-oss-20b --tokens 131072 \\\n"
            "        --ctx 1024 --device mps --dataset allenai/c4 \\\n"
            "        --json docs/data/moe-routing-....json --trace ~/llvq-moe/trace-....u8\n"
            "  This script fabricates no substitute. A run on a missing trace\n"
            "  would be a smokescreen."
        )
    with open(path, "rb") as f:
        head = json.loads(f.readline().decode("utf-8"))
        body = f.read()
    return head, body


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trace", help="temporal trace produced by moe_routing.py --trace")
    ap.add_argument("--selftest", action="store_true", help="the five V0.4 cases, no trace")
    ap.add_argument(
        "--alphas",
        default="0.2733,0.35,0.45,0.50,0.607,0.7161",
        help="fractions of resident cells to simulate",
    )
    a = ap.parse_args()

    if a.selftest or not a.trace:
        rc = selftest()
        if not a.trace:
            return rc
        if rc:
            return rc
        print()

    head, body = read_trace(a.trace)
    n_tokens = head["tokens"]
    n_layers = len(head["layer_names"])
    top_k = head["top_k"]
    n_experts = head["n_experts"]
    cells = n_layers * n_experts
    want = n_tokens * n_layers * top_k
    if n_experts > 256:
        sys.exit("uint16 trace: not implemented (rule set in advance, §2.1)")
    if len(body) != want:
        sys.exit(f"truncated trace: {len(body)} bytes for {want} expected")

    print(f"# trace   : {a.trace}")
    print(f"# sha256  : {hashlib.sha256(body).hexdigest()}")
    print(f"# model   : {head.get('model')}  transformers {head.get('transformers_version')}")
    print(f"# tokens  : {n_tokens}  layers {n_layers}  top_k {top_k}  experts {n_experts}")
    print(f"# cells   : {cells}   visits/token {n_layers * top_k}")
    print(f"# tokens_sha256 : {head.get('tokens_sha256')}")
    print()
    print(f"  {'α':>8}{'cells':>10}{'hit LRU':>10}{'miss/token':>12}{'compulsory':>14}")
    print("  " + "-" * 54)
    for alpha in [float(s) for s in a.alphas.split(",")]:
        cap = int(alpha * cells)  # ceil/floor convention: frozen to floor, and printed
        c = simulate(decode_order(body, n_tokens, n_layers, top_k, n_experts), cap)
        print(
            f"  {alpha:>8.4f}{cap:>10}{c.rate() * 100:>9.3f}%"
            f"{c.misses / n_tokens:>12.4f}{c.compulsory / max(c.misses, 1) * 100:>13.3f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
