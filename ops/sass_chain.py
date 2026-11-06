"""Critical path of a loop body in a cuobjdump -sass listing, under declared latencies.

    python3 -I ops/sass_chain.py <file.sass> <lo_hex> <hi_hex>

Builds the register dependency graph of the instructions in [lo, hi] as one
straight-line pass (branches ignored, values carried in from the previous
iteration taken as ready) and prints the longest chain in cycles, the
instruction count, and the opcodes on the chain. The latencies are
assumptions, Ada-class figures from public microbenchmarks, not measured on
the L40S: integer and FP32 ALU 4-6, shared load 30, global load hitting L1 35,
shuffle 30. A chain far shorter than the measured time per iteration says the
loop is not bound by its own dependency chain alone.
"""
import re
import sys
from collections import Counter

LINE = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(@!?U?P\w+\s+)?([A-Z0-9_.]+)\s*(.*?);")
REG = re.compile(r"\bR(\d+)(\.64)?")
LAT = [("LDS", 30), ("LDG", 35), ("LD.", 35), ("SHFL", 30), ("MUFU", 20), ("I2F", 6), ("F2F", 6), ("F2I", 6),
       ("IDP", 6), ("IMAD", 5), ("POPC", 6), ("FLO", 6), ("BREV", 6), ("S2R", 20), ("ST", 0), ("BRA", 0),
       ("BSSY", 0), ("BSYNC", 0), ("ISETP", 5), ("FSETP", 5)]


def lat(op):
    for p, v in LAT:
        if op.startswith(p):
            return v
    return 5


def regs(s):
    out = []
    for m in REG.finditer(s):
        r = int(m.group(1))
        out.append(r)
        if m.group(2):
            out.append(r + 1)
    return out


def main():
    lo, hi = int(sys.argv[2], 16), int(sys.argv[3], 16)
    body = []
    for raw in open(sys.argv[1]):
        m = LINE.search(raw)
        if m and lo <= int(m.group(1), 16) <= hi:
            body.append((m.group(3), m.group(4)))
    ready = {}  # register -> (cycle its value is ready, chain of opcodes)
    best = (0, [])
    for op, args in body:
        if op.startswith(("ST", "BRA", "BSSY", "BSYNC", "BAR", "WARPSYNC", "NOP", "EXIT")):
            dst, src = [], regs(args)
        else:
            parts = args.split(",", 1)
            dst = regs(parts[0]) if parts and parts[0].strip().startswith("R") else []
            src = regs(parts[1]) if len(parts) > 1 else []
            if op.startswith(("LD", "LDS")) and len(dst) == 1:
                width = re.search(r"\.(64|128)", op)
                n = {"64": 2, "128": 4}.get(width.group(1), 1) if width else 1
                dst = [dst[0] + i for i in range(n)]
        start, chain = 0, []
        for r in src:
            if r in ready and ready[r][0] > start:
                start, chain = ready[r]
        end = start + lat(op)
        for r in dst:
            if r != 255:
                ready[r] = (end, chain + [op.split(".")[0]])
        if end > best[0]:
            best = (end, chain + [op.split(".")[0]])
    print(f"{len(body)} instructions, critical chain {best[0]} cycles over {len(best[1])} instructions")
    print("  chain by opcode: " + ", ".join(f"{k} {v}" for k, v in Counter(best[1]).most_common()))


if __name__ == "__main__":
    main()
