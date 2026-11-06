"""Count the instructions of every loop in a cuobjdump -sass listing.

    python3 -I ops/sass_loops.py <file.sass>

A loop is a backward branch (BRA to a lower address). For each one: the
address range, the instruction count, and the counts by opcode (the mnemonic
before the first dot, plus the full mnemonic for loads and stores). Read off
the card's own cubins (step L1's cubin/), compiled by the driver's ptxas.
"""
import re
import sys
from collections import Counter

LINE = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(@!?U?P\w+\s+)?([A-Z0-9_.]+)(.*?);")


def parse(path):
    ins = []
    for raw in open(path):
        m = LINE.search(raw)
        if m:
            ins.append((int(m.group(1), 16), m.group(3), m.group(4).strip(), bool(m.group(2))))
    return ins


def main():
    ins = parse(sys.argv[1])
    loops = []
    for addr, op, args, _ in ins:
        if op.startswith("BRA"):
            t = re.search(r"0x([0-9a-f]+)", args)
            if t and int(t.group(1), 16) < addr:
                loops.append((int(t.group(1), 16), addr))
    print(f"{len(ins)} instructions, {len(loops)} backward branches")
    for lo, hi in sorted(loops, key=lambda x: x[1] - x[0]):
        body = [(op, args) for a, op, args, _ in ins if lo <= a <= hi]
        c = Counter()
        for op, args in body:
            base = op.split(".")[0]
            c[op if base in ("LDG", "LDS", "LD", "LDC", "STG", "STS", "ULDC") else base] += 1
        print(f"\nloop 0x{lo:04x}-0x{hi:04x}: {len(body)} instructions")
        print("  " + ", ".join(f"{k} {v}" for k, v in c.most_common()))


if __name__ == "__main__":
    main()
