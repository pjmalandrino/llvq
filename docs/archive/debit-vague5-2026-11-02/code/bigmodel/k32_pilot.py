# DRAFT, never run. Written read-only on 2026-11-02 for lead "bigmodel"
# (prereg-bigmodel-1.md, stage P0). Not part of the repository. Standard
# library only; `python3 -I k32_pilot.py ...` on paths passed as arguments.
#
# P0 re-validates the GGUF instrument on TODAY's llama.cpp build against the
# only llama.cpp value this repository holds, the 4B engine gate of 2026-08-30
# (70.36 micro, docs/data/mmlu-dumps/mmlu-4b-gguf-f16.csv), and anchors the Mac
# time per question that no journal records.
#
#   write   PLAN OUT        100 rows of PLAN, every 23rd (rows 0..2277), header
#                           kept, NO trailer (the run-fingerprint check of
#                           gguf_mmlu.py then skips: 100 != 2280)
#   compare PILOT GGUF OURS picks of the pilot dump against the 2026-08-30
#                           llama.cpp dump and against ours, on the 100 keys,
#                           qhash checked; exit 1 if more than 3 differ from
#                           the llama.cpp dump (the P0 gate)
#
# Every 23rd row of a plan ordered by subject, 40 rows a subject, touches all
# 57 subjects (43 twice, 14 once), so the prompt-length mix, which sets the
# time, is the plan's.

import csv
import sys

STEP, N_PILOT, MAX_DIFF = 23, 100, 3


def split(path):
    head, body = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("# end "):
                continue
            (head if line.startswith("#") else body).append(line)
    return head, body


def rows(path):
    _, body = split(path)
    return {(r["subject"], int(r["index"])): r for r in csv.DictReader(body)}


def write(plan, out):
    head, body = split(plan)
    columns, data = body[0], body[1:]
    picked = data[0:STEP * (N_PILOT - 1) + 1:STEP]
    if len(picked) != N_PILOT:
        raise SystemExit(f"picked {len(picked)} rows, expected {N_PILOT}")
    with open(out, "w", encoding="utf-8") as f:
        f.writelines(head)
        f.write("# pilot=every 23rd row of the plan, rows 0..2277, no trailer\n")
        f.write(columns)
        f.writelines(picked)
    subjects = {r.split(",", 1)[0] for r in picked}
    print(f"wrote {len(picked)} rows over {len(subjects)} subjects to {out}")


def compare(pilot, gguf, ours):
    p, g, o = rows(pilot), rows(gguf), rows(ours)
    if len(p) != N_PILOT:
        raise SystemExit(f"pilot holds {len(p)} rows, expected {N_PILOT}")
    for name, ref in (("llama.cpp 2026-08-30", g), ("ours f16", o)):
        bad = [k for k in p if k not in ref or ref[k]["qhash"] != p[k]["qhash"]]
        if bad:
            raise SystemExit(f"{name}: {len(bad)} pilot rows missing or with another qhash")
    d_g = sum(1 for k in p if p[k]["pick"] != g[k]["pick"])
    d_o = sum(1 for k in p if p[k]["pick"] != o[k]["pick"])
    print(f"picks different from llama.cpp 2026-08-30: {d_g} / {N_PILOT}  (gate: <= {MAX_DIFF})")
    print(f"picks different from ours f16:             {d_o} / {N_PILOT}  (reported)")
    if d_g > MAX_DIFF:
        print("P0 FAILS: the instrument moved; nothing else runs until the cause is written")
        sys.exit(1)
    print("P0 passes on picks; the four-letter count is gguf_mmlu.py's own refusal")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "write":
        write(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 5 and sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        raise SystemExit("usage: write PLAN OUT | compare PILOT GGUF OURS")
