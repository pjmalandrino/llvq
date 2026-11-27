"""Gate D-ii of prereg inkling-fenetre-2026-11-27: one arm's logits file against a reference's.

usage: python3 -I compare.py <ref.logits.f32> <arm.logits.f32> [<ref.json> <arm.json>]
Both files are inkrun's: rows of 200,058 f32 logits, one row per generated token. Prints the sha256 of each, the
largest |dlogit|, the largest |logit| of the reference, the rows whose argmax differ, and, given the two reports,
whether the 32 tokens after the prompt are the same.
"""
import hashlib, json, sys
from array import array

VOCAB = 200058


def load(p):
    a = array("f")
    with open(p, "rb") as f:
        a.frombytes(f.read())
    return a


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


ref, arm = sys.argv[1], sys.argv[2]
r, a = load(ref), load(arm)
assert len(r) == len(a) and len(r) % VOCAB == 0, (len(r), len(a))
rows = len(r) // VOCAB
print(f"ref {sha(ref)}  {ref}")
print(f"arm {sha(arm)}  {arm}")
dmax, lmax, moved, equal = 0.0, 0.0, [], 0
for i in range(rows):
    rr, aa = r[i * VOCAB:(i + 1) * VOCAB], a[i * VOCAB:(i + 1) * VOCAB]
    d = max(abs(x - y) for x, y in zip(rr, aa))
    dmax = max(dmax, d)
    lmax = max(lmax, max(abs(x) for x in rr))
    ar = max(range(VOCAB), key=rr.__getitem__)
    aa_ = max(range(VOCAB), key=aa.__getitem__)
    if ar != aa_:
        moved.append(i)
    equal += int(rr.tobytes() == aa.tobytes())
print(f"rows {rows}; bit for bit {equal}; max |dlogit| {dmax:.6g}; largest |logit| {lmax:.6g}; "
      f"relative {dmax / lmax:.3g}; argmax moved at rows {moved}")
if len(sys.argv) > 4:
    t1 = json.load(open(sys.argv[3]))["tokens"]
    t2 = json.load(open(sys.argv[4]))["tokens"]
    print(f"tokens after the prompt: {len(t1)} and {len(t2)}, same {t1 == t2}")
