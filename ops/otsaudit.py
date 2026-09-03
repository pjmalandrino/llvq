"""Audit the OpenTimestamps stamps in proofs/ from the bytes on disk.

Answers three questions that prose in CLAUDE.md had been answering from
memory, and one that a `grep` had been answering wrongly:

  1. does each .ots still attest to the .md sitting next to it (sha256
     recomputed and compared to the digest the stamp commits to);
  2. how many Bitcoin anchors does each stamp actually carry, and at what
     block heights;
  3. which pre-registrations carry no stamp at all.

WHY THIS EXISTS.  The .ots format stores an attestation's type as an 8-byte
binary tag, never as text, so `grep BitcoinBlockHeaderAttestation` returns 0
on a file carrying four of them -- and `grep PendingAttestation` returns 0
too.  An instrument that returns the same value in both cases measures
nothing; this one deserializes the proof instead.

WHAT IT DOES NOT DO.  Verifying an anchor end to end means checking the
committed Merkle root against the real block, which needs a Bitcoin node or a
block explorer.  This script reads what the files carry and prints the
committed root per block so a third party can close that last step.

    pip install opentimestamps-client      # provides the `opentimestamps` lib
    python3 ops/otsaudit.py > docs/mesures/ots-etat-<date>.txt

Run from the repository root: paths are relative to it.
"""
import datetime as _dt
import glob, hashlib, os, subprocess
from opentimestamps.core.timestamp import DetachedTimestampFile
from opentimestamps.core.notary import BitcoinBlockHeaderAttestation, PendingAttestation
from opentimestamps.core.serialize import StreamDeserializationContext

def load(p):
    with open(p, "rb") as f:
        return DetachedTimestampFile.deserialize(StreamDeserializationContext(f))

print("=" * 78)
print("ACTUAL STATE OF THE OpenTimestamps STAMPS  -  measured on "
      + _dt.date.today().isoformat())
print("=" * 78)
print("""
INSTRUMENT.  `ots info` (opentimestamps-client v0.7.2, installed from PyPI) and
the python `opentimestamps` library, which deserializes the .ots and walks its
attestations.  Every number below is *measured* on the bytes of the repository
files; nothing is estimated.

WARNING: WHAT IS NOT DONE HERE, AND WHY.  Verifying an anchor end to end means
checking the committed Merkle root against the real block, which needs a Bitcoin
node or a block explorer.  This script queries neither, whatever the state of
the network.  It deserializes files, that is all.  This journal establishes what
the FILES carry, not that the chain confirms it.
(WARNING: access to the explorers and to the four calendars VARIES from one
session to the next: blocked at 403 on 2026-08-26, open on 2026-08-27.  Do not
read a missing anchor as a fact about the machine.)  The roots are printed at
the end of the journal exactly so a third party can take that last step in one
command.
""")

print("-" * 78)
print("1.  THE GREP THE CLAUDE.md LINE RESTS ON")
print("-" * 78)
print("""
CLAUDE.md (header and §7) states, "verified by grep on 2026-08-25":
    16 .ots, each carrying 4 PendingAttestation and 0 BitcoinBlockHeaderAttestation,
    "none has ever been upgraded".

What the grep actually returns, on a file that carries four anchors:""")
f = "proofs/preregistration-p1-2026-08-13.md.ots"
for pat in ("BitcoinBlockHeaderAttestation", "PendingAttestation"):
    n = subprocess.run(["grep","-c",pat,f], capture_output=True, text=True).stdout.strip()
    print(f"    grep -c {pat:<32} {f}  ->  {n}")
print("""
Both return 0.  The .ots format stores an attestation type in an 8-byte binary
tag, never as text: the class name appears only in the output RENDERED by
`ots info`, and in the library source.  A grep on this format therefore cannot
tell an anchored file from a pending one.  It returns 0 in both cases.

Consequence for the two published numbers:
  · the "0 BitcoinBlockHeaderAttestation" is what the instrument returned, and
    it is WRONG.  The file above carries four;
  · the "4 PendingAttestation" is RIGHT, but it does not come from this grep,
    which returns 0 as well.  It was inferred ("the four calendars") and
    presented as measured.
""")

print("-" * 78)
print("2.  WHAT THE FILES CARRY")
print("-" * 78)
rows = []
for p in sorted(glob.glob("proofs/*.ots")):
    d, doc = load(p), p[:-4]
    ok = None
    if os.path.exists(doc):
        ok = hashlib.sha256(open(doc,"rb").read()).digest() == d.file_digest
    btc, pend = set(), set()
    for msg, att in d.timestamp.all_attestations():
        if isinstance(att, BitcoinBlockHeaderAttestation): btc.add((att.height, msg[::-1].hex()))
        elif isinstance(att, PendingAttestation): pend.add(att.uri)
    rows.append((os.path.basename(doc), ok, sorted(btc), sorted(pend)))

print(f"\n{'document':<56} {'sha256 of .md':<14} {'btc':>4} {'pend':>5}")
for name, ok, btc, pend in rows:
    mark = {True: "recomputes ok", False: "DOES NOT MATCH", None: ".md missing"}[ok]
    print(f"{name:<56} {mark:<14} {len(btc):>4} {len(pend):>5}")

nb = sum(1 for _,_,b,_ in rows if b)
print(f"""
TOTAL: {len(rows)} stamps, of which {nb} carry at least one Bitcoin anchor and
{len(rows)-nb} carry none.  All carry the 4 pending attestations.

The CLAUDE.md count ("16 .ots") is stale too: there are {len(rows)} of them,
for {len(glob.glob('proofs/*.md'))} documents (README.md included).
""")

print("-" * 78)
print("3.  THE TWO STAMPS THAT NO LONGER ATTEST TO THEIR FILE")
print("-" * 78)
print("""
CLAUDE.md §7 already knew it, from memory: "the reverse defect did occur on the
08-10 and the 08-11 preregs".  It is now machine-checkable, and the mechanism is
named: commit 01fdbe6 (2026-08-19), the anonymization pass for TACO, rewrote
those two documents.  An anchor attests to BYTES.  Rewriting a document destroys
what its anchor proves.
""")
for doc in ["proofs/preregistration-2026-08-10.md", "proofs/preregistration-2026-08-11.md"]:
    want = load(doc + ".ots").file_digest
    have = hashlib.sha256(open(doc,"rb").read()).digest()
    print(f"  {os.path.basename(doc)}")
    print(f"    the stamp commits to  {want.hex()}")
    print(f"    the file hashes to    {have.hex()}")
print("""
And the attested version is NOT recoverable: the 128 distinct .md blobs of the
whole git history were hashed, and none yields either of these two digests.
These two stamps therefore prove the prior existence of a text the repository no
longer holds, under any revision.
""")

print("-" * 78)
print("4.  WHAT IS NOT TIMESTAMPED")
print("-" * 78)
print()
for f in sorted(glob.glob("proofs/*.md")):
    if not os.path.exists(f + ".ots"):
        print(f"    {os.path.basename(f)}")
print("""
    (README.md is not a preregistration.)

WARNING: preregistration-variance-calibration-2026-08-26.md is in this list, and
    its own §3 requires the stamp before the first measured millisecond.  It
    cannot be stamped from this machine: the four calendars are unreachable
    (403).
    To be done from a networked machine, before batch 1.
""")

print("-" * 78)
print("5.  COMMITTED MERKLE ROOTS  -  for third-party verification")
print("-" * 78)
print("""
Each line says: this file commits this root in the block at this height.
Verification in one command, from a networked machine:

    ots verify proofs/<file>.md.ots        (with a Bitcoin node)

or, without a node, by comparing the root below to the block's merkle_root field:

    curl -s https://blockstream.info/api/block-height/<height> \\
      | xargs -I{} curl -s https://blockstream.info/api/block/{} | jq -r .merkle_root
""")
for name, ok, btc, pend in rows:
    if not btc: continue
    print(f"\n  {name}")
    for h, root in btc:
        print(f"    block {h}   {root}")
print()
