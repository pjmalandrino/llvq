# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""No new tracked file over 1 MB: `python3 ops/check_big_files.py`, exit 0 when none.

The tracked tree weighs about 209 MB, 70 % of it raw directories under
`docs/mesures/` (audit of 2026-11-07, §2.8). A raw file over 1 MB goes to the
bucket, with its sha256 in the journal (docs/METHODE.md). The 80 files that
were already over the line on 2026-11-08 are listed in `ops/big-files.allow`
and stay; any other one fails this check, naming its size. A file that has to
be tracked anyway is added to the list in the same commit, which makes the
choice visible in review.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIMIT = 1 << 20


def main() -> int:
    allow = set((ROOT / "ops/big-files.allow").read_text().split())
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    found = []
    for name in out.decode().split("\0"):
        p = ROOT / name
        if name and p.is_file() and not p.is_symlink() and p.stat().st_size > LIMIT and name not in allow:
            found.append((p.stat().st_size, name))
    for size, name in sorted(found, reverse=True):
        print(f"big files: {name} is {size / 1e6:.1f} MB; put it in the bucket and its sha256 in the journal, "
              "or list it in ops/big-files.allow", file=sys.stderr)
    if not found:
        print(f"no new tracked file over 1 MB ({len(allow)} listed)")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
