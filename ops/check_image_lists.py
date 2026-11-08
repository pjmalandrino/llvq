# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""The CUDA image's lists agree: `python3 ops/check_image_lists.py`, exit 0 when they do.

The trap has bitten four times. A binary built by `cargo build --bin` in
`ops/Dockerfile.cuda` but absent from its runtime `COPY` does not exist on the
card, and a `COPY --from=build /src/<path>` whose path is not in `UPLOAD_ALLOW`
of `ops/run.py` kills the Space after its twelve-minute compile. This script
checks, with the standard library only:

1. every `--bin` built is copied, and every binary copied is built;
2. every other `COPY --from=build` source matches a pattern of `UPLOAD_ALLOW`;
3. every binary `ops/jobs/image-smoke-14b.sh` checks on the target is built.
"""
import ast
import re
import sys
from fnmatch import fnmatchcase
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def logical_lines(text: str) -> list[str]:
    """Dockerfile lines with their backslash continuations joined."""
    return re.sub(r"\\\n", " ", text).splitlines()


def built_bins(dockerfile: str) -> set[str]:
    bins = set()
    for line in logical_lines(dockerfile):
        if re.match(r"\s*RUN\s+cargo build\b", line):
            bins.update(re.findall(r"--bin\s+([A-Za-z0-9_-]+)", line))
    return bins


def copy_sources(dockerfile: str) -> list[str]:
    sources = []
    for line in logical_lines(dockerfile):
        words = line.split()
        if len(words) > 2 and words[0] == "COPY" and words[1] == "--from=build":
            sources.extend(words[2:-1])
    return sources


def upload_allow(run_py: str) -> tuple[str, ...]:
    for node in ast.parse(run_py).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "UPLOAD_ALLOW" for t in node.targets
        ):
            return tuple(ast.literal_eval(node.value))
    raise SystemExit("ops/run.py: no UPLOAD_ALLOW tuple")


def smoke_bins(script: str) -> set[str]:
    m = re.search(r"for b in ([^;]*?);\s*do\s*\n\s*test -x \"/usr/local/bin/\$b\"", script.replace("\\\n", " "))
    if not m:
        raise SystemExit("ops/jobs/image-smoke-14b.sh: the `for b in ...; do test -x` loop moved")
    return set(m.group(1).split())


def problems(dockerfile: str, run_py: str, smoke: str) -> list[str]:
    out = []
    built = built_bins(dockerfile)
    sources = copy_sources(dockerfile)
    copied = {s.rsplit("/", 1)[1] for s in sources if s.startswith("/src/target/release/")}
    if not built:
        out.append("no `cargo build --bin` found in ops/Dockerfile.cuda")
    for b in sorted(built - copied):
        out.append(f"built but not copied into the runtime image: {b}")
    for b in sorted(copied - built):
        out.append(f"copied but never built: {b}")
    allow = upload_allow(run_py)
    for s in sources:
        if s.startswith("/src/target/"):
            continue
        rel = s.removeprefix("/src/").rstrip("/")
        if not any(fnmatchcase(rel, p) or fnmatchcase(rel + "/x", p) for p in allow):
            out.append(f"COPY --from=build {s}: {rel} matches no pattern of UPLOAD_ALLOW in ops/run.py")
    for b in sorted(smoke_bins(smoke) - built):
        out.append(f"ops/jobs/image-smoke-14b.sh checks {b}, which the image does not build")
    return out


def main() -> int:
    found = problems(
        (ROOT / "ops/Dockerfile.cuda").read_text(),
        (ROOT / "ops/run.py").read_text(),
        (ROOT / "ops/jobs/image-smoke-14b.sh").read_text(),
    )
    for p in found:
        print(f"image lists: {p}", file=sys.stderr)
    if not found:
        print("image lists agree")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
