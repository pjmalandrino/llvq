# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""The image recipes' lists agree: `python3 ops/check_image_lists.py`, exit 0 when they do.

The trap has bitten four times. A binary built by `cargo build --bin` in
`ops/Dockerfile.cuda` but absent from its runtime `COPY` does not exist on the
card, and a `COPY --from=build /src/<path>` whose path `ops/run.py` does not
upload (in `UPLOAD_ALLOW` and not in `UPLOAD_IGNORE`) kills the Space after its
twelve-minute compile. This script checks, with the standard library only, in
`ops/Dockerfile.cuda` and in the CPU recipe `ops/Dockerfile`:

1. every `--bin` built by a `cargo build` inside a `RUN` is copied, and every
   binary copied is built. The `cargo build` may stand anywhere on the logical
   line: after an environment prefix, in an `&&` chain, across continuations;
   `--bin x` and `--bin=x` are both read;
2. every other `COPY --from=build` source is uploaded by `ops/run.py`;
3. every binary `ops/jobs/image-smoke-14b.sh` checks on the target is built
   (CUDA image only).

Before it reads the real files, it runs its own mutants: each one is a form of
the trap inserted in the real `ops/Dockerfile.cuda`, and each must be reported.
"""
import ast
import re
import sys
from fnmatch import fnmatchcase
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# A shell command ends at one of these; `cargo build`'s flags are read up to it.
SHELL_SEPARATOR = r"(?:&&|\|\||;|\|)"


def logical_lines(text: str) -> list[str]:
    """Dockerfile lines, comment lines dropped, backslash continuations joined.

    Docker removes a comment line even inside a continuation, so it is dropped
    before the join, and a `cargo build --bin x` quoted in a comment counts for
    nothing.
    """
    kept = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
    return re.sub(r"\\\n", " ", "\n".join(kept)).splitlines()


def built_bins(dockerfile: str) -> set[str]:
    bins = set()
    for line in logical_lines(dockerfile):
        words = line.split(None, 1)
        if not words or words[0].upper() != "RUN":
            continue
        for cmd in re.finditer(rf"\bcargo\s+build\b(.*?)(?={SHELL_SEPARATOR}|$)", line):
            bins.update(re.findall(r"--bin(?:\s+|=)([A-Za-z0-9_-]+)", cmd.group(1)))
    return bins


def copy_sources(dockerfile: str) -> list[str]:
    """The sources of every `COPY --from=build`, whatever the order of its flags."""
    sources = []
    for line in logical_lines(dockerfile):
        words = line.split()
        if not words or words[0].upper() != "COPY":
            continue
        flags = [w for w in words[1:] if w.startswith("--")]
        rest = [w for w in words[1:] if not w.startswith("--")]
        if "--from=build" in flags and len(rest) > 1:
            sources.extend(rest[:-1])
    return sources


def tuple_of(run_py: str, name: str) -> tuple[str, ...]:
    for node in ast.parse(run_py).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return tuple(ast.literal_eval(node.value))
    raise SystemExit(f"ops/run.py: no {name} tuple")


def upload_allow(run_py: str) -> tuple[str, ...]:
    return tuple_of(run_py, "UPLOAD_ALLOW")


def uploaded(path: str, allow: tuple[str, ...], ignore: tuple[str, ...]) -> bool:
    """Whether `ops/run.py` uploads `path`, as its perimeter filter does."""
    return any(fnmatchcase(path, p) for p in allow) and not any(fnmatchcase(path, p) for p in ignore)


def smoke_bins(script: str) -> set[str]:
    m = re.search(r"for b in ([^;]*?);\s*do\s*\n\s*test -x \"/usr/local/bin/\$b\"", script.replace("\\\n", " "))
    if not m:
        raise SystemExit("ops/jobs/image-smoke-14b.sh: the `for b in ...; do test -x` loop moved")
    return set(m.group(1).split())


def problems(dockerfile: str, run_py: str, smoke: str | None, name: str = "ops/Dockerfile.cuda") -> list[str]:
    out = []
    built = built_bins(dockerfile)
    sources = copy_sources(dockerfile)
    copied = {s.rsplit("/", 1)[1] for s in sources if s.startswith("/src/target/release/")}
    if not built:
        out.append(f"no `cargo build --bin` found in {name}")
    for b in sorted(built - copied):
        out.append(f"{name}: built but not copied into the runtime image: {b}")
    for b in sorted(copied - built):
        out.append(f"{name}: copied but never built: {b}")
    allow = upload_allow(run_py)
    ignore = tuple_of(run_py, "UPLOAD_IGNORE")
    for s in sources:
        if s.startswith("/src/target/"):
            continue
        rel = s.removeprefix("/src/").rstrip("/")
        # A directory source is uploaded when a file inside it would be.
        if not (uploaded(rel, allow, ignore) or uploaded(rel + "/x", allow, ignore)):
            out.append(f"{name}: COPY --from=build {s}: ops/run.py does not upload {rel} "
                       "(no pattern of UPLOAD_ALLOW, or one of UPLOAD_IGNORE)")
    if smoke is not None:
        for b in sorted(smoke_bins(smoke) - built):
            out.append(f"ops/jobs/image-smoke-14b.sh checks {b}, which the image does not build")
    return out


def mutants(dockerfile: str) -> list[tuple[str, str, str | None]]:
    """(name, mutated recipe, word the report must name, or None for a mutant that must stay clean).

    The lines go after the first `RUN cargo build` and after the last
    `COPY --from=build`, wherever those stand in the real file.
    """
    lines = dockerfile.splitlines(keepends=True)
    first_run = next(i for i, l in enumerate(lines) if re.match(r"RUN\s+cargo build\b", l))
    while lines[first_run].rstrip().endswith("\\"):
        first_run += 1
    last_copy = max(i for i, l in enumerate(lines) if l.startswith("COPY --from=build"))
    while lines[last_copy].rstrip().endswith("\\"):
        last_copy += 1

    def after_run(text: str) -> str:
        return "".join(lines[: first_run + 1] + [text + "\n"] + lines[first_run + 1:])

    def after_copy(text: str) -> str:
        return "".join(lines[: last_copy + 1] + [text + "\n"] + lines[last_copy + 1:])

    build = "cargo build --release --locked -p llvq-llm"
    return [
        ("plain RUN", after_run(f"RUN {build} --bin newbin"), "newbin"),
        ("environment prefix", after_run(f"RUN CARGO_BUILD_JOBS=2 {build} --bin newbin"), "newbin"),
        ("&& chain", after_run(f"RUN set -e && {build} --bin newbin"), "newbin"),
        ("continuations", after_run(f"RUN set -e \\\n && {build} \\\n      --bin newbin"), "newbin"),
        ("comment inside a continuation",
         after_run(f"RUN {build} \\\n# a comment\n      --bin newbin"), "newbin"),
        ("--bin=x", after_run(f"RUN {build} --bin=newbin"), "newbin"),
        ("copied, never built", after_copy("COPY --from=build /src/target/release/ghost /usr/local/bin/"), "ghost"),
        ("allowed but in UPLOAD_IGNORE", after_copy("COPY --from=build /src/llvq-tetra/README.md /x"), "llvq-tetra"),
        ("ignored directory", after_copy("COPY --from=build /src/llvq-tetra/ /x/"), "llvq-tetra"),
        ("--chmod before the source", after_copy("COPY --from=build --chmod=755 /src/docs/ETAT.md /x"), "docs/ETAT.md"),
        ("--chmod before --from", after_copy("COPY --chmod=755 --from=build /src/docs/ETAT.md /x"), "docs/ETAT.md"),
        ("a bin named in a comment", after_run(f"# RUN {build} --bin commentbin"), None),
        ("--bins is not --bin", after_run(f"RUN {build} --bins"), None),
    ]


def self_test(dockerfile: str, run_py: str, smoke: str) -> list[str]:
    """The mutants each check must catch, run on the real recipe. Returns the failures."""
    clean = problems(dockerfile, run_py, smoke)
    failures = []
    for name, mutated, word in mutants(dockerfile):
        found = [p for p in problems(mutated, run_py, smoke) if p not in clean]
        if word is None and found:
            failures.append(f"self-test: mutant '{name}' must stay clean, got {found}")
        elif word is not None and not any(word in p for p in found):
            failures.append(f"self-test: mutant '{name}' was not reported (wanted a problem naming {word})")
    return failures


def main() -> int:
    cuda = (ROOT / "ops/Dockerfile.cuda").read_text()
    run_py = (ROOT / "ops/run.py").read_text()
    smoke = (ROOT / "ops/jobs/image-smoke-14b.sh").read_text()
    found = self_test(cuda, run_py, smoke)
    if not found:
        found = problems(cuda, run_py, smoke) + problems(
            (ROOT / "ops/Dockerfile").read_text(), run_py, None, name="ops/Dockerfile"
        )
    for p in found:
        print(f"image lists: {p}", file=sys.stderr)
    if not found:
        print(f"image lists agree (self-test: {len(mutants(cuda))} mutants)")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
