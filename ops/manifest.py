# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface-hub>=1.26"]
# ///
"""Tie every published number to its proof.

Why this file exists: three headline figures of this repository (2.39 GB of
resident memory, 129.8 tok/s, the three decimals of 21.691 ms) are carried by
**no log at all** — `docs/archive/plan-de-test-papier.md` §5, point 12 names them. They were
typed into a table from a terminal that has since scrolled away. Nobody can
say today which command produced them, on which object, from which commit.

That is not a documentation problem, it is a measurement problem: a number
whose provenance is lost cannot be defended, corrected, or reproduced — and
this project has already had to retract 2.0653 b/weight, 92.24 % retention and
a macro-averaged MMLU, each time because the number outlived the conditions
that produced it.

## The one thing this file exists to prevent

A number entering the paper without a line in `ops/manifest.jsonl` that says
which command produced it, on which bytes, from which commit, in which job.

`record` derives what it can rather than trusting what it is told: the git
commit, whether the tree was clean, the sha256 of the log, the sha256 of the
measured object. **It refuses a measure taken from a dirty tree**, because a
number produced by uncommitted code cannot be reproduced by anyone, including
its author. `--allow-dirty` exists for exploration and marks the entry as
such — and `verify` rejects those entries unless told otherwise.

`verify` is the lethal part. A manifest nobody can re-check is worth no more
than no manifest: it re-reads every log, re-hashes it, re-hashes and re-sizes
every locally available object, and asks git whether each commit still exists.
`--selftest` proves the verifier actually detects corruption instead of nodding
along — it records a measure, verifies it, then breaks the log, deletes it,
corrupts the object, forges a commit, and requires a failure with the *right*
cause each time.

## The hole that hashing alone leaves, and how it is closed

Hashing the log proves the log has not moved. It does **not** prove the number
came from it. Nothing stopped `"value": 16.9415` from being edited to `12.0`
in a committed manifest whose every hash still matched — and the value is the
one field that ends up printed in the paper. So `record` also searches the log
for the digits it is being asked to record, stores the answer in
`value_in_log`, and `verify` searches again: an entry that claimed its number
was in its log and no longer is, is a **failure**. That closes the edit, and
the field is *required* so that deleting it is a failure too.

It is a presence check, not a parse — it says the digits are physically in the
bytes being hashed, nothing more. `false` is legitimate (a ratio computed from
two logged numbers, a size logged in GB and recorded in bytes) and is recorded,
warned about, and marked in `report`. What is not legitimate is nobody
noticing.

## On job identifiers and their URLs

`ops/run.py` never builds a job URL. It prints the one the API hands back
(`cmd_launch` and `cmd_oracle`, `job.url` off the object `run_job` returns),
and takes the identifier as an opaque positional everywhere else
(`cmd_monitor`, `cmd_watch`, both forwarding `job_id=` verbatim to
`inspect_job` / `fetch_job_logs`). The reason is structural: `JobInfo` sets
`self.url = f"{self.endpoint}/jobs/{self.owner.name}/{self.id}"`, so the URL
needs the **namespace**, which the identifier does not carry. A URL built here
from an id alone would be a guess.

(Symbols rather than line numbers on purpose: the references this file
inherited — `run.py:374` for that print, `run.py:303` for the flush below —
had both already drifted by ~45 lines.)

So this file does the same: the id is stored verbatim, and the URL is either
resolved through `inspect_job` (recorded as `api`) or supplied by hand
(recorded as `declare`). Never assembled.

WARNING: and per `docs/archive/plan-de-test-v2-cuda.md` §6.1, a job URL redirects an
anonymous visitor to a login form. It is a pointer for an auditor who has been
granted access, **not** a proof opposable to a reader. The proof is the log,
and what makes the log a proof is its hash sitting in a committed file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "ops" / "manifest.jsonl"

# Bumped only when an existing field changes meaning. Adding a field does not
# bump it: `verify` requires a known set and tolerates extras.
SCHEMA = 1

# The three arms of the campaign (`docs/archive/plan-de-test-v2-cuda.md` §2.1), plus
# the ones its arbitration no. 2 dropped, kept so that an old entry still reads,
# and so that a typo (`A` for `A1`) is rejected instead of silently splitting a
# comparison in two.
ARMS: dict[str, str] = {
    "A1": "LLVQ 2 bits, Pier-Jean/Qwen3-4B-LLVQ-2bit, qwen3-4b-llvq.bin",
    "A2": "LLVQ 2 bits, int4 embedding, ~/q4b-e4.llvq (no comparator)",
    "B0": "official AWQ 4 bits, Qwen/Qwen3-4B-AWQ",
    "B1": "GPTQ w2 g128 sym (dropped by the operator, §9 no. 2)",
    "B2": "GPTQ w3 g128 sym (dropped by the operator, §9 no. 2)",
    "B3": "bitsandbytes NF4 bs64 (dropped by the operator, §9 no. 2)",
    "C": "original f16, Qwen/Qwen3-4B",
    "ctrl": "control (oracle, L1-L4, CI, C8), not a comparison arm",
}

REQUIRED_FIELDS = ("schema", "recorded_at", "metric", "value", "value_in_log",
                   "unit", "arm", "object", "command", "log", "git")

# How many bytes of a log are searched for the recorded number. Every log this
# repository has produced is a few kilobytes; the cap only exists so that a
# runaway log cannot be read into memory whole. It is applied identically at
# record and at verify time, so the two always agree.
LOG_SEARCH_BYTES = 32 << 20


class ManifestError(RuntimeError):
    """A refusal the operator is meant to read and act on, not a crash."""


# --- primitives -------------------------------------------------------------


def sha256_file(path: Path) -> tuple[str, int]:
    """`(hexdigest, bytes)`, streamed — logs are small, artifacts are 1.8 GB."""
    h = hashlib.sha256()
    total = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            total += len(chunk)
    return h.hexdigest(), total


def store_path(path: Path, root: Path = ROOT) -> str:
    """Repo-relative when possible: a manifest is committed, and an absolute
    `/Users/<someone>/…` in it is unresolvable for every other reader."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


def locate(stored: str, root: Path = ROOT) -> Path | None:
    """Inverse of `store_path`, tolerant of both conventions."""
    candidate = root / stored          # an absolute `stored` wins on its own
    if candidate.is_file():
        return candidate
    direct = Path(stored).expanduser()
    return direct if direct.is_file() else None


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_ts(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def short(digest: str | None, n: int = 8) -> str:
    return digest[:n] if digest else "—"


def thousands(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("s" if n > 1 else "")


def fmt_value(value: int | float) -> str:
    """The number exactly as recorded.

    `repr` on a float round-trips in Python; a formatted one does not. A
    provenance table must show the number that was recorded, not a rounding of
    it — this repo has already published 2.0702, 2.1696 and 2.1595 for one
    file, and the only defence is that each is printed exactly as recorded.
    """
    return repr(value) if isinstance(value, float) else str(value)


def read_text_head(path: Path, limit: int = LOG_SEARCH_BYTES) -> str:
    """The first `limit` bytes of a file, decoded leniently.

    `errors="replace"` on purpose: a log is whatever the run printed, and a
    stray byte from a progress bar must not turn a provenance check into a
    traceback.
    """
    with path.open("rb") as f:
        return f.read(limit).decode("utf-8", errors="replace")


# Digit groups written with a space — `thousands()` produces them, and so do
# the tables of this repository. Only groups of exactly three are collapsed:
# collapsing any digit-space-digit would turn `ppl 4096 12` into `409612` and
# invent evidence that is not there.
_GROUPING = re.compile("(?<=\\d)[ \u00a0\u202f](?=\\d{3}(?!\\d))")


def value_evidence(text: str, value: int | float) -> str | None:
    """The literal spelling of `value` found in `text`, or `None`.

    A **presence** check, deliberately, and its limits are the point:

    * it proves the digits an entry claims are physically inside the bytes that
      entry hashes. That is what makes a hand-edited `value` detectable — the
      hash alone never could, since the log is untouched;
    * it proves nothing about *which* number in the log is the measure. A log
      containing `16.9415` twice, once as a result and once as a seed, satisfies
      it. Closing that would take parsing the log, which means a schema per
      binary, which is the machinery this file exists to avoid.

    Accepted spellings: the canonical `repr`, the same with a decimal comma, and
    either of those inside space-grouped digits. The logs this scans are frozen
    and French, so the comma and the space grouping both stay in scope. A log printing
    *more* decimals than were recorded matches on the prefix; one printing
    fewer does not, and says so rather than guessing.
    """
    canon = fmt_value(value)
    needles = [canon]
    if "." in canon:
        needles.append(canon.replace(".", ","))
    # U+2212 is what the repo's own tables use for a minus sign.
    normalized = text.replace("−", "-")
    haystacks = (normalized, _GROUPING.sub("", normalized))
    for needle in needles:
        for hay in haystacks:
            if needle in hay:
                return needle
    return None


def parse_value(text: str) -> int | float:
    """Numbers as the project writes them, including the French decimal comma.

    `16,9415` is how the frozen journals spell it, and refusing that would
    guarantee somebody re-types the number by hand, which is the exact failure
    this file exists to stop. The living tables now spell it `16.9415`.

    WARNING: one comma and no dot is read as a decimal comma. That is the
    repo's convention, not a proof: `1,770` meant as an English thousands
    separator would be read as 1.77 and no rule can tell the two apart
    (`×1,386` is a real three-digit decimal in this very repository). The
    defence is not here. It is `value_in_log`, which goes looking for the
    digits in the log and says so when it cannot find them.
    """
    text = text.strip()
    if text.count(",") == 1 and "." not in text:
        text = text.replace(",", ".")
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"value is not numeric: {text!r}") from exc


def parse_kv(text: str) -> tuple[str, str]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"expected key=value, got {text!r}")
    key, _, val = text.partition("=")
    return key.strip(), val.strip()


# --- git --------------------------------------------------------------------


@dataclass(frozen=True)
class GitState:
    """What the working tree was when a measure was taken.

    A value object rather than a call, so that `--selftest` can exercise the
    dirty-tree refusal **without** dirtying the caller's tree — the tree is
    dirty right now, and a self-test whose verdict depends on that is a
    self-test that tests nothing.
    """

    commit: str
    dirty: bool
    dirty_files: tuple[str, ...] = ()


def git_out(root: Path, *args: str) -> str:
    proc = subprocess.run(("git", "-C", str(root), *args),
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise ManifestError(
            f"`git {' '.join(args)}` failed: {proc.stderr.strip() or 'code ' + str(proc.returncode)}")
    return proc.stdout


def git_state(root: Path = ROOT) -> GitState:
    commit = git_out(root, "rev-parse", "HEAD").strip()
    lines = [row for row in git_out(root, "status", "--porcelain").splitlines() if row.strip()]
    # `--porcelain` prefixes each path with two status columns and a space.
    return GitState(commit=commit, dirty=bool(lines),
                    dirty_files=tuple(row[3:] for row in lines))


def commit_exists(commit: str, root: Path = ROOT) -> bool:
    proc = subprocess.run(("git", "-C", str(root), "cat-file", "-e", f"{commit}^{{commit}}"),
                          capture_output=True, text=True)
    return proc.returncode == 0


# --- the job side -----------------------------------------------------------


def resolve_job(job_id: str) -> dict:
    """Everything the Hub will say about a job, or nothing at all.

    Never assembles a URL (see the module docstring): either the API returns
    one, or the entry carries the identifier alone and says so. A failure here
    is not fatal. Losing the network must not cost the operator the record of
    a measure that has already been paid for.
    """
    out: dict = {"id": job_id, "url": None, "url_source": "indisponible"}
    try:
        from huggingface_hub import inspect_job
    except ImportError:
        return out
    try:
        info = inspect_job(job_id=job_id)
    except Exception as exc:  # network, token, unknown id — all non-fatal here
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    out["url"] = info.url
    out["url_source"] = "api"
    owner = getattr(info, "owner", None)
    if owner is not None:
        out["owner"] = getattr(owner, "name", None)
    status = getattr(info, "status", None)
    if status is not None:
        out["stage"] = getattr(status, "stage", None)
    out["flavor"] = getattr(info, "flavor", None)
    durations = getattr(info, "durations", None)
    if durations is not None:
        # The three of them, per plan §6.4: the billing page says "Starting or
        # Running" and `JobDurations` has no "starting" — so which one is the
        # billed figure is an open question, and the manifest keeps all three
        # rather than picking a winner it cannot justify.
        out["durations"] = {k: getattr(durations, k, None)
                            for k in ("scheduling_secs", "running_secs", "total_secs")}
    return out


# --- record -----------------------------------------------------------------


def append_entry(manifest: Path, entry: dict) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


def read_entries(manifest: Path) -> list[tuple[int, str]]:
    """`(line number, raw text)`, blank lines dropped, JSON left unparsed so
    that `verify` can report an unreadable line instead of dying on it."""
    if not manifest.is_file():
        return []
    out = []
    with manifest.open("r", encoding="utf-8") as f:
        for i, raw in enumerate(f, start=1):
            if raw.strip():
                out.append((i, raw))
    return out


def record(manifest: Path, *, git_st: GitState, metric: str, value: int | float,
           unit: str, arm: str, command: str, log: Path, obj: str,
           obj_sha: str | None = None, obj_rev: str | None = None,
           job_id: str | None = None, job_url: str | None = None,
           versions: dict | None = None, claim: str | None = None,
           notes: str | None = None, allow_dirty: bool = False,
           root: Path = ROOT, timestamp: str | None = None) -> dict:
    """Build and append one entry. Raises `ManifestError` on every refusal.

    Deliberately takes `git_st` instead of calling `git_state()`: see the
    docstring of `GitState`.
    """
    if arm not in ARMS:
        raise ManifestError(f"unknown arm: {arm!r}. Known: {', '.join(ARMS)}")
    if not metric.strip():
        raise ManifestError("empty metric")
    if not command.strip():
        raise ManifestError("empty command: a number without its command cannot be found again")
    if git_st.dirty and not allow_dirty:
        listing = "\n".join(f"    {f}" for f in git_st.dirty_files[:20])
        more = f"\n    ... and {len(git_st.dirty_files) - 20} more" if len(git_st.dirty_files) > 20 else ""
        raise ManifestError(
            "dirty working tree: no paper number comes out of an uncommitted tree.\n"
            f"  {len(git_st.dirty_files)} modified file(s):\n{listing}{more}\n"
            "  Commit, then run the MEASURE again (not just the recording).\n"
            "  `--allow-dirty` records anyway, marks the entry, and `verify` will refuse it.")

    if obj_sha is not None:
        # The `declare` path is the one with no bytes to check against, so the
        # shape is all there is to check. A sha256 is 64 hex characters; a
        # truncated copy-paste would otherwise sit in the manifest looking like
        # provenance and matching nothing, forever.
        obj_sha = obj_sha.strip().lower()
        if len(obj_sha) != 64 or obj_sha.strip("0123456789abcdef"):
            raise ManifestError(
                f"--object-sha256 is not a sha256 (64 hexadecimal characters): {obj_sha!r}")

    if job_url and not job_id:
        # The URL of a job contains its identifier, so a URL with no id is
        # provenance that points at a job the entry cannot name. Silently
        # dropping it, which is what happened, is the failure mode this file
        # is built against.
        raise ManifestError(
            "--job-url without --job: the URL points at a job the entry does not name")

    log = Path(log).expanduser()
    if not log.is_file():
        raise ManifestError(f"log not found: {log}")
    log_sha, log_bytes = sha256_file(log)
    evidence = value_evidence(read_text_head(log), value)

    obj_path = Path(obj).expanduser()
    obj_block: dict = {"ref": obj, "revision": obj_rev}
    if obj_path.is_file():
        computed, obj_bytes = sha256_file(obj_path)
        if obj_sha and obj_sha != computed:
            raise ManifestError(
                f"the sha256 declared for the object does not match the file:\n"
                f"  declared {obj_sha}\n  computed {computed}")
        obj_block.update(ref=store_path(obj_path, root), sha256=computed,
                         bytes=obj_bytes, sha256_source="calcule")
    elif obj_sha:
        obj_block.update(sha256=obj_sha, bytes=None, sha256_source="declare")
    else:
        obj_block.update(sha256=None, bytes=None, sha256_source="indisponible")

    job_block = None
    if job_id:
        job_block = resolve_job(job_id)
        if job_url:
            job_block["url"] = job_url
            job_block["url_source"] = "declare"

    entry = {
        "schema": SCHEMA,
        "recorded_at": timestamp or now_utc(),
        "metric": metric,
        "claim": claim,
        "value": value,
        # Derived, never supplied: `record` looks for the digits itself. This
        # is what makes a hand-edited `value` detectable at all — see the
        # module docstring.
        "value_in_log": evidence is not None,
        "unit": unit,
        "arm": arm,
        "object": obj_block,
        "command": command,
        "log": {"path": store_path(log, root), "sha256": log_sha, "bytes": log_bytes},
        "job": job_block,
        "git": {"commit": git_st.commit, "dirty": git_st.dirty,
                "dirty_files": list(git_st.dirty_files) if git_st.dirty else []},
        "versions": dict(versions or {}),
        "notes": notes,
    }
    append_entry(manifest, entry)
    return entry


# --- verify -----------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    level: str          # "FAIL" or "warn"
    line: int
    kind: str           # see KINDS, the machine-readable cause
    message: str


# What a finding is *about*, kept separate from how it is worded. Callers,
# `--selftest` first among them, assert on the kind: a check that matched a
# substring of a sentence would go green the day somebody rewords it.
KINDS = ("log", "objet", "commit", "sale", "structure", "horodatage", "valeur",
         "bras", "chiffre")


@dataclass
class VerifyReport:
    findings: list[Finding] = field(default_factory=list)
    checked: int = 0

    @property
    def failures(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "FAIL"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "warn"]

    def fail(self, line: int, kind: str, message: str) -> None:
        self.findings.append(Finding("FAIL", line, kind, message))

    def warn(self, line: int, kind: str, message: str) -> None:
        self.findings.append(Finding("warn", line, kind, message))


def verify(manifest: Path, *, root: Path = ROOT, allow_dirty: bool = False) -> VerifyReport:
    """Re-check everything that is re-checkable, and say what is not.

    Three **verdict** classes, never mixed. They are not the plan's three
    *proof* classes (§6.2, existence / validity / environment), which are a
    different taxonomy; what this implements is §6.4's rule: verify fails on a
    hash that does not come back and on a dirty tree.
      * **FAIL**: something that was true at record time is false now. A log
        that vanished or changed, an object whose bytes or size moved, a value
        that has left its log, a commit that no longer exists, a dirty entry.
        The manifest no longer holds.
      * **warn**: something that cannot be re-checked here. An object absent
        from this machine, an unknown arm, two values for the same measure, a
        number that was never literally in its log.
      * silence: everything that matched.
    """
    report = VerifyReport()
    previous_ts: datetime | None = None
    seen: dict[tuple[str, str], tuple[int, int | float]] = {}

    for line_no, raw in read_entries(manifest):
        report.checked += 1
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError as exc:
            report.fail(line_no, "structure", f"unreadable line (invalid JSON: {exc.msg})")
            continue
        if not isinstance(entry, dict):
            report.fail(line_no, "structure", "unreadable line (the entry is not a JSON object)")
            continue

        missing = [k for k in REQUIRED_FIELDS if k not in entry]
        if missing:
            report.fail(line_no, "structure",
                        f"required fields missing: {', '.join(missing)}")
            continue
        # Present is not the same as usable. The threat model here is a
        # hand-edited manifest, and a hand edit turns `"git": {...}` into
        # `"git": "abc"` as easily as it changes a digit — after which every
        # check below raised an AttributeError and the operator got a traceback
        # instead of the refusal this file promises.
        malformed = [k for k in ("object", "log", "git") if not isinstance(entry[k], dict)]
        if not isinstance(entry["value"], (int, float)) or isinstance(entry["value"], bool):
            malformed.append("value")
        if malformed:
            report.fail(line_no, "structure",
                        f"fields of unexpected type: {', '.join(sorted(malformed))}")
            continue
        if entry["schema"] != SCHEMA:
            report.warn(line_no, "structure",
                        f"schema {entry['schema']} ≠ {SCHEMA} (entry from an earlier version)")

        label = f"{entry['metric']}/{entry['arm']}"

        # -- timestamp
        try:
            ts = parse_ts(entry["recorded_at"])
        except (ValueError, TypeError):
            report.fail(line_no, "horodatage",
                        f"{label}: unreadable timestamp ({entry['recorded_at']!r})")
            ts = None
        if ts and previous_ts and ts < previous_ts:
            # Append-only means non-decreasing. A step backwards is either a
            # hand-edit or a clock jump; both deserve a look.
            report.warn(line_no, "horodatage",
                        f"{label}: timestamp earlier than the previous entry")
        previous_ts = ts or previous_ts

        # -- repository
        git_block = entry["git"]
        commit = git_block.get("commit")
        if not commit:
            report.fail(line_no, "commit", f"{label}: no commit recorded")
        elif not commit_exists(commit, root):
            report.fail(line_no, "commit", f"{label}: commit {short(commit, 12)} unknown to the repo")
        if git_block.get("dirty"):
            msg = (f"{label}: measure taken on a dirty tree "
                   f"({len(git_block.get('dirty_files') or [])} file(s)), not publishable")
            if allow_dirty:
                report.warn(line_no, "sale", msg)
            else:
                report.fail(line_no, "sale", msg)

        # -- the log IS the proof, so its disappearance is a failure, not a note
        log_block = entry["log"]
        log_path = locate(log_block.get("path", ""), root)
        log_intact = False
        if log_path is None:
            report.fail(line_no, "log", f"{label}: log not found ({log_block.get('path')!r})")
        else:
            digest, size = sha256_file(log_path)
            if digest != log_block.get("sha256"):
                report.fail(line_no, "log",
                            f"{label}: the log changed, sha256 {short(log_block.get('sha256'), 12)} "
                            f"expected, {short(digest, 12)} read ({log_path})")
            elif size != log_block.get("bytes"):
                report.fail(line_no, "log", f"{label}: log size inconsistent with the entry")
            else:
                log_intact = True

        # -- the number itself. Every check above passes on an entry whose
        #    `value` was edited by hand, because nothing above ever looks at it:
        #    the log is untouched, so its hash still matches. This is the one
        #    field that ends up in the paper, and it is the last one to become
        #    re-checkable.
        if entry["value_in_log"]:
            if log_intact and value_evidence(read_text_head(log_path), entry["value"]) is None:
                report.fail(line_no, "chiffre",
                            f"{label}: {fmt_value(entry['value'])} is not in the "
                            f"log, though the entry claims to have found it there. The "
                            f"value was edited after the recording")
        else:
            report.warn(line_no, "chiffre",
                        f"{label}: {fmt_value(entry['value'])} does not appear "
                        f"literally in the log. Derived or converted figure, to be "
                        f"justified in full")

        # -- the measured object: absent ≠ corrupted, and that nuance is the result
        obj = entry["object"]
        obj_path = locate(obj.get("ref", ""), root) if obj.get("ref") else None
        if obj.get("sha256") is None:
            report.warn(line_no, "objet", f"{label}: object without sha256 ({obj.get('ref')!r})")
        elif obj_path is None:
            report.warn(line_no, "objet",
                        f"{label}: object absent from this machine, sha256 not re-checked "
                        f"({obj.get('ref')!r})")
        else:
            digest, obj_size = sha256_file(obj_path)
            if digest != obj["sha256"]:
                report.fail(line_no, "objet",
                            f"{label}: the measured object changed, sha256 {short(obj['sha256'], 12)} "
                            f"expected, {short(digest, 12)} read ({obj_path})")
            elif obj.get("bytes") is not None and obj_size != obj["bytes"]:
                # The object's byte count is not bookkeeping here: it IS a
                # published figure. Axis 1 of the campaign is 1,770,527,533
                # bytes against 2,666,027,672 bytes. The log's size was already
                # checked for exactly this reason; leaving the object's
                # unchecked left the hole on the side whose number reaches the
                # paper.
                report.fail(line_no, "objet",
                            f"{label}: object size inconsistent with the entry, "
                            f"{thousands(obj['bytes'])} bytes declared, {thousands(obj_size)} bytes read "
                            f"({obj_path})")

        # -- cross-entry consistency
        if entry["arm"] not in ARMS:
            report.warn(line_no, "bras", f"{label}: unknown arm {entry['arm']!r}")
        key = (entry["metric"], entry["arm"])
        if key in seen and seen[key][1] != entry["value"]:
            first_line, first_value = seen[key]
            report.warn(line_no, "valeur",
                        f"{label}: two values for the same measure, {first_value} "
                        f"(line {first_line}) then {entry['value']}. `report` will show "
                        f"only the most recent one")
        seen.setdefault(key, (line_no, entry["value"]))

    return report


# --- report -----------------------------------------------------------------


def md_cell(text: str) -> str:
    return str(text).replace("|", r"\|").replace("\n", " ")


def load_entries(manifest: Path) -> list[dict]:
    out = []
    for _, raw in read_entries(manifest):
        try:
            out.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return out


def latest_per_measure(entries: list[dict]) -> tuple[list[dict], int]:
    """Newest entry per `(metric, arm)`, and how many were hidden.

    Append-only means re-measurements pile up. Showing them all in a table
    meant for a paper is how a stale number gets pasted; showing only the last
    one silently is how a disagreement gets buried. So: last one shown, count
    of hidden ones printed under the table.
    """
    best: dict[tuple[str, str], dict] = {}
    hidden = 0
    for entry in entries:
        key = (entry.get("metric"), entry.get("arm"))
        current = best.get(key)
        if current is None:
            best[key] = entry
        else:
            hidden += 1
            if entry.get("recorded_at", "") >= current.get("recorded_at", ""):
                best[key] = entry
    return list(best.values()), hidden


def render_report(entries: list[dict], *, show_all: bool, full: bool) -> str:
    hidden = 0
    if not show_all:
        entries, hidden = latest_per_measure(entries)
    entries = sorted(entries, key=lambda e: (str(e.get("metric")), str(e.get("arm")),
                                             str(e.get("recorded_at"))))

    lines = ["| metric | arm | value | unit | object | commit | job | log | date |",
             "|---|---|---|---:|---|---|---|---|---|"]
    derived = 0
    for entry in entries:
        git_block = entry.get("git") or {}
        commit = short(git_block.get("commit"), 7) + ("+dirty" if git_block.get("dirty") else "")
        job = (entry.get("job") or {}).get("id")
        # A number that is not literally in its log is not necessarily wrong,
        # but the reader of a provenance table is entitled to know which ones
        # the log backs verbatim and which ones somebody computed.
        mark = "" if entry.get("value_in_log") else " †"
        derived += 1 if mark else 0
        lines.append(
            f"| `{md_cell(entry.get('metric'))}` "
            f"| {md_cell(entry.get('arm'))} "
            f"| {md_cell(fmt_value(entry.get('value')))}{mark} "
            f"| {md_cell(entry.get('unit') or '—')} "
            f"| `{short((entry.get('object') or {}).get('sha256'))}` "
            f"| `{commit}` "
            f"| `{short(job)}` "
            f"| `{short((entry.get('log') or {}).get('sha256'))}` "
            f"| {md_cell(str(entry.get('recorded_at'))[:10])} |")

    lines.append("")
    lines.append("Provenance: `object` = truncated sha256 of the measured object · `commit` = "
                 "repository tree at the time of the measure (`+dirty` = uncommitted, **number "
                 "not publishable**) · `job` = HF Jobs identifier, whose URL requires access to "
                 "the namespace · `log` = truncated sha256 of the log. Everything is "
                 "re-checkable with `uv run ops/manifest.py verify`.")
    if derived:
        lines.append("")
        lines.append(f"† {plural(derived, 'value')}: not literally present in its own log "
                     f"(derived or converted figure). The hash guarantees the log, not the "
                     f"derivation. Justify it explicitly.")
    if hidden:
        lines.append("")
        lines.append(f"*{hidden} earlier entry(ies) hidden. Only the most recent one for each "
                     f"(metric, arm) is shown. `--all` to see them.*")

    if full:
        lines.append("")
        lines.append("### Detailed provenance")
        for entry in entries:
            obj = entry.get("object") or {}
            log_block = entry.get("log") or {}
            git_block = entry.get("git") or {}
            job = entry.get("job") or {}
            lines.append("")
            lines.append(f"#### `{entry.get('metric')}` — arm {entry.get('arm')}")
            lines.append("")
            lines.append(f"- value: **{fmt_value(entry.get('value'))}** {entry.get('unit') or ''}".rstrip())
            if entry.get("claim"):
                lines.append(f"- claim: `[[claim:{entry['claim']}]]`")
            octets = f", {thousands(obj['bytes'])} bytes" if obj.get("bytes") else ""
            revision = f", revision `{obj['revision']}`" if obj.get("revision") else ""
            lines.append(f"- object: `{obj.get('ref')}`{revision} — sha256 `{obj.get('sha256') or '—'}`"
                         f" ({obj.get('sha256_source')}){octets}")
            lines.append(f"- command: `{md_cell(entry.get('command'))}`")
            lines.append(f"- log: `{log_block.get('path')}` — sha256 `{log_block.get('sha256')}`"
                         f", {thousands(log_block.get('bytes') or 0)} bytes — "
                         + ("the number is literally in it"
                            if entry.get("value_in_log")
                            else "**the number is not in it** (derived or converted)"))
            if job.get("id"):
                url = job.get("url") or "URL not resolved"
                lines.append(f"- job: `{job['id']}` — {url} ({job.get('url_source')})")
                if job.get("durations"):
                    lines.append(f"  - durations: {json.dumps(job['durations'], ensure_ascii=False)}")
            lines.append(f"- repository: commit `{git_block.get('commit')}`, tree "
                         f"{'**dirty**' if git_block.get('dirty') else 'clean'}")
            if entry.get("versions"):
                lines.append("- versions: "
                             + ", ".join(f"{k}={v}" for k, v in sorted(entry["versions"].items())))
            if entry.get("notes"):
                lines.append(f"- note: {md_cell(entry['notes'])}")
            lines.append(f"- recorded: {entry.get('recorded_at')}")
    return "\n".join(lines)


# --- commands ---------------------------------------------------------------


def cmd_record(args) -> int:
    try:
        git_st = git_state(ROOT)
        entry = record(
            args.manifest, git_st=git_st, metric=args.metric, value=args.value,
            unit=args.unit, arm=args.arm, command=args.command, log=args.log,
            obj=args.object, obj_sha=args.object_sha256, obj_rev=args.object_rev,
            job_id=args.job, job_url=args.job_url,
            versions=dict(args.version or []), claim=args.claim, notes=args.notes,
            allow_dirty=args.allow_dirty)
    except ManifestError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1

    previous = [e for e in load_entries(args.manifest)[:-1]
                if e.get("metric") == args.metric and e.get("arm") == args.arm
                and e.get("value") != args.value]
    obj = entry["object"]
    log_block = entry["log"]
    print(f"recorded: {entry['metric']} = {fmt_value(entry['value'])} {entry['unit']} "
          f"(arm {entry['arm']})")
    octets = f"  {thousands(obj['bytes'])} bytes" if obj.get("bytes") else ""
    print(f"  object  {obj['ref']}\n          sha {short(obj['sha256'], 16)} "
          f"({obj['sha256_source']}){octets}")
    print(f"  log     {log_block['path']}\n          sha {short(log_block['sha256'], 16)}  "
          f"{thousands(log_block['bytes'])} bytes")
    print(f"  commit  {short(entry['git']['commit'], 12)} "
          f"({'DIRTY TREE' if entry['git']['dirty'] else 'clean tree'})")
    if entry["job"]:
        job = entry["job"]
        print(f"  job     {job['id']}  {job.get('url') or 'URL not resolved'} ({job['url_source']})")
    print(f"  → {store_path(args.manifest)} "
          f"({plural(len(load_entries(args.manifest)), 'record')})")

    if not entry["value_in_log"]:
        print(f"\n  WARNING  {fmt_value(entry['value'])} does not appear in {log_block['path']}."
              "\n      The hash proves the log has not moved, not that the number comes from it."
              "\n      If the value is derived (ratio, unit conversion), say so in"
              "\n      `--notes`; otherwise it is the wrong log, or the wrong number.")
    if obj["sha256_source"] == "indisponible":
        print("\n  WARNING  object without sha256: the entry says which number, not on which bytes."
              "\n      Pass `--object-sha256` again as soon as the object is reachable.")
    if previous:
        print(f"\n  WARNING  {len(previous)} earlier entry(ies) for {args.metric}/{args.arm} "
              f"with a DIFFERENT value: {', '.join(fmt_value(e['value']) for e in previous)}."
              f"\n      `report` will show only the most recent one. Check which one is right.")
    if entry["git"]["dirty"]:
        print("\n  WARNING  entry marked \"dirty tree\": `verify` will refuse it "
              "(unless `verify --allow-dirty`). It cannot go into the paper.")
    return 0


def cmd_verify(args) -> int:
    manifest = args.manifest
    if not manifest.is_file():
        print(f"no manifest at {store_path(manifest)}", file=sys.stderr)
        return 1
    report = verify(manifest, root=ROOT, allow_dirty=args.allow_dirty)

    for finding in report.findings:
        prefix = "FAIL " if finding.level == "FAIL" else "warn "
        print(f"{prefix} line {finding.line}: {finding.message}")
    clean = report.checked - len({f.line for f in report.failures})
    print(f"\n{plural(report.checked, 'record')} checked · {plural(clean, 'record')} with no "
          f"failure · {plural(len(report.failures), 'failure')} · "
          f"{plural(len(report.warnings), 'warning')}")
    if not report.failures:
        return 0

    # Flush first. The findings above are on stdout and the verdict below on
    # stderr; unflushed they arrive in the wrong order and the verdict reads as
    # an accusation without evidence. Same trap, same fix as the cost refusal
    # in `run.py`'s `cmd_launch`.
    sys.stdout.flush()
    kinds = {f.kind for f in report.failures}
    # Three failure modes with nothing in common, and conflating them sends the
    # operator hunting for a corruption that never happened.
    if kinds - {"sale", "chiffre"}:
        print("\nThe manifest no longer holds: a log, an object or a commit has moved "
              "since the recording. The numbers concerned are no longer backed by "
              "their proof.", file=sys.stderr)
    if "chiffre" in kinds:
        print("\nA value has left its log while the log itself is intact: the entry "
              "was edited by hand after the recording. That is the one forgery the "
              "hash does not see, and it is the one that reaches the paper "
              "directly.", file=sys.stderr)
    if "sale" in kinds:
        print("\nMeasures were taken on a dirty working tree: they are recorded but "
              "not publishable, since nobody can rebuild the code that produced "
              "them. `verify --allow-dirty` downgrades them to a "
              "warning.", file=sys.stderr)
    return 1


def cmd_report(args) -> int:
    entries = load_entries(args.manifest)
    if args.metric:
        entries = [e for e in entries if e.get("metric") == args.metric]
    if args.arm:
        entries = [e for e in entries if e.get("arm") == args.arm]
    if not entries:
        print("no measure recorded", file=sys.stderr)
        return 1
    print(render_report(entries, show_all=args.all, full=args.full))
    return 0


# --- selftest ---------------------------------------------------------------


def check(ok: bool, condition: bool, label: str) -> bool:
    print(("ok    " if condition else "FAIL  ") + label)
    return ok and condition


def has_fail(report: VerifyReport, kind: str) -> bool:
    """A failure is only useful if it names the right cause.

    On the kind, never on the wording: an assertion matching a substring of a
    French sentence passes again the day the sentence is rephrased, which is
    the same disease as a monotonicity test that accepts a no-op (CLAUDE.md §5).
    """
    assert kind in KINDS, f"unknown cause: {kind}"
    return any(f.kind == kind for f in report.failures)


def cmd_selftest(args) -> int:
    """Prove the verifier detects corruption. A verifier that does not is decoration.

    Same discipline as `run.py cmd_selftest`, and the same reason: the value of
    this file rests entirely on `verify` being able to fail. Every assertion
    below breaks **one** thing and requires a failure that names it — a check
    that merely counts failures would pass on a verifier that fails at random.
    """
    ok = True
    tmp = Path(tempfile.mkdtemp(prefix="llvq-manifest-selftest-"))
    try:
        head = git_state(ROOT)
        clean = GitState(commit=head.commit, dirty=False)
        dirty = GitState(commit=head.commit, dirty=True, dirty_files=("CLAUDE.md",))

        manifest = tmp / "manifest.jsonl"
        log = tmp / "run.log"
        log.write_text("ppl = 16.9415 over 12 windows\nbaseline ppl = 12.2361\n",
                       encoding="utf-8")
        obj = tmp / "objet.bin"
        obj.write_bytes(bytes(range(256)) * 16)

        common = dict(metric="selftest_ppl", unit="ppl", arm="ctrl",
                      command="LLVQ_DTYPE=f16 ppl 4096 12 cuda", log=log, obj=str(obj))

        # 1. an unclean tree is refused, and refused *before* writing anything.
        try:
            record(manifest, git_st=dirty, value=16.9415, **common)
            ok = check(ok, False, "a dirty tree is refused")
        except ManifestError:
            ok = check(ok, True, "a dirty tree is refused")
        ok = check(ok, not manifest.exists(), "a refusal writes nothing to the manifest")

        # 2. nominal path.
        entry = record(manifest, git_st=clean, value=16.9415, **common)
        report = verify(manifest, root=ROOT)
        ok = check(ok, not report.failures, "a sound entry passes verify")
        ok = check(ok, entry["value_in_log"] is True,
                   "record finds the number in the log")

        # 3. append-only: a second entry must not cost the first.
        record(manifest, git_st=clean, value=12.2361,
               **{**common, "metric": "selftest_baseline"})
        ok = check(ok, len(load_entries(manifest)) == 2, "the manifest is append-only")

        # 4. THE lock: the log's hash. Two corruptions, and the first one is
        #    length-preserving **on purpose** — an appended byte also changes
        #    the recorded size, so it alone would pass on a verifier that never
        #    compares a digest at all. A silent in-place edit is the realistic
        #    threat anyway: nobody appends to a log, people "fix" them.
        original = log.read_bytes()
        flipped = bytearray(original)
        flipped[0] ^= 0x01
        log.write_bytes(bytes(flipped))
        ok = check(ok, has_fail(verify(manifest, root=ROOT), "log"),
                   "a log edited at constant size makes verify fail")
        log.write_bytes(original + b" ")
        ok = check(ok, has_fail(verify(manifest, root=ROOT), "log"),
                   "a lengthened log makes verify fail")
        log.write_bytes(original)
        ok = check(ok, not verify(manifest, root=ROOT).failures,
                   "the restored log makes verify pass again")

        # 5. a log that disappeared is a failure too — "does it still exist?"
        log.unlink()
        ok = check(ok, has_fail(verify(manifest, root=ROOT), "log"),
                   "a vanished log makes verify fail")
        log.write_bytes(original)

        # 6. the measured object is hashed for the same reason as the log.
        obj_original = obj.read_bytes()
        obj.write_bytes(obj_original[:-1] + bytes([obj_original[-1] ^ 0xFF]))
        ok = check(ok, has_fail(verify(manifest, root=ROOT), "objet"),
                   "a corrupted object makes verify fail")
        obj.write_bytes(obj_original)

        # 7. a commit the repository does not know is not provenance.
        forged = json.loads(read_entries(manifest)[0][1])
        forged["git"] = {"commit": "0" * 40, "dirty": False, "dirty_files": []}
        forged["metric"] = "selftest_commit_forge"
        append_entry(manifest, forged)
        ok = check(ok, has_fail(verify(manifest, root=ROOT), "commit"),
                   "a nonexistent commit makes verify fail")

        # 8. a dirty entry is recordable, and rejected by default.
        dirty_manifest = tmp / "dirty.jsonl"
        entry = record(dirty_manifest, git_st=dirty, value=1.0, allow_dirty=True, **common)
        ok = check(ok, entry["git"]["dirty"] is True, "--allow-dirty marks the entry")
        ok = check(ok, has_fail(verify(dirty_manifest, root=ROOT), "sale"),
                   "a dirty entry makes verify fail by default")
        ok = check(ok, not verify(dirty_manifest, root=ROOT, allow_dirty=True).failures,
                   "verify --allow-dirty accepts it as a warning")

        # 9. an unreadable line is a failure, not a silent skip.
        broken = tmp / "broken.jsonl"
        broken.write_text("{ this is not JSON\n", encoding="utf-8")
        ok = check(ok, has_fail(verify(broken, root=ROOT), "structure"),
                   "an unreadable line makes verify fail")

        # 10. a truncated entry cannot pass as a complete one.
        amputated = tmp / "amputee.jsonl"
        partial = json.loads(read_entries(manifest)[0][1])
        partial.pop("log")
        append_entry(amputated, partial)
        ok = check(ok, has_fail(verify(amputated, root=ROOT), "structure"),
                   "a truncated entry makes verify fail")

        # 10bis. …and an entry whose fields have the wrong TYPE must be refused,
        #        not crash. A hand edit breaks a type as easily as a digit, and
        #        a traceback is not a refusal an operator can act on.
        # WARNING: one file per case, numbered. `append_entry` appends: two
        #    cases sharing a filename means the second one is verified against
        #    the first one's failure and passes for free. That is exactly how
        #    the boolean case below went green on a verifier that took booleans.
        for case, (label, key, junk) in enumerate((
                ("git block", "git", "abc"),
                ("log block", "log", None),
                ("object block", "object", []),
                ("text value", "value", "16,9415"),
                # `isinstance(True, int)` is True in Python, so a boolean walks
                # straight through a numeric check written the obvious way.
                ("boolean value", "value", True))):
            typed = tmp / f"type-{case}.jsonl"
            entry = json.loads(read_entries(manifest)[0][1])
            entry[key] = junk
            append_entry(typed, entry)
            try:
                ok = check(ok, has_fail(verify(typed, root=ROOT), "structure"),
                           f"{label} of unexpected type makes verify fail")
            except Exception as exc:  # a traceback is a failure of this check
                ok = check(ok, False, f"{label} of unexpected type crashes verify: {exc!r}")

        # 11. the report shows the newest value, and says how many it hides.
        record(manifest, git_st=clean, value=16.9415, **common)
        record(manifest, git_st=clean, value=17.5, **common)
        rendered = render_report(load_entries(manifest), show_all=False, full=False)
        ok = check(ok, "17.5" in rendered and "hidden" in rendered,
                   "report shows the most recent value and flags the hidden ones")

        # 12. the French decimal comma round-trips; a thousands separator would
        #     not be unambiguous, and is left alone on purpose.
        ok = check(ok, parse_value("16,9415") == 16.9415 and parse_value("1770527533") == 1770527533,
                   "parse_value reads the decimal comma and keeps integers exact")

        # 13. absent ≠ corrupted. An object living on the Hub (arm B0) is
        #     recordable with a declared sha, warns that it could not be
        #     re-hashed here, and must NOT fail. This is the check that pins
        #     the FAIL/warn boundary: without it, a verifier that downgraded
        #     every failure to a warning — and therefore exited 0 on a
        #     corrupted log — would still satisfy every assertion above.
        remote = tmp / "remote.jsonl"
        record(remote, git_st=clean, value=4.15625, allow_dirty=False,
               **{**common, "metric": "selftest_remote", "unit": "b/weight",
                  "obj": "Qwen/Qwen3-4B-AWQ", "obj_sha": "74d4bd2b" + "0" * 56})
        remote_report = verify(remote, root=ROOT)
        ok = check(ok, not remote_report.failures,
                   "a remote object, not hashable here, does not make verify fail")
        ok = check(ok, any("object absent" in f.message for f in remote_report.warnings),
                   "a remote object is flagged as a warning")
        ok = check(ok, not has_fail(remote_report, "objet"),
                   "a warning is never counted as a failure")

        # 14. a declared sha that contradicts the local bytes is a refusal, not
        #     a preference. This is how one records the 4B's sha against the
        #     8B's file — the very confusion `docs/fiche-4b.md` was written to
        #     end. `record` hashes what is in front of it and trusts that.
        wrong = tmp / "wrong.jsonl"
        try:
            record(wrong, git_st=clean, value=1.0,
                   **{**common, "obj_sha": "dead" + "0" * 60})
            ok = check(ok, False, "a declared sha that is wrong is refused")
        except ManifestError:
            ok = check(ok, not wrong.exists(),
                       "a declared sha that is wrong is refused, writing nothing")

        # …and a truncated one is refused on its shape alone. Tested on a
        # REMOTE object on purpose: on a local file the mismatch above would
        # catch it anyway, so a local fixture would leave the shape check
        # untested — a mutant deleting it survived exactly that way.
        for label, bad in (("truncated", "aa"), ("non-hexadecimal", "z" * 64)):
            try:
                record(wrong, git_st=clean, value=1.0,
                       **{**common, "obj": "Qwen/Qwen3-4B", "obj_sha": bad})
                ok = check(ok, False, f"a declared sha that is {label} is refused")
            except ManifestError:
                ok = check(ok, True, f"a declared sha that is {label} is refused")

        # 15. the remaining refusals of `record`. Each is a 2 a.m. typo, and
        #     each would otherwise produce an entry that `verify` can never
        #     repair: a mistyped arm silently splits a comparison in two, an
        #     empty command loses the reproduction, an absent log loses the
        #     proof. Argparse blocks the first two at the CLI; `record` is
        #     called programmatically too, and the guard belongs where the
        #     entry is built.
        guard = tmp / "guard.jsonl"
        for label, override in (("unknown arm", {"arm": "A"}),
                                ("empty command", {"command": "   "}),
                                ("missing log", {"log": tmp / "pas-de-log.txt"}),
                                ("job URL without an identifier",
                                 {"job_url": "https://huggingface.co/jobs/x/y"})):
            try:
                record(guard, git_st=clean, value=1.0, **{**common, **override})
                ok = check(ok, False, f"record refuses: {label}")
            except ManifestError:
                ok = check(ok, not guard.exists(), f"record refuses: {label}")

        # 16bis. the recorded size must agree with the hashed file. This can
        #        only ever fire on an entry edited by hand — a corrupted log
        #        changes the digest first — which is precisely the threat a
        #        tamper-evident manifest is for. It went untested until a
        #        mutant deleting the branch survived.
        resized = tmp / "taille.jsonl"
        entry = json.loads(read_entries(manifest)[0][1])
        entry["log"]["bytes"] = entry["log"]["bytes"] + 1
        append_entry(resized, entry)
        ok = check(ok, has_fail(verify(resized, root=ROOT), "log"),
                   "a log size rewritten by hand makes verify fail")

        # 16ter. a blank line is tolerated, not treated as a broken entry.
        #        Somebody will open this file in an editor one day; the
        #        manifest must survive a stray newline without crying wolf,
        #        and that tolerance has to be pinned or it is accidental.
        spaced = tmp / "espace.jsonl"
        good = read_entries(manifest)[0][1]
        spaced.write_text(good + "\n" + good, encoding="utf-8")
        ok = check(ok, not verify(spaced, root=ROOT).failures,
                   "a blank line does not make verify fail")

        # 16. the comparison is on the WHOLE digest, not on what gets printed.
        #     `short()` lives in this file and shows 8 characters in every
        #     report; comparing the displayed prefix instead of the recorded
        #     value is one careless refactor away, and it would accept any
        #     forgery agreeing on 8 hex characters. This entry carries hashes
        #     that agree on their first 8 and differ on all the others.
        def prefix_forgery(digest: str) -> str:
            return digest[:8] + "".join("0" if c != "0" else "1" for c in digest[8:])

        forged_hash = tmp / "prefixe.jsonl"
        entry = json.loads(read_entries(manifest)[0][1])
        entry["log"]["sha256"] = prefix_forgery(entry["log"]["sha256"])
        entry["object"]["sha256"] = prefix_forgery(entry["object"]["sha256"])
        append_entry(forged_hash, entry)
        prefix_report = verify(forged_hash, root=ROOT)
        ok = check(ok, has_fail(prefix_report, "log"),
                   "a log sha matching on 8 characters is rejected")
        ok = check(ok, has_fail(prefix_report, "objet"),
                   "an object sha matching on 8 characters is rejected")

        # 17. the object's SIZE, for the same reason the log's is checked — and
        #     it matters more here: the object's byte count is not bookkeeping,
        #     it IS a published figure (axis 1 is 1,770,527,533 bytes against
        #     2,666,027,672 bytes). Hashes and sizes are checked on the log side and
        #     were not on the object side; a mutant deleting the object branch
        #     survived every assertion above.
        obj_resized = tmp / "objet-taille.jsonl"
        entry = json.loads(read_entries(manifest)[0][1])
        entry["object"]["bytes"] = entry["object"]["bytes"] + 1
        append_entry(obj_resized, entry)
        ok = check(ok, has_fail(verify(obj_resized, root=ROOT), "objet"),
                   "an object size rewritten by hand makes verify fail")

        # 18. THE number. Everything above passes on an entry whose `value` was
        #     edited by hand: the log is untouched, so every hash still matches
        #     and the verifier had nothing to say about the one field that ends
        #     up in the paper. Measured before the fix: 0 failure, 0 warning.
        edited = tmp / "valeur.jsonl"
        entry = json.loads(read_entries(manifest)[0][1])
        entry["value"] = 12.0
        append_entry(edited, entry)
        edited_report = verify(edited, root=ROOT)
        ok = check(ok, has_fail(edited_report, "chiffre"),
                   "a value rewritten by hand makes verify fail")
        ok = check(ok, not has_fail(edited_report, "log"),
                   "…and the failure accuses the value, not the log, which has not moved")

        #     …and deleting the field must not be a way out. That is why it is
        #     required: an omission is a structural failure, not a silence.
        stripped = tmp / "sans-champ.jsonl"
        entry = json.loads(read_entries(manifest)[0][1])
        entry["value"] = 12.0
        entry.pop("value_in_log")
        append_entry(stripped, entry)
        ok = check(ok, has_fail(verify(stripped, root=ROOT), "structure"),
                   "deleting value_in_log does not bypass the check")

        # 19. a value that was never in its log is recorded as such, warned
        #     about, and NOT failed — a ratio or a unit conversion is legitimate
        #     provenance. This pins the boundary: a verifier that failed here
        #     would satisfy check 18 while making the tool unusable, and one
        #     that stayed silent would make check 18 unreachable.
        derived_mf = tmp / "derive.jsonl"
        derived_entry = record(derived_mf, git_st=clean, value=1.5058,
                               **{**common, "metric": "selftest_ratio", "unit": "×"})
        ok = check(ok, derived_entry["value_in_log"] is False,
                   "a number absent from the log is recorded as such")
        derived_report = verify(derived_mf, root=ROOT)
        ok = check(ok, not derived_report.failures,
                   "a derived number does not make verify fail")
        ok = check(ok, any(f.kind == "chiffre" for f in derived_report.warnings),
                   "a derived number is flagged as a warning")
        ok = check(ok, "†" in render_report(load_entries(derived_mf), show_all=True, full=False),
                   "report marks the number the log does not carry")

        # 20. the matcher itself. A presence check that matched everything
        #     would make 18 pass on a broken verifier, so it needs its own
        #     negative controls — the same reason `golay_stage_is_load_bearing`
        #     exists (CLAUDE.md §5).
        cases: tuple[tuple[str, str, int | float, bool], ...] = (
            ("decimal comma", "ppl = 16,9415 over 12 windows", 16.9415, True),
            ("decimal point", "ppl = 16.9415", 16.9415, True),
            ("more decimals in the log", "ppl = 16.94153", 16.9415, True),
            ("fewer decimals in the log", "ppl = 16.94", 16.9415, False),
            ("integer grouped by spaces", "size 1 770 527 533 bytes", 1770527533, True),
            ("bare integer", "size 1770527533 bytes", 1770527533, True),
            ("typographic minus sign", "drop −14.33 pp", -14.33, True),
            ("a different number", "ppl = 12.2361", 16.9415, False),
            ("a prefix is not a proof", "ppl = 16.94", 16.941, False),
            # The trap the grouping rule is written against: collapsing any
            # digit-space-digit would turn this into `409612` and invent a
            # match for a value nobody logged.
            ("no merging outside groups of three", "ppl 4096 12 cuda", 409612, False),
        )
        for label, text, value, expected in cases:
            ok = check(ok, (value_evidence(text, value) is not None) is expected,
                       f"value_evidence — {label}")
    except ManifestError as exc:
        print(f"FAIL  selftest interrupted: {exc}")
        ok = False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + ("selftest green: verify can fail, so it verifies."
                  if ok else "selftest RED: do not trust the manifest."))
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--manifest", type=Path, default=MANIFEST,
                   help=f"default: {store_path(MANIFEST)}")
    p.add_argument("--selftest", action="store_true",
                   help="record, verify, corrupt, require the failure")
    sub = p.add_subparsers(dest="cmd")

    r = sub.add_parser("record", help="record a measure and its provenance")
    r.add_argument("--metric", required=True, help="e.g. ppl_wikitext2_ctx4096_w73")
    r.add_argument("--value", required=True, type=parse_value,
                   help="decimal comma accepted (16,9415)")
    r.add_argument("--unit", required=True, help="e.g. ppl, pp, bytes, b/weight, tok/s, s")
    r.add_argument("--arm", required=True, choices=sorted(ARMS),
                   help="; ".join(f"{k} = {v}" for k, v in ARMS.items()))
    r.add_argument("--log", required=True, type=Path, help="the log that carries the number")
    r.add_argument("--object", required=True,
                   help="the measured object: local path (hashed) or Hub reference")
    r.add_argument("--command", required=True, help="the EXACT command, environment included")
    r.add_argument("--object-sha256", default=None,
                   help="sha of the file if it is not reachable here; checked if it is")
    r.add_argument("--object-rev", default=None, help="Hub revision of the object")
    r.add_argument("--job", default=None, help="HF Jobs identifier; the URL is resolved by the API")
    r.add_argument("--job-url", default=None,
                   help="URL of the job if the API is out of reach (marked \"declare\")")
    r.add_argument("--version", action="append", type=parse_kv, metavar="KEY=VALUE",
                   help="rustc=…, cuda=…, driver=…, candle=… (repeatable)")
    r.add_argument("--claim", default=None, help="[[claim:ID]] identifier on the paper side")
    r.add_argument("--notes", default=None)
    r.add_argument("--allow-dirty", action="store_true",
                   help="record from a dirty tree; the entry is marked and verify refuses it")
    r.set_defaults(fn=cmd_record)

    v = sub.add_parser("verify", help="re-check hashes, commits and cleanliness")
    v.add_argument("--allow-dirty", action="store_true",
                   help="downgrade \"dirty tree\" entries to a warning")
    v.set_defaults(fn=cmd_verify)

    rp = sub.add_parser("report", help="markdown table, each row with its provenance")
    rp.add_argument("--metric", default=None)
    rp.add_argument("--arm", default=None, choices=sorted(ARMS))
    rp.add_argument("--all", action="store_true",
                    help="also show the superseded measures")
    rp.add_argument("--full", action="store_true",
                    help="add the detailed provenance under the table")
    rp.set_defaults(fn=cmd_report)

    st = sub.add_parser("selftest", help="alias for --selftest")
    st.set_defaults(fn=cmd_selftest)

    args = p.parse_args()
    if args.selftest:
        return cmd_selftest(args)
    if not getattr(args, "fn", None):
        p.print_help()
        return 2
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
