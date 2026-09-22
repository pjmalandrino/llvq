# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface-hub>=1.26", "requests"]
# ///
"""Read the Space's last build log and check the rebuild did what it was published for. $0, read-only.

    uv run space-build-log.py [OUT.txt]

Prints the stage, the sha, the duration from "Build Queued" to the last DONE, and one line per
marker. Exits 1 if any marker is missing, or if the log is not of the Space's current sha. The log is streamed (SSE); the stream is cut after
40 s, which is enough for a finished build (687 lines for a963a020, fetched in < 5 s).
"""
import json
import re
import sys
import time
from datetime import datetime

import requests
from huggingface_hub import HfApi, get_token

SPACE = "Pier-Jean/llvq-runner-cuda"
MARKERS = {
    "cargo line seal+export+rowscale": r"--> RUN cargo build --release --locked -p llvq-llm --bin seal --bin export --bin rowscale",
    # Names the two new binaries: a marker on `seal` alone also matched a963a020's COPY, so it
    # could not tell the rebuilt image from the old one.
    "runtime COPY with export+rowscale": r"--> COPY --from=build .*?/src/target/release/seal\s+/src/target/release/export\s+/src/target/release/rowscale\s",
    "8B+14B configs test -f": r"--> RUN test -f /usr/local/share/llvq/configs/qwen3-8b-tetra-q5\.json",
    "export+rowscale RUN smoke": r"--> RUN /usr/local/bin/export 2>&1",
    "image pushed": r"--> Pushing image",
}

s = HfApi().space_info(SPACE)
stage = s.runtime.stage if s.runtime else None
print(f"space {SPACE} sha {s.sha} stage {stage} last_modified {s.last_modified}")

lines, start = [], time.time()
url = f"https://huggingface.co/api/spaces/{SPACE}/logs/build"
with requests.get(url, headers={"Authorization": f"Bearer {get_token()}"}, stream=True, timeout=(10, 20)) as r:
    r.raise_for_status()
    try:
        for raw in r.iter_lines(decode_unicode=True):
            if raw and raw.startswith("data:"):
                try:
                    d = json.loads(raw[5:])
                    lines.append(f"{d.get('timestamp')} {d.get('data', '').rstrip()}")
                except json.JSONDecodeError:
                    lines.append(raw)
            if time.time() - start > 40:
                break
    except requests.exceptions.RequestException:
        pass  # the server keeps the stream open after the last line; a read timeout ends it

if len(sys.argv) > 1:
    open(sys.argv[1], "w").write("\n".join(lines) + "\n")

text = "\n".join(lines)
head = re.search(r"Build Queued at .*? / Commit SHA: (\w+)", text)
print(f"{len(lines)} log lines; build of {head.group(1) if head else '?'}")
# Fatal, not a warning: a log of another commit proves nothing about the image a Job gets
# now. Right after `publish` the Space sha moves at once while the served log (and stage)
# can still be the previous build's; a second publish would leave a first rebuild's log,
# markers and all, in front of an unbuilt commit.
wrong_build = head is None or not s.sha.startswith(head.group(1))
if wrong_build:
    print(f"REFUSED: the log is of {head.group(1) if head else 'no identifiable build'}, the "
          f"Space is at {s.sha[:7]}: the build of the latest commit has not started, or its "
          "log is not the one served")
stamps = [datetime.fromisoformat(l.split(" ", 1)[0].replace("Z", "+00:00"))
          for l in lines if re.match(r"^\d{4}-\d\d-\d\dT", l)]
if stamps:
    print(f"build span: {(max(stamps) - min(stamps)).total_seconds() / 60:.1f} min "
          f"({min(stamps):%H:%M:%S} -> {max(stamps):%H:%M:%S} UTC)")
bad = 0
for name, pat in MARKERS.items():
    idx = next((i for i, l in enumerate(lines) if re.search(pat, l)), None)
    done = False
    if idx is not None:
        # The step's own verdict: the first DONE/CACHED/ERROR after its "-->" line, before
        # the next step starts. BuildKit prints the steps of this recipe one after another.
        for l in lines[idx + 1:]:
            if re.search(r" --> ", l):
                break
            if re.search(r" DONE [0-9.]+s$| CACHED$", l):
                done = True
                break
            if re.search(r"ERROR", l):
                break
    print(f"  {'ok     ' if done else 'MISSING'} {name}")
    bad += not done
for l in lines:
    if re.search(r"error(\[|:)|ERROR|failed to", l):
        print("  log:", l[:200])
sys.exit(1 if bad or wrong_build or stage in ("BUILDING", "BUILD_ERROR", "CONFIG_ERROR") else 0)
