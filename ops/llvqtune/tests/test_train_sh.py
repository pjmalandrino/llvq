"""train.sh, run for real, with `python -m llvqtune` replaced by a stand-in.

The stand-in records every llvqtune call and writes the probe's `closed`
record with a rate, a first loss and a gauge chosen by the test. Every other
`python` call (the parsers, the arithmetic) goes to the real interpreter, so
the script's own logic is what is under test.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

TRAIN_SH = Path(__file__).resolve().parent.parent / "train.sh"

SHIM = r"""#!/usr/bin/env bash
if [ "${1:-}" = "-m" ] && [ "${2:-}" = "llvqtune" ]; then
  printf '%s\n' "$*" >> "$SHIM_LOG"
  journal=""; prev=""
  for a in "$@"; do
    if [ "$prev" = "--journal" ]; then journal="$a"; fi
    prev="$a"
  done
  gauge='{"max_memory_allocated": 74000000000}'
  if [ -n "${SHIM_NOGAUGE:-}" ]; then gauge=null; fi
  if [ -n "${SHIM_RATE:-}" ]; then
    printf '{"event": "closed", "seconds_per_step": %s, "first_loss": %s, "gauge": %s}\n' \
      "$SHIM_RATE" "${SHIM_KL:-0.3}" "$gauge" >> "$journal"
  fi
  exit "${SHIM_EXIT:-0}"
fi
exec "$REAL_PYTHON" "$@"
"""

TRUNCATING_CP = r"""#!/usr/bin/env bash
# A `cp` that loses the second half of every file, as a bucket read might.
for last; do :; done
for src in "$@"; do
  [ "$src" = "$last" ] && break
  [ "$src" = "-v" ] && continue
  n=$(wc -c < "$src")
  head -c $(( n / 2 )) "$src" > "$last/$(basename "$src")"
done
"""


def _tool(bin_dir: Path, name: str, body: str) -> None:
    path = bin_dir / name
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def train(tmp_path: Path, extra_tools: dict[str, str] | None = None, **env):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    _tool(bin_dir, "python", SHIM)
    for name, body in (extra_tools or {}).items():
        _tool(bin_dir, name, body)
    export = tmp_path / "export"
    export.mkdir(exist_ok=True)
    (export / "model.safetensors").write_bytes(b"\x00" * 4096)
    (export / "config.json").write_text("{}")
    log = tmp_path / "calls.log"
    full = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "REAL_PYTHON": sys.executable,
        "SHIM_LOG": str(log),
        "EXPORT": str(export),
        "OUT": str(tmp_path / "out"),
        "TEACHER": "Qwen/Qwen3-8B",
        "SHIM_RATE": "0.5",
    }
    for k, v in env.items():
        if v is None:
            full.pop(k, None)
        else:
            full[k] = v
    done = subprocess.run(
        ["bash", str(TRAIN_SH)], env=full, cwd=tmp_path,
        capture_output=True, text=True, timeout=60,
    )
    calls = log.read_text().splitlines() if log.exists() else []
    return done, calls


def steps_of(call: str) -> int:
    words = call.split()
    return int(words[words.index("--steps") + 1])


def student_of(call: str) -> str:
    words = call.split()
    return words[words.index("--student") + 1]


def corpus_of(call: str) -> str:
    """The value argparse would read: the LAST `--corpus` on the line.

    Reading the first would let a second, contradicting flag appended further
    down pass unseen, which is exactly the probe-on-other-text mistake.
    """
    words = call.split()
    last = len(words) - 1 - words[::-1].index("--corpus")
    return words[last + 1]


def test_the_default_corpus_is_the_one_every_published_arm_ran_on(tmp_path):
    done, calls = train(tmp_path, BUDGET="100")
    assert done.returncode == 0, done.stdout + done.stderr
    assert [corpus_of(c) for c in calls] == ["dclm", "dclm"]


def test_the_probe_reads_the_corpus_the_run_will_read(tmp_path):
    """A rate measured on other text prices the wrong run."""
    done, calls = train(tmp_path, BUDGET="100", CORPUS="mmlu-aux")
    assert done.returncode == 0, done.stdout + done.stderr
    assert [corpus_of(c) for c in calls] == ["mmlu-aux", "mmlu-aux"]
    assert "corpus mmlu-aux" in done.stdout


def test_the_mix_ratio_reaches_both_calls(tmp_path):
    done, calls = train(tmp_path, BUDGET="100", CORPUS="mix", MIX_RATIO="0.25")
    assert done.returncode == 0, done.stdout + done.stderr
    for call in calls:
        words = call.split()
        assert words[words.index("--mix-ratio") + 1] == "0.25"


def test_without_steps_the_budget_decides(tmp_path):
    done, calls = train(tmp_path, BUDGET="100")
    assert done.returncode == 0, done.stdout + done.stderr
    assert len(calls) == 2
    assert steps_of(calls[0]) == 6          # the probe
    assert steps_of(calls[1]) == 200        # 100 s / 0.5 s


def test_steps_bypasses_the_budget(tmp_path):
    """The 8B must read the 4B's 9,507 steps, whatever its card's rate."""
    done, calls = train(tmp_path, BUDGET="100", STEPS="9507")
    assert done.returncode == 0, done.stdout + done.stderr
    assert steps_of(calls[1]) == 9507
    assert "BUDGET is not read" in done.stdout


def test_a_malformed_steps_is_refused_before_the_probe(tmp_path):
    for bad in ("abc", "0", "-5", "9507.0"):
        done, calls = train(tmp_path / bad.replace(".", "_"), STEPS=bad)
        assert done.returncode == 2, (bad, done.stdout + done.stderr)
        assert calls == [], bad


def test_the_probe_must_close_even_when_steps_is_given(tmp_path):
    """A probe that died (an OOM, say) left no rate: no training, STEPS or not."""
    done, calls = train(tmp_path, STEPS="9507", SHIM_RATE="", SHIM_EXIT="1")
    assert done.returncode == 2
    assert len(calls) == 1
    assert "wrote no rate" in done.stdout


def test_a_refused_probe_stops_the_run(tmp_path):
    """Exit 2 is a refused wiring, the teacher pairing among them."""
    done, calls = train(tmp_path, STEPS="9507", SHIM_EXIT="2")
    assert done.returncode == 2
    assert len(calls) == 1
    assert "refused its wiring" in done.stdout


def test_there_is_no_default_teacher(tmp_path):
    done, calls = train(tmp_path, TEACHER=None)
    assert done.returncode != 0
    assert calls == []
    assert "Qwen/Qwen3-8B" in done.stderr   # the message names an example


def test_the_teacher_given_is_the_teacher_passed(tmp_path):
    done, calls = train(tmp_path, TEACHER="Qwen/Qwen3-8B")
    assert done.returncode == 0, done.stdout + done.stderr
    assert all("--teacher Qwen/Qwen3-8B" in c for c in calls)


def test_the_projection_over_the_ceiling_is_refused(tmp_path):
    done, calls = train(tmp_path, STEPS="9507", MAX_TRAIN_SECONDS="1000")
    assert done.returncode == 3
    assert len(calls) == 1                  # the probe only


def test_the_projection_under_the_ceiling_runs(tmp_path):
    done, calls = train(tmp_path, STEPS="9507", MAX_TRAIN_SECONDS="8400")
    assert done.returncode == 0, done.stdout + done.stderr   # 4,753 s projected
    assert len(calls) == 2


def test_a_first_loss_over_the_ceiling_is_refused(tmp_path):
    done, calls = train(tmp_path, STEPS="9507", SHIM_KL="0.9", MAX_FIRST_KL="0.5")
    assert done.returncode == 4
    assert len(calls) == 1


def test_a_first_loss_under_the_ceiling_runs(tmp_path):
    done, calls = train(tmp_path, STEPS="9507", SHIM_KL="0.3", MAX_FIRST_KL="0.5")
    assert done.returncode == 0, done.stdout + done.stderr
    assert len(calls) == 2


def test_the_probe_gauge_is_printed(tmp_path):
    done, _ = train(tmp_path)
    assert '"max_memory_allocated":74000000000' in done.stdout


def test_a_probe_without_a_gauge_stops_the_run(tmp_path):
    """On cuda the gauge is always bound; its absence is a wiring fault."""
    done, calls = train(tmp_path, STEPS="9507", SHIM_NOGAUGE="1")
    assert done.returncode == 5
    assert len(calls) == 1


def test_staging_copies_and_trains_from_the_copy(tmp_path):
    stage = tmp_path / "stage"
    done, calls = train(tmp_path, STAGE=str(stage))
    assert done.returncode == 0, done.stdout + done.stderr
    assert (stage / "model.safetensors").stat().st_size == 4096
    assert [student_of(c) for c in calls] == [str(stage), str(stage)]


def test_without_stage_the_export_is_read_where_it_is(tmp_path):
    done, calls = train(tmp_path)
    assert [student_of(c) for c in calls] == [str(tmp_path / "export")] * 2


def test_a_truncated_copy_is_refused(tmp_path):
    done, calls = train(tmp_path, STAGE=str(tmp_path / "stage"),
                        extra_tools={"cp": TRUNCATING_CP})
    assert done.returncode != 0
    assert calls == []
    assert "staged; refusing" in done.stdout + done.stderr


@pytest.mark.skipif(not TRAIN_SH.exists(), reason="train.sh sits beside tests/")
def test_the_script_parses():
    assert subprocess.run(["bash", "-n", str(TRAIN_SH)]).returncode == 0
