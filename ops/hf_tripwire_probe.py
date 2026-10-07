#!/usr/bin/env python3
"""Can a published LLVQ repository refuse to load instead of loading garbage?

    <a python with llvq-tetra installed> ops/hf_tripwire_probe.py \\
        llvq-tetra/tests/fixtures/mini

`transformers` has no entry-point discovery for quantizers: `quantizers/auto.py`
carries no such mechanism, and the only `entry_points` in the whole package is
unrelated. So the method is registered by `import llvq_tetra` and by nothing
else. A tool that imports only `transformers` reads `quant_method: "llvq"`,
prints "Unknown quantization type ... we will skip the quantization", SKIPS it,
reinitializes every weight it then finds missing, and runs. On the published 4B
that is 254 missing keys, 1119 unexpected, and a forward pass whose logits look
perfectly normal. A benchmark would score a randomly initialized model and
publish the number as ours.

This probe asks whether the repository can make that failure loud, by trying
three layouts against four ways of loading. It writes nothing outside a temp
directory and publishes nothing.

Run it with an interpreter that HAS llvq-tetra installed: the `bare` case is
about the user's own `import`, not about the package being absent.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

CONFIGURATION = '''\
"""Importing this registers the LLVQ quantization method."""
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

try:
    import llvq_tetra  # noqa: F401
except ImportError as e:
    raise ImportError(
        "This model is in the LLVQ Tetra format and needs its reader: "
        "`pip install llvq-tetra`."
    ) from e


class LlvqQwen3Config(Qwen3Config):
    model_type = "{model_type}"
'''

MODELING = '''\
"""Importing this registers the LLVQ quantization method."""
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
{config_import}
try:
    import llvq_tetra  # noqa: F401
except ImportError as e:
    raise ImportError(
        "This model is in the LLVQ Tetra format and needs its reader: "
        "`pip install llvq-tetra`."
    ) from e


class LlvqQwen3ForCausalLM(Qwen3ForCausalLM):
    {config_class}
'''

# One case per way a caller reaches the model. `bare` is the one that matters:
# a benchmark harness, which imports transformers and nothing of ours.
CASE = '''\
import sys
import torch
from transformers import AutoModelForCausalLM

if "imported" in sys.argv[2]:
    import llvq_tetra  # noqa: F401

kw = {"dtype": torch.float32, "output_loading_info": True}
if "flag" in sys.argv[2]:
    kw["trust_remote_code"] = True

try:
    model, info = AutoModelForCausalLM.from_pretrained(sys.argv[1], **kw)
except Exception as e:
    print(f"REFUSED {type(e).__name__}: {str(e).splitlines()[0][:110]}")
    sys.exit(0)

n = sum(1 for m in model.modules()
        if type(m).__name__ in ("TetraLinear", "Int4Linear"))
clean = not info["missing_keys"] and not info["unexpected_keys"]
print(f"LOADED {n} records, keys clean {clean} "
      f"-> {'correct' if (n and clean) else 'A RANDOM MODEL'}")
'''

LAYOUTS = {
    "as published": {},
    "auto_map on the model": {
        "auto_map": {"AutoModelForCausalLM": "modeling_llvq.LlvqQwen3ForCausalLM"},
    },
    "auto_map and an unknown model_type": {
        "model_type": "llvq_qwen3",
        "auto_map": {
            "AutoConfig": "configuration_llvq.LlvqQwen3Config",
            "AutoModelForCausalLM": "modeling_llvq.LlvqQwen3ForCausalLM",
        },
    },
}
CASES = ("bare", "flag", "imported")


def build(src: Path, dest: Path, patch: dict) -> None:
    shutil.copytree(src, dest, dirs_exist_ok=True)
    cfg = json.loads((dest / "config.json").read_text())
    cfg.update(patch)
    (dest / "config.json").write_text(json.dumps(cfg, indent=1))
    model_type = patch.get("model_type", cfg["model_type"])
    (dest / "configuration_llvq.py").write_text(CONFIGURATION.format(model_type=model_type))
    # `config_class` on the model must match what `AutoConfig` resolves. Declaring
    # it while `AutoConfig` is NOT mapped makes transformers refuse on an
    # inconsistency that has nothing to do with the question asked here.
    mapped = "AutoConfig" in patch.get("auto_map", {})
    (dest / "modeling_llvq.py").write_text(MODELING.format(
        config_import="\nfrom .configuration_llvq import LlvqQwen3Config\n" if mapped else "",
        config_class="config_class = LlvqQwen3Config" if mapped else "pass",
    ))


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    src = Path(argv[1])
    if not (src / "config.json").exists():
        print(f"REFUSED: {src} is not a packed model", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory() as tmp:
        runner = Path(tmp) / "case.py"
        runner.write_text(CASE)
        width = max(len(c) for c in CASES)
        for name, patch in LAYOUTS.items():
            print(f"\n== {name} ==")
            d = Path(tmp) / name.replace(" ", "-")
            build(src, d, patch)
            for case in CASES:
                # A subprocess per case: registration is a module-level side
                # effect, so one interpreter cannot answer twice.
                r = subprocess.run(
                    [sys.executable, str(runner), str(d), case],
                    capture_output=True, text=True, timeout=900,
                    # `bare` must not inherit a yes: the prompt is the refusal.
                    env={"HF_HUB_DISABLE_TELEMETRY": "1", "PATH": "/usr/bin:/bin"},
                    stdin=subprocess.DEVNULL,
                )
                # Not `startswith`: when a repository needs remote code,
                # transformers prints "Do you wish to run the custom code?"
                # with no trailing newline, so the verdict lands mid-line.
                line = next(
                    (ln[ln.index("LOADED") if "LOADED" in ln else ln.index("REFUSED"):]
                     for ln in r.stdout.splitlines()
                     if "LOADED" in ln or "REFUSED" in ln),
                    f"no verdict, exit {r.returncode}: {r.stderr.strip().splitlines()[-1:]}",
                )
                print(f"  {case:<{width}}  {line}")

    print(textwrap.dedent("""
        Read the `bare` row of each layout. It is the only one a third-party tool
        takes, and the only one where silence is dangerous."""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
