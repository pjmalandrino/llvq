# Prereg. Stage 5, the TestPyPI half: what a stranger gets from an index

Status: written and committed on 2026-10-07, BEFORE the upload to TestPyPI and before the first load.
Operator go given 2026-10-07 on steps 1 to 5 of the PyPI route. Cost: 0 $, Mac, no paid job.
Plan: `docs/plan-transformers.md`. Earlier gates: stage 1, `proofs/preregistration-hf-quantizer-2026-09-30.md`;
the cleanroom of 2026-10-01 and 10-02, `docs/mesures/hf-cleanroom-4b-2026-10-01.txt`, which had no prereg and says so.

A timestamped prereg is no longer edited. Any departure goes in
`proofs/preregistration-hf-testpypi-2026-10-07-ECARTS.md`, beside it and never into it.

## 1. Question

Does `llvq-tetra` 0.1.0, uploaded to TestPyPI and installed from it into an empty environment, load the published 4B
and give the greedy tokens of `bin/run`?

Every check so far installed from a directory or from a local wheel. This is the first time the bytes go through an
index. The upload to PyPI cannot be taken back, so TestPyPI rehearses it with the same files.

## 2. The objects

The distributions, built by `uv build` from commit `1c64ecb` (`llvq-tetra/` is unchanged since):

| file | bytes | sha256 |
|---|---|---|
| `llvq_tetra-0.1.0-py3-none-any.whl` | 112,934 | `04c25b1dfb4778b171f1d9328adfc2789786d8e8d9e105bd28eb909d26f9240c` |
| `llvq_tetra-0.1.0.tar.gz` | 226,381 | `5be7aee67532202f220215d410c98addb9fd9df2d58deaa415e1ff0819f0351e` |

Both pass `twine check --strict`.

The model: `Pier-Jean/Qwen3-4B-LLVQ-Tetra` from the Hub, unauthenticated, into an empty `HF_HOME`.

The reference: `bin/run` on `~/q4b-sealed-2026-09-23/qwen3-4b-sealed.bin`, sha256 `886391a8c03f66dc...`, the file the
published directory was packed from. f32 on the CPU, 64 new tokens on its four prompts, 256 ids, dumped by
`LLVQ_RUN_DUMP`. It is regenerated today because the dump of stage 1 lived in `/tmp` and is gone.

## 3. Setup

```bash
# the operator, once, with a TestPyPI token: the upload
cd llvq-tetra && uv publish --publish-url https://test.pypi.org/legacy/ dist/llvq_tetra-0.1.0*

# the reference
LLVQ_DTYPE=f32 LLVQ_RUN_DUMP=$S/run-tokens-f32.json \
  cargo run --release -p llvq-llm --bin run -- ~/q4b-sealed-2026-09-23/qwen3-4b-sealed.bin cpu 64

# the environment: dependencies from PyPI, the package alone from TestPyPI
uv venv $V --python 3.12
VIRTUAL_ENV=$V uv pip install torch==2.14.1 transformers==5.18.0
VIRTUAL_ENV=$V uv pip install --no-deps --index-url https://test.pypi.org/simple/ llvq-tetra==0.1.0
VIRTUAL_ENV=$V uv pip check

# the check
env -u HF_TOKEN HF_HOME=$S/hf-empty $V/bin/python ops/hf_cleanroom_check.py \
  llvq-tetra/tests/fixtures/mini Pier-Jean/Qwen3-4B-LLVQ-Tetra $S/run-tokens-f32.json
```

`--no-deps` on TestPyPI is deliberate: anyone can upload a project named `torch` there, so nothing but our package
comes from it.

## 4. Controls

1. The wheel TestPyPI serves back has the sha256 of section 2.
2. `uv pip check` reports nothing: the installed torch and transformers meet the declared floors.
3. In a second empty environment, `uv pip install --dry-run --find-links llvq-tetra/dist llvq-tetra==0.1.0` resolves
   torch and transformers from PyPI on its own. That is what the card's one line, `pip install llvq-tetra`, relies on.
4. `llvq_tetra` is imported from the environment's `site-packages`; the script refuses otherwise.
5. The `mini` fixture loads with no missing, unexpected or mismatched key.
6. The download is unauthenticated: `HF_TOKEN` unset, `HF_HOME` empty.

If one fails, no verdict is published.

## 5. What gets published, and what does not get compared

Published: the two sha256 comparisons, the install log, the key lists, the 256 ids against the reference.

Not compared: any time or speed. The load and generation times are printed and are not a number of this prereg.

## 6. Decision rule

| result | verdict |
|---|---|
| TestPyPI refuses the upload | defect named, fixed, rebuilt; a rehearsal upload after that goes out as `0.1.0.devN`, which never reaches PyPI |
| a control fails | stop, named defect, no verdict |
| controls green, 256 ids of 256 equal | the TestPyPI half passes; the PyPI upload of 0.1.0 goes to the operator |
| controls green, at least one id differs | stop: the served bytes or the reader differ from stage 1; no upload to PyPI |
| otherwise | not settled, operator decision |

Three plausible results, each in one row: a refusal on the metadata version, row 1; a sha256 mismatch, row 2; 256 of
256, row 3.

## 7. Signed prediction

256 ids of 256, and TestPyPI accepts the upload. The arithmetic is the one that passed stage 1 and the cleanroom of
2026-10-02; the five changes of 2026-10-07 touch registration, the import guard and the metadata, not a weight. The
flaw: the wheel declares `Metadata-Version: 2.5`, written by hatchling 1.32.4, and nothing here has yet shown that
TestPyPI takes that version. If it does not, row 1.
