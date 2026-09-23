# Runbook: the `row_norms` arm, end to end

Written 2026-09-22 in a cloud container with no GPU, no Metal and no Hugging Face access. Everything
below that touches a card, the bucket, the Mac or the sealed file has therefore **never been run**.
What has run: the Python and Rust tests, their mutants, and a synthetic toy
([row-norms-synthetique](mesures/row-norms-synthetique-2026-09-22.txt)).

Prereg: [BROUILLON-preregistration-dclm-rownorms-2026-09-22](../proofs/BROUILLON-preregistration-dclm-rownorms-2026-09-22.md).
Cost: **about $4.72** (training ~$4.00, scoring ~$0.72), plus $0 of Mac. Nothing is launched
without the operator's go (hard rule 1).

## What exists and what it does

| piece | where | state |
|---|---|---|
| training mode `row_norms` | `ops/llvqtune/llvqtune/trainables/row_norms.py` | 8 tests, 5 mutants dead |
| `MODE` in the training script | `ops/llvqtune/train.sh` | default `row_scales`, so the 61.11 job replays unchanged |
| norm fold | `llvq-llm/src/bin/rowscale.rs`, kind `row_norms` | 18 tests, 5 mutants dead, clippy clean |
| training job | `ops/jobs/dclm-rownorms.sh` | the control's job, `MODE=row_norms` added |
| scoring job | `ops/jobs/dclm-rownorms-mmlu.sh` | modelled on `tetranu-ft-mmlu.sh` |

## 0. Check the code on the Mac

```bash
git fetch origin claude/qualite-pistes-manquantes-45u143
git checkout claude/qualite-pistes-manquantes-45u143
cd ops/llvqtune && uv run --extra torch --with pytest python -m pytest tests -q   # 47 passed, 0 skipped
cd ../.. && cargo test -p llvq-llm --bin rowscale                                                     # 18 passed
cargo clippy -p llvq-llm --bin rowscale --all-targets                                                 # 0 warnings
```

## 1. Prereg, before anything is billed

1. Read the draft. The prediction (§3) and the decision rule (§4) are the author's; change them
   now or never.
2. `git mv proofs/BROUILLON-preregistration-dclm-rownorms-2026-09-22.md proofs/preregistration-dclm-rownorms-2026-09-22.md`,
   and remove the "DRAFT" paragraph at its top.
3. Commit, then `ots stamp proofs/preregistration-dclm-rownorms-2026-09-22.md`, commit the `.ots`.
4. `shasum -a 256` of the stamped `.md` into the `sha256 <FILL AFTER STAMPING>` line of both job
   scripts. Commit.

## 2. Inputs on the bucket

The export is reused from the control arm. Check it before pricing anything else (rule 9):

```bash
hf buckets ls hf://buckets/Pier-Jean/jobs-artifacts/dclm-export-2026-09-19/
```

Package the trainer as the control did: `train.sh` and the `llvqtune/` package at the archive's
root, because the job runs `cd /tmp/src && PYTHONPATH=/tmp/src bash train.sh`.

```bash
tar czf /tmp/llvqtune-2026-09-22.tgz -C ops/llvqtune train.sh llvqtune
tar tzf /tmp/llvqtune-2026-09-22.tgz | grep -E '^train.sh$|row_norms.py$'   # both lines
hf buckets cp /tmp/llvqtune-2026-09-22.tgz hf://buckets/Pier-Jean/jobs-artifacts/llvqtune-2026-09-22.tgz
```

The `hf buckets cp SRC DST` form was read from `hf buckets cp --help` (huggingface_hub installed
2026-09-22). The command used for the 2026-09-19 upload is not recorded in the repository.

## 3. Training, ~$4.00

```bash
bash ops/jobs/dclm-rownorms.sh          # prints the job id
uv run ops/run.py monitor <job_id> --flavor l40sx1
```

**Stop condition, within the first minutes.** When `probe.jsonl` closes, its `first_loss` must
read **0.35267 within 1 %** (the control's value). Outside: `hf jobs cancel <job_id>`. The
routing is wrong and the arm is void.

The job prints `mode row_norms: free: 0 added parameters` and `wiring accepted: … 216 matrices`;
anything else is a stop too.

Fetch the results into the repository:

```bash
mkdir -p docs/mesures/dclm-rownorms-2026-09-22-brut
for f in sigma.json journal.jsonl probe.jsonl; do
  hf buckets cp hf://buckets/Pier-Jean/jobs-artifacts/dclm-rownorms-2026-09-22/$f docs/mesures/dclm-rownorms-2026-09-22-brut/$f
done
```

## 4. Fold, on the Mac, $0

Base: `~/qwen3-4b-dclm.bin`, sha256 starting `471f3988`, 1,794,564,765 bytes.

```bash
B=~/qwen3-4b-dclm.bin; D=docs/mesures/dclm-rownorms-2026-09-22-brut
shasum -a 256 $B | cut -c1-8                                   # 471f3988

# Control 2: the same export with every value at 1.0 must fold to a byte-identical file
python3 - $D/sigma.json /tmp/ones.json <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); r = d.get("result", d)
for k in ("sigma", "tau"):
    r[k] = {n: [1.0] * len(v) for n, v in r[k].items()}
json.dump(d, open(sys.argv[2], "w"))
PY
cargo run --release -p llvq-llm --bin rowscale -- $B /tmp/ones.bin /tmp/ones.json
cmp $B /tmp/ones.bin && echo IDEMPOTENT

# The fold itself
cargo run --release -p llvq-llm --bin rowscale -- $B ~/qwen3-4b-dclm-rownorms.bin $D/sigma.json
#   expect: 216 scaled, 0 untouched, 36 int4 passed through
#           73 norms scaled of 73 named
stat -f%z ~/qwen3-4b-dclm-rownorms.bin                         # 1794564765 (control 3)
shasum -a 256 ~/qwen3-4b-dclm-rownorms.bin

hf buckets cp ~/qwen3-4b-dclm-rownorms.bin \
  hf://buckets/Pier-Jean/jobs-artifacts/dclm-rownorms-2026-09-22/qwen3-4b-dclm-rownorms.bin
```

## 5. Scoring, ~$0.72

```bash
bash ops/jobs/dclm-rownorms-mmlu.sh
hf buckets cp hf://buckets/Pier-Jean/jobs-artifacts/dclm-rownorms-mmlu-2026-09-22/mmlu-4b-dclm-rownorms-FULL.csv docs/data/mmlu-dumps/
hf buckets cp hf://buckets/Pier-Jean/jobs-artifacts/dclm-rownorms-mmlu-2026-09-22/out.txt $D/out-mmlu.txt
```

The log must show `oracle` MATCH (control 4) and `14042 questions scored out of 14042`.

## 6. Reading, $0

```bash
# the prereg's comparison: against the row_scales control, 61.11
cargo run --release -p llvq-llm --bin mmlupair -- \
  docs/data/mmlu-dumps/mmlu-4b-dclm-ft-FULL.csv docs/data/mmlu-dumps/mmlu-4b-dclm-rownorms-FULL.csv
cargo run --release -p llvq-llm --bin mmlupair -- --no-fpc \
  docs/data/mmlu-dumps/mmlu-4b-dclm-ft-FULL.csv docs/data/mmlu-dumps/mmlu-4b-dclm-rownorms-FULL.csv
# and against the untrained base, 57.95, for the total
cargo run --release -p llvq-llm --bin mmlupair -- --no-fpc \
  docs/data/mmlu-dumps/mmlu-4b-dclm-FULL.csv docs/data/mmlu-dumps/mmlu-4b-dclm-rownorms-FULL.csv
```

Apply the prereg's §4 table as written. Secondary, never a gate (prereg §7):

```bash
LLVQ_DTYPE=f16 cargo run --release -p llvq-llm --features metal --bin ppl -- 4096 12 metal ~/qwen3-4b-dclm-rownorms.bin
```

## 7. What the record owes afterwards

- `docs/mesures/dclm-rownorms-2026-09-22.txt`: the result in the shape of
  `dclm-rowscales-2026-09-20.txt`, both jobs, the four controls, the prediction scored.
- Two rows in `docs/data/jobs.csv`, id, minutes, dollars.
- Any deviation in `proofs/preregistration-dclm-rownorms-2026-09-22-ECARTS.md`.
- `docs/HISTORIQUE.md`, one entry; `docs/ROADMAP-QUALITY.md` row 14 and L28 updated with the
  outcome.
