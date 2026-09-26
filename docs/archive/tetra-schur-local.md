# Tetra Schur local diagnostic

The diagnostic compares gain choices and their GPTQ continuations on retained projection rows.
It does not estimate MMLU or change the served encoder's default.

## What is implemented

`tetra_schur` separates local token preparation, capture and replay.
The code is in `llvq-quant/src/schur.rs` and `llvq-llm/src/tetra_diag.rs`.

| arm | choice at the common state |
|---|---|
| A | nearest gain to the compensated source norm |
| B | nearest gain to the source projection onto the chosen unit direction |
| C | lower conditional quadratic cost among the same two reconstructions |

Each shadow block calls the direction encoder once. Only A advances the witness.
At selected blocks, both gains start from independent copies of the same state.
Both continuations then use A until the row ends. Later directions may differ.
The row scale and fitted centroids remain fixed. The partial tail keeps its compensation.
Exact ties select gain zero. Floating-point near-ties are checked with relative tolerances in the tests.

The scorer solves `v U_BB = x - q` and returns `sum(v²)`.
The global continuous lower bound adds the accumulated fixed-prefix cost.
`rollout_excess` includes discrete constraints and the continuation algorithm's search error.
It is not an exact discrete gap, except on the exhaustively solved test toys.

Capture uses the original checkpoint in f32 through every Transformer layer.
This choice isolates the row optimizer and avoids full-model re-encoding.
It does not reproduce the activation distribution of a sequentially quantized model.
Validation activations come from a separate corpus split, through that same checkpoint.
The metric uses the requested natural-basis shrinkage, input rotation and damping.
Validation output error uses the raw reserved activations, without shrinkage or damping.

`tetrapost` with group scales, Design C or resume is refused before model loading in `smoke`.
The library also rejects these modes. Tetra's post-shape reprojection is explicitly refused.
The codebook fingerprint, tables, disk format and kernels are unchanged.

## Prepare without inference

Run from the repository root. All inputs must already exist locally.
The preparation script pins Qwen3-0.6B and the cached Wikitext train/validation snapshot.
It writes exact token IDs, non-overlapping offsets, corpus fingerprints and a SHA-256 preparation manifest.
It samples complete token chunks from a fixed text prefix using each recorded seed.

```bash
nice -n 10 cargo build --release -p llvq-llm --features metal,fast-linalg --bin tetra_schur
python3 ops/prepare_tetra_schur.py "$HOME/tetra-schur-pilot-2026-09-14"
```

Preparation creates `seed-1/plan.json` and `seed-2/plan.json` with inspection reports.
The script never launches capture, replay, a download or a model evaluation.
The completed pilot follows [the timestamped preregistration](../proofs/preregistration-tetra-schur-2026-09-14.md).
The original output directory already exists. Further runs require fresh paths and operator approval.

## Capture after operator approval

The oracle runs automatically on the requested backend before activation collection.
Capture refuses an existing output directory. Partial output has no completion marker.

```bash
pilot_dir="$HOME/tetra-schur-pilot-2026-09-14"
/usr/bin/time -l nice -n 10 target/release/tetra_schur capture \
  "$pilot_dir/seed-1/plan.json" "$pilot_dir/seed-1/capture" \
  > "$pilot_dir/seed-1/capture.log" 2>&1
```

Repeat for seed 2 under the approved total budget.
Keep the raw log: `/usr/bin/time -l` records maximum resident memory on macOS.
Capture records checkpoint and executable fingerprints, source copies, git commit and the tracked diff.
The FNV-1a fingerprints detect accidental drift; they are not cryptographic attestations.
The preregistration's timestamp and SHA-256 manifest provide the separate immutable record.

## Replay retained inputs

Each `layer-*.json` names binary f64 arrays for the metric, reserved activations and original rows.
Its metric is stored before damping; replay applies damping exactly once.
It also records matrix-fitted gains and the selected row and block indices.
The arrays stay in the capture directory. Replay refuses corruption and existing result directories.

```bash
capture_dir="$pilot_dir/seed-1/capture"
for bundle in "$capture_dir"/layer-*.json; do
  stem="$(basename "$bundle" .json)"
  /usr/bin/time -l nice -n 10 target/release/tetra_schur replay \
    "$bundle" "$pilot_dir/seed-1/$stem" \
    > "$pilot_dir/seed-1/$stem.log" 2>&1 || break
done
```

Replay runs on the CPU, using `fast-linalg` for factorization when compiled above.
It stores the witness, every selected compensated state, both complete reconstructions and their codes.
The per-row file includes both local scores, prefix cost, continuation losses and reserved-output losses.
Each projection summary reports disagreements and average selection regret for A, B and C.
Times are measured per row and factorization. Array storage and encoder-call budgets are computed.
No peak-memory figure is inferred from the array count.

Compare projection summaries within each depth and seed. Blocks are correlated observations.
The evenly spaced rows are a fixed diagnostic sample, not a random sample of all model weights.
No aggregate confidence interval or population claim is supported by this pilot.

## Software verification

110 tests passed, plus nine repeats under `fast-linalg` (*measured*, [verification journal](mesures/tetra-schur-code-2026-09-14.txt)).
Four intentional mutants were detected. Metal compilation and Clippy passed.
The pretrained-model pilot is complete; see [results and limitations](tetra-schur-pilot-2026-09-14.md).

```bash
cargo test -p llvq-quant
cargo test -p llvq-quant --features fast-linalg --test schur
cargo test -p llvq-llm --test tetra_diag --test tetra_wiring --bin smoke
cargo check -p llvq-llm --features metal,fast-linalg --all-targets
cargo clippy --all-targets -- -D warnings
git diff --check
```

Tests cover independent elimination, a nonempty prefix, damping, exhaustive tiny suffixes and an exact tail.
They also cover anisotropic gain-ranking reversal, legacy-code equivalence and the experimental artifact round trip.
The end-to-end fixture builds a tiny checkpoint locally and runs the oracle, capture and deterministic replay.
Its outputs are correctness fixtures, not measurements of a pretrained model's quality.

## Audit retained measurements

```bash
uv run --offline --with numpy python ops/analyze_tetra_schur.py \
  "$HOME/tetra-schur-pilot-2026-09-14" /tmp/tetra-schur-audit.json
```

The output path must be new. This checks retained arrays against independent dense algebra and recomputes the summaries.
It performs no model inference or new candidate selection.
