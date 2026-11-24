# Audit of llvq-tetra and its kernels, 2026-11-21

Five defects are fixed in the 0.2.0 line, each with a test that fails before the fix.
Four more are fixed in the unmerged CUDA bindings, and none of those was compiled.
Nothing is committed, the version is unchanged, and no shipped shader or served kernel changed by a byte.

Scope: A, the package on `main` (worktree `audit-tetra-kernel`); B, commit `2cbec01` (worktree `audit-tetra-cuda`), read statically; C, the fork `transformers-llvq`, read only.

## Test summary

Before the first edit: `uv run --group dev pytest`, 49 passed, 0 failed, 0 skipped, 43.8 s.
After the last edit: 72 passed, 0 failed, 0 skipped, 27.7 s.
The 23 new tests are in `tests/test_refusals.py` (17), `tests/test_cuda_sources.py` (2), `tests/test_load.py` (2), `tests/test_mps.py` (1) and `tests/test_quantizer.py` (1).

Real model, one heavy run under the local load-guard wrapper: `python -m llvq_tetra.checkmps` on `<staging-dir>/qwen3-4b-tetra`, after every fix.
It loaded the 4B through `from_pretrained` with the new checkpoint check active and printed "all gates green": G-A 252 of 252 bit-identical, G-D 252 of 252, G-B1 151,936 of 151,936 rows, G-B2 worst row 2.48e-06, G-C worst row 1.53e-07 (*measured*, log in the session scratchpad).

## Part A, fixed

### A1. Defect: a broken checkpoint loads instead of being refused

`llvq_tetra/integration.py:67` (`replace_modules`) and `:226` (`finalize`), before the fix.
A packed directory with one quantized tensor missing loaded without an error.
`transformers` fills a missing key with `torch.empty_like` (`modeling_utils.py`, `_move_missing_keys_from_meta_to_device`), that is uninitialised memory, and only lists it in `missing_keys`.
With a quantizer present it also skips its own shape check (`core_model_loading.py`, the test `hf_quantizer is None`), so a tensor of the wrong length is assigned as it comes.
Measured on the 148 KB fixture, before the fix:

| tensor changed | device | outcome |
|---|---|---|
| `q_proj.row_scales` dropped | cpu, mps | loaded, logits finite |
| `q_proj.codes` dropped | mps | loaded, logits finite |
| `q_proj.row_scales` cut by 8 | mps | loaded, `missing_keys` and `mismatched_keys` both empty |
| `down_proj.scales` cut by 8 | mps | loaded, both lists empty |
| `q_proj.tail` cut by 8 | mps | loaded, logits not finite |
| a rotation's `signs` dropped | cpu | loaded, logits not finite |

On Metal the cut-short cases have the kernel read past the end of a buffer, silently.
Fix: `replace_modules` records the shape of every buffer the LLVQ modules declare (`declared_tensors`).
With the `checkpoint_files` that `transformers` passes to `_process_model_before_weight_loading`, `require_checkpoint` (`integration.py:129`) reads the safetensors headers and refuses a missing or wrong-shaped tensor before any weight is read.
`finalize` calls `require_shapes` (`integration.py:153`), which catches a wrong shape when no files were handed over.
`replace_modules` takes `checkpoint_files` as an optional last argument, so the fork's call still works.
Test: `tests/test_refusals.py`, eight broken checkpoints on cpu and on mps, plus the intact fixture.
Before: 16 failed, 1 passed.
After: 17 passed.
Mutants: without the header check, 10 fail (the dropped cases; the cut-short ones are still caught by `require_shapes`); without both checks, 16 fail.
The real 4B still loads (`checkmps` above), so the check does not refuse a whole file.

### A2. Defect: `torch.inference_mode()` raises on Metal

`llvq_tetra/mps.py:421` and `:436`.
The shared rotation kept its result keyed on `x._version`, and an inference tensor has no version counter.
Every forward under `torch.inference_mode()` raised "Inference tensors do not track version counter" at the first projection.
Fix: an inference tensor is rotated again instead of kept, because nothing proves it unchanged.
The cost is two extra rotation dispatches per layer group under `inference_mode` only; `no_grad` is unchanged.
Test: `test_inference_mode_gives_the_no_grad_logits` in `tests/test_mps.py`, which requires equal logits.
Before: fails with the RuntimeError.
After: passes, logits `torch.equal`.

### A3. Defect: `LLVQ_HF_TILE` sizes the CUDA kernel and not its shared memory

`llvq_tetra/metal.py:129` (before) compiled `-DTILE_BLOCKS=$LLVQ_HF_TILE`.
`integration.py` always passed `TILE_BLOCKS = 64` as the tile, and `csrc/tetra_cuda.cu` sizes shared memory from that argument.
With `LLVQ_HF_TILE=128` the kernel stages 128 blocks into shared memory sized for 64.
Fix: `metal.cuda_tile()` (`metal.py:166`) reads the variable once, cached, and refuses anything but a power of two in 32..=512.
The build and `integration.tile_for("cuda")` (`integration.py:250`) both use it.
`tetra_cuda.cu` now refuses a `tile` that differs from its compiled `TILE_BLOCKS` (three lines, not compiled tonight).
Test: `test_the_cuda_tile_is_the_one_the_kernel_is_compiled_with` in `tests/test_quantizer.py`.
Mutant: `tile_for` returning 64 fails it.

### A4. Defect: the CUDA source copies are never checked by digest

`llvq_tetra/metal.py:122` (before) checked that the six CUDA files exist.
The Metal shaders are checked by sha256 before every compile; the CUDA copies were not, though `tetra-tables.json` records their digests.
A drifted copy compiled and ran.
Fix: `require_cuda_sources()` (`metal.py:90`) checks existence and sha256, and the CUDA build calls it.
Test: `tests/test_cuda_sources.py`, the shipped copies pass and a copy with one appended line is refused by name.
Mutant: skipping the comparison fails the second test.

### A5. Gate gap: the `from_pretrained` route was never held to the Rust digests

`tests/test_dequant.py` holds `PackedModel.dequantize` to the digests, on `tiny`.
The module path (`modules.py`, `TetraLinear.materialize`) reads the same bytes through the `transformers` loader, which casts floating tensors to the dtype of the buffer it fills.
Nothing checked that path bit for bit, though `fixtures/mini/llvq-dense-digest.json` is in the repository.
Added: `test_the_loaded_weights_are_the_rust_decoders_bit_for_bit` in `tests/test_load.py`, at f32 (eight SHA-256, seven records and the embedding) and at f16 (the f32 values narrowed once).
It passes on the code as it was, so this closes a gate and fixes no defect.
Mutant: narrowing `row_scales` to f32 inside `materialize` fails the f32 case.

## Part A, proposed and not applied

| id | severity | where | what happens | proposal, gain, risk, gate |
|---|---|---|---|---|
| P1 | defect | `integration.py:31-64`, `:246` | `device_map="cuda:1"` reduces to `"cuda"`: the buffers and `model.to` go to the current device, not to card 1 | carry the full device string from `target_device`; gate: a two-card load test; not testable on this Mac |
| P2 | waste | `fused.py:207` | on CUDA, one `tv_tetra48` dispatch and one f16-to-f32 cast per token, then `torch.stack`: a prompt of T tokens costs T dispatches per projection | bind `tv_tetra48_rows_h`, the served four-row prefill kernel; gain on time to first token only, unmeasured; gate: tokens equal row by row, as G-D does on Metal |
| P3 | waste | `mps.py:346` | `vocab_logits` dispatches once per row; `generate` keeps one row, but a perplexity run over T positions pays T dispatches | a rows variant of `tv_emb_q4`; gate G-B2 |
| P4 | hygiene | `fused.py:217`, `:298` | the non-mps, non-cuda branches call `torch.ops.llvq.*` (the Objective-C++ ops); a model moved to the CPU after loading fails inside an operator | refuse by name when the device is neither mps nor cuda |
| P5 | defect, docs | `README.md:37-46` | says `LLVQ_HF_FUSED=1` is the switch and that the fused path needs ninja and Xcode tools on a Mac; in 0.2.0 `device_map` is the switch and Metal compiles through `torch.mps.compile_shader` with no compiler; the 2.75 GB figure predates the 4-bit vocabulary | rewrite the two paragraphs from the code; the README is a living document, so not edited here |
| P6 | hygiene | `pyproject.toml:57-59` | the comment ties ninja to `LLVQ_HF_FUSED=1`; ninja is needed only for CUDA and for the 0.1.0 gates (`checkdecode`, G-A) | reword the comment |
| P7 | hygiene | `integration.py:299` | on CUDA the 84 int4 layers go dense with a `warnings.warn`, which is shown once and filtered by many harnesses | log the count at load as well; no behaviour change |
| P8 | gate gap | `checkdecode.py` | it gates the 0.1.0 Objective-C++ decode (`tetra48_probe`), not the `compile_shader` path 0.2.0 serves; that path is gated by `checkmps` G-A against the 0.1.0 dispatch | name the chain in the README of the gates, or drop `checkdecode` from the list of 0.2.0 gates |

Waste looked for and not found on the fixture: a decode step costs 0.59 ms per forward on the mini fixture, and the LLVQ modules take 23 ms of 148 ms of Python time over 200 forwards (*measured*, cProfile).
The libraries, tables and `invnorm` are built once and cached (`lru_cache`), and there is no `.cpu()`, `.item()` or numpy hop in the Metal forward.
The transcode at load (`split_stream` plus `to_tetra48`) costs 17 ms and peaks at 78 MB for the largest 4B matrix, 9,728 by 106 blocks (*measured*), so it is not worth touching.

## Part B, the CUDA bindings of `2cbec01` (static, nothing compiled)

Argument order, grid, block, shared bytes, dtypes, stream and launch checks match the served callers in `llvq-llm/src/fused_cuda.rs` (`launch_tetra48_h` :2275, `launch_q4_h` :2173, the tied head :1301-1319, `launch_rot_rows` via `llvq-cuda/src/gpu.rs:832`).
The shipped include closure is complete, ten files, byte-identical to the repository's.
Fixed in `audit-tetra-cuda/llvq-tetra/llvq_tetra/csrc/tetra_cuda.cu` (+33/-1):

* B1, defect: no `CUDAGuard` in any of the four wrappers (:104, :158, :201, :249); a tensor on another card than the current one is launched on the wrong device; added a guard before allocation, and a same-device check on every input.
* B2, defect: `tile` is not checked against the compiled `TILE_BLOCKS` (:133), the same hole as A3; added the refusal.
* B3, defect: the Tetra `tail` was checked for dtype only (:121); added contiguity, device and `numel >= d_out·tail_w`.
* B4, hygiene: the rotation's row offset is u32 in the kernel; added `n_rows·n <= UINT32_MAX`.

Proposed there: `pose_shared` calls `cudaFuncSetAttribute` on every launch where Rust sets it once at load (`fused_cuda.rs:641`, `:772`); a per-device static flag would remove it.
Proposed there: the Tetra wrapper checks no table sizes and no `invnorm` length, unlike the int4 ones.
None of part B is proven to compile until `nvcc` runs on a card.

## Part C, the fork `transformers-llvq` (read only, nothing edited)

The fork shares A1: `quantizer_llvq.py:92-105` delegates to `replace_modules` and `finalize` and checks nothing.
Patch to propose: take `checkpoint_files` in `_process_model_before_weight_loading` and pass it as the fourth argument of `replace_modules`, which now does the check.
Its tiny test asserts `mismatched_keys` is empty (`test_llvq.py:83-84`), which can never fail while a quantizer is present; a negative test with one tensor dropped belongs beside it.
The tiny test downloads `<owner>/tiny-qwen3-llvq` on every run and is not `@slow` (`test_llvq.py:74`), against `ISSUE.md:76`, which promises no download.
`LlvqConfig` defaults `block_dim=24` and `code_order="msb_first_dense"`, where the package refuses a config without `code_order`.
The install error does not name the floor 0.2.0, so a user with 0.1.0 reads "requires llvq-tetra".
`LLVQ_HF_FUSED` and `LLVQ_HF_DEVICE` reach the in-tree loader through `target_device(None)`, undocumented.
The uncommitted docs claim "1.39 GB on an Apple GPU" (`llvq.md:61`) with no journal behind it.
The Metal test does not count the 84 `Int4FusedLinear`.

## Not run

* Anything on CUDA: the A3 and A4 Python paths are tested, the three C++ lines in `main`'s `tetra_cuda.cu` and all of part B are not compiled.
* `checkdense`, `checkdecode`, `checkq4`, `checkq4guards` on the 4B: no fix touches the dense decode, the Objective-C++ op or the int4 kernels, and `checkmps` covers the load path the fixes changed.
* The CPU load of the 4B (16 GB at f32); the header check is the same code on both devices, and the Metal load declares more tensors (the 4-bit vocabulary).
* The fork's own test suite.

## Carried onto `main`, 2026-11-24

Part A and the part of B that applies were carried onto `b0e19d2`, which already held the code audit of 2026-11-18.
That audit had fixed A3 its own way, by refusing `LLVQ_HF_TILE` above 64.
The merge keeps both: `cuda_tile()` feeds the build and the caller, and accepts a power of two from 32 to 64 only, so no value refused on `main` became accepted.
`main` ships only the `tv_tetra48` binding, so B1 and B3 apply to that wrapper alone, and B2 is the A3 check, kept once.
The `tv_q4`, `tv_emb_q4` and `rot_apply_rows` guards of B, and B4, have no wrapper on `main` to land in and were left out.
