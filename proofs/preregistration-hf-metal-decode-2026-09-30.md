# Prereg. Stage 2 of the transformers plan: the Tetra decode as a torch op on Metal

Status: written and committed on 2026-09-30, BEFORE the first dispatch.
Operator go given 2026-09-30, on the decode op and not on the fused matvec. Cost: 0 $, Mac.
Plan: `docs/plan-transformers.md`. Stage 1: `proofs/preregistration-hf-quantizer-2026-09-30.md`,
journal `docs/mesures/hf-quantizer-4b-2026-09-30.txt`.

A timestamped prereg is no longer edited. Any departure goes in
`proofs/preregistration-hf-metal-decode-2026-09-30-ECARTS.md`, beside it and never into it.

## 1. The plan's gate for this stage is unachievable, and why

The plan asks for an op that "matches stage 1's dequant bit for bit". Stage 1's dequantization is
an **f64** chain, by the format's own design: reconstruct in f64, restore the tail, un-rotate in
f64, narrow once. Metal has no f64. The repository already says so where it met the same wall:
`llvq-llm/kernels/llvq_rot.metal` computes `1/sqrt(m)` "in f64 and narrowed once on the host".

So no correct Metal implementation can meet that gate. This is a property of the hardware and not
a defect, which is why the plan's own line, "that is a defect to fix, not a tolerance to widen",
does not apply and the gate is restated below rather than widened. The operator took that decision
on 2026-09-30.

What is left exactly gateable is the part of the chain that is **integer**: the decode from a
47-bit label to 24 lattice coordinates. That is this stage.

## 2. What is not attempted, and what blocks it

The fused matvec, our served kernel, is not ported here. Two reasons, both facts:

- Its reference cannot be stage 1. A matvec in f32 is gated against a host reference that
  reproduces the kernel's own summation order, which is what `llvq-metal`'s
  `tetra48_matvec_matches_host.rs` already does in Rust.
- The Metal fused path refuses all three sealed files at load. `MetalRuntime::upload_int4` stages
  `d_in · 4` bytes against a 32 KB threadgroup limit, and every sealed file carries int4
  `down_proj` at `d_in` 9,728 or more. The fix, tiling the staging by slices of 256 columns so
  each lane's accumulation order is preserved, is written down and not done.

Both belong to a stage 2 bis, with its own prereg and its own go.

## 3. The object and the setup

Input: `~/q4b-hf-2026-09-28/`, packed at commit `0c4197f`, `model.safetensors` sha256
`a28348cafa82d23d...`. 168 Tetra records, 118,665,216 blocks in total.

References, both already pinned to `llvq_search::tetra`:

- the MSL decode, by `llvq-metal/tests/tetra48_matches_rust.rs`, which demands equality and not a
  tolerance;
- the numpy decode, by control 2 of stage 1: 100,000 labels, 0 differences.

So the two sides of this stage's gate are each pinned to the same Rust reference, and the gate
closes the triangle.

```bash
# the shader and its tables, shipped with the package from one source
cargo run --release -p llvq-llm --bin tetratables -- llvq-hf/llvqhf/data/tetra-tables.safetensors

# the op, built once, then the gate over every block of the 4B
cd llvq-hf && uv run --group dev pytest
cd llvq-hf && uv run --group dev python -m llvqhf.checkdecode ~/q4b-hf-2026-09-28
```

## 4. Three decisions taken before the code

| decision | what is done | why not the other |
|---|---|---|
| one MSL source | the decode entry point is added to `llvq-llm/kernels/llvq_tetra48.metal`, the served shader, and the package ships a **copy** whose sha256 it checks | a second MSL decode in the package would be a second implementation to keep bit-exact, which is the argument that put the tables in the package rather than in every model |
| the shader's table layout | `bin/tetratables` also writes the shader-shaped tables, `branches` packed as 1,024 u16 beside the two u8 arrays numpy reads | letting Python repack them would put the layout decision in two places, and the repack is where a mistake would land |
| the op's boundary | `torch.ops.llvq.tetra_decode(codes, tables…) -> int8[n, 24]`, points only. No gain, no scale, no rotation | the rest of the chain is f64 and stays on the host. An op that returned weights would have to be judged against a reference it cannot equal |

## 5. The gate

**Every block of the 4B, exactly.** For each of the 168 Tetra records, the op's points on MPS equal
the numpy decode's, element for element, over 118,665,216 blocks and 2,847,965,184 coordinates.
Integer values: there is no tolerance to widen and none is defined.

The op is also checked on the 100,000 sampled labels of stage 1's control 2 and on the origin word.

## 6. Controls

1. The shader the package ships is the repository's, by sha256, checked in the fast loop.
2. The op refuses a label above 47 bits and a code stream whose length is not `6 · nblocks`.
3. One mutant at least on the gate: a permuted table, a section swapped in the MSL entry point.
   Each must be caught and named in the journal.
4. `cargo clippy --all-targets` silent, `cargo test` green in the fast loop, `pytest` green.
5. Stage 1's two gates still pass on the same directory, unchanged by this lot.

## 7. Signed predictions

**The gate passes exactly, on all 118,665,216 blocks.** Both sides are pinned to
`llvq_search::tetra`, so the only new thing between them is the table re-layout of decision §4, and
that is where a failure would land. The prediction is exactness.

**The MPS decode of the whole model takes under 60 s**, and the numpy comparison dominates the wall
time rather than the dispatch.

**The extension compiles in under 3 minutes** on this Mac, once.

**No memory and no speed number comes out of this stage.** The op returns points; the dequantization
still runs in f64 on the CPU, so the loaded model stays dense and nothing about throughput is
measurable here.

I have been wrong on signed predictions four times in this repository, the last on 2026-09-28 by a
factor of two. This one is scored the same way.

## 8. What this stage cannot establish

- Nothing about the fused matvec, which is stage 2 bis, nor about the int4 staging wall.
- Nothing about tokens: the op does not touch the forward pass. Stage 1's 256 ids stand and are
  re-checked as control 5, not re-derived.
- Nothing about speed or memory, here or at any stage of this plan.
- Nothing about CUDA, which is stage 4, where the wall and the reference are different.
- Nothing about the Kernel Hub packaging, which is stage 3 and needs this op to exist first.
