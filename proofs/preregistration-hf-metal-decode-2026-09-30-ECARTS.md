# Deviations from the stage 2 prereg of 2026-09-30

The prereg is `proofs/preregistration-hf-metal-decode-2026-09-30.md`, sha256 `9d89e8bb971c3ec2`,
timestamped before the first dispatch and never edited. The journal is
`docs/mesures/hf-metal-decode-4b-2026-09-30.txt`.

Three departures. The first two were found while writing the code, the third after the numbers.

## 1. No MSL was written: the entry point already existed

§4, first decision, says "the decode entry point is added to `llvq-llm/kernels/llvq_tetra48.metal`".
Nothing was added. `tetra48_probe` is already in that file, with its own comment saying why: "one
thread a block, no tile, no reduction: it exists so the decoder can be judged on its own, before
any matvec is written". It is also what `llvq-metal/tests/tetra48_matches_rust.rs` already compares
against `llvq_search::tetra`.

So the served shader is untouched by this stage, which is better than what the prereg planned: the
op is a binding with no arithmetic of its own, and the gate's two sides were already pinned to the
same Rust reference before the stage began.

## 2. The op's signature carries the stream, not the words

§4, third decision, writes the boundary as `tetra_decode(codes, tables…) -> int8[n, 24]`. The
`torch.ops.llvq.tetra_decode` schema takes the **transcoded** words, the four table blobs, `d_out`,
`nblocks`, `row_stride_u32` and the shader source, and returns `(float32[n, 24], int32[n])`: the
shader writes floats and its shell index, and the C++ side adds nothing.

The Python-facing function is the boundary §4 describes, `codes` in and `int8[n, 24]` out. The cast
is exact, the shader writes integers and the codebook's largest coordinate is 10. The transcode
from the disk stream to the served layout happens in `llvqhf.metal.to_tetra48`, on the host, and it
is where the one real defect of this stage was: a byte reversal instead of moving the gain bit from
bit 0 of the disk value to bit 47 of the word. Twelve blocks of twelve wrong, caught on the first
dispatch.

## 3. One signed prediction missed

§7 predicts that "the numpy comparison dominates the wall time rather than the dispatch". The
comparison costs nothing, 0.00 s on the largest record. What dominates is the numpy **decode**,
0.31 s against the op's 0.10 s end to end. The sentence named the wrong one of the two numpy costs.

The other four predictions of §7 held, including the exactness of the gate and the 60 s bound, at
38.4 s for both sides over 118,665,216 blocks.
