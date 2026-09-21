# Preregistration. What the device port cost on a card

**Written, committed and TIMESTAMPED on 2026-09-21, BEFORE the run.**
Operator go given 2026-09-21. Cost: about **$0.30** on l40sx1, timeout 1 h.

## 1. Why this run exists

Between 2026-09-20 and 2026-09-21 the served dispatch was rewritten. `model.rs` held
`Arc<FusedRuntime>` beside `Arc<FusedProj>`, both CUDA types, behind 40
`cfg(all(target_os = "linux", feature = "cuda"))` sites. It now holds `Arc<dyn LatticeProj>`
and names no backend; the adapters live in `fused_cuda.rs` and the port in `device.rs`
(`85a7ec9`, `01c5c66`, `e1d2c9e`).

Every method body forwards to the inherent method that already shipped, so the claim is that
the arithmetic and the call order are untouched and only the DISPATCH changed. That claim has
been reviewed adversarially (16 findings, none a behaviour change) and is exercised by 15 CPU
tests with 7 mutants. None of that runs on a card.

The cost of dynamic dispatch is *computed* at about 0.025 % of a decode step and has never
been *measured*. Until it is, the next Tetra throughput number is not comparable to the last
one.

## 2. The confound, and how it is controlled

The reference is `docs/mesures/dclm-ft-fusedrun-2026-09-20.txt`: **98.3 tok/s [97.5, 98.6] in
1.39 GB**, job `6aaf855552d0dbd7f1d72e3b`, l40sx1. That run did not pin `LLVQ_TILE_BLOCKS` and
its journal records **tile 128**.

Since 2026-09-20 the served default reads `TILE_BY_SM`, which is **64** on sm_89. Re-running
the reference script verbatim would therefore move for two reasons at once: the port, which is
what this run is for, and the tile, which is already measured and worth +16.1 %.

So the job runs two arms in one process:

| arm | tile | what it answers |
|---|---|---|
| A | `LLVQ_TILE_BLOCKS=128`, pinned | the port's cost, comparable trait for trait to 98.3 |
| B | unset, so 64 | the served number to publish after this lot |

Everything else is held: the same image, the same file
(`/out/dclm-ft-2026-09-19/qwen3-4b-dclm-ft.bin`, sha256 `f8c1c903b753fe34...`), the same
flags `LLVQ_FUSED_LAYOUT=tetra48 LLVQ_EMBED=q8 LLVQ_ROT_SHARE=1 LLVQ_FUSE=0 LLVQ_KV=f16`.

## 3. Signed prediction

**Arm A lands in [97.0, 98.8] tok/s**, so no measurable effect.

The reasoning, stated so it can be scored rather than admired: the port adds one indirect call
per matvec, per rotation and per embedding lookup. The served 4B issues 252 matvecs and 144
rotations a token, call it about 500 indirect calls. At 2 to 5 ns each that is 1.0 to 2.5 us
against a decode step of 10.17 ms at 98.3 tok/s, i.e. **0.010 % to 0.025 %**. The reference's
own round-to-round range is [97.5, 98.6], which is 1.1 %, forty times wider.

**Arm B lands in [110, 120] tok/s.** Tile 64 is worth +16.1 % on the Tetra arm's kernel time
(`tuile-l40s-2026-09-20`), and 67 % of that arm is launch overhead the tile does not touch, so
the end-to-end gain must be smaller than 16.1 %.

**VRAM stays at 1.39 GB on both arms**, to three digits. The port moved no allocation.

I have been wrong on signed predictions three times in this repository, once by a factor of
ten, and each is recorded. This one is scored the same way.

## 4. The gate, which is not the throughput

**256 tokens identical to the dense arm, in the same process, on both arms.** A refactor of the
served dispatch that changes one token is dead whatever it measures. `oracle` runs first, per
hard rule 10.

Throughput is secondary and is reported with its round-to-round range, formed round by round,
never as a quotient of two minima.

## 5. What would refute the claim

- Arm A outside [97.0, 98.8]: the dispatch costs something, and the computed figure is wrong.
- Any token differing from the dense arm: the port changed the arithmetic. The lot is reverted,
  not patched.
- VRAM away from 1.39 GB: an allocation moved, which nothing in the diff should do.
- Arm B below arm A: the tile row for sm_89 does not transport to this object.

## 6. What this run cannot establish

Nothing about Metal. Nothing about the 8B or any other model. Nothing about the segmented
kernel, which still fails its bit-exact comparison and stays behind `LLVQ_SEG_TETRA`, off.
Nothing about quality: this is a throughput and identity run, and MMLU is not scored here.

## 7. Deviations

Any departure from this file goes in
`proofs/preregistration-port-device-2026-09-21-ECARTS.md`, beside it and never into it.
