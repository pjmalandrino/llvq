# Preregistration. M7: where a Metal decode token goes, by op class

Written on 2026-11-06 in the morning on branch `claude/metal-nuit-3c4257`, once the code it describes existed. To be
timestamped before its first measurement, then never edited; departures go to
`preregistration-metal-nuit-2026-11-06-ECARTS.md` under an M7 heading. Cost **$0**, local, under the operator's go of
2026-11-06 ("ok fait le je veux qu'on bombarde sur METAL aussi").

## 1. Why

The sealed 4B decodes at 52 tok/s on the M3 Max (`lutd`, M2), reading about 72 GB/s of weights (*computed*). The
`blocks` phase holds 18.0 ms of a 19.2 ms token (phased, off protocol). Nobody knows how those 18 ms split between the
Tetra matvec, the int4 matvec, the rotations, the norms and the attention. Every Metal kernel lever after this one is
chosen on that split, so it is measured first.

## 2. What runs

`metalrun profile` (doc comment of `llvq-llm/src/bin/metalrun.rs`). Binary sha256
`9907a0736969e260975ce89b00766367a28bb6466456742e449dafb62512c8a7`, built from the commit that carries this file.
Quiet machine (no build from 15 minutes before the first run to the last), nice 10.

1. P1: `metalrun profile <sealed> configs/qwen3-4b-tetra-e4.json lutd 11`
2. P2: `metalrun profile <sealed> configs/qwen3-4b-tetra-e4.json lut 11` (the served Metal kernels, for the Tetra
   kernel's own GB/s)

Each run first checks that `generate_phased` returns the same 64 tokens as `generate`, then prints the `blocks`
phase, the full op sequence minus the empty one, the remainder, and per class (norm, rot, tetra.q, tetra.k,
tetra.gate, tetra.up, tetra.down, int4.v, int4.o, int4.down, act, resid) the marginal cost, the cost alone, its share
of `blocks` and, for the matvec classes, GB/s alone from bytes computed from the shapes.

## 3. Gates

1. Same 64 tokens from `generate_phased` and `generate` (the binary stops otherwise).
2. The full sequence is not longer than the `blocks` phase (median against median); otherwise the split is
   journaled as unusable and the next step is chosen on the alone costs only.
3. The Tetra and int4 bytes computed from the shapes sum to within 2 % of the loader's 1.16 GB of projections.

## 4. Rules

- The class with the largest marginal cost is the next kernel attacked, bit for bit with the served kernel.
- A matvec class reading under 50 % of the machine's nominal 400 GB/s alone is called *kernel-bound*, else
  *bandwidth-bound*. A bandwidth-bound class is not attacked by kernel work.
- If the remainder (attention, RoPE, q/k norms, cache, reshapes) is the largest share, the next lever is the op count
  of that remainder, not a matvec.
- Every number here is off protocol (closed by a synchronize). It chooses work; it is never quoted as a rate.

## 5. Predictions, signed

*Estimated*.

| quantity | prediction | why |
|---|---|---|
| Tetra classes, marginal, share of `blocks` | 55 to 75 % | 749 MB at the kernel's ~22 % of bandwidth |
| int4 classes, marginal, share | 15 to 30 % | 410 MB, a tiled kernel near its own roofline |
| rot + norm + act + resid | 3 to 10 % | small vectors, about 250 launches |
| remainder | 5 to 15 % | attention at a 6-token cache, RoPE, reshapes |
| Tetra alone, GB/s | 60 to 110 | kernel-bound |
| int4 alone, GB/s | 150 to 300 | |
