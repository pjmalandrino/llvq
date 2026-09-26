# Preregistration — Spherical GPTQ feedback on Tetra: does it move perplexity? Qwen3-0.6B

**Written and TIMESTAMPED (`ots stamp`) on 2026-09-22, BEFORE the first run.** Operator go,
2026-09-22, verbatim: "go", on the plan of the same day (implementation, first tests, then the
0.6B in 28 blocks, one variable). The commit that carries this file follows on the operator's go.

🚨 **This file is not edited again**: the stamp attests these bytes at this date. A fact it gets
wrong is written *beside* it, in a `-ECARTS.md`.

**What is measured**: HEAD `5d36d52` plus the uncommitted working-tree patch whose
`git diff HEAD -- '*.rs' CLAUDE.md | sha256` starts `36759b409a772c54` (16 files, +190/−16):
`GptqConfig::spherical_feedback` and the `sph` mode of `smoke`. Nine tests in
`llvq-quant/tests/g5_spherical.rs`, one dead-flag mutant killed by three of them.

**Cost: $0, about 31 min of Mac now** (2 runs, *estimated* from the 31 min the two-arm tetrapost
pair took on this protocol, `docs/mesures/tetrapost-ppl-0.6b-2026-09-15.txt`), **plus at most
1 h 05 more** if §4 says continue. No other Metal job runs beside it. Project total unchanged:
at least $176.80 (*computed*, `proofs/preregistration-dclm-8b-2026-09-21.md`).

---

## 1. The question

Every file this repository has produced is the paper's "Euclidean GPTQ" row: the block is
snapped to a gain level and the residual the Hessian correction propagates is formed against the
snapped block, so the radial error of a two-level code is chased by the correction. The paper's
loop (Algorithm 3 line 5) keeps every block on its exact norm during the loop and propagates the
angular error only; Table 9 reads +1.9 pp of MMLU for that on the family matching ours
(shape-gain with gain, input rotation, 34.1 → 36.0) and 191.90 → 6.90 of perplexity without
rotation (`docs/llvq-paper-notes.md`).

`spherical_feedback` does exactly that under the coded gain: the stored block stays on the gain
grid — same word, same file, same rate, same decoder, pinned bit for bit by
`stored_blocks_still_seal_bit_for_bit` — and only the residual `E` changes, to `w − q·‖w‖/‖q‖`.
It is design C without its closed-form solve, the one cell the design C refutation of 2026-08-07
(×1.99, `docs/mesures/m3-gate-design-c-2026-08-07.txt`) did not isolate, because that arm carried
the solve that alone degraded ×1.52 (`docs/HISTORIQUE.md`, 2026-07).

## 2. The signal that argues against it, named before the run

Four times a magnitude lever that won on a local proxy composed worse over 28 layers: design C,
`group_scales`, gptq2, tetrapost. All four changed the **stored** magnitude. This one does not,
but it under-compensates: the columns after a block are corrected for less error than the file
actually carries. If the radial error of the two-level code is large enough that chasing it is
what holds the layer together, B loses. The repository holds no measurement of either share.

## 3. The setup

Qwen3-0.6B, 28 blocks, calibration wikitext-2 train 64×2048 (prefix, no seed), evaluation 12
windows ×2048, f32, Metal, rotation on, damping 1e-2, `tetra` codebook — the tetrapost protocol,
which is the M1 protocol. One variable between the arms: the mode positional.

```
LLVQ_MODEL=Qwen/Qwen3-0.6B LLVQ_THREADS=12 nice -n 10 \
  target/release/smoke 64 2048 12 2048 metal {nogs|sph} tetra 999 rot
```

Arm A is `nogs`, the published loop. Arm B is `sph`. A is re-run rather than reused: the
encoder at 5d36d52 is not the encoder at 09e0f65 that read 41.8875. Both arms run from one
binary, copied out of the checkout before the first run, in one script, A then B. Replication is
by calibration seed: seeds 1 and 2 if §4 says continue.

## 4. The decision rule, written before the first number

Read as perplexity excess of B over A at equal seed.

| First pair | Action |
|---|---|
| B below A by ≥ 1.0 % | Continue: seeds 1 and 2. Same sign on all three → propose the 4B, with its own stamped prereg |
| within ±1.0 % | One more seed. Still within ±1.0 % → close, record the null |
| B above A by ≥ 1.0 % | Close, and record a **fifth** case of a local reading composing worse |

Nothing here licenses a change to the served encoder, an MMLU claim, or a 4B run: the 4B is a
separate operator decision, and the re-encoding bar of 2.92 pp of MMLU is above the effect the
paper reads.

## 5. Signed prediction

**B below A by 1 % to 4 % of perplexity**, same sign on the three seeds. Central value −2 %.

Named against me: any move beyond 8 % in either direction means the flag is not doing what §1
says it does, and the measurement is to be doubted before the hypothesis. A rise, or an unstable
sign, refutes the prediction outright, and §2 is why that outcome is live.

## 6. Controls

1. Both arms print the **same effective b/weight** (2.1656 expected): the flag touches no bit.
2. Both arms print the **same f32 baseline perplexity** (19.5038 expected, the M1 value).
3. Both arms print the **same weight count** (440,401,920) and the same configuration line up to
   the one word `spherical feedback on/off`.
4. Arm A replays 41.8875 within the encoder drift the repository already documents; a value far
   from it is reported as a drift of the encoder, not of this lever.
5. Raw logs are kept whole under `docs/mesures/sph-ppl-0.6b-2026-09-22-brut/` and are never
   summarized before commit.
