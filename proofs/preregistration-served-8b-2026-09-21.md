# Preregistration. The 8B paper-2 object on a card: kernel bench and served decode (2026-09-21)

**Written on 2026-09-21 and TIMESTAMPED (`ots stamp`) BEFORE either job is launched.** The
commit that carries it follows on the operator's go. Go and cap as in
`preregistration-census-8b-2026-09-21.md`: the whole 8B chain, $20, solo unless a big problem.

Not edited again. A fact it gets wrong goes beside it, in a `-ECARTS.md`.

**Cost.** Two jobs on l40sx1, image `a963a020` frozen. `planesbench`: $0.27 central, **hard cap
$0.90** (30 min). Served decode: $0.65 central, **hard cap $1.35** (45 min). Ceilings of the
whole 8B chain: census 4.05 + training 11.25 + FT census 1.80 + these 2.25 = **$19.35 ≤ $20**
(*computed*). Launchers: `ops/jobs/bench-8b.sh`, `ops/jobs/served-8b.sh`; config
`configs/qwen3-8b-tetra-q5.json` (the 4B's served values, a test pins them; the image does not
carry the file, so the job writes it from the command line and checks its sha256).

## Question

Does the Tetra kernel serve a Qwen3-8B file, give the dense arm's tokens, and at what speed and
VRAM? No Tetra file above 4B has ever run through `tv_tetra48_h`.

## Setup

1. **`planesbench`**, ball file first (`qwen3-8b-c12-77e76284/qwen3-8b-llvq.bin`, the only 8B
   `Planes14` file), Tetra second: **the base** `qwen3-8b-dclm.bin`. The fold changes row scales
   only, and `rowscale.rs:17-18` states that no index byte changes, so the base and the trained
   file present the kernel the same stream; the base lets the bench run while the training does.
   Arms `fp16, planes14, nullk, tetra48`, tile unset (64 on sm_89), seven rounds, two dropped,
   ratios formed round by round. Oracle first.
2. **Served decode** on **the trained file**, `PHASE=full DOOR=1`: oracle; object bytes and
   sha256; the 57-question served door through the kernel; the prefill gate at 203 tokens;
   `fusedrun` 256 tokens against the dense arm of the same process at the served flags
   (`tetra48`, q8 embedding, `ROT_SHARE=1`, `FUSE=0`, KV f16), five rounds; the same at
   `LLVQ_EMBED=f16`, the same-head arm (hard rule 4).

## Controls

1. Oracle MATCH; object bytes and sha256 equal to the Mac's.
2. The log says `tile 64 (served: measured optimum for sm_89)` and `36 int4`.
3. The prefill gate: same argmax over 203 tokens batched and one by one.
4. The door dump carries `# arithmetic=served kernel` and `# layout=tetra48`.
5. `planesbench`'s f64 row check passes on every arm.

## What gets published, and what does not get compared

Published: tok/s median and range for the q8 arm, the f16 same-head arm and the dense arm, their
ratios round by round, GB on card, the first divergence position; `planesbench` ms, GB read and
GB/s per arm, Tetra against FP16 and `Planes14` in the same process. The two `f16 lm_head` lines
of `planesbench` are struck: `planesbench.rs:3537, 3554` hard-code the 4B head. Not compared:
anything across cards or against another process's numbers.

## Decision rule

| result | reading | what follows |
|---|---|---|
| controls pass, first divergence absent or after token 5 | the kernel serves the 8B | publish; the chain closes |
| controls pass, first divergence at token 5 or earlier | a defect in the served path at 8B | report ("gros problème"); no speed is published as the object's |
| a control fails or a job dies | no number from that job | diagnose; a relaunch only if the $20 still covers its ceiling |
| otherwise | not settled | operator decision |

## Signed prediction

| quantity | point | interval |
|---|---|---|
| q8 arm, tok/s | **80** | [70, 90] |
| dense arm, tok/s | **26.5** | [25, 28] |
| f16 same-head arm against dense | **×1.43** | [×1.25, ×1.70] |
| GB on card, q8 arm | **3.2** | [3.0, 3.5] |
| first divergence, q8 arm | **none in 256** | after token 32 |
| `planesbench` FP16 | **21 ms** | [19, 23] |
| `planesbench` Tetra against FP16 | **×3.2** | [×2.6, ×4.2] |

Rationale: the 8B `Planes14` served 75.5 tok/s in 5.41 GB and the dense f16 path 26.4 tok/s
(*measured*, ETAT §2); at the 4B, Tetra and `Planes14` served within 1 % of each other and
Tetra read ×3.21 FP16 in `planesbench` at tile 64 (*measured*, `tuile-l40s-2026-09-20.txt`);
FP16 scales with the weights, ×1.91. The q8 arm holds 3.14 GB of the bare 8B plus the int4
`v_proj` (*computed*). Flaw: no 8B Tetra has ever run through the kernel; the head at 8B is
untied, 1.24 GB per table, which weighs on the same-head arm more than at the 4B.
