# Preregistration — does the Euclidean gain rule move perplexity? Qwen3-0.6B

**Written, committed and TIMESTAMPED on 2026-09-15, BEFORE the first run.**
Operator go given 2026-09-15 for the plan whose step 1 this is.

🚨 **This file is not edited again**: the stamp attests these bytes at this date.
A fact it gets wrong is written *beside* it, in a `-ECARTS.md` (CLAUDE.md §7).

**What is measured**: the code at `09e0f65`, encoder switch `tetrapost`.

**Cost: $0, about 1 h of Mac now** (2 runs; *estimated* from the 26 to 40 min per
0.6B run measured on 12 runs, `docs/mesures/m1-hessienne-shrink-2026-09-02.txt`),
**plus at most 2 h more** if §4 says continue. No other Metal job runs beside it.

---

## 1. The question

`docs/mesures/gain-desaccord-reel-2026-09-14.txt` measures that the served gain
rule and the Euclidean optimum disagree on 10.119 % of real compensated blocks,
and that switching rule is worth **−1.13 % of squared reconstruction error** at
identical rate, format and decoder, in 12 cells of 12.

That is a *local* quantity, on the block, before the GPTQ loop compensates for
it. This asks the only question that matters next: does it move perplexity.

## 2. The signal that argues against it, named before the run

The Schur pilot already replayed both gains to the end of the row on 144 branch
sites. Against the served rule, the Euclidean rule's mean rollout regret **fell
31.47 % on seed 1 and rose 92.17 % on seed 2**
(`docs/mesures/tetra-schur-pilot-2026-09-14.txt`). After continuation, on that
sample, it is not stably better.

So the local gain is solid and its propagation is not. The repository also holds
three cases where a better local proxy composed worse (`ETAT.md` §7). This run
exists because those two facts cannot be reconciled by argument.

## 3. The setup

Qwen3-0.6B, 28 blocks, calibration wikitext-2 train 64×2048, evaluation 12
windows ×2048, f32, Metal, `nogs`, rotation on — the protocol of the M1 A/B, and
the published path. One variable between the two arms: the codebook word.

```
LLVQ_MODEL=Qwen/Qwen3-0.6B LLVQ_THREADS=12 nice -n 10 \
  cargo run --release -p llvq-llm --features metal,fast-linalg --bin smoke -- \
  64 2048 12 2048 metal nogs {tetra|tetrapost} 999 rot
```

Arm A is `tetra`, the served rule. Arm B is `tetrapost`, the Euclidean rule.
Replication is by calibration seed: the first pair runs on the prefix (no
`LLVQ_CALIB_SEED`), then seeds 1 and 2 if §4 says continue.

## 4. The decision rule, written before the first number

Read on the median of the arms at equal seed, as perplexity excess of B over A.

| Result over the three seeds | Action |
|---|---|
| B below A by ≥ 0.5 % and the same sign on all three | Continue: 4B, two arms, with its own stamped prereg |
| within ±0.5 %, or sign unstable across seeds | Close the lead. Record the negative result |
| B above A by ≥ 0.5 % on the median | Close, and record a **fourth** case of a local proxy composing worse |

The first pair alone does not decide. A first pair inside ±0.5 %, or of the wrong
sign, stops the queue there and the remaining two seeds are not run.

## 5. Signed prediction

**B below A by 0.2 % to 1.2 % of perplexity**, same sign on the three seeds.

Named against me: any move beyond 3 % in either direction means the switch is not
doing what §1 says it does, and the measurement is to be doubted before the
hypothesis. A rise, or an unstable sign, refutes the prediction outright — and §2
is why that outcome is live rather than a formality.

## 6. Controls

1. Both arms print the **same effective b/weight**: the switch touches the choice
   of one bit, never the rate. A difference voids the comparison.
2. Both arms print the **same f32 baseline perplexity**: same evaluation windows.
3. `tetra` is re-run rather than reused, so both arms come from one process
   generation on one machine.
4. Raw logs are kept whole and are never summarized before commit.

Control 1 of the M1 protocol — replaying a known published value — has no
counterpart here: no Tetra 0.6B perplexity exists in any journal. Arm A is that
reference, and it is stated as new rather than as a replay.

## 7. What a positive result would and would not license

It licenses one 4B arm, on perplexity. It licenses **no MMLU claim**: the
sampling error is 1.339 pp and the expected effect is far under it, so an MMLU
arm on this lever reports noise until row A of `ROADMAP-QUALITY.md` lands. It
licenses no change to the served encoder, which is an operator decision on a
fundamental criterion.
