# Preregistration. The served door of the sealed 4B under its served config, 4E with the v4 row loop

**Written on 2026-11-05.** Stamped with OpenTimestamps before the first measurement, then never edited; a deviation
goes beside it in `proofs/preregistration-served-v4-2026-11-05-ECARTS.md`.

Operator go: "Oui, lance" (2026-11-05), for this reading at the cost below. Cost: **$0.36 expected, $0.75 at worst**
on l40sx1 (*estimated*, §8). Registry before the run: 265 rows, $272.64 (*computed*, `docs/data/jobs.csv` on
`integration/lot1`).

**Base commit:** the commit of `integration/lot1` that carries this file and `ops/jobs/served-v4.sh`. **Image:** the
Space sha built from it, passed as `IMAGE_SHA`; the launcher refuses another.

## 1. Why this run exists

- The operator made `configs/qwen3-4b-tetra-e4-4e-v4.json` the 4B's served config on 2026-11-05 (commit e897aec).
- It is the 4E config read by served-4e with one field more, `"rows": "v4"`. The v4 row loop computes the same
  arithmetic as v3 and the eager path: same 64 tokens at 5 and 8,192 in the measurement form (*measured*,
  `docs/mesures/shaker-4e-2026-11-05.txt`).
- What no run has read: the served door and the sweep under this config, and its text against the v3 config's.

## 2. What runs

One job, l40sx1, `ops/jobs/served-v4.sh`, timeout 25 min. The sealed 4B (`qwen3-4b-sealed.bin`, sha256 `886391a8…`),
the config at its sha256 as committed.

1. Oracle on Qwen3-0.6B (rule 10).
2. The door: `LLVQ_CONFIG=<v4 config> fusedrun <sealed> 256`.
3. The sweep: the same with `LLVQ_CONTEXTS=5,2048,4096,8192`.
4. The GSM8K pilot: the 50 problems of served-4e's pilot through the config.

## 3. Gates, each one voids the run

1. The image builds; `oracle` prints MATCH.
2. The object and the config at their bytes and sha256.
3. The door and the sweep print the provenance `decode persistent (4E rows v4, one launch a token run)`, the install
   line `row form 2`, RoPE equal to candle's, the 512-thread symbol with 0 B local; the door compiles the served unit
   `6193094310bc…`.
4. 4E's tokens equal the eager path's under the config at the door (256 of 256) and at the 4 lengths.
5. The pilot dump ends on fingerprint `94c7b98876288901` under the v4 provenance.
6. No `LLVQ_*` variable in the job environment.

## 4. Outcomes reported, never gated

- The door: 4E and eager rates and their ratio round by round; the decode alone.
- The sweep: tok/s median and range at each length, and the time to the first token.
- Token agreement with the v3 config (served-4e): the door's fingerprint against `61d0654532d35120`, the sweep's
  against `61d0654532d35120`, `a4645c00f879ae80`, `cc9b3414c4d67ba6`, `9aa830caa48cb332`.
- The pilot: correct count, and completions identical to the v3 config's dump (sha256 `1512a903…`).

## 5. Signed predictions

*Estimated* from served-4e (v3 at the door) times the shaker's v4/v3 ratios.

| quantity | point | interval |
|---|---|---|
| 4E against eager, door and the 4 lengths | **identical** | |
| door, tok/s (prompt and decode) | **179** | [172, 190] |
| decode alone, door, tok/s | **181.5** | [175, 192] |
| 4E / eager, door, round by round | **x1.42** | [1.34, 1.50] |
| sweep, tok/s at 5 / 2,048 / 4,096 / 8,192 | **181.5 / 170.6 / 161.8 / 138.8** | [175, 192] / [163, 180] / [155, 171] / [134, 145] |
| tokens against the v3 config, door and 4 lengths | **identical** | |
| pilot: completions identical to the v3 config's, of 50 | **50** | |
| pilot: correct, of 50 | **40** | |

## 6. Decision rules

- **A, identity.** 4E equals the eager path at the door and the 4 lengths, or the run is void.
- **B, the same text as v3.** The tokens and the pilot completions should equal the v3 config's. A difference is a
  defect of the v4 loop, not a tolerance: it is reported to the operator at once, with the proposal to serve the v3
  config until it is found.
- **C, speed.** The door's rate and the sweep are the served numbers of the 4B's served config.
- **G.** A miss is reported as a miss, with the term of §5 it refutes.

## 7. What this run does not establish

A full GSM8K census under 4E; the 8B and the 14B; Metal; any card but the L40S.

## 8. Cost

*Estimated* on l40sx1 at $1.80/h: container and oracle 4 min, door 2.5 min, sweep 5 min, pilot 2 min: about
12 min, **$0.36**. Timeout 25 min: **$0.75**.
