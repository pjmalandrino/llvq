# Preregistration. The served door of the sealed 4B under the v6 config, 4E with the v6 row loop

**Written on 2026-11-07.** Stamped with OpenTimestamps before the first measurement, then never edited; a deviation
goes beside it in `proofs/preregistration-served-v6-2026-11-07-ECARTS.md`.

Operator go for the code: "v6 c'est pas une découverte, on le sait, on a mesuré … tu peux implémenter v6 sur le 4"
(2026-11-07). The launch needs its own go. Cost: **$0.42 expected, $0.75 at worst** on l40sx1 (*estimated*, §8).
Registry before the run: 267 rows, $273.30 (*computed*, `docs/data/jobs.csv` at the commit that carries this file).

**Base commit:** the commit of `claude/serve-4e-v6-6e15c0` that carries this file and `ops/jobs/served-v6.sh`, or
the merge that brings it in. **Image:** the Space sha rebuilt from it, passed as `IMAGE_SHA`; the launcher refuses
another. An image built before that commit refuses `"rows": "v6"`.

## 1. Why this run exists

- `configs/qwen3-4b-tetra-e4-4e-v6.json` is the 4B's served config, `qwen3-4b-tetra-e4-4e-v4.json`, with
  `"rows": "v6"` and nothing else different (the test `the_v6_config_is_the_v4_config_with_the_v6_rows`).
- v6 is v4 with ln1, ln2 and the down's input in one round trip a thread (row form 4, `llvq_pers.cuh` section 7b).
- In the measurement form, v6 decodes 212.6 tok/s against 190.2 for v4 after a 5-token prompt (x1.118 round by
  round), and 152.0 against 138.9 at 8,192 (x1.094). It gives the same 64 tokens as the eager path at both contexts
  (*measured*, `docs/mesures/pers-4e-v6-2026-11-05.txt`).
- What no run has read: the served door and the sweep under this config, and its text against the v4 config's.

## 2. What runs

One job, l40sx1, `ops/jobs/served-v6.sh`, timeout 25 min. The sealed 4B (`qwen3-4b-sealed.bin`, sha256 `886391a8…`),
the config at its sha256 as committed.

1. Oracle on Qwen3-0.6B (rule 10).
2. The door: `LLVQ_CONFIG=<v6 config> fusedrun <sealed> 256`.
3. The sweep: the same with `LLVQ_CONTEXTS=5,2048,4096,8192`.
4. The GSM8K pilot: the 50 problems of the served-v4 pilot through the config.

## 3. Gates, each one voids the run

1. The image builds; `oracle` prints MATCH.
2. The object and the config at their bytes and sha256.
3. The door, the sweep and the pilot print:
   - the provenance `decode persistent (4E rows v6, one launch a token run)`;
   - the install line `row form 4`;
   - `4E v6: the 75 prologue inputs are 16-byte aligned; the vector path runs on every layer`;
   - RoPE equal to candle's;
   - the 512-thread symbol with 0 B local.

   The door also compiles the served unit `6193094310bc…`.
4. 4E's tokens equal the eager path's under the config: 256 of 256 at the door, and at each of the 4 lengths.
5. The pilot dump ends on fingerprint `94c7b98876288901`, and its header carries `"rows":"v6"`.
6. No `LLVQ_*` variable in the job environment.

## 4. Outcomes reported, never gated

- The door: 4E and eager rates and their ratio round by round; the decode alone.
- The sweep: tok/s median and range at each length, and the time to the first token.
- Token agreement with the v4 config (served-v4). The door's 256 tokens against `61d0654532d35120`. The sweep's against
  `61d0654532d35120`, `a4645c00f879ae80`, `cc9b3414c4d67ba6` and `9aa830caa48cb332`.
- The pilot: correct count, and completions identical to the v4 config's dump (sha256 `a7203fa6…`).

## 5. Signed predictions

*Estimated*: served-v4's readings times the v6 / v4 ratios of pers-4e-v6. At 2,048 and 4,096 the ratio is
interpolated linearly in the prompt length between x1.118 (5) and x1.094 (8,192): x1.112 and x1.106.

| quantity | point | interval |
|---|---|---|
| 4E against eager, door and the 4 lengths | **identical** | |
| door, tok/s (prompt and decode) | **199** | [191, 207] |
| decode alone, door, tok/s | **202** | [194, 210] |
| 4E / eager, door, round by round | **x1.57** | [1.50, 1.64] |
| sweep, tok/s at 5 / 2,048 / 4,096 / 8,192 | **202.1 / 188.5 / 179.3 / 152.6** | [194, 210] / [181, 196] / [172, 187] / [146, 159] |
| tokens against the v4 config, door and 4 lengths | **identical** | |
| pilot: completions identical to the v4 config's, of 50 | **50** | |
| pilot: correct, of 50 | **40** | |

## 6. Decision rules

- **A, identity.** 4E equals the eager path at the door and the 4 lengths, or the run is void.
- **B, the same text as v4.** The 256 tokens at the door and at each length, and the 50 pilot completions, should
  equal the v4 config's. A difference is a defect of the v6 loop, not a tolerance. It is reported to the operator at
  once, and v4 stays served.
- **C, speed.** If A and B hold, the door's rate and the sweep are the numbers v6 would serve. Making v6 the served
  config is the operator's decision on this reading.
- **G.** A miss is reported as a miss, with the term of §5 it refutes.

## 7. What this run does not establish

A full GSM8K census under 4E; any MMLU reading under 4E; the 8B and the 14B; Metal; any card but the L40S.

## 8. Cost

*Estimated* on l40sx1 at $1.80/h, from served-v4, which billed 14 min for the same three parts: about 14 min,
**$0.42**. Timeout 25 min: **$0.75**. The Space rebuild from the branch is not a job and is not counted here.
