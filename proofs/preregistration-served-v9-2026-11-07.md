# Preregistration. The served door of the sealed 4B under the v9 config, the int4 bytes held once

**Written on 2026-11-07.** Stamped with OpenTimestamps before the first measurement, then never edited; a deviation
goes beside it in `proofs/preregistration-served-v9-2026-11-07-ECARTS.md`.

Operator go for the code: "Tu vas déjà me faire v9 avec un seul copié int4 pour que je puisse avoir la v9 sur main
sur les 4, 8 et 14" (2026-11-07). The launch needs its own go. Cost: **$0.48 expected, $0.90 at worst** on l40sx1
(*estimated*, §8). Registry before the run: 271 rows, $274.46 (*computed*, `docs/data/jobs.csv` at the commit that
carries this file). The rows-grid microbench of another branch (at most $0.90) is not in that registry.

**Base commit:** the commit of `claude/serve-v9-single-int4-7aba47` that carries this file and `ops/jobs/served-v9.sh`,
or the merge that brings it in. **Image:** the Space sha rebuilt from it, passed as `IMAGE_SHA`; the launcher refuses
another. An image built before that commit refuses `"rows": "v9"` and has no tiled int4 reader.

## 1. Why this run exists

- `configs/qwen3-4b-tetra-e4-4e-v9.json` is `qwen3-4b-tetra-e4-4e-v6.json` with `"rows": "v9"` and nothing else
  different (the test `the_v9_config_is_the_v6_config_with_the_v9_rows`).
- v9 is v8 with 4E's int4 rows (v, o, the int4 downs) read from tiled words (row form 7, K11 of `docs/plan-300.md`).
  In the measurement form it decodes 238.6 tok/s against 213.3 for v6 after a 5-token prompt (x1.118 round by
  round), and 163.5 against 154.2 at 8,192 (x1.060), the same 64 tokens as the eager path at both contexts
  (*measured*, `docs/mesures/step3-4e-2026-11-07.txt`). That form held the int4 bytes twice on the card, 0.385 GB
  more (the install line of the same journal).
- Under the v9 config the loader now writes the int4 words tiled on the host and uploads them once
  (`fused_cuda::upload_int4`, `llvq_cuda::pers::q4_tile_words`). The eager path, which prefills, reads them through
  `tv_q4_ht` / `tv_q4_hht` (`llvq-llm/kernels/tv_q4_tile.cu`): the same products in the same order, only the
  addresses move. On the host, both tiled entry points return the served ones' bytes at every int4 width of the three
  sealed files, f32 before the narrowing included, and 9 mutants of the tiled reader are killed
  (`llvq-cuda/tests/pers_matches_served.rs`). The sealed file does not change, nor its 2.7320 b/param for the whole
  model: the tiling is a permutation inside each row (*computed*).
- What no run has read: the tiled readers on the card, the v9 config at the served door, the card memory it holds,
  and its text against the v6 config's.

## 2. What runs

One job, l40sx1, `ops/jobs/served-v9.sh`, timeout 30 min. The sealed 4B (`qwen3-4b-sealed.bin`, sha256 `886391a8…`),
both configs at their sha256 as committed (v9 `29b3ef32…`, v6 `ffbeb415…`, the one served-v6 read).

1. Oracle on Qwen3-0.6B (rule 10).
2. The v6 config's door: `LLVQ_CONFIG=<v6 config> fusedrun <sealed> 256`, for its card memory and its tokens.
3. The v9 door: `LLVQ_CONFIG=<v9 config> fusedrun <sealed> 256`.
4. The sweep: the same with `LLVQ_CONTEXTS=5,2048,4096,8192`.
5. The GSM8K pilot: the 50 problems of the served-v4 pilot through the v9 config.

## 3. Gates, each one voids the run

1. The image builds; `oracle` prints MATCH.
2. The object and the two configs at their bytes and sha256.
3. The v6 door compiles the served unit `6193094310bc…`, loads no tiled word, prints its card memory, and its 4E
   tokens equal the eager path's, 256 of 256.
4. The v9 door, the sweep and the pilot print:
   - the provenance `decode persistent (4E rows v9, one launch a token run)`;
   - `int4 layout: tiled (LLVQ_CONFIG), 84 records, words 0.385 GB on the card, held once`;
   - the load-time checks on the card: `tv_q4_hht equals tv_q4_ht`, and `tv_q4_hht on the tiled words equals tv_q4_hh
     on the file's order` on the three int4 widths, 2,560, 4,096 and 9,728;
   - the install line of row form 7 with the module sha256 `d9da09aa…`, the one step3-4e timed;
   - `4E v9: the 76 prologue inputs are 16-byte aligned; the vector path runs on every layer`;
   - `4E v9: the int4 records are read in place, tiled at load and held once on the card (36 layers)`, and no line
     `relaid into tiled words`;
   - RoPE equal to candle's, and the 512-thread symbol with 0 B local.

   The v9 door also compiles the v9 served unit `e269ea5d11b7…` (106,947 B, 12 parts) and prints its card memory.
5. 4E's tokens equal the eager path's under the v9 config: 256 of 256 at the door, and at each of the 4 lengths.
6. The pilot dump ends on fingerprint `94c7b98876288901`, and its header carries `"rows":"v9"`.
7. No `LLVQ_*` variable in the job environment.

## 4. Outcomes reported, never gated

- The v9 door: 4E and eager rates and their ratio round by round; the decode alone.
- The sweep: tok/s median and range at each length, and the time to the first token.
- The card memory after the load and the decoder install (`cuMemGetInfo`, the whole context), v9 against v6 in the
  same job.
- Token agreement with the v6 config (served-v6). The v6 and v9 doors' 256 tokens against `61d0654532d35120`. The
  sweep's against `61d0654532d35120`, `a4645c00f879ae80`, `cc9b3414c4d67ba6` and `9aa830caa48cb332`.
- The pilot: correct count, and completions identical to the v6 config's dump (sha256 `fc279103…`).

## 5. Signed predictions

*Estimated*: served-v6's readings times the v9 / v6 ratios of step3-4e. At 2,048 and 4,096 the ratio is interpolated
linearly in the prompt length between x1.118 (5) and x1.060 (8,192): x1.104 and x1.089. The eager path now reads
tiled int4 words; its rate is predicted unchanged within 4 %.

| quantity | point | interval |
|---|---|---|
| 4E against eager, door and the 4 lengths | **identical** | |
| v9 door, tok/s (prompt and decode) | **220** | [210, 230] |
| decode alone, v9 door, tok/s | **224** | [214, 234] |
| eager rate, v9 door, tok/s | **125.7** | [120.7, 130.7] |
| 4E / eager, v9 door, round by round | **x1.75** | [1.65, 1.85] |
| sweep, tok/s at 5 / 2,048 / 4,096 / 8,192 | **225.6 / 208.7 / 194.7 / 160.0** | [215, 236] / [199, 218] / [185, 204] / [152, 168] |
| card memory, v9 minus v6, GB | **0.00** | [-0.10, +0.10] |
| tokens against the v6 config, v6 door, v9 door and 4 lengths | **identical** | |
| pilot: completions identical to the v6 config's, of 50 | **50** | |
| pilot: correct, of 50 | **40** | |

The memory interval is decisive: two copies would read +0.385 GB.

## 6. Decision rules

- **A, identity.** 4E equals the eager path at the v6 door, the v9 door and the 4 lengths, or the run is void.
- **B, the same text as v6.** The 256 tokens at both doors and at each length, and the 50 pilot completions, should
  equal the v6 config's. A difference is a defect of the tiled readers or of the v9 loop, not a tolerance. It is
  reported to the operator at once, and v9 is not proposed.
- **C, one copy.** v9 minus v6 inside [-0.10, +0.10] GB holds the claim that the card holds the int4 bytes once. Above
  +0.10 it is refuted, and v9 is not proposed until the cause is named.
- **D, speed.** If A, B and C hold, the door's rate and the sweep are the numbers v9 would serve. Making v9 the served
  config of the 4B is the operator's decision on this reading.
- **G.** A miss is reported as a miss, with the term of §5 it refutes.

## 7. What this run does not establish

The 8B and the 14B under 4E: the install refuses both (the 8B's o_proj is Tetra and its head untied; the 14B's head
is untied), and `served.rs` refuses their shapes by name. A full GSM8K census under 4E; any MMLU reading under 4E;
Metal; any card but the L40S.

## 8. Cost

*Estimated* on l40sx1 at $1.80/h, from served-v6, which billed 825 s for the door, the sweep and the pilot, plus its
door part again (115 s) for the v6 door: about 16 min, **$0.48**. Timeout 30 min: **$0.90**. The Space rebuild from
the branch is not a job and is not counted here.
