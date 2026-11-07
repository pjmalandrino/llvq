# Preregistration. The served doors of the sealed 8B and 14B under their v9 configs

**Written on 2026-11-07.** Stamped with OpenTimestamps before the first measurement, then never edited; a deviation
goes beside it in `proofs/preregistration-served-v9-8b-14b-2026-11-07-ECARTS.md`.

Operator go, 2026-11-07: "Tu me fait 4,8 et 14 en V9 et tu envoie les tests point barre", with the doors launched
without a further go while their total stays under about $5. Cost: **8B $1.05, 14B $1.80 expected; $1.80 and $2.85
at worst** on l40sx1 (*estimated*, §8). Registry before these runs: 271 rows, $274.46 (*computed*,
`docs/data/jobs.csv`); the project total the operator's session gave is $274.71.

**Base commit:** the commit of `claude/serve-v9-single-int4-7aba47` that carries this file and
`ops/jobs/served-v9-big.sh`. **Image:** the Space sha rebuilt from it, passed as `IMAGE_SHA`.

## 1. Why these runs exist

- `configs/qwen3-8b-tetra-e4-4e-v9.json` and `configs/qwen3-14b-tetra-e4-4e-v9.json` carry the 4B v9 config's
  fields: attention `kernel`, decode `persistent`, rows `v9`. The 8B and the 14B serve today their grouped configs.
- 4E now reads what those files hold: the head through pointers of its own (both heads are untied), and the 8B's
  `o_proj` in Tetra under its own rotation (a `Pers4eOT` image a layer). On the host, the token loop of v4, v6, v8
  and v9 with a Tetra o and an untied head equals the served chain byte for byte, and 5 mutants of the new guards are
  killed (`llvq-cuda/tests/pers_matches_served.rs`). v9 asks 96,064 B of shared memory a CTA at 14B, 85,952 at 8B,
  of the 101,376 an L40S CTA may ask (*computed*, `the_4e_v9_shared_bytes_at_the_three_sealed_shapes`).
- Under the v9 config, 4E and the eager path both run the kernel attention form, so 4E owes the eager path's tokens.
  The kernel form below 1,280 has no T8 at 8B or 14B, and the served grouped form is not the kernel form: part 4 is
  the minimal quality check of `CLAUDE.md`, the paired perplexity inside the +-0.12 % band of `docs/ETAT.md` §4.

## 2. What runs

One job a size, l40sx1, `SIZE=8b|14b ops/jobs/served-v9-big.sh`. The sealed 8B (`qwen3-8b-sealed-B.bin`, sha256
`7bdb9a55…`) or 14B (`qwen3-14b-sealed.bin`, sha256 `61db37fe…`), both configs at their sha256 as committed.

0. The grouped config at the door: `LLVQ_CONFIG=<grouped> fusedrun <sealed> 256`.
1. The v9 door: `LLVQ_CONFIG=<v9> fusedrun <sealed> 256`.
2. The sweep: the same with `LLVQ_CONTEXTS=5,2048,4096,8192`.
3. The GSM8K pilot: the 50 problems the 4B pilots ran, through the v9 config.
4. `LLVQ_GQA_PPL=4096,6,60,kernel0`: the kernel form on every decode row against the grouped form, paired WikiText-2
   decode-path perplexity on 6 windows of 4,096 from window index 60.

## 3. Gates, each one voids the run

1. The image builds; `oracle` prints MATCH. The object and the two configs at their bytes and sha256.
2. Every v9 log (door, sweep, pilot) prints: the provenance `decode persistent (4E rows v9, one launch a token
   run)`; the v9 served unit `e269ea5d…`; `int4 layout: tiled (LLVQ_CONFIG), N records … held once` with N = 53 (8B)
   or 99 (14B); the tiled reader checked against the served one on the size's 2 int4 widths; the 4E v9 module
   `1b422b62…`; `4E: head on a table of its own (untied) (151936 rows)`; at 8B, `4E: 36 Tetra o_proj read under their
   own rotation, 0 int4`; the 76 (8B) or 84 (14B) prologue inputs aligned; the int4 records read in place and no
   relay line; RoPE equal to candle's; the 512-thread symbol with 0 B local.
3. 4E's tokens equal the eager path's under the v9 config: 256 of 256 at the door and at each of the 4 lengths.
4. The pilot dump ends on fingerprint `94c7b98876288901`, its header carries `"rows":"v9"`.
5. Part 4 prints the kernel replay check on the card (`attn_dec_g4_u64` at 8B, `attn_dec_g5_u64` at 14B), the kernel
   taking positions 0 and later, and window 1's prefill path bit for bit between the arms.
6. No `LLVQ_*` variable in the job environment.

## 4. Outcomes reported, never gated

The v9 and grouped door rates (two processes, one job), the decode alone, the sweep, the card memory of each config,
the tokens of each config (the grouped config's are not owed), the pilot's correct count against the grouped census
rows of the same 50 problems, the perplexity interval and the top-1 agreement.

## 5. Signed predictions

*Estimated*, wide on purpose: no 4E reading exists at these sizes. The 4B's v9 decodes about x1.75 over its eager path
(step3-4e); the 8B's Tetra o runs the unstaged path.

| quantity | 8B | 14B |
|---|---|---|
| 4E against eager, door and the 4 lengths | **identical** | **identical** |
| grouped door, tok/s | **106** [100, 112] | **63** [59, 67] |
| v9 door, tok/s | **140** [110, 175] | **85** [68, 105] |
| sweep at 8,192, v9, tok/s | **100** [75, 130] | **62** [48, 80] |
| pilot correct, of 50 (grouped census: 46 and 46) | **46** [43, 49] | **46** [43, 49] |
| perplexity, kernel from 0 against grouped | **+0.00 %** inside [-0.05, +0.05] | **+0.00 %** inside [-0.05, +0.05] |

## 6. Decision rules

- **A, identity.** 4E equals the eager path at the door and the 4 lengths, or the size's run is void.
- **B, quality.** The perplexity interval inside +-0.12 % licenses the kernel form from position 0 at that size, the
  form the v9 config serves. Outside, the v9 config of that size is not proposed.
- **C, serving.** If A and B hold, the v9 door's rate and sweep are what the size would serve. The operator's message
  of 2026-11-07 asks for the served configs to be switched on these readings; the docs record it.
- **G.** A miss is reported as a miss, with the term of §5 it refutes.

## 7. What these runs do not establish

A full GSM8K census or an MMLU under 4E; the kernel form's perplexity on windows shorter than 4,096 in isolation;
Metal; any card but the L40S.

## 8. Cost

*Estimated* on l40sx1 at $1.80/h, from gqa-8b (44 min) and gqa-14b (74 min), which loaded each object several times
with a perplexity band, and served-v6 (825 s for the 4B's door, sweep and pilot): 8B about 35 min, **$1.05**, timeout
60 min (**$1.80**); 14B about 60 min, **$1.80**, timeout 95 min (**$2.85**). With the 4B door ($0.48), about $3.33,
under the operator's $5.
