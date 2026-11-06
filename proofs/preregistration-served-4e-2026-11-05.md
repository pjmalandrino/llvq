# Preregistration. The sealed 4B served by 4E: the served door, the context sweep and a GSM8K pilot under `configs/qwen3-4b-tetra-e4-4e.json`

**NOT STAMPED. NOT LAUNCHED.** Written on 2026-11-05 on branch `claude/serene-bohr-1edf69`, once the code it describes
existed. To be timestamped before the first measurement, then never edited; a departure goes to
`preregistration-served-4e-2026-11-05-ECARTS.md`.

The launch is the operator's, through the integration session. Cost: about **$0.69** on l40sx1 (*estimated*, §8).
Timeout 45 min: **$1.35** at worst (*computed* at $1.80/h). Spent before it: see `docs/data/jobs.csv` at launch.

## 1. Why this run exists

On 2026-11-05 the operator granted the T8 extension of the kernel attention form below 1,280 on the sealed 4B
(`docs/mesures/attn-t8-below-4b-2026-11-05.txt`, *measured*):
- paired WikiText-2 decode-path perplexity against the grouped form: +0.0318 % [-0.0151 ; +0.0786] on 32 windows of
  1,280 and +0.0018 % [-0.0053 ; +0.0088] on 12 windows of 4,096, inside +-0.12 %;
- so the kernel form is licensed at every position on this object. The 8B and 14B are not covered.

4E computes exactly that form and decodes the eager path's tokens under it:
- 179.2 tok/s against 133.0 after a 5-token prompt;
- 135.4 against 104.6 at 8,192;
- `LLVQ_PERS_4E`'s measurement form (*measured*, `docs/mesures/pers-4e-v3-2026-11-05.txt`).

No served config could name it until now. This run reads the served object under the config that does:
- its identity to the eager path under the same config;
- its speed at four context lengths;
- its tokens against the served config's;
- a first task reading, the GSM8K pilot.

## 2. What the code holds at the commit that carries this file

- `llvq_llm::served`: an optional `decode` field. Absent or `"eager"` is the eager path, as every earlier config;
  `"persistent"` is 4E. It never changes what is computed. The config's threshold of the kernel form is 0 on 4E
  and on the eager path alike, so 4E owes the eager path's tokens. It is refused:
  - beside any attention but `kernel`, and beside any other field than the served 4B's (tetra48, q4, 1, 0, f16);
  - on Metal;
  - on any shape but the Qwen3-4B's;
  - beside any of 17 measurement-mode variables.
- `llvq_llm::served_decode`: the engine. The eager prefill runs in chunks, then 4E launches of at most 64 tokens.
  Under GSM8K's two stop tokens, the tokens are cut before the first stop of either kind. 4E stops on one of them;
  the other is cut after the launch, which changes no token.
- `bin/fusedrun`:
  - the served door under a persistent config: the eager reference under the same config, then 1 + 5 rounds of
    eager and 4E. 4E's tokens must equal the eager path's in every round, or the run is void;
  - `LLVQ_CONTEXTS` decodes through the engine, with one eager run a length compared token for token;
  - the eager door now prints its token ids.
- `bin/gsm8k` decodes through 4E under the persistent config; the eager loop is unchanged.
- `configs/qwen3-4b-tetra-e4-4e.json`: the kernel config with `"decode": "persistent"`. Not the served default.

Host checks at $0:
- `tests/served_config.rs`: 25 tests, 6 of them new for the field;
- the model test: apply refuses a toy model, and the form puts every decode row on the kernel;
- `served_decode`: the launch loop equals the plain greedy loop over 375 cases of stops, launch sizes and lengths;
- 12 mutants of the new guards, all killed;
- CUDA half type-checked in `llvq-check`.
The persistent path itself runs only on a card, where 4E's identity was read three times (v1 to v3).

## 3. What runs

One image built from the commit that carries this file. One job, l40sx1, `ops/jobs/served-4e.sh`. The sealed 4B
(`sealed-4b-2026-09-23/qwen3-4b-sealed.bin`, sha256 `886391a8…`) is checked by size, then by sha256. Both configs are
embedded in the job and checked by sha256.

1. **Oracle** on Qwen3-0.6B (rule 10).
2. `LLVQ_CONFIG=<4E config> fusedrun <sealed> 256`: the door, 256 tokens after the 5-token prompt.
3. `LLVQ_CONFIG=<4E config> LLVQ_CONTEXTS=5,2048,4096,8192 fusedrun <sealed> 256`: the sweep.
4. The same door and sweep under `configs/qwen3-4b-tetra-e4-kernel-from1280.json`, the served config.
5. `LLVQ_CONFIG=<4E config> gsm8k <sealed> cuda 50`: the pilot's 50 problems, dataset revision `740312ad…`.
   - They are compared with the grouped census rows at the same indices
     (`docs/mesures/gsm8k-grouped-2026-11-04-brut/gsm8k-4b-grouped.jsonl`, sha256 `eda98402…`, 40 of 50 correct).
   - The comparison is embedded in the job as index, correctness and a completion digest.

## 4. Gates

Each voids the run:

1. The image builds; `oracle` prints MATCH.
2. The object at its bytes and sha256; both configs at theirs.
3. The 4E door:
   - prints the provenance `attention kernel from position 0 on grow(256), decode persistent`;
   - compiles the served unit `6193094310bc…`;
   - loads `pers_4e_tokens_f*_t512` with 0 B local, and finds a RoPE form equal to candle's;
   - prints `4E tokens equal the eager path's under this config, 256 of 256`.
4. The 4E sweep prints the same identity at each of its 4 lengths, and 4 table rows.
5. The served config's door and sweep print `attention kernel from position 1280`.
6. The pilot's dump ends on the pilot's prompt fingerprint `94c7b98876288901`, under the 4E provenance, with 4E
   loaded.
7. No `LLVQ_*` variable in the job environment.

## 5. Outcomes reported, never gated

- The door under each config, tok/s median and range (prompt and decode, as the door times a generate).
- Under the 4E config, round by round:
  - 4E and eager rates, and their ratio;
  - the decode alone.
- The sweep under each config, tok/s median and range per length, and the time to the first token.
- Token agreement between the two configs: at the door, and at each length.
- The GSM8K pilot:
  - correct counts under the 4E config and in the census;
  - identical completions;
  - discordant problems, McNemar's exact p, and the delta in pp.

## 6. Signed predictions

*Estimated* from the 4E v3 and attn-t8-r2 journals; nothing fitted to a number of this run.

| quantity | point | interval |
|---|---|---|
| 4E against eager, same config, door and the 4 lengths | **identical** | |
| 4E door, tok/s (prompt and decode) | **176** | [168, 182] |
| 4E decode alone, door, tok/s | **179** | [174, 184] |
| eager door under the 4E config, tok/s | **132** | [127, 137] |
| 4E / eager, door, round by round | **x1.33** | [1.25, 1.40] |
| served config door, tok/s | **133** | [128, 137] |
| 4E sweep, tok/s at 5 / 2,048 / 4,096 / 8,192 | **179 / 166 / 154 / 135** | [172, 186] / [155, 175] / [143, 165] / [129, 141] |
| served sweep, tok/s at the same lengths | **134.7 / 120.5 / 117.6 / 104.0** | each +-3 % |
| tokens of the two configs at 2,048, 4,096, 8,192 | **identical** | |
| tokens of the two configs at the 5-token door and length 5 | **identical** | or a first divergence anywhere |
| pilot: identical completions, of 50 | **32** | [20, 45] |
| pilot: correct under the 4E config, of 50 | **40** | [37, 43] |

At 2,048 and later, every decode position of both configs is past 1,280. Both prefills of a chunk wider than the
kernel's rows take the grouped path. So the two configs run the same arithmetic there: that line is a consequence of
the code, read on the card. At the 5-token prompt they differ by design: the prefill rows and every decode row on
the kernel, against the grouped form.

## 7. Decision rules

- **A, identity.** If 4E's tokens equal the eager path's under the 4E config at the door and at the 4 lengths, the
  persistent engine serves the config's arithmetic on this card. Otherwise the run is void, and the first divergence
  goes to the host with the card's tokens.
- **B, speed.** The 4E door's rate and its round-by-round ratio to the eager path under the same config are the
  served numbers of this config. The served config's door is printed beside, from another process, with no ratio
  formed round by round across the two (rule 7).
- **C, agreement past 1,280.** The two configs' tokens at 2,048, 4,096 and 8,192 should be identical. A divergence
  there is a defect, not a tolerance: the config is not proposed until it is found. At length 5 the agreement is
  reported, inside the T8 granted.
- **D, the GSM8K pilot.** Correct counts within 2 problems (4 pp) and McNemar p >= 0.05 against the census: the pilot
  does not object. The full census through the 4E config (1,319 problems, its own job) is the task-level check of
  the extended form. Outside that band, the full census runs before any proposal.
- **E, serving.** Proposing this config as the served default is the operator's decision. Rules A and C must hold
  and D must not object. This run changes no served file.
- **G.** A miss is reported as a miss, with the term of §6 it refutes. No threshold moves after the clock.

## 8. Cost

*Estimated* on l40sx1 at $1.80/h:

| step | time |
|---|---|
| container and image | 3 min |
| oracle | 1 to 2 min |
| 4E door (load, 4E install and RoPE check, 1 + 6 eager and 6 4E generates of 256) | 2.5 min |
| 4E sweep (6 rounds a length, plus one eager run a length; the 8,192 prefill about 31 s each) | 6.5 min |
| served door and sweep | 7 min |
| GSM8K pilot, 50 problems | 2 min |

About 23 min, **$0.69**. Timeout 45 min: **$1.35** at worst.

## 9. What this run does not establish

- A GSM8K score under the extended form: 50 problems are a pilot, the census is 1,319.
- MMLU, which scores prompts and decodes nothing: the engine cannot move it.
- The 8B or the 14B, which the T8 does not cover and the config refuses.
- Any card but the L40S.
