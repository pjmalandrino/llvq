# Preregistration. A second architecture: the 252-matrix bench and the served files on an A100

**Written, committed and TIMESTAMPED on 2026-10-06, BEFORE the run.**
Operator go given 2026-10-06, for a test that holds as a second architecture. Cost: about
**$3.35**, *estimated* at the $2.50/h `a100-large` rate the ledger shows (`a4-a100-banc`, 6 min,
$0.25), over three jobs: the bench about 30 min, $1.25, timeout 45 min; the served files about
30 min, $1.25, timeout 50 min; vLLM about 20 min, $0.83, timeout 35 min. At worst $5.42. Campaign
cap $9, of which $0.50 is spent and up to $2.70 is committed to `banc-252`, so the worst case of
the campaign is $8.62.

Measured code, three images. The bench runs on the standard runner image
`97a2b62a6d0c8911dcd8e4d26a40461f0196ccc9` (commit `5333ac8`), the image of `banc-252`.
The served files run on `Pier-Jean/llvq-runner-cuda-sm80`, rebuilt for this run from the
commit that carries this prereg with `ops/run.py publish --cuda --compute-cap 80`: `oracle` and
`fusedrun` run candle, whose kernels the image compiles for one compute capability, and the
standard image targets 89. That Space's sha is written in the journal before its job launches.
Between `5333ac8` and this commit the kernel sources differ by comments only. vLLM runs
`vllm/vllm-openai:v0.26.0` at digest `sha256:ffb2d59b...`, the image of every published vLLM speed.

## 1. Why this run exists

Every number of paper 2 comes from one NVIDIA L40S (Ada, sm_89, GDDR6 at 0.86 TB/s). An A100
(Ampere, sm_80, HBM2e at 2.0 TB/s) differs in the one ratio the paper's speed rests on: memory
bandwidth against compute per SM. The paper's limitation says that on an A100 none of our earlier
lattice kernels beat FP16, and that `Tetra` never ran there.

Rule 9 first. `f4-a100-2026-08-18` timed the earlier layouts on this card at tile 128: nullk
4.107 ms, FP16 6.915, AWQ 3.793, `Planes14` 8.742. It has no `Tetra` arm and no served run. Those
numbers are the prediction, not the result.

## 2. What runs

**Jobs 1 and 2, `ops/jobs/a100-runner.sh`, `a100-large`, `LLVQ_NVRTC_ARCH=compute_80` on every
command.** `PART=bench` runs steps 2 and 3 on the standard image, as F4 and A4 ran `planesbench`
on this card. `PART=served` runs steps 1 and 4 on the sm80 image.

1. `oracle Qwen/Qwen3-0.6B 64 cuda` (hard rule 10).
2. The ten-arm bench of `preregistration-banc-252-2026-10-06.md`, same two files, same phases, the
   tile unset. sm_80 has no measured row, so the tile is the fallback 128, and the header must
   say so.
3. The tile sweep, `fp16,planes14,nullk,tetra48` at 128, 64 and 32, one process each.
4. The three sealed files, as `ops/jobs/paper-served.sh` ran them on the L40S: the prefill gate
   through the served config, then the q4-table arm and the f16-table arm, 256 tokens each, each
   against the dense reconstruction in the same process.

**Job 3, `ops/jobs/a100-vllm.sh`, the vLLM image, `a100-large`.** `ops/awq_speed.py --arms f16,awq_marlin` at 4B, 8B and
14B, both arms interleaved in one process per size (80 GB holds both), 2 warm-up and 5 timed
rounds, 128 tokens, as on the L40S.

Quality is not re-measured: MMLU, GSM8K and perplexity are properties of the file and the
arithmetic, and the f64 check of step 2 and the dense comparison of step 4 tie the kernel to the
reconstruction on this card.

## 3. Gates, each one voids its part

1. The card reports `A100` and compute capability 8.0.
2. Oracle `MATCH` on the card.
3. `tetra48: 252 of 252 matrices matched by name` in every bench process, and every arm's worst
   error under its threshold.
4. The bench header reads `tile 128 (served fallback: sm_80 has no measured row)`.
5. Each sealed file at its byte count and its published sha256: `886391a8`, `7bdb9a55`, `61db37fe`.
6. The prefill gate's argmax agrees with the dense arm on each size.
7. vLLM reports 0.26.0, and the uploaded script matches its sha256.

## 4. Signed prediction

Bench, medians in ms at the fallback tile 128:

| arm | point | interval | basis |
|---|---|---|---|
| FP16 | 6.92 | [6.7, 7.1] | F4 |
| AWQ | 3.79 | [3.7, 3.9] | F4 |
| `Planes14` | 8.74 | [8.5, 9.0] | F4 |
| nullk | 4.11 | [4.0, 4.2] | F4 |
| **`Tetra`, 252 matrices** | **7.5** | [6.5, 9.0] | L40S 4.355 at tile 128 times 1.73, the A100/L40S ratio of the F4 floor and of `Planes14` |

So `Tetra` is predicted **slower than FP16** on this card, 0.92 times its speed [0.77, 1.06], and
about half the speed of AWQ. The tile sweep moves it by less than on the L40S, a range of 6 %
[2, 15], because an Ampere SM has 192 KiB of L1 and shared memory where Ada has 128.

Served decode, tokens per second, q4 tables, 256 tokens:

| size | ours, our engine | FP16, vLLM | AWQ, vLLM |
|---|---|---|---|
| 4B | 75 [55, 95] | 165 [130, 200] | 240 [190, 290] |
| 8B | 60 [45, 75] | 95 [75, 115] | 165 [130, 200] |
| 14B | 37 [27, 47] | 55 [45, 65] | 115 [90, 140] |

Ours below vLLM's FP16 at every size is the point prediction. The same-head ratio against our own
dense arm falls under 1 at 4B, 0.9 [0.6, 1.2], where the L40S read 1.27.

## 5. What is published, whatever the numbers

The ten-arm table and the sweep of this card go into the paper beside the L40S ones, never
divided by them (hard rule 5). The served speeds go in per engine, never as a ratio across
engines. The paper's limitation "we did not run `Tetra` there" is replaced by what this run
measured.

## 6. What refutes what

- `Tetra` inside [6.5, 9.0] and above FP16's time: the speed result does not transfer to a
  high-bandwidth card, as for every earlier layout. The paper says the speed gain holds on the
  L40S and is lost on the A100, and the memory saving holds on both.
- `Tetra` under FP16's time: the decode got cheap enough to win even where bandwidth is plentiful,
  against F4's pattern. Read against the floor first: if nullk moved, the card or the image did.
- `Tetra` under AWQ's time: not expected, since the floor alone is already above AWQ on this card.
- Our served 4B above vLLM's FP16: the same conclusion as the second line, end to end.
