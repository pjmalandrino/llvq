# Preregistration. The prefill's card memory after an 8,192-token prompt: the exact fixes and the kernel prefill

**Written on 2026-11-08.** Stamped with OpenTimestamps before the first measurement, then never edited; a deviation
goes beside it in `proofs/preregistration-pic-vram-8k-2026-11-08-ECARTS.md`. **Not stamped and not launched at the
time of writing: nothing runs before the operator's go.**

Cost: **$1.75 expected, $2.40 at worst** (80 min timeout) on l40sx1 (*estimated* from the duel's Tetra job, 29 billed
minutes for six sweeps, `docs/data/jobs.csv`). Registry total on this branch's base: $280.04 over 278 rows
(*computed*); $297.59 over 291 rows once the duel's PR is merged.

**Base commit:** the commit of `claude/pic-vram-8k-1835fd` that carries this file and `ops/jobs/pic-vram-8k.sh`.
**Image:** the Space `Pier-Jean/llvq-runner-cuda` republished from that commit, passed as `IMAGE_SHA`. The Space
carries another session's branch at the time of writing: check `hf jobs ps` before republishing it.

## 1. Question

The duel of 2026-11-08 measured 11.48 / 12.66 / 17.09 GB of peak card memory after an 8,192-token prompt at 4B / 8B /
14B under the served v9 configs, against 3.52 / 4.71 / 6.90 for llama.cpp IQ2_S (*measured*,
`docs/mesures/duel-iq2-2026-11-08.txt`). `docs/mesures/pic-vram-8k-2026-11-08.txt` attributes the excess to the
grouped prefill's score tensors and to the stream-ordered pool, and the branch changes two things:

1. **Exact**, in the served path from this commit: the masked softmax written back in place in blocks of 1,024 rows
   (one score tensor alive instead of three), the causal mask built in f16 on the host, a stream sync before each
   prefill chunk and after each cache growth. The claim: the same logits bit for bit, a lower peak.
2. **Not served**: `"prefill_attention": "kernel"` sends the prompt's rows to the decode-attention kernel, 8 a call,
   and makes no score tensor. The claim: a peak near llama.cpp's, and logits that do not depend on the chunking.

Does the card agree, and by how much does the peak fall?

## 2. Arms

One process each, the duel's sampler (`nvidia-smi --query-gpu=memory.used -lms 100`, peak minus idle), `fusedrun
<sealed> 256` with `LLVQ_CONTEXTS=8192`, the duel's sealed files at their sha256.

| arm | config | sizes |
|---|---|---|
| served | `configs/qwen3-<size>-tetra-e4-4e-v9.json` | 4B, 8B, 14B |
| whole | the same, with `LLVQ_SOFTMAX_WHOLE=1` (the whole-tensor softmax, nothing else) | 4B |
| kprefill | `configs/qwen3-<size>-tetra-e4-4e-v9-kprefill.json` | 4B, 8B, 14B |

Each sweep prints the time to first token, the decode speed (median of 5), the 4E-against-eager check, a fingerprint
of the prompt's last-position f32 logits, and under kprefill the same logits in chunks of 512 against chunks of 4,096.

## 3. Predictions (signed before the run)

Peak minus idle, GB (*computed* in `ops/pic_vram_8k.py`; the pool's behaviour is *estimated*, so the band is +-0.5 GB).
The kprefill peak sits at the chunk's MLP (gate, up, act and product alive, 0.38 / 0.50 / 0.70 GB), not at the
attention. The bug audit's recount, 3.91 / 5.47 / 8.14, held the attention's locals and the last chunk's hidden
states alive under the MLP, which the code did until this branch frees them early; with them freed, the count
below is the code's:

| arm | 4B | 8B | 14B |
|---|---|---|---|
| served | 5.65 | 7.09 | 10.08 |
| whole | 7.8 (two score tensors since the lifetimes fix, plus the syncs) | - | - |
| kprefill | 3.67 | 5.20 | 7.81 |

- served: last-position logits fingerprint equal to the whole arm's at 4B; time to first token within +5 % of the
  duel's 34.48 / 45.98 / 80.31 s (the in-place pass and the syncs, *estimated* +0.2 to +0.6 s).
- kprefill: chunks of 512 and of 4,096 bit for bit; fingerprint different from served (the forms are not bit for bit);
  time to first token between 0.9 and 1.1 times served's (*estimated*: the kernel's attention against cuBLAS's, 2.2 to
  2.55 s of 34.5 at 4B, `docs/archive/debit-vague5-2026-11-02/prefill.md`).

## 4. Gates and decisions

- **G1, the exact claim.** 4B served and whole print the same logits fingerprint, and both print 256 of 256 4E tokens
  equal to the eager path's. If they differ, the in-place softmax is not bit for bit on the card: the branch reverts
  it before merging, whatever the memory.
- **G2, the memory.** served inside its band at each size. Outside it, the decomposition is wrong somewhere; the
  journal says where before anything else is claimed.
- **G3, the kernel prefill.** kprefill inside its band, and chunks of 512 bit for bit with chunks of 4,096. If both
  hold, its prompt rows are its decode rows, whose attention form passed the +-0.12 % band by decode
  (`docs/ETAT.md` §4), and serving it is put to the operator with its time to first token. If the chunkings differ,
  serving it needs a paired perplexity run of its own, costed then.
- Nothing becomes served by this run. The operator decides on the kprefill configs after reading it.

## 5. What this run does not measure

Quality (no perplexity, MMLU or GSM8K: the exact arm moves no bit, the kernel arm is G3's question), batch, any
context but 8,192, `llvq serve` itself (it prefills through the same `hidden_cached` in chunks of 4,096, so the same
peak is expected, *computed* from the code path), Metal.
