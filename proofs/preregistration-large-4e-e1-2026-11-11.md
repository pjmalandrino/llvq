# Prereg E1 of the large-4E budget: where 4E v9 asks L2 for later rows, sealed 4B

Written on 2026-11-11 on branch `claude/arbre-nuit-2026-11-10`, once the code it describes existed. Timestamped before
the first measurement, then never edited; a departure goes to `preregistration-large-4e-e1-2026-11-11-ECARTS.md`.

Go: the operator, card spending inside a **$15** cap of which **$10.31** is spent; relayed by the orchestrating
session. Registry before this job: **$317.31** (`docs/data/jobs.csv`). Cost: about **$0.54** on l40sx1 (*estimated*,
"Cost"), timeout 29 min, **$0.87** at worst; $11.18 of $15 at worst after it.

Measured code: the commit that holds this file. The image is published from a clean detached checkout of the commit
that holds this file's stamp.

## Question

`docs/mesures/large-4e-budget-2026-11-11.txt` §1b reads (*estimated*, not proven) that 4E v9's attention phase takes
791 us at a short context, and a third of its bytes at 8,192, because every CTA asks L2 for its next o and gate/up
rows during it (`prefetch_next` in `pers_4e.cu`, 670 MB a token, *computed*). Does removing that prefetch shorten the
attention phase, and does moving the prefetch to the gate/up phase raise tok/s at the bit?

## The arms (pers.rs::p4e_e1_edits, applied as text; the served v9 module does not move)

The served v9 module's text is pinned (252,757 B, sha256 1b422b62, `the_4e_v9_module_is_the_one_the_doors_gate_on`)
and no config changes. Three bench modules are built from it by text edits, named in `LLVQ_PERS_4E=64:<arms>`:

- **e1:v9**, the reference: v9 plus a logits symbol (`pers_4e_body` gains a fourth template parameter, false by
  default, so the token and stamp symbols keep their text; the logits symbols set it and write the head's binary16
  logits through `pers_head_argmax`'s existing `logits` pointer, at the launch's stamp pointer).
- **e1:v9np**: e1:v9 with the two `prefetch_next()` calls of the attention phase removed. Nothing else. Per token it
  asks L2 for 670.3 MB less (o 200.5, gate/up 469.8, *computed*); the q/k/v prefetch before ln1 (173.8 MB) and the
  down prefetch before the inner wait (315.4 MB) stay.
- **e1:v9mv**: e1:v9np, plus: during the barrier before layer l, one thread of each CTA (thread 32) computes that
  CTA's v9 shares of layer l + 1's q, k, v and o rows into header u32 10..25 of its shared memory; right after ln2 of
  layer l (pers_segment_body, at the start of the gate/up rows) every thread of the CTA asks L2 for them. The q/k/v
  prefetch before ln1 runs at layer 0 only, with layer 0's o share beside it. Gate/up rows are not prefetched. Bytes
  asked of L2 a token: 374.3 MB (q/k/v 173.8, o 200.5) against v9's 844.1 (*computed*), plus the unchanged down.

Departures from the note's E1 build, decided from the code: the moved prefetch is issued at the start of the gate/up
rows, after ln2, not after the first gate/up pair (`start(k)` runs on lane 0 only, so a per-pair hook would issue
the prefetch from one lane a warp); the KV loads get no evict-first hint (that is not a prefetch statement and would
change attention's load instructions); the note's arm v9p0 is e1:v9np here.

Host gates passed before this file (the commit's tests): the three arms' host twin equals the served chain on v9's
commands (`the_4e_e1_arms_are_the_served_chain`, prefetch is nothing on the host, so this covers mv's header slots,
its staging and its layer-0 branch); a mutant that moves mv's slots onto the v7 offset is killed; a structural check
(`p4e_e1_structure`) holds that, past the lines that name a prefetch or carry the `e1-pf` tag and assign nothing, and
past mv's helper block (no kernel, atomic, barrier or grid wait, stores to its slots only), each arm's module is the
reference's line for line, and kills seven mutants; `cargo clippy` for the Linux target, zero warnings.

## Setup

One image, one job, `ops/jobs/large-4e-e1.sh`, l40sx1 only (no twin). Output `large-4e-e1-2026-11-11/` in
`hf://buckets/Pier-Jean/jobs-artifacts/`. The sealed 4B (sha256 886391a8), `configs/qwen3-4b-tetra-e4-4e-v9.json`.
Each step its own process, non-fatal for the next; the 100 ms nvidia-smi sampler of `ops/jobs/duel-iq2.sh` around
each.

0. Controls: `oracle` on Qwen3-0.6B; the file and the config at their sha256; no `LLVQ_*` in the job environment.
1. The served v9 config at the door: `LLVQ_CONFIG=<v9> LLVQ_CONTEXTS=5 fusedrun <sealed> 64`.
2. `LLVQ_PERS_4E=64:v9,v9np,v9mv fusedrun <sealed> 64` with the served flags on the command line (layout tetra48,
   embed q4, rot_share 1, fuse 0), as step3-4e: at a 5-token prompt (measurement form, the kernel on every row) and
   at 8,192 (the served config's form), the eager path's 64 tokens, each arm's against them, 2 + 5 rounds with the
   eager path and every arm in every round, medians and ranges formed round by round, then one stamped launch an arm
   (phase walls: attention, o, gate/up, q/k/v and the rest), then one logits launch an arm (64 positions x 151,936
   logits, compared with e1:v9's bit for bit). The int4 words are relaid on the card as in step3-4e (measurement
   form, the int4 bytes twice).
3. `LLVQ_PERS_4E=64:<arm>` for v9, v9np and v9mv, one process each: the peak VRAM of each arm by the sampler, and a
   second logits fingerprint per arm and context, compared across processes.

## Controls

Each voids the run: oracle MATCH; the inputs at their sha256; no `LLVQ_*` in the job env; the served door's 4E module
is the served v9 module (252,757 B, sha256 1b422b62); each arm's tokens equal the eager path's at both contexts (an arm
that diverges leaves the run and its numbers are void, the bench's own rule). If the image fails to build, nothing is
launched.

## Signed prediction

Signed by the operator, relayed verbatim by the orchestrating session:

- logits bit for bit equal to v9 on both arms at both contexts; tokens equal; peak VRAM unchanged within 0.02 GB.
- v9 in this job: 226 tok/s [220, 232] at 5 tokens, 158 [152, 165] at 8,192.
- arm np: attention phase drops from about 790 µs to 600 µs [540, 700] at a short context; token time changes by -0.08 ms [-0.20, +0.05]; 230 tok/s [222, 237]; at 8,192: 165 tok/s [156, 176].
- arm mv: token time -0.14 ms [-0.29, -0.03] at a short context, 233 tok/s [226, 241]; at 8,192: 171 tok/s [162, 186].

Readings each line is compared with: tok/s and token times are the bench's (step 2) medians of e1:v9, e1:v9np and
e1:v9mv; the attention phase is the stamped launch's attention wall; the peak VRAM is step 3's three processes. The
door's v9 (step 1) is printed beside, at 5 tokens only. Note, not a prediction: step3-4e read the bench's v9 at 238.6
and 163.5 tok/s, and the door read 227.7 and 156.4 (spec-mem-m3).

## Decision rule

| result | outcome |
|---|---|
| an arm's logits bit-identical to e1:v9 at both contexts and its median tok/s above e1:v9's round by round (ratio range above 1.000) at both contexts | the arm PASSES |
| an arm PASSES with a gain of at least 2 % at 5 tokens or 5 % at 8,192 (median ratio round by round) | worth serving; the operator decides |
| any logit differs | the arm is a bug, INVALID; the journal leads with it |
| e1:v9np's stamped attention wall at 5 tokens not below e1:v9's | the note's reading (the prefetch makes the attention wall) is refuted, written as such |

## What gets published, and what is not established

Published: the module lines (digests, registers, local bytes), the rates and ratios, the stamp tables, the logits
lines, the sampler files and peaks. Not established: a served rate for the arms (no config names them); 8B, 14B;
another card; L2 residency of the prefetched lines (no counter is read).

## Cost

step3-4e loaded six bench modules in 389 s and ran both contexts in 8.4 min (*measured*). Here: the door (one module,
about 2 min), the three-arm process (about 5 min), three one-arm processes (about 2.5 min each), oracle and start:
about 18 min, **$0.54** at $1.80/h (*estimated*). Timeout 29 min, **$0.87**.
