# Prereg chain-runtime-door: `llvq` serves the 4B by the chain prefill (DRAFT, 2026-11-15)

Status: DRAFT, not timestamped. The lead rereads it, signs the predictions below and stamps it before any run.
Cost: about $0.24 (*estimated*), $0.47 if the module cache misses; hard cap 25 min, $0.75.
Measured code: branch `dev/trois-gains-2026-11-15`, commit to fill at the stamp. Job script
`ops/jobs/chain-runtime-door.sh`. The image must be rebuilt from that commit.

A timestamped prereg is no longer edited. A deviation is written in `<name>-ECARTS.md`.

## Question

Does `llvq serve`, with no variable set, load the chain config for the 4B on CUDA and answer through `/api/chat`
with the texts the eager prefill gives, sooner on a long prompt; does `LLVQ_PROFILE=eager` load the v9 config; are
other profiles refused?
The chain was measured at `fusedrun`'s door and in `gsm8k` (`docs/mesures/chain-prefill-served-2026-11-15.txt`).
`llvq` itself has never run it on a card. This gates the merge of the serving decision.

## Setup

One L40S, the sealed 4B imported by `llvq pull qwen3:4b --from <sealed>`. The module cache is a copy of the last
job's cubins (`chain-prefill-served-2026-11-15/modcache/`), `LLVQ_MODULE_CACHE` pointing at it.

```
(1) llvq serve; POST /api/generate {model, stream: false}; GET /api/ps          # load check
(2) door_client.py: /api/chat streamed, temperature 0, think false, num_predict 256, thinking off;
    the 8,191-token prompt (spec-long-8k-v1) and the first house prompt, alternated, 1 discarded + 3 timed rounds
(3) LLVQ_PROFILE=eager llvq serve; the same requests
(4) LLVQ_PROFILE=bogus, then chain: the load through the API
```

Alternating the two prompts keeps the conversation cache from reusing the long prompt: each request reuses only the
chat template's head, which `prompt_eval_cached_count` shows.

## Controls

1. oracle MATCH; the sealed file and the two prompt sets at their sha256; no `LLVQ_*` in the job environment.
2. Load check, or the job stops: `/api/ps` names `qwen3-4b-tetra-e4-4e-v9-chain.json`; the server's log names it and
   the served 4E v9 module (252,757 bytes, 1b422b62).
3. The log says whether the two tree modules came from the cache. If the key misses (another driver or card), the
   modules compile, the load takes about 509 s, and the job goes on: the timing of requests is unaffected.
4. Texts identical across rounds within each server.

## What gets published, and what does not get compared

Published: the config each server loaded; load time each; for each prompt and server the time to the first content
chunk on the client, median and range over 3 rounds, and the server's `prompt_eval_duration`; whether the chain's
texts equal the eager prefill's, and where they part if not; the two refusals. Not compared: these times with
`fusedrun`'s, which time no HTTP, tokenizer or streaming.

## Decision rule

| reading | verdict |
|---|---|
| controls hold; the way back loads v9; both refusals name `LLVQ_PROFILE`; long prompt first content, chain over eager, <= 0.30; texts identical for both prompts | PASSES: the merge may go ahead |
| a control fails, or the way back does not load v9, or a profile is not refused | FAILS: fix before the merge |
| controls hold, a text parts | NOT SETTLED: reported with where it parts; the band of §4 of ETAT is the operator's licence, the merge is his call |
| otherwise | not settled, operator decision |

## Signed prediction

To be signed by the lead before the stamp.

- Load with the cache: ...
- First content, long prompt, chain and eager: ...
- Texts identical: ...

## Reading table (lead's, filled after the run)

| quantity | predicted | measured | inside |
|---|---|---|---|
| | | | |

## Cost, from measured times (*estimated*)

oracle about 60 s; import of 1.42 GB and the cache copy about 20 s; two loads of 55.2 s (chain-prefill-served,
cached) and two refused loads of a few seconds; the chain's requests 4 x (6.74 + 256 / 160.5) s on the long prompt
plus 4 x about 1.2 s on the short one, about 40 s; the eager prefill's 4 x (37.22 + 256 / 155.5) s plus the short
ones, about 160 s; servers' starts and stops about 60 s. About 480 s, $0.24. A miss adds about 454 s (509.2 - 55.2),
$0.47 in all.
