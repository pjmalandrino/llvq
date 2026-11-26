# Prereg: a prefix cache on disk in the `llvq` server, Inkling 2 Mini on the Mac (2026-11-26)

Written before the first measurement of this lot. Not edited after its timestamp; deviations go in `-ECARTS.md`
beside it.

## Question

On this Mac (M3 Max, 64 GB, Metal) the server gives the first token of an 8,791-token prompt after 145 to 169 s
(*measured*, `docs/mesures/inkling-mac-serve-2026-11-26.txt` and `docs/mesures/inkling-mac-decode-2026-11-26.txt`).
The server already reuses a conversation in progress. A new conversation with the same system prompt and tools and
another first message matches no held segment, because the first segment holds the system prompt, the tools and the
first message together, so it is read again in full; so is any prompt after a restart, and after a rollback the
bounded sliding slots refuse. Plan item P7: save the cache after the static block to disk, and load it when a prompt
starts with the same tokens. How much of the first token's wait does it remove, at what disk and memory cost, and
does it keep every logit bit?

## Design (decided by the operator)

1. Save point. `L` is the end of the static block: what the template renders before the first message that is not a
   system message (the system prompt and the tools). It is found as the longest common prefix of the rendered ids of
   the request and of the ids rendered from its leading system messages alone (same tools, same effort). `b` is the
   largest multiple of the prompt piece (512 tokens, `PROMPT_PIECE`) at most `L`. The cache is saved at `b`, not at
   `L`: the cuts of the read stay those of today's read, so no logit changes. Up to 511 tokens are read again after a
   load. Nothing is saved when `b` is under 1,024 tokens.
2. Save. During the first read, at `b`: the whole cache state to disk. Per slot: keys and values of positions
   `base..len` in their stored dtype (raw bytes), `len`, `base`, the buffer capacity, the bound, the four convolution
   states (read after the rings are handed back, as `mark` does). Plus the tokens `0..b`. Atomic write (temporary file
   then rename), mode 0600, in `<store>/prefix/`.
3. Key and header. Format version, model digest (the local file's sha256, as in the manifest), sha256 of the running
   executable, stack and the settings that change bits (piece, score rows, slot slack, backend, where the bias is
   built), sha256 of the tokens `0..b`, sha256 of the whole file. Any mismatch or damaged file: refused by name in the
   server log, then the normal read. No panic, never a half-loaded state.
4. Load. Before reading a prompt from position `from`: the longest snapshot whose tokens are a prefix of the prompt
   and whose `b` exceeds what the cache in memory keeps (also when the rollback is refused). If there is one: cache
   emptied, state loaded, one held segment for `0..b` with its mark, then the rest is read. Later segments live as
   today. The `cached` count of the answer includes these tokens.
5. Disk. A byte cap, 4 GiB by default, least recently used evicted first. `llvq serve --prefix-cache on|off` and
   `--prefix-cache-max <size>`. On by default on Metal only; off by default on CUDA and the CPU, where it compiles and
   can be turned on (CUDA) or is refused by name (no long-prompt settings).
6. The `.llvq` format and the indexing are not touched.

## Gates

- Same bits, small model. A test in `llvq-llm` on the tiny model: read `0..b`, save, load into a new cache, read the
  rest; every logit equal byte for byte to the read without the cut, with bounded sliding slots, and on Metal with the
  fused glue and attention whose rings hold the states.
- Same bits, real model. `inkrun` on the 9,000-token prompt of lot 1, saved and loaded at `b` = 8,704: logits sha256
  `0140733aceb0789513f4c76dc922b6401b2f919c96f11d018adafb492fcdf226` (reference `b9000` of the earlier lots).
- Mutants, each seen red with its site counted: a different token accepted by the key comparison; the integrity check
  skipped; the save point moved by one piece; a setting removed from the key.
- Server. The 13 checks of `ops/inkling/serve-mac.sh` still pass. New checks on a request shaped like a coding agent:
  a system prompt and tools of about 8,800 tokens built deterministically, then a short user message.
  - P-a: cold server, first request: full read, file written.
  - P-b: server restarted, same static block, another user message: `cached` at least `b`, and the answer at
    temperature 0 is the text a server without the cache gives for the same request.
  - P-c: same server, a new conversation with a third message: same reading, no restart.
  - P-d: file damaged by one byte: refused by name, full read, same answer.
  - P-e: system prompt changed by one word: no load.
  - P-f: cache off by the switch: today's behaviour.

## Measurements

First token and load time in 3 repetitions of P-b, median and range. File size, save time, footprint of the server
process (`footprint`, as `serve-mac.sh`). Each `.machine` line carries load, GPU and `pmset -g therm`. The cold reads
this lot needs anyway alternate between `CANDLE_METAL_COMPUTE_PER_BUFFER=50` and the lot 3 default (30); their table
is reported and the default is not changed here.

## Signed predictions (the coordinating session, Claude Fable 5.2, 2026-11-26; copied as given)

"- F1 : fichier de 0,4 à 0,8 Go pour `b` = 8 704 (*calculé* : 4 096 octets par position et par créneau, 8 créneaux
pleins, 40 glissants à 2 048 positions au plus, soit 0,45 à 0,62 Go).
- F2 : premier jeton en P-b de 3 à 12 s, contre 145 à 169 s ; probabilité 0,6 d'être sous 8 s.
- F3 : la sauvegarde ajoute moins de 3 s à la première lecture.
- F4 : en P-c, même fourchette que F2.
- F5 : pic Metal inchangé à ±0,1 Go ; l'empreinte du processus peut monter le temps de l'écriture, mesure-la."

## Rules

One heavy job at a time under `~/.llvq-guard/heavy.sh`, load under 6 before each launch. The store of every test
server is a work directory under the raw folder (`LLVQ_HOME`), never `~/.llvq`. No paid run, no network call other
than `ots stamp`.

## Cost

$0. Local only, on the operator's Mac, which stays in use.
