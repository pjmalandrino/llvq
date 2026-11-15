# Deviations from the prereg of R3 (KV ladder)

Beside `preregistration-recherche-debit-kv-ladder-2026-11-14.md` (sha256 8913d843, stamped, not edited). Written on
2026-11-14, before the second attempt, and stamped before it.

## 1. Why a second attempt

The four jobs ran once (6af8a005, 6af8a007, 6af8a009, 6af8a00b) and are void: each died in its load check on a 404.
The job script pinned WikiText-2 at `740312add88f781978c0658806c59bc2815b9866`. That is the commit of
`openai/gsm8k`, not of `Salesforce/wikitext`. The main session copied it from `ops/jobs/attn-t8-r2.sh`, where it pins
GSM8K, and wrote it into §4 of the prereg. No window was scored and no arm has a verdict.

## 2. What changes

| prereg | second attempt |
|---|---|
| §4: WikiText-2 test at revision 740312ad | at `b08601e04326c79dfdd32d625aee71d232d685c3`, the commit of `Salesforce/wikitext` (main on 2026-11-14, the snapshot named in tetra-schur-pilot-2026-09-14.txt) |
| §4: output `/out/recherche-debit-kv-ladder-2026-11-14/<arm>` | `/out/recherche-debit-kv-ladder-2026-11-14/<arm>-r2` |
| cap: four 60 min timeouts, $7.20 | four 57 min timeouts, $6.84, so that the void attempt and the second stay under $7.20 |

`ops/jobs/recherche-debit-kv-capture.sh` carries the same wrong commit and is not run; it is not corrected here.

## 3. What does not change

The arms, the image (2990f36c, commit eb54ac5: the script alone changed), the sealed file, the flags, the windows
(0 to 11 of 8,192), the reading of §5 and the predictions of §7. The token fingerprint the run prints is the record
of what was scored.
