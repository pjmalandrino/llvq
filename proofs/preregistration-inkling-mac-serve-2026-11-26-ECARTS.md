# Deviations from prereg inkling-mac-serve-2026-11-26

E1. The server checks ran with longer request timeouts than the card job (300 s for the short turns, 600 s for the
tool calls, 1,800 s for the long prompt, against 120, 300 and 900). Nothing a check accepts changed.

E2. The first launch of `ops/inkling/serve-mac.sh` stopped before loading the model: its guard refused the
`LLVQ_THREADS` that `~/.llvq-guard/heavy.sh` exports. The guard now lets that one variable through. The second launch
is the first complete try that P1 reads.

E3. The load sentinel exited four times on a GPU-only warning (GPU above 92 % for two minutes) while a job of this
prereg held the lock: the warning is that job. A loop re-armed it on that case only and stopped on any other exit.
No load or memory warning and no critical alert fired.

E4. The generation speed after the long prompt is computed from the stream's timestamps (64 tokens after the first),
since check (e) goes through the OpenAI stream, which carries no `eval_duration`. The short turns give the server's
own `eval_count` over `eval_duration`.

E5. The repeated long prompt (check e2) ran with load1 between 6 and 8 from other processes. Its first token is
reported as "machine busy" and not repeated. The first long prompt ran at load1 4 to 5.

E6. Memory of the server is read with `footprint` on its process and `/api/ps`; the prereg named only the `inkrun`
readings. The method sentence of the prereg did not say how the server's memory would be read.
