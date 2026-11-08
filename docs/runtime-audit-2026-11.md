# Runtime audit, 2026-11

The audit found eight bugs in `llvq-runtime` and fixed all eight. Each fix came with a test that failed on the code
before it. Each fix also faced a mutant, and the twelve mutants were all killed. The runtime now costs 1.9 µs a token on
the engine's thread at 7,936 tokens of answer, against 679 µs before. It takes 0.43 ms from the HTTP request to the
prefill (*measured*, [runtime-overhead](mesures/runtime-overhead-2026-11-07.txt)). The rest of a token's time is the
model's. The audit covers `llvq-runtime` and `llvq-llm/src/session.rs` at `529100a`. Nothing ran on a card.

## Bugs fixed

| bug | what a user saw | test that failed first |
|---|---|---|
| A job taken off the queue looked idle for an instant. | The eviction of `LLVQ_MAX_LOADED` could drop an engine as it started a job, and load a second model beside it. | `a_job_being_taken_is_never_seen_idle`: 666 idle sightings in 200,000 jobs |
| `llvq stop`, `keep_alive: 0`, `/api/delete` and a re-pulled manifest dropped a busy engine. | Its thread kept the model in memory, out of `/api/ps` and of the cap. The next request loaded a second copy. | `stopping_a_busy_model_keeps_one_copy_until_its_answer_ends`: `/api/ps` empty while the model answered |
| A request whose client left while it was queued still ran its template and prefix. | The cache of the live conversation was cut to that request's prompt. The next turn re-read 513 of 516 positions. | `a_request_abandoned_in_the_queue_leaves_the_cache_alone` |
| `total_duration` added the load to a wait that already held it. | The first answer reported 1.24 s for a 0.60 s load and a 1-token answer. | `the_first_answer_counts_its_load_once` |
| Two pulls of one file appended to one partial file. | One of two concurrent pulls failed. By the code, a pull still appending after the other's rename writes into the registered blob. | `two_pulls_of_one_file_at_once_register_the_file_whole` |
| The Hub download had no read timeout. | A connection that stopped sending held the pull forever. | `a_hub_that_stops_sending_fails_the_pull_instead_of_hanging_it`: still hanging after 10 s |
| `llvq pull --from` hard-linked the source into the store. | A re-seal written in place to the same path changed the registered blob after its check. | `an_imported_blob_does_not_change_with_its_source` |
| A NUL character in an answer broke the chat page's Markdown renderer. | The whole answer showed as an error. | `ops/llvq_chat_markdown_check.js` (Node, not in CI) |

The fixes, in the same order:

- The worker marks itself busy before it decrements the queue.
- A busy engine retires instead of being dropped: it takes no new model, counts against the cap, and the reaper drops it
  once idle. A new request for the same model takes it back, as Ollama does.
- The worker skips a job whose event channel is closed.
- `total_duration` is the time since the request, or the run plus the load, whichever is longer.
- `Store::pull` and `Store::import` hold an OS file lock per blob, `<home>/locks/sha256-<hex>.lock`. It covers two
  servers, two CLIs, or one of each. The in-process `pull_lock` is gone.
- The download has a 30 s connect timeout and a 60 s read timeout (`Store::read_timeout`).
- `import` copies the file, checks the copy, then renames it. `std::fs::copy` clones on APFS, btrfs and XFS, so the
  copy costs no space there.
- The renderer turns a NUL into U+FFFD before it marks code spans with NUL.

Most mutants revert one fix. The others remove one guard of the retire logic, of the import or of the text stream. Two
of them first survived and pointed at real gaps. The reaper and the worker both retired an engine, two mechanisms where one was needed; the worker's was
removed. The text stream's fallback hid a stream that had stopped being incremental; its test now bounds the window.

## Checked and found sound

- A client that leaves a streamed answer stops the engine within one token, or one 4E launch of 64 tokens. A client that
  leaves a non-streamed answer does too: hyper drops the handler when the socket closes. The new test
  `a_client_that_leaves_a_non_streamed_answer_stops_the_engine` keeps it so.
- Every path of `Worker::run_inner` ends in `Done` or `Error`, or in a closed channel. No stream is left open.
- The prefix kept is computed on ids, never on the memo. The memo only maps a message's role and text to the ids last
  rendered or generated for it. An edited message, a changed system message, think on or off, or a raw prompt can make
  the memo miss. It never puts wrong ids in the cache.
- Stop strings never cut inside a character, and a split character is held until it is whole.
- The page escapes model text before any markup. A `javascript:` link stays text, and a quote cannot leave an `href`.
- `Session::answer` feeds and lists the same ids on both decode paths, including the 4E launch that stops on a stop
  token.

## Found and not fixed

- **A client that stays connected and stops reading blocks the engine.** The engine blocks once the 64-event channel
  and the socket buffers are full, after a few thousand tokens. Every later request then queues behind it. `llvq run |
  less` left unscrolled is one way to get there. The fix is a send deadline that treats the client as gone.
- **A long prefill cannot be cancelled.** A client that leaves during it holds the engine until the prefill ends. The fix
  is a closed-channel check between the chunks of `Session::feed`, which needs a callback in `llvq-llm`.
- **`Session::feed` failing between two layers would leave the caches longer than `ids`.** `truncate(n)` returns early
  when `n` equals `ids.len()`, so the next feed would append after stale positions. No test reaches a mid-layer failure.
  After a failed feed, every cache could be cut back to `ids.len()`.
- **Control tokens written in a message are read as control tokens.** A user message holding `<|im_end|>` closes its turn.
  Ollama and `transformers` tokenize their templates the same way. Changing it would change the prompt `bin/chat`
  builds.
- **Five route tests depend on timing.** The tests are `a_client_that_leaves_stops_the_engine` and four of the new ones.
  Their margins are wide: 200 to 300 ms steps inside answers of about 1.5 s, and a waiting time of a quarter of the
  abandoned answer. A loaded runner could still flake them.

## Runtime cost

All numbers are *measured* on an Apple M3 Max shared with other sessions, with Qwen3's real tokenizer for the text
work. They come from `cargo run --release -p llvq-runtime --example overhead`, journal
[runtime-overhead](mesures/runtime-overhead-2026-11-07.txt).

| piece | before | after |
|---|---|---|
| text work a token, at 1,024 / 4,096 / 7,936 tokens of answer | 101 / 381 / 679 µs | 1.0 / 1.6 / 1.9 µs |
| the same with 3 stop strings | 107 / 384 / 717 µs | 2.3 / 6.2 / 10.5 µs |
| text work over a whole 8,192-token answer | 4.64 s | 32.1 ms |
| one send on the engine's thread | 2.7 µs | unchanged |
| one chunk through channel, JSON and HTTP | 2.7 µs a chunk, 18 µs latency | unchanged |
| request to the prefill, loaded idle model | 0.43 ms | unchanged |
| 20 messages of 8,314 tokens to ids | 5.8 ms cold, 19 µs from the memo | unchanged |

The text work decoded the whole answer at every token: about 85 ns a token already said. `TextStream` now decodes only the ids
since the last whole character. The text it gives is the whole-list decode, byte for byte, at every token. The
journal checks this on 8,192 tokens of docs and on 4,096 random ids.

The 0.43 ms before the prefill splits into 64 µs of HTTP floor, 85 µs to read the manifest, 15 µs of template and 3 µs
to cut the cache. The rest, about 0.26 ms, is the hand-off between the async runtime and the engine's thread.

What this was worth to the served rate is small, and it is *computed* from the table. Under 4E the text work runs once
per launch of 64 tokens, so 679 µs at 7,936 tokens was 11 µs a token, 0.2% of 4.5 ms (4B v9 at 221.5 tok/s,
[served-v9](mesures/served-v9-2026-11-07.txt)). On Metal's eager path it runs every token. At 27.7 tok/s near 2,100
positions (36 ms a token, study M1), the 180 µs it cost there was 0.5%. The text work and the stream flush are not why
the Metal chat slows down with the conversation: below 300 positions they cost at most 30 µs a token.

## Optimizations done

- **Incremental decode**, above: the text work at 7,936 tokens goes from 0.7 ms to 2 to 11 µs a token. The tokens and the
  text are unchanged.
- **The last chunk leaves before the cache is closed.** The engine wrote `<|im_end|>` and the newline into the cache,
  one model call of up to 3 tokens, before it sent `done`. It now sends `done` first, so a non-streamed answer and an
  OpenAI `[DONE]` arrive one call sooner. A request sent at once queues behind that call, which it would have waited
  for anyway. A failure there resets the cache instead of answering 500 after the answer.

## What remains, by estimated gain

Runtime:

1. **One cache for every conversation.** Two clients that alternate re-read their whole history at every turn. A few
   cached sessions keyed by prefix would save the whole prefill on each switch (*estimated*), at the memory of one KV
   cache each.
2. **Cancel between prefill chunks**, above. It frees the engine sooner, and the tokens stay the same.
3. **A send deadline**, above. A robustness fix, worth nothing in tok/s.

Below 1% each and not worth a change: the manifest read per request, the double JSON serialization of a chunk, and the
stop-string scan over the whole text.

Model and kernel, outside the runtime:

1. **Metal attention.** The chat falls from 55.6 to 27.7 tok/s by 2,100 positions under the expanded form (study M1).
2. **4E's launch of 64 tokens.** The client receives text in bursts of 64 tokens: every 0.29 s at 4B, 0.45 s at 8B,
   0.74 s at 14B (*computed* from [served-v9](mesures/served-v9-2026-11-07.txt), 142.5 and 86.3 tok/s). A stop string
   costs up to 63 tokens of compute past it, and a client that leaves is seen once a launch. A shorter first launch
   would show text sooner at little cost in rate (*estimated*), and it belongs in `served_decode.rs`.
