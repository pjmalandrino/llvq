# Deviations from the prereg of the prefix cache on the Mac (2026-11-26)

Each line says what differs from `preregistration-inkling-mac-prefix-2026-11-26.md` and why.

- E1. The file's integrity digest is the sha256 of the list of the sha256 of each 64 MiB chunk, not one sha256 of
  the whole file. The first real reading (commit d7d903d) spent 3.17 s of a 3.49 s load and most of a 3.81 s save in
  one sha256 stream: the `sha2` crate of the lock file runs its portable code on this Mac, about 445 MB/s, and its
  hardware backend needs a crate not on this machine (no network). The chunks are hashed on six threads. Every byte
  is still covered, in order (`the_digest_covers_every_chunk_in_order`).
- E2. A loaded slot's key and value buffers are made in one device allocation each, from a host buffer with zeros
  past the held rows, instead of an upload of the held rows into zeroed buffers. The first gate run read 0.115 GB more
  Metal allocation after the prompt than `b9000` (10.82 against 10.71 GB). The numbers of this lot are those of the
  second code (commit 4d74299) unless a line says otherwise.
- E3. The file is 1.41 GB, not 0.4 to 0.8 GB: the keys and values are held in f32 on this stack (8,192 bytes a
  position and a slot), and the bounded sliding slots hold between 2,048 and 3,072 positions after a piece. F1 is
  read as written, outside its range.
- E4. A fifth mutant was added beyond the four of the prereg: the loaded buffers sized to the rows they hold, not to
  the saved capacity (M5). Run once with the test's byte round trip removed (M5b) to see whether the bit comparison
  alone catches it.
- E5. P-a is also compared with the answer of the server without the cache, as P-b and P-c are.
- E6. The 13 checks of `ops/inkling/serve-mac.sh` were run twice: on d7d903d, then on the final code.
- E7. The P-a check of the first server run read the prefix directory at the end of the script, after P-e had
  written a second file, and said FAIL on "one file". The listing taken right after P-a (`pa.prefix-ls.txt`) holds
  one file, `-00008704-`; the check now reads that listing, and was run again on the recorded files, with no new
  measurement (`prefix/recheck-pa.txt`; the first summary is kept as `prefix/summary-run.txt`).
- E8. The answers of the 48-token turns end inside the reasoning block, so the same-answer checks compare the
  reasoning as well as the text and the calls. The edit was made while the servers ran, before any check was read.
