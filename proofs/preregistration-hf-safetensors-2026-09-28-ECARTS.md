# Deviations from the stage 0 prereg of 2026-09-28

The prereg is `proofs/preregistration-hf-safetensors-2026-09-28.md`, sha256
`cce6728c34d7ba7a`, timestamped before the conversion and never edited. The journal is
`docs/mesures/hf-safetensors-4b-2026-09-28.txt`.

Three departures, all found while writing the code, none after seeing a result.

## 1. Control 4 cannot hold for `config.json`, and was split

§6 control 4 reads: "`config.json` and `tokenizer.json` come out byte for byte identical to
the blobs." That is impossible for `config.json` by stage 0's own definition, which puts a
`quantization_config` block in it. The prereg contradicts itself on that line.

What is checked instead:

- `tokenizer.json`, byte for byte against the blob's SHA-256. Passes.
- `config.json`, key for key against the sealed blob **parsed**, with `quantization_config`
  required to be the only added key. Passes over 26 keys.

Parsed values and not text, so no number's formatting enters the claim: `1e-06` and `1e-6`
are the same float on both sides. `llvq-digest.json` carries the sealed `config.json` under
`config_base` for that comparison, and marks each blob `written_verbatim` true or false so a
reader cannot compare the wrong pair.

## 2. Two digests more than §5 lists, on purpose

§5 lists one digest for the codes, over "the code stream verbatim". The implementation also
digests `indices`, as u64 little-endian, and `gains`, as u32 little-endian, both derived by
the reader from the packed bytes.

Reason: a digest over the payload bytes alone proves that a byte array survived a copy, which
is not what the gate claims. The reader now has to unpack 48-bit words MSB-first to pass,
which is the format decision of §3 being checked rather than asserted. A mutant that assembles
the words little-endian fails on `indices` and passes on `codes`.

This is strictly more checking than was preregistered, and it is recorded here because it is
still a departure from the file.

## 3. One signed prediction missed

§8 predicts `quantization_config` between 20 and 60 KB. Measured: `config.json` is 131,296 B,
essentially all of it that block. The miss is 2.2 times the top of the interval.

The cause is pretty printing, one key per line, about 500 B a record where the estimate assumed
a compact 150 B. The three other predictions of §8 held: no dtype failure, 1.422 GB of
directory inside [1.40 ; 1.45], 96 rotations against "about 100".

Nothing in the decision rule of §7 reads this number, so the verdict is unaffected. Whether the
block is compacted, or the record table moves to a side file, is an open decision for stage 1.
