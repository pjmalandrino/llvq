# Deviations from preregistration-metal-nuit-2026-11-06.md

The prereg (sha256 `e8342893…`, stamped 2026-11-05 23:35) is not edited. Its departures are written here.

## É1. M2's arm order: `lut,lutd,D`, not `D,lut,<after>`

§3 step 5 names the confirmation run `metalrun time <sealed> <config> D,lut,<after> 256 7`. `metalrun` takes its first
arm as the reference of the bit gate and of every ratio, and refuses a dense reference by name. The run as written
loaded its three arms and stopped before any timed round: "Error: the reference arm of the bit gate is a fused arm"
(raw output kept beside the journal). No rate was read. M2 ran as `lut,lutd,D`: the same three arms, F0 as the
reference, which is what §5 asks M2 to confirm against. The arm order inside a round differs from the prereg's, and
the same order holds every round, as METHODE requires.

## É2. M3 ran before M2

The command chain that held M2 ran the off-protocol phased pass of `lutd` and then M3 after M2 stopped on É1. M2 ran
after M3, on the same quiet machine. §3 step 6 says M3 runs "after M2"; it ran before it. M3 decides nothing (§5), and
M2's reading does not depend on M3.

## É3. Header times of the M4 and M5 preregs

`preregistration-metal-nuit-m4-2026-11-06.md` says it was written "at 00:35"; its OpenTimestamps stamp was submitted
at 00:34. `preregistration-metal-nuit-m5-2026-11-06.md` says "at about 01:30"; it was stamped at 01:16 and its run
started at 01:16. Both headers state a time after the stamp; both stamps precede the first measurement of their run.
Nothing else in either file is affected.

## É4. M6 and the night's code after M1-M3

The M6 prereg tests code written after the night prereg was stamped (the Metal attention kernel, commit `dd9abe8`, and
`Served::apply_measurement`). It carries its own stamp; the night prereg's arms and verdicts are not reopened by it.

## É5. M6's NOT SERVED kernel config is the kernel from 1,280, not from 0

M6 §4: the band held (pooled +0.0051 %, every window inside ±0.12 %) and the kernel is a wash at 4,096
(x1.037 [0.996-1.093]), so the rule names the kernel form as the Metal candidate and asks for a NOT SERVED config for
it. A config on the eager engine cannot say "from position 0": that threshold comes only with `"decode":
"persistent"`, 4E, which has no Metal kernel. `configs/qwen3-4b-tetra-e4-metal-kernel-NOT-SERVED.json` therefore
names the kernel form with the eager threshold of 1,280. Below 1,280 it runs the grouped arithmetic, measured bit for
bit with the expanded form on Metal (M5); from 1,280 the kernel, whose band M6 read over positions 0 to 4,095. That
config was not timed as such. The grouped config is kept beside it.

## É6. M7's marginal costs came out negative: the instrument, not the machine

M7 ran on 2026-11-06 at 10:17 (P1 `lutd`, P2 `lut`), after a first attempt at 10:08 was called off before any run
because another application held the GPU at 79 % and `MTLCompilerService` at 77 % CPU. At 10:17 the GPU read 31 %
idle-state and other processes 30 % CPU; the operator's `chat` process was loaded and idle.

Gates 1 to 3 held: same 64 tokens, the full sequence (10.042 ms P1, 10.377 P2) shorter than the `blocks` phase
(15.951, 16.487), Tetra plus int4 bytes 1,158.3 MB against the loader's 1.16 GB. But every class's marginal cost was
negative, about -1.2 to -3.9 ms: the sequence ran slower with a class left out than with every class in. That is
impossible for the work and points at the instrument: the configs with a class left out freed their outputs every
other layer instead of every layer (the sink cleared past 12 entries), and every one of them followed a short config.
The marginal column and the shares built on it are therefore not used. What M7 does give, and is used: the remainder
of 5.9 to 6.1 ms (`blocks` minus the full sequence), and each matvec class's GB/s alone.

The instrument is fixed in the commit after M7's (outputs freed at every layer in every config, an untimed full pass
before each reading, three passes a reading, the config order turned by one each round) and rerun under M7b, stamped
before its run.

## É7. M7b's marginals are still negative: the class-removal split does not work on this machine

M7b ran at 12:15 with the fixed instrument. Gates 1 to 3 held; gate 4 failed (norm -3.31 ms, int4.down -1.58, sum of
marginals -5.38 ms against a sequence of 9.62). Taking a class out of the replay makes the rest slower, with outputs
freed every layer, a warm pass and the order turned, so the cause is not the sink. The likeliest one is the GPU clock
following the load: a lighter sequence lets it drop. Not tested. Per M7b's gate 4, the marginal column is not used;
the next lever was chosen on the remainder (6.3 ms) and on the alone GB/s, and then on M8's prefill reading.

## É8. M8 ran with other applications open

M8 §2 asks for a quiet machine. Builds were stopped 15 minutes before; Activity Monitor and Battle.net were open, and
other processes read 79 to 218 % CPU between S1's arms. The arms are interleaved round by round, so the ratios carry
the load on both sides; the absolute rates sit beside M2's (54.1 tok/s for `lutd` here, 52.1 at M2).

## É9. M9 ran at 13:01, after the M10 code was built

M9's binary was pinned at 12:40 and its run scheduled for 12:56. The int4 prefill kernel was then written and built
(last build 12:45:19), so the run was called off before it started and moved to 13:01, fifteen minutes after that
build. M9 ran its pinned binary, `a68aa617…`, as stamped; M10 followed in the same window.

## É10. M12 ran early, and its first pair carried the shader compile

The operator asked to start at once ("let's go launch now"), at 14:42, about two minutes after the last build and
before the prereg's 15 quiet minutes. The first pair is kept in the journal but not used: D1a was the first process
since the build, its first token took 4,398 ms of pipeline compilation, and that CPU time sits in D1a only, so the
difference understates the host time. The pair was rerun at once with warm shaders, and that second pair is the
reading. Its real-time difference (15.3 ms a token) matches the printed decode rate, which the first pair's did not.
