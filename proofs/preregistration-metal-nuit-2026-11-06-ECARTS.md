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
