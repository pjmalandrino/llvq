# Deviations from preregistration-llvq-cpu-install-2026-11-25.md

## E1, 2026-11-25: a second run on an archive with the AVX2 kernel of b4a6684

The first run (6b0615aad0582cfbbefd2fae) passed its criterion and read the 4B at 0.5 to 0.6 tok/s. Two later jobs
showed that its instance was an outlier (the same commit ran seven times slower there than on another instance of
the same flavor) and that the AVX2 Tetra kernel had a factor of 1.73 left at 8 threads
(`docs/mesures/llvq-cpu-x86-2026-11-25.txt`). The same job is run again, `RETRY=2`, on the archive the workflow
`llvq archives` built from 532c56e (run 45525372204). Nothing else changes: same script, same criterion.

Signed before the run: the 4B decodes at 2.5 to 5 tok/s on 8 vCPU if the instance is like that of job
6b063689d0582cfbbefd5aa4 (182 ms of matvecs a token there, and the token has more in it than matvecs).
Cost: under $0.02, *estimated*.
