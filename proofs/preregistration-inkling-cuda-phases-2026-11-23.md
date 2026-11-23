# Prereg: Inkling 2 Mini on CUDA, where a decode step's time goes (2026-11-23)

Written before the run. Not edited after its timestamp; deviations go in `-ECARTS.md` beside it.

## Question

After the four-launch experts (0.1093 s a step on an L4, `inkling-cuda-moe4-2026-11-23.txt`), which part of a decode
step holds the time: the attention, the glue (norms, block attention, gates, convolutions, residuals), the routing,
the experts, the head? The answer orders the next two ports, the glue kernels and the attention kernels.

## Object and method

The object of `preregistration-inkling-cuda-moe4-2026-11-23.md`, arm `four`. `inkrun --phases 12`: the first 12
decode steps are split by part with the device drained at every boundary, so those steps run slower; the 19 steps
after them run untimed. Job `ops/jobs/inkling-cuda-phases.sh`, `l4x1`, the image of the moe4 run.

## Reading rule

The split is read as shares of the profiled steps' sum of parts, and trusted only if the untimed median step is
within 0.100 to 0.120 s (the moe4 run's figure) and the profiled total is under twice the untimed one. No verdict:
this is a measurement that orders work.

## Signed prediction (the coordinator, before the run)

| part | share predicted |
|---|---|
| glue and routing together | 40 to 60 % |
| attention | 20 to 35 % |
| experts | 10 to 20 % |
| head and the rest | 5 to 15 % |

The coordinator's prediction of the moe4 step was wrong (the experts held one third of a step, not two thirds), so
these shares are held loosely.

## Cost (*estimated*)

`l4x1` at $0.80/h, about 4 minutes: $0.05, $0.27 at the 20-minute timeout. Card spend before the run: $2.87 of $25.
