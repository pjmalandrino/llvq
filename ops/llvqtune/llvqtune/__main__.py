"""Wire the parts and run. The only file that knows about all four layers.

```text
uv run --project ops/llvqtune -m llvqtune \
  --student ~/qwen3-4b-export --teacher Qwen/Qwen3-4B \
  --mode row_scales --objective kl --steps 200 --out /tmp/sigma.json
```

`--student` is the directory `llvq-llm --bin export` wrote. `--dry-run` wires
everything, prints the rate the run would cost, and stops before the first
batch. Hard rule 1 asks for the cost up front, so that is the default way to
look at a new configuration.
"""

from __future__ import annotations

import argparse
import sys


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="llvqtune", description=__doc__)
    p.add_argument("--student", required=True, help="exported f16 checkpoint")
    p.add_argument("--teacher", default=None, help="dense reference, for kl")
    p.add_argument("--mode", default="row_scales", help="training mode")
    p.add_argument("--objective", default="kl", choices=("kl", "ce"))
    p.add_argument("--rank", type=int, default=16, help="low_rank only")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint-every", type=int, default=0,
                   help="steps between partial writes of the export")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="auto", choices=("auto", "f32", "f16", "bf16"))
    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument("--commute", action="store_true",
                   help="scale the output instead of building the weight; "
                        "measured slower on MPS, never read on a card")
    p.add_argument("--out", required=True, help="where the result is written")
    p.add_argument("--journal", default=None, help="jsonl journal path")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--repeat-one-batch", action="store_true",
                   help="overfit a single batch; the gradient sanity check")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .adapters.dclm_corpus import DclmCorpus
    from .adapters.json_sink import JsonSink
    from .adapters.jsonl_recorder import JsonlRecorder
    from .adapters.objectives import CrossEntropy, KLDistillation
    from .adapters.torch_model import TorchModel, TorchTeacher
    from .adapters.torch_optimizer import Adam
    from .domain.loop import Plan, check, run
    from .domain.schedule import WarmupCosine
    from .trainables import build

    widths = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}
    dtype = (
        widths[args.dtype]
        if args.dtype != "auto"
        else (torch.bfloat16 if args.device.startswith("cuda") else torch.float32)
    )
    student = AutoModelForCausalLM.from_pretrained(
        args.student, torch_dtype=dtype
    ).to(args.device)
    if args.grad_checkpoint:
        student.gradient_checkpointing_enable()

    shapes = TorchModel.shapes_for(student)
    if args.mode == "low_rank":
        full = {n: (r, student.get_submodule(n).weight.shape[1])
                for n, (r, _) in shapes.items()}
        trainable = build("low_rank", shapes=full, rank=args.rank,
                          device=args.device)
    elif args.mode == "free_params":
        tails = {}
        for name, (_, lattice) in shapes.items():
            w = student.get_submodule(name).weight
            tails[name] = w[:, lattice:].detach().float().cpu()
        trainable = build("free_params", shapes=shapes, tails=tails,
                          device=args.device)
    else:
        trainable = build(args.mode, shapes=shapes, device=args.device)

    model = TorchModel(student, trainable, commute=args.commute)
    cost = trainable.cost(model.param_count)
    print(f"mode {trainable.name}: {cost}", file=sys.stderr)

    objective = KLDistillation() if args.objective == "kl" else CrossEntropy()
    teacher = None
    if objective.needs_teacher:
        if args.teacher is None:
            print("--objective kl needs --teacher", file=sys.stderr)
            return 2
        dense = AutoModelForCausalLM.from_pretrained(
            args.teacher, torch_dtype=dtype
        ).to(args.device)
        teacher = TorchTeacher(dense)

    tokenizer = AutoTokenizer.from_pretrained(args.teacher or args.student)
    corpus = DclmCorpus(
        tokenizer, args.batch_size, args.seq_len, device=args.device
    )
    if args.repeat_one_batch:
        from .adapters.repeat_corpus import RepeatCorpus

        corpus = RepeatCorpus(corpus, warm_seed=args.seed)
    plan = Plan(
        steps=args.steps, seed=args.seed,
        checkpoint_every=args.checkpoint_every,
    )

    check(trainable=trainable, objective=objective, teacher=teacher,
          corpus=corpus, plan=plan)
    tokens = corpus.tokens_per_batch * plan.steps
    print(f"wiring accepted: {tokens} tokens, {len(model.matrices)} matrices",
          file=sys.stderr)
    if args.dry_run:
        print("dry run, nothing was trained", file=sys.stderr)
        return 0

    sink = JsonSink(args.out)
    outcome = run(
        model=model,
        trainable=trainable,
        objective=objective,
        optimizer=Adam(trainable.parameters()),
        schedule=WarmupCosine(args.lr, plan.steps, args.warmup),
        corpus=corpus,
        recorder=JsonlRecorder(args.journal),
        plan=plan,
        teacher=teacher,
        sink=sink,
    )
    path = sink.write(outcome.export, outcome.cost)
    print(f"wrote {path}", file=sys.stderr)
    return 0 if outcome.improved else 1


if __name__ == "__main__":
    raise SystemExit(main())
