"""The training loop, written in ports only.

It imports no framework and no file format. Swapping torch for anything else,
or the corpus for another, changes an adapter and leaves this file alone. That
is the whole reason the package is laid out this way.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from ..ports.corpus import CorpusPort
from ..ports.model import ModelPort
from ..ports.optimizer import OptimizerPort
from ..ports.recorder import RecorderPort
from ..ports.sink import SinkPort
from ..ports.teacher import TeacherPort
from .bits import BitCost
from .objective import Objective
from .trainable import Trainable


class Schedule(Protocol):
    def at(self, step: int) -> float: ...


@dataclass(frozen=True)
class Plan:
    """Everything a run needs to be replayed from its journal."""

    steps: int
    seed: int
    log_every: int = 10
    checkpoint_every: int = 0
    """Steps between two partial writes. 0 disables them.

    A run measured in hours must not be all or nothing. Writing the export
    periodically costs one serialization and turns a crash at hour eleven
    into a shorter run rather than a lost one.
    """

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.log_every <= 0:
            raise ValueError("log_every must be positive")
        if self.checkpoint_every < 0:
            raise ValueError("checkpoint_every is a count of steps, or 0")


@dataclass(frozen=True)
class Outcome:
    steps_run: int
    first_loss: float
    last_loss: float
    losses: tuple[float, ...]
    seconds: float
    """Wall time spent in the loop, loading excluded.

    A job that picks its own step count needs the rate, and the rate read
    from the outside includes pulling two checkpoints. On 2026-09-19 that
    contamination was the whole error: a step was priced from a measurement
    that was not the step.
    """
    cost: BitCost
    export: dict[str, object]

    @property
    def seconds_per_step(self) -> float:
        return self.seconds / self.steps_run if self.steps_run else 0.0

    @property
    def improved(self) -> bool:
        """True when the last third of the run sits below the first third.

        Only meaningful when the corpus repeats a fixed batch. On a streaming
        corpus each step reads different text, so two losses differ by the
        text as much as by the parameters and this says nothing. That confound
        is what `adapters.repeat_corpus.RepeatCorpus` exists to remove.
        """
        n = len(self.losses)
        if n < 3:
            return self.last_loss < self.first_loss
        k = max(1, n // 3)
        head = sum(self.losses[:k]) / k
        tail = sum(self.losses[-k:]) / k
        return tail < head


class WiringError(RuntimeError):
    """A run whose parts do not fit. Raised before the first batch is read."""


def check(
    *,
    trainable: Trainable,
    objective: Objective,
    teacher: TeacherPort | None,
    corpus: CorpusPort,
    plan: Plan,
) -> None:
    """Refuse an impossible run now rather than at step 400.

    Every condition here cost nothing to check and costs a GPU hour to
    discover. Hard rule 1 asks for the cost up front; this is the same idea
    applied to the wiring.
    """
    if objective.needs_teacher and teacher is None:
        raise WiringError(
            f"objective {objective.name!r} reads reference logits "
            "and no teacher is bound"
        )
    if not objective.needs_teacher and teacher is not None:
        raise WiringError(
            f"objective {objective.name!r} ignores the teacher, "
            "so binding one would bill a forward pass for nothing"
        )
    if not trainable.parameters():
        raise WiringError(
            f"trainable {trainable.name!r} exposes no parameter; "
            "nothing would move"
        )
    if corpus.tokens_per_batch <= 0:
        raise WiringError("corpus reports a non-positive batch size")
    if plan.steps <= 0:
        raise WiringError("plan asks for no step")


def run(
    *,
    model: ModelPort,
    trainable: Trainable,
    objective: Objective,
    optimizer: OptimizerPort,
    schedule: Schedule,
    corpus: CorpusPort,
    recorder: RecorderPort,
    plan: Plan,
    teacher: TeacherPort | None = None,
    sink: SinkPort | None = None,
) -> Outcome:
    """Train `trainable` against `objective`. Returns what to write back."""
    check(
        trainable=trainable,
        objective=objective,
        teacher=teacher,
        corpus=corpus,
        plan=plan,
    )

    cost = trainable.cost(model.param_count)
    recorder.opened(
        {
            "trainable": trainable.name,
            "objective": objective.name,
            "steps": plan.steps,
            "seed": plan.seed,
            "tokens_per_batch": corpus.tokens_per_batch,
            "tokens_total": corpus.tokens_per_batch * plan.steps,
            "cost": str(cost),
            "b_per_param_whole_model": cost.b_per_param_whole_model,
        }
    )

    first: float | None = None
    last = 0.0
    done = 0
    losses: list[float] = []
    checkpoints: list[str] = []
    started = time.monotonic()

    for index, ids in enumerate(corpus.batches(plan.steps, plan.seed)):
        if index >= plan.steps:
            break
        lr = schedule.at(index)
        optimizer.zero_grad()
        student = model.forward(ids)
        reference = teacher.logits(ids) if teacher is not None else None
        loss = objective(student, reference, ids)
        optimizer.backward(loss)
        optimizer.step(lr)

        last = objective.value(loss)
        losses.append(last)
        if first is None:
            first = last
        done = index + 1
        if done % plan.log_every == 0 or done == plan.steps:
            recorder.step(done, last, lr)
        if (
            sink is not None
            and plan.checkpoint_every
            and done % plan.checkpoint_every == 0
            and done != plan.steps
        ):
            path = sink.write(trainable.export(), cost)
            recorder.step(done, last, lr)
            checkpoints.append(path)

    elapsed = time.monotonic() - started
    if first is None:
        raise WiringError("the corpus yielded no batch")

    payload = trainable.export()
    recorder.closed(
        {
            "steps_run": done,
            "first_loss": first,
            "last_loss": last,
            "losses": losses,
            "checkpoints": len(checkpoints),
            "seconds": round(elapsed, 3),
            "seconds_per_step": round(elapsed / done, 4) if done else 0.0,
            "cost": str(cost),
        }
    )
    return Outcome(
        steps_run=done,
        first_loss=first,
        last_loss=last,
        losses=tuple(losses),
        seconds=elapsed,
        cost=cost,
        export=payload,
    )
