"""The loop, exercised with no framework bound."""

import pytest

from llvqtune.domain.loop import Plan, WiringError, run
from llvqtune.domain.schedule import Constant

from fakes import (
    FakeCorpus,
    FakeGauge,
    FakeSink,
    FakeModel,
    FakeObjective,
    FakeOptimizer,
    FakeRecorder,
    FakeTeacher,
    FakeTrainable,
)


def wire(**overrides):
    parts = dict(
        model=FakeModel(),
        trainable=FakeTrainable(),
        objective=FakeObjective(),
        optimizer=FakeOptimizer(),
        schedule=Constant(1e-3),
        corpus=FakeCorpus(),
        recorder=FakeRecorder(),
        plan=Plan(steps=5, seed=0, log_every=1),
    )
    parts.update(overrides)
    return parts


def test_it_runs_the_steps_it_was_asked_for():
    optimizer = FakeOptimizer()
    recorder = FakeRecorder()
    outcome = run(**wire(optimizer=optimizer, recorder=recorder))
    assert outcome.steps_run == 5
    assert optimizer.steps == 5
    assert len(recorder.steps) == 5


def test_the_header_states_the_cost_before_the_first_step():
    recorder = FakeRecorder()
    run(**wire(recorder=recorder))
    assert recorder.header["b_per_param_whole_model"] == 0.0
    assert recorder.header["tokens_total"] == 8 * 5


def test_an_objective_that_needs_a_teacher_refuses_to_run_without_one():
    with pytest.raises(WiringError, match="no teacher"):
        run(**wire(objective=FakeObjective(needs_teacher=True)))


def test_a_teacher_nobody_reads_is_refused():
    with pytest.raises(WiringError, match="ignores the teacher"):
        run(**wire(teacher=FakeTeacher()))


def test_a_trainable_with_no_parameter_is_refused():
    with pytest.raises(WiringError, match="no parameter"):
        run(**wire(trainable=FakeTrainable(params=0)))


def test_an_empty_corpus_is_refused():
    with pytest.raises(WiringError, match="no batch"):
        run(**wire(corpus=FakeCorpus(available=0)))


def test_a_plan_of_no_step_is_refused():
    with pytest.raises(ValueError):
        Plan(steps=0, seed=0)


def test_the_schedule_sets_the_rate_every_step():
    optimizer = FakeOptimizer()
    run(**wire(optimizer=optimizer, schedule=Constant(7e-4)))
    assert optimizer.rates == [7e-4] * 5


def test_the_outcome_carries_what_the_write_back_reads():
    outcome = run(**wire())
    assert outcome.export == {"kind": "fake"}
    assert outcome.cost.is_free
    assert outcome.improved


def test_checkpointing_writes_on_the_stride_and_not_on_the_last_step():
    """The final write belongs to the caller, so the loop must not duplicate it."""
    sink = FakeSink()
    run(**wire(plan=Plan(steps=6, seed=0, log_every=1, checkpoint_every=2), sink=sink))
    assert len(sink.writes) == 2  # steps 2 and 4, never step 6


def test_checkpointing_is_off_by_default():
    sink = FakeSink()
    run(**wire(sink=sink))
    assert sink.writes == []


def test_a_negative_stride_is_refused():
    with pytest.raises(ValueError, match="count of steps"):
        Plan(steps=4, seed=0, checkpoint_every=-1)


def test_a_run_with_no_sink_ignores_the_stride():
    outcome = run(**wire(plan=Plan(steps=4, seed=0, log_every=1, checkpoint_every=2)))
    assert outcome.steps_run == 4


def test_the_loop_reports_the_time_it_spent_in_itself():
    outcome = run(**wire())
    assert outcome.seconds >= 0.0
    assert outcome.seconds_per_step == outcome.seconds / outcome.steps_run


def test_the_summary_carries_the_rate_a_job_needs_to_size_itself():
    recorder = FakeRecorder()
    run(**wire(recorder=recorder))
    assert "seconds_per_step" in recorder.summary
    assert recorder.summary["seconds_per_step"] >= 0.0


def test_the_summary_carries_the_gauge_when_one_is_bound():
    """The 8B was sized for its card by computation; the peak must be read."""
    recorder = FakeRecorder()
    gauge = FakeGauge()
    run(**wire(recorder=recorder, gauge=gauge))
    assert recorder.summary["gauge"] == {"max_memory_allocated": 1000 * gauge.reads}


def test_every_checkpoint_reads_the_gauge():
    recorder = FakeRecorder()
    gauge = FakeGauge()
    sink = FakeSink()
    run(**wire(
        plan=Plan(steps=6, seed=0, log_every=1, checkpoint_every=2),
        sink=sink, recorder=recorder, gauge=gauge,
    ))
    assert [(i, g) for i, _, g in recorder.checkpoints] == [
        (2, {"max_memory_allocated": 1000}),
        (4, {"max_memory_allocated": 2000}),
    ]
    assert [p for _, p, _ in recorder.checkpoints] == [
        "/tmp/fake-1.json", "/tmp/fake-2.json",
    ]
    assert gauge.reads == 3  # two checkpoints and the summary


def test_without_a_gauge_the_journal_keeps_its_shape():
    """The 4B journals carry no gauge; a run with none must write the same."""
    recorder = FakeRecorder()
    sink = FakeSink()
    run(**wire(
        plan=Plan(steps=4, seed=0, log_every=1, checkpoint_every=2),
        sink=sink, recorder=recorder,
    ))
    assert "gauge" not in recorder.summary
    assert recorder.checkpoints == [(2, "/tmp/fake-1.json", None)]
