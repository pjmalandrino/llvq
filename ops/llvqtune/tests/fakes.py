"""Stand-ins for every port, so the loop is tested without a framework."""

from __future__ import annotations

from llvqtune.domain.bits import BitCost


class FakeTensor:
    def __init__(self, value: float) -> None:
        self.value = value


class FakeModel:
    def __init__(self, params: int = 1000) -> None:
        self.param_count = params
        self.seen = 0

    @property
    def matrices(self):
        return ["m0"]

    def rows_of(self, matrix):
        return 4

    def lattice_columns_of(self, matrix):
        return 24

    def forward(self, ids):
        self.seen += 1
        return FakeTensor(1.0 / self.seen)


class FakeTrainable:
    def __init__(self, cost_bits: int = 0, params: int = 1) -> None:
        self._cost_bits = cost_bits
        self._params = [FakeTensor(1.0)] * params

    name = "fake"

    def parameters(self):
        return self._params

    def weight(self, matrix, frozen):
        return frozen

    def cost(self, model_params):
        return BitCost(
            added_params=0 if self._cost_bits == 0 else 1,
            added_bits=self._cost_bits,
            model_params=model_params,
        )

    def export(self):
        return {"kind": "fake"}


class FakeObjective:
    def __init__(self, needs_teacher: bool = False) -> None:
        self._needs = needs_teacher

    name = "fake_objective"

    @property
    def needs_teacher(self):
        return self._needs

    def __call__(self, student, teacher, targets):
        return student

    def value(self, loss):
        return float(loss.value)


class FakeOptimizer:
    def __init__(self) -> None:
        self.steps = 0
        self.rates: list[float] = []

    def zero_grad(self):
        pass

    def backward(self, loss):
        pass

    def step(self, lr):
        self.steps += 1
        self.rates.append(lr)


class FakeCorpus:
    def __init__(self, tokens: int = 8, available: int = 1000) -> None:
        self.tokens_per_batch = tokens
        self._available = available

    def batches(self, count, seed):
        for index in range(min(count, self._available)):
            yield FakeTensor(float(index))


class FakeTeacher:
    def logits(self, ids):
        return FakeTensor(0.0)


class FakeRecorder:
    def __init__(self) -> None:
        self.header = None
        self.steps: list[tuple[int, float, float]] = []
        self.summary = None

    def opened(self, header):
        self.header = header

    def step(self, index, loss, lr):
        self.steps.append((index, loss, lr))

    def closed(self, summary):
        self.summary = summary


class FakeSink:
    def __init__(self) -> None:
        self.writes: list[dict] = []

    def write(self, payload, cost):
        self.writes.append(payload)
        return f"/tmp/fake-{len(self.writes)}.json"
