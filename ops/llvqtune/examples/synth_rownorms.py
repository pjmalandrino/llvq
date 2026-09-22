"""Synthetic first test of row_norms against row_scales. Not a quality claim.

Teacher: a tiny Qwen3 trained on a sparse Markov chain so its distributions
are peaked. Student: the teacher with q,k,o,gate,up,down replaced by a crude
shape-gain quantization (per 24-block: norm * sign(v)/sqrt(24)); v_proj kept
dense, as it is excluded from training in the real run. Both arms run the
package's own loop, objective, optimizer and schedule, on the same batches.
Read: held-out KL(teacher || student) on fresh sequences. Journal:
docs/mesures/row-norms-synthetique-2026-09-22.txt. Run from ops/llvqtune:
`uv run python examples/synth_rownorms.py 300`, about 3 min on 4 CPU.
"""

import json
import math
import sys

import torch
import transformers

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from llvqtune.adapters.objectives import KLDistillation  # noqa: E402
from llvqtune.adapters.torch_model import TorchModel, TorchTeacher  # noqa: E402
from llvqtune.adapters.torch_optimizer import Adam  # noqa: E402
from llvqtune.domain.loop import Plan, run  # noqa: E402
from llvqtune.domain.schedule import WarmupCosine  # noqa: E402
from llvqtune.trainables import build  # noqa: E402

torch.set_num_threads(4)
V, SEQ, BATCH = 256, 64, 16


def chain(seed):
    g = torch.Generator().manual_seed(seed)
    succ = torch.randint(0, V, (V, 4), generator=g)
    prob = torch.softmax(torch.randn(V, 4, generator=g) * 2, -1)
    return succ, prob


def sample(succ, prob, n, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.empty(n, SEQ, dtype=torch.long)
    x[:, 0] = torch.randint(0, V, (n,), generator=g)
    for t in range(1, SEQ):
        k = torch.multinomial(prob[x[:, t - 1]], 1, generator=g).squeeze(1)
        x[:, t] = succ[x[:, t - 1], k]
    return x


def make_teacher(succ, prob):
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(
        vocab_size=V, hidden_size=120, intermediate_size=240, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        max_position_embeddings=SEQ, tie_word_embeddings=True,
    )
    m = transformers.Qwen3ForCausalLM(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for step in range(400):
        ids = sample(succ, prob, 32, 1000 + step)
        loss = m(input_ids=ids, labels=ids).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"teacher trained, last CE {loss.item():.3f}", file=sys.stderr)
    return m.eval()


def quantize(m):
    with torch.no_grad():
        for name, mod in m.named_modules():
            if not isinstance(mod, torch.nn.Linear) or name.endswith("v_proj") or name == "lm_head":
                continue
            w = mod.weight
            cols = w.shape[1] - w.shape[1] % 24
            b = w[:, :cols].reshape(w.shape[0], -1, 24)
            norm = b.norm(dim=-1, keepdim=True)
            w[:, :cols] = (norm * torch.sign(b) / math.sqrt(24)).reshape(w.shape[0], cols)
    return m


class Stream:
    def __init__(self, succ, prob):
        self.succ, self.prob = succ, prob

    @property
    def tokens_per_batch(self):
        return BATCH * SEQ

    def batches(self, count, seed):
        for i in range(count):
            yield sample(self.succ, self.prob, BATCH, seed * 100000 + i)


class Quiet:
    def opened(self, h): pass
    def step(self, i, loss, lr): pass
    def closed(self, s): pass


class NoSink:
    def write(self, payload, cost): return ""


@torch.no_grad()
def heldout_kl(student, teacher, held):
    return float(KLDistillation()(student(input_ids=held).logits, teacher(input_ids=held).logits))


def arm(mode, teacher_state, teacher, succ, prob, held, seed, steps):
    student = transformers.Qwen3ForCausalLM(teacher.config)
    student.load_state_dict(teacher_state)
    quantize(student.eval())
    for p in student.parameters():
        p.requires_grad_(False)
    shapes = TorchModel.shapes_for(student, types=("q_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"))
    kw = {"shapes": shapes}
    if mode == "row_norms":
        kw["norms"] = TorchModel.norm_widths_for(student)
    trainable = build(mode, **kw)
    model = TorchModel(student, trainable)
    before = heldout_kl(student, teacher, held)
    if mode != "none":
        run(model=model, trainable=trainable, objective=KLDistillation(),
            optimizer=Adam(trainable.parameters()), schedule=WarmupCosine(1e-2, steps, 20),
            corpus=Stream(succ, prob), recorder=Quiet(), plan=Plan(steps=steps, seed=seed),
            teacher=TorchTeacher(teacher), sink=NoSink())
    return before, heldout_kl(student, teacher, held)


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    succ, prob = chain(7)
    teacher = make_teacher(succ, prob)
    state = {k: v.clone() for k, v in teacher.state_dict().items()}
    held = sample(succ, prob, 64, 999_999)
    out = []
    for seed in (0, 1, 2):
        for mode in ("row_scales", "row_norms"):
            b, a = arm(mode, state, teacher, succ, prob, held, seed, steps)
            out.append({"seed": seed, "mode": mode, "kl_before": b, "kl_after": a})
            print(json.dumps(out[-1]), flush=True)


if __name__ == "__main__":
    main()
