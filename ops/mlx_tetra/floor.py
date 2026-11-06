# /// script
# requires-python = ">=3.10"
# dependencies = ["mlx==0.29.3"]
# ///
# mlx 0.29.3 is the version this reading ran (journal metal-nuit-2026-11-06, MLX prototype).
# The engine and its other scripts run mlx 0.32.4: this floor is not re-read on that version.
"""The cost of one dependent dispatch in MLX on this Mac: chains of tiny ops."""
import statistics, time
import mlx.core as mx
N = 400
def chain(f, x0):
    def once():
        x = x0
        for _ in range(N):
            x = f(x)
        mx.eval(x)
    once()
    ts = []
    for _ in range(5):
        t = time.perf_counter(); once(); ts.append((time.perf_counter() - t) * 1e6 / N)
    return statistics.median(ts), min(ts), max(ts)
x = mx.zeros((1, 2560), dtype=mx.float16); mx.eval(x)
print("add 1, 2560 f16       %.2f us [%.2f-%.2f]" % chain(lambda v: v + 1, x))
print("astype f32 then f16   %.2f us [%.2f-%.2f]" % chain(lambda v: v.astype(mx.float32).astype(mx.float16), x))
y = mx.zeros((1, 9728), dtype=mx.float16); mx.eval(y)
print("add 1, 9728 f16       %.2f us [%.2f-%.2f]" % chain(lambda v: v + 1, y))
