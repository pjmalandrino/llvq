# `broadcast_matmul` copies the whole rhs when only the batch dims need broadcasting

<!--
ARCHIVE — verbatim record of what was posted as huggingface/candle#3871 on
2026-08-09. Do not edit: the point of this file is that it matches the issue.
If the upstream text needs to change, change it upstream and re-copy here.

The reproducer and the patch are inlined below so the issue body was
self-contained (relative links do not resolve from candle's tracker). The
same two artifacts also live beside this file as ./repro/ and
./candle-broadcast-matmul.patch, which are the editable copies.
-->

---

### What happens

`Tensor::broadcast_matmul` on a `(b, m, k) @ (k, n)` product takes the `(false, true)`
arm and runs

```rust
lhs.matmul(&rhs.broadcast_as(&r_shape)?.contiguous()?)
```

so it builds a `(b, k, n)` copy of the rhs on every call. When the rhs is a weight
matrix that is a full copy of the weights, per call. And since the rhs is usually a
transposed view (`w.t()`), the copy is a strided gather, not a memcpy.

The code is in `candle-core/src/tensor.rs`, right under the
`// TODO: Avoid concretising the broadcasted matrixes via contiguous.` that is already
there. Same on `main` @ `6f74e7c` (line 1559) and on the released 0.9.2 (line 1550).

The case I ran into is an output head: `h.broadcast_matmul(&w.t()?)` with
`h: (1, 1, hidden)` and `w: (vocab, hidden)`. For Qwen3-4B (vocab 151936, hidden 2560,
f16) that is 778 MB copied per decoded token. For an untied 8B head, 1.24 GB.

To be clear about the scope: `candle_nn::Linear::forward` already avoids this
deliberately,

```rust
// When possible, we avoid using a broadcasted matmul as it is much slower
// than the standard matmul for the cuda and cpu backends.
```

by folding the leading dims of `x` into its row dimension. So models built on `Linear`,
including the ones in `candle_transformers`, do not pay it. What I am reporting is that
the primitive still has the trap: write the mathematically identical
`h.broadcast_matmul(&w.t()?)` in your own code and you silently get an `O(k*n)` copy per
call, with nothing at the call site to suggest it. The fix below is the same fold, moved
into the primitive, so the workaround in `candle-nn` stops being the thing everyone
depends on.

### Reproducer

Self-contained, CPU only, no model needed. `cargo run --release -- 151936 2560 1 f16 5`
times `h.broadcast_matmul(&w.t()?)` against the manual fold
`h.reshape((m, k))?.matmul(&w.t()?)?.reshape(..)` on the same tensors.

Measured on 4 x86-64 vCPU, candle `main` @ `6f74e7c`, `--release`, best of N:

| lhs | rhs | dtype | `broadcast_matmul` | manual fold |
|---|---|---|---|---|
| `(1, 1, 2560)` | `(151936, 2560)ᵀ` | f16 | 8 104 ms | 76.6 ms |
| `(1, 1, 2560)` | `(151936, 2560)ᵀ` | f32 | 23 663 ms | 151 ms |
| `(1, 8, 2560)` | `(151936, 2560)ᵀ` | f16 | 9 247 ms | 267 ms |

Those CPU numbers are dominated by a single-threaded strided copy, so do not read the
ratio as a GPU ratio. What holds on any backend is the shape of the cost: `O(b*k*n)`
bytes written then read back per call, for a product that only needs to read `O(k*n)`
bytes once.

For an accelerator number: I hit this call in a decode loop on an L40S (Qwen3-4B, f16,
batch 1). With per-phase device fences, the head phase measures 26.7 ms/token against
13.3 ms for all 36 transformer blocks combined. One `broadcast_matmul` costing twice the
rest of the model. I found it by reading the path and timing the phase, not with a
profiler, so I can only say the copy is whatever `copy_strided_src` dispatches for the
dtype (`ucopy_f16` here). Raw log:
[phases-2026-08-07.txt](https://github.com/pjmalandrino/llvq/blob/main/docs/mesures/phases-2026-08-07.txt).
That call site is mine, not a `candle_transformers` model, see the note above.

<details>
<summary>repro/src/main.rs (deps: <code>candle-core = "=0.9.2"</code>, or a path to your checkout)</summary>

```rust
// Minimal reproducer: Tensor::broadcast_matmul materialises the broadcast
// operand, so a (b, m, k) x (k, n) product copies the whole rhs on every call.
//
// usage: cargo run --release -- <vocab> <hidden> <rows> <dtype: f16|f32> <iters>
// CHECK=1 also reports each path's error against the f32 product of the same
// inputs, which is where the f16 accumulation gap shows up.
use candle_core::{DType, Device, Tensor};
use std::time::Instant;

fn main() -> candle_core::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let n: usize = a.get(1).map_or(151_936, |s| s.parse().unwrap()); // vocab
    let k: usize = a.get(2).map_or(2_560, |s| s.parse().unwrap()); // hidden
    let m: usize = a.get(3).map_or(1, |s| s.parse().unwrap()); // tokens in flight
    let dt = match a.get(4).map(|s| s.as_str()) {
        Some("f32") => DType::F32,
        _ => DType::F16,
    };
    let iters: usize = a.get(5).map_or(5, |s| s.parse().unwrap());
    let dev = Device::Cpu;

    // The lm_head weight, exactly as a checkpoint stores it: (vocab, hidden).
    let w = Tensor::rand(-1f32, 1f32, (n, k), &dev)?.to_dtype(dt)?;
    // Hidden states with a leading batch dim, exactly as a decoder emits them.
    let h = Tensor::rand(-1f32, 1f32, (1, m, k), &dev)?.to_dtype(dt)?;
    let bytes = (n * k * dt.size_in_bytes()) as f64 / 1e6;
    println!("weight {n}x{k} {dt:?} = {bytes:.1} MB, lhs (1,{m},{k}), {iters} iters");

    let mut slow = f64::MAX;
    let mut fast = f64::MAX;
    let (mut a_out, mut b_out) = (None, None);
    for _ in 0..iters {
        let t = Instant::now();
        let y = h.broadcast_matmul(&w.t()?)?;
        slow = slow.min(t.elapsed().as_secs_f64());
        a_out = Some(y);

        let t = Instant::now();
        let y = h
            .reshape((m, k))?
            .matmul(&w.t()?)?
            .reshape((1, m, n))?;
        fast = fast.min(t.elapsed().as_secs_f64());
        b_out = Some(y);
    }
    let (a_out, b_out): (Tensor, Tensor) = (
        a_out.unwrap().to_dtype(DType::F32)?,
        b_out.unwrap().to_dtype(DType::F32)?,
    );
    let scale = b_out.abs()?.max_all()?.to_scalar::<f32>()?;
    let d = (&a_out - &b_out)?.abs()?.max_all()?.to_scalar::<f32>()?;
    println!("broadcast_matmul : {:9.3} ms", slow * 1e3);
    println!(
        "reshape + matmul : {:9.3} ms  ({:.1}x faster)",
        fast * 1e3,
        slow / fast
    );
    println!(
        "max |diff|       : {d:e}  (max |out| = {scale:e}, rel = {:e})",
        d / scale
    );
    // Which of the two is closer to the truth? Same product in f32.
    if dt == DType::F16 && std::env::var("CHECK").is_ok() {
        let truth = h
            .to_dtype(DType::F32)?
            .reshape((m, k))?
            .matmul(&w.to_dtype(DType::F32)?.t()?)?
            .reshape((1, m, n))?;
        let err = |t: &Tensor| -> candle_core::Result<f32> {
            Ok((t - &truth)?.abs()?.max_all()?.to_scalar::<f32>()? / scale)
        };
        println!(
            "vs f32 truth     : broadcast rel {:e}, fold rel {:e}",
            err(&a_out)?,
            err(&b_out)?
        );
    }
    Ok(())
}
```

</details>

### It is also less accurate in f16

Same reproducer with `CHECK=1` on `32768 x 2560`, f16 inputs, error against the f32
product of the same inputs, relative to `max |out|`:

```
max |diff|       : 9.6875e-1  (max |out| = 6.975e1, rel = 1.3888889e-2)
vs f32 truth     : broadcast rel 1.3748005e-2, fold rel 3.414906e-4
```

The folded path lands about 40x closer to the f32 result. The batched path looks like it
accumulates in a narrower type than the 2-D one. I did not chase it further, since the
fix removes that path anyway.

### Proposed fix

A rank-2 rhs is only broadcast over the batch dims, so the whole product is one 2-D
matmul once the leading dims of the lhs are folded into its rows. Same thing
`Linear::forward` does, with the same `is_contiguous()` guard:

```rust
(false, true) if rhs.rank() == 2 && lhs.is_contiguous() => {
    let (lhs_dims, rhs_dims) = (lhs.dims(), rhs.dims());
    let (m, k) = (lhs_dims[lhs.rank() - 2], lhs_dims[lhs.rank() - 1]);
    let n = rhs_dims[1];
    let batch: usize = lhs_dims[..lhs.rank() - 2].iter().product();
    let mut out_dims = lhs_dims.to_vec();
    out_dims.pop();
    out_dims.push(n);
    lhs.reshape((batch * m, k))?.matmul(rhs)?.reshape(out_dims)
}
```

With it, `broadcast_matmul` and the manual fold are the same code, so the table above
collapses to 81.0 ms (f16) and 154 ms (f32), bit-identical between the two.

On `main` @ `6f74e7c`, with the patch below applied:

* `cargo test -p candle-core --release` passes in full, `grad_tests` included
  (`reshape` and `matmul` are both differentiable, so autograd is unaffected).
* `cargo fmt -p candle-core -- --check` is clean.
* The added `broadcast_matmul_rank2_rhs` test checks the folded result against the
  per-batch products it stands for, over rank-3 and rank-4 lhs, `m = 1`, and a
  non-contiguous lhs that has to fall back. It dies if the fold is wrong: mutating
  `reshape((batch * m, k))` to `reshape((m, batch * k))` makes it fail.

Happy to send this as a PR.

<details>
<summary>Patch (fix + test), <code>git diff</code> against <code>main</code> @ <code>6f74e7c</code></summary>

```diff
diff --git a/candle-core/src/tensor.rs b/candle-core/src/tensor.rs
index 7070a00..e9eb310 100644
--- a/candle-core/src/tensor.rs
+++ b/candle-core/src/tensor.rs
@@ -1562,6 +1562,22 @@ impl Tensor {
                 .broadcast_as(&l_shape)?
                 .contiguous()?
                 .matmul(&rhs.broadcast_as(&r_shape)?.contiguous()?),
+            // A rank-2 rhs is only broadcast over the batch dimensions, so the
+            // whole product is one 2D matmul once the leading dims of lhs are
+            // folded into its row dimension. Broadcasting the rhs instead would
+            // copy it `batch` times -- for an lm_head that is the entire
+            // vocabulary matrix, per call. Same trick, and same contiguity
+            // guard, as `candle_nn::Linear::forward`.
+            (false, true) if rhs.rank() == 2 && lhs.is_contiguous() => {
+                let (lhs_dims, rhs_dims) = (lhs.dims(), rhs.dims());
+                let (m, k) = (lhs_dims[lhs.rank() - 2], lhs_dims[lhs.rank() - 1]);
+                let n = rhs_dims[1];
+                let batch: usize = lhs_dims[..lhs.rank() - 2].iter().product();
+                let mut out_dims = lhs_dims.to_vec();
+                out_dims.pop();
+                out_dims.push(n);
+                lhs.reshape((batch * m, k))?.matmul(rhs)?.reshape(out_dims)
+            }
             (false, true) => lhs.matmul(&rhs.broadcast_as(&r_shape)?.contiguous()?),
             (true, false) => lhs.broadcast_as(&l_shape)?.contiguous()?.matmul(rhs),
             (false, false) => lhs.matmul(rhs),
diff --git a/candle-core/tests/matmul_tests.rs b/candle-core/tests/matmul_tests.rs
index c6b3e59..c3e7371 100644
--- a/candle-core/tests/matmul_tests.rs
+++ b/candle-core/tests/matmul_tests.rs
@@ -149,6 +149,36 @@ fn zero_matmul_device_validation(device: &Device) -> Result<()> {
     Ok(())
 }
 
+// A rank-2 rhs is broadcast over the batch dims only, which `broadcast_matmul`
+// folds into a single 2D matmul instead of copying the rhs. Check the folded
+// path against the per-batch products it stands for, contiguous lhs (folded)
+// and non-contiguous lhs (fallback) alike.
+fn broadcast_matmul_rank2_rhs(device: &Device) -> Result<()> {
+    let rhs = Tensor::randn(0f32, 1f32, (5, 2), device)?;
+    for lhs in [
+        Tensor::randn(0f32, 1f32, (3, 4, 5), device)?,
+        Tensor::randn(0f32, 1f32, (3, 6, 4, 5), device)?,
+        Tensor::randn(0f32, 1f32, (1, 1, 5), device)?,
+        Tensor::randn(0f32, 1f32, (3, 5, 4), device)?.transpose(1, 2)?,
+    ] {
+        let out = lhs.broadcast_matmul(&rhs)?;
+        let mut dims = lhs.dims().to_vec();
+        let n = dims.len();
+        dims[n - 1] = 2;
+        assert_eq!(out.dims(), dims.as_slice());
+        // Same product, computed the way the doc comment describes it.
+        let batch: usize = lhs.dims()[..n - 2].iter().product();
+        let (m, k) = (lhs.dims()[n - 2], lhs.dims()[n - 1]);
+        let flat = lhs.reshape((batch, m, k))?;
+        let out = out.reshape((batch, m, 2))?;
+        for b in 0..batch {
+            let diff = (out.i(b)? - flat.i(b)?.matmul(&rhs)?)?.sqr()?.sum_all()?;
+            assert!(diff.to_vec0::<f32>()? < 1e-6);
+        }
+    }
+    Ok(())
+}
+
 #[test]
 fn tensor_dot() -> Result<()> {
     let lhs = Tensor::new(&[1., 2., 3.], &Device::Cpu)?;
@@ -227,5 +257,11 @@ test_device!(
     zero_matmul_device_validation_gpu,
     zero_matmul_device_validation_metal
 );
+test_device!(
+    broadcast_matmul_rank2_rhs,
+    broadcast_matmul_rank2_rhs_cpu,
+    broadcast_matmul_rank2_rhs_gpu,
+    broadcast_matmul_rank2_rhs_metal
+);
 test_device!(squeeze_mm, squeeze_mm_cpu, squeeze_mm_gpu, squeeze_mm_metal);
 test_device!(mm_layout, mm_layout_cpu, mm_layout_gpu, mm_layout_metal);
```

</details>

### What I deliberately left out

* Only the `(false, true)` arm with a rank-2 rhs is folded. A rhs of rank > 2 whose batch
  dims are all 1 (say `(1, k, n)`) still takes the copy path. It could be squeezed first,
  but that needs care about which reshapes are free.
* `(true, false)` and `(true, true)` still concretise. The original TODO covers them too
  and they do not have a one-line answer.
* The `is_contiguous()` guard is there because `reshape` on a non-contiguous lhs would
  itself copy. Copying the lhs is usually much cheaper than copying the rhs, but not
  always, so this keeps the change to the case where it is free. Same reasoning as
  `Linear::forward`.

Say the word if you would rather have the general fix in one go.

### Related

* #3253 (open) asks for stride-0 broadcast dims in `matmul`, the general case. Different
  symptom (an error rather than a copy), same underlying limitation.
* #1965 (closed) asks for avoiding `.contiguous()` before `matmul`.
* #513 is the request that added broadcasting to matmul in the first place.

None of them reports the rhs materialization for a rank-2 rhs.
