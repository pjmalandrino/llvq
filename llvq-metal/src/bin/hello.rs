//! Does the plumbing work at all?
//!
//! Before measuring anything about Leech decoding, establish that a shader
//! compiles, runs, produces correct output, and reports plausible timings.
//! Three numbers come out of it that everything downstream depends on: the
//! SIMD width (the lane-op accounting is meaningless without it), the
//! threadgroup ceiling, and the machine's integer throughput — the denominator
//! of every "operations per block" budget quoted so far.
//!
//! Run: `cargo run --release -p llvq-metal --bin hello`

const SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Correctness probe: something whose answer cannot be guessed from the shape.
kernel void square_plus_index(device const uint* in  [[buffer(0)]],
                              device uint*       out [[buffer(1)]],
                              uint               gid [[thread_position_in_grid]])
{
    uint v = in[gid];
    out[gid] = v * v + gid;
}

// Integer throughput probe: a dependent chain of multiply-adds, so the
// compiler cannot vectorise it away and the result depends on every step.
// ITERS is the chain length; each iteration is 2 integer ops.
kernel void int_throughput(device const uint* in  [[buffer(0)]],
                           device uint*       out [[buffer(1)]],
                           constant uint&     iters [[buffer(2)]],
                           uint               gid [[thread_position_in_grid]])
{
    uint a = in[gid];
    for (uint i = 0; i < iters; ++i) {
        a = a * 1664525u + 1013904223u;
    }
    out[gid] = a;
}
"#;

fn main() -> Result<(), String> {
    let k = llvq_metal::Kernel::new(SRC, "square_plus_index")?;
    println!("GPU: {}", k.device_name());
    println!(
        "  SIMD width            {}\n  max threads/group     {}",
        k.simd_width(),
        k.max_threads_per_group()
    );

    // ---- correctness ----
    const N: usize = 1 << 20;
    let input: Vec<u32> = (0..N as u32).collect();
    let bin = k.buffer(&input);
    let bout = k.empty::<u32>(N);
    k.dispatch(N as u64, 256, |enc| {
        enc.set_buffer(0, Some(&bin), 0);
        enc.set_buffer(1, Some(&bout), 0);
    });
    let got: Vec<u32> = unsafe { k.read(&bout, N) };
    for i in [0usize, 1, 12345, N - 1] {
        let want = (i as u32).wrapping_mul(i as u32).wrapping_add(i as u32);
        assert_eq!(got[i], want, "wrong result at {i}");
    }
    println!("\n  OK: shader compiled, ran, result verified on {N} elements");

    // ---- integer throughput ----
    let k2 = llvq_metal::Kernel::new(SRC, "int_throughput")?;
    const ITERS: u32 = 4096;
    let iters_buf = k2.buffer(&[ITERS]);
    let overhead = k2.overhead(20);
    let t = k2.time(N as u64, 256, 3, 10, |enc| {
        enc.set_buffer(0, Some(&bin), 0);
        enc.set_buffer(1, Some(&bout), 0);
        enc.set_buffer(2, Some(&iters_buf), 0);
    });
    // Two integer ops per iteration, one chain per thread.
    let ops = N as f64 * ITERS as f64 * 2.0;
    let net = t.seconds - overhead;
    let rate = ops / net;
    // Check the result is real: the kernel must have run.
    let out: Vec<u32> = unsafe { k2.read(&bout, N) };
    let mut want = input[12345];
    for _ in 0..ITERS {
        want = want.wrapping_mul(1664525).wrapping_add(1013904223);
    }
    assert_eq!(out[12345], want, "the throughput kernel did not run");
    println!(
        "\n  submission overhead   {:.3} ms (empty dispatch)\n  \
         raw time              {:.3} ms\n  \
         net time              {:.3} ms for {:.1} G op",
        overhead * 1e3,
        t.seconds * 1e3,
        net * 1e3,
        ops / 1e9
    );
    println!("\n  integer throughput    {:.2} T op/s", rate / 1e12);
    println!(
        "\n  This is the denominator of every \"operations per block\" budget.\n  \
         Earlier estimates assumed ~5.7 T op/s."
    );
    Ok(())
}
