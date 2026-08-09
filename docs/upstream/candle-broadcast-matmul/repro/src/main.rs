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
