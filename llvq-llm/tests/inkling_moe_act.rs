//! Gates of `inkling::moe_act`: the activation of the fused one-row MoE in
//! one launch, against the candle operations it replaces, on Metal.
//!
//! The kernel is meant to give candle's bits. The gates compare bits:
//!   - each of its three factors (the gate's softcap, the sigmoid, the up's
//!     softcap) against candle's operations, on special values, on a grid of
//!     the range the model uses and on random bit patterns; and, by name, on
//!     every one of the 2^32 f32 values (`probe_is_candle_on_every_f32`);
//!   - the whole activation, f16 out, against candle's composite on random
//!     pairs at the shape of a call and on pairs built from special values.
//!
//! A NaN equals a NaN whatever its payload; everything else is compared by
//! its bits, the sign of a zero included.
#![cfg(all(target_os = "macos", feature = "metal"))]

use candle_core::{DType, Device, Result, Tensor};
use llvq_llm::inkling::moe_act::MoeAct;

/// The caps of the checkpoint's config (`mlp_gate_softcap`, `mlp_up_softcap`).
const CAP_GATE: f64 = 4.0;
const CAP_UP: f64 = 25.0;

fn metal() -> Device {
    Device::new_metal(0).expect("a Metal device")
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn unif(&mut self) -> f64 {
        ((self.next() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f32 {
        let (u, v) = (self.unif(), self.unif());
        ((-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()) as f32
    }
}

/// `Inkling2LoopSoftcap` as `resident.rs` runs it.
fn softcap(x: &Tensor, cap: f64) -> Result<Tensor> {
    (x / cap)?.tanh()? * cap
}

fn same(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

/// The values whose three factors differ from candle's, per factor, and the
/// first few of them.
fn probe_against_candle(act: &MoeAct, dev: &Device, xs: Vec<f32>) -> Result<([usize; 3], Vec<String>)> {
    let n = xs.len();
    let x = Tensor::from_vec(xs.clone(), n, dev)?;
    let got = act.probe(&x, CAP_GATE, CAP_UP)?.to_vec2::<f32>()?;
    let want = [
        softcap(&x, CAP_GATE)?.to_vec1::<f32>()?,
        candle_nn::ops::sigmoid(&x)?.to_vec1::<f32>()?,
        softcap(&x, CAP_UP)?.to_vec1::<f32>()?,
    ];
    let names = ["softcap gate", "sigmoid", "softcap up"];
    let mut bad = [0usize; 3];
    let mut first = vec![];
    for f in 0..3 {
        for i in 0..n {
            if !same(got[f][i], want[f][i]) {
                bad[f] += 1;
                if first.len() < 12 {
                    first.push(format!(
                        "{} of {:e} ({:#010x}): kernel {:e} ({:#010x}), candle {:e} ({:#010x})",
                        names[f],
                        xs[i],
                        xs[i].to_bits(),
                        got[f][i],
                        got[f][i].to_bits(),
                        want[f][i],
                        want[f][i].to_bits()
                    ));
                }
            }
        }
    }
    Ok((bad, first))
}

fn specials() -> Vec<f32> {
    let mut v = vec![
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::MAX,
        f32::MIN,
        f32::EPSILON,
        1.0,
        -1.0,
        f32::from_bits(1),
        f32::from_bits(0x8000_0001),
        f32::from_bits(0x007f_ffff),
    ];
    for e in -40..40 {
        for s in [1.0f32, -1.0] {
            v.push(s * 2f32.powi(e));
            v.push(s * 1.5 * 2f32.powi(e));
            v.push(s * 10f32.powi(e / 2));
        }
    }
    v
}

/// The three factors on special values, on a grid of [-64, 64] and on random
/// bit patterns: candle's bits.
#[test]
fn probe_is_candle_on_structured_values() -> Result<()> {
    let dev = metal();
    let act = MoeAct::new(&dev)?;
    let mut xs = specials();
    let grid = 1usize << 20;
    xs.extend((0..grid).map(|i| -64.0 + 128.0 * (i as f32 + 0.5) / grid as f32));
    let mut rng = Rng(0xAC7_0001);
    xs.extend((0..1usize << 20).map(|_| f32::from_bits(rng.next() as u32)));
    xs.extend((0..1usize << 18).map(|_| 3.0 * rng.normal()));
    let n = xs.len();
    let (bad, first) = probe_against_candle(&act, &dev, xs)?;
    println!("probe on {n} values: differing softcap gate {}, sigmoid {}, softcap up {}", bad[0], bad[1], bad[2]);
    for l in &first {
        println!("  {l}");
    }
    assert_eq!(bad, [0, 0, 0], "the factors are not candle's bits");
    Ok(())
}

/// The three factors on every f32 value, 2^24 at a time. Run by name.
#[test]
#[ignore = "every f32 value through the GPU, about a minute; run by name under the lock"]
fn probe_is_candle_on_every_f32() -> Result<()> {
    let dev = metal();
    let act = MoeAct::new(&dev)?;
    let chunk = 1u64 << 24;
    let mut bad = [0usize; 3];
    let mut first = vec![];
    let t = std::time::Instant::now();
    let mut base = 0u64;
    while base < 1u64 << 32 {
        let xs: Vec<f32> = (base..base + chunk).map(|b| f32::from_bits(b as u32)).collect();
        let (b, f) = probe_against_candle(&act, &dev, xs)?;
        for i in 0..3 {
            bad[i] += b[i];
        }
        if first.len() < 12 {
            first.extend(f);
        }
        base += chunk;
    }
    println!(
        "probe on every f32 value (2^32) in {:.0} s: differing softcap gate {}, sigmoid {}, softcap up {}",
        t.elapsed().as_secs_f64(),
        bad[0],
        bad[1],
        bad[2]
    );
    for l in first.iter().take(12) {
        println!("  {l}");
    }
    assert_eq!(bad, [0, 0, 0], "the factors are not candle's bits on every value");
    Ok(())
}

/// candle's composite as `resident.rs` writes it, f16 out.
fn candle_act(gu: &Tensor) -> Result<Vec<u16>> {
    let (g, u) = (gu.get(0)?, gu.get(1)?);
    let act = softcap(&g, CAP_GATE)?.mul(&candle_nn::ops::sigmoid(&g)?)?.mul(&softcap(&u, CAP_UP)?)?.contiguous()?;
    Ok(act.to_dtype(DType::F16)?.flatten_all()?.to_vec1::<half::f16>()?.iter().map(|h| h.to_bits()).collect())
}

fn kernel_act(act: &MoeAct, gu: &Tensor) -> Result<Vec<u16>> {
    let out = act.forward(gu, CAP_GATE, CAP_UP)?;
    assert_eq!(out.dtype(), DType::F16, "the rotation reads f16");
    Ok(out.flatten_all()?.to_vec1::<half::f16>()?.iter().map(|h| h.to_bits()).collect())
}

fn same16(a: u16, b: u16) -> bool {
    let nan = |x: u16| x & 0x7c00 == 0x7c00 && x & 0x03ff != 0;
    a == b || (nan(a) && nan(b))
}

/// The whole activation, f16 out, against candle's composite: 64 draws at the
/// shape of a call (8 routed rows of 1,152) at four scales, and every pair of
/// special values.
#[test]
fn act_is_candle_on_pairs() -> Result<()> {
    let dev = metal();
    let act = MoeAct::new(&dev)?;
    let (rows, inter) = (8usize, 1152usize);
    let mut rng = Rng(0xAC7_0002);
    let mut compared = 0usize;
    for draw in 0..64 {
        let scale = [0.3f32, 1.0, 4.0, 30.0][draw % 4];
        let v: Vec<f32> = (0..2 * rows * inter).map(|_| scale * rng.normal()).collect();
        let gu = Tensor::from_vec(v.clone(), (2, rows, inter), &dev)?;
        let (got, want) = (kernel_act(&act, &gu)?, candle_act(&gu)?);
        assert!(want.iter().any(|&b| b & 0x7fff != 0), "draw {draw}: zeros would pass every equality");
        if let Some(i) = (0..got.len()).find(|&i| !same16(got[i], want[i])) {
            panic!(
                "draw {draw} element {i}: gate {:e} up {:e}: kernel {:#06x}, candle {:#06x}",
                v[i],
                v[rows * inter + i],
                got[i],
                want[i]
            );
        }
        compared += got.len();
    }
    let s = specials();
    let n = s.len();
    let mut v = Vec::with_capacity(2 * n * n);
    for &g in &s {
        v.extend(std::iter::repeat_n(g, n));
    }
    for _ in 0..n {
        v.extend_from_slice(&s);
    }
    let gu = Tensor::from_vec(v.clone(), (2, n, n), &dev)?;
    let (got, want) = (kernel_act(&act, &gu)?, candle_act(&gu)?);
    if let Some(i) = (0..got.len()).find(|&i| !same16(got[i], want[i])) {
        panic!("special pair {i}: gate {:e} up {:e}: kernel {:#06x}, candle {:#06x}", v[i], v[n * n + i], got[i], want[i]);
    }
    compared += got.len();
    println!("activation: {compared} pairs, f16 out, candle's bits");
    Ok(())
}

/// The kernel refuses what it cannot run: a shape that is not `[2, rows, inter]`.
#[test]
fn act_refuses_another_shape() -> Result<()> {
    let dev = metal();
    let act = MoeAct::new(&dev)?;
    let x = Tensor::zeros((3, 4, 5), DType::F32, &dev)?;
    assert!(act.forward(&x, CAP_GATE, CAP_UP).is_err());
    Ok(())
}
