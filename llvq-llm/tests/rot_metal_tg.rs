//! The threadgroup-staged rotation (`+r`, `rot_apply_tg_metal`) against the
//! served one (`rot_apply_metal`), reached through candle, every f32 equal:
//! the same sign flips, Walsh-Hadamard stages and mix, staged in threadgroup
//! memory instead of device memory. The widths are the sealed 4B's rotated
//! inputs (2,560: q, k, v, gate, up; 9,728: down, past the threadgroup wall,
//! so it must keep the served kernel) and a power of two (no mix step).

#![cfg(all(target_os = "macos", feature = "metal"))]

use candle_core::{DType, Device, Tensor};
use llvq_bench::f1::rank::{branch_words, prefix_bytes, suffix_bytes, RankTable};
use llvq_bench::f1::Trellis;
use llvq_core::SplitMix64;
use llvq_llm::fused::RotationTables;
use llvq_llm::fused_metal::{MetalRuntime, MetalTuning};

fn runtime(dev: &Device) -> MetalRuntime {
    let table = RankTable::build();
    let tr = Trellis::new();
    let invnorm = vec![0.0f32; 32];
    MetalRuntime::new(dev, &table.rows, &prefix_bytes(&tr), &branch_words(&tr), &suffix_bytes(&tr), &invnorm, 64, false)
        .expect("the runtime builds")
}

fn bits(t: &Tensor) -> Vec<u32> {
    t.flatten_all().unwrap().to_vec1::<f32>().unwrap().iter().map(|v| v.to_bits()).collect()
}

#[test]
fn the_staged_rotation_returns_the_served_bits() {
    let dev = Device::new_metal(0).expect("a Metal device");
    let mut rt = runtime(&dev);
    for (i, n) in [2560usize, 2048, 9728].into_iter().enumerate() {
        let tables = RotationTables::build(n, 0x0B57_0000 + i as u64).expect("tables");
        let rot = rt.upload_rotation(&tables).expect("uploads");
        let mut rng = SplitMix64::new(0x0B57_1000 + i as u64);
        // An f16 activation at a non-zero offset, as a decode step's view can be.
        let xs: Vec<half::f16> = (0..2 * n).map(|_| half::f16::from_f32(rng.next_gaussian() as f32)).collect();
        let x = Tensor::from_vec(xs, (2, n), &dev).unwrap().narrow(0, 1, 1).unwrap();
        rt.set_tuning(MetalTuning::default()).unwrap();
        let want = rt.rotate(&rot, "000.q_proj", &x).expect("served rotation");
        assert_eq!(want.dtype(), DType::F32);
        let want = bits(&want);
        assert!(want.iter().any(|v| *v != 0), "zeros would pass every equality");
        rt.set_tuning(MetalTuning { rot_tg: true, ..Default::default() }).unwrap();
        let got = bits(&rt.rotate(&rot, "000.q_proj", &x).expect("staged rotation"));
        assert_eq!(got.iter().zip(&want).position(|(a, b)| a != b), None, "width {n}");
    }
    // 2,560 and 2,048 fit and took the staged kernel; 9,728 did not fit.
    assert_eq!(rt.compiled_pipelines(), vec!["rot_apply_metal", "rot_apply_tg_metal"]);
}
