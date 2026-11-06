//! `kernels/attn_dec.metal` against the canonical replay (`attn::canon`, the
//! text of `llvq-cuda/src/attn_ref.rs`), every output bit equal.
//!
//! The card's kernel returns the replay's bits (`llvq-cuda/tests/
//! attn_dec_matches_rust.rs`), so equality here makes the Metal kernel the
//! card kernel's twin in output, whatever their mappings.
//!
//! The lengths cross every boundary the arithmetic has: a tile (8), a unit
//! (64), a split (256), several splits, the served threshold (1,280) and the
//! longest context measured on Metal (4,096). The rows are a decode row and
//! verify calls up to KMAX. K and V are views of a cache with spare capacity,
//! as the Grow store hands them.
//!
//! Mutation run: `docs/mesures/metal-nuit-2026-11-06.txt`.

#![cfg(all(target_os = "macos", feature = "metal"))]

use candle_core::Device;
use llvq_llm::attn_metal::{reference, MetalAttn};

fn kernel() -> (Device, MetalAttn) {
    let dev = Device::new_metal(0).expect("a Metal device");
    let k = MetalAttn::new(&dev, 8, 4).expect("compiles and passes its own self-check");
    (dev, k)
}

fn gate(k: &MetalAttn, dev: &Device, n_kv: usize, groups: usize, len: usize, rows: usize, seed: u64) {
    let want = reference(n_kv, groups, len, rows, seed);
    let got = k.attend_fixture(dev, n_kv, groups, len, rows, seed).expect("the kernel runs");
    assert_eq!(got.len(), want.len());
    if let Some(i) = got.iter().zip(&want).position(|(a, b)| a != b) {
        panic!(
            "n_kv {n_kv} groups {groups} len {len} rows {rows}: output {i} (head {}, row {}) is {:#06x}, the replay says {:#06x}",
            i / (rows * 128),
            (i / 128) % rows,
            got[i],
            want[i]
        );
    }
    assert!(got.iter().any(|b| *b != 0), "zeros would pass every equality");
}

/// The 4B's geometry, a decode row, every boundary up to 1,281.
#[test]
fn a_decode_row_matches_the_replay_across_every_boundary() {
    let (dev, k) = kernel();
    for (i, len) in [1usize, 2, 7, 8, 9, 63, 64, 65, 127, 255, 256, 257, 511, 512, 513, 777, 1279, 1280, 1281]
        .into_iter()
        .enumerate()
    {
        gate(&k, &dev, 8, 4, len, 1, 0xA77E_0000 + i as u64);
    }
}

/// Verify calls of 2 to 8 rows, whose rows end in different tiles, units and
/// splits of one call.
#[test]
fn verify_rows_match_the_replay() {
    let (dev, k) = kernel();
    for (i, (len, rows)) in [(8usize, 8usize), (9, 2), (66, 3), (260, 8), (300, 5), (1024, 8)].into_iter().enumerate() {
        gate(&k, &dev, 8, 4, len, rows, 0xA77E_1000 + i as u64);
    }
}

/// The long contexts M4 measured, 16 splits at 4,096.
#[test]
fn long_contexts_match_the_replay() {
    let (dev, k) = kernel();
    gate(&k, &dev, 8, 4, 2048, 1, 0xA77E_2000);
    gate(&k, &dev, 8, 4, 4096, 1, 0xA77E_2001);
    gate(&k, &dev, 8, 4, 4096, 8, 0xA77E_2002);
}

/// Another geometry: the 8B and 14B have 8 KV heads too, so this is the
/// generality check, 2 KV heads of 8 query heads each.
#[test]
fn another_head_geometry_matches_the_replay() {
    let (dev, k) = kernel();
    gate(&k, &dev, 2, 8, 333, 4, 0xA77E_3000);
}

/// The same call twice gives the same bits: the combine reads no state a
/// concurrent threadgroup could still be writing.
#[test]
fn repeated_calls_give_the_same_bits() {
    let (dev, k) = kernel();
    let a = k.attend_fixture(&dev, 8, 4, 3000, 8, 0xA77E_4000).unwrap();
    for _ in 0..8 {
        assert_eq!(k.attend_fixture(&dev, 8, 4, 3000, 8, 0xA77E_4000).unwrap(), a);
    }
}

/// `attn::kernel_for` hands out this kernel on a Metal device.
#[test]
fn kernel_for_builds_the_metal_kernel() {
    let dev = Device::new_metal(0).expect("a Metal device");
    let k = llvq_llm::attn::kernel_for(&dev, 8, 4, 128).expect("built and checked");
    assert_eq!(k.name(), "attn_dec.metal");
    assert!(llvq_llm::attn::kernel_for(&dev, 8, 4, 64).is_err(), "head dim 64 is refused");
}
