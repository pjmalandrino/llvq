// DRAFT, never compiled or run. Wave 5 deep dive, lead "mac", 2026-11-02.
//
// Proposed as `llvq-llm/tests/q4_metal_candle.rs`: the int4 kernels reached
// through candle, tiled and untiled, against the host lane-order reference.
//
// What this proves that the shader tests do not. `llvq-metal/tests/
// q4_matches_host.rs` and `q4_tiled_matches_served.rs` drive the MSL through
// metal-rs, which the shipped path cannot use: candle reaches Metal through
// objc2-metal. The q4_matches_host header says so ("Nothing runs through
// candle"). `tetra48_metal_candle.rs` closes that gap for Tetra; nothing closes
// it for int4 today, tiled or untiled. What is left unproved is the BINDING:
// buffer 7, the threadgroup length, the routing between the two kernels, the
// activation offset, the output allocation, the pipeline map.
//
// Why not "same op on two devices" as for Tetra: `Int4Op::cpu_fwd` bails
// (fused_metal.rs:1016-1018), so the reference is the host lane-order one of
// q4_matches_host.rs, built from `Int4Matrix::to_f32`. If the branch prefers
// to give `Int4Op` a CPU arm under `keep_host`, this file becomes the Tetra
// file's shape and the reference moves into the op.
//
// Written against the API proposed in code/mac/fused_metal_wall.rs:
// `MetalRuntime::upload_int4_tiled(.., tile)` and `MetalInt4Proj::tile_cols()`.
//
// Mutants of fused_metal.rs this file must kill:
//   - buffer 7 bound to `tile / 2`            -> claim 1 (lanes interleave differently)
//   - the routing inverted for d_in <= 8192    -> the tile_cols assertion (the bits
//     would not move: the tiled kernel equals the untiled one by construction)
//   - the activation offset bound in elements  -> claim 2
//   - "tv_q4_metal_tiled" missing from source_of -> claim 1 (no pipeline)
//   - threadgroup length `d_in * 4` on the tiled path -> claim 1 (Metal refuses
//     a length above 32,768 B; the test fails or the process aborts, which
//     still counts as killed and is recorded as such)
// Not planted: buffer 7 unbound or a zero tile, which loop forever on the GPU.

#![cfg(all(target_os = "macos", feature = "metal"))]

use candle_core::{DType, Device, Tensor};
use llvq_artifact::Int4Matrix;
use llvq_llm::fused_metal::MetalRuntime;

const TETRA_TILE: usize = 64;

// plain_runtime(dev, tile): verbatim from tetra48_metal_candle.rs (the int4
// arm needs a runtime, and a runtime needs decoder tables).
fn plain_runtime(_dev: &Device, _tile: usize) -> MetalRuntime {
    unimplemented!("copy from tetra48_metal_candle.rs::plain_runtime")
}
// fixture and reference: verbatim from llvq-metal/tests/q4_matches_host.rs,
// with the activation drawn as f16 bit patterns so the widened f32 is the
// model's.
fn fixture(_seed: u64, _d_out: usize, _d_in: usize) -> (Int4Matrix, Vec<f32>) {
    unimplemented!("copy from q4_matches_host.rs::fixture")
}
fn reference(_m: &Int4Matrix, _x: &[f32]) -> Vec<f32> {
    unimplemented!("copy from q4_matches_host.rs::reference")
}

fn words(m: &Int4Matrix) -> Vec<u32> {
    m.packed.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn device() -> Option<Device> {
    Device::new_metal(0).ok()
}

fn same_bits(got: &[f32], want: &[f32], what: &str) {
    if let Some((i, (g, w))) = got.iter().zip(want).enumerate().find(|(_, (g, w))| g.to_bits() != w.to_bits()) {
        panic!("{what}: row {i} reads {g} against {w}");
    }
}

/// Claim 1: the six widths of the sealed files, through the candle binding.
#[test]
fn the_binding_matches_the_host_at_the_six_widths() {
    let Some(dev) = device() else { panic!("no Metal device: this test refuses to skip") };
    let rt = plain_runtime(&dev, TETRA_TILE);
    let shapes = [
        (21u64, 1024usize, 2560usize, false),
        (22, 2560, 4096, false),
        (23, 5120, 5120, false),
        (24, 2560, 9728, true),
        (25, 4096, 12288, true),
        (26, 5120, 17408, true),
    ];
    for (seed, d_out, d_in, past_the_wall) in shapes {
        let (m, x) = fixture(seed, d_out, d_in);
        let want = reference(&m, &x);
        for tile in [2048usize, 8192] {
            let proj = rt
                .upload_int4_tiled(&m.name, d_out, d_in, m.groups_per_row(), &words(&m), &m.scales, &m.biases, tile)
                .expect("uploads");
            // The routing, read from the projection: a tiled kernel on a record
            // that fits would give the same bits, so only this line catches it.
            assert_eq!(proj.tile_cols(), past_the_wall.then_some(tile), "{d_out}x{d_in} routed wrong");
            let xt = Tensor::from_slice(&x, (1, d_in), &dev).unwrap();
            let got: Vec<f32> = rt.matvec_int4(&proj, &xt).unwrap().flatten_all().unwrap().to_vec1().unwrap();
            same_bits(&got, &want, &format!("{d_out}x{d_in} at tile {tile}"));
        }
    }
}

/// Claim 2: the activation is bound at its offset, in bytes. A narrowed row of
/// a two-row tensor starts `d_in` floats into its buffer.
#[test]
fn the_activation_is_read_at_its_offset() {
    let Some(dev) = device() else { panic!("no Metal device: this test refuses to skip") };
    let rt = plain_runtime(&dev, TETRA_TILE);
    for (seed, d_out, d_in) in [(31u64, 2560usize, 4096usize), (32, 2560, 9728)] {
        let (m, x) = fixture(seed, d_out, d_in);
        let (_, decoy) = fixture(seed + 100, d_out, d_in);
        let proj = rt
            .upload_int4_tiled(&m.name, d_out, d_in, m.groups_per_row(), &words(&m), &m.scales, &m.biases, 2048)
            .unwrap();
        let both: Vec<f32> = decoy.iter().chain(&x).copied().collect();
        let two = Tensor::from_slice(&both, (2, d_in), &dev).unwrap();
        let got: Vec<f32> = rt
            .matvec_int4(&proj, &two.narrow(0, 1, 1).unwrap())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        same_bits(&got, &reference(&m, &x), &format!("{d_out}x{d_in}, second row"));
    }
}

/// Claim 3: the model hands the int4 arm an f16 activation, which
/// `matvec_int4` widens (fused_metal.rs:997-1003). The widened values are the
/// reference's input.
#[test]
fn an_f16_activation_is_widened_exactly() {
    let Some(dev) = device() else { panic!("no Metal device: this test refuses to skip") };
    let rt = plain_runtime(&dev, TETRA_TILE);
    let (m, x) = fixture(41, 2560, 9728);
    let proj = rt
        .upload_int4_tiled(&m.name, m.d_out, m.d_in, m.groups_per_row(), &words(&m), &m.scales, &m.biases, 2048)
        .unwrap();
    let x16 = Tensor::from_slice(&x, (1, m.d_in), &dev).unwrap().to_dtype(DType::F16).unwrap();
    let widened: Vec<f32> = x16.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(widened, x, "the fixture's activation must be f16-representable");
    let got: Vec<f32> = rt.matvec_int4(&proj, &x16).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    same_bits(&got, &reference(&m, &widened), "f16 activation");
}

/// Claim 4: a bad tile is refused by name when the upload starts, whether or
/// not the record needs a tile.
#[test]
fn a_bad_tile_is_refused_by_name() {
    let Some(dev) = device() else { panic!("no Metal device: this test refuses to skip") };
    let rt = plain_runtime(&dev, TETRA_TILE);
    for (d_out, d_in) in [(1024usize, 2560usize), (2560, 9728)] {
        let (m, _) = fixture(51, d_out, d_in);
        for bad in [0usize, 300, 16384] {
            let e = rt
                .upload_int4_tiled(&m.name, d_out, d_in, m.groups_per_row(), &words(&m), &m.scales, &m.biases, bad)
                .err()
                .unwrap_or_else(|| panic!("tile {bad} accepted for {d_out}x{d_in}"))
                .to_string();
            assert!(e.contains(&bad.to_string()), "{e}");
        }
    }
}
