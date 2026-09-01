//! The A3 occupancy kernels, compiled by `clang++`, and their two scalar
//! helpers diffed against the Rust mirrors in `llvq_cuda::occ`.
//!
//! Two things, per the pattern of `planes_decoder_matches_rust.rs`:
//!
//! 1. **The kernel text compiles as host C++** — all seven entry points and
//!    every template instantiation the device build will make. A syntax or
//!    type error costs two seconds here instead of a billed job. It has to
//!    be said what this is not: `__syncthreads`, `warp_sum`, the ticket
//!    atomic and `__syncthreads_or` are shims, so nothing here says the
//!    kernels are *correct*. That is the bench's job, on the card, against
//!    `tv_planes_seg` bit for bit and the f64 reference.
//!
//! 2. **The kernel's arithmetic is the host's arithmetic.** `occ_xs_index`
//!    and `occ_slice` are executed on the CPU over every staged index of a
//!    tile and over every `(nblocks, nsplit, s)` the 4B can produce, and
//!    must return what `occ::xs_index` and `occ::slice` return. The bench
//!    sizes grids, shared memory and slices with the Rust side; the kernel
//!    addresses with the C++ side; a disagreement between the two would
//!    stage the wrong floats or leave a block out, silently.

use llvq_cuda::occ;
use std::io::Write;
use std::process::{Command, Stdio};

fn le_u32(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[test]
fn the_occupancy_kernels_compile_and_their_helpers_match_rust() {
    let out = std::env::temp_dir().join("llvq_host_planes_occ");
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let st = Command::new("clang++")
        .args(["-std=c++17", "-O2", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror"])
        .arg(dir.join("host_planes_occ.cpp"))
        .arg("-o")
        .arg(&out)
        .status()
        .expect("clang++ is on PATH");
    assert!(st.success(), "the A3 occupancy CUDA sources do not compile as host C++");

    // ---- fixture ----
    // Every staged index of a full tile, then the slice table of the four
    // fused widths at both split factors plus adversarial widths.
    let idx: Vec<u32> = (0..(occ::XS_DIM * llvq_cuda::TILE_BLOCKS) as u32).collect();
    let mut slices: Vec<(u32, u32, u32)> = Vec::new();
    let widths = [106u32, 170, 405, 1, 2, 7, 9, 31, 32, 33, 127, 128, 129, 255, 256, 257, 1000];
    for &nb in &widths {
        for factor in 1..=2 {
            let ns = occ::sk_nsplit(nb, factor);
            for s in 0..ns {
                slices.push((nb, ns, s));
            }
        }
        let ns = nb + 3;
        for s in 0..ns {
            slices.push((nb, ns, s));
        }
    }
    let mut fixture = Vec::new();
    fixture.extend_from_slice(&le_u32(&[idx.len() as u32, slices.len() as u32]));
    fixture.extend_from_slice(&le_u32(&idx));
    let flat: Vec<u32> = slices.iter().flat_map(|&(a, b, c)| [a, b, c]).collect();
    fixture.extend_from_slice(&le_u32(&flat));

    let mut child = Command::new(&out)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("host probe runs");
    child.stdin.take().expect("stdin").write_all(&fixture).expect("fixture written");
    let res = child.wait_with_output().expect("host probe finishes");
    assert!(res.status.success(), "host probe failed");

    let got: Vec<u32> = res
        .stdout
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(got.len(), idx.len() * 2 + slices.len() * 2, "the probe wrote the wrong count");

    // ---- the comparison ----
    for (k, &i) in idx.iter().enumerate() {
        assert_eq!(got[k], occ::xs_index(i as usize, occ::XS_DIM) as u32, "xs_index<24>({i})");
        assert_eq!(
            got[idx.len() + k],
            occ::xs_index(i as usize, occ::XS_PAD) as u32,
            "xs_index<28>({i})"
        );
    }
    let base = idx.len() * 2;
    for (k, &(nb, ns, s)) in slices.iter().enumerate() {
        let (lo, hi) = occ::slice(nb, ns, s);
        assert_eq!((got[base + 2 * k], got[base + 2 * k + 1]), (lo, hi), "slice({nb}, {ns}, {s})");
    }
}
