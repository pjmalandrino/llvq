//! The L2 bench's copy of the served NVRTC unit is the served door's.
//!
//! `llvq_cuda::pers::served_unit_text` rebuilds the 4B's served unit inside
//! `planesbench`, so its `tv_tetra48_h` is the reference the persistent arms
//! owe their bits to. The served door assembles the same unit in
//! `fused_cuda::FusedRuntime::new` from the loaders below, in this order:
//! the defines, llvq_slot/matvec/llvq_rot/rotate, the Tetra48 layout's parts,
//! the q4 embedding, the int4 matvec. If either side moves, this fails here
//! rather than as a digest mismatch on a rented card.

// `llvq-cuda` is a dev-dependency on macOS only (see `llvq-llm/Cargo.toml`): on a
// Linux CPU build it would pull `cudarc`, whose build script needs `nvcc`.
// Under `--features cuda` it is a normal dependency, so the file runs there.
#![cfg(any(target_os = "macos", feature = "cuda"))]

use llvq_llm::fused::{load_emb_q4_sources, load_int4_sources, load_planes_sources, FusedLayout};

#[test]
fn the_bench_copy_is_the_served_unit() {
    assert!(std::env::var("LLVQ_KERNEL_DIR").is_err(), "this test reads the embedded sources only");
    // `llvq_cuda::load_sources_many` is Linux-only; its embedded texts are
    // these four files, read here from the same paths.
    let base = [
        include_str!("../../llvq-cuda/kernels/llvq_slot.cuh"),
        include_str!("../../llvq-cuda/kernels/matvec.cu"),
        include_str!("../../llvq-cuda/kernels/llvq_rot.cuh"),
        include_str!("../../llvq-cuda/kernels/rotate.cu"),
    ];
    let (planes, _) = load_planes_sources(FusedLayout::Tetra48).unwrap();
    let (emb, _) = load_emb_q4_sources().unwrap();
    let (int4, _) = load_int4_sources().unwrap();
    let defines = llvq_cuda::pers::SERVED_DEFINES_SM89;
    let mut parts: Vec<&str> = vec![defines];
    parts.extend(base);
    parts.extend(planes.iter().map(String::as_str));
    parts.push(emb.as_str());
    parts.push(int4.as_str());
    let door = parts.join("\n");
    let bench = llvq_cuda::pers::served_unit_text(defines);
    assert_eq!(door.len(), bench.len(), "the two assemblies differ in length");
    assert!(door == bench, "the bench's served unit is not the served door's");
}
