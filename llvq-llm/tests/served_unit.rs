//! The served translation unit compiles — assembled the way the runtime
//! assembles it, on a machine with no card.
//!
//! ## The gap this closes
//!
//! `bin/cuhcheck` sweeps `llvq-cuda/kernels` and stops there, so the kernels
//! this crate ships — `tv_tetra48_h.cu`, `tv_q4_h.cu`, `tv_planes_h.cu` — were
//! parsed by nothing. `fused_cuda.rs` does not compile here either (it needs
//! nvcc and a Linux toolchain), so the first reader of that concatenation was
//! NVRTC, on a rented card, at the end of a load. That is how `tv_planes_seg_h`
//! was found missing on 2026-09-10, for $0.03.
//!
//! ## Assembled, never included
//!
//! The texts are concatenated exactly as `FusedRuntime::new` concatenates
//! them, and clang++ is given NO include path to either kernel directory.
//! Every source carries `#ifndef X / #include "x.cuh" / #endif`; with the
//! directory on the path a missing entry is silently pulled from disk and the
//! check passes. NVRTC has no filesystem. `bin/cuhcheck` learned that on
//! 2026-09-10 — five mutations, five exit 0 — and the same rule applies here.
//!
//! It is a PARSE. `-DLLVQ_HOST_BUILD=1` drops the inline-PTX branches and the
//! shim's warp primitives return their argument, so nothing executed here is
//! the device's arithmetic. What that arithmetic does is
//! `llvq-cuda/tests/tetra48_matches_rust.rs`'s business.

use llvq_llm::fused::{load_int4_sources, load_planes_sources, planes_source_names, FusedLayout};
use std::process::Command;

/// The host shim plus what a `.cu` expects from CUDA and the shim does not
/// declare. Same list as `bin/cuhcheck`'s, plus the two `tv_q4_h.cu` needs.
const SUPPLEMENT: &str = r#"#include "host_shim.h"
struct float4 { float x, y, z, w; };
static inline unsigned __shfl_xor_sync(unsigned, unsigned v, int, int = 32) { return v; }
static inline void __syncwarp(unsigned = 0xffffffffu) {}
static inline float atomicAdd(float* a, float v) { float o = *a; *a += v; return o; }
static inline float __shfl_down_sync(unsigned, float v, int, int = 32) { return v; }
static inline float __fmul_rn(float a, float b) { return a * b; }
static inline float __fadd_rn(float a, float b) { return a + b; }
"#;

/// The four every unit carries, in `FusedRuntime::new`'s order.
const BASE: [&str; 4] = ["llvq_slot.cuh", "matvec.cu", "llvq_rot.cuh", "rotate.cu"];

/// One source's text, from wherever it is embedded — the two crates hold
/// different halves and neither knows the other's.
fn text(layout: FusedLayout, name: &str) -> String {
    if let Ok(s) = llvq_cuda::embedded_source(name) {
        return s.to_string();
    }
    if name == "tv_q4_h.cu" {
        let (cu, _) = load_int4_sources().expect("the int4 source is embedded");
        return cu;
    }
    let (planes, _) = load_planes_sources(layout).expect("the layout's sources are embedded");
    let names = planes_source_names(layout);
    let i = names
        .iter()
        .position(|n| *n == name)
        .unwrap_or_else(|| panic!("no embedded copy of {name} for {layout:?}"));
    planes[i].clone()
}

/// The concatenation the host would hand NVRTC, minus the `#define`.
fn assembled(layout: FusedLayout, names: &[&str]) -> String {
    names.iter().map(|n| text(layout, n)).collect::<Vec<_>>().join("\n")
}

/// The `extern "C" __global__` entry points a text defines.
///
/// A source nothing else includes — a `.cu` holding only an entry point — can
/// be dropped without a parse error, because no guard fires for it. What
/// disappears then is the SYMBOL, and the host finds that out from the driver
/// at the end of a load. The drop test below accepts either failure.
fn entry_points(src: &str) -> Vec<String> {
    let mut v = Vec::new();
    for (i, _) in src.match_indices("__global__ void ") {
        let rest = &src[i + "__global__ void ".len()..];
        if let Some(end) = rest.find('(') {
            let n = rest[..end].trim();
            if !n.is_empty() && n.chars().all(|c| c.is_alphanumeric() || c == '_') {
                v.push(n.to_string());
            }
        }
    }
    v
}

/// Concatenate, write, and hand it to clang++ with no kernel include path.
fn parses(layout: FusedLayout, tag: &str, names: &[&str]) -> Result<(), String> {
    let mut src = String::from(SUPPLEMENT);
    // The host injects the tile before anything else; `matvec.cu` `#error`s
    // without it.
    src.push_str(&format!("#define TILE_BLOCKS {}u\n", llvq_cuda::TILE_BLOCKS));
    src.push_str(&assembled(layout, names));
    src.push_str("\nint main(){return 0;}\n");

    // 🕳️ The shim is COPIED into the sandbox rather than reached through its
    // own directory, and that is the whole difference between a check and a
    // decoration. Every served source guards its include as
    // `#include "../../llvq-cuda/kernels/x.cuh"` — a RELATIVE path. Pointing
    // `-I` at `llvq-cuda/tests` lets clang++ walk `../..` out of it and back
    // into the kernel tree, so a source dropped from the list is pulled off
    // the disk and the unit parses. Measured here on 2026-09-10: dropping
    // `llvq_tetra48.cuh` passed. It is the same defect `bin/cuhcheck` carried,
    // arriving by a different door.
    let dir = std::env::temp_dir().join(format!("llvq_served_unit_{tag}"));
    std::fs::create_dir_all(&dir).expect("sandbox");
    let shim_src = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("llvq-cuda")
        .join("tests")
        .join("host_shim.h");
    std::fs::copy(&shim_src, dir.join("host_shim.h")).expect("copy the shim");
    let tu = dir.join("unit.cpp");
    std::fs::write(&tu, src).expect("write the translation unit");
    let out = Command::new("c++")
        .args(["-std=c++17", "-fsyntax-only", "-DLLVQ_HOST_BUILD=1", "-I"])
        .arg(&dir)
        .arg(&tu)
        .output()
        .expect("a C++ compiler is on PATH");
    match out.status.success() {
        true => Ok(()),
        false => Err(String::from_utf8_lossy(&out.stderr).to_string()),
    }
}

/// The list the runtime hands NVRTC for the served object of 2026-09-08: the
/// base four, the Tetra chain, and the int4 kernel its `v_proj` needs.
fn served_names() -> Vec<&'static str> {
    let mut v: Vec<&'static str> = BASE.to_vec();
    v.extend(planes_source_names(FusedLayout::Tetra48));
    v.push("tv_q4_h.cu");
    v
}

#[test]
fn the_served_translation_unit_parses() {
    if let Err(e) = parses(FusedLayout::Tetra48, "served", &served_names()) {
        panic!("the served unit does not parse:\n{e}");
    }
}

/// Every layout's unit, so a source added to one and forgotten in another is
/// caught here and not by a job.
#[test]
fn every_layout_s_translation_unit_parses() {
    for layout in [
        FusedLayout::Slot32,
        FusedLayout::Planes14,
        FusedLayout::Planes12x,
        FusedLayout::Golay70,
        FusedLayout::Tetra48,
    ] {
        let mut names: Vec<&'static str> = BASE.to_vec();
        names.extend(planes_source_names(layout));
        let tag = format!("{layout:?}").to_lowercase();
        if let Err(e) = parses(layout, &tag, &names) {
            panic!("the {tag} unit does not parse:\n{e}");
        }
    }
}

/// The served unit carries both Tetra entry points and the int4 one.
///
/// The prefill kernel is ADDITIONAL: `tv_tetra48_h` stays in the unit
/// untouched, so every decode-time number keeps the kernel that produced it.
#[test]
fn the_served_unit_carries_the_kernels_the_runtime_looks_up() {
    let e = entry_points(&assembled(FusedLayout::Tetra48, &served_names()));
    for want in ["rot_apply", "tv_tetra48_h", "tv_tetra48_rows_h", "tv_q4_h"] {
        assert!(e.contains(&want.to_string()), "the served unit defines no `{want}`: {e:?}");
    }
}

/// 🚨 And the check is not decorative: dropping any source must break
/// something.
///
/// This is the property `bin/cuhcheck` did not have until 2026-09-10. With the
/// kernel directory on the include path a missing entry is resolved from disk
/// and the parse succeeds — five such mutations exited 0. Here the texts are
/// concatenated and no path is given, so an unsatisfied guard fires the
/// `#include` and clang++ cannot find the file, exactly as NVRTC cannot.
///
/// A leaf source breaks nothing at parse time and loses its entry point
/// instead, which is what the driver reports at the end of a load. Either
/// counts; neither breaking is the failure.
#[test]
fn dropping_any_source_breaks_the_served_unit() {
    let layout = FusedLayout::Tetra48;
    let names = served_names();
    let full = entry_points(&assembled(layout, &names));
    assert!(full.len() >= 4, "the served unit defines only {full:?}");
    for drop in 0..names.len() {
        let kept: Vec<&str> =
            names.iter().enumerate().filter(|(i, _)| *i != drop).map(|(_, n)| *n).collect();
        let broke = parses(layout, &format!("drop{drop}"), &kept).is_err();
        let left = entry_points(&assembled(layout, &kept));
        let lost = full.iter().any(|e| !left.contains(e));
        assert!(
            broke || lost,
            "dropping `{}` neither breaks the parse nor removes an entry point: \
             the host could forget it and nothing here would say so",
            names[drop]
        );
    }
}
