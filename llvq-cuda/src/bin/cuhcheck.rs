//! Do the CUDA headers of this crate parse at all?
//!
//! The twin of `llvq-metal`'s `bin/mslcheck`, and it exists for the same reason
//! that one does: to get a verdict in seconds on a machine that cannot run the
//! thing, rather than discovering a typo at job start on a rented card.
//!
//! Until now this repository declared that impossible for the *headers*.
//! `llvq_e1c.cuh` says so in its own header: *"Nothing in this file has been
//! compiled. […] It cannot catch a syntax error, a launch configuration, a
//! register spill or a race. Those surface at job start."* The last three
//! still do. The first does not have to.
//!
//! **And it was already half-solved.** `tests/host_shim.h` turns the CUDA
//! keywords into ordinary C++ so that `tests/*_decoder_matches_rust.rs` can
//! **compile and run** the kernel text on this Mac, a stronger check than
//! this one, and it covers the four kernels that have a probe. What it does
//! not cover is a header nobody wrote a probe for, which is where `llvq_e1c.cuh`
//! sat and where `llvq_e1v.cuh` would sit. This binary is the cheap sweep over
//! **all** of them; a probe remains the real thing wherever one exists.
//!
//! It reuses `tests/host_shim.h` rather than carrying a second one. The few
//! intrinsics that shim leaves out are supplemented **in the generated
//! translation unit**, not hoisted into it: two probes already define
//! `atomicAdd` locally (`host_planes12.cpp`, `host_golay70.cpp`, both saying
//! "host_shim.h does not provide"), and a third definition in the shared header
//! would collide with theirs.
//!
//! ## What a green line here means, and what it does not
//!
//! It means the header is well-formed C++: balanced braces, declared
//! identifiers, agreeing types, no call to a function nobody wrote.
//!
//! It does **not** mean the kernel is correct. The shim's
//! `__shfl_up_sync` returns its own argument, so anything built on a warp scan
//! parses beautifully and would compute garbage if run. Correctness is the
//! mirror test's statement, against the Rust decoder.
//!
//! Nor does it mean nvcc will accept it. Clang's C++ is not CUDA C++: a
//! header that fails here would certainly fail nvcc, one that passes may still
//! fail it. This is a filter, not a compiler, with the same standing this crate's
//! own comments give `preflight`.
//!
//! Run: `cargo run --release -p llvq-cuda --bin cuhcheck`

use std::io::Write;
use std::process::Command;

/// What `tests/host_shim.h` leaves out, supplied per translation unit.
///
/// Every entry is here because some kernel calls it and the shim does not
/// declare it. None of them is a semantics: `__shfl_up_sync` returns its own
/// argument, so a warp scan built on it parses and would compute garbage. This
/// is a parse, and the file that says what a parse is worth is this binary's
/// own header.
const SUPPLEMENT: &str = r#"#include "host_shim.h"
struct float4 { float x, y, z, w; };
static inline unsigned __shfl_xor_sync(unsigned, unsigned v, int, int = 32) { return v; }
static inline void __syncwarp(unsigned = 0xffffffffu) {}
static inline float atomicAdd(float* a, float v) { float o = *a; *a += v; return o; }
"#;

/// The units, each a list of sources concatenated **in the order the host
/// concatenates them**, and what each is for.
///
/// Most headers stand alone. `llvq_floor.cuh` does not: it uses `TILE_BLOCKS`
/// and the tiling `matvec.cu` defines, so it is assembled here from the very
/// list `bin/matvec` and `bin/graphbench` hand to `load_sources_many`. Same
/// discipline as `mslcheck`, whose `anchors.metal` is appended to
/// `PAYLOAD_MSL`: a unit that passed here and failed there would mean the two
/// assemblies had drifted, which is a thing worth learning.
///
/// The `.cu` files carry `#ifndef` guards that pull their dependencies from
/// disk. `planes.cu` says so in its own header, *"and only resolve from disk
/// under a host clang++ syntax check"*. They need no list of their own.
const UNITS: [(&str, &[&str], &str); 15] = [
    ("llvq_slot.cuh", &["llvq_slot.cuh"], "Slot32, the fallback layout"),
    ("llvq_planes.cuh", &["llvq_planes.cuh"], "Planes14, the served layout"),
    ("llvq_planes12.cuh", &["llvq_planes12.cuh"], "Planes12x, the sparse overlay"),
    ("llvq_golay.cuh", &["llvq_golay.cuh"], "Golay70, dropped, kept as a curve point"),
    (
        "llvq_floor.cuh",
        &["llvq_slot.cuh", "matvec.cu", "llvq_floor.cuh"],
        "the floor, assembled the way bin/matvec assembles it",
    ),
    ("llvq_rot.cuh", &["llvq_rot.cuh"], "the incoherence rotation"),
    ("llvq_e1c.cuh", &["llvq_e1c.cuh"], "E1c, transposed on the warp"),
    ("llvq_e1v.cuh", &["llvq_e1v.cuh"], "E1v, the CNS, row-aligned (P1c)"),
    ("matvec.cu", &["matvec.cu"], "the fused matvec, Slot32"),
    ("planes.cu", &["planes.cu"], "the fused matvec, Planes14"),
    ("planes12.cu", &["planes12.cu"], "the fused matvec, Planes12x"),
    ("rotate.cu", &["rotate.cu"], "the rotation, outside the loop"),
    ("e1v.cu", &["e1v.cu"], "the fused matvec, E1v row-aligned (P1c)"),
    ("nullk.cu", &["nullk.cu"], "the floor: same pass, no weight read (P4)"),
    // A3. Its `#ifndef` guards pull matvec.cu and llvq_planes.cuh from disk;
    // the seven entry points instantiate every template the device build will.
    ("planes_occ.cu", &["planes_occ.cu"], "A3: the occupancy variants of the fused Planes14"),
];

fn main() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let dir = root.join("kernels");
    let shim_dir = root.join("tests");
    let shim = shim_dir.join("host_shim.h");
    assert!(shim.exists(), "tests/host_shim.h is missing: {}", shim.display());

    // The compiler is a hard requirement, not an optional nicety. A run that
    // "skipped because no compiler" would print the same green as a run that
    // checked, which is the failure mode the dossier's §5 is about.
    let cc = std::env::var("CXX").unwrap_or_else(|_| "c++".to_string());
    let probe = Command::new(&cc).arg("--version").output();
    match probe {
        Ok(o) if o.status.success() => {
            let v = String::from_utf8_lossy(&o.stdout);
            println!("compiler: {} ({})", cc, v.lines().next().unwrap_or(""));
        }
        _ => {
            eprintln!(
                "no C++ compiler at `{cc}`. This binary does not know how to degrade \
                 quietly:\na green line with no check would be worse than no line at \
                 all.\nInstall the Command Line Tools, or point $CXX."
            );
            std::process::exit(1);
        }
    }
    println!("shim: {}\n", shim.display());

    let tmp = std::env::temp_dir().join(format!("llvq-cuhcheck-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).expect("temp dir");

    let mut bad = 0;
    for (label, sources, what) in UNITS {
        print!("  {label:<20} ");
        std::io::stdout().flush().ok();
        let tu = tmp.join(format!("{label}.cpp"));
        let mut src = String::from(SUPPLEMENT);
        for s in sources {
            src.push_str(&format!("#include \"{s}\"\n"));
        }
        src.push_str("int main(){return 0;}\n");
        std::fs::write(&tu, src).expect("write translation unit");

        let out = Command::new(&cc)
            // `LLVQ_HOST_BUILD` is the crate's OWN host mode, not an
            // invention of this binary: `llvq_rot.cuh` and `matvec.cu` both
            // branch on it to replace an inline-PTX `cvt.f32.f16` with a
            // software widening. Without it a host parser stops on the asm
            // constraints. It also means the PTX branches are NOT parsed
            // here, by anything, ever, short of nvcc.
            .args(["-std=c++17", "-fsyntax-only", "-DLLVQ_HOST_BUILD=1", "-I"])
            .arg(&shim_dir)
            // The tiling `matvec.cu` `#error`s without, read from the one place
            // it is defined rather than retyped here.
            .arg(format!("-DTILE_BLOCKS={}u", llvq_cuda::TILE_BLOCKS))
            .arg("-I")
            .arg(&dir)
            .arg(&tu)
            .output()
            .expect("run the compiler");
        if out.status.success() {
            println!("parse: {what}");
        } else {
            bad += 1;
            println!("FAIL:  {what}");
            for line in String::from_utf8_lossy(&out.stderr).lines().take(20) {
                println!("      {line}");
            }
        }
    }

    let _ = std::fs::remove_dir_all(&tmp);
    println!(
        "\nWARNING: a parse is not an nvcc compile, and above all it is no proof of \
         correctness.\n   The shim makes `__shfl_up_sync` return its own argument, and \
         -DLLVQ_HOST_BUILD drops the PTX\n   branches, which nothing here reads. Correctness \
         is the mirror test's business, against the\n   Rust decoder; execution is a card's."
    );
    if bad > 0 {
        std::process::exit(1);
    }
}
