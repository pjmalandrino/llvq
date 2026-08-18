//! Does the MSL of the P1 arms compile at all?
//!
//! `xcrun metal` is not installed on this machine, but `Kernel::new` hands the
//! source to Metal's **runtime** compiler — which is the compiler that matters,
//! since that is the one the bench will use. This binary exists to get a
//! syntax and semantics verdict in seconds rather than discovering it inside a
//! bench that also allocates 200 MB and sweeps a 981 MB file.
//!
//! **Every entry point is listed, not every file.** A library that builds says
//! nothing about a kernel it contains: a failure is reported per entry point.
//! `binomial_walk.metal` carries two (the arm and its instrumented twin) and
//! `anchors.metal` three, so the list is longer than the directory.
//!
//! `anchors.metal` is not self-contained — it is appended to
//! `llvq_metal::PAYLOAD_MSL`, which defines `DIM`, the cursor, `ClassRec` and
//! `decode_payload`. It is assembled here exactly as `bin/rankbench` assembles
//! it, from the same two pieces, so a check that passed here and failed there
//! would mean the two assemblies had drifted.
//!
//! It proves nothing about correctness. A shader that compiles can still
//! decode garbage; that is what V0 is for.
//!
//! Run: `cargo run --release -p llvq-metal --bin mslcheck`

fn main() {
    let anchors = format!(
        "{}{}",
        llvq_metal::PAYLOAD_MSL,
        include_str!("../../shaders/anchors.metal")
    );
    let cascade_uniform = include_str!("../../shaders/cascade_uniform.metal");
    let cascade_archive = include_str!("../../shaders/cascade_archive.metal");
    let binomial_walk = include_str!("../../shaders/binomial_walk.metal");
    let binomial_block = include_str!("../../shaders/binomial_block.metal");
    let block_flat = include_str!("../../shaders/binomial_block_flat.metal");
    let e1v_flux = include_str!("../../shaders/e1v_flux.metal");

    let cases: [(&str, &str, &str); 10] = [
        ("sol", &anchors, "floor96"),
        ("sol-rang (É3a)", &anchors, "floor_rank"),
        ("masques", &anchors, "decode_f96"),
        ("cascade_archive", cascade_archive, "cascade_archive"),
        ("cascade_uniform", cascade_uniform, "cascade_uniform"),
        ("binomial_walk", binomial_walk, "decode_walk"),
        ("binomial_walk (twin)", binomial_walk, "walk_arrangement"),
        ("marche-bloc (P1b)", binomial_block, "decode_block"),
        ("marche-bloc-plat (É1)", block_flat, "decode_block_flat"),
        // P1c. This one carries a named unknown: `simd_prefix_exclusive_sum`
        // may not exist in the runtime's MSL, and if it does not the arm is
        // written another way and MEASURES SOMETHING ELSE — a serial prefix sum
        // is not a warp-scan (pre-registration §6). This binary is where that
        // is found out, in three seconds rather than inside a bench that also
        // sweeps a 981 MB file.
        ("e1v-flux (P1c)", e1v_flux, "decode_e1v"),
    ];

    let mut bad = 0;
    for (name, src, entry) in cases {
        print!("  {name:<22} ");
        match llvq_metal::Kernel::new(src, entry) {
            Ok(k) => println!(
                "compile — {} , simd {} , max group {}",
                k.device_name(),
                k.simd_width(),
                k.max_threads_per_group()
            ),
            Err(e) => {
                bad += 1;
                println!("ÉCHEC");
                for line in e.lines().take(40) {
                    println!("      {line}");
                }
            }
        }
    }
    if bad > 0 {
        std::process::exit(1);
    }
}
