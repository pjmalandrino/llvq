//! Does the MSL of the P1 arms compile at all?
//!
//! `xcrun metal` is not installed on this machine, but `Kernel::new` hands the
//! source to Metal's **runtime** compiler — which is the compiler that matters,
//! since that is the one the bench will use. This binary exists to get a
//! syntax and semantics verdict in seconds rather than discovering it inside a
//! bench that also allocates 200 MB and sweeps a 981 MB file.
//!
//! It proves nothing about correctness. A shader that compiles can still
//! decode garbage; that is what V0 is for.
//!
//! Run: `cargo run --release -p llvq-metal --bin mslcheck`

fn main() {
    let cases: [(&str, &str, &str); 4] = [
        (
            "cascade_uniform",
            include_str!("../../shaders/cascade_uniform.metal"),
            "cascade_uniform",
        ),
        (
            "cascade_archive",
            include_str!("../../shaders/cascade_archive.metal"),
            "cascade_archive",
        ),
        (
            "binomial_walk",
            include_str!("../../shaders/binomial_walk.metal"),
            "decode_walk",
        ),
        // The instrumented twin (§11) is a separate entry point of the same
        // source, so a library that builds says nothing about it: an entry
        // point that fails to compile is reported per entry point, not per
        // file. Listed here so it cannot rot unnoticed.
        (
            "binomial_walk (twin)",
            include_str!("../../shaders/binomial_walk.metal"),
            "walk_arrangement",
        ),
    ];

    let mut bad = 0;
    for (name, src, entry) in cases {
        print!("  {name:<18} ");
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
