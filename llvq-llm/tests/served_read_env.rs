//! `Served::read` against the real process environment.
//!
//! `check_env` and `Served::of` are tested on closures in `served_config.rs`.
//! This file tests the one function that chains them on the shell every
//! binary runs under: a variable that contradicts the file, or that a served
//! run refuses beside it, must be refused through `Served::read` itself.
//!
//! It mutates the process environment, so it holds a single test and is its
//! own test binary: no other test can run beside it and read the variables
//! it sets.

use llvq_llm::fused::EmbedMode;
use llvq_llm::served::{Served, BELOW_THE_DOOR, MEASUREMENT_MODES};

/// Every variable `check_env` consults, cleared so the verdict does not
/// depend on what the developer's shell exports.
fn clear() {
    let spoken = ["LLVQ_FUSED_LAYOUT", "LLVQ_EMBED", "LLVQ_ROT_SHARE", "LLVQ_FUSE", "LLVQ_KV"];
    for v in spoken.iter().chain(&MEASUREMENT_MODES).chain(&BELOW_THE_DOOR) {
        std::env::remove_var(v);
    }
}

#[test]
fn served_read_refuses_what_the_environment_contradicts() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../configs/qwen3-4b-tetra-e4-4e-v9.json");
    clear();
    let s = Served::read(&path).expect("a clean environment reads the served config");
    assert_eq!(s.embed, EmbedMode::Q4);

    // A variable that agrees with the file is accepted.
    std::env::set_var("LLVQ_EMBED", "q4");
    Served::read(&path).expect("LLVQ_EMBED=q4 agrees with the file");

    // One that contradicts it is refused by name.
    std::env::set_var("LLVQ_EMBED", "q8");
    let e = Served::read(&path).expect_err("LLVQ_EMBED=q8 contradicts the file");
    assert!(e.contains("LLVQ_EMBED") && e.contains("contradicts"), "{e}");
    clear();

    // A variable below the served door is refused beside any config.
    std::env::set_var("LLVQ_TILE_BLOCKS", "64");
    let e = Served::read(&path).expect_err("LLVQ_TILE_BLOCKS beside LLVQ_CONFIG");
    assert!(e.contains("LLVQ_TILE_BLOCKS"), "{e}");
    clear();

    // A measurement mode is refused beside a config that names 4E.
    std::env::set_var("LLVQ_PERS_4E", "1");
    let e = Served::read(&path).expect_err("LLVQ_PERS_4E beside the persistent decode");
    assert!(e.contains("LLVQ_PERS_4E") && e.contains("measurement mode"), "{e}");
    clear();

    Served::read(&path).expect("the environment is clean again");
}
