//! The Inkling engine is a served door: a variable that reaches below it is
//! refused by name before anything is read, as beside a Qwen3 config. On CUDA
//! `LLVQ_KERNEL_DIR` swaps the text of the expert kernels and on Metal
//! `LLVQ_METAL_KERNEL` swaps their matvec; set beside `llvq serve`, either
//! changed the served Inkling object with no word.
//!
//! A test binary of its own: it sets a variable in the process environment,
//! which no other test may see.

use llvq_runtime::library::ModelName;
use llvq_runtime::store::Store;

#[test]
fn a_variable_below_the_door_is_refused_before_the_load() {
    let dir = std::env::temp_dir().join(format!("llvq-runtime-test-ink-door-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let store = Store::at(dir.clone());
    let name = ModelName::parse("ink:test").unwrap();
    // No such file: before the check, the load failed on reading it.
    let missing = dir.join("no-such-inkling.json");
    std::env::set_var("LLVQ_KERNEL_DIR", "/nowhere");
    let r = llvq_runtime::inkling::load_local(&name, &missing, &store);
    std::env::remove_var("LLVQ_KERNEL_DIR");
    let e = match r {
        Err(e) => format!("{e:#}"),
        Ok(_) => panic!("a model loaded from a file that does not exist"),
    };
    assert!(e.contains("LLVQ_KERNEL_DIR") && e.contains("below the served door"), "{e}");
    let _ = std::fs::remove_dir_all(&dir);
}
