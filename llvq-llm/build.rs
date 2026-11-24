//! One cfg for the code both GPU backends of Inkling share: `ink_glue` is set
//! on a macOS build with the `metal` feature and on a Linux build with the
//! `cuda` feature, the two places `inkling::glue` has kernels for.
fn main() {
    println!("cargo::rustc-check-cfg=cfg(ink_glue)");
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let has = |f: &str| std::env::var_os(format!("CARGO_FEATURE_{f}")).is_some();
    if (os == "macos" && has("METAL")) || (os == "linux" && has("CUDA")) {
        println!("cargo::rustc-cfg=ink_glue");
    }
}
