//! The commit the binary is built from, for `/api/version` and `llvq -v`.
//! `unknown` outside a git checkout, never a build failure.

fn main() {
    let commit = std::process::Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=LLVQ_BUILD_COMMIT={commit}");
    println!("cargo:rerun-if-changed=../.git/HEAD");
}
