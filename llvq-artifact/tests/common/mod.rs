//! The one thing the sealed-archive sweeps must agree on: what happens when
//! the archive is not on this machine.
//!
//! Everything else these files share is duplicated on purpose: `read_bits`,
//! the residue rules and the fixtures are each restated from the spec, so that
//! a bug in the implementation under test cannot also be in the yardstick.
//! A filesystem lookup has no spec to restate — and three copies of the
//! message below would be three chances for it to drift, which is how a
//! partial correction ends up looking more checked than it is.

use std::path::{Path, PathBuf};

/// Where to look when the archive lives somewhere other than `$HOME`.
///
/// A pointer, deliberately not an escape hatch: it can move the search, never
/// satisfy it. There is no value of this variable that makes a sweep pass
/// without reading an archive.
const ENV: &str = "LLVQ_SEALED_ARTIFACT";

/// Default name under `$HOME`: the working `leech1c12` archive of the
/// published Qwen3-4B, 981 MB of projections.
const DEFAULT_NAME: &str = "llvq-q4b.llvq";

const HOWTO: &str = "\
This sweep checks a runtime layout against the real Qwen3-4B archive, block by
block, and it is #[ignore]d so it never runs by accident. Being here means it
was asked for by name or with --include-ignored — so it fails rather than
printing SKIP and reporting `ok`.

Point it at a `leech1c12` archive (v1, v2 or v3 magic — the quantized section
of the published sealed model is byte-identical to the working
~/llvq-q4b.llvq, so either file works):

    LLVQ_SEALED_ARTIFACT=/path/to/qwen3-4b-llvq.bin \\
        cargo test --release -p llvq-artifact -- --include-ignored

See LAUNCH_ME.md for the download, or README.md for the `bin/smoke` invocation
that produces one. To leave the sweep out deliberately, drop --include-ignored:
it is then reported as `ignored`, which is a declaration, not a silent pass.";

/// Path of the sealed 4B archive, or a loud failure.
///
/// The form this replaces was `eprintln!("SKIP …"); return;`, which reported
/// `ok` on every machine without the file — an assertion that does not
/// exercise what it claims to cover, the exact defect CLAUDE.md §5 is about.
/// This repo is checked out on more than one machine, so that green was a
/// statement about the filesystem dressed up as a statement about the format.
///
/// The fix is in two halves and needs both: the sweeps are `#[ignore]`d, so
/// their absence from the default run is *declared* rather than silent, and a
/// missing archive on an explicit run is a failure that names the file.
pub fn sealed_artifact_path() -> PathBuf {
    let (path, origin) = match (std::env::var_os(ENV), std::env::var_os("HOME")) {
        (Some(p), _) => (PathBuf::from(p), ENV),
        (None, Some(h)) => (Path::new(&h).join(DEFAULT_NAME), "$HOME/llvq-q4b.llvq"),
        (None, None) => {
            panic!("neither ${ENV} nor $HOME is set, so there is no path to try.\n\n{HOWTO}")
        }
    };
    assert!(
        path.exists(),
        "the sealed archive is not on this machine: {} (from {origin})\n\n{HOWTO}",
        path.display()
    );
    path
}
