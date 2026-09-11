//! The served configuration, as a file rather than as five environment
//! variables each with a default.
//!
//! ## Why a file
//!
//! The object this repository ships is not a `.llvq` alone. It is a `.llvq`
//! plus the handful of choices that decide how it is read: which VRAM layout,
//! whether the embedding is quantized, whether the group's rotation is
//! hoisted. Those lived in `LLVQ_FUSED_LAYOUT`, `LLVQ_EMBED`, `LLVQ_ROT_SHARE`
//! and `LLVQ_FUSE`, each with a default, and the default of every one of them
//! is **not** the served value. A run that forgot one measured something else
//! and said nothing — which is exactly what happened to the F1e §0 Tetra runs
//! of 2026-09-10, measured at `FUSE=0 ROT_SHARE=0` and set beside a 100.6 tok/s
//! bar that had been measured at the served `ROT_SHARE=1 FUSE=1` on `Planes14`
//! (`docs/mesures/d1-fusion-servie-2026-08-24.txt`, `f1e0-2026-09-10.txt` §0).
//!
//! So the served values ship as a file beside the artifact, and the file is
//! the authority. There is no built-in served default here: a runner pointed
//! at no config is a runner in measurement mode, doing what it always did.
//!
//! ## The vocabulary is not restated
//!
//! Every field goes through the SAME `parse` the environment variable goes
//! through — [`FusedLayout::parse`], [`EmbedMode::parse`], [`RotShare::parse`],
//! [`FuseMode::parse`], [`KvMode::parse`]. A second spelling of "tetra48"
//! would be a second thing to keep in step, and this file has no opinion about
//! what a layout is called.
//!
//! ## What it refuses
//!
//! * an unknown key — a typo in `"embeding"` must not read as "the default";
//! * a missing key — there are no defaults here, that is the point;
//! * an empty string — `parse` reads `Some("")` as the default, which is the
//!   one way a file could still silently mean `planes14`;
//! * an environment variable that DISAGREES with the file. Not a precedence
//!   rule, a refusal: whichever way one resolved it, half the readers of the
//!   run would believe the other. The comparison goes THROUGH the parsers, so
//!   `LLVQ_FUSE=""` beside `"fuse": "0"` agrees — both mean `Off`;
//! * two variables that reach BELOW the served door and that no field of this
//!   file carries: `LLVQ_KERNEL_DIR` swaps the kernel text, `LLVQ_TILE_BLOCKS`
//!   moves a served constant. Either one set beside `LLVQ_CONFIG` is refused
//!   by name. `LLVQ_NVRTC_ARCH` is deliberately NOT refused: it names the card,
//!   not the object, and an A100 needs it; the runtime prints it on its
//!   `NVRTC source:` line so no served log is silent about its target.
//!
//! ## Pure where it can be
//!
//! [`Served::of`] reads no environment. The environment check is
//! [`check_env`], and it takes its lookup as an argument, so the contradiction
//! branch is tested with a closure and the test suite's result does not
//! depend on what the developer's shell happens to export — which it did,
//! four tests out of six, until the audit of 2026-09-11 found it.

use std::path::{Path, PathBuf};

use crate::fused::{EmbedMode, FuseMode, FusedLayout};
use crate::kvq::KvMode;
use crate::rotplan::RotShare;

/// The file on disk. Strings, because the vocabulary belongs to the parsers.
#[derive(serde::Deserialize, serde::Serialize, Debug, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ServedFile {
    pub layout: String,
    pub embed: String,
    pub rot_share: String,
    pub fuse: String,
    pub kv: String,
    /// Free text, carried so a config can say what object it belongs to. Read
    /// by nothing and printed with the rest.
    #[serde(default)]
    pub note: Option<String>,
}

/// The same thing resolved, plus where it came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Served {
    pub layout: FusedLayout,
    pub embed: EmbedMode,
    pub rot_share: RotShare,
    pub fuse: FuseMode,
    pub kv: KvMode,
    pub path: PathBuf,
    pub note: Option<String>,
}

/// The variables this file speaks for, in the order the file lists them.
const SPOKEN_FOR: [&str; 5] =
    ["LLVQ_FUSED_LAYOUT", "LLVQ_EMBED", "LLVQ_ROT_SHARE", "LLVQ_FUSE", "LLVQ_KV"];

/// The variables that reach below the served door and that this file cannot
/// carry. Set beside `LLVQ_CONFIG`, each is refused by name.
pub const BELOW_THE_DOOR: [&str; 2] = ["LLVQ_KERNEL_DIR", "LLVQ_TILE_BLOCKS"];

/// `Some(value)` from a non-empty string, `Err` from an empty one.
///
/// The parsers all read `Some("")` as their default, so an empty field in a
/// file that exists to have no defaults is the one hole worth closing by hand.
fn field<'a>(key: &str, v: &'a str) -> Result<Option<&'a str>, String> {
    match v.trim().is_empty() {
        true => Err(format!(
            "{key} is empty. This file has no defaults: write the value, or delete \
             the file and pass the environment variable."
        )),
        false => Ok(Some(v)),
    }
}

/// One variable against one field, compared as what they MEAN.
///
/// Through `parse` on both sides: an environment value that does not parse is
/// its own error (it would have been at the variable's own read anyway), and
/// two spellings of one mode — `""` and `"0"` for `FuseMode::Off` — agree.
fn same<T: PartialEq + std::fmt::Debug>(
    var: &str,
    env: Option<String>,
    file_value: &str,
    parse: fn(Option<&str>) -> Result<T, String>,
) -> Result<(), String> {
    let Some(env) = env else { return Ok(()) };
    let from_env = parse(Some(env.as_str())).map_err(|e| format!("{var} beside LLVQ_CONFIG: {e}"))?;
    let from_file = parse(Some(file_value))?;
    match from_env == from_file {
        true => Ok(()),
        false => Err(format!(
            "{var}={env:?} contradicts the served config, which says {file_value:?}. \
             Unset it, or change the file — this is not a precedence question, it is \
             two answers to one question."
        )),
    }
}

/// The environment against the file, with the environment handed in.
///
/// `lookup` is `|v| std::env::var(v).ok()` in production and a closure in a
/// test. Called by [`Served::read`] and nowhere inside [`Served::of`], which
/// stays pure.
pub fn check_env(
    f: &ServedFile,
    lookup: impl Fn(&str) -> Option<String>,
) -> Result<(), String> {
    for var in BELOW_THE_DOOR {
        if let Some(v) = lookup(var) {
            return Err(format!(
                "{var}={v:?} beside LLVQ_CONFIG. That variable reaches below the served \
                 door — it changes what the config cannot describe — so a served run \
                 refuses it by name. Unset it, or unset LLVQ_CONFIG and run the bench."
            ));
        }
    }
    let [layout, embed, rot_share, fuse, kv] = SPOKEN_FOR;
    same(layout, lookup(layout), &f.layout, FusedLayout::parse)?;
    same(embed, lookup(embed), &f.embed, EmbedMode::parse)?;
    same(rot_share, lookup(rot_share), &f.rot_share, RotShare::parse)?;
    same(fuse, lookup(fuse), &f.fuse, FuseMode::parse)?;
    same(kv, lookup(kv), &f.kv, KvMode::parse)?;
    Ok(())
}

impl Served {
    /// Resolve from `LLVQ_CONFIG`, or `None` when it is unset.
    ///
    /// `None` is not a failure and not a fallback to the served values: it is
    /// a runner in measurement mode, reading the same environment variables it
    /// always read.
    pub fn from_env() -> Result<Option<Self>, String> {
        match std::env::var("LLVQ_CONFIG") {
            Err(_) => Ok(None),
            Ok(p) if p.trim().is_empty() => Err(
                "LLVQ_CONFIG is set to the empty string. Point it at a config file, \
                 or unset it."
                    .to_string(),
            ),
            Ok(p) => Self::read(Path::new(&p)).map(Some),
        }
    }

    /// The file, parsed and nothing else. Disk, no environment.
    pub fn read_file(path: &Path) -> Result<ServedFile, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Read the file, check it against the real environment, resolve it.
    ///
    /// The one function here that consults the shell, and therefore the one a
    /// test must not call: `the_shipped_config_is_the_served_object` reads
    /// through [`Self::read_file`] and resolves through [`Self::of`], so its
    /// verdict is about the file and not about what the developer exported.
    pub fn read(path: &Path) -> Result<Self, String> {
        let f = Self::read_file(path)?;
        check_env(&f, |v| std::env::var(v).ok())
            .map_err(|e| format!("{}: {e}", path.display()))?;
        Self::of(f, path.to_path_buf()).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// The file's strings through the parsers. Pure: no environment, no disk.
    pub fn of(f: ServedFile, path: PathBuf) -> Result<Self, String> {
        Ok(Self {
            layout: FusedLayout::parse(field("layout", &f.layout)?)?,
            embed: EmbedMode::parse(field("embed", &f.embed)?)?,
            rot_share: RotShare::parse(field("rot_share", &f.rot_share)?)?,
            fuse: FuseMode::parse(field("fuse", &f.fuse)?)?,
            kv: KvMode::parse(field("kv", &f.kv)?)?,
            path,
            note: f.note,
        })
    }

    /// What a runtime prints as the source of its choices, in place of the
    /// variable names it prints in measurement mode.
    pub fn source(&self) -> String {
        format!("LLVQ_CONFIG={}", self.path.display())
    }

    /// One line, printed by every binary that resolves one — the provenance a
    /// reader needs to know which object produced a number.
    pub fn provenance(&self) -> String {
        let note = match &self.note {
            Some(n) => format!(" — {n}"),
            None => String::new(),
        };
        format!(
            "served config: {} (layout {}, embed {}, rot_share {}, fuse {}, kv {}){note}",
            self.path.display(),
            self.layout.name(),
            self.embed.name(),
            self.rot_share.name(),
            self.fuse.name(),
            self.kv.name(),
        )
    }
}
