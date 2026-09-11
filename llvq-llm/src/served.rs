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
//! and said nothing — which is exactly what happened to the 100.6 tok/s bar,
//! measured at `FUSE=0 ROT_SHARE=0` and compared against a served 1/1.
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
//!   run would believe the other.

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

/// An environment variable that disagrees with the file is refused by name.
fn agrees(var: &str, file_value: &str) -> Result<(), String> {
    match std::env::var(var) {
        Err(_) => Ok(()),
        Ok(env) if env == file_value => Ok(()),
        Ok(env) => Err(format!(
            "{var}={env:?} contradicts the served config, which says {file_value:?}. \
             Unset it, or change the file — this is not a precedence question, it is \
             two answers to one question."
        )),
    }
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

    pub fn read(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let f: ServedFile = serde_json::from_str(&text)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        Self::of(f, path.to_path_buf()).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn of(f: ServedFile, path: PathBuf) -> Result<Self, String> {
        for (var, value) in [
            ("LLVQ_FUSED_LAYOUT", &f.layout),
            ("LLVQ_EMBED", &f.embed),
            ("LLVQ_ROT_SHARE", &f.rot_share),
            ("LLVQ_FUSE", &f.fuse),
            ("LLVQ_KV", &f.kv),
        ] {
            agrees(var, value)?;
        }
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
