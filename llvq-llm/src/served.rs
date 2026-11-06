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
//! * a missing key, among the five required ones — there are no defaults
//!   here, that is the point. `attention` is the one exception, added on
//!   2026-11-02: it is optional because its absence means the form every
//!   published number ran, so the six configs that predate it keep serving
//!   the same logits. A present-but-empty `attention` is refused like any
//!   other empty field;
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
//! ## The decode engine
//!
//! `decode` is the second optional field, added on 2026-11-05. Absent, a
//! token is decoded by the eager path, as every config written before it.
//! `"persistent"` decodes it in 4E's one cooperative launch
//! (`crate::persist4e_cuda`). It never changes what is computed. Its config
//! sends every decode row to the attention kernel, from position 0: the
//! operator's T8 extension below 1,280 granted that form on the sealed 4B
//! (`docs/mesures/attn-t8-below-4b-2026-11-05.txt`). The eager path computes
//! the same form, and 4E owes its tokens bit for bit, so a binary that ignores
//! the field serves the same tokens, slower. It is refused:
//! * beside any attention but `kernel`, and beside any other field than the
//!   served 4B's (tetra48, q4, rot_share 1, fuse 0, kv f16): 4E reads those;
//! * on Metal, where 4E has no kernel;
//! * on any shape but the Qwen3-4B's, the one object it has run on and the
//!   only one the T8 covers ([`check_persistent_object`]);
//! * beside any measurement-mode variable ([`MEASUREMENT_MODES`]).
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
use crate::kvq::{AttnHeads, KvMode, KvStore};
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
    /// How attention reads the cached heads. The one OPTIONAL field that
    /// changes what runs, and optional on purpose: absent means `expanded`,
    /// the form every published number ran, so the six configs written before
    /// this field existed serve the same logits they always served. See
    /// [`crate::kvq::AttnHeads::parse`] for why that default is honest where
    /// the five required fields' defaults were not.
    #[serde(default)]
    pub attention: Option<String>,
    /// How a decode token runs: absent is the eager path, `"persistent"` is
    /// 4E. Optional for the reason `attention` is: absent is what every
    /// earlier config served. See [`Decode::parse`].
    #[serde(default)]
    pub decode: Option<String>,
    /// 4E's row loop, beside `"decode": "persistent"` only: absent is v3,
    /// the form the served-4e door ran; `"v4"` is the v2 loop inside the v3
    /// structure. See [`Rows4e::parse`].
    #[serde(default)]
    pub rows: Option<String>,
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
    /// How attention reads the cached heads.
    pub attention: AttnHeads,
    /// The cache growth store, read from `attention` and never written beside
    /// it ([`AttnHeads::store`]).
    pub store: KvStore,
    /// How a decode token runs.
    pub decode: Decode,
    /// 4E's row loop; `V3` under the eager decode, where it is not read.
    pub rows: Rows4e,
    pub path: PathBuf,
    pub note: Option<String>,
}

/// The decode engine a served config names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decode {
    /// The eager path, one launch a kernel: every config before 2026-11-05.
    Eager,
    /// 4E: the whole token in one cooperative launch, every decode row of the
    /// attention on the kernel from position 0.
    Persistent,
}

impl Decode {
    /// Absent is `Eager`. The empty string is refused, like every empty field
    /// of a file that has no defaults; so is any other word.
    pub fn parse(v: Option<&str>) -> Result<Self, String> {
        match v.map(str::trim) {
            None => Ok(Self::Eager),
            Some("eager") => Ok(Self::Eager),
            Some("persistent") => Ok(Self::Persistent),
            Some("") => Err("decode is empty. Write \"persistent\", or delete the key for the eager path.".into()),
            Some(o) => Err(format!("decode {o:?}: the engines are \"eager\" (or the key absent) and \"persistent\"")),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Eager => "eager",
            Self::Persistent => "persistent",
        }
    }

    /// The first position the kernel form sends to its kernel under this
    /// engine: the constant for the eager configs, 0 for 4E's (the T8
    /// extension of 2026-11-05).
    pub fn kernel_from(self) -> usize {
        match self {
            Self::Eager => crate::kvq::KERNEL_FROM,
            Self::Persistent => 0,
        }
    }
}

/// 4E's row loop, the module's row form (`llvq_cuda::pers::P4E_ROWS_FORMS`).
/// Every form owes the eager path's tokens bit for bit; they differ in speed
/// only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rows4e {
    /// Form 1, the loop the served-4e door ran (`docs/mesures/served-4e-2026-11-05.txt`).
    V3,
    /// Form 2, the v2 loop inside the v3 structure: x1.060 over v3 after a
    /// 5-token prompt, same tokens (measured, `docs/mesures/shaker-4e-2026-11-05.txt`).
    V4,
}

impl Rows4e {
    /// Absent is `V3`. `"v5"` is refused by name: it measured x1.001 over v4
    /// and the shaker's rule B kept it a measurement. Any other word, and the
    /// empty string, are refused.
    pub fn parse(v: Option<&str>) -> Result<Self, String> {
        match v.map(str::trim) {
            None | Some("v3") => Ok(Self::V3),
            Some("v4") => Ok(Self::V4),
            Some("v5") => Err("rows \"v5\": measured at x1.001 over v4 (shaker-4e-2026-11-05) and kept a \
                               measurement; the served forms are \"v3\" and \"v4\""
                .into()),
            Some("") => Err("rows is empty. Write \"v3\" or \"v4\", or delete the key for v3.".into()),
            Some(o) => Err(format!("rows {o:?}: the served forms of 4E's row loop are \"v3\" and \"v4\"")),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::V3 => "v3",
            Self::V4 => "v4",
        }
    }

    /// The module's row form number.
    pub fn form(self) -> u32 {
        match self {
            Self::V3 => 1,
            Self::V4 => 2,
        }
    }
}

/// The measurement modes of the binaries that read a served config: each turns
/// a run into a bench of something. Beside a config that names the persistent
/// engine each is refused by name ([`check_env`]); `LLVQ_CONTEXTS` and
/// `LLVQ_PREFILL_TOKENS` are not here, since they time the served object as
/// served with a longer prompt.
pub const MEASUREMENT_MODES: [&str; 17] = [
    "LLVQ_KV_PREALLOC", "LLVQ_GRAPH_AB", "LLVQ_GRAPH_DIAG", "LLVQ_KV_AB", "LLVQ_FUSE_AB", "LLVQ_TIME_PHASES",
    "LLVQ_TIME_EVENTS", "LLVQ_SEG_ARMS", "LLVQ_BENCH_ARMS", "LLVQ_TRIM_AB", "LLVQ_GQA_AB", "LLVQ_GQA_PPL",
    "LLVQ_INT4_WIDE_CHECK", "LLVQ_GESTURE_AB", "LLVQ_BOUNDARY_AB", "LLVQ_PERS_SEGMENT", "LLVQ_PERS_4E",
];

/// The persistent engine against the model's shape: the Qwen3-4B's, the one
/// object 4E has run on and the one the T8 extension covers. Pure, on the
/// config candle read from the file.
pub fn check_persistent_object(cfg: &candle_transformers::models::qwen3::Config) -> Result<(), String> {
    let got = (
        cfg.num_hidden_layers,
        cfg.hidden_size,
        cfg.intermediate_size,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
        cfg.vocab_size,
    );
    let want = (36, 2560, 9728, 32, 8, 128, 151_936);
    match got == want {
        true => Ok(()),
        false => Err(format!(
            "decode \"persistent\" on a model of (layers, hidden, intermediate, heads, kv heads, head dim, \
             vocab) = {got:?}. 4E has run, and the kernel form's T8 extension below 1,280 was granted, on the \
             sealed Qwen3-4B only {want:?}: the 8B and the 14B are not covered. Serve this object with the \
             eager engine (delete the decode key)."
        )),
    }
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
    if Decode::parse(f.decode.as_deref()) == Ok(Decode::Persistent) {
        for var in MEASUREMENT_MODES {
            if let Some(v) = lookup(var) {
                return Err(format!(
                    "{var}={v:?} beside a served config that names the persistent decode: that is a \
                     measurement mode, and the served 4E path runs one engine with nothing to compare. \
                     Unset it, or unset LLVQ_CONFIG and run the bench."
                ));
            }
        }
    }
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
        // `attention` goes through its own parse and NOT through `field`: it
        // is the one optional field, so `None` is a value here (`expanded`)
        // and the empty string is refused inside the parse instead.
        let attention = AttnHeads::parse(f.attention.as_deref())?;
        let decode = Decode::parse(f.decode.as_deref())?;
        let rows = Rows4e::parse(f.rows.as_deref())?;
        if f.rows.is_some() && decode != Decode::Persistent {
            return Err(format!(
                "rows {:?} beside the eager decode: the field names 4E's row loop. Write \"decode\": \
                 \"persistent\", or delete the rows key.",
                rows.label()
            ));
        }
        let layout = FusedLayout::parse(field("layout", &f.layout)?)?;
        let embed = EmbedMode::parse(field("embed", &f.embed)?)?;
        let rot_share = RotShare::parse(field("rot_share", &f.rot_share)?)?;
        let fuse = FuseMode::parse(field("fuse", &f.fuse)?)?;
        let kv = KvMode::parse(field("kv", &f.kv)?)?;
        if decode == Decode::Persistent {
            if attention != AttnHeads::Kernel {
                return Err(format!(
                    "decode \"persistent\" beside attention {:?}: 4E computes the kernel form's attention \
                     and nothing else. Write \"attention\": \"kernel\", or delete the decode key.",
                    attention.label()
                ));
            }
            let served_4b = (FusedLayout::Tetra48, EmbedMode::Q4, RotShare::On, FuseMode::Off, KvMode::F16);
            if (layout, embed, rot_share, fuse, kv) != served_4b {
                return Err(format!(
                    "decode \"persistent\" with layout {}, embed {}, rot_share {}, fuse {}, kv {}: 4E reads the \
                     served 4B's fields, tetra48, q4, 1, 0 and f16, and no other",
                    layout.name(),
                    embed.name(),
                    rot_share.name(),
                    fuse.name(),
                    kv.name()
                ));
            }
        }
        Ok(Self {
            layout,
            embed,
            rot_share,
            fuse,
            kv,
            attention,
            store: attention.store(),
            decode,
            rows,
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
        // The kernel form names its threshold: the benches of 2026-11-04 ran
        // a "kernel" with none, and their lines must not read as this one.
        let attention = match self.attention {
            AttnHeads::Kernel => format!("kernel from position {}", self.decode.kernel_from()),
            other => other.label().to_string(),
        };
        // The engine is named only when it is not the eager one, and the row
        // loop only when it is not v3, so every line an earlier config printed
        // reads as it did.
        let decode = match (self.decode, self.rows) {
            (Decode::Eager, _) => String::new(),
            (Decode::Persistent, Rows4e::V3) => ", decode persistent (4E, one launch a token run)".to_string(),
            (Decode::Persistent, r) => format!(", decode persistent (4E rows {}, one launch a token run)", r.label()),
        };
        format!(
            "served config: {} (layout {}, embed {}, rot_share {}, fuse {}, kv {}, \
             attention {attention} on {}{decode}){note}",
            self.path.display(),
            self.layout.name(),
            self.embed.name(),
            self.rot_share.name(),
            self.fuse.name(),
            self.kv.name(),
            self.store.label(),
        )
    }

    /// The served attention form against the device it would run on.
    ///
    /// `Grouped` is measured on one card, the L40S, through three runs of
    /// 2026-10-30 and 2026-11-01. On Metal it has never run: no throughput, no
    /// perplexity, and no comparison of what Metal's own matmuls round to. The
    /// form is backend-independent code in `model.rs`, which is exactly why a
    /// silent acceptance would be easy and wrong — it would serve on the Mac a
    /// form whose tolerance gate was passed on an NVIDIA card.
    ///
    /// So: refused by name, with the measurement that would lift the refusal.
    /// `metal` is handed in rather than read from a device so the refusal is
    /// testable on a machine that has no Metal device, and so this function
    /// stays pure like the rest of this module.
    pub fn check_device(&self, metal: bool) -> Result<(), String> {
        if self.decode == Decode::Persistent && metal {
            return Err(format!(
                "{}: decode \"persistent\" on Metal. 4E is one cooperative CUDA launch \
                 (llvq-cuda/kernels/pers_4e.cu); no Metal kernel exists. Serve this object with the \
                 eager engine on Metal.",
                self.path.display(),
            ));
        }
        match (self.attention, metal) {
            (AttnHeads::Expanded, _) | (_, false) => Ok(()),
            (AttnHeads::Kernel, true) => Err(format!(
                "{}: attention \"kernel\" on Metal. The split-KV decode-attention kernel is \
                 CUDA only (llvq-cuda/kernels/attn_dec.cu); no Metal kernel exists. Serve this \
                 object with the expanded form on Metal.",
                self.path.display(),
            )),
            (form, true) => Err(format!(
                "{}: attention {:?} on Metal. That form has never been measured on \
                 Metal — no throughput, no perplexity, and no reading of what Metal \
                 rounds to — and the +-0.12 % tolerance it passed was measured on an \
                 L40S (gqa-ppl-2026-11-01). Serve this object with the expanded form, \
                 or measure the grouped one on Metal first.",
                self.path.display(),
                form.label(),
            )),
        }
    }

    /// Put a loaded model on what this config says: the store, then the form.
    ///
    /// The one place the pair is applied, called by the served door of
    /// `bin/fusedrun`, `bin/mmlu` and `bin/gsm8k`. Three copies would be three
    /// chances for one binary to serve `grouped` while another serves
    /// `expanded` on the same file, which is the defect class
    /// [`Served`] exists to close.
    pub fn apply(
        &self,
        model: &mut crate::model::Qwen3,
        device: &candle_core::Device,
    ) -> Result<(), String> {
        self.check_device(device.is_metal())?;
        if self.decode == Decode::Persistent {
            check_persistent_object(model.config()).map_err(|e| format!("{}: {e}", self.path.display()))?;
        }
        self.apply_form(model, device)
    }

    /// [`Self::apply`] without the device refusal: the measurement door of
    /// `bin/metalrun`, which times and scores on Metal the attention forms
    /// [`Self::check_device`] refuses there. No served binary calls it: the
    /// refusal stands for every served door until the operator lifts it. 4E
    /// is still refused, since no Metal kernel exists for it.
    pub fn apply_measurement(
        &self,
        model: &mut crate::model::Qwen3,
        device: &candle_core::Device,
    ) -> Result<(), String> {
        if self.decode == Decode::Persistent {
            return Err(format!(
                "{}: decode \"persistent\" has no Metal kernel, not even as a measurement",
                self.path.display()
            ));
        }
        self.apply_form(model, device)
    }

    /// [`Self::apply`] past its device and object checks: the store, the form,
    /// the threshold and the kernel. Split out so the host tests reach the
    /// threshold on a toy model, which the object check refuses.
    pub(crate) fn apply_form(
        &self,
        model: &mut crate::model::Qwen3,
        device: &candle_core::Device,
    ) -> Result<(), String> {
        model.set_kv_store(self.store);
        model.set_attn_heads(self.attention);
        // The threshold of the kernel form: the constant, or 0 under a config
        // that names 4E. Applied to the eager path too, which prefills and is
        // the engine any binary without 4E decodes with: the same arithmetic.
        model.set_served_kernel_from(self.decode.kernel_from());
        // The kernel form is the one form that needs a device object: built
        // here, once, and on CUDA checked against the CPU replay bit for bit
        // before the first token (`crate::attn::kernel_for`).
        let kernel = match self.attention {
            AttnHeads::Kernel => {
                let (n_kv, groups, hd) = model.attn_geometry();
                Some(crate::attn::kernel_for(device, n_kv, groups, hd)?)
            }
            AttnHeads::Expanded | AttnHeads::Shared | AttnHeads::Grouped => None,
        };
        model.set_attn_kernel(kernel);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Which backend loads a sealed artifact.
//
// One place, so a second backend registers here and nowhere else. Before the
// device port every caller carried its own
// `#[cfg(all(target_os = "linux", feature = "cuda"))]` around a direct
// `fused_cuda::load_*` call: `bin/mmlu` once and `bin/fusedrun` five times.
// Those `cfg`s are what a Metal lot would otherwise have to reopen, one by
// one, in two binaries.
//
// TWO arms since 2026-09-21: `fused_cuda` on Linux and `fused_metal` on macOS.
// Neither caller changed when the second landed, which is what the shape was
// for. A third backend adds a third arm and touches nothing else.
// ---------------------------------------------------------------------------

/// The served object, loaded by whichever backend this build carries.
///
/// The argument list is `fused_cuda::load_resolved`'s, unchanged, because a
/// second backend reads the same config file and must answer the same
/// questions.
#[allow(clippy::too_many_arguments)]
pub fn load_resolved(
    path: &str,
    device: &candle_core::Device,
    dtype: candle_core::DType,
    layout: crate::fused::FusedLayout,
    emode: crate::fused::EmbedMode,
    share: crate::rotplan::RotShare,
    fuse: crate::fused::FuseMode,
    kv: crate::kvq::KvMode,
    origin: Option<&str>,
) -> candle_core::Result<crate::fused::FusedSealed> {
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    {
        crate::fused_cuda::load_resolved(path, device, dtype, layout, emode, share, fuse, kv, origin)
    }
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        crate::fused_metal::load_resolved(path, device, dtype, layout, emode, share, fuse, kv, origin)
    }
    #[cfg(not(any(
        all(target_os = "linux", feature = "cuda"),
        all(target_os = "macos", feature = "metal")
    )))]
    {
        let _ = (path, device, dtype, layout, emode, share, fuse, kv, origin);
        candle_core::bail!(
            "this build carries no device backend: a sealed artifact can be served \
             through the fused path on a CUDA build on Linux or a Metal build on macOS. \
             Otherwise take the dense path through sealed::load"
        )
    }
}

/// [`load_resolved`]'s measurement door: the layout, the embedding mode and
/// the hoist come from the environment rather than from a config file.
pub fn load_with(
    path: &str,
    device: &candle_core::Device,
    dtype: candle_core::DType,
    fuse: crate::fused::FuseMode,
) -> candle_core::Result<crate::fused::FusedSealed> {
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    {
        crate::fused_cuda::load_with(path, device, dtype, fuse)
    }
    #[cfg(not(all(target_os = "linux", feature = "cuda")))]
    {
        // The measurement door stays CUDA-only. It reads the layout, the
        // embedding mode and the hoist from the environment, and every A/B in
        // `docs/mesures/` turns one of those between two processes on a card.
        // A Metal arm here would let a Mac answer a question the journals ask
        // of an L40S.
        let _ = (path, device, dtype, fuse);
        candle_core::bail!(
            "this build carries no device backend for the measurement door, which is \
             CUDA-only. Use served::load_resolved with a config file instead"
        )
    }
}
