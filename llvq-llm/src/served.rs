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
//! * seven variables that reach BELOW the served door and that no field of
//!   this file carries ([`BELOW_THE_DOOR`]): `LLVQ_KERNEL_DIR` swaps the kernel
//!   text, `LLVQ_TILE_BLOCKS` and `LLVQ_PREFILL` move served constants,
//!   `LLVQ_METAL_KERNEL` swaps the Metal matvec, and `LLVQ_GUIDED_WEIGHTS`,
//!   `LLVQ_GUIDED_GROUPS` and `LLVQ_GUIDED_CLIP` change how an object is
//!   encoded. Any one set beside `LLVQ_CONFIG` is refused by name, and beside
//!   a served model of a family with no such file ([`check_door`], the
//!   runtime's Inkling engine). `LLVQ_NVRTC_ARCH` is deliberately NOT refused: it names the card,
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
//! * on any shape but the sealed Qwen3 shapes, 4B, 8B and 14B, and the 0.6B drafter
//!   ([`check_persistent_object`]);
//! * beside any measurement-mode variable ([`MEASUREMENT_MODES`]).
//!
//! ## The prefill form
//!
//! `prefill` is the third optional field, added on 2026-11-15. Absent, a
//! prompt is prefilled by the eager path, as every config written before it.
//! `"chain"` feeds it through the tree verify pass and hands over to 4E v9
//! ([`Prefill`]). It is refused beside anything but 4E v9, beside
//! `prefill_attention` and `batch`, on Metal, and on any shape but the 4B's.
//! It is not bit for bit with the eager prefill; the decode after it is.
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
use crate::kvq::{AttnHeads, KvMode, KvStore, PrefillAttn};
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
    /// structure; `"v6"` is v4 with vector prologues; `"v9"` is v8 with the
    /// int4 rows on tiled words, which also lays the int4 words tiled on the
    /// card ([`Rows4e::int4_layout`]). See [`Rows4e::parse`].
    #[serde(default)]
    pub rows: Option<String>,
    /// How a prefill chunk is attended, beside `"attention": "kernel"` only:
    /// absent is `"grouped"`, what every config before 2026-11-08 served;
    /// `"kernel"` sends the prompt's rows to the decode-attention kernel, 8 a
    /// call, and makes no score tensor. See [`PrefillAttn`].
    #[serde(default)]
    pub prefill_attention: Option<String>,
    /// How a prompt is prefilled, beside `"decode": "persistent"` and
    /// `"rows": "v9"` only: absent is the eager prefill, what every config
    /// before 2026-11-15 served; `"chain"` feeds the prompt through the tree
    /// verify pass as chains of known tokens ([`CHAIN_ARM`]). See [`Prefill`].
    #[serde(default)]
    pub prefill: Option<String>,
    /// The cache 4E's decode holds, beside `"decode": "persistent"` and
    /// `"rows": "v9"` only: absent is binary16, what every config before
    /// 2026-11-15 served; `"int8"` is the NOT SERVED form of research arm
    /// v9k8 with the binary16 caches released at the hand-over. See
    /// [`DecodeCache`].
    #[serde(default)]
    pub decode_cache: Option<String>,
    /// The batched decode (`crate::batch`): how many sequences one step
    /// carries at most, and how many positions a sequence holds. Absent, the
    /// object is served one sequence at a time, as every config before it.
    /// See [`BatchServe`].
    #[serde(default)]
    pub batch: Option<BatchFile>,
    /// The Metal kernels this config serves, spelled as the `bin/metalrun`
    /// arm that measured them (`"lutd+h+pq"`). Absent is
    /// `MetalTuning::default()`, what every config before 2026-11-26 served.
    /// A config that names it is a Metal config: refused off Metal, and the
    /// one kind of config through which Metal serves the grouped form. See
    /// [`MetalKernels`].
    #[serde(default)]
    pub metal_kernels: Option<String>,
    /// Free text, carried so a config can say what object it belongs to. Read
    /// by nothing and printed with the rest.
    #[serde(default)]
    pub note: Option<String>,
}

/// The `batch` field of a config file.
#[derive(serde::Deserialize, serde::Serialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct BatchFile {
    /// Sequences one step carries at most.
    pub max: usize,
    /// Positions a sequence holds at most, prompt and answer.
    pub context: usize,
}

/// The batched decode a config names: up to `max` sequences a step through
/// the tensor-core GEMM (`crate::batch::BatchProjs::Gemm`), each in a cache
/// slot of `context` positions. What it changes: the sum order of the
/// projections, so a sequence's tokens are not bit for bit those of batch 1;
/// the licence is the T8 band (the gate of preregistration-batch-l40s). Beside
/// `decode: persistent`, a request decoded alone runs on 4E and joins the
/// batch when another arrives (`llvq serve`). It is refused:
/// * beside any attention but `kernel`, the form its attention computes from
///   position 0, and any kv but `f16`, the cache it holds;
/// * beside `fuse: 1`, which has no batched kernel;
/// * with `max` outside 1..=128 (the widest entry) or `context` outside
///   256..=40,960 (Qwen3's RoPE table).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchServe {
    pub max: usize,
    pub context: usize,
}

/// The widest batch the GEMM's entries take (`batch_cuda::GEMM_MAX_ROWS`).
pub const BATCH_MAX_ROWS: usize = 128;

impl BatchServe {
    pub fn of(f: BatchFile) -> Result<Self, String> {
        if !(1..=BATCH_MAX_ROWS).contains(&f.max) {
            return Err(format!("batch max {}: between 1 and {BATCH_MAX_ROWS}", f.max));
        }
        if !(256..=40_960).contains(&f.context) {
            return Err(format!("batch context {}: between 256 and 40,960 positions", f.context));
        }
        Ok(Self { max: f.max, context: f.context })
    }

    /// The same, with `max` replaced by an operator's choice at serve time
    /// (`llvq serve`'s `LLVQ_BATCH`), under the same bounds.
    pub fn with_max(self, max: usize) -> Result<Self, String> {
        Self::of(BatchFile { max, context: self.context })
    }
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
    /// How the kernel form attends a prefill chunk; `Grouped` under any other
    /// form, where it is not read.
    pub prefill: PrefillAttn,
    /// How a prompt is prefilled: the eager path, or chains through the tree
    /// verify pass.
    pub prefill_form: Prefill,
    /// The cache 4E's decode holds: binary16, or int8 alone (NOT SERVED).
    pub decode_cache: DecodeCache,
    /// The batched decode, when the config names one.
    pub batch: Option<BatchServe>,
    /// The Metal kernels the config names; `Default` when it names none.
    pub metal_kernels: MetalKernels,
    pub path: PathBuf,
    pub note: Option<String>,
}

/// The Metal kernels a served config names, in `metal_kernels`.
///
/// A closed list: a name enters it once `bin/metalrun` has gated its kernels
/// bit for bit against the served ones on the sealed files it serves. The
/// spelling is the bench's arm, read by `fused_metal::MetalTuning::parse_arm`,
/// so the arm a journal timed is the one a config serves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalKernels {
    /// `MetalTuning::default()`: `tv_tetra48_metal_lut`, f32 io, one prompt
    /// row a launch. What an absent field means.
    Default,
    /// `lutd+h+pq`: the Tetra matvec `tv_tetra48_metal_lutd`, the half-io
    /// twins, and the prefill kernels of 8 rows a launch for the Tetra and the
    /// int4 records (`metal-nuit-2026-11-06` M2, M8, M9, M10: 253 of 253
    /// records bit for bit on the 4B; `qwen-metal-servi-2026-11-26` for the
    /// 8B and the 14B).
    LutdHPq,
}

impl MetalKernels {
    /// The names a config may write.
    pub const NAMES: [&'static str; 1] = ["lutd+h+pq"];

    /// Absent is [`MetalKernels::Default`]; any other spelling, the empty
    /// one included, is refused with the list.
    pub fn parse(v: Option<&str>) -> Result<Self, String> {
        match v {
            None => Ok(Self::Default),
            Some("lutd+h+pq") => Ok(Self::LutdHPq),
            Some(other) => Err(format!(
                "metal_kernels {other:?}: a config names one of {:?}, or leaves the key out for the default                  Metal kernels",
                Self::NAMES
            )),
        }
    }

    /// The `bin/metalrun` arm, `None` for the default kernels.
    pub fn arm(self) -> Option<&'static str> {
        match self {
            Self::Default => None,
            Self::LutdHPq => Some("lutd+h+pq"),
        }
    }
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
    /// of a file that has no defaults; so is any other word. The value is
    /// read as written, as [`AttnHeads::parse`] reads `attention`:
    /// `" persistent"` is refused, not trimmed into a value.
    pub fn parse(v: Option<&str>) -> Result<Self, String> {
        match v {
            None => Ok(Self::Eager),
            Some("eager") => Ok(Self::Eager),
            Some("persistent") => Ok(Self::Persistent),
            Some(s) if s.trim().is_empty() => {
                Err("decode is empty. Write \"persistent\", or delete the key for the eager path.".into())
            }
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

    /// The positions this engine can decode to, at most, beside the model's
    /// own RoPE table: none for the eager path, 4E's attention scratch for
    /// the persistent one ([`crate::served_decode::P4E_MAX_CAP`]). A harness
    /// checks its longest prompt plus its cap against it BEFORE the first
    /// problem, so a long one is refused at start and not hours into a run.
    pub fn max_positions(self) -> Option<usize> {
        match self {
            Self::Eager => None,
            Self::Persistent => Some(crate::served_decode::P4E_MAX_CAP),
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
    /// Form 4, v4 with ln1, ln2 and the down's input in one round trip a
    /// thread: x1.118 over v4 after a 5-token prompt and x1.094 at 8,192, same
    /// tokens (measured, `docs/mesures/pers-4e-v6-2026-11-05.txt`). The
    /// install refuses it when an input of a vector prologue is not 16-byte
    /// aligned (`llvq_cuda::pers::p4e_v6_inputs`), so it never times as v4
    /// under v6's name.
    V6,
    /// Form 7, v8 (v6 with the launch's constants staged during the barriers
    /// and attention's steps in a cp.async ring) with the int4 rows read from
    /// tiled words: x1.035 over v8 and x1.118 over v6 after a 5-token prompt,
    /// x1.060 over v6 at 8,192, same tokens (measured,
    /// `docs/mesures/step3-4e-2026-11-07.txt`, measurement form). Served with
    /// the int4 words tiled and held once ([`Rows4e::int4_layout`]): the
    /// eager path reads the same words through `tv_q4_hht`.
    V9,
}

impl Rows4e {
    /// Absent is `V3`. `"v5"` is refused by name: it measured x1.001 over v4
    /// and the shaker's rule B kept it a measurement. Any other word, and the
    /// empty string, are refused. Read as written, like [`Decode::parse`].
    pub fn parse(v: Option<&str>) -> Result<Self, String> {
        const FORMS: &str = "the served forms are \"v3\", \"v4\", \"v6\" and \"v9\"";
        match v {
            None | Some("v3") => Ok(Self::V3),
            Some("v4") => Ok(Self::V4),
            Some("v6") => Ok(Self::V6),
            Some("v9") => Ok(Self::V9),
            Some("v5") => Err(format!(
                "rows \"v5\": measured at x1.001 over v4 (shaker-4e-2026-11-05) and kept a measurement; {FORMS}"
            )),
            Some("v7") => Err(format!(
                "rows \"v7\": its module holds 24 B of local memory at 128 registers under the card's ptxas and \
                 does not install (step2-4e-2026-11-07); {FORMS}"
            )),
            Some("v8") => Err(format!(
                "rows \"v8\": a measurement, v9's base, which reads x1.035 over it at the bit \
                 (step3-4e-2026-11-07); {FORMS}"
            )),
            Some(s) if s.trim().is_empty() => {
                Err("rows is empty. Write \"v3\", \"v4\", \"v6\" or \"v9\", or delete the key for v3.".into())
            }
            Some(o) => Err(format!("rows {o:?}: {FORMS} of 4E's row loop")),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::V3 => "v3",
            Self::V4 => "v4",
            Self::V6 => "v6",
            Self::V9 => "v9",
        }
    }

    /// The module's row form number.
    pub fn form(self) -> u32 {
        match self {
            Self::V3 => 1,
            Self::V4 => 2,
            Self::V6 => 4,
            Self::V9 => 7,
        }
    }

    /// The order of the int4 words on the card under this row loop: tiled
    /// for v9, which reads them so, and the file's for every other form,
    /// which reads the file's. Not a field of its own, so a config cannot
    /// name v9 with the file's order (two copies on the card) or v6 with the
    /// tiled one (wrong words).
    pub fn int4_layout(self) -> crate::fused::Int4Layout {
        match self {
            Self::V9 => crate::fused::Int4Layout::Tiled,
            Self::V3 | Self::V4 | Self::V6 => crate::fused::Int4Layout::Served,
        }
    }
}

/// How a served config prefills a prompt.
///
/// `Chain` feeds the prompt as known tokens through the tree verify pass, 32
/// rows a launch, on the words 4E and the eager path hold, read in place, then
/// hands over to 4E v9 with a one-row launch (`Runtime4e::chain_prefill`). The
/// 4B's first token after 1,024 / 4,096 / 8,191 tokens took 0.420 / 2.393 /
/// 6.485 s against 3.951 / 17.641 / 37.037 for the eager prefill, at a phase
/// peak of 3.560 GB against 5.946 at 8,191 (*measured*, L40S,
/// `docs/mesures/large-prefill-ft-2026-11-11.txt`, `large-prefill-2026-11-11.txt`).
/// It is not bit for bit with the eager prefill: its 16-row form read +0.0007 %
/// of paired WikiText-2 perplexity against eager, and 41 of 47 texts identical.
/// The decode after it is 4E v9 unchanged, which owes the eager decode's tokens
/// on the caches the chain left. Refused:
/// * beside any decode but `persistent` with rows `v9` (the pass reads the int4
///   words tiled, row form 7, and hands over to 4E);
/// * beside `prefill_attention`, which names how the EAGER prefill attends;
/// * beside `batch`, whose pool prefills through the eager path;
/// * on Metal ([`Served::check_device`]);
/// * on any shape but the sealed 4B's ([`check_chain_object`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Prefill {
    /// The eager path, in chunks of `crate::model::MAX_PREFILL_ROWS`.
    Eager,
    /// Chains through the tree verify pass: [`CHAIN_ARM`], [`CHAIN_ROWS`] a launch.
    Chain,
}

/// The tree arm the chain prefill runs: a12ipg32ft (the 32-row pass on the
/// served words read in place, the cache read once for a group of 4 rows, the
/// attention tile's full-tile path) on the explicit plan the large-prefill-ft
/// job measured on the 4B, slices q/k/v 8, o 10, gate/up 8, down 20, head 6.
pub const CHAIN_ARM: &str = "a12ipg32ft@8.10.8.20.6";

/// The plan of [`CHAIN_ARM`]: its five slice counts, written once here.
pub const CHAIN_PLAN: [u32; 5] = [8, 10, 8, 20, 6];

/// The rows one chain launch carries: a12ipg32ft's row cap, what the
/// large-prefill-ft job measured.
pub const CHAIN_ROWS: usize = 32;

impl Prefill {
    /// Absent is `Eager`. Read as written, like [`Decode::parse`]; the empty
    /// string and any other word are refused.
    pub fn parse(v: Option<&str>) -> Result<Self, String> {
        match v {
            None | Some("eager") => Ok(Self::Eager),
            Some("chain") => Ok(Self::Chain),
            Some(s) if s.trim().is_empty() => {
                Err("prefill is empty. Write \"chain\", or delete the key for the eager prefill.".into())
            }
            Some(o) => Err(format!("prefill {o:?}: the forms are \"eager\" (or the key absent) and \"chain\"")),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Eager => "eager",
            Self::Chain => "chain",
        }
    }
}

/// The chain prefill against the model's shape: the sealed 4B's only. The
/// pass's slices ([`CHAIN_PLAN`]) and every measurement of it are the 4B's;
/// the tree install refuses the 8B's Tetra o_proj
/// (`Runtime4e::install_tiled`); the 14B's was never installed. Pure, like
/// [`check_persistent_object`].
pub fn check_chain_object(cfg: &candle_transformers::models::qwen3::Config) -> Result<(), String> {
    let got = (
        cfg.num_hidden_layers,
        cfg.hidden_size,
        cfg.intermediate_size,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
        cfg.vocab_size,
    );
    const FOUR_B: (usize, usize, usize, usize, usize, usize, usize) = (36, 2560, 9728, 32, 8, 128, 151_936);
    match got == FOUR_B {
        true => Ok(()),
        false => Err(format!(
            "prefill \"chain\" on a model of (layers, hidden, intermediate, heads, kv heads, head dim, vocab) = \
             {got:?}. The chain prefill carries the sealed 4B's shape {FOUR_B:?} only: its slices {CHAIN_PLAN:?} \
             and its measurements are the 4B's, the tree pass refuses the 8B's Tetra o_proj, and the 14B's was \
             never installed. Delete the prefill key for the eager prefill."
        )),
    }
}

/// The cache 4E's decode holds.
///
/// `Int8` is research arm v9k8 (`llvq-cuda/kernels/pers_4e_k8.cuh`): a
/// group-affine int8 cache filled from the prefill's binary16 caches at the
/// hand-over, each layer's binary16 cache then released, so the int8 caches
/// are the only copy the decode holds (`Runtime4e::set_k8_release`). The
/// round trip held the +-0.12 % tolerance at 8,192 on the 4B, +0.0126 %
/// [-0.0485, +0.0737] (*measured*,
/// `docs/mesures/recherche-debit-kv-ladder-2026-11-14.txt`). Its token symbol
/// holds 8 B of local memory at 128 registers on the L40S, and 4E's load rule
/// (`llvq_cuda::pers::p4e_token_symbol_loads`) refuses that: whether a served
/// module may hold local memory is the operator's decision. So every served
/// door refuses it ([`Served::decoder`]), and only
/// [`Served::decoder_measurement`] builds it, waiving that rule as the
/// `+local` measurement mode does. Refused beside anything but 4E v9.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeCache {
    /// Binary16, the served cache.
    F16,
    /// Int8 alone, the binary16 caches released at the hand-over. NOT SERVED.
    Int8,
}

impl DecodeCache {
    /// Absent is `F16`. Read as written; the empty string and any other word
    /// are refused.
    pub fn parse(v: Option<&str>) -> Result<Self, String> {
        match v {
            None | Some("f16") => Ok(Self::F16),
            Some("int8") => Ok(Self::Int8),
            Some(s) if s.trim().is_empty() => {
                Err("decode_cache is empty. Write \"int8\", or delete the key for the binary16 cache.".into())
            }
            Some(o) => Err(format!("decode_cache {o:?}: the caches are \"f16\" (or the key absent) and \"int8\"")),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Int8 => "int8",
        }
    }
}

/// The measurement modes of the binaries that read a served config: each turns
/// a run into a bench of something. Beside a config that names the persistent
/// engine each is refused by name ([`check_env`]); `LLVQ_CONTEXTS` and
/// `LLVQ_PREFILL_TOKENS` are not here, since they time the served object as
/// served with a longer prompt.
pub const MEASUREMENT_MODES: [&str; 7] = [
    "LLVQ_BENCH_ARMS", "LLVQ_GQA_AB", "LLVQ_GQA_PPL", "LLVQ_KV_CAPTURE", "LLVQ_KV_LADDER", "LLVQ_INT4_WIDE_CHECK",
    "LLVQ_PERS_4E",
];

/// Variables the binaries read once and no longer do. The code behind them was
/// deleted by the cleanup of 2026-11-08 (`docs/audit-architecture-2026-11.md`
/// §4.2); the tag `pre-clean-2026-11` still holds it. Set, each is refused by
/// name, with or without a config, so a command copied from an old journal
/// fails instead of measuring something else.
/// The last four are `planesbench`'s, which `llvq_cuda::REMOVED_BENCH_VARS`
/// refuses on its side.
pub const REMOVED: [&str; 15] = [
    "LLVQ_KV_PREALLOC", "LLVQ_STEPPED", "LLVQ_GRAPH_AB", "LLVQ_GRAPH_DIAG", "LLVQ_KV_AB", "LLVQ_FUSE_AB",
    "LLVQ_PERS_SEGMENT", "LLVQ_TIME_PHASES", "LLVQ_TRIM_AB", "LLVQ_GESTURE_AB", "LLVQ_BOUNDARY_AB",
    "LLVQ_PERS_ARMS", "LLVQ_PERS_MICRO", "LLVQ_SEG_ARMS", "LLVQ_TIME_EVENTS",
];

/// Refuse every name of [`REMOVED`] that `lookup` finds set. `bin/fusedrun`,
/// `bin/gsm8k`, `bin/chat` and `bin/run` call it first, and [`check_env`]
/// calls it for every served binary.
pub fn check_removed(lookup: impl Fn(&str) -> Option<String>) -> Result<(), String> {
    for var in REMOVED {
        if let Some(v) = lookup(var) {
            return Err(format!(
                "{var}={v:?}: this variable was removed on 2026-11-08 with the code it drove. \
                 The tag pre-clean-2026-11 still has it; unset it here."
            ));
        }
    }
    Ok(())
}

/// The persistent engine against the model's shape: the sealed Qwen3 shapes
/// 4E reads (the 4B's since 2026-11-05; the 8B's, with its Tetra o and its
/// untied head, and the 14B's, with its untied head, since 2026-11-07), and the
/// 0.6B's, the speculative decode's drafter (2026-11-07; its tied head is the
/// 4B's case). Any other shape is refused. Pure, on the config candle read
/// from the file.
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
    type Shape = (usize, usize, usize, usize, usize, usize, usize);
    const SEALED: [(&str, Shape); 4] = [
        ("0.6B drafter", (28, 1024, 3072, 16, 8, 128, 151_936)),
        ("4B", (36, 2560, 9728, 32, 8, 128, 151_936)),
        ("8B", (36, 4096, 12288, 32, 8, 128, 151_936)),
        ("14B", (40, 5120, 17408, 40, 8, 128, 151_936)),
    ];
    match SEALED.iter().any(|(_, s)| *s == got) {
        true => Ok(()),
        false => Err(format!(
            "decode \"persistent\" on a model of (layers, hidden, intermediate, heads, kv heads, head dim, \
             vocab) = {got:?}. 4E reads the sealed Qwen3 shapes only, the 4B, the 8B and the 14B, and the \
             0.6B drafter {SEALED:?}. Serve this object with the eager engine (delete the decode key)."
        )),
    }
}

/// The variables this file speaks for, in the order the file lists them.
const SPOKEN_FOR: [&str; 5] =
    ["LLVQ_FUSED_LAYOUT", "LLVQ_EMBED", "LLVQ_ROT_SHARE", "LLVQ_FUSE", "LLVQ_KV"];

/// The variables that reach below the served door and that this file cannot
/// carry. Set beside `LLVQ_CONFIG`, each is refused by name.
///
/// `LLVQ_PREFILL` recompiles the prefill kernel at another `rows:tile` pair,
/// and `LLVQ_METAL_KERNEL` swaps the Metal matvec: both are measurement modes
/// that change the compiled object without a field to say so.
///
/// `LLVQ_GUIDED_WEIGHTS`, `LLVQ_GUIDED_GROUPS` and `LLVQ_GUIDED_CLIP` change how an object is
/// encoded, which a served config, reading an object already made, has no field for
/// (`crate::guided`).
pub const BELOW_THE_DOOR: [&str; 7] = [
    "LLVQ_KERNEL_DIR",
    "LLVQ_TILE_BLOCKS",
    "LLVQ_PREFILL",
    "LLVQ_METAL_KERNEL",
    crate::guided::ENV_DIR,
    crate::guided::ENV_GROUPS,
    crate::guided::ENV_CLIP,
];

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
    check_removed(&lookup)?;
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
    below_the_door("LLVQ_CONFIG", "Unset it, or unset LLVQ_CONFIG and run the bench.", &lookup)?;
    let [layout, embed, rot_share, fuse, kv] = SPOKEN_FOR;
    same(layout, lookup(layout), &f.layout, FusedLayout::parse)?;
    same(embed, lookup(embed), &f.embed, EmbedMode::parse)?;
    same(rot_share, lookup(rot_share), &f.rot_share, RotShare::parse)?;
    same(fuse, lookup(fuse), &f.fuse, FuseMode::parse)?;
    same(kv, lookup(kv), &f.kv, KvMode::parse)?;
    Ok(())
}

/// The first name of [`BELOW_THE_DOOR`] that `lookup` finds set, refused
/// beside `beside`, the error ending on `remedy`.
fn below_the_door(beside: &str, remedy: &str, lookup: &impl Fn(&str) -> Option<String>) -> Result<(), String> {
    for var in BELOW_THE_DOOR {
        if let Some(v) = lookup(var) {
            return Err(format!(
                "{var}={v:?} beside {beside}. That variable reaches below the served \
                 door — it changes what the config cannot describe — so a served run \
                 refuses it by name. {remedy}"
            ));
        }
    }
    Ok(())
}

/// The refusals of [`check_env`] that need no [`ServedFile`], for a served
/// door of a family that has none (the runtime's Inkling engine): every name
/// of [`REMOVED`] and of [`BELOW_THE_DOOR`] that `lookup` finds set, refused
/// by name beside `beside`, what is being served. The lists stay here, one
/// copy for every door. [`MEASUREMENT_MODES`] are not refused: [`check_env`]
/// refuses them beside the persistent decode only, which is Qwen3's.
pub fn check_door(beside: &str, lookup: impl Fn(&str) -> Option<String>) -> Result<(), String> {
    check_removed(&lookup)?;
    below_the_door(beside, "Unset it.", &lookup)
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
        let prefill = PrefillAttn::parse(f.prefill_attention.as_deref())?;
        if f.prefill_attention.is_some() && attention != AttnHeads::Kernel {
            return Err(format!(
                "prefill_attention {:?} beside attention {:?}: the field says how the kernel form attends \
                 a prefill chunk. Write \"attention\": \"kernel\", or delete the prefill_attention key.",
                prefill.label(),
                attention.label()
            ));
        }
        if f.rows.is_some() && decode != Decode::Persistent {
            return Err(format!(
                "rows {:?} beside the eager decode: the field names 4E's row loop. Write \"decode\": \
                 \"persistent\", or delete the rows key.",
                rows.label()
            ));
        }
        let decode_cache = DecodeCache::parse(f.decode_cache.as_deref())?;
        if decode_cache == DecodeCache::Int8 && (decode != Decode::Persistent || rows != Rows4e::V9) {
            return Err(format!(
                "decode_cache \"int8\" beside decode {:?} rows {:?}: the int8 cache is 4E v9's research arm v9k8. Write \
                 \"decode\": \"persistent\" and \"rows\": \"v9\", or delete the decode_cache key.",
                decode.label(),
                rows.label()
            ));
        }
        if decode_cache == DecodeCache::Int8 && f.batch.is_some() {
            return Err("decode_cache \"int8\" beside batch: the batch pool holds binary16 caches.".into());
        }
        let metal_kernels = MetalKernels::parse(f.metal_kernels.as_deref())?;
        if metal_kernels != MetalKernels::Default && (decode != Decode::Eager || f.batch.is_some()) {
            return Err(format!(
                "metal_kernels {:?} beside decode {:?}{}: the field names the kernels of the eager engine on \
                 Metal, which carries neither 4E nor the batch. Write a config of its own for the card.",
                metal_kernels.arm().unwrap_or_default(),
                decode.label(),
                if f.batch.is_some() { " and a batch" } else { "" }
            ));
        }
        let prefill_form = Prefill::parse(f.prefill.as_deref())?;
        if prefill_form == Prefill::Chain {
            if decode != Decode::Persistent || rows != Rows4e::V9 {
                return Err(format!(
                    "prefill \"chain\" beside decode {:?} rows {:?}: the tree pass reads the int4 words tiled (row \
                     form 7) and hands the prompt over to 4E. Write \"decode\": \"persistent\" and \"rows\": \"v9\", \
                     or delete the prefill key.",
                    decode.label(),
                    rows.label()
                ));
            }
            if f.prefill_attention.is_some() {
                return Err(format!(
                    "prefill \"chain\" beside prefill_attention {:?}: that field names how the eager prefill \
                     attends, and the chain prefill runs no eager prefill. Delete one of the two keys.",
                    prefill.label()
                ));
            }
            if f.batch.is_some() {
                return Err(
                    "prefill \"chain\" beside batch: the batch pool prefills through the eager path. Write a config \
                     of its own for the batched decode."
                        .into(),
                );
            }
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
        let batch = match f.batch {
            None => None,
            Some(b) => {
                let why = if attention != AttnHeads::Kernel {
                    Some("an attention other than \"kernel\": the batched attention is the kernel form from position 0")
                } else if kv != KvMode::F16 {
                    Some("a kv other than \"f16\": the batched cache is f16")
                } else if fuse != FuseMode::Off {
                    Some("fuse \"1\": a fused group has no batched kernel")
                } else {
                    None
                };
                if let Some(why) = why {
                    return Err(format!("batch beside {why}. Write a config of its own for the batched decode."));
                }
                Some(BatchServe::of(b)?)
            }
        };
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
            prefill,
            prefill_form,
            decode_cache,
            batch,
            metal_kernels,
            path,
            note: f.note,
        })
    }

    /// The order of the int4 words on the card: tiled under rows "v9", the
    /// file's otherwise ([`Rows4e::int4_layout`]). Handed to the loader by
    /// every served door, so the card holds the int4 bytes once.
    pub fn int4_layout(&self) -> crate::fused::Int4Layout {
        self.rows.int4_layout()
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
        // Named only when it is not the grouped prefill, so every line an
        // earlier config printed reads as it did.
        let decode = match self.prefill {
            PrefillAttn::Grouped => decode,
            PrefillAttn::Kernel => format!("{decode}, prefill attention kernel (8 rows a call)"),
        };
        let decode = match self.batch {
            None => decode,
            Some(b) => format!("{decode}, batch of up to {} sequences of {} positions (gemm)", b.max, b.context),
        };
        let decode = match self.decode_cache {
            DecodeCache::F16 => decode,
            DecodeCache::Int8 => format!("{decode}, decode cache int8 alone (arm v9k8, NOT SERVED)"),
        };
        // Named only under the chain, so every line an earlier config printed
        // reads as it did.
        let decode = match self.prefill_form {
            Prefill::Eager => decode,
            Prefill::Chain => format!("{decode}, prefill chain (tree pass {CHAIN_ARM}, {CHAIN_ROWS} rows a launch)"),
        };
        // Named only when the config names Metal kernels, so every line an
        // earlier config printed reads as it did.
        let decode = match self.metal_kernels.arm() {
            None => decode,
            Some(arm) => format!("{decode}, Metal kernels {arm}"),
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
    /// `Grouped` is licensed on the card by a tolerance (L40S, 2026-10-30 and
    /// 2026-11-01). On Metal it returned the expanded form's logits bit for
    /// bit (`metal-nuit-2026-11-06` M3 and M5 on the 4B,
    /// `qwen-metal-servi-2026-11-26` on the 8B and the 14B), and the operator
    /// served it there on 2026-11-26. Metal accepts it only from a config that
    /// names its Metal kernels: those configs exist for the objects that
    /// passed that identity gate, and a card config that writes `grouped` was
    /// never read against Metal's rounding. The kernel form stays refused on
    /// Metal: no T8 of it there has been granted.
    ///
    /// A config that names Metal kernels is refused off Metal: on a card the
    /// grouped form is a tolerance, not an identity, and the 8B and 14B never
    /// ran it there. `metal` is handed in rather than read from a device so the refusal is
    /// testable on a machine that has no Metal device, and so this function
    /// stays pure like the rest of this module.
    pub fn check_device(&self, metal: bool) -> Result<(), String> {
        if self.batch.is_some() && metal {
            return Err(format!(
                "{}: batch on Metal. The batched decode's kernels are CUDA (kernels/tv_batch_gemm.cu); \
                 serve this object one sequence at a time on Metal.",
                self.path.display(),
            ));
        }
        if self.prefill_form == Prefill::Chain && metal {
            return Err(format!(
                "{}: prefill \"chain\" on Metal. The tree verify pass is a cooperative CUDA launch \
                 (llvq-cuda/kernels/pers_4et.cu); no Metal kernel exists. Serve this object with the eager \
                 prefill on Metal.",
                self.path.display(),
            ));
        }
        if self.decode == Decode::Persistent && metal {
            return Err(format!(
                "{}: decode \"persistent\" on Metal. 4E is one cooperative CUDA launch \
                 (llvq-cuda/kernels/pers_4e.cu); no Metal kernel exists. Serve this object with the \
                 eager engine on Metal.",
                self.path.display(),
            ));
        }
        if self.metal_kernels != MetalKernels::Default && !metal {
            return Err(format!(
                "{}: metal_kernels {:?} off Metal. This is a Metal config; serve the object with its card or CPU \
                 config.",
                self.path.display(),
                self.metal_kernels.arm().unwrap_or_default(),
            ));
        }
        match (self.attention, metal) {
            (AttnHeads::Expanded, _) | (_, false) => Ok(()),
            (AttnHeads::Grouped, true) if self.metal_kernels != MetalKernels::Default => Ok(()),
            (AttnHeads::Kernel, true) => Err(format!(
                "{}: attention \"kernel\" on Metal. kernels/attn_dec.metal returns the \
                 canonical replay's bits (tests/attn_metal.rs), but no T8 of the form on Metal \
                 has been granted. Serve this object with the expanded form on Metal; \
                 bin/metalrun measures the kernel form.",
                self.path.display(),
            )),
            (form, true) => Err(format!(
                "{}: attention {:?} on Metal from a config that names no Metal kernels. Its +-0.12 % \
                 tolerance was measured on an L40S (gqa-ppl-2026-11-01); Metal serves the grouped form \
                 only through a Metal config, one that names its kernels in metal_kernels and exists for \
                 an object that read bit for bit with the expanded form on Metal (qwen3-*-tetra-e4-metal.json). \
                 Serve this object with its Metal config or the expanded form; bin/metalrun measures the others.",
                self.path.display(),
                form.label(),
            )),
        }
    }

    /// Put a loaded model on what this config says: the store, then the form.
    ///
    /// The one place the pair is applied, called by the served door of
    /// `bin/fusedrun` and `bin/mmlu`, and through [`Self::serve`] by
    /// `bin/gsm8k` and `bin/chat`. Four copies would be four chances for one
    /// binary to serve `grouped` while another serves `expanded` on the same
    /// file, which is the defect class [`Served`] exists to close.
    pub fn apply(
        &self,
        model: &mut crate::model::Qwen3,
        device: &candle_core::Device,
    ) -> Result<(), String> {
        self.check_device(device.is_metal())?;
        self.check_metal_kernels(model, device)?;
        if self.decode == Decode::Persistent {
            check_persistent_object(model.config()).map_err(|e| format!("{}: {e}", self.path.display()))?;
        }
        if self.prefill_form == Prefill::Chain {
            check_chain_object(model.config()).map_err(|e| format!("{}: {e}", self.path.display()))?;
        }
        self.apply_form(model, device)
    }

    /// On Metal, under a config that names its kernels: the runtime under
    /// `model` launches those kernels. A door that loaded through
    /// [`load_resolved`], which knows nothing of the field, is refused here
    /// by name instead of serving the default kernels under a config that
    /// says otherwise. Nothing is checked under the default, so the bench's
    /// arms keep their own kernels under the older configs.
    pub fn check_metal_kernels(
        &self,
        model: &crate::model::Qwen3,
        device: &candle_core::Device,
    ) -> Result<(), String> {
        if self.metal_kernels == MetalKernels::Default || !device.is_metal() {
            return Ok(());
        }
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let want = crate::fused_metal::MetalTuning::of_served(self.metal_kernels).map_err(|e| e.to_string())?;
            match crate::fused_metal::tuning_of(model) {
                Some(got) if got == want => Ok(()),
                got => Err(format!(
                    "{}: the config names the Metal kernels {:?} ({want:?}) and the loaded model launches {got:?}. \
                     Load it through served::load_served, which hands the config's kernels to the loader.",
                    self.path.display(),
                    self.metal_kernels.arm().unwrap_or_default(),
                )),
            }
        }
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = model;
            Err(format!("{}: a Metal device in a build without the Metal backend", self.path.display()))
        }
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
        model.set_prefill_attention(self.prefill);
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
    int4: crate::fused::Int4Layout,
    origin: Option<&str>,
) -> candle_core::Result<crate::fused::FusedSealed> {
    // The CPU backend, in every build: a machine without a card, or a
    // device build asked for the CPU.
    if device.is_cpu() {
        return crate::fused_cpu::load_resolved(path, dtype, layout, emode, share, fuse, kv, int4, origin);
    }
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    {
        crate::fused_cuda::load_resolved(path, device, dtype, layout, emode, share, fuse, kv, int4, origin)
    }
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        // The tiled order exists for 4E v9, which has no Metal kernel; the
        // Metal int4 kernels read the file's order.
        if int4 != crate::fused::Int4Layout::Served {
            candle_core::bail!("int4 layout {} on Metal: the tiled order is 4E v9's, a CUDA kernel", int4.name());
        }
        crate::fused_metal::load_resolved(path, device, dtype, layout, emode, share, fuse, kv, origin)
    }
    #[cfg(not(any(
        all(target_os = "linux", feature = "cuda"),
        all(target_os = "macos", feature = "metal")
    )))]
    {
        let _ = (path, device, dtype, layout, emode, share, fuse, kv, int4, origin);
        candle_core::bail!(
            "this build carries no device backend: a sealed artifact can be served \
             through the fused path on a CUDA build on Linux, a Metal build on macOS, \
             or on the CPU (Device::Cpu) in any build"
        )
    }
}

/// The served object as `cfg` names it: [`load_resolved`] with every field
/// of the config, the Metal kernels included. The door a runtime that reads
/// a config whole calls (`llvq-runtime`, `bin/metalrun`'s `cfg:` arms).
pub fn load_served(
    cfg: &Served,
    path: &str,
    device: &candle_core::Device,
    dtype: candle_core::DType,
    origin: Option<&str>,
) -> candle_core::Result<crate::fused::FusedSealed> {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        if device.is_metal() && cfg.metal_kernels != MetalKernels::Default {
            let tuning = crate::fused_metal::MetalTuning::of_served(cfg.metal_kernels)?;
            if cfg.int4_layout() != crate::fused::Int4Layout::Served {
                candle_core::bail!("int4 layout {} on Metal: the tiled order is 4E v9's, a CUDA kernel", cfg.int4_layout().name());
            }
            return crate::fused_metal::load_resolved_tuned(
                path, device, dtype, cfg.layout, cfg.embed, cfg.rot_share, cfg.fuse, cfg.kv, origin, Some(tuning),
            );
        }
    }
    load_resolved(path, device, dtype, cfg.layout, cfg.embed, cfg.rot_share, cfg.fuse, cfg.kv, cfg.int4_layout(), origin)
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
