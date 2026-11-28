//! The two choices a fused load resolves first: which runtime layout the
//! matrices take ([`FusedLayout`]) and how the embedding tables sit on the
//! device ([`EmbedMode`]), with the q8 and q4 tables and their footprint report.

/// Which runtime layout the fused path reads.
///
/// Resolved once, from `LLVQ_FUSED_LAYOUT`, before any transcoding: the whole
/// model is one layout, the kernel is chosen by it, and the two cannot drift
/// apart because every [`HostStream`] carries its variant with it.
///
/// `Planes14` is the default — the reference layout since C1 (2026-08-06,
/// 1.14× over `Slot32` at identical decoded content, 4.804 against 5.510
/// b/weight on the published 4B). `Planes12x` is the sparse overlay measured
/// at the bench on 2026-08-07 (4.342 b/weight, 1.98× against FP16, exact
/// reconstruction) and wired here so it can be measured *in the model*;
/// ⚠️ on the 8B it does not pay — `Planes14` + a q8 embedding already sits
/// under the AWQ 4-bit at 5.322 b/param — and the reason it is here anyway is
/// the 14B, where the carried share falls back to ~10.5 %
/// (`docs/mesures/rtbits-planes-8b-2026-08-09.txt`). `Slot32` stays as the
/// comparison arm and the fallback, bit-identical to what shipped before the
/// switch existed.
///
/// `Golay70` is the E2 layout, wired 2026-08-11 as step 5 of the v2 campaign
/// (`docs/archive/passation-golay70-2026-08-11.md`): the 9-byte main stream whose
/// per-slot decode goes through the Golay codeword rank, plus the exact
/// exception records — 3.589 b/weight at the bench, the only layout whose
/// whole-model b/param beats the deployed AWQ at every scale
/// (`docs/archive/projections-golay70-2026-08-11.md` §2). Its kernel is the v2
/// block-prologue decoder; whether it is SERVED is decided by the
/// pre-registered criterion of `proofs/preregistration-2026-08-11.md`, not
/// by this wiring — being selectable is what makes it measurable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FusedLayout {
    Planes14,
    Planes12x,
    Slot32,
    Golay70,
    /// The Tetra word, read as it is written. The only layout here that does
    /// **not** unfold a ball index into a kernel-shaped record — it addresses
    /// blocks row-strided rather than flat, because a six-byte record is not
    /// u32-aligned and a row must start on a boundary or every row after the
    /// first reads at a shifted phase (`llvq_artifact::tetra48`).
    Tetra48,
}

impl FusedLayout {
    /// Parse the value of `LLVQ_FUSED_LAYOUT`. `None` (unset) and the empty
    /// string mean the default; anything else must name a layout exactly —
    /// a typo silently falling back to a default would make an A/B lie.
    pub fn parse(v: Option<&str>) -> Result<Self, String> {
        match v {
            None | Some("") => Ok(Self::Planes14),
            Some("planes14") => Ok(Self::Planes14),
            Some("planes12x") => Ok(Self::Planes12x),
            Some("slot32") => Ok(Self::Slot32),
            Some("golay70") => Ok(Self::Golay70),
            Some("tetra48") => Ok(Self::Tetra48),
            Some(other) => Err(format!(
                "LLVQ_FUSED_LAYOUT={other}: accepted values \"planes14\" (default), \
                 \"planes12x\", \"slot32\", \"golay70\" and \"tetra48\""
            )),
        }
    }

    /// Resolve from the environment.
    pub fn from_env() -> Result<Self, String> {
        let v = std::env::var("LLVQ_FUSED_LAYOUT").ok();
        Self::parse(v.as_deref())
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Planes14 => "planes14",
            Self::Planes12x => "planes12x",
            Self::Slot32 => "slot32",
            Self::Golay70 => "golay70",
            Self::Tetra48 => "tetra48",
        }
    }
}

/// How the carried embedding tables — the input table, and the `lm_head` when
/// the model unties it — sit on the device.
///
/// Resolved once, from `LLVQ_EMBED`, and printed next to the bytes it
/// decides. `F16` is the shipped behaviour — the embedding decoded to an f16
/// tensor, 778.1 MB on the 4B. `Q8` keeps it as the int8 g64 payload lot B
/// validated (ppl 16.9379 identical, MMLU within sigma): 388.96 MB of int8
/// plus 24.31 MB of f16 scales and biases, read by two dedicated kernels.
/// `Q4` is the same scheme at four bits, the payload `bin/embedq q4` writes
/// and the 2026-09-23 census scored (`embed-q4-swap-2026-09-23.txt`): 194.48
/// MB of nibbles plus the same 24.31 MB of scales and biases, read by two
/// kernels of its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbedMode {
    F16,
    Q8,
    Q4,
}

impl EmbedMode {
    /// Parse the value of `LLVQ_EMBED`. Same contract as [`FusedLayout`]:
    /// unset and empty mean the default, anything else must name a mode
    /// exactly — a typo silently falling back would make an A/B lie.
    pub fn parse(v: Option<&str>) -> Result<Self, String> {
        match v {
            None | Some("") => Ok(Self::F16),
            Some("f16") => Ok(Self::F16),
            Some("q8") => Ok(Self::Q8),
            Some("q4") => Ok(Self::Q4),
            Some(other) => Err(format!(
                "LLVQ_EMBED={other}: accepted values \"f16\" (default), \"q8\" and \"q4\""
            )),
        }
    }

    /// Resolve from the environment.
    pub fn from_env() -> Result<Self, String> {
        let v = std::env::var("LLVQ_EMBED").ok();
        Self::parse(v.as_deref())
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Q8 => "q8",
            Self::Q4 => "q4",
        }
    }

    /// The width of the quantized payload, `None` when the tables stay f16.
    pub fn bits(&self) -> Option<u8> {
        match self {
            Self::F16 => None,
            Self::Q8 => Some(8),
            Self::Q4 => Some(4),
        }
    }
}

/// The group width of the q8 embedding path — MLX's scheme, and the one
/// `bin/embedq` wrote into the artifact lot B scored. The kernels hard-code
/// it (`c >> 6`), so it is a constant, not a knob.
pub const EMBED_GROUP: usize = 64;

/// Turn a carried tensor into the int8 g64 payload the embedding kernels read.
///
/// Two accepted inputs, one output:
///
///  * an f16 tensor is quantized **by the same function `bin/embedq` calls**
///    (`embedquant::quantize_affine`, bits = 8, group = 64) — that call, not a
///    reimplementation, is what transfers lot B's quality verdict to this
///    path;
///  * a tensor already stored as int8 g64 (an artifact `embedq` produced) is
///    passed through byte-identical;
///  * anything else — int4, another group — is refused rather than silently
///    requantized: the validated object is q8 g64 and nothing next to it.
pub fn embed_q8(t: llvq_artifact::RawTensor) -> Result<llvq_artifact::RawTensor, String> {
    embed_at(t, 8)
}

/// [`embed_q8`] at four bits: an f16 tensor is quantized by the function
/// `bin/embedq q4` calls, a tensor already int4 g64 passes through
/// byte-identical, and anything else is refused. A q8 file is not
/// requantized to q4, nor the reverse: each would score an object nobody
/// measured.
pub fn embed_q4(t: llvq_artifact::RawTensor) -> Result<llvq_artifact::RawTensor, String> {
    embed_at(t, 4)
}

/// The rule [`embed_q8`] and [`embed_q4`] share, at `bits`.
fn embed_at(t: llvq_artifact::RawTensor, bits: u8) -> Result<llvq_artifact::RawTensor, String> {
    match &t.data {
        llvq_artifact::RawData::F16(_) => {
            crate::embedquant::quantize_affine(&t, bits, EMBED_GROUP).map_err(|e| e.to_string())
        }
        llvq_artifact::RawData::Quant(q) if q.bits == bits && q.group == EMBED_GROUP => Ok(t),
        llvq_artifact::RawData::Quant(q) => Err(format!(
            "{}: int{} g{} carried by the file, the q{bits} path reads only int{bits} g{EMBED_GROUP}",
            t.name, q.bits, q.group
        )),
    }
}

/// Device bytes of the q8 embedding: `(packed, scales + biases)`.
///
/// Pure arithmetic on the dims, so the announced footprint is testable
/// without a card: on `[151936, 2560]` this is 388,956,160 + 24,309,760
/// bytes — the 413.3 MB the mission statement quotes.
pub fn q8_device_bytes(dims: &[usize]) -> (u64, u64) {
    let row_len = dims.last().copied().unwrap_or(1).max(1);
    let n: usize = dims.iter().product();
    let rows = n / row_len;
    let gpr = row_len.div_ceil(EMBED_GROUP);
    (n as u64, (rows * gpr) as u64 * 4)
}

/// Device bytes of the q4 embedding: `(packed, scales + biases)`. Half the
/// payload of [`q8_device_bytes`] and the same side data: on `[151936, 2560]`
/// 194,478,080 + 24,309,760 bytes.
pub fn q4_device_bytes(dims: &[usize]) -> (u64, u64) {
    let (n, sb) = q8_device_bytes(dims);
    (n.div_ceil(2), sb)
}

/// The carried tensor holding the input embedding.
pub const EMBED_NAME: &str = "model.embed_tokens.weight";
/// The carried tensor holding the output projection. Present only when the
/// model unties its two ends — Qwen3-4B has `tie_word_embeddings = true` and
/// carries no such tensor, Qwen3-8B has it `false` and carries a second table
/// of the same shape and different values.
pub const HEAD_NAME: &str = "lm_head.weight";

/// The embedding tables the q8 path takes off the carried list, quantized.
///
/// One table when the model ties its two ends: a single device buffer feeds
/// the gather at the input and the `lm_head` at the output, which is the −365
/// MB lot B validated on the 4B. Two tables when it unties them: they are
/// different weights, so they are different buffers, and nothing in this path
/// may substitute one for the other.
pub struct EmbedTables {
    /// [`EMBED_NAME`], int8 g64.
    pub embed: llvq_artifact::RawTensor,
    /// [`HEAD_NAME`], int8 g64 — `None` exactly when the two ends are tied.
    pub head: Option<llvq_artifact::RawTensor>,
}

impl EmbedTables {
    /// The distinct payloads to upload, in buffer order.
    pub fn buffers(&self) -> Vec<&llvq_artifact::RawTensor> {
        let mut v = vec![&self.embed];
        v.extend(self.head.iter());
        v
    }

    /// Which uploaded buffer each end of the model reads: `(embedding, head)`.
    ///
    /// The wiring is returned as a **value** rather than decided at the call
    /// site, and that is deliberate. The call site sits behind
    /// `cfg(target_os = "linux", feature = "cuda")` and compiles on no machine
    /// this suite runs on, so an `lm_head` pointed at the embedding's buffer
    /// would be caught by nothing until a billed job produced plausible wrong
    /// logits. As data, it is pinned by a test that runs anywhere.
    pub fn wiring(&self) -> (usize, usize) {
        match self.head {
            None => (0, 0),
            Some(_) => (0, 1),
        }
    }
}

/// Take the embedding tables off the carried list and turn them into the int8
/// g64 payload the embedding kernels read.
///
/// `tie` is the model's `tie_word_embeddings`. Under `false` the file must
/// carry [`HEAD_NAME`] too, and its absence is an error that names it — never
/// a silent fall back on the embedding, which would produce a model that runs
/// and is wrong.
///
/// Both tables go through [`embed_q8`], so both accept either an f16 tensor
/// (quantized by the very function `bin/embedq` calls) or bytes `embedq`
/// already wrote, passed through byte-identical.
pub fn take_embed_tables(
    raw: &mut Vec<llvq_artifact::RawTensor>,
    tie: bool,
) -> Result<EmbedTables, String> {
    take_embed_tables_at(raw, tie, 8)
}

/// [`take_embed_tables`] at `bits` (8 or 4), through `embed_at`.
pub fn take_embed_tables_at(
    raw: &mut Vec<llvq_artifact::RawTensor>,
    tie: bool,
    bits: u8,
) -> Result<EmbedTables, String> {
    fn take(
        raw: &mut Vec<llvq_artifact::RawTensor>,
        name: &str,
    ) -> Option<llvq_artifact::RawTensor> {
        // `swap_remove` reorders what is left, so the second lookup below
        // searches the mutated vector by name rather than reusing an index.
        let i = raw.iter().position(|t| t.name == name)?;
        Some(raw.swap_remove(i))
    }
    let embed = take(raw, EMBED_NAME).ok_or_else(|| format!("does not carry {EMBED_NAME}"))?;
    let embed = embed_at(embed, bits)?;
    let head = if tie {
        None
    } else {
        let t = take(raw, HEAD_NAME).ok_or_else(|| {
            format!(
                "tie_word_embeddings=false, but the file does not carry {HEAD_NAME}. The \
                 two ends are untied, and the q{bits} path never substitutes the embedding \
                 for a missing lm_head"
            )
        })?;
        Some(embed_at(t, bits)?)
    };
    Ok(EmbedTables { embed, head })
}

/// The embedding tables still carried as ordinary tensors, embedding first and
/// `lm_head` second whatever order the file wrote them in.
pub fn carried_embed_tables(raw: &[llvq_artifact::RawTensor]) -> Vec<&llvq_artifact::RawTensor> {
    [EMBED_NAME, HEAD_NAME]
        .iter()
        .filter_map(|n| raw.iter().find(|t| t.name == *n))
        .collect()
}

/// What the embedding tables cost on the device, and the line that says so.
///
/// Arithmetic on dims alone, so the announced footprint is checkable without a
/// card. It exists because the line it replaces announced **one** table's
/// bytes on a model carrying two: with the ends untied it under-reported by a
/// whole table — 1.24 GB on the 8B — while the total printed just below it was
/// right. Half-true is the worst kind of wrong in a footprint report, because
/// the two lines then contradict each other and neither is obviously the liar.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbedReport {
    pub mode: EmbedMode,
    /// One entry per table resident on the device, in buffer order:
    /// `(name, payload bytes, scale/bias bytes)`. The last is zero at f16.
    pub tables: Vec<(String, u64, u64)>,
}

impl EmbedReport {
    /// Build the report for the tables the device will actually hold.
    pub fn new(mode: EmbedMode, tables: &[&llvq_artifact::RawTensor]) -> Self {
        let tables = tables
            .iter()
            .map(|t| match mode {
                EmbedMode::F16 => (t.name.clone(), t.len() as u64 * 2, 0),
                EmbedMode::Q8 => {
                    let (packed, sb) = q8_device_bytes(&t.dims);
                    (t.name.clone(), packed, sb)
                }
                EmbedMode::Q4 => {
                    let (packed, sb) = q4_device_bytes(&t.dims);
                    (t.name.clone(), packed, sb)
                }
            })
            .collect();
        Self { mode, tables }
    }

    /// Payload bytes over every table.
    pub fn packed(&self) -> u64 {
        self.tables.iter().map(|&(_, p, _)| p).sum()
    }

    /// Scale and bias bytes over every table; zero at f16.
    pub fn meta(&self) -> u64 {
        self.tables.iter().map(|&(_, _, s)| s).sum()
    }

    /// Device bytes of every embedding table together.
    pub fn total(&self) -> u64 {
        self.packed() + self.meta()
    }

    /// The load-time line — it names how many tables it is counting, so a
    /// reader can tell a tied model from an untied one without doing division.
    ///
    /// `source` is what decided the mode: `"LLVQ_EMBED"` in bench mode, the
    /// served config's name on the served path. Printed in place of the
    /// variable name, for the reason `fused_cuda::load_resolved` gives: a
    /// served log must not attribute a choice to a variable nobody set.
    pub fn line(&self, source: &str) -> String {
        let names: Vec<&str> = self.tables.iter().map(|(n, ..)| n.as_str()).collect();
        let which = if names.len() == 1 {
            format!("1 table ({}, lm_head tied on it)", names[0])
        } else {
            format!("{} tables ({})", names.len(), names.join(" + "))
        };
        match self.mode {
            EmbedMode::F16 => format!(
                "embedding: f16 ({source}), {which}, {:.1} MB on the card",
                self.total() as f64 / 1e6
            ),
            EmbedMode::Q8 => format!(
                "embedding: q8 g64 ({source}), {which}, {:.1} MB on the card \
                 (int8 {:.1} + scales/biases {:.1})",
                self.total() as f64 / 1e6,
                self.packed() as f64 / 1e6,
                self.meta() as f64 / 1e6
            ),
            EmbedMode::Q4 => format!(
                "embedding: q4 g64 ({source}), {which}, {:.1} MB on the card \
                 (int4 {:.1} + scales/biases {:.1})",
                self.total() as f64 / 1e6,
                self.packed() as f64 / 1e6,
                self.meta() as f64 / 1e6
            ),
        }
    }
}
