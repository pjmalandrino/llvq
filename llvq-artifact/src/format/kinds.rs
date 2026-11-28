//! The format's vocabulary: the magic words and versions, the code kinds a
//! record can be, the set a header declares, and the maps that read them.
//! One file because every other part of the format is written against these.

use crate::{Error, Result};
use core::fmt;
use llvq_core::Point;
use llvq_search::index::Indexer;
use llvq_search::tetra::{Tetra, LABEL_BITS};
use std::sync::OnceLock;
// Named only by the documentation's links, which resolved in the one file.
#[cfg(doc)]
use super::checks::index_width;
#[cfg(doc)]
use super::{
    read_header, read_matrix_raw, write_matrix_int4, write_matrix_raw, ArtifactWriter, Int4Matrix,
};

/// Format identifier of [`DEFAULT_VERSION`], what [`ArtifactWriter::new`]
/// emits.
///
/// `LVQ1` held quantized projections and nothing else, so a file needed the
/// original checkpoint beside it to run. `LVQ2` adds the raw tensors and the
/// blobs that make it self-contained. `LVQ3` tags each raw tensor with its
/// encoding so the embedding can be carried group-affine quantized instead of
/// f16 (see [`crate::sealed`]). `LVQ4` appends the writer's codebook
/// fingerprint to the header (see [`crate::codebook`]), the first field of
/// the whole format that describes what an index *means* rather than how
/// wide it is. `LVQ5` appends the Tetra fingerprint, the file's default
/// [`CodeKind`] and the [`KindSet`] its records may be — and gives every
/// record a kind of its own, which is what a mixed file needs. All
/// five are readable; `LVQ4` is written for Ball, `LVQ5` for Tetra. The
/// matrix records are identical across the first four versions — which is
/// what lets a tool copy them between two of those files untouched; a `LVQ5`
/// record adds its code kind, so a copy across that boundary passes each
/// file's own version to [`read_matrix_raw`] and [`write_matrix_raw`].
pub const MAGIC: &[u8; 4] = MAGIC_V4;
pub const MAGIC_V5: &[u8; 4] = b"LVQ5";
pub const MAGIC_V4: &[u8; 4] = b"LVQ4";
pub const MAGIC_V3: &[u8; 4] = b"LVQ3";
pub const MAGIC_V2: &[u8; 4] = b"LVQ2";
pub const MAGIC_V1: &[u8; 4] = b"LVQ1";

/// The newest version this crate reads and writes.
pub const VERSION: u32 = 5;

/// The version [`ArtifactWriter::new`] emits.
///
/// Deliberately not [`VERSION`]: a Ball file gains nothing from a v5 header,
/// and holding the default at 4 keeps every Ball file — the served 4B's
/// path included — byte-identical to what the same writer produced before
/// Tetra existed (`the_default_writer_still_writes_v4`). A Tetra file cannot
/// be a v4 file, and [`ArtifactWriter::with_kind`] picks its version from
/// the kind.
pub const DEFAULT_VERSION: u32 = 4;

/// First version whose header carries a codebook fingerprint.
///
/// Files below it were written before the field existed and are read without
/// the check — grandfathered deliberately: the published Qwen3-4B artifact is
/// `LVQ2`, and a reader that refused it would be worse than the hole it
/// closes. What protects *those* files is the pinned fingerprint in the test
/// suite, not a field they cannot have.
pub const FIRST_FINGERPRINTED_VERSION: u32 = 4;

/// First version whose header carries a default [`CodeKind`], the set of
/// kinds its records may be ([`KindSet`]) and the Tetra fingerprint, and whose
/// **records** each carry their own kind. Every file below it is a Ball file
/// of Ball records: there was nothing else to be.
pub const FIRST_KINDED_VERSION: u32 = 5;

/// What a Tetra record carries in its `shell_cap` field.
///
/// The field is not read for Tetra; 12 keeps `index_bits` at 47 so a reader
/// that ignored the kind would at least fail loudly on the fingerprint, not
/// silently on the width — its 47-bit reads would stay aligned with the
/// stream, and every one of them would be an index into the wrong map. The
/// reader consults the record's kind BEFORE any width ([`read_matrix_raw`]),
/// takes Tetra's width from [`LABEL_BITS`], and refuses a Tetra record whose
/// field holds anything else: such a record was never written by this crate.
pub const TETRA_SHELL_CAP: u32 = 12;

/// Which map turns a matrix's indices into lattice points.
///
/// Stored as a `u32` tag in the v5 header (the file's default) and in every
/// v5 record (`0 = Ball`, `1 = Tetra`); a tag this build does not know is
/// [`Error::UnknownCodeKind`], refused where it is read — past an unknown
/// kind neither the header's set nor the record's width means anything.
/// Every file before v5 is `Ball`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CodeKind {
    /// The v1 ball: 383 classes of `Λ₂₄(13)`, indexed by
    /// [`llvq_search::index::Indexer`], fingerprinted by
    /// [`crate::codebook_fingerprint`].
    Ball,
    /// The three-section trellis word of [`llvq_search::tetra`], fingerprinted
    /// by [`crate::codebook::tetra_fingerprint`].
    Tetra,
    /// Affine int4 over groups of [`INT4G128_GROUP`] weights, stored rather
    /// than indexed: no map, no fingerprint, nothing to be wrong about but the
    /// arithmetic. [`Int4Matrix`] is the record.
    Int4G128,
}

/// The tag of Q5's `int4 g128` matrices, kept as a name because the promise
/// it carried was about the *number*: a file written when tag 2 was reserved
/// and a file written now cannot disagree about what a 2 means.
/// [`write_matrix_int4`] is the writer that promise was waiting for.
/// `the_int4_tag_is_the_int4_kind` holds the two halves together.
pub const RESERVED_INT4G128_TAG: u32 = CodeKind::Int4G128.tag();

/// What an [`CodeKind::Int4G128`] record carries in its `shell_cap` field.
///
/// A sentinel and not a cap: the record indexes no lattice, so the field has
/// nothing true to say and is made to say something impossible instead.
/// `u32::MAX` is above [`llvq_search::classes::MAX_SHELL`], so an int4 record
/// whose tag is corrupted to Ball is refused by [`index_width`] before any
/// class is enumerated, and it is not [`TETRA_SHELL_CAP`], so the same record
/// read as Tetra is refused too. `the_sentinel_cap_is_illegal_for_both_lattice_kinds`
/// is that argument, checked.
pub const INT4G128_SHELL_CAP: u32 = u32::MAX;

/// Bits per weight of an [`CodeKind::Int4G128`] record. The only value the
/// reader and the writer accept.
pub const INT4G128_BITS: u8 = 4;

/// Weights per scale/bias pair. 128 to match the AWQ w4 g128 the cost table
/// compares against; `llvq_llm::sealed::Q4_GROUP` is derived from this
/// constant rather than written twice.
pub const INT4G128_GROUP: usize = 128;

impl CodeKind {
    /// Every kind this build knows, in tag order — what [`KindSet`] iterates
    /// and what a reader of an unknown tag is being measured against.
    pub const ALL: [CodeKind; 3] = [CodeKind::Ball, CodeKind::Tetra, CodeKind::Int4G128];

    /// The header and record tag.
    pub const fn tag(self) -> u32 {
        match self {
            Self::Ball => 0,
            Self::Tetra => 1,
            Self::Int4G128 => 2,
        }
    }

    /// The kind a tag names, or [`Error::UnknownCodeKind`] for tag 3 and
    /// above.
    pub fn from_tag(tag: u32) -> Result<Self> {
        match tag {
            0 => Ok(Self::Ball),
            1 => Ok(Self::Tetra),
            2 => Ok(Self::Int4G128),
            _ => Err(Error::UnknownCodeKind { tag }),
        }
    }

    /// This kind's bit in a [`KindSet`]: `1 << tag`, so the set and the tag
    /// can never drift apart.
    pub const fn bit(self) -> u32 {
        1u32 << self.tag()
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Ball => "Ball",
            Self::Tetra => "Tetra",
            Self::Int4G128 => "Int4G128",
        }
    }

    /// The version [`ArtifactWriter::with_kind`] emits for this kind:
    /// [`DEFAULT_VERSION`] for Ball, [`FIRST_KINDED_VERSION`] for everything
    /// else. A v4 record has nowhere to put a kind tag.
    pub const fn default_version(self) -> u32 {
        match self {
            Self::Ball => DEFAULT_VERSION,
            Self::Tetra | Self::Int4G128 => FIRST_KINDED_VERSION,
        }
    }
}

impl fmt::Display for CodeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// The kinds a file declares its records may be — the v5 header's
/// `kinds_present` bitmask, `1 << tag` per kind.
///
/// It exists because the header is the only place a refusal can be cheap and
/// early. A tool with no runtime layout for Tetra must stop before it reads
/// the first record of a file that holds one Tetra matrix among four hundred
/// Ball ones; the default kind alone cannot tell it that, and walking the
/// records to find out is the read the refusal was meant to avoid.
///
/// It is a **declaration, checked**, not a tally: [`ArtifactWriter`] writes
/// the header before it sees a single matrix — a 14 GB stream cannot be
/// buffered and cannot be seeked — so the set is fixed at construction and
/// every [`ArtifactWriter::push_kind`] is refused unless the set already
/// contains its kind ([`Error::KindNotDeclared`]). That makes the stored set
/// an upper bound on what the records are, which is the safe direction: a
/// refusal reading it can be too cautious, never too permissive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KindSet(u32);

impl KindSet {
    /// The only set a file below [`FIRST_KINDED_VERSION`] can have.
    pub const BALL: Self = Self::of(CodeKind::Ball);

    pub const fn empty() -> Self {
        Self(0)
    }

    pub const fn of(kind: CodeKind) -> Self {
        Self(kind.bit())
    }

    /// The stored `u32`.
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// The set a stored `u32` names, or [`Error::UnknownCodeKind`] for the
    /// lowest bit this build cannot name — a newer writer's kind, or a
    /// corrupted field. Refused rather than masked off: a set silently
    /// narrowed to the bits we understand would let a file with one
    /// int4 g128 matrix pass a Ball-only refusal.
    pub fn from_bits(bits: u32) -> Result<Self> {
        let known = CodeKind::ALL.iter().fold(0u32, |m, k| m | k.bit());
        let unknown = bits & !known;
        if unknown != 0 {
            return Err(Error::UnknownCodeKind {
                tag: unknown.trailing_zeros(),
            });
        }
        Ok(Self(bits))
    }

    pub const fn contains(self, kind: CodeKind) -> bool {
        self.0 & kind.bit() != 0
    }

    pub const fn with(self, kind: CodeKind) -> Self {
        Self(self.0 | kind.bit())
    }

    pub fn insert(&mut self, kind: CodeKind) {
        self.0 |= kind.bit();
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Whether every record of the file is a v1 ball index — the question
    /// [`crate::runtime::require_ball_kinds`] asks, and the only one a
    /// transcoder of this crate can answer yes to.
    pub const fn is_ball_only(self) -> bool {
        self.0 == Self::BALL.0
    }

    /// The kinds in the set, in tag order.
    pub fn iter(self) -> impl Iterator<Item = CodeKind> {
        CodeKind::ALL.into_iter().filter(move |k| self.contains(*k))
    }
}

impl From<CodeKind> for KindSet {
    fn from(kind: CodeKind) -> Self {
        Self::of(kind)
    }
}

impl fmt::Display for KindSet {
    /// `Ball`, `Tetra`, `Ball+Tetra` — and `none` for the empty set, which no
    /// header may carry ([`read_header`] refuses one) but a message about a
    /// refused header still has to print.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            return f.write_str("none");
        }
        for (i, k) in self.iter().enumerate() {
            if i > 0 {
                f.write_str("+")?;
            }
            f.write_str(k.name())?;
        }
        Ok(())
    }
}

/// The map of one [`CodeKind`], built once per process and shared across
/// matrices — building either enumerates tables that have no business being
/// rebuilt per record (383 classes for the ball, a 16 KiB table and a
/// trellis, all re-derived and asserted, for Tetra).
pub enum Codebook {
    Ball(Box<Indexer>),
    Tetra(Box<Tetra>),
}

impl Codebook {
    /// The map of a lattice kind. [`CodeKind::Int4G128`] is refused rather
    /// than defaulted to the ball: an int4 record's bytes are weights, and a
    /// caller handed the ball's map for them would decode nonsense without an
    /// error.
    pub fn new(kind: CodeKind) -> Result<Self> {
        match kind {
            CodeKind::Ball => Ok(Self::Ball(Box::new(Indexer::new()))),
            CodeKind::Tetra => Ok(Self::Tetra(Box::new(Tetra::new()))),
            CodeKind::Int4G128 => Err(Error::NoCodebookForKind { kind }),
        }
    }

    pub fn kind(&self) -> CodeKind {
        match self {
            Self::Ball(_) => CodeKind::Ball,
            Self::Tetra(_) => CodeKind::Tetra,
        }
    }

    /// The index of a point, or `None` for a point the map has no word for.
    /// A Tetra word comes back with its gain bit (bit 47) clear.
    pub fn encode(&self, point: &Point) -> Option<u64> {
        match self {
            Self::Ball(ix) => ix.encode(point),
            Self::Tetra(tetra) => tetra.encode(point),
        }
    }

    /// The point of a stored `(index, gain)` pair.
    ///
    /// For Tetra the pair is put back together as the 48-bit word the kernel
    /// will read, `index | gain << 47`, before decoding — the gain bit is
    /// opaque to [`Tetra::decode`], but the word is the unit of the format,
    /// and a reader that assembled it with the gain at any other bit would
    /// decode a different point (`the_gain_bit_sits_at_bit_47_on_disk`).
    pub fn decode(&self, index: u64, gain: u32) -> Option<Point> {
        match self {
            Self::Ball(ix) => ix.decode(index),
            Self::Tetra(tetra) => Some(tetra.decode(index | u64::from(gain) << LABEL_BITS)),
        }
    }
}

/// The maps a file's records are read through, each built on first use.
///
/// A v5 file may hold records of more than one kind, so a reader cannot pick
/// its map up front; and neither map is free to build — 383 classes
/// enumerated for the ball, a 16 KiB table and a trellis re-derived and
/// asserted for Tetra — so a file with no Tetra record must not pay for Tetra's.
/// One lazy slot per kind is both. `OnceLock` rather than `Option` so the
/// maps come out of a `&self`: [`ArtifactWriter`] holds this beside the sink
/// it borrows mutably, and a `&mut self` here would make the two collide.
#[derive(Default)]
pub struct Codebooks {
    ball: OnceLock<Codebook>,
    tetra: OnceLock<Codebook>,
}

impl Codebooks {
    pub const fn new() -> Self {
        Self {
            ball: OnceLock::new(),
            tetra: OnceLock::new(),
        }
    }

    /// The map of `kind`, built on the first record that needs it, or
    /// [`Error::NoCodebookForKind`] for a kind that indexes nothing.
    pub fn get(&self, kind: CodeKind) -> Result<&Codebook> {
        let slot = match kind {
            CodeKind::Ball => &self.ball,
            CodeKind::Tetra => &self.tetra,
            CodeKind::Int4G128 => return Err(Error::NoCodebookForKind { kind }),
        };
        if let Some(cb) = slot.get() {
            return Ok(cb);
        }
        let built = Codebook::new(kind)?;
        Ok(slot.get_or_init(|| built))
    }

    /// Whether the map of `kind` has been built — the cheap way to see that a
    /// Ball-only file never touched Tetra's tables
    /// (`a_ball_only_file_never_builds_the_tetra_map`).
    pub fn is_built(&self, kind: CodeKind) -> bool {
        match kind {
            CodeKind::Ball => self.ball.get().is_some(),
            CodeKind::Tetra => self.tetra.get().is_some(),
            // Never built: there is no map to build.
            CodeKind::Int4G128 => false,
        }
    }
}
