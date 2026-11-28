//! The file header: what it says, how it is written, and how it is read,
//! fingerprints checked before a single record is touched. `read_all` sits
//! here because it is the header plus every record.

use super::checks::{get_u32, get_u64, put_u32, put_u64};
use super::decode::read_matrix_with;
use super::kinds::{
    CodeKind, Codebooks, KindSet, FIRST_FINGERPRINTED_VERSION, FIRST_KINDED_VERSION, MAGIC_V1,
    MAGIC_V2, MAGIC_V3, MAGIC_V4, MAGIC_V5,
};
use super::write::QuantizedMatrix;
use crate::{Error, Result};
use std::io::{Read, Write};
// Named only by the documentation's links, which resolved in the one file.
#[cfg(doc)]
use super::ArtifactWriter;

/// What a file's header says: which version, how many matrices follow, from
/// [`FIRST_FINGERPRINTED_VERSION`] on which codebook wrote it, and from
/// [`FIRST_KINDED_VERSION`] on which maps its records may belong to.
pub struct Header {
    /// 1 for projections only; 2+ for a file that has the extra sections,
    /// which may be empty (see [`Header::is_self_contained`]). 3 adds tagged
    /// raw-tensor encodings; 4 adds the codebook fingerprint; 5 adds the Tetra
    /// fingerprint, the default code kind and the set of kinds present.
    pub version: u32,
    pub matrices: u32,
    /// The writer's v1 codebook fingerprint, or `None` for a legacy file that
    /// predates the field.
    ///
    /// A `Some` here has already been checked against
    /// [`crate::codebook_fingerprint`]: [`read_header`] refuses a file whose
    /// codebook is not this build's, so reaching a `Header` at all means the
    /// indices about to be read mean what the writer meant.
    pub codebook: Option<u64>,
    /// The writer's Tetra fingerprint, checked the same way against
    /// [`crate::codebook::tetra_fingerprint`]; `None` below v5. Carried and
    /// checked by a v5 Ball file too — the header describes the build, not
    /// only the map in use.
    pub tetra: Option<u64>,
    /// The kind the file's writer used by default. It is **not** what a
    /// record is read as — the record says that itself — and it is not what a
    /// refusal reads either: see [`Self::kinds`].
    pub default_kind: CodeKind,
    /// The kinds this file declares its records may be. `Ball` alone for
    /// every version below v5, which had nothing else to be.
    pub kinds: KindSet,
}

impl Header {
    /// Whether the file's version has the sections for embeddings, norms and
    /// tokenizer (`version >= 2`). It does not say the sections are filled:
    /// [`ArtifactWriter::finish`] writes them empty in the current version, so
    /// a raw quantization output answers `true` and still needs the original
    /// checkpoint until `bin/seal` completes it.
    pub fn is_self_contained(&self) -> bool {
        self.version >= 2
    }

    /// The kind [`ArtifactWriter::push`] used, and nothing more. A tool that
    /// wants to know whether it can handle this file asks [`Self::kinds`]:
    /// the default of a file whose `v_proj` records are int4 is still Tetra,
    /// and a Tetra-only reader that trusted it would walk into the int4 record
    /// four hundred records later.
    pub fn default_kind(&self) -> CodeKind {
        self.default_kind
    }

    /// The kinds the file may hold — what every refusal is written against,
    /// available before the first record is read.
    pub fn kinds(&self) -> KindSet {
        self.kinds
    }

    /// Whether every record is a v1 ball index, the question the runtime
    /// layouts of [`crate::runtime`] ask.
    pub fn is_ball_only(&self) -> bool {
        self.kinds.is_ball_only()
    }
}

/// Write a file header for `version`, kind `Ball`.
///
/// Exposed because the header is no longer `magic + count`: from
/// [`FIRST_FINGERPRINTED_VERSION`] on it also carries the fingerprint, from
/// [`FIRST_KINDED_VERSION`] on a second one and the kind, and a tool that
/// rewrites a file's matrix section byte for byte has to reproduce the whole
/// of it. Hand-rolling the writes is exactly how a `LVQ4` magic ends up over
/// a `LVQ3` body.
pub fn write_header(w: &mut impl Write, version: u32, matrices: u32) -> Result<()> {
    write_header_kind(w, version, matrices, CodeKind::Ball)
}

/// [`write_header`] for one kind: that kind is the default and the only one
/// declared.
pub fn write_header_kind(
    w: &mut impl Write,
    version: u32,
    matrices: u32,
    kind: CodeKind,
) -> Result<()> {
    write_header_kinds(w, version, matrices, kind, KindSet::of(kind))
}

/// The general header write: a default kind and the set of kinds the file
/// declares.
///
/// Anything but `Ball` alone needs [`FIRST_KINDED_VERSION`] or later, and is
/// refused below it rather than written as a header whose records would read
/// as Ball. The default must be one of the declared kinds — a header naming a
/// default outside its own set describes no file this crate can write, and
/// reading one back is refused for the same reason.
pub fn write_header_kinds(
    w: &mut impl Write,
    version: u32,
    matrices: u32,
    kind: CodeKind,
    kinds: KindSet,
) -> Result<()> {
    let magic = match version {
        1 => MAGIC_V1,
        2 => MAGIC_V2,
        3 => MAGIC_V3,
        4 => MAGIC_V4,
        5 => MAGIC_V5,
        v => return Err(Error::UnknownVersion { version: v }),
    };
    if !kinds.contains(kind) {
        return Err(Error::Inconsistent {
            name: "header".into(),
            detail: format!(
                "default kind {kind} is not in the declared set {kinds}: the default is what \
                 every push writes, so a file that excludes it can hold no matrix"
            ),
        });
    }
    if !kinds.is_ball_only() && version < FIRST_KINDED_VERSION {
        return Err(Error::Inconsistent {
            name: "header".into(),
            detail: format!(
                "a {kinds} file needs format v{FIRST_KINDED_VERSION}: v{version} carries no \
                 code kind, and its records would read as Ball"
            ),
        });
    }
    w.write_all(magic)?;
    put_u32(w, matrices)?;
    if version >= FIRST_FINGERPRINTED_VERSION {
        put_u64(w, crate::codebook_fingerprint())?;
    }
    if version >= FIRST_KINDED_VERSION {
        put_u64(w, crate::codebook::tetra_fingerprint())?;
        put_u32(w, kind.tag())?;
        put_u32(w, kinds.bits())?;
    }
    Ok(())
}

/// Read the file header, refusing a file this build cannot interpret.
///
/// Separate from [`read_all`] because a 4B model's codes are 14 GB of lattice
/// points: anything that walks a real artifact has to do it one matrix at a
/// time, and holding them all was never an option.
///
/// The fingerprints are checked here rather than at first decode so that a
/// codebook disagreement costs a few header bytes of I/O instead of a
/// gigabyte, and so that every caller gets the check without asking for it.
/// The kind is read last and refused if unknown: past an unknown kind no
/// record has a width.
pub fn read_header(r: &mut impl Read) -> Result<Header> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)
        .map_err(|_| Error::Truncated { reading: "magic" })?;
    let version = if &magic == MAGIC_V5 {
        5
    } else if &magic == MAGIC_V4 {
        4
    } else if &magic == MAGIC_V3 {
        3
    } else if &magic == MAGIC_V2 {
        2
    } else if &magic == MAGIC_V1 {
        1
    } else {
        return Err(Error::NotAnArtifact { got: magic });
    };
    let matrices = get_u32(r, "matrix count")?;
    let codebook = if version >= FIRST_FINGERPRINTED_VERSION {
        Some(check_fingerprint(
            get_u64(r, "codebook fingerprint")?,
            crate::codebook_fingerprint(),
            "v1 ball",
        )?)
    } else {
        None
    };
    let (tetra, kind, kinds) = if version >= FIRST_KINDED_VERSION {
        let tetra = check_fingerprint(
            get_u64(r, "tetra fingerprint")?,
            crate::codebook::tetra_fingerprint(),
            "Tetra",
        )?;
        let kind = CodeKind::from_tag(get_u32(r, "code kind")?)?;
        let kinds = KindSet::from_bits(get_u32(r, "code kinds present")?)?;
        if !kinds.contains(kind) {
            return Err(Error::Inconsistent {
                name: "header".into(),
                detail: format!(
                    "default kind {kind} is not in the file's set {kinds}: one of the two \
                     fields is corrupt, and which one decides how every record is read"
                ),
            });
        }
        (Some(tetra), kind, kinds)
    } else {
        (None, CodeKind::Ball, KindSet::BALL)
    };
    Ok(Header {
        version,
        matrices,
        codebook,
        tetra,
        default_kind: kind,
        kinds,
    })
}

fn check_fingerprint(stored: u64, computed: u64, which: &'static str) -> Result<u64> {
    if stored != computed {
        return Err(Error::CodebookMismatch {
            which,
            stored,
            computed,
        });
    }
    Ok(stored)
}

/// Read every matrix back, each through the map its own record names. Only
/// safe for small models — see [`read_header`].
pub fn read_all(r: &mut impl Read) -> Result<Vec<QuantizedMatrix>> {
    let h = read_header(r)?;
    // At the header, not at the record: a file declaring int4 would otherwise
    // fail somewhere in the middle with `NotALatticeRecord`, which is correct
    // and late.
    if h.kinds.contains(CodeKind::Int4G128) {
        return Err(Error::Inconsistent {
            name: "header".into(),
            detail: format!(
                "read_all decodes lattice records; this file declares {} — use read_record",
                h.kinds
            ),
        });
    }
    let cbs = Codebooks::new();
    (0..h.matrices)
        .map(|_| read_matrix_with(r, h.version, &cbs))
        .collect()
}
