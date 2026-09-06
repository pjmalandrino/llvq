//! Errors, spelled out rather than erased into a string.
//!
//! Every variant names a way a file can be wrong, because the failure mode
//! this format has to avoid is the *silent* one: an index that does not fit
//! its slot is a perfectly valid index for a different lattice point, and a
//! reader that shrugs would hand back plausible, wrong weights.

use crate::{CodeKind, KindSet};
use core::fmt;

#[derive(Debug)]
pub enum Error {
    /// The file does not begin with [`crate::MAGIC`].
    NotAnArtifact { got: [u8; 4] },
    /// Ran out of bytes mid-structure.
    Truncated { reading: &'static str },
    /// A matrix name was not valid UTF-8.
    BadName,
    /// A tensor name did not look like `model.layers.{b}.{proj}.weight`.
    UnexpectedName { name: String },
    /// Internal inconsistency in a matrix handed to the writer.
    Inconsistent { name: String, detail: String },
    /// A lattice point that the codebook cannot index.
    PointOutsideCodebook { name: String },
    /// The index needs more bits than the declared shell cap allows —
    /// truncating it would silently select a different lattice point.
    IndexTooWide {
        name: String,
        index: u64,
        bits: u32,
    },
    /// An index read back from the stream is not a codebook member.
    IndexOutOfRange { name: String, index: u64 },
    /// A raw tensor record carries an encoding tag this reader does not know.
    BadRawEncoding { tag: u32 },
    /// The file was written against a different codebook than this build's.
    ///
    /// The one failure that has no visible symptom: every index would still be
    /// in range, every point it decodes to would still be a lattice point, and
    /// the weights would be wrong. `which` names the map — the v1 ball's
    /// ([`crate::codebook_fingerprint`]) or Tetra's
    /// ([`crate::codebook::tetra_fingerprint`]) — since a v5 header carries
    /// both and a reader that reported "a fingerprint" would leave the
    /// operator guessing which build to go and find.
    CodebookMismatch {
        which: &'static str,
        stored: u64,
        computed: u64,
    },
    /// A header version this writer cannot emit.
    UnknownVersion { version: u32 },
    /// A v5 header or record names a code kind this reader does not know: a
    /// newer writer, or a corrupted field. Refused where it is read, before
    /// any width is trusted — an unknown kind's records have no defined
    /// width, and an unknown bit in the header's set is a matrix this build
    /// cannot decode sitting somewhere in the file.
    ///
    /// [`crate::RESERVED_INT4G128_TAG`] is no longer one of these: tag 2 is
    /// [`CodeKind::Int4G128`], written by [`crate::write_matrix_int4`] and read
    /// by [`crate::read_record`]. Tag 3 and above are.
    UnknownCodeKind { tag: u32 },
    /// A record was pushed as a kind the file's header never declared.
    ///
    /// The header precedes every record and cannot be rewritten, so the set
    /// of kinds a file may hold is fixed by [`crate::ArtifactWriter`]'s
    /// constructor. Writing the record anyway would produce a file whose own
    /// refusals — which read that set — lie about what is in it.
    KindNotDeclared {
        name: String,
        kind: CodeKind,
        declared: KindSet,
    },
    /// A record of one kind reached an entry point that reads another: a
    /// Tetra record handed to [`crate::read_matrix`], which decodes v1 ball
    /// indices and nothing else. Not an `IndexOutOfRange` by luck — a
    /// refusal by name, before the labels are put through the wrong map.
    WrongCodeKind {
        name: String,
        want: CodeKind,
        got: CodeKind,
    },
    /// A record whose kind stores weights rather than lattice indices reached
    /// an entry point that reads `(index, gain)` pairs.
    ///
    /// An [`CodeKind::Int4G128`] record carries no centroids, no row scales
    /// and no tail: its payload is a group-affine int4 block. Read as a
    /// lattice record it would take the payload length out of the nibbles.
    /// The refusal is at [`crate::read_matrix_raw`], which is what every
    /// lattice-only tool of this repository already goes through, so one
    /// check covers all of them.
    NotALatticeRecord { name: String, kind: CodeKind },
    /// A kind that indexes no lattice was asked for its map.
    ///
    /// [`CodeKind::Int4G128`] stores its weights; there is nothing to index
    /// and nothing to look up. Returning the ball's map by default would hand
    /// a caller a decoder for bytes that are not indices.
    NoCodebookForKind { kind: CodeKind },
    /// Underlying I/O failure.
    Io(std::io::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NotAnArtifact { got } => {
                write!(f, "not an LLVQ artifact: magic is {got:?}")
            }
            Error::Truncated { reading } => write!(f, "file ends mid-{reading}"),
            Error::BadName => write!(f, "matrix name is not valid UTF-8"),
            Error::UnexpectedName { name } => {
                write!(f, "unexpected tensor name {name}")
            }
            Error::Inconsistent { name, detail } => write!(f, "{name}: {detail}"),
            Error::PointOutsideCodebook { name } => {
                write!(f, "{name}: a block's point is outside the codebook")
            }
            Error::IndexTooWide { name, index, bits } => write!(
                f,
                "{name}: index {index} does not fit in {bits} bits — the shell \
                 cap and the codes disagree, and a truncated index is a valid \
                 index for a different point"
            ),
            Error::IndexOutOfRange { name, index } => {
                write!(f, "{name}: index {index} is not a codebook member")
            }
            Error::BadRawEncoding { tag } => write!(
                f,
                "raw tensor encoding tag {tag} is unknown — file written by a \
                 newer writer, or corrupted"
            ),
            Error::CodebookMismatch {
                which,
                stored,
                computed,
            } => write!(
                f,
                "{which} codebook fingerprint {stored:#018x} does not match \
                 this build's {computed:#018x} — the file's indices were \
                 assigned by a different index map (Golay order, class order \
                 or mixed-radix composition for the ball; tetra, rows or \
                 columns for Tetra), so decoding them here would yield valid \
                 lattice points that are not the ones written"
            ),
            Error::UnknownVersion { version } => {
                write!(f, "no artifact format version {version} to write")
            }
            Error::UnknownCodeKind { tag } => write!(
                f,
                "code kind {tag} is unknown (0 = Ball, 1 = Tetra, 2 = Int4G128) \
                 — file written by a newer writer, or corrupted"
            ),
            Error::KindNotDeclared {
                name,
                kind,
                declared,
            } => write!(
                f,
                "{name}: a {kind} record under a header that declared {declared} — \
                 the header is written before the records and cannot be revised, \
                 so every kind a file may hold is declared when the writer is built"
            ),
            Error::WrongCodeKind { name, want, got } => write!(
                f,
                "{name}: a {got} record read through the {want} entry point — its \
                 indices are in range for {want} and mean nothing there"
            ),
            Error::NotALatticeRecord { name, kind } => write!(
                f,
                "{name}: a {kind} record read as a lattice matrix — its bytes are \
                 group-affine int4, not (index, gain) pairs; read it through \
                 read_record"
            ),
            Error::NoCodebookForKind { kind } => write!(
                f,
                "{kind} has no lattice map: its weights are stored, not indexed"
            ),
            Error::Io(e) => write!(f, "i/o: {e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}
