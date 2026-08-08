//! Errors, spelled out rather than erased into a string.
//!
//! Every variant names a way a file can be wrong, because the failure mode
//! this format has to avoid is the *silent* one: an index that does not fit
//! its slot is a perfectly valid index for a different lattice point, and a
//! reader that shrugs would hand back plausible, wrong weights.

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
    /// the weights would be wrong.
    CodebookMismatch { stored: u64, computed: u64 },
    /// A header version this writer cannot emit.
    UnknownVersion { version: u32 },
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
            Error::CodebookMismatch { stored, computed } => write!(
                f,
                "codebook fingerprint {stored:#018x} does not match this \
                 build's {computed:#018x} — the file's indices were assigned \
                 by a different index map (Golay order, class order or \
                 mixed-radix composition), so decoding them here would yield \
                 valid lattice points that are not the ones written"
            ),
            Error::UnknownVersion { version } => {
                write!(f, "no artifact format version {version} to write")
            }
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
