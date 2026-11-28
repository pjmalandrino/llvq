//! `ArtifactWriter`, the streaming writer of a whole model: the header up
//! front, the declared kinds checked at every push, the matrix count checked
//! when the file is closed.

use super::checks::{put_u32, u32_field};
use super::header::write_header_kinds;
use super::int4::{write_matrix_int4, Int4Matrix, Record};
use super::kinds::{CodeKind, Codebooks, KindSet, DEFAULT_VERSION};
use super::write::{write_matrix_raw, write_matrix_with, QuantizedMatrix, RawMatrix};
use crate::{Error, Result};
use std::io::Write;
// Named only by the documentation's links, which resolved in the one file.
#[cfg(doc)]
use super::FIRST_KINDED_VERSION;

/// Streaming writer for a whole model.
///
/// Streaming is what shapes the kind fields. The header is written by the
/// constructor, before the first matrix is seen, and a 14 GB stream can be
/// neither buffered nor seeked — so the set of kinds the file may hold is
/// **declared** at construction and enforced at every push
/// ([`Error::KindNotDeclared`]), rather than accumulated and patched in
/// afterwards. [`Self::kinds_used`] is what the records actually were.
pub struct ArtifactWriter<W: Write> {
    out: W,
    version: u32,
    default_kind: CodeKind,
    declared: KindSet,
    used: KindSet,
    codebooks: Codebooks,
    /// The count the header promised, checked against [`Self::matrices`]
    /// when the file is closed.
    declared_matrices: u32,
    pub matrices: u32,
    pub payload_bits: u64,
}

impl<W: Write> ArtifactWriter<W> {
    /// `n_matrices` is written up front so the reader can size itself. A
    /// Ball file at [`DEFAULT_VERSION`], byte for byte what it always was.
    pub fn new(out: W, n_matrices: u32) -> Result<Self> {
        Self::with_version_kind(out, DEFAULT_VERSION, n_matrices, CodeKind::Ball)
    }

    /// Same, at a chosen format version — for a tool that rewrites an existing
    /// Ball file and must not silently upgrade it.
    pub fn with_version(out: W, version: u32, n_matrices: u32) -> Result<Self> {
        Self::with_version_kind(out, version, n_matrices, CodeKind::Ball)
    }

    /// A writer for `kind` and nothing else, at the version the kind calls for
    /// ([`CodeKind::default_version`]): v4 for Ball, v5 for Tetra.
    pub fn with_kind(out: W, kind: CodeKind, n_matrices: u32) -> Result<Self> {
        Self::with_version_kind(out, kind.default_version(), n_matrices, kind)
    }

    /// A writer for one kind at a chosen version. A non-Ball writer below
    /// [`FIRST_KINDED_VERSION`] is refused: the header would have nowhere to
    /// say what its records are, and neither would the records.
    pub fn with_version_kind(
        out: W,
        version: u32,
        n_matrices: u32,
        kind: CodeKind,
    ) -> Result<Self> {
        Self::with_kinds(out, version, n_matrices, kind, KindSet::of(kind))
    }

    /// The general form: a default kind for [`Self::push`], and the set of
    /// kinds the file declares — what a mixed file needs, and what its
    /// refusals will be read from. The default must be in the set.
    pub fn with_kinds(
        mut out: W,
        version: u32,
        n_matrices: u32,
        default_kind: CodeKind,
        declared: KindSet,
    ) -> Result<Self> {
        write_header_kinds(&mut out, version, n_matrices, default_kind, declared)?;
        Ok(Self {
            out,
            version,
            default_kind,
            declared,
            used: KindSet::empty(),
            codebooks: Codebooks::new(),
            declared_matrices: n_matrices,
            matrices: 0,
            payload_bits: 0,
        })
    }

    /// The kind [`Self::push`] writes.
    pub fn default_kind(&self) -> CodeKind {
        self.default_kind
    }

    /// The kinds the header declares — an upper bound on the records, since
    /// it was written before them.
    pub fn kinds(&self) -> KindSet {
        self.declared
    }

    /// The kinds actually pushed so far. Not stored in the file: it is only
    /// known once the records are, and by then the header is bytes on a disk.
    pub fn kinds_used(&self) -> KindSet {
        self.used
    }

    /// Push a matrix as the file's default kind.
    pub fn push(&mut self, m: &QuantizedMatrix) -> Result<()> {
        self.push_kind(m, self.default_kind)
    }

    /// Push a matrix as `kind`, whatever the file's default is — the entry a
    /// mixed file is written through. Refused unless the header declared the
    /// kind, because the header cannot be rewritten to say so afterwards.
    pub fn push_kind(&mut self, m: &QuantizedMatrix, kind: CodeKind) -> Result<()> {
        self.declare(kind, &m.name)?;
        let bits = write_matrix_with(&mut self.out, self.version, self.codebooks.get(kind)?, m)?;
        self.payload_bits += bits;
        self.matrices += 1;
        Ok(())
    }

    /// Push a matrix straight from its undecoded codes, as the kind the record
    /// already is — see [`write_matrix_raw`]. Nothing else about it is looked
    /// at, beyond the invariants of its own kind.
    pub fn push_raw(&mut self, m: &RawMatrix) -> Result<()> {
        self.declare(m.kind, &m.name)?;
        self.payload_bits += write_matrix_raw(&mut self.out, self.version, m)?;
        self.matrices += 1;
        Ok(())
    }

    /// Push an int4 record. Declared like any other kind, so a writer built
    /// with `KindSet::of(Tetra)` refuses it before a byte is written: the
    /// header is already on the disk and cannot be told about it afterwards.
    pub fn push_int4(&mut self, m: &Int4Matrix) -> Result<()> {
        self.declare(CodeKind::Int4G128, &m.name)?;
        self.payload_bits += write_matrix_int4(&mut self.out, self.version, m)?;
        self.matrices += 1;
        Ok(())
    }

    /// Push a record of whatever kind it already is — what a passthrough of a
    /// mixed file needs, and the only push that copies an int4 record without
    /// re-quantizing it.
    pub fn push_record(&mut self, rec: &Record) -> Result<()> {
        match rec {
            Record::Lattice(m) => self.push_raw(m),
            Record::Int4(m) => self.push_int4(m),
        }
    }

    /// The gate every push goes through: a kind the header did not declare is
    /// refused before a byte of the record is written.
    fn declare(&mut self, kind: CodeKind, name: &str) -> Result<()> {
        if !self.declared.contains(kind) {
            return Err(Error::KindNotDeclared {
                name: name.to_string(),
                kind,
                declared: self.declared,
            });
        }
        self.used.insert(kind);
        Ok(())
    }

    /// The header's count against the records pushed, before the closing
    /// sections go out. Another count leaves a file every reader misparses:
    /// too few and the raw-tensor section is read as a record, too many and
    /// the extra record is read as the raw-tensor section.
    fn check_count(&self) -> Result<()> {
        if self.matrices != self.declared_matrices {
            return Err(Error::Inconsistent {
                name: "matrix count".to_string(),
                detail: format!(
                    "the header declares {} matrices and {} pushed",
                    self.declared_matrices, self.matrices
                ),
            });
        }
        Ok(())
    }

    /// Close the file with the sections that make it self-contained.
    ///
    /// Taking them here rather than streaming them is deliberate: the counts
    /// have to precede the data, and a writer that cannot seek must therefore
    /// know them before writing. They are also small next to the codes —
    /// except the embedding, which is one tensor.
    pub fn seal(
        mut self,
        raws: &[crate::RawTensor],
        blobs: &[crate::Blob],
    ) -> Result<(u64, u64)> {
        self.check_count()?;
        let n_raws = u32_field("raw tensor section", "tensor count", raws.len())?;
        let n_blobs = u32_field("blob section", "blob count", blobs.len())?;
        put_u32(&mut self.out, n_raws)?;
        let mut extra = 0u64;
        for t in raws {
            extra += crate::write_raw(&mut self.out, t)?;
        }
        put_u32(&mut self.out, n_blobs)?;
        for b in blobs {
            extra += crate::write_blob(&mut self.out, b)?;
        }
        self.out.flush()?;
        Ok((self.payload_bits, extra * 8))
    }

    /// Close without the extra sections, producing a projections-only file.
    ///
    /// Kept for the quantization run, which streams matrices as it goes and
    /// has no checkpoint tensors to hand. `bin/seal` completes such a file.
    pub fn finish(mut self) -> Result<u64> {
        self.check_count()?;
        put_u32(&mut self.out, 0)?;
        put_u32(&mut self.out, 0)?;
        self.out.flush()?;
        Ok(self.payload_bits)
    }
}
