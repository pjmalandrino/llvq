//! `FileSink`, the buffered writer that streams each quantized matrix into the
//! artifact as the block loop produces it, and that the resume path borrows to
//! copy a shard's records forward.

/// Streams matrices to disk as they are quantized. Buffered, because the
/// index stream is written in 6-byte units and 151 M of them through an
/// unbuffered `File` would be 151 M syscalls.
pub(super) struct FileSink {
    w: llvq_llm::artifact2::ArtifactWriter<std::io::BufWriter<std::fs::File>>,
    path: String,
}

impl FileSink {
    /// `kind` is the run's codebook's ([`llvq_llm::calib::Codebook::code_kind`])
    /// and is the file's default; `int4` says whether the run will also write
    /// int4 g128 records, which the header has to **declare** before the first
    /// matrix — it is written by the constructor and cannot be revised.
    ///
    /// The version follows the declared set: v4 for a Ball-only file — byte
    /// for byte the header every published file carries — and v5 as soon as
    /// anything else is in it. Declaring a kind the run then never writes
    /// would make every GPU binary refuse the file, so the flag is the run's
    /// own decision and not a precaution.
    pub(super) fn create(
        path: &str,
        n: u32,
        kind: llvq_llm::artifact2::CodeKind,
        int4: bool,
    ) -> anyhow::Result<Self> {
        let f = std::fs::File::create(path)?;
        let mut kinds = llvq_llm::artifact2::KindSet::of(kind);
        if int4 {
            kinds.insert(llvq_llm::artifact2::CodeKind::Int4G128);
        }
        let version = if kinds.is_ball_only() {
            llvq_artifact::DEFAULT_VERSION
        } else {
            llvq_artifact::FIRST_KINDED_VERSION
        };
        Ok(Self {
            w: llvq_llm::artifact2::ArtifactWriter::with_kinds(
                std::io::BufWriter::with_capacity(1 << 20, f),
                version,
                n,
                kind,
                kinds,
            )?,
            path: path.to_string(),
        })
    }
    /// The raw writer, for the resume path — which copies its input shard's
    /// records straight through rather than pushing decoded matrices.
    pub(super) fn writer(
        &mut self,
    ) -> &mut llvq_llm::artifact2::ArtifactWriter<std::io::BufWriter<std::fs::File>> {
        &mut self.w
    }
    pub(super) fn finish(self) -> anyhow::Result<(u64, String)> {
        let bits = self.w.finish()?;
        Ok((bits, self.path))
    }
}

impl llvq_llm::calib::MatrixSink for FileSink {
    fn push(&mut self, m: llvq_llm::artifact2::QuantizedMatrix) -> anyhow::Result<()> {
        // `llvq-artifact` has its own error type on purpose — it carries no
        // dependencies, `anyhow` included. The bridge converts.
        Ok(self.w.push(&m)?)
    }
    fn push_int4(&mut self, m: llvq_llm::artifact2::Int4Matrix) -> anyhow::Result<()> {
        Ok(self.w.push_int4(&m)?)
    }
}
