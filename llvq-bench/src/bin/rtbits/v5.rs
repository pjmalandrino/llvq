//! The rate accounting of a v5 file, Tetra words and int4 records, which
//! names no class and so shares none of the v1 accumulators: one report,
//! one file.

use super::*;

/// The rate accounting of a v5 file — Tetra words, int4 records, or both.
///
/// **A different report, not a relaxed one.** Everything above this function
/// keys on a v1 class: masks, level histograms, slot strides, the Planes14 and
/// E1c projections. A Tetra word names no class and an int4 record names no
/// lattice point at all, so none of that has a value here and none of it is
/// printed. What is left is the arithmetic — the one thing the dossier needs
/// an instrument for, since three independent hand reconstructions of this
/// file's b/param disagree in the fourth decimal.
///
/// Two accountings are printed side by side and the difference between them is
/// not cosmetic: `side_bits` charges the `KeepExact` tail at **f32**, and the
/// served path has stored it at **f16** since 2026-08-09
/// (`llvq_llm::fused::TAIL_BYTES = 2`). Every published kernel b/weight in the
/// record carries the f32 figure. The whole-model b/param published for the
/// three sealed files (paper 2, `docs/ETAT.md`) is the f16 row, q4 embedding
/// column. Printing both is how a reader sees which one a number came from
/// instead of guessing.
///
/// The lattice stream is counted twice as well. `lattice_stream_bits` is the
/// stream as the file stores it, 48 bits a block back to back.
/// `lattice_card_stream_bits` is the stream as the card holds it: a Tetra row
/// is padded to [`llvq_artifact::tetra48::stride_u32`], 0 to 4 bytes on the
/// shapes of the sealed files, and the served loader counts that pad. Every
/// line printed before the card count existed keeps the file count.
pub(super) struct V5Rate {
    /// Lattice records: their shapes, and the stream bits they spend.
    pub(super) lattice: Shapes,
    pub(super) lattice_stream_bits: u64,
    /// The same records with every Tetra row at the card's stride. A Ball
    /// record has no `tetra48` layout and is carried at its file count.
    pub(super) lattice_card_stream_bits: u64,
    pub(super) lattice_matrices: u64,
    /// int4 records: weights, and the bits the three arrays actually occupy.
    pub(super) int4_weights: u64,
    pub(super) int4_bits: u64,
    pub(super) int4_matrices: u64,
}

impl V5Rate {
    /// Bits per weight the kernel reads, over **all** projection weights.
    ///
    /// `tail_f32` selects the tail width: `true` reproduces the accounting of
    /// every published kernel b/weight, `false` the width the card actually
    /// holds, the one the published whole-model b/param counts.
    pub(super) fn kernel_bpw(&self, tail_f32: bool) -> f64 {
        let tail = if tail_f32 { F32_BITS } else { 16 };
        let side = self.lattice.tail_weights * tail + self.lattice.rows * F32_BITS;
        let bits = self.lattice_stream_bits + side + self.int4_bits;
        bits as f64 / (self.lattice.weights + self.int4_weights) as f64
    }

    /// [`Self::kernel_bpw`] at the f16 tail, with the lattice stream as the
    /// card holds it ([`Self::lattice_card_stream_bits`]) instead of as the
    /// file stores it.
    pub(super) fn card_kernel_bpw(&self) -> f64 {
        let side = self.lattice.tail_weights * 16 + self.lattice.rows * F32_BITS;
        let bits = self.lattice_card_stream_bits + side + self.int4_bits;
        bits as f64 / (self.lattice.weights + self.int4_weights) as f64
    }
}

/// Bits one Tetra matrix's words occupy on the card: `d_out` rows of
/// [`llvq_artifact::tetra48::stride_u32`] u32 each, the pad that rounds a row
/// to 8 bytes included. The file stores the same words without the pad,
/// `d_out · (d_in / 24) · 48` bits.
pub(super) fn tetra_card_stream_bits(d_out: usize, d_in: usize) -> u64 {
    (d_out * llvq_artifact::tetra48::stride_u32(d_in / DIM)) as u64 * 32
}

/// Walk a v5 file and print its rate accounting, then stop.
///
/// The header has already been read; this consumes the records. It prints no
/// class statistic, no layout projection and no traffic model, because none of
/// them has a meaning for a map that names no class — and printing a plausible
/// one would be worse than printing nothing.
pub(super) fn report_v5(path: &str, r: &mut impl Read, h: &llvq_artifact::Header, sidecar: Option<&str>) {
    let mut whole = Whole::default();
    let mut a = V5Rate {
        lattice: Shapes::default(),
        lattice_stream_bits: 0,
        lattice_card_stream_bits: 0,
        lattice_matrices: 0,
        int4_weights: 0,
        int4_bits: 0,
        int4_matrices: 0,
    };
    let mut gain_widths: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();

    for _ in 0..h.matrices {
        match llvq_artifact::read_record(r, h.version).expect("valid record") {
            llvq_artifact::Record::Lattice(m) => {
                a.lattice.push(m.d_out, m.d_in);
                // The word is the unit of the format: `LABEL_BITS` of index
                // and one gain bit, whatever the shell cap field says.
                let per_block = llvq_search::tetra::WORD_BITS as u64;
                let file_bits = (m.d_out * (m.d_in / DIM)) as u64 * per_block;
                a.lattice_stream_bits += file_bits;
                a.lattice_card_stream_bits += if m.kind == llvq_artifact::CodeKind::Tetra {
                    tetra_card_stream_bits(m.d_out, m.d_in)
                } else {
                    file_bits
                };
                a.lattice_matrices += 1;
                gain_widths.insert(m.centroids.len());
                whole.push_lattice(&m);
            }
            llvq_artifact::Record::Int4(q) => {
                whole.int4.add((q.d_out * q.d_in) as u64, q.bits());
                a.int4_weights += (q.d_out * q.d_in) as u64;
                // The three arrays as they are stored, not a nominal 4.25:
                // a shape whose `d_in` is not a multiple of the group would
                // spend more, and the file is what is being priced.
                a.int4_bits +=
                    (q.packed.len() as u64 + 2 * q.scales.len() as u64 + 2 * q.biases.len() as u64)
                        * 8;
                a.int4_matrices += 1;
            }
        }
    }

    // The carried tensors, read the way the ball path reads them: a u32 count
    // then one framed tensor each, and the same two names count as embedding.
    // Two readers of one section would be two chances to disagree about what
    // "b/param whole model" divides by.
    let (mut carried_params, mut carried_bits, mut carried_embed) = (0u64, 0u64, 0u64);
    if h.is_self_contained() {
        let mut b = [0u8; 4];
        r.read_exact(&mut b).expect("raw-tensor count");
        for _ in 0..u32::from_le_bytes(b) {
            let t = llvq_artifact::read_raw(r, h.version).expect("valid raw tensor");
            carried_params += t.len() as u64;
            carried_bits += t.bytes() * 8;
            whole.raw.add(t.len() as u64, t.bytes() * 8);
            if matches!(t.name.as_str(), "model.embed_tokens.weight" | "lm_head.weight") {
                carried_embed += t.len() as u64;
            }
        }
        // The blobs, for the seal's own description: a partial seal says
        // there how many native parameters it did not carry.
        r.read_exact(&mut b).expect("blob count");
        for _ in 0..u32::from_le_bytes(b) {
            let blob = llvq_artifact::read_blob(r).expect("valid blob");
            if blob.name == "inkling-seal.json" {
                let v = json::parse(&String::from_utf8_lossy(&blob.bytes)).expect("inkling-seal.json");
                // A text-only seal omits the vision tower by design; only an
                // omitted text tensor makes the seal partial.
                whole.omitted = v
                    .get("omitted_text_elements")
                    .or_else(|| v.get("omitted_elements"))
                    .and_then(json::Value::as_u64)
                    .unwrap_or(0);
                whole.seal_blob = true;
            }
        }
    }
    if let Some(sc) = sidecar {
        whole.read_sidecar(sc);
    }

    let lat_w = a.lattice.weights;
    let total_q = lat_w + a.int4_weights;
    println!("\n  ── {path} — format v{}, kinds {} ──", h.version, h.kinds());
    println!(
        "  {} lattice matrices, {lat_w} weights at {:.4} b/weight of stream",
        a.lattice_matrices,
        a.lattice_stream_bits as f64 / lat_w.max(1) as f64
    );
    println!(
        "  the stream above is the file's; on the card, Tetra rows at the tetra48 stride: \
         {} bits, {:.4} b/weight, {} pad bits the file does not store",
        a.lattice_card_stream_bits,
        a.lattice_card_stream_bits as f64 / lat_w.max(1) as f64,
        a.lattice_card_stream_bits - a.lattice_stream_bits
    );
    println!(
        "  {} int4 matrices,    {} weights at {:.4} b/weight, side data included",
        a.int4_matrices,
        a.int4_weights,
        a.int4_bits as f64 / a.int4_weights.max(1) as f64
    );
    println!(
        "  tail {} weights, {} row scales — the side data every arm uploads",
        a.lattice.tail_weights, a.lattice.rows
    );
    assert_eq!(
        gain_widths.iter().copied().collect::<Vec<_>>(),
        vec![2usize],
        "{path}: gain widths {gain_widths:?}; every decoder hard-codes one gain bit"
    );

    println!("\n  kernel b/weight, over {total_q} projection weights");
    println!(
        "    tail at f32 (the accounting of published kernel b/weight) ... {:.4}",
        a.kernel_bpw(true)
    );
    println!(
        "    tail at f16 (the width the card has held since 2026-08-09) .. {:.4}",
        a.kernel_bpw(false)
    );
    println!(
        "    tail at f16, Tetra rows at the card's tetra48 stride ........ {:.4}",
        a.card_kernel_bpw()
    );

    if carried_params > 0 {
        let other = carried_params - carried_embed;
        let embed_q8 = 8.0 + EMBED_GROUP_BITS / EMBED_GROUP;
        // The `LLVQ_EMBED=q4` path: int4 plus the same f16 pair per group of
        // 64, 4 + 32/64 = 4.5 bits, the encoding `embedq q4` writes.
        let embed_q4 = 4.0 + EMBED_GROUP_BITS / EMBED_GROUP;
        let lin_f32 = (total_q, a.kernel_bpw(true));
        let lin_f16 = (total_q, a.kernel_bpw(false));
        let lin_card = (total_q, a.card_kernel_bpw());
        println!(
            "\n  b/param WHOLE MODEL over {} parameters ({carried_embed} embedding, {other} norms)",
            total_q + carried_params
        );
        println!(
            "  {:<22}{:>14}{:>14}{:>14}",
            "tail width", "embed f16", "embed q8", "embed q4"
        );
        // The published whole-model b/param is the f16 row, embed q4 column.
        // The third row is the f16 row with the Tetra stream at the card's
        // row stride, the bits the card holds.
        for (name, lin) in [
            ("f32", lin_f32),
            ("f16 (served)", lin_f16),
            ("f16 + tetra48 stride", lin_card),
        ] {
            println!(
                "  {name:<22}{:>14.4}{:>14.4}{:>14.4}",
                model_bpw(&[lin, (carried_embed, 16.0), (other, 16.0)]),
                model_bpw(&[lin, (carried_embed, embed_q8), (other, 16.0)]),
                model_bpw(&[lin, (carried_embed, embed_q4), (other, 16.0)])
            );
        }
        println!(
            "  carried MEASURED in the file: {:.4} b/param",
            carried_bits as f64 / carried_params as f64
        );
    } else {
        println!("\n  projections-only file: no whole-model b/param without an embedding.");
    }

    whole.print(sidecar);

    println!(
        "\n  NO CLASS STATISTIC IS PRINTED. Every accumulator of this tool keys on a v1\n           class — masks, level histograms, slot strides, the Planes14 and E1c projections —\n           and a Tetra word names none. A plausible-looking histogram here would be worse\n           than its absence."
    );
}
