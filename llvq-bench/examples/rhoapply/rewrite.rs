//! The artifact side: decoding a record with scaled rows, the row-scale proof,
//! the rewrite that writes an arm, and the byte comparison against the source.

use super::*;

// ---------------------------------------------------------------------------
// Decoding a record
// ---------------------------------------------------------------------------

/// `raw` rebuilt in the natural basis, with every row scale multiplied by
/// `factor(i)`. The path is the shipped one — [`llvq_artifact::decode_matrix`]
/// — so what this bench measures is what the model would hold.
pub(super) fn decode_scaled(raw: &RawMatrix, cb: &Codebook, factor: &dyn Fn(usize) -> f64) -> Vec<f32> {
    let mut codes = Vec::with_capacity(raw.indices.len());
    for (&idx, &gain) in raw.indices.iter().zip(&raw.gains) {
        let point = cb
            .decode(idx, gain)
            .unwrap_or_else(|| panic!("{}: index {idx} outside the map", raw.name));
        codes.push(BlockCode { point, gain });
    }
    let q = QuantizedMatrix {
        name: raw.name.clone(),
        d_out: raw.d_out,
        d_in: raw.d_in,
        codes,
        row_scales: raw
            .row_scales
            .iter()
            .enumerate()
            .map(|(i, s)| s * factor(i))
            .collect(),
        centroids: raw.centroids.clone(),
        rotation_seed: raw.rotation_seed,
        shell_cap: raw.shell_cap,
        tail: raw.tail.clone(),
    };
    llvq_artifact::decode_matrix(&q)
}

// ---------------------------------------------------------------------------
// Step 1 — the foundation, proved on the real file
// ---------------------------------------------------------------------------

/// Prove, on a record taken from the artifact itself, that `row_scales[i] × ρ`
/// scales row `i` of the decoded matrix and nothing else.
///
/// Three one-matrix files are written and decoded: ρ = 1, ρ = 2 and ρ = 0 on a
/// single row. The claim has two halves, and each has its own check:
///
/// * **rows are independent** — every row `j ≠ i` must come back *bit
///   identical* across the three files. If the rotation mixed rows this fails,
///   and the whole bench would be void.
/// * **the row is affine in ρ** — `r(2) = 2·r(1) − r(0)` to rounding. The
///   intercept `r(0)` is the tail's image, and it is not zero, which is why
///   the map is affine rather than linear.
pub(super) fn prove_row_scale(raw: &RawMatrix, cb: &Codebook, out_dir: &Path) -> Result<(), String> {
    let row = raw.d_out / 3; // an interior row, not an edge case of either end
    println!("STEP 1 — row_scales[i] × ρ scales row i, and only row i");
    println!("  record  {} ({}x{})", raw.name, raw.d_out, raw.d_in);
    println!("  probe row {row}, tail width {}", raw.d_in % DIM);

    let mut decoded = Vec::new();
    for (tag, rho) in [("one", 1.0f64), ("two", 2.0), ("zero", 0.0)] {
        let path = out_dir.join(format!("rhoapply-proof-{tag}.llvq"));
        let f = File::create(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut w = BufWriter::with_capacity(1 << 20, f);
        llvq_artifact::write_header_kinds(
            &mut w,
            llvq_artifact::VERSION,
            1,
            raw.kind,
            llvq_artifact::KindSet::of(raw.kind),
        )
        .map_err(|e| e.to_string())?;
        let mut copy = clone_raw(raw);
        copy.row_scales[row] *= rho;
        llvq_artifact::write_matrix_raw(&mut w, llvq_artifact::VERSION, &copy)
            .map_err(|e| e.to_string())?;
        w.flush().map_err(|e| e.to_string())?;
        drop(w);

        // Read it back through the file, not from memory: the claim is about
        // what a reader of the artifact sees.
        let mut r = BufReader::with_capacity(1 << 20, File::open(&path).map_err(|e| e.to_string())?);
        let h = llvq_artifact::read_header(&mut r).map_err(|e| e.to_string())?;
        // A one-record file this function wrote two lines ago from a lattice
        // record, so `read_matrix_raw` is right here and `next_raw` would only
        // hide a bug.
        let back = llvq_artifact::read_matrix_raw(&mut r, h.version).map_err(|e| e.to_string())?;
        decoded.push(decode_scaled(&back, cb, &|_| 1.0));
        std::fs::remove_file(&path).map_err(|e| e.to_string())?;
    }
    let (r1, r2, r0) = (&decoded[0], &decoded[1], &decoded[2]);

    let n = raw.d_in;
    let mut differing_rows = Vec::new();
    for j in 0..raw.d_out {
        let a = &r1[j * n..(j + 1) * n];
        let b = &r2[j * n..(j + 1) * n];
        let c = &r0[j * n..(j + 1) * n];
        if a != b || a != c {
            differing_rows.push(j);
        }
    }
    println!(
        "  rows that moved when row {row} was rescaled: {:?} (bit-exact comparison over {} rows)",
        differing_rows, raw.d_out
    );
    if differing_rows != vec![row] {
        return Err(
            "the rotation does not keep rows independent, or the writer touched another row: \
             the whole bench is void"
                .into(),
        );
    }

    let a = &r1[row * n..(row + 1) * n];
    let b = &r2[row * n..(row + 1) * n];
    let c = &r0[row * n..(row + 1) * n];
    let scale = b.iter().fold(0.0f64, |m, v| m.max(f64::from(*v).abs()));
    let dev = a
        .iter()
        .zip(b)
        .zip(c)
        .fold(0.0f64, |m, ((x, y), z)| {
            m.max((2.0 * f64::from(*x) - f64::from(*z) - f64::from(*y)).abs())
        });
    let nb: f64 = a
        .iter()
        .zip(c)
        .map(|(x, z)| (f64::from(*x) - f64::from(*z)).powi(2))
        .sum();
    let nt: f64 = c.iter().map(|z| f64::from(*z).powi(2)).sum();
    println!("  affinity  max|r(2) − (2·r(1) − r(0))| / max|r(2)| = {:.3e}", dev / scale);
    println!(
        "  intercept ‖T‖²/(‖B‖²+‖T‖²) = {:.6e}  — the tail, which no row scale reaches",
        nt / (nb + nt)
    );
    if dev / scale > 1e-6 {
        return Err("the decoded row is not affine in the row scale".into());
    }
    println!("  PROVED: r_i(ρ) = ρ·B_i + T_i, rows independent.\n");
    Ok(())
}

fn clone_raw(m: &RawMatrix) -> RawMatrix {
    RawMatrix {
        name: m.name.clone(),
        d_out: m.d_out,
        d_in: m.d_in,
        kind: m.kind,
        indices: m.indices.clone(),
        gains: m.gains.clone(),
        row_scales: m.row_scales.clone(),
        centroids: m.centroids.clone(),
        rotation_seed: m.rotation_seed,
        shell_cap: m.shell_cap,
        tail: m.tail.clone(),
    }
}

// ---------------------------------------------------------------------------
// The rewrite
// ---------------------------------------------------------------------------

/// What one rewrite touched, so the tool can state it rather than imply it.
#[derive(Default, Clone, Copy)]
pub(super) struct RewriteCounts {
    /// Records whose `row_scales` were multiplied.
    pub(super) lattice: usize,
    /// Records copied through by kind, with no `row_scales` to touch.
    pub(super) int4: usize,
    /// Row scales multiplied, over every lattice record.
    pub(super) scales: u64,
}

/// Copy `src` to `dst` changing nothing but the contents of `row_scales`.
///
/// Every other field — the indices, the gains, the centroids, the rotation
/// seed, the tail, the shell cap, the header and its two fingerprints — goes
/// through [`llvq_artifact::read_record`] and [`llvq_artifact::write_record`]
/// untouched, whose lattice arm is the pair whose byte-identity is pinned by
/// `raw_passthrough_is_byte_identical`. Nothing is decoded and nothing is
/// re-encoded, and an **int4 g128** record — the served `v_proj` — is copied
/// through by kind with no `row_scales` to touch.
pub(super) fn rewrite(
    src: &Path,
    dst: &Path,
    factor: &dyn Fn(usize, usize) -> f64,
) -> Result<(String, u64, RewriteCounts), String> {
    let mut counts = RewriteCounts::default();
    let mut r = BufReader::with_capacity(1 << 22, File::open(src).map_err(|e| e.to_string())?);
    let h = llvq_artifact::read_header(&mut r).map_err(|e| e.to_string())?;
    let mut w = HashWriter {
        inner: BufWriter::with_capacity(1 << 22, File::create(dst).map_err(|e| e.to_string())?),
        hash: Sha256::new(),
        bytes: 0,
    };
    llvq_artifact::write_header_kinds(&mut w, h.version, h.matrices, h.default_kind, h.kinds)
        .map_err(|e| e.to_string())?;
    for mi in 0..h.matrices as usize {
        // `read_record`/`write_record` and not the raw pair: the served file is
        // mixed, and an int4 record has no `row_scales` for ρ to reach. It is
        // copied through by kind, so `v_proj` is left alone by construction.
        let mut rec = llvq_artifact::read_record(&mut r, h.version).map_err(|e| e.to_string())?;
        match &mut rec {
            llvq_artifact::Record::Lattice(m) => {
                counts.lattice += 1;
                counts.scales += m.row_scales.len() as u64;
                for (i, s) in m.row_scales.iter_mut().enumerate() {
                    *s *= factor(mi, i);
                }
            }
            // Nothing is touched, and nothing *can* be: the factor is never
            // asked for. Counted so the caller can say so out loud rather than
            // leaving the reader to trust a branch.
            llvq_artifact::Record::Int4(_) => counts.int4 += 1,
        }
        llvq_artifact::write_record(&mut w, h.version, &rec).map_err(|e| e.to_string())?;
    }
    // Everything after the last record, copied verbatim. On a v3+ file that
    // is at least the two section counts `ArtifactWriter::finish` writes
    // (eight zero bytes, and the first thing the idempotence control caught
    // missing); on a sealed file it is the raw tensors and the blobs. Copying
    // the bytes rather than re-serializing the sections is what keeps this
    // tool honest about the claim it makes: it edits `row_scales` and nothing
    // else, including the parts of the format it does not model.
    std::io::copy(&mut r, &mut w).map_err(|e| e.to_string())?;
    w.flush().map_err(|e| e.to_string())?;
    let bytes = w.bytes;
    Ok((w.hash.finish(), bytes, counts))
}

// ---------------------------------------------------------------------------
// The byte-by-byte control (M1b §4.2)
// ---------------------------------------------------------------------------

/// Byte ranges of `path` that hold `row_scales`, half-open, in file order.
///
/// The offset is derived from the record layout `put_record_head` writes —
/// name, `d_out`, `d_in`, shell cap, kind tag from v5, centroid count,
/// rotation seed and flag, then the centroids — and then **verified** against
/// the file: the eight bytes at each computed slot must be the bit pattern
/// [`llvq_artifact::read_matrix_raw`] returned for that scale. A layout model
/// off by one field would not survive that comparison, so the ranges this
/// returns are the row-scale zones and not a guess about them.
pub(super) fn row_scale_ranges(path: &Path) -> Result<Vec<(u64, u64)>, String> {
    let mut r = BufReader::with_capacity(1 << 22, File::open(path).map_err(|e| e.to_string())?);
    let h = llvq_artifact::read_header(&mut r).map_err(|e| e.to_string())?;
    let mut ranges = Vec::with_capacity(h.matrices as usize);
    let mut expect: Vec<Vec<u64>> = Vec::with_capacity(h.matrices as usize);
    for _ in 0..h.matrices {
        let before = r.stream_position().map_err(|e| e.to_string())?;
        // An int4 record spans bytes like any other and holds no `row_scales`;
        // it must be consumed so the next `before` is right, and skipped so the
        // model does not claim a zone inside it.
        let Some(m) = next_raw(&mut r, h.version)? else {
            continue;
        };
        let after = r.stream_position().map_err(|e| e.to_string())?;
        let kind_tag = u64::from(h.version >= llvq_artifact::FIRST_KINDED_VERSION);
        let prefix = 4 + m.name.len() as u64      // name length, then the name
            + 4 + 4 + 4                            // d_out, d_in, shell cap
            + 4 * kind_tag                         // the record's code kind
            + 4                                    // centroid count
            + 8 + 4                                // rotation seed, rotation flag
            + 8 * m.centroids.len() as u64;
        let start = before + prefix;
        let end = start + 8 * m.row_scales.len() as u64;
        if end > after {
            return Err(format!(
                "{}: the row-scale slot [{start}, {end}) runs past the record that ends at {after}",
                m.name
            ));
        }
        ranges.push((start, end));
        expect.push(m.row_scales.iter().map(|s| s.to_bits()).collect());
    }
    drop(r);

    let mut f = File::open(path).map_err(|e| e.to_string())?;
    let mut buf = Vec::new();
    for (&(start, end), scales) in ranges.iter().zip(&expect) {
        f.seek(SeekFrom::Start(start)).map_err(|e| e.to_string())?;
        buf.resize((end - start) as usize, 0u8);
        f.read_exact(&mut buf).map_err(|e| e.to_string())?;
        for (k, &bits) in scales.iter().enumerate() {
            let mut w = [0u8; 8];
            w.copy_from_slice(&buf[8 * k..8 * k + 8]);
            if u64::from_le_bytes(w) != bits {
                return Err(format!(
                    "the row-scale zone model is wrong: scale {k} at offset {} reads {:#018x}, \
                     the record reader says {bits:#018x}",
                    start + 8 * k as u64,
                    u64::from_le_bytes(w)
                ));
            }
        }
    }
    Ok(ranges)
}

/// Compare two files byte by byte and split the differing bytes into those
/// that fall inside `ranges` and those that fall outside.
///
/// Returns `(inside, outside, first_outside, total)`. The prereg's control is
/// `outside == 0`; `inside` is reported because a rewrite that changed nothing
/// at all would also pass `outside == 0`.
pub(super) fn diff_against(
    src: &Path,
    dst: &Path,
    ranges: &[(u64, u64)],
) -> Result<(u64, u64, Option<u64>, u64), String> {
    let mut a = BufReader::with_capacity(1 << 22, File::open(src).map_err(|e| e.to_string())?);
    let mut b = BufReader::with_capacity(1 << 22, File::open(dst).map_err(|e| e.to_string())?);
    let mut ba = vec![0u8; 1 << 22];
    let mut bb = vec![0u8; 1 << 22];
    let (mut inside, mut outside, mut first_outside, mut off) = (0u64, 0u64, None, 0u64);
    let mut ri = 0usize;
    loop {
        let n = read_up_to(&mut a, &mut ba)?;
        let m = read_up_to(&mut b, &mut bb)?;
        if n != m {
            return Err(format!("the two files disagree in length at offset {off}"));
        }
        if n == 0 {
            break;
        }
        // The common case is a chunk with no difference at all; the slice
        // comparison settles it without touching a single byte by hand.
        if ba[..n] != bb[..n] {
            for k in 0..n {
                if ba[k] == bb[k] {
                    continue;
                }
                let at = off + k as u64;
                while ri < ranges.len() && ranges[ri].1 <= at {
                    ri += 1;
                }
                if ri < ranges.len() && at >= ranges[ri].0 {
                    inside += 1;
                } else {
                    outside += 1;
                    first_outside.get_or_insert(at);
                }
            }
        }
        off += n as u64;
    }
    Ok((inside, outside, first_outside, off))
}

/// `read` until the buffer is full or the file ends. `Read::read` is allowed
/// to return short, and a short read here would misalign the two files.
fn read_up_to(r: &mut impl Read, buf: &mut [u8]) -> Result<usize, String> {
    let mut n = 0;
    while n < buf.len() {
        match r.read(&mut buf[n..]).map_err(|e| e.to_string())? {
            0 => break,
            k => n += k,
        }
    }
    Ok(n)
}
