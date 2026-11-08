//! One matrix's arms, built from its record, and the matrices of the run.

use super::*;

/// The f64 references and the AWQ buffers of one matrix, as `reference_rows` leaves them.
struct Reference {
    w16: Vec<u16>,
    y_ref: Vec<f64>,
    y16_ref: Vec<f64>,
    scale: Vec<f64>,
    aww: usize,
    awq_tail: usize,
    awq_w: Option<Vec<u32>>,
    awq_z: Option<Vec<u32>>,
    awq_s: Option<Vec<u16>>,
    y_awq_ref: Option<Vec<f64>>,
    awq_scale: Option<Vec<f64>>,
    xh_f64: Option<Vec<f64>>,
    y16h_ref: Option<Vec<f64>>,
    chunk: usize,
}

/// One matrix, every selected arm built and staged, with its f64 references: `run`'s `build`.
// The body was moved verbatim out of `run`, where these names were owned locals; they are
// reference parameters here, so `&name` borrows a reference. Kept as moved.
#[allow(clippy::too_many_arguments, clippy::needless_borrow)]
pub(super) fn build_mat(
    s: &Src,
    x: &[f32],
    fd: &FastDecoder,
    table: &ClassTable,
    union: ArmSet,
    searcher: &Searcher,
    g70_needed: bool,
    golay: &llvq_core::Golay,
    g70cls: &[Golay70Class],
    phase0: ArmSet,
    cuda: &Cuda,
    tetra: &std::collections::HashMap<String, TetraSrc>,
) -> Result<Mat, String> {
    let (d_out, d_in) = (s.d_out, s.d_in);
    let nblocks = d_in / DIM;
    let tail_w = d_in % DIM;
    assert_eq!(d_out % 8, 0, "{}: CUDA launches whole blocks", s.name);
    assert!(
        d_in <= x.len(),
        "{}: d_in {d_in} overruns the activation of {} floats. Raise XLEN above \
         and relaunch: extending the draw leaves every shorter matrix bit-identical",
        s.name,
        x.len()
    );
    // The Slot32 host transcode is NOT conditional: it is the exact
    // content every LLVQ arm is proved against, and the reference
    // loop below decodes it row by row. What the selection spares is
    // its DEVICE buffers, not this.
    let rt = transcode(&fd, &table, &s.indices, &s.gains, Layout::Slot32)
        .map_err(|e| e.to_string())?;
    // The Planes14 stream: a bit-level bijection of the Slot32
    // content, proved block by block in the reference loop below.
    let planes_data = union
        .has(arms::PLANES14)
        .then(|| planes14_from_slot32(&rt, &table));
    // The M2 overlay: 12-byte main stream (L = 5 blocks swapped for
    // their best L ≤ 4 direction) + exact exception records. The
    // swap searches inside, threaded — the expensive build step, and
    // the selection's biggest saving when the arm is out.
    let p12 = match union.has(arms::PLANES12X) {
        true => Some(
            transcode_planes12x(&fd, &table, &searcher, &s.indices, &s.gains)
                .map_err(|e| e.to_string())?,
        ),
        false => None,
    };
    // The E2 arm: 9-byte main stream (exception blocks holed to the
    // origin) + exact exception records — pure table lookups plus a
    // Golay rank per block, no search. ONE stream for v1 and v2.
    let g70 = g70_needed
        .then(|| golay70_transcode(&fd, &golay, &g70cls, &table, &s.indices, &s.gains));

    let Reference {
        w16, y_ref, y16_ref, scale, aww, awq_tail, awq_w, awq_z, awq_s, y_awq_ref, awq_scale, xh_f64,
        y16h_ref, chunk,
    } = reference_rows(
        s, x, table, union, d_out, d_in, nblocks, tail_w, &rt, &planes_data, &p12, &g70, g70cls, golay,
    );

    // ---- the stage: Dev for phase 1, Host for the ones after ----
    let up_or_hold_u32 = |v: Vec<u32>, in_p0: bool| -> Result<Staged<u32>, String> {
        Ok(if in_p0 { Staged::Dev(cuda.up_u32(&v)?) } else { Staged::Host(v) })
    };
    let up_or_hold_u16 = |v: Vec<u16>, in_p0: bool| -> Result<Staged<u16>, String> {
        Ok(if in_p0 { Staged::Dev(cuda.up_u16(&v)?) } else { Staged::Host(v) })
    };

    let slot = match union.has(arms::SLOT32) {
        true => {
            let mut bytes = rt.data.clone();
            bytes.extend_from_slice(&[0u8; 20]); // the five-word read of the last block
            while !bytes.len().is_multiple_of(4) {
                bytes.push(0);
            }
            let words: Vec<u32> = bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let in0 = phase0.has(arms::SLOT32);
            Some(SlotArm {
                words: up_or_hold_u32(words, in0)?,
                bases: up_or_hold_u32(rt.bases.clone(), in0)?,
                bytes: rt.data.len() as u64
                    + rt.bases.len() as u64 * 4
                    + (d_out * tail_w) as u64 * 4
                    + d_out as u64 * 4,
            })
        }
        false => None,
    };

    let planes = match planes_data {
        Some(pdata) => {
            let bytes_acc = pdata.len() as u64
                + (d_out * tail_w) as u64 * 4
                + d_out as u64 * 4;
            let mut pbytes = pdata;
            // The last block's four-word window reaches at most 2
            // bytes past the stream; pad 4 and align, mirroring the
            // slot padding.
            pbytes.extend_from_slice(&[0u8; 4]);
            while !pbytes.len().is_multiple_of(4) {
                pbytes.push(0);
            }
            let pwords: Vec<u32> = pbytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Some(PlanesArm {
                pwords: up_or_hold_u32(pwords, phase0.has(arms::PLANES14))?,
                bytes: bytes_acc,
            })
        }
        None => None,
    };

    let pad12 = |mut b: Vec<u8>| -> Vec<u32> {
        b.extend_from_slice(&[0u8; 4]);
        while !b.len().is_multiple_of(4) {
            b.push(0);
        }
        b.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };

    // The M2 arm's three device arrays and its byte accounting:
    // main stream + exception indices + exception records + the
    // same tail/rscale terms as the other LLVQ arms. Upload paddings
    // are NOT billed: the slot arm ignores its 20-byte pad and the
    // planes arm its 4-byte pad, so billing ours here would mix two
    // byte accountings — the exact mistake the K-1 lot eliminated.
    let p12 = match p12 {
        Some(p12) => {
            let n_exc = p12.exc_idx.len();
            let bytes_acc = p12.data.len() as u64
                + n_exc as u64 * 4
                + p12.exc_data.len() as u64
                + (d_out * tail_w) as u64 * 4
                + d_out as u64 * 4;
            let p12words: Vec<u32> = pad12(p12.data);
            let excwords: Vec<u32> = pad12(p12.exc_data);
            // cudarc refuses a zero-length upload; a matrix without
            // any L = 5 block gets a one-word dummy the kernel never
            // reads (n_exc == 0 spawns no correction CTA).
            let exc_idx_up: Vec<u32> = if p12.exc_idx.is_empty() {
                vec![0]
            } else {
                p12.exc_idx
            };
            let in0 = phase0.has(arms::PLANES12X);
            Some(P12Arm {
                words: up_or_hold_u32(p12words, in0)?,
                exc_idx: up_or_hold_u32(exc_idx_up, in0)?,
                exc_words: up_or_hold_u32(excwords, in0)?,
                n_exc,
                bytes: bytes_acc,
            })
        }
        None => None,
    };

    // The E2 arm's three device arrays and its byte accounting:
    // 72 bits per block + 144 per exception + the same tail/rscale
    // terms as every LLVQ arm; upload paddings NOT billed — the
    // unified rule. One set of buffers for BOTH golay70 arms.
    let g70 = match g70 {
        Some(g70) => {
            let n_gexc = g70.exc_idx.len();
            let bytes_acc = g70.data.len() as u64
                + n_gexc as u64 * GOLAY70_EXC_BYTES as u64
                + (d_out * tail_w) as u64 * 4
                + d_out as u64 * 4;
            let gwords: Vec<u32> = pad12(g70.data);
            let gexcwords: Vec<u32> = pad12(g70.exc_data);
            let gexc_idx_up: Vec<u32> = if g70.exc_idx.is_empty() {
                vec![0]
            } else {
                g70.exc_idx
            };
            let in0 =
                phase0.has(arms::GOLAY70V1) || phase0.has(arms::GOLAY70V2);
            Some(G70Arm {
                words: up_or_hold_u32(gwords, in0)?,
                exc_idx: up_or_hold_u32(gexc_idx_up, in0)?,
                exc_words: up_or_hold_u32(gexcwords, in0)?,
                n_exc: n_gexc,
                bytes: bytes_acc,
            })
        }
        None => None,
    };

    // P1c. The servable stream: `transcode_e1v_rows` through
    // `e1v_host.rs`, cut on `d_in / 24` blocks per row so that a group
    // never straddles a row, failing which no warp would read a single
    // group (X3). No search, no exception, a direct transcode.
    let e1v_arm = match union.has(arms::E1V) {
        true => {
            let em = e1v_mat(&fd, &golay, &s.indices, &s.gains, d_in)?;
            let in0 = phase0.has(arms::E1V);
            Some(E1vArm {
                row_blocks: em.row_blocks,
                // The arm's bill: the stream and its bases table, plus
                // the f32 tail and the row scales that EVERY LLVQ arm
                // uploads, the `slot32` accounting twenty lines above,
                // identical.
                bytes: em.bytes + (d_out * tail_w) as u64 * 4 + d_out as u64 * 4,
                data: up_or_hold_u32(em.data, in0)?,
                bases: up_or_hold_u32(em.bases, in0)?,
            })
        }
        false => None,
    };

    let awq = match (awq_w, awq_z, awq_s, y_awq_ref, awq_scale) {
        (Some(awq_w), Some(awq_z), Some(awq_s), Some(y_awq), Some(a_scale)) => {
            let in0 = phase0.has(arms::AWQ);
            Some(AwqArm {
                w: {
                    // The allocation tail, added HERE and nowhere
                    // else: the kernel reads up to `NPG*128` words
                    // past the start of the last row, and those words
                    // must exist. They are NOT billed, being an
                    // addressing margin and not a byte the algorithm
                    // carries, and billing them would inflate the
                    // competitor arm by a weight it does not bear.
                    let mut wup = awq_w;
                    wup.resize(d_out * aww + awq_tail, 0);
                    up_or_hold_u32(wup, in0)?
                },
                z: up_or_hold_u32(awq_z, in0)?,
                s: up_or_hold_u16(awq_s, in0)?,
                bytes: awq_bytes(d_out, d_in),
                y_ref: y_awq,
                scale: a_scale,
            })
        }
        _ => None,
    };

    let qtip = build_qtip_arm(s, d_out, d_in, union, &xh_f64, phase0, &up_or_hold_u32, &up_or_hold_u16)?;

    let tetra_arm = build_tetra_arm(
        s, tetra, x, d_out, d_in, nblocks, tail_w, chunk, phase0, cuda, &up_or_hold_u32,
    )?;

    Ok(Mat {
        name: s.name.clone(),
        d_out,
        d_in,
        nblocks,
        tail_w,
        slot,
        planes,
        e1v: e1v_arm,
        tetra: tetra_arm,
        p12,
        g70,
        awq,
        qtip,
        gscale: cuda.up_f32(&s.centroids)?,
        rscale: cuda.up_f32(&s.rscale)?,
        tail: cuda.up_f32(if s.tail.is_empty() { &[0.0f32] } else { &s.tail })?,
        w16: cuda.up_u16(&w16)?,
        f16_bytes: (d_out * d_in * 2) as u64,
        y_ref,
        y16_ref,
        y16h_ref,
        scale,
    })
}

/// Every row decoded on the CPU: the f64 references, the f16 weights, and the AWQ arm's
/// quantized buffers, with the block-by-block proofs of the arms that were built.
#[allow(clippy::too_many_arguments)]
fn reference_rows(
    s: &Src,
    x: &[f32],
    table: &ClassTable,
    union: ArmSet,
    d_out: usize,
    d_in: usize,
    nblocks: usize,
    tail_w: usize,
    rt: &llvq_artifact::runtime::RuntimeBlocks,
    planes_data: &Option<Vec<u8>>,
    p12: &Option<llvq_artifact::runtime::Planes12xBlocks>,
    g70: &Option<Golay70Blocks>,
    g70cls: &[Golay70Class],
    golay: &llvq_core::Golay,
) -> Reference {
    let mut w16 = vec![0u16; d_out * d_in];
    let mut y_ref = vec![0.0f64; d_out];
    let mut y16_ref = vec![0.0f64; d_out];
    let mut scale = vec![0.0f64; d_out];
    // ---- the AWQ arm's three buffers ----
    //
    // 🚨 The weight buffer carries an ALLOCATION TAIL, and forgetting
    // it is an `illegal memory access` after twenty minutes of
    // transcoding. The `float4` load of the weights is UNCONDITIONAL
    // in their kernel (`awq_gemv.cu`), whereas the guard
    // `inputs_ptr_delta + ic_0 < IC/8` protects only the input and the
    // accumulation. The highest u32 index reached is therefore
    // `(OC-1)*weight_w + NPG*128 - 1`, past `OC*weight_w` when `IC/8`
    // is not a multiple of `NPG*128`. The overruns of the inner rows
    // fall back into the next row, only the last one goes out, hence a
    // single tail and not a widened stride.
    let (aww, awz, aws) = awq_strides(d_in);
    // ⚠️ The tail is NOT allocated here but at upload. If it were,
    // `chunks_mut(chunk * aww)` could return one slice more than
    // `y_ref.chunks_mut(chunk)`, and `zip` stops on the shorter one,
    // so whole rows would silently go unquantized and the arm would
    // turn green on a buffer full of holes.
    let awq_tail = (awz * 128).saturating_sub(aww);
    let awq_on = union.has(arms::AWQ);
    let mut awq_w = awq_on.then(|| vec![0u32; d_out * aww]);
    let mut awq_z = awq_on.then(|| vec![0u32; d_out * awz]);
    let mut awq_s = awq_on.then(|| vec![0u16; d_out * aws]);
    let mut y_awq_ref = awq_on.then(|| vec![0.0f64; d_out]);
    let mut awq_scale = awq_on.then(|| vec![0.0f64; d_out]);
    // The activation as the AWQ kernel, and cublasf16, sees it:
    // binary16. The reference must be computed against this one, not
    // against the f32, otherwise the arm is billed for a deviation
    // that is not its own.
    let cublas_on = union.has(arms::CUBLASF16);
    // Same rule, and the same 2026-08-20 oversight: the QTIP arm
    // forms its reference against the activation ROUNDED to binary16,
    // because that is the one its kernel reads. Without this input
    // there is no reference to form at all.
    let qtip_on = union.has(arms::QTIP);
    let xh_f64: Option<Vec<f64>> = (awq_on || cublas_on || qtip_on)
        .then(|| x[..d_in].iter().map(|&v| f16_to_f64(f16_bits(v))).collect());
    // The cublasf16 arm's reference: f16 weights × binary16 input,
    // accumulated in f64. Neither `y_ref` (exact weights) nor
    // `y16_ref` (f16 weights but f32 input) describes what GemmEx
    // computes.
    let mut y16h_ref = cublas_on.then(|| vec![0.0f64; d_out]);
    let nthreads = std::thread::available_parallelism().map_or(8, |n| n.get());
    let chunk = d_out.div_ceil(nthreads);
    let n_chunks = d_out.div_ceil(chunk);
    // The per-thread slices of the optional buffers. ⚠️ The trap this
    // shape avoids: `chunks_mut` on an EMPTY Vec returns zero slices,
    // and the driver's `zip` would stop at the shortest one, so the
    // whole reference loop would be skipped in silence. Hence
    // `Vec<Option<&mut [T]>>` of length n_chunks EXACTLY, in every
    // case.
    fn opt_chunks<T>(
        v: &mut Option<Vec<T>>,
        chunk: usize,
        n: usize,
    ) -> Vec<Option<&mut [T]>> {
        match v.as_mut() {
            Some(v) => {
                let out: Vec<Option<&mut [T]>> =
                    v.chunks_mut(chunk).map(Some).collect();
                assert_eq!(out.len(), n, "optional slices misaligned");
                out
            }
            None => (0..n).map(|_| None).collect(),
        }
    }
    let awc_v = opt_chunks(&mut awq_w, chunk * aww, n_chunks);
    let azc_v = opt_chunks(&mut awq_z, chunk * awz, n_chunks);
    let asc_v = opt_chunks(&mut awq_s, chunk * aws, n_chunks);
    let yawc_v = opt_chunks(&mut y_awq_ref, chunk, n_chunks);
    let asqc_v = opt_chunks(&mut awq_scale, chunk, n_chunks);
    let y16hc_v = opt_chunks(&mut y16h_ref, chunk, n_chunks);
    std::thread::scope(|sc| {
        for (ci, (((((((((w16c, yc), y16c), scc), awc), azc), asc), yawc), asqc), y16hc)) in
            w16.chunks_mut(chunk * d_in)
                .zip(y_ref.chunks_mut(chunk))
                .zip(y16_ref.chunks_mut(chunk))
                .zip(scale.chunks_mut(chunk))
                .zip(awc_v)
                .zip(azc_v)
                .zip(asc_v)
                .zip(yawc_v)
                .zip(asqc_v)
                .zip(y16hc_v)
                .enumerate()
        {
            let (rt, table, x, src, planes) = (&rt, &table, &x, &s, &planes_data);
            let p12 = &p12;
            let (g70, g70cls, golay) = (&g70, &g70cls, &golay);
            let xh = &xh_f64;
            sc.spawn(move || {
                let (mut awc, mut azc, mut asc, mut yawc, mut asqc, mut y16hc) =
                    (awc, azc, asc, yawc, asqc, y16hc);
                let mut wrow = vec![0.0f64; d_in];
                for lr in 0..yc.len() {
                    let row = ci * chunk + lr;
                    wrow.fill(0.0);
                    for p in 0..nblocks {
                        let (pt, gain) = rt.decode_block(table, row * nblocks + p);
                        // The bijection proof: every block of the
                        // Planes14 stream decodes to exactly the
                        // point and gain the Slot32 stream carries.
                        // Each proof runs iff its arm is selected —
                        // it is part of that arm's build, and the
                        // published five/six-arm runs all ran it.
                        if let Some(planes) = planes {
                            let (ppt, pgain) =
                                planes14_decode_block(planes, table, row * nblocks + p);
                            assert_eq!(
                                (pt, gain),
                                (ppt, pgain),
                                "{}: block {}, Planes14 is not a bijection of \
                                 Slot32",
                                src.name,
                                row * nblocks + p
                            );
                        }
                        // The overlay proof: main stream + exception
                        // records reconstruct the exact block — the
                        // approximation must be invisible here.
                        if let Some(p12) = p12 {
                            let (xpt, xgain) =
                                p12.decode_block(table, row * nblocks + p);
                            assert_eq!(
                                (pt, gain),
                                (xpt, xgain),
                                "{}: block {}, the Planes12x overlay does not \
                                 reconstruct the exact block",
                                src.name,
                                row * nblocks + p
                            );
                        }
                        // The E2 proof: main stream + exception
                        // records reconstruct the exact block, and
                        // the origin-holing is invisible here.
                        if let Some(g70) = g70 {
                            let (gpt, ggain) = golay70_decode_block(
                                g70, g70cls, golay, table, row * nblocks + p,
                            );
                            assert_eq!(
                                (pt, gain),
                                (gpt, ggain),
                                "{}: block {}, Golay70 does not reconstruct the exact \
                                 block",
                                src.name,
                                row * nblocks + p
                            );
                        }
                        if let Some(shell) =
                            llvq_core::Leech::shell_index(&pt).filter(|&s| s > 0)
                        {
                            let k = src.centroids[gain as usize] as f64
                                * src.rscale[row] as f64
                                / ((16 * shell) as f64).sqrt();
                            for (i, &v) in pt.iter().enumerate() {
                                wrow[p * DIM + i] = v as f64 * k;
                            }
                        }
                    }
                    for t in 0..tail_w {
                        wrow[nblocks * DIM + t] = src.tail[row * tail_w + t] as f64;
                    }
                    let (mut a, mut b, mut bh, mut ss) = (0.0, 0.0, 0.0, 0.0);
                    for c in 0..d_in {
                        let xv = x[c] as f64;
                        let wv = wrow[c];
                        let hb = f16_bits(wv as f32);
                        w16c[lr * d_in + c] = hb;
                        a += wv * xv;
                        b += f16_to_f64(hb) * xv;
                        // binary16 input: the sum GemmEx sees.
                        if let Some(xhv) = xh.as_deref() {
                            bh += f16_to_f64(hb) * xhv[c];
                        }
                        ss += (wv * xv).abs();
                    }
                    yc[lr] = a;
                    y16c[lr] = b;
                    if let Some(y16hc) = y16hc.as_deref_mut() {
                        y16hc[lr] = bh;
                    }
                    scc[lr] = ss;
                    // The competitor arm, on the SAME weights: w4g128
                    // applied to what LLVQ reconstructed. This is not
                    // Qwen's AWQ checkpoint, it is a w4g128 faithful
                    // in TIME (the kernel has no data-dependent
                    // branch) and exact in REFERENCE, and it will
                    // carry no quality claim.
                    if let (Some(awc), Some(azc), Some(asc), Some(yawc), Some(asqc)) = (
                        awc.as_deref_mut(),
                        azc.as_deref_mut(),
                        asc.as_deref_mut(),
                        yawc.as_deref_mut(),
                        asqc.as_deref_mut(),
                    ) {
                        let xh = xh.as_ref().expect("xh_f64 built with the arm");
                        yawc[lr] = awq_quant_row(
                            &wrow,
                            xh,
                            d_in,
                            &mut awc[lr * aww..(lr + 1) * aww],
                            &mut azc[lr * awz..(lr + 1) * awz],
                            &mut asc[lr * aws..(lr + 1) * aws],
                        );
                        // Its error denominator, on its own weights.
                        let mut sa = 0.0f64;
                        for c in 0..d_in {
                            sa += (wrow[c] * xh[c]).abs();
                        }
                        asqc[lr] = sa;
                    }
                }
            });
        }
    });
    Reference {
        w16, y_ref, y16_ref, scale, aww, awq_tail, awq_w, awq_z, awq_s, y_awq_ref, awq_scale, xh_f64,
        y16h_ref, chunk,
    }
}

/// The QTIP arm of one matrix, when selected.
#[allow(clippy::too_many_arguments)]
fn build_qtip_arm(
    s: &Src,
    d_out: usize,
    d_in: usize,
    union: ArmSet,
    xh_f64: &Option<Vec<f64>>,
    phase0: ArmSet,
    up_or_hold_u32: &impl Fn(Vec<u32>, bool) -> Result<Staged<u32>, String>,
    up_or_hold_u16: &impl Fn(Vec<u16>, bool) -> Result<Staged<u16>, String>,
) -> Result<Option<QtipArm>, String> {
    // ---- the QTIP arm ----
    //
    // Built only when selected, like every other arm. Two things about
    // it are unlike the others and are stated here rather than left to
    // be inferred:
    //
    //  * the payload is pseudo-random, so the weights it decodes are
    //    nobody's weights. See `QtipArm`. The seed is derived from the
    //    shape so a matrix gets the same stream in every phase and in
    //    every process — a control run must reproduce the same object,
    //    and a clock-seeded buffer would silently break that.
    //  * the reference is exact, so this arm is held to OUR threshold
    //    (`TOL`), not to the looser `AWQ_TOL`. That looser bound exists
    //    because AWQ and cuBLAS write binary16 outputs; QTIP writes
    //    f32, so there is nothing to forgive.
    let qtip = match union.has(arms::QTIP) {
        true => {
            use llvq_cuda::qtip_host as qh;
            let kernel = qh::kernel_name(d_out, d_in).ok_or_else(|| {
                format!(
                    "{}: QTIP has no shim for {d_out}x{d_in}, add the \
                     shape to kernels/qtip_glue.cu and to \
                     qtip_host::QTIP_SHAPES rather than guess a name",
                    s.name
                )
            })?;
            // 🚨 Derived from the matrix NAME, not from its shape
            // alone. One seed per shape gave 5 distinct payloads
            // replicated 36 times: the 252 matrices would have
            // produced 21,504 genuinely distinct rows instead of the
            // 1,105,920 the verification claims to cover, and 36
            // replicas of one content test it once. The name stays
            // deterministic from run to run, so a control run
            // reproduces the same object.
            let mut seed = 0xF2_0000_0000u64 ^ ((d_out as u64) << 20) ^ d_in as u64;
            for b in s.name.as_bytes() {
                seed = seed.rotate_left(7) ^ u64::from(*b);
            }
            let buf = qh::pseudo_random_buffer(d_out, d_in, seed);
            let tlut = qh::pseudo_random_tlut(seed ^ 0x5EED);
            let xf = xh_f64.as_ref().expect("binary16 activation in f64 not built");
            let mut y = vec![0.0f64; d_out];
            let mut sc = vec![0.0f64; d_out];
            for row in 0..d_out {
                y[row] = qh::reference_row(&buf, d_in, row, xf, &tlut);
                sc[row] = (0..d_in)
                    .map(|c| (qh::weight_at(&buf, d_in, row, c, &tlut) as f64 * xf[c]).abs())
                    .sum();
            }
            // 512 half2, flat as 1024 binary16 words, in the order the
            // kernel indexes them: entry i is (x, y) at 2i and 2i+1.
            let mut cb = vec![0u16; qh::QTIP_TLUT_LEN * 2];
            for (i, &(a, b)) in tlut.iter().enumerate() {
                cb[2 * i] = f16_bits(a);
                cb[2 * i + 1] = f16_bits(b);
            }
            let in0 = phase0.has(arms::QTIP);
            Some(QtipArm {
                compressed: up_or_hold_u32(qtip_pack_u32(&buf), in0)?,
                codebook: up_or_hold_u16(cb, in0)?,
                bytes: qh::qtip_bytes(d_out, d_in),
                y_ref: y,
                scale: sc,
                kernel,
            })
        }
        false => None,
    };
    Ok(qtip)
}

/// The served Tetra arm of one matrix, from the second file, with its own f64 reference.
#[allow(clippy::too_many_arguments)]
fn build_tetra_arm(
    s: &Src,
    tetra: &std::collections::HashMap<String, TetraSrc>,
    x: &[f32],
    d_out: usize,
    d_in: usize,
    nblocks: usize,
    tail_w: usize,
    chunk: usize,
    phase0: ArmSet,
    cuda: &Cuda,
    up_or_hold_u32: &impl Fn(Vec<u32>, bool) -> Result<Staged<u32>, String>,
) -> Result<Option<TetraArm>, String> {
    // ---- the served Tetra arm, from the SECOND file --------------
    //
    // Matched by NAME, not by position: the two files hold the same
    // 252 projections of the same model, but 36 of them are int4 in
    // the served one and have no lattice arm at all. A positional
    // match would pair `v_proj` of one file with `o_proj` of the other
    // and decode a stream against another projection's scales — which
    // produces numbers, not an error.
    let tetra_arm = match tetra.get(&s.name) {
        None => None,
        Some(t) => {
            assert_eq!(
                (t.d_out, t.d_in),
                (d_out, d_in),
                "{}: the two files disagree on the shape",
                s.name
            );
            // The stream's own stride, checked against what the layout
            // derives rather than trusted — the E1v argument
            // (`:2185-2189`): a stride recomputed at launch is a second
            // formula that can drift from the one that wrote the bytes.
            assert_eq!(
                t.blocks.stride_u32,
                llvq_artifact::tetra48::stride_u32(nblocks),
                "{}: the stream's row stride is not the one the layout derives",
                s.name
            );
            let mut ty = vec![0.0f64; d_out];
            let mut tsc = vec![0.0f64; d_out];
            // Built once and shared: `Tetra::new()` builds the trio's
            // tables, and one per row-chunk would dominate the build.
            let td = Tetra::new();
            std::thread::scope(|sc| {
                for (ci, (yc, scc)) in
                    ty.chunks_mut(chunk).zip(tsc.chunks_mut(chunk)).enumerate()
                {
                    let (blocks, t, x, td) = (&t.blocks, t, &x, &td);
                    let name = s.name.as_str();
                    sc.spawn(move || {
                        let mut wrow = vec![0.0f64; d_in];
                        for lr in 0..yc.len() {
                            let row = ci * chunk + lr;
                            wrow.fill(0.0);
                            for b in 0..nblocks {
                                // Decoded back OUT of the packed bytes,
                                // never from the indices that produced
                                // them — those were dropped at read.
                                // A packing mistake is then a
                                // disagreement between two routes.
                                let (pt, gain) = blocks.decode_block(td, row, b);
                                let n2: i64 =
                                    pt.iter().map(|&v| (v as i64) * (v as i64)).sum();
                                if n2 == 0 {
                                    continue; // the origin, and its row is already zero
                                }
                                // `n2 = 16 m` on Λ₂₄, always. If it is
                                // not, the decode is wrong and every
                                // number after it is a fiction — the
                                // kernel indexes `invnorm[n2 >> 4]`
                                // and would silently take a neighbour.
                                assert_eq!(
                                    n2 % 16,
                                    0,
                                    "{}: block {b} of row {row} has |y|² = {n2}, not a \
                                     multiple of 16",
                                    name
                                );
                                let k = t.centroids[gain as usize] as f64
                                    * t.rscale[row] as f64
                                    / (n2 as f64).sqrt();
                                for (i, &v) in pt.iter().enumerate() {
                                    wrow[b * DIM + i] = v as f64 * k;
                                }
                            }
                            for w in 0..tail_w {
                                wrow[nblocks * DIM + w] = t.tail[row * tail_w + w] as f64;
                            }
                            let (mut a, mut ss) = (0.0f64, 0.0f64);
                            for c in 0..d_in {
                                let v = wrow[c] * x[c] as f64;
                                a += v;
                                ss += v.abs();
                            }
                            yc[lr] = a;
                            scc[lr] = ss;
                        }
                    });
                }
            });
            let words: Vec<u32> = t
                .blocks
                .data
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            assert_eq!(
                words.len(),
                d_out * t.blocks.stride_u32,
                "{}: the stream is not a whole number of row strides",
                s.name
            );
            Some(TetraArm {
                words: up_or_hold_u32(words, phase0.has(arms::TETRA48))?,
                stride_u32: t.blocks.stride_u32 as u32,
                gscale: cuda.up_f32(&t.centroids)?,
                rscale: cuda.up_f32(&t.rscale)?,
                tail: cuda.up_f32(if t.tail.is_empty() { &[0.0f32] } else { &t.tail })?,
                // The stream PLUS the tail and the row scales, like
                // every other arm here (`SlotArm` :2113, `PlanesArm`
                // :2122, `P12Arm` :2164): this column is what the
                // kernel READS for one matrix, not what the format
                // spends on codes. It was the stream alone, which
                // would have understated Tetra's b/weight against
                // every arm it is printed beside.
                //
                // And the PADDED length, where Planes14 counts its
                // unpadded stream — not an inconsistency, a difference
                // between the two layouts. Planes14 is flat at a
                // uniform 14-byte stride and its 4-byte pad is a read
                // window past the end; Tetra pads EVERY ROW to a u32
                // boundary and the kernel reads over that pad on every
                // row. `Tetra48Blocks::bits_per_weight` says the same,
                // and names the accounting error of 2026-07-31 that
                // omitting it would repeat.
                // The UNPADDED stream, which is this repository's
                // stated convention and not a choice made here: the
                // Planes14 arm ignores its 4-byte pad — "billing ours
                // here would mix two byte accountings, the exact
                // mistake the K-1 lot eliminated" — and
                // `e1v_host.rs:35` says it outright, "the landing pad
                // is not counted".
                //
                // Tetra's row pad is that same species. `stride_u32`
                // rounds each row to EIGHT bytes and its own doc says
                // why: "what covers the last block's two-word read
                // window". On the 4B's shapes `6·nblocks` is already a
                // multiple of 4 at nblocks 106 and 170, so none of
                // those bytes is alignment — they are landing window.
                //
                // The decisive check is the column itself: Planes14's
                // row reproduces ETAT's 4.8040. A column where five
                // rows reproduce the record and the sixth does not has
                // one wrong row, and billing the pad would print 2.159
                // for a file the dossier publishes at 2.1498.
                bytes: (llvq_artifact::tetra48::TETRA48_BYTES * d_out * nblocks) as u64
                    + (d_out * tail_w) as u64 * 4
                    + d_out as u64 * 4,
                pad_bytes: t.blocks.data.len() as u64
                    - (llvq_artifact::tetra48::TETRA48_BYTES * d_out * nblocks) as u64,
                weights: (d_out * d_in) as u64,
                y_ref: ty,
                scale: tsc,
            })
        }
    };
    Ok(tetra_arm)
}

/// The matrices of the run: the ball file's records, or the synthetic shapes past the L2.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(super) fn load_mats(
    build: &impl Fn(&Src) -> Result<Mat, String>,
    union: ArmSet,
    g70_needed: bool,
    tetra_path: &Option<String>,
    tetra: &std::collections::HashMap<String, TetraSrc>,
    tetra_int4: usize,
    reps: usize,
    rng: &mut SplitMix64,
) -> Result<(Instant, Vec<Mat>, Vec<Src>, u64, String), String> {
    println!(
        "\nBuild: Slot32 transcode (reference, always){}{}{}{}{}, exact-reconstruction \
         proofs of the arms that were built…",
        if union.has(arms::TETRA48) {
            ", Tetra48 (SECOND FILE — its own reference, decoded back out of the \
             packed words)"
        } else {
            ""
        },
        if union.has(arms::PLANES14) { ", Planes14" } else { "" },
        if union.has(arms::PLANES12X) {
            ", Planes12x (L = 5 → L ≤ 4 swap included)"
        } else {
            ""
        },
        if g70_needed { ", Golay70 (origin holes + E2 exceptions)" } else { "" },
        if union.has(arms::AWQ) { ", AWQ w4g128" } else { "" },
    );
    let t0 = Instant::now();
    let mut mats = Vec::new();
    // Kept alive past `build` for the A4 fusion arm, which concatenates the
    // *indices* of q/k/v (and gate/up) before transcoding, not their
    // transcoded streams. Empty on the synthetic path, which is what turns
    // the fusion arm off there: `q_proj#0` and `q_proj#1` are different
    // repetitions, not a layer, and fusing them would be meaningless.
    // Costs ~1.8 GB of host RAM on the 4B — the same price `bin/matvec`
    // already pays for the Slot32 fusion arm.
    let mut srcs: Vec<Src> = Vec::new();
    let mut n_weights = 0u64;
    let source;

    match std::env::args().nth(1) {
        // ---- the published model ----
        Some(path) => {
            let f = std::fs::File::open(&path).map_err(|e| format!("open {path}: {e}"))?;
            let mut r = std::io::BufReader::new(f);
            let h = llvq_artifact::read_header(&mut r).map_err(|e| e.to_string())?;
            // A v5 Tetra file has no runtime layout yet: refused here by name, never read as a Ball.
            llvq_artifact::runtime::require_ball_kinds(h.kinds(), "planesbench").map_err(|e| e.to_string())?;
            println!("  {path} — {} matrices", h.matrices);
            // The provenance line every published table is read through.
            // With two files it has to name BOTH, and their record counts:
            // a reader who sees one path will read the whole table as one
            // file's, and the Tetra row is not.
            source = match &tetra_path {
                None => format!("the published model ({path})"),
                Some(t) => format!(
                    "TWO FILES — ball: {path} ({} records); tetra: {t} ({} Tetra + \
                     {tetra_int4} int4 g128). The arms do not all read the same bytes",
                    h.matrices,
                    tetra.len()
                ),
            };
            for _ in 0..h.matrices {
                let m = llvq_artifact::read_matrix_raw(&mut r, h.version).map_err(|e| e.to_string())?;
                // The record's own kind, not only the header's declared set.
                llvq_artifact::runtime::require_ball(m.kind, "planesbench")
                    .map_err(|e| e.to_string())?;
                // Every decoder hard-codes one gain bit (`hdr >> 9`).
                assert_eq!(
                    m.centroids.len(),
                    2,
                    "{}: the kernels hard-code 1 gain bit",
                    m.name
                );
                n_weights += (m.d_out * m.d_in) as u64;
                srcs.push(Src {
                    name: m.name.clone(),
                    d_out: m.d_out,
                    d_in: m.d_in,
                    indices: m.indices,
                    gains: m.gains,
                    centroids: [m.centroids[0] as f32, m.centroids[1] as f32],
                    rscale: m.row_scales.iter().map(|&v| v as f32).collect(),
                    tail: m.tail.iter().map(|&v| v as f32).collect(),
                });
            }
            for s in &srcs {
                mats.push(build(s)?);
            }
            // 🚨 Nothing checked that the two files' names agreed, and a
            // total mismatch is the quietest failure this bench can have:
            // the Tetra arm covers no matrix, its bytes and its weights
            // are both zero, `bpw` divides by `max(1)` and prints
            // **0.000 b/weight** — under the 2.20 gate, which then passes
            // on nothing — while `verify_arm` reports its best possible
            // error and the timed pass launches nothing at all.
            //
            // So the match is a fact of the run, printed, and a Tetra
            // record with no home is refused. On the pure Tetra file that
            // is 252 of 252; on the served mixed one, 216 of 252, the 36
            // `v_proj` being int4 and having no lattice arm.
            if !tetra.is_empty() {
                let matched = mats.iter().filter(|m| m.tetra.is_some()).count();
                println!(
                    "  tetra48: {matched} of {} matrices matched by name, from {} Tetra \
                     records",
                    mats.len(),
                    tetra.len()
                );
                if matched != tetra.len() {
                    return Err(format!(
                        "the two files do not name the same projections: {} Tetra records, \
                         {matched} of them found a matrix in the ball file. A record with no \
                         home would drop out of the timed pass in silence, and the b/weight \
                         column would divide by weights nobody read.",
                        tetra.len()
                    ));
                }
            }
        }
        // ---- synthetic, real shapes, repeated past the L2 ----
        None => {
            source = format!("{reps} synthetic repetitions of the 7 shapes");
            for r in 0..reps {
                for &(name, d_out, d_in) in &SHAPES {
                    let n = d_out * (d_in / DIM);
                    n_weights += (d_out * d_in) as u64;
                    mats.push(build(&Src {
                        name: format!("{name}#{r}"),
                        d_out,
                        d_in,
                        indices: (0..n).map(|_| 1 + rng.next() % N13).collect(),
                        gains: (0..n).map(|_| (rng.next() & 1) as u32).collect(),
                        centroids: GSCALE,
                        rscale: (0..d_out)
                            .map(|_| 0.5 + rng.next_gaussian().abs() as f32)
                            .collect(),
                        tail: (0..d_out * (d_in % DIM))
                            .map(|_| rng.next_gaussian() as f32)
                            .collect(),
                    })?);
                }
            }
        }
    }
    Ok((t0, mats, srcs, n_weights, source))
}
