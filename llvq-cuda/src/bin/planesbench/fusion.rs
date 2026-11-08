//! Section A4: the q+k+v and gate+up fusion, its bit-exact check, the persistent sections, its cost.

use super::*;

/// The fused groups, built after the table so none of them is resident while it is timed.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_fused(
    srcs: &[Src],
    union: ArmSet,
    x: &[f32],
    fd: FastDecoder,
    table: ClassTable,
    seg_tetra: bool,
    tetra: &std::collections::HashMap<String, TetraSrc>,
    cuda: &Cuda,
) -> Result<Vec<FusedMat>, String> {
    let mut fused: Vec<FusedMat> = Vec::new();
    // The section requires BOTH arms it fuses: on a selection without
    // slot32 or without planes14 it has neither a stream to concatenate nor
    // a separate arm to compare against — it skips, and says so.
    let a4_on = union.has(arms::SLOT32) && union.has(arms::PLANES14);
    if !srcs.is_empty() && !a4_on {
        println!(
            "\n  section A4 (fusion) SKIPPED: it requires slot32 AND planes14 in the \
             LLVQ_BENCH_ARMS selection"
        );
    }
    if !srcs.is_empty() && a4_on {
        let names: Vec<&str> = srcs.iter().map(|s| s.name.as_str()).collect();
        let groups = seg_groups(&names)?;
        let t_fuse = Instant::now();
        for (key, idx) in &groups {
            let sp: Vec<SegPart<'_>> = idx
                .iter()
                .map(|&i| {
                    let s = &srcs[i];
                    SegPart {
                        d_out: s.d_out,
                        d_in: s.d_in,
                        indices: &s.indices,
                        gains: &s.gains,
                        centroids: s.centroids,
                        rscale: &s.rscale,
                        tail: &s.tail,
                    }
                })
                .collect();
            let seg = seg_concat(&sp);
            // The twin of `build`'s guard: a fused matrix reads the same
            // shared activation, and nothing else would notice it reading
            // past the end of it.
            assert!(
                seg.d_in <= x.len(),
                "{key}: d_in {} overruns the activation",
                seg.d_in
            );
            // Slot32 first, then Planes14 off it — the same two-step every
            // unfused matrix goes through, so the two arms differ in
            // geometry and in nothing else.
            let rt = transcode(&fd, &table, &seg.indices, &seg.gains, Layout::Slot32)
                .map_err(|e| e.to_string())?;
            let pdata = planes14_from_slot32(&rt, &table);
            let slot_bytes = rt.data.len() as u64
                + rt.bases.len() as u64 * 4
                + (seg.d_out * seg.tail_w) as u64 * 4
                + seg.d_out as u64 * 4
                // `gs_off`, as for the planes arm below.
                + seg.d_out as u64 * 4;
            // 🕳️ `gs_off` was missing. The FUSED kernel reads one extra
            // u32 a row naming that row's centroid pair
            // (`kernels/planes_seg.cu`) and the unfused one does not — it
            // is the only byte fusion ADDS. Without it the ledger below
            // printed `+0.00%` and the sentence beside it called that a
            // verification.
            let planes_bytes = pdata.len() as u64
                + (seg.d_out * seg.tail_w) as u64 * 4
                + seg.d_out as u64 * 4
                + seg.d_out as u64 * 4;

            let mut sbytes = rt.data.clone();
            sbytes.extend_from_slice(&[0u8; 20]); // the five-word read of the last block
            while !sbytes.len().is_multiple_of(4) {
                sbytes.push(0);
            }
            let swords: Vec<u32> = sbytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let mut pbytes = pdata;
            pbytes.extend_from_slice(&[0u8; 4]); // the four-word read of the last block
            while !pbytes.len().is_multiple_of(4) {
                pbytes.push(0);
            }
            let pwords: Vec<u32> = pbytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();

            // The Tetra stream of the same group, built from the SECOND
            // file. `seg` above is the FIRST file's concatenation: other
            // indices, other centroids, other row scales, other tail. The
            // separate Tetra launches read `tetra[name]`, so the fused one
            // must read the same members, concatenated in the same order
            // with the same `gs_off` rule. Until 2026-11-04 this block
            // transcoded `seg.indices` and launched on `seg`'s scales: the
            // check below then compared two different matrices, and failed
            // at row 0 on 2026-09-20.
            //
            // It costs no transcoding step of its own: each member is
            // already a row-strided stream, and rows sharing `d_in` share
            // `row_stride_u32`, so concatenating by rows is concatenating
            // the word arrays.
            //
            // Only built when the arm is selected. A group is skipped when
            // any member is missing from `tetra`, which is what the served
            // object's int4 `v_proj` does to a q+k+v group: the pair that
            // survives is q+k, and gate+up.
            let (ftetra, tstride, tetra_bytes) = match seg_tetra
                && union.has(arms::TETRA48)
                && idx.iter().all(|&i| tetra.contains_key(&srcs[i].name))
            {
                false => (None, 0u32, 0u64),
                true => {
                    let tp: Vec<SegTetraPart<'_>> = idx
                        .iter()
                        .map(|&i| {
                            let t = &tetra[&srcs[i].name];
                            SegTetraPart {
                                d_out: t.d_out,
                                d_in: t.d_in,
                                blocks: &t.blocks,
                                centroids: t.centroids,
                                rscale: &t.rscale,
                                tail: &t.tail,
                            }
                        })
                        .collect();
                    // The fused output is cut by the FIRST file's row
                    // counts (`mats[pi].d_out` below), so `seg` is passed
                    // for its boundaries only, and they are asserted.
                    // Proved on the Mac by
                    // `tests/tetra48_segment_matches_unfused.rs`.
                    let ts = seg_tetra48_group(key, &seg, &tp);
                    // The same ledger the two ball arms keep, with the same
                    // `gs_off` line: it is the only byte fusion ADDS.
                    let bytes = ts.words.len() as u64 * 4
                        + (ts.d_out * ts.tail_w) as u64 * 4
                        + ts.d_out as u64 * 4
                        + ts.d_out as u64 * 4;
                    let ft = FusedTetra {
                        words: cuda.up_u32(&ts.words)?,
                        gscale: cuda.up_f32(&ts.centroids)?,
                        gs_off: cuda.up_u32(&ts.gs_off)?,
                        rscale: cuda.up_f32(&ts.rscale)?,
                        tail: cuda
                            .up_f32(if ts.tail.is_empty() { &[0.0f32] } else { &ts.tail })?,
                    };
                    (Some(ft), ts.stride_u32 as u32, bytes)
                }
            };

            fused.push(FusedMat {
                name: key.clone(),
                d_out: seg.d_out,
                nblocks: seg.nblocks,
                tail_w: seg.tail_w,
                words: cuda.up_u32(&swords)?,
                bases: cuda.up_u32(&rt.bases)?,
                pwords: cuda.up_u32(&pwords)?,
                gscale: cuda.up_f32(&seg.centroids)?,
                gs_off: cuda.up_u32(&seg.gs_off)?,
                rscale: cuda.up_f32(&seg.rscale)?,
                tail: cuda
                    .up_f32(if seg.tail.is_empty() { &[0.0f32] } else { &seg.tail })?,
                slot_bytes,
                planes_bytes,
                tetra: ftetra,
                tstride,
                tetra_bytes,
                parts: idx.clone(),
            });
        }
        println!(
            "\n  fusion: {} groups ({} matrices → {}), transcoded in {:.0} s",
            fused.len(),
            fused.iter().map(|f| f.parts.len()).sum::<usize>(),
            fused.len(),
            t_fuse.elapsed().as_secs_f64()
        );
    }
    Ok(fused)
}

/// Section A4 on its own output buffer: the bit-exact check of every fused group, the
/// persistent sections, then the timed rounds and the cost.
// The body was moved verbatim out of `run`, where these names were owned locals; they are
// reference parameters here, so `&name` borrows a reference. Kept as moved.
#[allow(clippy::too_many_arguments, clippy::needless_borrow)]
pub(super) fn a4_section(
    mats: Vec<Mat>,
    fused: Vec<FusedMat>,
    d_y: cudarc::driver::CudaSlice<f32>,
    max_dout: usize,
    cuda: &Cuda,
    seg_tetra: bool,
    pers_flags: &llvq_cuda::persbench::Flags,
    src: &KernelSource,
    tile: llvq_cuda::tile::Tile,
    tetra_tabs: &Option<TetraTabs>,
    d_x: &cudarc::driver::CudaSlice<f32>,
    tetra: &std::collections::HashMap<String, TetraSrc>,
    x: &[f32],
    shared: u32,
    run_slot: &impl RunMat<f32>,
    run_planes: &impl RunMat<f32>,
    run_tetra48: &impl RunMat<f32>,
    run_slot_seg: &impl RunSeg<()>,
    run_planes_seg: &impl RunSeg<()>,
    run_tetra48_seg: &impl RunSeg<bool>,
) -> Result<(), String> {
    // An output buffer of this section's own: a fused arm writes up to
    // 19,456 rows, and that is exactly what we refuse to impose on the
    // table's `d_y`. The two sections no longer share anything.
    let a4_dout = fused
        .iter()
        .map(|f| f.d_out)
        .chain(std::iter::once(max_dout))
        .max()
        .expect("the chain carries at least max_dout");
    drop(d_y);
    let mut d_y = cuda.zeros_f32(a4_dout)?;

    if !fused.is_empty() {
        // A fused row runs the same blocks, in the same order, with the
        // same centroids as the unfused row it came from — nothing is
        // reassociated — so the outputs must agree BIT FOR BIT. A tolerance
        // here would let a wrong `gs_off` through: swapping two segments'
        // centroids moves some rows by ~2× and leaves the rest untouched,
        // which a global epsilon on a different row still passes. The
        // Slot32 arm established that this equality is achievable on this
        // card and this compiler (921,600 rows, 2026-08-05); the Planes14
        // arm is the same claim about `tv_planes_seg` and is *not*
        // established until this block runs.
        println!("\n  Fusion (A4) — output against the unfused matrices");
        // Counted and printed rather than assumed: on the served object
        // every q+k+v group is skipped, because `v_proj` is int4, and a
        // silent skip would read as a pass.
        let mut tetra_checked = 0usize;
        for fm in &fused {
            // Layout 2 is Tetra, appended for the rule every arrival
            // follows. It is the only one that can be ABSENT: a group whose
            // members are not all Tetra has no fused stream, and the
            // closure says so rather than launching on nothing.
            for layout in 0..3 {
                match layout {
                    0 => run_slot_seg(fm, &mut d_y)?,
                    1 => run_planes_seg(fm, &mut d_y)?,
                    _ => {
                        if !run_tetra48_seg(fm, &mut d_y)? {
                            continue;
                        }
                        tetra_checked += 1;
                    }
                }
                cuda.sync()?;
                let got = cuda.down_f32(&d_y)?;
                let mut at = 0usize;
                for &pi in &fm.parts {
                    let m = &mats[pi];
                    match layout {
                        0 => run_slot(m, &mut d_y)?,
                        1 => run_planes(m, &mut d_y)?,
                        _ => run_tetra48(m, &mut d_y)?,
                    }
                    cuda.sync()?;
                    let want = cuda.down_f32(&d_y)?;
                    if got[at..at + m.d_out] != want[..m.d_out] {
                        let bad =
                            (0..m.d_out).find(|&r| got[at + r] != want[r]).unwrap_or(0);
                        return Err(format!(
                            "{} / {} / {}: row {bad} is {} fused against {} \
                             separate",
                            fm.name,
                            m.name,
                            match layout {
                                0 => "Slot32",
                                1 => "Planes14",
                                _ => "Tetra48",
                            },
                            got[at + bad],
                            want[bad]
                        ));
                    }
                    at += m.d_out;
                }
            }
        }
        println!(
            "  bit-exact against the unfused arms: {} groups on Slot32 and Planes14, \
             {tetra_checked} on Tetra48",
            fused.len()
        );
        if tetra_checked == 0 {
            match seg_tetra {
                false => println!(
                    "  Tetra fusion not requested (LLVQ_SEG_TETRA unset): \
                     `tv_tetra48_seg` is compiled and not exercised here"
                ),
                true => println!(
                    "  ⚠️ LLVQ_SEG_TETRA=1 and NO Tetra group was checked: no fused group \
                     has a Tetra stream, so `tv_tetra48_seg` is compiled and UNPROVEN"
                ),
            }
        }
        println!(
            "  {} groups, {} rows — identical BIT FOR BIT, on Slot32 AND on Planes14",
            fused.len(),
            fused.iter().map(|f| f.d_out).sum::<usize>()
        );

        // Cost. Four arms, interleaved in every round, same process: LLVQ
        // against LLVQ on each layout, and the two layouts against each
        // other. No FP16 arm, deliberately — fusing the FP16 witness would
        // mean holding a second copy of 7.27 GB of f16 weights, so a
        // fused-LLVQ / unfused-FP16 ratio would credit the format for a
        // geometry change. Every number below is a DELTA, not a ratio.
        // Six series, not four: arms 4 and 5 are Tetra separate and Tetra
        // fused, appended after the historical four so no existing
        // denominator moves. They are timed in the SAME rounds, because a
        // delta of two minima from rounds that never coexisted is the
        // mistake this file documents.
        // ---- the persistent sections (L1, the row loops): built after
        // A4's proof, timed in A4's rounds ----
        // The codec outlives the section, whose f64 references borrow it.
        let codec = pers_flags.tetra_occ.then(llvq_search::tetra::Tetra::new);
        let mut pers_sec = None;
        if pers_flags.any() || pers_flags.dump_cubin.is_some() {
            let mut sec = llvq_cuda::persbench::Section::new(pers_flags.clone(), &cuda, &src.text, &tile.define())?;
            let tt = tetra_tabs.as_ref().expect("tetra48 tables not uploaded");
            let tables = llvq_cuda::persbench::Tables {
                rows: dptr(&tt.rows, &cuda),
                pref: dptr(&tt.prefixes, &cuda),
                br: dptr(&tt.branches, &cuda),
                suf: dptr(&tt.suffixes, &cuda),
                invnorm: dptr(&tt.invnorm, &cuda),
            };
            let xp = dptr(&d_x, &cuda);
            // The f64 value of row r of one Tetra matrix, from the second
            // file's own records: a row that misses its owed equality is
            // judged against it.
            let f64_of = |name: &str| {
                let src = &tetra[name];
                let x = &x;
                let codec = codec.as_ref().expect("built when a Tetra section is on");
                move |r: usize| {
                    let tail: Vec<f64> = (0..src.d_in % DIM).map(|i| src.tail[r * (src.d_in % DIM) + i] as f64).collect();
                    llvq_cuda::pers::tetra_row_f64(
                        src.d_in / DIM,
                        |j| src.blocks.decode_block(codec, r, j),
                        src.centroids,
                        src.rscale[r],
                        &tail,
                        x,
                    )
                }
            };
            if pers_flags.tetra_occ {
                let zeros = cuda.up_u32(&vec![0u32; max_dout.max(19456)])?;
                let mut sites = Vec::new();
                for fm in &fused {
                    let Some(t) = fm.tetra.as_ref() else { continue };
                    run_tetra48_seg(fm, &mut d_y)?;
                    cuda.sync()?;
                    let reference = cuda.down_f32(&d_y)?[..fm.d_out].to_vec();
                    let parts: Vec<(String, usize)> = fm.parts.iter().map(|&pi| (mats[pi].name.clone(), mats[pi].d_out)).collect();
                    let rows_of: Vec<_> = parts.iter().map(|(n, d)| (Box::new(f64_of(n)) as Box<dyn Fn(usize) -> (f64, f64)>, *d)).collect();
                    sites.push(llvq_cuda::persbench::PersallSite {
                        name: fm.name.clone(),
                        words: dptr(&t.words, &cuda),
                        gscale: dptr(&t.gscale, &cuda),
                        gs_off: dptr(&t.gs_off, &cuda),
                        rscale: dptr(&t.rscale, &cuda),
                        tail: dptr(&t.tail, &cuda),
                        d_out: fm.d_out as u32,
                        nblocks: fm.nblocks as u32,
                        tail_w: fm.tail_w as u32,
                        stride_u32: fm.tstride,
                        reference,
                        f64_row: Box::new(move |mut r: usize| {
                            for (f, d) in &rows_of {
                                if r < *d {
                                    return f(r);
                                }
                                r -= d;
                            }
                            unreachable!("row past the group")
                        }),
                    });
                }
                for (i, m) in mats.iter().enumerate() {
                    if fused.iter().any(|f| f.parts.contains(&i) && f.tetra.is_some()) {
                        continue;
                    }
                    let Some(a) = m.tetra.as_ref() else { continue };
                    run_tetra48(m, &mut d_y)?;
                    cuda.sync()?;
                    let reference = cuda.down_f32(&d_y)?[..m.d_out].to_vec();
                    sites.push(llvq_cuda::persbench::PersallSite {
                        name: m.name.clone(),
                        words: dptr(a.words.dev(), &cuda),
                        gscale: dptr(&a.gscale, &cuda),
                        gs_off: dptr(&zeros, &cuda),
                        rscale: dptr(&a.rscale, &cuda),
                        tail: dptr(&a.tail, &cuda),
                        d_out: m.d_out as u32,
                        nblocks: m.nblocks as u32,
                        tail_w: m.tail_w as u32,
                        stride_u32: a.stride_u32,
                        reference,
                        f64_row: Box::new(f64_of(&m.name)),
                    });
                }
                sec.persall_setup(sites, tables, xp, shared)?;
                std::mem::forget(zeros); // read by every timed launch; freed with the process
            }
            if pers_flags.trows {
                // The 252 matrices of the bare 4B file in served order:
                // 36 layers, each q, k, v, o, gate, up, down.
                const PROJS: [&str; 7] = [
                    "self_attn.q_proj",
                    "self_attn.k_proj",
                    "self_attn.v_proj",
                    "self_attn.o_proj",
                    "mlp.gate_proj",
                    "mlp.up_proj",
                    "mlp.down_proj",
                ];
                let mut sites = Vec::new();
                for name in (0..36).flat_map(|l| PROJS.iter().map(move |p| format!("model.layers.{l}.{p}.weight"))) {
                    let m = mats
                        .iter()
                        .find(|m| m.name == name)
                        .ok_or_else(|| format!("LLVQ_TROWS: no matrix {name} in the bench"))?;
                    let a = m.tetra.as_ref().ok_or_else(|| format!("LLVQ_TROWS: {name} has no Tetra stream"))?;
                    let src = &tetra[&name];
                    sites.push(llvq_cuda::persbench::RowSite {
                        name: name.clone(),
                        words: dptr(a.words.dev(), &cuda),
                        gscale: dptr(&a.gscale, &cuda),
                        rscale: dptr(&a.rscale, &cuda),
                        // The file's binary16 tail, widened to f32 by the
                        // reader and narrowed back here: exact both ways.
                        tail_f16: src.tail.iter().map(|&v| llvq_cuda::f16_bits(v)).collect(),
                        d_out: m.d_out as u32,
                        d_in: m.d_in as u32,
                        stride_u32: a.stride_u32,
                    });
                }
                sec.trows_setup(&sites, tables, xp, x.len(), &tile.define())?;
            }
            pers_sec = Some((sec, tables, xp));
        }

        a4_time_and_cost(
            mats, fused, d_y, pers_sec, cuda, run_slot, run_planes, run_tetra48, run_slot_seg, run_planes_seg,
            run_tetra48_seg,
        )?;
    }
    Ok(())
}

/// Section A4's rounds, six series interleaved, then the deltas and the bytes.
#[allow(clippy::too_many_arguments)]
fn a4_time_and_cost(
    mats: Vec<Mat>,
    fused: Vec<FusedMat>,
    mut d_y: cudarc::driver::CudaSlice<f32>,
    mut pers_sec: Option<(llvq_cuda::persbench::Section<'_>, llvq_cuda::persbench::Tables, u64)>,
    cuda: &Cuda,
    run_slot: &impl RunMat<f32>,
    run_planes: &impl RunMat<f32>,
    run_tetra48: &impl RunMat<f32>,
    run_slot_seg: &impl RunSeg<()>,
    run_planes_seg: &impl RunSeg<()>,
    run_tetra48_seg: &impl RunSeg<bool>,
) -> Result<(), String> {
    let mut tf: [Vec<f64>; 6] = [
        Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(),
    ];
    for rep in 0..ROUNDS {
        for (arm, ta) in tf.iter_mut().enumerate() {
            // Arms 4 and 5 are the Tetra pair. Without a fused stream
            // they would measure the same launches twice and print a
            // delta of zero, which reads as "the fusion is worth
            // nothing" rather than "the fusion did not run".
            if arm >= 4 && !fused.iter().any(|f| f.tetra.is_some()) {
                continue;
            }
            let tin = Instant::now();
            match arm {
                // 0/2: everything separate. 1/3: the fusible groups
                // fused, o_proj and down_proj exactly as before.
                0 => {
                    for m in &mats {
                        run_slot(m, &mut d_y)?;
                    }
                }
                1 => {
                    for fm in &fused {
                        run_slot_seg(fm, &mut d_y)?;
                    }
                    for (i, m) in mats.iter().enumerate() {
                        if !fused.iter().any(|f| f.parts.contains(&i)) {
                            run_slot(m, &mut d_y)?;
                        }
                    }
                }
                2 => {
                    for m in &mats {
                        run_planes(m, &mut d_y)?;
                    }
                }
                4 => {
                    for m in &mats {
                        run_tetra48(m, &mut d_y)?;
                    }
                }
                5 => {
                    // A group whose members are not all Tetra has no
                    // fused stream; its members run separately here, so
                    // arms 4 and 5 always cover the same matrices and
                    // their delta is the fusion and nothing else.
                    for fm in &fused {
                        if !run_tetra48_seg(fm, &mut d_y)? {
                            for &pi in &fm.parts {
                                run_tetra48(&mats[pi], &mut d_y)?;
                            }
                        }
                    }
                    for (i, m) in mats.iter().enumerate() {
                        if !fused.iter().any(|f| f.parts.contains(&i)) {
                            run_tetra48(m, &mut d_y)?;
                        }
                    }
                }
                _ => {
                    for fm in &fused {
                        run_planes_seg(fm, &mut d_y)?;
                    }
                    for (i, m) in mats.iter().enumerate() {
                        if !fused.iter().any(|f| f.parts.contains(&i)) {
                            run_planes(m, &mut d_y)?;
                        }
                    }
                }
            }
            cuda.sync()?;
            let s = tin.elapsed().as_secs_f64();
            if rep >= WARMUP {
                ta.push(s);
            }
        }
        if let Some((sec, tables, xp)) = pers_sec.as_mut() {
            sec.time_round(rep >= WARMUP, *tables, *xp)?;
        }
    }
    if let Some((sec, _, _)) = &pers_sec {
        sec.report(&tf[4], &tf[5]);
    }
    let [ts, tss, tp, tps, tt, tts] = tf;
    // Deltas formed ROUND BY ROUND, exactly as the ratios are: a
    // difference of two minima taken from rounds that never coexisted
    // is the mistake this repository documents.
    let ds: Vec<f64> = ts.iter().zip(&tss).map(|(a, b)| (a - b) * 1e3).collect();
    let dp: Vec<f64> = tp.iter().zip(&tps).map(|(a, b)| (a - b) * 1e3).collect();
    let rr: Vec<f64> = dp.iter().zip(&ds).map(|(a, b)| a / b).collect();
    let (_, s_sep, _) = spread(ts);
    let (_, s_fus, _) = spread(tss);
    let (_, p_sep, _) = spread(tp);
    let (_, p_fus, _) = spread(tps);
    let (ds_lo, ds_md, ds_hi) = spread(ds);
    let (dp_lo, dp_md, dp_hi) = spread(dp);
    let (rr_lo, rr_md, rr_hi) = spread(rr);
    // The Tetra delta, formed round by round like the other two —
    // and ONLY when the arms ran. Skipping arms 4 and 5 leaves their
    // series empty, and `spread` indexes the median: on 2026-09-20
    // that panicked at the very end of a 25-minute job, after the
    // table it exists for had already been printed. A `continue` was
    // added without reading what consumed its result.
    let tetra_timed = !tt.is_empty() && !tts.is_empty();
    let (dt_lo, dt_md, dt_hi, t_sep, t_fus) = match tetra_timed {
        false => (0.0, 0.0, 0.0, 0.0, 0.0),
        true => {
            let dt: Vec<f64> =
                tt.iter().zip(&tts).map(|(a, b)| (a - b) * 1e3).collect();
            let (_, ts_, _) = spread(tt);
            let (_, tf_, _) = spread(tts);
            let (lo, md, hi) = spread(dt);
            (lo, md, hi, ts_, tf_)
        }
    };

    let n_sep = mats.len();
    let n_fus = fused.len()
        + mats
            .iter()
            .enumerate()
            .filter(|(i, _)| !fused.iter().any(|f| f.parts.contains(i)))
            .count();
    let unfused_bytes = |sel: fn(&Mat) -> u64| -> u64 {
        mats.iter().map(sel).sum::<u64>()
    };
    let fused_bytes = |fsel: fn(&FusedMat) -> u64, sel: fn(&Mat) -> u64| -> u64 {
        fused.iter().map(fsel).sum::<u64>()
            + mats
                .iter()
                .enumerate()
                .filter(|(i, _)| !fused.iter().any(|f| f.parts.contains(i)))
                .map(|(_, m)| sel(m))
                .sum::<u64>()
    };
    let sb_sep = unfused_bytes(|m| arm_bytes(m, arms::SLOT32));
    let sb_fus = fused_bytes(|f| f.slot_bytes, |m| arm_bytes(m, arms::SLOT32));
    let pb_sep = unfused_bytes(|m| arm_bytes(m, arms::PLANES14));
    let pb_fus = fused_bytes(|f| f.planes_bytes, |m| arm_bytes(m, arms::PLANES14));
    // The only byte fusion adds, on either layout: one u32 a fused row.
    let gs_off_bytes: u64 = fused.iter().map(|f| f.gs_off.len() as u64 * 4).sum();

    println!(
        "\n  Cost — {ROUNDS} rounds, {WARMUP} discarded, FOUR arms interleaved, \
         medians\n  {}",
        "-".repeat(78)
    );
    println!(
        "  {:<34}{:>10.3} ms   {n_sep} launches",
        "Slot32, separate matrices",
        s_sep * 1e3
    );
    println!(
        "  {:<34}{:>10.3} ms   {n_fus} launches",
        "Slot32, q+k+v and gate+up fused",
        s_fus * 1e3
    );
    println!(
        "  {:<34}{:>10.3} ms   {n_sep} launches",
        "Planes14, separate matrices",
        p_sep * 1e3
    );
    println!(
        "  {:<34}{:>10.3} ms   {n_fus} launches",
        "Planes14, q+k+v and gate+up fused",
        p_fus * 1e3
    );
    println!("  {}", "-".repeat(78));
    // Tetra before the two historical gains, because its arm is the
    // one this section was extended for. Printed even when it is zero:
    // a fusion that fused nothing must say so, not disappear.
    let tb_sep = unfused_bytes(|m| arm_bytes(m, arms::TETRA48));
    let tb_fus = fused_bytes(|f| f.tetra_bytes, |m| arm_bytes(m, arms::TETRA48));
    if !tetra_timed {
        println!("  Tetra48 fusion: not timed (LLVQ_SEG_TETRA unset)");
    }
    if tetra_timed {
    println!(
        "  {:<34}{:>10.3} ms\n  {:<34}{:>10.3} ms",
        "Tetra48, separate matrices",
        t_sep * 1e3,
        "Tetra48, q+k and gate+up fused",
        t_fus * 1e3
    );
    println!(
        "  gain Tetra48  : {dt_md:.3} ms [{dt_lo:.3}–{dt_hi:.3}]  ({:.1}%)",
        if t_sep > 0.0 { dt_md / (t_sep * 1e3) * 100.0 } else { 0.0 }
    );
    println!(
        "  bytes read — Tetra48  : {:.3} GB fused against {:.3} separate ({:+.2}%)",
        tb_fus as f64 / 1e9,
        tb_sep as f64 / 1e9,
        if tb_sep > 0 {
            (tb_fus as f64 / tb_sep as f64 - 1.0) * 100.0
        } else {
            0.0
        }
    );
    }
    println!(
        "  gain Slot32   : {ds_md:.3} ms [{ds_lo:.3}–{ds_hi:.3}]  ({:.1}%)",
        100.0 * ds_md / (s_sep * 1e3)
    );
    println!(
        "  gain Planes14 : {dp_md:.3} ms [{dp_lo:.3}–{dp_hi:.3}]  ({:.1}%)",
        100.0 * dp_md / (p_sep * 1e3)
    );
    println!(
        "  Planes14 / Slot32 on the gain: {rr_md:.2}× [{rr_lo:.2}–{rr_hi:.2}]\n  \
         WARNING: ratio of two DIFFERENCES. Its spread is that of the two \
         numerators\n  combined, hence far wider than that of the ratios in the table \
         above. Read it\n  as an order of magnitude, never to two decimals. The two \
         \"gain\" lines are the\n  ones that carry the result."
    );
    println!(
        "  landmark: 108 fewer launches × 3.63 µs measured (a3-graph-2026-08-06)\n  \
         = 0.392 ms, independent of the layout. What exceeds that is occupancy."
    );
    println!(
        "  bytes read — Slot32   : {:.3} GB fused against {:.3} separate ({:+.2}%)\n  \
         bytes read — Planes14 : {:.3} GB fused against {:.3} separate ({:+.2}%)",
        sb_fus as f64 / 1e9,
        sb_sep as f64 / 1e9,
        100.0 * (sb_fus as f64 - sb_sep as f64) / sb_sep as f64,
        pb_fus as f64 / 1e9,
        pb_sep as f64 / 1e9,
        100.0 * (pb_fus as f64 - pb_sep as f64) / pb_sep as f64
    );
    println!(
        "  Slot32's STREAM can move: its stride is the widest record of a group of \
         32,\n  and the concatenation regroups at segment boundaries. Planes14's \
         cannot — 14\n  bytes per block, no bases table. What BOTH pay is `gs_off`, one \
         u32 a fused\n  row naming that row's centroid pair: {:.1} MB over the 72 \
         groups, and it is the\n  only byte fusion adds. It was missing from this ledger \
         until 2026-09-10, which\n  printed +0.00% and called that a verification \
         (tests/planes_segment_matches_unfused.rs).\n  Any gain BEYOND this would be a \
         confounder, not a bonus.",
        (gs_off_bytes as f64) / 1e6
    );
    println!(
        "\n  WARNING: THIS BLOCK PRODUCES NO RATIO AGAINST FP16, and authorizes\n  \
         none: the FP16 arm suffers the same underfill on k/v and would gain from\n  \
         the fusion too. Only the two LLVQ → LLVQ DELTAS above are measured."
    );
    Ok(())
}
