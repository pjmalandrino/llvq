// DRAFT, never compiled or run. Written read-only on 2026-11-02 for the
// persistent-kernel lead (wave5-deep/persistent.md, sections 3.12 and 5,
// step L1b), after the adversarial review asked what the 2026-09-20 failure
// of tv_tetra48_seg says about bit identity on the card.
//
// What the repository holds (read at HEAD 06e1602, $0):
//
//   * The failure, recorded in the header of
//     llvq-cuda/tests/tetra48_segment_matches_unfused.rs:10-12:
//         000.gateup / model.layers.0.mlp.gate_proj.weight / Tetra48:
//           row 0 is -0.3080684 fused against -0.57255346 separate
//     A 46.2 % difference at row 0 (computed). An association moved by the
//     compiler moves a row by a few ulp (the bench's worst f64 errors are
//     2e-8 of the sum of |w.x|, banc-252 sweep-64): this is a different
//     matrix, not a contraction.
//   * Why a different matrix: in planesbench section A4, the fused Tetra
//     stream is transcoded from the FIRST file's indices and gains
//     (`seg = seg_concat(&sp)` over `srcs`, the ball records,
//     planesbench.rs:3876-3893, then `transcode_tetra48(&seg.indices,
//     &seg.gains, ..)`, :3954-3990), and launched with the ball file's
//     centroids, gs_off, rscale and tail (`fm.gscale`, `fm.gs_off`,
//     `fm.rscale`, `fm.tail`, :3060-3071, :3990-4004). The unfused arm it is
//     compared with decodes the SECOND file's own words with the second file's
//     centroids, rscale and tail (`m.tetra`, :2910-2919, filled at
//     :1854-1903, whose struct comment says "The **second** file's centroid
//     pair, not the first's", :526-528). On 2026-09-20 the two files were
//     ball-ref-2026-09-20 and dclm-ft-2026-09-19: two encodes of the same
//     projection, so the comparison could not pass whatever the kernel did.
//     `transcode_tetra48` packs any index below 2^47 as a Tetra label
//     (llvq-artifact/src/tetra48.rs:153-172), so a ball index went through
//     unrefused and decoded as some other lattice point.
//   * Why the host tests did not see it: the four tests of commit 8493f4b
//     build the fused and the unfused stream from ONE synthetic source.
//   * Logs: the failing job is 6aafac8152d0dbd7f1d73830 (2026-09-20 09:50
//     UTC, l40sx1, 1,601 s running, ERROR), absent from docs/data/jobs.csv,
//     like its two siblings 6aaf9e00 (COMPLETED) and 6aafc951 (ERROR, the
//     t64 table). `HfApi.fetch_job_logs` returns 0 lines for all three today
//     (a 2026-10-06 job returns 98, a 2026-09-26 job 0: retention near 30
//     days), and the bucket's banc-tetra-fuse-2026-09-20/banc.txt was
//     overwritten at 12:20 UTC by 6aafc951 (12,116 B, the same size as
//     docs/mesures/banc-t64-2026-09-20-brut/banc.txt). The recorded line
//     above is all that survives, and it is enough.
//
// So the kernel is neither cleared nor convicted: it has never been compared
// with a matrix it was given. The fix below makes the comparison meaningful;
// the card then says whether tv_tetra48_seg is bit-exact.

// ---- 1. planesbench, section A4: build the Tetra fused stream from the Tetra file ----
//
// let tparts: Option<Vec<&TetraSrc>> = idx.iter().map(|&i| tetra.get(&srcs[i].name)).collect();
// let tetra_fused = match (seg_tetra && union.has(arms::TETRA48), tparts) {
//     (true, Some(tp)) => {
//         // Same d_in inside a group, so the same nblocks and the same row
//         // stride: concatenating by rows is concatenating the word arrays,
//         // exactly as tetra48_seg.cu's header argues, but of the RIGHT file.
//         let stride = tp[0].blocks.stride_u32;
//         assert!(tp.iter().all(|t| t.blocks.stride_u32 == stride && t.d_in == tp[0].d_in));
//         let mut bytes = Vec::new();
//         let (mut gscale, mut gs_off, mut rscale, mut tail) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
//         for (p, t) in tp.iter().enumerate() {
//             bytes.extend_from_slice(&t.blocks.data);
//             gscale.extend_from_slice(&t.centroids);             // the second file's pair
//             gs_off.extend(std::iter::repeat(2 * p as u32).take(t.d_out));
//             rscale.extend_from_slice(&t.rscale);
//             tail.extend_from_slice(&t.tail);
//         }
//         Some(TetraFused { words: up_u32(&bytes)?, stride, gscale: up_f32(&gscale)?,
//                           gs_off: up_u32(&gs_off)?, rscale: up_f32(&rscale)?, tail: up_f32(&tail)? })
//     }
//     _ => None,
// };
//
// and `run_tetra48_seg` launches with `fm.tetra_fused`'s six buffers instead
// of the ball file's `fm.gscale`, `fm.gs_off`, `fm.rscale`, `fm.tail`.

// ---- 2. the check becomes an invalidation, A3's rule (planesbench.rs:4377-4432) ----
//
// Slot32 and Planes14 keep their fatal equality: they are the established arms.
// For Tetra, an unequal row no longer stops the job:
//   * equal everywhere: "N rows identical BIT FOR BIT on Tetra48";
//   * unequal, and the fused row within 1e-5 of the sum of |w.x| of the
//     second file's f64 reference (`ty`, `tsc` of the build closure,
//     planesbench.rs:2460-2500): "WARNING: association moved by the compiler,
//     NOT a defect", with the first such row and both values; the Tetra
//     fusion arm is still timed;
//   * above 1e-5: "ALERT: ... TETRA FUSION INVALIDATED", the first row, both
//     values and the error; arms 4 and 5 of the cost block are not timed and
//     the block says so; every other number of the job stands.

// ---- 3. the host test that fails on HEAD and passes after the fix ----
//
// In llvq-cuda/tests/tetra48_segment_matches_unfused.rs, beside the four
// tests of 8493f4b, which keep their single source:
//
// #[test]
// fn the_fused_tetra_stream_is_the_tetra_files_not_the_ball_files() {
//     let mut rng = SplitMix64::new(0x5eed);
//     let (ball, d_in) = fixture(&mut rng);          // "first file": indices, centroids, rscale, tail
//     let (tetra, _) = fixture(&mut rng);            // "second file": another encode of the same shapes
//     assert_ne!(ball[0].indices, tetra[0].indices); // the fixture must make them differ
//     let fused = build_tetra_fused(&ball, &tetra);  // the planesbench helper, extracted to seg_host.rs
//     let stride = stride_u32(d_in / DIM);
//     let mut row0 = 0;
//     for (p, t) in tetra.iter().enumerate() {
//         let own = transcode_tetra48(&t.indices, &t.gains, t.d_out, d_in / DIM).unwrap();
//         for r in 0..t.d_out {
//             let a = &fused.words[(row0 + r) * stride * 4..(row0 + r + 1) * stride * 4];
//             let b = &own.data[r * stride * 4..(r + 1) * stride * 4];
//             assert_eq!(a, b, "part {p} row {r}: the fused word is not the second file's");
//             assert_eq!(fused.gs_off[row0 + r] as usize, 2 * p);
//             assert_eq!(fused.rscale[row0 + r], t.rscale[r]);
//         }
//         assert_eq!(&fused.gscale[2 * p..2 * p + 2], &t.centroids[..]);
//         row0 += t.d_out;
//     }
// }
//
// Mutant to kill before calling it green (CLAUDE.md rule 10): build the words
// from `ball` instead of `tetra`, as HEAD does; the test must fail at part 0,
// row 0.
