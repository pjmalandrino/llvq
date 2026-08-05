//! The fused kernel against its FP16 witness, on NVIDIA.
//!
//! `matvec <model.llvq>` measures **the published model**: one token's worth
//! of linear algebra over its 252 projections. With no argument it falls back
//! to synthetic blocks over the seven shapes of Qwen3-4B, repeated until a
//! pass exceeds the card's L2 — useful for iterating on the kernel without
//! moving 1.77 GB, and nothing else.
//!
//! The two paths share every line downstream of the input, so a difference
//! between them can only come from the data. And it does: uniform draws over
//! the cap-13 ball give a wider class mix than a real quantization (65.85 % of
//! the artifact's blocks have exactly 4 levels), hence wider group strides —
//! 5.742 b/weight synthetic against 5.510 measured on the file. **The
//! synthetic run reads more bytes per weight than the model does**, so it is
//! the pessimistic one; only the run with a path is the headline.
//!
//! ## The cache the measurement has to defeat
//!
//! The card reports its L2, and it turned out to be 100.7 MB — not the 48 or
//! 96 that third-party sources give. Getting that wrong is a trap this
//! repository already fell into: 11-17 MB buffers replayed 576 times inside a
//! 48 MB system cache made every earlier LLVQ figure optimistic. On the real
//! model a pass touches 2.5 GB of distinct weights and the question does not
//! arise; the synthetic path sizes itself against the value it read.
//!
//! ## Verification before timing
//!
//! Every row of every matrix is checked against an f64 CPU reference built
//! from the transcoded blocks. Errors are relative to Σ|wᵢ·xᵢ|.
//!
//! ⚠️ The CUDA and Metal outputs are **not** bit-comparable and must never be
//! diffed against each other: `simd_sum` and a `__shfl_xor_sync` butterfly
//! reduce in different orders, and the two compilers contract FMAs by their
//! own rules. Each is checked against the shared f64 reference; two worst
//! errors get published, not one delta.

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("matvec targets NVIDIA GPUs; there is nothing to run here.");
}

#[cfg(target_os = "linux")]
fn main() -> Result<(), String> {
    linux::run()
}

#[cfg(target_os = "linux")]
mod linux {
    use llvq_artifact::runtime::{transcode, ClassTable, Layout};
    use llvq_core::{SplitMix64, DIM};
    use llvq_cuda::gpu::{Cuda, KernelSource};
    use llvq_search::fastdec::FastDecoder;
    use llvq_search::index::N13;
    use std::time::Instant;

    /// Blocks staged per tile: 3072 columns, 12 KB. Emitted into the kernel
    /// source by the host so the staging size and the tiling are one constant.
    const TILE_BLOCKS: usize = 128;
    const THREADS: u32 = 256;
    const GSCALE: [f32; 2] = [0.625, 1.375];
    const TABLE_ENTRIES: usize = 512;
    const REC_WORDS: usize = 6;
    const TOL: f64 = 1e-5;
    const ROUNDS: usize = 7;
    const WARMUP: usize = 2;

    /// The seven projection shapes of Qwen3-4B, `(name, d_out, d_in)`.
    const SHAPES: [(&str, usize, usize); 7] = [
        ("q_proj", 4096, 2560),
        ("k_proj", 1024, 2560),
        ("v_proj", 1024, 2560),
        ("o_proj", 2560, 4096),
        ("gate_proj", 9728, 2560),
        ("up_proj", 9728, 2560),
        ("down_proj", 2560, 9728),
    ];

    /// One matrix's inputs, whatever they came from.
    ///
    /// The synthetic path and the artifact path differ only here; everything
    /// downstream — the f64 reference, the f16 rounding, the upload — is the
    /// same code, so a difference between the two runs can only come from the
    /// data.
    struct Src {
        name: String,
        d_out: usize,
        d_in: usize,
        indices: Vec<u64>,
        gains: Vec<u32>,
        centroids: [f32; 2],
        rscale: Vec<f32>,
        tail: Vec<f32>,
    }

    struct Mat {
        name: String,
        d_out: usize,
        d_in: usize,
        nblocks: usize,
        tail_w: usize,
        words: cudarc::driver::CudaSlice<u32>,
        bases: cudarc::driver::CudaSlice<u32>,
        gscale: cudarc::driver::CudaSlice<f32>,
        rscale: cudarc::driver::CudaSlice<f32>,
        tail: cudarc::driver::CudaSlice<f32>,
        w16: cudarc::driver::CudaSlice<u16>,
        slot_bytes: u64,
        f16_bytes: u64,
        y_ref: Vec<f64>,
        y16_ref: Vec<f64>,
        scale: Vec<f64>,
    }

    fn worst_error(got: &[f32], want: &[f64], scale: &[f64]) -> f64 {
        got.iter()
            .zip(want)
            .zip(scale)
            .map(|((&g, &w), &s)| (g as f64 - w).abs() / s.max(1e-12))
            .fold(0.0, f64::max)
    }

    fn spread(mut v: Vec<f64>) -> (f64, f64, f64) {
        v.sort_by(f64::total_cmp);
        (v[0], v[v.len() / 2], v[v.len() - 1])
    }

    /// f32 → binary16 bits, round to nearest even. Same function as
    /// `llvq_metal::f16_bits`; duplicated rather than shared because the two
    /// crates are platform-gated in opposite directions. The shared home is
    /// lot K0.
    fn f16_bits(x: f32) -> u16 {
        let b = x.to_bits();
        let sign = ((b >> 16) & 0x8000) as u16;
        let exp = ((b >> 23) & 0xff) as i32;
        let man = b & 0x7f_ffff;
        if exp == 255 {
            return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
        }
        let e16 = exp - 127 + 15;
        if e16 >= 31 {
            return sign | 0x7c00;
        }
        if e16 <= 0 {
            if e16 < -10 {
                return sign;
            }
            let man = man | 0x80_0000;
            let shift = (14 - e16) as u32;
            let half = 1u32 << (shift - 1);
            let mut v = man >> shift;
            let rem = man & ((1 << shift) - 1);
            if rem > half || (rem == half && v & 1 == 1) {
                v += 1;
            }
            return sign | v as u16;
        }
        let mut v = (sign as u32) | ((e16 as u32) << 10) | (man >> 13);
        let rem = man & 0x1fff;
        if rem > 0x1000 || (rem == 0x1000 && v & 1 == 1) {
            v += 1;
        }
        v as u16
    }

    fn f16_to_f64(h: u16) -> f64 {
        let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
        let exp = (h >> 10) & 0x1f;
        let man = (h & 0x3ff) as f64;
        sign * match exp {
            0 => man * (-24f64).exp2(),
            31 => {
                if man == 0.0 {
                    f64::INFINITY
                } else {
                    f64::NAN
                }
            }
            e => (1.0 + man / 1024.0) * ((e as f64) - 15.0).exp2(),
        }
    }

    pub fn run() -> Result<(), String> {
        // Three parts, concatenated in dependency order: `llvq_floor.cuh` uses
        // `warp_sum` and `TILE_BLOCKS`, both of which `matvec.cu` defines.
        // NVRTC has no filesystem, so the host does the including.
        let sources =
            llvq_cuda::load_sources_many(&["llvq_slot.cuh", "matvec.cu", "llvq_floor.cuh"])?;
        let defines = format!("#define TILE_BLOCKS {TILE_BLOCKS}u\n");
        let parts: Vec<&str> = std::iter::once(defines.as_str())
            .chain(sources.parts.iter().map(String::as_str))
            .collect();
        let src = KernelSource::new(&parts);
        println!("source NVRTC : {} octets, sha256 {}", src.text.len(), src.sha256);
        if let Some(d) = &sources.overridden_from {
            println!("  ⚠️ SOURCES SURCHARGÉES depuis {d}");
        }

        let cuda = Cuda::new(&src)?;
        let dev = cuda.device()?;
        println!(
            "\n{} — {} SM, L2 {:.1} Mo (lue), {} o de partagée par bloc",
            dev.name,
            dev.sm_count,
            dev.l2_bytes as f64 / 1e6,
            dev.shared_per_block
        );
        // `tv_slot` and `tv_f16` first, and their line is the one to compare
        // against earlier logs: adding the floor probes to the same NVRTC
        // translation unit must not change what the compiler does to them.
        for name in ["tv_slot", "tv_f16", "tv_floor_stream", "tv_floor_tab", "tv_floor_xs"] {
            let r = cuda.report(name)?;
            println!(
                "  {:<10} {:>3} registres, {} o locaux, sm_{}",
                r.name, r.num_regs, r.local_bytes, r.binary_version
            );
            if r.local_bytes != 0 {
                return Err(format!("{name} : {} octets de spill", r.local_bytes));
            }
        }

        // Enough repetitions that one pass streams past the L2 twice over.
        // Reading the cache size rather than assuming it is the whole point:
        // this card reports 100.7 MB where published figures say 48 or 96.
        let one_pass: u64 = SHAPES
            .iter()
            .map(|&(_, o, i)| (o * i) as u64 * 551 / 100 / 8)
            .sum();
        let reps = (2 * dev.l2_bytes as u64 / one_pass + 1).max(2) as usize;

        let fd = FastDecoder::new();
        let table = ClassTable::new(&fd, 1);
        assert!(24 + table.worst_width_slot() <= 160, "fenêtre de 5 mots dépassée");

        let mut tab = vec![0u32; TABLE_ENTRIES * REC_WORDS];
        for e in 0..TABLE_ENTRIES {
            tab[e * REC_WORDS + REC_WORDS - 1] = 1;
        }
        for ci in 0..fd.n_classes() {
            let lv = fd.levels(ci);
            let norm = ((16 * lv.shell) as f64).sqrt();
            let base = (1 + ci) * REC_WORDS;
            for k in 0..lv.len {
                tab[base + k] = ((lv.values[k] as f64 / norm) as f32).to_bits();
            }
            tab[base + REC_WORDS - 1] = lv.len as u32;
        }
        let d_tab = cuda.up_u32(&tab)?;

        let mut rng = SplitMix64::new(0x6_D07);
        let x: Vec<f32> = (0..16384).map(|_| rng.next_gaussian() as f32).collect();
        let d_x = cuda.up_f32(&x)?;

        // Everything downstream of `Src` is shared, so the synthetic run and
        // the artifact run differ only in their input.
        //
        // Takes `&Src` rather than `Src` since 2026-08-05: the fusion arm
        // re-reads the same indices to transcode a concatenated stream, so
        // the sources have to outlive the matrices built from them.
        let build = |s: &Src| -> Result<Mat, String> {
            let (d_out, d_in) = (s.d_out, s.d_in);
            let nblocks = d_in / DIM;
            let tail_w = d_in % DIM;
            assert_eq!(d_out % 8, 0, "{}: CUDA lance des blocs entiers", s.name);
            assert!(d_in <= x.len(), "{}: d_in {d_in} dépasse l'activation", s.name);
            let rt = transcode(&fd, &table, &s.indices, &s.gains, Layout::Slot32)
                .map_err(|e| e.to_string())?;

            let mut w16 = vec![0u16; d_out * d_in];
            let mut y_ref = vec![0.0f64; d_out];
            let mut y16_ref = vec![0.0f64; d_out];
            let mut scale = vec![0.0f64; d_out];
            let nthreads = std::thread::available_parallelism().map_or(8, |n| n.get());
            let chunk = d_out.div_ceil(nthreads);
            std::thread::scope(|sc| {
                for (ci, (((w16c, yc), y16c), scc)) in w16
                    .chunks_mut(chunk * d_in)
                    .zip(y_ref.chunks_mut(chunk))
                    .zip(y16_ref.chunks_mut(chunk))
                    .zip(scale.chunks_mut(chunk))
                    .enumerate()
                {
                    let (rt, table, x, src) = (&rt, &table, &x, &s);
                    sc.spawn(move || {
                        let mut wrow = vec![0.0f64; d_in];
                        for lr in 0..yc.len() {
                            let row = ci * chunk + lr;
                            wrow.fill(0.0);
                            for p in 0..nblocks {
                                let (pt, gain) = rt.decode_block(table, row * nblocks + p);
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
                            let (mut a, mut b, mut ss) = (0.0, 0.0, 0.0);
                            for c in 0..d_in {
                                let xv = x[c] as f64;
                                let wv = wrow[c];
                                let hb = f16_bits(wv as f32);
                                w16c[lr * d_in + c] = hb;
                                a += wv * xv;
                                b += f16_to_f64(hb) * xv;
                                ss += (wv * xv).abs();
                            }
                            yc[lr] = a;
                            y16c[lr] = b;
                            scc[lr] = ss;
                        }
                    });
                }
            });

            let mut bytes = rt.data.clone();
            bytes.extend_from_slice(&[0u8; 20]); // the five-word read of the last block
            while !bytes.len().is_multiple_of(4) {
                bytes.push(0);
            }
            let words: Vec<u32> = bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Ok(Mat {
                name: s.name.clone(),
                d_out,
                d_in,
                nblocks,
                tail_w,
                words: cuda.up_u32(&words)?,
                bases: cuda.up_u32(&rt.bases)?,
                gscale: cuda.up_f32(&s.centroids)?,
                rscale: cuda.up_f32(&s.rscale)?,
                tail: cuda.up_f32(if s.tail.is_empty() { &[0.0f32] } else { &s.tail })?,
                w16: cuda.up_u16(&w16)?,
                slot_bytes: rt.data.len() as u64
                    + rt.bases.len() as u64 * 4
                    + (d_out * tail_w) as u64 * 4
                    + d_out as u64 * 4,
                f16_bytes: (d_out * d_in * 2) as u64,
                y_ref,
                y16_ref,
                scale,
            })
        };

        println!("\nConstruction et transcodage…");
        let t0 = Instant::now();
        let mut mats = Vec::new();
        // Kept alive past `build`: the fusion arm re-transcodes a
        // concatenation of these. Empty on the synthetic path, which has no
        // layer names to group by.
        let mut srcs: Vec<Src> = Vec::new();
        let mut n_weights = 0u64;
        let source;

        match std::env::args().nth(1) {
            // ---- the published model ----
            Some(path) => {
                let f = std::fs::File::open(&path).map_err(|e| format!("open {path}: {e}"))?;
                let mut r = std::io::BufReader::new(f);
                let h = llvq_artifact::read_header(&mut r).map_err(|e| e.to_string())?;
                println!("  {path} — {} matrices", h.matrices);
                source = format!("le modèle publié ({path})");
                for _ in 0..h.matrices {
                    let m = llvq_artifact::read_matrix_raw(&mut r).map_err(|e| e.to_string())?;
                    // Every decoder hard-codes one gain bit (`hdr >> 9`).
                    // A file with a different gain width would transcode into
                    // a coherent stream and decode into garbage.
                    assert_eq!(
                        m.centroids.len(),
                        2,
                        "{}: les noyaux codent 1 bit de gain en dur",
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
            }
            // ---- synthetic, real shapes, repeated past the L2 ----
            None => {
                source = format!("{reps} répétitions synthétiques des 7 formes");
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
        // ---- the fusion arm ----
        //
        // q/k/v share an activation, and so do gate/up. Row-concatenating them
        // turns three grids of 512/128/128 blocks into one of 768, which is
        // what the 2026-08-05 attribution job says k_proj and v_proj are dying
        // of: 128 blocks against 852 resident slots, 157 GB/s, and a ratio
        // against FP16 of 1.06×.
        //
        // The concatenation is done on the *indices*, before transcoding, so
        // the fused matrix is one ordinary Slot32 stream — its groups of 32
        // simply straddle the segment boundaries, which the addressing has
        // always tolerated since they already straddle rows. Only the gain
        // centroids do not concatenate, and `gs_off` names each row's pair.
        struct Fused {
            name: String,
            d_out: usize,
            nblocks: usize,
            tail_w: usize,
            words: cudarc::driver::CudaSlice<u32>,
            bases: cudarc::driver::CudaSlice<u32>,
            gscale: cudarc::driver::CudaSlice<f32>,
            gs_off: cudarc::driver::CudaSlice<u32>,
            rscale: cudarc::driver::CudaSlice<f32>,
            tail: cudarc::driver::CudaSlice<f32>,
            slot_bytes: u64,
            /// Indices into `mats` of the matrices this one replaces, in row
            /// order — the fused output must equal their outputs concatenated.
            parts: Vec<usize>,
        }

        // `None` for anything that is not fusible; the rank fixes the row
        // order inside a group, so the concatenation is deterministic and the
        // per-part comparison below lines up.
        fn family(n: &str) -> Option<(&'static str, usize)> {
            for (i, p) in ["q_proj", "k_proj", "v_proj"].iter().enumerate() {
                if n.contains(p) {
                    return Some(("qkv", i));
                }
            }
            for (i, p) in ["gate_proj", "up_proj"].iter().enumerate() {
                if n.contains(p) {
                    return Some(("gateup", i));
                }
            }
            None
        }

        let mut fused: Vec<Fused> = Vec::new();
        if !srcs.is_empty() {
            let mut groups: Vec<(String, Vec<usize>)> = Vec::new();
            for (i, s) in srcs.iter().enumerate() {
                if let Some((fam, _)) = family(&s.name) {
                    let (layer, _) =
                        llvq_artifact::split_name(&s.name).map_err(|e| e.to_string())?;
                    let key = format!("{layer:03}.{fam}");
                    match groups.iter_mut().find(|(k, _)| *k == key) {
                        Some((_, v)) => v.push(i),
                        None => groups.push((key, vec![i])),
                    }
                }
            }
            for (_, v) in groups.iter_mut() {
                v.sort_by_key(|&i| family(&srcs[i].name).map_or(usize::MAX, |(_, r)| r));
            }

            let t_fuse = Instant::now();
            for (key, idx) in &groups {
                let d_in = srcs[idx[0]].d_in;
                assert!(
                    idx.iter().all(|&i| srcs[i].d_in == d_in),
                    "{key}: fusion exige un d_in commun"
                );
                let nblocks = d_in / DIM;
                let tail_w = d_in % DIM;
                let d_out: usize = idx.iter().map(|&i| srcs[i].d_out).sum();
                assert_eq!(d_out % 8, 0, "{key}: CUDA lance des blocs entiers");

                let mut indices = Vec::with_capacity(d_out * nblocks);
                let mut gains = Vec::with_capacity(d_out * nblocks);
                let mut rscale = Vec::with_capacity(d_out);
                let mut tail = Vec::with_capacity(d_out * tail_w);
                let mut gscale = Vec::with_capacity(2 * idx.len());
                let mut gs_off = Vec::with_capacity(d_out);
                for (seg, &i) in idx.iter().enumerate() {
                    let s = &srcs[i];
                    indices.extend_from_slice(&s.indices);
                    gains.extend_from_slice(&s.gains);
                    rscale.extend_from_slice(&s.rscale);
                    tail.extend_from_slice(&s.tail);
                    gscale.extend_from_slice(&s.centroids);
                    gs_off.extend(std::iter::repeat_n(2 * seg as u32, s.d_out));
                }
                let rt = transcode(&fd, &table, &indices, &gains, Layout::Slot32)
                    .map_err(|e| e.to_string())?;
                let mut bytes = rt.data.clone();
                bytes.extend_from_slice(&[0u8; 20]);
                while !bytes.len().is_multiple_of(4) {
                    bytes.push(0);
                }
                let words: Vec<u32> = bytes
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                fused.push(Fused {
                    name: key.clone(),
                    d_out,
                    nblocks,
                    tail_w,
                    words: cuda.up_u32(&words)?,
                    bases: cuda.up_u32(&rt.bases)?,
                    gscale: cuda.up_f32(&gscale)?,
                    gs_off: cuda.up_u32(&gs_off)?,
                    rscale: cuda.up_f32(&rscale)?,
                    tail: cuda.up_f32(if tail.is_empty() { &[0.0f32] } else { &tail })?,
                    slot_bytes: rt.data.len() as u64
                        + rt.bases.len() as u64 * 4
                        + (d_out * tail_w) as u64 * 4
                        + d_out as u64 * 4,
                    parts: idx.clone(),
                });
            }
            println!(
                "  fusion : {} groupes ({} matrices → {}), transcodés en {:.0} s",
                fused.len(),
                fused.iter().map(|f| f.parts.len()).sum::<usize>(),
                fused.len(),
                t_fuse.elapsed().as_secs_f64()
            );
        }

        let max_dout = mats
            .iter()
            .map(|m| m.d_out)
            .chain(fused.iter().map(|f| f.d_out))
            .max()
            .unwrap();
        let mut d_y = cuda.zeros_f32(max_dout)?;
        println!(
            "  {} matrices, {:.2} Md de poids, en {:.0} s",
            mats.len(),
            n_weights as f64 / 1e9,
            t0.elapsed().as_secs_f64()
        );

        let f_slot = cuda.func("tv_slot")?;
        let f_f16 = cuda.func("tv_f16")?;
        let shared = (TILE_BLOCKS * DIM * 4) as u32;

        let run_slot = |m: &Mat, y: &mut cudarc::driver::CudaSlice<f32>| -> Result<(), String> {
            cuda.launch_slot(
                &f_slot, &m.words, &m.bases, &d_tab, &m.gscale, &m.rscale, &m.tail, &d_x, y,
                m.nblocks as u32, m.tail_w as u32, m.d_out as u32, THREADS, shared,
            )
        };
        let run_f16 = |m: &Mat, y: &mut cudarc::driver::CudaSlice<f32>| -> Result<(), String> {
            cuda.launch_f16(&f_f16, &m.w16, &d_x, y, m.d_in as u32, m.d_out as u32, THREADS, shared)
        };

        println!("\nVérification de chaque ligne contre la référence f64…");
        let (mut wq, mut wf) = (0.0f64, 0.0f64);
        for m in &mats {
            run_slot(m, &mut d_y)?;
            cuda.sync()?;
            let got = cuda.down_f32(&d_y)?;
            let e = worst_error(&got[..m.d_out], &m.y_ref, &m.scale);
            assert!(e < TOL, "{} / LLVQ : {e:.2e}·Σ|w·x|", m.name);
            wq = wq.max(e);

            run_f16(m, &mut d_y)?;
            cuda.sync()?;
            let got = cuda.down_f32(&d_y)?;
            let e = worst_error(&got[..m.d_out], &m.y16_ref, &m.scale);
            assert!(e < TOL, "{} / FP16 : {e:.2e}·Σ|w·x|", m.name);
            wf = wf.max(e);
        }
        let rows: usize = mats.iter().map(|m| m.d_out).sum();
        println!(
            "  {rows} lignes, seuil {TOL:.0e} — pire erreur LLVQ {wq:.1e}, FP16 {wf:.1e} ·Σ|w·x|"
        );

        // One pass = all matrices, one stream, in order. The stream serializes
        // them, which is the dependency a transformer's layers really have.
        // Wall-clock around the pass and a synchronize, exactly as the Metal
        // bench times its command buffer — so the two protocols are the same
        // shape and the two ratios are comparable.
        let mut t_slot = Vec::new();
        let mut t_f16 = Vec::new();
        for rep in 0..ROUNDS {
            for arm in 0..2 {
                let t = Instant::now();
                for m in &mats {
                    if arm == 0 {
                        run_slot(m, &mut d_y)?;
                    } else {
                        run_f16(m, &mut d_y)?;
                    }
                }
                cuda.sync()?;
                let s = t.elapsed().as_secs_f64();
                if rep >= WARMUP {
                    if arm == 0 {
                        t_slot.push(s)
                    } else {
                        t_f16.push(s)
                    }
                }
            }
        }
        let ratio: Vec<f64> = t_f16.iter().zip(&t_slot).map(|(a, b)| a / b).collect();
        let (rlo, rmd, rhi) = spread(ratio);
        let (slo, smd, shi) = spread(t_slot.clone());
        let (flo, fmd, fhi) = spread(t_f16.clone());
        let sb: u64 = mats.iter().map(|m| m.slot_bytes).sum();
        let fb: u64 = mats.iter().map(|m| m.f16_bytes).sum();

        println!("\nUN TOKEN — {} matrices, un stream, bras entrelacés", mats.len());
        println!("  {ROUNDS} rounds, {WARMUP} jetés ; le rapport est formé ROUND PAR ROUND");
        println!("  {}", "-".repeat(80));
        println!(
            "  {:<22}{:>9}{:>9}{:>9}{:>9}{:>9}{:>9}",
            "format", "min ms", "méd ms", "max ms", "Go lus", "b/poids", "Go/s"
        );
        for (n, (lo, md, hi), b) in [
            ("FP16 (128 bits)", (flo, fmd, fhi), fb),
            ("LLVQ fusé (Slot32)", (slo, smd, shi), sb),
        ] {
            println!(
                "  {n:<22}{:>9.3}{:>9.3}{:>9.3}{:>9.2}{:>9.3}{:>9.0}",
                lo * 1e3,
                md * 1e3,
                hi * 1e3,
                b as f64 / 1e9,
                b as f64 * 8.0 / n_weights as f64,
                b as f64 / lo / 1e9
            );
        }
        println!("  {}", "-".repeat(80));
        println!("  vs FP16 : {rmd:.2}× [{rlo:.2}–{rhi:.2}]");
        let read_per_pass: u64 = mats.iter().map(|m| m.slot_bytes).sum();
        println!("\n  source : {source}");
        println!(
            "  {:.0} Mo de poids LLVQ distincts par passe, soit {:.1}× la L2 lue.\n  \
             Sous 1× on mesurerait le cache et pas la DRAM — le piège qui a rendu\n  \
             optimiste toute mesure LLVQ antérieure au 2026-07-31.",
            read_per_pass as f64 / 1e6,
            read_per_pass as f64 / dev.l2_bytes as f64
        );
        // The warning has to follow the data, not the binary. Printing it
        // unconditionally is how a run on the published model ends up filed
        // under "synthetic" by whoever reads the log six months later.
        if std::env::args().nth(1).is_none() {
            println!(
                "  ⚠️ blocs SYNTHÉTIQUES, tirés uniformément sur la boule m ≤ 13 : le chemin de\n  \
                 décodage est exercé à l'identique, mais le mélange de classes d'un vrai\n  \
                 artefact ne l'est pas, donc ses strides de groupe et son trafic d'octets\n  \
                 diffèrent. Ce rapport mesure le NOYAU, pas le modèle publié — passer le\n  \
                 chemin du .llvq en argument pour cela."
            );
        }
        // The tied lm_head is unquantized and read once per token by both
        // arms — the same constant added to each, and what caps the ratio.
        // The Metal bench prints this; without it a projections-only ratio
        // gets read as an end-to-end one, which is the risk this dossier
        // names first.
        let head_bytes = 389_070_848f64 * 2.0;
        let bw = fb as f64 / flo;
        let head_s = head_bytes / bw;
        println!(
            "\n  Avec le lm_head f16 non quantifié ({:.0} M poids, {:.2} ms sur les deux\n  \
             bras au débit mesuré) : {:.2}× au lieu de {rmd:.2}×. C'est lui qui plafonne\n  \
             le rapport, et attention — normes, activations, attention et rotation ne\n  \
             sont mesurées ni ici ni là.",
            389_070_848f64 / 1e6,
            head_s * 1e3,
            (flo + head_s) / (slo + head_s)
        );
        println!(
            "\n  ⚠️ à ne JAMAIS comparer au chiffre Metal ligne à ligne : les deux réductions\n  \
             somment dans des ordres différents et les deux compilateurs contractent les FMA\n  \
             selon leurs propres règles. Chacun contre sa référence f64, deux pires erreurs."
        );

        // ================= ATTRIBUTION =================
        //
        // Everything above is the published protocol, untouched. What follows
        // runs in *additional* rounds and answers a different question: the
        // aggregate says the LLVQ arm reads 2.50 GB at 431 GB/s where FP16
        // reads 7.27 at 662, but it cannot say which of the four candidates
        // the perf audit lists holds the ~2 ms between 3.78 and 5.805.
        //
        // Two instruments, and their costs are declared:
        //
        //   * the floor probes, timed exactly like the two arms — same rounds,
        //     same warmup, same wall-clock-around-a-pass. Their *differences*
        //     are the attribution; their absolute values are not baselines.
        //   * per-matrix CUDA events, which add a `cuEventRecord` between
        //     consecutive launches. That is not free, so these rounds are
        //     separate and their totals are NOT the published milliseconds.
        println!("\n\n================= ATTRIBUTION (hors protocole publié) =================");

        let floors = [
            ("sol — bases + 5 mots", cuda.func("tv_floor_stream")?),
            ("sol — + gather tab", cuda.func("tv_floor_tab")?),
            ("sol — + 24 lectures xs", cuda.func("tv_floor_xs")?),
        ];
        let run_floor = |f: &cudarc::driver::CudaFunction,
                         m: &Mat,
                         y: &mut cudarc::driver::CudaSlice<f32>|
         -> Result<(), String> {
            cuda.launch_floor(
                f, &m.words, &m.bases, &d_tab, &d_x, y, m.nblocks as u32, m.d_out as u32,
                THREADS, shared,
            )
        };

        // `t_submit` is read on the same passes: the CPU time of the launch
        // loop alone, before the synchronize. If it approaches the wall clock,
        // the card is starved and every kernel-level number is a red herring.
        let mut submit_slot = Vec::new();
        let mut floor_ms: Vec<Vec<f64>> = vec![Vec::new(); floors.len()];
        for rep in 0..ROUNDS {
            let t = Instant::now();
            for m in &mats {
                run_slot(m, &mut d_y)?;
            }
            let sub = t.elapsed().as_secs_f64();
            cuda.sync()?;
            if rep >= WARMUP {
                submit_slot.push(sub);
            }
            for (fi, (_, f)) in floors.iter().enumerate() {
                let t = Instant::now();
                for m in &mats {
                    run_floor(f, m, &mut d_y)?;
                }
                cuda.sync()?;
                let s = t.elapsed().as_secs_f64();
                if rep >= WARMUP {
                    floor_ms[fi].push(s);
                }
            }
        }

        let (_, sub_md, _) = spread(submit_slot.clone());
        println!(
            "\n  t_submit (LLVQ) : {:.3} ms de CPU pour {} lancements, soit {:.1} % du mur\n  \
             et {:.2} µs par lancement — le gap `g` que le dossier laissait dans [1 ; 4] µs.",
            sub_md * 1e3,
            mats.len(),
            100.0 * sub_md / smd,
            sub_md * 1e6 / mats.len() as f64
        );

        println!("\n  Décomposition — chaque étage ajoute UNE chose que `tv_slot` fait");
        println!("  {}", "-".repeat(72));
        println!("  {:<26}{:>10}{:>12}{:>12}", "étage", "méd ms", "Δ ms", "part des ~2 ms");
        println!("  {}", "-".repeat(72));
        let dram_floor = sb as f64 / (fb as f64 / flo) * 1e3; // 2.50 GB at the FP16 arm's rate
        let gisement = smd * 1e3 - dram_floor;
        let mut prev = dram_floor;
        println!("  {:<26}{:>10.3}{:>12}{:>12}", "plancher DRAM (calculé)", dram_floor, "—", "—");
        for (fi, (name, _)) in floors.iter().enumerate() {
            let (_, md, _) = spread(floor_ms[fi].clone());
            let md = md * 1e3;
            println!(
                "  {name:<26}{md:>10.3}{:>12.3}{:>11.0} %",
                md - prev,
                100.0 * (md - prev) / gisement
            );
            prev = md;
        }
        let smd_ms = smd * 1e3;
        println!(
            "  {:<26}{smd_ms:>10.3}{:>12.3}{:>11.0} %",
            "tv_slot (décodage)",
            smd_ms - prev,
            100.0 * (smd_ms - prev) / gisement
        );
        println!("  {}", "-".repeat(72));
        println!(
            "  gisement total {gisement:.3} ms au-dessus du plancher DRAM.\n  \
             ⚠️ les sols ne calculent AUCUN produit : ce sont des diagnostics, jamais\n  \
             des références. Aucun rapport ne doit être cité contre eux."
        );

        // ---- per-shape timing, by CUDA events ----
        //
        // One event after each launch; matrix i costs `elapsed(ev[i], ev[i+1])`.
        // Half the records of a start/stop pair, and the same information.
        let evs: Vec<cudarc::driver::CudaEvent> = (0..=mats.len())
            .map(|_| cuda.new_event())
            .collect::<Result<_, String>>()?;
        let mut shape_us: Vec<[f64; 2]> = vec![[0.0; 2]; mats.len()];
        let ev_rounds = 3usize;
        for rep in 0..(ev_rounds + 1) {
            // `[0, 1]` rather than `0..2`: `arm` indexes the *second* axis of
            // `shape_us`, which clippy's `needless_range_loop` reads as the
            // first and suggests iterating away.
            for arm in [0usize, 1] {
                evs[0].record(cuda.stream()).map_err(|e| format!("event: {e}"))?;
                for (i, m) in mats.iter().enumerate() {
                    if arm == 0 {
                        run_slot(m, &mut d_y)?;
                    } else {
                        run_f16(m, &mut d_y)?;
                    }
                    evs[i + 1].record(cuda.stream()).map_err(|e| format!("event: {e}"))?;
                }
                cuda.sync()?;
                if rep == 0 {
                    continue; // warmup
                }
                for i in 0..mats.len() {
                    let dt = evs[i]
                        .elapsed_ms(&evs[i + 1])
                        .map_err(|e| format!("elapsed: {e}"))?;
                    shape_us[i][arm] += dt as f64 * 1e3 / ev_rounds as f64;
                }
            }
        }

        // Aggregate by shape. Order of first appearance, which is the order of
        // the model — so a reader can line the table up against a block.
        let mut shapes: Vec<(usize, usize)> = Vec::new();
        for m in &mats {
            if !shapes.contains(&(m.d_out, m.d_in)) {
                shapes.push((m.d_out, m.d_in));
            }
        }
        println!("\n  Temps par forme — {ev_rounds} rounds, events CUDA");
        println!(
            "  ⚠️ un `cuEventRecord` entre deux lancements n'est pas gratuit : ces\n  \
             millisecondes ne sont PAS celles du tableau publié. Les lire en relatif."
        );
        println!("  {}", "-".repeat(86));
        println!(
            "  {:<8}{:>7}{:>7}{:>6}{:>11}{:>11}{:>10}{:>10}{:>9}",
            "forme", "d_out", "d_in", "n", "LLVQ µs", "FP16 µs", "LLVQ Go/s", "FP16 Go/s", "blocs"
        );
        println!("  {}", "-".repeat(86));
        for (d_out, d_in) in &shapes {
            let idx: Vec<usize> = mats
                .iter()
                .enumerate()
                .filter(|(_, m)| m.d_out == *d_out && m.d_in == *d_in)
                .map(|(i, _)| i)
                .collect();
            let n = idx.len();
            let us_q: f64 = idx.iter().map(|&i| shape_us[i][0]).sum::<f64>() / n as f64;
            let us_f: f64 = idx.iter().map(|&i| shape_us[i][1]).sum::<f64>() / n as f64;
            let sbytes = mats[idx[0]].slot_bytes as f64;
            let fbytes = mats[idx[0]].f16_bytes as f64;
            // Blocks resident: the grid this shape launches, against the
            // card's capacity. Underfill is the audit's third candidate and
            // this column is what confirms or kills it.
            let blocks = *d_out as u32 * 32 / THREADS;
            println!(
                "  {:<8}{d_out:>7}{d_in:>7}{n:>6}{us_q:>11.1}{us_f:>11.1}{:>10.0}{:>10.0}{blocks:>9}",
                "",
                sbytes / (us_q * 1e-6) / 1e9,
                fbytes / (us_f * 1e-6) / 1e9
            );
        }
        println!("  {}", "-".repeat(86));
        println!(
            "  capacité ≈ {} blocs résidents ({} SM × 6). Une forme très en dessous\n  \
             et lente désigne le sous-remplissage ; toutes à ~430 Go/s désigne le\n  \
             motif d'accès. C'est la lecture qui réécrit l'ordre de K7.",
            dev.sm_count * 6,
            dev.sm_count
        );

        // ---- fusion: correctness first, then cost ----
        if !fused.is_empty() {
            let f_seg = cuda.func("tv_slot_seg")?;
            let run_seg = |f: &Fused,
                           y: &mut cudarc::driver::CudaSlice<f32>|
             -> Result<(), String> {
                cuda.launch_slot_seg(
                    &f_seg, &f.words, &f.bases, &d_tab, &f.gscale, &f.gs_off, &f.rscale,
                    &f.tail, &d_x, y, f.nblocks as u32, f.tail_w as u32, f.d_out as u32,
                    THREADS, shared,
                )
            };

            // The fused row `r` runs the same blocks, in the same order, with
            // the same centroids as the unfused row it came from — nothing is
            // reassociated. So the outputs must agree BIT FOR BIT, and a
            // tolerance here would let a wrong `gs_off` through: swapping two
            // segments' centroids moves a result by ~2×, but an epsilon-based
            // check on a different row would still pass.
            println!("\n  Fusion — sortie contre les matrices non fusionnées");
            for f in &fused {
                run_seg(f, &mut d_y)?;
                cuda.sync()?;
                let got = cuda.down_f32(&d_y)?;
                let mut at = 0usize;
                for &pi in &f.parts {
                    let m = &mats[pi];
                    run_slot(m, &mut d_y)?;
                    cuda.sync()?;
                    let want = cuda.down_f32(&d_y)?;
                    if got[at..at + m.d_out] != want[..m.d_out] {
                        let bad = (0..m.d_out)
                            .find(|&r| got[at + r] != want[r])
                            .unwrap_or(0);
                        return Err(format!(
                            "{} / {} : ligne {bad} vaut {} fusionnée contre {} séparée",
                            f.name, m.name, got[at + bad], want[bad]
                        ));
                    }
                    at += m.d_out;
                }
            }
            println!(
                "  {} groupes, {} lignes — identiques AU BIT près",
                fused.len(),
                fused.iter().map(|f| f.d_out).sum::<usize>()
            );

            // Cost. The comparison is LLVQ against LLVQ: the same 252
            // matrices' worth of weights, launched as 252 kernels or as 144.
            //
            // ⚠️ No ratio against FP16 is formed here, and that is deliberate.
            // Fusing the FP16 arm too would mean rebuilding and holding a
            // second copy of 7.27 GB of f16 weights; until that is done, a
            // fused-LLVQ / unfused-FP16 ratio would credit the format for a
            // geometry change. The number below is a delta, not a ratio.
            let mut t_fused = Vec::new();
            let mut t_ref = Vec::new();
            for rep in 0..ROUNDS {
                for arm in 0..2 {
                    let t = Instant::now();
                    if arm == 0 {
                        for m in &mats {
                            run_slot(m, &mut d_y)?;
                        }
                    } else {
                        // The fusible matrices, fused; everything else — o_proj
                        // and down_proj — exactly as before.
                        for f in &fused {
                            run_seg(f, &mut d_y)?;
                        }
                        for (i, m) in mats.iter().enumerate() {
                            if !fused.iter().any(|f| f.parts.contains(&i)) {
                                run_slot(m, &mut d_y)?;
                            }
                        }
                    }
                    cuda.sync()?;
                    let s = t.elapsed().as_secs_f64();
                    if rep >= WARMUP {
                        if arm == 0 {
                            t_ref.push(s)
                        } else {
                            t_fused.push(s)
                        }
                    }
                }
            }
            let (_, rmd_ms, _) = spread(t_ref.clone());
            let (_, fmd_ms, _) = spread(t_fused.clone());
            let n_launch = fused.len()
                + mats
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !fused.iter().any(|f| f.parts.contains(i)))
                    .count();
            println!(
                "\n  Coût — {ROUNDS} rounds, {WARMUP} jetés, bras entrelacés\n  \
                 {:<28}{:>10.3} ms   {} lancements\n  \
                 {:<28}{:>10.3} ms   {} lancements\n  \
                 gain {:.3} ms ({:.1} %), soit {:.0} % du gisement de 2,039 ms",
                "LLVQ, matrices séparées",
                rmd_ms * 1e3,
                mats.len(),
                "LLVQ, q+k+v et gate+up fusés",
                fmd_ms * 1e3,
                n_launch,
                (rmd_ms - fmd_ms) * 1e3,
                100.0 * (rmd_ms - fmd_ms) / rmd_ms,
                100.0 * (rmd_ms - fmd_ms) * 1e3 / 2.039
            );
            let fb_slot: u64 = fused.iter().map(|f| f.slot_bytes).sum::<u64>()
                + mats
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !fused.iter().any(|f| f.parts.contains(i)))
                    .map(|(_, m)| m.slot_bytes)
                    .sum::<u64>();
            println!(
                "  octets lus : {:.3} Go fusé contre {:.3} Go séparé ({:+.2} %) — la\n  \
                 fusion re-transcode le flux, donc les strides de groupe changent aux\n  \
                 frontières de segment. Un gain d'octets serait un confondant, pas un\n  \
                 bonus : il est imprimé pour pouvoir le retrancher.",
                fb_slot as f64 / 1e9,
                sb as f64 / 1e9,
                100.0 * (fb_slot as f64 - sb as f64) / sb as f64
            );
        }
        Ok(())
    }
}
