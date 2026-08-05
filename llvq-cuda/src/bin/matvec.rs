//! The fused kernel against its FP16 witness, on NVIDIA.
//!
//! One token's worth of linear algebra over the seven projection shapes of
//! Qwen3-4B, repeated until a pass exceeds the card's L2 — which the card
//! itself reports, and which turned out to be 100.7 MB, not the 48 or 96 that
//! third-party sources give. Getting that wrong is the trap this repository
//! already fell into once: 11-17 MB buffers replayed 576 times inside a 48 MB
//! system cache made every earlier LLVQ figure optimistic.
//!
//! ## Why synthetic blocks, and what it costs
//!
//! The indices are uniform draws over the cap-13 ball rather than a real
//! quantization of real weights. That is honest for a *kernel* measurement —
//! the decode path, the five-word read, the group strides and the tiling are
//! all exercised identically — but it is **not** the published model, so the
//! ratio here is not the headline figure. The class mix of a real artifact is
//! not uniform (65.85 % of its blocks have 4 levels), so its group strides,
//! and therefore its byte traffic, differ. Wiring the sealed file in is the
//! next step and it changes only the input, not this code.
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

    struct Mat {
        name: String,
        d_out: usize,
        d_in: usize,
        nblocks: usize,
        tail_w: usize,
        words: cudarc::driver::CudaSlice<u32>,
        bases: cudarc::driver::CudaSlice<u32>,
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
        let sources = llvq_cuda::load_sources_named(&["llvq_slot.cuh", "matvec.cu"])?;
        let defines = format!("#define TILE_BLOCKS {TILE_BLOCKS}u\n");
        let src = KernelSource::new(&[&defines, &sources.slot, &sources.cu]);
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
        for name in ["tv_slot", "tv_f16"] {
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
        println!(
            "\n{reps} répétitions des 7 formes : {:.0} Mo de poids LLVQ distincts par passe,\n\
             soit {:.1}× la L2. En dessous de 1× on mesurerait le cache, pas la DRAM —\n\
             le piège qui a rendu optimiste toute mesure LLVQ antérieure au 2026-07-31.",
            (one_pass * reps as u64) as f64 / 1e6,
            (one_pass * reps as u64) as f64 / dev.l2_bytes as f64
        );

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
        let d_gscale = cuda.up_f32(&GSCALE)?;

        let mut rng = SplitMix64::new(0x6_D07);
        let max_din = SHAPES.iter().map(|&(_, _, i)| i).max().unwrap();
        let x: Vec<f32> = (0..max_din).map(|_| rng.next_gaussian() as f32).collect();
        let d_x = cuda.up_f32(&x)?;
        let max_dout = SHAPES.iter().map(|&(_, o, _)| o).max().unwrap();
        let mut d_y = cuda.zeros_f32(max_dout)?;

        println!("\nConstruction et transcodage…");
        let t0 = Instant::now();
        let mut mats = Vec::new();
        let mut n_weights = 0u64;
        for r in 0..reps {
            for &(name, d_out, d_in) in &SHAPES {
                let nblocks = d_in / DIM;
                let tail_w = d_in % DIM;
                assert_eq!(d_out % 8, 0, "{name}: CUDA lance des blocs entiers");
                let n = d_out * nblocks;
                let indices: Vec<u64> = (0..n).map(|_| 1 + rng.next() % N13).collect();
                let gains: Vec<u32> = (0..n).map(|_| (rng.next() & 1) as u32).collect();
                let rt = transcode(&fd, &table, &indices, &gains, Layout::Slot32)
                    .map_err(|e| e.to_string())?;

                let mut w16 = vec![0u16; d_out * d_in];
                let mut y_ref = vec![0.0f64; d_out];
                let mut y16_ref = vec![0.0f64; d_out];
                let mut scale = vec![0.0f64; d_out];
                let rscale: Vec<f32> =
                    (0..d_out).map(|_| 0.5 + rng.next_gaussian().abs() as f32).collect();
                let tailf: Vec<f32> =
                    (0..d_out * tail_w).map(|_| rng.next_gaussian() as f32).collect();

                let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
                let chunk = d_out.div_ceil(threads);
                std::thread::scope(|s| {
                    for (ci, (((w16c, yc), y16c), sc)) in w16
                        .chunks_mut(chunk * d_in)
                        .zip(y_ref.chunks_mut(chunk))
                        .zip(y16_ref.chunks_mut(chunk))
                        .zip(scale.chunks_mut(chunk))
                        .enumerate()
                    {
                        let (rt, table, x, rscale, tailf) = (&rt, &table, &x, &rscale, &tailf);
                        s.spawn(move || {
                            let mut wrow = vec![0.0f64; d_in];
                            for lr in 0..yc.len() {
                                let row = ci * chunk + lr;
                                wrow.fill(0.0);
                                for p in 0..nblocks {
                                    let (pt, gain) = rt.decode_block(table, row * nblocks + p);
                                    let id = llvq_core::Leech::shell_index(&pt).filter(|&s| s > 0);
                                    if let Some(shell) = id {
                                        let s = GSCALE[gain as usize] as f64
                                            * rscale[row] as f64
                                            / ((16 * shell) as f64).sqrt();
                                        for (i, &v) in pt.iter().enumerate() {
                                            wrow[p * DIM + i] = v as f64 * s;
                                        }
                                    }
                                }
                                for t in 0..tail_w {
                                    wrow[nblocks * DIM + t] = tailf[row * tail_w + t] as f64;
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
                                sc[lr] = ss;
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
                mats.push(Mat {
                    name: format!("{name}#{r}"),
                    d_out,
                    d_in,
                    nblocks,
                    tail_w,
                    words: cuda.up_u32(&words)?,
                    bases: cuda.up_u32(&rt.bases)?,
                    rscale: cuda.up_f32(&rscale)?,
                    tail: cuda.up_f32(if tailf.is_empty() { &[0.0f32] } else { &tailf })?,
                    w16: cuda.up_u16(&w16)?,
                    slot_bytes: rt.data.len() as u64
                        + rt.bases.len() as u64 * 4
                        + (d_out * tail_w) as u64 * 4
                        + d_out as u64 * 4,
                    f16_bytes: (d_out * d_in * 2) as u64,
                    y_ref,
                    y16_ref,
                    scale,
                });
                n_weights += (d_out * d_in) as u64;
            }
        }
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
                &f_slot, &m.words, &m.bases, &d_tab, &d_gscale, &m.rscale, &m.tail, &d_x, y,
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
        println!(
            "\n  ⚠️ blocs SYNTHÉTIQUES, tirés uniformément sur la boule m ≤ 13 : le chemin de\n  \
             décodage est exercé à l'identique, mais le mélange de classes d'un vrai artefact\n  \
             ne l'est pas, donc ses strides de groupe et son trafic d'octets diffèrent.\n  \
             Ce rapport mesure le NOYAU, ce n'est pas encore le chiffre du modèle publié.\n\
             \n  ⚠️ à ne JAMAIS comparer au chiffre Metal ligne à ligne : les deux réductions\n  \
             somment dans des ordres différents et les deux compilateurs contractent les FMA\n  \
             selon leurs propres règles. Chacun contre sa référence f64, deux pires erreurs."
        );
        Ok(())
    }
}
