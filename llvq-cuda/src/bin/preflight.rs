//! The first thing this port asks of a rented card, and the cheapest.
//!
//! It answers, in one job of a few minutes, every question
//! `docs/archive/portage-noyau-cuda.md` §6.2 files under "answered at the first job,
//! for a few cents" — and one the document does not ask, which is the only
//! one that matters: **does the ported decoder decide the same thing as the
//! Rust decoder, on a real GPU?**
//!
//! ## Why it needs no artifact
//!
//! The 4B file is 981 MB and a job is billed by the minute, so downloading it
//! to check a decoder would spend more on transfer than on measurement. The
//! fixture is built in-process from `FastDecoder`: both ends of **every one of
//! the 384 classes**, plus twenty thousand uniform draws over the whole cap-13
//! ball, plus the origin mid-stream. That is *harder* than the sealed file,
//! which is capped at Λ₂₄(12): the cap-13 ball contains the 5-level classes
//! whose 130-bit records are the worst case for the kernel's five-word read.
//!
//! ## What it compares, and why in integers
//!
//! `slot_dot` returns a float, and a float compared across two compilers can
//! only ever be "close". The **level index** of each slot and its **sign** are
//! exact, and they are where the mask arithmetic lives — a flipped sign or an
//! off-by-one mask is a plain integer difference, not a suspicious residue.
//! The dot product is checked too, but second, and against an f64 reference.
//!
//! One exception, and it is documented rather than papered over: for a block
//! whose class is the origin the record stops after its 10-bit header, so the
//! 24 bits the kernel reads as `smask` belong to *the following blocks*. This
//! is harmless — `len = 1` forces every level to 0, every value to 0.0f, and
//! the sign only ever produces −0.0f — but it means the raw sign bit of a zero
//! slot is meaningless. Signs are therefore compared only where the decoded
//! value is nonzero, which is the decoder's actual semantics.

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("preflight targets NVIDIA GPUs; there is nothing to run here.");
}

#[cfg(target_os = "linux")]
fn main() -> Result<(), String> {
    linux::run()
}

#[cfg(target_os = "linux")]
mod linux {
    use llvq_artifact::runtime::{transcode, ClassTable, Layout};
    use llvq_core::{SplitMix64, DIM};
    use llvq_cuda::gpu::{launch_decode_probe, launch_dot_probe, Cuda, Inputs, KernelSource};
    use llvq_search::fastdec::FastDecoder;
    use llvq_search::index::N13;

    /// Gain centroids. Arbitrary, but far apart and neither 0 nor 1: a
    /// misread gain bit must change the answer, and a value of 1 would hide
    /// one behind a no-op multiply.
    const GSCALE: [f32; 2] = [0.625, 1.375];

    /// Entries in the device class table. The class field is 9 bits, so 512
    /// values are addressable while only 384 exist. Allocating the full 512
    /// and leaving the tail as the origin costs 3 KB and turns a truncated or
    /// corrupt stream from an illegal address — which kills the context, in
    /// the middle of a billed job — into a block of zeros.
    const TABLE_ENTRIES: usize = 512;
    /// `float vals[5]; unsigned len;`
    const REC_WORDS: usize = 6;

    pub fn run() -> Result<(), String> {
        let sources = llvq_cuda::load_sources()?;
        let src = KernelSource::new(&[&sources.slot, &sources.cu]);
        println!("source NVRTC : {} octets, sha256 {}", src.text.len(), src.sha256);
        match &sources.overridden_from {
            None => println!("  sources embarquées (celles du commit)"),
            Some(d) => println!(
                "  ⚠️ SOURCES SURCHARGÉES depuis {d} — tout chiffre issu de ce run se \n                   rattache au sha256 ci-dessus et à rien d'autre."
            ),
        }

        let cuda = Cuda::new(&src)?;
        let dev = cuda.device()?;
        println!("\ncarte");
        println!("  {}", "-".repeat(70));
        println!("  nom                    {}", dev.name);
        println!(
            "  compute capability     {}.{}",
            dev.compute_cap.0, dev.compute_cap.1
        );
        println!("  SM                     {}", dev.sm_count);
        println!(
            "  L2                     {} octets ({:.1} Mo) — LU, jamais supposé",
            dev.l2_bytes,
            dev.l2_bytes as f64 / 1e6
        );
        println!(
            "  mémoire partagée       {} o/bloc, {} o/SM",
            dev.shared_per_block, dev.shared_per_sm
        );
        println!(
            "  bande passante fiche   {:.0} Go/s ({} bits à {:.0} MHz, DDR)",
            dev.nameplate_bandwidth_gbs(),
            dev.mem_bus_bits,
            dev.mem_clock_khz as f64 / 1e3
        );
        println!(
            "  ⚠️ ce dernier chiffre est une plaque signalétique, pas une mesure : l'ECC est\n  \
             active par défaut et en consomme une part. Tout « x % du pic » doit passer par le\n  \
             noyau sol, jamais par lui."
        );

        println!("\nnoyaux chargés");
        println!("  {}", "-".repeat(70));
        println!(
            "  {:<16}{:>8}{:>10}{:>10}{:>10}{:>10}",
            "noyau", "regs", "local o", "shared o", "sm_", "thr/bloc"
        );
        for name in ["decode_probe", "dot_probe", "floor_probe"] {
            let r = cuda.report(name)?;
            println!(
                "  {:<16}{:>8}{:>10}{:>10}{:>10}{:>10}",
                r.name, r.num_regs, r.local_bytes, r.shared_bytes, r.binary_version, r.max_threads
            );
            if r.local_bytes != 0 {
                return Err(format!(
                    "{name} : {} octets de mémoire locale. `slot_dot` tient ses quatre chaînes \
                     FMA dans d0..d3 ; si le compilateur n'a pas déroulé, elles deviennent un \
                     tableau indexé dynamiquement, donc de la mémoire locale, sur le chemin le \
                     plus chaud. Aucun test de justesse ne peut voir ça.",
                    r.local_bytes
                ));
            }
        }

        // ---- the fixture ----
        let fd = FastDecoder::new();
        let table = ClassTable::new(&fd, 1);
        assert!(
            24 + table.worst_width_slot() <= 160,
            "la table déborde la fenêtre de 5 mots du noyau"
        );
        assert!(
            fd.n_classes() < TABLE_ENTRIES,
            "{} classes plus l'origine ne tiennent pas dans {TABLE_ENTRIES} entrées",
            fd.n_classes()
        );

        let mut indices: Vec<u64> = Vec::new();
        for ci in 0..fd.n_classes() {
            let (first, last) = fd.class_range(ci);
            indices.push(first);
            indices.push(last);
        }
        let mut rng = SplitMix64::new(0x6_C0DE);
        indices.extend((0..20_000).map(|_| 1 + rng.next() % N13));
        indices.push(0); // the origin, mid-stream
        let gains: Vec<u32> = indices.iter().map(|_| (rng.next() & 1) as u32).collect();
        let n = indices.len();

        let rt = transcode(&fd, &table, &indices, &gains, Layout::Slot32)
            .map_err(|e| e.to_string())?;
        println!(
            "\nfixture : {n} blocs — les deux bornes de chacune des {} classes, {} tirages \
             uniformes\n  sur la boule m ≤ 13, plus l'origine. Plus large que le fichier scellé, \
             qui est plafonné à Λ₂₄(12).",
            fd.n_classes(),
            20_000
        );

        // The five-word read of the last block runs past the stream. Benign on
        // Metal; here it is an illegal address that kills the context.
        let mut bytes = rt.data.clone();
        bytes.extend_from_slice(&[0u8; 20]);
        while !bytes.len().is_multiple_of(4) {
            bytes.push(0);
        }
        let words: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let mut tab = vec![0u32; TABLE_ENTRIES * REC_WORDS];
        tab[REC_WORDS - 1] = 1; // entry 0: the origin, len = 1, values 0
        for e in 1..TABLE_ENTRIES {
            tab[e * REC_WORDS + REC_WORDS - 1] = 1; // unreachable entries: origin
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

        let x: Vec<f32> = (0..n * DIM).map(|_| rng.next_gaussian() as f32).collect();

        let d_words = cuda.up_u32(&words)?;
        let d_bases = cuda.up_u32(&rt.bases)?;
        let d_tab = cuda.up_u32(&tab)?;
        let d_gscale = cuda.up_f32(&GSCALE)?;
        let d_x = cuda.up_f32(&x)?;
        let mut d_slots = cuda.zeros_u32(n * DIM)?;
        let mut d_cls = cuda.zeros_u32(n * 2)?;
        let mut d_dot = cuda.zeros_f32(n)?;

        let f_decode = cuda.func("decode_probe")?;
        let f_dot = cuda.func("dot_probe")?;
        let inp = Inputs {
            words: &d_words,
            bases: &d_bases,
            tab: &d_tab,
            gscale: &d_gscale,
            x: &d_x,
            nblocks: n as u32,
        };
        launch_decode_probe(&cuda, &f_decode, &inp, &mut d_slots, &mut d_cls)?;
        launch_dot_probe(&cuda, &f_dot, &inp, &mut d_dot)?;
        cuda.sync()?;
        let got_slots = cuda.down_u32(&d_slots)?;
        let got_cls = cuda.down_u32(&d_cls)?;
        let got_dot = cuda.down_f32(&d_dot)?;

        // ---- the comparison ----
        let mut worst_dot = 0.0f64;
        let mut n_origin = 0usize;
        for b in 0..n {
            let (pt, gain) = rt.decode_block(&table, b);
            let idx = indices[b];
            let id = match fd.class_of(idx) {
                Some(ci) => 1 + ci,
                None => {
                    assert_eq!(idx, 0, "bloc {b} : index {idx} hors de la boule");
                    n_origin += 1;
                    0
                }
            };
            if got_cls[b * 2] as usize != id {
                return Err(format!(
                    "bloc {b} : le GPU lit la classe {} là où l'hôte écrit {id}. \
                     L'en-tête de 10 bits est mal lu — boutisme, ou décalage d'alignement.",
                    got_cls[b * 2]
                ));
            }
            if got_cls[b * 2 + 1] != gain {
                return Err(format!(
                    "bloc {b} : bit de gain {} contre {gain} attendu",
                    got_cls[b * 2 + 1]
                ));
            }

            let lv = (id > 0).then(|| fd.levels(id - 1));
            let mut want_dot = 0.0f64;
            let mut scale = 0.0f64;
            for j in 0..DIM {
                let v = pt[j];
                let lev = match lv {
                    None => 0usize,
                    Some(lv) => lv.values[..lv.len]
                        .iter()
                        .position(|&u| u == v.abs())
                        .ok_or_else(|| {
                            format!("bloc {b} slot {j} : |{v}| absent des niveaux de la classe")
                        })?,
                };
                let g = got_slots[b * DIM + j];
                if (g & 7) as usize != lev {
                    return Err(format!(
                        "bloc {b} slot {j} : niveau {} contre {lev} attendu (classe {id}). \
                         C'est l'arithmétique des masques qui diverge, pas un arrondi.",
                        g & 7
                    ));
                }
                // Only where the value is nonzero: a zero slot's sign bit is
                // meaningless, and for the origin it is not even part of the
                // record — see the module header.
                if v != 0 {
                    let want_sign = u32::from(v < 0);
                    if (g >> 3) & 1 != want_sign {
                        return Err(format!(
                            "bloc {b} slot {j} : signe {} contre {want_sign} attendu",
                            (g >> 3) & 1
                        ));
                    }
                }
                let lvv = lv.map_or(0.0, |lv| {
                    lv.values[lev].abs() as f64 / ((16 * lv.shell) as f64).sqrt()
                });
                let w = if v < 0 { -lvv } else { lvv } * GSCALE[gain as usize] as f64;
                want_dot += w * x[b * DIM + j] as f64;
                scale += (w * x[b * DIM + j] as f64).abs();
            }
            let e = (got_dot[b] as f64 - want_dot).abs() / scale.max(1e-12);
            worst_dot = worst_dot.max(e);
        }

        println!("\nvérification");
        println!("  {}", "-".repeat(70));
        println!("  {n} blocs, {} slots — classe, gain et niveau EXACTS", n * DIM);
        println!("  {n_origin} bloc origine, dont le champ de signes n'existe pas");
        println!("  produit scalaire : pire erreur {worst_dot:.2e}·Σ|w·x|");
        if worst_dot > 1e-6 {
            return Err(format!(
                "pire erreur {worst_dot:.2e} au-dessus de 1e-6. Le décodage est juste \
                 (les niveaux le sont), donc c'est l'arithmétique flottante qui diverge."
            ));
        }
        println!("\n✓ le décodeur porté décide exactement ce que décide le décodeur Rust.");
        Ok(())
    }
}
