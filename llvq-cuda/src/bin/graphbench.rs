//! Ce que coûte un lancement, et ce qu'un CUDA Graph en rend.
//!
//! L'audit de perf du 2026-08-05 fait de ce chiffre une **condition de
//! publication** : tout gain de noyau mesuré à travers 252 lancements est
//! dilué par le terme constant `252·g`, et `g` doit être connu avant tout
//! balayage. `bin/matvec` en a déjà donné une borne — `t_submit` = 1,85 µs de
//! CPU par lancement, 8 % du mur — mais la soumission n'est pas l'exécution :
//! le GPU paie aussi un coût par lancement que le CPU ne voit pas.
//!
//! ## Pourquoi un noyau minuscule
//!
//! On mesure le **lancement**, pas le trafic. Une matrice de 64 lignes lance
//! 8 blocs et lit quelques kilo-octets : le temps par itération est alors
//! presque entièrement du coût fixe, ce qui est précisément la grandeur
//! cherchée. Sur les vraies formes, le travail masquerait ce qu'on veut isoler.
//!
//! Les tampons sont des zéros. Ce n'est pas un raccourci : `bases` nul donne
//! un stride nul, donc tous les blocs lisent l'entrée 0 de la table — la
//! classe origine, parfaitement valide (`len = 1`, valeurs nulles). Le noyau
//! exécute son chemin complet sur des données licites.
//!
//! ## Les trois bras, et pourquoi il en faut trois
//!
//! `Cuda::new` prend `ctx.default_stream()` — le stream NULL legacy, que le
//! driver **refuse de capturer**. Un bras graph impose donc un stream frais,
//! et changer de stream change l'objet mesuré : le NULL stream porte une
//! synchronisation implicite contre tous les autres streams du contexte, pas
//! un stream frais. Comparer « graph sur stream frais » à un chiffre publié
//! pris sur le legacy créditerait le graph de ce que vaut le changement de
//! stream. D'où : legacy · frais · frais + graph, dans le même job.

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("graphbench targets NVIDIA GPUs; there is nothing to run here.");
}

#[cfg(target_os = "linux")]
fn main() -> Result<(), String> {
    linux::run()
}

#[cfg(target_os = "linux")]
mod linux {
    use llvq_cuda::gpu::{Cuda, KernelSource};
    use std::time::Instant;

    const TILE_BLOCKS: usize = 128;
    const THREADS: u32 = 256;
    const TABLE_ENTRIES: usize = 512;
    const REC_WORDS: usize = 6;
    /// Les 252 projections d'un token de Qwen3-4B — le compte qui donne son
    /// sens à `252·g`.
    const LAUNCHES: usize = 252;
    const ROUNDS: usize = 20;
    const WARMUP: usize = 5;
    /// 64 lignes = 8 blocs : assez pour que la grille soit légale, assez peu
    /// pour que le travail ne masque pas le lancement.
    const D_OUT: usize = 64;
    const D_IN: usize = 2560;

    fn spread(mut v: Vec<f64>) -> (f64, f64, f64) {
        v.sort_by(f64::total_cmp);
        (v[0], v[v.len() / 2], v[v.len() - 1])
    }

    pub fn run() -> Result<(), String> {
        let sources =
            llvq_cuda::load_sources_many(&["llvq_slot.cuh", "matvec.cu", "llvq_floor.cuh"])?;
        let defines = format!("#define TILE_BLOCKS {TILE_BLOCKS}u\n");
        let parts: Vec<&str> = std::iter::once(defines.as_str())
            .chain(sources.parts.iter().map(String::as_str))
            .collect();
        let src = KernelSource::new(&parts);
        println!("source NVRTC : {} octets, sha256 {}", src.text.len(), src.sha256);

        let nblocks = D_IN / 24;
        let mut results: Vec<(&str, f64, f64, f64)> = Vec::new();

        for (label, fresh) in [("stream legacy (défaut)", false), ("stream frais", true)] {
            let cuda = if fresh {
                Cuda::new_on_fresh_stream(&src)?
            } else {
                Cuda::new(&src)?
            };
            let dev = cuda.device()?;
            if !fresh {
                println!("\n{} — {} SM", dev.name, dev.sm_count);
            }
            let f = cuda.func("tv_floor_stream")?;

            let words = cuda.zeros_u32(1 << 16)?;
            let bases = cuda.zeros_u32(D_OUT * nblocks / 32 + 2)?;
            let tab = cuda.up_u32(&vec![1u32; TABLE_ENTRIES * REC_WORDS])?;
            let x = cuda.zeros_f32(D_IN)?;
            let mut y = cuda.zeros_f32(D_OUT)?;
            let shared = (TILE_BLOCKS * 24 * 4) as u32;

            let one = |y: &mut cudarc::driver::CudaSlice<f32>| -> Result<(), String> {
                cuda.launch_floor(
                    &f, &words, &bases, &tab, &x, y, nblocks as u32, D_OUT as u32, THREADS,
                    shared,
                )
            };

            // --- sans graph ---
            let mut us = Vec::new();
            for r in 0..ROUNDS {
                let t = Instant::now();
                for _ in 0..LAUNCHES {
                    one(&mut y)?;
                }
                cuda.sync()?;
                if r >= WARMUP {
                    us.push(t.elapsed().as_secs_f64() * 1e6 / LAUNCHES as f64);
                }
            }
            let (lo, md, hi) = spread(us);
            println!("  {label:<26} {md:7.2} µs/lancement  [{lo:.2}–{hi:.2}]");
            results.push((if fresh { "frais" } else { "legacy" }, lo, md, hi));

            // --- avec graph, seulement là où le driver l'accepte ---
            if fresh {
                let graph = cuda.capture(|| {
                    for _ in 0..LAUNCHES {
                        one(&mut y)?;
                    }
                    Ok(())
                })?;
                let mut us = Vec::new();
                for r in 0..ROUNDS {
                    let t = Instant::now();
                    graph.launch().map_err(|e| format!("graph launch: {e}"))?;
                    cuda.sync()?;
                    if r >= WARMUP {
                        us.push(t.elapsed().as_secs_f64() * 1e6 / LAUNCHES as f64);
                    }
                }
                let (lo, md, hi) = spread(us);
                println!("  {:<26} {md:7.2} µs/lancement  [{lo:.2}–{hi:.2}]", "frais + graph");
                results.push(("graph", lo, md, hi));
            }
        }

        // --- ce que ça vaut sur un token ---
        let get = |k: &str| results.iter().find(|(n, ..)| *n == k).map(|(_, _, m, _)| *m);
        if let (Some(legacy), Some(fresh), Some(graph)) = (get("legacy"), get("frais"), get("graph"))
        {
            println!("\n  Sur {LAUNCHES} lancements — ce que vaut un token de projections :");
            println!("    legacy         {:6.3} ms", legacy * LAUNCHES as f64 / 1e3);
            println!("    frais          {:6.3} ms", fresh * LAUNCHES as f64 / 1e3);
            println!("    frais + graph  {:6.3} ms", graph * LAUNCHES as f64 / 1e3);
            println!(
                "\n  Le graph rend {:.3} ms contre le stream frais, {:.3} contre le legacy.",
                (fresh - graph) * LAUNCHES as f64 / 1e3,
                (legacy - graph) * LAUNCHES as f64 / 1e3
            );
            println!(
                "  ⚠️ PLAFOND, pas un gain acquis : ce noyau ne fait presque rien, donc son\n  \
                 temps est presque tout du lancement. Sur `tv_slot`, dont le travail masque\n  \
                 une partie du coût fixe, l'économie réelle est plus faible — et le décode\n  \
                 complet a un cache KV à formes variables qu'un graph statique ne capture pas\n  \
                 (cf. A3(b) de la spec du lot A)."
            );
        }
        Ok(())
    }
}
