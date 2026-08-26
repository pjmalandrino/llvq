//! How far the chosen direction sits from the block, per ball cap.
//!
//! Diagnostic only — no perplexity, no threshold, no verdict. It measures a
//! property of code that already exists, on weights that already exist.
//!
//! The confound it prices: the gain code quantizes `‖w‖`, but the magnitude
//! that minimises `‖w − a·û‖²` at a fixed direction `û` is the **projection**
//! `⟨w, û⟩ = ‖w‖·cos θ`. Coding the norm therefore overshoots by `1/cos θ`
//! on every block. That would be harmless if `cos θ` were the same for every
//! arm — but a coarser direction codebook means a larger θ, so the bias moves
//! **with the arm under test**. This binary says by how much.
//!
//! ```text
//! cargo run --release -p llvq-llm --features metal --bin cosdiag
//! ```

use llvq_core::DIM;
use llvq_quant::quantizer::row_scale;
use llvq_search::generic::BallSearcher;
use llvq_search::Searcher;

fn main() -> anyhow::Result<()> {
    let repo = std::env::var("LLVQ_MODEL").unwrap_or_else(|_| "Qwen/Qwen3-0.6B".into());
    let sample: usize = std::env::var("COSDIAG_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4000);

    let ck = llvq_llm::loader::Checkpoint::fetch(&repo)?;
    let vb = ck.var_builder(candle_core::DType::F32, &candle_core::Device::Cpu)?;
    let h = ck.config.hidden_size;
    let inter = ck.config.intermediate_size;

    let picks: Vec<(&str, usize, usize)> = vec![
        ("model.layers.0.mlp.gate_proj.weight", inter, h),
        ("model.layers.13.mlp.gate_proj.weight", inter, h),
        ("model.layers.27.mlp.down_proj.weight", h, inter),
    ];

    println!("modèle : {repo}   échantillon : {sample} blocs par (matrice, boule)");
    println!("cos θ = ⟨w, û⟩ / ‖w‖ — û étant la direction que la recherche angulaire retient.");
    println!("Le code de gain quantifie ‖w‖ ; l'optimum à direction fixée est ‖w‖·cos θ.");
    println!("Le surcoût systématique est donc 1/cos θ − 1, en pour-cent.\n");

    for (name, d_out, d_in) in picks {
        let t = vb.get((d_out, d_in), name)?;
        let w: Vec<f64> = t
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(f64::from)
            .collect();

        // Blocks in row order, normalised by their row scale exactly as the
        // quantizer sees them. Deterministic stride so every cap sees the
        // same blocks.
        let nb_row = d_in / DIM;
        let total = d_out * nb_row;
        let stride = (total / sample).max(1);
        let mut blocks: Vec<[f64; DIM]> = Vec::new();
        let mut idx = 0usize;
        for i in 0..d_out {
            let row = &w[i * d_in..(i + 1) * d_in];
            if row_scale(row) <= 0.0 {
                continue;
            }
            for b in row.chunks_exact(DIM) {
                if idx.is_multiple_of(stride) && blocks.len() < sample {
                    let mut a = [0.0f64; DIM];
                    a.copy_from_slice(b);
                    if a.iter().any(|v| *v != 0.0) {
                        blocks.push(a);
                    }
                }
                idx += 1;
            }
        }

        println!("── {name}  ({d_out}×{d_in}) — {} blocs échantillonnés", blocks.len());
        println!("   boule  classes   cos θ moyen   p1        p50       p99       surcoût 1/cosθ−1");
        let s = Searcher::new();
        for cap in [13u32, 12, 11, 10] {
            let mut ball = BallSearcher::with_level_cap(llvq_search::generic::MAX_LEVELS_ANY);
            ball.set_shell_cap(cap);
            let cs = llvq_search::classes::enumerate_classes(cap);
            let ncls = cs.even.len() + cs.odd.len();
            let mut cos: Vec<f64> = Vec::with_capacity(blocks.len());
            for x in &blocks {
                let f = ball.nearest_angular(&s, x);
                if f.shell == 0 {
                    continue;
                }
                let nx = x.iter().map(|a| a * a).sum::<f64>().sqrt();
                let np = ((16 * f.shell) as f64).sqrt();
                cos.push(f.dot / (np * nx));
            }
            cos.sort_by(f64::total_cmp);
            let m = cos.iter().sum::<f64>() / cos.len() as f64;
            let q = |p: f64| cos[((p * (cos.len() - 1) as f64).round() as usize).min(cos.len() - 1)];
            println!(
                "   {cap:5}  {ncls:7}   {m:.6}      {:.6}  {:.6}  {:.6}   {:+.3} %",
                q(0.01),
                q(0.50),
                q(0.99),
                (1.0 / m - 1.0) * 100.0
            );
        }
        println!();
    }
    Ok(())
}
