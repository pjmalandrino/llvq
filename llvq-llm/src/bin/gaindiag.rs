//! Where the fitted gain levels actually land, on real weights.
//!
//! Diagnostic only — it produces no perplexity, no verdict and no threshold.
//! It inspects a deterministic function (`fit_gain_centroids`) on weights that
//! already exist, and prints where its levels sit. Nothing here is a
//! measurement in the sense the pre-registrations use.
//!
//! The question it answers: at `k = 1` Lloyd converges the single level onto
//! the **mean** of `g = ‖block‖ / row_scale`. At `k = 2` the two levels are the
//! means of their own partitions, so on a right-skewed `g` the bulk level sits
//! **below** that mean and most blocks are reconstructed short. If that is what
//! the numbers show, the two-level gain code — the configuration we serve — is
//! biased by construction, and the 48-bit split is not what the ladder was
//! measuring.
//!
//! ```text
//! LLVQ_MODEL=Qwen/Qwen3-0.6B cargo run --release -p llvq-llm --features metal --bin gaindiag
//! ```

use llvq_llm::loader::Checkpoint;
use llvq_quant::quantizer::{fit_gain_centroids, row_scale};

const DIM: usize = 24;

fn main() -> anyhow::Result<()> {
    let repo = std::env::var("LLVQ_MODEL").unwrap_or_else(|_| "Qwen/Qwen3-0.6B".into());
    let ck = Checkpoint::fetch(&repo)?;
    let vb = ck.var_builder(candle_core::DType::F32, &candle_core::Device::Cpu)?;

    let h = ck.config.hidden_size;
    let inter = ck.config.intermediate_size;
    let heads = ck.config.num_attention_heads;
    let kv = ck.config.num_key_value_heads;

    // Three shapes, three roles: a narrow attention projection, the widest
    // MLP input, and the one whose input is the MLP width.
    let picks: Vec<(&str, usize, usize)> = vec![
        ("model.layers.0.self_attn.q_proj.weight", heads * ck.config.head_dim, h),
        ("model.layers.0.mlp.gate_proj.weight", inter, h),
        ("model.layers.0.mlp.down_proj.weight", h, inter),
        ("model.layers.13.mlp.gate_proj.weight", inter, h),
        ("model.layers.27.mlp.down_proj.weight", h, inter),
    ];
    let _ = kv;

    println!("model: {repo}   (hidden {h}, intermediate {inter})");
    println!("g = ‖block of 24‖ / row_scale(row) — this is what the gain codebook quantizes\n");

    for (name, d_out, d_in) in picks {
        let t = match vb.get((d_out, d_in), name) {
            Ok(t) => t,
            Err(e) => {
                println!("{name}: missing ({e})\n");
                continue;
            }
        };
        let w: Vec<f64> = t
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(f64::from)
            .collect();

        // The exact population `fit_gain_centroids` fits on.
        let mut g = Vec::with_capacity(d_out * (d_in / DIM));
        for i in 0..d_out {
            let row = &w[i * d_in..(i + 1) * d_in];
            let s = row_scale(row);
            if s <= 0.0 {
                continue;
            }
            for b in row.chunks_exact(DIM) {
                let n = b.iter().map(|a| a * a).sum::<f64>().sqrt();
                if n > 0.0 {
                    g.push(n / s);
                }
            }
        }
        let n = g.len() as f64;
        let mean = g.iter().sum::<f64>() / n;
        let mut srt = g.clone();
        srt.sort_by(f64::total_cmp);
        let q = |p: f64| srt[((p * (srt.len() - 1) as f64).round() as usize).min(srt.len() - 1)];
        let var = g.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let skew = g.iter().map(|v| ((v - mean) / var.sqrt()).powi(3)).sum::<f64>() / n;

        println!("── {name}  ({d_out}×{d_in}, {} blocks)", g.len());
        println!(
            "   g: mean {mean:.6}  median {:.6}  p1 {:.6}  p99 {:.6}  skew {skew:+.3}",
            q(0.5),
            q(0.01),
            q(0.99)
        );

        for k in [0u32, 1, 2, 4] {
            let c = fit_gain_centroids(&w, d_out, d_in, DIM, k, 40);
            // Where each block lands, and what that does to its norm.
            let mut counts = vec![0u64; c.len()];
            let mut short = 0u64;
            for &v in &g {
                let j = (0..c.len())
                    .min_by(|&a, &b| (v - c[a]).abs().total_cmp(&(v - c[b]).abs()))
                    .unwrap();
                counts[j] += 1;
                if c[j] < v {
                    short += 1;
                }
            }
            let lvls = c
                .iter()
                .map(|v| format!("{v:.6}"))
                .collect::<Vec<_>>()
                .join("  ");
            let pct = counts
                .iter()
                .map(|&n2| format!("{:.1}%", 100.0 * n2 as f64 / n))
                .collect::<Vec<_>>()
                .join("  ");
            let rel = (c[0] / mean - 1.0) * 100.0;
            println!(
                "   k={k} ({} levels)  levels: {lvls}",
                c.len()
            );
            println!(
                "        population: {pct}   |  block reconstructed too short: {:.1}%  |  low level vs k=0 mean: {rel:+.2}%",
                100.0 * short as f64 / n
            );
        }
        println!();
    }
    Ok(())
}
