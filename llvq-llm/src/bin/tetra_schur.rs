//! Local preparation, capture and replay for the Tetra Schur pilot.
use anyhow::{bail, Context, Result};
use llvq_llm::tetra_diag::{self, Plan};
use std::path::Path;

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("inspect") if args.len()==2 => {
            let plan:Plan=tetra_diag::read_json(Path::new(&args[1]))?;
            println!("{}",serde_json::to_string_pretty(&tetra_diag::inspect(&plan)?)?);
        }
        Some("capture") if args.len()==3 => {
            let plan:Plan=tetra_diag::read_json(Path::new(&args[1]))?;
            tetra_diag::capture(&plan,Path::new(&args[2]))?;
        }
        Some("replay") if args.len()==3 => tetra_diag::replay(Path::new(&args[1]),Path::new(&args[2]))?,
        Some("tokens") if args.len()==7 => {
            tetra_diag::prepare_tokens(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),
                args[4].parse().context("count")?,args[5].parse().context("length")?,args[6].parse().context("seed")?)?;
        }
        _ => bail!("usage: tetra_schur inspect PLAN.json | capture PLAN.json NEW_DIR | replay BUNDLE.json NEW_DIR | tokens LOCAL_CHECKPOINT LOCAL.parquet NEW.json COUNT LENGTH SEED"),
    }
    Ok(())
}
