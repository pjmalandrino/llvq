//! The DFlash drafter port (`llvq_llm::dflash`) against the MLX reference.
//!
//! The fixture is written by `ops/dflash_fixture.py` from the sealed 4B
//! through the MLX door: two prompts, three rounds each, the context extended
//! round by round. f32 on both sides. Required: the same 15 argmax and the
//! same tree of 15 (token and parent) on every round, the reference's top 64
//! logits a row within 1e-4 of the row's largest, the final hidden states
//! within 1e-4 relative (reached: 4e-6 on Metal, 2026-11-10), and the dequantized table rows within 1e-6.
//!
//! Needs three files and fails naming the one that is missing:
//! `~/llvq-dflash-fixture-2026-11-10.safetensors`, `~/q4b-hf-2026-09-28/`
//! and the drafter in the local Hugging Face cache.

use std::path::PathBuf;

use candle_core::{DType, Device};
use llvq_llm::dflash::{self, DFlashDrafter};

fn home(p: &str) -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME")).join(p)
}

#[test]
#[ignore = "needs the fixture, the packed 4B table and the drafter's 1 GB of weights"]
fn drafter_matches_mlx_reference() {
    let fixture = home("llvq-dflash-fixture-2026-11-10.safetensors");
    assert!(fixture.exists(), "missing {}: run ops/dflash_fixture.py", fixture.display());
    let packed = home("q4b-hf-2026-09-28");
    assert!(packed.join("model.safetensors").exists(), "missing {}", packed.display());
    let dir = dflash::cached_snapshot()
        .unwrap_or_else(|| panic!("missing {}@{} in the Hugging Face cache", dflash::REPO, dflash::REVISION));
    let dev = Device::Cpu;
    let drafter = DFlashDrafter::load(&dir, &dev, DType::F32).unwrap();
    let vocab = dflash::vocab_from_packed(&packed, &dev, DType::F32).unwrap();
    let rep = dflash::check_fixture(&fixture, &drafter, &vocab).unwrap();
    eprintln!("{rep}");
    assert_eq!(rep.rounds, 6, "{rep}");
    assert_eq!(rep.chain_equal, rep.rounds, "{rep}");
    assert_eq!(rep.tree_equal, rep.rounds, "{rep}");
    assert!(rep.logit_rel < 1e-4, "{rep}");
    assert!(rep.hidden_rel < 1e-4, "{rep}");
    assert!(rep.table_abs < 1e-6, "{rep}");
}
