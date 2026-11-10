#![cfg(any(target_os = "macos", feature = "cuda"))]
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

/// The same fixture through the loop's `Drafter` contract: binary16 taps
/// delivered incrementally, the root last in `tokens`. The fixture's taps are
/// f16 values cast to f32 (the MLX door runs in f16), so the binary16 path
/// sees the same numbers and must return the same chain and tree.
#[test]
#[ignore = "needs the fixture, the packed 4B table and the drafter's 1 GB of weights"]
fn adapter_matches_mlx_reference() {
    use llvq_cuda::spec_loop::{DraftInput, Drafter};
    use llvq_llm::dflash_spec::{DFlashSpec, Shape};
    let fixture = home("llvq-dflash-fixture-2026-11-10.safetensors");
    assert!(fixture.exists(), "missing {}: run ops/dflash_fixture.py", fixture.display());
    let packed = home("q4b-hf-2026-09-28");
    assert!(packed.join("model.safetensors").exists(), "missing {}", packed.display());
    let dir = dflash::cached_snapshot()
        .unwrap_or_else(|| panic!("missing {}@{} in the Hugging Face cache", dflash::REPO, dflash::REVISION));
    let dev = Device::Cpu;
    let fx = candle_core::safetensors::load(&fixture, &dev).unwrap();
    let get = |k: String| fx.get(&k).unwrap_or_else(|| panic!("fixture: no {k}")).clone();
    let vocab = dflash::vocab_from_packed(&packed, &dev, DType::F32).unwrap();
    let hidden = vocab.table.dim(1).unwrap();
    for shape in [Shape::Chain(15), Shape::Tree(15)] {
        for p in 0..2 {
            let mut s = DFlashSpec::new(
                DFlashDrafter::load(&dir, &dev, DType::F32).unwrap(),
                Box::new(dflash::DenseVocab { table: vocab.table.clone() }),
                shape,
            )
            .unwrap();
            let ids = get(format!("p{p}.ids")).to_dtype(DType::U32).unwrap().to_vec1::<u32>().unwrap();
            let taps = get(format!("p{p}.taps")).to_dtype(DType::F16).unwrap();
            let f32_back = taps.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let orig = get(format!("p{p}.taps")).flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert_eq!(f32_back, orig, "the fixture's taps are not binary16 values");
            let bits: Vec<u16> =
                taps.flatten_all().unwrap().to_vec1::<half::f16>().unwrap().iter().map(|x| x.to_bits()).collect();
            let w = 5 * hidden;
            let mut done = 0;
            for r in 0..3 {
                let m = get(format!("p{p}.r{r}.m")).to_dtype(DType::U32).unwrap().to_vec1::<u32>().unwrap()[0] as usize;
                let nodes = s.draft(&DraftInput {
                    ctx_len: m,
                    tokens: &ids[..=m],
                    new_taps: &bits[done * w..m * w],
                    hidden,
                });
                done = m;
                assert!(s.last_error.is_none(), "{:?}", s.last_error);
                let want: Vec<(u32, u8)> = match shape {
                    Shape::Chain(_) => get(format!("p{p}.r{r}.argmax"))
                        .to_dtype(DType::U32)
                        .unwrap()
                        .to_vec1::<u32>()
                        .unwrap()
                        .into_iter()
                        .enumerate()
                        .map(|(i, t)| (t, i as u8))
                        .collect(),
                    Shape::Tree(_) => {
                        let t = get(format!("p{p}.r{r}.tree_tok")).to_dtype(DType::U32).unwrap().to_vec1::<u32>().unwrap();
                        let q = get(format!("p{p}.r{r}.tree_parent")).to_dtype(DType::I64).unwrap().to_vec1::<i64>().unwrap();
                        t.into_iter().zip(q).map(|(t, q)| (t, (q + 1) as u8)).collect()
                    }
                };
                assert_eq!(nodes, want, "prompt {p} round {r} {shape:?}");
            }
            assert_eq!(s.skipped, 0);
        }
    }
}

/// The card's table (`dflash::vocab_from_sealed`): the sealed file's int4
/// table is the packed directory's byte for byte, and its dequantized rows
/// are the fixture's rows exactly.
#[test]
#[ignore = "needs the sealed 4B, the packed 4B table and the fixture"]
fn the_sealed_table_is_the_packed_table_and_the_fixture_rows() {
    let sealed = home("q4b-sealed-2026-09-23/qwen3-4b-sealed.bin");
    assert!(sealed.exists(), "missing {}", sealed.display());
    let packed_dir = home("q4b-hf-2026-09-28");
    let fixture = home("llvq-dflash-fixture-2026-11-10.safetensors");
    assert!(fixture.exists(), "missing {}", fixture.display());
    let (packed, scales, biases, vocab, hidden) = dflash::sealed_q4_table(&sealed).unwrap();
    eprintln!("sealed table: {vocab} x {hidden}, {} packed bytes, {} scales", packed.len(), scales.len());
    // SAFETY: a read-only mmap of a file nothing in this process writes.
    let st = unsafe { candle_core::safetensors::MmapedSafetensors::new(packed_dir.join("model.safetensors")) }.unwrap();
    let get = |k: &str| st.load(k, &Device::Cpu).unwrap();
    let q = get("model.embed_tokens.qweight").flatten_all().unwrap().to_vec1::<u8>().unwrap();
    let s: Vec<u16> = get("model.embed_tokens.scales").flatten_all().unwrap().to_vec1::<half::f16>().unwrap().iter().map(|x| x.to_bits()).collect();
    let b: Vec<u16> = get("model.embed_tokens.biases").flatten_all().unwrap().to_vec1::<half::f16>().unwrap().iter().map(|x| x.to_bits()).collect();
    assert!(q == packed && s == scales && b == biases, "the sealed table and the packed directory's differ");
    eprintln!("the sealed file's table and the packed directory's are the same bytes");
    let fx = candle_core::safetensors::load(&fixture, &Device::Cpu).unwrap();
    let ids = fx["table.ids"].to_dtype(DType::U32).unwrap().to_vec1::<u32>().unwrap();
    let want = fx["table.rows"].flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let groups = scales.len() / vocab;
    for (i, &id) in ids.iter().enumerate() {
        let r = id as usize;
        let row = dflash::dequant_q4_f16(
            &packed[r * hidden / 2..(r + 1) * hidden / 2],
            &scales[r * groups..(r + 1) * groups],
            &biases[r * groups..(r + 1) * groups],
            hidden,
        );
        let got: Vec<f32> = row.iter().map(|x| x.to_f32()).collect();
        assert_eq!(got, want[i * hidden..(i + 1) * hidden], "row {id}");
    }
    eprintln!("{} fixture rows equal bit for bit", ids.len());
}
