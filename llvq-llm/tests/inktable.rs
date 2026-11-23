//! `bin/inktable` against the existing table path, on a real Qwen3 table.
//!
//! The existing path is two binaries: `bin/seal` narrows every carried
//! tensor to f16 through candle (`to_dtype(F16)`), and `bin/embedq q4`
//! requantizes the f16 embedding of a sealed file at int4 g64. This test
//! runs that path on the embedding of the smallest Qwen3 checkpoint in the
//! Hugging Face cache (Qwen3-0.6B, 151,936 × 1,024, BF16): the f16 narrowing
//! is `bin/seal`'s own lines, the requantization is the `embedq` binary
//! itself on a sealed file holding that one tensor. `bin/inktable` reads the
//! same table straight from the safetensors file. The claim: the two raw
//! tensors are the same bytes in the sealed stream's framing.
//!
//! Needs the checkpoint; nothing is downloaded. Missing, the test fails and
//! names it. Its files go to `$INKLING_LOT1B_DIR/inktable-test` (default
//! `~/llvq-inkling-mini/lot1-b`), about 0.4 GB.

use candle_core::{DType, Device};
use llvq_artifact::{read_header, read_raw, write_raw, ArtifactWriter, CodeKind, RawData, RawTensor};
use std::path::PathBuf;

const SNAP: &str = ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots";
const NAME: &str = "model.embed_tokens.weight";

fn checkpoint() -> PathBuf {
    let home = PathBuf::from(std::env::var("HOME").expect("HOME"));
    let snaps = home.join(SNAP);
    let found = std::fs::read_dir(&snaps)
        .ok()
        .and_then(|d| d.flatten().map(|e| e.path().join("model.safetensors")).find(|p| p.is_file()));
    found.unwrap_or_else(|| panic!("missing archive: {}/<rev>/model.safetensors (Qwen3-0.6B)", snaps.display()))
}

fn out_dir() -> PathBuf {
    let base = std::env::var("INKLING_LOT1B_DIR").map(PathBuf::from).unwrap_or_else(|_| {
        PathBuf::from(std::env::var("HOME").expect("HOME")).join("llvq-inkling-mini/lot1-b")
    });
    let d = base.join("inktable-test");
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn raw_bytes(t: &RawTensor) -> Vec<u8> {
    let mut b = Vec::new();
    write_raw(&mut b, t).unwrap();
    b
}

#[test]
#[ignore = "reads the Qwen3-0.6B checkpoint from the HF cache; run under the guard"]
fn inktable_reproduces_seal_then_embedq_on_qwen3_0_6b() {
    let ck = checkpoint();
    let dir = out_dir();

    // ---- the existing path: bin/seal's narrowing, then the embedq binary ----
    let shard = candle_core::safetensors::load(&ck, &Device::Cpu).unwrap();
    let t = &shard[NAME];
    assert_eq!(t.dtype(), DType::BF16, "the table is BF16 in the checkpoint");
    let dims = t.dims().to_vec();
    // `bin/seal.rs` lines 106-119, verbatim in substance.
    let data: Vec<u16> = t
        .to_dtype(DType::F16)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<half::f16>()
        .unwrap()
        .into_iter()
        .map(|v| v.to_bits())
        .collect();
    drop(shard);
    let sealed = dir.join("qwen3-0.6b-embed-f16.llvq");
    {
        let f = std::io::BufWriter::new(std::fs::File::create(&sealed).unwrap());
        let w = ArtifactWriter::with_kind(f, CodeKind::Tetra, 0).unwrap();
        w.seal(&[RawTensor { name: NAME.into(), dims: dims.clone(), data: RawData::F16(data) }], &[]).unwrap();
    }
    let requant = dir.join("qwen3-0.6b-embed-q4.llvq");
    let st = std::process::Command::new(env!("CARGO_BIN_EXE_embedq"))
        .args([sealed.to_str().unwrap(), requant.to_str().unwrap(), "q4"])
        .status()
        .unwrap();
    assert!(st.success(), "embedq failed");
    let mut r = std::io::BufReader::new(std::fs::File::open(&requant).unwrap());
    let h = read_header(&mut r).unwrap();
    assert_eq!(h.matrices, 0);
    let mut b4 = [0u8; 4];
    std::io::Read::read_exact(&mut r, &mut b4).unwrap();
    assert_eq!(u32::from_le_bytes(b4), 1);
    let want = read_raw(&mut r, h.version).unwrap();
    assert!(matches!(&want.data, RawData::Quant(q) if q.bits == 4 && q.group == 64), "embedq did not write int4 g64");

    // ---- inktable, straight from the safetensors file ----
    let out = dir.join("qwen3-0.6b-embed.llvqraw");
    let st = std::process::Command::new(env!("CARGO_BIN_EXE_inktable"))
        .args([ck.to_str().unwrap(), NAME, out.to_str().unwrap()])
        .status()
        .unwrap();
    assert!(st.success(), "inktable failed");
    let got = llvq_llm::inkseal::read_raw_file(&out).unwrap();
    assert_eq!(got.name, NAME, "a name without the model.llm. prefix is kept");
    assert_eq!(got.dims, dims);
    let (a, b) = (raw_bytes(&want), raw_bytes(&got));
    let differ = a.iter().zip(&b).filter(|(x, y)| x != y).count();
    assert_eq!(a.len(), b.len(), "framed sizes differ");
    assert_eq!(differ, 0, "{differ} of {} bytes differ", a.len());
    eprintln!("inktable = seal + embedq q4 on {NAME} {dims:?}: {} bytes identical", a.len());
    for p in [&sealed, &requant, &out] {
        std::fs::remove_file(p).unwrap();
    }
}
