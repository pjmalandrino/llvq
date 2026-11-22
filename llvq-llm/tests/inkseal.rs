//! `llvq_llm::inkseal`'s seal on a synthetic layer directory: the records
//! pass through byte for byte, `verify` holds every expert to its
//! reconstruction, a flipped code bit fails it, and the complement of a
//! native checkpoint lands where the module says (sidecar, raw section,
//! refusal, omission). The real gate runs on layer 6 of Inkling 2 Mini; this
//! test holds the same code to the same claims without the 4 GB of hats.

use llvq_artifact::{write_matrix_with, CodeKind, QuantizedMatrix, TETRA_SHELL_CAP};
use llvq_core::{SplitMix64, DIM};
use llvq_llm::inkseal::{self, SealInputs};
use llvq_quant::quantizer::BlockCode;
use llvq_search::tetra::{Tetra, LABEL_MASK};
use serde_json::json;
use std::io::Write;
use std::path::{Path, PathBuf};

const E: usize = 3;
/// Expert shapes: gate and up are `F × D`, down `D × F`; `F` has a tail.
const D: usize = 2 * DIM;
const F: usize = DIM + 8;

fn dir(name: &str) -> PathBuf {
    let p = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("inkseal").join(name);
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn matrix(name: String, d_out: usize, d_in: usize, rng: &mut SplitMix64, t: &Tetra) -> QuantizedMatrix {
    QuantizedMatrix {
        name,
        d_out,
        d_in,
        codes: (0..d_out * (d_in / DIM))
            .map(|_| BlockCode { point: t.decode(rng.next() & LABEL_MASK), gain: (rng.next() & 1) as u32 })
            .collect(),
        row_scales: (0..d_out).map(|_| 1e-3 + rng.next_f64()).collect(),
        centroids: vec![0.7, 1.1],
        rotation_seed: Some(rng.next()),
        shell_cap: TETRA_SHELL_CAP,
        tail: (0..d_out * (d_in % DIM)).map(|_| (rng.next_f64() - 0.5) as f32 as f64).collect(),
    }
}

fn put_f32(path: &Path, v: &[f32]) {
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    std::fs::write(path, b).unwrap();
}

/// A layer directory as `bin/extq` leaves it, with records whose codes are
/// random Tetra words and hats that are their decode.
fn layer(root: &Path, l: u64, seed: u64) -> PathBuf {
    let ld = root.join(format!("L{l:02}"));
    std::fs::create_dir_all(ld.join("codes")).unwrap();
    let t = Tetra::new();
    let tb = llvq_artifact::Codebook::new(CodeKind::Tetra).unwrap();
    let mut rng = SplitMix64::new(seed);
    let (mut w13, mut w2) = (Vec::new(), Vec::new());
    for e in 0..E as u64 {
        let mut buf = Vec::new();
        for (proj, o, i) in [("gate", F, D), ("up", F, D), ("down", D, F)] {
            let m = matrix(inkseal::expert_name(l, e, proj), o, i, &mut rng, &t);
            write_matrix_with(&mut buf, inkseal::VERSION, &tb, &m).unwrap();
            let d = llvq_artifact::decode_matrix(&m);
            if proj == "down" { w2.extend(d) } else { w13.extend(d) }
        }
        std::fs::write(inkseal::codes_path(&ld, e), buf).unwrap();
    }
    put_f32(&ld.join("w13_hat.f32"), &w13);
    put_f32(&ld.join("w2_hat.f32"), &w2);
    std::fs::write(ld.join("encode.json"), json!({"layer": l, "experts_total": E, "experts_done": E}).to_string()).unwrap();
    std::fs::write(ld.join("manifest.json"), json!({"layer": l}).to_string()).unwrap();
    std::fs::write(ld.join("VERIFIED"), "test").unwrap();
    ld
}

/// A native checkpoint in miniature: the experts of layers 1 and 2 (U8 and
/// F32 pieces), an attention weight, a BF16 norm, an F32 query, an F16 tensor
/// and a vision weight.
fn native(path: &Path) {
    let ts: Vec<(&str, &str, Vec<usize>)> = vec![
        ("model.llm.layers.1.mlp.experts.w13_weight", "U8", vec![E, 2 * F, D / 2]),
        ("model.llm.layers.1.mlp.experts.w13_weight_amax", "F32", vec![E]),
        ("model.llm.layers.1.mlp.experts.w2_weight", "U8", vec![E, D, F / 2]),
        ("model.llm.layers.2.mlp.experts.w13_weight", "U8", vec![E, 2 * F, D / 2]),
        ("model.llm.layers.2.mlp.experts.w13_weight_amax", "F32", vec![E]),
        ("model.llm.layers.2.mlp.experts.w2_weight", "U8", vec![E, D, F / 2]),
        ("model.llm.layers.1.attn.wq_du.weight", "BF16", vec![8, 16]),
        ("model.llm.norm.weight", "BF16", vec![16]),
        ("model.llm.attn_res.output_query", "F32", vec![16]),
        ("model.llm.layers.1.attn.odd", "F16", vec![4]),
        ("model.visual.proj_in.weight", "BF16", vec![4, 4]),
    ];
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    let mut rng = SplitMix64::new(9);
    for (name, dtype, shape) in &ts {
        let n: usize = shape.iter().product::<usize>() * inkseal::dtype_bytes(dtype).unwrap() as usize;
        let at = data.len();
        data.extend((0..n).map(|_| rng.next() as u8));
        header.insert(name.to_string(), json!({"dtype": dtype, "shape": shape, "data_offsets": [at, at + n]}));
    }
    let mut h = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    while !h.len().is_multiple_of(8) {
        h.push(b' ');
    }
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(&(h.len() as u64).to_le_bytes()).unwrap();
    f.write_all(&h).unwrap();
    f.write_all(&data).unwrap();
}

#[test]
fn records_pass_through_and_decode_to_the_hats() {
    let root = dir("pass");
    let l1 = layer(&root, 1, 11);
    let out = root.join("s.llvq");
    let r = inkseal::seal(&SealInputs { out: out.clone(), layer_dirs: vec![l1.clone()], ..Default::default() }).unwrap();
    assert_eq!((r.records, r.expert_records), (3 * E, 3 * E));
    // Byte for byte: the record section is the concatenation of the codes files.
    let sealed = std::fs::read(&out).unwrap();
    let mut want = Vec::new();
    for e in 0..E as u64 {
        want.extend(std::fs::read(inkseal::codes_path(&l1, e)).unwrap());
    }
    let header = 4 + 4 + 8 + 8 + 4 + 4;
    assert_eq!(&sealed[header..header + want.len()], &want[..], "a record moved");
    let v = inkseal::verify(&out, &[l1], 2).unwrap();
    assert_eq!((v.compared_records(), v.differing_weights(), v.decode_failures.len()), (3 * E, 0, 0));
    assert_eq!(v.compared_weights(), (E * 3 * F * D) as u64);
}

#[test]
fn a_flipped_code_bit_fails_verify() {
    let root = dir("flip");
    let l1 = layer(&root, 1, 12);
    let out = root.join("s.llvq");
    inkseal::seal(&SealInputs { out: out.clone(), layer_dirs: vec![l1.clone()], ..Default::default() }).unwrap();
    for (k, bit) in [(0usize, 3u64), (4, 47), (3 * E - 1, 100)] {
        let bad = root.join("flip.llvq");
        inkseal::flip_code_bit(&out, &bad, k, bit).unwrap();
        let v = inkseal::verify(&bad, std::slice::from_ref(&l1), 2).unwrap();
        assert!(v.differing_weights() > 0 || !v.decode_failures.is_empty(), "record {k} bit {bit}: no difference");
    }
}

#[test]
fn a_layer_missing_from_the_file_fails_verify() {
    let root = dir("missing");
    let (l1, l2) = (layer(&root, 1, 13), layer(&root, 2, 14));
    let out = root.join("s.llvq");
    inkseal::seal(&SealInputs { out: out.clone(), layer_dirs: vec![l1], ..Default::default() }).unwrap();
    assert!(inkseal::verify(&out, &[l2], 2).is_err(), "a layer with no record in the file passed");
}

#[test]
fn an_unverified_layer_is_refused() {
    let root = dir("unverified");
    let l1 = layer(&root, 1, 15);
    std::fs::remove_file(l1.join("VERIFIED")).unwrap();
    let e = inkseal::seal(&SealInputs { out: root.join("s.llvq"), layer_dirs: vec![l1], ..Default::default() });
    assert!(e.unwrap_err().to_string().contains("VERIFIED"));
}

#[test]
fn the_complement_goes_where_the_module_says() {
    let root = dir("complement");
    let l1 = layer(&root, 1, 16);
    let nat = root.join("native.safetensors");
    native(&nat);
    let base = SealInputs {
        out: root.join("s.llvq"),
        layer_dirs: vec![l1.clone()],
        native: Some(nat.clone()),
        sidecar: Some(root.join("s.sidecar.safetensors")),
        ..Default::default()
    };
    // Layer 2's NVFP4 experts have no replacement: refused unless partial.
    let e = inkseal::seal(&base).unwrap_err().to_string();
    assert!(e.contains("layers.2.mlp.experts") && e.contains("--partial"), "{e}");
    let r = inkseal::seal(&SealInputs { partial: true, ..base }).unwrap();
    // Omitted: layer 2's three expert tensors, the F32 amax included.
    assert_eq!(r.omitted_tensors.len(), 3, "{:?}", r.omitted_tensors);
    assert!(r.omitted_tensors.iter().all(|n| n.contains("layers.2.mlp.experts")));
    // The F16 tensor went to the raw section, renamed.
    assert_eq!(r.raws, 1);
    let sc = inkseal::read_safetensors(&root.join("s.sidecar.safetensors")).unwrap();
    let names: Vec<&str> = sc.tensors.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(
        names,
        vec![
            "model.attn_res.output_query",
            "model.layers.1.attn.wq_du.weight",
            "model.norm.weight",
            "model.visual.proj_in.weight"
        ]
    );
    assert_eq!(sc.metadata["llvq_partial"], "true");
    let (n, _) = inkseal::verify_sidecar(&root.join("s.sidecar.safetensors"), &nat).unwrap();
    assert_eq!(n, 4);
    let v = inkseal::verify(&root.join("s.llvq"), &[l1], 2).unwrap();
    assert_eq!((v.raws, v.blobs, v.differing_weights()), (1, 1, 0));
}

#[test]
fn text_only_omits_the_vision_tower() {
    let root = dir("textonly");
    let (l1, l2) = (layer(&root, 1, 17), layer(&root, 2, 18));
    let nat = root.join("native.safetensors");
    native(&nat);
    let r = inkseal::seal(&SealInputs {
        out: root.join("s.llvq"),
        layer_dirs: vec![l2, l1],
        native: Some(nat),
        sidecar: Some(root.join("s.sidecar.safetensors")),
        text_only: true,
        ..Default::default()
    })
    .unwrap();
    assert_eq!(r.omitted_tensors, vec!["model.visual.proj_in.weight".to_string()]);
    assert_eq!(r.omitted_elements, 16);
    assert_eq!(r.sidecar_tensors, 3);
}

/// Two `inkwh` groups (a Tetra matrix with a tail, an int4 one), sealed with a
/// layer and a raw table, then verified: records against inkwh's files,
/// the table against its file, the native accounting exactly once. A flipped
/// bit in each kind of payload is caught.
#[test]
fn inkwh_records_and_tables_are_sealed_and_verified() {
    let root = dir("wh");
    let l1 = layer(&root, 1, 21);
    let mut rng = SplitMix64::new(22);
    let mut wh_dirs = Vec::new();
    for (stream, n, d_out, kind, nat) in [("q", 56usize, 8usize, "tetra", "wq_du"), ("v", 128, 4, "int4", "wv_dv")] {
        let g = root.join(format!("in_{stream}"));
        std::fs::create_dir_all(&g).unwrap();
        let w: Vec<f32> = (0..d_out * n).map(|_| (rng.next_f64() - 0.5) as f32).collect();
        put_f32(&g.join("w.f32"), &w);
        let mut h = vec![0.0f64; n * n];
        for i in 0..n {
            h[i * n + i] = 3.0 + rng.next_f64();
        }
        let hb: Vec<u8> = h.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(g.join("h.f64"), hb).unwrap();
        let name = format!("model.layers.1.attn.{nat}.weight");
        let m = json!({"n": n, "h": "h.f64", "weight_sum": 3.0, "tokens": 3, "layer": 1, "stream": stream,
                       "matrices": [{"name": name, "w": "w.f32", "d_out": d_out, "kind": kind}]});
        std::fs::write(g.join("m.json"), m.to_string()).unwrap();
        let out = root.join(format!("L01_{stream}"));
        inkseal::run_wh(&g.join("m.json"), &out, 1).unwrap();
        wh_dirs.push(out);
    }
    // A table: an F16 tensor quantized at int4 g64, as inktable writes it.
    let tf = root.join("embed.llvqraw");
    let t = llvq_artifact::RawTensor {
        name: "model.embed.weight".into(),
        dims: vec![4, 128],
        data: llvq_artifact::RawData::F16((0..512).map(|_| half::f16::from_f64(rng.next_f64() - 0.5).to_bits()).collect()),
    };
    inkseal::write_raw_file(&tf, &llvq_llm::embedquant::quantize_affine(&t, 4, 64).unwrap()).unwrap();
    // The native file: layer 1's experts, the two attention matrices, the table, a norm.
    let nat = root.join("native.safetensors");
    {
        let ts: Vec<(&str, &str, Vec<usize>)> = vec![
            ("model.llm.layers.1.mlp.experts.w13_weight", "U8", vec![E, 2 * F, D / 2]),
            ("model.llm.layers.1.mlp.experts.w2_weight", "U8", vec![E, D, F / 2]),
            ("model.llm.layers.1.attn.wq_du.weight", "BF16", vec![8, 56]),
            ("model.llm.layers.1.attn.wv_dv.weight", "BF16", vec![4, 128]),
            ("model.llm.embed.weight", "BF16", vec![4, 128]),
            ("model.llm.norm.weight", "BF16", vec![16]),
        ];
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, dtype, shape) in &ts {
            let n: usize = shape.iter().product::<usize>() * inkseal::dtype_bytes(dtype).unwrap() as usize;
            let at = data.len();
            data.extend((0..n).map(|_| rng.next() as u8));
            header.insert(name.to_string(), json!({"dtype": dtype, "shape": shape, "data_offsets": [at, at + n]}));
        }
        let mut h = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        while !h.len().is_multiple_of(8) {
            h.push(b' ');
        }
        let mut f = std::fs::File::create(&nat).unwrap();
        f.write_all(&(h.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&h).unwrap();
        f.write_all(&data).unwrap();
    }
    let out = root.join("s.llvq");
    let sc = root.join("s.sidecar.safetensors");
    let r = inkseal::seal(&SealInputs {
        out: out.clone(),
        layer_dirs: vec![l1.clone()],
        wh_dirs: wh_dirs.clone(),
        raw_files: vec![tf.clone()],
        native: Some(nat.clone()),
        sidecar: Some(sc.clone()),
        ..Default::default()
    })
    .unwrap();
    let vin = |sealed: &Path| inkseal::VerifyInputs {
        sealed: sealed.to_path_buf(),
        layer_dirs: vec![l1.clone()],
        wh_dirs: wh_dirs.clone(),
        raw_files: vec![tf.clone()],
        sidecar: Some(sc.clone()),
        native: Some(nat.clone()),
        threads: 2,
    };
    let v = inkseal::verify_all(&vin(&out)).unwrap();
    assert!(v.failures().is_empty(), "{:?}", v.failures());
    assert_eq!((v.wh_records, v.raws_compared, v.native_tensors, v.text_omitted), (2, 1, 6, 0));
    assert_eq!((r.records, r.int4_records, r.raws, r.sidecar_tensors), (3 * E + 2, 1, 1, 1));
    assert_eq!(v.by_class["expert records"], 2);
    assert_eq!(v.by_class["named record"], 2);
    // Records are laid out: 3E expert records, then wq_du (Tetra), wv_dv (int4).
    for (raw, k, bit) in [(false, 3 * E, 5u64), (false, 3 * E + 1, 9), (true, 0, 33)] {
        let bad = root.join("flip.llvq");
        inkseal::flip_bit(&out, &bad, raw, k, bit).unwrap();
        let v = inkseal::verify_all(&vin(&bad)).unwrap();
        assert!(!v.failures().is_empty(), "flip raw={raw} k={k} bit={bit} passed");
    }
}
