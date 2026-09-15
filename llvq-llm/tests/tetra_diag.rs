#![forbid(unsafe_code)]

use candle_core::{DType, Device};
use candle_nn::{Activation, VarBuilder, VarMap};
use candle_transformers::models::qwen3::Config;
use llvq_llm::model::Qwen3;
use llvq_llm::tetra_diag::{self, Bundle, Plan, Windows};
use std::path::PathBuf;

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!(
            "llvq-schur-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&p).unwrap();
        Self(p)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn local_capture_replay_oracle_provenance_and_corruption_refusal() {
    let scratch = Scratch::new();
    let revision = "a".repeat(40);
    let checkpoint = scratch.0.join(&revision);
    std::fs::create_dir(&checkpoint).unwrap();
    let config = Config {
        vocab_size: 64,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        head_dim: 8,
        attention_bias: false,
        num_key_value_heads: 2,
        max_position_embeddings: 64,
        sliding_window: None,
        max_window_layers: 0,
        tie_word_embeddings: true,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-6,
        use_sliding_window: false,
        hidden_act: Activation::Silu,
    };
    let map = VarMap::new();
    let model = Qwen3::new(
        &config,
        VarBuilder::from_varmap(&map, DType::F32, &Device::Cpu),
        llvq_llm::kvq::KvMode::F16,
    )
    .unwrap();
    drop(model);
    map.save(checkpoint.join("model.safetensors")).unwrap();
    tetra_diag::write_json(
        &checkpoint.join("config.json"),
        &serde_json::json!({
            "vocab_size":64,"hidden_size":32,"intermediate_size":64,"num_hidden_layers":2,
            "num_attention_heads":4,"head_dim":8,"attention_bias":false,"num_key_value_heads":2,
            "max_position_embeddings":64,"sliding_window":null,"max_window_layers":0,
            "tie_word_embeddings":true,"rope_theta":10000.0,"rms_norm_eps":1e-6,
            "use_sliding_window":false,"hidden_act":"silu"
        }),
    )
    .unwrap();
    std::fs::write(checkpoint.join("tokenizer.json"), "{}").unwrap();
    let calibration = scratch.0.join("calibration.json");
    let validation = scratch.0.join("validation.json");
    for (p, offset, name) in [(&calibration, 0, "train"), (&validation, 16, "validation")] {
        tetra_diag::write_json(
            p,
            &Windows {
                source: PathBuf::from(name),
                source_fingerprint_fnv1a: name.into(),
                seed: 1,
                offsets: vec![0],
                ids: vec![(offset..offset + 16).collect()],
            },
        )
        .unwrap();
    }
    let plan = Plan {
        version: 1,
        checkpoint,
        revision,
        calibration,
        validation,
        device: "cpu".into(),
        layers: vec![0, 1],
        projections: vec!["self_attn.q_proj".into(), "mlp.gate_proj".into()],
        rows_per_projection: 2,
        snapshots_per_row: 1,
        damping: 0.02,
        h_shrink: 0.7,
        rotation_seed: 42,
    };
    let inspection = tetra_diag::inspect(&plan).unwrap();
    assert_eq!(inspection["row_count"], 8);
    let output = scratch.0.join("capture");
    tetra_diag::capture(&plan, &output).unwrap();
    assert!(output.join("capture-complete.json").exists());
    assert!(tetra_diag::capture(&plan, &output).is_err());
    let path = output.join("layer-1-self_attn-q_proj.json");
    let bundle: Bundle = tetra_diag::read_json(&path).unwrap();
    assert_eq!(bundle.row_ids, vec![0, 31]);
    assert_eq!(bundle.provenance["prefix"], "checkpoint-f32");
    let first = scratch.0.join("first");
    let second = scratch.0.join("second");
    tetra_diag::replay(&path, &first).unwrap();
    tetra_diag::replay(&path, &second).unwrap();
    for row in bundle.row_ids.iter() {
        let a: serde_json::Value =
            tetra_diag::read_json(&first.join(format!("row-{row}.json"))).unwrap();
        let b: serde_json::Value =
            tetra_diag::read_json(&second.join(format!("row-{row}.json"))).unwrap();
        assert_eq!(a["diagnostic"], b["diagnostic"]);
    }
    assert!(tetra_diag::replay(&path, &first).is_err());
    let arr = output.join(&bundle.hessian.name);
    let mut bytes = std::fs::read(&arr).unwrap();
    bytes[0] ^= 1;
    std::fs::write(&arr, bytes).unwrap();
    let error = tetra_diag::replay(&path, &scratch.0.join("corrupt"))
        .unwrap_err()
        .to_string();
    assert!(error.contains("fingerprint"), "{error}");
    let mut bad = plan.clone();
    bad.validation = bad.calibration.clone();
    assert!(tetra_diag::inspect(&bad)
        .unwrap_err()
        .to_string()
        .contains("distinct corpus"));
    bad = plan.clone();
    bad.layers = vec![0, 0];
    assert!(tetra_diag::inspect(&bad).is_err());
    bad = plan.clone();
    bad.projections = vec!["self_attn.v_proj".into()];
    assert!(tetra_diag::inspect(&bad).is_err());
    let mut val: Windows = tetra_diag::read_json(&plan.validation).unwrap();
    let cal: Windows = tetra_diag::read_json(&plan.calibration).unwrap();
    val.ids = cal.ids.clone();
    let leaked = scratch.0.join("leaked.json");
    tetra_diag::write_json(&leaked, &val).unwrap();
    bad = plan.clone();
    bad.validation = leaked;
    assert!(tetra_diag::inspect(&bad)
        .unwrap_err()
        .to_string()
        .contains("reused"));
    val.ids = vec![(16..32).collect(), (32..48).collect()];
    val.offsets = vec![0, 8];
    let overlap = scratch.0.join("overlap.json");
    tetra_diag::write_json(&overlap, &val).unwrap();
    bad.validation = overlap;
    assert!(tetra_diag::inspect(&bad)
        .unwrap_err()
        .to_string()
        .contains("overlapping"));
}

#[test]
fn selection_is_dimension_only_and_rejects_oversampling() {
    assert_eq!(tetra_diag::positions(42, 3).unwrap(), vec![0, 20, 41]);
    assert_eq!(tetra_diag::positions(32, 4).unwrap(), vec![0, 10, 20, 31]);
    assert!(tetra_diag::positions(2, 3).is_err());
    assert!(tetra_diag::positions(4, 0).is_err());
}
