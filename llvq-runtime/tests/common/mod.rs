//! A tiny Qwen3 behind the real server, on the CPU, for the route tests and
//! for `examples/tiny_server.rs`, which the Python client checks talk to.
//!
//! The weights are deterministic (SplitMix64), so two loads are the same
//! model and a reference run outside the server reads the same logits. The
//! embedding rows of `<|im_end|>` and `<|endoftext|>` are zero, and the head
//! is tied, so their logits are zero while 60-odd random logits compete: a
//! greedy answer never stops on its own and runs to its cap. That is what
//! lets a test time a cancelled answer against a whole one.

#![allow(dead_code)]

use candle_core::{DType, Device, Tensor};
use candle_nn::{Activation, VarBuilder, VarMap};
use candle_transformers::models::qwen3::Config;
use llvq_llm::kvq::KvMode;
use llvq_llm::model::Qwen3;
use llvq_llm::served_decode::Decoder;
use llvq_runtime::api::AppState;
use llvq_runtime::engine::{KeepAlive, Loaded, Loader, Models};
use llvq_runtime::library::{Configs, ModelName};
use llvq_runtime::store::{Manifest, Store};
use std::sync::Arc;
use std::time::Duration;

pub const SPECIAL: [&str; 7] = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>", "\n", "\n\n"];
pub const WORDS: [&str; 24] = [
    "system", "user", "assistant", "hello", "world", "how", "are", "you", "fine", "thanks", "the", "a", "cat",
    "dog", "sat", "on", "mat", "yes", "no", "why", "because", "END", "one", "two",
];

/// A word-level tokenizer with Qwen3's markers as added tokens.
pub fn tokenizer() -> tokenizers::Tokenizer {
    let mut vocab = serde_json::Map::new();
    let mut added = Vec::new();
    for (i, s) in SPECIAL.iter().enumerate() {
        vocab.insert((*s).into(), i.into());
        added.push(serde_json::json!({"id": i, "content": s, "single_word": false, "lstrip": false,
            "rstrip": false, "normalized": false, "special": true}));
    }
    vocab.insert("<unk>".into(), SPECIAL.len().into());
    for (i, w) in WORDS.iter().enumerate() {
        vocab.insert((*w).into(), (SPECIAL.len() + 1 + i).into());
    }
    let j = serde_json::json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": added,
        "normalizer": null, "pre_tokenizer": {"type": "WhitespaceSplit"}, "post_processor": null,
        "decoder": null, "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"}
    });
    tokenizers::Tokenizer::from_bytes(serde_json::to_vec(&j).unwrap()).expect("the tiny tokenizer builds")
}

pub fn config() -> Config {
    Config {
        vocab_size: 64,
        hidden_size: 64,
        intermediate_size: 128,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        head_dim: 16,
        attention_bias: false,
        num_key_value_heads: 2,
        max_position_embeddings: 4096,
        sliding_window: None,
        max_window_layers: 0,
        tie_word_embeddings: true,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-6,
        use_sliding_window: false,
        hidden_act: Activation::Silu,
    }
}

struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

/// The tiny model, the same weights every call.
pub fn model() -> Qwen3 {
    let map = VarMap::new();
    let vb = VarBuilder::from_varmap(&map, DType::F32, &Device::Cpu);
    let m = Qwen3::new(&config(), vb, KvMode::F16).expect("the tiny model builds");
    let data = map.data().lock().unwrap();
    let mut names: Vec<&String> = data.keys().collect();
    names.sort();
    let mut rng = SplitMix(0x11_7A_5E_ED);
    for n in names {
        let var = &data[n];
        let shape = var.shape().clone();
        let count = shape.elem_count();
        let mut v: Vec<f32> = match n.contains("norm") {
            true => vec![1.0; count],
            false => (0..count).map(|_| rng.uniform() * 0.5).collect(),
        };
        if n.contains("embed_tokens") {
            let hidden = config().hidden_size;
            for stop in [1usize, 2] {
                v[stop * hidden..(stop + 1) * hidden].iter_mut().for_each(|x| *x = 0.0);
            }
        }
        var.set(&Tensor::from_vec(v, shape, &Device::Cpu).unwrap()).unwrap();
    }
    drop(data);
    m
}

pub struct TinyLoader;

impl Loader for TinyLoader {
    fn load(&self, _: &ModelName, _: &Manifest, _: &Store) -> anyhow::Result<Loaded> {
        Ok(Loaded {
            qwen: model(),
            tokenizer: tokenizer(),
            decoder: Decoder::eager(),
            backend: "dense-cpu".into(),
            config: None,
            lines: Vec::new(),
        })
    }
}

pub fn manifest() -> Manifest {
    Manifest {
        schema: 1,
        sha256: "0".repeat(64),
        size: 1234,
        repo: "test/tiny".into(),
        revision: "0".repeat(40),
        file: "tiny.bin".into(),
        family: "qwen3".into(),
        parameter_size: "tiny".into(),
        bits_per_param: "32".into(),
        configs: Configs { cuda: None, metal: None, cpu: None },
        modified_at: 1_762_387_200,
    }
}

/// A store under a fresh temporary directory, with `tiny:test` registered.
pub fn store(tag: &str) -> Arc<Store> {
    let dir = std::env::temp_dir().join(format!("llvq-runtime-test-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let s = Store::at(dir);
    s.write_manifest(&ModelName::parse("tiny:test").unwrap(), &manifest()).unwrap();
    Arc::new(s)
}

/// The server on 127.0.0.1 at a free port, on its own runtime thread.
/// Returns the `host:port`.
pub fn serve(store: Arc<Store>, hub: &str) -> String {
    let models = Arc::new(Models::new(store, Arc::new(TinyLoader), 1, KeepAlive::For(Duration::from_secs(300))));
    let state = AppState::new(models, hub.to_string());
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
        rt.block_on(async move {
            let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            tx.send(l.local_addr().unwrap().to_string()).unwrap();
            llvq_runtime::api::serve(l, state).await.unwrap();
        });
    });
    rx.recv().unwrap()
}
