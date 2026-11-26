//! The prefix cache on disk behind the real server ([`llvq_runtime::prefix`]),
//! with a scripted model on the byte-level tokenizer whose state is the ids it
//! holds: the first read of a long static block saves it at a piece boundary,
//! a new conversation and a restarted server load it and read only the rest, a
//! damaged file is refused and the prompt read in full, a changed system
//! prompt loads nothing, and a server without the cache writes nothing.

mod common;
#[path = "common/ink.rs"]
mod ink;

use candle_core::{Device, Tensor};
use llvq_llm::inkchat::InkChat;
use llvq_runtime::client::Client;
use llvq_runtime::engine::{Built, Loaded, Loader};
use llvq_runtime::inkling::{InkEngine, TurnLm};
use llvq_runtime::library::{Configs, ModelName};
use llvq_runtime::prefix::{Key, PrefixCache, FORMAT, MIN_TOKENS};
use llvq_runtime::store::{Manifest, Store};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

const WORDS: &str = "<|message_model|><|content_text|>the cat sat<|end_message|><|content_model_end_sampling|>";

type Feeds = Arc<Mutex<Vec<Vec<u32>>>>;

struct Script {
    ctx: Vec<u32>,
    feeds: Feeds,
    words: Vec<u32>,
}

impl TurnLm for Script {
    type Mark = usize;

    fn feed(&mut self, ids: &[u32]) -> anyhow::Result<Tensor> {
        self.feeds.lock().unwrap().push(ids.to_vec());
        self.ctx.extend_from_slice(ids);
        let i = (1..self.words.len()).rev().find(|&i| self.ctx.ends_with(&self.words[..i])).unwrap_or(0);
        let mut v = vec![0f32; ink::VOCAB];
        v[self.words[i] as usize] = 1.0;
        Ok(Tensor::from_vec(v, (1, ink::VOCAB), &Device::Cpu)?)
    }

    fn held(&self) -> usize {
        self.ctx.len()
    }

    fn mark(&self) -> usize {
        self.ctx.len()
    }

    fn rollback(&mut self, m: &usize) -> anyhow::Result<()> {
        anyhow::ensure!(*m <= self.ctx.len(), "a mark past the cache");
        self.ctx.truncate(*m);
        Ok(())
    }

    fn reset(&mut self) {
        self.ctx.clear();
    }

    fn save_state(&self, w: &mut dyn std::io::Write) -> anyhow::Result<()> {
        w.write_all(&(self.ctx.len() as u64).to_le_bytes())?;
        for t in &self.ctx {
            w.write_all(&t.to_le_bytes())?;
        }
        Ok(())
    }

    fn load_state(&mut self, r: &mut dyn std::io::Read) -> anyhow::Result<()> {
        let mut b8 = [0u8; 8];
        r.read_exact(&mut b8)?;
        let n = u64::from_le_bytes(b8) as usize;
        let mut b = vec![0u8; 4 * n];
        r.read_exact(&mut b)?;
        self.ctx = b.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        Ok(())
    }
}

struct ScriptLoader {
    feeds: Feeds,
    prefix: Option<PrefixCache>,
}

impl Loader for ScriptLoader {
    fn load(&self, _: &ModelName, _: &Manifest, _: &Store) -> anyhow::Result<Loaded> {
        anyhow::bail!("not a qwen3")
    }

    fn build(&self, _: &ModelName, _: &Manifest, _: &Store) -> anyhow::Result<Built> {
        let chat = InkChat::new(ink::byte_tokenizer())?;
        let words = chat.tokenizer().encode(WORDS, false).unwrap().get_ids().to_vec();
        let lm = Script { ctx: Vec::new(), feeds: self.feeds.clone(), words };
        Ok(Built::Inkling(InkEngine::with_prefix(lm, chat, 16_384, "script", None, self.prefix.clone())))
    }
}

fn key() -> Key {
    Key {
        format: FORMAT,
        family: "inkling2".into(),
        model_sha256: "1".repeat(64),
        files_sha256: "3".repeat(64),
        build_sha256: "2".repeat(64),
        backend: "script".into(),
        stack: "fused".into(),
        prompt_piece: 512,
        score_rows: 128,
        kv_slack: 1024,
        bias: "host".into(),
    }
}

fn store_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("llvq-runtime-test-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

/// A server over the store in `dir` (the same files across "restarts"), the
/// prefix cache on or off.
fn serve(dir: &Path, on: bool) -> (Client, Feeds) {
    let store = Store::at(dir);
    let m = Manifest {
        schema: 1,
        sha256: "1".repeat(64),
        size: 1,
        repo: "local".into(),
        revision: "local".into(),
        file: "inkling.json".into(),
        family: "inkling2".into(),
        parameter_size: String::new(),
        bits_per_param: String::new(),
        configs: Configs { cuda: None, metal: None, cpu: None, cuda_batch: None, cuda_eager: None },
        modified_at: 1_762_387_200,
    };
    store.write_manifest(&ModelName::parse("ink:test").unwrap(), &m).unwrap();
    let feeds = Arc::new(Mutex::new(Vec::new()));
    let prefix = on.then(|| PrefixCache::new(dir.join("prefix"), 1 << 30, key()));
    let loader = ScriptLoader { feeds: feeds.clone(), prefix };
    let host = common::serve_with(Arc::new(store), "http://127.0.0.1:9", Arc::new(loader));
    (Client::new(&host), feeds)
}

/// A system prompt of about 2,300 bytes, so tokens on the byte-level
/// tokenizer, with `word` in its first sentence.
fn system(word: &str) -> Value {
    let mut s = format!("You are a {word} coding agent working in a repository.");
    for k in 0..40 {
        s.push_str(&format!(" Rule {k}: read the file before you change it, and run the tests after."));
    }
    json!({"role": "system", "content": s})
}

fn tools() -> Value {
    json!([{"type": "function", "function": {"name": "list_dir", "description": "List a directory.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}])
}

/// A native chat with the static block of `word` and the user message
/// `user`: the answer and the count of cached tokens.
fn ask(c: &Client, word: &str, user: &str) -> (String, u64, u64) {
    let body = json!({"model": "ink:test", "stream": false, "tools": tools(),
        "messages": [system(word), {"role": "user", "content": user}]});
    let r = c.send("POST", "/api/chat", &body).unwrap();
    let content = r["message"]["content"].as_str().unwrap_or_default().to_string();
    (content, r["prompt_eval_count"].as_u64().unwrap(), r["prompt_eval_cached_count"].as_u64().unwrap())
}

fn prefix_files(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir.join("prefix"))
        .map(|rd| rd.filter_map(|e| e.ok()).map(|e| e.path()).collect())
        .unwrap_or_default();
    v.sort();
    v
}

/// The `b` a prefix file's name carries.
fn b_of(p: &Path) -> usize {
    p.file_name().unwrap().to_string_lossy().split('-').nth(1).unwrap().parse().unwrap()
}

fn fed_since(feeds: &Feeds, from: usize) -> Vec<Vec<u32>> {
    feeds.lock().unwrap()[from..].to_vec()
}

#[test]
fn a_static_block_is_read_once_across_conversations_and_restarts() {
    let dir = store_dir("ink-prefix");

    // P-a: a cold server reads the whole prompt and writes one file, cut at a piece boundary.
    let (c, feeds) = serve(&dir, true);
    let (a, n, cached) = ask(&c, "careful", "list the files");
    assert_eq!(a, "the cat sat");
    assert_eq!(cached, 0);
    let files = prefix_files(&dir);
    assert_eq!(files.len(), 1, "{files:?}");
    let b = b_of(&files[0]);
    assert!(b >= MIN_TOKENS && b.is_multiple_of(512) && (b as u64) < n, "b {b} of {n}");
    let first = fed_since(&feeds, 0);
    assert_eq!(first[0].len(), b, "the first read is cut at the save point");
    let mut block = first[0].clone();
    block.extend_from_slice(&first[1]);
    assert_eq!(block.len() as u64, n, "the first read is the whole prompt, in two feeds");

    // P-c: the same server, a new conversation: the file is loaded, only the rest is read.
    let k = feeds.lock().unwrap().len();
    let (a, n2, cached) = ask(&c, "careful", "show the readme");
    assert_eq!(a, "the cat sat");
    assert_eq!(cached as usize, b, "a new conversation loads the static block");
    let rest = fed_since(&feeds, k);
    let read = rest.iter().find(|f| f.len() > 1).map_or(0, Vec::len);
    assert_eq!(read as u64, n2 - b as u64, "only what follows the static block is read");

    // P-b: a restarted server, the same static block and another message.
    let (c, feeds) = serve(&dir, true);
    let (a, n3, cached) = ask(&c, "careful", "what does main do");
    assert_eq!(a, "the cat sat");
    assert_eq!(cached as usize, b);
    assert_eq!(fed_since(&feeds, 0)[0].len() as u64, n3 - b as u64);
    assert_eq!(prefix_files(&dir).len(), 1, "a loaded prefix is not saved again");

    // P-d: one byte damaged, a restarted server: refused, the prompt read in full, the file written again.
    let mut bytes = std::fs::read(&files[0]).unwrap();
    let at = bytes.len() - 40;
    bytes[at] ^= 0x01;
    std::fs::write(&files[0], &bytes).unwrap();
    let (c, feeds) = serve(&dir, true);
    let (a, n4, cached) = ask(&c, "careful", "what does main do");
    assert_eq!(a, "the cat sat");
    assert_eq!(cached, 0, "a damaged file is not loaded");
    assert_eq!(fed_since(&feeds, 0)[0].len(), b, "the full read saves again at b");
    assert_eq!(fed_since(&feeds, 0)[0].len() + fed_since(&feeds, 0)[1].len(), n4 as usize);
    assert_ne!(std::fs::read(&files[0]).unwrap(), bytes, "the damaged file was not replaced");
    let (_, _, cached) = ask(&c, "careful", "and the tests");
    assert_eq!(cached as usize, b, "the file written again loads");

    // P-e: the system prompt changed by one word: nothing loaded, a second file.
    let (_, _, cached) = ask(&c, "patient", "list the files");
    assert_eq!(cached, 0, "another static block loads nothing");
    assert_eq!(prefix_files(&dir).len(), 2);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_switch_off_writes_and_loads_nothing() {
    let dir = store_dir("ink-prefix-off");
    let (c, feeds) = serve(&dir, false);
    let (_, n, cached) = ask(&c, "careful", "list the files");
    assert_eq!(cached, 0);
    let k = feeds.lock().unwrap().len();
    assert_eq!(fed_since(&feeds, 0)[0].len() as u64, n, "the prompt is read in one piece, as without the cache");
    let (_, n2, cached) = ask(&c, "careful", "show the readme");
    assert_eq!(cached, 0);
    let read = fed_since(&feeds, k).into_iter().find(|f| f.len() > 1).map_or(0, |f| f.len());
    assert_eq!(read as u64, n2, "a new conversation is read in full, as without the cache");
    assert!(!dir.join("prefix").exists(), "a server without the cache made its directory");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A static block under 1,024 tokens is not saved.
#[test]
fn a_short_static_block_is_not_saved() {
    let dir = store_dir("ink-prefix-short");
    let (c, _) = serve(&dir, true);
    let body = json!({"model": "ink:test", "stream": false,
        "messages": [{"role": "system", "content": "You are brief."}, {"role": "user", "content": "hi"}]});
    c.send("POST", "/api/chat", &body).unwrap();
    assert!(prefix_files(&dir).is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}
