//! `LLVQ_KV_CAPTURE`: what a decode row reads from the KV cache, dumped for an
//! offline scorer (`ops/kv_mass_score.py`). MEASUREMENT ONLY, never served.
//!
//! One window of a corpus goes through the eager engine one true token at a
//! time, teacher-forced, under whatever attention form the caller set on the
//! model (`bin/fusedrun` sets the kernel form on every row, the path of
//! `LLVQ_GQA_PPL=...,kernel0`). The first `ctx` tokens only fill the cache.
//! For each of the next `tokens` tokens, the post-RoPE query of every layer is
//! kept, `[n_heads][head_dim]` binary16. After the last token the cache of
//! every layer is dumped as stored: K, `[n_kv][L][head_dim]` binary16, and the
//! L2 norm of every V row, `[n_kv][L]` f32, with `L = ctx + tokens`.
//!
//! Which positions a row attends, read from `Block::attention_ctx`: the cache
//! is appended BEFORE any form attends, so the row of the token at absolute
//! position `p` attends positions `0..=p`, its own key included. Captured
//! token `t` (0-based) sits at `p = ctx + t` and attends the first
//! `ctx + t + 1` positions of the final dump; the cache is append-only, so
//! those positions hold the bytes that row read.
//!
//! A self-check rides along: for the last token, at layer `n_layers / 2`, the
//! attention output as the engine computed it (the input of `o_proj`) and that
//! layer's V as stored, so a scorer can recompute the output from the dump.

use std::path::Path;

use candle_core::{DType, Result, Tensor};

use crate::model::{Act, Capture, KvCache, Qwen3};

/// The corpora a capture reads, by the loaders of [`crate::corpus`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptureCorpus {
    /// WikiText-2 raw test, rows joined by blank lines: `LLVQ_GQA_PPL`'s text.
    Wikitext2Test,
    /// C4 validation, the EVALUATION shard (`corpus::C4Role::Evaluation`),
    /// documents joined by blank lines.
    C4,
    /// DCLM-edu, the shard the sealed files were CALIBRATED on
    /// (`corpus::dclm_edu_calibration`), documents joined by blank lines.
    DclmEdu,
}

impl CaptureCorpus {
    pub fn parse(v: &str) -> std::result::Result<Self, String> {
        match v {
            "wikitext2-test" => Ok(Self::Wikitext2Test),
            "c4" => Ok(Self::C4),
            "dclm-edu" => Ok(Self::DclmEdu),
            other => Err(format!("corpus {other:?}: accepted `wikitext2-test`, `c4`, `dclm-edu`")),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Wikitext2Test => "wikitext2-test",
            Self::C4 => "c4",
            Self::DclmEdu => "dclm-edu",
        }
    }

    /// The raw text, at least `min_chars` long where the corpus is read with
    /// a bound (WikiText-2 test is read whole).
    pub fn text(self, min_chars: usize) -> anyhow::Result<String> {
        Ok(match self {
            Self::Wikitext2Test => crate::corpus::wikitext2_test()?,
            Self::C4 => crate::corpus::c4_validation(min_chars)?,
            Self::DclmEdu => {
                let (text, stats) = crate::corpus::dclm_edu_calibration(min_chars)?;
                println!("  dclm-edu: {}", stats.report());
                text
            }
        })
    }
}

/// The value of `LLVQ_KV_CAPTURE`: `<ctx>,<tokens>,<corpus>,<window>`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvCaptureSpec {
    /// Tokens fed before the first captured one.
    pub ctx: usize,
    /// Tokens captured after them.
    pub tokens: usize,
    pub corpus: CaptureCorpus,
    /// Window index: tokens `window * (ctx + tokens)` to `(window + 1) * (ctx + tokens)` of the tokenized corpus.
    pub window: usize,
}

/// Most tokens a capture keeps: their queries are held on the host,
/// `tokens x 36 x 32 x 128 x 2` B at 4B, 302 MB at 1,024.
pub const MAX_TOKENS: usize = 1024;
/// Most positions a window may span (`ctx + tokens`).
pub const MAX_SPAN: usize = 32768;

impl KvCaptureSpec {
    /// Positions a window spans, and the length of the final dump.
    pub fn span(&self) -> usize {
        self.ctx + self.tokens
    }
}

pub fn parse_spec(v: &str) -> std::result::Result<KvCaptureSpec, String> {
    let parts: Vec<&str> = v.split(',').map(str::trim).collect();
    let [c, t, corpus, w] = parts.as_slice() else {
        return Err(format!(
            "LLVQ_KV_CAPTURE={v:?}: expected <ctx>,<tokens>,<corpus>,<window>, e.g. 8192,256,wikitext2-test,0"
        ));
    };
    let num = |s: &str| s.parse::<usize>().map_err(|e| format!("LLVQ_KV_CAPTURE={v:?}: {s:?}: {e}"));
    let spec = KvCaptureSpec {
        ctx: num(c)?,
        tokens: num(t)?,
        corpus: CaptureCorpus::parse(corpus).map_err(|e| format!("LLVQ_KV_CAPTURE={v:?}: {e}"))?,
        window: num(w)?,
    };
    if spec.ctx == 0 {
        return Err(format!("LLVQ_KV_CAPTURE={v:?}: a context of at least one token"));
    }
    if spec.tokens < 2 || spec.tokens > MAX_TOKENS {
        return Err(format!("LLVQ_KV_CAPTURE={v:?}: between 2 and {MAX_TOKENS} captured tokens"));
    }
    if spec.span() > MAX_SPAN {
        return Err(format!("LLVQ_KV_CAPTURE={v:?}: a window of {} positions, above {MAX_SPAN}", spec.span()));
    }
    Ok(spec)
}

/// The window's tokens, or a refusal naming what is missing: a corpus too
/// short is never silently shortened.
pub fn window_of<'a>(ids: &'a [u32], spec: &KvCaptureSpec) -> std::result::Result<&'a [u32], String> {
    let span = spec.span();
    let (lo, hi) = (spec.window * span, (spec.window + 1) * span);
    if ids.len() < hi {
        return Err(format!(
            "{}: window {} of {span} tokens needs tokens {lo}..{hi}, and the corpus as read holds {}",
            spec.corpus.name(),
            spec.window,
            ids.len()
        ));
    }
    Ok(&ids[lo..hi])
}

/// The capture hook: the queries of the captured tokens, and the self-check's
/// attention output. Reads no activation, so the blocks keep their paths.
struct QueryTap {
    on: bool,
    q: Vec<u16>,
    check_layer: usize,
    want_check: bool,
    check_out: Option<Vec<u16>>,
    calls: usize,
}

fn f16_bits(t: &Tensor) -> Result<Vec<u16>> {
    Ok(t.to_dtype(DType::F16)?.flatten_all()?.to_vec1::<half::f16>()?.into_iter().map(|h| h.to_bits()).collect())
}

impl Capture for QueryTap {
    fn on_activation(&mut self, layer: usize, act: Act, x: &Tensor) -> Result<()> {
        if self.want_check && act == Act::AttnOut && layer == self.check_layer {
            self.check_out = Some(f16_bits(x)?);
        }
        Ok(())
    }
    fn wants_activations(&self) -> bool {
        false
    }
    fn on_attention_query(&mut self, _layer: usize, q: &Tensor) -> Result<()> {
        if self.on {
            let (b, _, l, _) = q.dims4()?;
            if (b, l) != (1, 1) {
                candle_core::bail!("kv capture: a query of batch {b} and {l} rows; the capture takes decode rows only");
            }
            self.q.extend(f16_bits(q)?);
            self.calls += 1;
        }
        Ok(())
    }
}

/// What one window leaves: the queries, the caches, the self-check.
pub struct Captured {
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub ctx: usize,
    pub tokens: usize,
    /// `[tokens][n_layers][n_heads][head_dim]` binary16 bits.
    pub q: Vec<u16>,
    /// One cache a layer, `ctx + tokens` positions held.
    pub caches: Vec<KvCache>,
    pub check_layer: usize,
    /// The last token's attention output at `check_layer`, `[n_heads][head_dim]` binary16 bits.
    pub check_out: Vec<u16>,
    /// `-log p(next true token)` of captured token `t`, for `t < tokens - 1`
    /// (the last has no target inside the window). A sanity line, nothing more.
    pub nll: Vec<f64>,
}

/// Feed `window` (`ctx + tokens` ids) one token at a time through fresh caches
/// of the model's store and form, capturing as the module doc says.
/// `progress` is called with the number of tokens fed, every 1,024 and at the end.
pub fn capture(
    model: &Qwen3,
    window: &[u32],
    ctx: usize,
    tokens: usize,
    mut progress: impl FnMut(usize),
) -> anyhow::Result<Captured> {
    let span = ctx + tokens;
    anyhow::ensure!(window.len() == span, "kv capture: {} ids for a window of {span}", window.len());
    let cfg = model.config();
    let (n_layers, n_heads, n_kv, head_dim) =
        (cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim);
    let check_layer = n_layers / 2;
    let mut tap = QueryTap {
        on: false,
        q: Vec::with_capacity(tokens * n_layers * n_heads * head_dim),
        check_layer,
        want_check: false,
        check_out: None,
        calls: 0,
    };
    let mut caches = model.fresh_caches();
    let mut nll = Vec::with_capacity(tokens - 1);
    let dev = model.device().clone();
    for (p, &id) in window.iter().enumerate() {
        tap.on = p >= ctx;
        tap.want_check = p + 1 == span;
        let input = Tensor::from_slice(&[id], (1, 1), &dev)?;
        let h = model.hidden_cached(&input, p, &mut caches, &mut tap)?;
        if p >= ctx && p + 1 < span {
            nll.push(-crate::model::logp_of(&model.logits_last(&h)?, window[p + 1])?);
        }
        if (p + 1) % 1024 == 0 || p + 1 == span {
            dev.synchronize()?;
            progress(p + 1);
        }
    }
    anyhow::ensure!(
        tap.calls == tokens * n_layers,
        "kv capture: {} query calls, expected {tokens} tokens x {n_layers} layers",
        tap.calls
    );
    anyhow::ensure!(tap.q.len() == tokens * n_layers * n_heads * head_dim, "kv capture: {} query values", tap.q.len());
    for (i, c) in caches.iter().enumerate() {
        held_bits(c, span).map_err(|e| anyhow::anyhow!("kv capture: layer {i}: {e}"))?;
    }
    let check_out = tap.check_out.ok_or_else(|| anyhow::anyhow!("kv capture: the check layer's output was not seen"))?;
    anyhow::ensure!(check_out.len() == n_heads * head_dim, "kv capture: check output of {} values", check_out.len());
    Ok(Captured { n_layers, n_heads, n_kv, head_dim, ctx, tokens, q: tap.q, caches, check_layer, check_out, nll })
}

fn write_u16(path: &Path, v: &[u16]) -> std::io::Result<()> {
    let mut b = Vec::with_capacity(v.len() * 2);
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    std::fs::write(path, b)
}

fn write_f32(path: &Path, v: &[f32]) -> std::io::Result<()> {
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    std::fs::write(path, b)
}

/// What `meta.json` carries beyond the shapes: where the window came from.
pub struct Provenance<'a> {
    pub corpus: &'a str,
    pub window: usize,
    /// The window's ids, `ctx + tokens` of them.
    pub token_ids: &'a [u32],
    pub sealed_sha256: &'a str,
    /// The store and form the model ran, as their labels print.
    pub attention: &'a str,
}

/// The cache of one layer: K bits `[n_kv][L][hd]`, V bits, both as stored.
fn held_bits(c: &KvCache, span: usize) -> anyhow::Result<(Tensor, Tensor)> {
    let (k, v) = c.held()?.ok_or_else(|| anyhow::anyhow!("kv capture: an empty cache"))?;
    anyhow::ensure!(k.dim(2)? == span, "kv capture: a cache of {} positions, expected {span}", k.dim(2)?);
    Ok((k.contiguous()?, v.contiguous()?))
}

/// Write `dir` (created; refused if it exists): `q.bin`, `k-<layer>.bin`,
/// `vnorm-<layer>.bin`, `v-check.bin`, `out-check.bin`, `meta.json`.
/// Returns the bytes written.
pub fn write_capture(dir: &Path, cap: &Captured, prov: &Provenance) -> anyhow::Result<u64> {
    anyhow::ensure!(!dir.exists(), "kv capture: {} exists; a capture never overwrites", dir.display());
    let span = cap.ctx + cap.tokens;
    anyhow::ensure!(prov.token_ids.len() == span, "kv capture: {} token ids for {span} positions", prov.token_ids.len());
    std::fs::create_dir_all(dir)?;
    let mut bytes = 0u64;
    let mut put = |name: &str, f: &dyn Fn(&Path) -> std::io::Result<()>| -> anyhow::Result<()> {
        let p = dir.join(name);
        f(&p)?;
        bytes += std::fs::metadata(&p)?.len();
        Ok(())
    };
    put("q.bin", &|p| write_u16(p, &cap.q))?;
    put("out-check.bin", &|p| write_u16(p, &cap.check_out))?;
    for (l, c) in cap.caches.iter().enumerate() {
        let (k, v) = held_bits(c, span)?;
        let kb = f16_bits(&k)?;
        anyhow::ensure!(kb.len() == cap.n_kv * span * cap.head_dim, "kv capture: layer {l} K of {} values", kb.len());
        put(&format!("k-{l}.bin"), &|p| write_u16(p, &kb))?;
        let vn: Vec<f32> = v.to_dtype(DType::F32)?.sqr()?.sum_keepdim(3)?.sqrt()?.flatten_all()?.to_vec1::<f32>()?;
        anyhow::ensure!(vn.len() == cap.n_kv * span, "kv capture: layer {l} V norms of {} values", vn.len());
        put(&format!("vnorm-{l}.bin"), &|p| write_f32(p, &vn))?;
        if l == cap.check_layer {
            let vb = f16_bits(&v)?;
            put("v-check.bin", &|p| write_u16(p, &vb))?;
        }
    }
    let (nl, nh, nkv, hd, ctx, nt) = (cap.n_layers, cap.n_heads, cap.n_kv, cap.head_dim, cap.ctx, cap.tokens);
    let meta = serde_json::json!({
        "format": "llvq-kv-capture/1",
        "dtype": "binary16 little endian (q, k, v, out); f32 little endian (vnorm)",
        "n_layers": nl, "n_heads": nh, "n_kv": nkv, "head_dim": hd,
        "groups": nh / nkv,
        "ctx": ctx, "tokens": nt, "positions": span,
        "corpus": prov.corpus, "window": prov.window,
        "window_first_token": prov.window * span,
        "token_ids": prov.token_ids,
        "sealed_sha256": prov.sealed_sha256,
        "attention": prov.attention,
        "c": 1.0 / (hd as f64).sqrt(),
        "c_note": "score of key j for query head h: c * <q_h, k_j>, natural softmax; query head h reads KV head h / groups",
        "attended": "captured token t (0-based) sits at position ctx + t and attends positions 0..=ctx+t of the dump, its own key included (the cache is appended before attention)",
        "files": {
            "q.bin": [nt, nl, nh, hd],
            "k-<layer>.bin": [nkv, span, hd],
            "vnorm-<layer>.bin": [nkv, span],
            "v-check.bin": [nkv, span, hd],
            "out-check.bin": [nh, hd],
        },
        "check": {
            "token": nt - 1, "layer": cap.check_layer,
            "what": "out-check.bin is the attention output (input of o_proj) of the last captured token at this layer as the engine computed it; v-check.bin is this layer's V as stored",
        },
        "nll": cap.nll,
        "nll_note": "-log p(next true token) of captured tokens 0..tokens-2",
    });
    put("meta.json", &|p| std::fs::write(p, serde_json::to_string_pretty(&meta).expect("json")))?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_spec_parses_and_refuses() {
        let s = parse_spec("8192,256,wikitext2-test,0").unwrap();
        assert_eq!(s, KvCaptureSpec { ctx: 8192, tokens: 256, corpus: CaptureCorpus::Wikitext2Test, window: 0 });
        assert_eq!(s.span(), 8448);
        assert_eq!(parse_spec(" 8192 , 256 , c4 , 3 ").unwrap().corpus, CaptureCorpus::C4);
        assert_eq!(parse_spec("8192,256,dclm-edu,0").unwrap().corpus, CaptureCorpus::DclmEdu);
        for bad in [
            "",
            "8192,256,c4",
            "8192,256,c4,0,1",
            "8192,256,wikitext2,0",
            "0,256,c4,0",
            "8192,1,c4,0",
            "8192,1025,c4,0",
            "32000,1000,c4,0",
            "8192,x,c4,0",
            "8192,256,c4,-1",
        ] {
            assert!(parse_spec(bad).is_err(), "{bad:?} must be refused");
        }
    }

    #[test]
    fn a_window_is_taken_whole_or_refused() {
        let ids: Vec<u32> = (0..100).collect();
        let s = KvCaptureSpec { ctx: 20, tokens: 5, corpus: CaptureCorpus::C4, window: 3 };
        assert_eq!(window_of(&ids, &s).unwrap(), &ids[75..100]);
        let e = window_of(&ids, &KvCaptureSpec { window: 4, ..s }).unwrap_err();
        assert!(e.contains("needs tokens 100..125") && e.contains("holds 100"), "{e}");
    }

    fn read_u16(p: &Path) -> Vec<u16> {
        std::fs::read(p).unwrap().chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
    }
    fn read_f32(p: &Path) -> Vec<f32> {
        std::fs::read(p).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }
    fn f(b: u16) -> f64 {
        half::f16::from_bits(b).to_f64()
    }

    /// The attention output of one row in f64: `q` is `[nh][hd]` bits, `k`
    /// and `v` are `[nkv][span][hd]` bits, positions `0..len` attended.
    #[allow(clippy::too_many_arguments)]
    fn recompute(q: &[u16], k: &[u16], v: &[u16], nh: usize, nkv: usize, hd: usize, span: usize, len: usize) -> Vec<f64> {
        let c = 1.0 / (hd as f64).sqrt();
        let mut o = vec![0.0; nh * hd];
        for h in 0..nh {
            let g = h / (nh / nkv);
            let qh = &q[h * hd..][..hd];
            let s: Vec<f64> =
                (0..len).map(|j| c * (0..hd).map(|i| f(qh[i]) * f(k[(g * span + j) * hd + i])).sum::<f64>()).collect();
            let mx = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let w: Vec<f64> = s.iter().map(|x| (x - mx).exp()).collect();
            let z: f64 = w.iter().sum();
            for (j, wj) in w.iter().enumerate() {
                for i in 0..hd {
                    o[h * hd + i] += wj / z * f(v[(g * span + j) * hd + i]);
                }
            }
        }
        o
    }

    /// Largest absolute difference over the largest |engine output|.
    fn rel_err(o: &[f64], out: &[u16]) -> f64 {
        let scale = out.iter().map(|&b| f(b).abs()).fold(0.0, f64::max);
        o.iter().zip(out).map(|(a, &b)| (a - f(b)).abs()).fold(0.0, f64::max) / scale
    }

    /// A tiny model under the kernel form on every row (the CPU replay):
    /// the capture leaves the logits bit for bit, writes every file at its
    /// shape, its K prefix is the cache each row read, and the self-check's
    /// output is recomputed from the dumped query, K and V over positions
    /// `0..=p`, its own key included, and not over `0..p`.
    #[test]
    fn a_capture_is_the_rows_the_engine_read() {
        use crate::kvq::{AttnHeads, KvStore, GROW_CHUNK};
        let mut m = crate::model::tests_support::tiny_model_kernel();
        m.set_kv_store(KvStore::Grow(GROW_CHUNK));
        m.set_attn_heads(AttnHeads::Kernel);
        m.set_attn_kernel(Some(crate::attn::kernel_for(&candle_core::Device::Cpu, 2, 4, 128).unwrap()));
        m.set_kernel_from_for_measurement(0);
        let window: Vec<u32> = (0..14u32).map(|i| (i * 37 + 5) % 97).collect();
        let (ctx, tokens) = (9, 5);

        // The capture changes no output: the same feed with NoCapture.
        let mut plain = m.fresh_caches();
        let mut ref_logits = Vec::new();
        for (p, &id) in window.iter().enumerate() {
            let x = Tensor::from_slice(&[id], (1, 1), &candle_core::Device::Cpu).unwrap();
            let h = m.hidden_cached(&x, p, &mut plain, &mut crate::model::NoCapture).unwrap();
            ref_logits.push(m.logits_last(&h).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap());
        }
        let mut seen = Vec::new();
        let cap = capture(&m, &window, ctx, tokens, |n| seen.push(n)).unwrap();
        assert_eq!(seen, vec![14]);
        for t in 0..tokens - 1 {
            let p = ctx + t;
            let l = &ref_logits[p];
            let mx = l.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let z: f64 = l.iter().map(|&x| (x as f64 - mx).exp()).sum();
            let want = -(l[window[p + 1] as usize] as f64 - mx - z.ln());
            assert!((cap.nll[t] - want).abs() < 1e-4, "token {t}: nll {} against {want}", cap.nll[t]);
        }
        for (a, b) in cap.caches.iter().zip(&plain) {
            let (ka, va) = a.held().unwrap().unwrap();
            let (kb, vb) = b.held().unwrap().unwrap();
            assert_eq!(f16_bits(&ka).unwrap(), f16_bits(&kb).unwrap(), "K moved under the capture");
            assert_eq!(f16_bits(&va).unwrap(), f16_bits(&vb).unwrap(), "V moved under the capture");
        }

        let dir = std::env::temp_dir().join(format!("llvq-kvcap-{}-{}", std::process::id(), line!()));
        let _ = std::fs::remove_dir_all(&dir);
        let prov = Provenance { corpus: "c4", window: 0, token_ids: &window, sealed_sha256: "00", attention: "grow(256)+kernel" };
        let bytes = write_capture(&dir, &cap, &prov).unwrap();
        let e = write_capture(&dir, &cap, &prov).unwrap_err();
        assert!(e.to_string().contains("never overwrites"), "{e}");
        let (nl, nh, nkv, hd, span) = (2usize, 8usize, 2usize, 128usize, 14usize);
        let mut total = 0u64;
        let size = |n: &str| std::fs::metadata(dir.join(n)).unwrap().len();
        assert_eq!(size("q.bin"), (tokens * nl * nh * hd * 2) as u64);
        assert_eq!(size("out-check.bin"), (nh * hd * 2) as u64);
        assert_eq!(size("v-check.bin"), (nkv * span * hd * 2) as u64);
        total += size("q.bin") + size("out-check.bin") + size("v-check.bin") + size("meta.json");
        for l in 0..nl {
            assert_eq!(size(&format!("k-{l}.bin")), (nkv * span * hd * 2) as u64);
            assert_eq!(size(&format!("vnorm-{l}.bin")), (nkv * span * 4) as u64);
            total += size(&format!("k-{l}.bin")) + size(&format!("vnorm-{l}.bin"));
        }
        assert_eq!(bytes, total);
        let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("meta.json")).unwrap()).unwrap();
        assert_eq!(meta["positions"], 14);
        assert_eq!(meta["check"]["layer"], 1);
        assert_eq!(meta["token_ids"].as_array().unwrap().len(), 14);

        // The self-check, recomputed: last token, layer 1, positions 0..=13.
        let q = read_u16(&dir.join("q.bin"));
        let k = read_u16(&dir.join("k-1.bin"));
        let v = read_u16(&dir.join("v-check.bin"));
        let out = read_u16(&dir.join("out-check.bin"));
        let vn = read_f32(&dir.join("vnorm-1.bin"));
        let t = tokens - 1;
        let q_last = &q[((t * nl + 1) * nh) * hd..][..nh * hd];
        let with_own = rel_err(&recompute(q_last, &k, &v, nh, nkv, hd, span, span), &out);
        println!("self-check, 14 positions: rel err {with_own:e} over 0..=p");
        assert!(with_own < 4e-3, "the dumped rows do not give the engine's output: {with_own:e}");

        // The own key, where a tiny model can show it: three positions. Its
        // random weights give near-uniform attention and near-equal V rows, so
        // dropping the own key moves the output by 3x the rounding only, and
        // the query moves nothing (printed, not asserted). The query is checked
        // where attention is peaked: the scorer's self-check on the card's
        // `out-check.bin` (`ops/kv_mass_score.py`).
        let short = capture(&m, &window[..3], 1, 2, |_| ()).unwrap();
        let (k3, v3) = held_bits(&short.caches[1], 3).unwrap();
        let (k3, v3) = (f16_bits(&k3).unwrap(), f16_bits(&v3).unwrap());
        let q3 = |t: usize| &short.q[((t * nl + 1) * nh) * hd..][..nh * hd];
        let good = rel_err(&recompute(q3(1), &k3, &v3, nh, nkv, hd, 3, 3), &short.check_out);
        let no_own = rel_err(&recompute(q3(1), &k3, &v3, nh, nkv, hd, 3, 2), &short.check_out);
        let wrong_q = rel_err(&recompute(q3(0), &k3, &v3, nh, nkv, hd, 3, 3), &short.check_out);
        println!("self-check, 3 positions: rel err {good:e} over 0..=p, {no_own:e} over 0..p, {wrong_q:e} with the previous token's query");
        assert!(good < 4e-3, "three positions: {good:e}");
        assert!(no_own > 2.0 * good, "own key not distinguishable: {no_own:e} vs {good:e}");
        // V norms are the norms of the stored rows.
        for g in 0..nkv {
            for j in 0..span {
                let n = (0..hd).map(|i| f(v[(g * span + j) * hd + i]).powi(2)).sum::<f64>().sqrt();
                assert!((n - vn[g * span + j] as f64).abs() <= 1e-4 * n.max(1.0), "vnorm {g} {j}");
            }
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
