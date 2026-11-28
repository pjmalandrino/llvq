//! The unit tests of the layer pieces: the convolution state, the tiny random stack end to end and
//! step by step, and the static convolution against a direct sum.
use super::*;

/// The state kept after a prompt of several rows owns its `k - 1` rows and nothing else: a view into the
/// context would keep `[T + k - 1, C]` alive behind it (2.56 GB at 2,000 tokens on the real model).
#[test]
fn conv_state_after_a_prompt_owns_its_rows() -> Result<()> {
    let dev = Device::Cpu;
    let (t, k, c) = (10usize, 4usize, 6usize);
    let x = Tensor::arange(0f32, (t * c) as f32, &dev)?.reshape((t, c))?;
    let mut state = ConvState::Empty;
    let ctx = conv_context(&x, k, &mut state)?;
    assert_eq!(ctx.dims(), [t + k - 1, c]);
    let ConvState::Block(s) = &state else { panic!("a block state after several rows") };
    assert_eq!(s.dims(), [k - 1, c]);
    assert_eq!(s.layout().start_offset(), 0, "the state is a view into the context");
    assert_eq!(s.to_vec2::<f32>()?, ctx.narrow(0, t, k - 1)?.to_vec2::<f32>()?);
    Ok(())
}

fn tiny_cfg() -> InklingConfig {
    let v = serde_json::json!({
        "hidden_size": 48, "num_hidden_layers": 12, "vocab_size": 64, "unpadded_vocab_size": 60,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
        "swa_head_dim": 8, "swa_num_attention_heads": 4, "swa_num_key_value_heads": 2,
        "sliding_window_size": 4, "d_rel": 4, "rel_extent": 6,
        "log_scaling_n_floor": 8192, "log_scaling_alpha": 0.1, "rms_norm_eps": 1e-6,
        "qk_l2_clip_threshold": 22.6, "rel_l2_clip_threshold": 8.0, "sconv_kernel_size": 4,
        "final_logit_softcapping": 30.0, "n_layers_per_attn_res_block": 6,
        "n_routed_experts": 8, "num_experts_per_tok": 2, "intermediate_size": 24,
        "mlp_gate_softcap": 4.0, "mlp_up_softcap": 25.0, "route_scale": 8.0,
        "loop_config": {"core_start": 6, "core_end": 12, "num_passes": 3, "gate_rank": 4}
    });
    // One file a call: the tests of this module run in parallel threads of one process.
    static CALL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = CALL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("inkling_tiny_{}_{n}.json", std::process::id()));
    std::fs::write(&path, v.to_string()).unwrap();
    let c = InklingConfig::from_file(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    c
}

struct RandExperts(Vec<(Tensor, Tensor)>);
impl ExpertSource for RandExperts {
    fn count(&self) -> usize {
        self.0.len()
    }
    fn expert(&self, e: usize, _dev: &Device) -> Result<(Tensor, Tensor)> {
        Ok(self.0[e].clone())
    }
}

fn r(shape: &[usize]) -> Tensor {
    Tensor::randn(0f32, 0.2, shape, &Device::Cpu).unwrap()
}

fn layer(cfg: &InklingConfig, l: usize) -> LayerWeights {
    let h = cfg.hidden_size;
    let sliding = cfg.is_sliding(l);
    let (nh, nkv, hd) = cfg.attn_shape(l);
    let conv = |c: usize| {
        if sliding {
            KvConv::Static(r(&[c, 4]))
        } else {
            KvConv::Dynamic { static_w: r(&[4, c]), weight_proj: r(&[nkv * 4, h]) }
        }
    };
    let rank = cfg.loop_config.gate_rank;
    let mg = || MessageGate { down: r(&[rank, 2 * h]), forget: r(&[h, rank]), gain: r(&[h, rank]) };
    let mut lw = LayerWeights {
        index: l,
        sliding,
        boundary: cfg.is_block_boundary(l),
        attention_input: if l == 0 { None } else { Some(r(&[if sliding { 1 } else { 3 }, h])) },
        mlp_input: r(&[1, h]),
        norms_in: (0..if sliding { 1 } else { 3 }).map(|_| r(&[h])).collect(),
        post_attention_layernorm: r(&[h]),
        attn: AttnWeights {
            heads: nh,
            kv_heads: nkv,
            head_dim: hd,
            sliding,
            rel_extent: cfg.layer_rel_extent(l),
            wq: r(&[nh * hd, h]),
            q_scale: r(&[nh, hd]),
            wk: r(&[nkv * hd, h]),
            k_conv: conv(nkv * hd),
            wv: r(&[nkv * hd, h]),
            v_conv: conv(nkv * hd),
            gate: r(&[nh * hd, h]),
            wo: r(&[h, nh * hd]),
            sink: r(&[nh, h]),
            wr: r(&[nh * cfg.d_rel, h]),
            rel_scale: r(&[nh, cfg.d_rel]),
            rel_proj: r(&[cfg.d_rel, cfg.layer_rel_extent(l)]),
            #[cfg(ink_glue)]
            dense: None,
            cpu: None,
        },
        post_attention_convolution: r(&[h, 4]),
        post_mlp_convolution: r(&[h, 4]),
        router_w: r(&[cfg.n_routed_experts, h]),
        router_bias_host: vec![],
        router_bias: r(&[cfg.n_routed_experts]),
        message_gates: if cfg.is_loop_layer(l) {
            Some(MessageGates { q: mg(), k: mg(), v: mg(), mlp: mg() })
        } else {
            None
        },
    };
    lw.router_bias_host = lw.router_bias.to_vec1().unwrap();
    lw
}

/// Layers 0 to 6 of the tiny config on `x`, layers 6 (sliding loop) and
/// 11 (full loop) with messages; one slot cache per layer.
fn run_tiny(
    cfg: &InklingConfig,
    layers: &[LayerWeights],
    ex: &RandExperts,
    x: &Tensor,
    msgs: &[Tensor],
    caches: &mut [SlotCache],
) -> Result<Tensor> {
    let mut h = x.clone();
    let mut states = vec![];
    let t = x.dim(0)?;
    let msgs: Vec<Tensor> = msgs.iter().map(|m| m.narrow(0, m.dim(0).unwrap() - t, t).unwrap()).collect();
    for (lw, c) in layers.iter().zip(caches.iter_mut()) {
        let m = if lw.message_gates.is_some() { Some(&msgs[..]) } else { None };
        let (next, routing) = lw.forward(cfg, &h, &states, m, ex, c)?;
        assert_eq!(routing.indices.len(), t);
        if lw.boundary {
            states.push(h.clone());
        }
        h = next;
    }
    Ok(h)
}

fn tiny_stack() -> (InklingConfig, Vec<LayerWeights>, RandExperts) {
    let cfg = tiny_cfg();
    let (h, i) = (cfg.hidden_size, cfg.intermediate_size);
    let ex = RandExperts((0..cfg.n_routed_experts).map(|_| (r(&[2 * i, h]), r(&[h, i]))).collect());
    let layers: Vec<LayerWeights> = [0, 1, 2, 3, 4, 5, 6, 11].iter().map(|&l| layer(&cfg, l)).collect();
    (cfg, layers, ex)
}

/// Pre-loop layers 0 (sliding, boundary) to 5 (full) and two loop layers
/// with messages run end to end on random weights, finite, and causal:
/// changing the last token moves no earlier row beyond rounding.
#[test]
fn tiny_stack_runs_and_is_causal() -> Result<()> {
    let (cfg, layers, ex) = tiny_stack();
    let h = cfg.hidden_size;
    let t = 7;
    let x = r(&[t, h]);
    let msgs: Vec<Tensor> = (0..4).map(|_| r(&[t, h])).collect();
    let fresh = || vec![SlotCache::default(); layers.len()];
    let y = run_tiny(&cfg, &layers, &ex, &x, &msgs, &mut fresh())?;
    assert!(y.flatten_all()?.to_vec1::<f32>()?.iter().all(|v| v.is_finite()));
    let x2 = Tensor::cat(&[x.narrow(0, 0, t - 1)?, r(&[1, h])], 0)?;
    let y2 = run_tiny(&cfg, &layers, &ex, &x2, &msgs, &mut fresh())?;
    // Not bit for bit: when the last token routes elsewhere, an expert's
    // token batch changes size and the matmul may round its rows
    // differently. A causal leak moves rows by O(1), not by an ulp.
    let (d, scale) = max_abs_err(&y.narrow(0, 0, t - 1)?, &y2.narrow(0, 0, t - 1)?)?;
    assert!(d <= 1e-6 * scale, "an earlier row moved by {d:e} when only the last token changed");
    Ok(())
}

/// The decode path: a prompt of `t - 2` rows then two single-row steps on
/// the same slot caches give the rows a single call over `t` gives, past
/// the sliding window (4) and the full layers' relative extent (6).
#[test]
fn steps_on_the_cache_match_one_call() -> Result<()> {
    let (cfg, layers, ex) = tiny_stack();
    let h = cfg.hidden_size;
    let t = 9;
    let x = r(&[t, h]);
    let msgs: Vec<Tensor> = (0..4).map(|_| r(&[t, h])).collect();
    let whole = run_tiny(&cfg, &layers, &ex, &x, &msgs, &mut vec![SlotCache::default(); layers.len()])?;
    let mut caches = vec![SlotCache::default(); layers.len()];
    let msg_upto = |n: usize| -> Vec<Tensor> { msgs.iter().map(|m| m.narrow(0, 0, n).unwrap()).collect() };
    let mut rows = vec![run_tiny(&cfg, &layers, &ex, &x.narrow(0, 0, t - 2)?, &msg_upto(t - 2), &mut caches)?];
    for n in [t - 1, t] {
        rows.push(run_tiny(&cfg, &layers, &ex, &x.narrow(0, n - 1, 1)?, &msg_upto(n), &mut caches)?);
    }
    assert_eq!(caches[0].len(), t);
    let stepped = Tensor::cat(&rows, 0)?;
    let (d, scale) = max_abs_err(&stepped, &whole)?;
    assert!(d <= 1e-5 * scale, "decode drifts from the single call: {d:e} of {scale:e}");
    Ok(())
}

/// The ops the forward needs exist on Metal (`gather`, `index_add`,
/// `index_select`, the BF16 widening of the head) and the tiny stack, a
/// prompt then two steps, agrees with the CPU. Uses the GPU: run it alone,
/// under the guard's lock.
#[cfg(feature = "metal")]
#[test]
#[ignore = "uses the GPU; run under the guard's lock"]
fn tiny_stack_on_metal_matches_cpu() -> Result<()> {
    let (cfg, layers, ex) = tiny_stack();
    let mtl = Device::new_metal(0)?;
    let to = |t: &Tensor| t.to_device(&mtl).unwrap();
    let mv = |w: &Tensor| to(w);
    let move_layer = |lw: &LayerWeights| LayerWeights {
        index: lw.index,
        sliding: lw.sliding,
        boundary: lw.boundary,
        attention_input: lw.attention_input.as_ref().map(mv),
        mlp_input: mv(&lw.mlp_input),
        norms_in: lw.norms_in.iter().map(mv).collect(),
        post_attention_layernorm: mv(&lw.post_attention_layernorm),
        attn: AttnWeights {
            heads: lw.attn.heads,
            kv_heads: lw.attn.kv_heads,
            head_dim: lw.attn.head_dim,
            sliding: lw.attn.sliding,
            rel_extent: lw.attn.rel_extent,
            wq: mv(&lw.attn.wq),
            q_scale: mv(&lw.attn.q_scale),
            wk: mv(&lw.attn.wk),
            k_conv: match &lw.attn.k_conv {
                KvConv::Static(w) => KvConv::Static(mv(w)),
                KvConv::Dynamic { static_w, weight_proj } => {
                    KvConv::Dynamic { static_w: mv(static_w), weight_proj: mv(weight_proj) }
                }
            },
            wv: mv(&lw.attn.wv),
            v_conv: match &lw.attn.v_conv {
                KvConv::Static(w) => KvConv::Static(mv(w)),
                KvConv::Dynamic { static_w, weight_proj } => {
                    KvConv::Dynamic { static_w: mv(static_w), weight_proj: mv(weight_proj) }
                }
            },
            gate: mv(&lw.attn.gate),
            wo: mv(&lw.attn.wo),
            sink: mv(&lw.attn.sink),
            wr: mv(&lw.attn.wr),
            rel_scale: mv(&lw.attn.rel_scale),
            rel_proj: mv(&lw.attn.rel_proj),
            #[cfg(ink_glue)]
            dense: None,
            cpu: None,
        },
        post_attention_convolution: mv(&lw.post_attention_convolution),
        post_mlp_convolution: mv(&lw.post_mlp_convolution),
        router_w: mv(&lw.router_w),
        router_bias: mv(&lw.router_bias),
        router_bias_host: lw.router_bias_host.clone(),
        message_gates: lw.message_gates.as_ref().map(|g| {
            let m = |x: &MessageGate| MessageGate { down: mv(&x.down), forget: mv(&x.forget), gain: mv(&x.gain) };
            MessageGates { q: m(&g.q), k: m(&g.k), v: m(&g.v), mlp: m(&g.mlp) }
        }),
    };
    let mlayers: Vec<LayerWeights> = layers.iter().map(move_layer).collect();
    let mex = RandExperts(ex.0.iter().map(|(a, b)| (to(a), to(b))).collect());
    let h = cfg.hidden_size;
    let t = 9;
    let x = r(&[t, h]);
    let msgs: Vec<Tensor> = (0..4).map(|_| r(&[t, h])).collect();
    let cpu = run_tiny(&cfg, &layers, &ex, &x, &msgs, &mut vec![SlotCache::default(); layers.len()])?;
    let mmsgs: Vec<Tensor> = msgs.iter().map(to).collect();
    let mut caches = vec![SlotCache::default(); layers.len()];
    let upto = |n: usize| -> Vec<Tensor> { mmsgs.iter().map(|m| m.narrow(0, 0, n).unwrap()).collect() };
    let mut rows = vec![run_tiny(&cfg, &mlayers, &mex, &to(&x.narrow(0, 0, t - 2)?), &upto(t - 2), &mut caches)?];
    for n in [t - 1, t] {
        rows.push(run_tiny(&cfg, &mlayers, &mex, &to(&x.narrow(0, n - 1, 1)?), &upto(n), &mut caches)?);
    }
    let metal = Tensor::cat(&rows, 0)?.to_device(&Device::Cpu)?;
    let (d, scale) = max_abs_err(&metal, &cpu)?;
    assert!(d <= 1e-4 * scale, "Metal drifts from the CPU: {d:e} of {scale:e}");
    // the head's BF16 widening
    let w = Tensor::randn(0f32, 1.0, (8, h), &Device::Cpu)?.to_dtype(DType::BF16)?;
    let a = w.to_dtype(DType::F32)?;
    let b = w.to_device(&mtl)?.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
    assert_eq!(max_abs_err(&a, &b)?.0, 0.0, "BF16 to f32 on Metal");
    Ok(())
}

/// The static convolution against a direct transcription of `F.conv1d`
/// with left padding K - 1 (lines 316-322), plus residual and scale.
#[test]
fn short_conv_matches_direct_sum() -> Result<()> {
    let (t, c) = (6, 3);
    let x = r(&[t, c]);
    let w = r(&[c, 4]);
    let ours = short_conv(&x, &w, &mut ConvState::Empty)?.to_vec2::<f32>()?;
    let xv = x.to_vec2::<f32>()?;
    let wv = w.to_vec2::<f32>()?;
    for ti in 0..t {
        for ci in 0..c {
            let mut s = 0f32;
            for (k, wk) in wv[ci].iter().enumerate() {
                let src = ti as i64 + k as i64 - 3;
                if src >= 0 {
                    s += wk * xv[src as usize][ci];
                }
            }
            let want = (s + xv[ti][ci]) / 5f32.sqrt();
            assert!((ours[ti][ci] - want).abs() <= 1e-5, "t {ti} c {ci}: {} vs {want}", ours[ti][ci]);
        }
    }
    Ok(())
}
