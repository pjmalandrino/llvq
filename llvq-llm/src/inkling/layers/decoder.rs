//! One decoder layer: the message gates of the loop layers, [`LayerWeights`] with its loaders, the
//! candle forward and the fused one-row forward. One file because the three forwards share the
//! weights and the order of the reference's steps.
use super::*;

/// `Inkling2LoopMessageGate`, lines 79-96: `(1 - forget) * h + gain * message`,
/// both gates `tanh` of a rank-`gate_rank` projection of the two unweighted-
/// normed inputs side by side. Names per `conversion_mapping.py` 2685-2697.
pub struct MessageGate {
    /// `down_weight` `[rank, 2H]`.
    pub down: Tensor,
    /// `forget_weight` `[H, rank]`.
    pub forget: Tensor,
    /// `gain_weight` `[H, rank]` (`message_gate_proj`).
    pub gain: Tensor,
}

impl MessageGate {
    pub fn forward(&self, h: &Tensor, message: &Tensor, eps: f64) -> Result<Tensor> {
        let g = Tensor::cat(&[rms_unweighted(h, eps)?, rms_unweighted(message, eps)?], 1)?;
        let g = linear(&g, &self.down)?;
        let forget = linear(&g, &self.forget)?.tanh()?;
        let gain = linear(&g, &self.gain)?.tanh()?;
        (forget.neg()? + 1.0)?.mul(h)? + gain.mul(message)?
    }
}

/// The four message gates of a loop layer: query, key, value, MLP (lines 931-934).
pub struct MessageGates {
    pub q: MessageGate,
    pub k: MessageGate,
    pub v: MessageGate,
    pub mlp: MessageGate,
}

/// The weights of a decoder layer: sliding (lines 856-873, loop variant
/// 913-934) or full (lines 1000-1019, loop variant 1065-1088).
pub struct LayerWeights {
    pub index: usize,
    pub sliding: bool,
    pub boundary: bool,
    /// `[nq, H]`: 1 query in sliding layers, 3 in full ones; absent in layer 0 (lines 862, 1006).
    pub attention_input: Option<Tensor>,
    pub mlp_input: Tensor,
    /// `input_layernorm` in sliding layers; `q/k/v_input_layernorm` in full ones.
    pub norms_in: Vec<Tensor>,
    pub post_attention_layernorm: Tensor,
    pub attn: AttnWeights,
    pub post_attention_convolution: Tensor,
    pub post_mlp_convolution: Tensor,
    pub router_w: Tensor,
    pub router_bias: Tensor,
    /// `router_bias` read to the host once, at load.
    pub router_bias_host: Vec<f32>,
    /// Loop layers only (`controllers.*` in the file).
    pub message_gates: Option<MessageGates>,
}

fn conv_weight(ck: &NativeCheckpoint, name: &str, dev: &Device) -> Result<Tensor> {
    let w = ck.f32(name, dev)?; // [C, 1, K]
    let (c, one, k) = w.dims3()?;
    if one != 1 {
        candle_core::bail!("{name}: [{c}, {one}, {k}], expected a depthwise kernel");
    }
    w.reshape((c, k))
}

impl LayerWeights {
    /// Load layer `l` from the native file, names per `conversion_mapping.py`
    /// lines 353-416 (`inkling2_text`) and 2679-2706 (`inkling2_loop_text`).
    pub fn load(ck: &NativeCheckpoint, cfg: &InklingConfig, l: usize, dev: &Device) -> Result<Self> {
        Self::load_impl(ck, cfg, l, dev, false)
    }

    /// [`Self::load`] with the attention through the kernels of
    /// `super::attn_dense` (`kernels`, compiled once for the model): the seven
    /// attention matrices are read from the file in BF16 straight into the
    /// kernels' buffers, and no f32 copy of them is put on `dev`. With `row`,
    /// the scores over the cache run through `super::attn_scores` too.
    #[cfg(ink_glue)]
    pub fn load_dense(
        ck: &NativeCheckpoint,
        cfg: &InklingConfig,
        l: usize,
        dev: &Device,
        kernels: &std::sync::Arc<super::attn_dense::AttnDense>,
        row: Option<&std::sync::Arc<super::attn_dense::RowKernel>>,
    ) -> Result<Self> {
        let mut lw = Self::load_impl(ck, cfg, l, dev, true)?;
        let a = &mut lw.attn;
        let w = super::attn_dense::AttnDenseWeights::new(a, cfg, dev)?;
        let empty = Tensor::zeros(0, DType::F32, &Device::Cpu)?;
        for m in [&mut a.wq, &mut a.wr, &mut a.sink, &mut a.gate, &mut a.wk, &mut a.wv, &mut a.wo] {
            *m = empty.clone();
        }
        let mut fused = super::attn_dense::FusedAttn::new(kernels.clone(), w, dev);
        fused.row = row.cloned();
        a.dense = Some(fused);
        Ok(lw)
    }

    /// [`Self::load`] on the CPU with the attention products through the
    /// CPU kernels: the seven attention matrices are read from the file as
    /// it stores them and held in BF16 ([`CpuAttnProj`]), and no f32 copy of
    /// them is made. A file that stores them otherwise than in BF16 keeps
    /// them in f32 on candle, as [`Self::load`] does.
    pub fn load_cpu(ck: &NativeCheckpoint, cfg: &InklingConfig, l: usize) -> Result<Self> {
        let mut lw = Self::load_impl(ck, cfg, l, &Device::Cpu, true)?;
        let a = &mut lw.attn;
        let mats = [&mut a.wq, &mut a.wr, &mut a.sink, &mut a.gate, &mut a.wk, &mut a.wv, &mut a.wo];
        if mats.iter().any(|m| m.dtype() != DType::BF16) {
            for m in mats {
                *m = m.to_dtype(DType::F32)?;
            }
            return Ok(lw);
        }
        let p = |s: &str, t: &Tensor| crate::fused_cpu::CpuBf16::from_tensor(format!("layers.{l}.attn.{s}"), t);
        a.cpu = Some(CpuAttnProj {
            wq: p("wq_du", &a.wq)?,
            wr: p("wr_du", &a.wr)?,
            sink: p("attn_sink_proj", &a.sink)?,
            gate: p("gate_attn", &a.gate)?,
            wk: p("wk_dv", &a.wk)?,
            wv: p("wv_dv", &a.wv)?,
            wo: p("wo_ud", &a.wo)?,
        });
        let empty = Tensor::zeros(0, DType::F32, &Device::Cpu)?;
        for m in [&mut a.wq, &mut a.wr, &mut a.sink, &mut a.gate, &mut a.wk, &mut a.wv, &mut a.wo] {
            *m = empty.clone();
        }
        Ok(lw)
    }

    /// The body of [`Self::load`]; `host_bf16` reads the seven attention
    /// matrices as the file stores them, on the host, for [`Self::load_dense`]
    /// and [`Self::load_cpu`].
    fn load_impl(ck: &NativeCheckpoint, cfg: &InklingConfig, l: usize, dev: &Device, host_bf16: bool) -> Result<Self> {
        if l >= cfg.num_hidden_layers {
            candle_core::bail!("layer {l}: the model has {} layers", cfg.num_hidden_layers);
        }
        let p = |s: &str| format!("layers.{l}.{s}");
        let f = |s: &str| ck.f32(&p(s), dev);
        let big = |s: &str| if host_bf16 { ck.raw(&p(s), &Device::Cpu) } else { f(s) };
        let sliding = cfg.is_sliding(l);
        let (heads, kv_heads, head_dim) = cfg.attn_shape(l);
        let kv_conv = |which: &str| -> Result<KvConv> {
            if sliding {
                Ok(KvConv::Static(conv_weight(ck, &p(&format!("attn.{which}_sconv.weight")), dev)?))
            } else {
                Ok(KvConv::Dynamic {
                    static_w: f(&format!("attn.{which}_sconv.static_w"))?,
                    weight_proj: f(&format!("attn.{which}_sconv.weight_proj"))?,
                })
            }
        };
        let attn = AttnWeights {
            heads,
            kv_heads,
            head_dim,
            sliding,
            rel_extent: cfg.layer_rel_extent(l),
            wq: big("attn.wq_du.weight")?,
            q_scale: f("attn.q_l2_clip.scale")?,
            wk: big("attn.wk_dv.weight")?,
            k_conv: kv_conv("k")?,
            wv: big("attn.wv_dv.weight")?,
            v_conv: kv_conv("v")?,
            gate: big("attn.gate_attn.weight")?,
            wo: big("attn.wo_ud.weight")?,
            sink: big("attn.attn_sink_proj.weight")?,
            wr: big("attn.wr_du.weight")?,
            rel_scale: f("attn.rel_l2_clip.scale")?,
            rel_proj: f("attn.rel_logits_proj.proj")?,
            #[cfg(ink_glue)]
            dense: None,
            cpu: None,
        };
        // `attn_res.attn_queries.{i}` is layer i+1's (Inkling2IndexOffsetWeightRename, offset 1).
        let attention_input = if l > 0 {
            let q = ck.f32(&format!("attn_res.attn_queries.{}", l - 1), dev)?;
            let q = if q.rank() == 1 { q.unsqueeze(0)? } else { q };
            let want = if sliding { 1 } else { 3 };
            if q.dim(0)? != want {
                candle_core::bail!("layer {l}: {} attention-input queries, expected {want}", q.dim(0)?);
            }
            Some(q)
        } else {
            None
        };
        let norms_in = if sliding {
            vec![f("attn_norm.weight")?]
        } else {
            vec![f("q_input_norm.weight")?, f("k_input_norm.weight")?, f("v_input_norm.weight")?]
        };
        let message_gates = if cfg.is_loop_layer(l) {
            let gate = |which: &str| -> Result<MessageGate> {
                Ok(MessageGate {
                    down: f(&format!("controllers.{which}.down_weight"))?,
                    forget: f(&format!("controllers.{which}.forget_weight"))?,
                    gain: f(&format!("controllers.{which}.gain_weight"))?,
                })
            };
            // `controllers.attn` is the query gate (conversion_mapping.py line 2685)
            Some(MessageGates { q: gate("attn")?, k: gate("k")?, v: gate("v")?, mlp: gate("mlp")? })
        } else {
            None
        };
        Ok(Self {
            index: l,
            sliding,
            boundary: cfg.is_block_boundary(l),
            attention_input,
            mlp_input: ck.f32(&format!("attn_res.mlp_queries.{l}"), dev)?.unsqueeze(0)?,
            norms_in,
            post_attention_layernorm: f("mlp_norm.weight")?,
            attn,
            post_attention_convolution: conv_weight(ck, &p("attn_sconv.weight"), dev)?,
            post_mlp_convolution: conv_weight(ck, &p("mlp_sconv.weight"), dev)?,
            router_w: f("mlp.gate.weight")?,
            router_bias_host: f("mlp.gate.bias")?.to_vec1::<f32>()?,
            router_bias: f("mlp.gate.bias")?,
            message_gates,
        })
    }

    /// One visit of the decoder layer: sliding (lines 886-910, loop variant
    /// 957-997) or full (lines 1032-1062, loop variant 1111-1149).
    /// `block_states` are those the model holds before this layer; `messages`
    /// (query, key, value, MLP) are given in the loop passes after the first.
    /// Returns the layer output and the routing of its MoE.
    pub fn forward(
        &self,
        cfg: &InklingConfig,
        h: &Tensor,
        block_states: &[Tensor],
        messages: Option<&[Tensor]>,
        experts: &dyn MoeSource,
        cache: &mut SlotCache,
    ) -> Result<(Tensor, Routing)> {
        self.forward_chunked(cfg, h, block_states, messages, experts, cache, DEFAULT_ATTN_CHUNK)
    }

    /// [`Self::forward`] with the attention scores taken `attn_chunk` query
    /// rows at a time ([`AttnWeights::forward_chunked`]).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_chunked(
        &self,
        cfg: &InklingConfig,
        h: &Tensor,
        block_states: &[Tensor],
        messages: Option<&[Tensor]>,
        experts: &dyn MoeSource,
        cache: &mut SlotCache,
        attn_chunk: usize,
    ) -> Result<(Tensor, Routing)> {
        let eps = cfg.rms_norm_eps;
        let dev = h.device();
        let gates = match (messages, &self.message_gates) {
            (None, _) => None,
            (Some(m), Some(g)) if m.len() == 4 => Some((m, g)),
            _ => candle_core::bail!("layer {}: messages given to a layer without message gates", self.index),
        };
        // a slot the fused glue stepped hands its two rings back first
        cache.settle_rings()?;
        let block_residual = h;
        let attn_out = if self.sliding {
            let w = &self.norms_in[0];
            let (xq, xk, xv) = timed(GLUE, || -> Result<(Tensor, Tensor, Tensor)> {
                let x = match &self.attention_input {
                    Some(q) => block_attn_res(q, block_states, h)?.remove(0),
                    None => h.clone(),
                };
                match gates {
                    // lines 961-962
                    None => {
                        let xn = rms_norm(&x, w, eps)?;
                        Ok((xn.clone(), xn.clone(), xn))
                    }
                    // lines 964-967: one input, three gated copies, one norm
                    Some((m, g)) => Ok((
                        rms_norm(&g.q.forward(&x, &m[0], eps)?, w, eps)?,
                        rms_norm(&g.k.forward(&x, &m[1], eps)?, w, eps)?,
                        rms_norm(&g.v.forward(&x, &m[2], eps)?, w, eps)?,
                    )),
                }
            })?;
            mark(dev, Phase::NormsGates)?;
            self.attn.forward_chunked(cfg, &xq, &xk, &xv, cache, attn_chunk)?
        } else {
            let (xq, xk, xv) = timed(GLUE, || -> Result<(Tensor, Tensor, Tensor)> {
                let (mut xq, mut xk, mut xv) = match &self.attention_input {
                    Some(q) => {
                        let mut v = block_attn_res(q, block_states, h)?;
                        let xv = v.pop().expect("3 queries");
                        let xk = v.pop().expect("3 queries");
                        (v.pop().expect("3 queries"), xk, xv)
                    }
                    None => (h.clone(), h.clone(), h.clone()),
                };
                // lines 1115-1119
                if let Some((m, g)) = gates {
                    xq = g.q.forward(&xq, &m[0], eps)?;
                    xk = g.k.forward(&xk, &m[1], eps)?;
                    xv = g.v.forward(&xv, &m[2], eps)?;
                }
                Ok((
                    rms_norm(&xq, &self.norms_in[0], eps)?,
                    rms_norm(&xk, &self.norms_in[1], eps)?,
                    rms_norm(&xv, &self.norms_in[2], eps)?,
                ))
            })?;
            mark(dev, Phase::NormsGates)?;
            self.attn.forward_chunked(cfg, &xq, &xk, &xv, cache, attn_chunk)?
        };
        let a = timed(CONVS, || short_conv(&attn_out, &self.post_attention_convolution, &mut cache.conv_attn))?;
        mark(dev, Phase::Convolutions)?;

        // lines 901-905: at a block boundary the layer input becomes a block state and the residual restarts
        let mut states: Vec<Tensor> = block_states.to_vec();
        let (residual, xn) = timed(GLUE, || -> Result<(Tensor, Tensor)> {
            let residual = if self.boundary {
                states.push(block_residual.clone());
                a
            } else {
                (block_residual + a)?
            };
            let mut m_in = block_attn_res(&self.mlp_input, &states, &residual)?.remove(0);
            if let Some((m, g)) = gates {
                m_in = g.mlp.forward(&m_in, &m[3], eps)?; // lines 991-992
            }
            let xn = rms_norm(&m_in, &self.post_attention_layernorm, eps)?;
            Ok((residual, xn))
        })?;
        mark(dev, Phase::NormsGates)?;
        let routing = timed(ROUTER, || {
            route_host_bias(&xn, &self.router_w, &self.router_bias_host, cfg.num_experts_per_tok, cfg.route_scale)
        })?;
        mark(dev, Phase::Routing)?;
        let m = experts.moe(&xn, &routing, cfg)?;
        mark(dev, Phase::Moe)?;
        let m = timed(CONVS, || short_conv(&m, &self.post_mlp_convolution, &mut cache.conv_mlp))?;
        mark(dev, Phase::Convolutions)?;
        let out = timed(GLUE, || residual + m)?;
        mark(dev, Phase::NormsGates)?;
        Ok((out, routing))
    }

    /// [`Self::forward_chunked`] for one row on Metal, its small steps
    /// through the fused glue kernels (`super::glue`, plan item D7): `pre`
    /// for the block residual, message gates and input norms; the attention
    /// as candle runs it; `post_attn` for the post-attention convolution, the
    /// residual, the MLP input and the router scores; `top8` for the routing,
    /// read back once when `want_routing` asks for it and handed to the MoE
    /// on the device otherwise (the returned routing is then empty); the MoE
    /// as the source runs it; `post_mlp` for the post-MLP convolution and the
    /// residual.
    ///
    /// The post-attention and post-MLP states move into two device rings,
    /// filled from what the slot held (a prompt's block, a candle step's
    /// ring, or zeros). Class B against the candle glue: `pre`, the normed
    /// MLP input and the scores sum in another order.
    #[cfg(ink_glue)]
    #[allow(clippy::too_many_arguments)]
    pub fn forward_fused(
        &self,
        cfg: &InklingConfig,
        h: &Tensor,
        block_states: &[Tensor],
        messages: Option<&[Tensor]>,
        experts: &dyn MoeSource,
        cache: &mut SlotCache,
        glue: &std::sync::Arc<super::glue::Glue>,
        want_routing: bool,
    ) -> Result<(Tensor, Routing)> {
        use super::glue::{FusedRings, GateW, PostAttnInputs, PreInputs};
        fn gw(g: &MessageGate) -> GateW<'_> {
            GateW { down: &g.down, forget: &g.forget, gain: &g.gain }
        }
        let eps = cfg.rms_norm_eps;
        let dev = h.device();
        if h.dim(0)? != 1 {
            candle_core::bail!("layer {}: the fused glue takes one row, not {}", self.index, h.dim(0)?);
        }
        let gates = match (messages, &self.message_gates) {
            (None, _) => None,
            (Some(m), Some(g)) if m.len() == 4 => Some((m, g)),
            _ => candle_core::bail!("layer {}: messages given to a layer without message gates", self.index),
        };
        let c = self.post_attention_convolution.dim(0)?;
        let (k_attn, k_mlp) = (self.post_attention_convolution.dim(1)?, self.post_mlp_convolution.dim(1)?);
        if cache.fused.is_none() {
            let a = cache.conv_attn.block(k_attn)?;
            let m = cache.conv_mlp.block(k_mlp)?;
            cache.fused = Some(FusedRings::new(glue.clone(), dev, c, k_attn, a.as_ref(), k_mlp, m.as_ref())?);
        }
        let h = h.contiguous()?;
        let states = block_states.iter().map(|s| s.contiguous()).collect::<Result<Vec<_>>>()?;
        let msgs = match gates {
            Some((m, _)) => Some(m.iter().map(|t| t.contiguous()).collect::<Result<Vec<_>>>()?),
            None => None,
        };

        let pre = PreInputs {
            h: &h,
            states: &states,
            queries: self.attention_input.as_ref(),
            norms: &self.norms_in,
            gates: match (&msgs, gates) {
                (Some(m), Some((_, g))) => Some(([&m[0], &m[1], &m[2]], [gw(&g.q), gw(&g.k), gw(&g.v)])),
                _ => None,
            },
            sliding: self.sliding,
            eps,
        };
        let x = glue.pre(&pre)?;
        mark(dev, Phase::NormsGates)?;
        let attn_out = if x.dim(0)? == 1 {
            self.attn.forward_chunked(cfg, &x, &x, &x, cache, DEFAULT_ATTN_CHUNK)?
        } else {
            let (xq, xk, xv) = (x.narrow(0, 0, 1)?, x.narrow(0, 1, 1)?, x.narrow(0, 2, 1)?);
            self.attn.forward_chunked(cfg, &xq, &xk, &xv, cache, DEFAULT_ATTN_CHUNK)?
        };
        let attn_out = attn_out.contiguous()?;

        let rings = cache.fused.as_mut().expect("filled above");
        let post = PostAttnInputs {
            attn_out: &attn_out,
            h: &h,
            states: &states,
            boundary: self.boundary,
            conv_w: &self.post_attention_convolution,
            mlp_query: &self.mlp_input,
            gate: match (&msgs, gates) {
                (Some(m), Some((_, g))) => Some((&m[3], gw(&g.mlp))),
                _ => None,
            },
            norm: &self.post_attention_layernorm,
            router_w: &self.router_w,
            eps,
        };
        let pa = glue.post_attn(&post, &mut rings.attn)?;
        mark(dev, Phase::NormsGates)?;
        let k = cfg.num_experts_per_tok;
        let top = glue.top8(&pa.scores, &self.router_bias, k, cfg.route_scale)?;
        // Without a caller for the routing, hand it to the MoE on the device:
        // the host waits on nothing in this visit. Same ids, same weights.
        let on_device = if want_routing {
            None
        } else {
            let sel = glue.sel(&top, k)?;
            mark(dev, Phase::Routing)?;
            experts.moe_routed_on_device(&pa.xn, &sel, pa.scores.elem_count(), cfg)?
        };
        let (m, routing) = match on_device {
            Some(m) => (m.contiguous()?, Routing { indices: vec![], weights: vec![], margin: vec![], scores: vec![] }),
            None => {
                let r = top.read(k)?;
                let routing =
                    Routing { indices: vec![r.ids], weights: vec![r.weights], margin: vec![r.margin], scores: vec![r.sel] };
                mark(dev, Phase::Routing)?;
                (experts.moe(&pa.xn, &routing, cfg)?.contiguous()?, routing)
            }
        };
        mark(dev, Phase::Moe)?;
        let rings = cache.fused.as_mut().expect("filled above");
        let out = glue.post_mlp(&m, &pa.residual, &self.post_mlp_convolution, &mut rings.mlp)?;
        mark(dev, Phase::Convolutions)?;
        Ok((out, routing))
    }
}
