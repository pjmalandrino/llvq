//! The mixture of experts: the two seams that supply experts, the routing on the host and the
//! dense expert sum. One file because the router's output is the experts' input and both cite
//! the same reference lines.

use candle_core::{Device, Result, Tensor};

use crate::inkling::config::InklingConfig;
use crate::inkling::loader::NativeExperts;

use super::{linear, softcap};

/// Every expert of a layer as f32 matrices: `(gate_up [2I, H], down [H, I])`.
pub trait ExpertSource {
    fn count(&self) -> usize;
    fn expert(&self, e: usize, dev: &Device) -> Result<(Tensor, Tensor)>;
}

impl ExpertSource for NativeExperts<'_> {
    fn count(&self) -> usize {
        NativeExperts::count(self)
    }
    fn expert(&self, e: usize, dev: &Device) -> Result<(Tensor, Tensor)> {
        let (gu, down) = self.dequant(e);
        let (r13, h) = (self.w13.rows, self.w13.cols);
        Ok((Tensor::from_vec(gu, (r13, h), dev)?, Tensor::from_vec(down, (self.w2.rows, self.w2.cols), dev)?))
    }
}

/// The one seam of the MoE call: what computes `sum_k w_k expert_k(x)` for a
/// layer. Every dense [`ExpertSource`] is one, through [`experts_forward`];
/// `resident::ResidentExperts` (Metal) is the other, through
/// `resident::experts_forward`, from lattice codes kept on the device.
pub trait MoeSource {
    /// `x` `[T, H]` and its routing to the MoE output `[T, H]` on the device of `x`.
    fn moe(&self, x: &Tensor, routing: &Routing, cfg: &InklingConfig) -> Result<Tensor>;

    /// GPU kernel launches of the lattice path since the last call (zero
    /// for a dense source), and reset.
    fn take_launches(&self) -> usize {
        0
    }

    /// The MoE of one row whose routing is still on the device: `sel` is
    /// `[2k]` u32, the ids of the `k` routed experts, ascending and distinct,
    /// chosen among `experts`, then the bits of their weights. The same
    /// result as [`Self::moe`] on that routing, bit for bit. `None` when this
    /// source has no such path: the caller then reads the routing back and
    /// calls [`Self::moe`].
    fn moe_routed_on_device(
        &self,
        _x: &Tensor,
        _sel: &Tensor,
        _experts: usize,
        _cfg: &InklingConfig,
    ) -> Result<Option<Tensor>> {
        Ok(None)
    }

    /// For a source with a fused one-row path: run its activation in one
    /// launch (`true`) or through candle's operations, the reference of that
    /// launch's gate. The same bits either way; nothing to do elsewhere.
    fn set_fused_act(&mut self, _on: bool) {}

    /// For a source that has both: take the routed experts of one row in
    /// four launches (`true`) or one expert at a time, the reference of the
    /// four launches' gate. Meant to be the same bits; nothing to do elsewhere.
    fn set_four_launches(&mut self, _on: bool) {}

    /// For a source that has both: take a prompt's rows in one launch a step
    /// for the whole layer (`true`) or one expert and one chunk at a time, the
    /// reference of that path's gate. Meant to be the same bits.
    fn set_batch_rows(&mut self, _on: bool) {}
}

impl<S: ExpertSource> MoeSource for S {
    fn moe(&self, x: &Tensor, routing: &Routing, cfg: &InklingConfig) -> Result<Tensor> {
        experts_forward(x, routing, self, cfg)
    }
}

/// The routing of one MoE call: per token, the experts sorted ascending and
/// their weights aligned with them.
#[derive(Debug, Clone)]
pub struct Routing {
    pub indices: Vec<Vec<u32>>,
    pub weights: Vec<Vec<f32>>,
    /// Per token, the 8th minus the 9th largest `score + bias`.
    pub margin: Vec<f32>,
    /// Per token, `score + bias` of every expert: what the top-k selected on.
    pub scores: Vec<Vec<f32>>,
}

/// `Inkling2LoopTopKRouter.forward`, lines 1163-1172, in f32: sigmoid scores,
/// top-k on `score + e_score_correction_bias`, weights the scores of the
/// chosen experts normalised to sum 1 (plus 1e-20) times `route_scale`.
pub fn route(x: &Tensor, w_gate: &Tensor, bias: &Tensor, top_k: usize, scale: f64) -> Result<Routing> {
    route_host_bias(x, w_gate, &bias.to_vec1::<f32>()?, top_k, scale)
}

/// [`route`] with the bias already on the host: the layer reads it once at
/// load ([`LayerWeights::router_bias_host`]), not at every visit. The same
/// f32 values in the same additions, so the same routing bit for bit.
pub(super) fn route_host_bias(x: &Tensor, w_gate: &Tensor, bias: &[f32], top_k: usize, scale: f64) -> Result<Routing> {
    let scores = candle_nn::ops::sigmoid(&linear(x, w_gate)?)?.to_vec2::<f32>()?;
    if scores.first().is_some_and(|s| s.len() != bias.len()) {
        candle_core::bail!("route: {} scores and {} bias values", scores[0].len(), bias.len());
    }
    // One row at a time, each on its own: the rows of a prompt are sorted in parallel and gathered in order.
    let one = |s: &Vec<f32>| {
        let sel: Vec<f32> = s.iter().zip(bias).map(|(a, b)| a + b).collect();
        let mut order: Vec<usize> = (0..sel.len()).collect();
        order.sort_by(|&a, &b| sel[b].total_cmp(&sel[a]).then(a.cmp(&b)));
        let margin = sel[order[top_k - 1]] - sel[order[top_k]];
        let mut idx: Vec<usize> = order[..top_k].to_vec();
        idx.sort_unstable();
        let picked: Vec<f32> = idx.iter().map(|&i| s[i]).collect();
        let sum: f32 = picked.iter().sum::<f32>() + 1e-20;
        let weights: Vec<f32> = picked.iter().map(|w| w / sum * scale as f32).collect();
        (idx.iter().map(|&i| i as u32).collect::<Vec<u32>>(), weights, margin, sel)
    };
    let rows: Vec<_> = if scores.len() >= 64 {
        use rayon::prelude::*;
        scores.par_iter().map(one).collect()
    } else {
        scores.iter().map(one).collect()
    };
    let mut r = Routing { indices: vec![], weights: vec![], margin: vec![], scores: vec![] };
    for (idx, weights, margin, sel) in rows {
        r.indices.push(idx);
        r.weights.push(weights);
        r.margin.push(margin);
        r.scores.push(sel);
    }
    Ok(r)
}

/// `Inkling2LoopExperts.forward`, lines 787-807, with the softcapped SwiGLU of
/// lines 760-771: experts in ascending order, each over its tokens, the
/// weighted outputs accumulated in f32 by `index_add`.
pub fn experts_forward(
    x: &Tensor,
    routing: &Routing,
    src: &dyn ExpertSource,
    cfg: &InklingConfig,
) -> Result<Tensor> {
    let dev = x.device();
    let inter = cfg.intermediate_size;
    let mut out = x.zeros_like()?;
    if x.dim(0)? == 1 && routing.indices.len() == 1 {
        // One row: the experts in the ascending order of the general path
        // below, each on `x` itself (no `index_select`), its weighted output
        // added to the sum (`index_add` onto one row is that same addition),
        // the 8 weights uploaded once.
        let (idx, w) = (&routing.indices[0], &routing.weights[0]);
        let wts = Tensor::from_vec(w.clone(), (w.len(), 1), dev)?;
        let mut order: Vec<usize> = (0..idx.len()).collect();
        order.sort_by_key(|&j| idx[j]);
        for j in order {
            let (gu_w, down_w) = src.expert(idx[j] as usize, dev)?;
            let gu = linear(x, &gu_w)?;
            let g = gu.narrow(1, 0, inter)?.contiguous()?;
            let u = gu.narrow(1, inter, inter)?.contiguous()?;
            let act = softcap(&g, cfg.mlp_gate_softcap)?
                .mul(&candle_nn::ops::sigmoid(&g)?)?
                .mul(&softcap(&u, cfg.mlp_up_softcap)?)?;
            let y = linear(&act, &down_w)?.broadcast_mul(&wts.narrow(0, j, 1)?)?;
            out = (out + y)?;
        }
        return Ok(out);
    }
    let mut per_expert: Vec<Vec<(u32, f32)>> = vec![vec![]; src.count()];
    for (t, (idx, w)) in routing.indices.iter().zip(&routing.weights).enumerate() {
        for (&e, &wt) in idx.iter().zip(w) {
            per_expert[e as usize].push((t as u32, wt));
        }
    }
    for (e, toks) in per_expert.iter().enumerate() {
        if toks.is_empty() {
            continue;
        }
        let (gu_w, down_w) = src.expert(e, dev)?;
        let ids = Tensor::from_vec(toks.iter().map(|p| p.0).collect::<Vec<_>>(), toks.len(), dev)?;
        let wts = Tensor::from_vec(toks.iter().map(|p| p.1).collect::<Vec<_>>(), (toks.len(), 1), dev)?;
        let xs = x.index_select(&ids, 0)?;
        let gu = linear(&xs, &gu_w)?;
        let g = gu.narrow(1, 0, inter)?.contiguous()?;
        let u = gu.narrow(1, inter, inter)?.contiguous()?;
        // line 771: softcap(gate) * sigmoid(gate) * softcap(up)
        let act = softcap(&g, cfg.mlp_gate_softcap)?
            .mul(&candle_nn::ops::sigmoid(&g)?)?
            .mul(&softcap(&u, cfg.mlp_up_softcap)?)?;
        let y = linear(&act, &down_w)?.broadcast_mul(&wts)?;
        out = out.index_add(&ids, &y, 0)?;
    }
    Ok(out)
}
