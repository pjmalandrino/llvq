//! The short convolutions of a layer: the history a slot keeps for each, the static depthwise
//! form and the dynamic form of the full layers. One file because the three share [`ConvState`].

use candle_core::{IndexOp, Result, Tensor};

use super::linear;

/// The last `K - 1` inputs of one short convolution, oldest first
/// (`conv_state`, lines 297, 379, 438). A prompt leaves a block `[K - 1, C]`;
/// a one-row step keeps a fixed ring of `K - 1` row handles `[1, C]` and
/// replaces the oldest, with no concatenation and no copy. Nothing is written
/// in place: a clone, or a mark, keeps what it saw.
#[derive(Debug, Clone, Default)]
pub enum ConvState {
    /// No input yet: the history is zeros (the reference's left padding).
    #[default]
    Empty,
    /// `[K - 1, C]`, oldest row first.
    Block(Tensor),
    /// `K - 1` rows `[1, C]`; `rows[head]` is the oldest.
    Ring { rows: Vec<Tensor>, head: usize },
}

impl ConvState {
    pub fn is_empty(&self) -> bool {
        matches!(self, ConvState::Empty)
    }

    /// The state as one block `[k - 1, C]`, oldest row first; `None` when
    /// empty (zeros). A ring is concatenated in time order: the same values.
    pub fn block(&self, k: usize) -> Result<Option<Tensor>> {
        match self {
            ConvState::Empty => Ok(None),
            ConvState::Block(s) => Ok(Some(s.clone())),
            ConvState::Ring { rows, head } => {
                let ordered: Vec<Tensor> = (0..k - 1).map(|j| rows[(head + j) % (k - 1)].clone()).collect();
                Ok(Some(Tensor::cat(&ordered, 0)?))
            }
        }
    }

    /// The `k - 1` history rows `[1, C]`, oldest first; zeros rows like `x` when empty.
    fn rows(&self, x: &Tensor, k: usize) -> Result<Vec<Tensor>> {
        match self {
            ConvState::Empty => {
                let z = Tensor::zeros((1, x.dim(1)?), x.dtype(), x.device())?;
                Ok(vec![z; k - 1])
            }
            ConvState::Block(s) => (0..k - 1).map(|i| s.narrow(0, i, 1)).collect(),
            ConvState::Ring { rows, head } => Ok((0..k - 1).map(|j| rows[(head + j) % (k - 1)].clone()).collect()),
        }
    }

    /// After a one-row step on `x` `[1, C]`: `x` replaces the oldest row.
    fn push_row(&mut self, x: &Tensor, k: usize) -> Result<()> {
        let (mut rows, head) = match std::mem::take(self) {
            ConvState::Ring { rows, head } => (rows, head),
            other => (other.rows(x, k)?, 0),
        };
        rows[head] = x.clone();
        *self = ConvState::Ring { rows, head: (head + 1) % (k - 1) };
        Ok(())
    }
}

/// The convolution context: the `k - 1` inputs before `x` (zeros when
/// `state` is empty) then `x`, `[k - 1 + T, C]`. `state` becomes the last
/// `k - 1` rows (`update_conv_state` / `conv_state.copy_`, lines 297, 379, 438).
pub(super) fn conv_context(x: &Tensor, k: usize, state: &mut ConvState) -> Result<Tensor> {
    let t = x.dim(0)?;
    let ctx = match &*state {
        ConvState::Block(s) => Tensor::cat(&[s, x], 0)?,
        ConvState::Empty => x.pad_with_zeros(0, k - 1, 0)?,
        ring @ ConvState::Ring { .. } => {
            let mut parts = ring.rows(x, k)?;
            parts.push(x.clone());
            Tensor::cat(&parts, 0)?
        }
    };
    // The last rows in a buffer of their own. A narrow of `ctx` is a view, and candle's `contiguous` and `copy`
    // both keep the whole buffer behind a view that is already contiguous: the state of a prompt of T rows would
    // hold `[T + k - 1, C]` for as long as the cache lives, four times a slot.
    *state = ConvState::Block(if t > 1 {
        let rows = Tensor::arange(t as u32, (t + k - 1) as u32, ctx.device())?;
        ctx.index_select(&rows, 0)?
    } else {
        ctx.narrow(0, t, k - 1)?.contiguous()?
    });
    Ok(ctx)
}

/// `Inkling2LoopShortConvolution.forward`, lines 362-393: a depthwise causal
/// convolution, `w` `[C, K]` from the file's `[C, 1, K]`, tap `K - 1` on the
/// current token (`F.conv1d` with left padding `K - 1`, lines 316-322, or the
/// cached history in decode, lines 373-376), then the residual and the output
/// scale `1 / sqrt(K + 1)` (line 392).
pub fn short_conv(x: &Tensor, w: &Tensor, state: &mut ConvState) -> Result<Tensor> {
    let k = w.dim(1)?;
    let t = x.dim(0)?;
    let mut acc = x.zeros_like()?;
    if t == 1 {
        // One row: the history rows where they are, the same products and sums as the context below.
        let rows = state.rows(x, k)?;
        for (j, row) in rows.iter().chain(std::iter::once(x)).enumerate() {
            let tap = w.i((.., j))?; // [C]
            acc = (acc + row.broadcast_mul(&tap)?)?;
        }
        state.push_row(x, k)?;
    } else {
        let ctx = conv_context(x, k, state)?;
        for j in 0..k {
            let tap = w.i((.., j))?; // [C]
            acc = (acc + ctx.narrow(0, j, t)?.broadcast_mul(&tap)?)?;
        }
    }
    let out = (acc + x)?;
    // line 392: the output scale (absent from Inkling 1, per the reference comment)
    out / ((k + 1) as f64).sqrt()
}

/// `Inkling2LoopDynamicShortConvolution.forward`, lines 428-455, prompt and
/// decode alike (the reference has one path for both, lines 437-446).
/// Tap `lag` (backward distance, `static_weight[0]` on the current token) is
/// `static_weight[lag, c] + (dynamic_input @ weight_proj.T)[t, head(c) * K + lag]`,
/// shared by the channels of a head. Residual, then scale `1 / sqrt(2K + 1)` (line 454).
pub fn dynamic_conv(
    x: &Tensor,
    dynamic_input: &Tensor,
    static_w: &Tensor,
    weight_proj: &Tensor,
    heads: usize,
    state: &mut ConvState,
) -> Result<Tensor> {
    let (t, c) = x.dims2()?;
    let k = static_w.dim(0)?;
    let dw = linear(dynamic_input, weight_proj)?.reshape((t, heads, k))?; // line 430-431
    // One row reads the history rows where they are; a prompt reads the context.
    let (rows, ctx) = if t == 1 { (Some(state.rows(x, k)?), None) } else { (None, Some(conv_context(x, k, state)?)) };
    let mut acc = x.zeros_like()?;
    for lag in 0..k {
        let d = dw.i((.., .., lag))?.unsqueeze(2)?.broadcast_as((t, heads, c / heads))?.reshape((t, c))?;
        let w = d.broadcast_add(&static_w.i(lag)?)?; // line 432
        let src = match (&rows, &ctx) {
            (Some(_), _) if lag == 0 => x.clone(),
            (Some(rows), _) => rows[k - 1 - lag].clone(),
            (None, Some(ctx)) => ctx.narrow(0, k - 1 - lag, t)?,
            (None, None) => unreachable!("one of the two is set"),
        };
        acc = (acc + src.mul(&w)?)?; // lines 444-453
    }
    if t == 1 {
        state.push_row(x, k)?;
    }
    let out = (acc + x)?;
    out / ((2 * k + 1) as f64).sqrt()
}
