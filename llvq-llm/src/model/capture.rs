//! What a forward pass hands out: the activation capture points and the
//! per-phase timing of a fenced decode.

use super::*;

/// Which activation a capture callback is being handed.
///
/// Named after the tensor, not the matrix: `Attn` feeds `q_proj`, `k_proj`
/// and `v_proj` alike.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Act {
    /// `input_layernorm(x)` — input of q/k/v_proj. Width: `hidden_size`.
    Attn,
    /// The concatenated attention context — input of o_proj.
    /// Width: `head_dim · num_attention_heads`.
    AttnOut,
    /// `post_attention_layernorm(x)` — input of gate/up_proj.
    /// Width: `hidden_size`.
    Mlp,
    /// `act(gate(h)) · up(h)` — input of down_proj. Width: `intermediate_size`.
    MlpOut,
}

impl Act {
    pub const ALL: [Act; 4] = [Act::Attn, Act::AttnOut, Act::Mlp, Act::MlpOut];

    /// Stable index, so a rotation seed derived from it is reproducible
    /// across runs and across reorderings of `ALL`.
    pub fn index(&self) -> u64 {
        match self {
            Act::Attn => 0,
            Act::AttnOut => 1,
            Act::Mlp => 2,
            Act::MlpOut => 3,
        }
    }

    /// The matrices this activation feeds, as safetensors name suffixes.
    pub fn consumers(&self) -> &'static [&'static str] {
        match self {
            Act::Attn => &["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
            Act::AttnOut => &["self_attn.o_proj"],
            Act::Mlp => &["mlp.gate_proj", "mlp.up_proj"],
            Act::MlpOut => &["mlp.down_proj"],
        }
    }

    pub fn width(&self, cfg: &Config) -> usize {
        match self {
            Act::Attn | Act::Mlp => cfg.hidden_size,
            Act::AttnOut => cfg.head_dim * cfg.num_attention_heads,
            Act::MlpOut => cfg.intermediate_size,
        }
    }
}

/// Receives every linear-layer input as the forward pass produces it.
pub trait Capture {
    /// `x` is `(batch, seq, width)`; rows are samples for `AᵀA`.
    fn on_activation(&mut self, layer: usize, act: Act, x: &Tensor) -> Result<()>;
}

/// The do-nothing capture, for plain inference.
pub struct NoCapture;

impl Capture for NoCapture {
    fn on_activation(&mut self, _: usize, _: Act, _: &Tensor) -> Result<()> {
        Ok(())
    }
}

/// Per-phase wall time of a fenced greedy decode, milliseconds.
///
/// Produced by [`Qwen3::generate_phased`]; see its contract. A decode of `n`
/// tokens yields `n` samples for the head-side phases and `n − 1` for the
/// forward-side ones — the last emitted token needs no further forward.
#[derive(Default)]
pub struct PhaseReport {
    /// Token id → hidden state (`Embed::forward` plus the host→device id copy).
    pub embed_ms: Vec<f64>,
    /// The transformer blocks — attention, MLP, rotation if fused — plus the
    /// causal mask and the final norm.
    pub blocks_ms: Vec<f64>,
    /// `Head::project` plus the f32 upcast of the logits.
    pub head_ms: Vec<f64>,
    /// Argmax, device→host readback of the winning id, bookkeeping.
    pub rest_ms: Vec<f64>,
    /// Device fences issued. Pinned by a test: a fence dropped between two
    /// phases would silently charge one phase's work to the next, and no
    /// timing assertion could catch that on a fast host.
    pub fences: usize,
}

impl PhaseReport {
    /// `(median, min, max)` of one phase's samples. Even counts take the mean
    /// of the two central samples.
    pub fn stats(samples: &[f64]) -> (f64, f64, f64) {
        assert!(!samples.is_empty(), "no samples to summarise");
        let mut s = samples.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).expect("phase times are finite"));
        let n = s.len();
        let median = if n % 2 == 1 {
            s[n / 2]
        } else {
            (s[n / 2 - 1] + s[n / 2]) / 2.0
        };
        (median, s[0], s[n - 1])
    }
}

/// One phase boundary: a device fence, counted.
///
/// The synchronize and the count share these two lines on purpose — the test
/// that pins `fences` can then vouch that every boundary reached the device.
/// (A mutant keeping the count but dropping the sync is invisible on a
/// synchronous backend, where the sync is a no-op anyway; on CUDA it is the
/// pairing below that the count certifies.)
pub(super) fn fence(dev: &Device, n: &mut usize) -> Result<()> {
    dev.synchronize()?;
    *n += 1;
    Ok(())
}
