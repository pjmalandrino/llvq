//! The phase clock of `inkrun`'s measurement flag: the parts of a forward, the switch and the
//! accumulator one mark charges. One file because the three share two statics and nothing else.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use candle_core::{Device, Result};

/// The parts of a forward.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// The embedding rows and their norm.
    Embed,
    /// The output head and its softcap.
    Head,
    /// The logits read back to the host.
    Readback,
    /// Router product, sigmoid, read back, top-k on the host.
    Routing,
    /// The attention projections (query, key, value, relative, sink,
    /// gate, output) with their clips and scales.
    AttnProducts,
    /// Relative bias and mask, scores, sinks, softmax, value product.
    AttnScores,
    /// The cache append and the copies of what it holds.
    CacheCopies,
    /// The four short convolutions of a visit.
    Convolutions,
    /// Block residuals, message gates, norms, residual adds.
    NormsGates,
    /// The MoE call.
    Moe,
    /// Whatever no other mark charged.
    Other,
}

pub const ALL: [Phase; 11] = [
    Phase::Embed,
    Phase::Head,
    Phase::Readback,
    Phase::Routing,
    Phase::AttnProducts,
    Phase::AttnScores,
    Phase::CacheCopies,
    Phase::Convolutions,
    Phase::NormsGates,
    Phase::Moe,
    Phase::Other,
];

impl Phase {
    pub fn name(self) -> &'static str {
        match self {
            Phase::Embed => "embed",
            Phase::Head => "head",
            Phase::Readback => "readback",
            Phase::Routing => "routing",
            Phase::AttnProducts => "attention_products",
            Phase::AttnScores => "attention_scores_softmax",
            Phase::CacheCopies => "cache_append_copies",
            Phase::Convolutions => "convolutions",
            Phase::NormsGates => "norms_gates_residual",
            Phase::Moe => "moe",
            Phase::Other => "other",
        }
    }
}

static ON: AtomicBool = AtomicBool::new(false);
static STATE: Mutex<Option<(Instant, [f64; 11])>> = Mutex::new(None);

/// Drain `dev` and start charging from now.
pub fn start(dev: &Device) -> Result<()> {
    dev.synchronize()?;
    *STATE.lock().expect("no panic holds this lock") = Some((Instant::now(), [0.0; 11]));
    ON.store(true, Ordering::SeqCst);
    Ok(())
}

/// Stop and return the seconds charged to each part, in the order of [`ALL`].
pub fn stop() -> [f64; 11] {
    ON.store(false, Ordering::SeqCst);
    STATE.lock().expect("no panic holds this lock").take().map(|s| s.1).unwrap_or([0.0; 11])
}

/// Drain `dev` and charge the time since the last mark to `p`.
#[inline]
pub fn mark(dev: &Device, p: Phase) -> Result<()> {
    if !ON.load(Ordering::Relaxed) {
        return Ok(());
    }
    dev.synchronize()?;
    let now = Instant::now();
    if let Some((last, acc)) = STATE.lock().expect("no panic holds this lock").as_mut() {
        acc[p as usize] += now.duration_since(*last).as_secs_f64();
        *last = now;
    }
    Ok(())
}
